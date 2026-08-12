#!/usr/bin/env python3
"""
Offline Re-ID Tracker — BoTSORT + Persistent Gallery
======================================================
FIXED version for SINGLE-CAMERA tracking.
Key fixes:
1. Removed active_gids exclusion (single-camera: same person CAN re-appear)
2. Fixed probe buffer - no more "leader changed" reset
3. Use max-similarity-per-person instead of biased voting
4. Lowered bridge threshold for natural re-acquisition
5. Occlusion no longer wipes confirmation buffer
6. Added track-level GID persistence
7. Temporal memory boost for recently-seen persons
"""

import os
import sys
import json
import time
import threading
import queue
from pathlib import Path
from collections import defaultdict
from face_extractor import FaceExtractor
import json
from collections import Counter
from datetime import datetime, timedelta

import cv2
import numpy as np
import torch
import psycopg2
import faiss

os.environ["OPENCV_LOG_LEVEL"] = "QUIET"
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
    "rtsp_transport;tcp"
    "|buffer_size;2097152"
    "|max_delay;500000"
    "|reorder_queue_size;10"
)

try:
    from ultralytics import YOLO
    from boxmot import BoTSORT
except ImportError as e:
    sys.exit(f"[ERROR] Missing dependency: {e}")

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════
INPUT_SOURCE = "1"
OUTPUT_VIDEO = "none"

YOLO_MODEL   = "yolov8n.pt"  # Switched to Nano model for much faster CPU processing
REID_WEIGHTS = Path("osnet_ain_x1_0_msmt17.pt")
DEVICE       = "0" if torch.cuda.is_available() else "cpu"

CONF_THRESH    = 0.45
NMS_IOU_THRESH = 0.40
OCCLUSION_IOU  = 0.35

# [FIX] Two live tracks whose boxes overlap this much are almost certainly
# one physical person detected twice, not two people standing on top of
# each other — used to collapse ghost boxes before identity resolution.
DUPLICATE_BOX_IOU = 0.75

FRAME_STRIDE = 3  # Changed from 1 to 3: Skips frames to triple FPS and reduce AI lag
CAMERA_ID = "offline_video"
FLUSH_ON_START = False

DB_DIR           = Path("reid_database_stride2")
FAISS_INDEX_FILE = DB_DIR / "identities.faiss"
FAISS_FACE_INDEX_FILE = DB_DIR / "faces.faiss"
FACE_SIM_THRESH = 0.85

PG_DBNAME = "attendance_db"
PG_USER   = "postgres"
PG_PASS   = "YOUR_DB_PASSWORD"
PG_HOST   = "localhost"

# [FIX] Switched to pure deep features (512-dim OSNet) with Test-Time Augmentation (TTA)
# The HSV histogram was too sensitive to lighting. TTA gives a massive accuracy boost for free.
REID_VECTOR_DIM         = 512

# Thresholds re-tuned for pure 512-dim deep features (which have slightly sharper similarities)
REID_SIM_THRESH_MATCH   = 0.80   
REID_SIM_THRESH_TEMPORAL = 0.75  

RECONCILE_SIM_THRESH  = 0.82     
RECONCILE_SIM_SINGLE  = 0.85     

STICKY_FRAME_COUNT   = 1
STICKY_SWITCH_THRESH = 0.70      # Need strong evidence to switch

CONFIRM_FRAMES        = 1
PROBE_FRAMES          = 1        # Instantly assign IDs for live/choppy webcams
PROBE_AGREE_THRESHOLD = 1        # Need 1 vote to confirm

MAX_SAMPLES = 8

RECENTLY_LOST_TTL_FRAMES = 1500  # 60 seconds at 25fps
RECENTLY_LOST_MAX_DIST = 400     # Increased to handle track drops while walking
BRIDGE_MIN_SIM         = 0.68    # Fallback default

# [FIX] Continuous bridge-confidence curve (replaces old hard-cutoff tiers).
# Required similarity slides between these two bounds based on how spatially
# tight and how recent the reappearance is, instead of jumping off a cliff
# at fixed frame_gap/distance boundaries (e.g. the old d<150 cutoff, which
# rejected a person who reappeared at d=151-238px purely because they
# crossed an arbitrary line, even though nothing else was near that spot).
# [FIX] Tightened bridging limits to prevent false identity merging (contamination).
# ReID features are now stronger with TTA, so we don't need to dangerously drop
# the threshold to 0.38 anymore. The absolute lowest visual similarity we will ever
# accept for a bridge (even if they are standing in the exact same spot instantly) is 0.65.
BRIDGE_THRESH_CEILING = 0.80     # Required sim when far away / long gone
BRIDGE_THRESH_FLOOR   = 0.75     # Absolute minimum required sim, even under perfect conditions

# [FIX] Ambiguity guard: if the best bridge candidate isn't an obvious tight
# reacquisition (same spot, moments later) AND a different dormant identity
# scores nearly as well, that single frame isn't strong enough evidence to
# commit to either one — defer to PROBE voting instead, which demands
# multiple consistent frames before confirming an identity. Widened to 0.15:
# two candidates both sitting in the 0.5-0.6 range (inherently unreliable,
# neither a strong match) need a real, wide margin between them before
# trusting the higher one — a 0.599 vs 0.493 gap (~0.10) is exactly the
# "both mediocre, one barely ahead" case this is meant to catch.
BRIDGE_AMBIGUITY_MARGIN = 0.15

TEMPORAL_MEMORY_TTL    = 15.0    # Short window: only for occlusion re-entry, not full re-visits
TEMPORAL_MATCH_THRESH  = 0.72    # Must be clearly the same person even in temporal mode

MIN_CROP_BRIGHTNESS = 15
MIN_CROP_CONTRAST   = 10
EMBEDDING_CACHE_TTL  = 0.5

# [FIX] Crop geometry gates — reject legs-only, partial, and micro crops.
# A standing person seen from overhead CCTV typically has aspect ratio
# (height/width) >= 1.3. Crops below this are almost always legs-only or
# torso fragments caught at the frame edge. ReID on legs alone is noise.
MIN_CROP_ASPECT_RATIO = 0.5      # height/width — lowered to 0.5 to allow webcam head/shoulder crops
MIN_CROP_HEIGHT_PX    = 80       # absolute minimum pixels tall (after margins)
MIN_CROP_WIDTH_PX     = 30       # absolute minimum pixels wide (after margins)

AUDIT_DIR = Path("./reid_audit_stride2")
AUDIT_CROPS_PER_GID = 6
# [FIX] Minimum frames between two saved crops for the same GID, so the
# 6 crops are spread across the person's time on screen (different pose,
# angle, lighting) rather than 6 near-duplicate frames grabbed in a burst
# right after confirmation.
AUDIT_MIN_FRAME_GAP = 15

# [NEW] Track persistence: once assigned, keep GID unless strong evidence
TRACK_PERSISTENCE_FRAMES = 10    # Min frames before allowing switch
TRACK_SWITCH_MIN_SIM = 0.65      # Must match new GID better than this

# [NEW] Face-verified attendance (IN/OUT) logging
FACE_VERIFY_SIM_THRESH     = 0.75     # Same threshold already used to accept a face match on screen
ATTENDANCE_COOLDOWN_SEC    = 10 * 60  # 10 minute timer for IN/OUT events
REQUIRED_WORK_HOURS        = 8.0      # Minimum hours required to be marked 'Present'
ATTENDANCE_CHECK_INTERVAL_SEC = 5     # Throttle: touch the DB at most this often per person,
                                       # instead of on every single frame — the 10 min timer
                                       # doesn't need frame-level resolution, and hitting
                                       # Postgres 15-30x/sec per visible face is what was
                                       # causing the lag.
IST_OFFSET               = timedelta(hours=5, minutes=30)  # attendance timestamps are logged in IST, not UTC


def now_ist() -> datetime:
    """Current time in IST. Attendance in/out/date are all logged against this,
    not datetime.utcnow(), so 'today' matches the local calendar day the
    camera site actually runs on."""
    return datetime.utcnow() + IST_OFFSET


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def calculate_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0]); yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2]); yB = min(boxA[3], boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    aA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    aB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return inter / float(aA + aB - inter + 1e-5)


def check_crop_quality(crop: np.ndarray, bbox: tuple = None, frame_shape: tuple = None) -> tuple[bool, str]:
    if crop.size == 0:
        return False, "empty"
    h, w = crop.shape[:2]

    # [FIX] Reject crops that are too small to contain a recognizable person.
    if h < MIN_CROP_HEIGHT_PX:
        return False, f"too_short({h}px)"
    if w < MIN_CROP_WIDTH_PX:
        return False, f"too_narrow({w}px)"

    # [FIX] Edge-touching filter
    # If the bounding box touches the left or right edge of the video frame, the
    # person is partially off-screen (e.g. just an arm or leg). 
    # We do NOT check the bottom edge (y2) because it's completely normal for 
    # a person's feet to touch the bottom of a CCTV camera view.
    if bbox is not None and frame_shape is not None:
        x1, y1, x2, y2 = bbox
        H, W = frame_shape[:2]
        EDGE_MARGIN = 10
        if x1 < EDGE_MARGIN or x2 > W - EDGE_MARGIN:
            return False, "edge_fragment"

    # [FIX] Reject crops with bad aspect ratio (legs-only, or wide slivers).
    aspect = h / max(w, 1)
    if aspect < MIN_CROP_ASPECT_RATIO:
        return False, f"bad_aspect({aspect:.2f})"

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if len(crop.shape) == 3 else crop
    mean_bright = gray.mean()
    std_bright = gray.std()
    if mean_bright < MIN_CROP_BRIGHTNESS:
        return False, f"too_dark({mean_bright:.0f})"
    if std_bright < MIN_CROP_CONTRAST:
        return False, f"low_contrast({std_bright:.0f})"
    return True, "ok"


# ══════════════════════════════════════════════════════════════════════════════
# 1. CONFIRMATION BUFFER (with EMA for occlusion resilience)
# ══════════════════════════════════════════════════════════════════════════════
class ConfirmationBuffer:
    def __init__(self):
        self._bufs: dict[int, list[np.ndarray]] = defaultdict(list)
        self._ema: dict[int, np.ndarray] = {}  # [NEW] Exponential moving average
        self._ema_alpha = 0.7  # Weight for new embeddings

    def add(self, sid: int, emb: np.ndarray) -> np.ndarray | None:
        self._bufs[sid].append(emb)
        
        # [NEW] Update EMA
        if sid in self._ema:
            self._ema[sid] = self._ema_alpha * emb + (1 - self._ema_alpha) * self._ema[sid]
            self._ema[sid] /= np.linalg.norm(self._ema[sid])
        else:
            self._ema[sid] = emb.copy()
        
        if len(self._bufs[sid]) >= CONFIRM_FRAMES:
            # Use EMA if available, otherwise average
            if sid in self._ema and len(self._bufs[sid]) >= 3:
                avg = self._ema[sid].astype(np.float32)
            else:
                avg = np.mean(self._bufs[sid], axis=0).astype(np.float32)
            norm = np.linalg.norm(avg)
            del self._bufs[sid]
            return avg / norm if norm > 1e-6 else None
        return None

    def get_ema(self, sid: int) -> np.ndarray | None:
        """[NEW] Get EMA embedding even if not fully confirmed yet."""
        return self._ema.get(sid)

    def remove(self, sid: int):
        self._bufs.pop(sid, None)
        self._ema.pop(sid, None)


# ══════════════════════════════════════════════════════════════════════════════
# 2. REID EXTRACTOR
# ══════════════════════════════════════════════════════════════════════════════
class ReIDExtractor:
    def __init__(self, reid_weights: Path, device: str):
        print("[REID] Loading ReID model ...")
        # Initialize BoTSORT with weights to prevent boxmot crash, but we bypass execution in extract()
        self.tracker = BoTSORT(model_weights=reid_weights, device=device, fp16=False)
        self.model = self.tracker.model
        
        self._cache: dict[int, tuple[np.ndarray, float]] = {}
        print("[REID] ReID model bypassed (OSNet execution disabled for speed test).")

    def _color_histogram(crop: np.ndarray) -> np.ndarray:
        """Compute rotation-invariant HSV color histogram on the clothing crop.
        Uses only the torso zone (top 70%, center 80% width) to avoid head/hands.
        Returns a L2-normalized 1024-dim float32 vector (16×8×8 bins)."""
        h, w = crop.shape[:2]
        # Torso zone: top 70% of height, center 80% of width
        x_pad = max(1, int(w * 0.10))
        y_end = max(10, int(h * 0.70))
        torso = crop[:y_end, x_pad:w - x_pad]
        if torso.size == 0 or torso.shape[0] < 5 or torso.shape[1] < 5:
            torso = crop  # fallback
        hsv = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)
        # 16 Hue bins × 8 Saturation bins × 8 Value bins = 1024 bins
        hist = cv2.calcHist(
            [hsv], [0, 1, 2], None,
            [16, 8, 8],
            [0, 180, 0, 256, 0, 256]
        ).flatten().astype(np.float32)
        norm = np.linalg.norm(hist)
        return hist / norm if norm > 1e-6 else hist

    def extract(self, frame: np.ndarray, bbox: tuple, session_id: int,
                force: bool = False) -> tuple[np.ndarray | None, str]:
        if session_id in self._cache and not force:
            emb, ts = self._cache[session_id]
            if time.time() - ts < EMBEDDING_CACHE_TTL:
                return emb, "cached"

        x1, y1, x2, y2 = bbox
        H, W = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(W, x2), min(H, y2)
        bw, bh = x2 - x1, y2 - y1
        if bw < 20 or bh < 40:
            return None, "too_small"

        # Use standard margins for OSNet which expects full body crops
        mx = max(1, int(bw * 0.05)) # 5%
        mt = max(1, int(bh * 0.02)) # 2%
        mb = max(1, int(bh * 0.10)) # 10%
        crop = frame[max(0, y1 + mt):min(H, y2 - mb),
                     max(0, x1 + mx):min(W, x2 - mx)]
        if crop.size == 0 or crop.shape[0] < 10 or crop.shape[1] < 10:
            return None, "empty_crop"

        is_good, reason = check_crop_quality(crop, bbox=bbox, frame_shape=frame.shape)
        if not is_good:
            return None, reason

        # Remove CLAHE to prevent deep-frying small noisy crops
        crop_enhanced = crop

        hc, wc = crop_enhanced.shape[:2]
        try:
            # [FIX] Test-Time Augmentation (TTA)
            # Instead of relying on a weak color histogram, we run the lightweight
            # OSNet model twice: once on the original crop, and once on a horizontally
            # flipped crop. We then average the two feature vectors. 
            # This is a standard ReID trick that massively improves robustness to pose
            # changes (e.g. seeing the left profile vs right profile) for almost zero cost.
            
            # Pass 1: Original
            embs1 = self.model.get_features(
                np.array([[0, 0, wc, hc]], dtype=np.float32), crop_enhanced)
            if embs1 is None or len(embs1) == 0:
                return None, "no_features"
            vec1 = np.array(embs1[0]).flatten().astype(np.float32)

            # Pass 2: Flipped
            crop_flipped = cv2.flip(crop_enhanced, 1)
            embs2 = self.model.get_features(
                np.array([[0, 0, wc, hc]], dtype=np.float32), crop_flipped)
            vec2 = np.array(embs2[0]).flatten().astype(np.float32)

            # Average and normalize
            deep_vec = (vec1 + vec2) / 2.0
            deep_norm = np.linalg.norm(deep_vec)
            if deep_norm < 1e-6:
                return None, "zero_deep_vec"
            
            emb = deep_vec / deep_norm

            if emb is not None:
                self._cache[session_id] = (emb, time.time())
            return emb, "ok"
        except Exception as e:
            print(f"[REID] get_features error: {e}")
            return None, "error"

    def clear_cache_for(self, session_id: int):
        self._cache.pop(session_id, None)


# ══════════════════════════════════════════════════════════════════════════════
# 3. PERSISTENT IDENTITY DATABASE
# ══════════════════════════════════════════════════════════════════════════════
class PersistentIdentityDB:
    def __init__(self, vector_dim: int = 1536, flush_on_start: bool = False):
        DB_DIR.mkdir(parents=True, exist_ok=True)
        self.dim = vector_dim
        self._gid_sample_count: dict[str, int] = {}
        self._gid_to_fids: dict[str, set[int]] = {}
        self._recent_gids: dict[str, float] = {}
        # [NEW] Person centroids for better matching
        self._gid_centroids: dict[str, np.ndarray] = {}
        self.faiss_face_id_to_gid = {}

        # [NEW] Attendance throttling — avoids hitting Postgres on every
        # single frame for every recognized face (see record_attendance()).
        self._attendance_last_check: dict[str, float] = {}
        self._last_faiss_save = time.monotonic()
        
        # Asynchronous database queue to completely decouple video tracking from PostgreSQL latency
        self._db_queue = queue.Queue()
        self._db_worker_thread = threading.Thread(target=self._db_worker, daemon=True)
        self._db_worker_thread.start()

        self._init_postgres()
        if flush_on_start:
            self._wipe()
        self._load_faiss()
        print(f"[DB] Ready — {self.index.ntotal} vectors, "
              f"{self._count_persons()} known persons.")

    def _init_postgres(self):
        try:
            self.conn = psycopg2.connect(
                dbname=PG_DBNAME, user=PG_USER, password=PG_PASS, host=PG_HOST)
            self.conn.autocommit = True
            with self.conn.cursor() as cur:
                cur.execute("CREATE SEQUENCE IF NOT EXISTS offline_gid_seq START 1")
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS offline_persons (
                        gid VARCHAR(50) PRIMARY KEY,
                        first_seen TIMESTAMP NOT NULL,
                        last_seen  TIMESTAMP NOT NULL,
                        visit_count  INTEGER DEFAULT 1,
                        sample_count INTEGER DEFAULT 1
                    )""")
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS offline_visits (
                        id SERIAL PRIMARY KEY,
                        gid VARCHAR(50) NOT NULL,
                        seen_at    TIMESTAMP NOT NULL,
                        source_file VARCHAR(255)
                    )""")
                # [NEW] Face-verified Session attendance log
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS attendance_sessions (
                        id SERIAL PRIMARY KEY,
                        date DATE,
                        name VARCHAR(50) NOT NULL,
                        in_time TIMESTAMP NOT NULL,
                        out_time TIMESTAMP,
                        status VARCHAR(20) DEFAULT 'Absent'
                    )""")
                
                # Safely add the 'date' and 'status' columns if the table was created in the previous step
                try:
                    cur.execute("ALTER TABLE attendance_sessions ADD COLUMN date DATE")
                    cur.execute("UPDATE attendance_sessions SET date = DATE(in_time) WHERE date IS NULL")
                except psycopg2.errors.DuplicateColumn:
                    pass
                except psycopg2.errors.InFailedSqlTransaction:
                    self.conn.rollback() # If it fails, we need to rollback to continue
                
                try:
                    cur.execute("ALTER TABLE attendance_sessions ADD COLUMN status VARCHAR(20) DEFAULT 'Absent'")
                    cur.execute("UPDATE attendance_sessions SET status = 'Absent' WHERE status IS NULL")
                except psycopg2.errors.DuplicateColumn:
                    pass
                except psycopg2.errors.InFailedSqlTransaction:
                    self.conn.rollback()
                cur.execute("CREATE INDEX IF NOT EXISTS idx_attendance_name_time "
                            "ON attendance_sessions (name, out_time DESC)")
        except Exception as e:
            sys.exit(f"[ERROR] PostgreSQL: {e}")

    def _db_worker(self):
        """Background thread that consumes all database write operations to eliminate tracking lag."""
        try:
            worker_conn = psycopg2.connect(
                dbname=PG_DBNAME, user=PG_USER, password=PG_PASS, host=PG_HOST)
            worker_conn.autocommit = True
        except Exception as e:
            print(f"[DB WORKER FATAL] Could not connect to Postgres: {e}")
            return

        while True:
            try:
                task = self._db_queue.get()
                if task[0] == "attendance":
                    _, gid, face_sim, now = task
                    self._process_attendance_sync(worker_conn, gid, face_sim, now)
                elif task[0] == "visit":
                    _, gid, source_name, now = task
                    self._process_visit_sync(worker_conn, gid, source_name, now)
                elif task[0] == "register":
                    _, gid, source_name, now = task
                    self._process_register_sync(worker_conn, gid, source_name, now)
                elif task[0] == "add_sample":
                    _, gid = task
                    self._process_add_sample_sync(worker_conn, gid)
                self._db_queue.task_done()
            except Exception as e:
                print(f"[DB WORKER ERROR] {e}")

    def _wipe(self):
        print("[DB] FLUSH_ON_START=True — wiping offline gallery ...")
        with self.conn.cursor() as cur:
            cur.execute("TRUNCATE TABLE offline_persons, offline_visits, attendance_sessions CASCADE")
            cur.execute("ALTER SEQUENCE offline_gid_seq RESTART WITH 1")
        import shutil
        for f in [FAISS_INDEX_FILE, FAISS_INDEX_FILE.with_suffix(".meta.json"), FAISS_FACE_INDEX_FILE, FAISS_FACE_INDEX_FILE.with_suffix(".meta.json")]:
            if f.exists():
                f.unlink()
        if AUDIT_DIR.exists():
            shutil.rmtree(AUDIT_DIR)

    def _load_faiss(self):
        if FAISS_INDEX_FILE.exists():
            self.index = faiss.read_index(str(FAISS_INDEX_FILE))
            meta = FAISS_INDEX_FILE.with_suffix(".meta.json")
            raw = json.loads(meta.read_text()) if meta.exists() else {}
            self.faiss_id_to_gid = {int(k): v for k, v in raw.items()}
            for fid, gid in self.faiss_id_to_gid.items():
                self._gid_sample_count[gid] = self._gid_sample_count.get(gid, 0) + 1
                self._gid_to_fids.setdefault(gid, set()).add(fid)
            # Rebuild centroids
            self._rebuild_centroids()
        else:
            self.index = faiss.IndexFlatIP(self.dim)
            self.faiss_id_to_gid = {}

        if FAISS_FACE_INDEX_FILE.exists():
            self.face_index = faiss.read_index(str(FAISS_FACE_INDEX_FILE))
            meta = FAISS_FACE_INDEX_FILE.with_suffix(".meta.json")
            raw = json.loads(meta.read_text()) if meta.exists() else {}
            self.faiss_face_id_to_gid = {int(k): v for k, v in raw.items()}
        else:
            self.face_index = faiss.IndexFlatIP(512)
            self.faiss_face_id_to_gid = {}


    def _rebuild_centroids(self):
        """[NEW] Build person centroids from samples."""
        self._gid_centroids = {}
        for gid, fids in self._gid_to_fids.items():
            if len(fids) > 0:
                vectors = []
                for fid in fids:
                    # Extract vector from index (faiss doesn't support direct extraction easily)
                    # Use search with the vector itself as query
                    pass  # Will be built incrementally

    def _save_faiss(self):
        faiss.write_index(self.index, str(FAISS_INDEX_FILE))
        FAISS_INDEX_FILE.with_suffix(".meta.json").write_text(
            json.dumps(self.faiss_id_to_gid))

        faiss.write_index(self.face_index, str(FAISS_FACE_INDEX_FILE))
        FAISS_FACE_INDEX_FILE.with_suffix(".meta.json").write_text(json.dumps(self.faiss_face_id_to_gid))


    def _update_recent_gids(self, gid: str):
        self._recent_gids[gid] = time.time()

    def _expire_recent_gids(self):
        now = time.time()
        expired = [g for g, ts in self._recent_gids.items()
                   if now - ts > TEMPORAL_MEMORY_TTL]
        for g in expired:
            del self._recent_gids[g]

    def _is_recently_seen(self, gid: str) -> bool:
        self._expire_recent_gids()
        return gid in self._recent_gids

    def _find_hint_in_results(self, sims, fids, target_gid: str):
        for sim, fid in zip(sims[0], fids[0]):
            if fid == -1:
                break
            if self.faiss_id_to_gid.get(fid) == target_gid:
                return (float(sim), int(fid))
        return None

    def search_for_gid_similarity(self, embedding: np.ndarray, target_gid: str) -> float | None:
        vec = embedding.astype(np.float32).reshape(1, -1)
        if self.index.ntotal == 0:
            return None
        k = min(self.index.ntotal, MAX_SAMPLES * 10)
        sims, fids = self.index.search(vec, k)
        result = self._find_hint_in_results(sims, fids, target_gid)
        return result[0] if result else None

    
    def insert_face_vector(self, gid: str, embedding: np.ndarray):
        vec = embedding.astype(np.float32).reshape(1, -1)
        fid = self.face_index.ntotal
        self.face_index.add(vec)
        self.faiss_face_id_to_gid[fid] = gid
        self._save_faiss()

    def search_face(self, embedding: np.ndarray, active_gids=None):
        if self.face_index.ntotal == 0:
            return None, 0.0
        vec = embedding.astype(np.float32).reshape(1, -1)
        k = min(self.face_index.ntotal, 5)
        sims, fids = self.face_index.search(vec, k)
        if active_gids is None:
            active_gids = set()
        best_gid, best_sim = None, 0.0
        for sim, fid in zip(sims[0], fids[0]):
            if fid == -1: break
            gid = self.faiss_face_id_to_gid.get(fid)
            if gid not in active_gids and float(sim) > best_sim:
                best_sim = float(sim)
                best_gid = gid
        return best_gid, best_sim

    def _get_best_match_per_person(self, sims, fids, exclude_active=None):
        """[NEW] Get best similarity for each person, not each sample."""
        if exclude_active is None:
            exclude_active = set()
        
        person_best = {}  # gid -> max_sim
        for sim, fid in zip(sims[0], fids[0]):
            if fid == -1:
                break
            gid = self.faiss_id_to_gid.get(fid)
            if gid is None:
                continue
            sim_f = float(sim)
            # [FIX] Do NOT exclude active here. Let resolve_identity handle it with spatial awareness.
            if gid not in person_best or sim_f > person_best[gid]:
                person_best[gid] = sim_f
        
        # Sort by similarity descending
        return sorted(person_best.items(), key=lambda x: x[1], reverse=True)

    def resolve_identity(self,
                          embedding: np.ndarray,
                          center: tuple,
                          active_gid_centers: dict,
                          previous_gid: str | None = None,
                          previous_gid_frames: int = 0,
                          probe_candidates: list | None = None,
                          probe_attempt: int = 0,
                          source_name: str = "unknown") -> tuple[str, list]:
        active_gids = set(active_gid_centers.keys())
        vec = embedding.astype(np.float32).reshape(1, -1)

        if self.index.ntotal == 0:
            return self._register_new(vec, source_name=source_name), []

        k = min(MAX_SAMPLES * 10, self.index.ntotal)  # [FIX] Search more samples
        sims, fids = self.index.search(vec, k)

        # [FIX] Use per-person best match instead of raw sample voting
        person_matches = self._get_best_match_per_person(sims, fids, active_gids)

        # ── GID stickiness ──
        if previous_gid and previous_gid_frames >= STICKY_FRAME_COUNT:
            prev_sim = None
            best_other_gid = None
            best_other_sim = 0.0
            
            for gid, sim in person_matches:
                if gid == previous_gid:
                    prev_sim = sim
                else:
                    if sim > best_other_sim:
                        best_other_gid = gid
                        best_other_sim = sim
            
            # --- ONLINE CONTINUOUS VERIFICATION (CHECKER) ---
            # 1. Merge into older dormant profile if it strongly matches (e.g., bad angle fixed)
            if best_other_gid is not None and best_other_gid not in active_gids:
                if best_other_sim >= 0.68:
                    try:
                        best_num = int(best_other_gid.split('-')[1])
                        prev_num = int(previous_gid.split('-')[1])
                        if best_num < prev_num:
                            print(f"[ONLINE CHECKER] Self-Correcting: Merging newer {previous_gid} into older {best_other_gid} "
                                  f"(sim={best_other_sim:.3f})")
                            self._record_visit(best_other_gid)
                            self._update_recent_gids(best_other_gid)
                            self._maybe_add_sample(vec, best_other_gid)
                            return best_other_gid, []
                    except Exception:
                        pass
                        
            # 2. Correct accidental crossing swaps: if they look significantly more like someone else now
            if best_other_gid is not None and best_other_sim >= STICKY_SWITCH_THRESH:
                if best_other_gid not in active_gids:
                    # If current sim is very low, or the other is much higher
                    if prev_sim is None or (best_other_sim - prev_sim >= 0.10):
                        print(f"[ONLINE CHECKER] Swap Override: {previous_gid} (sim={prev_sim if prev_sim else 0.0:.3f}) -> {best_other_gid} (sim={best_other_sim:.3f})")
                        self._record_visit(best_other_gid, source_name)
                        self._update_recent_gids(best_other_gid)
                        self._maybe_add_sample(vec, best_other_gid)
                        return best_other_gid, []

            if prev_sim is not None and prev_sim >= 0.45:
                # Normal keep
                self._record_visit(previous_gid, source_name)
                self._update_recent_gids(previous_gid)
                self._maybe_add_sample(vec, previous_gid)
                return previous_gid, []
            
            if best_other_gid is None or best_other_sim < STICKY_SWITCH_THRESH:
                # Normal reject
                self._record_visit(previous_gid, source_name)
                self._update_recent_gids(previous_gid)
                return previous_gid, []
            else:
                if best_other_gid in active_gids:
                    return previous_gid, []
                print(f"[STICKY] Allowing switch {previous_gid} -> {best_other_gid} "
                      f"(sim={best_other_sim:.3f})")
                self._record_visit(best_other_gid, source_name)
                self._update_recent_gids(best_other_gid)
                self._maybe_add_sample(vec, best_other_gid)
                return best_other_gid, []

        # ── PROBE voting ──
        if probe_candidates is not None:
            new_candidates = []
            for gid, sim in person_matches:
                thresh = REID_SIM_THRESH_MATCH

                # [FIX] Hard concurrency veto: a GID currently held by a
                # different live track can never be voted for here. (The old
                # "dist <= 60 ghost-box" override could let a genuinely
                # different, nearby person get folded into an active GID.)
                if gid in active_gids:
                    continue

                if self._is_recently_seen(gid):
                    thresh = TEMPORAL_MATCH_THRESH
                    
                if sim >= thresh:
                    new_candidates.append((gid, sim))

            # Accumulate candidates WITHOUT decay — decay was destroying valid votes.
            # [FIX] But always re-filter out any GID that has since become
            # active on a different track, even if it was voted for earlier
            # while still dormant — otherwise a stale vote could still hand
            # an active identity to a second, different person.
            all_candidates = [c for c in (probe_candidates + new_candidates)
                               if c[0] not in active_gids]

            gid_votes = Counter(gid for gid, _ in all_candidates)
            gid_best_sim = {}  # Track best sim per gid
            for gid, sim in all_candidates:
                if gid not in gid_best_sim or sim > gid_best_sim[gid]:
                    gid_best_sim[gid] = sim

            if gid_votes:
                best_gid, best_count = gid_votes.most_common(1)[0]
                best_sim = gid_best_sim[best_gid]
                
                # Determine threshold for final confirmation
                # Use lower threshold for recently-seen GIDs (they just left)
                if self._is_recently_seen(best_gid):
                    final_thresh = TEMPORAL_MATCH_THRESH
                else:
                    final_thresh = REID_SIM_THRESH_MATCH

                # [FIX] Require vote majority AND confidence. best_gid can no
                # longer be in active_gids (already filtered above), so no
                # separate distance check is needed here.
                if best_count >= PROBE_AGREE_THRESHOLD and best_sim >= final_thresh:
                    print(f"[PROBE] Confirmed {best_gid} with {best_count} votes, sim={best_sim:.3f}")
                    self._record_visit(best_gid, source_name)
                    self._update_recent_gids(best_gid)
                    self._maybe_add_sample(vec, best_gid)
                    return best_gid, []
                
                print(f"[PROBE DEBUG] attempt {probe_attempt}: best={best_gid}({best_count} votes, sim={best_sim:.3f}). All votes: {all_candidates}")

            # [FIX] This timeout must fire regardless of whether gid_votes is
            # empty. Previously it only ran in the `else` branch below, but
            # once ANY candidate was ever recorded, gid_votes stayed non-empty
            # forever (candidates never decayed/expired), so this branch was
            # unreachable and probes could run 40+ attempts instead of
            # PROBE_FRAMES, accumulating stale/spurious votes until an
            # unrelated GID crossed the 3-vote bar by chance.
            if probe_attempt >= PROBE_FRAMES:
                print(f"[PROBE] No match after {probe_attempt} attempts — new person")
                return self._register_new(vec, exclude_gids=active_gids, source_name=source_name), []

            return f"PROBE-{probe_attempt}", all_candidates

        # ── PATH 3: Standard immediate match ──
        best_match_gid = None
        best_match_sim = 0.0
        
        for gid, sim in person_matches:
            # [FIX] Hard concurrency veto: a GID currently held by a different
            # live track can NEVER be reassigned here. The previous "override"
            # (allow if dist<150 or sim>=0.85) could hand one physical
            # identity to two different simultaneously-visible people.
            if gid in active_gids:
                continue

            thresh = REID_SIM_THRESH_MATCH
            if self._is_recently_seen(gid):
                thresh = TEMPORAL_MATCH_THRESH
            
            if sim >= thresh and sim > best_match_sim:
                best_match_sim = sim
                best_match_gid = gid

        if best_match_gid is not None:
            tier = "MATCH"
            if self._is_recently_seen(best_match_gid):
                tier = "TEMPORAL"
            print(f"[{tier}] {best_match_gid} sim={best_match_sim:.3f}")
            self._record_visit(best_match_gid, source_name)
            self._update_recent_gids(best_match_gid)
            self._maybe_add_sample(vec, best_match_gid)
            return best_match_gid, []

        return self._register_new(vec, exclude_gids=active_gids, source_name=source_name), []

    def _register_new(self, vec: np.ndarray, exclude_gids: set | None = None, source_name: str = "unknown") -> str:
        exclude_gids = exclude_gids or set()
        # Reconcile check
        if self.index.ntotal > 0:
            k = min(MAX_SAMPLES * 10, self.index.ntotal)
            sims, fids = self.index.search(vec, k)
            person_matches = self._get_best_match_per_person(sims, fids)
            
            for gid, sim in person_matches:
                if sim < RECONCILE_SIM_THRESH:
                    break
                # [FIX] Never reconcile into a GID that's currently active on
                # a different live track — that would hand one identity to
                # two simultaneously-visible people.
                if gid in exclude_gids:
                    continue
                n_samples = self._gid_sample_count.get(gid, 1)
                min_sim = RECONCILE_SIM_THRESH if n_samples >= 2 else RECONCILE_SIM_SINGLE
                if sim < min_sim:
                    continue
                print(f"[RECONCILE] {sim:.3f} >= {min_sim} — "
                      f"treating as existing {gid}, not a new person")
                self._record_visit(gid, source_name)
                self._update_recent_gids(gid)
                self._maybe_add_sample(vec, gid)
                return gid

        with self.conn.cursor() as cur:
            cur.execute("SELECT nextval('offline_gid_seq')")
            n = cur.fetchone()[0]
        gid = f"GID-{n:05d}"
        new_fid = self.index.ntotal
        now = datetime.utcnow()
        self.index.add(vec)
        self.faiss_id_to_gid[new_fid] = gid
        self._gid_to_fids.setdefault(gid, set()).add(new_fid)
        
        # Send DB Insert to background queue (Zero lag)
        self._db_queue.put(("register", gid, source_name, now))
        
        self._gid_sample_count[gid] = 1
        print(f"[DB] New person: {gid}")
        return gid

    def _process_register_sync(self, worker_conn, gid: str, source_name: str, now: datetime):
        with worker_conn.cursor() as cur:
            cur.execute(
                "INSERT INTO offline_persons (gid, first_seen, last_seen) VALUES (%s,%s,%s)",
                (gid, now, now))
            cur.execute(
                "INSERT INTO offline_visits (gid, seen_at, source_file) VALUES (%s,%s,%s)",
                (gid, now, source_name))

    def _maybe_add_sample(self, vec: np.ndarray, gid: str):
        vec = vec.astype(np.float32).reshape(1, -1)
        existing = list(self._gid_to_fids.get(gid, set()))
        if len(existing) >= MAX_SAMPLES:
            return
        existing_set = set(existing)
        if existing:
            k = min(len(existing) + 5, self.index.ntotal)
            sims, fids = self.index.search(vec, k)
            for sim, fid in zip(sims[0], fids[0]):
                if fid in existing_set and sim > 0.85:
                    return

        # [FIX] Cross-GID contamination guard: before adding this vector
        # to the target GID's gallery, verify it actually looks more like
        # the target GID than any other GID. If a track was (even briefly)
        # assigned the wrong identity, this prevents the wrong person's
        # embedding from being permanently baked into the gallery.
        if self.index.ntotal > 0:
            k_check = min(self.index.ntotal, MAX_SAMPLES * 10)
            check_sims, check_fids = self.index.search(vec, k_check)
            best_per_person = self._get_best_match_per_person(check_sims, check_fids)
            target_sim = 0.0
            best_other_gid = None
            best_other_sim = 0.0
            for match_gid, match_sim in best_per_person:
                if match_gid == gid:
                    target_sim = match_sim
                elif match_sim > best_other_sim:
                    best_other_gid = match_gid
                    best_other_sim = match_sim
            # If this vector matches a DIFFERENT GID more strongly than the
            # target, it's contamination — don't add it.
            if best_other_gid and best_other_sim > target_sim + 0.05:
                print(f"[CONTAMINATION GUARD] Blocked sample for {gid}: "
                      f"vec matches {best_other_gid}({best_other_sim:.3f}) > {gid}({target_sim:.3f})")
                return

        new_fid = self.index.ntotal
        self.index.add(vec)
        self.faiss_id_to_gid[new_fid] = gid
        self._gid_to_fids.setdefault(gid, set()).add(new_fid)
        
        # Send DB Update to background queue (Zero lag)
        self._db_queue.put(("add_sample", gid))
        
        # Throttle FAISS disk saves to once per 30 seconds to avoid disk I/O lag
        now_mono = time.monotonic()
        if now_mono - self._last_faiss_save > 30.0:
            self._save_faiss()
            self._last_faiss_save = now_mono
            
        print(f"[DB] New angle for {gid} ({len(existing) + 1}/{MAX_SAMPLES})")

    def _process_add_sample_sync(self, worker_conn, gid: str):
        with worker_conn.cursor() as cur:
            cur.execute("UPDATE offline_persons SET sample_count=sample_count+1 WHERE gid=%s", (gid,))

    def _record_visit(self, gid: str, source_name: str = "unknown"):
        # Send DB write to the background queue immediately (Zero lag)
        now = datetime.utcnow()
        self._db_queue.put(("visit", gid, source_name, now))
        self._update_recent_gids(gid)

    def _process_visit_sync(self, worker_conn, gid: str, source_name: str, now: datetime):
        with worker_conn.cursor() as cur:
            cur.execute(
                "UPDATE offline_persons SET last_seen=%s, visit_count=visit_count+1 WHERE gid=%s",
                (now, gid))
            cur.execute(
                "INSERT INTO offline_visits (gid, seen_at, source_file) VALUES (%s,%s,%s)",
                (gid, now, source_name))

    def record_attendance(self, gid: str, face_sim: float | None = None):
        """[NEW] One row per person per calendar day.
        - First sighting of the day: opens the row (in_time = now, out_time = NULL).
        - Inside the 10 min timer: ignored.
        - After the 10 min timer: out_time is set/extended to now.
        - Seen again later the SAME day (after another 10 min gap): the
          existing row's out_time is just extended again — no new row.
        - Seen again on a DIFFERENT day: a new row is opened for that day.

        Timestamps are IST (now_ist()), so the `date` column matches the
        local calendar day.

        Throttled to at most once every ATTENDANCE_CHECK_INTERVAL_SEC per
        gid — this is called from the per-frame hot path, and the 10-minute
        timer doesn't need frame-level resolution. Without this, a recognized
        face sitting in view was hitting Postgres 15-30x/sec, which is what
        was causing the lag.

        Uses a per-gid threading.Lock (cheap, in-process) rather than a
        Postgres advisory lock, since the callers are threads within this
        same process — a DB-level lock would add two extra network
        round-trips per call for no benefit here.
        """
        now_mono = time.monotonic()
        last_check = self._attendance_last_check.get(gid, 0.0)
        if now_mono - last_check < ATTENDANCE_CHECK_INTERVAL_SEC:
            return  # Too soon since we last touched the DB for this person — skip.
        self._attendance_last_check[gid] = now_mono

        # Send to async background worker (Zero lag)
        now = now_ist()
        self._db_queue.put(("attendance", gid, face_sim, now))

    def _process_attendance_sync(self, worker_conn, gid: str, face_sim: float | None, now: datetime):
        today = now.date()
        with worker_conn.cursor() as cur:
            # Find the very last session for this person
            cur.execute("""
                SELECT id, date, in_time, out_time FROM attendance_sessions 
                WHERE name = %s 
                ORDER BY in_time DESC LIMIT 1
            """, (gid,))
            row = cur.fetchone()

            if row:
                session_id, session_date, in_time, out_time = row

                if session_date == today:
                    # Calculate total duration so far
                    duration_hours = (now - in_time).total_seconds() / 3600.0
                    new_status = "Present" if duration_hours >= REQUIRED_WORK_HOURS else "Absent"
                    
                    # Same calendar day -> continuously push out_time forward.
                    cur.execute("UPDATE attendance_sessions SET out_time = %s, status = %s WHERE id = %s", (now, new_status, session_id))
                    
                    # Only print to console if they were gone for a while, to avoid spam
                    if out_time is not None:
                        elapsed_since_last_seen = (now - out_time).total_seconds()
                        if elapsed_since_last_seen > ATTENDANCE_COOLDOWN_SEC:
                            print(f"[ATTENDANCE] {gid} -> RE-ENTRY (out_time extended) @ {now.isoformat(sep=' ', timespec='seconds')}")
                    return
                # else: different day falls through to opening a new row below

            # Either no prior session, or the last one belongs to an earlier day -> open a fresh row for today.
            cur.execute("INSERT INTO attendance_sessions (date, name, in_time, out_time, status) VALUES (%s, %s, %s, %s, 'Absent')", (today, gid, now, now))
            print(f"[ATTENDANCE] {gid} -> IN @ {now.isoformat(sep=' ', timespec='seconds')}"
                  + (f" (face_sim={face_sim:.2f})" if face_sim is not None else ""))

    def _count_persons(self) -> int:
        with self.conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM offline_persons")
            return cur.fetchone()[0]

    def close(self):
        try:
            self._save_faiss()
        except Exception:
            pass
        self.conn.close()


# ══════════════════════════════════════════════════════════════════════════════
# 4. OFFLINE TRACKER
# ══════════════════════════════════════════════════════════════════════════════
class OfflineTracker:
    def __init__(self, identity_db: PersistentIdentityDB, reid: ReIDExtractor, face_ext: FaceExtractor = None, source_name: str = "unknown"):
        self.identity_db = identity_db
        self.reid = reid
        self.face_ext = face_ext
        self.source_name = source_name

        self.tracker = BoTSORT(
            model_weights=REID_WEIGHTS,
            device=DEVICE,
            fp16=False,
            track_high_thresh=0.3,
            track_low_thresh=0.05,
            new_track_thresh=0.3,
        )
        self.tracker.model = reid.model

        self.session_map: dict[int, str] = {}
        self.gid_frame_counts: dict[int, int] = defaultdict(int)
        self.confirm_buffer = ConfirmationBuffer()
        self.probe_buffers: dict[int, list] = defaultdict(list)
        self._probe_attempts: dict[int, int] = defaultdict(int)
        self._last_bboxes: dict[int, tuple] = {}
        self._recently_lost: dict[int, tuple] = {}
        self._track_gid_history: dict[int, list[str]] = defaultdict(list)
        self.active_merge_map: dict[str, str] = {}
        self.frame_count = 0
        
        # [NEW] Throttle InsightFace so it doesn't run on every single frame!
        self._last_face_check: dict[int, int] = {}

        self.total_new_persons = 0
        self._audit_counts: dict[str, int] = defaultdict(int)
        # [FIX] Tracks the frame_count a crop was last saved for a given GID,
        # so the crops that accumulate now that _save_audit_crop is called
        # every frame (see the fix below) are spread across the person's
        # visible duration — different pose/angle/lighting — instead of
        # bunching up within the same second or two.
        self._audit_last_saved_frame: dict[str, int] = {}

    def _try_bridge(self, sid: int, embedding: np.ndarray, bbox: tuple,
                     active_gids: set, active_gid_centers: dict) -> str | None:
        """Attempt to re-link a dormant/probing track to a recently-lost GID.

        [FIX] Refactored out of `update()` so it can be called on EVERY
        frame a track is dormant or PROBE-ing, not just the single frame
        it first appears. Previously this logic only ran when
        `session_map.get(sid)` was still `None`; the moment a track fell
        through to PROBE-0 it could never be bridged again, even though
        `_recently_lost` entries were still valid and later frames had
        better embeddings.
        """
        if not self._recently_lost:
            return None

        x1, y1, x2, y2 = bbox
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

        # [DEBUG] Print all distances to all recently lost
        for lost_sid, (lbbox, lgid, _ts, _lost_frame) in self._recently_lost.items():
            lx1, ly1, lx2, ly2 = lbbox
            lcx, lcy = (lx1 + lx2) / 2, (ly1 + ly2) / 2
            d = ((cx - lcx) ** 2 + (cy - lcy) ** 2) ** 0.5
            top_sim = self.identity_db.search_for_gid_similarity(embedding, lgid)
            sim_val = top_sim if top_sim is not None else 0.0
            frame_gap = self.frame_count - _lost_frame
            print(f"[DEBUG BRIDGE] sid={sid} lost_gid={lgid} dist={d:.0f} sim={sim_val:.3f} frame_gap={frame_gap}")

        valid_candidates = []
        for lost_sid, (lbbox, lgid, _ts, _lost_frame) in self._recently_lost.items():
            if lgid in active_gids:
                continue
            lx1, ly1, lx2, ly2 = lbbox
            lcx, lcy = (lx1 + lx2) / 2, (ly1 + ly2) / 2
            d = ((cx - lcx) ** 2 + (cy - lcy) ** 2) ** 0.5

            if d < RECENTLY_LOST_MAX_DIST:
                top_sim = self.identity_db.search_for_gid_similarity(embedding, lgid)
                if top_sim is not None:
                    frame_gap = self.frame_count - _lost_frame

                    # Physics Constraint: Max 100px base + 15px per frame of movement.
                    # This is the feasibility gate — is it even possible this is
                    # the same person given how far they could have walked?
                    plausible_dist = 100 + (frame_gap * 15)
                    if d < plausible_dist:
                        # [FIX] Continuous confidence blend instead of hard
                        # tiers. space_conf is 1.0 right at the last known
                        # spot, fading to 0 at RECENTLY_LOST_MAX_DIST.
                        # time_conf is 1.0 the instant they vanish, fading to
                        # 0 as they approach the TTL. Either strong recency
                        # or strong spatial tightness should lower the bar —
                        # a person standing near where they vanished 30s ago,
                        # with no one else around, is good evidence even if
                        # appearance similarity alone is middling.
                        # [FIX] Time-decay the spatial confidence.
                        # If a person disappeared 20 seconds ago, the fact that a new track
                        # appeared in the exact same spot is meaningless (anyone could walk there).
                        # Spatial confidence should only discount the visual threshold
                        # if the track was lost very recently (e.g., within 90 frames / 3 seconds).
                        time_decay_for_space = max(0.0, 1.0 - (frame_gap / 90.0))
                        space_conf = max(0.0, 1.0 - d / RECENTLY_LOST_MAX_DIST) * time_decay_for_space
                        
                        # Time confidence naturally decays over the 1500 frame TTL
                        time_conf = max(0.0, 1.0 - frame_gap / RECENTLY_LOST_TTL_FRAMES)
                        
                        confidence = max(space_conf, time_conf)
                        dynamic_thresh = (BRIDGE_THRESH_CEILING
                                          - (BRIDGE_THRESH_CEILING - BRIDGE_THRESH_FLOOR) * confidence)
                    else:
                        dynamic_thresh = BRIDGE_THRESH_CEILING

                    if top_sim >= dynamic_thresh:
                        valid_candidates.append((top_sim, d, lost_sid, lgid, frame_gap))
                    else:
                        print(f"Rejected bridge for sid={sid} -> {lgid} "
                              f"(dist={d:.0f}px, sim={top_sim:.3f}, needed={dynamic_thresh:.3f})")

        if not valid_candidates:
            return None

        # Sort by (is_tight_spatial_match, similarity)
        def sort_key(x):
            sim, dist, l_sid, l_gid, f_gap = x
            is_tight = 1 if (dist < 100 and f_gap < 50) else 0
            return (is_tight, sim)

        valid_candidates.sort(reverse=True, key=sort_key)
        best_sim, best_dist, best_sid, bridged_gid, best_gap = valid_candidates[0]
        is_tight = best_dist < 100 and best_gap < 50

        # [FIX] Ambiguity guard. Loosening the thresholds above (so a
        # genuine reappearance isn't rejected outright) also means several
        # dormant identities can clear their bar in the SAME frame, all
        # mediocre matches. Previously the winner was just "whichever has
        # the highest raw sim" — with no check on whether a different GID
        # was nearly as good a fit. That silently produces a confident-
        # looking merge into the wrong identity when the evidence is
        # actually ambiguous. Skip the obvious case (a tight, same-spot
        # reacquisition) and otherwise require the winner to clearly beat
        # the runner-up before committing on a single frame.
        if not is_tight and len(valid_candidates) > 1:
            second_sim = valid_candidates[1][0]
            if best_sim - second_sim < BRIDGE_AMBIGUITY_MARGIN:
                second_gid = valid_candidates[1][3]
                print(f"Ambiguous bridge for sid={sid}: {bridged_gid}(sim={best_sim:.3f}) "
                      f"vs {second_gid}(sim={second_sim:.3f}) — deferring to PROBE")
                return None

        # [FIX] Contamination guard for the bridge itself. Before we hand this
        # track over to a GID, make absolutely sure the embedding doesn't
        # actually match a DIFFERENT dormant identity significantly better.
        # This stops the bridge from assigning GID-A to someone who
        # clearly looks like GID-B, which corrupts tracking state.
        vec = embedding.astype(np.float32).reshape(1, -1)
        if self.identity_db.index.ntotal > 0:
            k_check = min(self.identity_db.index.ntotal, MAX_SAMPLES * 10)
            check_sims, check_fids = self.identity_db.index.search(vec, k_check)
            best_per_person = self.identity_db._get_best_match_per_person(check_sims, check_fids)
            best_other_gid = None
            best_other_sim = 0.0
            for match_gid, match_sim in best_per_person:
                if match_gid == bridged_gid:
                    continue
                if match_sim > best_other_sim:
                    best_other_gid = match_gid
                    best_other_sim = match_sim
            if best_other_gid and best_other_sim > best_sim + 0.05:
                print(f"Rejected bridge for sid={sid} -> {bridged_gid} "
                      f"(CONTAMINATION GUARD: matches {best_other_gid} @ {best_other_sim:.3f} > {bridged_gid} @ {best_sim:.3f})")
                return None

        self._recently_lost.pop(best_sid)
        self.session_map[sid] = bridged_gid
        self.gid_frame_counts[sid] = STICKY_FRAME_COUNT
        active_gids.add(bridged_gid)
        # [FIX] Update active_gid_centers in-place (not just the local
        # active_gids set) so any other track resolved later in THIS SAME
        # frame also sees this GID as claimed.
        active_gid_centers[bridged_gid] = (cx, cy)
        self.identity_db._update_recent_gids(bridged_gid)
        self.identity_db._maybe_add_sample(embedding, bridged_gid)
        print(f"Bridged re-acquired track: sid={sid} -> {bridged_gid} "
              f"(dist={best_dist:.0f}px, sim={best_sim:.3f})")
        # A successful bridge supersedes any PROBE state this track was in.
        self.probe_buffers.pop(sid, None)
        self._probe_attempts.pop(sid, None)
        return bridged_gid

    def update(self, frame: np.ndarray, dets: np.ndarray) -> list[dict]:
        self.frame_count += 1
        raw_tracks = self.tracker.update(dets, frame)
        if raw_tracks is None or len(raw_tracks) == 0:
            return []

        current_sids = {int(t[4]) for t in raw_tracks}
        lost_sids = set(self.session_map.keys()) - current_sids
        now_ts = time.time()
        for sid in lost_sids:
            gid = self.session_map.pop(sid, None)
            if gid and not gid.startswith("PENDING") and not gid.startswith("PROBE"):
                last_bbox = self._last_bboxes.get(sid)
                if last_bbox is not None:
                    # Store (bbox, gid, time, frame_count)
                    self._recently_lost[sid] = (last_bbox, gid, now_ts, self.frame_count)

                # [FIX] Enrich the gallery with the freshest look at this
                # person right as they leave, instead of only ever adding
                # samples on a later successful match/bridge. A GID that
                # goes dormant with only its original (possibly awkward
                # angle) registration sample is fragile to bridge back to
                # after any real gap: this is exactly what caused a dormant
                # GID with one old sample to sit at sim~0.5 against a real
                # reappearance of the same person later and get rejected,
                # spawning a duplicate GID instead of reconnecting.
                fresh_emb = self.confirm_buffer.get_ema(sid)
                if fresh_emb is None:
                    cached = self.reid._cache.get(sid)
                    if cached is not None:
                        fresh_emb = cached[0]
                if fresh_emb is not None:
                    self.identity_db._maybe_add_sample(fresh_emb, gid)

            # [FIX] Don't remove confirm_buffer on track loss - keep for re-acquisition
            # self.confirm_buffer.remove(sid)
            self.probe_buffers.pop(sid, None)
            self._probe_attempts.pop(sid, None)
            self.gid_frame_counts.pop(sid, None)
            self._last_bboxes.pop(sid, None)
            self.reid.clear_cache_for(sid)

        expired = [s for s, (_, _, ts, lost_frame) in self._recently_lost.items()
                   if self.frame_count - lost_frame > RECENTLY_LOST_TTL_FRAMES]
        for s in expired:
            del self._recently_lost[s]

        occluded: set[int] = set()
        for i, tA in enumerate(raw_tracks):
            for j, tB in enumerate(raw_tracks):
                if i != j and calculate_iou(tA[:4], tB[:4]) > OCCLUSION_IOU:
                    occluded.add(int(tA[4]))
                    occluded.add(int(tB[4]))

        # ── [FIX] Duplicate-box (ghost) detection ──
        # YOLO/BoTSORT occasionally emit two live tracks for the same
        # physical person (near-total bbox overlap) rather than one — this
        # is a detection artifact, not two people standing on top of each
        # other. Left alone, each track used to run identity resolution
        # independently and could land on two different GIDs for one body.
        # Instead: collapse each such pair into a primary + ghost, and let
        # the ghost simply mirror whatever GID the primary resolves to,
        # without running its own ReID/identity resolution at all.
        def _bbox_area(b):
            return max(0, b[2] - b[0]) * max(0, b[3] - b[1])

        duplicate_of: dict[int, int] = {}
        sids_in_order = [int(t[4]) for t in raw_tracks]
        bboxes_by_sid = {int(t[4]): tuple(map(int, t[:4])) for t in raw_tracks}
        for i in range(len(raw_tracks)):
            sid_a = sids_in_order[i]
            if sid_a in duplicate_of:
                continue
            for j in range(i + 1, len(raw_tracks)):
                sid_b = sids_in_order[j]
                if sid_b in duplicate_of:
                    continue
                if calculate_iou(bboxes_by_sid[sid_a], bboxes_by_sid[sid_b]) < DUPLICATE_BOX_IOU:
                    continue
                gid_a = self.session_map.get(sid_a)
                gid_b = self.session_map.get(sid_b)
                a_confirmed = bool(gid_a) and not gid_a.startswith(("PENDING", "PROBE"))
                b_confirmed = bool(gid_b) and not gid_b.startswith(("PENDING", "PROBE"))
                if a_confirmed and not b_confirmed:
                    primary, ghost = sid_a, sid_b
                elif b_confirmed and not a_confirmed:
                    primary, ghost = sid_b, sid_a
                elif _bbox_area(bboxes_by_sid[sid_a]) >= _bbox_area(bboxes_by_sid[sid_b]):
                    primary, ghost = sid_a, sid_b
                else:
                    primary, ghost = sid_b, sid_a
                duplicate_of[ghost] = primary
                print(f"[DUPLICATE BOX] sid={ghost} treated as ghost of sid={primary} "
                      f"(iou>={DUPLICATE_BOX_IOU:.2f})")

        # [FIX] Sort so every primary is resolved before any ghost that
        # mirrors it (stable sort preserves original relative order).
        ordered_tracks = sorted(raw_tracks, key=lambda t: int(t[4]) in duplicate_of)

        # [FIX] Single-camera: don't exclude active GIDs from matching
        # Active GIDs should still be searchable for re-identification
        active_gids: set[str] = set()
        active_gid_centers = {}
        for t in raw_tracks:
            gid = self.session_map.get(int(t[4]))
            if gid and not gid.startswith("PENDING") and not gid.startswith("PROBE"):
                active_gids.add(gid)
                x1, y1, x2, y2 = map(int, t[:4])
                active_gid_centers[gid] = ((x1 + x2) / 2, (y1 + y2) / 2)

        results = []
        for t in ordered_tracks:
            x1, y1, x2, y2 = map(int, t[:4])
            sid = int(t[4])
            bbox = (x1, y1, x2, y2)
            self._last_bboxes[sid] = bbox

            # --- [NEW] Face is the Primary ID (Absolute Truth) ---
            face_override_gid = None
            
            # [SPEED FIX] Only run heavy InsightFace if we don't know who this is yet, 
            # OR if we haven't checked their face in the last 30 frames (about 1-2 seconds of actual video).
            old_gid = self.session_map.get(sid)
            needs_identification = old_gid is None or old_gid.startswith(("PENDING", "PROBE"))
            frames_since_last_check = self.frame_count - self._last_face_check.get(sid, -999)
            
            should_run_face = needs_identification or (frames_since_last_check > 30)

            if self.face_ext is not None and should_run_face and not (sid in duplicate_of) and not (sid in occluded):
                self._last_face_check[sid] = self.frame_count
                fcrop = frame[max(0, y1):min(frame.shape[0], y2), max(0, x1):min(frame.shape[1], x2)]
                # Ensure height is at least decent for face detection
                if fcrop.size > 0 and (y2 - y1) >= 80:
                    # Ignore the print output of extract_face by suppressing stdout if needed, 
                    # but it's fine. We just extract face.
                    f_emb, _, fbox = self.face_ext.extract_face(fcrop)
                    if f_emb is not None:
                        f_gid, f_sim = self.identity_db.search_face(f_emb)
                        if f_gid and f_sim >= 0.45:
                            face_override_gid = f_gid
                            print(f"[{'FACE OVERRIDE'}] sid={sid} -> {f_gid} (sim={f_sim:.2f})")

                            # [NEW] This is the moment the person is FACE-verified.
                            # Log IN the first time we see them; if they're seen again
                            # within the cooldown window, do nothing; once the cooldown
                            # has elapsed, log the opposite event (OUT, then IN, ...).
                            if f_sim >= FACE_VERIFY_SIM_THRESH:
                                self.identity_db.record_attendance(f_gid, face_sim=f_sim)

            if face_override_gid is not None:
                old_gid = self.session_map.get(sid)
                if old_gid and old_gid != face_override_gid and not old_gid.startswith(("PENDING", "PROBE")):
                    print(f"[FACE CORRECTION] Mid-stream correction: {old_gid} -> {face_override_gid}")
                    # If the old GID was just a mistake, we might want to tell the user, but we just swap it.
                    
                self.session_map[sid] = face_override_gid
                self.probe_buffers.pop(sid, None)
                self._probe_attempts.pop(sid, None)
                
                gid = face_override_gid
                self.gid_frame_counts[sid] = self.gid_frame_counts.get(sid, 0) + 1
                active_gids.add(gid)
                active_gid_centers[gid] = ((x1 + x2) / 2, (y1 + y2) / 2)
                
                results.append({"bbox": bbox, "session_id": sid, "gid": gid, "occluded": False})
                continue

            if sid in duplicate_of:
                # [FIX] Ghost box: never run independent identity resolution.
                # Just mirror the primary's current GID for this frame,
                # whatever state it's in (confirmed, PENDING, or PROBE-N).
                primary_sid = duplicate_of[sid]
                mirrored_gid = self.session_map.get(primary_sid, f"PENDING-{primary_sid}")
                self.session_map[sid] = mirrored_gid
                self.gid_frame_counts[sid] = self.gid_frame_counts.get(primary_sid, 0)
                results.append({"bbox": bbox, "session_id": sid,
                                 "gid": mirrored_gid, "occluded": True})
                continue

            gid = self.session_map.get(sid)

            if gid is None or gid.startswith("PENDING") or gid.startswith("PROBE"):
                if sid in occluded:
                    # [FIX] Don't wipe confirm buffer on occlusion
                    # Instead, use EMA if available
                    ema_emb = self.confirm_buffer.get_ema(sid)
                    if ema_emb is not None and gid is not None and not gid.startswith(("PENDING", "PROBE")):
                        # Keep existing GID during occlusion if we have history
                        self.gid_frame_counts[sid] += 1
                        results.append({"bbox": bbox, "session_id": sid,
                                         "gid": gid, "occluded": True})
                        continue
                    
                    gid = f"PENDING-{sid}"
                    self.session_map[sid] = gid
                else:
                    embedding, quality_reason = self.reid.extract(frame, bbox, sid)

                    if embedding is None:
                        if sid in self.session_map and not self.session_map[sid].startswith(("PENDING", "PROBE")):
                            gid = self.session_map[sid]
                            self.gid_frame_counts[sid] += 1
                        else:
                            gid = f"PENDING-{sid}"
                            self.session_map[sid] = gid
                    else:
                        bridge_emb = self.confirm_buffer.get_ema(sid)
                        if bridge_emb is None:
                            bridge_emb = embedding
                        bridged_gid = self._try_bridge(sid, bridge_emb, bbox, active_gids, active_gid_centers)
                        if bridged_gid is not None:
                            gid = bridged_gid
                            results.append({"bbox": bbox, "session_id": sid,
                                             "gid": gid, "occluded": False})
                            continue
                        confirmed_vec = self.confirm_buffer.add(sid, embedding)
                        if confirmed_vec is not None:
                            previous_gid = self.session_map.get(sid)
                            previous_gid_frames = self.gid_frame_counts.get(sid, 0)
                            if previous_gid and previous_gid.startswith(("PENDING", "PROBE")):
                                previous_gid, previous_gid_frames = None, 0

                            probe_candidates = self.probe_buffers.get(sid, [])
                            probe_attempt = self._probe_attempts.get(sid, 0)

                            # Calculate center of current bbox
                            cx = (bbox[0] + bbox[2]) / 2
                            cy = (bbox[1] + bbox[3]) / 2
                            
                            real_gid, new_probe = self.identity_db.resolve_identity(
                                confirmed_vec,
                                center=(cx, cy),
                                active_gid_centers=active_gid_centers,
                                previous_gid=previous_gid,
                                previous_gid_frames=previous_gid_frames,
                                probe_candidates=probe_candidates,
                                probe_attempt=probe_attempt,
                                source_name=self.source_name,
                            )

                            if real_gid.startswith("PROBE"):
                                self.probe_buffers[sid] = new_probe
                                self._probe_attempts[sid] = probe_attempt + 1
                                gid = real_gid
                                self.session_map[sid] = gid
                            else:
                                self.probe_buffers.pop(sid, None)
                                self._probe_attempts.pop(sid, None)
                                self.gid_frame_counts[sid] = (
                                    self.gid_frame_counts.get(sid, 0) + 1
                                    if previous_gid == real_gid else 1
                                )
                                self.session_map[sid] = real_gid
                                gid = real_gid
                                if not real_gid.startswith(("PENDING", "PROBE")):
                                    active_gids.add(real_gid)
                                    # [FIX] Same as the bridging case: make
                                    # this claim visible to any other track
                                    # still to be resolved in this frame.
                                    active_gid_centers[real_gid] = (cx, cy)
                                    self._track_gid_history[sid].append(real_gid)
                                    print(f"Confirmed: {real_gid} (sticky_frames={self.gid_frame_counts[sid]})")
                                    self._save_audit_crop(frame, bbox, real_gid)
                        else:
                            gid = self.session_map.get(sid, f"PENDING-{sid}")
                            self.session_map[sid] = gid
            else:
                self.gid_frame_counts[sid] += 1
                # [FIX] Previously _save_audit_crop was only ever called once,
                # at the exact frame a track transitioned from PENDING/PROBE
                # into a confirmed GID (see the branch above). Every later
                # frame of that same track skipped this whole if/else's "if"
                # side, so the audit folder ended up with a single crop per
                # person no matter how long they were visible afterward.
                # Call it here too, on every frame the track is already
                # confirmed, so multiple crops accumulate up to
                # AUDIT_CROPS_PER_GID. Skip occluded frames — an overlapping
                # box would save a crop containing part of another person.
                if not gid.startswith(("PENDING", "PROBE")) and sid not in occluded:
                    self._save_audit_crop(frame, bbox, gid)

            results.append({"bbox": bbox, "session_id": sid,
                             "gid": gid, "occluded": sid in occluded})
                             
        # --- CONTINUOUS ONLINE VERIFICATION ---
        # Every 15 frames, check all active tracks to see if their EMA now matches an older dormant track
        if self.frame_count % 15 == 0:
            for sid, gid in self.session_map.items():
                if gid and not gid.startswith(("PENDING", "PROBE")) and sid in self.confirm_buffer._bufs:
                    ema_vec = self.confirm_buffer.get_ema(sid)
                    if ema_vec is not None:
                        # Find best match in DB
                        k = min(MAX_SAMPLES * 10, self.identity_db.index.ntotal)
                        if k > 0:
                            sims, fids = self.identity_db.search(ema_vec.astype(np.float32).reshape(1, -1), k)
                            person_matches = self.identity_db._get_best_match_per_person(sims, fids, active_gids)
                            
                            best_other_gid = None
                            best_other_sim = 0.0
                            for m_gid, m_sim in person_matches:
                                if m_gid != gid and m_gid not in active_gids:
                                    if m_sim > best_other_sim:
                                        best_other_sim = m_sim
                                        best_other_gid = m_gid
                            
                            # Merge threshold > 0.75
                            if best_other_gid and best_other_sim > 0.75:
                                print(f"[CONTINUOUS VERIFICATION] Mid-stream merge! {gid} -> {best_other_gid} (sim={best_other_sim:.3f})")
                                self.active_merge_map[gid] = best_other_gid
                                self.session_map[sid] = best_other_gid
                                
                                # Update the results for THIS frame so it's instantly correct
                                for res in results:
                                    if res["session_id"] == sid:
                                        res["gid"] = best_other_gid
                                        
        return results

    def _save_audit_crop(self, frame: np.ndarray, bbox: tuple, gid: str):
        """Save high-quality crops for offline clustering using a Dynamic Enrollment Zone.
        
        Enforces a strict geometric rule: Bounding box must be >= 200px tall and 
        at least 20px away from the left, right, and top edges of the frame.
        If these are met, we bypass margin trimming and save the perfect, full-body crop.
        """
        if self._audit_counts[gid] >= AUDIT_CROPS_PER_GID:
            return

        last_saved = self._audit_last_saved_frame.get(gid)
        if last_saved is not None and (self.frame_count - last_saved) < AUDIT_MIN_FRAME_GAP:
            return

        x1, y1, x2, y2 = bbox
        H, W = frame.shape[:2]
        
        # Dynamic Enrollment Zone Check
        bw, bh = x2 - x1, y2 - y1
        if bh < 200:
            return # Rule 1: Height must be >= 200px
            
        EDGE_MARGIN = 20
        if x1 < EDGE_MARGIN or y1 < EDGE_MARGIN or x2 > W - EDGE_MARGIN:
            return # Rule 2: Must be fully in frame (not touching left, top, or right edges)

        # Since it passed the strict dynamic rules, we take the full, untrimmed bounding box!
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(W, x2), min(H, y2)

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return

        # Standard quality gate (aspect ratio, brightness, contrast)
        is_good, reason = check_crop_quality(crop, bbox=bbox, frame_shape=frame.shape)
        if not is_good:
            return

        gid_dir = AUDIT_DIR / gid
        gid_dir.mkdir(parents=True, exist_ok=True)
        idx = self._audit_counts[gid]
        cv2.imwrite(str(gid_dir / f"{idx:02d}.jpg"), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])

        if self.face_ext is not None and bh >= 100:
            face_emb, face_img, _ = self.face_ext.extract_face(crop)
            if face_emb is not None:
                self.identity_db.insert_face_vector(gid, face_emb)
                cv2.imwrite(str(gid_dir / f"face_{idx:02d}.jpg"), face_img, [cv2.IMWRITE_JPEG_QUALITY, 95])

        self._audit_counts[gid] += 1
        self._audit_last_saved_frame[gid] = self.frame_count


# ══════════════════════════════════════════════════════════════════════════════
# 5. MAIN
# ══════════════════════════════════════════════════════════════════════════════
def get_dets(yolo, frame):
    r = yolo.predict(frame, conf=CONF_THRESH, iou=NMS_IOU_THRESH, verbose=False, classes=[0])
    if r[0].boxes is not None and len(r[0].boxes):
        b = r[0].boxes
        return np.hstack([b.xyxy.cpu().numpy(),
                           b.conf.cpu().numpy()[:, None],
                           b.cls.cpu().numpy()[:, None]])
    return np.empty((0, 6))


class ThreadedCamera:
    def __init__(self, src):
        if isinstance(src, int):
            self.cap = cv2.VideoCapture(src, cv2.CAP_DSHOW)
        else:
            self.cap = cv2.VideoCapture(src)
        self.ret, self.frame = self.cap.read()
        self.new_frame_event = threading.Event()
        self.new_frame_event.set()
        self.stopped = False
        self.t = threading.Thread(target=self.update, args=())
        self.t.daemon = True
        self.t.start()

    def update(self):
        while not self.stopped:
            ret, frame = self.cap.read()
            if not ret:
                self.stopped = True
            else:
                self.ret, self.frame = ret, frame
                self.new_frame_event.set()

    def read(self):
        self.new_frame_event.wait()
        self.new_frame_event.clear()
        return self.ret, self.frame.copy() if self.frame is not None else None

    def release(self):
        self.stopped = True
        self.t.join()
        self.cap.release()

    def get(self, propId):
        return self.cap.get(propId)

def run(input_source, output_video):
    print("\n=== Offline Re-ID Tracker (BoTSORT + Persistent Gallery) ===")

    is_live = False
    
    if str(input_source).isdigit():
        input_source = int(input_source)
        is_live = True
    elif isinstance(input_source, str) and input_source.startswith("rtsp://"):
        is_live = True
    elif isinstance(input_source, str) and not os.path.isfile(input_source):
        sys.exit(f"[ERROR] Video not found: {input_source}")

    # Use ThreadedCamera for ALL live streams to prevent buffer lag (process late)
    if is_live:
        print(f"[INIT] Starting threaded capture for live stream '{input_source}'...")
        cap = ThreadedCamera(input_source)
    else:
        cap = cv2.VideoCapture(input_source)
        
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    effective_fps = src_fps / FRAME_STRIDE
    print(f"[INFO] {INPUT_SOURCE}")
    print(f"[INFO] {W}x{H} @ {src_fps:.1f}fps, {total_frames} frames "
          f"(stride={FRAME_STRIDE} -> writing at {effective_fps:.1f}fps)")

    print("[INIT] Loading YOLO ...")
    yolo = YOLO(YOLO_MODEL)

    print("[INIT] Connecting to DB ...")
    identity_db = PersistentIdentityDB(vector_dim=REID_VECTOR_DIM, flush_on_start=FLUSH_ON_START)

    print("[INIT] Loading ReID model ...")
    reid = ReIDExtractor(REID_WEIGHTS, DEVICE)
    face_ext = FaceExtractor(DEVICE)

    tracker = OfflineTracker(identity_db, reid, face_ext)
    
    if OUTPUT_VIDEO and str(OUTPUT_VIDEO).lower() != "none":
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, effective_fps, (W, H))
    else:
        out = None
    
    rng = np.random.default_rng(42)
    gid_colors = {}
    def color_for(gid):
        if gid not in gid_colors:
            gid_colors[gid] = tuple(int(c) for c in rng.integers(80, 220, 3))
        return gid_colors[gid]

    frame_idx = 0
    processed = 0
    seen_gids: set[str] = set()
    t_start = time.time()
    frames_written_so_far = 0

    log_path = "D:/cdac project/reid/tracking_log_stride2.jsonl"
    with open(log_path, "w") as f:
        pass # clear the log file

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % FRAME_STRIDE != 0:
            frame_idx += 1
            continue

        dets = get_dets(yolo, frame)
        tracks = tracker.update(frame, dets)

        log_entry = {
            "frame_idx": frame_idx,
            "tracks": []
        }

        for t in tracks:
            log_entry["tracks"].append({
                "gid": t["gid"],
                "bbox": t["bbox"],
                "occluded": t["occluded"]
            })
            
        with open("D:/cdac project/reid/tracking_log_stride2.jsonl", "a") as f_log:
            f_log.write(json.dumps(log_entry) + "\n")

        for t in tracks:
            x1, y1, x2, y2 = map(int, t["bbox"])
            gid = t["gid"]
            occluded = t["occluded"]
            
            if not gid.startswith(("PENDING", "PROBE")):
                seen_gids.add(gid)
                
            # Render to frame
            is_p = gid.startswith(("PENDING", "PROBE"))
            if is_p:
                col, label = (160, 160, 160), "..."
            elif occluded:
                col, label = (0, 0, 210), gid
            else:
                col, label = color_for(gid), gid
                
            cv2.rectangle(frame, (x1, y1), (x2, y2), col, 2)
            
            try:
                fcrop = frame[max(0, int(y1)):min(frame.shape[0], int(y2)), max(0, int(x1)):min(frame.shape[1], int(x2))]
                if fcrop.size > 0:
                    f_emb, _, fbox = face_ext.extract_face(fcrop)
                    if fbox is not None:
                        fx1, fy1, fx2, fy2 = fbox
                        face_text = "[FACE DETECTED]"
                        if f_emb is not None:
                            f_gid, f_sim = identity_db.search_face(f_emb)
                            if f_gid:
                                if f_sim >= FACE_VERIFY_SIM_THRESH:
                                    face_text = f"[{f_gid} {f_sim:.2f}]"
                                    # NOTE: attendance is already recorded in update()'s
                                    # face-override branch — don't call record_attendance()
                                    # again here, this loop is draw-only.
                                else:
                                    face_text = f"[UNKNOWN FACE {f_sim:.2f}]"
                        cv2.rectangle(frame, (x1 + fx1, y1 + fy1), (x1 + fx2, y1 + fy2), (0, 255, 0), 2)
                        cv2.putText(frame, face_text, (x1 + fx1, max(y1, y1 + fy1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            except Exception as e:
                pass

            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
            ty = max(y1, th + 10)
            cv2.rectangle(frame, (x1, ty - th - 10), (x1 + tw + 8, ty), col, -1)
            cv2.putText(frame, label, (x1 + 4, ty - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

        if is_live:
            # Sync output video to real-time
            now = time.time()
            expected_total_frames = int((now - t_start) * effective_fps)
            frames_to_write = expected_total_frames - frames_written_so_far
            if frames_to_write < 1:
                frames_to_write = 1
            if out is not None:
                for _ in range(frames_to_write):
                    out.write(frame)
            frames_written_so_far += frames_to_write
        else:
            if out is not None:
                out.write(frame)
        
        cv2.imshow('Live Tracking', cv2.resize(frame, (W//2, H//2)))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        processed += 1
        frame_idx += 1

        if frame_idx % 100 == 0:
            elapsed = time.time() - t_start
            pct = 100 * frame_idx / max(1, total_frames)
            print(f"[PROGRESS] frame {frame_idx}/{total_frames} ({pct:.0f}%) "
                  f"— {elapsed:.0f}s elapsed, {len(seen_gids)} unique person(s) so far")

    cap.release()
    if out is not None:
        out.release()
    identity_db.close()

    print(f"\n[SUCCESS] Done. {len(seen_gids)} unique person(s) in this video.")
    print(f"          GIDs: {sorted(seen_gids)}")
    print(f"          Output Video: {output_video}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Offline Re-ID Tracker")
    parser.add_argument("--input", type=str, default=INPUT_SOURCE, help="Input video source path")
    parser.add_argument("--output", type=str, default=OUTPUT_VIDEO, help="Output video path")
    args = parser.parse_args()
    
    run(args.input, args.output)