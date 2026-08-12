import cv2
import sys
import numpy as np
from pathlib import Path

sys.path.append("D:/cdac project/reid")
from face_extractor import FaceExtractor
from offline_tracking_faces import PersistentIdentityDB

def enroll_auto():
    db_dir = Path("D:/cdac project/database")
    print(f"\n--- Starting AUTO Face Enrollment from {db_dir} ---")
    
    face_ext = FaceExtractor(device='cpu')
    face_ext.app.prepare(ctx_id=-1, det_size=(640, 640), det_thresh=0.1)
    db = PersistentIdentityDB(vector_dim=512, flush_on_start=False)
    
    for gid_dir in sorted(db_dir.iterdir()):
        if not gid_dir.is_dir(): continue
        
        gid = gid_dir.name
        print(f"\n>>> Auto Enrolling {gid} <<<")
        faces_saved = 0
        
        image_paths = []
        for ext in ["*.jpg", "*.jpeg", "*.png"]:
            image_paths.extend(gid_dir.glob(ext))
            
        for img_path in sorted(image_paths, reverse=True):
            img = cv2.imread(str(img_path))
            if img is None: continue
            
            # Feed the entire image directly without ROI
            faces = face_ext.app.get(img)
            if not faces:
                print(f"ERROR: ArcFace could not detect any face in {img_path.name}")
                continue
                
            faces.sort(key=lambda f: f.det_score, reverse=True)
            best_face = faces[0]
            
            embedding = best_face.normed_embedding
            if embedding is None:
                embedding = best_face.embedding
                embedding = embedding / np.linalg.norm(embedding)
                
            print(f"[+] SUCCESS: Extracted face embedding! (score: {best_face.det_score:.4f})")
            db.insert_face_vector(gid, embedding)
            print(f"[+] Saved to database for {gid}!")
            faces_saved += 1
            break # Only save 1 best face per person for now

        print(f"Total faces saved for {gid}: {faces_saved}")
        if faces_saved == 0:
            print(f"WARNING: No face saved for {gid}!")

if __name__ == "__main__":
    enroll_auto()
