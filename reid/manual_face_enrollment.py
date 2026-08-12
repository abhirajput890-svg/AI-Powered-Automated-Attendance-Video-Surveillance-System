import cv2
import sys
import numpy as np
from pathlib import Path

sys.path.append("D:/cdac project/reid")
from face_extractor import FaceExtractor
from offline_tracking_faces import PersistentIdentityDB

def enroll_manual_face(gid, image_path):
    print(f"\n--- Enrolling Face for {gid} ---")
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error loading {image_path}")
        return

    # Resize for display if too large
    display_img = img.copy()
    h, w = display_img.shape[:2]
    scale = 1.0
    if h > 800 or w > 1200:
        scale = min(800/h, 1200/w)
        display_img = cv2.resize(display_img, (0,0), fx=scale, fy=scale)

    print("Please draw a bounding box around the person's FRONT FACE.")
    print("Drag with mouse. Press ENTER or SPACE to confirm. Press 'c' to cancel.")
    
    # Let user select ROI
    window_name = f"Select FACE for {gid}"
    roi = cv2.selectROI(window_name, display_img, showCrosshair=True, fromCenter=False)
    cv2.destroyAllWindows()
    
    if roi == (0, 0, 0, 0):
        print("Selection cancelled.")
        return
        
    # Scale ROI back to original image size
    x, y, w_box, h_box = [int(v / scale) for v in roi]
    
    # Extract the face crop
    face_crop = img[y:y+h_box, x:x+w_box]
    
    print("\nRunning ArcFace on your selected region...")
    # Initialize face extractor
    face_ext = FaceExtractor(device='cpu')
    
    # We relax the strict filter just for manual enrollment since the user guarantees it's a face!
    face_ext.app.prepare(ctx_id=-1, det_size=(640, 640), det_thresh=0.1)
    
    # Get embedding
    faces = face_ext.app.get(face_crop)
    if not faces:
        print("ERROR: ArcFace could not detect any facial features in your selected box!")
        return
        
    faces.sort(key=lambda x: x.det_score, reverse=True)
    best_face = faces[0]
    
    embedding = best_face.normed_embedding
    if embedding is None:
        embedding = best_face.embedding
        embedding = embedding / np.linalg.norm(embedding)
        
    print(f"SUCCESS: Extracted 512-dim face embedding! (det_score: {best_face.det_score:.4f})")
    
    # Save to database
    # Connect to the DB (do NOT flush, we want to append or create new)
    db = PersistentIdentityDB(vector_dim=512, flush_on_start=False)
    
    # Remove any existing face vectors for this GID so we ONLY use your manual one
    # Note: FAISS doesn't support easy deletion, but we can just clear the Python tracking dicts for it
    # For a clean slate, let's just insert it.
    db.insert_face_vector(gid, embedding)
    
    print(f"Saved manual face embedding for {gid} to FAISS database!")
    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Manually enroll a face")
    parser.add_argument("--gid", required=True, help="Global ID (e.g. GID-00001)")
    parser.add_argument("--image", required=True, help="Path to the image containing the person")
    
    args = parser.parse_args()
    enroll_manual_face(args.gid, args.image)
