import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis

class FaceExtractor:
    def __init__(self, device='cpu'):
        self.device = device
        print(f"[FACE] Loading ArcFace (insightface buffalo_l) to {device}...")
        
        # insightface uses ctx_id: 0 for GPU, -1 for CPU
        ctx_id = 0 if 'cuda' in str(device).lower() or str(device) == '0' else -1
        
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if ctx_id == 0 else ['CPUExecutionProvider']
        self.app = FaceAnalysis(name='buffalo_l', providers=providers)
        self.app.prepare(ctx_id=ctx_id, det_size=(640, 640), det_thresh=0.35)
    
    def extract_face(self, bgr_image, pad_ratio=0.15):
        """
        Given a BGR image crop of a person, detects the face and returns a 512-dim embedding.
        Includes padding and profile-aware geometry validation.
        """
        if bgr_image is None or bgr_image.size == 0:
            return None, None, None
            
        H_orig, W_orig = bgr_image.shape[:2]

        # Apply padding around tight body crop to retain head context for profile faces
        if pad_ratio > 0:
            pad_h = int(H_orig * pad_ratio)
            pad_w = int(W_orig * pad_ratio)
            padded_img = cv2.copyMakeBorder(
                bgr_image, pad_h, pad_h, pad_w, pad_w,
                borderType=cv2.BORDER_REFLECT_101
            )
            target_img = padded_img
            offset_x, offset_y = pad_w, pad_h
        else:
            target_img = bgr_image
            offset_x, offset_y = 0, 0

        # CLAHE Contrast Enhancement to boost side profile face detection under shadows
        try:
            lab = cv2.cvtColor(target_img, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            cl = clahe.apply(l)
            enhanced_img = cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)
        except Exception:
            enhanced_img = target_img

        faces = self.app.get(enhanced_img)
        if not faces:
            faces = self.app.get(target_img)
        if not faces:
            return None, None, None

        # Filter out false detections (ears, hair, noise) using landmark geometry & confidence
        valid_faces = []
        for f in faces:
            if f.det_score < 0.45:
                continue
            box = f.bbox.astype(int)
            fw = max(1, box[2] - box[0])
            fh = max(1, box[3] - box[1])
            
            # Require minimum face box dimensions
            if fw < 16 or fh < 16:
                continue
                
            # Facial Landmark Geometry Validation
            kps = f.kps
            if kps is None or len(kps) < 5:
                continue
                
            left_eye, right_eye = kps[0], kps[1]
            nose = kps[2]
            mouth_left, mouth_right = kps[3], kps[4]
            
            eye_center = (left_eye + right_eye) / 2.0
            mouth_center = (mouth_left + mouth_right) / 2.0

            eye_dist = np.linalg.norm(left_eye - right_eye)
            nose_eye_dist = np.linalg.norm(nose - eye_center)
            vertical_span = np.linalg.norm(mouth_center - eye_center)
            
            # Ear / hair texture signature: vertical span is collapsed (<0.14 of face height) or eye dist < 5px
            if vertical_span < (0.14 * fh) and eye_dist < 6.0:
                continue

            # Real faces must have minimum eye-to-mouth vertical geometry
            if vertical_span < (0.12 * fh):
                continue
                
            valid_faces.append(f)

        if not valid_faces:
            return None, None, None
            
        # Pick the best valid face by detection score
        valid_faces.sort(key=lambda x: x.det_score, reverse=True)
        best_face = valid_faces[0]
        
        # Adjust bounding box coordinates back to original unpadded crop frame
        box = best_face.bbox.astype(int)
        x1 = max(0, box[0] - offset_x)
        y1 = max(0, box[1] - offset_y)
        x2 = min(W_orig, box[2] - offset_x)
        y2 = min(H_orig, box[3] - offset_y)
        
        face_crop_bgr = bgr_image[y1:y2, x1:x2]
        if face_crop_bgr.size == 0:
            face_crop_bgr = None
            
        # normed_embedding is exactly what FAISS IP search needs
        embedding_np = best_face.normed_embedding
        if embedding_np is None:
            embedding_np = best_face.embedding
            norm = np.linalg.norm(embedding_np)
            if norm > 0:
                embedding_np = embedding_np / norm
                
        return embedding_np, face_crop_bgr, [x1, y1, x2, y2]
