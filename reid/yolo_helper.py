import cv2
import numpy as np
from pathlib import Path

def get_yolo_crop(img_path: Path):
    """
    Given an image path (e.g. tracks/GID-00001/frames/20260710/12345.jpg),
    finds the corresponding YOLO label, loads the image, crops it to the bounding box,
    and returns the cropped BGR image.
    If no label is found, returns None.
    """
    label_path = Path(str(img_path).replace("frames", "labels").replace(".jpg", ".txt"))
    if not label_path.exists():
        return None
        
    try:
        with open(label_path, "r") as f:
            lines = f.readlines()
            if not lines:
                return None
            # Assume single class per frame in tracking crop
            parts = lines[0].strip().split()
            if len(parts) < 5:
                return None
                
            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])
    except Exception as e:
        print(f"Error reading {label_path}: {e}")
        return None

    img = cv2.imread(str(img_path))
    if img is None:
        return None
        
    H, W = img.shape[:2]
    
    # Convert normalized YOLO to pixel coords
    x_c = int(x_center * W)
    y_c = int(y_center * H)
    w = int(width * W)
    h = int(height * H)
    
    x1 = max(0, x_c - w // 2)
    y1 = max(0, y_c - h // 2)
    x2 = min(W, x_c + w // 2)
    y2 = min(H, y_c + h // 2)
    
    crop = img[y1:y2, x1:x2]
    if crop.size == 0:
        return None
        
    return crop
