# AI-Powered Automated Attendance & Video Surveillance System

This project is a high-performance, real-time computer vision pipeline for automated attendance tracking and intelligent video surveillance. 

It uses advanced Deep Learning models (**YOLOv8**, **InsightFace**) combined with high-speed spatial tracking (**BoTSORT**, Kalman Filters) and vector search (**FAISS**) to instantly recognize individuals in live video feeds. It safely logs attendance into a **PostgreSQL** database using a zero-lag asynchronous worker.

## Key Features
- **Real-Time Multi-Object Tracking:** Uses YOLOv8 for robust body detection and BoTSORT for smooth, high-fps spatial tracking, handling camera noise and occlusions gracefully.
- **Biometric Identity Verification:** Uses InsightFace (ArcFace) to extract 512-dimensional facial embeddings for high-accuracy identity resolution regardless of angle or lighting.
- **FAISS Vector Search Engine:** Implements Facebook AI Similarity Search (FAISS) for lightning-fast matching of live faces against a gallery of known identities.
- **Zero-Lag Asynchronous Database:** Attendance logs (in-time, out-time, total hours, present/absent status) are written to PostgreSQL via a dedicated background threading queue, completely decoupling database I/O from the live video feed.
- **Intelligent AI Throttling:** Dynamically throttles heavy neural network inferences once a person is identified, relying instead on lightweight Kalman filters. This drastically reduces CPU overhead and increases FPS.

## Technologies Used
* **Languages:** Python, SQL
* **Deep Learning:** PyTorch, YOLOv8 (Ultralytics), InsightFace, OSNet
* **Computer Vision:** OpenCV, BoTSORT
* **Data & Search:** PostgreSQL, FAISS, psycopg2, Pandas

## Project Structure
- `reid/offline_tracking_faces.py` - The core engine that orchestrates YOLO, BoTSORT, InsightFace, and PostgreSQL asynchronous logging.
- `reid/face_extractor.py` - Wraps the InsightFace `buffalo_l` model for facial detection and embedding extraction.
- `reid/auto_enroll.py` - Utility to automatically enroll new faces into the FAISS index database.
- `export_attendance.py` - Utility script to export Postgres attendance logs to formatted Excel files.

## Setup Instructions

1. **Clone the Repository**
2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Database Configuration:**
   - Install PostgreSQL.
   - Update `PG_PASS` and `PG_DBNAME` inside `reid/offline_tracking_faces.py` to match your local database credentials.
4. **Run the Tracker:**
   ```bash
   python reid/offline_tracking_faces.py --input "0"
   ```
   *(Replace `"0"` with a path to an `mp4` video file to process a pre-recorded video instead of a webcam).*
