# Realtime Image Classification (MobileNetV2 + OpenCV)

A simple real-time image classification demo that uses a pre-trained **MobileNetV2** model (ImageNet weights) to classify live webcam frames or video files. Predictions (class name + confidence) are displayed as overlays on the video feed.

---

## Installation

1. Make sure Python 3.8–3.11 is installed.
2. Install dependencies:
```bash
pip install -r requirements.txt
```

---

## Usage

Run with default webcam:
```bash
python main.py
```

Use a different camera index (e.g., 1):
```bash
python main.py --source 1
```

Use a video file:
```bash
python main.py --source path/to/video.mp4
```

Press `q` (or `Esc`) to exit the window.

---

## Notes & Tips

- The model downloads the ImageNet weights the first time you run; it may take a minute depending on connection.
- On CPU-only machines, inference is slower. Use smaller window or consider using a cloud environment (Google Colab) with GPU for faster experiments.
- If the camera doesn't open:
  - Check camera index (0, 1, ...).
  - Ensure no other app is using the webcam.
  - On Windows, allow camera access in Settings.

---

## Files

- `main.py` — realtime classification script
- `requirements.txt` — Python dependencies
- `README.md` — this file

---

