"""
Realtime Image Classification (MobileNetV2 + OpenCV)

Usage:
    python main.py            # uses default camera 0
    python main.py --source 1 # use camera index 1
    python main.py --source video.mp4  # use a video file

Press `q` to quit.
"""
import argparse
import time
import numpy as np
import cv2
import sys

import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions

def build_model():
    model = MobileNetV2(weights="imagenet")
    return model

def preprocess_frame_for_model(frame, target_size=(224, 224)):
    # frame is BGR (from OpenCV). Convert to RGB, resize, preprocess
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, target_size)
    x = np.expand_dims(img, axis=0).astype("float32")
    x = preprocess_input(x)
    return x

def draw_label(frame, text, origin=(10, 30), font_scale=0.7, thickness=2, bg_color=(0,0,0)):
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = origin
    # background rectangle
    cv2.rectangle(frame, (x-5, y - text_size[1] - 8), (x + text_size[0] + 5, y + 5), bg_color, -1)
    cv2.putText(frame, text, (x, y - 6), font, font_scale, (255,255,255), thickness, cv2.LINE_AA)

def main(args):
    # Build model
    print("[INFO] Loading MobileNetV2 (ImageNet weights) - this may take a moment...")
    model = build_model()
    print("[INFO] Model loaded.")

    # Open video source
    src = args.source
    try:
        # If numeric string, cast to int for camera index
        cam_index = int(src) if str(src).isdigit() else src
    except Exception:
        cam_index = src

    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        print(f"[ERROR] Could not open source: {src}")
        return

    fps_display_interval = 1  # seconds
    frame_count = 0
    start_time = time.time()
    fps = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[INFO] End of stream or cannot read frame.")
            break

        # Optionally resize for faster inference on CPU (maintain small cost)
        # Keep original for display clarity; model expects 224x224 anyway
        input_tensor = preprocess_frame_for_model(frame, target_size=(224,224))

        # Predict
        preds = model.predict(input_tensor, verbose=0)
        decoded = decode_predictions(preds, top=1)[0][0]  # (class_id, class_name, score)
        class_name = decoded[1]
        score = decoded[2]

        # Overlay result on frame
        label = f"{class_name}: {score*100:.1f}%"
        draw_label(frame, label, origin=(10,40), font_scale=0.8, thickness=2, bg_color=(0,0,0))

        # Calculate & display FPS
        frame_count += 1
        if (time.time() - start_time) > fps_display_interval:
            fps = frame_count / (time.time() - start_time)
            frame_count = 0
            start_time = time.time()
        draw_label(frame, f"FPS: {fps:.1f}", origin=(10,80), font_scale=0.7, thickness=2, bg_color=(0,0,0))

        cv2.imshow("Realtime CNN Classification (MobileNetV2)", frame)

        # Quit on 'q' or ESC
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Exiting...")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Realtime Image Classification with MobileNetV2")
    parser.add_argument("--source", type=str, default="0",
                        help="Video source. '0' or '1' for webcams, or path to video file")
    args = parser.parse_args()
    main(args)
