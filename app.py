from flask import Flask, request, jsonify
from ultralytics import YOLO
import base64
import numpy as np
import cv2
import os

app = Flask(__name__)

print("Loading YOLO model...")
model = YOLO("best.pt")
print("Model loaded successfully!")


@app.route("/")
def home():
    return "Porthole YOLO Backend is running!"


@app.route("/detect", methods=["POST"])
def detect():
    data = request.json

    if not data or "image" not in data:
        return jsonify({"error": "No image received"}), 400

    image_data = data["image"]

    if "," in image_data:
        image_data = image_data.split(",", 1)[1]

    image_bytes = base64.b64decode(image_data)
    image_array = np.frombuffer(image_bytes, dtype=np.uint8)
    frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    if frame is None:
        return jsonify({"error": "Invalid image"}), 400

    results = model(frame, conf=0.30, verbose=False)

    detections = []

    for result in results:
        for box in result.boxes:
            detections.append({
                "class": int(box.cls[0]),
                "confidence": float(box.conf[0]),
                "x1": float(box.xyxy[0][0]),
                "y1": float(box.xyxy[0][1]),
                "x2": float(box.xyxy[0][2]),
                "y2": float(box.xyxy[0][3])
            })

    return jsonify({"detections": detections})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
