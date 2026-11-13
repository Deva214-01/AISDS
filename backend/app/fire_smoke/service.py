from pathlib import Path
from threading import Lock
import base64

import cv2
import numpy as np
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODEL_PATH = PROJECT_ROOT / "ml" / "models" / "fire_smoke_best.pt"

_model = None
_model_lock = Lock()

def get_model():
    global _model
    with _model_lock:
        if _model is None:
            if not MODEL_PATH.is_file():
                raise FileNotFoundError(f"Fire/Smoke model not found: {MODEL_PATH}")
            _model = YOLO(str(MODEL_PATH))
    return _model

def detect_fire_smoke(image_bytes: bytes):
    image = cv2.imdecode(
        np.frombuffer(image_bytes, dtype=np.uint8),
        cv2.IMREAD_COLOR
    )
    if image is None:
        raise ValueError("Could not read image. Upload a valid JPG, PNG or WEBP.")

    model = get_model()
    result = model.predict(
        source=image,
        conf=0.25,
        imgsz=640,
        device="cpu",
        max_det=100,
        verbose=False
    )[0]

    detections = []
    counts = {"fire": 0, "smoke": 0}

    if result.boxes is not None:
        for box in result.boxes:
            class_id = int(box.cls[0].item())
            label = str(model.names[class_id]).lower()
            confidence = float(box.conf[0].item())
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

            if label not in counts:
                continue

            detections.append({
                "class_name": label,
                "confidence": round(confidence, 4),
                "box": [x1, y1, x2, y2]
            })
            counts[label] += 1

    annotated = result.plot()
    success, encoded = cv2.imencode(".jpg", annotated)
    if not success:
        raise RuntimeError("Could not encode annotated image.")

    return {
        "total_detections": len(detections),
        "fire_count": counts["fire"],
        "smoke_count": counts["smoke"],
        "class_counts": counts,
        "detections": detections,
        "annotated_image": base64.b64encode(encoded.tobytes()).decode("utf-8"),
        "image_width": int(image.shape[1]),
        "image_height": int(image.shape[0]),
        "model": MODEL_PATH.name
    }
