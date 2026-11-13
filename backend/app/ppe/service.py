
from pathlib import Path
from threading import Lock
import base64

import cv2
import numpy as np
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[3]

MODEL_PATH = (
    PROJECT_ROOT / "ml" / "models" /
    "ppe_best_v3_candidate.pt"
)

_model = None
_model_lock = Lock()

PPE_CLASSES = {
    "helmet", "glove", "goggles", "mask", "shoes"
}

MISSING_CLASSES = {
    "no_helmet", "no_glove", "no_goggles",
    "no_mask", "no_shoes"
}


def get_model():
    global _model

    with _model_lock:
        if _model is None:
            if not MODEL_PATH.is_file():
                raise FileNotFoundError(
                    f"PPE model not found: {MODEL_PATH}"
                )

            _model = YOLO(str(MODEL_PATH))

    return _model


def detect_ppe(image_bytes: bytes):

    image_array = np.frombuffer(
        image_bytes, dtype=np.uint8
    )

    image = cv2.imdecode(
        image_array, cv2.IMREAD_COLOR
    )

    if image is None:
        raise ValueError(
            "Could not read image. Upload a valid JPG, PNG or WEBP."
        )

    model = get_model()

    result = model.predict(
        source=image,
        conf=0.20,
        imgsz=640,
        iou=0.45,
        device="cpu",
        max_det=100,
        verbose=False
    )[0]

    detections = []
    counts = {}
    missing_counts = {}

    if result.boxes is not None:

        for box in result.boxes:

            class_id = int(box.cls[0].item())
            label = str(
                model.names[class_id]
            ).lower()

            confidence = float(
                box.conf[0].item()
            )

            x1, y1, x2, y2 = map(
                int, box.xyxy[0].tolist()
            )

            category = (
                "missing"
                if label in MISSING_CLASSES
                else "detected"
            )

            detections.append({
                "class_name": label,
                "confidence": round(confidence, 4),
                "box": [x1, y1, x2, y2],
                "category": category
            })

            counts[label] = counts.get(label, 0) + 1

            if label in MISSING_CLASSES:
                missing_counts[label] = (
                    missing_counts.get(label, 0) + 1
                )

    annotated = result.plot()

    success, encoded = cv2.imencode(
        ".jpg", annotated
    )

    if not success:
        raise RuntimeError(
            "Could not encode annotated image."
        )

    return {
        "total_detections": len(detections),
        "detections": detections,
        "class_counts": counts,
        "missing_equipment_counts": missing_counts,
        "annotated_image": base64.b64encode(
            encoded.tobytes()
        ).decode("utf-8"),
        "image_width": int(image.shape[1]),
        "image_height": int(image.shape[0]),
        "model": MODEL_PATH.name
    }
