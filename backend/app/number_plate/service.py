
from pathlib import Path
from threading import Lock
import re

import cv2
import numpy as np


BASE_DIR = Path(__file__).resolve().parents[3]

MODEL_PATH = (
    BASE_DIR / "ml" / "models" / "license_plate_detector.pt"
)

_model = None
_model_lock = Lock()

_reader = None
_reader_lock = Lock()

OCR_ALLOWLIST = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"


def normalize_indian_plate(text):
    text = re.sub(r"[^A-Z0-9]", "", (text or "").upper())

    # Standard Indian registration format and Bharat series.
    standard = r"^[A-Z]{2}[0-9]{1,2}[A-Z]{1,3}[0-9]{1,4}$"
    bharat = r"^[0-9]{2}BH[0-9]{1,4}[A-Z]{1,2}$"

    if re.fullmatch(standard, text) or re.fullmatch(bharat, text):
        return text

    return ""


def box_iou(a, b):
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    union = area_a + area_b - intersection

    return intersection / union if union else 0.0


def get_number_plate_model():
    global _model

    with _model_lock:
        if _model is None:
            if not MODEL_PATH.is_file():
                raise FileNotFoundError(
                    f"License plate model not found: {MODEL_PATH}"
                )

            # Load Ultralytics only when number plate detection is requested.
            from ultralytics import YOLO

            _model = YOLO(str(MODEL_PATH))

        return _model


def get_ocr_reader():
    global _reader

    with _reader_lock:
        if _reader is None:
            # Load EasyOCR only when OCR is requested.
            import easyocr

            print("Loading EasyOCR model on CPU...")

            _reader = easyocr.Reader(
                ["en"],
                gpu=False,
                verbose=False
            )

            print("EasyOCR model loaded successfully.")

        return _reader


def extract_plate_text(crop):
    if crop is None or crop.size == 0:
        return "", 0.0

    reader = get_ocr_reader()

    # Enlarge the plate for better character recognition.
    height, width = crop.shape[:2]

    if height < 100:
        scale = max(2, int(100 / max(height, 1)))

        crop = cv2.resize(
            crop,
            (width * scale, height * scale),
            interpolation=cv2.INTER_CUBIC
        )

    # Improve contrast.
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    gray = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    ).apply(gray)

    results = reader.readtext(
        gray,
        detail=1,
        paragraph=False,
        allowlist=OCR_ALLOWLIST
    )

    if not results:
        return "", 0.0

    raw_text = "".join(
        item[1] for item in results
    ).upper()

    plate_text = re.sub(
        r"[^A-Z0-9]",
        "",
        raw_text
    )

    if not plate_text:
        return "", 0.0

    confidence = sum(
        float(item[2]) for item in results
    ) / len(results)

    return plate_text, round(confidence, 4)


def detect_license_plates(image_bytes):
    image_array = np.frombuffer(
        image_bytes,
        dtype=np.uint8
    )

    image = cv2.imdecode(
        image_array,
        cv2.IMREAD_COLOR
    )

    if image is None:
        raise ValueError("Invalid or unsupported image file.")

    model = get_number_plate_model()

    results = model.predict(
        source=image,
        conf=0.10,
        imgsz=960,
        max_det=50,
        device="cpu",
        verbose=False
    )

    result = results[0]
    detections = []
    accepted_boxes = []

    if result.boxes is not None:
        for box in result.boxes:

            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0].tolist()
            )

            # Keep coordinates inside image boundaries.
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(image.shape[1], x2)
            y2 = min(image.shape[0], y2)

            # Add a small margin around the plate crop.
            pad_x = int((x2 - x1) * 0.05)
            pad_y = int((y2 - y1) * 0.15)

            crop_x1 = max(0, x1 - pad_x)
            crop_y1 = max(0, y1 - pad_y)
            crop_x2 = min(image.shape[1], x2 + pad_x)
            crop_y2 = min(image.shape[0], y2 + pad_y)

            plate_crop = image[
                crop_y1:crop_y2,
                crop_x1:crop_x2
            ]

            try:
                plate_text, ocr_confidence = extract_plate_text(
                    plate_crop
                )
            except Exception as exc:
                print("OCR error:", exc)
                plate_text = ""
                ocr_confidence = 0.0

            plate_text = normalize_indian_plate(plate_text)

            # Reject OCR output that is not a valid Indian registration format.
            if not plate_text:
                continue

            current_box = (x1, y1, x2, y2)

            # Suppress duplicate detections of the same physical plate.
            if any(
                box_iou(current_box, old_box) >= 0.45
                for old_box in accepted_boxes
            ):
                continue

            accepted_boxes.append(current_box)

            class_id = int(box.cls[0].item())

            detections.append({
                "class_id": class_id,
                "class_name": model.names[class_id],
                "confidence": round(
                    float(box.conf[0].item()), 4
                ),
                "plate_text": plate_text,
                "ocr_confidence": ocr_confidence,
                "ocr_status": (
                    "READ"
                    if plate_text
                    else "NOT_READ"
                ),
                "coordinates": {
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2
                }
            })

    annotated_image = result.plot()

    success, encoded_image = cv2.imencode(
        ".jpg",
        annotated_image
    )

    if not success:
        raise RuntimeError(
            "Could not encode annotated image."
        )

    return {
        "total_plates": len(detections),
        "detections": detections,
        "annotated_image": encoded_image.tobytes()
    }
