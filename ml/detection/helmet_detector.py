
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


# =========================================================
# PROJECT PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_MODEL_PATH = (
    PROJECT_ROOT / "ml" / "models" / "HelmetDetection.onnx"
)


# =========================================================
# HELMET DETECTION ENGINE
# =========================================================

class HelmetDetector:

    CLASSES = {
        0: "Rider",
        1: "Helmet",
        2: "Plate",
        3: "NoHelmet",
    }

    def __init__(
        self,
        model_path=None,
        confidence=0.50,
        iou_threshold=0.45,
        input_size=768,
    ):

        # Resolve model path
        model_path = (
            Path(model_path)
            if model_path
            else DEFAULT_MODEL_PATH
        )

        if not model_path.is_absolute():
            model_path = PROJECT_ROOT / model_path

        if not model_path.is_file():
            raise FileNotFoundError(
                f"Helmet ONNX model not found: {model_path}"
            )

        # Detection settings
        self.confidence = confidence
        self.iou_threshold = iou_threshold
        self.input_size = input_size

        # Load ONNX model
        self.session = ort.InferenceSession(
            str(model_path),
            providers=["CPUExecutionProvider"],
        )

        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        input_shape = self.session.get_inputs()[0].shape

        if len(input_shape) != 4:
            raise ValueError(
                f"Unexpected model input shape: {input_shape}"
            )

        if input_shape[1] != 3:
            raise ValueError(
                f"Expected 3 input channels, received {input_shape}"
            )

        if len(self.CLASSES) != 4:
            raise ValueError("Model class configuration is invalid.")

        print("Helmet ONNX model loaded successfully")
        print("Model:", model_path)
        print("Input:", self.input_name, input_shape)
        print("Output:", self.output_name)

    # =====================================================
    # IMAGE PREPROCESSING
    # =====================================================

    def preprocess(self, frame):

        height, width = frame.shape[:2]

        scale = min(
            self.input_size / width,
            self.input_size / height,
        )

        new_width = int(round(width * scale))
        new_height = int(round(height * scale))

        resized = cv2.resize(
            frame,
            (new_width, new_height),
            interpolation=cv2.INTER_LINEAR,
        )

        canvas = np.full(
            (self.input_size, self.input_size, 3),
            114,
            dtype=np.uint8,
        )

        pad_x = (self.input_size - new_width) // 2
        pad_y = (self.input_size - new_height) // 2

        canvas[
            pad_y:pad_y + new_height,
            pad_x:pad_x + new_width
        ] = resized

        # BGR to RGB
        image = cv2.cvtColor(
            canvas,
            cv2.COLOR_BGR2RGB,
        )

        # Normalize
        image = image.astype(np.float32) / 255.0

        # HWC to CHW
        image = np.transpose(
            image,
            (2, 0, 1),
        )

        # Add batch dimension
        image = np.expand_dims(
            image,
            axis=0,
        )

        return image, scale, pad_x, pad_y

    # =====================================================
    # DETECTION
    # =====================================================

    def detect(self, frame):

        if frame is None or not isinstance(frame, np.ndarray):
            return []

        if frame.ndim != 3 or frame.shape[2] != 3:
            return []

        height, width = frame.shape[:2]

        image, scale, pad_x, pad_y = self.preprocess(frame)

        # ONNX inference
        output = self.session.run(
            [self.output_name],
            {
                self.input_name: image,
            },
        )[0]

        if output.ndim != 3 or output.shape[1] != 8:
            raise ValueError(
                f"Unexpected model output shape: {output.shape}"
            )

        predictions = output[0].T

        boxes = []
        scores = []
        class_ids = []

        # =================================================
        # DECODE PREDICTIONS
        # =================================================

        for prediction in predictions:

            class_scores = prediction[4:8]

            class_id = int(
                np.argmax(class_scores)
            )

            confidence = float(
                class_scores[class_id]
            )

            if confidence < self.confidence:
                continue

            cx, cy, box_width, box_height = prediction[:4]

            # Convert coordinates to original frame
            x1 = (
                cx - box_width / 2 - pad_x
            ) / scale

            y1 = (
                cy - box_height / 2 - pad_y
            ) / scale

            x2 = (
                cx + box_width / 2 - pad_x
            ) / scale

            y2 = (
                cy + box_height / 2 - pad_y
            ) / scale

            x1 = max(0, min(int(x1), width - 1))
            y1 = max(0, min(int(y1), height - 1))
            x2 = max(0, min(int(x2), width - 1))
            y2 = max(0, min(int(y2), height - 1))

            if x2 <= x1 or y2 <= y1:
                continue

            boxes.append([
                x1,
                y1,
                x2 - x1,
                y2 - y1,
            ])

            scores.append(confidence)
            class_ids.append(class_id)

        # =================================================
        # NON-MAXIMUM SUPPRESSION
        # =================================================

        detections = []

        if not boxes:
            return detections

        indices = cv2.dnn.NMSBoxes(
            boxes,
            scores,
            self.confidence,
            self.iou_threshold,
        )

        if len(indices) == 0:
            return detections

        for index in np.array(indices).flatten():

            x, y, box_width, box_height = boxes[index]

            class_id = class_ids[index]

            detections.append({

                "class_id": class_id,

                "label": self.CLASSES[class_id],

                "confidence": round(
                    float(scores[index]),
                    4,
                ),

                "box": [
                    x,
                    y,
                    x + box_width,
                    y + box_height,
                ],

            })

        return detections

    # =====================================================
    # DRAW DETECTIONS
    # =====================================================

    def draw(self, frame, detections):

        output_frame = frame.copy()

        for detection in detections:

            x1, y1, x2, y2 = detection["box"]

            label = detection["label"]

            confidence = detection["confidence"]

            if label == "NoHelmet":
                color = (0, 0, 255)

            elif label == "Helmet":
                color = (0, 255, 0)

            elif label == "Rider":
                color = (255, 180, 0)

            else:
                color = (255, 255, 0)

            cv2.rectangle(
                output_frame,
                (x1, y1),
                (x2, y2),
                color,
                2,
            )

            text = f"{label} {confidence:.2f}"

            cv2.putText(
                output_frame,
                text,
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
            )

        return output_frame
