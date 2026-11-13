
import cv2
from ultralytics import YOLO
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODEL_PATH = PROJECT_ROOT / "ml" / "models" / "HelmetDetection.onnx"

CONFIDENCE = 0.15
IMAGE_SIZE = 768

def main():
    print("Loading helmet detection model...")

    model = YOLO(str(MODEL_PATH), task="detect")

    print("Model loaded successfully!")
    print("Opening webcam...")

    camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not camera.isOpened():
        print("ERROR: Could not open webcam.")
        return

    print("Webcam started. Press ESC to exit.")

    try:
        while True:
            success, frame = camera.read()

            if not success:
                print("ERROR: Could not read webcam frame.")
                break

            results = model.predict(
                frame,
                imgsz=IMAGE_SIZE,
                conf=CONFIDENCE,
                device="cpu",
                verbose=False
            )

            result = results[0]
            annotated_frame = result.plot()

            detected_classes = [
                model.names[int(box.cls[0])]
                for box in result.boxes
            ]

            if "NoHelmet" in detected_classes:
                status = "POSSIBLE NO-HELMET: REVIEW REQUIRED"
                color = (0, 0, 255)
            elif "Helmet" in detected_classes:
                status = "HELMET DETECTED"
                color = (0, 255, 0)
            else:
                status = "NO HELMET RESULT"
                color = (255, 255, 0)

            cv2.putText(
                annotated_frame,
                status,
                (10, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                color,
                2
            )

            cv2.imshow("TRAFFICX-AI | Helmet Detection Test", annotated_frame)

            if cv2.waitKey(1) & 0xFF == 27:
                break

    finally:
        camera.release()
        cv2.destroyAllWindows()
        print("Webcam stopped. Test finished.")


if __name__ == "__main__":
    main()
