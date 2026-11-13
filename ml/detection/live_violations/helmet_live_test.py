from pathlib import Path
import cv2
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[3]
MODEL = ROOT / "ml" / "models" / "best.pt"

def main():
    if not MODEL.exists():
        raise FileNotFoundError(f"Model not found: {MODEL}")

    print("Loading helmet model:", MODEL)
    model = YOLO(str(MODEL))
    print("Classes:", model.names)

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam. Close other camera applications.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Cannot read webcam frame.")
                break

            results = model.predict(
                frame,
                classes=None,
                imgsz=640,
                conf=0.10,
                device="cpu",
                verbose=False
            )

            annotated = results[0].plot()
            helmet = 0
            no_helmet = 0

            if results[0].boxes is not None:
                for cls in results[0].boxes.cls.tolist():
                    if int(cls) == 2:
                        helmet += 1
                    elif int(cls) == 6:
                        no_helmet += 1

            cv2.putText(annotated, f"Helmet: {helmet}", (15, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(annotated, f"No Helmet: {no_helmet}", (15, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            cv2.imshow("TRAFFICX-AI Helmet Test", annotated)

            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Webcam released.")

if __name__ == "__main__":
    main()
