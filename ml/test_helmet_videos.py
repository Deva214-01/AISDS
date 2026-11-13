
import cv2
from ultralytics import YOLO
from pathlib import Path

MODEL = Path(__file__).resolve().parent / "models" / "best.pt"

VIDEO = Path(
    r"C:\Users\Asus\Downloads\Dataset for the Detection of Helmet Use in Motorcycle Drivers\Dataset for the Detection of Helmet Use in Motorcycle Drivers\Without Helmet\p1.mp4"
)

model = YOLO(str(MODEL))

cap = cv2.VideoCapture(str(VIDEO))

if not cap.isOpened():
    raise RuntimeError("Cannot open video")

# Read a frame at 10 seconds.
cap.set(cv2.CAP_PROP_POS_MSEC, 10000)
ret, frame = cap.read()
cap.release()

if not ret:
    raise RuntimeError("Cannot extract frame")

print("Image shape:", frame.shape)

results = model.predict(
    frame,
    conf=0.05,
    imgsz=640,
    verbose=False
)

result = results[0]

print("All detections:", len(result.boxes))

for box in result.boxes:
    class_id = int(box.cls[0])
    confidence = float(box.conf[0])
    print(model.names[class_id], round(confidence, 3))

output = result.plot()

cv2.imwrite(
    str(Path(__file__).resolve().parent / "helmet_diagnostic.jpg"),
    output
)

print("Saved: ml/helmet_diagnostic.jpg")
