
import cv2
from ultralytics import YOLO

MODEL_PATH = "yolo11n.pt"
VIDEO_PATH = "videos/stationary_test.mp4"

FRAME_WIDTH = 960
FRAME_HEIGHT = 540
CONFIDENCE = 0.10

# COCO classes:
# 2 = car, 3 = motorcycle, 5 = bus, 7 = truck
VEHICLE_CLASSES = [2, 3, 5, 7]

model = YOLO(MODEL_PATH)
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    raise RuntimeError(f"Cannot open video: {VIDEO_PATH}")

frame_index = 0
frames_with_detections = 0
total_vehicle_detections = 0

print("YOLO detection-only test started")
print("Tracking: DISABLED")
print(f"Confidence threshold: {CONFIDENCE}")

try:
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame_index += 1
        frame = cv2.resize(
            frame, (FRAME_WIDTH, FRAME_HEIGHT)
        )

        result = model.predict(
            frame,
            classes=VEHICLE_CLASSES,
            conf=CONFIDENCE,
            imgsz=960,
            verbose=False,
        )[0]

        count = 0

        if result.boxes is not None:
            count = len(result.boxes)

            if count > 0:
                frames_with_detections += 1
                total_vehicle_detections += count

                for box in result.boxes:
                    x1, y1, x2, y2 = map(
                        int, box.xyxy[0].cpu().tolist()
                    )
                    confidence = float(box.conf[0])
                    class_id = int(box.cls[0])

                    class_name = model.names[class_id]
                    label = f"{class_name} {confidence:.2f}"

                    cv2.rectangle(
                        frame,
                        (x1, y1),
                        (x2, y2),
                        (0, 255, 0),
                        2,
                    )

                    cv2.putText(
                        frame,
                        label,
                        (x1, max(y1 - 8, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        (0, 255, 0),
                        2,
                    )

        if frame_index % 10 == 0:
            print(
                f"Frame={frame_index}, "
                f"vehicle_detections={count}"
            )

        cv2.putText(
            frame,
            f"Vehicle detections: {count}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
        )

        cv2.imshow("TRAFFICX-AI - Detection Debug", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()

print("\n===== DETECTION REPORT =====")
print("Frames processed:", frame_index)
print("Frames with detections:", frames_with_detections)
print("Total vehicle detections:", total_vehicle_detections)
print("============================")
