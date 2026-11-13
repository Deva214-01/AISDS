
import cv2
from ultralytics import YOLO

MODEL_PATH = "yolo11n.pt"
VIDEO_PATH = "videos/stationary_test.mp4"

FRAME_WIDTH = 960
FRAME_HEIGHT = 540
CONFIDENCE = 0.2

model = YOLO(MODEL_PATH)
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    raise RuntimeError(f"Cannot open video: {VIDEO_PATH}")

frame_index = 0
frames_with_detections = 0
frames_without_ids = 0
observations = {}
previous_ids = set()

print("Tracking continuity test started")

try:
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame_index += 1
        frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

        result = model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            classes=[2, 3, 5, 7],
            conf=CONFIDENCE,
            imgsz=960,
            verbose=False,
        )[0]

        current_ids = set()

        if result.boxes is not None and len(result.boxes) > 0:
            frames_with_detections += 1

            if result.boxes.id is None:
                frames_without_ids += 1
            else:
                boxes = result.boxes.xyxy.cpu().numpy()
                ids = result.boxes.id.int().cpu().tolist()

                for box, track_id in zip(boxes, ids):
                    x1, y1, x2, y2 = map(int, box)
                    current_ids.add(track_id)
                    observations[track_id] = (
                        observations.get(track_id, 0) + 1
                    )

                    cv2.rectangle(
                        frame, (x1, y1), (x2, y2),
                        (0, 255, 0), 2
                    )
                    cv2.putText(
                        frame, f"ID {track_id}",
                        (x1, max(y1 - 8, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 0), 2
                    )

        if frame_index % 10 == 0:
            print(
                f"Frame={frame_index}, "
                f"detections={len(result.boxes) if result.boxes is not None else 0}, "
                f"visible_IDs={sorted(current_ids)}, "
                f"total_IDs_seen={len(observations)}"
            )

        previous_ids = current_ids

        cv2.imshow("Tracking Continuity Test", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()

print("\n===== TRACKING REPORT =====")
print("Frames processed:", frame_index)
print("Frames with detections:", frames_with_detections)
print("Frames with detections but no IDs:", frames_without_ids)
print("Unique IDs:", len(observations))
print("Observations per ID:", dict(sorted(observations.items())))
print("===========================")
