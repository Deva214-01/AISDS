
import cv2
from ultralytics import YOLO

MODEL_PATH = "yolo11n.pt"
VIDEO_PATH = "videos/traffic.mp4"

FRAME_WIDTH = 640
FRAME_HEIGHT = 360
LINE_Y = 180

model = YOLO(MODEL_PATH)
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    raise RuntimeError(f"Cannot open video: {VIDEO_PATH}")

print("Traffic video loaded successfully")
print("Line-crossing test started")

# Track each vehicle's previous vertical position.
previous_positions = {}

# Record each vehicle that crosses the line.
crossed_ids = set()

frame_count = 0

try:
    while True:
        ret, frame = cap.read()

        if not ret:
            break

        frame_count += 1
        frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

        results = model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            classes=[2, 3, 5, 7],
            conf=0.3,
            imgsz=640,
            verbose=False,
        )

        result = results[0]

        # Draw the test line.
        cv2.line(
            frame,
            (0, LINE_Y),
            (FRAME_WIDTH, LINE_Y),
            (0, 0, 255),
            2,
        )

        boxes = result.boxes

        if boxes is not None and boxes.id is not None:
            coordinates = boxes.xyxy.cpu().numpy()
            track_ids = boxes.id.int().cpu().tolist()

            for box, track_id in zip(coordinates, track_ids):
                x1, y1, x2, y2 = map(int, box)

                # Use the center of the bounding box.
                center_y = (y1 + y2) // 2

                previous_y = previous_positions.get(track_id)

                crossed = False

                if previous_y is not None:
                    crossed = (
                        previous_y < LINE_Y <= center_y
                        or previous_y > LINE_Y >= center_y
                    )

                if crossed:
                    crossed_ids.add(track_id)

                previous_positions[track_id] = center_y

                color = (0, 255, 0)
                label = f"ID {track_id}"

                if track_id in crossed_ids:
                    color = (0, 0, 255)
                    label = f"ID {track_id} LINE CROSSED"

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    color,
                    2,
                )

                cv2.putText(
                    frame,
                    label,
                    (x1, max(y1 - 8, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    color,
                    2,
                )

        cv2.putText(
            frame,
            f"Unique vehicles crossing line: {len(crossed_ids)}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
        )

        cv2.imshow("TRAFFICX-AI - Line Crossing Test", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()

print("\n========== LINE CROSSING REPORT ==========")
print("Frames processed:", frame_count)
print("Unique track IDs crossing line:", len(crossed_ids))
print("Crossed vehicle IDs:", sorted(crossed_ids))
print("==========================================")
