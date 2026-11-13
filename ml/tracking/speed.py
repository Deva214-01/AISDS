import cv2
import math
import time
from ultralytics import YOLO

# Load YOLO model
model = YOLO("yolo11n.pt")

# Traffic video
video_path = "../videos/traffic.mp4"

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("❌ Video open nahi ho rahi")
    exit()

print("✅ Traffic video loaded")
print("🚗 Speed estimation started...")

# Store previous vehicle positions
previous_positions = {}

# Store previous timestamps
previous_times = {}

# Conversion factor
# Prototype value - needs calibration for real km/h
PIXEL_TO_METER = 0.05

while True:

    ret, frame = cap.read()

    if not ret:
        break

    # Resize for CPU
    frame = cv2.resize(frame, (640, 360))

    # Current time
    current_time = time.time()

    # YOLO tracking
    results = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        classes=[2, 3, 5, 7],
        conf=0.3,
        imgsz=640,
        verbose=False
    )

    result = results[0]

    if result.boxes.id is not None:

        boxes = result.boxes.xyxy.cpu().numpy()
        track_ids = result.boxes.id.int().cpu().tolist()

        for box, track_id in zip(boxes, track_ids):

            x1, y1, x2, y2 = map(int, box)

            # Vehicle center
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)

            current_position = (center_x, center_y)

            speed_kmh = 0

            if track_id in previous_positions:

                previous_position = previous_positions[track_id]

                distance_pixels = math.sqrt(
                    (center_x - previous_position[0]) ** 2
                    +
                    (center_y - previous_position[1]) ** 2
                )

                time_difference = (
                    current_time - previous_times[track_id]
                )

                if time_difference > 0:

                    distance_meters = (
                        distance_pixels * PIXEL_TO_METER
                    )

                    speed_mps = (
                        distance_meters / time_difference
                    )

                    speed_kmh = speed_mps * 3.6

            # Save position and time
            previous_positions[track_id] = current_position
            previous_times[track_id] = current_time

            # Draw center
            cv2.circle(
                frame,
                current_position,
                4,
                (0, 255, 0),
                -1
            )

            # Draw speed
            cv2.putText(
                frame,
                f"ID {track_id} | {speed_kmh:.1f} km/h",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2
            )

    cv2.imshow(
        "TRAFFICX AI - Speed Estimation",
        frame
    )

    # Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

print("✅ Speed estimation complete")