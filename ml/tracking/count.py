
import cv2
from ultralytics import YOLO

# Configuration
MODEL_PATH = "yolo11n.pt"
VIDEO_PATH = "videos/traffic.mp4"

model = YOLO(MODEL_PATH)
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    raise RuntimeError(f"Cannot open video: {VIDEO_PATH}")

print("Traffic video loaded successfully")
print("Vehicle counting started...")

# Unique vehicle IDs
counted_ids = set()

# Vehicle type IDs
car_ids = set()
motorcycle_ids = set()
bus_ids = set()
truck_ids = set()

# Frame counters
frame_index = 0
processed_frames = 0

try:
    while True:
        ret, frame = cap.read()

        if not ret:
            break

        frame_index += 1

        # Process every second frame to match tracking_test.py
        if frame_index % 2 != 0:
            continue

        # Resize for CPU performance
        frame = cv2.resize(frame, (640, 360))

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
        processed_frames += 1
        boxes = result.boxes

        if boxes is not None and boxes.id is not None:
            track_ids = boxes.id.int().cpu().tolist()
            classes = boxes.cls.int().cpu().tolist()

            for track_id, class_id in zip(track_ids, classes):
                counted_ids.add(track_id)

                if class_id == 2:
                    car_ids.add(track_id)
                elif class_id == 3:
                    motorcycle_ids.add(track_id)
                elif class_id == 5:
                    bus_ids.add(track_id)
                elif class_id == 7:
                    truck_ids.add(track_id)

        # Draw detections and tracking IDs
        annotated_frame = result.plot()

        # Calculate total from class-wise counts
        total = (
            len(car_ids)
            + len(motorcycle_ids)
            + len(bus_ids)
            + len(truck_ids)
        )

        # Display statistics
        cv2.putText(
            annotated_frame,
            f"Total Vehicles: {total}",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

        cv2.putText(
            annotated_frame,
            f"Cars: {len(car_ids)}",
            (20, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

        cv2.putText(
            annotated_frame,
            f"Motorcycles: {len(motorcycle_ids)}",
            (20, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

        cv2.putText(
            annotated_frame,
            f"Buses: {len(bus_ids)}",
            (20, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

        cv2.putText(
            annotated_frame,
            f"Trucks: {len(truck_ids)}",
            (20, 150),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

        cv2.imshow(
            "TRAFFICX AI - Vehicle Counting",
            annotated_frame,
        )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()

# Final results
print("\n========== FINAL TRAFFIC COUNT ==========")
print("Video frames read:", frame_index)
print("Frames processed:", processed_frames)
print("Cars:", len(car_ids))
print("Motorcycles:", len(motorcycle_ids))
print("Buses:", len(bus_ids))
print("Trucks:", len(truck_ids))
print("Total unique track IDs:", len(counted_ids))
print(
    "Sum of class counts:",
    len(car_ids) + len(motorcycle_ids)
    + len(bus_ids) + len(truck_ids),
)
print("=========================================")
