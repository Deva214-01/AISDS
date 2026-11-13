
import os
import cv2
from ultralytics import YOLO

VIDEO_PATH = "videos/traffic.mp4"
MODEL_PATH = "yolo11n.pt"
OUTPUT_PATH = "output/tracking_test.mp4"

# Load model and video
model = YOLO(MODEL_PATH)
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    raise RuntimeError(f"Cannot open video: {VIDEO_PATH}")

# Tracking statistics
seen_ids = set()
frame_index = 0
processed_frames = 0
frames_with_ids = 0
detections_without_ids = 0
total_detections = 0

class_track_ids = {
    2: set(),  # Cars
    3: set(),  # Motorcycles
    5: set(),  # Buses
    7: set(),  # Trucks
}

os.makedirs("output", exist_ok=True)
video_writer = None

try:
    while True:
        ok, frame = cap.read()

        if not ok:
            break

        frame_index += 1

        # Process every second frame
        if frame_index % 2 != 0:
            continue

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

        processed_frames += 1
        boxes = results[0].boxes

        # Draw bounding boxes and tracking IDs
        annotated_frame = results[0].plot()

        # Initialize video writer once
        if video_writer is None:
            height, width = annotated_frame.shape[:2]

            video_writer = cv2.VideoWriter(
                OUTPUT_PATH,
                cv2.VideoWriter_fourcc(*"mp4v"),
                30,
                (width, height),
            )

            if not video_writer.isOpened():
                raise RuntimeError(
                    f"Cannot create output video: {OUTPUT_PATH}"
                )

        video_writer.write(annotated_frame)

        # Update tracking statistics
        if boxes is not None:
            total_detections += len(boxes)

            if boxes.id is not None:
                ids = boxes.id.int().cpu().tolist()
                classes = boxes.cls.int().cpu().tolist()

                for track_id, cls in zip(ids, classes):
                    if cls in class_track_ids:
                        class_track_ids[cls].add(track_id)

                seen_ids.update(ids)
                frames_with_ids += 1
                detections_without_ids += len(boxes) - len(ids)
            else:
                detections_without_ids += len(boxes)

finally:
    cap.release()

    if video_writer is not None:
        video_writer.release()

# Print baseline statistics
print("\n===== TRACKING BASELINE =====")
print("Video frames read:", frame_index)
print("Frames processed:", processed_frames)
print("Unique track IDs observed:", len(seen_ids))
print("Total detections:", total_detections)
print("Detections without IDs:", detections_without_ids)
print("Frames with ID detections:", frames_with_ids)
print("=============================")

print("\n===== UNIQUE VEHICLES BY CLASS =====")
print("Cars:", len(class_track_ids[2]))
print("Motorcycles:", len(class_track_ids[3]))
print("Buses:", len(class_track_ids[5]))
print("Trucks:", len(class_track_ids[7]))
print("====================================")

print("\nAnnotated video saved to:", OUTPUT_PATH)
