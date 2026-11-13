
import os
import cv2
from ultralytics import YOLO
from itertools import combinations

# Independent collision test — does not modify the main traffic engine
MODEL_PATH = "yolo11n.pt"
VIDEO_PATH = "videos/headon-nn.mp4"
OUTPUT_PATH = "output/collision_review.mp4"

CONFIDENCE = 0.30
IMAGE_SIZE = 640
VEHICLE_CLASSES = [2, 3, 5, 7]  # car, motorcycle, bus, truck
OVERLAP_THRESHOLD = 0.05

os.makedirs("output", exist_ok=True)

if not os.path.isfile(VIDEO_PATH):
    raise FileNotFoundError(f"Video not found: {VIDEO_PATH}")

model = YOLO(MODEL_PATH)
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    raise RuntimeError(f"Cannot open video: {VIDEO_PATH}")

fps = cap.get(cv2.CAP_PROP_FPS)
if fps <= 0:
    fps = 30.0

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

writer = cv2.VideoWriter(
    OUTPUT_PATH,
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (width, height),
)

if not writer.isOpened():
    cap.release()
    raise RuntimeError(f"Cannot create output video: {OUTPUT_PATH}")

frame_number = 0
frames_with_vehicles = 0
possible_overlap_frames = 0

try:
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame_number += 1

        results = model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            classes=VEHICLE_CLASSES,
            conf=CONFIDENCE,
            imgsz=IMAGE_SIZE,
            verbose=False,
        )

        result = results[0]
        boxes = result.boxes

        if boxes is not None and len(boxes) > 0:
            frames_with_vehicles += 1

            xyxy = boxes.xyxy.cpu().numpy()
            ids = (
                boxes.id.int().cpu().tolist()
                if boxes.id is not None
                else [None] * len(xyxy)
            )

            # Draw detected vehicles and their tracking IDs.
            for box, track_id in zip(xyxy, ids):
                x1, y1, x2, y2 = map(int, box)
                label = f"Vehicle ID {track_id}" if track_id is not None else "Vehicle"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame, label, (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
                )

            # Preliminary visual screen for overlapping vehicle boxes.
            overlap_found = False

            for (i, a), (j, b) in combinations(enumerate(xyxy), 2):
                ax1, ay1, ax2, ay2 = a
                bx1, by1, bx2, by2 = b

                intersection_w = max(0, min(ax2, bx2) - max(ax1, bx1))
                intersection_h = max(0, min(ay2, by2) - max(ay1, by1))
                intersection = intersection_w * intersection_h

                area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
                area_b = max(1, (bx2 - bx1) * (by2 - by1))
                smaller_area = min(area_a, area_b)

                overlap_ratio = intersection / smaller_area

                if overlap_ratio >= OVERLAP_THRESHOLD:
                    overlap_found = True
                    cv2.putText(
                        frame,
                        "POSSIBLE OVERLAP - REVIEW",
                        (20, 35),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255),
                        2,
                    )

            if overlap_found:
                possible_overlap_frames += 1

        cv2.putText(
            frame,
            f"Frame: {frame_number}",
            (20, height - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

        writer.write(frame)

        # Preview is optional; press Q to stop early.
        cv2.imshow("Collision Test - Q to stop", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("Preview stopped by user; output may be incomplete.")
            break

finally:
    cap.release()
    writer.release()
    cv2.destroyAllWindows()

print("\n===== COLLISION SCREENING REPORT =====")
print(f"Frames processed: {frame_number}")
print(f"Frames with vehicle detections: {frames_with_vehicles}")
print(f"Frames with possible box overlap: {possible_overlap_frames}")
print(f"Review video: {OUTPUT_PATH}")
print("Note: Box overlap is not proof of a collision.")
