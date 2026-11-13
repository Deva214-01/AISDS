
import os
import cv2
from ultralytics import YOLO

# =========================
# CONFIGURATION
# =========================
MODEL_PATH = "yolo11n.pt"
VIDEO_PATH = "videos/incident_short_test.mp4"
OUTPUT_PATH = "output/incident_review.mp4"

FRAME_WIDTH = 960
FRAME_HEIGHT = 540
CONFIDENCE = 0.2
IMAGE_SIZE = 960

# Stationary detection settings
STOPPED_SECONDS = 5
MOVEMENT_THRESHOLD = 8

# If a track disappears for too long, restart its timer
MAX_TRACK_GAP_SECONDS = 1.0

DIAGNOSTIC_INTERVAL = 30
VEHICLE_CLASSES = [2, 3, 5, 7]  # car, motorcycle, bus, truck


# =========================
# INITIALIZE MODEL & VIDEO
# =========================
model = YOLO(MODEL_PATH)
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    raise RuntimeError(f"Cannot open video: {VIDEO_PATH}")

fps = cap.get(cv2.CAP_PROP_FPS)
if fps <= 0:
    fps = 30.0

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(
    OUTPUT_PATH,
    fourcc,
    fps,
    (FRAME_WIDTH, FRAME_HEIGHT),
)

if not writer.isOpened():
    cap.release()
    raise RuntimeError(f"Cannot create output video: {OUTPUT_PATH}")


# =========================
# TRACKING STATE
# =========================
vehicle_state = {}
incident_ids = set()

frame_index = 0
diagnostic_lines = 0
frames_with_detections = 0
frames_without_ids = 0
timer_reset_count = 0

print("TRAFFICX-AI incident detection test started")
print(f"Input video: {VIDEO_PATH}")
print(f"Output video: {OUTPUT_PATH}")
print(f"Video FPS: {fps:.2f}")
print(f"Stationary threshold: {STOPPED_SECONDS}s")
print(f"Movement threshold: {MOVEMENT_THRESHOLD}px")
print("-" * 55)


# =========================
# PROCESS VIDEO
# =========================
try:
    while True:
        ok, frame = cap.read()

        if not ok:
            break

        frame_index += 1
        video_time = (frame_index - 1) / fps

        frame = cv2.resize(
            frame,
            (FRAME_WIDTH, FRAME_HEIGHT),
        )

        result = model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            classes=VEHICLE_CLASSES,
            conf=CONFIDENCE,
            imgsz=IMAGE_SIZE,
            verbose=False,
        )[0]

        annotated = result.plot()
        boxes = result.boxes

        current_ids = set()

        if boxes is not None and len(boxes) > 0:
            frames_with_detections += 1

            if boxes.id is None:
                frames_without_ids += 1
            else:
                coordinates = boxes.xyxy.cpu().numpy()
                track_ids = boxes.id.int().cpu().tolist()

                for box, track_id in zip(coordinates, track_ids):
                    x1, y1, x2, y2 = map(int, box)

                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2

                    current_ids.add(track_id)
                    state = vehicle_state.get(track_id)

                    # First observation of this ID
                    if state is None:
                        state = {
                            "anchor_x": center_x,
                            "anchor_y": center_y,
                            "still_since": video_time,
                            "last_seen": video_time,
                            "observations": 0,
                            "timer_resets": 0,
                        }
                        vehicle_state[track_id] = state

                    gap = video_time - state["last_seen"]

                    # Do not count a long tracking gap as stationary time
                    if gap > MAX_TRACK_GAP_SECONDS:
                        state["anchor_x"] = center_x
                        state["anchor_y"] = center_y
                        state["still_since"] = video_time
                        state["timer_resets"] += 1
                        timer_reset_count += 1

                    dx = center_x - state["anchor_x"]
                    dy = center_y - state["anchor_y"]
                    anchor_distance = (dx * dx + dy * dy) ** 0.5

                    # Reset timer when movement exceeds the threshold
                    if anchor_distance > MOVEMENT_THRESHOLD:
                        state["anchor_x"] = center_x
                        state["anchor_y"] = center_y
                        state["still_since"] = video_time
                        state["timer_resets"] += 1
                        timer_reset_count += 1

                    state["last_seen"] = video_time
                    state["observations"] += 1

                    stopped_duration = max(
                        0.0,
                        video_time - state["still_since"],
                    )

                    if frame_index % DIAGNOSTIC_INTERVAL == 0:
                        print(
                            f"Frame={frame_index}, ID={track_id}, "
                            f"anchor={anchor_distance:.1f}px, "
                            f"stationary_time={stopped_duration:.2f}s"
                        )
                        diagnostic_lines += 1

                    # Flag a prolonged stationary vehicle for review
                    if stopped_duration >= STOPPED_SECONDS:
                        if track_id not in incident_ids:
                            print(
                                f"[STATIONARY ALERT] ID={track_id}, "
                                f"duration={stopped_duration:.2f}s, "
                                f"frame={frame_index}"
                            )

                        incident_ids.add(track_id)

                    # Custom overlay: red = stationary alert
                    if track_id in incident_ids:
                        color = (0, 0, 255)
                        label = (
                            f"ID {track_id} STOPPED "
                            f"{stopped_duration:.1f}s"
                        )
                    else:
                        color = (0, 255, 0)
                        label = f"ID {track_id}"

                    cv2.rectangle(
                        annotated,
                        (x1, y1),
                        (x2, y2),
                        color,
                        2,
                    )

                    cv2.putText(
                        annotated,
                        label,
                        (x1, max(y1 - 8, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        color,
                        2,
                    )

        # Video overlay
        cv2.putText(
            annotated,
            f"Stationary alerts: {len(incident_ids)}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
        )

        # Save every processed frame
        writer.write(annotated)

        # Show live preview
        cv2.imshow(
            "TRAFFICX-AI - Incident Detection Test",
            annotated,
        )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("Stopped early by user.")
            break

finally:
    cap.release()
    writer.release()
    cv2.destroyAllWindows()


# =========================
# FINAL REPORT
# =========================
print("\n===== INCIDENT TEST REPORT =====")
print("Frames processed:", frame_index)
print("Frames with detections:", frames_with_detections)
print("Frames with detections but no IDs:", frames_without_ids)
print("Tracked vehicle IDs:", len(vehicle_state))
print("Diagnostic lines printed:", diagnostic_lines)
print("Timer resets:", timer_reset_count)
print("Flagged vehicle IDs:", sorted(incident_ids))
print("Number of flagged IDs:", len(incident_ids))

print("\n===== PER-TRACK SUMMARY =====")

for track_id, state in sorted(vehicle_state.items()):
    duration = max(0.0, state["last_seen"] - state["still_since"])

    print(
        f"ID={track_id}, "
        f"observations={state['observations']}, "
        f"timer_resets={state['timer_resets']}, "
        f"last_stationary_period={duration:.2f}s, "
        f"flagged={track_id in incident_ids}"
    )

print("================================")
print(f"Annotated video saved to: {OUTPUT_PATH}")
