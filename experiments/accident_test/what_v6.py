import cv2
import csv
import json
import math
import numpy as np

from pathlib import Path
from collections import defaultdict
from ultralytics import YOLO


# ============================================================
# CONFIGURATION
# ============================================================

# These values are overridden by backend/app/accident/pipeline.py
# when an uploaded video is processed.
VIDEO_PATH = Path("videos/tbone.mp4")
OUTPUT_DIR = Path("experiments/accident_test")

MODEL_PATH = "yolov8n.pt"
CONFIDENCE = 0.25
IMAGE_SIZE = 640
DEVICE = "cpu"

# COCO vehicle class IDs:
# 2 = car, 3 = motorcycle, 5 = bus, 7 = truck
VEHICLE_CLASSES = [2, 3, 5, 7]
VEHICLE_NAMES = {
    2: "CAR",
    3: "MOTORCYCLE",
    5: "BUS",
    7: "TRUCK",
}

# Repeated box overlap is only a visual warning signal.
MIN_OVERLAP_IOU = 0.05
MIN_OVERLAP_OBSERVATIONS = 3
COLLISION_WINDOW_SECONDS = 0.5
ALERT_DISPLAY_SECONDS = 2.5

MIN_TRACK_OBSERVATIONS = 5


# ============================================================
# CANDIDATE EVENT TIME
# ============================================================

def detect_event(video_path):
    """Find a visually unusual frame-change time; this is not crash proof."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    previous = None
    scores = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if previous is not None:
            difference = cv2.absdiff(gray, previous)
            scores.append(float(np.mean(difference)))
        previous = gray

    cap.release()

    if not scores or fps <= 0:
        raise RuntimeError("Invalid video or FPS; cannot determine candidate time.")

    values = np.asarray(scores, dtype=np.float32)
    rolling = np.convolve(values, np.ones(5, dtype=np.float32) / 5, mode="same")
    std = float(np.std(rolling))

    if std > 0:
        z_scores = (rolling - np.mean(rolling)) / std
    else:
        z_scores = np.zeros_like(rolling)

    candidates = np.where(z_scores > 1.5)[0]
    if len(candidates):
        index = int(candidates[np.argmax(z_scores[candidates])])
    else:
        index = int(np.argmax(rolling))

    return (index + 1) / fps, float(z_scores[index])


# ============================================================
# GEOMETRY HELPERS
# ============================================================

def box_iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    x1 = max(ax1, bx1)
    y1 = max(ay1, by1)
    x2 = min(ax2, bx2)
    y2 = min(ay2, by2)

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - intersection

    return intersection / union if union > 0 else 0.0


def distance(a, b):
    return math.hypot(a["cx"] - b["cx"], a["cy"] - b["cy"])


def direction_angle(dx, dy):
    if abs(dx) < 1 and abs(dy) < 1:
        return None
    return math.degrees(math.atan2(dy, dx))


def angle_difference(a, b):
    diff = abs(a - b) % 360
    return min(diff, 360 - diff)


# ============================================================
# TRACK ANALYSIS
# ============================================================

def analyse_track(points, event_time):
    points = sorted(points, key=lambda point: point["time"])

    before = [point for point in points if point["time"] < event_time]
    after = [point for point in points if point["time"] >= event_time]

    def section_stats(section):
        if len(section) < 2:
            return None

        first = section[0]
        last = section[-1]
        elapsed = last["time"] - first["time"]

        if elapsed <= 0:
            return None

        dx = last["cx"] - first["cx"]
        dy = last["cy"] - first["cy"]

        return {
            "pixel_speed": math.hypot(dx, dy) / elapsed,
            "angle": direction_angle(dx, dy),
            "displacement": math.hypot(dx, dy),
        }

    before_stats = section_stats(before)
    after_stats = section_stats(after)

    direction_change = None
    speed_change = None

    if before_stats and after_stats:
        if (
            before_stats["angle"] is not None
            and after_stats["angle"] is not None
        ):
            direction_change = angle_difference(
                before_stats["angle"], after_stats["angle"]
            )

        speed_change = (
            after_stats["pixel_speed"] - before_stats["pixel_speed"]
        )

    return {
        "observations": len(points),
        "first_time": points[0]["time"],
        "last_time": points[-1]["time"],
        "before_event_observations": len(before),
        "after_event_observations": len(after),
        "before_pixel_speed": (
            before_stats["pixel_speed"] if before_stats else None
        ),
        "after_pixel_speed": (
            after_stats["pixel_speed"] if after_stats else None
        ),
        "pixel_speed_change": speed_change,
        "direction_change_degrees": direction_change,
    }


# ============================================================
# FILE HELPERS
# ============================================================

def save_csv(path, rows, fieldnames=None):
    """Save CSV, including a header when there are no records."""
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows and not fieldnames:
        print("No records to save:", path)
        return

    columns = fieldnames or list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        if rows:
            writer.writerows(rows)


def draw_alert_banner(frame, alert):
    """Draw a high-visibility warning banner onto the output video."""
    height, width = frame.shape[:2]
    banner_height = min(105, max(75, height // 5))

    overlay = frame.copy()
    cv2.rectangle(
        overlay,
        (0, 0),
        (width, banner_height),
        (0, 0, 180),
        thickness=-1,
    )
    cv2.addWeighted(overlay, 0.78, frame, 0.22, 0, frame)

    title = "POSSIBLE ACCIDENT DETECTED"
    details = (
        f'{alert["vehicle_a_type"]} #{alert["vehicle_a"]}  +  '
        f'{alert["vehicle_b_type"]} #{alert["vehicle_b"]}'
    )
    time_text = f'Time: {alert["timestamp"]:.2f}s | REVIEW REQUIRED'

    cv2.putText(
        frame, title, (18, 32),
        cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame, details, (18, 62),
        cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame, time_text, (18, 88),
        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 230, 230), 1,
        cv2.LINE_AA,
    )


# ============================================================
# MAIN PIPELINE
# ============================================================

def main():
    global VIDEO_PATH, OUTPUT_DIR

    VIDEO_PATH = Path(VIDEO_PATH)
    OUTPUT_DIR = Path(OUTPUT_DIR)

    if not VIDEO_PATH.exists():
        raise FileNotFoundError(f"Input video does not exist: {VIDEO_PATH}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = VIDEO_PATH.stem

    video_output = OUTPUT_DIR / f"{stem}_v6.mp4"
    track_output = OUTPUT_DIR / f"{stem}_v6_tracks.csv"
    pair_output = OUTPUT_DIR / f"{stem}_v6_pairs.csv"
    timeline_output = OUTPUT_DIR / f"{stem}_v6_timeline.csv"
    alerts_output = OUTPUT_DIR / f"{stem}_collision_alerts.json"

    # Read video metadata.
    cap = cv2.VideoCapture(str(VIDEO_PATH))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open input video: {VIDEO_PATH}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    if fps <= 0 or width <= 0 or height <= 0:
        raise RuntimeError("Video metadata is invalid (FPS or dimensions are zero).")

    duration = total_frames / fps if total_frames else 0.0

    print("\n========== VIDEO INFORMATION ==========")
    print("Video:", VIDEO_PATH)
    print("FPS:", round(fps, 2))
    print("Duration:", round(duration, 2))
    print("Resolution:", width, "x", height)

    print("\n========== CANDIDATE EVENT DETECTION ==========")
    event_time, z_score = detect_event(VIDEO_PATH)
    print("Candidate time:", round(event_time, 2))
    print("Z-score:", round(z_score, 3))
    if event_time < 1.0:
        print(
            "WARNING: Very little pre-event footage is available; "
            "before/after comparison may be unreliable."
        )

    print("\nLoading YOLO...")
    model = YOLO(MODEL_PATH)

    cap = cv2.VideoCapture(str(VIDEO_PATH))
    if not cap.isOpened():
        raise RuntimeError(f"Could not reopen input video: {VIDEO_PATH}")

    writer = cv2.VideoWriter(
        str(video_output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Could not create output video: {video_output}")

    tracks = defaultdict(list)
    pair_records = []
    pair_overlap_history = defaultdict(list)
    collision_alerts = []
    alerted_pairs = set()
    frame_number = 0

    print("\n========== FULL VIDEO TRACKING ==========")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            timestamp = frame_number / fps

            results = model.track(
                frame,
                persist=True,
                tracker="bytetrack.yaml",
                conf=CONFIDENCE,
                imgsz=IMAGE_SIZE,
                classes=VEHICLE_CLASSES,
                device=DEVICE,
                verbose=False,
            )

            current = []

            for result in results:
                if result.boxes is None or result.boxes.id is None:
                    continue

                boxes = result.boxes.xyxy.cpu().numpy()
                ids = result.boxes.id.cpu().numpy().astype(int)
                class_ids = result.boxes.cls.cpu().numpy().astype(int)

                for box, track_id, class_id in zip(boxes, ids, class_ids):
                    x1, y1, x2, y2 = map(int, box)

                    if x2 <= x1 or y2 <= y1:
                        continue

                    class_name = VEHICLE_NAMES.get(int(class_id), "VEHICLE")
                    item = {
                        "id": int(track_id),
                        "class_id": int(class_id),
                        "class_name": class_name,
                        "time": timestamp,
                        "cx": (x1 + x2) / 2,
                        "cy": (y1 + y2) / 2,
                        "box": (x1, y1, x2, y2),
                    }

                    current.append(item)
                    tracks[int(track_id)].append(item)

                    # Normal tracking box. Alert boxes are redrawn below
                    # if this vehicle is part of a possible collision pair.
                    normal_color = (0, 255, 0)
                    cv2.rectangle(
                        frame, (x1, y1), (x2, y2), normal_color, 2
                    )
                    cv2.putText(
                        frame,
                        f"{class_name} ID {track_id}",
                        (x1, max(20, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        normal_color,
                        2,
                        cv2.LINE_AA,
                    )

            # Pair-wise measurements and repeated-overlap heuristic.
            for i in range(len(current)):
                for j in range(i + 1, len(current)):
                    a = current[i]
                    b = current[j]
                    pair = tuple(sorted((a["id"], b["id"])))

                    iou = box_iou(a["box"], b["box"])
                    center_distance = distance(a, b)

                    pair_records.append({
                        "time": round(timestamp, 4),
                        "track_a": pair[0],
                        "track_b": pair[1],
                        "center_distance_px": center_distance,
                        "iou": iou,
                    })

                    if iou >= MIN_OVERLAP_IOU:
                        pair_overlap_history[pair].append(timestamp)

                    pair_overlap_history[pair] = [
                        t for t in pair_overlap_history[pair]
                        if timestamp - t <= COLLISION_WINDOW_SECONDS
                    ]

                    recent_overlaps = pair_overlap_history[pair]

                    if (
                        len(recent_overlaps) >= MIN_OVERLAP_OBSERVATIONS
                        and pair not in alerted_pairs
                    ):
                        alerted_pairs.add(pair)

                        alert = {
                            "vehicle_a": int(a["id"]),
                            "vehicle_b": int(b["id"]),
                            "vehicle_a_type": a["class_name"],
                            "vehicle_b_type": b["class_name"],
                            "timestamp": round(timestamp, 3),
                            "evidence": "Repeated bounding-box overlap",
                            "status": "POSSIBLE_ACCIDENT",
                        }
                        collision_alerts.append(alert)

                        print(
                            "\n!!! POSSIBLE ACCIDENT !!!",
                            a["class_name"], a["id"], "+",
                            b["class_name"], b["id"],
                            "at", round(timestamp, 2), "seconds",
                        )

            # Show alert for a short period after it is triggered.
            active_alerts = [
                alert for alert in collision_alerts
                if 0 <= timestamp - alert["timestamp"] <= ALERT_DISPLAY_SECONDS
            ]

            if active_alerts:
                alert = active_alerts[-1]
                involved_ids = {
                    alert["vehicle_a"],
                    alert["vehicle_b"],
                }

                # Red boxes around the two involved vehicles whenever
                # their tracks are visible in this frame.
                for vehicle in current:
                    if vehicle["id"] in involved_ids:
                        x1, y1, x2, y2 = vehicle["box"]
                        label = (
                            f'POSSIBLE ACCIDENT: '
                            f'{vehicle["class_name"]} #{vehicle["id"]}'
                        )
                        cv2.rectangle(
                            frame, (x1, y1), (x2, y2), (0, 0, 255), 4
                        )
                        cv2.putText(
                            frame,
                            label,
                            (x1, min(height - 8, y2 + 22)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.55,
                            (0, 0, 255),
                            2,
                            cv2.LINE_AA,
                        )

                draw_alert_banner(frame, alert)

            # IMPORTANT: write the frame after drawing all annotations.
            writer.write(frame)
            frame_number += 1

    finally:
        cap.release()
        writer.release()

    # ========================================================
    # BUILD TRACK REPORT
    # ========================================================

    track_rows = []

    for track_id, points in tracks.items():
        if len(points) < MIN_TRACK_OBSERVATIONS:
            continue

        stats = analyse_track(points, event_time)
        track_rows.append({
            "track_id": track_id,
            "observations": stats["observations"],
            "first_time": round(stats["first_time"], 3),
            "last_time": round(stats["last_time"], 3),
            "before_event_observations": stats["before_event_observations"],
            "after_event_observations": stats["after_event_observations"],
            "before_pixel_speed": (
                round(stats["before_pixel_speed"], 2)
                if stats["before_pixel_speed"] is not None else ""
            ),
            "after_pixel_speed": (
                round(stats["after_pixel_speed"], 2)
                if stats["after_pixel_speed"] is not None else ""
            ),
            "pixel_speed_change": (
                round(stats["pixel_speed_change"], 2)
                if stats["pixel_speed_change"] is not None else ""
            ),
            "direction_change_degrees": (
                round(stats["direction_change_degrees"], 2)
                if stats["direction_change_degrees"] is not None else ""
            ),
        })

    # ========================================================
    # BUILD PAIR TEMPORAL REPORT
    # ========================================================

    grouped_pairs = defaultdict(list)
    for record in pair_records:
        grouped_pairs[(record["track_a"], record["track_b"])].append(record)

    pair_rows = []

    for pair, records in grouped_pairs.items():
        records.sort(key=lambda row: row["time"])

        before = [row for row in records if row["time"] < event_time]
        after = [row for row in records if row["time"] >= event_time]

        all_distances = [row["center_distance_px"] for row in records]
        before_distances = [row["center_distance_px"] for row in before]
        after_distances = [row["center_distance_px"] for row in after]

        pair_rows.append({
            "track_a": pair[0],
            "track_b": pair[1],
            "observations": len(records),
            "before_event_observations": len(before),
            "after_event_observations": len(after),
            "minimum_distance_px": round(min(all_distances), 2),
            "before_mean_distance_px": (
                round(float(np.mean(before_distances)), 2)
                if before_distances else ""
            ),
            "after_mean_distance_px": (
                round(float(np.mean(after_distances)), 2)
                if after_distances else ""
            ),
            "maximum_iou": round(max(row["iou"] for row in records), 4),
            "overlap_observations": sum(row["iou"] > 0 for row in records),
        })

    # ========================================================
    # EVENT TIMELINE
    # ========================================================

    timeline_rows = []

    for offset in [-2, -1, 0, 1, 2]:
        target_time = event_time + offset
        if target_time < 0 or target_time > duration:
            continue

        nearby = [
            row for row in pair_records
            if abs(row["time"] - target_time) <= 0.5
        ]

        timeline_rows.append({
            "relative_time_sec": offset,
            "absolute_time_sec": round(target_time, 3),
            "pair_observations": len(nearby),
            "minimum_pair_distance_px": (
                round(min(row["center_distance_px"] for row in nearby), 2)
                if nearby else ""
            ),
            "overlap_observations": sum(row["iou"] > 0 for row in nearby),
        })

    # ========================================================
    # SAVE OUTPUTS
    # ========================================================

    save_csv(track_output, track_rows, fieldnames=[
        "track_id", "observations", "first_time", "last_time",
        "before_event_observations", "after_event_observations",
        "before_pixel_speed", "after_pixel_speed",
        "pixel_speed_change", "direction_change_degrees",
    ])

    save_csv(pair_output, pair_rows, fieldnames=[
        "track_a", "track_b", "observations",
        "before_event_observations", "after_event_observations",
        "minimum_distance_px", "before_mean_distance_px",
        "after_mean_distance_px", "maximum_iou", "overlap_observations",
    ])

    save_csv(timeline_output, timeline_rows, fieldnames=[
        "relative_time_sec", "absolute_time_sec", "pair_observations",
        "minimum_pair_distance_px", "overlap_observations",
    ])

    with open(alerts_output, "w", encoding="utf-8") as file:
        json.dump(collision_alerts, file, indent=4)

    print("\n========== ANALYSIS SUMMARY ==========")
    print("Frames processed:", frame_number)
    print("Unique tracks:", len(tracks))
    print("Reliable-length tracks:", len(track_rows))
    print("Vehicle pairs:", len(pair_rows))
    print("Possible visual alerts:", len(collision_alerts))

    print("\n========== OUTPUT FILES ==========")
    print("Annotated video:", video_output)
    print("Tracks:", track_output)
    print("Pairs:", pair_output)
    print("Timeline:", timeline_output)
    print("Collision alerts:", alerts_output)

    print(
        "\nNOTE: Repeated bounding-box overlap is an exploratory warning "
        "signal, not proof of a real collision. Review the video manually."
    )

    # Return useful data to the backend pipeline, while retaining file outputs.
    return {
        "candidate_event_time": round(float(event_time), 3),
        "candidate_event_z_score": round(float(z_score), 3),
        "collision_alerts": collision_alerts,
        "annotated_video": str(video_output),
        "collision_alerts_file": str(alerts_output),
    }


if __name__ == "__main__":
    main()
