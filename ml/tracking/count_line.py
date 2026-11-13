import cv2
from ultralytics import YOLO

# YOLO model
model = YOLO("yolo11n.pt")

# Traffic video
video_path = "../videos/traffic.mp4"

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("❌ Video open nahi ho rahi")
    exit()

print("✅ Traffic video loaded")
print("🚦 Counting line started...")

# Counting line
LINE_Y = 180

# Vehicles already counted
counted_ids = set()

# Vehicle counters
car_count = 0
motorcycle_count = 0
bus_count = 0
truck_count = 0

# Previous center positions
previous_positions = {}

while True:

    ret, frame = cap.read()

    if not ret:
        break

    # Resize for CPU
    frame = cv2.resize(frame, (640, 360))

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

    # Draw counting line
    cv2.line(
        frame,
        (0, LINE_Y),
        (640, LINE_Y),
        (0, 0, 255),
        3
    )

    if result.boxes.id is not None:

        boxes = result.boxes.xyxy.cpu().numpy()
        track_ids = result.boxes.id.int().cpu().tolist()
        classes = result.boxes.cls.int().cpu().tolist()

        for box, track_id, class_id in zip(
            boxes,
            track_ids,
            classes
        ):

            x1, y1, x2, y2 = map(int, box)

            # Vehicle center
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)

            # Previous position
            previous_y = previous_positions.get(track_id)

            # Save current position
            previous_positions[track_id] = center_y

            # Check line crossing
            if previous_y is not None:

                crossed = (
                    (previous_y < LINE_Y and center_y >= LINE_Y)
                     or
                    (previous_y > LINE_Y and center_y <= LINE_Y)
            )

                if crossed and track_id not in counted_ids:

                    counted_ids.add(track_id)

                    if class_id == 2:
                        car_count += 1

                    elif class_id == 3:
                        motorcycle_count += 1

                    elif class_id == 5:
                        bus_count += 1

                    elif class_id == 7:
                        truck_count += 1

            # Draw center point
            cv2.circle(
                frame,
                (center_x, center_y),
                4,
                (0, 255, 0),
                -1
            )

            # Draw ID
            cv2.putText(
                frame,
                f"ID {track_id}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2
            )

    # Total
    total = (
        car_count
        + motorcycle_count
        + bus_count
        + truck_count
    )

    # Display counts
    cv2.putText(
        frame,
        f"Cars: {car_count}",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Bikes: {motorcycle_count}",
        (20, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Buses: {bus_count}",
        (20, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Trucks: {truck_count}",
        (20, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Total Crossed: {total}",
        (20, 150),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2
    )

    cv2.imshow(
        "TRAFFICX AI - Traffic Counter",
        frame
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

print("\n========== TRAFFIC FLOW ==========")
print("Cars crossed:", car_count)
print("Motorcycles crossed:", motorcycle_count)
print("Buses crossed:", bus_count)
print("Trucks crossed:", truck_count)
print("Total vehicles crossed:", total)
print("==================================")