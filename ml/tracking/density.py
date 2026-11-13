import cv2
from ultralytics import YOLO

# Load model
model = YOLO("yolo11n.pt")

# Video
video_path = "../videos/traffic.mp4"

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("❌ Video open nahi ho rahi")
    exit()

print("✅ Traffic video loaded")
print("🚦 Traffic density analysis started...")

while True:

    ret, frame = cap.read()

    if not ret:
        break

    # Resize for CPU
    frame = cv2.resize(frame, (640, 360))

    # Vehicle detection
    results = model(
        frame,
        classes=[2, 3, 5, 7],
        conf=0.3,
        imgsz=640,
        verbose=False
    )

    result = results[0]

    # Number of vehicles currently visible
    vehicle_count = len(result.boxes)

    # Density calculation
    # 10 vehicles = approximately 100% density
    density = min(vehicle_count / 10 * 100, 100)

    # Determine congestion
    if density < 30:
        congestion = "LOW"

    elif density < 70:
        congestion = "MEDIUM"

    else:
        congestion = "HIGH"

    # Draw detections
    annotated_frame = result.plot()

    # Display vehicle count
    cv2.putText(
        annotated_frame,
        f"Vehicles: {vehicle_count}",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    # Display density
    cv2.putText(
        annotated_frame,
        f"Density: {density:.0f}%",
        (20, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    # Display congestion
    cv2.putText(
        annotated_frame,
        f"Congestion: {congestion}",
        (20, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2
    )

    cv2.imshow(
        "TRAFFICX AI - Traffic Density",
        annotated_frame
    )

    # Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

print("\n========== TRAFFIC ANALYSIS ==========")
print("Last Vehicle Count:", vehicle_count)
print(f"Last Density: {density:.0f}%")
print("Congestion:", congestion)
print("======================================")