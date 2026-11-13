import cv2
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
print("🚦 Vehicle tracking started...")

while True:

    ret, frame = cap.read()

    if not ret:
        print("✅ Tracking complete")
        break

    # Resize frame for faster CPU processing
    frame = cv2.resize(frame, (640, 360))

    # YOLO + ByteTrack
    results = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        classes=[2, 3, 5, 7],
        conf=0.3,
        imgsz=640,
        verbose=False
    )

    # Draw boxes and IDs
    annotated_frame = results[0].plot()

    # Display
    cv2.imshow("TRAFFICX AI - Vehicle Tracking", annotated_frame)

    # Q = quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()