import cv2
from ultralytics import YOLO

# Load YOLO model
model = YOLO("yolo11n.pt")

# Input traffic video
video_path = "../videos/traffic.mp4"

# Open video
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("❌ Video open nahi ho rahi")
    exit()

print("✅ Traffic video loaded successfully")

while True:

    ret, frame = cap.read()

    if not ret:
        print("✅ Video processing complete")
        break

    # YOLO detection
    results = model(frame, classes=[2, 3, 5, 7])

    # Draw detections
    annotated_frame = results[0].plot()

    # Display
    cv2.imshow("TRAFFICX AI - Vehicle Detection", annotated_frame)

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()