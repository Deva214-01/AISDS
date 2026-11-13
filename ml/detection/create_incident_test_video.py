
import cv2

SOURCE_VIDEO = "videos/traffic.mp4"
OUTPUT_VIDEO = "videos/stationary_test.mp4"

cap = cv2.VideoCapture(SOURCE_VIDEO)

if not cap.isOpened():
    raise RuntimeError(f"Cannot open video: {SOURCE_VIDEO}")

fps = cap.get(cv2.CAP_PROP_FPS)
if fps <= 0:
    fps = 30.0

# Read a frame from the middle of the source video.
cap.set(cv2.CAP_PROP_POS_FRAMES, 200)
ok, frame = cap.read()
cap.release()

if not ok:
    raise RuntimeError("Could not read the selected source frame.")

frame = cv2.resize(frame, (640, 360))

writer = cv2.VideoWriter(
    OUTPUT_VIDEO,
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (640, 360),
)

if not writer.isOpened():
    raise RuntimeError(f"Cannot create output video: {OUTPUT_VIDEO}")

# Repeat the same image for six seconds.
for _ in range(int(fps * 6)):
    writer.write(frame)

writer.release()

print("Created controlled test video:", OUTPUT_VIDEO)
print("Duration: 6 seconds")
