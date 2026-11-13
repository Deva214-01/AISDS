
import cv2

INPUT_VIDEO = "videos/stationary_test.mp4"
OUTPUT_VIDEO = "videos/incident_short_test.mp4"
DURATION_SECONDS = 30

cap = cv2.VideoCapture(INPUT_VIDEO)

if not cap.isOpened():
    raise RuntimeError(f"Cannot open video: {INPUT_VIDEO}")

fps = cap.get(cv2.CAP_PROP_FPS)
if fps <= 0:
    fps = 30.0

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(fps * DURATION_SECONDS)

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(
    OUTPUT_VIDEO, fourcc, fps, (width, height)
)

if not writer.isOpened():
    cap.release()
    raise RuntimeError("Could not create short test video")

written = 0

try:
    while written < total_frames:
        ok, frame = cap.read()
        if not ok:
            break

        writer.write(frame)
        written += 1
finally:
    cap.release()
    writer.release()

print(f"Frames written: {written}")
print(f"Duration: {written / fps:.2f} seconds")
print(f"Saved to: {OUTPUT_VIDEO}")

