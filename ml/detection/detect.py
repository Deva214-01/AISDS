from ultralytics import YOLO

# Load pretrained YOLO model
model = YOLO("yolo11n.pt")

# Test image
image_url = "https://ultralytics.com/images/bus.jpg"

# Run detection
results = model(image_url)

# Show detection result
for result in results:
    result.show()