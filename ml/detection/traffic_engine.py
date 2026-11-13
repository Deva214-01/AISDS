
import os
import cv2
from ultralytics import YOLO


class TrafficEngine:
    def __init__(self, video_path, output_path=None, stats_callback=None):
        # Input video
        self.video_path = os.path.abspath(video_path)
        self.stats_callback = stats_callback

        # Output video: TRA/TRA/output/traffic_annotated.mp4
        project_dir = os.path.dirname(
            os.path.dirname(
                os.path.abspath(__file__)
            )
        )

        if output_path is None:
            output_path = os.path.join(
                project_dir,
                "output",
                "traffic_annotated.mp4",
            )

        self.output_path = os.path.abspath(output_path)
        os.makedirs(
            os.path.dirname(self.output_path),
            exist_ok=True,
        )

        # YOLO model
        self.model = YOLO("yolo11n.pt")

        # Unique vehicle tracking IDs
        self.car_ids = set()
        self.motorcycle_ids = set()
        self.bus_ids = set()
        self.truck_ids = set()

        # Virtual monitoring line
        self.line_y = 180
        self.crossed_ids = set()

        # Speed estimation state
        self.previous_positions = {}
        self.speed_samples = []

        # Approximate calibration only
        # Replace with camera calibration for real-world speed.
        self.pixel_to_meter = 0.05

        # Video properties
        self.frame_width = 640
        self.frame_height = 360

    # ==========================================
    # PROCESS VIDEO AND SAVE ANNOTATED OUTPUT
    # ==========================================

    def process_video(self):
        if not os.path.isfile(self.video_path):
            print(f"Input video not found: {self.video_path}")
            return None

        cap = cv2.VideoCapture(self.video_path)

        if not cap.isOpened():
            print(f"Video cannot be opened: {self.video_path}")
            return None

        fps = cap.get(cv2.CAP_PROP_FPS)

        if not fps or fps <= 0:
            fps = 25.0

        # We process every second input frame.
        # Half FPS keeps approximately the original video duration.
        output_fps = max(fps / 2.0, 1.0)

        writer = None
        frame_index = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1

        print("Traffic Engine started...")
        print(f"Input video: {self.video_path}")
        print(f"Output video: {self.output_path}")

        try:
            while True:
                ret, frame = cap.read()

                if not ret:
                    break

                frame_index += 1

                # Process every second frame to reduce CPU usage
                if frame_index % 2 != 0:
                    continue

                frame = cv2.resize(
                    frame,
                    (self.frame_width, self.frame_height),
                )

                results = self.model.track(
                    frame,
                    persist=True,
                    tracker="bytetrack.yaml",
                    classes=[2, 3, 5, 7],
                    conf=0.3,
                    imgsz=640,
                    verbose=False,
                )

                result = results[0]

                # Initialize writer using the first processed frame
                if writer is None:
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

                    writer = cv2.VideoWriter(
                        self.output_path,
                        fourcc,
                        output_fps,
                        (self.frame_width, self.frame_height),
                    )

                    if not writer.isOpened():
                        raise RuntimeError(
                            "Could not create output video. "
                            "Check OpenCV codec support."
                        )

                # Draw virtual monitoring line
                cv2.line(
                    frame,
                    (0, self.line_y),
                    (self.frame_width, self.line_y),
                    (0, 0, 255),
                    2,
                )

                cv2.putText(
                    frame,
                    "MONITORING LINE",
                    (10, self.line_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (0, 0, 255),
                    1,
                )

                # Extract detections and tracking IDs
                boxes_object = result.boxes

                if (
                    boxes_object is not None
                    and len(boxes_object) > 0
                ):
                    track_ids = (
                        boxes_object.id.int().cpu().tolist()
                        if boxes_object.id is not None
                        else [None] * len(boxes_object)
                    )

                    classes = (
                        boxes_object.cls.int().cpu().tolist()
                    )

                    boxes = (
                        boxes_object.xyxy.cpu().tolist()
                    )

                    confidences = (
                        boxes_object.conf.cpu().tolist()
                    )

                    for track_id, class_id, box, confidence in zip(
                        track_ids,
                        classes,
                        boxes,
                        confidences,
                    ):
                        x1, y1, x2, y2 = map(int, box)

                        center_x = (x1 + x2) / 2
                        center_y = (y1 + y2) / 2

                        # Vehicle class labels and colors
                        if class_id == 2:
                            label = "CAR"
                            color = (0, 255, 0)

                        elif class_id == 3:
                            label = "MOTORCYCLE"
                            color = (255, 200, 0)

                        elif class_id == 5:
                            label = "BUS"
                            color = (255, 0, 0)

                        elif class_id == 7:
                            label = "TRUCK"
                            color = (0, 165, 255)

                        else:
                            continue

                        # Count unique tracked vehicles
                        # Statistics retain the original logic.
                        if track_id is not None:
                            if class_id == 2:
                                self.car_ids.add(track_id)
                            elif class_id == 3:
                                self.motorcycle_ids.add(track_id)
                            elif class_id == 5:
                                self.bus_ids.add(track_id)
                            elif class_id == 7:
                                self.truck_ids.add(track_id)

                        # Draw detection bounding box
                        cv2.rectangle(
                            frame,
                            (x1, y1),
                            (x2, y2),
                            color,
                            2,
                        )

                        # Display label, confidence and track ID
                        if track_id is not None:
                            label_text = (
                                f"{label} ID:{track_id} "
                                f"{confidence:.0%}"
                            )
                        else:
                            label_text = (
                                f"{label} {confidence:.0%}"
                            )

                        label_y = max(y1 - 8, 18)

                        cv2.putText(
                            frame,
                            label_text,
                            (x1, label_y),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.45,
                            color,
                            2,
                        )

                        # Line crossing and speed require a track ID
                        if track_id is None:
                            continue

                        previous = self.previous_positions.get(
                            track_id
                        )

                        if previous is not None:
                            prev_x, prev_y, prev_frame = previous

                            crossed = (
                                prev_y < self.line_y <= center_y
                                or
                                prev_y > self.line_y >= center_y
                            )

                            if (
                                crossed
                                and track_id not in self.crossed_ids
                            ):
                                self.crossed_ids.add(track_id)

                            # Approximate speed calculation
                            elapsed_seconds = (
                                frame_index - prev_frame
                            ) / fps

                            if elapsed_seconds > 0:
                                distance_pixels = (
                                    (center_x - prev_x) ** 2
                                    + (center_y - prev_y) ** 2
                                ) ** 0.5

                                if distance_pixels >= 2:
                                    speed_kmh = (
                                        distance_pixels
                                        * self.pixel_to_meter
                                        / elapsed_seconds
                                        * 3.6
                                    )

                                    if 1 <= speed_kmh <= 150:
                                        self.speed_samples.append(
                                            speed_kmh
                                        )

                        self.previous_positions[track_id] = (
                            center_x,
                            center_y,
                            frame_index,
                        )

                # Draw live processing statistics onto each output frame
                total = (
                    len(self.car_ids)
                    + len(self.motorcycle_ids)
                    + len(self.bus_ids)
                    + len(self.truck_ids)
                )

                cv2.rectangle(
                    frame,
                    (5, 5),
                    (235, 45),
                    (20, 20, 20),
                    -1,
                )

                cv2.putText(
                    frame,
                    f"Vehicles detected: {total}",
                    (12, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    2,
                )

                # Save every processed frame
                writer.write(frame)
                # Send updated statistics to the backend
                if self.stats_callback is not None:
                    stats = self.get_statistics()

                    stats["progress"] = round(
                        min(frame_index / total_frames * 100, 100), 1
                    )

                    stats["processed_frames"] = frame_index
                    stats["total_frames"] = total_frames

                    self.stats_callback(stats)

        finally:
            cap.release()

            if writer is not None:
                writer.release()

        # Confirm output file exists and is not empty
        if (
            writer is None
            or not os.path.isfile(self.output_path)
            or os.path.getsize(self.output_path) == 0
        ):
            print("Annotated output video was not generated.")
            return None

        print("Traffic Engine completed.")
        print(f"Annotated video saved: {self.output_path}")

        return self.get_statistics()

    # ==========================================
    # CALCULATE TRAFFIC STATISTICS
    # ==========================================

    def get_statistics(self):
        cars = len(self.car_ids)
        motorcycles = len(self.motorcycle_ids)
        buses = len(self.bus_ids)
        trucks = len(self.truck_ids)

        total = cars + motorcycles + buses + trucks

        # Prototype density, not calibrated road occupancy
        density = min(total / 50 * 100, 100)

        if density < 30:
            congestion = "LOW"
        elif density < 70:
            congestion = "MEDIUM"
        else:
            congestion = "HIGH"

        average_speed = (
            sum(self.speed_samples) / len(self.speed_samples)
            if self.speed_samples
            else 0
        )

        line_crossings = len(self.crossed_ids)

        return {
            "total_vehicles": total,
            "cars": cars,
            "motorcycles": motorcycles,
            "buses": buses,
            "trucks": trucks,
            "traffic_density": round(density),
            "congestion": congestion,
            "violations": 0,
            "line_crossings": line_crossings,
            "average_speed": round(average_speed, 1),
        }


# ==========================================
# RUN ENGINE DIRECTLY FOR TESTING
# ==========================================

if __name__ == "__main__":
    project_dir = os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))
        )
    )

    video_path = os.path.join(
        project_dir,
        "videos",
        "traffic.mp4",
    )

    output_path = os.path.join(
        project_dir,
        "output",
        "traffic_annotated.mp4",
    )

    engine = TrafficEngine(
        video_path=video_path,
        output_path=output_path,
    )

    data = engine.process_video()

    print("\n========== TRAFFIC ENGINE ==========")

    if data is not None:
        for key, value in data.items():
            print(f"{key}: {value}")

    print("====================================")
