from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, HTTPException, UploadFile, File
import shutil
import uuid
import subprocess
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
import json
from datetime import datetime
import os
import sys
import threading
import importlib
import cv2
from collections import Counter
import time
import base64
# ==========================================
# PROJECT PATHS
# ==========================================
BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
)
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)
ENGINE_OUTPUT_PATH = os.path.join(
    OUTPUT_DIR, "traffic_annotated.mp4"
)
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)
ANNOTATED_VIDEO_PATH = os.path.join(OUTPUT_DIR, "traffic_browser.mp4")
ML_DIR = os.path.join(BASE_DIR, "ml")
if ML_DIR not in sys.path:
    sys.path.insert(0, ML_DIR)
def get_traffic_engine_class():
    return importlib.import_module(
        "detection.traffic_engine"
    ).TrafficEngine


HISTORY_FILE = os.path.join(
    os.path.dirname(__file__),
    "history.json"
)
# ==========================================
# FASTAPI APP
# ==========================================
app = FastAPI(
    title="TRAFFICX AI",
    description="AI Traffic Intelligence System",
    version="1.0.0",
)

from .accident.router import router as accident_router
app.include_router(accident_router)

from .ppe.router import router as ppe_router
app.include_router(ppe_router)

from .number_plate.router import router as number_plate_router
app.include_router(number_plate_router)

from .fire_smoke.router import router as fire_smoke_router
app.include_router(fire_smoke_router)
# ==========================================
# CORS
# ==========================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        origin.strip()
        for origin in os.getenv(
            "FRONTEND_ORIGINS",
            "http://localhost:5173,http://127.0.0.1:5173"
        ).split(",")
        if origin.strip()
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$",
)
# ==========================================
# SHARED STATE
# ==========================================
traffic_lock = threading.Lock()
traffic_state = {
    "status": "idle",
    "total_vehicles": 0,
    "cars": 0,
    "motorcycles": 0,
    "buses": 0,
    "trucks": 0,
    "traffic_density": 0,
    "congestion": "UNKNOWN",
    "violations": 0,
    "line_crossings": 0,
    "average_speed": 0,
    "error": None,
    "worker_running": False,
}
# ==========================================
# LIVE CAMERA STATE
# ==========================================
camera_lock = threading.Lock()
camera_model = None
helmet_model = None
camera_running = False
camera_starting = False
camera_stats = {
    "status": "stopped",
    "total_vehicles": 0,
    "cars": 0,
    "motorcycles": 0,
    "buses": 0,
    "trucks": 0,
    "error": None,
}
camera_stop_event = threading.Event()
camera_stopped_event = threading.Event()
camera_stopped_event.set()

helmet_alerts = []
frame_counts = Counter()

camera_session_seen = set()
camera_session_counts = Counter()
camera_track_positions = {}

COUNT_LINE_RATIO = 0.65

helmet_alert_history = {}
HELMET_ALERT_COOLDOWN = 15
HELMET_CONFIRMATIONS = 3
helmet_confirmation_counts = {}

# Browser webcam state: frames arrive from the HTTPS frontend.
browser_frame_lock = threading.Lock()
browser_camera_active = False
browser_frame_count = 0

camera_session_started_at = None
camera_session_saved = True
# ==========================================
# BACKGROUND PROCESSING
# ==========================================
def process_traffic_in_background():
    global traffic_state
    try:
        video_path = os.path.join(
            BASE_DIR,
            "videos",
            "traffic.mp4",
        )
        if not os.path.isfile(video_path):
            raise FileNotFoundError(
                f"Traffic video not found: {video_path}"
            )
        print("TRAFFICX AI: Processing started")
        print(f"Video: {video_path}")
        def update_live_stats(data):
            with traffic_lock:
                traffic_state.update(data)
                traffic_state["status"] = "processing"
                traffic_state["error"] = None
        # Create the engine before processing
        engine = get_traffic_engine_class()(
            video_path,
            output_path=ENGINE_OUTPUT_PATH,
            stats_callback=update_live_stats,
        )
        data = engine.process_video()
        if data is None:
            raise RuntimeError(
                "Video could not be processed."
            )
        with traffic_lock:
            traffic_state.update(data)
            traffic_state.update({
                "status": "completed",
                "progress": 100,
                "error": None,
                "worker_running": False,
            })
            completed_record = dict(traffic_state)
        save_analysis_history(completed_record)
        print("TRAFFICX AI: Processing completed")
        print(f"Statistics: {data}")
    except Exception as exc:
        print(f"Traffic processing error: {exc}")
        with traffic_lock:
            traffic_state.update({
                "status": "error",
                "error": str(exc),
                "worker_running": False,
            })
# ==========================================
# HOME
# ==========================================
@app.get("/")
def home():
    return {
        "message": "TRAFFICX AI Backend is running",
        "status": "healthy",
    }
# ==========================================
# HEALTH CHECK
# ==========================================
@app.get("/health")
def health():
    return {"status": "healthy"}
# ==========================================
# TRAFFIC API
# ==========================================
@app.get("/api/traffic")
def traffic_data():
    with traffic_lock:
        return dict(traffic_state)
# ==========================================
# RESET FOR A NEW ANALYSIS
# ==========================================
@app.post("/api/traffic/reset")
def reset_traffic():
    with traffic_lock:
        if traffic_state["worker_running"]:
            return {
                "message": "Analysis is still running.",
                "status": traffic_state["status"],
            }
        traffic_state.update({
            "status": "idle",
            "total_vehicles": 0,
            "cars": 0,
            "motorcycles": 0,
            "buses": 0,
            "trucks": 0,
            "traffic_density": 0,
            "congestion": "UNKNOWN",
            "violations": 0,
            "line_crossings": 0,
            "average_speed": 0,
            "error": None,
            "worker_running": False,
            "progress": 0,
            "processed_frames": 0,
            "total_frames": 0,
        })
    return {
        "message": "Traffic state reset successfully",
        "status": "idle",
    }
@app.get("/api/traffic/video")
def get_annotated_video():
    if not os.path.exists(ANNOTATED_VIDEO_PATH):
        raise HTTPException(
            status_code=404,
            detail="Annotated video file not found"
        )
    return FileResponse(
        path=ANNOTATED_VIDEO_PATH,
        media_type="video/mp4",
        filename="traffic_annotated.mp4",
        content_disposition_type="inline"
    )
# ==========================================
# UPLOAD TRAFFIC VIDEO
# ==========================================
@app.post("/api/traffic/upload")
async def upload_traffic_video(file: UploadFile = File(...)):
    allowed_extensions = {".mp4", ".avi", ".mov", ".mkv"}
    original_name = os.path.basename(file.filename or "")
    extension = os.path.splitext(original_name)[1].lower()
    if extension not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail="Unsupported format. Upload MP4, AVI, MOV, or MKV."
        )
    with traffic_lock:
        if traffic_state["worker_running"]:
            raise HTTPException(
                status_code=409,
                detail="Traffic analysis is already running."
            )
        traffic_state.update({
            "status": "processing",
            "total_vehicles": 0,
            "cars": 0,
            "motorcycles": 0,
            "buses": 0,
            "trucks": 0,
            "traffic_density": 0,
            "congestion": "UNKNOWN",
            "violations": 0,
            "line_crossings": 0,
            "average_speed": 0,
            "progress": 0,
            "processed_frames": 0,
            "total_frames": 0,
            "error": None,
            "worker_running": True,
        })
    upload_path = os.path.join(
        UPLOAD_DIR, f"{uuid.uuid4().hex}{extension}"
    )
    try:
        with open(upload_path, "wb") as destination:
            shutil.copyfileobj(file.file, destination)
    except Exception as exc:
        with traffic_lock:
            traffic_state.update({
                "status": "error",
                "error": f"Could not save uploaded video: {exc}",
                "worker_running": False,
            })
        raise HTTPException(
            status_code=500,
            detail="Could not save uploaded video."
        )
    finally:
        await file.close()
    def process_uploaded_video():
        try:
            def update_live_stats(data):
                with traffic_lock:
                    traffic_state.update(data)
                    traffic_state["status"] = "processing"
                    traffic_state["error"] = None
            # 1. Process uploaded video with YOLO
            engine = get_traffic_engine_class()(
                upload_path,
                output_path=ENGINE_OUTPUT_PATH,
                stats_callback=update_live_stats,
            )
            data = engine.process_video()
            if data is None:
                raise RuntimeError(
                    "Uploaded video could not be processed."
                )
            if not os.path.isfile(ENGINE_OUTPUT_PATH):
                raise RuntimeError(
                    "Processed video file was not generated."
                )
            # 2. Convert to browser-compatible H.264 MP4
            import imageio_ffmpeg
            ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
            result = subprocess.run(
                [
                    ffmpeg_path,
                    "-y",
                    "-i", ENGINE_OUTPUT_PATH,
                    "-c:v", "libx264",
                    "-preset", "fast",
                    "-pix_fmt", "yuv420p",
                    "-movflags", "+faststart",
                    "-an",
                    ANNOTATED_VIDEO_PATH,
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode != 0:
                raise RuntimeError(
                    "FFmpeg conversion failed: "
                    + result.stderr[-2000:]
                )
            if (
                not os.path.isfile(ANNOTATED_VIDEO_PATH)
                or os.path.getsize(ANNOTATED_VIDEO_PATH) == 0
            ):
                raise RuntimeError(
                    "Browser-compatible output video is missing."
                )
            print("Uploaded video processing completed.")
            print(f"Browser video: {ANNOTATED_VIDEO_PATH}")
            # 3. Mark analysis as completed
            with traffic_lock:
                traffic_state.update(data)
                traffic_state.update({
                    "status": "completed",
                    "progress": 100,
                    "error": None,
                    "worker_running": False,
                })
            with traffic_lock:
                completed_record = dict(traffic_state)
            completed_record["video_filename"] = original_name
            completed_record["analysis_type"] = "uploaded_video"
            save_analysis_history(completed_record)
        except Exception as exc:
            print(f"Uploaded video processing error: {exc}")
            with traffic_lock:
                traffic_state.update({
                    "status": "error",
                    "error": str(exc),
                    "worker_running": False,
                })
        finally:
            if os.path.exists(upload_path):
                try:
                    os.remove(upload_path)
                except OSError as exc:
                    print(f"Could not remove upload: {exc}")
    # IMPORTANT: Start the thread OUTSIDE the function above
    threading.Thread(
        target=process_uploaded_video,
        daemon=True,
    ).start()
    return {
        "message": "Video uploaded successfully. Analysis started.",
        "status": "processing",
    }
@app.get("/api/traffic/video/download")
def download_traffic_video():
    if not os.path.isfile(ANNOTATED_VIDEO_PATH):
        raise HTTPException(
            status_code=404,
            detail="Processed video not found. Process a video first."
        )
    return FileResponse(
        path=ANNOTATED_VIDEO_PATH,
        media_type="video/mp4",
        filename="trafficx_processed_video.mp4",
        content_disposition_type="attachment",
    )
@app.get("/api/traffic/report/download")
def download_traffic_report():
    with traffic_lock:
        report = dict(traffic_state)
    report.pop("worker_running", None)
    return Response(
        content=json.dumps(report, indent=4),
        media_type="application/json",
        headers={
            "Content-Disposition":
                'attachment; filename="trafficx_report.json"'
        },
    )
def save_analysis_history(data):
    os.makedirs(os.path.dirname(HISTORY_FILE), exist_ok=True)
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            history = json.load(f)
        if not isinstance(history, list):
            history = []
    except (FileNotFoundError, json.JSONDecodeError):
        history = []
    record = dict(data)
    record["completed_at"] = datetime.now().astimezone().isoformat()
    history.insert(0, record)
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=4)
# ==========================================
# LIVE WEBCAM ANALYSIS
# ==========================================
VEHICLE_CLASSES = {
    2: "cars",
    3: "motorcycles",
    5: "buses",
    7: "trucks",
}
def get_camera_model():
    global camera_model
    if camera_model is None:
        ultralytics = importlib.import_module("ultralytics")
        camera_model = ultralytics.YOLO("yolo11n.pt")
    return camera_model
def get_helmet_model():
    global helmet_model

    if helmet_model is None:
        from ultralytics import YOLO

        project_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..")
        )

        ppe_path = os.path.join(
            project_root, "ml", "models", "ppe_best.pt"
        )

        if not os.path.isfile(ppe_path):
            raise FileNotFoundError(
                f"PPE V2 model not found: {ppe_path}"
            )

        helmet_model = YOLO(ppe_path)
        print("PPE V2 LIVE MODEL LOADED:", helmet_model.names)

    return helmet_model


def update_helmet_alerts(detections):
    global helmet_alerts, helmet_alert_history

    now = time.monotonic()
    confirmed_boxes = []

    missing_classes = {
        "no_helmet": "NO_HELMET",
        "no_glove": "NO_GLOVE",
        "no_goggles": "NO_GOGGLES",
        "no_mask": "NO_MASK",
        "no_shoes": "NO_SHOES",
    }

    for box in detections:
        class_id = int(box.cls[0].item())
        class_name = str(helmet_model.names[class_id]).lower()
        confidence = float(box.conf[0].item())

        if class_name not in missing_classes or confidence < 0.35:
            continue

        x1, y1, x2, y2 = map(
            int, box.xyxy[0].tolist()
        )

        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2

        matched_key = None

        for key, item in helmet_alert_history.items():
            old_x, old_y = item["center"]

            if (
                item.get("class_name") == class_name
                and abs(center_x - old_x) < 60
                and abs(center_y - old_y) < 60
                and now - item["last_seen"] < 3
            ):
                matched_key = key
                break

        if matched_key is None:
            matched_key = f"{class_name}_{center_x}_{center_y}"
            helmet_alert_history[matched_key] = {
                "class_name": class_name,
                "center": (center_x, center_y),
                "hits": 0,
                "last_seen": now,
                "last_alert": 0,
            }

        item = helmet_alert_history[matched_key]
        item["center"] = (center_x, center_y)
        item["last_seen"] = now
        item["hits"] += 1

        if item["hits"] >= HELMET_CONFIRMATIONS:
            confirmed_boxes.append(
                (x1, y1, x2, y2, confidence, class_name)
            )

            if now - item["last_alert"] >= HELMET_ALERT_COOLDOWN:
                with camera_lock:
                    helmet_alerts.insert(0, {
                        "id": str(uuid.uuid4()),
                        "timestamp": datetime.now().astimezone().isoformat(),
                        "type": missing_classes[class_name],
                        "confidence": round(confidence, 3),
                        "message": f"Possible {class_name.replace('_', ' ')} violation. Review required.",
                    })

                    helmet_alerts = helmet_alerts[:50]

                item["last_alert"] = now

    helmet_alert_history = {
        key: item
        for key, item in helmet_alert_history.items()
        if now - item["last_seen"] < 5
    }

    return confirmed_boxes
# ==========================================
# LIVE TRAFFIC INTELLIGENCE
# ==========================================
traffic_intelligence = {
    "status": "waiting",
    "occupancy_percent": 0.0,
    "congestion": "UNKNOWN",
    "alert": None,
    "last_updated": None,
    "recent_events": [],
}

CONGESTION_LOW_MAX = 5.0
CONGESTION_MEDIUM_MAX = 12.0
MAX_RECENT_TRAFFIC_EVENTS = 10


# Stable congestion detection
CONGESTION_CONFIRM_FRAMES = 5
congestion_candidate = "UNKNOWN"
congestion_candidate_frames = 0
CONGESTION_HOLD_SECONDS = 3.0
congestion_last_changed_at = 0.0


def calculate_traffic_intelligence(frame, frame_counts):
    """
    Estimate vehicle-box occupancy in the camera frame.
    This is a demo indicator, not calibrated road congestion.
    """
    global traffic_intelligence

    frame_height, frame_width = frame.shape[:2]
    frame_area = max(frame_height * frame_width, 1)

    results = get_camera_model().predict(
        frame,
        classes=list(VEHICLE_CLASSES.keys()),
        verbose=False
    )

    vehicle_area = 0

    if results and results[0].boxes is not None:
        for box in results[0].boxes.xyxy:
            x1, y1, x2, y2 = box.tolist()

            box_width = max(0, x2 - x1)
            box_height = max(0, y2 - y1)

            vehicle_area += box_width * box_height

    occupancy = min(100.0, (vehicle_area / frame_area) * 100)

    if occupancy < CONGESTION_LOW_MAX:
        level = "LOW"
    elif occupancy < CONGESTION_MEDIUM_MAX:
        level = "MEDIUM"
    else:
        level = "HIGH"

    now = datetime.now().astimezone().isoformat()

    with camera_lock:
        previous_level = traffic_intelligence["congestion"]
        alert = None

        if previous_level not in ("UNKNOWN", level):
            alert = f"Traffic changed from {previous_level} to {level}"

            traffic_intelligence["recent_events"].insert(0, {
                "timestamp": now,
                "message": alert,
                "congestion": level,
            })

            traffic_intelligence["recent_events"] = (
                traffic_intelligence["recent_events"][
                    :MAX_RECENT_TRAFFIC_EVENTS
                ]
            )

        traffic_intelligence.update({
            "status": "running",
            "occupancy_percent": round(occupancy, 2),
            "congestion": level,
            "alert": alert,
            "last_updated": now,
            "vehicle_counts": dict(frame_counts),
        })

    return level, occupancy

def generate_camera_frames():
    global camera_running, camera_starting, camera_stats
    global camera_session_seen, camera_session_counts
    global camera_track_positions
    global camera_session_started_at, camera_session_saved
    global congestion_last_changed_at

    global congestion_candidate, congestion_candidate_frames


    cap = None
    camera_stop_event.clear()
    camera_stopped_event.clear()
    camera_session_seen = set()
    camera_track_positions = {}
    camera_session_counts = Counter()
    camera_track_positions = {}
    COUNT_LINE_RATIO = 0.65

    congestion_candidate = "UNKNOWN"
    congestion_candidate_frames = 0

    camera_session_started_at = datetime.now().astimezone().isoformat()
    camera_session_saved = False

    with camera_lock:
        camera_stats.update({"status": "starting", "total_vehicles": 0, "cars": 0,
            "motorcycles": 0, "buses": 0, "trucks": 0, "error": None,
            "session_total_vehicles": 0, "session_cars": 0, "session_motorcycles": 0,
            "session_buses": 0, "session_trucks": 0})

    try:
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not cap.isOpened():
            raise RuntimeError("Could not open webcam. Check camera permission or whether another app is using it.")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        model = get_camera_model()
        helmet_detector = get_helmet_model()
        helmet_frame_interval = 5
        helmet_frame_count = 0
        last_helmet_detection = []
        confirmed_helmet_boxes = []
        with camera_lock:
            camera_running = True
            camera_starting = False
            camera_stats.update({"status": "running", "error": None})

        while not camera_stop_event.is_set():
            success, frame = cap.read()
            if not success or frame is None:
                print("Camera warning: Could not read a frame.")
                break

            frame_height, frame_width = frame.shape[:2]


            results = model.track(frame, persist=True, classes=list(VEHICLE_CLASSES.keys()),
                                  verbose=False, tracker="bytetrack.yaml")
            # Helmet detection on selected frames to reduce CPU usage
            helmet_frame_count += 1

            if helmet_frame_count % helmet_frame_interval == 0:
                helmet_results = helmet_detector.predict(
                    frame,
                    imgsz=640,
                    conf=0.15,
                    device="cpu",
                    verbose=False
                )

                last_helmet_detection = helmet_results[0].boxes

                raw_predictions = []
                for detected_box in last_helmet_detection:
                    detected_id = int(detected_box.cls[0].item())
                    detected_name = str(helmet_detector.names[detected_id])
                    detected_conf = float(detected_box.conf[0].item())
                    raw_predictions.append(
                        (detected_name, round(detected_conf, 3))
                    )

                print("LIVE PPE RAW PREDICTIONS:", raw_predictions, flush=True)

                confirmed_helmet_boxes = update_helmet_alerts(
                    last_helmet_detection
                )

            vehicle_box_area = 0.0
            if results and results[0].boxes is not None:
                boxes = results[0].boxes
                track_ids = boxes.id
                for index, box in enumerate(boxes):
                    category = VEHICLE_CLASSES.get(int(box.cls[0].item()))
                    if category:
                        frame_counts[category] += 1
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        box_area = max(0, x2 - x1) * max(0, y2 - y1)
                        vehicle_box_area += box_area
                        if track_ids is not None:
                            track_id = int(track_ids[index].item())

                            center_y = int((y1 + y2) / 2)
                            count_line_y = int(frame.shape[0] * COUNT_LINE_RATIO)

                            previous_y = camera_track_positions.get(track_id)

                            crossed_line = (
                                previous_y is not None
                                and (
                                    (previous_y < count_line_y <= center_y)
                                    or
                                    (previous_y > count_line_y >= center_y)
                                )
                            )

                            if crossed_line and track_id not in camera_session_seen:
                                camera_session_seen.add(track_id)
                                camera_session_counts[category] += 1

                            camera_track_positions[track_id] = center_y
                frame = results[0].plot()
                count_line_y = int(frame_height * COUNT_LINE_RATIO)

                cv2.line(
                    frame,
                    (0, count_line_y),
                    (frame_width, count_line_y),
                    (0, 255, 255),
                    2
                )

                cv2.putText(
                    frame,
                    "VEHICLE COUNTING LINE",
                    (10, count_line_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 255),
                    2
                )
                frame_height, frame_width = frame.shape[:2]
                frame_area = max(frame_height * frame_width, 1)

                occupancy = min(100.0, (vehicle_box_area / frame_area) * 100)


                if occupancy < CONGESTION_LOW_MAX:
                    detected_level = "LOW"
                elif occupancy < CONGESTION_MEDIUM_MAX:
                    detected_level = "MEDIUM"
                else:
                    detected_level = "HIGH"


                # Require five consecutive frames before changing the level.
                if detected_level == congestion_candidate:
                    congestion_candidate_frames += 1
                else:
                    congestion_candidate = detected_level
                    congestion_candidate_frames = 1

                current_level = traffic_intelligence["congestion"]
                now_monotonic = time.monotonic()

                if current_level == "UNKNOWN":
                    if congestion_candidate_frames >= CONGESTION_CONFIRM_FRAMES:
                        congestion_level = congestion_candidate
                        congestion_last_changed_at = now_monotonic
                    else:
                        congestion_level = "UNKNOWN"

                elif (
                    congestion_candidate_frames >= CONGESTION_CONFIRM_FRAMES
                    and congestion_candidate != current_level
                    and now_monotonic - congestion_last_changed_at
                        >= CONGESTION_HOLD_SECONDS
                ):
                    congestion_level = congestion_candidate
                    congestion_last_changed_at = now_monotonic

                else:
                    congestion_level = current_level


                now = datetime.now().astimezone().isoformat()


            with camera_lock:
                camera_stats.update({"status": "running", "total_vehicles": sum(frame_counts.values()),
                    "cars": frame_counts["cars"], "motorcycles": frame_counts["motorcycles"],
                    "buses": frame_counts["buses"], "trucks": frame_counts["trucks"],
                    "session_total_vehicles": sum(camera_session_counts.values()),
                    "session_cars": camera_session_counts["cars"],
                    "session_motorcycles": camera_session_counts["motorcycles"],
                    "session_buses": camera_session_counts["buses"],
                    "session_trucks": camera_session_counts["trucks"], "error": None})
                previous_level = traffic_intelligence["congestion"]

                if previous_level not in ("UNKNOWN", congestion_level):
                    alert_message = (
                        f"Traffic changed from {previous_level} to {congestion_level}"
                    )

                    traffic_intelligence["recent_events"].insert(0, {
                        "timestamp": now,
                        "message": alert_message,
                        "congestion": congestion_level,
                    })

                    traffic_intelligence["recent_events"] = (
                        traffic_intelligence["recent_events"][
                            :MAX_RECENT_TRAFFIC_EVENTS
                        ]
                    )

                    traffic_intelligence["alert"] = alert_message

                traffic_intelligence.update({
                    "status": "running",
                    "occupancy_percent": round(occupancy, 2),
                    "congestion": congestion_level,
                    "last_updated": now,
                    "vehicle_counts": {
                        "cars": frame_counts["cars"],
                        "motorcycles": frame_counts["motorcycles"],
                        "buses": frame_counts["buses"],
                        "trucks": frame_counts["trucks"],
                    },
                })


            # Draw confirmed PPE violations after vehicle annotations.
            for x1, y1, x2, y2, confidence, class_name in confirmed_helmet_boxes:
                label = f"{class_name.replace('_', ' ').upper()} {confidence:.2f}"

                cv2.rectangle(
                    frame, (x1, y1), (x2, y2), (0, 0, 255), 2
                )
                cv2.putText(
                    frame,
                    label,
                    (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    2
                )

            ok, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            if ok:
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"

    except Exception as exc:
        with camera_lock:
            camera_stats.update({"status": "error", "error": str(exc)})
        print(f"Camera error: {exc}")
    finally:
        if cap is not None:
            cap.release()
        if not camera_session_saved:
            try:
                save_analysis_history({"analysis_type": "live_camera_session",
                    "video_filename": "Laptop webcam session", "status": "completed",
                    "total_vehicles": sum(camera_session_counts.values()),
                    "cars": camera_session_counts["cars"], "motorcycles": camera_session_counts["motorcycles"],
                    "buses": camera_session_counts["buses"], "trucks": camera_session_counts["trucks"],
                    "traffic_density": 0, "congestion": "UNKNOWN", "violations": 0,
                    "line_crossings": 0, "average_speed": 0,
                    "session_started_at": camera_session_started_at,
                    "session_ended_at": datetime.now().astimezone().isoformat()})
            except Exception as exc:
                print(f"Could not save camera session history: {exc}")
            camera_session_saved = True
        with camera_lock:
            camera_running = False
            camera_starting = False
            if camera_stats["status"] != "error":
                camera_stats["status"] = "stopped"
            camera_stats.update({"total_vehicles": 0, "cars": 0, "motorcycles": 0, "buses": 0, "trucks": 0})
        camera_stopped_event.set()


@app.post("/api/camera/frame")
def browser_camera_frame(file: UploadFile = File(...)):
    global browser_camera_active, browser_frame_count
    global camera_running, camera_starting
    global camera_session_started_at, camera_session_saved
    global congestion_candidate, congestion_candidate_frames
    global congestion_last_changed_at

    with browser_frame_lock:
        try:
            raw = file.file.read(5 * 1024 * 1024 + 1)
            if not raw or len(raw) > 5 * 1024 * 1024:
                raise HTTPException(status_code=413, detail="Frame is empty or exceeds 5 MB.")

            import numpy as np
            image_array = np.frombuffer(raw, dtype=np.uint8)
            frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            if frame is None:
                raise HTTPException(status_code=400, detail="Invalid image frame.")

            if not browser_camera_active:
                if camera_running and not browser_camera_active:
                    raise HTTPException(status_code=409, detail="Another camera session is running.")

                browser_camera_active = True
                browser_frame_count = 0
                camera_session_seen.clear()
                camera_track_positions.clear()
                camera_session_counts.clear()
                helmet_alert_history.clear()
                congestion_candidate = "UNKNOWN"
                congestion_candidate_frames = 0
                congestion_last_changed_at = 0.0
                camera_session_started_at = datetime.now().astimezone().isoformat()
                camera_session_saved = False
                with camera_lock:
                    camera_stats.update({
                        "status": "starting", "error": None,
                        "total_vehicles": 0, "cars": 0,
                        "motorcycles": 0, "buses": 0, "trucks": 0,
                        "session_total_vehicles": 0,
                        "session_cars": 0, "session_motorcycles": 0,
                        "session_buses": 0, "session_trucks": 0
                    })
                    traffic_intelligence["status"] = "running"
                    traffic_intelligence["congestion"] = "UNKNOWN"

            camera_running = True
            camera_starting = False
            browser_frame_count += 1

            vehicle_model = get_camera_model()
            results = vehicle_model.track(
                frame,
                persist=True,
                classes=list(VEHICLE_CLASSES.keys()),
                verbose=False,
                tracker="bytetrack.yaml"
            )

            frame_height, frame_width = frame.shape[:2]
            count_line_y = int(frame_height * COUNT_LINE_RATIO)
            counts = Counter()
            vehicle_box_area = 0.0

            result = results[0]
            boxes = result.boxes
            if boxes is not None:
                track_ids = boxes.id
                for index, box in enumerate(boxes):
                    category = VEHICLE_CLASSES.get(int(box.cls[0].item()))
                    if not category:
                        continue

                    counts[category] += 1
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    vehicle_box_area += max(0, x2 - x1) * max(0, y2 - y1)

                    if track_ids is not None:
                        track_id = int(track_ids[index].item())
                        center_y = int((y1 + y2) / 2)
                        previous_y = camera_track_positions.get(track_id)
                        crossed = (
                            previous_y is not None and
                            ((previous_y < count_line_y <= center_y) or
                             (previous_y > count_line_y >= center_y))
                        )
                        if crossed and track_id not in camera_session_seen:
                            camera_session_seen.add(track_id)
                            camera_session_counts[category] += 1
                        camera_track_positions[track_id] = center_y

            annotated = result.plot() if boxes is not None else frame.copy()
            cv2.line(annotated, (0, count_line_y),
                     (frame_width, count_line_y), (0, 255, 255), 2)
            cv2.putText(annotated, "VEHICLE COUNTING LINE",
                        (10, max(20, count_line_y - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

            confirmed_helmet_boxes = []
            if browser_frame_count % 5 == 0:
                helmet_detector = get_helmet_model()
                helmet_results = helmet_detector.predict(
                    frame, imgsz=640, conf=0.15,
                    device="cpu", verbose=False
                )
                helmet_boxes = helmet_results[0].boxes

                if helmet_boxes is not None:
                    raw_ppe = [
                        (
                            helmet_detector.names[int(box.cls[0].item())],
                            round(float(box.conf[0].item()), 3)
                        )
                        for box in helmet_boxes
                    ]
                    print("LIVE PPE RAW PREDICTIONS:", raw_ppe, flush=True)

                    # Display all PPE predictions, including compliant helmet class.
                    for ppe_box in helmet_boxes:
                        class_id = int(ppe_box.cls[0].item())
                        class_name = str(helmet_detector.names[class_id]).lower()
                        confidence = float(ppe_box.conf[0].item())
                        x1, y1, x2, y2 = map(int, ppe_box.xyxy[0].tolist())

                        is_violation = class_name.startswith("no_")
                        color = (0, 0, 255) if is_violation else (0, 200, 0)
                        label = f"{class_name.upper()} {confidence:.2f}"

                        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(
                            annotated, label, (x1, max(y1 - 8, 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
                        )

                    confirmed_helmet_boxes = update_helmet_alerts(helmet_boxes)
                else:
                    print("LIVE PPE RAW PREDICTIONS: no detections", flush=True)

            for x1, y1, x2, y2, confidence, class_name in confirmed_helmet_boxes:
                label = f"{class_name.replace('_', ' ').upper()} {confidence:.2f}"
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(annotated, label, (x1, max(y1 - 10, 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            frame_area = max(frame_height * frame_width, 1)
            occupancy = min(100.0, vehicle_box_area / frame_area * 100.0)
            if occupancy < CONGESTION_LOW_MAX:
                detected_level = "LOW"
            elif occupancy < CONGESTION_MEDIUM_MAX:
                detected_level = "MEDIUM"
            else:
                detected_level = "HIGH"

            if detected_level == congestion_candidate:
                congestion_candidate_frames += 1
            else:
                congestion_candidate = detected_level
                congestion_candidate_frames = 1

            now_mono = time.monotonic()
            current_level = traffic_intelligence["congestion"]
            if current_level == "UNKNOWN":
                congestion_level = (
                    congestion_candidate
                    if congestion_candidate_frames >= CONGESTION_CONFIRM_FRAMES
                    else "UNKNOWN"
                )
                if congestion_level != "UNKNOWN":
                    congestion_last_changed_at = now_mono
            elif (
                congestion_candidate_frames >= CONGESTION_CONFIRM_FRAMES
                and congestion_candidate != current_level
                and now_mono - congestion_last_changed_at >= CONGESTION_HOLD_SECONDS
            ):
                congestion_level = congestion_candidate
                congestion_last_changed_at = now_mono
            else:
                congestion_level = current_level

            now_iso = datetime.now().astimezone().isoformat()
            with camera_lock:
                camera_stats.update({
                    "status": "running",
                    "total_vehicles": sum(counts.values()),
                    "cars": counts["cars"],
                    "motorcycles": counts["motorcycles"],
                    "buses": counts["buses"],
                    "trucks": counts["trucks"],
                    "session_total_vehicles": sum(camera_session_counts.values()),
                    "session_cars": camera_session_counts["cars"],
                    "session_motorcycles": camera_session_counts["motorcycles"],
                    "session_buses": camera_session_counts["buses"],
                    "session_trucks": camera_session_counts["trucks"],
                    "error": None
                })
                previous_level = traffic_intelligence["congestion"]
                if previous_level not in ("UNKNOWN", congestion_level):
                    message = f"Traffic changed from {previous_level} to {congestion_level}"
                    traffic_intelligence["recent_events"].insert(0, {
                        "timestamp": now_iso,
                        "message": message,
                        "congestion": congestion_level
                    })
                    traffic_intelligence["recent_events"] = (
                        traffic_intelligence["recent_events"][:MAX_RECENT_TRAFFIC_EVENTS]
                    )
                    traffic_intelligence["alert"] = message

                traffic_intelligence.update({
                    "status": "running",
                    "occupancy_percent": round(occupancy, 2),
                    "congestion": congestion_level,
                    "last_updated": now_iso,
                    "vehicle_counts": {
                        "cars": counts["cars"],
                        "motorcycles": counts["motorcycles"],
                        "buses": counts["buses"],
                        "trucks": counts["trucks"]
                    }
                })

            ok, buffer = cv2.imencode(
                ".jpg", annotated,
                [cv2.IMWRITE_JPEG_QUALITY, 65]
            )
            if not ok:
                raise RuntimeError("Could not encode annotated frame.")

            payload = {
                "image": base64.b64encode(buffer.tobytes()).decode("ascii"),
                "stats": dict(camera_stats)
            }
            return JSONResponse(content=payload)

        except HTTPException:
            raise
        except Exception as exc:
            with camera_lock:
                camera_stats.update({"status": "error", "error": str(exc)})
            print(f"Browser camera frame error: {exc}", flush=True)
            raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/camera/live")
def live_camera():
    global camera_starting
    with camera_lock:
        if camera_running or camera_starting:
            raise HTTPException(
                status_code=409,
                detail="Camera stream is already running."
            )
        camera_starting = True
    return StreamingResponse(
        generate_camera_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )

@app.get("/api/camera/helmet-alerts")
def get_helmet_alerts():
    with camera_lock:
        return {
            "total": len(helmet_alerts),
            "alerts": list(helmet_alerts),
        }


@app.get("/api/camera/stats")
def live_camera_stats():
    with camera_lock:
        return dict(camera_stats)

@app.get("/api/camera/stats")
def live_camera_stats():
    with camera_lock:
        return dict(camera_stats)
@app.get("/api/traffic/intelligence")
def get_traffic_intelligence():
    with camera_lock:
        return dict(traffic_intelligence)
@app.get("/api/camera/status")
def live_camera_status():
    with camera_lock:
        return {
            "running": camera_running,
            "status": camera_stats["status"],
            "error": camera_stats["error"],
        }
@app.post("/api/camera/stop")
def stop_camera():
    global browser_camera_active, camera_running, camera_starting
    with browser_frame_lock:
        browser_camera_active = False
        camera_running = False
        camera_starting = False
        with camera_lock:
            if camera_stats.get("status") != "error":
                camera_stats["status"] = "stopped"
    camera_stop_event.set()
    stopped = camera_stopped_event.wait(timeout=5)
    if not stopped:
        return {
            "status": "stopping",
            "message": "Camera is stopping. Please wait."
        }
    return {
        "status": "stopped",
        "message": "Camera closed successfully."
    }
@app.get("/api/traffic/history")
def get_traffic_history():
    if not os.path.isfile(HISTORY_FILE):
        return []
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            history = json.load(f)
        if not isinstance(history, list):
            raise HTTPException(
                status_code=500,
                detail="Invalid history file format."
            )
        return history
    except json.JSONDecodeError:
        raise HTTPException(
            status_code=500,
            detail="History file contains invalid JSON."
        )
@app.delete("/api/traffic/history")
def delete_traffic_history():
    try:
        os.makedirs(os.path.dirname(HISTORY_FILE), exist_ok=True)
        with open(HISTORY_FILE, "w", encoding="utf-8") as history_file:
            json.dump([], history_file, indent=4)
        return {"status": "success", "message": "Analysis history deleted successfully."}
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"Could not delete history: {exc}")
