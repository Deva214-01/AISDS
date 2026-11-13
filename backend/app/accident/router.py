
import shutil
import threading
import uuid
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, File, HTTPException, UploadFile
from fastapi.responses import FileResponse

from .pipeline import run_accident_pipeline


router = APIRouter(
    prefix="/api/accident",
    tags=["Accident Detection"]
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]

UPLOAD_DIR = PROJECT_ROOT / "uploads" / "accident"
OUTPUT_DIR = PROJECT_ROOT / "output" / "accident"
EXPERIMENT_DIR = PROJECT_ROOT / "experiments" / "accident_test"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}

jobs = {}
jobs_lock = threading.Lock()


@router.get("/health")
def accident_health():
    return {
        "module": "Accident Detection Pipeline",
        "status": "ready"
    }


def process_accident_job(job_id: str, video_path: Path, output_dir: Path):
    """Run accident analysis in a background task."""

    with jobs_lock:
        jobs[job_id]["status"] = "processing"
        jobs[job_id]["message"] = "YOLO vehicle detection and tracking started."

    try:
        result = run_accident_pipeline(video_path, output_dir)

        with jobs_lock:
            jobs[job_id].update({
                "status": "completed",
                "message": "Accident analysis completed.",
                "report": result["report"],
                "report_path": result["report_path"],
                "video_path": result["annotated_video_path"],
                "tracks_path": result["tracks_path"],
                "pairs_path": result["pairs_path"],
                "timeline_path": result["timeline_path"],
            })

    except Exception as exc:
        print(f"ACCIDENT JOB {job_id} FAILED: {exc}")

        with jobs_lock:
            jobs[job_id].update({
                "status": "failed",
                "message": str(exc)
            })


@router.post("/upload")
async def upload_accident_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    """Upload a video and start accident analysis."""

    original_name = Path(file.filename or "").name
    extension = Path(original_name).suffix.lower()

    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="Upload MP4, AVI, MOV or MKV video."
        )

    job_id = uuid.uuid4().hex

    video_path = UPLOAD_DIR / f"{job_id}{extension}"
    output_dir = OUTPUT_DIR / job_id

    try:
        with video_path.open("wb") as destination:
            shutil.copyfileobj(file.file, destination)
    except OSError as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Could not save uploaded video: {exc}"
        )
    finally:
        await file.close()

    with jobs_lock:
        jobs[job_id] = {
            "job_id": job_id,
            "filename": original_name,
            "status": "queued",
            "message": "Video uploaded. Waiting for processing.",
            "report": None,
        }

    background_tasks.add_task(
        process_accident_job,
        job_id,
        video_path,
        output_dir
    )

    return {
        "job_id": job_id,
        "filename": original_name,
        "status": "queued",
        "message": "Video uploaded successfully. Accident analysis started."
    }


@router.get("/jobs/{job_id}")
def get_accident_job(job_id: str):
    """Get processing status and report when available."""

    with jobs_lock:
        job = jobs.get(job_id)

        if job is None:
            raise HTTPException(
                status_code=404,
                detail="Accident analysis job not found."
            )

        return dict(job)


@router.get("/jobs/{job_id}/report")
def get_accident_job_report(job_id: str):
    with jobs_lock:
        job = jobs.get(job_id)

        if job is None:
            raise HTTPException(status_code=404, detail="Job not found.")

        if job["status"] != "completed":
            raise HTTPException(
                status_code=409,
                detail="Accident analysis is not completed yet."
            )

        report_path = Path(job["report_path"])

    if not report_path.is_file():
        raise HTTPException(status_code=404, detail="Report file not found.")

    return FileResponse(
        report_path,
        media_type="application/json",
        filename="accident_report.json"
    )


@router.get("/jobs/{job_id}/video")
def get_accident_annotated_video(job_id: str):
    with jobs_lock:
        job = jobs.get(job_id)

        if job is None:
            raise HTTPException(status_code=404, detail="Job not found.")

        if job["status"] != "completed":
            raise HTTPException(
                status_code=409,
                detail="Video processing is not completed."
            )

        video_path = job.get("video_path")

    if not video_path or not Path(video_path).is_file():
        raise HTTPException(
            status_code=404,
            detail="Annotated video not available."
        )

    return FileResponse(
        video_path,
        media_type="video/mp4",
        filename="accident_annotated.mp4"
    )


# Retain compatibility with the two existing experiment reports.
@router.get("/results/{video_name}")
def get_accident_result(video_name: str):
    allowed_videos = {"headon-nn", "tbone"}

    if video_name not in allowed_videos:
        raise HTTPException(
            status_code=400,
            detail="Unknown video. Use headon-nn or tbone."
        )

    result_file = EXPERIMENT_DIR / f"{video_name}_decision_v2.json"

    if not result_file.is_file():
        raise HTTPException(
            status_code=404,
            detail="Decision report not found."
        )

    import json

    try:
        with result_file.open("r", encoding="utf-8") as file:
            return json.load(file)
    except (OSError, json.JSONDecodeError):
        raise HTTPException(
            status_code=500,
            detail="Could not read accident decision report."
        )
