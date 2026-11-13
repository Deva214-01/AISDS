from fastapi import APIRouter, UploadFile, File, HTTPException
from .service import detect_fire_smoke, MODEL_PATH

router = APIRouter(prefix="/api/fire-smoke", tags=["Fire & Smoke Detection"])

@router.get("/health")
def fire_smoke_health():
    return {
        "status": "ready" if MODEL_PATH.is_file() else "model_missing",
        "module": "Fire & Smoke Detection",
        "model": MODEL_PATH.name
    }

@router.post("/detect")
async def fire_smoke_detect(file: UploadFile = File(...)):
    allowed = {".jpg", ".jpeg", ".png", ".webp"}
    filename = (file.filename or "").lower()

    if not any(filename.endswith(ext) for ext in allowed):
        raise HTTPException(status_code=400, detail="Upload JPG, JPEG, PNG or WEBP.")

    image_bytes = await file.read()
    await file.close()

    if not image_bytes:
        raise HTTPException(status_code=400, detail="Uploaded image is empty.")

    if len(image_bytes) > 10 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Image must be 10 MB or smaller.")

    try:
        return detect_fire_smoke(image_bytes)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Fire/Smoke detection failed: {exc}")
