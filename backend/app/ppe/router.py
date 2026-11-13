from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile

from .service import detect_ppe, MODEL_PATH

router = APIRouter(prefix="/api/ppe", tags=["Industrial PPE Detection"])

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
MAX_IMAGE_SIZE = 10 * 1024 * 1024


@router.get("/health")
def ppe_health():
    return {
        "module": "Industrial PPE Detection",
        "status": "ready" if MODEL_PATH.is_file() else "model_missing",
        "model": MODEL_PATH.name,
        "model_exists": MODEL_PATH.is_file()
    }


@router.post("/detect")
async def ppe_detect(file: UploadFile = File(...)):
    extension = Path(file.filename or "").suffix.lower()

    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="Upload JPG, JPEG, PNG or WEBP images only."
        )

    try:
        image_bytes = await file.read()

        if not image_bytes:
            raise HTTPException(status_code=400, detail="Uploaded image is empty.")

        if len(image_bytes) > MAX_IMAGE_SIZE:
            raise HTTPException(status_code=413, detail="Image must be below 10 MB.")

        result = detect_ppe(image_bytes)
        return {"success": True, **result}

    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        print("PPE detection error:", exc)
        raise HTTPException(status_code=500, detail="PPE detection failed.")
    finally:
        await file.close()
