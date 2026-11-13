import base64

from fastapi import APIRouter, File, HTTPException, UploadFile

from .service import detect_license_plates


router = APIRouter(
    prefix="/api/number-plate",
    tags=["Number Plate Detection"]
)


@router.get("/health")
def number_plate_health():
    return {
        "status": "ready",
        "module": "Number Plate Detection"
    }


@router.post("/detect")
async def detect_number_plate(file: UploadFile = File(...)):

    allowed_extensions = {
        ".jpg", ".jpeg", ".png", ".webp"
    }

    filename = file.filename or ""
    extension = filename.rsplit(".", 1)[-1].lower()

    if "." not in filename or f".{extension}" not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail="Upload JPG, JPEG, PNG or WEBP images only."
        )

    try:
        image_bytes = await file.read()

        if not image_bytes:
            raise HTTPException(
                status_code=400,
                detail="Uploaded image is empty."
            )

        if len(image_bytes) > 10 * 1024 * 1024:
            raise HTTPException(
                status_code=413,
                detail="Image size must be below 10 MB."
            )

        result = detect_license_plates(image_bytes)

        return {
            "success": True,
            "total_plates": result["total_plates"],
            "detections": result["detections"],
            "annotated_image": base64.b64encode(
                result["annotated_image"]
            ).decode("utf-8")
        }

    except HTTPException:
        raise

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc)
        )

    except Exception as exc:
        print("Number plate detection error:", exc)

        raise HTTPException(
            status_code=500,
            detail="Number plate detection failed."
        )

    finally:
        await file.close()
