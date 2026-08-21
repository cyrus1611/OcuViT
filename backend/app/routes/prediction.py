"""
Prediction API route — POST /api/predict
"""

import uuid
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, HTTPException

from app.services.inference import model_service
from app.services.preprocessing import validate_and_load_image
from app.services.explainability import generate_attention_map
from app.services.advisory import generate_advisory, DISCLAIMER
from app.services.laterality import detect_eye_laterality


router = APIRouter()

# Directory for temporarily storing uploaded images
_BACKEND_DIR = Path(__file__).resolve().parent.parent.parent
UPLOADS_DIR = _BACKEND_DIR / "uploads"
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)


@router.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Accept a fundus image, run ViT inference, and return results.
    """
    # 1. Validate and load image
    image = await validate_and_load_image(file)

    # 2. Save upload temporarily (with safe filename)
    safe_name = f"upload_{uuid.uuid4().hex[:12]}.png"
    upload_path = UPLOADS_DIR / safe_name
    image.save(upload_path)

    # 3. Detect eye laterality (Left Eye vs Right Eye)
    laterality = detect_eye_laterality(image)

    # 4. Run inference
    try:
        result = model_service.predict(image)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}",
        )

    # 5. Generate attention map
    try:
        heatmap_url = generate_attention_map(image, result["attentions"])
    except Exception as e:
        print(f"[WARNING] Attention map generation failed: {e}")
        heatmap_url = None

    # 6. Generate advisory
    advisory = generate_advisory(result["detections"])

    # 7. Build response
    return {
        "eye_laterality": laterality,
        "probabilities": result["probabilities"],
        "detections": result["detections"],
        "advisory": advisory,
        "heatmap_url": heatmap_url,
        "disclaimer": DISCLAIMER,
    }
