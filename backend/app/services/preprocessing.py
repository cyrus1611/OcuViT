"""
Image preprocessing and validation service.
"""

from PIL import Image
from io import BytesIO
from fastapi import UploadFile, HTTPException

# Allowed MIME types and extensions
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}
ALLOWED_MIME_TYPES = {"image/jpeg", "image/png"}
MAX_FILE_SIZE_MB = 10
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024


async def validate_and_load_image(file: UploadFile) -> Image.Image:
    """
    Validate an uploaded file and return a PIL Image in RGB mode.

    Raises HTTPException on validation failure.
    """
    # --- Check content type ---
    if file.content_type not in ALLOWED_MIME_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type '{file.content_type}'. Please upload a JPG, JPEG, or PNG image.",
        )

    # --- Check file extension ---
    filename = file.filename or ""
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file extension '.{ext}'. Supported formats: JPG, JPEG, PNG.",
        )

    # --- Read bytes and check size ---
    contents = await file.read()
    if len(contents) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=400,
            detail=f"File too large ({len(contents) / 1024 / 1024:.1f} MB). Maximum size is {MAX_FILE_SIZE_MB} MB.",
        )

    if len(contents) == 0:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is empty.",
        )

    # --- Try to open as image ---
    try:
        image = Image.open(BytesIO(contents))
        image.verify()  # Verify it's not corrupted
        # Re-open after verify (verify() can leave the image in an unusable state)
        image = Image.open(BytesIO(contents))
        image = image.convert("RGB")
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file appears to be corrupted or is not a valid image.",
        )

    return image
