"""
Eye laterality detection service — determines whether a fundus image is
Left Eye (OS - Oculi Sinister) or Right Eye (OD - Oculi Dexter).

Anatomical Rule:
- Optic Disc is located on the NASAL side (towards the nose).
- Macula is located on the TEMPORAL side (towards the ear).
- In a Left Eye (OS) fundus photograph, the Optic Disc is on the LEFT side of the image.
- In a Right Eye (OD) fundus photograph, the Optic Disc is on the RIGHT side of the image.
"""

import numpy as np
from PIL import Image


def detect_eye_laterality(image: Image.Image) -> dict:
    """
    Determine whether the fundus image belongs to the Left Eye (OS) or Right Eye (OD)
    based on the anatomical location of the optic disc (the brightest landmark).

    Parameters
    ----------
    image : PIL.Image
        The fundus image (RGB).

    Returns
    -------
    dict
        {
            "eye": "Left Eye (OS)" | "Right Eye (OD)",
            "code": "OS" | "OD",
            "disc_side": "left" | "right",
            "confidence": float (0.50 - 0.99),
            "description": str
        }
    """
    # Resize to standard analysis size for speed and robustness
    analysis_size = (300, 300)
    resized = image.resize(analysis_size).convert("RGB")
    arr = np.array(resized)

    # Fundus images are typically circular with dark background borders.
    # Exclude dark background pixels (luminance < 25)
    r = arr[:, :, 0].astype(np.float32)
    g = arr[:, :, 1].astype(np.float32)
    b = arr[:, :, 2].astype(np.float32)

    # Optic disc has high green & red intensity
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    foreground_mask = luminance > 30

    if not np.any(foreground_mask):
        # Fallback if image is entirely dark
        return {
            "eye": "Undetermined",
            "code": "Unknown",
            "disc_side": "unknown",
            "confidence": 0.50,
            "description": "Unable to clearly distinguish optic disc landmark.",
        }

    # Find the top 1.5% brightest pixels in the foreground
    fg_values = luminance[foreground_mask]
    threshold_val = np.percentile(fg_values, 98.5)
    bright_mask = (luminance >= threshold_val) & foreground_mask

    # Calculate horizontal centroid (x-coordinate) of bright optic disc candidates
    y_coords, x_coords = np.where(bright_mask)

    if len(x_coords) == 0:
        return {
            "eye": "Undetermined",
            "code": "Unknown",
            "disc_side": "unknown",
            "confidence": 0.50,
            "description": "Unable to localize optic disc center.",
        }

    center_x = analysis_size[0] / 2.0
    mean_disc_x = float(np.mean(x_coords))

    # Calculate distance from center normalized to [0, 1]
    offset_from_center = (mean_disc_x - center_x) / center_x
    confidence = min(0.99, max(0.65, 0.50 + abs(offset_from_center) * 0.70))

    if mean_disc_x < center_x:
        return {
            "eye": "Left Eye (OS)",
            "code": "OS",
            "disc_side": "left",
            "confidence": round(confidence, 2),
            "description": "Optic disc located on nasal (left) side, indicating a Left Eye (Oculus Sinister).",
        }
    else:
        return {
            "eye": "Right Eye (OD)",
            "code": "OD",
            "disc_side": "right",
            "confidence": round(confidence, 2),
            "description": "Optic disc located on nasal (right) side, indicating a Right Eye (Oculus Dexter).",
        }
