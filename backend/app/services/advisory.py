"""
Advisory service — generates educational advisory text based on detected diseases.
"""

# Disease-specific educational advisory messages.
# IMPORTANT: Never claim diagnosis.  Always use "The model detected a pattern…"
ADVISORY_MESSAGES: dict[str, str] = {
    "Normal": (
        "No strong disease pattern was detected by the model. "
        "Routine professional eye examinations are still recommended."
    ),
    "Diabetes": (
        "The model detected a pattern associated with diabetic eye disease. "
        "Professional ophthalmic evaluation is recommended."
    ),
    "Glaucoma": (
        "The model detected a pattern associated with glaucoma. "
        "Further evaluation by an eye-care professional is recommended."
    ),
    "Cataract": (
        "The model detected a pattern associated with cataract. "
        "Professional eye evaluation is recommended."
    ),
    "AMD": (
        "The model detected a pattern associated with age-related macular degeneration. "
        "Professional ophthalmic evaluation is recommended."
    ),
    "Hypertension": (
        "The model detected a pattern associated with hypertensive retinal changes. "
        "Professional evaluation is recommended."
    ),
    "Myopia": (
        "The model detected a pattern associated with myopia. "
        "Professional vision assessment is recommended."
    ),
    "Other": (
        "The model detected an ophthalmic pattern outside the primary categories. "
        "Professional evaluation is recommended."
    ),
}

DISCLAIMER = (
    "This AI tool is intended for educational and research screening purposes only. "
    "It does not provide a medical diagnosis and does not replace evaluation by a "
    "qualified healthcare professional. "
    "Model probabilities should not be interpreted as medical certainty."
)


def generate_advisory(detections: list[dict]) -> str:
    """
    Build a combined advisory string from the list of detection results.

    Parameters
    ----------
    detections : list[dict]
        Each dict has keys: disease, probability, threshold, detected.

    Returns
    -------
    str : combined advisory text for all detected conditions.
    """
    detected_diseases = [d["disease"] for d in detections if d["detected"]]

    if not detected_diseases:
        return ADVISORY_MESSAGES["Normal"]

    # If "Normal" is detected alongside other diseases, skip the Normal message
    # to avoid confusing output.
    disease_list = [d for d in detected_diseases if d != "Normal"]
    if not disease_list:
        # Only "Normal" was detected
        return ADVISORY_MESSAGES["Normal"]

    advisories = [ADVISORY_MESSAGES[d] for d in disease_list if d in ADVISORY_MESSAGES]
    return " ".join(advisories)
