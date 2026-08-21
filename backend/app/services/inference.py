"""
Inference service — loads the trained ViT model once and provides prediction.
"""

import json
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from transformers import ViTForImageClassification, ViTImageProcessor


# Resolve paths relative to the project root (two levels up from this file)
_BACKEND_DIR = Path(__file__).resolve().parent.parent.parent  # backend/
_PROJECT_ROOT = _BACKEND_DIR.parent  # OcuViT/
_THRESHOLDS_PATH = _BACKEND_DIR / "thresholds.json"

# Class names in index order (must match config.json id2label)
CLASS_NAMES = [
    "Normal", "Diabetes", "Glaucoma", "Cataract",
    "AMD", "Hypertension", "Myopia", "Other",
]


class ModelService:
    """Singleton-style service that holds the loaded ViT model, processor, and thresholds."""

    def __init__(self):
        self.model: ViTForImageClassification | None = None
        self.processor: ViTImageProcessor | None = None
        self.thresholds: dict[str, float] = {}
        self.device: torch.device = torch.device("cpu")
        self.loaded = False

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------
    def load(self) -> None:
        """Load model, processor, and thresholds from disk.  Called once at startup."""
        if self.loaded:
            return

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load the HuggingFace model from the project root which contains
        # config.json, model.safetensors, and preprocessor_config.json.
        print(f"[ModelService] Loading ViT model from {_PROJECT_ROOT} …")
        self.model = ViTForImageClassification.from_pretrained(
            str(_PROJECT_ROOT),
            local_files_only=True,
            attn_implementation="eager",
        )
        self.model.to(self.device)
        self.model.eval()
        print(f"[ModelService] Model loaded on {self.device}")

        # Load the image processor (matches training preprocessing)
        self.processor = ViTImageProcessor.from_pretrained(
            str(_PROJECT_ROOT),
            local_files_only=True,
        )

        # Load per-class thresholds
        with open(_THRESHOLDS_PATH, "r") as f:
            self.thresholds = json.load(f)
        print(f"[ModelService] Thresholds loaded: {self.thresholds}")

        self.loaded = True

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------
    @torch.no_grad()
    def predict(self, image: Image.Image) -> dict:
        """
        Run inference on a single PIL image.

        Returns
        -------
        dict with keys:
            probabilities : dict[str, float]   — sigmoid probability per class
            detections    : list[dict]          — classes where prob >= threshold
            attentions    : torch.Tensor        — raw attention weights (for explainability)
        """
        if not self.loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        # Preprocess using the saved ViTImageProcessor (matches training)
        inputs = self.processor(images=image, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device)

        # Forward pass with attention outputs for explainability
        outputs = self.model(pixel_values, output_attentions=True)
        logits = outputs.logits  # shape: (1, 8)

        # Multi-label: use sigmoid (NOT softmax)
        probs = torch.sigmoid(logits).cpu().numpy()[0]

        # Build probabilities dict
        probabilities = {}
        for i, name in enumerate(CLASS_NAMES):
            probabilities[name] = round(float(probs[i]), 4)

        # Apply per-class thresholds
        detections = []
        for name in CLASS_NAMES:
            prob = probabilities[name]
            threshold = self.thresholds.get(name, 0.5)
            detected = prob >= threshold
            detections.append({
                "disease": name,
                "probability": prob,
                "threshold": round(threshold, 4),
                "detected": detected,
            })

        return {
            "probabilities": probabilities,
            "detections": detections,
            "attentions": outputs.attentions,  # tuple of tensors, one per layer
        }


# Module-level singleton
model_service = ModelService()
