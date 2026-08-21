"""
Explainability service — generates ViT attention map overlays.
"""

import uuid
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.cm as cm
from pathlib import Path
from PIL import Image


# Output directory for attention maps
_BACKEND_DIR = Path(__file__).resolve().parent.parent.parent
ATTENTION_MAPS_DIR = _BACKEND_DIR / "results" / "attention_maps"
ATTENTION_MAPS_DIR.mkdir(parents=True, exist_ok=True)


def generate_attention_map(
    image: Image.Image,
    attentions: tuple,
) -> str:
    """
    Generate a ViT attention heatmap overlay on the original image.

    Parameters
    ----------
    image : PIL.Image
        The original input image (RGB).
    attentions : tuple of torch.Tensor
        Attention weights from all ViT layers (output_attentions=True).
        Each tensor shape: (batch, num_heads, seq_len, seq_len)

    Returns
    -------
    str : relative URL path to the saved heatmap image.
    """
    # Use the last layer's attention
    # Shape: (1, num_heads, seq_len, seq_len)
    last_layer_attn = attentions[-1]

    # Average across all attention heads → (1, seq_len, seq_len)
    attn_avg = last_layer_attn.mean(dim=1)

    # Extract CLS token (index 0) attention to all patch tokens
    # Shape: (1, seq_len) → take [0, 1:] to skip CLS-to-CLS
    cls_attn = attn_avg[0, 0, 1:].detach().cpu().numpy()

    # ViT-base-patch16-224: 224/16 = 14 patches per side → 196 patch tokens
    num_patches_side = 14
    assert cls_attn.shape[0] == num_patches_side * num_patches_side, (
        f"Expected {num_patches_side**2} patch tokens, got {cls_attn.shape[0]}"
    )

    # Reshape to 2D grid
    attn_map = cls_attn.reshape(num_patches_side, num_patches_side)

    # Normalize to [0, 1]
    attn_min = attn_map.min()
    attn_max = attn_map.max()
    if attn_max - attn_min > 1e-8:
        attn_map = (attn_map - attn_min) / (attn_max - attn_min)
    else:
        attn_map = np.zeros_like(attn_map)

    # Resize attention map to original image size
    original_size = image.size  # (width, height)
    attn_resized = np.array(
        Image.fromarray((attn_map * 255).astype(np.uint8)).resize(
            original_size, resample=Image.BILINEAR
        )
    ) / 255.0

    # Apply colormap (jet) to create RGBA heatmap
    heatmap_rgba = cm.jet(attn_resized)  # shape: (H, W, 4)
    heatmap_rgb = (heatmap_rgba[:, :, :3] * 255).astype(np.uint8)

    # Create overlay: blend original image with heatmap
    original_array = np.array(image.resize(original_size))
    alpha = 0.5
    overlay = (original_array * (1 - alpha) + heatmap_rgb * alpha).astype(np.uint8)

    # Save
    filename = f"attention_{uuid.uuid4().hex[:12]}.png"
    output_path = ATTENTION_MAPS_DIR / filename
    Image.fromarray(overlay).save(output_path)

    # Return URL path relative to static mount
    return f"/results/attention_maps/{filename}"
