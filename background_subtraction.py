import cv2
import numpy as np
from pathlib import Path
from typing import Optional

class BackgroundSubtraction:
    """Class to handle background subtraction processing"""
    
    def __init__(self, lower_threshold: int = 10, upper_threshold: int = 255):
        self.lower_threshold = lower_threshold
        self.upper_threshold = upper_threshold

    def process(self, background: np.ndarray, target: np.ndarray):
        """
        Subtract target from background, compute heatmap, and threshold.
        
        Returns:
            heatmap_color: The colorized heatmap of differences.
            mask: The binary mask after thresholding and morphological cleanup.
        """
        # Ensure same size
        if background.shape != target.shape:
            target = cv2.resize(target, (background.shape[1], background.shape[0]))
        
        # Calculate pixel-wise difference
        diff_px = target.astype(np.float32) - background.astype(np.float32)
        dist_px = np.sqrt(np.sum(diff_px**2, axis=2))
        
        # Normalize to 0-255
        norm_dist_px = cv2.normalize(dist_px, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        # Create heatmap
        heatmap_color = cv2.applyColorMap(norm_dist_px, cv2.COLORMAP_JET)
        
        # Apply threshold to create mask
        bg_mask = cv2.inRange(norm_dist_px, self.lower_threshold, self.upper_threshold)
        
        # Morphological operations to clean up mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        bg_mask = cv2.morphologyEx(bg_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        bg_mask = cv2.morphologyEx(bg_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        
        return heatmap_color, bg_mask


def get_background_from_config(config: dict, mode: str, base_dir: Path) -> Optional[np.ndarray]:
    """Load a mode-specific background image from config.

    Args:
        config: Configuration dictionary (parsed YAML).
        mode: Current mode; expects "realsense" or "top_camera".
        base_dir: Base directory to resolve relative paths.

    Returns:
        Background image as numpy array, or None if not found/loaded.
    """
    bg_config = config.get('background_image', {}) if isinstance(config, dict) else {}

    path = None
    if mode == "realsense":
        path = bg_config.get('realsense_path')
    elif mode == "top_camera":
        path = bg_config.get('http_path')
    else:
        path = bg_config.get('path')

    if not path:
        return None

    try:
        resolved = Path(path)
        if not resolved.is_absolute():
            resolved = base_dir / path

        if not resolved.exists():
            return None

        img = cv2.imread(str(resolved))
        return img
    except Exception:
        return None
