"""Trace-color computation: sample the artwork's own colors, build gradients,
and pick contrasting frame/keyline colors.

The trace is revealed from `recon[i]`, whose parameter i/N matches
`pixel_samples[i]` (source-image pixel coords) and `samples[i]`. So a color
array of length N, one RGB per recon point, lines up index-for-index with the
drawn path. Colors are computed once in the driver and passed into the scene,
so both the full and zoom renders share identical color data (frame-synced).
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

# a pleasant default cycle for trace_source: gradient when no list is given
DEFAULT_GRADIENT = ["#FF5A5F", "#FFB400", "#4ECDC4", "#556FB5"]


def hex_to_rgb(h: str) -> np.ndarray:
    h = h.lstrip("#")
    return np.array([int(h[i:i + 2], 16) for i in (0, 2, 4)], float) / 255.0


def rgb_to_hex(rgb: np.ndarray) -> str:
    r, g, b = (np.clip(rgb, 0.0, 1.0) * 255).round().astype(int)
    return f"#{r:02X}{g:02X}{b:02X}"


def luminance(rgb: np.ndarray) -> float:
    """Perceived luminance in 0..1 (Rec. 709 weights)."""
    return float(0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2])


def contrast_hex(bg_hex: str) -> str:
    """White on a dark background, near-black on a light one."""
    return "#FFFFFF" if luminance(hex_to_rgb(bg_hex)) < 0.5 else "#101010"


def _load_bgr(path: str | Path) -> np.ndarray:
    img = cv2.imdecode(np.frombuffer(Path(path).read_bytes(), np.uint8),
                       cv2.IMREAD_UNCHANGED)
    if img is None:
        raise SystemExit(f"Could not read color image: {path}")
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img


def detect_source_light(img: np.ndarray | None) -> bool:
    """True if the artwork's border (its background) reads as light."""
    if img is None:
        return True
    gray = img if img.ndim == 2 else cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2GRAY)
    border = np.concatenate([gray[0, :], gray[-1, :], gray[:, 0], gray[:, -1]])
    return float(border.mean()) >= 128.0


def sample_colors(color_img: np.ndarray, pixel_samples: np.ndarray,
                  line_shape: tuple[int, int]) -> np.ndarray:
    """One RGB (0..1) per path point, read from `color_img` at each point's
    position. `pixel_samples` are in the *line* image's pixel coords; we map
    them to fractions and index into the color image, so a separate colored
    original at a different resolution still lines up (same framing assumed)."""
    ch, cw = color_img.shape[:2]
    lh, lw = line_shape
    fx = np.clip(pixel_samples.real / max(lw, 1), 0.0, 1.0)
    fy = np.clip(pixel_samples.imag / max(lh, 1), 0.0, 1.0)
    px = np.clip((fx * (cw - 1)).round().astype(int), 0, cw - 1)
    py = np.clip((fy * (ch - 1)).round().astype(int), 0, ch - 1)
    bgr = color_img[py, px, :3].astype(np.float64) / 255.0
    return bgr[:, ::-1].copy()  # BGR -> RGB


def gradient_ramp(n: int, hexes: list[str], cycles: float) -> np.ndarray:
    """Smoothly cycle through `hexes` along the path, `cycles` times."""
    cols = np.array([hex_to_rgb(h) for h in hexes])
    m = len(cols)
    if m == 1:
        return np.repeat(cols, n, axis=0)
    t = (np.arange(n) / max(n, 1)) * cycles * m
    t = np.mod(t, m)
    i0 = np.floor(t).astype(int) % m
    i1 = (i0 + 1) % m
    f = (t - np.floor(t))[:, None]
    return cols[i0] * (1.0 - f) + cols[i1] * f


def build_trace_colors(s, data, line_image_path: str | Path) -> np.ndarray | None:
    """(N, 3) RGB array for the trace, or None for a single themed color.

    trace_source:
      theme    -> None (scene uses the single theme/trace color)
      gradient -> cycle through gradient_colors (or DEFAULT_GRADIENT)
      original -> sample color_image (else source_image, else the line art),
                  then blend toward the theme color by (1 - color_strength).
    """
    if s.trace_source == "theme":
        return None
    n = len(data.samples)
    theme = hex_to_rgb(s.trace_hex)

    if s.trace_source == "gradient":
        hexes = s.gradient_colors or DEFAULT_GRADIENT
        return gradient_ramp(n, hexes, s.gradient_cycles)

    # original: sample the artwork's own colors
    src = s.color_image or s.source_image or str(line_image_path)
    color_img = _load_bgr(src)
    if data.source_image is not None:
        line_shape = data.source_image.shape[:2]
    else:  # SVG line input: fall back to the path's own pixel extent
        line_shape = (int(data.pixel_samples.imag.max()) + 1,
                      int(data.pixel_samples.real.max()) + 1)
    cols = sample_colors(color_img, data.pixel_samples, line_shape)
    return (1.0 - s.color_strength) * theme + s.color_strength * cols
