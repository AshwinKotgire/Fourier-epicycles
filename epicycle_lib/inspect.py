"""Contour-preview PNG + human confirmation gate, run before every render."""
from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .image_to_path import PathData

BG = (22, 22, 22)
NORMAL = (80, 200, 90)
JUMP = (230, 70, 60)
TRACE = (255, 220, 60)
PANEL_H = 720
TITLE_H = 36


def _draw_polyline(draw: ImageDraw.ImageDraw, xs: np.ndarray, ys: np.ndarray,
                   mask: np.ndarray, normal_color, jump_color, width: int) -> None:
    for i in range(len(xs) - 1):
        color = jump_color if (mask[i] or mask[i + 1]) else normal_color
        draw.line([(xs[i], ys[i]), (xs[i + 1], ys[i + 1])], fill=color, width=width)


def _left_panel(data: PathData) -> Image.Image:
    pix = data.pixel_samples
    if data.source_image is not None:
        img = data.source_image
        rgb = img[:, :, 2::-1] if img.ndim == 3 else np.stack([img] * 3, axis=-1)
        base = Image.fromarray(rgb.astype(np.uint8)).convert("RGB")
        xs, ys = pix.real, pix.imag
    else:  # SVG input: white canvas fitted to the path bounds
        margin = 20
        w = int(pix.real.max() - pix.real.min()) + 2 * margin
        h = int(pix.imag.max() - pix.imag.min()) + 2 * margin
        base = Image.new("RGB", (max(w, 64), max(h, 64)), (250, 250, 250))
        xs = pix.real - pix.real.min() + margin
        ys = pix.imag - pix.imag.min() + margin
    scale = PANEL_H / base.height
    base = base.resize((max(1, int(base.width * scale)), PANEL_H))
    draw = ImageDraw.Draw(base)
    _draw_polyline(draw, xs * scale, ys * scale, data.jump_mask, NORMAL, JUMP, 2)
    return base


def _right_panel(recon: np.ndarray, jump_mask: np.ndarray,
                 aspect: float) -> Image.Image:
    w = max(64, int(PANEL_H * aspect))
    img = Image.new("RGB", (w, PANEL_H), BG)
    draw = ImageDraw.Draw(img)
    span_x = max(recon.real.max() - recon.real.min(), 1e-9)
    span_y = max(recon.imag.max() - recon.imag.min(), 1e-9)
    scale = 0.88 * min(w / span_x, PANEL_H / span_y)
    cx, cy = recon.real.mean(), recon.imag.mean()
    xs = (recon.real - cx) * scale + w / 2
    ys = (cy - recon.imag) * scale + PANEL_H / 2  # manim y up -> canvas y down
    _draw_polyline(draw, xs, ys, jump_mask, TRACE, (110, 90, 40), 2)
    return img


def write_inspect(data: PathData, recon: np.ndarray, n_vectors: int,
                  out_path: Path) -> Path:
    left = _left_panel(data)
    span_x = max(recon.real.max() - recon.real.min(), 1e-9)
    span_y = max(recon.imag.max() - recon.imag.min(), 1e-9)
    right = _right_panel(recon, data.jump_mask, span_x / span_y)

    gap = 16
    sheet = Image.new("RGB", (left.width + right.width + 3 * gap,
                              PANEL_H + TITLE_H + 2 * gap), BG)
    sheet.paste(left, (gap, TITLE_H + gap))
    sheet.paste(right, (left.width + 2 * gap, TITLE_H + gap))
    draw = ImageDraw.Draw(sheet)
    draw.text((gap, 12), f"extracted chain - {data.n_contours} contours "
                         f"(red = jumps)", fill=(200, 200, 200))
    draw.text((left.width + 2 * gap, 12),
              f"fourier reconstruction - {n_vectors} vectors", fill=(200, 200, 200))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)
    return out_path


def confirm_or_abort(png_path: Path, skip_prompt: bool) -> None:
    try:
        subprocess.Popen(["xdg-open", str(png_path)],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except OSError:
        pass
    print(f"Inspection preview: {png_path}")
    if skip_prompt:
        return
    answer = input("Proceed with render? [y/N] ").strip().lower()
    if answer not in ("y", "yes"):
        raise SystemExit("Aborted at inspection gate.")
