#!/usr/bin/env python
"""Fourier epicycle animation renderer (3Blue1Brown style).

Usage (inside the `manim` conda env):
    python epicycles.py assets/jolly_roger.png --duration 180 --detail high
    python epicycles.py assets/luffy.png --config configs/luffy.yaml --view zoom --preview

Each run produces <name>_full.mp4 and/or <name>_zoom.mp4 in output/<name>/,
rendered from one deterministic timeline so the two stay frame-synced.
"""
from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from epicycle_lib.settings import Settings


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("image", help="input PNG (near-white bg / darker lines) or SVG")
    p.add_argument("--config", help="per-image YAML; CLI flags override it")
    p.add_argument("--duration", type=float, help="seconds for one full draw")
    p.add_argument("--detail", choices=["low", "medium", "high"],
                   help="vector count preset: 50 / 150 / 400")
    p.add_argument("--vectors", type=int, help="explicit vector count")
    p.add_argument("--samples", type=int, help="path sample count (default auto)")
    p.add_argument("--mode", choices=["full", "outline"])
    p.add_argument("--extract", choices=["auto", "otsu", "alpha", "canny"])
    p.add_argument("--min-area", type=float, dest="min_area")
    p.add_argument("--canny-low", type=int, dest="canny_low")
    p.add_argument("--canny-high", type=int, dest="canny_high")
    p.add_argument("--smooth", type=float, help="contour simplification, 0..1")
    p.add_argument("--fill", type=float, help="height as fraction of frame")
    p.add_argument("--hold", type=float)
    p.add_argument("--end-hold", type=float, dest="end_hold")
    p.add_argument("--no-ghost", action="store_true")
    p.add_argument("--no-label", action="store_true")
    p.add_argument("--view", choices=["full", "zoom", "both"])
    p.add_argument("--zoom-factor", type=float, dest="zoom_factor")
    p.add_argument("--cam-lag", type=float, dest="cam_lag")
    p.add_argument("--fps", type=int)
    p.add_argument("--preview", action="store_true", help="854x480 @ 15fps")
    p.add_argument("--yes", action="store_true", help="skip the inspect prompt")
    p.add_argument("--via-svg", action="store_true", dest="via_svg",
                   help="route the bitmap through vtracer (quality A/B test)")
    p.add_argument("--double-lines", action="store_true", dest="double_lines",
                   help="draw both edges of stroked lines instead of one")
    p.add_argument("-o", "--output", help="output directory")
    return p.parse_args()


def build_settings(args: argparse.Namespace) -> Settings:
    s = Settings()
    if args.config:
        s.apply_yaml(args.config)
    scalar_flags = ["duration", "detail", "vectors", "samples", "mode", "extract",
                    "min_area", "canny_low", "canny_high", "smooth", "fill",
                    "hold", "end_hold", "view", "zoom_factor", "cam_lag",
                    "fps", "output"]
    for name in scalar_flags:
        value = getattr(args, name)
        if value is not None:
            setattr(s, name, value)
    if args.no_ghost:
        s.ghost = False
    if args.no_label:
        s.label = False
    for flag in ("preview", "yes", "via_svg", "double_lines"):
        if getattr(args, flag):
            setattr(s, flag, True)
    s.validate()
    return s


def render_view(view: str, coeffs, freqs, recon, jump_mask, s: Settings,
                outdir: Path, name: str) -> Path:
    from manim import tempconfig

    from epicycle_lib.scene import BG_COLOR, EpicycleScene

    px_w, px_h, fps = (854, 480, 15) if s.preview else (1920, 1080, s.fps)
    with tempconfig({
        "pixel_width": px_w, "pixel_height": px_h, "frame_rate": fps,
        "background_color": BG_COLOR,
        "disable_caching": True, "verbosity": "ERROR",
        "media_dir": str(outdir / "media"),
        "output_file": f"{name}_{view}",
    }):
        scene = EpicycleScene(coeffs, freqs, recon, jump_mask, s, view)
        scene.render()
        produced = Path(scene.renderer.file_writer.movie_file_path)
    final = outdir / f"{name}_{view}.mp4"
    if produced.resolve() != final.resolve():
        shutil.move(str(produced), str(final))
    # manim leaves per-animation fragments under media/; only the moved file
    # is a real video, so drop the scratch tree to avoid confusion
    shutil.rmtree(outdir / "media", ignore_errors=True)
    return final


def main() -> None:
    args = parse_args()
    s = build_settings(args)

    image_path = Path(args.image)
    if not image_path.exists():
        raise SystemExit(f"No such file: {image_path}")
    name = image_path.stem
    outdir = Path(s.output) if s.output else Path("output") / name
    outdir.mkdir(parents=True, exist_ok=True)

    from epicycle_lib.fourier import fourier_series
    from epicycle_lib.image_to_path import extract_path
    from epicycle_lib.inspect import confirm_or_abort, write_inspect

    t0 = time.perf_counter()
    data = extract_path(image_path, s, cache_dir=outdir)
    coeffs, freqs, recon = fourier_series(data.samples, s.n_vectors)
    print(f"Extracted {data.n_contours} contours -> {len(data.samples)} samples, "
          f"{len(coeffs)} vectors  ({time.perf_counter() - t0:.1f}s)")

    inspect_png = write_inspect(data, recon, len(coeffs), outdir / "inspect.png")
    confirm_or_abort(inspect_png, s.yes)

    views = ["full", "zoom"] if s.view == "both" else [s.view]
    for view in views:
        t0 = time.perf_counter()
        print(f"Rendering {view} view...")
        out = render_view(view, coeffs, freqs, recon, data.jump_mask, s,
                          outdir, name)
        print(f"  -> {out}  ({time.perf_counter() - t0:.1f}s)")


if __name__ == "__main__":
    main()
