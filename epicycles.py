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
import json
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from epicycle_lib.settings import THEMES, Settings


def ffmpeg_bin() -> str:
    env_ffmpeg = Path(sys.executable).parent / "ffmpeg"
    return str(env_ffmpeg) if env_ffmpeg.exists() else "ffmpeg"


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
    p.add_argument("--end-card", type=float, dest="end_card",
                   help=">0: scale the finished drawing to fill the frame "
                        "and hold this many seconds (outro shot)")
    p.add_argument("--start-t", type=float, dest="start_t",
                   help="render only from this timeline fraction (0..1)")
    p.add_argument("--end-t", type=float, dest="end_t",
                   help="render only up to this timeline fraction (0..1)")
    p.add_argument("--theme", choices=list(THEMES),
                   help="trace-color preset")
    p.add_argument("--trace-color", dest="trace_color",
                   help="hex trace color, overrides --theme (e.g. '#FF7043')")
    p.add_argument("--ghost-mode", choices=["static", "fadein", "late"],
                   dest="ghost_mode",
                   help="static = always visible; fadein = grows with "
                        "progress; late = appears in the last third")
    p.add_argument("--vignette", type=float,
                   help="0..1 radial background gradient strength")
    p.add_argument("--trace-fade", type=float, dest="trace_fade",
                   help="0 = trace stays bright; >0 dims the settled path "
                        "so a glowing head follows the pen")
    p.add_argument("--no-ghost", action="store_true")
    p.add_argument("--no-label", action="store_true")
    p.add_argument("--view", choices=["full", "zoom", "both", "pip"],
                   help="pip = zoom view inset into a corner of the full view")
    p.add_argument("--zoom-factor", type=float, dest="zoom_factor")
    p.add_argument("--cam-lag", type=float, dest="cam_lag")
    p.add_argument("--fps", type=int)
    p.add_argument("--parallel", action="store_true",
                   help="render full + zoom views as two parallel processes")
    p.add_argument("--pip-inset-label", action="store_true",
                   dest="pip_inset_label",
                   help="keep the corner label inside the pip inset "
                        "(suppressed by default)")
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
                    "hold", "end_hold", "end_card", "start_t", "end_t",
                    "theme", "trace_color", "ghost_mode", "vignette",
                    "trace_fade", "view", "zoom_factor", "cam_lag",
                    "fps", "output"]
    for name in scalar_flags:
        value = getattr(args, name)
        if value is not None:
            setattr(s, name, value)
    if args.no_ghost:
        s.ghost = False
    if args.no_label:
        s.label = False
    for flag in ("preview", "yes", "via_svg", "double_lines", "parallel",
                 "pip_inset_label"):
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
        "media_dir": str(outdir / f"media_{view}"),
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
    shutil.rmtree(outdir / f"media_{view}", ignore_errors=True)
    return final


def composite_pip(full: Path, zoom: Path, out: Path) -> None:
    """Overlay the zoom view (at 30% width, thin border) onto the full view."""
    ffmpeg = ffmpeg_bin()
    filt = ("[1:v]scale=trunc(iw*0.3/2)*2:-2,"
            "pad=iw+6:ih+6:3:3:color=#3a3a3a[inset];"
            "[0:v][inset]overlay=32:main_h-overlay_h-32")
    subprocess.run(
        [ffmpeg, "-y", "-v", "error", "-i", str(full), "-i", str(zoom),
         "-filter_complex", filt, "-c:v", "libx264", "-preset", "medium",
         "-crf", "18", "-pix_fmt", "yuv420p", str(out)],
        check=True)


def write_timeline(jump_mask, duration: float, path: Path) -> Path:
    """Per-contour draw times, so commentary can be scripted to the render."""
    def ts(sec: float) -> str:
        return f"{int(sec // 60)}:{sec % 60:04.1f}"

    n = len(jump_mask)
    segments = []
    contour = 0
    start = 0
    for i in range(1, n + 1):
        if i < n and jump_mask[i] == jump_mask[start]:
            continue
        t0, t1 = start / n * duration, i / n * duration
        if jump_mask[start]:
            segments.append({"type": "jump", "start": round(t0, 2),
                             "end": round(t1, 2)})
        else:
            contour += 1
            segments.append({"type": "contour", "index": contour,
                             "start": round(t0, 2), "end": round(t1, 2),
                             "start_ts": ts(t0), "end_ts": ts(t1)})
        start = i
    path.write_text(json.dumps(
        {"duration": duration, "segments": segments}, indent=2))
    return path


def run(image_path: Path, s: Settings) -> dict[str, Path]:
    """Full pipeline for one image. Returns {view: rendered mp4 path}."""
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

    timeline = write_timeline(data.jump_mask, s.duration,
                              outdir / "timeline.json")
    print(f"Draw timeline: {timeline}")

    views = ["full", "zoom"] if s.view in ("both", "pip") else [s.view]
    view_settings = {view: s for view in views}
    if s.view == "pip" and not s.pip_inset_label:
        view_settings["zoom"] = replace(s, label=False)

    rendered: dict[str, Path] = {}
    if s.parallel and len(views) > 1:
        t0 = time.perf_counter()
        print(f"Rendering {' + '.join(views)} views in parallel...")
        with ProcessPoolExecutor(max_workers=len(views)) as pool:
            futures = {view: pool.submit(render_view, view, coeffs, freqs,
                                         recon, data.jump_mask,
                                         view_settings[view], outdir, name)
                       for view in views}
            for view in views:
                rendered[view] = futures[view].result()
                print(f"  -> {rendered[view]}")
        print(f"  (both views: {time.perf_counter() - t0:.1f}s)")
    else:
        for view in views:
            t0 = time.perf_counter()
            print(f"Rendering {view} view...")
            out = render_view(view, coeffs, freqs, recon, data.jump_mask,
                              view_settings[view], outdir, name)
            rendered[view] = out
            print(f"  -> {out}  ({time.perf_counter() - t0:.1f}s)")

    if s.view == "pip":
        t0 = time.perf_counter()
        print("Compositing picture-in-picture...")
        pip_out = outdir / f"{name}_pip.mp4"
        composite_pip(rendered["full"], rendered["zoom"], pip_out)
        rendered["pip"] = pip_out
        print(f"  -> {pip_out}  ({time.perf_counter() - t0:.1f}s)")
    return rendered


def main() -> None:
    args = parse_args()
    s = build_settings(args)
    run(Path(args.image), s)


if __name__ == "__main__":
    main()
