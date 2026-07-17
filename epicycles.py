#!/usr/bin/env python
"""Fourier epicycle animation renderer (3Blue1Brown style).

Usage (inside the `manim` conda env):
    python epicycles.py assets/jolly_roger.png --duration 180 --detail high
    python epicycles.py assets/luffy.png --config configs/luffy.yaml --view zoom --preview

Project-folder workflow:
    python epicycles.py init nika_dance --image assets/nika_dance.png
        -> creates projects/nika_dance/config.yaml (fully documented; the
           image is copied into the folder). Edit it, then:
    python epicycles.py projects/nika_dance
        -> runs the whole pipeline from that config; outputs land in the
           project folder. CLI flags still override (e.g. --preview).

Each run produces <name>_full.mp4 and/or <name>_zoom.mp4 (plus _pip.mp4 for
--view pip), rendered from one deterministic timeline so views stay
frame-synced.
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


PROJECT_CONFIG_TEMPLATE = """\
# Epicycle project config - run with: python epicycles.py {folder}
# CLI flags override anything set here (e.g. --preview for a fast check).

image: {image}       # input PNG (near-white bg / darker lines) or SVG,
                     # relative to this folder (or to the repo root)

duration: 60         # seconds for one full draw of the shape
detail: high         # low = 50, medium = 150, high = 400 vectors
# vectors: 800       # explicit vector count (overrides detail)
# samples: 16384     # path sample count (default: max(4096, 10 * vectors));
                     # method points uses samples/4 tour points (min 1000)

mode: full           # full = all contours | outline = largest contour only
extract: auto        # auto | otsu | alpha | canny
method: contour      # contour = ordered findContours loops (default)
                     # vtracer = bitmap -> SVG splines
                     # points  = unordered point cloud, greedy NN tour
# min-area: 0.001    # drop contours below this fraction of image area
# canny-low: 60      # only used with extract: canny
# canny-high: 160
smooth: 0.2          # 0 = raw pixel contours .. 1 = heavy simplification
# dedup-ratio: 0.6   # stroke-dedup area-ratio gate; inner edges of thick
# dedup-dist: 0.02   # lines are dropped only if they also hug the outer edge
                     # within this fraction of image size (protects detail)

fill: 0.8            # drawing height as a fraction of frame height
hold: 0              # >0: hold after the draw before the machinery fades
end_hold: 0          # >0: hold on the finished drawing after the fade
end_card: 0          # >0: scale the drawing up to fill the frame and hold (s)
# start-t: 0         # render only a slice of the timeline (0..1)
# end-t: 1

theme: classic       # classic | straw-hat | zoro | ace
# trace-color: "#FF7043"  # explicit hex trace color, overrides theme
trace_fade: 0        # >0 dims the settled path -> glowing comet head (try 0.6)
vignette: 0          # 0..1 radial background gradient
arrow_width: 0.5     # vector shaft thickness scale (1.0 = original)
arrow_head: 0.5      # vector arrowhead size scale (1.0 = original)
arrow_opacity: 0.5   # vector arrow opacity scale (1.0 = original);
                     # arrows fade in decreasing order of size down the chain
circle_opacity: 0.5  # circle brightness scale (1.0 = original);
                     # larger circles stay brighter than smaller ones
circle_width: 1.0    # circle stroke width scale; also tapers down the chain
trace_glow: 1.0      # trace glow halo intensity, 0..2 (0 = no halo)
jump_opacity: 1.0    # jump-line visibility, 0..2 (0 = hidden)
ghost_opacity: 0.06  # ghost preview stroke opacity
tip_glow: 0.6        # glowing pen-tip highlight intensity (0 = off)
bloom: 0.35          # post-render neon bloom strength (0 = off)
ghost: true          # faint preview of the final shape from t=0
ghost_mode: static   # static | fadein | late
label: true          # "N rotating vectors" corner label

view: pip            # full | zoom | both | pip (zoom inset in the full view)
# pip_inset_label: false
zoom_factor: 8
# zoom-start: 4      # ease magnification from this factor to zoom_factor
                     # over the draw (default: fixed at zoom_factor)
cam_lag: 0           # zoom camera: 0 = locked on pen tip; >0 = lazy follow (s)
fps: 60              # production is 1920x1080 at this fps (--preview: 480p15)
parallel: true       # render full + zoom as two parallel processes

# --- motion / background ---
ease: none           # none | in | out | inout — draw-speed easing. NOTE:
                     # timeline.json assumes linear; keep 'none' if you script
                     # narration to exact contour times.
bg_color: "#161616"  # frame background color

# --- reference imagery & reveals ---
# source_image: original.png  # colored original; enables the *_reveal options
start_reveal: line   # none | line | original — faint intro reference
start_opacity: 0.12  # opacity when start_reveal is 'original'
end_reveal: line     # none | line | original — what the end card shows
end_crossfade: 1.0   # seconds to dissolve the drawing into the end image
                     # (end_card must be > 0 for any end reveal to show)

# --- intro / outro title cards (set a title to enable) ---
# intro_title: "TITLE"
# intro_subtitle: "subtitle"
# intro_seconds: 3
# outro_title: "Thanks for watching"
# outro_subtitle: "Subscribe for more"
# outro_seconds: 4

# --- audio: mux a music/narration bed into the finals ---
# audio: track.mp3

# --- vertical (9:16) short of a window of the deliverable ---
shorts: false        # true = also emit <name>_short.mp4 (1080x1920)
shorts_start: 0.0    # crop window as a fraction of the full video (0..1)
shorts_end: 1.0
"""


def init_project(argv: list[str]) -> None:
    p = argparse.ArgumentParser(
        prog="epicycles.py init",
        description="Scaffold a project folder with a documented config.")
    p.add_argument("name", help="project name -> projects/<name>/")
    p.add_argument("--image", help="input image; copied into the folder")
    args = p.parse_args(argv)

    folder = Path("projects") / args.name
    cfg_file = folder / "config.yaml"
    if cfg_file.exists():
        raise SystemExit(f"{cfg_file} already exists - edit it or pick "
                         "another name")
    folder.mkdir(parents=True, exist_ok=True)

    image_value = "CHANGE_ME.png   # <- point this at your input image"
    if args.image:
        src = Path(args.image)
        if not src.exists():
            raise SystemExit(f"No such image: {src}")
        shutil.copy2(src, folder / src.name)
        image_value = src.name
    cfg_file.write_text(PROJECT_CONFIG_TEMPLATE.format(
        folder=folder, image=image_value))
    print(f"Created {cfg_file}")
    print(f"Edit it, then render with: python epicycles.py {folder}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("image", help="input PNG (near-white bg / darker lines) or "
                                 "SVG, or a project folder with a config.yaml")
    p.add_argument("--config", help="per-image YAML; CLI flags override it")
    p.add_argument("--duration", type=float, help="seconds for one full draw")
    p.add_argument("--detail", choices=["low", "medium", "high"],
                   help="vector count preset: 50 / 150 / 400")
    p.add_argument("--vectors", type=int, help="explicit vector count")
    p.add_argument("--samples", type=int, help="path sample count (default auto)")
    p.add_argument("--mode", choices=["full", "outline"])
    p.add_argument("--extract", choices=["auto", "otsu", "alpha", "canny"])
    p.add_argument("--method", choices=["contour", "vtracer", "points"],
                   help="path tracing: contour = ordered findContours loops "
                        "(default); vtracer = bitmap -> SVG splines; points = "
                        "unordered point cloud, ordered by a greedy "
                        "nearest-neighbour tour at generation time")
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
    p.add_argument("--arrow-width", type=float, dest="arrow_width",
                   help="vector shaft thickness scale (default 0.5, 1.0 = original)")
    p.add_argument("--arrow-head", type=float, dest="arrow_head",
                   help="vector arrowhead size scale (default 0.5, 1.0 = original)")
    p.add_argument("--arrow-opacity", type=float, dest="arrow_opacity",
                   help="vector arrow opacity scale (default 0.5, 1.0 = original)")
    p.add_argument("--circle-opacity", type=float, dest="circle_opacity",
                   help="circle brightness scale (default 0.5, 1.0 = original)")
    p.add_argument("--circle-width", type=float, dest="circle_width",
                   help="circle stroke width scale; also tapers down the chain")
    p.add_argument("--trace-glow", type=float, dest="trace_glow",
                   help="trace glow halo intensity, 0..2 (0 = no halo)")
    p.add_argument("--jump-opacity", type=float, dest="jump_opacity",
                   help="jump-line visibility, 0..2 (0 = hidden)")
    p.add_argument("--ghost-opacity", type=float, dest="ghost_opacity",
                   help="ghost preview stroke opacity (default 0.06)")
    p.add_argument("--tip-glow", type=float, dest="tip_glow",
                   help="glowing pen-tip highlight intensity, 0..1 (0 = off)")
    p.add_argument("--bloom", type=float,
                   help="post-render neon bloom strength, 0..1 (0 = off)")
    p.add_argument("--zoom-start", type=float, dest="zoom_start",
                   help="ease zoom magnification from this factor to "
                        "--zoom-factor over the draw (default: fixed)")
    p.add_argument("--dedup-ratio", type=float, dest="dedup_ratio",
                   help="stroke-dedup area-ratio gate (default 0.6)")
    p.add_argument("--dedup-dist", type=float, dest="dedup_dist",
                   help="stroke-dedup max edge gap as a fraction of image "
                        "size (default 0.02); smaller keeps more detail")
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
                   help="deprecated alias for --method vtracer")
    p.add_argument("--double-lines", action="store_true", dest="double_lines",
                   help="draw both edges of stroked lines instead of one")
    p.add_argument("-o", "--output", help="output directory")
    return p.parse_args()


def build_settings(args: argparse.Namespace) -> tuple[Settings, str | None]:
    s = Settings()
    cfg_image = None
    if args.config:
        cfg_image = s.apply_yaml(args.config)
    scalar_flags = ["duration", "detail", "vectors", "samples", "mode", "extract",
                    "method", "min_area", "canny_low", "canny_high", "smooth", "fill",
                    "hold", "end_hold", "end_card", "start_t", "end_t",
                    "theme", "trace_color", "ghost_mode", "vignette",
                    "arrow_width", "arrow_head", "arrow_opacity",
                    "circle_opacity", "circle_width", "trace_glow",
                    "jump_opacity", "ghost_opacity", "tip_glow", "bloom",
                    "zoom_start", "dedup_ratio", "dedup_dist",
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
    if args.via_svg:
        s.method = "vtracer"
    for flag in ("preview", "yes", "double_lines", "parallel",
                 "pip_inset_label"):
        if getattr(args, flag):
            setattr(s, flag, True)
    s.validate()
    return s, cfg_image


def render_view(view: str, coeffs, freqs, recon, jump_mask, s: Settings,
                outdir: Path, name: str) -> Path:
    from manim import tempconfig

    from epicycle_lib.scene import BG_COLOR, EpicycleScene

    px_w, px_h, fps = (854, 480, 15) if s.preview else (1920, 1080, s.fps)
    with tempconfig({
        "pixel_width": px_w, "pixel_height": px_h, "frame_rate": fps,
        "background_color": s.bg_color or BG_COLOR,
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


def bloom_pass(video: Path, strength: float) -> None:
    """Re-encode with a neon bloom: bright pass -> blur -> screen blend."""
    tmp = video.with_name(video.stem + ".bloom.mp4")
    lut = "'max(val-140,0)*2.5'"
    filt = (f"[0:v]format=gbrp,split[base][bright];"
            f"[bright]lutrgb=r={lut}:g={lut}:b={lut},gblur=sigma=14[glow];"
            f"[base][glow]blend=all_mode=screen:all_opacity={strength:.3f},"
            f"format=yuv420p")
    subprocess.run(
        [ffmpeg_bin(), "-y", "-v", "error", "-i", str(video),
         "-filter_complex", filt, "-c:v", "libx264", "-preset", "medium",
         "-crf", "18", str(tmp)],
        check=True)
    tmp.replace(video)


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


def _find_font() -> str:
    """A bold TTF for title cards; falls back to any DejaVu/Ubuntu face."""
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        str(Path(sys.executable).parent.parent / "fonts" / "Ubuntu-M.ttf"),
    ]
    for c in candidates:
        if Path(c).exists():
            return c
    hits = list(Path("/usr/share/fonts").rglob("*.ttf"))
    return str(hits[0]) if hits else "sans"


def _probe_duration(video: Path) -> float:
    """Seconds of a video via ffprobe (0.0 if it can't be read)."""
    ffprobe = str(Path(ffmpeg_bin()).with_name("ffprobe"))
    out = subprocess.run(
        [ffprobe, "-v", "error", "-show_entries", "format=duration",
         "-of", "default=nw=1:nk=1", str(video)],
        capture_output=True, text=True)
    try:
        return float(out.stdout.strip())
    except ValueError:
        return 0.0


def _card_clip(title: str, subtitle: str | None, seconds: float, bg: str,
               w: int, h: int, fps: int, out: Path, workdir: Path) -> Path:
    """Render a solid-background title card (title + optional subtitle)."""
    font = _find_font()
    title_f = workdir / "_card_title.txt"
    title_f.write_text(title)
    draws = [(f"drawtext=fontfile='{font}':textfile='{title_f}':"
              f"fontcolor=white:fontsize={int(h * 0.085)}:"
              f"x=(w-text_w)/2:y=(h-text_h)/2-{int(h * 0.04)}")]
    if subtitle:
        sub_f = workdir / "_card_sub.txt"
        sub_f.write_text(subtitle)
        draws.append(f"drawtext=fontfile='{font}':textfile='{sub_f}':"
                     f"fontcolor=0xB0B0B0:fontsize={int(h * 0.042)}:"
                     f"x=(w-text_w)/2:y=(h-text_h)/2+{int(h * 0.06)}")
    vf = (f"color=c={bg}:s={w}x{h}:d={seconds}:r={fps},format=yuv420p,"
          + ",".join(draws))
    subprocess.run(
        [ffmpeg_bin(), "-y", "-v", "error", "-f", "lavfi", "-i", vf,
         "-c:v", "libx264", "-preset", "medium", "-crf", "18",
         "-pix_fmt", "yuv420p", str(out)],
        check=True)
    return out


def add_cards(video: Path, s: Settings, w: int, h: int, fps: int,
              workdir: Path) -> None:
    """Prepend an intro card and/or append an outro card to `video`."""
    bg = s.bg_color or "#161616"
    clips: list[Path] = []
    if s.intro_title and s.intro_seconds > 0:
        clips.append(_card_clip(s.intro_title, s.intro_subtitle, s.intro_seconds,
                                bg, w, h, fps, workdir / "_intro.mp4", workdir))
    clips.append(video)
    if s.outro_title and s.outro_seconds > 0:
        clips.append(_card_clip(s.outro_title, s.outro_subtitle, s.outro_seconds,
                                bg, w, h, fps, workdir / "_outro.mp4", workdir))
    if len(clips) == 1:
        return
    inputs: list[str] = []
    for c in clips:
        inputs += ["-i", str(c)]
    streams = "".join(f"[{i}:v]" for i in range(len(clips)))
    filt = f"{streams}concat=n={len(clips)}:v=1:a=0[v]"
    tmp = video.with_name(video.stem + ".carded.mp4")
    subprocess.run(
        [ffmpeg_bin(), "-y", "-v", "error", *inputs, "-filter_complex", filt,
         "-map", "[v]", "-c:v", "libx264", "-preset", "medium", "-crf", "18",
         "-pix_fmt", "yuv420p", str(tmp)],
        check=True)
    tmp.replace(video)
    for c in clips:
        if c != video and c.exists():
            c.unlink()


def mux_audio(video: Path, audio: Path) -> None:
    """Mux an audio track onto `video`, trimmed to the shorter of the two."""
    tmp = video.with_name(video.stem + ".audio.mp4")
    subprocess.run(
        [ffmpeg_bin(), "-y", "-v", "error", "-i", str(video), "-i", str(audio),
         "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac",
         "-b:a", "192k", "-shortest", str(tmp)],
        check=True)
    tmp.replace(video)


def make_short(video: Path, start_frac: float, end_frac: float, out: Path,
               h: int, fps: int) -> None:
    """Center-crop a 9:16 vertical Short from a time window of `video`."""
    dur = _probe_duration(video)
    t0, t1 = start_frac * dur, end_frac * dur
    # crop full height, 9:16 width, centered; then scale up to 1080x1920
    filt = ("crop=w=trunc(ih*9/16/2)*2:h=ih:x=(iw-ow)/2:y=0,"
            "scale=1080:1920,format=yuv420p")
    subprocess.run(
        [ffmpeg_bin(), "-y", "-v", "error", "-ss", f"{t0:.3f}", "-to",
         f"{t1:.3f}", "-i", str(video), "-vf", filt, "-r", str(fps),
         "-c:v", "libx264", "-preset", "medium", "-crf", "18",
         "-pix_fmt", "yuv420p", str(out)],
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

    # resolve optional assets relative to the project folder, then check them
    proj = image_path.parent
    for attr in ("source_image", "audio"):
        val = getattr(s, attr)
        if val and not Path(val).is_absolute():
            cand = proj / val
            setattr(s, attr, str(cand if cand.exists() else Path(val)))
        val = getattr(s, attr)
        if val and not Path(val).exists():
            raise SystemExit(f"{attr} not found: {val}")

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

    if s.bloom > 0:
        t0 = time.perf_counter()
        print(f"Applying bloom ({s.bloom:.2f})...")
        for view in views:
            bloom_pass(rendered[view], s.bloom)
        print(f"  ({time.perf_counter() - t0:.1f}s)")

    if s.view == "pip":
        t0 = time.perf_counter()
        print("Compositing picture-in-picture...")
        pip_out = outdir / f"{name}_pip.mp4"
        composite_pip(rendered["full"], rendered["zoom"], pip_out)
        rendered["pip"] = pip_out
        print(f"  -> {pip_out}  ({time.perf_counter() - t0:.1f}s)")

    # -- post: title cards, audio, vertical short (all config-gated) --
    px_w, px_h, fps = (854, 480, 15) if s.preview else (1920, 1080, s.fps)
    finals = [rendered["pip"]] if "pip" in rendered else [rendered[v] for v in views]
    if (s.intro_title or s.outro_title):
        print("Adding title cards...")
        for f in finals:
            add_cards(f, s, px_w, px_h, fps, outdir)
    if s.audio:
        print(f"Muxing audio ({Path(s.audio).name})...")
        for f in finals:
            mux_audio(f, Path(s.audio))
    if s.shorts:
        print("Rendering vertical short...")
        short_out = outdir / f"{name}_short.mp4"
        make_short(finals[0], s.shorts_start, s.shorts_end, short_out, px_h, fps)
        rendered["short"] = short_out
        print(f"  -> {short_out}")
    return rendered


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "init":
        init_project(sys.argv[2:])
        return

    args = parse_args()
    target = Path(args.image)
    if target.is_dir():
        if not args.config:
            cfg_file = target / "config.yaml"
            if not cfg_file.exists():
                raise SystemExit(f"{target} is a folder but has no "
                                 "config.yaml - create one with: "
                                 f"python epicycles.py init <name>")
            args.config = str(cfg_file)
    s, cfg_image = build_settings(args)

    if target.is_dir():
        if not cfg_image or "CHANGE_ME" in str(cfg_image):
            raise SystemExit(f"Set the 'image:' key in {args.config} first")
        in_project = target / str(cfg_image)
        image = in_project if in_project.exists() else Path(str(cfg_image))
        if s.output is None:
            s.output = str(target)
    else:
        image = target
    run(image, s)


if __name__ == "__main__":
    main()
