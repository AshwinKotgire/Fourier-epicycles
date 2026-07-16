#!/usr/bin/env python
"""Batch renderer: run several images from one YAML, optionally concat them.

Usage (inside the `manim` conda env):
    python epicycles_batch.py configs/batch_example.yaml --yes

Batch YAML format:
    defaults:            # any Settings keys, applied to every job
      duration: 60
      view: pip
    jobs:                # rendered in order; keys override defaults
      - image: assets/luffy.png
        duration: 90
        theme: straw-hat
      - image: assets/zoro.png
        theme: zoro
    concat: true                     # stitch the job videos into one file
    concat_output: output/episode1.mp4

Concat picks each job's pip video if present, else full, else the single
rendered view. All jobs must share resolution/fps (don't mix preview and
production jobs when concatenating).
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import fields
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

from epicycle_lib.settings import Settings
from epicycles import ffmpeg_bin, run


def job_settings(job: dict, defaults: dict, force_yes: bool) -> tuple[Path, Settings]:
    merged = {**defaults, **job}
    image = merged.pop("image", None)
    if image is None:
        raise SystemExit(f"Batch job missing 'image': {job}")
    s = Settings()
    valid = {f.name for f in fields(s)}
    for key, value in merged.items():
        name = key.replace("-", "_")
        if name not in valid:
            raise SystemExit(f"Unknown key in batch job {image}: {key!r}")
        setattr(s, name, value)
    if force_yes:
        s.yes = True
    s.validate()
    return Path(image), s


def pick_for_concat(rendered: dict[str, Path]) -> Path:
    for view in ("pip", "full", "zoom"):
        if view in rendered:
            return rendered[view]
    return next(iter(rendered.values()))


def concat_videos(videos: list[Path], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    list_file = out.with_suffix(".txt")
    list_file.write_text(
        "".join(f"file '{v.resolve()}'\n" for v in videos))
    subprocess.run(
        [ffmpeg_bin(), "-y", "-v", "error", "-f", "concat", "-safe", "0",
         "-i", str(list_file), "-c", "copy", str(out)],
        check=True)
    list_file.unlink()


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("batch_yaml", help="batch config (defaults/jobs/concat)")
    p.add_argument("--yes", action="store_true",
                   help="skip every inspection prompt")
    args = p.parse_args()

    cfg = yaml.safe_load(Path(args.batch_yaml).read_text()) or {}
    jobs = cfg.get("jobs")
    if not jobs:
        raise SystemExit(f"No 'jobs' list in {args.batch_yaml}")
    defaults = cfg.get("defaults", {})

    outputs = []
    for i, job in enumerate(jobs, 1):
        image, s = job_settings(job, defaults, args.yes)
        print(f"=== Job {i}/{len(jobs)}: {image} ===")
        rendered = run(image, s)
        outputs.append(pick_for_concat(rendered))

    if cfg.get("concat"):
        out = Path(cfg.get("concat_output", "output/batch.mp4"))
        print(f"Concatenating {len(outputs)} videos -> {out}")
        concat_videos(outputs, out)
        print(f"  -> {out}")


if __name__ == "__main__":
    main()
