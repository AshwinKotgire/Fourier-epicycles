"""Settings dataclass. Precedence: defaults < YAML config < CLI flags."""
from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path

import yaml

DETAIL_VECTORS = {"low": 50, "medium": 150, "high": 400}


@dataclass
class Settings:
    duration: float = 60.0        # seconds for one full draw of the shape
    detail: str = "medium"        # low | medium | high (50/150/400 vectors)
    vectors: int | None = None    # explicit vector count, overrides detail
    samples: int | None = None    # path sample count; None = max(4096, 10 * vectors)
    mode: str = "full"            # full = all contours | outline = largest only
    extract: str = "auto"         # auto | otsu | alpha | canny
    min_area: float = 0.001       # drop contours below this fraction of image area
    canny_low: int = 60
    canny_high: int = 160
    smooth: float = 0.2           # 0 = raw pixel contours .. 1 = heavy simplification
    fill: float = 0.8             # drawing height as fraction of frame height
    hold: float = 0.0             # seconds to hold after the draw completes
    end_hold: float = 0.0         # seconds to hold after circles fade out
    ghost: bool = True            # faint preview of the final shape from t=0
    label: bool = True            # "N rotating vectors" corner label
    view: str = "both"            # full | zoom | both
    zoom_factor: float = 8.0
    cam_lag: float = 0.0          # zoom camera: 0 = locked on pen tip; >0 = lazy follow (s)
    fps: int = 60
    preview: bool = False         # 854x480 @ 15fps fast iteration
    yes: bool = False             # skip the inspection confirmation prompt
    via_svg: bool = False         # route the bitmap through vtracer -> SVG -> samples
    double_lines: bool = False    # draw both edges of stroked lines (default: one)
    output: str | None = None     # output directory; default output/<image-stem>/

    @property
    def n_vectors(self) -> int:
        if self.vectors is not None:
            return int(self.vectors)
        return DETAIL_VECTORS[self.detail]

    @property
    def n_samples(self) -> int:
        if self.samples is not None:
            return int(self.samples)
        return max(4096, 10 * self.n_vectors)

    def apply_yaml(self, path: str | Path) -> None:
        cfg = yaml.safe_load(Path(path).read_text()) or {}
        valid = {f.name for f in fields(self)}
        for key, value in cfg.items():
            name = key.replace("-", "_")
            if name not in valid:
                raise SystemExit(f"Unknown key in {path}: {key!r}")
            setattr(self, name, value)

    def validate(self) -> None:
        if self.detail not in DETAIL_VECTORS:
            raise SystemExit(f"--detail must be one of {list(DETAIL_VECTORS)}")
        if self.mode not in ("full", "outline"):
            raise SystemExit("--mode must be full or outline")
        if self.extract not in ("auto", "otsu", "alpha", "canny"):
            raise SystemExit("--extract must be auto, otsu, alpha or canny")
        if self.view not in ("full", "zoom", "both"):
            raise SystemExit("--view must be full, zoom or both")
        if not 0.0 <= self.smooth <= 1.0:
            raise SystemExit("--smooth must be in 0..1")
        if not 0.0 < self.fill <= 1.0:
            raise SystemExit("--fill must be in (0, 1]")
