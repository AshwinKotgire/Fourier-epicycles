"""Settings dataclass. Precedence: defaults < YAML config < CLI flags."""
from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path

import yaml

DETAIL_VECTORS = {"low": 50, "medium": 150, "high": 400}

# Trace-color presets; --trace-color overrides the theme
THEMES = {
    "classic": "#FFFF00",     # 3b1b yellow
    "straw-hat": "#FFC94D",   # Luffy gold
    "zoro": "#7ED957",        # green
    "ace": "#FF7043",         # fire orange
}


@dataclass
class Settings:
    duration: float = 60.0        # seconds for one full draw of the shape
    detail: str = "medium"        # low | medium | high (50/150/400 vectors)
    vectors: int | None = None    # explicit vector count, overrides detail
    samples: int | None = None    # path sample count; None = max(4096, 10 * vectors)
    mode: str = "full"            # full = all contours | outline = largest only
    extract: str = "auto"         # auto | otsu | alpha | canny
    method: str = "contour"       # contour = findContours loops | vtracer = via
                                  # SVG splines | points = unordered point cloud,
                                  # ordered by a greedy nearest-neighbour tour
    min_area: float = 0.001       # drop contours below this fraction of image area
    canny_low: int = 60
    canny_high: int = 160
    smooth: float = 0.2           # 0 = raw pixel contours .. 1 = heavy simplification
    fill: float = 0.8             # drawing height as fraction of frame height
    hold: float = 0.0             # seconds to hold after the draw completes
    end_hold: float = 0.0         # seconds to hold after circles fade out
    end_card: float = 0.0         # >0: scale the finished drawing up and hold (s)
    start_t: float = 0.0          # render only the slice [start_t, end_t] of the
    end_t: float = 1.0            # timeline (0..1); great for tuning long draws
    trace_fade: float = 0.0       # 0 = trace stays bright; 1 = settled path dims hard
    theme: str = "classic"        # trace-color preset: classic|straw-hat|zoro|ace
    trace_color: str | None = None  # explicit hex trace color, overrides theme
    ghost: bool = True            # faint preview of the final shape from t=0
    ghost_mode: str = "static"    # static | fadein (grows with t) | late (last third)
    vignette: float = 0.0         # 0 = flat background; 1 = strong radial gradient
    arrow_width: float = 0.5      # vector shaft thickness scale (1.0 = original)
    arrow_head: float = 0.5       # vector arrowhead size scale (1.0 = original)
    arrow_opacity: float = 0.5    # vector arrow opacity scale (1.0 = original);
                                  # opacity tapers down the chain with magnitude
    circle_opacity: float = 0.5   # circle brightness scale (1.0 = original);
                                  # larger circles stay brighter than smaller
    label: bool = True            # "N rotating vectors" corner label
    view: str = "both"            # full | zoom | both | pip (zoom inset in full)
    zoom_factor: float = 8.0
    cam_lag: float = 0.0          # zoom camera: 0 = locked on pen tip; >0 = lazy follow (s)
    fps: int = 60
    preview: bool = False         # 854x480 @ 15fps fast iteration
    parallel: bool = False        # render full + zoom views as parallel processes
    pip_inset_label: bool = False  # keep the corner label inside the pip inset
    yes: bool = False             # skip the inspection confirmation prompt
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

    def apply_yaml(self, path: str | Path) -> str | None:
        """Apply a YAML config; returns its 'image' key (project configs
        name their input image there) or None."""
        cfg = yaml.safe_load(Path(path).read_text()) or {}
        image = cfg.pop("image", None)
        valid = {f.name for f in fields(self)}
        for key, value in cfg.items():
            name = key.replace("-", "_")
            if name not in valid:
                raise SystemExit(f"Unknown key in {path}: {key!r}")
            setattr(self, name, value)
        return image

    def validate(self) -> None:
        if self.detail not in DETAIL_VECTORS:
            raise SystemExit(f"--detail must be one of {list(DETAIL_VECTORS)}")
        if self.mode not in ("full", "outline"):
            raise SystemExit("--mode must be full or outline")
        if self.extract not in ("auto", "otsu", "alpha", "canny"):
            raise SystemExit("--extract must be auto, otsu, alpha or canny")
        if self.method not in ("contour", "vtracer", "points"):
            raise SystemExit("--method must be contour, vtracer or points")
        if self.view not in ("full", "zoom", "both", "pip"):
            raise SystemExit("--view must be full, zoom, both or pip")
        if not 0.0 <= self.smooth <= 1.0:
            raise SystemExit("--smooth must be in 0..1")
        if not 0.0 <= self.trace_fade <= 1.0:
            raise SystemExit("--trace-fade must be in 0..1")
        if self.theme not in THEMES:
            raise SystemExit(f"--theme must be one of {list(THEMES)}")
        if self.ghost_mode not in ("static", "fadein", "late"):
            raise SystemExit("--ghost-mode must be static, fadein or late")
        if not 0.0 <= self.vignette <= 1.0:
            raise SystemExit("--vignette must be in 0..1")
        for knob in ("arrow_width", "arrow_head", "arrow_opacity",
                     "circle_opacity"):
            if not 0.0 < getattr(self, knob) <= 2.0:
                raise SystemExit(f"--{knob.replace('_', '-')} must be in (0, 2]")
        if not 0.0 <= self.start_t < self.end_t <= 1.0:
            raise SystemExit("need 0 <= start-t < end-t <= 1")
        if not 0.0 < self.fill <= 1.0:
            raise SystemExit("--fill must be in (0, 1]")

    @property
    def trace_hex(self) -> str:
        return self.trace_color or THEMES[self.theme]
