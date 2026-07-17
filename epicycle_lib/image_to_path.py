"""PNG/SVG -> ordered complex sample chain + jump mask.

Pipeline: binarize (Otsu / alpha / Canny) -> contours -> order (largest first,
then greedy nearest-neighbour with each closed loop rotated to start at the
nearest point) -> chain with straight connector segments recorded as jumps ->
uniform arc-length resample -> rounded junction corners -> normalized manim
coordinates (centroid at origin, so the DFT's DC term vanishes).

Jumps are known by construction: we insert the connectors ourselves and carry
their sample-index ranges through resampling as a boolean mask. No speed
heuristics anywhere.
"""
from __future__ import annotations

import hashlib
import tempfile
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .settings import Settings

FRAME_H = 8.0                # manim default frame height in scene units
FRAME_W = FRAME_H * 16 / 9


@dataclass
class PathData:
    samples: np.ndarray        # complex, manim coords, uniform arc length, mean 0
    jump_mask: np.ndarray      # bool, True where the sample lies on a connector
    pixel_samples: np.ndarray  # complex, source pixel coords (y down), for inspect
    source_image: np.ndarray | None  # BGR image for the inspect background
    n_contours: int


# ---------------------------------------------------------------- binarize

def _binarize(img: np.ndarray, s: Settings) -> np.ndarray:
    """Return a uint8 mask with the linework as foreground (255)."""
    alpha = img[:, :, 3] if img.ndim == 3 and img.shape[2] == 4 else None
    mode = s.extract
    if mode == "auto":
        mode = "alpha" if alpha is not None and alpha.min() < 128 else "otsu"

    if mode == "alpha":
        if alpha is None:
            raise SystemExit("--extract alpha: image has no alpha channel")
        _, binary = cv2.threshold(alpha, 127, 255, cv2.THRESH_BINARY)
        return binary

    gray = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    if mode == "otsu":
        # Darker lines on a lighter (near-white) background -> invert so lines are fg.
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        return binary
    if mode == "canny":
        return cv2.Canny(gray, s.canny_low, s.canny_high)
    raise SystemExit(f"Unknown extract mode: {mode}")


# ---------------------------------------------------------------- contours

def _poly_area(z: np.ndarray) -> float:
    x, y = z.real, z.imag
    return 0.5 * abs(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def _smooth_contour(c: np.ndarray, smooth: float) -> np.ndarray:
    if smooth <= 0:
        return c
    eps = (smooth ** 2) * 0.02 * cv2.arcLength(c, True)
    if eps <= 0:
        return c
    approx = cv2.approxPolyDP(c, eps, True)
    return approx if len(approx) >= 3 else c


def _contours_from_binary(binary: np.ndarray, s: Settings) -> list[np.ndarray]:
    found, hierarchy = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    keep = [True] * len(found)
    if not s.double_lines and hierarchy is not None:
        # Stroked linework yields two contours per stroke: the outside and the
        # inside edge, nearly equal in enclosed area. Drop the inner edge so a
        # stroke is drawn as one line. Two gates protect genuine detail:
        # the area-ratio test (holes small relative to their parent survive)
        # AND a distance test - a real stroke's two edges run parallel a few
        # pixels apart *everywhere*, so an inner contour is only dropped when
        # 90% of its points lie within dedup_dist of the outer edge.
        max_gap = s.dedup_dist * max(binary.shape)
        for i, row in enumerate(hierarchy[0]):
            parent = row[3]
            if parent == -1:
                continue
            parent_area = cv2.contourArea(found[parent])
            if parent_area <= 0 or \
                    cv2.contourArea(found[i]) / parent_area <= s.dedup_ratio:
                continue
            child = found[i][:, 0, :].astype(np.float64)
            outer = found[parent][:, 0, :].astype(np.float64)
            child = child[::max(1, len(child) // 200)]
            outer = outer[::max(1, len(outer) // 400)]
            gaps = np.sqrt(((child[:, None, :] - outer[None, :, :]) ** 2)
                           .sum(-1)).min(axis=1)
            if np.quantile(gaps, 0.9) <= max_gap:
                keep[i] = False
    min_area = s.min_area * binary.shape[0] * binary.shape[1]
    candidates = [c for i, c in enumerate(found) if keep[i] and len(c) >= 8]
    if not candidates:
        raise SystemExit("No contours found — check the image / --extract mode")
    kept = [c for c in candidates if cv2.contourArea(c) >= min_area]
    if not kept:  # everything was tiny; keep the largest rather than fail
        kept = [max(candidates, key=cv2.contourArea)]
    contours = []
    for c in kept:
        pts = _smooth_contour(c, s.smooth)[:, 0, :].astype(np.float64)
        contours.append(pts[:, 0] + 1j * pts[:, 1])
    if s.mode == "outline":
        contours = [max(contours, key=_poly_area)]
    return contours


# ---------------------------------------------------------------- SVG input

def _contours_from_svg(svg_path: Path, s: Settings,
                       drop_full_canvas: tuple[float, float] | None = None) -> list[np.ndarray]:
    import svgelements

    svg = svgelements.SVG.parse(str(svg_path))
    contours = []
    for element in svg.elements():
        if not isinstance(element, svgelements.Path) or len(element) == 0:
            continue
        for sub in element.as_subpaths():
            sp = svgelements.Path(sub)
            length = sp.length(error=1e-3)
            if length < 4:
                continue
            n = max(16, int(length / 2))  # ~2 units of arc length per sample
            pts = np.array([complex(p.x, p.y)
                            for p in (sp.point(i / n) for i in range(n + 1))
                            if p is not None])
            if len(pts) < 8:
                continue
            if drop_full_canvas is not None:
                w = pts.real.max() - pts.real.min()
                h = pts.imag.max() - pts.imag.min()
                # vtracer sometimes emits a canvas-sized background rectangle
                if w * h >= 0.95 * drop_full_canvas[0] * drop_full_canvas[1]:
                    continue
            contours.append(pts)
    if not contours:
        raise SystemExit(f"No usable paths found in {svg_path}")
    min_area = s.min_area * (drop_full_canvas[0] * drop_full_canvas[1]
                             if drop_full_canvas else 0)
    if min_area > 0:
        kept = [c for c in contours if _poly_area(c) >= min_area]
        contours = kept or [max(contours, key=_poly_area)]
    if s.mode == "outline":
        contours = [max(contours, key=_poly_area)]
    return contours


def _contours_via_vtracer(binary: np.ndarray, s: Settings) -> list[np.ndarray]:
    import vtracer

    with tempfile.TemporaryDirectory() as tmp:
        png = Path(tmp) / "binary.png"
        svg = Path(tmp) / "traced.svg"
        cv2.imwrite(str(png), 255 - binary)  # vtracer traces dark shapes on light bg
        vtracer.convert_image_to_svg_py(
            str(png), str(svg),
            colormode="binary", mode="spline", filter_speckle=4,
        )
        h, w = binary.shape[:2]
        return _contours_from_svg(svg, s, drop_full_canvas=(float(w), float(h)))


# ---------------------------------------------------------------- ordering

def _order_contours(contours: list[np.ndarray]) -> list[np.ndarray]:
    """Largest first, then greedy nearest-neighbour; each closed loop is
    rotated (np.roll) so it starts at the point nearest the previous end."""
    remaining = sorted(contours, key=_poly_area, reverse=True)
    ordered = [remaining.pop(0)]
    current = ordered[0][0]  # loops end where they start
    while remaining:
        best_i, best_j, best_d = 0, 0, np.inf
        for i, c in enumerate(remaining):
            j = int(np.argmin(np.abs(c - current)))
            d = abs(c[j] - current)
            if d < best_d:
                best_i, best_j, best_d = i, j, d
        chosen = remaining.pop(best_i)
        chosen = np.roll(chosen, -best_j)
        ordered.append(chosen)
        current = chosen[0]
    return ordered


def _points_chain(contours: list[np.ndarray],
                  n_samples: int) -> tuple[np.ndarray, np.ndarray]:
    """The 'points' method: forget the contour structure, treat everything as
    one unordered point cloud, and let a greedy nearest-neighbour tour over the
    individual points decide the pen path. Edges much longer than the typical
    point spacing are flagged as jumps so the renderer dims them."""
    # Evenly-spaced points along every contour (smoothed contours keep only
    # sparse corners, which would skew the tour's typical edge length).
    target = max(1000, n_samples // 4)
    closed = [np.append(c, c[0]) for c in contours]
    total = sum(np.abs(np.diff(c)).sum() for c in closed)
    spacing = max(total / target, 1e-9)
    parts = []
    for c in closed:
        seg = np.abs(np.diff(c))
        cum = np.concatenate([[0.0], np.cumsum(seg)])
        if cum[-1] <= 0:
            continue
        m = max(8, int(round(cum[-1] / spacing)))
        svals = np.arange(m) * cum[-1] / m
        idx = np.clip(np.searchsorted(cum, svals, side="right") - 1,
                      0, len(seg) - 1)
        frac = (svals - cum[idx]) / np.maximum(seg[idx], 1e-12)
        parts.append(c[idx] + (c[idx + 1] - c[idx]) * frac)
    cloud = np.concatenate(parts)

    n = len(cloud)
    used = np.zeros(n, dtype=bool)
    order = np.empty(n, dtype=int)
    current = int(np.argmin(cloud.imag))  # deterministic start: topmost point
    for k in range(n):
        order[k] = current
        used[current] = True
        if k + 1 < n:
            d = np.abs(cloud - cloud[current])
            d[used] = np.inf
            current = int(np.argmin(d))

    tour = cloud[order]
    pts = np.append(tour, tour[0])  # close the chain
    seg = np.abs(np.diff(pts))
    positive = seg[seg > 0]
    median = np.median(positive) if len(positive) else 1.0
    long_edge = seg > 4 * median
    flags = np.zeros(len(pts), dtype=bool)
    flags[:-1] |= long_edge
    flags[1:] |= long_edge
    return pts, flags


# ---------------------------------------------------------------- chaining

def _build_chain(contours: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Concatenate closed loops with straight connectors; returns
    (points, per-point jump flag). The chain ends back at its start point."""
    pts_segments: list[np.ndarray] = []
    flag_segments: list[np.ndarray] = []

    def add(points: np.ndarray, is_jump: bool) -> None:
        pts_segments.append(points)
        flag_segments.append(np.full(len(points), is_jump))

    first_start = contours[0][0]
    for i, c in enumerate(contours):
        add(np.append(c, c[0]), False)  # traverse the loop and close it
        target = contours[i + 1][0] if i + 1 < len(contours) else first_start
        a, b = c[0], target
        length = abs(b - a)
        if length > 1e-9:
            n = max(2, int(length))  # ~1px connector spacing
            tvals = np.linspace(0, 1, n + 1)[1:-1]
            if len(tvals):
                add(a + (b - a) * tvals, True)
    add(np.array([first_start]), False)  # close the whole chain

    return np.concatenate(pts_segments), np.concatenate(flag_segments)


def _resample(pts: np.ndarray, point_jump: np.ndarray,
              n: int) -> tuple[np.ndarray, np.ndarray]:
    """Uniform arc-length resampling of the closed chain -> constant pen speed."""
    seg_len = np.abs(np.diff(pts))
    edge_jump = point_jump[:-1] | point_jump[1:]
    cum = np.concatenate([[0.0], np.cumsum(seg_len)])
    total = cum[-1]
    if total <= 0:
        raise SystemExit("Degenerate path: zero total length")
    svals = np.arange(n) * total / n
    idx = np.clip(np.searchsorted(cum, svals, side="right") - 1, 0, len(seg_len) - 1)
    frac = (svals - cum[idx]) / np.maximum(seg_len[idx], 1e-12)
    out = pts[idx] + (pts[idx + 1] - pts[idx]) * frac
    return out, edge_jump[idx]


def _round_junctions(pts: np.ndarray, mask: np.ndarray,
                     window: int = 4, passes: int = 3) -> np.ndarray:
    """Locally smooth the corners where connectors meet contours, to suppress
    Gibbs ringing in the truncated Fourier series."""
    n = len(pts)
    transitions = np.nonzero(mask != np.roll(mask, 1))[0]
    if len(transitions) == 0:
        return pts
    selected = np.zeros(n, dtype=bool)
    for b in transitions:
        selected[np.arange(b - window, b + window + 1) % n] = True
    smoothed = pts.copy()
    for _ in range(passes):
        avg = 0.25 * np.roll(smoothed, 1) + 0.5 * smoothed + 0.25 * np.roll(smoothed, -1)
        smoothed = np.where(selected, avg, smoothed)
    return smoothed


# ---------------------------------------------------------------- normalize

def _normalize(pixel_samples: np.ndarray, fill: float) -> np.ndarray:
    z = pixel_samples - pixel_samples.mean()
    z = z.real - 1j * z.imag  # image y points down, manim y points up
    height = max(z.imag.max() - z.imag.min(), 1e-9)
    width = max(z.real.max() - z.real.min(), 1e-9)
    factor = min(fill * FRAME_H / height, 0.95 * FRAME_W / width)
    z = z * factor
    return z - z.mean()


# ---------------------------------------------------------------- entry point

def extract_path(image_path: str | Path, s: Settings,
                 cache_dir: Path | None = None) -> PathData:
    image_path = Path(image_path)
    raw = image_path.read_bytes()
    n_samples = s.n_samples
    is_svg = image_path.suffix.lower() == ".svg"

    cache_file = None
    if cache_dir is not None:
        params = (hashlib.sha1(raw).hexdigest(), s.extract, s.mode, s.min_area,
                  s.canny_low, s.canny_high, s.smooth, s.fill, s.method,
                  s.double_lines, s.dedup_ratio, s.dedup_dist, n_samples)
        key = hashlib.sha1(repr(params).encode()).hexdigest()[:16]
        cache_file = cache_dir / f"cache_{key}.npz"

    source_image = None
    if not is_svg:
        source_image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_UNCHANGED)
        if source_image is None:
            raise SystemExit(f"Could not read image: {image_path}")

    if cache_file is not None and cache_file.exists():
        data = np.load(cache_file)
        return PathData(data["samples"], data["jump_mask"].astype(bool),
                        data["pixel_samples"], source_image, int(data["n_contours"]))

    if is_svg:
        contours = _contours_from_svg(image_path, s)
    else:
        binary = _binarize(source_image, s)
        contours = (_contours_via_vtracer(binary, s) if s.method == "vtracer"
                    else _contours_from_binary(binary, s))

    if s.method == "points":
        chain, flags = _points_chain(contours, n_samples)
    else:
        contours = _order_contours(contours)
        chain, flags = _build_chain(contours)
    pixel_samples, jump_mask = _resample(chain, flags, n_samples)
    pixel_samples = _round_junctions(pixel_samples, jump_mask)
    samples = _normalize(pixel_samples, s.fill)

    if cache_file is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_file, samples=samples, jump_mask=jump_mask,
                            pixel_samples=pixel_samples, n_contours=len(contours))

    return PathData(samples, jump_mask, pixel_samples, source_image, len(contours))
