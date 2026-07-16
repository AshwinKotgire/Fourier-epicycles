"""EpicycleScene: one scene class, two views sharing one deterministic timeline.

The tip position at time t is a pure function of t, and the trace is revealed
from the precomputed reconstruction at the same t, so the "full" and "zoom"
renders stay frame-for-frame in sync for picture-in-picture editing.
"""
from __future__ import annotations

import numpy as np
from manim import (DR, ORIGIN, RIGHT, TAU, WHITE, YELLOW, Circle, FadeOut,
                   Line, MovingCameraScene, Text, ValueTracker, VGroup,
                   VMobject, config, linear)

BG_COLOR = "#161616"
CIRCLE_COLOR = "#58C4DD"   # 3b1b light blue
VECTOR_COLOR = "#ffffff"
LABEL_COLOR = "#7a7a7a"

# Arrowheads: constant length for long arrows, proportional for short ones
HEAD_LENGTH = 0.18         # scene units, at zoom 1
HEAD_LENGTH_RATIO = 0.35   # head length never exceeds this fraction of the arrow
HEAD_WIDTH_RATIO = 0.45    # half-width of the head relative to its length

# (stroke width, opacity) layers: core trace + 3 stacked glow copies
TRACE_LAYERS = [(3.0, 1.0), (6.0, 0.12), (12.0, 0.06), (20.0, 0.03)]
JUMP_LAYERS = [(1.5, 0.15)]


def c2p(z: complex) -> np.ndarray:
    return np.array([z.real, z.imag, 0.0])


def path_points(zs: np.ndarray) -> np.ndarray:
    return np.column_stack([zs.real, zs.imag, np.zeros(len(zs))])


def mask_runs(mask: np.ndarray) -> list[tuple[int, int, bool]]:
    """Consecutive runs of the jump mask -> [(start, end_exclusive, is_jump)]."""
    runs = []
    start = 0
    for i in range(1, len(mask)):
        if mask[i] != mask[start]:
            runs.append((start, i, bool(mask[start])))
            start = i
    runs.append((start, len(mask), bool(mask[start])))
    return runs


class EpicycleScene(MovingCameraScene):
    def __init__(self, coeffs, freqs, recon, jump_mask, settings, view, **kwargs):
        self.coeffs = np.asarray(coeffs)
        self.freqs = np.asarray(freqs, dtype=np.float64)
        self.recon = np.asarray(recon)
        self.jump_mask = np.asarray(jump_mask, dtype=bool)
        self.s = settings
        self.view = view
        super().__init__(**kwargs)

    @staticmethod
    def _reveal_updater(template: VMobject, s0: float, s1: float,
                        tracker: ValueTracker):
        span = max(s1 - s0, 1e-9)
        state = {"last": None}

        def update(mobject: VMobject) -> None:
            alpha = float(np.clip((tracker.get_value() - s0) / span, 0.0, 1.0))
            if alpha == state["last"]:
                return
            state["last"] = alpha
            if alpha <= 0:
                mobject.set_points(np.zeros((0, 3)))
            else:
                mobject.pointwise_become_partial(template, 0, alpha)

        return update

    def construct(self):
        s = self.s
        coeffs, freqs = self.coeffs, self.freqs
        n_vec = len(coeffs)
        n = len(self.recon)
        zoomed = self.view == "zoom"
        cam_scale = s.zoom_factor if zoomed else 1.0

        def w(width: float) -> float:  # constant on-screen stroke thickness
            return width / cam_scale

        tracker = ValueTracker(0.0)

        def tip_z(t: float) -> complex:
            return complex(np.sum(coeffs * np.exp(TAU * 1j * freqs * t)))

        def chain_positions(t: float) -> np.ndarray:
            terms = coeffs * np.exp(TAU * 1j * freqs * t)
            return np.concatenate([[0j], np.cumsum(terms)])

        recon_ext = np.append(self.recon, self.recon[0])
        runs = mask_runs(self.jump_mask)

        # -- ghost of the finished shape (normal sub-paths only)
        if s.ghost:
            ghost = VGroup()
            for a, b, jump in runs:
                if jump:
                    continue
                vm = VMobject(stroke_color=WHITE, stroke_width=w(1.5),
                              stroke_opacity=0.08)
                vm.set_points_as_corners(path_points(recon_ext[a:b + 1]))
                ghost.add(vm)
            self.add(ghost)

        # -- trace, revealed by t; jumps faint, normals glowing yellow
        trace = VGroup()
        for a, b, jump in runs:
            template = VMobject()
            template.set_points_as_corners(path_points(recon_ext[a:b + 1]))
            for width, opacity in (JUMP_LAYERS if jump else TRACE_LAYERS):
                piece = VMobject(stroke_color=YELLOW, stroke_width=w(width),
                                 stroke_opacity=opacity)
                updater = self._reveal_updater(template, a / n, b / n, tracker)
                piece.add_updater(updater)
                updater(piece)
                trace.add(piece)
        self.add(trace)

        # -- circles: visibility decided in camera space (pixels on screen);
        #    larger circles slightly brighter, like the 3b1b original
        mags = np.abs(coeffs)
        px_per_unit = config.pixel_width / (config.frame_width / cam_scale)
        visibility = np.clip((mags * px_per_unit - 2.0) / 2.0, 0.0, 1.0)
        brightness = 0.25 + 0.4 * np.sqrt(mags / max(mags.max(), 1e-12))
        circles = VGroup()
        circle_indices = []
        for k in range(n_vec):
            if visibility[k] <= 0.01 or mags[k] <= 1e-6:
                continue
            circles.add(Circle(radius=mags[k], stroke_color=CIRCLE_COLOR,
                               stroke_width=w(1.2),
                               stroke_opacity=brightness[k] * visibility[k],
                               fill_opacity=0))
            circle_indices.append(k)

        # -- vector chain: white arrows (shaft + filled head), widths tapering
        #    with coefficient magnitude
        widths = 0.5 + 2.0 * np.sqrt(mags / max(mags.max(), 1e-12))
        head_max = HEAD_LENGTH / cam_scale
        shafts = VGroup(*[
            Line(ORIGIN, RIGHT, stroke_color=VECTOR_COLOR,
                 stroke_width=w(widths[k]), stroke_opacity=0.95)
            for k in range(n_vec)
        ])
        heads = VGroup(*[
            VMobject(stroke_width=0, fill_color=VECTOR_COLOR, fill_opacity=0.95)
            for _ in range(n_vec)
        ])
        vectors = VGroup(shafts, heads)

        def update_kinematics(_) -> None:
            pos = chain_positions(tracker.get_value())
            for k in range(n_vec):
                a, b = pos[k], pos[k + 1]
                length = abs(b - a)
                if length < 1e-8:
                    shafts[k].put_start_and_end_on(c2p(a), c2p(a + 1e-8))
                    heads[k].set_points(np.zeros((0, 3)))
                    continue
                direction = (b - a) / length
                head = min(head_max, HEAD_LENGTH_RATIO * length)
                base = b - direction * head
                shafts[k].put_start_and_end_on(c2p(a), c2p(base))
                side = direction * 1j * head * HEAD_WIDTH_RATIO
                heads[k].set_points_as_corners(
                    np.array([c2p(b), c2p(base + side), c2p(base - side), c2p(b)]))
            for circle, k in zip(circles, circle_indices):
                circle.move_to(c2p(pos[k]))

        vectors.add_updater(update_kinematics)
        update_kinematics(vectors)
        self.add(circles, vectors)

        # -- zoom camera: locked on the pen tip (cam_lag 0, the default);
        #    a positive cam_lag gives an exponential-damping lazy follow
        frame = self.camera.frame
        if zoomed:
            frame.set(width=config.frame_width / cam_scale)
            frame.move_to(c2p(tip_z(0.0)))
            if s.cam_lag <= 0:
                frame.add_updater(
                    lambda f: f.move_to(c2p(tip_z(tracker.get_value()))))
            else:
                def follow(f, dt: float) -> None:
                    if dt <= 0:
                        return
                    blend = 1.0 - np.exp(-dt / s.cam_lag)
                    f.move_to((1 - blend) * f.get_center()
                              + blend * c2p(tip_z(tracker.get_value())))

                frame.add_updater(follow)
            self.add(frame)

        # -- corner label
        if s.label:
            label = Text(f"{n_vec} rotating vectors", font_size=28,
                         color=LABEL_COLOR)
            if zoomed:
                label.scale(1.0 / cam_scale)
                pad = 0.25 / cam_scale

                def pin(m) -> None:
                    m.move_to(frame.get_corner(DR)
                              + np.array([-(m.width / 2 + pad),
                                          m.height / 2 + pad, 0.0]))

                label.add_updater(pin)
                pin(label)
            else:
                label.to_corner(DR, buff=0.3)
            self.add(label)

        # -- timeline: one uniform-speed draw filling the whole duration.
        #    The ending sequence (hold, fade the machinery, hold on the
        #    result) only runs when hold/end-hold are requested, so by
        #    default the video is exactly `duration` seconds long.
        self.play(tracker.animate.set_value(1.0), run_time=s.duration,
                  rate_func=linear)
        if s.hold > 0 or s.end_hold > 0:
            if s.hold > 0:
                self.wait(s.hold)
            vectors.clear_updaters()
            fading = [g for g in (vectors, circles) if len(g) > 0]
            if fading:
                self.play(*[FadeOut(g) for g in fading], run_time=3)
            if s.end_hold > 0:
                self.wait(s.end_hold)
