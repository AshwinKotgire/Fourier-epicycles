"""EpicycleScene: one scene class, two views sharing one deterministic timeline.

The tip position at time t is a pure function of t, and the trace is revealed
from the precomputed reconstruction at the same t, so the "full" and "zoom"
renders stay frame-for-frame in sync for picture-in-picture editing.
"""
from __future__ import annotations

import numpy as np
from manim import (DR, ORIGIN, RIGHT, TAU, WHITE, Circle, Dot, FadeOut,
                   ImageMobject, Line, MovingCameraScene, Text, ValueTracker,
                   VGroup, VMobject, config, linear)

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

    @staticmethod
    def _head_updater(template: VMobject, s0: float, s1: float,
                      tracker: ValueTracker, window: float):
        """Reveal only the last `window` (in global t) of the run — the
        glowing comet head that trails the pen when trace_fade is on."""
        span = max(s1 - s0, 1e-9)
        state = {"last": None}

        def update(mobject: VMobject) -> None:
            t = tracker.get_value()
            hi = float(np.clip((t - s0) / span, 0.0, 1.0))
            lo = float(np.clip((t - window - s0) / span, 0.0, 1.0))
            if (lo, hi) == state["last"]:
                return
            state["last"] = (lo, hi)
            if hi <= 0 or hi <= lo:
                mobject.set_points(np.zeros((0, 3)))
            else:
                mobject.pointwise_become_partial(template, lo, hi)

        return update

    def construct(self):
        s = self.s
        coeffs, freqs = self.coeffs, self.freqs
        n_vec = len(coeffs)
        n = len(self.recon)
        zoomed = self.view == "zoom"
        cam_scale = s.zoom_factor if zoomed else 1.0
        trace_hex = s.trace_hex
        frame = self.camera.frame

        def w(width: float) -> float:  # constant on-screen stroke thickness
            return width / cam_scale

        tracker = ValueTracker(s.start_t)

        # -- vignette: soft radial gradient, brighter center / darker corners.
        #    Added first so everything else draws on top; in the zoom view it
        #    rides along with the camera frame.
        if s.vignette > 0:
            v = float(s.vignette)
            h_res, w_res = 270, 480
            gy, gx = np.mgrid[0:h_res, 0:w_res]
            aspect = config.frame_width / config.frame_height
            nx = (gx / (w_res - 1) - 0.5) * 2 * aspect
            ny = (gy / (h_res - 1) - 0.5) * 2
            r = np.sqrt(nx ** 2 + ny ** 2) / np.sqrt(aspect ** 2 + 1)
            val = np.clip(22 + v * (14 - 30 * r), 0, 255).astype(np.uint8)
            bg_img = ImageMobject(np.stack([val] * 3, axis=-1))
            bg_img.stretch_to_fit_width(config.frame_width / cam_scale * 1.02)
            bg_img.stretch_to_fit_height(config.frame_height / cam_scale * 1.02)
            if zoomed:
                bg_img.add_updater(lambda m: m.move_to(frame.get_center()))
            self.add(bg_img)

        def tip_z(t: float) -> complex:
            return complex(np.sum(coeffs * np.exp(TAU * 1j * freqs * t)))

        def chain_positions(t: float) -> np.ndarray:
            terms = coeffs * np.exp(TAU * 1j * freqs * t)
            return np.concatenate([[0j], np.cumsum(terms)])

        recon_ext = np.append(self.recon, self.recon[0])
        runs = mask_runs(self.jump_mask)

        # -- ghost of the finished shape (normal sub-paths only).
        #    ghost_mode: static = always visible; fadein = grows with progress;
        #    late = only appears in the last third (protects the reveal)
        ghost = VGroup()
        if s.ghost and s.ghost_opacity > 0:
            g_op = s.ghost_opacity
            for a, b, jump in runs:
                if jump:
                    continue
                vm = VMobject(stroke_color=WHITE, stroke_width=w(1.5),
                              stroke_opacity=g_op)
                vm.set_points_as_corners(path_points(recon_ext[a:b + 1]))
                ghost.add(vm)
            if s.ghost_mode != "static":
                def ghost_opacity(t: float) -> float:
                    if s.ghost_mode == "fadein":
                        return g_op * t
                    return g_op * float(np.clip((t - 2 / 3) / 0.1, 0.0, 1.0))

                def update_ghost(g: VGroup) -> None:
                    g.set_stroke(opacity=ghost_opacity(tracker.get_value()))

                ghost.add_updater(update_ghost)
                update_ghost(ghost)
            self.add(ghost)

        # -- trace, revealed by t; jumps faint, normals glowing yellow.
        #    With trace_fade > 0 the settled path dims and only a comet head
        #    (the most recent ~8% of the timeline) keeps the full glow.
        trace = VGroup()
        fade = float(np.clip(s.trace_fade, 0.0, 1.0))
        head_window = 0.08
        # core layer at full strength; the halo layers scale with trace_glow,
        # jump lines with jump_opacity
        trace_layers = [TRACE_LAYERS[0]] + [
            (wd, min(1.0, op * s.trace_glow)) for wd, op in TRACE_LAYERS[1:]]
        jump_layers = [(wd, min(1.0, op * s.jump_opacity))
                       for wd, op in JUMP_LAYERS]
        for a, b, jump in runs:
            if jump and s.jump_opacity <= 0:
                continue
            template = VMobject()
            template.set_points_as_corners(path_points(recon_ext[a:b + 1]))
            if jump or fade <= 0:
                for width, opacity in (jump_layers if jump else trace_layers):
                    piece = VMobject(stroke_color=trace_hex, stroke_width=w(width),
                                     stroke_opacity=opacity)
                    updater = self._reveal_updater(template, a / n, b / n,
                                                   tracker)
                    piece.add_updater(updater)
                    updater(piece)
                    trace.add(piece)
                continue
            settled = VMobject(stroke_color=trace_hex,
                               stroke_width=w(TRACE_LAYERS[0][0]),
                               stroke_opacity=1.0 - 0.65 * fade)
            updater = self._reveal_updater(template, a / n, b / n, tracker)
            settled.add_updater(updater)
            updater(settled)
            trace.add(settled)
            for width, opacity in trace_layers:
                piece = VMobject(stroke_color=trace_hex, stroke_width=w(width),
                                 stroke_opacity=opacity)
                updater = self._head_updater(template, a / n, b / n, tracker,
                                             head_window)
                piece.add_updater(updater)
                updater(piece)
                trace.add(piece)
        self.add(trace)

        # -- circles: visibility decided in camera space (pixels on screen);
        #    larger circles slightly brighter, like the 3b1b original
        mags = np.abs(coeffs)
        mag_ratio = np.sqrt(mags / max(mags.max(), 1e-12))  # 1 -> 0 down the chain
        px_per_unit = config.pixel_width / (config.frame_width / cam_scale)
        visibility = np.clip((mags * px_per_unit - 2.0) / 2.0, 0.0, 1.0)
        brightness = np.minimum(1.0, (0.25 + 0.4 * mag_ratio) * s.circle_opacity)
        circles = VGroup()
        circle_indices = []
        for k in range(n_vec):
            if visibility[k] <= 0.01 or mags[k] <= 1e-6:
                continue
            circles.add(Circle(
                radius=mags[k], stroke_color=CIRCLE_COLOR,
                stroke_width=w(1.2 * (0.5 + 0.5 * mag_ratio[k])
                               * s.circle_width),
                stroke_opacity=brightness[k] * visibility[k],
                fill_opacity=0))
            circle_indices.append(k)

        # -- vector chain: white arrows (shaft + filled head), widths tapering
        #    with coefficient magnitude
        widths = (0.5 + 2.0 * mag_ratio) * s.arrow_width
        head_scale = s.arrow_head
        head_max = HEAD_LENGTH * head_scale / cam_scale
        # opacity tapers down the chain like the widths do, so the tiny fast
        # arrows near the pen fade instead of forming a bright cluster
        arrow_opacity = np.minimum(
            1.0, 0.95 * s.arrow_opacity * (0.45 + 0.55 * mag_ratio))
        shafts = VGroup(*[
            Line(ORIGIN, RIGHT, stroke_color=VECTOR_COLOR,
                 stroke_width=w(widths[k]), stroke_opacity=arrow_opacity[k])
            for k in range(n_vec)
        ])
        heads = VGroup(*[
            VMobject(stroke_width=0, fill_color=VECTOR_COLOR,
                     fill_opacity=arrow_opacity[k])
            for k in range(n_vec)
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
                head = min(head_max, HEAD_LENGTH_RATIO * head_scale * length)
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

        # -- pen-tip highlight: a bright core with a themed halo, so the eye
        #    always has a focal point at the pen
        tip_dot = VGroup()
        if s.tip_glow > 0:
            for radius, opacity, color in ((0.20, 0.12, trace_hex),
                                           (0.10, 0.30, trace_hex),
                                           (0.03, 1.00, WHITE)):
                tip_dot.add(Dot(radius=radius / cam_scale, fill_color=color,
                                fill_opacity=min(1.0, opacity * s.tip_glow),
                                stroke_width=0))

            def update_tip(m: VGroup) -> None:
                m.move_to(c2p(tip_z(tracker.get_value())))

            tip_dot.add_updater(update_tip)
            update_tip(tip_dot)
            self.add(tip_dot)

        # -- zoom camera: locked on the pen tip (cam_lag 0, the default);
        #    a positive cam_lag gives an exponential-damping lazy follow.
        #    zoom_start eases the magnification from zoom_start to zoom_factor
        #    across the draw (strokes render a touch thinner early on)
        if zoomed:
            z_end = s.zoom_factor
            z_start = s.zoom_start if s.zoom_start else z_end

            def cam_width(t: float) -> float:
                return config.frame_width / (z_start + (z_end - z_start) * t)

            frame.set(width=cam_width(s.start_t))
            frame.move_to(c2p(tip_z(s.start_t)))
            if s.cam_lag <= 0:
                def lock(f) -> None:
                    t = tracker.get_value()
                    f.set(width=cam_width(t))
                    f.move_to(c2p(tip_z(t)))

                frame.add_updater(lock)
            else:
                def follow(f, dt: float) -> None:
                    if dt <= 0:
                        return
                    t = tracker.get_value()
                    blend = 1.0 - np.exp(-dt / s.cam_lag)
                    f.set(width=cam_width(t))
                    f.move_to((1 - blend) * f.get_center()
                              + blend * c2p(tip_z(t)))

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

        # -- timeline: one uniform-speed draw. start_t/end_t render just a
        #    slice at the same speed (for tuning long draws). The ending
        #    sequence (hold, fade the machinery, end card, hold) only runs
        #    when requested, so by default the video is exactly the slice's
        #    share of `duration` seconds long.
        self.play(tracker.animate.set_value(s.end_t),
                  run_time=s.duration * (s.end_t - s.start_t),
                  rate_func=linear)
        if s.hold > 0 or s.end_hold > 0 or s.end_card > 0:
            if s.hold > 0:
                self.wait(s.hold)
            vectors.clear_updaters()
            tip_dot.clear_updaters()
            fading = [g for g in (vectors, circles, tip_dot) if len(g) > 0]
            if fading:
                self.play(*[FadeOut(g) for g in fading], run_time=3)
            if s.end_card > 0:
                # scale the finished drawing up to fill the frame and hold —
                # a natural outro shot. Identical timing in both views keeps
                # the full/zoom renders frame-synced.
                # spend the same time whether or not this view has a label,
                # so the full/zoom timelines never drift apart (0.6s is a
                # whole number of frames at both 15 and 60 fps)
                if s.label:
                    self.play(FadeOut(label), run_time=0.6)
                else:
                    self.wait(0.6)
                anims = []
                if zoomed:
                    frame.clear_updaters()
                    for vm in VGroup(trace, ghost).family_members_with_points():
                        vm.set_stroke(width=vm.get_stroke_width() * cam_scale)
                    anims.append(frame.animate.set(
                        width=config.frame_width).move_to(ORIGIN))
                drawing = VGroup(trace, ghost)
                span_h = max(drawing.height, 1e-6)
                span_w = max(drawing.width, 1e-6)
                factor = min(0.9 * config.frame_height / span_h,
                             0.95 * config.frame_width / span_w)
                anims.append(drawing.animate.scale(factor).move_to(ORIGIN))
                self.play(*anims, run_time=2)
                self.wait(s.end_card)
            if s.end_hold > 0:
                self.wait(s.end_hold)
