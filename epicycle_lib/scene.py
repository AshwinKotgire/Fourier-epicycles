"""EpicycleScene: one scene class, two views sharing one deterministic timeline.

The tip position at time t is a pure function of t, and the trace is revealed
from the precomputed reconstruction at the same t, so the "full" and "zoom"
renders stay frame-for-frame in sync for picture-in-picture editing.
"""
from __future__ import annotations

import numpy as np
from manim import (DR, ORIGIN, RIGHT, TAU, WHITE, Circle, Dot, FadeIn, FadeOut,
                   ImageMobject, Line, MovingCameraScene, Text, ValueTracker,
                   VGroup, VMobject, config, linear, rush_from, rush_into,
                   smooth)

from .color import contrast_hex, rgb_to_hex

# draw-speed easing presets (rush_into = ease-in, rush_from = ease-out)
EASE_FUNCS = {"none": linear, "in": rush_into, "out": rush_from, "inout": smooth}

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


def _fit_image(img, frac: float = 0.9) -> None:
    """Scale an ImageMobject to fill `frac` of the frame, centered, aspect kept."""
    img.height = config.frame_height * frac
    if img.width > config.frame_width * frac:
        img.width = config.frame_width * frac
    img.move_to(ORIGIN)


def slice_play_kwargs(sl: dict, run_time: float, ease, on_frame=None) -> dict:
    """run_time + rate_func that make this process render exactly frames
    [lo, hi) of the sequential draw, at bit-identical tracker values.

    Manim walks an animation over `np.arange(0, run_time, 1/fps)` and feeds
    each element to `rate_func(t / run_time)`, so the whole draw is just an
    indexable list of alphas. A slice re-plays a window of that list:

      * `alphas` is the exact global sequence the sequential run would use,
        so worker frame j reproduces global frame lo+j to the last bit.
      * the run_time is half a frame SHORT of count/fps. `np.arange` overshoots
        to count+1 elements for ~17% of counts (688 of the first 4000 at both
        15 and 60 fps), and an extra element at every seam is a duplicated
        frame — a visible hitch. Going half a step short always yields
        exactly `count`.
    """
    fps = config.frame_rate
    alphas = np.arange(0, run_time, 1.0 / fps) / run_time
    lo, count, last = sl["lo"], sl["hi"] - sl["lo"], len(alphas) - 1
    span = count - 0.5

    def rate(a: float) -> float:
        # a >= 1 is Animation.finish(), not a rendered frame: land on end_t
        # exactly, the way the unsliced draw leaves the tracker for the outro.
        if a >= 1.0:
            return ease(1.0)
        j = int(round(a * span))
        if on_frame is not None:
            on_frame(j)   # live progress for the parent's aggregate bar
        return ease(float(alphas[min(lo + j, last)]))

    return {"run_time": span / fps, "rate_func": rate}


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
    def __init__(self, coeffs, freqs, recon, jump_mask, settings, view,
                 trace_colors=None, slice_spec=None, progress_cb=None,
                 **kwargs):
        self.coeffs = np.asarray(coeffs)
        self.freqs = np.asarray(freqs, dtype=np.float64)
        self.recon = np.asarray(recon)
        self.jump_mask = np.asarray(jump_mask, dtype=bool)
        self.s = settings
        self.view = view
        # {"lo", "hi", "is_last"}: render only frames [lo, hi) of the draw, for
        # frame-range parallelism. None renders the whole timeline (the default).
        self.slice_spec = slice_spec
        # called with this slice's frame index as each frame is computed
        self.progress_cb = progress_cb
        # (N, 3) RGB per recon point, or None for a single themed color
        self.trace_colors = (np.asarray(trace_colors, dtype=np.float64)
                             if trace_colors is not None else None)
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
        circle_hex = s.circle_color or CIRCLE_COLOR
        vector_hex = s.vector_color or VECTOR_COLOR
        frame = self.camera.frame

        def w(width: float) -> float:  # constant on-screen stroke thickness
            return width / cam_scale

        tracker = ValueTracker(s.start_t)

        # -- vignette: soft radial gradient, brighter center / darker corners.
        #    Added first so everything else draws on top; in the zoom view it
        #    rides along with the camera frame.
        if s.vignette > 0 and not s.transparent:
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

        recon_ext = np.append(self.recon, self.recon[0])
        runs = mask_runs(self.jump_mask)

        # -- start reveal: a faint colored original as a reference, shown in
        #    the full view only (the zoom inset stays clean). Sits behind
        #    everything else.
        if s.start_reveal == "original" and s.source_image and not zoomed:
            ref = ImageMobject(s.source_image)
            _fit_image(ref, 0.9)
            ref.set_opacity(s.start_opacity)
            self.add(ref)

        # -- ghost of the finished shape (normal sub-paths only).
        #    ghost_mode: static = always visible; fadein = grows with progress;
        #    late = only appears in the last third (protects the reveal)
        ghost = VGroup()
        if s.ghost and s.ghost_opacity > 0 and s.start_reveal == "line":
            g_op = s.ghost_opacity
            for a, b, jump in runs:
                if jump:
                    continue
                vm = VMobject(stroke_color=WHITE, stroke_width=w(1.5 * s.trace_width),
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
        # trace_width scales every drawn-line stroke (core, halos, jumps,
        # keyline) so the whole line thins together and the glow stays in
        # proportion to the core it wraps.
        tw = s.trace_width
        core_w = TRACE_LAYERS[0][0] * tw
        trace_layers = [(TRACE_LAYERS[0][0] * tw, TRACE_LAYERS[0][1])] + [
            (wd * tw, min(1.0, op * s.trace_glow)) for wd, op in TRACE_LAYERS[1:]]
        jump_layers = [(wd * tw, min(1.0, op * s.jump_opacity))
                       for wd, op in JUMP_LAYERS]

        def add_reveal(template, s0, s1, color, width, opacity):
            piece = VMobject(stroke_color=color, stroke_width=w(width),
                             stroke_opacity=opacity)
            updater = self._reveal_updater(template, s0, s1, tracker)
            piece.add_updater(updater)
            updater(piece)
            trace.add(piece)

        def add_head(template, s0, s1, color):
            """The glowing comet head over one span, in that span's own hue."""
            for width, opacity in trace_layers:
                piece = VMobject(stroke_color=color, stroke_width=w(width),
                                 stroke_opacity=opacity)
                updater = self._head_updater(template, s0, s1, tracker,
                                             head_window)
                piece.add_updater(updater)
                updater(piece)
                trace.add(piece)

        if self.trace_colors is not None:
            # Colored trace: solid-color chunks along each non-jump run, each
            # chunk tinted by the mean sampled/gradient color for its span. A
            # contrasting keyline rim (optional) keeps any color legible.
            # With trace_fade > 0 each chunk settles dim and the comet head
            # rides over it in that chunk's own hue, so color and fade coexist.
            seg = max(1, n // max(1, s.color_segments))
            key_hex = contrast_hex(s.bg_color or BG_COLOR)
            for a, b, jump in runs:
                if jump:
                    if s.jump_opacity <= 0:
                        continue
                    template = VMobject()
                    template.set_points_as_corners(path_points(recon_ext[a:b + 1]))
                    for width, opacity in jump_layers:
                        add_reveal(template, a / n, b / n, trace_hex, width, opacity)
                    continue
                k = a
                while k < b:
                    k2 = min(b, k + seg)
                    col_hex = rgb_to_hex(self.trace_colors[k:k2].mean(axis=0))
                    template = VMobject()
                    template.set_points_as_corners(path_points(recon_ext[k:k2 + 1]))
                    s0, s1 = k / n, k2 / n
                    if s.keyline > 0:
                        add_reveal(template, s0, s1, key_hex, 6.0 * tw, s.keyline)
                    if fade <= 0:
                        add_reveal(template, s0, s1, col_hex, core_w, 1.0)
                        if s.trace_glow > 0:
                            add_reveal(template, s0, s1, col_hex, 9.0 * tw,
                                       min(1.0, 0.10 * s.trace_glow))
                    else:
                        add_reveal(template, s0, s1, col_hex, core_w,
                                   1.0 - 0.65 * fade)
                        add_head(template, s0, s1, col_hex)
                    k = k2
            self.add(trace)
        else:
            for a, b, jump in runs:
                if jump and s.jump_opacity <= 0:
                    continue
                template = VMobject()
                template.set_points_as_corners(path_points(recon_ext[a:b + 1]))
                if jump or fade <= 0:
                    for width, opacity in (jump_layers if jump else trace_layers):
                        add_reveal(template, a / n, b / n, trace_hex, width, opacity)
                    continue
                settled = VMobject(stroke_color=trace_hex,
                                   stroke_width=w(core_w),
                                   stroke_opacity=1.0 - 0.65 * fade)
                updater = self._reveal_updater(template, a / n, b / n, tracker)
                settled.add_updater(updater)
                updater(settled)
                trace.add(settled)
                add_head(template, a / n, b / n, trace_hex)
            self.add(trace)

        # -- circles: visibility decided in camera space (pixels on screen);
        #    larger circles slightly brighter, like the 3b1b original
        mags = np.abs(coeffs)
        mag_ratio = np.sqrt(mags / max(mags.max(), 1e-12))  # 1 -> 0 down the chain
        px_per_unit = config.pixel_width / (config.frame_width / cam_scale)
        visibility = np.clip((mags * px_per_unit - 2.0) / 2.0, 0.0, 1.0)
        brightness = np.minimum(1.0, (0.25 + 0.4 * mag_ratio) * s.circle_opacity)
        widths = (0.5 + 2.0 * mag_ratio) * s.arrow_width
        head_scale = s.arrow_head
        head_max = HEAD_LENGTH * head_scale / cam_scale
        # opacity tapers down the chain like the widths do, so the tiny fast
        # arrows near the pen fade instead of forming a bright cluster
        arrow_opacity = np.minimum(
            1.0, 0.95 * s.arrow_opacity * (0.45 + 0.55 * mag_ratio))

        # which epicycles to draw as circles/arrows.
        #
        # shown_vectors keeps only the N largest by magnitude and builds a
        # SHORTER chain from just those, so its last arrow lands on a partial
        # sum, not the pen; bridge_opacity draws the leftover gap (which can
        # be tens of pixels). arrow_min_px is the alternative: the chain stays
        # complete, so positions are exact, and we merely skip DRAWING the
        # segments too short to see, then close the chain with one final arrow
        # onto the true pen tip. Sub-pixel terms are individually invisible but
        # collectively move the pen, so that closing arrow is what keeps the
        # arrowhead and the drawn line in agreement.
        order = np.arange(n_vec)
        bridging = False
        truncating = False
        if s.shown_vectors is not None and s.shown_vectors < n_vec:
            order = np.sort(np.argsort(-mags)[:int(s.shown_vectors)])
            bridging = True
        elif s.arrow_min_px > 0:
            keep = np.nonzero(mags * px_per_unit > s.arrow_min_px)[0]
            if len(keep) and len(keep) < n_vec:
                order = keep
                truncating = True
        sub_coeffs = coeffs[order]
        sub_freqs = freqs[order]
        n_show = len(order)

        if truncating:
            # full chain, so position[p] is absolute and the tail beyond the
            # last drawn segment is closed by `bridge` below
            def chain_positions(t: float) -> np.ndarray:
                terms = coeffs * np.exp(TAU * 1j * freqs * t)
                pos = np.concatenate([[0j], np.cumsum(terms)])
                return pos[np.concatenate([order, [n_vec]])]
        else:
            def chain_positions(t: float) -> np.ndarray:
                terms = sub_coeffs * np.exp(TAU * 1j * sub_freqs * t)
                return np.concatenate([[0j], np.cumsum(terms)])

        circles = VGroup()
        circle_pos = []
        for p, k in enumerate(order):
            if visibility[k] <= 0.01 or mags[k] <= 1e-6:
                continue
            circles.add(Circle(
                radius=mags[k], stroke_color=circle_hex,
                stroke_width=w(1.2 * (0.5 + 0.5 * mag_ratio[k])
                               * s.circle_width),
                stroke_opacity=brightness[k] * visibility[k],
                fill_opacity=0))
            circle_pos.append(p)

        # -- vector chain: white arrows (shaft + filled head), widths tapering
        #    with coefficient magnitude
        shafts = VGroup(*[
            Line(ORIGIN, RIGHT, stroke_color=vector_hex,
                 stroke_width=w(widths[k]), stroke_opacity=arrow_opacity[k])
            for k in order
        ])
        heads = VGroup(*[
            VMobject(stroke_width=0, fill_color=vector_hex,
                     fill_opacity=arrow_opacity[k])
            for k in order
        ])
        bridge = VGroup()
        if bridging and s.bridge_opacity > 0:
            bridge.add(Line(ORIGIN, RIGHT, stroke_color=vector_hex,
                            stroke_width=w(0.8), stroke_opacity=s.bridge_opacity))
        vectors = VGroup(shafts, heads, bridge)

        def update_kinematics(_) -> None:
            pos = chain_positions(tracker.get_value())
            for p in range(n_show):
                a, b = pos[p], pos[p + 1]
                length = abs(b - a)
                if length < 1e-8:
                    shafts[p].put_start_and_end_on(c2p(a), c2p(a + 1e-8))
                    heads[p].set_points(np.zeros((0, 3)))
                    continue
                direction = (b - a) / length
                head = min(head_max, HEAD_LENGTH_RATIO * head_scale * length)
                base = b - direction * head
                shafts[p].put_start_and_end_on(c2p(a), c2p(base))
                side = direction * 1j * head * HEAD_WIDTH_RATIO
                heads[p].set_points_as_corners(
                    np.array([c2p(b), c2p(base + side), c2p(base - side), c2p(b)]))
            for circle, p in zip(circles, circle_pos):
                circle.move_to(c2p(pos[p]))
            if len(bridge) > 0:
                last = pos[-1]
                tipz = tip_z(tracker.get_value())
                if abs(tipz - last) < 1e-8:
                    bridge[0].set_points(np.zeros((0, 3)))
                else:
                    bridge[0].put_start_and_end_on(c2p(last), c2p(tipz))

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
                tip_dot.add(Dot(radius=radius * s.tip_size / cam_scale,
                                fill_color=color,
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
            label = Text(f"{n_show} rotating vectors", font_size=28,
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
        #
        # Under slice_spec this process renders only its window of the draw
        # and, unless it owns the last window, stops before the outro (which
        # is wall-clock, not t-driven, so it belongs to exactly one worker).
        ease_fn = EASE_FUNCS[s.ease]
        draw_run_time = s.duration * (s.end_t - s.start_t)
        if self.slice_spec is None:
            play_kwargs = {"run_time": draw_run_time, "rate_func": ease_fn}
            render_ending = True
        else:
            play_kwargs = slice_play_kwargs(self.slice_spec, draw_run_time,
                                            ease_fn, self.progress_cb)
            render_ending = self.slice_spec["is_last"]
        self.play(tracker.animate.set_value(s.end_t), **play_kwargs)
        if render_ending and (s.hold > 0 or s.end_hold > 0 or s.end_card > 0):
            if s.hold > 0:
                self.wait(s.hold)
            vectors.clear_updaters()
            tip_dot.clear_updaters()
            fading = [g for g in (vectors, circles, tip_dot) if len(g) > 0]
            if fading:
                self.play(*[FadeOut(g) for g in fading], run_time=3)
            if s.end_card > 0:
                # Outro shot, held for end_card seconds. end_reveal chooses:
                #   line     = scale the finished line drawing up to fill frame
                #   original = crossfade the drawing into the colored original
                #   none     = just settle and hold (reset zoom to the full view)
                # Identical timing in both views keeps full/zoom frame-synced.
                # spend the same 0.6s whether or not this view has a label, so
                # the timelines never drift (0.6s is a whole number of frames
                # at both 15 and 60 fps).
                if s.label:
                    self.play(FadeOut(label), run_time=0.6)
                else:
                    self.wait(0.6)
                drawing = VGroup(trace, ghost)
                if s.end_reveal == "original" and s.source_image:
                    img = ImageMobject(s.source_image)
                    _fit_image(img, 0.92)
                    anims = [FadeIn(img), FadeOut(drawing)]
                    if zoomed:
                        frame.clear_updaters()
                        anims.append(frame.animate.set(
                            width=config.frame_width).move_to(ORIGIN))
                    self.play(*anims, run_time=max(s.end_crossfade, 1e-3))
                    # Ken Burns: slow push-in / drift on the revealed image.
                    if (s.reveal_zoom > 1.0 or s.reveal_pan_x or s.reveal_pan_y):
                        shift = np.array([s.reveal_pan_x * config.frame_width * 0.1,
                                          s.reveal_pan_y * config.frame_height * 0.1,
                                          0.0])
                        self.play(img.animate.scale(s.reveal_zoom).shift(shift),
                                  run_time=max(s.end_card, 1e-3), rate_func=smooth)
                    else:
                        self.wait(s.end_card)
                elif s.end_reveal == "none":
                    if zoomed:
                        frame.clear_updaters()
                        self.play(frame.animate.set(
                            width=config.frame_width).move_to(ORIGIN), run_time=2)
                    else:
                        self.wait(2)
                    self.wait(s.end_card)
                else:  # line
                    anims = []
                    if zoomed:
                        frame.clear_updaters()
                        for vm in drawing.family_members_with_points():
                            vm.set_stroke(width=vm.get_stroke_width() * cam_scale)
                        anims.append(frame.animate.set(
                            width=config.frame_width).move_to(ORIGIN))
                    span_h = max(drawing.height, 1e-6)
                    span_w = max(drawing.width, 1e-6)
                    factor = min(0.9 * config.frame_height / span_h,
                                 0.95 * config.frame_width / span_w)
                    anims.append(drawing.animate.scale(factor).move_to(ORIGIN))
                    self.play(*anims, run_time=2)
                    self.wait(s.end_card)
            if s.end_hold > 0:
                self.wait(s.end_hold)
