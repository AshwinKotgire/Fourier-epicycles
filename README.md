# Epicycles

3Blue1Brown-style Fourier epicycle videos: rotating circles draw a piece of
line art, then (optionally) the drawing soaks into the colored original.

Everything runs inside the `manim` conda env (setup recipe:
`environment-setup.sh`). Run all commands from the repo root:

```bash
cd ~/epicycles
conda activate manim
```

---

## 1. Make a new project

You need the **line art** (dark lines on a light background, PNG) and,
optionally, the **colored original** it was made from. They do not have to
line up exactly.

```bash
# best starting point: copy a tuned project's look (here: mahadevi)
python epicycles.py init durga --image path/to/durga_lineart.png \
                               --original path/to/durga.jpeg \
                               --from projects/mahadevi

# or start from the generic, fully documented template
python epicycles.py init durga --image path/to/durga_lineart.png \
                               --original path/to/durga.jpeg
```

This creates `projects/durga/` with both images copied in and a
`config.yaml`.

| `init` option | What it does |
|---|---|
| `--image` | line art to draw (copied into the folder) |
| `--original` | colored image the drawing turns into at the end; switches on the bleed reveal |
| `--from projects/X` | start from project X's tuned config instead of the template: its whole look carries over, only `image` / `source_image` point at your new files |

Without `--from`, the config is the **complete template**: every setting is
listed with a comment, the less common ones commented out. With `--from`,
it is a copy of that project's config, which only lists what that project
changed. Anything not listed uses the engine default. Any config made
without `--from` doubles as the reference for every key.

---

## 2. The workflow: check → preview → tune → final

### Step 1: check the extraction (seconds)

```bash
python epicycles.py projects/durga --preview
```

Without `--yes` it stops after tracing and writes `projects/durga/inspect.png`:
left = the contours it found (red = jumps between strokes), right = what the
epicycles will actually draw. Read the two numbers at the bottom:

- **recon error**: under ~1px at 1080p is sharp; over 3px looks soft →
  raise `vectors` (and `samples` = 10 × vectors).
- **jumps**: share of the path spent jumping between strokes; keep under 15%.

Check the face. If the eyes or pupils are missing, lower `min_area` (faces
need `0.0`). Answer `n` to stop, edit the config, run again: tracing is cached,
so this loop is fast.

### Step 2: quick preview (minutes)

```bash
python epicycles.py projects/durga --preview --yes --duration 30
```

480p at 15 fps, with the draw squeezed into 30 s. Watch
`projects/durga/<image>_pip.mp4`. Drop `--duration` to preview at the real
pace.

Look at just one part of the timeline with `--start-t` / `--end-t` (0..1):

```bash
python epicycles.py projects/durga --preview --yes --start-t 0.95     # just the ending
python epicycles.py projects/durga --preview --yes --start-t 0.4 --end-t 0.45
```

### Step 3: tune

Try a value without editing anything. Most settings have a matching flag
(`trace_width` → `--trace-width`; `python epicycles.py --help` lists them):

```bash
python epicycles.py projects/durga --preview --yes --trace-width 0.3 --chain-fade 0.7
```

When you like it, write the value into `config.yaml` so it sticks. The
ending (`end_reveal`, `source_image`, `end_crossfade`), `gradient_colors`,
`bg_color`, `ease`, `label`, titles, audio and shorts are config-only. See the
tuning guide below.

### Step 4: final render

```bash
python epicycles.py projects/durga --yes
```

1080p at 60 fps. Outputs land in the project folder:

| File | What |
|---|---|
| `<image>_pip.mp4` | **the deliverable**: full frame with the zoomed pen as an inset |
| `<image>_full.mp4` / `<image>_zoom.mp4` | the two views separately (frame-synced, for editing) |
| `inspect.png`, `timeline.json` | extraction check; when each stroke is drawn |

With `frame_workers: 16` a ~150 s, 8000-vector draw takes on the order of
half an hour on this machine. The ending always renders as one piece, so it
sets a floor.

**Only one render per project folder at a time.** Two runs write the same
files.

---

## 3. Tuning guide

### Detail (what gets drawn)

| Setting | Effect | Good values |
|---|---|---|
| `min_area` | drops strokes smaller than this fraction of the image | `0.0` for faces (pupils!); `0.00003`+ to drop hatching/texture |
| `vectors` / `samples` | how faithful the drawing is (see recon error) | dense art: 8000 / 80000 |
| `duration` | seconds for the whole draw | 120–180 for detailed art |
| `fill` | drawing height as a fraction of the frame | 0.8–0.9 |
| `aspect` | `"16:9"` YouTube, `"1:1"`, `"4:5"`, `"9:16"` Shorts. **Keep the quotes** | |

### The line

| Setting | Effect | Good values |
|---|---|---|
| `trace_width` | line thickness. Dense art needs thin lines or strokes merge | 0.3–0.45 dense, 1.0 simple |
| `trace_glow` / `bloom` | glow around the line; too much smears fine detail | 0.3 / 0.2 dense |
| `trace_source: gradient` + `gradient_colors` | line color changes over the draw | `["#FFD54A", "#FF9933", "#E53935"]` gold→saffron→kumkum |
| `theme` | single color + pen-tip/jump color | `saffron`, `lotus`, `krishna`, `deep-red`, `violet`, `classic`… |
| `tip_size` / `tip_glow` | the glowing pen dot (only while drawing) | 0.4 / 0.5 for dense art |

### The circles and arrows (the show)

| Setting | Effect | Good values |
|---|---|---|
| `circle_color` | `"#58C4DD"` is the 3Blue1Brown blue | |
| `circle_opacity` / `circle_width` | circle brightness / thickness | 1.4 / 1.5 |
| `arrow_opacity` / `arrow_width` / `arrow_head` | arrow brightness / shaft / head | 1.2 / 0.9 / 0.8 |
| `chain_order` | `magnitude` = biggest circle first, shrinking toward the pen (cleaner on dense art); `frequency` = classic 3b1b order | `magnitude` |
| `chain_fade` | circles dim steadily toward the pen; the tail keeps `1 − chain_fade`, so it never detaches | 0.85 (0.7 = brighter tail, 0.95 = almost gone) |
| `arrow_min_px` | skip drawing circles/arrows smaller than this many pixels. The pen stays exact either way | 0.5 (0 = draw all; ~3× slower, looks the same) |

Dense art (hundreds of strokes) never gets the "few big circles" look. Its
detail is spread over thousands of similar-sized terms. `chain_order:
magnitude` + `chain_fade` is the best look for it. Avoid `shown_vectors` if
the arrow must sit on the pen: it shortens the chain, so the arrows end
somewhere else and a faint `bridge_opacity` line covers the gap.

### The ending

| Setting | Effect |
|---|---|
| `source_image` | the colored original (path relative to the line art) |
| `end_reveal` | `original` = turn into source_image · `line` = scale the drawing up · `none` |
| `end_transition` | `bleed` = color soaks outward from the lines · `dissolve` = grainy fade · `crossfade` = plain fade |
| `reveal_from` | `last_frame` = start straight off the final drawn frame · `clean` = fade the circles first (3 s) |
| `end_crossfade` | seconds for the transition (bleed looks best at 3–5) |
| `end_card` | seconds to hold on the result afterwards |
| `reveal_zoom` / `reveal_pan_x` / `reveal_pan_y` | slow Ken Burns push-in / drift during the hold (1.06 = gentle) |
| `end_hold` | extra seconds at the very end |

### Background and view

| Setting | Effect |
|---|---|
| `bg_color` | frame background, e.g. `"#140808"` dark maroon |
| `vignette` | radial darkening. **It paints a neutral grey backdrop over `bg_color`**, so set `0` if you want your background color to show |
| `view` | `pip` (deliverable) · `full` · `zoom` · `both` |
| `zoom_factor` / `zoom_start` | inset magnification, eased from `zoom_start` to `zoom_factor` |
| `frame_workers` | render the draw in N parallel processes (needs `cam_lag: 0`) |
| `transparent` | `.mov` with alpha for overlaying in an editor (`view: full`) |

### Extras

`intro_title` / `outro_title` (+ `_subtitle`, `_seconds`): title cards. The
font has no Devanagari, so use Latin text. `audio: track.mp3` muxes a
track. `shorts: true` also writes a 1080×1920 vertical crop
(`shorts_start`/`shorts_end` pick the window).

---

## 4. Several images in one go

**Series of projects (recommended):** keep one project folder per image under
a parent folder (projects can be nested), each fully tunable on its own, then
render them all with a loop. `projects/navaratri/` is set up this way; see
its `README.md`:

```bash
for d in projects/navaratri/*/; do python epicycles.py "$d" --yes; done
```

**Batch file:** one shared look in a single YAML, no per-image configs.

`epicycles_batch.py` renders a list of images with one shared look:

```bash
python epicycles_batch.py configs/navaratri_batch.yaml --yes
```

```yaml
defaults:            # any config key, applied to every job
  preview: false     # batch has no --preview flag: set true here to test
  frame_workers: 16  # must be a number here (0 = auto works only in epicycles.py)
  chain_fade: 0.85
  # ...
jobs:                # one entry per image; keys override defaults
  - image: projects/mahadevi/Orignal_images/mahadevi_lineart.png  # from repo root
    source_image: mahadevi.jpeg                                   # next to the image
    output: projects/mahadevi                                     # give each job its own
    min_area: 0.0
concat: false        # true = also stitch every job into one video
```

`configs/navaratri_batch.yaml` is a ready example. Tune each image as its
own project first (sections 1–3), then batch the final renders.

---

## Cheat sheet

```bash
conda activate manim                                             # once per terminal
python epicycles.py init NAME --image LINEART --original COLOR --from projects/mahadevi
python epicycles.py projects/NAME --preview                      # check inspect.png
python epicycles.py projects/NAME --preview --yes --duration 30  # quick look
python epicycles.py projects/NAME --preview --yes --start-t 0.95 # just the ending
python epicycles.py projects/NAME --yes                          # final 1080p60
python epicycles.py --help                                       # every flag
```
