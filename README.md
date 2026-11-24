# boardvu

A virtual camera of my blackboard. One process owns the real webcam and writes two v4l2loopback cameras:

| Device        | Name            | What                                                        |
|---------------|-----------------|-------------------------------------------------------------|
| `/dev/video10` | `boardvu raw`   | the webcam, passed through (1920×1080)                      |
| `/dev/video11` | `boardvu board` | the deskewed board at native resolution (about 710×948)     |

In **ghost mode** (the default) the board feed shows the board as it was last seen where I'm standing, with me
drawn faintly over it, so the writing stays readable when I'm in the way. See `PROJECT.md` for the plan.

## Setup: the virtual camera devices (once per boot)

boardvu writes into [v4l2loopback](https://github.com/v4l2loopback/v4l2loopback) devices, so the kernel module has
to be loaded first. Ubuntu ships it with the kernel (0.12.7 at the time of writing); otherwise install
`v4l2loopback-dkms`.

```sh
sudo modprobe v4l2loopback devices=2 video_nr=10,11 \
    card_label="boardvu raw","boardvu board" exclusive_caps=1,1 max_buffers=8
```

or `scripts/loopback.sh`, which reloads the module if it's already loaded. What the options do:

| Option | Why |
|---|---|
| `devices=2 video_nr=10,11` | two devices at fixed paths, `/dev/video10` (raw) and `/dev/video11` (board) |
| `card_label=...` | the names meeting apps show in their camera picker |
| `exclusive_caps=1,1` | **one value per device.** Devices advertise capture-only while boardvu is writing, which Chrome/Meet need to list them. They only appear in camera pickers while boardvu runs. |
| `max_buffers=8` | boardvu fills buffers in rotation, so readers get up to 7 frames of slack before a frame they're copying is overwritten (tearing). The default 2 gives one frame. |

To load it at every boot:

```sh
echo v4l2loopback | sudo tee /etc/modules-load.d/boardvu.conf
echo 'options v4l2loopback devices=2 video_nr=10,11 card_label="boardvu raw","boardvu board" exclusive_caps=1,1 max_buffers=8' \
    | sudo tee /etc/modprobe.d/boardvu.conf
```

**If a run fails with `setting YUYV ... Invalid argument`**, the driver is stuck. On 0.12.7 that happens if
something writes to an `exclusive_caps` device with plain `write()` calls (e.g. ffmpeg, or `boardvu --io write`) and
then exits. Reload the module with `scripts/loopback.sh`.

## Setup: the segmentation model (once)

```sh
scripts/fetch-model.sh   # models/rvm_mobilenetv3_fp32.onnx, Robust Video Matting (GPL-3.0)
```

## Setup: board light (optional)

boardvu can turn the board light on while it runs and off when it stops, through Home Assistant. Copy
`light.example.json` to `light.json` (gitignored) and fill in the URL, the light's entity id, and the `light.turn_on`
service data to use while running. The token is read from `HA_API_KEY` in the environment, or from a `HA_API_KEY=...`
line in `~/.tokens`. Without `light.json`, or with `--no-light`, the light is left alone.

## Use

Run it only when you need it: while it runs it holds `/dev/video0`, so nothing else (e.g. `~/coding/ml/learn/board.py`)
can use the camera. Ctrl-C to stop.

```sh
cargo run --release              # stream to /dev/video10 + /dev/video11
cargo run --release -- --no-raw  # board only
cargo run --release -- snap out/ # one frame -> out/raw.png (board quad outlined) + out/board.png
cargo run --release -- --no-ghost             # plain deskewed board, no segmentation
cargo run --release -- --ghost-opacity 0.4    # make me more visible
cargo run --release -- --look paper           # dark ink on white (also: natural, the default; plain)
```

`snap` averages 16 frames and writes every look: `raw.png`, `board.png`, `board_natural.png`, `board_paper.png`.

**Status display.** In a terminal, boardvu shows a full-screen status: the feeds (and whether anything is watching
them), the look and ghost, board tracking, frame rate and CPU, and a log. `--no-tui` (or running without a terminal)
prints plain log lines instead.

**Keys while streaming:** `l` cycles the look (plain → natural → paper), `g` steps me through hidden → ghost 0.1 → ghost 0.25 → ghost 0.5 → opaque (fully
shown, no ghosting). `p` toggles a picture-in-picture of me, `o` moves it to the next corner. `q`, Esc or Ctrl-C stops.

**Picture-in-picture.** A squircle crop of me, inset in a corner of the board feed (same resolution, so meeting
apps don't notice it coming and going). It follows my head using ghost mode's person matte (so it needs ghost
mode), framed about three head-widths wide so it looks the same at the board or at the desk, and fades out when I
leave the frame. `--pip` starts with it on; `--pip-size` (fraction of the feed's width, default 0.3) and
`--pip-corner` set it up.

### Looks

| Look | What |
|---|---|
| `natural` (default) | lighting flattened to an even board colour at a fixed brightness, chalk boosted, colours kept |
| `paper` | dark ink on white, chalk only (sticky notes and chalk-dust haze become paper); clearest in poor light |
| `plain` | the board as the camera sees it |

Chalk is found with a white top-hat (the image minus its morphological opening, which keeps only bright strokes
thinner than ~11 px), so big bright areas like sticky notes and haze aren't treated as ink. Lighting is a coarse
median estimate, refreshed twice a second, and divides out (gain capped at 3× so unlit corners don't turn to
noise). In ghost mode enhancement runs at ~10 Hz on the slowly changing board memory, and I'm drawn on top
every frame.

### How ghost mode works

- A worker thread runs Robust Video Matting on a 512×288 copy of the newest frame, at most 10 times a second
  (CPU, about 55 ms each), and publishes the newest matte; the video loop never waits for it.
- The matte is warped into board space, grown by 2.5% of the board height, and held for a few frames after I
  move, so the board memory never learns me.
- The board memory is a per-pixel running average, updated only where I'm not. Small differences update slowly,
  which denoises faint chalk. Big ones (new chalk, erasing) are only accepted once the pixel has been still for
  ~1/3 s, so a moving hand the matte missed is never absorbed, and then arrive in about half a second.
- Output: the memory, with the live frame blended in at `--ghost-opacity` where the matte says person.

The raw feed is only produced while another process has `/dev/video10` open (checked once a second), so it costs
nothing when unused. Typical CPU at 15 fps: about 2.2 cores (2.65 with the raw feed watched).

In the meeting, pick **boardvu board** (or **boardvu raw** for the normal camera view).

Other options: `--size WxH` (pad the board into a fixed canvas, e.g. 1920x1080; default is the board alone at its native
resolution), `--raw-size WxH`, `--focus auto|N`, `--cam`, `--raw`, `--board`, `--config`, `--seg-res`,
`--seg-ratio`, `--seg-threads`, `--seg-fps`, `--threads`, `--model`, `--light`.

## Calibration (`board.json`)

- `corners`: TL, TR, BR, BL of the writing surface (inner edge of the aluminium frame) in full-resolution
  (2560×1440) camera pixels.
- `rotate`: degrees to turn the camera picture clockwise to make it upright (0, 90, 180, 270), for a camera
  mounted on its side. Corners are always listed in the board's own order (its top-left first), so the board comes
  out upright; segmentation runs on the upright picture, and the raw feed is upright (portrait, e.g. 1080x1920).
- `focus`: locked `focus_absolute` (autofocus locks onto my head instead of the board).
- `aspect`: real board width/height. 0.75 is estimated from the square sticky notes. Measure it to be exact.

**Finding the board.** Along each edge of the current outline, boardvu looks across the edge for the strongest
step from dark board to bright frame, fits a line per edge with RANSAC (so sticky notes on the edge, a person in
front of it, or chalk near it are ignored), and intersects the lines for the corners.

```sh
cargo run --release -- calibrate --dry-run --dir out/   # find the board, write out/calib.png, change nothing
cargo run --release -- calibrate                        # ...and save the corners to board.json
cargo run --release -- calibrate --search 250           # the camera moved a lot (default search: 120 px)
cargo run --release -- calibrate --from "x,y x,y x,y x,y"  # start from a rough guess (TL TR BR BL)
cargo run --release -- calibrate --search 60 --from "x,y x,y x,y x,y"  # after turning the camera: set
                                                        # `rotate` first, give rough corners in board order
cargo run --release -- calibrate --set "x,y x,y x,y x,y"   # manual fallback: use exactly these corners
```

`calib.png` shows the new outline in green and the previous one in orange. It refuses to save an outline that can't be the board (a
corner outside the picture, not board-shaped, or an edge suddenly 1.6x longer or shorter), e.g. when part of the
board is out of view and a straight chalk line looks like an edge.

**While streaming** boardvu re-checks the edges every 2 s (average of 8 frames, ±15 px). If three checks in a row
agree that the board has moved more than 2 px, it re-aligns the warp, restarts the board memory, and saves the
new corners to `board.json`; the output size doesn't change, so meeting apps don't notice. A bigger move than
that prints a hint to run `calibrate`. `--no-track` turns this off.
