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
```

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
  (2560×1440) camera pixels. If the camera gets bumped, run `snap`, look at `raw.png`, and fix these.
  (AprilTag auto-calibration is build step 3.)
- `focus`: locked `focus_absolute` (autofocus locks onto my head instead of the board).
- `aspect`: real board width/height. 0.75 is estimated from the square sticky notes. Measure it to be exact.
