# boardvu

A virtual camera of my blackboard. One process owns the real webcam and writes two v4l2loopback cameras:

| Device        | Name            | What                                                        |
|---------------|-----------------|-------------------------------------------------------------|
| `/dev/video10` | `boardvu raw`   | the webcam, passed through (1920×1080)                      |
| `/dev/video11` | `boardvu board` | the deskewed board at native resolution (about 710×948)     |

See `PROJECT.md` for the plan. This is build step 1 (capture → deskew → virtual cameras).

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

## Use

Run it only when you need it: while it runs it holds `/dev/video0`, so nothing else (e.g. `~/coding/ml/learn/board.py`)
can use the camera. Ctrl-C to stop.

```sh
cargo run --release              # stream to /dev/video10 + /dev/video11
cargo run --release -- --no-raw  # board only
cargo run --release -- snap out/ # one frame -> out/raw.png (board quad outlined) + out/board.png
```

In the meeting, pick **boardvu board** (or **boardvu raw** for the normal camera view).

Options: `--size WxH` (pad the board into a fixed canvas, e.g. 1920x1080; default is the board alone at its native
resolution), `--raw-size WxH`, `--focus auto|N`, `--cam`, `--raw`, `--board`, `--config`.

## Calibration (`board.json`)

- `corners`: TL, TR, BR, BL of the writing surface (inner edge of the aluminium frame) in full-resolution
  (2560×1440) camera pixels. If the camera gets bumped, run `snap`, look at `raw.png`, and fix these.
  (AprilTag auto-calibration is build step 3.)
- `focus`: locked `focus_absolute` (autofocus locks onto my head instead of the board).
- `aspect`: real board width/height. 0.75 is estimated from the square sticky notes. Measure it to be exact.
