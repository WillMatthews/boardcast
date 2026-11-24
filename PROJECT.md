# boardcast

## The idea

Have a virtual webcam (teed off of my webcam) that can see my blackboard.
Crop out just the blackboard using de-skewing.
When I write on it - show only a "ghost" of me over the writing (so you can read writing on the board when I am blocking the view).
This should be another "virtual camera" that I can share to a meeting.

## Scope

- My green chalkboard (portrait, aluminium frame, sticky notes down the right edge), one webcam
  (Anker PowerConf C200, `/dev/video0`, 2560×1440 MJPEG), used live on video calls.
- Run on demand: start boardcast for a call, stop it afterwards. While it runs it owns `/dev/video0`;
  other tools (e.g. `~/coding/ml/learn/board.py`) use the camera when boardcast is stopped.
- Linux only (v4l2loopback). Written in Rust.

## Outputs

One process owns the real webcam and writes two v4l2loopback devices:

1. **Raw** - passthrough of the webcam, so meetings can still use it as a normal camera.
2. **Board** - the deskewed, cleaned blackboard with my ghost over it.

## Core features

- **Deskew / crop** - perspective-warp the whole board (to the inner edge of the frame, sticky notes included)
  to a flat rectangle. Starting calibration in `board.json` (full-res pixel corners TL, TR, BR, BL, plus a
  locked focus value: autofocus locks onto my head, not the board).
- **AprilTag auto-calibration** - 4 printed AprilTag markers on the board corners; corners are re-detected
  continuously so a bumped camera self-corrects. Manual 4-click calibration as a fallback.
- **Person segmentation** - per-pixel mask of me (ONNX model, e.g. MediaPipe selfie / RVM).
- **Persistent board memory** - a running "clean board" image, updated only where the mask says
  I'm not standing. Where I occlude the board, show the last-known ink.
- **Ghost** - draw me over the remembered board at ~20-30% opacity (tunable).
- **Lighting correction / chalk enhancement** - flatten uneven lighting (the board light makes a hot spot at
  the top) and chalk-dust smear, then boost chalk contrast. Two looks: natural (green board, boosted chalk)
  and "paper" (dark ink on white). Proven recipe in `~/coding/ml/learn/board.py` `enhance()`: subtract a
  large median-blur background, CLAHE, soft percentile threshold.
- **Temporal denoise** - average static board pixels across frames to sharpen a cheap webcam. Chalk is faint, so
  this matters; frame averaging beat learned super-resolution in `~/coding/ml/learn/experiments/sr/` (GAN models
  altered faint marks).

## Extras

- **Triggered snapshots** - no automatic snapshots. A snapshot is taken only when I "do something"
  on the board: hold my hand over a small hotspot drawn in a board corner for ~1s
  (dwell detection via the person mask). Saves a PNG of the clean board memory (no ghost).
  Brief on-screen flash in the board feed to confirm.
- **OCR** - optional, on snapshot: send the PNG to OpenAI `gpt-6-luna` (Responses API, image input)
  and save Markdown / LaTeX alongside the PNG. Async so it never stalls the video loop.
- **Face picture-in-picture** - small cut-out of me in a corner of the board feed (reuses the mask).
- **Auto-zoom** - gently pan/zoom toward where new ink is appearing.
- **Pointer highlight** - soft spotlight at my hand / marker tip.

## Build order

1. ~~Capture -> deskew (manual corners) -> board virtual cam + raw passthrough.~~ Done.
2. Segmentation + board memory + ghost. (Board memory as a running average also gives the temporal denoise.)
3. ~~AprilTag calibration.~~ Done without tags: edge detection on the board's own frame (`calibrate`, live tracking).
4. ~~Lighting correction / chalk enhancement + denoise.~~ Done (looks: natural, paper, plain).
5. Triggered snapshots + OCR.
6. PiP, auto-zoom, pointer.

## Stack (no OpenCV)

- Capture + virtual cam output: `v4l` (v4l2loopback devices)
- Decode: `zune-jpeg` (MJPEG) / hand-rolled YUYV
- Warp, blur, blending: `image` + `imageproc` / small hand-written kernels, `rayon` for parallelism
- Segmentation: `ort` (ONNX Runtime)
- Markers: `apriltag` crate (bindings to the small C AprilTag library)
- OCR: `reqwest` + `tokio`
