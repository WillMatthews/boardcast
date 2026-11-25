#!/usr/bin/env bash
# Export the YOLO pose model used to find your head for the picture-in-picture, to
# ${XDG_DATA_HOME:-~/.local/share}/boardcast/models/yolo11n-pose.onnx, where boardcast looks for it.
#
# Ultralytics' weights are AGPL-3.0, so they aren't committed here: this downloads them and exports ONNX with a
# dynamic input size, so boardcast can feed it the segmentation picture at its own size and shape (landscape or
# portrait), in a throwaway Python environment. (An older fixed-size export still works: boardcast letterboxes into
# it, at lower resolution.) Needs uv (https://docs.astral.sh/uv/). Without the model there's no picture-in-picture.
set -euo pipefail

dir="${XDG_DATA_HOME:-$HOME/.local/share}/boardcast/models"
out="$dir/yolo11n-pose.onnx"
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT

uv venv -q "$work/venv"
VIRTUAL_ENV="$work/venv" uv pip install -q "ultralytics==8.4.218" onnx onnxslim \
    --extra-index-url https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match
(cd "$work" && "$work/venv/bin/python" -I -c "
from ultralytics import YOLO
YOLO('yolo11n-pose.pt').export(format='onnx', imgsz=(288, 512), simplify=True, opset=17, dynamic=True)
")
mkdir -p "$dir"
mv "$work/yolo11n-pose.onnx" "$out"
echo "wrote $out"
