#!/usr/bin/env bash
# Export the YOLO pose model used to find your head for the picture-in-picture, to
# ${XDG_DATA_HOME:-~/.local/share}/boardvu/models/yolo11n-pose.onnx, where boardvu looks for it.
#
# Ultralytics' weights are AGPL-3.0, so they aren't committed here: this downloads them and exports ONNX at the
# input size boardvu uses (512x288, the upright camera picture scaled down), in a throwaway Python environment.
# Needs uv (https://docs.astral.sh/uv/). Without the model, the picture-in-picture follows the person matte.
set -euo pipefail

dir="${XDG_DATA_HOME:-$HOME/.local/share}/boardvu/models"
out="$dir/yolo11n-pose.onnx"
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT

uv venv -q "$work/venv"
VIRTUAL_ENV="$work/venv" uv pip install -q "ultralytics==8.4.218" onnx onnxslim \
    --extra-index-url https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match
(cd "$work" && "$work/venv/bin/python" -I -c "
from ultralytics import YOLO
YOLO('yolo11n-pose.pt').export(format='onnx', imgsz=(512, 288), simplify=True, opset=17, dynamic=False)
")
mkdir -p "$dir"
mv "$work/yolo11n-pose.onnx" "$out"
echo "wrote $out"
