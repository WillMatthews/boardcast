#!/usr/bin/env bash
# Fetch MediaPipe's hand landmark model (Apache-2.0) and convert it to ONNX: models/hand_landmarks.onnx.
#
# Used by the pointer to find my index fingertip. MediaPipe publishes it as TFLite inside hand_landmarker.task;
# this converts it with tf2onnx in a throwaway Python environment (needs uv: https://docs.astral.sh/uv/).
# Without the model, the pointer follows pose's wrist instead (rougher).
set -euo pipefail

cd "$(dirname "$0")/.."
out=models/hand_landmarks.onnx
url=https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT

curl -sSfL -o "$work/hand_landmarker.task" "$url"
(cd "$work" && python3 -I -c "import zipfile; zipfile.ZipFile('hand_landmarker.task').extract('hand_landmarks_detector.tflite')")
uv venv -q -p 3.11 "$work/venv"
VIRTUAL_ENV="$work/venv" uv pip install -q "tf2onnx==1.16.1" "tensorflow-cpu==2.15.1" "onnx==1.16.2" "numpy<2"
"$work/venv/bin/python" -I -m tf2onnx.convert --tflite "$work/hand_landmarks_detector.tflite" \
    --output "$work/hand_landmarks.onnx" --opset 17 >/dev/null 2>&1
mkdir -p models
mv "$work/hand_landmarks.onnx" "$out"
echo "wrote $out"
