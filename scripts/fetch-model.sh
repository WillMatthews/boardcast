#!/bin/sh
# Download the person-segmentation model: Robust Video Matting, MobileNetV3, fp32 ONNX (~15 MB, GPL-3.0).
# https://github.com/PeterL1n/RobustVideoMatting
set -e
# Saved to ${XDG_DATA_HOME:-~/.local/share}/boardvu/models, where boardvu looks for it.
dir="${XDG_DATA_HOME:-$HOME/.local/share}/boardvu/models"
mkdir -p "$dir"
f="$dir/rvm_mobilenetv3_fp32.onnx"
curl -fL --progress-bar -o "$f.part" \
    https://github.com/PeterL1n/RobustVideoMatting/releases/download/v1.0.0/rvm_mobilenetv3_fp32.onnx
echo "88d4531297118f595bf2fd60f6f566aec2e559393802d1f436c380f0cbbd2828  $f.part" | sha256sum -c --quiet
mv "$f.part" "$f"
echo "ok: $f"
