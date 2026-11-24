#!/bin/sh
# (Re)load v4l2loopback with the two boardvu virtual cameras. Needs sudo.
#   /dev/video10  "boardvu raw"    webcam passthrough
#   /dev/video11  "boardvu board"  the deskewed board
set -e
if lsmod | grep -q '^v4l2loopback'; then
    sudo modprobe -r v4l2loopback
fi
sudo modprobe v4l2loopback devices=2 video_nr=10,11 \
    card_label="boardvu raw","boardvu board" exclusive_caps=1,1 max_buffers=8
v4l2-ctl --list-devices | grep -A1 boardvu
