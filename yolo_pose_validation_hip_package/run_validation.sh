#!/usr/bin/env bash
set -euo pipefail

VIDEO="${1:-realsense_20260805_163743.mp4}"

python3 validate_yolo_pose.py "$VIDEO" \
  --models yolo11n-pose.pt yolo11s-pose.pt \
  --tracker botsort.yaml \
  --imgsz 640 \
  --conf 0.25 \
  --kpt-conf 0.35
