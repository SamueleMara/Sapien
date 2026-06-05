#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <urdf_path> <joint_name> [output_dir]"
  exit 1
fi

URDF_PATH="$1"
JOINT_NAME="$2"
OUTPUT_DIR="${3:-/data/output/demo}"

python src/render_urdf_multiview.py render \
  --urdf "$URDF_PATH" \
  --joint-name "$JOINT_NAME" \
  --q-start 0.0 \
  --q-end 1.0 \
  --num-frames 60 \
  --views-json configs/camera_views.example.json \
  --output-dir "$OUTPUT_DIR" \
  --save-video
