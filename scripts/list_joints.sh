#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <urdf_path>"
  exit 1
fi

URDF_PATH="$1"
python src/render_urdf_multiview.py list-joints --urdf "$URDF_PATH"
