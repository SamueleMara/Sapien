#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${IMAGE:-urdf-sapien-renderer:latest}"
WORKSPACE_DIR="${WORKSPACE_DIR:-$ROOT_DIR}"
ASSETS_DIR="${ASSETS_DIR:-$ROOT_DIR/assets}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/output}"

mkdir -p "$ASSETS_DIR" "$OUTPUT_DIR"

if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  echo "Image '$IMAGE' not found. Building it now..."
  docker build -t "$IMAGE" "$ROOT_DIR"
fi

docker run --rm -it \
  --gpus all \
  -e NVIDIA_DRIVER_CAPABILITIES=graphics,utility,compute,display \
  -e PYTHONUNBUFFERED=1 \
  -e QT_X11_NO_MITSHM=1 \
  -v "$WORKSPACE_DIR":/workspace \
  -v "$ASSETS_DIR":/data/assets \
  -v "$OUTPUT_DIR":/data/output \
  -w /workspace \
  "$IMAGE" \
  bash
