#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${IMAGE:-urdf-sapien-renderer:latest}"
WORKSPACE_DIR="${WORKSPACE_DIR:-$ROOT_DIR}"
ASSETS_DIR="${ASSETS_DIR:-$ROOT_DIR/assets}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/output}"

if [[ -z "${DISPLAY:-}" ]]; then
  echo "DISPLAY is not set. Start this from a graphical session (X11/XWayland)."
  exit 1
fi

if ! command -v xhost >/dev/null 2>&1; then
  echo "xhost is required for GUI mode. Install package: x11-xserver-utils"
  exit 1
fi

mkdir -p "$ASSETS_DIR" "$OUTPUT_DIR"

if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  echo "Image '$IMAGE' not found. Building it now..."
  docker build -t "$IMAGE" "$ROOT_DIR"
fi

xhost +si:localuser:root >/dev/null
cleanup() {
  xhost -si:localuser:root >/dev/null || true
}
trap cleanup EXIT

docker run --rm -it \
  --gpus all \
  -e DISPLAY="$DISPLAY" \
  -e NVIDIA_DRIVER_CAPABILITIES=graphics,utility,compute,display \
  -e PYTHONUNBUFFERED=1 \
  -e QT_X11_NO_MITSHM=1 \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  -v "$WORKSPACE_DIR":/workspace \
  -v "$ASSETS_DIR":/data/assets \
  -v "$OUTPUT_DIR":/data/output \
  -w /workspace \
  "$IMAGE" \
  bash
