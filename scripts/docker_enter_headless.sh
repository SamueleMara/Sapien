#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${IMAGE:-urdf-sapien-renderer:latest}"
WORKSPACE_DIR="${WORKSPACE_DIR:-$ROOT_DIR}"
# Resolve external mounts unless the caller supplied BOTH. No local fallback:
# that used to disguise errors and risk writing multi-GB renders to the NVMe.
if [[ -z "${ASSETS_DIR:-}" || -z "${OUTPUT_DIR:-}" ]]; then
  source "$ROOT_DIR/scripts/use_external_ssd.sh" || exit 1
fi
ASSETS_DIR="${ASSETS_DIR:?ASSETS_DIR must be a real, existing assets path}"
OUTPUT_DIR="${OUTPUT_DIR:?OUTPUT_DIR must be a real, existing output path}"

source "$ROOT_DIR/scripts/docker_mount_helpers.sh"
sapien_validate_mount_paths "$ASSETS_DIR" "$OUTPUT_DIR"

if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  echo "Image '$IMAGE' not found. Building it now..."
  docker build -t "$IMAGE" "$ROOT_DIR"
fi

docker run --rm -it \
  --gpus all \
  -e NVIDIA_DRIVER_CAPABILITIES=graphics,utility,compute,display \
  -e PYTHONUNBUFFERED=1 \
  -e QT_X11_NO_MITSHM=1 \
  --mount "type=bind,src=$WORKSPACE_DIR,dst=/workspace" \
  --mount "type=bind,src=$ASSETS_DIR,dst=/data/assets" \
  --mount "type=bind,src=$OUTPUT_DIR,dst=/data/output" \
  -w /workspace \
  "$IMAGE" \
  bash
