#!/usr/bin/env bash
# Run the HoI! converter in the same image as the SAPIEN renderer.
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${IMAGE:-urdf-sapien-renderer:latest}"
HOI_DIR="${HOI_DIR:-/media/samuelemara/Extreme SSD/Samuele/hoi!/hoi_dataset}"
HOI_OUTPUT_ROOT="${HOI_OUTPUT_ROOT:-/media/samuelemara/Extreme SSD/Samuele/Backup_09_10_26/Sapien/output/hoi}"
HOI_OUTPUT_ROOT="${HOI_OUTPUT_ROOT%/}"
OUTPUT_PARENT="$(dirname "$HOI_OUTPUT_ROOT")"
OUTPUT_NAME="$(basename "$HOI_OUTPUT_ROOT")"

source "$ROOT_DIR/scripts/docker_mount_helpers.sh"
# Bind the existing parent so --dry-run does not create the output directory.
if [[ "$HOI_DIR" != /* || "$HOI_OUTPUT_ROOT" != /* || "$OUTPUT_NAME" == . || "$OUTPUT_NAME" == .. || "$OUTPUT_NAME" == / ]]; then
  echo 'ERROR: HOI_DIR and HOI_OUTPUT_ROOT must be absolute paths to named directories.' >&2
  exit 1
fi
if [[ ! -d "$OUTPUT_PARENT" ]]; then
  echo "ERROR: Output parent must already exist: $OUTPUT_PARENT" >&2
  exit 1
fi
sapien_validate_mount_paths "$HOI_DIR" "$OUTPUT_PARENT"

# Cached builds also pick up changed requirements in an already-existing image.
docker build -t "$IMAGE" "$ROOT_DIR"
exec docker run --rm \
  --user "$(id -u):$(id -g)" \
  -e HOME=/tmp \
  -e PYTHONDONTWRITEBYTECODE=1 \
  --mount "type=bind,src=$ROOT_DIR,dst=/workspace,readonly" \
  --mount "type=bind,src=$HOI_DIR,dst=/data/hoi,readonly" \
  --mount "type=bind,src=$OUTPUT_PARENT,dst=/data/output" \
  -w /workspace "$IMAGE" \
  python3 -B scripts/convert_hoi_dataset.py \
  --input /data/hoi --output-root "/data/output/$OUTPUT_NAME" "$@"
