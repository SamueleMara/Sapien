#!/usr/bin/env bash
# Run on the host: segmentation uses the existing grounded-sam2 environment.
set -uo pipefail
HOI_OUTPUT_ROOT="${HOI_OUTPUT_ROOT:-/media/samuelemara/Extreme SSD/Samuele/Backup_09_10_26/Sapien/output/hoi}"
MASK_PYTHON="${MASK_PYTHON:-/home/samuelemara/miniconda3/envs/grounded-sam2/bin/python}"
MASK_SCRIPT="${MASK_SCRIPT:-/home/samuelemara/FreeArtGS/samuele/create_standard_masks.py}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HOI_SOURCE_ROOT="${HOI_SOURCE_ROOT:-/media/samuelemara/Extreme SSD/Samuele/hoi!/hoi_dataset}"
status=0
found=0
for scene in "$HOI_OUTPUT_ROOT"/fridge_*; do
  [[ -d "$scene/rgb" ]] || continue
  case "$(basename "$scene")" in
    fridge_1_obj001|fridge_1_obj002) prompt="grey fridge" ;;
    fridge_1_obj003|fridge_1_obj004) prompt="white fridge" ;;
    *) echo "No configured prompt for $scene" >&2; status=1; continue ;;
  esac
  found=1
  if ! "$MASK_PYTHON" -B "$ROOT_DIR/scripts/create_hoi_appliance_guidance.py" --input "$scene" --source-root "$HOI_SOURCE_ROOT"; then
    status=1; continue
  fi
  "$MASK_PYTHON" -B "$MASK_SCRIPT" --input "$scene" --prompt "$prompt" \
    --device cuda --leica-guidance "$@" || status=1
done
if [[ "$found" == 0 ]]; then
  echo "No fridge scene folders found under $HOI_OUTPUT_ROOT" >&2
  exit 1
fi
exit "$status"
