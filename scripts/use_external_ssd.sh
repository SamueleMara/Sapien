#!/usr/bin/env bash
# Source from the host shell (NOT sudo); automatically resolve an existing PartNet assets root.
#   source ./scripts/use_external_ssd.sh
# Or supply the precise source explicitly:
#   ASSETS_DIR="/path/to/actual/assets" source ./scripts/use_external_ssd.sh

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo "ERROR: use 'source ./scripts/use_external_ssd.sh' (without sudo)" >&2
  exit 2
fi

_sapien_sourced_setup() {
  local mount="/media/samuelemara/Extreme SSD"
  local output="$mount/Samuele/Backup_09_10_26/Sapien/output"
  local candidate parent
  local -a matches=()

  if ! mountpoint -q "$mount"; then
    echo "ERROR: External SSD is not mounted: $mount" >&2
    return 1
  fi
  if [[ ! -d "$output/preparation_8cams" ]]; then
    echo "ERROR: Confirmed output directory not found: $output/preparation_8cams" >&2
    return 1
  fi

  if [[ -n "${ASSETS_DIR:-}" ]]; then
    # Explicit paths always take precedence; never quietly switch to a different tree.
    if [[ ! -d "$ASSETS_DIR" ]]; then
      echo "ERROR: Explicit ASSETS_DIR is missing: $ASSETS_DIR" >&2
      return 1
    fi
    candidate="$ASSETS_DIR"
  else
    # Don't guess a backup date. Discover directories that really exist.
    # The source dataset used by the current PartNet pipeline has this name.
    shopt -s nullglob
    for parent in "$mount"/Samuele/Backup_*/Sapien/assets \
                  "$mount"/Samuele/Sapien/assets \
                  "$mount"/Sapien/assets; do
      if [[ -d "$parent/partnet_subset_fixed" ]]; then
        matches+=("$parent")
      fi
    done
    shopt -u nullglob

    if (( ${#matches[@]} == 0 )); then
      # Some source trees are stored under different nesting levels.
      # Search for the actual generated URDF directory (limited depth).
      while IFS= read -r -d '' parent; do
        candidate="${parent%/partnet_subset_fixed}"
        if [[ -d "$candidate" ]]; then
          local already=0 x
          for x in "${matches[@]}"; do
            [[ "$x" == "$candidate" ]] && already=1 && break
          done
          (( already )) || matches+=("$candidate")
        fi
      done < <(find "$mount/Samuele" -maxdepth 8 -type d -name partnet_subset_fixed -print0 2>/dev/null)
    fi

    if (( ${#matches[@]} != 1 )); then
      echo "ERROR: Could not uniquely determine your PartNet assets folder." >&2
      if (( ${#matches[@]} )); then
        printf '  Candidate: %s\n' "${matches[@]}" >&2
      else
        echo "  No directory named partnet_subset_fixed was found on the external SSD." >&2
      fi
      echo 'Locate the real URDF source with:' >&2
      printf '  find "%s/Samuele" -maxdepth 8 -type d -name partnet_subset_fixed -print\n' "$mount" >&2
      echo 'Then export ASSETS_DIR="<parent of partnet_subset_fixed>" and retry.' >&2
      return 1
    fi
    candidate="${matches[0]}"
  fi

  if [[ ! -d "$candidate/partnet_subset_fixed" ]]; then
    echo "ERROR: PartNet source not found at: $candidate/partnet_subset_fixed" >&2
    echo 'ASSETS_DIR must be the parent of the generated partnet_subset_fixed directory.' >&2
    return 1
  fi
  export ASSETS_DIR="$candidate"
  export OUTPUT_DIR="$output"
  export SAPIEN_SSD_MOUNT="$mount"
  printf 'ASSETS_DIR=%q\nOUTPUT_DIR=%q\n' "$ASSETS_DIR" "$OUTPUT_DIR"
  echo 'Inside Docker: /data/assets/partnet_subset_fixed and /data/output/preparation_8cams'
}
_sapien_sourced_setup
_sapien_result=$?
unset -f _sapien_sourced_setup
return "$_sapien_result"
