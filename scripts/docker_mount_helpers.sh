#!/usr/bin/env bash
# Shared helpers for launching Docker with safe host bind mounts.

sapien_require_external_mount() {
  local path="$1"
  if [[ "$path" == /media/* ]]; then
    local rest="${path#/media/}"
    local user="${rest%%/*}"
    if [[ "$rest" == "$user" ]]; then
      echo "ERROR: Invalid /media path: $path" >&2
      return 1
    fi
    rest="${rest#*/}"
    local volume="${rest%%/*}"
    local volume_mount="/media/$user/$volume"
    if ! mountpoint -q "$volume_mount"; then
      echo "ERROR: Removable drive is not mounted at $volume_mount" >&2
      return 1
    fi
  fi
}

sapien_validate_mount_paths() {
  local assets="$1" outputs="$2"
  sapien_require_external_mount "$assets" || return 1
  sapien_require_external_mount "$outputs" || return 1
  if [[ ! -d "$assets" ]]; then
    echo "ERROR: ASSETS_DIR does not exist: $assets" >&2
    return 1
  fi
  if [[ ! -d "$outputs" ]]; then
    # Create only after confirming any /media volume is actually mounted.
    mkdir -p "$outputs" || return 1
  fi
  if [[ ! -d "$outputs" || ! -w "$outputs" ]]; then
    echo "ERROR: OUTPUT_DIR is not a writable directory: $outputs" >&2
    return 1
  fi
}
