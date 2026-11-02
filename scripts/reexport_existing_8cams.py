#!/usr/bin/env python3
"""Regenerate initial + final COLMAP from previously rendered 8-camera datasets.

Only direct children of --root containing per-object metadata are treated as
existing dataset directories. No new dataset names are invented. Does not
rerender all the RGB-D frames; the COLMAP exporter replays endpoint frames.

Defaults to --dry-run to prevent unintended overwrites.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
import yaml


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default='/data/output/preparation_8cams',
                        help='Existing 8-camera dataset root mounted in Docker')
    parser.add_argument('--urdf-dir', default='/data/assets',
                        help='Location to recursively look for the original URDFs')
    parser.add_argument('--execute', action='store_true',
                        help='Write COLMAP outputs (otherwise only show the plan)')
    parser.add_argument('--sample-stride', type=int, default=8)
    parser.add_argument('--max-points-per-image', type=int, default=1000)
    return parser.parse_args()


def main():
    args = parse_args()
    root = Path(args.root).expanduser().resolve()
    urdf_dir = Path(args.urdf_dir).expanduser().resolve()
    exporter = Path(__file__).resolve().with_name('export_colmap_dataset.py')
    if not root.is_dir():
        raise SystemExit(f'ERROR: existing dataset root missing: {root}')
    if not urdf_dir.is_dir():
        raise SystemExit(f'ERROR: URDF/assets directory missing: {urdf_dir}')
    if not exporter.is_file():
        raise SystemExit(f'ERROR: exporter missing: {exporter}')
    if args.sample_stride < 1 or args.max_points_per_image < 0:
        raise SystemExit('ERROR: sample-stride must be >=1 and max-points >=0')

    selected = []
    for dataset in sorted(root.iterdir()):
        if not dataset.is_dir():
            continue
        model_configs = sorted(dataset.glob('*/metadata/job_config_resolved.yaml'))
        if not model_configs:
            continue
        wrong_counts = []
        for cfg_path in model_configs:
            try:
                cfg = yaml.safe_load(cfg_path.read_text())
                count = len(cfg['cameras'])
            except (OSError, ValueError, TypeError, KeyError) as exc:
                wrong_counts.append(f'{cfg_path.parent.parent.name}: invalid configuration ({exc})')
                continue
            if count != 8:
                wrong_counts.append(f'{cfg_path.parent.parent.name}: {count} cameras')
        if wrong_counts:
            print(f'[SKIP] {dataset.name} contains non-8-camera/invalid models:', flush=True)
            for msg in wrong_counts: print('       ' + msg, flush=True)
            continue
        selected.append((dataset, len(model_configs)))

    print(f'Existing dataset root: {root}', flush=True)
    print(f'URDF search root:      {urdf_dir}', flush=True)
    if not selected:
        raise SystemExit('No eligible 8-camera datasets found (expected */metadata/job_config_resolved.yaml).')
    print(f'Mode: {"EXECUTE: overwrite existing endpoint COLMAP/PLY" if args.execute else "DRY RUN: no files changed"}', flush=True)

    failures = []
    for dataset, count in selected:
        print(f'\n[DATASET] {dataset.name}: {count} model(s)', flush=True)
        cmd = [sys.executable, str(exporter),
               '--output-root', str(dataset),
               '--urdf-dir', str(urdf_dir),
               '--max-frames', '1',
               '--image-mode', 'copy',
               '--model-format', 'both',
               '--sample-stride', str(args.sample_stride),
               '--max-points-per-image', str(args.max_points_per_image)]
        print(' '.join(__import__('shlex').quote(item) for item in cmd), flush=True)
        if args.execute:
            exit_code = subprocess.call(cmd)
            if exit_code:
                failures.append(dataset.name)
                print(f'[FAILED] {dataset.name} (exit={exit_code})', flush=True)
    if failures:
        raise SystemExit(f'Failed datasets: {", ".join(failures)}')
    print('\nDone.' if args.execute else '\nDry run complete. Add --execute to write outputs.', flush=True)


if __name__ == '__main__':
    main()
