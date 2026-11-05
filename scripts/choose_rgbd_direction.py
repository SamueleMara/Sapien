#!/usr/bin/env python3
"""Choose a starting direction from masked RGB-D coverage; never reorder data.

Compare per-camera medians over the first/last 10% of the sequence. This measures
observed foreground pixels, not correspondence quality or unique 3D surfaces.
Use a scene folder/ZIP, or --root to process synthetic and real final_* scenes.
"""
import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import re
import sys
import tempfile
from zipfile import ZipFile

import numpy as np
from PIL import Image

sys.dont_write_bytecode = True
ALGORITHM_VERSION = 2
FRAME = re.compile(r'frame_(\d+)\.(png|npy)')
SKIP = {'rgb', 'mask', 'depth_npy', 'depth_png', 'metadata', 'colmap',
        'colmap_final', 'pointcloud', 'video_rgb', 'hoi'}


def decide(x, fraction=.1):
    """x: camera x frame counts, with equal pixel weights across views."""
    x = np.asarray(x)
    if not 0 < fraction <= .5 or x.ndim != 2 or 0 in x.shape or not np.isfinite(x).all() or np.any(x < 0):
        raise ValueError('Expected nonnegative finite camera x frame counts and fraction in (0, .5]')
    n = max(1, int(np.ceil(x.shape[1] * fraction)))
    first, last = np.median(x[:, :n], axis=1), np.median(x[:, -n:], axis=1)
    start, end = float(first.sum()), float(last.sum())
    direction = 'FORWARD' if start > end else 'BACKWARD' if end > start else 'TIE' if start > 0 else 'UNAVAILABLE'
    return dict(direction=direction, first_window_total=start, last_window_total=end,
                forward_initial_coverage_advantage_pct=round((start-end)/start*100, 2) if start else None,
                window_frames=n, first_per_camera=first.tolist(), last_per_camera=last.tolist())


def inventory(dataset, archive=None, fingerprint=True):
    """Validate every view/frame instead of silently intersecting partial masks."""
    entries = {}
    if archive:
        for item in archive.infolist():
            if item.filename in entries:raise ValueError(f'Duplicate ZIP entry: {item.filename}')
            entries[item.filename] = (item.file_size, item.CRC)
    else:
        for kind in ['mask', 'rgb', 'depth_npy', 'depth_png']:
            folder = dataset/kind
            if not folder.is_dir():continue
            for cam in sorted(folder.iterdir()):
                if not cam.is_dir() or not re.fullmatch(r'cam_\d+', cam.name):continue
                with os.scandir(cam) as scan:
                    for item in scan:
                        if FRAME.fullmatch(item.name):
                            entries[f'{kind}/{cam.name}/{item.name}'] = None
    groups = {}
    for name in entries:
        parts = name.split('/')
        if len(parts) != 3 or parts[0] not in {'mask', 'rgb', 'depth_npy', 'depth_png'} or not re.fullmatch(r'cam_\d+', parts[1]):continue
        match = FRAME.fullmatch(parts[2])
        if not match or match[2] != ('npy' if parts[0]=='depth_npy' else 'png'):continue
        frames = groups.setdefault((parts[0], parts[1]), {})
        index = int(match[1])
        if index in frames:raise ValueError(f'Duplicate frame ID: {name}')
        frames[index] = name
    cameras = sorted(cam for kind, cam in groups if kind=='mask')
    if not cameras:raise ValueError('Expected mask/cam_*/frame_*.png in the ZIP or folder')
    metadata_name = 'metadata/cameras.json'
    if archive and metadata_name in entries:
        metadata = json.loads(archive.read(metadata_name))
    elif not archive and (dataset/metadata_name).is_file():
        metadata = json.loads((dataset/metadata_name).read_text())
    else:metadata = None
    expected = sorted(c['name'] for c in metadata['cameras']) if metadata else cameras
    if cameras != expected:raise ValueError(f'Mask cameras differ from camera metadata: {cameras} vs {expected}')
    rgb_cameras = sorted(cam for kind, cam in groups if kind=='rgb')
    if rgb_cameras and rgb_cameras != cameras:raise ValueError('RGB and mask camera sets differ')
    frame_ids = sorted(groups['mask', cameras[0]])
    depths = {}
    for camera in cameras:
        expected_frames = set(frame_ids)
        for kind in ['mask'] + (['rgb'] if rgb_cameras else []):
            if set(groups.get((kind, camera), {})) != expected_frames:
                raise ValueError(f'{kind}/{camera}: frame set differs; incomplete or stale observations')
        kind = 'depth_npy' if groups.get(('depth_npy', camera)) else 'depth_png'
        if set(groups.get((kind, camera), {})) != expected_frames:
            raise ValueError(f'{kind}/{camera}: frame set differs from masks; missing or stale depth')
        depths[camera] = kind
    used = [name for camera in cameras for kind in ['mask', depths[camera]] for name in groups[kind, camera].values()]
    digest = None
    if fingerprint:
        signatures = []
        for name in sorted(used):
            if archive:signature = entries[name]
            else:
                stat = (dataset/name).stat()
                signature = (stat.st_size, stat.st_mtime_ns)
            signatures.append((name, signature))
        digest = hashlib.sha256(json.dumps(signatures,separators=(',', ':')).encode()).hexdigest()
    return cameras, frame_ids, groups, depths, digest


def analyze_scene(dataset, fraction=.1, out=None, resume=False, dry_run=False):
    dataset = Path(dataset).expanduser().resolve()
    if not 0 < fraction <= .5:raise ValueError('fraction must be in (0, .5]')
    if not dataset.exists():raise FileNotFoundError(dataset)
    if (dataset/'conversion_in_progress.json').exists():raise ValueError('RGB-D conversion is incomplete')
    output = Path(out).expanduser().resolve() if out else dataset/'metadata/direction'
    if dataset.is_file() and out is None:raise ValueError('ZIP input requires --out')
    if out is None and not output.resolve().is_relative_to(dataset):raise ValueError('Direction output escapes scene')
    if not (output/'.runtime/matplotlib').resolve().is_relative_to(output.resolve()):raise ValueError('Plot cache escapes direction output')
    archive = ZipFile(dataset) if dataset.is_file() else None
    try:
        cameras, frames, groups, sources, fingerprint = inventory(dataset, archive, fingerprint=not dry_run)
        signature = dict(algorithm_version=ALGORITHM_VERSION, endpoint_fraction=fraction, input_fingerprint=fingerprint)
        layout = dict(dataset=str(dataset), cameras=cameras, frames=len(frames), first_frame=frames[0],
                      last_frame=frames[-1], depth_source_by_camera=sources, **signature)
        if dry_run:
            print(f'{dataset.name}: {len(cameras)} cameras x {len(frames)} frames; layout OK; {output}', flush=True)
            return layout
        summary_path = output/'summary.json'
        if resume and all((output/name).is_file() for name in ['summary.json','per_frame.csv','coverage.png']):
            previous = json.loads(summary_path.read_text())
            if all(previous.get(key)==value for key,value in signature.items()) and previous.get('recommended_direction') in {'FORWARD','BACKWARD'}:
                print(f'{dataset.name}: reused {previous["recommended_direction"]}; {summary_path}', flush=True)
                return previous

        def read(name):
            if name.endswith('.npy'):
                if archive:return np.load(io.BytesIO(archive.read(name)), allow_pickle=False)
                return np.load(dataset/name, mmap_mode='r', allow_pickle=False)
            if archive:
                with Image.open(io.BytesIO(archive.read(name))) as image:return np.array(image)
            with Image.open(dataset/name) as image:return np.array(image)

        mask_count = np.zeros((len(cameras),len(frames)),dtype=np.int64)
        depth_count = np.zeros_like(mask_count)
        for ci, camera in enumerate(cameras):
            for ti, frame in enumerate(frames):
                mask = read(groups['mask', camera][frame])
                depth = read(groups[sources[camera], camera][frame])
                if mask.ndim!=2 or depth.ndim!=2 or mask.shape!=depth.shape:
                    raise ValueError(f'Shape mismatch or non-grayscale data: {camera}, {frame}')
                foreground = mask > 0
                mask_count[ci, ti] = np.count_nonzero(foreground)
                depth_count[ci, ti] = np.count_nonzero(foreground & np.isfinite(depth) & (depth>0))
            print(f'{dataset.name}: {camera} counted {len(frames)} frames',flush=True)
        result = dict(layout, mask_only=decide(mask_count,fraction), mask_and_valid_depth=decide(depth_count,fraction),
            interpretation='More initial valid foreground observations; not correspondence, unique-surface coverage, or training accuracy.',
            sequence_reversed=False)
        direction = result['mask_and_valid_depth']['direction']
        result['recommended_direction'] = 'FORWARD' if direction=='TIE' else None if direction=='UNAVAILABLE' else direction
        result['decision_note'] = ('Equal endpoint coverage; keep acquisition order.' if direction=='TIE' else
            'Neither endpoint has valid foreground depth; no RGB-D direction can be recommended.' if direction=='UNAVAILABLE' else
            'Initialize at the endpoint with more masked pixels with finite positive depth.')
        output.mkdir(parents=True,exist_ok=True)
        # Import only after selecting a scene-local cache; batch/CLI checks need no GUI.
        os.environ['MPLCONFIGDIR'] = str(output/'.runtime/matplotlib')
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        with tempfile.TemporaryDirectory(prefix='.direction-',dir=output) as temp:
            stage = Path(temp)
            (stage/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
            with (stage/'per_frame.csv').open('w',newline='') as stream:
                writer = csv.writer(stream)
                writer.writerow(['frame','mask_total','valid_depth_total']+[f'{c}_mask' for c in cameras]+[f'{c}_depth' for c in cameras])
                for i,frame in enumerate(frames):writer.writerow([frame,int(mask_count[:,i].sum()),int(depth_count[:,i].sum())]+mask_count[:,i].tolist()+depth_count[:,i].tolist())
            fig,ax = plt.subplots(figsize=(10.5,4.5))
            x = np.asarray(frames);n = result['mask_and_valid_depth']['window_frames']
            ax.plot(x,mask_count.sum(axis=0),label='Object mask pixels',alpha=.72)
            ax.plot(x,depth_count.sum(axis=0),label='Object pixels with valid depth',linewidth=2)
            ax.axvspan(x[0],x[n-1],color='gray',alpha=.12);ax.axvspan(x[-n],x[-1],color='gray',alpha=.12)
            ax.set(title=f'Observation-based starting direction: {direction}',xlabel='Frame',ylabel='Visible foreground pixels (sum of cameras)')
            ax.grid(alpha=.2);ax.legend();fig.tight_layout();fig.savefig(stage/'coverage.png',dpi=160);plt.close(fig)
            for name in ['per_frame.csv','coverage.png','summary.json']:(stage/name).replace(output/name)
        print(f'{dataset.name}: recommended={result["recommended_direction"]}, decision={direction}; {summary_path}',flush=True)
        return result
    finally:
        if archive:archive.close()


def discover_scenes(root, subsets):
    root = Path(root).expanduser().resolve()
    if not root.is_dir():raise FileNotFoundError(root)
    scenes = []
    for directory,children,_ in os.walk(root):
        scene = Path(directory)
        synthetic = (scene/'metadata/job_config_resolved.yaml').is_file()
        real = (scene/'metadata/real_conversion.json').is_file()
        if synthetic or real:
            if ('synthetic' in subsets and synthetic) or ('real-final' in subsets and real and re.fullmatch(r'final_.+_episode_\d+',scene.name)):
                if scene.is_symlink() or not scene.resolve().is_relative_to(root):raise ValueError(f'Scene escapes root: {scene}')
                scenes.append(scene)
            children[:] = []
        else:children[:] = [c for c in children if c not in SKIP and not c.startswith('.')]
    return sorted(scenes)


def finalize_scene(scene):
    result = analyze_scene(scene, resume=True)
    if result['recommended_direction'] is None:
        raise ValueError(f'No valid foreground depth at either endpoint: {scene}')
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset',type=Path,nargs='?',help='One standard scene folder or ZIP')
    parser.add_argument('--root',type=Path,help='Recursively discover prepared synthetic and/or real final_* episode scenes')
    parser.add_argument('--subset',choices=['synthetic','real-final'],nargs='+',default=['synthetic','real-final'])
    parser.add_argument('--out',type=Path,help='Single-scene report folder; default <scene>/metadata/direction')
    parser.add_argument('--fraction',type=float,default=.1)
    parser.add_argument('--resume',action='store_true',help='Reuse matching reports; changed mask/depth size/mtime or fraction triggers recounting')
    parser.add_argument('--dry-run',action='store_true',help='Check camera/frame layout only; no reads of pixel values and no writes')
    args = parser.parse_args(argv)
    if bool(args.dataset)==bool(args.root):parser.error('Choose a dataset or --root')
    if args.root and args.out:parser.error('--out is only valid for one dataset')
    if not 0 < args.fraction <= .5:parser.error('--fraction must be in (0, .5]')
    scenes = discover_scenes(args.root,args.subset) if args.root else [args.dataset]
    if not scenes:parser.error('No matching prepared scenes found')
    failures = 0
    for scene in scenes:
        try:
            result = analyze_scene(scene,args.fraction,args.out,args.resume,args.dry_run)
            if not args.dry_run and result.get('recommended_direction') is None:failures+=1
        except Exception as error:
            failures+=1;print(f'FAILED {scene}: {error}',file=sys.stderr,flush=True)
    print(f'{len(scenes)-failures}/{len(scenes)} scenes {"layout-validated" if args.dry_run else "processed"}; {failures} failures.',flush=True)
    return int(bool(failures))


if __name__=='__main__':sys.exit(main())
