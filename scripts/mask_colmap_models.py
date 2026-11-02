#!/usr/bin/env python3
"""Apply scene masks to initial/final COLMAP models; retain sparse_unmasked/0.

Existing-data mode is CPU-only. A point needs at least one foreground track;
background tracks are detached, preserving original image/point IDs and XY indices.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import shutil
import struct
import sys
import tempfile
from pathlib import Path, PurePosixPath

import numpy as np
from PIL import Image as PILImage
from read_write_model import Camera, Image, Point3D, CAMERA_MODELS, write_model


def read_binary_model(folder):
    def read(stream, fmt):
        size = struct.calcsize('<' + fmt)
        data = stream.read(size)
        if len(data) != size:
            raise ValueError(f'Truncated COLMAP model: {stream.name}')
        return struct.unpack('<' + fmt, data)
    cameras, images, points = {}, {}, {}
    camera_models = {m.model_id: m for m in CAMERA_MODELS}
    with (folder / 'cameras.bin').open('rb') as f:
        for _ in range(read(f, 'Q')[0]):
            cid, model, width, height = read(f, 'iiQQ')
            m = camera_models[model]
            cameras[cid] = Camera(cid, m.model_name, width, height, np.array(read(f, 'd' * m.num_params)))
    with (folder / 'images.bin').open('rb') as f:
        for _ in range(read(f, 'Q')[0]):
            values = read(f, 'idddddddi')
            name = bytearray()
            while True:
                char = f.read(1)
                if not char:
                    raise ValueError('Truncated COLMAP image name')
                if char == b'\0':
                    break
                name.extend(char)
            observations = [read(f, 'ddq') for _ in range(read(f, 'Q')[0])]
            xys = np.array([(x, y) for x, y, _ in observations], dtype=float).reshape(-1, 2)
            pids = np.array([pid for _, _, pid in observations], dtype=np.int64)
            images[values[0]] = Image(values[0], np.array(values[1:5]), np.array(values[5:8]), values[8], name.decode(), xys, pids)
    with (folder / 'points3D.bin').open('rb') as f:
        for _ in range(read(f, 'Q')[0]):
            values = read(f, 'QdddBBBdQ')
            tracks = [read(f, 'ii') for _ in range(values[-1])]
            points[values[0]] = Point3D(values[0], np.array(values[1:4]), np.array(values[4:7], dtype=np.uint8), values[7],
                                       np.array([t[0] for t in tracks], dtype=np.int32), np.array([t[1] for t in tracks], dtype=np.int32))
    return cameras, images, points


def read_model(folder):
    names = ['cameras', 'images', 'points3D']
    if all((folder / (name + '.bin')).is_file() for name in names):
        return read_binary_model(folder)
    if not all((folder / (name + '.txt')).is_file() for name in names):
        raise FileNotFoundError(f'Incomplete COLMAP model: {folder}')
    cameras, images, points = {}, {}, {}
    with (folder / 'cameras.txt').open() as stream:
        for line in stream:
            fields = line.split()
            if not fields or fields[0].startswith('#'):
                continue
            cid = int(fields[0])
            cameras[cid] = Camera(cid, fields[1], int(fields[2]), int(fields[3]), np.array(fields[4:], dtype=float))
    with (folder / 'images.txt').open() as stream:
        while True:
            line = stream.readline()
            if not line:
                break
            fields = line.split()
            if not fields or fields[0].startswith('#'):
                continue
            iid = int(fields[0])
            observations = stream.readline().split()
            if len(observations) % 3:
                raise ValueError('Invalid COLMAP image observations')
            xys = np.array([(float(observations[i]), float(observations[i+1])) for i in range(0, len(observations), 3)]).reshape(-1, 2)
            pids = np.array([int(observations[i+2]) for i in range(0, len(observations), 3)], dtype=np.int64)
            images[iid] = Image(iid, np.array(fields[1:5], dtype=float), np.array(fields[5:8], dtype=float), int(fields[8]), ' '.join(fields[9:]), xys, pids)
    with (folder / 'points3D.txt').open() as stream:
        for line in stream:
            fields = line.split()
            if not fields or fields[0].startswith('#'):
                continue
            pid = int(fields[0])
            points[pid] = Point3D(pid, np.array(fields[1:4], dtype=float), np.array(fields[4:7], dtype=np.uint8), float(fields[7]),
                                  np.array(fields[8::2], dtype=np.int32), np.array(fields[9::2], dtype=np.int32))
    return cameras, images, points


def select_endpoint(scene, model, cameras, images, points):
    """Keep one synchronized time per view, retaining IDs and valid track indices."""
    metadata_file = scene / 'metadata/cameras.json'
    if metadata_file.is_file():
        views = [camera['name'] for camera in json.loads(metadata_file.read_text())['cameras']]
    else:
        views = sorted({str(PurePosixPath(image.name).parent) for image in images.values()})
    frame = 0
    if model.endswith('_final'):
        conversion = scene / 'metadata/real_conversion.json'
        if conversion.is_file():
            data = json.loads(conversion.read_text())
            frame = data.get('output_frames', data['source_frames'])[-1]
        elif (scene / 'rgb').is_dir():
            last_frames = []
            for view in views:
                names = sorted((scene / 'rgb' / view).glob('frame_*.png'))
                if not names:
                    raise ValueError(f'Missing RGB sequence for view {view}')
                last_frames.append(int(names[-1].stem.removeprefix('frame_')))
            if len(set(last_frames)) != 1:
                raise ValueError('Views have different final frame indices')
            frame = last_frames[0]
        else:
            frame = max(int(PurePosixPath(image.name).stem.removeprefix('frame_')) for image in images.values())
    expected = {f'{view}/frame_{frame:06d}.png' for view in views}
    selected = {iid: image for iid, image in images.items() if image.name in expected}
    if {image.name for image in selected.values()} != expected or len(selected) != len(views):
        raise ValueError(f'{model} must contain exactly frame {frame} from every view ({len(views)} views)')
    kept_points = {}
    for pid, point in points.items():
        keep = np.array([iid in selected for iid in point.image_ids], dtype=bool)
        if keep.any():
            kept_points[pid] = point._replace(image_ids=point.image_ids[keep], point2D_idxs=point.point2D_idxs[keep])
    selected = {iid: image._replace(point3D_ids=np.array([pid if pid in kept_points else -1 for pid in image.point3D_ids], dtype=np.int64))
                for iid, image in selected.items()}
    used_cameras = {image.camera_id for image in selected.values()}
    return {cid: camera for cid, camera in cameras.items() if cid in used_cameras}, selected, kept_points, frame


def stage_images(scene, root, images, destination):
    """Publish endpoint image trees separately from the full RGB motion sequence."""
    destination.mkdir()
    for image in images.values():
        name = PurePosixPath(image.name)
        if name.is_absolute() or '..' in name.parts:
            raise ValueError(f'Unsafe image name: {image.name}')
        source = scene / 'rgb' / image.name
        if not source.is_file():
            source = root / 'images' / image.name
        target = destination / image.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)


def filter_model(cameras, images, points, mask_root):
    accepted, masks = {}, {}
    for iid, image in images.items():
        name = PurePosixPath(image.name)
        if name.is_absolute() or '..' in name.parts:
            raise ValueError(f'Unsafe COLMAP image name: {image.name}')
        path = mask_root / str(name)
        if not path.resolve().is_relative_to(mask_root.resolve()):
            raise ValueError(f'Mask escapes scene: {path}')
        data = path.read_bytes()  # Missing masks are errors, never treated as foreground.
        with PILImage.open(path) as source:
            mask = np.asarray(source)
        camera = cameras[image.camera_id]
        if mask.shape != (camera.height, camera.width) or mask.dtype != np.uint8 or not np.isin(mask, [0, 255]).all():
            raise ValueError(f'Expected camera-sized binary uint8 0/255 mask: {path}')
        masks[image.name] = hashlib.sha256(data).hexdigest()
        valid = np.isfinite(image.xys).all(axis=1)
        xy = np.zeros_like(image.xys, dtype=np.int64)
        xy[valid] = np.floor(image.xys[valid] + .5).astype(np.int64)
        valid &= (xy[:, 0] >= 0) & (xy[:, 0] < camera.width) & (xy[:, 1] >= 0) & (xy[:, 1] < camera.height)
        inside = np.zeros(len(xy), dtype=bool)
        inside[valid] = mask[xy[valid, 1], xy[valid, 0]] > 0
        accepted[iid] = inside
    masked_points = {}
    masked_images = {iid: image._replace(point3D_ids=np.full(len(image.point3D_ids), -1, dtype=np.int64)) for iid, image in images.items()}
    for pid, point in points.items():
        tracks = []
        for iid, index in zip(point.image_ids, point.point2D_idxs):
            if iid not in images or index < 0 or index >= len(images[iid].point3D_ids) or images[iid].point3D_ids[index] != pid:
                raise ValueError(f'Inconsistent COLMAP track: point {pid}, image {iid}, index {index}')
            if accepted[iid][index]:
                tracks.append((iid, index))
        if tracks:
            masked_points[pid] = point._replace(image_ids=np.array([t[0] for t in tracks], dtype=np.int32),
                                                point2D_idxs=np.array([t[1] for t in tracks], dtype=np.int32))
            for iid, index in tracks:
                masked_images[iid].point3D_ids[index] = pid
    if not masked_points:
        raise ValueError('Masks retain no 3D points; existing sparse models were not replaced')
    return masked_images, masked_points, dict(num_points3D_unmasked=len(points), num_points3D_masked=len(masked_points),
        retained_tracks=sum(len(p.image_ids) for p in masked_points.values()), mask_sha256=masks,
        rule='keep points with at least one foreground observation; detach background tracks; XY indices and IDs preserved')


def replace_directory(staged, destination):
    """Publish a complete model directory, retaining the previous one until installed."""
    previous = destination.with_name(destination.name + '.previous-masking')
    if previous.exists():
        raise ValueError(f'Interrupted earlier publication; inspect {previous}')
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        destination.rename(previous)
    try:
        staged.rename(destination)
    except BaseException:
        if previous.exists():
            previous.rename(destination)
        raise
    if previous.exists():
        shutil.rmtree(previous)


def write_pair(colmap_dir, cameras, images, points, mask_root, model_format="both"):
    """Fresh-export API: validate masks before publishing either model."""
    cameras, images, points, frame = select_endpoint(colmap_dir.parent, colmap_dir.name, cameras, images, points)
    masked_images, masked_points, stats = filter_model(cameras, images, points, mask_root)
    stats.update(endpoint_frame=frame, num_images=len(images), num_views=len(images))
    colmap_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.mask-export-', dir=colmap_dir) as temporary:
        temporary = Path(temporary)
        for name, ims, pts in [('unmasked', images, points), ('masked', masked_images, masked_points)]:
            folder = temporary / name
            folder.mkdir()
            for ext in (['.txt', '.bin'] if model_format == 'both' else ['.' + model_format]):
                write_model(cameras, ims, pts, str(folder), ext=ext)
        if (colmap_dir.parent / 'rgb').is_dir():
            stage_images(colmap_dir.parent, colmap_dir, images, temporary / 'images')
            replace_directory(temporary / 'images', colmap_dir / 'images')
        replace_directory(temporary / 'unmasked', colmap_dir / 'sparse_unmasked/0')
        replace_directory(temporary / 'masked', colmap_dir / 'sparse/0')
    return stats


def build_endpoint_from_saved_rgbd(scene, model):
    """Rebuild a missing synthetic endpoint using saved depth and calibrated cameras."""
    # Reuse the renderer's camera conventions, without importing/initializing SAPIEN.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
    from camera_utils import load_camera_specs
    from export_colmap_dataset import _build_colmap_cameras, _colmap_world_to_camera, _sample_valid_pixels
    from read_write_model import qvec2rotmat
    specs = load_camera_specs(str(scene / 'metadata/cameras.json'))
    cameras = _build_colmap_cameras(specs)
    common = None
    for camera in specs:
        names = {p.stem for p in (scene / 'depth_npy' / camera.name).glob('frame_*.npy')}
        names &= {p.stem for p in (scene / 'rgb' / camera.name).glob('frame_*.png')}
        common = names if common is None else common & names
    if not common:
        raise ValueError(f'No synchronized saved RGB-D frames: {scene}')
    frame = min(common) if model == 'colmap' else max(common)
    # Require the true endpoints, rather than silently accepting a truncated sequence.
    import yaml
    config = yaml.safe_load((scene / 'metadata/job_config_resolved.yaml').read_text())
    count = config.get('num_frames_resolved') or config.get('num_frames')
    if count and frame != f'frame_{0 if model == "colmap" else int(count)-1:06d}':
        raise ValueError(f'Missing expected {model} RGB-D endpoint; found {frame}')
    images, points = {}, {}
    for iid, camera in enumerate(specs, 1):
        depth = np.load(scene / 'depth_npy' / camera.name / (frame + '.npy'), allow_pickle=False)
        with PILImage.open(scene / 'rgb' / camera.name / (frame + '.png')) as image:
            rgb = np.asarray(image.convert('RGB'))
        if depth.shape != (camera.height, camera.width) or rgb.shape != (camera.height, camera.width, 3):
            raise ValueError(f'Saved RGB-D size differs from calibration: {camera.name}')
        if not np.issubdtype(depth.dtype, np.floating):
            raise ValueError('Expected metric floating-point depth_npy, not a PNG preview')
        ys, xs = _sample_valid_pixels(np.isfinite(depth) & (depth > 0), 8, 1000)
        qvec, tvec = _colmap_world_to_camera(camera)
        fx, fy, cx, cy = cameras[iid].params
        z = depth[ys, xs].astype(float)
        local = np.column_stack([(xs-cx)/fx*z, (ys-cy)/fy*z, z])
        world = (local-tvec) @ qvec2rotmat(qvec)
        ids = np.arange(len(points)+1, len(points)+len(z)+1, dtype=np.int64)
        for index, (pid, xyz, color) in enumerate(zip(ids, world, rgb[ys, xs])):
            points[int(pid)] = Point3D(int(pid), xyz, color, 0., np.array([iid], dtype=np.int32), np.array([index], dtype=np.int32))
        images[iid] = Image(iid, qvec, tvec, iid, f'{camera.name}/{frame}.png', np.column_stack([xs, ys]).astype(float), ids)
    return cameras, images, points


def apply_scene(scene, dry_run=False, build_missing_endpoints=False):
    scene = Path(scene).expanduser().resolve()
    # Validate and stage BOTH endpoints before publishing either one.
    with tempfile.TemporaryDirectory(prefix='.mask-colmap-', dir=None if dry_run else scene) as temporary:
        temporary = Path(temporary)
        pending = []
        for model in ['colmap', 'colmap_final']:
            root = scene / model
            unmasked = root / 'sparse_unmasked/0'
            if not unmasked.resolve().is_relative_to(scene) or not (root / 'sparse/0').resolve().is_relative_to(scene):
                raise ValueError(f'Sparse model escapes scene: {root}')
            original = unmasked if unmasked.exists() else root / 'sparse/0'
            rebuilt = False
            try:
                cameras, images, points = read_model(original)
            except FileNotFoundError:
                if not build_missing_endpoints or unmasked.exists():
                    raise
                cameras, images, points = build_endpoint_from_saved_rgbd(scene, model)
                rebuilt = True
            original_image_count = len(images)
            cameras, images, points, frame = select_endpoint(scene, model, cameras, images, points)
            masked_images, masked_points, stats = filter_model(cameras, images, points, scene / 'mask')
            stats.update(endpoint_frame=frame, num_images=len(images), num_views=len(images), num_cameras=len(cameras))
            if rebuilt:
                stats['geometry_source'] = 'saved metric depth_npy + RGB + camera calibration; no rendering, no part labels fabricated'
            stats.update(masking_pending=False, sparse_dir=str(root / 'sparse/0'), sparse_unmasked_dir=str(unmasked), num_points3D=len(masked_points))
            print(f'{scene.name}/{model}: frame {frame}, {len(images)} views; {len(points)} unmasked -> {len(masked_points)} masked points', flush=True)
            if dry_run:
                continue
            stage = temporary / model
            stage.mkdir()
            # Rewrite the unmasked endpoint too, including legacy all-time models.
            (stage / 'unmasked').mkdir()
            for ext in ['.txt', '.bin']:
                write_model(cameras, images, points, str(stage / 'unmasked'), ext=ext)
            stage_images(scene, root, images, stage / 'images')
            (stage / 'masked').mkdir()
            for ext in ['.txt', '.bin']:
                write_model(cameras, masked_images, masked_points, str(stage / 'masked'), ext=ext)
            stats['removed_temporal_images'] = original_image_count - len(images)
            pending.append((root, stage, stats))
        for root, stage, stats in pending:
            if stats['removed_temporal_images'] > 0:
                backup = scene / 'metadata/endpoint_model_backup' / root.name
                if not backup.exists():
                    backup.mkdir(parents=True)
                    for name in ['sparse', 'sparse_unmasked']:
                        if (root / name).exists():
                            shutil.copytree(root / name, backup / name)
                    for name in ['export_summary.json']:
                        if (root / name).exists():
                            shutil.copyfile(root / name, backup / name)
            replace_directory(stage / 'images', root / 'images')
            if (stage / 'unmasked').exists():
                replace_directory(stage / 'unmasked', root / 'sparse_unmasked/0')
            replace_directory(stage / 'masked', root / 'sparse/0')
            for path in [root / 'export_summary.json', scene / 'metadata' / ('colmap_export_final.json' if root.name == 'colmap_final' else 'colmap_export.json')]:
                data = json.loads(path.read_text()) if path.exists() else {}
                if 'point_part_counts' in data:
                    data['point_part_counts_unmasked'] = data.pop('point_part_counts')
                data.update(stats)
                data['mask_filtering'] = stats
                data['pointcloud_ply_filtering'] = 'PLY files unchanged; masked selection is in sparse/0'
                path.parent.mkdir(parents=True, exist_ok=True)
                temp = path.with_suffix('.json.mask-tmp')
                temp.write_text(json.dumps(data, indent=2) + '\n')
                temp.replace(path)
        conversion = scene / 'metadata/real_conversion.json'
        if pending and conversion.is_file():
            data = json.loads(conversion.read_text())
            data['colmap_masking'] = {root.name: stats for root, _, stats in pending}
            if 'final_colmap' in data:
                data['final_colmap'].update(pending[-1][2])
            temp = conversion.with_suffix('.json.mask-tmp')
            temp.write_text(json.dumps(data, indent=2) + '\n')
            temp.replace(conversion)
    return bool(pending)


def discover_scenes(root):
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    scenes = []
    skip = {'rgb', 'depth_npy', 'depth_png', 'mask', 'colmap', 'colmap_final', 'pointcloud', 'video_rgb', 'metadata'}
    for directory, children, _ in os.walk(root):
        path = Path(directory)
        if (path / 'metadata/job_config_resolved.yaml').is_file():
            scenes.append(path)
            children[:] = []
        else:
            children[:] = [name for name in children if name not in skip and not name.startswith('.')]
    return sorted(scenes)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--scene', type=Path, help='One real or synthetic scene containing colmap and colmap_final')
    group.add_argument('--root', type=Path, help='Recursively discover existing synthetic scenes from metadata/job_config_resolved.yaml')
    parser.add_argument('--build-missing-endpoints', action='store_true', help='Build missing synthetic endpoint models from saved metric RGB-D and cameras, on CPU')
    parser.add_argument('--dry-run', action='store_true', help='Validate both endpoints and masks without writing to datasets')
    args = parser.parse_args()
    scenes = [args.scene] if args.scene else discover_scenes(args.root)
    if not scenes:
        parser.error('No synthetic scene folders found')
    failures = []
    for scene in scenes:
        try:
            apply_scene(scene, dry_run=args.dry_run, build_missing_endpoints=args.build_missing_endpoints)
        except Exception as exc:
            failures.append(str(scene))
            print(f'FAILED {scene}: {exc}', flush=True)
    if failures:
        raise SystemExit(f'{len(failures)} failed scene(s): ' + ', '.join(failures))
    print(f'Validated {len(scenes)} scene(s).' if args.dry_run else f'Updated {len(scenes)} scene(s).', flush=True)


if __name__ == '__main__':
    main()
