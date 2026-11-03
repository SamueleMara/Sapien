"""Build real masked COLMAP endpoints from every valid foreground RGB-D pixel."""
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np
from PIL import Image as PILImage
from point_sampling import point_fraction, sample_pixels


def build_endpoint(scene, cameras, images, shared, fraction=1.):
    fraction = point_fraction(fraction)
    from read_write_model import qvec2rotmat

    result, points, counts, hashes, available = {}, {}, {}, {}, {}
    for iid, image in images.items():
        camera = cameras[image.camera_id]
        if camera.model != 'PINHOLE':
            raise ValueError('Dense real RGB-D export requires rectified PINHOLE cameras')
        name = Path(image.name)
        if name.is_absolute() or '..' in name.parts:
            raise ValueError(f'Unsafe image name: {name}')
        mask_path = scene / 'mask' / name
        with PILImage.open(mask_path) as source:
            mask = np.asarray(source)
        with PILImage.open(scene / 'rgb' / name) as source:
            rgb = np.asarray(source.convert('RGB'))
        depth = np.load(scene / 'depth_npy' / name.with_suffix('.npy'), allow_pickle=False)
        shape = (camera.height, camera.width)
        if mask.shape != shape or mask.dtype != np.uint8 or not np.isin(mask, [0, 255]).all():
            raise ValueError(f'Expected camera-sized binary uint8 mask: {mask_path}')
        if depth.shape != shape or not np.issubdtype(depth.dtype, np.floating) or rgb.shape != (*shape, 3):
            raise ValueError(f'Expected registered metric depth and RGB at calibrated resolution: {name}')
        fx, fy, cx, cy = camera.params
        if not np.isfinite(camera.params).all() or fx <= 0 or fy <= 0:
            raise ValueError(f'Invalid intrinsics: {name}')
        ys, xs = np.nonzero((mask > 0) & np.isfinite(depth) & (depth > 0))
        available[image.name] = len(xs)
        ys, xs = sample_pixels(ys, xs, fraction)
        z = depth[ys, xs].astype(float)
        local = np.column_stack([(xs-cx)/fx*z, (ys-cy)/fy*z, z])
        # COLMAP poses are world-to-camera: X_cam = R X_world + t.
        world = (local - image.tvec) @ qvec2rotmat(image.qvec)
        ids = np.arange(len(points)+1, len(points)+len(z)+1, dtype=np.int64)
        for index, (pid, xyz, color) in enumerate(zip(ids, world, rgb[ys, xs])):
            points[int(pid)] = shared.Point3D(int(pid), xyz, color, 0.,
                np.array([iid], dtype=np.int32), np.array([index], dtype=np.int32))
        result[iid] = image._replace(xys=np.column_stack([xs, ys]).astype(float), point3D_ids=ids)
        counts[image.name] = len(z)
        hashes[image.name] = hashlib.sha256(mask_path.read_bytes()).hexdigest()
    if not points:
        raise ValueError('No valid metric depth inside endpoint masks; existing outputs were not replaced')
    return result, points, dict(num_points3D_masked=len(points), retained_tracks=len(points),
        foreground_depth_pixels=available, sampled_depth_pixels=counts, mask_sha256=hashes,
        point_fraction=fraction, point_stride=1, voxel_size_m=None,
        rule='deterministically sample valid foreground depth pixels at point_fraction per view; no voxel averaging',
        geometry_source='saved registered metric depth_npy, RGB, and COLMAP endpoint poses')


def write_ply(path, points):
    """Write XYZ/RGB only; no fabricated surface normals."""
    ordered = list(points.values())
    array = np.empty(len(ordered), dtype=[('x','<f4'),('y','<f4'),('z','<f4'),
                                        ('red','u1'),('green','u1'),('blue','u1')])
    if ordered:
        xyz = np.array([p.xyz for p in ordered])
        rgb = np.array([p.rgb for p in ordered])
        for axis, name in enumerate(('x', 'y', 'z')): array[name] = xyz[:, axis]
        for axis, name in enumerate(('red', 'green', 'blue')): array[name] = rgb[:, axis]
    with path.open('wb') as stream:
        stream.write(('ply\nformat binary_little_endian 1.0\n'
            f'element vertex {len(array)}\nproperty float x\nproperty float y\nproperty float z\n'
            'property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n').encode())
        array.tofile(stream)


def apply_scene(scene, shared, fraction=None):
    scene = Path(scene).expanduser().resolve()
    conversion_path = scene / 'metadata/real_conversion.json'
    conversion = json.loads(conversion_path.read_text())
    fraction = point_fraction(fraction if fraction is not None else conversion.get('point_fraction') or 1.)
    if conversion.get('depth_output_unit') != 'float32 meters, rectified color-camera Z; zero means missing':
        raise ValueError('Dense real export requires confirmed registered metric color-camera Z depth')
    pending = []
    # Build and validate both endpoints before replacing any existing model.
    with tempfile.TemporaryDirectory(prefix='.dense-real-colmap-', dir=scene) as temp:
        temp = Path(temp)
        for model, basename in [('colmap', 'points3D'), ('colmap_final', 'points3D_final')]:
            root = scene / model
            for target in [root, root / 'sparse/0', root / 'sparse_unmasked/0', scene / 'pointcloud']:
                if not target.resolve().is_relative_to(scene):
                    raise ValueError(f'Output escapes scene: {target}')
            cameras, images, points = shared.read_model(root / 'sparse_unmasked/0')
            original_count = len(images)
            cameras, images, points, frame = shared.select_endpoint(scene, model, cameras, images, points)
            if len(images) != original_count:
                raise ValueError('Repair legacy all-time unmasked models before dense real export')
            masked_images, masked_points, stats = build_endpoint(scene, cameras, images, shared, fraction)
            stats.update(endpoint_frame=frame, num_images=len(images), num_views=len(images), num_cameras=len(cameras),
                num_points3D=len(masked_points), num_points3D_unmasked=len(points), masking_pending=False,
                sparse_dir=str(root / 'sparse/0'), sparse_unmasked_dir=str(root / 'sparse_unmasked/0'),
                pointcloud_ply=str(scene / 'pointcloud' / (basename+'.ply')),
                pointcloud_ply_unmasked=str(scene / 'pointcloud' / (basename+'_unmasked.ply')),
                pointcloud_ply_filtering='PLY matches dense masked sparse/0; separate unmasked PLY matches sparse_unmasked/0')
            stage = temp / model
            stage.mkdir()
            for ext in ['.txt', '.bin']:
                shared.write_model(cameras, masked_images, masked_points, str(stage), ext=ext)
            write_ply(stage / 'pointcloud.ply', masked_points)
            unmasked_ply = temp / (basename+'_unmasked.ply')
            write_ply(unmasked_ply, points)
            for path in [root / 'export_summary.json', scene / 'metadata' / ('colmap_export_final.json' if model == 'colmap_final' else 'colmap_export.json')]:
                data = json.loads(path.read_text()) if path.exists() else {}
                data.update(stats)
                data['mask_filtering'] = stats
                if 'fusion' in data:
                    data['unmasked_seed_fusion'] = data.pop('fusion')
                staged_json = temp / (model+'_'+path.name)
                staged_json.write_text(json.dumps(data, indent=2)+'\n')
                pending.append((staged_json, path))
            canonical = temp / (basename+'.ply')
            write_ply(canonical, masked_points)
            pending.extend([(unmasked_ply, scene/'pointcloud'/(basename+'_unmasked.ply')),
                            (canonical, scene/'pointcloud'/(basename+'.ply'))])
            conversion.setdefault('colmap_masking', {})[model] = stats
            if model == 'colmap_final': conversion.setdefault('final_colmap', {}).update(stats)
            print(f'{scene.name}/{model}: frame {frame}, {len(images)} views; {len(masked_points)} masked depth points (fraction={fraction:g})', flush=True)
        for model in ['colmap', 'colmap_final']:
            shared.replace_directory(temp/model, scene/model/'sparse/0')
        for source, target in pending:
            target.parent.mkdir(parents=True, exist_ok=True)
            source.replace(target)
        # The preserved sparse_unmasked model is untouched; refresh its PLY from its actual points.
        for model in ['colmap', 'colmap_final']:
            basename = 'points3D' if model == 'colmap' else 'points3D_final'
            shutil.copyfile(scene/'pointcloud'/(basename+'_unmasked.ply'), scene/model/'sparse_unmasked/0/pointcloud.ply')
        staged_json = temp/'real_conversion.json'
        conversion['point_fraction'] = fraction
        staged_json.write_text(json.dumps(conversion, indent=2)+'\n')
        staged_json.replace(conversion_path)
    return True
