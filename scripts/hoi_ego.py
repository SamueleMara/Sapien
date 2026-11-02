"""Read already-rectified HoI! Aria streams; preserve nonmetric depth semantics."""
import csv
import hashlib
import io
import json
import shutil
import zipfile

import cv2
import numpy as np
from scipy.spatial.transform import Rotation, Slerp


def packed_json(root, label, filename):
    path = root / label / filename
    if path.is_file():
        raw = path.read_bytes()
    else:
        with zipfile.ZipFile(root / (label + '.zip')) as archive:
            raw = archive.read(f'{label}/{filename}')
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def trajectory(root, start, end, stack):
    relative = 'slam/closed_loop_trajectory_aligned/data.csv'
    path = root / relative
    if path.is_file():
        stream = stack.enter_context(path.open())
    else:
        archive = stack.enter_context(zipfile.ZipFile(root / 'slam.zip'))
        stream = stack.enter_context(io.TextIOWrapper(archive.open(relative)))
    times, translations, quaternions = [], [], []
    for row in csv.DictReader(stream):
        stamp = int(row['timestamp'])
        if start <= stamp <= end:
            if 'quality_score' in row and float(row['quality_score']) <= 0:
                continue
            times.append(stamp)
            translations.append([float(row[f'{a}_world_device']) for a in ['tx', 'ty', 'tz']])
            quaternions.append([float(row[f'{a}_world_device']) for a in ['qx', 'qy', 'qz', 'qw']])
    times = np.asarray(times, dtype=np.int64)
    translations = np.asarray(translations, dtype=float)
    if len(times) < 2 or np.any(np.diff(times) <= 0) or not np.isfinite(translations).all():
        raise ValueError(f'{root}: insufficient or invalid aligned Aria trajectory')
    return times, translations, Rotation.from_quat(quaternions)


def prepare_ego(plan, args, metadata, frames, stack, stream_type, validate_pose):
    """Plan each moving view independently, retaining shared output frame indices."""
    output = []
    tolerance = round(args.tolerance_ms * 1e6)
    base = args.input / plan['location'] / args.recording_type
    for module in ['aria_human', 'aria_gripper']:
        candidates = sorted(p for p in base.iterdir() if p.is_dir() and p.name.split(' ')[0] == module)
        if not candidates:
            continue
        if len(candidates) != 1:
            raise ValueError(f'Ambiguous {module} directory: {base}')
        root = candidates[0] / (plan['session'] + '_vrs')
        if not root.is_dir():
            raise FileNotFoundError(f'Missing ego session: {root}')
        sync = metadata.read(root.relative_to(args.input) / 'time_sync_info.json')
        if sync.get('timestamps_aligned') is not True:
            raise ValueError(f'{root}: Aria timestamps are not aligned')
        calib, digest = packed_json(root, 'calib', 'calib.json')
        calib = calib['PINHOLE']
        if calib['model'] != 'PINHOLE' or np.any(np.asarray(calib['distortion']) != 0):
            raise ValueError('Aria camera_rgb must already be rectified to PINHOLE')
        k = np.asarray(calib['K'], dtype=float)
        if k.shape != (3, 3) or not np.isfinite(k).all() or min(k[0, 0], k[1, 1]) <= 0 or not np.allclose(k[2], [0, 0, 1]):
            raise ValueError('Invalid Aria intrinsics')
        # Matches the official registration: device->raw RGB->clockwise-rectified RGB.
        extrinsic = validate_pose(np.asarray(calib['T_device_camera'], dtype=float)) @ validate_pose(np.asarray(calib['pinhole_T_device_camera'], dtype=float))
        rgb = stack.enter_context(stream_type(root, 'camera_rgb', '.jpg'))
        depth = stack.enter_context(stream_type(root, 'camera_depth', '.npy'))
        stamps = np.asarray(sorted(set(rgb.members) & set(depth.members)), dtype=np.int64)
        targets = np.asarray([row['target_ns'] for row in frames], dtype=np.int64)
        stamps = stamps[(stamps >= targets[0] - tolerance) & (stamps <= targets[-1] + tolerance)]
        if not len(stamps):
            raise ValueError(f'{root}: no paired ego RGB/depth in object window')
        times, positions, rotations = trajectory(root, targets[0] - 2*tolerance, targets[-1] + 2*tolerance, stack)
        seconds = (times - times[0]) / 1e9
        slerp = Slerp(seconds, rotations)
        rows, previous = [], None
        for number, target in enumerate(targets):
            idx = np.searchsorted(stamps, target)
            idx = min([max(0, idx-1), min(idx, len(stamps)-1)], key=lambda i: abs(int(stamps[i])-int(target)))
            stamp = int(stamps[idx])
            if abs(stamp-int(target)) > tolerance or stamp == previous or not times[0] <= stamp <= times[-1]:
                continue
            right = int(np.searchsorted(times, stamp, side='right'))
            right = min(right, len(times)-1)
            left = max(0, right-1)
            if max(stamp-int(times[left]), int(times[right])-stamp) > tolerance:
                continue
            t = (stamp-int(times[0]))/1e9
            pose = np.eye(4)
            pose[:3, :3] = slerp(t).as_matrix()
            pose[:3, 3] = [np.interp(t, seconds, positions[:, axis]) for axis in range(3)]
            pose = validate_pose(pose @ extrinsic)
            rows.append(dict(frame_index=number, target_timestamp_ns=int(target), source_timestamp_ns=stamp,
                             matching_error_ns=stamp-int(target), pose_matrix_4x4=pose.tolist(),
                             world_to_camera_matrix_4x4=np.linalg.inv(pose).tolist(),
                             pose_bracket_timestamp_ns=[int(times[left]), int(times[right])]))
            previous = stamp
        if not rows:
            raise ValueError(f'{root}: no ego RGB-D frames with valid interpolated poses')
        target_k = k.copy()
        target_k[0] *= args.size[0]/int(calib['w'])
        target_k[1] *= args.size[1]/int(calib['h'])
        target_k[0,2] = (k[0,2]+.5)*args.size[0]/int(calib['w'])-.5
        target_k[1,2] = (k[1,2]+.5)*args.size[1]/int(calib['h'])-.5
        output.append(dict(module=module, root=root, rgb=rgb, depth=depth, rows=rows,
                           calib=calib, k=target_k, calibration_sha256=digest))
    return output


def export_ego(sensors, output, size, first_id, write_json, save_png):
    summaries = []
    for index, sensor in enumerate(sensors, first_id):
        camera = f'cam_{index:03d}'
        rows = []
        for entry in sensor['rows']:
            stamp = entry['source_timestamp_ns']
            rgb = cv2.imdecode(np.frombuffer(sensor['rgb'].read(stamp), np.uint8), cv2.IMREAD_COLOR)
            raw = np.load(io.BytesIO(sensor['depth'].read(stamp)), allow_pickle=False)
            calib = sensor['calib']
            if rgb is None or rgb.shape[:2] != (int(calib['h']), int(calib['w'])) or raw.ndim != 2 or raw.dtype != np.float32:
                raise ValueError(f'{sensor["root"]}: invalid rectified ego RGB/depth dimensions or dtype')
            if raw.shape[0]*rgb.shape[1] != raw.shape[1]*rgb.shape[0]:
                raise ValueError('Aria RGB/depth aspect ratios differ; registration cannot be assumed')
            if not np.isfinite(raw).all() or not np.any(raw > 0):
                raise ValueError('Invalid relative ego depth')
            name = f'frame_{entry["frame_index"]:06d}'
            rgb_path = f'ego/rgb/{camera}/{name}.png'
            depth_path = f'ego/depth_relative_npy/{camera}/{name}.npy'
            save_png(output/rgb_path, cv2.resize(rgb, tuple(size), interpolation=cv2.INTER_LINEAR))
            target = output/depth_path
            target.parent.mkdir(parents=True, exist_ok=True)
            np.save(target, cv2.resize(raw, tuple(size), interpolation=cv2.INTER_NEAREST_EXACT), allow_pickle=False)
            rows.append(dict(entry, rgb_path=rgb_path, depth_path=depth_path))
        summary = dict(camera=camera, source_camera_name=sensor['module'], camera_motion='moving',
                       width=size[0], height=size[1], intrinsics_matrix_3x3=sensor['k'].tolist(),
                       coordinate_frame='hoi_leica_world', pose_convention='opencv_optical_camera_to_world',
                       pose_source='aligned Aria MPS SLAM; translation interpolation and quaternion SLERP at RGB timestamp',
                       pose_is_estimated=True, calibration_sha256=sensor['calibration_sha256'],
                       depth_source='Depth-Anything-V2-base-hf predicted_depth', depth_units='relative inverse depth',
                       depth_is_metric=False, depth_projection_to_world_valid=False,
                       depth_warning='Must calibrate inverse-depth scale and shift before metric 3D projection; values are not meters.',
                       num_frames=len(rows), missing_frame_indices=sorted(set(range(sensor['total_frames']))-{row['frame_index'] for row in rows}),
                       poses_file=f'metadata/poses/{camera}.json')
        write_json(output/summary['poses_file'], dict(summary, frames=rows))
        summaries.append(summary)
    return summaries


def leica_part_path(plan, args):
    # Source output_ply is an absolute path on the authors' machine; resolve locally by instance index.
    path = args.input/plan['location']/'leica/001/instance_annotations_3d/points'/f'{plan["object_index"]:03d}.ply'
    if not path.is_file():
        raise FileNotFoundError(f'Missing selected Leica moving-part cloud: {path}')
    with path.open('rb') as stream:
        header = []
        for _ in range(100):
            line = stream.readline(1024).decode('ascii').strip()
            header.append(line)
            if line == 'end_header':
                break
        else:
            raise ValueError(f'Invalid PLY header: {path}')
    counts = [int(line.split()[-1]) for line in header if line.startswith('element vertex ')]
    if header[0] != 'ply' or len(counts) != 1 or counts[0] <= 0:
        raise ValueError(f'Empty/invalid Leica moving-part cloud: {path}')
    expected = plan['annotation'].get('num_points_3d')
    if expected is not None and int(expected) != counts[0]:
        raise ValueError('Leica PLY vertex count differs from selected object annotation')
    return path, counts[0]


def export_leica(plan, output, source, count, write_json):
    destination = output/'pointcloud/leica_moving_part.ply'
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    summary = dict(path='pointcloud/leica_moving_part.ply', source=str(source),
                   source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                   object_index=plan['object_index'], description=plan['description'], num_points=count,
                   coordinate_frame='hoi_leica_world', units='meters', transform_to_dataset_world=np.eye(4).tolist(),
                   selection='exact annotated instance points; no whole-scene cloud or bounding-box crop',
                   articulation_state='static Leica scan; not asserted equal to initial or final RGB-D state',
                   annotation_file='metadata/hoi_object.json',
                   projection='p_camera = inverse(T_world_camera) @ p_world; pixel = K @ p_camera / p_camera.z',
                   warning='Use pose at matching frame. The scanned moving part changes position during manipulation; it is not per-frame depth GT.')
    write_json(output/'metadata/leica_moving_part.json', summary)
    return summary
