"""Use the shared SAPIEN mask exporter for host-side real-data preparation."""
import os
import sys
from pathlib import Path
from point_sampling import point_fraction


def apply_scene_masks(scene, fraction=None):
    repo = Path(os.environ.get('SAPIEN_REPO', Path.home() / 'Joint3DGS/Sapien'))
    script_dir = repo / 'scripts'
    if not (script_dir / 'mask_colmap_models.py').is_file():
        raise FileNotFoundError(f'Missing shared mask exporter under {script_dir}; set SAPIEN_REPO to the Sapien checkout')
    sys.path.insert(0, str(script_dir))
    import mask_colmap_models as shared
    if (Path(scene) / 'metadata/real_conversion.json').is_file():
        from dense_real_colmap import apply_scene
        result = apply_scene(scene, shared, fraction)
    else:
        if fraction is not None:
            raise ValueError('--point-fraction currently applies to metric real recordings only')
        result = shared.apply_scene(scene)
    if result:
        from choose_rgbd_direction import finalize_scene
        finalize_scene(scene)
    return result


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Rebuild masked endpoint models using existing masks; no segmentation')
    parser.add_argument('--scene', required=True, type=Path)
    parser.add_argument('--point-fraction', type=point_fraction, help='Retain fraction (0, 1] of valid masked depth pixels; default: saved setting, or 1')
    args = parser.parse_args()
    apply_scene_masks(args.scene, args.point_fraction)
