# Correct masks in existing real datasets

These tools are copied from FreeArtGS into this Sapien checkout. The FreeArtGS
copies remain available. Execution uses the Sapien copies and its existing
`scripts/mask_colmap_models.py` and `scripts/read_write_model.py`; no FreeArtGS
checkout is needed for this correction workflow.

Files:

- `samuele/correct_real_masks.py`: batch correction and resume handling.
- `samuele/create_standard_masks.py`: text or spatially guided mask generation.
- `samuele/sam2_point_guidance.py`: per-camera boxes, points, crops, refinement.
- `samuele/robot_mask_guidance.py`: optional projected robot background points.
- `samuele/colmap_masking.py`, `dense_real_colmap.py`, `point_sampling.py`: rebuild both masked real RGB-D endpoint models and matching PLYs.
- `samuele/leica_mask_guidance.py`: optional Leica-guided object selection.
- `utils/local_storage.py`: runtime files and caches under the selected scene.
- `configs/masks/*.json`: recording-specific spatial guidance profiles.

## Correct only black boxes

```bash
/home/samuelemara/miniconda3/envs/grounded-sam2/bin/python -B \
  /home/samuelemara/Joint3DGS/Sapien/samuele/correct_real_masks.py \
  --root "/media/samuelemara/Extreme SSD/Samuele/Backup_09_10_26/Sapien/output/real" \
  --groups black_box \
  --overwrite
```

This selects 12 `final_black_box_episode_*` scenes and the older
`black_box_20261021_161154_episode_000003` scene. They use different spatial
profiles because the newer captures place the case on cardboard supports.
The profiles aim to exclude the supports while retaining the case and opening
lid. Validation used 117 sampled frames across all 13 scenes; it does not prove
perfect masks or cover every intermediate frame. Tool-contact boundaries may
still need adjustment.

## Correct only orange suitcases

The moving part in this recording is the telescopic handle. Use the profile
that includes both silver rails and the black grip throughout retraction:

```bash
/home/samuelemara/miniconda3/envs/grounded-sam2/bin/python -B \
  /home/samuelemara/Joint3DGS/Sapien/samuele/correct_real_masks.py \
  --root "/media/samuelemara/Extreme SSD/Samuele/Backup_09_10_26/Sapien/output/real" \
  --groups orange_suitcase \
  --overwrite
```

This selects all twelve `final_orange_suitecase_episode_*` scenes. The source
folder intentionally retains `suitecase`; the CLI group uses `orange_suitcase`.
`configs/masks/final_orange_suitecase.json` widens the bounds to include the
extended handle and adds foreground anchors at the rail attachments. It
excludes the cardboard stand and uses per-camera robot background points in
image coordinates. Robot-pose projection is disabled for this recording
because the projections were inconsistent across episodes.

Validation checked five times per camera across twelve episodes (180 sampled
frames), including extended, partly retracted, and retracted handle states.
All explicit point constraints passed; this is not a GT IoU measurement or an
exhaustive check of every frame. Small robot-tool/contact boundary leaks can
remain. The correction command writes masks and rebuilds both masked COLMAP
endpoints and matching PLYs. It does not reconvert RGB-D.

## Correct all supported groups

```bash
/home/samuelemara/miniconda3/envs/grounded-sam2/bin/python -B \
  /home/samuelemara/Joint3DGS/Sapien/samuele/correct_real_masks.py \
  --root "/media/samuelemara/Extreme SSD/Samuele/Backup_09_10_26/Sapien/output/real" \
  --groups monitor matchbox laptop robot_dog wooden_box black_box orange_suitcase \
  --parquet-python /home/samuelemara/miniconda3/envs/yolo26/bin/python \
  --overwrite
```

Currently selects 87 scenes. `--parquet-python` only decodes raw Parquet on CPU
when optional robot guidance needs refreshing. SAM2 runs on CUDA in the
`grounded-sam2` process. Its weights must already exist under
`/home/samuelemara/Grounded-SAM-2-autodistill`, or pass `--model-root`.
No models or dependencies are downloaded by these commands.

Add `--dry-run` to check scene/profile layout without writes or inference.
Replace `--overwrite` with `--resume` to continue an interrupted correction;
only masks recorded with matching settings are reused. Missing or unrecorded
masks are generated again. Limit `--groups` to the objects you want to correct.
For a checkout at a different location, set `SAPIEN_REPO` to its absolute path
so the shared COLMAP exporter can be found.

Outputs stay in the selected dataset tree:

- Each scene's `mask/cam_NNN/frame_NNNNNN.png` and `metadata/mask_generation.json`.
- Initial `colmap/sparse/0` and final `colmap_final/sparse/0` masked models.
- Matching `pointcloud/points3D.ply` and `pointcloud/points3D_final.ply`.
- Scene-local `.runtime/` caches and optional `metadata/robot_mask_guidance.json`.
- `mask_correction_report.json` under the selected real output root.

RGB, depth, source Parquet, and unmasked COLMAP models are preserved. A failed
scene is reported, and the other selected scenes continue. Existing RGB-D
conversion must be complete before correction. Point sampling uses each scene's
saved fraction (or every valid masked depth pixel when absent).

## Generate text-prompt masks for one scene

```bash
/home/samuelemara/miniconda3/envs/grounded-sam2/bin/python -B \
  /home/samuelemara/Joint3DGS/Sapien/samuele/create_standard_masks.py \
  --input "/absolute/path/to/converted/scene" \
  --prompt "black box" --device cuda --overwrite
```

For a reviewed spatial profile, add
`--prompt-points /home/samuelemara/Joint3DGS/Sapien/configs/masks/final_black_box.json`.
Only use profiles with their matching recording and calibrated image size.
Text labels do not replace spatial prompts in this mode. For profiles with
robot-state guidance enabled, use the batch correction command to prepare or
validate the required projected-pose cache first.

## CPU regression checks

```bash
cd /home/samuelemara/Joint3DGS/Sapien
/home/samuelemara/miniconda3/envs/grounded-sam2/bin/python -B -m unittest discover -s tests -p 'test_real_mask_corrections.py'
/home/samuelemara/miniconda3/envs/grounded-sam2/bin/python -B -m unittest discover -s tests -p 'test_sam2_point_guidance.py'
/home/samuelemara/miniconda3/envs/grounded-sam2/bin/python -B -m unittest discover -s tests -p 'test_standard_masks.py'
```


## Final direction check

After both masked endpoint models and PLYs are rebuilt successfully, correction
runs the CPU RGB-D direction check. Its recommendation and coverage diagnostics
are saved under each scene's `metadata/direction/`. Matching reports are reused;
changed mask/depth inputs trigger recounting. Incomplete camera/frame sets fail
rather than generating a recommendation from a subset. See the
[direction-selection guide](rgbd_direction.md) to run this step alone across all
synthetic scenes and real `final_*` episodes.
