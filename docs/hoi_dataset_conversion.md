# HoI! to the standard object dataset format

Converter: `scripts/convert_hoi_dataset.py`, launched inside the existing SAPIEN
Docker image by `scripts/docker_convert_hoi.sh`. The image installs NumPy,
SciPy, OpenCV and Pillow. Conversion uses CPU RGB-D processing; it does not
require a GPU, a host Conda environment, ROS, the Aria SDK, or bulk ZIP extraction.

## What was downloaded

This is [HoI!](https://hoi-dataset.ethz.ch/), with the
[official loaders](https://github.com/timengelbracht/hoi-dataset-tools/tree/main/data_processing/src/hoi/data_tools).
A folder such as `fridge_1` is a **location**, containing several articulated
objects and manipulation recordings. It is not one object sequence.

```text
hoi_dataset/fridge_1/
  gripper/                          # manipulation embodiment
    interaction_splitting_info_1-4_confirmed.json
    iphone_1/fridge_1_1-4_gripper/
      metadata                      # JSON, despite having no extension
      time_sync_info.json
      camera_rgb.zip                # camera_rgb/<timestamp_ns>.jpg
      camera_depth.zip              # camera_depth/<timestamp_ns>.npy
      poses_aligned.zip             # poses_aligned/data.csv
    iphone_2/fridge_1_1-4_gripper/    # second static view
    aria_human/                     # moving camera; not a fixed-view input
    aria_gripper/                   # moving camera; not a fixed-view input
    gripper/                        # ZED/force/tactile streams
  leica/001/
    instance_annotations_3d/instances.json
    instance_annotations_3d/points/  # static scan instance clouds
    images/instance_annotations/    # panorama masks, not dynamic video masks
    pano_tiles/                     # static scan views, not time frames
```

The downloaded sample contains iPhone ZIPs, poses, Aria streams, and Leica
annotations, but its original download script omitted the plain iPhone
`metadata`, `time_sync_info.json`, and confirmed interaction JSON. Conversion
needs these five files. `--fetch-missing-metadata` obtains only missing small
metadata from the official public index and caches it under
`<output-root>/.hoi_metadata/`, preserving source-relative paths. It never
extracts into or changes the input tree. A dry run fetches into memory only.
Ego export also requires each Aria module's `time_sync_info.json`; the same flag
fetches these two small files.
For offline use, `--metadata-root DIR` accepts an overlay with those relative
paths. The large RGB/depth ZIPs must already exist locally.

The supplied gripper ZED ZIP currently does not pass ZIP validation;
it is not required by this iPhone conversion. Aria depth and Leica static scans
are also not substituted for missing iPhone measurements.

## Object splitting and camera semantics

Default: one confirmed **opening** clip per object, using `iphone_1` and
`iphone_2` from the `gripper` embodiment. The sample's confirmed object mapping:

| Output scene | Source object index | Annotation | Matched frames at 30 FPS |
| --- | --- | --- | --- |
| `fridge_1_obj001` | 1 | silver fridge door | 261 |
| `fridge_1_obj002` | 2 | silver freezer door | 261 |
| `fridge_1_obj003` | 3 | white fridge door | 246 |
| `fridge_1_obj004` | 4 | white freezer door | 230 |

These counts come from a read-only plan with the current metadata, a 20 ms
matching tolerance, and both phones. The converter reads confirmed `start_ns`,
`end_ns`, object `index`, and open/close label. The sample's `duration_s` fields
are zero; durations are computed from start/end instead. It never concatenates
open/close repetitions or different objects into one moving sequence. If an
object has several matching windows, their session/window names are appended
to keep every output separate. Use different output roots for other actions or
embodiments.

The iPhone views occupy `cam_000`/`cam_001`; both available Aria streams
(`aria_human`, `aria_gripper`) follow as `cam_002`/`cam_003` in the same RGB/depth
folders. iPhone poses are checked for motion; each ego image has its own optical
pose. Both endpoint COLMAP models include all four endpoint cameras/images. ZED is not exported: its downloaded
archive does not pass ZIP validation. No missing stream is replaced by another
sensor or a guessed pose.

## Commands

Run from the host; this builds/updates `urdf-sapien-renderer:latest` using the
repository Dockerfile, then runs conversion inside that image:

```bash
cd /home/samuelemara/Joint3DGS/Sapien
bash scripts/docker_convert_hoi.sh --fetch-missing-metadata
```

This discovers all downloaded locations and writes object folders directly to:

```text
/media/samuelemara/Extreme SSD/Samuele/Backup_09_10_26/Sapien/output/hoi/
  fridge_1_obj001/
  fridge_1_obj002/
  fridge_1_obj003/
  fridge_1_obj004/
  .hoi_metadata/
  hoi_batch_summary.json
```

For a read-only conversion plan of just `fridge_1`, use:

```bash
bash scripts/docker_convert_hoi.sh \
  --locations fridge_1 --fetch-missing-metadata --dry-run
```

The launcher uses these defaults; override them with environment variables if
needed (single quotes protect the `!` in an interactive Bash input path):

```bash
HOI_DIR='/media/samuelemara/Extreme SSD/Samuele/hoi!/hoi_dataset' \
HOI_OUTPUT_ROOT='/media/samuelemara/Extreme SSD/Samuele/Backup_09_10_26/Sapien/output/hoi' \
  bash scripts/docker_convert_hoi.sh --fetch-missing-metadata
```

The source and repository are read-only bind mounts. The existing output parent
is mounted at `/data/output`; generated datasets and fetched metadata go only
under `/data/output/hoi`. The launcher checks external drives are mounted and
requires the output parent to exist before building. It runs as your host UID/GID,
so results are owned by you. A dry run does not create the `hoi` directory or
metadata cache, although Docker may build/update its image. Docker image layers
use Docker's normal storage, independently of the dataset output destination.
No GPU runtime flag is used for this CPU conversion. Extra converter arguments
are forwarded unchanged. For offline metadata, place its overlay in the input
tree before mounting and pass `--metadata-root /data/hoi/<overlay-directory>`.

Omit `--locations` to discover all downloaded locations containing the requested
embodiment. Select individual objects with `--object-ids 1 3`. Other options:
`--action close`, `--recording-type hand|wrist|umi`, `--fps 30`, `--size 640 480`,
`--tolerance-ms 20`, `--point-stride 8`, `--voxel-size 0.005`.
`--max-frames N` is only for a test in a separate output root: it truncates the
interaction, so its final endpoint is not the final state of the full action.
This is explicitly recorded as `smoke_test_truncated`.

To add ego views, per-camera poses and the selected Leica part to **existing**
completed object folders in the same output root:

```bash
bash scripts/docker_convert_hoi.sh --fetch-missing-metadata --upgrade-existing
```

Append `--dry-run` to validate the upgrade first. This reuses the saved static
frame timeline, camera selection, intrinsics, poses and resolution; it does not
resample/rewrite the existing iPhone RGB-D. Masks, endpoint geometry/point
clouds and original conversion metadata are preserved; endpoint COLMAP gains
the moving cameras/images at the correct endpoint poses. New auxiliary
exports are staged within each output scene before merging the cameras into
the common RGB/depth layout. The auxiliary export replaces generated ego data,
`metadata/poses/`, the selected Leica cloud and their auxiliary metadata.
`metadata/hoi_auxiliary_export.json` records the upgrade. Publication failures
roll back the auxiliary paths. Existing incomplete static conversions are
rejected, and source-window/camera mismatches are explicit errors. New scenes
in the same batch are converted normally.

Without `--upgrade-existing`, existing scene folders are refused. A failed conversion keeps `conversion_in_progress.json`;
`hoi_batch_summary.json` reports per-object successes/failures. No intermediate
archives, caches or generated scene files are written into the source dataset.

## Calibration, timestamps, and depth

- File timestamps are sorted **numerically**, not lexicographically. Different
  timestamp digit counts around 100 seconds must not reorder the recording.
- HoI! streams are time-aligned, not hardware synchronized. Nearest source
  frames are selected on a target time grid; target/source errors and
  cross-camera skew must fit the requested tolerance. Reused source frames and
  gaps outside that tolerance are dropped, not duplicated or renumbered by
  ordinal position. The original nanosecond values are saved.
- Record3D's `metadata/K` is column-major. Per-frame `fx,fy,cx,cy` are looked up
  using original metadata timestamps plus `time_sync_info.delta`. This matters
  when the recording was cropped: iPhone 2's surviving sequence does not start
  at metadata index zero. Exported timestamps are not offset a second time.
- Each RGB frame and its registered, lower-resolution depth are resampled to a
  **constant calibrated pinhole K per view**, accounting for autofocus changes.
  RGB uses linear interpolation; registered depth uses nearest-neighbor ray
  sampling. The input depth already shares the RGB field of view. No monocular
  depth is generated. Invalid/zero/nonfinite depth becomes zero.
- `depth_npy`: float32 optical camera Z in meters. `depth_png`: uint16
  millimeters, zero missing, saturated above 65.535 m. Use NPY for training.
- `poses_aligned` supplies optical camera-to-Leica-world transforms; COLMAP
  stores their inverses. No ARKit axis flip or identity pose is guessed.

## Final layout and endpoint invariant

```text
<output-root>/fridge_1_obj001/
  rgb/cam_000/frame_000000.png           # consecutive through last frame
  rgb/cam_001/frame_000000.png
  depth_npy/cam_000/frame_000000.npy
  depth_png/cam_000/frame_000000.png
  mask/cam_000/                        # empty until real masks are generated
  mask/cam_001/
  colmap/
    images/cam_000/frame_000000.png     # only initial time, every view
    images/cam_001/frame_000000.png
    sparse_unmasked/0/{cameras,images,points3D}.{txt,bin}
  colmap_final/
    images/cam_000/frame_000260.png     # only final time, every view
    images/cam_001/frame_000260.png
    sparse_unmasked/0/{cameras,images,points3D}.{txt,bin}
  rgb/cam_002/frame_000000.png            # Aria human
  rgb/cam_003/frame_000000.png            # Aria gripper
  depth_npy/cam_002/frame_000000.npy      # NONMETRIC relative inverse depth
  depth_npy/cam_003/frame_000000.npy
  depth_png/cam_002/frame_000000.png      # normalized preview, not millimeters
  mask/cam_002/                          # generated using same mask runner
  mask/cam_003/
  pointcloud/
    leica_moving_part.ply                # exact selected annotated instance
    points3D.ply
    points3D_final.ply
    views/start/cam_000.ply
    views/end/cam_000.ply
  metadata/
    cameras.json                        # all views; moving cameras point to pose files
    camera_streams.json                 # all exported cameras and depth semantics
    poses/cam_000.json                   # frame poses and RGB/depth paths
    poses/cam_001.json
    poses/cam_002.json
    poses/cam_003.json
    leica_moving_part.json               # units, frame, source hash and state warning
    frame_timestamps.csv
    hoi_object.json
    hoi_conversion.json
    colmap_export.json
    colmap_export_final.json
```

Each endpoint model has **four images total**, one per camera, regardless of
sequence length. Ego poses come from the matching endpoint frame. Its point
cloud merges only metric iPhone endpoint depth (5 mm voxel
fusion by default), with consistent COLMAP observations/tracks and measured
reprojection error. Initial/final clouds are reconstructed independently.
Leica points are not used as moving endpoint geometry.

`hoi_object.json` retains source labels/description, and `hoi_conversion.json`
records exact source windows, module mapping, calibration/timestamp provenance,
and metadata hashes. The scanned class label supplies a joint-type annotation;
no joint axis or metric articulation displacement is fabricated.

## Ego RGB/depth and per-frame poses

`metadata/camera_streams.json` indexes **every exported camera**, with its
intrinsics, image dimensions, metric-depth flag and `metadata/poses/cam_NNN.json`
path. Every pose file includes per-frame RGB/depth paths, output frame index,
source/target timestamps, matching error, camera-to-world and world-to-camera
4x4 matrices. All transforms use the same `hoi_leica_world` coordinate frame.
Static poses are repeated explicitly per frame; moving poses are never frozen.

Aria RGB is already rectified and rotated by the source preparation. The
converter uses `calib/calib.json` PINHOLE K, scales it with the exported image
size, and composes the aligned device trajectory with **both** the physical RGB
camera extrinsic and the clockwise rectification transform, matching the
[official spatial registration code](https://github.com/timengelbracht/hoi-dataset-tools/blob/main/data_processing/src/hoi/data_tools/spatial_registrator.py).
Translation is interpolated linearly and rotation uses quaternion SLERP at the
actual RGB timestamp. Invalid trajectory quality and gaps outside the matching
tolerance are excluded. Source timestamps are already aligned; their time-sync
delta is not applied again.

Ego RGB/depth pairs are matched independently to the existing static sequence's
target timestamps. Files retain the corresponding global `frame_NNNNNN` index;
missing matches remain absent and are listed in `missing_frame_indices`. A
missing ego frame does not remove a valid static frame or duplicate an ego image.
Camera IDs follow the selected static views, in human/gripper order when present.

**Aria depth is not metric RGB-D.** The official HoI! loader saves
`Depth-Anything-V2-base-hf`'s `predicted_depth` output, a relative inverse-depth
prediction. The
[model card](https://huggingface.co/depth-anything/Depth-Anything-V2-Base-hf)
identifies relative depth estimation. Its values are preserved in float32,
nearest-neighbor resized to the exported RGB resolution, under the common
`depth_npy/cam_002/` and `depth_npy/cam_003/`. Metadata explicitly sets `depth_is_metric=false` and
`depth_projection_to_world_valid=false`. The common `depth_npy/` namespace has **mixed depth semantics**: inspect each
camera's `depth_is_metric` flag. Ego `depth_png/` images are normalized previews,
not millimeters; ego maps are not fused into endpoint geometry.
Estimating inverse-depth scale and shift against metric anchors is a separate
calibration step; camera poses alone do not determine those parameters.

## Selected Leica moving-part point cloud and georeferencing

Each scene copies only
`<location>/leica/001/instance_annotations_3d/points/<object-index:03d>.ply`
to `pointcloud/leica_moving_part.ply`. This is the exact annotated moving-part
instance, not the whole appliance or scene and not a bounding-box crop. The
original binary/ASCII PLY data and coordinates are preserved byte for byte.
Missing/empty clouds and annotation/PLY point-count mismatches fail explicitly.
`metadata/leica_moving_part.json` records source SHA256, selected instance,
point count, meters, and the identity transform to `hoi_leica_world`.

For **metric iPhone optical depth** at pixel `(u,v)`, project using the exported
intrinsics and matching pose:

```python
p_camera = depth_m * numpy.linalg.inv(K) @ numpy.array([u, v, 1.0])
p_world = T_world_camera @ numpy.r_[p_camera, 1.0]
```

For a Leica point, use the inverse pose to project it into a camera image:

```python
p_camera = T_camera_world @ numpy.r_[p_leica, 1.0]
uvz = K @ p_camera[:3]
uv = uvz[:2] / uvz[2]  # only positive Z and image bounds are visible candidates
```

Both metric iPhone projections and the Leica instance are in the same world
frame. The cloud records a **static scan configuration**; the converter does not
claim that it matches every moving RGB-D frame, or even the clip's initial state.
Do not force the articulated part onto the static scan during motion, treat it
as per-frame depth GT, or infer video masks without visibility/state handling.

## Object masks and masked sparse models

HoI!'s Leica panorama masks are not masks for every iPhone video frame. The
converter therefore creates empty mask directories, not white placeholder
images. Before mask generation, only `sparse_unmasked/0` exists. Generate object
masks using the existing configurable-prompt runner, for example:

```bash
source /home/samuelemara/miniconda3/etc/profile.d/conda.sh
conda activate grounded-sam2
python -B /home/samuelemara/FreeArtGS/samuele/create_standard_masks.py \
  --input "/media/samuelemara/Extreme SSD/Samuele/Backup_09_10_26/Sapien/output/hoi/fridge_1_obj001" \
  --prompt "silver refrigerator"
```

Repeat for each object with a suitable whole-object prompt. The labels describe
moving doors; choose a prompt including the static appliance body if your
articulation training needs it. Silver objects are indices 1/2; white objects
are 3/4. A complete successful mask pass automatically publishes masked
`sparse/0` inside **both** `colmap/` and `colmap_final/`, preserving their
`sparse_unmasked/0` originals. Each variant still has exactly four endpoint images, one per physical view.
Inspect segmentation before treating the dataset as training-ready.

## Verification

```bash
# Converter checks in the SAPIEN image after the launcher has built it:
docker run --rm --user "$(id -u):$(id -g)" \
  --mount "type=bind,src=/home/samuelemara/Joint3DGS/Sapien,dst=/workspace,readonly" \
  -w /workspace urdf-sapien-renderer:latest \
  python3 -B -m unittest discover -s tests -p 'test_*hoi*.py' -v

# Launcher argument checks use only host Python's standard library:
python3 -B -m unittest discover -s tests -p test_docker_convert_hoi.py -v
```

Tests exercise ego pose interpolation and extrinsic composition, relative-depth
labeling, missing ego frame bookkeeping, exact Leica instance selection, and
ZIP/extracted inputs, confirmed per-object splitting, cropped
metadata indices, per-frame rectification, nearest-time matching with gaps,
metric depth, camera poses, one-image-per-view endpoints, point tracks, later
masked publication, dry-run/source preservation, refusal to overwrite, and
rejection of moving/unaligned cameras. Downloaded metadata was inspected and
all four real object plans validated without creating production outputs.

## Migrate existing ego folders and generate masks

Existing exports with `ego/rgb/` can be migrated without re-decoding source ZIPs:

```bash
python3 -B scripts/standardize_hoi_cameras.py --input \
  "/media/samuelemara/Extreme SSD/Samuele/Backup_09_10_26/Sapien/output/hoi"/fridge_*
```

Run the command above inside the SAPIEN container with equivalent `/data/output`
paths, or use host Python with the converter dependencies. Fresh conversion and
`--upgrade-existing` apply this migration automatically. The original iPhone
RGB/depth and existing masks remain intact. Legacy ego directories are moved;
conflicting common/legacy files are refused. Identical duplicates are removed.

Try the `fridge` prompt on one image per view, on CUDA:

```bash
/home/samuelemara/miniconda3/envs/grounded-sam2/bin/python -B \
  /home/samuelemara/FreeArtGS/samuele/create_standard_masks.py \
  --input "/media/samuelemara/Extreme SSD/Samuele/Backup_09_10_26/Sapien/output/hoi/fridge_1_obj001" \
  --prompt fridge --device cuda --max-frames 1
```

Remove `--max-frames 1` for the full sequence. Repeat for the other object folders.
The runner resumes masks with matching settings and RGB hashes. A full successful
pass filters both four-view endpoint models and preserves the unmasked variants.

## Acquisition start and Leica pose refinement

For these fridge recordings, the valid acquisition starts at original frame 22.
Prepare existing scenes inside the SAPIEN container:

```bash
python3 -B scripts/refine_hoi_geometry.py --start-frame 22 --input \
  /data/output/hoi/fridge_1_obj001 /data/output/hoi/fridge_1_obj002 \
  /data/output/hoi/fridge_1_obj003 /data/output/hoi/fridge_1_obj004
```

The retained frames are renumbered from zero. Per-frame metadata and the static
CSV retain original frame indices and source timestamps. Original metadata,
discarded frames, and previous COLMAP models are retained beneath
`metadata/before_start_frame_000022/`. The operation refuses a second, different
start index; repeating the same completed preparation does not trim or align twice.

Fixed iPhone poses are refined with bounded, robust, regularized point-to-plane
ICP against the annotated moving-part Leica cloud at the new initial frame.
Correspondences are restricted to a 10cm neighborhood; corrections are limited
to 5 degrees and 10cm. A flat door does not determine every pose direction, so
the original calibration acts as a prior. Failed/no-overlap/nonimproving fits
retain their original pose. Every fixed-camera frame receives the same correction.
Reports live in `metadata/hoi_leica_alignment.json`; distance values are geometric
registration residuals, **not joint-axis or independent GT errors**.

The initial unmasked COLMAP seed combines projected metric depth from cam_000/001
with Leica points supported by a positive camera projection and depth agreement
within 5cm. Leica observations are generated projections, not measured SfM
features. The full original selected Leica scan remains separately available.
The final seed uses final RGB-D only: the static scan is not silently copied
into the articulated final state. Both endpoint models contain all four views.
Aria depth remains nonmetric and is never fused as meter-valued geometry.

For new conversion or upgrade, add `--prepare-start-frame 22` to
`scripts/docker_convert_hoi.sh`; omit it for source datasets that do not have
this acquisition-start rule.

## Final fridge mask prompts and selection

Run on the host using the existing grounded-sam2 environment:

```bash
cd /home/samuelemara/Joint3DGS/Sapien
bash scripts/mask_hoi_datasets.sh --max-frames 1  # trial, first valid frame per camera
bash scripts/mask_hoi_datasets.sh                 # all retained frames, resumes matching masks
```

Objects 001/002 use **grey fridge**; objects 003/004 use **white fridge**.
`--leica-guidance` chooses only the GroundingDINO detection supported by the
selected annotated part projected through the current frame's calibrated pose.
Neighboring appliances are not unioned into the selected mask. SAM additionally
receives positive points from both annotated parts of the selected appliance and
negative points from the neighboring appliance. The user-defined groups are
001/002 (grey) and 003/004 (white). If no detection has sufficient reference
support, SAM can use a box derived from the projected Leica appliance instead;
this is explicitly recorded as `leica_box_fallback`, with no detector confidence
claimed. Insufficient visible reference remains explicitly empty. This static reference selects the appliance;
it is not asserted to follow the moving door or provide per-frame depth GT.

Old trial masks with another prompt/selection require explicit `--overwrite`;
pre-start masks and their manifests are preserved in the frame-22 backup.
Inspect per-frame selection status and empty counts in `metadata/mask_generation.json`.

The mask batch script creates `metadata/leica_appliance_guidance.json` from the
original Leica annotations (read-only `HOI_SOURCE_ROOT`). The file stores sampled
positive and negative world points for object selection; the selected object's
full PCD remains the only exported moving-part PCD. Prompts, guidance version,
pose/profile hashes, fallback status and coverage are recorded for reproducibility.
Existing masks require `--overwrite` when these settings change.

`metadata/synchronized_frames.json` lists the intersection of available camera
frames. Object 003 has two missing Aria-gripper frames; they are recorded rather
than replaced with repeated images or guessed poses.
