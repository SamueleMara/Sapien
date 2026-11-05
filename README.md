# SAPIEN Articulated Rendering Pipeline

> **External SSD setup (Ubuntu):** See [EXTERNAL_SSD.md](EXTERNAL_SSD.md) for the exact `Extreme SSD` mount paths, Docker launch commands, exFAT-compatible COLMAP export, and 64/100-camera examples.

This repository renders articulated URDF assets with SAPIEN, writes multiview
RGB/depth/mask sequences, and optionally exports a COLMAP-style sparse model
plus a part-labeled point cloud.

It supports four preparation workflows:

1. Render an existing URDF collection with fixed cameras and a moving joint.
2. Build definitive 2-link URDFs from PartNet-Mobility, then render them
   consistently and export the first-frame COLMAP data used by downstream 3DGS
   pipelines.
3. Convert native real-world RGB-D recordings and generate prompted masks.
4. Split HoI! recordings by confirmed object interactions and export static
   iPhone RGB-D views in the same endpoint dataset format.

## Three inputs, one dataset interface

| Workflow | Starting format | Preparation | Final scene folder |
| --- | --- | --- | --- |
| PARIS objects rendered in SAPIEN | Extracted PartNet-Mobility object directories with `mobility.urdf`, meshes, and hierarchy metadata; PARIS category/ID directories and `textured_objs/trans.json` for discovery | Generate two-link URDFs, choose cameras, render motion, export endpoints | `<render-root>/<category>_<id>/` |
| Real-world capture | Native `videos/`, `depth/`, `meta/`, and `data/`, including camera calibration | Rectify RGB, register depth, fuse endpoint point clouds, generate text-prompted masks | `<REAL_OUTPUT>/` |
| HoI! object interactions | Location/embodiment/session tree with timestamped iPhone RGB-D ZIPs, aligned poses, and confirmed object windows | Split by object/action, time-match the static views, rectify per-frame intrinsics, fuse endpoint clouds, then generate masks | `<output-root>/<location>_objNNN/` |

All three workflows produce `rgb/`, `depth_npy/`, `depth_png/`, `mask/`,
`colmap/`, `colmap_final/`, `pointcloud/`, and `metadata/`. Camera/frame names
match across RGB, metric depth, and masks. Use `depth_npy` for metric depth in
meters: **synthetic `depth_png` is a normalized preview**, whereas the real
converter writes millimeters to its depth PNGs. PLY names and provenance
metadata differ; synthetic data additionally has part labels and joint GT.
Real captures do not acquire joint GT from conversion or segmentation.

- [PARIS/SAPIEN: complete commands](#parissapien-from-source-assets-to-the-final-dataset)
- [Real recordings: complete commands](#real-rgb-d-recordings-native-format-to-masks)
- [Shared output contract](#output-structure)

## HoI! recordings: one standard dataset per object

Use `scripts/convert_hoi_dataset.py`. It reads timestamped ZIPs or extracted
streams directly, uses confirmed object indices and action windows, and exports
consecutive RGB/depth sequences from the two static iPhones. Each initial/final
COLMAP model contains exactly one image per camera at its endpoint, never all
time frames. Aria views follow the iPhones as `cam_002`/`cam_003` in the shared
`rgb/`, `depth_npy/`, `depth_png/` and `mask/` directories, with per-frame optical
poses in `metadata/poses/`.
`metadata/camera_streams.json` indexes all cameras. Aria depth is **nonmetric**
relative inverse depth and is labeled accordingly; it must be calibrated before
metric projection. Moving views require their per-frame pose metadata. Endpoint COLMAP includes
all four views with their actual endpoint poses; only metric iPhone depth is fused.

Each object also gets `pointcloud/leica_moving_part.ply`, copied exactly from its
Leica instance annotation in the same world frame as the camera poses.
`metadata/leica_moving_part.json` records units, source hash and the static scan
state limitation. The static scan is not per-frame moving-part ground truth.

[Full HoI! format explanation, commands, object mapping, calibration details,
and mask generation](docs/hoi_dataset_conversion.md).

Run conversion inside the existing SAPIEN Docker image:

```bash
cd /home/samuelemara/Joint3DGS/Sapien
bash scripts/docker_convert_hoi.sh --fetch-missing-metadata
```

The launcher builds/updates `urdf-sapien-renderer:latest`, mounts the HoI! input
read-only, and writes one scene per object directly under
`/media/samuelemara/Extreme SSD/Samuele/Backup_09_10_26/Sapien/output/hoi`.
It discovers all downloaded locations; the current `fridge_1` download produces
`fridge_1_obj001` through `fridge_1_obj004`. For already-converted folders, add
the new ego/pose/Leica exports without recreating static data:

```bash
bash scripts/docker_convert_hoi.sh --fetch-missing-metadata --upgrade-existing
```

This preserves existing iPhone RGB-D, masks and geometry and reuses the saved
frame timeline and resolution. Endpoint COLMAP gains the ego images/cameras
without inventing 3D points from relative depth. Append `--dry-run` to preview, or
`--locations fridge_1` to restrict discovery. `HOI_DIR` and `HOI_OUTPUT_ROOT`
override the default host paths; the output parent must already exist.

The existing download omitted plain iPhone calibration/time-sync metadata and
confirmed interactions. The explicit fetch flag obtains only those missing
small files and caches them under the output root. Conversion uses CPU inside
Docker, with no Conda activation or GPU runtime requirement. Masks are generated
separately; a successful mask pass publishes masked `sparse/0` models and
preserves `sparse_unmasked/0`. No fake masks or joint-axis GT are created.

## Masked and unmasked COLMAP models (initial and final)

Each endpoint now has two sparse models:

```text
<scene>/
  colmap/
    sparse/0/             # masked, used by training
    sparse_unmasked/0/    # full pre-mask model
    images/
  colmap_final/
    sparse/0/             # masked final state
    sparse_unmasked/0/    # full pre-mask final state
    images/
```

Each endpoint must contain **one image per view at one time**. Existing-data
filtering also repairs legacy models that incorrectly listed all time frames,
updates tracks and summaries, and replaces `colmap/images` with endpoint images
only. Old model files are backed up in `metadata/endpoint_model_backup/`.

Both sparse folders contain `cameras`, `images`, and `points3D` in text and
binary formats by default. The shared `images/` directories retain the original
RGB images. A 3D point survives if at least one of its recorded observations is
inside its corresponding `mask/<camera>/<frame>.png`. Background observations
are detached; points without foreground support are removed. Camera poses,
point coordinates/colors, IDs, and observation indices are preserved. Missing,
wrong-size, or nonbinary masks are errors; an empty filtered model is rejected.
The image/point tracks remain consistent in both text and binary models.

Fresh SAPIEN exports apply this automatically using the rendered masks. Fresh
real conversion first writes **only `sparse_unmasked/0`**; complete, successful
mask generation then publishes `sparse/0`. This prevents training from silently
using unmasked geometry while masks are still missing. The host-side real-data
mask runner uses this repository's `scripts/mask_colmap_models.py` (default
checkout `/home/samuelemara/Joint3DGS/Sapien`; set `SAPIEN_REPO` if moved).

For **existing outputs**, run the following commands on the host. These perform
CPU-only filtering of stored models, reconstructing missing synthetic endpoints
from saved RGB-D when requested: no rendering, model inference, or Docker/GPU
is needed. They preserve RGB/depth/masks and existing PLY files;
the masked selection is saved in the COLMAP sparse models. Synthetic pre-mask
models already exclude actors without articulated-object part labels, so some
scenes may retain all existing points.

```bash
source /home/samuelemara/miniconda3/etc/profile.d/conda.sh
# Use the host Python for CPU geometry/export steps.

# Existing real recording: both initial and final endpoints.
python3 -B /home/samuelemara/Joint3DGS/Sapien/scripts/mask_colmap_models.py \
  --scene "/media/samuelemara/Extreme SSD/Samuele/Backup_09_10_26/Sapien/output/real/fridge_plastic_20261016_163314"

# All 60 PARIS synthetic scenes, using their existing masks and saved RGB-D.
SAPIEN_OUTPUT="/media/samuelemara/Extreme SSD/Samuele/Backup_09_10_26/Sapien/output"
batch_status=0
for dataset in "$SAPIEN_OUTPUT"/preparation_8cams/partnet_subset_fixed_render* "$SAPIEN_OUTPUT"/preparation_64cams "$SAPIEN_OUTPUT"/preparation_100cams; do
  [ -d "$dataset" ] || continue
  python3 -B /home/samuelemara/Joint3DGS/Sapien/scripts/mask_colmap_models.py \
    --root "$dataset" --build-missing-endpoints || batch_status=1
done
[ "$batch_status" -eq 0 ]
```

`--root` discovers synthetic scenes by `metadata/job_config_resolved.yaml`, so
real recordings are handled separately. Add `--dry-run` to either command for
read-only validation and point counts. Matching masks must already exist. Endpoint models must exist unless
`--build-missing-endpoints` is supplied for saved synthetic RGB-D. Failed scenes are reported, other scenes
are still processed, and the command exits nonzero if any fail.

The first existing-data run copies the original sparse models into
`sparse_unmasked/0`. Later runs always start from those saved originals, so
edited masks can restore previously excluded points. Initial and final masks
are validated before publishing either endpoint. Summaries record both point
counts, the filtering rule, and hashes of all masks used. Do not train or modify
masks while publishing new sparse models. Existing `pointcloud/*.ply` remain
pre-mask exports; use `sparse/0/points3D.{txt,bin}` for the filtered points.

### Missing synthetic endpoint models: use saved RGB-D on CPU

The synthetic datasets already contain rendered masks; no SAM or mask inference
is needed. Their saved depth and existing sparse points may already contain
only foreground, so the masked and unmasked copies can be identical.

The current SSD has 20 PARIS scenes missing final models. Pass
`--build-missing-endpoints` to reconstruct missing endpoints from saved metric
`depth_npy`, RGB, and camera calibration. This does not load SAPIEN, rerender,
change masks, or invent background or part labels. It requires the true first
and last saved frames, and publishes both masked and unmasked COLMAP models.
Existing PLY files are preserved. Endpoint reconstruction samples depth with
stride 8 and a maximum of 1000 points per camera, matching exporter defaults.

The synthetic command above covers all 60 PARIS scenes: four 8-camera profiles
and the flat 64/100-camera roots. It excludes the unrelated `franka_joint1`
sample. Both filtering and missing-endpoint reconstruction run on the host CPU.

## PARIS/SAPIEN: from source assets to the final dataset

These commands render the ten PARIS-aligned objects listed in
`paris_subset_models.txt`: blade, foldchair, fridge, laptop, oven, scissor,
stapler, storage, USB, and washer. This rerenders their articulated assets;
it does not convert previously rendered PARIS images.

### 1. Host: mount the SSD assets/output and enter Docker

The extracted inputs on this machine are:

```text
<ASSETS_DIR>/
  partnet-mobility-v0/dataset/<id>/
    mobility.urdf
    textured_objs/
    result.json (and result_after_merging.json where available)
  PARIS/dataset/data/sapien/<category>/<id>/textured_objs/trans.json
```

Install Docker with NVIDIA Container Toolkit and a working NVIDIA driver first.
Then run on the **host**:

```bash
cd /home/samuelemara/Joint3DGS/Sapien
export ASSETS_DIR="/media/samuelemara/Extreme SSD/Samuele/Backup_09_10_26/Sapien/assets"
export OUTPUT_DIR="/media/samuelemara/Extreme SSD/Samuele/Backup_09_10_26/Sapien/output"
bash scripts/docker_enter_headless.sh
```

The launcher builds the image if missing and mounts the repository at
`/workspace`, SSD assets at `/data/assets`, and SSD output at `/data/output`.
All following synthetic commands run **inside that container**. Generated URDFs
are written to the SSD assets mount; renders and exported models go to the SSD
output mount.

```bash
cd /workspace
nvidia-smi
python3 -c "import sapien; print(sapien.__version__)"
python3 -c "import sapien.core as s; e=s.Engine(); r=s.SapienRenderer(offscreen_only=True); e.set_renderer(r); print('renderer init ok')"
```

### 2. Generate the PARIS subset URDFs and canonical manifest

Choose a fresh generated-assets folder to keep existing URDFs intact:

```bash
URDF_ROOT=/data/assets/partnet_subset_fixed_documented

python3 scripts/generate_partnet_urdfs_with_static_context.py \
  --partnet-root /data/assets/partnet-mobility-v0/dataset \
  --source-root /data/assets/PARIS/dataset/data/sapien \
  --subset-file /workspace/paris_subset_models.txt \
  --output-dir "$URDF_ROOT" \
  --canonical-manifest-out "$URDF_ROOT/partnet_canonical_manifest.json"
```

The subset file specifies both object IDs and active joint IDs. The generator
merges other geometry into the fixed link, preserving static context (including
the laptop's desk). It writes `<category>_<id>.urdf`, generation summaries, and
the canonical manifest. It can rewrite generated files if the target already
exists; use another folder for a separate preparation.

To discover **all objects in the extracted PARIS source** instead of the fixed
ten-object list, run this alternative generation command and use that folder as
`URDF_ROOT` in the rendering commands:

```bash
URDF_ROOT=/data/assets/paris_all_documented

python3 scripts/generate_partnet_urdfs_with_static_context.py \
  --partnet-root /data/assets/partnet-mobility-v0/dataset \
  --source-root /data/assets/PARIS/dataset/data/sapien \
  --paris \
  --output-dir "$URDF_ROOT" \
  --canonical-manifest-out "$URDF_ROOT/partnet_canonical_manifest.json"
```

PARIS discovery uses each `trans.json` joint selection; the matching full
PartNet-Mobility object must also be present.

### 3. Choose a camera config, or generate one

For the established eight-camera layout, no generation is necessary:

```bash
NUM_CAMS=8
CAMERA_JSON=/workspace/configs/cameras.PARIS.json
RENDER_ROOT=/data/output/preparation_8cams/partnet_subset_fixed_render_linear_documented
```

The shipped presets are eight circular views, 64 hemisphere views, and 100
hemisphere views. To select a hemisphere instead, replace the variables above:

```bash
NUM_CAMS=64
CAMERA_JSON=/workspace/configs/cameras.hemisphere_64.json
RENDER_ROOT=/data/output/preparation_64cams/partnet_subset_fixed_render_linear_documented
```

For 100 views, use `NUM_CAMS=100`,
`/workspace/configs/cameras.hemisphere_100.json`, and a distinct render root.
Camera count must match the config file. Increasing the count increases render
and export work.

To generate a custom circular config instead, run:

```bash
NUM_CAMS=8
CAMERA_JSON=/data/output/camera_configs/cameras.custom_8.json
RENDER_ROOT=/data/output/preparation_8cams/partnet_subset_fixed_render_linear_custom

python3 scripts/generate_cameras_json.py \
  --output "$CAMERA_JSON" \
  --num-cams "$NUM_CAMS" \
  --radius 4.0 --height 0.7 \
  --target 0 0 0.3 --up 0 0 1 \
  --width 640 --height-px 480 \
  --near 0.01 --far 10.0 --fov-y-deg 45
```

This example uses the shipped PARIS circle's radius/height/target. Adjust these
for your scene. Explicit intrinsics can replace FOV using
`--fx 525 --fy 525 --cx 320 --cy 240`. Camera names are `cam_000`, `cam_001`, etc.
The generated config is saved on the SSD; the shipped presets are preserved.

When a canonical manifest is supplied, the runner normally adjusts circular
camera radius/height/target per object. Shipped hemisphere configs retain their
geometry. For a custom config whose exact geometry you want preserved, add
`--no-use-canonical-camera` to the next command. Root poses and object-view
orientation overrides remain active; see `--object-view-profile` and
`--non-blade-orbit-axis` for explicit orientation control.

### 4. Render motion, masks, and both endpoint COLMAP models

Using the variables selected above:

```bash
python3 scripts/run.py \
  --urdf-dir "$URDF_ROOT" \
  --canonical-manifest-json "$URDF_ROOT/partnet_canonical_manifest.json" \
  --camera-config-json "$CAMERA_JSON" \
  --num-cams "$NUM_CAMS" \
  --output-root "$RENDER_ROOT" \
  --num-frames 60 \
  --velocity-profile linear \
  --save-video \
  --export-colmap \
  --colmap-max-frames 1 \
  --colmap-image-mode copy \
  --colmap-model-format both
```

To preview the planned render, append `--dry-run`. Without `--rewrite`, existing
object folders are skipped, including incomplete folders; inspect the batch
summary before treating a skipped object as complete. Add `--rewrite` only when
you intend to **delete and regenerate each discovered object's output**.
`copy` is required for images on this exFAT SSD because symlinks are unsupported.

Every object gets 60 RGB/depth/mask frames per camera plus optional videos.
SAPIEN provides masks directly; Grounded-SAM-2 is only needed for the real-data
workflow. `colmap/` uses frame 0, and `colmap_final/` independently exports frame
59 with the saved final joint state. `--colmap-max-frames 1` limits the initial
model, not the RGB-D motion sequence or the additional final export. Do not use
`--colmap-max-frames 0` for an endpoint reconstruction: it mixes moving states.

To change motion, replace `linear` with `ease-in`, `ease-out`, `ease-in-out`, or
`trapezoid` and use a distinct output root. For trapezoid, optionally add
`--velocity-profile-accel-fraction 0.2`. Joint limits select the default range;
`--joint-delta` explicitly overrides the displacement.

### 5. Re-export both endpoint models from existing renders

This replays the endpoint states in SAPIEN; it still needs the GPU and source
URDFs/meshes. It regenerates COLMAP/PLY exports without rerendering the full
RGB-D sequence:

```bash
python3 scripts/export_colmap_dataset.py \
  --output-root "$RENDER_ROOT" \
  --urdf-dir "$URDF_ROOT" \
  --max-frames 1 \
  --image-mode copy \
  --model-format both
```

Check `pipeline_summary.json` under the render root and
`metadata/colmap_export.json` / `metadata/colmap_export_final.json` inside each
object. Initial and final exports contain independent point sets. Synthetic
part labels and world-space joint metadata are in `pointcloud/`, as described
in the output section below.

## Real RGB-D recordings: native format to masks

This host-side workflow converts a calibrated real recording from native
`videos/`, `depth/`, `meta/`, and `data/` folders into the standard dataset
layout, then generates foreground masks with a text prompt. It runs in the
Conda environments below; the Docker instructions later in this README apply
to synthetic SAPIEN rendering.

Geometry/export commands below use host Python with NumPy, SciPy, OpenCV,
Pillow and PyYAML installed. Only mask inference uses `grounded-sam2`.

The native conversion tools currently live in
`/home/samuelemara/FreeArtGS/samuele/`. Mask generation and correction tools
are also available in this repository under `samuele/`, with the reviewed
profiles under `configs/masks/`. See [mask correction commands](docs/real_mask_corrections.md)
for correcting existing real datasets without reconversion.
The guide includes `--groups orange_suitcase` to retain the moving telescopic
handle and rails in all twelve `final_orange_suitecase_episode_*` scenes.
The required environments and local Grounded-SAM-2 weights must already be
installed. Run the following steps manually in the same terminal.

### 1. Select the recording, output folder, and prompt

```bash
source /home/samuelemara/miniconda3/etc/profile.d/conda.sh

REAL_INPUT="/media/samuelemara/Extreme SSD/Samuele/Backup_09_10_26/Sapien/assets/dataset_fridge_plastic_depth_20261016_163314"
REAL_OUTPUT="/media/samuelemara/Extreme SSD/Samuele/Backup_09_10_26/Sapien/output/real/fridge_plastic_20261016_163314_v2"
MASK_PROMPT="red fridge"
```

Full conversion requires a **new output directory**. The original
`fridge_plastic_20261016_163314` already exists; choose another unused name if
`_v2` also exists. Keep paths quoted because the SSD name contains a space.

### 2. Convert RGB, depth, camera calibration, and both COLMAP models

Activate the geometry environment explicitly:

```bash
# Use the host Python for CPU geometry/export steps.

# Optional: inspect the planned conversion without creating output files.
python3 -B /home/samuelemara/FreeArtGS/samuele/convert_real_recording.py \
  --input "$REAL_INPUT" --output "$REAL_OUTPUT" --dry-run

# Create the dataset (CPU conversion; no mask inference).
python3 -B /home/samuelemara/FreeArtGS/samuele/convert_real_recording.py \
  --input "$REAL_INPUT" --output "$REAL_OUTPUT"
```

The converter matches RGB frames to available depth frame IDs, rectifies RGB,
and registers depth to the rectified color camera. Defaults are 640 x 480,
episode 0, all synchronized frames, point sampling stride 8, and 5 mm fusion
voxels. Depth NPY files contain camera-Z in meters; depth PNGs contain
millimeters, with zero for missing depth.

`colmap/` contains exactly `frame_000000.png` from each camera and the fused
initial RGB-D point cloud. It does not list the other times in the sequence.
`colmap_final/` contains the final selected view from each camera and a
**separately fused final RGB-D point cloud**. Conversion writes their text and
binary camera, image, and point files into `sparse_unmasked/0`; the next mask
generation step creates the foreground-filtered `sparse/0` models. Fusion uses the supplied calibration;
it does not refine it. The recording's `CANDIDATE_NOT_VALIDATED` calibration
status is retained in the metadata.

### 3. Generate masks with Grounded-SAM-2 on the GPU

Switch to the segmentation environment explicitly:

```bash
conda activate grounded-sam2

python -B /home/samuelemara/Joint3DGS/Sapien/samuele/create_standard_masks.py \
  --input "$REAL_OUTPUT" \
  --prompt "$MASK_PROMPT" \
  --device cuda
```

`--prompt` is required: replace `red fridge` with the desired object description.
The default model root is `/home/samuelemara/Grounded-SAM-2-autodistill`; use
`--model-root` if the installed checkout and weights are elsewhere. CUDA is
required by the command above; CPU inference is available only when explicitly
requested with `--device cpu`.

Masks are binary PNGs (0 background, 255 foreground) with the same camera names,
frame IDs, and image dimensions as RGB. Existing masks with matching provenance
are reused. Add `--overwrite` to regenerate masks after changing the prompt or
settings. Empty predictions are recorded in `metadata/mask_generation.json`
and cause a nonzero exit status. RGB and depth are preserved. After a complete
successful mask pass, both masked COLMAP models are published to `sparse/0`;
the original models remain in `sparse_unmasked/0`.
Model runtime caches stay under `$REAL_OUTPUT/.runtime/`.

### Generated dataset structure

```text
<REAL_OUTPUT>/
  rgb/cam_000/frame_000000.png
  depth_npy/cam_000/frame_000000.npy
  depth_png/cam_000/frame_000000.png
  mask/cam_000/frame_000000.png
  colmap/
    images/cam_000/frame_000000.png
    sparse/0/{cameras,images,points3D}.{txt,bin}
    sparse_unmasked/0/{cameras,images,points3D}.{txt,bin}
    export_summary.json
  colmap_final/
    images/cam_000/frame_000327.png
    sparse/0/{cameras,images,points3D}.{txt,bin}
    sparse_unmasked/0/{cameras,images,points3D}.{txt,bin}
    export_summary.json
  pointcloud/
    points3D.ply
    points3D_final.ply
    views/start/cam_000.ply
    views/end/cam_000.ply
  metadata/
    cameras.json
    frame_timestamps.csv
    real_conversion.json
    colmap_export.json
    colmap_export_final.json
    mask_generation.json
    raw/
```

The example recording has three cameras (`cam_000` through `cam_002`), with
328 synchronized frames per camera, named consecutively from `frame_000000`
through `frame_000327`. These correspond to original capture indices
0, 2, ..., 654. The original indices remain in `source_frames` in
`metadata/real_conversion.json` and in the `source_frame_index` column of
`metadata/frame_timestamps.csv`; timestamps retain capture timing.
Other recordings can have different frame counts. This workflow creates
foreground masks, not ground-truth part labels or joint annotations.

### Add a missing final COLMAP model to an existing conversion

Only use this repair command when an already converted dataset lacks
`colmap_final/`. It leaves its existing RGB, depth, and masks in place:

```bash
source /home/samuelemara/miniconda3/etc/profile.d/conda.sh
# Use the host Python for CPU geometry/export steps.

python3 -B /home/samuelemara/FreeArtGS/samuele/convert_real_recording.py \
  --output "/media/samuelemara/Extreme SSD/Samuele/Backup_09_10_26/Sapien/output/real/fridge_plastic_20261016_163314" \
  --final-colmap-only
```

### Repair legacy nonconsecutive frame names

For real datasets generated by the older converter with names `0, 2, 4, ...`,
use the repair mode below. It changes all RGB, depth, mask, and COLMAP image
filenames to consecutive indices and updates both text/binary COLMAP image
references, mask manifest keys, endpoint metadata, and timestamp rows. It
preserves source indices separately and does not regenerate images or masks.

```bash
source /home/samuelemara/miniconda3/etc/profile.d/conda.sh
# Use the host Python for CPU geometry/export steps.
python3 -B /home/samuelemara/FreeArtGS/samuele/convert_real_recording.py \
  --output "/media/samuelemara/Extreme SSD/Samuele/Backup_09_10_26/Sapien/output/real/fridge_plastic_20261016_163314" \
  --renumber-frames
```

Add `--dry-run` to inspect first. Data files are checksum-verified after the
rename; original reference files and a resumable journal are saved in
`metadata/frame_renumbering_backup/`. If interrupted, rerun the same command.
Keep training and mask generation stopped during this repair. Fresh conversions
already use consecutive names and need no repair.

## What The Repo Contains

```text
src/
  render_urdf_multiview.py        direct SAPIEN CLI
  sapien_urdf_renderer.py         renderer, segmentation, frame sampling
  camera_utils.py                 camera loading and pose helpers
  config_types.py                 validated render-job schema
  io_utils.py                     output writers

scripts/
  docker_convert_hoi.sh            HoI! conversion in the SAPIEN Docker image
  convert_hoi_dataset.py           confirmed HoI! objects to standard RGB-D/COLMAP
  run.py                          main batch pipeline
  run_with_joint_delta.py         compatibility wrapper with same behavior as run.py
  export_colmap_dataset.py        COLMAP/PLY export from rendered outputs
  generate_partnet_urdfs.py       PartNet 2-link URDF generation
  generate_partnet_urdfs_with_static_context.py
                                  PartNet generation with extra static context meshes
  generate_cameras_json.py        camera-ring generator
  generate_sapien_cameras_json.py alternate camera generator
  list_joints.sh                  quick joint inspection helper
  docker_enter_headless.sh        start a headless GPU container
  docker_enter_gui.sh             start a GUI-enabled GPU container

configs/
  cameras.PARIS.json              camera layout used for the PARIS-style outputs
  cameras.pipeline.json           default pipeline camera config
  cameras.circle.json             simple circular example
  job.example.yaml                example render job config

assets/
  partnet-mobility-v0/            expected extracted PartNet dataset root
  *.urdf                          example robot/object URDFs

paris_subset_models.txt           subset list used for PARIS-aligned PartNet runs
output/                           rendered outputs and summaries
```

## Main Concepts

- `scripts/run.py` is the main entrypoint. It scans a URDF directory, picks one
  active 1-DoF joint per model, chooses a default `q_start/q_end` range from
  the joint limits, renders the sequence, and optionally exports COLMAP.
- `src/render_urdf_multiview.py` is the low-level single-job renderer.
- `scripts/export_colmap_dataset.py` replays rendered jobs to build
  `colmap/sparse/0` and `pointcloud/points3D_parts.ply`.
- For PartNet, the safest generator is
  `scripts/generate_partnet_urdfs_with_static_context.py`, because some objects
  contain visual meshes outside `mobility.urdf`. The classic example is
  `laptop/10211`, where the static desk/context geometry would otherwise be
  missing.

## Prerequisites

For synthetic rendering, run inside Docker. The real-recording workflow above
uses host Conda environments.

Host requirements:

- NVIDIA GPU with working host drivers
- Docker
- NVIDIA Container Toolkit so `docker run --gpus all ...` works

If `nvidia-smi` fails on the host or inside the container, SAPIEN headless
rendering will not work reliably.

## Installation

### Docker Installation

The container is the supported path. It installs:

- `sapien==2.2.2`
- `numpy`
- `imageio`
- `imageio-ffmpeg`
- `matplotlib`
- `opencv-python-headless`
- `Pillow`
- `PyYAML`
- `tqdm`

Start a headless shell:

```bash
./scripts/docker_enter_headless.sh
```

This script:

- builds `urdf-sapien-renderer:latest` if needed
- mounts the repo at `/workspace`
- mounts the configured SSD assets at `/data/assets` and outputs at `/data/output`
- enables GPU access with `--gpus all`

Alternative with Compose:

```bash
docker compose run --rm renderer bash
```

### Local Python Installation

Local execution is possible, but only if your machine already has a working
SAPIEN-compatible EGL/Vulkan/NVIDIA environment.

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip setuptools wheel
python3 -m pip install -r requirements.txt
export PYTHONPATH="$PWD/src"
```

The Docker image already sets `PYTHONPATH=/workspace/src`.

## Data Layout

For the PartNet workflow, the expected extracted dataset location is:

```text
/data/assets/partnet-mobility-v0/dataset
```

The PARIS-aligned subset file in this repo is:

```text
/workspace/paris_subset_models.txt
```

Each line has the form:

```text
<category>/<partnet_object_id> <joint_id>
```

Example:

```text
blade/103706 0
foldchair/102255 1
fridge/10905 0
laptop/10211 1
oven/101917 0
scissor/11100 0
stapler/103111 1
storage/45135 0
USB/100109 0
washer/103776 0
```

## Quick Environment Check

Run these inside the container before launching a long job:

```bash
python3 -c "import sapien; print(sapien.__version__)"
python3 -c "import sapien.core as sapien; e=sapien.Engine(); r=sapien.SapienRenderer(offscreen_only=True); e.set_renderer(r); print('renderer init ok')"
```

If the second command fails with Vulkan/EGL/NVIDIA errors, stop and fix the
runtime first.

## Workflow 1: Render An Existing URDF Directory

Render a URDF collection with the PARIS camera layout, save RGB/depth/mask/video,
and skip COLMAP:

```bash
python3 scripts/run.py \
  --urdf-dir /data/assets/definitive_urdf/definitive_urdf \
  --output-root /data/output/preparation_definitive_urdf \
  --camera-config-json /workspace/configs/cameras.PARIS.json \
  --num-frames 60 \
  --save-video \
  --no-export-colmap
```

Render the same dataset and export initial and final COLMAP and PLY outputs:

```bash
python3 scripts/run.py \
  --urdf-dir /data/assets/definitive_urdf/definitive_urdf \
  --output-root /data/output/preparation_definitive_urdf_colmap \
  --camera-config-json /workspace/configs/cameras.PARIS.json \
  --num-frames 60 \
  --save-video \
  --export-colmap \
  --colmap-max-frames 1
```

Important:

- `--colmap-max-frames 1` is the default.
- In that default mode, `colmap/` uses `frame_000000`; `colmap_final/`
  independently uses the last rendered frame.
- The initial model is independent from the dynamic velocity profile; the
  final export reuses the saved final joint state.

## Workflow 2: Generate PartNet 2-Link URDFs

Use the static-context generator when you want PartNet visuals to match the
full object, not just the meshes explicitly referenced in `mobility.urdf`.

```bash
python3 scripts/generate_partnet_urdfs_with_static_context.py \
  --partnet-root /data/assets/partnet-mobility-v0/dataset \
  --subset-file /workspace/paris_subset_models.txt \
  --output-dir /data/assets/partnet_subset_fixed \
  --canonical-manifest-out /data/assets/partnet_subset_fixed/partnet_canonical_manifest.json
```

Expected laptop check:

```text
[OK] laptop_10211: joint=joint_1 fixed_meshes=105 moving_meshes=3
```

That confirms the extra static context meshes were included.

If you intentionally want the older `mobility.urdf`-only visual behavior:

```bash
python3 scripts/generate_partnet_urdfs_with_static_context.py \
  --partnet-root /data/assets/partnet-mobility-v0/dataset \
  --subset-file /workspace/paris_subset_models.txt \
  --output-dir /data/assets/partnet_subset_no_context \
  --no-include-result-static-meshes
```

## Workflow 3: Render Generated PartNet URDFs

Recommended command for the generated subset:

```bash
python3 scripts/run.py \
  --urdf-dir /data/assets/partnet_subset_fixed \
  --canonical-manifest-json /data/assets/partnet_subset_fixed/partnet_canonical_manifest.json \
  --output-root /data/output/partnet_subset_fixed_render_trapezoid \
  --num-frames 60 \
  --velocity-profile trapezoid \
  --velocity-profile-accel-fraction 0.2 \
  --save-video \
  --export-colmap \
  --colmap-max-frames 1
```

This uses the default per-object joint range selected from each joint's limits.
No `--joint-delta` is needed.

Static sanity check with no motion:

```bash
python3 scripts/run.py \
  --urdf-dir /data/assets/partnet_subset_fixed \
  --canonical-manifest-json /data/assets/partnet_subset_fixed/partnet_canonical_manifest.json \
  --output-root /data/output/partnet_subset_fixed_static \
  --joint-delta 0 \
  --num-frames 2 \
  --no-export-colmap
```

## Velocity Profiles

The dynamic renderer supports these joint-space profiles:

```text
linear
ease-in
ease-out
ease-in-out
trapezoid
```

Examples:

```bash
python3 scripts/run.py \
  --urdf-dir /data/assets/partnet_subset_fixed \
  --canonical-manifest-json /data/assets/partnet_subset_fixed/partnet_canonical_manifest.json \
  --output-root /data/output/partnet_subset_fixed_render_linear \
  --num-frames 60 \
  --velocity-profile linear \
  --save-video \
  --no-export-colmap
```

```bash
python3 scripts/run.py \
  --urdf-dir /data/assets/partnet_subset_fixed \
  --canonical-manifest-json /data/assets/partnet_subset_fixed/partnet_canonical_manifest.json \
  --output-root /data/output/partnet_subset_fixed_render_ease_in_out \
  --num-frames 60 \
  --velocity-profile ease-in-out \
  --save-video \
  --no-export-colmap
```

```bash
python3 scripts/run.py \
  --urdf-dir /data/assets/partnet_subset_fixed \
  --canonical-manifest-json /data/assets/partnet_subset_fixed/partnet_canonical_manifest.json \
  --output-root /data/output/partnet_subset_fixed_render_trapezoid \
  --num-frames 60 \
  --velocity-profile trapezoid \
  --velocity-profile-accel-fraction 0.2 \
  --save-video \
  --no-export-colmap
```

Notes:

- The frame count is still fixed by `--num-frames` or by `--fps` plus
  `--duration-sec`.
- The profile changes only how `q` evolves across those frames.
- The first frame is always `q_start`.
- The last frame is always `q_end`.
- Each render also writes `metadata/joint_trajectory.png` with three subplots:
  position, speed, and acceleration over frame index.

## Workflow 4: Direct Single-Model Rendering

List active joints in one URDF:

```bash
python3 src/render_urdf_multiview.py list-joints \
  --urdf /data/assets/partnet_subset_fixed/laptop_10211.urdf \
  --as-json
```

Render one URDF directly:

```bash
python3 src/render_urdf_multiview.py render \
  --urdf /data/assets/partnet_subset_fixed/laptop_10211.urdf \
  --joint-name joint_1 \
  --q-start 0.0 \
  --q-end 0.35 \
  --num-frames 60 \
  --velocity-profile trapezoid \
  --velocity-profile-accel-fraction 0.2 \
  --views-json /workspace/configs/cameras.PARIS.json \
  --output-dir /data/output/laptop_direct_render \
  --save-video
```

Use this mode when you want full manual control over a single articulation.

## Workflow 5: Export COLMAP From Existing Renders

If a render output already exists and you only need COLMAP/PLY:

```bash
python3 scripts/export_colmap_dataset.py \
  --output-root /data/output/preparation_definitive_urdf_colmap \
  --urdf-dir /data/assets/definitive_urdf/definitive_urdf \
  --max-frames 1
```

Or for a single model:

```bash
python3 scripts/export_colmap_dataset.py \
  --model-output /data/output/preparation_definitive_urdf_colmap/USB_100109 \
  --urdf /data/assets/definitive_urdf/definitive_urdf/USB_100109.urdf \
  --max-frames 1
```

## Output Structure

The shared interface is RGB, metric NPY depth, masks, calibrated cameras, and
independent initial/final COLMAP models. Synthetic scenes add joint GT and
part-labeled point clouds; real-scene PLYs are named `points3D.ply` and
`points3D_final.ply`, without GT parts. Both real and synthetic models use exactly one time per view:
`colmap/` contains only frame 0 from every camera, and `colmap_final/` contains
only the final frame from every camera. For a three-camera recording, each
model has three image records and three image files, not cameras multiplied
by times. This holds for both `sparse/0` and `sparse_unmasked/0`.
The complete motion sequence stays in `rgb/`, `depth_npy/`, `depth_png/`, and `mask/`.

Each model render writes a directory like:

```text
<model_name>/
  rgb/<camera_name>/frame_000000.png
  depth_npy/<camera_name>/frame_000000.npy
  depth_png/<camera_name>/frame_000000.png
  mask/<camera_name>/frame_000000.png
  video_rgb/<camera_name>.mp4
  metadata/
    cameras.json
    frame_values.csv
    joint_trajectory.png
    job_config_resolved.yaml
    joint_info.json
    package_versions.json
    visual_materials.json
```

With COLMAP enabled, you also get:

```text
<model_name>/
  colmap/
    images/<camera_name>/frame_000000.png
    sparse/0/cameras.txt
    sparse/0/images.txt
    sparse/0/points3D.txt
    sparse/0/cameras.bin
    sparse/0/images.bin
    sparse/0/points3D.bin
    sparse_unmasked/0/{cameras,images,points3D}.{txt,bin}
    export_summary.json
  colmap_final/
    images/<camera_name>/frame_000059.png
    sparse/0/{cameras,images,points3D}.{txt,bin}
    sparse_unmasked/0/{cameras,images,points3D}.{txt,bin}
    export_summary.json
  pointcloud/
    points3D_parts.ply
    points3D_parts_final.ply
    joint_metadata.json
  metadata/
    colmap_export.json
    colmap_export_final.json
```

Frame 59 above assumes 60 rendered frames. Synthetic depth PNGs are normalized
visualizations; use NPY files for metric values.

Part labels in `points3D_parts.ply`:

```text
part=0  static/base link
part=1  moving/child link
```

The PLY and JSON metadata also store the rendered joint axis and point of
application in world coordinates.

`metadata/joint_trajectory.png` is computed in frame space:

- top subplot: joint position `q`
- middle subplot: finite-difference speed `dq / frame`
- bottom subplot: finite-difference acceleration `d²q / frame²`

## How Joint Motion Is Chosen

Batch mode in `scripts/run.py` works like this:

1. Scan the URDF and list active joints.
2. Pick the first active 1-DoF joint.
3. If joint limits exist, choose a default `q_start/q_end` from those limits.
4. Sample the chosen motion profile across `num_frames`.
5. Render each frame by setting the articulation `qpos` directly in SAPIEN.

If you pass `--joint-delta`, only the final displacement is overridden. If you
do not pass it, the default per-object range is used.

## Troubleshooting

Typical renderer failure:

```text
[svulkan2] [error] Vulkan is incompatible with your driver
```

or:

```text
nvidia-smi
No devices were found
```

Interpretation:

- the URDF is not the problem
- the container GPU runtime is the problem

Recommended checks:

```bash
./scripts/docker_enter_headless.sh
nvidia-smi
python3 -c "import sapien.core as sapien; e=sapien.Engine(); r=sapien.SapienRenderer(offscreen_only=True); e.set_renderer(r); print('renderer init ok')"
```

If that fails, fix Docker/NVIDIA first.

Other useful checks:

```bash
python3 src/render_urdf_multiview.py list-joints --urdf /data/assets/partnet_subset_fixed/laptop_10211.urdf --as-json
python3 scripts/run.py --help
python3 scripts/export_colmap_dataset.py --help
```

## Recommended Entry Points

Use these scripts in normal work:

- `scripts/docker_enter_headless.sh` to enter the container
- `scripts/generate_partnet_urdfs_with_static_context.py` to build PartNet URDFs
- `scripts/run.py` to render datasets
- `scripts/export_colmap_dataset.py` to export COLMAP/PLY from existing renders

Keep `scripts/run_with_joint_delta.py` only as a compatibility wrapper. New
commands should use `scripts/run.py`.

### HoI fridge acquisition and masks

For this acquisition, add `--prepare-start-frame 22` to the Docker conversion/upgrade
command. Original frame 22 becomes output frame zero for every view. The preparation
backs up pre-start data, refines fixed iPhone poses against the Leica part cloud,
and fuses supported Leica points into the initial COLMAP seed. Final geometry
uses final RGB-D, with no untransformed initial scan mixed into it.

On the host, run `bash scripts/mask_hoi_datasets.sh` to generate all masks using
`grey fridge` for objects 001/002 and `white fridge` for 003/004, selecting the
annotated appliance through Leica projection guidance. Add `--max-frames 1` for
a first-frame-per-view trial. The runner uses grounded-sam2 on CUDA.


## Final preparation step: choose the RGB-D direction

Successful synthetic export and real mask/COLMAP preparation now finish with
`scripts/choose_rgbd_direction.py`. It compares masked valid-depth coverage at
the two ends of the sequence and saves a forward/backward recommendation.
To process all existing synthetic scenes and all real `final_*` episodes:

```bash
/home/samuelemara/miniconda3/envs/grounded-sam2/bin/python -B \
  /home/samuelemara/Joint3DGS/Sapien/scripts/choose_rgbd_direction.py \
  --root "/media/samuelemara/Extreme SSD/Samuele/Backup_09_10_26/Sapien/output" \
  --subset synthetic real-final --resume
```

Results stay in each scene's `metadata/direction/`: `summary.json`,
`per_frame.csv`, and `coverage.png`. This CPU-only command does not regenerate
masks, geometry, or renders. Add `--dry-run` to check layouts without writes.
The recommendation does not reorder frames or launch training. See
[direction selection and commands](docs/rgbd_direction.md) for the heuristic,
strict camera/frame checks, resume behavior, and separate dataset selections.
