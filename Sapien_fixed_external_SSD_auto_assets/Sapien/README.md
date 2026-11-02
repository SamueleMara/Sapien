# SAPIEN Articulated Rendering Pipeline

> **External SSD setup (Ubuntu):** See [EXTERNAL_SSD.md](EXTERNAL_SSD.md) for the exact `Extreme SSD` mount paths, Docker launch commands, exFAT-compatible COLMAP export, and 64/100-camera examples.

This repository renders articulated URDF assets with SAPIEN, writes multiview
RGB/depth/mask sequences, and optionally exports a COLMAP-style sparse model
plus a part-labeled point cloud.

It is used here for two closely related workflows:

1. Render an existing URDF collection with fixed cameras and a moving joint.
2. Build definitive 2-link URDFs from PartNet-Mobility, then render them
   consistently and export the first-frame COLMAP data used by downstream 3DGS
   pipelines.

## What The Repo Contains

```text
src/
  render_urdf_multiview.py        direct SAPIEN CLI
  sapien_urdf_renderer.py         renderer, segmentation, frame sampling
  camera_utils.py                 camera loading and pose helpers
  config_types.py                 validated render-job schema
  io_utils.py                     output writers

scripts/
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

Recommended path: run everything inside Docker.

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
- mounts `assets/` and `output/`
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
/workspace/assets/partnet-mobility-v0/dataset
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
  --urdf-dir /workspace/assets/definitive_urdf/definitive_urdf \
  --output-root /workspace/output/preparation_definitive_urdf \
  --camera-config-json /workspace/configs/cameras.PARIS.json \
  --num-frames 60 \
  --save-video \
  --no-export-colmap
```

Render the same dataset and also export first-frame COLMAP and PLY outputs:

```bash
python3 scripts/run.py \
  --urdf-dir /workspace/assets/definitive_urdf/definitive_urdf \
  --output-root /workspace/output/preparation_definitive_urdf_colmap \
  --camera-config-json /workspace/configs/cameras.PARIS.json \
  --num-frames 60 \
  --save-video \
  --export-colmap \
  --colmap-max-frames 1
```

Important:

- `--colmap-max-frames 1` is the default.
- In that default mode, COLMAP and PLY use only `frame_000000`.
- Because only the first frame is used, COLMAP is independent from the chosen
  dynamic velocity profile.

## Workflow 2: Generate PartNet 2-Link URDFs

Use the static-context generator when you want PartNet visuals to match the
full object, not just the meshes explicitly referenced in `mobility.urdf`.

```bash
python3 scripts/generate_partnet_urdfs_with_static_context.py \
  --partnet-root /workspace/assets/partnet-mobility-v0/dataset \
  --subset-file /workspace/paris_subset_models.txt \
  --output-dir /workspace/assets/partnet_subset_fixed \
  --canonical-manifest-out /workspace/assets/partnet_subset_fixed/partnet_canonical_manifest.json
```

Expected laptop check:

```text
[OK] laptop_10211: joint=joint_1 fixed_meshes=105 moving_meshes=3
```

That confirms the extra static context meshes were included.

If you intentionally want the older `mobility.urdf`-only visual behavior:

```bash
python3 scripts/generate_partnet_urdfs_with_static_context.py \
  --partnet-root /workspace/assets/partnet-mobility-v0/dataset \
  --subset-file /workspace/paris_subset_models.txt \
  --output-dir /workspace/assets/partnet_subset_no_context \
  --no-include-result-static-meshes
```

## Workflow 3: Render Generated PartNet URDFs

Recommended command for the generated subset:

```bash
python3 scripts/run.py \
  --urdf-dir /workspace/assets/partnet_subset_fixed \
  --canonical-manifest-json /workspace/assets/partnet_subset_fixed/partnet_canonical_manifest.json \
  --output-root /workspace/output/partnet_subset_fixed_render_trapezoid \
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
  --urdf-dir /workspace/assets/partnet_subset_fixed \
  --canonical-manifest-json /workspace/assets/partnet_subset_fixed/partnet_canonical_manifest.json \
  --output-root /workspace/output/partnet_subset_fixed_static \
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
  --urdf-dir /workspace/assets/partnet_subset_fixed \
  --canonical-manifest-json /workspace/assets/partnet_subset_fixed/partnet_canonical_manifest.json \
  --output-root /workspace/output/partnet_subset_fixed_render_linear \
  --num-frames 60 \
  --velocity-profile linear \
  --save-video \
  --no-export-colmap
```

```bash
python3 scripts/run.py \
  --urdf-dir /workspace/assets/partnet_subset_fixed \
  --canonical-manifest-json /workspace/assets/partnet_subset_fixed/partnet_canonical_manifest.json \
  --output-root /workspace/output/partnet_subset_fixed_render_ease_in_out \
  --num-frames 60 \
  --velocity-profile ease-in-out \
  --save-video \
  --no-export-colmap
```

```bash
python3 scripts/run.py \
  --urdf-dir /workspace/assets/partnet_subset_fixed \
  --canonical-manifest-json /workspace/assets/partnet_subset_fixed/partnet_canonical_manifest.json \
  --output-root /workspace/output/partnet_subset_fixed_render_trapezoid \
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
  --urdf /workspace/assets/partnet_subset_fixed/laptop_10211.urdf \
  --as-json
```

Render one URDF directly:

```bash
python3 src/render_urdf_multiview.py render \
  --urdf /workspace/assets/partnet_subset_fixed/laptop_10211.urdf \
  --joint-name joint_1 \
  --q-start 0.0 \
  --q-end 0.35 \
  --num-frames 60 \
  --velocity-profile trapezoid \
  --velocity-profile-accel-fraction 0.2 \
  --views-json /workspace/configs/cameras.PARIS.json \
  --output-dir /workspace/output/laptop_direct_render \
  --save-video
```

Use this mode when you want full manual control over a single articulation.

## Workflow 5: Export COLMAP From Existing Renders

If a render output already exists and you only need COLMAP/PLY:

```bash
python3 scripts/export_colmap_dataset.py \
  --output-root /workspace/output/preparation_definitive_urdf_colmap \
  --urdf-dir /workspace/assets/definitive_urdf/definitive_urdf \
  --max-frames 1
```

Or for a single model:

```bash
python3 scripts/export_colmap_dataset.py \
  --model-output /workspace/output/preparation_definitive_urdf_colmap/USB_100109 \
  --urdf /workspace/assets/definitive_urdf/definitive_urdf/USB_100109.urdf \
  --max-frames 1
```

## Output Structure

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
    export_summary.json
  pointcloud/
    points3D_parts.ply
    joint_metadata.json
  metadata/
    colmap_export.json
```

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
python3 src/render_urdf_multiview.py list-joints --urdf /workspace/assets/partnet_subset_fixed/laptop_10211.urdf --as-json
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
