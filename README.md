# SAPIEN URDF Multi-View Renderer (Docker-Only Python Environment)

This project renders articulated URDFs with **SAPIEN 2.2.2** and exports per-frame:
- RGB PNG
- metric depth (`.npy`, meters)
- binary foreground mask PNG
- optional per-camera RGB MP4

Everything Python-related is installed **inside Docker only**. You do not need to install project Python dependencies on the host or in conda.

## Host assumptions

- Host OS: Ubuntu 24.04
- Host CUDA stack: NVIDIA driver supporting CUDA 12.8
- Docker + NVIDIA Container Toolkit installed

Container base image is `nvidia/cuda:12.8.1-runtime-ubuntu22.04`.
Running Ubuntu 22.04 inside Docker is fine on an Ubuntu 24.04 host.

## 1) Install Docker + NVIDIA runtime on Ubuntu 24

If Docker is not installed yet:

```bash
sudo apt-get update
sudo apt-get install -y docker.io docker-compose-v2
sudo systemctl enable --now docker
sudo usermod -aG docker $USER
```

Install NVIDIA Container Toolkit:

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

Log out/in after adding your user to `docker` group.

Quick host checks:

```bash
nvidia-smi
docker run --rm --gpus all nvidia/cuda:12.8.1-runtime-ubuntu22.04 nvidia-smi
```

## 2) Build the image

From this repo root:

```bash
docker build -t urdf-sapien-renderer:latest .
```

This installs all Python dependencies inside the image from `requirements.txt`.

## 3) Run container (headless/offscreen)

Quick launcher:

```bash
./scripts/docker_enter_headless.sh
```

It auto-builds `urdf-sapien-renderer:latest` if missing.
You can override mounts with env vars:

```bash
ASSETS_DIR=/abs/assets OUTPUT_DIR=/abs/output ./scripts/docker_enter_headless.sh
```

Manual command:

```bash
docker run --rm -it \
  --gpus all \
  -e NVIDIA_DRIVER_CAPABILITIES=graphics,utility,compute,display \
  -v /absolute/path/to/this/repo:/workspace \
  -v /absolute/path/to/assets:/data/assets \
  -v /absolute/path/to/output:/data/output \
  urdf-sapien-renderer:latest \
  bash
```

Inside container, validate SAPIEN:

```bash
python -c "import sapien; print(sapien.__version__)"
python -m sapien.example.offscreen
```

If `output.png` is created, offscreen rendering is working.

## 4) Run container with GUI (X11)

Only needed if you want GUI windows (viewer/debug), not required for dataset rendering.

Quick launcher:

```bash
./scripts/docker_enter_gui.sh
```

It configures `xhost` access for local docker root and revokes it on exit.

Manual command:

On host, allow local docker root to access X server:

```bash
xhost +si:localuser:root
```

Run GUI-enabled container:

```bash
docker run --rm -it \
  --gpus all \
  -e DISPLAY=$DISPLAY \
  -e QT_X11_NO_MITSHM=1 \
  -e NVIDIA_DRIVER_CAPABILITIES=graphics,utility,compute,display \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  -v /absolute/path/to/this/repo:/workspace \
  -v /absolute/path/to/assets:/data/assets \
  -v /absolute/path/to/output:/data/output \
  urdf-sapien-renderer:latest \
  bash
```

After you finish, you can revoke X access:

```bash
xhost -si:localuser:root
```

## 5) Docker Compose usage

Headless shell:

```bash
ASSETS_DIR=/absolute/path/to/assets OUTPUT_DIR=/absolute/path/to/output \
  docker compose run --rm renderer
```

GUI shell:

```bash
xhost +si:localuser:root
ASSETS_DIR=/absolute/path/to/assets OUTPUT_DIR=/absolute/path/to/output \
  docker compose --profile gui run --rm renderer-gui
```

## 6) Use the renderer CLI inside Docker

Recommended end-to-end pipeline (4 Python commands):

```bash
# 1) Verify SAPIEN inside Docker
python -c "import sapien; print(sapien.__version__)"

# 2) Generate camera views JSON
python scripts/generate_cameras_json.py \
  --output configs/cameras.circle.json \
  --num-cams 8 \
  --radius 1.4 \
  --height 0.7 \
  --target 0 0 0.3 \
  --width 640 \
  --height-px 480 \
  --near 0.01 \
  --far 10.0 \
  --fov-y-deg 45

# 3) Inspect available joints in your URDF
python src/render_urdf_multiview.py list-joints \
  --urdf /data/assets/model.urdf

# 4) Run the rendering pipeline
python src/render_urdf_multiview.py render \
  --urdf /data/assets/model.urdf \
  --joint-name hinge_joint \
  --q-start 0.0 \
  --q-end 1.0 \
  --num-frames 60 \
  --views-json configs/cameras.circle.json \
  --output-dir /data/output/circle_demo \
  --save-video
```

### List joints

```bash
python src/render_urdf_multiview.py list-joints \
  --urdf /data/assets/model.urdf
```

### Render from direct args

```bash
python src/render_urdf_multiview.py render \
  --urdf /data/assets/model.urdf \
  --joint-name hinge_joint \
  --q-start 0.0 \
  --q-end 1.57 \
  --num-frames 120 \
  --views-json configs/camera_views.example.json \
  --output-dir /data/output/demo \
  --save-video
```

### Render from YAML config

```bash
python src/render_urdf_multiview.py render-config \
  --config configs/job.example.yaml
```

## 7) Camera JSON generator integration

You provided `scripts/generate_cameras_json.py`; it is now integrated with the renderer camera schema.

Generate circular camera views:

```bash
python scripts/generate_cameras_json.py \
  --output configs/cameras.circle.json \
  --num-cams 8 \
  --radius 1.4 \
  --height 0.7 \
  --target 0 0 0.3 \
  --width 640 \
  --height-px 480 \
  --near 0.01 \
  --far 10.0 \
  --fov-y-deg 45
```

Then render with it:

```bash
python src/render_urdf_multiview.py render \
  --urdf /data/assets/model.urdf \
  --joint-name hinge_joint \
  --q-start 0.0 \
  --q-end 1.0 \
  --num-frames 60 \
  --views-json configs/cameras.circle.json \
  --output-dir /data/output/circle_demo
```

The loader now supports both:
- native camera entries (`width/height/near/far/...`)
- generator-style nested fields (`intrinsics`, `pose`, optional `extrinsics`)

## 8) Output layout

```text
output_dir/
  metadata/
    job_config_resolved.yaml
    joint_info.json
    cameras.json
    frame_values.csv
    package_versions.json
  rgb/<camera_name>/frame_000000.png
  depth_npy/<camera_name>/frame_000000.npy
  depth_png/<camera_name>/frame_000000.png
  mask/<camera_name>/frame_000000.png
  video_rgb/<camera_name>.mp4
```

Depth policy:
- `.npy` stores metric depth in meters
- invalid/background depth is saved as `0.0`

Mask policy:
- foreground `255`, background `0`
- articulation mask uses segmentation IDs of articulation links

## 9) Troubleshooting

### `Cannot find a suitable rendering device`
- Confirm `--gpus all` is used.
- Confirm NVIDIA Container Toolkit is installed.
- Confirm Docker runtime is configured (`nvidia-ctk runtime configure --runtime=docker`).
- Confirm container includes required libs (`libegl1`, `libxext6`, `libvulkan1`).

### `python -m sapien.example.offscreen` fails
- Run `python -c "import sapien; print(sapien.__version__)"` first.
- Check host driver with `nvidia-smi`.
- Re-test with: `docker run --rm --gpus all ...`.

### GUI does not open
- Ensure X11 socket mount is present: `-v /tmp/.X11-unix:/tmp/.X11-unix`.
- Ensure `DISPLAY` is passed.
- Ensure `xhost +si:localuser:root` was run on host.
- Wayland users may need XWayland enabled.

## 10) Full preparation pipeline over all URDFs

Inside the container (`/workspace`), run:

```bash
python3 /workspace/scripts/run.py
```

The runner auto-fixes stale mesh paths, including absolute paths (for example `/home/.../assets/meshes/...`) and broken relative paths (for example `../meshes/...`), by remapping them to the active runtime URDF directory (for example `/workspace/assets/.../meshes/...`).

This script now auto-detects paths in Docker with this priority:
1. `/workspace/assets`
2. `<repo>/assets`
3. `/data/assets`

So if your repo is mounted at `/workspace`, no extra flags are needed.

Useful options:
You can set the URDF source folder with --urdf-dir (for example when URDFs are not in /workspace/assets).


```bash
# dry run (prints all commands)
python3 /workspace/scripts/run.py --dry-run

# save videos for each URDF job
python3 /workspace/scripts/run.py --save-video

# force deterministic per-link colors when URDF visuals have no material color/texture
python3 /workspace/scripts/run.py \
  --urdf-dir /workspace/assets/sapien_unified/sapien_unified \
  --force-link-colors \
  --force-color-seed paris_v1

# explicit paths (if your mounts differ)
python3 /workspace/scripts/run.py \
  --urdf-dir /workspace/assets \
  --output-root /workspace/output/preparation_all \
  --views-json /workspace/configs/cameras.pipeline.json

# use an existing camera config JSON directly (no regeneration if file exists)
python3 /workspace/scripts/run.py \
  --urdf-dir /workspace/assets \
  --camera-config-json /workspace/configs/my_cameras.json
```

Camera config behavior:
- `--camera-config-json` is treated as input and used directly for rendering.
- `--views-json` is the generation output path used when `--camera-config-json` is not provided.
- Without `--skip-camera-generation`, `run.py` regenerates `--views-json` each run.
- `--force-link-colors` injects colors into visuals that lack effective material styling.
- `--force-color-seed` controls deterministic color assignment across runs.

Output summary is written to:

```text
/workspace/output/preparation_all/pipeline_summary.json
```
