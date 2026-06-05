# CONTEXT.md — Codex implementation brief for Dockerized SAPIEN multi-view articulated rendering

## Goal

Implement a **Docker-based, Python-first rendering toolchain** that:

1. Installs **SAPIEN 2.2.x** inside Docker.
2. Loads a **URDF articulation** from disk.
3. Lets the user choose:
   - the **active joint** by name,
   - the **initial** and **final** joint values,
   - either the **number of frames** or **fps + duration**,
   - one or more **camera views**.
4. Renders, for **every frame and every camera**:
   - **RGB** image,
   - **depth** image,
   - **foreground binary mask**.
5. Optionally writes an **MP4 video** for the RGB sequence of each camera.
6. Exposes a clean **Python API** and a **CLI**.

This must be robust enough for synthetic dataset generation.

---

## Why SAPIEN and not Open3D

Use **SAPIEN**, not Open3D, as the primary engine.

Reason:
- SAPIEN natively supports **URDF articulations** and joint state control.
- SAPIEN has an offscreen camera pipeline for **RGB, depth, and segmentation**.
- Open3D is useful as a renderer but is not the best fit for a native “load URDF and animate joints over time” workflow.

Target **SAPIEN 2.2.2** specifically, not 3.x.

Why:
- SAPIEN 2.2 docs directly show URDF loading, articulation control, `set_qpos`, and camera capture.
- SAPIEN 3.x introduced a **major API overhaul**, so pinning to 2.2.2 avoids ambiguity and keeps implementation aligned with stable docs.

---

## External facts Codex should respect

These are the assumptions behind the implementation and install process:

1. SAPIEN is installable from PyPI with:
   - `pip install sapien`
2. SAPIEN supports Linux and Python 3.7–3.11 in the 2.2 docs.
3. For offscreen rendering on a server, the minimum system dependencies called out are:
   - `libegl1`
   - `libxext6`
4. In NVIDIA Docker, GPU capabilities should include:
   - `graphics,utility,compute`
5. SAPIEN provides offscreen examples and URDF articulation tutorials.
6. SAPIEN’s `Position` texture should be used to derive depth.
7. Actor/segmentation output should be used to derive foreground masks.

Do not drift to a different engine unless there is a blocking issue.

---

## Deliverables Codex must create

Codex must create the following files:

1. `Dockerfile`
2. `docker-compose.yml` (optional but preferred)
3. `requirements.txt`
4. `src/render_urdf_multiview.py`
5. `src/sapien_urdf_renderer.py`
6. `src/camera_utils.py`
7. `src/io_utils.py`
8. `src/types.py` or equivalent dataclasses file
9. `configs/camera_views.example.json`
10. `configs/job.example.yaml`
11. `README.md`
12. `scripts/run_demo.sh`
13. `scripts/list_joints.sh`

Keep the implementation modular.

---

## Environment and container requirements

### Primary target
- Host OS: Ubuntu Linux with NVIDIA GPU
- Container runtime: Docker with NVIDIA Container Toolkit
- Renderer mode: **headless offscreen EGL**

### Base image
Use an NVIDIA CUDA runtime or devel image on Ubuntu 22.04.
A safe default is something like:
- `nvidia/cuda:12.1.1-runtime-ubuntu22.04`

A devel image is also acceptable if needed.

### Required apt packages
Install at least:
- `python3`
- `python3-pip`
- `python3-venv`
- `git`
- `ffmpeg`
- `libegl1`
- `libxext6`
- `libglib2.0-0`
- `libsm6`
- `libxrender1`
- `libgomp1`
- `ca-certificates`

Optional but useful:
- `vim`
- `curl`
- `unzip`

### Required environment variables
Set in Dockerfile:

```dockerfile
ENV NVIDIA_DRIVER_CAPABILITIES=graphics,utility,compute
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive
```

Do **not** rely on X11 display forwarding.
Use offscreen rendering only.

### Python dependencies
Pin versions conservatively.
At minimum:
- `sapien==2.2.2`
- `numpy`
- `imageio`
- `imageio-ffmpeg`
- `opencv-python-headless`
- `Pillow`
- `PyYAML`
- `tqdm`

Optional:
- `typer` or `click` for CLI
- `pydantic` for config validation

Do not introduce heavy unnecessary dependencies.

---

## Docker requirements

### Dockerfile behavior
The Docker image must:
1. Install all system deps.
2. Install Python deps.
3. Copy the project into `/workspace`.
4. Default working directory to `/workspace`.
5. Be runnable with mounted host folders for:
   - input URDF/assets
   - output renders

### Runtime assumptions
The container should be started with GPU access, e.g.:

```bash
docker run --rm -it \
  --gpus all \
  -v /absolute/path/to/project:/workspace \
  -v /absolute/path/to/assets:/data/assets \
  -v /absolute/path/to/output:/data/output \
  urdf-sapien-renderer:latest \
  bash
```

If using docker compose, it must request NVIDIA GPUs in a current-compatible way.

### Validation commands inside container
Codex must make the README tell the user to validate installation with:

```bash
python -c "import sapien; print(sapien.__version__)"
python -m sapien.example.offscreen
```

If `python -m sapien.example.offscreen` generates `output.png`, consider install successful.
Warnings about missing display can be acceptable in headless mode as long as rendering succeeds.

---

## Functional requirements

### Inputs
The renderer must support all of the following inputs.

#### Required
- `urdf_path: str`
- `joint_name: str`
- `q_start: float`
- `q_end: float`
- one of:
  - `num_frames: int`
  - or `fps: float` and `duration_sec: float`
- camera definitions file path
- output directory

#### Optional
- `joint_space: bool = True`
- `root_pose` for placing the whole articulation in world coordinates
- `image_width: int`
- `image_height: int`
- near plane
- far plane
- whether to save PNG depth preview
- whether to save `.npy` metric depth
- whether to save MP4
- video codec / fps override
- mask mode
- background color
- dry run / list-joints-only

### Camera definitions
Support multiple camera views from JSON or YAML.
Each camera entry must include:
- `name`
- either:
  - `pose_matrix_4x4`
  - or `(position, target, up)` look-at spec
- `width`
- `height`
- `fov_y_deg` or intrinsics
- near/far values

Preferred schema:

```json
{
  "cameras": [
    {
      "name": "cam_front",
      "width": 640,
      "height": 480,
      "fov_y_deg": 45.0,
      "near": 0.01,
      "far": 10.0,
      "position": [1.2, 0.0, 0.8],
      "target": [0.0, 0.0, 0.3],
      "up": [0.0, 0.0, 1.0]
    }
  ]
}
```

### Frame scheduling
Support both modes:

#### Mode A: exact frame count
If `num_frames` is provided, linearly interpolate from `q_start` to `q_end` over exactly that many frames, inclusive of endpoints.

#### Mode B: fps + duration
If `fps` and `duration_sec` are provided, compute:

```python
num_frames = round(fps * duration_sec)
```

Then generate a trajectory inclusive of endpoints.
Document clearly whether the final frame includes `q_end`.
It should.

### Joint interpolation
Assume the selected joint is a 1-DoF revolute or prismatic joint.
Interpolate linearly in joint coordinate:

```python
q_t = np.linspace(q_start, q_end, num_frames)
```

Do not try to support multi-DoF joints in v1.
If the chosen joint has DoF != 1, fail with a clear error.

### URDF loading
Load the URDF using SAPIEN’s URDF loader.
Use a **kinematic articulation** unless dynamic simulation is explicitly needed.
The articulation is being posed, not simulated.

Requirements:
- load successfully from host-mounted path
- support mesh assets referenced relative to the URDF location
- optionally apply a root pose transform

### Joint discovery
Provide a mode to print all active joints and exit.
For each active joint, print at least:
- joint name
- joint type if available
- index in active joint list
- DoF / limits if available

This is needed so the user can discover `joint_name`.

### Rendering outputs
For each frame and camera save:

#### RGB
- save as `uint8` PNG
- shape `(H, W, 3)`
- filename pattern:
  - `rgb/<camera_name>/frame_000000.png`

#### Depth
Save **metric depth** primarily as `.npy`.
Also optionally save a 16-bit PNG preview.

Preferred outputs:
- `depth_npy/<camera_name>/frame_000000.npy`
- `depth_png/<camera_name>/frame_000000.png`

Depth requirements:
- represent distance in **meters**
- background / invalid pixels should be either:
  - `0.0`
  - or `np.nan`

Pick one policy and document it consistently.
Preferred: use `0.0` for invalid/background in saved outputs.

Important:
- derive depth from SAPIEN camera output correctly
- do not store arbitrary normalized depth unless explicitly labeled as preview only

#### Foreground binary mask
Save as single-channel PNG:
- `mask/<camera_name>/frame_000000.png`

Mask values:
- foreground = `255`
- background = `0`

The foreground mask should correspond to the articulated object only.

### Video output
If enabled, write one RGB MP4 per camera:
- `video_rgb/<camera_name>.mp4`

Use `imageio` or ffmpeg.
Do not make video writing mandatory.

---

## Foreground mask definition

The mask must represent the URDF articulation and not the background.

### Default policy
Use actor-level segmentation from SAPIEN and mark pixels belonging to the articulation’s links as foreground.

### Important implementation detail
Do **not** simply use `segmentation > 0` if there may be other scene objects.
Instead:
1. enumerate all actors/entities belonging to the loaded articulation,
2. collect their actor/segmentation IDs,
3. create a binary mask where pixel label is in that set.

If actor IDs are not directly exposed in the most convenient form, implement a tested mapping layer.

### Empty background
If the scene only contains the articulation, a fallback of `segmentation > 0` may work, but do not rely on this as the primary logic.

---

## Depth extraction requirements

Depth extraction must be implemented carefully.

Use the SAPIEN camera output documented for position/depth retrieval.
The implementation must:
1. capture the relevant texture after rendering,
2. convert it to metric depth in camera coordinates,
3. save exact metric depth.

Also provide a helper function:

```python
def save_depth_png_preview(depth_m: np.ndarray, out_path: str, max_depth_m: float | None = None) -> None:
    ...
```

This preview should be clearly labeled as visualization only.
The `.npy` file is the authoritative depth output.

---

## Multi-view support

Support any number of cameras in one run.

Two valid implementation patterns:
1. create multiple mounted cameras in the scene, or
2. reuse one camera and update its pose per view.

Preferred:
- keep code simple and reliable
- if multiple cameras are supported cleanly, use them
- otherwise one camera reused sequentially is acceptable

The output folder must be grouped by camera name.

---

## Scene setup requirements

### Lighting
Add reasonable default lights so meshes render clearly.
Use at least:
- ambient light
- one or more directional or point lights

The scene should render with stable lighting across frames.

### Ground plane
Do **not** add a ground plane by default.
Reason: it complicates foreground masks.

Allow an optional `--add-ground` flag later if useful, but default should be no ground.

### Background
Default background should be black or another known constant.
Document it.
This is mainly for RGB appearance; mask must not rely on background color.

---

## Python API design

Codex must expose a reusable Python API, not only a CLI.

Preferred structure:

```python
class CameraSpec:
    name: str
    width: int
    height: int
    near: float
    far: float
    fov_y_deg: float | None
    position: list[float] | None
    target: list[float] | None
    up: list[float] | None
    pose_matrix_4x4: list[list[float]] | None

class RenderJobConfig:
    urdf_path: str
    joint_name: str
    q_start: float
    q_end: float
    num_frames: int | None
    fps: float | None
    duration_sec: float | None
    cameras: list[CameraSpec]
    output_dir: str
    save_video: bool = False
    save_depth_png: bool = True
    save_depth_npy: bool = True
    image_width: int | None = None
    image_height: int | None = None

class SapienURDFRenderer:
    def __init__(self, ...): ...
    def load_articulation(self, urdf_path: str, root_pose=None) -> None: ...
    def list_active_joints(self) -> list[dict]: ...
    def set_joint_value(self, joint_name: str, value: float) -> None: ...
    def render_frame(self, camera: CameraSpec) -> dict[str, np.ndarray]: ...
    def render_sequence(self, cfg: RenderJobConfig) -> None: ...
```

Use dataclasses if simpler.

---

## CLI requirements

The CLI must support at least these commands.

### 1. List joints

```bash
python src/render_urdf_multiview.py list-joints \
  --urdf /data/assets/model.urdf
```

This should print all active joints and exit.

### 2. Render sequence from direct args

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

### 3. Render from job config

```bash
python src/render_urdf_multiview.py render-config \
  --config configs/job.example.yaml
```

The YAML config mode is strongly preferred.

---

## Output directory layout

Codex must make the output deterministic and easy to parse.

Use this layout:

```text
output_dir/
  metadata/
    job_config_resolved.yaml
    joint_info.json
    cameras.json
    frame_values.csv
  rgb/
    cam_front/
      frame_000000.png
      ...
    cam_side/
      frame_000000.png
  depth_npy/
    cam_front/
      frame_000000.npy
  depth_png/
    cam_front/
      frame_000000.png
  mask/
    cam_front/
      frame_000000.png
  video_rgb/
    cam_front.mp4
```

### Metadata requirements
Save metadata files so results are reproducible.

At minimum save:
- resolved config
- actual framewise joint values
- selected joint info
- camera definitions
- package versions if possible

---

## Error handling expectations

Implementation must fail clearly for these cases:

1. URDF path does not exist.
2. URDF loads but references missing mesh assets.
3. Joint name not found.
4. Selected joint has DoF != 1.
5. No cameras provided.
6. Neither `num_frames` nor `fps + duration_sec` provided.
7. Both modes are invalid or inconsistent.
8. Output directory not writable.
9. SAPIEN rendering device unavailable in container.

When rendering device initialization fails, print a helpful message that mentions:
- checking `--gpus all`
- checking NVIDIA Container Toolkit
- checking `libegl1` and `libxext6`
- checking `NVIDIA_DRIVER_CAPABILITIES=graphics,utility,compute`

---

## Non-goals for v1

Do **not** implement these unless trivial:
- physics simulation
- collision checking
- contact dynamics
- multi-joint coupled trajectories
- motion planning
- randomization engine
- domain randomization
- COCO export
- ROS integration
- GUI viewer

Stay focused on deterministic articulated rendering.

---

## README requirements

The README must include:

1. What the tool does.
2. Why SAPIEN 2.2.2 is pinned.
3. How to build Docker image.
4. How to run the container with GPU.
5. How to validate SAPIEN offscreen rendering.
6. How to list joints from a URDF.
7. How to run a render job.
8. Exact output layout.
9. Troubleshooting section for common headless GPU errors.

### Troubleshooting section must mention
- missing GPU inside container
- `Cannot find a suitable rendering device`
- Vulkan/EGL-related failures
- missing `output.png` from `sapien.example.offscreen`
- user should verify host NVIDIA driver and docker GPU access

---

## Code quality requirements

- Python 3.10+ preferred
- type hints throughout
- small, testable helper functions
- no notebook-only code
- no hardcoded absolute paths
- no hidden global state
- clear logging
- progress bar during rendering

Prefer standard logging over ad hoc prints.

---

## Suggested implementation plan for Codex

1. Create Dockerfile and requirements.
2. Verify SAPIEN import and offscreen example inside container.
3. Implement config dataclasses and loaders.
4. Implement camera spec parsing.
5. Implement URDF loader and active joint discovery.
6. Implement single-frame render returning RGB/depth/segmentation.
7. Implement articulation-only foreground mask extraction.
8. Implement sequence loop and file saving.
9. Implement optional MP4 export.
10. Add CLI wrappers.
11. Write README and example configs.

---

## Acceptance criteria

The task is complete only if all of the following are true:

1. `docker build` succeeds.
2. Running the container with `--gpus all` allows:
   - `import sapien`
   - `python -m sapien.example.offscreen`
3. A URDF can be loaded.
4. Joint names can be listed.
5. A specified 1-DoF joint can be animated from start to end.
6. For each frame and view, RGB/depth/mask are saved.
7. Depth is saved in metric units as `.npy`.
8. Mask isolates the articulation, not just any non-background pixel.
9. The CLI works in both direct-args mode and config-file mode.
10. README instructions are sufficient for a fresh user to reproduce the setup.

---

## Example user workflow Codex should optimize for

```bash
# Build image
docker build -t urdf-sapien-renderer .

# Run container
docker run --rm -it \
  --gpus all \
  -v /home/user/urdf_project:/workspace \
  -v /home/user/datasets:/data \
  urdf-sapien-renderer

# Validate install
python -c "import sapien; print(sapien.__version__)"
python -m sapien.example.offscreen

# Inspect joints
python src/render_urdf_multiview.py list-joints --urdf /data/my_robot/model.urdf

# Render a job
python src/render_urdf_multiview.py render \
  --urdf /data/my_robot/model.urdf \
  --joint-name drawer_joint \
  --q-start 0.0 \
  --q-end 0.25 \
  --num-frames 60 \
  --views-json configs/camera_views.example.json \
  --output-dir /data/output/drawer_demo \
  --save-video
```

---

## Final instruction to Codex

Implement the full project exactly as described above, with Docker as the default installation path and SAPIEN 2.2.2 as the pinned rendering backend. Prefer correctness, reproducibility, and explicitness over clever abstractions.
