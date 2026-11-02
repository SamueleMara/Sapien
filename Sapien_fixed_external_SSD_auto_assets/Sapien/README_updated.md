# SAPIEN Output Generation

This repo uses SAPIEN inside Docker to generate the two output trees used here:

```text
/home/samuelemara/Joint3DGS/Sapien/output/preparation_definitive_urdf
/home/samuelemara/Joint3DGS/Sapien/output/preparation_definitive_urdf_colmap
```

Run the commands from inside the project container, where the repo is mounted at `/workspace`.

```bash
./scripts/docker_enter_headless.sh
```

Inside Docker, verify the renderer environment:

```bash
python -c "import sapien; print(sapien.__version__)"
```

## Renderer Troubleshooting

If the pipeline fails with messages like:

```text
[svulkan2] [error] Vulkan is incompatible with your driver
```

or:

```text
nvidia-smi
No devices were found
```

then the failure is in the container renderer runtime, not in the URDF or mesh selection code. In that state, `render_urdf_multiview.py list-joints` may still print valid JSON, but the actual `render` step will die at `Rendering: 0%`.

After a reboot, check the container again before launching a full dataset run:

```bash
./scripts/docker_enter_headless.sh
nvidia-smi
python -c "import sapien.core as sapien; e=sapien.Engine(); r=sapien.SapienRenderer(offscreen_only=True); e.set_renderer(r); print('renderer init ok')"
```

If `nvidia-smi` still reports no devices, stop there and fix the Docker/NVIDIA runtime first. Re-running the batch pipeline will not help.

`scripts/run.py` now performs a real render preflight before the batch loop. If the renderer is still broken, it should fail early instead of spending time iterating over every URDF.


## Generate PartNet URDFs With Static Context Meshes

Some PartNet-Mobility objects contain visual geometry that is present in
`result.json`, `result_after_merging.json`, `point_sample`, and `textured_objs`,
but is not referenced by `mobility.urdf`. For example, `laptop/10211` contains
the laptop plus the static table/desktop context. If the URDF generator reads
only `mobility.urdf`, those static context meshes are dropped and SAPIEN renders
only the articulated laptop parts.

Use `scripts/generate_partnet_urdfs_with_static_context.py` to generate the
2-link articulated URDFs while also adding the missing static context meshes to
`base_link`.

The subset file is a plain text list of objects and joint ids:

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

To generate the fixed URDF set from that subset:

```bash
python scripts/generate_partnet_urdfs_with_static_context.py \
  --partnet-root /workspace/assets/partnet-mobility-v0/dataset \
  --subset-file /workspace/paris_subset_models.txt \
  --output-dir /workspace/assets/partnet_subset_fixed \
  --canonical-manifest-out /workspace/assets/partnet_subset_fixed/partnet_canonical_manifest.json
```

For `laptop/10211`, the expected log line should include the static table/context
meshes:

```text
[OK] laptop_10211: joint=joint_1 fixed_meshes=105 moving_meshes=3
```

That gives `105 + 3 = 108` visual meshes, matching the complete PartNet object
instead of the smaller `mobility.urdf`-only visual set.

Static context meshes are included by default. To reproduce the old
`mobility.urdf`-only behavior, disable the extra context meshes:

```bash
python scripts/generate_partnet_urdfs_with_static_context.py \
  --partnet-root /workspace/assets/partnet-mobility-v0/dataset \
  --subset-file /workspace/paris_subset_models.txt \
  --output-dir /workspace/assets/partnet_subset_no_context \
  --no-include-result-static-meshes
```

## Rendering Generated PartNet URDFs

After generating `/workspace/assets/partnet_subset_fixed`, render from that folder
rather than from the old URDF directory.

Use `--joint-delta` to control how far the selected joint moves during the render:

```text
not provided       keep the previous automatic q_end selection
--joint-delta 0    keep the object static at q_start
--joint-delta -0.35 render from q_start to q_start - 0.35
--joint-delta 0.35 render from q_start to q_start + 0.35
```

The requested delta is clamped to the joint limits when limits are available.

Static one-frame sanity check for the full object, including static context:

```bash
python scripts/run_with_joint_delta.py \
  --urdf-dir /workspace/assets/partnet_subset_fixed \
  --canonical-manifest-json /workspace/assets/partnet_subset_fixed/partnet_canonical_manifest.json \
  --output-root /workspace/output/partnet_subset_fixed_static \
  --joint-delta 0 \
  --num-frames 1
```

Animated render with a small joint displacement:

```bash
python scripts/run_with_joint_delta.py \
  --urdf-dir /workspace/assets/partnet_subset_fixed \
  --canonical-manifest-json /workspace/assets/partnet_subset_fixed/partnet_canonical_manifest.json \
  --output-root /workspace/output/partnet_subset_fixed_render \
  --joint-delta -0.35 \
  --num-frames 60 \
  --save-video
```

## Base Output

Create `/workspace/output/preparation_definitive_urdf` with RGB, depth, mask, metadata, and optional videos, but without COLMAP/PLY post-export:

```bash
python /workspace/scripts/run.py \
  --urdf-dir /workspace/assets/definitive_urdf/definitive_urdf \
  --output-root /workspace/output/preparation_definitive_urdf \
  --camera-config-json /workspace/configs/cameras.PARIS.json \
  --num-frames 60 \
  --save-video \
  --no-export-colmap
```

Each model output has this structure:

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
    job_config_resolved.yaml
    joint_info.json
    package_versions.json
    visual_materials.json
```

## COLMAP Output

Create `/workspace/output/preparation_definitive_urdf_colmap` with the same rendered structure plus a COLMAP dataset and a part-labeled PLY point cloud:

```bash
python /workspace/scripts/run.py \
  --urdf-dir /workspace/assets/definitive_urdf/definitive_urdf \
  --output-root /workspace/output/preparation_definitive_urdf_colmap \
  --camera-config-json /workspace/configs/cameras.PARIS.json \
  --num-frames 60 \
  --save-video \
  --export-colmap \
  --colmap-max-frames 1
```

`--colmap-max-frames 1` means the COLMAP model and PLY are generated only from `frame_000000`. Use `--colmap-max-frames 0` to export all frames.

Additional output per model:

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

The PLY includes a `part` vertex field:

```text
part=0  static rendered part, selected joint parent / first URDF link, usually base_link
part=1  moving rendered part, selected joint child / second URDF link, usually moving_link
```

The labels come from rendered SAPIEN actor segmentation, not from color heuristics.

The point cloud also carries a global `joint_metadata` payload. In `points3D_parts.ply` it is encoded as a PLY header comment:

```text
comment joint_metadata {"type":"...","point_of_application":[...],"axis_vector":[...]}
```

The same payload is written to `pointcloud/joint_metadata.json` and embedded in `metadata/colmap_export.json`. It contains:

```text
type                  URDF joint type, for example revolute, prismatic, continuous
point_of_application  joint origin transformed into the rendered world frame
axis_vector           normalized joint axis transformed into the rendered world frame
```

The original URDF `origin xyz`, `origin rpy`, and `axis xyz` are kept in the metadata as `urdf_origin_xyz`, `urdf_origin_rpy`, and `urdf_axis`.

## Add COLMAP To Existing Base Output

If `/workspace/output/preparation_definitive_urdf` already exists and you only want to create the COLMAP version, copy the base tree and run the standalone exporter:

```bash
rsync -a --delete \
  /workspace/output/preparation_definitive_urdf/ \
  /workspace/output/preparation_definitive_urdf_colmap/

python /workspace/scripts/export_colmap_dataset.py \
  --output-root /workspace/output/preparation_definitive_urdf_colmap \
  --urdf-dir /workspace/assets/definitive_urdf/definitive_urdf \
  --max-frames 1
```

The exporter replays the first rendered frame in SAPIEN to recover actor segmentation and sample the part-labeled point cloud.

## Single-Model Test

For the PARIS subset line:

```text
USB/100109 0
```

use model folder `USB_100109` and URDF `USB_100109.urdf`.

```bash
python /workspace/scripts/export_colmap_dataset.py \
  --model-output /workspace/output/preparation_definitive_urdf_colmap/USB_100109 \
  --urdf /workspace/assets/definitive_urdf/definitive_urdf/USB_100109.urdf \
  --max-frames 1
```

Expected files after the test:

```text
/workspace/output/preparation_definitive_urdf_colmap/USB_100109/colmap/sparse/0/images.txt
/workspace/output/preparation_definitive_urdf_colmap/USB_100109/colmap/sparse/0/points3D.txt
/workspace/output/preparation_definitive_urdf_colmap/USB_100109/pointcloud/points3D_parts.ply
/workspace/output/preparation_definitive_urdf_colmap/USB_100109/pointcloud/joint_metadata.json
/workspace/output/preparation_definitive_urdf_colmap/USB_100109/metadata/colmap_export.json
```

## Relevant Code

```text
scripts/generate_partnet_urdfs_with_static_context.py  PartNet URDF generator with missing static context meshes
scripts/run_with_joint_delta.py                        batch renderer with optional --joint-delta control
scripts/run.py                                         full batch renderer and optional COLMAP/PLY export
scripts/export_colmap_dataset.py                       COLMAP/PLY exporter from rendered outputs
scripts/read_write_model.py                            COLMAP cameras/images/points3D writer
src/render_urdf_multiview.py                           direct renderer CLI
src/sapien_urdf_renderer.py                            SAPIEN rendering, depth, masks, segmentation
```
