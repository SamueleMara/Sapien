# FreeArtGS: Articulated Gaussian Splatting Under Free-moving Scenario

<div align="center">
  <p>
    <a href="https://daihangpku.github.io/">Hang Dai<sup>*</sup></a>,
    <a href="https://hwfan.io/about-me/">Hongwei Fan<sup>*</sup></a>,
    <a href="https://github.com/Zarad0X">Han Zhang<sup>*</sup></a>,
    <a href="https://github.com/Wu-dolores">Duojin Wu</a>,
    <a href="https://jiyao06.github.io/">Jiyao Zhang</a>,
    <a href="https://zsdonghao.github.io/">Hao Dong</a>
  </p>
</div>
<div align="center">
  (* indicates equal contribution)
</div>
<div align="center">
  <strong>CVPR 2026</strong>
</div>
<div align="center">
<a href="https://arxiv.org/abs/2603.22102">
  <img src="https://img.shields.io/badge/arXiv-2603.22102-b31b1b" alt="arXiv">
</a>
<a href="https://freeartgs.github.io/">
  <img src="https://img.shields.io/badge/Project_Page-FreeArtGS-green" alt="Project Page">
</a>
</div>

<!-- <p align="center">
    <img src="assets/method.png" alt="Teaser" style="center" />
</p> -->

This repository contains the official implementation of [FreeArtGS: Articulated Gaussian Splatting Under Free-moving Scenario](https://freeartgs.github.io/).

<p align="center">
  <img src="docs/pipeline.png" alt="Overview of FreeArtGS" width="90%">
</p>


## Installation

Please follow [docs/install.md](docs/install.md) to set up the environment and required third-party dependencies.

## Data Preparation

### Local real RGB-D recordings: native format to masks

The local tools convert the native `videos/`, `depth/`, `meta/`, and `data/`
recording into the standard RGB-D/COLMAP layout, then generate foreground masks.
Run these commands yourself, in order. The complete format and options are
documented in [Real recording conversion](docs/real_recording_conversion.md).

Set the recording, a **new output folder**, and the segmentation prompt:

```bash
REAL_INPUT="/media/samuelemara/Extreme SSD/Samuele/Backup_09_10_26/Sapien/assets/dataset_fridge_plastic_depth_20261016_163314"
REAL_OUTPUT="/media/samuelemara/Extreme SSD/Samuele/Backup_09_10_26/Sapien/output/real/fridge_plastic_20261016_163314_v2"
MASK_PROMPT="red fridge"
```

The original `fridge_plastic_20261016_163314` output already exists. The example
uses `_v2` to keep it intact; choose another unused folder if necessary.

1. Preview the conversion without writing files:

   ```bash
   /home/samuelemara/miniconda3/envs/reartgs2/bin/python -B \
     /home/samuelemara/FreeArtGS/samuele/convert_real_recording.py \
     --input "$REAL_INPUT" --output "$REAL_OUTPUT" --dry-run
   ```

2. Convert RGB/depth and build both COLMAP models (CPU):

   ```bash
   /home/samuelemara/miniconda3/envs/reartgs2/bin/python -B \
     /home/samuelemara/FreeArtGS/samuele/convert_real_recording.py \
     --input "$REAL_INPUT" --output "$REAL_OUTPUT"
   ```

   This writes rectified `rgb/`, registered `depth_npy/` in meters,
   `depth_png/` in millimeters, camera metadata, `colmap/`, and `colmap_final/`.
   The initial and final point clouds merge the cameras separately at the first
   and last selected frames, using 5 mm voxels. Per-camera clouds and merged
   `points3D.ply` / `points3D_final.ply` are saved under `pointcloud/`.
   The output folder must not already exist; no source files are changed.

3. Generate masks from the converted RGB (CUDA):

   ```bash
   /home/samuelemara/miniconda3/envs/grounded-sam2/bin/python -B \
     /home/samuelemara/FreeArtGS/samuele/create_standard_masks.py \
     --input "$REAL_OUTPUT" --prompt "$MASK_PROMPT"
   ```

   Change `MASK_PROMPT` to describe the whole object. The required `--prompt`
   argument controls detection; it is not hardcoded. Masks are written as
   `mask/cam_NNN/frame_NNNNNN.png`, binary 0/255, at the converted RGB resolution.
   Matching completed masks are reused. Empty masks are identified in
   `metadata/mask_generation.json` and produce a nonzero exit status.

To regenerate masks in the **existing** converted dataset, including those
already generated with the same prompt:

```bash
/home/samuelemara/miniconda3/envs/grounded-sam2/bin/python -B \
  /home/samuelemara/FreeArtGS/samuele/create_standard_masks.py \
  --input "/media/samuelemara/Extreme SSD/Samuele/Backup_09_10_26/Sapien/output/real/fridge_plastic_20261016_163314" \
  --prompt "red fridge" --overwrite
```

`--overwrite` replaces masks, so omit it to reuse matching results. Changing a
previously used prompt or thresholds requires `--overwrite`. Masks must use the
rectified output RGB, not the native MP4 frames. Calibration retains its original
`CANDIDATE_NOT_VALIDATED` status; no joint GT is fabricated.

### Official datasets

Download FreeArt-21:

```bash
pip install gdown
gdown 1hl6s3C3919gXKvHC0ZfgrxWqFCKM3XXf -O ./
unzip ./FreeArt-21_v1.zip -d .
```

Download the MonoMobility evaluation data:

```bash
gdown 1K0z1_Gtk9G_51_o61OAhBjo1hxrDCFey -O ./
unzip ./data.zip -d .
```

## Quick Start

For the multi-camera datasets already in this workspace, see
[Joint training with the local datasets](docs/local_datasets.md) to train each
object using all eight calibrated cameras in one shared model.
The local batch stores results, prepared datasets, converted inputs, configs and
runtime caches under the SSD target selected by `--outputs-root`.

FreeArtGS is driven by a single YAML config. The example below runs object `100109` from FreeArt-21 with [configs/freeart-21.yaml](configs/freeart-21.yaml).

### 1. Select the Matplotlib Backend

Set `MPLBACKEND` before running the pipeline. For interactive local runs:

```bash
export MPLBACKEND=TkAgg
```

For headless/server runs:

```bash
export MPLBACKEND=Agg
```

### 2. Run the Full Pipeline

The full pipeline performs preprocessing, two-part tracking, reconstruction, blending, articulation optimization, and evaluation:

```bash
bash pipelines/freeartgs.sh --config configs/freeart-21.yaml --object_name 100109
```

Intermediate and final results will be saved under `datasets/100109/` and `outputs/100109/`.

You can also run individual stages:

```bash
bash pipelines/freeartgs.sh --config configs/freeart-21.yaml --object_name 100109 --preprocess_only
bash pipelines/freeartgs.sh --config configs/freeart-21.yaml --object_name 100109 --tracking_only
bash pipelines/freeartgs.sh --config configs/freeart-21.yaml --object_name 100109 --reconstruction_only
```

### 3. Evaluate an Existing Result

After reconstruction has finished, run:

```bash
bash pipelines/evaluate_freeartgs.sh --object_name 100109
```

### 4. Visualize Results

Visualize the two-part blend result:

```bash
python reconstruction/visualize_twopart_blend.py \
  --input-dir outputs/100109/twopart_blend \
  --motion_part 0
```

Use a hard part mask instead of the default soft blending:

```bash
python reconstruction/visualize_twopart_blend.py \
  --input-dir outputs/100109/twopart_blend \
  --motion_part 0 \
  --hard \
  --threshold 0.5
```

Visualize the final articulated Gaussian result:

```bash
python reconstruction/visualize_articulation.py \
  --input-dir outputs/100109/splatfacto-art \
  --motion_part 0 \
  --soft
```

The articulation viewer can render with GSplat and record frames periodically:

```bash
python reconstruction/visualize_articulation.py \
  --input-dir outputs/100109/splatfacto-art \
  --motion_part 0 \
  --use-gsplat-render \
  --record \
  --record-interval 0.1
```

Common controls: `A/D` step backward/forward, `R` reset, `M` toggle the moving part, `J/K` adjust the mask threshold, and `S` toggle soft/hard visualization. The articulation viewer also supports `P` to save a capture and axis/origin adjustment keys printed in the terminal.

## Open-source TODO List

- [x] Release installation instructions.
- [x] Release FreeArt-21 data.
- [x] Release the main FreeArtGS pipeline config and scripts for FreeArt-21.
- [x] Release evaluation scripts.
- [x] Release visualization tools.
- [ ] Release the pipeline config and scripts for real data.
