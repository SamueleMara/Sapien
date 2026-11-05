# Choose the RGB-D starting direction

`scripts/choose_rgbd_direction.py` is the last preparation step after successful
mask/COLMAP export. It recommends starting from the endpoint with more valid
foreground observations. Synthetic `scripts/run.py` and the standalone COLMAP
exporter run it after export; render-only `run.py --no-export-colmap` runs it
after rendering. Real text-mask generation, spatial mask correction, and
`samuele/colmap_masking.py` run it after rebuilding both masked endpoint models.
Standalone `scripts/mask_colmap_models.py` also runs it after a successful scene
update. Dry runs do not generate direction outputs.

## Run on all existing synthetic and real final episodes

```bash
/home/samuelemara/miniconda3/envs/grounded-sam2/bin/python -B \
  /home/samuelemara/Joint3DGS/Sapien/scripts/choose_rgbd_direction.py \
  --root "/media/samuelemara/Extreme SSD/Samuele/Backup_09_10_26/Sapien/output" \
  --subset synthetic real-final \
  --resume
```

Discovery selects synthetic scenes with `metadata/job_config_resolved.yaml`
under every camera-count/motion-profile folder, and converted real scenes named
`final_*_episode_NNNNNN` with `metadata/real_conversion.json`. The present tree
contains 61 synthetic scenes (41 with eight cameras, ten with 64, ten with 100)
and 110 real final episodes. Older real recordings and HoI scenes are excluded.
No masks, renders, conversions, or COLMAP reconstruction are rerun by this
command. This check runs on CPU and reads all selected mask/depth frames.

For synthetic scenes only, use `--subset synthetic`; for real final episodes
only, use `--subset real-final`. Add `--dry-run` to check camera/frame layouts
without reading pixel values or writing files. A failed scene is reported, the
other scenes continue, and the command exits nonzero if any scene fails.

`--resume` reuses reports only when the algorithm version, endpoint fraction,
and input mask/depth filenames, sizes, and nanosecond modification times match.
Changed masks or depth trigger recounting. Omit `--resume` to force recounting.
The default endpoint fraction is 0.10; change it with `--fraction` in (0, 0.5].

For one scene:

```bash
/home/samuelemara/miniconda3/envs/grounded-sam2/bin/python -B \
  /home/samuelemara/Joint3DGS/Sapien/scripts/choose_rgbd_direction.py \
  "/absolute/path/to/scene"
```

A ZIP with paths directly under `mask/cam_*/`, `depth_npy/cam_*/` or
`depth_png/cam_*/` remains supported; give an explicit `--out` for ZIP input.

## Outputs and decision

Each scene receives:

```text
metadata/direction/
  summary.json
  per_frame.csv
  coverage.png
  .runtime/matplotlib/   # plot cache, also within the selected scene
```

`summary.json` contains `recommended_direction` (`FORWARD` or `BACKWARD`), the
raw `mask_and_valid_depth.direction`, both endpoint scores, per-camera scores,
frame/camera counts, depth sources, and the input fingerprint. `per_frame.csv`
contains mask and valid-depth pixel counts per camera and in total.

For each camera, take the median count over the first and last
`ceil(num_frames * fraction)` frames, then sum the camera medians. The higher
sum determines the recommendation. Metric `depth_npy` takes precedence over
PNG depth; valid depth is finite and strictly positive. The synthetic PNGs
are previews, so the NPY data avoid preview quantization affecting validity.
PNG-only input is supported when NPY data are absent for that view. A partial
NPY sequence is an error and does not silently fall back to PNGs.

Camera sets must match camera metadata (and RGB when present), and every view
must have exactly the same frame IDs for masks, RGB, and selected depth.
Partial masks, missing views, stale extra frames, incomplete conversions, and
shape mismatches are errors. The code does not silently intersect frame sets.

An exact nonzero tie records `TIE` and recommends `FORWARD` to retain acquisition
order. If neither endpoint has any valid foreground depth, the decision is
`UNAVAILABLE` and the recommendation is null; batch/pipeline execution reports
a failure. It does not pretend that zero-depth endpoints provide evidence.

This is a coverage heuristic. It does not measure correspondence, unique 3D
surfaces, joint accuracy, or training quality, and it is affected by mask quality.
The report is a recommendation: frame names, camera poses, joint GT, endpoint
models, and RGB-D order stay as recorded. Training software must consume the
recommendation to actually run backward; this step does not launch training.
