# Generic SAPIEN run interface

The **only** batch entry point is `python scripts/run.py`.

`--num-cams N`: 8 = PARIS, 64 = hemisphere, 100 = hemisphere. Other counts generate circular cameras. Existing/custom `--camera-config-json` must contain N cameras.

`--velocity-profile PROFILE`: linear, ease-in, ease-out, ease-in-out, trapezoid. Case and underscore-insensitive.

`--rewrite`: fully rerender each URDF, deleting the *existing per-model* output before regenerating RGB/depth/mask/video and initial/final COLMAP + PLY. Without it, existing objects are skipped. `--overwrite-existing` remains a deprecated alias.

`--output-root`: optional explicit target; otherwise the default is `/data/output/preparation_<N>cams/partnet_subset_fixed_render_<PROFILE>` in Docker. `--urdf-dir` is the source directory and must be chosen to include exactly the source models you intend to process.

For the existing 8-camera linear dataset (inside Docker, assuming source URDFs are in the path from this repo's PartNet instructions):

```bash
python scripts/run.py \
  --urdf-dir /data/assets/partnet_subset_fixed \
  --output-root /data/output/preparation_8cams/partnet_subset_fixed_render_linear \
  --num-cams 8 \
  --velocity-profile linear \
  --num-frames 60 \
  --save-video \
  --export-colmap \
  --colmap-image-mode copy \
  --colmap-max-frames 1 \
  --rewrite
```

The `colmap/` and `colmap_final/` outputs are independently reconstructed from the first and last motion frame, and `pointcloud/` contains their labeled PLYs. `--dry-run` previews without deletion. See `EXTERNAL_SSD.md` for mount details.
