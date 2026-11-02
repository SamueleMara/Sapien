# SAPIEN external SSD (actual machine)

The external drive is mounted at `/media/samuelemara/Extreme SSD`.
The **existing output** is in `Samuele/Backup_09_10_26/Sapien/output`.
Do **not** assume inputs are in `Backup_09_09_26`: the user confirmed this folder no longer exists.

The setup script **discovers** the parent of the existing `partnet_subset_fixed` URDF directory. If more than one candidate exists, it stops and prints them rather than choosing the wrong backup.

On the Ubuntu host:

```bash
cd ~/Joint3DGS/Sapien
source ./scripts/use_external_ssd.sh
bash ./scripts/docker_enter_headless.sh
```

Alternatively, skip `source` and simply execute `bash ./scripts/docker_enter_headless.sh`; the launcher calls the setup if either mount variable is missing.

To specify an explicit real input location:

```bash
export ASSETS_DIR='/actual/absolute/path/to/parent/of/partnet_subset_fixed'
export OUTPUT_DIR='/media/samuelemara/Extreme SSD/Samuele/Backup_09_10_26/Sapien/output'
bash ./scripts/docker_enter_headless.sh
```

Inside the container, overwrite the existing 8-camera linear dataset using the **single** generic `run.py`:

```bash
cd /workspace
python scripts/run.py \
  --urdf-dir /data/assets/partnet_subset_fixed \
  --output-root /data/output/preparation_8cams/partnet_subset_fixed_render_linear \
  --num-cams 8 \
  --velocity-profile linear \
  --num-frames 60 \
  --save-video --export-colmap --colmap-image-mode copy --colmap-max-frames 1 \
  --rewrite
```

`--rewrite` deletes existing per-object outputs before rerendering, so check the resolved output path before execution. The external SSD uses exFAT, so use `--colmap-image-mode copy`, not symlinks.
