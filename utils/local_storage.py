"""Keep local-batch working files and library caches under its output target."""
import argparse
import errno
import os
import shlex
import shutil
import tempfile
from pathlib import Path

DEFAULT_OUTPUTS_ROOT = Path("/media/samuelemara/Extreme SSD/Samuele/results_11_10_26/freeart")
MIN_FREE_BYTES = 10 * 1024**3


def runtime_environment(outputs_root):
    runtime = Path(outputs_root).resolve() / ".runtime"
    directories = {
        "TMPDIR": "tmp", "TEMP": "tmp", "TMP": "tmp",
        "XDG_CACHE_HOME": "cache", "XDG_CONFIG_HOME": "config", "XDG_DATA_HOME": "data",
        "MPLCONFIGDIR": "matplotlib", "TORCH_HOME": "torch",
        "TORCH_EXTENSIONS_DIR": "torch_extensions", "TORCHINDUCTOR_CACHE_DIR": "torchinductor",
        "HF_HOME": "huggingface", "HF_HUB_CACHE": "huggingface/hub",
        "HUGGINGFACE_HUB_CACHE": "huggingface/hub", "TRANSFORMERS_CACHE": "huggingface/transformers",
        "CUDA_CACHE_PATH": "cuda", "TRITON_CACHE_DIR": "triton", "NUMBA_CACHE_DIR": "numba",
    }
    env = {key: str(runtime / folder) for key, folder in directories.items()}
    env.update(PYTHONDONTWRITEBYTECODE="1", MPLBACKEND="Agg")
    return env


def ensure_free_space(outputs_root):
    free = shutil.disk_usage(outputs_root).free
    if free < MIN_FREE_BYTES:
        raise OSError(errno.ENOSPC, f"Only {free / 1024**3:.2f} GiB free at {outputs_root}; "
                      f"need at least {MIN_FREE_BYTES / 1024**3:.0f} GiB for the batch working data")


def configure_runtime(outputs_root, dry_run=False):
    root = Path(outputs_root).resolve()
    env = runtime_environment(root)
    directories = {value for key, value in env.items() if key not in ("PYTHONDONTWRITEBYTECODE", "MPLBACKEND")}
    for directory in directories:
        path = Path(directory)
        if not path.resolve().is_relative_to(root):
            raise ValueError(f"Runtime directory escapes the output target: {path}")
    if not dry_run:
        root.mkdir(parents=True, exist_ok=True)
        ensure_free_space(root)
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    os.environ.update(env)
    tempfile.tempdir = env["TMPDIR"]
    return env


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("--outputs-root", type=Path, default=DEFAULT_OUTPUTS_ROOT)
    parser.add_argument("--dry-run", action="store_true")
    args, remaining = parser.parse_known_args()
    env = configure_runtime(args.outputs_root.expanduser(), args.dry_run or "--help" in remaining or "-h" in remaining)
    for key, value in env.items():
        print(f"export {key}={shlex.quote(value)}")
