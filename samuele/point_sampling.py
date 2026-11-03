"""Deterministic sampling of valid RGB-D pixels; never change depth or RGB images."""
import argparse
import math

import numpy as np


def point_fraction(value):
    value = float(value)
    if not math.isfinite(value) or not 0 < value <= 1:
        raise argparse.ArgumentTypeError('Point fraction must be finite and in (0, 1]')
    return value


def sample_pixels(ys, xs, fraction):
    fraction = point_fraction(fraction)
    if len(xs) == 0 or fraction == 1:
        return ys, xs
    count = max(1, math.floor(len(xs) * fraction))
    # Evenly cover the valid pixel list instead of taking a contiguous patch.
    indices = np.linspace(0, len(xs)-1, count, dtype=np.int64)
    return ys[indices], xs[indices]
