"""Unification of the through-plane effective resolution (Z3).

Acquisitions thicker than the target are left untouched. Thinner acquisitions
are blurred in z with a Gaussian kernel sized so that the combined nominal
resolution equals the target, then downsampled to the target grid and linearly
interpolated back to the original number of slices. This removes the
slice-thickness cue without changing the geometry of the masks.
"""
import math

import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d

Z_TARGET_MM = 3.0


def z3_unify(volume, slice_thickness_mm, target_mm=Z_TARGET_MM):
    """Return the volume with a 3-mm effective through-plane resolution.

    volume : np.ndarray, normalised, z is the last axis.
    slice_thickness_mm : reconstructed slice thickness of the source scan.
    """
    if slice_thickness_mm >= target_mm - 0.05:
        return volume
    sigma_mm = math.sqrt(max(target_mm ** 2 - slice_thickness_mm ** 2, 0.0)) / 2.355
    if sigma_mm > 0.05:
        volume = gaussian_filter1d(
            volume, sigma_mm / slice_thickness_mm, axis=2, mode="nearest")
    z = volume.shape[2]
    z_target = max(8, int(round(z * slice_thickness_mm / target_mm)))
    if z_target < z:
        w = torch.from_numpy(np.ascontiguousarray(volume)).reshape(-1, 1, z)
        w = torch.nn.functional.interpolate(
            w, size=z_target, mode="linear", align_corners=False)
        w = torch.nn.functional.interpolate(
            w, size=z, mode="linear", align_corners=False)
        volume = w.reshape(volume.shape).numpy()
    return volume
