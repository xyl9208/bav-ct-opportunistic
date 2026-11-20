"""Valve-centred composite ROI and valve ROI.

1. CT and masks are moved to a canonical orientation.
2. The CT is windowed to [-150, 800] HU and mapped to [0, 1]; the z direction is
   unified to a 3-mm effective resolution (see bavct.preprocess.z3).
3. Root localisation on a 4x downsampled grid: candidates are
   aorta and dilate(heart, 10 mm) and anterior to the heart centroid; the valve
   plane is the 2nd percentile of their z coordinate, and the root centre is the
   centroid of candidates within 20 mm above it.
4. Composite ROI: 100-mm cube centred at (root_x, root_y, valve_z + 30 mm).
   Valve ROI: 56-mm cube centred at (root_x, root_y, valve_z + 8 mm).
   Both are resampled to 128^3 (isotropic 0.78 mm / 0.44 mm) as float16.
5. Quality control: mean of the central 64^3 block and the fraction of zero
   voxels; a case is flagged suspect_localization below 0.10 or above 0.5.

The manifest has one row per examination with the columns
exam_no, label, ct, aorta, heart (paths to NIfTI files).
"""
import argparse
import csv
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_dilation, zoom

from bavct import config
from bavct.preprocess.z3 import z3_unify

HU_LO, HU_HI = -150.0, 800.0
DS_XY = 4
COMPOSITE_MM = (100.0, 100.0, 100.0)
VALVE_MM = (56.0, 56.0, 56.0)
OUT_SHAPE = 128


def crop_physical_box(volume, spacing, center_vox, size_mm, pad_value):
    """Crop a physical box of the given size around center_vox.

    Returns (box, inside_fraction), or None if the box misses the volume.
    """
    shape = np.array(volume.shape)
    half = np.array([size_mm[i] / 2.0 / spacing[i] for i in range(3)])
    start = np.round(np.array(center_vox) - half).astype(int)
    end = np.round(np.array(center_vox) + half).astype(int)
    out = np.full(end - start, pad_value, dtype=np.float32)
    lo = np.maximum(start, 0)
    hi = np.minimum(end, shape)
    if np.any(hi <= lo):
        return None
    dst = lo - start
    out[dst[0]:dst[0] + hi[0] - lo[0],
        dst[1]:dst[1] + hi[1] - lo[1],
        dst[2]:dst[2] + hi[2] - lo[2]] = volume[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
    inside = float(np.prod(hi - lo) / np.prod(out.shape))
    return out, inside


def locate_root(aorta, heart, spacing):
    """Return (valve_z, root_x, root_y) in voxel coordinates."""
    ao_c = aorta[::DS_XY, ::DS_XY, :]
    he_c = heart[::DS_XY, ::DS_XY, :]
    sp_c = spacing * np.array([DS_XY, DS_XY, 1])
    it_xy = max(1, int(round(10.0 / sp_c[0])))
    it_z = max(1, int(round(10.0 / sp_c[2])))
    he_d = binary_dilation(he_c, structure=np.ones((3, 3, 3), bool),
                           iterations=max(it_xy, min(it_z, 5)))
    y_heart = np.argwhere(he_c)[:, 1].mean()
    cand = ao_c & he_d
    idx = np.argwhere(cand)
    idx = idx[idx[:, 1] > y_heart]
    if len(idx) < 10:
        idx = np.argwhere(cand)
        if len(idx) < 10:
            raise ValueError("no root candidate")
    valve_z = float(np.percentile(idx[:, 2], 2))
    near = idx[(idx[:, 2] >= valve_z) & (idx[:, 2] <= valve_z + 20.0 / sp_c[2])]
    if len(near) == 0:
        near = idx
    return valve_z, float(near[:, 0].mean() * DS_XY), float(near[:, 1].mean() * DS_XY)


def extract(ct, aorta, heart, spacing, out_shape=OUT_SHAPE):
    """Crop and resample both ROIs; returns (arrays, info)."""
    ct = np.clip(ct.astype(np.float32), HU_LO, HU_HI)
    ct = (ct - HU_LO) / (HU_HI - HU_LO)
    ct = z3_unify(ct, float(spacing[2]))

    valve_z, root_x, root_y = locate_root(aorta, heart, spacing)
    info = {"valve_z_vox": valve_z, "root_x": root_x, "root_y": root_y,
            "z_mm": round(float(spacing[2]), 2)}
    arrays = {}
    for name, size_mm, z_offset in (("c128", COMPOSITE_MM, 30.0),
                                    ("v128", VALVE_MM, 8.0)):
        centre = (root_x, root_y, valve_z + z_offset / spacing[2])
        box = crop_physical_box(ct, spacing, centre, size_mm, 0.0)
        if box is None:
            raise ValueError(name + " box outside volume")
        crop, inside = box
        out = zoom(crop, [out_shape / s for s in crop.shape], order=1)
        if out.shape != (out_shape,) * 3:
            pad = np.zeros((out_shape,) * 3, np.float32)
            s = [min(out_shape, d) for d in out.shape]
            pad[:s[0], :s[1], :s[2]] = out[:s[0], :s[1], :s[2]]
            out = pad
        arrays[name] = out.astype(np.float16)
        info[name + "_inside"] = round(inside, 3)

    centre = arrays["c128"][32:96, 32:96, 32:96]
    info["centre_mean"] = round(float(centre.mean()), 4)
    info["zero_frac"] = round(float((arrays["c128"] <= 1e-6).mean()), 4)
    info["status"] = ("suspect_localization"
                      if info["centre_mean"] < 0.10 or info["zero_frac"] > 0.5
                      else "ok")
    return arrays, info


def process_case(row, out_root):
    import nibabel as nib

    eid, label = row["exam_no"], row["label"]
    ct_img = nib.as_closest_canonical(nib.load(row["ct"]))
    ao_img = nib.as_closest_canonical(nib.load(row["aorta"]))
    he_img = nib.as_closest_canonical(nib.load(row["heart"]))
    if ao_img.shape != ct_img.shape:
        raise ValueError("mask shape mismatch")
    spacing = np.abs(np.array(ct_img.header.get_zooms()[:3], dtype=float))
    arrays, info = extract(np.asarray(ct_img.dataobj),
                           np.asarray(ao_img.dataobj) > 0,
                           np.asarray(he_img.dataobj) > 0, spacing)
    for name, arr in arrays.items():
        dst = out_root / name / label / (eid + ".npy")
        dst.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(dst), arr)
    return eid, info


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", default=str(config.WORK / "data" / "processed"))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.manifest, newline="", encoding="utf-8-sig")))
    if args.limit:
        rows = rows[:args.limit]
    out_root = Path(args.out)
    qc_path = config.RESULTS / "preprocess_qc.csv"
    qc_path.parent.mkdir(parents=True, exist_ok=True)

    with open(qc_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["exam_no", "status", "z_mm", "valve_z_vox", "root_x",
                         "root_y", "c128_inside", "v128_inside", "centre_mean",
                         "zero_frac"])
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(process_case, r, out_root): r["exam_no"] for r in rows}
            for future in as_completed(futures):
                eid = futures[future]
                try:
                    _, info = future.result()
                except Exception as exc:
                    print(eid + ": " + str(exc))
                    writer.writerow([eid, "error: " + str(exc).splitlines()[0]] + [""] * 8)
                    continue
                writer.writerow([eid, info["status"], info["z_mm"], info["valve_z_vox"],
                                 info["root_x"], info["root_y"], info["c128_inside"],
                                 info["v128_inside"], info["centre_mean"], info["zero_frac"]])
    print("done: " + str(len(rows)) + " examinations -> " + str(out_root))


if __name__ == "__main__":
    main()
