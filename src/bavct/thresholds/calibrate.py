#!/usr/bin/env python3
"""Threshold calibration from a score table.

Methods: the comparison cut-off of a cohort is the 95th (and 90th) percentile
of the score distribution of an independent set of non-BAV examinations from
the same site. Percentiles use NumPy's default (linear) interpolation.

The score table has one row per examination::

    exam_no,site,prob
    A001,anji,0.0123

Usage::

    python -m bavct.thresholds.calibrate --scores work/results/calib_scores.csv         --out work/results/site_thresholds.json --group-col site
"""
import argparse
import csv
import json

import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scores", required=True, help="CSV with one probability per row")
    ap.add_argument("--out", required=True, help="output JSON")
    ap.add_argument("--score-col", default="prob")
    ap.add_argument("--group-col", default=None,
                    help="optional column with the calibration set name")
    ap.add_argument("--percentiles", type=float, nargs="+", default=[95.0, 90.0])
    args = ap.parse_args()

    groups = {}
    with open(args.scores, newline="", encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            raw = (row.get(args.score_col) or "").strip()
            if not raw:
                continue
            key = row.get(args.group_col, "all") if args.group_col else "all"
            groups.setdefault(key, []).append(float(raw))

    out = {}
    for key, values in sorted(groups.items()):
        arr = np.asarray(values, dtype=float)
        entry = {"n": int(arr.size)}
        for p in args.percentiles:
            entry[f"p{p:g}"] = float(np.percentile(arr, p))
        out[key] = entry

    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    for key, entry in out.items():
        print(key, entry)


if __name__ == "__main__":
    main()
