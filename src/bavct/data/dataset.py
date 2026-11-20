#!/usr/bin/env python3
# Research code accompanying the BAV-CT opportunistic screening study.
# Paths are resolved through bavct.config (environment variables BAVCT_WORK
# and BAVCT_DATA). See docs/pipeline.md for the execution order.
"""Gen-4 数据集:多 ROI 源解析 + 层厚配额批内采样。

ROI 源:
  z3c128   : 8.18 Z3 复合 128³(A0-A4)
  native128: 原生分辨率复合 128³(A5;8.21 补做的新BAV → 8.14 composite128
             {label} → 8.14 CALIB)
  anat     : 解剖自适应盒 1.5mm 120×128×152(A6;8.21 产物)
层厚配额采样(A1/A2/A4/A5):3 桶(<1.5 / 1.5-2.95 / ≥2.95mm),每个样本对
按 BAV 桶分布抽桶,再在桶内 BAV/TAV 各自患者循环 → batch 内两类层厚构成一致。
"""
from bavct import config
import csv
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch

B21 = Path(str(config.WORK))
B18 = Path(str(config.WORK))
B14 = config.WORK
from bavct.data import legacy_dataset as ds7  # noqa: E402

MANIFEST = config.MANIFEST_CSV
ANAT_SHAPE = (120, 128, 152)
ANAT_VOX = 1.5


def load_manifest_gen4():
    rows = list(csv.DictReader(open(MANIFEST)))
    for r in rows:
        r["fold"] = int(r["fold"]) if r["fold"] else 0
        r["is_calib"] = r["is_calib"] == "1"
        r["y"] = 1 if r["label"] == "BAV" else 0
        r["z_mm"] = float(r["z_mm"]) if r["z_mm"] else 1.0
        r["t_eff"] = float(np.clip(max(r["z_mm"], 3.0), 3.0, 5.0))
        z = r["z_mm"]
        r["bucket"] = 0 if z < 1.5 else (1 if z < 2.95 else 2)
    return rows


def roi_path(roi, row):
    e, lab = row["exam_no"], row["label"]
    if roi == "z3c128":
        return B18 / "data" / "processed" / "c128" / lab / f"{e}.npy"
    if roi == "anat":
        return B21 / "data" / "processed" / "anat_roi" / lab / f"{e}.npy"
    if roi == "native128":
        for p in [B21 / "data" / "processed" / "native_c128" / lab / f"{e}.npy",
                  B14 / "data" / "processed" / "composite128" / lab / f"{e}.npy",
                  B14 / "data" / "processed" / "composite128" / "CALIB" / f"{e}.npy"]:
            if p.exists():
                return p
        return p
    raise ValueError(roi)


def has_roi(roi, row):
    return roi_path(roi, row).exists()


class Gen4ExamDataset(ds7.ExamDataset):
    """zaug_mode: residual(T~U(t_eff,5)) / off。体素尺寸按 ROI 源取。"""

    def __init__(self, rows, roi, augment=False, seed=0, zaug_mode="residual"):
        super().__init__(rows, roi="composite", augment=augment, seed=seed)
        self.roi_src = roi
        self.zaug_mode = zaug_mode
        self._cur_teff = 3.0

    def _load(self, row):
        return np.load(roi_path(self.roi_src, row)).astype(np.float32)

    def _zaug(self, v, rng):
        if self.zaug_mode == "off":
            return v
        t0 = self._cur_teff
        T = float(rng.uniform(t0, 5.0))
        vox = ANAT_VOX if self.roi_src == "anat" else 100.0 / 128
        sigma_mm = math.sqrt(max(T * T - t0 * t0, 0.0)) / 2.355
        if sigma_mm > 0.05:
            from scipy.ndimage import gaussian_filter1d
            v = gaussian_filter1d(v, sigma_mm / vox, axis=2, mode="nearest")
            z = v.shape[2]
            zt = max(8, int(round(z * vox / T)))
            if zt < z:
                t = torch.from_numpy(np.ascontiguousarray(v)).reshape(-1, 1, z)
                t = torch.nn.functional.interpolate(t, size=zt, mode="linear",
                                                    align_corners=False)
                t = torch.nn.functional.interpolate(t, size=z, mode="linear",
                                                    align_corners=False)
                v = t.reshape(v.shape[0], v.shape[1], z).numpy()
        return v


class Gen4Balanced(ds7.BalancedPatientDataset):
    """1:1 患者均衡(无配额)。"""

    def __init__(self, rows, roi, seed=0, zaug_mode="residual"):
        super().__init__(rows, roi="composite", seed=seed)
        self.inner = Gen4ExamDataset(rows, roi, augment=True, seed=seed,
                                     zaug_mode=zaug_mode)

    def __getitem__(self, i):
        base = self.epoch * self.n
        y = 1 if i % 2 == 0 else 0
        k = (base + i) // 2
        row = self._pick(y, k)
        self.inner._cur_teff = row["t_eff"]
        v = self.inner._load(row)
        v = self.inner._augment(v, np.random.default_rng((self.seed,
                                                          self.epoch, i)))
        return torch.from_numpy(v[None].copy()), y, i


class Gen4QuotaBalanced(torch.utils.data.Dataset):
    """层厚配额:样本对的桶按 BAV 桶分布循环抽取,桶内患者循环。

    epoch 长度 = 2 × TAV 训练患者数(与既有口径一致)。
    """

    def __init__(self, rows, roi, seed=0, zaug_mode="residual"):
        self.seed = seed
        self.epoch = 0
        self.inner = Gen4ExamDataset(rows, roi, augment=True, seed=seed,
                                     zaug_mode=zaug_mode)
        self.by = {}  # (y, bucket) -> {pid: [rows]}
        for r in rows:
            self.by.setdefault((r["y"], r["bucket"]), {}) \
                .setdefault(r["patient_id"], []).append(r)
        self.pats = {k: sorted(v) for k, v in self.by.items()}
        # BAV 桶分布(按患者数)
        bav_counts = [len(self.pats.get((1, b), [])) for b in range(3)]
        tot = sum(bav_counts)
        n_tav = sum(len(self.pats.get((0, b), [])) for b in range(3))
        self.n = 2 * n_tav
        # 桶序列:按比例的确定性循环(Bresenham 式),保证每个 batch 邻域构成稳定
        self.bucket_seq = []
        acc = [0.0] * 3
        for _ in range(self.n // 2):
            for b in range(3):
                acc[b] += bav_counts[b] / tot
            b = max(range(3), key=lambda x: acc[x])
            acc[b] -= 1.0
            self.bucket_seq.append(b)

    def set_epoch(self, e):
        self.epoch = e

    def __len__(self):
        return self.n

    def _pick(self, y, bucket, k):
        pats = self.pats[(y, bucket)]
        cycle, pos = divmod(k, len(pats))
        rng = random.Random((self.seed, y, bucket, cycle))
        order = list(pats)
        rng.shuffle(order)
        pid = order[pos]
        exams = self.by[(y, bucket)][pid]
        rng2 = random.Random((self.seed, y, bucket, cycle, pid))
        return rng2.choice(exams)

    def __getitem__(self, i):
        base = self.epoch * self.n
        y = 1 if i % 2 == 0 else 0
        k = (base + i) // 2
        bucket = self.bucket_seq[k % len(self.bucket_seq)]
        # 桶内序号:同桶第几次出现(跨 epoch 连续)
        kk = k // len(self.bucket_seq) * self.bucket_seq.count(bucket) + \
            sum(1 for b in self.bucket_seq[:k % len(self.bucket_seq)]
                if b == bucket)
        row = self._pick(y, bucket, kk)
        self.inner._cur_teff = row["t_eff"]
        v = self.inner._load(row)
        v = self.inner._augment(v, np.random.default_rng((self.seed,
                                                          self.epoch, i)))
        return torch.from_numpy(v[None].copy()), y, i
