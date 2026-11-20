#!/usr/bin/env python3
# Research code accompanying the BAV-CT opportunistic screening study.
# Paths are resolved through bavct.config (environment variables BAVCT_WORK
# and BAVCT_DATA). See docs/pipeline.md for the execution order.
"""数据集与患者均衡采样(方案 §5.3)。

- 批内 BAV:TAV = 1:1。
- BAV 患者均衡:按患者洗牌循环,患者内随机选一次检查。
- TAV 从大池按患者循环抽取(每患者一名额,患者内随机选检查),跨 epoch 覆盖全池。
- 验证/OOF 用原始队列构成(全部检查,不重采样)。
- 增强:小幅旋转/平移/缩放(单次 affine)、强度扰动、高斯噪声;禁止翻转。
"""
from bavct import config
import csv
import math
import random
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d
from torch.utils.data import Dataset

BASE = Path(str(config.WORK))
DATA = BASE / "data"
MANIFEST = config.MANIFEST_CSV

ROI_DIRS = {"composite": DATA / "processed" / "composite",
            "valve": DATA / "processed" / "valve"}


def load_manifest():
    rows = list(csv.DictReader(open(MANIFEST)))
    for r in rows:
        r["fold"] = int(r["fold"])
        r["y"] = 1 if r["label"] == "BAV" else 0
    return rows


def has_npy(row, roi):
    return (ROI_DIRS[roi] / row["label"] / f"{row['exam_no']}.npy").exists()


class ExamDataset(Dataset):
    """检查级数据集(验证/推理用,无增强)。"""

    def __init__(self, rows, roi="composite", augment=False, seed=0):
        self.rows = rows
        self.roi = roi
        self.augment = augment
        self.seed = seed

    def __len__(self):
        return len(self.rows)

    def _load(self, row):
        p = ROI_DIRS[self.roi] / row["label"] / f"{row['exam_no']}.npy"
        v = np.load(p).astype(np.float32)
        return v

    def _zaug(self, v, rng):
        """z向采集层厚随机化(v2):消除层厚-标签捷径。

        每样本 T~U(1,5)mm:先加 sqrt(T²-1)/2.355 mm 的 z 向高斯模糊
        (近似从 ~1mm 原生层厚降到 T),再按 T 降采样-插回模拟部分容积。
        """
        T = float(rng.uniform(1.0, 5.0))
        vox = (100.0 / 96) if self.roi == "composite" else (56.0 / 96)
        sigma_mm = math.sqrt(max(T * T - 1.0, 0.0)) / 2.355
        if sigma_mm > 0.05:
            v = gaussian_filter1d(v, sigma_mm / vox, axis=2, mode="nearest")
        if T > 1.5 * vox:
            z = v.shape[2]
            zt = max(8, int(round(z * vox / T)))
            # 仅沿 z 的 1D 线性重采样(等价于 trilinear,但快约3倍)
            t = torch.from_numpy(np.ascontiguousarray(v)).reshape(-1, 1, z)
            t = torch.nn.functional.interpolate(t, size=zt, mode="linear",
                                                align_corners=False)
            t = torch.nn.functional.interpolate(t, size=z, mode="linear",
                                                align_corners=False)
            v = t.reshape(v.shape[0], v.shape[1], z).numpy()
        return v

    def _augment(self, v, rng):
        v = self._zaug(v, rng)
        t = torch.from_numpy(v)[None, None]  # 1,1,D,H,W
        # 组合仿射: 旋转(每轴±10°) + 缩放0.9~1.1 + 平移±6voxel
        angles = rng.uniform(-10, 10, 3) * math.pi / 180
        scale = rng.uniform(0.9, 1.1)
        trans = rng.uniform(-6 / 48, 6 / 48, 3)  # 归一化坐标
        cx, cy, cz = np.cos(angles)
        sx, sy, sz = np.sin(angles)
        Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
        Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
        Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
        A = (Rx @ Ry @ Rz) / scale
        theta = torch.zeros(1, 3, 4)
        theta[0, :, :3] = torch.from_numpy(A).float()
        theta[0, :, 3] = torch.from_numpy(trans).float()
        grid = torch.nn.functional.affine_grid(theta, t.shape, align_corners=False)
        t = torch.nn.functional.grid_sample(t, grid, mode="bilinear",
                                            padding_mode="zeros", align_corners=False)
        v = t[0, 0].numpy()
        # 强度扰动
        if rng.random() < 0.8:
            v = v * rng.uniform(0.9, 1.1) + rng.uniform(-0.05, 0.05)
        if rng.random() < 0.5:
            v = np.clip(v, 0, None) ** rng.uniform(0.85, 1.18)
        if rng.random() < 0.5:
            v = v + rng.normal(0, rng.uniform(0.005, 0.02), v.shape).astype(np.float32)
        return np.clip(v, -0.2, 1.2).astype(np.float32)

    def __getitem__(self, i):
        row = self.rows[i]
        v = self._load(row)
        if self.augment:
            v = self._augment(v, np.random.default_rng((self.seed, i)))
        return torch.from_numpy(v[None].copy()), row["y"], i


class BalancedPatientDataset(Dataset):
    """训练用:索引 i 只决定 类别+顺序,实际样本由患者循环器给出。

    epoch 长度 = 2 × TAV训练患者数;偶数索引取 BAV、奇数取 TAV。
    """

    def __init__(self, rows, roi="composite", seed=0):
        self.roi = roi
        self.seed = seed
        self.epoch = 0
        self.by_pat = {0: {}, 1: {}}
        for r in rows:
            self.by_pat[r["y"]].setdefault(r["patient_id"], []).append(r)
        self.pats = {y: sorted(self.by_pat[y]) for y in (0, 1)}
        self.n = 2 * len(self.pats[0])
        self.inner = ExamDataset(rows, roi=roi, augment=True, seed=seed)

    def set_epoch(self, e):
        self.epoch = e

    def __len__(self):
        return self.n

    def _pick(self, y, k):
        pats = self.pats[y]
        cycle, pos = divmod(k, len(pats))
        rng = random.Random((self.seed, y, cycle))
        order = list(pats)
        rng.shuffle(order)
        pid = order[pos]
        exams = self.by_pat[y][pid]
        rng2 = random.Random((self.seed, y, cycle, pid))
        return rng2.choice(exams)

    def __getitem__(self, i):
        base = self.epoch * self.n
        y = 1 if i % 2 == 0 else 0
        k = (base + i) // 2
        row = self._pick(y, k)
        v = self.inner._load(row)
        v = self.inner._augment(v, np.random.default_rng((self.seed, self.epoch, i)))
        return torch.from_numpy(v[None].copy()), y, i
