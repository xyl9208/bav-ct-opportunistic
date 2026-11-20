#!/usr/bin/env python3
# Research code accompanying the BAV-CT opportunistic screening study.
# Paths are resolved through bavct.config (environment variables BAVCT_WORK
# and BAVCT_DATA). See docs/pipeline.md for the execution order.
"""Gen-4 训练:7 臂 × 5 折(配方主体 = Gen-3 c128,差异见 ARM 表)。

用法: train_gen4.py --arm a0..a6 --fold K [--workers N] [--resume]
输出 runs/<arm>/f<K>/{best.pt,last.pt,oof.csv,train_log.csv}
数据就绪门:等待 data/ready_<roisrc>.marker(A5/A6 预处理完成后由预处理脚本写)。
"""
from bavct import config
import argparse
import csv
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

B21 = Path(str(config.WORK))
from bavct.data.dataset import (ANAT_SHAPE, Gen4Balanced, Gen4ExamDataset,  # noqa
                          Gen4QuotaBalanced, has_roi, load_manifest_gen4)
from bavct.models.resnet3d import build_model  # noqa: E402

SEED0 = 20260821
ARMS = {
    #        roi         zaug        quota  epochs early_stop
    "a0": ("z3c128",    "residual", False, 40, True),
    "a1": ("z3c128",    "residual", True,  40, True),
    "a2": ("z3c128",    "off",      True,  40, True),
    "a3": ("z3c128",    "residual", False, 24, False),
    "a4": ("z3c128",    "residual", True,  24, False),
    "a5": ("native128", "off",      True,  40, True),
    "a6": ("anat",      "residual", False, 40, True),
}
WARMUP = 2
PATIENCE = 10
MICRO, ACCUM = 8, 3


def auc_score(y, s):
    from scipy.stats import rankdata
    y = np.asarray(y)
    s = np.asarray(s, float)
    n1 = int(y.sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(s)
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def sens_at_spec(y, s, spec=0.95):
    y = np.asarray(y)
    s = np.asarray(s, float)
    neg = np.sort(s[y == 0])
    if len(neg) == 0:
        return float("nan")
    thr = neg[min(len(neg) - 1, int(math.ceil(spec * len(neg))))]
    return float((s[y == 1] > thr).mean())


@torch.no_grad()
def collect(model, dl, device):
    model.eval()
    ys, lgs = [], []
    for x, y, _ in dl:
        x = x.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            z = model(x).float()
        lgs.extend(z.cpu().double().tolist())
        ys.extend(y.tolist())
    return ys, lgs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=sorted(ARMS))
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    roi, zaug, quota, EPOCHS, early = ARMS[args.arm]
    K = args.fold
    seed = SEED0 + K
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = True
    device = "cuda"

    # 数据就绪门(a5/a6 预处理由并行流生产)
    marker = B21 / "data" / f"ready_{roi}.marker"
    if roi != "z3c128":
        t0 = time.time()
        while not marker.exists():
            if time.time() - t0 > 6 * 3600:
                print(f"FAIL: data not ready {marker}", flush=True)
                sys.exit(3)
            time.sleep(60)

    out = B21 / "runs" / args.arm / f"f{K}"
    out.mkdir(parents=True, exist_ok=True)
    if (out / "oof.csv").exists():
        print("already done, skip")
        return

    rows = [r for r in load_manifest_gen4()
            if not r["is_calib"] and has_roi(roi, r)]
    tr = [r for r in rows if r["fold"] != K]
    va = [r for r in rows if r["fold"] == K]
    print(f"{args.arm} f{K} roi={roi} quota={quota} ep={EPOCHS} "
          f"train={len(tr)} val={len(va)}", flush=True)

    if quota:
        train_ds = Gen4QuotaBalanced(tr, roi, seed=seed, zaug_mode=zaug)
    else:
        train_ds = Gen4Balanced(tr, roi, seed=seed, zaug_mode=zaug)
    val_ds = Gen4ExamDataset(va, roi, augment=False)
    train_dl = DataLoader(train_ds, batch_size=MICRO, shuffle=False,
                          num_workers=args.workers, pin_memory=True,
                          drop_last=True, persistent_workers=False)
    val_dl = DataLoader(val_ds, batch_size=MICRO, shuffle=False,
                        num_workers=4, pin_memory=True)

    shape = ANAT_SHAPE if roi == "anat" else (128, 128, 128)
    model = build_model(False, input_shape=shape).to(device)
    model = model.to(memory_format=torch.channels_last_3d)
    crit = nn.BCEWithLogitsLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    steps_per_epoch = len(train_ds) // (MICRO * ACCUM)
    total_steps = EPOCHS * steps_per_epoch
    warm_steps = WARMUP * steps_per_epoch

    def lr_factor(step):
        if step < warm_steps:
            return step / max(1, warm_steps)
        p = (step - warm_steps) / max(1, total_steps - warm_steps)
        return 0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * p))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_factor)
    log_f = open(out / "train_log.csv", "a", newline="")
    lw = csv.writer(log_f)
    if log_f.tell() == 0:
        lw.writerow(["stage", "epoch", "train_loss", "val_auc", "val_sens95",
                     "sec"])
    best = (-1.0, -1.0)
    best_ep = -1
    step = 0
    start_ep = 0
    if args.resume and (out / "last.pt").exists():
        ck = torch.load(out / "last.pt", map_location=device,
                        weights_only=True)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        start_ep = ck["epoch"] + 1
        step = ck["step"]
        best = tuple(ck["best"])
        best_ep = ck["best_ep"]
        print(f"resumed ep{ck['epoch']}", flush=True)

    for ep in range(start_ep, EPOCHS):
        train_ds.set_epoch(ep + 100)
        model.train()
        t0 = time.time()
        tot, nb, mb = 0.0, 0, 0
        opt.zero_grad(set_to_none=True)
        for x, y, _ in train_dl:
            x = x.to(device, non_blocking=True)
            y = y.float().to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = crit(model(x).float(), y) / ACCUM
            loss.backward()
            mb += 1
            if mb % ACCUM == 0:
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                nb += 1
            tot += loss.item() * ACCUM
        ys, lgs = collect(model, val_dl, device)
        s = [1 / (1 + math.exp(-z)) for z in lgs]
        auc, s95 = auc_score(ys, s), sens_at_spec(ys, s)
        lw.writerow(["S2", ep, round(tot / max(mb, 1), 5), round(auc, 5),
                     round(s95, 5), int(time.time() - t0)])
        log_f.flush()
        print(f"ep{ep} loss={tot/max(mb,1):.4f} AUC={auc:.4f} "
              f"s95={s95:.3f} {int(time.time()-t0)}s", flush=True)
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "sched": sched.state_dict(), "epoch": ep, "step": step,
                    "best": best, "best_ep": best_ep}, out / "last.pt")
        key = (round(auc, 6), round(s95, 6))
        if key > best:
            best = key
            best_ep = ep
            torch.save({"model": model.state_dict(), "epoch": ep,
                        "val_auc": auc}, out / "best.pt")
        if early and ep - best_ep >= PATIENCE and ep >= WARMUP + 5:
            print(f"early stop ep{ep} (best {best_ep})", flush=True)
            break
    log_f.close()

    ck = torch.load(out / "best.pt", map_location=device, weights_only=True)
    model.load_state_dict(ck["model"])
    ys, lgs = collect(model, val_dl, device)
    with open(out / "oof.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["exam_no", "patient_id", "true_label", "fold",
                    "logit", "score"])
        for r, z in zip(va, lgs):
            w.writerow([r["exam_no"], r["patient_id"], r["y"], K,
                        f"{z:.8f}", f"{1/(1+np.exp(-z)):.10f}"])
    s = [1 / (1 + math.exp(-z)) for z in lgs]
    print(f"FOLD DONE {args.arm} f{K} best_ep={ck['epoch']} "
          f"OOF_AUC={auc_score(ys, s):.4f} s95={sens_at_spec(ys, s):.3f}",
          flush=True)


if __name__ == "__main__":
    main()
