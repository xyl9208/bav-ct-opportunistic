#!/usr/bin/env python3
# Research code accompanying the BAV-CT opportunistic screening study.
# Paths are resolved through bavct.config (environment variables BAVCT_WORK
# and BAVCT_DATA). See docs/pipeline.md for the execution order.
"""b_r18n 全链评估:开发OOF配对(vs a1)→ 独立校准定阈值 → 外部测试双口径。

外部双口径:剔除后(当前主目录)与不剔除(含 剔除/ 子目录)。
输出 results/r18n_eval_report.md
"""
from bavct import config
import csv, json, math, os, sys
from pathlib import Path
import numpy as np
import torch
from scipy.stats import rankdata

B21 = Path(str(config.WORK))
B22 = Path(str(config.WORK))
B18 = Path(str(config.WORK))
EXT = Path(str(config.EXTERNAL))
from bavct.models.resnet_native import build_r18_native  # noqa
DEV = "cuda"


def auc(y, s):
    y = np.asarray(y); s = np.asarray(s, float)
    n1 = y.sum(); n0 = len(y) - n1; r = rankdata(s)
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def honest_se(y, s, z):
    thick = z >= 2.95; se = n = 0
    for m in (~thick, thick):
        neg = np.sort(s[(y == 0) & m])
        t = neg[min(len(neg) - 1, int(math.ceil(0.95 * len(neg))))]
        se += int((s[(y == 1) & m] > t).sum()); n += int(((y == 1) & m).sum())
    return se / n


def main():
    L = ["# b_r18n(原生 3D ResNet-18)全链评估", ""]
    # ---- 1. 开发 OOF 配对 vs a1 ----
    zmm = {r["exam_no"]: float(r["z_mm"]) if r["z_mm"] else 1.0
           for r in csv.DictReader(open(B21 / "data" / "gen4_manifest_folds.csv"))}
    def pool(base, arm):
        d = {}
        for k in range(1, 6):
            for r in csv.DictReader(open(f"{base}/runs/{arm}/f{k}/oof.csv")):
                d[r["exam_no"]] = (int(r["true_label"]), float(r["logit"]))
        return d
    A = pool(B21, "a1"); B = pool(B22, "b_r18n")
    common = sorted(set(A) & set(B))
    y = np.array([A[e][0] for e in common]); z = np.array([zmm[e] for e in common])
    sa = np.array([A[e][1] for e in common]); sb = np.array([B[e][1] for e in common])
    rng = np.random.default_rng(11); d = []
    for _ in range(2000):
        i = rng.integers(0, len(common), len(common))
        if y[i].sum() == 0: continue
        d.append(honest_se(y[i], sb[i], z[i]) - honest_se(y[i], sa[i], z[i]))
    L += ["## 1. 开发集配对(n=%d)" % len(common), "",
          "| 臂 | AUC | 诚实 Se@Sp95 |", "|---|---|---|",
          f"| a1(部署,膨胀卷积) | {auc(y,sa):.4f} | {100*honest_se(y,sa,z):.1f}% |",
          f"| **b_r18n(原生)** | {auc(y,sb):.4f} | {100*honest_se(y,sb,z):.1f}% |",
          f"| Δ | {auc(y,sb)-auc(y,sa):+.4f} | "
          f"{100*(honest_se(y,sb,z)-honest_se(y,sa,z)):+.1f}pp "
          f"({100*np.percentile(d,2.5):+.1f}~{100*np.percentile(d,97.5):+.1f}) |", ""]

    # ---- 2. 载入模型 ----
    models = []
    for k in range(1, 6):
        m = build_r18_native().to(DEV).eval()
        ck = torch.load(B22 / "runs" / "b_r18n" / f"f{k}" / "best.pt",
                        map_location=DEV, weights_only=True)
        m.load_state_dict(ck["model"]); models.append(m)

    @torch.no_grad()
    def prob(paths):
        out = []
        for i in range(0, len(paths), 16):
            x = torch.from_numpy(np.stack([np.load(p).astype(np.float32)
                                           for p in paths[i:i+16]])[:, None]).to(DEV)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                ps = [torch.sigmoid(m(x).float()).cpu().numpy() for m in models]
            out.append(np.mean(ps, axis=0).ravel())
        return np.concatenate(out) if out else np.array([])

    # ---- 3. 本中心独立校准集定阈值 ----
    cal = [r for r in csv.DictReader(open(B21 / "data" / "gen4_manifest_folds.csv"))
           if r["is_calib"] == "1"]
    cp = [B18 / "data/processed/c128" / r["label"] / f"{r['exam_no']}.npy" for r in cal]
    cp = [p for p in cp if p.exists()]
    cprob = prob(cp)
    cen95, cen90 = float(np.percentile(cprob, 95)), float(np.percentile(cprob, 90))
    L += [f"## 2. 本中心阈值(独立校准 n={len(cp)}):P95={cen95:.6f} P90={cen90:.6f}", ""]

    # ---- 4. 站点校准 ----
    site = {}
    for ctr in ("anji", "quzhou"):
        ps = sorted((B21 / "data/processed/calib_c128" / ctr).glob("*.npy"))
        pr = prob(ps)
        site[ctr] = {"n": len(ps), "p95": float(np.percentile(pr, 95)),
                     "p90": float(np.percentile(pr, 90))}
        L.append(f"- {ctr} 本地阈值(n={len(ps)}):P95={site[ctr]['p95']:.6f} "
                 f"P90={site[ctr]['p90']:.6f}")
    L.append("")

    # ---- 5. 外部测试双口径 ----
    def ids(p): return {f[:-7] for f in os.listdir(p) if f.endswith(".nii.gz")}
    P = B21 / "data/processed/v4_c128"
    L += ["## 3. 外部测试(双口径)", "",
          "| 中心 | 口径 | BAV n | AUC | 本地P90 Se/Sp | 本地P95 Se/Sp | 本中心P95 Se/Sp |",
          "|---|---|---|---|---|---|---|"]
    for ctr, cn, bd, td in [("anji", "安吉", "安吉BAV", "安吉TAV"),
                            ("quzhou", "衢州", "衢州BAV", "衢州TAV")]:
        tav_ids = sorted(ids(EXT / td))
        tp = [P / f"{ctr}_TAV" / f"{e}.npy" for e in tav_ids]
        tp = [p for p in tp if p.exists()]
        tprob = prob(tp)
        nr_t = len(tav_ids) - len(tp)  # 含不合格,近似;特异度按意向诊断
        keep_ids = sorted(ids(EXT / bd))
        exd = EXT / bd / "剔除"
        ex_ids = sorted(ids(exd)) if exd.is_dir() else []
        for mode, label, bl in [("keep", "剔除后", keep_ids),
                                ("full", "不剔除", keep_ids + ex_ids)]:
            bp = [P / f"{ctr}_BAV" / f"{e}.npy" for e in bl]
            bp = [p for p in bp if p.exists()]
            bprob = prob(bp)
            nb_all = len(bl)   # 意向诊断分母(缺 npy 计漏)
            row = f"| {cn} | {label} | {nb_all} | {auc([1]*len(bprob)+[0]*len(tprob), list(bprob)+list(tprob)):.4f} |"
            for thr in (site[ctr]["p90"], site[ctr]["p95"], cen95):
                se = float((bprob > thr).sum()) / nb_all
                sp = float((tprob <= thr).sum()) / (len(tprob) + nr_t)
                row += f" {100*se:.1f}%/{100*sp:.1f}% |"
            L.append(row)
    L.append("")
    (B21 / "results" / "r18n_eval_report.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
