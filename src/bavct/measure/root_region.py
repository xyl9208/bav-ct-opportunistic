#!/usr/bin/env python3
# Research code accompanying the BAV-CT opportunistic screening study.
# Paths are resolved through bavct.config (environment variables BAVCT_WORK
# and BAVCT_DATA). See docs/pipeline.md for the execution order.
"""主动脉根部区域定位:
1) 粗网格上找靠近心脏、位于心脏质心前方的升主动脉,取其中段 z_ref 与 xy 中心;
2) 全分辨率从 z_ref 逐层向下追踪同一 2D 连通体(重叠 ≥20%)直到主动脉掩膜终止 → 瓣环水平 z_bot;
3) 根部段 = 追踪到的最低 30 mm 连通体,平面内外扩 4 mm;向下延伸 10 mm 且延伸段外扩 6 mm(覆盖瓣环/瓣叶),并限制在 aorta∪heart 掩膜内。
返回 (root_mask(bool, 全分辨率), info dict);失败返回 (None, info)。"""
import numpy as np
from scipy import ndimage as ndi

def root_region(ao, he, sp, L_mm=30.0, dil_mm=4.0, below_mm=10.0, below_dil_mm=6.0):
    info = {}
    ao_c, he_c = ao[::4, ::4, :], he[::4, ::4, :]; spc = sp * np.array([4, 4, 1])
    it = max(1, int(round(10.0 / spc[0])))
    hed = ndi.binary_dilation(he_c, np.ones((3, 3, 3), bool), iterations=max(it, min(int(round(10.0 / spc[2])), 5)))
    if he_c.sum() == 0: info['status'] = 'no_heart'; return None, info
    yh = np.argwhere(he_c)[:, 1].mean()
    idx = np.argwhere(ao_c & hed); i2 = idx[idx[:, 1] > yh]
    if len(i2) >= 10: idx = i2
    if len(idx) < 10: info['status'] = 'no_ascending'; return None, info
    z_ref = int(np.median(idx[:, 2])); cx, cy = idx[:, 0].mean() * 4, idx[:, 1].mean() * 4
    def comps(z):
        lab, n = ndi.label(ao[:, :, z]); return [lab == c for c in range(1, n + 1) if (lab == c).sum() >= 20]
    cands = comps(z_ref)
    if not cands: info['status'] = 'no_ref_slice'; return None, info
    cur = min(cands, key=lambda m: (np.argwhere(m).mean(0)[0] - cx) ** 2 + (np.argwhere(m).mean(0)[1] - cy) ** 2)
    track = {z_ref: cur}
    for z in range(z_ref - 1, -1, -1):
        best, bo = None, 0
        for m in comps(z):
            ov = (m & cur).sum()
            if ov > bo: best, bo = m, ov
        if best is None or bo < 0.2 * min(best.sum(), cur.sum()): break
        track[z] = best; cur = best
    z_bot = min(track); z_top = min(z_bot + int(round(L_mm / sp[2])), max(track))
    seg = np.zeros_like(ao)
    for z in range(z_bot, z_top + 1):
        if z in track: seg[:, :, z] = track[z]
    if seg.sum() < 50 or (z_top - z_bot + 1) * sp[2] < 10:
        info['status'] = 'no_root_seg'; return None, info
    # 向下延伸 below_mm(用最底层形状,外扩 below_dil_mm 覆盖瓣环/瓣叶),限制在心脏∪主动脉掩膜内
    nb = int(round(below_mm / sp[2])); z_ext0 = max(z_bot - nb, 0)
    for z in range(z_ext0, z_bot):
        seg[:, :, z] = track[z_bot]
    bb = ndi.find_objects(seg.astype(np.uint8))[0]
    pad = [int(round(below_dil_mm / s)) + 2 for s in sp]
    sl = tuple(slice(max(b.start - p_, 0), min(b.stop + p_, n_)) for b, p_, n_ in zip(bb, pad, seg.shape))
    sub = seg[sl]; n_xy = max(1, int(round(dil_mm / sp[0]))); n_xy2 = max(1, int(round(below_dil_mm / sp[0])))
    z0 = sl[2].start
    reg = np.stack([ndi.binary_dilation(sub[:, :, k], iterations=(n_xy2 if (z0 + k) < z_bot else n_xy)) if sub[:, :, k].any() else sub[:, :, k] for k in range(sub.shape[2])], axis=2)
    reg &= (ao[sl] | he[sl])
    root = np.zeros_like(ao); root[sl] = reg
    info.update(status='ok', z_ref=z_ref, z_bot=z_bot, z_top=z_top, root_len_mm=round((z_top - z_bot + 1) * sp[2], 1), root_vol_mm3=round(float(root.sum() * sp.prod()), 0))
    return root, info
