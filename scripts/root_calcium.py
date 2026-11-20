#!/usr/bin/env python3
# Research code accompanying the BAV-CT opportunistic screening study.
# Paths are resolved through bavct.config (environment variables BAVCT_WORK
# and BAVCT_DATA). See docs/pipeline.md for the execution order.
"""主动脉根部钙化斑块容积 v3:根部区域定位(root_region.py)+ 区域内钙化体积。
用法: python3 calc_root_v3.py test|tav|missdx [workers](tav = 测试集 TAV 三队列,掩膜同 masks_fine/<cohort>__<exam>);输出 results/calc_root_v3_<set>.csv,QC 图抽样到 results/calc_root_v3_qc_<set>/。"""
from bavct import config
import csv, sys, time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import numpy as np, nibabel as nib
from scipy import ndimage as ndi
from bavct.measure.root_region import root_region
SET = sys.argv[1]; NW = int(sys.argv[2]) if len(sys.argv) > 2 else 6
R = Path(str(config.RESULTS)); OUT = R / f'calc_root_v3_{SET}.csv'; QC = R / f'calc_root_v3_qc_{SET}'; QC.mkdir(exist_ok=True)
if SET in ('test', 'tav'):
    SRC = {'internal_BAV': str(config.COHORTS['internal_BAV']), 'anji_BAV': str(config.COHORTS['anji_BAV']), 'quzhou_BAV': str(config.COHORTS['quzhou_BAV'])}
    if SET == 'tav': SRC = {'internal_TAV': str(config.COHORTS['internal_TAV']), 'anji_TAV': str(config.COHORTS['anji_TAV']), 'quzhou_TAV': str(config.COHORTS['quzhou_TAV'])}
    FM = Path(str(config.MASKS_FINE)); tasks = [(d.name.split('__', 1)[0], d.name.split('__', 1)[1]) for d in sorted(FM.iterdir()) if d.name.split('__', 1)[0] in SRC and (d / 'aorta.nii.gz').exists()]
    ctpath = lambda coh, eid: f'{SRC[coh]}/{eid}.nii.gz'; mdir = lambda coh, eid: FM / f'{coh}__{eid}'
else:
    W = Path(str(config.MISS_DX)); FM = W / 'masks_fine'; tasks = [('missdx', d.name) for d in sorted(FM.iterdir()) if (d / 'aorta.nii.gz').exists() and (W / 'nii' / f'{d.name}.nii.gz').exists()]
    ctpath = lambda coh, eid: str(W / 'nii' / f'{eid}.nii.gz'); mdir = lambda coh, eid: FM / eid
COLS = ['exam_no', 'cohort', 'status', 'aorta_median_hu', 'aorta_noise_madsd', 'enhanced_flag', 'root_status', 'root_len_mm', 'root_region_vol_mm3', 'root_ca_vol_mm3', 'root_ca_n_comp', 'root_ca_max_hu', 'sp_z_mm']
def clean(ca):
    lab, n = ndi.label(ca)
    if n: sizes = ndi.sum(ca, lab, range(1, n + 1)); ca = ca & np.isin(lab, [i + 1 for i, s in enumerate(sizes) if s >= 3]); lab, n = ndi.label(ca)
    return ca, n
def one(args):
    coh, eid, qc = args
    r = dict.fromkeys(COLS, ''); r.update(exam_no=eid, cohort=coh)
    try:
        img = nib.as_closest_canonical(nib.load(ctpath(coh, eid))); ct = np.asarray(img.dataobj)   # 保持 int16,省内存(OOM 后改)
        md = mdir(coh, eid)
        ao = np.asarray(nib.as_closest_canonical(nib.load(str(md / 'aorta.nii.gz'))).dataobj) > 0
        he = np.asarray(nib.as_closest_canonical(nib.load(str(md / 'heart.nii.gz'))).dataobj) > 0
        if ao.shape != ct.shape: r['status'] = 'shape_mismatch'; return r
        sp = np.abs(np.array(img.header.get_zooms()[:3], float))
        shift = -1024.0 if np.percentile(ct[::8, ::8, ::4], 1) > -500 else 0.0
        if ao.sum() < 100 or he.sum() < 100: r['status'] = 'mask_small'; return r
        lab, n = ndi.label(he)
        if n > 1: sizes = ndi.sum(he, lab, range(1, n + 1)); he = lab == (int(np.argmax(sizes)) + 1)
        lab, n = ndi.label(ao)
        if n > 1:
            sizes = ndi.sum(ao, lab, range(1, n + 1)) * sp.prod(); keep = [i + 1 for i, s in enumerate(sizes) if s >= 5000]; ao = np.isin(lab, keep) if keep else lab == (int(np.argmax(sizes)) + 1)
        v = ct[ao].astype(np.float32) + shift; amed = float(np.median(v)); madsd = float(1.4826 * np.median(np.abs(v - amed))); del v
        r.update(aorta_median_hu=round(amed, 0), aorta_noise_madsd=round(madsd, 0), enhanced_flag=int(amed > 100), sp_z_mm=round(float(sp[2]), 2))
        root, info = root_region(ao, he, sp); r['root_status'] = info.get('status')
        if root is None: r['status'] = 'no_root'; return r
        bb = ndi.find_objects(root.astype(np.uint8))[0]; pad = [int(round(3.0 / s)) + 1 for s in sp]
        sl = tuple(slice(max(b.start - p_, 0), min(b.stop + p_, n_)) for b, p_, n_ in zip(bb, pad, ct.shape))
        ctb, rb = ct[sl].astype(np.float32) + shift, root[sl]; cts = ndi.gaussian_filter(ctb, sigma=1.0 / sp)
        ca, n = clean(rb & (cts >= 130))
        r.update(status='ok', root_len_mm=info['root_len_mm'], root_region_vol_mm3=info['root_vol_mm3'], root_ca_vol_mm3=round(float(ca.sum() * sp.prod()), 1), root_ca_n_comp=int(n), root_ca_max_hu=round(float(ctb[ca].max()), 0) if n else 0)
        if qc:
            import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
            zr = np.where(rb.any(axis=(0, 1)))[0]; k = int(np.percentile(zr, 40))
            fig, ax = plt.subplots(1, 2, figsize=(9, 4.5))
            for a_ in ax: a_.imshow(np.clip(ctb[:, :, k].T, -150, 800), cmap='gray', vmin=-150, vmax=800, origin='lower'); a_.axis('off')
            ax[0].contour(rb[:, :, k].T, levels=[0.5], colors='magenta', linewidths=1.0)
            m = ca[:, :, k].T; ax[1].imshow(np.ma.masked_where(~m, m), cmap='autumn', origin='lower', alpha=0.95)
            fig.suptitle(f"{coh} {eid} root {info['root_len_mm']} mm  Ca {r['root_ca_vol_mm3']} mm3  noise {madsd:.0f}", fontsize=9); fig.tight_layout(); fig.savefig(QC / f'{coh}__{eid}.png', dpi=100); plt.close(fig)
        return r
    except Exception as e:
        r['status'] = f'err:{str(e)[:60]}'; return r
def main():
    done = {row['exam_no'] for row in csv.DictReader(open(OUT))} if OUT.exists() else set()
    todo = [(c, e, abs(hash(e)) % 50 == 0) for c, e in tasks if e not in done]
    print(f'{SET}: 待算 {len(todo)}(已完成 {len(done)})', flush=True); new = not OUT.exists()
    with open(OUT, 'a', newline='') as fh, ProcessPoolExecutor(NW) as ex:
        w = csv.DictWriter(fh, fieldnames=COLS)
        if new: w.writeheader()
        t0 = time.time(); k = 0
        for f in as_completed([ex.submit(one, t) for t in todo]):
            w.writerow(f.result()); fh.flush(); k += 1
            if k % 200 == 0: print(f'{k}/{len(todo)} {time.time()-t0:.0f}s', flush=True)
    print('完成', flush=True)
if __name__ == '__main__': main()
