#!/usr/bin/env python3
# Research code accompanying the BAV-CT opportunistic screening study.
# Paths are resolved through bavct.config (environment variables BAVCT_WORK
# and BAVCT_DATA). See docs/pipeline.md for the execution order.
"""GPU 版 aorta/heart 精细分割,逐步复现 TotalSegmentator(fast=False, roi_subset=[aorta,heart])的完整流程:
① 6 mm 粗模型(Dataset298)→ 主动脉∪心脏包围盒 + 20 mm 边距裁剪(原始方向)② 裁剪块规范方向→1.5 mm→cardiac 模型(Dataset293)推理
③ 标签最近邻回采→撤销规范方向→回贴原始网格→保存 aorta/heart(原始方向、原始 affine)。重采样全部在 GPU 完成。
重采样坐标约定与 scipy.ndimage.zoom 一致(角点对齐:c_j = j·(n_in−1)/(n_out−1)),图像三次 B 样条(镜像边界,GPU 可分离矩阵实现),标签最近邻 floor(c+0.5)。
用法: python3 seg_gpu.py <cohort> <shard> <nshard> [--validate] [--mode cubA|linA|lin0] [--reverse]  (cohort 名取自 $BAVCT_WORK/data/cohorts.csv)"""
from bavct import config
import os, sys, csv, time, warnings
os.environ.setdefault('nnUNet_raw', '/tmp'); os.environ.setdefault('nnUNet_preprocessed', '/tmp'); os.environ.setdefault('nnUNet_results', os.path.expanduser('~/.totalsegmentator/nnunet/results'))
warnings.filterwarnings('ignore')
from pathlib import Path
import numpy as np, nibabel as nib, torch, torch.nn.functional as F
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

B21 = Path(str(config.WORK))
SRC = {'internal_BAV': str(config.COHORTS['internal_BAV']), 'anji_BAV': str(config.COHORTS['anji_BAV']), 'quzhou_BAV': str(config.COHORTS['quzhou_BAV']),
       'internal_TAV': str(config.COHORTS['internal_TAV']), 'anji_TAV': str(config.COHORTS['anji_TAV']), 'quzhou_TAV': str(config.COHORTS['quzhou_TAV'])}
MODEL = Path.home() / '.totalsegmentator/nnunet/results/Dataset293_TotalSegmentator_part3_cardiac_1559subj/nnUNetTrainerNoMirroring__nnUNetPlans__3d_fullres'
TARGET = 1.5
MODE = 'cubA'   # cubA: 三次 B 样条+角点对齐(=scipy zoom order 3);linA: 三线性+角点对齐;lin0: 旧版(torch 半像素约定)

_MCACHE = {}
def _coords(n_in, n_out):
    return torch.arange(n_out, dtype=torch.float64, device='cuda') * ((n_in - 1) / (n_out - 1) if n_out > 1 else 0.0)

def _mirror(i, n):
    i = i.clone(); p = 2 * (n - 1)
    if n == 1: return torch.zeros_like(i)
    i = torch.remainder(i, p); i = torch.where(i >= n, p - i, i); return i

def cubic_matrix(n_in, n_out):
    """(n_out × n_in) 矩阵:三次 B 样条插值(含预滤波逆矩阵),镜像边界,角点对齐坐标。"""
    key = ('cub', n_in, n_out)
    if key in _MCACHE: return _MCACHE[key]
    c = _coords(n_in, n_out); f = torch.floor(c); t = c - f
    W = torch.zeros(n_out, n_in, dtype=torch.float64, device='cuda')
    w = [(1 - t) ** 3 / 6, (3 * t ** 3 - 6 * t ** 2 + 4) / 6, (-3 * t ** 3 + 3 * t ** 2 + 3 * t + 1) / 6, t ** 3 / 6]
    for k in range(4):
        idx = _mirror((f + k - 1).long(), n_in)
        W.index_put_((torch.arange(n_out, device='cuda'), idx), w[k], accumulate=True)
    B = torch.zeros(n_in, n_in, dtype=torch.float64, device='cuda')
    for i in range(n_in):
        for d, v in ((-1, 1 / 6), (0, 2 / 3), (1, 1 / 6)):
            j = _mirror(torch.tensor([i + d], device='cuda'), n_in).item(); B[i, j] += v
    M = (W @ torch.linalg.inv(B)).float(); _MCACHE[key] = M; return M

def nearest_index(n_in, n_out):
    key = ('nn', n_in, n_out)
    if key not in _MCACHE: _MCACHE[key] = torch.floor(_coords(n_in, n_out) + 0.5).long().clamp_(0, n_in - 1)
    return _MCACHE[key]

def resample_image(t, new_shape):
    """t: (X,Y,Z) float32 cuda → new_shape。"""
    if MODE == 'lin0': return F.interpolate(t[None, None], size=new_shape, mode='trilinear', align_corners=False)[0, 0]
    if MODE == 'linA': return F.interpolate(t[None, None], size=new_shape, mode='trilinear', align_corners=True)[0, 0]
    X, Y, Z = t.shape; x, y, z = new_shape
    t = torch.einsum('ax,xyz->ayz', cubic_matrix(X, x), t)
    t = torch.einsum('by,xyz->xbz', cubic_matrix(Y, y), t)
    t = torch.einsum('cz,xyz->xyc', cubic_matrix(Z, z), t)
    return t

def resample_label(s, shp):
    if MODE == 'lin0': return F.interpolate(s[None, None].float(), size=shp, mode='nearest')[0, 0].to(torch.uint8)
    ix, iy, iz = (nearest_index(n_in, n_out) for n_in, n_out in zip(s.shape, shp))
    return s[ix][:, iy][:, :, iz]


    p = nnUNetPredictor(tile_step_size=0.5, use_gaussian=True, use_mirroring=False, perform_everything_on_device=True, device=torch.device('cuda'), verbose=False, verbose_preprocessing=False, allow_tqdm=False)
    p.initialize_from_trained_model_folder(str(MODEL), use_folds=(0,), checkpoint_name='checkpoint_final.pth')
    return p

MODEL6 = Path.home() / '.totalsegmentator/nnunet/results/Dataset298_TotalSegmentator_total_6mm_1559subj/nnUNetTrainer_4000epochs_NoMirroring__nnUNetPlans__3d_fullres'
AORTA_ID, HEART_ID = 52, 51      # class_map['total']
CROP = True
from nibabel.orientations import io_orientation, axcodes2ornt, ornt_transform, apply_orientation

def load_predictor():
    def mk(folder):
        p = nnUNetPredictor(tile_step_size=0.5, use_gaussian=True, use_mirroring=False, perform_everything_on_device=True, device=torch.device('cuda'), verbose=False, verbose_preprocessing=False, allow_tqdm=False)
        p.initialize_from_trained_model_folder(str(folder), use_folds=(0,), checkpoint_name='checkpoint_final.pth'); return p
    return {'fine': mk(MODEL), 'coarse': mk(MODEL6)}

def _predict(pred, arr_can, sp_can, target):
    """arr_can: 规范方向 (X,Y,Z) 数组;返回同网格的标签数组(uint8)。等价 change_spacing(order 3,int32)→nnUNet→change_spacing(order 0)。"""
    shp = arr_can.shape; new_shape = tuple(int(round(s * z / target)) for s, z in zip(shp, sp_can))
    with torch.no_grad():
        t = resample_image(torch.from_numpy(np.ascontiguousarray(arr_can, dtype=np.float32)).cuda(), new_shape)
        img_r = t.cpu().numpy().astype(np.int32).astype(np.float32); del t
    seg = pred.predict_single_npy_array(img_r.transpose(2, 1, 0)[None], {'spacing': (target,) * 3}, None, None, False)
    seg = np.asarray(seg).transpose(2, 1, 0)
    with torch.no_grad():
        return resample_label(torch.from_numpy(seg.astype(np.uint8)).cuda(), shp).cpu().numpy()

def _canon(arr, affine):
    """→ (规范方向数组, 规范方向 zooms, 反变换 ornt)。"""
    ornt = io_orientation(affine); arr_c = apply_orientation(arr, ornt)
    aff_c = affine @ nib.orientations.inv_ornt_aff(ornt, arr.shape)
    sp_c = np.sqrt((aff_c[:3, :3] ** 2).sum(0))
    back = ornt_transform(axcodes2ornt('RAS'), ornt)
    return arr_c, sp_c, back

def segment(preds, nii_path):
    img = nib.load(nii_path); arr = np.asarray(img.dataobj).astype(np.float32); aff = img.affine; sp = np.abs(np.array(img.header.get_zooms()[:3], float))
    if CROP:
        arr_c, sp_c, back = _canon(arr, aff)
        seg6 = _predict(preds['coarse'], arr_c, sp_c, 6.0)
        seg6 = apply_orientation(seg6, back)
        m = (seg6 == AORTA_ID) | (seg6 == HEART_ID)
        if m.sum() == 0:
            bbox = [[0, s] for s in arr.shape]
        else:
            add = (np.array([20, 20, 20]) / sp).astype(int); w = np.where(m)
            bbox = [[max(0, int(w[i].min()) - add[i]), min(arr.shape[i], int(w[i].max()) + 1 + add[i])] for i in range(3)]
        sub = arr[bbox[0][0]:bbox[0][1], bbox[1][0]:bbox[1][1], bbox[2][0]:bbox[2][1]].astype(np.int32).astype(np.float32)
    else:
        bbox = [[0, s] for s in arr.shape]; sub = arr
    sub_c, sp_c, back = _canon(sub, aff)     # 裁剪不改变方向;affine 平移不影响 ornt
    segf = apply_orientation(_predict(preds['fine'], sub_c, sp_c, TARGET), back)
    full = np.zeros(arr.shape, np.uint8); full[bbox[0][0]:bbox[0][1], bbox[1][0]:bbox[1][1], bbox[2][0]:bbox[2][1]] = segf
    return full, aff, sp

def save_masks(back, affine, dst):
    dst.mkdir(parents=True, exist_ok=True)
    for name, lab in (('heart', 1), ('aorta', 2), ('bct', 4), ('sca_l', 6)):
        m = (back == lab).astype(np.uint8)
        nib.save(nib.Nifti1Image(m, affine), dst / f'{name}.nii.gz')

def main():
    global MODE
    coh, shard, nshard = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]); validate = '--validate' in sys.argv
    global CROP
    if '--mode' in sys.argv: MODE = sys.argv[sys.argv.index('--mode') + 1]
    if '--nocrop' in sys.argv: CROP = False
    FM = Path(str(config.MASKS_FINE)) if not validate else Path(f'{config.WORK}/masks_gpu_validate_{MODE}{"_crop" if CROP else ""}')
    rows = [r for r in csv.DictReader(open(str(config.COHORT_ROSTER))) if r['cohort'] == coh]
    rows = [r for k, r in enumerate(rows) if k % nshard == shard]
    if validate: rows = rows[:20]
    if '--reverse' in sys.argv: rows = rows[::-1]   # 与正向进程同分片反向跑,中间会合
    pred = load_predictor(); t0 = time.time(); k = 0
    for r in rows:
        eid = r['exam_no']; dst = FM / f'{coh}__{eid}'; nii = f"{SRC[coh]}/{eid}.nii.gz"
        if (dst / 'aorta.nii.gz').exists() or not Path(nii).exists(): continue
        try:
            back, aff, sp = segment(pred, nii); save_masks(back, aff, dst); k += 1
            if k % 10 == 0: print(f'[{coh} s{shard}] {k}/{len(rows)} {time.time()-t0:.0f}s ({(time.time()-t0)/k:.1f} s/例)', flush=True)
        except Exception as e:
            print(f'[{coh} s{shard}] {eid} ERR {str(e)[:80]}', flush=True)
    print(f'[{coh} s{shard}] 完成 {k} 例 {time.time()-t0:.0f}s', flush=True)

if __name__ == '__main__':
    main()
