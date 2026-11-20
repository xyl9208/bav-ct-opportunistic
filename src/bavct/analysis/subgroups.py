#!/usr/bin/env python3
# Research code accompanying the BAV-CT opportunistic screening study.
# Paths are resolved through bavct.config (environment variables BAVCT_WORK
# and BAVCT_DATA). See docs/pipeline.md for the execution order.
"""亚组分析数据准备:b_r18n 逐例分数 + 协变量总表。

覆盖:内部 temporal_BAV/TAV_A + 外部 anji/quzhou BAV/TAV(当前名册)。
协变量:年龄/性别/层厚/重建核/年份 + 升主直径(mask测径) + 根部钙化占比(npy)
      + 心超严重度标签(仅内部,ct_records)。
输出 results/subgroup_dataset.csv
"""
from bavct import config
import csv, json, os, re, sys
from pathlib import Path
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
B21=Path(str(config.WORK)); B22=Path(str(config.WORK))
B18=Path(str(config.WORK)); B7=Path(str(config.WORK))
EXT=str(config.EXTERNAL)
INT=str(config.INTERNAL)
sys.path.insert(0,str(config.SRC_DIR))

def asc_diam(margs):
    """与 missdx_aorta_diam 同法。margs=(eid, mask_dir)"""
    eid,md=margs
    try:
        import nibabel as nib
        from scipy.ndimage import binary_dilation, label
        md=Path(md)
        ao=np.asarray(nib.as_closest_canonical(nib.load(str(md/'aorta.nii.gz'))).dataobj)>0
        he=np.asarray(nib.as_closest_canonical(nib.load(str(md/'heart.nii.gz'))).dataobj)>0
        if ao.sum()<100 or he.sum()<100: return eid,None
        sp=np.abs(np.array(nib.load(str(md/'aorta.nii.gz')).header.get_zooms()[:3],float))
        ao_c=ao[::4,::4,:]; he_c=he[::4,::4,:]; spc=sp*np.array([4,4,1])
        it=max(1,int(round(10.0/spc[0])))
        hed=binary_dilation(he_c,np.ones((3,3,3),bool),iterations=max(it,min(int(round(10.0/spc[2])),5)))
        yh=np.argwhere(he_c)[:,1].mean()
        idx=np.argwhere(ao_c&hed); i2=idx[idx[:,1]>yh]
        if len(i2)>=10: idx=i2
        if len(idx)<10: return eid,None
        vz=int(np.percentile(idx[:,2],2)); yh_full=np.argwhere(he)[:,1].mean()
        best=0.0; vox=sp[0]*sp[1]
        for z in range(max(vz+int(10/sp[2]),0), min(vz+int(70/sp[2]),ao.shape[2])):
            sl=ao[:,:,z]
            if sl.sum()<20: continue
            lab,n=label(sl)
            for c in range(1,n+1):
                comp=(lab==c)
                if comp.sum()<20: continue
                if np.argwhere(comp)[:,1].mean()<=yh_full: continue
                d=2*np.sqrt(comp.sum()*vox/np.pi)
                best=max(best,d)
        return eid, round(float(best),1) if best>0 else None
    except Exception: return eid,None

def calc_frac(args):
    eid,npy=args
    try:
        v=np.load(npy).astype(np.float32)
        c=v[40:88,40:88,40:88]      # 根部中心区
        return eid, round(float((c>0.60).mean()*100),3)   # >0.6≈HU>420
    except Exception: return eid,None

def main():
    import torch
    from bavct.models.resnet_native import build_r18_native
    ms=[]
    for k in range(1,6):
        m=build_r18_native().cuda().eval()
        m.load_state_dict(torch.load(B22/f'runs/b_r18n/f{k}/best.pt',map_location='cuda',weights_only=True)['model'])
        ms.append(m)
    @torch.no_grad()
    def prob(paths):
        out=[]
        for i in range(0,len(paths),24):
            x=torch.from_numpy(np.stack([np.load(p).astype(np.float32) for p in paths[i:i+24]])[:,None]).cuda()
            with torch.autocast('cuda',dtype=torch.bfloat16):
                out.append(np.mean([torch.sigmoid(m(x).float()).cpu().numpy() for m in ms],axis=0).ravel())
        return np.concatenate(out) if out else np.array([])

    # ---- 队列定义:名册 + npy/mask 路径 ----
    def ids(d): return sorted(f[:-7] for f in os.listdir(d) if f.endswith('.nii.gz'))
    cohorts={}
    cohorts['internal_BAV']=(ids(f'{INT}/浙一BAV'), B18/'data/processed/c128/temporal_BAV', B7/'data/masks/temporal_BAV', 1)
    # 内部 TAV_A:名册取分数表(目录名不同),用 manifest
    tavA=[r['exam_no'] for r in csv.DictReader(open(B7/'data/test_manifest_temporal_TAV_A.csv'))]
    cohorts['internal_TAV']=(tavA, B18/'data/processed/c128/temporal_TAV_A', B7/'data/masks/temporal_TAV_A', 0)
    for ctr,bd,td in [('anji','安吉BAV','安吉TAV'),('quzhou','衢州BAV','衢州TAV')]:
        cohorts[f'{ctr}_BAV']=(ids(f'{EXT}/{bd}'), B21/f'data/processed/v4_c128/{ctr}_BAV', B21/'data/v4_masks', 1)
        cohorts[f'{ctr}_TAV']=(ids(f'{EXT}/{td}'), B21/f'data/processed/v4_c128/{ctr}_TAV', B21/'data/v4_masks', 0)

    # ---- 元数据:内部 manifest,外部 jsonl/分数表 ----
    man={}
    for coh,f in [('internal_BAV','test_manifest_temporal_BAV.csv'),('internal_TAV','test_manifest_temporal_TAV_A.csv')]:
        for r in csv.DictReader(open(B7/'data'/f)):
            man[r['exam_no']]={'age':r.get('age_years',''),'sex':r.get('gender',''),
                              'date':(r.get('exam_datetime','') or '')[:10],'patient_id':r.get('patient_id','')}
    for site,fs in [('anji',['anji_TAV_metadata.jsonl','anji_BAV_metadata.jsonl']),
                    ('quzhou',['quzhou_TAV_metadata.jsonl','quzhou_BAV_metadata.jsonl'])]:
        for fn in fs:
            try:
                for ln in open(f'{config.METADATA_DIR}/{fn}'):
                    try:
                        r=json.loads(ln)
                        man.setdefault(r['检查号'],{})
                        man[r['检查号']].update({'age':r.get('年龄',''),'sex':r.get('性别',''),
                          'date':(r.get('检查日期','') or '')[:10],'kernel':r.get('重建核',''),
                          'z':r.get('层厚mm','')})
                    except Exception: pass
            except FileNotFoundError: pass
    zmap={}
    for coh in ('temporal_BAV','temporal_TAV_A'):
        for r in csv.DictReader(open(B21/f'results/gen4_test_scores_{coh}.csv')):
            if r['z_mm']: zmap[r['exam_no']]=r['z_mm']
    for coh in ('anji_BAV','anji_TAV','quzhou_BAV','quzhou_TAV'):
        for r in csv.DictReader(open(B21/f'results/v4_test_scores_{coh}.csv')):
            if r['z_mm']: zmap[r['exam_no']]=r['z_mm']

    rows_out=[]
    for coh,(elist,procd,maskd,lab) in cohorts.items():
        pth=[(e, Path(procd)/f'{e}.npy') for e in elist]
        ok=[(e,p) for e,p in pth if p.exists()]
        pr=prob([p for _,p in ok]); pmap=dict(zip([e for e,_ in ok],pr))
        print(f'{coh}: scored {len(pmap)}/{len(elist)}',flush=True)
        # 钙化占比
        cal={}
        with ProcessPoolExecutor(max_workers=12) as ex:
            for f in as_completed([ex.submit(calc_frac,(e,str(p))) for e,p in ok]):
                e,v=f.result(); cal[e]=v
        # 升主直径
        dm={}
        margs=[]
        for e in elist:
            md=Path(maskd)/e if 'v4_masks' not in str(maskd) else Path(maskd)/f'{coh}__{e}'
            if (md/'aorta.nii.gz').exists(): margs.append((e,str(md)))
        with ProcessPoolExecutor(max_workers=12) as ex:
            for f in as_completed([ex.submit(asc_diam,a) for a in margs]):
                e,v=f.result(); dm[e]=v
        print(f'{coh}: diam {sum(1 for v in dm.values() if v)}/{len(elist)}',flush=True)
        for e in elist:
            m=man.get(e,{})
            rows_out.append({'cohort':coh,'exam_no':e,'label':lab,
                'prob':f"{pmap[e]:.6f}" if e in pmap else '',
                'age':m.get('age',''),'sex':m.get('sex',''),
                'z_mm':zmap.get(e, m.get('z','')),'kernel':m.get('kernel',''),
                'exam_date':m.get('date',''),'patient_id':m.get('patient_id',''),
                'asc_diam_mm':dm.get(e,''),
                'root_calc_pct':cal.get(e,'')})
    with open(B21/'results/subgroup_dataset.csv','w',newline='') as fh:
        w=csv.DictWriter(fh,fieldnames=list(rows_out[0].keys())); w.writeheader(); w.writerows(rows_out)
    print(f'SUBGROUP PREP DONE {len(rows_out)} 行',flush=True)

if __name__=='__main__': main()
