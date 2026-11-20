#!/usr/bin/env python3
# Research code accompanying the BAV-CT opportunistic screening study.
# Paths are resolved through bavct.config (environment variables BAVCT_WORK
# and BAVCT_DATA). See docs/pipeline.md for the execution order.
"""Discrimination, calibration and subgroup statistics for the deployed model:

1. AUC + DeLong 95%CI(每中心);2. Se/Sp + Wilson CI(P95/P90);
3. PPV/NPV/LR± @患病率 0.5/1/2%;4. 决策曲线(净获益);
5. 校准(logistic 斜率/截距,内部);6. 三中心 Se 异质性(Cochran Q);
7. 亚组 Se@P95 + 交互 p(森林图数据)。
输出 results/stats_report.md + forest_data.csv
"""

from bavct import config
import re
import csv, json, math
from pathlib import Path
import numpy as np
from scipy import stats
B21=Path(str(config.WORK))

def delong_ci(y,s):
    y=np.asarray(y); s=np.asarray(s,float)
    pos=s[y==1]; neg=s[y==0]; m,n=len(pos),len(neg)
    mw=(pos[:,None]>neg[None,:]).astype(float)+0.5*(pos[:,None]==neg[None,:])
    auc=mw.mean()
    v10=mw.mean(axis=1); v01=mw.mean(axis=0)
    var=v10.var(ddof=1)/m+v01.var(ddof=1)/n
    se=math.sqrt(max(var,1e-12))
    return auc, auc-1.96*se, auc+1.96*se
def wilson(k,n):
    if n==0: return 0,0,0
    p=k/n; z=1.96
    den=1+z*z/n; c=(p+z*z/(2*n))/den
    h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return p, c-h, c+h

def main():
    T=json.load(open(B21/'results/r18n_thresholds.json'))
    rows=list(csv.DictReader(open(B21/'results/subgroup_dataset.csv')))
    for r in rows: r['p']=float(r['prob']) if r['prob'] else None
    L=["# Discrimination and calibration statistics (deployed model)",""]
    coh_map={'内部时间外':('internal_BAV','internal_TAV',T['central']),
             '安吉':('anji_BAV','anji_TAV',T['site']['anji']),
             '衢州':('quzhou_BAV','quzhou_TAV',T['site']['quzhou'])}
    se_by={}
    L.append("## 1. 判别性能\n")
    L.append("| 队列 | AUC (DeLong 95%CI) | 运行点 | Se (Wilson CI) | Sp (Wilson CI) | LR+ | LR− |")
    L.append("|---|---|---|---|---|---|---|")
    for name,(bc,tc,thr) in coh_map.items():
        b=[r['p'] for r in rows if r['cohort']==bc and r['p'] is not None]
        t=[r['p'] for r in rows if r['cohort']==tc and r['p'] is not None]
        nb=sum(1 for r in rows if r['cohort']==bc); nt=sum(1 for r in rows if r['cohort']==tc)
        y=[1]*len(b)+[0]*len(t)
        auc,lo,hi=delong_ci(y,b+t)
        for tag in ('p95','p90'):
            th=thr[tag]
            k_se=sum(1 for p in b if p>th); k_sp=sum(1 for p in t if p<=th)
            se,selo,sehi=wilson(k_se,nb); sp,splo,sphi=wilson(k_sp,nt)
            lrp=se/max(1-sp,1e-9); lrn=(1-se)/max(sp,1e-9)
            se_by.setdefault(tag,{})[name]=(k_se,nb)
            aucs=f"{auc:.4f} ({lo:.4f}–{hi:.4f})" if tag=='p95' else ''
            L.append(f"| {name} | {aucs} | {tag.upper()} | {100*se:.1f}% ({100*selo:.1f}–{100*sehi:.1f}) | "
                     f"{100*sp:.1f}% ({100*splo:.1f}–{100*sphi:.1f}) | {lrp:.1f} | {lrn:.2f} |")
    # 2. PPV/NPV
    L.append("\n## 2. 预测值(内部 P95 运行点,按患病率情景)\n")
    b=[r['p'] for r in rows if r['cohort']=='internal_BAV' and r['p'] is not None]
    t=[r['p'] for r in rows if r['cohort']=='internal_TAV' and r['p'] is not None]
    th=T['central']['p95']
    se=sum(1 for p in b if p>th)/len(b); sp=sum(1 for p in t if p<=th)/len(t)
    L.append("| 患病率 | PPV | NPV | NNR(筛出1例需复核数) |"); L.append("|---|---|---|---|")
    for pr in (0.005,0.01,0.02):
        ppv=se*pr/(se*pr+(1-sp)*(1-pr)); npv=sp*(1-pr)/(sp*(1-pr)+(1-se)*pr)
        L.append(f"| {100*pr:.1f}% | {100*ppv:.1f}% | {100*npv:.2f}% | {1/ppv:.1f} |")
    # 3. 决策曲线
    L.append("\n## 3. 决策曲线(内部,患病率 1% 情景,净获益 ×1000)\n")
    L.append("| 阈值概率 | AI 净获益 | 全转诊 | 不转诊 |"); L.append("|---|---|---|---|")
    prev=0.01
    for pt in (0.02,0.05,0.10,0.20):
        w=pt/(1-pt)
        nbn=se*prev - (1-sp)*(1-prev)*w
        alln=prev-(1-prev)*w
        L.append(f"| {pt:.2f} | {1000*nbn:.2f} | {1000*alln:.2f} | 0 |")
    # 4. 校准
    L.append("\n## 4. 校准(内部,logistic 再校准)\n")
    y=np.array([1]*len(b)+[0]*len(t)); s=np.clip(np.array(b+t),1e-6,1-1e-6)
    lo_=np.log(s/(1-s))
    from numpy.polynomial import polynomial as P
    # 简易 logistic 回归(牛顿法)
    X=np.column_stack([np.ones_like(lo_),lo_]); beta=np.zeros(2)
    for _ in range(50):
        mu=1/(1+np.exp(-X@beta)); Wt=mu*(1-mu)
        beta+=np.linalg.solve((X*Wt[:,None]).T@X+1e-6*np.eye(2), X.T@(y-mu))
    L.append(f"- 校准斜率 {beta[1]:.3f}(理想=1),截距 {beta[0]:.3f}(理想=0;受测试集患病率 {y.mean():.1%} 影响,部署需按目标患病率重标定)")
    brier=float(np.mean((s-y)**2)); L.append(f"- Brier 分数 {brier:.4f}")
    # 5. 异质性
    L.append("\n## 5. 三中心敏感度异质性(P95)\n")
    ks=[v for v in se_by['p95'].values()]
    p_pool=sum(k for k,_ in ks)/sum(n for _,n in ks)
    Q=sum((k-n*p_pool)**2/max(n*p_pool*(1-p_pool),1e-9) for k,n in ks)
    pq=1-stats.chi2.cdf(Q,df=len(ks)-1)
    L.append(f"- Cochran Q={Q:.2f}, df=2, p={pq:.3f}" + ("(存在异质性)" if pq<0.05 else "(无显著异质性)"))
    # 6. 亚组森林数据(合并三中心 BAV,Se@各自P95;TAV 侧 Sp 同理)
    L.append("\n## 6. 亚组敏感度(P95,三中心合并,阈值各用本地)\n")
    fr=[]
    def th_of(coh):
        return {'internal_BAV':T['central']['p95'],'anji_BAV':T['site']['anji']['p95'],
                'quzhou_BAV':T['site']['quzhou']['p95']}[coh]
    bav=[r for r in rows if r['label']=='1' and r['p'] is not None]
    for r in bav: r['hit']=1 if r['p']>th_of(r['cohort']) else 0
    def sub(name, fn):
        groups={}
        for r in bav:
            g=fn(r)
            if g: groups.setdefault(g,[]).append(r['hit'])
        out=[]
        for g,v in sorted(groups.items()):
            se,lo_,hi_=wilson(sum(v),len(v))
            out.append((g,len(v),se,lo_,hi_))
            fr.append({'subgroup':name,'level':g,'n':len(v),'se':round(100*se,1),
                       'lo':round(100*lo_,1),'hi':round(100*hi_,1)})
        return out
    def band(z):
        try: z=float(z)
        except: return ''
        return '<1.5mm' if z<1.5 else ('1.5-3mm' if z<2.95 else '≥3mm')
    def ageg(a):
        try: a=float(re.sub(r'[^0-9.]','',str(a)))   # 外部中心 DICOM 年龄如 '083Y'
        except: return ''
        if a<18: return ''   # 作者核对:<18 岁(DICOM 字段异常)不计入年龄亚组分母
        return '<40' if a<40 else ('40-59' if a<60 else '≥60')
    def diamg(d):
        try: d=float(d)
        except: return ''
        return '<40mm' if d<40 else ('40-45mm' if d<45 else '≥45mm')
    _cf=B21/'results/calc_subgroups.csv'   # 钙化 v2(算法定量;疑似增强/高噪声例无分组→不进亚组)
    CAT={r['exam_no']:r for r in csv.DictReader(open(_cf))} if _cf.exists() else {}
    def calcg(c):
        try: return '有钙化' if float(c)>0.5 else '无/少钙化'
        except: return ''
    subs=[('层厚带',lambda r: band(r['z_mm'])),('年龄',lambda r: ageg(r['age'])),
          ('性别',lambda r: r['sex'] or ''),('升主直径',lambda r: diamg(r['asc_diam_mm'])),
          ('主动脉根部钙化斑块容积',lambda r: CAT.get(r['exam_no'],{}).get('root_cat',''))]
    L.append("| 亚组 | 层级 | n | Se | 95%CI |"); L.append("|---|---|---|---|---|")
    for nm,fn in subs:
        for g,n,se,lo_,hi_ in sub(nm,fn):
            L.append(f"| {nm} | {g} | {n} | {100*se:.1f}% | {100*lo_:.1f}–{100*hi_:.1f} |")
    with open(B21/'results/forest_data.csv','w',newline='') as fh:
        w=csv.DictWriter(fh,fieldnames=['subgroup','level','n','se','lo','hi']); w.writeheader(); w.writerows(fr)
    (B21/'results/stats_report.md').write_text("\n".join(L)+"\n")
    print("\n".join(L))

if __name__=='__main__': main()
