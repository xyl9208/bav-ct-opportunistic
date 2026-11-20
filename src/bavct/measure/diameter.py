#!/usr/bin/env python3
# Research code accompanying the BAV-CT opportunistic screening study.
# Paths are resolved through bavct.config (environment variables BAVCT_WORK
# and BAVCT_DATA). See docs/pipeline.md for the execution order.
"""升主动脉测径 v3:mask 骨架中心线 + 垂直截面最大过心弦。

1. 主动脉 mask 3D 骨架化(skeletonize_3d);
2. 根部锚点 = 瓣平面处升主质心;骨架图上取离锚点最近的骨架点为起点;
3. BFS 沿骨架走到全局最高点(弓顶),该路径 = 升主段(天然止于弓部);
4. 路径平滑 → 切向 → 垂直平面 0.5mm 重采样 → 过质心最大弦;
5. 报告 = 瓣上5mm 至 弓顶下10mm 路径段内的最大值。
"""
import numpy as np
from collections import deque

def skel_centerline_full(ao, root_vox, sp, iso=1.5):
    """EDT 加权最短路中心线:各向同性重采样 → 代价=1/(EDT+0.5)^2 →
    Dijkstra(root→弓顶),路径自然走管腔正中。返回原生体素坐标路径。"""
    from scipy.ndimage import zoom, distance_transform_edt
    from skimage.graph import route_through_array
    f=sp/iso
    iso_shape=tuple(max(2,int(round(s*fi))) for s,fi in zip(ao.shape,f))
    aoi=zoom(ao.astype(np.float32),[i/s for i,s in zip(iso_shape,ao.shape)],order=1)>0.5
    if aoi.sum()<50: return None
    edt=distance_transform_edt(aoi)*iso
    cost=np.where(aoi, 1.0/(edt+0.5)**2, 1e6)
    s0=tuple(int(round(root_vox[k]*sp[k]/iso)) for k in range(3))
    s0=tuple(min(max(s,0),d-1) for s,d in zip(s0,iso_shape))
    # 起点吸附:根部 z 平面上下 8mm 内、且距锚点 25mm 内的 EDT 最大点(管腔正中)
    zr=s0[2]; z1,z2=max(0,zr-2),min(iso_shape[2],zr+int(8/iso)+1)
    slab=np.zeros_like(aoi); slab[:,:,z1:z2]=aoi[:,:,z1:z2]
    cand=np.argwhere(slab)
    if len(cand):
        near=cand[np.linalg.norm((cand-np.array(s0))*1.0,axis=1)*iso<=25]
        if len(near): cand=near
        start=tuple(cand[edt[cand[:,0],cand[:,1],cand[:,2]].argmax()])
    else:
        cand=np.argwhere(aoi)
        start=tuple(cand[((cand-np.array(s0))**2).sum(1).argmin()])
    # 终点:弓顶区域(最高 4mm 内)EDT 最大点
    ztop=np.argwhere(aoi)[:,2].max()
    z3=max(0,ztop-int(4/iso))
    top=np.argwhere(aoi[:,:,z3:ztop+1])
    top=np.column_stack([top[:,0],top[:,1],top[:,2]+z3])
    end=tuple(top[edt[top[:,0],top[:,1],top[:,2]].argmax()])
    try:
        path,_=route_through_array(cost,start,end,fully_connected=True,geometric=True)
    except Exception:
        return None
    P=np.array(path,float)          # iso 体素
    return P*iso/sp[None,:]         # → 原生体素坐标

def skel_centerline(ao, root_vox, sp, iso=1.5):
    """与 skel_centerline_full 逐位等价的包围盒版:
    iso 网格与 scipy.zoom(grid_mode=False)完全相同(输出索引 o → 输入坐标 o·(n−1)/(m−1)),
    只对覆盖主动脉包围盒(外扩 1 个 iso 体素)的输出块用 map_coordinates(order=1)取值;
    块外的 zoom 结果必为 0(插值邻点全为背景),块内 EDT 与全图相同(最近背景体素必在块内),
    Dijkstra 路径不会经过块外(块外代价 1e6)。所有坐标仍用全图 iso 索引表示。"""
    from scipy.ndimage import map_coordinates, distance_transform_edt
    from skimage.graph import route_through_array
    f=sp/iso
    iso_shape=tuple(max(2,int(round(s*fi))) for s,fi in zip(ao.shape,f))
    zf=[(n-1)/(m-1) if m>1 else 1.0 for n,m in zip(ao.shape,iso_shape)]   # scipy zoom 的坐标因子
    nz=np.nonzero(ao)
    if len(nz[0])==0: return None
    lo=[]; hi=[]
    for k in range(3):
        x0=int(nz[k].min()); x1=int(nz[k].max())+1
        o0=int(np.floor((x0-1)/zf[k])) if x0>0 else 0
        o1=int(np.ceil(x1/zf[k]))
        lo.append(max(0,o0-1)); hi.append(min(iso_shape[k]-1,o1+1))
    off=np.array(lo)
    grids=np.meshgrid(*[np.arange(lo[k],hi[k]+1,dtype=np.float64)*zf[k] for k in range(3)],indexing='ij')
    coords=np.stack([g.ravel() for g in grids])
    aoi=(map_coordinates(ao.astype(np.float32),coords,order=1,mode='constant',cval=0.).reshape(grids[0].shape))>0.5
    if aoi.sum()<50: return None
    edt=distance_transform_edt(aoi)*iso
    cost=np.where(aoi, 1.0/(edt+0.5)**2, 1e6)
    s0=tuple(int(round(root_vox[k]*sp[k]/iso)) for k in range(3))
    s0=tuple(min(max(s,0),d-1) for s,d in zip(s0,iso_shape))
    zr=s0[2]; z1,z2=max(0,zr-2),min(iso_shape[2],zr+int(8/iso)+1)
    b1,b2=max(z1,lo[2])-lo[2],max(min(z2,hi[2]+1)-lo[2],0)
    slab=np.zeros_like(aoi)
    if b2>b1: slab[:,:,b1:b2]=aoi[:,:,b1:b2]
    cand=np.argwhere(slab)+off
    if len(cand):
        near=cand[np.linalg.norm((cand-np.array(s0))*1.0,axis=1)*iso<=25]
        if len(near): cand=near
        cl=cand-off
        start=tuple(cand[edt[cl[:,0],cl[:,1],cl[:,2]].argmax()])
    else:
        cand=np.argwhere(aoi)+off
        start=tuple(cand[((cand-np.array(s0))**2).sum(1).argmin()])
    ztop=int(np.argwhere(aoi)[:,2].max())+lo[2]
    z3=max(0,ztop-int(4/iso))
    top=np.argwhere(aoi[:,:,z3-lo[2]:ztop+1-lo[2]])
    top=np.column_stack([top[:,0]+lo[0],top[:,1]+lo[1],top[:,2]+z3])
    tl_=top-off
    end=tuple(top[edt[tl_[:,0],tl_[:,1],tl_[:,2]].argmax()])
    try:
        path,_=route_through_array(cost,tuple(np.array(start)-off),tuple(np.array(end)-off),fully_connected=True,geometric=True)
    except Exception:
        return None
    P=np.array(path,float)+off        # 全图 iso 体素
    return P*iso/sp[None,:]           # → 原生体素坐标

def skel_diam(ao, he, sp, sinus_from=10.0, arch_margin=20.0):
    from scipy.ndimage import binary_dilation, gaussian_filter1d, map_coordinates, label as lab3
    if ao.sum()<100 or he.sum()<100: return None,'empty'
    # 根部锚点(同部署定位)
    ao_c=ao[::4,::4,:]; he_c=he[::4,::4,:]; spc=sp*np.array([4,4,1])
    it=max(1,int(round(10.0/spc[0])))
    hed=binary_dilation(he_c,np.ones((3,3,3),bool),iterations=max(it,min(int(round(10.0/spc[2])),5)))
    yh=np.argwhere(he_c)[:,1].mean()
    idx=np.argwhere(ao_c&hed); i2=idx[idx[:,1]>yh]
    if len(i2)>=10: idx=i2
    if len(idx)<10: return None,'no root'
    vz=float(np.percentile(idx[:,2],2))
    sel=idx[(idx[:,2]>=vz)&(idx[:,2]<=vz+20.0/spc[2])]
    if len(sel)==0: sel=idx
    root=np.array([sel[:,0].mean()*4, sel[:,1].mean()*4, vz])
    path=skel_centerline(ao, root, sp)
    if path is None or len(path)<8: return None,'skel fail'
    Pmm=path*sp[None,:]
    # 弧长均匀重采样(2mm)后再平滑,切向稳定
    seg=np.linalg.norm(np.diff(Pmm,axis=0),axis=1)
    s=np.concatenate([[0],np.cumsum(seg)])
    if s[-1]<20: return None,'path too short'
    tl=min(4.0,0.05*s[-1]); th_=min(5.0,0.06*s[-1])
    ss=np.arange(tl,max(s[-1]-th_,tl+2.0),2.0)    # 两端修剪(短路径自适应)
    Pmm=np.stack([np.interp(ss,s,Pmm[:,k]) for k in range(3)],axis=1)
    for k in range(3): Pmm[:,k]=gaussian_filter1d(Pmm[:,k],3)   # σ=6mm
    # 切向 = 局部 PCA 主方向(±8mm 窗,4个采样点),比逐点差分稳
    T=np.zeros_like(Pmm)
    for i in range(len(Pmm)):
        lo_=max(0,i-4); hi_=min(len(Pmm),i+5)
        seg_=Pmm[lo_:hi_]-Pmm[lo_:hi_].mean(axis=0)
        _,_,Vt=np.linalg.svd(seg_,full_matrices=False)
        d=Vt[0]
        if d[2]<0: d=-d
        T[i]=d
    T/=np.linalg.norm(T,axis=1,keepdims=True)+1e-9
    z_top=Pmm[:,2].max(); z_valve=vz*sp[2]
    g=np.arange(-42,42.5,0.5); GU,GV=np.meshgrid(g,g,indexing='ij')
    best=0.0
    aof=ao.astype(np.float32)   # 提到循环外(原每个平面重复转换整幅掩膜;结果逐位相同)
    for i in range(len(Pmm)):
        zmm=Pmm[i][2]
        span=max(z_top-z_valve,1.0)
        am=min(arch_margin,0.25*span); sf=min(sinus_from,0.2*span)
        if zmm < z_valve+sf or zmm > z_top-am: continue
        t=T[i]
        u=np.cross(t,[0,0,1.0]); nu=np.linalg.norm(u)
        u=(u/nu) if nu>1e-6 else np.array([1.0,0,0])
        v=np.cross(t,u)
        plane=Pmm[i][None,None,:]+GU[...,None]*u[None,None,:]+GV[...,None]*v[None,None,:]
        vox=(plane/sp[None,None,:]).reshape(-1,3).T
        m2=(map_coordinates(aof,vox,order=1,mode='constant',cval=0.).reshape(GU.shape))>=0.5
        if not m2[len(g)//2,len(g)//2]: continue
        LL,_=lab3(m2)
        comp2=LL==LL[len(g)//2,len(g)//2]
        p2=np.argwhere(comp2).astype(float)*0.5-42
        c2=p2.mean(axis=0); rel=p2-c2
        eq=2*np.sqrt(comp2.sum()*0.25/np.pi)
        from scipy.ndimage import distance_transform_edt as _edt
        r_in=_edt(comp2)[len(g)//2,len(g)//2]*0.5    # 心点内切半径(mm)
        if r_in<3: continue
        for a in np.arange(0,180,3):
            th=np.deg2rad(a); uu=np.array([np.cos(th),np.sin(th)])
            tp=rel@uu; pp=np.abs(rel@np.array([-uu[1],uu[0]]))
            m=pp<=0.6
            if m.sum()<2: continue
            L=tp[m].max()-tp[m].min()
            if L>1.6*eq: continue          # 椭圆度守卫:斜切伪影
            if L>3.0*r_in: continue        # 内切圆守卫:双管腔合并/根部粘连
            if L>best:
                best=L
                info=(Pmm[i].copy(),t.copy(),u.copy(),v.copy(),c2.copy(),th,tp[m].min(),tp[m].max())
    if best<=0: return None,'no plane'
    return (round(float(best),1), info, Pmm), 'ok'
