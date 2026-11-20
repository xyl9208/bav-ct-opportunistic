#!/usr/bin/env python3
# Research code accompanying the BAV-CT opportunistic screening study.
# Paths are resolved through bavct.config (environment variables BAVCT_WORK
# and BAVCT_DATA). See docs/pipeline.md for the execution order.
"""Gen-5 主干:SE-R18、ConvNeXt-V2-tiny-3D(含 GRN)、编码器加载工具。"""
import sys

import torch
import torch.nn as nn

from bavct.models.resnet3d import build_model  # noqa: E402  (R18 部署同款)


# ---------------- SE-R18 ----------------
class SE3d(nn.Module):
    def __init__(self, ch, r=16):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(ch, max(ch // r, 4)), nn.ReLU(),
                                nn.Linear(max(ch // r, 4), ch), nn.Sigmoid())

    def forward(self, x):
        w = self.fc(x.mean(dim=(2, 3, 4)))
        return x * w[:, :, None, None, None]


class SEWrap(nn.Module):
    """包住一个残差 block:block 输出后接 SE 重标定。"""

    def __init__(self, block, ch):
        super().__init__()
        self.block = block
        self.se = SE3d(ch)

    def forward(self, x):
        return self.se(self.block(x))


def build_se_r18(input_shape=(128, 128, 128)):
    m = build_model(False, input_shape=input_shape)
    for lname, ch in [("layer1", 64), ("layer2", 128),
                      ("layer3", 256), ("layer4", 512)]:
        layer = getattr(m, lname)
        for i, blk in enumerate(layer):
            layer[i] = SEWrap(blk, ch)
    return m


# ---------------- ConvNeXt-V2-3D ----------------
class GRN3d(nn.Module):
    """Global Response Normalization(ConvNeXt-V2),通道最后布局。"""

    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, 1, dim))

    def forward(self, x):  # x: (N, D, H, W, C)
        gx = torch.norm(x, p=2, dim=(1, 2, 3), keepdim=True)
        nx = gx / (gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * nx) + self.beta + x


class CNXBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dw = nn.Conv3d(dim, dim, 7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pw1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.grn = GRN3d(4 * dim)
        self.pw2 = nn.Linear(4 * dim, dim)

    def forward(self, x):  # (N,C,D,H,W)
        r = x
        x = self.dw(x).permute(0, 2, 3, 4, 1)  # →(N,D,H,W,C)
        x = self.pw2(self.grn(self.act(self.pw1(self.norm(x)))))
        return r + x.permute(0, 4, 1, 2, 3)


class LayerNormCF(nn.Module):
    """channels-first LayerNorm。"""

    def __init__(self, dim):
        super().__init__()
        self.ln = nn.LayerNorm(dim, eps=1e-6)

    def forward(self, x):
        return self.ln(x.permute(0, 2, 3, 4, 1)).permute(0, 4, 1, 2, 3)


class ConvNeXtV2_3D(nn.Module):
    def __init__(self, in_ch=1, depths=(2, 2, 6, 2), dims=(64, 128, 256, 512),
                 num_out=1):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv3d(in_ch, dims[0], 4, stride=4),
                                  LayerNormCF(dims[0]))
        self.downs = nn.ModuleList()
        self.stages = nn.ModuleList()
        for i in range(4):
            self.stages.append(nn.Sequential(
                *[CNXBlock(dims[i]) for _ in range(depths[i])]))
            if i < 3:
                self.downs.append(nn.Sequential(
                    LayerNormCF(dims[i]),
                    nn.Conv3d(dims[i], dims[i + 1], 2, stride=2)))
        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(dims[-1], num_out)

    def features(self, x):
        x = self.stem(x)
        for i in range(4):
            x = self.stages[i](x)
            if i < 3:
                x = self.downs[i](x)
        return x  # (N,512,4,4,4) @128³

    def forward(self, x):
        x = self.features(x)
        x = x.mean(dim=(2, 3, 4))
        return self.fc(self.dropout(self.norm(x))).squeeze(1)


def build_cnx(input_shape=(128, 128, 128)):
    return ConvNeXtV2_3D()


# ---------------- 编码器加载 ----------------
def load_encoder(model, enc_path, prefix_map=None):
    """把预训练编码器权重灌入分类模型(仅匹配的键,fc 保持随机)。"""
    enc = torch.load(enc_path, map_location="cpu", weights_only=True)
    state = enc.get("encoder", enc)
    msd = model.state_dict()
    hit = {}
    for k, v in state.items():
        k2 = (prefix_map or {}).get(k, k)
        if k2 in msd and msd[k2].shape == v.shape:
            hit[k2] = v
    msd.update(hit)
    model.load_state_dict(msd)
    print(f"encoder loaded: {len(hit)}/{len(state)} tensors")
    return model


if __name__ == "__main__":
    x = torch.randn(1, 1, 128, 128, 128)
    for name, m in [("se_r18", build_se_r18()), ("cnx", build_cnx())]:
        n = sum(p.numel() for p in m.parameters()) / 1e6
        with torch.no_grad():
            y = m(x)
        print(name, f"params={n:.1f}M out={tuple(y.shape) or 'scalar'}")


# ---------------- 原生 3D ResNet-18(stride 下采样,无膨胀)----------------
class _BasicBlock(nn.Module):
    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.c1 = nn.Conv3d(cin, cout, 3, stride, 1, bias=False)
        self.b1 = nn.BatchNorm3d(cout)
        self.c2 = nn.Conv3d(cout, cout, 3, 1, 1, bias=False)
        self.b2 = nn.BatchNorm3d(cout)
        self.relu = nn.ReLU(inplace=True)
        self.down = (nn.Sequential(nn.Conv3d(cin, cout, 1, stride, bias=False),
                                   nn.BatchNorm3d(cout))
                     if (stride != 1 or cin != cout) else None)

    def forward(self, x):
        r = x if self.down is None else self.down(x)
        x = self.relu(self.b1(self.c1(x)))
        return self.relu(self.b2(self.c2(x)) + r)


class R18Native(nn.Module):
    """标准 3D ResNet-18(Hara 风格),128³ → 4³ 特征图,较部署版快约 40×。"""

    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv3d(1, 64, 7, 2, 3, bias=False),
                                  nn.BatchNorm3d(64), nn.ReLU(True),
                                  nn.MaxPool3d(3, 2, 1))
        L = lambda a, b, s: nn.Sequential(_BasicBlock(a, b, s),
                                          _BasicBlock(b, b))
        self.l1, self.l2 = L(64, 64, 1), L(64, 128, 2)
        self.l3, self.l4 = L(128, 256, 2), L(256, 512, 2)
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(512, 1)

    def features(self, x):
        return self.l4(self.l3(self.l2(self.l1(self.stem(x)))))

    def forward(self, x):
        h = self.pool(self.features(x)).flatten(1)
        return self.fc(self.dropout(h)).squeeze(1)


def build_r18_native(input_shape=(128, 128, 128)):
    return R18Native()
