#!/usr/bin/env python3
# Research code accompanying the BAV-CT opportunistic screening study.
# Paths are resolved through bavct.config (environment variables BAVCT_WORK
# and BAVCT_DATA). See docs/pipeline.md for the execution order.
"""MedicalNet 3D ResNet-18 分类器(方案 §5.1/§6.1)。

M2: MedicalNet 预训练;M1: 同构随机初始化;A1: 同 M2 但输入为瓣膜小ROI。
单 logit 输出 + BCEWithLogits。
"""
from bavct import config
import sys

import torch
import torch.nn as nn

MEDICALNET_DIR = str(config.MEDICALNET_DIR)
PRETRAIN_PTH = str(config.MEDICALNET_WEIGHTS)






class ResNet18Classifier(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        m = backbone
        self.conv1, self.bn1, self.relu = m.conv1, m.bn1, m.relu
        self.maxpool = m.maxpool
        self.layer1, self.layer2 = m.layer1, m.layer2
        self.layer3, self.layer4 = m.layer3, m.layer4
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(512, 1)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        x = self.pool(x).flatten(1)
        return self.fc(self.dropout(x)).squeeze(1)




def _load_medicalnet():
    import sys

    sys.path.insert(0, str(config.MEDICALNET_DIR))
    try:
        from models import resnet as medicalnet_resnet
    except ImportError as exc:
        raise ImportError(
            'MedicalNet backbone not found; set BAVCT_MEDICALNET_DIR to a local '
            'clone of the MedicalNet repository (only needed for this arm).'
        ) from exc
    return medicalnet_resnet


def build_model(pretrained: bool, input_shape=(96, 96, 96)):
    medicalnet_resnet = _load_medicalnet()
    backbone = medicalnet_resnet.resnet18(
        sample_input_D=input_shape[0],
        sample_input_H=input_shape[1],
        sample_input_W=input_shape[2],
        shortcut_type="A",
        num_seg_classes=2,
    )
    if pretrained:
        ckpt = torch.load(PRETRAIN_PTH, map_location="cpu", weights_only=True)
        state = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
        state = {k.replace("module.", ""): v for k, v in state.items()}
        missing, unexpected = backbone.load_state_dict(state, strict=False)
        print(f"pretrain loaded, missing={len(missing)} unexpected={len(unexpected)}")
    return ResNet18Classifier(backbone)
