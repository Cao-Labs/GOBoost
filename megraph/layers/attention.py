#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : attention.py
# Author : Honghua Dong
# Email  : dhh19951@gmail.com
#
# Distributed under terms of the MIT license.

import torch
import torch.nn as nn
from megraph.torch_utils import get_activation


class AttentionWeightLayer(nn.Module):
    """[Experimental] The attention weight layer for fusing multiple features."""

    def __init__(self, feat_dim, activation):
        super().__init__()
        self.fc = nn.Linear(feat_dim, 1, bias=False)
        self.act = get_activation(activation)

    def forward(self, xs):
        x = torch.stack(xs, dim=-2)  # [N, len(xs), feat_dim]
        x = self.fc(x).squeeze(-1)  # [N, len(xs)]
        w = self.act(x)
        return w
