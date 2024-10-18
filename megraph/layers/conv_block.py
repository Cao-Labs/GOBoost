#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : conv_block.py
# Author : Honghua Dong
# Email  : dhh19951@gmail.com
#
# Distributed under terms of the MIT license.

import torch
import torch.nn as nn

__all__ = ["ConvBlock"]

from dgl import DGLGraph
from megraph.representation import MultiFeatures


class ConvBlock(nn.Module):
    def __init__(self, conv, norms=None, act=None, dropout=None):
        super(ConvBlock, self).__init__()
        self.conv = conv
        self.norms = norms
        self.act = act
        self.dropout = dropout

    def forward(self, graph: DGLGraph, features: MultiFeatures) -> MultiFeatures:
        features = self.conv(graph, features)
        return features.apply_fn(self.norms).apply_fn(self.act).apply_fn(self.dropout)

    def get_output_dims(self):
        return self.conv.get_output_dims()


class RGCNBlock(nn.Module):
    def __init__(self, conv, norms=None, act=None, dropout=None, inter_connect="edge"):
        super(RGCNBlock, self).__init__()
        self.conv = conv
        self.norms = norms
        self.act = act
        self.dropout = dropout
        if inter_connect == "edge":
            self.forward = self.forward_v1
        elif inter_connect == "addnode":
            self.forward = self.forward_v2

    def forward_v1(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        node_num: int,
    ) -> MultiFeatures:
        feat = self.conv(x, edge_index, edge_type)
        feat = x[:node_num]
        features = MultiFeatures([None, feat, None])
        return features.apply_fn(self.norms).apply_fn(self.act).apply_fn(self.dropout)

    def forward_v2(
        self,
        x: torch.Tensor,
        parent: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> MultiFeatures:
        feat = self.conv(x, parent, edge_index, edge_attr)
        features = MultiFeatures([None, feat, None])
        return features.apply_fn(self.norms).apply_fn(self.act).apply_fn(self.dropout)
