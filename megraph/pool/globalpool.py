#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : globalpool.py
# Author : Honghua Dong
# Email  : dhh19951@gmail.com
#
# Distributed under terms of the MIT license.

import torch.nn as nn
from dgl.nn.pytorch.glob import (
    AvgPooling,
    GlobalAttentionPooling,
    MaxPooling,
    SumPooling,
)
from dgl.readout import max_edges, mean_edges, softmax_edges, sum_edges

__all__ = ["get_global_pooling", "get_global_edge_pooling"]

GLOBAL_POOLING = dict(
    sum=SumPooling,
    mean=AvgPooling,
    max=MaxPooling,
    att=GlobalAttentionPooling,
)


class BaseEdgePooling(nn.Module):
    """Base class for pooling among all edges."""

    def __init__(self):
        super().__init__()

    def get_readout_func(self):
        raise NotImplementedError

    def forward(self, graph, feat):
        with graph.local_scope():
            graph.edata["h"] = feat
            readout = self.get_readout_func()(graph, "h")
            return readout


class SumEdgePooling(BaseEdgePooling):
    def get_readout_func(self):
        return sum_edges


class AvgEdgePooling(BaseEdgePooling):
    def get_readout_func(self):
        return mean_edges


class MaxEdgePooling(BaseEdgePooling):
    def get_readout_func(self):
        return max_edges


GLOBAL_EDGE_POOLING = dict(
    sum=SumEdgePooling,
    mean=AvgEdgePooling,
    max=MaxEdgePooling,
)


def get_global_pooling(method, ndim=None):
    if method == "att":
        gate_nn = nn.Linear(ndim, 1, bias=True)
        return GlobalAttentionPooling(gate_nn)
    return GLOBAL_POOLING[method]()


def get_global_edge_pooling(method):
    return GLOBAL_EDGE_POOLING[method]()
