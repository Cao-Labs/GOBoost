#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : layers.py
# Author : Honghua Dong
# Email  : dhh19951@gmail.com
#
# Distributed under terms of the MIT license.

import dgl
import dgl.function as fn
import torch
import torch.nn as nn
from dgl import DGLGraph
from dgl.nn.functional import edge_softmax
from dgl.nn.pytorch import (AGNNConv, APPNPConv, ChebConv, GATv2Conv, GINConv,
                            GraphConv, SAGEConv, SGConv, TAGConv)
from dgl.nn.pytorch.conv.pnaconv import AGGREGATORS, SCALERS, PNAConvTower
from dgl.utils import expand_as_pair
from megraph.pool.globalpool import get_global_edge_pooling, get_global_pooling
from megraph.representation import MultiFeatures
from megraph.torch_utils import apply_trans, sum_not_none_elements
from megraph.utils import apply_fn_on_list

from . import register_function
from .base import BaseGraphLayer

@register_function("gcn")
class GCNLayer(BaseGraphLayer):
    def __init__(self, input_dims, output_dims, **kwargs):
        super(GCNLayer, self).__init__(output_dims)
        in_feats = input_dims[1]
        out_feats = output_dims[1]
        self.conv = GraphConv(in_feats, out_feats, **kwargs)

    def update_nodes(self, graph: DGLGraph, features: MultiFeatures):
        return features.replace_nodes_features(
            self.conv(graph, features.nodes_features)
        )
