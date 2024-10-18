#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : base.py
# Author : Honghua Dong
# Email  : dhh19951@gmail.com
#
# Distributed under terms of the MIT license.

import dgl
import torch
import torch.nn as nn
from dgl import DGLGraph
from megraph.args_utils import ArgsBuilder
from megraph.representation import MultiFeatures

__all__ = ["BaseGraphLayer"]


class BaseGraphLayer(nn.Module, ArgsBuilder):
    r"""The base graph layer"""

    def __init__(self, output_dims):
        super(BaseGraphLayer, self).__init__()
        self.output_dims = output_dims

    def pre_ops(self, graph: DGLGraph, features: MultiFeatures):
        return features

    def update_edges(self, graph: DGLGraph, features: MultiFeatures):
        return features

    def update_nodes(self, graph: DGLGraph, features: MultiFeatures):
        return features

    def update_global(self, graph: DGLGraph, features: MultiFeatures):
        return features

    def post_ops(self, graph: DGLGraph, features: MultiFeatures):
        return features

    def forward(self, graph: DGLGraph, features: MultiFeatures):
        with graph.local_scope():
            features = self.pre_ops(graph, features)
            features = self.update_edges(graph, features)
            features = self.update_nodes(graph, features)
            features = self.update_global(graph, features)
            features = self.post_ops(graph, features)
        return features

    def get_output_dims(self):
        return self.output_dims

    __hyperparams__ = []
    __parser__ = None
    __prefix__ = "--"

    @classmethod
    def register_layer_args(cls, parser, prefix=None):
        pass
