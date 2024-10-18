#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : mee.py
# Author : Honghua Dong
# Email  : dhh19951@gmail.com
#
# Distributed under terms of the MIT license.

import torch.nn.functional as F
import dgl
import torch
import torch.nn as nn
from dgl import DGLGraph
from megraph.representation import MultiFeatures
from megraph.torch_utils import get_index_of_fused_groups
from megraph.utils import check_len, residual_when_same_shape_on_list

__all__ = ["MeeLayer"]


class MeeLayer(nn.Module):
    def __init__(
        self,
        intra_convs,
        inter_convs,
        pooling=None,
        inter_fusion_fcs=None,
        cross_update_method="conv",
        start_height=0,
        end_height=None,
        vertical_first=False,
        stem_beta=1.0,
        branch_beta=0.5,
        keep_beta=1.0,
        cross_beta=1.0,
        dropout=None,
        dropout_after_residual=False,
        eps=0.2,
        perturbed=False,
    ):
        super(MeeLayer, self).__init__()
        self.intra_convs = intra_convs
        if cross_update_method == "pool":
            inter_convs = [None] * len(inter_convs)
        if cross_update_method == "conv":
            pooling = None
        self.inter_convs = inter_convs
        self.pooling = pooling
        self.inter_fusion_fcs = inter_fusion_fcs
        self.cross_update_method = cross_update_method
        self.start_height = start_height
        self.end_height = end_height
        self.vertical_first = vertical_first
        self.stem_beta = stem_beta
        self.branch_beta = branch_beta
        self.keep_beta = keep_beta
        self.cross_beta = cross_beta
        self.dropout = dropout
        self.dropout_after_residual = dropout_after_residual
        self.eps = eps
        self.perturbed = perturbed

    def get_conv(self, convs, i, height):
        """Get the conv layer for the i-th height, can use shared convs"""
        if i < 0 or i >= height:
            raise ValueError("index must be in [0, height)")
        return convs[i * len(convs) // height]

    def get_fusion_fc(self, fusion_fcs, i, height):
        """Get the fusion function for the i-th height"""
        if i < 0 or i >= height:
            raise ValueError("index must be in [0, height)")
        if fusion_fcs is None:
            return None
        return fusion_fcs[i * len(fusion_fcs) // height]

    def _cross_update(
        self,
        conv,
        fusion_fc,
        xg: DGLGraph,
        yg: DGLGraph,
        inter_g: DGLGraph,
        cluster: torch.Tensor,  # from larger graph to smaller graph
        x: MultiFeatures,
        y: MultiFeatures,
    ):
        """cross update features from two consecutive heights"""
        nx = x.nodes_features
        ny = y.nodes_features

        if self.cross_update_method in ["conv", "combine"]:
            x_ns, y_ns = xg.batch_num_nodes(), yg.batch_num_nodes()
            x_idx, y_idx = get_index_of_fused_groups(x_ns, y_ns, device=nx.device)
            cat_xy = torch.cat([nx, ny], dim=0)
            nz = torch.zeros_like(cat_xy)
            nz[x_idx] = nx
            nz[y_idx] = ny

            z = MultiFeatures([None, nz, None])
            z = conv(inter_g, z)
            nz = z.nodes_features
            new_nx = nz[x_idx]
            new_ny = nz[y_idx]
        if self.cross_update_method in ["pool", "combine"]:
            new_nx = self.pooling.unpool(ny, yg, inter_g, cluster)
        if self.cross_update_method in ["pool"]:
            new_ny = self.pooling.pool(nx, yg, cluster)
        if fusion_fc is None:
            x = x.replace_nodes_features(nx * self.keep_beta + new_nx * self.cross_beta)  # 残差连接
            y = y.replace_nodes_features(ny * self.keep_beta + new_ny * self.cross_beta)  # 残差连接
        else:

            def fuse(xs):
                w = fusion_fc(xs)
                return sum([x * w[:, i : i + 1] for i, x in enumerate(xs)])

            x = x.replace_nodes_features(fuse([nx, new_nx]))
            y = y.replace_nodes_features(fuse([ny, new_ny]))
        return x, y

    def _rightward_update(self, height, inds, intra_graphs, xs):
        """update along intra-graphs by intra-convs"""
        res = []
        for i in range(height):
            x = xs[i]
            if inds[0] <= i <= inds[-1]:
                x = self.get_conv(self.intra_convs, i, height)(intra_graphs[i], x)
            res.append(x)
        return res

    def _vertical_update(self, height, inds, intra_graphs, inter_graphs, clusters, xs):
        """Update along inter-graphs by inter-convs"""
        for i in inds:
            if i + 1 >= height:
                continue
            xs[i], xs[i + 1] = self._cross_update(
                self.get_conv(self.inter_convs, i, height - 1),
                self.get_fusion_fc(self.inter_fusion_fcs, i, height),
                intra_graphs[i],
                intra_graphs[i + 1],
                inter_graphs[i],
                clusters[i],
                xs[i],
                xs[i + 1],
            )

    def _residual(self, xs, ys, stem_beta=None, branch_beta=None):
        if stem_beta is None:
            stem_beta = self.stem_beta
        if branch_beta is None:
            branch_beta = self.branch_beta
        return [
            x.residual_when_same_shape(y, stem_beta, branch_beta)
            for x, y in zip(xs, ys)
        ]

    def forward(self, height, intra_graphs, inter_graphs, clusters, features):
        check_len(intra_graphs, height)
        check_len(inter_graphs, height - 1)
        check_len(clusters, height - 1)
        check_len(features, height)

        sh = self.start_height
        eh = height - 1 if self.end_height is None else self.end_height
        xs = features

        if not self.vertical_first:
            xs = self._rightward_update(height, [sh, eh], intra_graphs, xs)

        inds = list(range(sh, eh))

        self._vertical_update(height, inds, intra_graphs, inter_graphs, clusters, xs)
        if len(inds) > 1:
            inds = inds[::-1]
            self._vertical_update(
                height, inds, intra_graphs, inter_graphs, clusters, xs
            )

        if self.vertical_first:
            xs = self._rightward_update(height, [sh, eh], intra_graphs, xs)
        if not self.dropout_after_residual:
            xs = [x.apply_fn(self.dropout) for x in xs]
        xs = self._residual(features, xs)
        if self.dropout_after_residual:
            xs = [x.apply_fn(self.dropout) for x in xs]
        return xs
