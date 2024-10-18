#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : edgepool_dgl.py
# Author : Honghua Dong
# Email  : dhh19951@gmail.com
#
# Distributed under terms of the MIT license.

# Improved edgepool based on the implementation in pytorch_geometric

import bisect
import time
from collections import namedtuple
from typing import Any, Dict

import dgl
import dgl.function as fn
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_scatter
from dgl import DGLGraph
from dgl.nn.functional import edge_softmax
from megraph.torch_utils import get_index_of_fused_groups
from torch_geometric.utils import softmax
from torch_sparse import coalesce

from .cluster import compute_cluster_with_graph


class EdgePooling(nn.Module):
    r"""The edge pooling operator improved from the `"Towards Graph Pooling by Edge
    Contraction" <https://graphreason.github.io/papers/17.pdf>`_ and
    `"Edge Contraction Pooling for Graph Neural Networks"
    <https://arxiv.org/abs/1905.10990>`_ papers.

    Args:
        node_feat_dim (int): Size of each input node feature.
        edge_feat_dim (int, optional): Size of each input edge feature.
        edge_score_method (str, optional): The method to apply to compute the
            edge score from raw edge scores. By default, this is the softmax
            over all incoming edges for each node. If set to "sigmoid", the
            edge score is computed as :math:`sigmoid(raw_edge_score)`. If set
            to "tanh", the edge score is computed as :math:`tanh(raw_edge_score)`.
            Included options are :obj:`softmax`, :obj:`sigmoid`, and :obj:`tanh`.
            (default: :obj:`softmax`)
        node_ratio (float, optional): The ratio of nodes to keep in the pooled.
        degree_ratio (float, optional): Not Effective Now.
        topedge_ratio (float, optional): The ratio of top edges to use.
        dropout (float, optional): The probability with
            which to drop edge scores during training. (default: :obj:`0`)
        edge_aggr (str, optional): The aggregation method for edge features to
            use when pooling. (default: :obj:`add`)
        node_aggr (str, optional): The aggregation method for node features to
            use when pooling. (default: :obj:`sum`)
        pool_with_cluster_score (bool, optional): Whether to mul cluster score
            to pooled node features. (default: :obj:`False`)
        unpool_with_cluster_score (bool, optional): Whether to div cluster score
            to unpooled node features. (default: :obj:`False`)
        add_to_edge_score (float, optional): This is added to each
            computed edge score. Adding this greatly helps with unpool
            stability. (default: :obj:`0.5`)
    """

    def __init__(
        self,
        node_feat_dim,
        edge_feat_dim=0,
        edge_score_method="softmax",
        node_ratio=None,
        degree_ratio=None,
        topedge_ratio=None,
        cluster_size_limit=None,
        dropout=0,
        edge_aggr="add",
        node_aggr="sum",
        pool_with_cluster_score=False,
        unpool_with_cluster_score=False,
        remove_self_loop=True,
        add_to_edge_score=0.5,
        noise_scale=0.001,
    ):
        super().__init__()
        self.node_feat_dim = node_feat_dim
        self.edge_feat_dim = edge_feat_dim
        self.edge_score_method = edge_score_method
        self.node_ratio = node_ratio
        self.topedge_ratio = topedge_ratio
        self.cluster_size_limit = cluster_size_limit
        self.dropout = nn.Dropout(dropout)
        self.edge_aggr = edge_aggr
        self.node_aggr = node_aggr
        self.pool_with_cluster_score = pool_with_cluster_score
        self.unpool_with_cluster_score = unpool_with_cluster_score
        self.remove_self_loop = remove_self_loop
        self.add_to_edge_score = add_to_edge_score
        self.noise_scale = noise_scale

        self.linear_src = nn.Linear(node_feat_dim, 1, bias=False)
        self.linear_dst = nn.Linear(node_feat_dim, 1, bias=False)
        edge_feat_dim = edge_feat_dim if edge_feat_dim > 0 else 1  # when augmented
        self.linear_edge = nn.Linear(edge_feat_dim, 1, bias=False)

        # self.reset_parameters()

    def reset_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                module.reset_parameters()

    def forward(
        self,
        g: DGLGraph,
        node_feat: torch.Tensor,
        edge_feat: torch.Tensor = None,
        cache: Dict[str, Any] = None,
    ):
        r"""Forward computation which computes the raw edge score, normalizes
        it, and merges the edges.

        Args:
            g (DGLGraph): The graph to perform edge pooling on.
            node_feat (Tensor): The input node features of shape :math:`(N, D_n)`
            edge_feat (Tensor, optional): The input edge features of shape :math:`(N, D_e)`

        Return types:
            * **intra_graph** *(DGLGraph)* - The intra-graph representing the
              pooled graph.
            * **inter_graph** *(DGLGraph)* - The inter-graph representing the
              clustering mapping.
            * **cluster** *(Tensor)* - The node cluster mapping from original graph
              to pooled graph.
        """
        with g.local_scope():  # 一个上下文管理器，它保证在这个代码块中对图 g 的修改不会影响到外部的图。
            g.srcdata["out_src"] = self.linear_src(node_feat)
            g.dstdata["out_dst"] = self.linear_dst(node_feat)
            # 通过线性变换计算源节点和目标节点的输出特征，并将结果存储在图的节点数据中。
            g.apply_edges(fn.u_add_v("out_src", "out_dst", "edge_score"))
            # 应用一个函数 fn.u_add_v 到图的每一条边上，该函数将源节点和目标节点的输出特征相加，计算出原始的边分数，并将结果存储在边数据中。
            e = g.edata["edge_score"].view(-1)
            # 获取边分数，并将其形状变为一维。
            if edge_feat is not None:
                e += self.linear_edge(edge_feat).view(-1)  #没有边特征跳过这一行代码
            e = self.dropout(e)
            if self.edge_score_method == "softmax":
                e = edge_softmax(g, e)
            elif self.edge_score_method == "sigmoid":
                e = F.sigmoid(e)
            elif self.edge_score_method == "tanh":
                e = F.tanh(e)
            else:
                raise ValueError(
                    "Unknown edge score method: {}".format(self.edge_score_method)
                )
            e = e + self.add_to_edge_score

        with g.local_scope():
            g.edata["edge_score"] = e
            intra_graph, inter_graph, cluster, new_edge_feat = self.__merge_edges__(
                g, edge_feat
            )
        return intra_graph, inter_graph, cluster, new_edge_feat


    def compute_new_edges(self, g: DGLGraph, cluster, n_nodes, edge_feat):
        edges = torch.stack(g.edges(), dim=0)
        new_edges, edge_feat = coalesce(
            cluster[edges], edge_feat, n_nodes, n_nodes, op=self.edge_aggr
        )
        src, dst = new_edges[0], new_edges[1]
        if self.remove_self_loop:
            mask = src != dst
            src, dst = src[mask], dst[mask]
            if edge_feat is not None:
                edge_feat = edge_feat[mask]
        return src, dst, edge_feat

    def __merge_edges__(self, g: DGLGraph, edge_feat: torch.Tensor = None):
        num_nodes = g.num_nodes()
        batch_num_nodes_tensor = g.batch_num_nodes()
        batch_num_nodes = batch_num_nodes_tensor.tolist()
        dtype = batch_num_nodes_tensor.dtype
        device = batch_num_nodes_tensor.device
        new_batch_num_nodes, cluster, cluster_score = compute_cluster_with_graph(
            g,
            self.node_ratio,
            self.topedge_ratio,
            self.cluster_size_limit,
            self.noise_scale,
        )
        new_num_nodes = sum(new_batch_num_nodes)
        # Collapse the edges
        src, dst, new_edge_feat = self.compute_new_edges(
            g, cluster, new_num_nodes, edge_feat
        )

        # compute the batch for new edges
        # All cpu oprs (otherwise will be slower)
        src_list = src.tolist()  # src is sorted
        new_batch_num_edges = []
        if len(batch_num_nodes) == 1:
            new_batch_num_edges.append(len(src_list))
        else:
            cur_n, last_idx = 0, 0
            for n in new_batch_num_nodes:
                cur_n += n
                cur_idx = bisect.bisect_left(src_list, cur_n)
                new_batch_num_edges.append(cur_idx - last_idx)
                last_idx = cur_idx

        new_batch_num_nodes_tensor = torch.tensor(
            new_batch_num_nodes, dtype=dtype, device=device
        )
        new_batch_num_edges_tensor = torch.tensor(
            new_batch_num_edges, dtype=dtype, device=device
        )
        intra_graph = dgl.graph((src, dst), num_nodes=new_num_nodes)
        intra_graph.set_batch_num_nodes(new_batch_num_nodes_tensor)
        intra_graph.set_batch_num_edges(new_batch_num_edges_tensor)

        # Build the inter-graph (i -> j + num_nodes for each graph) and compute new_x
        old_nodes_idx, new_nodes_idx = get_index_of_fused_groups(
            batch_num_nodes, new_batch_num_nodes, device
        )
        new_src = old_nodes_idx

        new_dst = cluster.clone()
        last_n, cur_n = 0, 0
        for n in batch_num_nodes:
            last_n = cur_n
            cur_n += n
            new_dst[last_n:cur_n] += cur_n

        def fuse(x, y, n):
            z = x.new_zeros(n * 2)
            z[0::2] = x
            z[1::2] = y
            return z

        inter_src = fuse(new_src, new_dst, cur_n)
        inter_dst = fuse(new_dst, new_src, cur_n)
        inter_graph = dgl.graph(
            (inter_src, inter_dst), num_nodes=num_nodes + new_num_nodes
        )
        inter_graph.set_batch_num_nodes(
            batch_num_nodes_tensor + new_batch_num_nodes_tensor
        )
        inter_graph.set_batch_num_edges(batch_num_nodes_tensor * 2)
        intra_graph.ndata["cluster_score"] = cluster_score
        return intra_graph, inter_graph, cluster, new_edge_feat

    def get_node_aggr(self, agg=None):
        agg = agg or self.node_aggr
        if agg == "sum":
            return torch_scatter.scatter_add
        return getattr(torch_scatter, f"scatter_{agg}")

    def pool(
        self,
        x: torch.Tensor,
        intra_graph: DGLGraph,
        cluster: torch.Tensor,
        agg: str = None,
    ):
        """
        Return types:
            * **x** *(Tensor)* - The pooled node features.
        """
        num_nodes = intra_graph.num_nodes()
        node_aggr = self.get_node_aggr(agg)
        new_x = node_aggr(x, cluster, dim=0, dim_size=num_nodes)
        if self.pool_with_cluster_score:
            new_x = new_x * intra_graph.ndata["cluster_score"].view(-1, 1)
        return new_x

    def unpool(
        self,
        x: torch.Tensor,
        intra_graph: DGLGraph,
        inter_graph: DGLGraph,
        cluster: torch.Tensor,
    ):
        r"""Unpools a previous edge pooling step.

        For unpooling, :obj:`x` should be of same shape as those produced by
        this layer's :func:`forward` function. Then, it will produce an
        unpooled feature :obj:`x`.

        Args:
            x (Tensor): The node features.
            intra_graph (DGLGraph): The intra-graph computed in forwrad function.
            inter_graph (DGLGraph): The inter-graph computed in forwrad function.
            cluster (Tensor): The cluster assignment of the nodes.

        Return types:
            * **x** *(Tensor)* - The unpooled node features.
        """

        if self.unpool_with_cluster_score:
            x = x / intra_graph.ndata["cluster_score"].view(-1, 1)
        new_x = x[cluster]

        return new_x

    def extra_repr(self):
        """Set the extra representation of the module,
        which will come into effect when printing the model.
        """
        summary = f"in={self.node_feat_dim}, score_func={self.edge_score_method}, "
        if self.add_to_edge_score != 0:
            summary += f"add_to_edge_score={self.add_to_edge_score}, "
        return summary
