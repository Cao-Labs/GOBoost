#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : random_pool.py
# Author : Honghua Dong
# Email  : dhh19951@gmail.com
#
# Distributed under terms of the MIT license.

# Improved edgepool based on the implementation in pytorch_geometric

import bisect
import time
from typing import Any, Dict

import dgl
import dgl.function as fn
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_scatter
from dgl import DGLGraph
from megraph.torch_utils import get_index_of_fused_groups
from torch_sparse import coalesce

from .cluster import compute_cluster_with_graph


class RandomPooling(nn.Module):
    r"""
    Args:
        edge_aggr (str, optional): The aggregation method for edge features to
            use when pooling. (default: :obj:`add`)
        node_aggr (str, optional): The aggregation method for node features to
            use when pooling. (default: :obj:`sum`)
        pool_with_cluster_score (bool, optional): Whether to mul cluster score
            to pooled node features. (default: :obj:`False`)
        unpool_with_cluster_score (bool, optional): Whether to div cluster score
            to unpooled node features. (default: :obj:`False`)
    """

    def __init__(
        self,
        node_ratio=0.5,
        edge_aggr="add",
        node_aggr="sum",
        remove_self_loop=True,
    ):
        super().__init__()
        self.node_ratio = node_ratio
        self.edge_aggr = edge_aggr
        self.node_aggr = node_aggr
        self.remove_self_loop = remove_self_loop

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

        with g.local_scope():
            intra_graph, inter_graph, cluster, new_edge_feat = self.__merge_edges__(
                g, edge_feat, cache
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

    def __merge_edges__(
        self, g: DGLGraph, edge_feat: torch.Tensor = None, cache: Dict[str, Any] = None
    ):
        num_nodes = g.num_nodes()
        batch_num_nodes_tensor = g.batch_num_nodes()
        batch_num_nodes = batch_num_nodes_tensor.tolist()
        dtype = batch_num_nodes_tensor.dtype
        device = batch_num_nodes_tensor.device

        cluster, new_batch_num_nodes = [], []
        total_new_nodes = 0
        node_ratio = self.node_ratio or 0.5
        for n in batch_num_nodes:
            n_clusters = int(n * node_ratio)
            temp_cluster = np.random.randint(0, n_clusters, size=n)
            cluster.append(temp_cluster + total_new_nodes)
            total_new_nodes += n_clusters
            new_batch_num_nodes.append(n_clusters)

        new_num_nodes = sum(new_batch_num_nodes)
        cluster = torch.from_numpy(np.hstack(cluster)).to(device)
        cluster_score = torch.ones(new_num_nodes, device=device)

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

        new_x = x[cluster]

        return new_x
