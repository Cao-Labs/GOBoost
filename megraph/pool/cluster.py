#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : cluster.py
# Author : Honghua Dong
# Email  : dhh19951@gmail.com
#
# Distributed under terms of the MIT license.

"""Use taichi to speed up edge contraction computation"""

import dgl
import taichi as ti
import torch

# NOTE: TO use taichi, you need to call ti.init() in your main file.


@ti.func
def find_root(f: ti.template(), x):
    """Find the root of the tree that x belongs to."""
    root = x
    while f[root] != root:
        root = f[root]
    while f[x] != root:  # Shortcut
        f[x] = root
        x = f[x]
    return root


@ti.kernel
def iter_edges(
    batch_num_nodes: ti.types.ndarray(),
    batch_num_edges: ti.types.ndarray(),
    cur_nstart: ti.types.ndarray(),
    cur_estart: ti.types.ndarray(),
    src: ti.types.ndarray(),
    dst: ti.types.ndarray(),
    scores: ti.types.ndarray(),
    edge_index: ti.types.ndarray(),
    node_ratio: ti.f64,
    topedge_ratio: ti.f64,
    cluster_size_limit: int,
    f: ti.types.ndarray(),
    s: ti.types.ndarray(),
    cnt: ti.types.ndarray(),
    root: ti.types.ndarray(),
    is_root: ti.types.ndarray(),
    idx: ti.types.ndarray(),
    new_batch_num_nodes: ti.types.ndarray(),
):
    """Use taichi kernel to speed up, compute the clustering result in batch."""
    for i in range(batch_num_nodes.shape[0]):
        n_nodes, n_edges = batch_num_nodes[i], batch_num_edges[i]
        num, target_n_nodes = n_nodes, int(n_nodes * node_ratio)
        n_top_edges = n_edges
        if topedge_ratio < 1.0:
            n_top_edges = int(n_edges * topedge_ratio)
        for j in range(n_top_edges):
            k = edge_index[cur_estart[i] + j]
            u = src[k]
            v = dst[k]
            if u == v:  # ignore self-loop
                continue
            x = find_root(f, u)
            y = find_root(f, v)
            if x == y:  # They already in the same cluster
                continue
            if cnt[x] + cnt[y] > cluster_size_limit:
                continue
            # Otherwise, contract the edge between x and y.
            # turned off to when compare results
            if ti.random() < 0.5:
                x, y = y, x
            # Merge cluster x to cluster y
            f[x] = y
            # The sum and count info are stored in the new root y
            cnt[y] += cnt[x]
            s[y] += s[x] + scores[k]
            num -= 1
            if num == target_n_nodes:
                break
        num_new_nodes = 0
        for j in range(cur_nstart[i], cur_nstart[i] + n_nodes):
            root[j] = find_root(f, j)
            if root[j] == j:
                is_root[j] = 1
                idx[j] = num_new_nodes
                num_new_nodes += 1
        new_batch_num_nodes[i] = num_new_nodes


def compute_cluster(
    node_ratio,
    topedge_ratio,
    cluster_size_limit,
    num_nodes,
    num_edges,
    batch_num_nodes,
    batch_num_edges,
    src,
    dst,
    edge_argsort,
    edge_scores,
):
    device = edge_scores.device

    batch_size = len(batch_num_nodes)
    batch_num_nodes = torch.tensor(batch_num_nodes, device=device)
    batch_num_edges = torch.tensor(batch_num_edges, device=device)
    cur_nstart = torch.zeros(batch_size, device=device, dtype=torch.long)
    cur_estart = torch.zeros(batch_size, device=device, dtype=torch.long)
    for i in range(batch_size - 1):
        cur_nstart[i + 1] = cur_nstart[i] + batch_num_nodes[i]
        cur_estart[i + 1] = cur_estart[i] + batch_num_edges[i]
    # The Disjoint-set data structure: f(father), s(sum), cnt(count).
    f = torch.arange(num_nodes, device=device)
    s = torch.zeros(num_nodes, device=device, dtype=torch.float)
    cnt = torch.ones(num_nodes, device=device, dtype=torch.long)
    # The Outputs of iter_edges
    root = torch.zeros(num_nodes, device=device, dtype=torch.long)
    is_root = torch.zeros(num_nodes, device=device, dtype=torch.long)
    idx = torch.zeros(num_nodes, device=device, dtype=torch.long)
    new_batch_num_nodes = torch.zeros(batch_size, device=device, dtype=torch.long)

    iter_edges(
        batch_num_nodes,
        batch_num_edges,
        cur_nstart,
        cur_estart,
        src,
        dst,
        edge_scores,
        edge_argsort,
        node_ratio,
        topedge_ratio,
        cluster_size_limit,
        f,
        s,
        cnt,
        root,
        is_root,
        idx,
        new_batch_num_nodes,
    )
    cur_idx1, cur_idx2 = 0, 0
    for n1, n2 in zip(batch_num_nodes, new_batch_num_nodes):
        idx[cur_idx1 : cur_idx1 + n1] += cur_idx2
        cur_idx1 += n1
        cur_idx2 += n2
    cluster = idx[root]
    roots = root[is_root.bool()]
    cluster_score = (s[roots] + 1.0) / cnt[roots]
    return new_batch_num_nodes.tolist(), cluster, cluster_score


# -----------------------------------------------------------------------
# -----------------------------------------------------------------------
# -----------------------------------------------------------------------


def compute_cluster_with_graph(
    g: dgl.DGLGraph,
    node_ratio: float = None,
    topedge_ratio: float = None,
    cluster_size_limit: int = None,
    noise_scale: float = None,
):
    num_nodes, num_edges = g.num_nodes(), g.num_edges()
    batch_num_nodes = g.batch_num_nodes().tolist()
    batch_num_edges = g.batch_num_edges().tolist()

    if node_ratio is None:
        node_ratio = 0.0
    if cluster_size_limit is None:
        cluster_size_limit = num_nodes
    if node_ratio <= 0.0 and cluster_size_limit >= num_nodes:
        cluster_size_limit = 2
    if topedge_ratio is None:
        topedge_ratio = 10.0  # >1.0 means no limit

    edge_scores = g.edata["edge_score"]
    src, dst = g.edges()

    # The edge score are in range [-1, 1]
    # Make bias to the edge scores so that edges in early graph goes first
    addition = torch.zeros_like(edge_scores)
    cur_idx = 0
    for i, ne in enumerate(batch_num_edges):
        addition[cur_idx : cur_idx + ne] = -i * 5.0
        cur_idx += ne
    edge_scores_for_sort = edge_scores + addition
    if noise_scale is not None:
        noise = torch.rand_like(edge_scores) * noise_scale
        edge_scores_for_sort = edge_scores_for_sort + noise
    edge_argsort = torch.argsort(edge_scores_for_sort, descending=True)
    return compute_cluster(
        node_ratio,
        topedge_ratio,
        cluster_size_limit,
        num_nodes,
        num_edges,
        batch_num_nodes,
        batch_num_edges,
        src,
        dst,
        edge_argsort,
        edge_scores,
    )
