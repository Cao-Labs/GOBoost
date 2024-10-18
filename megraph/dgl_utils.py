#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : dgl_utils.py
# Author : Honghua Dong, Jiawei Xu
# Email  : dhh19951@gmail.com, 1138217508@qq.com
#
# Distributed under terms of the MIT license.

import dgl
import dgl.backend as F
import numpy as np
import torch

# See also dgl.transforms.add_reverse_edges
# from dgl.transforms import add_reverse_edges


def add_reverse_edges(g: dgl.DGLGraph):
    if g.batch_size > 1:
        gs = dgl.unbatch(g)
        new_gs = [add_reverse_edges(g) for g in gs]
        return dgl.batch(new_gs)

    src, dst = g.edges()
    g.add_edges(dst, src)
    return g


def find_root(f, x):
    """Find the root of the tree that x belongs to."""
    if f[x] == x:
        return x
    fx = find_root(f, f[x])
    f[x] = fx  # Shortcut
    return fx


def get_num_disjoint(g: dgl.DGLGraph):
    """Get the number of disjoint components in the graph using Disjoint-set
    data structure."""

    num = n = g.num_nodes()
    src, dst = g.edges()
    f = np.arange(n)
    for i, j in zip(src, dst):
        x, y = i.item(), j.item()
        x = find_root(f, x)
        y = find_root(f, y)
        if x != y:
            if np.random.randint(2) == 0:
                x, y = y, x
            f[x] = y
            num -= 1
    return num


def position_to_encoding(pos, dim):
    # Compute the positional encodings as in Transformer.
    if dim % 2 == 1:
        raise ValueError("positional encoding dim must be even")
    pe = torch.zeros(pos.shape[0], dim, dtype=torch.float)
    pos = pos.unsqueeze(-1)
    div_term = torch.exp(
        torch.arange(0, dim, 2, dtype=torch.float) * -(np.log(10000.0) / dim)
    )
    pe[:, 0::2] = torch.sin(pos * div_term)
    pe[:, 1::2] = torch.cos(pos * div_term)
    return pe


def bfs_pe(g: dgl.DGLGraph, dim, num_rep):
    """Bfs Positional Encoding, compute the distance from randomly sampled source node
    as position, convert the position to encoding similar to Transformer."""
    n = g.num_nodes()
    res = []
    for _ in range(num_rep):
        pos = torch.zeros(n, dtype=torch.float)
        num_remains = n
        idx = np.arange(n)
        remains = np.ones(n, dtype="bool")
        while num_remains > 0:
            choice = np.random.choice(num_remains)
            s = idx[remains][choice]
            for i, nodes in enumerate(dgl.bfs_nodes_generator(g, s)):
                pos[nodes] = i
                num_remains -= len(nodes)
                remains[nodes.numpy()] = 0
        res.append(position_to_encoding(pos, dim))
    return torch.cat(res, dim=-1)


def positioal_encoding(g: dgl.DGLGraph, pe_type, pe_dim, pe_rep=1):
    if pe_type == "laplacian":
        pe = dgl.laplacian_pe(g, pe_dim)
    elif pe_type == "random_walk":
        pe = dgl.random_walk_pe(g, pe_dim)
    elif pe_type == "bfs":
        if pe_dim % pe_rep != 0:
            raise ValueError("pe_dim must be divisible by pe_rep for bfs pe")
        pe = bfs_pe(g, pe_dim // pe_rep, pe_rep)
    else:
        raise ValueError("Unknown Positional Encoding: {}".format(pe_type))
    return pe


def add_self_loop(g: dgl.DGLGraph, etype=None):
    # See alse dgl.add_self_loop
    etype = g.to_canonical_etype(etype)
    assert etype[0] == etype[2], (
        "add_self_loop does not support unidirectional bipartite graphs: {}."
        "Please make sure the types of head node and tail node are identical."
        "".format(etype)
    )
    nodes = g.nodes(etype[0])
    g.add_edges(nodes, nodes, etype=etype)
    return g


def remove_self_loop(g: dgl.DGLGraph, etype=None):
    # See also dgl.remove_self_loop
    etype = g.to_canonical_etype(etype)
    assert etype[0] == etype[2], (
        "remove_self_loop does not support unidirectional bipartite graphs: {}."
        "Please make sure the types of head node and tail node are identical."
        "".format(etype)
    )
    u, v = g.edges(form="uv", order="eid", etype=etype)
    self_loop_eids = F.tensor(F.nonzero_1d(u == v), dtype=F.dtype(u))
    g.remove_edges(self_loop_eids, etype=etype, store_ids=False)
    return g


def augment_graph_if_below_thresh(graph: dgl.DGLGraph, edge_feat=None, thresh=None):
    batch_num_nodes = graph.batch_num_nodes()
    if thresh is None or torch.square(batch_num_nodes).float().mean() >= thresh:
        return graph, edge_feat, False  # thresh not defined or too much cost
    # To fully connected graph
    device = batch_num_nodes.device
    with graph.local_scope():
        if edge_feat is not None:
            graph.edata["x"] = edge_feat
        unbatch_graphs = dgl.unbatch(graph)

    augmented_graphs = []
    for g in unbatch_graphs:
        n = g.num_nodes()
        nodes = torch.arange(n, device=device)
        edge_feat = g.edata.get("x", None)
        # use all one feature is edge_feat is None
        if edge_feat is None:
            edge_feat = torch.ones((g.num_edges(), 1), device=device)
        # edge_feat for new edges are all zero
        new_ex = torch.zeros((n**2, edge_feat.shape[-1]), device=device)
        src, dst = g.edges()
        new_ex[src * n + dst] = edge_feat

        # build new graph
        src, dst = torch.meshgrid(nodes, nodes)  # for torch >=1.10.0, indexing="ij"
        src, dst = src.flatten(), dst.flatten()
        mask = src != dst
        src, dst, new_ex = src[mask], dst[mask], new_ex[mask]
        new_g = dgl.graph((src, dst), num_nodes=n)
        new_g.edata["x"] = new_ex
        augmented_graphs.append(new_g)

    graph = dgl.batch(augmented_graphs)
    return graph, graph.edata["x"], True


if __name__ == "__main__":
    g = dgl.graph(([0, 1, 1, 2, 3, 4, 4, 5], [1, 0, 2, 1, 4, 3, 5, 4]))
    # g.ndata["feat"] = torch.ones(g.num_nodes(), 1)
    # g = positioal_encoding(g, "bfs", 6, pe_rep=2)
    # print(g.ndata["feat"])
    from IPython import embed

    embed()
