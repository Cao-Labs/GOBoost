#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : encoder.py
# Author : Honghua Dong, Jiawei Xu
# Email  : dhh19951@gmail.com, 1138217508@qq.com
#
# Distributed under terms of the MIT license.

# Modified based on https://github.com/snap-stanford/ogb/blob/master/ogb/graphproppred/mol_encoder.py

import torch.nn as nn
from ogb.utils.features import get_atom_feature_dims, get_bond_feature_dims

full_atom_feature_dims = get_atom_feature_dims()
full_bond_feature_dims = get_bond_feature_dims()


class EncoderBase(nn.Module):
    def __init__(self, emb_dim, feature_dims):
        super(EncoderBase, self).__init__()
        self.embeddings = nn.ModuleList()

        for _, dim in enumerate(feature_dims):
            emb = nn.Embedding(dim, emb_dim)
            nn.init.xavier_uniform_(emb.weight.data)
            self.embeddings.append(emb)

    def forward(self, x):
        embed_x = 0
        for i in range(x.shape[1]):
            embed_x += self.embeddings[i](x[:, i])
        return embed_x


class AtomEncoder(EncoderBase):
    def __init__(self, emb_dim):
        super(AtomEncoder, self).__init__(emb_dim, full_atom_feature_dims)


class BondEncoder(EncoderBase):
    def __init__(self, emb_dim):
        super(BondEncoder, self).__init__(emb_dim, full_bond_feature_dims)


def get_input_embedding(
    input_dim,
    output_dim,
    embed_method="linear",
    is_node_feat=True,
    bias=True,
):
    if embed_method == "linear":
        return nn.Linear(input_dim, output_dim, bias=bias)
    elif embed_method == "embed":
        return nn.Embedding(input_dim, output_dim)
    elif embed_method == "mol":
        return AtomEncoder(output_dim) if is_node_feat else BondEncoder(output_dim)
    else:
        raise ValueError("Unknown embedding method: {}".format(embed_method))
