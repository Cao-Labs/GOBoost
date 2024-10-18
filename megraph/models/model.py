#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : model.py
# Author : Honghua Dong, Jiawei Xu
# Email  : dhh19951@gmail.com, 1138217508@qq.com
#
# Distributed under terms of the MIT license.

from functools import partial

import torch
import torch.nn as nn
from dgl import DGLGraph
from megraph.args_utils import ArgsBuilder
from megraph.layers import ConvBlock, MLPLayer, get_input_embedding
from megraph.pool import get_global_pooling
from megraph.representation.features import MultiFeatures
from megraph.torch_utils import get_activation, get_norm_layer

__all__ = ["GraphModel", "MultiFeaturesModel"]


class GraphModel(nn.Module, ArgsBuilder):
    """Base class for graph models."""
    def __init__(self):
        super(GraphModel, self).__init__()

    __hyperparams__ = []
    __parser__ = None
    __prefix__ = "--"

    @classmethod
    def register_model_args(cls, parser, prefix=None):
        cls._set_parser_and_prefix(parser, prefix)


class MultiFeaturesModel(GraphModel):
    def __init__(
        self,
        input_dims,
        output_dims,
        pe_dim,
        task,
        build_conv,
        n_layers,
        g_hidden,
        n_hidden,
        e_hidden,
        norm_layer="layer",
        activation="relu",
        dropout=0.2,
        allow_zero_in_degree=True,
        embed_method={},
        use_input_embedding=False,
        use_pe_embedding=False,
        pe_op="cat",
        pe_hidden=None,
        use_scales=False,
        global_pool_methods=["mean"],
        use_global_pool_scales=False,
        last_hidden_dims=[512*4],
        last_simple=False,
        last_mlp=False,
        num_heads=None,
    ):
        super(MultiFeaturesModel, self).__init__()
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.pe_dim = pe_dim
        self.task = task
        self.build_conv = build_conv
        self.num_layers = n_layers

        self.hidden_dims = [g_hidden, n_hidden, e_hidden]

        self.activation = get_activation(activation)
        self.dropout = nn.Dropout(dropout)
        self.norm_layer = norm_layer
        self.allow_zero_in_degree = allow_zero_in_degree
        if "node" not in embed_method:
            embed_method["node"] = "linear"
        if "edge" not in embed_method:
            embed_method["edge"] = embed_method["node"]
        self.embed_method = embed_method
        self.use_input_embedding = use_input_embedding
        self.use_pe_embedding = use_pe_embedding
        self.pe_op = pe_op
        self.pe_hidden = pe_hidden
        self.use_scales = use_scales
        self.global_pool_methods = global_pool_methods
        self.use_global_pool_scales = use_global_pool_scales
        self.last_hidden_dims = last_hidden_dims
        self.last_simple = last_simple
        self.last_mlp = last_mlp
        if num_heads is not None and len(num_heads) == 1:
            num_heads = num_heads * (n_layers + 1) + [1]
        self.num_heads = num_heads
        if self.use_scales:
            self.scales = nn.Parameter(torch.empty(n_layers + 2))
            nn.init.ones_(self.scales)
        if use_global_pool_scales:
            num_gps = len(global_pool_methods)
            self.global_pool_scales = nn.Parameter(torch.empty(num_gps))
            nn.init.ones_(self.global_pool_scales)

    def get_embed(self, in_dim, out_dim, feat="node", bias=True, is_input=True):
        embed_method = self.embed_method.get(feat, "linear")
        embed = (
            get_input_embedding(in_dim, out_dim, embed_method, feat == "node", bias)
            if is_input
            else nn.Linear(in_dim, out_dim, bias=bias)
        )
        return embed, out_dim

    def get_input_dims_after_encoder(self):
        g_dim, n_dim, e_dim = self.input_dims[:3]
        g_hidden, n_hidden, e_hidden = self.hidden_dims[:3]

        # Positional Encoding
        pe_dim = self.pe_dim
        pe_hidden = self.pe_hidden
        if pe_hidden is None:
            pe_hidden = pe_dim if self.pe_op == "fill" else n_hidden
        if self.use_pe_embedding and pe_dim > 0:
            self.pe_embed, pe_dim = self.get_embed(pe_dim, pe_hidden, is_input=False)
        feat_hidden = n_hidden - pe_dim if self.pe_op == "fill" else n_hidden
        if feat_hidden <= 0:
            raise ValueError("not enough embedding dismension leaved for feature")

        if self.use_input_embedding:
            self.node_embed, n_dim = self.get_embed(n_dim, feat_hidden)
            if e_dim > 0 and e_hidden > 0:
                self.edge_embed, e_dim = self.get_embed(e_dim, e_hidden, feat="edge")

        if self.pe_op in ["cat", "fill"]:
            n_dim += pe_dim
        elif self.pe_op == "add":
            if pe_dim != n_dim:
                raise ValueError(
                    "pe_dim must be equal to node_feat_dim when pe_op is add."
                )
        # else: do nothing
        self.pos_enc_dim = pe_dim
        return [g_dim, n_dim, e_dim]

    def get_inputs(self, graph: DGLGraph):
        input_e_dim = self.input_dims[2]
        gfeat = None
        nfeat = graph.ndata["feat"]
        efeat = graph.edata.get("feat", None) if input_e_dim > 0 else None
        if self.use_input_embedding:
            if self.embed_method.get("node", "linear") != "linear":
                nfeat = nfeat.to(torch.int64)
            nfeat = self.node_embed(nfeat)
            if efeat is not None:
                if self.embed_method.get("edge", "linear") != "linear":
                    efeat = efeat.to(torch.int64)
                efeat = self.edge_embed(efeat)

        pe = graph.ndata.get("pe", None)
        if pe is not None:
            if self.use_pe_embedding:
                pe = self.pe_embed(pe)
            if self.pe_op in ["cat", "fill"]:
                nfeat = torch.cat([nfeat, pe], dim=-1)
            elif self.pe_op == "add":
                nfeat = nfeat + pe
        self.pos_enc = pe
        return MultiFeatures([gfeat, nfeat, efeat])

    def get_conv(self, ind, input_dims, output_dims):
        build_conv = self.build_conv
        if self.num_heads is not None:
            build_conv = partial(build_conv, num_heads=self.num_heads[ind])
        if self.allow_zero_in_degree:
            build_conv = partial(build_conv, allow_zero_in_degree=True)
        return build_conv(input_dims=input_dims, output_dims=output_dims)

    def get_conv_block(self, conv, use_dropout=True):
        output_dims = conv.get_output_dims()
        # n_dims = [output_dims[1]]
        norms = nn.ModuleList(
            [get_norm_layer(self.norm_layer, hid) for hid in output_dims]
        )
        drop = self.dropout if use_dropout else None
        return ConvBlock(conv, norms=norms, act=self.activation, dropout=drop)

    def apply_post_layer_oprs(self, features: MultiFeatures, ind: int):
        if self.use_scales:
            features = features * self.scales[ind]
        return features

    def get_mlp_layer(self, input_dim, output_dim, hidden_dims):
        norm_layer = None if self.last_simple else self.norm_layer
        dropout = None if self.last_simple else self.dropout
        return MLPLayer(
            input_dim,
            output_dim,
            hidden_dims,
            norm_layer=norm_layer,
            dropout=dropout,
            activation=self.activation,
        )

    def prepare_last_layer(self, current_dims):
        if self.task == "gpred":
            self.global_pools = nn.ModuleList()
            for gp in self.global_pool_methods:
                self.global_pools.append(get_global_pooling(gp, current_dims[1]))
            current_dim = current_dims[1] * len(self.global_pool_methods)
            self.last_layer = self.get_mlp_layer(
                current_dim, self.output_dims[0], self.last_hidden_dims
            )
        else:
            if self.last_mlp:
                self.last_layer = self.get_mlp_layer(
                    current_dims[1], self.output_dims[1], self.last_hidden_dims
                )
            else:
                self.last_layer = self.get_conv(-1, current_dims, self.output_dims)

    def apply_last_layer(self, graph: DGLGraph, features: MultiFeatures):
        if self.task == "gpred":
            nfeat = features.nodes_features
            x = [gp(graph, nfeat) for gp in self.global_pools]
            if self.use_global_pool_scales:
                x = [v * self.global_pool_scales[i] for i, v in enumerate(x)]
            x = torch.cat(x, dim=-1)
            logits = self.last_layer(x)
            logits = self.apply_post_layer_oprs(logits, ind=-1)
        else:
            if self.last_mlp:
                logits = self.last_layer(features.nodes_features)
            else:
                features = self.last_layer(graph, features)
                logits = features.nodes_features
        return logits

    def apply_global_pooling(self, graph: DGLGraph, features: MultiFeatures):
        if self.task == "gpred":
            nfeat = features.nodes_features
            x = [gp(graph, nfeat) for gp in self.global_pools]
            if self.use_global_pool_scales:
                x = [v * self.global_pool_scales[i] for i, v in enumerate(x)]
            x = torch.cat(x, dim=-1)
        else:
            return None
        return x

    @classmethod
    def register_model_args(cls, parser, prefix=None):
        super().register_model_args(parser, prefix=prefix)
        cls._add_argument(
            "n_layers",
            "-nl",
            type=int,
            default=2,
            help="Number of layers (do not include the first and last layer)",
        )
        cls._add_argument(
            "g_hidden",
            "-ghd",
            type=int,
            default=0,  # Disable by default
            help="Hidden dim for global feature",
        )
        cls._add_argument(
            "n_hidden",
            "-nhd",
            type=int,
            default=256,
            help="Hidden dim for node feature",
        )
        cls._add_argument(
            "e_hidden",
            "-ehd",
            type=int,
            default=None,  # Same as n_hidden
            help="Hidden dim for edge feature",
        )
        cls._add_argument(
            "norm_layer",
            "-nm",
            choices=["batch", "layer", "instance", "none"],
            default="none",  # recommend to use layer norm, or instance norm.
            help="The norm layer to use",
        )
        cls._add_argument(
            "activation", "-act", type=str, default="relu", help="Activation"
        )
        cls._add_argument("dropout", "-drop", type=float, default=0.5, help="Drop rate")
        cls._add_argument(
            "allow_zero_in_degree",
            "-az",
            action="store_true",
            help="Allow zero in degree",
        )
        cls._add_argument(
            "global_pool_methods",
            "-gp",
            type=str,
            default=["mean"],
            nargs="+",
            choices=["mean", "max", "sum", "att"],
            help="Global pooling operation",
        )
        cls._add_argument(
            "use_global_pool_scales",
            "-gs",
            action="store_true",
            help="Enable the scales after global pooling",
        )
        cls._add_argument(
            "last_hidden_dims",
            "-lhd",
            type=int,
            default=[],
            nargs="*",
            help="The hidden dims for the last mlp after global pooling",
        )
        cls._add_argument(
            "last_simple",
            "-lsimple",
            action="store_true",
            help="Use simple mlp for the last layer (no dropout and norm)",
        )
        cls._add_argument(
            "last_mlp",
            "-lmlp",
            action="store_true",
            help="Use MLP for last layer in node/edge prediction task",
        )
        cls._add_argument(
            "use_input_embedding",
            "-embed",
            action="store_true",
            help="Use input embedding at the begining",
        )
        cls._add_argument(
            "use_pe_embedding",
            "-pemb",
            action="store_true",
            help="Use embedding for positional encoding",
        )
        cls._add_argument(
            "pe_op",
            "-peop",
            type=str,
            default="cat",
            choices=["add", "cat", "fill", "none"],
            help="positional encoding op",
        )
        cls._add_argument(
            "pe_hidden",
            "-pehd",
            type=int,
            default=None,
            help="the ratio of positional encoding for the whole embedding dim when pe_op is cat",
        )
        cls._add_argument(
            "use_scales",
            "-scale",
            action="store_true",
            help="Use scale layer after each layer",
        )
        cls._add_argument(
            "num_heads",
            "-nhead",
            type=int,
            nargs="+",
            default=None,
            help="Attention heads, should be of length: 1 or n_layers + 2",
        )
