#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : mlp.py
# Author : Honghua Dong
# Email  : dhh19951@gmail.com
#
# Distributed under terms of the MIT license.

# Modified from https://github.com/vacancy/Jacinle/blob/master/jactorch/nn/cnn/layers.py

import torch.nn as nn
from megraph.torch_utils import get_activation, get_dropout, get_norm_layer


class LinearLayer(nn.Sequential):
    def __init__(
        self,
        in_features,
        out_features,
        norm_layer="none",
        dropout=None,
        bias=True,
        activation=None,
    ):
        no_norm_layer = norm_layer is None or norm_layer == "none"
        if bias is None:
            bias = no_norm_layer

        modules = [nn.Linear(in_features, out_features, bias=bias)]
        if not no_norm_layer:
            modules.append(get_norm_layer(norm_layer, out_features))
        if dropout is not None:
            modules.append(get_dropout(dropout))
        if activation is not None:
            modules.append(get_activation(activation))
        super().__init__(*modules)

        self.in_features = in_features
        self.out_features = out_features

    @property
    def input_dim(self):
        return self.in_features

    @property
    def output_dim(self):
        return self.out_features

    def reset_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                module.reset_parameters()


class MLPLayer(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        hidden_dims,
        norm_layer="none",
        dropout=None,
        activation="relu",
        flatten=True,
        last_activation=False,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims
        self.norm_layer = norm_layer
        self.dropout = dropout
        self.activation = activation
        self.flatten = flatten
        self.last_activation = last_activation

        if hidden_dims is None:
            hidden_dims = []
        elif type(hidden_dims) is int:
            hidden_dims = [hidden_dims]

        dims = [input_dim]
        dims.extend(hidden_dims)
        dims.append(output_dim)

        def build_linear_layer(input_dim, output_dim):
            return LinearLayer(
                input_dim,
                output_dim,
                norm_layer=self.norm_layer,
                dropout=self.dropout,
                activation=self.activation,
            )

        nr_hiddens = len(hidden_dims)  # len=1
        modules = [build_linear_layer(dims[i], dims[i + 1]) for i in range(nr_hiddens)]
        if self.last_activation:
            layer = build_linear_layer(dims[-2], dims[-1])
        else:
            layer = nn.Linear(dims[-2], dims[-1], bias=True)
        modules.append(layer)
        self.mlp = nn.Sequential(*modules)

    def reset_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                module.reset_parameters()

    def forward(self, input):
        if self.flatten:
            input = input.view(input.size(0), -1)
        return self.mlp(input)
