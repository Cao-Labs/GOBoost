#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : features.py
# Author : Honghua Dong
# Email  : dhh19951@gmail.com
#
# Distributed under terms of the MIT license.

import copy

import torch.nn as nn
from megraph.torch_utils import apply_modules_on_list
from megraph.utils import (
    all_same_shape,
    apply_fn_on_list,
    residual_when_same_shape_on_list,
)


class MultiFeatures(object):
    """The features container for multi features on the graph.

    Note that the first dim of global features should be batch size."""

    def __init__(self, features):
        self._features = features

    def get_copy(self):
        return MultiFeatures(copy.copy(self._features))
        # 创建一个新的MultiFeatures对象，它的_features属性是当前对象_features属性的一个副本。

    @property  # 这是一个装饰器，它可以将一个方法转换为属性。
    def features(self):
        return self._features

    def set_features(self, features):
        self._features = features

    @property
    def edges_features(self):
        return self[2]

    def set_edges_features(self, value):
        self[2] = value
        return self

    def replace_edges_features(self, value):
        return self.get_copy().set_edges_features(value)

    @property
    def nodes_features(self):
        return self[1]

    def set_nodes_features(self, value):
        self[1] = value
        return self

    def replace_nodes_features(self, value):
        return self.get_copy().set_nodes_features(value)

    @property
    def global_features(self):
        return self[0]

    def set_global_features(self, value):
        self[0] = value
        return self

    def replace_global_features(self, value):
        return self.get_copy().set_global_features(value)

    def get_global_nodes_edges_features(self):
        return self._features[:3]

    def apply_fn(self, fn):
        if isinstance(fn, nn.ModuleList):
            self._features = apply_modules_on_list(fn, self._features)
        else:
            self._features = apply_fn_on_list(fn, self._features)
        return self

    def residual_when_same_shape(self, other, stem_beta=1.0, branch_beta=1.0):
        if branch_beta is not None:
            return MultiFeatures(
                residual_when_same_shape_on_list(
                    self, other, x_beta=stem_beta, y_beta=branch_beta
                )
            )
        # Otherwise replace with other
        return other

    def get_nary_features(self, index):
        if index >= len(self):
            return None
        return self[index]

    def __getitem__(self, index):
        if index >= len(self):
            raise StopIteration(f"Index {index} is out of range")
        return self._features[index]

    def __setitem__(self, index, value):
        if index >= len(self):
            raise ValueError(f"Index {index} is out of range")
        self._features[index] = value

    def __mul__(self, other):
        mul = lambda x: None if x is None else x * other
        return MultiFeatures(apply_fn_on_list(mul, self._features))

    def __add__(self, other):
        if not all_same_shape(self, other):
            raise ValueError("The shapes of two MultiFeatures are not all the same")
        return MultiFeatures(residual_when_same_shape_on_list(self, other))

    def __radd__(self, other: int):
        # Support sum([list of features])
        add = lambda x: None if x is None else x + other
        return MultiFeatures(apply_fn_on_list(add, self._features))

    def __len__(self):
        return len(self._features)

    def __str__(self):
        return "\n".join(
            [f"feature {i}: {f}" for i, f in enumerate(self._features) if f is not None]
        )
