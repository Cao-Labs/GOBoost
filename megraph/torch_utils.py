#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : torch_utils.py
# Author : Honghua Dong
# Email  : dhh19951@gmail.com
#
# Distributed under terms of the MIT license.

import random

import numpy as np
import torch
import torch.nn as nn
from jactorch.nn.quickaccess import get_activation as jac_get_act


class InstanceNorm1d(nn.InstanceNorm1d):
    def __init__(self, dim, **kwargs):
        super().__init__(dim, **kwargs)

    def forward(self, input):
        if input.dim() == 2:
            return super().forward(input.unsqueeze(0)).squeeze(0)
        return super().forward(input)


def set_global_seed(args):
    """Set the global seeds for reproducibility."""
    seed = args.seed
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available() and args.gpu_id >= 0:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            torch.cuda.manual_seed(seed)


def get_activation(act):
    """Get the activation module given name."""
    if isinstance(act, nn.Module):
        return act
    if type(act) is not str:
        raise ValueError("act should be a string or a nn.Module")
    act_lower = act.lower()
    if act_lower == "relu":
        return nn.ReLU()
    if act_lower == "elu":
        return nn.ELU()
    if act_lower == "softmax":
        return nn.Softmax(-1)
    return jac_get_act(act)


def get_dropout(dropout):
    """Get the dropout module."""
    if isinstance(dropout, nn.Module):
        return dropout
    return nn.Dropout(dropout)


def get_norm_layer(norm, dim):
    """Get the Normalization Layer, can be layer-norm, batch-norm, instance-norm."""
    # NOTE: Only support 1d norm, maybe support more than 1d in the future.
    if isinstance(norm, nn.Module):
        return norm
    if norm is None or norm == "none":
        return None
    if norm == "layer":
        return nn.LayerNorm(dim)
    if norm == "batch":
        return nn.BatchNorm1d(dim)
    if norm == "instance":
        return InstanceNorm1d(dim)
    raise ValueError("Invalid norm layer: {}".format(norm))


def get_num_params(model):
    """Calculate the number of params of a given model."""
    return np.sum([np.prod(list(param.data.size())) for param in model.parameters()])


def apply_modules_on_list(fn, a):
    """Apply modules on a list of tensors while allowing the elements to be None. 在允许元素为None的情况下，对张量列表应用模块。"""

    def get(f, x):
        if f is None:
            return x
        return None if x is None else f(x)

    if len(fn) != len(a):
        print(f'fn=[{fn}')
        print(f'a={a}')
        print(f'len(fn)={len(fn)}')
        print(f'len(a)={len(a)}')
        raise ValueError("modules and args should have the same length")
    return [get(f, x) for f, x in zip(fn, a)]


def apply_trans(trans, x):
    """Apply a transformation on x, returns None when either of them is None."""
    if (x is None) or (trans is None):
        return None
    return trans(x)


def sum_not_none_elements(a):
    """Sum over a list of tensors while ignoring the None elements."""
    a = [x for x in a if x is not None]
    return torch.sum(torch.stack(a, dim=0), dim=0) if len(a) else None


def tolist(x):
    """Convert a torch tensor to list."""
    if type(x) is torch.Tensor:
        return x.tolist()
    return x


def get_index_of_fused_groups(x_ns, y_ns, device):
    """Get the indexs of two fused groups."""
    x_ns = tolist(x_ns)
    y_ns = tolist(y_ns)
    if len(x_ns) != len(y_ns):
        raise ValueError("x_ns and y_ns should have the same length")
    x_idx, y_idx = [], []
    cur = 0
    for xn, yn in zip(x_ns, y_ns):
        x_idx.append(torch.arange(xn, device=device) + cur)
        # 创建一个从 0 到 xn 的整数序列，然后将 cur 加到这个序列的每个元素上，最后将结果添加到 x_idx 列表中。
        cur += xn
        y_idx.append(torch.arange(yn, device=device) + cur)
        cur += yn
    return torch.cat(x_idx), torch.cat(y_idx)
# cat将一个列表的张量连接成一个张量，也就是一个batch中所有图的所有节点都有对于的index
