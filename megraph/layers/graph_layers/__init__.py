#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : __init__.py
# Author : Honghua Dong
# Email  : dhh19951@gmail.com
#
# Distributed under terms of the MIT license.

from megraph.io_utils import import_dir_files

from .base import BaseGraphLayer

__LAYER_DICT__ = {}


def register_layers_args(parser):
    parser.add_argument(
        "--layer",
        "-ly",
        type=str,
        default="gcn",
        choices=list(__LAYER_DICT__.keys()),
        help="layer to use",
    )


def layer_factory(name):
    if name is None:
        return None
    return __LAYER_DICT__[name]


def register_function(name):
    def register_function_fn(cls):
        if name in __LAYER_DICT__:
            raise ValueError(f"Name {name} already registered!")
        if not issubclass(cls, BaseGraphLayer):
            raise ValueError(f"Class {cls} is not a subclass of {BaseGraphLayer}")
        __LAYER_DICT__[name] = cls
        print(f"Layer registered: [{name}]")
        return cls

    return register_function_fn


import_dir_files(__file__)
