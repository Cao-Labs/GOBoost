#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : __init__.py
# Author : Honghua Dong
# Email  : dhh19951@gmail.com
#
# Distributed under terms of the MIT license.

__all__ = ["model_factory", "register_models_args"]

from megraph.io_utils import import_dir_files

from .model import GraphModel

__MODEL_DICT__ = dict()


def register_models_args(parser):
    parser.add_argument(
        "--model",
        "-md",
        type=str,
        default="megraph",
        choices=list(__MODEL_DICT__.keys()),
        help="model to use",
    )


def model_factory(name):
    return __MODEL_DICT__[name]


def register_function(name):
    def register_function_fn(cls):
        if name in __MODEL_DICT__:
            raise ValueError(f"Name {name} already registered!")
        if not issubclass(cls, GraphModel):
            raise ValueError(f"Class {cls} is not a subclass of {GraphModel}")
        __MODEL_DICT__[name] = cls
        print(f"Model registered: [{name}]")
        return cls

    return register_function_fn


import_dir_files(__file__)
