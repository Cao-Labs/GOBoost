#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : args.py
# Author : Honghua Dong
# Email  : dhh19951@gmail.com
#
# Distributed under terms of the MIT license.

import git

from .io_utils import get_default_config, get_raw_cmdline

__all__ = ["ArgsBuilder", "add_git_and_cmd_line_info", "get_args_and_model"]


class ArgsBuilder(object):
    """A meta-class to be inherit that support args register and setup from args"""

    __hyperparams__ = []
    __parser__ = None
    __prefix__ = "--"

    @classmethod
    def _set_parser_and_prefix(cls, parser, prefix):
        cls.__parser__ = parser
        if prefix is None:
            prefix = "--"
        else:
            prefix = f"--{prefix}-"
        cls.__prefix__ = prefix

    @classmethod
    def _add_argument(cls, name, *args, **kwargs):
        cls.__hyperparams__.append(name)
        name = name.replace("_", "-")
        cls.__parser__.add_argument(cls.__prefix__ + name, *args, **kwargs)

    @classmethod
    def from_args(cls, args, prefix=None, **kwargs):
        if prefix is None:
            prefix = ""
        else:
            prefix = str(prefix) + "_"
        print(f"From Args: {cls.__name__} with {kwargs}")
        init_params = {k: getattr(args, prefix + k) for k in cls.__hyperparams__}
        init_params.update(kwargs)
        return cls(**init_params)


def add_git_and_cmd_line_info(args):
    args.raw_cmdline = get_raw_cmdline()
    try:
        args.git_version = git.Repo().head.object.hexsha
    except Exception:
        print("No git info detected")
    return args


def get_args_and_model(parser, layer_factory, model_factory):
    args_, _ = parser.parse_known_args()
    graph_layer = layer_factory(args_.layer)
    if graph_layer is not None:
        graph_layer.register_layer_args(parser)
    graph_model = model_factory(args_.model)
    graph_model.register_model_args(parser)
    config = get_default_config(args_)
    parser.set_defaults(**config)
    args = parser.parse_args()
    args = add_git_and_cmd_line_info(args)
    return args, graph_layer, graph_model
