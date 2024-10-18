#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : utils.py
# Author : Honghua Dong
# Email  : dhh19951@gmail.com
#
# Distributed under terms of the MIT license.

import os.path as osp
import time

from tqdm import tqdm

from .logger import configure, dump_params, logger

MINIMAL_TQDM_LEN = 100


def get_localtime_str():
    return time.strftime("%Y_%m_%d__%H_%M_%S", time.localtime())


def my_tqdm(a, *args, **kwargs):
    if len(a) < MINIMAL_TQDM_LEN:
        return a
    return tqdm(a, *args, **kwargs)


def check_len(x, n):
    if len(x) != n:
        raise ValueError(f"Expect {n} elements, but got {len(x)}")


def get_tuple_n(x, n, tp=int):
    if x is None:
        return None
    assert tp is not list
    if type(x) is tp:
        x = [x] * n
    if len(x) == 1 and type(x[0]) is tp:
        x = x * n
    if len(x) != n:
        raise ValueError(f"parameters should be {tp} or list of N elements")
    for i in x:
        if type(i) is not tp:
            raise ValueError(f"elements of list should be {tp}")
    return x


def all_same_shape(xs, ys):
    if len(xs) != len(ys):
        raise ValueError("Expect same length")
    for x, y in zip(xs, ys):
        if x is not None and y is not None and x.shape != y.shape:
            return False
    return True


def residual_when_same_shape(x, y, x_beta=1.0, y_beta=1.0):
    if y is None:
        return x
    if x is None:
        return y
    if x.shape == y.shape:
        return x * x_beta + y * y_beta
    # Otherwise replace with y
    return y


def residual_when_same_shape_on_list(xs, ys, x_beta=1.0, y_beta=1.0):
    return [residual_when_same_shape(x, y, x_beta, y_beta) for x, y in zip(xs, ys)]


def apply_fn_on_list(fn, a):
    """Apply fn on each element of a list"""
    if fn is None:
        return a

    def get(fn, x):
        return None if x is None else fn(x)

    return [get(fn, x) for x in a]


def register_logging_args(parser):
    logging = parser.add_argument_group("logging")
    logging.add_argument("--dump-dir", "-du", type=str, default="./results")
    logging.add_argument("--subdir", "-sub", type=str, default=None)
    logging.add_argument(
        "--exp-name", "-ename", type=str, default=None, help="exp name"
    )
    logging.add_argument(
        "--extra-name", "-ext", type=str, default=None, help="extra name"
    )
    logging.add_argument(
        "--use-tensorboard", "-tb", action="store_true", help="use tensorboard"
    )


def get_dataset_name(args):
    dataset_name = args.dataset_name
    if args.dataset_subname is not None:
        dataset_name += "_" + args.dataset_subname
    return dataset_name


def get_dump_dir(args):
    subdir = args.subdir
    if args.debug:
        subdir = "debug"
    dataset_name = get_dataset_name(args)
    if subdir is None:
        subdir = dataset_name
    dump_dir = osp.join(args.dump_dir, subdir)

    if getattr(args, "exp_name", None):
        exp_name = args.exp_name
    else:
        exp_name = f"{args.model}"
        if args.layer is not None:
            exp_name += f"_{args.layer}"
        if args.debug:
            exp_name += f"_{dataset_name}"
        if hasattr(args, "max_height"):
            exp_name += f"_h{args.max_height}"
        exp_name += f"_n{args.n_layers}"
        if getattr(args, "extra_name", None):
            exp_name += f"_{args.extra_name}"
        exp_name += f"_{args.local_time}"
        if args.seed is not None:
            exp_name += f"_seed{args.seed}"

    dump_dir = osp.expanduser(osp.join(dump_dir, exp_name))
    return dump_dir


def set_logger(args):
    args.local_time = get_localtime_str()
    dump_dir = get_dump_dir(args)
    format_strings = ["stdout", "log", "csv"]
    if args.use_tensorboard:
        format_strings.append("tensorboard")
    configure(logger, dump_dir, format_strings)
    dump_params(logger.get_dir(), vars(args))
    return dump_dir
