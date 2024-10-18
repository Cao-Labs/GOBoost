#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : os_utils.py
# Author : Honghua Dong
# Email  : dhh19951@gmail.com
#
# Distributed under terms of the MIT license.

import importlib
import os
import os.path as osp
from importlib.machinery import SourceFileLoader

main_module_name = osp.basename(osp.dirname(__file__))


def get_module_name(file):
    if file.endswith(".py") and not file.startswith("_"):
        return file[: file.find(".py")]
    return None


def get_module_path(dir):
    basename = osp.basename(dir)
    if dir == "/":
        raise Exception("Failed to initialize module")
    if basename == main_module_name:
        return basename + "."
    else:
        return get_module_path(osp.dirname(dir)) + basename + "."


def import_dir_files(file):
    abs_dir = osp.dirname(file)
    module_path = get_module_path(abs_dir)
    for file in os.listdir(abs_dir):
        module_name = get_module_name(file)
        if module_name is not None:
            module = importlib.import_module(module_path + module_name)


def get_raw_cmdline():
    with open("/proc/self/cmdline") as f:
        x = f.readlines()
    if x is None or len(x) == 0:
        return None
    return x[0].replace("\x00", " ")


def found_config_file(filename):
    print(f"looking for config file: {filename}")
    if osp.exists(filename):
        print(f"Found config file: [{filename}]")
        return True
    return False


def read_config_file(filename, folder=None):
    if folder is not None:
        filename = osp.join(folder, filename)
    if found_config_file(filename):
        return SourceFileLoader("config", filename).load_module().CONFIG
    return None


def merge_names(first, second, sep="_"):
    if second is not None:
        return first + sep + second
    return first


def get_default_config_filenames(model_name, conv_name, dataset_name, dataset_subname):
    """Get default config files with increasing priority."""
    dataset_fullname = merge_names(dataset_name, dataset_subname)
    model_fullname = merge_names(model_name, conv_name)

    names = [dataset_name, dataset_fullname, model_name, model_fullname]
    names.append(merge_names(model_name, dataset_name))
    names.append(merge_names(model_fullname, dataset_name))
    names.append(merge_names(model_name, dataset_fullname))
    names.append(merge_names(model_fullname, dataset_fullname))
    # Keep Unique
    on_list = set()
    filenames = []
    for name in names:
        if name not in on_list:
            on_list.add(name)
            filenames.append(f"cfg_{name}.py")
    return filenames


def get_default_config(args):
    dataset_name = args.dataset_name
    dataset_subname = args.dataset_subname
    model_name = args.model
    conv_name = args.layer

    # Config
    cfg_file = args.config_file
    if cfg_file is not None:
        config = read_config_file(cfg_file)
        if config is None:
            cfg_file = None
            print(
                f"[Warning] Could not found {cfg_file}, "
                "fall back to default config files."
            )
        else:
            config["config_file"] = cfg_file
    if cfg_file is None:
        cfg_files = get_default_config_filenames(
            model_name, conv_name, dataset_name, dataset_subname
        )
        config = {}
        found_files = []
        for f in cfg_files:
            new_config = read_config_file(f, folder=args.configs_dir)
            if new_config is not None:
                print(f"Overwrite default config using {f}:")
                print(new_config)
                config.update(new_config)
                found_files.append(f)
        config["config_file"] = found_files
    return config
