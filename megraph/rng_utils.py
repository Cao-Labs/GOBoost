#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : rng_utils.py
# Author : Honghua Dong
# Email  : dhh19951@gmail.com
#
# Distributed under terms of the MIT license.

from typing import Dict, List, Union

import numpy.random as random


def sample_between_min_max(min_max: List[Union[int, float]]) -> Union[int, float]:
    """Sample a number within [min, max]."""
    mi, ma = min_max
    if type(mi) is int:
        return random.randint(mi, ma + 1)
    return random.rand() * (ma - mi) + mi


def sample_partition(n: int, m: int, method: str = "sep") -> List[int]:
    """Sample a partition of n objects into m parts."""
    if n < 0 or m <= 0:
        raise ValueError(f"No valid partition for {n} objects and {m} parts.")
    support_methods = ["sep", "iter"]
    if not (method in support_methods):
        raise ValueError(
            f"Invalid method {method}, only {support_methods} are supported."
        )
    if method == "sep":
        sep = [0, n]
        for i in range(m - 1):
            sep.append(sample_between_min_max([0, n]))
        sep = sorted(sep)
        return [sep[i + 1] - sep[i] for i in range(m)]
    else:
        parts = []
        for i in range(m):
            c = sample_between_min_max([0, n])
            n -= c
            parts.append(c)
        return parts


def sample_from_mixture(mix):
    return random.choice(list(mix.keys()), p=list(mix.values()))
