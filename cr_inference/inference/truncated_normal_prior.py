#!/usr/bin/env python3

# SPDX-License-Identifier: GPL-2.0+ OR BSD-2-Clause

from functools import partial
from typing import Callable, Optional

from jax import numpy as jnp
from jax.tree_util import Partial, tree_map

from nifty8.re.num.stats_distributions import _normal_to_standard, _standard_to_normal, _standard_to_uniform
from nifty8.re.model import WrappedCall
from nifty8.re.prior import _format_doc

exp = partial(tree_map, jnp.exp)
sqrt = partial(tree_map, jnp.sqrt)
log = partial(tree_map, jnp.log)
log1p = partial(tree_map, jnp.log1p)

def _uniform_to_standard(y, *, a_min, scale):
    from jax.scipy.stats import norm

    return tree_map(norm.ppf, (y - a_min) / scale)


def _standard_to_truncated_normal(xi, *, mean, std, a_min, a_max):
    from jax.scipy.stats import norm

    min_norm_cdf = norm.cdf((a_min - mean) / std)
    max_norm_cdf = norm.cdf((a_max - mean) / std)

    return _standard_to_normal(tree_map(norm.ppf, min_norm_cdf + _standard_to_uniform(xi, a_min=0, scale=1.0) * (max_norm_cdf - min_norm_cdf)), mean=mean, std=std)

def _truncated_normal_to_standard(y, *, mean, std, a_min, a_max):
    from jax.scipy.stats import norm

    min_norm_cdf = norm.cdf((a_min - mean) / std)
    max_norm_cdf = norm.cdf((a_max - mean) / std)

    return _uniform_to_standard(norm.cdf(_normal_to_standard(y, mean=mean, std=std)) - min_norm_cdf / (max_norm_cdf - min_norm_cdf), a_min=0, scale=1.0)

def truncated_normal_prior(mean, std, a_min, a_max) -> Partial:
    """Match standard normally distributed random variables to non-standard variables, truncated within a range."""
    return Partial(_standard_to_truncated_normal, mean=mean, std=std, a_min=a_min, a_max=a_max)

def truncated_normal_invprior(mean, std, a_min, a_max) -> Partial:
    """Match standard normally distributed random variables to non-standard variables, turncated within a range."""
    return Partial(_truncated_normal_to_standard, mean=mean, std=std, a_min=a_min, a_max=a_max)


class TruncatedNormalPrior(WrappedCall):
    @_format_doc
    def __init__(self, mean, std, a_min, a_max, **kwargs):
        """Transforms standard normally distributed random variables to a
        log-normal distribution.

        Parameters
        ----------
        mean : tree-like structure with arithmetics
            Mean of the log-normal distribution.
        std : tree-like structure with arithmetics
            Standard deviation of the log-normal distribution.
        a_min : tree-like structure with arithmetics
            Minimum value.
        a_max : tree-like structure with arithmetics
            Maximum value.
        {_doc_shared}
        """


        self.mean = mean
        self.std = std
        self.low = self.a_min = a_min
        self.high = self.a_max = a_max

        call = truncated_normal_prior(self.mean, self.std, self.a_min, self.a_max)
        super().__init__(call, white_init=True, **kwargs)
