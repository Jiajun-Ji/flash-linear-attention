# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import torch

from fla.modules.conv import (
    ImplicitLongConvolution,
    LongConvolution,
    PositionalEmbedding,
    ShortConvolution,
    causal_conv1d,
    fft_conv,
)
from fla.modules.conv.cp import CausalConv1dFunctionCP, causal_conv1d_cp
from fla.modules.conv.cuda import FastCausalConv1dFn, fast_causal_conv1d_fn
from fla.modules.conv.triton import (
    CausalConv1dFunction,
    causal_conv1d_bwd,
    causal_conv1d_fwd,
    causal_conv1d_update_states,
)
from fla.modules.conv.triton import (
    causal_conv1d_update as _causal_conv1d_update_default,
)
from fla.modules.conv.triton.deterministic_update import deterministic_causal_conv1d_update


def causal_conv1d_update(
    x: torch.Tensor,
    cache: torch.Tensor,
    residual: torch.Tensor | None = None,
    weight: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    activation: str | None = None,
    cache_indices: torch.Tensor | None = None,
    compact_cache: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    if cache_indices is not None or compact_cache:
        return deterministic_causal_conv1d_update(
            x=x,
            cache=cache,
            cache_indices=cache_indices,
            residual=residual,
            weight=weight,
            bias=bias,
            activation=activation,
            compact_cache=compact_cache,
        )
    return _causal_conv1d_update_default(
        x=x, cache=cache, residual=residual, weight=weight, bias=bias, activation=activation,
    )

__all__ = [
    'CausalConv1dFunction',
    'CausalConv1dFunctionCP',
    'FastCausalConv1dFn',
    'ImplicitLongConvolution',
    'LongConvolution',
    'PositionalEmbedding',
    'ShortConvolution',
    'causal_conv1d',
    'causal_conv1d_bwd',
    'causal_conv1d_cp',
    'causal_conv1d_fwd',
    'causal_conv1d_update',
    'causal_conv1d_update_states',
    'fast_causal_conv1d_fn',
    'fft_conv',
]
