# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import pytest
import torch

from fla.modules.convolution import causal_conv1d_update
from fla.utils import assert_close, device


@pytest.mark.parametrize('compact_cache', [False, True])
@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
def test_deterministic_conv_update(compact_cache: bool, dtype: torch.dtype):
    torch.manual_seed(42)
    N, D, W = 3, 32, 4
    x = torch.randn((N, D), device=device, dtype=dtype)
    weight = torch.randn((D, W), device=device, dtype=dtype)
    bias = torch.randn((D,), device=device, dtype=dtype)
    slots = torch.tensor([2, -1, 4], device=device, dtype=torch.int32)
    cache_shape = (5, W - 1, D) if compact_cache else (5, D, W)
    cache = torch.randn(cache_shape, device=device, dtype=dtype)
    expected_cache = cache.clone()
    expected = torch.empty_like(x)
    for row, slot in enumerate(slots.tolist()):
        if slot >= 0:
            previous = expected_cache[slot].clone() if compact_cache else expected_cache[slot, :, 1:].T.contiguous()
        else:
            previous = torch.zeros((W - 1, D), device=device, dtype=dtype)
        state = torch.cat((previous, x[row:row + 1]), dim=0)
        result = torch.zeros((D,), device=device, dtype=torch.float32)
        for tap in range(W):
            result += state[tap].float() * weight[:, tap].float()
        expected[row] = (result + bias.float()).to(dtype)
        if slot >= 0:
            expected_cache[slot] = state[1:] if compact_cache else state.T

    actual, final_cache = causal_conv1d_update(
        x=x, cache=cache, cache_indices=slots, weight=weight, bias=bias, compact_cache=compact_cache,
    )
    assert final_cache is cache
    assert_close('output', expected, actual, 0.005)
    assert_close('state', expected_cache, cache, 0.005)
