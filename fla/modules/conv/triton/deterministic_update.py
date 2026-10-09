# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import torch
import triton
import triton.language as tl

from fla.utils import input_guard


@triton.heuristics({
    'HAS_BIAS': lambda args: args['bias'] is not None,
    'HAS_RESIDUAL': lambda args: args['residual'] is not None,
    'HAS_CACHE_INDICES': lambda args: args['cache_indices'] is not None,
})
@triton.jit
def deterministic_causal_conv1d_update_kernel(
    x,
    cache,
    cache_indices,
    residual,
    weight,
    bias,
    y,
    stride_x_n: tl.constexpr,
    stride_x_d: tl.constexpr,
    stride_y_n: tl.constexpr,
    stride_y_d: tl.constexpr,
    D: tl.constexpr,
    W: tl.constexpr,
    NUM_SLOTS: tl.constexpr,
    BD: tl.constexpr,
    COMPACT_CACHE: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    HAS_RESIDUAL: tl.constexpr,
    HAS_CACHE_INDICES: tl.constexpr,
    ACTIVATION: tl.constexpr,
):
    i_d = tl.program_id(0).to(tl.int64)
    i_n = tl.program_id(1).to(tl.int64)
    i_cache = tl.load(cache_indices + i_n).to(tl.int64) if HAS_CACHE_INDICES else i_n
    valid_cache = (i_cache >= 0) & (i_cache < NUM_SLOTS)
    i_cache = tl.where(valid_cache, i_cache, 0)
    o_d = i_d * BD + tl.arange(0, BD)
    mask_d = o_d < D
    b_x = tl.load(x + i_n * stride_x_n + o_d * stride_x_d, mask=mask_d, other=0).to(tl.float32)
    b_y = tl.zeros((BD,), dtype=tl.float32)

    for j in tl.static_range(0, W):
        if j < W - 1:
            if COMPACT_CACHE:
                p_state = cache + i_cache * (W - 1) * D + j * D + o_d
            else:
                p_state = cache + i_cache * D * W + o_d * W + j + 1
            b_state = tl.load(p_state, mask=valid_cache & mask_d, other=0).to(tl.float32)
        else:
            b_state = b_x

        b_weight = tl.load(weight + o_d * W + j, mask=mask_d, other=0).to(tl.float32)
        b_y += b_state * b_weight

        if COMPACT_CACHE:
            if j > 0:
                p_write = cache + i_cache * (W - 1) * D + (j - 1) * D + o_d
                tl.store(
                    p_write,
                    tl.cast(b_state, dtype=cache.dtype.element_ty, fp_downcast_rounding='rtne'),
                    mask=valid_cache & mask_d,
                )
        else:
            p_write = cache + i_cache * D * W + o_d * W + j
            tl.store(
                p_write,
                tl.cast(b_state, dtype=cache.dtype.element_ty, fp_downcast_rounding='rtne'),
                mask=valid_cache & mask_d,
            )

    if HAS_BIAS:
        b_y += tl.load(bias + o_d, mask=mask_d, other=0).to(tl.float32)
    if ACTIVATION == 'swish' or ACTIVATION == 'silu':
        b_y *= tl.sigmoid(b_y)
    if HAS_RESIDUAL:
        b_y += tl.load(residual + i_n * D + o_d, mask=mask_d, other=0).to(tl.float32)
    tl.store(
        y + i_n * stride_y_n + o_d * stride_y_d,
        tl.cast(b_y, dtype=y.dtype.element_ty, fp_downcast_rounding='rtne'),
        mask=mask_d,
    )


@input_guard(no_guard_contiguous=['x'])
def deterministic_causal_conv1d_update(
    x: torch.Tensor,
    cache: torch.Tensor,
    cache_indices: torch.Tensor | None = None,
    residual: torch.Tensor | None = None,
    weight: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    activation: str | None = None,
    compact_cache: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    if x.ndim not in (2, 3) or weight is None:
        raise ValueError('Expected a rank-2 or rank-3 input and a convolution weight.')
    D, W = weight.shape
    if x.shape[-1] != D or W < 2:
        raise ValueError('Expected x[..., D] and weight[D, W] with W >= 2.')
    if x.ndim == 3 and x.shape[0] != 1 and x.shape[1] != 1:
        raise ValueError('Rank-3 decode input must have a singleton batch or time dimension.')
    N = x.numel() // D
    expected_tail = (W - 1, D) if compact_cache else (D, W)
    if cache.ndim != 3 or tuple(cache.shape[1:]) != expected_tail or not cache.is_contiguous():
        raise ValueError(f'Expected contiguous cache[slots, {expected_tail[0]}, {expected_tail[1]}].')
    if cache_indices is not None and (cache_indices.ndim != 1 or cache_indices.numel() != N):
        raise ValueError(f'Expected {N} cache indices.')

    if x.ndim == 2:
        stride_x_n, stride_x_d = x.stride()
    elif x.shape[0] == 1:
        stride_x_n, stride_x_d = x.stride()[1:]
    else:
        stride_x_n, stride_x_d = x.stride()[0], x.stride()[2]
    y = torch.empty_like(x, memory_format=torch.contiguous_format)
    if y.ndim == 2:
        stride_y_n, stride_y_d = y.stride()
    elif y.shape[0] == 1:
        stride_y_n, stride_y_d = y.stride()[1:]
    else:
        stride_y_n, stride_y_d = y.stride()[0], y.stride()[2]

    BD = 64 if compact_cache else 8
    deterministic_causal_conv1d_update_kernel[(triton.cdiv(D, BD), N)](
        x=x,
        cache=cache,
        cache_indices=cache_indices,
        residual=residual,
        weight=weight,
        bias=bias,
        y=y,
        stride_x_n=stride_x_n,
        stride_x_d=stride_x_d,
        stride_y_n=stride_y_n,
        stride_y_d=stride_y_d,
        D=D,
        W=W,
        NUM_SLOTS=cache.shape[0],
        BD=BD,
        COMPACT_CACHE=compact_cache,
        ACTIVATION=activation,
        num_warps=4,
    )
    return y.view_as(x), cache
