# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Backward kernels for fused recurrent KDA.

The main kernel replays the recurrence instead of materializing every
``[K, V]`` state. Each program owns one sequence, value head and V tile and
runs three token sweeps:

1. reverse the state adjoint and produce ``dv``, ``dbeta`` and the update part
   of ``dk``;
2. replay states forward and produce ``dq``, the read part of ``dk`` and a
   local log-decay contribution;
3. scan those local contributions backwards to produce the log-decay gradient.

This keeps saved activation memory at O(T * (K + V)) rather than O(T * K * V).
"""

import torch
import triton
import triton.language as tl

from fla.modules.activations import sigmoid_bwd
from fla.modules.l2norm import l2norm_bwd, l2norm_fwd
from fla.ops.kda.gate import kda_gate_bwd
from fla.ops.utils.op import exp
from fla.ops.utils.softplus import softplus


@triton.heuristics(
    {
        "USE_INITIAL_STATE": lambda args: args["h0"] is not None,
        "STORE_INITIAL_STATE_GRADIENT": lambda args: args["dh0"] is not None,
        "USE_FINAL_STATE_GRADIENT": lambda args: args["dht"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
        "HAS_DT_BIAS": lambda args: args["dt_bias"] is not None,
        "USE_LOWER_BOUND": lambda args: args["lower_bound"] is not None,
    }
)
@triton.jit(do_not_specialize=["T"])
def fused_recurrent_kda_bwd_kernel(
    q,
    k,
    g,
    beta,
    A_log,
    dt_bias,
    residual,
    h0,
    do,
    dht,
    dq,
    dk,
    dv,
    dg,
    db,
    dh0,
    cu_seqlens,
    lower_bound,
    scale: tl.constexpr,
    T: tl.int64,
    B: tl.constexpr,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    IS_BETA_VECTOR: tl.constexpr,
    USE_QK_L2NORM_IN_KERNEL: tl.constexpr,
    USE_GATE_IN_KERNEL: tl.constexpr,
    APPLY_BETA_SIGMOID: tl.constexpr,
    ALLOW_NEG_EIGVAL: tl.constexpr,
    TRANSPOSE_STATE: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_INITIAL_STATE_GRADIENT: tl.constexpr,
    USE_FINAL_STATE_GRADIENT: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    HAS_DT_BIAS: tl.constexpr,
    USE_LOWER_BOUND: tl.constexpr,
):
    i_v = tl.program_id(0).to(tl.int64)
    i_nh = tl.program_id(1).to(tl.int64)
    i_n, i_hv = i_nh // HV, i_nh % HV
    i_h = i_hv // (HV // H)

    all_tokens = B * T
    if IS_VARLEN:
        bos = tl.load(cu_seqlens + i_n).to(tl.int64)
        eos = tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        seq_len = eos - bos
    else:
        bos = i_n * T
        seq_len = T

    o_k = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    m_k = o_k < K
    m_v = o_v < V
    if TRANSPOSE_STATE:
        m_h = m_v[:, None] & m_k[None, :]
        b_dht = tl.zeros([BV, BK], dtype=tl.float32)
    else:
        m_h = m_k[:, None] & m_v[None, :]
        b_dht = tl.zeros([BK, BV], dtype=tl.float32)

    if USE_FINAL_STATE_GRADIENT:
        if TRANSPOSE_STATE:
            p_dht = dht + (i_n * HV + i_hv) * K * V + o_v[:, None] * K + o_k[None, :]
        else:
            p_dht = dht + (i_n * HV + i_hv) * K * V + o_k[:, None] * V + o_v[None, :]
        b_dht += tl.load(p_dht, mask=m_h, other=0).to(tl.float32)

    # Empty varlen sequences still have the identity final-state path.
    if seq_len == 0:
        if STORE_INITIAL_STATE_GRADIENT:
            if TRANSPOSE_STATE:
                p_dh0 = dh0 + (i_n * HV + i_hv) * K * V + o_v[:, None] * K + o_k[None, :]
            else:
                p_dh0 = dh0 + (i_n * HV + i_hv) * K * V + o_k[:, None] * V + o_v[None, :]
            tl.store(p_dh0, b_dht, mask=m_h)
        return

    # Pass 1: reverse state-adjoint sweep. ``dk`` temporarily stores the
    # update contribution and ``dv`` stores d(residual).
    token = bos + seq_len - 1
    p_q = q + (token * H + i_h) * K + o_k
    p_k = k + (token * H + i_h) * K + o_k
    p_g = g + (token * HV + i_hv) * K + o_k
    p_r = residual + (token * HV + i_hv) * V + o_v
    p_do = do + (token * HV + i_hv) * V + o_v
    p_dv = dv + (token * HV + i_hv) * V + o_v
    p_dk = dk + ((i_v * all_tokens + token) * HV + i_hv) * K + o_k
    if IS_BETA_VECTOR:
        p_beta = beta + (token * HV + i_hv) * V + o_v
        p_db = db + (token * HV + i_hv) * V + o_v
    else:
        p_beta = beta + token * HV + i_hv
        p_db = db + (i_v * all_tokens + token) * HV + i_hv

    b_dh = b_dht
    for _ in range(0, seq_len):
        b_q = tl.load(p_q, mask=m_k, other=0).to(tl.float32)
        b_k = tl.load(p_k, mask=m_k, other=0).to(tl.float32)
        if USE_QK_L2NORM_IN_KERNEL:
            b_q *= tl.rsqrt(tl.sum(b_q * b_q) + 1e-6)
            b_k *= tl.rsqrt(tl.sum(b_k * b_k) + 1e-6)
        b_q *= scale

        b_log_decay = tl.load(p_g, mask=m_k, other=0).to(tl.float32)
        if USE_GATE_IN_KERNEL:
            b_A = tl.load(A_log + i_h).to(tl.float32)
            if HAS_DT_BIAS:
                b_log_decay += tl.load(dt_bias + i_h * K + o_k, mask=m_k, other=0).to(tl.float32)
            if USE_LOWER_BOUND:
                b_log_decay = lower_bound * tl.sigmoid(exp(b_A) * b_log_decay)
            else:
                b_log_decay = -exp(b_A) * softplus(b_log_decay)
        b_decay = exp(b_log_decay)

        b_r = tl.load(p_r, mask=m_v, other=0).to(tl.float32)
        b_do = tl.load(p_do, mask=m_v, other=0).to(tl.float32)
        if IS_BETA_VECTOR:
            b_beta = tl.load(p_beta, mask=m_v, other=0).to(tl.float32)
        else:
            b_beta = tl.load(p_beta).to(tl.float32)
        if APPLY_BETA_SIGMOID:
            b_beta = tl.sigmoid(b_beta)
            if ALLOW_NEG_EIGVAL:
                b_beta *= 2.0

        if TRANSPOSE_STATE:
            b_dh_total = b_dh + b_do[:, None] * b_q[None, :]
            b_du = tl.sum(b_dh_total * b_k[None, :], axis=1)
            b_u = b_beta * b_r
            b_dk_update = tl.sum(b_dh_total * b_u[:, None], axis=0)
            b_dr = b_beta * b_du
            b_dp = b_dh_total - b_dr[:, None] * b_k[None, :]
            b_dh = b_dp * b_decay[None, :]
        else:
            b_dh_total = b_dh + b_q[:, None] * b_do[None, :]
            b_du = tl.sum(b_dh_total * b_k[:, None], axis=0)
            b_u = b_beta * b_r
            b_dk_update = tl.sum(b_dh_total * b_u[None, :], axis=1)
            b_dr = b_beta * b_du
            b_dp = b_dh_total - b_k[:, None] * b_dr[None, :]
            b_dh = b_dp * b_decay[:, None]

        b_db = b_du * b_r
        tl.store(p_dk, b_dk_update, mask=m_k)
        tl.store(p_dv, b_dr, mask=m_v)
        if IS_BETA_VECTOR:
            tl.store(p_db, b_db, mask=m_v)
        else:
            tl.store(p_db, tl.sum(b_db))

        p_q -= H * K
        p_k -= H * K
        p_g -= HV * K
        p_r -= HV * V
        p_do -= HV * V
        p_dv -= HV * V
        p_dk -= HV * K
        p_beta -= HV * (V if IS_BETA_VECTOR else 1)
        p_db -= HV * (V if IS_BETA_VECTOR else 1)

    if STORE_INITIAL_STATE_GRADIENT:
        if TRANSPOSE_STATE:
            p_dh0 = dh0 + (i_n * HV + i_hv) * K * V + o_v[:, None] * K + o_k[None, :]
        else:
            p_dh0 = dh0 + (i_n * HV + i_hv) * K * V + o_k[:, None] * V + o_v[None, :]
        tl.store(p_dh0, b_dh, mask=m_h)

    tl.debug_barrier()

    # Pass 2: replay states forward. Complete dq/dk and store the local part
    # of d(log_decay). No per-token [K, V] state is materialized.
    if TRANSPOSE_STATE:
        b_h = tl.zeros([BV, BK], dtype=tl.float32)
    else:
        b_h = tl.zeros([BK, BV], dtype=tl.float32)
    if USE_INITIAL_STATE:
        if TRANSPOSE_STATE:
            p_h0 = h0 + (i_n * HV + i_hv) * K * V + o_v[:, None] * K + o_k[None, :]
        else:
            p_h0 = h0 + (i_n * HV + i_hv) * K * V + o_k[:, None] * V + o_v[None, :]
        b_h += tl.load(p_h0, mask=m_h, other=0).to(tl.float32)

    token = bos
    p_q = q + (token * H + i_h) * K + o_k
    p_k = k + (token * H + i_h) * K + o_k
    p_g = g + (token * HV + i_hv) * K + o_k
    p_r = residual + (token * HV + i_hv) * V + o_v
    p_do = do + (token * HV + i_hv) * V + o_v
    p_dv = dv + (token * HV + i_hv) * V + o_v
    p_dq = dq + ((i_v * all_tokens + token) * HV + i_hv) * K + o_k
    p_dk = dk + ((i_v * all_tokens + token) * HV + i_hv) * K + o_k
    p_dg = dg + ((i_v * all_tokens + token) * HV + i_hv) * K + o_k
    if IS_BETA_VECTOR:
        p_beta = beta + (token * HV + i_hv) * V + o_v
    else:
        p_beta = beta + token * HV + i_hv

    for _ in range(0, seq_len):
        b_q = tl.load(p_q, mask=m_k, other=0).to(tl.float32)
        b_k = tl.load(p_k, mask=m_k, other=0).to(tl.float32)
        if USE_QK_L2NORM_IN_KERNEL:
            b_q *= tl.rsqrt(tl.sum(b_q * b_q) + 1e-6)
            b_k *= tl.rsqrt(tl.sum(b_k * b_k) + 1e-6)

        b_log_decay = tl.load(p_g, mask=m_k, other=0).to(tl.float32)
        if USE_GATE_IN_KERNEL:
            b_A = tl.load(A_log + i_h).to(tl.float32)
            if HAS_DT_BIAS:
                b_log_decay += tl.load(dt_bias + i_h * K + o_k, mask=m_k, other=0).to(tl.float32)
            if USE_LOWER_BOUND:
                b_log_decay = lower_bound * tl.sigmoid(exp(b_A) * b_log_decay)
            else:
                b_log_decay = -exp(b_A) * softplus(b_log_decay)
        b_decay = exp(b_log_decay)

        b_r = tl.load(p_r, mask=m_v, other=0).to(tl.float32)
        b_do = tl.load(p_do, mask=m_v, other=0).to(tl.float32)
        b_dr = tl.load(p_dv, mask=m_v, other=0).to(tl.float32)
        b_dk_update = tl.load(p_dk, mask=m_k, other=0).to(tl.float32)
        if IS_BETA_VECTOR:
            b_beta = tl.load(p_beta, mask=m_v, other=0).to(tl.float32)
        else:
            b_beta = tl.load(p_beta).to(tl.float32)
        if APPLY_BETA_SIGMOID:
            b_beta = tl.sigmoid(b_beta)
            if ALLOW_NEG_EIGVAL:
                b_beta *= 2.0
        b_u = b_beta * b_r

        if TRANSPOSE_STATE:
            b_p = b_h * b_decay[None, :]
            b_h = b_p + b_u[:, None] * b_k[None, :]
            b_dq = tl.sum(b_h * b_do[:, None], axis=0) * scale
            b_dk_read = -tl.sum(b_p * b_dr[:, None], axis=0)
        else:
            b_p = b_h * b_decay[:, None]
            b_h = b_p + b_k[:, None] * b_u[None, :]
            b_dq = tl.sum(b_h * b_do[None, :], axis=1) * scale
            b_dk_read = -tl.sum(b_p * b_dr[None, :], axis=1)

        b_dk = b_dk_update + b_dk_read
        # q is normalized but not scaled here. Since b_dq includes scale,
        # q*dq is the invariant inner product for the actual forward query.
        b_gate_local = b_q * b_dq - b_k * b_dk_update + b_k * b_dk_read
        tl.store(p_dq, b_dq, mask=m_k)
        tl.store(p_dk, b_dk, mask=m_k)
        tl.store(p_dg, b_gate_local, mask=m_k)

        p_q += H * K
        p_k += H * K
        p_g += HV * K
        p_r += HV * V
        p_do += HV * V
        p_dv += HV * V
        p_dq += HV * K
        p_dk += HV * K
        p_dg += HV * K
        p_beta += HV * (V if IS_BETA_VECTOR else 1)

    if TRANSPOSE_STATE:
        b_gate_carry = tl.sum(b_dht * b_h, axis=0)
    else:
        b_gate_carry = tl.sum(b_dht * b_h, axis=1)

    tl.debug_barrier()

    # Pass 3: reverse suffix scan for d(log_decay).
    token = bos + seq_len - 1
    p_dg = dg + ((i_v * all_tokens + token) * HV + i_hv) * K + o_k
    for _ in range(0, seq_len):
        b_gate_carry += tl.load(p_dg, mask=m_k, other=0).to(tl.float32)
        tl.store(p_dg, b_gate_carry, mask=m_k)
        p_dg -= HV * K


def _validate_backward_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    residual: torch.Tensor,
    do: torch.Tensor,
    dht: torch.Tensor | None,
    initial_state: torch.Tensor | None,
    transpose_state_layout: bool,
    cu_seqlens: torch.LongTensor | None,
) -> tuple[int, int, int, int, int, int, int]:
    if q.ndim != 4:
        raise ValueError(f"Expected `q` to be rank 4, got shape {q.shape}.")
    B, T, H, K = q.shape
    if k.shape != q.shape:
        raise ValueError(f"Expected `k` to have shape {q.shape}, got {k.shape}.")
    if K > 256:
        raise ValueError(f"Fused recurrent KDA backward supports K <= 256, got K={K}.")
    if v.ndim != 4 or v.shape[:2] != (B, T):
        raise ValueError(f"Expected `v` to start with shape {(B, T)}, got {v.shape}.")
    HV, V = v.shape[2:]
    if H <= 0 or HV < H or HV % H != 0:
        raise ValueError(f"Expected HV to be a positive multiple of H, got H={H}, HV={HV}.")
    if g.shape != (B, T, HV, K):
        raise ValueError(f"Expected `g` to have shape {(B, T, HV, K)}, got {g.shape}.")
    scalar_beta_shape = (B, T, HV)
    vector_beta_shape = (B, T, HV, V)
    if beta.shape not in (scalar_beta_shape, vector_beta_shape):
        raise ValueError(
            f"Expected `beta` to have shape {scalar_beta_shape} or {vector_beta_shape}, got {beta.shape}."
        )
    if residual.shape != v.shape or residual.dtype != torch.float32:
        raise ValueError(f"Expected FP32 `residual` with shape {v.shape}, got {residual.shape} / {residual.dtype}.")
    if do.shape != v.shape:
        raise ValueError(f"Expected `do` to have shape {v.shape}, got {do.shape}.")

    if cu_seqlens is None:
        N = B
    else:
        if B != 1:
            raise ValueError(f"Varlen KDA requires B == 1, got B={B}.")
        if cu_seqlens.ndim != 1 or cu_seqlens.dtype not in (torch.int32, torch.int64):
            raise ValueError("`cu_seqlens` must be a rank-1 int32 or int64 tensor.")
        N = len(cu_seqlens) - 1
    state_shape = (N, HV, V, K) if transpose_state_layout else (N, HV, K, V)
    if initial_state is not None and initial_state.shape != state_shape:
        raise ValueError(f"Expected `initial_state` to have shape {state_shape}, got {initial_state.shape}.")
    if dht is not None and dht.shape != state_shape:
        raise ValueError(f"Expected `dht` to have shape {state_shape}, got {dht.shape}.")
    return B, T, H, HV, K, V, N


def _raw_gate_bwd(
    g: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor | None,
    dlog_decay: torch.Tensor,
    H: int,
    HV: int,
    K: int,
    lower_bound: float | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Apply the internal FLA raw-gate convention under GVA.

    The internal recurrent forward shares ``A_log`` and ``dt_bias`` across
    value heads that map to the same query/key head. ``kda_gate_bwd`` operates
    head-wise, so expand the small parameters to HV and reduce their gradients
    back to H after the Triton helper runs.
    """
    group_size = HV // H
    A_log_hv = A_log.repeat_interleave(group_size)
    dt_bias_hv = None
    if dt_bias is not None:
        dt_bias_hv = dt_bias.view(H, K).repeat_interleave(group_size, dim=0).reshape(HV * K)
    dg, dA_hv, dbias_hv = kda_gate_bwd(
        g=g,
        A_log=A_log_hv,
        dt_bias=dt_bias_hv,
        dyg=dlog_decay,
        lower_bound=lower_bound,
    )
    dA = dA_hv.view(H, group_size).sum(1).to(A_log)
    dbias = None
    if dbias_hv is not None:
        dbias = dbias_hv.view(H, group_size, K).sum(1).reshape_as(dt_bias).to(dt_bias)
    return dg, dA, dbias


@torch.compiler.disable
def fused_recurrent_kda_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    residual: torch.Tensor,
    do: torch.Tensor,
    dht: torch.Tensor | None = None,
    A_log: torch.Tensor | None = None,
    dt_bias: torch.Tensor | None = None,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    use_qk_l2norm_in_kernel: bool = False,
    use_gate_in_kernel: bool = False,
    use_beta_sigmoid_in_kernel: bool = False,
    allow_neg_eigval: bool = False,
    lower_bound: float | None = None,
    transpose_state_layout: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
]:
    """Run the complete three-pass Triton backward for recurrent KDA."""
    B, T, H, HV, K, V, N = _validate_backward_inputs(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        residual=residual,
        do=do,
        dht=dht,
        initial_state=initial_state,
        transpose_state_layout=transpose_state_layout,
        cu_seqlens=cu_seqlens,
    )
    if scale is None:
        scale = K ** -0.5
    if allow_neg_eigval and not use_beta_sigmoid_in_kernel:
        raise ValueError("`allow_neg_eigval=True` requires `use_beta_sigmoid_in_kernel=True`.")
    if use_gate_in_kernel:
        if A_log is None or A_log.numel() != H:
            raise ValueError(f"Raw gate activation requires `A_log` with {H} elements.")
        if dt_bias is not None and dt_bias.numel() != H * K:
            raise ValueError(f"Expected `dt_bias` to have {H * K} elements, got {dt_bias.numel()}.")

    if T == 0:
        dq = torch.zeros_like(q, dtype=torch.float32)
        dk = torch.zeros_like(k, dtype=torch.float32)
        dv = torch.zeros_like(v, dtype=torch.float32)
        dg = torch.zeros_like(g, dtype=torch.float32)
        db = torch.zeros_like(beta, dtype=torch.float32)
        dA = torch.zeros_like(A_log, dtype=torch.float32) if use_gate_in_kernel else None
        dbias = torch.zeros_like(dt_bias, dtype=torch.float32) if use_gate_in_kernel and dt_bias is not None else None
        dh0 = None
        if initial_state is not None:
            dh0 = torch.zeros_like(initial_state, dtype=torch.float32)
            if dht is not None:
                dh0.copy_(dht)
        return dq, dk, dv, dg, db, dA, dbias, dh0

    BK = triton.next_power_of_2(K)
    BV = 32
    NV = triton.cdiv(V, BV)
    all_tokens = B * T

    # V-tile partials. dq/dk reduce both V tiles and (for GVA) value heads;
    # dg only reduces V tiles because the decay is value-head-specific.
    dq_partial = q.new_empty(NV, all_tokens, HV, K, dtype=torch.float32)
    dk_partial = k.new_empty(NV, all_tokens, HV, K, dtype=torch.float32)
    dg_partial = g.new_empty(NV, all_tokens, HV, K, dtype=torch.float32)
    if beta.ndim == v.ndim:
        db_work = beta.new_empty(beta.shape, dtype=torch.float32)
    else:
        db_work = beta.new_empty(NV, all_tokens, HV, dtype=torch.float32)
    dv = v.new_empty(v.shape, dtype=torch.float32)
    dh0 = torch.empty_like(initial_state, dtype=torch.float32) if initial_state is not None else None

    grid = (NV, N * HV)
    fused_recurrent_kda_bwd_kernel[grid](
        q=q,
        k=k,
        g=g,
        beta=beta,
        A_log=A_log,
        dt_bias=dt_bias,
        residual=residual,
        h0=initial_state,
        do=do,
        dht=dht,
        dq=dq_partial,
        dk=dk_partial,
        dv=dv,
        dg=dg_partial,
        db=db_work,
        dh0=dh0,
        cu_seqlens=cu_seqlens,
        lower_bound=lower_bound,
        scale=scale,
        T=T,
        B=B,
        H=H,
        HV=HV,
        K=K,
        V=V,
        BK=BK,
        BV=BV,
        IS_BETA_VECTOR=beta.ndim == v.ndim,
        USE_QK_L2NORM_IN_KERNEL=use_qk_l2norm_in_kernel,
        USE_GATE_IN_KERNEL=use_gate_in_kernel,
        APPLY_BETA_SIGMOID=use_beta_sigmoid_in_kernel,
        ALLOW_NEG_EIGVAL=allow_neg_eigval,
        TRANSPOSE_STATE=transpose_state_layout,
        num_warps=4,
        num_stages=2,
    )

    dq_hv = dq_partial.sum(0).view(B, T, H, HV // H, K).sum(3)
    dk_hv = dk_partial.sum(0).view(B, T, H, HV // H, K).sum(3)
    dlog_decay = dg_partial.sum(0).view_as(g)
    db = db_work if beta.ndim == v.ndim else db_work.sum(0).view_as(beta)

    if use_qk_l2norm_in_kernel:
        q_norm, q_rstd = l2norm_fwd(q, output_dtype=torch.float32)
        k_norm, k_rstd = l2norm_fwd(k, output_dtype=torch.float32)
        dq = l2norm_bwd(q_norm, q_rstd, dq_hv)
        dk = l2norm_bwd(k_norm, k_rstd, dk_hv)
    else:
        dq, dk = dq_hv, dk_hv

    if use_gate_in_kernel:
        dg, dA, dbias = _raw_gate_bwd(
            g=g,
            A_log=A_log,
            dt_bias=dt_bias,
            dlog_decay=dlog_decay,
            H=H,
            HV=HV,
            K=K,
            lower_bound=lower_bound,
        )
    else:
        dg, dA, dbias = dlog_decay, None, None

    if use_beta_sigmoid_in_kernel:
        scale_beta = 2.0 if allow_neg_eigval else 1.0
        db = sigmoid_bwd(beta, db * scale_beta)

    return dq, dk, dv, dg, db, dA, dbias, dh0
