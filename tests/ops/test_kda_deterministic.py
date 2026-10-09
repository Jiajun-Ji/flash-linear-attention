# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import pytest
import torch
import torch.nn.functional as F

from fla.ops.kda import deterministic_fused_recurrent_kda, fused_recurrent_kda
from fla.ops.kda.naive import naive_recurrent_kda
from fla.utils import assert_close, device


def test_deterministic_fused_recurrent_training():
    torch.manual_seed(42)
    shape = (1, 7, 2, 32)
    q = torch.randn(shape, device=device, requires_grad=True)
    k = torch.randn(shape, device=device, requires_grad=True)
    v = torch.randn(shape, device=device, requires_grad=True)
    g = F.logsigmoid(torch.randn(shape, device=device)).detach().requires_grad_(True)
    beta = torch.rand((1, 7, 2), device=device, requires_grad=True)
    h0 = torch.randn((1, 2, 32, 32), device=device, requires_grad=True)
    inputs = (q, k, v, g, beta, h0)

    actual, actual_state = fused_recurrent_kda(
        q=q, k=k, v=v, g=g, beta=beta, initial_state=h0, output_final_state=True,
    )
    expected, expected_state = naive_recurrent_kda(
        q=q, k=k, v=v, g=g, beta=beta, initial_state=h0, output_final_state=True,
    )
    assert_close('output', expected, actual, 0.005)
    assert_close('state', expected_state, actual_state, 0.005)

    actual_grads = torch.autograd.grad(actual.sum() + actual_state.sum(), inputs, retain_graph=True)
    expected_grads = torch.autograd.grad(expected.sum() + expected_state.sum(), inputs)
    for name, reference, result in zip(('q', 'k', 'v', 'g', 'beta', 'state'), expected_grads, actual_grads):
        assert_close(name, reference, result, 0.005)


def test_deterministic_fused_recurrent_raw_gate_backward():
    torch.manual_seed(42)
    shape = (1, 5, 2, 32)
    q = torch.randn(shape, device=device, requires_grad=True)
    k = torch.randn(shape, device=device, requires_grad=True)
    v = torch.randn(shape, device=device, requires_grad=True)
    raw_g = torch.randn(shape, device=device, requires_grad=True)
    raw_beta = torch.randn((1, 5, 2), device=device, requires_grad=True)
    A_log = torch.randn((2,), device=device, requires_grad=True)
    dt_bias = torch.randn((2, 32), device=device, requires_grad=True)
    h0 = torch.randn((1, 2, 32, 32), device=device, requires_grad=True)
    inputs = (q, k, v, raw_g, raw_beta, A_log, dt_bias, h0)

    actual, actual_state = fused_recurrent_kda(
        q=q, k=k, v=v, g=raw_g, beta=raw_beta, A_log=A_log, dt_bias=dt_bias,
        initial_state=h0, output_final_state=True, use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=True, use_beta_sigmoid_in_kernel=True, lower_bound=-5.0,
    )
    activated_g = -5.0 * torch.sigmoid(A_log.exp()[None, None, :, None] * (raw_g + dt_bias))
    expected, expected_state = naive_recurrent_kda(
        q=F.normalize(q, dim=-1), k=F.normalize(k, dim=-1), v=v, g=activated_g,
        beta=raw_beta.sigmoid(), initial_state=h0, output_final_state=True,
    )
    assert_close('output', expected, actual, 0.005)
    assert_close('state', expected_state, actual_state, 0.005)

    actual_grads = torch.autograd.grad(actual.sum() + actual_state.sum(), inputs, retain_graph=True)
    expected_grads = torch.autograd.grad(expected.sum() + expected_state.sum(), inputs)
    for name, reference, result in zip(
        ('q', 'k', 'v', 'g', 'beta', 'A_log', 'dt_bias', 'state'), expected_grads, actual_grads,
    ):
        assert_close(name, reference, result, 0.005)


def test_deterministic_fused_recurrent_vfirst_state_pool():
    torch.manual_seed(42)
    shape = (2, 1, 1, 32)
    q = torch.randn(shape, device=device)
    k = torch.randn(shape, device=device)
    v = torch.randn(shape, device=device)
    g = F.logsigmoid(torch.randn(shape, device=device))
    beta = torch.rand((2, 1, 1), device=device)
    slots = torch.tensor([2, -1], device=device, dtype=torch.int32)
    pool = torch.randn((4, 1, 32, 32), device=device)
    before = pool.clone()

    actual, final_pool = deterministic_fused_recurrent_kda(
        q=q, k=k, v=v, g=g, beta=beta, initial_state=pool,
        output_final_state=True, inplace_final_state=True,
        ssm_state_indices=slots, vfirst_state_pool=True,
    )
    first, first_state = naive_recurrent_kda(
        q=q[:1], k=k[:1], v=v[:1], g=g[:1], beta=beta[:1],
        initial_state=before[2:3].transpose(-1, -2).contiguous(), output_final_state=True,
    )
    second, _ = naive_recurrent_kda(q=q[1:], k=k[1:], v=v[1:], g=g[1:], beta=beta[1:])
    assert final_pool is pool
    assert_close('first output', first, actual[:1], 0.005)
    assert_close('masked output', second, actual[1:], 0.005)
    assert_close('first state', first_state.transpose(-1, -2), pool[2:3], 0.005)
    assert torch.equal(pool[:2], before[:2])
    assert torch.equal(pool[3:], before[3:])

    wrapped_pool = before.clone()
    with pytest.warns(DeprecationWarning, match='transpose_state_layout'):
        wrapped, final_wrapped_pool = fused_recurrent_kda(
            q=q, k=k, v=v, g=g, beta=beta, initial_state=wrapped_pool,
            output_final_state=True, inplace_final_state=True,
            ssm_state_indices=slots, transpose_state_layout=False, vfirst_state_pool=True,
        )
    assert final_wrapped_pool is wrapped_pool
    assert torch.equal(wrapped, actual)
    assert torch.equal(wrapped_pool, pool)
