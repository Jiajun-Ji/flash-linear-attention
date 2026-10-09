# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Framework-neutral input normalization for fused recurrent KDA."""

from __future__ import annotations

import torch


def prepare_fused_recurrent_kda_gate_beta(
    *,
    q: torch.Tensor,
    v: torch.Tensor,
    gate: torch.Tensor,
    beta: torch.Tensor,
    beta_is_raw: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Normalize raw SGL gate/beta layouts to the fused-recurrent contract.

    The raw-beta FP32 sigmoid and GVA H→HV expansion belong with the KDA
    operator contract, not an individual framework adapter.
    """

    _, total_tokens, num_q_heads, key_dim = q.shape
    num_v_heads = v.shape[2]
    if num_v_heads < num_q_heads or num_v_heads % num_q_heads:
        raise ValueError(f"Expected HV to be a multiple of H, got H={num_q_heads}, HV={num_v_heads}")
    if gate.ndim == 2 and tuple(gate.shape) == (total_tokens, num_q_heads * key_dim):
        gate = gate.reshape(1, total_tokens, num_q_heads, key_dim)
    elif gate.ndim == 3 and tuple(gate.shape) == (total_tokens, num_q_heads, key_dim):
        gate = gate.unsqueeze(0)
    if gate.ndim != 4 or gate.shape[:2] != (1, total_tokens):
        raise ValueError(f"Unsupported raw-gate shape {tuple(gate.shape)}")
    if beta_is_raw:
        beta = beta.float().sigmoid()
    if beta.ndim == 2 and beta.shape[0] == total_tokens:
        beta = beta.unsqueeze(0)
    if beta.ndim not in (3, 4) or beta.shape[:2] != (1, total_tokens):
        raise ValueError(f"Unsupported beta shape {tuple(beta.shape)}")
    group_size = num_v_heads // num_q_heads
    if gate.shape[2] == num_q_heads and group_size > 1:
        gate = gate.repeat_interleave(group_size, dim=2)
    if beta.shape[2] == num_q_heads and group_size > 1:
        beta = beta.repeat_interleave(group_size, dim=2)
    return gate, beta


__all__ = ["prepare_fused_recurrent_kda_gate_beta"]
