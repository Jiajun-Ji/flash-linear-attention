# Deterministic KDA and Conv kernels

This release adds deterministic recurrent KDA and causal Conv decode kernels on upstream FLA `37a6b1c6290e5240f6f0d80419d08a7aac27e548`. Ordinary inference calls retain the upstream default kernels.

| Entry point | Input/state contract | Output |
| --- | --- | --- |
| `fla.ops.kda.deterministic_fused_recurrent_kda` | KDA `q, k, v, g, beta`; optional canonical `[N, HV, K, V]` FP32 state, or a V-first `[slots, HV, V, K]` pool with `ssm_state_indices` and `vfirst_state_pool=True` | output plus final state; autograd supports canonical state only |
| `fla.ops.kda.fused_recurrent_kda` | Existing upstream API; training calls and explicit state-pool/`transpose_state_layout` calls route to the deterministic variant | same return structure |
| `fla.modules.convolution.causal_conv1d_update` | Existing upstream API; `cache_indices` or `compact_cache=True` selects fixed tap-order decode with `[slots, W-1, D]` compact or `[slots, D, W]` full cache | output and in-place cache |

The recurrent backward replays the state instead of saving `[T, K, V]` states and supports `K <= 256`. The V-first state-pool mode is inference-only. Negative or out-of-range state slots are masked. The Conv update processes taps in increasing order and masks invalid cache slots.

The deterministic path uses recurrent KDA. Chunk KDA retains its upstream implementation. Framework integrations are responsible for request and cache metadata.

Validation commands:

```bash
FLA_CI_ENV=0 pytest -q tests/ops/test_kda_deterministic.py tests/modules/test_deterministic_conv.py
FLA_CI_ENV=0 pytest -q tests/ops/test_kda.py tests/modules/test_conv.py
```

On an NVIDIA H20, the focused deterministic tests passed (7/7), as did the upstream fused-recurrent KDA cases (26/26) and Conv update cases (16/16). The full `tests/ops/test_kda.py` run was interrupted during a long-running chunk-path case, so it is not recorded as a full-suite pass. No performance claim is attached to this branch.
