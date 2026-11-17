# Reconstructed Spec — attention-core (what the code does today)
Source: reading HEAD `0d473ff`; verified numerically where noted. Observed behaviour, not intent.

## Requirements
- R1 WHEN `chunkwise_flash_linear_attention` is called with `q,k,v` `[B,H,T,D]` (or 3D) and decay logit `w` THEN output `O_t = phi(q_t) S_t`, `S_t = gamma S_{t-1} + phi(k_t)^T v_t`, `gamma = sigmoid(w)` (verified vs brute force, 1e-13, chunk sizes 4/16/64).
- R2 WHEN `is_causal=False` THEN output = forward scan + reverse scan - diagonal term.
- R3 WHEN a mask is given THEN padded positions have `phi(k)=0`, `v=0`, `gamma=1` and output is zeroed; padding does not decay the state.
- R4 WHEN `past_state` is given THEN the scan starts from it and returns the final state (module path only).
- R5 `SubQAttentionFunction.backward` returns exact gradients for q, k, v, w, saving boundary states only (29 tests pass, incl. gradcheck, bidirectional, masked).
- R6 WHEN the C++ engine `forward` runs on `[B,T,D]` float32 THEN state is zero-initialised, carried across arena chunks, decay is a scalar `gamma` passed directly (not a logit).
- R7 C++ `backward` returns dQ, dK, dV (CPU: exact chunked adjoint scan, zero initial state) and optionally d_gamma/d_w.
- R8 GQA: K/V heads are repeat-interleaved to H heads in `SubQLinearAttention`.

## Invariants
- I1 `phi(x)=ELU(x)+1 > 0`; `phi'(x)=1 if x>0 else exp(x)`.
- I2 `0 < gamma < 1` in the module (sigmoid of a logit); C++ accepts any float `gamma` unchecked.
- I3 Chunk size does not change results (Python: tested; C++: fixed at 64, untested otherwise).
- I4 Memory: backward stores `(T/C+1)*D*D` boundary states plus `C*D*D` scratch.
- I5 Zero initial state is assumed by every backward implementation.
- I6 C++ engine API is 3D-only, single shared scalar decay, causal-only, no mask.

## Git archaeology (9 commits touch the perimeter, 19 repo-wide)
- Design moved under pressure in `866d5e3` (kernel fusion, HIP), `be46932` (BF16 abstraction), `17a5d4c` (mutex around FFI), `dc52a52` (GQA, padding, bidirectional), `0d473ff` (FP16 decay logit). Features landed in the Python path while the C++ kernels stayed 3D, scalar-decay and causal-only, so the two paths diverged.
- Churn hotspot: `tests/test_equivalence.py` (7 edits), still covering only `decay=1.0`.
- No reverts or dead branches. `master` is the only branch; CI targets `main`.

## Decision inventory (none has an ADR; `docs/adr/` does not exist)
| # | Decision | Where |
|---|---|---|
| D1 | ELU+1 feature map (not softmax or random features) | feature_map.cpp, autograd.py:15 |
| D2 | Decay = sigmoid of a learnable logit, per head, optionally input-gated | autograd.py:608-642 |
| D3 | Chunk size 64, boundary-state checkpointing for backward | attention_*.cpp, autograd.py |
| D4 | Dual implementation: C++ engine (inference/FFI) vs Python autograd (training) | repo layout |
| D5 | Bidirectional = forward + reverse - diagonal | autograd.py:233 |
| D6 | Padding freezes state (gamma=1, k=v=0) | autograd.py:116-122 |
| D7 | Subnormal flush at 1e-15 | kernels, bindings |
| D8 | FP32 state; BF16 only as an enum | bindings.cpp:199 |
