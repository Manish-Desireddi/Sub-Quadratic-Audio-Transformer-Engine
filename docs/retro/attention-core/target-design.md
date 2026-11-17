# Target Design: attention-core (P0/P1 rows)

## Target shape
1. **One backward core.** `subq::subquadratic_attention_backward<T>` is the only C++ backward. `bindings.cpp` and the legacy `run_attention_backward` call it with separate output buffers (G-004, G-006). It takes an optional initial state (G-003).
2. **GPU honesty first.** Until a real GPU adjoint scan exists, GPU backward raises an explicit unsupported error instead of returning wrong gradients (G-001). The real port is its own plan (effort L).
3. **Numerics.** Replace the 1e-15 "flush" with a true subnormal bound, or remove it (G-005). One shared helper for forward, backward and bindings.
4. **Masks.** `_normalize_attention_mask` rejects 4D masks that vary along the key axis instead of silently dropping them (G-002), then gets an explicit contract (G-010).
5. **Safety net before refactor (G-011, G-012).** Add C++-vs-Python decayed forward and gradient tests (T>64, D up to 64, nonzero initial state). Run pytest and ctest in CI on `master`.

## Fix sequence
G-012 (CI, so later rows are guarded), G-011 (failing tests first), G-002, G-005, G-003/G-004, G-006/G-007, G-001 (guard now, port later), then P2/P3 rows interleaved.

## What does not change
- Python `chunkwise_flash_linear_attention` forward and `SubQAttentionFunction` backward math (matches brute force to 1e-13; 29 tests pass at HEAD).
- Public API: `SubQEngine.forward/backward/*_raw`, `SubQLinearAttention`, decay as sigmoid(logit).
- ELU+1 feature map, default chunk size 64, boundary-state checkpoint scheme.

## Retro-ADRs to backfill (all `status: proposed (reconstructed)`; owner must confirm)
- A001 ELU+1 linear attention and sigmoid decay logit
- A002 dual C++/Python implementation and parity target
- A003 chunk size and checkpointing
- A004 bidirectional = forward + reverse - diagonal
- A005 padding freezes state
- A006 flush threshold
- A007 BF16 stance

Observed history and inferred intent stay separate; nothing is marked accepted without the owner.

## Approval needed before Phase 5
GPU backward behaviour change (error instead of wrong result), backward context API change, 4D mask rejection (behaviour change), CI trigger change.
