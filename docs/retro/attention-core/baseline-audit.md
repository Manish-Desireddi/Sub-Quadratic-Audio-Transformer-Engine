# Baseline Audit — attention-core
Baseline commit: `0d473ff` (worktree branch `branch/audio-transformer-subquadratic-4c0824`). Audited 2026-11-17. No code was changed.

## Environment caveats (affect how to read results)
- The editable install `subq_audio` points at the **main checkout** (`C:\Users\manis\Project\SubQuadraticNeuralNetwork`), which has 23 uncommitted modified files (709+/578-, incl. `autograd.py`). A plain `pytest` therefore tests uncommitted code, not HEAD.
- The worktree has no built `_C` extension and no C++ compiler on PATH (`cl`, `g++`, `ninja` absent), so **C++ tests (Catch2) and the extension-backed Python tests were not run** against HEAD.
- HEAD `autograd.py` was exercised by loading it directly with `_C` stubbed (scratchpad `runwt.py`).

## Test / CI snapshot
| Check | Result |
|---|---|
| `tests/test_autograd.py` on HEAD code (C ext stubbed) | 29 passed |
| Same suite via editable install (main checkout, uncommitted code) | `test_grad_vs_reference_autograd` FAILS: chunkwise forward differs from the loop reference by ~211 (max abs). HEAD code matches brute force to 1e-13, so the regression is in the uncommitted working copy, not HEAD |
| `test_api`, `test_model`, `test_equivalence` (main checkout) | 20 passed before the first failure (run was `-x`) |
| C++ `test_attention` | not run (no compiler). Only 3 cases; backward case is T=2, D=2 |
| CI `.github/workflows/tests.yml` | triggers on PRs to `main`; repo default branch is `master`. Runs ctest and `tests/benchmark_audio.py`; **never runs pytest** |
| TODO/FIXME/HACK in perimeter | 0 |

## Findings (file:line)
1. `src/kernels/attention_backward.cpp:249-289, 320-329` GPU backward kernel is a stub: `dQ=dO*phi'(Q)`, `dK=dO*phi'(K)`, `dV=phi(K)*dO`. It ignores state, decay and V, so it is not the gradient. It also overwrites `ctx.Q/K/V` with gradients.
2. `src/kernels/attention_backward.cpp:335-371` legacy CPU `run_attention_backward` reads `dO` from `ctx.O` and overwrites the Q/K/V inputs with gradients. Decay logit is reconstructed with clamps (`:348`), and `d_gamma` is discarded.
3. `src/kernels/attention_backward.cpp:50-53` aliasing guards are `assert`, compiled out in Release.
4. Forward carries input state `S` (`attention_forward.cpp:92`, `bindings.cpp:151-209` chunk-carries it), but backward assumes a zero initial state (`attention_backward.cpp:88`) and ignores `ctx.S`. Gradients are wrong whenever initial state is nonzero.
5. The "subnormal flush" threshold is `1e-15`, which is not subnormal (float subnormals are below ~1e-38). It silently zeroes legitimate values: `attention_forward.cpp:46,164`, `attention_backward.cpp:110,149`, `bindings.cpp:~330`.
6. The backward algorithm is copy-pasted a second time in `python/bindings.cpp:299-~440` (float-only, different clamping). Three forward/backward implementations now exist: C++ kernel, bindings copy, Python autograd.
7. `python/bindings.cpp:284-286` raw `new float[]` for dQ/dK/dV, freed only through later capsules; an exception between allocation and capsule creation leaks.
8. `include/kernels.hpp:46` declares `run_attention_forward_raw<T>`; no definition exists anywhere.
9. `src/kernels/attention_forward.cpp:157-163` `std::pow` evaluated inside the triple loop (D*D*C pow calls per chunk); `:108-109` heap alloc per chunk; `:8-9` relies on transitive `<vector>/<algorithm>`.
10. `src/kernels/feature_map.cpp:30-33` `int` index math overflows past 2^31 elements; same pattern `attention_forward.cpp:65-67`.
11. `attention_forward.cpp:12-61` GPU forward: `threads = d_model` (fails above 1024), no multi-head, relies on caller zeroing `S`; comment says BF16 kernel "not implemented" but `bindings.cpp:199` selects `PrecisionMode::BF16` and passes it through.
12. `autograd.py:46-49, 65-69` mask normalisation is heuristic: a float mask with any value < -1 is treated as additive, otherwise as 0/1. A 4D `[B,1,L,S]` mask is collapsed over the key axis, so **key-padding masks are silently ignored** (verified: keys 4-5 padded, every query reported valid).
13. `autograd.py:96-114, 262-280` decay shape is inferred from shape coincidence (`shape[0]==H` vs T; `shape[1]==T` vs H). Ambiguous when H==T.
14. `autograd.py:646-651` `SubQLinearAttention.forward` calls `chunkwise_flash_linear_attention` directly, not `SubQAttentionFunction`. The analytic O(T/C)-memory backward is therefore not used by the module; torch autograd taping is.
15. `autograd.py:~453-455` `SubQAttentionFunction` never accepts `past_state`, so cached/streaming use has no custom gradient path.
16. Tests: `tests/test_equivalence.py:33-39` compares C++ to reference only at `decay_factor=1.0`; the decayed C++ forward has no reference test. No test links C++ backward to the Python backward.
17. File headers read "All rights reserved" alongside MIT text (all files in perimeter).
18. Editable install points at another checkout (see caveats): dev-environment hazard that already produced a misleading result during this audit.
