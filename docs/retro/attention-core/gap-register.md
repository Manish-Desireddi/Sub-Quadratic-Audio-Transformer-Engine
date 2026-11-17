# Gap Register: attention-core (Audio Transformer, sub-quadratic scaling)

Current phase: 5 (executing; G-012, G-011, G-002 done; G-005, G-003, G-004 awaiting CI)
Baseline: `0d473ff`. Artifacts: `baseline-audit.md`, `reconstructed-spec.md`, `target-design.md`.

## Perimeter
- Paths: `include/attention.hpp`, `include/kernels.hpp`, `src/kernels/{attention_forward,attention_backward,feature_map}.cpp`, `python/bindings.cpp`, `python/subq_audio/autograd.py`
- Anchor tests: `tests/test_attention.cpp`, `tests/test_equivalence.py`, `tests/test_autograd.py`, `tests/reference.py`
- Callers: `python/subq_audio/{model,nn,surgery}.py`, `src/engine.cpp`. Blast radius: every forward/backward, surgery, DDP training.
- Other sub-slices not started: B streaming-ingest, C weights/memory, D packaging/CI.

## Register
Triage: FF = fix-forward, RW = rewrite, LD = leave-documented (wont-fix).

| ID | Axis | Gap | Evidence | Forward fix | Sev | Eff | Triage | Status |
|----|------|-----|----------|-------------|-----|-----|--------|--------|
| G-001 | practice | GPU backward kernel is not the real gradient (ignores state, decay, V) and overwrites inputs | attention_backward.cpp:249-289,320-329 | Raise "unsupported" on GPU backward now; port the CPU adjoint scan as its own plan; write to separate buffers | P0 | L | RW | open |
| G-002 | practice | 4D key-padding masks silently ignored (collapsed over key axis) | autograd.py:65-69; probe: all 6 queries valid with keys 4-5 padded | Reject 4D masks that vary along the key axis, or derive key validity explicitly; add test | P1 | S | FF | done (fixed in _normalize_attention_mask; tests/test_mask_normalization.py; 35 pass on HEAD code) |
| G-003 | practice | Backward assumes zero initial state while forward accepts `S`; wrong gradients with nonzero state | attention_backward.cpp:88 vs attention_forward.cpp:92; bindings.cpp:151-209 | Accept initial state in backward ctx, or throw when nonzero; add test | P1 | M | FF | done in source (AttentionBackwardContext::S0, d_gamma t=0 term; Catch2 finite-difference test, T=70; awaiting CI) |
| G-004 | practice | Legacy `run_attention_backward` reads dO from `ctx.O` and overwrites Q/K/V; `d_gamma` dropped | attention_backward.cpp:335-371 | Route through `AttentionBackwardContext` with separate outputs; deprecate legacy entry | P1 | S | FF | done in source (run_attention_backward_ex, legacy wraps it, runtime alias check; Catch2 test; awaiting CI) |
| G-005 | practice | "Subnormal flush" at 1e-15 zeroes real values (not the subnormal range) | attention_forward.cpp:46,164; attention_backward.cpp:110,149; bindings.cpp:~330 | Use a true subnormal bound or drop it and rely on FTZ/DAZ; record in ADR | P1 | S | FF | done in source (helper subq::is_subnormal_or_zero in kernels.hpp, 8 sites; NOT compiled or run: no toolchain here; re-run tests/test_cpp_python_parity.py on a HEAD build) |
| G-006 | practice | Backward algorithm duplicated in bindings.cpp (3 implementations total) | bindings.cpp:299-~440 | Bindings call `subq::subquadratic_attention_backward`; delete copy; add cross-check vs Python backward | P2 | M | FF | open |
| G-007 | practice | Aliasing checks are `assert` only; raw `new[]` can leak on exception | attention_backward.cpp:50-53; bindings.cpp:284-286 | Runtime checks that throw; own buffers with `unique_ptr`/capsule before use | P2 | S | FF | open |
| G-008 | feature | C++ engine lacks multi-head, GQA, mask, bidirectional, per-head/gated decay, state input; Python has all | reconstructed-spec I6; attention.hpp:14-21 | Decide parity target (ADR A002); until then validate inputs and document the C++ subset | P2 | L | LD then FF | open |
| G-009 | practice | Module bypasses the analytic backward (O(T/C) memory claim only holds for `SubQAttentionFunction`); no custom grad for `past_state` | autograd.py:646-651, ~453 | Route module through the Function (add `past_state`) or correct the claims | P2 | M | FF | open |
| G-010 | thinking | Mask float-vs-additive heuristic and decay-shape inference by coincidence (H==T ambiguity) | autograd.py:46-49, 96-114, 262-280 | Explicit parameters or strict validation with errors | P2 | M | FF | open |
| G-011 | feature | No test ties decayed C++ forward or C++ backward to the Python reference; equivalence test only at decay=1.0 | test_equivalence.py:33-39; test_attention.cpp (T=2, D=2) | Add decayed-forward and gradient-vs-Python tests with T>64, D up to 64, nonzero initial state | P1 | M | FF | done (tests added: tests/test_cpp_python_parity.py, 11 pass vs installed binary; nonzero-initial-state case not reachable from Python API, still open in C++ test) |
| G-012 | feature | CI never runs pytest and targets `main` while the default branch is `master` | .github/workflows/tests.yml | Trigger on `master`; run pytest (CPU) plus ctest | P1 | S | FF | done (workflow edited, not yet run in CI) |
| G-013 | thinking | No ADRs: decay logit, chunk 64, dual implementation, bidirectional formula, flush threshold, BF16 stance all undocumented | `docs/adr/` absent | Backfill as `proposed (reconstructed)` | P2 | M | FF | open |
| G-014 | practice | `std::pow` in inner loops, per-chunk heap allocation, missing includes | attention_forward.cpp:108-109,157-163,8-9 | Precompute decay powers once per chunk; hoist buffers; add includes | P3 | S | FF | open |
| G-015 | practice | `int` index overflow in feature map and forward launch math | feature_map.cpp:30-33; attention_forward.cpp:65-67 | Use `size_t`; add large-shape guard | P3 | S | FF | open |
| G-016 | selection | `run_attention_forward_raw<T>` declared but never defined; BF16 selected but unimplemented in kernels | kernels.hpp:46; attention_forward.cpp:82-84; bindings.cpp:199 | Remove dead declaration; make BF16 an explicit error until implemented | P3 | S | FF | open |
| G-017 | practice | "All rights reserved" headers contradict MIT | headers in all perimeter files | Normalise to `SPDX-License-Identifier: MIT` | P3 | S | FF | open |
| G-018 | practice | Editable install points at another checkout, so tests can silently run uncommitted code | pip metadata; hit during this audit | Document per-worktree `pip install -e .` in CONTRIBUTING; CI installs from the checkout | P2 | S | FF | open |
| G-019 | practice | Uncommitted edits in the main checkout break `test_grad_vs_reference_autograd` (forward off by ~211); HEAD passes | main checkout `git diff` of `python/subq_audio/autograd.py` | Owner to review or discard those edits (outside this HEAD baseline) | P1 | S | LD | open |
