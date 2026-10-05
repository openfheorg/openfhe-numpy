# Linear-transform optimization (issue #41): report

Branch `opt-lt` (off `dev` @ `1e2a822`), commits `38c7cd9` and `5e57876`.
All builds and measurements were done on ICE-Lake: 2× Xeon Platinum 8360Y, 72 cores, SMT off, 125 GB RAM.
OpenFHE C++ v1.6.0 and the `openfhe` 1.6.0 Python wheel were used unchanged; only openfhe-numpy changed.

## Summary

| Encrypted BERT-Tiny (secure profile) | Before | After |
|---|---|---|
| Forward pass | 1619.2 s | **409.5 s (3.95×)** |
| ct @ pt matmul (26 calls) | 696.6 s | **16.9 s (41×)** |
| ct @ ct matmul (8 calls) | 471.8 s | **28.2 s (17×)** |
| Bootstraps | 38 | **30** |
| RSS after matmul keygen | 79.6 GB | **62.5 GB** |
| RSS after forward pass | 85.5 GB | **76.0 GB** |
| Prediction / \|FHE − sim\| | positive / 5.4e-7 | positive / 2.6e-7 |

## Optimizations

1. **One linear-transform subroutine.** `EvalLinearTransform(ct, offsets, diagonals, algo)` in `numpy_lintrans.{h,cpp}` computes y = Σ diag_t ⊙ Rot(offset_t)(x). Sigma, Tau, Phi, Psi and transpose (square and rectangular) are now thin wrappers that build their diagonals and call it. Three strategies are available:
   - **NAIVE:** one `EvalRotate` plus one `EvalMult` per diagonal. This is the legacy behaviour.
   - **BSGS:** baby-step giant-step.
     - Every baby rotation is hoisted from one digit decomposition (`EvalFastRotationPrecompute` / `EvalFastRotation`).
     - Diagonals are pre-rotated in the clear.
     - Giant steps use Horner accumulation over a single stride, so only one giant key is needed.
   - **BSGS_DH:** double-hoisted BSGS, modeled on the CKKS-bootstrapping linear transforms (`SWITCHCKKSRNS::EvalLTWithPrecomputeSwitch`).
     - Baby rotations and plaintext products stay in the extended basis Q_l·P (`EvalFastRotationExt`, `EvalMultExt`).
     - Giant steps use `EvalHornerGiantRotate`.
     - A single `KeySwitchDown` runs at the end.
   - **Planner:** finds the arithmetic progression of the offsets and picks a baby/giant split. It uses only rotations whose keys already exist, and only when the split saves key switches; otherwise every rotation is hoisted in a single group. As a result, existing key-generation calls keep working unchanged.
   - **Diagonal encoding:** diagonals are encoded at the ciphertext's level, which avoids the legacy level-0 encode followed by an adjust step.
   - **Parallelism:** baby rotations and per-group products run in parallel under OpenMP.
2. **Direct ct @ pt and pt @ ct.**
   - X·W with a plaintext W is a single slot-linear map with 2d − 1 diagonals: Y[i][j] = Σ_δ X[i][j+δ]·W[j+δ][j]. It is evaluated as one BSGS transform that needs only existing SIGMA keys (pt @ ct needs only TAU keys).
   - It uses **1 multiplicative level instead of 3**. This is why BERT now needs 8 fewer bootstraps.
3. **Hoisted JKLS ct @ ct.**
   - Sigma(A) and Tau(B) are BSGS transforms.
   - All 2(d − 1) Phi rotations share one decomposition of Sigma(A), and all Psi rotations share one decomposition of Tau(B).
   - The k-loop runs in parallel with per-thread partial sums.
   - Products are accumulated with `EvalMultNoRelin`, followed by a single `Relinearize`.
4. **Canonical row shifts when batch = d².**
   - Rotations by d·k and d·k − d² are the same permutation of the packed data. Tau and Psi therefore merge their wrap and no-wrap masks onto the d·k rotation.
   - Key generation skips the d·k − d² keys. For d = 128 that is 509 → 383 rotation indices, about 16–17 GB less RSS.
   - Evaluation always uses d·k, which exists under both key sets, so old and new key sets stay interchangeable.
5. **Robustness.**
   - Errors inside OpenMP regions are captured and rethrown, so they reach Python as `RuntimeError` instead of aborting the process.
   - Rotation keys are checked serially before any parallel loop.
   - Non-relinearized (3-element) inputs are rejected, as `EvalRotate` already does.
   - The single-group path streams terms, so its memory use does not grow with the number of diagonals.

### Defaults and controls

| Control | Default | Notes |
|---|---|---|
| `SetLinTransAlgo` / `OPENFHE_NUMPY_LT_ALGO` = `naive\|bsgs\|dh\|auto` | `auto` → BSGS | BSGS_DH needs HYBRID and FLEXIBLEAUTO/FLEXIBLEAUTOEXT/FIXEDAUTO; otherwise it falls back to BSGS |
| `SetLinTransBabyStep` / `OPENFHE_NUMPY_LT_BSTEP` | next power of two ≥ max(√span, span/4) | 64 for 255 diagonals, 32 for Tau at d = 128 |
| `SetLinTransThreads` / `OPENFHE_NUMPY_LT_THREADS` | OpenMP maximum | upper bound for the outer parallel loops |
| `OPENFHE_NUMPY_LT_PAR_MIN` | 4 | smaller loops run sequentially, so OpenFHE's own per-tower threads stay active |
| `OPENFHE_NUMPY_CANONICAL_KEYS=0` | off | restores the legacy d·k − d² keys |

New Python names: `LinTransAlgo`, `Set/GetLinTransAlgo`, `Set/GetLinTransBabyStep`, `Set/GetLinTransThreads`, `EvalLinearTransform`, `LinearTransformRotationIndices`. Existing signatures are unchanged. FIXEDMANUAL keeps the legacy JKLS matmul, so manual rescaling counts do not change.

## Micro-benchmark (BERT shapes)

Parameters:
- Ring 2^16, 16384 slots, depth 26, 50-bit scaling, 4 large digits, HYBRID, FLEXIBLEAUTO, sparse secret.
- 128 × 128 inputs encrypted at level 16 of 26, i.e. right after a bootstrap.
- Only the matmul keys are generated.
- Times are the minimum over 2–3 runs.

| Build / strategy | ct @ pt | ct @ ct | Levels (ct @ pt) | RSS of matmul keys |
|---|---|---|---|---|
| original wheel | 24.44 s | 48.52 s | 3 | 68.9 GB |
| opt-lt NAIVE | 24.65 s | 36.63 s | 3 | 52.7 GB |
| opt-lt BSGS, b = 16 / 32 / 64 | 1.00 / 0.69 / 0.60 s | 3.25 / 2.75 / 2.66 s | 1 | 52.7 GB |
| opt-lt BSGS_DH, b = 16 / 32 / 64 | 1.16 / 0.83 / 0.73 s | 3.65 / 3.12 / 3.04 s | 1 | 52.7 GB |
| **opt-lt default (BSGS, tuned b)** | **0.62 s (39×)** | **2.70 s (18×)** | 1 | 52.7 GB |

ct @ ct versus thread count, at the then-default baby step b = 16:

| Strategy | T = 16 | T = 36 | T = 72 |
|---|---|---|---|
| BSGS | 5.04 s | 3.70 s | 3.25 s |
| DH | 5.14 s | 4.02 s | 3.65 s |

T = 72 is fastest, so the default stays at the OpenMP maximum.

**Why BSGS beats BSGS_DH here.** Bootstrapping precomputes its diagonal plaintexts once. openfhe-numpy encodes them on every call, and encoding in the larger Q_l·P basis costs more than the ModDowns that double hoisting saves. BSGS_DH would pay off if diagonals were cached across calls, for example fixed weights reused over many inferences.

## End-to-end: enc-bert-tiny-onp

Command: `run_fhe.py --profile secure`, run from an unmodified copy of the app. VmRSS and VmHWM were sampled every 2 s.

| Operation | Before: count, time | After (BSGS): count, time | After (DH): count, time |
|---|---|---|---|
| matmul_ctpt | 26, 696.59 s | 26, 16.90 s | 26, — |
| matmul_ctct | 8, 471.78 s | 8, 28.16 s | 8, — |
| bootstrap | 38, 380.09 s | 30, 294.35 s | 30, 296.52 s |
| transpose (app's own `EvalRotate` code) | 2, 47.33 s | 2, 46.75 s | 2, 47.38 s |
| **forward pass** | **1619.2 s** | **409.5 s** | **421.6 s** |
| matmul keygen | 7.4 s | 4.7 s | 4.7 s |
| RSS after forward pass | 85.5 GB | 76.0 GB | 75.8 GB |
| peak VmHWM (2 s sampling) | not sampled | 72.4 GiB | 72.3 GiB |

The DH run's report was summarized only, so its matmul rows are not recorded.

Accuracy:
- Encrypted logits are [-1.78773823, 1.28152434]; the baseline gave [-1.78773774, 1.28152441].
- |FHE − exact| is 9.807e-5 (baseline 9.800e-5).
- Per-stage encrypted-vs-simulator error is ≤ 1.3e-5, the same magnitude as the baseline.

Bootstrapping is now 73% of the run. The next-largest cost is the app's own composed transpose (11.6%). It calls `EvalRotate` directly, so this change does not affect it; `onp.transpose` itself is now BSGS-based.

ICE-Lake is a shared host: the load average was about 15 when the default run started.

## Validation

- **openfhe-numpy test suite** (`tests/run_tests.py`):

  | Build | Passed | Wall time |
  |---|---|---|
  | `dev` baseline | 61/61 | 473 s |
  | opt-lt NAIVE | 61/61 | 474 s |
  | opt-lt BSGS | 61/61 | 453 s |
  | opt-lt DH | 61/61 | 454 s |

- **Examples:** 19/23 Python examples and both C++ demos pass under every strategy. The other four fail identically on unmodified `dev`, so they were already broken:
  - `simple_matrix_vector_product`
  - `block_tensor/simple_block_matrix_vector_product`
  - `simple_slicing`
  - `simple_vector_operations`
- **Smoke test against numpy:** 264 checks, all passing. It covers:
  - NAIVE, BSGS and DH.
  - Batch sizes S = d² < N/2, S = 2d², S = 4d² and S = d² = N/2.
  - Scaling: FLEXIBLEAUTO, FLEXIBLEAUTOEXT, FIXEDAUTO, FIXEDMANUAL.
  - Key switching: HYBRID and BV.
  - Operations: Sigma, Tau, Phi, Psi, transpose, generic linear transform, ct @ ct, ct @ pt, pt @ ct.
  - For the linear-transform operations, level, noise-scale degree and scaling factor are identical to NAIVE.
- **Robustness checks:** missing keys raise errors, 3-element inputs are rejected, short diagonals work, and rectangular transpose is correct.
- **Review:** an adversarial review in two lenses, (1) correctness and (2) concurrency, memory and ABI. Its findings are fixed in `38c7cd9`.

## Behaviour changes to note

- ct @ pt and pt @ ct consume 1 level instead of 3. This applies under BSGS and DH; NAIVE and FIXEDMANUAL are unchanged.
- When batch = d², `EvalSquareMatMultRotateKeyGen` and `EvalLinTransKeyGen(TAU/PSI)` generate fewer keys. Code that rotates by d·k − d² using those keys must either use d·k or set `OPENFHE_NUMPY_CANONICAL_KEYS=0`.
- ct @ ct holds up to about 1.3 GB of extra transient memory at BERT sizes, from its parallel k-loop.
