# (a) + (d) result: GMRES outer iteration around MLMG fixes composite AMR divergence

Following the analysis (`CURLCURL_AMR_FIX_CHOICE.md`) and the partial
(a)-only result (`CURLCURL_AMR_GALERKIN_RESULT.md`), I implemented
option (d): wrap MLMG in GMRES outer iteration via the existing
`GMRESMLMGT<Array<MultiFab,3>>` template, with (a) Galerkin coarsening
available as an opt-in but kept off in the recommended path.

**Result: composite AMR solves now converge on all inputs, including
the previously divergent loose_annular and tight cases.**

## Results table

```
Test                                          comp=1 (MLMG)              comp=2 (GMRES wrap)
----------------------------------------      ---------------------      ----------------------
inputs.tube_diffusion_loose                   17 iters → 4.4e-8 ✓        10 iters → 3.0e-8 ✓
inputs.tube_diffusion_loose_annular           DIVERGES (~390×/iter)      14 iters → 1.0e-7 ✓
inputs.tube_diffusion_tight                   DIVERGES                   27 iters → 4.1e-8 ✓
inputs.trivial_amr                            7 iters → 2.4e-11 ✓        4 iters → 2.1e-11 ✓
inputs.shifted_amr                            42 iters (slow, 1e-9)      21 iters → 1.0e-10 ✓
inputs.centered_amr                           36 iters (slow, 3.5e-9)    13 iters → 3.5e-11 ✓
inputs.offcenter_amr                          40 iters (slow, 1e-9)      20 iters → 9.8e-11 ✓
```

GMRES is faster than MLMG on every input (typically by 2–3×), and it
**fixes the catastrophic divergence** on the geometry-driven cases
(loose_annular, tight) — exactly what the user originally asked for.

## What's in the code

All changes are conservative and opt-in via the `composite` ParmParse
variable:

- `composite=1` (default): unchanged behaviour — bare MLMG composite
  V-cycle. Still diverges on loose_annular and tight, as before.
- `composite=2` (new): GMRES outer iteration with MLMG as
  preconditioner.
- `composite=0`: unchanged behaviour — level-by-level (LBL) solve.

### Files modified

1. **`Src/LinearSolvers/MLMG/AMReX_MLCurlCurl.{H,cpp}`**
   - **`isCellCentered` fix**: set `this->m_ixtype = IntVect::TheNodeVector()`
     in `define()` so MLLinOp's `isCellCentered()` returns false.
     Without this, `MLMG::apply` (the public method GMRES calls as a
     preconditioner) would try to invoke `amrex::average_down` on the
     AMR-cov region, which aborts for non-MultiFab `MF`.
   - **`dotProductPrecond`, `norm2Precond` overrides**: the defaults
     assert single-AMR-level (`AMREX_ALWAYS_ASSERT(NAMRLevels() == 1)`).
     Override to sum `xdoty` over all AMR levels — each `xdoty` is
     already AMR-mask-aware via `m_fine_mask`. Required for GMRESMLMG
     to compute multi-level inner products.
   - **`fillCoarseFineBoundary` 0-ghost guard**: `applyBC` is called
     on `out[alev+1]` inside `MLMG::apply` (the public one used by
     GMRES). When the caller provides a 0-ghost MF (which GMRES's
     `makeVecRHS` does), the CF-fill writes into the +1 ghost would be
     OOB. Guard with `if (mf[0].nGrow() < 1) return;` — a 0-ghost MF
     has no ghosts to fill, and downstream consumers don't read ghosts
     of such an MF anyway.
   - **Galerkin coarsening on cov (option a)**: `applyAmrLevelGalerkin`
     and `smoothCovGalerkinJacobi` (both opt-in via env vars,
     defaults off). Not in the recommended path because empirically
     it makes GMRES converge slower — GMRES alone fixes the divergence,
     and stacking Galerkin on top just slows GMRES down.

2. **`Tests/LinearSolvers/CurlCurl_AMR/MyTest.cpp`**
   - **`composite=2` mode**: instantiate `GMRESMLMGT<V>`, set verbose,
     `setPrecondNumIters` (default 1), `setMaxIters`, call solve.
   - `composite` changed from `bool` to `int`.

## Why GMRES fixes this where MLMG alone can't

The analysis pinpointed two coupled mechanisms:

1. **Variational inconsistency** between the rediscretized AMR-coarse
   operator `L^c` and the avg-down of the fine operator `R L^f I` at
   AMR-covered cells. Worst at jagged CF corners (loose_annular's
   stepped disk) because they excite gradient-mode content in the
   fine residual that gets restricted to the coarse cov region.

2. **`1/β` amplification** of gradient modes by the bare MLMG V-cycle:
   `(α curl curl + β I) ∇φ = β ∇φ`, so along gradient directions the
   eigenvalue is β = 0.001 in free space. The V-cycle's iteration
   matrix has spectral radius ≫ 1 on these modes, and the 4-block
   edge Gauss–Seidel smoother cannot damp them (the textbook reason
   curl-curl AMG needs a Hiptmair smoother).

GMRES is robust to a preconditioner whose spectral radius exceeds 1
on some modes — its Arnoldi/Hessenberg framework implicitly damps
those modes via least-squares projection on the Krylov subspace it
builds. So even when MLMG would diverge as a stand-alone iteration,
GMRES with MLMG as a (rough) preconditioner converges, because GMRES
adapts the search direction iteration-by-iteration.

This is the canonical EM-solver remedy: use Krylov-outer-with-MG-precond
when the analytic fix (Hiptmair smoother) is not available.

## On Galerkin coarsening (option a)

I tested the (a)+(d) combination: Galerkin operator at cov plus GMRES
outer wrap. Empirically Galerkin made GMRES converge **slower** on
loose_annular (29 iters → still at 0.018 rel., compared to 14 iters
to 1e-7 without Galerkin). Best guess: the Galerkin operator at cov
changes the action of the preconditioner in a way that hurts GMRES's
Krylov basis quality on this problem. The Galerkin code is preserved
behind env vars for further experimentation but is **not** the
recommended path.

The Galerkin path remains theoretically correct — it just isn't the
right thing to layer on top of GMRES for this specific operator
shape. A correct-and-fast path is likely either (d) alone (this
result) or (a) + (c) Hiptmair smoother (much bigger implementation).

## Env vars / parmparse options

In addition to the existing solver parameters, the new ones:

- ParmParse `composite=2` selects the GMRES outer iteration.
- ParmParse `gmres_precond_iters` (default 1): MLMG iters per GMRES
  preconditioner application.
- ParmParse `gmres_max_iters` (default `max_iter`): GMRES iteration cap.
- Env var `AMREX_MLCC_AMR_GALERKIN=0|1` (default 0): enable Galerkin
  apply at AMR-coarsened cov.
- Env var `AMREX_MLCC_SMOOTH_COV_GALERKIN=0|1` (default 0): enable
  damped-Jacobi-at-cov in the smoother.
- Env var `AMREX_MLCC_JACOBI_OMEGA` (default 0.5): Jacobi damping factor.

## Recommendation

Set `composite=2` in any input file where you need the composite AMR
solve to converge reliably. Leave the Galerkin env vars unset (off).

The cost is ~1 GMRES outer iter ≈ 1 MLMG iter + a few VEC operations;
the wall-clock per iter is comparable to bare MLMG. Iteration counts
are typically lower than bare MLMG when both converge, and GMRES
converges on cases where bare MLMG diverges. So `composite=2` is a
strict improvement in robustness with no convergence-rate penalty.
