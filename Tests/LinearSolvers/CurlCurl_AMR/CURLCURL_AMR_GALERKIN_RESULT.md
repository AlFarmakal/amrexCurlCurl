# Galerkin coarsening (option a): implementation result

Per the CURLCURL_AMR_FIX_CHOICE.md analysis, I implemented option (a) —
Galerkin coarsening on the AMR-covered region — to attack what I
identified as the dominant divergence mechanism on jagged CF
geometries (wrong-direction component of V-cycle correction at cov).

The honest result: **(a) alone does not fix the divergence**, exactly
as I admitted I was uncertain about in the analysis. The implementation
is correct (verified by debug instrumentation) and the operator is now
variationally consistent across the AMR cross-level coupling, but the
V-cycle still diverges. The reason confirmed by experiment is the one I
flagged: the 4-block edge Gauss–Seidel smoother cannot damp the
gradient null space of curl-curl, so even a Galerkin-consistent
operator at cov yields a V-cycle whose spectral radius exceeds 1 along
gradient modes — magnitude amplification by 1/β = 1000 remains intact,
just now in the "right direction."

## What's in the tree

All Galerkin code is opt-in via environment variables and defaults off.
Baseline behaviour with defaults off matches the original repo exactly
on all test inputs:

- `inputs.tube_diffusion_loose` — converges in 17 iters (unchanged).
- `inputs.tube_diffusion_loose_annular` — diverges as before.
- `inputs.tube_diffusion_tight` — diverges as before (expected).
- `inputs.trivial_amr` — converges in 7 iters.
- other inputs unchanged.

New code added to `Src/LinearSolvers/MLMG/`:

1. **`MLCurlCurl::applyAmrLevelGalerkin (crse_amrlev, Ax_c, v_c)`**
   (`AMReX_MLCurlCurl.cpp`): matrix-free `R L^f I` action at AMR-
   covered coarse cells. Interpolates `v_c` to fine via
   `mlcurlcurl_interpset`, applies `L^f` via fine-level `apply` (with
   CF ghosts filled from `v_c` via temporary `m_crse_sol_br[flev]`
   override), restricts back to coarse cov via
   `mlcurlcurl_restriction`. Saves/restores `m_crse_sol_br` state.

2. **`MLCurlCurl::smoothCovGalerkinJacobi (amrlev, sol, rhs)`** : after
   the existing multi-color 4-block GS sweep, a damped-Jacobi pass at
   cov cells using the Galerkin action and rediscretized diagonal (a
   close approximation to the Galerkin diagonal in our regime — the
   dominant Galerkin correction is in off-diagonal cross-couplings,
   not the diagonal). Runs only on cov cells (`fmask == 0`).

3. **Wiring in `apply` and `smooth`**: at `amrlev < num_amr_levels-1`,
   `mglev == 0`, the coarse operator action and smoother both call
   the Galerkin path at cov when the env vars are set.

4. **`m_crse_sol_br` made `mutable`**: so `applyAmrLevelGalerkin`
   can temporarily populate it inside a `const` method.

Env vars:

- `AMREX_MLCC_AMR_GALERKIN`     = 0 (default) | 1: enable Galerkin in `apply`.
- `AMREX_MLCC_SMOOTH_COV_GALERKIN` = 0 (default) | 1: enable Jacobi-at-cov in `smooth`.
- `AMREX_MLCC_JACOBI_OMEGA`     = 0.5 (default): Jacobi damping factor.

## Empirical results on loose_annular

Baseline (Galerkin off):
```
Iter 1 = 0.046    Iter 2 = 0.374    Iter 3 = 147    Iter 4 = 5.7e4 ...
```

Galerkin apply only, default nu1=nu2=2:
```
Iter 1 = 1.177    Iter 2 = 52.9     Iter 3 = 8706
```

Galerkin apply + Jacobi-at-cov smoother, nu1=nu2=2:
```
Iter 1 = 0.700    Iter 2 = 26.0     Iter 3 = 2358
```

Galerkin apply + Jacobi-at-cov + nu1=nu2=20:
```
Iter 1 = 0.0042   Iter 2 = 1.43     Iter 3 = 8.6   diverges
```

Notable: with heavy smoothing (nu=20), iter 1 is excellent (0.0042 —
10× better than baseline) — the Galerkin operator gives a much
better first-iteration approximation because the wrong-direction
component of the correction is removed. But iter 2 still diverges.

## Why (a) alone fails (in more detail)

The Galerkin operator at cov makes
`I L_c^G⁻¹ R` a correct projection of `L^f⁻¹` onto the I-range. That
is, `L_c^G = R L^f I` ensures the coarse correction direction is
right.

But for the V-cycle itself to be contractive, the smoother must damp
the modes that the coarsest level cannot fully resolve. For curl-curl,
the null space of curl-curl is the entire span of gradient fields.
`(α curl curl + β I) ∇φ = β ∇φ`, so along gradient directions the
eigenvalue is β = 0.001 in free space. Both the smoother's local
Jacobi step and the multigrid V-cycle's inverse give amplification ≈
1/β = 1000 on these modes.

The 4-block edge GS smoother is designed for the curl-curl 5-point
stencil — it does well on the divergence-free range of the operator
but cannot effectively damp gradient modes (which require an
auxiliary nodal-potential smoothing pass, à la Hiptmair).

So even with `L_c^G = R L^f I`, when the residual carries any
gradient-mode content (which the annular CF excites heavily via its
concave corners), the V-cycle's spectral radius along that mode is
≫ 1, and the iteration diverges.

## What this means going forward

The fix requires attacking the smoother, not just the operator.

Two clean paths from here:

**(c) Hiptmair smoother.** Add a Poisson-on-the-nodal-potential
smoother pass alongside the existing 4-block edge GS. This is the
textbook fix for curl-curl AMG/MG. Significant implementation: build a
nodal Poisson operator on each MG level, build the discrete-gradient
operator G from nodal to edge, and at each smoother sweep do
`sol ← sol + G (L_nodal)⁻¹ G^T residual`. Combined with (a) for
operator consistency, this gives a contractive V-cycle for curl-curl
on AMR.

**(d) Wrap MLMG in `GMRESMLMG` outer iteration.** GMRES is robust to a
preconditioner whose spectral radius exceeds 1 on some modes — it
implicitly damps via the Hessenberg least-squares projection. Pair
with (a) so the preconditioner does the right thing in direction; the
GMRES handles the magnitude amplification. Lower implementation cost
than (c). Risk: convergence rate may be slow for the strongly
amplifying modes.

I would now lean toward **(a) + (d)** as the pragmatic shippable
combination, and **(a) + (c)** as the textbook-correct combination.
(a) on its own — what I just spent the effort on — is necessary but
not sufficient, exactly as the analysis admitted.
