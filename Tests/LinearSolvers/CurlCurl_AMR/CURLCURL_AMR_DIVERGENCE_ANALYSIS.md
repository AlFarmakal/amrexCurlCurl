# CurlCurl AMR composite-solve divergence: analysis

Investigation of why `inputs.tube_diffusion_loose_annular` diverges
catastrophically under `composite=1`, while `inputs.tube_diffusion_loose`
converges cleanly. Both inputs are identical except for the level-1 grid
geometry: loose uses a clean rectangular CF patch (`tube_refine_kind=box`,
4 sharp CF corners), loose_annular uses a stepped disk
(`tube_refine_kind=disk`, many concave corners on a tile-stepped circular
CF). The annular case puts the CF in free space (β = 1/η_FS = 0.001), well
outside the tube wall.

## Reproduction

- `inputs.tube_diffusion_loose` converges in 17 iters: ratios 0.046 → 4e-8.
- `inputs.tube_diffusion_loose_annular` diverges by ~390× per iter from
  iter 3 onward: 0.046 → 0.374 → 147 → 5.7e4 → ... → 1e20 by iter 10.
- Iter 1 residual ratio is **identical** (0.046) — the first V-cycle on
  `sol=0` reaches the same point either way. The divergent eigenmode is
  only excited once `sol` carries non-trivial structure.

## Localization

I instrumented `MLMG::oneIter` (per-step `(|res|, |sol|, |cor|)` norms,
both masked-uncov and all-cells) and `MLCurlCurl::reflux` /
`avgDownResAmr` (per-stage uncov/cov-split norms).

The blowup is fully localised to the **F/V-cycle on AMR level 0** (oneIter
step "04-after mgFcycle"):

| iter | `|res[0]|`<br>(before V-cycle) | `|cor[0]|`<br>(from V-cycle) | ratio |
|------|----:|----:|---:|
| 1    | 1.06 | 0.68 | 0.6×   |
| 2    | 0.85 | **364** | **428×** |

The V-cycle amplifies a moderate residual into a huge correction in iter
2. `sol[0] += cor[0]` makes sol huge; `interpCorrection(1)` propagates it
to `sol[1]`; the next iteration's `computeResidual(finest)` sees a hugely
inconsistent residual. Diverges.

Confirmed independent of MG hierarchy: `max_coarsening_level=0` (lev-0
V-cycle becomes a single bicgstab bottom solve, no MG at all) reproduces
the divergence at the same factor. The amplification source is not within
the lev-0 MG hierarchy.

## Two contributing mechanisms

### (i) `composite_sol` CF discontinuity inside `reflux`

`reflux` builds `composite_sol` as the user's `crse_sol` overlaid with
`avg-down(fine_sol)` at covered cells. By the time `reflux` runs inside
`computeResWithCrseSolFineCor`, `fine_sol` has just been updated by
`miniCycle`, but `crse_sol` is still at the previous iter's
`averageDownAndSync` snapshot. The cov side gets fresh avg-down, the uncov
side stays stale, so there is a jump across the CF boundary.

Iter-2 norms of `composite_sol` (uncov vs cov side of CF):

| case          | uncov | cov  | jump   |
|---------------|------:|-----:|-------:|
| loose         | 1.33  | 1.40 | 5%     |
| loose_annular | 0.22  | 0.60 | 170%   |

When `L^c` is then applied to `composite_sol` and the result overwrites
`res[0]` at uncov, the spurious residual at uncov-near-CF is:

| case          | reflux-induced `|res|` uncov |
|---------------|----:|
| loose         | 0.005 |
| loose_annular | **0.79** |

This is the CF-jump artifact. **But disabling this overwrite alone does
not stop the divergence** — confirmed via env-var experiment. So it is a
contributor, not the dominant cause.

### (ii) `avgDownResAmr` × rediscretized `L^c` × free-space β

`avgDownResAmr` deposits `R(fine_rescor)` at the cov region of `res[0]`.
The V-cycle on lev 0 then inverts this with the rediscretized `L^c`. In
free-space cov cells, β = 0.001, so `|L^c^-1|` ≈ 1/β = 1000 along
gradient-mode directions (the kernel of curl-curl). Any gradient content
in `R(fine_rescor)` is amplified by ~1000×.

The interpolated coarse correction then doesn't cancel the fine
residual it was meant to correct — it injects gradient-mode noise into
`sol[1]` near the CF. The next iteration's fine residual is **worse**.
Divergence.

The annular geometry triggers this because its jagged stepped CF has many
concave corners that the fine smoother can't fully damp; the residual
arriving at `avgDownResAmr` carries gradient content. The box geometry
has a smooth 4-corner CF whose residual stays nearly gradient-free.

## Confirmations

- Disabling **both** `avgDownResAmr` overwrite and `reflux` uncov
  overwrite (FAC-style): residual stabilises at ratio ~0.058 (stagnates
  rather than diverges). Skipping just `avgDownResAmr` produces a slower
  alternating divergence (×1.3, ×7, ×1.3, ×7, ...).
- Reducing β contrast (`tube_eta_FS=10` → β=0.1): divergence factor drops
  from ~390× to ~3.4× per iter — directly proportional to 1/β, confirming
  the amplification mechanism.
- Heavy fine smoothing (nu1=nu2=20): iter 1 better (0.005) but divergence
  still kicks in by iter 5. Smoother strength alone doesn't fix it.
- PCG inside the 4-block edge smoother (`setUsePCG(true)`): no effect.
- Damping `cor[0]` by 0.01 before propagation: delays divergence to iter
  7 (residual decreases through iter 6, then resumes growing). Damping
  suppresses magnitude but not direction error.
- LBL (`composite=0`) converges cleanly on the annular case (lev 0: 12
  iters to 4e-8; lev 1: 14 iters to 7.9e-8) — confirms the trouble is
  purely in the composite cross-AMR coupling.

## Candidate fix paths

None are one-line changes.

**(a) Galerkin coarse operator on the cov region.**
Replace the rediscretized `L^c` at AMR-coarsened cov cells with the
variational coarsening `R L^f I`. The current `L^c` at cov uses
`α/dx_c²` and `avg-down(β)`; this is not the algebraic Galerkin product.
For curl-curl with extreme β contrast, the directional mismatch between
`L^c` and `R L^f I` is exactly what makes `L^c^-1 R(fine_rescor)` an
inconsistent (rather than just amplified) correction. Requires modifying
`apply` and the smoother at amrlev=0 mglev=0 to detect cov cells and
apply a Galerkin stencil.

**(b) Flux-difference reflux at CF edges (MLABec-style).**
Replace the current "re-apply `L^c` to `composite_sol` and overwrite res
at all uncov" with a true conservative flux correction at the CF coarse
edges only. For staggered curl-curl, the "flux" at an Ex/Ey edge is the
curl_z value at the perpendicular nodes; compute coarse-stencil curl_z
and avg-fine curl_z at the CF nodes, take the difference, add to the
coarse residual only at the adjacent CF coarse edges. This is what
`EdgeFluxRegister` is designed for and what MLABec does. Significant
implementation work but follows established AMReX patterns.

**(c) Hiptmair smoother (or equivalent) for curl-curl V-cycles.**
The 1/β amplification of gradient modes is intrinsic to the
edge-discretised curl-curl operator. Hiptmair (1998) alternates the
edge-based 4-block smoother with a nodal Poisson smooth on an auxiliary
potential field, where gradient modes live. This is the standard
remedy in EM AMG literature (HYPRE's AMS preconditioner is built on
it). Largest implementation effort.

**(d) Wrap `MLMG` in `GMRESMLMG` outer iteration.**
`GMRESMLMG` already exists. GMRES tolerates a preconditioner with
spectral radius > 1 on some modes — it will adapt the search direction
even when MLMG produces wrong-magnitude corrections. Lowest
implementation cost: only the test driver changes. Doesn't fix the
root cause; risk that if MLMG's directions are catastrophically wrong,
GMRES converges very slowly or stagnates.

## Recommendation

See companion analysis: this file is the findings; the (a) vs (b)
choice is non-obvious and is discussed separately.

## What I left in the tree

Nothing. All exploratory diagnostic instrumentation (in `MLMG.H`,
`MLCurlCurl.cpp`, `MyTest.cpp`) has been reverted. The working tree only
contains this analysis file.
