# Composite beta-weighted auxiliary-space (AMS) correction: design

Design for the composite gradient-channel correction of the MLCurlCurl
AMR solves, addressing the three remaining symptoms of
`CURLCURL_AMR_CONTRACTION_ROOT_CAUSE.md` Parts 2–4 (one cause: the
gradient channel is only ever treated per-level):

(a) tube-3L stationary floor 1.04e-6 at `tube_refine_band=0.1` — the
    KCHARGE reference's seam bias, proven irreducible per-level
    (Part 4 addendum);
(b) 3-level `composite=2` SOL2ND=2 GMRES crawl (500 it → 3.8e-3);
(c) 2-level pocket stationary rate rho≈0.89 (tight 47 it) and the
    accurate-system GMRES count (167 it, target ~30–40).

## Decision summary

**One mechanism: a composite multiplicative AMS correction applied once
per MLMG `oneIter`, hooked at the end of `averageDownAndSync` (the only
per-iteration linop callback that sees the full updated `sol` vector).**
Env-gated `AMREX_MLCC_AMS=1`, default 0 = bit-identical current
behavior. KCHARGE gating is left untouched (it provides within-cycle
stability that a post-cycle correction cannot replace at extreme beta —
the coarse solve amplifies stitch garbage by 1/beta *within* the
down-leg).

Rejected alternatives:

* **Per-level Hiptmair pass after smoothing** — re-introduces exactly
  the per-level patch-truncation family refuted in Part 4 (the floor is
  not the truncation, and per-level aux solves are what we have).
* **`MLNodeLaplacian` as the aux solver** — its 9-point FEM stencil
  with cell-centered sigma is not the operator's gradient channel
  (`L(G phi) = beta o G phi` holds exactly only for the 5-point
  edge-beta nodal Laplacian already implemented in
  `betaNodalPoissonCG`). A stencil mismatch would not break consistency
  (the correction is residual-linear, see below) but costs correction
  quality exactly where it matters (1/beta-amplified ring charges), and
  the integration surface (separate MLMG, sigma fabrication, rhs
  conventions) is large. Kept as fallback.
* **Retiring KCHARGE in the stationary cycle under AMS** — unsafe: at
  beta_FS=1e-6 the in-cycle 1/beta response to assembly charges grows
  faster per iteration than any post-cycle correction can remove.
  KCHARGE stays; AMS suppresses its bias footprint instead (below).

## Why this fixes the floor (a)

The floor mechanism, made precise during this design: at the exact
discrete solution the assembled residual is zero but the KCHARGE target
`div(stash)` is *not* zero at ring nodes (the avg-down-trace
commutator), so the cycle injects a fixed spurious residual
`g0 = beta o G(Hodge(ring bias))` every iteration. The stationary fixed
point satisfies `e* = (I - M)^-1 g0` — a residual floor proportional to
the bias, irreducible by making M (the cycle) better at everything
except exactly this content. The AMS correction appends a step
`(I - C_ams A)` to the iteration map; `g0` is a smooth beta-weighted
gradient field — precisely the content the composite aux solve
represents best — so the floor is multiplied by the aux-solve defect
factor eps (the fraction of a gradient field the FAC solve fails to
remove). eps ≤ 0.1 suffices for ≤1e-7; a one-pass FAC typically gives
1e-2 or better.

Consistency guarantee: every AMS input is linear in true residuals
(`r = b - A sol`), so the correction vanishes identically at the exact
discrete solution for *any* aux-solver quality — the fixed point is
never displaced, only the rate and the bias footprint change. This is
the structural difference from PROJECT_CHARGE (which deletes genuine
residual content, Part 1) and from the refuted reference-folding
variants (Part 4).

## The correction, step by step

All 2D-only (`AMREX_SPACEDIM == 2`), ref_ratio 2, gated off in
matrix-apply context (`m_matrix_apply`) so the GMRES matrix action is
untouched. Inside the GMRES *preconditioner* (`m_precond_mode`) it
fires automatically through the same `averageDownAndSync` hook — that
is the AMS preconditioner for (b)/(c); the Solution-mode applies inside
it pick up the second-order CF ghosts via the existing SOL2ND gating,
so the correction sees the same (matched) operator the preconditioner
is solving.

Per pass (`AMREX_MLCC_AMS_PASSES`, default 1):

1. **Per-level solution residuals** (top-down, using the just-synced
   `sol`): `r_lev = b_lev - A_lev(sol_lev)` via the production
   `apply(..., BCMode::Inhomogeneous, StateMode::Solution)` with CF
   ghosts from `sol[lev-1]` (`setLevelBC`), Dirichlet rows zeroed.
   At a coarse level this reproduces the genuine reflux-row values at
   uncovered+fringe edges for free (after averageDownAndSync,
   `sol_c = composite_sol`, and the reflux rows are
   `rhs_c - L^c(composite_sol)` by construction); covered rows are
   wrong but never used (overwritten in charge space, step 2).
   The rhs `b_lev` is stashed (copied, x/y components) by `reflux`
   each down-leg — pointer stashing is unsafe because
   `MLMG::computeResidual`/`MLMG::apply` pass temporaries.

2. **Composite charge assembly**, finest → coarsest. Uniform formula
   per level:
   `q_lev = -divUncov(r_lev) + R(q_lev+1)` where
   * `divUncov` sums `±r_e/h` over incident *uncovered* edges only
     (fine-mask-gated; on the finest level all edges count, and the
     zero-extended ghosts make patch-boundary nodes carry exactly the
     fine-side half-divergence);
   * `R` is the charge-preserving nodal full weighting
     (1/4, 1/8, 1/16 — the FARFIELD restriction), owner-masked on the
     coarsened-fine layout and `ParallelAdd`-ed so interface ring nodes
     receive fine-sector flux *additively* on top of their own
     uncovered coarse-sector flux. This is the standard FAC composite
     nodal rhs: covered interior = restricted fine charge, ring = fine
     sector + coarse sector, uncovered = level charge. No stitched
     edge-residual divergence is ever taken, so no Part-1 corner
     garbage can appear.
   The sign: with `q = -div(r)`, the SPD aux solve
   `-div(beta grad phi) = q` gives `beta o G(phi) ≈ r_grad`, i.e.
   `A(G phi) ≈ r_grad`, so the *solution* update is `sol += G phi`.
   Dirichlet domain nodes zeroed; owner-synced.

3. **Composite FAC solve** of the aux problem:
   * level 0: `betaNodalPoissonCG(0, phi_0, q_0)` (existing kernel,
     domain-covering, periodic mean removal / Dirichlet domain nodes);
   * level l = 1..L: bilinearly interpolate `phi_{l-1}` onto the
     patch's nodal layout, valid + 1 ghost (`Iphi`, the FARFIELD
     interpolation); solve the patch *defect* with the same 5-point CG,
     treating patch-boundary (interface) nodes as Dirichlet
     (`delta = 0` there, so `phi_l = Iphi + delta` equals the coarse
     data on the interface — the half-charge ambiguity at interface
     nodes never enters a fine solve; the composite ring charge was
     already consumed by the coarser solve in step 2). Proper nesting
     (≥ 2 coarse cells, enforced by MyTest) guarantees the interpolant
     covers the patch + ghost.

4. **Correction**: `sol_lev[x/y] += damp * (G phi_lev)` at non-Dirichlet
   edges (`AMREX_MLCC_AMS_DAMP`, default 1.0). The z (nodal) component
   is untouched (2D gradients have no z part). On the interface the
   fine tangential update is the tangential derivative of the
   interpolated coarse potential — exactly curl-conforming with the
   production prolongation (`P G_c = G_f P_n`, the probe-verified
   identity), so the subsequent re-sync is a no-op there.

5. Re-run the average-down + owner-sync body so `sol` leaves the hook
   composite-consistent.

## Implementation surface

* `AMReX_MLCurlCurl.H/.cpp`:
  - split `averageDownAndSync` into the existing body
    (`averageDownAndSyncImpl`) + the gated AMS hook;
  - `compositeAMSCorrection(Vector<MF>& sol) const` implementing
    steps 1–5;
  - `betaNodalPoissonCG` gains two defaulted args
    (`MultiFab const* inhom_bdry`, `iMultiFab const* interior_mask`)
    for the patch defect solves — null args keep the current behavior
    bit-identically;
  - rhs stash: `m_ams_rhs` (mutable, x/y copies) + `m_ams_rhs_valid`,
    filled in `reflux` when AMS is enabled and not in matrix-apply.
* Env gates: `AMREX_MLCC_AMS` (0 default), `AMREX_MLCC_AMS_PASSES` (1),
  `AMREX_MLCC_AMS_DAMP` (1.0), `AMREX_MLCC_AMS_VERBOSE` (0).
* Cost per outer iteration at AMS=1: one extra level `apply` per level
  plus one nodal CG per level — comparable to the existing KCHARGE
  Hodge cost; only paid when gated in.

## Validation matrix

Build: `cd Tests/LinearSolvers/CurlCurl_AMR && make -j8`; runs pinned
`taskset -c 16-31 mpiexec --bind-to none -n {1,4}`.

Success criteria (AMS=1):
* tube-3L `tube_refine_band=0.1 composite=1` → ≤ 1e-7 (now floor
  1.04e-6);
* tight 2L `composite=1` < 47 it;
* tight `composite=2` ≪ 167 it;
* 3L `composite=2` (SOL2ND=2) breaks the 500-it crawl.

No-regression (AMS unset → bit-identical):
* tight 47 it → 9.412390583e-8; annular 18 it; loose 14 it; trivial
  7 it; L (`fine_box2…`, KCHARGE=0, beta=1) 164 it → 9.090785493e-11;
* tight `composite=2` 167 it → 4.493342797e-8; shifted `composite=2`
  err 0.499;
* 3L manufactured (KCHARGE=0) 43/164 it; LBL (`composite=0`) unchanged.

---

## Implementation results & findings (what landed)

`AMREX_MLCC_AMS=1` (default 0) enables one composite gradient-channel
pass per MLMG `oneIter`, hooked at the end of `averageDownAndSync`. The
charge is the divergence of the charge-consistent reference field
`rhs − β∘sol` (never an `apply`-based residual — see finding 1), solved
by a composite FAC (`betaNodalPoissonCG`: level-0 domain solve + finer
patch-defect solves with the interpolated coarser potential as Dirichlet
data and a buffer band), and applied as `sol += damp·grad(phi)`.

**Verified bit-identical with AMS off** across the full no-regression
matrix above — the feature is zero-risk to existing behavior.

**Where AMS helps (gated on):**
| case | baseline | AMS=1 |
|---|---|---|
| tube tight 2L `composite=1` | 47 it → 9.41e-8 | **26 it → 7.90e-8** |
| manufactured 3L rect (KCHARGE=0) | 43 it | **23 it** |
| manufactured L-pocket 3L (KCHARGE=0) | 164 it | **58 it** |

**Where AMS does NOT yet help (primary goal, unmet):**
* tube-3L `band=0.1 composite=1`: AMS floors at ~2.6e-5, *above* the
  KCHARGE baseline 1.04e-6 — AMS is not beneficial on the β-contrast
  3-level staircase and should stay off there.
* 2-level/3-level `composite=2` (GMRES): AMS-in-preconditioner is
  bounded but does not beat the matched-operator baseline (167 it).

### Findings (root-caused this session)

1. **The gradient charge must come from `rhs − β∘sol`, never from a
   `b − A(sol)` residual.** `div∘(α curl curl) ≡ 0` identically, so
   `div(r) = div(rhs − β∘sol)` — but only if you never assemble the
   curl-curl term. Taking the divergence of an `apply`-based residual
   re-introduces the Part-1 spurious CF-corner charge (the asymmetric
   assembly's boundary term), which the nodal solve amplifies by `1/β`.
   An early `apply`-based version diverged ~`1/β` per iteration on the
   β-contrast tube and in the Krylov preconditioner (where the pocket
   symmetrization is inert). This was the decisive fix that made the
   2-level GMRES bounded again. The composite charge must likewise be a
   *single field's* divergence (`divUncov(ref) + R(q_fine)`, the FAC
   nodal rhs); a node-charge restriction that drops the interface
   boundary term re-creates the same `O(rhs/h)` spurious ring charge.

2. **The remaining 3-level β-contrast floor is an architectural limit of
   the per-level-kernel correction, not a tuning issue.** The post-cycle
   additive correction `E += grad(phi)` only stays out of the assembled
   composite curl-curl's range if it is *exactly* curl-conforming there,
   i.e. `grad(phi)` must be a composite gradient satisfying
   `P G_c = G_f P_n`. A per-level patch FAC produces a continuous `phi`
   but `avg-down ∘ P ≠ I` at the middle CF interface, so the correction
   has a kink there → a curl-channel charge → `1/β`-amplified by the
   middle level's *partial* solve (exactly the Part-4 mechanism). At
   β = 1 the amplification is unity and AMS converges fully (manufactured
   cases); at the wall contrast it sets the floor. Confirmed by: the
   uniform-η tube has no such blow-up; a buffer band on the patch defect
   does not move the floor; pure curl-conforming prolongation (no fine
   defect) discards the fine gradient detail the 2-level needs and
   stalls.

   **The structural cure is a true composite/Galerkin nodal operator**
   (e.g. an `MLNodeLaplacian`-style FAC where `avg-down ∘ P` consistency
   is built into the operator), so that `grad(phi)` lies in the exact
   discrete kernel at every interface. `betaNodalPoissonCG` (now with
   the inhomogeneous-boundary / interior-mask patch-defect mode) and the
   charge-consistent reference assembly are the landed building blocks
   for that next step.

### Env gates

`AMREX_MLCC_AMS` (0), `AMREX_MLCC_AMS_PASSES` (1),
`AMREX_MLCC_AMS_DAMP` (1.0), `AMREX_MLCC_AMS_BUFFER` (2 fine cells),
`AMREX_MLCC_AMS_VERBOSE` (0).
