# Curl-curl AMR composite: root cause of non-contraction (pure GMG)

Investigation of the *stationary* composite V-cycle (`composite=1`, no
Krylov wrap), answering: which AMR-operator inconsistency breaks
contraction, and can the cycle be made consistent **and** contracting by
fixing the AMR-level operators?

**Answer: yes — the contraction defect is a single, now-proven mechanism:
the composite residual assembly (reflux rows at uncovered coarse edges +
restricted fine residual at covered edges) creates O(α/h²)-scale point
charges — spurious discrete-divergence sources — at CF *corner* nodes.
True curl-curl residuals have only β-scale divergence, so the coarse
solve's gradient channel (eigenvalue β) responds to these charges with a
smooth spurious gradient correction of amplitude ~charge/β. At
convex-corner-only geometries the per-corner charges nearly cancel in
± pairs (slow convergence / stalls); at re-entrant (staircase) corners
adjacent charges have the *same sign* and reinforce ⇒ ρ ≈ δ(geometry)/β
and divergence. Removing the charge (a β-weighted discrete Hodge
projection of the assembled coarse residual, env-gated test code)
eliminates divergence on every previously-divergent 2-level case.**

This supersedes the earlier working theories: it is *not* the smoother
(no Hiptmair needed for contraction at 2 levels), *not* the
rediscretized-vs-Galerkin covered-region operator, *not* reflux
snapshot staleness, and *not* the first-order CF prolongation (that is
the separate, independent *accuracy* defect of the fixed point — see
`ROOT_CAUSE_AND_FIX.md`).

## Experiment matrix (key discriminators)

All on this branch, `Tests/LinearSolvers/CurlCurl_AMR`, `make -j`,
`mpiexec -n {1,4} ./main2d.gnu.MPI.ex <inputs> composite=1 …`:

| # | configuration | result | kills hypothesis / shows |
|---|---------------|--------|--------------------------|
| E1 | tube **box** CF (4 convex corners), β_FS = 1e-3 / 1e-4 / 1e-5 / 1e-6 | conv 17 it / conv 29 it / **diverges ρ≈3.8** / **diverges ρ≈45** | staircase not required; ρ ∝ 1/β with δ_box ≈ 4e-5 |
| E2 | manufactured **rect** patch, uniform β = 1e-2 … 1e-6 (`beta_scalar=`) | never diverges (stalls at 1/β-scaled floor) | uniform-β rect: δ < 1e-6 |
| E3 | tube **jagged** CF with **uniform** β=1e-3 (`tube_eta=1000`) | diverges ~400×/it, same as with wall | β-contrast/wall not required; staircase flips δ to ~0.4 |
| C1 | LBL (`composite=0`) box CF at β=1e-6 | converges 12+16 it | within-level machinery (AFW-type gs4 smoother, intra-level MG) is β-robust; defect is cross-level |
| R2 | **minimal reproducer**: manufactured rect + one extra box ⇒ single re-entrant corner (`fine_box2_*`), uniform β | **diverges from iter 1**, iter-1 ratio = 862/β exactly (β = 1 → 862!) | one re-entrant corner suffices; even β=1 diverges; seed is β-independent, response ∝ 1/β |
| C2 | rect multi-box (`max_grid_size=16`) | bit-identical to single-box | shared-face edge duplication innocent |
| C3 | LBL on the L geometry | converges 5+31 it | cross-level confirmed again |

## Probes (gated, default-off, in `MyTest.cpp`)

* `probe_grad=1` — feeds an exact discrete coarse gradient G_cφ through
  the production cross-level coupling.
  - **P-path** (`interpolationAmr` + interpset ghost fill via
    `solutionResidual`): defect ≤ 2e-11 *even at the re-entrant corner*.
    The prolongation pair (interpadd interior + interpset ghosts, same
    weights, same coarse data) is exactly curl-conforming:
    P G_c = G_f P_n with P_n bilinear. **Prolongation exonerated.**
  - **A-path** (compatible composite gradient through
    `solutionResidual(0)` + `reflux` + `avgDownResAmr`): fine rows clean
    (2e-10); assembled coarse rows defect ~0.2·β|∇φ| spread along faces —
    β-scaled, same for rect and L. Smooth-compatible inputs do *not*
    excite the defect.
* `probe_cycle=N` (+ `probe_win_*` window dump) — N fixed MLMG
  iterations on the real rhs, then true-residual / solution argmax and a
  corner-window dump of curl(sol) and sol[0]-vs-avgdown.
  On the L after 1 iter: residual max 4.3e5 at exactly the two interface
  fine edges incident at the re-entrant corner; sol carries an O(50)
  smooth blob; the coarse cor[0] is a smooth O(40) field — 40× the
  solution scale at β=1 — i.e. the coarse solve was fed a residual with
  massively wrong gradient-channel content (invisible to ∞-norms:
  0.04 absolute suffices at β=1e-3).
* `probe_div=1` — **the decisive instrument**: the discrete divergence
  (charge map) of the assembled coarse residual. A pure curl-channel
  residual has exactly zero discrete divergence; div(rhs) ≈ 1e-11.
  Measured charges of the iter-1 assembly (β=1, units of the 502 rhs
  norm):

  ```
  rect:  div(40,8) = -16258   div(8,40) = +15698     (± pair, cancels)
  L:     div(40,24) = -10626  div(41,24) = -6441     (pocket: SAME sign)
         div(8,40)  = +15698
  ```

  Same-order charges appear for *every* snapshot variant tried
  (fresh-slave recompute, old-snapshot reflux) ⇒ the charge is
  structural in the *stitching* of the two residual representations
  (the adjoint identity div_c(Pᵀ r_f) = P_nᵀ(div_f r_f) holds in the
  covered interior but its boundary term at interface nodes is dropped),
  not a staleness artifact. This is why all snapshot fixes failed —
  including two tried and refuted during this investigation.

## Proof by repair

`AMREX_MLCC_PROJECT_CHARGE=1` (env-gated, default off; test code at the
end of `MLCurlCurl::avgDownResAmr`): compute q = div(cres), restrict to
the CF ring (+1 node), solve the **β-weighted** nodal Poisson
−div(β∇φ) = q (diagonal-preconditioned CG), subtract the operator-range
gradient field β∘Gφ from cres. The β-weighting matters: for variable β
the operator's gradient range is β∘Gφ and the response to charge is
governed by the β-weighted Hodge structure (the AMS auxiliary operator);
with a plain Laplacian the tube cases keep diverging.

| case (2-level, `composite=1`) | baseline | + charge projection |
|---|---|---|
| tube tight (staircase, β_FS=1e-3) | diverges ~390×/it | converges to **1.5e-6** rel, stuck (tol 1e-7) |
| tube loose_annular (staircase) | diverges | **1.05e-7** rel (≈ tol) |
| tube loose box | 17 it | **15 it** ✓ |
| tube loose box, β_FS = 1e-6 | diverges | **converges 14 it** ✓ |
| trivial_amr | 7 it | 7 it ✓ |
| manufactured L (1 re-entrant corner) | diverges from iter 1, ∀β | contracts, stalls at 0.17 |
| manufactured shifted rect | 42 it (err 47) | stalls at 0.13 (regression) |
| tube tight 3-level | diverges by iter 4 | diverges at iter 44 (helped, not fixed) |

Reading: divergence is eliminated wherever the 2-level charge is
removed — ρ < 1 restored at every β and geometry. The remaining stalls
are the projection's *consistency offset*: the genuine corner-mode
gradient residual (β-scale, the thing the cycle must correct to reach
the fixed point) also lives on the ring and is projected away with the
spurious charge. Where corner errors are large relative to the rhs
(manufactured, β=1) the stall is large; on the tube it is ~1e-6–1e-7;
where charges were tiny it vanishes (box cases now converge fully).
The 3-level case additionally needs a proper aux solve at the middle
level (the test-grade CG runs on the patch BA with ad-hoc BCs).

## What this means

1. **The composite V-cycle is not fundamentally blocked.** The
   contraction defect is localized, mechanistic, and removable. No
   Hiptmair/AMS smoother is required for 2-level contraction — the
   existing vertex-patch (AFW-type) gs4 smoother plus intra-level MG is
   β-robust (LBL at β=1e-6 converges), and the cross-level prolongation
   is already exactly curl-conforming.
2. **The production fix is a charge-consistent assembly, not a
   projection.** The assembly must stop *creating* the spurious corner
   charge while keeping the genuine β-scale divergence content:
   restore the dropped boundary term of the restriction at interface
   nodes (equivalently: make the uncov-reflux rows and the Pᵀ-restricted
   cov rows agree on the discrete charge balance at every interface
   node, transporting the corner boundary-term charge instead of
   depositing it). The Hodge projection here serves as the mechanism
   proof and an emergency stabilizer.
3. The **accuracy** defect (first-order interpset ⇒ corner-wrong fixed
   point, err 47) is independent and untouched; fixing contraction does
   not fix accuracy and vice versa. A production composite needs both:
   charge-consistent assembly (contraction) + the 2nd-order CF ghosts on
   the *solution* path only (accuracy; `StateMode` already distinguishes
   the paths in `apply`).

## Failed hypotheses (tested and refuted this session)

* Fresh-snapshot reflux (recompute fine_res against composite_sol with
  refreshed slave register): diverges *worse* (~9× at iter 2). Removed.
* Old-snapshot reflux (no avgdown overlay): same-order divergence.
  Removed.
* Charge from shared-face edge duplication: refuted (C2).
* Smoother cannot damp gradients / needs Hiptmair (from
  `CURLCURL_AMR_GALERKIN_RESULT.md`): refuted for 2-level contraction
  by C1/C3 — the gs4 4-block smoother is a vertex-patch AFW smoother and
  handles the gradient space fine within a level.

## What's in the tree

* `MLCurlCurl::avgDownResAmr`: env-gated `AMREX_MLCC_PROJECT_CHARGE`
  (default off; =1 ring-restricted, =2 everywhere) — mechanism test /
  emergency stabilizer. Default behavior bit-identical to baseline
  (verified: tight diverges identically, loose 17 it).
* `MyTest.cpp`: gated probes `probe_grad`, `probe_cycle`, `probe_div`,
  `probe_win_*`; L/step geometry via `fine_box2_{lo,hi}_{x,y}`
  (manufactured only). `MyTest.H`: members for the second box.

## Reproduce the key results

```bash
cd Tests/LinearSolvers/CurlCurl_AMR && make -j

# minimal reproducer: one re-entrant corner, uniform beta=1, diverges iter 1:
mpiexec -n 1 ./main2d.gnu.MPI.ex inputs.shifted_amr beta_scalar=1.0 composite=1 \
  fine_box2_lo_x=40 fine_box2_lo_y=8 fine_box2_hi_x=55 fine_box2_hi_y=23 plot_dir=/tmp/L

# charge map (the smoking gun):
mpiexec -n 1 ./main2d.gnu.MPI.ex inputs.shifted_amr beta_scalar=1.0 probe_grad=1 probe_div=1 \
  fine_box2_lo_x=40 fine_box2_lo_y=8 fine_box2_hi_x=55 fine_box2_hi_y=23 plot_dir=/tmp/L

# proof by repair on the original divergent cases:
AMREX_MLCC_PROJECT_CHARGE=1 mpiexec -n 4 ./main2d.gnu.MPI.ex inputs.tube_diffusion_tight  composite=1 plot_dir=/tmp/t
AMREX_MLCC_PROJECT_CHARGE=1 mpiexec -n 4 ./main2d.gnu.MPI.ex inputs.tube_diffusion_loose tube_eta_FS=1.0e6 composite=1 plot_dir=/tmp/b
```

---

# Part 2: the charge-consistent assembly (implementation + results)

Follow-up session: implement the production fix specified above and
test it. Outcome: **every previously-divergent composite case now
converges under the Krylov mode (`composite=2`), including to the
second-order-accurate answer**; the pure stationary cycle
(`composite=1`) is bounded everywhere with a small documented residual
floor on jagged CF, and its remaining blocker is precisely
characterized (pocket-ghost coupling asymmetry, see below).

## What was implemented

1. **Charge-consistent assembly** (`AMREX_MLCC_KCHARGE`, default on).
   Phase 1 (`reflux`): stash the reference field
   `rhs_c − β∘composite_sol`, whose discrete divergence equals the true
   composite residual's charge exactly (the α curl(curl) part
   telescopes to zero identically, for any coefficients). Phase 2
   (`avgDownResAmr`): correct the assembled residual's charge to that
   target via a ring-restricted β-weighted Hodge subtraction
   (`hodgeNeutralizeCharge`, diagonal-preconditioned CG on the
   β-weighted nodal Laplacian). Verified: the corrected assembly's
   leftover charge scales exactly ∝ β (15698 → 6.5 at β=1 → 0.0064 at
   β=1e-3 on the L corner).

2. **Second-order solution path under Krylov**
   (`AMREX_MLCC_SOL2ND`, default 2): Solution-mode applies inside the
   GMRES context (matrix action AND the preconditioner's residuals —
   gated by `beginMatrixApply`/`m_precond_mode`) use the second-order
   CF ghosts, so GMRES converges to the second-order-accurate composite
   fixed point with a *matched* preconditioner. Matched preconditioning
   is decisively better than preconditioning the accurate system with
   the first-order cycle (tube tight: 167 it vs stall at 2.7e-6). The
   stationary cycle and all Correction-mode applies keep the
   lowest-order curl-conforming fill. New hook:
   `MLLinOpT::begin/endMatrixApply`, fired by `GMRESMLMGT::apply`.

3. `gmres_precond_iters` default raised to 4 in the test driver (the
   accurate system's corner near-null modes need the stronger
   preconditioner; 1 suffices for `AMREX_MLCC_SOL2ND=0`).

## Results (tol_rel: tube 1e-7, manufactured 1e-10)

| case | baseline composite | now: `composite=2` (GMRES) | now: `composite=1` (stationary) |
|---|---|---|---|
| tube tight | diverges 390×/it | **167 it → 4.5e-8** (accurate op) / 48 it (1st-order op) | bounded, floor 4.7e-5 |
| tube loose_annular | diverges | **162 it → 9.5e-8** | bounded, floor 1.4e-5 |
| tube loose | 17 it (wrong-ish corners) | 310 it @p1 | **14 it ✓** |
| tube loose, β_FS=1e-6 | diverges | **74 it ✓** (1st-order op; accurate op stalls: Hiptmair-gap regime) | **14 it ✓** |
| manufactured shifted | 42 it, **err 47** | **100 it → 8.7e-11, err 0.499** | stalls 0.13 (floor) |
| manufactured L (1 re-entrant corner) | diverges from iter 1 at ANY β | **173 it → 4.3e-11** (β=1e-3) | bounded, floor ∝1/β |
| trivial_amr | 7 it | 4 it, err 1.2e-3 | 7 it ✓ |
| LBL (`composite=0`) | — | — | unchanged (15 it) ✓ |

Iteration-count note: the first-order system converges in 27–74 outer
iterations (p=1); the second-order-accurate system currently needs
~100–170 with p=4. Reaching ~30–40 on the accurate system requires an
auxiliary-space (AMS-style) gradient-channel correction in the
preconditioner — the β-weighted nodal solver this branch already
contains is exactly the needed building block.

## The remaining stationary (`composite=1`) blocker, exactly

Residual-side corrections cannot push the stationary floors to zero —
there is a structural no-go: a correction that vanishes at the fixed
point reproduces the variational charge, which at re-entrant corners IS
the garbage (it is the boundary term of the *asymmetric* composite
operator). The asymmetry was localized exactly with two new probes:

* `probe_asym` (impulse columns of the assembled operator's charge
  map): flat-face and convex-corner interface rows deposit clean
  parent-edge charge dipoles that telescope along each face; at a
  re-entrant pocket, exactly four rows misroute dipole endpoints
  across the two faces.
* `probe_w` (row-to-charge map of the restriction alone): perfectly
  regular even at the corner — the misrouting is in the OPERATOR's
  couplings, not the deposit weights.
* Hand-derivation through the mixed pocket ghosts (the ghost edges
  whose interpset stencil reads 0.5·slave + 0.5·uncov) shows: the x–x
  couplings and all flat-face pairings are exactly weighted-symmetric
  (ratio 1/4 = the FD measure), but the **dxy cross-couplings at the
  pocket have a sign clash**: A(x2→y1) = −a·dxy/2 vs A(y1→x2) =
  +a·dxy/2 through the corresponding ghosts. Weighted symmetry forces
  the slave weight a → 0, but consistency (Σw = 1) then overshoots the
  fine↔uncov coupling by exactly 2×: pure ghost-weight tuning is
  over-determined. A solution exists in the larger design space
  (linear-extrapolation-type pocket fills with signed weights, plus
  matched reflux-row compensation with diagonal rebalancing) — a small
  solvable linear system over ~10 coefficients per corner orientation.
  That derivation + corner-aware kernels is the remaining work for
  full stationary convergence; the impulse probes above are the
  verification harness for it.

## Tried and refuted in this session

* Trace-ring deposits (`mlcurlcurl_restriction_cc`, kept as
  `AMREX_MLCC_CC_ASSEMBLY=1` diagnostic): reshapes the corner charge
  into a near-dipole but does not remove it.
* K-only split (residual's curl-part stitch): misses the rhs-stitch
  charge, which dominates at iteration 1.
* Clip-envelope (subtract only charge exceeding c·|target|): the
  envelope inherits the target's corner bias; sub-envelope garbage
  re-ignites divergence.
* sol2nd on the stationary solution path: does not lower the tube
  floors and regresses manufactured cases (the (P2nd−P1st)(cor) ring
  jolt) — hence the m_precond_mode gating.
* Direct dipole re-routing linear in corner-row residuals: wrong frame
  (the impulse columns are solution-linear, not residual-linear).

## Reproduce

```bash
cd Tests/LinearSolvers/CurlCurl_AMR && make -j

# converged + accurate composite (the headline):
mpiexec -n 4 ./main2d.gnu.MPI.ex inputs.tube_diffusion_tight composite=2 plot_dir=/tmp/t
mpiexec -n 4 ./main2d.gnu.MPI.ex inputs.shifted_amr composite=2 plot_dir=/tmp/s   # err 0.5, not 47

# fast first-order composite:
AMREX_MLCC_SOL2ND=0 mpiexec -n 4 ./main2d.gnu.MPI.ex inputs.tube_diffusion_tight composite=2 gmres_precond_iters=1 plot_dir=/tmp/t1

# stationary cycle: bounded with floor (no more divergence):
mpiexec -n 4 ./main2d.gnu.MPI.ex inputs.tube_diffusion_tight composite=1 plot_dir=/tmp/t2

# asymmetry probes (serial):
AMREX_MLCC_KCHARGE=0 mpiexec -n 1 ./main2d.gnu.MPI.ex inputs.shifted_amr beta_scalar=1.0 \
  probe_grad=1 probe_asym=1 fine_box2_lo_x=40 fine_box2_lo_y=8 fine_box2_hi_x=55 fine_box2_hi_y=23 plot_dir=/tmp/a
```
