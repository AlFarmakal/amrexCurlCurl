# (a) Galerkin coarsening vs (b) flux-difference reflux: which is right?

Sober comparison of the two principled fix paths from the analysis
document. Goal: pick one without ambiguity and explain why, or admit
where I am not sure.

## What each option actually does

### (a) Galerkin coarse operator on the cov region

Replace `L^c` (rediscretized: `α/dx_c²` Laplacian-style coupling with
`avg-down(β)`) with the algebraic product `L_c^G = R L^f I`, evaluated
only on covered cells of AMR level 0. This is the standard "Galerkin
coarsening" of algebraic multigrid, but applied across AMR levels rather
than across MG levels.

Concretely:

- For every uncov coarse edge: keep `L^c` as-is.
- For every cov coarse edge: the smoother and `apply` use the algebraic
  Galerkin stencil. That stencil has a wider footprint than the
  rediscretized 5-point pattern (in 2D, ratio-2: typically 9-point or
  wider) because `R L^f I` reaches into all fine cells touched by `I`
  and back through `R`.
- Construction options: build the sparse Galerkin matrix once at
  `prepareForSolve`, or compute stencil entries on the fly per edge in
  the apply/smoother kernels.

### (b) Flux-difference reflux at CF coarse edges

Replace the current "re-apply `L^c` to `composite_sol` and overwrite res
at all uncov" approach with a true conservative flux correction
restricted to CF coarse edges, mirroring `MLCellLinOp::reflux` /
`MLABecLaplacian` / `EdgeFluxRegister`.

For staggered curl-curl, the "flux" through an Ex edge is the curl_z
field at the two perpendicular coarse nodes. The reflux step:

1. Compute coarse curl_z at coarse nodes on the CF boundary from
   `crse_sol`.
2. Compute fine curl_z at corresponding fine nodes from `fine_sol`,
   average down to the coarse nodes.
3. Take the difference (coarse_curl_z − avg_fine_curl_z) at the CF
   nodes. Multiply by `α/dy` to get the flux mismatch in the L^c
   stencil. Add this to `res[0]` only at the two CF coarse Ex edges
   incident on each CF node. Similarly for Ey edges (mismatch in `α/dx`).
4. `L^c` itself is untouched everywhere.
5. `avgDownResAmr` still overwrites cov of `res[0]` with `R(fine_rescor)`,
   same as today.

## Does each option actually fix the observed divergence?

This is the part where I am going to be skeptical rather than confident.

### The divergence mechanism, restated precisely

From the diagnostics, the divergent eigenmode is:

1. After fine miniCycle, `fine_rescor` retains gradient-mode content at
   fine valid cells adjacent to concave CF corners. (The 4-block edge GS
   does not damp gradient modes well; the loose case does not excite
   gradient modes strongly because its CF is smooth.)
2. `avgDownResAmr` restricts `fine_rescor` to coarse cov. `R(fine_rescor)`
   at coarse cov free-space cells inherits gradient content.
3. The V-cycle on lev 0 inverts this with `L^c` rediscretized.
   `L^c` has eigenvalue β = 0.001 along gradient directions in free
   space, so `L^c^-1` amplifies gradient content by ~1/β = 1000.
4. `interpCorrection(1)` lifts the gradient-mode `cor[0]` to fine `sol[1]`.
5. The next iteration's fine residual is **worse**, not better, because
   the lifted gradient correction doesn't cancel the fine residual — it
   adds noise. Diverges.

The 1/β amplification is intrinsic to the curl-curl operator's gradient
null space: `(α curl curl + β I) ∇φ = β ∇φ`. Inverting an operator whose
spectrum includes β = 0.001 is itself a "1/β amplifier" for gradient
content. This is independent of grid resolution and AMR structure.

### Does (a) fix it?

Honest answer: **probably yes in direction, possibly not in magnitude. I
am not certain.**

What (a) buys you: variational consistency between `R`, `L^c` (now
Galerkin), and `I`. Then `L_c^G = R L^f I`, so
`I L_c^G^-1 R x = I (R L^f I)^-1 R x` is the best approximation of
`L^f^-1 x` in the range of `I`. The coarse correction `I cor[0]` now
*correctly* cancels the projected component of `fine_rescor`. There is
no "wrong-direction" amplification.

But — and this is the part I want to be explicit about — `L_c^G` still
has eigenvalue β = 0.001 along gradient modes in free space. The
**magnitude** of `cor[0]` along gradient directions is still ~1000× the
gradient component of `R(fine_rescor)`. The point is that this large
correction is now in the *right* direction to cancel the fine residual,
so the iteration converges instead of diverging.

The reason this might still not be enough: in a 2-level AMR setup, the
coarse correction is computed once per outer iter; whether that one
correction step is enough to drive the iteration to fixed-point convergence
depends on the smoother's effectiveness on the within-fine V-cycle, and
on the conditioning of `I L_c^G^-1 R L^f`. For curl-curl with the 4-block
edge GS smoother (which does not damp gradient modes well), I cannot
prove from first principles that a single Galerkin-coarse correction
restores convergence — even though variational consistency is necessary.

**What I am sure of:** (a) is necessary for an asymptotically convergent
iteration. It fixes the "wrong direction" component of the spectral
radius.

**What I am not sure of:** whether (a) alone suffices, or whether
the 4-block edge smoother is also too weak on gradient modes and needs
Hiptmair-style augmentation.

### Does (b) fix it?

Honest answer: **probably not on its own, and I am fairly confident
about this.**

What (b) buys you: a correct, conservative residual at the CF coarse
edges. The current `reflux` introduces spurious residual at uncov-near-CF
cells (the 0.79 at iter 2) because `L^c[composite_sol]` reads across the
CF jump in `composite_sol`. (b) eliminates this spurious contribution.

But the divergence is not driven by the spurious uncov contribution.
This was tested directly with the env-var experiment:
`AMREX_MLCC_SKIP_REFLUX_UNCOV=1` removes the uncov overwrite entirely
(stronger than what (b) would do — (b) keeps a small targeted correction
at CF edges, while the env var skips even that). The result: divergence
at the **same factor** (0.046 → 0.374 → 147 → ...). The spurious
uncov contribution is a 0.79 perturbation; the divergent factor stays
unchanged because the dominant amplification comes from the cov side.

(b) does not touch how `R(fine_rescor)` is deposited at cov, and does
not touch the cov-region operator. The 1/β amplification of gradient
content in `R(fine_rescor)` × rediscretized `L^c` happens exactly as
before. So (b) leaves the dominant divergence mechanism untouched.

**What I am sure of:** (b) is *necessary* to make `reflux` not introduce
the CF-jump artifact, and it brings the AMR composite operator into line
with the rest of AMReX's MLMG. As a code-quality and correctness fix
it is right.

**What I am sure of:** (b) is *not sufficient* on its own for the
annular case. The cov-side amplification will persist.

## Verdict

If forced to pick one and only one, **(a) is the more correct fix**.
It addresses the dominant mechanism (cov amplification × variational
inconsistency). (b) addresses a real but secondary mechanism (CF-jump
spurious uncov residual) and matches AMReX architecture, but does not
attack the inversion-direction issue.

This is not a comfortable verdict, because:

1. (a) is significantly more invasive: the Galerkin stencil at cov has
   to be constructed, the smoother needs a new code path, and the wider
   stencil has to be handled correctly at the cov/uncov interface.
2. (b) is the path AMReX-natural code would take, and it is what every
   other AMR cross-level operator in this codebase does.
3. The empirical test of "skipping both" (FAC mode) stagnates rather
   than converges, suggesting that without proper cross-level coupling
   we lose the V-cycle's contraction. Both (a) and (b) restore *some*
   coupling; only (a) makes that coupling variationally consistent.

I am explicitly uncertain about two things:

- **Whether (a) alone converges quickly, or whether it converges at all,
  for the annular case.** Variational consistency is necessary but not
  sufficient for convergence; the conditioning of the iteration matrix
  matters, and that depends on how the 4-block edge GS smoother interacts
  with gradient modes. I have not done the analysis to prove or disprove
  contraction. The honest expectation is "it should converge slowly,
  similar to loose's box case, but I cannot rule out that it stagnates
  too."
- **Whether the AMReX MLMG framework cleanly supports (a)** — i.e.,
  whether a per-cell-type-conditional stencil at amrlev=0 mglev=0 is
  expressible without invading other linop implementations or violating
  invariants assumed by MLMG. I would need to read more of MLMG's
  internals to commit. (b) clearly fits the framework.

## What I would actually do, if asked to commit

If the priority is correctness and you accept the implementation cost,
implement (a). It is the right answer.

If the priority is "ship something that does not diverge and is
maintainable", do **(b) + (d)**: implement flux-difference reflux
properly (matches MLABec, fixes the CF-jump artifact, improves
correctness), then wrap the whole MLMG in `GMRESMLMG` (handles the
remaining 1/β amplification via Krylov stabilisation, even if MLMG's
spectral radius stays >1 on gradient modes). This combination is well-
trodden in EM solver practice — many production codes use a Krylov
outer iteration around an MG preconditioner exactly because the MG
alone isn't contractive on the curl-curl null space, and the analytic
fix (Hiptmair smoother) is too painful.

I lean toward (b)+(d) as the pragmatic answer, while acknowledging (a)
is more mathematically principled. The user's call.
