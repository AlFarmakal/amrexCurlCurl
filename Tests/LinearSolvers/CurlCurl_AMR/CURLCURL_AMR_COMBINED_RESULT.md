# Curl-curl AMR: combining the GMRES wrap with a 2nd-order CF prolongation

This branch combines two independent pieces of work on the same base:

* the **GMRES outer wrap** for the composite solve (`composite=2`, from
  `curl-curl-gmres`), and
* a **second-order coarse→fine ghost prolongation** (`mlcurlcurl_interp_cfghost`,
  from `curlcurl-amr-tests`).

It also corrects an important misconception in the GMRES result doc.

## The misconception the combination exposes

`CURLCURL_AMR_GMRES_RESULT.md` validated `composite=2` on **residuals only**.
Measured against the manufactured ground truth, the GMRES wrap converges to the
**corner-wrong answer** — the composite operator is inconsistent at CF corners
(first-order, piecewise-constant-along-the-edge prolongation injects an O(dx)
ghost error that curl-curl's gradient near-null-space amplifies by 1/β). So a
fast residual drop hides an O(1) solution error:

```
inputs.shifted_amr,  composite=2 (GMRES, low-order operator):
    GMRES converges:  139 iters -> 1e-10 rel
    actual error:     lev1 Ex max = 47   <-- WRONG
```

Accuracy is a property of the **operator/discretisation**, not of the outer
solver. GMRES (a Krylov method) converges to the same discrete solution MLMG
would — it cannot fix an inconsistent operator. The fix has to be in the
prolongation.

## What the 2nd-order prolongation does, per solve mode

| mode | operator | result |
|------|----------|--------|
| `composite=0` (level-by-level) | **2nd-order** CF ghost | **accurate (O(dx²)) and robust** |
| `composite=2` (GMRES wrap)     | low-order CF ghost (gated off) | fast, but corner-wrong (unchanged from gmres branch) |
| `composite=1` (stationary)     | low-order | diverges on jagged CF (unchanged) |

`composite=0` + 2nd-order prolongation, manufactured `shifted`, refined:

```
n_cell   lev1 Ex err (max)     order
  64       0.00549
 128       0.00136             ~2.0
 256       0.000338            ~2.0
```

vs ~first-order (0.120 / 0.057 / 0.028) with the original prolongation, and vs
**47** for the GMRES composite path. Level-by-level also converges for the
jagged tube (2- and 3-level) in a handful of V-cycles per level.

## Why the 2nd-order prolongation is gated OFF for composite/GMRES

Dropping the consistent (2nd-order) operator into the composite path makes the
GMRES wrap **much slower and, at small β, stall** — the opposite of helpful:

```
composite=2 (GMRES), shifted:     low-order 139 it (err 47)  ->  2nd-order 108 it (err 0.5)
composite=2 (GMRES), tube tight:  low-order  27 it           ->  2nd-order 191 it
composite=2 (GMRES), 3-level tube: low-order converges       ->  2nd-order STALLS
```

β-sweep (manufactured 3-level, `composite=2`, 2nd-order operator) pins the
cause to the gradient near-null-space, not the AMR transfer order:

```
β = 1     -> 131 iters (converges)
β = 0.01  -> stalls
β = 0.001 -> stalls            (tube free space is β = 1/η_FS = 1e-3)
```

The consistent operator is accurate but **not curl-conforming**, so it injects
gradient modes; the low-order V-cycle is a poor preconditioner for them, and at
small β they dominate. Raising the AMR transfers (`reflux`/`avgDownResAmr`) to
2nd order does **not** fix this (β=1 already converges with the low-order
transfers). A composite solve that is *both* accurate *and* fast needs an
**AMS / Hiptmair-style auxiliary-space preconditioner** for the small-β
gradient space — the EM-solver state of the art, and a separate research-level
effort.

So the prolongation is enabled only where it is unambiguously beneficial:
single-level / level-by-level (`m_num_amr_levels == 1`), where the CF data is a
fixed Dirichlet BC. `composite=2` is left bit-identical to the gmres branch
(still fast, still corner-wrong) so nothing regresses.

Note: GMRES does **not** help the level-by-level path — its per-level V-cycle
already converges in a handful of iterations, GMRES cannot improve its accuracy
(same operator), and `GMRESMLMG` as written drops the external inhomogeneous
Dirichlet CF data in precond mode (it is built for composite/homogeneous
solves), producing a wrong answer if naively wrapped around an LBL level.

## Recommendation

Use **`composite=0` (level-by-level) with the 2nd-order prolongation**: it is
accurate (2nd order), robust, and converges for the jagged tube and 3 levels.
`composite=2` remains available for cases where two-way composite coupling is
wanted and the corner-region accuracy / small-β convergence limitations are
acceptable. Making composite both accurate and fast is the AMS/Hiptmair
project, scoped but not undertaken here.

See the sibling `curlcurl-amr-tests` branch / `ROOT_CAUSE_AND_FIX.md` for the
full root-cause investigation (localisation, tolerance/refinement invariance,
transfer decomposition, discriminator, β-mechanism).
