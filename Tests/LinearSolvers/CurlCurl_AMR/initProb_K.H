#ifndef INIT_PROB_K_H_
#define INIT_PROB_K_H_

#include <AMReX_FArrayBox.H>

//! β-profile selector codes — plain ints so they cross the GPU boundary.
constexpr int BetaProfileUniform   = 0;
constexpr int BetaProfileSmooth10  = 1;
constexpr int BetaProfileSmooth100 = 2;

#if (AMREX_SPACEDIM == 1)

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real eval_beta (amrex::Real x, amrex::Real /*y*/, amrex::Real /*z*/, int profile) noexcept
{
  using amrex::Real;
  constexpr Real pi = amrex::Math::pi<Real>();
  if (profile == BetaProfileSmooth10)
  {
    return Real(1.0) + Real(9.0) * Real(0.5) *
           (Real(1.0) + std::sin(Real(2.0)*pi*x));
  }
  if (profile == BetaProfileSmooth100)
  {
    return Real(1.0) + Real(99.0) * Real(0.5) *
           (Real(1.0) + std::sin(Real(2.0)*pi*x));
  }
  return Real(1.0);
}

//! 1D manufactured E:
//!   E_x = 0          (cell-centred — decoupled β·Ex = rhs identity)
//!   E_y = sin(2πx)   (nodal — pure 1D Laplacian per Ey: −α∂²Ey/∂x² + β Ey)
//!   E_z = cos(2πx)   (nodal — pure 1D Laplacian per Ez)
//! Each Ey, Ez component is exactly MLABecLaplacian per scalar component
//! in 1D (the operator collapses to −α∇² + β). The curl-curl coefficient
//! per component is 4π²α (only one non-zero 2nd derivative in 1D).
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void actual_init_prob (int i, int j, int k,
                       amrex::GpuArray<amrex::Array4<amrex::Real>,3> const& rhs,
                       amrex::GpuArray<amrex::Array4<amrex::Real>,3> const& sol,
                       amrex::GpuArray<amrex::Array4<amrex::Real>,3> const& bcoef,
                       amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> const& problo,
                       amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> const& dx,
                       amrex::Real alpha, int beta_profile)
{
  using amrex::Real;
  constexpr Real pi = amrex::Math::pi<Real>();
  amrex::ignore_unused(j, k);

  Real xnd = problo[0] + Real(i) * dx[0];
  Real xcc = xnd + Real(0.5) * dx[0];

  Real cce = Real(4.0) * pi * pi * alpha;

  // Ex edge — type (0): position xcc.
  if (sol[0].contains(i,j,k))
  {
    sol[0](i,j,k) = Real(0.0);
    Real bx = eval_beta(xcc, Real(0.0), Real(0.0), beta_profile);
    if (bcoef[0].contains(i,j,k)) bcoef[0](i,j,k) = bx;
    if (rhs[0].contains(i,j,k))   rhs[0]  (i,j,k) = Real(0.0);
  }

  // Ey edge — type (1): position xnd.
  if (sol[1].contains(i,j,k))
  {
    Real Ey = std::sin(Real(2.0)*pi*xnd);
    sol[1](i,j,k) = Ey;
    Real by = eval_beta(xnd, Real(0.0), Real(0.0), beta_profile);
    if (bcoef[1].contains(i,j,k)) bcoef[1](i,j,k) = by;
    if (rhs[1].contains(i,j,k))   rhs[1]  (i,j,k) = (cce + by) * Ey;
  }

  // Ez edge — type (1): position xnd.
  if (sol[2].contains(i,j,k))
  {
    Real Ez = std::cos(Real(2.0)*pi*xnd);
    sol[2](i,j,k) = Ez;
    Real bz = eval_beta(xnd, Real(0.0), Real(0.0), beta_profile);
    if (bcoef[2].contains(i,j,k)) bcoef[2](i,j,k) = bz;
    if (rhs[2].contains(i,j,k))   rhs[2]  (i,j,k) = (cce + bz) * Ez;
  }
}

#elif (AMREX_SPACEDIM == 2)

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real eval_beta (amrex::Real x, amrex::Real y, amrex::Real /*z*/, int profile) noexcept
{
  using amrex::Real;
  constexpr Real pi = amrex::Math::pi<Real>();
  if (profile == BetaProfileSmooth10)
  {
    return Real(1.0) + Real(9.0) * Real(0.5) *
           (Real(1.0) + std::sin(Real(2.0)*pi*x) * std::sin(Real(2.0)*pi*y));
  }
  if (profile == BetaProfileSmooth100)
  {
    return Real(1.0) + Real(99.0) * Real(0.5) *
           (Real(1.0) + std::sin(Real(2.0)*pi*x) * std::sin(Real(2.0)*pi*y));
  }
  return Real(1.0);
}

//! 2D manufactured E from F_z = sin(2πx)·sin(2πy):
//!   E_x = +2π · sin(2πx) · cos(2πy)
//!   E_y = -2π · cos(2πx) · sin(2πy)
//!   E_z = 0  (decouples in 2D)
//! Discrete operator: L(E) = α ∇×∇×E + β E = -α ∇²E + β E (∇·E = 0).
//! For scalar α, the curl-curl part is exactly 8π²α·E pointwise even
//! when β varies, so the analytic RHS is:
//!   rhs_x = (8π²α + β_x(x,y)) · E_x(x,y)
//!   rhs_y = (8π²α + β_y(x,y)) · E_y(x,y)
//!   rhs_z = 0
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void actual_init_prob (int i, int j, int k,
                       amrex::GpuArray<amrex::Array4<amrex::Real>,3> const& rhs,
                       amrex::GpuArray<amrex::Array4<amrex::Real>,3> const& sol,
                       amrex::GpuArray<amrex::Array4<amrex::Real>,3> const& bcoef,
                       amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> const& problo,
                       amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> const& dx,
                       amrex::Real alpha, int beta_profile)
{
  using amrex::Real;
  constexpr Real pi = amrex::Math::pi<Real>();

  Real xnd = problo[0] + Real(i) * dx[0];
  Real xcc = xnd + Real(0.5) * dx[0];
  Real ynd = problo[1] + Real(j) * dx[1];
  Real ycc = ynd + Real(0.5) * dx[1];
  amrex::ignore_unused(k);

  Real cce = Real(8.0) * pi * pi * alpha;

  // Ex edge — type (0,1): position (xcc, ynd).
  if (sol[0].contains(i,j,k))
  {
    Real Ex = Real(2.0) * pi * std::sin(Real(2.0)*pi*xcc) * std::cos(Real(2.0)*pi*ynd);
    sol[0](i,j,k) = Ex;
    Real bx = eval_beta(xcc, ynd, Real(0.0), beta_profile);
    if (bcoef[0].contains(i,j,k)) bcoef[0](i,j,k) = bx;
    if (rhs[0].contains(i,j,k))   rhs[0]  (i,j,k) = (cce + bx) * Ex;
  }

  // Ey edge — type (1,0): position (xnd, ycc).
  if (sol[1].contains(i,j,k))
  {
    Real Ey = -Real(2.0) * pi * std::cos(Real(2.0)*pi*xnd) * std::sin(Real(2.0)*pi*ycc);
    sol[1](i,j,k) = Ey;
    Real by = eval_beta(xnd, ycc, Real(0.0), beta_profile);
    if (bcoef[1].contains(i,j,k)) bcoef[1](i,j,k) = by;
    if (rhs[1].contains(i,j,k))   rhs[1]  (i,j,k) = (cce + by) * Ey;
  }

  // Ez edge — type (1,1): position (xnd, ynd). Decouples in 2D.
  if (sol[2].contains(i,j,k))
  {
    sol[2](i,j,k) = Real(0.0);
    if (bcoef[2].contains(i,j,k)) bcoef[2](i,j,k) = eval_beta(xnd, ynd, Real(0.0), beta_profile);
    if (rhs[2].contains(i,j,k))   rhs[2]  (i,j,k) = Real(0.0);
  }
}

#elif (AMREX_SPACEDIM == 3)

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real eval_beta (amrex::Real x, amrex::Real y, amrex::Real z, int profile) noexcept
{
  using amrex::Real;
  constexpr Real pi = amrex::Math::pi<Real>();
  if (profile == BetaProfileSmooth10)
  {
    return Real(1.0) + Real(9.0) * Real(0.5) *
           (Real(1.0) + std::sin(Real(2.0)*pi*x) *
                        std::sin(Real(2.0)*pi*y) *
                        std::sin(Real(2.0)*pi*z));
  }
  if (profile == BetaProfileSmooth100)
  {
    return Real(1.0) + Real(99.0) * Real(0.5) *
           (Real(1.0) + std::sin(Real(2.0)*pi*x) *
                        std::sin(Real(2.0)*pi*y) *
                        std::sin(Real(2.0)*pi*z));
  }
  return Real(1.0);
}

//! 3D manufactured solenoidal E:
//!   E_x = sin(2πy) cos(2πz)
//!   E_y = sin(2πz) cos(2πx)
//!   E_z = sin(2πx) cos(2πy)
//! ∇·E = 0 by construction (each component depends only on the other two).
//! ∇²E_x = -8π² · E_x (two non-zero second derivatives, each = -(2π)² · E_x);
//! same for E_y, E_z. So ∇×∇×E = -∇²E = 8π² · E pointwise.
//! Discrete RHS for L(E)=α∇×∇×E+βE:
//!   rhs_α = (8π²α + β_α(pos_α)) · E_α(pos_α).
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void actual_init_prob (int i, int j, int k,
                       amrex::GpuArray<amrex::Array4<amrex::Real>,3> const& rhs,
                       amrex::GpuArray<amrex::Array4<amrex::Real>,3> const& sol,
                       amrex::GpuArray<amrex::Array4<amrex::Real>,3> const& bcoef,
                       amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> const& problo,
                       amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> const& dx,
                       amrex::Real alpha, int beta_profile)
{
  using amrex::Real;
  constexpr Real pi = amrex::Math::pi<Real>();

  Real xnd = problo[0] + Real(i) * dx[0];
  Real xcc = xnd + Real(0.5) * dx[0];
  Real ynd = problo[1] + Real(j) * dx[1];
  Real ycc = ynd + Real(0.5) * dx[1];
  Real znd = problo[2] + Real(k) * dx[2];
  Real zcc = znd + Real(0.5) * dx[2];

  Real cce = Real(8.0) * pi * pi * alpha;

  // Ex edge — type (0,1,1): position (xcc, ynd, znd).
  if (sol[0].contains(i,j,k))
  {
    Real Ex = std::sin(Real(2.0)*pi*ynd) * std::cos(Real(2.0)*pi*znd);
    sol[0](i,j,k) = Ex;
    Real bx = eval_beta(xcc, ynd, znd, beta_profile);
    if (bcoef[0].contains(i,j,k)) bcoef[0](i,j,k) = bx;
    if (rhs[0].contains(i,j,k))   rhs[0]  (i,j,k) = (cce + bx) * Ex;
  }

  // Ey edge — type (1,0,1): position (xnd, ycc, znd).
  if (sol[1].contains(i,j,k))
  {
    Real Ey = std::sin(Real(2.0)*pi*znd) * std::cos(Real(2.0)*pi*xnd);
    sol[1](i,j,k) = Ey;
    Real by = eval_beta(xnd, ycc, znd, beta_profile);
    if (bcoef[1].contains(i,j,k)) bcoef[1](i,j,k) = by;
    if (rhs[1].contains(i,j,k))   rhs[1]  (i,j,k) = (cce + by) * Ey;
  }

  // Ez edge — type (1,1,0): position (xnd, ynd, zcc).
  if (sol[2].contains(i,j,k))
  {
    Real Ez = std::sin(Real(2.0)*pi*xnd) * std::cos(Real(2.0)*pi*ynd);
    sol[2](i,j,k) = Ez;
    Real bz = eval_beta(xnd, ynd, zcc, beta_profile);
    if (bcoef[2].contains(i,j,k)) bcoef[2](i,j,k) = bz;
    if (rhs[2].contains(i,j,k))   rhs[2]  (i,j,k) = (cce + bz) * Ez;
  }
}

#endif

// ===========================================================================
// Tube diffusion problem: faithful port of CNS-AMReX
// settings/MHDDiffusion/diffusion_nonuniform_tube physics.
// Solves L E = curl B_initial with
//   L = dt * curl(curl E) + (1/eta) * E
//   (per ImplicitFD setup: setScalars(dt, 1.0) + setBeta(1/eta))
// B_initial: Bz = B0 if r > r2 else 0; Bx = By = 0.
// eta(phi) = eta_tube / (1 - alpha*cos(phi)) inside annulus r1<r<r2,
//   eta = eta_FS elsewhere.
// Coefficients: bcoef = 1/eta at edge midpoint.
// sol = 0 (Dirichlet); no analytical exact solution.
// ===========================================================================

#if (AMREX_SPACEDIM == 2)
// eta evaluated at CELL CENTRE (i, j) with cell-centre position (xcc, ycc).
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real tube_eta (amrex::Real x, amrex::Real y,
                      amrex::Real r1, amrex::Real r2,
                      amrex::Real eta_tube, amrex::Real eta_FS,
                      amrex::Real alpha_aniso) noexcept
{
    using amrex::Real;
    Real r = std::sqrt(x*x + y*y);
    if (r > r1 && r < r2) {
        Real phi = std::atan2(y, x);
        Real denom = Real(1.0) - alpha_aniso * std::cos(phi);
        if (denom < Real(1.0e-12)) denom = Real(1.0e-12);
        return eta_tube / denom;
    }
    return eta_FS;
}

// Harmonic average of two cell-centre eta values (matches CNS-AMReX's
// CoeffAvg::harmonic_average_face). For eta=0 either side: returns 0.
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real harmonic2 (amrex::Real a, amrex::Real b) noexcept
{
    using amrex::Real;
    if (a + b <= Real(0.0)) return Real(0.0);
    return Real(2.0) * a * b / (a + b);
}

// Harmonic average over 4 cell-centre values (node-averaging, mirrors
// CoeffAvg::harmonic_average_node behaviour). 4*a*b*c*d / sum-of-products.
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real harmonic4 (amrex::Real a, amrex::Real b,
                       amrex::Real c, amrex::Real d) noexcept
{
    using amrex::Real;
    // Generalized: 4 / (1/a + 1/b + 1/c + 1/d).
    Real inv_sum = Real(0.0);
    if (a > Real(0.0)) inv_sum += Real(1.0)/a; else return Real(0.0);
    if (b > Real(0.0)) inv_sum += Real(1.0)/b; else return Real(0.0);
    if (c > Real(0.0)) inv_sum += Real(1.0)/c; else return Real(0.0);
    if (d > Real(0.0)) inv_sum += Real(1.0)/d; else return Real(0.0);
    return Real(4.0) / inv_sum;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real tube_Bz_initial (amrex::Real x, amrex::Real y,
                             amrex::Real r2, amrex::Real B0) noexcept
{
    using amrex::Real;
    Real r = std::sqrt(x*x + y*y);
    return (r > r2) ? B0 : Real(0.0);
}

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void tube_init_prob (int i, int j, int k,
                     amrex::GpuArray<amrex::Array4<amrex::Real>,3> const& rhs,
                     amrex::GpuArray<amrex::Array4<amrex::Real>,3> const& sol,
                     amrex::GpuArray<amrex::Array4<amrex::Real>,3> const& bcoef,
                     amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> const& problo,
                     amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> const& dx,
                     amrex::Real r1, amrex::Real r2,
                     amrex::Real eta_tube, amrex::Real eta_FS,
                     amrex::Real alpha_aniso, amrex::Real B0)
{
    using amrex::Real;
    amrex::ignore_unused(k);
    Real xnd = problo[0] + Real(i) * dx[0];
    Real xcc = xnd + Real(0.5) * dx[0];
    Real ynd = problo[1] + Real(j) * dx[1];
    Real ycc = ynd + Real(0.5) * dx[1];

    // Cell-centred eta at the 4 cells around this Yee node (i, j):
    //   center cell (i,j)   at (xcc, ycc)        = (xnd + dx/2, ynd + dy/2)
    //   south  cell (i,j-1) at (xcc, ycc - dy)
    //   west   cell (i-1,j) at (xcc - dx, ycc)
    //   SW     cell (i-1,j-1) at (xcc - dx, ycc - dy)
    Real const eta_C  = tube_eta(xcc,         ycc,         r1, r2, eta_tube, eta_FS, alpha_aniso);
    Real const eta_S  = tube_eta(xcc,         ycc - dx[1], r1, r2, eta_tube, eta_FS, alpha_aniso);
    Real const eta_W  = tube_eta(xcc - dx[0], ycc,         r1, r2, eta_tube, eta_FS, alpha_aniso);
    Real const eta_SW = tube_eta(xcc - dx[0], ycc - dx[1], r1, r2, eta_tube, eta_FS, alpha_aniso);

    // Ex edge — type (0,1): position (xcc, ynd). Edge spans cells C (above)
    // and S (below). Harmonic average of eta across those two cells.
    if (sol[0].contains(i,j,k))
    {
        sol[0](i,j,k) = Real(0.0);
        Real eta_edge = harmonic2(eta_C, eta_S);
        if (bcoef[0].contains(i,j,k)) {
            bcoef[0](i,j,k) = (eta_edge > Real(0.0)) ? Real(1.0)/eta_edge : Real(1.0e30);
        }
        if (rhs[0].contains(i,j,k)) {
            Real Bz_above = tube_Bz_initial(xcc, ycc,         r2, B0);
            Real Bz_below = tube_Bz_initial(xcc, ycc - dx[1], r2, B0);
            rhs[0](i,j,k) = (Bz_above - Bz_below) / dx[1];
        }
    }
    // Ey edge — type (1,0): position (xnd, ycc). Edge spans cells C (east)
    // and W (west). Harmonic average.
    if (sol[1].contains(i,j,k))
    {
        sol[1](i,j,k) = Real(0.0);
        Real eta_edge = harmonic2(eta_C, eta_W);
        if (bcoef[1].contains(i,j,k)) {
            bcoef[1](i,j,k) = (eta_edge > Real(0.0)) ? Real(1.0)/eta_edge : Real(1.0e30);
        }
        if (rhs[1].contains(i,j,k)) {
            Real Bz_east = tube_Bz_initial(xcc,         ycc, r2, B0);
            Real Bz_west = tube_Bz_initial(xcc - dx[0], ycc, r2, B0);
            rhs[1](i,j,k) = -(Bz_east - Bz_west) / dx[0];
        }
    }
    // Ez edge — type (1,1): position (xnd, ynd). 4-cell harmonic average.
    if (sol[2].contains(i,j,k))
    {
        sol[2](i,j,k) = Real(0.0);
        Real eta_node = harmonic4(eta_C, eta_S, eta_W, eta_SW);
        if (bcoef[2].contains(i,j,k)) {
            bcoef[2](i,j,k) = (eta_node > Real(0.0)) ? Real(1.0)/eta_node : Real(1.0e30);
        }
        if (rhs[2].contains(i,j,k))   rhs[2](i,j,k) = Real(0.0);
    }
}
#endif

#endif
