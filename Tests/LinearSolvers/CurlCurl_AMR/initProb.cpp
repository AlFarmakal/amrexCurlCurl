#include "initProb_K.H"

#include "MyTest.H"

using namespace amrex;

void
MyTest::initProb ()
{
  const bool is_tube = (problem == "tube");

  int profile_code = BetaProfileUniform;
  if (beta_profile == "smooth_10x")  profile_code = BetaProfileSmooth10;
  if (beta_profile == "smooth_100x") profile_code = BetaProfileSmooth100;

  const int nlevels = static_cast<int>(geom.size());
  for (int lev = 0; lev < nlevels; ++lev)
  {
    const auto prob_lo = geom[lev].ProbLoArray();
    const auto dx      = geom[lev].CellSizeArray();
    const auto a       = alpha;
    const int  pcode   = profile_code;
    // Tube parameters (captured by lambda).
    const Real r1 = tube_r1;
    const Real r2 = tube_r2;
    const Real eta_t = tube_eta;
    const Real eta_FS = tube_eta_FS;
    const Real al_an = tube_alpha_aniso;
    const Real B0 = tube_B0;

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(rhs[lev][0], TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
      // Grow by 1 in valid edge directions to also fill exact-MF ghosts.
      const Box& gbx = mfi.tilebox(IntVect(1), IntVect(1));
      GpuArray<Array4<Real>,3> rhsfab{rhs[lev][0].array(mfi),
                                      rhs[lev][1].array(mfi),
                                      rhs[lev][2].array(mfi)};
      GpuArray<Array4<Real>,3> solfab{solution[lev][0].array(mfi),
                                      solution[lev][1].array(mfi),
                                      solution[lev][2].array(mfi)};
      GpuArray<Array4<Real>,3> bcfab {bcoef[lev][0].array(mfi),
                                      bcoef[lev][1].array(mfi),
                                      bcoef[lev][2].array(mfi)};
      if (is_tube) {
#if (AMREX_SPACEDIM == 2)
        amrex::ParallelFor(gbx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
          tube_init_prob(i, j, k, rhsfab, solfab, bcfab, prob_lo, dx,
                         r1, r2, eta_t, eta_FS, al_an, B0);
        });
#else
        amrex::Abort("Tube test is 2D-only");
#endif
      } else {
        amrex::ParallelFor(gbx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
          actual_init_prob(i, j, k, rhsfab, solfab, bcfab, prob_lo, dx, a, pcode);
        });
      }
    }
  }
}
