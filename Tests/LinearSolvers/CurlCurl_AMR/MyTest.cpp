// Curl-curl AMR solver test driver.
//
// Two solve modes, selected by ParmParse top-level `composite` (default = 1):
//   composite=1  MLMG composite cross-level V-cycle (default).
//   composite=0  Per-level solves: coarse first, then each fine level
//                with the next-coarser solution as Dirichlet CF data.
//
// AMR scenarios are driven by `inputs.*` files in this directory.

#include <AMReX_MLCurlCurl.H>

#include "MyTest.H"
#include "initProb_K.H"

#include <AMReX_GMRES_MLMG.H>
#include <AMReX_MLABecLaplacian.H>
#include <AMReX_MLMG.H>
#include <AMReX_ParmParse.H>
#include <AMReX_PlotFileUtil.H>
#include <AMReX_Utility.H>

#include <fstream>

using namespace amrex;

MyTest::MyTest ()
{
  readParameters();
  initData();
}

void
MyTest::solve ()
{
  using V = Array<MultiFab,3>;

  LPInfo info;
  info.setAgglomeration(agglomeration);
  info.setConsolidation(consolidation);
  info.setMaxCoarseningLevel(max_coarsening_level);

  MLCurlCurl mlcc(geom, grids, dmap, info);

  bool const is_tube = (problem == "tube");

  // Tube: Dirichlet 0 BC. Manufactured: periodic.
  LinOpBCType const bc_type = is_tube ? LinOpBCType::Dirichlet
                                       : LinOpBCType::Periodic;
  Array<LinOpBCType,AMREX_SPACEDIM> lobc{AMREX_D_DECL(bc_type, bc_type, bc_type)};
  Array<LinOpBCType,AMREX_SPACEDIM> hibc = lobc;
  mlcc.setDomainBC(lobc, hibc);

  // Diagnostic toggle: scalar-β bypasses the m_bcoefs path entirely.
  // Only meaningful when beta_profile == "uniform" (since the test's
  // analytic rhs assumes the uniform value).
  bool use_scalar_beta = false;
  {
    ParmParse ppd;
    ppd.query("use_scalar_beta", use_scalar_beta);
  }
  Real beta_scalar = Real(1.0);
  { ParmParse ppb; ppb.query("beta_scalar", beta_scalar); }
  amrex::Print() << "[MyTest] alpha=" << alpha
                 << " beta_scalar=" << beta_scalar << "\n";
  // Tube convention: setScalars(dt, 1.0) — m_alpha = dt (scalar curl curl
  // coeff), m_beta = 1.0 (variable bcoef = 1/eta provides per-edge β).
  // Manufactured convention: setScalars(alpha, 1.0) — alpha scales curl curl,
  // m_beta=1.0 is uniform on top of bcoef = 1.0.
  if (is_tube) {
      mlcc.setScalars(tube_dt, Real(1.0));
      amrex::Print() << "[MyTest tube] dt=" << tube_dt
                     << " eta_tube=" << tube_eta
                     << " eta_FS=" << tube_eta_FS
                     << " alpha_aniso=" << tube_alpha_aniso
                     << " r1=" << tube_r1 << " r2=" << tube_r2 << "\n";
  } else {
      mlcc.setScalars(alpha, Real(1.0));
  }
  if (beta_scalar != Real(1.0)) {
      for (int lev = 0; lev < (int)geom.size(); ++lev) {
          for (int idim = 0; idim < 3; ++idim) {
              bcoef[lev][idim].mult(beta_scalar, 0, 1,
                                    bcoef[lev][idim].nGrow());
              // rhs originally = (8π²α + 1)·E_exact. With β scaled to
              // beta_scalar, rhs must be (8π²α + beta_scalar)·E_exact.
              // Correction: rhs += (beta_scalar - 1)·E_exact.
              MultiFab::Saxpy(rhs[lev][idim], beta_scalar - Real(1.0),
                              exact[lev][idim], 0, 0, 1, IntVect(0));
          }
      }
  }

  if (!use_scalar_beta)
  {
    Vector<Array<MultiFab const*,3>> bcoef_ptrs(geom.size());
    for (int lev = 0; lev < (int)geom.size(); ++lev)
    {
      bcoef_ptrs[lev] = {&bcoef[lev][0], &bcoef[lev][1], &bcoef[lev][2]};
    }
    mlcc.setBeta(bcoef_ptrs);
  }

  for (int lev = 0; lev < (int)geom.size(); ++lev)
  {
    mlcc.setLevelBC(lev, &solution[lev]);
  }

  Vector<V*> rhs_ptrs(geom.size());
  for (int lev = 0; lev < (int)geom.size(); ++lev) { rhs_ptrs[lev] = &rhs[lev]; }
  mlcc.prepareRHS(rhs_ptrs);

  MLMGT<V> mlmg(mlcc);
  mlmg.setMaxIter(max_iter);
  mlmg.setVerbose(verbose);
  mlmg.setBottomVerbose(bottom_verbose);
  // Bottom-solver iteration cap. Default in MLMG is ~200; tube problem
  // with extreme eta contrast needs more — set via parmparse.
  // Also allow switching bottom solver kind: bicgstab (default), smoother,
  // cg, etc. For very ill-conditioned operators, "smoother" can be more
  // robust than BiCGStab.
  {
    int bottom_max_iter = -1;
    Real bottom_tol = -1.0;
    std::string bottom_solver = "";
    ParmParse pp;
    pp.query("bottom_max_iter", bottom_max_iter);
    pp.query("bottom_tol", bottom_tol);
    pp.query("bottom_solver", bottom_solver);
    if (bottom_max_iter > 0) mlmg.setBottomMaxIter(bottom_max_iter);
    if (bottom_tol > 0)      mlmg.setBottomTolerance(bottom_tol);
    if (bottom_solver == "smoother") mlmg.setBottomSolver(BottomSolver::smoother);
    else if (bottom_solver == "cg")  mlmg.setBottomSolver(BottomSolver::cg);
    else if (bottom_solver == "bicgstab") mlmg.setBottomSolver(BottomSolver::bicgstab);
    else if (bottom_solver == "cgbicg") mlmg.setBottomSolver(BottomSolver::cgbicg);
  }
  // Smoother-sweep tuning (curl-curl V-cycle convergence is slow; experiment
  // with bigger nu1/nu2 to see if it stabilises composite).
  {
    int nu1 = 2, nu2 = 2, nub = 0, nuf = 0;
    ParmParse pp;
    pp.query("nu1", nu1);
    pp.query("nu2", nu2);
    pp.query("nub", nub);
    pp.query("nuf", nuf);
    mlmg.setPreSmooth(nu1);
    mlmg.setPostSmooth(nu2);
    if (nub > 0) mlmg.setBottomSmooth(nub);
    if (nuf > 0) mlmg.setFinalSmooth(nuf);
  }

  // Diagnostic: optionally start from the exact solution. If exact is a
  // fixed point of the discrete composite system, residual stays at the
  // (small) per-level discretisation residual and iter count drops to 0.
  // Set init_from_exact=1 in the inputs file to enable.
  bool init_from_exact = false;
  {
    ParmParse ppx;
    ppx.query("init_from_exact", init_from_exact);
  }
  for (int lev = 0; lev < (int)geom.size(); ++lev)
  {
    if (init_from_exact) {
      for (int idim = 0; idim < 3; ++idim) {
        solution[lev][idim].LocalCopy(exact[lev][idim], 0, 0, 1, IntVect(1));
      }
    } else {
      for (auto& mf : solution[lev]) { mf.setVal(Real(0.0)); }
    }
  }

  Vector<V*>       sol_ptrs(geom.size());
  Vector<V const*> rhs_const_ptrs(geom.size());
  for (int lev = 0; lev < (int)geom.size(); ++lev)
  {
    sol_ptrs[lev]       = &solution[lev];
    rhs_const_ptrs[lev] = &rhs[lev];
  }

  // Bz at cell centres before the timestep (initial condition). For the
  // tube test this is the prescribed analytic field tube_Bz_initial; for
  // other problems we just write 0.
  Vector<MultiFab> bz_initial(geom.size());
  {
    Real const r2  = tube_r2;
    Real const B0  = tube_B0;
    bool const is_tube_local = is_tube;
    for (int lev = 0; lev < (int)geom.size(); ++lev) {
      bz_initial[lev].define(grids[lev], dmap[lev], 1, 0);
      bz_initial[lev].setVal(0.0);
      if (is_tube_local) {
        auto const dx      = geom[lev].CellSizeArray();
        auto const problo  = geom[lev].ProbLoArray();
        auto const& bz_a   = bz_initial[lev].arrays();
        ParallelFor(bz_initial[lev],
            [=] AMREX_GPU_DEVICE (int bno, int i, int j, int k)
        {
          Real x = problo[0] + (Real(i) + Real(0.5)) * dx[0];
          Real y = problo[1] + (Real(j) + Real(0.5)) * dx[1];
          bz_a[bno](i, j, k) = tube_Bz_initial(x, y, r2, B0);
        });
      }
    }
    Gpu::streamSynchronize();
  }

  // Pre-solve plot: sol=0, residual=rhs (since L(0)=0). Bz = initial.
  if (!plot_dir.empty()) {
    writePlotfile("initial", solution, exact, rhs, rhs, &bz_initial);
  }

  // Force MLMG-internal state (cfmask, fine_mask, lusolver) to be built so
  // diagnostic residuals see the same operator MLMG actually drives.
  mlcc.prepareForSolve();

  // Diagnostic: mimic MLMG's exact pre-iteration sequence on the user's
  // solution[] arrays in-place (after possibly being set to exact above).
  // MLMG aliases sol[alev] to user's solution[alev], so MLMG's
  // averageDownSolutionRHS modifies the same memory. We're allowed to
  // mutate the same way here for diagnostic purposes — solution[] gets
  // reset to zero or exact below.
  bool const run_mlmg_mimic = true;
  if (run_mlmg_mimic && geom.size() > 1)
  {
    Vector<V> diag_sol(geom.size());
    Vector<V> diag_rhs(geom.size());
    Vector<V> diag_resid(geom.size());
    for (int lev = 0; lev < (int)geom.size(); ++lev)
    {
      for (int idim = 0; idim < 3; ++idim) {
        diag_sol[lev][idim].define(solution[lev][idim].boxArray(),
                                   solution[lev][idim].DistributionMap(),
                                   1, IntVect(1));
        diag_sol[lev][idim].LocalCopy(solution[lev][idim], 0, 0, 1, IntVect(1));
        diag_rhs[lev][idim].define(rhs[lev][idim].boxArray(),
                                   rhs[lev][idim].DistributionMap(),
                                   1, IntVect(0));
        diag_rhs[lev][idim].LocalCopy(rhs[lev][idim], 0, 0, 1, IntVect(0));
        diag_resid[lev][idim].define(rhs[lev][idim].boxArray(),
                                     rhs[lev][idim].DistributionMap(),
                                     1, IntVect(1));
        diag_resid[lev][idim].setBndry(Real(0.0));
      }
    }
    // Step 1: averageDownSolutionRHS, mimicking MLMG prepareForSolve.
    for (int falev = (int)geom.size() - 1; falev > 0; --falev) {
      mlcc.averageDownSolutionRHS(falev-1, diag_sol[falev-1], diag_rhs[falev-1],
                                  diag_sol[falev], diag_rhs[falev]);
    }
    auto probe_Ey16_16 = [&](char const* tag) {
#if (AMREX_SPACEDIM == 2)
      for (MFIter mfi(diag_resid[1][1]); mfi.isValid(); ++mfi) {
        Array4<Real const> const& a = diag_resid[1][1].const_array(mfi);
        if (mfi.validbox().contains(IntVect(16,16))) {
          amrex::AllPrint() << "    [TRACE " << tag << "] diag_resid[1][1](16,16) = "
                            << a(16,16,0) << '\n';
        }
      }
#else
      amrex::ignore_unused(tag);
#endif
    };
    // Step 2: solutionResidual on each level finest → coarsest.
    for (int lev = (int)geom.size() - 1; lev >= 0; --lev) {
      const V* crse_bcdata = (lev > 0) ? &diag_sol[lev-1] : nullptr;
      mlcc.solutionResidual(lev, diag_resid[lev], diag_sol[lev],
                            diag_rhs[lev], crse_bcdata);
      probe_Ey16_16(amrex::Concatenate("after solRes lev=", lev).c_str());
    }
    // Step 3: reflux at coarse level (matching computeMLResidual).
    if (geom.size() > 1) {
      mlcc.reflux(0, diag_resid[0], diag_sol[0], diag_rhs[0],
                  diag_resid[1], diag_sol[1], diag_rhs[1]);
      probe_Ey16_16("after reflux");
    }

    // Print per-level, per-component norms — over ALL cells AND over uncovered
    // (mask-aware, what MLMG actually drives).
    constexpr int n_active = AMREX_SPACEDIM;
    Array<std::string,3> names_all{"Ex", "Ey", "Ez"};
    for (int lev = 0; lev < (int)geom.size(); ++lev)
    {
      for (int idim = 0; idim < n_active; ++idim) {
        auto allmax = diag_resid[lev][idim].norminf();
        amrex::Print() << "  MLMG-mimic-resid lev " << lev << " " << names_all[idim]
                       << " allcells max: " << allmax << '\n';
      }
    }
#if (AMREX_SPACEDIM == 2)
    // Probe specific Ey residual values at CF-adjacent cells (2D shifted_amr).
    if (geom.size() > 1) {
      for (MFIter mfi(diag_resid[1][1]); mfi.isValid(); ++mfi) {
        Array4<Real const> const& a = diag_resid[1][1].const_array(mfi);
        Box const& vbx = mfi.validbox();
        for (int ii : {16, 17, 18, 79, 80}) {
          for (int jj : {16, 17, 18, 79}) {
            if (vbx.contains(IntVect(ii,jj))) {
              amrex::AllPrint() << "    [diag_resid Ey] (" << ii << "," << jj
                                << ") = " << a(ii,jj,0) << '\n';
            }
          }
        }
      }
    }
#endif
    // Use linop's masked normInf (same call MLMG makes).
    for (int lev = 0; lev < (int)geom.size(); ++lev)
    {
      auto m = mlcc.normInf(lev, diag_resid[lev], false);
      amrex::Print() << "  MLMG-mimic-resid lev " << lev
                     << " linop.normInf (masked): " << m << '\n';
    }
  }

  // Diagnostic: per-level residual evaluated at the manufactured exact
  // solution. Tests whether the per-level discrete operator is consistent
  // with the PDE. If small (O(dx^2)) on both levels, the per-level operators
  // and CF ghost interp are right; the bug must live in AMR coupling
  // (reflux / avgDownResAmr / averageDownAndSync). If large, the per-level
  // operator (or CF interp from coarse-exact) is broken.
  {
    Vector<V> exact_copy(geom.size());
    Vector<V> rhs_copy(geom.size());
    Vector<V> exact_resid(geom.size());
    for (int lev = 0; lev < (int)geom.size(); ++lev)
    {
      for (int idim = 0; idim < 3; ++idim) {
        exact_copy[lev][idim].define(exact[lev][idim].boxArray(),
                                     exact[lev][idim].DistributionMap(), 1, 1);
        exact_copy[lev][idim].LocalCopy(exact[lev][idim], 0, 0, 1, IntVect(1));
        rhs_copy[lev][idim].define(rhs[lev][idim].boxArray(),
                                   rhs[lev][idim].DistributionMap(), 1, 0);
        rhs_copy[lev][idim].LocalCopy(rhs[lev][idim], 0, 0, 1, IntVect(0));
        exact_resid[lev][idim].define(rhs[lev][idim].boxArray(),
                                      rhs[lev][idim].DistributionMap(), 1, 0);
      }
    }
    // Mimic mlmg's prepareForSolve normalization: avg-down sol[covered]
    // and rhs[covered] from finer levels. mlmg does this BEFORE residual.
    for (int falev = (int)geom.size() - 1; falev > 0; --falev) {
      mlcc.averageDownSolutionRHS(falev-1,
                                  exact_copy[falev-1], rhs_copy[falev-1],
                                  exact_copy[falev],   rhs_copy[falev]);
    }
    for (int lev = 0; lev < (int)geom.size(); ++lev)
    {
      const V* crse_bcdata = (lev > 0) ? &exact_copy[lev-1] : nullptr;
      mlcc.solutionResidual(lev, exact_resid[lev], exact_copy[lev],
                            rhs_copy[lev], crse_bcdata);
      auto dvol = AMREX_D_TERM(geom[lev].CellSize(0), * geom[lev].CellSize(1), * geom[lev].CellSize(2));
      Array<std::string,3> names_all{"Ex","Ey","Ez"};
      for (int idim = 0; idim < 3; ++idim) {
        auto e0 = exact_resid[lev][idim].norminf();
        auto e2 = exact_resid[lev][idim].norm2(0, geom[lev].periodicity());
        e2 *= std::sqrt(dvol);
        amrex::Print() << "  EXACT-residual lev " << lev << " " << names_all[idim]
                       << " (max, L2): " << e0 << "  " << e2 << '\n';
      }
    }

    // Composite residual at exact: mimic mlmg.computeMLResidual — bare
    // per-level then reflux on the coarse residual at uncovered ring.
    if (geom.size() > 1)
    {
      Vector<V> exact_resid_comp(geom.size());
      for (int lev = 0; lev < (int)geom.size(); ++lev) {
        for (int idim = 0; idim < 3; ++idim) {
          // ng=1 required: reflux's applyBC(homog) iterates with IntVect(1).
          exact_resid_comp[lev][idim].define(rhs[lev][idim].boxArray(),
                                             rhs[lev][idim].DistributionMap(),
                                             1, IntVect(1));
          exact_resid_comp[lev][idim].setBndry(Real(0.0));
          exact_resid_comp[lev][idim].LocalCopy(exact_resid[lev][idim],
                                                0, 0, 1, IntVect(0));
        }
      }
      mlcc.reflux(0, exact_resid_comp[0], exact_copy[0], rhs_copy[0],
                  exact_resid_comp[1], exact_copy[1], rhs_copy[1]);
      auto dvol0 = AMREX_D_TERM(geom[0].CellSize(0), * geom[0].CellSize(1), * geom[0].CellSize(2));
      Array<std::string,3> names_all{"Ex","Ey","Ez"};
      for (int idim = 0; idim < 3; ++idim) {
        auto e0 = exact_resid_comp[0][idim].norminf();
        auto e2 = exact_resid_comp[0][idim].norm2(0, geom[0].periodicity());
        e2 *= std::sqrt(dvol0);
        amrex::Print() << "  COMPOSITE-residual-at-exact lev 0 " << names_all[idim]
                       << " (max, L2): " << e0 << "  " << e2 << '\n';
      }
    }
  }

  Real tol_abs = Real(0.0);

  // Solve mode: composite (default), level-by-level, or GMRES-wrapped.
  // composite=2: GMRES outer iteration with MLMG as preconditioner.
  //   Stabilizes the iteration on near-singular curl-curl operators
  //   (β << 1 in free space) where the bare MLMG V-cycle has spectral
  //   radius > 1 along gradient modes. Pair with the Galerkin env vars
  //   AMREX_MLCC_AMR_GALERKIN=1 and AMREX_MLCC_SMOOTH_COV_GALERKIN=1
  //   so the preconditioner direction is variationally consistent.
  // composite=1: existing MLMG composite cross-AMR V-cycle.
  // composite=0: per-level single-AMR-level solves; coarse first, then each
  //   fine level with setCoarseFineBC pointing to the just-solved coarser
  //   level. Mirrors CNS DiffusiveMethod::globalFieldCompositeCurlCurlSolve=0.
  int composite = 1;
  {
    ParmParse pp;
    pp.query("composite", composite);
  }

  if (composite == 2) {
    GMRESMLMGT<V> gmsolver(mlmg);
    gmsolver.setVerbose(verbose);
    {
      int gmres_precond_iters = 1;
      int gmres_max_iters = max_iter;
      // Default Krylov restart scales with AMR depth: each extra AMR
      // level coarsening adds spectral spread to the MLMG-preconditioned
      // operator (curl-curl gradient modes amplify ~1/β = 1000 per
      // coarsening), so the Krylov subspace needs to be longer to damp
      // the worst-conditioned modes between restarts. GMRES default is
      // 30, which is plenty for 2 AMR levels but stagnates badly for 3+.
      // Cap at 200 — beyond that the orthogonalisation cost dominates.
      int gmres_restart = std::min(200, std::max(30, 100 * (int)geom.size()));
      ParmParse pp;
      pp.query("gmres_precond_iters", gmres_precond_iters);
      pp.query("gmres_max_iters",      gmres_max_iters);
      pp.query("gmres_restart",        gmres_restart);
      gmsolver.setPrecondNumIters(gmres_precond_iters);
      gmsolver.setMaxIters(gmres_max_iters);
      gmsolver.setRestartLength(gmres_restart);
    }
    gmsolver.solve(sol_ptrs, rhs_const_ptrs, tol_rel, tol_abs);
  } else if (composite == 1) {
    mlmg.solve(sol_ptrs, rhs_const_ptrs, tol_rel, tol_abs);
  } else {
    for (int lev = 0; lev < (int)geom.size(); ++lev)
    {
      LPInfo lev_info;
      lev_info.setAgglomeration(agglomeration);
      lev_info.setConsolidation(consolidation);
      lev_info.setMaxCoarseningLevel(max_coarsening_level);

      Vector<Geometry>            lev_geom{geom[lev]};
      Vector<BoxArray>            lev_grids{grids[lev]};
      Vector<DistributionMapping> lev_dmap{dmap[lev]};

      MLCurlCurl mlcc_lvl(lev_geom, lev_grids, lev_dmap, lev_info);
      mlcc_lvl.setDomainBC(lobc, hibc);
      mlcc_lvl.setScalars(is_tube ? tube_dt : alpha, Real(1.0));

      if (!use_scalar_beta) {
        Vector<Array<MultiFab const*,3>> bc_ptr(1);
        bc_ptr[0] = {&bcoef[lev][0], &bcoef[lev][1], &bcoef[lev][2]};
        mlcc_lvl.setBeta(bc_ptr);
      }

      // Prescribe coarse-side Dirichlet from the just-solved coarser level.
      if (lev > 0) {
        mlcc_lvl.setCoarseFineBC(&solution[lev-1], ref_ratio);
      }
      mlcc_lvl.setLevelBC(0, nullptr);

      Vector<V*> rhs_lvl_ptrs(1);
      rhs_lvl_ptrs[0] = &rhs[lev];
      mlcc_lvl.prepareRHS(rhs_lvl_ptrs);

      MLMGT<V> mlmg_lvl(mlcc_lvl);
      mlmg_lvl.setMaxIter(max_iter);
      mlmg_lvl.setVerbose(verbose);
      mlmg_lvl.setBottomVerbose(bottom_verbose);
      {
        int bottom_max_iter_lvl = -1;
        Real bottom_tol_lvl = -1.0;
        std::string bottom_solver_lvl = "";
        int nu1_lvl = 2, nu2_lvl = 2, nub_lvl = 0, nuf_lvl = 0;
        ParmParse pp;
        pp.query("bottom_max_iter", bottom_max_iter_lvl);
        pp.query("bottom_tol", bottom_tol_lvl);
        pp.query("bottom_solver", bottom_solver_lvl);
        pp.query("nu1", nu1_lvl);
        pp.query("nu2", nu2_lvl);
        pp.query("nub", nub_lvl);
        pp.query("nuf", nuf_lvl);
        if (bottom_max_iter_lvl > 0) mlmg_lvl.setBottomMaxIter(bottom_max_iter_lvl);
        if (bottom_tol_lvl > 0)      mlmg_lvl.setBottomTolerance(bottom_tol_lvl);
        if (bottom_solver_lvl == "smoother") mlmg_lvl.setBottomSolver(BottomSolver::smoother);
        else if (bottom_solver_lvl == "cg")  mlmg_lvl.setBottomSolver(BottomSolver::cg);
        else if (bottom_solver_lvl == "bicgstab") mlmg_lvl.setBottomSolver(BottomSolver::bicgstab);
        else if (bottom_solver_lvl == "cgbicg") mlmg_lvl.setBottomSolver(BottomSolver::cgbicg);
        mlmg_lvl.setPreSmooth(nu1_lvl);
        mlmg_lvl.setPostSmooth(nu2_lvl);
        if (nub_lvl > 0) mlmg_lvl.setBottomSmooth(nub_lvl);
        if (nuf_lvl > 0) mlmg_lvl.setFinalSmooth(nuf_lvl);
      }

      Vector<V*>       sol_lvl{&solution[lev]};
      Vector<V const*> rhs_lvl{&rhs[lev]};
      amrex::Print() << "Level-by-level solve: amrlev=" << lev << '\n';
      mlmg_lvl.solve(sol_lvl, rhs_lvl, tol_rel, tol_abs);
    }
  }

  // Per-AMR-level residual = rhs - L(sol). MLMGT::apply / compResidual are
  // not instantiated for the Array<MultiFab,3> MF type used by MLCurlCurl,
  // and the linop's per-level solutionResidual is what mlmg actually drives,
  // so it is the most faithful diagnostic anyway.
  Vector<V> residual_storage(geom.size());
  for (int lev = 0; lev < (int)geom.size(); ++lev)
  {
    for (int idim = 0; idim < 3; ++idim) {
      residual_storage[lev][idim].define(rhs[lev][idim].boxArray(),
                                         rhs[lev][idim].DistributionMap(), 1, 0);
    }
    const V* crse_bcdata = (lev > 0) ? &solution[lev-1] : nullptr;
    mlcc.solutionResidual(lev, residual_storage[lev], solution[lev],
                          rhs[lev], crse_bcdata);
  }

  // Bz at end of timestep: Bz_new = Bz_old - dt * curl(E)_z, where
  // curl(E)_z at cell centre (i,j) = (Ey(i+1,j) - Ey(i,j))/dx
  //                                  - (Ex(i,j+1) - Ex(i,j))/dy.
  // dt = tube_dt for the tube test (matches MLMG's m_alpha). For non-tube
  // problems we still emit the curl_z*dt update with dt=tube_dt so the
  // user gets a meaningful "rate" view.
  Vector<MultiFab> bz_final(geom.size());
  {
    Real const dt = tube_dt;
    for (int lev = 0; lev < (int)geom.size(); ++lev) {
      bz_final[lev].define(grids[lev], dmap[lev], 1, 0);
      auto const dxinv = geom[lev].InvCellSizeArray();
      auto const& bzin_a = bz_initial[lev].const_arrays();
      auto const& bzfn_a = bz_final[lev].arrays();
      auto const& ex_a   = solution[lev][0].const_arrays();
      auto const& ey_a   = solution[lev][1].const_arrays();
      ParallelFor(bz_final[lev],
          [=] AMREX_GPU_DEVICE (int bno, int i, int j, int k)
      {
        Real const curl_z =
              (ey_a[bno](i+1, j  , k) - ey_a[bno](i, j, k)) * dxinv[0]
            - (ex_a[bno](i  , j+1, k) - ex_a[bno](i, j, k)) * dxinv[1];
        bzfn_a[bno](i, j, k) = bzin_a[bno](i, j, k) - dt * curl_z;
      });
    }
    Gpu::streamSynchronize();
  }

  if (!plot_dir.empty()) {
    writePlotfile("final", solution, exact, rhs, residual_storage, &bz_final);
    // Write a VisIt .visit movie file pointing at plt_initial then plt_final
    // so VisIt cycles through the two states.
    if (ParallelDescriptor::IOProcessor()) {
      std::ofstream vfs(plot_dir + "/movie.visit");
      vfs << "plt_initial/Header\n" << "plt_final/Header\n";
    }
    ParallelDescriptor::Barrier();
  }

  // Ghost-fill diagnostic. For each level, sample sol valid+ghost values at
  // boundary cells and report whether the ghost-fill mechanism produces
  // consistent values at: (a) domain Dirichlet boundary, (b) internal CF
  // boundary, (c) corner where both meet. Enable with print_ghost_diag=1.
  {
    bool print_ghost_diag = false;
    ParmParse ppgd;
    ppgd.query("print_ghost_diag", print_ghost_diag);
    if (print_ghost_diag) {
      for (int lev = 0; lev < (int)geom.size(); ++lev) {
        Box const dom = geom[lev].Domain();
        IntVect const dlo = dom.smallEnd();
        IntVect const dhi = dom.bigEnd();
        amrex::Print() << "\n=== Ghost-fill diag lev " << lev
                       << " domain=[" << dlo[0] << "," << dlo[1]
                       << "]..[" << dhi[0] << "," << dhi[1] << "] ===\n";
        Array<std::string,3> names_all{"Ex", "Ey", "Ez"};
        for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
          auto const& mf = solution[lev][idim];
          Box const edom = amrex::convert(dom, mf.ixType());
          IntVect const elo = edom.smallEnd();
          IntVect const ehi = edom.bigEnd();
          amrex::Print() << "-- " << names_all[idim] << " edge domain=["
                         << elo[0] << "," << elo[1] << "]..[" << ehi[0]
                         << "," << ehi[1] << "], type=" << mf.ixType() << "\n";
          // Walk MFIter, find boxes touching domain corner / CF corner.
          for (MFIter mfi(mf); mfi.isValid(); ++mfi) {
            Box const& vbx = mfi.validbox();
            Box const gbx = amrex::grow(vbx, 1);
            Array4<Real const> const& a = mf.const_array(mfi);
            auto sample = [&](char const* tag, int i, int j) {
              if (gbx.contains(IntVect(i,j))) {
                int valid = vbx.contains(IntVect(i,j)) ? 1 : 0;
                int in_edom = edom.contains(IntVect(i,j)) ? 1 : 0;
                amrex::AllPrint() << "  [" << tag << "] ("
                                  << i << "," << j << ") val=" << a(i,j,0)
                                  << " (valid=" << valid
                                  << ", in_edge_domain=" << in_edom << ")\n";
              }
            };
            // Domain lo-x boundary stripe (i=elo[0]): edges at left
            if (vbx.smallEnd(0) == elo[0]) {
              sample("dom-lo-x valid", elo[0],   vbx.smallEnd(1));
              sample("dom-lo-x ghost", elo[0]-1, vbx.smallEnd(1));
            }
            // Domain lo-y boundary stripe (j=elo[1]): edges at bottom
            if (vbx.smallEnd(1) == elo[1]) {
              sample("dom-lo-y valid",  vbx.smallEnd(0), elo[1]);
              sample("dom-lo-y ghost",  vbx.smallEnd(0), elo[1]-1);
            }
            // Patch hi-x boundary (potential internal CF) where vbx ends
            // away from domain hi: sample valid + ghost just outside patch
            if (vbx.bigEnd(0) < ehi[0]) {
              sample("patch-hi-x valid", vbx.bigEnd(0),   vbx.smallEnd(1));
              sample("patch-hi-x ghost", vbx.bigEnd(0)+1, vbx.smallEnd(1));
            }
            if (vbx.bigEnd(1) < ehi[1]) {
              sample("patch-hi-y valid", vbx.smallEnd(0), vbx.bigEnd(1));
              sample("patch-hi-y ghost", vbx.smallEnd(0), vbx.bigEnd(1)+1);
            }
          }
        }
      }
    }
  }

  // Per-level error norms. In 2D Ez decouples and is identically zero. In
  // 1D Ex is the trivial β·Ex=rhs identity; Ey and Ez are the non-trivial
  // scalar Laplacians. In 3D all three components are non-trivial. We
  // report all 3 components in every dim — trivial-zero entries are
  // informative as a sanity check.
  for (int lev = 0; lev < (int)geom.size(); ++lev)
  {
    auto dvol = AMREX_D_TERM(geom[lev].CellSize(0), * geom[lev].CellSize(1), * geom[lev].CellSize(2));
    Array<std::string,3> names_all{"Ex", "Ey", "Ez"};
    for (int idim = 0; idim < 3; ++idim)
    {
      MultiFab::Subtract(solution[lev][idim], exact[lev][idim], 0, 0, 1, 0);
      auto e0 = solution[lev][idim].norminf();
      auto e2 = solution[lev][idim].norm2(0, geom[lev].periodicity());
      e2 *= std::sqrt(dvol);
      amrex::Print() << "  lev " << lev << " " << names_all[idim]
                     << " errors (max, L2): " << e0 << "  " << e2 << '\n';
    }
  }

  // Optional cross-validation: solve the same scalar BVP with MLABecLap
  // on a CC grid covering the same geom/grids/dmap. Verifies that the
  // edge-centred fix1=2 reflux gives the same composite-vs-LBL behaviour
  // as MLABecLap's native flux-difference reflux on the equivalent CC
  // problem. 1D-only.
  bool xref = false;
  {
    ParmParse pp;
    pp.query("mlabec_xref", xref);
  }
  if (xref) { solveABecLapXref(); }
}

void
MyTest::writePlotfile (const std::string& label,
                       const Vector<Array<MultiFab,3>>& sol,
                       const Vector<Array<MultiFab,3>>& exct,
                       const Vector<Array<MultiFab,3>>& rhsv,
                       const Vector<Array<MultiFab,3>>& resid,
                       const Vector<MultiFab>* bz_cc) const
{
#if (AMREX_SPACEDIM == 1)
  // 1D: skip plotfile generation. Error norms and residual diagnostics
  // are emitted directly by the solve() driver; 1D test cases are
  // validated by those prints, not by visual inspection.
  amrex::ignore_unused(label, sol, exct, rhsv, resid);
  return;
#else
  // Components per E field: 2 in 2D (Ex, Ey — Ez decouples), 3 in 3D.
  // Base cc components = 5 fields × NEDGE: sol, exact, rhs, res, abserr.
  // If bz_cc is supplied (tube test: Bz at cell centres), append one more
  // scalar component "Bz".
  constexpr int NEDGE = AMREX_SPACEDIM;
  Vector<std::string> varnames;
#if (AMREX_SPACEDIM == 2)
  varnames = {
    "sol_Ex", "sol_Ey",
    "exact_Ex", "exact_Ey",
    "rhs_Ex", "rhs_Ey",
    "res_Ex", "res_Ey",
    "abserr_Ex", "abserr_Ey"
  };
#else
  varnames = {
    "sol_Ex", "sol_Ey", "sol_Ez",
    "exact_Ex", "exact_Ey", "exact_Ez",
    "rhs_Ex", "rhs_Ey", "rhs_Ez",
    "res_Ex", "res_Ey", "res_Ez",
    "abserr_Ex", "abserr_Ey", "abserr_Ez"
  };
#endif
  bool const have_bz = (bz_cc != nullptr);
  if (have_bz) { varnames.push_back("Bz"); }
  const int ncomp = static_cast<int>(varnames.size());
  const int bz_comp = 5 * NEDGE;  // valid only when have_bz

  const int nl = static_cast<int>(geom.size());
  Vector<MultiFab> plot_mf(nl);
  Vector<const MultiFab*> plot_ptrs(nl);
  Vector<int> level_steps(nl, 0);
  Vector<IntVect> rr(std::max(nl - 1, 0), IntVect(ref_ratio));

  for (int lev = 0; lev < nl; ++lev)
  {
    plot_mf[lev].define(grids[lev], dmap[lev], ncomp, 0);
    plot_mf[lev].setVal(0.0);
    plot_ptrs[lev] = &plot_mf[lev];

    // For each of the 5 source vector fields, manually average the
    // edge-centred components onto cell centres into NEDGE consecutive comps
    // starting at dcomp_base (Ex,Ey[,Ez]).
    auto edge_to_cc = [&](const Array<MultiFab,3>& Earr,
                          MultiFab& cc, int dcomp_base)
    {
#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
      for (MFIter mfi(cc, TilingIfNotGPU()); mfi.isValid(); ++mfi)
      {
        const Box& bx = mfi.tilebox();
        Array4<Real> const& cca = cc.array(mfi);
        Array4<Real const> const& exa = Earr[0].const_array(mfi);
        Array4<Real const> const& eya = Earr[1].const_array(mfi);
        Array4<Real const> const& eza = Earr[2].const_array(mfi);
        const int d = dcomp_base;
        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
#if (AMREX_SPACEDIM == 2)
          // 2D: Ex on (0,1) avg in y; Ey on (1,0) avg in x.
          cca(i,j,k,d  ) = Real(0.5) * (exa(i,j,k) + exa(i,j+1,k));
          cca(i,j,k,d+1) = Real(0.5) * (eya(i,j,k) + eya(i+1,j,k));
          amrex::ignore_unused(eza);
#else
          // 3D: Ex on (0,1,1) avg in y,z; Ey on (1,0,1) avg in x,z;
          //     Ez on (1,1,0) avg in x,y.
          cca(i,j,k,d  ) = Real(0.25) *
            (exa(i,j  ,k  ) + exa(i,j+1,k  ) +
             exa(i,j  ,k+1) + exa(i,j+1,k+1));
          cca(i,j,k,d+1) = Real(0.25) *
            (eya(i  ,j,k  ) + eya(i+1,j,k  ) +
             eya(i  ,j,k+1) + eya(i+1,j,k+1));
          cca(i,j,k,d+2) = Real(0.25) *
            (eza(i  ,j  ,k) + eza(i+1,j  ,k) +
             eza(i  ,j+1,k) + eza(i+1,j+1,k));
#endif
        });
      }
    };

    edge_to_cc(sol  [lev], plot_mf[lev], 0*NEDGE);
    edge_to_cc(exct [lev], plot_mf[lev], 1*NEDGE);
    edge_to_cc(rhsv [lev], plot_mf[lev], 2*NEDGE);
    edge_to_cc(resid[lev], plot_mf[lev], 3*NEDGE);

    // abs(error) = |sol - exact| computed as cell-centered diff of the already
    // averaged plot_mf entries.
    constexpr int abserr_base = 4*NEDGE;
    MultiFab::Copy    (plot_mf[lev], plot_mf[lev], 0,        abserr_base, NEDGE, 0);
    MultiFab::Subtract(plot_mf[lev], plot_mf[lev], 1*NEDGE,  abserr_base, NEDGE, 0);
    auto abs_inplace = [&](int comp)
    {
      MultiFab& mf = plot_mf[lev];
#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
      for (MFIter mfi(mf, TilingIfNotGPU()); mfi.isValid(); ++mfi)
      {
        const Box& bx = mfi.tilebox();
        Array4<Real> const& a = mf.array(mfi);
        const int c = comp;
        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
          a(i,j,k,c) = std::abs(a(i,j,k,c));
        });
      }
    };
    for (int c = 0; c < NEDGE; ++c) { abs_inplace(abserr_base + c); }

    if (have_bz) {
      // bz_cc is per-AMR-level, cell-centered, 1 component.
      AMREX_ALWAYS_ASSERT((*bz_cc)[lev].nComp() == 1);
      MultiFab::Copy(plot_mf[lev], (*bz_cc)[lev], 0, bz_comp, 1, 0);
    }
  }

  if (ParallelDescriptor::IOProcessor()) {
    amrex::UtilCreateDirectory(plot_dir, 0755);
  }
  ParallelDescriptor::Barrier();
  const std::string fname = plot_dir + "/plt_" + label;
  amrex::Print() << "  Writing plotfile: " << fname << '\n';
  amrex::WriteMultiLevelPlotfile(fname, nl, plot_ptrs, varnames,
                                 geom, Real(0.0), level_steps, rr);
#endif // AMREX_SPACEDIM != 1
}

void
MyTest::readParameters ()
{
  ParmParse pp;
  pp.query("n_cell", n_cell);
  pp.query("max_grid_size", max_grid_size);
  pp.query("ref_ratio", ref_ratio);
  pp.query("nlevels", nlevels);
  pp.query("fine_box_lo", fine_box_lo);
  pp.query("fine_box_hi", fine_box_hi);
  pp.query("fine_box_lo_l2", fine_box_lo_l2);
  pp.query("fine_box_hi_l2", fine_box_hi_l2);
  pp.query("verbose", verbose);
  pp.query("bottom_verbose", bottom_verbose);
  pp.query("max_iter", max_iter);
  pp.query("agglomeration", agglomeration);
  pp.query("consolidation", consolidation);
  pp.query("max_coarsening_level", max_coarsening_level);
  pp.query("alpha", alpha);
  pp.query("tol_rel", tol_rel);
  pp.query("beta_profile", beta_profile);
  pp.query("plot_dir", plot_dir);
  pp.query("problem", problem);
  pp.query("tube_r1", tube_r1);
  pp.query("tube_r2", tube_r2);
  pp.query("tube_eta", tube_eta);
  pp.query("tube_eta_FS", tube_eta_FS);
  pp.query("tube_alpha_aniso", tube_alpha_aniso);
  pp.query("tube_B0", tube_B0);
  pp.query("tube_dt", tube_dt);
  pp.query("tube_refine_band", tube_refine_band);
  pp.query("tube_refine_kind", tube_refine_kind);
  pp.query("tube_refine_box_lo_x", tube_refine_box_lo_x);
  pp.query("tube_refine_box_lo_y", tube_refine_box_lo_y);
  pp.query("tube_refine_box_hi_x", tube_refine_box_hi_x);
  pp.query("tube_refine_box_hi_y", tube_refine_box_hi_y);
  pp.query("prob_lo_x", prob_lo_x);
  pp.query("prob_lo_y", prob_lo_y);
  pp.query("prob_hi_x", prob_hi_x);
  pp.query("prob_hi_y", prob_hi_y);
}

void
MyTest::initData ()
{
  geom.resize(nlevels);
  grids.resize(nlevels);
  dmap.resize(nlevels);
  solution.resize(nlevels);
  exact.resize(nlevels);
  rhs.resize(nlevels);
  bcoef.resize(nlevels);

  bool const is_tube = (problem == "tube");

  // Domain and BC: tube uses [prob_lo, prob_hi]² with Dirichlet (non-
  // periodic); manufactured uses unit square periodic.
  RealBox rb = is_tube
      ? RealBox({AMREX_D_DECL(prob_lo_x, prob_lo_y, 0.)},
                {AMREX_D_DECL(prob_hi_x, prob_hi_y, 1.)})
      : RealBox({AMREX_D_DECL(0., 0., 0.)},
                {AMREX_D_DECL(1., 1., 1.)});
  Array<int,AMREX_SPACEDIM> is_periodic = is_tube
      ? Array<int,AMREX_SPACEDIM>{AMREX_D_DECL(0, 0, 0)}
      : Array<int,AMREX_SPACEDIM>{AMREX_D_DECL(1, 1, 1)};
  Geometry::Setup(&rb, 0, is_periodic.data());

  // Level-0 (coarse): full domain.
  Box domain0(IntVect(0), IntVect(n_cell - 1));
  geom[0].define(domain0);
  grids[0].define(domain0);
  grids[0].maxSize(max_grid_size);
  dmap[0].define(grids[0]);

  if (nlevels > 1)
  {
    if (is_tube) {
      // Annular refinement: tile coarse domain with max_grid_size boxes,
      // keep tiles whose cells intersect the band r in
      // [tube_r1 - band, tube_r2 + band].
      auto const dx_c = geom[0].CellSizeArray();
      Real const xlo = prob_lo_x, ylo = prob_lo_y;
      // "annular": ring around tube wall.
      // "disk": entire interior + tube wall + buffer.
      // "box": rectangular patch defined by tube_refine_box_{lo,hi}_{x,y}.
      enum { KIND_ANNULAR, KIND_DISK, KIND_BOX } refine_kind = KIND_ANNULAR;
      if (tube_refine_kind == "disk") refine_kind = KIND_DISK;
      else if (tube_refine_kind == "box") refine_kind = KIND_BOX;
      Real const r_lo = (refine_kind == KIND_DISK) ? Real(0.0)
                                                   : (tube_r1 - tube_refine_band);
      Real const r_hi = tube_r2 + tube_refine_band;
      Real const bx_lo = tube_refine_box_lo_x;
      Real const by_lo = tube_refine_box_lo_y;
      Real const bx_hi = tube_refine_box_hi_x;
      Real const by_hi = tube_refine_box_hi_y;
      BoxList bl;
      int const tile = max_grid_size;
      for (int j_lo = 0; j_lo < n_cell; j_lo += tile)
      for (int i_lo = 0; i_lo < n_cell; i_lo += tile) {
          Box t(IntVect(AMREX_D_DECL(i_lo, j_lo, 0)),
                IntVect(AMREX_D_DECL(std::min(i_lo+tile-1, n_cell-1),
                                     std::min(j_lo+tile-1, n_cell-1), 0)));
          bool keep = false;
          for (int j = t.smallEnd(1); j <= t.bigEnd(1) && !keep; ++j)
          for (int i = t.smallEnd(0); i <= t.bigEnd(0) && !keep; ++i) {
              Real x = xlo + (Real(i) + Real(0.5)) * dx_c[0];
              Real y = ylo + (Real(j) + Real(0.5)) * dx_c[1];
              if (refine_kind == KIND_BOX) {
                  if (x >= bx_lo && x <= bx_hi
                   && y >= by_lo && y <= by_hi) keep = true;
              } else {
                  Real r = std::sqrt(x*x + y*y);
                  if (r >= r_lo && r < r_hi) keep = true;
              }
          }
          if (keep) bl.push_back(t);
      }
      if (bl.size() == 0) {
          amrex::Abort("Tube annular refinement: no tiles intersect the "
                       "refine band — increase tube_refine_band or n_cell.");
      }
      BoxArray fine_coarse(bl);
      Box fine_domain_box = amrex::refine(fine_coarse.minimalBox(),
                                          ref_ratio);
      Box fine_geom_domain = amrex::refine(domain0, ref_ratio);
      geom[1].define(fine_geom_domain);
      // Refine the BoxArray of tiles to give the actual fine BoxArray.
      grids[1] = amrex::refine(fine_coarse, ref_ratio);
      grids[1].maxSize(max_grid_size * ref_ratio);
      dmap[1].define(grids[1]);
      amrex::ignore_unused(fine_domain_box);
    } else {
      // Level-1 (fine): user-controllable patch in coarse-cell coords.
      // -1 means default to centered-half [n_cell/4, 3*n_cell/4 - 1].
      int lo = (fine_box_lo >= 0) ? fine_box_lo : n_cell / 4;
      int hi = (fine_box_hi >= 0) ? fine_box_hi : (3 * n_cell / 4 - 1);
      Box fine_coarse_region{IntVect(lo), IntVect(hi)};
      Box fine_domain = amrex::refine(fine_coarse_region, ref_ratio);
      Box fine_geom_domain = amrex::refine(domain0, ref_ratio);
      geom[1].define(fine_geom_domain);
      grids[1].define(fine_domain);
      grids[1].maxSize(max_grid_size * ref_ratio);
      dmap[1].define(grids[1]);
    }
  }

  if (nlevels > 2)
  {
    Box fine2_geom_domain = amrex::refine(geom[1].Domain(), ref_ratio);
    geom[2].define(fine2_geom_domain);
    if (is_tube) {
      // Tube: annular refinement at lev 2, tighter band than lev 1.
      // Default: keep tiles whose cells intersect the tube wall plus a
      // half-band buffer, r ∈ [r1 - band/2, r2 + band/2]. Same tube_refine_kind
      // selector as lev 1 (annular / disk / box).
      auto const dx_l1 = geom[1].CellSizeArray();
      Real const xlo = prob_lo_x, ylo = prob_lo_y;
      Real const band_l2 = Real(0.5) * tube_refine_band;
      Real const r_lo = (tube_refine_kind == "disk") ? Real(0.0)
                                                     : (tube_r1 - band_l2);
      Real const r_hi = tube_r2 + band_l2;
      Real const bx_lo = tube_refine_box_lo_x;
      Real const by_lo = tube_refine_box_lo_y;
      Real const bx_hi = tube_refine_box_hi_x;
      Real const by_hi = tube_refine_box_hi_y;
      enum { KIND_ANNULAR, KIND_DISK, KIND_BOX } refine_kind = KIND_ANNULAR;
      if (tube_refine_kind == "disk") refine_kind = KIND_DISK;
      else if (tube_refine_kind == "box") refine_kind = KIND_BOX;
      // Walk the lev-1 BoxArray, tile each lev-1 box by max_grid_size, keep
      // tiles touching the lev-2 refinement region.
      BoxArray const& lev1_ba = grids[1];
      int const tile = max_grid_size;
      BoxList bl2;
      for (int ib = 0; ib < lev1_ba.size(); ++ib) {
        Box const& b = lev1_ba[ib];
        for (int j_lo = b.smallEnd(1); j_lo <= b.bigEnd(1); j_lo += tile)
        for (int i_lo = b.smallEnd(0); i_lo <= b.bigEnd(0); i_lo += tile) {
          Box t(IntVect(AMREX_D_DECL(i_lo, j_lo, 0)),
                IntVect(AMREX_D_DECL(std::min(i_lo+tile-1, b.bigEnd(0)),
                                     std::min(j_lo+tile-1, b.bigEnd(1)), 0)));
          bool keep = false;
          for (int j = t.smallEnd(1); j <= t.bigEnd(1) && !keep; ++j)
          for (int i = t.smallEnd(0); i <= t.bigEnd(0) && !keep; ++i) {
            Real x = xlo + (Real(i) + Real(0.5)) * dx_l1[0];
            Real y = ylo + (Real(j) + Real(0.5)) * dx_l1[1];
            if (refine_kind == KIND_BOX) {
              if (x >= bx_lo && x <= bx_hi
               && y >= by_lo && y <= by_hi) keep = true;
            } else {
              Real r = std::sqrt(x*x + y*y);
              if (r >= r_lo && r < r_hi) keep = true;
            }
          }
          if (keep) bl2.push_back(t);
        }
      }
      if (bl2.size() == 0) {
        amrex::Abort("Tube nlevels>=3: no tiles intersect lev-2 refinement "
                     "band — narrow tube_refine_band or supply explicit "
                     "fine_box_*_l2.");
      }
      BoxArray fine2_l1(bl2);
      grids[2] = amrex::refine(fine2_l1, ref_ratio);
      grids[2].maxSize(max_grid_size * ref_ratio * ref_ratio);
      dmap[2].define(grids[2]);
    } else {
      // Manufactured: centered-half of lev 1's fine region.
      int lev1_lo = (fine_box_lo >= 0) ? fine_box_lo : n_cell / 4;
      int lev1_hi = (fine_box_hi >= 0) ? fine_box_hi : (3 * n_cell / 4 - 1);
      int lev1_lo_f = lev1_lo * ref_ratio;
      int lev1_hi_f = (lev1_hi + 1) * ref_ratio - 1;
      int lev1_size = lev1_hi_f - lev1_lo_f + 1;
      int default_l2_lo = lev1_lo_f + lev1_size / 4;
      int default_l2_hi = lev1_lo_f + (3 * lev1_size) / 4 - 1;
      int lo2 = (fine_box_lo_l2 >= 0) ? fine_box_lo_l2 : default_l2_lo;
      int hi2 = (fine_box_hi_l2 >= 0) ? fine_box_hi_l2 : default_l2_hi;
      Box fine2_l1_region{IntVect(lo2), IntVect(hi2)};
      Box fine2_domain = amrex::refine(fine2_l1_region, ref_ratio);
      grids[2].define(fine2_domain);
      grids[2].maxSize(max_grid_size * ref_ratio * ref_ratio);
      dmap[2].define(grids[2]);
    }
  }

  // Edge index types (2D): Ex (0,1), Ey (1,0), Ez (1,1).
  Array<IntVect,3> etype
  {
    IntVect(AMREX_D_DECL(0, 1, 1)),
    IntVect(AMREX_D_DECL(1, 0, 1)),
    IntVect(AMREX_D_DECL(1, 1, 0))
  };

  for (int lev = 0; lev < nlevels; ++lev)
  {
    for (int idim = 0; idim < 3; ++idim)
    {
      BoxArray eba = amrex::convert(grids[lev], etype[idim]);
      solution[lev][idim].define(eba, dmap[lev], 1, 1);
      exact   [lev][idim].define(eba, dmap[lev], 1, 1);
      rhs     [lev][idim].define(eba, dmap[lev], 1, 0);
      // bcoef holds 1 ghost so initProb (which writes to tilebox(IntVect(1)))
      // can fill ghost edges with the analytical β formula. This gives
      // MLCurlCurl::setBeta valid β at CF interface / domain-boundary ghost
      // edges, which the 4-block smoother reads at near-boundary nodes.
      bcoef   [lev][idim].define(eba, dmap[lev], 1, 1);
    }
  }

  initProb();

  for (int lev = 0; lev < nlevels; ++lev)
  {
    for (int idim = 0; idim < 3; ++idim)
    {
      exact[lev][idim].LocalCopy(solution[lev][idim], 0, 0, 1, IntVect(1));
    }
  }
}

void
MyTest::solveABecLapXref ()
{
#if (AMREX_SPACEDIM != 1)
  amrex::Print() << "[mlabec_xref] only implemented for SPACEDIM==1; skipping.\n";
#else
  using amrex::Real;
  constexpr Real pi = amrex::Math::pi<Real>();

  // Operator: −α ∂²u/∂x² + β u = (4π²α + β) sin(2πx). Mirrors the
  // MLCurlCurl-1D Ey component. β scalar = 1.0 matches the MLCurlCurl
  // test's setScalars(alpha, 1.0).
  Real const cce      = Real(4.0) * pi * pi * alpha;
  Real const beta_val = Real(1.0);

  int const nlev = static_cast<int>(geom.size());

  Vector<MultiFab> cc_sol  (nlev);
  Vector<MultiFab> cc_rhs  (nlev);
  Vector<MultiFab> cc_exact(nlev);
  Vector<MultiFab> cc_acoef(nlev);

  auto init_problem = [&]() {
    for (int lev = 0; lev < nlev; ++lev) {
      cc_sol  [lev].define(grids[lev], dmap[lev], 1, 1);
      cc_rhs  [lev].define(grids[lev], dmap[lev], 1, 0);
      cc_exact[lev].define(grids[lev], dmap[lev], 1, 1);
      cc_acoef[lev].define(grids[lev], dmap[lev], 1, 0);

      cc_sol  [lev].setVal(Real(0.0));
      cc_acoef[lev].setVal(Real(1.0));

      auto const& dx     = geom[lev].CellSizeArray();
      auto const& problo = geom[lev].ProbLoArray();
      auto const& exact_a = cc_exact[lev].arrays();
      auto const& rhs_a   = cc_rhs  [lev].arrays();
      ParallelFor(cc_exact[lev], IntVect(1),
        [=] AMREX_GPU_DEVICE (int bno, int i, int j, int k)
      {
        Real xcc = problo[0] + (Real(i) + Real(0.5)) * dx[0];
        Real u   = std::sin(Real(2.0) * pi * xcc);
        exact_a[bno](i,j,k) = u;
        if (rhs_a[bno].contains(i,j,k)) {
          rhs_a[bno](i,j,k) = (cce + beta_val) * u;
        }
      });
    }
    Gpu::streamSynchronize();
  };

  auto report_errors = [&](char const* tag) {
    for (int lev = 0; lev < nlev; ++lev) {
      MultiFab err(grids[lev], dmap[lev], 1, 0);
      err.LocalCopy(cc_sol[lev], 0, 0, 1, IntVect(0));
      MultiFab::Subtract(err, cc_exact[lev], 0, 0, 1, 0);
      auto e0 = err.norminf();
      auto e2 = err.norm2(0, geom[lev].periodicity());
      e2 *= std::sqrt(geom[lev].CellSize(0));
      amrex::Print() << "  [mlabec_xref " << tag << "] lev " << lev
                     << " errors (max, L2): " << e0 << "  " << e2 << '\n';
    }
  };

  // Periodic in x — same as the MLCurlCurl test (geom is also globally
  // periodic so we can't switch to Dirichlet without rebuilding geom).
  Array<LinOpBCType,AMREX_SPACEDIM> lobc{LinOpBCType::Periodic};
  Array<LinOpBCType,AMREX_SPACEDIM> hibc = lobc;

  LPInfo info;
  info.setAgglomeration(agglomeration);
  info.setConsolidation(consolidation);
  info.setMaxCoarseningLevel(max_coarsening_level);

  auto setup_one_level_coefs = [&](MLABecLaplacian& mlabec, int storage_lev,
                                   int data_lev)
  {
    mlabec.setACoeffs(storage_lev, cc_acoef[data_lev]);
    Array<MultiFab,AMREX_SPACEDIM> face_b;
    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
      BoxArray fba = amrex::convert(grids[data_lev],
                                    IntVect::TheDimensionVector(idim));
      face_b[idim].define(fba, dmap[data_lev], 1, 0);
      face_b[idim].setVal(Real(1.0));
    }
    mlabec.setBCoeffs(storage_lev, GetArrOfConstPtrs(face_b));
  };

  // ---- LBL solve ----
  init_problem();
  for (int lev = 0; lev < nlev; ++lev)
  {
    MLABecLaplacian mlabec({geom[lev]}, {grids[lev]}, {dmap[lev]}, info);
    mlabec.setDomainBC(lobc, hibc);
    if (lev > 0) {
      mlabec.setCoarseFineBC(&cc_sol[lev-1], ref_ratio);
    }
    mlabec.setLevelBC(0, &cc_sol[lev]);
    mlabec.setScalars(beta_val, alpha);
    setup_one_level_coefs(mlabec, /*storage_lev=*/0, /*data_lev=*/lev);

    MLMG mlmg(mlabec);
    mlmg.setMaxIter(max_iter);
    mlmg.setVerbose(verbose);
    mlmg.setBottomVerbose(bottom_verbose);
    amrex::Print() << "[mlabec_xref] LBL solve lev=" << lev << '\n';
    mlmg.solve({&cc_sol[lev]}, {&cc_rhs[lev]}, tol_rel, Real(0.0));
  }
  report_errors("LBL");

  // ---- Composite solve ----
  init_problem();
  {
    MLABecLaplacian mlabec(geom, grids, dmap, info);
    mlabec.setDomainBC(lobc, hibc);
    for (int lev = 0; lev < nlev; ++lev) {
      mlabec.setLevelBC(lev, &cc_sol[lev]);
    }
    mlabec.setScalars(beta_val, alpha);
    for (int lev = 0; lev < nlev; ++lev) {
      setup_one_level_coefs(mlabec, /*storage_lev=*/lev, /*data_lev=*/lev);
    }

    MLMG mlmg(mlabec);
    mlmg.setMaxIter(max_iter);
    mlmg.setVerbose(verbose);
    mlmg.setBottomVerbose(bottom_verbose);
    amrex::Print() << "[mlabec_xref] composite solve\n";
    mlmg.solve(GetVecOfPtrs(cc_sol), GetVecOfConstPtrs(cc_rhs),
               tol_rel, Real(0.0));
  }
  report_errors("composite");
#endif
}
