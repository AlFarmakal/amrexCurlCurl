#ifndef MY_TEST_H_
#define MY_TEST_H_

#include <AMReX_MultiFab.H>

#include <string>

class MyTest
{
public:

  MyTest ();

  void solve ();

  // public for cuda
  void initProb ();

private:

  void readParameters ();
  void initData ();

  //! Phase-1 cross-validation: solve the same 1D scalar BVP
  //! −α u'' + β u = (4π²α + β) sin(2πx) on [0,1] (Dirichlet u=0 at x=0,1)
  //! using MLABecLaplacian on a CC grid covering the same geom/grids/dmap.
  //! Runs both LBL and composite, reports per-level max/L2 errors. Allows
  //! checking whether MLABecLap's native flux-difference reflux gives errors
  //! comparable to MLCurlCurl-1D fix1=2 — i.e. whether fix1=2 is the
  //! correct edge-centred analog of MLABecLap's CC composite reflux.
  void solveABecLapXref ();

  //! Build a per-level cell-centered plot MF with components for solution,
  //! exact, rhs, residual, and abs(error) for both Ex and Ey edge fields.
  //! Caller-supplied source MFs are sampled at edges and averaged to cc.
  void writePlotfile (const std::string& label,
                      const amrex::Vector<amrex::Array<amrex::MultiFab,3>>& sol,
                      const amrex::Vector<amrex::Array<amrex::MultiFab,3>>& exct,
                      const amrex::Vector<amrex::Array<amrex::MultiFab,3>>& rhsv,
                      const amrex::Vector<amrex::Array<amrex::MultiFab,3>>& resid,
                      const amrex::Vector<amrex::MultiFab>* bz_cc = nullptr) const;

  //! Coarse domain extent in cells per direction.
  int n_cell = 64;
  //! Per-fab maximum size on each AMR level.
  int max_grid_size = 32;
  //! AMR refinement ratio between level 0 and level 1.
  int ref_ratio = 2;
  //! Number of AMR levels (1 → single-level; 2 → coarse + fine patch).
  int nlevels = 2;
  //! Lower corner of the fine patch in coarse-cell coordinates. -1 →
  //! default to centered-half (n_cell/4). Only used when nlevels > 1.
  int fine_box_lo = -1;
  //! Upper corner of the fine patch in coarse-cell coordinates
  //! (inclusive). -1 → default to centered-half (3*n_cell/4 - 1).
  int fine_box_hi = -1;
  //! Lower / upper corner of the level-2 patch in level-1 cell coords
  //! (inclusive). -1 → default to centered-half of the level-1 fine
  //! region. Only used when nlevels > 2.
  int fine_box_lo_l2 = -1;
  int fine_box_hi_l2 = -1;
  //! Optional second level-1 box (coarse-cell coords, inclusive,
  //! disjoint from box 1) for an L/step-shaped fine region with
  //! re-entrant CF corners. -1 → disabled. Manufactured problem only.
  int fine_box2_lo_x = -1;
  int fine_box2_lo_y = -1;
  int fine_box2_hi_x = -1;
  int fine_box2_hi_y = -1;
  //! Max iterations passed to MLMG.
  int max_iter = 200;
  //! MLMG verbosity.
  int verbose = 2;
  //! Bottom-solver verbosity.
  int bottom_verbose = 0;
  //! AMReX agglomeration knobs.
  bool agglomeration = true;
  bool consolidation = true;
  int  max_coarsening_level = 30;

  //! Selector: "uniform", "smooth_10x", "smooth_100x".
  std::string beta_profile = "uniform";
  //! Plotfile output directory (empty → no plotfiles written).
  std::string plot_dir = "";
  //! Constant scalar coefficient on curl(curl).
  amrex::Real alpha = 1.0;
  //! Convergence tolerance (relative).
  amrex::Real tol_rel = 1.0e-10;

  //! Tube-diffusion problem (faithful port of CNS-AMReX
  //! settings/MHDDiffusion/diffusion_nonuniform_tube). Selected by
  //! problem=tube. Overrides domain/BC/IC/refinement/scalars.
  std::string problem = "manufactured";
  amrex::Real tube_r1 = 0.45;
  amrex::Real tube_r2 = 0.5;
  amrex::Real tube_eta = 0.001;
  amrex::Real tube_eta_FS = 1000.0;
  amrex::Real tube_alpha_aniso = 0.5;
  amrex::Real tube_B0 = 1.0;
  amrex::Real tube_dt = 0.01;
  amrex::Real tube_refine_band = 0.05;
  std::string tube_refine_kind = "annular"; // "annular", "disk", or "box"
  // Box-refinement physical extents (used when tube_refine_kind == "box").
  amrex::Real tube_refine_box_lo_x = -0.7;
  amrex::Real tube_refine_box_lo_y = -0.7;
  amrex::Real tube_refine_box_hi_x =  0.7;
  amrex::Real tube_refine_box_hi_y =  0.7;
  amrex::Real prob_lo_x = 0.0, prob_lo_y = 0.0;
  amrex::Real prob_hi_x = 1.0, prob_hi_y = 1.0;

  //! Per-AMR-level geometries / grids / dmaps.
  amrex::Vector<amrex::Geometry>            geom;
  amrex::Vector<amrex::BoxArray>            grids;
  amrex::Vector<amrex::DistributionMapping> dmap;

  //! Per-AMR-level edge MFs (3 components per level).
  amrex::Vector<amrex::Array<amrex::MultiFab,3>> solution;
  amrex::Vector<amrex::Array<amrex::MultiFab,3>> exact;
  amrex::Vector<amrex::Array<amrex::MultiFab,3>> rhs;
  amrex::Vector<amrex::Array<amrex::MultiFab,3>> bcoef;
};

#endif
