#ifndef AMREX_ML_CURL_CURL_H_
#define AMREX_ML_CURL_CURL_H_
#include <AMReX_Config.H>

#include <AMReX_MLLinOp.H>
#include <AMReX_MLCurlCurl_K.H>

namespace amrex {

/**
 * \brief curl (alpha curl E) + beta E = rhs
 *
 * Here E is an Array of 3 MultiFabs on the staggered grid.  The coefficient
 * \f$\alpha\f$ can be supplied either as a positive scalar (via setScalars)
 * or as a nodal MultiFab (via setAlpha), and \f$\beta\f$ can be either a
 * non-negative scalar or an edge-centered MultiFab (via setBeta).
 *
 * It's the caller's responsibility to make sure rhs has consistent nodal
 * data. If needed, one could call prepareRHS for this.
 *
 * The smoother is based on the 4-color Gauss-Seidel smoother of Li
 * et. al. 2020.  "An Efficient Preconditioner for 3-D Finite Difference
 * Modeling of the Electromagnetic Diffusion Process in the Frequency
 * Domain", IEEE Transactions on Geoscience and Remote Sensing, 58, 500-509.
 *
 * Supports composite solves across multiple AMR levels.
 */
class MLCurlCurl
    : public MLLinOpT<Array<MultiFab,3> >
{
public:
    using MF        = Array<MultiFab,3>;
    using RT        = typename MLLinOpT<MF>::RT;
    using BCType    = typename MLLinOpT<MF>::BCType;
    using BCMode    = typename MLLinOpT<MF>::BCMode;
    using StateMode = typename MLLinOpT<MF>::StateMode;
    using Location  = typename MLLinOpT<MF>::Location;

    MLCurlCurl () = default;
    MLCurlCurl (const Vector<Geometry>& a_geom,
                const Vector<BoxArray>& a_grids,
                const Vector<DistributionMapping>& a_dmap,
                const LPInfo& a_info = LPInfo());

    void define (const Vector<Geometry>& a_geom,
                 const Vector<BoxArray>& a_grids,
                 const Vector<DistributionMapping>& a_dmap,
                 const LPInfo& a_info = LPInfo());

    //! Set scalar coefficients.  This clears any previously supplied nodal alpha
    //! or edge-centered beta MultiFabs, so call setAlpha/setBeta afterwards if
    //! you still need spatially varying coefficients.
    void setScalars (RT a_alpha, RT a_beta) noexcept;

    //! This is needed only if there is variable beta coefficient.  Call this
    //! after setScalars if both APIs are used, because setScalars clears the
    //! cached beta MultiFabs.
    void setBeta (const Vector<Array<MultiFab const*,3>>& a_bcoefs);

    //! Set variable alpha defined on nodal MultiFab.  Call this after
    //! setScalars if both APIs are used, because setScalars clears the cached
    //! nodal alpha fields.
    void setAlpha (const Vector<MultiFab const*>& a_acoeffs);

    //! Synchronize RHS on nodal points. If the user can guarantee it, this
    //! function does not need to be called.
    void prepareRHS (Vector<MF*> const& rhs) const;

    void setDirichletNodesToZero (int amrlev, int mglev, MF& a_mf) const override;

    [[nodiscard]] std::string name () const override {
        return std::string("curl of curl");
    }

    bool setUsePCG (bool flag) { return std::exchange(m_use_pcg, flag); }

    void setLevelBC (int amrlev, const MF* levelbcdata,
                     const MF* robinbc_a = nullptr,
                     const MF* robinbc_b = nullptr,
                     const MF* robinbc_f = nullptr) override;

    void restriction (int amrlev, int cmglev, MF& crse, MF& fine) const override;

    void interpolation (int amrlev, int fmglev, MF& fine, const MF& crse) const override;

    void apply (int amrlev, int mglev, MF& out, MF& in, BCMode bc_mode,
                StateMode s_mode, const MLMGBndryT<MF>* bndry=nullptr) const override;

    void smooth (int amrlev, int mglev, MF& sol, const MF& rhs,
                 bool skip_fillboundary, int niter) const override;

    void solutionResidual (int amrlev, MF& resid, MF& x, const MF& b,
                           const MF* crse_bcdata=nullptr) override;

    void correctionResidual (int amrlev, int mglev, MF& resid, MF& x,
                             const MF& b, BCMode bc_mode,
                             const MF* crse_bcdata=nullptr) override;

    [[nodiscard]] bool needsUpdate () const override {
        return (m_needs_update || MLLinOpT<Array<MultiFab,3>>::needsUpdate());
    }
    void update () override;

    void prepareForSolve () override;

    [[nodiscard]] bool isSingular (int /*amrlev*/) const override { return false; }
    [[nodiscard]] bool isBottomSingular () const override { return false; }

    RT xdoty (int amrlev, int mglev, const MF& x, const MF& y, bool local) const override;

    [[nodiscard]] RT normInf (int amrlev, MF const& mf, bool local) const override;

    //! The outer Krylov matrix action sees the second-order CF-ghost
    //! solution path (accurate composite operator); the preconditioner's
    //! internal cycle keeps the lowest-order curl-conforming fill it is
    //! built around. See apply().
    void beginMatrixApply () override { m_matrix_apply = true; }
    void endMatrixApply () override { m_matrix_apply = false; }

    //! Multi-AMR-level dot product for GMRESMLMG. Sums xdoty over all
    //! AMR levels (each xdoty excludes covered coarse edges via the
    //! AMR fine_mask), then reduces across MPI ranks.
    RT dotProductPrecond (Vector<MF const*> const& x,
                          Vector<MF const*> const& y) const override;

    //! Multi-AMR-level L2 norm for GMRESMLMG.
    RT norm2Precond (Vector<MF const*> const& x) const override;

    void averageDownAndSync (Vector<MF>& sol) const override;

    //! Average fine-level solution and RHS down to coarse level.
    void averageDownSolutionRHS (int camrlev, MF& crse_sol, MF& crse_rhs,
                                 const MF& fine_sol, const MF& fine_rhs) override;

    //! Restrict fine residual to covered coarse edges.
    void avgDownResAmr (int clev, MF& cres, MF const& fres) const override;

    //! Prolongate coarse AMR correction to fine AMR level.
    void interpolationAmr (int famrlev, MF& fine, const MF& crse,
                           IntVect const& nghost) const override;

    //! Fix coarse residual at fringe edges (C/F boundary).
    void reflux (int crse_amrlev, MF& res,
                 const MF& crse_sol, const MF& crse_rhs,
                 MF& fine_res, MF& fine_sol, const MF& fine_rhs) const override;

    [[nodiscard]] IntVect getNGrowVectRestriction () const override {
        return IntVect(1);
    }

    void make (Vector<Vector<MF> >& mf, IntVect const& ng) const override;

    [[nodiscard]] MF make (int amrlev, int mglev, IntVect const& ng) const override;

    [[nodiscard]] MF makeAlias (MF const& mf) const override;

    [[nodiscard]] MF makeCoarseMG (int amrlev, int mglev, IntVect const& ng) const override;

    [[nodiscard]] MF makeCoarseAmr (int famrlev, IntVect const& ng) const override;

// public for cuda

#if (AMREX_SPACEDIM > 1)
    void smooth4 (int amrlev, int mglev, MF& sol, MF const& rhs, int color) const;
#else
    void smooth1D (int amrlev, int mglev, MF& sol, MF const& rhs, int color) const;
#endif

    void compresid (int amrlev, int mglev, MF& resid, MF const& b) const;

    void applyPhysBC (int amrlev, int mglev, MultiFab& mf, CurlCurlStateType type) const;

    //! Galerkin operator action on the AMR-coarsened covered region.
    //! Computes (R L^f I) v at covered coarse cells, writing into Ax_c.
    //! Used to make the coarse operator at cov variationally consistent
    //! with the fine operator — necessary for stable composite V-cycles
    //! on curl-curl with small β in free space.
    void applyAmrLevelGalerkin (int crse_amrlev, MF& Ax_c, MF const& v_c) const;

    //! Damped-Jacobi smoother pass at AMR-covered cells using the
    //! Galerkin operator action. Run after the standard 4-block GS to
    //! make the smoother consistent with apply() at cov, so the V-cycle
    //! on the AMR-coarse level contracts.
    void smoothCovGalerkinJacobi (int amrlev, MF& sol, MF const& rhs) const;

private:

    //! Fill ghost cells: FillBoundary + physical BC + CF BC.
    //! When homogeneous is true, CF ghosts are filled with zero
    //! (correction equation). When false, they are filled from stored
    //! coarse data (solution equation). sol2nd selects the second-order
    //! CF-ghost prolongation (solution path of multi-level composite
    //! solves; the correction path keeps the lowest-order curl-
    //! conforming interpolation).
    void applyBC (int amrlev, int mglev, MF& in, CurlCurlStateType type,
                  bool homogeneous = false, bool sol2nd = false) const;

    //! Fill CF ghost cells using m_cfmask to avoid corrupting same-level
    //! neighbor ghosts. When homogeneous is true, writes zero. When false,
    //! interpolates from m_crse_sol_br (second order when sol2nd).
    void fillCoarseFineBoundary (int amrlev, MF& mf, bool homogeneous,
                                 bool sol2nd = false) const;

    //! Build CF masks for all AMR/MG levels.
    void buildCFMasks ();

    //! Solve the beta-weighted nodal Poisson -div(beta grad phi) = q on
    //! level clev (diagonal-preconditioned CG) and add beta o grad(phi)
    //! to cres, removing the charge q from div(cres). If ring_only, q is
    //! first restricted to the CF interface ring (+1 node). Used by the
    //! charge-consistent composite assembly and by the diagnostic charge
    //! projection in avgDownResAmr.
    void hodgeNeutralizeCharge (int clev, MF& cres, MultiFab& q,
                                bool ring_only) const;

    //! Build fine coverage masks for all AMR levels except the finest.
    void buildFineMask ();

    [[nodiscard]] iMultiFab const& getDotMask (int amrlev, int mglev, int idim) const;
    [[nodiscard]] iMultiFab const& getFineMask (int amrlev, int idim) const;

    [[nodiscard]] CurlCurlDirichletInfo getDirichletInfo (int amrlev, int mglev) const;
    [[nodiscard]] CurlCurlSymmetryInfo getSymmetryInfo (int amrlev, int mglev) const;

    void update_lusolver ();

    void clearAlphaMultiFab ();
    void clearBetaMultiFab ();

    RT m_alpha = std::numeric_limits<RT>::lowest();
    RT m_beta  = std::numeric_limits<RT>::lowest();

    Array<IntVect,3> m_etype
#if (AMREX_SPACEDIM == 3)
         {IntVect(0,1,1), IntVect(1,0,1), IntVect(1,1,0)};
#elif (AMREX_SPACEDIM == 2)
         {IntVect(0,1), IntVect(1,0), IntVect(1,1)};
#else
         {IntVect(0), IntVect(1), IntVect(1)};
#endif

    mutable Vector<Vector<Array<std::unique_ptr<iMultiFab>,3>>> m_dotmask;
    static constexpr int m_ncomp = 1;
    Vector<Vector<std::unique_ptr<Gpu::DeviceScalar
                                  <LUSolver<AMREX_SPACEDIM*2,RT>>>>> m_lusolver;
    Vector<Vector<Array<std::unique_ptr<MultiFab>,3>>> m_bcoefs;
    Vector<Vector<Array<std::unique_ptr<MultiFab>,3>>> m_acoefs;
    bool m_use_pcg = false;
    bool m_needs_update = true;

    // --- Multilevel AMR support ---

    //! CF masks: 0=valid or same-level neighbor ghost, 1=true CF ghost.
    //! [amrlev][mglev][idim], 1 ghost cell.
    Vector<Vector<Array<std::unique_ptr<iMultiFab>,3>>> m_cfmask;

    //! Coarse solution on coarsened-fine layout for CF ghost interpolation.
    //! [amrlev][idim], with 1 ghost cell. Only for amrlev > 0.
    //! Marked mutable so applyAmrLevelGalerkin (const) can temporarily
    //! populate it with the coarse vector during a Galerkin action and
    //! restore afterwards.
    mutable Vector<Array<std::unique_ptr<MultiFab>,3>> m_crse_sol_br;

    //! Whether m_crse_sol_br holds valid data.
    mutable Vector<int> m_has_cf_data;

    //! Fine coverage mask: 0=covered by finer level, 1=not covered.
    //! [amrlev][idim]. Only for amrlev < finest.
    mutable Vector<Array<std::unique_ptr<iMultiFab>,3>> m_fine_mask;

    //! Stash of the alpha*curl(curl) components of the fields entering
    //! the composite residual assembly, per coarse AMR level: filled by
    //! reflux, consumed by avgDownResAmr for the charge-consistent
    //! assembly correction (see avgDownResAmr). [crse_amrlev][idim].
    mutable Vector<Array<MultiFab,3>> m_kstash_crse;
    mutable Vector<Array<MultiFab,3>> m_kstash_fine;
    mutable Vector<int> m_kstash_valid;

    //! True while the outer Krylov matrix action is being applied
    //! (begin/endMatrixApply); selects the second-order CF-ghost
    //! solution path in apply().
    bool m_matrix_apply = false;
};

}

#endif