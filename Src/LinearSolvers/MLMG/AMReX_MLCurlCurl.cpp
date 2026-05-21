#include <AMReX_MLCurlCurl.H>
#include <AMReX_Math.H>
#include <AMReX_MLNodeLinOp_K.H>

namespace amrex {

MLCurlCurl::MLCurlCurl (const Vector<Geometry>& a_geom,
                        const Vector<BoxArray>& a_grids,
                        const Vector<DistributionMapping>& a_dmap,
                        const LPInfo& a_info)
{
    define(a_geom, a_grids, a_dmap, a_info);
}

void MLCurlCurl::define (const Vector<Geometry>& a_geom,
                         const Vector<BoxArray>& a_grids,
                         const Vector<DistributionMapping>& a_dmap,
                         const LPInfo& a_info)
{
    MLLinOpT<MF>::define(a_geom, a_grids, a_dmap, a_info, {});

    // Curl-curl is edge-centered (each E component lives on a distinct
    // edge type). The MLLinOp base treats an unset m_ixtype = (0,0,0)
    // as cell-centered, which makes MLMG::apply call cell-centered
    // average_down routines on the AMR-cov region — aborting for
    // non-MultiFab MF (Array<MultiFab,3>) when used as GMRES
    // preconditioner. Set m_ixtype to a non-cell value (any non-zero
    // IntVect suffices) so isCellCentered() returns false and
    // MLMG::apply skips the cell-centered average_down path.
    // (For non-GMRES paths this is harmless — m_ixtype only feeds
    // isCellCentered and the default MLLinOpT::make* methods which
    // MLCurlCurl overrides.)
    this->m_ixtype = IntVect::TheNodeVector();

    m_dotmask.resize(this->m_num_amr_levels);
    for (int amrlev = 0; amrlev < m_num_amr_levels; ++amrlev) {
        m_dotmask[amrlev].resize(this->m_num_mg_levels[amrlev]);
    }

    m_bcoefs.resize(this->m_num_amr_levels);
    for (int amrlev = 0; amrlev < m_num_amr_levels; ++amrlev) {
        m_bcoefs[amrlev].resize(this->m_num_mg_levels[amrlev]);
    }

    m_acoefs.resize(this->m_num_amr_levels);
    for (int amrlev = 0; amrlev < m_num_amr_levels; ++amrlev) {
        m_acoefs[amrlev].resize(this->m_num_mg_levels[amrlev]);
    }

    m_lusolver.resize(this->m_num_amr_levels);
    for (int amrlev = 0; amrlev < m_num_amr_levels; ++amrlev) {
        m_lusolver[amrlev].resize(this->m_num_mg_levels[amrlev]);
    }

    // Multilevel structures
    m_cfmask.resize(this->m_num_amr_levels);
    for (int amrlev = 0; amrlev < m_num_amr_levels; ++amrlev) {
        m_cfmask[amrlev].resize(this->m_num_mg_levels[amrlev]);
    }

    m_crse_sol_br.resize(this->m_num_amr_levels);
    m_has_cf_data.resize(this->m_num_amr_levels, 0);
    m_fine_mask.resize(this->m_num_amr_levels);
}

void MLCurlCurl::setScalars (RT a_alpha, RT a_beta) noexcept
{
    m_needs_update = true;
    m_alpha = a_alpha;
    m_beta = a_beta;
    clearAlphaMultiFab();
    clearBetaMultiFab();
    AMREX_ASSERT(m_beta > RT(0));
}

void MLCurlCurl::setBeta (const Vector<Array<MultiFab const*,3>>& a_bcoefs)
{
    m_needs_update = true;

    Array<IntVect,3> ng;
    for (int idim = 0; idim < 3; ++idim) {
        ng[idim] = IntVect(1) - m_etype[idim];
    }

    // Copy user-provided beta at MG level 0 on each AMR level.
    // Copy caller's ghost cells too (up to ng[idim]) so fine-level CF-ghost
    // values filled by the caller (from material data) make it into our
    // internal state; otherwise smooth4's betax(i-1,j,k) read at a CF edge
    // sees the default-initialised (NaN in DEBUG) contents.
    for (int amrlev = 0; amrlev < m_num_amr_levels; ++amrlev) {
        for (int idim = 0; idim < 3; ++idim) {
            if (m_bcoefs[amrlev][0][idim] == nullptr) {
                m_bcoefs[amrlev][0][idim] = std::make_unique<MultiFab>
                    (a_bcoefs[amrlev][idim]->boxArray(),
                     a_bcoefs[amrlev][idim]->DistributionMap(), 1, ng[idim]);
            }
            IntVect const copy_ng = amrex::min(
                ng[idim], a_bcoefs[amrlev][idim]->nGrowVect());
            MultiFab::Copy(*m_bcoefs[amrlev][0][idim],
                           *a_bcoefs[amrlev][idim], 0, 0, 1, copy_ng);
            m_bcoefs[amrlev][0][idim]->FillBoundary(
                m_geom[amrlev][0].periodicity());
        }
    }

    // Average beta from fine AMR levels to coarse AMR levels.
    // This ensures the coarse operator under the fine grid sees
    // correctly averaged coefficients.
    for (int amrlev = m_num_amr_levels - 1; amrlev > 0; --amrlev) {
        IntVect ratio(this->AMRRefRatio(amrlev-1));
        for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
            BoxArray cba = m_bcoefs[amrlev][0][idim]->boxArray();
            cba.coarsen(ratio);
            MultiFab ctmp(cba, m_bcoefs[amrlev][0][idim]->DistributionMap(),
                          1, 0);
            average_down_edges(*m_bcoefs[amrlev][0][idim], ctmp, ratio);
            m_bcoefs[amrlev-1][0][idim]->ParallelCopy(ctmp, 0, 0, 1);
            m_bcoefs[amrlev-1][0][idim]->FillBoundary(
                m_geom[amrlev-1][0].periodicity());
        }
#if (AMREX_SPACEDIM < 3)
        {
            BoxArray cba = m_bcoefs[amrlev][0][2]->boxArray();
            cba.coarsen(ratio);
            MultiFab ctmp(cba, m_bcoefs[amrlev][0][2]->DistributionMap(),
                          1, 0);
            average_down_nodal(*m_bcoefs[amrlev][0][2], ctmp, ratio);
            m_bcoefs[amrlev-1][0][2]->ParallelCopy(ctmp, 0, 0, 1);
        }
#endif
#if (AMREX_SPACEDIM == 1)
        {
            BoxArray cba = m_bcoefs[amrlev][0][1]->boxArray();
            cba.coarsen(ratio);
            MultiFab ctmp(cba, m_bcoefs[amrlev][0][1]->DistributionMap(),
                          1, 0);
            average_down_nodal(*m_bcoefs[amrlev][0][1], ctmp, ratio);
            m_bcoefs[amrlev-1][0][1]->ParallelCopy(ctmp, 0, 0, 1);
        }
#endif
    }

    // Coarsen through MG levels within each AMR level.
    for (int amrlev = 0; amrlev < m_num_amr_levels; ++amrlev) {
        for (int mglev = 1; mglev < m_num_mg_levels[amrlev]; ++mglev) {
            IntVect ratio = (amrlev > 0) ? IntVect(2)
                : mg_coarsen_ratio_vec[mglev-1];
            for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
                if (m_bcoefs[amrlev][mglev][idim] == nullptr) {
                    m_bcoefs[amrlev][mglev][idim] = std::make_unique<MultiFab>
                        (amrex::convert(m_grids[amrlev][mglev], m_etype[idim]),
                         m_dmap[amrlev][mglev], 1, ng[idim]);
                }
                average_down_edges(*m_bcoefs[amrlev][mglev-1][idim],
                                   *m_bcoefs[amrlev][mglev  ][idim], ratio);
                m_bcoefs[amrlev][mglev][idim]->FillBoundary(
                    m_geom[amrlev][mglev].periodicity());
            }
#if (AMREX_SPACEDIM < 3)
            if (m_bcoefs[amrlev][mglev][2] == nullptr) {
                m_bcoefs[amrlev][mglev][2] = std::make_unique<MultiFab>
                    (amrex::convert(m_grids[amrlev][mglev], m_etype[2]),
                     m_dmap[amrlev][mglev], 1, 0);
            }
            average_down_nodal(*m_bcoefs[amrlev][mglev-1][2],
                               *m_bcoefs[amrlev][mglev  ][2], ratio);
#endif
#if (AMREX_SPACEDIM == 1)
            if (m_bcoefs[amrlev][mglev][1] == nullptr) {
                m_bcoefs[amrlev][mglev][1] = std::make_unique<MultiFab>
                    (amrex::convert(m_grids[amrlev][mglev], m_etype[1]),
                     m_dmap[amrlev][mglev], 1, 0);
            }
            average_down_nodal(*m_bcoefs[amrlev][mglev-1][1],
                               *m_bcoefs[amrlev][mglev  ][1], ratio);
#endif
        }
    }

    for (auto& amrvec : m_lusolver) {
        for (auto& mgptr : amrvec) {
            mgptr.reset();
        }
    }
}

void MLCurlCurl::setAlpha (const Vector<MultiFab const*>& a_acoeffs)
{
    AMREX_ALWAYS_ASSERT(static_cast<int>(a_acoeffs.size()) == m_num_amr_levels);

    m_needs_update = true;

    auto lobc = LoBC();
    auto hibc = HiBC();
    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
        if (lobc[idim] != LinOpBCType::Periodic) { lobc[idim] = LinOpBCType::Neumann; }
        if (hibc[idim] != LinOpBCType::Periodic) { hibc[idim] = LinOpBCType::Neumann; }
    }

    for (int amrlev = 0; amrlev < m_num_amr_levels; ++amrlev) {
        AMREX_ALWAYS_ASSERT(a_acoeffs[amrlev]->is_nodal());
        MultiFab nodal_alpha(a_acoeffs[amrlev]->boxArray(),
                             a_acoeffs[amrlev]->DistributionMap(), 1, 1);
        MultiFab::Copy(nodal_alpha, *a_acoeffs[amrlev], 0, 0, 1, 0);
        nodal_alpha.FillBoundaryAndSync(m_geom[amrlev][0].periodicity());

        for (int mglev = 0; mglev < m_num_mg_levels[amrlev]; ++mglev) {

            Box nd_domain = amrex::surroundingNodes(m_geom[amrlev][mglev].Domain());
#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
            for (MFIter mfi(nodal_alpha); mfi.isValid(); ++mfi) {
                auto const& afab = nodal_alpha.array(mfi);
                Box const& box = mfi.validbox();
                mlndlap_applybc(box, afab, nd_domain, lobc, hibc);
            }

            auto const& anode = nodal_alpha.const_arrays();

            GpuArray<MultiArray4<Real>,3> aface;
            for (int idim = 0; idim < 3; ++idim) {
                IntVect typ(0);
                if (idim < AMREX_SPACEDIM) { typ[idim] = 1; }
                m_acoefs[amrlev][mglev][idim] = std::make_unique<MultiFab>
                    (amrex::convert(m_grids[amrlev][mglev], typ),
                     m_dmap[amrlev][mglev], 1, 1);
                aface[idim] = m_acoefs[amrlev][mglev][idim]->arrays();
            }

            amrex::ParallelFor(nodal_alpha, IntVect(1),
                               [=] AMREX_GPU_DEVICE (int b, int i, int j, int k)
            {
                auto const& an = anode[b];
                auto const& ax = aface[0][b];
                auto const& ay = aface[1][b];
                auto const& az = aface[2][b];
                if (ax.contains(i,j,k)) {
#if (AMREX_SPACEDIM == 1)
                    ax(i,0,0) = an(i,0,0);
#elif (AMREX_SPACEDIM == 2)
                    ax(i,j,0) = Real(0.5)*(an(i,j,0)+an(i,j+1,0));
#else
                    ax(i,j,k) = Real(0.25)*(an(i,j,k)+an(i,j+1,k)+an(i,j,k+1)+an(i,j+1,k+1));
#endif
                }
                if (ay.contains(i,j,k)) {
#if (AMREX_SPACEDIM == 1)
                    ay(i,0,0) = Real(0.5)*(an(i,0,0)+an(i+1,0,0));
#elif (AMREX_SPACEDIM == 2)
                    ay(i,j,0) = Real(0.5)*(an(i,j,0)+an(i+1,j,0));
#else
                    ay(i,j,k) = Real(0.25)*(an(i,j,k)+an(i+1,j,k)+an(i,j,k+1)+an(i+1,j,k+1));
#endif
                }
                if (az.contains(i,j,k)) {
#if (AMREX_SPACEDIM == 1)
                    az(i,0,0) = Real(0.5)*(an(i,0,0)+an(i+1,0,0));
#elif (AMREX_SPACEDIM >= 2)
                    az(i,j,k) = Real(0.25)*(an(i,j,k)+an(i+1,j,k)+an(i,j+1,k)+an(i+1,j+1,k));
#endif
                }
            });

            if (mglev+1 < m_num_mg_levels[amrlev]) {
                MultiFab tmp(amrex::convert(m_grids[amrlev][mglev+1], IntVect(1)),
                             m_dmap[amrlev][mglev+1], 1, 1);
                IntVect ratio = (amrlev > 0) ? IntVect(2) : mg_coarsen_ratio_vec[mglev];
                average_down_nodal(nodal_alpha, tmp, ratio);
                std::swap(nodal_alpha, tmp);
                nodal_alpha.FillBoundary(m_geom[amrlev][mglev+1].periodicity());
            }
        }
    }

    for (auto& amrvec : m_lusolver) {
        for (auto& mgptr : amrvec) {
            mgptr.reset();
        }
    }
}

void MLCurlCurl::clearAlphaMultiFab ()
{
    for (auto& amrvec : m_acoefs) {
        for (auto& arr : amrvec) {
            for (auto& mf : arr) {
                mf.reset();
            }
        }
    }
}

void MLCurlCurl::clearBetaMultiFab ()
{
    for (auto& amrvec : m_bcoefs) {
        for (auto& arr : amrvec) {
            for (auto& mf : arr) {
                mf.reset();
            }
        }
    }
}

void MLCurlCurl::prepareRHS (Vector<MF*> const& rhs) const
{
    for (int amrlev = 0; amrlev < m_num_amr_levels; ++amrlev) {
        for (auto& mf : *rhs[amrlev]) {
            mf.OverrideSync(m_geom[amrlev][0].periodicity());
        }
    }
}

void MLCurlCurl::setDirichletNodesToZero (int amrlev, int mglev, MF& a_mf) const
{
    MFItInfo mfi_info{};
#ifdef AMREX_USE_GPU
    Vector<Array4BoxTag<RT>> tags;
    mfi_info.DisableDeviceSync();
#endif

    for (auto& mf : a_mf)
    {
        auto const idxtype = mf.ixType();
        Box const domain = amrex::convert(m_geom[amrlev][mglev].Domain(), idxtype);

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
        for (MFIter mfi(mf,mfi_info); mfi.isValid(); ++mfi) {
            auto const& vbx = mfi.validbox();
            auto const& a = mf.array(mfi);
            for (OrientationIter oit; oit; ++oit) {
                Orientation const face = oit();
                int const idim = face.coordDir();
                bool is_dirichlet = face.isLow()
                    ? m_lobc[0][idim] == LinOpBCType::Dirichlet
                    : m_hibc[0][idim] == LinOpBCType::Dirichlet;
                if (is_dirichlet && domain[face] == vbx[face] &&
                    idxtype.nodeCentered(idim))
                {
                    Box b = vbx;
                    b.setRange(idim, vbx[face], 1);
#ifdef AMREX_USE_GPU
                    tags.emplace_back(Array4BoxTag<RT>{.dfab = a, .dbox = b});
#else
                    amrex::LoopOnCpu(b, [&] (int i, int j, int k)
                    {
                        a(i,j,k) = RT(0.0);
                    });
#endif
                }
            }
        }
    }

#ifdef AMREX_USE_GPU
    ParallelFor(tags,
    [=] AMREX_GPU_DEVICE (int i, int j, int k, Array4BoxTag<RT> const& tag) noexcept
    {
        tag.dfab(i,j,k) = RT(0.0);
    });
#endif
}

void MLCurlCurl::setLevelBC (int amrlev, const MF* levelbcdata,
                             const MF* robinbc_a, const MF* robinbc_b,
                             const MF* robinbc_f)
{
    amrex::ignore_unused(robinbc_a, robinbc_b, robinbc_f);

#if MLCC_CF_GALERKIN
    // Single-level + setCoarseFineBC case: treat the pre-registered
    // m_coarse_data_for_bc as the "crse bcdata" for this amrlev=0 solve.
    if (amrlev == 0) {
        if (!this->needsCoarseDataForBC()) { return; }
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
            this->m_coarse_data_for_bc != nullptr,
            "MLCurlCurl::setLevelBC: setCoarseFineBC must be called before "
            "setLevelBC(0, ...) when needsCoarseDataForBC() is true.");

        IntVect const ratio = this->m_coarse_data_crse_ratio;

        if (m_crse_sol_br[amrlev][0] == nullptr) {
            BoxArray crse_ba = m_grids[amrlev][0];
            crse_ba.coarsen(ratio);
            for (int idim = 0; idim < 3; ++idim) {
                m_crse_sol_br[amrlev][idim] = std::make_unique<MultiFab>
                    (amrex::convert(crse_ba, m_etype[idim]),
                     m_dmap[amrlev][0], 1, 1);
            }
        }

        for (int idim = 0; idim < 3; ++idim) {
            m_crse_sol_br[amrlev][idim]->ParallelCopy(
                (*this->m_coarse_data_for_bc)[idim], 0, 0, 1,
                IntVect(0), IntVect(1),
                m_geom[amrlev][0].periodicity());
        }
        m_has_cf_data[amrlev] = 1;
        return;
    }
#else
    if (amrlev == 0) { return; }
#endif

    IntVect ratio(this->AMRRefRatio(amrlev-1));

    // Allocate boundary register if needed
    if (m_crse_sol_br[amrlev][0] == nullptr) {
        BoxArray crse_ba = m_grids[amrlev][0];
        crse_ba.coarsen(ratio);
        for (int idim = 0; idim < 3; ++idim) {
            m_crse_sol_br[amrlev][idim] = std::make_unique<MultiFab>
                (amrex::convert(crse_ba, m_etype[idim]),
                 m_dmap[amrlev][0], 1, 1);
        }
    }

    if (levelbcdata != nullptr) {
        for (int idim = 0; idim < 3; ++idim) {
            m_crse_sol_br[amrlev][idim]->ParallelCopy(
                (*levelbcdata)[idim], 0, 0, 1,
                IntVect(0), IntVect(1),
                m_geom[amrlev-1][0].periodicity());
        }
        m_has_cf_data[amrlev] = 1;
    } else {
        for (int idim = 0; idim < 3; ++idim) {
            if (m_crse_sol_br[amrlev][idim]) {
                m_crse_sol_br[amrlev][idim]->setVal(Real(0.0));
            }
        }
        m_has_cf_data[amrlev] = 0;
    }
}

// ========================================================================
// CF mask building
// ========================================================================

void MLCurlCurl::buildCFMasks ()
{
    for (int amrlev = 0; amrlev < m_num_amr_levels; ++amrlev)
    {
        for (int mglev = 0; mglev < m_num_mg_levels[amrlev]; ++mglev)
        {
            auto const& period = m_geom[amrlev][mglev].periodicity();

            for (int idim = 0; idim < 3; ++idim)
            {
                BoxArray const edgeBA = amrex::convert(
                    m_grids[amrlev][mglev], m_etype[idim]);
                m_cfmask[amrlev][mglev][idim] =
                    std::make_unique<iMultiFab>(
                        edgeBA, m_dmap[amrlev][mglev], 1, 1);

                // Start: all ghosts marked as CF (1), valid cells as 1
                m_cfmask[amrlev][mglev][idim]->setVal(1);

                // Set valid cells to 0
#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
                for (MFIter mfi(*m_cfmask[amrlev][mglev][idim],
                                TilingIfNotGPU()); mfi.isValid(); ++mfi)
                {
                    Box const& vbx = mfi.validbox();
                    auto const& mask =
                        m_cfmask[amrlev][mglev][idim]->array(mfi);
                    ParallelFor(vbx,
                    [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                    {
                        mask(i,j,k) = 0;
                    });
                }

                // FillBoundary: ghost cells covered by same-level
                // neighbor valid cells become 0
                m_cfmask[amrlev][mglev][idim]->FillBoundary(period);

                // Ghost cells outside the physical domain are NOT CF
                auto const ixtype = m_etype[idim];
                Box const domain = amrex::convert(
                    m_geom[amrlev][mglev].Domain(), ixtype);

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
                for (MFIter mfi(*m_cfmask[amrlev][mglev][idim],
                                TilingIfNotGPU()); mfi.isValid(); ++mfi)
                {
                    Box const& gbx = amrex::grow(mfi.validbox(), 1);
                    auto const& mask =
                        m_cfmask[amrlev][mglev][idim]->array(mfi);
                    auto const dom = domain;
                    ParallelFor(gbx,
                    [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                    {
                        if (!dom.contains(IntVect(AMREX_D_DECL(i,j,k))))
                        {
                            mask(i,j,k) = 0;
                        }
                    });
                }
            }
        }
    }
}

// ========================================================================
// CF ghost fill
// ========================================================================

void MLCurlCurl::fillCoarseFineBoundary (int amrlev, MF& mf,
                                          bool homogeneous) const
{
    if (m_cfmask[amrlev][0][0] == nullptr) { return; }

    // mf may be a 0-ghost MultiFab (e.g., GMRES Krylov vector allocated
    // via makeVecRHS). CF-ghost fill writes into the +1 ghost, which
    // would be out-of-bounds; consumers of such a 0-ghost MF do not
    // read ghosts anyway, so skip.
    if (mf[0].nGrow() < 1) { return; }

#if MLCC_CF_GALERKIN
    IntVect ratio;
    if (amrlev == 0) {
        if (!this->needsCoarseDataForBC()) { return; }
        ratio = this->m_coarse_data_crse_ratio;
    } else {
        ratio = IntVect(this->AMRRefRatio(amrlev-1));
    }
#else
    if (amrlev == 0) { return; }
    IntVect ratio(this->AMRRefRatio(amrlev-1));
#endif

    for (int idim = 0; idim < 3; ++idim)
    {
        if (m_cfmask[amrlev][0][idim] == nullptr) { continue; }

        if (homogeneous || !m_has_cf_data[amrlev]
            || m_crse_sol_br[amrlev][0] == nullptr)
        {
            // Zero fill at CF ghosts
            auto const& sol = mf[idim].arrays();
            auto const& mask =
                m_cfmask[amrlev][0][idim]->const_arrays();
            ParallelFor(mf[idim], IntVect(1),
                [=] AMREX_GPU_DEVICE (int bno, int i, int j, int k)
            {
                if (mask[bno](i,j,k) != 0) {
                    sol[bno](i,j,k) = Real(0.0);
                }
            });
        }
        else
        {
            // Inhomogeneous: interpolate from coarse data at masked cells.
            // Build temporary on coarsened-fine layout, ParallelCopy from
            // boundary register, interpolate to fine.
            MultiFab const& crse = *m_crse_sol_br[amrlev][idim];
            BoxArray crse_ba = mf[idim].boxArray();
            crse_ba.coarsen(ratio);
            MultiFab crse_on_cfba(amrex::convert(crse_ba, m_etype[idim]),
                                  mf[idim].DistributionMap(), 1, 1);
            crse_on_cfba.ParallelCopy(crse, 0, 0, 1,
                                      IntVect(1), IntVect(1));

            auto const& sol = mf[idim].arrays();
            auto const& crsearr = crse_on_cfba.const_arrays();
            auto const& mask =
                m_cfmask[amrlev][0][idim]->const_arrays();
            int const dir = idim;
            ParallelFor(mf[idim], IntVect(1),
                [=] AMREX_GPU_DEVICE (int bno, int i, int j, int k)
            {
                if (mask[bno](i,j,k) != 0) {
                    mlcurlcurl_interpset(dir, i, j, k,
                                         sol[bno], crsearr[bno]);
                }
            });
        }
        Gpu::streamSynchronize();
    }
}

// ========================================================================
// applyBC with homogeneous flag
// ========================================================================

void MLCurlCurl::applyBC (int amrlev, int mglev, MF& in,
                           CurlCurlStateType type, bool homogeneous) const
{
    int nmfs = 3;
#if (AMREX_SPACEDIM == 2)
    if (CurlCurlStateType::b == type) { nmfs = 2; }
#elif (AMREX_SPACEDIM == 1)
    if (CurlCurlStateType::b == type) { nmfs = 1; }
#endif
    Vector<MultiFab*> mfs(nmfs);
    for (int imf = 0; imf < nmfs; ++imf) {
        mfs[imf] = in.data() + imf;
    }

    // Step 1: FillBoundary — same-level ghost exchange
    FillBoundary(mfs, this->m_geom[amrlev][mglev].periodicity());

    // Step 2: CF ghost fill (masked — only true CF ghosts are touched)
#if MLCC_CF_GALERKIN
    // Also fire at amrlev==0 for single-level + setCoarseFineBC, so that
    // CF ghosts get zeroed (homog) or interpolated-coarse (inhomog).
    bool const cf_fill =
        (mglev == 0) && (type != CurlCurlStateType::b)
        && ((amrlev > 0) ||
            (amrlev == 0 && this->needsCoarseDataForBC()));
    if (cf_fill) {
        fillCoarseFineBoundary(amrlev, in, homogeneous);
    }
#else
    if (amrlev > 0 && mglev == 0 && type != CurlCurlStateType::b) {
        fillCoarseFineBoundary(amrlev, in, homogeneous);
    }
#endif

    // Step 3: Physical BC
    for (auto* mf : mfs) {
        applyPhysBC(amrlev, mglev, *mf, type);
    }
}

// ========================================================================
// applyAmrLevelGalerkin — matrix-free (R L^f I) action at AMR-covered cells
// ========================================================================
//
// For staggered curl-curl with small β in free space, the AMR-coarsened
// rediscretized L^c is NOT variationally consistent with avg-down(L^f).
// The off-diagonal Ex–Ey cross-couplings of L_G = R L^f I differ
// structurally from L^c (Galerkin halves the same-cell coupling and
// zeros the cross-cell coupling that L^c writes as -α/dx/dy). This
// inconsistency, combined with the gradient null space of curl-curl
// (eigenvalue β = 0.001 in free space ⇒ |L^c⁻¹| ≈ 1000 along
// gradient modes), turns the V-cycle's cor at AMR-covered cells into
// inconsistent gradient-mode noise that diverges the iteration.
//
// This routine restores variational consistency by computing the
// Galerkin operator action directly: interpolate v from coarse to fine
// (mlcurlcurl_interpset), apply L^f via the fine-level apply (which
// fills CF ghosts from v), then restrict back (mlcurlcurl_restriction).
// The result is written into Ax_c at AMR-covered coarse cells via
// ParallelCopy from the coarsened-fine layout — uncov cells of Ax_c
// are untouched.
//
// Cost: one extra fine-level apply per V-cycle apply on the AMR-coarse
// level. Roughly doubles the cost of the V-cycle.
//
void MLCurlCurl::applyAmrLevelGalerkin (int crse_amrlev, MF& Ax_c,
                                        MF const& v_c) const
{
    BL_PROFILE("MLCurlCurl::applyAmrLevelGalerkin()");

    AMREX_ASSERT(crse_amrlev + 1 < m_num_amr_levels);

    int const flev = crse_amrlev + 1;
    IntVect const ratio(this->AMRRefRatio(crse_amrlev));
    AMREX_ALWAYS_ASSERT(ratio == 2);

    // Save state of m_crse_sol_br[flev] and m_has_cf_data[flev]. We
    // temporarily overwrite them so apply(flev, ...) below fills CF
    // ghosts from v_c. Restore on exit.
    int const saved_has_cf = m_has_cf_data[flev];
    Array<MultiFab,3> saved_brdata;
    bool const brdata_was_alloc = (m_crse_sol_br[flev][0] != nullptr);
    if (brdata_was_alloc) {
        for (int idim = 0; idim < 3; ++idim) {
            saved_brdata[idim].define(m_crse_sol_br[flev][idim]->boxArray(),
                                      m_crse_sol_br[flev][idim]->DistributionMap(),
                                      1, 1);
            MultiFab::Copy(saved_brdata[idim], *m_crse_sol_br[flev][idim],
                           0, 0, 1, IntVect(1));
        }
    } else {
        BoxArray crse_ba = m_grids[flev][0];
        crse_ba.coarsen(ratio);
        for (int idim = 0; idim < 3; ++idim) {
            m_crse_sol_br[flev][idim] = std::make_unique<MultiFab>
                (amrex::convert(crse_ba, m_etype[idim]),
                 m_dmap[flev][0], 1, 1);
        }
    }

    // Populate m_crse_sol_br[flev] from v_c. setVal(0) first so any
    // dest cells (e.g. ghosts outside the source's valid domain) that
    // ParallelCopy doesn't touch are well-defined rather than picking
    // up uninitialised memory from a freshly-allocated MultiFab.
    for (int idim = 0; idim < 3; ++idim) {
        m_crse_sol_br[flev][idim]->setVal(Real(0.0));
        m_crse_sol_br[flev][idim]->ParallelCopy(
            v_c[idim], 0, 0, 1,
            IntVect(0), IntVect(1),
            m_geom[crse_amrlev][0].periodicity());
    }
    m_has_cf_data[flev] = 1;

    // Build Iv (= I v_c) on fine layout with 1 ghost. Use the
    // interpolation kernel mlcurlcurl_interpset to fill valid fine
    // cells from m_crse_sol_br[flev]. Fine CF ghosts and other ghosts
    // are left at 0 here; they will be filled correctly by applyBC
    // inside the apply(flev, ...) call below (which fires
    // fillCoarseFineBoundary from m_crse_sol_br[flev] and
    // FillBoundary + applyPhysBC).
    //
    // Iteration must stay within m_crse_sol_br[flev]'s valid+1 ghost
    // range: interpset can read up to (ic+1, jc+1) from a fine cell at
    // (i, j) → max coarse index read = coarsen(i_fine_max, 2) + 1.
    // Iterating ng=0 (valid only) keeps the coarse reads within bounds.
    Array<MultiFab,3> Iv;
    for (int idim = 0; idim < 3; ++idim) {
        Iv[idim].define(amrex::convert(m_grids[flev][0], m_etype[idim]),
                        m_dmap[flev][0], 1, 1);
        Iv[idim].setVal(Real(0.0));

        auto const& iv_a = Iv[idim].arrays();
        auto const& crse_a = m_crse_sol_br[flev][idim]->const_arrays();
        int const dir = idim;
        ParallelFor(Iv[idim], IntVect(0),
            [=] AMREX_GPU_DEVICE (int bno, int i, int j, int k)
        {
            mlcurlcurl_interpset(dir, i, j, k, iv_a[bno], crse_a[bno]);
        });
    }
    Gpu::streamSynchronize();

    // L^f Iv on fine layout. apply() will fire applyBC, which sets CF
    // ghosts from m_crse_sol_br[flev] (= v_c). The CF ghosts we just
    // filled via interpset are overwritten, but with the same values.
    //
    // Allocate Lf_Iv with 1 ghost: the subsequent restriction kernel
    // mlcurlcurl_restriction reads fine cells at (ii±1, jj±1, kk±1)
    // (nodal restriction has the widest reach), so coarse cells at the
    // boundary of the coarsened-fine BA need the +1 fine-ghost cells.
    // setVal(0) on the ghosts; apply() writes valid only, and 0 ghost
    // values give the natural zero-extension for the restriction at
    // the coarsened-fine BA boundary.
    MF Lf_Iv;
    for (int idim = 0; idim < 3; ++idim) {
        Lf_Iv[idim].define(amrex::convert(m_grids[flev][0], m_etype[idim]),
                           m_dmap[flev][0], 1, 1);
        Lf_Iv[idim].setVal(Real(0.0));
    }
    apply(flev, 0, Lf_Iv, Iv, BCMode::Inhomogeneous, StateMode::Solution);

    // R Lf_Iv on coarsened-fine layout.
    BoxArray cba = m_grids[flev][0]; cba.coarsen(ratio);
    auto dinfo = getDirichletInfo(flev, 0);
    Array<MultiFab,3> R_Lf_Iv;
    for (int idim = 0; idim < 3; ++idim) {
        R_Lf_Iv[idim].define(amrex::convert(cba, m_etype[idim]),
                             m_dmap[flev][0], 1, 0);
        auto const& crsema = R_Lf_Iv[idim].arrays();
        auto const& finema = Lf_Iv[idim].const_arrays();
        int const dir = idim;
        ParallelFor(R_Lf_Iv[idim],
            [=] AMREX_GPU_DEVICE (int bno, int i, int j, int k)
        {
            mlcurlcurl_restriction(dir, i, j, k,
                                   crsema[bno], finema[bno], dinfo);
        });
    }
    Gpu::streamSynchronize();

    // Write into Ax_c at AMR-covered cells via ParallelCopy.
    for (int idim = 0; idim < 3; ++idim) {
        Ax_c[idim].ParallelCopy(R_Lf_Iv[idim], 0, 0, 1);
    }

    // Restore m_crse_sol_br[flev] and m_has_cf_data[flev].
    if (brdata_was_alloc) {
        for (int idim = 0; idim < 3; ++idim) {
            MultiFab::Copy(*m_crse_sol_br[flev][idim], saved_brdata[idim],
                           0, 0, 1, IntVect(1));
        }
    } else {
        for (int idim = 0; idim < 3; ++idim) {
            m_crse_sol_br[flev][idim]->setVal(Real(0.0));
        }
    }
    m_has_cf_data[flev] = saved_has_cf;
}

// ========================================================================
// smoothCovGalerkinJacobi — damped Jacobi at AMR-covered cells using
// the Galerkin operator. Pair this with the existing 4-block GS so the
// smoother is consistent with apply() at both uncov and cov.
// ========================================================================
void MLCurlCurl::smoothCovGalerkinJacobi (int amrlev, MF& sol, MF const& rhs) const
{
    BL_PROFILE("MLCurlCurl::smoothCovGalerkinJacobi()");

    // L_G[sol] on the coarse layout (overwrites only cov cells).
    MF Lg_sol;
    for (int idim = 0; idim < 3; ++idim) {
        Lg_sol[idim].define(sol[idim].boxArray(),
                            sol[idim].DistributionMap(), 1, 0);
        Lg_sol[idim].setVal(Real(0.0));
    }
    applyAmrLevelGalerkin(amrlev, Lg_sol, sol);

    // Damped Jacobi at cov:
    //   sol_cov += omega * (rhs_cov - L_G sol_cov) / diag
    // The rediscretized diagonal is a close approximation to the
    // Galerkin diagonal in our regime (the dominant Galerkin
    // correction is in off-diagonal cross-couplings).
    auto dxinv = this->m_geom[amrlev][0].InvCellSizeArray();
    auto adxinv = dxinv;
    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
        adxinv[idim] *= std::sqrt(m_alpha);
    }
    Real const dxx = adxinv[0] * adxinv[0];
#if (AMREX_SPACEDIM >= 2)
    Real const dyy = adxinv[1] * adxinv[1];
#else
    Real const dyy = Real(0.0);
#endif
#if (AMREX_SPACEDIM == 3)
    Real const dzz = adxinv[2] * adxinv[2];
#else
    Real const dzz = Real(0.0);
#endif
    Real const b_scalar = m_beta;
    bool const has_beta = (m_bcoefs[amrlev][0][0] != nullptr);

    auto dinfo = getDirichletInfo(amrlev, 0);
    static Real const omega = [] {
        char const* s = std::getenv("AMREX_MLCC_JACOBI_OMEGA");
        return s ? Real(std::atof(s)) : Real(0.5);
    }();

    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
        auto const& sol_a  = sol[idim].arrays();
        auto const& rhs_a  = rhs[idim].const_arrays();
        auto const& lg_a   = Lg_sol[idim].const_arrays();
        auto const& fmask  = m_fine_mask[amrlev][idim]->const_arrays();
        Array4<Real const> empty;
        auto const& beta_a = has_beta
            ? m_bcoefs[amrlev][0][idim]->const_arrays()
            : MultiArray4<Real const>{};
        int const dir = idim;
        // Per-edge rediscretized diagonal:
        //   Ex (idim=0): 2*dyy (+ 2*dzz in 3D) + β
        //   Ey (idim=1): 2*dxx (+ 2*dzz in 3D) + β
        //   Ez (idim=2): 2*(dxx + dyy) + β   (only 2D here for now)
        Real const diag_curl =
            (idim == 0) ? Real(2.0)*(dyy + dzz)
          : (idim == 1) ? Real(2.0)*(dxx + dzz)
          :               Real(2.0)*(dxx + dyy);

        ParallelFor(sol[idim],
            [=] AMREX_GPU_DEVICE (int bno, int i, int j, int k)
        {
            // cov cells only (fmask == 0 means covered by finer level).
            if (fmask[bno](i,j,k) != 0) return;
            if (dinfo.is_dirichlet_edge(dir, i, j, k)) return;

            Real const beta_local = has_beta ? beta_a[bno](i,j,k) : b_scalar;
            Real const diag = diag_curl + beta_local;
            Real const r = rhs_a[bno](i,j,k) - lg_a[bno](i,j,k);
            sol_a[bno](i,j,k) += omega * r / diag;
        });
    }
    Gpu::streamSynchronize();
}

// ========================================================================
// restriction, interpolation, interpolationAmr
// ========================================================================

void MLCurlCurl::restriction (int amrlev, int cmglev, MF& crse, MF& fine) const
{
    IntVect ratio = (amrlev > 0) ? IntVect(2)
        : this->mg_coarsen_ratio_vec[cmglev-1];
    AMREX_ALWAYS_ASSERT(ratio == 2);

    // Correction residual → homogeneous CF BCs
    applyBC(amrlev, cmglev-1, fine, CurlCurlStateType::r,
            /*homogeneous=*/true);

    auto dinfo = getDirichletInfo(amrlev,cmglev-1);

    for (int idim = 0; idim < 3; ++idim) {
        bool need_parallel_copy = !amrex::isMFIterSafe(crse[idim], fine[idim]);
        MultiFab cfine;
        if (need_parallel_copy) {
            BoxArray const& ba = amrex::coarsen(fine[idim].boxArray(), 2);
            cfine.define(ba, fine[idim].DistributionMap(), 1, 0);
        }

        MultiFab* pcrse = (need_parallel_copy) ? &cfine : &(crse[idim]);

        auto const& crsema = pcrse->arrays();
        auto const& finema = fine[idim].const_arrays();
        ParallelFor(*pcrse, [=] AMREX_GPU_DEVICE (int bno, int i, int j, int k)
        {
            mlcurlcurl_restriction(idim,i,j,k,crsema[bno],finema[bno],dinfo);
        });
        Gpu::streamSynchronize();

        if (need_parallel_copy) {
            crse[idim].ParallelCopy(cfine);
        }
    }
}

void MLCurlCurl::interpolation (int amrlev, int fmglev, MF& fine,
                                const MF& crse) const
{
    IntVect ratio = (amrlev > 0) ? IntVect(2)
        : this->mg_coarsen_ratio_vec[fmglev];
    AMREX_ALWAYS_ASSERT(ratio == 2);

    auto dinfo = getDirichletInfo(amrlev,fmglev);

    for (int idim = 0; idim < 3; ++idim) {
        bool need_parallel_copy = !amrex::isMFIterSafe(crse[idim], fine[idim]);
        MultiFab cfine;
        MultiFab const* cmf = &(crse[idim]);
        if (need_parallel_copy) {
            BoxArray const& ba = amrex::coarsen(fine[idim].boxArray(), 2);
            cfine.define(ba, fine[idim].DistributionMap(), 1, 0);
            cfine.ParallelCopy(crse[idim]);
            cmf = &cfine;
        }
        auto const& finema = fine[idim].arrays();
        auto const& crsema = cmf->const_arrays();
        ParallelFor(fine[idim],
            [=] AMREX_GPU_DEVICE (int bno, int i, int j, int k)
        {
            if (!dinfo.is_dirichlet_edge(idim,i,j,k)) {
                mlcurlcurl_interpadd(idim,i,j,k,finema[bno],crsema[bno]);
            }
        });
        Gpu::streamSynchronize();
    }
}

void MLCurlCurl::interpolationAmr (int famrlev, MF& fine, const MF& crse,
                                   IntVect const& nghost) const
{
    BL_PROFILE("MLCurlCurl::interpolationAmr()");

    IntVect ratio(this->AMRRefRatio(famrlev-1));
    AMREX_ALWAYS_ASSERT(ratio == 2);

    auto dinfo = getDirichletInfo(famrlev, 0);

    for (int idim = 0; idim < 3; ++idim)
    {
        BoxArray cfba = fine[idim].boxArray();
        if (nghost != IntVect(0)) { cfba.grow(nghost); }
        cfba.coarsen(ratio);

        MultiFab cfine(cfba, fine[idim].DistributionMap(), 1, 0);
        cfine.ParallelCopy(crse[idim], 0, 0, 1);

        fine[idim].setVal(Real(0.0));
        auto const& finema = fine[idim].arrays();
        auto const& crsema = cfine.const_arrays();
        auto const ng = nghost;
        auto const fdomain = amrex::convert(
            m_geom[famrlev][0].Domain(), m_etype[idim]);
        ParallelFor(fine[idim], ng,
            [=] AMREX_GPU_DEVICE (int bno, int i, int j, int k)
        {
            if (fdomain.contains(i,j,k)) {
                if (!dinfo.is_dirichlet_edge(idim, i, j, k)) {
                    mlcurlcurl_interpadd(idim, i, j, k,
                                         finema[bno], crsema[bno]);
                }
            }
        });
        Gpu::streamSynchronize();
    }
}

// ========================================================================
// apply — passes homogeneous from BCMode to applyBC
// ========================================================================

void
MLCurlCurl::apply (int amrlev, int mglev, MF& out, MF& in, BCMode bc_mode,
                   StateMode /*s_mode*/, const MLMGBndryT<MF>* /*bndry*/) const
{
    bool const homogeneous = (bc_mode == BCMode::Homogeneous);
    applyBC(amrlev, mglev, in, CurlCurlStateType::x, homogeneous);

    auto dxinv = this->m_geom[amrlev][mglev].InvCellSizeArray();
    auto adxinv = dxinv;
    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
        adxinv[idim] *= std::sqrt(m_alpha);
    }
    auto const b = m_beta;
    bool const has_beta = (m_bcoefs[amrlev][mglev][0] != nullptr);
    bool const has_alpha = (m_acoefs[amrlev][mglev][0] != nullptr);

    auto dinfo = getDirichletInfo(amrlev,mglev);
    int const coord = 0;

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(out[0],TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        Box const& xbx = mfi.tilebox(out[0].ixType().toIntVect());
        Box const& ybx = mfi.tilebox(out[1].ixType().toIntVect());
        Box const& zbx = mfi.tilebox(out[2].ixType().toIntVect());
        auto const& xout = out[0].array(mfi);
        auto const& yout = out[1].array(mfi);
        auto const& zout = out[2].array(mfi);
        auto const& xin = in[0].array(mfi);
        auto const& yin = in[1].array(mfi);
        auto const& zin = in[2].array(mfi);

        if (has_alpha) {
            Array4<Real const> bcx, bcy, bcz;
            if (has_beta) {
                bcx = m_bcoefs[amrlev][mglev][0]->const_array(mfi);
                bcy = m_bcoefs[amrlev][mglev][1]->const_array(mfi);
                bcz = m_bcoefs[amrlev][mglev][2]->const_array(mfi);
            }
            auto const afx = m_acoefs[amrlev][mglev][0]->const_array(mfi);
            auto const afy = m_acoefs[amrlev][mglev][1]->const_array(mfi);
            auto const afz = m_acoefs[amrlev][mglev][2]->const_array(mfi);
            amrex::ParallelFor(xbx, ybx, zbx,
            [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                if (dinfo.is_dirichlet_x_edge(i,j,k)) {
                    xout(i,j,k) = Real(0.0);
                } else {
                    Real beta = bcx ? bcx(i,j,k) : b;
                    mlcurlcurl_adotx_x_alpha(i,j,k,xout,xin,yin,zin,
                                             afy,afz,beta,dxinv);
                }
            },
            [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                if (dinfo.is_dirichlet_y_edge(i,j,k)) {
                    yout(i,j,k) = Real(0.0);
                } else {
                    Real beta = bcy ? bcy(i,j,k) : b;
                    mlcurlcurl_adotx_y_alpha(i,j,k,yout,xin,yin,zin,
                                             afx,afz,beta,dxinv
#if (AMREX_SPACEDIM < 3)
                                             ,coord
#endif
                                             );
                }
            },
            [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                if (dinfo.is_dirichlet_z_edge(i,j,k)) {
                    zout(i,j,k) = Real(0.0);
                } else {
                    Real beta = bcz ? bcz(i,j,k) : b;
                    mlcurlcurl_adotx_z_alpha(i,j,k,zout,xin,yin,zin,
                                             afx,afy,beta,dxinv
#if (AMREX_SPACEDIM < 3)
                                             ,coord
#endif
                                             );
                }
            });
        } else if (m_bcoefs[amrlev][mglev][0]) {
            auto const& bcx = m_bcoefs[amrlev][mglev][0]->const_array(mfi);
            auto const& bcy = m_bcoefs[amrlev][mglev][1]->const_array(mfi);
            auto const& bcz = m_bcoefs[amrlev][mglev][2]->const_array(mfi);
            amrex::ParallelFor(xbx, ybx, zbx,
            [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                if (dinfo.is_dirichlet_x_edge(i,j,k)) {
                    xout(i,j,k) = Real(0.0);
                } else {
                    mlcurlcurl_adotx_x(i,j,k,xout,xin,yin,zin,bcx(i,j,k),adxinv);
                }
            },
            [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                if (dinfo.is_dirichlet_y_edge(i,j,k)) {
                    yout(i,j,k) = Real(0.0);
                } else {
                    mlcurlcurl_adotx_y(i,j,k,yout,xin,yin,zin,bcy(i,j,k),adxinv);
                }
            },
            [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                if (dinfo.is_dirichlet_z_edge(i,j,k)) {
                    zout(i,j,k) = Real(0.0);
                } else {
                    mlcurlcurl_adotx_z(i,j,k,zout,xin,yin,zin,bcz(i,j,k),adxinv);
                }
            });
        } else {
            amrex::ParallelFor(xbx, ybx, zbx,
            [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                if (dinfo.is_dirichlet_x_edge(i,j,k)) {
                    xout(i,j,k) = Real(0.0);
                } else {
                    mlcurlcurl_adotx_x(i,j,k,xout,xin,yin,zin,b,adxinv);
                }
            },
            [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                if (dinfo.is_dirichlet_y_edge(i,j,k)) {
                    yout(i,j,k) = Real(0.0);
                } else {
                    mlcurlcurl_adotx_y(i,j,k,yout,xin,yin,zin,b,adxinv);
                }
            },
            [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                if (dinfo.is_dirichlet_z_edge(i,j,k)) {
                    zout(i,j,k) = Real(0.0);
                } else {
                    mlcurlcurl_adotx_z(i,j,k,zout,xin,yin,zin,b,adxinv);
                }
            });
        }

#if MLCC_CF_GALERKIN
#if (AMREX_SPACEDIM >= 2)
        // Galerkin diagonal correction at MG coarse levels (mglev>0) on
        // AMR levels with CF ghosts. Matches the kernel-level correction
        // applied inside smooth4's HasCF block — the operator must be
        // consistent with the smoother for the V-cycle to converge.
        // At mglev=0 we skip the diagonal correction here; the
        // applyAmrLevelGalerkin call below handles AMR-cov consistency.
        if (mglev > 0 && m_cfmask[amrlev][mglev][0] != nullptr)
        {
            auto const& exm = m_cfmask[amrlev][mglev][0]->const_array(mfi);
            auto const& eym = m_cfmask[amrlev][mglev][1]->const_array(mfi);
            Real const gdxx = adxinv[0]*adxinv[0];
            Real const gdyy = adxinv[1]*adxinv[1];
#if (AMREX_SPACEDIM == 3)
            auto const& ezm = m_cfmask[amrlev][mglev][2]->const_array(mfi);
            Real const gdzz = adxinv[2]*adxinv[2];
#endif
            amrex::ParallelFor(xbx,
            [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                if (dinfo.is_dirichlet_x_edge(i,j,k)) { return; }
                Real extra = Real(0.0);
                if (exm(i,j+1,k) == 1) { extra += gdyy; }
                if (exm(i,j-1,k) == 1) { extra += gdyy; }
#if (AMREX_SPACEDIM == 3)
                if (exm(i,j,k+1) == 1) { extra += gdzz; }
                if (exm(i,j,k-1) == 1) { extra += gdzz; }
#endif
                xout(i,j,k) += extra * xin(i,j,k);
            });
            amrex::ParallelFor(ybx,
            [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                if (dinfo.is_dirichlet_y_edge(i,j,k)) { return; }
                Real extra = Real(0.0);
                if (eym(i+1,j,k) == 1) { extra += gdxx; }
                if (eym(i-1,j,k) == 1) { extra += gdxx; }
#if (AMREX_SPACEDIM == 3)
                if (eym(i,j,k+1) == 1) { extra += gdzz; }
                if (eym(i,j,k-1) == 1) { extra += gdzz; }
#endif
                yout(i,j,k) += extra * yin(i,j,k);
            });
#if (AMREX_SPACEDIM == 3)
            amrex::ParallelFor(zbx,
            [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                if (dinfo.is_dirichlet_z_edge(i,j,k)) { return; }
                Real extra = Real(0.0);
                if (ezm(i+1,j,k) == 1) { extra += gdxx; }
                if (ezm(i-1,j,k) == 1) { extra += gdxx; }
                if (ezm(i,j+1,k) == 1) { extra += gdyy; }
                if (ezm(i,j-1,k) == 1) { extra += gdyy; }
                zout(i,j,k) += extra * zin(i,j,k);
            });
#endif
        }
#endif
#endif
    }

#if MLCC_CF_GALERKIN
#if (AMREX_SPACEDIM >= 2)
    // At mglev=0 on a non-finest AMR level, override Ax at the
    // AMR-covered region with the algebraic Galerkin action R L^f I
    // applied to in. This makes the AMR-coarse operator variationally
    // consistent with the fine operator over cov, removing the
    // wrong-direction component of V-cycle corrections that otherwise
    // drive composite divergence on jagged CF geometries.
    // Opt-in via env var; default off because (a) alone doesn't restore
    // composite-V-cycle convergence on curl-curl with small β in free
    // space — the 4-block edge GS smoother cannot damp the gradient
    // null space, so even a variationally-consistent operator at cov
    // does not give a contractive V-cycle. Documented in the analysis
    // file; needs a Hiptmair-style smoother to be useful.
    static int s_mlcc_galerkin_amr = [] {
        char const* s = std::getenv("AMREX_MLCC_AMR_GALERKIN");
        return s ? std::atoi(s) : 0;
    }();
    if (s_mlcc_galerkin_amr
        && mglev == 0
        && amrlev < m_num_amr_levels - 1
        && m_fine_mask[amrlev][0] != nullptr)
    {
        applyAmrLevelGalerkin(amrlev, out, in);
    }
#endif
#endif
}

// ========================================================================
// smooth — always homogeneous CF BCs (correction equation)
// ========================================================================

void MLCurlCurl::smooth (int amrlev, int mglev, MF& sol, const MF& rhs,
                         bool skip_fillboundary, int niter) const
{
    AMREX_ASSERT(rhs[0].nGrowVect().allGE(1));

    applyBC(amrlev, mglev, const_cast<MF&>(rhs), CurlCurlStateType::b);
#if (AMREX_SPACEDIM == 1)
    int ncolors = 2;
#else
    int ncolors = 4;
#endif

    for (int i = 0; i < niter; ++i) {
        for (int color = 0; color < ncolors; ++color) {
            if (!skip_fillboundary) {
                // Correction equation → homogeneous CF BCs
                applyBC(amrlev, mglev, sol, CurlCurlStateType::x,
                        /*homogeneous=*/true);
            }
            skip_fillboundary = false;
#if (AMREX_SPACEDIM == 1)
            smooth1D(amrlev, mglev, sol, rhs, color);
#else
            smooth4(amrlev, mglev, sol, rhs, color);
#endif
        }

#if MLCC_CF_GALERKIN
#if (AMREX_SPACEDIM >= 2)
        // After one full multi-color GS sweep, add a damped-Jacobi pass
        // at AMR-covered cells using the Galerkin operator R L^f I to
        // match the apply() operator at cov. Without this, smooth4
        // (rediscretized) and apply (Galerkin at cov) disagree → V-cycle
        // does not contract.
        //
        // We compute L_G[sol] via applyAmrLevelGalerkin into a temporary,
        // then update sol_cov += omega * (rhs - L_G sol) / diag, where
        // diag is the rediscretized diagonal (a close approximation to
        // the Galerkin diagonal in our regime — the dominant Galerkin
        // correction is in off-diagonal cross-couplings, not the
        // diagonal).
        static int s_smooth_cov_galerkin = [] {
            char const* s = std::getenv("AMREX_MLCC_SMOOTH_COV_GALERKIN");
            return s ? std::atoi(s) : 0;
        }();
        if (s_smooth_cov_galerkin
            && mglev == 0
            && amrlev < m_num_amr_levels - 1
            && m_fine_mask[amrlev][0] != nullptr)
        {
            smoothCovGalerkinJacobi(amrlev, sol, rhs);
        }
#endif
#endif
    }
}

// smooth4 and smooth1D are UNCHANGED from the original — the smoother
// updates ALL valid edges. CF ghost values provide the boundary condition
// through the stencil, just like physical boundary ghosts.

#if (AMREX_SPACEDIM == 1)
void MLCurlCurl::smooth1D (int amrlev, int mglev, MF& sol, MF const& rhs,
                           int color) const
{
    auto const& ex = sol[0].arrays();
    auto const& ey = sol[1].arrays();
    auto const& ez = sol[2].arrays();
    auto const& rhsx = rhs[0].const_arrays();
    auto const& rhsy = rhs[1].const_arrays();
    auto const& rhsz = rhs[2].const_arrays();

    auto b = m_beta;

    auto dinfo = getDirichletInfo(amrlev,mglev);
    auto dxinv = this->m_geom[amrlev][mglev].InvCellSizeArray();
    auto adxinv = dxinv;
    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
        adxinv[idim] *= std::sqrt(m_alpha);
    }

    int xhi = this->m_geom[amrlev][mglev].Domain().bigEnd(0);

    MultiFab nmf(amrex::convert(rhs[0].boxArray(),IntVect(1)),
                 rhs[0].DistributionMap(), 1, 0, MFInfo().SetAlloc(false));

    bool const has_beta = (m_bcoefs[amrlev][mglev][0] != nullptr);
    bool const has_alpha = (m_acoefs[amrlev][mglev][0] != nullptr);
    int const coord = 0;

    if (has_alpha && has_beta) {
        auto const& acy = m_acoefs[amrlev][mglev][1]->const_arrays();
        auto const& acz = m_acoefs[amrlev][mglev][2]->const_arrays();
        auto const& bcx = m_bcoefs[amrlev][mglev][0]->const_arrays();
        auto const& bcy = m_bcoefs[amrlev][mglev][1]->const_arrays();
        auto const& bcz = m_bcoefs[amrlev][mglev][2]->const_arrays();
        ParallelFor( nmf, [=] AMREX_GPU_DEVICE(int bno, int i, int j, int k)
        {
            bool valid_x = i <= xhi;
            mlcurlcurl_smooth_1d_alpha_beta(i,j,k,ex[bno],ey[bno],ez[bno],
                                            rhsx[bno],rhsy[bno],rhsz[bno],
                                            bcx[bno],bcy[bno],bcz[bno],
                                            dxinv,color,dinfo,valid_x,coord,
                                            acy[bno],acz[bno]);
        });
    } else if (has_alpha && !has_beta) {
        auto const& acy = m_acoefs[amrlev][mglev][1]->const_arrays();
        auto const& acz = m_acoefs[amrlev][mglev][2]->const_arrays();
        ParallelFor( nmf, [=] AMREX_GPU_DEVICE(int bno, int i, int j, int k)
        {
            bool valid_x = i <= xhi;
            mlcurlcurl_smooth_1d_alpha(i,j,k,ex[bno],ey[bno],ez[bno],
                                       rhsx[bno],rhsy[bno],rhsz[bno],
                                       b,
                                       dxinv,color,dinfo,valid_x,coord,
                                       acy[bno],acz[bno]);
        });
    } else if (has_beta) {
        auto const& bcx = m_bcoefs[amrlev][mglev][0]->const_arrays();
        auto const& bcy = m_bcoefs[amrlev][mglev][1]->const_arrays();
        auto const& bcz = m_bcoefs[amrlev][mglev][2]->const_arrays();
        ParallelFor( nmf, [=] AMREX_GPU_DEVICE(int bno, int i, int j, int k)
        {
            bool valid_x = i <= xhi;
            mlcurlcurl_1D(i,j,k,ex[bno],ey[bno],ez[bno],
                          rhsx[bno],rhsy[bno],rhsz[bno],
                          bcx[bno],bcy[bno],bcz[bno],
                          adxinv,color,dinfo,valid_x);
        });
    } else {
        ParallelFor( nmf, [=] AMREX_GPU_DEVICE(int bno, int i, int j, int k)
        {
            bool valid_x = i <= xhi;
            mlcurlcurl_1D(i,j,k,ex[bno],ey[bno],ez[bno],
                          rhsx[bno],rhsy[bno],rhsz[bno],
                          b,adxinv,color,dinfo,valid_x);
        });
    }
    Gpu::streamSynchronize();
}
#endif

#if (AMREX_SPACEDIM > 1)
void MLCurlCurl::smooth4 (int amrlev, int mglev, MF& sol, MF const& rhs,
                          int color) const
{
    auto const& ex = sol[0].arrays();
    auto const& ey = sol[1].arrays();
    auto const& ez = sol[2].arrays();
    auto const& rhsx = rhs[0].const_arrays();
    auto const& rhsy = rhs[1].const_arrays();
    auto const& rhsz = rhs[2].const_arrays();

#if (AMREX_SPACEDIM == 2)
    auto b = m_beta;
#endif

    auto dxinv = this->m_geom[amrlev][mglev].InvCellSizeArray();
    auto adxinv = dxinv;
    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
        adxinv[idim] *= std::sqrt(m_alpha);
    }

    bool const has_beta = (m_bcoefs[amrlev][mglev][0] != nullptr);
    bool const has_alpha = (m_acoefs[amrlev][mglev][0] != nullptr);

    auto dinfo = getDirichletInfo(amrlev,mglev);
    auto sinfo = getSymmetryInfo(amrlev,mglev);

    MultiFab nmf(amrex::convert(rhs[0].boxArray(),IntVect(1)),
                 rhs[0].DistributionMap(), 1, 0, MFInfo().SetAlloc(false));
    if (m_lusolver[amrlev][mglev] && !has_alpha && !has_beta) {
        auto* plusolver = m_lusolver[amrlev][mglev]->dataPtr();
        ParallelFor(nmf, [=] AMREX_GPU_DEVICE (int bno, int i, int j, int k)
        {
            mlcurlcurl_gs4_lu(i,j,k,ex[bno],ey[bno],ez[bno],
                              rhsx[bno],rhsy[bno],rhsz[bno],
#if (AMREX_SPACEDIM == 2)
                              b,
#endif
                              adxinv,color,*plusolver,dinfo,sinfo);
        });
    } else if (has_alpha && has_beta) {
        auto const& acx = m_acoefs[amrlev][mglev][0]->const_arrays();
        auto const& acy = m_acoefs[amrlev][mglev][1]->const_arrays();
        auto const& acz = m_acoefs[amrlev][mglev][2]->const_arrays();
        auto const bb = m_beta;
        auto const& bcx = m_bcoefs[amrlev][mglev][0]->const_arrays();
        auto const& bcy = m_bcoefs[amrlev][mglev][1]->const_arrays();
        auto const& bcz = m_bcoefs[amrlev][mglev][2]->const_arrays();
        ParallelFor(nmf, [=] AMREX_GPU_DEVICE (int bno, int i, int j, int k)
        {
            mlcurlcurl_gs4_alpha(i,j,k,ex[bno],ey[bno],ez[bno],
                                 rhsx[bno],rhsy[bno],rhsz[bno],
                                 dxinv,color,
                                 acx[bno],acy[bno],acz[bno],
                                 bcx[bno],bcy[bno],bcz[bno],
                                 bb,dinfo,sinfo);
        });
    } else if (has_alpha && !has_beta) {
        auto const& acx = m_acoefs[amrlev][mglev][0]->const_arrays();
        auto const& acy = m_acoefs[amrlev][mglev][1]->const_arrays();
        auto const& acz = m_acoefs[amrlev][mglev][2]->const_arrays();
        auto const bb = m_beta;
        ParallelFor(nmf, [=] AMREX_GPU_DEVICE (int bno, int i, int j, int k)
        {
            Array4<Real const> empty;
            mlcurlcurl_gs4_alpha(i,j,k,ex[bno],ey[bno],ez[bno],
                                 rhsx[bno],rhsy[bno],rhsz[bno],
                                 dxinv,color,
                                 acx[bno],acy[bno],acz[bno],
                                 empty,empty,empty,
                                 bb,dinfo,sinfo);
        });
    } else {
        auto const& bcx = m_bcoefs[amrlev][mglev][0]->const_arrays();
        auto const& bcy = m_bcoefs[amrlev][mglev][1]->const_arrays();
        auto const& bcz = m_bcoefs[amrlev][mglev][2]->const_arrays();

        // Bug-#1 fix: when the cfmask exists for this (amrlev, mglev),
        // dispatch to the HasCF=true variant of mlcurlcurl_gs4 so that
        // the per-node 4-block LU enforces v=0 on CF-ghost edge unknowns
        // instead of solving for them (only to have applyBC zero them
        // back to 0 between sweeps, leaving the rest of the block
        // inconsistent). The cfmask is built only for multi-AMR-level
        // solves; for single-level it's null and we use the HasCF=false
        // variant (original behaviour).
        bool const have_cf =
#if MLCC_CF_GALERKIN
            // Include single-level + setCoarseFineBC case (m_num_amr_levels==1
            // but cfmask has been built via the needsCoarseDataForBC branch
            // in prepareForSolve).
            (m_cfmask[amrlev][mglev][0] != nullptr)
         && (m_cfmask[amrlev][mglev][1] != nullptr)
         && (m_cfmask[amrlev][mglev][2] != nullptr);
#else
            (m_num_amr_levels > 1)
         && (m_cfmask[amrlev][mglev][0] != nullptr)
         && (m_cfmask[amrlev][mglev][1] != nullptr)
         && (m_cfmask[amrlev][mglev][2] != nullptr);
#endif

        bool const mg_coarse = (mglev > 0);

        if (m_use_pcg) {
            if (have_cf) {
                auto const& cfmx = m_cfmask[amrlev][mglev][0]->const_arrays();
                auto const& cfmy = m_cfmask[amrlev][mglev][1]->const_arrays();
                auto const& cfmz = m_cfmask[amrlev][mglev][2]->const_arrays();
                ParallelFor(nmf, [=] AMREX_GPU_DEVICE (int bno, int i, int j, int k)
                {
                    mlcurlcurl_gs4<true,true>(i,j,k,ex[bno],ey[bno],ez[bno],
                                              rhsx[bno],rhsy[bno],rhsz[bno],
                                              adxinv,color,bcx[bno],bcy[bno],bcz[bno],
                                              dinfo,sinfo,
                                              cfmx[bno],cfmy[bno],cfmz[bno],
                                              mg_coarse);
                });
            } else {
                ParallelFor(nmf, [=] AMREX_GPU_DEVICE (int bno, int i, int j, int k)
                {
                    mlcurlcurl_gs4<true>(i,j,k,ex[bno],ey[bno],ez[bno],
                                         rhsx[bno],rhsy[bno],rhsz[bno],
                                         adxinv,color,bcx[bno],bcy[bno],bcz[bno],
                                         dinfo,sinfo);
                });
            }
        } else {
            if (have_cf) {
                auto const& cfmx = m_cfmask[amrlev][mglev][0]->const_arrays();
                auto const& cfmy = m_cfmask[amrlev][mglev][1]->const_arrays();
                auto const& cfmz = m_cfmask[amrlev][mglev][2]->const_arrays();
                ParallelFor(nmf, [=] AMREX_GPU_DEVICE (int bno, int i, int j, int k)
                {
                    mlcurlcurl_gs4<false,true>(i,j,k,ex[bno],ey[bno],ez[bno],
                                               rhsx[bno],rhsy[bno],rhsz[bno],
                                               adxinv,color,bcx[bno],bcy[bno],bcz[bno],
                                               dinfo,sinfo,
                                               cfmx[bno],cfmy[bno],cfmz[bno],
                                               mg_coarse);
                });
            } else {
                ParallelFor(nmf, [=] AMREX_GPU_DEVICE (int bno, int i, int j, int k)
                {
                    mlcurlcurl_gs4<false>(i,j,k,ex[bno],ey[bno],ez[bno],
                                          rhsx[bno],rhsy[bno],rhsz[bno],
                                          adxinv,color,bcx[bno],bcy[bno],bcz[bno],
                                          dinfo,sinfo);
                });
            }
        }
    }
    Gpu::streamSynchronize();
}
#endif

// ========================================================================
// solutionResidual — inhomogeneous CF BCs from coarse data
// ========================================================================

void MLCurlCurl::solutionResidual (int amrlev, MF& resid, MF& x, const MF& b,
                                   const MF* crse_bcdata)
{
    BL_PROFILE("MLCurlCurl::solutionResidual()");

    // Store coarse BC data for inhomogeneous CF ghost fill
    if (amrlev > 0 && crse_bcdata != nullptr) {
        setLevelBC(amrlev, crse_bcdata);
    }

    const int mglev = 0;
    // Inhomogeneous → CF ghosts filled from coarse solution
    apply(amrlev, mglev, resid, x, BCMode::Inhomogeneous, StateMode::Solution);
    compresid(amrlev, mglev, resid, b);
}

// ========================================================================
// correctionResidual — passes through BCMode
// ========================================================================

void MLCurlCurl::correctionResidual (int amrlev, int mglev, MF& resid, MF& x,
                                     const MF& b, BCMode bc_mode,
                                     const MF* crse_bcdata)
{
    if (amrlev > 0 && mglev == 0 && crse_bcdata != nullptr) {
        setLevelBC(amrlev, crse_bcdata);
    } else if (amrlev > 0 && mglev == 0) {
        setLevelBC(amrlev, nullptr); // homogeneous
    }

    apply(amrlev, mglev, resid, x, bc_mode, StateMode::Correction);
    compresid(amrlev, mglev, resid, b);
}

void MLCurlCurl::compresid (int amrlev, int mglev, MF& resid, MF const& b) const
{
    auto dinfo = getDirichletInfo(amrlev,mglev);

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(resid[0],TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        Box const& xbx = mfi.tilebox(resid[0].ixType().toIntVect());
        Box const& ybx = mfi.tilebox(resid[1].ixType().toIntVect());
        Box const& zbx = mfi.tilebox(resid[2].ixType().toIntVect());
        auto const& resx = resid[0].array(mfi);
        auto const& resy = resid[1].array(mfi);
        auto const& resz = resid[2].array(mfi);
        auto const& bx = b[0].array(mfi);
        auto const& by = b[1].array(mfi);
        auto const& bz = b[2].array(mfi);
        amrex::ParallelFor(xbx, ybx, zbx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k)
        {
            if (dinfo.is_dirichlet_x_edge(i,j,k)) {
                resx(i,j,k) = Real(0.0);
            } else {
                resx(i,j,k) = bx(i,j,k) - resx(i,j,k);
            }
        },
        [=] AMREX_GPU_DEVICE (int i, int j, int k)
        {
            if (dinfo.is_dirichlet_y_edge(i,j,k)) {
                resy(i,j,k) = Real(0.0);
            } else {
                resy(i,j,k) = by(i,j,k) - resy(i,j,k);
            }
        },
        [=] AMREX_GPU_DEVICE (int i, int j, int k)
        {
            if (dinfo.is_dirichlet_z_edge(i,j,k)) {
                resz(i,j,k) = Real(0.0);
            } else {
                resz(i,j,k) = bz(i,j,k) - resz(i,j,k);
            }
        });
    }
}

// ========================================================================
// reflux — fix fringe edges of coarse residual
// ========================================================================

void MLCurlCurl::reflux (int crse_amrlev, MF& res,
                         const MF& crse_sol, const MF& crse_rhs,
                         MF& fine_res, MF& fine_sol,
                         const MF& fine_rhs) const
{
    BL_PROFILE("MLCurlCurl::reflux()");
    amrex::ignore_unused(fine_rhs);

    IntVect ratio(this->AMRRefRatio(crse_amrlev));
    int const famrlev = crse_amrlev + 1;

    // Same-level FillBoundary + physical BCs + zero CF ghost (CF ghost
    // will be refilled below from the composite-corrected coarse residual).
    applyBC(famrlev, 0, fine_res, CurlCurlStateType::r,
            /*homogeneous=*/true);

    // Build composite solution: crse_sol with covered region replaced
    // by averaged-down fine_sol. Note: after averageDownAndSync (called
    // by MLMG before compResidual), crse_sol already contains the
    // averaged-down fine sol in the covered region. So composite_sol
    // is effectively crse_sol. We build it explicitly for correctness.
    MF composite_sol;
    for (int idim = 0; idim < 3; ++idim) {
        composite_sol[idim].define(crse_sol[idim].boxArray(),
                                   crse_sol[idim].DistributionMap(),
                                   1, IntVect(1));
        MultiFab::Copy(composite_sol[idim], crse_sol[idim], 0, 0, 1, IntVect(1));
    }

    // Average fine solution to covered coarse region
    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
        BoxArray cba = fine_sol[idim].boxArray();
        cba.coarsen(ratio);
        MultiFab ctmp(amrex::convert(cba, m_etype[idim]),
                      fine_sol[idim].DistributionMap(), 1, 0);
        average_down_edges(fine_sol[idim], ctmp, ratio);
        composite_sol[idim].ParallelCopy(ctmp, 0, 0, 1);
    }
#if (AMREX_SPACEDIM < 3)
    {
        BoxArray cba = fine_sol[2].boxArray();
        cba.coarsen(ratio);
        MultiFab ctmp(amrex::convert(cba, m_etype[2]),
                      fine_sol[2].DistributionMap(), 1, 0);
        average_down_nodal(fine_sol[2], ctmp, ratio);
        composite_sol[2].ParallelCopy(ctmp, 0, 0, 1);
    }
#endif
#if (AMREX_SPACEDIM == 1)
    {
        BoxArray cba = fine_sol[1].boxArray();
        cba.coarsen(ratio);
        MultiFab ctmp(amrex::convert(cba, m_etype[1]),
                      fine_sol[1].DistributionMap(), 1, 0);
        average_down_nodal(fine_sol[1], ctmp, ratio);
        composite_sol[1].ParallelCopy(ctmp, 0, 0, 1);
    }
#endif

    // Re-apply coarse operator to composite solution.
    // applyBC inside apply will FillBoundary (propagating averaged-down
    // data to ghost cells) and applyPhysBC.
    MF Ax_comp;
    for (int idim = 0; idim < 3; ++idim) {
        Ax_comp[idim].define(res[idim].boxArray(),
                             res[idim].DistributionMap(), 1, 0);
    }
    apply(crse_amrlev, 0, Ax_comp, composite_sol,
          BCMode::Inhomogeneous, StateMode::Solution);

    // Overwrite coarse residual at uncovered edges with the
    // composite-stencil value crse_rhs - L^c[composite_sol]. Covered
    // coarse edges are handled by avgDownResAmr restricting the fine
    // residual.
    auto dinfo = getDirichletInfo(crse_amrlev, 0);

    if (crse_amrlev < m_num_amr_levels - 1
        && m_fine_mask[crse_amrlev][0] != nullptr)
    {
        for (int idim = 0; idim < 3; ++idim) {
            auto const& res_a  = res[idim].arrays();
            auto const& rhs_a  = crse_rhs[idim].const_arrays();
            auto const& ax_a   = Ax_comp[idim].const_arrays();
            auto const& fmask  = m_fine_mask[crse_amrlev][idim]->const_arrays();
            ParallelFor(res[idim],
                [=] AMREX_GPU_DEVICE (int bno, int i, int j, int k)
            {
                if (fmask[bno](i,j,k) != 0
                 && !dinfo.is_dirichlet_edge(idim, i, j, k)) {
                    res_a[bno](i,j,k) = rhs_a[bno](i,j,k)
                                       - ax_a[bno](i,j,k);
                }
            });
        }
        Gpu::streamSynchronize();
    }
}

// ========================================================================
// avgDownResAmr — restrict fine residual to covered coarse edges
// ========================================================================

void MLCurlCurl::avgDownResAmr (int clev, MF& cres, MF const& fres) const
{
    BL_PROFILE("MLCurlCurl::avgDownResAmr()");

    int const flev = clev + 1;
    IntVect ratio(this->AMRRefRatio(clev));
    auto dinfo = getDirichletInfo(flev, 0);

    // mlcurlcurl_restriction reads fres at (ii±1, jj±1, kk±1). Coarse
    // cells at the boundary of the coarsened-fine BoxArray need the +1
    // fine-side ghost. Most callers pass fres with ≥1 ghost; GMRES's
    // makeVecRHS allocates 0-ghost vectors though, so when MLMG::apply
    // routes through this routine the fres argument may have no
    // ghosts. In that case, materialise a local 1-ghost copy with
    // zero-extended ghosts so the restriction stencil reads in bounds.
    bool const need_ghost_copy = (fres[0].nGrow() < 1);
    MF fres_local;
    MF const* pfres = &fres;
    if (need_ghost_copy) {
        for (int idim = 0; idim < 3; ++idim) {
            fres_local[idim].define(fres[idim].boxArray(),
                                    fres[idim].DistributionMap(), 1, 1);
            fres_local[idim].setVal(Real(0.0));
            MultiFab::Copy(fres_local[idim], fres[idim], 0, 0, 1, IntVect(0));
        }
        pfres = &fres_local;
    }
    MF const& fres_use = *pfres;

    for (int idim = 0; idim < 3; ++idim) {
        BoxArray cfba = fres_use[idim].boxArray();
        cfba.coarsen(ratio);

        MultiFab crse_from_fine(amrex::convert(cfba, m_etype[idim]),
                                fres_use[idim].DistributionMap(), 1, 0);

        auto const& crsema = crse_from_fine.arrays();
        auto const& finema = fres_use[idim].const_arrays();
        ParallelFor(crse_from_fine,
            [=] AMREX_GPU_DEVICE (int bno, int i, int j, int k)
        {
            mlcurlcurl_restriction(idim, i, j, k,
                                   crsema[bno], finema[bno], dinfo);
        });
        Gpu::streamSynchronize();

        // Overwrite covered region of coarse residual
        cres[idim].ParallelCopy(crse_from_fine, 0, 0, 1);
    }
}

// ========================================================================
// averageDownSolutionRHS
// ========================================================================

void MLCurlCurl::averageDownSolutionRHS (int camrlev, MF& crse_sol,
                                          MF& crse_rhs,
                                          const MF& fine_sol,
                                          const MF& fine_rhs)
{
    BL_PROFILE("MLCurlCurl::averageDownSolutionRHS()");

    IntVect ratio(this->AMRRefRatio(camrlev));

    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
        average_down_edges(fine_sol[idim], crse_sol[idim], ratio);
        average_down_edges(fine_rhs[idim], crse_rhs[idim], ratio);
    }
#if (AMREX_SPACEDIM < 3)
    average_down_nodal(fine_sol[2], crse_sol[2], ratio);
    average_down_nodal(fine_rhs[2], crse_rhs[2], ratio);
#endif
#if (AMREX_SPACEDIM == 1)
    average_down_nodal(fine_sol[1], crse_sol[1], ratio);
    average_down_nodal(fine_rhs[1], crse_rhs[1], ratio);
#endif

    auto const& period = m_geom[camrlev][0].periodicity();
    for (int idim = 0; idim < 3; ++idim) {
        amrex::OverrideSync(crse_sol[idim],
                            getDotMask(camrlev, 0, idim), period);
        amrex::OverrideSync(crse_rhs[idim],
                            getDotMask(camrlev, 0, idim), period);
    }
    for (int idim = 0; idim < 3; ++idim) {
    crse_sol[idim].FillBoundary(period);
    crse_rhs[idim].FillBoundary(period);
    }
}

// ========================================================================
// averageDownAndSync
// ========================================================================

void MLCurlCurl::averageDownAndSync (Vector<MF>& sol) const
{
    BL_PROFILE("MLCurlCurl::averageDownAndSync()");

    // Average down from fine to coarse AMR levels
    for (int falev = int(sol.size()) - 1; falev > 0; --falev) {
        IntVect ratio(this->AMRRefRatio(falev-1));
        for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
            average_down_edges(sol[falev][idim], sol[falev-1][idim], ratio);
        }
#if (AMREX_SPACEDIM < 3)
        average_down_nodal(sol[falev][2], sol[falev-1][2], ratio);
#endif
#if (AMREX_SPACEDIM == 1)
        average_down_nodal(sol[falev][1], sol[falev-1][1], ratio);
#endif
    }

    // Sync each level
    for (int amrlev = 0; amrlev < int(sol.size()); ++amrlev) {
        for (int idim = 0; idim < 3; ++idim) {
            amrex::OverrideSync(sol[amrlev][idim],
                                getDotMask(amrlev, 0, idim),
                                this->m_geom[amrlev][0].periodicity());
        }
        for (int idim = 0; idim < 3; ++idim) {
        sol[amrlev][idim].FillBoundary(this->m_geom[amrlev][0].periodicity());
        }
    }
}

// ========================================================================
// xdoty, normInf, dotProductPrecond, norm2Precond — exclude covered
// coarse edges. The *Precond variants are multi-AMR-level (sum over
// AMR levels) and are used by GMRESMLMG when MLMG is used as a
// preconditioner inside an outer Krylov method.
// ========================================================================

Real MLCurlCurl::dotProductPrecond (Vector<MF const*> const& x,
                                    Vector<MF const*> const& y) const
{
    Real local_sum = Real(0.0);
    for (int alev = 0; alev < m_num_amr_levels; ++alev) {
        local_sum += xdoty(alev, 0, *x[alev], *y[alev], /*local=*/true);
    }
    ParallelAllReduce::Sum(local_sum, ParallelContext::CommunicatorSub());
    return local_sum;
}

Real MLCurlCurl::norm2Precond (Vector<MF const*> const& x) const
{
    Real s = dotProductPrecond(x, x);
    return std::sqrt(s);
}

Real MLCurlCurl::xdoty (int amrlev, int mglev, const MF& x, const MF& y,
                        bool local) const
{
    auto result = Real(0.0);
    for (int idim = 0; idim < 3; ++idim) {
        auto rtmp = MultiFab::Dot(getDotMask(amrlev,mglev,idim),
                                  x[idim], 0, y[idim], 0, 1, 0, true);
        result += rtmp;
    }

    // Subtract contribution from edges covered by finer levels
    if (mglev == 0 && amrlev < m_num_amr_levels - 1
        && m_fine_mask[amrlev][0] != nullptr)
    {
        for (int idim = 0; idim < 3; ++idim) {
            auto const& xma = x[idim].const_arrays();
            auto const& yma = y[idim].const_arrays();
            auto const& dma = getDotMask(amrlev,mglev,idim).const_arrays();
            auto const& fma = m_fine_mask[amrlev][idim]->const_arrays();

            ReduceOps<ReduceOpSum> reduce_op;
            ReduceData<Real> reduce_data(reduce_op);
            using ReduceTuple = typename decltype(reduce_data)::Type;

            reduce_op.eval(x[idim], IntVect(0), reduce_data,
                [=] AMREX_GPU_DEVICE (int bno, int i, int j, int k)
                    -> ReduceTuple
            {
                if (fma[bno](i,j,k) == 0) {
                    return {dma[bno](i,j,k) * xma[bno](i,j,k) * yma[bno](i,j,k)};
                }
                return {Real(0.0)};
            });

            ReduceTuple rv = reduce_data.value(reduce_op);
            result -= amrex::get<0>(rv);
        }
    }

    if (!local) {
        ParallelAllReduce::Sum(result, ParallelContext::CommunicatorSub());
    }
    return result;
}

Real MLCurlCurl::normInf (int amrlev, MF const& mf, bool local) const
{
    Real result = Real(0.0);

    if (amrlev < m_num_amr_levels - 1 && m_fine_mask[amrlev][0] != nullptr) {
        for (int idim = 0; idim < 3; ++idim) {
            auto const& mfa = mf[idim].const_arrays();
            auto const& fma = m_fine_mask[amrlev][idim]->const_arrays();

            ReduceOps<ReduceOpMax> reduce_op;
            ReduceData<Real> reduce_data(reduce_op);
            using ReduceTuple = typename decltype(reduce_data)::Type;

            reduce_op.eval(mf[idim], IntVect(0), reduce_data,
                [=] AMREX_GPU_DEVICE (int bno, int i, int j, int k)
                    -> ReduceTuple
            {
                if (fma[bno](i,j,k) != 0) {
                    return {amrex::Math::abs(mfa[bno](i,j,k))};
                }
                return {Real(0.0)};
            });

            ReduceTuple rv = reduce_data.value(reduce_op);
            result = std::max(result, amrex::get<0>(rv));
        }
    } else {
        result = amrex::norminf(mf, 0, m_ncomp, IntVect(0), true);
    }

    if (!local) {
        ParallelAllReduce::Max(result, ParallelContext::CommunicatorSub());
    }
    return result;
}

// ========================================================================
// buildFineMask
// ========================================================================

void MLCurlCurl::buildFineMask ()
{
    m_fine_mask.resize(m_num_amr_levels);

    for (int amrlev = 0; amrlev < m_num_amr_levels - 1; ++amrlev) {
        IntVect ratio(this->AMRRefRatio(amrlev));

        for (int idim = 0; idim < 3; ++idim) {
            BoxArray cfba = amrex::convert(m_grids[amrlev+1][0], m_etype[idim]);
            cfba.coarsen(ratio);

            BoxArray const& cba = amrex::convert(m_grids[amrlev][0],
                                                  m_etype[idim]);

            m_fine_mask[amrlev][idim] = std::make_unique<iMultiFab>(
                cba, m_dmap[amrlev][0], 1, 0);
            m_fine_mask[amrlev][idim]->setVal(1);

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
            for (MFIter mfi(*m_fine_mask[amrlev][idim]); mfi.isValid(); ++mfi)
            {
                auto const& mask = m_fine_mask[amrlev][idim]->array(mfi);
                Box const& vbx = mfi.validbox();
                for (int j = 0; j < cfba.size(); ++j) {
                    Box const& isect = vbx & cfba[j];
                    if (isect.ok()) {
                        amrex::ParallelFor(isect,
                            [=] AMREX_GPU_DEVICE (int ii, int jj, int kk)
                        {
                            mask(ii,jj,kk) = 0;
                        });
                    }
                }
            }
            Gpu::streamSynchronize();
        }
    }
}

// ========================================================================
// prepareForSolve, update
// ========================================================================

void MLCurlCurl::prepareForSolve ()
{
    update_lusolver();
#if MLCC_CF_GALERKIN
    // Also build CF masks when a single-level MLCurlCurl is being used with
    // setCoarseFineBC (the caller's level-by-level pattern in ImplicitFD.cpp).
    // Without this, m_num_amr_levels==1 skips cfmask construction, have_cf
    // stays false, smooth4 runs the non-HasCF kernel, and CF-ghost β/ex
    // reads crash on DEBUG/FPE.
    bool const want_cfmask = (m_num_amr_levels > 1) || this->needsCoarseDataForBC();
#else
    bool const want_cfmask = (m_num_amr_levels > 1);
#endif
    if (want_cfmask) {
        buildCFMasks();
    }
    if (m_num_amr_levels > 1) {
        buildFineMask();
    }
}

void MLCurlCurl::update ()
{
    if (MLLinOpT<Array<MultiFab,3>>::needsUpdate()) {
        MLLinOpT<Array<MultiFab,3>>::update();
    }

    if (m_needs_update) {
        update_lusolver();
#if MLCC_CF_GALERKIN
        bool const want_cfmask = (m_num_amr_levels > 1) || this->needsCoarseDataForBC();
#else
        bool const want_cfmask = (m_num_amr_levels > 1);
#endif
        if (want_cfmask) {
            buildCFMasks();
        }
        if (m_num_amr_levels > 1) {
            buildFineMask();
        }
        m_needs_update = false;
    }
}

// ========================================================================
// make, makeAlias, makeCoarseMG, makeCoarseAmr
// ========================================================================

void MLCurlCurl::make (Vector<Vector<MF> >& mf, IntVect const& ng) const
{
    MLLinOpT<MF>::make(mf, ng);
}

Array<MultiFab,3>
MLCurlCurl::make (int amrlev, int mglev, IntVect const& ng) const
{
    MF r;
    for (int idim = 0; idim < 3; ++idim) {
        r[idim].define(amrex::convert(this->m_grids[amrlev][mglev], m_etype[idim]),
                       this->m_dmap[amrlev][mglev], m_ncomp, ng, MFInfo(),
                       *(this->m_factory)[amrlev][mglev]);
    }
    return r;
}

Array<MultiFab,3>
MLCurlCurl::makeAlias (MF const& mf) const
{
    MF r;
    for (int idim = 0; idim < 3; ++idim) {
        r[idim] = MultiFab(mf[idim], amrex::make_alias, 0, mf[idim].nComp());
    }
    return r;
}

Array<MultiFab,3>
MLCurlCurl::makeCoarseMG (int amrlev, int mglev, IntVect const& ng) const
{
    BoxArray cba = this->m_grids[amrlev][mglev];
    IntVect ratio = (amrlev > 0) ? IntVect(2) : this->mg_coarsen_ratio_vec[mglev];
    cba.coarsen(ratio);

    MF r;
    for (int idim = 0; idim < 3; ++idim) {
        r[idim].define(amrex::convert(cba, m_etype[idim]),
                       this->m_dmap[amrlev][mglev], m_ncomp, ng);
    }
    return r;
}

Array<MultiFab,3>
MLCurlCurl::makeCoarseAmr (int famrlev, IntVect const& ng) const
{
    BoxArray cba = this->m_grids[famrlev][0];
    IntVect ratio(this->AMRRefRatio(famrlev-1));
    cba.coarsen(ratio);

    MF r;
    for (int idim = 0; idim < 3; ++idim) {
        r[idim].define(amrex::convert(cba, m_etype[idim]),
                       this->m_dmap[famrlev][0], m_ncomp, ng);
    }
    return r;
}

// ========================================================================
// getDotMask, getFineMask, getDirichletInfo, getSymmetryInfo
// ========================================================================

iMultiFab const& MLCurlCurl::getDotMask (int amrlev, int mglev, int idim) const
{
    if (m_dotmask[amrlev][mglev][idim] == nullptr) {
        MultiFab tmp(amrex::convert(this->m_grids[amrlev][mglev], m_etype[idim]),
                     this->m_dmap[amrlev][mglev], 1, 0, MFInfo().SetAlloc(false));
        m_dotmask[amrlev][mglev][idim] =
            tmp.OwnerMask(this->m_geom[amrlev][mglev].periodicity());
    }
    return *m_dotmask[amrlev][mglev][idim];
}

iMultiFab const& MLCurlCurl::getFineMask (int amrlev, int idim) const
{
    AMREX_ASSERT(amrlev < m_num_amr_levels - 1);
    AMREX_ASSERT(m_fine_mask[amrlev][idim] != nullptr);
    return *m_fine_mask[amrlev][idim];
}

CurlCurlDirichletInfo MLCurlCurl::getDirichletInfo (int amrlev, int mglev) const
{
    auto helper = [&] (int idim, int face) -> int
    {
#if (AMREX_SPACEDIM == 2)
        if (idim == 2) { return std::numeric_limits<int>::lowest(); }
#elif (AMREX_SPACEDIM == 1)
        if (idim > 0) { return std::numeric_limits<int>::lowest(); }
#endif
        if (face == 0) {
            if (m_lobc[0][idim] == LinOpBCType::Dirichlet) {
                return m_geom[amrlev][mglev].Domain().smallEnd(idim);
            } else {
                return std::numeric_limits<int>::lowest();
            }
        } else {
            if (m_hibc[0][idim] == LinOpBCType::Dirichlet) {
                return m_geom[amrlev][mglev].Domain().bigEnd(idim) + 1;
            } else {
                return std::numeric_limits<int>::max();
            }
        }
    };

    return CurlCurlDirichletInfo{.dirichlet_lo = IntVect(AMREX_D_DECL(helper(0,0),
                                                                      helper(1,0),
                                                                      helper(2,0))),
                                 .dirichlet_hi = IntVect(AMREX_D_DECL(helper(0,1),
                                                                      helper(1,1),
                                                                      helper(2,1)))};
}

CurlCurlSymmetryInfo MLCurlCurl::getSymmetryInfo (int amrlev, int mglev) const
{
    auto helper = [&] (int idim, int face) -> int
    {
#if (AMREX_SPACEDIM == 2)
        if (idim == 2) { return std::numeric_limits<int>::lowest(); }
#elif (AMREX_SPACEDIM == 1)
        if (idim > 0) { return std::numeric_limits<int>::lowest(); }
#endif
        if (face == 0) {
            if (m_lobc[0][idim] == LinOpBCType::symmetry) {
                return m_geom[amrlev][mglev].Domain().smallEnd(idim);
            } else {
                return std::numeric_limits<int>::lowest();
            }
        } else {
            if (m_hibc[0][idim] == LinOpBCType::symmetry) {
                return m_geom[amrlev][mglev].Domain().bigEnd(idim) + 1;
            } else {
                return std::numeric_limits<int>::max();
            }
        }
    };

    return CurlCurlSymmetryInfo{.symmetry_lo = IntVect(AMREX_D_DECL(helper(0,0),
                                                                    helper(1,0),
                                                                    helper(2,0))),
                                .symmetry_hi = IntVect(AMREX_D_DECL(helper(0,1),
                                                                    helper(1,1),
                                                                    helper(2,1)))};
}

// ========================================================================
// update_lusolver (unchanged from original)
// ========================================================================

void MLCurlCurl::update_lusolver ()
{
#if (AMREX_SPACEDIM > 1)
    // There is no global LU Solver that can be built for variable alpha or beta.
    if (m_bcoefs[0][0][0] == nullptr && m_acoefs[0][0][0] == nullptr) {
        for (int amrlev = 0;  amrlev < m_num_amr_levels; ++amrlev) {
            for (int mglev = 0; mglev < m_num_mg_levels[amrlev]; ++mglev) {
                auto const& dxinv = this->m_geom[amrlev][mglev].InvCellSizeArray();
                Real dxx = dxinv[0]*dxinv[0];
                Real dyy = dxinv[1]*dxinv[1];
                Real dxy = dxinv[0]*dxinv[1];
#if (AMREX_SPACEDIM == 2)
                Array2D<Real,0,3,0,3,Order::C> A
                    {m_alpha*dyy*Real(2.0) + m_beta,
                     Real(0.0),
                    -m_alpha*dxy,
                     m_alpha*dxy,
                     Real(0.0),
                     m_alpha*dyy*Real(2.0) + m_beta,
                     m_alpha*dxy,
                    -m_alpha*dxy,
                    -m_alpha*dxy,
                     m_alpha*dxy,
                     m_alpha*dxx*Real(2.0) + m_beta,
                     Real(0.0),
                     m_alpha*dxy,
                    -m_alpha*dxy,
                     Real(0.0),
                     m_alpha*dxx*Real(2.0) + m_beta};
#else
                Real dzz = dxinv[2]*dxinv[2];
                Real dxz = dxinv[0]*dxinv[2];
                Real dyz = dxinv[1]*dxinv[2];

                Array2D<Real,0,5,0,5,Order::C> A
                    {m_alpha*(dyy+dzz)*Real(2.0) + m_beta,
                     Real(0.0),
                    -m_alpha*dxy,
                     m_alpha*dxy,
                    -m_alpha*dxz,
                     m_alpha*dxz,
                     Real(0.0),
                     m_alpha*(dyy+dzz)*Real(2.0) + m_beta,
                     m_alpha*dxy,
                    -m_alpha*dxy,
                     m_alpha*dxz,
                    -m_alpha*dxz,
                    -m_alpha*dxy,
                     m_alpha*dxy,
                     m_alpha*(dxx+dzz)*Real(2.0) + m_beta,
                     Real(0.0),
                    -m_alpha*dyz,
                     m_alpha*dyz,
                     m_alpha*dxy,
                    -m_alpha*dxy,
                     Real(0.0),
                     m_alpha*(dxx+dzz)*Real(2.0) + m_beta,
                     m_alpha*dyz,
                    -m_alpha*dyz,
                    -m_alpha*dxz,
                     m_alpha*dxz,
                    -m_alpha*dyz,
                     m_alpha*dyz,
                     m_alpha*(dxx+dyy)*Real(2.0) + m_beta,
                     Real(0.0),
                     m_alpha*dxz,
                    -m_alpha*dxz,
                     m_alpha*dyz,
                    -m_alpha*dyz,
                     Real(0.0),
                     m_alpha*(dxx+dyy)*Real(2.0) + m_beta};
#endif

                m_lusolver[amrlev][mglev]
                    = std::make_unique<Gpu::DeviceScalar
                                       <LUSolver<AMREX_SPACEDIM*2,RT>>>(A);
            }
        }
    }
#endif
}

// ========================================================================
// applyPhysBC (unchanged from original)
// ========================================================================

void MLCurlCurl::applyPhysBC (int amrlev, int mglev, MultiFab& mf,
                               CurlCurlStateType type) const
{
    if (CurlCurlStateType::b == type) { return; }

    auto const idxtype = mf.ixType();
    Box const domain = amrex::convert(this->m_geom[amrlev][mglev].Domain(), idxtype);
    Box const gdomain = amrex::convert(
        this->m_geom[amrlev][mglev].growPeriodicDomain(1), idxtype);

    MFItInfo mfi_info{};

#ifdef AMREX_USE_GPU
    Vector<Array4BoxOrientationTag<RT>> tags;
    mfi_info.DisableDeviceSync();
#endif

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(mf,mfi_info); mfi.isValid(); ++mfi) {
        auto const& vbx = mfi.validbox();
        auto const& a = mf.array(mfi);
        for (OrientationIter oit; oit; ++oit) {
            Orientation const face = oit();
            int const idim = face.coordDir();
            bool is_symmetric = face.isLow()
                ? m_lobc[0][idim] == LinOpBCType::symmetry
                : m_hibc[0][idim] == LinOpBCType::symmetry;
            if (domain[face] == vbx[face] && is_symmetric &&
                ((type == CurlCurlStateType::x) ||
                 (type == CurlCurlStateType::r && idxtype.nodeCentered(idim))))
            {
                Box b = vbx;
                for (int jdim = 0; jdim < AMREX_SPACEDIM; ++jdim) {
                    if (jdim == idim) {
                        int shift = face.isLow() ? -1 : 1;
                        b.setRange(jdim, domain[face] + shift, 1);
                    } else {
                        if (b.smallEnd(jdim) > gdomain.smallEnd(jdim)) {
                            b.growLo(jdim);
                        }
                        if (b.bigEnd(jdim) < gdomain.bigEnd(jdim)) {
                            b.growHi(jdim);
                        }
                    }
                }
#ifdef AMREX_USE_GPU
                tags.emplace_back(Array4BoxOrientationTag<RT>{.fab = a, .bx = b, .face = face});
#else
                amrex::LoopOnCpu(b, [&] (int i, int j, int k)
                {
                    mlcurlcurl_bc_symmetry(i, j, k, face, idxtype, a);
                });
#endif
            }
        }
    }

#ifdef AMREX_USE_GPU
    ParallelFor(tags,
    [=] AMREX_GPU_DEVICE (int i, int j, int k,
                          Array4BoxOrientationTag<RT> const& tag) noexcept
    {
        mlcurlcurl_bc_symmetry(i, j, k, tag.face, idxtype, tag.fab);
    });
#endif

    if (CurlCurlStateType::r == type) {
        auto sinfo = getSymmetryInfo(amrlev,mglev);

#ifdef AMREX_USE_GPU
        Vector<Array4BoxOffsetTag<RT>> tags2;
#endif

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
        for (MFIter mfi(mf,mfi_info); mfi.isValid(); ++mfi) {
            auto const& vbx = mfi.validbox();
            auto const& a = mf.array(mfi);
            for (int idim = 0; idim < AMREX_SPACEDIM-1; ++idim) {
                for (int jdim = idim+1; jdim < AMREX_SPACEDIM; ++jdim) {
                    if (idxtype.nodeCentered(idim) &&
                        idxtype.nodeCentered(jdim))
                    {
                        for (int iside = 0; iside < 2; ++iside) {
                            int ii = (iside == 0) ? vbx.smallEnd(idim) : vbx.bigEnd(idim);
                            for (int jside = 0; jside < 2; ++jside) {
                                int jj = (jside == 0) ? vbx.smallEnd(jdim) : vbx.bigEnd(jdim);
                                if (sinfo.is_symmetric(idim,iside,ii) &&
                                    sinfo.is_symmetric(jdim,jside,jj))
                                {
                                    IntVect oiv(0);
                                    oiv[idim] = (iside == 0) ? 2 : -2;
                                    oiv[jdim] = (jside == 0) ? 2 : -2;
                                    Dim3 offset = oiv.dim3();

                                    Box bb = vbx;
                                    if (iside == 0) {
                                        bb.setRange(idim,vbx.smallEnd(idim)-1);
                                    } else {
                                        bb.setRange(idim,vbx.bigEnd(idim)+1);
                                    }
                                    if (jside == 0) {
                                        bb.setRange(jdim,vbx.smallEnd(jdim)-1);
                                    } else {
                                        bb.setRange(jdim,vbx.bigEnd(jdim)+1);
                                    }
#ifdef AMREX_USE_GPU
                                    tags2.emplace_back(Array4BoxOffsetTag<RT>{.fab = a, .bx = bb, .offset = offset});
#else
                                    amrex::LoopOnCpu(bb, [&] (int i, int j, int k)
                                    {
                                        a(i,j,k) = a(i+offset.x,j+offset.y,k+offset.z);
                                    });
#endif
                                }
                            }
                        }
                    }
                }
            }
        }

#ifdef AMREX_USE_GPU
        ParallelFor(tags2,
        [=] AMREX_GPU_DEVICE (int i, int j, int k,
                              Array4BoxOffsetTag<RT> const& tag)
        {
            tag.fab(i,j,k) = tag.fab(i+tag.offset.x,j+tag.offset.y,k+tag.offset.z);
        });
#endif
    }
}

}