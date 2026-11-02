/*---------------------------------------------------------------------------*\
     ██╗████████╗██╗  ██╗ █████╗  ██████╗ █████╗       ███████╗██╗   ██╗
     ██║╚══██╔══╝██║  ██║██╔══██╗██╔════╝██╔══██╗      ██╔════╝██║   ██║
     ██║   ██║   ███████║███████║██║     ███████║█████╗█████╗  ██║   ██║
     ██║   ██║   ██╔══██║██╔══██║██║     ██╔══██║╚════╝██╔══╝  ╚██╗ ██╔╝
     ██║   ██║   ██║  ██║██║  ██║╚██████╗██║  ██║      ██║      ╚████╔╝
     ╚═╝   ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝╚═╝  ╚═╝      ╚═╝       ╚═══╝

 * In real Time Highly Advanced Computational Applications for Finite Volumes
 * Copyright (C) 2026 by the ITHACA-FV authors
-------------------------------------------------------------------------------
License
    This file is part of ITHACA-FV
    ITHACA-FV is free software: you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    ITHACA-FV is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
    GNU Lesser General Public License for more details.
    You should have received a copy of the GNU Lesser General Public License
    along with ITHACA-FV. If not, see <http://www.gnu.org/licenses/>.
Description
    Settings of the LSPG reduced problems, read from ITHACAdict. Each settings struct reads and
    validates its own entries, so that the defaults and the checks of an option live in one place.
SourceFiles
    LSPGSettings.C
\*---------------------------------------------------------------------------*/

#ifndef LSPGSettings_H
#define LSPGSettings_H

#include "fvCFD.H"

/// @brief Preconditioner of the LSPG residual. For a detailed discussion on preconditioned LSPG, see
// Lindsay, et al. “Preconditioned Least-Squares Petrov-Galerkin Reduced Order Models.” https://doi.org/10.48550/arXiv.2203.12180.

enum class ResidualPreconditionerType {
    //- Volume-integrated residuals weighted by 1/sqrt(V), block weights calibrated at t0
    None,

    //- r_i = sqrt(V_i) R_i / (s_f |a_ii|): R_i volume-integrated residual, a_ii matrix diagonal (frozen per
    //  time step), s_f field scale (RMS of the snapshots, or preconditionerScales)
    Jacobi
};

/// General settings of an LSPG reduced problem (ITHACAdict, top level)
struct ROMSettings {
    word romID = "";
    bool scaleResidualWithCellVolume = true;
    bool debug = false;
    bool writeResidualFields = false;

    //- Relative finite-difference step for the Jacobian, h_j = jacobianStep * max(|y_j|, 1)
    scalar jacobianStep = 1e-5;
    //- Order of the finite-difference Jacobian: 1 forward (n + 1 residual evaluations), 2 central (2n)
    label jacobianOrder = 2;

    ResidualPreconditionerType residualPreconditioner = ResidualPreconditionerType::None;

    static ROMSettings read(const dictionary& dict);
};

/// Affine reference q_ref(mu) of the reduced fields, q = q_ref(mu) + sum_i a_i phi_i
enum class AffineReferenceType {
    //- No reference: the modes span the fields
    None,
    //- Lifting functions scaled by the boundary values (bcMethod lift)
    Lift,
    //- Initial conditions of the offline runs, blended by inverse distance weighting on the parameters
    InitialCondition
};

/// Settings of the affine reference (ITHACAdict: bcMethod, snapshotCollectionMethod,
/// parametricDistanceExponent, maxInterpolatedSolutions)
struct AffineReferenceSettings {
    AffineReferenceType type = AffineReferenceType::InitialCondition;
    //- Exponent of the parametric distance in the inverse distance weighting
    scalar idwExponent = 1.0;
    //- Number of nearest runs used (ties included), 0: all
    label idwNeighbours = 0;

    static AffineReferenceSettings read(const dictionary& dict);
};

struct LMSolverSettings {
    label maxIterations = 10;
    //- gtol: largest cosine between the residual and a Jacobian column, |J_j^T r| / (|J_j| |r|)
    scalar gradientTol = 1e-4;
    //- ftol: relative reduction of |r|^2, actual and predicted
    scalar reductionTol = 1e-6;
    //- xtol: step size relative to |y|; 0 disables it
    scalar stepTol = 0;
    //- Jacobian reuse (jacobianReuse adaptive): the Jacobian is kept across iterations and time steps, with
    //  Broyden updates, and recomputed when a step fails or is poor, or before convergence is declared
    bool reuseJacobian = false;
    //- With reuse: an accepted step whose actual to predicted reduction is below refreshRatio recomputes the Jacobian
    scalar refreshRatio = 0.25;

    static LMSolverSettings read(const dictionary& dict);
};

/// Settings of the solver diagnostics run when the LM solve of a time step fails (ITHACAdict, lmDiagnostics)
struct LMDiagnosticsSettings {
    bool enabled = false;
    //- LM statuses that trigger a diagnostic dump ("Running" = maximum iterations reached)
    wordList triggers;
    //- Maximum number of dumps per online solve
    label maxDumps = 3;
    //- Steps of the scans, in scaled coordinates y
    scalarList steps;
    //- Scan every reduced coordinate (2 residual evaluations per coordinate and step)
    bool coordinateScan = true;

    static LMDiagnosticsSettings read(const dictionary& dict);
};

#endif
