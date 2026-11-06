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
Class
    NLSSettings
Description
    Settings and results of the modular nonlinear least-squares solvers (ITHACAdict, nonlinearSolver).
SourceFiles
    NLSSettings.C
\*---------------------------------------------------------------------------*/

#ifndef NLSSettings_H
#define NLSSettings_H

#include "fvCFD.H"
#include <string>

namespace ITHACAnls
{

/// @brief Reason why a solve stopped
enum class NLSStatus
{
    //- Maximum number of iterations reached
    MaxIterations,
    //- gradientTol: every Jacobian column is nearly orthogonal to the residual (stationarity)
    GradientSmall,
    //- reductionTol: relative reduction of |r|^2 below the tolerance
    ReductionSmall,
    //- stepTol: step below the tolerance, relative to |y|
    StepSmall,
    //- Levenberg-Marquardt: no step reduces the residual, even with the largest damping
    DampingExploded,
    //- Gauss-Newton: no admissible step length found
    LineSearchFailed,
    //- Gauss-Newton: the linear least-squares problem of the step could not be solved
    LinearSolveFailed
};

/// @brief Status as a string, in the naming of Eigen's (MINPACK) Levenberg-Marquardt, e.g. "XtolTooSmall". The
/// triggers of lmDiagnostics (ITHACAdict) are matched against these strings
std::string toString(NLSStatus status);

/// @brief Outcome of a solve
struct NLSResult
{
    NLSStatus status = NLSStatus::MaxIterations;
    //- Iterations (linear solves of a step, rejected steps included)
    int iterations = 0;
    //- Residual norm at the initial guess
    double initialNorm = 0;
};

/// @brief Options of the nonlinear least-squares solver, ITHACAdict:
/// @code
/// nonlinearSolver
/// {
///     type            gaussNewton;        // levenbergMarquardt | gaussNewton
///     linearAlgebra   qr;                 // normalEquations | qr
///     jacobian        finiteDifference;   // finiteDifference
///     jacobianReuse   none;               // none
///     lineSearch      backtracking;       // gaussNewton: fixed | backtracking
///     damping         nielsen;            // levenbergMarquardt: nielsen
///     jacobianStep 1e-5; jacobianOrder 1;   // finite differences: relative step, 1 forward | 2 central
///     maxIterations 10; gradientTol 1e-6; reductionTol 1e-6; stepTol 0;
/// }
/// @endcode
/// Every entry is optional: the defaults below are a Levenberg-Marquardt solver on the normal equations.
struct NLSSettings
{
    word type = "levenbergMarquardt";
    word linearAlgebra = "normalEquations";
    word jacobian = "finiteDifference";
    word jacobianReuse = "none";
    word lineSearch = "fixed";
    word damping = "nielsen";

    // *** Stopping criteria *** //
    label maxIterations = 10;
    //- Largest cosine between the residual and a Jacobian column
    scalar gradientTol = 1e-4;
    //- Relative reduction of |r|^2
    scalar reductionTol = 1e-6;
    //- Step size relative to |y|; 0 disables it
    scalar stepTol = 0;

    // *** Finite-difference Jacobian *** //
    //- h_j = jacobianStep * max(|y_j|, 1)
    scalar jacobianStep = 1e-5;
    //- 1 forward, 2 central
    label jacobianOrder = 2;

    // *** Levenberg-Marquardt *** //
    //- Initial damping of every solve
    scalar lambda0 = 1e-3;
    //- The solve stops when the damping exceeds this value
    scalar lambdaMax = 1e16;
    //- A step is accepted if the ratio of the actual to the predicted reduction is above this value
    scalar acceptRatio = 1e-4;

    // *** Gauss-Newton step length *** //
    //- fixed: the step length
    scalar stepLength = 1;
    //- backtracking: sufficient decrease constant c1 of the Armijo condition, f(y + a d) <= f + 2 c1 a g.d
    scalar armijoC1 = 1e-4;
    //- backtracking: factor applied to the step length after a failed trial
    scalar backtrackShrink = 0.5;
    //- backtracking: maximum number of reductions of the step length
    label maxBacktracks = 20;

    /// @brief Reads and validates the nonlinearSolver dictionary (an empty dictionary gives the defaults)
    static NLSSettings read(const dictionary& dict);
};

} // namespace ITHACAnls

#endif
