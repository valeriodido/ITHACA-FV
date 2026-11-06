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
    GaussNewtonSolver
Description
    Gauss-Newton solver of the nonlinear least-squares problem.
SourceFiles
    GaussNewtonSolver.C
\*---------------------------------------------------------------------------*/

#ifndef GaussNewtonSolver_H
#define GaussNewtonSolver_H

#include "NonlinearLSSolver.H"

namespace ITHACAnls
{

/// @brief Gauss-Newton iteration with a step length rule. Each iteration computes the Jacobian, the step
/// d that minimises |r + J d|^2 (StepSolver, lambda = 0) and moves to y + alpha d, with alpha chosen by the
/// StepLengthRule (fixed, Armijo backtracking). With the QR step solver the least-squares problem is solved
/// without forming J^T J.
///
/// Stopping, with the criteria of NLSSettings:
///   - gradientTol:  largest cosine between the residual and a Jacobian column
///   - reductionTol: relative reduction of |r|^2 in 0 <= (f - fTrial) / f <= reductionTol. A fixed step
///                   that increases the residual is taken (it is the pure Gauss-Newton iteration) but is not
///                   convergence
///   - stepTol:      alpha |d| relative to |y|
///   - LineSearchFailed, LinearSolveFailed
class GaussNewtonSolver : public IterativeNLSSolver
{
public:
    GaussNewtonSolver(
        LeastSquaresProblem& problem,
        std::unique_ptr<JacobianProvider> jacobian,
        std::unique_ptr<StepSolver> stepSolver,
        std::unique_ptr<StepLengthRule> stepLength,
        const NLSSettings& settings)
        : IterativeNLSSolver(problem, std::move(jacobian), std::move(stepSolver), settings),
          stepLength_(std::move(stepLength))
    {}

    NLSResult solve(Eigen::VectorXd& y) override;

private:
    std::unique_ptr<StepLengthRule> stepLength_;
};

} // namespace ITHACAnls

#endif
