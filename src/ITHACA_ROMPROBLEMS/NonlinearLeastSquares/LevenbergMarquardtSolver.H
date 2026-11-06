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
    LevenbergMarquardtSolver
Description
    Levenberg-Marquardt solver of the nonlinear least-squares problem.
SourceFiles
    LevenbergMarquardtSolver.C
\*---------------------------------------------------------------------------*/

#ifndef LevenbergMarquardtSolver_H
#define LevenbergMarquardtSolver_H

#include "NonlinearLSSolver.H"

namespace ITHACAnls
{

/// @brief Levenberg-Marquardt iteration on the parts of NLSInterfaces.H. Each iteration solves the damped
/// problem min |r + J d|^2 + lambda |D d|^2 (Marquardt scaling D^2 = diag(J^T J)) with the StepSolver and
/// accepts the step if the ratio rho of the actual to the predicted reduction of |r|^2 is above acceptRatio.
/// The damping follows Nielsen's update: after an accepted step lambda *= max(1/3, 1 - (2 rho - 1)^3), after a
/// rejected one lambda *= nu and nu *= 2. A rejected step keeps the Jacobian (and the factorisation of the
/// StepSolver), only the damping changes. The Jacobian is recomputed after every accepted step.
class LevenbergMarquardtSolver : public IterativeNLSSolver
{
public:
    LevenbergMarquardtSolver(
        LeastSquaresProblem& problem,
        std::unique_ptr<JacobianProvider> jacobian,
        std::unique_ptr<StepSolver> stepSolver,
        const NLSSettings& settings)
        : IterativeNLSSolver(problem, std::move(jacobian), std::move(stepSolver), settings)
    {}

    NLSResult solve(Eigen::VectorXd& y) override;
};

} // namespace ITHACAnls

#endif
