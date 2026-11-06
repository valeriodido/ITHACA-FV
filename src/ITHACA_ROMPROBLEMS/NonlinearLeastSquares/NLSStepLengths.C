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
    NLSStepLengths
Description
    Step length rules of the Gauss-Newton solver.
SourceFiles
    NLSStepLengths.C
\*---------------------------------------------------------------------------*/

#include "NLSStepLengths.H"
#include "ParallelLinearAlgebra.H"

namespace ITHACAnls
{

StepLengthResult FixedStepLength::compute(
    LeastSquaresProblem& problem,
    const Linearization& lin,
    [[maybe_unused]] const StepSolver& stepSolver,
    const Eigen::VectorXd& delta,
    Eigen::VectorXd& rTrial,
    scalar& fTrial)
{
    problem.residual(lin.y + alpha_ * delta, rTrial);
    fTrial = globalSquaredNorm(rTrial);
    return StepLengthResult{ alpha_, true, 1 };
}


StepLengthResult BacktrackingStepLength::compute(
    LeastSquaresProblem& problem,
    const Linearization& lin,
    const StepSolver& stepSolver,
    const Eigen::VectorXd& delta,
    Eigen::VectorXd& rTrial,
    scalar& fTrial)
{
    // Slope of f = |r|^2 along the step, negative for a descent direction. Global, as f
    const scalar slope = 2 * stepSolver.gradient().dot(delta);

    scalar alpha = 1;
    label nfev = 0;
    for (label k = 0; k <= maxBacktracks_; k++)
    {
        problem.residual(lin.y + alpha * delta, rTrial);
        fTrial = globalSquaredNorm(rTrial);
        nfev++;
        // A trial that is not finite does not satisfy the condition: the comparison is false for NaN
        if (fTrial <= lin.f + c1_ * alpha * slope)
        {
            return StepLengthResult{ alpha, true, nfev };
        }
        alpha *= shrink_;
    }
    return StepLengthResult{ alpha, false, nfev };
}

} // namespace ITHACAnls
