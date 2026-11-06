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

#ifndef NLSStepLengths_H
#define NLSStepLengths_H

#include "fvCFD.H"
#include "NLSInterfaces.H"

namespace ITHACAnls
{

/// @brief The step is always taken with the same length (1: the pure Gauss-Newton step). The step is always
/// accepted, even if it increases the residual
class FixedStepLength : public StepLengthRule
{
public:
    explicit FixedStepLength(scalar alpha = 1) : alpha_(alpha)
    {}

    StepLengthResult compute(
        LeastSquaresProblem& problem,
        const Linearization& lin,
        const StepSolver& stepSolver,
        const Eigen::VectorXd& delta,
        Eigen::VectorXd& rTrial,
        scalar& fTrial) override;

private:
    const scalar alpha_;
};


/// @brief Backtracking line search with the Armijo (sufficient decrease) condition
///     f(y + a d) <= f(y) + c1 a f'(0),   f'(0) = 2 g.d  (f = |r|^2, g = J^T r)
/// starting from the full step a = 1, the length is multiplied by `shrink` after each failed trial. Each trial
/// costs a residual evaluation. The search fails after maxBacktracks reductions, or at once if d is not a
/// descent direction and no length satisfies the condition
class BacktrackingStepLength : public StepLengthRule
{
public:
    BacktrackingStepLength(scalar c1, scalar shrink, label maxBacktracks)
        : c1_(c1), shrink_(shrink), maxBacktracks_(maxBacktracks)
    {}

    StepLengthResult compute(
        LeastSquaresProblem& problem,
        const Linearization& lin,
        const StepSolver& stepSolver,
        const Eigen::VectorXd& delta,
        Eigen::VectorXd& rTrial,
        scalar& fTrial) override;

private:
    const scalar c1_;
    const scalar shrink_;
    const label maxBacktracks_;
};

} // namespace ITHACAnls

#endif
