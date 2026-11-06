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

#include "GaussNewtonSolver.H"

namespace ITHACAnls
{

NLSResult GaussNewtonSolver::solve(Eigen::VectorXd& y)
{
    NLSResult result;
    beginSolve();

    Eigen::VectorXd r(problem_.nResiduals());
    scalar f = evaluate(y, r);
    result.initialNorm = std::sqrt(f);
    const Eigen::MatrixXd* J = &computeJacobian(y, r);

    Eigen::VectorXd delta, rTrial(r.size());
    while (true)
    {
        const Linearization lin{ *J, r, y, f, jacobian_->isFresh() };
        stepSolver_->prepare(lin);

        // The decisions below use global quantities only, so that every processor takes the same path
        gradientNorm_ = stepSolver_->gradientNorm();
        if (gradientNorm_ <= settings_.gradientTol)
        {
            result.status = NLSStatus::GradientSmall;
            break;
        }
        if (result.iterations >= settings_.maxIterations)
        {
            result.status = NLSStatus::MaxIterations;
            break;
        }

        result.iterations++;
        if (!stepSolver_->solve(0, delta))
        {
            result.status = NLSStatus::LinearSolveFailed;
            break;
        }
        if (stepSolver_->rankDeficient())
        {
            Info << "Gauss-Newton: the Jacobian is numerically rank deficient, minimum-norm step" << endl;
        }

        scalar fTrial = 0;
        const StepLengthResult step = stepLength_->compute(problem_, lin, *stepSolver_, delta, rTrial, fTrial);
        nfevAdd(step.nfev);
        if (!step.accepted)
        {
            result.status = NLSStatus::LineSearchFailed;
            break;
        }

        const scalar relReduction = (f - fTrial) / f;
        const Eigen::VectorXd move = step.alpha * delta;
        const bool smallStep = settings_.stepTol > 0 && move.norm() <= settings_.stepTol * (y.norm() + settings_.stepTol);

        jacobian_->accept(move, r, rTrial);
        y += move;
        r = rTrial;
        f = fTrial;

        if (smallStep || (relReduction >= 0 && relReduction <= settings_.reductionTol))
        {
            result.status = smallStep ? NLSStatus::StepSmall : NLSStatus::ReductionSmall;
            break;
        }
        J = &computeJacobian(y, r);
    }
    return result;
}

} // namespace ITHACAnls
