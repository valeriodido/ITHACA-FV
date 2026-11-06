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

#include "LevenbergMarquardtSolver.H"

namespace ITHACAnls
{

NLSResult LevenbergMarquardtSolver::solve(Eigen::VectorXd& y)
{
    NLSResult result;
    beginSolve();

    Eigen::VectorXd r(problem_.nResiduals());
    scalar f = evaluate(y, r);
    result.initialNorm = std::sqrt(f);
    const Eigen::MatrixXd* J = &computeJacobian(y, r);

    scalar lambda = settings_.lambda0;
    scalar nu = 2;
    // J and r change only after an accepted step: the StepSolver factorises them once, a rejected step only
    // changes the damping
    bool linearizationChanged = true;
    Eigen::VectorXd delta, rTrial(r.size());

    while (true)
    {
        if (linearizationChanged)
        {
            const Linearization lin{ *J, r, y, f, jacobian_->isFresh() };
            stepSolver_->prepare(lin);
            gradientNorm_ = stepSolver_->gradientNorm();
            linearizationChanged = false;
        }

        // The decisions use global quantities only, so that every processor takes the same path
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

        const bool solved = stepSolver_->solve(lambda, delta);
        result.iterations++;

        // Reduction of |r|^2 predicted by the linear model, and actual. A step that could not be computed is rejected
        const scalar predicted = solved ? stepSolver_->predictedReduction(delta) : -1;
        const Eigen::VectorXd yTrial = y + delta;
        scalar fTrial = 0;
        scalar rho = -1;
        if (solved)
        {
            fTrial = evaluate(yTrial, rTrial);
            rho = predicted > 0 ? (f - fTrial) / predicted : -1;
        }

        if (rho > settings_.acceptRatio)
        {
            const scalar relReduction = (f - fTrial) / f;
            const bool smallStep = settings_.stepTol > 0 && delta.norm() <= settings_.stepTol * (y.norm() + settings_.stepTol);

            jacobian_->accept(delta, r, rTrial);
            y = yTrial;
            r = rTrial;
            f = fTrial;
            lambda *= std::max(1.0 / 3.0, 1 - std::pow(2 * rho - 1, 3));
            nu = 2;

            if (relReduction <= settings_.reductionTol || smallStep)
            {
                result.status = smallStep ? NLSStatus::StepSmall : NLSStatus::ReductionSmall;
                break;
            }
            J = &computeJacobian(y, r);
            linearizationChanged = true;
        } else
        {
            lambda *= nu;
            nu *= 2;
            if (lambda > settings_.lambdaMax)
            {
                result.status = NLSStatus::DampingExploded;
                break;
            }
        }
    }
    return result;
}

} // namespace ITHACAnls
