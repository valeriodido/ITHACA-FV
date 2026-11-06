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
    NonlinearLSSolver
Description
    Nonlinear least-squares solver interface and factory. This is the only file that knows all the
    concrete parts: the rest of the code only sees NLSInterfaces.H and NonlinearLSSolver.H.
SourceFiles
    NonlinearLSSolver.C
\*---------------------------------------------------------------------------*/

#include "NonlinearLSSolver.H"
#include "ITHACAassert.H"
#include "ParallelLinearAlgebra.H"
#include "NLSJacobians.H"
#include "NLSStepSolvers.H"
#include "NLSStepLengths.H"
#include "GaussNewtonSolver.H"
#include "LevenbergMarquardtSolver.H"

namespace ITHACAnls
{

std::unique_ptr<NonlinearLSSolver> NonlinearLSSolver::New(
    LeastSquaresProblem& problem,
    const NLSSettings& settings)
{
    std::unique_ptr<JacobianProvider> jacobian;
    if (settings.jacobian == "finiteDifference")
    {
        jacobian.reset(new FiniteDifferenceJacobian(
            settings.jacobianStep, settings.jacobianOrder, problem.nResiduals(), problem.nCoordinates()));
    }

    std::unique_ptr<StepSolver> stepSolver;
    if (settings.linearAlgebra == "normalEquations")
    {
        stepSolver.reset(new NormalEquationsStepSolver());
    } else if (settings.linearAlgebra == "qr")
    {
        stepSolver.reset(new QRStepSolver());
    }

    if (!jacobian || !stepSolver)
    {
        FatalErrorInFunction
            << "jacobian '" << settings.jacobian << "' or linearAlgebra '" << settings.linearAlgebra
            << "' is not implemented." << exit(FatalError);
    }

    if (settings.type == "gaussNewton")
    {
        std::unique_ptr<StepLengthRule> rule;
        if (settings.lineSearch == "fixed")
        {
            rule.reset(new FixedStepLength(settings.stepLength));
        } else if (settings.lineSearch == "backtracking")
        {
            rule.reset(new BacktrackingStepLength(
                settings.armijoC1, settings.backtrackShrink, settings.maxBacktracks));
        } else
        {
            FatalErrorInFunction
                << "lineSearch '" << settings.lineSearch << "' is not implemented." << exit(FatalError);
        }
        return std::unique_ptr<NonlinearLSSolver>(new GaussNewtonSolver(
            problem, std::move(jacobian), std::move(stepSolver), std::move(rule), settings));
    }

    if (settings.type == "levenbergMarquardt")
    {
        return std::unique_ptr<NonlinearLSSolver>(new LevenbergMarquardtSolver(
            problem, std::move(jacobian), std::move(stepSolver), settings));
    }

    FatalErrorInFunction << "nonlinearSolver type '" << settings.type << "' is not implemented." << exit(FatalError);
    return nullptr;
}


IterativeNLSSolver::IterativeNLSSolver(
    LeastSquaresProblem& problem,
    std::unique_ptr<JacobianProvider> jacobian,
    std::unique_ptr<StepSolver> stepSolver,
    const NLSSettings& settings)
    : problem_(problem),
      jacobian_(std::move(jacobian)),
      stepSolver_(std::move(stepSolver)),
      settings_(settings)
{}


void IterativeNLSSolver::beginSolve()
{
    nfevDirect_ = 0;
    njev_ = 0;
    jacobianEvaluationsAtStart_ = jacobian_->residualEvaluations();
    gradientNorm_ = 0;
    problem_.beginSolve();
}


scalar IterativeNLSSolver::evaluate(const Eigen::VectorXd& y, Eigen::VectorXd& r)
{
    problem_.residual(y, r);
    nfevDirect_++;
    return globalSquaredNorm(r);
}


const Eigen::MatrixXd& IterativeNLSSolver::computeJacobian(const Eigen::VectorXd& y, const Eigen::VectorXd& r)
{
    njev_++;
    return jacobian_->evaluate(problem_, y, r);
}

} // namespace ITHACAnls
