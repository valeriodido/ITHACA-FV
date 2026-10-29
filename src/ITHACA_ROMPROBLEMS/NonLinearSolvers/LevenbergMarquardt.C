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
    LevenbergMarquardt
Description
    Class holding the functor to enable usage of Eigen's Levenberg-Marquardt 
    algorithm for solving non-linear least squares problems.
    Used in the solution of Least Squares Petrov-Galerkin (LSPG) reduced order models.
    TODO: Check MPI parallelization
SourceFiles
    LevenbergMarquardt.H
\*---------------------------------------------------------------------------*/

#include "LevenbergMarquardt.H"

std::string lmStatusToString(Eigen::LevenbergMarquardtSpace::Status status)
{
    using namespace Eigen::LevenbergMarquardtSpace;
    switch (status)
    {
        case NotStarted:                    return "NotStarted";
        case Running:                       return "Running";
        case ImproperInputParameters:       return "ImproperInputParameters";
        case RelativeReductionTooSmall:     return "RelativeReductionTooSmall (ftol)";
        case RelativeErrorTooSmall:         return "RelativeErrorTooSmall (xtol)";
        case RelativeErrorAndReductionTooSmall: return "RelativeErrorAndReductionTooSmall (ftol+xtol)";
        case CosinusTooSmall:               return "CosinusTooSmall (gtol)";
        case TooManyFunctionEvaluation:     return "TooManyFunctionEvaluation (maxfev)";
        case FtolTooSmall:                  return "FtolTooSmall";
        case XtolTooSmall:                  return "XtolTooSmall";
        case GtolTooSmall:                  return "GtolTooSmall";
        case UserAsked:                     return "UserAsked";
        default:                            return "Unknown";
    }
}

LMDriver::LMDriver(
    ReducedLSPG* reducedProblem,
    int nCoordinates,
    int nResiduals,
    const LMSolverSettings& settings)
    : functor_(reducedProblem, nCoordinates, nResiduals),
      lm_(functor_),
      maxIterations_(settings.maxIterations)
{
    lm_.parameters.ftol = settings.reductionTol;
    lm_.parameters.xtol = settings.stepTol;
    lm_.parameters.gtol = settings.gradientTol;
    lm_.parameters.maxfev = nCoordinates * 500;
}

LMStepResult LMDriver::solve(Eigen::VectorXd& y)
{
    using namespace Eigen::LevenbergMarquardtSpace;
    LMStepResult result;
    result.status = lm_.minimizeInit(y);
    // Over all the processors, unlike fvec, which only holds the rows of this processor
    result.initialNorm = functor_.reducedProblem_->residualNorm();
    while ((result.status == NotStarted || result.status == Running) && result.iterations < maxIterations_)
    {
        result.status = lm_.minimizeOneStep(y);
        result.iterations++;
    }
    return result;
}
