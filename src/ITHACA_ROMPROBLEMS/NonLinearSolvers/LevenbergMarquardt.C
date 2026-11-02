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

namespace
{
/// Sum over the processors, the same on every processor
template <class Derived>
void sumOverProcessors(Eigen::MatrixBase<Derived>& m)
{
    if (!Pstream::parRun())
    {
        return;
    }
    scalarField f(m.size());
    Eigen::Map<Eigen::Matrix<scalar, Eigen::Dynamic, Eigen::Dynamic>>(f.data(), m.rows(), m.cols()) = m;
    reduce(f, sumOp<scalarField>());
    m = Eigen::Map<const Eigen::Matrix<scalar, Eigen::Dynamic, Eigen::Dynamic>>(f.cdata(), m.rows(), m.cols());
}

/// |v|^2 over the processors
double globalSquaredNorm(const Eigen::VectorXd& v)
{
    return returnReduce(v.squaredNorm(), sumOp<scalar>());
}
}


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


std::string lmStatusToString(const LMStatus status)
{
    switch (status)
    {
        case LMStatus::Running:                   return "Running";
        case LMStatus::CosinusTooSmall:           return "CosinusTooSmall (gtol)";
        case LMStatus::RelativeReductionTooSmall: return "RelativeReductionTooSmall (ftol)";
        case LMStatus::RelativeErrorTooSmall:     return "RelativeErrorTooSmall (xtol)";
        case LMStatus::XtolTooSmall:              return "XtolTooSmall";
    }
    return "Unknown";
}


LMDriver::LMDriver(
    ReducedLSPG* reducedProblem,
    int nCoordinates,
    int nResiduals,
    const LMSolverSettings& settings)
    : functor_(reducedProblem, nCoordinates, nResiduals),
      settings_(settings),
      J_(nResiduals, nCoordinates)
{}


void LMDriver::computeJacobian(const Eigen::VectorXd& y)
{
    nfev_ += functor_.df(y, J_);
    njev_++;
    rowScaling_ = functor_.reducedProblem_->residualRowScaling();
    fresh_ = true;
}


LMStepResult LMDriver::solve(Eigen::VectorXd& y)
{
    const bool reuse = settings_.reuseJacobian;
    LMStepResult result;
    nfev_ = 0;
    njev_ = 0;

    // The first evaluation of the time step also refreshes the residual weights (preconditioner)
    Eigen::VectorXd r(functor_.values());
    functor_(y, r);
    nfev_++;
    double f = globalSquaredNorm(r);
    result.initialNorm = Foam::sqrt(f);

    if (reuse && rowScaling_.size() == r.size())
    {
        // The Jacobian of the previous time step, computed at its solution, which is the initial guess of this
        // one: its rows are rescaled to the residual weights of this time step
        const Eigen::VectorXd scaling = functor_.reducedProblem_->residualRowScaling();
        J_ = scaling.cwiseQuotient(rowScaling_).asDiagonal() * J_;
        rowScaling_ = scaling;
        fresh_ = false;
    } else
    {
        computeJacobian(y);
    }

    double lambda = 1e-3;
    double nu = 2;
    while (true)
    {
        // Normal equations, summed over the processors
        Eigen::MatrixXd A = J_.transpose() * J_;
        Eigen::VectorXd g = J_.transpose() * r;
        sumOverProcessors(A);
        sumOverProcessors(g);

        // Stationarity: largest cosine between the residual and a Jacobian column
        gnorm_ = 0;
        for (label j = 0; j < g.size(); j++)
        {
            const double colNorm = Foam::sqrt(A(j, j));
            if (colNorm > 0 && f > 0)
            {
                gnorm_ = std::max(gnorm_, std::abs(g(j)) / (colNorm * Foam::sqrt(f)));
            }
        }
        if (gnorm_ <= settings_.gradientTol)
        {
            if (fresh_)
            {
                result.status = LMStatus::CosinusTooSmall;
                break;
            }
            // Only a fresh Jacobian decides stationarity
            computeJacobian(y);
            continue;
        }
        if (result.iterations >= settings_.maxIterations)
        {
            result.status = LMStatus::Running;
            break;
        }

        // Damped step, (A + lambda diag(A)) d = -g
        const Eigen::VectorXd D = A.diagonal().cwiseMax(1e-12 * A.diagonal().maxCoeff());
        Eigen::MatrixXd M = A;
        M.diagonal() += lambda * D;
        const Eigen::VectorXd delta = M.ldlt().solve(-g);
        result.iterations++;

        // Reduction of |r|^2 predicted by the linear model, and actual
        const double predicted = -(2 * g.dot(delta) + delta.dot(A * delta));
        const Eigen::VectorXd yTrial = y + delta;
        Eigen::VectorXd rTrial(r.size());
        functor_(yTrial, rTrial);
        nfev_++;
        const double fTrial = globalSquaredNorm(rTrial);
        const double rho = predicted > 0 ? (f - fTrial) / predicted : -1;

        if (rho > 1e-4)
        {
            const bool stepFromFresh = fresh_;
            const double relReduction = (f - fTrial) / f;
            const bool smallStep = settings_.stepTol > 0 && delta.norm() <= settings_.stepTol * (y.norm() + settings_.stepTol);
            if (reuse)
            {
                // Broyden update, J += (dr - J d) d^T / |d|^2, on the rows of this processor
                J_.noalias() += ((rTrial - r) - J_ * delta) * (delta.transpose() / delta.squaredNorm());
                fresh_ = false;
            }
            y = yTrial;
            r = rTrial;
            f = fTrial;
            lambda *= std::max(1.0 / 3.0, 1 - std::pow(2 * rho - 1, 3));
            nu = 2;

            if (relReduction <= settings_.reductionTol || smallStep)
            {
                if (!reuse || stepFromFresh)
                {
                    result.status = smallStep ? LMStatus::RelativeErrorTooSmall : LMStatus::RelativeReductionTooSmall;
                    break;
                }
                // The step came from a reused Jacobian: go on with a fresh one
                computeJacobian(y);
            } else if (!reuse || rho < settings_.refreshRatio)
            {
                computeJacobian(y);
            }
        } else
        {
            lambda *= nu;
            nu *= 2;
            if (reuse && !fresh_)
            {
                computeJacobian(y);
            } else if (lambda > 1e16)
            {
                result.status = LMStatus::XtolTooSmall;
                break;
            }
        }
    }
    return result;
}
