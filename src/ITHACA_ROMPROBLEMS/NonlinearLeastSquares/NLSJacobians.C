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
    NLSJacobians
Description
    Jacobians of the residual of the nonlinear least-squares solvers.
SourceFiles
    NLSJacobians.C
\*---------------------------------------------------------------------------*/

#include "NLSJacobians.H"
#include "ITHACAassert.H"

namespace ITHACAnls
{

FiniteDifferenceJacobian::FiniteDifferenceJacobian(
    const scalar relativeStep,
    const label order,
    const label nResiduals,
    const label nCoordinates)
    : relativeStep_(relativeStep),
      order_(order),
      J_(nResiduals, nCoordinates)
{
    M_Assert(order == 1 || order == 2, "FiniteDifferenceJacobian: the order must be 1 (forward) or 2 (central).");
}


const Eigen::MatrixXd& FiniteDifferenceJacobian::evaluate(
    LeastSquaresProblem& problem,
    const Eigen::VectorXd& y,
    const Eigen::VectorXd& r)
{
    const bool central = order_ == 2;
    Eigen::VectorXd yPert = y;
    Eigen::VectorXd fPlus(J_.rows()), fMinus(J_.rows());

    for (label j = 0; j < y.size(); ++j)
    {
        const scalar h = relativeStep_ * std::max(std::abs(y(j)), 1.0);
        yPert(j) = y(j) + h;
        problem.residual(yPert, fPlus);
        nfev_++;
        if (central)
        {
            yPert(j) = y(j) - h;
            problem.residual(yPert, fMinus);
            nfev_++;
            J_.col(j) = (fPlus - fMinus) / (2.0 * h);
        } else
        {
            J_.col(j) = (fPlus - r) / h;
        }
        yPert(j) = y(j);
    }
    return J_;
}

} // namespace ITHACAnls
