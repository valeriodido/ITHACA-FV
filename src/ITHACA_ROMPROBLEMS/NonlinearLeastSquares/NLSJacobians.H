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

#ifndef NLSJacobians_H
#define NLSJacobians_H

#include "fvCFD.H"
#include "NLSInterfaces.H"

namespace ITHACAnls
{

/// @brief Finite-difference Jacobian, column j is the directional derivative of the residual along the
/// coordinate j (a Jacobian-vector product with the unit vector e_j). The step is
/// h_j = relativeStep * max(|y_j|, 1), the floor keeps h away from zero for coefficients close to zero.
///   - order 1: forward differences, (r(y + h e_j) - r(y)) / h, error O(h), n evaluations (r(y) is given)
///   - order 2: central differences, error O(h^2), 2n evaluations
/// The evaluations are local to the processor: every processor perturbs the same coordinates and computes
/// the columns of its own rows
class FiniteDifferenceJacobian : public JacobianProvider
{
public:
    FiniteDifferenceJacobian(scalar relativeStep, label order, label nResiduals, label nCoordinates);

    const Eigen::MatrixXd& evaluate(
        LeastSquaresProblem& problem,
        const Eigen::VectorXd& y,
        const Eigen::VectorXd& r) override;

    label residualEvaluations() const override
    {
        return nfev_;
    }

private:
    const scalar relativeStep_;
    const label order_;
    Eigen::MatrixXd J_;
    label nfev_ = 0;
};

} // namespace ITHACAnls

#endif
