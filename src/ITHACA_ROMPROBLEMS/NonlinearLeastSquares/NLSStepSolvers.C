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
    NLSStepSolvers
Description
    Linear algebra of the steps of the nonlinear least-squares solvers.
SourceFiles
    NLSStepSolvers.C
\*---------------------------------------------------------------------------*/

#include "NLSStepSolvers.H"
#include "ParallelLinearAlgebra.H"

namespace ITHACAnls
{

namespace
{
/// Largest cosine between the residual and a Jacobian column, max_j |g_j| / (|J_j| |r|), with the squared
/// column norms colNorm2_j = (J^T J)_jj and f = |r|^2. Columns with zero norm do not count
scalar stationarity(const Eigen::VectorXd& g, const Eigen::VectorXd& colNorm2, const scalar f)
{
    scalar gnorm = 0;
    for (label j = 0; j < g.size(); j++)
    {
        const scalar colNorm = std::sqrt(colNorm2(j));
        if (colNorm > 0 && f > 0)
        {
            gnorm = std::max(gnorm, std::abs(g(j)) / (colNorm * std::sqrt(f)));
        }
    }
    return gnorm;
}

/// Marquardt scaling of the damping: the diagonal of J^T J, floored so that no coordinate is undamped
Eigen::VectorXd dampingScaling(const Eigen::VectorXd& colNorm2)
{
    if (colNorm2.size() == 0)
    {
        return colNorm2;
    }
    return colNorm2.cwiseMax(1e-12 * colNorm2.maxCoeff());
}
}


// * * * * * * * * * * * * * NormalEquationsStepSolver * * * * * * * * * * * * * //

void NormalEquationsStepSolver::prepare(const Linearization& lin)
{
    A_ = lin.J.transpose() * lin.J;
    g_ = lin.J.transpose() * lin.r;
    sumOverProcessors(A_);
    sumOverProcessors(g_);

    const Eigen::VectorXd colNorm2 = A_.diagonal();
    gradientNorm_ = stationarity(g_, colNorm2, lin.f);
    D_ = dampingScaling(colNorm2);
}


bool NormalEquationsStepSolver::solve(const scalar lambda, Eigen::VectorXd& delta)
{
    Eigen::MatrixXd M = A_;
    M.diagonal() += lambda * D_;
    delta = M.ldlt().solve(-g_);
    return delta.allFinite();
}


scalar NormalEquationsStepSolver::predictedReduction(const Eigen::VectorXd& delta) const
{
    return -(2 * g_.dot(delta) + delta.dot(A_ * delta));
}


// * * * * * * * * * * * * * * * QRStepSolver * * * * * * * * * * * * * * * * //

void QRStepSolver::prepare(const Linearization& lin)
{
    tsqr(lin.J, lin.r, R_, c_, rho_);
    g_ = R_.transpose() * c_;

    // (J^T J)_jj = |J_j|^2 = |R_j|^2
    const Eigen::VectorXd colNorm2 = R_.colwise().squaredNorm().transpose();
    gradientNorm_ = stationarity(g_, colNorm2, lin.f);
    D_ = dampingScaling(colNorm2);
}


bool QRStepSolver::solve(const scalar lambda, Eigen::VectorXd& delta)
{
    const label n = R_.cols();
    rankDeficient_ = false;
    if (n == 0)
    {
        delta.resize(0);
        return true;
    }

    if (lambda <= 0)
    {
        // Gauss-Newton: R d = -c. A numerically singular R is solved in the minimum-norm sense
        const Eigen::VectorXd diag = R_.diagonal().cwiseAbs();
        if (diag.minCoeff() <= 1e-13 * diag.maxCoeff())
        {
            rankDeficient_ = true;
            Eigen::CompleteOrthogonalDecomposition<Eigen::MatrixXd> cod(R_);
            cod.setThreshold(1e-13);
            delta = cod.solve(-c_);
        } else
        {
            delta = R_.triangularView<Eigen::Upper>().solve(-c_);
        }
    } else
    {
        // min | [R; sqrt(lambda) D] d + [c; 0] |
        Eigen::MatrixXd augmented = Eigen::MatrixXd::Zero(2 * n, n);
        augmented.topRows(n) = R_;
        augmented.bottomRows(n).diagonal() = (lambda * D_).cwiseSqrt();
        Eigen::VectorXd rhs = Eigen::VectorXd::Zero(2 * n);
        rhs.head(n) = -c_;
        delta = Eigen::HouseholderQR<Eigen::MatrixXd>(augmented).solve(rhs);
    }
    return delta.allFinite();
}


scalar QRStepSolver::predictedReduction(const Eigen::VectorXd& delta) const
{
    return -(2 * g_.dot(delta) + (R_.triangularView<Eigen::Upper>() * delta).squaredNorm());
}

} // namespace ITHACAnls
