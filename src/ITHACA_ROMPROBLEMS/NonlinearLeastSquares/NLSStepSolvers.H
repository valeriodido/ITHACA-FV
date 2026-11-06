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
    Linear algebra of the steps of the nonlinear least-squares solvers: the (damped) linear least-squares problem
        min_d |r + J d|^2 + lambda |D d|^2,   D^2 = diag(J^T J) (Marquardt scaling, floored)
    solved with the normal equations or with a QR factorisation of the Jacobian. Both are parallel: the rows of J
    and r are distributed over the processors.
SourceFiles
    NLSStepSolvers.C
\*---------------------------------------------------------------------------*/

#ifndef NLSStepSolvers_H
#define NLSStepSolvers_H

#include "fvCFD.H"
#include "NLSInterfaces.H"

namespace ITHACAnls
{

/// @brief Normal equations (J^T J + lambda D^2) d = -J^T r, with J^T J and J^T r summed over the processors and
/// factorised by LDLT. Cheap, but the conditioning is the square of the one of J
class NormalEquationsStepSolver : public StepSolver
{
public:
    void prepare(const Linearization& lin) override;
    const Eigen::VectorXd& gradient() const override { return g_; }
    scalar gradientNorm() const override { return gradientNorm_; }
    bool solve(scalar lambda, Eigen::VectorXd& delta) override;
    scalar predictedReduction(const Eigen::VectorXd& delta) const override;

private:
    //- J^T J and J^T r over the processors
    Eigen::MatrixXd A_;
    Eigen::VectorXd g_;
    //- Marquardt scaling of the damping
    Eigen::VectorXd D_;
    scalar gradientNorm_ = 0;
};


/// @brief QR factorisation of the Jacobian: TSQR of [J | r] (tsqr in ParallelLinearAlgebra.H) gives the small
/// triangular R, c = Q^T r and the gradient J^T r = R^T c, with the conditioning of J.
///   - Gauss-Newton (lambda = 0): back substitution of R d = -c. If R is numerically singular, the minimum-norm
///     solution of a complete orthogonal decomposition of R is used (rankDeficient)
///   - Damped (lambda > 0): QR of the stacked [R; sqrt(lambda) D], which is small (2n x n) and does not touch J again,
///     so that the steps with different damping are cheap and need no communication
/// R and c are identical on every processor, and so are the steps
class QRStepSolver : public StepSolver
{
public:
    void prepare(const Linearization& lin) override;
    const Eigen::VectorXd& gradient() const override { return g_; }
    scalar gradientNorm() const override { return gradientNorm_; }
    bool solve(scalar lambda, Eigen::VectorXd& delta) override;
    scalar predictedReduction(const Eigen::VectorXd& delta) const override;
    bool rankDeficient() const override { return rankDeficient_; }

private:
    //- Triangular factor of J, c = Q^T r, and the residual of the linear least-squares problem
    Eigen::MatrixXd R_;
    Eigen::VectorXd c_;
    scalar rho_ = 0;
    Eigen::VectorXd g_;
    //- Marquardt scaling of the damping
    Eigen::VectorXd D_;
    scalar gradientNorm_ = 0;
    bool rankDeficient_ = false;
};

} // namespace ITHACAnls

#endif
