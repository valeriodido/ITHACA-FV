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
    ParallelLinearAlgebra
Description
    Linear algebra of matrices distributed by rows over the processors, as the Jacobian of the LSPG problem.
    All the functions are no-ops/serial when the run is not parallel.
SourceFiles
    ParallelLinearAlgebra.C
\*---------------------------------------------------------------------------*/

#ifndef ParallelLinearAlgebra_H
#define ParallelLinearAlgebra_H

#include "fvCFD.H"
#include <Eigen/Dense>

namespace ITHACAnls
{

/// @brief Sum of m over the processors, the same on every processor. A matrix with the same size on every
/// processor is needed
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

/// @brief |v|^2 over the processors, for a vector distributed by rows
double globalSquaredNorm(const Eigen::VectorXd& v);

/// @brief Tall-skinny QR (TSQR) of the matrix [J | r] distributed by rows over the processors, J is m x n and
/// r has m entries. The matrix is never formed globally and Q is never computed: every processor factorises its
/// rows, the small triangular factors are stacked and factorised again. With [J | r] = Q [R c; 0 rho]:
///   - J = Q1 R, so J^T J = R^T R and the conditioning is the one of J instead of J^T J
///   - c = Q1^T r and J^T r = R^T c
///   - the minimum of |r + J d|^2 over d is rho^2, reached at R d = -c
/// R, c and rho are identical on every processor. The communication is nProcs (n + 1)^2 numbers per processor.
/// @param[in]  J local rows of the Jacobian
/// @param[in]  r local rows of the residual
/// @param[out] R n x n upper triangular
/// @param[out] c n entries
/// @param[out] rho residual of the linear least-squares problem (up to the sign)
void tsqr(
    const Eigen::MatrixXd& J,
    const Eigen::VectorXd& r,
    Eigen::MatrixXd& R,
    Eigen::VectorXd& c,
    scalar& rho);

} // namespace ITHACAnls

#endif
