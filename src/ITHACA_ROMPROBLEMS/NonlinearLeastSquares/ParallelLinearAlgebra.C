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
    Linear algebra of matrices distributed by rows over the processors.
SourceFiles
    ParallelLinearAlgebra.C
\*---------------------------------------------------------------------------*/

#include "ParallelLinearAlgebra.H"

namespace ITHACAnls
{

double globalSquaredNorm(const Eigen::VectorXd& v)
{
    return returnReduce(v.squaredNorm(), sumOp<scalar>());
}


void tsqr(
    const Eigen::MatrixXd& J,
    const Eigen::VectorXd& r,
    Eigen::MatrixXd& R,
    Eigen::VectorXd& c,
    scalar& rho)
{
    const label n = J.cols();
    const label m = J.rows();
    const label q = n + 1;

    Eigen::MatrixXd Jr(m, q);
    Jr << J, r;

    // Local factor. A processor with fewer than n + 1 rows has a trapezoidal factor with k rows, completed by
    // zero rows: they do not change the stacked R^T R
    const Eigen::HouseholderQR<Eigen::MatrixXd> localQR(Jr);
    const label k = std::min<label>(m, q);
    Eigen::MatrixXd Rloc = Eigen::MatrixXd::Zero(q, q);
    Rloc.topRows(k) = localQR.matrixQR().topRows(k).triangularView<Eigen::Upper>();

    Eigen::MatrixXd Rglob = Rloc;
    if (Pstream::parRun())
    {
        // Every processor puts its factor in its own block of the stack and the blocks are summed, so that every
        // processor holds the whole stack and computes the same global factor
        Eigen::MatrixXd stacked = Eigen::MatrixXd::Zero(Pstream::nProcs() * q, q);
        stacked.block(Pstream::myProcNo() * q, 0, q, q) = Rloc;
        sumOverProcessors(stacked);
        const Eigen::HouseholderQR<Eigen::MatrixXd> globalQR(stacked);
        Rglob = globalQR.matrixQR().topRows(q).triangularView<Eigen::Upper>();
    }

    R = Rglob.topLeftCorner(n, n);
    c = Rglob.topRightCorner(n, 1);
    rho = Rglob(n, n);
}

} // namespace ITHACAnls
