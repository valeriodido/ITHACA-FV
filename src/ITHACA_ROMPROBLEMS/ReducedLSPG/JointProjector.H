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
    JointProjector
Description
    Projection onto the stacked POD. The stacked modes are not orthonormal field by field: the shared
    coefficients a of the fields s_f solve G a = sum_f w_f F_f^T V_f s_f, with G = sum_f w_f F_f^T V_f F_f,
    in the weighted inner product sum_f w_f <., .>_V the modes were computed with. F_f is the block of the
    stacked modes that belongs to field f (one mode per column), V_f the cell volumes.
    The products F_f^T V_f F_f and F_f^T V_f s_f are computed with ITHACA (getMassMatrix, Modes::project),
    as in the stacked POD of the FOM; only the factorisation of G is stored.
SourceFiles
    JointProjector.C
\*---------------------------------------------------------------------------*/

#ifndef JointProjector_H
#define JointProjector_H

#include "ReducedField.H"

class JointProjector
{
public:
    /// @param fields Field set; the fields with joint() are projected together
    /// @param weights Weight w_f of each field of the set in the inner product (0 outside the stacked POD)
    JointProjector(ReducedFieldSet& fields, const Eigen::VectorXd& weights);

    //- Number of joint fields
    label size() const
    {
        return fields_.size();
    }

    //- Shared coefficients from the Galerkin projections P_f = F_f^T V_f s_f of the joint fields, in set
    //  order, already summed over the processors (one column per snapshot): a = G^-1 sum_f w_f P_f
    Eigen::MatrixXd solve(const List<Eigen::MatrixXd>& projections) const;

    //- Shared coefficients of the current fields of the set
    Eigen::VectorXd coefficients(ReducedFieldSet& fields) const;

private:
    //- Indices of the joint fields in the set
    labelList fields_;

    //- w_f of each joint field
    scalarList weights_;

    Eigen::LDLT<Eigen::MatrixXd> gram_;
};

#endif
