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
Description
    Residual b - A psi of an fvMatrix, also on coupled (processor, cyclic) patches.

    fvMatrix::residual() (OpenFOAM v2412) adds the contribution of the cells across the coupled patches twice:
    once in addBoundarySource, with couples = true, and once through the coupled interfaces in
    lduMatrix::residual. The solvers call addBoundarySource(source, false) instead. The duplicate is the same
    product of the boundary coefficients and the neighbour field, removed here: in serial without cyclic
    patches the result is fvMatrix::residual()
\*---------------------------------------------------------------------------*/

#ifndef MatrixResidual_H
#define MatrixResidual_H

#include "fvMatrix.H"

template <class Type>
tmp<Field<Type>> matrixResidual(const fvMatrix<Type>& eqn)
{
    tmp<Field<Type>> tres = eqn.residual();
    Field<Type>& res = tres.ref();
    const auto& bpsi = eqn.psi().boundaryField();
    forAll(bpsi, patchi)
    {
        if (bpsi[patchi].coupled())
        {
            const Field<Type> pnf(bpsi[patchi].patchNeighbourField());
            const Field<Type>& pbc = eqn.boundaryCoeffs()[patchi];
            const labelUList& cells = eqn.lduAddr().patchAddr(patchi);
            forAll(cells, facei)
            {
                res[cells[facei]] -= cmptMultiply(pbc[facei], pnf[facei]);
            }
        }
    }
    return tres;
}

#endif
