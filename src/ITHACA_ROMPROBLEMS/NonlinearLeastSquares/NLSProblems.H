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
    NLSProblems
Description
    Least-squares problems of the nonlinear least-squares solvers.
SourceFiles
    NLSProblems.H
\*---------------------------------------------------------------------------*/

#ifndef NLSProblems_H
#define NLSProblems_H

#include "fvCFD.H"
#include "NLSInterfaces.H"
#include "ReducedLSPG.H"

namespace ITHACAnls
{

/// @brief The LSPG problem of a time step: the weighted residual of a ReducedLSPG model as a function of the scaled
/// reduced coordinates y, r(y) = assembleResidual(fullState(y)). The residual is the one of this processor's cells
class ReducedLSPGProblem : public LeastSquaresProblem
{
public:
    ReducedLSPGProblem(::ReducedLSPG& rom, label nCoordinates, label nResiduals)
        : rom_(rom), nCoordinates_(nCoordinates), nResiduals_(nResiduals)
    {}

    label nCoordinates() const override
    {
        return nCoordinates_;
    }

    label nResiduals() const override
    {
        return nResiduals_;
    }

    void residual(const Eigen::VectorXd& y, Eigen::VectorXd& r) const override
    {
        r = rom_.assembleResidual(rom_.fullState(y));
    }

    //- The first residual evaluation of the solve rebuilds the preconditioner of the residual
    void beginSolve() override
    {
        rom_.requestResidualPreconditionerUpdate();
    }

private:
    //- The model is modified by the residual evaluations (fields, boundary conditions)
    ::ReducedLSPG& rom_;
    const label nCoordinates_;
    const label nResiduals_;
};

} // namespace ITHACAnls

#endif
