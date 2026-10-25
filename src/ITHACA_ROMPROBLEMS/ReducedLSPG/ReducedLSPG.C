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
    ReducedLSPG
Description
    Class where the base of a Least-Squares Petrov-Galerkin reduced problem for the unsteady flow is implemented.
SourceFiles
    ReducedLSPG.C
\*---------------------------------------------------------------------------*/

#include "ReducedLSPG.H"
#include "ITHACAassert.H"

ReducedLSPG::ReducedLSPG(std::shared_ptr<ITHACAcontext> context): context_(std::move(context))
{
    if (!context_)
    {
        FatalErrorInFunction
            << "Context pointer is null. Please provide a valid ITHACAcontext object."
            << exit(FatalError);
    }
    ITHACAdict = new IOdictionary(
        IOobject(
            "ITHACAdict",
            runTime().system(),
            mesh(),
            IOobject::MUST_READ,
            IOobject::NO_WRITE
        )
    );
    romSettings_ = ROMSettings
    {
        ITHACAdict->lookupOrDefault<word>("bcMethod", "lift"),
        ITHACAdict->lookupOrDefault<word>("romID", ""),
        ITHACAdict->lookupOrDefault<label>("snapshotCollectionMethod", 1),
        ITHACAdict->lookupOrDefault<bool>("scaleResidualWithCellVolume", true),
        ITHACAdict->lookupOrDefault<bool>("debug", false)
    };

    M_Assert(romSettings_.bcMethod == "lift" || romSettings_.bcMethod == "None", "Invalid bcMethod specified in ITHACAdict. Valid options are 'lift' or 'none'.");
    M_Assert(romSettings_.bcMethod == "None" || romSettings_.snapshotCollectionMethod == 0, "If using bcMethod 'lift', snapshotCollectionMethod must be set to 0 in ITHACAdict.");
    nCells = mesh().nCells();
}

void ReducedLSPG::assembleJacobian(const Eigen::VectorXd& state,
    const Eigen::VectorXd& residual,
    Eigen::MatrixXd& jacobian,
    const Time& runTime)
{
    const double eps = 1e-6;
    const double absoluteFloor_ = 1e-8;

    for (int i = 0; i < state.size(); i++)
    {
        const double h = std::sqrt(eps) * std::max({ std::abs(state[i]), absoluteFloor_ });
        Eigen::VectorXd state_plus = state;
        state_plus[i] += h;
        const Eigen::VectorXd residual_plus = assembleResidual(state_plus);
        jacobian.col(i) = (residual_plus - residual) / h;
    }
    return;
}

template<class Type>
void ReducedLSPG::writeResidualField(
  const word& name,
  const fvMatrix<Type>& eqn,
  const Field<Type>& R,
  const fvMesh& mesh,
  const bool perUnitVolume
)
{
    typedef GeometricField<Type, fvPatchField, volMesh> fieldType;

    // fvMatrix residuals are integrated over the cell volume
    const dimensionSet dims =
        perUnitVolume ? eqn.dimensions()/dimVolume : eqn.dimensions();

    fieldType res
    (
        IOobject
        (
            name,
            mesh.time().timeName(),
            mesh,
            IOobject::NO_READ,
            IOobject::NO_WRITE
        ),
        mesh,
        dimensioned<Type>("zero", dims, Zero),
        zeroGradientFvPatchField<Type>::typeName
    );

    res.primitiveFieldRef() = R;
    if (perUnitVolume)
    {
        res.primitiveFieldRef() /= mesh.V();
    }
    res.correctBoundaryConditions();

    res.write();
}
// Template instantiation for the types we expect to use
template void ReducedLSPG::writeResidualField<scalar>(
    const word& name,
    const fvMatrix<scalar>& eqn,
    const Field<scalar>& R,
    const fvMesh& mesh,
    const bool perUnitVolume
);
template void ReducedLSPG::writeResidualField<vector>(
    const word& name,
    const fvMatrix<vector>& eqn,
    const Field<vector>& R,
    const fvMesh& mesh,
    const bool perUnitVolume
);


argList& ReducedLSPG::args() const
{
    return context_->args();
}

fvMesh& ReducedLSPG::mesh() const
{
    return context_->mesh();
}

Time& ReducedLSPG::runTime() const
{
    return context_->runTime();
}

pimpleControl& ReducedLSPG::pimple() const
{
    return context_->pimple();
}

fv::options& ReducedLSPG::fvOptions() const
{
    return context_->fvOptions();
}

IOMRFZoneList& ReducedLSPG::MRF() const
{
    return context_->MRF();
}
