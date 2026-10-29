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
    Equations of the incompressible Navier-Stokes residual, assembled as in the final PIMPLE iteration
    of pimpleFoam (no relaxation): the momentum equation without the pressure gradient, and the pressure
    equation of the PISO correction built from it. Shared by the LSPG residual and by the pressure
    residual diagnostics (PressureResidualCheck.H), so that both assemble the same equations.
SourceFiles
    IncompressibleNSEquations.C
\*---------------------------------------------------------------------------*/

#ifndef IncompressibleNSEquations_H
#define IncompressibleNSEquations_H

#include "fvCFD.H"
#include "fvOptions.H"
#include "IOMRFZoneList.H"

namespace IncompressibleNSEquations
{

//- Momentum equation without the pressure gradient: ddt(U) + div(phi, U) + MRF + divDevReff == fvOptions.
//  divDevReff is the stress term of the closure, evaluated after the MRF velocity correction
template <class Closure>
tmp<fvVectorMatrix> momentum(
    volVectorField& U,
    const surfaceScalarField& phi,
    IOMRFZoneList& MRF,
    fv::options& fvOptions,
    const Closure& closure)
{
    MRF.correctBoundaryVelocity(U);
    tmp<fvVectorMatrix> tUEqn(
        fvm::ddt(U) + fvm::div(phi, U) + MRF.DDt(U) + closure.divDevReff() == fvOptions(U));
    fvOptions.constrain(tUEqn.ref());
    return tUEqn;
}

//- Pressure equation of the PISO correction, from a momentum equation without the pressure gradient.
//  Constrains the pressure boundary conditions (fixedFluxPressure) as pimpleFoam does; the boundary
//  conditions of p are not corrected
class PressureSystem
{
public:
    PressureSystem(
        const fvVectorMatrix& UEqn,
        volVectorField& U,
        volScalarField& p,
        const surfaceScalarField& phi,
        const IOMRFZoneList& MRF,
        const label pRefCell,
        const scalar pRefValue);

    fvScalarMatrix& pEqn()
    {
        return pEqn_();
    }

    const fvScalarMatrix& pEqn() const
    {
        return pEqn_();
    }

    //- Rhie-Chow flux of the current (U, p), phiHbyA - pEqn.flux()
    tmp<surfaceScalarField> flux() const
    {
        return phiHbyA - pEqn_().flux();
    }

    //- 1/a_P of the momentum equation
    const volScalarField rAU;

    //- H/a_P, constrained on the fixed-value velocity patches
    const volVectorField HbyA;

    //- interpolate(rAU) ddtCorr(U, phi)
    const surfaceScalarField ddtCorrFlux;

    //- flux(HbyA) + ddtCorrFlux, made relative (MRF) and adjusted (closed domains)
    surfaceScalarField phiHbyA;

private:
    autoPtr<fvScalarMatrix> pEqn_;
};

}

#endif
