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
    Diagnostics of the pressure (continuity) residual of the LSPG ROM.
    The pressure equation is assembled exactly as in ReducedLSPGUnsteadyNSTurb/pEqn.H, and the
    residual is split into its terms and into regions (cells next to each patch, interior).
    Used on the FOM states in memory (TestLSPG, end of each converged time step) and on the
    same states read back from disk (V1 consistency test), with identical code and norms.
    Norms are Jacobi-weighted as in the ROM preconditioner, sqrt(V) R / |a_ii|, i.e. pressure
    increments [m2/s2] times sqrt(V), not divided by the field scale.
\*---------------------------------------------------------------------------*/

#ifndef PressureResidualCheck_H
#define PressureResidualCheck_H

#include "fvCFD.H"
#include "IOMRFZoneList.H"
#include "constrainHbyA.H"
#include "constrainPressure.H"
#include "adjustPhi.H"
#include <ostream>
#include <string>

namespace PressureResidualCheck
{

//- Per-cell terms of the pressure residual, volume-integrated (units of pEqn.source())
struct Terms
{
    //- pEqn.residual(), as in the ROM
    scalarField R;
    //- fvc::laplacian(rAU, p) V
    scalarField lap;
    //- fvc::div(fvc::flux(HbyA)) V
    scalarField divHbyA;
    //- fvc::div(interpolate(rAU) ddtCorr(U, phi)) V
    scalarField divDdtCorr;
    //- |a_ii| V of the pressure equation (Jacobi scaling)
    scalarField diag;
    //- U - (HbyA - rAU grad p): zero at a converged PISO/PIMPLE state
    vectorField momentumCorrector;
};

//- Assembles the pressure equation as ReducedLSPGUnsteadyNSTurb/pEqn.H. UEqn must not contain grad(p)
inline Terms compute(
    const fvVectorMatrix& UEqn,
    volVectorField& U,
    volScalarField& p,
    const surfaceScalarField& phi,
    const IOMRFZoneList& MRF,
    const label pRefCell,
    const scalar pRefValue)
{
    const fvMesh& mesh = U.mesh();
    const scalarField& V = mesh.V().field();

    volScalarField rAU(1.0 / UEqn.A());
    volVectorField HbyA(constrainHbyA(rAU * UEqn.H(), U, p));
    const surfaceScalarField ddtCorrFlux(fvc::interpolate(rAU) * fvc::ddtCorr(U, phi));
    surfaceScalarField phiHbyA("phiHbyA", fvc::flux(HbyA) + ddtCorrFlux);
    MRF.makeRelative(phiHbyA);
    adjustPhi(phiHbyA, U, p);
    constrainPressure(p, U, phiHbyA, rAU, MRF);

    fvScalarMatrix pEqn(fvm::laplacian(rAU, p) == fvc::div(phiHbyA));
    pEqn.setReference(pRefCell, pRefValue);

    Terms t;
    t.R = pEqn.residual();
    t.lap = fvc::laplacian(rAU, p)().primitiveField() * V;
    t.divHbyA = fvc::div(fvc::flux(HbyA))().primitiveField() * V;
    t.divDdtCorr = fvc::div(ddtCorrFlux)().primitiveField() * V;
    t.diag = mag(pEqn.A()().primitiveField()) * V;
    t.momentumCorrector = U.primitiveField() - (HbyA.primitiveField() - rAU.primitiveField() * fvc::grad(p)().primitiveField());
    return t;
}

//- Cell masks: cells next to each non-empty patch, and the interior (not next to any of them)
inline void regions(const fvMesh& mesh, wordList& names, List<boolList>& masks)
{
    names.clear();
    masks.clear();
    boolList interior(mesh.nCells(), true);
    forAll(mesh.boundary(), patchi)
    {
        const fvPatch& patch = mesh.boundary()[patchi];
        if (patch.size() == 0 || isA<emptyFvPatch>(patch) || patch.coupled())
        {
            continue;
        }
        boolList mask(mesh.nCells(), false);
        for (const label celli : patch.faceCells())
        {
            mask[celli] = true;
            interior[celli] = false;
        }
        names.append(patch.name());
        masks.append(mask);
    }
    names.append("interior");
    masks.append(interior);
}

//- Jacobi-weighted norm sqrt(sum_i V_i (r_i / |a_ii|)^2) over the masked cells (all if mask empty)
inline scalar weightedNorm(const scalarField& r, const Terms& t, const scalarField& V, const boolList& mask = boolList())
{
    scalar s = 0;
    forAll(r, celli)
    {
        if (mask.empty() || mask[celli])
        {
            s += V[celli] * sqr(r[celli] / max(t.diag[celli], VSMALL));
        }
    }
    return Foam::sqrt(returnReduce(s, sumOp<scalar>()));
}

inline scalar volumeNorm(const vectorField& f, const scalarField& V)
{
    return Foam::sqrt(returnReduce(sum(V * magSqr(f)), sumOp<scalar>()));
}

inline scalar volumeNorm(const scalarField& f, const scalarField& V)
{
    return Foam::sqrt(returnReduce(sum(V * sqr(f)), sumOp<scalar>()));
}

inline std::string header(const fvMesh& mesh)
{
    wordList names;
    List<boolList> masks;
    regions(mesh, names, masks);
    std::string h =
        "# Jacobi-weighted norms sqrt(sum V (r/|a_ii|)^2), pressure increments, not divided by the field scale\n"
        "# relR = |R| / |divHbyA + divDdtCorr| ; closure = |R -+ (divHbyA + divDdtCorr - lap)| / |R|\n"
        "# relMomCorr = |U - (HbyA - rAU grad p)|_V / |U|_V\n"
        "# time R relR lap divHbyA divDdtCorr closure relMomCorr";
    for (const word& n : names)
    {
        h += " R_" + n;
    }
    return h;
}

//- One row: time, global norms, then the residual norm in each region
inline void writeRow(std::ostream& os, const scalar time, const Terms& t, const fvMesh& mesh, const vectorField& U)
{
    const scalarField& V = mesh.V().field();
    wordList names;
    List<boolList> masks;
    regions(mesh, names, masks);

    const scalarField rhs(t.divHbyA + t.divDdtCorr);
    const scalar nR = weightedNorm(t.R, t, V);
    // Sign convention of fvMatrix::residual() aside, R should be +-(divHbyA + divDdtCorr - lap)
    const scalar closure = min(weightedNorm(t.R - (rhs - t.lap), t, V), weightedNorm(t.R + (rhs - t.lap), t, V))
        / max(nR, VSMALL);

    os << time << " " << nR << " " << nR / max(weightedNorm(rhs, t, V), VSMALL) << " "
       << weightedNorm(t.lap, t, V) << " " << weightedNorm(t.divHbyA, t, V) << " "
       << weightedNorm(t.divDdtCorr, t, V) << " " << closure << " "
       << volumeNorm(t.momentumCorrector, V) / max(volumeNorm(U, V), VSMALL);
    for (const boolList& mask : masks)
    {
        os << " " << weightedNorm(t.R, t, V, mask);
    }
}

} // End namespace PressureResidualCheck

#endif
