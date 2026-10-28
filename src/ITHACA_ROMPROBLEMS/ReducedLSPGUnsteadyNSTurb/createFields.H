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

 \*---------------------------------------------------------------------------*/

IOdictionary transportProperties(
    IOobject(
        "transportProperties",
        runTime.constant(),
        mesh,
        IOobject::MUST_READ_IF_MODIFIED,
        IOobject::NO_WRITE));

// Laminar viscocity [m2/s]
_nu = autoPtr<dimensionedScalar>(
    new dimensionedScalar(
        "nu",
        dimViscosity,
        transportProperties.lookup("nu")));
dimensionedScalar& nu = _nu();

// Turbulent viscocity [m2/s]
_nut = autoPtr<volScalarField>(
    new volScalarField(
        IOobject(
            "nut_r",
            runTime.timeName(),
            mesh,
            IOobject::MUST_READ,
            IOobject::AUTO_WRITE),
        mesh));
volScalarField& nut = _nut();

_U = autoPtr<volVectorField>(
    new volVectorField(
        IOobject(
            "U_r",
            runTime.timeName(),
            mesh,
            IOobject::MUST_READ,
            IOobject::AUTO_WRITE),
        mesh));
volVectorField& U = _U();

_phi = autoPtr<surfaceScalarField>(
    new surfaceScalarField(
        IOobject(
            "phi_r",
            runTime.timeName(),
            mesh,
            IOobject::NO_READ,
            IOobject::AUTO_WRITE),
        linearInterpolate(U) & mesh.Sf()));
surfaceScalarField& phi = _phi();


_p = autoPtr<volScalarField>(
    new volScalarField(
        IOobject(
            "p_r",
            runTime.timeName(),
            mesh,
            IOobject::MUST_READ,
            IOobject::AUTO_WRITE),
        mesh));
volScalarField& p = _p();

_laminarTransport = autoPtr<singlePhaseTransportModel>(
    new singlePhaseTransportModel(U, phi));
singlePhaseTransportModel& laminarTransport = _laminarTransport();

_pRefCell = 0;
_pRefValue = 0.0;
setRefCell(p, pimple.dict(), _pRefCell, _pRefValue);

_k = autoPtr<volScalarField>(
    new volScalarField(
        IOobject(
            "k_r",
            runTime.timeName(),
            mesh,
            IOobject::MUST_READ,
            IOobject::AUTO_WRITE),
        mesh));

_omega = autoPtr<volScalarField>(
    new volScalarField(
        IOobject(
            "omega_r",
            runTime.timeName(),
            mesh,
            IOobject::MUST_READ,
            IOobject::AUTO_WRITE),
        mesh));