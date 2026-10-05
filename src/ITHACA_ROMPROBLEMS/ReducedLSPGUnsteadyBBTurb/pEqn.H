/*---------------------------------------------------------------------------*\
     ██╗████████╗██╗  ██╗ █████╗  ██████╗ █████╗       ███████╗██╗   ██╗
     ██║╚══██╔══╝██║  ██║██╔══██╗██╔════╝██╔══██╗      ██╔════╝██║   ██║
     ██║   ██║   ███████║███████║██║     ███████║█████╗█████╗  ██║   ██║
     ██║   ██║   ██╔══██║██╔══██║██║     ██╔══██║╚════╝██╔══╝  ╚██╗ ██╔╝
     ██║   ██║   ██║  ██║██║  ██║╚██████╗██║  ██║      ██║      ╚████╔╝
     ╚═╝   ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝╚═╝  ╚═╝      ╚═╝       ╚═══╝

 * In real Time Highly Advanced Computational Applications for Finite Volumes
 * Copyright (C) 2017 by the ITHACA-FV authors
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

// Remember - In the LSPG loop we don't need to solve or compute the corrector loop
// Also, we don't solve the pressure equation, we just need to assemble it to compute the residual
volScalarField rAU("rAU", 1.0 / UEqn.A());
surfaceScalarField rAUf("rAUf", fvc::interpolate(rAU));
volVectorField HbyA(constrainHbyA(rAU * UEqn.H(), U, p_rgh));

surfaceScalarField phig(-rAUf* ghf * fvc::snGrad(rhok) * mesh.magSf());

surfaceScalarField phiHbyA
(
    "phiHbyA",
    fvc::flux(HbyA)
    + rAUf * fvc::ddtCorr(U, phi)
    + phig
);
MRF.makeRelative(phiHbyA);
// Update the pressure BCs to ensure flux consistency
constrainPressure(p_rgh, U, phiHbyA, rAUf, MRF);
fvScalarMatrix p_rghEqn
(
    fvm::laplacian(rAUf, p_rgh) == fvc::div(phiHbyA)
);
p_rghEqn.setReference(pRefCell, getRefCellValue(p_rgh, pRefCell));

// #include "continuityErrs.H"
p = p_rgh + rhok * gh;
