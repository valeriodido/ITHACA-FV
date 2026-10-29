#include "IncompressibleNSEquations.H"
#include "constrainHbyA.H"
#include "constrainPressure.H"
#include "adjustPhi.H"

IncompressibleNSEquations::PressureSystem::PressureSystem(
    const fvVectorMatrix& UEqn,
    volVectorField& U,
    volScalarField& p,
    const surfaceScalarField& phi,
    const IOMRFZoneList& MRF,
    const label pRefCell,
    const scalar pRefValue)
    : rAU(1.0 / UEqn.A()),
      HbyA(constrainHbyA(rAU * UEqn.H(), U, p)),
      ddtCorrFlux(fvc::interpolate(rAU) * fvc::ddtCorr(U, phi)),
      phiHbyA("phiHbyA", fvc::flux(HbyA) + ddtCorrFlux)
{
    MRF.makeRelative(phiHbyA);
    adjustPhi(phiHbyA, U, p);

    // Update the pressure BCs to ensure flux consistency
    constrainPressure(p, U, phiHbyA, rAU, MRF);

    pEqn_.reset(new fvScalarMatrix(fvm::laplacian(rAU, p) == fvc::div(phiHbyA)));
    pEqn_->setReference(pRefCell, pRefValue);
}
