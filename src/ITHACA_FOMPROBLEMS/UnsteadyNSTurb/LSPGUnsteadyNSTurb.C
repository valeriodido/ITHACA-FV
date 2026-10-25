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

/// \file
/// Source file of the LSPGUnsteadyNSTurb class.

#include "LSPGUnsteadyNSTurb.H"
#include "viscosityModel.H"
#include "calculatedFvPatchField.H" // Used to implement BCs
#include "fvCFD.H"
#include <functional>
#include <cmath>
#include "ITHACAPOD.H"
#include "pisoControl.H"
#include "simpleControl.H"

// * * * * * * * * * * * * * * * Constructors * * * * * * * * * * * * * * * * //
LSPGUnsteadyNSTurb::LSPGUnsteadyNSTurb(std::shared_ptr<ITHACAcontext> context): LSPGProblem(std::move(context))
{
    fvMesh& mesh = this->mesh();
    Time& runTime = this->runTime();
    pimpleControl& pimple = this->pimple();
    fv::options& fvOptions = this->fvOptions();
#include "createLSPGFields.H"
#include "createUfIfPresent.H"
    readITHACAdict();
    _k = autoPtr<volScalarField>(
        new volScalarField(
            IOobject(
                "k",
                runTime.timeName(),
                mesh,
                IOobject::MUST_READ,
                IOobject::NO_WRITE),
            mesh));
    _omega = autoPtr<volScalarField>(
        new volScalarField(
            IOobject(
                "omega",
                runTime.timeName(),
                mesh,
                IOobject::MUST_READ,
                IOobject::NO_WRITE),
            mesh));
}

void LSPGUnsteadyNSTurb::readITHACAdict()
{
    bcMethod = ITHACAdict->lookupOrDefault<word>("bcMethod", "None");
    snapshotCollectionMethod = ITHACAdict->lookupOrDefault<label>("snapshotCollectionMethod", 0);
    turbulenceModel = ITHACAdict->lookupOrDefault<word>("turbulenceModel", "kOmegaSST");

    M_Assert(bcMethod == "lift" || bcMethod == "None", "The BC method must be set to 'lift' or 'None' in ITHACAdict. Other methods are not implemented yet.");
    M_Assert(snapshotCollectionMethod == 0 || snapshotCollectionMethod == 1, "The snapshot collection method must be either 0 (no IC subtraction) or 1 (IC subtraction).");
    M_Assert(bcMethod == "None" || snapshotCollectionMethod == 0, "If the BC method is set to lift, the snapshot collection method must be set to 0 (no IC subtraction).");
    M_Assert(turbulenceModel == "kOmegaSST" || turbulenceModel == "RBF", "The turbulence model must be set to 'kOmegaSST' or 'RBF' in ITHACAdict. Other models are not implemented yet.");


    NUmodes_ = ITHACAdict->lookupOrDefault<label>("NmodesUproj", 5);
    NPmodes_ = ITHACAdict->lookupOrDefault<label>("NmodesPproj", 5);
    NNutModes_ = ITHACAdict->lookupOrDefault<label>("NmodesNutproj", 5);
    NkModes_ = ITHACAdict->lookupOrDefault<label>("NmodesKproj", 5);
    NomegaModes_ = ITHACAdict->lookupOrDefault<label>("NmodesOmegaproj", 5);

    Info << "### INFO ### " << nl
         << "BC method: " << bcMethod << nl
         << "Number of velocity modes for projection: " << NUmodes_ << nl
         << "Number of pressure modes for projection: " << NPmodes_ << nl
         << "Number of eddy viscosity modes for projection: " << NNutModes_ << nl
         << "Number of turbulent kinetic energy modes for projection: " << NkModes_ << nl
         << "Number of omega modes for projection: " << NomegaModes_ << endl;
}

// * * * * * * * * * * * * * * Full Order Methods * * * * * * * * * * * * * * //
void LSPGUnsteadyNSTurb::truthSolve(const List<scalar> mu_now, label nSample)
{
    Info << nl << "### Offline solve for parameter set: " << mu_now << nl << endl;
    fvMesh& mesh = this->mesh();
    Time& runTime = this->runTime();
    pimpleControl& pimple = this->pimple();
#include "initContinuityErrs.H"
    fv::options& fvOptions = this->fvOptions();
    IOMRFZoneList& MRF = this->MRF();
    singlePhaseTransportModel& laminarTransport = _laminarTransport();
    volScalarField& p = _p();
    volVectorField& U = _U();
    volScalarField& nut = _nut();
    volScalarField& k = _k();
    volScalarField& omega = _omega();
    surfaceScalarField& phi = _phi();

    runTime.setEndTime(finalTime);
    runTime.setTime(startTime, 0);
    runTime.setDeltaT(timeStep);
    nextWrite = startTime + writeEvery;
    label nSavedTimesteps = std::ceil((finalTime - startTime) / writeEvery);
    label nTotalTimesteps = (finalTime - startTime) / timeStep;

    M_Assert(nSavedTimesteps <= nTotalTimesteps,
        "The number of saved timesteps must be less than or equal to the total number of timesteps. Please adjust the parameters accordingly.");
    M_Assert(timeSnapshots.size() > nSample,
        "The timeSnapshots list does not have enough space for the current sample index.");

    timeSnapshots[nSample].resize(nSavedTimesteps);
    label stepCounter = 0;
    Info << "Starting time loop." << nl << endl;

    while (runTime.run())
    {
#include "readTimeControls.H"
#include "CourantNo.H"
#include "setDeltaT.H"
        runTime++;
        Info << "Time = " << runTime.timeName() << nl << endl;

        while (pimple.loop())
        {
#include "UEqn.H"
            while (pimple.correct())
            {
#include "pEqn.H"
            }

            if (pimple.turbCorr())
            {
                laminarTransport.correct();
                turbulence->correct();
            }
        }

        Info << "ExecutionTime = " << runTime.elapsedCpuTime() << " s"
             << "  ClockTime = " << runTime.elapsedClockTime() << " s"
             << nl << endl;

        if (checkWrite(runTime))
        {
            nut = turbulence->nut();
            k = turbulence->k();
            omega = turbulence->omega();
            ITHACAstream::exportSolution(U, name(counter), "./ITHACAoutput/Offline/");
            ITHACAstream::exportSolution(p, name(counter), "./ITHACAoutput/Offline/");
            ITHACAstream::exportSolution(phi, name(counter), "./ITHACAoutput/Offline/");
            ITHACAstream::exportSolution(nut, name(counter), "./ITHACAoutput/Offline/");
            ITHACAstream::exportSolution(k, name(counter), "./ITHACAoutput/Offline/");
            ITHACAstream::exportSolution(omega, name(counter), "./ITHACAoutput/Offline/");
            M_Assert(stepCounter < nSavedTimesteps,
                "The stepCounter has exceeded the allocated size for timeSnapshots.");
            timeSnapshots[nSample](stepCounter) = runTime.value();
            stepCounter++;
            Ufield.append(U.clone());
            Phifield.append(phi.clone());
            Pfield.append(p.clone());
            Nutfield.append(nut.clone());
            kfield.append(k.clone());
            omegafield.append(omega.clone());
            nextWrite += writeEvery;
            writeMu(mu_now);
            counter++;
        }
    }
}

void LSPGUnsteadyNSTurb::truthSolve(fileName folder)
{
    fvMesh& mesh = this->mesh();
    Time& runTime = this->runTime();
    pimpleControl& pimple = this->pimple();
    fv::options& fvOptions = this->fvOptions();
    IOMRFZoneList& MRF = this->MRF();
    singlePhaseTransportModel& laminarTransport = _laminarTransport();

    scalar cumulativeContErr = 0.0;
#include "initContinuityErrs.H"

    surfaceScalarField& phi = _phi();
    volScalarField& p = _p();
    volVectorField& U = _U();
    volScalarField& nut = _nut();
    volScalarField& k = _k();
    volScalarField& omega = _omega();

    instantList Times = runTime.times();
    runTime.setEndTime(finalTime);
    runTime.setTime(startTime, 0);
    runTime.setDeltaT(timeStep);
    nextWrite = startTime + writeEvery;

    while (runTime.run())
    {
#include "readTimeControls.H"
#include "CourantNo.H"
#include "setDeltaT.H"
        runTime++;
        // runTime.setEndTime(finalTime + timeStep);
        Info << "Time = " << runTime.timeName() << nl << endl;

        while (pimple.loop())
        {
#include "UEqn.H"

            while (pimple.correct())
            {
#include "pEqn.H"
            }

            if (pimple.turbCorr())
            {
                laminarTransport.correct();
                turbulence->correct();
            }
        }

        Info << "ExecutionTime = " << runTime.elapsedCpuTime() << " s"
             << "  ClockTime = " << runTime.elapsedClockTime() << " s"
             << nl << endl;

        if (checkWrite(runTime))
        {
            nut = turbulence->nut();
            k = turbulence->k();
            omega = turbulence->omega();
            ITHACAstream::exportSolution(U, name(counter), folder);
            ITHACAstream::exportSolution(p, name(counter), folder);
            ITHACAstream::exportSolution(nut, name(counter), folder);
            ITHACAstream::exportSolution(k, name(counter), folder);
            ITHACAstream::exportSolution(omega, name(counter), folder);
            std::ofstream of(folder + "/" + name(counter) + "/" + runTime.timeName());
            Ufield.append(U.clone());
            Pfield.append(p.clone());
            Nutfield.append(nut.clone());
            kfield.append(k.clone());
            omegafield.append(omega.clone());
            nextWrite += writeEvery;
            counter++;
        }
    }
}

// * * * * * * * * * * * * * * RBF Prep Methods * * * * * * * * * * * * * * //
void LSPGUnsteadyNSTurb::offlineRBFInterpolation()
{
    const Eigen::MatrixXd coeffL2nut = ITHACAutilities::getCoeffs(fluctNutfield, nutmodes,
        NNutModes_);

    const label dimInputRBF = ITHACAdict->lookupOrDefault<label>("dimInputRBF", 0);
    const bool derivativeInRBF = ITHACAdict->lookupOrDefault<bool>("derivativeInRBF", false);

    M_Assert(dimInputRBF <= NUmodes_,
        "The dimension of the input to the RBF must be less than or equal to the number of velocity modes.");
    M_Assert(dimInputRBF >= 0,
        "The dimension of the input to the RBF must be greater than or equal to zero.");

    label inputModes = dimInputRBF;
    Eigen::MatrixXd coeffL2vel;

    if (inputModes == 0)
    {
        inputModes = NUmodes_;
    }

    if (bcMethod == "lift")
    {
        coeffL2vel = ITHACAutilities::getCoeffs(Uomfield, Umodes,
            inputModes); // Returns a [modes x snapshots]
    } else
    {
        coeffL2vel = ITHACAutilities::getCoeffs(Ufield, Umodes,
            inputModes); // Returns a [modes x snapshots]
    }

    List<Eigen::MatrixXd> velDerCoeff(2);
    // Returns a list of two matrices: [0] = velocity derivative coeffs, [1] = eddy viscosity coeffs. Each matrix is [snapshots x coeffs]
    if (derivativeInRBF)
    {
        velDerCoeff = velDerivativeCoeff(coeffL2vel.transpose(), coeffL2nut.transpose(),
            timeSnapshots);
    } else
    {
        velDerCoeff[0] = coeffL2vel.transpose();
        velDerCoeff[1] = coeffL2nut.transpose();
    }

    if (Pstream::master())
    {
        ITHACAutilities::createSymLink("./ITHACAoutput/Debug");
        ITHACAstream::exportMatrix(velDerCoeff[0], "A_RBF", "numpy",
            "./ITHACAoutput/Debug/");
        ITHACAstream::exportMatrix(velDerCoeff[1], "G_RBF", "numpy",
            "./ITHACAoutput/Debug/");
    }

    if (ITHACAutilities::check_file("./ITHACAoutput/shapeParameters"))
    {
        Info << "### MESSAGE - Reading RBF shape parameters from file is not yet implemented for mathtoolbox usage. Using the one provided in the dictionary."
             << endl;
    }

    rbfSplines.resize(NNutModes_);
    Eigen::MatrixXd x = velDerCoeff[0].transpose();

    const dictionary viscDict = ITHACAdict->subDict("viscDict");
    for (label i = 0; i < NNutModes_; i++)
    {
        rbfSplines[i] = std::make_shared<ithacaInterpolator>(viscDict);
        const Eigen::VectorXd y = velDerCoeff[1].col(i);
        // rbfSplines[i]->optimizeShapeParameter(x, y, 5);
        rbfSplines[i]->fit(x, y);
        Info << "### INTERPOLATION - Fitting ithacaInterpolator for mode " << i + 1 << " completed. Here's some information:" << endl;
        rbfSplines[i]->printInfo();
    }
}

List<Eigen::MatrixXd> LSPGUnsteadyNSTurb::velDerivativeCoeff(
    const Eigen::MatrixXd& A, const Eigen::MatrixXd& G,
    const List<Eigen::VectorXd>& snapshotTimes) // LLM helped, check that this is decent code
{
    const label velCoeffsNum = A.cols();
    const label parsSamplesNum = snapshotTimes.size();
    label newRowsNum = 0;

    for (label j = 0; j < parsSamplesNum; ++j)
    {
        newRowsNum += (snapshotTimes[j].size() - 1);
    }

    List<Eigen::MatrixXd> newCoeffs;
    newCoeffs.setSize(2);
    const label newColsNum = 2 * velCoeffsNum;
    newCoeffs[0].resize(newRowsNum, newColsNum);
    newCoeffs[1].resize(newRowsNum, G.cols());
    label outOffset = 0;
    label blockStart = 0;
    // Pre-alias sub-matrices for readability and safety
    Eigen::MatrixXd& C0 = newCoeffs[0];
    Eigen::MatrixXd& C1 = newCoeffs[1];

    for (label j = 0; j < parsSamplesNum; ++j)
    {
        const Eigen::VectorXd& timeSnap = snapshotTimes[j];
        const label nSnaps = timeSnap.size();
        const label rowsPerBlock = nSnaps - 1;

        if (rowsPerBlock <= 0)
        {
            continue;
        }

        // Direct reference blocks into matrix A (zero allocation)
        auto b0 = A.middleRows(blockStart, rowsPerBlock);
        auto b2 = A.middleRows(blockStart + 1, rowsPerBlock);
        // Compute time steps: timeSnap[k+1] - timeSnap[k]
        // Uses Eigen block expression to avoid tail/head array copies
        auto deltaT = timeSnap.tail(rowsPerBlock) - timeSnap.head(rowsPerBlock);
        // Direct block assignment for newCoeffs[0] left half (b2)
        C0.block(outOffset, 0, rowsPerBlock, velCoeffsNum) = b2;
        // Direct vectorized calculation into newCoeffs[0] right half (derivative)
        // Eliminates intermediate matrices b0, b2, derivative, and bNew completely
        C0.block(outOffset, velCoeffsNum, rowsPerBlock, velCoeffsNum) =
            (b2 - b0).array().colwise() / deltaT.array();
        // Direct block assignment for newCoeffs[1]
        C1.middleRows(outOffset, rowsPerBlock) = G.middleRows(blockStart + 1,
            rowsPerBlock);
        outOffset += rowsPerBlock;
        blockStart += nSnaps;
    }

    return newCoeffs;
}

void LSPGUnsteadyNSTurb::splitEddyViscositySnapshots()
{
    const label nSamples = timeSnapshots.size();
    const label totalSnapshots = Nutfield.size();
    avgNutfield.setSize(nSamples);
    fluctNutfield.setSize(totalSnapshots);

    if (snapshotCollectionMethod == 0)
    {
        label globalIndex = 0;
        label flatIndex = 0;
        for (label i = 0; i < nSamples; i++)
        {
            label nSnap = timeSnapshots[i].size();
            M_Assert(nSnap > 0, "Each parameter sample must have at least one snapshot");
            avgNutfield.set(i, new volScalarField(IOobject("avgNut", Nutfield[globalIndex].time().timeName(), Nutfield[globalIndex].mesh(), IOobject::NO_READ, IOobject::NO_WRITE), Nutfield[globalIndex]));
            volScalarField& avg = avgNutfield[i];

            for (label j = 1; j < nSnap; j++)
            {
                avg += Nutfield[globalIndex + j];
            }

            avg /= scalar(nSnap);

            for (label j = 0; j < nSnap; j++)
            {
                fluctNutfield.set(flatIndex,
                    new volScalarField(Nutfield[globalIndex + j] - avg));
                fluctNutfield[flatIndex].rename("fluctNut");
                flatIndex++;
            }
            globalIndex += nSnap;
        }
    } else
    {
        M_Assert(avgNutfield.size() == Nut0field.size(), "avgNutfield and Nut0field must have the same size when using snapshotCollectionMethod 1.");
        M_Assert(fluctNutfield.size() == Nutfield.size(), "fluctNutfield and Nutfield must have the same size when using snapshotCollectionMethod 1.");

        for (label i = 0; i < Nut0field.size(); ++i)
        {
            avgNutfield.set(i, Nut0field[i].clone());
        }

        for (label i = 0; i < Nutfield.size(); ++i)
        {
            fluctNutfield.set(i, Nutfield[i].clone());
        }
    }
}

// * * * * * * * * * * * * * * Preparation of modes * * * * * * * * * *  //
void LSPGUnsteadyNSTurb::prepareModes()
{

}

// * * * * * * * * * * * * * * Restart Methods * * * * * * * * * * * * * //

void LSPGUnsteadyNSTurb::restart()
{
    Info << "### Restarting the offline infrastructure." << endl;
    Time& runTime = this->runTime();
    fvMesh& mesh = this->mesh();
    pimpleControl& pimple = this->pimple();
    fv::options& fvOptions = this->fvOptions();

    runTime.setTime(0, 0);
    // Read transportProperties dictionary
    IOdictionary transportProperties(
        IOobject(
            "transportProperties",
            runTime.constant(),
            mesh,
            IOobject::MUST_READ_IF_MODIFIED,
            IOobject::NO_WRITE));

    // Now we reset the fields, reading from disk (OpenFOAM folders)
    volVectorField U_new(IOobject("U", runTime.timeName(), mesh,
                             IOobject::MUST_READ, IOobject::NO_WRITE),
        mesh);
    _U() = U_new;
    volScalarField nut_new(IOobject("nut", runTime.timeName(), mesh,
                               IOobject::MUST_READ, IOobject::NO_WRITE),
        mesh);
    _nut() = nut_new;

    volScalarField UliftBC_new(IOobject("UliftBC", runTime.timeName(), mesh,
                                   IOobject::MUST_READ, IOobject::NO_WRITE),
        mesh);
    _UliftBC() = UliftBC_new;
    _laminarTransport.clear();
    _laminarTransport = autoPtr<singlePhaseTransportModel>(new singlePhaseTransportModel(_U(), _phi()));
    turbulence.clear();
    turbulence = autoPtr<incompressible::turbulenceModel>(incompressible::turbulenceModel::New(_U(), _phi(), _laminarTransport()));
    turbulence->validate();
    Info << "Restart complete." << endl;
}

void LSPGUnsteadyNSTurb::resizeModes()
{
    Umodes.resize(NUmodes_);
    nutmodes.resize(NNutModes_);
    Pmodes.resize(NPmodes_);
    Phimodes.resize(NUmodes_);
    kmodes.resize(NkModes_);
    omegamodes.resize(NomegaModes_);
}

void LSPGUnsteadyNSTurb::computePOD()
{
    int nModesU = ITHACAdict->lookupOrDefault<int>("NmodesUout", 0);
    int nModesP = ITHACAdict->lookupOrDefault<int>("NmodesPout", 0);
    int nModesK = ITHACAdict->lookupOrDefault<int>("NmodesKout", 0);
    int nModesOmega = ITHACAdict->lookupOrDefault<int>("NmodesOmegaout", 0);
    int nModesNut = ITHACAdict->lookupOrDefault<int>("NmodesNutout", 0);
    if (bcMethod == "lift")
    {
        setupLift();
        ITHACAPOD::getModes(Uomfield, Umodes, _U().name(), podex, 0, 0, nModesU, true);
        getPhiModes(Umodes, Uomfield, Phimodes, Phiomfield);
    } else
    {
        ITHACAPOD::getModes(Ufield, Umodes, _U().name(), podex, 0, 0, nModesU, true);
        getPhiModes(Umodes, Ufield, Phimodes, Phifield);
    }
    ITHACAPOD::getModes(omegafield, omegamodes, "omega", podex, 0, 0, nModesOmega, false);
    ITHACAPOD::getModes(kfield, kmodes, "k", podex, 0, 0, nModesK, true);
    ITHACAPOD::getModes(Pfield, Pmodes, _p().name(), podex, 0, 0, nModesP, true);
    ITHACAPOD::getModes(fluctNutfield, nutmodes, fluctNutfield[0].name(), podex, 0, 0, nModesNut, true);
    // Assert that the modes are not empty
    M_Assert(Umodes.size() > 0, "Velocity modes are empty.");
    M_Assert(nutmodes.size() > 0, "Nut modes are empty.");
    M_Assert(Pmodes.size() > 0, "Pressure modes are empty.");
    M_Assert(Phimodes.size() > 0, "Phi modes are empty.");
}

void LSPGUnsteadyNSTurb::setupLift()
{
    liftSolve();
    computeLift(Ufield, liftfield, Uomfield);
    homogenizePhi(Phifield, liftfieldphi, Phiomfield);

    ITHACAstream::exportFields(liftfield, "./ITHACAoutput/Lift", "ULift");
}

void LSPGUnsteadyNSTurb::homogenizePhi(
    const PtrList<surfaceScalarField>& phifields,
    const PtrList<surfaceScalarField>& phifieldsLift,
    PtrList<surfaceScalarField>& phifieldsHomogenized)
{
    scalar phi_bc = 0.0;
    scalar phi_bc_lift = 0.0;
    scalar patch_area = 0.0;

    const auto surface_areas = phifields[0].mesh().magSf().boundaryField();

    for (label i = 0; i < inletIndex.rows(); i++)
    {
        label patchID = inletIndex(i, 0);
        patch_area = gSum(surface_areas[patchID]);
        phi_bc_lift = gSum(phifieldsLift[i].boundaryField()[patchID]) / patch_area;
        Info << "### Homogenization of phi for patch " << patchID << " with area " << patch_area << " and phi_bc_lift " << phi_bc_lift << endl;

        for (label j = 0; j < phifields.size(); j++)
        {
            if (i == 0)
            {
                phi_bc = gSum(phifields[j].boundaryField()[patchID]) / patch_area;
                surfaceScalarField tempPhi("phi", phifields[j] - phifieldsLift[i] * phi_bc / phi_bc_lift);
                phifieldsHomogenized.append(tempPhi.clone());
            } else
            {
                phi_bc = gSum(phifields[j].boundaryField()[patchID]) / patch_area;
                surfaceScalarField tempPhi("phi", phifieldsHomogenized[j] - phifieldsLift[i] * phi_bc / phi_bc_lift);
                phifieldsHomogenized.set(j, tempPhi.clone());
            }
        }
    }
}

void LSPGUnsteadyNSTurb::getPhiModes(
    volVectorModes& Umodes,
    PtrList<volVectorField>& Uomfield,
    surfaceScalarModes& Phimodes,
    PtrList<surfaceScalarField>& Phiomfield)
{
    Info << "### MESSAGE - Computing phi modes from velocity modes." << endl;
    Phimodes.setSize(Umodes.size());
    // The phi modes are computed directly from the velocity modes, to avoid differences due to inner products
    Eigen::MatrixXd WU = ITHACAutilities::getCoeffs(Uomfield, Umodes).transpose();
    for (label i = 0; i < WU.cols(); i++)
    {
        WU.col(i) /= WU.col(i).squaredNorm();
    }

    for (label i = 0; i < Umodes.size(); i++)
    {
        surfaceScalarField phiMode("phi", 0 * Phiomfield[0]);
        for (label n = 0; n < Uomfield.size(); n++)
        {
            phiMode += WU(n, i) * Phiomfield[n];
        }
        Phimodes.set(i, phiMode.clone());
    }
    ITHACAstream::exportFields(Phimodes, "./ITHACAoutput/POD", "phi");
}

void LSPGUnsteadyNSTurb::liftSolve()
{
    for (label k = 0; k < inletIndex.rows(); k++)
    {
        Time& runTime = this->runTime();
        fvMesh& mesh = this->mesh();
        IOMRFZoneList& MRF = this->MRF();

        surfaceScalarField& phi = _phi();
        pisoControl potentialFlow(mesh, "potentialFlow");
        volVectorField& U = _U();
        volScalarField& UliftBC = _UliftBC();
        label BCind = inletIndex(k, 0);
        volVectorField Ulift("Ulift" + name(k), U);
        instantList Times = runTime.times();
        runTime.setTime(Times[1], 1);
        Info << "Solving a lifting Problem" << endl;
        Vector<double> v1(0, 0, 0);
        v1[inletIndex(k, 1)] = 1;
        Vector<double> v0(0, 0, 0);

        for (label j = 0; j < U.boundaryField().size(); j++)
        {
            if (j == BCind)
            {
                assignBC(Ulift, j, v1);
            } else if (U.boundaryField()[BCind].type() == "fixedValue")
            {
                assignBC(Ulift, j, v0);
            } else
            {
            }

            assignIF(Ulift, v0);
            phi = linearInterpolate(Ulift) & mesh.Sf();
        }

        Info << "Constructing velocity potential field Phi\n"
             << endl;
        volScalarField Phi(
            IOobject(
                "Phi",
                runTime.timeName(),
                mesh,
                IOobject::READ_IF_PRESENT,
                IOobject::NO_WRITE),
            mesh,
            dimensionedScalar("Phi", dimLength * dimVelocity, 0),
            UliftBC.boundaryField().types());
        label PhiRefCell = 0;
        scalar PhiRefValue = 0.0;
        setRefCell(
            Phi,
            potentialFlow.dict(),
            PhiRefCell,
            PhiRefValue);
        mesh.setFluxRequired(Phi.name());
        runTime.functionObjects().start();
        MRF.makeRelative(phi);
        adjustPhi(phi, Ulift, UliftBC);

        while (potentialFlow.correctNonOrthogonal())
        {
            fvScalarMatrix PhiEqn(
                fvm::laplacian(dimensionedScalar("1", dimless, 1), Phi) ==
                fvc::div(phi));
            PhiEqn.setReference(PhiRefCell, PhiRefValue);
            PhiEqn.solve();

            if (potentialFlow.finalNonOrthogonalIter())
            {
                phi -= PhiEqn.flux();
            }
        }

        MRF.makeAbsolute(phi);
        Info << "Continuity error = "
             << mag(fvc::div(phi))().weightedAverage(mesh.V()).value()
             << endl;
        Ulift = fvc::reconstruct(phi);
        Ulift.correctBoundaryConditions();
        Info << "Interpolated velocity error = "
             << (sqrt(sum(sqr((fvc::interpolate(U) & mesh.Sf()) - phi))) / sum(mesh.magSf())).value()
             << endl;
        Ulift.write();
        liftfield.append(Ulift.clone());
        liftfieldphi.append(phi.clone());
    }
}


void LSPGUnsteadyNSTurb::switchOffAutoWrite()
{
    Time& runTime = this->runTime();
    _U->writeOpt(IOobject::NO_WRITE);
    _p->writeOpt(IOobject::NO_WRITE);
    _nut->writeOpt(IOobject::NO_WRITE);
    _k->writeOpt(IOobject::NO_WRITE);
    _omega->writeOpt(IOobject::NO_WRITE);
}

void LSPGUnsteadyNSTurb::loadFieldsFromDisk(
    const word offlinepath)
{
    ITHACAutilities::createSymLink(offlinepath);
    ITHACAstream::read_fields(Ufield, _U(), offlinepath);
    ITHACAstream::read_fields(Phifield, _phi(), offlinepath);
    ITHACAstream::read_fields(Pfield, _p(), offlinepath);
    ITHACAstream::read_fields(Nutfield, _nut(), offlinepath);
    ITHACAstream::read_fields(kfield, _k(), offlinepath);
    ITHACAstream::read_fields(omegafield, _omega(), offlinepath);
}

void LSPGUnsteadyNSTurb::prepareSnapshots()
{
    M_Assert(timeSnapshots.size() > 0, "No time snapshots have been collected. Please run the truth solve first.");
    M_Assert(Ufield.size() > 0, "No velocity snapshots have been collected. Please run the truth solve first.");
    if (snapshotCollectionMethod == 1)
    {
        // Here we center each computed trajectory around its initial condition
        for (label i = 0; i < mu.cols(); i++) // looping over each trajectory
        {
            U0field.set(i, Ufield[i * timeSnapshots[i].size()].clone());
            Phi0field.set(i, Phifield[i * timeSnapshots[i].size()].clone());
            P0field.set(i, Pfield[i * timeSnapshots[i].size()].clone());
            Nut0field.set(i, Nutfield[i * timeSnapshots[i].size()].clone());
            k0field.set(i, kfield[i * timeSnapshots[i].size()].clone());
            omega0field.set(i, omegafield[i * timeSnapshots[i].size()].clone());
            for (label j = 0; j < timeSnapshots[i].size(); j++)
            {
                Ufield[i * timeSnapshots[i].size() + j] -= U0field[i];
                Pfield[i * timeSnapshots[i].size() + j] -= P0field[i];
                Nutfield[i * timeSnapshots[i].size() + j] -= Nut0field[i];
                Phifield[i * timeSnapshots[i].size() + j] -= Phi0field[i];
                kfield[i * timeSnapshots[i].size() + j] -= k0field[i];
                omegafield[i * timeSnapshots[i].size() + j] -= omega0field[i];
            }
        }
    }
}

void LSPGUnsteadyNSTurb::prepareParametersMu(const word& velocity_file)
{
    const Eigen::MatrixXd boundary_condition = ITHACAstream::readMatrix(velocity_file);

    M_Assert(boundary_condition.rows() > 0, "Boundary condition matrix must have at least one row.");
    M_Assert(boundary_condition.cols() > 0, "Boundary condition matrix must have at least one column.");

    Pnumber = boundary_condition.cols();
    Tnumber = boundary_condition.rows();
    if (snapshotCollectionMethod == 1)
    {
        U0field.setSize(Tnumber);
        Phi0field.setSize(Tnumber);
        P0field.setSize(Tnumber);
        Nut0field.setSize(Tnumber);
        k0field.setSize(Tnumber);
        omega0field.setSize(Tnumber);
    }
    setParameters(); // Sets mu to be (Pnumber x Tnumber)

    for (label i = 0; i < boundary_condition.cols(); i++)
    {
        double minVal = boundary_condition.col(i).minCoeff();
        double maxVal = boundary_condition.col(i).maxCoeff();
        mu_range.row(i) << minVal, maxVal;
    }
    for (label i = 0; i < boundary_condition.rows(); i++)
    {
        mu.col(i).head(boundary_condition.cols()) =
            boundary_condition.row(i).transpose();
    }
}
