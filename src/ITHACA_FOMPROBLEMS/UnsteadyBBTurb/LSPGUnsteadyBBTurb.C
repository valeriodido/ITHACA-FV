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
/// Source file of the LSPGUnsteadyBBTurb class.

#include "LSPGUnsteadyBBTurb.H"
#include "viscosityModel.H"
#include "alphatJayatillekeWallFunctionFvPatchScalarField.H" // Used to implement BCs
#include "calculatedFvPatchField.H" // Used to implement BCs
#include "fvCFD.H"
#include <functional>
#include <cmath>
#include "ITHACAPOD.H"
#include "pisoControl.H"
#include "simpleControl.H"

// * * * * * * * * * * * * * * * Constructors * * * * * * * * * * * * * * * * //
LSPGUnsteadyBBTurb::LSPGUnsteadyBBTurb(std::shared_ptr<ITHACAcontext> context): LSPGProblem(std::move(context))
{
    fvMesh& mesh = this->mesh();
    Time& runTime = this->runTime();
    pimpleControl& pimple = this->pimple();
    fv::options& fvOptions = this->fvOptions();
#include "createFields.H"
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

void LSPGUnsteadyBBTurb::readITHACAdict()
{
    bcMethod = ITHACAdict->lookupOrDefault<word>("bcMethod", "None");
    snapshotCollectionMethod = ITHACAdict->lookupOrDefault<label>("snapshotCollectionMethod", 0);

    M_Assert(bcMethod == "lift" || bcMethod == "None", "The BC method must be set to 'lift' or 'None' in ITHACAdict. Other methods are not implemented yet.");
    M_Assert(snapshotCollectionMethod == 0 || snapshotCollectionMethod == 1, "The snapshot collection method must be either 0 (no IC subtraction) or 1 (IC subtraction).");
    M_Assert(bcMethod == "None" || snapshotCollectionMethod == 0, "If the BC method is set to lift, the snapshot collection method must be set to 0 (no IC subtraction).");

    NUmodes_ = ITHACAdict->lookupOrDefault<label>("NmodesUproj", 5);
    NTmodes_ = ITHACAdict->lookupOrDefault<label>("NmodesTproj", 5);
    NPrghmodes_ = ITHACAdict->lookupOrDefault<label>("NmodesPrghproj", 5);
    NNutModes_ = ITHACAdict->lookupOrDefault<label>("NmodesNutproj", 5);

    Info << "### INFO ### " << nl
         << "BC method: " << bcMethod << nl
         << "Number of velocity modes for projection: " << NUmodes_ << nl
         << "Number of temperature modes for projection: " << NTmodes_ << nl
         << "Number of pressure modes for projection: " << NPrghmodes_ << nl
         << "Number of eddy viscosity modes for projection: " << NNutModes_ << endl;
}

// * * * * * * * * * * * * * * Full Order Methods * * * * * * * * * * * * * * //
void LSPGUnsteadyBBTurb::truthSolve(const List<scalar> mu_now, label nSample)
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
    volScalarField& p_rgh = _p_rgh();
    volScalarField& T = _T();
    volScalarField& nut = _nut();
    volScalarField& k = _k();
    volScalarField& omega = _omega();
    volScalarField& alphat = _alphat();
    volScalarField& rhok = _rhok();
    volScalarField& gh = _gh();
    surfaceScalarField& ghf = _ghf();
    surfaceScalarField& phi = _phi();
    dimensionedScalar& beta = _beta();
    dimensionedScalar& TRef = _TRef();
    dimensionedScalar& Pr = _Pr();
    dimensionedScalar& Prt = _Prt();

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
#include "TEqn.H"
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
            ITHACAstream::exportSolution(phi, name(counter), "./ITHACAoutput/Offline/");
            ITHACAstream::exportSolution(p_rgh, name(counter), "./ITHACAoutput/Offline/");
            ITHACAstream::exportSolution(T, name(counter), "./ITHACAoutput/Offline/");
            ITHACAstream::exportSolution(nut, name(counter), "./ITHACAoutput/Offline/");
            ITHACAstream::exportSolution(k, name(counter), "./ITHACAoutput/Offline/");
            ITHACAstream::exportSolution(omega, name(counter), "./ITHACAoutput/Offline/");
            M_Assert(stepCounter < nSavedTimesteps,
                "The stepCounter has exceeded the allocated size for timeSnapshots.");
            timeSnapshots[nSample](stepCounter) = runTime.value();
            stepCounter++;
            Ufield.append(U.clone());
            Phifield.append(phi.clone());
            Prghfield.append(p_rgh.clone());
            Tfield.append(T.clone());
            Nutfield.append(nut.clone());
            kfield.append(k.clone());
            omegafield.append(omega.clone());
            nextWrite += writeEvery;
            writeMu(mu_now);
            counter++;
        }
    }
}

void LSPGUnsteadyBBTurb::truthSolve(fileName folder)
{
    fvMesh& mesh = this->mesh();
    Time& runTime = this->runTime();
    pimpleControl& pimple = this->pimple();
    fv::options& fvOptions = this->fvOptions();
    IOMRFZoneList& MRF = this->MRF();
    singlePhaseTransportModel& laminarTransport = _laminarTransport();

    scalar cumulativeContErr = 0.0;
#include "initContinuityErrs.H"

    volScalarField& p = _p();
    volVectorField& U = _U();
    volScalarField& p_rgh = _p_rgh();
    volScalarField& T = _T();
    volScalarField& nut = _nut();
    volScalarField& k = _k();
    volScalarField& omega = _omega();
    volScalarField& alphat = _alphat();
    volScalarField& rhok = _rhok();
    volScalarField& gh = _gh();
    surfaceScalarField& ghf = _ghf();
    surfaceScalarField& phi = _phi();
    dimensionedScalar& beta = _beta();
    dimensionedScalar& TRef = _TRef();
    dimensionedScalar& Pr = _Pr();
    dimensionedScalar& Prt = _Prt();

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
#include "TEqn.H"

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
            ITHACAstream::exportSolution(p_rgh, name(counter), folder);
            ITHACAstream::exportSolution(T, name(counter), folder);
            ITHACAstream::exportSolution(nut, name(counter), folder);
            ITHACAstream::exportSolution(k, name(counter), folder);
            ITHACAstream::exportSolution(omega, name(counter), folder);
            std::ofstream of(folder + "/" + name(counter) + "/" + runTime.timeName());
            Ufield.append(U.clone());
            Prghfield.append(p_rgh.clone());
            Tfield.append(T.clone());
            Nutfield.append(nut.clone());
            kfield.append(k.clone());
            omegafield.append(omega.clone());
            nextWrite += writeEvery;
            counter++;
        }
    }
}

// * * * * * * * * * * * * * * RBF Prep Methods * * * * * * * * * * * * * * //
void LSPGUnsteadyBBTurb::offlineRBFInterpolation()
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

    Info << "Shape of the L2 velocity coeff matrix: " << coeffL2vel.rows() << " x "
         << coeffL2vel.cols() << nl
         << "Shape of the L2 eddy viscosity coeff matrix: " << coeffL2nut.rows() << " x " << coeffL2nut.cols() << endl;

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

List<Eigen::MatrixXd> LSPGUnsteadyBBTurb::velDerivativeCoeff(
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

void LSPGUnsteadyBBTurb::splitEddyViscositySnapshots()
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
        Info << "avgNutfield.size() = " << avgNutfield.size() << ", Nut0field.size() = " << Nut0field.size() << endl;
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
void LSPGUnsteadyBBTurb::prepareModes()
{

}

// * * * * * * * * * * * * * * Restart Methods * * * * * * * * * * * * * //

void LSPGUnsteadyBBTurb::restart()
{
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
    // Update constants
    _nu() = dimensionedScalar("nu", dimViscosity, transportProperties.lookup("nu"));
    _Pr() = dimensionedScalar("Pr", dimless, transportProperties.lookup("Pr"));
    _Prt() = dimensionedScalar("Prt", dimless, transportProperties.lookup("Prt"));
    _beta() = dimensionedScalar("beta", dimless / dimTemperature,
        transportProperties.lookup("beta"));
    _TRef() = dimensionedScalar("TRef", dimTemperature,
        transportProperties.lookup("TRef"));
    // Now we reset the fields, reading from disk (OpenFOAM folders)
    volVectorField U_new(IOobject("U", runTime.timeName(), mesh,
                             IOobject::MUST_READ, IOobject::NO_WRITE),
        mesh);
    _U() = U_new;
    // _U().correctBoundaryConditions();
    volScalarField T_new(IOobject("T", runTime.timeName(), mesh,
                             IOobject::MUST_READ, IOobject::NO_WRITE),
        mesh);
    _T() = T_new;
    // _T().correctBoundaryConditions();
    volScalarField p_rgh_new(IOobject("p_rgh", runTime.timeName(), mesh,
                                 IOobject::MUST_READ, IOobject::NO_WRITE),
        mesh);
    _p_rgh() = p_rgh_new;
    // _p_rgh().correctBoundaryConditions();
    volScalarField nut_new(IOobject("nut", runTime.timeName(), mesh,
                               IOobject::MUST_READ, IOobject::NO_WRITE),
        mesh);
    _nut() = nut_new;
    // _nut().correctBoundaryConditions();
    volScalarField alphat_new(IOobject("alphat", runTime.timeName(), mesh,
                                  IOobject::MUST_READ, IOobject::NO_WRITE),
        mesh);
    _alphat() = alphat_new;
    // _alphat().correctBoundaryConditions();
    volScalarField UliftBC_new(IOobject("UliftBC", runTime.timeName(), mesh,
                                   IOobject::MUST_READ, IOobject::NO_WRITE),
        mesh);
    _UliftBC() = UliftBC_new;
    _phi() = linearInterpolate(_U()) & mesh.Sf();
    _rhok() = 1.0 - _beta() * (_T() - _TRef());
    _laminarTransport.clear();
    _laminarTransport = autoPtr<singlePhaseTransportModel>(new singlePhaseTransportModel(_U(), _phi()));
    turbulence.clear();
    turbulence = autoPtr<incompressible::turbulenceModel>(incompressible::turbulenceModel::New(_U(), _phi(), _laminarTransport()));
    turbulence->validate();
    _p() = _p_rgh() + _rhok() * _gh();
    setRefCell(_p(), _p_rgh(), pimple.dict(), pRefCell, pRefValue);

    if (_p_rgh().needReference())
    {
        _p() += dimensionedScalar("p", _p().dimensions(),
            pRefValue - getRefCellValue(_p(), pRefCell));
    }

    mesh.setFluxRequired(_p_rgh().name());
    Info << "Restart complete." << endl;
}

void LSPGUnsteadyBBTurb::resizeModes()
{
    Umodes.resize(NUmodes_);
    Tmodes.resize(NTmodes_);
    nutmodes.resize(NNutModes_);
    Prghmodes.resize(NPrghmodes_);
    Phimodes.resize(NUmodes_);
    omegamodes.resize(NNutModes_);
    kmodes.resize(NNutModes_);
}

void LSPGUnsteadyBBTurb::computePOD()
{
    int nModesU = ITHACAdict->lookupOrDefault<int>("NmodesUout", 0);
    int nModesPrgh = ITHACAdict->lookupOrDefault<int>("NmodesPrghout", 0);
    int nModesT = ITHACAdict->lookupOrDefault<int>("NmodesTout", 0);
    int nModesNut = ITHACAdict->lookupOrDefault<int>("NmodesNutout", 0);
    if (bcMethod == "lift")
    {
        setupLift();
        ITHACAPOD::getModes(Uomfield, Umodes, _U().name(), podex, 0, 0, nModesU, true);
        ITHACAPOD::getModes(Tomfield, Tmodes, _T().name(), podex, 0, 0, nModesT, true);
        getPhiModes(Umodes, Uomfield, Phimodes, Phiomfield);
    } else
    {
        ITHACAPOD::getModes(Ufield, Umodes, _U().name(), podex, 0, 0, nModesU, true);
        ITHACAPOD::getModes(Tfield, Tmodes, _T().name(), podex, 0, 0, nModesT, true);
        getPhiModes(Umodes, Ufield, Phimodes, Phifield);
    }
    ITHACAPOD::getModes(omegafield, omegamodes, "omega", podex, 0, 0, nModesNut, false);
    ITHACAPOD::getModes(kfield, kmodes, "k", podex, 0, 0, nModesNut, false);
    ITHACAPOD::getModes(Prghfield, Prghmodes, _p_rgh().name(), podex, 0, 0, nModesPrgh, false);
    ITHACAPOD::getModes(fluctNutfield, nutmodes, fluctNutfield[0].name(), podex, 0, 0, nModesNut, true);
    // Assert that the modes are not empty
    M_Assert(Umodes.size() > 0, "Velocity modes are empty.");
    M_Assert(Tmodes.size() > 0, "Temperature modes are empty.");
    M_Assert(Phimodes.size() > 0, "Phi modes are empty.");
    M_Assert(Prghmodes.size() > 0, "Pressure modes are empty.");
}

void LSPGUnsteadyBBTurb::setupLift()
{
    liftSolve();
    liftSolveT();
    computeLift(Ufield, liftfield, Uomfield);
    computeLiftT(Tfield, liftfieldT, Tomfield);
    homogenizePhi(Phifield, liftfieldphi, Phiomfield);

    ITHACAstream::exportFields(liftfield, "./ITHACAoutput/Lift", "ULift");
    ITHACAstream::exportFields(liftfieldT, "./ITHACAoutput/Lift", "TLift");
}

void LSPGUnsteadyBBTurb::homogenizePhi(
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

void LSPGUnsteadyBBTurb::getPhiModes(
    volVectorModes& Umodes,
    PtrList<volVectorField>& Uomfield,
    surfaceScalarModes& Phimodes,
    PtrList<surfaceScalarField>& Phiomfield)
{
    Info << "### MESSAGE - Computing phi modes from velocity modes." << endl;
    Phimodes.setSize(Umodes.size());
    // The phi modes are computed directly from the velocity modes, to avoid differences due to inner products
    // Φ_i = Σₙ W^U_ni · phi_hom^n (lol LLM maths)
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

void LSPGUnsteadyBBTurb::liftSolve()
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

void LSPGUnsteadyBBTurb::liftSolveT()
{
    for (label k = 0; k < inletIndexT.rows(); k++)
    {
        Time& runTime = this->runTime();
        fvMesh& mesh = this->mesh();

        volScalarField& T = _T();
        label BCind = inletIndexT(k, 0);
        volScalarField Tlift("Tlift" + name(k), T);

        instantList Times = runTime.times();
        runTime.setTime(Times[1], 1);
        Info << "Solving Temperature Lifting with Potential Flow Proxy for Inlet " << BCind << endl;

        // Build an advective proxy flux from potential velocity lifting field (Ulift[k])
        volVectorField Uproxy = liftfield[0]; // Re-use the potential velocity lift field k
        surfaceScalarField phiProxy("phiProxy", linearInterpolate(Uproxy) & mesh.Sf());

        scalar t1 = 1.0;
        scalar t0 = 0.0;

        forAll(Tlift.boundaryField(), j)
        {
            if (j == BCind)
            {
                assignBC(Tlift, j, t1);
            } else if (Tlift.boundaryField()[j].type() == "fixedValue" || Tlift.boundaryField()[j].type() == "surfaceNormalFixedValue")
            {
                assignBC(Tlift, j, t0);
            }
        }

        assignIF(Tlift, t0);
        Tlift.correctBoundaryConditions();

        dimensionedScalar& Pr = _Pr();
        dimensionedScalar& Prt = _Prt();
        volScalarField& alphat = _alphat();
        alphat = turbulence->nut() / Prt;
        alphat.correctBoundaryConditions();
        volScalarField alphaEff("alphaEff", turbulence->nu() / Pr + alphat);
        simpleControl simple(mesh);

        Tlift.storePrevIter();

        while (simple.correctNonOrthogonal())
        {
            fvScalarMatrix TEqn(
                fvm::div(phiProxy, Tlift) - fvm::laplacian(alphaEff, Tlift));
            TEqn.relax();
            TEqn.solve();
        }

        Tlift.write();
        liftfieldT.append(Tlift.clone());
    }
}

void LSPGUnsteadyBBTurb::switchOffAutoWrite()
{
    Time& runTime = this->runTime();
    _U->writeOpt(IOobject::NO_WRITE);
    _T->writeOpt(IOobject::NO_WRITE);
    _p_rgh->writeOpt(IOobject::NO_WRITE);
    _nut->writeOpt(IOobject::NO_WRITE);
    _alphat->writeOpt(IOobject::NO_WRITE);
}

void LSPGUnsteadyBBTurb::loadFieldsFromDisk(
    const word offlinepath)
{
    ITHACAutilities::createSymLink(offlinepath);
    ITHACAstream::read_fields(Ufield, _U(), offlinepath);
    ITHACAstream::read_fields(Phifield, _phi(), offlinepath);
    ITHACAstream::read_fields(Prghfield, _p_rgh(), offlinepath);
    ITHACAstream::read_fields(Tfield, _T(), offlinepath);
    ITHACAstream::read_fields(Nutfield, _nut(), offlinepath);
    ITHACAstream::read_fields(kfield, _k(), offlinepath);
    ITHACAstream::read_fields(omegafield, _omega(), offlinepath);
}

void LSPGUnsteadyBBTurb::prepareSnapshots()
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
            Prgh0field.set(i, Prghfield[i * timeSnapshots[i].size()].clone());
            T0field.set(i, Tfield[i * timeSnapshots[i].size()].clone());
            Nut0field.set(i, Nutfield[i * timeSnapshots[i].size()].clone());
            for (label j = 0; j < timeSnapshots[i].size(); j++)
            {
                Ufield[i * timeSnapshots[i].size() + j] -= U0field[i];
                Tfield[i * timeSnapshots[i].size() + j] -= T0field[i];
                Prghfield[i * timeSnapshots[i].size() + j] -= Prgh0field[i];
                Nutfield[i * timeSnapshots[i].size() + j] -= Nut0field[i];
                Phifield[i * timeSnapshots[i].size() + j] -= Phi0field[i];
            }
        }
        Info << "### MESSAGE - Centering of snapshots around initial condition completed." << endl;
    }
}

void LSPGUnsteadyBBTurb::prepareParametersMu(const word& velocity_file, const word& temperature_file)
{
    const Eigen::MatrixXd boundary_condition = ITHACAstream::readMatrix(velocity_file);
    const Eigen::MatrixXd boundary_condition_T = ITHACAstream::readMatrix(temperature_file);

    M_Assert(boundary_condition.rows() == boundary_condition_T.rows(), "Number of parameter sets for BCs and Temperature BCs must be the same!");
    M_Assert(boundary_condition.rows() > 0, "Boundary condition matrix must have at least one row.");
    M_Assert(boundary_condition.cols() > 0, "Boundary condition matrix must have at least one column.");
    M_Assert(boundary_condition_T.cols() > 0, "Temperature boundary condition matrix must have at least one column.");
    M_Assert(boundary_condition_T.rows() > 0, "Temperature boundary condition matrix must have at least one row.");

    Pnumber = boundary_condition.cols() + boundary_condition_T.cols();
    Tnumber = boundary_condition.rows();
    if (snapshotCollectionMethod == 1)
    {
        U0field.setSize(Tnumber);
        Phi0field.setSize(Tnumber);
        Prgh0field.setSize(Tnumber);
        T0field.setSize(Tnumber);
        Nut0field.setSize(Tnumber);
    }
    setParameters(); // Sets mu to be (Pnumber x Tnumber)

    for (label i = 0; i < boundary_condition.cols(); i++)
    {
        double minVal = boundary_condition.col(i).minCoeff();
        double maxVal = boundary_condition.col(i).maxCoeff();
        mu_range.row(i) << minVal, maxVal;
    }
    for (label i = 0; i < boundary_condition_T.cols(); i++)
    {
        double minVal = boundary_condition_T.col(i).minCoeff();
        double maxVal = boundary_condition_T.col(i).maxCoeff();
        mu_range.row(boundary_condition.cols() + i) << minVal, maxVal;
    }
    for (label i = 0; i < boundary_condition.rows(); i++)
    {
        mu.col(i).head(boundary_condition.cols()) =
            boundary_condition.row(i).transpose();

        mu.col(i).tail(boundary_condition_T.cols()) =
            boundary_condition_T.row(i).transpose();
    }
}
