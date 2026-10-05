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
/// Source file of the UnsteadyBBTurb class.

#include "UnsteadyBBTurb.H"
#include "viscosityModel.H"
#include "alphatJayatillekeWallFunctionFvPatchScalarField.H" // Used to implement BCs
#include "calculatedFvPatchField.H" // Used to implement BCs
#include "fvCFD.H"
#include <functional>
#include <cmath>

// * * * * * * * * * * * * * * * Constructors * * * * * * * * * * * * * * * * //

// Constructor
UnsteadyBBTurb::UnsteadyBBTurb() { }

UnsteadyBBTurb::UnsteadyBBTurb(int argc, char* argv[])
{
    _args = autoPtr<argList>(
                new argList(argc, argv));

    if (!_args->checkRootCase())
    {
        Foam::FatalError.exit();
    }

    argList& args = _args();
#include "createTime.H"
#include "createMesh.H"
    _pimple = autoPtr<pimpleControl>(
                  new pimpleControl(
                      mesh));
    pimpleControl& pimple = _pimple();
#include "createFields.H"
_MRF = autoPtr<IOMRFZoneList>(
           new IOMRFZoneList(mesh));
#include "createFvOptions.H"
    ITHACAdict = new IOdictionary(
        IOobject(
            "ITHACAdict",
            runTime.system(),
            mesh,
            IOobject::MUST_READ,
            IOobject::NO_WRITE));
    // Are these calls necessary? Some of these also present in the constructor of unsteadyNS. Check please.
    method = ITHACAdict->lookupOrDefault<word>("method", "supremizer");
    bcMethod = ITHACAdict->lookupOrDefault<word>("bcMethod", "lift");
    timeDependentBC = ITHACAdict->lookupOrDefault<bool>("timeDependentBC", false);
    derivativeInRBF = ITHACAdict->lookupOrDefault<bool>("derivativeInRBF", true);
    centerSnapshots = ITHACAdict->lookupOrDefault<bool>("centerSnapshots", false);
    supremizerApproach = ITHACAdict->lookupOrDefault<word>("supremizerApproach",
        "modes");
    M_Assert(method == "supremizer" || method == "PPE" || method == "VMB",
             "A method must be set to either 'supremizer', 'VMB' or 'PPE' in ITHACAdict");
    M_Assert(bcMethod == "lift" || bcMethod == "penalty"
             || bcMethod == "Gunzburger",
                         "The BC method must be set to lift, penalty or Gunzburger in ITHACAdict");
    turbulence->validate();
    offline = ITHACAutilities::check_off();
    podex = ITHACAutilities::check_pod();
    supex = ITHACAutilities::check_sup();
    viscDict = ITHACAdict->subDict("viscDict");
    NUmodes = ITHACAdict->lookupOrDefault<label>("NmodesUproj", 10);
    NTmodes = ITHACAdict->lookupOrDefault<label>("NmodesTproj", 5);

    if (method == "VMB")
    {
        NPrghmodes = NUmodes;
    }
    else
    {
        NPrghmodes = ITHACAdict->lookupOrDefault<label>("NmodesPrghproj", 5);
    }

    NNutModes = ITHACAdict->lookupOrDefault<label>("NmodesNutproj", 5);
    NSUPmodes = ITHACAdict->lookupOrDefault<label>("NmodesSUPproj", 5);
    if (method != "supremizer")
    {
        NSUPmodes = 0;
    }
    dimInputRBF = ITHACAdict->lookupOrDefault<label>("dimInputRBF", 0);
    M_Assert(dimInputRBF <= NUmodes + NSUPmodes,
             "The dimension of the input to the RBF must be less than or equal to the number of velocity modes.");
    M_Assert(dimInputRBF >= 0,
             "The dimension of the input to the RBF must be greater than or equal to zero.");
    Info << "### INFO ### " << nl << "Method: " << method << nl
         << "BC method: " << bcMethod << nl
         << "Time dependent BCs: " << timeDependentBC << nl
         << "Supremizer approach: " << supremizerApproach << nl
         << "Derivative in RBF for nut interpolation: " << derivativeInRBF << nl
         << "Number of velocity modes for projection: " << NUmodes << nl
         << "Number of temperature modes for projection: " << NTmodes << nl
         << "Number of pressure modes for projection: " << NPrghmodes << nl
         << "Number of eddy viscosity modes for projection: " << NNutModes << nl
         << "Number of supremizer modes for projection: " << NSUPmodes << endl;
}

// * * * * * * * * * * * * * * Full Order Methods * * * * * * * * * * * * * * //
void UnsteadyBBTurb::truthSolve(const List<scalar> mu_now, label nSample)
{
    Time& runTime = _runTime();
    fvMesh& mesh = _mesh();
    Info << "Created mesh and runTime references." << nl << endl;
    #include "initContinuityErrs.H"
    fv::options& fvOptions = _fvOptions();
    singlePhaseTransportModel& laminarTransport = _laminarTransport();
    volScalarField& p = _p();
    volVectorField& U = _U();
    volScalarField& p_rgh = _p_rgh();
    volScalarField& T = _T();
    volScalarField& nut = _nut();
    volScalarField& alphat = _alphat();
    volScalarField& rhok = _rhok();
    volScalarField& gh = _gh();
    // volScalarField& Sourceterm = _Sourceterm();
    surfaceScalarField& ghf = _ghf();
    surfaceScalarField& phi = _phi();
    pimpleControl& pimple = _pimple();
    IOMRFZoneList& MRF = _MRF();
    dimensionedScalar& beta = _beta();
    dimensionedScalar& TRef = _TRef();
    dimensionedScalar& Pr = _Pr();
    dimensionedScalar& Prt = _Prt();
    // Here we handle the time settings. In other instances in ITHACA-FV,
    // runTime.setTime() uses as argument (Times[1],0). Don't know the reason, ask for clarification.
    instantList Times = runTime.times();
    runTime.setEndTime(finalTime);
    runTime.setTime(startTime, 0);
    runTime.setDeltaT(timeStep);
    nextWrite = startTime + writeEvery;
    label nSavedTimesteps = (finalTime - startTime) / writeEvery;
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
            ITHACAstream::exportSolution(U, name(counter), "./ITHACAoutput/Offline/");
            ITHACAstream::exportSolution(p, name(counter), "./ITHACAoutput/Offline/");
            ITHACAstream::exportSolution(p_rgh, name(counter), "./ITHACAoutput/Offline/");
            ITHACAstream::exportSolution(T, name(counter), "./ITHACAoutput/Offline/");
            ITHACAstream::exportSolution(nut, name(counter), "./ITHACAoutput/Offline/");

            timeSnapshots[nSample](stepCounter) = runTime.value();
            stepCounter++;
            Ufield.append(U.clone());
            Prghfield.append(p_rgh.clone());
            Tfield.append(T.clone());
            Nutfield.append(nut.clone());
            nextWrite += writeEvery;
            writeMu(mu_now);
            counter++;
        }
    }
}

void UnsteadyBBTurb::truthSolve(fileName folder)
{
    Time& runTime = _runTime();
    fvMesh& mesh = _mesh();
#include "initContinuityErrs.H"
    fv::options& fvOptions = _fvOptions();
    singlePhaseTransportModel& laminarTransport = _laminarTransport();
    volScalarField& p = _p();
    volVectorField& U = _U();
    volScalarField& p_rgh = _p_rgh();
    volScalarField& T = _T();
    volScalarField& nut = _nut();
    volScalarField& alphat = _alphat();
    volScalarField& rhok = _rhok();
    volScalarField& gh = _gh();
    surfaceScalarField& ghf = _ghf();
    surfaceScalarField& phi = _phi();
    pimpleControl& pimple = _pimple();
    IOMRFZoneList& MRF = _MRF();
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
            ITHACAstream::exportSolution(U, name(counter), folder);
            ITHACAstream::exportSolution(p, name(counter), folder);
            ITHACAstream::exportSolution(p_rgh, name(counter), folder);
            ITHACAstream::exportSolution(T, name(counter), folder);
            ITHACAstream::exportSolution(nut, name(counter), folder);
            std::ofstream of(folder + "/" + name(counter) + "/" + runTime.timeName());
            Ufield.append(U.clone());
            Prghfield.append(p_rgh.clone());
            Tfield.append(T.clone());
            Nutfield.append(nut.clone());
            nextWrite += writeEvery;
            counter++;
        }
    }
}

void UnsteadyBBTurb::solvesupremizer(word type)
{
    Info << "Called the supremizer for UnsteadyBBTurb." << nl << endl;
    M_Assert(type == "modes" || type == "snapshots",
             "You must specify the variable type with either snapshots or modes");
    PtrList<volScalarField> P_sup;

    if (type == "snapshots")
    {
        P_sup = Prghfield;
    }
    else
    {
        P_sup = P_rghmodes.toPtrList();
    }

    if (supex == 1)
    {
        volVectorField U = _U();
        volVectorField Usup(
            IOobject(
                "Usup",
                U.time().timeName(),
                U.mesh(),
                IOobject::NO_READ,
                IOobject::AUTO_WRITE),
            U.mesh(),
            dimensionedVector("zero", U.dimensions(), vector::zero));

        if (type == "snapshots")
        {
            ITHACAstream::read_fields(supfield, Usup, "./ITHACAoutput/supfield/");
        }
        else
        {
            ITHACAstream::read_fields(supmodes, Usup, "./ITHACAoutput/supremizer/");
        }
    }
    else
    {
        volVectorField U = _U();
        volVectorField Usup(
            IOobject(
                "Usup",
                U.time().timeName(),
                U.mesh(),
                IOobject::NO_READ,
                IOobject::AUTO_WRITE),
            U.mesh(),
            dimensionedVector("zero", U.dimensions(), vector::zero));
        dimensionedScalar nu_fake(
            "nu_fake",
            dimensionSet(0, 2, -1, 0, 0, 0, 0),
            scalar(1));
        Vector<double> v(0, 0, 0);

        for (label i = 0; i < Usup.boundaryField().size(); i++)
        {
            if (Usup.boundaryField()[i].type() != "processor")
            {
                ITHACAutilities::changeBCtype(Usup, "fixedValue", i);
                assignBC(Usup, i, v);
                assignIF(Usup, v);
            }
        }

        if (type == "snapshots")
        {
            for (label i = 0; i < P_sup.size(); i++)
            {
                fvVectorMatrix u_sup_eqn(
                    -fvm::laplacian(nu_fake, Usup));
                solve(
                    u_sup_eqn == fvc::grad(P_sup[i]));
                supfield.append(Usup.clone());
                ITHACAstream::exportSolution(Usup, name(i + 1), "./ITHACAoutput/supfield/");
            }

            ITHACAutilities::createSymLink("./ITHACAoutput/supfield");
        }
        else
        {
            for (label i = 0; i < P_sup.size(); i++)
            {
                fvVectorMatrix u_sup_eqn(
                    -fvm::laplacian(nu_fake, Usup));
                solve(
                    u_sup_eqn == fvc::grad(P_sup[i]));
                supmodes.append(Usup.clone());
                ITHACAstream::exportSolution(Usup, name(i + 1), "./ITHACAoutput/supremizer/");
            }

            ITHACAutilities::createSymLink("./ITHACAoutput/supremizer");
        }
    }
}

// * * * * * * * * * * * * * * Load or compute matrices * * * * * * * * * * * //
void UnsteadyBBTurb::loadOrCompute(Eigen::MatrixXd& matrix, const word& folder,
                                   const word& nameBase, const word& suffix,
                                   std::function<Eigen::MatrixXd()> generator)
{
    word fileName = folder + "/" + nameBase + "_" + suffix;

    if (ITHACAutilities::check_file(fileName))
    {
        ITHACAstream::ReadDenseMatrix(matrix, folder + "/", nameBase + "_" + suffix);
        Info << "Loaded matrix " << nameBase << "_" << suffix << " from file." << nl << endl;
    }
    else
    {
        matrix = generator();
    }
}

void UnsteadyBBTurb::loadOrCompute(Eigen::Tensor<double, 3>& tensor,
                                   const word& folder, const word& nameBase, const word& suffix,
                                   std::function<Eigen::Tensor<double, 3>()> generator)
{
    word fileName = folder + "/" + nameBase + "_" + suffix + "_t";

    if (ITHACAutilities::check_file(fileName))
    {
        ITHACAstream::ReadDenseTensor(tensor, folder + "/",
                                      nameBase + "_" + suffix + "_t");
    }
    else
    {
        tensor = generator();
    }
}

void UnsteadyBBTurb::assembleCommonMatrices()
{
    // Initialize the shared pointer to the common matrices struct
    pCommonMatrices = std::make_shared<CommonMatrices>();
    word suffix1 = name(liftfield.size()) + "_" + name(NUmodes) + "_" + name(
                       NSUPmodes);
    word suffix2 = name(liftfield.size()) + "_" + name(NUmodes) + "_" + name(
                       NSUPmodes) + "_" + name(NPrghmodes);
                       
    word suffix3 = name(liftfield.size()) + "_" + name(NUmodes) + "_" + name(
                       NSUPmodes) + "_" + name(liftfieldT.size()) + "_" + name(NTmodes);
    word suffix5 = name(liftfield.size()) + "_" + name(NTmodes);
    word suffix6 = name(liftfield.size()) + "_" + name(NUmodes) + "_" + name(
                       NSUPmodes) + "_" + name(NNutModes);
    word suffix7 = name(liftfield.size()) + "_" + name(NTmodes) + "_" + name(
                       NNutModes);
    word matrixFolder = "./ITHACAoutput/Matrices";
    loadOrCompute(pCommonMatrices->M, matrixFolder, "M", suffix1, [this]()
    {
        return mass_term(NUmodes, NSUPmodes);
    });
    loadOrCompute(pCommonMatrices->B, matrixFolder, "B", suffix1, [this]()
    {
        return diffusive_term(NUmodes, NSUPmodes);
    });
    loadOrCompute(pCommonMatrices->BTurb, matrixFolder, "BT", suffix1, [this]()
    {
        return BTturbulence(NUmodes, NSUPmodes);
    });
    loadOrCompute(pCommonMatrices->K, matrixFolder, "K", suffix2, [this]()
    {
        return pressureGradientTerm(NUmodes, NPrghmodes, NSUPmodes);
    });
    loadOrCompute(pCommonMatrices->H, matrixFolder, "H", suffix3, [this]()
    {
        return buoyancyTerm(NUmodes, NTmodes, NSUPmodes);
    });
    loadOrCompute(pCommonMatrices->W, matrixFolder, "W", suffix5, [this]()
    {
        return massTermTemperature(NTmodes);
    });
    loadOrCompute(pCommonMatrices->Y, matrixFolder, "Y", suffix5, [this]()
    {
        return diffusiveTermTemperature(NUmodes, NTmodes, NSUPmodes);
    });
    loadOrCompute(pCommonMatrices->C, matrixFolder, "C", suffix1, [this]()
    {
        return convective_term_tens(NUmodes, NSUPmodes);
    });
    loadOrCompute(pCommonMatrices->CTurb1, matrixFolder, "CT1",
                  suffix6, [this]()
    {
        return turbulenceTensor1(NUmodes, NSUPmodes, NNutModes);
    });
    loadOrCompute(pCommonMatrices->CTurb2, matrixFolder, "CT2",
                  suffix6, [this]()
    {
        return turbulenceTensor2(NUmodes, NSUPmodes, NNutModes);
    });
    loadOrCompute(pCommonMatrices->AveCTurb1, matrixFolder, "CT1Ave",
                  suffix6, [this]()
    {
        return turbulenceAveTensor1(NUmodes, NSUPmodes);
    });
    loadOrCompute(pCommonMatrices->AveCTurb2, matrixFolder, "CT2Ave",
                  suffix6, [this]()
    {
        return turbulenceAveTensor2(NUmodes, NSUPmodes);
    });
    loadOrCompute(pCommonMatrices->Q, matrixFolder, "Q", suffix2, [this]()
    {
        return convectiveTensorTemperature(NUmodes, NTmodes, NSUPmodes);
    });
    loadOrCompute(pCommonMatrices->YTurb, matrixFolder, "YTurb", suffix7, [this]()
    {
        return temperatureTurbulenceTensor(NTmodes, NNutModes);
    });
    loadOrCompute(pCommonMatrices->AveYTurb, matrixFolder, "AveYTurb",
                  suffix7, [this]()
    {
        return turbulenceTemperatureAveTensor(NTmodes);
    });
    pCommonMatrices->BTotal = pCommonMatrices->B + pCommonMatrices->BTurb;
    label sizeTensorC = NUmodes + NSUPmodes + liftfield.size();
    label sizeTestSpace = testFunctionsU.size();
    pCommonMatrices->CTotal.resize(sizeTestSpace, NNutModes, sizeTensorC);
    pCommonMatrices->CTotal = pCommonMatrices->CTurb1 + pCommonMatrices->CTurb2;
    pCommonMatrices->CTotalAve.resize(sizeTestSpace, avgNutfield.size(),
                                      sizeTensorC);
    pCommonMatrices->CTotalAve = pCommonMatrices->AveCTurb1 +
                                 pCommonMatrices->AveCTurb2;
}

void UnsteadyBBTurb::assembleSupremizerMatrices()
{
    pSupremizerMatrices = std::make_shared<SupremizerMatrices>();
    word suffix = name(liftfield.size()) + "_" + name(NUmodes) + "_" + name(
                      NSUPmodes) + "_" + name(NPrghmodes);
    word matrixFolder = "./ITHACAoutput/Matrices";
    loadOrCompute(pSupremizerMatrices->P, matrixFolder, "P", suffix, [this]()
    {
        return divergenceTerm(NUmodes, NPrghmodes, NSUPmodes);
    });
}

void UnsteadyBBTurb::assemblePPEMatrices()
{
    pPPEMatrices = std::make_shared<PPEMatrices>();
    word suffix1 = name(liftfield.size()) + "_" + name(NUmodes) + "_" + name(
                       NSUPmodes) + "_" + name(NPrghmodes);
    word suffix2 = name(liftfield.size()) + "_" + name(NUmodes) + "_" + name(
                       NSUPmodes) + "_" + name(NPrghmodes) + "_" + name(NTmodes) + "_" + name(
                       NNutModes);
    word matrixFolder = "./ITHACAoutput/Matrices";
    loadOrCompute(pPPEMatrices->D, matrixFolder, "D", suffix1, [this]()
    {
        return laplacianPPE(NPrghmodes);
    });
    loadOrCompute(pPPEMatrices->HP, matrixFolder, "HP", suffix1, [this]()
    {
        return buoyancyTermPPE(NPrghmodes, NTmodes);
    });
    loadOrCompute(pPPEMatrices->nuBC, matrixFolder, "nuBC", suffix1, [this]()
    {
        return BC3PPE(NUmodes, NPrghmodes);
    });
    loadOrCompute(pPPEMatrices->timedepBC, matrixFolder, "timedepBC",
                  suffix1, [this]()
    {
        return BC4PPE(NUmodes, NPrghmodes);
    });
    loadOrCompute(pPPEMatrices->G, matrixFolder, "G", suffix2, [this]()
    {
        return divMomentum(NUmodes, NPrghmodes);
    });
    loadOrCompute(pPPEMatrices->PPEAveCT1, matrixFolder, "PPEAveCT1",
                  suffix2, [this]()
    {
        return turbulencePPEAveTensor1(NUmodes, NPrghmodes);
    });
    loadOrCompute(pPPEMatrices->PPECT1, matrixFolder, "PPECT1", suffix2, [this]()
    {
        return turbulencePPETensor1(NUmodes, NPrghmodes, NNutModes);
    });
    loadOrCompute(pPPEMatrices->PPEAveCT2, matrixFolder, "PPEAveCT2",
                  suffix2, [this]()
    {
        return turbulencePPEAveTensor2(NUmodes, NPrghmodes);
    });
    loadOrCompute(pPPEMatrices->PPECT2, matrixFolder, "PPECT2", suffix2, [this]()
    {
        return turbulencePPETensor2(NUmodes, NPrghmodes, NNutModes);
    });
    label cSize = NUmodes + liftfield.size();
    pPPEMatrices->CTotalPPEAve.resize(NPrghmodes, avgNutfield.size(), cSize);
    pPPEMatrices->CTotalPPEAve = pPPEMatrices->PPEAveCT1 + pPPEMatrices->PPEAveCT2;
    pPPEMatrices->CTotalPPEFluct.resize(NPrghmodes, NNutModes, cSize);
    pPPEMatrices->CTotalPPEFluct = pPPEMatrices->PPECT1 + pPPEMatrices->PPECT2;
}

void UnsteadyBBTurb::assemblePenaltyMatrices()
{
    pPenaltyMatrices = std::make_shared<PenaltyMatrices>();
    pPenaltyMatrices->bcVelVec = bcVelocityVec(NUmodes, NSUPmodes);
    pPenaltyMatrices->bcVelMat = bcVelocityMat(NUmodes, NSUPmodes);
    pPenaltyMatrices->bcTempVec = bcTemperatureVec(NTmodes);
    pPenaltyMatrices->bcTempMat = bcTemperatureMat(NTmodes);
}

void UnsteadyBBTurb::removeMean()
{
    // Uavg = ITHACAutilities::computeAverage(Ufield);
    UavgPtr.reset(new volVectorField(
                      IOobject(
                          "Uavg",
                          _runTime().timeName(),
                          _mesh(),
                          IOobject::NO_READ,
                          IOobject::AUTO_WRITE),
                      Ufield[0]));
    UavgPtr() *= 0.0;
    TavgPtr.reset(new volScalarField(
                      IOobject(
                          "Tavg",
                          _runTime().timeName(),
                          _mesh(),
                          IOobject::NO_READ,
                          IOobject::AUTO_WRITE),
                      Tfield[0]));
    TavgPtr() *= 0.0;
    p_rghavgPtr.reset(new volScalarField(
                          IOobject(
                              "p_rghavg",
                              _runTime().timeName(),
                              _mesh(),
                              IOobject::NO_READ,
                              IOobject::AUTO_WRITE),
                          Prghfield[0]));
    p_rghavgPtr() *= 0.0;

    for (label i = 0; i < Ufield.size(); i++)
    {
        UavgPtr() += Ufield[i];
    }

    UavgPtr() /= Ufield.size();

    for (label patchi = 0; patchi < UavgPtr().boundaryField().size(); patchi++)
    {
        UavgPtr().boundaryFieldRef()[patchi] == 0.0 * Ufield[0].boundaryField()[patchi];

        for (label i = 0; i < Ufield.size(); i++)
        {
            UavgPtr().boundaryFieldRef()[patchi] == UavgPtr().boundaryField()[patchi] +
            Ufield[i].boundaryField()[patchi];
        }

        UavgPtr().boundaryFieldRef()[patchi] == UavgPtr().boundaryField()[patchi] /
        scalar(Ufield.size());
    }

    for (label i = 0; i < Ufield.size(); i++)
    {
        Ufield[i] -= UavgPtr();

        for (label patchi = 0; patchi < UavgPtr().boundaryField().size(); patchi++)
        {
            Ufield[i].boundaryFieldRef()[patchi] == Ufield[i].boundaryField()[patchi] -
            UavgPtr().boundaryField()[patchi];
        }
    }

    // Tavg = ITHACAutilities::computeAverage(Tfield);
    for (label i = 0; i < Tfield.size(); i++)
    {
        TavgPtr() += Tfield[i];
    }

    TavgPtr() /= Tfield.size();

    for (label patchi = 0; patchi < TavgPtr().boundaryField().size(); patchi++)
    {
        TavgPtr().boundaryFieldRef()[patchi] == 0.0 * Tfield[0].boundaryField()[patchi];

        for (label i = 0; i < Tfield.size(); i++)
        {
            TavgPtr().boundaryFieldRef()[patchi] == TavgPtr().boundaryField()[patchi] +
            Tfield[i].boundaryField()[patchi];
        }

        TavgPtr().boundaryFieldRef()[patchi] == TavgPtr().boundaryField()[patchi] /
        scalar(Tfield.size());
    }

    for (label i = 0; i < Tfield.size(); i++)
    {
        Tfield[i] -= TavgPtr();

        for (label patchi = 0; patchi < TavgPtr().boundaryField().size(); patchi++)
        {
            Tfield[i].boundaryFieldRef()[patchi] == Tfield[i].boundaryField()[patchi] -
            TavgPtr().boundaryField()[patchi];
        }
    }

    for (label i = 0; i < Prghfield.size(); i++)
    {
        p_rghavgPtr() += Prghfield[i];
    }

    p_rghavgPtr() /= Prghfield.size();

    for (label patchi = 0; patchi < p_rghavgPtr().boundaryField().size(); patchi++)
    {
        p_rghavgPtr().boundaryFieldRef()[patchi] == 0.0 *
        Prghfield[0].boundaryField()[patchi];

        for (label i = 0; i < Prghfield.size(); i++)
        {
            p_rghavgPtr().boundaryFieldRef()[patchi] ==
            p_rghavgPtr().boundaryField()[patchi] + Prghfield[i].boundaryField()[patchi];
        }

        p_rghavgPtr().boundaryFieldRef()[patchi] ==
        p_rghavgPtr().boundaryField()[patchi] / scalar(Prghfield.size());
    }

    for (label i = 0; i < Prghfield.size(); i++)
    {
        Prghfield[i] -= p_rghavgPtr();

        for (label patchi = 0; patchi < p_rghavgPtr().boundaryField().size(); patchi++)
        {
            Prghfield[i].boundaryFieldRef()[patchi] == Prghfield[i].boundaryField()[patchi]
            - p_rghavgPtr().boundaryField()[patchi];
        }
    }

    ITHACAstream::exportSolution(UavgPtr(), "Uavg", "./ITHACAoutput/Debug/");
    ITHACAstream::exportSolution(TavgPtr(), "Tavg", "./ITHACAoutput/Debug/");
    ITHACAstream::exportSolution(p_rghavgPtr(), "p_rghavg",
                                 "./ITHACAoutput/Debug/");
}

// * * * * * * * * * * * * * * Projection Methods * * * * * * * * * * * * * * //

void UnsteadyBBTurb::project()
{
    prepareModes();
    assembleCommonMatrices();

    if (method == "supremizer")
    {
        M_Assert(NSUPmodes > 0,
                 "The supremizer method requires at least one supremizer mode. Please set the number of supremizer modes to a positive integer in ITHACAdict.");
        assembleSupremizerMatrices();
    }
    else if (method == "PPE")
    {
        M_Assert(NSUPmodes == 0,
                 "The PPE method is not compatible with supremizer modes. Please set the number of supremizer modes to 0 in ITHACAdict.");
        assemblePPEMatrices();
    }

    if (bcMethod == "penalty")
    {
        assemblePenaltyMatrices();
    }

    offlineRBFInterpolation();
}

// * * * * * * * * * * * * * * RBF Prep Methods * * * * * * * * * * * * * * //
void UnsteadyBBTurb::offlineRBFInterpolation()
{
    Eigen::MatrixXd weights;
    Eigen::MatrixXd coeffL2nut = ITHACAutilities::getCoeffs(fluctNutfield, nutmodes,
        NNutModes);
    Eigen::MatrixXd coeffL2vel;
    coeffL2vel.resize(0, 0);
    label inputModes = dimInputRBF;

    if (inputModes == 0)
    {
        inputModes = NUmodes;
    }

    if (bcMethod == "lift")
    {
        coeffL2vel = ITHACAutilities::getCoeffs(Uomfield, L_U_SUPmodes,
                                                inputModes); // Returns a [modes x snapshots]
        skipRBFIndex = liftfield.size();
    }
    else
    {
        coeffL2vel = ITHACAutilities::getCoeffs(Ufield, L_U_SUPmodes,
                                                inputModes); // Returns a [modes x snapshots]
    }

    Info << "Shape of the L2 velocity coeff matrix: " << coeffL2vel.rows() << " x "
         << coeffL2vel.cols() << endl;
    Info << "Shape of the L2 eddy viscosity coeff matrix: " << coeffL2nut.rows() <<
            " x " << coeffL2nut.cols() << endl;
    List<Eigen::MatrixXd> velDerCoeff(2);

    // Returns a list of two matrices: [0] = velocity derivative coeffs, [1] = eddy viscosity coeffs. Each matrix is [snapshots x coeffs]
    if (derivativeInRBF)
    {
        velDerCoeff = velDerivativeCoeff(coeffL2vel.transpose(), coeffL2nut.transpose(),
                                         timeSnapshots);
        dimA = velDerCoeff[0].cols();
    }
    else
    {
        velDerCoeff[0] = coeffL2vel.transpose();
        velDerCoeff[1] = coeffL2nut.transpose();
        dimA = velDerCoeff[0].cols();
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

    rbfSplines.resize(NNutModes);
    Eigen::MatrixXd x = velDerCoeff[0].transpose();

    for (label i = 0; i < NNutModes; i++)
    {
        // Create a RBF interpolator instance
        rbfSplines[i] = std::make_shared<ithacaInterpolator>(viscDict);
        Eigen::VectorXd y = velDerCoeff[1].col(i);
        // rbfSplines[i]->optimizeShapeParameter(x, y, 5);
        rbfSplines[i]->fit(x, y);
        Info << "### INTERPOLATION - Fitting ithacaInterpolator for mode " << i + 1 <<
             " completed. Here's some information:" << endl;
        rbfSplines[i]->printInfo();
    }
}

List<Eigen::MatrixXd> UnsteadyBBTurb::velDerivativeCoeff(
    const Eigen::MatrixXd& A,
    const Eigen::MatrixXd& G,
    const List<Eigen::VectorXd>& snapshotTimes)
{
    const label velCoeffsNum = A.cols();
    const label parsSamplesNum = snapshotTimes.size();
    // 1. Calculate total rows without storing per-sample counts in a vector
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

void UnsteadyBBTurb::splitEddyViscositySnapshots()
{
    const label nSamples = timeSnapshots.size();
    avgNutfield.setSize(nSamples);
    const label totalSnapshots = Nutfield.size();
    fluctNutfield.setSize(totalSnapshots);
    label globalIndex = 0;
    label flatIndex = 0;

    for (label i = 0; i < nSamples; i++)
    {
        label nSnap = timeSnapshots[i].size();
        M_Assert(nSnap > 0, "Each parameter sample must have at least one snapshot");
        // 1. Calculate Average
        // Initialize with the first snapshot of the current sample
        avgNutfield.set(i, new volScalarField(
                            IOobject(
                                "avgNut",
                                Nutfield[globalIndex].time().timeName(),
                                Nutfield[globalIndex].mesh(),
                                IOobject::NO_READ,
                                IOobject::NO_WRITE),
                            Nutfield[globalIndex]));
        volScalarField& avg = avgNutfield[i];

        for (label j = 1; j < nSnap; j++)
        {
            avg += Nutfield[globalIndex + j];
        }

        avg /= scalar(nSnap);

        // 2. Calculate Fluctuations directly
        for (label j = 0; j < nSnap; j++)
        {
            fluctNutfield.set(flatIndex,
                              new volScalarField(Nutfield[globalIndex + j] - avg));
            fluctNutfield[flatIndex].rename("fluctNut");
            flatIndex++;
        }

        globalIndex += nSnap;
    }

    if (DEBUG_MODE)
    {
        ITHACAutilities::createSymLink("./ITHACAoutput/Debug");
        ITHACAstream::exportFields(avgNutfield, "./ITHACAoutput/Debug/", "avgNutfield");
        ITHACAstream::exportFields(fluctNutfield, "./ITHACAoutput/Debug/",
                                   "fluctNutfield");
    }
}

// * * * * * * * * * * * * * * Matrices Methods * * * * * * * * * * * * * * //

Eigen::MatrixXd UnsteadyBBTurb::mass_term(label NUmodes, 
  label NSUPmodes)
{
    label testFunctionSize = testFunctionsU.size();
    label Msize = NUmodes + NSUPmodes + liftfield.size();
    Eigen::MatrixXd M_matrix(testFunctionSize, Msize);

    // Project everything
    for (label i = 0; i < testFunctionSize; i++)
    {
        for (label j = 0; j < Msize; j++)
        {
            M_matrix(i, j) = fvc::domainIntegrate(
                                 testFunctionsU[i] &  L_U_SUPmodes[j]).value();
        }
    }

    if (Pstream::parRun())
    {
        reduce(M_matrix, sumOp<Eigen::MatrixXd>());
    }

    if (Pstream::master())
    {
        ITHACAstream::SaveDenseMatrix(M_matrix, "./ITHACAoutput/Matrices/",
                                      "M_" + name(liftfield.size()) + "_" + name(NUmodes) + "_" + name(NSUPmodes));
        ITHACAstream::exportMatrix(M_matrix, "M_matrix", "numpy",
                                   "./ITHACAoutput/Matrices/python/");
    }

    Info << "### [MATRICES] - Assembled the mass term matrix M of size " <<
         M_matrix.rows() << " x " << M_matrix.cols() << endl;
    return M_matrix;
}

Eigen::MatrixXd UnsteadyBBTurb::diffusive_term(label NUmodes,
        label NSUPmodes)
{
    label Bsize = NUmodes + NSUPmodes + liftfield.size();
    label testFunctionSize = testFunctionsU.size();
    Eigen::MatrixXd B_matrix(testFunctionSize, Bsize);

    for (label j = 0; j < Bsize; j++)
    {
        tmp<volVectorField> laplacianField = fvc::laplacian(dimensionedScalar("1",
            dimless, 1), L_U_SUPmodes[j]);

        for (label i = 0; i < testFunctionSize; i++)
        {
            B_matrix(i, j) = fvc::domainIntegrate(testFunctionsU[i] &
                                                  laplacianField()).value();
        }

        laplacianField.clear(); // Clear the temporary field to free memory
    }

    if (Pstream::parRun())
    {
        reduce(B_matrix, sumOp<Eigen::MatrixXd>());
    }

    if (Pstream::master())
    {
        Info << "### [MATRICES] - Assembled the diffusive term matrix B of size " <<
             B_matrix.rows() << " x " << B_matrix.cols() << endl;
        ITHACAstream::SaveDenseMatrix(B_matrix, "./ITHACAoutput/Matrices/",
                                      "B_" + name(liftfield.size()) + "_" + name(NUmodes) + "_" + name(NSUPmodes));
        ITHACAstream::exportMatrix(B_matrix, "B_matrix", "numpy",
                                   "./ITHACAoutput/Matrices/python/");
    }

    return B_matrix;
}

Eigen::MatrixXd UnsteadyBBTurb::pressureGradientTerm(label NU,
        label NPrgh, label NSUP)
{
    label K1size = testFunctionsU.size();
    label K2size = NPrgh;
    Eigen::MatrixXd K_matrix(K1size, K2size);

    for (label j = 0; j < K2size; j++)
    {
        tmp<volVectorField> gradP = fvc::reconstruct(
                                        fvc::snGrad(P_rghmodes[j]) * P_rghmodes[j].mesh().magSf()
                                    );

        for (label i = 0; i < K1size; i++)
        {
            K_matrix(i, j) = fvc::domainIntegrate(testFunctionsU[i] & gradP()).value();
        }

        gradP.clear(); // Clear the temporary field to free memory
    }

    if (Pstream::parRun())
    {
        reduce(K_matrix, sumOp<Eigen::MatrixXd>());
    }

    if (Pstream::master())
    {
        // Export the matrix
        ITHACAstream::SaveDenseMatrix(K_matrix, "./ITHACAoutput/Matrices/",
                                      "K_" + name(liftfield.size()) + "_" + name(NU) + "_" + name(NSUP) + "_" + name(
                                          NPrgh));
        ITHACAstream::exportMatrix(K_matrix, "K_matrix", "numpy",
                                   "./ITHACAoutput/Matrices/python/");
    }

    Info << "### [MATRICES] - Assembled the pressure gradient term matrix K of size "
         << K_matrix.rows() << " x " << K_matrix.cols() << endl;
    return K_matrix;
}

Eigen::MatrixXd UnsteadyBBTurb::diffusiveTermTemperature(label NU,
        label NT, label NSUP)
{
    label Ysize = NT + liftfieldT.size();
    label testFunctionSize = testFunctionsT.size();
    Eigen::MatrixXd Y_matrix(testFunctionSize, Ysize);

    // More efficient than the reverse
    for (label j = 0; j < Ysize; j++)
    {
        tmp<volScalarField> laplacianField = fvc::laplacian(dimensionedScalar("1",
            dimless, 1), L_Tmodes[j]);

        for (label i = 0; i < testFunctionSize; i++)
        {
            Y_matrix(i, j) = fvc::domainIntegrate(testFunctionsT[i] *
                                                  laplacianField()).value();
        }

        laplacianField.clear(); // Clear the temporary field to free memory
    }

    if (Pstream::parRun())
    {
        reduce(Y_matrix, sumOp<Eigen::MatrixXd>());
    }

    if (Pstream::master())
    {
        // Export the matrix
        ITHACAstream::SaveDenseMatrix(Y_matrix, "./ITHACAoutput/Matrices/",
                                      "Y_" + name(liftfieldT.size()) + "_" + name(NT));
        ITHACAstream::exportMatrix(Y_matrix, "Y_matrix", "numpy",
                                   "./ITHACAoutput/Matrices/python/");
    }

    Info << "### [MATRICES] - Assembled the diffusive term matrix Y of size " <<
         Y_matrix.rows() << " x " << Y_matrix.cols() << endl;
    return Y_matrix;
}

Eigen::MatrixXd UnsteadyBBTurb::divergenceTerm(label NU, label NPrgh,
        label NSUP)
{
    label P1size = NPrgh;
    label P2size = NU + NSUP + liftfield.size();
    Eigen::MatrixXd P_matrix(P1size, P2size);

    for (label j = 0; j < P2size; j++)
    {
        tmp<volScalarField> divField = fvc::div(L_U_SUPmodes[j]);

        for (label i = 0; i < P1size; i++)
        {
            P_matrix(i, j) = fvc::domainIntegrate(P_rghmodes[i] * divField()).value();
        }

        divField.clear();
    }

    if (Pstream::parRun())
    {
        reduce(P_matrix, sumOp<Eigen::MatrixXd>());
    }

    if (Pstream::master())
    {
        // Export the matrix
        ITHACAstream::SaveDenseMatrix(P_matrix, "./ITHACAoutput/Matrices/",
                                      "P_" + name(liftfield.size()) + "_" + name(NU) + "_" + name(NSUP) + "_" + name(
                                          NPrgh));
        ITHACAstream::exportMatrix(P_matrix, "P_matrix", "numpy",
                                   "./ITHACAoutput/Matrices/python/");
    }

    Info << "### [MATRICES] - Assembled the divergence term matrix P of size " <<
         P_matrix.rows() << " x " << P_matrix.cols() << endl;
    return P_matrix;
}

Eigen::MatrixXd UnsteadyBBTurb::buoyancyTerm(label NU, label NT,
        label NSUP)
{
    label H1size = testFunctionsU.size();
    label H2size = NT + liftfieldT.size();
    Eigen::MatrixXd H_matrix(H1size, H2size);
    dimensionedScalar beta = _beta();
    dimensionedScalar TRef = _TRef();
    dimensionedVector g = _g();
    surfaceScalarField& ghf = _ghf();

    for (label j = 0; j < H2size; j++)
    {
        tmp<volVectorField> buoyancyField = fvc::reconstruct(ghf * fvc::snGrad((
                -beta * (L_Tmodes[j]))) * L_Tmodes[j].mesh().magSf());

        for (label i = 0; i < H1size; i++)
        {
            H_matrix(i, j) = fvc::domainIntegrate(testFunctionsU[i] &
                                                  buoyancyField()).value();
        }

        buoyancyField.clear(); // Clear the temporary field to free memory
    }

    if (Pstream::parRun())
    {
        reduce(H_matrix, sumOp<Eigen::MatrixXd>());
    }

    if (Pstream::master())
    {
        // Export the matrix
        ITHACAstream::SaveDenseMatrix(H_matrix, "./ITHACAoutput/Matrices/",
                                      "H_" + name(liftfield.size()) + "_" + name(NU) + "_" + name(NSUP) + "_" + name(
                                          liftfieldT.size()) + "_" + name(NT));
        ITHACAstream::exportMatrix(H_matrix, "H_matrix", "numpy",
                                   "./ITHACAoutput/Matrices/python/");
    }

    Info << "### [MATRICES] - Assembled the buoyancy term matrix H of size " <<
         H_matrix.rows() << " x " << H_matrix.cols() << endl;
    return H_matrix;
}

Eigen::MatrixXd UnsteadyBBTurb::massTermTemperature(label NT)
{
    label Wsize = testFunctionsT.size();
    label Tsize = NT + liftfieldT.size();
    Eigen::MatrixXd W_matrix(Wsize, Tsize);

    for (label i = 0; i < Wsize; i++)
    {
        for (label j = 0; j < Tsize; j++)
        {
            W_matrix(i, j) = fvc::domainIntegrate(testFunctionsT[i] * L_Tmodes[j]).value();
        }
    }

    if (Pstream::parRun())
    {
        reduce(W_matrix, sumOp<Eigen::MatrixXd>());
    }

    if (Pstream::master())
    {
        ITHACAstream::SaveDenseMatrix(W_matrix, "./ITHACAoutput/Matrices/",
                                      "W_" + name(liftfieldT.size()) + "_" + name(NT));
        ITHACAstream::exportMatrix(W_matrix, "W_matrix", "numpy",
                                   "./ITHACAoutput/Matrices/python/");
    }

    Info << "### [MATRICES] - Assembled the mass term matrix W of size " <<
         W_matrix.rows() << " x " << W_matrix.cols() << endl;
    return W_matrix;
}

Eigen::MatrixXd UnsteadyBBTurb::BTturbulence(label NU, label NSUP)
{
    label btSize = NU + NSUP + liftfield.size();
    label testFunctionSize = testFunctionsU.size();
    Eigen::MatrixXd btMatrix(testFunctionSize, btSize);

    for (label j = 0; j < btSize; j++)
    {
        tmp<volVectorField> divDev2Field = fvc::div(dev2((T(fvc::grad(
                L_U_SUPmodes[j])))));

        for (label i = 0; i < testFunctionSize; i++)
        {
            btMatrix(i, j) = fvc::domainIntegrate(testFunctionsU[i] &
                                                  divDev2Field()).value();
        }

        divDev2Field.clear(); // Clear the temporary field to free memory
    }

    if (Pstream::parRun())
    {
        reduce(btMatrix, sumOp<Eigen::MatrixXd>());
    }

    if (Pstream::master())
    {
        ITHACAstream::SaveDenseMatrix(btMatrix, "./ITHACAoutput/Matrices/",
                                      "BT_" + name(liftfield.size()) + "_" + name(NU) + "_" + name(NSUP));
        ITHACAstream::exportMatrix(btMatrix, "BT_matrix", "numpy",
                                   "./ITHACAoutput/Matrices/python/");
    }

    Info << "### [MATRICES] - Assembled the turbulence term matrix BT of size " <<
         btMatrix.rows() << " x " << btMatrix.cols() << endl;
    return btMatrix;
}

Eigen::Tensor<double, 3> UnsteadyBBTurb::convective_term_tens(label NUmodes, label NSUPmodes)
{
    label Csize = NUmodes + NSUPmodes + liftfield.size();
    label testFunctionSize = testFunctionsU.size();
    Eigen::Tensor<double, 3> C_tensor(testFunctionSize, Csize, Csize);

    // More efficient looping
    for (label j = 0; j < Csize; j++)
    {
        for (label k = 0; k < Csize; k++)
        {
            tmp<volVectorField> divField;

            if (fluxMethod == "consistent")
            {
                divField = fvc::div(L_PHImodes[j], L_U_SUPmodes[k]);
            }
            else
            {
                divField = fvc::div(linearInterpolate(L_U_SUPmodes[j]) &
                                    L_U_SUPmodes[j].mesh().Sf(), L_U_SUPmodes[k]);
            }

            for (label i = 0; i < testFunctionSize; i++)
            {
                C_tensor(i, j, k) = fvc::domainIntegrate(testFunctionsU[i] &
                    divField()).value();
            }

            divField.clear();
        }
    }

    if (Pstream::master())
    {
        // Export the tensor
        ITHACAstream::SaveDenseTensor(C_tensor, "./ITHACAoutput/Matrices/",
                                      "C_" + name(liftfield.size()) + "_" + name(NUmodes) + "_" + name(
                                          NSUPmodes) + "_t");
        ITHACAstream::exportTensor(C_tensor, "C_tensor", "numpy",
                                   "./ITHACAoutput/Matrices/python/");
    }

    if (Pstream::parRun())
    {
        reduce(C_tensor, sumOp<Eigen::Tensor<double, 3>>());
    }

    Info << "### [MATRICES] - Assembled the convective term tensor C of size " <<
         C_tensor.dimension(0) << " x " << C_tensor.dimension(1) << " x " <<
         C_tensor.dimension(2) << endl;
    return C_tensor;
}

Eigen::Tensor<double, 3> UnsteadyBBTurb::temperatureTurbulenceTensor(label NT,
        label Nnut)
{
    label Stsize = NT + liftfieldT.size();
    label testFunctionSize = testFunctionsT.size();
    Eigen::Tensor<double, 3> YT_tensor(testFunctionSize, Nnut, Stsize);

    for (label k = 0; k < Stsize; k++)
    {
        for (label j = 0; j < Nnut; j++)
        {
            tmp<volScalarField> laplacianField = fvc::laplacian(nutmodes[j], L_Tmodes[k]);

            for (label i = 0; i < testFunctionSize; i++)
            {
                YT_tensor(i, j, k) = fvc::domainIntegrate(testFunctionsT[i] *
                    laplacianField()).value();
            }

            laplacianField.clear(); // Clear the temporary field to free memory
        }
    }

    if (Pstream::parRun())
    {
        reduce(YT_tensor, sumOp<Eigen::Tensor<double, 3>>());
    }

    if (Pstream::master())
    {
        ITHACAstream::SaveDenseTensor(YT_tensor, "./ITHACAoutput/Matrices/",
                                      "YT_" + name(liftfield.size()) + "_" + name(NT) + "_" + name(Nnut) + "_t");
        ITHACAstream::exportTensor(YT_tensor, "YT_tensor", "numpy",
                                   "./ITHACAoutput/Matrices/python/");
    }

    Info << "### [MATRICES] - Assembled the turbulence term tensor YT of size " <<
         YT_tensor.dimension(0) << " x " << YT_tensor.dimension(1) << " x " <<
         YT_tensor.dimension(2) << endl;
    return YT_tensor;
}

Eigen::Tensor<double, 3> UnsteadyBBTurb::turbulenceTensor1(label NU, label NSUP,
        label Nnut)
{
    label cSize = NU + NSUP + liftfield.size();
    label testFunctionSize = testFunctionsU.size();
    Eigen::Tensor<double, 3> ct1Tensor(testFunctionSize, Nnut, cSize);

    for (label k = 0; k < cSize; k++)
    {
        for (label j = 0; j < Nnut; j++)
        {
            tmp<volVectorField> laplacianField = fvc::laplacian(nutmodes[j],
                L_U_SUPmodes[k]);

            for (label i = 0; i < testFunctionSize; i++)
            {
                ct1Tensor(i, j, k) = fvc::domainIntegrate(testFunctionsU[i] &
                    laplacianField()).value();
            }

            laplacianField.clear(); // Clear the temporary field to free memory
        }
    }

    if (Pstream::parRun())
    {
        reduce(ct1Tensor, sumOp<Eigen::Tensor<double, 3>>());
    }

    if (Pstream::master())
    {
        ITHACAstream::SaveDenseTensor(ct1Tensor, "./ITHACAoutput/Matrices/",
                                      "CT1_" + name(liftfield.size()) + "_" + name(NU) + "_" + name(
                                          NSUP) + "_" + name(Nnut));
        ITHACAstream::exportTensor(ct1Tensor, "CT1_tensor", "numpy",
                                   "./ITHACAoutput/Matrices/python/");
    }

    Info << "### [MATRICES] - Assembled the turbulence term tensor CT1 of size " <<
         ct1Tensor.dimension(0) << " x " << ct1Tensor.dimension(1) << " x " <<
         ct1Tensor.dimension(2) << endl;
    return ct1Tensor;
}

Eigen::Tensor<double, 3> UnsteadyBBTurb::turbulenceAveTensor1(label NU,
        label NSUP)
{
    label cSize = NU + NSUP + liftfield.size();
    label testFunctionSize = testFunctionsU.size();
    label samplesNumber = avgNutfield.size();
    Eigen::Tensor<double, 3> ct1AveTensor(testFunctionSize, samplesNumber, cSize);

    for (label k = 0; k < cSize; k++)
    {
        for (label j = 0; j < samplesNumber; j++)
        {
            tmp<volVectorField> laplacianField = fvc::laplacian(avgNutfield[j],
                L_U_SUPmodes[k]);

            for (label i = 0; i < testFunctionSize; i++)
            {
                ct1AveTensor(i, j, k) = fvc::domainIntegrate(testFunctionsU[i] &
                    laplacianField()).value();
            }

            laplacianField.clear(); // Clear the temporary field to free memory
        }
    }

    if (Pstream::parRun())
    {
        reduce(ct1AveTensor, sumOp<Eigen::Tensor<double, 3>>());
    }

    if (Pstream::master())
    {
        ITHACAstream::SaveDenseTensor(ct1AveTensor, "./ITHACAoutput/Matrices/",
                                      "CT1Ave_" + name(liftfield.size()) + "_" + name(NU) + "_" + name(NSUP));
        ITHACAstream::exportTensor(ct1AveTensor, "CT1_ave_tensor", "numpy",
                                   "./ITHACAoutput/Matrices/python/");
    }

    Info << "### [MATRICES] - Assembled the turbulence term tensor CT1Ave of size "
         << ct1AveTensor.dimension(0) << " x " << ct1AveTensor.dimension(
             1) << " x " << ct1AveTensor.dimension(2) << endl;
    return ct1AveTensor;
}

Eigen::Tensor<double, 3> UnsteadyBBTurb::turbulenceTensor2(label NU, label NSUP,
        label Nnut)
{
    label cSize = NU + NSUP + liftfield.size();
    label testFunctionSize = testFunctionsU.size();
    Eigen::Tensor<double, 3> ct2Tensor(testFunctionSize, Nnut, cSize);

    for (label k = 0; k < cSize; k++)
    {
        for (label j = 0; j < Nnut; j++)
        {
            tmp<volVectorField> divField = fvc::div(nutmodes[j] * dev2((fvc::grad(
                    L_U_SUPmodes[k]))().T()));

            for (label i = 0; i < testFunctionSize; i++)
            {
                ct2Tensor(i, j, k) = fvc::domainIntegrate(testFunctionsU[i] &
                    divField()).value();
            }

            divField.clear(); // Clear the temporary field to free memory
        }
    }

    if (Pstream::parRun())
    {
        reduce(ct2Tensor, sumOp<Eigen::Tensor<double, 3>>());
    }

    if (Pstream::master())
    {
        ITHACAstream::SaveDenseTensor(ct2Tensor, "./ITHACAoutput/Matrices/",
                                      "CT2_" + name(liftfield.size()) + "_" + name(NU) + "_" + name(
                                          NSUP) + "_" + name(Nnut));
        ITHACAstream::exportTensor(ct2Tensor, "CT2_tensor", "numpy",
                                   "./ITHACAoutput/Matrices/python/");
    }

    Info << "### [MATRICES] - Assembled the turbulence term tensor CT2 of size " <<
         ct2Tensor.dimension(0) << " x " << ct2Tensor.dimension(1) << " x " <<
         ct2Tensor.dimension(2) << endl;
    return ct2Tensor;
}

Eigen::Tensor<double, 3> UnsteadyBBTurb::turbulenceAveTensor2(label NU,
        label NSUP)
{
    label cSize = NU + NSUP + liftfield.size();
    label testFunctionSize = testFunctionsU.size();
    label samplesNumber = avgNutfield.size();
    Eigen::Tensor<double, 3> ct2AveTensor(testFunctionSize, samplesNumber, cSize);

    for (label k = 0; k < cSize; k++)
    {
        for (label j = 0; j < samplesNumber; j++)
        {
            tmp<volVectorField> divField = fvc::div(avgNutfield[j] * dev2((fvc::grad(
                    L_U_SUPmodes[k]))().T()));

            for (label i = 0; i < testFunctionSize; i++)
            {
                ct2AveTensor(i, j, k) = fvc::domainIntegrate(testFunctionsU[i] &
                    divField()).value();
            }

            divField.clear(); // Clear the temporary field to free memory
        }
    }

    if (Pstream::parRun())
    {
        reduce(ct2AveTensor, sumOp<Eigen::Tensor<double, 3>>());
    }

    if (Pstream::master())
    {
        ITHACAstream::SaveDenseTensor(ct2AveTensor, "./ITHACAoutput/Matrices/",
                                      "CT2Ave_" + name(liftfield.size()) + "_" + name(NU) + "_" + name(NSUP));
        ITHACAstream::exportTensor(ct2AveTensor, "CT2_ave_tensor", "numpy",
                                   "./ITHACAoutput/Matrices/python/");
    }

    Info << "### [MATRICES] - Assembled the turbulence term tensor CT2Ave of size "
         << ct2AveTensor.dimension(0) << " x " << ct2AveTensor.dimension(
             1) << " x " << ct2AveTensor.dimension(2) << endl;
    return ct2AveTensor;
}

Eigen::Tensor<double, 3> UnsteadyBBTurb::turbulenceTemperatureAveTensor(
    label NT)
{
    label ySize = NT + liftfieldT.size();
    label testFunctionSize = testFunctionsT.size();
    label samplesNumber = avgNutfield.size();
    Eigen::Tensor<double, 3> YTAveTensor(testFunctionSize, samplesNumber, ySize);

    for (label k = 0; k < ySize; k++)
    {
        for (label j = 0; j < samplesNumber; j++)
        {
            tmp<volScalarField> laplacianField = fvc::laplacian(avgNutfield[j],
                L_Tmodes[k]);

            for (label i = 0; i < testFunctionSize; i++)
            {
                YTAveTensor(i, j, k) = fvc::domainIntegrate(testFunctionsT[i] *
                    laplacianField()).value();
            }

            laplacianField.clear();
        }
    }

    if (Pstream::parRun())
    {
        reduce(YTAveTensor, sumOp<Eigen::Tensor<double, 3>>());
    }

    if (Pstream::master())
    {
        ITHACAstream::SaveDenseTensor(YTAveTensor, "./ITHACAoutput/Matrices/",
                                      "YT_ave_" + name(liftfieldT.size()) + "_" + name(NT) + "_t");
        ITHACAstream::exportTensor(YTAveTensor, "YT_ave_tensor", "numpy",
                                   "./ITHACAoutput/Matrices/python/");
    }

    Info << "### [MATRICES] - Assembled the turbulence term tensor YT_ave of size "
         << YTAveTensor.dimension(0) << " x " << YTAveTensor.dimension(
             1) << " x " << YTAveTensor.dimension(2) << endl;
    return YTAveTensor;
}

// * * * * * * * * * * * * * * Energy Eq. Methods * * * * * * * * * * * * * //

Eigen::Tensor<double, 3> UnsteadyBBTurb::convectiveTensorTemperature(label NU,
        label NT, label NSUP)
{
    label Qsize = NU + liftfield.size() + NSUP;
    label Qsizet = NT + liftfieldT.size();
    label testFunctionSize = testFunctionsT.size();
    Eigen::Tensor<double, 3> Q_tensor(testFunctionSize, Qsize, Qsizet);

    for (label k = 0; k < Qsizet; k++)
    {
        for (label j = 0; j < Qsize; j++)
        {
            tmp<volScalarField> divField = fvc::div(fvc::interpolate(
                    L_U_SUPmodes[j]) & L_U_SUPmodes[j].mesh().Sf(), L_Tmodes[k]);

            for (label i = 0; i < testFunctionSize; i++)
            {
                Q_tensor(i, j, k) = fvc::domainIntegrate(testFunctionsT[i] *
                    divField()).value();
            }

            divField.clear();
        }
    }

    if (Pstream::parRun())
    {
        reduce(Q_tensor, sumOp<Eigen::Tensor<double, 3>>());
    }

    if (Pstream::master())
    {
        ITHACAstream::SaveDenseTensor(Q_tensor, "./ITHACAoutput/Matrices/",
                                      "Q_" + name(liftfield.size()) + "_" + name(NU) + "_" +
                                      name(NSUP) + "_" + name(liftfieldT.size()) + "_" + name(NT) + "_t");
        ITHACAstream::exportTensor(Q_tensor, "Q_tensor", "numpy",
                                   "./ITHACAoutput/Matrices/python/");
    }

    Info << "### [MATRICES] - Assembled the convective term tensor Q of size " <<
         Q_tensor.dimension(0) << " x " << Q_tensor.dimension(1) << " x " <<
         Q_tensor.dimension(2) << endl;
    return Q_tensor;
}

Eigen::MatrixXd UnsteadyBBTurb::laplacianPPE(label NPrgh)
{
    label Dsize = NPrgh;
    Eigen::MatrixXd D_matrix(Dsize, Dsize);
    // Improved, employing the symmetry
    PtrList<volVectorField> gradP(Dsize);

    for (label i = 0; i < Dsize; i++)
    {
        gradP.set(i, fvc::grad(P_rghmodes[i]));
    }

    for (label i = 0; i < Dsize; i++)
    {
        for (label j = i; j < Dsize; j++)
        {
            D_matrix(i, j) = fvc::domainIntegrate(gradP[i] & gradP[j]).value();

            if (i != j)
            {
                D_matrix(j, i) = D_matrix(i, j); // Exploit symmetry
            }
        }
    }

    if (Pstream::parRun())
    {
        reduce(D_matrix, sumOp<Eigen::MatrixXd>());
    }

    if (Pstream::master())
    {
        ITHACAstream::SaveDenseMatrix(D_matrix, "./ITHACAoutput/Matrices/",
                                      "D_" + name(NPrghmodes));
        ITHACAstream::exportMatrix(D_matrix, "D_matrix", "numpy",
                                   "./ITHACAoutput/Matrices/python/");
    }

    Info << "### [MATRICES] - Assembled the laplacian term matrix D of size " <<
         D_matrix.rows() << " x " << D_matrix.cols() << endl;
    return D_matrix;
}

Eigen::Tensor<double, 3> UnsteadyBBTurb::divMomentum(label NU, label NPrgh)
{
    label g1Size = NPrgh + liftfieldP.size();
    label g2Size = NU + liftfield.size();
    Eigen::Tensor<double, 3> gTensor(g1Size, g2Size, g2Size);
    PtrList<volVectorField> gradP(g1Size);

    for (label i = 0; i < g1Size; i++)
    {
        gradP.set(i, fvc::grad(P_rghmodes[i]));
    }

    for (label j = 0; j < g2Size; j++)
    {
        for (label k = 0; k < g2Size; k++)
        {
            tmp<volVectorField> divField;

            if (fluxMethod == "consistent")
            {
                divField = fvc::div(L_PHImodes[j], L_U_SUPmodes[k]);
            }
            else
            {
                divField = fvc::div(linearInterpolate(L_U_SUPmodes[j]) &
                                    L_U_SUPmodes[j].mesh().Sf(), L_U_SUPmodes[k]);
            }

            for (label i = 0; i < g1Size; i++)
            {
                gTensor(i, j, k) = fvc::domainIntegrate(gradP[i] & divField()).value();
            }

            divField.clear();
        }
    }

    if (Pstream::parRun())
    {
        reduce(gTensor, sumOp<Eigen::Tensor<double, 3>>());
    }

    if (Pstream::master())
    {
        // Export the tensor
        ITHACAstream::SaveDenseTensor(gTensor, "./ITHACAoutput/Matrices/",
                                      "G_" + name(liftfield.size()) + "_" + name(NU) + "_" + name(NPrgh) + "_t");
        ITHACAstream::exportTensor(gTensor, "G_tensor", "numpy",
                                   "./ITHACAoutput/Matrices/python/");
    }

    Info << "### [MATRICES] - Assembled the divergence term tensor G of size " <<
         gTensor.dimension(0) << " x " << gTensor.dimension(1) << " x " <<
         gTensor.dimension(2) << endl;
    return gTensor;
}

Eigen::MatrixXd UnsteadyBBTurb::buoyancyTermPPE(label NPrgh, label NT)
{
    label H1size = NPrgh;
    label H2size = NT + liftfieldT.size();
    Eigen::MatrixXd HP_matrix(H1size, H2size);
    dimensionedScalar beta = _beta();
    surfaceScalarField& ghf = _ghf();
    // Pressure gradient vectors
    PtrList<volVectorField> gradP(H1size);

    for (label i = 0; i < H1size; i++)
    {
        gradP.set(i, fvc::reconstruct(fvc::snGrad(P_rghmodes[i]) *
                                      P_rghmodes[i].mesh().magSf()));
    }

    // Buoyancy force
    PtrList<volVectorField> buoyancyForce(H2size);

    for (label j = 0; j < H2size; j++)
    {
        buoyancyForce.set(j, fvc::reconstruct(ghf * fvc::snGrad(-(beta *
            (L_Tmodes[j]))) * L_Tmodes[j].mesh().magSf()));
    }

    // Inner product
    for (label i = 0; i < H1size; i++)
    {
        for (label j = 0; j < H2size; j++)
        {
            HP_matrix(i, j) = fvc::domainIntegrate(gradP[i] & buoyancyForce[j]).value();
        }
    }

    if (Pstream::parRun())
    {
        reduce(HP_matrix, sumOp<Eigen::MatrixXd>());
    }

    if (Pstream::master())
    {
        ITHACAstream::SaveDenseMatrix(HP_matrix, "./ITHACAoutput/Matrices/",
                                      "HP_" + name(NPrgh) + "_" + name(liftfieldT.size()) + "_" + name(NT));
        ITHACAstream::exportMatrix(HP_matrix, "HP_matrix", "numpy",
                                   "./ITHACAoutput/Matrices/python/");
    }

    Info << "### [MATRICES] - Assembled the buoyancy term matrix HP of size " <<
         HP_matrix.rows() << " x " << HP_matrix.cols() << endl;
    return HP_matrix;
}

Eigen::MatrixXd UnsteadyBBTurb::BC1PPE(label NU, label NPrgh)
{
    label P_BC1size = NPrgh;
    label P_BC2size = NU + liftfield.size();
    Eigen::MatrixXd BC1_matrix(P_BC1size, P_BC2size);
    const fvMesh& mesh = L_U_SUPmodes[0].mesh();

    for (label i = 0; i < P_BC1size; i++)
    {
        for (label j = 0; j < P_BC2size; j++)
        {
            surfaceScalarField lpl((fvc::interpolate(fvc::laplacian(
                    L_U_SUPmodes[j])) &
                                    mesh.Sf()) *
                                   fvc::interpolate(P_rghmodes[i]));
            double s = 0;

            for (label k = 0; k < lpl.boundaryField().size(); k++)
            {
                if (lpl.boundaryField()[k].type() != "processor")
                {
                    s += gSum(lpl.boundaryField()[k]);
                }
            }

            BC1_matrix(i, j) = s;
        }
    }

    if (Pstream::parRun())
    {
        reduce(BC1_matrix, sumOp<Eigen::MatrixXd>());
    }

    if (Pstream::master())
    {
        ITHACAstream::SaveDenseMatrix(BC1_matrix, "./ITHACAoutput/Matrices/",
                                      "BC1_" + name(liftfield.size()) + "_" + name(NU) + "_" + name(NPrgh));
        ITHACAstream::exportMatrix(BC1_matrix, "BC1_matrix", "numpy",
                                   "./ITHACAoutput/Matrices/python/");
    }

    return BC1_matrix;
}

Eigen::Tensor<double, 3> UnsteadyBBTurb::BC2PPE(label NU, label NPrgh)
{
    label pressureBC1Size = NPrgh;
    label pressureBC2Size = NU + liftfield.size();
    const fvMesh& mesh = L_U_SUPmodes[0].mesh();
    Eigen::Tensor<double, 3> bc2Tensor(pressureBC1Size, pressureBC2Size,
                                       pressureBC2Size);

    for (label i = 0; i < pressureBC1Size; i++)
    {
        for (label j = 0; j < pressureBC2Size; j++)
        {
            for (label k = 0; k < pressureBC2Size; k++)
            {
                surfaceScalarField div_m(fvc::interpolate(fvc::div(fvc::interpolate(
                        L_U_SUPmodes[j]) &
                    mesh.Sf(),
                    L_U_SUPmodes[k])) &
                                         mesh.Sf() * fvc::interpolate(P_rghmodes[i]));
                double s = 0;

                for (label k = 0; k < div_m.boundaryField().size(); k++)
                {
                    if (div_m.boundaryField()[k].type() != "processor")
                    {
                        s += gSum(div_m.boundaryField()[k]);
                    }
                }

                bc2Tensor(i, j, k) = s;
            }
        }
    }

    if (Pstream::parRun())
    {
        reduce(bc2Tensor, sumOp<Eigen::Tensor<double, 3>>());
    }

    if (Pstream::master())
    {
        // Export the tensor
        ITHACAstream::SaveDenseTensor(bc2Tensor, "./ITHACAoutput/Matrices/",
                                      "BC2_" + name(liftfield.size()) + "_" + name(NU) + "_" + name(NPrgh) + "_t");
        ITHACAstream::exportTensor(bc2Tensor, "BC2_tensor", "numpy",
                                   "./ITHACAoutput/Matrices/python/");
    }

    return bc2Tensor;
}

Eigen::MatrixXd UnsteadyBBTurb::BC3PPE(label NU, label NPrgh)
{
    label P3_BC1size = NPrgh;
    label P3_BC2size = NU + liftfield.size();
    Eigen::MatrixXd BC3_matrix(P3_BC1size, P3_BC2size);
    const fvMesh& mesh = L_U_SUPmodes[0].mesh();
    surfaceVectorField n(mesh.Sf() / mesh.magSf());

    for (label i = 0; i < P3_BC1size; i++)
    {
        for (label j = 0; j < P3_BC2size; j++)
        {
            surfaceVectorField BC3 = fvc::interpolate(fvc::curl(L_U_SUPmodes[j])).ref();
            surfaceVectorField BC4 = (n ^ fvc::interpolate(fvc::grad(P_rghmodes[i]))).ref();
            surfaceScalarField BC5 = ((BC3 & BC4) * mesh.magSf()).ref();
            double s = 0;

            for (label k = 0; k < BC5.boundaryField().size(); k++)
            {
                if (BC5.boundaryField()[k].type() != "processor")
                {
                    s += gSum(BC5.boundaryField()[k]);
                }
            }

            BC3_matrix(i, j) = s;
        }
    }

    if (Pstream::parRun())
    {
        reduce(BC3_matrix, sumOp<Eigen::MatrixXd>());
    }

    if (Pstream::master())
    {
        ITHACAstream::SaveDenseMatrix(BC3_matrix, "./ITHACAoutput/Matrices/",
                                      "BC3_" + name(liftfield.size()) + "_" + name(NU) + "_" + name(NPrgh));
        ITHACAstream::exportMatrix(BC3_matrix, "BC3_matrix", "numpy",
                                   "./ITHACAoutput/Matrices/python/");
    }

    return BC3_matrix;
}

Eigen::MatrixXd UnsteadyBBTurb::BC4PPE(label NU, label NPrgh)
{
    label prghSpaceSize = NPrgh;
    label uSpaceSize = NU + liftfield.size();
    Eigen::MatrixXd BC4_matrix(prghSpaceSize, uSpaceSize);
    const fvMesh& mesh = L_U_SUPmodes[0].mesh();
    surfaceVectorField n(mesh.Sf() / mesh.magSf());

    for (label i = 0; i < prghSpaceSize; i++)
    {
        for (label j = 0; j < uSpaceSize; j++)
        {
            surfaceScalarField prghTerm = fvc::interpolate(P_rghmodes[i]).ref();
            surfaceScalarField uNormalTerm = (n & fvc::interpolate(L_U_SUPmodes[j])).ref();
            surfaceScalarField product = ((prghTerm * uNormalTerm) * mesh.magSf()).ref();
            double s = 0;

            for (label k = 0; k < product.boundaryField().size(); k++)
            {
                if (product.boundaryField()[k].type() != "processor")
                {
                    s += gSum(product.boundaryField()[k]);
                }
            }

            BC4_matrix(i, j) = s;
        }
    }

    if (Pstream::parRun())
    {
        reduce(BC4_matrix, sumOp<Eigen::MatrixXd>());
    }

    if (Pstream::master())
    {
        ITHACAstream::SaveDenseMatrix(BC4_matrix, "./ITHACAoutput/Matrices/",
                                      "BC4_" + name(liftfield.size()) + "_" + name(NU) + "_" + name(NPrgh));
        ITHACAstream::exportMatrix(BC4_matrix, "BC4_matrix", "numpy",
                                   "./ITHACAoutput/Matrices/python/");
    }

    return BC4_matrix;
}
// * * * * * * * * * * * * * * PPE Turbulent Methods * * * * * * * * * * * * * //

Eigen::Tensor<double, 3> UnsteadyBBTurb::turbulencePPEAveTensor1(label NU,
        label NPrgh)
{
    const label cSize = NU + liftfield.size();
    const label nAvg = avgNutfield.size();
    Eigen::Tensor<double, 3> ct1PPEAveTensor(NPrgh, nAvg, cSize);
    PtrList<volVectorField> gradP(NPrgh);

    for (label i = 0; i < NPrgh; ++i)
    {
        gradP.set(i, fvc::grad(P_rghmodes[i]));
    }

    for (label j = 0; j < nAvg; ++j)
    {
        for (label k = 0; k < cSize; ++k)
        {
            tmp<volVectorField> laplacianField = fvc::laplacian(avgNutfield[j],
                L_U_SUPmodes[k]);

            for (label i = 0; i < NPrgh; ++i)
            {
                ct1PPEAveTensor(i, j, k) = fvc::domainIntegrate(gradP[i] &
                    laplacianField()).value();
            }

            laplacianField.clear();
        }
    }

    if (Pstream::parRun())
    {
        reduce(ct1PPEAveTensor, sumOp<Eigen::Tensor<double, 3>>());
    }

    if (Pstream::master())
    {
        ITHACAstream::SaveDenseTensor(
            ct1PPEAveTensor,
            "./ITHACAoutput/Matrices/",
            "ct1PPEAve_" + name(liftfield.size()) + "_" + name(NU) + "_" + name(
                NPrgh) + "_t");
        ITHACAstream::exportTensor(ct1PPEAveTensor, "ct1PPEAve_tensor", "numpy",
                                   "./ITHACAoutput/Matrices/python/");
    }

    Info << "### [MATRICES] - Assembled the turbulence term tensor ct1PPEAve of size "
         << ct1PPEAveTensor.dimension(0) << " x " << ct1PPEAveTensor.dimension(
             1) << " x " << ct1PPEAveTensor.dimension(2) << endl;
    return ct1PPEAveTensor;
}

Eigen::Tensor<double, 3> UnsteadyBBTurb::turbulencePPETensor1(label NU,
        label NPrgh, label Nnut)
{
    const label cSize = NU + liftfield.size();
    Eigen::Tensor<double, 3> ct1PPEFluctTensor(NPrgh, Nnut, cSize);
    PtrList<volVectorField> gradP(NPrgh);

    for (label i = 0; i < NPrgh; ++i)
    {
        gradP.set(i, fvc::grad(P_rghmodes[i]));
    }

    for (label j = 0; j < Nnut; ++j)
    {
        for (label k = 0; k < cSize; ++k)
        {
            tmp<volVectorField> laplacianField = fvc::laplacian(nutmodes[j],
                L_U_SUPmodes[k]);

            for (label i = 0; i < NPrgh; ++i)
            {
                ct1PPEFluctTensor(i, j, k) = fvc::domainIntegrate(gradP[i] &
                    laplacianField()).value();
            }

            laplacianField.clear();
        }
    }

    if (Pstream::parRun())
    {
        reduce(ct1PPEFluctTensor, sumOp<Eigen::Tensor<double, 3>>());
    }

    if (Pstream::master())
    {
        ITHACAstream::SaveDenseTensor(
            ct1PPEFluctTensor,
            "./ITHACAoutput/Matrices/",
            "ct1PPEFluct_" + name(liftfield.size()) + "_" + name(NU) + "_" + name(
                NPrgh) + "_t");
        ITHACAstream::exportTensor(ct1PPEFluctTensor, "ct1PPEFluct_tensor", "numpy",
                                   "./ITHACAoutput/Matrices/python/");
    }

    Info << "### [MATRICES] - Assembled the turbulence term tensor ct1PPEFluct of size "
         << ct1PPEFluctTensor.dimension(0) << " x " << ct1PPEFluctTensor.dimension(
             1) << " x " << ct1PPEFluctTensor.dimension(2) << endl;
    return ct1PPEFluctTensor;
}

Eigen::Tensor<double, 3> UnsteadyBBTurb::turbulencePPEAveTensor2(label NU,
        label NPrgh)
{
    const label cSize = NU + liftfield.size();
    const label nAvg = avgNutfield.size();
    Eigen::Tensor<double, 3> ct2PPEAveTensor(NPrgh, nAvg, cSize);
    PtrList<volVectorField> gradP(NPrgh);

    for (label i = 0; i < NPrgh; ++i)
    {
        gradP.set(i, fvc::grad(P_rghmodes[i]));
    }

    for (label j = 0; j < nAvg; ++j)
    {
        for (label k = 0; k < cSize; ++k)
        {
            tmp<volVectorField> divField = fvc::div(avgNutfield[j] * dev2((fvc::grad(
                    L_U_SUPmodes[k]))().T()));

            for (label i = 0; i < NPrgh; ++i)
            {
                ct2PPEAveTensor(i, j, k) = fvc::domainIntegrate(gradP[i] & divField()).value();
            }

            divField.clear();
        }
    }

    if (Pstream::parRun())
    {
        reduce(ct2PPEAveTensor, sumOp<Eigen::Tensor<double, 3>>());
    }

    if (Pstream::master())
    {
        ITHACAstream::SaveDenseTensor(
            ct2PPEAveTensor,
            "./ITHACAoutput/Matrices/",
            "ct2PPEAve_" + name(liftfield.size()) + "_" + name(NU) + "_" + name(
                NPrgh) + "_t");
        ITHACAstream::exportTensor(ct2PPEAveTensor, "ct2PPEAve_tensor", "numpy",
                                   "./ITHACAoutput/Matrices/python/");
    }

    Info << "### [MATRICES] - Assembled the turbulence term tensor ct2PPEAve of size "
         << ct2PPEAveTensor.dimension(0) << " x " << ct2PPEAveTensor.dimension(
             1) << " x " << ct2PPEAveTensor.dimension(2) << endl;
    return ct2PPEAveTensor;
}

Eigen::Tensor<double, 3> UnsteadyBBTurb::turbulencePPETensor2(label NU,
        label NPrgh, label Nnut)
{
    const label cSize = NU + liftfield.size();
    Eigen::Tensor<double, 3> ct2PPEFluctTensor(NPrgh, Nnut, cSize);
    PtrList<volVectorField> gradP(NPrgh);

    for (label i = 0; i < NPrgh; ++i)
    {
        gradP.set(i, fvc::grad(P_rghmodes[i]));
    }

    for (label j = 0; j < Nnut; ++j)
    {
        for (label k = 0; k < cSize; ++k)
        {
            tmp<volVectorField> divField = fvc::div(nutmodes[j] * dev2((fvc::grad(
                    L_U_SUPmodes[k]))().T()));

            for (label i = 0; i < NPrgh; ++i)
            {
                ct2PPEFluctTensor(i, j, k) = fvc::domainIntegrate(gradP[i] &
                    divField()).value();
            }

            divField.clear();
        }
    }

    if (Pstream::parRun())
    {
        reduce(ct2PPEFluctTensor, sumOp<Eigen::Tensor<double, 3>>());
    }

    if (Pstream::master())
    {
        ITHACAstream::SaveDenseTensor(
            ct2PPEFluctTensor,
            "./ITHACAoutput/Matrices/",
            "ct2PPEFluct_" + name(liftfield.size()) + "_" + name(NU) + "_" + name(
                NPrgh) + "_t");
        ITHACAstream::exportTensor(ct2PPEFluctTensor, "ct2PPEFluct_tensor", "numpy",
                                   "./ITHACAoutput/Matrices/python/");
    }

    Info << "### [MATRICES] - Assembled the turbulence term tensor ct2PPEFluct of size "
         << ct2PPEFluctTensor.dimension(0) << " x " << ct2PPEFluctTensor.dimension(
             1) << " x " << ct2PPEFluctTensor.dimension(2) << endl;
    return ct2PPEFluctTensor;
}

// * * * * * * * * * * * * * * Penalty term methods * * * * * * * * * * * * * //

Eigen::MatrixXd UnsteadyBBTurb::bcTemperatureVec(const label NT)
{
    label BCTsize = inletIndexT.rows();
    Eigen::MatrixXd bcTempVec(NT, BCTsize);

    for (label i = 0; i < BCTsize; i++)
    {
        label BCind = inletIndexT(i);

        for (label j = 0; j < NT; j++)
        {
            bcTempVec(j, i) = gSum(L_Tmodes[j].boundaryField()[BCind] *
                                   L_Tmodes[j].mesh().magSf().boundaryField()[BCind]);
        }
    }

    if (Pstream::parRun())
    {
        reduce(bcTempVec, sumOp<Eigen::MatrixXd>());
    }

    if (Pstream::master())
    {
        ITHACAstream::SaveDenseMatrix(bcTempVec, "./ITHACAoutput/Matrices/",
                                      "bcTempVec_" + name(NT));
        ITHACAstream::exportMatrix(bcTempVec, "bcTempVec", "numpy",
                                   "./ITHACAoutput/Matrices/python/");
    }

    return bcTempVec;
}


Eigen::Tensor<double, 3> UnsteadyBBTurb::bcTemperatureMat(const label NT)
{
    label BCTsize = inletIndexT.rows();
    Eigen::Tensor<double, 3> bcTempMat(BCTsize, NT, NT);

    for (label k = 0; k < BCTsize; k++)
    {
        label BCind = inletIndexT(k);

        for (label i = 0; i < NT; i++)
        {
            for (label j = 0; j < NT; j++)
            {
                bcTempMat(k, i, j) = gSum(L_Tmodes[i].boundaryField()[BCind] *
                                          L_Tmodes[j].boundaryField()[BCind] *
                                          L_Tmodes[i].mesh().magSf().boundaryField()[BCind]);
            }
        }
    }

    if (Pstream::parRun())
    {
        reduce(bcTempMat, sumOp<Eigen::Tensor<double, 3>>());
    }

    if (Pstream::master())
    {
        ITHACAstream::SaveDenseTensor(bcTempMat, "./ITHACAoutput/Matrices/",
                                      "bcTempMat_" + name(NT));
        ITHACAstream::exportTensor(bcTempMat, "bcTempMat", "numpy",
                                   "./ITHACAoutput/Matrices/python/");
    }

    return bcTempMat;
}

// * * * * * * * * * * * * * * Preparation of modes * * * * * * * * * *  //
void UnsteadyBBTurb::prepareModes()
{
    L_U_SUPmodes.resize(0);
    L_Tmodes.resize(0);

    if (centerSnapshots)
    {
        L_U_SUPmodes.append(UavgPtr->clone());
        L_Tmodes.append(TavgPtr->clone());
        // Add the Prgh mean field to the top of the volScalarModes (derivated from autoPtr<volScalarField>), without substitutuing
        int oldPsize = P_rghmodes.size();
        P_rghmodes.resize(oldPsize + 1);

        for (label i = oldPsize; i > 0; i--)
        {
            P_rghmodes.set(i, P_rghmodes.release(i - 1));
        }

        P_rghmodes.set(0, p_rghavgPtr->clone());
    }

    if (liftfield.size() != 0)
    {
        for (label k = 0; k < liftfield.size(); k++)
        {
            L_U_SUPmodes.append(liftfield[k].clone());
        }
    }

    if (NUmodes != 0)
    {
        for (label k = 0; k < NUmodes; k++)
        {
            L_U_SUPmodes.append(Umodes[k].clone());
        }
    }

    if (NSUPmodes != 0)
    {
        for (label k = 0; k < NSUPmodes; k++)
        {
            L_U_SUPmodes.append(supmodes[k].clone());
        }
    }

    if (liftfieldT.size() != 0)
    {
        for (label k = 0; k < liftfieldT.size(); k++)
        {
            L_Tmodes.append(liftfieldT[k].clone());
        }
    }

    if (NTmodes != 0)
    {
        for (label k = 0; k < NTmodes; k++)
        {
            L_Tmodes.append(Tmodes[k].clone());
        }
    }

    if (bcMethod == "Gunzburger")
    {
        computeTestFunctionsBC();
    }
    else
    {
        // If we are not using the method, we must use the standard L_U_SUPmodes/L_Tmodes as test functions.
        // Avoid copying the modes if not necessary
        testFunctionsU.resize(0);
        testFunctionsT.resize(0);

        for (label k = 0; k < L_U_SUPmodes.size(); k++)
        {
            testFunctionsU.append(L_U_SUPmodes[k].clone());
        }

        for (label k = 0; k < L_Tmodes.size(); k++)
        {
            testFunctionsT.append(L_Tmodes[k].clone());
        }
    }
}

// * * * * * * * * * * * * * * Restart Methods * * * * * * * * * * * * * //

void UnsteadyBBTurb::restart()
{
    Time& runTime = _runTime();
    fvMesh& mesh = _mesh();
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
                                  IOobject::MUST_READ, IOobject::NO_WRITE), mesh);
    _U() = U_new;
    // _U().correctBoundaryConditions();
    volScalarField T_new(IOobject("T", runTime.timeName(), mesh,
                                  IOobject::MUST_READ, IOobject::NO_WRITE), mesh);
    _T() = T_new;
    // _T().correctBoundaryConditions();
    volScalarField p_rgh_new(IOobject("p_rgh", runTime.timeName(), mesh,
                                      IOobject::MUST_READ, IOobject::NO_WRITE), mesh);
    _p_rgh() = p_rgh_new;
    // _p_rgh().correctBoundaryConditions();
    volScalarField nut_new(IOobject("nut", runTime.timeName(), mesh,
                                    IOobject::MUST_READ, IOobject::NO_WRITE), mesh);
    _nut() = nut_new;
    // _nut().correctBoundaryConditions();
    volScalarField alphat_new(IOobject("alphat", runTime.timeName(), mesh,
                                       IOobject::MUST_READ, IOobject::NO_WRITE), mesh);
    _alphat() = alphat_new;
    // _alphat().correctBoundaryConditions();
    volScalarField UliftBC_new(IOobject("UliftBC", runTime.timeName(), mesh,
                                        IOobject::MUST_READ, IOobject::NO_WRITE), mesh);
    _UliftBC() = UliftBC_new;
    _phi() = linearInterpolate(_U()) & mesh.Sf();
    _rhok() = 1.0 - _beta() * (_T() - _TRef());
    _laminarTransport.clear();
    _laminarTransport = autoPtr<singlePhaseTransportModel>(new
        singlePhaseTransportModel(_U(), _phi()));
    turbulence.clear();
    turbulence = autoPtr<incompressible::turbulenceModel>
                 (incompressible::turbulenceModel::New(_U(), _phi(), _laminarTransport()));
    turbulence->validate();
    _p() = _p_rgh() + _rhok() * _gh();
    // Fix p reference
    pimpleControl& pimple = _pimple();
    setRefCell(_p(), _p_rgh(), pimple.dict(), pRefCell, pRefValue);

    if (_p_rgh().needReference())
    {
        _p() += dimensionedScalar("p", _p().dimensions(),
                                  pRefValue - getRefCellValue(_p(), pRefCell));
    }

    mesh.setFluxRequired(_p_rgh().name());
    Info << "Restart complete." << endl;
}

void UnsteadyBBTurb::resizeModes()
{
    Umodes.resize(NUmodes);
    Tmodes.resize(NTmodes);
    nutmodes.resize(NNutModes);
    P_rghmodes.resize(NPrghmodes);
}

void UnsteadyBBTurb::computeTestFunctionsBC()
{
    // The test functions are computed using QR factorization of the modes. The B matrix is a (Nmodes x NBC) matrix, where each mode is evaluated on a point of the boundary
    const label nBC = inletIndex.rows();
    const label totalUmodes = NUmodes + NSUPmodes;
    const label nUTest = totalUmodes - nBC;

    GunzburgerBCMatrixVelocity.resize(NUmodes, nBC);
    const label nBCT = inletIndexT.rows();
    const label nTTest = NTmodes - nBCT;
    GunzburgerBCMatrixTemperature.resize(NTmodes, nBCT);
    M_Assert(nUTest > 0,
             "Not enough velocity modes to satisfy the boundary conditions. Increase NUmodes or reduce the number of boundary conditions.");
    M_Assert(nTTest > 0,
             "Not enough temperature modes to satisfy the boundary conditions. Increase NTmodes or reduce the number of boundary conditions.");
    scalarField areaU(nBC, 0.0);

    for (label j = 0; j < nBC; j++)
    {
        label BCind = inletIndex(j, 0);
        areaU[j] = gSum(Umodes[0].mesh().magSf().boundaryField()[BCind]);
    }

    for (label j = 0; j < nBC; j++)
    {
        label BCind = inletIndex(j, 0);
        label BCcomp = inletIndex(j, 1);

        for (label i = 0; i < totalUmodes; i++)
        {
            scalar avg = gSum(
                             L_U_SUPmodes[i].boundaryField()[BCind].component(BCcomp) *
                             L_U_SUPmodes[i].mesh().magSf().boundaryField()[BCind]
                         ) / areaU[j];
            GunzburgerBCMatrixVelocity(i, j) = avg;
        }
    }

    // Now, we want to determine the linear combination psi_l of the POD basis that vanish on the boundary
    // We use the property that the last (NUmodes-NBC) columns of the complete Q matrix form a nullspace of B
    Eigen::ColPivHouseholderQR<Eigen::MatrixXd> qr(GunzburgerBCMatrixVelocity);
    Eigen::MatrixXd Q = qr.householderQ() * Eigen::MatrixXd::Identity(totalUmodes,
        totalUmodes);
    Eigen::MatrixXd psiCoeffs = Q.rightCols(nUTest);
    testFunctionsU.setSize(nUTest);

    for (label i = 0; i < nUTest; i++)
    {
        testFunctionsU.set(i, volVectorField::New("testFunctionU" + name(i),
            Umodes[0].mesh(), dimensionedVector(Umodes[0].dimensions(), vector(0, 0, 0))));

        for (label j = 0; j < totalUmodes; j++)
        {
            testFunctionsU[i] += psiCoeffs(j, i) * L_U_SUPmodes[j];
        }

        for (label j = 0; j < nBC; j++)
        {
            label BCind = inletIndex(j, 0);
            vector bcValue(0, 0, 0);
            ITHACAutilities::assignBC(testFunctionsU[i], BCind, bcValue);
        }
    }

    scalarField areaT(nBCT, 0.0);

    for (label j = 0; j < nBCT; j++)
    {
        label BCind = inletIndexT(j, 0);
        areaT[j] = gSum(Tmodes[0].mesh().magSf().boundaryField()[BCind]);
    }

    for (label j = 0; j < nBCT; j++)
    {
        label BCind = inletIndexT(j, 0);

        for (label i = 0; i < NTmodes; i++)
        {
            scalar avg = gSum(
                             Tmodes[i].boundaryField()[BCind] *
                             Tmodes[i].mesh().magSf().boundaryField()[BCind]
                         ) / areaT[j];
            GunzburgerBCMatrixTemperature(i, j) = avg;
        }
    }

    Eigen::ColPivHouseholderQR<Eigen::MatrixXd> qrT(GunzburgerBCMatrixTemperature);
    Eigen::MatrixXd QT = qrT.householderQ() * Eigen::MatrixXd::Identity(NTmodes,
        NTmodes);
    Eigen::MatrixXd psiCoeffsT = QT.rightCols(nTTest);
    testFunctionsT.setSize(nTTest);

    for (label i = 0; i < nTTest; i++)
    {
        testFunctionsT.set(
            i,
            volScalarField::New(
                "testT_" + name(i),
                Tmodes[0].mesh(),
                dimensionedScalar(Tmodes[0].dimensions(), Zero)
            )
        );

        for (label j = 0; j < NTmodes; ++j)
        {
            testFunctionsT[i] += psiCoeffsT(j, i) * Tmodes[j];
        }

        // Clean up boundary floating-point noise
        for (label j = 0; j < nBCT; ++j)
        {
            const label BCind = inletIndexT(j, 0);
            ITHACAutilities::assignBC(testFunctionsT[i], BCind, 0.0);
        }
    }

    // Assign the matrices to the shared pointers for later use
    pGunzburgerMatrices = std::make_shared<GunzburgerMatrices>();
    pGunzburgerMatrices->bcTempMat = GunzburgerBCMatrixTemperature;
    pGunzburgerMatrices->bcVelMat = GunzburgerBCMatrixVelocity;

    // ITHACAstream::exportFields(testFunctionsU, "./ITHACAoutput/testFunctions/", testFunctionsU[0].name()); // For Debugging
    // ITHACAstream::exportFields(testFunctionsT, "./ITHACAoutput/testFunctions/", testFunctionsT[0].name());

    for (label i = 0; i < testFunctionsU.size(); i++)
    {
        for (label j = 0; j < inletIndex.rows(); j++)
        {
            label BCind = inletIndex(j, 0);
            label BCcomp = inletIndex(j, 1);
            scalar area = gSum(testFunctionsU[i].mesh().magSf().boundaryField()[BCind]);
            scalar avg = gSum(testFunctionsU[i].boundaryField()[BCind].component(
                                  BCcomp) * testFunctionsU[i].mesh().magSf().boundaryField()[BCind]) / area;

            if (mag(avg) > 1e-6)
            {
                WarningIn("UnsteadyBBTurb::computeTestFunctionsBC()")
                        << "Test function " << i << " is not zero on the boundary condition " << j <<
                   " (average value: " << avg << ")" << endl;
            }
        }
    }

    for (label i = 0; i < testFunctionsT.size(); i++)
    {
        for (label j = 0; j < inletIndexT.rows(); j++)
        {
            label BCind = inletIndexT(j, 0);
            scalar area = gSum(testFunctionsT[i].mesh().magSf().boundaryField()[BCind]);
            scalar avg = gSum(testFunctionsT[i].boundaryField()[BCind] *
                              testFunctionsT[i].mesh().magSf().boundaryField()[BCind]) / area;

            if (mag(avg) > 1e-6)
            {
                WarningIn("UnsteadyBBTurb::computeTestFunctionsBC()")
                        << "Test function " << i << " is not zero on the boundary condition " << j <<
                   " (average value: " << avg << ")" << endl;
            }
        }
    }
}
