#include "ReducedLSPGUnsteadyNSTurb.H"
#include "Foam2Eigen.H" // Needed to convert OpenFOAM fields to Eigen vectors and matrices
#include "ITHACAassign.H"
#include <fstream>
#include <iomanip>

namespace
{
/// RMS over the snapshots (columns) of each coefficient (rows)
Eigen::VectorXd rowRMS(const Eigen::MatrixXd& coeffs)
{
    Eigen::VectorXd rms = (coeffs.rowwise().squaredNorm() / scalar(coeffs.cols())).cwiseSqrt();

    // A mode that is (numerically) absent from the snapshots gets the largest scale instead of ~0
    const double maxRms = rms.maxCoeff() > 0 ? rms.maxCoeff() : 1.0;
    for (label i = 0; i < rms.size(); i++)
    {
        if (!(rms(i) > 1e-12 * maxRms))
        {
            rms(i) = maxRms;
        }
    }
    return rms;
}

/// RMS over the snapshots of the projection coefficient of each mode
template <class Type>
Eigen::VectorXd coefficientRMS(
    PtrList<GeometricField<Type, fvPatchField, volMesh>>& snapshots,
    PtrList<GeometricField<Type, fvPatchField, volMesh>>& modes)
{
    M_Assert(snapshots.size() > 0, "coefficientRMS: no snapshots available to compute the state scaling.");
    return rowRMS(ITHACAutilities::getCoeffs(snapshots, modes, modes.size()));
}

/// Adds w F^T V F to G and w F^T V S to B, with F the modes and S the snapshots (columns)
template <class Type>
void addJointProjection(
    PtrList<GeometricField<Type, fvPatchField, volMesh>>& modes,
    const Eigen::MatrixXd& snapshots,
    const scalar w,
    Eigen::MatrixXd& G,
    Eigen::MatrixXd& B)
{
    const Eigen::MatrixXd F = Foam2Eigen::PtrList2Eigen(modes);
    const Eigen::VectorXd V = ITHACAutilities::getMassMatrixFV(modes[0]);
    const Eigen::MatrixXd FtV = F.transpose() * V.asDiagonal();
    G += w * FtV * F;
    B += w * FtV * snapshots;
}

/// RMS over the snapshots of sqrt(sum_cells |f|^2 V / sum V)
template <class Type>
scalar fieldRMS(const PtrList<GeometricField<Type, fvPatchField, volMesh>>& snapshots)
{
    M_Assert(snapshots.size() > 0, "fieldRMS: no snapshots available to compute the field scale.");
    const scalarField& V = snapshots[0].mesh().V().field();
    const scalar Vtot = sum(V);
    scalar sumSq = 0;
    forAll(snapshots, i)
    {
        sumSq += sum(magSqr(snapshots[i].primitiveField()) * V) / Vtot;
    }
    return Foam::sqrt(sumSq / snapshots.size());
}
}


ReducedLSPGUnsteadyNSTurb::ReducedLSPGUnsteadyNSTurb(
    std::shared_ptr<ITHACAcontext> context,
    const LSPGUnsteadyNSTurb& problem): ReducedLSPG(std::move(context))
{
    captureFOMData(problem);

    Time& runTime = this->runTime();
    fvMesh& mesh = this->mesh();
    pimpleControl& pimple = this->pimple();

#include "createFields.H"
    // turbulence = autoPtr<RBFTurbulence>(new RBFTurbulence(nut, nu)); // RBF approach

    volScalarField& k = _k();
    volScalarField& omega = _omega();

    createTurbulenceModel();

    const label state_size = numberOfModes_.velocity +
        numberOfModes_.pressure +
        numberOfModes_.k + numberOfModes_.omega;

    // The Rhie-Chow flux phi = phiHbyA - pEqn.flux() needs the pressure flux
    mesh.setFluxRequired(_p().name());

    const dictionary& solverDict = ITHACAdict->subOrEmptyDict("lmSolver");
    lmSettings_.maxIterations = solverDict.getOrDefault<label>("maxIterations", lmSettings_.maxIterations);
    lmSettings_.gradientTol = solverDict.getOrDefault<scalar>("gradientTol", lmSettings_.gradientTol);
    lmSettings_.reductionTol = solverDict.getOrDefault<scalar>("reductionTol", lmSettings_.reductionTol);
    lmSettings_.stepTol = solverDict.getOrDefault<scalar>("stepTol", lmSettings_.stepTol);
    M_Assert(lmSettings_.maxIterations > 0, "lmSolver: maxIterations must be positive.");
    M_Assert(lmSettings_.gradientTol >= 0 && lmSettings_.reductionTol >= 0 && lmSettings_.stepTol >= 0,
        "lmSolver: the tolerances must be non-negative.");
    Info << "### LM stopping criteria: gradientTol " << lmSettings_.gradientTol << ", reductionTol "
         << lmSettings_.reductionTol << ", stepTol " << lmSettings_.stepTol << ", maxIterations "
         << lmSettings_.maxIterations << endl;

    const dictionary& diagDict = ITHACAdict->subOrEmptyDict("lmDiagnostics");
    lmDiagnostics_.enabled = diagDict.getOrDefault<bool>("enabled", false);
    lmDiagnostics_.triggers = diagDict.getOrDefault<wordList>("triggers", wordList({ "XtolTooSmall" }));
    lmDiagnostics_.maxDumps = diagDict.getOrDefault<label>("maxDumps", 3);
    lmDiagnostics_.steps = diagDict.getOrDefault<scalarList>("steps",
        scalarList({ 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8 }));
    lmDiagnostics_.coordinateScan = diagDict.getOrDefault<bool>("coordinateScan", true);

    idwExponent_ = ITHACAdict->getOrDefault<scalar>("parametricDistanceExponent", 1.0);
    idwNeighbours_ = ITHACAdict->getOrDefault<label>("maxInterpolatedSolutions", 0);
    M_Assert(idwExponent_ > 0, "parametricDistanceExponent must be positive.");
    Info << "### Affine reference: inverse distance weighting, exponent " << idwExponent_ << ", "
         << (idwNeighbours_ > 0 ? std::to_string(idwNeighbours_) + " nearest runs" : std::string("all runs")).c_str()
         << endl;

    currentState_ = Eigen::VectorXd::Zero(state_size);
    oldState_ = Eigen::VectorXd::Zero(state_size);
    buildStateMap();
    currentCoords_ = Eigen::VectorXd::Zero(stateMap_.size() ? stateMap_.cols() : state_size);

    // Scaled coordinates y = a / rms(a_training) are O(1) for every mode and field
    M_Assert(coefficientScale_.size() == currentCoords_.size(), "Coefficient scale size does not match the number of reduced coordinates.");
    stateScaling_ = coefficientScale_;
    Info << "### State scaling (RMS of the training coefficients): " << stateScaling_.transpose() << endl;

    std::vector<BlockSpec> blocks {
        { "U_r", 3 * nCells, 1.0 },
        { "p_r", nCells, 1.0 },
        { "k_r", nCells, 1.0 },
        { "omega_r", nCells, 1.0 }
    };
    if (romSettings_.fluxResidual)
    {
        blocks.push_back({ "phi_r", mesh.nInternalFaces(), 1.0 });
    }
    residualWorkspace_ = BlockResidual { blocks };

    iU_ = residualWorkspace_.index("U_r");
    iP_ = residualWorkspace_.index("p_r");
    iK_ = residualWorkspace_.index("k_r");
    iOmega_ = residualWorkspace_.index("omega_r");
    iPhi_ = romSettings_.fluxResidual ? residualWorkspace_.index("phi_r") : -1;
    if (romSettings_.scaleResidualWithCellVolume)
    {
        const Eigen::VectorXd wScalar = Foam2Eigen::field2Eigen(mesh.V().field()).cwiseSqrt().cwiseInverse();
        vectorField wVecField(nCells);
        forAll(wVecField, celli)
        {
            const scalar w = 1.0 / Foam::sqrt(mesh.V()[celli]);
            wVecField[celli] = vector(w, w, w);
        }
        const Eigen::VectorXd wVector = Foam2Eigen::field2Eigen(wVecField);

        residualWorkspace_.setBlockRowWeights(iU_, wVector);
        residualWorkspace_.setBlockRowWeights(iP_, wScalar);
        residualWorkspace_.setBlockRowWeights(iK_, wScalar);
        residualWorkspace_.setBlockRowWeights(iOmega_, wScalar);

        if (iPhi_ >= 0)
        {
            // The flux residual is in m3/s: divided by |Sf| it is a normal velocity, then weighted by the
            // square root of the face volume |Sf| |d|, as the cell blocks are weighted by sqrt(V)
            const scalarField& magSf = mesh.magSf().primitiveField();
            const surfaceVectorField delta(mesh.delta());
            Eigen::VectorXd wPhi(mesh.nInternalFaces());
            for (label facei = 0; facei < mesh.nInternalFaces(); facei++)
            {
                wPhi(facei) = Foam::sqrt(magSf[facei] * mag(delta[facei])) / magSf[facei];
            }
            residualWorkspace_.setBlockRowWeights(iPhi_, wPhi);
        }
    }

    if (romSettings_.residualPreconditioner == "jacobi")
    {
        // Each preconditioned block is an increment of its unknown: normalise by the field scale.
        // The phi block, through its row weights, is a normal velocity
        residualWorkspace_.setBlockWeight(iU_, 1.0 / fieldScale_(0));
        residualWorkspace_.setBlockWeight(iP_, 1.0 / fieldScale_(1));
        residualWorkspace_.setBlockWeight(iK_, 1.0 / fieldScale_(2));
        residualWorkspace_.setBlockWeight(iOmega_, 1.0 / fieldScale_(3));
        if (iPhi_ >= 0)
        {
            residualWorkspace_.setBlockWeight(iPhi_, 1.0 / fieldScale_(0));
        }
        Info << "### Jacobi-preconditioned residual, field scales (U p k omega): " << fieldScale_.transpose() << endl;
    }
}

void ReducedLSPGUnsteadyNSTurb::captureFOMData(
    const LSPGUnsteadyNSTurb& problem)
{
    interpolationSettings_ = InterpolationSettings {
        problem.ITHACAdict->lookupOrDefault<int>("firstRBFIndex", 0),
        problem.ITHACAdict->lookupOrDefault<bool>("derivativeInRBF", false),
        problem.ITHACAdict->lookupOrDefault<label>("dimInputRBF", 0),
    };

    numberOfModes_ = NumberOfModes {
        problem.ITHACAdict->lookupOrDefault<label>("NmodesUproj", 0),
        problem.ITHACAdict->lookupOrDefault<label>("NmodesPproj", 0),
        problem.ITHACAdict->lookupOrDefault<label>("NmodesKproj", 0),
        problem.ITHACAdict->lookupOrDefault<label>("NmodesOmegaproj", 0),
    };
    M_Assert(numberOfModes_.velocity > 0, "Number of velocity modes must be greater than zero.");
    M_Assert(numberOfModes_.pressure > 0, "Number of pressure modes must be greater than zero.");
    M_Assert(numberOfModes_.k > 0, "Number of k modes must be greater than zero.");
    M_Assert(numberOfModes_.omega > 0, "Number of omega modes must be greater than zero.");

    mu_ = problem.mu;
    rbfSplines_ = problem.rbfSplines;

    // I don't think here we are actually safe from deleting the FOM, check later
    Umodes_ = problem.Umodes;
    M_Assert(Umodes_.size() == numberOfModes_.velocity, "Number of velocity modes does not match the size of the velocity modes matrix.");
    Phimodes_ = problem.Phimodes;
    M_Assert(Phimodes_.size() == numberOfModes_.velocity, "Number of velocity modes does not match the size of the face flux modes matrix.");
    Pmodes_ = problem.Pmodes;
    M_Assert(Pmodes_.size() == numberOfModes_.pressure, "Number of pressure modes does not match the size of the pressure modes matrix.");
    kModes_ = problem.kmodes;
    Info << "Number of k modes: " << kModes_.size() << nl;
    Info << "Requested number of k modes: " << numberOfModes_.k << nl;
    M_Assert(kModes_.size() == numberOfModes_.k, "Number of k modes does not match the size of the k modes matrix.");
    omegaModes_ = problem.omegamodes;
    M_Assert(omegaModes_.size() == numberOfModes_.omega, "Number of omega modes does not match the size of the omega modes matrix.");
    // Nutmodes_ = problem.nutmodes;
    // M_Assert(Nutmodes_.size() == numberOfModes_.nut, "Number of nut modes does not match the size of the nut modes matrix.");

    liftFields_ = problem.liftfield;
    liftFieldsPhi_ = problem.liftfieldphi;

    // PtrLists containing the initial condition of each offline trajectory.
    U0field_ = problem.U0field;
    Phi0field_ = problem.Phi0field;
    P0field_ = problem.P0field;
    // Nut0field_ = problem.Nut0field;
    k0field_ = problem.k0field;
    omega0field_ = problem.omega0field;

    // nutFields_ = problem.fluctNutfield;
    // avgNutFields_ = problem.avgNutfield;

    inletIndex_ = problem.inletIndex;

    // Stacked POD: the fields of jointPOD share their coefficients
    const wordList fieldNames({ "U", "p", "k", "omega" });
    forAll(fieldNames, f)
    {
        jointField_[f] = problem.jointPOD.found(fieldNames[f]);
    }
    jointWeights_ = problem.jointPODWeights;

    // RMS of the training values of each reduced coordinate, used to scale the reduced state.
    // Computed on the unnormalised snapshots, i.e. on the coefficients the ROM actually sees.

    // This const_cast is quite ugly, but is needed since getCoeffs accepts a non-const PtrList, and we don't want to copy the snapshots.
    LSPGUnsteadyNSTurb& fom = const_cast<LSPGUnsteadyNSTurb&>(problem);
    PtrList<volVectorField>& Usnapshots = (romSettings_.bcMethod == "lift") ? fom.Uomfield : fom.Ufield;
    std::vector<Eigen::VectorXd> scales;
    if (problem.jointPOD.size())
    {
        List<Eigen::MatrixXd> snapshots(4);
        snapshots[0] = Foam2Eigen::PtrList2Eigen(Usnapshots);
        if (jointField_[1])
            snapshots[1] = Foam2Eigen::PtrList2Eigen(fom.Pfield);
        if (jointField_[2])
            snapshots[2] = Foam2Eigen::PtrList2Eigen(fom.kfield);
        if (jointField_[3])
            snapshots[3] = Foam2Eigen::PtrList2Eigen(fom.omegafield);
        scales.push_back(rowRMS(jointCoefficients(snapshots)));
    }
    if (!jointField_[0])
        scales.push_back(coefficientRMS(Usnapshots, Umodes_));
    if (!jointField_[1])
        scales.push_back(coefficientRMS(fom.Pfield, Pmodes_));
    if (!jointField_[2])
        scales.push_back(coefficientRMS(fom.kfield, kModes_));
    if (!jointField_[3])
        scales.push_back(coefficientRMS(fom.omegafield, omegaModes_));
    label nCoords = 0;
    for (const Eigen::VectorXd& s : scales)
    {
        nCoords += s.size();
    }
    coefficientScale_.resize(nCoords);
    nCoords = 0;
    for (const Eigen::VectorXd& s : scales)
    {
        coefficientScale_.segment(nCoords, s.size()) = s;
        nCoords += s.size();
    }

    // Field scales of the preconditioned residual: RMS over the snapshots of the volume-weighted L2 norm
    fieldScale_.resize(4);
    fieldScale_ << fieldRMS(Usnapshots), fieldRMS(fom.Pfield), fieldRMS(fom.kfield), fieldRMS(fom.omegafield);
    const scalarList userScales = problem.ITHACAdict->lookupOrDefault<scalarList>("preconditionerScales", scalarList());
    if (userScales.size())
    {
        M_Assert(userScales.size() == 4, "preconditionerScales must have 4 entries: (U p k omega).");
        for (label i = 0; i < 4; i++)
        {
            fieldScale_(i) = userScales[i];
        }
    }
    M_Assert(fieldScale_.minCoeff() > 0, "The preconditioner field scales must be positive.");
}

void ReducedLSPGUnsteadyNSTurb::createTurbulenceModel()
{
    const dictionary* turbDictPtr = nullptr;
    fvMesh& mesh = this->mesh();

    autoPtr<IOdictionary> localDict;

    for (const word& name : { word("momentumTransport"), word("turbulenceProperties") })
    {
        if (mesh.foundObject<IOdictionary>(name))
        {
            turbDictPtr = &mesh.lookupObject<IOdictionary>(name);
            break;
        }
    }

    if (!turbDictPtr)
    {
        for (const word& name : { word("momentumTransport"), word("turbulenceProperties") })
        {
            IOobject io(
                name, mesh.time().constant(), mesh,
                IOobject::MUST_READ, IOobject::NO_WRITE);
            if (io.typeHeaderOk<IOdictionary>(true))
            {
                localDict.reset(new IOdictionary(io));
                turbDictPtr = localDict.get();
                break;
            }
        }
    }

    if (!turbDictPtr)
    {
        FatalErrorInFunction
            << "Neither momentumTransport nor turbulenceProperties found in "
            << mesh.time().constant() << exit(FatalError);
    }

    const word simType = turbDictPtr->getOrDefault<word>("simulationType", "RAS");
    if (simType != "RAS")
    {
        FatalErrorInFunction
            << "simulationType " << simType << " not supported, only RAS"
            << exit(FatalError);
    }
    const dictionary& rasDict = turbDictPtr->subDict("RAS");

    const word modelType =
        rasDict.found("model") ? rasDict.get<word>("model")
                               : rasDict.get<word>("RASModel");

    ITHACATurbulenceModels::ResidualEddyViscosityTurbulenceModel::SharedFields shared {
        *_nu, *_nut, *_U, *_phi
    };

    if (modelType == "kOmegaSST")
    {
        turbulence.reset(
            new ITHACATurbulenceModels::ResidualAssemblerSST(
                mesh,
                mesh.time(),
                shared,
                kModes_,
                omegaModes_,
                k0field_,
                omega0field_,
                rasDict.optionalSubDict("kOmegaSSTCoeffs"),
                ITHACAdict->lookupOrDefault<scalar>("turbulenceBoundFactor", 0.1)));
    } else
    {
        FatalErrorInFunction
            << "Unsupported RAS model " << modelType
            << ". Available: kOmegaSST" << exit(FatalError);
    }
}

void ReducedLSPGUnsteadyNSTurb::solveOnline(const word& bcROM, const word& folder)
{
    if (Pstream::master())
    {
        mkDir(folder);
        ITHACAutilities::createSymLink(folder);
    }
    resetToInitialCondition();
    boundaryConditions_ = BoundaryConditions { bcROM, "linear" };
    nDiagnosticDumps_ = 0;
    // Residual fields go to the ROM case folder (folder/0 is a link to the case 0 directory)
    residualOutputFolder_ = folder;
    Time& runTime = this->runTime();
    fvMesh& mesh = this->mesh();

#include "readTimeControls.H"
#include "initContinuityErrs.H"

    initializePODCoeffsFromFields();
    updatePreconditioner_ = true;
    assembleResidual(currentState_, true, true);

    Info << "\nInitial residual norm: " << residualWorkspace_.vector().norm() << nl
         << "  - U residual norm " << residualWorkspace_.blockNorm(iU_) << nl
         << "  - p residual norm " << residualWorkspace_.blockNorm(iP_) << nl
         << "  - k residual norm " << residualWorkspace_.blockNorm(iK_) << nl
         << "  - omega residual norm " << residualWorkspace_.blockNorm(iOmega_) << nl
         << "  - phi residual norm " << phiBlockNorm() << nl << endl;
    if (romSettings_.residualPreconditioner == "none")
    {
        residualWorkspace_.calibrateWeights(1e-4);
    }

    // Per-step residual history, for post-processing (e.g. the consistency test)
    std::ofstream residualLog;
    if (Pstream::master())
    {
        residualLog.open(folder + "/residualHistory.dat");
        residualLog << std::scientific << std::setprecision(8)
                    << "# deltaT " << runTime.deltaTValue() << nl
                    << "# block weights (U p k omega phi): " << residualBlockWeights().transpose() << nl
                    << "# gradientCosine: max_j |J_j^T r| / (|J_j| |r|) at the last Jacobian evaluation" << nl
                    << "# time initialTotal finalTotal U p k omega phi lmIterations nfev gradientCosine status" << std::endl;
    }

    // The functor provides its own finite-difference Jacobian (LevenbergMarquardtFunctor::df)
    LevenbergMarquardtFunctor lmFunctor(this, currentCoords_.size(), residualWorkspace_.size());
    Eigen::LevenbergMarquardt<LevenbergMarquardtFunctor> lm(lmFunctor);

    lm.parameters.ftol = lmSettings_.reductionTol;
    lm.parameters.xtol = lmSettings_.stepTol;
    lm.parameters.gtol = lmSettings_.gradientTol;
    lm.parameters.maxfev = currentCoords_.size() * 500;

    // Own write schedule: Time::setTime() does not reset the write index of runTime, so after an
    // offline run up to endTime, runTime.writeTime() would never fire again from t = 0
    const scalar writeInterval = runTime.controlDict().get<scalar>("writeInterval");
    scalar nextWrite = runTime.value() + writeInterval;

    auto start = std::chrono::high_resolution_clock::now();
    while (runTime.run())
    {
        runTime++;
        Info << "Time = " << runTime.time().value() << nl;
        boundaryConditions_.updateTimeDependentBC(runTime.time().value());

        Eigen::VectorXd scaledState = currentCoords_.cwiseQuotient(stateScaling_);
        // minimizeInit evaluates the residual at the initial guess: the preconditioner is built there
        updatePreconditioner_ = true;
        Eigen::LevenbergMarquardtSpace::Status status = lm.minimizeInit(scaledState);
        const double initialResidualNorm = residualWorkspace_.vector().norm();
        int lmIterations = 0;

        // Every stop other than Running is decided inside LM: gtol (stationarity, checked on the Jacobian at
        // the current point before stepping), ftol (reduction) or xtol (trust region), or a failure status
        while ((status == Eigen::LevenbergMarquardtSpace::NotStarted || status == Eigen::LevenbergMarquardtSpace::Running)
            && lmIterations < lmSettings_.maxIterations)
        {
            status = lm.minimizeOneStep(scaledState);
            lmIterations++;
        }
        if (status == Eigen::LevenbergMarquardtSpace::Running)
        {
            Info << "LM reached maximum iterations (" << lmSettings_.maxIterations << ") without converging." << nl;
        }
        Info << "LM stopped after " << lmIterations << " iterations (status = " << lmStatusToString(status)
             << ", gradient cosine = " << lm.gnorm << ")" << nl;

        if (lmDiagnosticsTriggered(lmStatusToString(status)))
        {
            runLMDiagnostics(lmFunctor, scaledState, lmStatusToString(status), folder);
        }

        currentCoords_ = scaledState.cwiseProduct(stateScaling_);
        currentState_ = fullState(scaledState);
        // Final assembly at the converged state; optionally writes the residual fields in folder/<time>
        assembleResidual(currentState_, true, romSettings_.writeResidualFields);

        volScalarField contErr(fvc::div(_phi()));
        scalar sumLocalContErr = runTime.deltaTValue() *
            mag(contErr)().weightedAverage(mesh.V()).value();
        scalar globalContErr = runTime.deltaTValue() *
            contErr.weightedAverage(mesh.V()).value();
        cumulativeContErr += globalContErr;

        Info << "  - U residual norm " << residualWorkspace_.blockNorm(iU_) << nl
             << "  - p residual norm " << residualWorkspace_.blockNorm(iP_) << nl
             << "  - k residual norm " << residualWorkspace_.blockNorm(iK_) << nl
             << "  - omega residual norm " << residualWorkspace_.blockNorm(iOmega_) << nl
             << "  - phi residual norm " << phiBlockNorm() << nl
             << "  - Initial residual norm: " << initialResidualNorm << nl
             << "  - Final residual norm: " << residualWorkspace_.vector().norm() << nl
             << "  - time step continuity errors : sum local = " << sumLocalContErr
             << "  |  global = " << globalContErr
             << "  |  cumulative = " << cumulativeContErr << nl
             << "  - Function evaluations: " << lm.nfev << endl;

        if (Pstream::master())
        {
            residualLog << runTime.value() << " " << initialResidualNorm << " "
                        << residualWorkspace_.vector().norm() << " "
                        << residualWorkspace_.blockNorm(iU_) << " "
                        << residualWorkspace_.blockNorm(iP_) << " "
                        << residualWorkspace_.blockNorm(iK_) << " "
                        << residualWorkspace_.blockNorm(iOmega_) << " "
                        << phiBlockNorm() << " "
                        << lmIterations << " " << lm.nfev << " " << lm.gnorm << " "
                        << lmStatusToString(status) << std::endl;
        }

        // oldState_ = currentState_;

        if (runTime.value() >= nextWrite - 0.5 * runTime.deltaTValue())
        {
            ITHACAstream::exportSolution(_U(), name(runTime.time().value()), folder);
            ITHACAstream::exportSolution(_p(), name(runTime.time().value()), folder);
            turbulence->saveFieldsToDisk(runTime, folder);
            nextWrite += writeInterval;
        }
    }
    residualOutputFolder_ = "";
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    Info << "Total computation time: " << duration.count() << " ms" << endl;
}

Eigen::VectorXd ReducedLSPGUnsteadyNSTurb::assembleResidual(
    const Eigen::VectorXd& state,
    const bool correctBCs,
    const bool writeResidual)
{
#include "createReferences.H"

    reconstructReducedFields(state, U, p, phi, correctBCs); // Carries out the linear combination
    turbulence->correctNut();
    // interpolateNutCoeffs(nut, Nutmodes_, state); // Nut coefficients are interpolated using an RBF from the velocity coefficients, then the field is reconstructed using the modes

    return residualWithCurrentFields(writeResidual);
}

PressureResidualCheck::Terms ReducedLSPGUnsteadyNSTurb::pressureResidualTerms()
{
#include "createReferences.H"
#include "UEqn.H"
    return PressureResidualCheck::compute(UEqn, U, p, phi, MRF, _pRefCell, _pRefValue);
}

Eigen::VectorXd ReducedLSPGUnsteadyNSTurb::residualWithCurrentFields(const bool writeResidual)
{
#include "createReferences.H"

    // interpolateNutCoeffs(nut, Nutmodes_, state); // Nut coefficients are interpolated using an RBF from the velocity coefficients, then the field is reconstructed using the modes
#include "UEqn.H"
#include "pEqn.H"

    const Field<scalar> Rp = pEqn.residual();

    // Face-flux residual (He et al. 2021, eq. 1.8): the reconstructed flux against the Rhie-Chow flux
    // of the current (U, p). Assembled only, from the same UEqn and pEqn
    if (iPhi_ >= 0)
    {
        const surfaceScalarField phiRC(phiHbyA - pEqn.flux());
        const Field<scalar> Rphi(phi.primitiveField() - phiRC.primitiveField());
        residualWorkspace_.set(iPhi_, Foam2Eigen::field2Eigen(Rphi));
        if (writeResidual)
        {
            surfaceScalarField res("residual_phi", (phi - phiRC) / mesh.magSf());
            exportResidual(res);
        }
    }

    UEqn += fvc::grad(p);
    const Field<vector> Ru = UEqn.residual();

    PtrList<fvScalarMatrix> turbEqs = turbulence->assembleEquations(fvOptions);
    const Field<scalar> Rk = turbEqs[0].residual();
    const Field<scalar> Romega = turbEqs[1].residual();

    if (writeResidual)
    {
        writeResidualField<vector>("residual_U", UEqn, Ru, mesh);
        writeResidualField<scalar>("residual_p", pEqn, Rp, mesh);
        writeResidualField<scalar>("residual_k", turbEqs[0], Rk, mesh);
        writeResidualField<scalar>("residual_omega", turbEqs[1], Romega, mesh);
    }

    if (romSettings_.residualPreconditioner == "jacobi" && updatePreconditioner_)
    {
        setJacobiPreconditioner(UEqn, pEqn, turbEqs);
        updatePreconditioner_ = false;
    }

    residualWorkspace_.set(iU_, Foam2Eigen::field2Eigen(Ru));
    residualWorkspace_.set(iP_, Foam2Eigen::field2Eigen(Rp));
    residualWorkspace_.set(iK_, Foam2Eigen::field2Eigen(Rk));
    residualWorkspace_.set(iOmega_, Foam2Eigen::field2Eigen(Romega));

    return residualWorkspace_.vector();
}

void ReducedLSPGUnsteadyNSTurb::setJacobiPreconditioner(
    const fvVectorMatrix& UEqn,
    const fvScalarMatrix& pEqn,
    const PtrList<fvScalarMatrix>& turbEqs)
{
    const scalarField& V = mesh().V().field();
    auto rowWeights = [&](const scalarField& A) -> Eigen::VectorXd {
        Eigen::VectorXd w(A.size());
        forAll(A, celli)
        {
            const scalar a = max(mag(A[celli]) * V[celli], VSMALL);
            w(celli) = Foam::sqrt(V[celli]) / a;
        }
        return w;
    };

    const Eigen::VectorXd wU = rowWeights(UEqn.A()().primitiveField());
    Eigen::VectorXd wUvec(3 * wU.size());
    for (label celli = 0; celli < wU.size(); celli++)
    {
        wUvec.segment(3 * celli, 3).setConstant(wU(celli));
    }
    residualWorkspace_.setBlockRowWeights(iU_, wUvec);
    residualWorkspace_.setBlockRowWeights(iP_, rowWeights(pEqn.A()().primitiveField()));
    residualWorkspace_.setBlockRowWeights(iK_, rowWeights(turbEqs[0].A()().primitiveField()));
    residualWorkspace_.setBlockRowWeights(iOmega_, rowWeights(turbEqs[1].A()().primitiveField()));
}

double ReducedLSPGUnsteadyNSTurb::phiBlockNorm() const
{
    return iPhi_ >= 0 ? residualWorkspace_.blockNorm(iPhi_) : 0.0;
}

void ReducedLSPGUnsteadyNSTurb::reconstructReducedFields(
    const Eigen::VectorXd& state,
    volVectorField& velocity_field,
    volScalarField& pressure_field,
    surfaceScalarField& phi_field,
    bool correctBCs)
{
    velocity_field = Umodes_.reconstruct(velocity_field, state.head(numberOfModes_.velocity), "U_r");
    phi_field = Phimodes_.reconstruct(phi_field, state.head(numberOfModes_.velocity), "phi_r");
    pressure_field = Pmodes_.reconstruct(pressure_field, state.segment(numberOfModes_.velocity, numberOfModes_.pressure), "p_r");
    turbulence->reconstructTurbulenceFields(state.tail(numberOfModes_.k + numberOfModes_.omega));

    if (romSettings_.bcMethod == "lift")
    {
        const Eigen::VectorXd& bcs = boundaryConditions_.getCurrentBCs();
        for (label i = 0; i < inletIndex_.rows(); ++i)
        {
            const label patchID = inletIndex_(i, 0);
            const label cmpt = inletIndex_(i, 1);
            vector velBC = { 0.0, 0.0, 0.0 };
            velBC[cmpt] = bcs(i);
            ITHACAutilities::assignBC(velocity_field, patchID, velBC);
            velocity_field += liftFields_[i] * bcs(i);
            phi_field += liftFieldsPhi_[i] * bcs(i);
        }
    } else if (romSettings_.snapshotCollectionMethod == 1) // Consistent method by carlberg, snapshots are centered to their initial condition for the offline trajectories
    {
        const Eigen::VectorXd& bcs = boundaryConditions_.getCurrentBCs();
        const Eigen::VectorXd weights = interpolateIDW(bcs);
        for (int k = 0; k < weights.size(); k++)
        {
            velocity_field += weights(k) * U0field_[k];
            phi_field += weights(k) * Phi0field_[k];
            pressure_field += weights(k) * P0field_[k];
        }
        turbulence->addInitialZeroField(weights);

        for (label i = 0; i < inletIndex_.rows(); ++i)
        {
            const label patchID = inletIndex_(i, 0);
            const label cmpt = inletIndex_(i, 1);
            vector velBC = { 0.0, 0.0, 0.0 };
            velBC[cmpt] = bcs(i);
            ITHACAutilities::assignBC(velocity_field, patchID, velBC);
            phi_field.boundaryFieldRef()[patchID] = velocity_field.boundaryField()[patchID] & velocity_field.mesh().Sf().boundaryField()[patchID];
        }
    }
    if (correctBCs)
    {
        pressure_field.correctBoundaryConditions();
    }

    turbulence->boundTurbulenceFields();
}

void ReducedLSPGUnsteadyNSTurb::initializePODCoeffsFromFields()
{
    volVectorField& U = _U();
    volScalarField& p = _p();
    surfaceScalarField& phi = _phi();

    if (romSettings_.bcMethod == "lift")
    {
        for (int i = 0; i < liftFields_.size(); i++)
        {
            U -= boundaryConditions_.getCurrentBCs()(i) * liftFields_[i];
            phi -= boundaryConditions_.getCurrentBCs()(i) * liftFieldsPhi_[i];
        }
    } else if (romSettings_.snapshotCollectionMethod == 1) // Snapshots obtained by subtracting the initial snapshot
    {
        const Eigen::VectorXd weights = interpolateIDW(boundaryConditions_.getCurrentBCs());
        for (int j = 0; j < weights.size(); j++)
        {
            U -= weights(j) * U0field_[j];
            phi -= weights(j) * Phi0field_[j];
            p -= weights(j) * P0field_[j];
        }
        turbulence->removeInitialZeroField(weights);
    }
    currentState_.head(numberOfModes_.velocity) = ITHACAutilities::getCoeffs(U, Umodes_);
    currentState_.segment(numberOfModes_.velocity, numberOfModes_.pressure) = ITHACAutilities::getCoeffs(p, Pmodes_);
    turbulence->computePODCoeffsFromFields(currentState_.tail(numberOfModes_.k + numberOfModes_.omega));

    if (stateMap_.size())
    {
        // The stacked modes are not orthonormal field by field: one weighted projection for the whole group
        // (k_r and omega_r are the fluctuations here, as the turbulence model removed their initial fields)
        List<Eigen::MatrixXd> fields(4);
        fields[0] = Foam2Eigen::field2Eigen(U);
        if (jointField_[1])
            fields[1] = Foam2Eigen::field2Eigen(p);
        if (jointField_[2])
            fields[2] = Foam2Eigen::field2Eigen(_k());
        if (jointField_[3])
            fields[3] = Foam2Eigen::field2Eigen(_omega());
        const Eigen::VectorXd a = jointCoefficients(fields).col(0);

        const labelList fieldModes({ numberOfModes_.velocity, numberOfModes_.pressure, numberOfModes_.k, numberOfModes_.omega });
        label offset = 0;
        forAll(fieldModes, f)
        {
            if (jointField_[f])
            {
                currentState_.segment(offset, fieldModes[f]) = a;
            }
            offset += fieldModes[f];
        }
    }
    currentCoords_ = reducedCoordinates(currentState_);
}

void ReducedLSPGUnsteadyNSTurb::resetToInitialCondition()
{
    Time& runTime = this->runTime();
    fvMesh& mesh = this->mesh();
    runTime.setTime(runTime.startTime().value(), runTime.startTimeIndex());

    auto resetFromDisk = [&](auto& field) {
        using FieldType = std::decay_t<decltype(field)>;
        const FieldType fieldOnDisk(
            IOobject(field.name(), runTime.timeName(), mesh,
                IOobject::MUST_READ, IOobject::NO_WRITE, IOobject::NO_REGISTER),
            mesh);
        // == forces the boundary values too
        field == fieldOnDisk;
        field.clearOldTimes();
    };
    resetFromDisk(_U());
    resetFromDisk(_p());
    resetFromDisk(_k());
    resetFromDisk(_omega());
    resetFromDisk(_nut());

    // phi_r is not read: as in createFields.H
    _phi() == (linearInterpolate(_U()) & mesh.Sf());
    _phi().clearOldTimes();
}

void ReducedLSPGUnsteadyNSTurb::buildStateMap()
{
    stateMap_.resize(0, 0);
    if (!jointField_[0])
    {
        return;
    }
    const labelList fieldModes({ numberOfModes_.velocity, numberOfModes_.pressure, numberOfModes_.k, numberOfModes_.omega });
    const label nJoint = numberOfModes_.velocity;
    label nState = 0;
    label nCoords = nJoint;
    forAll(fieldModes, f)
    {
        M_Assert(!jointField_[f] || fieldModes[f] == nJoint,
            "The fields of the stacked POD must have the same number of modes.");
        nState += fieldModes[f];
        nCoords += jointField_[f] ? 0 : fieldModes[f];
    }

    stateMap_ = Eigen::MatrixXd::Zero(nState, nCoords);
    label row = 0;
    label col = nJoint;
    forAll(fieldModes, f)
    {
        for (label i = 0; i < fieldModes[f]; i++)
        {
            stateMap_(row + i, jointField_[f] ? i : col + i) = 1.0;
        }
        row += fieldModes[f];
        col += jointField_[f] ? 0 : fieldModes[f];
    }
    Info << "### Stacked POD: " << nCoords << " reduced coordinates for a state of size " << nState << endl;
}

Eigen::VectorXd ReducedLSPGUnsteadyNSTurb::reducedCoordinates(const Eigen::VectorXd& state) const
{
    if (!stateMap_.size())
    {
        return state;
    }
    // E has 0/1 entries and orthogonal columns: (E^T E)^-1 E^T averages the copies of a shared coefficient
    const Eigen::VectorXd copies = stateMap_.colwise().sum().transpose();
    return (stateMap_.transpose() * state).cwiseQuotient(copies);
}

Eigen::MatrixXd ReducedLSPGUnsteadyNSTurb::jointCoefficients(const List<Eigen::MatrixXd>& snapshots)
{
    const label n = numberOfModes_.velocity;
    const label m = snapshots[0].cols();
    Eigen::MatrixXd G = Eigen::MatrixXd::Zero(n, n);
    Eigen::MatrixXd B = Eigen::MatrixXd::Zero(n, m);
    addJointProjection(Umodes_, snapshots[0], jointWeights_(0), G, B);
    if (jointField_[1])
        addJointProjection(Pmodes_, snapshots[1], jointWeights_(1), G, B);
    if (jointField_[2])
        addJointProjection(kModes_, snapshots[2], jointWeights_(2), G, B);
    if (jointField_[3])
        addJointProjection(omegaModes_, snapshots[3], jointWeights_(3), G, B);
    if (Pstream::parRun())
    {
        reduce(G, sumOp<Eigen::MatrixXd>());
        reduce(B, sumOp<Eigen::MatrixXd>());
    }
    return G.ldlt().solve(B);
}
// currentState_.segment(numberOfModes_.velocity + numberOfModes_.pressure, numberOfModes_.k) = ITHACAutilities::getCoeffs(k, kModes_);
// currentState_.tail(numberOfModes_.omega) = ITHACAutilities::getCoeffs(omega, omegaModes_);
// currentNutCoeffs_ = ITHACAutilities::getCoeffs(nutFields_[0], Nutmodes_);
// currentNutAvgCoeffs_ = interpolateIDW(boundaryConditions_.getCurrentBCs());
// }


void ReducedLSPGUnsteadyNSTurb::interpolateNutCoeffs(volScalarField& nut_field, volScalarModes& nut_modes, const Eigen::VectorXd& state)
{
    // Uses RBF to interpolate nut coefficients, then sums the Inverse Distance Weighted average fields
    const label inputRBFSize =
        interpolationSettings_.dimInputRBF > 0
        ? interpolationSettings_.dimInputRBF
        : numberOfModes_.velocity;

    const Eigen::VectorXd velocity_coeffs = state.head(inputRBFSize);
    if (interpolationSettings_.derivative)
    {
        const Eigen::VectorXd velocity_coeffs_derivative = (state - oldState_).head(inputRBFSize) / runTime().deltaTValue();
        Eigen::VectorXd inputVector = Eigen::VectorXd(inputRBFSize * 2);
        inputVector << velocity_coeffs, velocity_coeffs_derivative;
        for (int j = 0; j < nut_modes.size(); j++)
        {
            currentNutCoeffs_(j) = rbfSplines_[j]->predict(inputVector);
        }
    } else
    {
        for (int j = 0; j < nut_modes.size(); j++)
        {
            currentNutCoeffs_(j) = rbfSplines_[j]->predict(velocity_coeffs);
        }
    }
    nut_field = nut_modes.reconstruct(nut_field, currentNutCoeffs_, "nut_r");
    for (int k = 0; k < currentNutAvgCoeffs_.size(); k++)
    {
        nut_field += currentNutAvgCoeffs_(k) * avgNutFields_[k];
    }
    // Clip to ensure non-negative values
    nut_field.internalFieldRef() = Foam::max(nut_field.internalFieldRef(), nut_field.internalFieldRef() * 0.0);
    nut_field.correctBoundaryConditions();
}


Eigen::VectorXd ReducedLSPGUnsteadyNSTurb::interpolateIDW(const Eigen::VectorXd&
        input_parameters) const
{
    M_Assert(input_parameters.size() == mu_.rows(),
        "Input parameters size to the linear interpolation does not match the expected size.");
    const label n_samples = mu_.cols();
    Eigen::VectorXd weights(n_samples);

    // Parameters normalised by their range over the training runs, so that they weigh equally in the distance
    const Eigen::VectorXd lower = mu_.rowwise().minCoeff();
    const Eigen::VectorXd range = mu_.rowwise().maxCoeff() - lower;
    auto normalised = [&](const Eigen::VectorXd& x) {
        Eigen::VectorXd y = Eigen::VectorXd::Zero(x.size());
        for (label j = 0; j < x.size(); j++)
        {
            y(j) = range(j) > 1e-15 ? (x(j) - lower(j)) / range(j) : 0.0;
        }
        return y;
    };

    const Eigen::VectorXd target = normalised(input_parameters);
    Eigen::VectorXd distances(n_samples);
    for (label i = 0; i < n_samples; i++)
    {
        distances(i) = std::pow((target - normalised(mu_.col(i))).norm(), idwExponent_);
        if (distances(i) <= 1e-10)
        {
            // A training parameter
            weights.setZero();
            weights(i) = 1.0;
            return weights;
        }
    }

    // Only the idwNeighbours_ nearest runs (all the runs at the cut-off distance are kept)
    scalar cutOff = GREAT;
    if (idwNeighbours_ > 0 && idwNeighbours_ < n_samples)
    {
        std::vector<scalar> sorted(distances.data(), distances.data() + n_samples);
        std::sort(sorted.begin(), sorted.end());
        cutOff = sorted[idwNeighbours_ - 1];
    }
    for (label i = 0; i < n_samples; i++)
    {
        weights(i) = distances(i) <= cutOff ? 1.0 / distances(i) : 0.0;
    }

    const double weightSum = weights.sum();
    if (weightSum > 1e-10)
    {
        weights /= weightSum;
    } else
    {
        weights.setConstant(1.0 / n_samples);
    }

    return weights;
}


Eigen::VectorXd ReducedLSPGUnsteadyNSTurb::residualBlockWeights() const
{
    Eigen::VectorXd weights(5);
    weights << residualWorkspace_.blockWeight(iU_),
        residualWorkspace_.blockWeight(iP_),
        residualWorkspace_.blockWeight(iK_),
        residualWorkspace_.blockWeight(iOmega_),
        iPhi_ >= 0 ? residualWorkspace_.blockWeight(iPhi_) : 0.0;
    return weights;
}


Eigen::MatrixXd ReducedLSPGUnsteadyNSTurb::evaluateResidualOnTrajectory(
    const word& bcROM,
    const PtrList<volVectorField>& Ufom,
    const PtrList<surfaceScalarField>& phiFom,
    const PtrList<volScalarField>& pFom,
    const PtrList<volScalarField>& kFom,
    const PtrList<volScalarField>& omegaFom,
    const Eigen::VectorXd& times,
    const scalar deltaT,
    const word& mode,
    const fileName& pressureCheckFile,
    const PtrList<volScalarField>* nutFom)
{
    M_Assert(mode == "exact" || mode == "projected",
        "evaluateResidualOnTrajectory: mode must be 'exact' or 'projected'.");
    const label nSteps = Ufom.size();
    M_Assert(phiFom.size() == nSteps && pFom.size() == nSteps && kFom.size() == nSteps && omegaFom.size() == nSteps && times.size() == nSteps,
        "evaluateResidualOnTrajectory: all field lists and the time vector must have the same size.");
    M_Assert(nSteps >= 3, "evaluateResidualOnTrajectory: at least 3 consecutive states are needed.");

    Time& runTime = this->runTime();
    volVectorField& U = _U();
    volScalarField& p = _p();
    surfaceScalarField& phi = _phi();
    volScalarField& k = _k();
    volScalarField& omega = _omega();

    const scalar savedTime = runTime.value();
    const label savedIndex = runTime.timeIndex();
    const scalar savedDeltaT = runTime.deltaTValue();

    boundaryConditions_ = BoundaryConditions { bcROM, "linear" };

    // Two increments ending at times(0), so that deltaT0 = deltaT for the backward scheme
    runTime.setDeltaT(deltaT, false);
    runTime.setTime(times(0) - 2 * deltaT, savedIndex + 1);
    runTime++;
    runTime++;

    Eigen::MatrixXd out(nSteps - 2, 7);

    // Pressure residual diagnostics, exact mode only
    std::ofstream pCheck;
    if (mode == "exact" && !pressureCheckFile.empty() && Pstream::master())
    {
        pCheck.open(pressureCheckFile);
        pCheck << std::scientific << std::setprecision(8) << PressureResidualCheck::header(mesh())
               << " relNut relPhiOld relUOld relUOldOld" << std::endl;
    }

    // Every block of the weighted residual split by region, and its largest cell (location of the inconsistency)
    wordList regionNames;
    List<boolList> regionMasks;
    PressureResidualCheck::regions(mesh(), regionNames, regionMasks);
    const wordList blockNames({ "U", "p", "k", "omega" });
    const labelList blockIds({ iU_, iP_, iK_, iOmega_ });
    std::ofstream blockCheck;
    if (pCheck.is_open())
    {
        blockCheck.open(pressureCheckFile.path() / "blockRegions_disk.dat");
        blockCheck << "# weighted residual (as minimised by LM) per block and region; max = largest cell: value, x, y\n# time";
        for (const word& b : blockNames)
        {
            for (const word& r : regionNames)
            {
                blockCheck << " " << b << "_" << r;
            }
            blockCheck << " " << b << "_maxValue " << b << "_maxX " << b << "_maxY";
        }
        // Cells whose FOM value is the neighbour average Foam::bound assigns to negative cells after the solve
        blockCheck << " kBoundedCells omegaBoundedCells kMaxCellBounded";
        blockCheck << std::endl;
    }
    for (label n = 0; n < nSteps; n++)
    {
        if (n > 0)
        {
            runTime.setTime(times(n), runTime.timeIndex() + 1);
        }
        boundaryConditions_.updateTimeDependentBC(times(n));

        // The first modification at a new time index stores the previous state as oldTime,
        // so the old time levels are the previous (exact or projected) FOM states
        U == Ufom[n];
        p == pFom[n];
        phi == phiFom[n];
        k == kFom[n];
        omega == omegaFom[n];

        updatePreconditioner_ = true;
        if (mode == "projected")
        {
            initializePODCoeffsFromFields();
            assembleResidual(currentState_, true, false);
        } else
        {
            turbulence->correctNut();
            residualWithCurrentFields();
        }

        if (pCheck.is_open() && n >= 2)
        {
            // nut recomputed by the ROM from the FOM (U, k, omega) against the FOM nut, and the old time
            // levels used by ddt and ddtCorr against the previous FOM states
            const scalarField& V = mesh().V().field();
            auto relV = [&](const auto& a, const auto& b) {
                return PressureResidualCheck::volumeNorm(a - b, V) / max(PressureResidualCheck::volumeNorm(b, V), VSMALL);
            };
            const scalar relNut = nutFom ? relV(_nut().primitiveField(), (*nutFom)[n].primitiveField()) : -1;
            const scalarField& phiOld = phi.oldTime().primitiveField();
            const scalarField& phiPrev = phiFom[n - 1].primitiveField();
            const scalar relPhiOld = Foam::sqrt(sum(sqr(phiOld - phiPrev)) / max(sum(sqr(phiPrev)), VSMALL));
            const scalar relUOld = relV(U.oldTime().primitiveField(), Ufom[n - 1].primitiveField());
            const scalar relUOldOld = relV(U.oldTime().oldTime().primitiveField(), Ufom[n - 2].primitiveField());

            const PressureResidualCheck::Terms terms = pressureResidualTerms();
            PressureResidualCheck::writeRow(pCheck, times(n), terms, mesh(), U.primitiveField());
            pCheck << " " << relNut << " " << relPhiOld << " " << relUOld << " " << relUOldOld << std::endl;

            blockCheck << std::scientific << std::setprecision(8) << times(n);
            const vectorField& C = mesh().C().primitiveField();
            forAll(blockIds, b)
            {
                const Eigen::Ref<const Eigen::VectorXd> r = residualWorkspace_.block(blockIds[b]);
                const label nPerCell = r.size() / mesh().nCells();
                scalarField cellSqr(mesh().nCells(), 0.0);
                forAll(cellSqr, celli)
                {
                    cellSqr[celli] = r.segment(nPerCell * celli, nPerCell).squaredNorm();
                }
                for (const boolList& mask : regionMasks)
                {
                    scalar s2 = 0;
                    forAll(cellSqr, celli)
                    {
                        s2 += mask[celli] ? cellSqr[celli] : 0;
                    }
                    blockCheck << " " << Foam::sqrt(s2);
                }
                const label imax = findMax(cellSqr);
                blockCheck << " " << Foam::sqrt(cellSqr[imax]) << " " << C[imax].x() << " " << C[imax].y();
            }

            // Foam::bound sets a negative cell to fvc::average(max(f, fMin)) evaluated with the cell itself at fMin
            // (kOmegaSST: kMin = omegaMin = SMALL): k_P = sum_f |Sf| (w_P fMin + (1 - w_P) f_N) / sum_f |Sf|,
            // with w_P the linear-interpolation weight of P. Internal faces only (cells next to a patch are skipped)
            auto boundedCells = [this](const volScalarField& f) {
                const fvMesh& m = mesh();
                const labelUList& own = m.owner();
                const labelUList& nei = m.neighbour();
                const scalarField& w = m.weights().primitiveField();
                const scalarField& magSf = m.magSf().primitiveField();
                scalarField sumA(m.nCells(), 0.0), sumNb(m.nCells(), 0.0), sumW(m.nCells(), 0.0);
                forAll(own, facei)
                {
                    const label P = own[facei], N = nei[facei];
                    sumA[P] += magSf[facei];
                    sumA[N] += magSf[facei];
                    sumNb[P] += magSf[facei] * (1 - w[facei]) * f[N];
                    sumNb[N] += magSf[facei] * w[facei] * f[P];
                    sumW[P] += magSf[facei] * w[facei];
                    sumW[N] += magSf[facei] * (1 - w[facei]);
                }
                boolList onPatch(m.nCells(), false);
                forAll(m.boundary(), patchi)
                {
                    for (const label celli : m.boundary()[patchi].faceCells())
                    {
                        onPatch[celli] = true;
                    }
                }
                boolList mask(f.size(), false);
                forAll(mask, celli)
                {
                    if (!onPatch[celli] && sumA[celli] > 0)
                    {
                        const scalar avg = (sumNb[celli] + sumW[celli] * SMALL) / sumA[celli];
                        mask[celli] = mag(f[celli] - avg) <= 1e-9 * max(mag(avg), VSMALL);
                    }
                }
                return mask;
            };
            const boolList kBounded = boundedCells(kFom[n]);
            const boolList omegaBounded = boundedCells(omegaFom[n]);
            label kMaxCell = 0;
            {
                const Eigen::Ref<const Eigen::VectorXd> r = residualWorkspace_.block(iK_);
                r.cwiseAbs().maxCoeff(&kMaxCell);
            }
            blockCheck << " " << std::count(kBounded.begin(), kBounded.end(), true)
                       << " " << std::count(omegaBounded.begin(), omegaBounded.end(), true)
                       << " " << kBounded[kMaxCell];
            blockCheck << std::endl;
        }

        if (n >= 2)
        {
            out(n - 2, 0) = times(n);
            out(n - 2, 1) = residualWorkspace_.vector().norm();
            out(n - 2, 2) = residualWorkspace_.blockNorm(iU_);
            out(n - 2, 3) = residualWorkspace_.blockNorm(iP_);
            out(n - 2, 4) = residualWorkspace_.blockNorm(iK_);
            out(n - 2, 5) = residualWorkspace_.blockNorm(iOmega_);
            out(n - 2, 6) = phiBlockNorm();
        }
    }

    runTime.setDeltaT(savedDeltaT, false);
    runTime.setTime(savedTime, savedIndex);
    return out;
}
