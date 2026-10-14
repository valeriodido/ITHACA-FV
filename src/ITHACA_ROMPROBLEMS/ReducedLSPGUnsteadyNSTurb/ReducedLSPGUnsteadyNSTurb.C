#include "ReducedLSPGUnsteadyNSTurb.H"
#include "Foam2Eigen.H" // Needed to convert OpenFOAM fields to Eigen vectors and matrices
#include "ITHACAassign.H"


ReducedLSPGUnsteadyNSTurb::ReducedLSPGUnsteadyNSTurb(
    std::shared_ptr<ITHACAcontext> context,
    const LSPGUnsteadyNSTurb& problem): ReducedLSPG(std::move(context))
{
    captureFOMData(problem);

    Time& runTime = this->runTime();
    fvMesh& mesh = this->mesh();
    pimpleControl& pimple = this->pimple();

#include "createFields.H"

    currentState_ = Eigen::VectorXd::Zero(
        numberOfModes_.velocity +
        numberOfModes_.pressure);
    currentNutCoeffs_ = Eigen::VectorXd::Zero(numberOfModes_.nut);
    if (Pstream::master())
    {
        readEigenvalues();
    }
    
    residualWorkspace_ = BlockResidual {
      { { "U_r", 3 * nCells, 1.0 },
        { "p_r", nCells, 1.0 },
      }
    };
    
    iU_ = residualWorkspace_.index("U_r");
    iP_ = residualWorkspace_.index("p_r");

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
        problem.ITHACAdict->lookupOrDefault<label>("NmodesNutproj", 0)
    };
    M_Assert(numberOfModes_.velocity > 0, "Number of velocity modes must be greater than zero.");
    M_Assert(numberOfModes_.pressure > 0, "Number of pressure modes must be greater than zero.");
    M_Assert(numberOfModes_.nut > 0, "Number of nut modes must be greater than zero.");

    mu_ = problem.mu;
    rbfSplines_ = problem.rbfSplines;

    // I don't think here we are actually safe from deleting the FOM, check later
    Umodes_ = problem.Umodes;
    M_Assert(Umodes_.size() == numberOfModes_.velocity, "Number of velocity modes does not match the size of the velocity modes matrix.");
    Phimodes_ = problem.Phimodes;
    M_Assert(Phimodes_.size() == numberOfModes_.velocity, "Number of velocity modes does not match the size of the face flux modes matrix.");
    Pmodes_ = problem.Pmodes;
    M_Assert(Pmodes_.size() == numberOfModes_.pressure, "Number of pressure modes does not match the size of the pressure modes matrix.");
    Nutmodes_ = problem.nutmodes;
    M_Assert(Nutmodes_.size() == numberOfModes_.nut, "Number of nut modes does not match the size of the nut modes matrix.");

    liftFields_ = problem.liftfield;
    liftFieldsPhi_ = problem.liftfieldphi;
    // PtrLists containing the initial condition of each offline trajectory.
    U0field_ = problem.U0field;
    Phi0field_ = problem.Phi0field;
    P0field_ = problem.P0field;
    Nut0field_ = problem.Nut0field;

    nutFields_ = problem.fluctNutfield;
    avgNutFields_ = problem.avgNutfield;

    inletIndex_ = problem.inletIndex;
}

void ReducedLSPGUnsteadyNSTurb::solveOnline(
    const word& bcROM,
    const word& folder)
{
    if (Pstream::master())
    {
        mkDir(folder);
        ITHACAutilities::createSymLink(folder);
    }
    boundaryConditions_ = BoundaryConditions { bcROM, "linear" };
    const GaussNewtonSettings gn_settings = GaussNewtonSettings { 5, 1e-3, 1e-7 };
    Time& runTime = this->runTime();
#include "readTimeControls.H"

    initializePODCoeffsFromFields(); // Here we cannot correct the BCs for pRgh
    assembleResidual(currentState_, false); // Here we cannot correct the BCs for pRgh

    Info << "Initial residual norm: " << residualWorkspace_.vector().norm() << nl
         << "  - U residual norm " << residualWorkspace_.blockNorm(iU_) << nl
         << "  - p residual norm " << residualWorkspace_.blockNorm(iP_) << endl;
    residualWorkspace_.calibrateWeights(1e-8);

    LevenbergMarquardtFunctor lmFunctor(this, currentState_.size(), residualWorkspace_.size());
    Eigen::NumericalDiff<LevenbergMarquardtFunctor> numDiff(lmFunctor);
    Eigen::LevenbergMarquardt<Eigen::NumericalDiff<LevenbergMarquardtFunctor>> lm(numDiff);

    lm.parameters.ftol = 1e-8;
    lm.parameters.xtol = 1e-8;
    lm.parameters.gtol = 1e-10;
    lm.parameters.maxfev = 800;

    auto start = std::chrono::high_resolution_clock::now();
    while (runTime.run())
    {
        runTime++;
        Info << "Time = " << runTime.time().value() << nl;
        boundaryConditions_.updateTimeDependentBC(runTime.time().value());

        Eigen::VectorXd scaledState = currentState_.cwiseQuotient(stateScaling_);
        Eigen::LevenbergMarquardtSpace::Status status = lm.minimizeInit(scaledState);
        const double initialResidualNorm = residualWorkspace_.vector().norm();
        for (int gnIter = 0; gnIter < gn_settings.maxIter; gnIter++)
        {
            const double residualNorm = lm.fnorm;

            if (residualNorm < gn_settings.tol)
            {
                Info << "LM converged at iteration " << gnIter << nl;
                break;
            }

            status = lm.minimizeOneStep(scaledState);

            if (status != Eigen::LevenbergMarquardtSpace::Running)
            {
                Info << "LM stopped internally at iteration " << gnIter
                     << " (status = " << lmStatusToString(status) << ")" << nl;
                break;
            }
        }

        currentState_ = scaledState.cwiseProduct(stateScaling_);
        assembleResidual(currentState_);
        Info << "  - U residual norm " << residualWorkspace_.blockNorm(iU_) << nl
             << "  - p residual norm " << residualWorkspace_.blockNorm(iP_) << nl
             << "  - Initial residual norm: " << initialResidualNorm << nl
             << "  - Final residual norm: " << lm.fnorm << endl;

        if (runTime.writeTime())
        {
            ITHACAstream::exportSolution(_U(), name(runTime.time().value()), folder);
            ITHACAstream::exportSolution(_p(), name(runTime.time().value()), folder);
            ITHACAstream::exportSolution(_nut(), name(runTime.time().value()), folder);
        }
    }
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
    interpolateNutCoeffs(nut, Nutmodes_, state); // Nut coefficients are interpolated using an RBF from the velocity coefficients, then the field is reconstructed using the modes
#include "UEqn.H"
#include "pEqn.H"

    const Field<vector> Ru = UEqn.residual();
    const Field<scalar> Rp = pEqn.residual();

    if (writeResidual)
    {
        writeResidualField<vector>("residual_U", UEqn, Ru, mesh);
        writeResidualField<scalar>("residual_p", pEqn, Rp, mesh);
    }

    residualWorkspace_.set(iU_, Foam2Eigen::field2Eigen(Ru));
    residualWorkspace_.set(iP_, Foam2Eigen::field2Eigen(Rp));
    return residualWorkspace_.vector();
}

void ReducedLSPGUnsteadyNSTurb::assembleJacobian(const Eigen::VectorXd& state,
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

Eigen::VectorXd ReducedLSPGUnsteadyNSTurb::residualWithCurrentFields()
{
#include "createReferences.H"

#include "UEqn.H"
#include "pEqn.H"

    const Field<vector> Ru = UEqn.residual();
    const Field<scalar> Rp = pEqn.residual();

    residualWorkspace_.set(iU_, Foam2Eigen::field2Eigen(Ru));
    residualWorkspace_.set(iP_, Foam2Eigen::field2Eigen(Rp));

    return residualWorkspace_.vector();
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
    if (correctBCs) // On the first iteration, calling correctBoundaryConditions() raises the --> FOAM FATAL ERROR: updateCoeffs(const scalarField& snGradp) MUST be called ...
    {
        pressure_field.correctBoundaryConditions();
    }
}


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
        FatalErrorInFunction << "Derivative interpolation for nut coefficients is not implemented yet." << exit(FatalError);
    } else
    {
        for (int j = 0; j < numberOfModes_.nut; j++)
        {
            currentNutCoeffs_(j) = rbfSplines_[j]->predict(velocity_coeffs);
        }
    }
    nut_field = nut_modes.reconstruct(nut_field, currentNutCoeffs_, "nut_r");
    for (int k = 0; k < currentNutAvgCoeffs_.size(); k++)
    {
        nut_field += currentNutAvgCoeffs_(k) * avgNutFields_[k];
    }
}


Eigen::VectorXd ReducedLSPGUnsteadyNSTurb::interpolateIDW(const Eigen::VectorXd&
        input_parameters)
{
    M_Assert(input_parameters.size() == mu_.rows(),
        "Input parameters size to the linear interpolation does not match the expected size.");
    const label n_samples = mu_.cols();
    Eigen::VectorXd weights(n_samples);

    for (label i = 0; i < n_samples; i++)
    {
        weights(i) = 1.0 / ((input_parameters - mu_.col(i)).norm() + 1e-10);
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
        }
    } else if (romSettings_.snapshotCollectionMethod == 1) // Snapshots obtained by subtracting the initial snapshot
    {
        const Eigen::VectorXd weights = interpolateIDW(boundaryConditions_.getCurrentBCs());
        for (int k = 0; k < weights.size(); k++)
        {
            Info << "Weight of the trajectory " << k << " is " << weights(k) << endl;
            U -= weights(k) * U0field_[k];
            phi -= weights(k) * Phi0field_[k];
            p -= weights(k) * P0field_[k];
        }
    }
    Info << "Initializing the POD coefficients from the current fields in memory" << endl;
    currentState_.head(numberOfModes_.velocity) = ITHACAutilities::getCoeffs(U, Umodes_);
    currentState_.tail(numberOfModes_.pressure) = ITHACAutilities::getCoeffs(p, Pmodes_);
    currentNutCoeffs_ = ITHACAutilities::getCoeffs(nutFields_[0], Nutmodes_);
    currentNutAvgCoeffs_ = interpolateIDW(boundaryConditions_.getCurrentBCs());
    Info << "Initialized the POD coefficients" << endl;
}


void ReducedLSPGUnsteadyNSTurb::readEigenvalues()
{
    Eigen::VectorXd uEigenvalues_, pEigenvalues_, nutEigenvalues_;
    uEigenvalues_ = Eigen::VectorXd::Zero(numberOfModes_.velocity);
    pEigenvalues_ = Eigen::VectorXd::Zero(numberOfModes_.pressure);
    nutEigenvalues_ = Eigen::VectorXd::Zero(numberOfModes_.nut);
    std::ifstream uFile("ITHACAoutput/POD/Eigenvalues_U");
    M_Assert(uFile.is_open(),
        "Could not open file ITHACAoutput/POD/Eigenvalues_U. Please make sure the file exists and is readable.");
    std::string line;
    std::getline(uFile, line); // Ignore first line
    std::getline(uFile, line); // Ignore second line

    for (int i = 0; i < numberOfModes_.velocity; i++)
    {
        std::getline(uFile, line);
        uEigenvalues_(i) = std::stod(line);
    }

    std::ifstream pFile("ITHACAoutput/POD/Eigenvalues_p");
    M_Assert(pFile.is_open(),
        "Could not open file ITHACAoutput/POD/Eigenvalues_p. Please make sure the file exists and is readable.");
    std::getline(pFile, line); // Ignore first line
    std::getline(pFile, line); // Ignore second line

    for (int i = 0; i < numberOfModes_.pressure; i++)
    {
        std::getline(pFile, line);
        pEigenvalues_(i) = std::stod(line);
    }

    Info << "### EIGS - Velocity eigenvalues: " << uEigenvalues_.transpose() << nl
         << "### EIGS - Pressure eigenvalues: " << pEigenvalues_.transpose() << endl;

    eigenvalues_ = Eigen::VectorXd::Zero(
        numberOfModes_.velocity + numberOfModes_.pressure);
    stateScaling_ = Eigen::VectorXd::Zero(
        numberOfModes_.velocity + numberOfModes_.pressure);
    eigenvalues_ << uEigenvalues_, pEigenvalues_;

    stateScaling_ = eigenvalues_.array().sqrt();

    M_Assert(stateScaling_.size() == currentState_.size(),
        "State scaling size does not match current state size.");
    M_Assert(stateScaling_.minCoeff() > 1e-10,
        "State scaling has a zero or negative value, which is not allowed.");
}
