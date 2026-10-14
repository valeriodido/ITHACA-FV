#include "ReducedLSPGUnsteadyBBTurb.H"
#include "Foam2Eigen.H" // Needed to convert OpenFOAM fields to Eigen vectors and matrices
#include "ITHACAassign.H"


ReducedLSPGUnsteadyBBTurb::ReducedLSPGUnsteadyBBTurb(
    std::shared_ptr<ITHACAcontext> context,
    const LSPGUnsteadyBBTurb& problem): ReducedLSPG(std::move(context))
{
    captureFOMData(problem);

    Time& runTime = this->runTime();
    fvMesh& mesh = this->mesh();
    pimpleControl& pimple = this->pimple();

#include "createFields.H"

    currentState_ = Eigen::VectorXd::Zero(
        numberOfModes_.velocity +
        numberOfModes_.pressure +
        numberOfModes_.temperature);
    currentNutCoeffs_ = Eigen::VectorXd::Zero(numberOfModes_.nut);
    if (Pstream::master())
    {
        readEigenvalues();
    }
    
    residualWorkspace_ = BlockResidual {
      { { "U_r", 3 * nCells, 1.0 },
            { "p_rgh_r", nCells, 1.0 },
            { "T_r", nCells, 1.0 } }
    };
    
    iU_ = residualWorkspace_.index("U_r");
    iP_ = residualWorkspace_.index("p_rgh_r");
    iT_ = residualWorkspace_.index("T_r");

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
      residualWorkspace_.setBlockRowWeights(iT_, wScalar);
    }
}

void ReducedLSPGUnsteadyBBTurb::captureFOMData(
    const LSPGUnsteadyBBTurb& problem)
{
    interpolationSettings_ = InterpolationSettings {
        problem.ITHACAdict->lookupOrDefault<int>("firstRBFIndex", 0),
        problem.ITHACAdict->lookupOrDefault<bool>("derivativeInRBF", false),
        problem.ITHACAdict->lookupOrDefault<label>("dimInputRBF", 0),
    };

    numberOfModes_ = NumberOfModes {
        problem.ITHACAdict->lookupOrDefault<label>("NmodesUproj", 0),
        problem.ITHACAdict->lookupOrDefault<label>("NmodesPrghproj", 0),
        problem.ITHACAdict->lookupOrDefault<label>("NmodesTproj", 0),
        problem.ITHACAdict->lookupOrDefault<label>("NmodesNutproj", 0)
    };

    mu_ = problem.mu;
    rbfSplines_ = problem.rbfSplines;

    // I don't think here we are actually safe from deleting the FOM, check later
    Umodes_ = problem.Umodes;
    Phimodes_ = problem.Phimodes;
    Prghmodes_ = problem.Prghmodes;
    Tmodes_ = problem.Tmodes;
    Nutmodes_ = problem.nutmodes;

    liftFields_ = problem.liftfield;
    liftFieldsT_ = problem.liftfieldT;
    liftFieldsPhi_ = problem.liftfieldphi;
    // PtrLists containing the initial condition of each offline trajectory.
    U0field_ = problem.U0field;
    Phi0field_ = problem.Phi0field;
    Prgh0field_ = problem.Prgh0field;
    T0field_ = problem.T0field;
    Nut0field_ = problem.Nut0field;

    nutFields_ = problem.fluctNutfield;
    avgNutFields_ = problem.avgNutfield;

    inletIndex_ = problem.inletIndex;
    inletIndexT_ = problem.inletIndexT;
}

void ReducedLSPGUnsteadyBBTurb::solveOnline(
    const word& bcROM,
    const word& bcROMT,
    const word& folder)
{
    if (Pstream::master())
    {
        mkDir(folder);
        ITHACAutilities::createSymLink(folder);
    }
    boundaryConditionsU_ = BoundaryConditions { bcROM, "linear" };
    boundaryConditionsT_ = BoundaryConditions { bcROMT, "linear" };
    const GaussNewtonSettings gn_settings = GaussNewtonSettings { 5, 1e-3, 1e-7 };
    Time& runTime = this->runTime();
#include "readTimeControls.H"

    initializePODCoeffsFromFields(); // Here we cannot correct the BCs for pRgh
    assembleResidual(currentState_, false); // Here we cannot correct the BCs for pRgh

    Info << "Initial residual norm: " << residualWorkspace_.vector().norm() << nl
         << "  - U residual norm " << residualWorkspace_.blockNorm(iU_) << nl
         << "  - p_rgh residual norm " << residualWorkspace_.blockNorm(iP_) << nl
         << "  - T residual norm " << residualWorkspace_.blockNorm(iT_) << endl;
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
        Info << "Time = " << runTime.time().value() << nl << endl;
        boundaryConditionsU_.updateTimeDependentBC(runTime.time().value());
        boundaryConditionsT_.updateTimeDependentBC(runTime.time().value());

        Eigen::VectorXd scaledState = currentState_.cwiseQuotient(stateScaling_);
        Eigen::LevenbergMarquardtSpace::Status status = lm.minimizeInit(scaledState);

        for (int gnIter = 0; gnIter < gn_settings.maxIter; gnIter++)
        {
            const double residualNorm = lm.fnorm;

            if (residualNorm < gn_settings.tol)
            {
                Info << "LM converged at iteration " << gnIter << endl;
                break;
            }

            status = lm.minimizeOneStep(scaledState);

            if (status != Eigen::LevenbergMarquardtSpace::Running)
            {
                Info << "LM stopped internally at iteration " << gnIter
                     << " (status = " << lmStatusToString(status) << ")" << endl;
                break;
            }
        }

        currentState_ = scaledState.cwiseProduct(stateScaling_);
        assembleResidual(currentState_);
        Info << "  - U residual norm " << residualWorkspace_.blockNorm(iU_) << nl
             << "  - p_rgh residual norm " << residualWorkspace_.blockNorm(iP_) << nl
             << "  - T residual norm " << residualWorkspace_.blockNorm(iT_) << endl;

        if (runTime.writeTime())
        {
            ITHACAstream::exportSolution(_U(), name(runTime.time().value()), folder);
            ITHACAstream::exportSolution(_p_rgh(), name(runTime.time().value()), folder);
            ITHACAstream::exportSolution(_T(), name(runTime.time().value()), folder);
            ITHACAstream::exportSolution(_nut(), name(runTime.time().value()), folder);
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    Info << "Total computation time: " << duration.count() << " ms" << endl;
}

Eigen::VectorXd ReducedLSPGUnsteadyBBTurb::assembleResidual(
    const Eigen::VectorXd& state,
    const bool correctBCs,
    const bool writeResidual)
{
#include "createReferences.H"

    reconstructReducedFields(state, U, p_rgh, T, phi, correctBCs); // Carries out the linear combination
    interpolateNutCoeffs(nut, Nutmodes_, state); // Nut coefficients are interpolated using an RBF from the velocity coefficients, then the field is reconstructed using the modes
#include "UEqn.H"
#include "TEqn.H"
#include "pEqn.H"

    volVectorField Usrc = -fvc::reconstruct(
        (
            ghf * fvc::snGrad(rhok) + fvc::snGrad(p_rgh)) *
        mesh.magSf());
    UEqn -= Usrc;

    const Field<vector> Ru = UEqn.residual();
    const Field<scalar> RT = TEqn.residual();
    const Field<scalar> Rp = p_rghEqn.residual();

    if (writeResidual)
    {
        writeResidualField<vector>("residual_U", UEqn, Ru, mesh);
        writeResidualField<scalar>("residual_p_rgh", p_rghEqn, Rp, mesh);
        writeResidualField<scalar>("residual_T", TEqn, RT, mesh);
    }

    residualWorkspace_.set(iU_, Foam2Eigen::field2Eigen(Ru));
    residualWorkspace_.set(iP_, Foam2Eigen::field2Eigen(Rp));
    residualWorkspace_.set(iT_, Foam2Eigen::field2Eigen(RT));
    return residualWorkspace_.vector();
}

void ReducedLSPGUnsteadyBBTurb::assembleJacobian(const Eigen::VectorXd& state,
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

Eigen::VectorXd ReducedLSPGUnsteadyBBTurb::residualWithCurrentFields()
{
#include "createReferences.H"

#include "UEqn.H"
#include "TEqn.H"
#include "pEqn.H"

    volVectorField Usrc = -fvc::reconstruct(
        (
            ghf * fvc::snGrad(rhok) + fvc::snGrad(p_rgh)) *
        mesh.magSf());
    UEqn -= Usrc;

    const Field<vector> Ru = UEqn.residual();
    const Field<scalar> RT = TEqn.residual();
    const Field<scalar> Rp = p_rghEqn.residual();

    residualWorkspace_.set(iU_, Foam2Eigen::field2Eigen(Ru));
    residualWorkspace_.set(iP_, Foam2Eigen::field2Eigen(Rp));
    residualWorkspace_.set(iT_, Foam2Eigen::field2Eigen(RT));

    return residualWorkspace_.vector();
}


void ReducedLSPGUnsteadyBBTurb::reconstructReducedFields(
    const Eigen::VectorXd& state,
    volVectorField& velocity_field,
    volScalarField& pressure_field,
    volScalarField& temperature_field,
    surfaceScalarField& phi_field,
    bool correctBCs)
{
    velocity_field = Umodes_.reconstruct(velocity_field, state.head(numberOfModes_.velocity), "U_r");
    phi_field = Phimodes_.reconstruct(phi_field, state.head(numberOfModes_.velocity), "phi_r");
    temperature_field = Tmodes_.reconstruct(temperature_field, state.tail(numberOfModes_.temperature), "T_r");
    pressure_field = Prghmodes_.reconstruct(pressure_field, state.segment(numberOfModes_.velocity, numberOfModes_.pressure), "p_rgh_r", false);
    
    const Eigen::VectorXd& bcsU = boundaryConditionsU_.getCurrentBCs();
    const Eigen::VectorXd& bcsT = boundaryConditionsT_.getCurrentBCs();
    Eigen::VectorXd aggregatedBCs = Eigen::VectorXd(bcsU.size() + bcsT.size());
    aggregatedBCs << bcsU, bcsT;
    
    if (romSettings_.bcMethod == "lift")
    {
        for (label i = 0; i < inletIndex_.rows(); ++i)
        {
            const label patchID = inletIndex_(i, 0);
            const label cmpt = inletIndex_(i, 1);
            vector velBC = { 0.0, 0.0, 0.0 };
            velBC[cmpt] = bcsU(i);
            ITHACAutilities::assignBC(velocity_field, patchID, velBC);
            velocity_field += liftFields_[i] * bcsU(i);
            phi_field += liftFieldsPhi_[i] * bcsU(i);
        }
        for (label i = 0; i < inletIndexT_.rows(); ++i)
        {
            const label patchID = inletIndexT_(i, 0);
            ITHACAutilities::assignBC(temperature_field, patchID, bcsT(i));
            temperature_field += liftFieldsT_[i] * bcsT(i);
        }
    } else if (romSettings_.snapshotCollectionMethod == 1) // Consistent method by carlberg, snapshots are centered to their initial condition for the offline trajectories
    {
        const Eigen::VectorXd weights = interpolateIDW(aggregatedBCs);
        for (int k = 0; k < weights.size(); k++)
        {
            velocity_field += weights(k) * U0field_[k];
            phi_field += weights(k) * Phi0field_[k];
            pressure_field += weights(k) * Prgh0field_[k];
            temperature_field += weights(k) * T0field_[k];
        }
        for (label i = 0; i < inletIndex_.rows(); ++i)
        {
            const label patchID = inletIndex_(i, 0);
            const label cmpt = inletIndex_(i, 1);
            vector velBC = { 0.0, 0.0, 0.0 };
            velBC[cmpt] = bcsU(i);
            ITHACAutilities::assignBC(velocity_field, patchID, velBC);
            phi_field.boundaryFieldRef()[patchID] = velocity_field.boundaryField()[patchID] & velocity_field.mesh().Sf().boundaryField()[patchID];
        }
        for (label i = 0; i < inletIndexT_.rows(); ++i)
        {
            const label patchID = inletIndexT_(i, 0);
            ITHACAutilities::assignBC(temperature_field, patchID, bcsT(i));
        }
    }
    if (correctBCs) // On the first iteration, calling correctBoundaryConditions() raises the --> FOAM FATAL ERROR: updateCoeffs(const scalarField& snGradp) MUST be called ...
    {
        pressure_field.correctBoundaryConditions();
    }
}


void ReducedLSPGUnsteadyBBTurb::interpolateNutCoeffs(volScalarField& nut_field, volScalarModes& nut_modes, const Eigen::VectorXd& state)
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


Eigen::VectorXd ReducedLSPGUnsteadyBBTurb::interpolateIDW(const Eigen::VectorXd&
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

void ReducedLSPGUnsteadyBBTurb::initializePODCoeffsFromFields()
{
    volVectorField& U = _U();
    volScalarField& p_rgh = _p_rgh();
    volScalarField& T = _T();
    surfaceScalarField& phi = _phi();
    if (romSettings_.bcMethod == "lift")
    {
        for (int i = 0; i < liftFields_.size(); i++)
        {
            U -= boundaryConditionsU_.getCurrentBCs()(i) * liftFields_[i];
        }
        for (int i = 0; i < liftFieldsT_.size(); i++)
        {
            T -= boundaryConditionsT_.getCurrentBCs()(i) * liftFieldsT_[i];
        }
    } else if (romSettings_.snapshotCollectionMethod == 1) // Snapshots obtained by subtracting the initial snapshot
    {
        const Eigen::VectorXd weights = interpolateIDW(boundaryConditionsU_.getCurrentBCs().head(3));
        for (int k = 0; k < weights.size(); k++)
        {
            Info << "Weight of the trajectory " << k << " is " << weights(k) << endl;
            U -= weights(k) * U0field_[k];
            phi -= weights(k) * Phi0field_[k];
            p_rgh -= weights(k) * Prgh0field_[k];
            T -= weights(k) * T0field_[k];
        }
    }
    currentState_.head(numberOfModes_.velocity) = ITHACAutilities::getCoeffs(U, Umodes_);
    currentState_.segment(numberOfModes_.velocity, numberOfModes_.pressure) = ITHACAutilities::getCoeffs(p_rgh, Prghmodes_);
    currentState_.tail(numberOfModes_.temperature) = ITHACAutilities::getCoeffs(T, Tmodes_);
    currentNutCoeffs_ = ITHACAutilities::getCoeffs(nutFields_[0], Nutmodes_);
    currentNutAvgCoeffs_ = interpolateIDW(boundaryConditionsU_.getCurrentBCs().head(3));
}


void ReducedLSPGUnsteadyBBTurb::readEigenvalues()
{
    Eigen::VectorXd uEigenvalues_, pEigenvalues_, tEigenvalues_, nutEigenvalues_;
    uEigenvalues_ = Eigen::VectorXd::Zero(numberOfModes_.velocity);
    pEigenvalues_ = Eigen::VectorXd::Zero(numberOfModes_.pressure);
    tEigenvalues_ = Eigen::VectorXd::Zero(numberOfModes_.temperature);
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

    std::ifstream pFile("ITHACAoutput/POD/Eigenvalues_p_rgh");
    M_Assert(pFile.is_open(),
        "Could not open file ITHACAoutput/POD/Eigenvalues_p_rgh. Please make sure the file exists and is readable.");
    std::getline(pFile, line); // Ignore first line
    std::getline(pFile, line); // Ignore second line

    for (int i = 0; i < numberOfModes_.pressure; i++)
    {
        std::getline(pFile, line);
        pEigenvalues_(i) = std::stod(line);
    }

    std::ifstream tFile("ITHACAoutput/POD/Eigenvalues_T");
    M_Assert(tFile.is_open(),
        "Could not open file ITHACAoutput/POD/Eigenvalues_T. Please make sure the file exists and is readable.");
    std::getline(tFile, line); // Ignore first line
    std::getline(tFile, line); // Ignore second line

    for (int i = 0; i < numberOfModes_.temperature; i++)
    {
        std::getline(tFile, line);
        tEigenvalues_(i) = std::stod(line);
    }

    // std::ifstream nutFile("ITHACAoutput/POD/Eigenvalues_nut");
    // M_Assert(nutFile.is_open(),
    //     "Could not open file ITHACAoutput/POD/Eigenvalues_nut. Please make sure the file exists and is readable.");
    // std::getline(nutFile, line); // Ignore first line
    // std::getline(nutFile, line); // Ignore second line

    // for (int i = 0; i < numberOfModes_.nut; i++)
    // {
    //     std::getline(nutFile, line);
    //     nutEigenvalues_(i) = std::stod(line);
    // }

    Info << "### EIGS - Velocity eigenvalues: " << uEigenvalues_.transpose() << nl
         << "### EIGS - Pressure eigenvalues: " << pEigenvalues_.transpose() << nl
         << "### EIGS - Temperature eigenvalues: " << tEigenvalues_.transpose() << endl;
    //  << "### EIGS - Turbulent viscosity eigenvalues: " << nutEigenvalues_.transpose() << endl;

    eigenvalues_ = Eigen::VectorXd::Zero(
        numberOfModes_.velocity + numberOfModes_.pressure + numberOfModes_.temperature);
    stateScaling_ = Eigen::VectorXd::Zero(
        numberOfModes_.velocity + numberOfModes_.pressure + numberOfModes_.temperature);
    eigenvalues_ << uEigenvalues_, pEigenvalues_, tEigenvalues_;
    // The state scaling is sqrt(eigenvalue)
    stateScaling_ = eigenvalues_.array().sqrt();
    M_Assert(stateScaling_.size() == currentState_.size(),
        "State scaling size does not match current state size.");
    // Make sure no scaling factor is zero to avoid division by zero
    M_Assert(stateScaling_.minCoeff() > 1e-10,
        "State scaling has a zero or negative value, which is not allowed.");
}

void ReducedLSPGUnsteadyBBTurb::setDimensionParameters(const double u_ref, const double l_ref, const double t_ref)
{
    Info << "Setting dimension parameters: u_ref = " << u_ref << ", l_ref = " << l_ref << ", t_ref = " << t_ref << endl;
    dimensionParameters_.u_ref = u_ref;
    dimensionParameters_.l_ref = l_ref;
    dimensionParameters_.t_ref = t_ref;
}
