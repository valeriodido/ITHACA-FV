#include "ReducedLSPGUnsteadyBBTurb.H"
#include "Foam2Eigen.H" // Needed to convert OpenFOAM fields to Eigen vectors and matrices
#include "ITHACAassign.H"


ReducedLSPGUnsteadyBBTurb::ReducedLSPGUnsteadyBBTurb(
  std::shared_ptr<ITHACAcontext> context,
  const LSPGUnsteadyBBTurb& problem
)
: ReducedLSPG(std::move(context))
{
  captureFOMData(problem);

  Time& runTime = this->runTime();
  fvMesh& mesh = this->mesh();
  pimpleControl& pimple = this->pimple();

  #include "createFields.H"

  currentState_ = Eigen::VectorXd::Zero(
    numberOfModes_.velocity +
    numberOfModes_.pressure + 
    numberOfModes_.temperature
  );
  currentNutCoeffs_ = Eigen::VectorXd::Zero(numberOfModes_.nut);
  if (Pstream::master())
  {
    readEigenvalues();
  }

  const label nCells = mesh.nCells();

  Eigen::VectorXd wScalar = Foam2Eigen::field2Eigen(mesh.V().field()).cwiseSqrt().cwiseInverse();
  vectorField wVecField(nCells);
  forAll(wVecField, celli)
  {
    const scalar w = 1.0 / Foam::sqrt(mesh.V()[celli]);
    wVecField[celli] = vector(w, w, w);
  }
  Eigen::VectorXd wVector = Foam2Eigen::field2Eigen(wVecField);

  residualWorkspace_ = BlockResidual
  {
    {
      {"U_r", 3*nCells, 1.0},
      {"p_rgh_r", nCells, 1.0},
      {"T_r", nCells, 1.0}
    }
  };

  iU_ = residualWorkspace_.index("U_r");
  iP_ = residualWorkspace_.index("p_rgh_r");
  iT_ = residualWorkspace_.index("T_r");

  residualWorkspace_.setBlockRowWeights(iU_, wVector);
  residualWorkspace_.setBlockRowWeights(iP_, wScalar);
  residualWorkspace_.setBlockRowWeights(iT_, wScalar);
}

void ReducedLSPGUnsteadyBBTurb::captureFOMData(
  const LSPGUnsteadyBBTurb& problem)
{
  interpolationSettings_ = InterpolationSettings
  {
      problem.ITHACAdict->lookupOrDefault<int>("firstRBFIndex", 0),
      problem.ITHACAdict->lookupOrDefault<bool>("derivativeInRBF", false),
      problem.ITHACAdict->lookupOrDefault<label>("dimInputRBF", 0),
  };

  romSettings_ = ROMSettings
  {
      ITHACAdict->lookupOrDefault<word>("bcMethod", "lift"),
      ITHACAdict->lookupOrDefault<word>("caseIdentifier", "")
  };
  numberOfModes_ = NumberOfModes
  {
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
  mkDir(folder);
  ITHACAutilities::createSymLink(folder);
  Info << "Starting Reduced LSPG solution" << endl;
  boundaryConditions_ = BoundaryConditions{bcROM, bcROMT, "linear"};
  GaussNewtonSettings gn_settings = GaussNewtonSettings{5, 1e-2, 5e-5};
  Time& runTime = this->runTime();
  #include "readTimeControls.H"
  // Maybe here we need: #include "initContinuityErrs.H" - Check later
  
  initializePODCoeffsFromFields(); // Here we cannot correct the BCs for pRgh
  assembleResidual(currentState_, false); // Here we cannot correct the BCs for pRgh
  assembleResidual(currentState_, true); // Here we can correct the BCs for pRgh, as the snGrad is filled
  residualWorkspace_.calibrateWeights(1e-8);

  LevenbergMarquardtFunctor lmFunctor(this, currentState_.size(), residualWorkspace_.size());
  Eigen::NumericalDiff<LevenbergMarquardtFunctor> numDiff(lmFunctor);
  Eigen::LevenbergMarquardt<Eigen::NumericalDiff<LevenbergMarquardtFunctor>> lm(numDiff);

  lm.parameters.ftol   = 1e-8;
  lm.parameters.xtol   = 1e-8;
  lm.parameters.gtol   = 1e-10;
  lm.parameters.maxfev = 500;

  auto start = std::chrono::high_resolution_clock::now();
  while (runTime.run())
  {
    runTime++;
    Info << "Time = " << runTime.time().value() << nl << endl;
    boundaryConditions_.updateTimeDependentBC(runTime.time().value());

    Eigen::VectorXd scaledState = currentState_.cwiseQuotient(stateScaling_);
    // minimizeInit evaluates the residual and Jacobian once at currentState_
    // and sets lm's internal state up for iterating from there.
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
    // Called for logging purposes
    assembleResidual(currentState_);
    Info << "  - U residual norm " << residualWorkspace_.blockNorm(iU_) << nl
         << "  - p_rgh residual norm " << residualWorkspace_.blockNorm(iP_) << nl
         << "  - T residual norm " << residualWorkspace_.blockNorm(iT_) << endl;

    // runTime.write();
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
  bool correctBCs)
{
  fvMesh& mesh = this->mesh();
  fv::options& fvOptions = this->fvOptions();
  IOMRFZoneList& MRF = this->MRF();

  volVectorField& U = _U();
  volScalarField& p_rgh = _p_rgh();
  volScalarField& T = _T();
  volScalarField& nut = _nut();
  volScalarField& alphat = _alphat();
  volScalarField& rhok = _rhok();
  volScalarField& gh = _gh();
  surfaceScalarField& ghf = _ghf();
  surfaceScalarField& phi = _phi();
  dimensionedScalar& beta = _beta();
  dimensionedScalar& TRef = _TRef();
  dimensionedScalar& Pr = _Pr();
  dimensionedScalar& Prt = _Prt();

  reconstructReducedFields(state, U, p_rgh, T, phi, correctBCs);
  // Info << "max|div(phi)| = " << max(mag(fvc::div(phi))).value() << endl;
  
  interpolateNutCoeffs(nut, Nutmodes_, state);
  volScalarField nuEff("nuEffROM", _nu() + nut);
  MRF.correctBoundaryVelocity(U);
  fvVectorMatrix UEqn
  (
      fvm::ddt(U) + fvm::div(phi, U)
      + MRF.DDt(U)
      // + turbulence->divDevReff(U) // Since we are using a custom turbulence, we have to express this fully
      - fvc::div(nuEff*dev2(Foam::T(fvc::grad(U)))) - fvm::laplacian(nuEff, U)
      ==
      fvOptions(U)
  );
  fvOptions.constrain(UEqn);
  alphat = nut / Prt;
  alphat.correctBoundaryConditions();
  volScalarField alphaEff("alphaEff", _nu() / Pr + alphat);
  fvScalarMatrix TEqn
  (
      fvm::ddt(T)
      + fvm::div(phi, T)
      - fvm::laplacian(alphaEff, T)
      ==
      fvOptions(T)
  );
  fvOptions.constrain(TEqn);
  rhok = 1.0 - beta * (T - TRef);

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
  constrainPressure(p_rgh, U, phiHbyA, rAUf, MRF);
  fvScalarMatrix p_rghEqn
  (
      fvm::laplacian(rAUf, p_rgh) == fvc::div(phiHbyA)
  );
  if (p_rgh.needReference())
  {
    Info << "p_rghEqn: reference cell required" << endl;
    const scalar pRefRgh = pRefValue -rhok[pRefCell] * gh[pRefCell];
    p_rghEqn.setReference(pRefCell, pRefRgh);
  }
  
  volVectorField Usrc = -fvc::reconstruct
  (
      (
          ghf * fvc::snGrad(rhok)
          + fvc::snGrad(p_rgh)
      ) * mesh.magSf()
  );
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

  if (romSettings_.bcMethod == "lift")
  {
    const Eigen::VectorXd& bcs = boundaryConditions_.getCurrentBCs();
    for (label i = 0; i < inletIndex_.rows(); ++i)
    {
      const label patchID = inletIndex_(i, 0);
      const label cmpt = inletIndex_(i, 1);
      vector velBC = {0.0, 0.0, 0.0};
      velBC[cmpt] = bcs(i);
      ITHACAutilities::assignBC(velocity_field, patchID, velBC);
      velocity_field += liftFields_[i] * bcs(i);
      phi_field += liftFieldsPhi_[i] * bcs(i);
    }
    for (label i = 0; i < inletIndexT_.rows(); ++i)
    {
      label patchID = inletIndexT_(i, 0);
      ITHACAutilities::assignBC(temperature_field, patchID, boundaryConditions_.getCurrentBCs()(i + liftFields_.size()));
      temperature_field += liftFieldsT_[i] * boundaryConditions_.getCurrentBCs()(i + liftFields_.size());
    }
  }
  if (correctBCs) // On the first iteration, calling correctBoundaryConditions() raises the --> FOAM FATAL ERROR: updateCoeffs(const scalarField& snGradp) MUST be called before updateCoeffs() or evaluate() to set the boundary gradient.
  {
    pressure_field = Prghmodes_.reconstruct(pressure_field, state.segment(numberOfModes_.velocity, numberOfModes_.pressure), "p_rgh_r", false);
    pressure_field.correctBoundaryConditions();
  }
  else
  {
    pressure_field = Prghmodes_.reconstruct(pressure_field, state.segment(numberOfModes_.velocity, numberOfModes_.pressure), "p_rgh_r", false);
  }
}


void ReducedLSPGUnsteadyBBTurb::interpolateNutCoeffs(volScalarField& nut_field, volScalarModes& nut_modes, const Eigen::VectorXd& state)
{
  const label inputRBFSize =
    interpolationSettings_.dimInputRBF > 0
  ? interpolationSettings_.dimInputRBF
  : numberOfModes_.velocity;

  Eigen::VectorXd velocity_coeffs = state.head(inputRBFSize);
  if (interpolationSettings_.derivative)
  {
    FatalErrorInFunction << "Derivative interpolation for nut coefficients is not implemented yet." << exit(FatalError);
  }
  else
  {
    for (int j=0; j < numberOfModes_.nut; j++)
    {
      currentNutCoeffs_(j) = rbfSplines_[j]->predict(velocity_coeffs);
    }
  }
  nut_field = nut_modes.reconstruct(nut_field, currentNutCoeffs_, "nut_r");
  for (int k = 0; k < currentNutAvgCoeffs_.size(); k++)
  {
      nut_field += currentNutAvgCoeffs_(k) * avgNutFields_[k];
  }
  nut_field.correctBoundaryConditions(); // This is probably not needed, as it should be called by reconstruct()
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
    }
    else
    {
        weights.setConstant(1.0 / n_samples);
    }

    for (int j = 0; j < weights.size(); j++)
    {
        if (std::abs(weights(j)) < 1e-6)
        {
            weights(j) = 0.0;
        }
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
          U -= boundaryConditions_.getCurrentBCs()(i) * liftFields_[i];
        }
        for (int i = 0; i < liftFieldsT_.size(); i++)
        {
          T -= boundaryConditions_.getCurrentBCs()(i + liftFields_.size()) * liftFieldsT_[i];
        }
    }
    currentState_.head(numberOfModes_.velocity) = ITHACAutilities::getCoeffs(U, Umodes_);
    currentState_.segment(numberOfModes_.velocity, numberOfModes_.pressure) = ITHACAutilities::getCoeffs(p_rgh, Prghmodes_);
    currentState_.tail(numberOfModes_.temperature) = ITHACAutilities::getCoeffs(T, Tmodes_);
    currentNutCoeffs_ = ITHACAutilities::getCoeffs(nutFields_[0], Nutmodes_);
    currentNutAvgCoeffs_ = interpolateIDW(boundaryConditions_.getCurrentBCs().head(3));
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
  uEigenvalues_.resize(numberOfModes_.velocity);

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
  pEigenvalues_.resize(numberOfModes_.pressure);

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
  tEigenvalues_.resize(numberOfModes_.temperature);

  for (int i = 0; i < numberOfModes_.temperature; i++)
  {
      std::getline(tFile, line);
      tEigenvalues_(i) = std::stod(line);
  }

  std::ifstream nutFile("ITHACAoutput/POD/Eigenvalues_fluctNut");
  M_Assert(nutFile.is_open(),
          "Could not open file ITHACAoutput/POD/Eigenvalues_fluctNut. Please make sure the file exists and is readable.");
  std::getline(nutFile, line); // Ignore first line
  std::getline(nutFile, line); // Ignore second line
  nutEigenvalues_.resize(numberOfModes_.nut);

  for (int i = 0; i < numberOfModes_.nut; i++)
  {
      std::getline(nutFile, line);
      nutEigenvalues_(i) = std::stod(line);
  }

  Info << "### EIGS - Velocity eigenvalues: " << uEigenvalues_.transpose() << nl
        << "### EIGS - Pressure eigenvalues: " << pEigenvalues_.transpose() << nl
        << "### EIGS - Temperature eigenvalues: " << tEigenvalues_.transpose() << nl
        << "### EIGS - FluctNut eigenvalues: " << nutEigenvalues_.transpose() << endl;
      
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
