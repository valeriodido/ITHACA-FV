#include "LSPGSettings.H"
#include "ITHACAassert.H"

ROMSettings ROMSettings::read(const dictionary& dict)
{
    ROMSettings s;
    s.romID = dict.lookupOrDefault<word>("romID", s.romID);
    s.scaleResidualWithCellVolume = dict.lookupOrDefault<bool>("scaleResidualWithCellVolume", s.scaleResidualWithCellVolume);
    s.debug = dict.lookupOrDefault<bool>("debug", s.debug);
    s.writeResidualFields = dict.lookupOrDefault<bool>("writeResidualFields", s.writeResidualFields);
    s.jacobianStep = dict.lookupOrDefault<scalar>("jacobianStep", s.jacobianStep);
    s.jacobianOrder = dict.lookupOrDefault<label>("jacobianOrder", s.jacobianOrder);

    const word preconditioner = dict.lookupOrDefault<word>("residualPreconditioner", "none");
    M_Assert(preconditioner == "none" || preconditioner == "jacobi",
        "Invalid residualPreconditioner in ITHACAdict. Valid options are 'none' or 'jacobi'.");
    s.residualPreconditioner =
        preconditioner == "jacobi" ? ResidualPreconditionerType::Jacobi : ResidualPreconditionerType::None;

    M_Assert(s.jacobianOrder == 1 || s.jacobianOrder == 2,
        "jacobianOrder must be 1 (forward) or 2 (central) in ITHACAdict.");
    M_Assert(s.jacobianStep > 0, "jacobianStep must be > 0.");
    return s;
}

AffineReferenceSettings AffineReferenceSettings::read(const dictionary& dict)
{
    AffineReferenceSettings s;
    const word bcMethod = dict.lookupOrDefault<word>("bcMethod", "lift");
    const label snapshotCollectionMethod = dict.lookupOrDefault<label>("snapshotCollectionMethod", 1);
    M_Assert(bcMethod == "lift" || bcMethod == "None",
        "Invalid bcMethod specified in ITHACAdict. Valid options are 'lift' or 'none'.");
    M_Assert(bcMethod == "None" || snapshotCollectionMethod == 0,
        "If using bcMethod 'lift', snapshotCollectionMethod must be set to 0 in ITHACAdict.");

    if (bcMethod == "lift")
    {
        s.type = AffineReferenceType::Lift;
    } else if (snapshotCollectionMethod == 1)
    {
        s.type = AffineReferenceType::InitialCondition;
    } else
    {
        s.type = AffineReferenceType::None;
    }

    s.idwExponent = dict.getOrDefault<scalar>("parametricDistanceExponent", s.idwExponent);
    s.idwNeighbours = dict.getOrDefault<label>("maxInterpolatedSolutions", s.idwNeighbours);
    M_Assert(s.idwExponent > 0, "parametricDistanceExponent must be positive.");
    return s;
}

LMSolverSettings LMSolverSettings::read(const dictionary& dict)
{
    LMSolverSettings s;
    s.maxIterations = dict.getOrDefault<label>("maxIterations", s.maxIterations);
    s.gradientTol = dict.getOrDefault<scalar>("gradientTol", s.gradientTol);
    s.reductionTol = dict.getOrDefault<scalar>("reductionTol", s.reductionTol);
    s.stepTol = dict.getOrDefault<scalar>("stepTol", s.stepTol);
    const word reuse = dict.getOrDefault<word>("jacobianReuse", "none");
    M_Assert(reuse == "none" || reuse == "adaptive", "lmSolver: jacobianReuse must be 'none' or 'adaptive'.");
    s.reuseJacobian = reuse == "adaptive";
    s.refreshRatio = dict.getOrDefault<scalar>("refreshRatio", s.refreshRatio);
    M_Assert(s.maxIterations > 0, "lmSolver: maxIterations must be positive.");
    M_Assert(s.gradientTol >= 0 && s.reductionTol >= 0 && s.stepTol >= 0,
        "lmSolver: the tolerances must be non-negative.");
    return s;
}

LMDiagnosticsSettings LMDiagnosticsSettings::read(const dictionary& dict)
{
    LMDiagnosticsSettings s;
    s.enabled = dict.getOrDefault<bool>("enabled", false);
    s.triggers = dict.getOrDefault<wordList>("triggers", wordList({ "XtolTooSmall" }));
    s.maxDumps = dict.getOrDefault<label>("maxDumps", 3);
    s.steps = dict.getOrDefault<scalarList>("steps",
        scalarList({ 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8 }));
    s.coordinateScan = dict.getOrDefault<bool>("coordinateScan", true);
    return s;
}
