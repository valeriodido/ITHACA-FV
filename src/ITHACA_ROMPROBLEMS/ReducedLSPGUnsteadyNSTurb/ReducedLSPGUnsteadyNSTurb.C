#include "ReducedLSPGUnsteadyNSTurb.H"
#include "IncompressibleNSEquations.H"
#include "ITHACAcoeffsMass.H"
#include "ITHACAassign.H"
#include <cctype>

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

/// RMS over the snapshots of sqrt(sum_cells |f|^2 V / sum V)
template <class Type>
scalar fieldRMS(const PtrList<GeometricField<Type, fvPatchField, volMesh>>& snapshots)
{
    M_Assert(snapshots.size() > 0, "fieldRMS: no snapshots available to compute the field scale.");
    const scalarField& V = snapshots[0].mesh().V().field();
    const scalar Vtot = gSum(V);
    scalar sumSq = 0;
    forAll(snapshots, i)
    {
        sumSq += gSum(magSqr(snapshots[i].primitiveField()) * V) / Vtot;
    }
    return Foam::sqrt(sumSq / snapshots.size());
}

/// Vectors stacked one after the other
Eigen::VectorXd concatenate(const std::vector<Eigen::VectorXd>& parts)
{
    Eigen::Index n = 0;
    for (const Eigen::VectorXd& v : parts)
    {
        n += v.size();
    }
    Eigen::VectorXd out(n);
    n = 0;
    for (const Eigen::VectorXd& v : parts)
    {
        out.segment(n, v.size()) = v;
        n += v.size();
    }
    return out;
}


/// Number of projection modes of a field in ITHACAdict: Nmodes<Name>proj, e.g. NmodesUproj, NmodesOmegaproj
label requestedModes(const dictionary& dict, const word& name)
{
    word key(name);
    key[0] = std::toupper(key[0]);
    const label n = dict.lookupOrDefault<label>("Nmodes" + key + "proj", 0);
    M_Assert(n > 0, ("Number of " + name + " modes (Nmodes" + key + "proj) must be greater than zero.").c_str());
    return n;
}

/// ROM name of a reduced field from the name of its OpenFOAM field, e.g. k_r -> k
word reducedName(const word& fieldName)
{
    const bool suffix = fieldName.size() > 2 && fieldName.compare(fieldName.size() - 2, 2, "_r") == 0;
    return suffix ? word(fieldName.substr(0, fieldName.size() - 2)) : fieldName;
}

/// Offline data of a reduced field: its modes, the snapshots the POD was computed on, and the reference
/// fields of the affine reference
template <class Type>
struct OfflineFieldData {
    const Modes<Type, fvPatchField, volMesh>& modes;
    PtrList<GeometricField<Type, fvPatchField, volMesh>>& snapshots;
    const PtrList<GeometricField<Type, fvPatchField, volMesh>>& references;
};

/// Offline data of a transported field of the closure
OfflineFieldData<scalar> turbulenceData(LSPGUnsteadyNSTurb& fom, const word& name, const bool initialConditions)
{
    static const PtrList<volScalarField> none;
    if (name == "k")
    {
        return { fom.kmodes, fom.kfield, initialConditions ? fom.k0field : none };
    }
    if (name == "omega")
    {
        return { fom.omegamodes, fom.omegafield, initialConditions ? fom.omega0field : none };
    }
    FatalErrorInFunction << "No offline data for the transported field " << name << exit(FatalError);
    return { fom.kmodes, fom.kfield, none };
}

/// Position of a field in the inner product weights of the stacked POD (jointPODWeights), -1 if absent
label jointWeightIndex(const word& name)
{
    return wordList({ "U", "p", "k", "omega" }).find(name);
}

/// Field of the case read from the current time directory, registered on the mesh and written automatically
template <class FieldType>
FieldType* readField(const word& name, const fvMesh& mesh)
{
    return new FieldType(
        IOobject(name, mesh.time().timeName(), mesh, IOobject::MUST_READ, IOobject::AUTO_WRITE),
        mesh);
}
}


ReducedLSPGUnsteadyNSTurb::ReducedLSPGUnsteadyNSTurb(
    std::shared_ptr<ITHACAcontext> context,
    const LSPGUnsteadyNSTurb& problem): ReducedLSPG(std::move(context)),
                                        fluxResidual_(ITHACAdict->lookupOrDefault<bool>("fluxResidual", true)),
                                        inletIndex_(problem.inletIndex),
                                        affine_(affineSettings_, problem.mu)
{
    createFields();
    createTurbulenceModel(problem);
    const TrainingStatistics stats = createReducedFields(problem);
    createStateMap(problem);
    setStateScaling(stats);
    setFieldScales(stats);
    createResidual();

    lmDiagnostics_.reset(new LMDiagnostics(
        LMDiagnosticsSettings::read(ITHACAdict->subOrEmptyDict("lmDiagnostics")),
        residual_,
        reportedBlocks(),
        reportedBlockNames(),
        coordinateLabels(),
        phi_(),
        [this]() { return turbulence_->clippedCells(); }));
}


void ReducedLSPGUnsteadyNSTurb::createFields()
{
    Time& runTime = this->runTime();
    fvMesh& mesh = this->mesh();

    IOdictionary transportProperties(
        IOobject(
            "transportProperties",
            runTime.constant(),
            mesh,
            IOobject::MUST_READ_IF_MODIFIED,
            IOobject::NO_WRITE));

    // Laminar viscosity [m2/s]
    nu_.reset(new dimensionedScalar("nu", dimViscosity, transportProperties.lookup("nu")));

    // Turbulent viscosity [m2/s]
    nut_.reset(readField<volScalarField>("nut_r", mesh));
    U_.reset(readField<volVectorField>("U_r", mesh));
    p_.reset(readField<volScalarField>("p_r", mesh));

    phi_.reset(new surfaceScalarField(
        IOobject("phi_r", runTime.timeName(), mesh, IOobject::NO_READ, IOobject::AUTO_WRITE),
        linearInterpolate(U_()) & mesh.Sf()));

    setRefCell(p_(), pimple().dict(), pRefCell_, pRefValue_);
    // The Rhie-Chow flux phi = phiHbyA - pEqn.flux() needs the pressure flux
    mesh.setFluxRequired(p_().name());
}


void ReducedLSPGUnsteadyNSTurb::createTurbulenceModel(const LSPGUnsteadyNSTurb& problem)
{
    fvMesh& mesh = this->mesh();
    const dictionary* turbDictPtr = nullptr;
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
        nu_(), nut_(), U_(), phi_()
    };

    if (modelType == "kOmegaSST")
    {
        // The transported fields of the closure, read from the case start time
        turbulenceFields_.push_back(readField<volScalarField>("k_r", mesh));
        turbulenceFields_.push_back(readField<volScalarField>("omega_r", mesh));
        turbulence_.reset(
            new ITHACATurbulenceModels::ResidualAssemblerSST(
                mesh,
                mesh.time(),
                shared,
                problem.k0field,
                problem.omega0field,
                rasDict.optionalSubDict("kOmegaSSTCoeffs"),
                ITHACAdict->lookupOrDefault<scalar>("turbulenceBoundFactor", 0.1)));
    } else
    {
        FatalErrorInFunction
            << "Unsupported RAS model " << modelType
            << ". Available: kOmegaSST" << exit(FatalError);
    }

    const wordList& transported = turbulence_->transportedFields();
    M_Assert(transported.size() == turbulenceFields_.size(), "The closure transports fields that were not created.");
    forAll(transported, i)
    {
        M_Assert(transported[i] == turbulenceFields_[i].name(), "The fields of the closure are not in the expected order.");
    }
}


ReducedLSPGUnsteadyNSTurb::TrainingStatistics ReducedLSPGUnsteadyNSTurb::createReducedFields(
    const LSPGUnsteadyNSTurb& problem)
{
    const AffineReferenceType reference = affine_.type();
    const bool lift = reference == AffineReferenceType::Lift;
    const bool initialConditions = reference == AffineReferenceType::InitialCondition;
    M_Assert(problem.jointPOD.empty() || problem.jointPOD.found("U"), "The stacked POD (jointPOD) must contain U.");

    LSPGUnsteadyNSTurb& fom = const_cast<LSPGUnsteadyNSTurb&>(problem); // Remove the constness 
    const PtrList<volVectorField> noVector;
    const PtrList<volScalarField> noScalar;
    const PtrList<surfaceScalarField> noSurface;

    // Statistics computed on the unnormalised snapshots, i.e. on the coefficients the ROM actually sees
    TrainingStatistics stats;

    auto addField = [&](auto* field, auto& snapshots) {
        fields_.append(field);
        stats.fieldRMS.push_back(fieldRMS(snapshots));
        if (field->joint())
        {
            stats.jointProjections.push_back(ITHACAutilities::getMassMatrix(field->modes(), snapshots));
        } else
        {
            stats.segregatedRMS.push_back(coefficientRMS(snapshots, field->modes()));
        }
    };
    auto checkModes = [&](const label available, const word& name) {
        M_Assert(available == requestedModes(*ITHACAdict, name),
            ("Number of " + name + " modes does not match the size of the " + name + " modes.").c_str());
    };

    checkModes(problem.Umodes.size(), "U");
    addField(new volVectorReducedField(
                 "U", U_(), problem.Umodes,
                 lift ? problem.liftfield : initialConditions ? problem.U0field
                                                              : noVector,
                 fields_.stateSize(), problem.jointPOD.found("U")),
        lift ? fom.Uomfield : fom.Ufield);

    // The face flux shares the coefficients of the velocity
    M_Assert(problem.Phimodes.size() == problem.Umodes.size(),
        "Number of velocity modes does not match the size of the face flux modes matrix.");
    phiField_.reset(new surfaceScalarReducedField(
        "phi", phi_(), problem.Phimodes,
        lift ? problem.liftfieldphi : initialConditions ? problem.Phi0field
                                                        : noSurface,
        fields_[0].offset(), false));

    // No BC correction at reconstruction: the pressure BCs are corrected after the affine reference is added
    checkModes(problem.Pmodes.size(), "p");
    addField(new volScalarReducedField(
                 "p", p_(), problem.Pmodes, initialConditions ? problem.P0field : noScalar,
                 fields_.stateSize(), problem.jointPOD.found("p"), false),
        fom.Pfield);

    forAll(turbulenceFields_, i)
    {
        const word name = reducedName(turbulenceFields_[i].name());
        const OfflineFieldData<scalar> data = turbulenceData(fom, name, initialConditions);
        checkModes(data.modes.size(), name);
        addField(new volScalarReducedField(
                     name, turbulenceFields_[i], data.modes, data.references,
                     fields_.stateSize(), problem.jointPOD.found(name)),
            data.snapshots);
    }

    for (label f = 0; f < fields_.size(); f++)
    {
        referencedFields_.push_back(&fields_[f]);
    }
    referencedFields_.push_back(phiField_.get());

    return stats;
}


void ReducedLSPGUnsteadyNSTurb::createStateMap(const LSPGUnsteadyNSTurb& problem)
{
    // Stacked POD: the joint fields share one set of coefficients, the reduced coordinates
    stateMap_ = StateMap(fields_);
    if (!problem.jointPOD.empty())
    {
        Eigen::VectorXd weights = Eigen::VectorXd::Zero(fields_.size());
        for (label f = 0; f < fields_.size(); f++)
        {
            const label w = jointWeightIndex(fields_[f].name());
            weights(f) = fields_[f].joint() && w >= 0 ? problem.jointPODWeights(w) : 0.0;
        }
        joint_.reset(new JointProjector(fields_, weights));
    }

    currentState_ = Eigen::VectorXd::Zero(fields_.stateSize());
    currentCoords_ = Eigen::VectorXd::Zero(stateMap_.nCoordinates(fields_.stateSize()));
}


void ReducedLSPGUnsteadyNSTurb::setStateScaling(const TrainingStatistics& stats)
{
    // Same order as the reduced coordinates (StateMap): the shared coefficients, then the other fields
    std::vector<Eigen::VectorXd> scales;
    if (joint_)
    {
        scales.push_back(rowRMS(joint_->solve(stats.jointProjections)));
    }
    scales.insert(scales.end(), stats.segregatedRMS.begin(), stats.segregatedRMS.end());

    // Scaled coordinates y = c / rms(c_training) are O(1) for every mode and field
    stateScaling_ = concatenate(scales);
    M_Assert(stateScaling_.size() == currentCoords_.size(),
        "Coefficient scale size does not match the number of reduced coordinates.");
    Info << "### State scaling (RMS of the training coefficients): " << stateScaling_.transpose() << endl;
}


void ReducedLSPGUnsteadyNSTurb::setFieldScales(const TrainingStatistics& stats)
{
    // Field scales of the preconditioned residual: RMS over the snapshots of the volume-weighted L2 norm
    fieldScale_ = Eigen::Map<const Eigen::VectorXd>(stats.fieldRMS.cdata(), stats.fieldRMS.size());
    const scalarList userScales = ITHACAdict->lookupOrDefault<scalarList>("preconditionerScales", scalarList());
    if (userScales.size())
    {
        M_Assert(userScales.size() == fields_.size(),
            "preconditionerScales must have one entry per reduced field: (U p k omega).");
        forAll(userScales, i)
        {
            fieldScale_(i) = userScales[i];
        }
    }
    M_Assert(fieldScale_.minCoeff() > 0, "The preconditioner field scales must be positive.");
}


void ReducedLSPGUnsteadyNSTurb::createResidual()
{
    const fvMesh& mesh = this->mesh();
    std::vector<BlockSpec> blocks {
        { "U", 3 * nCells, 1.0 },
        { "p", nCells, 1.0 }
    };
    for (label f = 2; f < fields_.size(); f++)
    {
        blocks.push_back({ fields_[f].name(), nCells, 1.0 });
    }
    if (fluxResidual_)
    {
        blocks.push_back({ "phi", mesh.nInternalFaces(), 1.0 });
    }
    residual_ = BlockResidual { blocks };

    // Block of each field of fields_, in order
    labelList fieldBlocks(fields_.size());
    forAll(fieldBlocks, f)
    {
        fieldBlocks[f] = residual_.index(fields_[f].name());
    }
    iU_ = fieldBlocks[0];
    iP_ = fieldBlocks[1];
    iTurbulence_ = labelList(SubList<label>(fieldBlocks, fields_.size() - 2, 2));
    iPhi_ = fluxResidual_ ? residual_.index("phi") : -1;

    if (romSettings_.scaleResidualWithCellVolume)
    {
        setCellVolumeRowWeights(iU_, 3);
        for (label f = 1; f < fields_.size(); f++)
        {
            setCellVolumeRowWeights(fieldBlocks[f], 1);
        }

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
            residual_.setBlockRowWeights(iPhi_, wPhi);
        }
    }

    if (romSettings_.residualPreconditioner == ResidualPreconditionerType::Jacobi)
    {
        forAll(fieldBlocks, f)
        {
            residual_.setBlockWeight(fieldBlocks[f], 1.0 / fieldScale_(f));
        }
        if (iPhi_ >= 0)
        {
            residual_.setBlockWeight(iPhi_, 1.0 / fieldScale_(0));
        }
        Info << "### Jacobi-preconditioned residual, field scales (" << fields_.names() << "): "
             << fieldScale_.transpose() << endl;
    }

    // Logs and residualBlockWeights: every field block, then phi (0 without fluxResidual)
    labelList reported(fieldBlocks);
    reported.push_back(iPhi_);
    wordList names(fields_.names());
    names.push_back("phi");
    setReportedBlocks(reported, names);
}


void ReducedLSPGUnsteadyNSTurb::solveOnline(const word& bcROM, const word& folder)
{
    resetToInitialCondition();
    boundaryConditions_ = BoundaryConditions { bcROM, "linear" };
    lmDiagnostics_->reset();
    cumulativeContErr_ = 0;
    runOnline(folder);
}


void ReducedLSPGUnsteadyNSTurb::updateBoundaryConditions(const scalar time)
{
    boundaryConditions_.updateTimeDependentBC(time);
}


void ReducedLSPGUnsteadyNSTurb::timeStepCompleted()
{
    const Time& runTime = this->runTime();
    const volScalarField contErr(fvc::div(phi_()));
    const scalar sumLocalContErr = runTime.deltaTValue() *
        mag(contErr)().weightedAverage(mesh().V()).value();
    const scalar globalContErr = runTime.deltaTValue() *
        contErr.weightedAverage(mesh().V()).value();
    cumulativeContErr_ += globalContErr;

    Info << "  - time step continuity errors : sum local = " << sumLocalContErr
         << "  |  global = " << globalContErr
         << "  |  cumulative = " << cumulativeContErr_ << nl;
}


void ReducedLSPGUnsteadyNSTurb::writeFields(const word& folder) const
{
    fields_.write(name(runTime().time().value()), folder);
}


void ReducedLSPGUnsteadyNSTurb::diagnoseSolve(
    const LevenbergMarquardtFunctor& functor,
    const Eigen::VectorXd& y,
    const std::string& status,
    const word& folder)
{
    if (lmDiagnostics_->triggered(status))
    {
        lmDiagnostics_->run(functor, y, status, folder, runTime().value());
    }
}


Eigen::VectorXd ReducedLSPGUnsteadyNSTurb::assembleResidual(
    const Eigen::VectorXd& state,
    const bool correctBCs,
    const bool writeResidual)
{
    reconstructFields(state, correctBCs);
    turbulence_->correctNut();
    return residualWithCurrentFields(writeResidual);
}


Eigen::VectorXd ReducedLSPGUnsteadyNSTurb::residualWithCurrentFields(const bool writeResidual)
{
    volVectorField& U = U_();
    volScalarField& p = p_();
    surfaceScalarField& phi = phi_();

    tmp<fvVectorMatrix> tUEqn = IncompressibleNSEquations::momentum(U, phi, MRF(), fvOptions(), turbulence_());
    fvVectorMatrix& UEqn = tUEqn.ref();
    const IncompressibleNSEquations::PressureSystem ps(UEqn, U, p, phi, MRF(), pRefCell_, pRefValue_);
    p.correctBoundaryConditions();

    collector_.begin(writeResidual);

    const scalarField Rp(ps.pEqn().residual());
    collector_.add(iP_, ps.pEqn(), Rp);

    // Face-flux residual (He et al. 2021, eq. 1.8): the reconstructed flux against the Rhie-Chow flux
    // of the current (U, p). Assembled only, from the same UEqn and pEqn
    if (iPhi_ >= 0)
    {
        const surfaceScalarField phiRC(ps.flux());
        collector_.add(iPhi_, surfaceScalarField("Rphi", phi - phiRC));
    }

    UEqn += fvc::grad(p);
    const vectorField Ru(UEqn.residual());
    collector_.add(iU_, UEqn, Ru);

    PtrList<fvScalarMatrix> turbEqs = turbulence_->assembleEquations(fvOptions());
    forAll(turbEqs, i)
    {
        const scalarField R(turbEqs[i].residual());
        collector_.add(iTurbulence_[i], turbEqs[i], R);
    }

    collector_.end();
    return residual_.vector();
}


PressureResidualCheck::Terms ReducedLSPGUnsteadyNSTurb::pressureResidualTerms()
{
    tmp<fvVectorMatrix> tUEqn = IncompressibleNSEquations::momentum(U_(), phi_(), MRF(), fvOptions(), turbulence_());
    return PressureResidualCheck::compute(tUEqn(), U_(), p_(), phi_(), MRF(), pRefCell_, pRefValue_);
}


void ReducedLSPGUnsteadyNSTurb::reconstructFields(const Eigen::VectorXd& state, const bool correctBCs)
{
    fields_.reconstruct(state);
    phiField_->reconstruct(state);

    updateAffineReference();
    for (ReducedFieldBase& f : referencedFields_)
    {
        f.addReference();
    }
    imposeInletConditions();

    if (correctBCs)
    {
        p_().correctBoundaryConditions();
    }

    turbulence_->constrainTransportedFields();
}


void ReducedLSPGUnsteadyNSTurb::updateAffineReference()
{
    affine_.update(boundaryConditions_.getCurrentBCs(), referencedFields_);
}


void ReducedLSPGUnsteadyNSTurb::imposeInletConditions()
{
    const AffineReferenceType reference = affine_.type();
    if (reference == AffineReferenceType::None)
    {
        return;
    }
    volVectorField& U = U_();
    surfaceScalarField& phi = phi_();
    const Eigen::VectorXd bcs = boundaryConditions_.getCurrentBCs();
    for (label i = 0; i < inletIndex_.rows(); ++i)
    {
        const label patchID = inletIndex_(i, 0);
        const label cmpt = inletIndex_(i, 1);
        vector velBC = { 0.0, 0.0, 0.0 };
        velBC[cmpt] = bcs(i);
        ITHACAutilities::assignBC(U, patchID, velBC);
        // With the lifting functions, the inlet flux is in the lifted phi
        if (reference == AffineReferenceType::InitialCondition)
        {
            phi.boundaryFieldRef()[patchID] = U.boundaryField()[patchID] & U.mesh().Sf().boundaryField()[patchID];
        }
    }
}


void ReducedLSPGUnsteadyNSTurb::initializePODCoeffsFromFields()
{
    updateAffineReference();
    for (ReducedFieldBase& f : referencedFields_)
    {
        f.removeReference();
    }
    fields_.project(currentState_);

    if (joint_)
    {
        // The stacked modes are not orthonormal field by field: one weighted projection for the whole group
        const Eigen::VectorXd a = joint_->coefficients(fields_);
        for (label f = 0; f < fields_.size(); f++)
        {
            if (fields_[f].joint())
            {
                currentState_.segment(fields_[f].offset(), fields_[f].nModes()) = a;
            }
        }
    }
    currentCoords_ = stateMap_.reduce(currentState_);
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
    resetFromDisk(U_());
    resetFromDisk(p_());
    for (volScalarField& f : turbulenceFields_)
    {
        resetFromDisk(f);
    }
    resetFromDisk(nut_());

    // phi_r is not read: as in createFields
    phi_() == (linearInterpolate(U_()) & mesh.Sf());
    phi_().clearOldTimes();
}


wordList ReducedLSPGUnsteadyNSTurb::coordinateLabels() const
{
    DynamicList<word> labels;
    if (!stateMap_.identity())
    {
        for (label f = 0; f < fields_.size(); f++)
        {
            if (fields_[f].joint())
            {
                for (label i = 0; i < fields_[f].nModes(); i++)
                {
                    labels.push_back("joint" + name(i + 1));
                }
                break;
            }
        }
    }
    for (label f = 0; f < fields_.size(); f++)
    {
        if (stateMap_.identity() || !fields_[f].joint())
        {
            for (label i = 0; i < fields_[f].nModes(); i++)
            {
                labels.push_back(fields_[f].name() + name(i + 1));
            }
        }
    }
    return wordList(std::move(labels));
}
