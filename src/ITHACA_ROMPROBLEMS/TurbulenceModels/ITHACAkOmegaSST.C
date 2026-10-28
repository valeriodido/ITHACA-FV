#include "ITHACAkOmegaSST.H"
#include <cmath>
#include "omegaWallFunctionFvPatchScalarField.H"
#include "nutWallFunctionFvPatchScalarField.H"
#include "wallFunctionCoefficients.H"
#include "StringStream.H"
#include "HashSet.H"

namespace Foam
{
namespace ITHACATurbulenceModels
{
    ResidualAssemblerSST::ResidualAssemblerSST(
        const fvMesh& mesh,
        const Time& runTime,
        const SharedFields& shared,
        volScalarModes& kModes,
        volScalarModes& omegaModes,
        PtrList<volScalarField> k0Fields,
        PtrList<volScalarField> omega0Fields,
        const dictionary& kOmegaSSTCoeffs,
        const scalar boundFactor,
        const word& GName):

        ResidualEddyViscosityTurbulenceModel(mesh, runTime, shared),

        kModes_(kModes),
        omegaModes_(omegaModes),

        k0Fields_(k0Fields),
        omega0Fields_(omega0Fields),

        k_(mesh.lookupObjectRef<volScalarField>("k_r")),
        omega_(mesh.lookupObjectRef<volScalarField>("omega_r")),

        GName_(GName),
        y_(wallDist::New(mesh).y()),

        fieldNames_({ "k_r", "omega_r" }),

        alphaK1_("alphaK1", dimless, kOmegaSSTCoeffs.getOrDefault<scalar>("alphaK1", 0.85)),
        alphaK2_("alphaK2", dimless, kOmegaSSTCoeffs.getOrDefault<scalar>("alphaK2", 1.0)),
        alphaOmega1_("alphaOmega1", dimless, kOmegaSSTCoeffs.getOrDefault<scalar>("alphaOmega1", 0.5)),
        alphaOmega2_("alphaOmega2", dimless, kOmegaSSTCoeffs.getOrDefault<scalar>("alphaOmega2", 0.856)),
        gamma1_("gamma1", dimless, kOmegaSSTCoeffs.getOrDefault<scalar>("gamma1", 5.0 / 9.0)),
        gamma2_("gamma2", dimless, kOmegaSSTCoeffs.getOrDefault<scalar>("gamma2", 0.44)),
        beta1_("beta1", dimless, kOmegaSSTCoeffs.getOrDefault<scalar>("beta1", 0.075)),
        beta2_("beta2", dimless, kOmegaSSTCoeffs.getOrDefault<scalar>("beta2", 0.0828)),
        betaStar_("betaStar", dimless, kOmegaSSTCoeffs.getOrDefault<scalar>("betaStar", 0.09)),
        a1_("a1", dimless, kOmegaSSTCoeffs.getOrDefault<scalar>("a1", 0.31)),
        b1_("b1", dimless, kOmegaSSTCoeffs.getOrDefault<scalar>("b1", 1.0)),
        c1_("c1", dimless, kOmegaSSTCoeffs.getOrDefault<scalar>("c1", 10.0)),
        F3_(kOmegaSSTCoeffs.getOrDefault<Switch>("F3", false)),
        omegaMin_("omegaMin", dimless / dimTime, SMALL),
        kMin_("kMin", sqr(dimVelocity), SMALL),
        yWall_(mesh)
    {
        forAll(nut_.boundaryField(), patchi)
        {
            if (isA<nutWallFunctionFvPatchScalarField>(nut_.boundaryField()[patchi]))
            {
                FatalErrorInFunction
                    << "nut wall functions (patch " << mesh.boundary()[patchi].name()
                    << ") are not supported by the ROM: they would read the FOM turbulence model."
                    << exit(FatalError);
            }
        }

        // Not registered: OpenFOAM's omegaWallFunction finds the production field by name (kOmegaSST:G) in the
        // mesh registry. A registered ROM field under that name shadows the FOM's own G, which then fails to
        // register (duplicate names are silently rejected): the FOM wall functions write G0 into the ROM field,
        // and every FOM solve after the ROM construction loses the wall-function production in the wall cells
        G_.reset(
            new volScalarField::Internal(
                IOobject(
                    GName_, mesh.time().timeName(), mesh,
                    IOobject::NO_READ, IOobject::NO_WRITE, IOobject::NO_REGISTER
                    ),
                mesh,
                dimensionedScalar("G", dimensionSet(0, 2, -3, 0, 0, 0, 0), 0.0)));

        cacheDirichletValues(k_, kDirichletPatches_, kDirichletValues_);
        cacheDirichletValues(omega_, omegaDirichletPatches_, omegaDirichletValues_);

        // Must come after cacheDirichletValues: the wall patches become fixedValue here
        setupOmegaWallFunctions();

        // Data-based floors: boundFactor times the smallest positive value of the initial field and of the
        // initial snapshots of the training runs (the FOM never goes below these values)
        auto minPositive = [](const volScalarField& f, scalar current)
        {
            forAll(f, celli)
            {
                if (f[celli] > 0)
                {
                    current = min(current, f[celli]);
                }
            }
            return current;
        };
        scalar kRef = minPositive(k_, GREAT);
        scalar omegaRef = minPositive(omega_, GREAT);
        forAll(k0Fields_, i)
        {
            kRef = minPositive(k0Fields_[i], kRef);
        }
        forAll(omega0Fields_, i)
        {
            omegaRef = minPositive(omega0Fields_[i], omegaRef);
        }
        if (kRef < GREAT)
        {
            kMin_.value() = max(SMALL, boundFactor * kRef);
        }
        if (omegaRef < GREAT)
        {
            omegaMin_.value() = max(SMALL, boundFactor * omegaRef);
        }
        Info << "ResidualAssemblerSST: lower bounds kMin " << kMin_.value() << ", omegaMin " << omegaMin_.value()
             << " (boundFactor " << boundFactor << ")" << nl;
    }

    template <class T>
    T ResidualAssemblerSST::blend(const T& F1, const dimensionedScalar& a, const dimensionedScalar& b) const
    {
        return F1 * (a - b) + b;
    }

    tmp<volScalarField> ResidualAssemblerSST::F1(
        const volScalarField& CDkOmega) const
    {
        const volScalarField CDkOmegaPlus(
            max(CDkOmega, dimensionedScalar("1e-10", dimless / sqr(dimTime), 1.0e-10)));

        const volScalarField arg1(
            min(
                min(
                    max(
                        sqrt(k_) / (betaStar_ * omega_ * y_),
                        scalar(500) * nu_ / (sqr(y_) * omega_)),
                    (4 * alphaOmega2_) * k_ / (CDkOmegaPlus * sqr(y_))),
                scalar(10)));
        return tanh(pow4(arg1));
    }

    tmp<volScalarField> ResidualAssemblerSST::F23() const
    {
        const volScalarField arg2(
            min(
                max(
                    (scalar(2) / betaStar_) * sqrt(k_) / (omega_ * y_),
                    scalar(500) * nu_ / (sqr(y_) * omega_)),
                scalar(100)));
        tmp<volScalarField> f2 = tanh(sqr(arg2));

        if (F3_)
        {
            const volScalarField arg3(min(150 * nu_ / (omega_ * sqr(y_)), scalar(10)));
            f2.ref() *= (1 - tanh(pow4(arg3)));
        }
        return f2;
    }

    void ResidualAssemblerSST::correctNut()
    {
        const volScalarField S2(2 * magSqr(symm(fvc::grad(U_))));
        const volScalarField F2(F23());
        nut_ = a1_ * k_ / max(a1_ * omega_, b1_ * F2 * sqrt(S2));
        nut_.correctBoundaryConditions();
    }

    PtrList<fvScalarMatrix> ResidualAssemblerSST::assembleEquations(
        fv::options& fvOptions)
    {
        const volScalarField::Internal divU(fvc::div(fvc::absolute(phi_, U_))());

        const tmp<volTensorField> tgradU = fvc::grad(U_);
        const volScalarField S2(2 * magSqr(symm(tgradU())));
        volScalarField::Internal GbyNu0((tgradU() && dev(twoSymm(tgradU()))));

        volScalarField::Internal& G = G_();
        G = nut_() * GbyNu0;

        // Same order as kOmegaSST::correct(): updates omega (and G) at the wall
        applyOmegaWallFunctions(G);

        const volScalarField CDkOmega(
            (2 * alphaOmega2_) * (fvc::grad(k_) & fvc::grad(omega_)) / omega_);
        const volScalarField F1v(F1(CDkOmega));
        const volScalarField F23v(F23());

        const volScalarField::Internal gamma(blend(F1v(), gamma1_, gamma2_));
        const volScalarField::Internal beta(blend(F1v(), beta1_, beta2_));

        const volScalarField DkEff("DkEff", blend(F1v, alphaK1_, alphaK2_) * nut_ + nu_);

        const volScalarField DomegaEff("DomegaEff", blend(F1v, alphaOmega1_, alphaOmega2_) * nut_ + nu_);

        // omega production limiter
        GbyNu0 = min(
            GbyNu0,
            (c1_ / a1_) * betaStar_ * omega_() * max(a1_ * omega_(), b1_ * F23v() * sqrt(S2())));


        tmp<fvScalarMatrix> omegaEqn(
            fvm::ddt(omega_) + fvm::div(phi_, omega_) - fvm::laplacian(DomegaEff, omega_) ==
            gamma * GbyNu0 - fvm::SuSp((2.0 / 3.0) * gamma * divU, omega_) - fvm::Sp(beta * omega_(), omega_) - fvm::SuSp((F1v() - scalar(1)) * CDkOmega() / omega_(), omega_) + fvOptions(omega_));

        fvOptions.constrain(omegaEqn.ref());
        omegaEqn.ref().boundaryManipulate(omega_.boundaryFieldRef());
        // Equivalent of omegaWallFunction::manipulateMatrix: fix omega in the wall-adjacent cells
        if (omegaWallCells_.size())
        {
            omegaEqn.ref().setValues(
                omegaWallCells_,
                scalarField(omega_.primitiveField(), omegaWallCells_));
        }

        tmp<fvScalarMatrix> kEqn(
            fvm::ddt(k_) + fvm::div(phi_, k_) - fvm::laplacian(DkEff, k_) ==
            min(G, (c1_ * betaStar_) * k_() * omega_()) - fvm::SuSp((2.0 / 3.0) * divU, k_) - fvm::Sp(betaStar_ * omega_(), k_) + fvOptions(k_));
        fvOptions.constrain(kEqn.ref());
        kEqn.ref().boundaryManipulate(k_.boundaryFieldRef());

        PtrList<fvScalarMatrix> eqs(2);
        
        eqs.set(0, kEqn.ptr());
        eqs.set(1, omegaEqn.ptr());

        return eqs;
    }

    void ResidualAssemblerSST::reconstructTurbulenceFields(
        const Eigen::VectorXd& stateCoeffs)
    {
        const label nKModes = kModes_.size();
        const label nOmegaModes = omegaModes_.size();

        const Eigen::VectorXd kCoeffs = stateCoeffs.head(nKModes);
        const Eigen::VectorXd omegaCoeffs = stateCoeffs.segment(nKModes, nOmegaModes);

        k_ = kModes_.reconstruct(k_, kCoeffs, "k_r");
        omega_ = omegaModes_.reconstruct(omega_, omegaCoeffs, "omega_r");

        imposeDirichletValues(k_, kDirichletPatches_, kDirichletValues_);
        imposeDirichletValues(omega_, omegaDirichletPatches_, omegaDirichletValues_);
    }

    void ResidualAssemblerSST::cacheDirichletValues(
        const volScalarField& field,
        labelList& patches,
        List<scalarField>& values)
    {
        DynamicList<label> ids;
        forAll(field.boundaryField(), patchi)
        {
            // Exact type match: wall functions derive from fixedValue but compute their own values
            if (field.boundaryField()[patchi].type() == fixedValueFvPatchScalarField::typeName)
            {
                ids.append(patchi);
            }
        }
        patches.transfer(ids);
        values.setSize(patches.size());
        forAll(patches, i)
        {
            values[i] = field.boundaryField()[patches[i]];
        }
    }

    void ResidualAssemblerSST::imposeDirichletValues(
        volScalarField& field,
        const labelList& patches,
        const List<scalarField>& values)
    {
        forAll(patches, i)
        {
            field.boundaryFieldRef()[patches[i]] == values[i];
        }
    }

    void ResidualAssemblerSST::setupOmegaWallFunctions()
    {
        const fvMesh& mesh = omega_.mesh();
        volScalarField::Boundary& bf = omega_.boundaryFieldRef();

        // Number of wall-function faces seen by each cell (omegaWallFunction::createAveragingWeights)
        scalarField faceCount(mesh.nCells(), 0.0);
        DynamicList<label> wallPatchIDs;
        forAll(bf, patchi)
        {
            if (isA<omegaWallFunctionFvPatchScalarField>(bf[patchi]))
            {
                wallPatchIDs.append(patchi);
                for (const label celli : bf[patchi].patch().faceCells())
                {
                    faceCount[celli] += 1.0;
                }
            }
        }

        DynamicList<label> wallCells;
        for (const label patchi : wallPatchIDs)
        {
            // The blending coefficients are protected in OpenFOAM: read them back from the patch entries
            OStringStream os;
            bf[patchi].write(os);
            IStringStream is(os.str());
            const dictionary dict(is);
            const wallFunctionCoefficients wallCoeffs(dict);

            OmegaWallPatch wp;
            wp.patchi = patchi;
            wp.blending = dict.getOrDefault<word>("blending", "binomial");
            wp.n = dict.getOrDefault<scalar>("n", 2.0);
            wp.beta1 = dict.getOrDefault<scalar>("beta1", 0.075);
            wp.Cmu = wallCoeffs.Cmu();
            wp.kappa = wallCoeffs.kappa();
            wp.yPlusLam = wallCoeffs.yPlusLam();

            if (!wordList({ "stepwise", "max", "binomial", "exponential", "tanh" }).found(wp.blending))
            {
                FatalErrorInFunction
                    << "Unknown omega wall-function blending " << wp.blending
                    << " on patch " << mesh.boundary()[patchi].name() << exit(FatalError);
            }

            const labelUList& faceCells = bf[patchi].patch().faceCells();
            wp.cornerWeights = 1.0 / scalarField(faceCount, faceCells);
            wallCells.append(faceCells);

            Info << "ResidualAssemblerSST: omega wall function on patch "
                 << mesh.boundary()[patchi].name() << " (blending " << wp.blending
                 << ", n " << wp.n << ", Cmu " << wp.Cmu << ", kappa " << wp.kappa
                 << ", yPlusLam " << wp.yPlusLam << ")" << nl;

            omegaWallPatches_.push_back(wp);

            // Plain fixedValue patch: its value is set by applyOmegaWallFunctions
            const scalarField faceValues(bf[patchi]);
            bf.set(patchi, new fixedValueFvPatchScalarField(mesh.boundary()[patchi], omega_.internalField()));
            bf[patchi] == faceValues;
        }

        omegaWallCells_ = labelHashSet(wallCells).sortedToc();
    }

    void ResidualAssemblerSST::applyOmegaWallFunctions(volScalarField::Internal& G)
    {
        if (omegaWallPatches_.empty())
        {
            return;
        }

        const fvMesh& mesh = omega_.mesh();
        const scalar nu = nu_.value();

        // Accumulated, corner-weighted contributions (omegaWallFunction::calculate)
        scalarField G0(mesh.nCells(), 0.0);
        scalarField omega0(mesh.nCells(), 0.0);

        for (const OmegaWallPatch& wp : omegaWallPatches_)
        {
            const labelUList& faceCells = mesh.boundary()[wp.patchi].faceCells();
            const scalarField& y = yWall_[wp.patchi];
            const scalarField& nutw = nut_.boundaryField()[wp.patchi];
            const scalarField magGradUw(mag(U_.boundaryField()[wp.patchi].snGrad()));
            const scalar Cmu25 = pow025(wp.Cmu);

            forAll(faceCells, facei)
            {
                const label celli = faceCells[facei];
                const scalar w = wp.cornerWeights[facei];
                const scalar sqrtk = sqrt(max(k_[celli], scalar(0)));
                const scalar yPlus = Cmu25 * y[facei] * sqrtk / nu;
                const scalar omegaVis = 6.0 * nu / (wp.beta1 * sqr(y[facei]));
                const scalar omegaLog = sqrtk / (Cmu25 * wp.kappa * y[facei]);

                scalar omegaFace = 0.0;
                if (wp.blending == "stepwise")
                {
                    omegaFace = (yPlus > wp.yPlusLam) ? omegaLog : omegaVis;
                }
                else if (wp.blending == "binomial")
                {
                    omegaFace = pow(pow(omegaVis, wp.n) + pow(omegaLog, wp.n), scalar(1) / wp.n);
                }
                else if (wp.blending == "max")
                {
                    omegaFace = max(omegaVis, omegaLog);
                }
                else if (wp.blending == "exponential")
                {
                    const scalar Gamma = 0.01 * pow4(yPlus) / (1 + 5 * yPlus);
                    const scalar invGamma = scalar(1) / (Gamma + ROOTVSMALL);
                    omegaFace = omegaVis * exp(-Gamma) + omegaLog * exp(-invGamma);
                }
                else // tanh
                {
                    const scalar b1 = omegaVis + omegaLog;
                    const scalar b2 = pow(pow(omegaVis, 1.2) + pow(omegaLog, 1.2), 1.0 / 1.2);
                    const scalar phiTanh = tanh(pow4(0.1 * yPlus));
                    omegaFace = phiTanh * b1 + (1 - phiTanh) * b2;
                }
                omega0[celli] += w * omegaFace;

                if (wp.blending != "stepwise" || yPlus > wp.yPlusLam)
                {
                    G0[celli] += w * (nutw[facei] + nu) * magGradUw[facei] * Cmu25 * sqrtk
                        / (wp.kappa * y[facei]);
                }
            }
        }

        scalarField& omegaCells = omega_.primitiveFieldRef();
        for (const label celli : omegaWallCells_)
        {
            G[celli] = G0[celli];
            omegaCells[celli] = omega0[celli];
        }

        // Zero-gradient condition on the wall faces
        for (const OmegaWallPatch& wp : omegaWallPatches_)
        {
            omega_.boundaryFieldRef()[wp.patchi] ==
                scalarField(omega0, mesh.boundary()[wp.patchi].faceCells());
        }
    }

    void ResidualAssemblerSST::boundField(volScalarField& field, const dimensionedScalar& fieldMin, boolList& clipped)
    {
        clipped.setSize(field.size());
        forAll(clipped, celli)
        {
            clipped[celli] = field[celli] < fieldMin.value();
        }

        // Smooth floor b(x) = m + m softplus((x - m)/m): C-infinity and >= m everywhere, b(x) = x to within
        // 1e-4 relative for x >= 10 m (FOM values are, by the choice of m). Foam::bound, used by the FOM, is
        // discontinuous at x = 0 (negative cells jump to the neighbour average): harmless in a solver, but the
        // LSPG Jacobian is a finite difference of the residual, and LM gets stuck on the jump
        const scalar m = fieldMin.value();
        scalarField& f = field.primitiveFieldRef();
        forAll(f, celli)
        {
            const scalar z = (f[celli] - m) / m;
            f[celli] = m + m * (z > 30 ? z : std::log1p(std::exp(z)));
        }
        field.correctBoundaryConditions();

        // = is a no-op on fixedValue patches: Dirichlet values (e.g. k = 0 on walls) are left untouched, as in the FOM
        field.boundaryFieldRef() = max(field.boundaryField(), fieldMin.value());
    }

    void ResidualAssemblerSST::boundTurbulenceFields()
    {
        boundField(k_, kMin_, kClipped_);
        boundField(omega_, omegaMin_, omegaClipped_);

        // The wall-function patches are fixedValue here (setupOmegaWallFunctions), so boundField leaves their
        // face values as reconstructed, possibly <= 0: with zero shear (slip walls) nut = a1 k/max(a1 omega, ...)
        // would divide by zero. Give them the zero-gradient value the wall function imposes; the wall-function
        // values themselves are set in applyOmegaWallFunctions
        for (const OmegaWallPatch& wp : omegaWallPatches_)
        {
            omega_.boundaryFieldRef()[wp.patchi] == omega_.boundaryField()[wp.patchi].patchInternalField();
        }
    }

    void ResidualAssemblerSST::computePODCoeffsFromFields(Eigen::Ref<Eigen::VectorXd> stateCoeffs)
    {
        const label nKModes = kModes_.size();
        const label nOmegaModes = omegaModes_.size();

        Eigen::VectorXd kCoeffs = ITHACAutilities::getCoeffs(k_, kModes_);
        Eigen::VectorXd omegaCoeffs = ITHACAutilities::getCoeffs(omega_, omegaModes_);

        stateCoeffs.head(nKModes) = kCoeffs;
        stateCoeffs.segment(nKModes, nOmegaModes) = omegaCoeffs;
    }

    void ResidualAssemblerSST::removeInitialZeroField(const Eigen::VectorXd weights)
    {
        for (label i = 0; i < weights.size(); ++i)
        {
            k_ -= weights(i) * k0Fields_[i];
            omega_ -= weights(i) * omega0Fields_[i];
        }
    }

    void ResidualAssemblerSST::addInitialZeroField(const Eigen::VectorXd weights)
    {
        for (label i = 0; i < weights.size(); ++i)
        {
            k_ += weights(i) * k0Fields_[i];
            omega_ += weights(i) * omega0Fields_[i];
        }
    }

    void ResidualAssemblerSST::saveFieldsToDisk(const Time& runTime, const word folder) const
    {
      ITHACAstream::exportSolution(k_, name(runTime.time().value()), folder);
      ITHACAstream::exportSolution(omega_, name(runTime.time().value()), folder);
    }
}
}
