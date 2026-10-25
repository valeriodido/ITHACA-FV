#include "ITHACAkOmegaSST.H"

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
        omegaMin_("omegaMin", dimless / dimTime, SMALL)
    {
        if (mesh.foundObject<volScalarField::Internal>(GName_))
        {
            FatalErrorInFunction
                << GName_ << " is already registered; reuse it instead." << exit(FatalError);
        }

        G_.reset(
            new volScalarField::Internal(
                IOobject(
                    GName_, mesh.time().timeName(), mesh,
                    IOobject::NO_READ, IOobject::NO_WRITE, true // registered
                    ),
                mesh,
                dimensionedScalar("G", dimensionSet(0, 2, -3, 0, 0, 0, 0), 0.0)));
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
        omega_.boundaryFieldRef().updateCoeffs();

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
    }

    void ResidualAssemblerSST::boundTurbulenceFields()
    {

        const dimensionedScalar kMin("kMin", k_.dimensions(), SMALL);

        k_.internalFieldRef() = Foam::max(k_.internalFieldRef(), kMin);
        omega_.internalFieldRef() = Foam::max(omega_.internalFieldRef(), omegaMin_);
        k_.correctBoundaryConditions();
        omega_.correctBoundaryConditions();

        forAll(k_.boundaryFieldRef(), patchi)
        {
            scalarField& kPatch = k_.boundaryFieldRef()[patchi];
            scalarField& omegaPatch = omega_.boundaryFieldRef()[patchi];

            forAll(kPatch, facei)
            {
                kPatch[facei] = max(kPatch[facei], scalar(SMALL));
                omegaPatch[facei] = max(omegaPatch[facei], scalar(SMALL));
            }
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