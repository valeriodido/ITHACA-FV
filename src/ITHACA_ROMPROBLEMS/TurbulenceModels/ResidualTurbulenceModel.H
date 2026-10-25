// Implements a base class for turbulence models that compute the residual of the turbulence equations.
// To be used in LSPG ROMs.
// Specific turbulence models should inherit from this class

#ifndef RESIDUALTURBULENCEMODEL_H
#define RESIDUALTURBULENCEMODEL_H

#include "fvCFD.H"
#include "fvOptions.H"
#include "tmp.H"
#include <Eigen/Dense>

namespace Foam
{
namespace ITHACATurbulenceModels
{
    class ResidualEddyViscosityTurbulenceModel
    {
    public:
        struct SharedFields {
            const dimensionedScalar& nu;
            volScalarField& nut;
            const volVectorField& U;
            const surfaceScalarField& phi;
        };

        struct ReducedModes 
        {
          virtual ~ReducedModes() = default;
        };

        ResidualEddyViscosityTurbulenceModel(
            const fvMesh& mesh,
            const Time& runTime,
            const SharedFields& sharedFields);

        virtual ~ResidualEddyViscosityTurbulenceModel() = default;

        virtual const wordList& transportedFields() const = 0;

        virtual void correctNut() = 0;

        // Returns one matrix per transported field, in the same order as transportedFields()
        virtual PtrList<fvScalarMatrix> assembleEquations(fv::options& fvOptions) = 0;

        // Linear Eddy Viscosity stress divergence term
        tmp<fvVectorMatrix> divDevReff() const;

        // Base method to reconstruct from POD
        virtual void reconstructTurbulenceFields(const Eigen::VectorXd& stateCoeffs) = 0;

        // Base method to bound turbulence fields to ensure physicality
        virtual void boundTurbulenceFields() = 0;

        // Base method to compute the initial POD coefficients from fields
        virtual void computePODCoeffsFromFields(Eigen::Ref<Eigen::VectorXd> stateCoeffs) = 0;

        // Removes the initial 0 field from the turbulent quantities - See consistent POD from Carlberg
        virtual void removeInitialZeroField(const Eigen::VectorXd weights) = 0;

        // Adds the initial 0 field to the turbulent quantities - See consistent POD from Carlberg
        virtual void addInitialZeroField(const Eigen::VectorXd weights) = 0;

        // Saves the current fields to disk for post-processing
        virtual void saveFieldsToDisk(const Time& runTime, const word folder) const = 0;

    protected:
        const fvMesh& mesh_;
        const dimensionedScalar& nu_;
        volScalarField& nut_;
        const volVectorField& U_;
        const surfaceScalarField& phi_;
    };
}
}

#endif
