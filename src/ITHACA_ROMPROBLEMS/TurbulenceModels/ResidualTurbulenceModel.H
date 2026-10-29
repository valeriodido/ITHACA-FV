// Base class of the eddy-viscosity closures of the LSPG ROMs: they assemble the residual of their transport
// equations from the current fields, without solving them. The closure only describes the physics; the
// reduced representation of its transported fields (modes, affine reference, output) is handled by the ROM.
// Specific turbulence models should inherit from this class

#ifndef RESIDUALTURBULENCEMODEL_H
#define RESIDUALTURBULENCEMODEL_H

#include "fvCFD.H"
#include "fvOptions.H"
#include "tmp.H"

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

        // Called after the transported fields were set from the reduced state: restores their Dirichlet
        // values (mode reconstruction overwrites them) and bounds them to ensure physicality
        virtual void constrainTransportedFields() = 0;

        // Cells clipped by the last constrainTransportedFields call, one mask per transported field
        // in the order of transportedFields(). Used by the solver diagnostics; empty if not tracked
        virtual List<boolList> clippedCells() const
        {
            return List<boolList>();
        }

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
