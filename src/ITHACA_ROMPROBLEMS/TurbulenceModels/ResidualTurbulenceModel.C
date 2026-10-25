#include "ResidualTurbulenceModel.H"

namespace Foam
{
namespace ITHACATurbulenceModels
{
    ResidualEddyViscosityTurbulenceModel::ResidualEddyViscosityTurbulenceModel(
        const fvMesh& mesh,
        const Time& runTime,
        const SharedFields& sharedFields): mesh_(mesh),
                                           nu_(sharedFields.nu),
                                           nut_(sharedFields.nut),
                                           U_(sharedFields.U),
                                           phi_(sharedFields.phi)
    {
    }

    tmp<fvVectorMatrix> ResidualEddyViscosityTurbulenceModel::divDevReff() const
    {
        const volScalarField nuEff("nuEff", nu_ + nut_);
        return (
            -fvc::div(nuEff * dev2(T(fvc::grad(U_)))) 
            - fvm::laplacian(nuEff, U_)
        );
    }
}
}