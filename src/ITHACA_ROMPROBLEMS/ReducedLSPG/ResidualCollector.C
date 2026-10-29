#include "ResidualCollector.H"
#include "zeroGradientFvPatchFields.H"

template <class Type>
void LSPGResidualOutput::writeMatrixResidual(
    const word& name,
    const fvMatrix<Type>& eqn,
    const Field<Type>& R,
    const fvMesh& mesh,
    const word& folder,
    const bool perUnitVolume)
{
    typedef GeometricField<Type, fvPatchField, volMesh> fieldType;

    const dimensionSet dims =
        perUnitVolume ? eqn.dimensions() / dimVolume : eqn.dimensions();

    fieldType res(
        IOobject(
            name,
            mesh.time().timeName(),
            mesh,
            IOobject::NO_READ,
            IOobject::NO_WRITE),
        mesh,
        dimensioned<Type>("zero", dims, Zero),
        zeroGradientFvPatchField<Type>::typeName);

    res.primitiveFieldRef() = R;
    if (perUnitVolume)
    {
        res.primitiveFieldRef() /= mesh.V();
    }
    res.correctBoundaryConditions();

    exportField(res, folder);
}

template void LSPGResidualOutput::writeMatrixResidual<scalar>(
    const word&, const fvMatrix<scalar>&, const Field<scalar>&, const fvMesh&, const word&, const bool);
template void LSPGResidualOutput::writeMatrixResidual<vector>(
    const word&, const fvMatrix<vector>&, const Field<vector>&, const fvMesh&, const word&, const bool);


void ResidualCollector::add(const label block, const surfaceScalarField& R)
{
    residual_.rawBlock(block) = Eigen::Map<const Eigen::VectorXd>(R.primitiveField().cdata(), R.primitiveField().size());

    if (writeFields_)
    {
        surfaceScalarField res(outputName(block), R / mesh_.magSf());
        LSPGResidualOutput::exportField(res, outputFolder_);
    }
}

Eigen::VectorXd ResidualCollector::jacobiRowWeights(const scalarField& A, const label nComponents) const
{
    const scalarField& V = mesh_.V().field();
    Eigen::VectorXd w(nComponents * A.size());
    forAll(A, celli)
    {
        const scalar a = max(mag(A[celli]) * V[celli], VSMALL);
        w.segment(nComponents * celli, nComponents).setConstant(Foam::sqrt(V[celli]) / a);
    }
    return w;
}
