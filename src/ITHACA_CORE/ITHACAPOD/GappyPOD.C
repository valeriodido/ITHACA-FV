#include "GappyPOD.H"
#include "ITHACAstream.H"

// Small helper to extract the component of a field value, whether it's a scalar or a vector.
static scalar componentOf(const scalar& value, direction)
{
    return value;
}
template <class T>
static scalar componentOf(const T& v, direction dir)
{
    return v.component(dir);
}

template <class Type, template <class> class PatchField, class GeoMesh>
GappyPOD<Type, PatchField, GeoMesh>::GappyPOD(
    const PtrList<GeometricField<Type, PatchField, GeoMesh>>& modes,
    label nModes): modes_(modes),
                   nModes_(nModes < 0 ? modes.size() : nModes),
                   nCells_(modes.size() ? modes[0].primitiveField().size() : 0),
                   offlineComputed_(false)
{
    M_Assert(nModes_ <= modes.size(), "nModes must be less than or equal to the number of modes provided.");
    M_Assert(nCells_ > 0, "Modes must have non-zero size.");
    M_Assert(modes.size() > 0, "Modes list for the GappyPOD must not be empty.");
}

template <class Type, template <class> class PatchField, class GeoMesh>
void GappyPOD<Type, PatchField, GeoMesh>::setMask()
{
    labelHashSet seen;
    forAll(cellIDs_, k)
    {
        if (cellIDs_[k] < 0 || cellIDs_[k] >= nCells_)
            FatalErrorInFunction << "cell " << cellIDs_[k] << " out of range" << exit(FatalError);
        if (!seen.insert(cellIDs_[k]))
            FatalErrorInFunction << "duplicate cell " << cellIDs_[k] << exit(FatalError);
    }
    if (cellIDs_.size() * nComp_ < nModes_)
    {
      WarningInFunction << "fewer masked DOFs than modes: problem is underdetermined" << endl;
    }

    modesMasked_.resize(0, 0);
    pInvM_.resize(0, 0);
    offlineComputed_ = false;
}

template <class Type, template <class> class PatchField, class GeoMesh>
void GappyPOD<Type, PatchField, GeoMesh>::offline(
    const fileName& folder,
    bool loadIfExists)
{
    M_Assert(cellIDs_.size() > 0, "Mask must be set before calling offline() in Gappy POD.");

    if (offlineComputed_)
        return;

    if (loadIfExists)
    {
        FatalErrorInFunction << "Loading offline data from file is not yet implemented." << exit(FatalError);
    }

    if (!offlineComputed_)
    {
        buildMaskedModes();
        computePseudoInverse("BDCSVD");
    }
}

template <class Type, template <class> class PatchField, class GeoMesh>
GeometricField<Type, PatchField, GeoMesh> GappyPOD<Type, PatchField, GeoMesh>::reconstruct(
    const GeometricField<Type, PatchField, GeoMesh>& sample,
    word fieldName) const
{
    return reconstructToField(reconstructCoefficients(gather(sample)), sample.instance(), fieldName);
}

template <class Type, template <class> class PatchField, class GeoMesh>
Eigen::VectorXd GappyPOD<Type, PatchField, GeoMesh>::reconstruct(
    const Eigen::VectorXd& sample) const
{
    FatalErrorInFunction << "Reconstruction to Eigen::VectorXd is not yet implemented." << exit(FatalError);
}

template <class Type, template <class> class PatchField, class GeoMesh>
Eigen::VectorXd GappyPOD<Type, PatchField, GeoMesh>::reconstructCoefficients(
    const Eigen::VectorXd& sample) const
{
    // TODO: Evaluate the impact of the assertions in a hot loop such as GNAT.
    M_Assert(offlineComputed_,
        "Offline step must be performed before calling reconstructCoefficients() in Gappy POD.");
    M_Assert(
        sample.size() == cellIDs_.size() * nComp_,
        "Sample size does not match the number of masked DOFs.");

    return pInvM_ * sample;
}

template <class Type, template <class> class PatchField, class GeoMesh>
GeometricField<Type, PatchField, GeoMesh> GappyPOD<Type, PatchField, GeoMesh>::reconstructToField(
    const Eigen::VectorXd& coefficients, word instance, word name) const
{
    M_Assert(offlineComputed_,
        "Offline step must be performed before calling reconstructToField() in Gappy POD.");
    M_Assert(coefficients.size() == nModes_,
        "Coefficients size does not match the number of modes.");

    GeometricField<Type, PatchField, GeoMesh> reconstructedField(
        IOobject(name.empty() ? "gappyReconstructed" : name,
            instance,
            modes_[0].mesh(),
            IOobject::NO_READ,
            IOobject::NO_WRITE),
        modes_[0] * coefficients(0));

    for (label modeIdx = 1; modeIdx < nModes_; ++modeIdx)
    {
        reconstructedField += modes_[modeIdx] * coefficients(modeIdx);
    }
    return reconstructedField;
}

template <class Type, template <class> class PatchField, class GeoMesh>
Eigen::VectorXd GappyPOD<Type, PatchField, GeoMesh>::gather(
    const GeometricField<Type, PatchField, GeoMesh>& fullField) const
{
    M_Assert(cellIDs_.size() > 0, "Mask must be set before calling sample() in Gappy POD.");
    M_Assert(fullField.primitiveField().size() == nCells_, "Full field size does not match the number of cells in the modes.");

    Eigen::VectorXd sampled(cellIDs_.size() * nComp_);
    const auto& field = fullField.primitiveField();
    forAll(cellIDs_, i)
    {
        for (int comp = 0; comp < nComp_; ++comp)
        {
            sampled(i * nComp_ + comp) = componentOf(field[cellIDs_[i]], comp);
        }
    }
    return sampled;
}

template <class Type, template <class> class PatchField, class GeoMesh>
bool GappyPOD<Type, PatchField, GeoMesh>::isOfflineComputed() const
{
    return offlineComputed_;
}

template <class Type, template <class> class PatchField, class GeoMesh>
volScalarField GappyPOD<Type, PatchField, GeoMesh>::getSampleMesh(word name) const
{
    M_Assert(cellIDs_.size() > 0, "Mask must be set before calling getSampleMesh() in Gappy POD.");
    volScalarField sampleMesh(
        IOobject(name,
            modes_[0].instance(),
            modes_[0].mesh(),
            IOobject::NO_READ,
            IOobject::NO_WRITE),
        modes_[0].mesh(),
        dimensionedScalar("zero", dimless, 0.0));

    forAll(cellIDs_, i)
    {
        sampleMesh[cellIDs_[i]] = 1.0;
    }
    return sampleMesh;
}

template <class Type, template <class> class PatchField, class GeoMesh>
double GappyPOD<Type, PatchField, GeoMesh>::getConditionNumber() const
{
    M_Assert(offlineComputed_,
        "Offline step must be performed before calling getConditionNumber() in Gappy POD.");
    return conditionNumber_;
}

template <class Type, template <class> class PatchField, class GeoMesh>
void GappyPOD<Type, PatchField, GeoMesh>::randomSample(label nSamples, label seed)
{
    labelList all(identity(nCells_));
    Random rndGen(seed);
    for (label i = nCells_ - 1; i > 0; i--)
    {
        label j = rndGen.position<label>(0, i);
        Foam::Swap(all[i], all[j]);
    }
    all.resize(nSamples);
    cellIDs_ = all;
}

template <class Type, template <class> class PatchField, class GeoMesh>
void GappyPOD<Type, PatchField, GeoMesh>::buildMaskedModes()
{
    modesMasked_.resize(cellIDs_.size() * nComp_, nModes_);
    for (label modeIdx = 0; modeIdx < nModes_; ++modeIdx)
    {
        const auto& mode = modes_[modeIdx].primitiveField();
        forAll(cellIDs_, i)
        {
            for (int comp = 0; comp < nComp_; ++comp)
            {
                modesMasked_(i * nComp_ + comp, modeIdx) = componentOf(mode[cellIDs_[i]], comp);
            }
        }
    }
}

template <class Type, template <class> class PatchField, class GeoMesh>
void GappyPOD<Type, PatchField, GeoMesh>::computePseudoInverse(word method, scalar tolerance)
{
    if (method == "COD")
    {
        Eigen::CompleteOrthogonalDecomposition<Eigen::MatrixXd> cod(modesMasked_);
        cod.setThreshold(tolerance);
        pInvM_ = cod.pseudoInverse();
        offlineComputed_ = true;

        Eigen::JacobiSVD<Eigen::MatrixXd> svd(modesMasked_, Eigen::ComputeThinU | Eigen::ComputeThinV);
        conditionNumber_ = svd.singularValues()(0) / svd.singularValues()(svd.singularValues().size() - 1);

    } else if (method == "JacobiSVD")
    {
        Eigen::JacobiSVD<Eigen::MatrixXd> svd(modesMasked_, Eigen::ComputeThinU | Eigen::ComputeThinV);
        const auto& singularValues = svd.singularValues();
        Eigen::VectorXd singularValuesInv(singularValues.size());
        for (int i = 0; i < singularValues.size(); ++i)
        {
            if (singularValues(i) > tolerance)
                singularValuesInv(i) = 1.0 / singularValues(i);
            else
                singularValuesInv(i) = 0.0;
        }
        pInvM_ = svd.matrixV() * singularValuesInv.asDiagonal() * svd.matrixU().transpose();
        offlineComputed_ = true;

        conditionNumber_ = singularValues(0) / singularValues(singularValues.size() - 1);
    } else if (method == "BDCSVD")
    {
        Eigen::BDCSVD<Eigen::MatrixXd> svd(modesMasked_, Eigen::ComputeThinU | Eigen::ComputeThinV);
        const auto& singularValues = svd.singularValues();
        Eigen::VectorXd singularValuesInv(singularValues.size());
        for (int i = 0; i < singularValues.size(); ++i)
        {
            if (singularValues(i) > tolerance)
                singularValuesInv(i) = 1.0 / singularValues(i);
            else
                singularValuesInv(i) = 0.0;
        }
        pInvM_ = svd.matrixV() * singularValuesInv.asDiagonal() * svd.matrixU().transpose();
        offlineComputed_ = true;

        conditionNumber_ = singularValues(0) / singularValues(singularValues.size() - 1);
    } else
    {
        FatalErrorInFunction << "Unknown method for computing pseudo-inverse: " << method << exit(FatalError);
    }
    Info << "GappyPOD: Condition number of the inverted masked modes matrix: " << conditionNumber_ << endl;
}

// Template specializations
template class GappyPOD<scalar, fvPatchField, volMesh>;
template class GappyPOD<vector, fvPatchField, volMesh>;
