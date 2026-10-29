#include "ReducedField.H"
#include "ITHACAcoeffsMass.H"
#include "ITHACAstream.H"
#include "ITHACAassert.H"

// * * * * * * * * * * * * * * * * ReducedField  * * * * * * * * * * * * * * //

template <class Type, template <class> class PatchField, class GeoMesh>
ReducedField<Type, PatchField, GeoMesh>::ReducedField(
    const word& name,
    FieldType& field,
    const ModesType& modes,
    const PtrList<FieldType>& references,
    const label offset,
    const bool joint,
    const bool correctBCs): ReducedFieldBase(name, offset, modes.size(), joint),
                            field_(field),
                            modes_(modes),
                            references_(references),
                            correctBCs_(correctBCs)
{ }

template <class Type, template <class> class PatchField, class GeoMesh>
void ReducedField<Type, PatchField, GeoMesh>::reconstruct(const Eigen::Ref<const Eigen::VectorXd>& state)
{
    modes_.reconstructInPlace(field_, state.segment(offset(), nModes()), field_.name(), correctBCs_);
}

template <class Type, template <class> class PatchField, class GeoMesh>
void ReducedField<Type, PatchField, GeoMesh>::project(Eigen::Ref<Eigen::VectorXd> state)
{
    state.segment(offset(), nModes()) = ITHACAutilities::getCoeffs(field_, modes_);
}

template <class Type, template <class> class PatchField, class GeoMesh>
void ReducedField<Type, PatchField, GeoMesh>::setReferenceWeights(const Eigen::VectorXd& weights)
{
    if (references_.empty())
    {
        return;
    }
    M_Assert(weights.size() > 0 && weights.size() <= references_.size(),
        "ReducedField: more reference weights than reference fields.");
    // Computed once per change of the weights (once per time step), instead of at every residual evaluation
    tmp<FieldType> tsum(weights(0) * references_[0]);
    for (label r = 1; r < weights.size(); r++)
    {
        tsum = tsum + weights(r) * references_[r];
    }
    reference_.reset(new FieldType(
        IOobject(field_.name() + "Reference", field_.time().timeName(), field_.db(),
            IOobject::NO_READ, IOobject::NO_WRITE, IOobject::NO_REGISTER),
        tsum));
}

template <class Type, template <class> class PatchField, class GeoMesh>
void ReducedField<Type, PatchField, GeoMesh>::addReference()
{
    if (reference_)
    {
        field_ += reference_();
    }
}

template <class Type, template <class> class PatchField, class GeoMesh>
void ReducedField<Type, PatchField, GeoMesh>::removeReference()
{
    if (reference_)
    {
        field_ -= reference_();
    }
}

template <class Type, template <class> class PatchField, class GeoMesh>
Eigen::MatrixXd ReducedField<Type, PatchField, GeoMesh>::gram()
{
    return ITHACAutilities::getMassMatrix(modes_);
}

template <class Type, template <class> class PatchField, class GeoMesh>
Eigen::VectorXd ReducedField<Type, PatchField, GeoMesh>::galerkinProjection()
{
    return modes_.project(field_, nModes(), "G");
}

template <class Type, template <class> class PatchField, class GeoMesh>
void ReducedField<Type, PatchField, GeoMesh>::write(const word& timeName, const word& folder) const
{
    ITHACAstream::exportSolution(field_, timeName, folder);
}

template class ReducedField<scalar, fvPatchField, volMesh>;
template class ReducedField<vector, fvPatchField, volMesh>;
template class ReducedField<scalar, fvsPatchField, surfaceMesh>;

// * * * * * * * * * * * * * * * * ReducedFieldSet * * * * * * * * * * * * * //

void ReducedFieldSet::append(ReducedFieldBase* field)
{
    M_Assert(field->offset() == stateSize(), "ReducedFieldSet: the fields must be contiguous in the state.");
    fields_.push_back(field);
}

label ReducedFieldSet::find(const word& name) const
{
    forAll(fields_, i)
    {
        if (fields_[i].name() == name)
        {
            return i;
        }
    }
    return -1;
}

label ReducedFieldSet::stateSize() const
{
    return fields_.empty() ? 0 : fields_.back().offset() + fields_.back().nModes();
}

wordList ReducedFieldSet::names() const
{
    wordList n(fields_.size());
    forAll(fields_, i)
    {
        n[i] = fields_[i].name();
    }
    return n;
}

void ReducedFieldSet::reconstruct(const Eigen::Ref<const Eigen::VectorXd>& state)
{
    for (ReducedFieldBase& f : fields_)
    {
        f.reconstruct(state);
    }
}

void ReducedFieldSet::project(Eigen::Ref<Eigen::VectorXd> state)
{
    for (ReducedFieldBase& f : fields_)
    {
        f.project(state);
    }
}

void ReducedFieldSet::write(const word& timeName, const word& folder) const
{
    for (const ReducedFieldBase& f : fields_)
    {
        f.write(timeName, folder);
    }
}
