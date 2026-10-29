/*---------------------------------------------------------------------------*\
     ██╗████████╗██╗  ██╗ █████╗  ██████╗ █████╗       ███████╗██╗   ██╗
     ██║╚══██╔══╝██║  ██║██╔══██╗██╔════╝██╔══██╗      ██╔════╝██║   ██║
     ██║   ██║   ███████║███████║██║     ███████║█████╗█████╗  ██║   ██║
     ██║   ██║   ██╔══██║██╔══██║██║     ██╔══██║╚════╝██╔══╝  ╚██╗ ██╔╝
     ██║   ██║   ██║  ██║██║  ██║╚██████╗██║  ██║      ██║      ╚████╔╝
     ╚═╝   ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝╚═╝  ╚═╝      ╚═╝       ╚═══╝

 * In real Time Highly Advanced Computational Applications for Finite Volumes
 * Copyright (C) 2026 by the ITHACA-FV authors
-------------------------------------------------------------------------------
License
    This file is part of ITHACA-FV
    ITHACA-FV is free software: you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    ITHACA-FV is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
    GNU Lesser General Public License for more details.
    You should have received a copy of the GNU Lesser General Public License
    along with ITHACA-FV. If not, see <http://www.gnu.org/licenses/>.
Class
    ReducedFieldBase, ReducedField, ReducedFieldSet
Description
    A reduced field is an OpenFOAM field approximated as q = q_ref + sum_i a_i phi_i: it holds the field,
    its modes phi_i, the position of its coefficients a_i in the state vector, and the affine reference
    q_ref = sum_r w_r q_r (lifting functions or initial conditions of the offline runs, see AffineReference).
    ReducedFieldSet is the ordered list of the fields whose coefficients make up the state vector. The models
    only declare their fields: reconstruction, projection, affine reference and output are written once here.
SourceFiles
    ReducedField.C
\*---------------------------------------------------------------------------*/

#ifndef ReducedField_H
#define ReducedField_H

#include "fvCFD.H"
#include "Modes.H"
#include <Eigen/Dense>

/// Type-independent interface of a reduced field
class ReducedFieldBase
{
public:
    /// @param name Name of the field in the reduced problem (labels, residual blocks), e.g. "U"
    /// @param offset Position of the first coefficient of the field in the state vector
    /// @param nModes Number of modes (coefficients) of the field
    /// @param joint Whether the field belongs to the stacked POD, whose fields share one set of coefficients
    ReducedFieldBase(const word& name, const label offset, const label nModes, const bool joint): name_(name), offset_(offset), nModes_(nModes), joint_(joint)
    { }

    virtual ~ReducedFieldBase() = default;

    const word& name() const
    {
        return name_;
    }

    label offset() const
    {
        return offset_;
    }

    label nModes() const
    {
        return nModes_;
    }

    bool joint() const
    {
        return joint_;
    }

    //- Linear combination of the modes. The affine reference is not added
    virtual void reconstruct(const Eigen::Ref<const Eigen::VectorXd>& state) = 0;

    //- Writes the projection coefficients of the current field in state.segment(offset, nModes)
    virtual void project(Eigen::Ref<Eigen::VectorXd> state) = 0;

    //- Recomputes the affine reference sum_r w_r q_r. No-op for a field without reference fields
    virtual void setReferenceWeights(const Eigen::VectorXd& weights) = 0;

    //- Adds the affine reference to the field
    virtual void addReference() = 0;

    //- Subtracts the affine reference from the field
    virtual void removeReference() = 0;

    //- Gram matrix of the modes in the volume-weighted L2 product, F^T V F (summed over processors)
    virtual Eigen::MatrixXd gram() = 0;

    //- Galerkin projection of the current field on the modes, F^T V s (local to this processor)
    virtual Eigen::VectorXd galerkinProjection() = 0;

    //- Exports the field to folder/timeName
    virtual void write(const word& timeName, const word& folder) const = 0;

private:
    word name_;
    label offset_;
    label nModes_;
    bool joint_;
};


/// Reduced field of a given type. Instantiated in ReducedField.C for the typedefs below
template <class Type, template <class> class PatchField, class GeoMesh>
class ReducedField : public ReducedFieldBase
{
public:
    typedef GeometricField<Type, PatchField, GeoMesh> FieldType;
    typedef Modes<Type, PatchField, GeoMesh> ModesType;

    /// @param field The reduced field, owned by the reduced problem
    /// @param modes Modes of the field (copied)
    /// @param references Reference fields q_r of the affine reference (copied), empty for none
    /// @param correctBCs Whether reconstruct() corrects the boundary conditions
    ReducedField(
        const word& name,
        FieldType& field,
        const ModesType& modes,
        const PtrList<FieldType>& references,
        const label offset,
        const bool joint,
        const bool correctBCs = true);

    FieldType& field()
    {
        return field_;
    }

    ModesType& modes()
    {
        return modes_;
    }

    void reconstruct(const Eigen::Ref<const Eigen::VectorXd>& state) override;

    void project(Eigen::Ref<Eigen::VectorXd> state) override;

    void setReferenceWeights(const Eigen::VectorXd& weights) override;

    void addReference() override;

    void removeReference() override;

    Eigen::MatrixXd gram() override;

    Eigen::VectorXd galerkinProjection() override;

    void write(const word& timeName, const word& folder) const override;

private:
    FieldType& field_;
    ModesType modes_;
    PtrList<FieldType> references_;
    autoPtr<FieldType> reference_;
    bool correctBCs_;
};

typedef ReducedField<scalar, fvPatchField, volMesh> volScalarReducedField;
typedef ReducedField<vector, fvPatchField, volMesh> volVectorReducedField;
typedef ReducedField<scalar, fvsPatchField, surfaceMesh> surfaceScalarReducedField;


/// Ordered fields whose coefficients make up the state vector, x = [a_field0; a_field1; ...]
class ReducedFieldSet
{
public:
    //- Appends a field, which must start where the previous one ends. Takes ownership
    void append(ReducedFieldBase* field);

    label size() const
    {
        return fields_.size();
    }

    ReducedFieldBase& operator[](const label i)
    {
        return fields_[i];
    }

    const ReducedFieldBase& operator[](const label i) const
    {
        return fields_[i];
    }

    //- Index of the field with the given name, -1 if absent
    label find(const word& name) const;

    label stateSize() const;

    wordList names() const;

    void reconstruct(const Eigen::Ref<const Eigen::VectorXd>& state);

    void project(Eigen::Ref<Eigen::VectorXd> state);

    void write(const word& timeName, const word& folder) const;

private:
    PtrList<ReducedFieldBase> fields_;
};

#endif
