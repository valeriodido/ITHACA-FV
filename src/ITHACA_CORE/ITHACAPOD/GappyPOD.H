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
    GappyPOD
Description
    Implementation of Gappy POD class to be used both as a standalone
    technique (for example for CFD post-processing) and as a tool in
     reduced order modeling.
SourceFiles
    GappyPOD.C
\*---------------------------------------------------------------------------*/

/// \file
/// Header file of the GappyPOD class.
/// \dir
/// Directory containing the header and source files for the GappyPOD class.

#ifndef GappyPOD_H
#define GappyPOD_H

#include "fvCFD.H"
#include <Eigen/Dense>

template <class Type, template <class> class PatchField, class GeoMesh>
class GappyPOD
{
public:
    /// @brief Constructor, that takes in a list of modes.
    /// We avoid copying the modes, and instead store a reference to them.
    explicit GappyPOD(const PtrList<GeometricField<Type, PatchField, GeoMesh>>& modes, label nModes = -1);

    /// @brief Sets the sample matrix/mask matrix, which is a sparse matrix that selects the sample points from the full field.
    void setMask();

    /// @brief Computes the pseudo-inverse of the matrix M.
    /// This is used in a least-squares sense to solve for the coefficients of the modes given the sample points.
    /// Ma = b. M = <modes, modes>_maskedSpace, b = <sample, modes>_maskedSpace
    void offline(const fileName& folder, bool loadIfExists = true);

    /// @brief Reconstruct the field to OpenFOAM format
    GeometricField<Type, PatchField, GeoMesh> reconstruct(
        const GeometricField<Type, PatchField, GeoMesh>& sample,
        word fieldName = "") const;

    /// @brief Reconstruct the field to Eigen format
    Eigen::VectorXd reconstruct(
        const Eigen::VectorXd& sample) const;

    /// @brief Reconstructs the coefficients only, without reconstructing the full field.
    Eigen::VectorXd reconstructCoefficients(
        const Eigen::VectorXd& sample) const;

    /// @brief Reconstruct to field. Check, ITHACAutilities might already have a function for this.
    GeometricField<Type, PatchField, GeoMesh> reconstructToField(
        const Eigen::VectorXd& coefficients,
        word instance,
        word name = "") const;

    /// @brief Extracts the measured values from the full field, according to the mask matrix.
    Eigen::VectorXd gather(
        const GeometricField<Type, PatchField, GeoMesh>& fullField) const;

    /// @brief  Returns whether the offline step has been performed and the pseudo-inverse is available.
    /// @return Boolean indicating whether the offline step has been performed and the pseudo-inverse is available.
    bool isOfflineComputed() const;

    /// @brief Returns a volScalarField containing the sample points
    /// @param word name: Name of the field to be created
    volScalarField getSampleMesh(word name = "gappySampleMesh") const;

    /// @brief Returns the condition number of the inverted matrix
    double getConditionNumber() const;

    /// @brief Returns a randomly sampled subset of the mesh via a labelList
    /// @param label nSamples: Number of samples to extract
    /// @param label seed: Seed for the random number generator
    void randomSample(label nSamples, label seed = 42);

private:
    /// @brief Builds the masked modes
    void buildMaskedModes();

    /// @brief Computes the pseudo-inverse of the matrix M.
    /// Here we use the SVD, I think it is more robust since we might have very small amount of probes.
    /// @param method Method to use for computing the pseudo-inverse. Default is "JacobiSVD". "COD" also available, uses Eigen's Complete Orthogonal Decomposition.
    /// @param tolerance Tolerance for the pseudo-inverse computation. Default is 1e-6.
    void computePseudoInverse(word method = "COD", scalar tolerance = 1e-6);

    /// @brief Non-owning reference to the modez
    const PtrList<GeometricField<Type, PatchField, GeoMesh>>& modes_;

    /// @brief OpenFOAM cell IDs of the sample points.
    labelList cellIDs_;

    /// @brief Cached matrix M
    Eigen::MatrixXd matrixM_;

    /// @brief Cached pseudo-inverse of matrix M
    Eigen::MatrixXd pInvM_;

    /// @brief Masked modes matrix
    Eigen::MatrixXd modesMasked_;

    /// @brief Number of modes
    label nModes_;

    /// @brief Number of sample points
    label nSamplePoints_;

    /// @brief Number of cells in the mesh
    label nCells_;

    /// @brief Whether the offline step has been performed and the pseudo-inverse is available.
    bool offlineComputed_ = false;

    /// @brief Condition number of the inverted matrix
    double conditionNumber_ = 0.0;

    // Number of components in the field
    static constexpr int nComp_ = pTraits<Type>::nComponents;
};

#endif
