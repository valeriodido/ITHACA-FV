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
    AffineReference
Description
    Affine reference q_ref(mu) = sum_r w_r q_r of the reduced fields (see AffineReferenceType):
      Lift             : w = the boundary values, q_r the lifting functions
      InitialCondition : w = inverse distance weights of the offline runs, q_r their initial conditions
    The reference fields q_r are held by each ReducedField. The reference depends only on the boundary
    values, so it is recomputed only when they change (once per time step), not at every residual evaluation.
SourceFiles
    AffineReference.C
\*---------------------------------------------------------------------------*/

#ifndef AffineReference_H
#define AffineReference_H

#include "LSPGSettings.H"
#include "InverseDistanceWeights.H"
#include "ReducedField.H"

/* 
* @brief Affine reference for the reduced fields.
* At the moment is very bare-bones, and we have either a lifting options (for bcs) or an initial condition option.
* Eventually I think we can support both at the same time, but for now it is like this.
*/
class AffineReference
{
public:
    AffineReference() = default;

    /// @param mu Parameters of the offline runs, one column per run (inverse distance weighting)
    AffineReference(const AffineReferenceSettings& settings, const Eigen::MatrixXd& mu);

    AffineReferenceType type() const
    {
        return settings_.type;
    }

    //- Weights of the offline runs at the given parameters
    const InverseDistanceWeights& parametricWeights() const
    {
        return idw_;
    }

    //- Weights w of the reference fields at the boundary values bcs (empty without a reference)
    Eigen::VectorXd weights(const Eigen::VectorXd& bcs) const;

    //- Sets the reference of the fields at the boundary values bcs, if they changed since the last call
    void update(const Eigen::VectorXd& bcs, UPtrList<ReducedFieldBase>& fields);

private:
    AffineReferenceSettings settings_;

    InverseDistanceWeights idw_;

    //- Boundary values of the current reference
    Eigen::VectorXd bcs_;

    bool valid_ = false;
};

#endif
