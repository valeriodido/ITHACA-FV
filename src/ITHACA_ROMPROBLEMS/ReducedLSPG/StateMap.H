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
    StateMap
Description
    Map x = E c from the reduced coordinates c solved for by LM to the state vector x of the field
    coefficients. With a stacked POD, the fields of the stacked group share one set of coefficients: E copies
    each shared coordinate into the slot of every field of the group. The reduced coordinates are
    [shared coefficients; coefficients of the other fields, in the order of the field set].
    E has one nonzero (1) per row, so it is stored as the coordinate of each state entry.
    This was born because I was experimenting with a POD for each field, in the spirit of Galerkin.
SourceFiles
    StateMap.C
\*---------------------------------------------------------------------------*/

#ifndef StateMap_H
#define StateMap_H

#include "ReducedField.H"

class StateMap
{
public:
    //- Identity map
    StateMap() = default;

    //- Map of a field set: identity if no field is in the stacked POD
    explicit StateMap(const ReducedFieldSet& fields);

    bool identity() const
    {
        return source_.empty();
    }

    label nCoordinates(const label stateSize) const
    {
        return identity() ? stateSize : nCoordinates_;
    }

    //- State x = E c
    Eigen::VectorXd expand(const Eigen::VectorXd& coordinates) const;

    //- Coordinates of a state, (E^T E)^-1 E^T x: the copies of a shared coefficient are averaged
    Eigen::VectorXd reduce(const Eigen::VectorXd& state) const;

private:
    //- Coordinate of each state entry
    labelList source_;

    //- Number of state entries of each coordinate
    labelList copies_;

    label nCoordinates_ = 0;
};

#endif
