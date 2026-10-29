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
    InverseDistanceWeights
Description
    Weights of the offline runs at a new parameter, w_r ~ 1/d_r^exponent, with the distance d_r computed
    on the parameters normalised by their range over the training runs (so that they weigh equally).
SourceFiles
    InverseDistanceWeights.C
\*---------------------------------------------------------------------------*/

#ifndef InverseDistanceWeights_H
#define InverseDistanceWeights_H

#include "fvCFD.H"
#include <Eigen/Dense>

class InverseDistanceWeights
{
public:
    InverseDistanceWeights() = default;

    /// @param samples Parameters of the offline runs, one column per run
    /// @param exponent Exponent of the distance
    /// @param neighbours Only the nearest runs are used (all the runs at the cut-off distance are kept); 0: all
    InverseDistanceWeights(const Eigen::MatrixXd& samples, const scalar exponent, const label neighbours);

    /// @brief Weights of the offline runs at the given parameters, summing to 1
    Eigen::VectorXd operator()(const Eigen::VectorXd& parameters) const;

    label nSamples() const
    {
        return samples_.cols();
    }

private:
    Eigen::VectorXd normalised(const Eigen::VectorXd& x) const;

    Eigen::MatrixXd samples_;
    Eigen::VectorXd lower_;
    Eigen::VectorXd range_;
    //- Normalised parameters of the offline runs
    Eigen::MatrixXd normalisedSamples_;
    scalar exponent_ = 1.0;
    label neighbours_ = 0;
};

#endif
