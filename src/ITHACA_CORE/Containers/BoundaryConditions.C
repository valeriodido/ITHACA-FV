/*---------------------------------------------------------------------------*\
v/*---------------------------------------------------------------------------*\
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

\*---------------------------------------------------------------------------*/

/// \file
/// Source file of the BoundaryConditions class

#include "BoundaryConditions.H"
#include "ITHACAstream.H"

// * * * * * * * * * * * * * * * Constructors * * * * * * * * * * * * * * * * //

// Constructor
BoundaryConditions::BoundaryConditions(const Eigen::MatrixXd& BCMatrix,  word timeDepMethod):
    BCMatrix_(BCMatrix),
    timeDepMethod_(timeDepMethod)
{
    M_Assert(BCMatrix_.cols() > 1, "BCMatrix must have at least two columns: one for time and at least one for BC values.");
    timeDep_ = false;

    if (BCMatrix.rows() > 1)
    {
        timeDep_ = true;
        timestepsBC_ = BCMatrix.col(0);
    }

    currentBC_ = BCMatrix.row(0).tail(BCMatrix_.cols() - 1);
    nBCs_ = currentBC_.size();
}

BoundaryConditions::BoundaryConditions(const Eigen::VectorXd& BCMatrix):
    BCMatrix_(BCMatrix)
{
    timeDep_ = false;

    if (BCMatrix.rows() > 1)
    {
        timeDep_ = true;
        timestepsBC_ = BCMatrix.col(0);
    }

    currentBC_ = BCMatrix.row(0).tail(BCMatrix_.cols() - 1);
    nBCs_ = currentBC_.size();
}

BoundaryConditions::BoundaryConditions(const word BCFile, word timeDepMethod):
    timeDepMethod_(timeDepMethod)
{
    BCMatrix_ = ITHACAstream::readMatrix(BCFile);

    if (BCMatrix_.rows() > 1)
    {
        timeDep_ = true;
        timestepsBC_ = BCMatrix_.col(0);
    }

    currentBC_ = BCMatrix_.row(0).tail(BCMatrix_.cols() - 1);
    nBCs_ = currentBC_.size();
}

// Methods
void BoundaryConditions::updateTimeDependentBC(const scalar currentTime)
{
    if (timeDep_)
    {
        if (timeDepMethod_ == "linear")
        {
            if (currentTime >= timestepsBC_.tail(1).value())
            {
                currentBC_ = BCMatrix_.row(BCMatrix_.rows() - 1).tail(BCMatrix_.cols() - 1);
            } else if (currentTime <= timestepsBC_(0))
            {
                currentBC_ = BCMatrix_.row(0).tail(BCMatrix_.cols() - 1);
            } else
            {
                auto it = std::upper_bound(timestepsBC_.data(),
                    timestepsBC_.data() + timestepsBC_.size(),
                    currentTime);
                Eigen::Index k = std::distance(timestepsBC_.data(), it);

                // Safety check - Should not happen due to how the solver in the ROM is written, but still...
                if (k > 0)
                {
                    double t0 = timestepsBC_(k - 1);
                    double t1 = timestepsBC_(k);
                    double alpha = (currentTime - t0) / (t1 - t0);
                    Eigen::VectorXd val0 = BCMatrix_.row(k - 1).tail(
                        BCMatrix_.cols() - 1);
                    Eigen::VectorXd val1 = BCMatrix_.row(k).tail(BCMatrix_.cols() -
                        1);
                    currentBC_ = val0 + alpha * (val1 - val0);
                }
            }
        }
    }
}

double BoundaryConditions::linearInterpolate(const double t0, const double t1,
    const double v0, const double v1, const scalar currentTime)
{
    return v0 + (v1 - v0) * (currentTime - t0) / (t1 - t0);
}

void BoundaryConditions::correctLiftingCoeffs(Eigen::VectorXd& y,
    const int segment_start)
{
    y.segment(segment_start, nBCs_) = currentBC_;
}

// ************************************************************************ //
