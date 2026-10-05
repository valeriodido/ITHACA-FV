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
BoundaryConditions::BoundaryConditions(const Eigen::MatrixXd& velocityBC,
    const Eigen::MatrixXd& temperatureBC,
    word timeDepMethod):
    temperatureBCMatrix_(temperatureBC),
    velocityBCMatrix_(velocityBC),
    timeDepMethod_(timeDepMethod)
{
    temperatureTimeDep_ = false;
    velocityTimeDep_ = false;

    if (temperatureBC.rows() > 1)
    {
        temperatureTimeDep_ = true;
        timestepsTempBC_ = temperatureBC.col(0);
    }

    if (velocityBC.rows() > 1)
    {
        velocityTimeDep_ = true;
        timestepsVelBC_ = velocityBC.col(0);
    }

    currentTemperatureBC = temperatureBC.row(0).tail(temperatureBC.cols() - 1);
    currentVelocityBC = velocityBC.row(0).tail(velocityBC.cols() - 1);
}

BoundaryConditions::BoundaryConditions(const Eigen::VectorXd& velocityBC,
    const Eigen::VectorXd& temperatureBC):
    temperatureBCMatrix_(temperatureBC),
    velocityBCMatrix_(velocityBC)
{
    temperatureTimeDep_ = false;
    velocityTimeDep_ = false;
    std::cout << "### DEBUG --- Initializing BoundaryConditions with Eigen::VectorXd. Assuming time-independent BCs."
              << std::endl;
    currentTemperatureBC = temperatureBC.tail(temperatureBC.size() - 1);
    currentVelocityBC = velocityBC.tail(velocityBC.size() - 1);
    std::cout << "### DEBUG --- Current temperature BC: " << currentTemperatureBC.transpose() << std::endl;
}

BoundaryConditions::BoundaryConditions(const word velocityBCFile,
    const word temperatureBCFile, word timeDepMethod):
    timeDepMethod_(timeDepMethod)
{
    velocityBCMatrix_ = ITHACAstream::readMatrix(velocityBCFile);
    temperatureBCMatrix_ = ITHACAstream::readMatrix(temperatureBCFile);

    temperatureTimeDep_ = false;
    velocityTimeDep_ = false;

    if (temperatureBCMatrix_.rows() > 1)
    {
        temperatureTimeDep_ = true;
        timestepsTempBC_ = temperatureBCMatrix_.col(0);
    }

    if (velocityBCMatrix_.rows() > 1)
    {
        velocityTimeDep_ = true;
        timestepsVelBC_ = velocityBCMatrix_.col(0);
    }

    currentTemperatureBC = temperatureBCMatrix_.row(0).tail(temperatureBCMatrix_.cols() - 1);
    currentVelocityBC = velocityBCMatrix_.row(0).tail(velocityBCMatrix_.cols() - 1);
}

// Methods
void BoundaryConditions::updateTimeDependentBC(const scalar currentTime)
{
    if (temperatureTimeDep_)
    {
        if (timeDepMethod_ == "linear")
        {
            if (currentTime >= timestepsTempBC_.tail(1).value())
            {
                currentTemperatureBC = temperatureBCMatrix_.row(temperatureBCMatrix_.rows() -
                                                               1)
                                           .tail(temperatureBCMatrix_.cols() - 1);
            } else if (currentTime <= timestepsTempBC_(0))
            {
                currentTemperatureBC = temperatureBCMatrix_.row(0).tail(
                    temperatureBCMatrix_.cols() - 1);
            } else
            {
                auto it = std::upper_bound(timestepsTempBC_.data(),
                    timestepsTempBC_.data() + timestepsTempBC_.size(),
                    currentTime);
                Eigen::Index k = std::distance(timestepsTempBC_.data(), it);

                // Safety check - Should not happen due to how the solver in the ROM is written, but still...
                // Maybe remove for performance, all gas no brakes
                if (k > 0)
                {
                    double t0 = timestepsTempBC_(k - 1);
                    double t1 = timestepsTempBC_(k);
                    double alpha = (currentTime - t0) / (t1 - t0);
                    Eigen::VectorXd val0 = temperatureBCMatrix_.row(k - 1).tail(
                        temperatureBCMatrix_.cols() - 1);
                    Eigen::VectorXd val1 = temperatureBCMatrix_.row(k).tail(
                        temperatureBCMatrix_.cols() - 1);
                    currentTemperatureBC = val0 + alpha * (val1 - val0);
                }
            }
        }
    }

    if (velocityTimeDep_)
    {
        if (timeDepMethod_ == "linear")
        {
            if (currentTime >= timestepsVelBC_.tail(1).value())
            {
                currentVelocityBC = velocityBCMatrix_.row(velocityBCMatrix_.rows() - 1).tail(velocityBCMatrix_.cols() - 1);
            } else if (currentTime <= timestepsVelBC_(0))
            {
                currentVelocityBC = velocityBCMatrix_.row(0).tail(velocityBCMatrix_.cols() - 1);
            } else
            {
                auto it = std::upper_bound(timestepsVelBC_.data(),
                    timestepsVelBC_.data() + timestepsVelBC_.size(),
                    currentTime);
                Eigen::Index k = std::distance(timestepsVelBC_.data(), it);

                // Safety check - Should not happen due to how the solver in the ROM is written, but still...
                if (k > 0)
                {
                    double t0 = timestepsVelBC_(k - 1);
                    double t1 = timestepsVelBC_(k);
                    double alpha = (currentTime - t0) / (t1 - t0);
                    Eigen::VectorXd val0 = velocityBCMatrix_.row(k - 1).tail(
                        velocityBCMatrix_.cols() - 1);
                    Eigen::VectorXd val1 = velocityBCMatrix_.row(k).tail(velocityBCMatrix_.cols() -
                        1);
                    currentVelocityBC = val0 + alpha * (val1 - val0);
                }
            }
        }
    }
}

double BoundaryConditions::linearInterpolate(const double t0, const double t1,
    const double v0, const double v1, const scalar currentTime)
{
    double interpolatedValue = v0 + (v1 - v0) * (currentTime - t0) / (t1 - t0);
    return interpolatedValue;
}

void BoundaryConditions::correctLiftingCoeffs(Eigen::VectorXd& y,
    const int N_BC, const int N_BC_t, const int Nphi_u, const int Nphi_prgh)
{
    y.head(N_BC) = currentVelocityBC.head(N_BC);
    y.segment(Nphi_u + Nphi_prgh, N_BC_t) = currentTemperatureBC.head(N_BC_t);
}

// ************************************************************************ //
