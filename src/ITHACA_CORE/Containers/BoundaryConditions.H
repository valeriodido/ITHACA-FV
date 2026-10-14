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
    BoundaryConditions
Description
    A class to handle various boundary conditions for Reduced Order Models (time-dependent, temperature, ...).
    At the moment this is only implemented for linear time-dependent BCs, where both temperature and velocity are given.
    The input BCs are given as Eigen::MatrixXd, where the first column is a series of time steps, and the other columns are the BC values at those time steps.
SourceFiles
    BoundaryConditions.C
\*---------------------------------------------------------------------------*/

/// \file
/// Header file of the BoundaryConditions class.
/// \dir
/// Directory containing the header and source files for the BoundaryConditions class

#ifndef BoundaryConditions_H
#define BoundaryConditions_H

#include "fvCFD.H"
#include "IOmanip.H"
#include <Eigen/Dense>

/*---------------------------------------------------------------------------*\
                        Class BoundaryConditions Declaration
\*---------------------------------------------------------------------------*/
/// Class where a general treatment of boundary conditions for ROMs is implemented
/*
The idea is that one instance of this objects is creted for each field that needs bc
*/

class BoundaryConditions
{
private:
    /// Matrix that stores the BCs
    Eigen::MatrixXd BCMatrix_;

    /// Vector that stores the time instances at which the BCs change
    Eigen::VectorXd timestepsBC_;

    /// Boolean to know if time-dep
    bool timeDep_ = false;

    /// Word that stores how to treat time-dependent BCs. At the moment only "linear" is implemented
    word timeDepMethod_;

    /// Vector that stores the current velocity BCs
    Eigen::VectorXd currentBC_;

    /// Number of BCs
    int nBCs_;

    /// @brief    Method to linearly interpolate a value between two values
    ///
    double linearInterpolate(const double t0, const double t1, const double v0,
        const double v1, const scalar currentTime);

public:
    // Constructors
    BoundaryConditions() { };

    BoundaryConditions(const Eigen::MatrixXd& BCMatrix,  word timeDepMethod);
    BoundaryConditions(const Eigen::VectorXd& BCMatrix);
    BoundaryConditions(const word BCFile, word timeDepMethod = "linear");

    // Destructor
    ~BoundaryConditions() { };

    /// @brief    Method to update time-dependent boundary conditions
    ///
    /// @param[in]  currentTime  Scalar of the current time
    ///
    void updateTimeDependentBC(const scalar currentTime);

    /// @brief    Returns the current BCs sequentially as a single Eigen::VectorXd, with the velocity BCs first and then the temperature BCs
    Eigen::VectorXd getCurrentBCs() const
    {
        return currentBC_;
    }

    /// @brief    Corrects the first N_BC and N_BC_t coefficients of the reduced solution vector, so that
    ///           they match the current boundary conditions when using the lifting function approach.
    ///
    /// @param[in]  y The reduced coefficient vector to be corrected. Is a Eigen::VectorXd
    /// @param[in]  segment_start The index of the first coefficient to be corrected. Is an int
    
    void correctLiftingCoeffs(Eigen::VectorXd& y, const int segment_start);
};

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif
