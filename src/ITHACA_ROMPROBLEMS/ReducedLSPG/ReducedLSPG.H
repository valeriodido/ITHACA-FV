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
    ReducedLSPG
Description
    Class where the base of a Least-Squares Petrov-Galerkin reduced problem for the unsteady flow is implemented.
SourceFiles
    ReducedLSPG.C
\*---------------------------------------------------------------------------*/

/// \file
/// Header file of the ReducedLSPG class.
/// \dir
/// Directory containing the header and source files for the ReducedLSPG class

#ifndef ReducedLSPG_H
#define ReducedLSPG_H

#include "fvCFD.H"
#include "pimpleControl.H"
#include "fvOptions.H"
#include "ITHACAcontext.H"
#include "BlockResidual.H"
#include <Eigen/Dense>
#include <memory>

struct ROMSettings
{
  word bcMethod = "lift";
  word romID = "";
};

/*
Base class for the Reduced Least-Squares Petrov-Galerkin (LSPG) problem
Here we only implement the constructor and the destructor, which initialize quantities that are common to all (most) reduced problems.
Detailed implementation, such as field creation, is left to the derived classes.
*/
class ReducedLSPG
{
  public:
    /// @brief  Deleted default constructor
    ///
    /// @details This constructor is deleted to prevent the creation of an object of this class without a valid context.
    ReducedLSPG() = delete;
    
    /// @brief Constructor, takes a context object
    /// @param context Shared pointer to the ITHACAcontext object, which contains the mesh, time, fvOptions, and other necessary objects for the FOM problem.
    explicit ReducedLSPG(std::shared_ptr<ITHACAcontext> context);

    /// @brief Destructor
    virtual ~ReducedLSPG() = default;

    /// @brief Virtual function to be overridden to assemble the residual
    virtual Eigen::VectorXd assembleResidual(const Eigen::VectorXd& state, bool correctBCs = true) = 0;

    /// @brief Returns the scaling factor
    const Eigen::VectorXd& getStateScaling() const
    {
        return stateScaling_;
    }

    const word& getID() const
    {
        return romID_;
    }

  protected:
    /// @brief Shared pointer to the ITHACAcontext object
    std::shared_ptr<ITHACAcontext> context_;

    /// @brief Pointer to the ITHACAdict object
    IOdictionary* ITHACAdict;

    /// @brief Access to the argList object
    /// @return Reference to the argList object
    argList& args() const;
    
    /// @brief Access to the fvMesh object
    /// @return Reference to the fvMesh object
    fvMesh& mesh() const;

    /// @brief Access to the Time object
    /// @return Reference to the Time object
    Time& runTime() const;

    /// @brief  Access to the pimpleControl object
    /// @return Reference to the pimpleControl object 
    pimpleControl& pimple() const;

    /// @brief Access to the fvOptions object
    /// @return Reference to the fvOptions object
    fv::options& fvOptions() const;

    /// @brief Access to the MRF object
    /// @return Reference to the MRF object
    IOMRFZoneList& MRF() const;

    /// Current state vector
    Eigen::VectorXd currentState_;

    /// State scaling vector (could be for example the POD singular values)
    Eigen::VectorXd stateScaling_;

    /// Word identifier. It literally just identifies the ROM for post-processing purposes.
    word romID_;

    // *** Time step management - These are called by openfoam Time object *** //
    //- Whether to adjust the time step based on the Courant number (or custom criteria?)
    bool adjustTimeStep;

    //- Maximum Courant number allowed
    scalar maxCo;

    //- Maximum time step allowed
    scalar maxDeltaT;
};

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif
