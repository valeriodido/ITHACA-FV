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
#include "ITHACAstream.H"
#include <Eigen/Dense>
#include <memory>

struct ROMSettings {
    word bcMethod = "lift";
    word romID = "";
    //- Specifier on how the snapshots were collected. Affects the reconstruction
    label snapshotCollectionMethod = 1;
    bool scaleResidualWithCellVolume = true;
    //- Debug flag
    bool debug_ = false;
    //- Adds the face-flux residual phi - (phiHbyA - pEqn.flux()) on the internal faces
    //  (He et al. 2021, eq. 1.8), which ties the reconstructed flux to the Rhie-Chow flux of (U, p)
    bool fluxResidual = true;
    //- Writes the residual fields at every time step of solveOnline, in the ROM case folder
    bool writeResidualFields = false;
    //- Relative finite-difference step for the Jacobian, h_j = jacobianStep * max(|y_j|, 1)
    scalar jacobianStep = 1e-5;
    //- Order of the finite-difference Jacobian: 1 forward (n + 1 residual evaluations), 2 central (2n)
    label jacobianOrder = 2;
    //- Residual preconditioning (Lindsay, Fike, Tezaur, Carlberg 2022, preconditioned LSPG):
    //  "none"   : volume-integrated residuals weighted by 1/sqrt(V), block weights calibrated at t0
    //  "jacobi" : each row divided by its matrix diagonal (frozen within a time step), i.e. an approximate
    //             local increment of the unknown, weighted by sqrt(V) and divided by a field scale
    word residualPreconditioner = "none";
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
    virtual Eigen::VectorXd assembleResidual(
        const Eigen::VectorXd& state,
        const bool correctBCs = true,
        const bool writeResidual = false) = 0;

    /// @brief Initializes the POD coefficients from the current fields in memory
    virtual void initializePODCoeffsFromFields() = 0;

    /// @brief Returns the scaling factor
    const Eigen::VectorXd& getStateScaling() const
    {
        return stateScaling_;
    }

    /// @brief Full state x = E (s .* y) from the scaled reduced coordinates y, with E = stateMap_
    /// (identity if stateMap_ is empty)
    Eigen::VectorXd fullState(const Eigen::VectorXd& y) const
    {
        const Eigen::VectorXd x = stateScaling_.cwiseProduct(y);
        return stateMap_.size() ? Eigen::VectorXd(stateMap_ * x) : x;
    }

    /// @brief Returns the relative finite-difference step for the Jacobian
    double jacobianStep() const
    {
        return romSettings_.jacobianStep;
    }

    /// @brief Returns the order of the finite-difference Jacobian (1 forward, 2 central)
    label jacobianOrder() const
    {
        return romSettings_.jacobianOrder;
    }

    /// @brief Returns the ID of the ROM
    const word& getID() const
    {
        return romSettings_.romID;
    }

protected:
    template <class Type>
    void writeResidualField(
        const word& name,
        const fvMatrix<Type>& eqn,
        const Field<Type>& R,
        const fvMesh& mesh,
        const bool perUnitVolume = true);

    /// @brief Writes a residual field: in residualOutputFolder_/<time> if set, else in the case time directory
    template <class FieldType>
    void exportResidual(FieldType& res) const
    {
        if (residualOutputFolder_.empty())
        {
            res.write();
        } else
        {
            ITHACAstream::exportSolution(res, name(res.time().value()), residualOutputFolder_);
        }
    }

    /// @brief Output folder of the residual fields (empty: case time directory)
    word residualOutputFolder_;

protected:
    /// @brief Shared pointer to the ITHACAcontext object
    std::shared_ptr<ITHACAcontext> context_;

    /// @brief Pointer to the ITHACAdict object
    IOdictionary* ITHACAdict;

    /// @brief Settings
    ROMSettings romSettings_;

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

    /// State scaling vector (could be for example the POD singular values), one entry per reduced coordinate
    Eigen::VectorXd stateScaling_;

    /// Maps the reduced coordinates onto the full state, e.g. a coefficient shared by the fields of a
    /// stacked POD is copied into the slot of every field. Empty: the reduced coordinates are the state
    Eigen::MatrixXd stateMap_;

    // *** Time step management - These are called by openfoam Time object *** //
    //- Whether to adjust the time step based on the Courant number (or custom criteria?)
    bool adjustTimeStep;

    //- Maximum Courant number allowed
    scalar maxCo;

    //- Maximum time step allowed
    scalar maxDeltaT;

    //- Number of cell elements
    label nCells;
};

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif
