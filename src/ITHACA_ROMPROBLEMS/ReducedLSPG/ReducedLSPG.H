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
    Base of the Least-Squares Petrov-Galerkin (LSPG) reduced problems. Physics-agnostic: it manages the
    settings, the reduced state, the block residual and the online solve (one Levenberg-Marquardt solve
    per time step). A model implements assembleResidual and initializePODCoeffsFromFields, and the interface to
    of the time loop it needs (boundary conditions, output, diagnostics).
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
#include "ResidualCollector.H"
#include "LSPGSettings.H"
#include "StateMap.H"
#include "ITHACAstream.H"
#include <Eigen/Dense>
#include <memory>

struct LevenbergMarquardtFunctor;

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

    ReducedLSPG(const ReducedLSPG&) = delete;
    ReducedLSPG& operator=(const ReducedLSPG&) = delete;

    /// @brief Destructor
    virtual ~ReducedLSPG() = default;

    /// @brief Assembles the residual at the given state (field coefficients)
    virtual Eigen::VectorXd assembleResidual(
        const Eigen::VectorXd& state,
        const bool correctBCs = true,
        const bool writeResidual = false) = 0;

    /// @brief Initializes the POD coefficients from the current fields in memory
    virtual void initializePODCoeffsFromFields() = 0;

    /// @brief Full state x = Ey from the scaled reduced coordinates y, with E = stateMap_
    /// stateMap_ maps the reduced coordinates onto the state.
    /// @details Imagine we compute the joint POD on field U and field p, with 2 modes each.
    /// The full state will be: [y1, y2, y1, y2], where the first two entries are the coefficients of U and the last two are the coefficients of p.
    /// The stateMap_ will be: [0, 1, 0, 1], which maps the first and third entries of the full state to the first reduced coordinate y1, and the second and fourth entries to the second reduced coordinate y2.
    /// The coordinates will be: [y1, y2]
    Eigen::VectorXd fullState(const Eigen::VectorXd& y) const
    {
        return stateMap_.expand(stateScaling_.cwiseProduct(y));
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

    /// @brief Norm of the weighted residual at the last evaluation, over all the processors
    double residualNorm() const
    {
        return residual_.norm();
    }

    /// @brief Weight of every row of the residual minimised by LM (BlockResidual::rowScaling)
    Eigen::VectorXd residualRowScaling() const
    {
        return residual_.rowScaling();
    }

    /// @brief Block weights currently applied to the reported blocks of the residual (0 for an absent block)
    Eigen::VectorXd residualBlockWeights() const;

protected:
    // *** Online solve *** //

    /// @brief Time loop of the online solve, results in folder: one LM solve per time step from the
    /// current fields (initializePODCoeffsFromFields). The model sets its fields and boundary conditions before
    void runOnline(const word& folder);

    //- Called at every time step, before the solve
    virtual void updateBoundaryConditions(const scalar time)
    {}

    //- Called at every time step, after the final assembly at the converged state
    virtual void timeStepCompleted()
    {}

    //- Writes the reduced fields at the write times of the online solve
    virtual void writeFields(const word& folder) const
    {}

    //- Called when the LM solve of a time step stopped, at the solution y (scaled coordinates)
    virtual void diagnoseSolve(
        const LevenbergMarquardtFunctor& functor,
        const Eigen::VectorXd& y,
        const std::string& status,
        const word& folder)
    {}

    // *** Residual *** //

    /// @brief Sets the blocks reported in the logs and in residualBlockWeights, in this order. A block
    /// index -1 is reported as 0 (e.g. an optional block that is switched off)
    void setReportedBlocks(const labelList& blocks, const wordList& names);

    /// @brief Rows of a cell block weighted by 1/sqrt(V), one weight per component
    void setCellVolumeRowWeights(const label block, const label nComponents);

    const labelList& reportedBlocks() const
    {
        return reportedBlocks_;
    }

    const wordList& reportedBlockNames() const
    {
        return reportedBlockNames_;
    }

    /// @brief Norm of a reported block, 0 if absent
    double reportedBlockNorm(const label i) const;

    template <class Type>
    void writeResidualField(
        const word& name,
        const fvMatrix<Type>& eqn,
        const Field<Type>& R,
        const fvMesh& mesh,
        const bool perUnitVolume = true)
    {
        LSPGResidualOutput::writeMatrixResidual<Type>(name, eqn, R, mesh, collector_.outputFolder(), perUnitVolume);
    }

    // *** Access *** //

    /// @brief Shared pointer to the ITHACAcontext object
    std::shared_ptr<ITHACAcontext> context_;

    /// @brief ITHACAdict
    autoPtr<IOdictionary> ITHACAdict;

    /// @brief Settings
    ROMSettings romSettings_;

    /// @brief Settings of the affine reference of the fields
    AffineReferenceSettings affineSettings_;

    /// @brief Stopping criteria of the LM solve
    LMSolverSettings lmSettings_;

    argList& args() const;
    fvMesh& mesh() const;
    Time& runTime() const;
    pimpleControl& pimple() const;
    fv::options& fvOptions() const;
    IOMRFZoneList& MRF() const;

    // *** State *** //

    /// Current state vector (field coefficients)
    Eigen::VectorXd currentState_;

    /// Reduced coordinates solved for by LM (unscaled), currentState_ = stateMap_ * currentCoords_
    Eigen::VectorXd currentCoords_;

    /// State scaling vector, one entry per reduced coordinate: LM solves for y = c / stateScaling_
    Eigen::VectorXd stateScaling_;

    /// Maps the reduced coordinates onto the state (stacked POD)
    StateMap stateMap_;

    // *** Residual *** //

    /// Weighted residual, filled through collector_
    BlockResidual residual_;

    /// Fills residual_ from the equations of the model
    ResidualCollector collector_;

    // *** Time step management - These are called by openfoam Time object *** //
    //- Whether to adjust the time step based on the Courant number (or custom criteria?)
    bool adjustTimeStep;

    //- Maximum Courant number allowed
    scalar maxCo;

    //- Maximum time step allowed
    scalar maxDeltaT;

    //- Number of cell elements
    label nCells;

private:
    //- Writes the norm of every reported block
    void printBlockNorms() const;

    labelList reportedBlocks_;
    wordList reportedBlockNames_;
};

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif
