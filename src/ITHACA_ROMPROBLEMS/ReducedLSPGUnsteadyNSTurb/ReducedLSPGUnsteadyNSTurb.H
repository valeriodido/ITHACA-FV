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
    ReducedLSPGUnsteadyNSTurb
Description
    Least-Squares Petrov-Galerkin reduced problem of the unsteady, incompressible, turbulent (RANS) flow.
    The reduced fields are U, p and the transported fields of the eddy-viscosity closure (k, omega for
    kOmegaSST); the face flux phi is reconstructed from the velocity coefficients. The residual blocks are
    the momentum, pressure and closure equations, and optionally the face flux (fluxResidual).
    The online solve itself (time loop, Levenberg-Marquardt) is in ReducedLSPG.
SourceFiles
    ReducedLSPGUnsteadyNSTurb.C
    ResidualOnTrajectory.C
\*---------------------------------------------------------------------------*/

/// \file
/// Header file of the ReducedLSPGUnsteadyNSTurb class.
/// \dir
/// Directory containing the header and source files for the ReducedLSPGUnsteadyNSTurb class

#ifndef ReducedLSPGUnsteadyNSTurb_H
#define ReducedLSPGUnsteadyNSTurb_H

#include "fvCFD.H"
#include "ResidualTurbulenceModel.H"
#include "ITHACAkOmegaSST.H"
#include "ITHACAcontext.H"
#include "LSPGUnsteadyNSTurb.H"
#include "ReducedLSPG.H"
#include "ReducedField.H"
#include "JointProjector.H"
#include "AffineReference.H"
#include "BoundaryConditions.H"
#include "LevenbergMarquardt.H"
#include "LMDiagnostics.H"
#include "PressureResidualCheck.H" // Used for residual diagnostics in the FOM evaluation of the ROM residual
#include <Eigen/Dense>
#include <memory>


class ReducedLSPGUnsteadyNSTurb : public ReducedLSPG
{
    // *** Methods *** //
public:
    //- Empty constructor - Cannot be used to create an object of this class
    ReducedLSPGUnsteadyNSTurb() = delete;

    //- Constructor from the context and the offline (FOM) problem, which holds the modes and snapshots
    ReducedLSPGUnsteadyNSTurb(std::shared_ptr<ITHACAcontext> context,
        const LSPGUnsteadyNSTurb& problem);

    //- Default destructor
    virtual ~ReducedLSPGUnsteadyNSTurb() = default;

    //- Given the boundary conditions file, solves the reduced problem; results in folder
    void solveOnline(const word& bcROM, const word& folder = "ITHACAoutput/OnlineLSPGROM/");

    /// @brief Consistency diagnostic (Carlberg et al. 2011): evaluates the ROM residual
    /// on a sequence of consecutive FOM states, all separated by deltaT.
    /// @param mode "exact"     : FOM fields, nut recomputed from (U, k, omega) as in the ROM residual
    ///             "projected" : FOM fields projected onto the ROM affine space
    /// @return [nSteps-2 x 7] matrix: time, total, U, p, k, omega, phi norms, weighted as minimised by LM
    /// (phi is 0 without fluxResidual).
    /// The first two steps are skipped, since they lack a valid old-old time level.
    /// @param pressureCheckFile "exact" mode only, if not empty: writes the pressure residual split into
    /// terms and regions (PressureResidualCheck.H), plus nut and old-time level checks against nutFom,
    /// phiFom and Ufom (nutFom may be null)
    Eigen::MatrixXd evaluateResidualOnTrajectory(
        const word& bcROM,
        const PtrList<volVectorField>& Ufom,
        const PtrList<surfaceScalarField>& phiFom,
        const PtrList<volScalarField>& pFom,
        const PtrList<volScalarField>& kFom,
        const PtrList<volScalarField>& omegaFom,
        const Eigen::VectorXd& times,
        const scalar deltaT,
        const word& mode,
        const fileName& pressureCheckFile = fileName(),
        const PtrList<volScalarField>* nutFom = nullptr);

    /// @brief Weights of the offline runs at the given parameters.
    /// Settings in ITHACAdict: parametricDistanceExponent (default 1) and maxInterpolatedSolutions
    /// (only the nearest runs, ties included; default 0 = all).
    Eigen::VectorXd interpolateIDW(const Eigen::VectorXd& input_parameters) const
    {
        return affine_.parametricWeights()(input_parameters);
    }

private:
    // *** Construction *** //

    //- Creates the ROM fields (U_r, p_r, phi_r, nut_r and the closure fields) from the case start time
    void createFields();

    //- Creates the eddy-viscosity closure from momentumTransport or turbulenceProperties
    void createTurbulenceModel(const LSPGUnsteadyNSTurb& problem);

    //- Statistics of the training snapshots of each field, gathered while the fields are declared (only then
    //  are the concrete field types known)
    struct TrainingStatistics
    {
        //- RMS of the volume-weighted L2 norm of the snapshots, per field of fields_
        DynamicList<scalar> fieldRMS;
        //- RMS of the projection coefficients of the fields outside the stacked POD, in set order
        std::vector<Eigen::VectorXd> segregatedRMS;
        //- F_f^T V_f S_f of the joint fields, in set order, for the joint coefficients of the snapshots
        List<Eigen::MatrixXd> jointProjections;
    };

    //- Declares the reduced fields (U, p, closure fields) and the face flux from the modes of the FOM;
    //  returns the statistics of their training snapshots
    TrainingStatistics createReducedFields(const LSPGUnsteadyNSTurb& problem);

    //- Stacked POD: the state map and the projector of the joint fields; sizes currentState_ and currentCoords_
    void createStateMap(const LSPGUnsteadyNSTurb& problem);

    //- State scaling: RMS of the training coefficients, joint first, then the other fields. Needs joint_
    void setStateScaling(const TrainingStatistics& stats);

    //- Field scales of the Jacobi-preconditioned residual: RMS of the training snapshots, or preconditionerScales
    void setFieldScales(const TrainingStatistics& stats);

    //- Creates the residual blocks and their weights
    void createResidual();

    // *** ReducedLSPG interface *** //

    //- Assembles the residual vector at the given state
    Eigen::VectorXd assembleResidual(const Eigen::VectorXd& state,
        const bool correctBCs = true,
        const bool writeResidual = false) override;

    //- Projects the current fields onto the affine space of the ROM: sets currentState_ and currentCoords_
    void initializePODCoeffsFromFields() override;

    void updateBoundaryConditions(const scalar time) override;

    //- Continuity errors of the converged time step
    void timeStepCompleted() override;

    void writeFields(const word& folder) const override;

    //- Runs the LM diagnostics, if the status triggers them
    void diagnoseSolve(
        const LevenbergMarquardtFunctor& functor,
        const Eigen::VectorXd& y,
        const std::string& status,
        const word& folder) override;

    // *** Residual *** //

    //- Assembles the residual vector using the current fields in memory
    Eigen::VectorXd residualWithCurrentFields(const bool writeResidual = false);

    //- Terms of the pressure residual at the current fields, assembled as in residualWithCurrentFields
    PressureResidualCheck::Terms pressureResidualTerms();

    // *** Fields *** //

    //- Sets the fields from the state: modes, affine reference, boundary conditions and closure constraints
    void reconstructFields(const Eigen::VectorXd& state, const bool correctBCs = true);

    //- Sets the affine reference of the fields at the current boundary values
    void updateAffineReference();

    //- Imposes the current inlet velocity on the inlet patches (inletIndex)
    void imposeInletConditions();

    //- Resets the time to the start time and the fields (U_r, p_r, closure fields, nut_r, phi_r) to the
    //  initial condition on disk, without old time levels, as after construction.
    void resetToInitialCondition();

    //- Names of the reduced coordinates: the shared coefficients of the stacked POD, then the other fields
    wordList coordinateLabels() const;

    // *** Data *** //

    //- Adds the face-flux residual phi - (phiHbyA - pEqn.flux()) on the internal faces
    //  (He et al. 2021, eq. 1.8), which ties the reconstructed flux to the Rhie-Chow flux of (U, p)
    bool fluxResidual_;

    //- Inlet patch indices and velocity component of each boundary value
    Eigen::MatrixXi inletIndex_;

    //- Boundary values of the online solve
    BoundaryConditions boundaryConditions_;

    //- Affine reference of the fields
    AffineReference affine_;

    // Fields - Used in the creation of the residual for the LSPG
    autoPtr<dimensionedScalar> nu_;
    autoPtr<volVectorField> U_;
    autoPtr<volScalarField> p_;
    autoPtr<surfaceScalarField> phi_;
    autoPtr<volScalarField> nut_;
    //- Transported fields of the closure (k_r, omega_r)
    PtrList<volScalarField> turbulenceFields_;

    ///- Reference cell and value for pressure
    label pRefCell_ = 0;
    scalar pRefValue_ = 0.0;

    //- Eddy-viscosity closure
    autoPtr<ITHACATurbulenceModels::ResidualEddyViscosityTurbulenceModel> turbulence_;

    //- Reduced fields of the state, in this order: U p and the closure fields
    ReducedFieldSet fields_;

    //- Face flux, reconstructed from the velocity coefficients
    autoPtr<surfaceScalarReducedField> phiField_;

    //- Every field with an affine reference (fields_ and phi)
    UPtrList<ReducedFieldBase> referencedFields_;

    //- Projector of the stacked POD (null without one)
    autoPtr<JointProjector> joint_;

    //- Scale of each field of fields_ in the Jacobi-preconditioned residual: RMS of the training snapshots
    //  (the fields the POD was computed on), or preconditionerScales in ITHACAdict
    Eigen::VectorXd fieldScale_;

    //- Residual blocks
    label iU_ = -1;
    label iP_ = -1;
    //- Block of each closure equation, in the order of transportedFields()
    labelList iTurbulence_;
    //- Face-flux block, -1 if fluxResidual is off
    label iPhi_ = -1;

    //- Cumulative continuity error of the online solve
    scalar cumulativeContErr_ = 0;

    //- Solver diagnostics
    autoPtr<LMDiagnostics> lmDiagnostics_;
};

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif
