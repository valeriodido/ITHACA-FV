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
    Class where a Least-Squares Petrov-Galerkin reduced problem for the unsteady flow is implemented.
    Gradually, this class will be extended to handle the buoyant turbulent case.
SourceFiles
    ReducedLSPGUnsteadyNSTurb.C
\*---------------------------------------------------------------------------*/

/// \file
/// Header file of the ReducedLSPGUnsteadyNSTurb class.
/// \dir
/// Directory containing the header and source files for the ReducedLSPGUnsteadyNSTurb class

#ifndef ReducedLSPGUnsteadyNSTurb_H
#define ReducedLSPGUnsteadyNSTurb_H

#include "fvCFD.H"
#include "singlePhaseTransportModel.H"
#include "turbulentTransportModel.H"
#include "ResidualTurbulenceModel.H"
#include "ITHACAkOmegaSST.H"
#include "ITHACAcontext.H"
#include "ithacaInterpolator.H"
#include "LSPGUnsteadyNSTurb.H"
#include "ReducedLSPG.H"
#include "Modes.H"
#include "BoundaryConditions.H"
#include "RBFTurbulence.H"
#include "LevenbergMarquardt.H"
#include "PressureResidualCheck.H" // Used for residual diagnostics in the FOM evaluation of the ROM residual
#include <Eigen/Dense>
#include <memory>

struct InterpolationSettings {
    int firstRBFIndex = 0;
    bool derivative = false;
    int dimInputRBF = 0;
};

struct NumberOfModes {
    int velocity;
    int pressure;
    int k;
    int omega;
};

/// Stopping criteria of the LM solve of each time step (ITHACAdict, lmSolver). The LSPG residual does not
/// vanish at the solution, so convergence is stationarity, not a small residual norm
struct LMSolverSettings {
    //- Maximum LM iterations (Jacobian evaluations) per time step
    label maxIterations = 10;
    //- gtol: largest cosine between the residual and a Jacobian column, |J_j^T r| / (|J_j| |r|)
    scalar gradientTol = 1e-4;
    //- ftol: relative reduction of |r|^2, actual and predicted
    scalar reductionTol = 1e-6;
    //- xtol: trust-region radius relative to |y| (scaled by the Jacobian column norms); 0 disables it
    scalar stepTol = 0;
};

/// Settings of the solver diagnostics run when the LM solve of a time step fails (ITHACAdict, lmDiagnostics)
struct LMDiagnosticsSettings {
    bool enabled = false;
    //- LM statuses that trigger a diagnostic dump ("Running" = maximum iterations reached)
    wordList triggers;
    //- Maximum number of dumps per online solve
    label maxDumps = 3;
    //- Steps of the scans, in scaled coordinates y
    scalarList steps;
    //- Scan every reduced coordinate (2 residual evaluations per coordinate and step)
    bool coordinateScan = true;
};


/// Class where a reduced problem for the <b> unsteady, turbulent, buoyant Navier-stokes </b> problem is implemented.
/** In this class are implemented the methods to solve a reduced version of the unsteady, turbulent, buoyant Navier-stokes equations. This class must be used together with the FOM class */
class ReducedLSPGUnsteadyNSTurb : public ReducedLSPG
{
    // *** Methods *** //
public:
    //- Empty constructor - Cannot be used to create an object of this class
    ReducedLSPGUnsteadyNSTurb() = delete;

    //- Constructor that takes, other than the FOM, argc and argv
    ReducedLSPGUnsteadyNSTurb(std::shared_ptr<ITHACAcontext> context,
        const LSPGUnsteadyNSTurb& problem);

    //- Default destructor
    virtual ~ReducedLSPGUnsteadyNSTurb() = default;

    //- Given the boundary conditions and a timeManager, solves the reduced problem
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

    /// @brief Block weights (U, p, k, omega, phi) currently applied to the residual (phi is 0 without fluxResidual)
    Eigen::VectorXd residualBlockWeights() const;

    /// @brief Weights of the offline runs at the given parameters.
    /// Settings in ITHACAdict: parametricDistanceExponent (default 1) and maxInterpolatedSolutions
    /// (only the nearest runs, ties included; default 0 = all).
    Eigen::VectorXd interpolateIDW(const Eigen::VectorXd& input_parameters) const;

private:
    /// @brief Captures the relevant FOM data and stores it internally
    /// ideally the FOM could then be destroyed and the ROM would still work. This is still wip
    void captureFOMData(const LSPGUnsteadyNSTurb& problem);

    //- Assembles the residual vector
    Eigen::VectorXd assembleResidual(const Eigen::VectorXd& state,
        const bool correctBCs = true,
        const bool writeResidual = false) override;

    //- Terms of the pressure residual at the current fields, assembled as in residualWithCurrentFields
    PressureResidualCheck::Terms pressureResidualTerms();

    //- Assembles the residual vector using the current fields in memory
    Eigen::VectorXd residualWithCurrentFields(const bool writeResidual = false);

    //- Norm of the face-flux block, 0 if fluxResidual is off
    double phiBlockNorm() const;

    //- Row weights of the Jacobi preconditioner from the diagonals of the assembled equations
    //- Check against Linsday (Carlberg) paper
    void setJacobiPreconditioner(
        const fvVectorMatrix& UEqn,
        const fvScalarMatrix& pEqn,
        const PtrList<fvScalarMatrix>& turbEqs);

    // *** Solver diagnostics (LMDiagnostics.C) *** //

    //- Whether the LM status of a time step triggers a diagnostic dump
    bool lmDiagnosticsTriggered(const std::string& status) const;

    //- Diagnoses the reduced problem at the point y where LM stopped. Writes in folder/lmDiagnostics/<time>/:
    //  the singular values of the Jacobian, residual scans along the Gauss-Newton and steepest-descent
    //  directions and along every reduced coordinate (kinks, jumps), the counts of k/omega cells crossing
    //  the clipping bound and of faces where phi changes sign, and a summary with a heuristic interpretation
    void runLMDiagnostics(
        const LevenbergMarquardtFunctor& functor,
        const Eigen::VectorXd& y,
        const std::string& status,
        const word& folder);

    //- Reconstructs the reduced solution fields from the currentState_ vector
    void reconstructReducedFields(
        const Eigen::VectorXd& state,
        volVectorField& velocity_field,
        volScalarField& pressure_field,
        surfaceScalarField& phi_field,
        bool correctBCs = true);

    //- Interpolates the turbulent eddy viscosity coefficients from the RBF interpolation to the current state
    void interpolateNutCoeffs(
        volScalarField& nut_field,
        volScalarModes& nut_modes,
        const Eigen::VectorXd& state);

    void createTurbulenceModel();


    ///- Method to initializa the POD coefficients by reconstructing fields at a selected time
    /// The method reads the fields from disk and projects them onto the modes to get the initial coefficients
    void initializePODCoeffsFromFields();

    //- Resets the time to the start time and the reduced fields (U_r, p_r, k_r, omega_r, nut_r, phi_r) to the
    //  initial condition on disk, without old time levels, as after construction.
    void resetToInitialCondition();

    //- Builds stateMap_: the reduced coordinates are [shared coefficients of the stacked POD fields;
    //  coefficients of the other fields in the order U p k omega]. Empty map without a stacked POD
    void buildStateMap();

    //- Reduced coordinates of a full state, the inverse of stateMap_ on its range
    Eigen::VectorXd reducedCoordinates(const Eigen::VectorXd& state) const;

    //- Coefficients of the stacked POD: projection of the snapshots (columns, Foam2Eigen layout, in the
    //  order U p k omega; the fields outside the stacked POD are not used) in the weighted inner product
    //  sum_f w_f <., .>_V the modes were computed with
    Eigen::MatrixXd jointCoefficients(const List<Eigen::MatrixXd>& snapshots);

private:
    //- Parameters for the offline solution
    Eigen::MatrixXd mu_;

    //- Inlet Patch indices for velocity lifting
    Eigen::MatrixXi inletIndex_;

    //- Old state vector, used in the RBF interpolation if requested
    Eigen::VectorXd oldState_;

    //- RBF for the interpolation
    std::vector<std::shared_ptr<ithacaInterpolator>> rbfSplines_;

    //- Velocity modes
    volVectorModes Umodes_;

    //- Phimodes
    surfaceScalarModes Phimodes_;

    //- Velocity lifting function
    PtrList<volVectorField> liftFields_;

    //- Phi lifting function
    PtrList<surfaceScalarField> liftFieldsPhi_;

    //- Pressure modes
    volScalarModes Pmodes_;

    //- Eddy viscosity modes
    volScalarModes Nutmodes_;

    //- Turbulent kinetic energy modes
    volScalarModes kModes_;

    //- Turbulent omega modes
    volScalarModes omegaModes_;

    //- Velocity initial conditions
    PtrList<volVectorField> U0field_;

    //- Flux initial conditions
    PtrList<surfaceScalarField> Phi0field_;

    //- Pressure initial conditions
    PtrList<volScalarField> P0field_;

    //- Eddy viscosity initial conditions
    PtrList<volScalarField> Nut0field_;

    //- Turbulent kinetic energy initial conditions
    PtrList<volScalarField> k0field_;

    //- Turbulent omega initial conditions
    PtrList<volScalarField> omega0field_;

    //- Fluctuating eddy viscosity fields
    PtrList<volScalarField> nutFields_;

    //- Averaged eddy viscosity fields
    PtrList<volScalarField> avgNutFields_;

    ///- Interpolation settings
    InterpolationSettings interpolationSettings_;

    ///- Number of modes for the reduced problem
    NumberOfModes numberOfModes_;

    ///- Boundary conditions for the reduced problem
    BoundaryConditions boundaryConditions_;

    ///- Current nut coefficients vector (fluctuating part)
    Eigen::VectorXd currentNutCoeffs_;

    ///- Current nut coefficients vector (averaged part)
    Eigen::VectorXd currentNutAvgCoeffs_;

    // *** Fields - Used in the creation of the residual for the LSPG. *** //
    ///- Pointer to the kinematic viscosity field
    autoPtr<dimensionedScalar> _nu;

    ///- Pointer to the turbulent viscosity field
    autoPtr<volScalarField> _nut;

    ///- Pointer to the turbulent kinetic energy field
    autoPtr<volScalarField> _k;

    ///- Pointer to the turbulent omega field
    autoPtr<volScalarField> _omega;

    ///- Pointer to the reduced velocity field
    autoPtr<volVectorField> _U;

    ///- Pointer to the reduced face flux field
    autoPtr<surfaceScalarField> _phi;

    ///- Laminar transport pointer
    autoPtr<singlePhaseTransportModel> _laminarTransport;

    ///- Turbulent transport pointer
    // autoPtr<RBFTurbulence> turbulence;
    // autoPtr<ITHACATurbulenceModels::<incompressible::turbulenceModel>> turbulence;
    autoPtr<ITHACATurbulenceModels::ResidualEddyViscosityTurbulenceModel> turbulence;

    ///- Pointer to the pressure field
    autoPtr<volScalarField> _p;

    ///- Reference cell for pressure
    label _pRefCell;

    ///- Reference value for pressure
    scalar _pRefValue;

    ///- RMS of the training values of each reduced coordinate, used as state scaling
    Eigen::VectorXd coefficientScale_;

    ///- Reduced coordinates solved for by LM (unscaled), currentState_ = stateMap_ * currentCoords_
    Eigen::VectorXd currentCoords_;

    ///- Fields (U p k omega) of the stacked POD, which share one set of coefficients
    boolList jointField_ = boolList(4, false);

    ///- Weights of (U p k omega) in the inner product of the stacked POD
    Eigen::VectorXd jointWeights_;

    ///- Field scales (U, p, k, omega) of the Jacobi-preconditioned residual: RMS of the training snapshots
    ///  (the fields the POD was computed on), or preconditionerScales in ITHACAdict
    Eigen::VectorXd fieldScale_;

    ///- Recompute the Jacobi preconditioner at the next residual evaluation (set once per time step)
    bool updatePreconditioner_ = true;

    ///- Workspace for the residual
    BlockResidual residualWorkspace_;

    ///- Block index for U
    label iU_;

    ///- Block index for p
    label iP_;

    ///- Block index for k
    label iK_;

    ///- Block index for omega
    label iOmega_;

    ///- Block index for the face flux, -1 if fluxResidual is off
    label iPhi_ = -1;

    ///- Inverse distance weighting of the affine reference: exponent of the distance, and number of nearest
    ///  runs used (0: all)
    scalar idwExponent_ = 1.0;
    label idwNeighbours_ = 0;

    ///- Stopping criteria of the LM solve
    LMSolverSettings lmSettings_;

    ///- Solver diagnostics settings, and dumps written so far in the current online solve
    LMDiagnosticsSettings lmDiagnostics_;
    label nDiagnosticDumps_ = 0;
};

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif
