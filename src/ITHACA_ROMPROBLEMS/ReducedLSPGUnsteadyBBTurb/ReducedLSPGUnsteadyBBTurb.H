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
    ReducedLSPGUnsteadyBBTurb
Description
    Class where a Least-Squares Petrov-Galerkin reduced problem for the unsteady flow is implemented.
    Gradually, this class will be extended to handle the buoyant turbulent case.
SourceFiles
    ReducedLSPGUnsteadyBBTurb.C
\*---------------------------------------------------------------------------*/

/// \file
/// Header file of the ReducedLSPGUnsteadyBBTurb class.
/// \dir
/// Directory containing the header and source files for the ReducedLSPGUnsteadyBBTurb class

#ifndef ReducedLSPGUnsteadyBBTurb_H
#define ReducedLSPGUnsteadyBBTurb_H

#include "fvCFD.H"
#include "singlePhaseTransportModel.H"
#include "turbulentTransportModel.H"
#include "ITHACAcontext.H"
#include "ithacaInterpolator.H"
#include "LSPGUnsteadyBBTurb.H"
#include "ReducedLSPG.H"
#include "Modes.H"
#include "BoundaryConditions.H"
#include "RBFTurbulence.H"
#include "LevenbergMarquardt.H"
#include <Eigen/Dense>
#include <memory>

struct InterpolationSettings
{
    int firstRBFIndex = 0;
    bool derivative = false;
    int dimInputRBF = 0;
};

struct NumberOfModes
{
    int velocity = 0;
    int pressure = 0;
    int temperature = 0;
    int nut = 0;
};

struct GaussNewtonSettings
{
    int maxIter = 5;
    float tol = 1e-2;
    float stagnationTol = 1e-3;
};


/// Class where a reduced problem for the <b> unsteady, turbulent, buoyant Navier-stokes </b> problem is implemented.
/** In this class are implemented the methods to solve a reduced version of the unsteady, turbulent, buoyant Navier-stokes equations. This class must be used together with the FOM class */
class ReducedLSPGUnsteadyBBTurb : public ReducedLSPG
{
  // *** Methods *** //
  public:
    //- Empty constructor - Cannot be used to create an object of this class
    ReducedLSPGUnsteadyBBTurb() = delete;
    
    //- Constructor that takes, other than the FOM, argc and argv
    ReducedLSPGUnsteadyBBTurb(std::shared_ptr<ITHACAcontext> context,
      const LSPGUnsteadyBBTurb& problem);

    //- Default destructor
    virtual ~ReducedLSPGUnsteadyBBTurb() = default;

    //- Given the boundary conditions and a timeManager, solves the reduced problem
    void solveOnline(const word& bcROM, const word& bcROMT, const word& folder = "ITHACAoutput/OnlineLSPGROM/");

  private:
    /// @brief Captures the relevant FOM data and stores it internally
    /// ideally the FOM could then be destroyed and the ROM would still work. This is still wip
    void captureFOMData(const LSPGUnsteadyBBTurb& problem);

    //- Assembles the residual vector
    Eigen::VectorXd assembleResidual(const Eigen::VectorXd& state, bool correctBCs = true) override;

    //- Reconstructs the reduced solution fields from the currentState_ vector
    void reconstructReducedFields(
      const Eigen::VectorXd& state,
      volVectorField& velocity_field,
      volScalarField& pressure_field,
      volScalarField& temperature_field,
      surfaceScalarField& phi_field,
      bool correctBCs = true);
  
    //- Interpolates the turbulent eddy viscosity coefficients from the RBF interpolation to the current state
    void interpolateNutCoeffs(
      volScalarField& nut_field,
      volScalarModes& nut_modes,
      const Eigen::VectorXd& state);

    //- Inverse Distance Weighting interpolation of the input parameters to get the current state vector
    Eigen::VectorXd interpolateIDW(const Eigen::VectorXd& input_parameters);

    ///- Reads eigenvalues and fills the eigenvalues_ vector
    void readEigenvalues();

    ///- Method to initializa the POD coefficients by reconstructing fields at a selected time
    /// The method reads the fields from disk and projects them onto the modes to get the initial coefficients
    void initializePODCoeffsFromFields();

  private:
    //- Parameters for the offline solution
    Eigen::MatrixXd mu_;

    //- Inlet Patch indices for velocity lifting
    Eigen::MatrixXi inletIndex_;

    //- Inlet Patch indices for temperature lifting
    Eigen::MatrixXi inletIndexT_;

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
    volScalarModes Prghmodes_;
    
    //- Temperature modes
    volScalarModes Tmodes_;

    //- Temperature lifting function
    PtrList<volScalarField> liftFieldsT_;

    //- Eddy viscosity modes
    volScalarModes Nutmodes_;

    //- Turbulent kinetic energy modes
    volScalarModes kModes_;

    //- Turbulent omega modes
    volScalarModes omegaModes_;

    //- Fluctuating eddy viscosity fields
    PtrList<volScalarField> nutFields_;

    //- Averaged eddy viscosity fields
    PtrList<volScalarField> avgNutFields_;

    ///- Settings for the ROM
    ROMSettings romSettings_;

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

    ///- Pointer to the Prandtl number 
    autoPtr<dimensionedScalar> _Pr;

    ///- Pointer to the turbulent Prandtl number
    autoPtr<dimensionedScalar> _Prt;

    ///- Pointer to the thermal expansion coefficient
    autoPtr<dimensionedScalar> _beta;

    ///- Pointer to the reference temperature
    autoPtr<dimensionedScalar> _TRef;

    ///- Gravitational constant
    autoPtr<dimensionedVector> _g;

    ///- Pointer to hRef
    autoPtr<dimensionedScalar> _hRef;

    ///- Pointer ot ghRef
    autoPtr<dimensionedScalar> _ghRef;

    ///- Pointer to field gh
    autoPtr<volScalarField> _gh;

    ///- Pointer to field ghf
    autoPtr<surfaceScalarField> _ghf;

    ///- Pointer to the reduced temperature field
    autoPtr<volScalarField> _T;

    ///- Pointer to the reduced p_rgh
    autoPtr<volScalarField> _p_rgh;

    ///- Pointer to the reduced velocity field
    autoPtr<volVectorField> _U;

    ///- Pointer to the reduced face flux field
    autoPtr<surfaceScalarField> _phi;

    ///- Laminar transport pointer
    autoPtr<singlePhaseTransportModel> _laminarTransport;

    ///- Turbulent transport pointer
    // autoPtr<incompressible::turbulenceModel> turbulence; // For conformity, can we rename this to _...?
    autoPtr<RBFTurbulence> turbulence;

    ///- Pointer to kinematic density
    autoPtr<volScalarField> _rhok;

    ///- Pointer to the kinematic turbulent thermal conductivity
    autoPtr<volScalarField> _alphat;

    ///- Pointer to the pressure field
    autoPtr<volScalarField> _p;

    ///- Reference cell for pressure
    label pRefCell;
    
    ///- Reference value for pressure
    scalar pRefValue;

    ///- Modes eigenvalues, to be used for normalization of the modes coefficients
    Eigen::VectorXd eigenvalues_;

    ///- Workspace for the residual
    BlockResidual residualWorkspace_;

    ///- Block index for U
    label iU_;

    label iP_;

    label iT_;
};

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif
