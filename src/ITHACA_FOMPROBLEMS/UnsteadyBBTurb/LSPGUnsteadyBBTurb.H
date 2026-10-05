/*---------------------------------------------------------------------------*\
     ██╗████████╗██╗  ██╗ █████╗  ██████╗ █████╗       ███████╗██╗   ██╗
     ██║╚══██╔══╝██║  ██║██╔══██╗██╔════╝██╔══██╗      ██╔════╝██║   ██║
     ██║   ██║   ███████║███████║██║     ███████║█████╗█████╗  ██║   ██║
     ██║   ██║   ██╔══██║██╔══██║██║     ██╔══██║╚════╝██╔══╝  ╚██╗ ██╔╝
     ██║   ██║   ██║  ██║██║  ██║╚██████╗██║  ██║      ██║      ╚████╔╝
     ╚═╝   ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝╚═╝  ╚═╝      ╚═╝       ╚═══╝

 * In real Time Highly Advanced Computational Applications for Finite Volumes
 * Copyright (C) 2017 by the ITHACA-FV authors
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
    LSPGUnsteadyBBTurb
Description
    Implementation for the full order model used to prepare a Least Squares Petrov Galerkin ROM of an
    unsteady, turbulent, buoyant flow.
    Turbulence is treated using an RBF interpolation procedure for the eddy viscosity fluctuation coefficients.
    The interpolation is performed using a radial basis function (RBF) approach.

SourceFiles
    LSPGUnsteadyBBTurb.C
\*---------------------------------------------------------------------------*/

/// \file
/// Header file of the LSPGUnsteadyBBTurb class.
/// \dir
/// Directory containing the header and source files for the LSPGUnsteadyBBTurb class.

#ifndef LSPGUnsteadyBBTurb_H
#define LSPGUnsteadyBBTurb_H
#include "fvCFD.H"
#include "singlePhaseTransportModel.H"
#include "turbulentTransportModel.H"
#include "pimpleControl.H"
#include "IOmanip.H"
#include "fvOptions.H"
#include "ithacaInterpolator.H"
#include "ITHACAcontext.H"
#include "LSPGProblem.H"
#include "UnsteadyProblem.H"
#include "Modes.H"
#include <memory>
#include <iostream>
#include <functional>

/*---------------------------------------------------------------------------*\
                        Class LSPGUnsteadyBBTurb Declaration
\*---------------------------------------------------------------------------*/

/* Implementation of a parametrized full order <b> unsteady, turbulent Boussinesq problem </b>.
In this class are implemented the methods for the offline solve of a unsteady Boussinesq problem (two way coupling between momentum and energy equation)
Differing from the non-LSPG version of this class, here we do not assemble matrices.
Moreover, we treat boundary conditions using a lifting approach, without however appending modes and lifting together.
*/

class LSPGUnsteadyBBTurb : public LSPGProblem, public UnsteadyProblem
{
    public:
        /// Constructor
        explicit LSPGUnsteadyBBTurb(std::shared_ptr<ITHACAcontext> context);

        ~LSPGUnsteadyBBTurb() { };

                //--------------------------------------------------------------------------
        /// @brief     Method to correctly prepare the modes (supremizer and lifting if necessary), be it PPE, Supremizer or VMB
        ///
        void prepareModes();

        //--------------------------------------------------------------------------
        /// @brief   Solves the CFD problem for parameters mu_now. Differently
        /// from UnsteadyBB, here the eddy viscosity field is also computed and saved.
        /// @param[in]  mu_now  The actual value of the parameter for this truthSolve. Used only
        /// @param[in]  nSample  The sample number for which the truth solve is performed.
        void truthSolve(const List<scalar> mu_now, label nSample);

        //--------------------------------------------------------------------------
        /// @brief   Solves the CFD problem for parameters mu_now. Differently
        /// from UnsteadyBB, here the eddy viscosity field is also computed and saved.
        void truthSolve(fileName folder = "./ITHACAOutput/Offline");

        /// @brief      A method to compute the two matrices needed for the RBF interpolation by combining
        /// the velocity L2 projection coefficients and their time derivatives [a, a_dot]. The time derivatives
        /// are computed using a finite difference scheme according to the list of time instants provided snapshotTimes.
        ///
        /// @param[in]  A            The velocity L2 projection coefficients
        /// @param[in]  G            The eddy viscoisty L2 projection coefficients
        /// @param[in]  snapshotTimes A list of Eigen vectors containing the time instants for each snapshot
        ///
        /// @return     The matrix of the combined velocity L2 projection coefficients starting from the second
        /// snapshots for each parameter sample together with their time derivatives and also the matrix of the
        /// eddy viscosity L2 projection coefficients associated with the first matrix
        ///
        List<Eigen::MatrixXd> velDerivativeCoeff(const Eigen::MatrixXd& A,
                const Eigen::MatrixXd& G, const List<Eigen::VectorXd>& snapshotTimes);

        //--------------------------------------------------------------------------
        /// @brief     Splits the eddy viscosity field snapshots in average and fluctuating part
        ///
        void splitEddyViscositySnapshots();

        //--------------------------------------------------------------------------
        /// @brief     Carries out the RBF interpolation
        ///
        void offlineRBFInterpolation();

        //--------------------------------------------------------------------------
        /// @brief      Resets the fields after a truthSolve, reading from disk the initial conditions.
        ///
        void restart();

        //--------------------------------------------------------------------------
        /// @brief      Resizes the modes before the Online phase, according to the number of modes chosen
        ///
        void resizeModes();

        //--------------------------------------------------------------------------
        /// @brief      Assembles the POD modes
        ///
        void computePOD();

        //--------------------------------------------------------------------------
        /// @brief      Complete set up of the lift approach
        ///
        void setupLift();

        //--------------------------------------------------------------------------
        /// @brief      Switch off the AUTO_WRITE options so that successive ROM runs
        /// do not write U, and other fields. (They write U_r, ...)
        ///
        void switchOffAutoWrite();

        //--------------------------------------------------------------------------
        /// @brief      Loads the fields from disk
        ///
        void loadFieldsFromDisk(word offlinepath = "./ITHACAOutput/Offline");
        
    private:
        ///- Reads the dictionary ITHACAdict and sets the relevant parameters
        void readITHACAdict();
        
        //--------------------------------------------------------------------------
        /// @brief      Centers the snapshots before POD
        void removeMean();
        
        //--------------------------------------------------------------------------
        /// @brief      Solves the lift problem for the velocity
        ///
        void liftSolve();

        //--------------------------------------------------------------------------
        /// @brief      Solves the lift problem for the temperature
        ///
        void liftSolveT();

        //--------------------------------------------------------------------------
        /// @brief      Homogenizes the snapshots for flux
        ///
        void homogenizePhi(
          const PtrList<surfaceScalarField>& phifields,
          const PtrList<surfaceScalarField>& phifieldsLift,
          PtrList<surfaceScalarField>& phifieldsHomogenized);

        //--------------------------------------------------------------------------
        /// @brief      Computes the phi modes with the same linear combination of the velocity modes
        ///
        void getPhiModes(
          volVectorModes& Umodes,
          PtrList<volVectorField>& Uomfield,
          surfaceScalarModes& Phimodes,
          PtrList<surfaceScalarField>& Phiomfield
        );

    private:
        //- Number of velocity modes
        label NUmodes_;

        //- Number of pressure modes
        label NPrghmodes_;

        //- Number of temperature modes
        label NTmodes_;

        //- Number of eddy viscosity modes
        label NNutModes_;

        ///- Reference cell for pressure
        label pRefCell;
        scalar pRefValue;
        
        /// Identifier for bcMethod, which can be "lift", "penalty" or "Gunzburger"
        word bcMethod;

        ///- Whether we mean-center
        bool centerSnapshots;

        /// adjustTimeStep
        bool adjustTimeStep;

        /// maxCourant
        scalar maxCo;

        /// maxDeltaT
        scalar maxDeltaT;
    
    protected:
        /// List of time snapshots for each parameter sample
        List<Eigen::VectorXd> timeSnapshots;
    
    public:
        //- Velocity snapshots
        PtrList<volVectorField> Ufield;

        //- Homogenized velocity snapshots
        PtrList<volVectorField> Uomfield;

        //- Flux field
        PtrList<surfaceScalarField> Phifield;

        //- Flux homogenized field
        PtrList<surfaceScalarField> Phiomfield;

        //- Lift fields for velocity
        PtrList<volVectorField> liftfield;

        //- Lift homogenized fields for phi
        PtrList<surfaceScalarField> liftfieldphi;

        //- Pressure snapshots
        PtrList<volScalarField> Prghfield;

        //- Temperature snapshots
        PtrList<volScalarField> Tfield;

        //- Homogenized temperature snapshots
        PtrList<volScalarField> Tomfield;

        //- Lift fields for temperature
        PtrList<volScalarField> liftfieldT;

        /// Snapshots for eddy viscosity
        PtrList<volScalarField> Nutfield;

        /// Snapshots for turbulent kinetic energy
        PtrList<volScalarField> kfield;

        /// Snapshots for omega
        PtrList<volScalarField> omegafield;

        //- Velocity POD modes
        volVectorModes Umodes;

        //- Flux POD modes
        surfaceScalarModes Phimodes;

        //- Pressure POD modes
        volScalarModes Prghmodes;

        //- Temperature POD modes
        volScalarModes Tmodes;

        //- Eddy viscosity POD modes
        volScalarModes nutmodes;

        //- Kinetic energy POD modes
        volScalarModes kmodes;

        //- Omega POD modes
        volScalarModes omegamodes;

        // Snapshot average of velocity
        autoPtr<volVectorField> UavgPtr;

        // Snapshot average of temperature
        autoPtr<volScalarField> TavgPtr;

        // Snapshot average of pressure
        autoPtr<volScalarField> p_rghavgPtr;

        /// List of fluctuating eddy viscosity fields. One for each parameter sample
        PtrList<volScalarField> fluctNutfield;

        /// List of averaged eddy viscosity fields. One for each parameter sample
        PtrList<volScalarField> avgNutfield;

        /// Interpolators for nut coefficient interpolation
        std::vector<std::shared_ptr<ithacaInterpolator>> rbfSplines;

        // Dimension of the interpolation matrix A
        label dimA;

    public:
        ///- Laminar transport pointer
        autoPtr<singlePhaseTransportModel> _laminarTransport;

        ///- Turbulent transport pointer
        autoPtr<incompressible::turbulenceModel> turbulence; // For conformity, can we rename this to _..., by changing the fields?

        ///- Pointer to the reduced velocity field
        autoPtr<volVectorField> _U;

        ///- Pointer to the ULiftBC field
        autoPtr<volScalarField> _UliftBC;
        
        ///- Pointer to the reduced face flux field
        autoPtr<surfaceScalarField> _phi;
        
        ///- Pointer to the pressure field
        autoPtr<volScalarField> _p;
        
        ///- Pointer to the reduced p_rgh
        autoPtr<volScalarField> _p_rgh;
        
        ///- Pointer to the reduced temperature field
        autoPtr<volScalarField> _T;
        
        ///- Pointer to kinematic density
        autoPtr<volScalarField> _rhok;
        
        ///- Pointer to the kinematic turbulent thermal conductivity
        autoPtr<volScalarField> _alphat;
        
        ///- Pointer to the turbulent viscosity field
        autoPtr<volScalarField> _nut;

        ///- Pointer to the turbulent kinetic energy field
        autoPtr<volScalarField> _k;

        ///- Pointer to the turbulent omega field
        autoPtr<volScalarField> _omega;
        
        ///- Pointer to the kinematic viscosity field
        autoPtr<dimensionedScalar> _nu;

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

};

#endif
