
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
    kOmega
Description
    Turbulence model (k-OmegaSST) to be used to assemble the residual in a Least Squares Petrov-Galerkin
    Reduced Order Model.
    This leverages the existing OpenFOAM infrastructure, but does not solve the turbulence equations, only assembles the residual for the ROM to be minimized.
SourceFiles
    ITHACAkOmegaSST.C
\*---------------------------------------------------------------------------*/

/// \file
/// Header file of the ITHACAkOmegaSST class.
/// \dir
/// Directory containing the header and source files for the ITHACAkOmegaSST class/*

#ifndef ITHACAkOmegaSST_H
#define ITHACAkOmegaSST_H

#include "fvCFD.H"
#include "ResidualTurbulenceModel.H"
#include "fvOptions.H"
#include "wallDist.H"
#include "Modes.H"

namespace Foam
{
namespace ITHACATurbulenceModels
{
    class ResidualAssemblerSST : public ResidualEddyViscosityTurbulenceModel
    {
    public:
        ResidualAssemblerSST(
            const fvMesh& mesh,
            const Time& runTime,
            const SharedFields& shared,
            volScalarModes& kModes,
            volScalarModes& omegaModes,
            PtrList<volScalarField> k0Fields,
            PtrList<volScalarField> omega0Fields,
            const dictionary& kOmegaSSTCoeffs,
            const word& GName = "kOmegaSST:G");

        virtual ~ResidualAssemblerSST() = default;

        virtual const wordList& transportedFields() const override
        {
            return fieldNames_;
        }

        virtual void correctNut() override;

        virtual PtrList<fvScalarMatrix> assembleEquations(fv::options& fvOptions);

        virtual void reconstructTurbulenceFields(
            const Eigen::VectorXd& stateCoeffs) override;

        virtual void boundTurbulenceFields() override;

        virtual void computePODCoeffsFromFields(Eigen::Ref<Eigen::VectorXd> stateCoeffs) override;

        virtual void removeInitialZeroField(const Eigen::VectorXd weights) override;

        virtual void addInitialZeroField(const Eigen::VectorXd weights) override;

        virtual void saveFieldsToDisk(const Time& runTime, const word folder) const override;

    private:
        volScalarField& k_;
        volScalarModes& kModes_;
        PtrList<volScalarField> k0Fields_;
        volScalarField& omega_;
        volScalarModes& omegaModes_;
        PtrList<volScalarField> omega0Fields_;
                
        const word GName_;
        autoPtr<volScalarField::Internal> G_;
        const volScalarField y_;

        wordList fieldNames_;

        dimensionedScalar alphaK1_, alphaK2_, alphaOmega1_, alphaOmega2_;
        dimensionedScalar gamma1_, gamma2_, beta1_, beta2_, betaStar_;
        dimensionedScalar a1_, b1_, c1_;
        Switch F3_;
        dimensionedScalar omegaMin_;

        // Blending
        template <class T>
        T blend(const T& F1, const dimensionedScalar& a, const dimensionedScalar& b) const;

        tmp<volScalarField> F1(const volScalarField& CDkOmega) const;

        tmp<volScalarField> F23() const;
    };

}
}

#endif
