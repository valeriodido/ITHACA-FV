
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
#include "fixedValueFvPatchFields.H"
#include "nearWallDist.H"
#include <vector>

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
            const scalar boundFactor = 0.1,
            const word& GName = "ResidualAssemblerSST:G");

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

        virtual List<boolList> clippedCells() const override
        {
            return List<boolList>({ kClipped_, omegaClipped_ });
        }

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

        //- Lower bound of k. kMin_ and omegaMin_ are boundFactor times the smallest positive value of the
        //  initial and training fields: the FOM floor (SMALL) is harmless where k and omega come from a
        //  solve, but a reconstructed omega can sit at it, and 1/omega terms (CDkOmega, F1) then reach 1e15
        dimensionedScalar kMin_;

        //- Smooth lower bound of a field: m + m softplus((x - m)/m), identical to x for x >> m.
        //  clipped marks the cells whose reconstructed value is below fieldMin (diagnostics)
        static void boundField(volScalarField& field, const dimensionedScalar& fieldMin, boolList& clipped);

        // Dirichlet (fixedValue) patch values, cached from the initial fields.
        // Mode reconstruction overwrites them, and += / -= are no-ops on fixedValue patches,
        // so they must be re-imposed after every reconstruction.
        labelList kDirichletPatches_;
        List<scalarField> kDirichletValues_;
        labelList omegaDirichletPatches_;
        List<scalarField> omegaDirichletValues_;

        static void cacheDirichletValues(
            const volScalarField& field,
            labelList& patches,
            List<scalarField>& values);

        static void imposeDirichletValues(
            volScalarField& field,
            const labelList& patches,
            const List<scalarField>& values);

        // omega wall functions, reimplemented on the ROM fields.
        // OpenFOAM's omegaWallFunction looks up the turbulenceModel registered on the mesh,
        // which is the FOM one: it would compute the near-wall omega and G from the FOM k, nut, U.
        // The omegaWallFunction patches of omega_ are therefore replaced by fixedValue patches
        // and their behaviour (OpenFOAM v2412 omegaWallFunctionFvPatchScalarField) is reproduced here.
        struct OmegaWallPatch
        {
            label patchi;
            scalarField cornerWeights;
            word blending;
            scalar n;
            scalar beta1;
            scalar Cmu;
            scalar kappa;
            scalar yPlusLam;
        };

        std::vector<OmegaWallPatch> omegaWallPatches_;

        //- Unique cells adjacent to omega wall-function patches (rows fixed in the omega equation)
        labelList omegaWallCells_;

        //- Cells clipped by the last boundTurbulenceFields call (diagnostics)
        boolList kClipped_;
        boolList omegaClipped_;

        //- Wall distance of the near-wall cell centres
        const nearWallDist yWall_;

        //- Replaces the omegaWallFunction patches and stores their coefficients
        void setupOmegaWallFunctions();

        //- Sets omega and G in the wall-adjacent cells and omega on the wall faces
        void applyOmegaWallFunctions(volScalarField::Internal& G);

        // Blending
        template <class T>
        T blend(const T& F1, const dimensionedScalar& a, const dimensionedScalar& b) const;

        tmp<volScalarField> F1(const volScalarField& CDkOmega) const;

        tmp<volScalarField> F23() const;
    };

}
}

#endif
