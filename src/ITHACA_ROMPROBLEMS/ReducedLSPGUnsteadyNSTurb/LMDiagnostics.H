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
    LMDiagnostics
Description
    Solver diagnostics of the LSPG ROM of the incompressible turbulent flow, run when the
    Levenberg-Marquardt solve of a time step stops with one of the trigger statuses (ITHACAdict,
    lmDiagnostics). See LMDiagnostics.C for what is written and how to read it.
    The diagnostics only evaluate the residual through the LM functor, and read the face flux and the
    cells clipped by the closure: they do not depend on the internals of the ROM.
SourceFiles
    LMDiagnostics.C
\*---------------------------------------------------------------------------*/

#ifndef LMDiagnostics_H
#define LMDiagnostics_H

#include "fvCFD.H"
#include "BlockResidual.H"
#include "LSPGSettings.H"
#include "LevenbergMarquardt.H"
#include <functional>

class LMDiagnostics
{
public:
    /// @param residual Residual filled by the functor evaluations
    /// @param blocks Blocks of the tables (-1: absent, reported as 0), with their names
    /// @param coordinateLabels Name of each reduced coordinate
    /// @param phi Face flux of the ROM, set by the functor evaluations
    /// @param clippedCells Cells clipped by the closure at the last evaluation, one mask per transported field
    LMDiagnostics(
        const LMDiagnosticsSettings& settings,
        const BlockResidual& residual,
        const labelList& blocks,
        const wordList& blockNames,
        const wordList& coordinateLabels,
        const surfaceScalarField& phi,
        std::function<List<boolList>()> clippedCells);

    //- Starts a new online solve: resets the number of dumps
    void reset()
    {
        nDumps_ = 0;
    }

    //- Whether the LM status of a time step triggers a diagnostic dump
    bool triggered(const std::string& status) const;

    //- Diagnoses the reduced problem at the point y where LM stopped. Writes in folder/lmDiagnostics/<time>/:
    //  the singular values of the Jacobian, residual scans along the Gauss-Newton and steepest-descent
    //  directions and along every reduced coordinate (kinks, jumps), the counts of k/omega cells crossing
    //  the clipping bound and of faces where phi changes sign, and a summary with a heuristic interpretation
    void run(
        const LevenbergMarquardtFunctor& functor,
        const Eigen::VectorXd& y,
        const std::string& status,
        const word& folder,
        const scalar time);

private:
    LMDiagnosticsSettings settings_;
    const BlockResidual& residual_;
    labelList blocks_;
    wordList blockNames_;
    wordList coordinateLabels_;
    const surfaceScalarField& phi_;
    std::function<List<boolList>()> clippedCells_;

    //- Dumps written so far in the current online solve
    label nDumps_ = 0;
};

#endif
