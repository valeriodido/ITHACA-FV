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
#include "ReducedLSPG.H"
#include "LevenbergMarquardt.H"
#include "ITHACAassert.H"
#include "ITHACAsystem.H"
#include "Foam2Eigen.H"
#include <chrono>
#include <fstream>
#include <iomanip>

namespace
{
std::shared_ptr<ITHACAcontext> checkedContext(std::shared_ptr<ITHACAcontext> context)
{
    if (!context)
    {
        FatalErrorInFunction
            << "Context pointer is null. Please provide a valid ITHACAcontext object."
            << exit(FatalError);
    }
    return context;
}
}

ReducedLSPG::ReducedLSPG(std::shared_ptr<ITHACAcontext> context): context_(checkedContext(std::move(context))),
                                                                  collector_(residual_, context_->mesh())
{
    ITHACAdict.reset(new IOdictionary(
        IOobject(
            "ITHACAdict",
            runTime().system(),
            mesh(),
            IOobject::MUST_READ,
            IOobject::NO_WRITE)));
    romSettings_ = ROMSettings::read(*ITHACAdict);
    affineSettings_ = AffineReferenceSettings::read(*ITHACAdict);
    lmSettings_ = LMSolverSettings::read(ITHACAdict->subOrEmptyDict("lmSolver"));
    Info << "### LM stopping criteria: gradientTol " << lmSettings_.gradientTol << ", reductionTol "
         << lmSettings_.reductionTol << ", stepTol " << lmSettings_.stepTol << ", maxIterations "
         << lmSettings_.maxIterations << endl;

    collector_.setPreconditioner(romSettings_.residualPreconditioner);
    nCells = mesh().nCells();
}


void ReducedLSPG::setReportedBlocks(const labelList& blocks, const wordList& names)
{
    M_Assert(blocks.size() == names.size(), "setReportedBlocks: one name per block is needed.");
    reportedBlocks_ = blocks;
    reportedBlockNames_ = names;
}

void ReducedLSPG::setCellVolumeRowWeights(const label block, const label nComponents)
{
    // Used to approximate a L2 norm of the residual block
    const Eigen::VectorXd w = Foam2Eigen::field2Eigen(mesh().V().field()).cwiseSqrt().cwiseInverse();
    Eigen::VectorXd rowWeights(nComponents * w.size());
    for (label celli = 0; celli < w.size(); celli++)
    {
        rowWeights.segment(nComponents * celli, nComponents).setConstant(w(celli));
    }
    residual_.setBlockRowWeights(block, rowWeights);
}

double ReducedLSPG::reportedBlockNorm(const label i) const
{
    // If the block is not present we report 0. This is useful for when we have optional blocks that 
    // are activated at runtime (for example via ITHACAdict).
    return reportedBlocks_[i] >= 0 ? residual_.blockNorm(reportedBlocks_[i]) : 0.0;
}

Eigen::VectorXd ReducedLSPG::residualBlockWeights() const
{
    Eigen::VectorXd weights(reportedBlocks_.size());
    forAll(reportedBlocks_, i)
    {
        weights(i) = reportedBlocks_[i] >= 0 ? residual_.blockWeight(reportedBlocks_[i]) : 0.0;
    }
    return weights;
}

void ReducedLSPG::printBlockNorms() const
{
    forAll(reportedBlocks_, i)
    {
        Info << "  - " << reportedBlockNames_[i] << " residual norm " << reportedBlockNorm(i) << nl;
    }
}


void ReducedLSPG::runOnline(const word& folder)
{
    if (Pstream::master())
    {
        mkDir(folder);
        ITHACAutilities::createSymLink(folder);
    }
    // Residual fields go to the ROM case folder (folder/0 is a link to the case 0 directory)
    collector_.setOutputFolder(folder);
    Time& runTime = this->runTime();

#include "readTimeControls.H"

    initializePODCoeffsFromFields();
    collector_.requestPreconditionerUpdate();
    assembleResidual(currentState_, true, false);

    Info << "\nInitial residual norm: " << residual_.norm() << nl;
    printBlockNorms();
    Info << endl;
    if (romSettings_.residualPreconditioner == ResidualPreconditionerType::None)
    {
        residual_.calibrateWeights(1e-4);
    }

    // Per-step residual history, for post-processing (e.g. the consistency test)
    std::ofstream residualLog;
    std::string blockNames;
    forAll(reportedBlockNames_, i)
    {
        blockNames += std::string(i ? " " : "") + reportedBlockNames_[i];
    }
    if (Pstream::master())
    {
        residualLog.open(folder + "/residualHistory.dat");
        residualLog << std::scientific << std::setprecision(8)
                    << "# deltaT " << runTime.deltaTValue() << nl
                    << "# block weights (" << blockNames << "): " << residualBlockWeights().transpose() << nl
                    << "# gradientCosine: max_j |J_j^T r| / (|J_j| |r|) at the last Jacobian evaluation" << nl
                    << "# time initialTotal finalTotal";
        for (const word& name : reportedBlockNames_)
        {
            residualLog << " " << name;
        }
        residualLog << " lmIterations nfev gradientCosine status" << std::endl;
    }

    // The functor provides its own finite-difference Jacobian (LevenbergMarquardtFunctor::df)
    // At the moment i don't think this is a big advantage over Eigen's df, but it allows a bit
    // more freedom in the future.
    LMDriver lm(this, currentCoords_.size(), residual_.size(), lmSettings_);

    const scalar writeInterval = runTime.controlDict().get<scalar>("writeInterval");
    scalar nextWrite = runTime.value() + writeInterval;

    auto start = std::chrono::high_resolution_clock::now();
    while (runTime.run())
    {
        runTime++;
        Info << "Time = " << runTime.time().value() << nl;
        updateBoundaryConditions(runTime.time().value());

        Eigen::VectorXd scaledState = currentCoords_.cwiseQuotient(stateScaling_);
        // The first residual evaluation of the solve builds the preconditioner of the time step
        collector_.requestPreconditionerUpdate();
        const LMStepResult result = lm.solve(scaledState);
        const std::string status = lmStatusToString(result.status);
        if (result.status == Eigen::LevenbergMarquardtSpace::Running)
        {
            Info << "LM reached maximum iterations (" << lmSettings_.maxIterations << ") without converging." << nl;
        }
        Info << "LM stopped after " << result.iterations << " iterations (status = " << status
             << ", gradient cosine = " << lm.gnorm() << ")" << nl;

        diagnoseSolve(lm.functor(), scaledState, status, folder);

        currentCoords_ = scaledState.cwiseProduct(stateScaling_);
        currentState_ = fullState(scaledState);
        // Final assembly at the converged state; optionally writes the residual fields in folder/<time>
        assembleResidual(currentState_, true, romSettings_.writeResidualFields);

        printBlockNorms();
        Info << "  - Initial residual norm: " << result.initialNorm << nl
             << "  - Final residual norm: " << residual_.norm() << nl
             << "  - Function evaluations: " << lm.nfev() << nl;
        timeStepCompleted();
        Info << endl;

        // The norms are reductions over the processors: computed on every processor, written by the master
        const scalar finalNorm = residual_.norm();
        scalarList blockNorms(reportedBlocks_.size());
        forAll(reportedBlocks_, i)
        {
            blockNorms[i] = reportedBlockNorm(i);
        }
        if (Pstream::master())
        {
            residualLog << runTime.value() << " " << result.initialNorm << " " << finalNorm;
            for (const scalar norm : blockNorms)
            {
                residualLog << " " << norm;
            }
            residualLog << " " << result.iterations << " " << lm.nfev() << " " << lm.gnorm() << " "
                        << status << std::endl;
        }

        if (runTime.value() >= nextWrite - 0.5 * runTime.deltaTValue())
        {
            writeFields(folder);
            nextWrite += writeInterval;
        }
    }
    collector_.setOutputFolder("");
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    Info << "Total computation time: " << duration.count() << " ms" << endl;
}


argList& ReducedLSPG::args() const
{
    return context_->args();
}

fvMesh& ReducedLSPG::mesh() const
{
    return context_->mesh();
}

Time& ReducedLSPG::runTime() const
{
    return context_->runTime();
}

pimpleControl& ReducedLSPG::pimple() const
{
    return context_->pimple();
}

fv::options& ReducedLSPG::fvOptions() const
{
    return context_->fvOptions();
}

IOMRFZoneList& ReducedLSPG::MRF() const
{
    return context_->MRF();
}
