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
    NonlinearLSSolver
Description
    Nonlinear least-squares solver, min_y |r(y)|^2, interface used by the ReducedLSPG time loop, and the
    base class of the solvers assembled from the parts of NLSInterfaces.H. The solver is selected at run
    time by NonlinearLSSolver::New from NLSSettings (ITHACAdict, nonlinearSolver).
SourceFiles
    NonlinearLSSolver.C
\*---------------------------------------------------------------------------*/

#ifndef NonlinearLSSolver_H
#define NonlinearLSSolver_H

#include "fvCFD.H"
#include "NLSInterfaces.H"
#include "NLSSettings.H"
#include <memory>

namespace ITHACAnls
{

/// @brief Interface of a nonlinear least-squares solver. The same object is reused for every time step.
class NonlinearLSSolver
{
public:
    virtual ~NonlinearLSSolver() = default;

    /// @brief Minimises |r(y)|^2 from the initial guess y; y is the solution on return
    virtual NLSResult solve(Eigen::VectorXd& y) = 0;

    //- Residual evaluations of the last solve, finite-difference Jacobians included
    virtual label nfev() const = 0;

    //- Jacobians of the last solve
    virtual label njev() const = 0;

    //- Largest cosine between the residual and a Jacobian column at the last iteration
    virtual scalar gradientNorm() const = 0;

    /// @brief Builds the solver selected by the settings
    static std::unique_ptr<NonlinearLSSolver> New(LeastSquaresProblem& problem, const NLSSettings& settings);
};


/// @brief Parts and bookkeeping that the iterative solvers share: the problem, the Jacobian, the linear algebra
/// of the steps, the settings and the counters of the evaluations. The loop itself (acceptance of the steps,
/// stopping) belongs to the solvers
class IterativeNLSSolver : public NonlinearLSSolver
{
public:
    label nfev() const override
    {
        return nfevDirect_ + jacobian_->residualEvaluations() - jacobianEvaluationsAtStart_;
    }

    label njev() const override
    {
        return njev_;
    }

    scalar gradientNorm() const override
    {
        return gradientNorm_;
    }

protected:
    IterativeNLSSolver(
        LeastSquaresProblem& problem,
        std::unique_ptr<JacobianProvider> jacobian,
        std::unique_ptr<StepSolver> stepSolver,
        const NLSSettings& settings);

    //- Resets the counters and tells the problem that a solve begins. To be called first by solve
    void beginSolve();

    //- Residual at y, returns |r|^2 over the processors
    scalar evaluate(const Eigen::VectorXd& y, Eigen::VectorXd& r);

    //- Counts residual evaluations that were not made by evaluate (e.g. by a step length rule)
    void nfevAdd(const label n)
    {
        nfevDirect_ += n;
    }

    //- Jacobian at y, where r is the residual
    const Eigen::MatrixXd& computeJacobian(const Eigen::VectorXd& y, const Eigen::VectorXd& r);

    LeastSquaresProblem& problem_;
    std::unique_ptr<JacobianProvider> jacobian_;
    std::unique_ptr<StepSolver> stepSolver_;
    const NLSSettings settings_;
    scalar gradientNorm_ = 0;

private:
    label nfevDirect_ = 0;
    label jacobianEvaluationsAtStart_ = 0;
    label njev_ = 0;
};

} // namespace ITHACAnls

#endif
