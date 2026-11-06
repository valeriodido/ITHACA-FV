#ifndef NLSINTERFACE_H
#define NLSINTERFACE_H

#include "scalar.H"
#include "label.H"
#include <Eigen/Dense>


namespace ITHACAnls
{

// Conventions for parallel:
// r, J are local to the processor
// f, g, R, c are global quantities, identical on all processors
// f = |r|^2, g = J^T r

struct Linearization {
    const Eigen::MatrixXd& J;
    const Eigen::VectorXd& r;
    const Eigen::VectorXd& y;
    scalar f;
    bool jacobianFresh;
};

class LeastSquaresProblem
{
public:
    virtual ~LeastSquaresProblem() = default;
    virtual label nCoordinates() const = 0;
    virtual label nResiduals() const = 0; // local rows
    virtual void residual(
        const Eigen::VectorXd& x,
        Eigen::VectorXd& r) const = 0;
    virtual void beginSolve() { }
};

class JacobianProvider
{
public:
    virtual ~JacobianProvider() = default;
    virtual const Eigen::MatrixXd& evaluate(
        LeastSquaresProblem& lsProblem,
        const Eigen::VectorXd& y,
        const Eigen::VectorXd& r) = 0;
    virtual void accept(
        const Eigen::VectorXd& delta,
        const Eigen::VectorXd& rOld,
        const Eigen::VectorXd& rNew) { }
    virtual void invalidate() { }
    virtual bool isFresh() const { return true; }
    virtual label residualEvaluations() const = 0;
};

class StepSolver
{
public:
    virtual ~StepSolver() = default;
    virtual void prepare(const Linearization& lin) = 0;
    virtual const Eigen::VectorXd& gradient() const = 0;
    virtual scalar gradientNorm() const = 0;
    virtual bool solve(scalar lambda, Eigen::VectorXd& delta) = 0;
    virtual scalar predictedReduction(const Eigen::VectorXd& delta) const = 0;
    virtual bool rankDeficient() const { return false; }
};

struct StepLengthResult {
    scalar alpha;
    bool accepted;
    label nfev;
};

class StepLengthRule
{
public:
    virtual ~StepLengthRule() = default;
    virtual StepLengthResult compute(
        LeastSquaresProblem& lsProblem,
        const Linearization& lin,
        const StepSolver& stepSolver,
        const Eigen::VectorXd& delta,
        Eigen::VectorXd& rTrial,
        scalar& fTrial) = 0;
};

}

#endif // NLSINTERFACE_H
