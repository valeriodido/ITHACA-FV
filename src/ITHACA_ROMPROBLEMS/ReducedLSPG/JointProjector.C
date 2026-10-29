#include "JointProjector.H"
#include "ITHACAassert.H"

JointProjector::JointProjector(ReducedFieldSet& fields, const Eigen::VectorXd& weights)
{
    M_Assert(weights.size() == fields.size(), "JointProjector: one weight per field of the set is needed.");
    DynamicList<label> joint; // OpenFOAM growable lists
    for (label f = 0; f < fields.size(); f++)
    {
        if (fields[f].joint())
        {
            joint.push_back(f);
        }
    }
    fields_.transfer(joint);
    M_Assert(fields_.size() > 0, "JointProjector: no field of the set is in the stacked POD.");

    // G = sum_f w_f F_f^T V_f F_f, each term already summed over the processors
    const label n = fields[fields_[0]].nModes();
    Eigen::MatrixXd G = Eigen::MatrixXd::Zero(n, n);
    weights_.setSize(fields_.size());
    forAll(fields_, j)
    {
        weights_[j] = weights(fields_[j]);
        G += weights_[j] * fields[fields_[j]].gram();
    }
    gram_.compute(G);
}

Eigen::MatrixXd JointProjector::solve(const List<Eigen::MatrixXd>& projections) const
{
    M_Assert(projections.size() == fields_.size(), "JointProjector: one projection per joint field is needed.");
    Eigen::MatrixXd B = Eigen::MatrixXd::Zero(gram_.rows(), projections[0].cols());
    forAll(projections, j)
    {
        B += weights_[j] * projections[j];
    }
    return gram_.solve(B); // Solves G a = B
}

Eigen::VectorXd JointProjector::coefficients(ReducedFieldSet& fields) const
{
    List<Eigen::MatrixXd> projections(fields_.size());
    forAll(fields_, j)
    {
        projections[j] = fields[fields_[j]].galerkinProjection();
        // Modes::project does not sum over the processors
        if (Pstream::parRun())
        {
            reduce(projections[j], sumOp<Eigen::MatrixXd>());
        }
    }
    return solve(projections).col(0);
}
