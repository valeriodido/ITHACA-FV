#include "StateMap.H"

StateMap::StateMap(const ReducedFieldSet& fields)
{
    label nJoint = -1; // Number of joint variables (i.e. the number of coordinates shared)
    for (label f = 0; f < fields.size(); f++)
    {
        if (fields[f].joint())
        {
            if (nJoint < 0)
            {
                nJoint = fields[f].nModes();
            }
            M_Assert(fields[f].nModes() == nJoint,
                "The fields of the stacked POD must have the same number of modes.");
        }
    }
    if (nJoint < 0)
    {
        return;
    }

    const label nState = fields.stateSize();
    // source_ is a vector that tells were to copy each coordinate into the full state vector.
    source_.setSize(nState);
    label col = nJoint;
    for (label f = 0; f < fields.size(); f++)
    {
        for (label i = 0; i < fields[f].nModes(); i++)
        {
            source_[fields[f].offset() + i] = fields[f].joint() ? i : col + i;
        }
        col += fields[f].joint() ? 0 : fields[f].nModes();
    }
    nCoordinates_ = col;

    copies_.setSize(nCoordinates_, 0);
    for (const label c : source_)
    {
        copies_[c]++;
    }
    Info << "### Stacked POD: " << nCoordinates_ << " reduced coordinates for a state of size " << nState << endl;
}

Eigen::VectorXd StateMap::expand(const Eigen::VectorXd& coordinates) const
{
    if (identity())
    {
        return coordinates;
    }
    Eigen::VectorXd state(source_.size());
    forAll(source_, i)
    {
        state(i) = coordinates(source_[i]);
    }
    return state;
}

Eigen::VectorXd StateMap::reduce(const Eigen::VectorXd& state) const
{
    if (identity())
    {
        return state;
    }
    Eigen::VectorXd coordinates = Eigen::VectorXd::Zero(nCoordinates_);
    forAll(source_, i)
    {
        coordinates(source_[i]) += state(i);
    }
    forAll(copies_, c)
    {
        coordinates(c) /= copies_[c];
    }
    return coordinates;
}
