#include "AffineReference.H"

AffineReference::AffineReference(const AffineReferenceSettings& settings, const Eigen::MatrixXd& mu)
    : settings_(settings),
      idw_(mu, settings.idwExponent, settings.idwNeighbours)
{
    if (settings_.type == AffineReferenceType::InitialCondition)
    {
        Info << "### Affine reference: inverse distance weighting, exponent " << settings_.idwExponent << ", "
             << (settings_.idwNeighbours > 0 ? std::to_string(settings_.idwNeighbours) + " nearest runs"
                                             : std::string("all runs")).c_str()
             << endl;
    }
}

Eigen::VectorXd AffineReference::weights(const Eigen::VectorXd& bcs) const
{
    switch (settings_.type)
    {
        case AffineReferenceType::Lift:
            return bcs;
        case AffineReferenceType::InitialCondition:
            return idw_(bcs);
        default:
            return Eigen::VectorXd();
    }
}

void AffineReference::update(const Eigen::VectorXd& bcs, UPtrList<ReducedFieldBase>& fields)
{
    if (settings_.type == AffineReferenceType::None)
    {
        return;
    }
    if (valid_ && bcs.size() == bcs_.size() && bcs == bcs_)
    {
        return;
    }
    const Eigen::VectorXd w = weights(bcs);
    for (ReducedFieldBase& f : fields)
    {
        f.setReferenceWeights(w);
    }
    bcs_ = bcs;
    valid_ = true;
}
