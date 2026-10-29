#include "InverseDistanceWeights.H"
#include "ITHACAassert.H"
#include <algorithm>
#include <vector>

InverseDistanceWeights::InverseDistanceWeights(
    const Eigen::MatrixXd& samples,
    const scalar exponent,
    const label neighbours)
    : samples_(samples),
      lower_(samples.rowwise().minCoeff()),
      range_(samples.rowwise().maxCoeff() - lower_),
      exponent_(exponent),
      neighbours_(neighbours)
{
    normalisedSamples_.resize(samples_.rows(), samples_.cols());
    for (label i = 0; i < samples_.cols(); i++)
    {
        normalisedSamples_.col(i) = normalised(samples_.col(i));
    }
}

Eigen::VectorXd InverseDistanceWeights::normalised(const Eigen::VectorXd& x) const
{
    Eigen::VectorXd y = Eigen::VectorXd::Zero(x.size());
    for (label j = 0; j < x.size(); j++)
    {
        y(j) = range_(j) > 1e-15 ? (x(j) - lower_(j)) / range_(j) : 0.0;
    }
    return y;
}

Eigen::VectorXd InverseDistanceWeights::operator()(const Eigen::VectorXd& parameters) const
{
    M_Assert(parameters.size() == samples_.rows(),
        "Input parameters size to the linear interpolation does not match the expected size.");
    const label nSamples = samples_.cols();
    Eigen::VectorXd weights(nSamples);

    const Eigen::VectorXd target = normalised(parameters);
    Eigen::VectorXd distances(nSamples);
    for (label i = 0; i < nSamples; i++)
    {
        distances(i) = std::pow((target - normalisedSamples_.col(i)).norm(), exponent_);
        if (distances(i) <= 1e-10)
        {
            // A training parameter
            weights.setZero();
            weights(i) = 1.0;
            return weights;
        }
    }

    scalar cutOff = GREAT;
    if (neighbours_ > 0 && neighbours_ < nSamples)
    {
        std::vector<scalar> sorted(distances.data(), distances.data() + nSamples);
        std::sort(sorted.begin(), sorted.end());
        cutOff = sorted[neighbours_ - 1];
    }
    for (label i = 0; i < nSamples; i++)
    {
        weights(i) = distances(i) <= cutOff ? 1.0 / distances(i) : 0.0;
    }

    const double weightSum = weights.sum();
    if (weightSum > 1e-10)
    {
        weights /= weightSum;
    } else
    {
        weights.setConstant(1.0 / nSamples);
    }

    return weights;
}
