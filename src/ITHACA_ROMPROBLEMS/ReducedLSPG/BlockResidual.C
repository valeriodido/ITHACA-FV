#include "ITHACAassert.H"
#include "BlockResidual.H"

BlockResidual::BlockResidual(std::vector<BlockSpec> specs)
{
  Eigen::Index offset = 0;
 
  blocks_.reserve(specs.size());
  
  for (BlockSpec& spec : specs)
  {
    M_Assert(spec.size > 0, "Block has size <= 0");
    M_Assert(spec.weight > 0, "Block has weight <= 0");
    M_Assert(spec.rowWeights.size() == 0 || spec.rowWeights.size() == spec.size, "Block has rowWeights size mismatch with block size");
    // Should actually be:
    //  M_Assert(spec.rowWeights.size() == 0 || spec.rowWeights.size() == spec.size, ...);
    for (const Block& otherblock : blocks_)
    {
      M_Assert(spec.name != otherblock.name, "Duplicate block name found.");
    }

    Block block;
    block.name = spec.name;
    block.offset = offset;
    block.size = spec.size;
    block.weight = spec.weight;
    block.rowWeights = spec.rowWeights;
    block.rowWeights = std::move(spec.rowWeights);
    blocks_.push_back(std::move(block));

    offset += spec.size;
  }

  raw_ = Eigen::VectorXd::Zero(offset);
  scaled_ = raw_;
  isScaled_ = false;
}

int BlockResidual::size() const
{
  return static_cast<int>(raw_.size());
}

const BlockResidual::Block& BlockResidual::at(int block) const
{
    M_Assert(block >= 0 && block < numBlocks(), "BlockResidual: block index out of range");
    return blocks_[static_cast<std::size_t>(block)];
}

int BlockResidual::index(const std::string& name) const
{
    for (std::size_t i = 0; i < blocks_.size(); ++i)
    {
        if (blocks_[i].name == name)
        {
            return static_cast<int>(i);
        }
    }
    M_Assert(false, "BlockResidual: block name not found");
    return -1;
}

void BlockResidual::set(int block_index, const Eigen::Ref<const Eigen::VectorXd>& raw_data)
{
    const Block& block = at(block_index);
    M_Assert(raw_data.size() == block.size, "BlockResidual: block has size mismatch with input data");
    raw_.segment(block.offset, block.size) = raw_data;
    isScaled_ = false;
}


Eigen::Ref<Eigen::VectorXd> BlockResidual::rawBlock(int block_index)
{
    const Block& block = at(block_index);
    isScaled_ = false;
    return raw_.segment(block.offset, block.size);
}

void BlockResidual::setBlockWeight(int block_index, double weight)
{
    at(block_index);
    blocks_[static_cast<std::size_t>(block_index)].weight = weight;
    isScaled_ = false;
}

void BlockResidual::setBlockRowWeights(int block_index, const Eigen::VectorXd& rowWeights)
{
    Block& block = blocks_[static_cast<std::size_t>(block_index)];
    M_Assert(rowWeights.size() == block.size, "BlockResidual: block has size mismatch with input rowWeights");
    block.rowWeights = rowWeights;
    isScaled_ = false;
}

int BlockResidual::numBlocks() const
{
    return static_cast<int>(blocks_.size());
}

void BlockResidual::calibrateWeights(double minRelative)
{
  const std::size_t nBlocks = blocks_.size();
  std::vector<double> norms(nBlocks);
  double maxNorm = 0.0;

  for (std::size_t i = 0; i < nBlocks; ++i)
  {
    const Block& block = blocks_[i];
    double sq = 0.0;
    if (block.size > 0)
    {
      const auto seg = raw_.segment(block.offset, block.size);
      sq = (block.rowWeights.size() > 0)
        ? (block.rowWeights.array() * seg.array()).square().sum()
        : seg.squaredNorm();
    }
    norms[i] = std::sqrt(sq); // Watch out parallelization here
    maxNorm = std::max(maxNorm, norms[i]);
  }

  if (!(maxNorm > 0.0) || !std::isfinite(maxNorm))
  {
    return;
  }

  const double floorValue = minRelative * maxNorm;
  for (std::size_t i = 0; i < nBlocks; ++i)
  {
    blocks_[i].weight = 1.0 / std::max(norms[i], floorValue);
  }

  isScaled_ = false;
}

void BlockResidual::applyScaling() const
{
  if (isScaled_)
  {
    return;
  }

  for (const Block& block : blocks_)
  {
    if (block.size == 0)
    {
      continue;
    }
    const auto in = raw_.segment(block.offset, block.size);
    auto out = scaled_.segment(block.offset, block.size);
    if (block.rowWeights.size() > 0)
    {
      out = block.weight * in.cwiseProduct(block.rowWeights);
    }
    else
    {
      out = block.weight * in;
    }
  }
  isScaled_ = true;
}

double BlockResidual::norm() const
{
  // A OpenMPI safe way must be implemented here
  return vector().norm();
}

double BlockResidual::blockNorm(int block_index) const
{
  const Block& block = at(block_index);
  applyScaling();
  return scaled_.segment(block.offset, block.size).norm();
}

const Eigen::VectorXd& BlockResidual::vector() const
{
  applyScaling();
  return scaled_;
}
