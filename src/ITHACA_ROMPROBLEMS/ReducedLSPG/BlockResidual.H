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
    BlockResidual
Description
    Base class to create a block of a residual vector. Blocks can be assembled together to form a residual (e.g. velocity, pressure, temperature, etc.)
SourceFiles
    BlockResidual.C
\*---------------------------------------------------------------------------*/

#ifndef BlockResidual_H
#define BlockResidual_H

#include <Eigen/Dense>

struct BlockSpec
{
    std::string name;
    Eigen::Index size;
    double weight = 1.0;
    Eigen::VectorXd rowWeights; // Optional per-entry weights
};

class BlockResidual
{
  public:
    /// @brief Default constructor, must provide a vector of BlockSpec
    ///
    BlockResidual() = default;

    /// @brief Constructs. 
    /// @param specs Vector of BlockSpec, each with a name, size and optional weight and rowWeights. The order of the blocks in the vector will be the order in which they are assembled in the residual.
    explicit BlockResidual(std::vector<BlockSpec> specs);

    /// @brief Returns the size of the residual vector
    /// @return int of the size of the residual vector
    int size() const;

    /// @brief Returns the index of a block given its name
    /// @param name Name of the block
    /// @return Index of the block in the vector of blocks
    int index(const std::string& name) const;

    /// @brief Fills the block, raw and unscaled
    /// @param block_index Index of the block to fill
    /// @param raw_data Data to fill the block with. Can be data.head(size) or data.tail(size) or any other Eigen::Ref of the correct size
    void set(int block_index, const Eigen::Ref<const Eigen::VectorXd>& raw_data);

    /// @brief Returns the raw block, unscaled
    /// @param block_index Index of the block to return
    /// @return Eigen::Ref<Eigen::VectorXd> of the raw block
    Eigen::Ref<Eigen::VectorXd> rawBlock(int block_index);

    /// @brief Returns the scaled residual, with weights applied. Ready for the solver.
    /// @return const Eigen::VectorXd& of the scaled block
    const Eigen::VectorXd& vector() const;

    /// @brief Returns a specific block, scaled and ready for the solver.
    /// @param block_index Index of the block to return
    /// @return Eigen::Ref<const Eigen::VectorXd> of the scaled block
    Eigen::Ref<const Eigen::VectorXd> block(int block_index) const;

    /// @brief Returns the norm
    /// @return double of the norm of the scaled residual
    double norm() const;

    /// @brief Returns the norm of a specific block
    /// @param block_index Index of the block to return the norm of
    /// @return double of the norm of the scaled block
    double blockNorm(int block_index) const;

    /// @brief Sets the weight of a specific block
    /// @param block_index Index of the block to set the weight of
    /// @param weight Weight to set
    void setBlockWeight(int block_index, double weight);

    /// @brief Sets the per-entry weights of a specific block
    /// @param block_index Index of the block to set the per-entry weights of
    /// @param rowWeights Eigen::VectorXd of the per-entry weights to set. Must
    void setBlockRowWeights(int block_index, const Eigen::VectorXd& rowWeights);

    int numBlocks() const;

    /// @brief Calibrates the weights as 1/||block||
    void calibrateWeights(double minRelative=1e-8);
    // Hyper-reduction will go here

  private:
    struct Block
    {
      std::string name;
      Eigen::Index offset = 0;
      Eigen::Index size = 0;
      double weight = 1.0;
      Eigen::VectorXd rowWeights; // Optional per-entry weights
    };

    /// @brief Returns a specific block
    const Block& at(int block_index) const;
    
    /// @brief Apply scaling
    void applyScaling() const;

    /// @brief Vector of blocks
    std::vector<Block> blocks_;

    /// @brief Raw residual vector
    Eigen::VectorXd raw_;

    // Caches
    mutable Eigen::VectorXd scaled_;

    /// @brief Boolean if the vector is scaled or not
    mutable bool isScaled_;
};

#endif
