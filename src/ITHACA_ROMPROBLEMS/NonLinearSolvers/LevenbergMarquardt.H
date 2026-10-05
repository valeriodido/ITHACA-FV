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
    LevenbergMarquardt
Description
    Class holding the functor to enable usage of Eigen's Levenberg-Marquardt 
    algorithm for solving non-linear least squares problems.
    Used in the solution of Least Squares Petrov-Galerkin (LSPG) reduced order models.
    TODO: Check MPI parallelization
SourceFiles
    LevenbergMarquardt.H
\*---------------------------------------------------------------------------*/

/// \file
/// Header file of the LevenbergMarquardt class.
/// \dir
/// Directory containing the header and source files for the LevenbergMarquardt class.

#ifndef LevenbergMarquardt_H
#define LevenbergMarquardt_H

#include <Eigen/Dense>
#include <ReducedLSPG.H>
#include <unsupported/Eigen/NonLinearOptimization>
#include <unsupported/Eigen/NumericalDiff>

/// @brief Functor class for the Levenberg-Marquardt algorithm using Eigen's Levenberg-Marquardt implementation.
struct LevenbergMarquardtFunctor
{
  using Scalar = double;
  using InputType = Eigen::VectorXd;
  using ValueType = Eigen::VectorXd;
  using JacobianType = Eigen::MatrixXd;

  LevenbergMarquardtFunctor(ReducedLSPG* reducedProblem, int n_inputs, int n_values)
    : reducedProblem_(reducedProblem), n_inputs(n_inputs), n_values(n_values) {}

  enum {
    InputsAtCompileTime = Eigen::Dynamic,
    ValuesAtCompileTime = Eigen::Dynamic
  };

  /// @brief  Pointer to the ReducedLSPG problem instance.
  ReducedLSPG* reducedProblem_;
  
  /// @brief  Number of inputs (size of the input vector).
  int n_inputs;

  /// @brief  Number of values (size of the output vector).
  int n_values;

  int inputs() const { return n_inputs; }
  int values() const { return n_values; }

  int operator()(const Eigen::VectorXd& y, Eigen::VectorXd& fvec) const
  {
    const Eigen::VectorXd x = reducedProblem_->getStateScaling().cwiseProduct(y);
    fvec = reducedProblem_->assembleResidual(x);
    return 0;
  }
};

/// @brief Converts the Levenberg-Marquardt status enum to a string representation.
std::string lmStatusToString(Eigen::LevenbergMarquardtSpace::Status status);

#endif
