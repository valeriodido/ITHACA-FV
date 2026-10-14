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
\*---------------------------------------------------------------------------*/

#include "ITHACAcontext.H"
#include "LSPGUnsteadyNSTurb.H"
#include "ReducedLSPGUnsteadyNSTurb.H"
#include "ITHACAstream.H"
#include <memory>

#include "ITHACAutilities.H"
#include <Eigen/Dense>

class OfflineFOM : public LSPGUnsteadyNSTurb
{
public:
  explicit OfflineFOM(std::shared_ptr<ITHACAcontext> context)
      : LSPGUnsteadyNSTurb(context) {}

  void offlineSolve(word offlinepath, word timeSettingsFile)
  {
    if ((offline) && (ITHACAutilities::check_folder(offlinepath) == true))
    {
      loadFieldsFromDisk(offlinepath);
      ITHACAstream::ReadDenseMatrixList(timeSnapshots, "./ITHACAoutput/Offline/", "timeSnapshots");
    }
    else
    {
      List<scalar> mu_now(Pnumber);
      Vector<double> inl(1, 0, 0);
      timeSnapshots.setSize(Tnumber);
      label number_velocity_bc = inletIndex.rows();
      Eigen::MatrixXd time_settings = ITHACAstream::readMatrix(timeSettingsFile);

      M_Assert(mu.rows() == number_velocity_bc, "Number of rows in mu matrix must be equal to the total number of parameteric boundary conditions");
      M_Assert(mu.cols() == Tnumber, "Number of columns in mu matrix must be equal to the number of parameter samples (Tnumber)");
      M_Assert(time_settings.cols() == 4, "Time settings matrix must have 4 columns: startTime, finalTime, timeStep, writeEvery");
      M_Assert(time_settings.rows() == Tnumber, "Time settings matrix must have the same number of rows as the number of parameter samples (Tnumber)");

      volVectorField &U = _U();

      for (label solve_idx = 0; solve_idx < Tnumber; solve_idx++)
      {
        for (label j = 0; j < number_velocity_bc; j++)
        {
          inl[0] = mu(j, solve_idx);
          assignBC(U, inletIndex(j, 0), inl);
        }
        for (label j = 0; j < Pnumber; j++)
        {
          mu_now[j] = mu(j, solve_idx);
        }
        startTime = time_settings(solve_idx, 0);
        finalTime = time_settings(solve_idx, 1);
        timeStep = time_settings(solve_idx, 2);
        writeEvery = time_settings(solve_idx, 3);
        truthSolve(mu_now, solve_idx);
        restart();
      }
      if (Pstream::master())
      {
        ITHACAstream::SaveDenseMatrixList(timeSnapshots, offlinepath, "timeSnapshots");
      }
    }
  }
};

template <typename FieldType>
void compareFields(
    const word &path1,
    const word &fieldName1,
    const word &path2,
    const word &fieldName2,
    const word &outputFile,
    const std::shared_ptr<ITHACAcontext> &context,
    const word &caseID,
    const word &romID)
{
  PtrList<FieldType> field1 = context->loadFieldsFromCaseFolder<FieldType>(fieldName1, path1);
  PtrList<FieldType> field2 = context->loadFieldsFromCaseFolder<FieldType>(fieldName2, path2);

  Eigen::MatrixXd l2Error = ITHACAutilities::errorL2Rel(field1, field2);
  ITHACAstream::exportMatrix(l2Error, outputFile, "eigen", "./ITHACAoutput/l2error/" + romID + caseID);
  Info << "-- Maximum L2 error for field " << fieldName1 << " and " << fieldName2 << ": " << l2Error.maxCoeff() << nl
       << "-- Average L2 error for field " << fieldName1 << " and " << fieldName2 << ": " << l2Error.mean() << endl;
}

int main(int argc, char *argv[])
{
  // Create the context, which reads mesh, time and so on
  auto context = std::make_shared<ITHACAcontext>(argc, argv);
  OfflineFOM offlineFOM(context);

  offlineFOM.prepareParametersMu("./input_parameters/offline/BC");
  offlineFOM.inletIndex.resize(1, 2);
  offlineFOM.inletIndex << 0, 0;

  offlineFOM.offlineSolve("./ITHACAoutput/Offline/", "./input_parameters/offline/timeSettings");
  offlineFOM.prepareSnapshots(); // Subtracts the initial condition from the snaps if promptedto
  offlineFOM.splitEddyViscositySnapshots();
  offlineFOM.computePOD(); // Automatically lifts if needed

  offlineFOM.prepareModes();
  offlineFOM.offlineRBFInterpolation();
  offlineFOM.resizeModes();
  offlineFOM.switchOffAutoWrite();

  std::vector<std::string> suffixes = {"01"};
  std::vector<std::string> fieldNames = {"U", "p", "nut"};
  std::vector<std::string> romFieldNames = {"U_r", "p_r", "nut_r"};

  ReducedLSPGUnsteadyNSTurb reducedProblem(context, offlineFOM);

  for (int i = 0; i < suffixes.size(); i++)
  {
    word bcROM("./input_parameters/ROM/ROM_BC_" + suffixes[i]);
    word casePath("./ITHACAoutput/OnlineLSPGROM" + suffixes[i] + "_" + reducedProblem.getID() + "/");
    word fomCasePath("./ITHACAoutput/OnlineFOM" + suffixes[i] + "/");

    reducedProblem.solveOnline(bcROM, casePath);

    for (int j = 0; j < fieldNames.size(); j++)
    {
      if (fieldNames[j] == "U")
      {
        compareFields<volVectorField>(fomCasePath, fieldNames[j], casePath, romFieldNames[j], "L2ErrorLSPGU", context, reducedProblem.getID(), suffixes[i]);
      }
      else
      {
        compareFields<volScalarField>(fomCasePath, fieldNames[j], casePath, romFieldNames[j], "L2ErrorLSPG" + fieldNames[j], context, reducedProblem.getID(), suffixes[i]);
      }
    }
  }
  return 0;
}
