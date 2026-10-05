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
    ReducedLSPG
Description
    Class where the base of a Least-Squares Petrov-Galerkin reduced problem for the unsteady flow is implemented.
SourceFiles
    ReducedLSPG.C
\*---------------------------------------------------------------------------*/

#include "ReducedLSPG.H"

ReducedLSPG::ReducedLSPG(std::shared_ptr<ITHACAcontext> context)
: context_(std::move(context))
{
  if (!context_)
  {
    FatalErrorInFunction
      << "Context pointer is null. Please provide a valid ITHACAcontext object."
      << exit(FatalError);
  }
  ITHACAdict = new IOdictionary
  (
      IOobject
      (
          "ITHACAdict",
          runTime().system(),
          mesh(),
          IOobject::MUST_READ,
          IOobject::NO_WRITE
      )
  );

  romID_ = ITHACAdict->lookupOrDefault<word>("romID", "");
}

argList& ReducedLSPG::args() const
{
    return context_->args();
}

fvMesh& ReducedLSPG::mesh() const
{
    return context_->mesh();
}

Time& ReducedLSPG::runTime() const
{
    return context_->runTime();
}

pimpleControl& ReducedLSPG::pimple() const
{
    return context_->pimple();
}

fv::options& ReducedLSPG::fvOptions() const
{
    return context_->fvOptions();
}

IOMRFZoneList& ReducedLSPG::MRF() const
{
    return context_->MRF();
}
