/*---------------------------------------------------------------------------*\
     ██╗████████╗██╗  ██╗ █████╗  ██████╗ █████╗       ███████╗██╗   ██╗
     ██║╚══██╔══╝██║  ██║██╔══██╗██╔════╝██╔══██╗      ██╔════╝██║   ██║
     ██║   ██║   ███████║███████║██║     ███████║█████╗█████╗  ██║   ██║
     ██║   ██║   ██╔══██║██╔══██║██║     ██╔══██║╚════╝██╔══╝  ╚██╗ ██╔╝
     ██║   ██║   ██║  ██║██║  ██║╚██████╗██║  ██║      ██║      ╚████╔╝
     ╚═╝   ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝╚═╝  ╚═╝      ╚═╝       ╚═══╝

 * In real Time Highly Advanced Computational Applications for Finite Volumes
 * Copyright (C) 2017 by the ITHACA-FV authors
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
    LSPGProblem
Description
    Base class for a LSPG Full Order Model problem
SourceFiles
    LSPGProblem.C
\*---------------------------------------------------------------------------*/

/// \file
/// Header file of the LSPGProblem class.
/// \dir
/// Directory containing the header and source files for the LSPGProblem class.

#ifndef LSPGProblem_H
#define LSPGProblem_H

#include "fvCFD.H"
#include "reductionProblem.H"
#include "ITHACAcontext.H"

/*---------------------------------------------------------------------------*\
                        Class LSPGProblem Declaration
\*---------------------------------------------------------------------------*/

/* 
In this class are implemented the methods for the offline solve of a steady NS problem.
This class is oriented to LSPG reduction, and implements a new way of sharing data between FOM and ROM so that ROM can outlive the FOM.
This include smart ways of passing necessary context (mesh and time), modes, and eventually, snapshots, to the ROM solver.
*/
class LSPGProblem: public reductionProblem
{
    public:
        // Constructor that takes in a context object, which contains the mesh, time, fvOptions, and other necessary objects for the FOM problem.
        explicit LSPGProblem(std::shared_ptr<ITHACAcontext> context);

        // Convenience constructor that takes in argc and argv, and creates a context object internally.
        LSPGProblem(int argc, char* argv[]);

        virtual ~LSPGProblem() = default;

        ITHACAcontext& context() const
        {
            return *context_;
        }

        argList& args() const
        {
            return context_->args();
        }

        fvMesh& mesh() const
        {
            return context_->mesh();
        }

        Time& runTime() const
        {
            return context_->runTime();
        }

        pimpleControl& pimple() const
        {
            return context_->pimple();
        }

        fv::options& fvOptions() const
        {
            return context_->fvOptions();
        }

        IOMRFZoneList& MRF() const
        {
            return context_->MRF();
        }

    protected:
        std::shared_ptr<ITHACAcontext> context_;

};

#endif
