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
    ITHACAcontext
Description
    Sets up a shared context, which creates the Time and fvMesh objects.
    This can be shared/passed between ROMs and FOMs, so that one can outlive the other.
SourceFiles
    ITHACAcontext.C
\*---------------------------------------------------------------------------*/

/// \file
/// Header file of the ITHACAcontext class.
/// \dir
/// Directory containing the header and source files for the ITHACAcontext class.

#ifndef ITHACAcontext_H
#define ITHACAcontext_H

#include "fvCFD.H"
#include "pimpleControl.H"
#include "fvOptions.H"
#include "singlePhaseTransportModel.H"
#include "turbulentTransportModel.H"

class ITHACAcontext
{
  public:
    //- Constructor
    ITHACAcontext(int argc, char *argv[]);

    //- Destructor
    ~ITHACAcontext();

    //- Access to the argList
    argList& args();

    //- Access to the Time 
    Time& runTime(); 

    //- Access to the fvMesh
    fvMesh& mesh();

    //- Access to the pimpleControl
    pimpleControl& pimple();

    //- Access to the fvOptions
    fv::options& fvOptions();

    //- Access to the MRF
    IOMRFZoneList& MRF();


    /// @brief Loads the selected fields from the specified case folder, given by the context.
    /// @tparam FieldType The type of the field to be loaded (e.g., volVectorField, volScalarField, surfaceScalarField).
    /// @param fieldName The name of the field to be loaded.
    /// @param folder The folder (relative to the case root) from which to load the
    ///
    template <typename FieldType>
    PtrList<FieldType> loadFieldsFromCaseFolder(const word& fieldName, const fileName& folder);

  private:
    ///- Pointer to the argList object
    autoPtr<argList> _args;  
    
    ///- Pointer to the Time object
    autoPtr<Time> _runTime;

    ///- Pointer to the fvMesh object
    autoPtr<fvMesh> _mesh;
    
    ///- Pointer to the pimpleControl object
    autoPtr<pimpleControl> _pimple;
    
    ///- Pointer to the fvOptions object
    autoPtr<fv::options> _fvOptions;

    ///- Pointer to the MRF object
    autoPtr<IOMRFZoneList> _MRF;
};

#endif
