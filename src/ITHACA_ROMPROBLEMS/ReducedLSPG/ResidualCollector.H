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
    ResidualCollector
Description
    Gathers the residuals of the separate equations of a model into the blocks of the LSPG residual.
    The model only states which equation goes in which block; for each block the collector
      - stores the residual in the BlockResidual,
      - refreshes the Jacobi row weights from the matrix diagonal, when requested (once per time step),
      - writes the residual field, when requested (residual_<block name>).
SourceFiles
    ResidualCollector.C
\*---------------------------------------------------------------------------*/

#ifndef ResidualCollector_H
#define ResidualCollector_H

#include "fvCFD.H"
#include "BlockResidual.H"
#include "LSPGSettings.H"
#include "ITHACAstream.H"

namespace LSPGResidualOutput
{
//- Writes a residual field: in folder/<time> if folder is not empty, else in the case time directory
template <class FieldType>
void exportField(FieldType& res, const word& folder)
{
    if (folder.empty())
    {
        res.write();
    } else
    {
        ITHACAstream::exportSolution(res, name(res.time().value()), folder);
    }
}

//- Writes the residual R of the equation eqn as a cell field. fvMatrix residuals are integrated over the
//  cell volume: perUnitVolume divides them by V
template <class Type>
void writeMatrixResidual(
    const word& name,
    const fvMatrix<Type>& eqn,
    const Field<Type>& R,
    const fvMesh& mesh,
    const word& folder,
    const bool perUnitVolume = true);
}


class ResidualCollector
{
public:
    ResidualCollector(BlockResidual& residual, const fvMesh& mesh)
        : residual_(residual), mesh_(mesh)
    {}

    void setPreconditioner(const ResidualPreconditionerType type)
    {
        preconditioner_ = type;
    }

    //- Folder of the residual fields (empty: case time directory)
    void setOutputFolder(const word& folder)
    {
        outputFolder_ = folder;
    }

    const word& outputFolder() const
    {
        return outputFolder_;
    }

    //- The next evaluation recomputes the Jacobi row weights
    void requestPreconditionerUpdate()
    {
        updatePreconditioner_ = true;
    }

    //- Starts the collection of one residual evaluation
    void begin(const bool writeFields)
    {
        writeFields_ = writeFields;
    }

    //- Residual R of the matrix equation eqn
    template <class Type>
    void add(const label block, const fvMatrix<Type>& eqn, const Field<Type>& R);

    //- Residual of a face block (internal faces), written per unit face area
    void add(const label block, const surfaceScalarField& R);

    //- Ends the collection: the Jacobi row weights are frozen until the next request
    void end()
    {
        if (preconditioner_ == ResidualPreconditionerType::Jacobi)
        {
            updatePreconditioner_ = false;
        }
        writeFields_ = false;
    }

private:
    //- Jacobi row weights sqrt(V) / (|a_ii| V), repeated for each component
    Eigen::VectorXd jacobiRowWeights(const scalarField& A, const label nComponents) const;

    word outputName(const label block) const
    {
        return "residual_" + word(residual_.name(block));
    }

    BlockResidual& residual_;
    const fvMesh& mesh_;
    ResidualPreconditionerType preconditioner_ = ResidualPreconditionerType::None;
    word outputFolder_;
    bool updatePreconditioner_ = true;
    bool writeFields_ = false;
};


template <class Type>
void ResidualCollector::add(const label block, const fvMatrix<Type>& eqn, const Field<Type>& R)
{
    const label nComponents = pTraits<Type>::nComponents;
    residual_.rawBlock(block) =
        Eigen::Map<const Eigen::VectorXd>(reinterpret_cast<const scalar*>(R.cdata()), R.size() * nComponents);

    if (writeFields_)
    {
        LSPGResidualOutput::writeMatrixResidual<Type>(outputName(block), eqn, R, mesh_, outputFolder_);
    }

    if (preconditioner_ == ResidualPreconditionerType::Jacobi && updatePreconditioner_)
    {
        residual_.setBlockRowWeights(block, jacobiRowWeights(eqn.A()().primitiveField(), nComponents));
    }
}

#endif
