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
    NLSSettings
Description
    Settings and results of the modular nonlinear least-squares solvers.
SourceFiles
    NLSSettings.C
\*---------------------------------------------------------------------------*/

#include "NLSSettings.H"
#include "ITHACAassert.H"

namespace ITHACAnls
{

std::string toString(const NLSStatus status)
{
    switch (status)
    {
        case NLSStatus::MaxIterations:    return "Running";
        case NLSStatus::GradientSmall:    return "CosinusTooSmall (gtol)";
        case NLSStatus::ReductionSmall:   return "RelativeReductionTooSmall (ftol)";
        case NLSStatus::StepSmall:        return "RelativeErrorTooSmall (xtol)";
        case NLSStatus::DampingExploded:  return "XtolTooSmall";
        case NLSStatus::LineSearchFailed: return "LineSearchFailed";
        case NLSStatus::LinearSolveFailed: return "LinearSolveFailed";
    }
    return "Unknown";
}

namespace
{
/// Reads the word `key` and checks it against the choices
word readChoice(
    const dictionary& dict,
    const word& key,
    const word& defaultValue,
    const wordList& choices)
{
    const word value = dict.getOrDefault<word>(key, defaultValue);
    if (!choices.found(value))
    {
        FatalIOErrorInFunction(dict)
            << "nonlinearSolver: " << key << " '" << value << "' is not available. Valid choices: "
            << choices << exit(FatalIOError);
    }
    return value;
}
}

NLSSettings NLSSettings::read(const dictionary& dict)
{
    NLSSettings s;

    s.type = readChoice(dict, "type", s.type, wordList({ "levenbergMarquardt", "gaussNewton" }));
    s.linearAlgebra = readChoice(dict, "linearAlgebra", s.linearAlgebra, wordList({ "normalEquations", "qr" }));
    s.jacobian = readChoice(dict, "jacobian", s.jacobian, wordList({ "finiteDifference" }));
    s.jacobianReuse = readChoice(dict, "jacobianReuse", s.jacobianReuse, wordList({ "none" }));
    s.lineSearch = readChoice(dict, "lineSearch", s.lineSearch, wordList({ "fixed", "backtracking" }));
    s.damping = readChoice(dict, "damping", s.damping, wordList({ "nielsen" }));

    // Options of the other solver type are ignored: tell the user
    const bool gn = s.type == "gaussNewton";
    for (const word& key : wordList({ "lineSearch", "stepLength", "armijoC1", "backtrackShrink", "maxBacktracks" }))
    {
        if (!gn && dict.found(key))
        {
            WarningInFunction << "nonlinearSolver: '" << key << "' is only used by gaussNewton, ignored" << endl;
        }
    }
    for (const word& key : wordList({ "damping", "lambda0", "lambdaMax", "acceptRatio" }))
    {
        if (gn && dict.found(key))
        {
            WarningInFunction << "nonlinearSolver: '" << key << "' is only used by levenbergMarquardt, ignored" << endl;
        }
    }

    s.maxIterations = dict.getOrDefault<label>("maxIterations", s.maxIterations);
    s.gradientTol = dict.getOrDefault<scalar>("gradientTol", s.gradientTol);
    s.reductionTol = dict.getOrDefault<scalar>("reductionTol", s.reductionTol);
    s.stepTol = dict.getOrDefault<scalar>("stepTol", s.stepTol);

    s.jacobianStep = dict.getOrDefault<scalar>("jacobianStep", s.jacobianStep);
    s.jacobianOrder = dict.getOrDefault<label>("jacobianOrder", s.jacobianOrder);

    s.lambda0 = dict.getOrDefault<scalar>("lambda0", s.lambda0);
    s.lambdaMax = dict.getOrDefault<scalar>("lambdaMax", s.lambdaMax);
    s.acceptRatio = dict.getOrDefault<scalar>("acceptRatio", s.acceptRatio);

    s.stepLength = dict.getOrDefault<scalar>("stepLength", s.stepLength);
    s.armijoC1 = dict.getOrDefault<scalar>("armijoC1", s.armijoC1);
    s.backtrackShrink = dict.getOrDefault<scalar>("backtrackShrink", s.backtrackShrink);
    s.maxBacktracks = dict.getOrDefault<label>("maxBacktracks", s.maxBacktracks);

    M_Assert(s.maxIterations > 0, "nonlinearSolver: maxIterations must be positive.");
    M_Assert(s.gradientTol >= 0 && s.reductionTol >= 0 && s.stepTol >= 0,
        "nonlinearSolver: the tolerances must be non-negative.");
    M_Assert(s.jacobianStep > 0, "nonlinearSolver: jacobianStep must be positive.");
    M_Assert(s.jacobianOrder == 1 || s.jacobianOrder == 2, "nonlinearSolver: jacobianOrder must be 1 or 2.");
    M_Assert(s.lambda0 > 0 && s.lambdaMax > s.lambda0, "nonlinearSolver: need 0 < lambda0 < lambdaMax.");
    M_Assert(s.stepLength > 0, "nonlinearSolver: stepLength must be positive.");
    M_Assert(s.armijoC1 > 0 && s.armijoC1 < 1, "nonlinearSolver: armijoC1 must be in (0, 1).");
    M_Assert(s.backtrackShrink > 0 && s.backtrackShrink < 1, "nonlinearSolver: backtrackShrink must be in (0, 1).");
    M_Assert(s.maxBacktracks >= 0, "nonlinearSolver: maxBacktracks must be non-negative.");
    return s;
}

} // namespace ITHACAnls
