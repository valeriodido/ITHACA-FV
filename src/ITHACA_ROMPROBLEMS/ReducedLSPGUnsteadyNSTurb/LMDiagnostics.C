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
Description
    Solver diagnostics of ReducedLSPGUnsteadyNSTurb, run when the Levenberg-Marquardt solve of a
    time step stops with one of the trigger statuses (ITHACAdict, lmDiagnostics).

    LM stops with XtolTooSmall when every trial step is rejected: the linear model built from the
    finite-difference Jacobian predicts a decrease that the residual does not deliver. The dump
    separates the possible causes:
      - ill-conditioning      : singular values of J (singularValues.dat)
      - wrong local model     : residual along the Gauss-Newton and steepest-descent directions,
                                actual against predicted reduction (direction_*.dat)
      - non-smooth residual   : kink indicator per block along every reduced coordinate
                                (coordinates*.dat), with the counts of k/omega cells crossing
                                the clipping bound and of faces where phi changes sign

    Kink indicator of a block b, coordinate j, step s:
        K = || (r(y + s e_j) - r(y)) + (r(y - s e_j) - r(y)) ||_b / || r(y + s e_j) - r(y - s e_j) ||_b
    For a smooth residual K = O(s) -> 0; for a kink K = O(1) as s -> 0; for a jump the denominator
    does not vanish with s.
\*---------------------------------------------------------------------------*/

#include "ReducedLSPGUnsteadyNSTurb.H"
#include <Eigen/SVD>
#include <fstream>
#include <iomanip>
#include <limits>

namespace
{
/// Residual vector and discrete state of one evaluation
struct Evaluation
{
    Eigen::VectorXd r;
    //- Clipped cells, one mask per transported turbulence field (k, omega)
    List<boolList> clipped;
    //- Sign of the face flux on the internal faces
    boolList phiPositive;
};

/// Number of entries that differ between two masks
label nSwitched(const boolList& a, const boolList& b)
{
    if (a.size() != b.size())
    {
        return 0;
    }
    label n = 0;
    forAll(a, i)
    {
        if (a[i] != b[i])
        {
            n++;
        }
    }
    return n;
}

label nTrue(const boolList& a)
{
    label n = 0;
    forAll(a, i)
    {
        if (a[i])
        {
            n++;
        }
    }
    return n;
}

/// Kink indicator ||(rPlus - r0) + (rMinus - r0)|| / ||rPlus - rMinus|| of two vector segments
double kinkIndicator(
    const Eigen::Ref<const Eigen::VectorXd>& rPlus,
    const Eigen::Ref<const Eigen::VectorXd>& rMinus,
    const Eigen::Ref<const Eigen::VectorXd>& r0)
{
    const double num = (rPlus + rMinus - 2.0 * r0).norm();
    const double den = (rPlus - rMinus).norm();
    if (den > 0)
    {
        return num / den;
    }
    return num > 0 ? std::numeric_limits<double>::infinity() : 0.0;
}
}


bool ReducedLSPGUnsteadyNSTurb::lmDiagnosticsTriggered(const std::string& status) const
{
    if (!lmDiagnostics_.enabled || nDiagnosticDumps_ >= lmDiagnostics_.maxDumps)
    {
        return false;
    }
    for (const word& trigger : lmDiagnostics_.triggers)
    {
        // lmStatusToString may append the tolerance name, e.g. "RelativeReductionTooSmall (ftol)"
        if (status.rfind(trigger, 0) == 0)
        {
            return true;
        }
    }
    return false;
}


void ReducedLSPGUnsteadyNSTurb::runLMDiagnostics(
    const LevenbergMarquardtFunctor& functor,
    const Eigen::VectorXd& y,
    const std::string& status,
    const word& folder)
{
    nDiagnosticDumps_++;
    const scalar time = runTime().value();
    const word dumpFolder(folder + "/lmDiagnostics/" + name(time) + "/");
    if (Pstream::master())
    {
        mkDir(dumpFolder);
    }
    Info << "### LM diagnostics at t = " << time << " (status " << status.c_str() << "), written to "
         << dumpFolder << endl;

    const label n = y.size();
    const label m = residualWorkspace_.size();
    const scalarList& steps = lmDiagnostics_.steps;
    const scalar smallestStep = min(steps);

    // Blocks of the tables, in this order; phi is reported as 0 without the flux residual
    const List<label> blocks({ iU_, iP_, iK_, iOmega_, iPhi_ });
    const wordList blockNames({ "U", "p", "k", "omega", "phi" });
    const label nBlocks = blocks.size();

    auto segment = [&](const Eigen::VectorXd& v, const label b) -> Eigen::Ref<const Eigen::VectorXd>
    {
        return v.segment(residualWorkspace_.blockOffset(blocks[b]), residualWorkspace_.blockSize(blocks[b]));
    };

    auto evaluate = [&](const Eigen::VectorXd& yy) -> Evaluation
    {
        Evaluation e;
        e.r.resize(m);
        functor(yy, e.r);
        e.clipped = turbulence->clippedCells();
        const scalarField& phiI = _phi().primitiveField();
        e.phiPositive.setSize(phiI.size());
        forAll(phiI, facei)
        {
            e.phiPositive[facei] = phiI[facei] > 0;
        }
        return e;
    };

    // Switches between two evaluations: k clipping, omega clipping, phi sign
    auto switches = [&](const Evaluation& a, const Evaluation& b) -> FixedList<label, 3>
    {
        FixedList<label, 3> s(0);
        if (a.clipped.size() >= 2 && b.clipped.size() >= 2)
        {
            s[0] = nSwitched(a.clipped[0], b.clipped[0]);
            s[1] = nSwitched(a.clipped[1], b.clipped[1]);
        }
        s[2] = nSwitched(a.phiPositive, b.phiPositive);
        return s;
    };

    // Labels of the reduced coordinates
    wordList coordinateLabels(n);
    {
        // Same order as stateMap_: the shared coefficients of the stacked POD first, then the other fields
        const wordList fieldNames({ "U", "p", "k", "omega" });
        const labelList fieldModes({ numberOfModes_.velocity, numberOfModes_.pressure, numberOfModes_.k, numberOfModes_.omega });
        label j = 0;
        if (stateMap_.size())
        {
            for (label i = 0; i < numberOfModes_.velocity; i++) coordinateLabels[j++] = "joint" + name(i + 1);
        }
        forAll(fieldNames, f)
        {
            if (jointField_[f]) continue;
            for (label i = 0; i < fieldModes[f]; i++) coordinateLabels[j++] = fieldNames[f] + name(i + 1);
        }
    }

    std::ostringstream summary;
    summary << std::scientific << std::setprecision(3);

    // ------------------------------------------------------------------ 1. Point where LM stopped
    const Evaluation base = evaluate(y);
    const double r0 = base.r.norm();
    const scalarField& phiBase = _phi().primitiveField();
    const scalar phiMax = max(mag(phiBase));
    label nNearZeroPhi = 0;
    forAll(phiBase, facei)
    {
        if (mag(phiBase[facei]) < 1e-6 * phiMax)
        {
            nNearZeroPhi++;
        }
    }
    const label nClippedK = base.clipped.size() >= 2 ? nTrue(base.clipped[0]) : -1;
    const label nClippedOmega = base.clipped.size() >= 2 ? nTrue(base.clipped[1]) : -1;

    summary << "=== LM diagnostics at t = " << time << ", status " << status << "\n\n"
            << "Reduced coordinates: " << n << ", residual rows: " << m << ", ||r(y)|| = " << r0 << "\n"
            << "||y||_inf = " << y.cwiseAbs().maxCoeff() << " (scaled coordinates, O(1) in the training range)\n"
            << "Clipped cells at y: k " << nClippedK << ", omega " << nClippedOmega
            << " | internal faces with |phi| < 1e-6 max|phi|: " << nNearZeroPhi << "\n";
    summary << "Blocks at y:";
    for (label b = 0; b < nBlocks; b++)
    {
        summary << " " << blockNames[b] << " " << (blocks[b] >= 0 ? segment(base.r, b).norm() : 0.0);
    }
    summary << "\n\n";

    // ------------------------------------------------------------------ 2. Jacobian conditioning
    Eigen::MatrixXd J(m, n);
    functor.df(y, J);
    Eigen::BDCSVD<Eigen::MatrixXd> svd(J, Eigen::ComputeThinV);
    const Eigen::VectorXd sv = svd.singularValues();
    const Eigen::MatrixXd& V = svd.matrixV();
    const double svMax = sv(0);
    const double svMin = sv(n - 1);
    const double cond = svMin > 0 ? svMax / svMin : std::numeric_limits<double>::infinity();

    if (Pstream::master())
    {
        std::ofstream out(dumpFolder + "singularValues.dat");
        out << std::scientific << std::setprecision(8)
            << "# index sigma sigma/sigma_max | dominant coordinate of the right singular vector, its weight\n";
        for (label i = 0; i < n; i++)
        {
            Eigen::Index jMax;
            const double w = V.col(i).cwiseAbs().maxCoeff(&jMax);
            out << i + 1 << " " << sv(i) << " " << sv(i) / svMax << " " << coordinateLabels[jMax] << " " << w << "\n";
        }
    }

    label nSmallSv = 0;
    for (label i = 0; i < n; i++)
    {
        if (sv(i) < 1e-6 * svMax)
        {
            nSmallSv++;
        }
    }
    summary << "Jacobian (scaled coordinates): sigma_max " << svMax << ", sigma_min " << svMin
            << ", condition number " << cond << ", singular values below 1e-6 sigma_max: " << nSmallSv << "\n"
            << "Weakest directions (dominant coordinates):";
    for (label i = n - 1; i >= max(label(0), n - 3); i--)
    {
        Eigen::Index jMax;
        V.col(i).cwiseAbs().maxCoeff(&jMax);
        summary << " " << coordinateLabels[jMax] << " (sigma/sigma_max " << sv(i) / svMax << ")";
    }
    summary << "\n\n";

    // ------------------------------------------------------------------ 3. Direction scans
    const Eigen::VectorXd dGN = J.colPivHouseholderQr().solve(-base.r);
    const Eigen::VectorXd dSD = -(J.transpose() * base.r);
    const List<std::pair<word, Eigen::VectorXd>> directions({ { "gaussNewton", dGN }, { "steepestDescent", dSD } });
    // Smallest step with an actual decrease, per direction (-1 if none)
    List<scalar> firstDecrease(directions.size(), -1);

    forAll(directions, dirI)
    {
        const word& dirName = directions[dirI].first;
        const double dNorm = directions[dirI].second.norm();
        std::ofstream out;
        if (Pstream::master())
        {
            out.open(dumpFolder + "direction_" + dirName + ".dat");
            out << std::scientific << std::setprecision(8)
                << "# unit direction, ||d|| = " << dNorm << "\n"
                << "# s total U p k omega phi predictedTotal actualReduction predictedReduction "
                << "switchesK switchesOmega switchesPhi\n";
        }
        summary << "Direction " << dirName << " (||d|| = " << dNorm << "):\n";
        if (!(dNorm > 0))
        {
            summary << "  zero direction, skipped\n";
            continue;
        }
        const Eigen::VectorXd dHat = directions[dirI].second / dNorm;
        const Eigen::VectorXd JdHat = J * dHat;
        scalar smallestDecrease = -1;
        double ratioAtSmallest = 0;

        // Largest step first, so that the table reads as s -> 0
        scalarList sortedSteps(steps);
        Foam::sort(sortedSteps, std::greater<scalar>());
        for (const scalar s : sortedSteps)
        {
            const Evaluation e = evaluate(y + s * dHat);
            const double total = e.r.norm();
            const double predicted = (base.r + s * JdHat).norm();
            const double actualReduction = r0 - total;
            const double predictedReduction = r0 - predicted;
            const FixedList<label, 3> sw = switches(base, e);
            if (actualReduction > 0)
            {
                smallestDecrease = s;
            }
            if (s == smallestStep)
            {
                ratioAtSmallest = predictedReduction != 0 ? actualReduction / predictedReduction : 0;
            }
            if (Pstream::master())
            {
                out << s << " " << total;
                for (label b = 0; b < nBlocks; b++)
                {
                    out << " " << (blocks[b] >= 0 ? segment(e.r, b).norm() : 0.0);
                }
                out << " " << predicted << " " << actualReduction << " " << predictedReduction
                    << " " << sw[0] << " " << sw[1] << " " << sw[2] << "\n";
            }
        }
        firstDecrease[dirI] = smallestDecrease;
        summary << "  smallest step with an actual decrease: "
                << (smallestDecrease > 0 ? std::to_string(smallestDecrease) : std::string("none")) << "\n"
                << "  actual / predicted reduction at s = " << smallestStep << ": " << ratioAtSmallest
                << " (-> 1 for a smooth residual and a correct Jacobian)\n";
    }
    summary << "\n";

    // ------------------------------------------------------------------ 4. Coordinate scans
    // Per step: max kink indicator over the coordinates (total and per block), max switches
    Eigen::MatrixXd maxKink = Eigen::MatrixXd::Zero(steps.size(), nBlocks + 1);
    Eigen::MatrixXi maxSwitches = Eigen::MatrixXi::Zero(steps.size(), 3);
    // Per coordinate, at the smallest step: kink indicators and column norm
    Eigen::MatrixXd smallStepKink = Eigen::MatrixXd::Zero(n, nBlocks + 2);

    if (lmDiagnostics_.coordinateScan)
    {
        std::ofstream out;
        if (Pstream::master())
        {
            out.open(dumpFolder + "coordinates.dat");
            out << std::scientific << std::setprecision(6)
                << "# j label s Ktotal KU Kp Kk Komega Kphi columnNorm switchesK switchesOmega switchesPhi\n";
        }
        for (label j = 0; j < n; j++)
        {
            forAll(steps, sI)
            {
                const scalar s = steps[sI];
                Eigen::VectorXd yPert = y;
                yPert(j) = y(j) + s;
                const Evaluation ePlus = evaluate(yPert);
                yPert(j) = y(j) - s;
                const Evaluation eMinus = evaluate(yPert);

                Eigen::VectorXd K(nBlocks + 1);
                K(0) = kinkIndicator(ePlus.r, eMinus.r, base.r);
                for (label b = 0; b < nBlocks; b++)
                {
                    K(b + 1) = blocks[b] >= 0 ? kinkIndicator(segment(ePlus.r, b), segment(eMinus.r, b), segment(base.r, b)) : 0.0;
                }
                const double columnNorm = (ePlus.r - eMinus.r).norm() / (2 * s);
                const FixedList<label, 3> swPlus = switches(base, ePlus);
                const FixedList<label, 3> swMinus = switches(base, eMinus);
                FixedList<label, 3> sw;
                for (label c = 0; c < 3; c++)
                {
                    sw[c] = max(swPlus[c], swMinus[c]);
                    maxSwitches(sI, c) = max(maxSwitches(sI, c), sw[c]);
                }
                for (label c = 0; c <= nBlocks; c++)
                {
                    maxKink(sI, c) = std::max(maxKink(sI, c), K(c));
                }
                if (s == smallestStep)
                {
                    smallStepKink.row(j).head(nBlocks + 1) = K.transpose();
                    smallStepKink(j, nBlocks + 1) = columnNorm;
                }
                if (Pstream::master())
                {
                    out << j + 1 << " " << coordinateLabels[j] << " " << s;
                    for (label c = 0; c <= nBlocks; c++)
                    {
                        out << " " << K(c);
                    }
                    out << " " << columnNorm << " " << sw[0] << " " << sw[1] << " " << sw[2] << "\n";
                }
            }
        }

        if (Pstream::master())
        {
            std::ofstream outMax(dumpFolder + "coordinates_max.dat");
            outMax << std::scientific << std::setprecision(6)
                   << "# max over the coordinates: s Ktotal KU Kp Kk Komega Kphi switchesK switchesOmega switchesPhi\n";
            forAll(steps, sI)
            {
                outMax << steps[sI];
                for (label c = 0; c <= nBlocks; c++)
                {
                    outMax << " " << maxKink(sI, c);
                }
                outMax << " " << maxSwitches(sI, 0) << " " << maxSwitches(sI, 1) << " " << maxSwitches(sI, 2) << "\n";
            }

            std::ofstream outSmall(dumpFolder + "coordinates_smallStep.dat");
            outSmall << std::scientific << std::setprecision(6)
                     << "# s = " << smallestStep << ": j label Ktotal KU Kp Kk Komega Kphi columnNorm\n";
            for (label j = 0; j < n; j++)
            {
                outSmall << j + 1 << " " << coordinateLabels[j];
                for (label c = 0; c <= nBlocks + 1; c++)
                {
                    outSmall << " " << smallStepKink(j, c);
                }
                outSmall << "\n";
            }
        }

        // Summary of the coordinate scan
        label smallI = 0;
        forAll(steps, sI)
        {
            if (steps[sI] == smallestStep)
            {
                smallI = sI;
            }
        }
        summary << "Coordinate scan at s = " << smallestStep << " (max over coordinates):\n  kink indicator total "
                << maxKink(smallI, 0);
        for (label b = 0; b < nBlocks; b++)
        {
            summary << ", " << blockNames[b] << " " << maxKink(smallI, b + 1);
        }
        summary << "\n  switches: k clipping " << maxSwitches(smallI, 0) << ", omega clipping " << maxSwitches(smallI, 1)
                << ", phi sign " << maxSwitches(smallI, 2) << "\n  non-smooth coordinates (Ktotal > 0.1):";
        label nKinked = 0;
        for (label j = 0; j < n; j++)
        {
            if (smallStepKink(j, 0) > 0.1)
            {
                Eigen::Index bMax;
                smallStepKink.row(j).segment(1, nBlocks).maxCoeff(&bMax);
                summary << " " << coordinateLabels[j] << "(" << blockNames[bMax] << ")";
                nKinked++;
            }
        }
        summary << (nKinked ? "" : " none") << "\n\n";
    }

    // ------------------------------------------------------------------ 5. Interpretation
    summary << "Interpretation (heuristic)\n";
    if (cond > 1e8)
    {
        summary << "- J is ill-conditioned (cond " << cond << "): some coordinates barely change the residual,\n"
                << "  and the Gauss-Newton step along them is unreliable (see the weakest directions above).\n";
    }
    const bool sdDecrease = firstDecrease[1] > 0;
    const bool gnDecrease = firstDecrease[0] > 0;
    if (!sdDecrease)
    {
        summary << "- No decrease even along the steepest-descent direction -J^T r: the Jacobian does not\n"
                << "  describe the residual at y (non-smooth residual or wrong finite differences).\n";
    } else if (!gnDecrease)
    {
        summary << "- A decrease exists along -J^T r but not along the Gauss-Newton step: the step relies on\n"
                << "  poorly determined directions (conditioning).\n";
    } else
    {
        summary << "- Both directions decrease the residual for small enough steps: LM stopped on its own\n"
                << "  tolerances/trust region rather than on a broken local model.\n";
    }
    if (lmDiagnostics_.coordinateScan)
    {
        label smallI = 0;
        forAll(steps, sI)
        {
            if (steps[sI] == smallestStep)
            {
                smallI = sI;
            }
        }
        if (maxKink(smallI, 0) > 0.1)
        {
            summary << "- The residual is non-smooth at y (kink indicator " << maxKink(smallI, 0) << " at s = " << smallestStep << ").\n";
            if (maxSwitches(smallI, 0) + maxSwitches(smallI, 1) > 0)
            {
                summary << "  k/omega cells cross the clipping bound within the smallest step: clipping and sqrt(k).\n";
            }
            if (maxSwitches(smallI, 2) > 0)
            {
                summary << "  Faces change the sign of phi within the smallest step: upwind switching of the convection schemes.\n";
            }
            if (maxSwitches(smallI, 0) + maxSwitches(smallI, 1) + maxSwitches(smallI, 2) == 0)
            {
                summary << "  No clipping or upwind switch within the smallest step: limiters (SST min/max, TVD) or wall functions.\n";
            }
        } else
        {
            summary << "- The residual is smooth down to s = " << smallestStep << " along every coordinate.\n";
        }
    }

    if (Pstream::master())
    {
        std::ofstream out(dumpFolder + "summary.log");
        out << summary.str();

        // One line per dump, to compare the failures of a run
        const word indexFile(folder + "/lmDiagnostics/index.dat");
        const bool newIndex = !isFile(indexFile);
        std::ofstream index(indexFile, std::ios::app);
        if (newIndex)
        {
            index << "# time status residual conditionNumber clippedK clippedOmega firstDecreaseGN firstDecreaseSD\n";
        }
        index << std::scientific << std::setprecision(6) << time << " " << word(status).replaceAll(" ", "_") << " " << r0 << " " << cond
              << " " << nClippedK << " " << nClippedOmega << " " << firstDecrease[0] << " " << firstDecrease[1] << "\n";
    }
    Info << summary.str().c_str() << endl;
}
