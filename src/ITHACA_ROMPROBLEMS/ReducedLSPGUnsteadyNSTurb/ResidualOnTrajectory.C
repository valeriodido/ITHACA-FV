// Consistency diagnostic of ReducedLSPGUnsteadyNSTurb (Carlberg et al. 2011): the ROM residual on a FOM trajectory

#include "ReducedLSPGUnsteadyNSTurb.H"
#include "TimeStateGuard.H"
#include <fstream>
#include <iomanip>

namespace
{
/// Cells whose value is the neighbour average Foam::bound assigns to negative cells after a FOM solve.
/// Foam::bound sets a negative cell to fvc::average(max(f, fMin)) evaluated with the cell itself at fMin
/// (kOmegaSST: kMin = omegaMin = SMALL): f_P = sum_f |Sf| (w_P fMin + (1 - w_P) f_N) / sum_f |Sf|,
/// with w_P the linear-interpolation weight of P. Internal faces only (cells next to a patch, processor patches
/// included, are skipped)
boolList boundedCells(const volScalarField& f)
{
    const fvMesh& m = f.mesh();
    const labelUList& own = m.owner();
    const labelUList& nei = m.neighbour();
    const scalarField& w = m.weights().primitiveField();
    const scalarField& magSf = m.magSf().primitiveField();
    scalarField sumA(m.nCells(), 0.0), sumNb(m.nCells(), 0.0), sumW(m.nCells(), 0.0);
    forAll(own, facei)
    {
        const label P = own[facei], N = nei[facei];
        sumA[P] += magSf[facei];
        sumA[N] += magSf[facei];
        sumNb[P] += magSf[facei] * (1 - w[facei]) * f[N];
        sumNb[N] += magSf[facei] * w[facei] * f[P];
        sumW[P] += magSf[facei] * w[facei];
        sumW[N] += magSf[facei] * (1 - w[facei]);
    }
    boolList onPatch(m.nCells(), false);
    forAll(m.boundary(), patchi)
    {
        for (const label celli : m.boundary()[patchi].faceCells())
        {
            onPatch[celli] = true;
        }
    }
    boolList mask(f.size(), false);
    forAll(mask, celli)
    {
        if (!onPatch[celli] && sumA[celli] > 0)
        {
            const scalar avg = (sumNb[celli] + sumW[celli] * SMALL) / sumA[celli];
            mask[celli] = mag(f[celli] - avg) <= 1e-9 * max(mag(avg), VSMALL);
        }
    }
    return mask;
}
}


Eigen::MatrixXd ReducedLSPGUnsteadyNSTurb::evaluateResidualOnTrajectory(
    const word& bcROM,
    const PtrList<volVectorField>& Ufom,
    const PtrList<surfaceScalarField>& phiFom,
    const PtrList<volScalarField>& pFom,
    const PtrList<volScalarField>& kFom,
    const PtrList<volScalarField>& omegaFom,
    const Eigen::VectorXd& times,
    const scalar deltaT,
    const word& mode,
    const fileName& pressureCheckFile,
    const PtrList<volScalarField>* nutFom)
{
    M_Assert(mode == "exact" || mode == "projected",
        "evaluateResidualOnTrajectory: mode must be 'exact' or 'projected'.");
    const label nSteps = Ufom.size();
    M_Assert(phiFom.size() == nSteps && pFom.size() == nSteps && kFom.size() == nSteps && omegaFom.size() == nSteps && times.size() == nSteps,
        "evaluateResidualOnTrajectory: all field lists and the time vector must have the same size.");
    M_Assert(nSteps >= 3, "evaluateResidualOnTrajectory: at least 3 consecutive states are needed.");
    M_Assert(turbulenceFields_.size() == 2, "evaluateResidualOnTrajectory: a k-omega closure is needed.");

    Time& runTime = this->runTime();
    const fvMesh& mesh = this->mesh();
    volVectorField& U = U_();
    volScalarField& p = p_();
    surfaceScalarField& phi = phi_();
    volScalarField& k = turbulenceFields_[0];
    volScalarField& omega = turbulenceFields_[1];

    // The time and the time step are restored on return
    const TimeStateGuard timeGuard(runTime);
    const label savedIndex = runTime.timeIndex();

    boundaryConditions_ = BoundaryConditions { bcROM, "linear" };

    // Two increments ending at times(0), so that deltaT0 = deltaT for the backward scheme
    runTime.setDeltaT(deltaT, false);
    runTime.setTime(times(0) - 2 * deltaT, savedIndex + 1);
    runTime++;
    runTime++;

    // Columns: time, total, then every reported block (U p k omega phi)
    const labelList& reported = reportedBlocks();
    Eigen::MatrixXd out(nSteps - 2, 2 + reported.size());

    // Pressure residual diagnostics, exact mode only. The files are written by the master only
    const bool pressureCheck = mode == "exact" && !pressureCheckFile.empty();
    std::ofstream pCheck;
    if (pressureCheck && Pstream::master())
    {
        pCheck.open(pressureCheckFile);
        pCheck << std::scientific << std::setprecision(8) << PressureResidualCheck::header(mesh)
               << " relNut relPhiOld relUOld relUOldOld" << std::endl;
    }

    // Every cell block of the weighted residual split by region, and its largest cell (location of the inconsistency)
    wordList regionNames;
    List<boolList> regionMasks;
    PressureResidualCheck::regions(mesh, regionNames, regionMasks);
    const wordList blockNames(fields_.names());
    labelList blockIds(fields_.size());
    forAll(blockIds, b)
    {
        blockIds[b] = residual_.index(blockNames[b]);
    }
    std::ofstream blockCheck;
    if (pCheck.is_open())
    {
        blockCheck.open(pressureCheckFile.path() / "blockRegions_disk.dat");
        blockCheck << "# weighted residual (as minimised by LM) per block and region; max = largest cell: value, x, y\n# time";
        for (const word& b : blockNames)
        {
            for (const word& r : regionNames)
            {
                blockCheck << " " << b << "_" << r;
            }
            blockCheck << " " << b << "_maxValue " << b << "_maxX " << b << "_maxY";
        }
        // Cells whose FOM value is the neighbour average Foam::bound assigns to negative cells after the solve
        blockCheck << " kBoundedCells omegaBoundedCells kMaxCellBounded";
        blockCheck << std::endl;
    }
    for (label n = 0; n < nSteps; n++)
    {
        if (n > 0)
        {
            runTime.setTime(times(n), runTime.timeIndex() + 1);
        }
        boundaryConditions_.updateTimeDependentBC(times(n));

        // The first modification at a new time index stores the previous state as oldTime,
        // so the old time levels are the previous (exact or projected) FOM states
        U == Ufom[n];
        p == pFom[n];
        phi == phiFom[n];
        k == kFom[n];
        omega == omegaFom[n];

        collector_.requestPreconditionerUpdate();
        if (mode == "projected")
        {
            initializePODCoeffsFromFields();
            assembleResidual(currentState_, true, false);
        } else
        {
            turbulence_->correctNut();
            residualWithCurrentFields();
        }

        // Evaluated on every processor (the norms are collective); only the master writes
        if (pressureCheck && n >= 2)
        {
            // nut recomputed by the ROM from the FOM (U, k, omega) against the FOM nut, and the old time
            // levels used by ddt and ddtCorr against the previous FOM states
            const scalarField& V = mesh.V().field();
            auto relV = [&](const auto& a, const auto& b) {
                return PressureResidualCheck::volumeNorm(a - b, V) / max(PressureResidualCheck::volumeNorm(b, V), VSMALL);
            };
            const scalar relNut = nutFom ? relV(nut_().primitiveField(), (*nutFom)[n].primitiveField()) : -1;
            const scalarField& phiOld = phi.oldTime().primitiveField();
            const scalarField& phiPrev = phiFom[n - 1].primitiveField();
            const scalar relPhiOld = Foam::sqrt(gSum(sqr(phiOld - phiPrev)) / max(gSum(sqr(phiPrev)), VSMALL));
            const scalar relUOld = relV(U.oldTime().primitiveField(), Ufom[n - 1].primitiveField());
            const scalar relUOldOld = relV(U.oldTime().oldTime().primitiveField(), Ufom[n - 2].primitiveField());

            const PressureResidualCheck::Terms terms = pressureResidualTerms();
            PressureResidualCheck::writeRow(pCheck, times(n), terms, mesh, U.primitiveField());
            pCheck << " " << relNut << " " << relPhiOld << " " << relUOld << " " << relUOldOld << std::endl;

            blockCheck << std::scientific << std::setprecision(8) << times(n);
            const vectorField& C = mesh.C().primitiveField();
            forAll(blockIds, b)
            {
                const Eigen::Ref<const Eigen::VectorXd> r = residual_.block(blockIds[b]);
                const label nPerCell = r.size() / mesh.nCells();
                scalarField cellSqr(mesh.nCells(), 0.0);
                forAll(cellSqr, celli)
                {
                    cellSqr[celli] = r.segment(nPerCell * celli, nPerCell).squaredNorm();
                }
                for (const boolList& mask : regionMasks)
                {
                    scalar s2 = 0;
                    forAll(cellSqr, celli)
                    {
                        s2 += mask[celli] ? cellSqr[celli] : 0;
                    }
                    blockCheck << " " << Foam::sqrt(returnReduce(s2, sumOp<scalar>()));
                }
                // Largest cell over all the processors, and its centre (sent by the processor that holds it)
                const label imax = findMax(cellSqr);
                const scalar localMax = imax >= 0 ? cellSqr[imax] : -GREAT;
                const scalar globalMax = returnReduce(localMax, maxOp<scalar>());
                const bool holdsMax = imax >= 0 && localMax == globalMax;
                blockCheck << " " << Foam::sqrt(globalMax)
                           << " " << returnReduce(holdsMax ? C[imax].x() : -GREAT, maxOp<scalar>())
                           << " " << returnReduce(holdsMax ? C[imax].y() : -GREAT, maxOp<scalar>());
            }

            const boolList kBounded = boundedCells(kFom[n]);
            const boolList omegaBounded = boundedCells(omegaFom[n]);
            // Whether the cell with the largest k residual (over all the processors) is a bounded cell
            label kMaxCell = 0;
            const scalar kLocalMax = residual_.block(iTurbulence_[0]).cwiseAbs().maxCoeff(&kMaxCell);
            const bool holdsKMax = kLocalMax == returnReduce(kLocalMax, maxOp<scalar>());
            const label kMaxCellBounded = returnReduce(label(holdsKMax && kBounded[kMaxCell]), maxOp<label>());
            const label nKBounded = returnReduce(label(std::count(kBounded.begin(), kBounded.end(), true)), sumOp<label>());
            const label nOmegaBounded =
                returnReduce(label(std::count(omegaBounded.begin(), omegaBounded.end(), true)), sumOp<label>());
            blockCheck << " " << nKBounded << " " << nOmegaBounded << " " << kMaxCellBounded;
            blockCheck << std::endl;
        }

        if (n >= 2)
        {
            out(n - 2, 0) = times(n);
            out(n - 2, 1) = residual_.norm();
            forAll(reported, i)
            {
                out(n - 2, 2 + i) = reportedBlockNorm(i);
            }
        }
    }

    return out;
}
