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
#include "PressureResidualCheck.H"
#include "ITHACAstream.H"
#include <memory>

#include "ITHACAutilities.H"
#include <Eigen/Dense>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <cmath>

class OfflineFOM : public LSPGUnsteadyNSTurb
{
public:
    explicit OfflineFOM(std::shared_ptr<ITHACAcontext> context): LSPGUnsteadyNSTurb(context) { }

    void offlineSolve(word offlinepath, word timeSettingsFile)
    {
        if ((offline) && (ITHACAutilities::check_folder(offlinepath) == true))
        {
            loadFieldsFromDisk(offlinepath);
            ITHACAstream::ReadDenseMatrixList(timeSnapshots, "./ITHACAoutput/Offline/", "timeSnapshots");
        } else
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

            volVectorField& U = _U();

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

    /// FOM trajectory written at every time step, needed by the consistency test.
    /// Reuses the data if the folder already exists.
    /// pimpleOverride: entries merged into the PIMPLE dictionary of system/fvSolution for this run only
    /// (e.g. more outer correctors and a residualControl, to converge every time step). The offline
    /// snapshots are not affected. The settings are stored in the folder and checked when it is reused.
    /// writePrecision: ASCII precision of the trajectory fields (the residual is sensitive to round-off in phi, U).
    /// While the trajectory is computed, the pressure residual of every converged state is evaluated in memory
    /// and written to <folder>/pressureCheck_inMemory.dat (PressureResidualCheck.H)
    void consistencySolve(const word& folder, scalar bcValue, scalar tStart, scalar tEnd, scalar dt,
        const dictionary& pimpleOverride, const label writePrecision)
    {
        OStringStream settings;
        settings << "bcValue " << bcValue << "; startTime " << tStart << "; endTime " << tEnd
                 << "; deltaT " << dt << "; writePrecision " << writePrecision
                 << "; pimpleOverride " << pimpleOverride;
        const fileName settingsFile(folder + "/V1settings");

        if (ITHACAutilities::check_folder(folder))
        {
            IFstream is(settingsFile);
            string stored;
            if (is.good())
            {
                std::ostringstream ss;
                ss << is.stdStream().rdbuf();
                stored = ss.str();
            }
            if (stored != settings.str())
            {
                FatalErrorInFunction
                    << "The FOM trajectory in " << folder << " was computed with different settings:" << nl
                    << "  stored : " << stored.c_str() << nl << "  current: " << settings.str().c_str() << nl
                    << "Delete the folder to recompute it." << exit(FatalError);
            }
            Info << "### V1 - Reusing the FOM trajectory in " << folder << endl;
            return;
        }

        // Temporary PIMPLE settings, restored after the run. pimpleControl re-reads the dictionary
        // at every outer loop, so the in-memory change is picked up without touching fvSolution
        dictionary& pimpleDict = const_cast<dictionary&>(mesh().solutionDict()).subDict("PIMPLE");
        const dictionary pimpleSaved(pimpleDict);
        pimpleDict.merge(pimpleOverride);
        Info << "### V1 - PIMPLE settings of the FOM trajectory:" << nl << pimpleDict << endl;

        restart();
        Vector<double> inl(0, 0, 0);
        inl[inletIndex(0, 1)] = bcValue;
        assignBC(_U(), inletIndex(0, 0), inl);
        startTime = tStart;
        finalTime = tEnd;
        timeStep = dt;
        writeEvery = dt;

        const unsigned savedPrecision = IOstream::defaultPrecision(writePrecision);
        mkDir(folder);
        if (Pstream::master())
        {
            pressureCheck_.open(folder + "/pressureCheck_inMemory.dat");
            pressureCheck_ << std::scientific << std::setprecision(8)
                           << PressureResidualCheck::header(mesh()) << std::endl;

            turbulenceCheck_.open(folder + "/turbulenceCheck_inMemory.dat");
            turbulenceCheck_ << std::scientific << std::setprecision(8)
                             << "# change of k and omega over one extra turbulence->correct() at the converged state\n"
                             << "# (zero at a fixed point of the PIMPLE loop), |dk|_V / |k|_V per region; max = largest |dk|/k: value, x, y\n"
                             << "# time";
            wordList names;
            List<boolList> masks;
            PressureResidualCheck::regions(mesh(), names, masks);
            for (const word& f : { "k", "omega" })
            {
                for (const word& r : names)
                {
                    turbulenceCheck_ << " d" << f << "_" << r;
                }
                turbulenceCheck_ << " d" << f << "_maxRel d" << f << "_maxX d" << f << "_maxY";
            }
            turbulenceCheck_ << std::endl;
        }
        pressureCheckEnabled_ = true;
        truthSolve(folder);
        pressureCheckEnabled_ = false;
        pressureCheck_.close();
        turbulenceCheck_.close();
        IOstream::defaultPrecision(savedPrecision);
        restart();

        pimpleDict = pimpleSaved;

        if (Pstream::master())
        {
            OFstream os(settingsFile);
            os << settings.str().c_str();
        }
    }

protected:
    /// Pressure residual of the converged FOM state, assembled in memory exactly as in the ROM
    /// (ReducedLSPGUnsteadyNSTurb/UEqn.H + pEqn.H): no disk round-off, no reconstruction of old time levels
    void timeStepCompleted() override
    {
        if (!pressureCheckEnabled_)
        {
            return;
        }
        volVectorField& U = _U();
        volScalarField& p = _p();
        surfaceScalarField& phi = _phi();
        IOMRFZoneList& MRF = this->MRF();
        fv::options& fvOptions = this->fvOptions();

        // As the ROM: final iteration, so no relaxation
        fvVectorMatrix UEqn(
            fvm::ddt(U) + fvm::div(phi, U) + MRF.DDt(U) + turbulence->divDevReff(U) == fvOptions(U));
        fvOptions.constrain(UEqn);

        // constrainPressure only acts on fixedFluxPressure patches, as in the FOM pEqn.H
        const PressureResidualCheck::Terms terms =
            PressureResidualCheck::compute(UEqn, U, p, phi, MRF, pressureRefCell(), pressureRefValue());
        if (pressureCheck_.is_open())
        {
            PressureResidualCheck::writeRow(pressureCheck_, runTime().value(), terms, mesh(), U.primitiveField());
            pressureCheck_ << std::endl;
        }

        // Fixed-point test of the turbulence equations: one more correct() must not change k and omega.
        // The state is restored afterwards, so the trajectory is unaffected
        volScalarField& kT = const_cast<volScalarField&>(turbulence->k().cref());
        volScalarField& omegaT = const_cast<volScalarField&>(turbulence->omega().cref());
        volScalarField& nutT = const_cast<volScalarField&>(turbulence->nut().cref());
        const volScalarField kSaved("kSaved", kT);
        const volScalarField omegaSaved("omegaSaved", omegaT);
        const volScalarField nutSaved("nutSaved", nutT);

        turbulence->correct();
        const scalarField dk(kT.primitiveField() - kSaved.primitiveField());
        const scalarField domega(omegaT.primitiveField() - omegaSaved.primitiveField());

        kT == kSaved;
        omegaT == omegaSaved;
        nutT == nutSaved;

        if (turbulenceCheck_.is_open())
        {
            const scalarField& V = mesh().V().field();
            const vectorField& C = mesh().C().primitiveField();
            wordList names;
            List<boolList> masks;
            PressureResidualCheck::regions(mesh(), names, masks);
            turbulenceCheck_ << runTime().value();
            auto write = [&](const scalarField& d, const scalarField& f)
            {
                for (const boolList& mask : masks)
                {
                    scalar sd = 0, sf = 0;
                    forAll(d, celli)
                    {
                        if (mask[celli])
                        {
                            sd += V[celli] * sqr(d[celli]);
                            sf += V[celli] * sqr(f[celli]);
                        }
                    }
                    turbulenceCheck_ << " " << Foam::sqrt(sd / max(sf, VSMALL));
                }
                const scalarField rel(mag(d) / max(mag(f), VSMALL));
                const label imax = findMax(rel);
                turbulenceCheck_ << " " << rel[imax] << " " << C[imax].x() << " " << C[imax].y();
            };
            write(dk, kSaved.primitiveField());
            write(domega, omegaSaved.primitiveField());
            turbulenceCheck_ << std::endl;
        }
    }

private:
    bool pressureCheckEnabled_ = false;
    std::ofstream pressureCheck_;
    std::ofstream turbulenceCheck_;
};

// * * * * * * * * * * * * * * * Output layout * * * * * * * * * * * * * * * //

/// Inputs and results of one ROM (caseID = romID in ITHACAdict) for one parametric run (suffix).
///   Inputs, shared by all the ROMs:
///     input_parameters/ROM/ROM_BC_<suffix>          boundary condition of the run
///     ITHACAoutput/OnlineFOM<suffix>/                FOM reference fields
///     ITHACAoutput/OnlineFOM<suffix>/forceCoeffs.dat FOM force coefficients (written by the FOM run, for the plots)
///   Results:
///     ITHACAoutput/ROM/<caseID>/<suffix>/            ROM fields (time folders), residualHistory.dat, lmDiagnostics/
///     ITHACAoutput/ROM/<caseID>/<suffix>/forceCoeffs.dat
///     ITHACAoutput/ROM/<caseID>/<suffix>/errors/     L2 errors against the FOM and projection errors (*_mat.txt)
///     ITHACAoutput/ROM/<caseID>/V1_<fomLabel>/       consistency test (one per ROM, on the run of consistencyTest/suffix)
struct RunPaths
{
    word caseID;
    word suffix;

    word bcFile() const { return "./input_parameters/ROM/ROM_BC_" + suffix; }
    word fomFolder() const { return "./ITHACAoutput/OnlineFOM" + suffix + "/"; }
    word caseFolder() const { return "./ITHACAoutput/ROM/" + caseID + "/"; }
    word romFolder() const { return caseFolder() + suffix + "/"; }
    word errorFolder() const { return romFolder() + "errors/"; }
    word forceFile() const { return romFolder() + "forceCoeffs.dat"; }
    word v1Folder(const word& fomLabel) const { return caseFolder() + "V1_" + fomLabel + "/"; }
};

// * * * * * * * * * * * * V1 consistency test helpers * * * * * * * * * * * //

/// Reads the first nCols numeric columns of a whitespace-separated table, skipping '#' lines
Eigen::MatrixXd readTable(const std::string& file, const int nCols)
{
    std::ifstream in(file);
    std::vector<std::vector<double>> rows;
    std::string line;
    while (std::getline(in, line))
    {
        if (line.empty() || line[0] == '#')
        {
            continue;
        }
        std::istringstream ls(line);
        std::vector<double> row(nCols);
        bool complete = true;
        for (int c = 0; c < nCols && complete; c++)
        {
            complete = static_cast<bool>(ls >> row[c]);
        }
        if (complete)
        {
            rows.push_back(row);
        }
    }
    Eigen::MatrixXd table(rows.size(), nCols);
    for (size_t r = 0; r < rows.size(); r++)
    {
        for (int c = 0; c < nCols; c++)
        {
            table(r, c) = rows[r][c];
        }
    }
    return table;
}

/// Reads the value of a "# key value" header line, or returns -1 if absent
double readHeaderValue(const std::string& file, const std::string& key)
{
    std::ifstream in(file);
    std::string line;
    while (std::getline(in, line))
    {
        std::istringstream ls(line);
        std::string hash, name;
        double value;
        if (ls >> hash >> name >> value && hash == "#" && name == key)
        {
            return value;
        }
    }
    return -1;
}

/// Mean (row 0) and max (row 1) of the columns 1..end, over the rows with tFrom <= time (column 0) <= tTo
Eigen::MatrixXd columnStats(const Eigen::MatrixXd& table, const scalar tFrom, const scalar tTo = GREAT)
{
    Eigen::MatrixXd stats = Eigen::MatrixXd::Zero(2, table.cols() - 1);
    label count = 0;
    for (label r = 0; r < table.rows(); r++)
    {
        if (table(r, 0) < tFrom || table(r, 0) > tTo)
        {
            continue;
        }
        for (label c = 1; c < table.cols(); c++)
        {
            stats(0, c - 1) += table(r, c);
            stats(1, c - 1) = std::max(stats(1, c - 1), table(r, c));
        }
        count++;
    }
    if (count > 0)
    {
        stats.row(0) /= count;
    }
    return stats;
}

void writeTable(const std::string& file, const Eigen::MatrixXd& table, const std::string& header)
{
    std::ofstream out(file);
    out << std::scientific << std::setprecision(8) << header << "\n";
    for (label r = 0; r < table.rows(); r++)
    {
        for (label c = 0; c < table.cols(); c++)
        {
            out << table(r, c) << (c + 1 < table.cols() ? " " : "\n");
        }
    }
}

/// Removes the first entry of a loaded field list when it holds one more state than expected
template <class FieldType>
void dropLeadingState(PtrList<FieldType>& fields, const label expected)
{
    if (fields.size() != expected + 1)
    {
        return;
    }
    PtrList<FieldType> trimmed(expected);
    for (label i = 0; i < expected; i++)
    {
        trimmed.set(i, fields.release(i + 1));
    }
    fields.transfer(trimmed);
}

/// V1 test: evaluates the ROM residual on a FOM trajectory sampled at every time step.
/// If the FOM states are far from roots of the ROM residual, no basis can make the ROM reproduce the FOM
/// (Carlberg et al. 2011, Prop. 1), and the accuracy is limited by the residual definition.
/// The FOM trajectory is cached in ITHACAoutput/V1/<fomLabel> (shared by all the ROMs), the results go to
/// ITHACAoutput/ROM/<caseID>/V1_<fomLabel>, and the ROM run compared with it is the parametric run <suffix>
/// (consistencyTest/suffix in ITHACAdict, default: the first run)
void runConsistencyTest(
    OfflineFOM& fom,
    ReducedLSPGUnsteadyNSTurb& rom,
    const std::shared_ptr<ITHACAcontext>& context,
    const word& defaultSuffix)
{
    const dictionary& dict = fom.ITHACAdict->subOrEmptyDict("consistencyTest");
    if (!dict.getOrDefault<bool>("enabled", false))
    {
        return;
    }

    // Read as a string: an unquoted suffix such as 01 would be parsed as a number
    const RunPaths run { rom.getID(), word(dict.getOrDefault<string>("suffix", defaultSuffix)) };
    const scalar bcValue = dict.get<scalar>("bcValue");
    const word bcROM(run.bcFile());
    const scalar tStart = dict.getOrDefault<scalar>("startTime", 0.0);
    const scalar tEnd = dict.get<scalar>("endTime");
    const scalar dt = dict.get<scalar>("deltaT");
    const scalar statsFrom = dict.getOrDefault<scalar>("statsFrom", tStart + 0.1 * (tEnd - tStart));

    // FOM trajectory label: one folder per PIMPLE setting, e.g. FOM (fvSolution) and FOMconverged
    const dictionary& pimpleOverride = dict.subOrEmptyDict("pimpleOverride");
    const word fomLabel = dict.getOrDefault<word>("fomLabel", pimpleOverride.empty() ? "FOM" : "FOMconverged");

    const word fomFolder("./ITHACAoutput/V1/" + fomLabel + "/");
    const word resultsFolder(run.v1Folder(fomLabel));
    mkDir(resultsFolder);

    Info << nl << "### V1 - Consistency test: ROM residual evaluated on FOM states (" << fomLabel << ")" << endl;
    const label writePrecision = dict.getOrDefault<label>("writePrecision", 15);
    fom.consistencySolve(fomFolder, bcValue, tStart, tEnd, dt, pimpleOverride, writePrecision);

    PtrList<volVectorField> Ufom = context->loadFieldsFromCaseFolder<volVectorField>("U", fomFolder);
    PtrList<volScalarField> pFom = context->loadFieldsFromCaseFolder<volScalarField>("p", fomFolder);
    PtrList<surfaceScalarField> phiFom = context->loadFieldsFromCaseFolder<surfaceScalarField>("phi", fomFolder);
    PtrList<volScalarField> kFom = context->loadFieldsFromCaseFolder<volScalarField>("k", fomFolder);
    PtrList<volScalarField> omegaFom = context->loadFieldsFromCaseFolder<volScalarField>("omega", fomFolder);
    PtrList<volScalarField> nutFom = context->loadFieldsFromCaseFolder<volScalarField>("nut", fomFolder);

    // ITHACA links the case's 0/ directory into its output folders, and the loader reads it as
    // a first time step: drop it, since the trajectory states are only the numbered exports
    const label expected = std::round((tEnd - tStart) / dt);
    if (isDir(fomFolder + "0"))
    {
        dropLeadingState(Ufom, expected);
        dropLeadingState(pFom, expected);
        dropLeadingState(phiFom, expected);
        dropLeadingState(kFom, expected);
        dropLeadingState(omegaFom, expected);
        dropLeadingState(nutFom, expected);
    }

    const label nStates = Ufom.size();
    if (nStates != expected || pFom.size() != expected || phiFom.size() != expected
        || kFom.size() != expected || omegaFom.size() != expected)
    {
        FatalErrorInFunction
            << "Found " << nStates << " FOM states in " << fomFolder << ", expected " << expected
            << ". Delete the folder to recompute the FOM trajectory with the current settings." << exit(FatalError);
    }

    // truthSolve writes the first state one time step after the start
    Eigen::VectorXd times(nStates);
    for (label i = 0; i < nStates; i++)
    {
        times(i) = tStart + (i + 1) * dt;
    }

    const std::vector<std::string> modes = { "exact", "projected" };
    std::vector<Eigen::MatrixXd> results;
    for (const std::string& mode : modes)
    {
        Info << "### V1 - Evaluating the ROM residual, mode " << mode << endl;
        results.push_back(rom.evaluateResidualOnTrajectory(
            bcROM, Ufom, phiFom, pFom, kFom, omegaFom, times, dt, mode,
            resultsFolder + "pressureCheck_disk.dat", nutFom.size() == nStates ? &nutFom : nullptr));
        if (Pstream::master())
        {
            writeTable(resultsFolder + "residual_" + mode + ".dat", results.back(), "# time total U p k omega phi");
        }
    }

    if (!Pstream::master())
    {
        return;
    }

    // ROM residual history: time initialTotal finalTotal U p k omega phi
    const std::string romHistory = run.romFolder() + "residualHistory.dat";
    const Eigen::MatrixXd romTable = readTable(romHistory, 8);
    const bool haveROM = romTable.rows() > 0;
    Eigen::MatrixXd romFinal;
    if (haveROM)
    {
        romFinal.resize(romTable.rows(), 7);
        romFinal.col(0) = romTable.col(0);
        romFinal.rightCols(6) = romTable.rightCols(6);
    }
    const double romDeltaT = haveROM ? readHeaderValue(romHistory, "deltaT") : -1;

    std::ofstream summary(resultsFolder + "consistencySummary.log");
    summary << std::scientific << std::setprecision(4);
    summary << "=== V1 consistency test (Carlberg et al. 2011, Prop. 1) ===\n\n"
        << "FOM trajectory     : inlet velocity " << bcValue << ", t in [" << tStart << ", " << tEnd
        << "], deltaT_FOM = " << dt << ", " << nStates << " states\n"
        << "FOM trajectory     : " << fomLabel << ", PIMPLE override: "
        << (pimpleOverride.empty() ? std::string("none (system/fvSolution)") : std::string("see ") + fomFolder + "V1settings") << "\n"
        << "ROM run            : " << (haveROM ? romHistory : "not found, ROM comparisons skipped") << "\n";
    if (haveROM)
    {
        summary << "deltaT_ROM         : " << romDeltaT << "\n";
        if (std::abs(romDeltaT - dt) > 1e-12 * dt)
        {
            summary << "WARNING            : deltaT_ROM != deltaT_FOM. The residuals contain a ddt term ~ 1/deltaT,\n"
                << "                     so the ROM and FOM norms are not directly comparable. Run the ROM\n"
                << "                     with deltaT_FOM for a clean comparison.\n";
        }
    }
    summary << "Block weights      : (U p k omega phi) = " << rom.residualBlockWeights().transpose() << "\n"
        << "Statistics over t in [" << statsFrom << ", " << tEnd << "]\n\n";

    const std::vector<std::string> blockNames = { "total", "U", "p", "k", "omega", "phi" };
    summary << std::left << std::setw(20) << "case (mean / max)";
    for (const std::string& b : blockNames)
    {
        summary << std::setw(26) << b;
    }
    summary << "\n";

    auto printStats = [&](const std::string& caseName, const Eigen::MatrixXd& stats)
    {
        summary << std::setw(20) << caseName;
        for (label c = 0; c < stats.cols(); c++)
        {
            std::ostringstream cell;
            cell << std::scientific << std::setprecision(3) << stats(0, c) << " / " << stats(1, c);
            summary << std::setw(26) << cell.str();
        }
        summary << "\n";
    };

    Eigen::MatrixXd romStats;
    if (haveROM)
    {
        // The ROM may run longer than the FOM trajectory: compare over the same window
        romStats = columnStats(romFinal, statsFrom, tEnd + SMALL);
        printStats("ROM (converged)", romStats);
    }
    std::vector<Eigen::MatrixXd> stats;
    for (size_t m = 0; m < modes.size(); m++)
    {
        stats.push_back(columnStats(results[m], statsFrom));
        printStats("FOM " + modes[m], stats.back());
    }

    // Mean total residuals, used for the interpretation
    const double exact = stats[0](0, 0);
    const double projected = stats[1](0, 0);

    summary << "\nRatios of the mean total residual\n"
        << "  projected / exact  : " << projected / exact << "\n";
    if (haveROM)
    {
        summary << "  exact / ROM        : " << exact / romStats(0, 0) << "\n"
            << "  projected / ROM    : " << projected / romStats(0, 0) << "\n";
    }

    // Dominant block of the inconsistency
    label dominant = 1;
    for (label c = 2; c < stats[0].cols(); c++)
    {
        if (stats[0](0, c) > stats[0](0, dominant))
        {
            dominant = c;
        }
    }

    summary << "\nInterpretation (heuristic)\n";
    if (haveROM)
    {
        if (exact / romStats(0, 0) > 0.1)
        {
            summary << "- The exact FOM states leave a residual of " << 100 * exact / romStats(0, 0)
                << " % of the ROM converged residual: the FOM is not a near-root of the ROM residual.\n"
                << "  The residual definition, not the basis, limits the accuracy.\n";
        }
        else
        {
            summary << "- The FOM states are near-roots of the ROM residual: the residual definition is consistent,\n"
                << "  and the ROM error comes from basis truncation or from the nonlinear solver.\n";
        }
        // Margin: the ROM minimises along its own history, so small differences are not meaningful
        if (projected < 0.8 * romStats(0, 0))
        {
            summary << "- The projected FOM trajectory has a clearly lower residual (< 80 %) than the ROM solution:\n"
                << "  check the LM convergence (status, iterations in residualHistory.dat).\n";
        }
    }
    summary << "- Largest block of the exact-FOM residual: " << blockNames[dominant] << "\n";
    summary << "- projected/exact = " << projected / exact
        << ": the increase due to the basis truncation alone.\n";

    summary.close();
    Info << "### V1 - Summary written to " << resultsFolder << "consistencySummary.log" << nl
         << "### V1 - Plot with: gnuplot -e \"caseID='" << run.caseID << "'; run='" << run.suffix << "'; v1='"
         << fomLabel << "'\" consistencyPlots.gp" << endl;
}

// * * * * * * * * * * * * * * Errors against the FOM * * * * * * * * * * * * //

/// Whether the FOM reference of the run has been computed: at least one written time besides 0
bool haveFOMReference(const RunPaths& run)
{
    const fileNameList dirs = readDir(run.fomFolder(), fileName::DIRECTORY);
    for (const fileName& d : dirs)
    {
        scalar t;
        if (readScalar(d, t) && t > 0)
        {
            return true;
        }
    }
    return false;
}

/// Relative L2 error, at every written time, between the FOM field <fieldName> of the run and the ROM field
/// <fieldName>_r. Written to errors/L2ErrorLSPG<fieldName>_mat.txt
template <typename FieldType>
void compareWithFOM(const RunPaths& run, const word& fieldName, const std::shared_ptr<ITHACAcontext>& context)
{
    PtrList<FieldType> fom = context->loadFieldsFromCaseFolder<FieldType>(fieldName, run.fomFolder());
    PtrList<FieldType> rom = context->loadFieldsFromCaseFolder<FieldType>(fieldName + "_r", run.romFolder());

    Eigen::MatrixXd l2Error = ITHACAutilities::errorL2Rel(fom, rom);
    ITHACAstream::exportMatrix(l2Error, "L2ErrorLSPG" + fieldName, "eigen", run.errorFolder());
    Info << "-- " << fieldName << ": maximum L2 error " << l2Error.maxCoeff()
         << ", average L2 error " << l2Error.mean() << endl;
}


/// Pressure error with the volume-weighted mean removed from both fields. If it is much smaller than
/// the plain L2 error, the ROM pressure is mostly off by a uniform level (gauge). The difference of the
/// means is exported as well (errors/L2ErrorLSPGp_meanRemoved_mat.txt, errors/pMeanOffset_mat.txt)
void comparePressureWithoutMean(const RunPaths& run, const std::shared_ptr<ITHACAcontext>& context)
{
    PtrList<volScalarField> pFom = context->loadFieldsFromCaseFolder<volScalarField>("p", run.fomFolder());
    PtrList<volScalarField> pRom = context->loadFieldsFromCaseFolder<volScalarField>("p_r", run.romFolder());
    const label n = min(pFom.size(), pRom.size());

    PtrList<volScalarField> fomNoMean(n);
    PtrList<volScalarField> romNoMean(n);
    Eigen::MatrixXd meanOffset(n, 1);
    for (label i = 0; i < n; i++)
    {
        const scalar meanFom = pFom[i].weightedAverage(pFom[i].mesh().V()).value();
        const scalar meanRom = pRom[i].weightedAverage(pRom[i].mesh().V()).value();
        fomNoMean.set(i, new volScalarField("pNoMean", pFom[i] - dimensionedScalar(pFom[i].dimensions(), meanFom)));
        romNoMean.set(i, new volScalarField("pNoMean", pRom[i] - dimensionedScalar(pRom[i].dimensions(), meanRom)));
        meanOffset(i, 0) = meanRom - meanFom;
    }

    Eigen::MatrixXd l2Error = ITHACAutilities::errorL2Rel(fomNoMean, romNoMean);
    ITHACAstream::exportMatrix(l2Error, "L2ErrorLSPGp_meanRemoved", "eigen", run.errorFolder());
    ITHACAstream::exportMatrix(meanOffset, "pMeanOffset", "eigen", run.errorFolder());
    Info << "-- p with the mean removed: maximum L2 error " << l2Error.maxCoeff()
         << ", average L2 error " << l2Error.mean() << nl
         << "-- Maximum |mean(p_ROM) - mean(p_FOM)|: " << meanOffset.cwiseAbs().maxCoeff() << endl;
}


/// Weights of the offline runs in the affine reference of the ROM, at the parameter of the run (value of
/// ROM_BC_<suffix> at the first time)
Eigen::VectorXd referenceWeights(const ReducedLSPGUnsteadyNSTurb& rom, const RunPaths& run)
{
    const Eigen::MatrixXd bc = ITHACAstream::readMatrix(run.bcFile());
    const Eigen::VectorXd muRun = bc.row(0).tail(bc.cols() - 1).transpose();
    return rom.interpolateIDW(muRun);
}


/// Projection error of the FOM fields of the run: best approximation in the affine space
/// sum_r w_r q0_r + span(modes) (the reference blended as in the ROM), field by field. For a stacked POD this
/// is the best approximation of each field in the span of its part of the stacked modes.
/// Written to errors/L2errorProj<fieldName>_mat.txt
template <class FieldType>
void projectionError(
    const RunPaths& run,
    const word& fieldName,
    PtrList<FieldType>& modes,
    const PtrList<FieldType>& references,
    const Eigen::VectorXd& weights,
    const std::shared_ptr<ITHACAcontext>& context)
{
    PtrList<FieldType> fom = context->loadFieldsFromCaseFolder<FieldType>(fieldName, run.fomFolder());
    FieldType reference("reference", weights(0) * references[0]);
    for (label r = 1; r < weights.size(); r++)
    {
        reference += weights(r) * references[r];
    }

    PtrList<FieldType> fluctuations(fom.size());
    forAll(fom, i)
    {
        fluctuations.set(i, new FieldType("fluctuation", fom[i] - reference));
    }
    const label nModes = modes.size();
    Eigen::MatrixXd coeffs = ITHACAutilities::getCoeffs(fluctuations, modes, nModes);
    PtrList<FieldType> reconstructed = ITHACAutilities::reconstructFromCoeff(modes, coeffs, nModes);
    forAll(reconstructed, i)
    {
        reconstructed[i] += reference;
    }

    Eigen::MatrixXd l2Error = ITHACAutilities::errorL2Rel(fom, reconstructed);
    ITHACAstream::exportMatrix(l2Error, "L2errorProj" + fieldName, "eigen", run.errorFolder());
}


// * * * * * * * * * * * * * * Force coefficients * * * * * * * * * * * * * * //

/// Force coefficients of the ROM fields of one run, written to RunPaths::forceFile().
/// The forceCoeffs settings are read from system/forceCoeffsROM. The function object is added to the in-memory
/// controlDict under a name unique to the run: Time::run() creates it, with a new output file, at the start of
/// solveOnline. finish() destroys it, which closes the file, and moves the file to the results of the run.
/// Without system/forceCoeffsROM nothing is computed.
class RunForceCoefficients
{
public:
    RunForceCoefficients(Time& runTime, const dictionary& settings, const RunPaths& run)
        : runTime_(runTime),
          name_("forceCoeffs_" + run.caseID + "_" + run.suffix),
          target_(run.forceFile()),
          active_(!settings.empty())
    {
        if (!active_)
        {
            return;
        }
        functions().set(name_, settings);
        runTime_.functionObjects().clear();
    }

    void finish()
    {
        if (!active_)
        {
            return;
        }
        runTime_.functionObjects().clear();
        functions().remove(name_);

        // forceCoeffs writes postProcessing/<name>/<start time>/coefficient.dat
        const fileName dir(runTime_.globalPath() / functionObject::outputPrefix / name_);
        const fileNameList times = readDir(dir, fileName::DIRECTORY);
        const fileName source = times.size() ? dir / times[0] / "coefficient.dat" : fileName();
        if (Pstream::master())
        {
            if (!source.empty() && isFile(source))
            {
                mv(source, target_);
                rmDir(dir);
                Info << "-- Force coefficients written to " << target_ << endl;
            } else
            {
                WarningInFunction << "No force coefficients found in " << dir << endl;
            }
        }
    }

private:
    dictionary& functions()
    {
        return const_cast<dictionary&>(runTime_.controlDict()).subDictOrAdd("functions");
    }

    Time& runTime_;
    const word name_;
    const fileName target_;
    const bool active_;
};


/// The function objects of system/controlDict are meant for standalone FOM runs (e.g. liftAndDragFOM): inside
/// TestLSPG they would act on the FOM fields kept in memory. They are switched off here
void disableControlDictFunctionObjects(Time& runTime)
{
    const_cast<dictionary&>(runTime.controlDict()).set("functions", dictionary());
    runTime.functionObjects().clear();
}


/// forceCoeffs settings for the ROM runs (system/forceCoeffsROM), empty if the file does not exist
dictionary readForceCoeffsSettings(const Time& runTime)
{
    IOobject io("forceCoeffsROM", runTime.system(), runTime, IOobject::MUST_READ, IOobject::NO_WRITE,
        IOobject::NO_REGISTER);
    if (!io.typeHeaderOk<IOdictionary>(true))
    {
        WarningInFunction << "system/forceCoeffsROM not found: the force coefficients of the ROM are not computed" << endl;
        return dictionary();
    }
    return dictionary(IOdictionary(io));
}


// * * * * * * * * * * * * * * * * * * Main * * * * * * * * * * * * * * * * //

int main(int argc, char* argv[])
{
    // Create the context, which reads mesh, time and so on
    auto context = std::make_shared<ITHACAcontext>(argc, argv);
    OfflineFOM offlineFOM(context);
    disableControlDictFunctionObjects(context->runTime());

    // Offline phase: FOM snapshots and modes, shared by all the ROMs and runs
    offlineFOM.prepareParametersMu("./input_parameters/offline/BC");
    offlineFOM.inletIndex.resize(1, 2);
    offlineFOM.inletIndex << 0, 0;

    offlineFOM.offlineSolve("./ITHACAoutput/Offline/", "./input_parameters/offline/timeSettings");
    offlineFOM.prepareSnapshots();
    offlineFOM.splitEddyViscositySnapshots();
    offlineFOM.computePOD();

    offlineFOM.prepareModes();
    offlineFOM.offlineRBFInterpolation();

    offlineFOM.resizeModes();
    offlineFOM.switchOffAutoWrite();

    // Online phase: one ROM (romID in ITHACAdict), solved for every parametric run.
    // Run <suffix> needs input_parameters/ROM/ROM_BC_<suffix> and the FOM reference ITHACAoutput/OnlineFOM<suffix>
    const wordList suffixes({ "01", "04" });

    ReducedLSPGUnsteadyNSTurb reducedProblem(context, offlineFOM);
    const word caseID = reducedProblem.getID();
    const dictionary forceSettings = readForceCoeffsSettings(context->runTime());

    for (const word& suffix : suffixes)
    {
        const RunPaths run { caseID, suffix };
        Info << nl << "### ROM " << caseID << ", run " << suffix << ": results in " << run.romFolder() << endl;
        mkDir(run.errorFolder());

        RunForceCoefficients forces(context->runTime(), forceSettings, run);
        reducedProblem.solveOnline(run.bcFile(), run.romFolder());
        forces.finish();

        if (!haveFOMReference(run))
        {
            WarningInFunction << "No FOM reference in " << run.fomFolder() << ": errors of run " << suffix
                              << " not computed" << endl;
            continue;
        }
        compareWithFOM<volVectorField>(run, "U", context);
        compareWithFOM<volScalarField>(run, "p", context);
        compareWithFOM<volScalarField>(run, "k", context);
        compareWithFOM<volScalarField>(run, "omega", context);
        comparePressureWithoutMean(run, context);

        const Eigen::VectorXd weights = referenceWeights(reducedProblem, run);
        std::ostringstream weightsText;
        weightsText << weights.transpose();
        Info << "-- Weights of the offline runs in the reference: " << weightsText.str().c_str() << endl;
        projectionError(run, "U", offlineFOM.Umodes, offlineFOM.U0field, weights, context);
        projectionError(run, "p", offlineFOM.Pmodes, offlineFOM.P0field, weights, context);
        projectionError(run, "k", offlineFOM.kmodes, offlineFOM.k0field, weights, context);
        projectionError(run, "omega", offlineFOM.omegamodes, offlineFOM.omega0field, weights, context);
    }
    Info << nl << "### Plot with: gnuplot -e \"cases='" << caseID << "'; run='" << suffixes[0] << "'\" errorPlots.gp" << endl;

    // V1: consistency test, enabled by the consistencyTest subdictionary of ITHACAdict.
    // Runs last because it moves the shared time and fields.
    runConsistencyTest(offlineFOM, reducedProblem, context, suffixes[0]);

    return 0;
}
