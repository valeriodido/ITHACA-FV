// Tests of Modes::reconstructInPlace() against Modes::reconstruct()
#include "fvCFD.H"
#include "pointFields.H"
#include "valuePointPatchField.H"
#include "Modes.H"
#include "ITHACAparameters.H"
#include "../OpenFOAMTest.hpp"
#include <gtest/gtest.h>
#include <chrono>
#include <iomanip>
#include <type_traits>

/*
The in-place reconstruction must give the same field as field = modes.reconstruct(field, ...), boundary
values included, for every instantiated Modes type: volScalarModes, volVectorModes, surfaceScalarModes
and pointVectorModes. The modes are not a POD basis: any set of fields exercises the same code paths.
They carry non-trivial values on every patch, and the patches cover the boundary treatments of
reconstruct(): fixedValue, zeroGradient, fixedGradient, calculated and empty.
The timing test reports the cost of both versions on fields with two stored old-time levels,
as in a transient solver with a second-order time scheme.
*/

namespace
{
const scalar pi = constant::mathematical::pi;

// Smooth, mode-dependent, sign-changing function
scalar modeFunction(const label k, const point& p)
{
    const scalar a = (k % 4 + 1) * pi;
    const scalar b = (k / 4 + 1) * pi;
    return Foam::sin(a * p.x() + 0.3 * k) * Foam::cos(b * p.y()) + 0.1 * k * p.x();
}

vector modeVector(const label k, const point& p)
{
    return vector(modeFunction(k, p), modeFunction(k + 7, p), 0.5 * modeFunction(k + 3, p));
}

// Patch types by patch name; empty patches keep the empty type, the others default to calculated
wordList patchTypes(const wordList& patchNames, const wordList& constraintTypes,
                    const HashTable<word>& types)
{
    wordList out(patchNames.size(), word("calculated"));
    forAll(patchNames, patchi)
    {
        if (constraintTypes[patchi] == "empty")
        {
            out[patchi] = "empty";
        }
        else if (types.found(patchNames[patchi]))
        {
            out[patchi] = types[patchNames[patchi]];
        }
    }
    return out;
}

// Largest difference between two fields, internal and boundary values
template<class Type, template<class> class PatchField, class GeoMesh>
scalar maxDifference(
    const GeometricField<Type, PatchField, GeoMesh>& a,
    const GeometricField<Type, PatchField, GeoMesh>& b)
{
    scalar diff = 0;
    if (a.size())
    {
        diff = Foam::max(Foam::mag(a.primitiveField() - b.primitiveField()));
    }

    forAll(a.boundaryField(), patchi)
    {
        if constexpr (std::is_same<GeoMesh, pointMesh>::value)
        {
            // Only value point patches store values
            if (isA<valuePointPatchField<Type>>(a.boundaryField()[patchi]))
            {
                const Field<Type>& va = refCast<const valuePointPatchField<Type>>(a.boundaryField()[patchi]);
                const Field<Type>& vb = refCast<const valuePointPatchField<Type>>(b.boundaryField()[patchi]);
                if (va.size())
                {
                    diff = Foam::max(diff, Foam::max(Foam::mag(va - vb)));
                }
            }
        }
        else
        {
            if (a.boundaryField()[patchi].size())
            {
                diff = Foam::max(diff,
                    Foam::max(Foam::mag(a.boundaryField()[patchi] - b.boundaryField()[patchi])));
            }
        }
    }

    // reconstruct() sets the gradient of fixedGradient patches
    if constexpr (std::is_same<GeoMesh, volMesh>::value && std::is_same<Type, scalar>::value)
    {
        forAll(a.boundaryField(), patchi)
        {
            if (isA<fixedGradientFvPatchScalarField>(a.boundaryField()[patchi]) && a.boundaryField()[patchi].size())
            {
                const scalarField& ga = refCast<const fixedGradientFvPatchScalarField>(a.boundaryField()[patchi]).gradient();
                const scalarField& gb = refCast<const fixedGradientFvPatchScalarField>(b.boundaryField()[patchi]).gradient();
                diff = Foam::max(diff, Foam::max(Foam::mag(ga - gb)));
            }
        }
    }
    return diff;
}

// Largest absolute value of a field, to make the differences relative
template<class Type, template<class> class PatchField, class GeoMesh>
scalar maxMagnitude(const GeometricField<Type, PatchField, GeoMesh>& f)
{
    return f.size() ? Foam::max(Foam::mag(f.primitiveField())) : 0;
}
}


class TestCase
{
public:
    TestCase(int argc, char* argv[])
    {
        argList::noBanner();
        _args = autoPtr<argList>(new argList(argc, argv, true, true, false));
        if (!_args->checkRootCase())
        {
            Foam::FatalError.exit();
        }
        _runTime = autoPtr<Time>(new Time(Time::controlDictName, _args()));
        _mesh = autoPtr<fvMesh>(
            new fvMesh(
                IOobject(
                    fvMesh::defaultRegion,
                    _runTime->timeName(),
                    _runTime->time(),
                    IOobject::MUST_READ)));
        // assignBC reads the warning settings from the ITHACAparameters instance
        ITHACAparameters::getInstance(*_mesh, *_runTime);
    }

    Time& runTime() { return *_runTime; }
    fvMesh& mesh() { return *_mesh; }

    // Not registered: several fields of the same name coexist in the tests
    IOobject io(const word& name) const
    {
        return IOobject(name, _runTime->timeName(), *_mesh, IOobject::NO_READ, IOobject::NO_WRITE, false);
    }

    wordList fvPatchTypes(const HashTable<word>& types) const
    {
        return patchTypes(_mesh->boundaryMesh().names(), _mesh->boundaryMesh().types(), types);
    }

    // *** Templates, with the boundary types under test *** //

    volScalarField scalarTemplate(const word& name) const
    {
        HashTable<word> types;
        types.insert("inlet", "fixedValue");
        types.insert("outlet", "zeroGradient");
        types.insert("walls", "fixedGradient");
        return volScalarField(io(name), *_mesh, dimensionedScalar(dimless, 0), fvPatchTypes(types));
    }

    volVectorField vectorTemplate(const word& name) const
    {
        HashTable<word> types;
        types.insert("inlet", "fixedValue");
        types.insert("outlet", "zeroGradient");
        types.insert("walls", "fixedValue");
        return volVectorField(io(name), *_mesh, dimensionedVector(dimless, Zero), fvPatchTypes(types));
    }

    surfaceScalarField surfaceTemplate(const word& name) const
    {
        return surfaceScalarField(io(name), *_mesh, dimensionedScalar(dimless, 0), fvPatchTypes(HashTable<word>()));
    }

    pointVectorField pointTemplate(const word& name) const
    {
        const pointMesh& pMesh = pointMesh::New(*_mesh);
        wordList names(pMesh.boundary().size());
        wordList constraints(pMesh.boundary().size());
        forAll(pMesh.boundary(), patchi)
        {
            names[patchi] = pMesh.boundary()[patchi].name();
            constraints[patchi] = pMesh.boundary()[patchi].type();
        }
        HashTable<word> types;
        types.insert("inlet", "calculated");
        types.insert("outlet", "zeroGradient");
        types.insert("walls", "fixedValue");
        return pointVectorField(io(name), pMesh, dimensionedVector(dimless, Zero),
                                patchTypes(names, constraints, types));
    }

    // *** Mode sets, with values on every patch *** //

    volScalarModes scalarModes(const word& name, const label nModes) const
    {
        PtrList<volScalarField> fields(nModes);
        const volVectorField& C = _mesh->C();
        for (label k = 0; k < nModes; k++)
        {
            fields.set(k, new volScalarField(scalarTemplate(name)));
            volScalarField& f = fields[k];
            forAll(f, celli)
            {
                f[celli] = modeFunction(k, C[celli]);
            }
            forAll(f.boundaryField(), patchi)
            {
                const vectorField& Cf = C.boundaryField()[patchi];
                scalarField values(Cf.size());
                forAll(values, facei)
                {
                    values[facei] = modeFunction(k, Cf[facei]);
                }
                f.boundaryFieldRef()[patchi] == values;
            }
        }
        volScalarModes modes;
        modes = fields;
        return modes;
    }

    volVectorModes vectorModes(const word& name, const label nModes) const
    {
        PtrList<volVectorField> fields(nModes);
        const volVectorField& C = _mesh->C();
        for (label k = 0; k < nModes; k++)
        {
            fields.set(k, new volVectorField(vectorTemplate(name)));
            volVectorField& f = fields[k];
            forAll(f, celli)
            {
                f[celli] = modeVector(k, C[celli]);
            }
            forAll(f.boundaryField(), patchi)
            {
                const vectorField& Cf = C.boundaryField()[patchi];
                vectorField values(Cf.size());
                forAll(values, facei)
                {
                    values[facei] = modeVector(k, Cf[facei]);
                }
                f.boundaryFieldRef()[patchi] == values;
            }
        }
        volVectorModes modes;
        modes = fields;
        return modes;
    }

    surfaceScalarModes surfaceModes(const word& name, const label nModes) const
    {
        PtrList<surfaceScalarField> fields(nModes);
        const surfaceVectorField& Cf = _mesh->Cf();
        for (label k = 0; k < nModes; k++)
        {
            fields.set(k, new surfaceScalarField(surfaceTemplate(name)));
            surfaceScalarField& f = fields[k];
            forAll(f, facei)
            {
                f[facei] = modeFunction(k, Cf[facei]);
            }
            forAll(f.boundaryField(), patchi)
            {
                const vectorField& Cp = Cf.boundaryField()[patchi];
                scalarField values(Cp.size());
                forAll(values, facei)
                {
                    values[facei] = modeFunction(k, Cp[facei]);
                }
                f.boundaryFieldRef()[patchi] == values;
            }
        }
        surfaceScalarModes modes;
        modes = fields;
        return modes;
    }

    pointVectorModes pointModes(const word& name, const label nModes) const
    {
        PtrList<pointVectorField> fields(nModes);
        const pointField& points = _mesh->points();
        for (label k = 0; k < nModes; k++)
        {
            fields.set(k, new pointVectorField(pointTemplate(name)));
            pointVectorField& f = fields[k];
            forAll(f, pointi)
            {
                f[pointi] = modeVector(k, points[pointi]);
            }
            forAll(f.boundaryField(), patchi)
            {
                if (isA<valuePointPatchField<vector>>(f.boundaryField()[patchi]))
                {
                    valuePointPatchField<vector>& pf =
                        refCast<valuePointPatchField<vector>>(f.boundaryFieldRef()[patchi]);
                    pf == pf.patchInternalField()();
                }
            }
        }
        pointVectorModes modes;
        modes = fields;
        return modes;
    }

private:
    autoPtr<argList> _args;
    autoPtr<Time> _runTime;
    autoPtr<fvMesh> _mesh;
};


class ModesReconstructFixture : public testing::Test
{
protected:
    void SetUp() override
    {
        testCase = std::make_unique<TestCase>(OpenFOAMTest::argc, OpenFOAMTest::argv);
        testCase->runTime().setTime(0, 0);
        std::srand(42);
    }

    // Reconstructs with both versions from nSets random coefficient vectors and compares the results.
    // Returns the largest relative difference
    template<class Type, template<class> class PatchField, class GeoMesh>
    scalar compareReconstructions(
        Modes<Type, PatchField, GeoMesh>& modes,
        const GeometricField<Type, PatchField, GeoMesh>& fieldTemplate,
        const label nCoeffs,
        const word& name,
        const bool correctBC,
        const label nSets = 5)
    {
        typedef GeometricField<Type, PatchField, GeoMesh> FieldType;
        scalar maxRelDiff = 0;
        for (label s = 0; s < nSets; s++)
        {
            const Eigen::VectorXd coeffs = Eigen::VectorXd::Random(nCoeffs);
            FieldType original(fieldTemplate);
            FieldType inPlace(fieldTemplate);

            original = modes.reconstruct(original, coeffs, name, correctBC);
            modes.reconstructInPlace(inPlace, coeffs, name, correctBC);

            EXPECT_EQ(original.name(), inPlace.name());
            const scalar scale = Foam::max(maxMagnitude(original), VSMALL);
            maxRelDiff = Foam::max(maxRelDiff, maxDifference(original, inPlace) / scale);
        }
        return maxRelDiff;
    }

    std::unique_ptr<TestCase> testCase;
    const label nModes = 15;
    const scalar tolerance = 1e-12;
};


TEST_F(ModesReconstructFixture, VolScalarField)
{
    volScalarModes modes = testCase->scalarModes("p", nModes);
    const volScalarField fieldTemplate(testCase->scalarTemplate("p"));
    EXPECT_LT(compareReconstructions(modes, fieldTemplate, nModes, "p_rec", true), tolerance);
    EXPECT_LT(compareReconstructions(modes, fieldTemplate, nModes, "p_rec", false), tolerance);
    // Fewer coefficients than modes
    EXPECT_LT(compareReconstructions(modes, fieldTemplate, nModes - 5, "p", true), tolerance);
}

TEST_F(ModesReconstructFixture, VolScalarFieldNutClipping)
{
    // reconstruct() clips the internal field when the input is named nut, the boundary when Name is nut
    volScalarModes modes = testCase->scalarModes("nut", nModes);
    const volScalarField fieldTemplate(testCase->scalarTemplate("nut"));
    EXPECT_LT(compareReconstructions(modes, fieldTemplate, nModes, "nut", true), tolerance);
    EXPECT_LT(compareReconstructions(modes, fieldTemplate, nModes, "nut_rec", true), tolerance);

    // The clipping is actually exercised: the same modes and coefficients give negative values under another name
    const Eigen::VectorXd coeffs = Eigen::VectorXd::Random(nModes);
    volScalarModes pModes = testCase->scalarModes("p", nModes);
    volScalarField p(testCase->scalarTemplate("p"));
    volScalarField nut(fieldTemplate);
    pModes.reconstructInPlace(p, coeffs, "p", true);
    modes.reconstructInPlace(nut, coeffs, "nut", true);
    EXPECT_LT(Foam::min(p.primitiveField()), 0) << "The test modes should produce negative values";
    EXPECT_GE(Foam::min(nut.primitiveField()), 0);
}

TEST_F(ModesReconstructFixture, VolVectorField)
{
    volVectorModes modes = testCase->vectorModes("U", nModes);
    const volVectorField fieldTemplate(testCase->vectorTemplate("U"));
    EXPECT_LT(compareReconstructions(modes, fieldTemplate, nModes, "U_rec", true), tolerance);
    EXPECT_LT(compareReconstructions(modes, fieldTemplate, nModes, "U_rec", false), tolerance);
    EXPECT_LT(compareReconstructions(modes, fieldTemplate, nModes - 5, "U", true), tolerance);
}

TEST_F(ModesReconstructFixture, SurfaceScalarField)
{
    surfaceScalarModes modes = testCase->surfaceModes("phi", nModes);
    const surfaceScalarField fieldTemplate(testCase->surfaceTemplate("phi"));
    EXPECT_LT(compareReconstructions(modes, fieldTemplate, nModes, "phi_rec", true), tolerance);
    EXPECT_LT(compareReconstructions(modes, fieldTemplate, nModes, "phi_rec", false), tolerance);
    EXPECT_LT(compareReconstructions(modes, fieldTemplate, nModes - 5, "phi", true), tolerance);
}

TEST_F(ModesReconstructFixture, PointVectorField)
{
    pointVectorModes modes = testCase->pointModes("pointDisplacement", nModes);
    const pointVectorField fieldTemplate(testCase->pointTemplate("pointDisplacement"));
    EXPECT_LT(compareReconstructions(modes, fieldTemplate, nModes, "pointDisplacement_rec", true), tolerance);
    EXPECT_LT(compareReconstructions(modes, fieldTemplate, nModes, "pointDisplacement_rec", false), tolerance);
    EXPECT_LT(compareReconstructions(modes, fieldTemplate, nModes - 5, "pointDisplacement", true), tolerance);
}

TEST_F(ModesReconstructFixture, OldTimeLevels)
{
    // Both versions store the previous state as old time at the first modification of a new time step
    volVectorModes modes = testCase->vectorModes("U", nModes);
    volVectorField original(testCase->vectorTemplate("U"));
    volVectorField inPlace(testCase->vectorTemplate("U"));
    original.oldTime().oldTime();
    inPlace.oldTime().oldTime();

    Time& runTime = testCase->runTime();
    for (label step = 0; step < 3; step++)
    {
        runTime++;
        // Several reconstructions per time step, as in a nonlinear solver
        for (label it = 0; it < 3; it++)
        {
            const Eigen::VectorXd coeffs = Eigen::VectorXd::Random(nModes);
            original = modes.reconstruct(original, coeffs, "U", true);
            modes.reconstructInPlace(inPlace, coeffs, "U", true);
        }
        const scalar scale = maxMagnitude(original);
        EXPECT_EQ(original.timeIndex(), inPlace.timeIndex());
        EXPECT_LT(maxDifference(original, inPlace) / scale, tolerance);
        EXPECT_LT(maxDifference(original.oldTime(), inPlace.oldTime()) / scale, tolerance);
        EXPECT_LT(maxDifference(original.oldTime().oldTime(), inPlace.oldTime().oldTime()) / scale, tolerance);
    }
}


// *** Timings *** //

namespace
{
struct Timing
{
    std::string type;
    label size;
    double originalMs;
    double inPlaceMs;
};

std::vector<Timing> timings;

// Mean time per call of field = reconstruct(field, ...) and of reconstructInPlace(field, ...), on a field
// with two old-time levels. The coefficients change at every call, as in a nonlinear solver
template<class Type, template<class> class PatchField, class GeoMesh>
Timing timeReconstructions(
    const std::string& type,
    Modes<Type, PatchField, GeoMesh>& modes,
    const GeometricField<Type, PatchField, GeoMesh>& fieldTemplate,
    const label nModes,
    const label nRepetitions)
{
    typedef std::chrono::steady_clock clock;
    GeometricField<Type, PatchField, GeoMesh> original(fieldTemplate);
    GeometricField<Type, PatchField, GeoMesh> inPlace(fieldTemplate);
    original.oldTime().oldTime();
    inPlace.oldTime().oldTime();
    const word name = fieldTemplate.name();
    const Eigen::MatrixXd coeffs = Eigen::MatrixXd::Random(nModes, nRepetitions);

    // Warm up, also builds EigenModes
    original = modes.reconstruct(original, coeffs.col(0), name, true);
    modes.reconstructInPlace(inPlace, coeffs.col(0), name, true);

    const auto t0 = clock::now();
    for (label r = 0; r < nRepetitions; r++)
    {
        original = modes.reconstruct(original, coeffs.col(r), name, true);
    }
    const auto t1 = clock::now();
    for (label r = 0; r < nRepetitions; r++)
    {
        modes.reconstructInPlace(inPlace, coeffs.col(r), name, true);
    }
    const auto t2 = clock::now();

    const double toMs = 1e-3 / nRepetitions;
    return Timing
    {
        type,
        fieldTemplate.size(),
        std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() * toMs,
        std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() * toMs
    };
}
}

TEST_F(ModesReconstructFixture, Timing)
{
    const label nRepetitions = 200;
    timings.clear();
    {
        volScalarModes modes = testCase->scalarModes("p", nModes);
        timings.push_back(timeReconstructions("volScalarField", modes, testCase->scalarTemplate("p"), nModes, nRepetitions));
    }
    {
        volVectorModes modes = testCase->vectorModes("U", nModes);
        timings.push_back(timeReconstructions("volVectorField", modes, testCase->vectorTemplate("U"), nModes, nRepetitions));
    }
    {
        surfaceScalarModes modes = testCase->surfaceModes("phi", nModes);
        timings.push_back(timeReconstructions("surfaceScalarField", modes, testCase->surfaceTemplate("phi"), nModes, nRepetitions));
    }
    {
        pointVectorModes modes = testCase->pointModes("pointDisplacement", nModes);
        timings.push_back(timeReconstructions("pointVectorField", modes, testCase->pointTemplate("pointDisplacement"), nModes, nRepetitions));
    }

    std::cout << "\nMean time per reconstruction (" << nModes << " modes, " << nRepetitions
              << " repetitions, 2 old-time levels)\n"
              << std::left << std::setw(20) << "field" << std::right << std::setw(10) << "size"
              << std::setw(16) << "original [ms]" << std::setw(16) << "inPlace [ms]" << std::setw(10) << "speedup" << "\n";
    for (const Timing& t : timings)
    {
        std::cout << std::left << std::setw(20) << t.type << std::right << std::setw(10) << t.size
                  << std::fixed << std::setprecision(4)
                  << std::setw(16) << t.originalMs << std::setw(16) << t.inPlaceMs
                  << std::setprecision(2) << std::setw(9) << t.originalMs / t.inPlaceMs << "x\n";
        RecordProperty(t.type + "_original_ms", std::to_string(t.originalMs));
        RecordProperty(t.type + "_inPlace_ms", std::to_string(t.inPlaceMs));
    }
    std::cout << std::endl;
}


int main(int argc, char** argv)
{
    OpenFOAMTest::setArguments(argc, argv);
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
