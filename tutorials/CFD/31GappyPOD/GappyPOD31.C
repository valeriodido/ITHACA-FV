// Tutorial class for Gappy POD

#include "fvCFD.H"
#include "ITHACAutilities.H"
#include "ITHACAerror.H"
#include "Modes.H"
#include "ITHACAPOD.H"
#include "GappyPOD.H"

class GappyPODExample
{
 // Small class that holds the fvMesh, the Time object and has utilities to generate snapshots
  public:
    GappyPODExample(int argc, char *argv[])
    {
      // Standard OpenFOAM boilerplate to set up the Time and fvMesh objects
      _args = autoPtr<argList>(new argList(argc, argv));
      if (!_args->checkRootCase())
      {
        Foam::FatalError.exit();
      }
      _runTime = autoPtr<Time>(
        new Time(Time::controlDictName, _args()));
      
      _mesh = autoPtr<fvMesh>(
        new fvMesh(
          IOobject(
            fvMesh::defaultRegion,
            _runTime->timeName(),
            _runTime->time(),
            IOobject::MUST_READ
          )
        )
      );

      _para = ITHACAparameters::getInstance(*_mesh, *_runTime);
    }

    label nSnapshots;
    label nPODmodes;
    instantList snapshotTimes_;
    labelList snapshotIndices_;

    void generateSnapshots()
    {
      // Generate snapshots of a simple function f(x, y; μ) = Σ_{k=1..K}  (1/k) · sin(kπx)·sin(kπy) · cos(kμ + φ_k)
      label K = 10; // number of harmonics
      scalarList phi = {0.3, 1.1, 2.0, 0.7, 2.6, 3.0, 0.86, 1.6, 2.3, 0.9};   // fixed phases
      M_Assert(phi.size() == K, "phi.size() must be equal to K");
      _Tfield.setSize(nSnapshots);
      snapshotTimes_.setSize(nSnapshots);
      snapshotIndices_.setSize(nSnapshots);

      M_Assert(nSnapshots > 1, "nSnapshots must be greater than 1");
      M_Assert(nPODmodes > 0, "nPODmodes must be greater than 0");

      for (label i = 0; i < nSnapshots; i++)
      {
        runTime()++;
        const scalar mu =
            2.0 * constant::mathematical::pi * scalar(i) / scalar(nSnapshots - 1);
        snapshotTimes_[i] = instant(_runTime->value());
        snapshotIndices_[i] = _runTime->timeIndex();
        _Tfield.set
        (
            i,
            new volScalarField
            (
                IOobject
                (
                    "f",
                    _runTime->timeName(),
                    *_mesh,
                    IOobject::NO_READ,
                    IOobject::NO_WRITE
                ),
                *_mesh,
                dimensionedScalar("zero", dimless, 0.0)
            )
        );
    
        scalarField& values = _Tfield[i].primitiveFieldRef();
    
        forAll(values, cellI)
        {
            const scalar x = _mesh->C()[cellI].x();
            const scalar y = _mesh->C()[cellI].y();
    
            for (label k = 0; k < K; k++)
            {
                const scalar harmonic = scalar(k + 1);
    
                values[cellI] +=
                    (1.0 / harmonic)
                  * Foam::sin(harmonic * constant::mathematical::pi * x)
                  * Foam::sin(harmonic * constant::mathematical::pi * y)
                  * Foam::cos(harmonic * mu + phi[k]);
            }
        }
        _Tfield[i].correctBoundaryConditions(); // Unnecessary for this example
        _Tfield[i].write();
      }
    }
    
    void computePODmodes()
    {
      ITHACAPOD::getModes(_Tfield, _modes, _Tfield[0].name(), false, 0, 0, nPODmodes, false);
    }
    
    Time& runTime() { return *_runTime; }
    fvMesh& mesh() { return *_mesh; }
    PtrList<volScalarField>& Tfield() { return _Tfield; }
    PtrList<volScalarField>& modes() { return _modes; }

  private:
    ITHACAparameters* _para;
    autoPtr<argList> _args;
    autoPtr<Time> _runTime;
    autoPtr<fvMesh> _mesh;
    PtrList<volScalarField> _Tfield;
    PtrList<volScalarField> _modes;
};

labelList randomMask(
  const label nCells,
  const label m,
  const label seed = 42)
{
  labelList all(identity(nCells));
  Random rndGen(seed);
  for (label i = nCells - 1; i > 0; i--)
  {
    label j = rndGen.position<label>(0, i);
    Foam::Swap(all[i], all[j]);
  }
  all.resize(m);
  return all;
}

int main(int argc, char *argv[])
{
    GappyPODExample example(argc, argv);
    Time& runTime = example.runTime();
    fvMesh& mesh = example.mesh();

    example.nSnapshots = 50;
    // Note that in this case, the k-th mode,
    //  for k > n_harmonics, will be numerical noise.
    example.nPODmodes = 10;
    example.generateSnapshots();
    example.computePODmodes();

    label maskSize = 20; // Number of sample points for Gappy POD
    labelList modeSize = {2, 4, 6, 8, 10};  // Number of modes to use for Gappy POD
    Eigen::MatrixXd conditionNumbers(modeSize.size(), 2);

    for (label i = 0; i < modeSize.size(); i++)
    {
      word identifier = Foam::name(modeSize[i]) + "modes_" + Foam::name(maskSize) + "samples";
      Info << "Running Gappy POD with " << modeSize[i] << " modes and " << maskSize << " sample points." << endl;
      M_Assert(modeSize[i] <= example.nPODmodes, "modeSize must be less than or equal to nPODmodes");
      GappyPOD<scalar, fvPatchField, volMesh> gappyPOD(example.modes(), modeSize[i]);
      gappyPOD.randomSample(maskSize);
      gappyPOD.setMask();
      gappyPOD.offline("./ITHACAoutput/gappyPOD", false);
      
      volScalarField sampleMesh = gappyPOD.getSampleMesh("gappySampleMesh");
      ITHACAstream::exportSolution(sampleMesh, "1", "./ITHACAoutput/gappyMesh", "randomMesh");
      conditionNumbers(i, 0) = modeSize[i];
      conditionNumbers(i, 1) = gappyPOD.getConditionNumber();

      PtrList<volScalarField> reconstructedFields(example.nSnapshots);
      for (label i = 0; i < example.nSnapshots; i++)
      {
        runTime.setTime(example.snapshotTimes_[i], example.snapshotIndices_[i]);
        reconstructedFields.set
        (
            i,
            new volScalarField(
            gappyPOD.reconstruct(example.Tfield()[i], "f_reconstructed" + identifier)
            )
        );
        reconstructedFields[i].write();
      }
      Eigen::MatrixXd l2_error = ITHACAutilities::errorL2Rel(example.Tfield(), reconstructedFields);
      ITHACAstream::exportMatrix(l2_error, "l2_error_" + identifier, "eigen", "./ITHACAoutput/l2_error");
    }
    ITHACAstream::exportMatrix(conditionNumbers, "condition_numbers", "eigen", "./ITHACAoutput/condition_numbers");

    return 0;
}
