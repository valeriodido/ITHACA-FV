fvMesh& mesh = this->mesh();
fv::options& fvOptions = this->fvOptions();
IOMRFZoneList& MRF = this->MRF();

volVectorField& U = _U();
volScalarField& p_rgh = _p_rgh();
volScalarField& T = _T();
volScalarField& nut = _nut();
volScalarField& alphat = _alphat();
volScalarField& rhok = _rhok();
volScalarField& gh = _gh();
surfaceScalarField& ghf = _ghf();
surfaceScalarField& phi = _phi();
dimensionedScalar& beta = _beta();
dimensionedScalar& TRef = _TRef();
dimensionedScalar& Pr = _Pr();
dimensionedScalar& Prt = _Prt();