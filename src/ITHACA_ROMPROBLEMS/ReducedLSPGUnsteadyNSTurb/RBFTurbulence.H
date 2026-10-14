#ifndef RBFTURBULENCE_H
#define RBFTURBULENCE_H

#include "fvCFD.H"

class RBFTurbulence
{
  private:
   const volScalarField& nut_;
   const dimensionedScalar& nu_;

  public:
    RBFTurbulence(const volScalarField& nut, const dimensionedScalar& nu)
    : nut_(nut), nu_(nu)
    {}

    tmp<volScalarField> nuEff() const
    {
        return tmp<volScalarField>
        (
            new volScalarField("nuEffROM", nu_ + nut_)
        );
    }
};

#endif
