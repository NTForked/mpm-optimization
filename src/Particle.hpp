// Copyright (c) 2016 University of Minnesota
//
// MPM-OPTIMIZATION Uses the BSD 2-Clause License (http://www.opensource.org/licenses/BSD-2-Clause)
// Redistribution and use in source and binary forms, with or without modification, are
// permitted provided that the following conditions are met:
// 1. Redistributions of source code must retain the above copyright notice, this list of
// conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice, this list
// of conditions and the following disclaimer in the documentation and/or other materials
// provided with the distribution.
// THIS SOFTWARE IS PROVIDED "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR  A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE UNIVERSITY OF MINNESOTA, DULUTH OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA,
// OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER
// IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
// OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// By Matt Overby (http://www.mattoverby.net)

#ifndef MPM_PARTICLE_H
#define MPM_PARTICLE_H 1

#include <Banana/Setup.h>
#include <Banana/LinearAlgebra/ImplicitQRSVD.h>
#include <Banana/LinearAlgebra/LinaHelpers.h>

#include <iostream>
#include <memory>
#include "Interp.hpp"

using namespace Banana;

namespace mpm {
// Projection, Singular Values, SVD's U, SVD's V transpose
static inline void oriented_svd(const Mat3x3r& F, Vec3r& S, Mat3x3r& U, Mat3x3r& Vt)
{
    QRSVD::svd(F, U, S, Vt);
    Vt = glm::transpose(Vt);
    Mat3x3r J = Mat3x3r(1.0);
    J[2][2] = -1.0;

    // Check for inversion
    if(glm::determinant(U) < 0.0) { U = U * J; S[2] *= -1.0; }
    if(glm::determinant(Vt) < 0.0) { Vt = J * Vt; S[2] *= -1.0; }
}   // end oriented svd

// Particle (material point) class
class Particle {
public:
    Particle() : m(0.1), vol(0.0), v(0, 0, 0), x(0, 0, 0)
    {
        Fe    = Mat3x3r(1.0);
        tempP = Mat3x3r(1.0);
        B     = Mat3x3r(0);
        D     = Mat3x3r(0);
    }

    virtual Mat3x3r get_piola_stress(Mat3x3r& currF) const  = 0;
    virtual Real    get_energy_density(Mat3x3r& newF) const = 0;
    virtual Mat3x3r get_deform_grad() { return Fe; }
    virtual void    update_deform_grad(Mat3x3r velocity_grad, Real timestep_s)
    {
        Fe = (Mat3x3r(1.0) + timestep_s * velocity_grad) * Fe;
    }

    Real    m;     // mass
    Real    vol;   // rest volume, computed at first timestep
    Vec3r   v;     // velocity
    Vec3r   x;     // location
    Mat3x3r B;     // affine matrix (eq 176 course notes)
    Mat3x3r D;     // helper matrix, set in p_to_g mass (eq 174 course notes)

    Mat3x3r tempP; // temporary piola stress tensor computed by solver
    Mat3x3r Fe;    // elastic deformation gradient
protected:
};

// NeoHookean Particle
class pNeoHookean : public Particle {
public:

    pNeoHookean() : mu(10), lambda(10) {}
    Real mu;
    Real lambda;

    Mat3x3r get_piola_stress(Mat3x3r& currF) const
    {
        // Fix inversions:
        Mat3x3r U, Vt, Ftemp;
        Vec3r   S;
        oriented_svd(currF, S, U, Vt);
        if(S[2] < 0.0) { S[2] *= -1.0; }
        Ftemp = U * LinaHelpers::diagMatrix(S) * Vt;

        // Compute Piola stress tensor:
        Real J = glm::determinant(Ftemp);
        assert(isreal(J));
        assert(J > 0.0);
        Mat3x3r Fit = glm::transpose(glm::inverse(Ftemp));           // F^(-T)
        Mat3x3r P   = mu * (Ftemp - Fit) + lambda * (log(J) * Fit);
        for(int i = 0; i < 3; ++i) {
            for(int j = 0; j < 3; ++j) {
                assert(isreal(P[i][ j]));
            }
        }
        return P;
    }     // end compute piola stress tensor

    Real get_energy_density(Mat3x3r& newF) const
    {
        // Fix inversions:
        Mat3x3r U, Vt, Ftemp;
        Vec3r   S;
        oriented_svd(newF, S, U, Vt);
        if(S[2] < 0.0) { S[2] *= -1.0; }
        Ftemp = U * LinaHelpers::diagMatrix(S) * Vt;

        // Compute energy density:
        Real J = glm::determinant(Ftemp);
        assert(isreal(J));
        assert(J > 0.0);
        Real t1 = 0.5 * mu * (LinaHelpers::trace(glm::transpose(Ftemp) * Ftemp) - 3.0);
        Real t2 = -mu* log(J);
        Real t3 = 0.5 * lambda * (log(J) * log(J));
        assert(isreal(t1));
        assert(isreal(t2));
        assert(isreal(t3));
        return (t1 + t2 + t3);
    }   // end compute energy density
};    // end class neohookean particle
}     // end namespace mpm

#endif
