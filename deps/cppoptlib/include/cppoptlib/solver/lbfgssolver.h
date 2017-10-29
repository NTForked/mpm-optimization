/*
 *  mylbfgssolver.h
 *
 *
 *  All rights are retained by the authors and the University of Minnesota.
 *
 *  Author: Ioannis Karamouzas
 *  Contact: ioannis@cs.umn.edu
 */

#include <iostream>

#include "isolver.h"
#include "../linesearch/morethuente.h"

#include <Eigen/Dense>

#ifndef MYLBFGSSOLVER_H_
#define MYLBFGSSOLVER_H_

namespace cppoptlib {
/**
 * @brief  LBFGS implementation based on Nocedal & Wright Numerical Optimization book (Section 7.2)
 * @tparam T scalar type
 * @tparam P problem type
 * @tparam Ord order of solver
 */

template<typename T>
class lbfgssolver : public ISolver<T, 1> {
public:
    void minimize(Problem<T>& objFunc, Vector<T>& x0)
    {
        int   _m      = std::min(int(this->settings_.maxIter), 10);
        int   _noVars = x0.size();
        float _eps_g  = this->settings_.gradTol;
        float _eps_x  = 1e-8;


        Eigen::MatrixXf s = Eigen::MatrixXf::Zero(_noVars, _m);
        Eigen::MatrixXf y = Eigen::MatrixXf::Zero(_noVars, _m);

        Vector<float> alpha = Vector<float>::Zero(_m);
        Vector<float> rho   = Vector<float>::Zero(_m);
        Vector<float> grad(_noVars), q(_noVars), grad_old(_noVars), x_old(_noVars);

        //	float f = objFunc.value(x0);
        float f              = objFunc.value_gradient(x0, grad);
        float gamma_k        = this->settings_.init_hess;
        float gradNorm       = 0;
        float alpha_init     = std::min(1.0, 1.0 / grad.lpNorm<Eigen::Infinity>());
        int   globIter       = 0;
        int   maxiter        = this->settings_.maxIter;
        float new_hess_guess = 1.0; // only changed if we converged to a solution

        for(int k = 0; k < maxiter; k++) {
            x_old    = x0;
            grad_old = grad;
            q        = grad;
            globIter++;

            //L - BFGS first - loop recursion
            int iter = std::min(_m, k);
            for(int i = iter - 1; i >= 0; --i) {
                rho(i)   = 1.0 / ((s.col(i)).dot(y.col(i)));
                alpha(i) = rho(i) * (s.col(i)).dot(q);
                q        = q - alpha(i) * y.col(i);
            }

            //L - BFGS second - loop recursion
            q = gamma_k * q;
            for(int i = 0; i < iter; ++i) {
                float beta = rho(i) * q.dot(y.col(i));
                q = q + (alpha(i) - beta) * s.col(i);
            }

            // is there a descent
            float dir = q.dot(grad);
            if(dir < 1e-4) {
                q          = grad;
                maxiter   -= k;
                k          = 0;
                alpha_init = std::min(1.0, 1.0 / grad.lpNorm<Eigen::Infinity>());
            }

            const float rate = MoreThuente<T, decltype(objFunc), 1>::linesearch(x0, -q,  objFunc, alpha_init);
            //		const float rate = linesearch(objFunc, x0, -q, f, grad, 1.0);
            x0 = x0 - rate * q;
            if((x_old - x0).squaredNorm() < _eps_x) {
                //			std::cout << "x diff norm: " << (x_old - x0).squaredNorm() << std::endl;
                break;
            }     // usually this is a problem so exit

            //		f = objFunc.value(x0);
            f = objFunc.value_gradient(x0, grad);

            gradNorm = grad.lpNorm<Eigen::Infinity>();
            if(gradNorm < _eps_g) {
                // Only change hessian guess if we break out the loop via convergence.
                //			std::cout << "grad norm: " << gradNorm << std::endl;
                new_hess_guess = gamma_k;
                break;
            }

            Vector<float> s_temp = x0 - x_old;
            Vector<float> y_temp = grad - grad_old;

            // update the history
            if(k < _m) {
                s.col(k) = s_temp;
                y.col(k) = y_temp;
            } else {
                s.leftCols(_m - 1) = s.rightCols(_m - 1).eval();
                s.rightCols(1)     = s_temp;
                y.leftCols(_m - 1) = y.rightCols(_m - 1).eval();
                y.rightCols(1)     = y_temp;
            }


            gamma_k    = s_temp.dot(y_temp) / y_temp.dot(y_temp);
            alpha_init = 1.0;
        }

        this->n_iters             = globIter;
        this->settings_.init_hess = new_hess_guess;
    }   // end minimize
};
}
/* namespace cppoptlib */

#endif /* MYLBFGSSOLVER_H_ */
