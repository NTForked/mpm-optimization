// CppNumericalSolver
#ifndef META_H
#define META_H

#include <Eigen/Dense>

namespace cppoptlib {
template<typename RealType>
using EgVector = Eigen::Matrix<RealType, Eigen::Dynamic, 1>;

template<typename RealType>
using EgMatrix = Eigen::Matrix<RealType, Eigen::Dynamic, Eigen::Dynamic>;


template<class RealType>
struct Options
{
    RealType gradTol;
    RealType rate;
    RealType init_hess;
    size_t   maxIter;
    size_t   m;
    bool     store_obj;
    bool     store_runtime;
    bool     use_max_iters;

    Options()
    {
        rate          = RealType(0.00005);
        maxIter       = RealType(100000);
        gradTol       = RealType(1e-4);
        m             = 10;
        store_obj     = false;
        store_runtime = false;
        use_max_iters = false;
        init_hess     = RealType(1.0); // only used by lbfgs
    }
};

//template<typename T>
//bool checkConvergence(T val_new, T val_old, Vector<T> grad, Vector<T> x_new, Vector<T> x_old)
//{
//    T ftol = 1e-10;
//    T gtol = 1e-8;
//    T xtol = 1e-32;
//
//    // value crit.
//    if((x_new - x_old).cwiseAbs().maxCoeff() < xtol) {
//        return true;
//    }
//
//    // // absol. crit
//    if(abs(val_new - val_old) / (abs(val_new) + ftol) < ftol) {
//        std::cout << abs(val_new - val_old) / (abs(val_new) + ftol) << std::endl;
//        std::cout << val_new << std::endl;
//        std::cout << val_old << std::endl;
//        std::cout << abs(val_new - val_old) / (abs(val_new) + ftol) << std::endl;
//        return true;
//    }
//
//    // gradient crit
//    T g = grad.template lpNorm<Eigen::Infinity>();
//    if(g < gtol) {
//        return true;
//    }
//    return false;
//}
}
#endif /* META_H */
