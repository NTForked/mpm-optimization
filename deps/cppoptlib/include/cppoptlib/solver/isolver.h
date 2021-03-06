// CppNumericalSolver
#ifndef ISOLVER_H_
#define ISOLVER_H_

#include <functional>
#include <chrono>

#include "isolver.h"
#include "../meta.h"
#include "../problem.h"

namespace cppoptlib {

template<typename T, int Ord>
class ISolver {
protected:
  const int order_ = Ord;
 public:
  Options settings_;
  int n_iters;
  std::vector<T> obj_vals;
  std::vector<double> runtimes;
  ISolver() {
    settings_ = Options();
  }

  /**
   * @brief minimize an objective function given a gradient (and optinal a hessian)
   * @details this is just the abstract interface
   *
   * @param x0 starting point
   * @param funObjective objective function
   * @param funGradient gradient function
   * @param funcHession hessian function
   */
  virtual void minimize(Problem<T> &objFunc, Vector<T> & x0) = 0;

};

} /* namespace cppoptlib */

#endif /* ISOLVER_H_ */
