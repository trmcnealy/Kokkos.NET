
#include "Tests.hpp"

template<class ExecutionSpace>
static void TestGaussNewton()
{
    const int n = 2;

    const double precision          = 1.0E-09;
    const int    maximum_iterations = 500;

    Kokkos::Extension::Vector<double, ExecutionSpace> x0("x0", n);
    x0[0] = -3.0;
    x0[1] = 2.0;

    Kokkos::Extension::Vector<double, ExecutionSpace> xmin("xmin", n);
    xmin[0] = -10.0;
    xmin[1] = -10.0;

    Kokkos::Extension::Vector<double, ExecutionSpace> xmax("xmax", n);
    xmax[0] = 10.0;
    xmax[1] = 10.0;

    rosenbrock<double, ExecutionSpace> func;

    typedef decltype(func) rosenbrock_t;

    Kokkos::Extension::Vector<double, ExecutionSpace> results = GaussNewton(precision, maximum_iterations, 1, n, func, x0, xmin, xmax);

    std::cout << results << std::endl;
}
