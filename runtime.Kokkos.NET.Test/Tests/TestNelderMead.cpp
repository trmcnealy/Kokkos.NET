
#include "Tests.hpp"

#include <Measure.hpp>

using namespace NumericalMethods::Algorithms;

template<class ExecutionSpace>
extern void TestNelderMead()
{
    const int n = 2;

    const double reqmin = 1.0E-016;
    const int    konvge = 1;
    const int    kcount = 500;

    Kokkos::View<double*, ExecutionSpace> step("step", n);
    step[0] = 0.1;
    step[1] = 0.1;

    Kokkos::View<double*, ExecutionSpace> x0("x0", n);
    x0[0] = -3.0;
    x0[1] = 2.0;

    Kokkos::View<double*, ExecutionSpace> xmin("xmin", n);
    xmin[0] = -10.0;
    xmin[1] = -10.0;

    Kokkos::View<double*, ExecutionSpace> xmax("xmax", n);
    xmax[0] = 10.0;
    xmax[1] = 10.0;

    NelderMeadOptions<double> options(reqmin);//, konvge, kcount, step);

    rosenbrock<double, ExecutionSpace> func;

    typedef decltype(func) rosenbrock_t;

    //NumericalMethods::Algorithms::NelderMead<double, ExecutionSpace, rosenbrock_t> nelderMead();

    NelderMeadResults<double, ExecutionSpace> results(x0.size());

    {
        System::Measure measure("NelderMead");

        results = NelderMead<double, ExecutionSpace, rosenbrock_t>::DeviceSolve(options, 1, func, x0, xmin, xmax);
    }

    std::cout << "ICount:" << results.ICount << std::endl;
    std::cout << "NumRes:" << results.NumRes << std::endl;
    std::cout << "IFault:" << results.IFault << std::endl;
    std::cout << "YNewLo:" << results.YNewLo << std::endl;
    std::cout << "XMin(0):" << results.XMin(0) << std::endl;
    std::cout << "XMin(1):" << results.XMin(1) << std::endl;
}

template __declspec(dllexport) void TestNelderMead<EXECUTION_SPACE>();