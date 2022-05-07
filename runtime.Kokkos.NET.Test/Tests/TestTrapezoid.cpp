
#include "Tests.hpp"

template<class ExecutionSpace>
static void TestTrapezoid()
{
    Vector<double, ExecutionSpace> x("x", 5);
    x(0) = 0;
    x(1) = 2;
    x(2) = 4;
    x(3) = 6;
    x(4) = 8;
    Vector<double, ExecutionSpace> y("y", 5);
    y(0) = 3;
    y(1) = 7;
    y(2) = 11;
    y(3) = 9;
    y(4) = 3;

    double result = NumericalMethods::Calculus::trapezoid(x, y);

    std::cout << result << std::endl;
}
