
#include "Tests.hpp"

template<class ExecutionSpace>
static void TestPInv()
{

    const uint32 M = 2;
    const uint32 N = 3;

    const double rcond = 1E-15;

    Kokkos::Extension::Matrix<double, ExecutionSpace> A("A", M, N);

    A(0, 0) = 1.0;
    A(0, 1) = 3.0;
    A(0, 2) = 5.0;

    A(1, 0) = 2.0;
    A(1, 1) = 4.0;
    A(1, 2) = 6.0;

    std::cout << A << std::endl;

    const Kokkos::Extension::Matrix<double, ExecutionSpace> A_pinv = pinverse(A, rcond);

    std::cout << A_pinv << std::endl;
}
