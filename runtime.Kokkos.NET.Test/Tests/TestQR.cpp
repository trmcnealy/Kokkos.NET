
#include "Tests.hpp"

template<class ExecutionSpace>
static void TestQR()
{
    using Matrix = Kokkos::Extension::Matrix<double, ExecutionSpace>;
    using Vector = Kokkos::Extension::Vector<double, ExecutionSpace>;

    Matrix A("A", 3, 5);

    A(0, 0) = 0.41;
    A(1, 0) = 0.94;
    A(2, 0) = 0.92;

    A(0, 1) = 0.41;
    A(1, 1) = 0.89;
    A(2, 1) = 0.06;

    A(0, 2) = 0.35;
    A(1, 2) = 0.81;
    A(2, 2) = 0.01;

    A(0, 3) = 0.14;
    A(1, 3) = 0.20;
    A(2, 3) = 0.20;

    A(0, 4) = 0.60;
    A(1, 4) = 0.27;
    A(2, 4) = 0.20;

    // Matrix A("A", 5, 3);

    // A(0, 0) = 0.41;
    // A(1, 0) = 0.41;
    // A(2, 0) = 0.35;
    // A(3, 0) = 0.14;
    // A(4, 0) = 0.60;

    // A(0, 1) = 0.94;
    // A(1, 1) = 0.89;
    // A(2, 1) = 0.81;
    // A(3, 1) = 0.20;
    // A(4, 1) = 0.27;

    // A(0, 2) = 0.92;
    // A(1, 2) = 0.06;
    // A(2, 2) = 0.01;
    // A(3, 2) = 0.20;
    // A(4, 2) = 0.20;

    std::cout << A << std::endl;

    NumericalMethods::Algebra::QRDecomposition<double, ExecutionSpace> qr(A);

    std::cout << qr.qt << std::endl;
    std::cout << qr.r << std::endl;
}
