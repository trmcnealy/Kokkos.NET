
#include "Tests.hpp"


template<class ExecutionSpace>
static void TestLU()
{
    using Matrix = Kokkos::Extension::Matrix<double, ExecutionSpace>;

    Matrix A("A", 3, 3);

    A(0, 0) = 2.0;
    A(1, 0) = -5.0;
    A(2, 0) = 1.0;

    A(0, 1) = -1.0;
    A(1, 1) = 3.0;
    A(2, 1) = -1.0;

    A(0, 2) = 3.0;
    A(1, 2) = 4.0;
    A(2, 2) = 2.0;

    std::cout << A << std::endl;

    NumericalMethods::Algebra::LUD<double, ExecutionSpace> lud(A);

    std::cout << lud.lu << std::endl;

    Matrix invA("invA", 3, 3);

    lud.Inverse(invA);

    std::cout << invA << std::endl;

    Matrix identity = A * invA;

    std::cout << identity << std::endl;
}


template __declspec(dllexport) void TestLU<EXECUTION_SPACE>();
//template __declspec(dllexport) void TestLU<Kokkos::Cuda>();
//template __declspec(dllexport) void TestLU<Kokkos::OpenMP>();
//template __declspec(dllexport) void TestLU<Kokkos::Serial>();
