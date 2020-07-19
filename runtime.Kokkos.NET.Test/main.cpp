
#include <iostream>
#include <runtime.Kokkos/Extensions.hpp>

using namespace Kokkos::Extension;
using namespace Kokkos::LinearAlgebra;

template<typename DataType, class ExecutionSpace>
void test1()
{
    Matrix<DataType, ExecutionSpace> A("A", 3, 3);

    A(0, 0) = 36;
    A(1, 0) = 30;
    A(2, 0) = 18;

    A(0, 1) = 30;
    A(1, 1) = 41;
    A(2, 1) = 23;

    A(0, 2) = 18;
    A(1, 2) = 23;
    A(2, 2) = 14;

    Vector<DataType, ExecutionSpace> b("b", 3);

    b(0) = 288;
    b(1) = 296;
    b(2) = 173;

    Vector<DataType, ExecutionSpace> x = Cholesky<DataType, ExecutionSpace>(A, b);

    for(size_type i = 0; i < x.extent(0); i++)
    {
        std::cout << x(i) << std::endl;
    }

    std::cout << std::endl;

    x(0) = 5;
    x(1) = 3;
    x(2) = 1;

    Vector<DataType, ExecutionSpace> b2 = A * x;

    for(size_type i = 0; i < x.extent(0); i++)
    {
        std::cout << b2(i) << std::endl;
    }

    std::cout << std::endl;
}

template<typename DataType, class ExecutionSpace>
void test2()
{
    Matrix<DataType, ExecutionSpace> A("A", 2, 2);

    A(0, 0) = -1;
    A(0, 1) = 4;

    A(1, 0) = 2;
    A(1, 1) = 3;

    Matrix<DataType, ExecutionSpace> X("X", 2, 2);

    X(0, 0) = 9;
    X(0, 1) = -3;

    X(1, 0) = 6;
    X(1, 1) = 1;

    Matrix<DataType, ExecutionSpace> B = A * X;

    for(size_type i = 0; i < B.extent(0); i++)
    {
        for(size_type j = 0; j < B.extent(1); j++)
        {
            std::cout << B(i, j) << std::endl;
        }
    }

    std::cout << std::endl;
}

template<typename DataType, class ExecutionSpace>
void test3()
{
    // Matrix<DataType, ExecutionSpace> A("A", 2, 2);

    // A(0, 0) = -1;
    // A(0, 1) = 4;

    // A(1, 0) = 2;
    // A(1, 1) = 3;

    // Matrix<DataType, ExecutionSpace> B("B", 2, 2);

    // B(0, 0) = 9;
    // B(0, 1) = -3;

    // B(1, 0) = 6;
    // B(1, 1) = 1;

    Matrix<DataType, ExecutionSpace> I = Identity<DataType, ExecutionSpace>(2, 2);

    // Matrix<DataType, ExecutionSpace> X = A / B;

    for(size_type i = 0; i < I.extent(0); i++)
    {
        for(size_type j = 0; j < I.extent(1); j++)
        {
            std::cout << I(i, j) << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;

    Vector<DataType, ExecutionSpace> r0 = row(I, 0);

    for(size_type i = 0; i < r0.extent(0); i++)
    {
        std::cout << r0(i) << std::endl;
    }

    std::cout << std::endl;

    auto c0 = column(I, 0);

    for(size_type i = 0; i < c0.extent(0); i++)
    {
        std::cout << c0(i) << std::endl;
    }

    std::cout << std::endl;
}

#include <Analyzes/kNearestNeighbor.hpp>

template<typename DataType, class ExecutionSpace>
void test4()
{
    Kokkos::View<DataType* [5], typename ExecutionSpace::array_layout, ExecutionSpace> dataset("data", 10000);

    for(size_type i = 0; i < dataset.extent(0); i++)
    {
        dataset(i, 0) = 1000.0 * ((DataType)std::rand() / (DataType)RAND_MAX);
        dataset(i, 1) = 100.0 * ((DataType)std::rand() / (DataType)RAND_MAX);
        dataset(i, 2) = 0.001 * ((DataType)std::rand() / (DataType)RAND_MAX);
        dataset(i, 3) = 0.0001 * ((DataType)std::rand() / (DataType)RAND_MAX);
        dataset(i, 4) = ((DataType)std::rand() / (DataType)RAND_MAX);
    }

    for(size_type i = 0; i < min<size_type>(100, dataset.extent(0)); i++)
    {
        std::cout << dataset(i, 0) << std::endl;
    }
    std::cout << std::endl;

    Kokkos::View<DataType**, typename ExecutionSpace::array_layout, ExecutionSpace> distances = kNearestNeighbor<DataType, ExecutionSpace, 5>(1, dataset);

    for(size_type i = 0; i < min<size_type>(100, distances.extent(0)); i++)
    {
        for(size_type j = 0; j < min<size_type>(100, distances.extent(1)); j++)
        {
            std::cout << distances(i, j) << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

int main(int argc, char* argv[])
{
    Kokkos::initialize(argc, argv);

    // test4<double, Kokkos::Serial>();

    test4<double, Kokkos::OpenMP>();

    test4<double, Kokkos::Cuda>();

    Kokkos::finalize_all();

    std::cout << "Press any key to exit." << std::endl;
    getchar();

    return 0;
}
