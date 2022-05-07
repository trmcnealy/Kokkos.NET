
#include "Tests.hpp"

template<class ExecutionSpace>
static void TestSVD()
{
    using Matrix = Kokkos::Extension::Matrix<double, ExecutionSpace>;
    using Vector = Kokkos::Extension::Vector<double, ExecutionSpace>;

    Matrix A("A", 5, 3);

    // A(0, 0) = 0.41;
    // A(1, 0) = 0.94;
    // A(2, 0) = 0.92;

    // A(0, 1) = 0.41;
    // A(1, 1) = 0.89;
    // A(2, 1) = 0.06;

    // A(0, 2) = 0.35;
    // A(1, 2) = 0.81;
    // A(2, 2) = 0.01;

    // A(0, 3) = 0.14;
    // A(1, 3) = 0.20;
    // A(2, 3) = 0.20;

    // A(0, 4) = 0.60;
    // A(1, 4) = 0.27;
    // A(2, 4) = 0.20;

    A(0, 0) = 0.41;
    A(1, 0) = 0.41;
    A(2, 0) = 0.35;
    A(3, 0) = 0.14;
    A(4, 0) = 0.60;

    A(0, 1) = 0.94;
    A(1, 1) = 0.89;
    A(2, 1) = 0.81;
    A(3, 1) = 0.20;
    A(4, 1) = 0.27;

    A(0, 2) = 0.92;
    A(1, 2) = 0.06;
    A(2, 2) = 0.01;
    A(3, 2) = 0.20;
    A(4, 2) = 0.20;

    std::cout << A << std::endl;

    NumericalMethods::Algebra::SVD<double, ExecutionSpace> svd(A);

    std::cout << svd.u << std::endl;
    std::cout << svd.v << std::endl;
    std::cout << svd.w << std::endl;

    Matrix C("C", svd.u.ncolumns(), svd.u.ncolumns());

    KokkosBlas::gemm("T", "N", 1.0, svd.u.View(), svd.u.View(), 1.0, C.View());

    std::cout << C << std::endl;

    Matrix Cv("Cv", svd.v.nrows(), svd.v.nrows());

    KokkosBlas::gemm("T", "N", 1.0, svd.v.View(), svd.v.View(), 1.0, Cv.View());

    std::cout << Cv << std::endl;
}
