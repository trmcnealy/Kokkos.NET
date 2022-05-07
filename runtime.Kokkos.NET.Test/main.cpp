//-Bdynamic, $(PACKAGE_CACHE)../kokkos/lib/libkokkoscore.dll.a,$(PACKAGE_CACHE)../kokkos/lib/libkokkoscontainers.dll.a, $(PACKAGE_CACHE)../kokkos/lib/libkokkoskernels.a

//-Bdynamic,-lkokkoscore.dll,-lkokkoscontainers.dll,-lkokkoskernels.dll,-lkokkoskernels.dll,-Bstatic,-lteuchoscore
//-LC:/AssemblyCache/kokkos/lib,-LC:/AssemblyCache/Trilinos/lib

#include <Types.hpp>

#include "Tests/Tests.hpp"

//#include <Solvers/RungeKuttaMethods.hpp>


//#include <runtime.Kokkos/ViewTypes.hpp>
//#include <runtime.Kokkos/Extensions.hpp>
//#include <Geometry/Delaunay.hpp>
//#include "Tpetra_TestingUtilities.hpp"

//#include <Tpetra_Core.hpp>
//#include <Tpetra_MultiVector.hpp>
//#include <Tpetra_Operator.hpp>
//#include <Tpetra_Vector.hpp>
//#include <Kokkos_ArithTraits.hpp>
//#include <KokkosBlas.hpp>

//#include <Tpetra_MultiVector_fwd.hpp>
//#include <Tpetra_Vector_fwd.hpp>
//#include <Tpetra_FEMultiVector_fwd.hpp>
//#include <Tpetra_DistObject.hpp>
//#include <Tpetra_Map_fwd.hpp>
//#include <Tpetra_Details_Behavior.hpp>
//#include <Kokkos_DualView.hpp>
//#include <Teuchos_BLAS_types.hpp>
//#include <Teuchos_DataAccess.hpp>
//#include <Teuchos_Range1D.hpp>
//#include <Kokkos_ArithTraits.hpp>
//#include <Kokkos_InnerProductSpaceTraits.hpp>
//#include <Tpetra_KokkosRefactor_Details_MultiVectorLocalDeepCopy.hpp>
//#include <Tpetra_Access.hpp>

#include <iostream>
#include <vector>
#include <iterator>
#include <algorithm>
#include <array>
#include <random>
#include <chrono>

#include "_LinkLibraries.hpp"

//link_trilinos






int main(int argc, char** argv)
{
    std::cout << argv[0] << std::endl;

    const int32 num_threads      = 64;
    const int32 num_numa         = 1;
    const int32 device_id        = 0;
    const int32 ndevices         = 1;
    const int32 skip_device      = 9999;
    const bool  disable_warnings = false;

    Kokkos::InitArguments arguments{};
    arguments.num_threads      = num_threads;
    arguments.num_numa         = num_numa;
    arguments.device_id        = device_id;
    arguments.ndevices         = ndevices;
    arguments.skip_device      = skip_device;
    arguments.disable_warnings = disable_warnings;

    Kokkos::ScopeGuard kokkos(arguments);
    {
        //TestCombinations2<EXECUTION_SPACE>();
    
        TestNelderMead<EXECUTION_SPACE>();
    }

    std::cout << "Press any key to exit." << std::endl;
    getchar();

    return 0;
}

//// const int32 num_threads      = 64;
//// const int32 num_numa         = 1;
//// const int32 device_id        = 0;
//// const int32 ndevices         = 1;
//// const int32 skip_device      = 9999;
//// const bool  disable_warnings = false;

//// Kokkos::InitArguments arguments{};
//// arguments.num_threads      = num_threads;
//// arguments.num_numa         = num_numa;
//// arguments.device_id        = device_id;
//// arguments.ndevices         = ndevices;
//// arguments.skip_device      = skip_device;
//// arguments.disable_warnings = disable_warnings;

//// Kokkos::ScopeGuard kokkos(arguments);
////{
////    Kokkos::View<Point<fp64, 2>*, Kokkos::Cuda::array_layout, Kokkos::Cuda> vertices("v", 12);

////    uint32 i = 0;

////    vertices(i++) = (Point<fp64, 2>*)Kokkos::kokkos_malloc<Kokkos::Cuda>(sizeof(Point<fp64, 2>));

////    vertices(i++) = Teuchos::rcp(new Point<fp64, 2>(7.0, 3.0));
////    vertices(i++) = Teuchos::rcp(new Point<fp64, 2>(4.0, 7.0));
////    vertices(i++) = Teuchos::rcp(new Point<fp64, 2>(5.0, 13.0));
////    vertices(i++) = Teuchos::rcp(new Point<fp64, 2>(2.0, 7.0));
////    vertices(i++) = Teuchos::rcp(new Point<fp64, 2>(6.0, 9.0));
////    vertices(i++) = Teuchos::rcp(new Point<fp64, 2>(12.0, 8.0));
////    vertices(i++) = Teuchos::rcp(new Point<fp64, 2>(3.0, 4.0));
////    vertices(i++) = Teuchos::rcp(new Point<fp64, 2>(6.0, 6.0));
////    vertices(i++) = Teuchos::rcp(new Point<fp64, 2>(3.0, 10.0));
////    vertices(i++) = Teuchos::rcp(new Point<fp64, 2>(8.0, 7.0));
////    vertices(i++) = Teuchos::rcp(new Point<fp64, 2>(5.0, 13.0));
////    vertices(i++) = Teuchos::rcp(new Point<fp64, 2>(10.0, 6.0));

////    Kokkos::View<Triangle<fp64, 2>*, Kokkos::Cuda::array_layout, Kokkos::Cuda> triangles = Triangulate<fp64, Kokkos::Cuda, 2>(vertices);

////    for (uint32 j = 0; j <= triangles.extent(0); ++j)
////    {
////        std::cout << triangles(j) << std::endl;
////    }
////}

// int numberPoints = 40;

// std::default_random_engine             eng(std::random_device{}());
// std::uniform_real_distribution<double> dist_w(0, 800);
// std::uniform_real_distribution<double> dist_h(0, 600);

// std::cout << "Generating " << numberPoints << " random points" << std::endl;

// std::vector<Vector<double, 2>> points;
// for (int i = 0; i < numberPoints; ++i)
//{
//    points.push_back(Vector<double, 2>{dist_w(eng), dist_h(eng)});
//}

// Delaunay<double>                       triangulation;
// const auto                             start     = std::chrono::high_resolution_clock::now();
// const std::vector<Triangle<double, 2>> triangles = triangulation.triangulate(points);
// const auto                             end       = std::chrono::high_resolution_clock::now();
// const std::chrono::duration<double>    diff      = end - start;

// std::cout << triangles.size() << " triangles generated in " << diff.count() << "s\n";
// const std::vector<Segment<double, 2>> edges = triangulation.getEdges();

// uint32 i    = 0;
// values(i++) = 0.21354;
// values(i++) = 1.21354;
// values(i++) = 2.21354;
// values(i++) = 3.21354;
// values(i++) = 4.21354;
// values(i++) = 5.21354;
// values(i++) = 6.21354;
// values(i++) = 7.21354;
// values(i++) = 8.21354;
// values(i) = 9.21354;

// bool found_value0 = Kokkos::Extension::Contains<fp64, uint32, Kokkos::Cuda>(values, 8.21354);

// std::cout << "found_value:" << found_value0 << ":shoudl be 1" << std::endl;

// bool found_value1 = Kokkos::Extension::Contains<fp64, uint32, Kokkos::Cuda>(values, 18.21354);

// std::cout << "found_value:" << found_value1 << ":shoudl be 0"<< std::endl;

// template<FloatingPoint DataType>
// static constexpr DataType y1x(const DataType t)
//{
//    const DataType value = Default<DataType>(20.0) / (Default<DataType>(1.0) + Default<DataType>(19.0) * System::exp(Default<DataType>(-0.25) * t));
//
//    return value;
//}
//
// template<FloatingPoint DataType, class ExecutionSpace, typename LayoutType = typename ExecutionSpace::array_layout>
// void f1(const DataType t, const Kokkos::View<DataType*, LayoutType, ExecutionSpace>& y, const Kokkos::View<DataType*, LayoutType, ExecutionSpace>& yp)
//{
//    yp[0] = Default<DataType>(0.25) * y[0] * (Default<DataType>(1.0) - y[0] / Default<DataType>(20.0));
//}
//
// template<FloatingPoint DataType, class ExecutionSpace, typename LayoutType = typename ExecutionSpace::array_layout>
// void test01()
//{
//    std::cout << "\n";
//    std::cout << "TEST01\n";
//    std::cout << "  Solve a scalar equation using RKF:\n";
//    std::cout << "\n";
//    std::cout << "  Y' = 0.25 * Y * ( 1 - Y / 20 )\n";
//    std::cout << "\n";
//
//    int neqn = 1;
//
//    Kokkos::View<DataType*, LayoutType, ExecutionSpace> y("y", neqn);
//    Kokkos::View<DataType*, LayoutType, ExecutionSpace> yp("yp", neqn);
//
//    DataType abserr = System::sqrt(Constants<DataType>::Epsilon());
//    DataType relerr = System::sqrt(Constants<DataType>::Epsilon());
//
//    int flag = 1;
//
//    DataType t_start = Default<DataType>(0.0);
//    DataType t_stop  = Default<DataType>(20.0);
//
//    int n_step = 5;
//
//    DataType t     = Default<DataType>(0.0);
//    DataType t_out = Default<DataType>(0.0);
//
//    y[0] = Default<DataType>(1.0);
//
//    f1(t, y, yp);
//
//    std::cout << "\n";
//    std::cout << "FLAG             T          Y         Y'          Y_Exact         Error\n";
//    std::cout << "\n";
//
//    std::cout << std::setw(4) << flag << "  " << std::setw(12) << t << "  " << std::setw(12) << y[0] << "  " << std::setw(12) << yp[0] << "  " << std::setw(12) << y1x(t) << "  " << std::setw(12) << y[0] - y1x(t) << "\n";
//
//    for (int i_step = 1; i_step <= n_step; i_step++)
//    {
//        t = ((DataType)(n_step - i_step + 1) * t_start + (DataType)(i_step - 1) * t_stop) / (DataType)(n_step);
//
//        t_out = ((DataType)(n_step - i_step) * t_start + (DataType)(i_step)*t_stop) / (DataType)(n_step);
//
//        flag = NumericalMethods::Solvers::rkf45<DataType, ExecutionSpace>(&f1<DataType, ExecutionSpace>, neqn, y, yp, &t, t_out, &relerr, abserr, flag);
//
//        std::cout << std::setw(4) << flag << "  " << std::setw(12) << t << "  " << std::setw(12) << y[0] << "  " << std::setw(12) << yp[0] << "  " << std::setw(12) << y1x(t) << "  " << std::setw(12) << y[0] - y1x(t) << "\n";
//    }
//}
//
// template<FloatingPoint DataType, class ExecutionSpace, typename LayoutType = typename ExecutionSpace::array_layout>
// static void f2(const DataType t, const Kokkos::View<DataType*, LayoutType, ExecutionSpace>& y, const Kokkos::View<DataType*, LayoutType, ExecutionSpace>& yp)
//{
//    yp[0] = y[1];
//    yp[1] = -y[0];
//}
//
// template<FloatingPoint DataType, class ExecutionSpace, typename LayoutType = typename ExecutionSpace::array_layout>
// void test02()
//{
//    int32    flag;
//    int32    n_step;
//    int32    neqn;
//    DataType abserr;
//    DataType relerr;
//    DataType t;
//    DataType t_out;
//    DataType t_start;
//    DataType t_stop;
//
//    std::cout << "\n";
//    std::cout << "TEST02\n";
//    std::cout << "  Solve a vector equation using RKF:\n";
//    std::cout << "\n";
//    std::cout << "  Y'(1) =  Y(2)\n";
//    std::cout << "  Y'(2) = -Y(1)\n";
//    std::cout << "\n";
//    std::cout << "\n";
//    std::cout << "  This system is equivalent to the following\n";
//    std::cout << "  second order system:\n";
//    std::cout << "\n";
//    std::cout << "  Z\" = - Z.\n";
//
//    neqn = 2;
//
//    Kokkos::View<DataType*, LayoutType, ExecutionSpace> y("y", neqn);
//    Kokkos::View<DataType*, LayoutType, ExecutionSpace> yp("yp", neqn);
//
//    abserr = System::sqrt(Constants<DataType>::Epsilon());
//    relerr = System::sqrt(Constants<DataType>::Epsilon());
//
//    flag = 1;
//
//    t_start = 0.0;
//    t_stop  = 2.0 * 3.14159265;
//
//    n_step = 12;
//
//    t     = 0.0;
//    t_out = 0.0;
//
//    y[0] = 1.0;
//    y[1] = 0.0;
//
//    std::cout << "\n";
//    std::cout << "FLAG             T          Y(1)       Y(2)\n";
//    std::cout << "\n";
//
//    std::cout << std::setw(4) << flag << "  " << std::setw(12) << t << "  " << std::setw(12) << y[0] << "  " << std::setw(12) << y[1] << "\n";
//
//    for (int32 i_step = 1; i_step <= n_step; i_step++)
//    {
//        t = ((DataType)(n_step - i_step + 1) * t_start + (DataType)(i_step - 1) * t_stop) / (DataType)(n_step);
//
//        t_out = ((DataType)(n_step - i_step) * t_start + (DataType)(i_step)*t_stop) / (DataType)(n_step);
//
//        flag = NumericalMethods::Solvers::rkf45<DataType, ExecutionSpace>(&f2<DataType, ExecutionSpace>, neqn, y, yp, &t, t_out, &relerr, abserr, flag);
//
//        std::cout << std::setw(4) << flag << "  " << std::setw(12) << t << "  " << std::setw(12) << y[0] << "  " << std::setw(12) << y[1] << "\n";
//    }
//}
