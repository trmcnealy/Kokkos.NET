//-Bdynamic, $(PACKAGE_CACHE)../kokkos/lib/libkokkoscore.dll.a,$(PACKAGE_CACHE)../kokkos/lib/libkokkoscontainers.dll.a, $(PACKAGE_CACHE)../kokkos/lib/libkokkoskernels.a

//-Bdynamic,-lkokkoscore.dll,-lkokkoscontainers.dll,-lkokkoskernels.dll,-lkokkoskernels.dll,-Bstatic,-lteuchoscore
//-LC:/AssemblyCache/kokkos/lib,-LC:/AssemblyCache/Trilinos/lib

#pragma clang diagnostic ignored "-Wimplicit-const-int-float-conversion"

#include <Types.hpp>

#include <runtime.Kokkos/ViewTypes.hpp>
#include <runtime.Kokkos/Extensions.hpp>

//#include <Sacado.hpp>

//#include <Fraction.hpp>

#include <ValueLimits.hpp>
//#include <Algorithms/Optimization/LevMarFit.hpp>
#include <Algorithms/Smoothing.hpp>

#include <Algebra/LU.hpp>
#include <Algebra/SVD.hpp>
#include <Algebra/QR.hpp>

#include <Calculus/Integration.hpp>
//#include <Array.hpp>
//#include <Statistics/Random.hpp>
#include <Combinations.hpp>
#include <Analyze/PrincipalComponentsAnalysis.hpp>
//#include <Algorithms/Optimization/LevenbergMarquardt.hpp>
#include <Algorithms/Optimization/Newton.hpp>
#include <Algorithms/Optimization/GlobalNewton.hpp>
#include <Algorithms/Optimization/NelderMeadOptimizer.hpp>
#include <Solvers/RungeKuttaMethods.hpp>

//#include <Kokkos_UnorderedMap.hpp>

#include <KokkosBlas.hpp>
#include <KokkosSparse_CrsMatrix.hpp>
#include <KokkosSparse_spmv.hpp>

#include <cstring>
#include <iostream>

//#include <Teuchos_RCP.hpp>

// using namespace std::string_literals;
//
//
// namespace Internal
//{
//    constexpr char forwardslash = '/';
//    constexpr char backslash    = '\\';
//}

// MSBuild runtime.Kokkos.NET.vcxproj -p:Configuration=Debug;Platform=x64 -bl:output.binlog

// template<typename DataType, class ExecutionSpace>
// static void test4();
//
// template<typename DataType, class ExecutionSpace, unsigned Size>
// static void test6();
//
// template<typename DataType, class ExecutionSpace>
// static void test();

// using AtomicTrait       = Kokkos::MemoryTraits<Kokkos::Atomic>;
// using RandomAccessTrait = Kokkos::MemoryTraits<Kokkos::RandomAccess>;
//
using EXECUTION_SPACE = Kokkos::Cuda;
//using EXECUTION_SPACE = Kokkos::OpenMP;
// using EXECUTION_SPACE = Kokkos::Serial;
//// using Array            = System::Array<double, EXECUTION_SPACE>;
// using LayoutType = EXECUTION_SPACE::array_layout;
// using AtomicView       = Kokkos::View<double*, LayoutType, EXECUTION_SPACE, AtomicTrait>;
// using RandomAccessView = Kokkos::View<double*, LayoutType, EXECUTION_SPACE, RandomAccessTrait>;
// using View             = Kokkos::View<double*, LayoutType, EXECUTION_SPACE>;
//// using StringDictionary = NumericalMethods::DataStorage::Dictionary<std::string, Array*, EXECUTION_SPACE>;
//// using ViewDictionary   = NumericalMethods::DataStorage::Dictionary<std::string, AtomicView, EXECUTION_SPACE>;
//
// typedef Kokkos::UnorderedMap<std::string, size_type, EXECUTION_SPACE> map_type;

// template<typename DataType>
// KOKKOS_INLINE_FUNCTION DataType rosenbrock_function(const uint32 equation_index, const Kokkos::View<DataType*, EXECUTION_SPACE>& x, const uint32 ts)
//{
//    const DataType fx1 = x[1] - x[0] * x[0];
//    const DataType fx2 = 1.0 - x[0];
//
//    const DataType fx = 100.0 * fx1 * fx1 + fx2 * fx2;
//
//    return fx;
//}

template<typename DataType, class ExecutionSpace>
struct rosenbrock
{
    KOKKOS_INLINE_FUNCTION DataType operator()(const Kokkos::View<DataType*, ExecutionSpace>& x) const
    {
        const DataType fx1 = x[1] - (x[0] * x[0]);
        const DataType fx2 = 1.0 - x[0];

        const DataType fx = (100.0 * (fx1 * fx1)) + (fx2 * fx2);

        return fx;
    }

    KOKKOS_INLINE_FUNCTION DataType operator()(const Kokkos::View<DataType*, ExecutionSpace>& x, const uint32 equation_index) const
    {
        const DataType fx1 = x[1] - (x[0] * x[0]);
        const DataType fx2 = 1.0 - x[0];

        const DataType fx = (100.0 * (fx1 * fx1)) + (fx2 * fx2);

        return fx;
    }

    KOKKOS_INLINE_FUNCTION DataType operator()(const NumericalMethods::Algorithms::GaussNewtonJacobian&, const Kokkos::View<DataType*, ExecutionSpace>& x, const uint32 equation_index) const
    {
        if (equation_index == 0)
        {
            return 4.0 * 100.0 * (std::pow(x[0], 2) - x[1]) * (x[0] + 2.0 * (x[0] - 1.0));
        }

        return -2.0 * 100.0 * (std::pow(x[0], 2) - x[1]);
    }

    KOKKOS_INLINE_FUNCTION DataType operator()(const Kokkos::View<DataType*, ExecutionSpace>& x, const uint32 equation_index, const uint32 ts) const
    {
        const DataType fx1 = x[1] - (x[0] * x[0]);
        const DataType fx2 = 1.0 - x[0];

        const DataType fx = (100.0 * (fx1 * fx1)) + (fx2 * fx2);

        return fx;
    }
};

template<typename DataType, class ExecutionSpace>
struct sin_func
{
    KOKKOS_INLINE_FUNCTION DataType operator()(const Kokkos::View<DataType*, ExecutionSpace>& x) const
    {
        return std::sin(x[0]);
    }
};

template<typename DataType>
static DataType func(const DataType& x)
{
    DataType r = std::sin(x);
    return r;
}

template<typename DataType>
static DataType func_dx(const DataType& x)
{
    DataType r = std::cos(x);
    return r;
}

// using namespace Kokkos;

using namespace NumericalMethods::Algorithms;

template<class ExecutionSpace>
static void TestGramSchmidt();
template<class ExecutionSpace>
static void TestPCA();
template<class ExecutionSpace>
static void TestCombinations2();
template<class ExecutionSpace>
static void TestNelderMead();
template<class ExecutionSpace>
static void TestGaussNewton();
template<class ExecutionSpace>
static void TestPInv();

template<FloatingPoint DataType,
         class ExecutionSpace,
         typename Ordinal    = size_type,
         typename Offset     = size_type,
         typename DeviceType = Kokkos::Device<ExecutionSpace, typename ExecutionSpace::memory_space>>
using SparseMatrixType = KokkosSparse::CrsMatrix<DataType, Ordinal, DeviceType, void, Offset>;

int main(int argc, char** argv)
{

    std::cout << argv[0] << std::endl;

    //// void*           ptr;
    //// const size_type arg_alloc_size = 80;
    //// cudaError_t err = cudaMallocManaged(&ptr, arg_alloc_size, cudaMemAttachGlobal);

    const int  num_threads      = 64;
    const int  num_numa         = 1;
    const int  device_id        = 0;
    const int  ndevices         = 1;
    const int  skip_device      = 9999;
    const bool disable_warnings = false;

    Kokkos::InitArguments arguments{};
    arguments.num_threads      = num_threads;
    arguments.num_numa         = num_numa;
    arguments.device_id        = device_id;
    arguments.ndevices         = ndevices;
    arguments.skip_device      = skip_device;
    arguments.disable_warnings = disable_warnings;

    

    Kokkos::ScopeGuard kokkos(arguments);
    {
        //Kokkos::print_configuration(std::cout, true);

        // std::cout << Kokkos::hwloc::get_available_numa_count() << std::endl;
        // std::cout << Kokkos::hwloc::get_available_cores_per_numa() << std::endl;
        // std::cout << Kokkos::hwloc::get_available_threads_per_core() << std::endl;

        // SYSTEM_INFO sysinfo;
        // GetSystemInfo(&sysinfo);
        // std::cout << sysinfo.dwNumberOfProcessors << std::endl;

        // static const int FadStride = std::is_same_v<ExecutionSpace, Kokkos::Cuda> ? 32 : 1;

        // const int num_deriv = 1;

        //// typedef Sacado::Fad::DFad<double> FadType;
        // typedef Sacado::Fad::SFad<double, num_deriv>                        FadType;
        // typedef Sacado::Fad::ViewFad<double, num_deriv, FadStride, FadType> ViewFadType;
        //// typedef Sacado::Fad::SFad<FadType, 1>                               HessianType;
        //// typedef HessianType                                                 ScalarT;

        // typedef Kokkos::LayoutContiguous<ExecutionSpace::array_layout, FadStride> ContLayout;
        // typedef Kokkos::View<FadType*, ExecutionSpace>                            ViewType;

        // Kokkos::View<FadType*> v("x", 2);

        //// auto seed_second_deriv = [](int num_vars, int index, double xi, double vi) -> HessianType
        ////{
        ////    typedef HessianType SecondFadType;

        ////    SecondFadType x   = SecondFadType(1, FadType(num_vars, index, xi));
        ////    x.fastAccessDx(0) = vi;

        ////    return x;
        ////};

        //// double x_val  = 0.25;
        //// double dx_val = 2.0;

        //// for (int i = 0; i < 2; ++i)
        ////{
        ////    x(i) = seed_second_deriv(2, 0, x_val, dx_val);
        ////}

        //// Kokkos::View<double**, ExecutionSpace> v("v", 2, num_deriv);

        //// ViewType v(Kokkos::View<double**, ExecutionSpace>("", 2, 2).data(), 2);

        //// Kokkos::deep_copy(v, 0.0);

        // v(0) = FadType(1, 1.25 / 4.0);
        // v(1) = FadType(1, 2.0);

        // v(0).fastAccessDx(0) = 1.0;
        // //v(1).fastAccessDx(1) = 1.0;

        //// ViewFadType afad((double*)Kokkos::kokkos_malloc<ExecutionSpace>(sizeof(double) * 2 * num_deriv), num_deriv);

        //// afad(0).val() = 1.25 / 4.0;
        //// afad(1).val() = 2.0;

        // FadType rfad = func<FadType>(v(0));

        // std::cout << rfad.val() << std::endl;
        // std::cout << rfad.fastAccessDx(0) << std::endl;

        // std::cout << func_dx(1.25 / 4.0) << std::endl;

        //// using Matrix = Kokkos::Extension::Matrix<double, ExecutionSpace>;
        //// using Vector = Kokkos::Extension::Vector<double, ExecutionSpace>;

        //// Vector x("x", 1);
        //// x[0] = 1.0;

        //// Vector jacobian("jacobian", 1);

        //// sin_func<double> sin_f;

        //// typedef decltype(sin_f) sin_t;

        //// NumericalMethods::Calculus::Derivative<NumericalMethods::Calculus::DerivativeType::CentralDifference>::dFdx(sin_f, jacobian, x);

        //// std::cout << jacobian << std::endl;

        //// std::cout << std::cos(x[0]) << std::endl;

        //// const uint32 M = 2;
        //// const uint32 N = 2;

        //// Kokkos::Extension::Matrix<double, ExecutionSpace> A("A", M, N);

        //// A(0, 0) = 4.0;
        //// A(0, 1) = 1.0;

        //// A(1, 0) = 1.0;
        //// A(1, 1) = 3.0;

        ////
        //// std::cout << -pinverse(A) << std::endl;

        // TestPCA();
        //// TestPInv();
        // TestNelderMead();
        TestGramSchmidt<EXECUTION_SPACE>();

        //// TestCombinations2();

        ////// using Matrix = Kokkos::Extension::Matrix<double, ExecutionSpace>;
        //////
        ////// Matrix A("A", 2, 3);

        ////// A(0, 0) = 1.0;
        ////// A(0, 1) = 2.0;
        ////// A(0, 2) = 3.0;
        ////// A(1, 0) = 4.0;
        ////// A(1, 1) = 5.0;
        ////// A(1, 2) = 6.0;

        ////// Matrix B("B", 3, 2);

        ////// B(0, 0) = 10.0;
        ////// B(0, 1) = 11.0;
        ////// B(1, 0) = 20.0;
        ////// B(1, 1) = 21.0;
        ////// B(2, 0) = 30.0;
        ////// B(2, 1) = 31.0;

        ////// Matrix C = A * B;
        //////
        ////// std::cout << A << std::endl;
        ////// std::cout << B << std::endl;
        ////// std::cout << C << std::endl;

        ////// std::cout << A << std::endl;

        ////// Matrix I = inverse(A);

        ////// std::cout << I << std::endl;

        //// rosenbrock<double> rosenbrock_func;

        //// typedef decltype(rosenbrock_func) rosenbrock_t;

        //// using LevenbergMarquardt = NumericalMethods::Algorithms::GlobalNewton<double, ExecutionSpace, rosenbrock_t>;

        //// const uint32               iteration_max = 100;
        //// const double               tolerance     = 0.000000001;
        //// LevenbergMarquardt::Vector xmin("xmin", 2);
        //// xmin(0ull) = -5.0;
        //// xmin[1]    = -5.0;
        //// LevenbergMarquardt::Vector xmax("xmax", 2);
        //// xmax[0] = 5.0;
        //// xmax[1] = 5.0;
        //// LevenbergMarquardt::DataVector actual_data("", 1, 1);
        //// actual_data(0, 0) = 0.0;
        //// LevenbergMarquardt::Vector actual_time("", 1);
        //// actual_time[0] = 1.0;
        //// LevenbergMarquardt::Vector weights("", 1);
        //// weights[0] = 1.0;

        //// LevenbergMarquardt::ConstDataVector const_actual_data = actual_data;
        //// LevenbergMarquardt::ConstVector     const_actual_time = actual_time;
        //// LevenbergMarquardt::ConstVector     const_weights     = weights;

        ////// LevenbergMarquardt LM(iteration_max, tolerance, xmin, xmax, const_actual_data, const_actual_time, const_weights, rosenbrock_func);
        //// GlobalNewton LM(xmin, xmax, const_actual_data, const_actual_time, const_weights, rosenbrock_func);

        //// LevenbergMarquardt::Vector x("x", 2);
        //// x[0] = -3.0;
        //// x[1] = -3.0;

        //// std::cout << x << std::endl;

        //// LM.Solve(x);

        //// std::cout << x << std::endl;

        ////// Kokkos::Extension::Vector<double, ExecutionSpace> means("means", 2);

        ////// means = Kokkos::Extension::SumByColumn(A);

        ////// std::cout << means << std::endl;

        ////// means /= double(2);

        ////// std::cout << means << std::endl;

        ////// Kokkos::Extension::Matrix<double, ExecutionSpace> B = (1.0 / (double(2) - 1.0)) * A;

        ////// std::cout << B << std::endl;
    }

    ////// Kokkos::print_configuration(std::cout);

    ////// const std::string oilName = STRING("Oil");

    ////// std::cout << oilData(0) << std::endl;

    ////// ViewDictionary vDictionary(10);
    ////// vDictionary.Add(oilName, oilData);

    ////// std::cout << (vDictionary.ContainsKey(oilName) ? TEXT("Yes") : TEXT("No")) << std::endl;

    ////// Array* oilData =  new Array(1);
    ////// oilData->operator[](0) = 0.5416;

    ////// std::cout << oilData->operator[](0) << std::endl;

    ////// const std::string oilName = STRING("Oil");

    //// map_type map(10);

    //// const auto result = map.insert("Oil", 5);

    //// if (result.failed())
    ////{
    ////    std::cout << TEXT("insert failed") << std::endl;
    ////}

    ////
    //// View oilData("Oil", 10);

    ////

    ////// StringDictionary strDictionary(10);

    ////// strDictionary.Add(oilName, oilData);

    //// if (map.exists("Oil"))
    ////{

    ////    const uint32_t index = map.find("Oil");

    ////    const auto gid = map.key_at(index);

    ////    const auto lid = map.value_at(index);

    ////    std::cout << lid << std::endl;

    ////    oilData(lid) = 0.5416;

    ////    std::cout << oilData(lid) << std::endl;

    ////}

    //////// test4<double, Kokkos::Serial>();

    //////// test4<double, Kokkos::OpenMP>();

    //////// test4<double, ExecutionSpace>();

    ////// test<double, ExecutionSpace>();

    std::cout << "Press any key to exit." << std::endl;
    getchar();

    return 0;
}

using namespace Kokkos::Extension;

//#include <windows.h>
//
// static unsigned long CountSetBits(const unsigned long long bitMask)
//{
//    const unsigned long LSHIFT  = sizeof(unsigned long long) * 8 - 1;
//    unsigned long long  bitTest = static_cast<unsigned long long>(1) << LSHIFT;
//
//    unsigned long bitSetCount = 0;
//    for (unsigned long i = 0; i <= LSHIFT; ++i)
//    {
//        bitSetCount += ((bitMask & bitTest) ? 1 : 0);
//        bitTest /= 2;
//    }
//
//    return bitSetCount;
//}
//
// int GetProcessorInfo(unsigned* available_numa_count, unsigned* available_cores_per_numa, unsigned* available_threads_per_core)
//{
//    const unsigned nSLPI = sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION);
//
//    BOOL              done = FALSE;
//    DWORD             success;
//    PCACHE_DESCRIPTOR Cache;
//
//    PSYSTEM_LOGICAL_PROCESSOR_INFORMATION buffer = nullptr;
//    PSYSTEM_LOGICAL_PROCESSOR_INFORMATION ptr    = nullptr;
//
//    DWORD returnLength          = 0;
//    DWORD logicalProcessorCount = 0;
//    DWORD numaNodeCount         = 0;
//    DWORD processorCoreCount    = 0;
//    DWORD processorL1CacheCount = 0;
//    DWORD processorL2CacheCount = 0;
//    DWORD processorL3CacheCount = 0;
//    DWORD processorPackageCount = 0;
//    DWORD byteOffset            = 0;
//
//    while (!done)
//    {
//        success = GetLogicalProcessorInformation(buffer, &returnLength);
//
//        if (success == FALSE)
//        {
//            if (GetLastError() == ERROR_INSUFFICIENT_BUFFER)
//            {
//                if (buffer)
//                {
//                    free(buffer);
//                }
//
//                buffer = (PSYSTEM_LOGICAL_PROCESSOR_INFORMATION)malloc(returnLength);
//
//                if (buffer == nullptr)
//                {
//                    std::cerr << TEXT("\nError: Allocation failure\n");
//                    return (2);
//                }
//            }
//            else
//            {
//                std::cerr << TEXT("\nError %d\n") << GetLastError();
//                return (3);
//            }
//        }
//        else
//        {
//            done = TRUE;
//        }
//    }
//
//    ptr = buffer;
//
//    while (byteOffset + nSLPI <= returnLength)
//    {
//        switch (ptr->Relationship)
//        {
//            case RelationNumaNode:
//            {
//                // Non-NUMA systems report a single record of this type.
//                numaNodeCount++;
//                break;
//            }
//            case RelationProcessorCore:
//            {
//                processorCoreCount++;
//
//                // A hyperthreaded core supplies more than one logical processor.
//                logicalProcessorCount += CountSetBits(ptr->ProcessorMask);
//                break;
//            }
//            case RelationCache:
//            {
//                // Cache data is in ptr->Cache, one CACHE_DESCRIPTOR structure for each cache.
//                Cache = &ptr->Cache;
//
//                if (Cache->Level == 1)
//                {
//                    processorL1CacheCount++;
//                }
//                else if (Cache->Level == 2)
//                {
//                    processorL2CacheCount++;
//                }
//                else if (Cache->Level == 3)
//                {
//                    processorL3CacheCount++;
//                }
//                break;
//            }
//            case RelationProcessorPackage:
//            {
//                // Logical processors share a physical package.
//                processorPackageCount++;
//                break;
//            }
//            default:
//            {
//                std::cerr << TEXT("\nError: Unsupported LOGICAL_PROCESSOR_RELATIONSHIP value.\n");
//                break;
//            }
//        }
//
//        byteOffset += nSLPI;
//        ptr++;
//    }
//
//    *available_numa_count       = numaNodeCount;
//    *available_cores_per_numa   = processorCoreCount;
//    *available_threads_per_core = logicalProcessorCount / processorCoreCount;
//
//    return 0;
//}
//
// void GetInfo()
//{
//    unsigned available_numa_count=0;
//    unsigned available_cores_per_numa=0;
//    unsigned available_threads_per_core=0;
//
//    GetProcessorInfo(&available_numa_count, &available_cores_per_numa, &available_threads_per_core);
//
//    std::cout << "available_numa_count:" << available_numa_count << std::endl;
//    std::cout << "available_cores_per_numa:" << available_cores_per_numa << std::endl;
//    std::cout << "available_threads_per_core:" << available_threads_per_core << std::endl;
//}
//
// template<typename DataType, class ExecutionSpace, typename Layout>
//__inline static NdArray* RcpViewToNdArrayRank1(void* instance) noexcept
//{
//    typedef Kokkos::View<DataType*, Layout, ExecutionSpace> view_type;
//
//    typedef NdArrayTraits<typename view_type::traits::value_type, typename view_type::traits::execution_space, typename view_type::traits::array_layout, 1> ndarray_traits;
//
//    Teuchos::RCP<view_type>* view = reinterpret_cast<Teuchos::RCP<view_type>*>(instance);
//
//    NdArray* ndArray = new NdArray(ndarray_traits::data_type,
//                                   1,
//                                   ndarray_traits::layout,
//                                   ndarray_traits::execution_space,
//                                   (*view)->data(),
//                                   NativeString((*view)->label().size(), (*view)->label().c_str()));
//
//    ndArray->dims[0] = (*view)->extent(0);
//
//    ndArray->strides[0] = (*view)->stride(0);
//
//    return ndArray;
//}
//
// static void MatrixTest()
//{
//    Matrix<double, ExecutionSpace>* matrix = new Matrix<double, ExecutionSpace>("matrix", 2, 3);
//
//    Teuchos::RCP<Matrix<double, ExecutionSpace>>* instance = new Teuchos::RCP<Matrix<double, ExecutionSpace>>(matrix);
//
//    NdArray* ndArray = RcpViewToNdArrayRank1<double, ExecutionSpace, ExecutionSpace::array_layout>(instance);
//
//    std::cout << ndArray->label.Bytes << std::endl;
//
//    const Matrix<double, ExecutionSpace> lhs("lhs", 2, 3);
//    lhs(0, 0) = 1.0;
//    lhs(0, 1) = 2.0;
//    lhs(0, 2) = 3.0;
//    lhs(1, 0) = 4.0;
//    lhs(1, 1) = 5.0;
//    lhs(1, 2) = 6.0;
//
//    const Matrix<double, ExecutionSpace> rhs("rhs", 3, 2);
//    rhs(0, 0) = 10.0;
//    rhs(0, 1) = 11.0;
//    rhs(1, 0) = 20.0;
//    rhs(1, 1) = 21.0;
//    rhs(2, 0) = 30.0;
//    rhs(2, 1) = 31.0;
//
//    const Matrix<double, ExecutionSpace> eqs = lhs * rhs;
//
//    for (size_type i = 0; i < eqs.extent(0); ++i)
//    {
//        for (size_type j = 0; j < eqs.extent(1); ++j)
//        {
//            std::cout << eqs(i, j) << " ";
//        }
//        std::cout << std::endl;
//    }
//
//    const Matrix<double, ExecutionSpace> lhs_v("lhs_v", 3, 3);
//    lhs_v(0, 0) = 3.0;
//    lhs_v(0, 1) = 2.0;
//    lhs_v(0, 2) = 0.0;
//    lhs_v(1, 0) = 0.0;
//    lhs_v(1, 1) = 4.0;
//    lhs_v(1, 2) = 1.0;
//    lhs_v(2, 0) = 2.0;
//    lhs_v(2, 1) = 0.0;
//    lhs_v(2, 2) = 1.0;
//
//    const Vector<double, ExecutionSpace> rhs_v("rhs_v", 3);
//    rhs_v(0) = 4.0;
//    rhs_v(1) = 3.0;
//    rhs_v(2) = 1.0;
//
//    const Vector<double, ExecutionSpace> eqs_v = lhs_v * rhs_v;
//
//    for (size_type i = 0; i < eqs_v.extent(0); ++i)
//    {
//        std::cout << eqs_v(i) << " " << std::endl;
//    }
//}

namespace Kokkos
{
    template<class Type>
    struct Dot
    {
        using execution_space = typename Type::execution_space;

        static_assert(static_cast<unsigned>(Type::Rank) == static_cast<unsigned>(1), "Dot static_assert Fail: Rank != 1");

        using value_type = double;

#if 1
        typename Type::const_type X;
        typename Type::const_type Y;
#else
        Type            X;
        Type            Y;
#endif

        Dot(const Type& arg_x, const Type& arg_y) : X(arg_x), Y(arg_y) {}

        KOKKOS_INLINE_FUNCTION void operator()(int i, value_type& update) const
        {
            update += X[i] * Y[i];
        }

        KOKKOS_INLINE_FUNCTION static void join(volatile value_type& update, const volatile value_type& source)
        {
            update += source;
        }

        KOKKOS_INLINE_FUNCTION static void init(value_type& update)
        {
            update = 0;
        }
    };

    template<class Type>
    struct DotSingle
    {
        using execution_space = typename Type::execution_space;

        static_assert(static_cast<unsigned>(Type::Rank) == static_cast<unsigned>(1), "DotSingle static_assert Fail: Rank != 1");

        using value_type = double;

#if 1
        typename Type::const_type X;
#else
        Type            X;
#endif

        DotSingle(const Type& arg_x) : X(arg_x) {}

        KOKKOS_INLINE_FUNCTION void operator()(int i, value_type& update) const
        {
            const typename Type::value_type& x = X[i];
            update += x * x;
        }

        KOKKOS_INLINE_FUNCTION static void join(volatile value_type& update, const volatile value_type& source)
        {
            update += source;
        }

        KOKKOS_INLINE_FUNCTION static void init(value_type& update)
        {
            update = 0;
        }
    };

    template<class ScalarType, class VectorType>
    struct Scale
    {
        using execution_space = typename VectorType::execution_space;

        static_assert(static_cast<unsigned>(ScalarType::Rank) == static_cast<unsigned>(0), "Scale static_assert Fail: ScalarType::Rank != 0");

        static_assert(static_cast<unsigned>(VectorType::Rank) == static_cast<unsigned>(1), "Scale static_assert Fail: VectorType::Rank != 1");

#if 1
        typename ScalarType::const_type alpha;
#else
        ScalarType      alpha;
#endif

        VectorType Y;

        Scale(const ScalarType& arg_alpha, const VectorType& arg_Y) : alpha(arg_alpha), Y(arg_Y) {}

        KOKKOS_INLINE_FUNCTION void operator()(int i) const
        {
            Y[i] *= alpha();
        }
    };

    template<class ScalarType, class ConstVectorType, class VectorType>
    struct AXPBY
    {
        using execution_space = typename VectorType::execution_space;

        static_assert(static_cast<unsigned>(ScalarType::Rank) == static_cast<unsigned>(0), "AXPBY static_assert Fail: ScalarType::Rank != 0");

        static_assert(static_cast<unsigned>(ConstVectorType::Rank) == static_cast<unsigned>(1), "AXPBY static_assert Fail: ConstVectorType::Rank != 1");

        static_assert(static_cast<unsigned>(VectorType::Rank) == static_cast<unsigned>(1), "AXPBY static_assert Fail: VectorType::Rank != 1");

#if 1
        typename ScalarType::const_type      alpha, beta;
        typename ConstVectorType::const_type X;
#else
        ScalarType      alpha, beta;
        ConstVectorType X;
#endif

        VectorType Y;

        AXPBY(const ScalarType& arg_alpha, const ConstVectorType& arg_X, const ScalarType& arg_beta, const VectorType& arg_Y) : alpha(arg_alpha), beta(arg_beta), X(arg_X), Y(arg_Y) {}

        KOKKOS_INLINE_FUNCTION void operator()(int i) const
        {
            Y[i] = alpha() * X[i] + beta() * Y[i];
        }
    };

    template<class ConstScalarType, class ConstVectorType, class VectorType>
    __inline static void axpby(const ConstScalarType& alpha, const ConstVectorType& X, const ConstScalarType& beta, const VectorType& Y)
    {
        using functor = AXPBY<ConstScalarType, ConstVectorType, VectorType>;

        parallel_for(Y.extent(0), functor(alpha, X, beta, Y));
    }

    /** \brief  Y *= alpha */
    template<class ConstScalarType, class VectorType>
    __inline static void scale(const ConstScalarType& alpha, const VectorType& Y)
    {
        using functor = Scale<ConstScalarType, VectorType>;

        parallel_for(Y.extent(0), functor(alpha, Y));
    }

    template<class ConstVectorType, class Finalize>
    __inline static void dot(const ConstVectorType& X, const ConstVectorType& Y, const Finalize& finalize)
    {
        using functor = Dot<ConstVectorType>;

        parallel_reduce(X.extent(0), functor(X, Y), finalize);
    }

    template<class ConstVectorType, class Finalize>
    __inline static void dot(const ConstVectorType& X, const Finalize& finalize)
    {
        using functor = DotSingle<ConstVectorType>;

        parallel_reduce(X.extent(0), functor(X), finalize);
    }
}

// Reduction   : result = dot( Q(:,j) , Q(:,j) );
// PostProcess : R(j,j) = result ; inv = 1 / result ;
template<class VectorView, class ValueView>
struct InvNorm2 : public Kokkos::DotSingle<VectorView>
{
    using value_type = typename Kokkos::DotSingle<VectorView>::value_type;

    ValueView Rjj;
    ValueView inv;

    InvNorm2(const VectorView& argX, const ValueView& argR, const ValueView& argInv) : Kokkos::DotSingle<VectorView>(argX), Rjj(argR), inv(argInv) {}

    KOKKOS_INLINE_FUNCTION void final(value_type& result) const
    {
        result = Kokkos::Experimental::sqrt(result);
        Rjj()  = result;
        inv()  = (0 < result) ? 1.0 / result : 0;
    }
};

template<class VectorView, class ValueView>
inline static void invnorm2(const VectorView& x, const ValueView& r, const ValueView& r_inv)
{
    Kokkos::parallel_reduce(Kokkos::RangePolicy<typename VectorView::traits::execution_space>(0, x.extent(0)), InvNorm2<VectorView, ValueView>(x, r, r_inv));
}

// PostProcess : tmp = - ( R(j,k) = result );
template<class VectorView, class ValueView>
struct DotM : public Kokkos::Dot<VectorView>
{
    using value_type = typename Kokkos::Dot<VectorView>::value_type;

    ValueView Rjk;
    ValueView tmp;

    DotM(const VectorView& argX, const VectorView& argY, const ValueView& argR, const ValueView& argTmp) : Kokkos::Dot<VectorView>(argX, argY), Rjk(argR), tmp(argTmp) {}

    KOKKOS_INLINE_FUNCTION void final(value_type& result) const
    {
        Rjk() = result;
        tmp() = -result;
    }
};

template<class VectorView, class ValueView>
inline static void dot_neg(const VectorView& x, const VectorView& y, const ValueView& r, const ValueView& r_neg)
{
    Kokkos::parallel_reduce(Kokkos::RangePolicy<typename VectorView::traits::execution_space>(0, x.extent(0)), DotM<VectorView, ValueView>(x, y, r, r_neg));
}

template<typename Scalar, class DeviceType>
struct ModifiedGramSchmidt
{
    using execution_space = DeviceType;
    using size_type       = typename execution_space::size_type;

    using multivector_type = Kokkos::View<Scalar**, Kokkos::LayoutLeft, execution_space>;

    using vector_type = Kokkos::View<Scalar*, Kokkos::LayoutLeft, execution_space>;

    using value_view = Kokkos::View<Scalar, Kokkos::LayoutLeft, execution_space>;

    multivector_type Q;
    multivector_type R;

    static double factorization(const multivector_type Q_, const multivector_type R_)
    {
        const size_type count = Q_.extent(1);
        value_view      tmp("tmp");
        value_view      one("one");

        Kokkos::deep_copy(one, (Scalar)1);

        Kokkos::Timer timer;

        for (size_type j = 0; j < count; ++j)
        {
            // Reduction   : tmp = dot( Q(:,j) , Q(:,j) );
            // PostProcess : tmp = std::sqrt( tmp ); R(j,j) = tmp ; tmp = 1 / tmp ;
            const vector_type Qj  = Kokkos::subview(Q_, Kokkos::ALL(), j);
            const value_view  Rjj = Kokkos::subview(R_, j, j);

            invnorm2(Qj, Rjj, tmp);

            // Q(:,j) *= ( 1 / R(j,j) ); => Q(:,j) *= tmp ;
            Kokkos::scale(tmp, Qj);

            for (size_type k = j + 1; k < count; ++k)
            {
                const vector_type Qk  = Kokkos::subview(Q_, Kokkos::ALL(), k);
                const value_view  Rjk = Kokkos::subview(R_, j, k);

                // Reduction   : R(j,k) = dot( Q(:,j) , Q(:,k) );
                // PostProcess : tmp = - R(j,k);
                dot_neg(Qj, Qk, Rjk, tmp);

                // Q(:,k) -= R(j,k) * Q(:,j); => Q(:,k) += tmp * Q(:,j)
                Kokkos::axpby(tmp, Qj, one, Qk);
            }
        }

        execution_space().fence();

        return timer.seconds();
    }

    //--------------------------------------------------------------------------

    static double test(const size_type length, const size_type count, const size_t iter = 1)
    {
        multivector_type Q_("Q", length, count);
        multivector_type R_("R", count, count);

        typename multivector_type::HostMirror A = Kokkos::create_mirror(Q_);

        // Create and fill A on the host

        for (size_type j = 0; j < count; ++j)
        {
            for (size_type i = 0; i < length; ++i)
            {
                A(i, j) = (i + 1) * (j + 1);
            }
        }

        double dt_min = 0;

        for (size_t i = 0; i < iter; ++i)
        {
            Kokkos::deep_copy(Q_, A);

            // A = Q * R

            const double dt = factorization(Q_, R_);

            if (0 == i)
                dt_min = dt;
            else
                dt_min = dt < dt_min ? dt : dt_min;
        }

        return dt_min;
    }
};

template<class DeviceType>
void run_test_gramschmidt(const int exp_beg, const int exp_end, const int num_trials, const char deviceTypeName[])
{
    std::string label_gramschmidt;
    label_gramschmidt.append("\"GramSchmidt< double , ");
    label_gramschmidt.append(deviceTypeName);
    label_gramschmidt.append(" >\"");

    for (int i = exp_beg; i < exp_end; ++i)
    {
        double min_seconds = 0.0;
        double max_seconds = 0.0;
        double avg_seconds = 0.0;

        const int parallel_work_length = 1 << i;

        for (int j = 0; j < num_trials; ++j)
        {
            const double seconds = ModifiedGramSchmidt<double, DeviceType>::test(parallel_work_length, 32);

            if (0 == j)
            {
                min_seconds = seconds;
                max_seconds = seconds;
            }
            else
            {
                if (seconds < min_seconds)
                    min_seconds = seconds;
                if (seconds > max_seconds)
                    max_seconds = seconds;
            }
            avg_seconds += seconds;
        }
        avg_seconds /= num_trials;

        std::cout << label_gramschmidt << " , " << parallel_work_length << " , " << min_seconds << " , " << (min_seconds / parallel_work_length) << std::endl;
    }
}

template<class ExecutionSpace>
static void TestGramSchmidt()
{
    const int exp_beg    = 10;
    const int exp_end    = 20;
    const int num_trials = 5;

    run_test_gramschmidt<ExecutionSpace>(exp_beg, exp_end, num_trials, ExecutionSpace::name());

    //const double seconds = ModifiedGramSchmidt<double, ExecutionSpace>::test(1 << exp_end, 32);
}

template<class ExecutionSpace>
static void TestPCA()
{
    using Matrix = Kokkos::Extension::Matrix<double, ExecutionSpace>;
    using Vector = Kokkos::Extension::Vector<double, ExecutionSpace>;

    Matrix A("A", 329, 9);

#pragma region Matrix Values
    A(0, 0)   = 521.0;
    A(0, 1)   = 6200.0;
    A(0, 2)   = 237.0;
    A(0, 3)   = 923.0;
    A(0, 4)   = 4031.0;
    A(0, 5)   = 2757.0;
    A(0, 6)   = 996.0;
    A(0, 7)   = 1405.0;
    A(0, 8)   = 7633.0;
    A(1, 0)   = 575.0;
    A(1, 1)   = 8138.0;
    A(1, 2)   = 1656.0;
    A(1, 3)   = 886.0;
    A(1, 4)   = 4883.0;
    A(1, 5)   = 2438.0;
    A(1, 6)   = 5564.0;
    A(1, 7)   = 2632.0;
    A(1, 8)   = 4350.0;
    A(2, 0)   = 468.0;
    A(2, 1)   = 7339.0;
    A(2, 2)   = 618.0;
    A(2, 3)   = 970.0;
    A(2, 4)   = 2531.0;
    A(2, 5)   = 2560.0;
    A(2, 6)   = 237.0;
    A(2, 7)   = 859.0;
    A(2, 8)   = 5250.0;
    A(3, 0)   = 476.0;
    A(3, 1)   = 7908.0;
    A(3, 2)   = 1431.0;
    A(3, 3)   = 610.0;
    A(3, 4)   = 6883.0;
    A(3, 5)   = 3399.0;
    A(3, 6)   = 4655.0;
    A(3, 7)   = 1617.0;
    A(3, 8)   = 5864.0;
    A(4, 0)   = 659.0;
    A(4, 1)   = 8393.0;
    A(4, 2)   = 1853.0;
    A(4, 3)   = 1483.0;
    A(4, 4)   = 6558.0;
    A(4, 5)   = 3026.0;
    A(4, 6)   = 4496.0;
    A(4, 7)   = 2612.0;
    A(4, 8)   = 5727.0;
    A(5, 0)   = 520.0;
    A(5, 1)   = 5819.0;
    A(5, 2)   = 640.0;
    A(5, 3)   = 727.0;
    A(5, 4)   = 2444.0;
    A(5, 5)   = 2972.0;
    A(5, 6)   = 334.0;
    A(5, 7)   = 1018.0;
    A(5, 8)   = 5254.0;
    A(6, 0)   = 559.0;
    A(6, 1)   = 8288.0;
    A(6, 2)   = 621.0;
    A(6, 3)   = 514.0;
    A(6, 4)   = 2881.0;
    A(6, 5)   = 3144.0;
    A(6, 6)   = 2333.0;
    A(6, 7)   = 1117.0;
    A(6, 8)   = 5097.0;
    A(7, 0)   = 537.0;
    A(7, 1)   = 6487.0;
    A(7, 2)   = 965.0;
    A(7, 3)   = 706.0;
    A(7, 4)   = 4975.0;
    A(7, 5)   = 2945.0;
    A(7, 6)   = 1487.0;
    A(7, 7)   = 1280.0;
    A(7, 8)   = 5795.0;
    A(8, 0)   = 561.0;
    A(8, 1)   = 6191.0;
    A(8, 2)   = 432.0;
    A(8, 3)   = 399.0;
    A(8, 4)   = 4246.0;
    A(8, 5)   = 2778.0;
    A(8, 6)   = 256.0;
    A(8, 7)   = 1210.0;
    A(8, 8)   = 4230.0;
    A(9, 0)   = 609.0;
    A(9, 1)   = 6546.0;
    A(9, 2)   = 669.0;
    A(9, 3)   = 1073.0;
    A(9, 4)   = 4902.0;
    A(9, 5)   = 2852.0;
    A(9, 6)   = 1235.0;
    A(9, 7)   = 1109.0;
    A(9, 8)   = 6241.0;
    A(10, 0)  = 885.0;
    A(10, 1)  = 16047.0;
    A(10, 2)  = 2025.0;
    A(10, 3)  = 983.0;
    A(10, 4)  = 3954.0;
    A(10, 5)  = 2843.0;
    A(10, 6)  = 5632.0;
    A(10, 7)  = 3156.0;
    A(10, 8)  = 6220.0;
    A(11, 0)  = 195.0;
    A(11, 1)  = 12175.0;
    A(11, 2)  = 601.0;
    A(11, 3)  = 1223.0;
    A(11, 4)  = 5091.0;
    A(11, 5)  = 2414.0;
    A(11, 6)  = 2346.0;
    A(11, 7)  = 3000.0;
    A(11, 8)  = 7668.0;
    A(12, 0)  = 530.0;
    A(12, 1)  = 5704.0;
    A(12, 2)  = 580.0;
    A(12, 3)  = 878.0;
    A(12, 4)  = 2865.0;
    A(12, 5)  = 2469.0;
    A(12, 6)  = 430.0;
    A(12, 7)  = 838.0;
    A(12, 8)  = 3370.0;
    A(13, 0)  = 591.0;
    A(13, 1)  = 5725.0;
    A(13, 2)  = 820.0;
    A(13, 3)  = 975.0;
    A(13, 4)  = 2707.0;
    A(13, 5)  = 2772.0;
    A(13, 6)  = 169.0;
    A(13, 7)  = 613.0;
    A(13, 8)  = 4262.0;
    A(14, 0)  = 546.0;
    A(14, 1)  = 11014.0;
    A(14, 2)  = 2508.0;
    A(14, 3)  = 1067.0;
    A(14, 4)  = 3433.0;
    A(14, 5)  = 3346.0;
    A(14, 6)  = 7559.0;
    A(14, 7)  = 2288.0;
    A(14, 8)  = 4579.0;
    A(15, 0)  = 560.0;
    A(15, 1)  = 5530.0;
    A(15, 2)  = 598.0;
    A(15, 3)  = 1125.0;
    A(15, 4)  = 3051.0;
    A(15, 5)  = 2189.0;
    A(15, 6)  = 268.0;
    A(15, 7)  = 1165.0;
    A(15, 8)  = 4730.0;
    A(16, 0)  = 396.0;
    A(16, 1)  = 7877.0;
    A(16, 2)  = 833.0;
    A(16, 3)  = 525.0;
    A(16, 4)  = 3298.0;
    A(16, 5)  = 2844.0;
    A(16, 6)  = 1166.0;
    A(16, 7)  = 2315.0;
    A(16, 8)  = 5275.0;
    A(17, 0)  = 694.0;
    A(17, 1)  = 6722.0;
    A(17, 2)  = 1204.0;
    A(17, 3)  = 566.0;
    A(17, 4)  = 5086.0;
    A(17, 5)  = 2990.0;
    A(17, 6)  = 1391.0;
    A(17, 7)  = 1542.0;
    A(17, 8)  = 5196.0;
    A(18, 0)  = 601.0;
    A(18, 1)  = 6691.0;
    A(18, 2)  = 605.0;
    A(18, 3)  = 933.0;
    A(18, 4)  = 1866.0;
    A(18, 5)  = 2646.0;
    A(18, 6)  = 3546.0;
    A(18, 7)  = 1001.0;
    A(18, 8)  = 5193.0;
    A(19, 0)  = 696.0;
    A(19, 1)  = 8316.0;
    A(19, 2)  = 3195.0;
    A(19, 3)  = 1308.0;
    A(19, 4)  = 8409.0;
    A(19, 5)  = 3057.0;
    A(19, 6)  = 7559.0;
    A(19, 7)  = 1362.0;
    A(19, 8)  = 6315.0;
    A(20, 0)  = 615.0;
    A(20, 1)  = 11074.0;
    A(20, 2)  = 637.0;
    A(20, 3)  = 1878.0;
    A(20, 4)  = 3556.0;
    A(20, 5)  = 2929.0;
    A(20, 6)  = 621.0;
    A(20, 7)  = 2711.0;
    A(20, 8)  = 8107.0;
    A(21, 0)  = 534.0;
    A(21, 1)  = 6292.0;
    A(21, 2)  = 1798.0;
    A(21, 3)  = 872.0;
    A(21, 4)  = 2523.0;
    A(21, 5)  = 2915.0;
    A(21, 6)  = 1047.0;
    A(21, 7)  = 913.0;
    A(21, 8)  = 5431.0;
    A(22, 0)  = 474.0;
    A(22, 1)  = 10384.0;
    A(22, 2)  = 1203.0;
    A(22, 3)  = 821.0;
    A(22, 4)  = 3943.0;
    A(22, 5)  = 2208.0;
    A(22, 6)  = 3857.0;
    A(22, 7)  = 1800.0;
    A(22, 8)  = 5097.0;
    A(23, 0)  = 435.0;
    A(23, 1)  = 8831.0;
    A(23, 2)  = 782.0;
    A(23, 3)  = 1049.0;
    A(23, 4)  = 3670.0;
    A(23, 5)  = 3063.0;
    A(23, 6)  = 5355.0;
    A(23, 7)  = 1063.0;
    A(23, 8)  = 7439.0;
    A(24, 0)  = 560.0;
    A(24, 1)  = 8068.0;
    A(24, 2)  = 420.0;
    A(24, 3)  = 1561.0;
    A(24, 4)  = 3725.0;
    A(24, 5)  = 2564.0;
    A(24, 6)  = 1222.0;
    A(24, 7)  = 1568.0;
    A(24, 8)  = 6056.0;
    A(25, 0)  = 567.0;
    A(25, 1)  = 9148.0;
    A(25, 2)  = 3562.0;
    A(25, 3)  = 1730.0;
    A(25, 4)  = 7405.0;
    A(25, 5)  = 3471.0;
    A(25, 6)  = 9788.0;
    A(25, 7)  = 2925.0;
    A(25, 8)  = 5503.0;
    A(26, 0)  = 451.0;
    A(26, 1)  = 7277.0;
    A(26, 2)  = 780.0;
    A(26, 3)  = 651.0;
    A(26, 4)  = 5613.0;
    A(26, 5)  = 2934.0;
    A(26, 6)  = 1995.0;
    A(26, 7)  = 2148.0;
    A(26, 8)  = 5172.0;
    A(27, 0)  = 427.0;
    A(27, 1)  = 8083.0;
    A(27, 2)  = 342.0;
    A(27, 3)  = 1565.0;
    A(27, 4)  = 3329.0;
    A(27, 5)  = 2635.0;
    A(27, 6)  = 4237.0;
    A(27, 7)  = 1413.0;
    A(27, 8)  = 6308.0;
    A(28, 0)  = 527.0;
    A(28, 1)  = 6342.0;
    A(28, 2)  = 900.0;
    A(28, 3)  = 1031.0;
    A(28, 4)  = 4652.0;
    A(28, 5)  = 2483.0;
    A(28, 6)  = 354.0;
    A(28, 7)  = 1648.0;
    A(28, 8)  = 4008.0;
    A(29, 0)  = 423.0;
    A(29, 1)  = 6288.0;
    A(29, 2)  = 616.0;
    A(29, 3)  = 1313.0;
    A(29, 4)  = 2782.0;
    A(29, 5)  = 2745.0;
    A(29, 6)  = 1795.0;
    A(29, 7)  = 1813.0;
    A(29, 8)  = 6019.0;
    A(30, 0)  = 586.0;
    A(30, 1)  = 7866.0;
    A(30, 2)  = 861.0;
    A(30, 3)  = 310.0;
    A(30, 4)  = 2960.0;
    A(30, 5)  = 2535.0;
    A(30, 6)  = 1284.0;
    A(30, 7)  = 1480.0;
    A(30, 8)  = 3119.0;
    A(31, 0)  = 772.0;
    A(31, 1)  = 8329.0;
    A(31, 2)  = 240.0;
    A(31, 3)  = 825.0;
    A(31, 4)  = 3776.0;
    A(31, 5)  = 2778.0;
    A(31, 6)  = 1302.0;
    A(31, 7)  = 3200.0;
    A(31, 8)  = 4247.0;
    A(32, 0)  = 566.0;
    A(32, 1)  = 6761.0;
    A(32, 2)  = 570.0;
    A(32, 3)  = 1190.0;
    A(32, 4)  = 2989.0;
    A(32, 5)  = 2545.0;
    A(32, 6)  = 79.0;
    A(32, 7)  = 1477.0;
    A(32, 8)  = 3635.0;
    A(33, 0)  = 559.0;
    A(33, 1)  = 14607.0;
    A(33, 2)  = 2661.0;
    A(33, 3)  = 857.0;
    A(33, 4)  = 3511.0;
    A(33, 5)  = 3653.0;
    A(33, 6)  = 9304.0;
    A(33, 7)  = 1918.0;
    A(33, 8)  = 6016.0;
    A(34, 0)  = 452.0;
    A(34, 1)  = 8315.0;
    A(34, 2)  = 479.0;
    A(34, 3)  = 810.0;
    A(34, 4)  = 6285.0;
    A(34, 5)  = 3008.0;
    A(34, 6)  = 778.0;
    A(34, 7)  = 2046.0;
    A(34, 8)  = 5913.0;
    A(35, 0)  = 584.0;
    A(35, 1)  = 6458.0;
    A(35, 2)  = 441.0;
    A(35, 3)  = 810.0;
    A(35, 4)  = 2516.0;
    A(35, 5)  = 2592.0;
    A(35, 6)  = 679.0;
    A(35, 7)  = 2106.0;
    A(35, 8)  = 5801.0;
    A(36, 0)  = 550.0;
    A(36, 1)  = 8257.0;
    A(36, 2)  = 1007.0;
    A(36, 3)  = 415.0;
    A(36, 4)  = 4529.0;
    A(36, 5)  = 3052.0;
    A(36, 6)  = 1599.0;
    A(36, 7)  = 1722.0;
    A(36, 8)  = 5614.0;
    A(37, 0)  = 612.0;
    A(37, 1)  = 6811.0;
    A(37, 2)  = 1692.0;
    A(37, 3)  = 1123.0;
    A(37, 4)  = 5177.0;
    A(37, 5)  = 2851.0;
    A(37, 6)  = 3958.0;
    A(37, 7)  = 1234.0;
    A(37, 8)  = 4843.0;
    A(38, 0)  = 149.0;
    A(38, 1)  = 8365.0;
    A(38, 2)  = 804.0;
    A(38, 3)  = 413.0;
    A(38, 4)  = 4303.0;
    A(38, 5)  = 2686.0;
    A(38, 6)  = 1211.0;
    A(38, 7)  = 1630.0;
    A(38, 8)  = 6019.0;
    A(39, 0)  = 558.0;
    A(39, 1)  = 7056.0;
    A(39, 2)  = 731.0;
    A(39, 3)  = 657.0;
    A(39, 4)  = 1746.0;
    A(39, 5)  = 2873.0;
    A(39, 6)  = 2152.0;
    A(39, 7)  = 1990.0;
    A(39, 8)  = 4829.0;
    A(40, 0)  = 487.0;
    A(40, 1)  = 8654.0;
    A(40, 2)  = 815.0;
    A(40, 3)  = 673.0;
    A(40, 4)  = 5889.0;
    A(40, 5)  = 2854.0;
    A(40, 6)  = 1470.0;
    A(40, 7)  = 1605.0;
    A(40, 8)  = 5863.0;
    A(41, 0)  = 592.0;
    A(41, 1)  = 8221.0;
    A(41, 2)  = 453.0;
    A(41, 3)  = 880.0;
    A(41, 4)  = 6575.0;
    A(41, 5)  = 2391.0;
    A(41, 6)  = 2385.0;
    A(41, 7)  = 1672.0;
    A(41, 8)  = 4633.0;
    A(42, 0)  = 623.0;
    A(42, 1)  = 11609.0;
    A(42, 2)  = 5301.0;
    A(42, 3)  = 1215.0;
    A(42, 4)  = 6801.0;
    A(42, 5)  = 3479.0;
    A(42, 6)  = 21042.0;
    A(42, 7)  = 3066.0;
    A(42, 8)  = 6363.0;
    A(43, 0)  = 459.0;
    A(43, 1)  = 11914.0;
    A(43, 2)  = 962.0;
    A(43, 3)  = 1088.0;
    A(43, 4)  = 7108.0;
    A(43, 5)  = 2587.0;
    A(43, 6)  = 3663.0;
    A(43, 7)  = 4012.0;
    A(43, 8)  = 7127.0;
    A(44, 0)  = 440.0;
    A(44, 1)  = 8242.0;
    A(44, 2)  = 333.0;
    A(44, 3)  = 1093.0;
    A(44, 4)  = 3805.0;
    A(44, 5)  = 2712.0;
    A(44, 6)  = 154.0;
    A(44, 7)  = 1349.0;
    A(44, 8)  = 7437.0;
    A(45, 0)  = 423.0;
    A(45, 1)  = 8394.0;
    A(45, 2)  = 438.0;
    A(45, 3)  = 768.0;
    A(45, 4)  = 2391.0;
    A(45, 5)  = 2718.0;
    A(45, 6)  = 1506.0;
    A(45, 7)  = 1512.0;
    A(45, 8)  = 6020.0;
    A(46, 0)  = 808.0;
    A(46, 1)  = 9060.0;
    A(46, 2)  = 310.0;
    A(46, 3)  = 651.0;
    A(46, 4)  = 1670.0;
    A(46, 5)  = 2544.0;
    A(46, 6)  = 382.0;
    A(46, 7)  = 1973.0;
    A(46, 8)  = 5671.0;
    A(47, 0)  = 648.0;
    A(47, 1)  = 13429.0;
    A(47, 2)  = 2550.0;
    A(47, 3)  = 943.0;
    A(47, 4)  = 3197.0;
    A(47, 5)  = 3029.0;
    A(47, 6)  = 8368.0;
    A(47, 7)  = 1913.0;
    A(47, 8)  = 7197.0;
    A(48, 0)  = 516.0;
    A(48, 1)  = 10041.0;
    A(48, 2)  = 975.0;
    A(48, 3)  = 545.0;
    A(48, 4)  = 4495.0;
    A(48, 5)  = 2628.0;
    A(48, 6)  = 514.0;
    A(48, 7)  = 777.0;
    A(48, 8)  = 6527.0;
    A(49, 0)  = 575.0;
    A(49, 1)  = 8263.0;
    A(49, 2)  = 916.0;
    A(49, 3)  = 1336.0;
    A(49, 4)  = 3810.0;
    A(49, 5)  = 2729.0;
    A(49, 6)  = 2001.0;
    A(49, 7)  = 1217.0;
    A(49, 8)  = 6900.0;
    A(50, 0)  = 440.0;
    A(50, 1)  = 5376.0;
    A(50, 2)  = 91.0;
    A(50, 3)  = 974.0;
    A(50, 4)  = 3119.0;
    A(50, 5)  = 2413.0;
    A(50, 6)  = 162.0;
    A(50, 7)  = 3000.0;
    A(50, 8)  = 4968.0;
    A(51, 0)  = 383.0;
    A(51, 1)  = 8228.0;
    A(51, 2)  = 640.0;
    A(51, 3)  = 1016.0;
    A(51, 4)  = 2530.0;
    A(51, 5)  = 2973.0;
    A(51, 6)  = 2002.0;
    A(51, 7)  = 1413.0;
    A(51, 8)  = 8040.0;
    A(52, 0)  = 571.0;
    A(52, 1)  = 8064.0;
    A(52, 2)  = 2465.0;
    A(52, 3)  = 971.0;
    A(52, 4)  = 5384.0;
    A(52, 5)  = 3121.0;
    A(52, 6)  = 8567.0;
    A(52, 7)  = 2441.0;
    A(52, 8)  = 5047.0;
    A(53, 0)  = 637.0;
    A(53, 1)  = 6179.0;
    A(53, 2)  = 994.0;
    A(53, 3)  = 707.0;
    A(53, 4)  = 1910.0;
    A(53, 5)  = 2519.0;
    A(53, 6)  = 131.0;
    A(53, 7)  = 701.0;
    A(53, 8)  = 5680.0;
    A(54, 0)  = 383.0;
    A(54, 1)  = 9673.0;
    A(54, 2)  = 1809.0;
    A(54, 3)  = 494.0;
    A(54, 4)  = 7146.0;
    A(54, 5)  = 3323.0;
    A(54, 6)  = 1741.0;
    A(54, 7)  = 3357.0;
    A(54, 8)  = 6726.0;
    A(55, 0)  = 575.0;
    A(55, 1)  = 7332.0;
    A(55, 2)  = 443.0;
    A(55, 3)  = 650.0;
    A(55, 4)  = 4279.0;
    A(55, 5)  = 2754.0;
    A(55, 6)  = 989.0;
    A(55, 7)  = 1157.0;
    A(55, 8)  = 4847.0;
    A(56, 0)  = 401.0;
    A(56, 1)  = 9839.0;
    A(56, 2)  = 345.0;
    A(56, 3)  = 989.0;
    A(56, 4)  = 4410.0;
    A(56, 5)  = 2453.0;
    A(56, 6)  = 303.0;
    A(56, 7)  = 1435.0;
    A(56, 8)  = 6303.0;
    A(57, 0)  = 434.0;
    A(57, 1)  = 7774.0;
    A(57, 2)  = 837.0;
    A(57, 3)  = 714.0;
    A(57, 4)  = 5270.0;
    A(57, 5)  = 2619.0;
    A(57, 6)  = 904.0;
    A(57, 7)  = 1501.0;
    A(57, 8)  = 5009.0;
    A(58, 0)  = 525.0;
    A(58, 1)  = 8627.0;
    A(58, 2)  = 672.0;
    A(58, 3)  = 1022.0;
    A(58, 4)  = 7447.0;
    A(58, 5)  = 3147.0;
    A(58, 6)  = 2203.0;
    A(58, 7)  = 1700.0;
    A(58, 8)  = 5485.0;
    A(59, 0)  = 569.0;
    A(59, 1)  = 7402.0;
    A(59, 2)  = 1463.0;
    A(59, 3)  = 1495.0;
    A(59, 4)  = 4207.0;
    A(59, 5)  = 3164.0;
    A(59, 6)  = 2993.0;
    A(59, 7)  = 2561.0;
    A(59, 8)  = 5153.0;
    A(60, 0)  = 627.0;
    A(60, 1)  = 7789.0;
    A(60, 2)  = 708.0;
    A(60, 3)  = 721.0;
    A(60, 4)  = 5470.0;
    A(60, 5)  = 2894.0;
    A(60, 6)  = 2605.0;
    A(60, 7)  = 844.0;
    A(60, 8)  = 5257.0;
    A(61, 0)  = 644.0;
    A(61, 1)  = 7169.0;
    A(61, 2)  = 999.0;
    A(61, 3)  = 1273.0;
    A(61, 4)  = 6099.0;
    A(61, 5)  = 3031.0;
    A(61, 6)  = 4313.0;
    A(61, 7)  = 1236.0;
    A(61, 8)  = 5671.0;
    A(62, 0)  = 618.0;
    A(62, 1)  = 9531.0;
    A(62, 2)  = 1348.0;
    A(62, 3)  = 756.0;
    A(62, 4)  = 6041.0;
    A(62, 5)  = 3489.0;
    A(62, 6)  = 1422.0;
    A(62, 7)  = 1704.0;
    A(62, 8)  = 6055.0;
    A(63, 0)  = 576.0;
    A(63, 1)  = 6189.0;
    A(63, 2)  = 564.0;
    A(63, 3)  = 946.0;
    A(63, 4)  = 3401.0;
    A(63, 5)  = 2415.0;
    A(63, 6)  = 2483.0;
    A(63, 7)  = 1238.0;
    A(63, 8)  = 4487.0;
    A(64, 0)  = 514.0;
    A(64, 1)  = 10913.0;
    A(64, 2)  = 5766.0;
    A(64, 3)  = 1034.0;
    A(64, 4)  = 7742.0;
    A(64, 5)  = 3486.0;
    A(64, 6)  = 24846.0;
    A(64, 7)  = 2856.0;
    A(64, 8)  = 5205.0;
    A(65, 0)  = 603.0;
    A(65, 1)  = 8587.0;
    A(65, 2)  = 243.0;
    A(65, 3)  = 947.0;
    A(65, 4)  = 4067.0;
    A(65, 5)  = 3126.0;
    A(65, 6)  = 1647.0;
    A(65, 7)  = 1543.0;
    A(65, 8)  = 5307.0;
    A(66, 0)  = 584.0;
    A(66, 1)  = 8143.0;
    A(66, 2)  = 2138.0;
    A(66, 3)  = 978.0;
    A(66, 4)  = 5748.0;
    A(66, 5)  = 2918.0;
    A(66, 6)  = 9688.0;
    A(66, 7)  = 2451.0;
    A(66, 8)  = 5270.0;
    A(67, 0)  = 544.0;
    A(67, 1)  = 6007.0;
    A(67, 2)  = 446.0;
    A(67, 3)  = 736.0;
    A(67, 4)  = 2226.0;
    A(67, 5)  = 2654.0;
    A(67, 6)  = 111.0;
    A(67, 7)  = 2219.0;
    A(67, 8)  = 4880.0;
    A(68, 0)  = 579.0;
    A(68, 1)  = 9168.0;
    A(68, 2)  = 3167.0;
    A(68, 3)  = 1138.0;
    A(68, 4)  = 7333.0;
    A(68, 5)  = 2972.0;
    A(68, 6)  = 12679.0;
    A(68, 7)  = 3300.0;
    A(68, 8)  = 4879.0;
    A(69, 0)  = 526.0;
    A(69, 1)  = 8509.0;
    A(69, 2)  = 721.0;
    A(69, 3)  = 1086.0;
    A(69, 4)  = 3389.0;
    A(69, 5)  = 2754.0;
    A(69, 6)  = 1749.0;
    A(69, 7)  = 2375.0;
    A(69, 8)  = 7699.0;
    A(70, 0)  = 541.0;
    A(70, 1)  = 7702.0;
    A(70, 2)  = 1951.0;
    A(70, 3)  = 1065.0;
    A(70, 4)  = 3893.0;
    A(70, 5)  = 2377.0;
    A(70, 6)  = 2882.0;
    A(70, 7)  = 1331.0;
    A(70, 8)  = 5147.0;
    A(71, 0)  = 526.0;
    A(71, 1)  = 7519.0;
    A(71, 2)  = 1421.0;
    A(71, 3)  = 1524.0;
    A(71, 4)  = 5859.0;
    A(71, 5)  = 2908.0;
    A(71, 6)  = 2489.0;
    A(71, 7)  = 1484.0;
    A(71, 8)  = 5279.0;
    A(72, 0)  = 517.0;
    A(72, 1)  = 5817.0;
    A(72, 2)  = 833.0;
    A(72, 3)  = 820.0;
    A(72, 4)  = 2995.0;
    A(72, 5)  = 2665.0;
    A(72, 6)  = 1861.0;
    A(72, 7)  = 1214.0;
    A(72, 8)  = 4812.0;
    A(73, 0)  = 558.0;
    A(73, 1)  = 8093.0;
    A(73, 2)  = 1837.0;
    A(73, 3)  = 1092.0;
    A(73, 4)  = 4364.0;
    A(73, 5)  = 2928.0;
    A(73, 6)  = 6648.0;
    A(73, 7)  = 2020.0;
    A(73, 8)  = 5165.0;
    A(74, 0)  = 362.0;
    A(74, 1)  = 6929.0;
    A(74, 2)  = 458.0;
    A(74, 3)  = 1335.0;
    A(74, 4)  = 3626.0;
    A(74, 5)  = 2840.0;
    A(74, 6)  = 1992.0;
    A(74, 7)  = 2037.0;
    A(74, 8)  = 6690.0;
    A(75, 0)  = 591.0;
    A(75, 1)  = 6054.0;
    A(75, 2)  = 760.0;
    A(75, 3)  = 337.0;
    A(75, 4)  = 3709.0;
    A(75, 5)  = 3363.0;
    A(75, 6)  = 373.0;
    A(75, 7)  = 1036.0;
    A(75, 8)  = 4741.0;
    A(76, 0)  = 544.0;
    A(76, 1)  = 9318.0;
    A(76, 2)  = 2825.0;
    A(76, 3)  = 1529.0;
    A(76, 4)  = 6213.0;
    A(76, 5)  = 3269.0;
    A(76, 6)  = 10438.0;
    A(76, 7)  = 2310.0;
    A(76, 8)  = 7710.0;
    A(77, 0)  = 569.0;
    A(77, 1)  = 14420.0;
    A(77, 2)  = 2350.0;
    A(77, 3)  = 548.0;
    A(77, 4)  = 2715.0;
    A(77, 5)  = 3029.0;
    A(77, 6)  = 7415.0;
    A(77, 7)  = 1572.0;
    A(77, 8)  = 7060.0;
    A(78, 0)  = 545.0;
    A(78, 1)  = 5709.0;
    A(78, 2)  = 593.0;
    A(78, 3)  = 379.0;
    A(78, 4)  = 3161.0;
    A(78, 5)  = 2943.0;
    A(78, 6)  = 85.0;
    A(78, 7)  = 501.0;
    A(78, 8)  = 4491.0;
    A(79, 0)  = 440.0;
    A(79, 1)  = 8083.0;
    A(79, 2)  = 1113.0;
    A(79, 3)  = 834.0;
    A(79, 4)  = 3907.0;
    A(79, 5)  = 2901.0;
    A(79, 6)  = 1017.0;
    A(79, 7)  = 1920.0;
    A(79, 8)  = 4997.0;
    A(80, 0)  = 544.0;
    A(80, 1)  = 7635.0;
    A(80, 2)  = 2253.0;
    A(80, 3)  = 1151.0;
    A(80, 4)  = 4775.0;
    A(80, 5)  = 2772.0;
    A(80, 6)  = 6935.0;
    A(80, 7)  = 1122.0;
    A(80, 8)  = 4532.0;
    A(81, 0)  = 561.0;
    A(81, 1)  = 7203.0;
    A(81, 2)  = 723.0;
    A(81, 3)  = 1347.0;
    A(81, 4)  = 4117.0;
    A(81, 5)  = 2612.0;
    A(81, 6)  = 809.0;
    A(81, 7)  = 3967.0;
    A(81, 8)  = 6592.0;
    A(82, 0)  = 480.0;
    A(82, 1)  = 7395.0;
    A(82, 2)  = 732.0;
    A(82, 3)  = 897.0;
    A(82, 4)  = 3867.0;
    A(82, 5)  = 2683.0;
    A(82, 6)  = 298.0;
    A(82, 7)  = 1222.0;
    A(82, 8)  = 4274.0;
    A(83, 0)  = 521.0;
    A(83, 1)  = 10789.0;
    A(83, 2)  = 2533.0;
    A(83, 3)  = 1365.0;
    A(83, 4)  = 8145.0;
    A(83, 5)  = 3145.0;
    A(83, 6)  = 8477.0;
    A(83, 7)  = 2324.0;
    A(83, 8)  = 7164.0;
    A(84, 0)  = 444.0;
    A(84, 1)  = 8028.0;
    A(84, 2)  = 1256.0;
    A(84, 3)  = 1044.0;
    A(84, 4)  = 5521.0;
    A(84, 5)  = 2613.0;
    A(84, 6)  = 1857.0;
    A(84, 7)  = 1802.0;
    A(84, 8)  = 5346.0;
    A(85, 0)  = 536.0;
    A(85, 1)  = 8525.0;
    A(85, 2)  = 4142.0;
    A(85, 3)  = 1587.0;
    A(85, 4)  = 4808.0;
    A(85, 5)  = 3064.0;
    A(85, 6)  = 10389.0;
    A(85, 7)  = 2483.0;
    A(85, 8)  = 3904.0;
    A(86, 0)  = 336.0;
    A(86, 1)  = 5708.0;
    A(86, 2)  = 593.0;
    A(86, 3)  = 930.0;
    A(86, 4)  = 2232.0;
    A(86, 5)  = 2230.0;
    A(86, 6)  = 117.0;
    A(86, 7)  = 714.0;
    A(86, 8)  = 5453.0;
    A(87, 0)  = 419.0;
    A(87, 1)  = 7993.0;
    A(87, 2)  = 640.0;
    A(87, 3)  = 571.0;
    A(87, 4)  = 3668.0;
    A(87, 5)  = 2701.0;
    A(87, 6)  = 340.0;
    A(87, 7)  = 1587.0;
    A(87, 8)  = 3949.0;
    A(88, 0)  = 193.0;
    A(88, 1)  = 6040.0;
    A(88, 2)  = 1159.0;
    A(88, 3)  = 488.0;
    A(88, 4)  = 5205.0;
    A(88, 5)  = 2619.0;
    A(88, 6)  = 2377.0;
    A(88, 7)  = 3107.0;
    A(88, 8)  = 3922.0;
    A(89, 0)  = 537.0;
    A(89, 1)  = 6501.0;
    A(89, 2)  = 444.0;
    A(89, 3)  = 1096.0;
    A(89, 4)  = 6539.0;
    A(89, 5)  = 2630.0;
    A(89, 6)  = 904.0;
    A(89, 7)  = 1610.0;
    A(89, 8)  = 6113.0;
    A(90, 0)  = 257.0;
    A(90, 1)  = 7078.0;
    A(90, 2)  = 798.0;
    A(90, 3)  = 433.0;
    A(90, 4)  = 3197.0;
    A(90, 5)  = 2960.0;
    A(90, 6)  = 1807.0;
    A(90, 7)  = 1397.0;
    A(90, 8)  = 5348.0;
    A(91, 0)  = 592.0;
    A(91, 1)  = 7343.0;
    A(91, 2)  = 528.0;
    A(91, 3)  = 1323.0;
    A(91, 4)  = 3705.0;
    A(91, 5)  = 2479.0;
    A(91, 6)  = 3800.0;
    A(91, 7)  = 1101.0;
    A(91, 8)  = 5080.0;
    A(92, 0)  = 521.0;
    A(92, 1)  = 6573.0;
    A(92, 2)  = 596.0;
    A(92, 3)  = 524.0;
    A(92, 4)  = 4168.0;
    A(92, 5)  = 2537.0;
    A(92, 6)  = 353.0;
    A(92, 7)  = 1023.0;
    A(92, 8)  = 4214.0;
    A(93, 0)  = 467.0;
    A(93, 1)  = 7078.0;
    A(93, 2)  = 562.0;
    A(93, 3)  = 582.0;
    A(93, 4)  = 3324.0;
    A(93, 5)  = 3000.0;
    A(93, 6)  = 1048.0;
    A(93, 7)  = 1600.0;
    A(93, 8)  = 4813.0;
    A(94, 0)  = 461.0;
    A(94, 1)  = 6829.0;
    A(94, 2)  = 626.0;
    A(94, 3)  = 845.0;
    A(94, 4)  = 2312.0;
    A(94, 5)  = 2764.0;
    A(94, 6)  = 215.0;
    A(94, 7)  = 1200.0;
    A(94, 8)  = 8268.0;
    A(95, 0)  = 605.0;
    A(95, 1)  = 7715.0;
    A(95, 2)  = 529.0;
    A(95, 3)  = 635.0;
    A(95, 4)  = 5754.0;
    A(95, 5)  = 2641.0;
    A(95, 6)  = 2032.0;
    A(95, 7)  = 1340.0;
    A(95, 8)  = 4299.0;
    A(96, 0)  = 741.0;
    A(96, 1)  = 9370.0;
    A(96, 2)  = 539.0;
    A(96, 3)  = 874.0;
    A(96, 4)  = 5293.0;
    A(96, 5)  = 3118.0;
    A(96, 6)  = 2631.0;
    A(96, 7)  = 3400.0;
    A(96, 8)  = 3045.0;
    A(97, 0)  = 550.0;
    A(97, 1)  = 6743.0;
    A(97, 2)  = 783.0;
    A(97, 3)  = 864.0;
    A(97, 4)  = 3496.0;
    A(97, 5)  = 2797.0;
    A(97, 6)  = 1876.0;
    A(97, 7)  = 1622.0;
    A(97, 8)  = 5206.0;
    A(98, 0)  = 643.0;
    A(98, 1)  = 9017.0;
    A(98, 2)  = 900.0;
    A(98, 3)  = 861.0;
    A(98, 4)  = 4602.0;
    A(98, 5)  = 2439.0;
    A(98, 6)  = 749.0;
    A(98, 7)  = 2005.0;
    A(98, 8)  = 4884.0;
    A(99, 0)  = 148.0;
    A(99, 1)  = 8168.0;
    A(99, 2)  = 920.0;
    A(99, 3)  = 503.0;
    A(99, 4)  = 6325.0;
    A(99, 5)  = 2506.0;
    A(99, 6)  = 2111.0;
    A(99, 7)  = 1414.0;
    A(99, 8)  = 5594.0;
    A(100, 0) = 561.0;
    A(100, 1) = 6274.0;
    A(100, 2) = 872.0;
    A(100, 3) = 1150.0;
    A(100, 4) = 4402.0;
    A(100, 5) = 3051.0;
    A(100, 6) = 844.0;
    A(100, 7) = 709.0;
    A(100, 8) = 5255.0;
    A(101, 0) = 549.0;
    A(101, 1) = 6686.0;
    A(101, 2) = 594.0;
    A(101, 3) = 545.0;
    A(101, 4) = 3581.0;
    A(101, 5) = 2334.0;
    A(101, 6) = 1915.0;
    A(101, 7) = 1695.0;
    A(101, 8) = 4631.0;
    A(102, 0) = 507.0;
    A(102, 1) = 8252.0;
    A(102, 2) = 655.0;
    A(102, 3) = 655.0;
    A(102, 4) = 2244.0;
    A(102, 5) = 2799.0;
    A(102, 6) = 270.0;
    A(102, 7) = 790.0;
    A(102, 8) = 5098.0;
    A(103, 0) = 540.0;
    A(103, 1) = 7204.0;
    A(103, 2) = 724.0;
    A(103, 3) = 1671.0;
    A(103, 4) = 4912.0;
    A(103, 5) = 2511.0;
    A(103, 6) = 2163.0;
    A(103, 7) = 1355.0;
    A(103, 8) = 3724.0;
    A(104, 0) = 546.0;
    A(104, 1) = 5962.0;
    A(104, 2) = 607.0;
    A(104, 3) = 516.0;
    A(104, 4) = 1454.0;
    A(104, 5) = 2427.0;
    A(104, 6) = 1021.0;
    A(104, 7) = 994.0;
    A(104, 8) = 4492.0;
    A(105, 0) = 552.0;
    A(105, 1) = 6508.0;
    A(105, 2) = 818.0;
    A(105, 3) = 1334.0;
    A(105, 4) = 4963.0;
    A(105, 5) = 3109.0;
    A(105, 6) = 628.0;
    A(105, 7) = 800.0;
    A(105, 8) = 4842.0;
    A(106, 0) = 490.0;
    A(106, 1) = 9951.0;
    A(106, 2) = 731.0;
    A(106, 3) = 744.0;
    A(106, 4) = 2637.0;
    A(106, 5) = 2413.0;
    A(106, 6) = 1609.0;
    A(106, 7) = 4200.0;
    A(106, 8) = 6631.0;
    A(107, 0) = 572.0;
    A(107, 1) = 10810.0;
    A(107, 2) = 1252.0;
    A(107, 3) = 1536.0;
    A(107, 4) = 4186.0;
    A(107, 5) = 2734.0;
    A(107, 6) = 2027.0;
    A(107, 7) = 2455.0;
    A(107, 8) = 7136.0;
    A(108, 0) = 342.0;
    A(108, 1) = 9298.0;
    A(108, 2) = 546.0;
    A(108, 3) = 787.0;
    A(108, 4) = 4583.0;
    A(108, 5) = 2729.0;
    A(108, 6) = 380.0;
    A(108, 7) = 4005.0;
    A(108, 8) = 7166.0;
    A(109, 0) = 602.0;
    A(109, 1) = 8842.0;
    A(109, 2) = 527.0;
    A(109, 3) = 1422.0;
    A(109, 4) = 2143.0;
    A(109, 5) = 3154.0;
    A(109, 6) = 368.0;
    A(109, 7) = 2058.0;
    A(109, 8) = 7973.0;
    A(110, 0) = 482.0;
    A(110, 1) = 5784.0;
    A(110, 2) = 466.0;
    A(110, 3) = 663.0;
    A(110, 4) = 3092.0;
    A(110, 5) = 2927.0;
    A(110, 6) = 145.0;
    A(110, 7) = 1736.0;
    A(110, 8) = 4849.0;
    A(111, 0) = 536.0;
    A(111, 1) = 7554.0;
    A(111, 2) = 484.0;
    A(111, 3) = 544.0;
    A(111, 4) = 2886.0;
    A(111, 5) = 2809.0;
    A(111, 6) = 87.0;
    A(111, 7) = 2092.0;
    A(111, 8) = 6342.0;
    A(112, 0) = 509.0;
    A(112, 1) = 6733.0;
    A(112, 2) = 1060.0;
    A(112, 3) = 710.0;
    A(112, 4) = 5416.0;
    A(112, 5) = 2772.0;
    A(112, 6) = 2846.0;
    A(112, 7) = 1711.0;
    A(112, 8) = 4195.0;
    A(113, 0) = 528.0;
    A(113, 1) = 7956.0;
    A(113, 2) = 1038.0;
    A(113, 3) = 1348.0;
    A(113, 4) = 4472.0;
    A(113, 5) = 2627.0;
    A(113, 6) = 6466.0;
    A(113, 7) = 2366.0;
    A(113, 8) = 6862.0;
    A(114, 0) = 559.0;
    A(114, 1) = 9291.0;
    A(114, 2) = 369.0;
    A(114, 3) = 1483.0;
    A(114, 4) = 4388.0;
    A(114, 5) = 2407.0;
    A(114, 6) = 3596.0;
    A(114, 7) = 2984.0;
    A(114, 8) = 5746.0;
    A(115, 0) = 526.0;
    A(115, 1) = 5382.0;
    A(115, 2) = 622.0;
    A(115, 3) = 749.0;
    A(115, 4) = 2174.0;
    A(115, 5) = 2299.0;
    A(115, 6) = 153.0;
    A(115, 7) = 300.0;
    A(115, 8) = 4220.0;
    A(116, 0) = 402.0;
    A(116, 1) = 7388.0;
    A(116, 2) = 1731.0;
    A(116, 3) = 1658.0;
    A(116, 4) = 3527.0;
    A(116, 5) = 3094.0;
    A(116, 6) = 3335.0;
    A(116, 7) = 1237.0;
    A(116, 8) = 5739.0;
    A(117, 0) = 727.0;
    A(117, 1) = 7767.0;
    A(117, 2) = 1437.0;
    A(117, 3) = 1213.0;
    A(117, 4) = 3423.0;
    A(117, 5) = 2809.0;
    A(117, 6) = 1756.0;
    A(117, 7) = 3000.0;
    A(117, 8) = 6026.0;
    A(118, 0) = 483.0;
    A(118, 1) = 7641.0;
    A(118, 2) = 1364.0;
    A(118, 3) = 996.0;
    A(118, 4) = 5855.0;
    A(118, 5) = 2526.0;
    A(118, 6) = 4115.0;
    A(118, 7) = 1940.0;
    A(118, 8) = 3826.0;
    A(119, 0) = 476.0;
    A(119, 1) = 7120.0;
    A(119, 2) = 43.0;
    A(119, 3) = 568.0;
    A(119, 4) = 2241.0;
    A(119, 5) = 2674.0;
    A(119, 6) = 603.0;
    A(119, 7) = 1883.0;
    A(119, 8) = 5166.0;
    A(120, 0) = 105.0;
    A(120, 1) = 7898.0;
    A(120, 2) = 1109.0;
    A(120, 3) = 401.0;
    A(120, 4) = 5587.0;
    A(120, 5) = 2721.0;
    A(120, 6) = 1921.0;
    A(120, 7) = 1304.0;
    A(120, 8) = 5646.0;
    A(121, 0) = 513.0;
    A(121, 1) = 7780.0;
    A(121, 2) = 1274.0;
    A(121, 3) = 952.0;
    A(121, 4) = 3454.0;
    A(121, 5) = 2705.0;
    A(121, 6) = 3255.0;
    A(121, 7) = 1909.0;
    A(121, 8) = 4848.0;
    A(122, 0) = 410.0;
    A(122, 1) = 7143.0;
    A(122, 2) = 667.0;
    A(122, 3) = 792.0;
    A(122, 4) = 3747.0;
    A(122, 5) = 2737.0;
    A(122, 6) = 401.0;
    A(122, 7) = 2176.0;
    A(122, 8) = 4697.0;
    A(123, 0) = 490.0;
    A(123, 1) = 8218.0;
    A(123, 2) = 706.0;
    A(123, 3) = 994.0;
    A(123, 4) = 1641.0;
    A(123, 5) = 2854.0;
    A(123, 6) = 1254.0;
    A(123, 7) = 739.0;
    A(123, 8) = 5443.0;
    A(124, 0) = 367.0;
    A(124, 1) = 8401.0;
    A(124, 2) = 916.0;
    A(124, 3) = 583.0;
    A(124, 4) = 3793.0;
    A(124, 5) = 2622.0;
    A(124, 6) = 2547.0;
    A(124, 7) = 1925.0;
    A(124, 8) = 5650.0;
    A(125, 0) = 626.0;
    A(125, 1) = 7064.0;
    A(125, 2) = 1694.0;
    A(125, 3) = 967.0;
    A(125, 4) = 4453.0;
    A(125, 5) = 3090.0;
    A(125, 6) = 4188.0;
    A(125, 7) = 1651.0;
    A(125, 8) = 5204.0;
    A(126, 0) = 655.0;
    A(126, 1) = 6336.0;
    A(126, 2) = 1260.0;
    A(126, 3) = 1185.0;
    A(126, 4) = 3950.0;
    A(126, 5) = 3236.0;
    A(126, 6) = 2569.0;
    A(126, 7) = 1410.0;
    A(126, 8) = 5012.0;
    A(127, 0) = 568.0;
    A(127, 1) = 7763.0;
    A(127, 2) = 818.0;
    A(127, 3) = 627.0;
    A(127, 4) = 3431.0;
    A(127, 5) = 2990.0;
    A(127, 6) = 825.0;
    A(127, 7) = 1491.0;
    A(127, 8) = 4477.0;
    A(128, 0) = 542.0;
    A(128, 1) = 8227.0;
    A(128, 2) = 1135.0;
    A(128, 3) = 892.0;
    A(128, 4) = 3338.0;
    A(128, 5) = 2747.0;
    A(128, 6) = 2316.0;
    A(128, 7) = 1604.0;
    A(128, 8) = 4618.0;
    A(129, 0) = 556.0;
    A(129, 1) = 7891.0;
    A(129, 2) = 2087.0;
    A(129, 3) = 629.0;
    A(129, 4) = 6164.0;
    A(129, 5) = 3224.0;
    A(129, 6) = 3083.0;
    A(129, 7) = 1532.0;
    A(129, 8) = 5322.0;
    A(130, 0) = 516.0;
    A(130, 1) = 11652.0;
    A(130, 2) = 2521.0;
    A(130, 3) = 1279.0;
    A(130, 4) = 7120.0;
    A(130, 5) = 3628.0;
    A(130, 6) = 3616.0;
    A(130, 7) = 1790.0;
    A(130, 8) = 6307.0;
    A(131, 0) = 623.0;
    A(131, 1) = 6760.0;
    A(131, 2) = 1006.0;
    A(131, 3) = 765.0;
    A(131, 4) = 2703.0;
    A(131, 5) = 2726.0;
    A(131, 6) = 188.0;
    A(131, 7) = 797.0;
    A(131, 8) = 4728.0;
    A(132, 0) = 717.0;
    A(132, 1) = 17021.0;
    A(132, 2) = 1298.0;
    A(132, 3) = 891.0;
    A(132, 4) = 5911.0;
    A(132, 5) = 2502.0;
    A(132, 6) = 7168.0;
    A(132, 7) = 3703.0;
    A(132, 8) = 5187.0;
    A(133, 0) = 427.0;
    A(133, 1) = 7094.0;
    A(133, 2) = 583.0;
    A(133, 3) = 400.0;
    A(133, 4) = 1145.0;
    A(133, 5) = 1995.0;
    A(133, 6) = 725.0;
    A(133, 7) = 2700.0;
    A(133, 8) = 6662.0;
    A(134, 0) = 424.0;
    A(134, 1) = 9760.0;
    A(134, 2) = 2467.0;
    A(134, 3) = 1499.0;
    A(134, 4) = 4626.0;
    A(134, 5) = 3271.0;
    A(134, 6) = 11073.0;
    A(134, 7) = 1825.0;
    A(134, 8) = 7464.0;
    A(135, 0) = 636.0;
    A(135, 1) = 6632.0;
    A(135, 2) = 875.0;
    A(135, 3) = 665.0;
    A(135, 4) = 4001.0;
    A(135, 5) = 2525.0;
    A(135, 6) = 2195.0;
    A(135, 7) = 840.0;
    A(135, 8) = 4383.0;
    A(136, 0) = 600.0;
    A(136, 1) = 6283.0;
    A(136, 2) = 685.0;
    A(136, 3) = 924.0;
    A(136, 4) = 2661.0;
    A(136, 5) = 2257.0;
    A(136, 6) = 1921.0;
    A(136, 7) = 1075.0;
    A(136, 8) = 6412.0;
    A(137, 0) = 557.0;
    A(137, 1) = 7012.0;
    A(137, 2) = 2243.0;
    A(137, 3) = 1000.0;
    A(137, 4) = 5804.0;
    A(137, 5) = 2690.0;
    A(137, 6) = 6348.0;
    A(137, 7) = 1906.0;
    A(137, 8) = 5082.0;
    A(138, 0) = 434.0;
    A(138, 1) = 9429.0;
    A(138, 2) = 2437.0;
    A(138, 3) = 830.0;
    A(138, 4) = 2770.0;
    A(138, 5) = 2842.0;
    A(138, 6) = 2255.0;
    A(138, 7) = 1506.0;
    A(138, 8) = 5165.0;
    A(139, 0) = 518.0;
    A(139, 1) = 6794.0;
    A(139, 2) = 679.0;
    A(139, 3) = 1021.0;
    A(139, 4) = 4800.0;
    A(139, 5) = 2654.0;
    A(139, 6) = 323.0;
    A(139, 7) = 1933.0;
    A(139, 8) = 3822.0;
    A(140, 0) = 412.0;
    A(140, 1) = 7245.0;
    A(140, 2) = 1792.0;
    A(140, 3) = 1091.0;
    A(140, 4) = 4917.0;
    A(140, 5) = 3130.0;
    A(140, 6) = 3209.0;
    A(140, 7) = 1427.0;
    A(140, 8) = 5186.0;
    A(141, 0) = 457.0;
    A(141, 1) = 6626.0;
    A(141, 2) = 1181.0;
    A(141, 3) = 1211.0;
    A(141, 4) = 5611.0;
    A(141, 5) = 3048.0;
    A(141, 6) = 2162.0;
    A(141, 7) = 2884.0;
    A(141, 8) = 6139.0;
    A(142, 0) = 564.0;
    A(142, 1) = 6111.0;
    A(142, 2) = 740.0;
    A(142, 3) = 967.0;
    A(142, 4) = 1780.0;
    A(142, 5) = 2646.0;
    A(142, 6) = 567.0;
    A(142, 7) = 1177.0;
    A(142, 8) = 6386.0;
    A(143, 0) = 466.0;
    A(143, 1) = 7447.0;
    A(143, 2) = 700.0;
    A(143, 3) = 858.0;
    A(143, 4) = 3092.0;
    A(143, 5) = 2532.0;
    A(143, 6) = 1092.0;
    A(143, 7) = 1615.0;
    A(143, 8) = 4371.0;
    A(144, 0) = 601.0;
    A(144, 1) = 8810.0;
    A(144, 2) = 1759.0;
    A(144, 3) = 1434.0;
    A(144, 4) = 4982.0;
    A(144, 5) = 2574.0;
    A(144, 6) = 7420.0;
    A(144, 7) = 1001.0;
    A(144, 8) = 4889.0;
    A(145, 0) = 663.0;
    A(145, 1) = 6119.0;
    A(145, 2) = 1152.0;
    A(145, 3) = 424.0;
    A(145, 4) = 2532.0;
    A(145, 5) = 2925.0;
    A(145, 6) = 1925.0;
    A(145, 7) = 2155.0;
    A(145, 8) = 4903.0;
    A(146, 0) = 547.0;
    A(146, 1) = 6524.0;
    A(146, 2) = 731.0;
    A(146, 3) = 353.0;
    A(146, 4) = 4343.0;
    A(146, 5) = 2691.0;
    A(146, 6) = 666.0;
    A(146, 7) = 903.0;
    A(146, 8) = 4181.0;
    A(147, 0) = 479.0;
    A(147, 1) = 9327.0;
    A(147, 2) = 1058.0;
    A(147, 3) = 837.0;
    A(147, 4) = 4645.0;
    A(147, 5) = 2868.0;
    A(147, 6) = 3177.0;
    A(147, 7) = 1636.0;
    A(147, 8) = 4631.0;
    A(148, 0) = 580.0;
    A(148, 1) = 5159.0;
    A(148, 2) = 500.0;
    A(148, 3) = 628.0;
    A(148, 4) = 2335.0;
    A(148, 5) = 2871.0;
    A(148, 6) = 80.0;
    A(148, 7) = 801.0;
    A(148, 8) = 5324.0;
    A(149, 0) = 527.0;
    A(149, 1) = 7919.0;
    A(149, 2) = 1043.0;
    A(149, 3) = 1120.0;
    A(149, 4) = 5419.0;
    A(149, 5) = 2896.0;
    A(149, 6) = 2071.0;
    A(149, 7) = 2163.0;
    A(149, 8) = 4794.0;
    A(150, 0) = 483.0;
    A(150, 1) = 7230.0;
    A(150, 2) = 609.0;
    A(150, 3) = 976.0;
    A(150, 4) = 3444.0;
    A(150, 5) = 2855.0;
    A(150, 6) = 75.0;
    A(150, 7) = 1119.0;
    A(150, 8) = 5579.0;
    A(151, 0) = 549.0;
    A(151, 1) = 8126.0;
    A(151, 2) = 1711.0;
    A(151, 3) = 1142.0;
    A(151, 4) = 5006.0;
    A(151, 5) = 3028.0;
    A(151, 6) = 1167.0;
    A(151, 7) = 1045.0;
    A(151, 8) = 6166.0;
    A(152, 0) = 549.0;
    A(152, 1) = 7076.0;
    A(152, 2) = 1939.0;
    A(152, 3) = 1468.0;
    A(152, 4) = 5869.0;
    A(152, 5) = 2949.0;
    A(152, 6) = 5553.0;
    A(152, 7) = 2043.0;
    A(152, 8) = 4865.0;
    A(153, 0) = 496.0;
    A(153, 1) = 8516.0;
    A(153, 2) = 1067.0;
    A(153, 3) = 911.0;
    A(153, 4) = 4473.0;
    A(153, 5) = 2918.0;
    A(153, 6) = 3844.0;
    A(153, 7) = 2224.0;
    A(153, 8) = 5176.0;
    A(154, 0) = 365.0;
    A(154, 1) = 6463.0;
    A(154, 2) = 398.0;
    A(154, 3) = 733.0;
    A(154, 4) = 2862.0;
    A(154, 5) = 3167.0;
    A(154, 6) = 920.0;
    A(154, 7) = 931.0;
    A(154, 8) = 6331.0;
    A(155, 0) = 670.0;
    A(155, 1) = 6692.0;
    A(155, 2) = 960.0;
    A(155, 3) = 622.0;
    A(155, 4) = 4273.0;
    A(155, 5) = 2761.0;
    A(155, 6) = 3309.0;
    A(155, 7) = 2514.0;
    A(155, 8) = 5537.0;
    A(156, 0) = 512.0;
    A(156, 1) = 6616.0;
    A(156, 2) = 596.0;
    A(156, 3) = 413.0;
    A(156, 4) = 1817.0;
    A(156, 5) = 2904.0;
    A(156, 6) = 285.0;
    A(156, 7) = 1000.0;
    A(156, 8) = 3429.0;
    A(157, 0) = 352.0;
    A(157, 1) = 8310.0;
    A(157, 2) = 686.0;
    A(157, 3) = 676.0;
    A(157, 4) = 6096.0;
    A(157, 5) = 3027.0;
    A(157, 6) = 1466.0;
    A(157, 7) = 1953.0;
    A(157, 8) = 5648.0;
    A(158, 0) = 494.0;
    A(158, 1) = 7778.0;
    A(158, 2) = 655.0;
    A(158, 3) = 465.0;
    A(158, 4) = 4956.0;
    A(158, 5) = 2945.0;
    A(158, 6) = 2235.0;
    A(158, 7) = 1814.0;
    A(158, 8) = 4333.0;
    A(159, 0) = 429.0;
    A(159, 1) = 8572.0;
    A(159, 2) = 548.0;
    A(159, 3) = 1030.0;
    A(159, 4) = 5268.0;
    A(159, 5) = 2305.0;
    A(159, 6) = 1772.0;
    A(159, 7) = 1734.0;
    A(159, 8) = 9702.0;
    A(160, 0) = 469.0;
    A(160, 1) = 6921.0;
    A(160, 2) = 314.0;
    A(160, 3) = 1093.0;
    A(160, 4) = 3549.0;
    A(160, 5) = 2336.0;
    A(160, 6) = 1456.0;
    A(160, 7) = 1855.0;
    A(160, 8) = 5872.0;
    A(161, 0) = 514.0;
    A(161, 1) = 13282.0;
    A(161, 2) = 1237.0;
    A(161, 3) = 822.0;
    A(161, 4) = 3422.0;
    A(161, 5) = 2607.0;
    A(161, 6) = 3746.0;
    A(161, 7) = 2435.0;
    A(161, 8) = 5755.0;
    A(162, 0) = 307.0;
    A(162, 1) = 6680.0;
    A(162, 2) = 323.0;
    A(162, 3) = 1373.0;
    A(162, 4) = 3412.0;
    A(162, 5) = 2998.0;
    A(162, 6) = 309.0;
    A(162, 7) = 2513.0;
    A(162, 8) = 5594.0;
    A(163, 0) = 559.0;
    A(163, 1) = 8631.0;
    A(163, 2) = 1111.0;
    A(163, 3) = 413.0;
    A(163, 4) = 3908.0;
    A(163, 5) = 3097.0;
    A(163, 6) = 1015.0;
    A(163, 7) = 1147.0;
    A(163, 8) = 5120.0;
    A(164, 0) = 480.0;
    A(164, 1) = 7907.0;
    A(164, 2) = 1371.0;
    A(164, 3) = 894.0;
    A(164, 4) = 5557.0;
    A(164, 5) = 2891.0;
    A(164, 6) = 4206.0;
    A(164, 7) = 1609.0;
    A(164, 8) = 4747.0;
    A(165, 0) = 424.0;
    A(165, 1) = 6152.0;
    A(165, 2) = 465.0;
    A(165, 3) = 1050.0;
    A(165, 4) = 3322.0;
    A(165, 5) = 2827.0;
    A(165, 6) = 150.0;
    A(165, 7) = 702.0;
    A(165, 8) = 5264.0;
    A(166, 0) = 552.0;
    A(166, 1) = 6962.0;
    A(166, 2) = 588.0;
    A(166, 3) = 1457.0;
    A(166, 4) = 2989.0;
    A(166, 5) = 2736.0;
    A(166, 6) = 2804.0;
    A(166, 7) = 1609.0;
    A(166, 8) = 5341.0;
    A(167, 0) = 556.0;
    A(167, 1) = 9906.0;
    A(167, 2) = 412.0;
    A(167, 3) = 1913.0;
    A(167, 4) = 5900.0;
    A(167, 5) = 2241.0;
    A(167, 6) = 1586.0;
    A(167, 7) = 3996.0;
    A(167, 8) = 6035.0;
    A(168, 0) = 513.0;
    A(168, 1) = 7497.0;
    A(168, 2) = 621.0;
    A(168, 3) = 1018.0;
    A(168, 4) = 2931.0;
    A(168, 5) = 2700.0;
    A(168, 6) = 3150.0;
    A(168, 7) = 1752.0;
    A(168, 8) = 4573.0;
    A(169, 0) = 548.0;
    A(169, 1) = 10414.0;
    A(169, 2) = 1202.0;
    A(169, 3) = 909.0;
    A(169, 4) = 3575.0;
    A(169, 5) = 2479.0;
    A(169, 6) = 2111.0;
    A(169, 7) = 1879.0;
    A(169, 8) = 6527.0;
    A(170, 0) = 479.0;
    A(170, 1) = 5850.0;
    A(170, 2) = 477.0;
    A(170, 3) = 1156.0;
    A(170, 4) = 2366.0;
    A(170, 5) = 2375.0;
    A(170, 6) = 1280.0;
    A(170, 7) = 1757.0;
    A(170, 8) = 6105.0;
    A(171, 0) = 490.0;
    A(171, 1) = 6876.0;
    A(171, 2) = 759.0;
    A(171, 3) = 764.0;
    A(171, 4) = 2941.0;
    A(171, 5) = 2694.0;
    A(171, 6) = 736.0;
    A(171, 7) = 1853.0;
    A(171, 8) = 4444.0;
    A(172, 0) = 635.0;
    A(172, 1) = 8340.0;
    A(172, 2) = 1860.0;
    A(172, 3) = 1055.0;
    A(172, 4) = 4080.0;
    A(172, 5) = 2861.0;
    A(172, 6) = 3596.0;
    A(172, 7) = 1403.0;
    A(172, 8) = 6245.0;
    A(173, 0) = 522.0;
    A(173, 1) = 6986.0;
    A(173, 2) = 741.0;
    A(173, 3) = 855.0;
    A(173, 4) = 4084.0;
    A(173, 5) = 2629.0;
    A(173, 6) = 1352.0;
    A(173, 7) = 1428.0;
    A(173, 8) = 4313.0;
    A(174, 0) = 398.0;
    A(174, 1) = 8256.0;
    A(174, 2) = 775.0;
    A(174, 3) = 789.0;
    A(174, 4) = 5618.0;
    A(174, 5) = 2878.0;
    A(174, 6) = 4523.0;
    A(174, 7) = 1804.0;
    A(174, 8) = 4908.0;
    A(175, 0) = 497.0;
    A(175, 1) = 7270.0;
    A(175, 2) = 1861.0;
    A(175, 3) = 1328.0;
    A(175, 4) = 4186.0;
    A(175, 5) = 2581.0;
    A(175, 6) = 2180.0;
    A(175, 7) = 1462.0;
    A(175, 8) = 5273.0;
    A(176, 0) = 500.0;
    A(176, 1) = 6608.0;
    A(176, 2) = 509.0;
    A(176, 3) = 976.0;
    A(176, 4) = 2680.0;
    A(176, 5) = 2816.0;
    A(176, 6) = 334.0;
    A(176, 7) = 834.0;
    A(176, 8) = 6898.0;
    A(177, 0) = 579.0;
    A(177, 1) = 8309.0;
    A(177, 2) = 1105.0;
    A(177, 3) = 609.0;
    A(177, 4) = 3629.0;
    A(177, 5) = 2582.0;
    A(177, 6) = 2565.0;
    A(177, 7) = 1602.0;
    A(177, 8) = 3301.0;
    A(178, 0) = 885.0;
    A(178, 1) = 13868.0;
    A(178, 2) = 5153.0;
    A(178, 3) = 1960.0;
    A(178, 4) = 4345.0;
    A(178, 5) = 3195.0;
    A(178, 6) = 23567.0;
    A(178, 7) = 3948.0;
    A(178, 8) = 5316.0;
    A(179, 0) = 616.0;
    A(179, 1) = 6812.0;
    A(179, 2) = 2111.0;
    A(179, 3) = 937.0;
    A(179, 4) = 5420.0;
    A(179, 5) = 3028.0;
    A(179, 6) = 4916.0;
    A(179, 7) = 1942.0;
    A(179, 8) = 5402.0;
    A(180, 0) = 526.0;
    A(180, 1) = 9640.0;
    A(180, 2) = 1083.0;
    A(180, 3) = 819.0;
    A(180, 4) = 3820.0;
    A(180, 5) = 2479.0;
    A(180, 6) = 3057.0;
    A(180, 7) = 1129.0;
    A(180, 8) = 6651.0;
    A(181, 0) = 604.0;
    A(181, 1) = 6990.0;
    A(181, 2) = 900.0;
    A(181, 3) = 1608.0;
    A(181, 4) = 4158.0;
    A(181, 5) = 2545.0;
    A(181, 6) = 3402.0;
    A(181, 7) = 1702.0;
    A(181, 8) = 5923.0;
    A(182, 0) = 642.0;
    A(182, 1) = 6934.0;
    A(182, 2) = 732.0;
    A(182, 3) = 643.0;
    A(182, 4) = 4909.0;
    A(182, 5) = 2803.0;
    A(182, 6) = 1079.0;
    A(182, 7) = 1439.0;
    A(182, 8) = 4926.0;
    A(183, 0) = 447.0;
    A(183, 1) = 6235.0;
    A(183, 2) = 593.0;
    A(183, 3) = 783.0;
    A(183, 4) = 3144.0;
    A(183, 5) = 2651.0;
    A(183, 6) = 1435.0;
    A(183, 7) = 1204.0;
    A(183, 8) = 5659.0;
    A(184, 0) = 378.0;
    A(184, 1) = 9897.0;
    A(184, 2) = 2168.0;
    A(184, 3) = 779.0;
    A(184, 4) = 6084.0;
    A(184, 5) = 3047.0;
    A(184, 6) = 5123.0;
    A(184, 7) = 1944.0;
    A(184, 8) = 5448.0;
    A(185, 0) = 404.0;
    A(185, 1) = 9860.0;
    A(185, 2) = 737.0;
    A(185, 3) = 633.0;
    A(185, 4) = 4595.0;
    A(185, 5) = 2728.0;
    A(185, 6) = 475.0;
    A(185, 7) = 837.0;
    A(185, 8) = 7101.0;
    A(186, 0) = 558.0;
    A(186, 1) = 6881.0;
    A(186, 2) = 303.0;
    A(186, 3) = 1072.0;
    A(186, 4) = 2876.0;
    A(186, 5) = 2871.0;
    A(186, 6) = 554.0;
    A(186, 7) = 1133.0;
    A(186, 8) = 4386.0;
    A(187, 0) = 238.0;
    A(187, 1) = 5345.0;
    A(187, 2) = 372.0;
    A(187, 3) = 836.0;
    A(187, 4) = 2117.0;
    A(187, 5) = 2644.0;
    A(187, 6) = 1231.0;
    A(187, 7) = 1059.0;
    A(187, 8) = 5739.0;
    A(188, 0) = 611.0;
    A(188, 1) = 9008.0;
    A(188, 2) = 256.0;
    A(188, 3) = 728.0;
    A(188, 4) = 3512.0;
    A(188, 5) = 2797.0;
    A(188, 6) = 1856.0;
    A(188, 7) = 1416.0;
    A(188, 8) = 3692.0;
    A(189, 0) = 582.0;
    A(189, 1) = 8721.0;
    A(189, 2) = 517.0;
    A(189, 3) = 1039.0;
    A(189, 4) = 2560.0;
    A(189, 5) = 2814.0;
    A(189, 6) = 437.0;
    A(189, 7) = 3800.0;
    A(189, 8) = 7089.0;
    A(190, 0) = 514.0;
    A(190, 1) = 7015.0;
    A(190, 2) = 2043.0;
    A(190, 3) = 1488.0;
    A(190, 4) = 6247.0;
    A(190, 5) = 2804.0;
    A(190, 6) = 4486.0;
    A(190, 7) = 1994.0;
    A(190, 8) = 5160.0;
    A(191, 0) = 634.0;
    A(191, 1) = 10267.0;
    A(191, 2) = 2314.0;
    A(191, 3) = 2459.0;
    A(191, 4) = 5202.0;
    A(191, 5) = 2879.0;
    A(191, 6) = 4837.0;
    A(191, 7) = 4300.0;
    A(191, 8) = 5840.0;
    A(192, 0) = 559.0;
    A(192, 1) = 12135.0;
    A(192, 2) = 2589.0;
    A(192, 3) = 691.0;
    A(192, 4) = 4198.0;
    A(192, 5) = 3539.0;
    A(192, 6) = 8058.0;
    A(192, 7) = 1596.0;
    A(192, 8) = 6324.0;
    A(193, 0) = 593.0;
    A(193, 1) = 11652.0;
    A(193, 2) = 884.0;
    A(193, 3) = 646.0;
    A(193, 4) = 4636.0;
    A(193, 5) = 3128.0;
    A(193, 6) = 730.0;
    A(193, 7) = 1682.0;
    A(193, 8) = 6307.0;
    A(194, 0) = 603.0;
    A(194, 1) = 8672.0;
    A(194, 2) = 97.0;
    A(194, 3) = 1166.0;
    A(194, 4) = 5310.0;
    A(194, 5) = 2416.0;
    A(194, 6) = 438.0;
    A(194, 7) = 1502.0;
    A(194, 8) = 9980.0;
    A(195, 0) = 460.0;
    A(195, 1) = 10176.0;
    A(195, 2) = 3053.0;
    A(195, 3) = 826.0;
    A(195, 4) = 4945.0;
    A(195, 5) = 3044.0;
    A(195, 6) = 8766.0;
    A(195, 7) = 2902.0;
    A(195, 8) = 4982.0;
    A(196, 0) = 293.0;
    A(196, 1) = 9559.0;
    A(196, 2) = 3934.0;
    A(196, 3) = 906.0;
    A(196, 4) = 5606.0;
    A(196, 5) = 3013.0;
    A(196, 6) = 11714.0;
    A(196, 7) = 2158.0;
    A(196, 8) = 5843.0;
    A(197, 0) = 442.0;
    A(197, 1) = 6704.0;
    A(197, 2) = 1469.0;
    A(197, 3) = 1511.0;
    A(197, 4) = 3345.0;
    A(197, 5) = 2779.0;
    A(197, 6) = 1764.0;
    A(197, 7) = 2164.0;
    A(197, 8) = 4565.0;
    A(198, 0) = 639.0;
    A(198, 1) = 8630.0;
    A(198, 2) = 347.0;
    A(198, 3) = 1154.0;
    A(198, 4) = 2000.0;
    A(198, 5) = 2616.0;
    A(198, 6) = 631.0;
    A(198, 7) = 833.0;
    A(198, 8) = 5107.0;
    A(199, 0) = 615.0;
    A(199, 1) = 11660.0;
    A(199, 2) = 2482.0;
    A(199, 3) = 819.0;
    A(199, 4) = 2690.0;
    A(199, 5) = 2787.0;
    A(199, 6) = 7563.0;
    A(199, 7) = 3544.0;
    A(199, 8) = 6154.0;
    A(200, 0) = 455.0;
    A(200, 1) = 6190.0;
    A(200, 2) = 331.0;
    A(200, 3) = 957.0;
    A(200, 4) = 3606.0;
    A(200, 5) = 2453.0;
    A(200, 6) = 1528.0;
    A(200, 7) = 1541.0;
    A(200, 8) = 5537.0;
    A(201, 0) = 483.0;
    A(201, 1) = 6754.0;
    A(201, 2) = 832.0;
    A(201, 3) = 815.0;
    A(201, 4) = 3509.0;
    A(201, 5) = 2388.0;
    A(201, 6) = 1374.0;
    A(201, 7) = 1112.0;
    A(201, 8) = 4892.0;
    A(202, 0) = 530.0;
    A(202, 1) = 5800.0;
    A(202, 2) = 949.0;
    A(202, 3) = 783.0;
    A(202, 4) = 4325.0;
    A(202, 5) = 2965.0;
    A(202, 6) = 2498.0;
    A(202, 7) = 1428.0;
    A(202, 8) = 3980.0;
    A(203, 0) = 580.0;
    A(203, 1) = 6391.0;
    A(203, 2) = 699.0;
    A(203, 3) = 1537.0;
    A(203, 4) = 3353.0;
    A(203, 5) = 2630.0;
    A(203, 6) = 529.0;
    A(203, 7) = 2666.0;
    A(203, 8) = 3708.0;
    A(204, 0) = 538.0;
    A(204, 1) = 10757.0;
    A(204, 2) = 853.0;
    A(204, 3) = 452.0;
    A(204, 4) = 3320.0;
    A(204, 5) = 2728.0;
    A(204, 6) = 2122.0;
    A(204, 7) = 1523.0;
    A(204, 8) = 6962.0;
    A(205, 0) = 600.0;
    A(205, 1) = 7800.0;
    A(205, 2) = 1850.0;
    A(205, 3) = 984.0;
    A(205, 4) = 5030.0;
    A(205, 5) = 2763.0;
    A(205, 6) = 4342.0;
    A(205, 7) = 1849.0;
    A(205, 8) = 5938.0;
    A(206, 0) = 656.0;
    A(206, 1) = 11138.0;
    A(206, 2) = 3919.0;
    A(206, 3) = 566.0;
    A(206, 4) = 2119.0;
    A(206, 5) = 3234.0;
    A(206, 6) = 8640.0;
    A(206, 7) = 3705.0;
    A(206, 8) = 7371.0;
    A(207, 0) = 643.0;
    A(207, 1) = 8087.0;
    A(207, 2) = 519.0;
    A(207, 3) = 1012.0;
    A(207, 4) = 3219.0;
    A(207, 5) = 2439.0;
    A(207, 6) = 766.0;
    A(207, 7) = 1450.0;
    A(207, 8) = 4937.0;
    A(208, 0) = 516.0;
    A(208, 1) = 10509.0;
    A(208, 2) = 1245.0;
    A(208, 3) = 903.0;
    A(208, 4) = 4900.0;
    A(208, 5) = 3128.0;
    A(208, 6) = 1360.0;
    A(208, 7) = 1217.0;
    A(208, 8) = 6470.0;
    A(209, 0) = 583.0;
    A(209, 1) = 11460.0;
    A(209, 2) = 2068.0;
    A(209, 3) = 893.0;
    A(209, 4) = 5938.0;
    A(209, 5) = 3495.0;
    A(209, 6) = 7852.0;
    A(209, 7) = 1604.0;
    A(209, 8) = 5478.0;
    A(210, 0) = 583.0;
    A(210, 1) = 10218.0;
    A(210, 2) = 556.0;
    A(210, 3) = 633.0;
    A(210, 4) = 4505.0;
    A(210, 5) = 3244.0;
    A(210, 6) = 1164.0;
    A(210, 7) = 2281.0;
    A(210, 8) = 6672.0;
    A(211, 0) = 498.0;
    A(211, 1) = 8515.0;
    A(211, 2) = 2586.0;
    A(211, 3) = 1604.0;
    A(211, 4) = 4579.0;
    A(211, 5) = 2995.0;
    A(211, 6) = 7978.0;
    A(211, 7) = 3500.0;
    A(211, 8) = 6453.0;
    A(212, 0) = 638.0;
    A(212, 1) = 13358.0;
    A(212, 2) = 7850.0;
    A(212, 3) = 2498.0;
    A(212, 4) = 8625.0;
    A(212, 5) = 2984.0;
    A(212, 6) = 56745.0;
    A(212, 7) = 3579.0;
    A(212, 8) = 5338.0;
    A(213, 0) = 601.0;
    A(213, 1) = 14220.0;
    A(213, 2) = 4106.0;
    A(213, 3) = 1461.0;
    A(213, 4) = 3514.0;
    A(213, 5) = 3362.0;
    A(213, 6) = 14224.0;
    A(213, 7) = 1818.0;
    A(213, 8) = 5690.0;
    A(214, 0) = 554.0;
    A(214, 1) = 7686.0;
    A(214, 2) = 507.0;
    A(214, 3) = 775.0;
    A(214, 4) = 5561.0;
    A(214, 5) = 2538.0;
    A(214, 6) = 966.0;
    A(214, 7) = 1873.0;
    A(214, 8) = 4463.0;
    A(215, 0) = 632.0;
    A(215, 1) = 8568.0;
    A(215, 2) = 1932.0;
    A(215, 3) = 997.0;
    A(215, 4) = 3215.0;
    A(215, 5) = 3014.0;
    A(215, 6) = 7087.0;
    A(215, 7) = 2964.0;
    A(215, 8) = 5866.0;
    A(216, 0) = 648.0;
    A(216, 1) = 20151.0;
    A(216, 2) = 2530.0;
    A(216, 3) = 625.0;
    A(216, 4) = 3536.0;
    A(216, 5) = 3029.0;
    A(216, 6) = 7273.0;
    A(216, 7) = 2268.0;
    A(216, 8) = 6432.0;
    A(217, 0) = 910.0;
    A(217, 1) = 13135.0;
    A(217, 2) = 2362.0;
    A(217, 3) = 1533.0;
    A(217, 4) = 6430.0;
    A(217, 5) = 2646.0;
    A(217, 6) = 6162.0;
    A(217, 7) = 2394.0;
    A(217, 8) = 5457.0;
    A(218, 0) = 333.0;
    A(218, 1) = 6750.0;
    A(218, 2) = 489.0;
    A(218, 3) = 1327.0;
    A(218, 4) = 3798.0;
    A(218, 5) = 2864.0;
    A(218, 6) = 266.0;
    A(218, 7) = 3095.0;
    A(218, 8) = 7060.0;
    A(219, 0) = 603.0;
    A(219, 1) = 6689.0;
    A(219, 2) = 384.0;
    A(219, 3) = 1698.0;
    A(219, 4) = 1944.0;
    A(219, 5) = 2721.0;
    A(219, 6) = 259.0;
    A(219, 7) = 900.0;
    A(219, 8) = 7565.0;
    A(220, 0) = 554.0;
    A(220, 1) = 7186.0;
    A(220, 2) = 1623.0;
    A(220, 3) = 1297.0;
    A(220, 4) = 4459.0;
    A(220, 5) = 2908.0;
    A(220, 6) = 4843.0;
    A(220, 7) = 1742.0;
    A(220, 8) = 8119.0;
    A(221, 0) = 726.0;
    A(221, 1) = 8263.0;
    A(221, 2) = 338.0;
    A(221, 3) = 752.0;
    A(221, 4) = 4083.0;
    A(221, 5) = 2625.0;
    A(221, 6) = 708.0;
    A(221, 7) = 1451.0;
    A(221, 8) = 4912.0;
    A(222, 0) = 440.0;
    A(222, 1) = 7128.0;
    A(222, 2) = 2559.0;
    A(222, 3) = 1008.0;
    A(222, 4) = 5806.0;
    A(222, 5) = 3069.0;
    A(222, 6) = 3787.0;
    A(222, 7) = 1977.0;
    A(222, 8) = 5853.0;
    A(223, 0) = 509.0;
    A(223, 1) = 10173.0;
    A(223, 2) = 1574.0;
    A(223, 3) = 804.0;
    A(223, 4) = 3299.0;
    A(223, 5) = 2754.0;
    A(223, 6) = 6987.0;
    A(223, 7) = 1866.0;
    A(223, 8) = 5822.0;
    A(224, 0) = 457.0;
    A(224, 1) = 8196.0;
    A(224, 2) = 765.0;
    A(224, 3) = 1671.0;
    A(224, 4) = 5887.0;
    A(224, 5) = 2976.0;
    A(224, 6) = 2681.0;
    A(224, 7) = 2881.0;
    A(224, 8) = 7413.0;
    A(225, 0) = 524.0;
    A(225, 1) = 6760.0;
    A(225, 2) = 812.0;
    A(225, 3) = 568.0;
    A(225, 4) = 2273.0;
    A(225, 5) = 2491.0;
    A(225, 6) = 804.0;
    A(225, 7) = 852.0;
    A(225, 8) = 6062.0;
    A(226, 0) = 890.0;
    A(226, 1) = 14000.0;
    A(226, 2) = 1106.0;
    A(226, 3) = 791.0;
    A(226, 4) = 2238.0;
    A(226, 5) = 2155.0;
    A(226, 6) = 2769.0;
    A(226, 7) = 2135.0;
    A(226, 8) = 5514.0;
    A(227, 0) = 536.0;
    A(227, 1) = 6373.0;
    A(227, 2) = 201.0;
    A(227, 3) = 1344.0;
    A(227, 4) = 2778.0;
    A(227, 5) = 2500.0;
    A(227, 6) = 755.0;
    A(227, 7) = 2089.0;
    A(227, 8) = 6083.0;
    A(228, 0) = 617.0;
    A(228, 1) = 6657.0;
    A(228, 2) = 665.0;
    A(228, 3) = 488.0;
    A(228, 4) = 4399.0;
    A(228, 5) = 2503.0;
    A(228, 6) = 91.0;
    A(228, 7) = 1148.0;
    A(228, 8) = 5187.0;
    A(229, 0) = 584.0;
    A(229, 1) = 6248.0;
    A(229, 2) = 593.0;
    A(229, 3) = 591.0;
    A(229, 4) = 1750.0;
    A(229, 5) = 1701.0;
    A(229, 6) = 155.0;
    A(229, 7) = 1956.0;
    A(229, 8) = 4491.0;
    A(230, 0) = 586.0;
    A(230, 1) = 9462.0;
    A(230, 2) = 1117.0;
    A(230, 3) = 744.0;
    A(230, 4) = 4738.0;
    A(230, 5) = 3058.0;
    A(230, 6) = 480.0;
    A(230, 7) = 1513.0;
    A(230, 8) = 5154.0;
    A(231, 0) = 536.0;
    A(231, 1) = 6479.0;
    A(231, 2) = 563.0;
    A(231, 3) = 1472.0;
    A(231, 4) = 2918.0;
    A(231, 5) = 2914.0;
    A(231, 6) = 1954.0;
    A(231, 7) = 2160.0;
    A(231, 8) = 6029.0;
    A(232, 0) = 491.0;
    A(232, 1) = 8388.0;
    A(232, 2) = 1184.0;
    A(232, 3) = 921.0;
    A(232, 4) = 2967.0;
    A(232, 5) = 2452.0;
    A(232, 6) = 2294.0;
    A(232, 7) = 1688.0;
    A(232, 8) = 4073.0;
    A(233, 0) = 630.0;
    A(233, 1) = 8310.0;
    A(233, 2) = 5158.0;
    A(233, 3) = 1059.0;
    A(233, 4) = 5903.0;
    A(233, 5) = 3781.0;
    A(233, 6) = 17270.0;
    A(233, 7) = 1979.0;
    A(233, 8) = 5638.0;
    A(234, 0) = 536.0;
    A(234, 1) = 8921.0;
    A(234, 2) = 1584.0;
    A(234, 3) = 1268.0;
    A(234, 4) = 4729.0;
    A(234, 5) = 2942.0;
    A(234, 6) = 4573.0;
    A(234, 7) = 2472.0;
    A(234, 8) = 6415.0;
    A(235, 0) = 463.0;
    A(235, 1) = 5674.0;
    A(235, 2) = 617.0;
    A(235, 3) = 1169.0;
    A(235, 4) = 1671.0;
    A(235, 5) = 2554.0;
    A(235, 6) = 373.0;
    A(235, 7) = 793.0;
    A(235, 8) = 4247.0;
    A(236, 0) = 586.0;
    A(236, 1) = 8099.0;
    A(236, 2) = 3413.0;
    A(236, 3) = 687.0;
    A(236, 4) = 5616.0;
    A(236, 5) = 3544.0;
    A(236, 6) = 11069.0;
    A(236, 7) = 2145.0;
    A(236, 8) = 5261.0;
    A(237, 0) = 482.0;
    A(237, 1) = 7807.0;
    A(237, 2) = 694.0;
    A(237, 3) = 638.0;
    A(237, 4) = 3759.0;
    A(237, 5) = 3264.0;
    A(237, 6) = 228.0;
    A(237, 7) = 1420.0;
    A(237, 8) = 5483.0;
    A(238, 0) = 483.0;
    A(238, 1) = 8100.0;
    A(238, 2) = 834.0;
    A(238, 3) = 823.0;
    A(238, 4) = 5185.0;
    A(238, 5) = 2973.0;
    A(238, 6) = 2351.0;
    A(238, 7) = 3366.0;
    A(238, 8) = 6186.0;
    A(239, 0) = 768.0;
    A(239, 1) = 9912.0;
    A(239, 2) = 1590.0;
    A(239, 3) = 1504.0;
    A(239, 4) = 5947.0;
    A(239, 5) = 3343.0;
    A(239, 6) = 5160.0;
    A(239, 7) = 2532.0;
    A(239, 8) = 4535.0;
    A(240, 0) = 469.0;
    A(240, 1) = 9966.0;
    A(240, 2) = 596.0;
    A(240, 3) = 475.0;
    A(240, 4) = 2321.0;
    A(240, 5) = 3026.0;
    A(240, 6) = 165.0;
    A(240, 7) = 1390.0;
    A(240, 8) = 8367.0;
    A(241, 0) = 488.0;
    A(241, 1) = 9981.0;
    A(241, 2) = 355.0;
    A(241, 3) = 633.0;
    A(241, 4) = 4166.0;
    A(241, 5) = 2898.0;
    A(241, 6) = 785.0;
    A(241, 7) = 1670.0;
    A(241, 8) = 6746.0;
    A(242, 0) = 586.0;
    A(242, 1) = 9274.0;
    A(242, 2) = 2467.0;
    A(242, 3) = 998.0;
    A(242, 4) = 5474.0;
    A(242, 5) = 3558.0;
    A(242, 6) = 6152.0;
    A(242, 7) = 2263.0;
    A(242, 8) = 5154.0;
    A(243, 0) = 500.0;
    A(243, 1) = 9321.0;
    A(243, 2) = 198.0;
    A(243, 3) = 485.0;
    A(243, 4) = 4546.0;
    A(243, 5) = 2618.0;
    A(243, 6) = 1985.0;
    A(243, 7) = 3300.0;
    A(243, 8) = 3459.0;
    A(244, 0) = 497.0;
    A(244, 1) = 6637.0;
    A(244, 2) = 468.0;
    A(244, 3) = 1181.0;
    A(244, 4) = 3501.0;
    A(244, 5) = 2653.0;
    A(244, 6) = 1307.0;
    A(244, 7) = 1619.0;
    A(244, 8) = 4646.0;
    A(245, 0) = 496.0;
    A(245, 1) = 8943.0;
    A(245, 2) = 931.0;
    A(245, 3) = 1055.0;
    A(245, 4) = 3558.0;
    A(245, 5) = 2732.0;
    A(245, 6) = 1171.0;
    A(245, 7) = 2016.0;
    A(245, 8) = 4415.0;
    A(246, 0) = 647.0;
    A(246, 1) = 8230.0;
    A(246, 2) = 3476.0;
    A(246, 3) = 981.0;
    A(246, 4) = 6544.0;
    A(246, 5) = 3455.0;
    A(246, 6) = 5730.0;
    A(246, 7) = 1606.0;
    A(246, 8) = 6405.0;
    A(247, 0) = 614.0;
    A(247, 1) = 7614.0;
    A(247, 2) = 1154.0;
    A(247, 3) = 522.0;
    A(247, 4) = 3120.0;
    A(247, 5) = 3028.0;
    A(247, 6) = 1108.0;
    A(247, 7) = 1549.0;
    A(247, 8) = 5587.0;
    A(248, 0) = 664.0;
    A(248, 1) = 8584.0;
    A(248, 2) = 274.0;
    A(248, 3) = 892.0;
    A(248, 4) = 5727.0;
    A(248, 5) = 2471.0;
    A(248, 6) = 845.0;
    A(248, 7) = 2424.0;
    A(248, 8) = 4459.0;
    A(249, 0) = 535.0;
    A(249, 1) = 12449.0;
    A(249, 2) = 615.0;
    A(249, 3) = 1116.0;
    A(249, 4) = 6767.0;
    A(249, 5) = 2529.0;
    A(249, 6) = 2210.0;
    A(249, 7) = 2386.0;
    A(249, 8) = 5677.0;
    A(250, 0) = 664.0;
    A(250, 1) = 8461.0;
    A(250, 2) = 300.0;
    A(250, 3) = 779.0;
    A(250, 4) = 4714.0;
    A(250, 5) = 1728.0;
    A(250, 6) = 529.0;
    A(250, 7) = 1204.0;
    A(250, 8) = 5326.0;
    A(251, 0) = 585.0;
    A(251, 1) = 8343.0;
    A(251, 2) = 2448.0;
    A(251, 3) = 1076.0;
    A(251, 4) = 6680.0;
    A(251, 5) = 2940.0;
    A(251, 6) = 5697.0;
    A(251, 7) = 1943.0;
    A(251, 8) = 5870.0;
    A(252, 0) = 615.0;
    A(252, 1) = 9754.0;
    A(252, 2) = 2201.0;
    A(252, 3) = 1475.0;
    A(252, 4) = 3141.0;
    A(252, 5) = 2596.0;
    A(252, 6) = 5327.0;
    A(252, 7) = 1918.0;
    A(252, 8) = 4923.0;
    A(253, 0) = 652.0;
    A(253, 1) = 7476.0;
    A(253, 2) = 1036.0;
    A(253, 3) = 784.0;
    A(253, 4) = 3872.0;
    A(253, 5) = 2723.0;
    A(253, 6) = 1263.0;
    A(253, 7) = 2036.0;
    A(253, 8) = 5287.0;
    A(254, 0) = 308.0;
    A(254, 1) = 9193.0;
    A(254, 2) = 2966.0;
    A(254, 3) = 437.0;
    A(254, 4) = 4399.0;
    A(254, 5) = 2134.0;
    A(254, 6) = 769.0;
    A(254, 7) = 1503.0;
    A(254, 8) = 6099.0;
    A(255, 0) = 536.0;
    A(255, 1) = 8609.0;
    A(255, 2) = 1969.0;
    A(255, 3) = 894.0;
    A(255, 4) = 5165.0;
    A(255, 5) = 3582.0;
    A(255, 6) = 6956.0;
    A(255, 7) = 2659.0;
    A(255, 8) = 6304.0;
    A(256, 0) = 466.0;
    A(256, 1) = 7584.0;
    A(256, 2) = 969.0;
    A(256, 3) = 1156.0;
    A(256, 4) = 2987.0;
    A(256, 5) = 2680.0;
    A(256, 6) = 1026.0;
    A(256, 7) = 933.0;
    A(256, 8) = 4592.0;
    A(257, 0) = 576.0;
    A(257, 1) = 9855.0;
    A(257, 2) = 1027.0;
    A(257, 3) = 1363.0;
    A(257, 4) = 5097.0;
    A(257, 5) = 2793.0;
    A(257, 6) = 4483.0;
    A(257, 7) = 2306.0;
    A(257, 8) = 5309.0;
    A(258, 0) = 515.0;
    A(258, 1) = 7368.0;
    A(258, 2) = 1022.0;
    A(258, 3) = 1068.0;
    A(258, 4) = 3186.0;
    A(258, 5) = 2772.0;
    A(258, 6) = 1708.0;
    A(258, 7) = 2059.0;
    A(258, 8) = 3709.0;
    A(259, 0) = 195.0;
    A(259, 1) = 7235.0;
    A(259, 2) = 603.0;
    A(259, 3) = 343.0;
    A(259, 4) = 4565.0;
    A(259, 5) = 2502.0;
    A(259, 6) = 1871.0;
    A(259, 7) = 1572.0;
    A(259, 8) = 5488.0;
    A(260, 0) = 475.0;
    A(260, 1) = 5589.0;
    A(260, 2) = 223.0;
    A(260, 3) = 969.0;
    A(260, 4) = 2689.0;
    A(260, 5) = 2927.0;
    A(260, 6) = 879.0;
    A(260, 7) = 1265.0;
    A(260, 8) = 5991.0;
    A(261, 0) = 537.0;
    A(261, 1) = 7605.0;
    A(261, 2) = 2850.0;
    A(261, 3) = 1306.0;
    A(261, 4) = 7119.0;
    A(261, 5) = 3530.0;
    A(261, 6) = 8896.0;
    A(261, 7) = 2243.0;
    A(261, 8) = 5800.0;
    A(262, 0) = 716.0;
    A(262, 1) = 8378.0;
    A(262, 2) = 749.0;
    A(262, 3) = 1014.0;
    A(262, 4) = 4732.0;
    A(262, 5) = 3278.0;
    A(262, 6) = 691.0;
    A(262, 7) = 1873.0;
    A(262, 8) = 3835.0;
    A(263, 0) = 644.0;
    A(263, 1) = 11622.0;
    A(263, 2) = 1232.0;
    A(263, 3) = 490.0;
    A(263, 4) = 3459.0;
    A(263, 5) = 2729.0;
    A(263, 6) = 3276.0;
    A(263, 7) = 2234.0;
    A(263, 8) = 6309.0;
    A(264, 0) = 843.0;
    A(264, 1) = 13838.0;
    A(264, 2) = 352.0;
    A(264, 3) = 1107.0;
    A(264, 4) = 4160.0;
    A(264, 5) = 2439.0;
    A(264, 6) = 1004.0;
    A(264, 7) = 3179.0;
    A(264, 8) = 5656.0;
    A(265, 0) = 541.0;
    A(265, 1) = 9466.0;
    A(265, 2) = 1631.0;
    A(265, 3) = 969.0;
    A(265, 4) = 6228.0;
    A(265, 5) = 2340.0;
    A(265, 6) = 5528.0;
    A(265, 7) = 3900.0;
    A(265, 8) = 4942.0;
    A(266, 0) = 488.0;
    A(266, 1) = 6321.0;
    A(266, 2) = 236.0;
    A(266, 3) = 1032.0;
    A(266, 4) = 2938.0;
    A(266, 5) = 2707.0;
    A(266, 6) = 1301.0;
    A(266, 7) = 1136.0;
    A(266, 8) = 7720.0;
    A(267, 0) = 398.0;
    A(267, 1) = 6898.0;
    A(267, 2) = 1337.0;
    A(267, 3) = 1197.0;
    A(267, 4) = 5387.0;
    A(267, 5) = 2938.0;
    A(267, 6) = 4295.0;
    A(267, 7) = 1509.0;
    A(267, 8) = 6873.0;
    A(268, 0) = 903.0;
    A(268, 1) = 14465.0;
    A(268, 2) = 2416.0;
    A(268, 3) = 1099.0;
    A(268, 4) = 5489.0;
    A(268, 5) = 2794.0;
    A(268, 6) = 8818.0;
    A(268, 7) = 3347.0;
    A(268, 8) = 5489.0;
    A(269, 0) = 910.0;
    A(269, 1) = 17158.0;
    A(269, 2) = 3726.0;
    A(269, 3) = 1619.0;
    A(269, 4) = 8299.0;
    A(269, 5) = 3371.0;
    A(269, 6) = 14226.0;
    A(269, 7) = 4600.0;
    A(269, 8) = 6063.0;
    A(270, 0) = 850.0;
    A(270, 1) = 16048.0;
    A(270, 2) = 2117.0;
    A(270, 3) = 1065.0;
    A(270, 4) = 5224.0;
    A(270, 5) = 2709.0;
    A(270, 6) = 6446.0;
    A(270, 7) = 1964.0;
    A(270, 8) = 7270.0;
    A(271, 0) = 855.0;
    A(271, 1) = 15547.0;
    A(271, 2) = 532.0;
    A(271, 3) = 1026.0;
    A(271, 4) = 5662.0;
    A(271, 5) = 2719.0;
    A(271, 6) = 2684.0;
    A(271, 7) = 3300.0;
    A(271, 8) = 5821.0;
    A(272, 0) = 843.0;
    A(272, 1) = 14303.0;
    A(272, 2) = 1035.0;
    A(272, 3) = 964.0;
    A(272, 4) = 5010.0;
    A(272, 5) = 2611.0;
    A(272, 6) = 3748.0;
    A(272, 7) = 1703.0;
    A(272, 8) = 5335.0;
    A(273, 0) = 732.0;
    A(273, 1) = 12931.0;
    A(273, 2) = 1052.0;
    A(273, 3) = 912.0;
    A(273, 4) = 3313.0;
    A(273, 5) = 2722.0;
    A(273, 6) = 3457.0;
    A(273, 7) = 2255.0;
    A(273, 8) = 5703.0;
    A(274, 0) = 391.0;
    A(274, 1) = 9560.0;
    A(274, 2) = 801.0;
    A(274, 3) = 939.0;
    A(274, 4) = 3742.0;
    A(274, 5) = 2626.0;
    A(274, 6) = 817.0;
    A(274, 7) = 2535.0;
    A(274, 8) = 7715.0;
    A(275, 0) = 542.0;
    A(275, 1) = 6896.0;
    A(275, 2) = 1084.0;
    A(275, 3) = 1614.0;
    A(275, 4) = 5958.0;
    A(275, 5) = 2456.0;
    A(275, 6) = 2262.0;
    A(275, 7) = 2237.0;
    A(275, 8) = 5591.0;
    A(276, 0) = 575.0;
    A(276, 1) = 6697.0;
    A(276, 2) = 1219.0;
    A(276, 3) = 372.0;
    A(276, 4) = 3683.0;
    A(276, 5) = 3230.0;
    A(276, 6) = 1832.0;
    A(276, 7) = 1386.0;
    A(276, 8) = 4907.0;
    A(277, 0) = 808.0;
    A(277, 1) = 10183.0;
    A(277, 2) = 2715.0;
    A(277, 3) = 1170.0;
    A(277, 4) = 6634.0;
    A(277, 5) = 2710.0;
    A(277, 6) = 9577.0;
    A(277, 7) = 4800.0;
    A(277, 8) = 5901.0;
    A(278, 0) = 570.0;
    A(278, 1) = 6697.0;
    A(278, 2) = 700.0;
    A(278, 3) = 384.0;
    A(278, 4) = 2017.0;
    A(278, 5) = 3022.0;
    A(278, 6) = 52.0;
    A(278, 7) = 1100.0;
    A(278, 8) = 4055.0;
    A(279, 0) = 442.0;
    A(279, 1) = 8121.0;
    A(279, 2) = 593.0;
    A(279, 3) = 450.0;
    A(279, 4) = 3458.0;
    A(279, 5) = 2557.0;
    A(279, 6) = 268.0;
    A(279, 7) = 1316.0;
    A(279, 8) = 4765.0;
    A(280, 0) = 524.0;
    A(280, 1) = 5722.0;
    A(280, 2) = 394.0;
    A(280, 3) = 1035.0;
    A(280, 4) = 1922.0;
    A(280, 5) = 2652.0;
    A(280, 6) = 68.0;
    A(280, 7) = 937.0;
    A(280, 8) = 6213.0;
    A(281, 0) = 508.0;
    A(281, 1) = 6534.0;
    A(281, 2) = 1445.0;
    A(281, 3) = 1197.0;
    A(281, 4) = 4401.0;
    A(281, 5) = 2858.0;
    A(281, 6) = 2826.0;
    A(281, 7) = 1389.0;
    A(281, 8) = 6585.0;
    A(282, 0) = 385.0;
    A(282, 1) = 6528.0;
    A(282, 2) = 846.0;
    A(282, 3) = 759.0;
    A(282, 4) = 4316.0;
    A(282, 5) = 2673.0;
    A(282, 6) = 1393.0;
    A(282, 7) = 1359.0;
    A(282, 8) = 4648.0;
    A(283, 0) = 276.0;
    A(283, 1) = 7983.0;
    A(283, 2) = 1041.0;
    A(283, 3) = 556.0;
    A(283, 4) = 6271.0;
    A(283, 5) = 2651.0;
    A(283, 6) = 465.0;
    A(283, 7) = 1324.0;
    A(283, 8) = 5204.0;
    A(284, 0) = 545.0;
    A(284, 1) = 5938.0;
    A(284, 2) = 830.0;
    A(284, 3) = 1038.0;
    A(284, 4) = 5634.0;
    A(284, 5) = 2874.0;
    A(284, 6) = 2672.0;
    A(284, 7) = 1819.0;
    A(284, 8) = 5056.0;
    A(285, 0) = 574.0;
    A(285, 1) = 6927.0;
    A(285, 2) = 497.0;
    A(285, 3) = 869.0;
    A(285, 4) = 5534.0;
    A(285, 5) = 2774.0;
    A(285, 6) = 2988.0;
    A(285, 7) = 1517.0;
    A(285, 8) = 4722.0;
    A(286, 0) = 524.0;
    A(286, 1) = 7882.0;
    A(286, 2) = 1877.0;
    A(286, 3) = 1225.0;
    A(286, 4) = 6172.0;
    A(286, 5) = 3078.0;
    A(286, 6) = 1983.0;
    A(286, 7) = 1536.0;
    A(286, 8) = 5384.0;
    A(287, 0) = 453.0;
    A(287, 1) = 8039.0;
    A(287, 2) = 710.0;
    A(287, 3) = 1212.0;
    A(287, 4) = 6159.0;
    A(287, 5) = 3525.0;
    A(287, 6) = 3466.0;
    A(287, 7) = 1514.0;
    A(287, 8) = 5289.0;
    A(288, 0) = 544.0;
    A(288, 1) = 6343.0;
    A(288, 2) = 577.0;
    A(288, 3) = 892.0;
    A(288, 4) = 3828.0;
    A(288, 5) = 2709.0;
    A(288, 6) = 1634.0;
    A(288, 7) = 1737.0;
    A(288, 8) = 5932.0;
    A(289, 0) = 648.0;
    A(289, 1) = 23640.0;
    A(289, 2) = 2610.0;
    A(289, 3) = 835.0;
    A(289, 4) = 3110.0;
    A(289, 5) = 3029.0;
    A(289, 6) = 7865.0;
    A(289, 7) = 1729.0;
    A(289, 8) = 6158.0;
    A(290, 0) = 575.0;
    A(290, 1) = 8405.0;
    A(290, 2) = 612.0;
    A(290, 3) = 540.0;
    A(290, 4) = 2740.0;
    A(290, 5) = 3169.0;
    A(290, 6) = 1271.0;
    A(290, 7) = 1200.0;
    A(290, 8) = 4677.0;
    A(291, 0) = 542.0;
    A(291, 1) = 6578.0;
    A(291, 2) = 505.0;
    A(291, 3) = 418.0;
    A(291, 4) = 1532.0;
    A(291, 5) = 2672.0;
    A(291, 6) = 147.0;
    A(291, 7) = 1460.0;
    A(291, 8) = 3744.0;
    A(292, 0) = 625.0;
    A(292, 1) = 8474.0;
    A(292, 2) = 342.0;
    A(292, 3) = 1395.0;
    A(292, 4) = 4427.0;
    A(292, 5) = 2155.0;
    A(292, 6) = 1579.0;
    A(292, 7) = 1630.0;
    A(292, 8) = 5672.0;
    A(293, 0) = 548.0;
    A(293, 1) = 7670.0;
    A(293, 2) = 1040.0;
    A(293, 3) = 689.0;
    A(293, 4) = 6951.0;
    A(293, 5) = 3144.0;
    A(293, 6) = 5080.0;
    A(293, 7) = 2851.0;
    A(293, 8) = 4474.0;
    A(294, 0) = 808.0;
    A(294, 1) = 7770.0;
    A(294, 2) = 539.0;
    A(294, 3) = 1162.0;
    A(294, 4) = 4730.0;
    A(294, 5) = 2546.0;
    A(294, 6) = 4297.0;
    A(294, 7) = 4000.0;
    A(294, 8) = 4887.0;
    A(295, 0) = 404.0;
    A(295, 1) = 8029.0;
    A(295, 2) = 370.0;
    A(295, 3) = 1161.0;
    A(295, 4) = 5530.0;
    A(295, 5) = 2790.0;
    A(295, 6) = 2181.0;
    A(295, 7) = 1936.0;
    A(295, 8) = 6021.0;
    A(296, 0) = 440.0;
    A(296, 1) = 7442.0;
    A(296, 2) = 1189.0;
    A(296, 3) = 1493.0;
    A(296, 4) = 5588.0;
    A(296, 5) = 3044.0;
    A(296, 6) = 5040.0;
    A(296, 7) = 2943.0;
    A(296, 8) = 7256.0;
    A(297, 0) = 557.0;
    A(297, 1) = 5527.0;
    A(297, 2) = 453.0;
    A(297, 3) = 630.0;
    A(297, 4) = 3550.0;
    A(297, 5) = 3012.0;
    A(297, 6) = 1226.0;
    A(297, 7) = 1401.0;
    A(297, 8) = 4353.0;
    A(298, 0) = 467.0;
    A(298, 1) = 5717.0;
    A(298, 2) = 343.0;
    A(298, 3) = 822.0;
    A(298, 4) = 2537.0;
    A(298, 5) = 2899.0;
    A(298, 6) = 63.0;
    A(298, 7) = 669.0;
    A(298, 8) = 4772.0;
    A(299, 0) = 518.0;
    A(299, 1) = 7767.0;
    A(299, 2) = 1738.0;
    A(299, 3) = 998.0;
    A(299, 4) = 5323.0;
    A(299, 5) = 2852.0;
    A(299, 6) = 4389.0;
    A(299, 7) = 1952.0;
    A(299, 8) = 4534.0;
    A(300, 0) = 501.0;
    A(300, 1) = 7110.0;
    A(300, 2) = 1148.0;
    A(300, 3) = 999.0;
    A(300, 4) = 5348.0;
    A(300, 5) = 2795.0;
    A(300, 6) = 1632.0;
    A(300, 7) = 1141.0;
    A(300, 8) = 5464.0;
    A(301, 0) = 636.0;
    A(301, 1) = 10616.0;
    A(301, 2) = 1372.0;
    A(301, 3) = 1181.0;
    A(301, 4) = 4786.0;
    A(301, 5) = 3311.0;
    A(301, 6) = 5029.0;
    A(301, 7) = 1646.0;
    A(301, 8) = 5772.0;
    A(302, 0) = 589.0;
    A(302, 1) = 8548.0;
    A(302, 2) = 1259.0;
    A(302, 3) = 1400.0;
    A(302, 4) = 4397.0;
    A(302, 5) = 2685.0;
    A(302, 6) = 4889.0;
    A(302, 7) = 3131.0;
    A(302, 8) = 6147.0;
    A(303, 0) = 530.0;
    A(303, 1) = 7498.0;
    A(303, 2) = 1581.0;
    A(303, 3) = 1080.0;
    A(303, 4) = 3758.0;
    A(303, 5) = 2628.0;
    A(303, 6) = 4248.0;
    A(303, 7) = 2024.0;
    A(303, 8) = 7115.0;
    A(304, 0) = 470.0;
    A(304, 1) = 6464.0;
    A(304, 2) = 674.0;
    A(304, 3) = 1014.0;
    A(304, 4) = 4723.0;
    A(304, 5) = 2390.0;
    A(304, 6) = 1432.0;
    A(304, 7) = 1090.0;
    A(304, 8) = 4900.0;
    A(305, 0) = 500.0;
    A(305, 1) = 7298.0;
    A(305, 2) = 672.0;
    A(305, 3) = 955.0;
    A(305, 4) = 3460.0;
    A(305, 5) = 3283.0;
    A(305, 6) = 404.0;
    A(305, 7) = 631.0;
    A(305, 8) = 7327.0;
    A(306, 0) = 548.0;
    A(306, 1) = 6744.0;
    A(306, 2) = 391.0;
    A(306, 3) = 400.0;
    A(306, 4) = 4592.0;
    A(306, 5) = 2970.0;
    A(306, 6) = 858.0;
    A(306, 7) = 1750.0;
    A(306, 8) = 5226.0;
    A(307, 0) = 821.0;
    A(307, 1) = 10503.0;
    A(307, 2) = 1079.0;
    A(307, 3) = 964.0;
    A(307, 4) = 4153.0;
    A(307, 5) = 2498.0;
    A(307, 6) = 2962.0;
    A(307, 7) = 1559.0;
    A(307, 8) = 5819.0;
    A(308, 0) = 768.0;
    A(308, 1) = 9015.0;
    A(308, 2) = 517.0;
    A(308, 3) = 752.0;
    A(308, 4) = 3817.0;
    A(308, 5) = 2332.0;
    A(308, 6) = 1557.0;
    A(308, 7) = 1464.0;
    A(308, 8) = 4571.0;
    A(309, 0) = 336.0;
    A(309, 1) = 7143.0;
    A(309, 2) = 260.0;
    A(309, 3) = 1092.0;
    A(309, 4) = 2407.0;
    A(309, 5) = 2696.0;
    A(309, 6) = 87.0;
    A(309, 7) = 1410.0;
    A(309, 8) = 7599.0;
    A(310, 0) = 615.0;
    A(310, 1) = 7295.0;
    A(310, 2) = 807.0;
    A(310, 3) = 1135.0;
    A(310, 4) = 4133.0;
    A(310, 5) = 2747.0;
    A(310, 6) = 2097.0;
    A(310, 7) = 1474.0;
    A(310, 8) = 5023.0;
    A(311, 0) = 543.0;
    A(311, 1) = 7778.0;
    A(311, 2) = 210.0;
    A(311, 3) = 1132.0;
    A(311, 4) = 3094.0;
    A(311, 5) = 2128.0;
    A(311, 6) = 511.0;
    A(311, 7) = 2800.0;
    A(311, 8) = 5563.0;
    A(312, 0) = 412.0;
    A(312, 1) = 6106.0;
    A(312, 2) = 538.0;
    A(312, 3) = 1166.0;
    A(312, 4) = 3018.0;
    A(312, 5) = 2867.0;
    A(312, 6) = 1141.0;
    A(312, 7) = 1248.0;
    A(312, 8) = 6259.0;
    A(313, 0) = 631.0;
    A(313, 1) = 13724.0;
    A(313, 2) = 4361.0;
    A(313, 3) = 1317.0;
    A(313, 4) = 8236.0;
    A(313, 5) = 3635.0;
    A(313, 6) = 21701.0;
    A(313, 7) = 1578.0;
    A(313, 8) = 6072.0;
    A(314, 0) = 569.0;
    A(314, 1) = 10024.0;
    A(314, 2) = 1218.0;
    A(314, 3) = 789.0;
    A(314, 4) = 2434.0;
    A(314, 5) = 2995.0;
    A(314, 6) = 318.0;
    A(314, 7) = 946.0;
    A(314, 8) = 5656.0;
    A(315, 0) = 347.0;
    A(315, 1) = 7881.0;
    A(315, 2) = 925.0;
    A(315, 3) = 700.0;
    A(315, 4) = 3351.0;
    A(315, 5) = 2889.0;
    A(315, 6) = 3000.0;
    A(315, 7) = 1900.0;
    A(315, 8) = 4407.0;
    A(316, 0) = 308.0;
    A(316, 1) = 7642.0;
    A(316, 2) = 818.0;
    A(316, 3) = 442.0;
    A(316, 4) = 3496.0;
    A(316, 5) = 2749.0;
    A(316, 6) = 761.0;
    A(316, 7) = 1654.0;
    A(316, 8) = 4300.0;
    A(317, 0) = 509.0;
    A(317, 1) = 10512.0;
    A(317, 2) = 375.0;
    A(317, 3) = 1783.0;
    A(317, 4) = 5201.0;
    A(317, 5) = 3224.0;
    A(317, 6) = 2888.0;
    A(317, 7) = 3772.0;
    A(317, 8) = 7992.0;
    A(318, 0) = 542.0;
    A(318, 1) = 6576.0;
    A(318, 2) = 791.0;
    A(318, 3) = 308.0;
    A(318, 4) = 2450.0;
    A(318, 5) = 3002.0;
    A(318, 6) = 422.0;
    A(318, 7) = 1271.0;
    A(318, 8) = 4740.0;
    A(319, 0) = 494.0;
    A(319, 1) = 7061.0;
    A(319, 2) = 806.0;
    A(319, 3) = 1164.0;
    A(319, 4) = 3933.0;
    A(319, 5) = 2981.0;
    A(319, 6) = 2987.0;
    A(319, 7) = 1508.0;
    A(319, 8) = 6036.0;
    A(320, 0) = 456.0;
    A(320, 1) = 6404.0;
    A(320, 2) = 549.0;
    A(320, 3) = 1179.0;
    A(320, 4) = 2793.0;
    A(320, 5) = 2747.0;
    A(320, 6) = 599.0;
    A(320, 7) = 1126.0;
    A(320, 8) = 6805.0;
    A(321, 0) = 558.0;
    A(321, 1) = 7284.0;
    A(321, 2) = 860.0;
    A(321, 3) = 464.0;
    A(321, 4) = 3097.0;
    A(321, 5) = 2906.0;
    A(321, 6) = 196.0;
    A(321, 7) = 726.0;
    A(321, 8) = 3288.0;
    A(322, 0) = 597.0;
    A(322, 1) = 7927.0;
    A(322, 2) = 1445.0;
    A(322, 3) = 1115.0;
    A(322, 4) = 4532.0;
    A(322, 5) = 3112.0;
    A(322, 6) = 4545.0;
    A(322, 7) = 1923.0;
    A(322, 8) = 6174.0;
    A(323, 0) = 564.0;
    A(323, 1) = 6858.0;
    A(323, 2) = 1099.0;
    A(323, 3) = 1423.0;
    A(323, 4) = 2904.0;
    A(323, 5) = 2876.0;
    A(323, 6) = 1077.0;
    A(323, 7) = 2668.0;
    A(323, 8) = 5390.0;
    A(324, 0) = 562.0;
    A(324, 1) = 8715.0;
    A(324, 2) = 1805.0;
    A(324, 3) = 680.0;
    A(324, 4) = 3643.0;
    A(324, 5) = 3299.0;
    A(324, 6) = 1784.0;
    A(324, 7) = 910.0;
    A(324, 8) = 5040.0;
    A(325, 0) = 535.0;
    A(325, 1) = 6440.0;
    A(325, 2) = 317.0;
    A(325, 3) = 1106.0;
    A(325, 4) = 3731.0;
    A(325, 5) = 2491.0;
    A(325, 6) = 996.0;
    A(325, 7) = 2140.0;
    A(325, 8) = 4986.0;
    A(326, 0) = 540.0;
    A(326, 1) = 8371.0;
    A(326, 2) = 713.0;
    A(326, 3) = 440.0;
    A(326, 4) = 2267.0;
    A(326, 5) = 2903.0;
    A(326, 6) = 1022.0;
    A(326, 7) = 842.0;
    A(326, 8) = 4946.0;
    A(327, 0) = 570.0;
    A(327, 1) = 7021.0;
    A(327, 2) = 1097.0;
    A(327, 3) = 938.0;
    A(327, 4) = 3374.0;
    A(327, 5) = 2920.0;
    A(327, 6) = 2797.0;
    A(327, 7) = 1327.0;
    A(327, 8) = 3894.0;
    A(328, 0) = 608.0;
    A(328, 1) = 7875.0;
    A(328, 2) = 212.0;
    A(328, 3) = 1179.0;
    A(328, 4) = 2768.0;
    A(328, 5) = 2387.0;
    A(328, 6) = 122.0;
    A(328, 7) = 918.0;
    A(328, 8) = 4694.0;
#pragma endregion

    // const size_type nRows = 10;
    // const size_type nColumns = 2;

    // Matrix A("A", nColumns, nRows);

    // A(0ULL, 0ULL) = 2.5; A(1ULL, 0ULL) = 2.4;
    // A(0ULL, 1ULL) = 0.5; A(1ULL, 1ULL) = 0.7;
    // A(0ULL, 2ULL) = 2.2; A(1ULL, 2ULL) = 2.9;
    // A(0ULL, 3ULL) = 1.9; A(1ULL, 3ULL) = 2.2;
    // A(0ULL, 4ULL) = 3.1; A(1ULL, 4ULL) = 3.0;
    // A(0ULL, 5ULL) = 2.3; A(1ULL, 5ULL) = 2.7;
    // A(0ULL, 6ULL) = 2.0; A(1ULL, 6ULL) = 1.6;
    // A(0ULL, 7ULL) = 1.0; A(1ULL, 7ULL) = 1.1;
    // A(0ULL, 8ULL) = 1.5; A(1ULL, 8ULL) = 1.6;
    // A(0ULL, 9ULL) = 1.1; A(1ULL, 9ULL) = 0.9;

    // std::cout << A << std::endl;

    Statistics::Analyze::PCAResult<Matrix, Matrix, Matrix> pca_results = Statistics::Analyze::PrincipalComponentsAnalysis<double, ExecutionSpace>(A);

    Matrix eigenVectors = pca_results.EigenVectors;
    Matrix eigenValues  = pca_results.EigenValues;
    Matrix pCAResults   = pca_results.PCAResults;

    std::cout << eigenVectors << std::endl;
    std::cout << eigenValues << std::endl;

    std::cout << pCAResults << std::endl;

    //			eigenVectors/Component	V	[9x9]
    //			9	0.995145	-0.022933	0.00137187	-0.0876895	0.00941882	-0.0168656	0.000598585	-0.00503159	0.0327178
    //			8	-0.042138	-0.0121185	0.24136	-0.266827	0.0415077	-0.929159	-0.0159493	-0.0187807	0.054398
    //			7	0.0814085	0.0266878	0.137061	0.94478	-0.0135454	-0.241155	-0.0429604	-0.127117	-0.070161
    //			6	-0.00118662	-0.0486382	0.929493	-0.0539762	-0.0922351	0.253188	-0.167554	0.173348	0.00515218
    //			5	0.0162782	-0.0838423	-0.159076	0.116014	-0.14665	-0.106256	0.00867376	0.954262	-0.102241
    //			4	-0.0263107	-0.177751	-0.0265616	0.0990354	-0.038397	0.0216394	0.0277547	0.0690328	0.974536
    //			3	0.0066923	0.0826419	-0.027761	-0.0376109	-0.971532	-0.0415077	0.151027	-0.149572	-0.0127433
    //			2	0.0154595	0.937207	-0.0205399	-0.0109019	0.0187573	-0.00139588	-0.282261	0.103848	0.17336
    //			1	0.00641635	0.269142	0.178319	0.0281343	0.149302	0.0251909	0.93086	0.069824	0.0251308
    //
    //
    //	Proportion		Eigenvalue	D	[9x9]
    //	100.00%	0.03%	10962.60	10962.6	0	0	0	0	0	0	0	0
    //	99.97%	0.21%	66995.90	0	66995.9	0	0	0	0	0	0	0
    //	99.76%	0.29%	92809.90	0	0	92809.9	0	0	0	0	0	0
    //	99.47%	0.74%	240852.00	0	0	0	240852	0	0	0	0	0
    //	98.73%	1.48%	478338.00	0	0	0	0	478338	0	0	0	0
    //	97.26%	3.32%	1076360.00	0	0	0	0	0	1.08E+06	0	0	0
    //	93.94%	5.05%	1638040.00	0	0	0	0	0	0	1.64E+06	0	0
    //	88.88%	13.59%	4408000.00	0	0	0	0	0	0	0	4.41E+06	0
    //	75.29%	75.29%	24413700.00	0	0	0	0	0	0	0	0	2.44E+07
    //			32426058.40
}

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

template<class ExecutionSpace>
static void TestNelderMead()
{
    const int n = 2;

    const double reqmin = 1.0E-016;
    const int    konvge = 1;
    const int    kcount = 500;

    Kokkos::View<double*, ExecutionSpace> step("step", n);
    step[0] = 1.0;
    step[1] = 1.0;

    Kokkos::View<double*, ExecutionSpace> x0("x0", n);
    x0[0] = -3.0;
    x0[1] = 2.0;

    Kokkos::View<double*, ExecutionSpace> xmin("xmin", n);
    xmin[0] = -10.0;
    xmin[1] = -10.0;

    Kokkos::View<double*, ExecutionSpace> xmax("xmax", n);
    xmax[0] = 10.0;
    xmax[1] = 10.0;

    NelderMeadOptions<double, ExecutionSpace> options(reqmin, konvge, kcount, step);

    rosenbrock<double, ExecutionSpace> func;

    typedef decltype(func) rosenbrock_t;

    NelderMeadResults<double, ExecutionSpace> results = NelderMead<double, ExecutionSpace, rosenbrock_t>::Solve(options, func, x0, xmin, xmax);

    std::cout << "ICount:" << results.ICount << std::endl;
    std::cout << "NumRes:" << results.NumRes << std::endl;
    std::cout << "IFault:" << results.IFault << std::endl;
    std::cout << "YNewLo:" << results.YNewLo << std::endl;
    std::cout << "XMin(0):" << results.XMin(0) << std::endl;
    std::cout << "XMin(1):" << results.XMin(1) << std::endl;
}

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

template<class ExecutionSpace>
static void TestCombinations()
{

    // typedef std::vector<int>          Set;
    // typedef std::vector<Set>          SetOfSets;
    // typedef SetOfSets::const_iterator SetOfSetsCIt;
    // const SetOfSets                   data = {{2, 4}, {1, 3, 8}, {7, 5}};
    //{
    //    std::cout << "First to last-------" << std::endl;
    //    typedef Combinations<SetOfSetsCIt> Combinations;
    //    Combinations                       cmbs = make_combinations(data);
    //    {
    //        std::cout << "Forward:" << std::endl;
    //        for (const auto& combination : cmbs)
    //        {
    //            for (const auto& elemIt : combination)
    //                std::cout << *elemIt << " ";
    //            std::cout << std::endl;
    //        }
    //    }
    //    {
    //        std::cout << "Reverse:" << std::endl;
    //        for (Combinations::const_reverse_iterator combIt = cmbs.crbegin(); combIt != cmbs.crend(); ++combIt)
    //        {
    //            for (const auto& elemIt : *combIt)
    //                std::cout << *elemIt << " ";
    //            std::cout << std::endl;
    //        }
    //    }
    //}
    //{
    //    std::cout << "Last to first-------" << std::endl;
    //    typedef SetOfSets::const_reverse_iterator SetOfSetsCRIt;
    //    typedef Combinations<SetOfSetsCRIt>       Combinations;
    //    Combinations                              cmbs(data.crbegin(), data.crend());
    //    {
    //        std::cout << "Forward:" << std::endl;
    //        for (Combinations::const_iterator cmbIt = cmbs.begin(); cmbIt != cmbs.end(); ++cmbIt)
    //        {
    //            Combinations::Combination c = *cmbIt;
    //            std::reverse(c.begin(), c.end());
    //            for (const auto& it : c)
    //                std::cout << *it << " ";
    //            std::cout << std::endl;
    //        }
    //    }
    //    {
    //        std::cout << "Reverse:" << std::endl;
    //        for (Combinations::const_reverse_iterator cmbIt = cmbs.crbegin(); cmbIt != cmbs.crend(); ++cmbIt)
    //        {
    //            Combinations::Combination c = *cmbIt;
    //            std::reverse(c.begin(), c.end());
    //            for (const auto& it : c)
    //                std::cout << *it << " ";
    //            std::cout << std::endl;
    //        }
    //    }
    //}
}

template<class ExecutionSpace>
static void TestCombinations2()
{
    auto km = ValueLimits<double>(0.0001, 0.01);
    auto kF = ValueLimits<double>(10.0, 1000.0);
    auto kf = ValueLimits<double>(0.01, 10.0);
    auto ye = ValueLimits<double>(100.0, 1000.0);
    auto LF = ValueLimits<double>(50.0, 250.0);
    auto Lf = ValueLimits<double>(10.0, 150.0);

    const int32 n = 3;

    typedef std::vector<double>       Set;
    typedef std::vector<Set>          SetOfSets;
    typedef SetOfSets::const_iterator SetOfSetsCIt;

    auto kmV = std::LinearSpacing<double>(km.GetLower(), km.GetUpper(), n + 2);
    auto kFV = std::LinearSpacing<double>(kF.GetLower(), kF.GetUpper(), n + 2);
    auto kfV = std::LinearSpacing<double>(kf.GetLower(), kf.GetUpper(), n + 2);
    auto yeV = std::LinearSpacing<double>(ye.GetLower(), ye.GetUpper(), n + 2);
    auto LFV = std::LinearSpacing<double>(LF.GetLower(), LF.GetUpper(), n + 2);
    auto LfV = std::LinearSpacing<double>(Lf.GetLower(), Lf.GetUpper(), n + 2);

    kmV.erase(kmV.begin(), kmV.begin() + 1);
    kmV.erase(kmV.end() - 1, kmV.end());

    kFV.erase(kFV.begin(), kFV.begin() + 1);
    kFV.erase(kFV.end() - 1, kFV.end());

    kfV.erase(kfV.begin(), kfV.begin() + 1);
    kfV.erase(kfV.end() - 1, kfV.end());

    yeV.erase(yeV.begin(), yeV.begin() + 1);
    yeV.erase(yeV.end() - 1, yeV.end());

    LFV.erase(LFV.begin(), LFV.begin() + 1);
    LFV.erase(LFV.end() - 1, LFV.end());

    LfV.erase(LfV.begin(), LfV.begin() + 1);
    LfV.erase(LfV.end() - 1, LfV.end());

    for (const auto& item : kmV)
    {
        std::cout << std::to_string(item) << " ";
    }
    std::cout << std::endl;
    for (const auto& item : kFV)
    {
        std::cout << std::to_string(item) << " ";
    }
    std::cout << std::endl;
    for (const auto& item : kfV)
    {
        std::cout << std::to_string(item) << " ";
    }
    std::cout << std::endl;
    for (const auto& item : yeV)
    {
        std::cout << std::to_string(item) << " ";
    }
    std::cout << std::endl;
    for (const auto& item : LFV)
    {
        std::cout << std::to_string(item) << " ";
    }
    std::cout << std::endl;
    for (const auto& item : LfV)
    {
        std::cout << std::to_string(item) << " ";
    }
    std::cout << std::endl;

    const SetOfSets data = {kmV, kFV, kfV, yeV, LFV, LfV};

    {
        typedef NumericalMethods::Algorithms::Internal::Combinations<SetOfSetsCIt> Combinations;
        Combinations                                                               cmbs = NumericalMethods::Algorithms::Internal::make_combinations(data);

        std::cout << "km"
                  << " "
                  << "kF"
                  << " "
                  << "kf"
                  << " "
                  << "ye"
                  << " "
                  << "LF"
                  << " "
                  << "Lf" << std::endl;

        for (const auto& combination : cmbs)
        {
            for (const auto& elemIt : combination)
            {
                std::cout << std::to_string(*elemIt) << " ";
            }
            std::cout << std::endl;
        }

        // int swarm_index    = 0;
        // int particle_index = 0;
        // int parameter_index;

        // for (const auto& combination : cmbs)
        //{
        //    parameter_index = 0;

        //    for (const auto& elemIt : combination)
        //    {
        //        // std::cout << std::to_string(*elemIt) << " ";
        //        std::cout << std::to_string(swarm_index) << " " << std::to_string(particle_index) << " " << std::to_string(parameter_index) << std::endl;
        //        ++parameter_index;
        //    }
        //    // std::cout << std::endl;

        //    if (particle_index == 64 - 1)
        //    {
        //        ++swarm_index;
        //        particle_index = 0;
        //    }
        //    else
        //    {
        //        ++particle_index;
        //    }
        //}
    }
}

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

    lud.inverse(invA);

    std::cout << invA << std::endl;

    Matrix identity = A * invA;

    std::cout << identity << std::endl;
}

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

template<class ExecutionSpace>
static void TestKolmogorovZurbenko()
{
    // const int n = 100;

    // Vector<double, ExecutionSpace> x("x", n);

    // Vector<double, ExecutionSpace> y("y", n);

    // for (size_t i = 0; i < x.extent(0); ++i)
    //{
    //    x(i) = i + 1;
    //    y(i) = x(i) + x(i) * sin(x(i));
    //}

    // const Vector<double, ExecutionSpace> new_y = KolmogorovZurbenko(x, y, 20);

    // std::cout << y << std::endl;
    // std::cout << new_y << std::endl;
}

template<class ExecutionSpace>
static void TestNestedHybridLoop()
{
    // const size_t nAgents    = 5;
    // const size_t nParticles = 100;

    // Vector<double, ExecutionSpace> x("x", nAgents * nParticles);

    ////#pragma omp parallel num_threads(nAgents)
    ////        for (size_t i = 0; i < nAgents; ++i)
    ////        {
    ////            cudaStream_t stream;
    ////            cudaStreamCreate(&stream);
    ////            Kokkos::Cuda space0(stream);
    ////
    ////            Kokkos::parallel_for(Kokkos::RangePolicy<ExecutionSpace>(space0, 0, nParticles), [=] __host__ __device__(const size_t i0) {
    ////                x(i + nAgents * i0) = 1.0 * i0;
    ////            });
    ////
    ////            space0.fence();
    ////        }

    // Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::OpenMP>(0, nAgents), [=](const size_t i) {
    //    cudaStream_t stream;
    //    cudaStreamCreate(&stream);
    //    Kokkos::Cuda space0(stream);

    //    Kokkos::parallel_for(Kokkos::RangePolicy<ExecutionSpace>(space0, 0, nParticles), [=] __host__ __device__(const size_t i0) {
    //        x(i + nAgents * i0) = 1.0 * i0;
    //    });

    //    space0.fence();
    //});

    // for (size_t i = 0; i < nAgents; ++i)
    //{
    //    for (size_t i0 = 0; i0 < nParticles; i0++)
    //    {
    //        std::cout << x(i + nAgents * i0) << " " << std::endl;
    //    }
    //}
}

template<class ExecutionSpace>
static void TestFractions()
{

    // const Fraction zero = 0;

    // std::cout << zero << std::endl;

    // const Fraction quarter = 0.25;

    // std::cout << quarter << std::endl;

    // const Fraction half = 0.5;

    // std::cout << half << std::endl;

    // const Fraction threequarters = 0.75;

    // std::cout << threequarters << std::endl;

    // const Fraction one = 1;

    // std::cout << one << std::endl;

    // const Fraction one_half = one + half;

    // std::cout << one_half << std::endl;

    // const Fraction three = 2 * one_half;

    // std::cout << three << std::endl;

    // const Fraction one_quarter = (three / 2) - 0.25;

    // std::cout << one_quarter << std::endl;

    // const Fraction two = three - one;

    // std::cout << two << std::endl;
}

// void test_extension()
//{
//    Kokkos::View<double*, ExecutionSpace> data("atomic", 10);
//
//    data(0) = Constants<double>::Max();
//
//    std::cout << TEXT("data equals ") << data(0) << std::endl;
//
//    double greater_than_value = 1.0;
//
//    if (Kokkos::atomic_greater_than_fetch(&data(0), greater_than_value))
//    {
//        std::cout << TEXT("data is > ") << greater_than_value << std::endl;
//        std::cout << TEXT("data now equals ") << data(0) << std::endl;
//    }
//
//    double greater_than_equal_value = 1.0;
//
//    if (Kokkos::atomic_greater_than_equal_fetch(&data(0), greater_than_equal_value))
//    {
//        std::cout << TEXT("data is >= ") << greater_than_equal_value << std::endl;
//        std::cout << TEXT("data now equals ") << data(0) << std::endl;
//    }
//
//    double less_than_value = 10.0;
//
//    if (Kokkos::atomic_less_than_fetch(&data(0), less_than_value))
//    {
//        std::cout << TEXT("data is < ") << less_than_value << std::endl;
//        std::cout << TEXT("data now equals ") << data(0) << std::endl;
//    }
//
//    double less_than_equal_value = 10.0;
//
//    if (Kokkos::atomic_less_than_equal_fetch(&data(0), less_than_equal_value))
//    {
//        std::cout << TEXT("data is <= ") << less_than_equal_value << std::endl;
//        std::cout << TEXT("data now equals ") << data(0) << std::endl;
//    }
//
//    double equal_value = 10.0;
//
//    if (Kokkos::atomic_equal_to_fetch(&data(0), equal_value))
//    {
//        std::cout << TEXT("data is == ") << equal_value << std::endl;
//        std::cout << TEXT("data now equals ") << data(0) << std::endl;
//    }
//
//    double not_equal_value = 1.0;
//
//    if (Kokkos::atomic_not_equal_to_fetch(&data(0), not_equal_value))
//    {
//        std::cout << TEXT("data is != ") << not_equal_value << std::endl;
//        std::cout << TEXT("data now equals ") << data(0) << std::endl;
//    }
//}

//#include <KokkosBlas1_fill.hpp>
//#include <runtime.Kokkos/Extensions.hpp>
//
//#include <PetroleumModels/RelativePermeabilityModels.hpp>
//
//#include <iostream>
//
// using namespace Kokkos::Extension;
// using namespace Kokkos::LinearAlgebra;

// template<typename DataType, class ExecutionSpace>
// void test1()
//{
//    Matrix<DataType, ExecutionSpace> A("A", 3, 3);
//
//    A(0, 0) = 36;
//    A(1, 0) = 30;
//    A(2, 0) = 18;
//
//    A(0, 1) = 30;
//    A(1, 1) = 41;
//    A(2, 1) = 23;
//
//    A(0, 2) = 18;
//    A(1, 2) = 23;
//    A(2, 2) = 14;
//
//    Vector<DataType, ExecutionSpace> b("b", 3);
//
//    b(0) = 288;
//    b(1) = 296;
//    b(2) = 173;
//
//    Vector<DataType, ExecutionSpace> x = Cholesky<DataType, ExecutionSpace>(A, b);
//
//    for(size_type i = 0; i < x.extent(0); ++i)
//    {
//        std::cout << x(i) << std::endl;
//    }
//
//    std::cout << std::endl;
//
//    x(0) = 5;
//    x(1) = 3;
//    x(2) = 1;
//
//    Vector<DataType, ExecutionSpace> b2 = A * x;
//
//    for(size_type i = 0; i < x.extent(0); ++i)
//    {
//        std::cout << b2(i) << std::endl;
//    }
//
//    std::cout << std::endl;
//}
//
// template<typename DataType, class ExecutionSpace>
// void test2()
//{
//    Matrix<DataType, ExecutionSpace> A("A", 2, 2);
//
//    A(0, 0) = -1;
//    A(0, 1) = 4;
//
//    A(1, 0) = 2;
//    A(1, 1) = 3;
//
//    Matrix<DataType, ExecutionSpace> X("X", 2, 2);
//
//    X(0, 0) = 9;
//    X(0, 1) = -3;
//
//    X(1, 0) = 6;
//    X(1, 1) = 1;
//
//    Matrix<DataType, ExecutionSpace> B = A * X;
//
//    for(size_type i = 0; i < B.extent(0); ++i)
//    {
//        for(size_type j = 0; j < B.extent(1); ++j)
//        {
//            std::cout << B(i, j) << std::endl;
//        }
//    }
//
//    std::cout << std::endl;
//}
//
// template<typename DataType, class ExecutionSpace>
// void test3()
//{
//    // Matrix<DataType, ExecutionSpace> A("A", 2, 2);
//
//    // A(0, 0) = -1;
//    // A(0, 1) = 4;
//
//    // A(1, 0) = 2;
//    // A(1, 1) = 3;
//
//    // Matrix<DataType, ExecutionSpace> B("B", 2, 2);
//
//    // B(0, 0) = 9;
//    // B(0, 1) = -3;
//
//    // B(1, 0) = 6;
//    // B(1, 1) = 1;
//
//    Matrix<DataType, ExecutionSpace> I = Identity<DataType, ExecutionSpace>(2, 2);
//
//    // Matrix<DataType, ExecutionSpace> X = A / B;
//
//    for(size_type i = 0; i < I.extent(0); ++i)
//    {
//        for(size_type j = 0; j < I.extent(1); ++j)
//        {
//            std::cout << I(i, j) << " ";
//        }
//        std::cout << std::endl;
//    }
//    std::cout << std::endl;
//
//    Vector<DataType, ExecutionSpace> r0 = row(I, 0);
//
//    for(size_type i = 0; i < r 0.extent(0); ++i)
//    {
//        std::cout << r0(i) << std::endl;
//    }
//
//    std::cout << std::endl;
//
//    auto c0 = column(I, 0);
//
//    for(size_type i = 0; i < c 0.extent(0); ++i)
//    {
//        std::cout << c0(i) << std::endl;
//    }
//
//    std::cout << std::endl;
//}
//
//#include <Analyzes/kNearestNeighbor.hpp>
//
// template<typename DataType, class ExecutionSpace>
// void test4()
//{
//    Kokkos::View<DataType* [5], typename ExecutionSpace::array_layout, ExecutionSpace> dataset("data", 10000);
//
//    for(size_type i = 0; i < dataset.extent(0); ++i)
//    {
//        dataset(i, 0) = 100 0.0 * ((DataType)std::rand() / (DataType)RAND_MAX);
//        dataset(i, 1) = 10 0.0 * ((DataType)std::rand() / (DataType)RAND_MAX);
//        dataset(i, 2) =  0.001 * ((DataType)std::rand() / (DataType)RAND_MAX);
//        dataset(i, 3) =  0.0001 * ((DataType)std::rand() / (DataType)RAND_MAX);
//        dataset(i, 4) = ((DataType)std::rand() / (DataType)RAND_MAX);
//    }
//
//    for(size_type i = 0; i < min<size_type>(100, dataset.extent(0)); ++i)
//    {
//        std::cout << dataset(i, 0) << std::endl;
//    }
//    std::cout << std::endl;
//
//    Kokkos::View<DataType**, typename ExecutionSpace::array_layout, ExecutionSpace> distances = kNearestNeighbor<DataType, ExecutionSpace, 5>(1, dataset);
//
//    for(size_type i = 0; i < min<size_type>(100, distances.extent(0)); ++i)
//    {
//        for(size_type j = 0; j < min<size_type>(100, distances.extent(1)); ++j)
//        {
//            std::cout << distances(i, j) << " ";
//        }
//        std::cout << std::endl;
//    }
//    std::cout << std::endl;
//}

// int main(int argc, char* argv[])
//{
//    Kokkos::initialize(argc, argv);
//
//    // test4<double, Kokkos::Serial>();
//
//    // test4<double, Kokkos::OpenMP>();
//
//    // test4<double, ExecutionSpace>();
//
//    test5<double, Kokkos::Cuda, 100>();
//
//    Kokkos::finalize_all();
//
//    std::cout << "Press any key to exit." << std::endl;
//    getchar();
//
//    return 0;
//}

///// <summary>
///// file:///R:/Rrelperm/inst/doc/RelativePermeability.html
///// </summary>
// template<typename DataType, class ExecutionSpace, unsigned Size>
// void test5()
//{
//    Kokkos::View<DataType* [4], typename ExecutionSpace::array_layout, ExecutionSpace> data("data", Size);
//
//    Kokkos::View<DataType** [4], typename ExecutionSpace::array_layout, ExecutionSpace> data_matrix("data_matrix", Size, Size);
//
//    using RelativePermeability = Petroleum::Reservoir::RelativePermeability<DataType, ExecutionSpace>;
//
//    DataType s_wcon  = 0.13;
//    DataType s_wcrit = 0.13;
//    DataType s_oirw  = 0.2;
//    DataType s_orw   = 0.2;
//    DataType s_oirg  = 0.1;
//    DataType s_org   = 0.1;
//    DataType s_irw   = 0.2;
//
//    DataType s_gcon  = 0.0;
//    DataType s_gcrit = 0.0;
//
//    DataType k_rwiro = 0.45;
//    DataType k_rocw  = 1.0;
//    DataType k_rgcl  = 0.35;
//    DataType k_rogcg = 1.0;
//
//    DataType s_w = 0.0;
//    DataType s_g = 0.0;
//
//    DataType n_w  = 4.25;
//    DataType n_g  = 3.0;
//    DataType n_ow = 2.5;
//    DataType n_og = 2.0;
//
//    // KokkosBlas::fill(data,  0.0);
//    // RelativePermeability::kr2p_ow(data, s_wcon, s_wcrit, s_oirw, s_orw, k_rwiro, k_rocw, n_w, n_ow);
//
//    ////--------------
//    // for (size_type i0 = 0; i0 < data.extent(0); ++i0)
//    //{
//    //
//    //    std::cout << data(i0, 0) << " " << data(i0, 1) << " " << data(i0, 2) << " " << data(i0, 3) << std::endl;
//    //}
//
//    ////--------------
//
//    // KokkosBlas::fill(data,  0.0);
//
//    // RelativePermeability::kr2p_gl(data, s_wcon, s_oirg, s_org, s_gcon, s_gcrit, k_rgcl, k_rogcg, n_g, n_og);
//
//    ////--------------
//    // for (size_type i0 = 0; i0 < data.extent(0); ++i0)
//    //{
//    //    std::cout << data(i0, 0) << " " << data(i0, 1) << " " << data(i0, 2) << " " << data(i0, 3) << std::endl;
//    //}
//    // std::cout << std::endl;
//    //--------------
//
//    // KokkosBlas::fill(data_matrix,  0.0);
//
//    // RelativePermeability::kr3p_StoneI_So(data_matrix, s_wcon, s_wcrit, s_oirw, s_orw, s_oirg, s_org, s_gcon, s_gcrit, k_rwiro, k_rocw, k_rgcl, n_w, n_ow, n_g, n_og);
//
//    ////--------------
//    // for (size_type i0 = 0; i0 < data_matrix.extent(0); ++i0)
//    //{
//    //    for (size_type i1 = 0; i1 < data_matrix.extent(0); ++i1)
//    //    {
//    //        std::cout << data_matrix(i0, i1, 0) << " " << data_matrix(i0, i1, 1) << " " << data_matrix(i0, i1, 2) << " " << data_matrix(i0, i1, 3) << std::endl;
//    //    }
//    //}
//    // std::cout << std::endl;
//    ////--------------
//
//    // KokkosBlas::fill(data_matrix,  0.0);
//
//    // RelativePermeability::kr3p_StoneI_SwSg(data_matrix, s_wcon, s_wcrit, s_irw, s_orw, s_oirg, s_org, s_gcon, s_gcrit, k_rwiro, k_rocw, k_rgcl, n_w, n_ow, n_g, n_og);
//
//    ////--------------
//    // for (size_type i0 = 0; i0 < data_matrix.extent(0); ++i0)
//    //{
//    //    std::cout << data_matrix(i0, 0) << " " << data_matrix(i0, 1) << " " << data_matrix(i0, 2) << " " << data_matrix(i0, 3) << std::endl;
//    //}
//    // std::cout << std::endl;
//    ////--------------
//
//    KokkosBlas::fill(data_matrix, 0.0);
//
//    RelativePermeability::kr3p_StoneII_So(data_matrix, s_wcon, s_wcrit, s_irw, s_orw, s_oirg, s_org, s_gcon, s_gcrit, k_rwiro, k_rocw, k_rgcl, n_w, n_ow, n_g, n_og);
//
//    //--------------
//    for (size_type i0 = 0; i0 < data_matrix.extent(0); ++i0)
//    {
//        for (size_type i1 = 0; i1 < data_matrix.extent(0); ++i1)
//        {
//            std::cout << data_matrix(i0, i1, 0) << " " << data_matrix(i0, i1, 1) << " " << data_matrix(i0, i1, 2) << " " << data_matrix(i0, i1, 3) << std::endl;
//        }
//    }
//    std::cout << std::endl;
//    //--------------
//
//    KokkosBlas::fill(data_matrix, 0.0);
//
//    RelativePermeability::kr3p_StoneII_SwSg(data_matrix, s_wcon, s_wcrit, s_irw, s_orw, s_oirg, s_org, s_gcon, s_gcrit, k_rwiro, k_rocw, k_rgcl, n_w, n_ow, n_g, n_og);
//
//    //--------------
//    for (size_type i0 = 0; i0 < data_matrix.extent(0); ++i0)
//    {
//        for (size_type i1 = 0; i1 < data_matrix.extent(0); ++i1)
//        {
//            std::cout << data_matrix(i0, i1, 0) << " " << data_matrix(i0, i1, 1) << " " << data_matrix(i0, i1, 2) << " " << data_matrix(i0, i1, 3) << std::endl;
//        }
//    }
//    std::cout << std::endl;
//    //--------------
//
//    ////KokkosBlas::fill(data,  0.0);
//
//    ////DataType RelativePermeability::krow2p_BC(s_w, s_wcon, s_orw, k_rocw, n_ow);
//
//    ////DataType RelativePermeability::krgl2p_BC(s_g, s_wcon, s_org, s_gcon, k_rogcg, n_og);
//}

// template<typename DataType, class ExecutionSpace>
// static void test()
//{
//    Kokkos::View<DataType*, typename ExecutionSpace::array_layout, ExecutionSpace> data("data", 100);
//
//    NumericalMethods::Statistics::GenerateRandomNumbers(data);
//
//    for (size_type i0 = 0; i0 < data.extent(0); ++i0)
//    {
//        std::cout << data(i0) << " ";
//    }
//
//    std::cout << std::endl;
//    std::cout << std::endl;
//
//    const int64 index = Kokkos::Extension::BinarySearch<DataType, ExecutionSpace>(data, 0.5, true);
//
//    for (size_type i0 = 0; i0 < data.extent(0); ++i0)
//    {
//        std::cout << data(i0) << " ";
//    }
//
//    std::cout << std::endl;
//    std::cout << std::endl;
//
//    std::cout << index << std::endl;
//
//    const int64 index1 = Kokkos::Extension::BinarySearch<DataType, ExecutionSpace>(data, 1.5, false);
//
//    std::cout << std::endl;
//    std::cout << std::endl;
//
//    std::cout << index1 << std::endl;
//
//    const int64 index2 = Kokkos::Extension::BinarySearch<DataType, ExecutionSpace>(data, -0.5, false);
//
//    std::cout << std::endl;
//    std::cout << std::endl;
//
//    std::cout << index2 << std::endl;
//
//    const int64 index3 = Kokkos::Extension::BinarySearch<DataType, ExecutionSpace>(data, 0.01, false);
//
//    std::cout << std::endl;
//    std::cout << std::endl;
//
//    std::cout << index3 << std::endl;
//}
//
// template<typename DataType, class ExecutionSpace>
// static void test4()
//{
//    Kokkos::View<DataType****, typename ExecutionSpace::array_layout, ExecutionSpace> data("data", 100, 100, 100, 6);
//
//    NumericalMethods::Statistics::GenerateRandomNumbers(data);
//
//    for (size_type i0 = 0; i0 < data.extent(0); ++i0)
//    {
//        for (size_type i1 = 0; i1 < data.extent(1); ++i1)
//        {
//            for (size_type i2 = 0; i2 < data.extent(2); ++i2)
//            {
//                for (size_type i3 = 0; i3 < data.extent(3); ++i3)
//                {
//                    std::cout << data(i0, i1, i2, i3) << " ";
//                }
//                std::cout << std::endl;
//            }
//            std::cout << std::endl;
//        }
//        std::cout << std::endl;
//    }
//}
//
// template<typename DataType, class ExecutionSpace, unsigned Size>
// static void test6()
//{
//    Kokkos::View<DataType******, typename ExecutionSpace::array_layout, ExecutionSpace> data("data", Size, Size, Size, Size, Size, Size);
//
//    NumericalMethods::Statistics::GenerateRandomNumbers(data);
//
//    for (size_type i0 = 0; i0 < data.extent(0); ++i0)
//    {
//        for (size_type i1 = 0; i1 < data.extent(1); ++i1)
//        {
//            for (size_type i2 = 0; i2 < data.extent(2); ++i2)
//            {
//                for (size_type i3 = 0; i3 < data.extent(3); ++i3)
//                {
//                    for (size_type i4 = 0; i4 < data.extent(4); ++i4)
//                    {
//                        for (size_type i5 = 0; i5 < data.extent(5); ++i5)
//                        {
//                            std::cout << data(i0, i1, i2, i3, i4, i5) << " ";
//                        }
//                        std::cout << std::endl;
//                    }
//                    std::cout << std::endl;
//                }
//                std::cout << std::endl;
//            }
//            std::cout << std::endl;
//        }
//        std::cout << std::endl;
//    }
//}

// void* ___chkstk_ms            = 0;

//#pragma comment(linker, "/EXPORT:_InterlockedCompareExchange64")
