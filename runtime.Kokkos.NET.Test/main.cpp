//-Bdynamic,-lkokkoscore.dll,-lkokkoscontainers.dll,-lkokkoskernels.dll,-lkokkoskernels.dll,-Bstatic,-lteuchoscore
//-LC:/AssemblyCache/kokkos/lib,-LC:/AssemblyCache/Trilinos/lib

#include <runtime.Kokkos/ViewTypes.hpp>
#include <runtime.Kokkos/Extensions.hpp>
//#include <Array.hpp>
//#include <Statistics/Random.hpp>
//#include <DataStorage/Dictionary.hpp>

//#include <Kokkos_UnorderedMap.hpp>

#include <iostream>
#include <cstring>
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

using AtomicTrait       = Kokkos::MemoryTraits<Kokkos::Atomic>;
using RandomAccessTrait = Kokkos::MemoryTraits<Kokkos::RandomAccess>;

//using ExecutionSpace   = Kokkos::Cuda;
using ExecutionSpace   = Kokkos::Serial;
//using Array            = System::Array<double, ExecutionSpace>;
using LayoutType       = ExecutionSpace::array_layout;
using AtomicView       = Kokkos::View<double*, LayoutType, ExecutionSpace, AtomicTrait>;
using RandomAccessView = Kokkos::View<double*, LayoutType, ExecutionSpace, RandomAccessTrait>;
using View = Kokkos::View<double*, LayoutType, ExecutionSpace>;
//using StringDictionary = NumericalMethods::DataStorage::Dictionary<std::string, Array*, ExecutionSpace>;
//using ViewDictionary   = NumericalMethods::DataStorage::Dictionary<std::string, AtomicView, ExecutionSpace>;

typedef Kokkos::UnorderedMap<std::string, size_type, ExecutionSpace> map_type;

//using namespace Kokkos;
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
//    for (size_type i = 0; i < eqs.extent(0); i++)
//    {
//        for (size_type j = 0; j < eqs.extent(1); j++)
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
//    for (size_type i = 0; i < eqs_v.extent(0); i++)
//    {
//        std::cout << eqs_v(i) << " " << std::endl;
//    }
//}

int main(int argc, char** argv)
{
    // void*           ptr;
    // const size_type arg_alloc_size = 80;
    // cudaError_t err = cudaMallocManaged(&ptr, arg_alloc_size, cudaMemAttachGlobal);

    const int  num_threads      = 4;
    const int  num_numa         = 1;
    const int  device_id        = 0;
    const int  ndevices         = 1;
    const int  skip_device      = 9999;
    const bool disable_warnings = false;

    Kokkos::InitArguments arguments;
    //arguments.num_threads      = num_threads;
    //arguments.num_numa         = num_numa;
    //arguments.device_id        = device_id;
    //arguments.ndevices         = ndevices;
    //arguments.skip_device      = skip_device;
    //arguments.disable_warnings = disable_warnings;

    Kokkos::ScopeGuard kokkos(arguments);
    {
        Kokkos::View<double*, ExecutionSpace> data("atomic", 10);

        data(0) = Constants<double>::Max();
        
        std::cout << TEXT("data equals ") << data(0) << std::endl;

        double value = 1.0;

        if (Kokkos::atomic_greater_than_fetch(&data(0), value))
        {
            std::cout << TEXT("data is > ") << value << std::endl;
            std::cout << TEXT("data now equals ") << data(0) << std::endl;
        }
    }

    //// Kokkos::print_configuration(std::cout);

    //// const std::string oilName = STRING("Oil");

    //// std::cout << oilData(0) << std::endl;

    //// ViewDictionary vDictionary(10);
    //// vDictionary.Add(oilName, oilData);

    //// std::cout << (vDictionary.ContainsKey(oilName) ? TEXT("Yes") : TEXT("No")) << std::endl;

    //// Array* oilData =  new Array(1);
    //// oilData->operator[](0) = 0.5416;

    //// std::cout << oilData->operator[](0) << std::endl;

    //// const std::string oilName = STRING("Oil");

    // map_type map(10);

    // const auto result = map.insert("Oil", 5);

    // if (result.failed())
    //{
    //    std::cout << TEXT("insert failed") << std::endl;
    //}

    //
    // View oilData("Oil", 10);

    //

    //// StringDictionary strDictionary(10);

    //// strDictionary.Add(oilName, oilData);

    // if (map.exists("Oil"))
    //{

    //    const uint32_t index = map.find("Oil");

    //    const auto gid = map.key_at(index);

    //    const auto lid = map.value_at(index);

    //    std::cout << lid << std::endl;

    //    oilData(lid) = 0.5416;

    //    std::cout << oilData(lid) << std::endl;

    //}

    ////// test4<double, Kokkos::Serial>();

    ////// test4<double, Kokkos::OpenMP>();

    ////// test4<double, ExecutionSpace>();

    //// test<double, ExecutionSpace>();

    std::cout << "Press any key to exit." << std::endl;
    getchar();

    return 0;
}

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
//    for(size_type i = 0; i < x.extent(0); i++)
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
//    for(size_type i = 0; i < x.extent(0); i++)
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
//    for(size_type i = 0; i < B.extent(0); i++)
//    {
//        for(size_type j = 0; j < B.extent(1); j++)
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
//    for(size_type i = 0; i < I.extent(0); i++)
//    {
//        for(size_type j = 0; j < I.extent(1); j++)
//        {
//            std::cout << I(i, j) << " ";
//        }
//        std::cout << std::endl;
//    }
//    std::cout << std::endl;
//
//    Vector<DataType, ExecutionSpace> r0 = row(I, 0);
//
//    for(size_type i = 0; i < r 0.extent(0); i++)
//    {
//        std::cout << r0(i) << std::endl;
//    }
//
//    std::cout << std::endl;
//
//    auto c0 = column(I, 0);
//
//    for(size_type i = 0; i < c 0.extent(0); i++)
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
//    for(size_type i = 0; i < dataset.extent(0); i++)
//    {
//        dataset(i, 0) = 100 0.0 * ((DataType)std::rand() / (DataType)RAND_MAX);
//        dataset(i, 1) = 10 0.0 * ((DataType)std::rand() / (DataType)RAND_MAX);
//        dataset(i, 2) =  0.001 * ((DataType)std::rand() / (DataType)RAND_MAX);
//        dataset(i, 3) =  0.0001 * ((DataType)std::rand() / (DataType)RAND_MAX);
//        dataset(i, 4) = ((DataType)std::rand() / (DataType)RAND_MAX);
//    }
//
//    for(size_type i = 0; i < min<size_type>(100, dataset.extent(0)); i++)
//    {
//        std::cout << dataset(i, 0) << std::endl;
//    }
//    std::cout << std::endl;
//
//    Kokkos::View<DataType**, typename ExecutionSpace::array_layout, ExecutionSpace> distances = kNearestNeighbor<DataType, ExecutionSpace, 5>(1, dataset);
//
//    for(size_type i = 0; i < min<size_type>(100, distances.extent(0)); i++)
//    {
//        for(size_type j = 0; j < min<size_type>(100, distances.extent(1)); j++)
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

//template<typename DataType, class ExecutionSpace>
//static void test()
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
//template<typename DataType, class ExecutionSpace>
//static void test4()
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
//template<typename DataType, class ExecutionSpace, unsigned Size>
//static void test6()
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
