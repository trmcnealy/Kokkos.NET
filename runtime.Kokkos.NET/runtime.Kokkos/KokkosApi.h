#pragma once

#include <runtime.Kokkos/ViewTypes.hpp>
#include <StdExtensions.hpp>

#include <ValueType.hpp>

#undef _In_
#define _In_ __attribute__((annotate("In")))

#undef _Out_
#define _Out_ __attribute__((annotate("Out")))

#undef _Inout_
#define _Inout_ __attribute__((annotate("InOut")))

KOKKOS_NET_API_EXTERN void* Allocate(const ExecutionSpaceKind execution_space, const size_type arg_alloc_size) noexcept;

KOKKOS_NET_API_EXTERN void* Reallocate(const ExecutionSpaceKind execution_space, void* instance, const size_type arg_alloc_size) noexcept;

KOKKOS_NET_API_EXTERN void Copy(const ExecutionSpaceKind src_execution_space, void* src, const ExecutionSpaceKind dest_execution_space, void* dest, const size_type size_in_bytes) noexcept;

KOKKOS_NET_API_EXTERN void Free(const ExecutionSpaceKind execution_space, void* instance) noexcept;

KOKKOS_NET_API_EXTERN void Initialize(const int narg, char* arg[]) noexcept;

KOKKOS_NET_API_EXTERN void InitializeSerial() noexcept;

KOKKOS_NET_API_EXTERN void InitializeOpenMP(const int num_threads) noexcept;

KOKKOS_NET_API_EXTERN void InitializeCuda(const int use_gpu) noexcept;

KOKKOS_NET_API_EXTERN void InitializeThreads(const int num_cpu_threads, const int gpu_device_id) noexcept;

KOKKOS_NET_API_EXTERN void InitializeArguments(const Kokkos::InitArguments arguments) noexcept;

KOKKOS_NET_API_EXTERN void Finalize() noexcept;

KOKKOS_NET_API_EXTERN void FinalizeSerial() noexcept;

KOKKOS_NET_API_EXTERN void FinalizeOpenMP() noexcept;

KOKKOS_NET_API_EXTERN void FinalizeCuda() noexcept;

KOKKOS_NET_API_EXTERN void FinalizeAll() noexcept;

KOKKOS_NET_API_EXTERN bool IsInitialized() noexcept;

KOKKOS_NET_API_EXTERN void PrintConfiguration(const bool detail = false) noexcept;

KOKKOS_NET_API_EXTERN unsigned int CudaGetDeviceCount() noexcept;

KOKKOS_NET_API_EXTERN unsigned int CudaGetComputeCapability(unsigned int device_id) noexcept;

KOKKOS_NET_API_EXTERN void CreateViewRank0(void* instance, NdArray* ndArray) noexcept;

KOKKOS_NET_API_EXTERN void CreateViewRank1(void* instance, NdArray* ndArray, const size_type n0) noexcept;

KOKKOS_NET_API_EXTERN void CreateViewRank2(void* instance, NdArray* ndArray, const size_type n0, const size_type n1) noexcept;

KOKKOS_NET_API_EXTERN void CreateViewRank3(void* instance, NdArray* ndArray, const size_type n0, const size_type n1, const size_type n2) noexcept;

KOKKOS_NET_API_EXTERN void CreateViewRank4(void* instance, NdArray* ndArray, const size_type n0, const size_type n1, const size_type n2, const size_type n3) noexcept;

KOKKOS_NET_API_EXTERN void CreateViewRank5(void* instance, NdArray* ndArray, const size_type n0, const size_type n1, const size_type n2, const size_type n3, const size_type n4) noexcept;

KOKKOS_NET_API_EXTERN void CreateViewRank6(void*           instance,
                                           NdArray*        ndArray,
                                           const size_type n0,
                                           const size_type n1,
                                           const size_type n2,
                                           const size_type n3,
                                           const size_type n4,
                                           const size_type n5) noexcept;

KOKKOS_NET_API_EXTERN void CreateViewRank7(void*           instance,
                                           NdArray*        ndArray,
                                           const size_type n0,
                                           const size_type n1,
                                           const size_type n2,
                                           const size_type n3,
                                           const size_type n4,
                                           const size_type n5,
                                           const size_type n6) noexcept;

KOKKOS_NET_API_EXTERN void CreateViewRank8(void*           instance,
                                           NdArray*        ndArray,
                                           const size_type n0,
                                           const size_type n1,
                                           const size_type n2,
                                           const size_type n3,
                                           const size_type n4,
                                           const size_type n5,
                                           const size_type n6,
                                           const size_type n7) noexcept;

KOKKOS_NET_API_EXTERN void CreateView(void* instance, NdArray* ndArray) noexcept;

KOKKOS_NET_API_EXTERN const NativeString GetLabel(void* instance, const NdArray ndArray) noexcept;

KOKKOS_NET_API_EXTERN uint64 GetSize(void* instance, const NdArray ndArray) noexcept;

KOKKOS_NET_API_EXTERN uint64 GetStride(void* instance, const NdArray ndArray, const uint32 dim) noexcept;

KOKKOS_NET_API_EXTERN uint64 GetExtent(void* instance, const NdArray ndArray, const uint32 dim) noexcept;

KOKKOS_NET_API_EXTERN void CopyTo(void* instance, const NdArray ndArray, ValueType values[]) noexcept;

KOKKOS_NET_API_EXTERN ValueType GetValue(void*           instance,
                                         const NdArray   ndArray,
                                         const size_type i0,
                                         const size_type i1 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
                                         const size_type i2 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
                                         const size_type i3 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
                                         const size_type i4 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
                                         const size_type i5 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
                                         const size_type i6 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
                                         const size_type i7 = KOKKOS_IMPL_CTOR_DEFAULT_ARG) noexcept;

KOKKOS_NET_API_EXTERN void SetValue(void*           instance,
                                    const NdArray   ndArray,
                                    const ValueType value,
                                    const size_type i0,
                                    const size_type i1 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
                                    const size_type i2 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
                                    const size_type i3 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
                                    const size_type i4 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
                                    const size_type i5 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
                                    const size_type i6 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
                                    const size_type i7 = KOKKOS_IMPL_CTOR_DEFAULT_ARG) noexcept;

KOKKOS_NET_API_EXTERN void RcpViewToNdArray(void*                    instance,
                                            const ExecutionSpaceKind execution_space,
                                            const LayoutKind         layout,
                                            const DataTypeKind       data_type,
                                            const uint16             rank,
                                            NdArray*                 ndArray) noexcept;

KOKKOS_NET_API_EXTERN NdArray* ViewToNdArray(void* instance, const ExecutionSpaceKind execution_space, const LayoutKind layout, const DataTypeKind data_type, const uint16 rank) noexcept;

struct KokkosApi
{
    void* (*Allocate)(const ExecutionSpaceKind, const size_type) noexcept;

    void* (*Reallocate)(const ExecutionSpaceKind, void*, const size_type) noexcept;

    void (*Copy)(const ExecutionSpaceKind, void*, const ExecutionSpaceKind, void*, const size_type size_in_bytes) noexcept;

    void (*Free)(const ExecutionSpaceKind, void*) noexcept;

    void (*Initialize)(const int, char*[]) noexcept;

    void (*InitializeSerial)() noexcept;

    void (*InitializeOpenMP)(const int num_threads) noexcept;

    void (*InitializeCuda)(const int use_gpu) noexcept;

    void (*InitializeThreads)(int, int) noexcept;

    void (*InitializeArguments)(const Kokkos::InitArguments) noexcept;

    void (*Finalize)() noexcept;

    void (*FinalizeSerial)() noexcept;

    void (*FinalizeOpenMP)() noexcept;

    void (*FinalizeCuda)() noexcept;

    void (*FinalizeAll)() noexcept;

    bool (*IsInitialized)() noexcept;

    void (*PrintConfiguration)(const bool) noexcept;

    unsigned int (*CudaGetDeviceCount)() noexcept;

    unsigned int (*CudaGetComputeCapability)(unsigned int) noexcept;

    void (*CreateViewRank0)(void*, NdArray*) noexcept;

    void (*CreateViewRank1)(void*, NdArray*, const size_type) noexcept;

    void (*CreateViewRank2)(void*, NdArray*, const size_type, const size_type) noexcept;

    void (*CreateViewRank3)(void*, NdArray*, const size_type, const size_type, const size_type) noexcept;

    void (*CreateViewRank4)(void*, NdArray*, const size_type, const size_type, const size_type, const size_type) noexcept;

    void (*CreateViewRank5)(void*, NdArray*, const size_type, const size_type, const size_type, const size_type, const size_type) noexcept;

    void (*CreateViewRank6)(void*, NdArray*, const size_type, const size_type, const size_type, const size_type, const size_type, const size_type) noexcept;

    void (*CreateViewRank7)(void*, NdArray*, const size_type, const size_type, const size_type, const size_type, const size_type, const size_type, const size_type) noexcept;

    void (*CreateViewRank8)(void*, NdArray*, const size_type, const size_type, const size_type, const size_type, const size_type, const size_type, const size_type, const size_type) noexcept;

    void (*CreateView)(void*, NdArray*) noexcept;

    const NativeString (*GetLabel)(void* instance, const NdArray) noexcept;

    uint64 (*GetSize)(void*, const NdArray) noexcept;

    uint64 (*GetStride)(void*, const NdArray, const uint32) noexcept;

    uint64 (*GetExtent)(void*, const NdArray, const uint32) noexcept;

    void (*CopyTo)(void*, const NdArray, ValueType[]) noexcept;

    ValueType (*GetValue)(void*, const NdArray, const size_type, const size_type, const size_type, const size_type, const size_type, const size_type, const size_type, const size_type) noexcept;

    void (*SetValue)(void*,
                     const NdArray,
                     const ValueType,
                     const size_type,
                     const size_type,
                     const size_type,
                     const size_type,
                     const size_type,
                     const size_type,
                     const size_type,
                     const size_type) noexcept;

    void (*RcpViewToNdArray)(void*, const ExecutionSpaceKind, const LayoutKind, const DataTypeKind, const uint16, NdArray*) noexcept;

    NdArray* (*ViewToNdArray)(void*, const ExecutionSpaceKind, const LayoutKind, const DataTypeKind, const uint16) noexcept;
};

#define KOKKOS_POLICY_TAG(NAME, RANK)                                                                                                                                                                  \
    struct NAME##Tag                                                                                                                                                                                   \
    {                                                                                                                                                                                                  \
    };                                                                                                                                                                                                 \
    typedef Kokkos::TeamPolicy<ExecutionSpace, Kokkos::Schedule<Kokkos::Static>, NAME##Tag>                                                      NAME##_TeamPolicyType;                                \
    typedef typename NAME##_TeamPolicyType::member_type                                                                                          NAME##_TeamMemberType;                                \
    typedef Kokkos::MDRangePolicy<ExecutionSpace, Kokkos::Rank<RANK>, Kokkos::IndexType<size_type>, Kokkos::Schedule<Kokkos::Static>, NAME##Tag> NAME##Rank##RANK##_MDRangeType;                       \
    typedef typename NAME##Rank##RANK##_MDRangeType::point_type                                                                                  NAME##Rank##RANK##_PointType;                         \
    typedef Kokkos::RangePolicy<ExecutionSpace, Kokkos::IndexType<size_type>, Kokkos::Schedule<Kokkos::Static>, NAME##Tag>                       NAME##_RangeType;

// inline void* operator new(size_type size)
// {
//     //#if defined(__CUDA_ARCH__)
//     //    return Kokkos::kokkos_malloc<Kokkos::Cuda::memory_space>(size);
//     //#else
//     return Kokkos::kokkos_malloc<Kokkos::DefaultExecutionSpace::memory_space>(size);
// }
//
// inline void* operator new[](const size_type size) { return operator new(size); }

// void* operator new(const size_type size, void* ptr) throw() { return ptr; }
//
// void* operator new[](const size_type size, void* ptr) throw() { return ptr; }

// template<typename DataType, class ExecutionSpace>
// void* operator new(const size_type size, void* ptr, const std::string label, const size_type n0, const size_type n1, const size_type n2) throw()
//{
//    return new(ptr) Kokkos::View<DataType***, typename ExecutionSpace::array_layout, ExecutionSpace>(label, n0, n1, n2);
//}
//
// template<typename DataType, class ExecutionSpace>
// void* operator new[](const size_type size, void* ptr, const std::string label, const size_type n0, const size_type n1, const size_type n2) throw()
//{
//    return new(ptr) Kokkos::View<DataType***, typename ExecutionSpace::array_layout, ExecutionSpace>(label, n0, n1, n2);
//}

// inline void operator delete(void* ptr) noexcept
// {
//     if(ptr != nullptr)
//     {
//         return Kokkos::kokkos_free<Kokkos::DefaultExecutionSpace::memory_space>(ptr);
//     }
// }
//
// inline void operator delete[](void* ptr) noexcept
// {
//     if(ptr != nullptr)
//     {
//         return operator delete(ptr);
//     }
// }

// inline void operator delete(void* ptr, const size_type size) noexcept
//{
//    if(ptr != nullptr)
//    {
//        return operator delete(ptr);
//    }
//}
//
// inline void operator delete[](void* ptr, const size_type size) noexcept
//{
//    if(ptr != nullptr)
//    {
//        return operator delete(ptr);
//    }
//}
