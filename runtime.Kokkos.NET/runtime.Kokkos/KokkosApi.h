#pragma once

#include "ViewTypes.hpp"

__inline static bool isNullTerminating(const std::string& str) { return str[str.size() - 1] == '\0'; }

__inline static bool isNullTerminating(const int& length, char* const bytes) { return bytes[length - 1] == '\0'; }

struct NativeString
{
    const int Length;

    int8* const Bytes;

    NativeString(const std::string& str);

    NativeString(const int& length, char* const bytes);

    std::string ToString() const;
};

union ValueType
{
    float  Single;
    double Double;
    bool   Bool;
    int8   Int8;
    uint8  UInt8;
    int16  Int16;
    uint16 UInt16;
    int32  Int32;
    uint32 UInt32;
    int64  Int64;
    uint64 UInt64;

    ValueType(const float& value) { Single = value; }

    ValueType(const double& value) { Double = value; }

    ValueType(const bool& value) { Bool = value; }

    ValueType(const int8& value) { Int8 = value; }

    ValueType(const uint8& value) { UInt8 = value; }

    ValueType(const int16& value) { Int16 = value; }

    ValueType(const uint16& value) { UInt16 = value; }

    ValueType(const int32& value) { Int32 = value; }

    ValueType(const uint32& value) { UInt32 = value; }

    ValueType(const int64& value) { Int64 = value; }

    ValueType(const uint64& value) { UInt64 = value; }

    ValueType(float&& value) { Single = std::move(value); }

    ValueType(double&& value) { Double = std::move(value); }

    ValueType(bool&& value) { Bool = std::move(value); }

    ValueType(int8&& value) { Int8 = std::move(value); }

    ValueType(uint8&& value) { UInt8 = std::move(value); }

    ValueType(int16&& value) { Int16 = std::move(value); }

    ValueType(uint16&& value) { UInt16 = std::move(value); }

    ValueType(int32&& value) { Int32 = std::move(value); }

    ValueType(uint32&& value) { UInt32 = std::move(value); }

    ValueType(int64&& value) { Int64 = std::move(value); }

    ValueType(uint64&& value) { UInt64 = std::move(value); }

    ValueType& operator=(const float& value)
    {
        Single = value;
        return *this;
    }

    ValueType& operator=(const double& value)
    {
        Double = value;
        return *this;
    }

    ValueType& operator=(const bool& value)
    {
        Bool = value;
        return *this;
    }

    ValueType& operator=(const int8& value)
    {
        Int8 = value;

        return *this;
    }

    ValueType& operator=(const uint8& value)
    {
        UInt8 = value;
        return *this;
    }

    ValueType& operator=(const int16& value)
    {
        Int16 = value;
        return *this;
    }

    ValueType& operator=(const uint16& value)
    {
        UInt16 = value;
        return *this;
    }

    ValueType& operator=(const int32& value)
    {
        Int32 = value;
        return *this;
    }

    ValueType& operator=(const uint32& value)
    {
        UInt32 = value;
        return *this;
    }

    ValueType& operator=(const int64& value)
    {
        Int64 = value;
        return *this;
    }

    ValueType& operator=(const uint64& value)
    {
        UInt64 = value;
        return *this;
    }

    ValueType& operator=(float&& value)
    {
        Single = std::move(value);
        return *this;
    }

    ValueType& operator=(double&& value)
    {
        Double = std::move(value);
        return *this;
    }

    ValueType& operator=(bool&& value)
    {
        Bool = std::move(value);
        return *this;
    }

    ValueType& operator=(int8&& value)
    {
        Int8 = std::move(value);

        return *this;
    }

    ValueType& operator=(uint8&& value)
    {
        UInt8 = std::move(value);
        return *this;
    }

    ValueType& operator=(int16&& value)
    {
        Int16 = std::move(value);
        return *this;
    }

    ValueType& operator=(uint16&& value)
    {
        UInt16 = std::move(value);
        return *this;
    }

    ValueType& operator=(int32&& value)
    {
        Int32 = std::move(value);
        return *this;
    }

    ValueType& operator=(uint32&& value)
    {
        UInt32 = std::move(value);
        return *this;
    }

    ValueType& operator=(int64&& value)
    {
        Int64 = std::move(value);
        return *this;
    }

    ValueType& operator=(uint64&& value)
    {
        UInt64 = std::move(value);
        return *this;
    }

    __inline operator float() const { return Single; }
    __inline operator double() const { return Double; }
    __inline operator bool() const { return Bool; }
    __inline operator int8() const { return Int8; }
    __inline operator uint8() const { return UInt8; }
    __inline operator int16() const { return Int16; }
    __inline operator uint16() const { return UInt16; }
    __inline operator int32() const { return Int32; }
    __inline operator uint32() const { return UInt32; }
    __inline operator int64() const { return Int64; }
    __inline operator uint64() const { return UInt64; }
};

#undef _In_
#define _In_ __attribute__((annotate("In")))

#undef _Out_
#define _Out_ __attribute__((annotate("Out")))

#undef _Inout_
#define _Inout_ __attribute__((annotate("InOut")))

KOKKOS_NET_API_EXTERN void* Allocate(const ExecutionSpaceKind& execution_space, const size_type& arg_alloc_size) noexcept;

KOKKOS_NET_API_EXTERN void* Reallocate(const ExecutionSpaceKind& execution_space, void* instance, const size_type& arg_alloc_size) noexcept;

KOKKOS_NET_API_EXTERN void Free(const ExecutionSpaceKind& execution_space, void* instance) noexcept;

KOKKOS_NET_API_EXTERN void Initialize(int& narg, char* arg[]) noexcept;

KOKKOS_NET_API_EXTERN void InitializeThreads(int num_cpu_threads, int gpu_device_id) noexcept;

KOKKOS_NET_API_EXTERN void InitializeArguments(const Kokkos::InitArguments& arguments) noexcept;

KOKKOS_NET_API_EXTERN void Finalize() noexcept;

KOKKOS_NET_API_EXTERN void FinalizeAll() noexcept;

KOKKOS_NET_API_EXTERN bool IsInitialized() noexcept;

KOKKOS_NET_API_EXTERN void PrintConfiguration(const bool& detail = false) noexcept;

KOKKOS_NET_API_EXTERN unsigned int CudaGetDeviceCount() noexcept;

KOKKOS_NET_API_EXTERN unsigned int CudaGetComputeCapability(unsigned int device_id) noexcept;

KOKKOS_NET_API_EXTERN void CreateViewRank0(void* instance, NdArray& ndArray) noexcept;

KOKKOS_NET_API_EXTERN void CreateViewRank1(void* instance, NdArray& ndArray, const size_type& n0) noexcept;

KOKKOS_NET_API_EXTERN void CreateViewRank2(void* instance, NdArray& ndArray, const size_type& n0, const size_type& n1) noexcept;

KOKKOS_NET_API_EXTERN void CreateViewRank3(void* instance, NdArray& ndArray, const size_type& n0, const size_type& n1, const size_type& n2) noexcept;

KOKKOS_NET_API_EXTERN void CreateView(void* instance, NdArray& ndArray) noexcept;

KOKKOS_NET_API_EXTERN const NativeString GetLabel(void* instance, const NdArray& ndArray) noexcept;

KOKKOS_NET_API_EXTERN uint64 GetSize(void* instance, const NdArray& ndArray) noexcept;

KOKKOS_NET_API_EXTERN uint64 GetStride(void* instance, const NdArray& ndArray, const uint32& dim) noexcept;

KOKKOS_NET_API_EXTERN uint64 GetExtent(void* instance, const NdArray& ndArray, const uint32& dim) noexcept;

KOKKOS_NET_API_EXTERN void CopyTo(void* instance, const NdArray& ndArray, ValueType* values) noexcept;

KOKKOS_NET_API_EXTERN ValueType GetValue(void* instance, const NdArray& ndArray, const size_type& i0, const size_type& i1, const size_type& i2) noexcept;

KOKKOS_NET_API_EXTERN void SetValue(void* instance, const NdArray& ndArray, const ValueType& value, const size_type& i0, const size_type& i1, const size_type& i2) noexcept;

struct KokkosApi
{
    void* (*Allocate)(const ExecutionSpaceKind&, const size_type&) noexcept;

    void* (*Reallocate)(const ExecutionSpaceKind&, void*, const size_type&) noexcept;

    void (*Free)(const ExecutionSpaceKind&, void*) noexcept;

    void (*Initialize)(int&, char*[]) noexcept;

    void (*InitializeThreads)(int, int) noexcept;

    void (*InitializeArguments)(const Kokkos::InitArguments&) noexcept;

    void (*Finalize)() noexcept;

    void (*FinalizeAll)() noexcept;

    bool (*IsInitialized)() noexcept;

    void (*PrintConfiguration)(const bool&) noexcept;

    unsigned int (*CudaGetDeviceCount)() noexcept;

    unsigned int (*CudaGetComputeCapability)(unsigned int) noexcept;

    void (*CreateViewRank0)(void*, NdArray&) noexcept;

    void (*CreateViewRank1)(void*, NdArray&, const size_type&) noexcept;

    void (*CreateViewRank2)(void*, NdArray&, const size_type&, const size_type&) noexcept;

    void (*CreateViewRank3)(void*, NdArray&, const size_type&, const size_type&, const size_type&) noexcept;

    void (*CreateView)(void*, NdArray&) noexcept;

    const NativeString (*GetLabel)(void* instance, const NdArray&) noexcept;

    uint64 (*GetSize)(void*, const NdArray&) noexcept;

    uint64 (*GetStride)(void*, const NdArray&, const uint32&) noexcept;

    uint64 (*GetExtent)(void*, const NdArray&, const uint32&) noexcept;

    void (*CopyTo)(void*, const NdArray&, ValueType*) noexcept;

    ValueType (*GetValue)(void*, const NdArray&, const size_type&, const size_type&, const size_type&) noexcept;

    void (*SetValue)(void*, const NdArray&, const ValueType&, const size_type&, const size_type&, const size_type&) noexcept;
};

__forceinline void* operator new(size_type size)
{
    //#if defined(__CUDA_ARCH__)
    //    return Kokkos::kokkos_malloc<Kokkos::Cuda::memory_space>(size);
    //#else
    return Kokkos::kokkos_malloc<Kokkos::Serial::memory_space>(size);
}

__forceinline void* operator new[](const size_type size) { return operator new(size); }

//__forceinline void* operator new(const size_type size, void* ptr) throw() { return ptr; }
//
//__forceinline void* operator new[](const size_type size, void* ptr) throw() { return ptr; }

//template<typename DataType, class ExecutionSpace>
//__forceinline void* operator new(const size_type size, void* ptr, const std::string& label, const size_type& n0, const size_type& n1, const size_type& n2) throw()
//{
//    return new(ptr) Kokkos::View<DataType***, typename ExecutionSpace::array_layout, ExecutionSpace>(label, n0, n1, n2);
//}
//
//template<typename DataType, class ExecutionSpace>
//__forceinline void* operator new[](const size_type size, void* ptr, const std::string& label, const size_type& n0, const size_type& n1, const size_type& n2) throw()
//{
//    return new(ptr) Kokkos::View<DataType***, typename ExecutionSpace::array_layout, ExecutionSpace>(label, n0, n1, n2);
//}

__forceinline void operator delete(void* ptr) noexcept
{
    if(ptr != nullptr)
    {
        return Kokkos::kokkos_free<Kokkos::Serial::memory_space>(ptr);
    }
}

__forceinline void operator delete[](void* ptr) noexcept
{
    if(ptr != nullptr)
    {
        return operator delete(ptr);
    }
}

__forceinline void operator delete(void* ptr, const size_type size) noexcept
{
    if(ptr != nullptr)
    {
        return operator delete(ptr);
    }
}

__forceinline void operator delete[](void* ptr, const size_type size) noexcept
{
    if(ptr != nullptr)
    {
        return operator delete(ptr);
    }
}
