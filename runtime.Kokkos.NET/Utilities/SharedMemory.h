#pragma once

#ifndef EXTERN
#    define EXTERN extern
#endif

#ifndef EXTERN_C
#    define EXTERN_C extern "C"
#endif

#if defined(_MSC_VER)
#    define EXPORT_TYPE __declspec(dllexport)
#    define IMPORT_TYPE __declspec(dllimport)
#elif  defined(__clang__)
#    define EXPORT_TYPE __attribute__((visibility("default"))) __declspec(dllexport)
#    define IMPORT_TYPE __declspec(dllimport)
#else
#    define EXPORT_TYPE __attribute__((visibility("default")))
#    define IMPORT_TYPE
#endif

#ifdef KOKKOS_NET_EXPORTS

#    ifndef KOKKOS_NET_API_EXPORT
#        define KOKKOS_NET_API_EXPORT EXPORT_TYPE
#    endif

#    ifndef KOKKOS_NET_API_EXTERNC
#        define KOKKOS_NET_API_EXTERNC EXTERN_C EXPORT_TYPE
#    endif

#    ifndef KOKKOS_NET_API_EXTERN
#        define KOKKOS_NET_API_EXTERN EXTERN EXPORT_TYPE
#    endif

#    ifndef KOKKOS_NET_API_IMPORT
#        define KOKKOS_NET_API_IMPORT IMPORT_TYPE
#    endif

#else

#    ifndef KOKKOS_NET_API_EXPORT
#        define KOKKOS_NET_API_EXPORT IMPORT_TYPE
#    endif

#    ifndef KOKKOS_NET_API_EXTERNC
#        define KOKKOS_NET_API_EXTERNC EXTERN_C IMPORT_TYPE
#    endif

#    ifndef KOKKOS_NET_API_EXTERN
#        define KOKKOS_NET_API_EXTERN EXTERN IMPORT_TYPE
#    endif

#    ifndef KOKKOS_NET_API_IMPORT
#        define KOKKOS_NET_API_IMPORT EXPORT_TYPE
#    endif

#endif

using size_type = unsigned long long;

struct SharedMemoryData;

KOKKOS_NET_API_EXTERNC int SharedMemoryCreate(const wchar_t* name, size_type size, SharedMemoryData* data);

KOKKOS_NET_API_EXTERNC int SharedMemoryOpen(const wchar_t* name, size_type size, SharedMemoryData* data);

KOKKOS_NET_API_EXTERNC int SharedMemoryResize(const wchar_t* name, size_type size, SharedMemoryData* data);

KOKKOS_NET_API_EXTERNC void SharedMemoryClose(SharedMemoryData* data);

#if !defined(__wasm32__)

KOKKOS_NET_API_EXTERNC int SharedMemoryRegisterWithCuda(SharedMemoryData* data);

#endif

#if !defined(__wasm32__)

#    include <Types.hpp>
#    include <Print.hpp>
#    include <Array.hpp>

#    include <runtime.Kokkos/ViewTypes.hpp>

#else

#    define KOKKOS_INLINE_FUNCTION __inline

#endif

struct SharedMemoryData
{
    size_type Size;
    void*     HostAddress;
    void*     DeviceAddress;
#if defined(_WINDOWS)
    void* Handle;
#else
    int Handle;
#endif

    KOKKOS_INLINE_FUNCTION constexpr SharedMemoryData()  = default;
    KOKKOS_INLINE_FUNCTION constexpr ~SharedMemoryData() = default;

    KOKKOS_INLINE_FUNCTION constexpr SharedMemoryData(const SharedMemoryData& other) : Size(other.Size), HostAddress(other.HostAddress), DeviceAddress(other.DeviceAddress), Handle(other.Handle) {}

    KOKKOS_INLINE_FUNCTION constexpr SharedMemoryData(SharedMemoryData&& other) noexcept : Size(other.Size), HostAddress(other.HostAddress), DeviceAddress(other.DeviceAddress), Handle(other.Handle) {}

    KOKKOS_INLINE_FUNCTION constexpr SharedMemoryData& operator=(const SharedMemoryData& other)
    {
        if (this == &other)
        {
            return *this;
        }
        Size          = other.Size;
        HostAddress   = other.HostAddress;
        DeviceAddress = other.DeviceAddress;
        Handle        = other.Handle;
        return *this;
    }

    KOKKOS_INLINE_FUNCTION constexpr SharedMemoryData& operator=(SharedMemoryData&& other) noexcept
    {
        if (this == &other)
        {
            return *this;
        }
        Size          = other.Size;
        HostAddress   = other.HostAddress;
        DeviceAddress = other.DeviceAddress;
        Handle        = other.Handle;
        return *this;
    }

    ////~SharedMemoryData()
    ////{
    ////    if (HostAddress != nullptr || Handle != nullptr)
    ////    {
    ////        SharedMemoryClose(this);
    ////    }
    ////}
    // KOKKOS_INLINE_FUNCTION constexpr SharedMemoryData(const SharedMemoryData& other)     = delete;
    // KOKKOS_INLINE_FUNCTION constexpr SharedMemoryData(SharedMemoryData&& other) noexcept = delete;
    // KOKKOS_INLINE_FUNCTION constexpr SharedMemoryData& operator=(const SharedMemoryData& other) = delete;
    // KOKKOS_INLINE_FUNCTION constexpr SharedMemoryData& operator=(SharedMemoryData&& other) noexcept = delete;
};
