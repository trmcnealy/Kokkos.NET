#pragma once

#include "KokkosAPI.hpp"

#include <cuda.h>
#include <cuda_runtime_api.h>

#include <iostream>

enum cnmemStatus
{
    CNMEM_STATUS_SUCCESS = 0,
    CNMEM_STATUS_CUDA_ERROR,
    CNMEM_STATUS_INVALID_ARGUMENT,
    CNMEM_STATUS_NOT_INITIALIZED,
    CNMEM_STATUS_OUT_OF_MEMORY,
    CNMEM_STATUS_UNKNOWN_ERROR
};

enum cnmemManagerFlags
{
    CNMEM_FLAGS_DEFAULT      = 0,
    CNMEM_FLAGS_CANNOT_GROW  = 1,
    CNMEM_FLAGS_CANNOT_STEAL = 2,
    CNMEM_FLAGS_MANAGED      = 4,
};

struct cnmemDevice
{
    int           device;
    uint64   size;
    int           numStreams;
    cudaStream_t* streams;
    uint64*  streamSizes;
};

KOKKOS_NET_API_EXTERNC cnmemStatus cnmemInit(const int numDevices, const cnmemDevice* devices, unsigned flags);

KOKKOS_NET_API_EXTERNC cnmemStatus cnmemFinalize();

KOKKOS_NET_API_EXTERNC cnmemStatus cnmemRetain();

KOKKOS_NET_API_EXTERNC cnmemStatus cnmemRelease();

KOKKOS_NET_API_EXTERNC cnmemStatus cnmemRegisterStream(cudaStream_t stream);

KOKKOS_NET_API_EXTERNC cnmemStatus cnmemMalloc(void** ptr, uint64 size, cudaStream_t stream);

KOKKOS_NET_API_EXTERNC cnmemStatus cnmemFree(void* ptr, cudaStream_t stream);

KOKKOS_NET_API_EXTERNC cnmemStatus cnmemMemGetInfo(uint64* freeMem, uint64* totalMem, cudaStream_t stream);

KOKKOS_NET_API_EXTERNC cnmemStatus cnmemPrintMemoryState(FILE* file, cudaStream_t stream);

KOKKOS_NET_API_EXTERNC const char* cnmemGetErrorString(cnmemStatus status);
