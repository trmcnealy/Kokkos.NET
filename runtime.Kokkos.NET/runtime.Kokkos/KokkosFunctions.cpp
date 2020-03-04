
#include "runtime.Kokkos/KokkosApi.h"

void* Allocate(const ExecutionSpaceKind& execution_space, const size_type& arg_alloc_size) noexcept
{
    switch(execution_space)
    {
        case ExecutionSpaceKind::Serial:
        {
            return Kokkos::kokkos_malloc<Kokkos::Serial::memory_space>(arg_alloc_size);
        }
        case ExecutionSpaceKind::OpenMP:
        {
            return Kokkos::kokkos_malloc<Kokkos::OpenMP::memory_space>(arg_alloc_size);
        }
        case ExecutionSpaceKind::Cuda:
        {
            return Kokkos::kokkos_malloc<Kokkos::Cuda::memory_space>(arg_alloc_size);
        }
        default:
        {
            std::cout << "Allocate ExecutionSpace is not supported." << std::endl;
            return nullptr;
        }
    }
}

void* Reallocate(const ExecutionSpaceKind& execution_space, void* instance, const size_type& arg_alloc_size) noexcept
{
    switch(execution_space)
    {
        case ExecutionSpaceKind::Serial:
        {
            return Kokkos::kokkos_realloc<Kokkos::Serial::memory_space>(instance, arg_alloc_size);
        }
        case ExecutionSpaceKind::OpenMP:
        {
            return Kokkos::kokkos_realloc<Kokkos::OpenMP::memory_space>(instance, arg_alloc_size);
        }
        case ExecutionSpaceKind::Cuda:
        {
            return Kokkos::kokkos_realloc<Kokkos::Cuda::memory_space>(instance, arg_alloc_size);
        }
        default:
        {
            std::cout << "Allocate ExecutionSpace is not supported." << std::endl;
            return nullptr;
        }
    }
}

void Free(const ExecutionSpaceKind& execution_space, void* instance) noexcept
{
    switch(execution_space)
    {
        case ExecutionSpaceKind::Serial:
        {
            Kokkos::kokkos_free<Kokkos::Serial::memory_space>(instance);
            break;
        }
        case ExecutionSpaceKind::OpenMP:
        {
            Kokkos::kokkos_free<Kokkos::OpenMP::memory_space>(instance);
            break;
        }
        case ExecutionSpaceKind::Cuda:
        {
            Kokkos::kokkos_free<Kokkos::Cuda::memory_space>(instance);
            break;
        }
        default:
        {
            std::cout << "Allocate ExecutionSpace is not supported." << std::endl;
        }
    }
}

void Initialize(int& narg, char* arg[]) noexcept
{
    std::cout << "Initializing Kokkos." << std::endl;

    Kokkos::initialize(narg, arg);
}

void InitializeThreads(const int num_cpu_threads, const int gpu_device_id) noexcept
{
    std::cout << "Initializing Kokkos." << std::endl;

    Kokkos::InitArguments arguments;
    arguments.num_threads      = num_cpu_threads;
    arguments.num_numa         = 1;
    arguments.device_id        = gpu_device_id;
    arguments.ndevices         = 1;
    arguments.skip_device      = 9999;
    arguments.disable_warnings = false;

    Kokkos::initialize(arguments);
}

void InitializeArguments(const Kokkos::InitArguments& arguments) noexcept
{
    std::cout << "Initializing Kokkos." << std::endl;

    Kokkos::initialize(arguments);
}

void Finalize() noexcept
{
    std::cout << "Shuting down Kokkos." << std::endl;

    Kokkos::finalize();
}

void FinalizeAll() noexcept
{
    std::cout << "Shuting down Kokkos." << std::endl;

    Kokkos::finalize_all();
}

__attribute__((flatten)) bool IsInitialized() noexcept { return Kokkos::is_initialized(); }

void PrintConfiguration(const bool& detail) noexcept { Kokkos::print_configuration(std::cout, detail); }

#include <cuda_runtime.h>

unsigned int CudaGetDeviceCount() noexcept
{
    int deviceCount = 0;

    const cudaError_t error_id = cudaGetDeviceCount(&deviceCount);

    return deviceCount;
}

unsigned int CudaGetComputeCapability(unsigned int device_id) noexcept
{
    cudaSetDevice(device_id);

    cudaDeviceProp    deviceProp;
    const cudaError_t error_id = cudaGetDeviceProperties(&deviceProp, device_id);

    if(error_id == cudaSuccess)
    {
        return deviceProp.major * 100 + deviceProp.minor * 10;
    }

    return 0;
}
