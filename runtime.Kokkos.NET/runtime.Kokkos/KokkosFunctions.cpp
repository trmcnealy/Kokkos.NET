
#include "runtime.Kokkos/KokkosApi.h"

void* Allocate(const ExecutionSpaceKind execution_space, const size_type arg_alloc_size) noexcept
{
    switch (execution_space)
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

void* Reallocate(const ExecutionSpaceKind execution_space, void* instance, const size_type arg_alloc_size) noexcept
{
    // const ExecutionSpaceKind kind = execution_space;

    // using ExecutionSpaceType = typename ToTrait<decltype(kind)>::ExecutionSpace;

    // return Kokkos::kokkos_realloc<typename ExecutionSpaceType::memory_space>(instance, arg_alloc_size);

    switch (execution_space)
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

void Copy(const ExecutionSpaceKind src_execution_space, void* src, const ExecutionSpaceKind dest_execution_space, void* dest, const size_type size_in_bytes) noexcept
{
    switch (src_execution_space)
    {
        case ExecutionSpaceKind::Serial:
        {
            switch (dest_execution_space)
            {
                case ExecutionSpaceKind::Serial:
                {
                    cudaMemcpy(dest, src, size_in_bytes, cudaMemcpyHostToHost);
                    break;
                }
                case ExecutionSpaceKind::OpenMP:
                {
                    cudaMemcpy(dest, src, size_in_bytes, cudaMemcpyHostToHost);
                    break;
                }
                case ExecutionSpaceKind::Cuda:
                {
                    cudaMemcpy(dest, src, size_in_bytes, cudaMemcpyHostToDevice);
                    break;
                }
                default:
                {
                    std::cout << "Copy destination ExecutionSpace is not supported." << std::endl;
                }
            }
            break;
        }
        case ExecutionSpaceKind::OpenMP:
        {
            switch (dest_execution_space)
            {
                case ExecutionSpaceKind::Serial:
                {
                    cudaMemcpy(dest, src, size_in_bytes, cudaMemcpyHostToHost);
                    break;
                }
                case ExecutionSpaceKind::OpenMP:
                {
                    cudaMemcpy(dest, src, size_in_bytes, cudaMemcpyHostToHost);
                    break;
                }
                case ExecutionSpaceKind::Cuda:
                {
                    cudaMemcpy(dest, src, size_in_bytes, cudaMemcpyHostToDevice);
                    break;
                }
                default:
                {
                    std::cout << "Copy destination ExecutionSpace is not supported." << std::endl;
                }
            }
            break;
        }
        case ExecutionSpaceKind::Cuda:
        {
            switch (dest_execution_space)
            {
                case ExecutionSpaceKind::Serial:
                {
                    cudaMemcpy(dest, src, size_in_bytes, cudaMemcpyDeviceToHost);
                    break;
                }
                case ExecutionSpaceKind::OpenMP:
                {
                    cudaMemcpy(dest, src, size_in_bytes, cudaMemcpyDeviceToHost);
                    break;
                }
                case ExecutionSpaceKind::Cuda:
                {
                    cudaMemcpy(dest, src, size_in_bytes, cudaMemcpyDeviceToDevice);
                    break;
                }
                default:
                {
                    std::cout << "Copy destination ExecutionSpace is not supported." << std::endl;
                }
            }
            break;
        }
        default:
        {
            std::cout << "Copy source ExecutionSpace is not supported." << std::endl;
        }
    }
}

void Free(const ExecutionSpaceKind execution_space, void* instance) noexcept
{
    switch (execution_space)
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

void Initialize(const int narg, char* arg[]) noexcept
{
    cudaDeviceReset();

    std::cout << "Initializing Kokkos." << std::endl;

    Kokkos::initialize((int&)narg, arg);
}

void InitializeSerial() noexcept
{
    std::cout << "Initializing Kokkos::Serial." << std::endl;

    Kokkos::Serial::impl_initialize();
}

void InitializeOpenMP(const int num_threads) noexcept
{
    std::cout << "Initializing Kokkos::OpenMP." << std::endl;

    Kokkos::OpenMP::impl_initialize(num_threads);
}

void InitializeCuda(const int use_gpu) noexcept
{
    cudaDeviceReset();

    std::cout << "Initializing Kokkos::Cuda." << std::endl;

    if (use_gpu > -1)
    {
        Kokkos::Cuda::impl_initialize(Kokkos::Cuda::SelectDevice(use_gpu));
    }
    else
    {
        Kokkos::Cuda::impl_initialize();
    }
}

void InitializeThreads(const int num_cpu_threads, const int gpu_device_id) noexcept
{
    cudaDeviceReset();

    std::cout << "Initializing Kokkos." << std::endl;

    Kokkos::InitArguments arguments;
    arguments.num_threads      = num_cpu_threads;
    arguments.num_numa         = 1;
    arguments.device_id        = gpu_device_id;
    arguments.ndevices         = 1;
    arguments.skip_device      = 9999;
    arguments.disable_warnings = false;
    
    try
    {
        Kokkos::initialize(arguments);
    }
    catch (const std::exception& exp)
    {
        std::cerr << exp.what() << std::endl;
    }
    catch (const std::logic_error& exp)
    {
        std::cerr << exp.what() << std::endl;
    }
    catch (const std::runtime_error& exp)
    {
        std::cerr << exp.what() << std::endl;
    }
    catch (const std::bad_exception& exp)
    {
        std::cerr << exp.what() << std::endl;
    }
    catch (...)
    {
        std::cerr << "Caught unknown exception." << std::endl;
    }
}

void InitializeArguments(Kokkos::InitArguments args) noexcept
{
    cudaDeviceReset();

    std::cout << "Initializing Kokkos." << std::endl;

    Kokkos::InitArguments arguments;
    arguments.num_threads      = args.num_threads;
    arguments.num_numa         = args.num_numa;
    arguments.device_id        = args.device_id;
    arguments.ndevices         = args.ndevices;
    arguments.skip_device      = args.skip_device;
    arguments.disable_warnings = args.disable_warnings;

    try
    {
        Kokkos::initialize(arguments);
    }
    catch (const std::exception& exp)
    {
        std::cerr << exp.what() << std::endl;
    }
    catch (const std::logic_error& exp)
    {
        std::cerr << exp.what() << std::endl;
    }
    catch (const std::runtime_error& exp)
    {
        std::cerr << exp.what() << std::endl;
    }
    catch (const std::bad_exception& exp)
    {
        std::cerr << exp.what() << std::endl;
    }
    catch (...)
    {
        std::cerr << "Caught unknown exception." << std::endl;
    }
}

void Finalize() noexcept
{
    std::cout << "Shuting down Kokkos." << std::endl;

    Kokkos::finalize();
}

void FinalizeSerial() noexcept
{
    std::cout << "Shuting down Kokkos::Serial." << std::endl;

    Kokkos::Serial::impl_finalize();
}

void FinalizeOpenMP() noexcept
{
    std::cout << "Shuting down Kokkos::OpenMP." << std::endl;

    Kokkos::OpenMP::impl_finalize();
}

void FinalizeCuda() noexcept
{
    std::cout << "Shuting down Kokkos::Cuda." << std::endl;

    Kokkos::Cuda::impl_finalize();
}

void FinalizeAll() noexcept
{
    std::cout << "Shuting down Kokkos." << std::endl;

    Kokkos::finalize_all();
}

__attribute__((flatten)) bool IsInitialized() noexcept
{
    return Kokkos::is_initialized();
}

void PrintConfiguration(const bool detail) noexcept
{
    Kokkos::print_configuration(std::cout, detail);
}

#include <cuda_runtime.h>

unsigned int CudaGetDeviceCount() noexcept
{
    int deviceCount = 0;

    const cudaError_t error_id = cudaGetDeviceCount(&deviceCount);

    return deviceCount;
}

unsigned int CudaGetComputeCapability(const unsigned int device_id) noexcept
{
    cudaSetDevice(device_id);

    cudaDeviceProp    deviceProp;
    const cudaError_t error_id = cudaGetDeviceProperties(&deviceProp, device_id);

    if (error_id == cudaSuccess)
    {
        return deviceProp.major * 100 + deviceProp.minor * 10;
    }

    return 0;
}
