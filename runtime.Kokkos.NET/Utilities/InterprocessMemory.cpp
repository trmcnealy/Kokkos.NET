
#include "InterprocessMemory.h"

#include <MathExtensions.hpp>
#include <Exceptions.h>

#include <windef.h>
#include <winbase.h>

#include <cuda.h>

std::vector<CUdevice> GetAllSupportedCudaDevices(CUdevice cuDevice)
{
    int num_cuda_devices;
    int capable;
    int attributeVal;

    CHECK(cuDeviceGetCount(&num_cuda_devices));

    std::vector<CUdevice> supportedCudaDevices;

    supportedCudaDevices.push_back(cuDevice);

    for (int dev = 0; dev < num_cuda_devices; dev++)
    {
        capable      = 0;
        attributeVal = 0;

        if (dev == cuDevice)
        {
            continue;
        }

        CHECK(cuDeviceCanAccessPeer(&capable, cuDevice, dev));

        if (!capable)
        {
            continue;
        }

        CHECK(cuDeviceGetAttribute(&attributeVal, CU_DEVICE_ATTRIBUTE_VIRTUAL_ADDRESS_MANAGEMENT_SUPPORTED, cuDevice));

        if (attributeVal == 0)
        {
            continue;
        }

        supportedCudaDevices.push_back(dev);
    }
    return supportedCudaDevices;
}

void* IpcCreate(CONST(ExecutionSpaceKind) execution_space, CONST(size_type) size, CONST(NativeString) label) noexcept
{
    switch (execution_space)
    {
        case ExecutionSpaceKind::Serial:
        {
            void* handle = CreateFileMappingA(INVALID_HANDLE_VALUE, nullptr, PAGE_READWRITE, 0, static_cast<uint32>(size), label.Bytes);

#if defined(__CUDACC__) || defined(__NVCC__)

            CHECK(cuMemHostRegister(handle, size, CU_MEMHOSTREGISTER_PORTABLE));

#endif

            Kokkos::InterprocessMemory<Kokkos::Serial>* ipc = new Kokkos::InterprocessMemory<Kokkos::Serial>(handle, handle, size, label.Bytes);

            return new Teuchos::RCP<Kokkos::InterprocessMemory<Kokkos::Serial>>(ipc);
        }
        case ExecutionSpaceKind::OpenMP:
        {
            void* handle = CreateFileMappingA(INVALID_HANDLE_VALUE, nullptr, PAGE_READWRITE, 0, static_cast<uint32>(size), label.Bytes);

            Kokkos::InterprocessMemory<Kokkos::OpenMP>* ipc = new Kokkos::InterprocessMemory<Kokkos::OpenMP>(handle, handle, size, label.Bytes);

            return new Teuchos::RCP<Kokkos::InterprocessMemory<Kokkos::OpenMP>>(ipc);
        }
        case ExecutionSpaceKind::Cuda:
        {
            const int32 deviceID = Kokkos::Cuda().cuda_device();

            CUdevice cuDevice;
            CHECK(cuDeviceGet(&cuDevice, deviceID));

            CUdeviceptr* dptr = nullptr;

            const std::vector<CUdevice> supportedCudaDevices = GetAllSupportedCudaDevices(cuDevice);

            const size_type nDevices        = supportedCudaDevices.size();
            size_type       min_granularity = 0;

            CUmemAllocationProp prop = {};
            prop.type                = CU_MEM_ALLOCATION_TYPE_PINNED;
            prop.location.type       = CU_MEM_LOCATION_TYPE_DEVICE;

            for (size_type devicesIdx = 0; devicesIdx < nDevices; ++devicesIdx)
            {
                size_type granularity = 0;

                prop.location.id = supportedCudaDevices[devicesIdx];

                CHECK(cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));

                if (min_granularity < granularity)
                {
                    min_granularity = granularity;
                }
            }

            const size_type allocationSize = round_up(size, nDevices * min_granularity);
            const size_type stripeSize     = allocationSize / nDevices;

            CHECK(cuMemAddressReserve(dptr, size, 0, 0, 0));

            void* shareableHandle;

            for (size_type devicesIdx = 0; devicesIdx < nDevices; ++devicesIdx)
            {
                prop.location.id = supportedCudaDevices[devicesIdx];

                CUmemGenericAllocationHandle allocationHandle;
                CHECK(cuMemCreate(&allocationHandle, stripeSize, &prop, 0));

                if (cuDevice == supportedCudaDevices[devicesIdx])
                {
                    CHECK(cuMemExportToShareableHandle(&shareableHandle, allocationHandle, CUmemAllocationHandleType::CU_MEM_HANDLE_TYPE_WIN32, 0));
                }

                CHECK(cuMemMap(*dptr + (stripeSize * devicesIdx), stripeSize, 0, allocationHandle, 0));

                CHECK(cuMemRelease(allocationHandle));
            }

            CUmemAccessDesc accessDescriptor = {};
            accessDescriptor.location.type   = CU_MEM_LOCATION_TYPE_DEVICE;
            accessDescriptor.location.id     = cuDevice;
            accessDescriptor.flags           = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;

            CHECK(cuMemSetAccess(*dptr, size, &accessDescriptor, 1));

            void* memoryPtr = reinterpret_cast<void*>(dptr); // Kokkos::kokkos_malloc<Kokkos::Cuda::memory_space>(size);

            // CUipcMemHandle handle;

            // CUresult error = cuIpcGetMemHandle(&handle, reinterpret_cast<CUdeviceptr>(memoryPtr));

            Kokkos::InterprocessMemory<Kokkos::Cuda>* ipc = new Kokkos::InterprocessMemory<Kokkos::Cuda>(memoryPtr, shareableHandle, allocationSize, label.Bytes);

            return new Teuchos::RCP<Kokkos::InterprocessMemory<Kokkos::Cuda>>(ipc);
        }
        case ExecutionSpaceKind::Unknown:
        {
            break;
        }
    }

    std::cout << "The ExecutionSpace is not supported." << std::endl;
    return nullptr;
}

void* IpcCreateFrom(CONST(ExecutionSpaceKind) execution_space, void* memoryPtr, CONST(size_type) size, CONST(NativeString) label) noexcept
{
    switch (execution_space)
    {
        case ExecutionSpaceKind::Serial:
        {
            void* handle = CreateFileMappingA(INVALID_HANDLE_VALUE, nullptr, PAGE_READWRITE, 0, static_cast<uint32>(size), label.Bytes);

            Kokkos::InterprocessMemory<Kokkos::Serial>* ipc = new Kokkos::InterprocessMemory<Kokkos::Serial>(handle, handle, size, label.Bytes);

            return new Teuchos::RCP<Kokkos::InterprocessMemory<Kokkos::Serial>>(ipc);
        }
        case ExecutionSpaceKind::OpenMP:
        {
            void* handle = CreateFileMappingA(INVALID_HANDLE_VALUE, nullptr, PAGE_READWRITE, 0, static_cast<uint32>(size), label.Bytes);

            Kokkos::InterprocessMemory<Kokkos::OpenMP>* ipc = new Kokkos::InterprocessMemory<Kokkos::OpenMP>(handle, handle, size, label.Bytes);

            return new Teuchos::RCP<Kokkos::InterprocessMemory<Kokkos::OpenMP>>(ipc);
        }
        case ExecutionSpaceKind::Cuda:
        {
            CUdeviceptr* dptr = reinterpret_cast<CUdeviceptr*>(memoryPtr);

            const int32 deviceID = Kokkos::Cuda().cuda_device();

            CUdevice cuDevice;
            CHECK(cuDeviceGet(&cuDevice, deviceID));

            const std::vector<CUdevice> supportedCudaDevices = GetAllSupportedCudaDevices(cuDevice);

            const size_type nDevices        = supportedCudaDevices.size();
            size_type       min_granularity = 0;

            CUmemAllocationProp prop = {};
            prop.type                = CU_MEM_ALLOCATION_TYPE_PINNED;
            prop.location.type       = CU_MEM_LOCATION_TYPE_DEVICE;

            for (size_type devicesIdx = 0; devicesIdx < nDevices; ++devicesIdx)
            {
                size_type granularity = 0;

                prop.location.id = supportedCudaDevices[devicesIdx];

                CHECK(cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));

                if (min_granularity < granularity)
                {
                    min_granularity = granularity;
                }
            }

            const size_type allocationSize = round_up(size, nDevices * min_granularity);
            const size_type stripeSize     = allocationSize / nDevices;

            void* shareableHandle;

            for (size_type devicesIdx = 0; devicesIdx < nDevices; ++devicesIdx)
            {
                prop.location.id = supportedCudaDevices[devicesIdx];

                CUmemGenericAllocationHandle allocationHandle;
                CHECK(cuMemCreate(&allocationHandle, stripeSize, &prop, 0));

                if (cuDevice == supportedCudaDevices[devicesIdx])
                {
                    CHECK(cuMemExportToShareableHandle(&shareableHandle, allocationHandle, CUmemAllocationHandleType::CU_MEM_HANDLE_TYPE_WIN32, 0));
                }

                CHECK(cuMemMap(*dptr + (stripeSize * devicesIdx), stripeSize, 0, allocationHandle, 0));

                CHECK(cuMemRelease(allocationHandle));
            }

            CUmemAccessDesc accessDescriptor = {};
            accessDescriptor.location.type   = CU_MEM_LOCATION_TYPE_DEVICE;
            accessDescriptor.location.id     = cuDevice;
            accessDescriptor.flags           = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;

            CHECK(cuMemSetAccess(*dptr, size, &accessDescriptor, 1));

            // CUipcMemHandle handle;

            // CUresult error = cuIpcGetMemHandle(&handle, reinterpret_cast<CUdeviceptr>(memoryPtr));

            Kokkos::InterprocessMemory<Kokkos::Cuda>* ipc = new Kokkos::InterprocessMemory<Kokkos::Cuda>(memoryPtr, shareableHandle, allocationSize, label.Bytes);

            return new Teuchos::RCP<Kokkos::InterprocessMemory<Kokkos::Cuda>>(ipc);
        }
        case ExecutionSpaceKind::Unknown:
        {
            break;
        }
    }

    std::cout << "The ExecutionSpace is not supported." << std::endl;
    return nullptr;
}

void* IpcOpenExisting(CONST(ExecutionSpaceKind) execution_space, void* instance) noexcept
{
    switch (execution_space)
    {
        case ExecutionSpaceKind::Serial:
        {
            const Teuchos::RCP<Kokkos::InterprocessMemory<Kokkos::Serial>> other_ipc = *reinterpret_cast<Teuchos::RCP<Kokkos::InterprocessMemory<Kokkos::Serial>>*>(instance);

            void* handle = OpenFileMappingA(FILE_MAP_ALL_ACCESS, FALSE, other_ipc->GetLabel());

            LARGE_INTEGER large_integer;
            GetFileSizeEx(handle, &large_integer);
            const size_type size = large_integer.QuadPart;

            Kokkos::InterprocessMemory<Kokkos::Serial>* ipc = new Kokkos::InterprocessMemory<Kokkos::Serial>(handle, handle, size, other_ipc->GetLabel());

            return new Teuchos::RCP<Kokkos::InterprocessMemory<Kokkos::Serial>>(ipc);
        }
        case ExecutionSpaceKind::OpenMP:
        {
            const Teuchos::RCP<Kokkos::InterprocessMemory<Kokkos::OpenMP>> other_ipc = *reinterpret_cast<Teuchos::RCP<Kokkos::InterprocessMemory<Kokkos::OpenMP>>*>(instance);

            void* handle = OpenFileMappingA(FILE_MAP_ALL_ACCESS, FALSE, other_ipc->GetLabel());

            LARGE_INTEGER large_integer;
            GetFileSizeEx(handle, &large_integer);
            const size_type size = large_integer.QuadPart;

            Kokkos::InterprocessMemory<Kokkos::OpenMP>* ipc = new Kokkos::InterprocessMemory<Kokkos::OpenMP>(handle, handle, size, other_ipc->GetLabel());

            return new Teuchos::RCP<Kokkos::InterprocessMemory<Kokkos::OpenMP>>(ipc);
        }
        case ExecutionSpaceKind::Cuda:
        {
            const Teuchos::RCP<Kokkos::InterprocessMemory<Kokkos::Cuda>> other_ipc = *reinterpret_cast<Teuchos::RCP<Kokkos::InterprocessMemory<Kokkos::Cuda>>*>(instance);

            // CUresult error = cuIpcOpenMemHandle(&memoryPtr, *handle, CU_IPC_MEM_LAZY_ENABLE_PEER_ACCESS);

            Kokkos::InterprocessMemory<Kokkos::Cuda>* ipc = new Kokkos::InterprocessMemory<Kokkos::Cuda>(other_ipc->GetMemoryPointer(), other_ipc->GetDeviceHandle(), other_ipc->GetSize(), other_ipc->GetLabel());

            return new Teuchos::RCP<Kokkos::InterprocessMemory<Kokkos::Cuda>>(ipc);
        }
        case ExecutionSpaceKind::Unknown:
        {
            break;
        }
    }

    std::cout << "The ExecutionSpace is not supported." << std::endl;
    return nullptr;
}

void IpcDestory(CONST(ExecutionSpaceKind) execution_space, void* instance) noexcept
{
    switch (execution_space)
    {
        case ExecutionSpaceKind::Serial:
        {
            Teuchos::RCP<Kokkos::InterprocessMemory<Kokkos::Serial>> ipc = *reinterpret_cast<Teuchos::RCP<Kokkos::InterprocessMemory<Kokkos::Serial>>*>(instance);

            if (ipc->GetDeviceHandle())
            {
#ifdef _WINDOWS
                ::UnmapViewOfFile(ipc->GetDeviceHandle());
#else
                ::munmap(ipc->GetDeviceHandle(), _filesize);
#endif
            }

            if (ipc->GetMemoryPointer())
            {
                Kokkos::kokkos_free<Kokkos::Serial::memory_space>(ipc->GetMemoryPointer());
            }

            Kokkos::kokkos_free<Kokkos::Serial::memory_space>(instance);

            ipc = Teuchos::null;
            break;
        }
        case ExecutionSpaceKind::OpenMP:
        {
            Teuchos::RCP<Kokkos::InterprocessMemory<Kokkos::OpenMP>> ipc = *reinterpret_cast<Teuchos::RCP<Kokkos::InterprocessMemory<Kokkos::OpenMP>>*>(instance);

            if (ipc->GetDeviceHandle())
            {
#ifdef _WINDOWS
                ::UnmapViewOfFile(ipc->GetDeviceHandle());
#else
                ::munmap(ipc->GetDeviceHandle(), _filesize);
#endif
            }

            if (ipc->GetMemoryPointer())
            {
                Kokkos::kokkos_free<Kokkos::OpenMP::memory_space>(ipc->GetMemoryPointer());
            }

            Kokkos::kokkos_free<Kokkos::OpenMP::memory_space>(instance);

            ipc = Teuchos::null;
            break;
        }
        case ExecutionSpaceKind::Cuda:
        {
            Teuchos::RCP<Kokkos::InterprocessMemory<Kokkos::Cuda>> ipc = *reinterpret_cast<Teuchos::RCP<Kokkos::InterprocessMemory<Kokkos::Cuda>>*>(instance);

            CHECK(cuMemUnmap(reinterpret_cast<CUdeviceptr>(ipc->GetMemoryPointer()), ipc->GetSize()));
            CHECK(cuMemAddressFree(reinterpret_cast<CUdeviceptr>(ipc->GetMemoryPointer()), ipc->GetSize()));

            // if (ipc->GetDeviceHandle())
            //{
            //    cuIpcCloseMemHandle(reinterpret_cast<CUdeviceptr>(ipc->GetDeviceHandle()));
            //}

            // if (ipc->GetMemoryPointer())
            //{
            //    Kokkos::kokkos_free<Kokkos::Cuda::memory_space>(ipc->GetMemoryPointer());
            //}

            ipc = Teuchos::null;
            break;
        }
        case ExecutionSpaceKind::Unknown:
        default:
        {
            std::cout << "The ExecutionSpace is not supported." << std::endl;
        }
    }
}

void IpcClose(CONST(ExecutionSpaceKind) execution_space, void* instance) noexcept
{

    switch (execution_space)
    {
        case ExecutionSpaceKind::Serial:
        {
            const Teuchos::RCP<Kokkos::InterprocessMemory<Kokkos::Serial>> ipc = *reinterpret_cast<Teuchos::RCP<Kokkos::InterprocessMemory<Kokkos::Serial>>*>(instance);

            if (ipc->GetDeviceHandle())
            {
#ifdef _WINDOWS
                ::UnmapViewOfFile(ipc->GetDeviceHandle());
#else
                ::munmap(ipc->GetDeviceHandle(), _filesize);
#endif
            }
        }
        case ExecutionSpaceKind::OpenMP:
        {
            const Teuchos::RCP<Kokkos::InterprocessMemory<Kokkos::OpenMP>> ipc = *reinterpret_cast<Teuchos::RCP<Kokkos::InterprocessMemory<Kokkos::OpenMP>>*>(instance);

            if (ipc->GetDeviceHandle())
            {
#ifdef _WINDOWS
                ::UnmapViewOfFile(ipc->GetDeviceHandle());
#else
                ::munmap(ipc->GetDeviceHandle(), _filesize);
#endif
            }
        }
        case ExecutionSpaceKind::Cuda:
        {
            const Teuchos::RCP<Kokkos::InterprocessMemory<Kokkos::Cuda>> ipc = *reinterpret_cast<Teuchos::RCP<Kokkos::InterprocessMemory<Kokkos::Cuda>>*>(instance);

            CHECK(cuMemUnmap(reinterpret_cast<CUdeviceptr>(ipc->GetMemoryPointer()), ipc->GetSize()));
            CHECK(cuMemAddressFree(reinterpret_cast<CUdeviceptr>(ipc->GetMemoryPointer()), ipc->GetSize()));

            // CUresult error = cuIpcCloseMemHandle(reinterpret_cast<CUdeviceptr>(ipc->GetDeviceHandle()));
        }
        case ExecutionSpaceKind::Unknown:
        {
            break;
        }
    }

    std::cout << "The ExecutionSpace is not supported." << std::endl;
}

void* IpcGetMemoryPointer(CONST(ExecutionSpaceKind) execution_space, void* instance) noexcept
{

    switch (execution_space)
    {
        case ExecutionSpaceKind::Serial:
        {
            const Teuchos::RCP<Kokkos::InterprocessMemory<Kokkos::Serial>> ipc = *reinterpret_cast<Teuchos::RCP<Kokkos::InterprocessMemory<Kokkos::Serial>>*>(instance);

            return ipc->GetMemoryPointer();
        }
        case ExecutionSpaceKind::OpenMP:
        {
            const Teuchos::RCP<Kokkos::InterprocessMemory<Kokkos::OpenMP>> ipc = *reinterpret_cast<Teuchos::RCP<Kokkos::InterprocessMemory<Kokkos::OpenMP>>*>(instance);

            return ipc->GetMemoryPointer();
        }
        case ExecutionSpaceKind::Cuda:
        {
            const Teuchos::RCP<Kokkos::InterprocessMemory<Kokkos::Cuda>> ipc = *reinterpret_cast<Teuchos::RCP<Kokkos::InterprocessMemory<Kokkos::Cuda>>*>(instance);

            return ipc->GetMemoryPointer();
        }
        case ExecutionSpaceKind::Unknown:
        {
            break;
        }
    }

    std::cout << "The ExecutionSpace is not supported." << std::endl;
    return nullptr;
}

void* IpcGetDeviceHandle(CONST(ExecutionSpaceKind) execution_space, void* instance) noexcept
{

    switch (execution_space)
    {
        case ExecutionSpaceKind::Serial:
        {
            const Teuchos::RCP<Kokkos::InterprocessMemory<Kokkos::Serial>> ipc = *reinterpret_cast<Teuchos::RCP<Kokkos::InterprocessMemory<Kokkos::Serial>>*>(instance);

            return ipc->GetDeviceHandle();
        }
        case ExecutionSpaceKind::OpenMP:
        {
            const Teuchos::RCP<Kokkos::InterprocessMemory<Kokkos::OpenMP>> ipc = *reinterpret_cast<Teuchos::RCP<Kokkos::InterprocessMemory<Kokkos::OpenMP>>*>(instance);

            return ipc->GetDeviceHandle();
        }
        case ExecutionSpaceKind::Cuda:
        {
            const Teuchos::RCP<Kokkos::InterprocessMemory<Kokkos::Cuda>> ipc = *reinterpret_cast<Teuchos::RCP<Kokkos::InterprocessMemory<Kokkos::Cuda>>*>(instance);

            return ipc->GetDeviceHandle();
        }
        case ExecutionSpaceKind::Unknown:
        {
            break;
        }
    }

    std::cout << "The ExecutionSpace is not supported." << std::endl;
    return nullptr;
}

size_type IpcGetSize(CONST(ExecutionSpaceKind) execution_space, void* instance) noexcept
{

    switch (execution_space)
    {
        case ExecutionSpaceKind::Serial:
        case ExecutionSpaceKind::OpenMP:
        {
            const Teuchos::RCP<Kokkos::InterprocessMemory<Kokkos::OpenMP>> ipc = *reinterpret_cast<Teuchos::RCP<Kokkos::InterprocessMemory<Kokkos::OpenMP>>*>(instance);

            return ipc->GetSize();
        }
        case ExecutionSpaceKind::Cuda:
        {
            const Teuchos::RCP<Kokkos::InterprocessMemory<Kokkos::Cuda>> ipc = *reinterpret_cast<Teuchos::RCP<Kokkos::InterprocessMemory<Kokkos::Cuda>>*>(instance);

            return ipc->GetSize();
        }
        case ExecutionSpaceKind::Unknown:
        {
            break;
        }
    }

    std::cout << "The ExecutionSpace is not supported." << std::endl;
    return 0;
}

#define DEF_TEMPLATE(TYPE_NAME, TYPE, EXECUTION_SPACE)                                                                                                                                                                               \
    case DataTypeKind::TYPE_NAME:                                                                                                                                                                                                    \
    {                                                                                                                                                                                                                                \
        typedef Kokkos::InterprocessMemory<Kokkos::EXECUTION_SPACE> interprocess_memory_type;                                                                                                                                        \
        void* ptr = (*reinterpret_cast<Teuchos::RCP<interprocess_memory_type>*>(instance))->MakeViewFromPointer<TYPE>(arg_N0, arg_N1, arg_N2, arg_N3, arg_N4, arg_N5, arg_N6, arg_N7);                                               \
        return ptr;                                                                                                                                                                                                                  \
    }

#define TEMPLATE_RANK0(DEF, EXECUTION_SPACE)                                                                                                                                                                                         \
    DEF(Single, float, EXECUTION_SPACE)                                                                                                                                                                                              \
    DEF(Double, double, EXECUTION_SPACE)                                                                                                                                                                                             \
    DEF(Bool, bool, EXECUTION_SPACE)                                                                                                                                                                                                 \
    DEF(Int8, int8, EXECUTION_SPACE)                                                                                                                                                                                                 \
    DEF(UInt8, uint8, EXECUTION_SPACE)                                                                                                                                                                                               \
    DEF(Int16, int16, EXECUTION_SPACE)                                                                                                                                                                                               \
    DEF(UInt16, uint16, EXECUTION_SPACE)                                                                                                                                                                                             \
    DEF(Int32, int32, EXECUTION_SPACE)                                                                                                                                                                                               \
    DEF(UInt32, uint32, EXECUTION_SPACE)                                                                                                                                                                                             \
    DEF(Int64, int64, EXECUTION_SPACE)                                                                                                                                                                                               \
    DEF(UInt64, uint64, EXECUTION_SPACE)                                                                                                                                                                                             \
    DEF(Char, wchar_t, EXECUTION_SPACE)

#define TEMPLATE_RANK1(DEF, EXECUTION_SPACE)                                                                                                                                                                                         \
    DEF(Single, float, EXECUTION_SPACE)                                                                                                                                                                                              \
    DEF(Double, double, EXECUTION_SPACE)                                                                                                                                                                                             \
    DEF(Bool, bool, EXECUTION_SPACE)                                                                                                                                                                                                 \
    DEF(Int8, int8, EXECUTION_SPACE)                                                                                                                                                                                                 \
    DEF(UInt8, uint8, EXECUTION_SPACE)                                                                                                                                                                                               \
    DEF(Int16, int16, EXECUTION_SPACE)                                                                                                                                                                                               \
    DEF(UInt16, uint16, EXECUTION_SPACE)                                                                                                                                                                                             \
    DEF(Int32, int32, EXECUTION_SPACE)                                                                                                                                                                                               \
    DEF(UInt32, uint32, EXECUTION_SPACE)                                                                                                                                                                                             \
    DEF(Int64, int64, EXECUTION_SPACE)                                                                                                                                                                                               \
    DEF(UInt64, uint64, EXECUTION_SPACE)                                                                                                                                                                                             \
    DEF(Char, wchar_t, EXECUTION_SPACE)

#define TEMPLATE_RANK2(DEF, EXECUTION_SPACE)                                                                                                                                                                                         \
    DEF(Single, float, EXECUTION_SPACE)                                                                                                                                                                                              \
    DEF(Double, double, EXECUTION_SPACE)                                                                                                                                                                                             \
    DEF(Bool, bool, EXECUTION_SPACE)                                                                                                                                                                                                 \
    DEF(Int8, int8, EXECUTION_SPACE)                                                                                                                                                                                                 \
    DEF(UInt8, uint8, EXECUTION_SPACE)                                                                                                                                                                                               \
    DEF(Int16, int16, EXECUTION_SPACE)                                                                                                                                                                                               \
    DEF(UInt16, uint16, EXECUTION_SPACE)                                                                                                                                                                                             \
    DEF(Int32, int32, EXECUTION_SPACE)                                                                                                                                                                                               \
    DEF(UInt32, uint32, EXECUTION_SPACE)                                                                                                                                                                                             \
    DEF(Int64, int64, EXECUTION_SPACE)                                                                                                                                                                                               \
    DEF(UInt64, uint64, EXECUTION_SPACE)                                                                                                                                                                                             \
    DEF(Char, wchar_t, EXECUTION_SPACE)

#define TEMPLATE_RANK3(DEF, EXECUTION_SPACE)                                                                                                                                                                                         \
    DEF(Single, float, EXECUTION_SPACE)                                                                                                                                                                                              \
    DEF(Double, double, EXECUTION_SPACE)                                                                                                                                                                                             \
    DEF(Bool, bool, EXECUTION_SPACE)                                                                                                                                                                                                 \
    DEF(Int8, int8, EXECUTION_SPACE)                                                                                                                                                                                                 \
    DEF(UInt8, uint8, EXECUTION_SPACE)                                                                                                                                                                                               \
    DEF(Int16, int16, EXECUTION_SPACE)                                                                                                                                                                                               \
    DEF(UInt16, uint16, EXECUTION_SPACE)                                                                                                                                                                                             \
    DEF(Int32, int32, EXECUTION_SPACE)                                                                                                                                                                                               \
    DEF(UInt32, uint32, EXECUTION_SPACE)                                                                                                                                                                                             \
    DEF(Int64, int64, EXECUTION_SPACE)                                                                                                                                                                                               \
    DEF(UInt64, uint64, EXECUTION_SPACE)                                                                                                                                                                                             \
    DEF(Char, wchar_t, EXECUTION_SPACE)

#define TEMPLATE_RANK4(DEF, EXECUTION_SPACE)                                                                                                                                                                                         \
    DEF(Single, float, EXECUTION_SPACE)                                                                                                                                                                                              \
    DEF(Double, double, EXECUTION_SPACE)                                                                                                                                                                                             \
    DEF(Bool, bool, EXECUTION_SPACE)                                                                                                                                                                                                 \
    DEF(Int8, int8, EXECUTION_SPACE)                                                                                                                                                                                                 \
    DEF(UInt8, uint8, EXECUTION_SPACE)                                                                                                                                                                                               \
    DEF(Int16, int16, EXECUTION_SPACE)                                                                                                                                                                                               \
    DEF(UInt16, uint16, EXECUTION_SPACE)                                                                                                                                                                                             \
    DEF(Int32, int32, EXECUTION_SPACE)                                                                                                                                                                                               \
    DEF(UInt32, uint32, EXECUTION_SPACE)                                                                                                                                                                                             \
    DEF(Int64, int64, EXECUTION_SPACE)                                                                                                                                                                                               \
    DEF(UInt64, uint64, EXECUTION_SPACE)                                                                                                                                                                                             \
    DEF(Char, wchar_t, EXECUTION_SPACE)

#define TEMPLATE_RANK5(DEF, EXECUTION_SPACE)                                                                                                                                                                                         \
    DEF(Single, float, EXECUTION_SPACE)                                                                                                                                                                                              \
    DEF(Double, double, EXECUTION_SPACE)                                                                                                                                                                                             \
    DEF(Bool, bool, EXECUTION_SPACE)                                                                                                                                                                                                 \
    DEF(Int8, int8, EXECUTION_SPACE)                                                                                                                                                                                                 \
    DEF(UInt8, uint8, EXECUTION_SPACE)                                                                                                                                                                                               \
    DEF(Int16, int16, EXECUTION_SPACE)                                                                                                                                                                                               \
    DEF(UInt16, uint16, EXECUTION_SPACE)                                                                                                                                                                                             \
    DEF(Int32, int32, EXECUTION_SPACE)                                                                                                                                                                                               \
    DEF(UInt32, uint32, EXECUTION_SPACE)                                                                                                                                                                                             \
    DEF(Int64, int64, EXECUTION_SPACE)                                                                                                                                                                                               \
    DEF(UInt64, uint64, EXECUTION_SPACE)                                                                                                                                                                                             \
    DEF(Char, wchar_t, EXECUTION_SPACE)

#define TEMPLATE_RANK6(DEF, EXECUTION_SPACE)                                                                                                                                                                                         \
    DEF(Single, float, EXECUTION_SPACE)                                                                                                                                                                                              \
    DEF(Double, double, EXECUTION_SPACE)                                                                                                                                                                                             \
    DEF(Bool, bool, EXECUTION_SPACE)                                                                                                                                                                                                 \
    DEF(Int8, int8, EXECUTION_SPACE)                                                                                                                                                                                                 \
    DEF(UInt8, uint8, EXECUTION_SPACE)                                                                                                                                                                                               \
    DEF(Int16, int16, EXECUTION_SPACE)                                                                                                                                                                                               \
    DEF(UInt16, uint16, EXECUTION_SPACE)                                                                                                                                                                                             \
    DEF(Int32, int32, EXECUTION_SPACE)                                                                                                                                                                                               \
    DEF(UInt32, uint32, EXECUTION_SPACE)                                                                                                                                                                                             \
    DEF(Int64, int64, EXECUTION_SPACE)                                                                                                                                                                                               \
    DEF(UInt64, uint64, EXECUTION_SPACE)                                                                                                                                                                                             \
    DEF(Char, wchar_t, EXECUTION_SPACE)

#define TEMPLATE_RANK7(DEF, EXECUTION_SPACE)                                                                                                                                                                                         \
    DEF(Single, float, EXECUTION_SPACE)                                                                                                                                                                                              \
    DEF(Double, double, EXECUTION_SPACE)                                                                                                                                                                                             \
    DEF(Bool, bool, EXECUTION_SPACE)                                                                                                                                                                                                 \
    DEF(Int8, int8, EXECUTION_SPACE)                                                                                                                                                                                                 \
    DEF(UInt8, uint8, EXECUTION_SPACE)                                                                                                                                                                                               \
    DEF(Int16, int16, EXECUTION_SPACE)                                                                                                                                                                                               \
    DEF(UInt16, uint16, EXECUTION_SPACE)                                                                                                                                                                                             \
    DEF(Int32, int32, EXECUTION_SPACE)                                                                                                                                                                                               \
    DEF(UInt32, uint32, EXECUTION_SPACE)                                                                                                                                                                                             \
    DEF(Int64, int64, EXECUTION_SPACE)                                                                                                                                                                                               \
    DEF(UInt64, uint64, EXECUTION_SPACE)                                                                                                                                                                                             \
    DEF(Char, wchar_t, EXECUTION_SPACE)

#define TEMPLATE_RANK8(DEF, EXECUTION_SPACE)                                                                                                                                                                                         \
    DEF(Single, float, EXECUTION_SPACE)                                                                                                                                                                                              \
    DEF(Double, double, EXECUTION_SPACE)                                                                                                                                                                                             \
    DEF(Bool, bool, EXECUTION_SPACE)                                                                                                                                                                                                 \
    DEF(Int8, int8, EXECUTION_SPACE)                                                                                                                                                                                                 \
    DEF(UInt8, uint8, EXECUTION_SPACE)                                                                                                                                                                                               \
    DEF(Int16, int16, EXECUTION_SPACE)                                                                                                                                                                                               \
    DEF(UInt16, uint16, EXECUTION_SPACE)                                                                                                                                                                                             \
    DEF(Int32, int32, EXECUTION_SPACE)                                                                                                                                                                                               \
    DEF(UInt32, uint32, EXECUTION_SPACE)                                                                                                                                                                                             \
    DEF(Int64, int64, EXECUTION_SPACE)                                                                                                                                                                                               \
    DEF(UInt64, uint64, EXECUTION_SPACE)                                                                                                                                                                                             \
    DEF(Char, wchar_t, EXECUTION_SPACE)

void* IpcMakeViewFromPointer(CONST(ExecutionSpaceKind) execution_space,
                             CONST(DataTypeKind) data_type,
                             void* instance,
                             CONST(size_type) arg_N0,
                             CONST(size_type) arg_N1,
                             CONST(size_type) arg_N2,
                             CONST(size_type) arg_N3,
                             CONST(size_type) arg_N4,
                             CONST(size_type) arg_N5,
                             CONST(size_type) arg_N6,
                             CONST(size_type) arg_N7) noexcept
{
    const size_type rank = Kokkos::GetRank(arg_N0, arg_N1, arg_N2, arg_N3, arg_N4, arg_N5, arg_N6, arg_N7);

    switch (execution_space)
    {
        case ExecutionSpaceKind::Serial:
        {
            switch (rank)
            {
                case 0:
                {
                    switch (data_type)
                    {
                        TEMPLATE_RANK0(DEF_TEMPLATE, Serial)
                        default:
                        {
                            std::cout << "MakeViewFromPointer::Serial, DataType is not supported." << std::endl;
                        }
                    }
                    break;
                }
                case 1:
                {
                    switch (data_type)
                    {
                        TEMPLATE_RANK1(DEF_TEMPLATE, Serial)
                        default:
                        {
                            std::cout << "MakeViewFromPointer::Serial, DataType is not supported." << std::endl;
                        }
                    }
                    break;
                }
                case 2:
                {
                    switch (data_type)
                    {
                        TEMPLATE_RANK2(DEF_TEMPLATE, Serial)
                        default:
                        {
                            std::cout << "MakeViewFromPointer::Serial, DataType is not supported." << std::endl;
                        }
                    }
                    break;
                }
                case 3:
                {
                    switch (data_type)
                    {
                        TEMPLATE_RANK3(DEF_TEMPLATE, Serial)
                        default:
                        {
                            std::cout << "MakeViewFromPointer::Serial, DataType is not supported." << std::endl;
                        }
                    }
                    break;
                }
                case 4:
                {
                    switch (data_type)
                    {
                        TEMPLATE_RANK4(DEF_TEMPLATE, Serial)
                        default:
                        {
                            std::cout << "MakeViewFromPointer::Serial, DataType is not supported." << std::endl;
                        }
                    }
                    break;
                }
                case 5:
                {
                    switch (data_type)
                    {
                        TEMPLATE_RANK5(DEF_TEMPLATE, Serial)
                        default:
                        {
                            std::cout << "MakeViewFromPointer::Serial, DataType is not supported." << std::endl;
                        }
                    }
                    break;
                }
                case 6:
                {
                    switch (data_type)
                    {
                        TEMPLATE_RANK6(DEF_TEMPLATE, Serial)
                        default:
                        {
                            std::cout << "MakeViewFromPointer::Serial, DataType is not supported." << std::endl;
                        }
                    }
                    break;
                }
                case 7:
                {
                    switch (data_type)
                    {
                        TEMPLATE_RANK7(DEF_TEMPLATE, Serial)
                        default:
                        {
                            std::cout << "MakeViewFromPointer::Serial, DataType is not supported." << std::endl;
                        }
                    }
                    break;
                }
                case 8:
                {
                    switch (data_type)
                    {
                        TEMPLATE_RANK8(DEF_TEMPLATE, Serial)
                        default:
                        {
                            std::cout << "MakeViewFromPointer::Serial, DataType is not supported." << std::endl;
                        }
                    }
                    break;
                }
                default:
                {
                    std::cout << "MakeViewFromPointer::Serial, Rank is not supported." << std::endl;
                }
            }
        }
        case ExecutionSpaceKind::OpenMP:
        {
            switch (rank)
            {
                case 0:
                {
                    switch (data_type)
                    {
                        TEMPLATE_RANK0(DEF_TEMPLATE, OpenMP)
                        default:
                        {
                            std::cout << "MakeViewFromPointer::OpenMP, DataType is not supported." << std::endl;
                        }
                    }
                    break;
                }
                case 1:
                {
                    switch (data_type)
                    {
                        TEMPLATE_RANK1(DEF_TEMPLATE, OpenMP)
                        default:
                        {
                            std::cout << "MakeViewFromPointer::OpenMP, DataType is not supported." << std::endl;
                        }
                    }
                    break;
                }
                case 2:
                {
                    switch (data_type)
                    {
                        TEMPLATE_RANK2(DEF_TEMPLATE, OpenMP)
                        default:
                        {
                            std::cout << "MakeViewFromPointer::OpenMP, DataType is not supported." << std::endl;
                        }
                    }
                    break;
                }
                case 3:
                {
                    switch (data_type)
                    {
                        TEMPLATE_RANK3(DEF_TEMPLATE, OpenMP)
                        default:
                        {
                            std::cout << "MakeViewFromPointer::OpenMP, DataType is not supported." << std::endl;
                        }
                    }
                    break;
                }
                case 4:
                {
                    switch (data_type)
                    {
                        TEMPLATE_RANK4(DEF_TEMPLATE, OpenMP)
                        default:
                        {
                            std::cout << "MakeViewFromPointer::OpenMP, DataType is not supported." << std::endl;
                        }
                    }
                    break;
                }
                case 5:
                {
                    switch (data_type)
                    {
                        TEMPLATE_RANK5(DEF_TEMPLATE, OpenMP)
                        default:
                        {
                            std::cout << "MakeViewFromPointer::OpenMP, DataType is not supported." << std::endl;
                        }
                    }
                    break;
                }
                case 6:
                {
                    switch (data_type)
                    {
                        TEMPLATE_RANK6(DEF_TEMPLATE, OpenMP)
                        default:
                        {
                            std::cout << "MakeViewFromPointer::OpenMP, DataType is not supported." << std::endl;
                        }
                    }
                    break;
                }
                case 7:
                {
                    switch (data_type)
                    {
                        TEMPLATE_RANK7(DEF_TEMPLATE, OpenMP)
                        default:
                        {
                            std::cout << "MakeViewFromPointer::OpenMP, DataType is not supported." << std::endl;
                        }
                    }
                    break;
                }
                case 8:
                {
                    switch (data_type)
                    {
                        TEMPLATE_RANK8(DEF_TEMPLATE, OpenMP)
                        default:
                        {
                            std::cout << "MakeViewFromPointer::OpenMP, DataType is not supported." << std::endl;
                        }
                    }
                    break;
                }
                default:
                {
                    std::cout << "MakeViewFromPointer::OpenMP, Rank is not supported." << std::endl;
                }
            }
        }
        case ExecutionSpaceKind::Cuda:
        {
            switch (rank)
            {
                case 0:
                {
                    switch (data_type)
                    {
                        TEMPLATE_RANK0(DEF_TEMPLATE, Cuda)
                        default:
                        {
                            std::cout << "MakeViewFromPointer::Cuda, DataType is not supported." << std::endl;
                        }
                    }
                    break;
                }
                case 1:
                {
                    switch (data_type)
                    {
                        TEMPLATE_RANK1(DEF_TEMPLATE, Cuda)
                        default:
                        {
                            std::cout << "MakeViewFromPointer::Cuda, DataType is not supported." << std::endl;
                        }
                    }
                    break;
                }
                case 2:
                {
                    switch (data_type)
                    {
                        TEMPLATE_RANK2(DEF_TEMPLATE, Cuda)
                        default:
                        {
                            std::cout << "MakeViewFromPointer::Cuda, DataType is not supported." << std::endl;
                        }
                    }
                    break;
                }
                case 3:
                {
                    switch (data_type)
                    {
                        TEMPLATE_RANK3(DEF_TEMPLATE, Cuda)
                        default:
                        {
                            std::cout << "MakeViewFromPointer::Cuda, DataType is not supported." << std::endl;
                        }
                    }
                    break;
                }
                case 4:
                {
                    switch (data_type)
                    {
                        TEMPLATE_RANK4(DEF_TEMPLATE, Cuda)
                        default:
                        {
                            std::cout << "MakeViewFromPointer::Cuda, DataType is not supported." << std::endl;
                        }
                    }
                    break;
                }
                case 5:
                {
                    switch (data_type)
                    {
                        TEMPLATE_RANK5(DEF_TEMPLATE, Cuda)
                        default:
                        {
                            std::cout << "MakeViewFromPointer::Cuda, DataType is not supported." << std::endl;
                        }
                    }
                    break;
                }
                case 6:
                {
                    switch (data_type)
                    {
                        TEMPLATE_RANK6(DEF_TEMPLATE, Cuda)
                        default:
                        {
                            std::cout << "MakeViewFromPointer::Cuda, DataType is not supported." << std::endl;
                        }
                    }
                    break;
                }
                case 7:
                {
                    switch (data_type)
                    {
                        TEMPLATE_RANK7(DEF_TEMPLATE, Cuda)
                        default:
                        {
                            std::cout << "MakeViewFromPointer::Cuda, DataType is not supported." << std::endl;
                        }
                    }
                    break;
                }
                case 8:
                {
                    switch (data_type)
                    {
                        TEMPLATE_RANK8(DEF_TEMPLATE, Cuda)
                        default:
                        {
                            std::cout << "MakeViewFromPointer::Cuda, DataType is not supported." << std::endl;
                        }
                    }
                    break;
                }
                default:
                {
                    std::cout << "MakeViewFromPointer::Cuda, Rank is not supported." << std::endl;
                }
            }
        }
        default:
        {
            std::cout << "MakeViewFromPointer ExecutionSpace is not supported." << std::endl;
        }
    }

    return nullptr;
}

#undef DEF_TEMPLATE

#define DEF_TEMPLATE(TYPE_NAME, TYPE, EXECUTION_SPACE)                                                                                                                                                                               \
    case DataTypeKind::TYPE_NAME:                                                                                                                                                                                                    \
    {                                                                                                                                                                                                                                \
        typedef Kokkos::InterprocessMemory<Kokkos::EXECUTION_SPACE> interprocess_memory_type;                                                                                                                                        \
        void* ptr = (*reinterpret_cast<Teuchos::RCP<interprocess_memory_type>*>(instance))->MakeViewFromHandle<TYPE>(arg_N0, arg_N1, arg_N2, arg_N3, arg_N4, arg_N5, arg_N6, arg_N7);                                                \
        return ptr;                                                                                                                                                                                                                  \
    }

void* IpcMakeViewFromHandle(CONST(ExecutionSpaceKind) execution_space,
                            CONST(DataTypeKind) data_type,
                            void* instance,
                            CONST(size_type) arg_N0,
                            CONST(size_type) arg_N1,
                            CONST(size_type) arg_N2,
                            CONST(size_type) arg_N3,
                            CONST(size_type) arg_N4,
                            CONST(size_type) arg_N5,
                            CONST(size_type) arg_N6,
                            CONST(size_type) arg_N7) noexcept
{
    const size_type rank = Kokkos::GetRank(arg_N0, arg_N1, arg_N2, arg_N3, arg_N4, arg_N5, arg_N6, arg_N7);

    switch (execution_space)
    {
        case ExecutionSpaceKind::Serial:
        {
            switch (rank)
            {
                case 0:
                {
                    switch (data_type)
                    {
                        TEMPLATE_RANK0(DEF_TEMPLATE, Serial)
                        default:
                        {
                            std::cout << "MakeViewFromPointer::Serial, DataType is not supported." << std::endl;
                        }
                    }
                    break;
                }
                case 1:
                {
                    switch (data_type)
                    {
                        TEMPLATE_RANK1(DEF_TEMPLATE, Serial)
                        default:
                        {
                            std::cout << "MakeViewFromPointer::Serial, DataType is not supported." << std::endl;
                        }
                    }
                    break;
                }
                case 2:
                {
                    switch (data_type)
                    {
                        TEMPLATE_RANK2(DEF_TEMPLATE, Serial)
                        default:
                        {
                            std::cout << "MakeViewFromPointer::Serial, DataType is not supported." << std::endl;
                        }
                    }
                    break;
                }
                case 3:
                {
                    switch (data_type)
                    {
                        TEMPLATE_RANK3(DEF_TEMPLATE, Serial)
                        default:
                        {
                            std::cout << "MakeViewFromPointer::Serial, DataType is not supported." << std::endl;
                        }
                    }
                    break;
                }
                case 4:
                {
                    switch (data_type)
                    {
                        TEMPLATE_RANK4(DEF_TEMPLATE, Serial)
                        default:
                        {
                            std::cout << "MakeViewFromPointer::Serial, DataType is not supported." << std::endl;
                        }
                    }
                    break;
                }
                case 5:
                {
                    switch (data_type)
                    {
                        TEMPLATE_RANK5(DEF_TEMPLATE, Serial)
                        default:
                        {
                            std::cout << "MakeViewFromPointer::Serial, DataType is not supported." << std::endl;
                        }
                    }
                    break;
                }
                case 6:
                {
                    switch (data_type)
                    {
                        TEMPLATE_RANK6(DEF_TEMPLATE, Serial)
                        default:
                        {
                            std::cout << "MakeViewFromPointer::Serial, DataType is not supported." << std::endl;
                        }
                    }
                    break;
                }
                case 7:
                {
                    switch (data_type)
                    {
                        TEMPLATE_RANK7(DEF_TEMPLATE, Serial)
                        default:
                        {
                            std::cout << "MakeViewFromPointer::Serial, DataType is not supported." << std::endl;
                        }
                    }
                    break;
                }
                case 8:
                {
                    switch (data_type)
                    {
                        TEMPLATE_RANK8(DEF_TEMPLATE, Serial)
                        default:
                        {
                            std::cout << "MakeViewFromPointer::Serial, DataType is not supported." << std::endl;
                        }
                    }
                    break;
                }
                default:
                {
                    std::cout << "MakeViewFromPointer::Serial, Rank is not supported." << std::endl;
                }
            }
        }
        case ExecutionSpaceKind::OpenMP:
        {
            switch (rank)
            {
                case 0:
                {
                    switch (data_type)
                    {
                        TEMPLATE_RANK0(DEF_TEMPLATE, OpenMP)
                        default:
                        {
                            std::cout << "MakeViewFromPointer::OpenMP, DataType is not supported." << std::endl;
                        }
                    }
                    break;
                }
                case 1:
                {
                    switch (data_type)
                    {
                        TEMPLATE_RANK1(DEF_TEMPLATE, OpenMP)
                        default:
                        {
                            std::cout << "MakeViewFromPointer::OpenMP, DataType is not supported." << std::endl;
                        }
                    }
                    break;
                }
                case 2:
                {
                    switch (data_type)
                    {
                        TEMPLATE_RANK2(DEF_TEMPLATE, OpenMP)
                        default:
                        {
                            std::cout << "MakeViewFromPointer::OpenMP, DataType is not supported." << std::endl;
                        }
                    }
                    break;
                }
                case 3:
                {
                    switch (data_type)
                    {
                        TEMPLATE_RANK3(DEF_TEMPLATE, OpenMP)
                        default:
                        {
                            std::cout << "MakeViewFromPointer::OpenMP, DataType is not supported." << std::endl;
                        }
                    }
                    break;
                }
                case 4:
                {
                    switch (data_type)
                    {
                        TEMPLATE_RANK4(DEF_TEMPLATE, OpenMP)
                        default:
                        {
                            std::cout << "MakeViewFromPointer::OpenMP, DataType is not supported." << std::endl;
                        }
                    }
                    break;
                }
                case 5:
                {
                    switch (data_type)
                    {
                        TEMPLATE_RANK5(DEF_TEMPLATE, OpenMP)
                        default:
                        {
                            std::cout << "MakeViewFromPointer::OpenMP, DataType is not supported." << std::endl;
                        }
                    }
                    break;
                }
                case 6:
                {
                    switch (data_type)
                    {
                        TEMPLATE_RANK6(DEF_TEMPLATE, OpenMP)
                        default:
                        {
                            std::cout << "MakeViewFromPointer::OpenMP, DataType is not supported." << std::endl;
                        }
                    }
                    break;
                }
                case 7:
                {
                    switch (data_type)
                    {
                        TEMPLATE_RANK7(DEF_TEMPLATE, OpenMP)
                        default:
                        {
                            std::cout << "MakeViewFromPointer::OpenMP, DataType is not supported." << std::endl;
                        }
                    }
                    break;
                }
                case 8:
                {
                    switch (data_type)
                    {
                        TEMPLATE_RANK8(DEF_TEMPLATE, OpenMP)
                        default:
                        {
                            std::cout << "MakeViewFromPointer::OpenMP, DataType is not supported." << std::endl;
                        }
                    }
                    break;
                }
                default:
                {
                    std::cout << "MakeViewFromPointer::OpenMP, Rank is not supported." << std::endl;
                }
            }
        }
        case ExecutionSpaceKind::Cuda:
        {
            switch (rank)
            {
                case 0:
                {
                    switch (data_type)
                    {
                        TEMPLATE_RANK0(DEF_TEMPLATE, Cuda)
                        default:
                        {
                            std::cout << "MakeViewFromPointer::Cuda, DataType is not supported." << std::endl;
                        }
                    }
                    break;
                }
                case 1:
                {
                    switch (data_type)
                    {
                        TEMPLATE_RANK1(DEF_TEMPLATE, Cuda)
                        default:
                        {
                            std::cout << "MakeViewFromPointer::Cuda, DataType is not supported." << std::endl;
                        }
                    }
                    break;
                }
                case 2:
                {
                    switch (data_type)
                    {
                        TEMPLATE_RANK2(DEF_TEMPLATE, Cuda)
                        default:
                        {
                            std::cout << "MakeViewFromPointer::Cuda, DataType is not supported." << std::endl;
                        }
                    }
                    break;
                }
                case 3:
                {
                    switch (data_type)
                    {
                        TEMPLATE_RANK3(DEF_TEMPLATE, Cuda)
                        default:
                        {
                            std::cout << "MakeViewFromPointer::Cuda, DataType is not supported." << std::endl;
                        }
                    }
                    break;
                }
                case 4:
                {
                    switch (data_type)
                    {
                        TEMPLATE_RANK4(DEF_TEMPLATE, Cuda)
                        default:
                        {
                            std::cout << "MakeViewFromPointer::Cuda, DataType is not supported." << std::endl;
                        }
                    }
                    break;
                }
                case 5:
                {
                    switch (data_type)
                    {
                        TEMPLATE_RANK5(DEF_TEMPLATE, Cuda)
                        default:
                        {
                            std::cout << "MakeViewFromPointer::Cuda, DataType is not supported." << std::endl;
                        }
                    }
                    break;
                }
                case 6:
                {
                    switch (data_type)
                    {
                        TEMPLATE_RANK6(DEF_TEMPLATE, Cuda)
                        default:
                        {
                            std::cout << "MakeViewFromPointer::Cuda, DataType is not supported." << std::endl;
                        }
                    }
                    break;
                }
                case 7:
                {
                    switch (data_type)
                    {
                        TEMPLATE_RANK7(DEF_TEMPLATE, Cuda)
                        default:
                        {
                            std::cout << "MakeViewFromPointer::Cuda, DataType is not supported." << std::endl;
                        }
                    }
                    break;
                }
                case 8:
                {
                    switch (data_type)
                    {
                        TEMPLATE_RANK8(DEF_TEMPLATE, Cuda)
                        default:
                        {
                            std::cout << "MakeViewFromPointer::Cuda, DataType is not supported." << std::endl;
                        }
                    }
                    break;
                }
                default:
                {
                    std::cout << "MakeViewFromPointer::Cuda, Rank is not supported." << std::endl;
                }
            }
        }
        default:
        {
            std::cout << "MakeViewFromPointer ExecutionSpace is not supported." << std::endl;
        }
    }

    return nullptr;
}

#undef DEF_TEMPLATE
#undef TEMPLATE_RANK0
#undef TEMPLATE_RANK1
#undef TEMPLATE_RANK2
#undef TEMPLATE_RANK3
#undef TEMPLATE_RANK4
#undef TEMPLATE_RANK5
#undef TEMPLATE_RANK6
#undef TEMPLATE_RANK7
#undef TEMPLATE_RANK8
