
#include "InterprocessMemory.h"

#include <windef.h>
#include <winbase.h>

#include <cuda.h>

void* IpcCreate(REF(ExecutionSpaceKind) execution_space, REF(size_type) size, REF(NativeString) label) noexcept
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
            void* memoryPtr = Kokkos::kokkos_malloc<Kokkos::Cuda::memory_space>(size);

            CUipcMemHandle handle;

            CUresult error = cuIpcGetMemHandle(&handle, reinterpret_cast<CUdeviceptr>(memoryPtr));

            Kokkos::InterprocessMemory<Kokkos::Cuda>* ipc = new Kokkos::InterprocessMemory<Kokkos::Cuda>(memoryPtr, &handle, size, label.Bytes);

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

void* IpcCreateFrom(REF(ExecutionSpaceKind) execution_space, void* memoryPtr, REF(size_type) size, REF(NativeString) label) noexcept
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
            CUipcMemHandle handle;

            CUresult error = cuIpcGetMemHandle(&handle, reinterpret_cast<CUdeviceptr>(memoryPtr));

            Kokkos::InterprocessMemory<Kokkos::Cuda>* ipc = new Kokkos::InterprocessMemory<Kokkos::Cuda>(memoryPtr, &handle, size, label.Bytes);

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

void* IpcOpenExisting(REF(ExecutionSpaceKind) execution_space, void* instance) noexcept
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

            CUdeviceptr           memoryPtr;
            const CUipcMemHandle* handle = reinterpret_cast<const CUipcMemHandle*>(other_ipc->GetDeviceHandle());

            CUresult error = cuIpcOpenMemHandle(&memoryPtr, *handle, CU_IPC_MEM_LAZY_ENABLE_PEER_ACCESS);

            Kokkos::InterprocessMemory<Kokkos::Cuda>* ipc = new Kokkos::InterprocessMemory<Kokkos::Cuda>(reinterpret_cast<void*>(memoryPtr),
                                                                                                         &handle,
                                                                                                         other_ipc->GetSize(),
                                                                                                         other_ipc->GetLabel());

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

void IpcDestory(REF(ExecutionSpaceKind) execution_space, void* instance) noexcept
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

            if (ipc->GetDeviceHandle())
            {
                cuIpcCloseMemHandle(reinterpret_cast<CUdeviceptr>(ipc->GetDeviceHandle()));
            }

            if (ipc->GetMemoryPointer())
            {
                Kokkos::kokkos_free<Kokkos::Cuda::memory_space>(ipc->GetMemoryPointer());
            }

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

void IpcClose(REF(ExecutionSpaceKind) execution_space, void* instance) noexcept
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
        }
        case ExecutionSpaceKind::Cuda:
        {
            Teuchos::RCP<Kokkos::InterprocessMemory<Kokkos::Cuda>> ipc = *reinterpret_cast<Teuchos::RCP<Kokkos::InterprocessMemory<Kokkos::Cuda>>*>(instance);

            CUresult error = cuIpcCloseMemHandle(reinterpret_cast<CUdeviceptr>(ipc->GetDeviceHandle()));
        }
        case ExecutionSpaceKind::Unknown:
        {
            break;
        }
    }

    std::cout << "The ExecutionSpace is not supported." << std::endl;
}

void* IpcGetMemoryPointer(REF(ExecutionSpaceKind) execution_space, void* instance) noexcept
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

void* IpcGetDeviceHandle(REF(ExecutionSpaceKind) execution_space, void* instance) noexcept
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

size_type IpcGetSize(REF(ExecutionSpaceKind) execution_space, void* instance) noexcept
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

#define DEF_TEMPLATE(TYPE_NAME, TYPE, EXECUTION_SPACE)                                                                                                                        \
    case DataTypeKind::TYPE_NAME:                                                                                                                                             \
    {                                                                                                                                                                         \
        typedef Kokkos::InterprocessMemory<Kokkos::EXECUTION_SPACE> interprocess_memory_type;                                                                                 \
        void*                                                       ptr = (*reinterpret_cast<Teuchos::RCP<interprocess_memory_type>*>(instance))                              \
                        ->MakeViewFromPointer<TYPE>(arg_N0, arg_N1, arg_N2, arg_N3, arg_N4, arg_N5, arg_N6, arg_N7);                                                          \
        return ptr;                                                                                                                                                           \
    }

#define TEMPLATE_RANK0(DEF, EXECUTION_SPACE)                                                                                                                                  \
    DEF(Single, float, EXECUTION_SPACE)                                                                                                                                       \
    DEF(Double, double, EXECUTION_SPACE)                                                                                                                                      \
    DEF(Bool, bool, EXECUTION_SPACE)                                                                                                                                          \
    DEF(Int8, int8, EXECUTION_SPACE)                                                                                                                                          \
    DEF(UInt8, uint8, EXECUTION_SPACE)                                                                                                                                        \
    DEF(Int16, int16, EXECUTION_SPACE)                                                                                                                                        \
    DEF(UInt16, uint16, EXECUTION_SPACE)                                                                                                                                      \
    DEF(Int32, int32, EXECUTION_SPACE)                                                                                                                                        \
    DEF(UInt32, uint32, EXECUTION_SPACE)                                                                                                                                      \
    DEF(Int64, int64, EXECUTION_SPACE)                                                                                                                                        \
    DEF(UInt64, uint64, EXECUTION_SPACE)                                                                                                                                      \
    DEF(Char, wchar_t, EXECUTION_SPACE)

#define TEMPLATE_RANK1(DEF, EXECUTION_SPACE)                                                                                                                                  \
    DEF(Single, float, EXECUTION_SPACE)                                                                                                                                       \
    DEF(Double, double, EXECUTION_SPACE)                                                                                                                                      \
    DEF(Bool, bool, EXECUTION_SPACE)                                                                                                                                          \
    DEF(Int8, int8, EXECUTION_SPACE)                                                                                                                                          \
    DEF(UInt8, uint8, EXECUTION_SPACE)                                                                                                                                        \
    DEF(Int16, int16, EXECUTION_SPACE)                                                                                                                                        \
    DEF(UInt16, uint16, EXECUTION_SPACE)                                                                                                                                      \
    DEF(Int32, int32, EXECUTION_SPACE)                                                                                                                                        \
    DEF(UInt32, uint32, EXECUTION_SPACE)                                                                                                                                      \
    DEF(Int64, int64, EXECUTION_SPACE)                                                                                                                                        \
    DEF(UInt64, uint64, EXECUTION_SPACE)                                                                                                                                      \
    DEF(Char, wchar_t, EXECUTION_SPACE)

#define TEMPLATE_RANK2(DEF, EXECUTION_SPACE)                                                                                                                                  \
    DEF(Single, float, EXECUTION_SPACE)                                                                                                                                       \
    DEF(Double, double, EXECUTION_SPACE)                                                                                                                                      \
    DEF(Bool, bool, EXECUTION_SPACE)                                                                                                                                          \
    DEF(Int8, int8, EXECUTION_SPACE)                                                                                                                                          \
    DEF(UInt8, uint8, EXECUTION_SPACE)                                                                                                                                        \
    DEF(Int16, int16, EXECUTION_SPACE)                                                                                                                                        \
    DEF(UInt16, uint16, EXECUTION_SPACE)                                                                                                                                      \
    DEF(Int32, int32, EXECUTION_SPACE)                                                                                                                                        \
    DEF(UInt32, uint32, EXECUTION_SPACE)                                                                                                                                      \
    DEF(Int64, int64, EXECUTION_SPACE)                                                                                                                                        \
    DEF(UInt64, uint64, EXECUTION_SPACE)                                                                                                                                      \
    DEF(Char, wchar_t, EXECUTION_SPACE)

#define TEMPLATE_RANK3(DEF, EXECUTION_SPACE)                                                                                                                                  \
    DEF(Single, float, EXECUTION_SPACE)                                                                                                                                       \
    DEF(Double, double, EXECUTION_SPACE)                                                                                                                                      \
    DEF(Bool, bool, EXECUTION_SPACE)                                                                                                                                          \
    DEF(Int8, int8, EXECUTION_SPACE)                                                                                                                                          \
    DEF(UInt8, uint8, EXECUTION_SPACE)                                                                                                                                        \
    DEF(Int16, int16, EXECUTION_SPACE)                                                                                                                                        \
    DEF(UInt16, uint16, EXECUTION_SPACE)                                                                                                                                      \
    DEF(Int32, int32, EXECUTION_SPACE)                                                                                                                                        \
    DEF(UInt32, uint32, EXECUTION_SPACE)                                                                                                                                      \
    DEF(Int64, int64, EXECUTION_SPACE)                                                                                                                                        \
    DEF(UInt64, uint64, EXECUTION_SPACE)                                                                                                                                      \
    DEF(Char, wchar_t, EXECUTION_SPACE)

#define TEMPLATE_RANK4(DEF, EXECUTION_SPACE)                                                                                                                                  \
    DEF(Single, float, EXECUTION_SPACE)                                                                                                                                       \
    DEF(Double, double, EXECUTION_SPACE)                                                                                                                                      \
    DEF(Bool, bool, EXECUTION_SPACE)                                                                                                                                          \
    DEF(Int8, int8, EXECUTION_SPACE)                                                                                                                                          \
    DEF(UInt8, uint8, EXECUTION_SPACE)                                                                                                                                        \
    DEF(Int16, int16, EXECUTION_SPACE)                                                                                                                                        \
    DEF(UInt16, uint16, EXECUTION_SPACE)                                                                                                                                      \
    DEF(Int32, int32, EXECUTION_SPACE)                                                                                                                                        \
    DEF(UInt32, uint32, EXECUTION_SPACE)                                                                                                                                      \
    DEF(Int64, int64, EXECUTION_SPACE)                                                                                                                                        \
    DEF(UInt64, uint64, EXECUTION_SPACE)                                                                                                                                      \
    DEF(Char, wchar_t, EXECUTION_SPACE)

#define TEMPLATE_RANK5(DEF, EXECUTION_SPACE)                                                                                                                                  \
    DEF(Single, float, EXECUTION_SPACE)                                                                                                                                       \
    DEF(Double, double, EXECUTION_SPACE)                                                                                                                                      \
    DEF(Bool, bool, EXECUTION_SPACE)                                                                                                                                          \
    DEF(Int8, int8, EXECUTION_SPACE)                                                                                                                                          \
    DEF(UInt8, uint8, EXECUTION_SPACE)                                                                                                                                        \
    DEF(Int16, int16, EXECUTION_SPACE)                                                                                                                                        \
    DEF(UInt16, uint16, EXECUTION_SPACE)                                                                                                                                      \
    DEF(Int32, int32, EXECUTION_SPACE)                                                                                                                                        \
    DEF(UInt32, uint32, EXECUTION_SPACE)                                                                                                                                      \
    DEF(Int64, int64, EXECUTION_SPACE)                                                                                                                                        \
    DEF(UInt64, uint64, EXECUTION_SPACE)                                                                                                                                      \
    DEF(Char, wchar_t, EXECUTION_SPACE)

#define TEMPLATE_RANK6(DEF, EXECUTION_SPACE)                                                                                                                                  \
    DEF(Single, float, EXECUTION_SPACE)                                                                                                                                       \
    DEF(Double, double, EXECUTION_SPACE)                                                                                                                                      \
    DEF(Bool, bool, EXECUTION_SPACE)                                                                                                                                          \
    DEF(Int8, int8, EXECUTION_SPACE)                                                                                                                                          \
    DEF(UInt8, uint8, EXECUTION_SPACE)                                                                                                                                        \
    DEF(Int16, int16, EXECUTION_SPACE)                                                                                                                                        \
    DEF(UInt16, uint16, EXECUTION_SPACE)                                                                                                                                      \
    DEF(Int32, int32, EXECUTION_SPACE)                                                                                                                                        \
    DEF(UInt32, uint32, EXECUTION_SPACE)                                                                                                                                      \
    DEF(Int64, int64, EXECUTION_SPACE)                                                                                                                                        \
    DEF(UInt64, uint64, EXECUTION_SPACE)                                                                                                                                      \
    DEF(Char, wchar_t, EXECUTION_SPACE)

#define TEMPLATE_RANK7(DEF, EXECUTION_SPACE)                                                                                                                                  \
    DEF(Single, float, EXECUTION_SPACE)                                                                                                                                       \
    DEF(Double, double, EXECUTION_SPACE)                                                                                                                                      \
    DEF(Bool, bool, EXECUTION_SPACE)                                                                                                                                          \
    DEF(Int8, int8, EXECUTION_SPACE)                                                                                                                                          \
    DEF(UInt8, uint8, EXECUTION_SPACE)                                                                                                                                        \
    DEF(Int16, int16, EXECUTION_SPACE)                                                                                                                                        \
    DEF(UInt16, uint16, EXECUTION_SPACE)                                                                                                                                      \
    DEF(Int32, int32, EXECUTION_SPACE)                                                                                                                                        \
    DEF(UInt32, uint32, EXECUTION_SPACE)                                                                                                                                      \
    DEF(Int64, int64, EXECUTION_SPACE)                                                                                                                                        \
    DEF(UInt64, uint64, EXECUTION_SPACE)                                                                                                                                      \
    DEF(Char, wchar_t, EXECUTION_SPACE)

#define TEMPLATE_RANK8(DEF, EXECUTION_SPACE)                                                                                                                                  \
    DEF(Single, float, EXECUTION_SPACE)                                                                                                                                       \
    DEF(Double, double, EXECUTION_SPACE)                                                                                                                                      \
    DEF(Bool, bool, EXECUTION_SPACE)                                                                                                                                          \
    DEF(Int8, int8, EXECUTION_SPACE)                                                                                                                                          \
    DEF(UInt8, uint8, EXECUTION_SPACE)                                                                                                                                        \
    DEF(Int16, int16, EXECUTION_SPACE)                                                                                                                                        \
    DEF(UInt16, uint16, EXECUTION_SPACE)                                                                                                                                      \
    DEF(Int32, int32, EXECUTION_SPACE)                                                                                                                                        \
    DEF(UInt32, uint32, EXECUTION_SPACE)                                                                                                                                      \
    DEF(Int64, int64, EXECUTION_SPACE)                                                                                                                                        \
    DEF(UInt64, uint64, EXECUTION_SPACE)                                                                                                                                      \
    DEF(Char, wchar_t, EXECUTION_SPACE)

void* IpcMakeViewFromPointer(REF(ExecutionSpaceKind) execution_space,
                             REF(DataTypeKind) data_type,
                             void* instance,
                             REF(size_type) arg_N0,
                             REF(size_type) arg_N1,
                             REF(size_type) arg_N2,
                             REF(size_type) arg_N3,
                             REF(size_type) arg_N4,
                             REF(size_type) arg_N5,
                             REF(size_type) arg_N6,
                             REF(size_type) arg_N7) noexcept
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

#define DEF_TEMPLATE(TYPE_NAME, TYPE, EXECUTION_SPACE)                                                                                                                        \
    case DataTypeKind::TYPE_NAME:                                                                                                                                             \
    {                                                                                                                                                                         \
        typedef Kokkos::InterprocessMemory<Kokkos::EXECUTION_SPACE> interprocess_memory_type;                                                                                 \
        void*                                                       ptr = (*reinterpret_cast<Teuchos::RCP<interprocess_memory_type>*>(instance))                              \
                        ->MakeViewFromHandle<TYPE>(arg_N0, arg_N1, arg_N2, arg_N3, arg_N4, arg_N5, arg_N6, arg_N7);                                                           \
        return ptr;                                                                                                                                                           \
    }

void* IpcMakeViewFromHandle(REF(ExecutionSpaceKind) execution_space,
                            REF(DataTypeKind) data_type,
                            void* instance,
                            REF(size_type) arg_N0,
                            REF(size_type) arg_N1,
                            REF(size_type) arg_N2,
                            REF(size_type) arg_N3,
                            REF(size_type) arg_N4,
                            REF(size_type) arg_N5,
                            REF(size_type) arg_N6,
                            REF(size_type) arg_N7) noexcept
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
