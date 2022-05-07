
#include "SharedMemory.h"

#if defined(_WINDOWS)
#    include <memoryapi.h>
#    include <errhandlingapi.h>
#    include <handleapi.h>
#else
#    include <stdlib.h>
#    include <stdio.h>
#    include <fcntl.h>
#    include <sys/mman.h>
#    include <unistd.h>
#    include <errno.h>
#    include <sys/types.h>
#    include <sys/socket.h>
#    include <memory.h>
#    include <sys/un.h>
#endif

#if !defined(__wasm32__)
#    include <cuda_runtime.h>
#endif

static int wchar_len(const wchar_t* name)
{
    int len = 0;

    const wchar_t* nameptr = name;

    while (nameptr[len] != '\0')
    {
        ++len;
    }

    return len;
}

int SharedMemoryCreate(const wchar_t* name, size_type size, SharedMemoryData* data_out)
{
    int result = 0;

    SharedMemoryData data{};

    data.Size = size;

#if defined(_WINDOWS)
    data.Handle = CreateFileMappingW(INVALID_HANDLE_VALUE, nullptr, SEC_COMMIT | PAGE_READWRITE, 0, (DWORD)size, name);

    if (data.Handle == nullptr)
    {
        result = GetLastError();
        printf("SharedMemoryCreate::CreateFileMappingW error:%i\n", result);
    }

    data.HostAddress = MapViewOfFile(data.Handle, FILE_MAP_ALL_ACCESS, 0, 0, size);

    if (data.HostAddress == nullptr)
    {
        result = GetLastError();
        printf("SharedMemoryCreate::MapViewOfFile error:%i\n", result);
    }
#else
    const int name_len  = wchar_len(name);
    char*     char_name = new char[name_len];

    wcstombs(char_name, name, name_len);

    data.Handle = shm_open(char_name, O_RDWR | O_CREAT, 0777);

    if (data.Handle < 0)
    {
        result = errno;
    }

    result = ftruncate(data.Handle, size);

    data.HostAddress = mmap(0, size, PROT_READ | PROT_WRITE, MAP_SHARED, data.Handle, 0);

    if (data.HostAddress == NULL)
    {
        result = errno;
    }
#endif

    (*data_out) = data;

    return result;
}

int SharedMemoryOpen(const wchar_t* name, size_type size, SharedMemoryData* data_out)
{
    int result = 0;

    SharedMemoryData data{};

    data.Size = size;

#if defined(_WINDOWS)
    data.Handle = OpenFileMappingW(FILE_MAP_ALL_ACCESS, FALSE, name);

    if (data.Handle == nullptr)
    {
        result = GetLastError();
        printf("SharedMemoryOpen::OpenFileMappingW error:%i\n", result);
    }

    data.HostAddress = MapViewOfFile(data.Handle, FILE_MAP_ALL_ACCESS, 0, 0, size);

    if (data.HostAddress == nullptr)
    {
        result = GetLastError();
        printf("SharedMemoryOpen::MapViewOfFile error:%i\n", result);
    }
#else
    const int name_len  = wchar_len(name);
    char*     char_name = new char[name_len];

    wcstombs(char_name, name, name_len);

    data.Handle = shm_open(char_name, O_RDWR, 0777);

    if (data.Handle < 0)
    {
        result = errno;
    }

    data.HostAddress = mmap(0, size, PROT_READ | PROT_WRITE, MAP_SHARED, data.Handle, 0);

    if (data.HostAddress == NULL)
    {
        result = errno;
    }
#endif

    (*data_out) = data;

    return result;
}

int SharedMemoryResize(const wchar_t* name, size_type size, SharedMemoryData* data)
{
    int result = 0;

    const LPVOID tempAddress = VirtualAlloc(nullptr, data->Size, MEM_COMMIT | MEM_PHYSICAL, PAGE_READWRITE);

    MoveMemory(tempAddress, data->HostAddress, data->Size);

    bool isRegisteredWithCuda = false;
    if (data->DeviceAddress)
    {
        isRegisteredWithCuda = true;
    }

    SharedMemoryClose(data);

    result = SharedMemoryCreate(name, size, data);

    MoveMemory(data->HostAddress, tempAddress, data->Size);

    if (isRegisteredWithCuda)
    {
        result += SharedMemoryRegisterWithCuda(data);
    }

    return result;
}

void SharedMemoryClose(SharedMemoryData* data)
{
    if (data->HostAddress)
    {
#if defined(_WINDOWS)
        UnmapViewOfFile(data->HostAddress);
#else
        munmap(data->HostAddress, data->Size);
#endif
        data->HostAddress = nullptr;
    }

    if (data->Handle)
    {
#if defined(_WINDOWS)
        CloseHandle(data->Handle);
#else
        close(data->Handle);
#endif
        data->Handle = nullptr;
    }

#if !defined(__wasm32__)
    if (data->DeviceAddress)
    {
        cudaHostUnregister(data->DeviceAddress);
        data->DeviceAddress = nullptr;
    }
#endif
}

#if !defined(__wasm32__)

#    include <cuda_runtime.h>

int SharedMemoryRegisterWithCuda(SharedMemoryData* data)
{
    cudaError error = cudaHostRegister(data->HostAddress, data->Size, cudaHostRegisterMapped);

    if (error == cudaSuccess)
    {
        error = cudaHostGetDevicePointer(&data->DeviceAddress, data->HostAddress, 0);
    }

    return error;
}

#endif
