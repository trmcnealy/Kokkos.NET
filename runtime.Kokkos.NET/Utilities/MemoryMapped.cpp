// //////////////////////////////////////////////////////////
// MemoryMapped.cpp
// Copyright (c) 2013 Stephan Brumme. All rights reserved.
// see http://create.stephan-brumme.com/disclaimer.html
//

#include "MemoryMapped.h"


// // OS-specific
// #ifdef _WINDOWS
// // Windows
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#    include <windows.h>
// #else
// // Linux
// // enable large file support on 32 bit systems
// #    ifndef _LARGEFILE64_SOURCE
// #        define _LARGEFILE64_SOURCE
// #    endif
// #    ifdef _FILE_OFFSET_BITS
// #        undef _FILE_OFFSET_BITS
// #    endif
// #    define _FILE_OFFSET_BITS 64
// // and include needed headers
// #    include <sys/stat.h>
// #    include <sys/mman.h>
// #    include <fcntl.h>
// #    include <errno.h>
// #    include <unistd.h>
// #endif

#include <cmath>
#include <stdexcept>
#include <cstdio>

//#include <crt/host_defines.h>

/// do nothing, must use open()
MemoryMapped::MemoryMapped() :
    _filename(),
    _filesize(0),
    _hint(Normal),
    _mappedBytes(0),
    _file(nullptr),
#ifdef _WINDOWS
    _mappedFile(nullptr),
#endif
    _mappedView(nullptr)
{
}

/// open file, mappedBytes = 0 maps the whole file
MemoryMapped::MemoryMapped(const std::string& filename, const unsigned __int64 mappedBytes, const CacheHint hint) :
    _filename(filename),
    _filesize(0),
    _hint(hint),
    _mappedBytes(mappedBytes),
    _file(nullptr),
#ifdef _WINDOWS
    _mappedFile(nullptr),
#endif
    _mappedView(nullptr)
{
    Open(filename, mappedBytes, hint);
}

/// close file (see close() )
MemoryMapped::~MemoryMapped() { Close(); }

/// open file
bool MemoryMapped::Open(const std::string& filename, const unsigned __int64 mappedBytes, const CacheHint hint)
{
    // already open ?
    if(IsValid())
        return false;

    _file     = nullptr;
    _filesize = 0;
    _hint     = hint;
#ifdef _WINDOWS
    _mappedFile = nullptr;
#endif
    _mappedView = nullptr;

#ifdef _WINDOWS
    // Windows

    DWORD winHint = 0;
    switch(_hint)
    {
        case Normal: winHint = FILE_ATTRIBUTE_NORMAL; break;
        case SequentialScan: winHint = FILE_FLAG_SEQUENTIAL_SCAN; break;
        case RandomAccess: winHint = FILE_FLAG_RANDOM_ACCESS; break;
        default: break;
    }

    // open file
    _file = ::CreateFileA(filename.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, winHint, nullptr);

    if(!_file)
        return false;

    // file size
    LARGE_INTEGER result;

    if(!GetFileSizeEx(_file, &result))
        return false;

    _filesize = static_cast<uint64>(result.QuadPart);

    // convert to mapped mode
    _mappedFile = ::CreateFileMapping(_file, nullptr, PAGE_READONLY, 0, 0, nullptr);

    if(!_mappedFile)
        return false;

#else

    // Linux

    // open file
    _file = ::open(filename.c_str(), O_RDONLY | O_LARGEFILE);
    if(_file == -1)
    {
        _file = nullptr;
        return false;
    }

    // file size
    struct stat64 statInfo;
    if(fstat64(_file, &statInfo) < 0)
        return false;

    _filesize = statInfo.st_size;
#endif

    // initial mapping
    Remap(0, mappedBytes);

    if(!_mappedView)
        return false;

    // everything's fine
    return true;
}

/// close file
void MemoryMapped::Close()
{
    // kill pointer
    if(_mappedView)
    {
#ifdef _WINDOWS
        ::UnmapViewOfFile(_mappedView);
#else
        ::munmap(_mappedView, _filesize);
#endif
        _mappedView = nullptr;
    }

#ifdef _WINDOWS
    if(_mappedFile)
    {
        ::CloseHandle(_mappedFile);
        _mappedFile = nullptr;
    }
#endif

    // close underlying file
    if(_file)
    {
#ifdef _WINDOWS
        ::CloseHandle(_file);
#else
        ::close(_file);
#endif
        _file = nullptr;
    }

    _filesize = 0;
}

/// access position, no range checking (faster)
unsigned char MemoryMapped::operator[](const unsigned __int64 offset) const { return ((unsigned char*)_mappedView)[offset]; }

/// access position, including range checking
unsigned char MemoryMapped::At(const unsigned __int64 offset) const
{
    // checks
    if(!_mappedView)
        throw std::invalid_argument("No view mapped");

    if(offset >= _filesize)
        throw std::out_of_range("View is not large enough");

    return operator[](offset);
}

/// raw access
const unsigned char* MemoryMapped::GetData() const { return (const unsigned char*)_mappedView; }

/// true, if file successfully opened
bool MemoryMapped::IsValid() const { return _mappedView != nullptr; }

/// get file size
uint64 MemoryMapped::Size() const { return _filesize; }

/// get number of actually mapped bytes
unsigned __int64 MemoryMapped::MappedSize() const { return _mappedBytes; }

/// replace mapping by a new one of the same file, offset MUST be a multiple of the page size
bool MemoryMapped::Remap(const uint64 offset, unsigned __int64 mappedBytes)
{
    if(!_file)
        return false;

    if(mappedBytes == WholeFile)
        mappedBytes = _filesize;

    // close old mapping
    if(_mappedView)
    {
#ifdef _WINDOWS
        ::UnmapViewOfFile(_mappedView);
#else
        ::munmap(_mappedView, _mappedBytes);
#endif
        _mappedView = nullptr;
    }

    // don't go further than end of file
    if(offset > _filesize)
        return false;
    if(offset + mappedBytes > _filesize)
        mappedBytes = static_cast<unsigned __int64>(_filesize - offset);

#ifdef _WINDOWS
    // Windows

    const DWORD offsetLow  = DWORD(offset & 0xFFFFFFFF);
    const DWORD offsetHigh = DWORD(offset >> 32);
    _mappedBytes           = mappedBytes;

    // get memory address
    _mappedView = ::MapViewOfFile(_mappedFile, FILE_MAP_READ, offsetHigh, offsetLow, mappedBytes);

    if(_mappedView == nullptr)
    {
        _mappedBytes = 0;
        _mappedView  = nullptr;
        return false;
    }

    return true;

#else

    // Linux
    // new mapping
    _mappedView = ::mmap64(NULL, mappedBytes, PROT_READ, MAP_SHARED, _file, offset);
    if(_mappedView == MAP_FAILED)
    {
        _mappedBytes = 0;
        _mappedView  = nullptr;
        return false;
    }

    _mappedBytes = mappedBytes;

    // tweak performance
    int linuxHint = 0;
    switch(_hint)
    {
        case Normal: linuxHint = MADV_NORMAL; break;
        case SequentialScan: linuxHint = MADV_SEQUENTIAL; break;
        case RandomAccess: linuxHint = MADV_RANDOM; break;
        default: break;
    }
    // assume that file will be accessed soon
    // linuxHint |= MADV_WILLNEED;
    // assume that file will be large
    // linuxHint |= MADV_HUGEPAGE;

    ::madvise(_mappedView, _mappedBytes, linuxHint);

    return true;
#endif
}

/// get OS page size (for remap)
int MemoryMapped::getpagesize()
{
#ifdef _WINDOWS
    SYSTEM_INFO sysInfo;
    GetSystemInfo(&sysInfo);
    return sysInfo.dwAllocationGranularity;
#else
    return sysconf(_SC_PAGESIZE); //::getpagesize();
#endif
}

KOKKOS_NET_API_EXTERNC MemoryMapped* Create() noexcept { return new MemoryMapped(); }

KOKKOS_NET_API_EXTERNC MemoryMapped* CreateAndOpen(const char* filename, const unsigned __int64 mappedBytes, const MemoryMapped::CacheHint hint) noexcept
{
    return new MemoryMapped(std::string(filename), mappedBytes, hint);
}

KOKKOS_NET_API_EXTERNC void Destory(MemoryMapped* mm) noexcept { delete mm; }

KOKKOS_NET_API_EXTERNC bool Open(MemoryMapped* mm, const char* filename, const unsigned __int64 mappedBytes, const MemoryMapped::CacheHint hint) noexcept
{
    return mm->Open(std::string(filename), mappedBytes, hint);
}

KOKKOS_NET_API_EXTERNC void Close(MemoryMapped* mm) { mm->Close(); }

// KOKKOS_NET_API_EXTERN unsigned char operator[](unsigned __int64 offset) noexcept;

KOKKOS_NET_API_EXTERNC unsigned char At(MemoryMapped* mm, const unsigned __int64 offset) noexcept { return mm->At(offset); }

KOKKOS_NET_API_EXTERNC const unsigned char* GetData(MemoryMapped* mm) noexcept { return mm->GetData(); }

KOKKOS_NET_API_EXTERNC bool IsValid(MemoryMapped* mm) noexcept { return mm->IsValid(); }

KOKKOS_NET_API_EXTERNC uint64 Size(MemoryMapped* mm) noexcept { return mm->Size(); }

KOKKOS_NET_API_EXTERNC unsigned __int64 MappedSize(MemoryMapped* mm) noexcept { return mm->MappedSize(); }

KOKKOS_NET_API_EXTERNC bool Remap(MemoryMapped* mm, const uint64 offset, const unsigned __int64 mappedBytes) { return mm->Remap(offset, mappedBytes); }

//#include "KokkosAPI.hpp"
//
//#include "cuda_runtime_api.h"
//
//#define ROWNUM 11997996
//#define REDUCTION_BLOCK_SIZE 512
//#define SEGSIZE ROWNUM / 2
//
// struct entry
//{
//    char words[11];
//};
//
// void mmread_date(const std::string& filename, struct entry* date_h)
//{
//    const MemoryMapped data(filename);
//
//    if(!data.IsValid())
//    {
//        printf("Failed to read file.\n");
//        return;
//    }
//
//    char* buffer = (char*)data.GetData();
//
//    int index = 0;
//
//    for(uint64 i = 0; i < data.Size(); i += 11)
//    { // length of date(10) + '\n'(1) = 11
//        for(int j = 0; j < 10; ++j)
//            date_h[index].words[j] = buffer[i + j];
//
//        date_h[index].words[10] = '\0';
//        index++;
//    }
//}
//
///*
//   Read floats from file using MemoryMapping library.
//*/
// void mmread_float(const std::string& filename, float* dest)
//{
//    const MemoryMapped data(filename);
//
//    if(!data.IsValid())
//    {
//        printf("Failed to read file.\n");
//        return;
//    }
//
//    char* buffer = (char*)data.GetData();
//
//    int  index    = 0;
//    int  start    = 0;
//    char temp[15] = {};
//
//    for(uint64_t i = 0; i < data.Size(); ++i)
//    {
//        if(buffer[i] == '\n')
//        {
//            const int length = i - start;
//
//            for(int j = 0; j < length; ++j)
//                temp[j] = buffer[start + j];
//
//            temp[length] = '\0';
//            dest[index]  = atof(temp);
//            start        = i + 1;
//            index++;
//        }
//    }
//}
//
///*
//   Query_Scan --
//      This kernel scans disc_d, qty_d, price_d, and date_d
//      according to the mySQL Q6 query from the TPC-H specification, described below:
//
//   select sum(l_extendedprice*l_discount)
//   as revenue from lineitem
//   where(l_shipdate >= '1994-01-01' and
//         l_shipdate < '1995-01-01'  and
//         l_discount >= 0.05         and
//         l_discount <= 0.0750       and
//         l_quantity < 24;
//*/
//__global__ void query_scan(float* disc_d, float* qty_d, float* price_d, struct entry* date_d, float* revenue_d)
//{
//    int index = blockDim.x * blockIdx.x + threadIdx.x;
//    if(index < SEGSIZE)
//    {
//        int count = 0;
//        if(date_d[index].words[3] == '4')
//        {
//            count++;
//        }
//
//        if((disc_d[index] >= 0.0500) && (disc_d[index] <= 0.0750))
//        {
//            count++;
//        }
//
//        if(qty_d[index] < 24)
//        {
//            count++;
//        }
//
//        if(count == 3)
//        {
//            revenue_d[index] = price_d[index] * disc_d[index];
//        }
//        else
//            revenue_d[index] = 0;
//    }
//
//    __syncthreads();
//}
//
///*
//
//   Query_Reduction --
//      Performs a reduction sum on revenue_d, storing the respective sum
//         for the current block at block_sum_d[i]
//*/
//__global__ void query_reduction(float* revenue_d, float* block_sum_d)
//{
//    __shared__ float partialSum[2 * REDUCTION_BLOCK_SIZE];
//
//    int t     = threadIdx.x;
//    int start = 2 * blockDim.x * blockIdx.x;
//
//    if(start + t < SEGSIZE)
//    {
//        partialSum[t] = revenue_d[start + t];
//    }
//    else
//    {
//        partialSum[t] = 0.0;
//    }
//
//    if(start + t + REDUCTION_BLOCK_SIZE < SEGSIZE)
//    {
//        partialSum[t + blockDim.x] = revenue_d[start + t + blockDim.x];
//    }
//    else
//    {
//        partialSum[t + blockDim.x] = 0.0;
//    }
//
//    int stride;
//    for(stride = blockDim.x; stride >= 1; stride >>= 1)
//    {
//        __syncthreads();
//
//        if(t < stride)
//        {
//            partialSum[t] += partialSum[t + stride];
//        }
//    }
//
//    if(stride == 0)
//    {
//        block_sum_d[blockIdx.x] = partialSum[0];
//    }
//}
//
///*
//
//   main() --
//      Main entry point to the program.
//*/
// int main()
//{
//    // Declare pointers for TPC-H benchmark data to be imported.
//    // This is for discount, quantity, price, and date attributes
//    //    for the Q6 query.
//    float*        disc_h; // = (float*)malloc(ROWNUM*sizeof(float));
//    float*        qty_h; // = (float*)malloc(ROWNUM*sizeof(float));
//    float*        price_h; //= (float*)malloc(ROWNUM*sizeof(float));
//    struct entry* date_h; // =(struct entry*) malloc(ROWNUM*sizeof(struct entry));
//
//    // We're using pinned memory
//    cudaHostAlloc((void**)&disc_h, ROWNUM * sizeof(float), 0);
//    cudaHostAlloc((void**)&qty_h, ROWNUM * sizeof(float), 0);
//    cudaHostAlloc((void**)&price_h, ROWNUM * sizeof(float), 0);
//    cudaHostAlloc((void**)&date_h, ROWNUM * sizeof(struct entry), 0);
//
//    float total_revenue = 0.0;
//
//    // define variable for stream 0
//    float *       disc_d0, *qty_d0, *price_d0, *revenue_d0;
//    struct entry* date_d0;
//
//    // define variable for stream 1
//    float *       disc_d1, *qty_d1, *price_d1, *revenue_d1;
//    struct entry* date_d1;
//
//    dim3 dim_grid, dim_block;
//
//    // allocate device memory for stream 0
//    cudaMalloc((void**)&disc_d0, SEGSIZE * sizeof(float));
//    cudaMalloc((void**)&qty_d0, SEGSIZE * sizeof(float));
//    cudaMalloc((void**)&price_d0, SEGSIZE * sizeof(float));
//    cudaMalloc((void**)&date_d0, SEGSIZE * sizeof(struct entry));
//    cudaMalloc((void**)&revenue_d0, SEGSIZE * sizeof(float));
//
//    // allocate device memory for stream 1
//    cudaMalloc((void**)&disc_d1, SEGSIZE * sizeof(float));
//    cudaMalloc((void**)&qty_d1, SEGSIZE * sizeof(float));
//    cudaMalloc((void**)&price_d1, SEGSIZE * sizeof(float));
//    cudaMalloc((void**)&date_d1, SEGSIZE * sizeof(struct entry));
//    cudaMalloc((void**)&revenue_d1, SEGSIZE * sizeof(float));
//
//    // measure the total response time
//    float       responseTime;
//    cudaEvent_t start0, stop0;
//    cudaEventCreate(&start0);
//    cudaEventCreate(&stop0);
//    cudaEventRecord(start0, 0);
//
//    // data transfer from disk to host memory using memory mapping library
//    mmread_float("L_DISCOUNT.txt", disc_h);
//    mmread_float("L_QUANTITY.txt", qty_h);
//    mmread_float("L_EXTENDEDPRICE.txt", price_h);
//    mmread_date("L_SHIPDATE.txt", date_h);
//
//    // define multi-streams
//    cudaStream_t stream0, stream1;
//    cudaStreamCreate(&stream0);
//    cudaStreamCreate(&stream1);
//
//    // decide query_scan kernel function's dimension
//    dim_block.x = 1024;
//    dim_block.y = dim_block.z = 1;
//
//    dim_grid.x = SEGSIZE / 1024;
//    if(SEGSIZE % 1024 != 0)
//    {
//        dim_grid.x++;
//    }
//    dim_grid.y = dim_grid.z = 1;
//
//    // decide query_reduction kernel function's dimension
//    int    num_blocks = ceil((float)SEGSIZE / (REDUCTION_BLOCK_SIZE * 2));
//    float *block_sum_d0, *block_sum_d1;
//
//    // allocate memory for block_sum_h
//    float* block_sum_h = (float*)malloc(2 * num_blocks * sizeof(float));
//
//    // allocate meory for block_sum_d0 and block_sum_d1
//    cudaMalloc((void**)&block_sum_d0, num_blocks * sizeof(float));
//    cudaMalloc((void**)&block_sum_d1, num_blocks * sizeof(float));
//
//    // call kernel function
//    // Record GPU Time.
//    float       gpuTimeDMATime;
//    cudaEvent_t start2, stop2;
//    cudaEventCreate(&start2);
//    cudaEventCreate(&stop2);
//    cudaEventRecord(start2, 0);
//
//    // call kenerl query_scan function to scan the whole database table
//    // fetch the required data tuples
//
//    // copy data from host to device for stream 0
//    cudaMemcpyAsync(disc_d0, disc_h, SEGSIZE * sizeof(float), cudaMemcpyHostToDevice, stream0);
//    cudaMemcpyAsync(qty_d0, qty_h, SEGSIZE * sizeof(float), cudaMemcpyHostToDevice, stream0);
//    cudaMemcpyAsync(price_d0, price_h, SEGSIZE * sizeof(float), cudaMemcpyHostToDevice, stream0);
//    cudaMemcpyAsync(date_d0, date_h, SEGSIZE * sizeof(struct entry), cudaMemcpyHostToDevice, stream0);
//
//    // copy data from host to device for stream 1
//    cudaMemcpyAsync(disc_d1, disc_h + SEGSIZE, SEGSIZE * sizeof(float), cudaMemcpyHostToDevice, stream1);
//    cudaMemcpyAsync(qty_d1, qty_h + SEGSIZE, SEGSIZE * sizeof(float), cudaMemcpyHostToDevice, stream1);
//    cudaMemcpyAsync(price_d1, price_h + SEGSIZE, SEGSIZE * sizeof(float), cudaMemcpyHostToDevice, stream1);
//    cudaMemcpyAsync(date_d1, date_h + SEGSIZE, SEGSIZE * sizeof(struct entry), cudaMemcpyHostToDevice, stream1);
//
//    // query_scan<<<dim_grid, dim_block>>>(disc_d, qty_d, price_d, date_d, revenue_d);
//    query_scan<<<dim_grid, dim_block, 0, stream0>>>(disc_d0, qty_d0, price_d0, date_d0, revenue_d0);
//    query_scan<<<dim_grid, dim_block, 0, stream1>>>(disc_d1, qty_d1, price_d1, date_d1, revenue_d1);
//
//    // Synchronize between kernel calls.
//    cudaDeviceSynchronize();
//
//    query_reduction<<<num_blocks, REDUCTION_BLOCK_SIZE, 0, stream0>>>(revenue_d0, block_sum_d0);
//    query_reduction<<<num_blocks, REDUCTION_BLOCK_SIZE, 0, stream1>>>(revenue_d1, block_sum_d1);
//
//    // Mark GPU end time
//    cudaEventRecord(stop2, 0);
//    cudaEventSynchronize(stop2);
//    cudaEventElapsedTime(&gpuTimeDMATime, start2, stop2);
//
//    // Copy results back to host, calculate total revenue.
//    cudaMemcpyAsync(block_sum_h, block_sum_d0, num_blocks * sizeof(float), cudaMemcpyDeviceToHost, stream0);
//    cudaMemcpyAsync(block_sum_h + num_blocks, block_sum_d1, num_blocks * sizeof(float), cudaMemcpyDeviceToHost, stream1);
//
//    // measure execution time, GPU time, and CPU time
//    float       cpuTime;
//    cudaEvent_t start1, stop1;
//    cudaEventCreate(&start1);
//    cudaEventCreate(&stop1);
//    cudaEventRecord(start1, nullptr);
//
//    for(int i = 0; i < 2 * num_blocks; ++i)
//    {
//        total_revenue = total_revenue + block_sum_h[i];
//    }
//
//    // Mark execution end time
//    cudaEventRecord(stop1, 0);
//    cudaEventSynchronize(stop1);
//    cudaEventElapsedTime(&cpuTime, start1, stop1);
//
//    // Output query result
//    printf(std::endl);
//    printf("+----------+\n");
//    printf("| revenue  |\n");
//    printf("+----------+\n");
//    printf(" %f\n", total_revenue);
//    printf("+----------+\n");
//
//    // Free all allocated resources
//    cudaFreeHost(disc_h);
//    cudaFreeHost(qty_h);
//    cudaFreeHost(price_h);
//    cudaFreeHost(date_h);
//
//    free(block_sum_h);
//
//    cudaFree(disc_d0);
//    cudaFree(qty_d0);
//    cudaFree(price_d0);
//    cudaFree(date_d0);
//    cudaFree(revenue_d0);
//    cudaFree(block_sum_d0);
//
//    cudaFree(disc_d1);
//    cudaFree(qty_d1);
//    cudaFree(price_d1);
//    cudaFree(date_d1);
//    cudaFree(revenue_d1);
//    cudaFree(block_sum_d1);
//
//    // Stop timer for response time
//    cudaEventRecord(stop0, 0);
//    cudaEventSynchronize(stop0);
//    cudaEventElapsedTime(&responseTime, start0, stop0);
//
//    printf("the total response time is: %f ms\n", responseTime);
//    printf("the gpu + DMA time is: %f ms\n", gpuTimeDMATime);
//    printf("the cpu time is: %f ms\n", cpuTime);
//    // printf("the DMA time is: %f ms\n", DMATime);
//    printf("the IO time is: %f ms\n", responseTime - gpuTimeDMATime - cpuTime);
//
//    return 0;
//}
