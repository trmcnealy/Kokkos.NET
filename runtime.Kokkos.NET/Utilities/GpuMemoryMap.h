#pragma once

#include "KokkosAPI.hpp"

#include <cuda.h>

#include <iostream>

__inline static void checkDrvError(const CUresult res, const char* tok, const char* file, const unsigned line)
{
    if(res != CUDA_SUCCESS)
    {
        const char* errStr = nullptr;
        (void)cuGetErrorString(res, &errStr);
        std::cerr << file << ':' << line << ' ' << tok << "failed (" << (unsigned)res << "): " << errStr << std::endl;
        abort();
    }
}

#define CHECK_DRV(x) checkDrvError(x, #x, __FILE__, __LINE__);

class MMAPAllocation
{
public:
    size_type                    sz;
    CUmemGenericAllocationHandle hdl;
    CUmemAccessDesc              accessDesc;
    CUdeviceptr                  ptr;

    MMAPAllocation(const size_type size, const int dev = 0)
    {
        size_type           aligned_sz;
        CUmemAllocationProp prop = {};
        prop.type                = CU_MEM_ALLOCATION_TYPE_PINNED;
        prop.location.type       = CU_MEM_LOCATION_TYPE_DEVICE;
        prop.location.id         = dev;
        accessDesc.location      = prop.location;
        accessDesc.flags         = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;

        CHECK_DRV(cuMemGetAllocationGranularity(&aligned_sz, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));
        sz = ((size + aligned_sz - 1) / aligned_sz) * aligned_sz;

        CHECK_DRV(cuMemAddressReserve(&ptr, sz, 0ULL, 0ULL, 0ULL));
        CHECK_DRV(cuMemCreate(&hdl, sz, &prop, 0));
        CHECK_DRV(cuMemMap(ptr, sz, 0ULL, hdl, 0ULL));
        CHECK_DRV(cuMemSetAccess(ptr, sz, &accessDesc, 1ULL));
    }
    ~MMAPAllocation()
    {
        CHECK_DRV(cuMemUnmap(ptr, sz));
        CHECK_DRV(cuMemAddressFree(ptr, sz));
        CHECK_DRV(cuMemRelease(hdl));
    }
};
