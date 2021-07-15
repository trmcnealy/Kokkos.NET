
#include <runtime.Kokkos/KokkosApi.h>

// clang-format off
static constexpr KokkosApi kokkos_api_version_1 =
{
    &Allocate,
    &Reallocate,
    &Copy,
    &Free,
    &Initialize,
    &InitializeSerial,
    &InitializeOpenMP,
    &InitializeCuda,
    &InitializeThreads,
    &InitializeArguments,
    &Finalize,
    &FinalizeSerial,
    &FinalizeOpenMP,
    &FinalizeCuda,
    &FinalizeAll,
    &IsInitialized,
    &PrintConfiguration,
    &CudaGetDeviceCount,
    &CudaGetComputeCapability,
    &CreateViewRank0,
    &CreateViewRank1,
    &CreateViewRank2,
    &CreateViewRank3,
    &CreateViewRank4,
    &CreateViewRank5,
    &CreateViewRank6,
    &CreateViewRank7,
    &CreateViewRank8,
    &CreateView,
    &GetLabel,
    &GetSize,
    &GetStride,
    &GetExtent,
    &CopyTo,
    &GetValue,
    &SetValue,
    &RcpViewToNdArray,
    &ViewToNdArray
};
// clang-format on

KOKKOS_NET_API_EXTERNC const KokkosApi* GetApi(const uint32 version)
{
    if (version == 1)
    {
        return &kokkos_api_version_1;
    }
    return nullptr;
}

// KokkosDotNET::KokkosView_TemplateManager* templateManager = new KokkosDotNET::KokkosView_TemplateManager();

