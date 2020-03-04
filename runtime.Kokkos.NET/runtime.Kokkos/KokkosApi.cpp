
#include "runtime.Kokkos/KokkosApi.h"

// clang-format off
static constexpr KokkosApi kokkos_api_version_1 = 
{
    &Allocate,
    &Reallocate,
    &Free,
    &Initialize,
    &InitializeThreads,
    &InitializeArguments,
    &Finalize,
    &FinalizeAll,
    &IsInitialized,
    &PrintConfiguration,
    &CudaGetDeviceCount,
    &CudaGetComputeCapability,
    &CreateViewRank0,
    &CreateViewRank1,
    &CreateViewRank2,
    &CreateViewRank3,
    &CreateView,
    &GetLabel,
    &GetSize,
    &GetStride,
    &GetExtent,
    &CopyTo,
    &GetValue,
    &SetValue,
    &ViewToNdArray
};
// clang-format on

KOKKOS_NET_API_EXTERNC const KokkosApi* GetApi(const uint32& version)
{
    if(version == 1)
    {
        return &kokkos_api_version_1;
    }
    return nullptr;
}

NativeString::NativeString(const std::string& str) :
    Length(isNullTerminating(str) ? str.size() : str.size() + 1), Bytes((int8*)Kokkos::kokkos_malloc<Kokkos::Serial::memory_space>(Length))
{
    // memcpy(const_cast<int8*>(Bytes), str.c_str(), isNullTerminating(str) ? str.size() - 1 : str.size());

    int index = 0;

    while(str[index] != '\0')
    {
        Bytes[index] = str[index];
        ++index;
    }
}

NativeString::NativeString(const int& length, char* const bytes) :
    Length(isNullTerminating(length, bytes) ? length : length + 1), Bytes((int8*)Kokkos::kokkos_malloc<Kokkos::Serial::memory_space>(Length))
{
    // memcpy(const_cast<int8*>(Bytes), bytes, isNullTerminating(length, bytes) ? length - 1 : length);

    int index = 0;

    while(bytes[index] != '\0')
    {
        Bytes[index] = bytes[index];
        ++index;
    }
}

std::string NativeString::ToString() const { return std::string(Bytes, isNullTerminating(Length, Bytes) ? Length - 1 : Length); }

// KokkosDotNET::KokkosView_TemplateManager* templateManager = new KokkosDotNET::KokkosView_TemplateManager();
