
#include "runtime.Kokkos/KokkosApi.h"

KOKKOS_NET_API_EXTERNC uint32 GetNumaCount() noexcept
{
    return Kokkos::hwloc::get_available_numa_count();
}

KOKKOS_NET_API_EXTERNC uint32 GetCoresPerNuma() noexcept
{
    return Kokkos::hwloc::get_available_numa_count();
}

KOKKOS_NET_API_EXTERNC uint32 GetThreadsPerCore() noexcept
{
    return Kokkos::hwloc::get_available_numa_count();
}
