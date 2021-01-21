
#include "runtime.Kokkos/KokkosApi.h"

#include "GpuCsvReader.hpp"

KOKKOS_NET_API_EXTERNC void* CountLineEndingsSerial(void* instance)
{

    typedef Kokkos::View<wchar_t*, Kokkos::Serial::array_layout, Kokkos::Serial>         view_type;
    typedef Kokkos::View<uint64*, Kokkos::Serial::array_layout, Kokkos::Serial> view_index_type;

    view_type& view = *(*reinterpret_cast<Teuchos::RCP<view_type>*>(instance));

    view_index_type lineEndingsView = Kokkos::Extension::CountLineEndings(view);

    view_index_type* lineEndings_ptr = new view_index_type("CountLineEndings", lineEndingsView.extent(0));

    for (uint64 i0 = 0; i0 < lineEndingsView.extent(0); ++i0)
    {
        (*lineEndings_ptr)(i0) = lineEndingsView(i0);
    }

    return (void*)new Teuchos::RCP<view_index_type>(lineEndings_ptr);
}

KOKKOS_NET_API_EXTERNC void* CountLineEndingsOpenMP(void* instance)
{

    typedef Kokkos::View<wchar_t*, Kokkos::OpenMP::array_layout, Kokkos::OpenMP>         view_type;
    typedef Kokkos::View<uint64*, Kokkos::OpenMP::array_layout, Kokkos::OpenMP> view_index_type;

    view_type& view = *(*reinterpret_cast<Teuchos::RCP<view_type>*>(instance));

    view_index_type lineEndingsView = Kokkos::Extension::CountLineEndings(view);

    view_index_type* lineEndings_ptr = new view_index_type("CountLineEndings", lineEndingsView.extent(0));

    for (uint64 i0 = 0; i0 < lineEndingsView.extent(0); ++i0)
    {
        (*lineEndings_ptr)(i0) = lineEndingsView(i0);
    }

    return (void*)new Teuchos::RCP<view_index_type>(lineEndings_ptr);
}

KOKKOS_NET_API_EXTERNC void* CountLineEndingsCuda(void* instance)
{

    typedef Kokkos::View<wchar_t*, Kokkos::Cuda::array_layout, Kokkos::Cuda>         view_type;
    typedef Kokkos::View<uint64*, Kokkos::Cuda::array_layout, Kokkos::Cuda> view_index_type;

    view_type& view = *(*reinterpret_cast<Teuchos::RCP<view_type>*>(instance));

    view_index_type lineEndingsView = Kokkos::Extension::CountLineEndings(view);

    view_index_type* lineEndings_ptr = new view_index_type("CountLineEndings", lineEndingsView.extent(0));

    for (uint64 i0 = 0; i0 < lineEndingsView.extent(0); ++i0)
    {
        (*lineEndings_ptr)(i0) = lineEndingsView(i0);
    }

    return (void*)new Teuchos::RCP<view_index_type>(lineEndings_ptr);
}
