#pragma once

#include "runtime.Kokkos/ViewTypes.hpp"

/// <summary>
/// https://github.com/kokkos/kokkos/wiki/View
/// </summary>
void ParallelViews()
{
    Kokkos::View<int*, Kokkos::Device<Kokkos::Cuda, Kokkos::CudaUVMSpace>>   a("a", 1000);
    Kokkos::View<int*, Kokkos::Device<Kokkos::OpenMP, Kokkos::CudaUVMSpace>> b("b", 1000);

    typedef Kokkos::RangePolicy<Kokkos::OpenMP, int> range_t;

    range_t range(range_t::member_type(0), range_t::member_type(1000));

    Kokkos::parallel_for("my kernel label", range, [=] __host__ __device__(const range_t::member_type i) {
        for (int j = 0; j < numInner; ++j)
        {
            outer[i][j] = 10.0 * double(i) + double(j);
        }
    });

    //    	                Serial	OpenMP	OpenMP	Cuda	ROCm
    // HostSpace	            x	    x	    x	    -	    -
    // HBWSpace	            x	    x	    x	    -	    -
    // CudaSpace	            -	    -	    -	    x	    -
    // CudaUVMSpace	        x	    x	    x	    x	    -
    // CudaHostPinnedSpace	x	    x	    x	    x	    -
    // ROCmSpace	            -	    -	    -	    -	    x
    // ROCmHostPinnedSpace	x	    x	    x	    -	    x
}

void ViewOfViews()
{
    using Kokkos::Cuda;
    using Kokkos::CudaSpace;
    using Kokkos::CudaUVMSpace;
    using Kokkos::View;
    using Kokkos::view_alloc;
    using Kokkos::WithoutInitializing;

    using inner_view_type = View<double*, CudaSpace>;
    using outer_view_type = View<inner_view_type*, CudaUVMSpace>;

    const int       numOuter = 5;
    const int       numInner = 4;
    outer_view_type outer(view_alloc(std::string("Outer"), WithoutInitializing), numOuter);

    // Create inner Views on host, outside of a parallel region, uninitialized
    for (int k = 0; k < numOuter; ++k)
    {
        const std::string label = std::string("Inner ") + std::to_string(k);
        new (&outer[k]) inner_view_type(view_alloc(label, WithoutInitializing), numInner);
    }

    // Outer and inner views are now ready for use on device

    Kokkos::RangePolicy<Cuda, int> range(0, numOuter);

    Kokkos::parallel_for("my kernel label", range, [=] __host__ __device__(const int i) {
        for (int j = 0; j < numInner; ++j)
        {
            outer[i][j] = 10.0 * double(i) + double(j);
        }
    });

    // Fence before deallocation on host, to make sure
    // that the device kernel is done first.
    // Note the new fence syntax that requires an instance.
    // This will work with other CUDA streams, etc.
    Cuda().fence();

    // Destroy inner Views, again on host, outside of a parallel region.
    for (int k = 0; k < 5; ++k)
    {
        outer[k].~inner_view_type();
    }

    // You're better off disposing of outer immediately.
    outer = outer_view_type();
}
