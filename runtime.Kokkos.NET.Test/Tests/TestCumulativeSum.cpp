
#include <Types.hpp>

#include <runtime.Kokkos/ViewTypes.hpp>
#include <runtime.Kokkos/Extensions.hpp>

void TestCumulativeSum()
{
    const int  num_threads      = 64;
    const int  num_numa         = 1;
    const int  device_id        = 0;
    const int  ndevices         = 1;
    const int  skip_device      = 9999;
    const bool disable_warnings = false;

    Kokkos::InitArguments arguments{};
    arguments.num_threads      = num_threads;
    arguments.num_numa         = num_numa;
    arguments.device_id        = device_id;
    arguments.ndevices         = ndevices;
    arguments.skip_device      = skip_device;
    arguments.disable_warnings = disable_warnings;

    Kokkos::ScopeGuard kokkos(arguments);
    {
        Kokkos::Extension::Vector<double, Kokkos::Cuda> x("x", 5); // assume filled with input values

        x[0] = 1;
        x[1] = 2;
        x[2] = 3;
        x[3] = 4;
        x[4] = 5;

        Kokkos::Extension::Vector<double, Kokkos::Cuda> sum_x = Kokkos::CumulativeSum(x);

        // Kokkos::View<float*> sum_x("sum_x", 5);

        // Kokkos::deep_copy(sum_x, x);

        // const size_t N = x.extent(0);

        // Kokkos::parallel_scan(N,
        //                      [=] __host__ __device__(const int i, float& update, const bool final)
        //                      {
        //                          const float val_i = x(i);

        //                          update += val_i;

        //                          if (final)
        //                          {
        //                              sum_x(i) = update;
        //                          }

        //                      });

        std::cout << sum_x << std::endl;


    }
}
