
#include "Tests.hpp"

template<class ExecutionSpace>
static void TestNestedHybridLoop()
{
    // const size_t nAgents    = 5;
    // const size_t nParticles = 100;

    // Vector<double, ExecutionSpace> x("x", nAgents * nParticles);

    ////#pragma omp parallel num_threads(nAgents)
    ////        for (size_t i = 0; i < nAgents; ++i)
    ////        {
    ////            cudaStream_t stream;
    ////            cudaStreamCreate(&stream);
    ////            Kokkos::Cuda space0(stream);
    ////
    ////            Kokkos::parallel_for(Kokkos::RangePolicy<ExecutionSpace>(space0, 0, nParticles), [=] __host__ __device__(const size_t i0) {
    ////                x(i + nAgents * i0) = 1.0 * i0;
    ////            });
    ////
    ////            space0.fence();
    ////        }

    // Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::OpenMP>(0, nAgents), [=](const size_t i) {
    //    cudaStream_t stream;
    //    cudaStreamCreate(&stream);
    //    Kokkos::Cuda space0(stream);

    //    Kokkos::parallel_for(Kokkos::RangePolicy<ExecutionSpace>(space0, 0, nParticles), [=] __host__ __device__(const size_t i0) {
    //        x(i + nAgents * i0) = 1.0 * i0;
    //    });

    //    space0.fence();
    //});

    // for (size_t i = 0; i < nAgents; ++i)
    //{
    //    for (size_t i0 = 0; i0 < nParticles; i0++)
    //    {
    //        std::cout << x(i + nAgents * i0) << " " << std::endl;
    //    }
    //}
}
