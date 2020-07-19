#pragma once

#include "runtime.Kokkos/Extensions.hpp"
//#include "NumericalMethods/Statistics/UniformRandom.hpp"

#include <Kokkos_Sort.hpp>

namespace NearestNeighbor
{
    template<size_type Dimensions>
    struct Tag
    {
    };

    template<typename DataType, class ExecutionSpace, size_type Dimensions>
    struct DistanceFunctor
    {
        static_assert(Dimensions >= 1 && Dimensions <= 10, "Dimensions must be between 1 && 10");

        using Dataset = Kokkos::View<DataType* [Dimensions], typename ExecutionSpace::array_layout, ExecutionSpace>;

        using Matrix = Kokkos::View<DataType**, typename ExecutionSpace::array_layout, ExecutionSpace>;

        Dataset _dataset;
        Matrix  _distances;

        DistanceFunctor(const Dataset& dataset) : _dataset(dataset), _distances("distances", dataset.extent(0), dataset.extent(0)) {}

        KOKKOS_INLINE_FUNCTION void operator()(const Tag<1>&, const size_type i, const size_type j) const
        {
            _distances(i, j) = 0.0;

            if(i == j)
            {
                return;
            }

            _distances(i, j) += pow(_dataset(i, 0) - _dataset(j, 0), 2);
        }

        KOKKOS_INLINE_FUNCTION void operator()(const Tag<2>&, const size_type i, const size_type j) const
        {
            _distances(i, j) = 0.0;

            if(i == j)
            {
                return;
            }

            _distances(i, j) += pow(_dataset(i, 0) - _dataset(j, 0), 2);
            _distances(i, j) += pow(_dataset(i, 1) - _dataset(j, 1), 2);

            _distances(i, j) = sqrt(_distances(i, j));
        }

        KOKKOS_INLINE_FUNCTION void operator()(const Tag<3>&, const size_type i, const size_type j) const
        {
            _distances(i, j) = 0.0;

            if(i == j)
            {
                return;
            }

            _distances(i, j) += pow(_dataset(i, 0) - _dataset(j, 0), 2);
            _distances(i, j) += pow(_dataset(i, 1) - _dataset(j, 1), 2);
            _distances(i, j) += pow(_dataset(i, 2) - _dataset(j, 2), 2);

            _distances(i, j) = sqrt(_distances(i, j));
        }

        KOKKOS_INLINE_FUNCTION void operator()(const Tag<4>&, const size_type i, const size_type j) const
        {
            _distances(i, j) = 0.0;

            if(i == j)
            {
                return;
            }

            _distances(i, j) += pow(_dataset(i, 0) - _dataset(j, 0), 2);
            _distances(i, j) += pow(_dataset(i, 1) - _dataset(j, 1), 2);
            _distances(i, j) += pow(_dataset(i, 2) - _dataset(j, 2), 2);
            _distances(i, j) += pow(_dataset(i, 3) - _dataset(j, 3), 2);

            _distances(i, j) = sqrt(_distances(i, j));
        }

        KOKKOS_INLINE_FUNCTION void operator()(const Tag<5>&, const size_type i, const size_type j) const
        {
            _distances(i, j) = 0.0;

            if(i == j)
            {
                return;
            }

            _distances(i, j) += pow(_dataset(i, 0) - _dataset(j, 0), 2);
            _distances(i, j) += pow(_dataset(i, 1) - _dataset(j, 1), 2);
            _distances(i, j) += pow(_dataset(i, 2) - _dataset(j, 2), 2);
            _distances(i, j) += pow(_dataset(i, 3) - _dataset(j, 3), 2);
            _distances(i, j) += pow(_dataset(i, 4) - _dataset(j, 4), 2);

            _distances(i, j) = sqrt(_distances(i, j));
        }

        KOKKOS_INLINE_FUNCTION void operator()(const Tag<6>&, const size_type i, const size_type j) const
        {
            _distances(i, j) = 0.0;

            if(i == j)
            {
                return;
            }

            _distances(i, j) += pow(_dataset(i, 0) - _dataset(j, 0), 2);
            _distances(i, j) += pow(_dataset(i, 1) - _dataset(j, 1), 2);
            _distances(i, j) += pow(_dataset(i, 2) - _dataset(j, 2), 2);
            _distances(i, j) += pow(_dataset(i, 3) - _dataset(j, 3), 2);
            _distances(i, j) += pow(_dataset(i, 4) - _dataset(j, 4), 2);
            _distances(i, j) += pow(_dataset(i, 5) - _dataset(j, 5), 2);

            _distances(i, j) = sqrt(_distances(i, j));
        }

        KOKKOS_INLINE_FUNCTION void operator()(const Tag<7>&, const size_type i, const size_type j) const
        {
            _distances(i, j) = 0.0;

            if(i == j)
            {
                return;
            }

            _distances(i, j) += pow(_dataset(i, 0) - _dataset(j, 0), 2);
            _distances(i, j) += pow(_dataset(i, 1) - _dataset(j, 1), 2);
            _distances(i, j) += pow(_dataset(i, 2) - _dataset(j, 2), 2);
            _distances(i, j) += pow(_dataset(i, 3) - _dataset(j, 3), 2);
            _distances(i, j) += pow(_dataset(i, 4) - _dataset(j, 4), 2);
            _distances(i, j) += pow(_dataset(i, 5) - _dataset(j, 5), 2);
            _distances(i, j) += pow(_dataset(i, 6) - _dataset(j, 6), 2);

            _distances(i, j) = sqrt(_distances(i, j));
        }

        KOKKOS_INLINE_FUNCTION void operator()(const Tag<8>&, const size_type i, const size_type j) const
        {
            _distances(i, j) = 0.0;

            if(i == j)
            {
                return;
            }

            _distances(i, j) += pow(_dataset(i, 0) - _dataset(j, 0), 2);
            _distances(i, j) += pow(_dataset(i, 1) - _dataset(j, 1), 2);
            _distances(i, j) += pow(_dataset(i, 2) - _dataset(j, 2), 2);
            _distances(i, j) += pow(_dataset(i, 3) - _dataset(j, 3), 2);
            _distances(i, j) += pow(_dataset(i, 4) - _dataset(j, 4), 2);
            _distances(i, j) += pow(_dataset(i, 5) - _dataset(j, 5), 2);
            _distances(i, j) += pow(_dataset(i, 6) - _dataset(j, 6), 2);
            _distances(i, j) += pow(_dataset(i, 7) - _dataset(j, 7), 2);

            _distances(i, j) = sqrt(_distances(i, j));
        }

        KOKKOS_INLINE_FUNCTION void operator()(const Tag<9>&, const size_type i, const size_type j) const
        {
            _distances(i, j) = 0.0;

            if(i == j)
            {
                return;
            }

            _distances(i, j) += pow(_dataset(i, 0) - _dataset(j, 0), 2);
            _distances(i, j) += pow(_dataset(i, 1) - _dataset(j, 1), 2);
            _distances(i, j) += pow(_dataset(i, 2) - _dataset(j, 2), 2);
            _distances(i, j) += pow(_dataset(i, 3) - _dataset(j, 3), 2);
            _distances(i, j) += pow(_dataset(i, 4) - _dataset(j, 4), 2);
            _distances(i, j) += pow(_dataset(i, 5) - _dataset(j, 5), 2);
            _distances(i, j) += pow(_dataset(i, 6) - _dataset(j, 6), 2);
            _distances(i, j) += pow(_dataset(i, 7) - _dataset(j, 7), 2);
            _distances(i, j) += pow(_dataset(i, 8) - _dataset(j, 8), 2);

            _distances(i, j) = sqrt(_distances(i, j));
        }

        KOKKOS_INLINE_FUNCTION void operator()(const Tag<10>&, const size_type i, const size_type j) const
        {
            _distances(i, j) = 0.0;

            if(i == j)
            {
                return;
            }

            _distances(i, j) += pow(_dataset(i, 0) - _dataset(j, 0), 2);
            _distances(i, j) += pow(_dataset(i, 1) - _dataset(j, 1), 2);
            _distances(i, j) += pow(_dataset(i, 2) - _dataset(j, 2), 2);
            _distances(i, j) += pow(_dataset(i, 3) - _dataset(j, 3), 2);
            _distances(i, j) += pow(_dataset(i, 4) - _dataset(j, 4), 2);
            _distances(i, j) += pow(_dataset(i, 5) - _dataset(j, 5), 2);
            _distances(i, j) += pow(_dataset(i, 6) - _dataset(j, 6), 2);
            _distances(i, j) += pow(_dataset(i, 7) - _dataset(j, 7), 2);
            _distances(i, j) += pow(_dataset(i, 8) - _dataset(j, 8), 2);
            _distances(i, j) += pow(_dataset(i, 9) - _dataset(j, 9), 2);

            _distances(i, j) = sqrt(_distances(i, j));
        }
    };

}

template<typename DataType, class ExecutionSpace, size_type Dimensions>
__inline static Kokkos::View<DataType**, typename ExecutionSpace::array_layout, ExecutionSpace> kNearestNeighbor(
    const int                                                                                         k,
    const Kokkos::View<DataType* [Dimensions], typename ExecutionSpace::array_layout, ExecutionSpace> dataset)
{
    using mdrange_type = Kokkos::MDRangePolicy<ExecutionSpace, Kokkos::Rank<2>, Kokkos::IndexType<size_type>, NearestNeighbor::Tag<Dimensions>>;

    using point_type = typename mdrange_type::point_type;

    mdrange_type policy(point_type {{0, 0}}, point_type {{dataset.extent(0), dataset.extent(0)}});

    using DistanceFunctor = NearestNeighbor::DistanceFunctor<DataType, ExecutionSpace, Dimensions>;
    DistanceFunctor f(dataset);

    Kokkos::parallel_for("Distance", policy, f);

    Kokkos::View<DataType**, typename ExecutionSpace::array_layout, ExecutionSpace> distances("distances", dataset.extent(0), dataset.extent(0));
    Kokkos::deep_copy(distances, f._distances);

    for(size_type i = 0; i < dataset.extent(0); i++)
    {
        auto _row = Kokkos::Extension::row(distances, i);

        Kokkos::sort(_row);
    }

    // Kokkos::View<int*, typename ExecutionSpace::array_layout, ExecutionSpace> classification("classification", dataset.extent(0));

    // const Kokkos::Random_XorShift1024_Pool<ExecutionSpace> pool(Kokkos::Impl::clock_tic());
    // Kokkos::fill_random(classification, pool, k);

    // for (int i = 0; i < k; i++)
    //{
    //    if (arr[i].val == 0)
    //        freq1++;
    //    else if (arr[i].val == 1)
    //        freq2++;
    //}

    return distances;
}
