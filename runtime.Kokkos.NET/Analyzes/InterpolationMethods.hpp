#pragma once

#include "runtime.Kokkos/ViewTypes.hpp"
#include "runtime.Kokkos/Extensions.hpp"

#include <MathExtensions.hpp>

#include <KokkosBlas.hpp>

#include <StdExtensions.hpp>

#include <Print.hpp>

//#include <Algebra/Eigenvalue.hpp>

KOKKOS_NET_API_EXTERNC void* Shepard2dSingle(void* xd_rcp_view_ptr, void* zd_rcp_view_ptr, const float& p, void* xi_rcp_view_ptr, const ExecutionSpaceKind& execution_space) noexcept;

KOKKOS_NET_API_EXTERNC void* Shepard2dDouble(void* xd_rcp_view_ptr, void* zd_rcp_view_ptr, const double& p, void* xi_rcp_view_ptr, const ExecutionSpaceKind& execution_space) noexcept;

namespace Interpolation
{
    template<typename DataType, class ExecutionSpace, size_type M>
    using PointVector = Kokkos::View<DataType* [M], typename ExecutionSpace::array_layout, ExecutionSpace>;

    template<typename DataType, class ExecutionSpace, size_type M>
    struct ShepardFunctor
    {
        PointVector<DataType, ExecutionSpace, M>            xd;
        Kokkos::Extension::Vector<DataType, ExecutionSpace> zd;
        PointVector<DataType, ExecutionSpace, M>            xi;
        Kokkos::Extension::Vector<DataType, ExecutionSpace> zi;
        Kokkos::Extension::Matrix<DataType, ExecutionSpace> w;
        const DataType                                      p;
        const size_type                                     nd;
        const size_type                                     ni;

        ShepardFunctor(PointVector<DataType, ExecutionSpace, M>&            xd,
                       Kokkos::Extension::Vector<DataType, ExecutionSpace>& zd,
                       const DataType&                                      p,
                       PointVector<DataType, ExecutionSpace, M>&            xi) :
            xd(xd), zd(zd), xi(xi), p(p), nd(xd.extent(0)), ni(xi.extent(0))
        {
            zi = Kokkos::Extension::Vector<DataType, ExecutionSpace>("zi", ni);
            w  = Kokkos::Extension::Matrix<DataType, ExecutionSpace>("w", nd, ni);
        }

        KOKKOS_INLINE_FUNCTION void operator()(const size_type& i) const
        {
            DataType s;
            DataType t;
            int64    z = -1;

            for(size_type j = 0; j < nd; ++j)
            {
                t = 0.0;

#pragma clang loop unroll(enable)
                for(size_type i2 = 0; i2 < M; ++i2)
                {
                    t += pow(xi(i, i2) - xd(j, i2), 2.0);
                }

                w(j, i) = sqrt(t);

                if(w(j, i) == 0.0)
                {
                    z = j;
                    break;
                }
            }

            if(z != -1)
            {
                for(size_type j = 0; j < nd; ++j)
                {
                    w(j, i) = 0.0;
                }

                w(z, i) = 1.0;
            }
            else
            {
                for(size_type j = 0; j < nd; ++j)
                {
                    w(j, i) = 1.0 / pow(w(j, i), p);
                }

                s = 0.0;

                for(size_type j = 0; j < nd; ++j)
                {
                    s += w(j, i);
                }

                for(size_type j = 0; j < nd; ++j)
                {
                    w(j, i) /= s;
                }
            }

            if(i == 0)
            {
                for(size_type j = 0; j < nd; ++j)
                {
                    System::out << w(j, i) << System::endl;
                }
            }

            zi(i) = Kokkos::Extension::inner_product<DataType, ExecutionSpace>(Kokkos::subview(w, Kokkos::ALL, i), zd);
        }

        // if(p == 0.0)
        //{
        //    for(size_type j = 0; j < nd; ++j)
        //    {
        //        w[j] = 1.0 / (DataType)nd;
        //    }
        //}
    };

    /// <summary>
    /// SHEPARD_INTERP_ND evaluates a multidimensional Shepard interpolant.
    /// </summary>
    /// <param name="xd">the data point[M*ND], M: the spatial dimension, ND: the number of data points</param>
    /// <param name="zd">data values[ND]</param>
    /// <param name="p">power</param>
    /// <param name="xi">interpolation points[M*NI], NI: the number of interpolation points</param>
    /// <returns>interpolated values[NI]</returns>
    template<typename DataType, class ExecutionSpace, size_type M>
    static Kokkos::Extension::Vector<DataType, ExecutionSpace> ShepardNd(PointVector<DataType, ExecutionSpace, M>&            xd,
                                                                         Kokkos::Extension::Vector<DataType, ExecutionSpace>& zd,
                                                                         const DataType&                                      p,
                                                                         PointVector<DataType, ExecutionSpace, M>&            xi)
    {
        const size_type nd = xd.extent(0);
        const size_type ni = xi.extent(0);

        ShepardFunctor<DataType, ExecutionSpace, M> functor(xd, zd, p, xi);

        Kokkos::parallel_for(Kokkos::RangePolicy<ExecutionSpace, Kokkos::IndexType<size_type>>(0, ni), functor);

        return functor.zi;
    }

}
