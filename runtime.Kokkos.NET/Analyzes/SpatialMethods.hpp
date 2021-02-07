#pragma once

#include "runtime.Kokkos/ViewTypes.hpp"
#include "runtime.Kokkos/Extensions.hpp"

#include <Constants.hpp>

#include <MathExtensions.hpp>

#include <KokkosBlas.hpp>

#include <StdExtensions.hpp>

#include <Print.hpp>

#include <intrin.h>

//#include <Algebra/Eigenvalue.hpp>

KOKKOS_NET_API_EXTERNC void* NearestNeighborSingle(void* latlongdegrees_rcp_view_ptr, const ExecutionSpaceKind& execution_space) noexcept;

KOKKOS_NET_API_EXTERNC void* NearestNeighborDouble(void* latlongdegrees_rcp_view_ptr, const ExecutionSpaceKind& execution_space) noexcept;

namespace Spatial
{
    static constexpr uint32 SurfaceIdx = 0;
    static constexpr uint32 BottomIdx  = 1;

    static constexpr uint32 LatitudeIdx  = 0;
    static constexpr uint32 LongitudeIdx = 1;

    static constexpr uint32 XIdx = 0;
    static constexpr uint32 YIdx = 1;

    template<typename DataType, class ExecutionSpace>
    using LineVector = Kokkos::View<DataType* [2][2], typename ExecutionSpace::array_layout, ExecutionSpace>;

    template<typename DataType, class ExecutionSpace>
    using PointVector = Kokkos::View<DataType* [2], typename ExecutionSpace::array_layout, ExecutionSpace>;

    template<typename DataType, class ExecutionSpace>
    using ValueVector = Kokkos::View<DataType*, typename ExecutionSpace::array_layout, ExecutionSpace>;

    /// <summary>
    /// Feet
    /// </summary>
    template<typename DataType>
    KOKKOS_FORCEINLINE_FUNCTION static constexpr DataType EarthCircumference()
    {
        return 131479718.77428;
    }

    template<typename DataType>
    KOKKOS_FORCEINLINE_FUNCTION static constexpr DataType EarthCircumferencePerDegree()
    {
        return EarthCircumference<DataType>() / 360.0;
    }

    template<typename DataType>
    KOKKOS_INLINE_FUNCTION static DataType DegreeToRadian(REF(DataType) degree)
    {
        return (Constants<DataType>::PI() / 180.0) * degree;
    }

    // template<typename DataType>
    // KOKKOS_FORCEINLINE_FUNCTION static constexpr DataType DistanceConversionFactor()
    //{
    //    return 6076.11549;
    //}
    //
    // enum EllipsoidKind
    //{
    //    Sphere = 0,
    //    WGS84,
    //    NAD27,
    //    International,
    //    Krasovsky,
    //    Bessel,
    //    WGS72,
    //    WGS66,
    //    FaiSphere
    //};
    //
    // template<typename DataType>
    // struct Ellipsoid
    //{
    //    const char* name;
    //    DataType    a;
    //    DataType    invf;
    //
    //    Ellipsoid(const char* const name, REF(DataType) a, REF(DataType) invf) : name(name), a(a), invf(invf) {}
    //};
    //
    // template<typename DataType>
    // KOKKOS_INLINE_FUNCTION static DataType GetEllipsoid(REF(EllipsoidKind) kind)
    //{
    //    switch(kind)
    //    {
    //        case Sphere:
    //        {
    //            return Ellipsoid<DataType>("Sphere", 180.0 * 60.0 / Constants<DataType>::PI(), Constants<DataType>::Infinity());
    //        }
    //        case NAD27:
    //        {
    //            return Ellipsoid<DataType>("NAD27", 6378.2064 / 1.852, 294.9786982138);
    //        }
    //        case International:
    //        {
    //            return Ellipsoid<DataType>("International", 6378.388 / 1.852, 297.0);
    //        }
    //        case Krasovsky:
    //        {
    //            return Ellipsoid<DataType>("Krasovsky", 6378.245 / 1.852, 298.3);
    //        }
    //        case Bessel:
    //        {
    //            return Ellipsoid<DataType>("Bessel", 6377.397155 / 1.852, 299.1528);
    //        }
    //        case WGS72:
    //        {
    //            return Ellipsoid<DataType>("WGS72", 6378.135 / 1.852, 298.26);
    //        }
    //        case WGS66:
    //        {
    //            return Ellipsoid<DataType>("WGS66", 6378.145 / 1.852, 298.25);
    //        }
    //        case FaiSphere:
    //        {
    //            return Ellipsoid<DataType>("FAI sphere", 6371.0 / 1.852, 1000000000.0);
    //        }
    //        case WGS84:
    //        default:
    //        {
    //            return Ellipsoid<DataType>("WGS84", 6378.137 / 1.852, 298.257223563);
    //        }
    //    }
    //}
    //
    // template<typename DataType>
    // KOKKOS_INLINE_FUNCTION static DataType CrsDistance(REF(DataType) lat1, REF(DataType) lon1, REF(DataType) lat2, REF(DataType) lon2)
    //{
    //    return acos(sin(lat1) * sin(lat2) + cos(lat1) * cos(lat2) * cos(lon1 - lon2));
    //}
    //
    // template<typename DataType>
    // KOKKOS_INLINE_FUNCTION static DataType CrsDistance(REF(DataType) lat1, REF(DataType) lon1, REF(DataType) lat2, REF(DataType) lon2, REF(Ellipsoid<DataType>) ellipse)
    //{
    //    const DataType EPS     = 0.00000000005;
    //    const uint32   MAXITER = 100;
    //
    //    const DataType a = ellipse.a;
    //    const DataType f = 1.0 / ellipse.invf;
    //
    //    DataType sx;
    //    DataType cx;
    //    DataType sy;
    //    DataType cy;
    //    DataType y;
    //    DataType sa;
    //    DataType c2a;
    //    DataType cz;
    //    DataType e;
    //    DataType c;
    //    DataType s;
    //
    //    DataType r   = 1.0 - f;
    //    DataType tu1 = r * tan(lat1);
    //    DataType tu2 = r * tan(lat2);
    //    DataType cu1 = 1.0 / sqrt(1. + tu1 * tu1);
    //    DataType su1 = cu1 * tu1;
    //    DataType cu2 = 1.0 / sqrt(1. + tu2 * tu2);
    //    DataType s1  = cu1 * cu2;
    //    DataType b1  = s1 * tu2;
    //    DataType f1  = b1 * tu1;
    //    DataType x   = lon2 - lon1;
    //    DataType d   = x + 1; // force one pass
    //
    //    uint32 iter = 1;
    //
    //    while((abs(d - x) > EPS) && (iter < MAXITER))
    //    {
    //        iter += 1;
    //
    //        sx = sin(x);
    //
    //        cx  = cos(x);
    //        tu1 = cu2 * sx;
    //        tu2 = b1 - su1 * cu2 * cx;
    //        sy  = sqrt(tu1 * tu1 + tu2 * tu2);
    //        cy  = s1 * cx + f1;
    //        y   = atan2(sy, cy);
    //        sa  = s1 * sx / sy;
    //        c2a = 1.0 - sa * sa;
    //        cz  = f1 + f1;
    //
    //        if(c2a > 0.0)
    //        {
    //            cz = cy - cz / c2a;
    //        }
    //
    //        e = cz * cz * 2.0 - 1.0;
    //        c = ((-3.0 * c2a + 4.0) * f + 4.0) * c2a * f / 16.0;
    //        d = x;
    //        x = ((e * cy * c + cz) * sy * c + y) * sa;
    //        x = (1.0 - c) * x * f + lon2 - lon1;
    //    }
    //
    //    x = sqrt((1.0 / (r * r) - 1.0) * c2a + 1.0);
    //    x += 1.0;
    //    x = (x - 2.0) / x;
    //    c = 1.0 - x;
    //    c = (x * x / 4.0 + 1.0) / c;
    //    d = (0.375 * x * x - 1.0) * x;
    //    x = e * cy;
    //
    //    return ((((sy * sy * 4.0 - 3.0) * (1.0 - e - e) * cz * d / 6.0 - x) * d / 4.0 + cz) * sy * d + y) * c * a * r;
    //}

    struct Length
    {
    };

    struct Distance
    {
    };

    struct Neighbor
    {
    };

    template<typename DataType>
    KOKKOS_INLINE_FUNCTION static DataType MidPoint(REF(DataType) a, REF(DataType) b)
    {
        return (a + b) / 2.0;
    }

    template<typename DataType>
    KOKKOS_INLINE_FUNCTION static DataType LongitudeToX(REF(DataType) lat, REF(DataType) lon)
    {
        return lon * EarthCircumferencePerDegree<DataType>() * cos(DegreeToRadian<DataType>(lat));
    }

    template<typename DataType>
    KOKKOS_INLINE_FUNCTION static DataType LatitudeToY(REF(DataType) lat)
    {
        return lat * EarthCircumferencePerDegree<DataType>();
    }

    template<typename DataType>
    KOKKOS_INLINE_FUNCTION static DataType AreaFromPoints(REF(DataType) x0, REF(DataType) y0, REF(DataType) x1, REF(DataType) y1, REF(DataType) x2, REF(DataType) y2)
    {
        return 0.5 * abs(((x0 - x2) * (y1 - y0)) - ((x0 - x1) * (y2 - y0)));
    }

    template<typename DataType>
    KOKKOS_INLINE_FUNCTION static DataType DistanceBetweenPointAndLines(REF(DataType) x0,
                                                                        REF(DataType) y0,
                                                                        REF(DataType) Px1,
                                                                        REF(DataType) Py1,
                                                                        REF(DataType) Px2,
                                                                        REF(DataType) Py2)
    {
        if(((Py2 - Py1) <= Constants<DataType>::Epsilon()) && ((Px2 - Px1) <= Constants<DataType>::Epsilon()))
        {
            return sqrt(pow(Py1 - y0, 2) + pow(Px1 - x0, 2));
        }

        return (abs(((Py2 - Py1) * x0) - ((Px2 - Px1) * y0) + (Px2 * Py1) - (Py2 * Px1))) / (sqrt(pow(Py2 - Py1, 2) + pow(Px2 - Px1, 2)));

        // const DataType area = AreaFromPoints(x0, y0, Px1, Py1, Px2, Py2);

        // return distance;
    }

    // template<typename DataType>
    // KOKKOS_INLINE_FUNCTION static DataType LineNormal(REF(DataType) x0, REF(DataType) y0, REF(DataType) x1, REF(DataType) y1)
    //{
    //    const DataType dx=x1-x0;
    //    const DataType dy=y1-y0;
    //
    //    //dy' = dx' * (-dx/dy)
    //    //dx' = dy' * (-dy/dx)
    //
    //    normals
    //
    //    (-dy, dx)
    //    (dy, -dx)
    //
    //    return lat * EarthCircumferencePerDegree<DataType>();
    //}

    template<typename DataType, class ExecutionSpace>
    struct NearestNeighborFunctor
    {
        const size_type N;

        const LineVector<DataType, ExecutionSpace> latlongdegrees;

        LineVector<DataType, ExecutionSpace> Cartesian;

        Kokkos::Extension::Vector<DataType, ExecutionSpace> Lengths;

        Kokkos::Extension::Matrix<DataType, ExecutionSpace> Distances;

        Kokkos::Extension::Vector<DataType, ExecutionSpace> Neighbors;

        NearestNeighborFunctor(const LineVector<DataType, ExecutionSpace>& latlongdegrees) : N(latlongdegrees.extent(0)), latlongdegrees(latlongdegrees)
        {
            Cartesian = LineVector<DataType, ExecutionSpace>("Cartesian", N);
            Lengths   = Kokkos::Extension::Vector<DataType, ExecutionSpace>("Lengths", N);
            Distances = Kokkos::Extension::Matrix<DataType, ExecutionSpace>("Distances", N, N);
            Neighbors = Kokkos::Extension::Vector<DataType, ExecutionSpace>("Neighbors", N);
        }

        KOKKOS_INLINE_FUNCTION void operator()(REF(Length), const size_type& i) const
        {
            Cartesian(i, SurfaceIdx, XIdx) = LongitudeToX<DataType>(latlongdegrees(i, SurfaceIdx, LatitudeIdx), latlongdegrees(i, SurfaceIdx, LongitudeIdx));

            Cartesian(i, SurfaceIdx, YIdx) = LatitudeToY<DataType>(latlongdegrees(i, SurfaceIdx, LatitudeIdx));

            Cartesian(i, BottomIdx, XIdx) = LongitudeToX<DataType>(latlongdegrees(i, BottomIdx, LatitudeIdx), latlongdegrees(i, BottomIdx, LongitudeIdx));
            Cartesian(i, BottomIdx, YIdx) = LatitudeToY<DataType>(latlongdegrees(i, BottomIdx, LatitudeIdx));

            Lengths(i) = sqrt(pow(Cartesian(i, BottomIdx, XIdx) - Cartesian(i, SurfaceIdx, XIdx), 2) + pow(Cartesian(i, BottomIdx, YIdx) - Cartesian(i, SurfaceIdx, YIdx), 2));
        }

        KOKKOS_INLINE_FUNCTION void operator()(REF(Distance), const size_type& i, const size_type& j) const
        {
            if(i == j)
            {
                Distances(i, j) = 0.0;
                return;
            }

            // if(Lengths(j) < 100.0)
            //{
            //    Distances(i, j) = Constants<DataType>::Max();
            //    return;
            //}

            const DataType wellbore = DistanceBetweenPointAndLines(Cartesian(j, SurfaceIdx, XIdx),
                                                                   Cartesian(j, SurfaceIdx, YIdx),
                                                                   Cartesian(i, SurfaceIdx, XIdx),
                                                                   Cartesian(i, SurfaceIdx, YIdx),
                                                                   Cartesian(i, BottomIdx, XIdx),
                                                                   Cartesian(i, BottomIdx, YIdx));

            const DataType surface = sqrt(pow(Cartesian(i, SurfaceIdx, YIdx) - Cartesian(j, SurfaceIdx, YIdx), 2) +
                                          pow(Cartesian(i, SurfaceIdx, XIdx) - Cartesian(j, SurfaceIdx, XIdx), 2));

            const DataType bottom = sqrt(pow(Cartesian(i, BottomIdx, YIdx) - Cartesian(j, BottomIdx, YIdx), 2) +
                                         pow(Cartesian(i, BottomIdx, XIdx) - Cartesian(j, BottomIdx, XIdx), 2));

            Distances(i, j) = max(surface, max(wellbore, bottom));
        }

        KOKKOS_INLINE_FUNCTION void operator()(REF(Neighbor), const size_type& i) const
        {
            DataType neighbor = Constants<DataType>::Max();

            for(size_type j = 0; j < N; j++)
            {
                if(Distances(i, j) < 1.0)
                {
                    continue;
                }

                if(Distances(i, j) < neighbor)
                {
                    neighbor = Distances(i, j);
                }
            }

            Neighbors(i) = neighbor;
        }
    };

    template<typename DataType, class ExecutionSpace>
    static Kokkos::Extension::Vector<DataType, ExecutionSpace> NearestNeighbor(const LineVector<DataType, ExecutionSpace>& latlongdegrees)
    {
        const size_type length = latlongdegrees.extent(0);

        NearestNeighborFunctor<DataType, ExecutionSpace> functor(latlongdegrees);

        //

        Kokkos::RangePolicy<ExecutionSpace, Kokkos::IndexType<size_type>, Length> LengthRange(0, length);

        Kokkos::parallel_for(LengthRange, functor);
        Kokkos::fence();

        //

        using mdrange_type = Kokkos::MDRangePolicy<ExecutionSpace, Kokkos::Rank<2>, Kokkos::IndexType<size_type>, Distance>;

        using point_type = typename mdrange_type::point_type;

        mdrange_type policy(point_type {{0, 0}}, point_type {{length, length}});

        Kokkos::parallel_for(policy, functor);
        Kokkos::fence();

        //

        Kokkos::RangePolicy<ExecutionSpace, Kokkos::IndexType<size_type>, Neighbor> NeighborRange(0, length);

        Kokkos::parallel_for(NeighborRange, functor);
        Kokkos::fence();

        //

        return functor.Neighbors;
    }

}

namespace Spatial
{

}
