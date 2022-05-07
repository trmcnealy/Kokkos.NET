#pragma once

#include <runtime.Kokkos/ViewTypes.hpp>

#include <Constants.hpp>

#include <Geometry/Point.hpp>
#include <Geometry/Sphere.hpp>
#include <Geometry/Box.hpp>
#include <Geometry/Vector.hpp>

namespace Geometry
{

    // is_valid: Point
    template<typename TDataType>
    KOKKOS_INLINE_FUNCTION static constexpr bool is_valid(Point<TDataType> const&)
    {
        return true;
    }

    // is_valid: Sphere
    template<typename TDataType>
    KOKKOS_INLINE_FUNCTION static constexpr bool is_valid(Sphere<TDataType> const& s)
    {
        return static_cast<TDataType>(0) <= s.radius();
    }

    // is_valid: Box
    template<typename TDataType>
    KOKKOS_INLINE_FUNCTION static constexpr bool is_valid(Box<TDataType> const& b)
    {
        return b.min_corner()[0] <= b.max_corner()[0] && b.min_corner()[1] <= b.max_corner()[1] && b.min_corner()[2] <= b.max_corner()[2];
    }

    // Point
    template<typename TDataType>
    KOKKOS_INLINE_FUNCTION static constexpr Point<TDataType> const& min_corner(Point<TDataType> const& p)
    {
        return p;
    }

    template<typename TDataType>
    KOKKOS_INLINE_FUNCTION static constexpr Point<TDataType> const& max_corner(Point<TDataType> const& p)
    {
        return p;
    }

    template<typename TDataType>
    KOKKOS_INLINE_FUNCTION static constexpr Point<TDataType> const& center(Point<TDataType> const& p)
    {
        return p;
    }

    // Point,index
    template<typename TDataType>
    KOKKOS_INLINE_FUNCTION static constexpr TDataType min_corner(Point<TDataType> const& p, int index)
    {
        return p[index];
    }

    template<typename TDataType>
    KOKKOS_INLINE_FUNCTION static constexpr TDataType max_corner(Point<TDataType> const& p, int index)
    {
        return p[index];
    }

    template<typename TDataType>
    KOKKOS_INLINE_FUNCTION static constexpr TDataType center(Point<TDataType> const& p, int index)
    {
        return p[index];
    }

    // Sphere
    template<typename TDataType>
    KOKKOS_INLINE_FUNCTION static constexpr Point<TDataType> const min_corner(Sphere<TDataType> const& s)
    {
        const Point<TDataType>& c = s.center();
        const TDataType         r = s.radius();
        return Point<TDataType>(c[0] - r, c[1] - r, c[2] - r);
    }

    template<typename TDataType>
    KOKKOS_INLINE_FUNCTION static constexpr Point<TDataType> const max_corner(Sphere<TDataType> const& s)
    {
        const Point<TDataType>& c = s.center();
        const TDataType         r = s.radius();
        return Point<TDataType>(c[0] + r, c[1] + r, c[2] + r);
    }

    template<typename TDataType>
    KOKKOS_INLINE_FUNCTION static constexpr Point<TDataType> const& center(Sphere<TDataType> const& s)
    {
        return s.center();
    }

    // Sphere,index
    template<typename TDataType>
    KOKKOS_INLINE_FUNCTION static constexpr TDataType min_corner(Sphere<TDataType> const& s, int index)
    {
        return s.center()[index] - s.radius();
    }

    template<typename TDataType>
    KOKKOS_INLINE_FUNCTION static constexpr TDataType max_corner(Sphere<TDataType> const& s, int index)
    {
        return s.center()[index] + s.radius();
    }

    template<typename TDataType>
    KOKKOS_INLINE_FUNCTION static constexpr TDataType center(Sphere<TDataType> const& s, int index)
    {
        return s.center()[index];
    }

    // Box
    template<typename TDataType>
    KOKKOS_INLINE_FUNCTION static constexpr Point<TDataType> const& min_corner(Box<TDataType> const& b)
    {
        return b.min_corner();
    }

    template<typename TDataType>
    KOKKOS_INLINE_FUNCTION static constexpr Point<TDataType> const& max_corner(Box<TDataType> const& b)
    {
        return b.max_corner();
    }

    template<typename TDataType>
    KOKKOS_INLINE_FUNCTION static constexpr Point<TDataType> const center(Box<TDataType> const& b)
    {
        const Point<TDataType>& l = b.min_corner();
        const Point<TDataType>& u = b.max_corner();
        return Point<TDataType>((l[0] + u[0]) / 2, (l[1] + u[1]) / 2, (l[2] + u[2]) / 2);
    }

    // Box,index
    template<typename TDataType>
    KOKKOS_INLINE_FUNCTION static constexpr TDataType min_corner(Box<TDataType> const& b, int index)
    {
        return b.min_corner()[index];
    }

    template<typename TDataType>
    KOKKOS_INLINE_FUNCTION static constexpr TDataType max_corner(Box<TDataType> const& b, int index)
    {
        return b.max_corner()[index];
    }

    template<typename TDataType>
    KOKKOS_INLINE_FUNCTION static constexpr TDataType center(Box<TDataType> const& b, int index)
    {
        return (b.min_corner()[index] + b.max_corner()[index]) / 2;
    }

    // intersects: Point,Point
    template<typename TDataType>
    KOKKOS_INLINE_FUNCTION static constexpr bool intersects(Point<TDataType> const& a, Point<TDataType> const& b)
    {
        return (a == b);
    }

    // intersects: Point,Sphere
    template<typename TDataType>
    KOKKOS_INLINE_FUNCTION static constexpr bool intersects(Point<TDataType> const& a, Sphere<TDataType> const& b)
    {
        const TDataType dist2 = (a[0] - b.center()[0]) * (a[0] - b.center()[0]) + (a[1] - b.center()[1]) * (a[1] - b.center()[1]) + (a[2] - b.center()[2]) * (a[2] - b.center()[2]);
        return (dist2 <= b.radius() * b.radius());
    }

    // intersects: Sphere,Point
    template<typename TDataType>
    KOKKOS_INLINE_FUNCTION static constexpr bool intersects(Sphere<TDataType> const& a, Point<TDataType> const& b)
    {
        return intersects(b, a);
    }

    // intersects: Point,Box
    template<typename TDataType>
    KOKKOS_INLINE_FUNCTION static constexpr bool intersects(Point<TDataType> const& a, Box<TDataType> const& b)
    {
        return b.min_corner()[0] <= a[0] && a[0] <= b.max_corner()[0] && b.min_corner()[1] <= a[1] && a[1] <= b.max_corner()[1] && b.min_corner()[2] <= a[2] &&
               a[2] <= b.max_corner()[2];
    }

    // intersects: Box,Point
    template<typename TDataType>
    KOKKOS_INLINE_FUNCTION static constexpr bool intersects(Box<TDataType> const& a, Point<TDataType> const& b)
    {
        return intersects(b, a);
    }

    // intersects: Sphere,Sphere
    template<typename TDataType>
    KOKKOS_INLINE_FUNCTION static constexpr bool intersects(Sphere<TDataType> const& a, Sphere<TDataType> const& b)
    {
        const Point<TDataType>& ac    = a.center();
        const Point<TDataType>& bc    = b.center();
        const TDataType         r2    = (a.radius() + b.radius()) * (a.radius() + b.radius());
        const TDataType         dist2 = (ac[0] - bc[0]) * (ac[0] - bc[0]) + (ac[1] - bc[1]) * (ac[1] - bc[1]) + (ac[2] - bc[2]) * (ac[2] - bc[2]);
        return dist2 < r2;
    }

    // intersects: Sphere,Box
    template<typename T1, typename T2>
    KOKKOS_INLINE_FUNCTION static constexpr bool intersects(Sphere<T1> const& a, Box<T2> const& b)
    {
        const Point<T1>& ac   = a.center();
        const Point<T2>& bmin = b.min_corner();
        const Point<T2>& bmax = b.max_corner();

        const T1 r2 = a.radius() * a.radius();

        // check that the nearest point in the bounding box is within the sphere
        T1 dmin = 0;
        for (int i = 0; i < 3; ++i)
        {
            if (ac[i] < bmin[i])
                dmin += (ac[i] - bmin[i]) * (ac[i] - bmin[i]);
            else if (ac[i] > bmax[i])
                dmin += (ac[i] - bmax[i]) * (ac[i] - bmax[i]);
        }
        return dmin <= r2;
    }

    // intersects: Box,Sphere
    template<typename T1, typename T2>
    KOKKOS_INLINE_FUNCTION static constexpr bool intersects(Box<T1> const& a, Sphere<T2> const& b)
    {
        return intersects(b, a);
    }

    // intersects: Box,Box
    template<typename T1, typename T2>
    KOKKOS_INLINE_FUNCTION static constexpr bool intersects(Box<T1> const& a, Box<T2> const& b)
    {
        const Point<T1>& amin = a.min_corner();
        const Point<T1>& amax = a.max_corner();

        const Point<T2>& bmin = b.min_corner();
        const Point<T2>& bmax = b.max_corner();

        // check that the boxes are not disjoint
        return !((amax[0] < bmin[0]) || (bmax[0] < amin[0]) || (amax[1] < bmin[1]) || (bmax[1] < amin[1]) || (amax[2] < bmin[2]) || (bmax[2] < amin[2]));
    }

    template<typename TDataType, typename U>
    KOKKOS_INLINE_FUNCTION static constexpr void scale_by(Sphere<TDataType>& s, U const& c)
    {
        s.set_radius(s.radius() * c);
    }

    template<typename TDataType, typename U>
    KOKKOS_INLINE_FUNCTION static void scale_by(Box<TDataType>& b, U const& c)
    {
        Point<TDataType>& min_corner = b.min_corner();
        Point<TDataType>& max_corner = b.max_corner();
        const U   factor     = (c - 1) / 2;
        for (int i = 0; i < 3; ++i)
        {
            const TDataType d = factor * (max_corner[i] - min_corner[i]);
            min_corner[i] -= d;
            max_corner[i] += d;
        }
    }

    template<typename TDataType1, typename TDataType2>
    KOKKOS_INLINE_FUNCTION static constexpr void add_to_box(Box<TDataType1>& box, const Box<TDataType2>& addBox)
    {
        box.set_box(System::min(box.get_x_min(), static_cast<TDataType1>(addBox.get_x_min())),
                    System::min(box.get_y_min(), static_cast<TDataType1>(addBox.get_y_min())),
                    System::min(box.get_z_min(), static_cast<TDataType1>(addBox.get_z_min())),
                    System::max(box.get_x_max(), static_cast<TDataType1>(addBox.get_x_max())),
                    System::max(box.get_y_max(), static_cast<TDataType1>(addBox.get_y_max())),
                    System::max(box.get_z_max(), static_cast<TDataType1>(addBox.get_z_max())));
    }

    template<typename TDataType1, typename TDataType2>
    KOKKOS_INLINE_FUNCTION static constexpr void add_to_box(Box<TDataType1>& box, const Sphere<TDataType2>& addBox)
    {
        box.set_box(System::min(box.get_x_min(), addBox.get_x_min()),
                    System::min(box.get_y_min(), addBox.get_y_min()),
                    System::min(box.get_z_min(), addBox.get_z_min()),
                    System::max(box.get_x_max(), addBox.get_x_max()),
                    System::max(box.get_y_max(), addBox.get_y_max()),
                    System::max(box.get_z_max(), addBox.get_z_max()));
    }

    template<typename TDataType1, typename TDataType2>
    KOKKOS_INLINE_FUNCTION static constexpr void add_to_box(Box<TDataType1>& box, const Point<TDataType2>& addBox)
    {
        box.set_box(System::min(box.get_x_min(), addBox.get_x_min()),
                    System::min(box.get_y_min(), addBox.get_y_min()),
                    System::min(box.get_z_min(), addBox.get_z_min()),
                    System::max(box.get_x_max(), addBox.get_x_max()),
                    System::max(box.get_y_max(), addBox.get_y_max()),
                    System::max(box.get_z_max(), addBox.get_z_max()));
    }

    // This algorithm is based off the minimum circle for a triangle blog post
    // by Christer Ericson at http://realtimecollisiondetection.net/blog/?p=20
    template<typename TDataType>
    static Sphere<TDataType> minimumBoundingSphere(const Point<TDataType>& ptA, const Point<TDataType>& ptB, const Point<TDataType>& ptC)
    {
        typedef Vector<TDataType, 3> Vec;

        Vec a = Vec(ptA[0], ptA[1], ptA[2]);
        Vec b = Vec(ptB[0], ptB[1], ptB[2]);
        Vec c = Vec(ptC[0], ptC[1], ptC[2]);

        Vec AB = b - a;
        Vec AC = c - a;

        TDataType dotABAB = KokkosBlas::dot(AB, AB);
        TDataType dotABAC = KokkosBlas::dot(AB, AC);
        TDataType dotACAC = KokkosBlas::dot(AC, AC);

        TDataType d           = 2.0 * (dotABAB * dotACAC - dotABAC * dotABAC);
        Vec  referencePt = a;

        Vec center;

        if (std::abs(d) <= 100 * Constants<TDataType>::Epsilon())
        {
            // a, b, and c lie on a line. Circle center is center of AABB of the
            // points, and radius is distance from circle center to AABB corner
            Box<TDataType> aabb = Box<TDataType>(Point<TDataType>(a[0], a[1], a[2]), Point<TDataType>(b[0], b[1], b[2]));

            add_to_box(aabb, Point<TDataType>(c[0], c[1], c[2]));

            Point<TDataType> minCornerPt = aabb.min_corner();
            Point<TDataType> maxCornerPt = aabb.max_corner();
            Vec         minCorner   = Vec(minCornerPt[0], minCornerPt[1], minCornerPt[2]);
            Vec         maxCorner   = Vec(maxCornerPt[0], maxCornerPt[1], maxCornerPt[2]);
            center                  = 0.5 * (minCorner + maxCorner);
            referencePt             = minCorner;
        }
        else
        {
            TDataType s = (dotABAB * dotACAC - dotACAC * dotABAC) / d;
            TDataType t = (dotACAC * dotABAB - dotABAB * dotABAC) / d;

            // s controls height over AC, t over AB, (1-s-t) over BC
            if (s <= 0.0f)
            {
                center = 0.5 * (a + c);
            }
            else if (t <= 0.0f)
            {
                center = 0.5 * (a + b);
            }
            else if (s + t >= 1.0)
            {
                center      = 0.5 * (b + c);
                referencePt = b;
            }
            else
            {
                center = a + s * (b - a) + t * (c - a);
            }
        }

        TDataType radius = std::sqrt(KokkosBlas::dot(center - referencePt, center - referencePt));

        return Sphere<TDataType>(Point<TDataType>(center[0], center[1], center[2]), radius);
    }

}
