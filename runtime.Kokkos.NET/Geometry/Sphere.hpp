#pragma once

#include <runtime.Kokkos/ViewTypes.hpp>

#include <Geometry/Point.hpp>

namespace Geometry
{

    template<typename TDataType>
    struct Sphere
    {
        typedef TDataType         value_type;
        typedef Point<value_type> point_type;
        static const int          Dim = 3;

    private:
        point_type m_center;
        value_type m_radius;

    public:
        Sphere(const point_type& x_center = point_type(), const value_type& x_radius = static_cast<value_type>(-1)) : m_center(x_center), m_radius(x_radius) {}

        KOKKOS_INLINE_FUNCTION constexpr void set_box(const Sphere& val)
        {
            m_center = val.m_center;
            m_radius = val.m_radius;
        }

        KOKKOS_INLINE_FUNCTION constexpr const point_type& center() const
        {
            return m_center;
        }
        KOKKOS_INLINE_FUNCTION constexpr point_type& center()
        {
            return m_center;
        }

        KOKKOS_INLINE_FUNCTION constexpr const value_type& radius() const
        {
            return m_radius;
        }
        KOKKOS_INLINE_FUNCTION constexpr value_type& radius()
        {
            return m_radius;
        }

        KOKKOS_INLINE_FUNCTION constexpr void set_center(const point_type& c)
        {
            m_center = c;
        }
        KOKKOS_INLINE_FUNCTION constexpr void set_radius(const value_type& r)
        {
            m_radius = r;
        }

        KOKKOS_INLINE_FUNCTION constexpr bool operator==(const Sphere<value_type>& s) const
        {
            return m_radius == s.m_radius && m_center == s.m_center;
        }

        KOKKOS_INLINE_FUNCTION constexpr bool operator!=(const Sphere<value_type>& s) const
        {
            return !(*this == s);
        }

        KOKKOS_INLINE_FUNCTION constexpr value_type get_x_min() const
        {
            return m_center[0] - m_radius;
        }
        KOKKOS_INLINE_FUNCTION constexpr value_type get_y_min() const
        {
            return m_center[1] - m_radius;
        }
        KOKKOS_INLINE_FUNCTION constexpr value_type get_z_min() const
        {
            return m_center[2] - m_radius;
        }
        KOKKOS_INLINE_FUNCTION constexpr value_type get_x_max() const
        {
            return m_center[0] + m_radius;
        }
        KOKKOS_INLINE_FUNCTION constexpr value_type get_y_max() const
        {
            return m_center[1] + m_radius;
        }
        KOKKOS_INLINE_FUNCTION constexpr value_type get_z_max() const
        {
            return m_center[2] + m_radius;
        }

        friend std::ostream& operator<<(std::ostream& out, const Sphere<value_type>& s)
        {
            out << "{" << s.center() << ":" << s.radius() << "}";
            return out;
        }
    };

}
