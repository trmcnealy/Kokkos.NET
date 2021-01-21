#pragma once

#include <runtime.Kokkos/ViewTypes.hpp>

#include <iosfwd>

namespace Geometry
{
    template<typename TDataType>
    struct Point
    {
        typedef TDataType     value_type;
        static const unsigned Dim = 3;

    private:
        value_type m_value[Dim];

    public:
        Point(value_type x = value_type(), value_type y = value_type(), value_type z = value_type()) : m_value{x, y, z} {}

        KOKKOS_FORCEINLINE_FUNCTION constexpr void operator=(const Point<value_type>& pt)
        {
            for (unsigned i = 0; i < Dim; ++i)
            {
                m_value[i] = pt.m_value[i];
            }
        }

        KOKKOS_INLINE_FUNCTION constexpr value_type& operator[](size_t index)
        {
            return m_value[index];
        }

        KOKKOS_INLINE_FUNCTION constexpr const value_type& operator[](size_t index) const
        {
            return m_value[index];
        }

        KOKKOS_INLINE_FUNCTION constexpr bool operator==(const Point<value_type>& p) const
        {
            return m_value[0] == p.m_value[0] && m_value[1] == p.m_value[1] && m_value[2] == p.m_value[2];
        }

        KOKKOS_INLINE_FUNCTION constexpr bool operator!=(const Point<value_type>& p) const
        {
            return !(*this == p);
        }

        KOKKOS_INLINE_FUNCTION constexpr value_type get_x_min() const
        {
            return m_value[0];
        }
        KOKKOS_INLINE_FUNCTION constexpr value_type get_y_min() const
        {
            return m_value[1];
        }
        KOKKOS_INLINE_FUNCTION constexpr value_type get_z_min() const
        {
            return m_value[2];
        }
        KOKKOS_INLINE_FUNCTION constexpr value_type get_x_max() const
        {
            return m_value[0];
        }
        KOKKOS_INLINE_FUNCTION constexpr value_type get_y_max() const
        {
            return m_value[1];
        }
        KOKKOS_INLINE_FUNCTION constexpr value_type get_z_max() const
        {
            return m_value[2];
        }

        friend std::ostream& operator<<(std::ostream& out, const Point<value_type>& p)
        {
            out << "(" << p[0] << "," << p[1] << "," << p[2] << ")";
            return out;
        }
    };
}
