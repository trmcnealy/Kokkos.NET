#pragma once

#include <runtime.Kokkos/ViewTypes.hpp>

#include <Constants.hpp>

#include <Geometry/Point.hpp>

namespace Geometry
{
    template<typename TDataType>
    struct Box
    {
        typedef TDataType         value_type;
        typedef Point<value_type> point_type;
        static const int          Dim = 3;

    private:
        point_type m_min_corner;
        point_type m_max_corner;

    public:
        constexpr Box(const point_type& x_min_corner = point_type(Constants<TDataType>::Max(), Constants<TDataType>::Max(), Constants<TDataType>::Max()),
                      const point_type& x_max_corner = point_type(Constants<TDataType>::Lowest(), Constants<TDataType>::Lowest(), Constants<TDataType>::Lowest())) :
            m_min_corner(x_min_corner),
            m_max_corner(x_max_corner)
        {
        }

        constexpr Box(const TDataType x_min, const TDataType y_min, const TDataType z_min, const TDataType x_max, const TDataType y_max, const TDataType z_max) :
            m_min_corner(x_min, y_min, z_min),
            m_max_corner(x_max, y_max, z_max)
        {
        }

        KOKKOS_INLINE_FUNCTION constexpr const point_type& min_corner() const
        {
            return m_min_corner;
        }
        KOKKOS_INLINE_FUNCTION constexpr point_type& min_corner()
        {
            return m_min_corner;
        }
        KOKKOS_INLINE_FUNCTION constexpr const point_type& max_corner() const
        {
            return m_max_corner;
        }
        KOKKOS_INLINE_FUNCTION constexpr point_type& max_corner()
        {
            return m_max_corner;
        }

        KOKKOS_INLINE_FUNCTION constexpr value_type get_x_min() const
        {
            return m_min_corner[0];
        }
        KOKKOS_INLINE_FUNCTION constexpr value_type get_y_min() const
        {
            return m_min_corner[1];
        }
        KOKKOS_INLINE_FUNCTION constexpr value_type get_z_min() const
        {
            return m_min_corner[2];
        }
        KOKKOS_INLINE_FUNCTION constexpr value_type get_x_max() const
        {
            return m_max_corner[0];
        }
        KOKKOS_INLINE_FUNCTION constexpr value_type get_y_max() const
        {
            return m_max_corner[1];
        }
        KOKKOS_INLINE_FUNCTION constexpr value_type get_z_max() const
        {
            return m_max_corner[2];
        }

        KOKKOS_INLINE_FUNCTION constexpr void set_min_corner(const point_type& x_min_corner)
        {
            m_min_corner = x_min_corner;
        }
        KOKKOS_INLINE_FUNCTION constexpr void set_max_corner(const point_type& x_max_corner)
        {
            m_max_corner = x_max_corner;
        }

        KOKKOS_INLINE_FUNCTION constexpr void set_box(const value_type x1,
                                                      const value_type y1,
                                                      const value_type z1,
                                                      const value_type x2,
                                                      const value_type y2,
                                                      const value_type z2)
        {
            set_min_corner(point_type(x1, y1, z1));
            set_max_corner(point_type(x2, y2, z2));
        }

        KOKKOS_INLINE_FUNCTION constexpr bool operator==(const Box<value_type>& b) const
        {
            return m_min_corner == b.m_min_corner && m_max_corner == b.m_max_corner;
        }

        KOKKOS_INLINE_FUNCTION constexpr bool operator!=(const Box<value_type>& b) const
        {
            return !(*this == b);
        }

        KOKKOS_INLINE_FUNCTION constexpr void set_box(const Box& b)
        {
            set_min_corner(b.min_corner());
            set_max_corner(b.max_corner());
        }

        friend std::ostream& operator<<(std::ostream& out, const Box<value_type>& b)
        {
            out << "{" << b.min_corner() << "->" << b.max_corner() << "}";
            return out;
        }
    };
}
