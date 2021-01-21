#pragma once

#include <runtime.Kokkos/ViewTypes.hpp>

#include <StdExtensions.hpp>

//#include <array>
//#include <cmath>
//#include <type_traits>
//#include <ostream>
//#include <iterator>
//#include <cassert>
//#include <string>
//#include <limits>

namespace Geometry
{

    enum class MemberInit
    {
        NONE
    };

    template<class TDataType, unsigned DIM>
    struct Vector
    {

        // typedef typename std::array<TDataType, DIM>::const_iterator const_iterator;
        // typedef typename std::array<TDataType, DIM>::iterator       iterator;

        typedef TDataType*       iterator;
        typedef const TDataType* const_iterator;

        TDataType vec[DIM];

        static const Vector<TDataType, DIM> ZERO;

        Vector(const Vector<TDataType, DIM>& rhs)
        {
            vec = rhs.vec;
        }
        Vector(Vector<TDataType, DIM>&&) = default;

        explicit Vector(const TDataType* rhs)
        {
            Assert(rhs);
            for (unsigned i = 0; i < DIM; ++i)
            {
                vec[i] = rhs[i];
            }
        }
        Vector(const TDataType* rhs, const unsigned len)
        {
            Assert(len == 0 || rhs);
            Assert(DIM >= len);
            for (unsigned i = 0; i < len; ++i)
            {
                vec[i] = rhs[i];
            }
            for (unsigned i = len; i < DIM; ++i)
            {
                vec[i] = 0;
            }
        }
        Vector(const TDataType x, const TDataType y, const TDataType z)
        {
            static_assert(DIM == 3, "Invalid dimension");

            vec[0] = x;
            vec[1] = y;
            vec[2] = z;
        }
        Vector(const TDataType x, const TDataType y)
        {
            static_assert(DIM == 2, "Invalid dimension");

            vec[0] = x;
            vec[1] = y;
        }
        explicit Vector(const TDataType x)
        {
            static_assert(DIM == 1, "Invalid dimension");

            vec[0] = x;
        }

        constexpr Vector() : vec{} {}

        Vector(MemberInit) {}

        KOKKOS_INLINE_FUNCTION constexpr Vector<TDataType, DIM>& operator=(const Vector<TDataType, DIM>& rhs)
        {
            vec = rhs.vec;
            return *this;
        }
        KOKKOS_INLINE_FUNCTION constexpr Vector<TDataType, DIM>& operator=(Vector<TDataType, DIM>&&) = default;
        KOKKOS_INLINE_FUNCTION constexpr Vector<TDataType, DIM>& operator-=(const Vector<TDataType, DIM>& rhs)
        {
            for (unsigned i = 0; i < DIM; ++i)
            {
                vec[i] -= rhs.vec[i];
            }
            return *this;
        }
        KOKKOS_INLINE_FUNCTION constexpr Vector<TDataType, DIM>& operator+=(const Vector<TDataType, DIM>& rhs)
        {
            for (unsigned i = 0; i < DIM; ++i)
            {
                vec[i] += rhs.vec[i];
            }
            return *this;
        }
        KOKKOS_INLINE_FUNCTION constexpr Vector<TDataType, DIM>& operator*=(const TDataType rhs)
        {
            for (unsigned i = 0; i < DIM; ++i)
            {
                vec[i] *= rhs;
            }
            return *this;
        }
        KOKKOS_INLINE_FUNCTION constexpr Vector<TDataType, DIM>& operator/=(const TDataType rhs)
        {
            for (unsigned i = 0; i < DIM; ++i)
            {
                vec[i] /= rhs;
            }
            return *this;
        }

        KOKKOS_INLINE_FUNCTION Vector<TDataType, DIM> operator-() const
        {
            Vector<TDataType, DIM> result(MemberInit::NONE);
            for (unsigned i = 0; i < DIM; ++i)
            {
                result.vec[i] = -vec[i];
            }
            return result;
        }
        KOKKOS_INLINE_FUNCTION Vector<TDataType, DIM> operator+(const Vector<TDataType, DIM>& rhs) const
        {
            Vector<TDataType, DIM> result(MemberInit::NONE);
            for (unsigned i = 0; i < DIM; ++i)
            {
                result.vec[i] = vec[i] + rhs.vec[i];
            }
            return result;
        }
        KOKKOS_INLINE_FUNCTION Vector<TDataType, DIM> operator-(const Vector<TDataType, DIM>& rhs) const
        {
            Vector<TDataType, DIM> result(MemberInit::NONE);
            for (unsigned i = 0; i < DIM; ++i)
            {
                result.vec[i] = vec[i] - rhs.vec[i];
            }
            return result;
        }

        KOKKOS_INLINE_FUNCTION constexpr bool zero_length() const
        {
            for (unsigned i = 0; i < DIM; ++i)
            {
                if (vec[i] != 0.0)
                {
                    return false;
                }
            }
            return true;
        }

        KOKKOS_INLINE_FUNCTION constexpr bool operator==(const Vector<TDataType, DIM>& rhs) const
        {
            // This is a strict equality!
            for (unsigned i = 0; i < DIM; ++i)
            {
                if (vec[i] != rhs.vec[i])
                {
                    return false;
                }
            }
            return true;
        }
        KOKKOS_INLINE_FUNCTION constexpr bool operator!=(const Vector<TDataType, DIM>& rhs) const
        {
            for (unsigned i = 0; i < DIM; ++i)
            {
                if (vec[i] != rhs.vec[i])
                {
                    return true;
                }
            }
            return false;
        }
        KOKKOS_INLINE_FUNCTION constexpr bool operator<(const Vector<TDataType, DIM>& rhs) const
        {
            for (unsigned i = 0; i < DIM; ++i)
            {
                if (vec[i] < rhs.vec[i])
                {
                    return true;
                }
                if (vec[i] > rhs.vec[i])
                {
                    return false;
                }
            }
            return false;
        }

        KOKKOS_INLINE_FUNCTION constexpr TDataType unitize()
        {
            const TDataType len = length();
            if (len > 0.0)
            {
                const TDataType inv_length = 1.0 / len;
                for (unsigned i = 0; i < DIM; ++i)
                {
                    vec[i] *= inv_length;
                }
            }
            else
            {
                for (unsigned i = 0; i < DIM; ++i)
                {
                    vec[i] = std::nan();
                }
            }
            return len;
        }
        KOKKOS_INLINE_FUNCTION constexpr TDataType length() const
        {
            TDataType lengthSquared = 0.0;
            for (unsigned i = 0; i < DIM; ++i)
            {
                lengthSquared += vec[i] * vec[i];
            }
            return std::sqrt(lengthSquared);
        }
        KOKKOS_INLINE_FUNCTION constexpr TDataType length_squared() const
        {
            TDataType lengthSquared = 0.0;
            for (unsigned i = 0; i < DIM; ++i)
            {
                lengthSquared += vec[i] * vec[i];
            }
            return lengthSquared;
        }
        KOKKOS_INLINE_FUNCTION constexpr Vector<TDataType, DIM> unit_vector() const
        {
            Vector<TDataType, DIM> result(MemberInit::NONE);
            for (unsigned i = 0; i < DIM; ++i)
            {
                result.vec[i] = vec[i];
            }
            result.unitize();
            return result;
        }

        KOKKOS_INLINE_FUNCTION constexpr const TDataType& operator[](const unsigned i) const
        {
            return vec[i];
        }
        KOKKOS_INLINE_FUNCTION constexpr TDataType& operator[](const unsigned i)
        {
            return vec[i];
        }
        KOKKOS_INLINE_FUNCTION constexpr TDataType* data()
        {
            return &vec[0];
        }
        KOKKOS_INLINE_FUNCTION constexpr const TDataType* data() const
        {
            return &vec[0];
        }

        friend std::ostream& operator<<(std::ostream& out, const Vector<TDataType, DIM>& rhs)
        {
            out << "Vector" << DIM << "d: ";
            for (unsigned i = 0; i < DIM; ++i)
            {
                out << rhs.vec[i] << " ";
            }
            return out;
        }

        std::string to_string() const
        {
            std::string output;
            for (size_t i = 0; i < DIM; i++)
            {
                output += std::to_string(vec[i]);
                if (i != DIM - 1)
                {
                    output += " ";
                }
            }
            return output;
        }

        KOKKOS_INLINE_FUNCTION constexpr unsigned dimension() const
        {
            return DIM;
        }
        KOKKOS_INLINE_FUNCTION constexpr unsigned size() const
        {
            return DIM;
        }

        KOKKOS_INLINE_FUNCTION constexpr iterator begin()
        {
            return &vec[0];
        }

        KOKKOS_INLINE_FUNCTION constexpr const_iterator begin() const
        {
            return &vec[0];
        }

        KOKKOS_INLINE_FUNCTION constexpr iterator end()
        {
            return (&vec[0]) + DIM;
        }

        KOKKOS_INLINE_FUNCTION constexpr const_iterator end() const
        {
            return (&vec[0]) + DIM;
        }
    };

    template<class TDataType, unsigned DIM>
    const Vector<TDataType, DIM> Vector<TDataType, DIM>::ZERO = Vector<TDataType, DIM>();

    template<class TDataType, unsigned DIM, class T>
    KOKKOS_INLINE_FUNCTION constexpr Vector<TDataType, DIM> operator*(const T scalar, const Vector<TDataType, DIM>& rhs)
    {
        Vector<TDataType, DIM> result(MemberInit::NONE);
        for (unsigned i = 0; i < DIM; ++i)
        {
            result[i] = scalar * rhs[i];
        }
        return result;
    }

    template<class TDataType, unsigned DIM, class T>
    KOKKOS_INLINE_FUNCTION constexpr Vector<TDataType, DIM> operator*(const Vector<TDataType, DIM>& rhs, const T scalar)
    {
        Vector<TDataType, DIM> result(MemberInit::NONE);
        for (unsigned i = 0; i < DIM; ++i)
        {
            result[i] = scalar * rhs[i];
        }
        return result;
    }

    template<class TDataType, unsigned DIM, class T>
    KOKKOS_INLINE_FUNCTION constexpr Vector<TDataType, DIM> operator/(const Vector<TDataType, DIM>& rhs, const T scalar)
    {
        Vector<TDataType, DIM> result(MemberInit::NONE);
        for (unsigned i = 0; i < DIM; ++i)
        {
            result[i] = rhs[i] / scalar;
        }
        return result;
    }

    template<class TDataType, unsigned DIM>
    KOKKOS_INLINE_FUNCTION constexpr TDataType Dot(const Vector<TDataType, DIM>& a, const Vector<TDataType, DIM>& b)
    {
        TDataType dot = 0.0;
        for (unsigned i = 0; i < DIM; ++i)
        {
            dot += a[i] * b[i];
        }
        return dot;
    }

    template<class TDataType, unsigned DIM>
    KOKKOS_INLINE_FUNCTION constexpr Vector<TDataType, DIM> Cross(const Vector<TDataType, DIM>& a, const Vector<TDataType, DIM>& b)
    {
        static_assert(DIM == 3, "Invalid dimension");
        return Vector<TDataType, DIM>(a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]);
    }

    /// Optimized Cross Products to Unit Vectors
    template<class TDataType, unsigned DIM>
    KOKKOS_INLINE_FUNCTION constexpr Vector<TDataType, DIM> crossX(const Vector<TDataType, DIM>& v)
    {
        static_assert(DIM == 3, "Invalid dimension");
        return Vector<TDataType, DIM>(0.0, v[2], -v[1]);
    }

    template<class TDataType, unsigned DIM>
    KOKKOS_INLINE_FUNCTION constexpr Vector<TDataType, DIM> crossY(const Vector<TDataType, DIM>& v)
    {
        static_assert(DIM == 3, "Invalid dimension");
        return Vector<TDataType, DIM>(-v[2], 0.0, v[0]);
    }

    template<class TDataType, unsigned DIM>
    KOKKOS_INLINE_FUNCTION constexpr Vector<TDataType, DIM> crossZ(const Vector<TDataType, DIM>& v)
    {
        static_assert(DIM == 3, "Invalid dimension");
        return Vector<TDataType, DIM>(v[1], -v[0], 0.0);
    }
}

namespace Geometry
{
    typedef Vector<float, 3> Vector3f;
    typedef Vector<float, 2> Vector2f;
    typedef Vector<float, 1> Vector1f;

    typedef Vector<double, 3> Vector3d;
    typedef Vector<double, 2> Vector2d;
    typedef Vector<double, 1> Vector1d;

}
