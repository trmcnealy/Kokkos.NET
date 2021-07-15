#pragma once

#include "runtime.Kokkos/ViewTypes.hpp"

#include <Concepts.hpp>

namespace Internal
{
    template<typename T>
    KOKKOS_FORCEINLINE_FUNCTION static constexpr T abs(const T x) noexcept
    {
        return (x > 0) ? x : -x;
    }

    template<typename T>
    KOKKOS_FORCEINLINE_FUNCTION static constexpr int64 sign(const T x) noexcept
    {
        return (x == 0) ? 0 : (x < 0 ? -1 : 1);
    }

    
    KOKKOS_FORCEINLINE_FUNCTION static constexpr int64 gcd(const int64 x, const int64 y) noexcept
    {
        return (y == 0) ? x : ((x == 0) ? 1 : (gcd(y, x % y)));
    }

    KOKKOS_FORCEINLINE_FUNCTION static constexpr int64 lcm(const int64 x, const int64 y) noexcept
    {
        return x / (gcd(x, y) * y);
    }


}

template<typename Lhs, typename Rhs>
struct ratio_add;

template<typename Lhs, typename Rhs>
struct ratio_subtract;

template<typename Lhs, typename Rhs>
struct ratio_multiply;

template<typename Lhs, typename Rhs>
struct ratio_divide;

template<typename Lhs, typename Rhs>
struct ratio_gcd;

template<typename Lhs, typename Rhs>
struct ratio_lcm;

template<typename R>
struct ratio_negate;

template<typename R>
struct ratio_abs;

template<typename R>
struct ratio_sign;

template<typename R, int64 P>
struct ratio_power;

template<typename Lhs, typename Rhs>
struct ratio_equal;

template<typename Lhs, typename Rhs>
struct ratio_not_equal;

template<typename Lhs, typename Rhs>
struct ratio_less;

template<typename Lhs, typename Rhs>
struct ratio_less_equal;

template<typename Lhs, typename Rhs>
struct ratio_greater;

template<typename Lhs, typename Rhs>
struct ratio_greater_equal;

template<int64 Numerator, int64 Denominator = 1>
class Ratio
{
    inline static constexpr int64 AbsNumerator = Internal::abs(Numerator);
    inline static constexpr int64 AbsDenominator = Internal::abs(Denominator);
    inline static constexpr int64 SignRatio  = Internal::sign(Numerator) * Internal::sign(Denominator);

    inline static constexpr int64 GreatestCommonDivisor = Internal::gcd(AbsNumerator, AbsDenominator);

public:
    inline static constexpr int64 Num = SignRatio * AbsNumerator / GreatestCommonDivisor;
    inline static constexpr int64 Den = AbsDenominator / GreatestCommonDivisor;

    typedef Ratio<Num, Den> Type;

    template<typename R>
    static constexpr auto scale(R factor)
    {
        return (factor * static_cast<R>(Num)) / static_cast<R>(Den);
    }
};

typedef Ratio<1ULL, 1000000000000000000ULL> Atto;
typedef Ratio<1ULL, 1000000000000000ULL>    Femto;
typedef Ratio<1ULL, 1000000000000ULL>       Pico;
typedef Ratio<1ULL, 1000000000ULL>          Nano;
typedef Ratio<1ULL, 1000000ULL>             Micro;
typedef Ratio<1ULL, 1000ULL>                Milli;
typedef Ratio<1ULL, 100ULL>                 Centi;
typedef Ratio<1ULL, 10ULL>                  Deci;
typedef Ratio<10ULL, 1ULL>                  Deca;
typedef Ratio<100ULL, 1ULL>                 Hecto;
typedef Ratio<1000ULL, 1ULL>                Kilo;
typedef Ratio<1000000ULL, 1ULL>             Mega;
typedef Ratio<1000000000ULL, 1ULL>          Giga;
typedef Ratio<1000000000000ULL, 1ULL>       Tera;
typedef Ratio<1000000000000000ULL, 1ULL>    Peta;
typedef Ratio<1000000000000000000ULL, 1ULL> Exa;

#define BytesInKiB 1024ULL

typedef Ratio<BytesInKiB>            KiB;
typedef Ratio<KiB::Num * BytesInKiB> MeB;
typedef Ratio<MeB::Num * BytesInKiB> GiB;
typedef Ratio<GiB::Num * BytesInKiB> TeB;
typedef Ratio<TeB::Num * BytesInKiB> PeB;
typedef Ratio<PeB::Num * BytesInKiB> ExB;
