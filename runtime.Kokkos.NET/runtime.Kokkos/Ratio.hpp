#pragma once

#include "runtime.Kokkos/ViewTypes.hpp"

#include <Concepts.hpp>

namespace Internal
{
    template<int64 X, int64 Y>
    struct GreatestCommonDivisor
    {
        static constexpr int64 Value = GreatestCommonDivisor<Y, X % Y>::Value;
    };

    template<int64 X>
    struct GreatestCommonDivisor<X, 0>
    {
        static constexpr int64 Value = X;
    };

    template<>
    struct GreatestCommonDivisor<0, 0>
    {
        static constexpr int64 Value = 1;
    };

    template<int64 X, int64 Y>
    struct LeastCommonMultiple
    {
        static constexpr int64 Value = X / GreatestCommonDivisor<X, Y>::Value * Y;
    };

    // template<int64 X>
    // struct static_abs
    //{
    //    static constexpr const int64 Value = X > 0 ? X : -X;
    //};

    // template<int64 X>
    // struct static_sign
    //{
    //    static constexpr const int64 Value = X == 0 ? 0 : (X < 0 ? -1 : 1);
    //};

    template<typename T>
    KOKKOS_FORCEINLINE_FUNCTION static constexpr T abs(const T x) noexcept
    {
        return x > 0 ? x : -x;
    }

    template<typename T>
    KOKKOS_FORCEINLINE_FUNCTION static constexpr int64 sign(const T x) noexcept
    {
        return x == 0 ? 0 : (x < 0 ? -1 : 1);
    }

    template<int64 X, int64 Y, int64 = sign(Y)>
    class ll_add;

    template<int64 X, int64 Y>
    class ll_add<X, Y, 1>
    {
        static constexpr int64 Min = (1LL << (int64)(sizeof(int64) * CHAR_BIT - 1LL)) + 1LL;
        static constexpr int64 Max = -Min;

    public:
        static constexpr int64 Value = X + Y;
    };

    template<int64 X, int64 Y>
    class ll_add<X, Y, 0>
    {
    public:
        static constexpr int64 Value = X;
    };

    template<int64 X, int64 Y>
    class ll_add<X, Y, -1>
    {
        static constexpr int64 Min = (1LL << (int64)(sizeof(int64) * CHAR_BIT - 1LL)) + 1LL;
        static constexpr int64 Max = -Min;

    public:
        static constexpr int64 Value = X + Y;
    };

    template<int64 X, int64 Y, int64 = sign(Y)>
    class ll_sub;

    template<int64 X, int64 Y>
    class ll_sub<X, Y, 1>
    {
        static constexpr int64 Min = (1LL << (int64)(sizeof(int64) * CHAR_BIT - 1LL)) + 1LL;
        static constexpr int64 Max = -Min;

    public:
        static constexpr int64 Value = X - Y;
    };

    template<int64 X, int64 Y>
    class ll_sub<X, Y, 0>
    {
    public:
        static constexpr int64 Value = X;
    };

    template<int64 X, int64 Y>
    class ll_sub<X, Y, -1>
    {
        static constexpr int64 Min = (1LL << (int64)(sizeof(int64) * CHAR_BIT - 1LL)) + 1LL;
        static constexpr int64 Max = -Min;

    public:
        static constexpr int64 Value = X - Y;
    };

    template<int64 X, int64 Y>
    class ll_mul
    {
        static constexpr int64 NaN = (1LL << (int64)(sizeof(int64) * CHAR_BIT - 1LL));
        static constexpr int64 Min = NaN + 1;
        static constexpr int64 Max = -Min;
        static constexpr int64 a_x = abs(X);
        static constexpr int64 a_y = abs(Y);

    public:
        static constexpr int64 Value = X * Y;
    };

    template<int64 Y>
    class ll_mul<0, Y>
    {
    public:
        static constexpr int64 Value = 0;
    };

    template<int64 X>
    class ll_mul<X, 0>
    {
    public:
        static constexpr int64 Value = 0;
    };

    template<>
    class ll_mul<0, 0>
    {
    public:
        static constexpr int64 Value = 0;
    };

    template<int64 X, int64 Y>
    class ll_div
    {
        static constexpr int64 NaN = (1LL << (int64)(sizeof(int64) * CHAR_BIT - 1LL));
        static constexpr int64 Min = NaN + 1;
        static constexpr int64 Max = -Min;

    public:
        static constexpr int64 Value = X / Y;
    };
}

template<int64 Numerator, int64 Denominator = 1>
class Ratio
{
    static constexpr int64 na = Internal::abs(Numerator);
    static constexpr int64 da = Internal::abs(Denominator);
    static constexpr int64 s  = Internal::sign(Numerator) * Internal::sign(Denominator);

    static constexpr int64 gcd = Internal::GreatestCommonDivisor<na, da>::Value;

public:
    static constexpr int64 Num = s * na / gcd;
    static constexpr int64 Den = da / gcd;

    typedef Ratio<Num, Den> Type;

    template<typename R>
    static constexpr auto scale(R factor)
    {
        return (factor * static_cast<R>(Num)) / static_cast<R>(Den);
    }
};

template<int64 Numerator, int64 Denominator>
constexpr int64 Ratio<Numerator, Denominator>::Num;

template<int64 Numerator, int64 Denominator>
constexpr int64 Ratio<Numerator, Denominator>::Den;

typedef Ratio<1LL, 1000000000000000000LL> Atto;
typedef Ratio<1LL, 1000000000000000LL>    Femto;
typedef Ratio<1LL, 1000000000000LL>       Pico;
typedef Ratio<1LL, 1000000000LL>          Nano;
typedef Ratio<1LL, 1000000LL>             Micro;
typedef Ratio<1LL, 1000LL>                Milli;
typedef Ratio<1LL, 100LL>                 Centi;
typedef Ratio<1LL, 10LL>                  Deci;
typedef Ratio<10LL, 1LL>                  Deca;
typedef Ratio<100LL, 1LL>                 Hecto;
typedef Ratio<1000LL, 1LL>                Kilo;
typedef Ratio<1000000LL, 1LL>             Mega;
typedef Ratio<1000000000LL, 1LL>          Giga;
typedef Ratio<1000000000000LL, 1LL>       Tera;
typedef Ratio<1000000000000000LL, 1LL>    Peta;
typedef Ratio<1000000000000000000LL, 1LL> Exa;

namespace Internal
{
    template<typename LHS, typename RHS>
    struct RatioMultiply
    {
    private:
        static const int64 gcd_n1_d2 = GreatestCommonDivisor<LHS::Num, RHS::Den>::Value;
        static const int64 gcd_d1_n2 = GreatestCommonDivisor<LHS::Den, RHS::Num>::Value;

    public:
        typedef typename Ratio<ll_mul<LHS::Num / gcd_n1_d2, RHS::Num / gcd_d1_n2>::Value, ll_mul<RHS::Den / gcd_n1_d2, LHS::Den / gcd_d1_n2>::Value>::Type Type;
    };
}

template<typename LHS, typename RHS>
using RatioMultiply = typename Internal::RatioMultiply<LHS, RHS>::Type;

namespace Internal
{
    template<typename LHS, typename RHS>
    struct RatioDivide
    {
    private:
        static const int64 gcd_n1_n2 = GreatestCommonDivisor<LHS::Num, RHS::Num>::Value;
        static const int64 gcd_d1_d2 = GreatestCommonDivisor<LHS::Den, RHS::Den>::Value;

    public:
        typedef typename Ratio<ll_mul<LHS::Num / gcd_n1_n2, RHS::Den / gcd_d1_d2>::Value, ll_mul<RHS::Num / gcd_n1_n2, LHS::Den / gcd_d1_d2>::Value>::Type Type;
    };

}

template<typename LHS, typename RHS>
using RatioDivide = typename Internal::RatioDivide<LHS, RHS>::Type;

namespace Internal
{
    template<typename LHS, typename RHS>
    struct RatioAdd
    {
    private:
        static const int64 gcd_n1_n2 = GreatestCommonDivisor<LHS::Num, RHS::Num>::Value;
        static const int64 gcd_d1_d2 = GreatestCommonDivisor<LHS::Den, RHS::Den>::Value;

    public:
        typedef typename RatioMultiply<
            Ratio<gcd_n1_n2, LHS::Den / gcd_d1_d2>,
            Ratio<ll_add<ll_mul<LHS::Num / gcd_n1_n2, RHS::Den / gcd_d1_d2>::Value, ll_mul<RHS::Num / gcd_n1_n2, LHS::Den / gcd_d1_d2>::Value>::Value, RHS::Den>>::Type Type;
    };

}

template<typename LHS, typename RHS>
using RatioAdd = typename Internal::RatioAdd<LHS, RHS>::Type;

namespace Internal
{
    template<typename LHS, typename RHS>
    struct RatioSubtract
    {
    private:
        static const int64 gcd_n1_n2 = GreatestCommonDivisor<LHS::Num, RHS::Num>::Value;
        static const int64 gcd_d1_d2 = GreatestCommonDivisor<LHS::Den, RHS::Den>::Value;

    public:
        typedef typename RatioMultiply<
            Ratio<gcd_n1_n2, LHS::Den / gcd_d1_d2>,
            Ratio<ll_sub<ll_mul<LHS::Num / gcd_n1_n2, RHS::Den / gcd_d1_d2>::Value, ll_mul<RHS::Num / gcd_n1_n2, LHS::Den / gcd_d1_d2>::Value>::Value, RHS::Den>>::Type Type;
    };

}

template<typename LHS, typename RHS>
using RatioSubtract = typename Internal::RatioSubtract<LHS, RHS>::Type;

namespace Internal
{
    template<typename RatioT>
    struct RatioInvert;

    template<int64 Num, int64 Den>
    struct RatioInvert<Ratio<Num, Den>>
    {
        using Type = Ratio<-Num, Den>;
    };
}

template<typename RatioT>
using RatioInvert = typename Internal::RatioInvert<RatioT>::Type;

namespace Internal
{
    template<typename R, int64 N>
    struct RatioPow
    {
        using Type = typename RatioMultiply<typename RatioPow<R, N - 1>::Type, R>::Type;
    };

    template<typename R>
    struct RatioPow<R, 1>
    {
        using Type = R;
    };

    template<typename R>
    struct RatioPow<R, 0>
    {
        using Type = Ratio<1>;
    };
}

template<typename R, int64 N>
using RatioPow = typename Internal::RatioPow<R, N>::Type;

namespace Internal
{
    KOKKOS_FORCEINLINE_FUNCTION static constexpr int64 Sqrt(const int64 v, const int64 l, const int64 r)
    {
        if (l == r)
        {
            return r;
        }

        const auto mid = (r + l) / 2;

        if (mid * mid >= v)
        {
            return Sqrt(v, l, mid);
        }

        return Sqrt(v, mid + 1, r);
    }

    KOKKOS_FORCEINLINE_FUNCTION static constexpr int64 Sqrt(const int64 v)
    {
        return Sqrt(v, 1, v);
    }

    template<typename R>
    struct RatioSqrt
    {
        using Type = Ratio<Sqrt(R::num), Sqrt(R::den)>;
    };

    template<int64 Den>
    struct RatioSqrt<Ratio<0, Den>>
    {
        using Type = Ratio<0>;
    };

}

template<typename R>
using RatioSqrt = typename Internal::RatioSqrt<R>::Type;

template<typename LHS, typename RHS>
struct RatioEqual : std::bool_constant<(LHS::Num == RHS::Num && LHS::Den == RHS::Den)>
{
};

template<typename LHS, typename RHS>
struct RatioNotEqual : std::bool_constant<(!RatioEqual<LHS, RHS>::Value)>
{
};

namespace Internal
{
    template<typename LHS,
             typename RHS,
             bool  _Odd = false,
             int64 _Q1  = LHS::Num / LHS::Den,
             int64 _M1  = LHS::Num % LHS::Den,
             int64 _Q2  = RHS::Num / RHS::Den,
             int64 _M2  = RHS::Num % RHS::Den>
    struct RatioLess
    {
        static const bool Value = _Odd ? _Q2 < _Q1 : _Q1 < _Q2;
    };

    template<typename LHS, typename RHS, bool _Odd, int64 _Qp>
    struct RatioLess<LHS, RHS, _Odd, _Qp, 0, _Qp, 0>
    {
        static const bool Value = false;
    };

    template<typename LHS, typename RHS, bool _Odd, int64 _Qp, int64 _M2>
    struct RatioLess<LHS, RHS, _Odd, _Qp, 0, _Qp, _M2>
    {
        static const bool Value = !_Odd;
    };

    template<typename LHS, typename RHS, bool _Odd, int64 _Qp, int64 _M1>
    struct RatioLess<LHS, RHS, _Odd, _Qp, _M1, _Qp, 0>
    {
        static const bool Value = _Odd;
    };

    template<typename LHS, typename RHS, bool _Odd, int64 _Qp, int64 _M1, int64 _M2>
    struct RatioLess<LHS, RHS, _Odd, _Qp, _M1, _Qp, _M2>
    {
        static const bool Value = RatioLess<Ratio<LHS::Den, _M1>, Ratio<RHS::Den, _M2>, !_Odd>::Value;
    };

    template<typename LHS, typename RHS, int64 S1 = Internal::sign(LHS::Num), int64 S2 = Internal::sign(RHS::Num)>
    struct RatioLess_t
    {
        static const bool Value = S1 < S2;
    };

    template<typename LHS, typename RHS>
    struct RatioLess_t<LHS, RHS, 1LL, 1LL>
    {
        static const bool Value = RatioLess<LHS, RHS>::Value;
    };

    template<typename LHS, typename RHS>
    struct RatioLess_t<LHS, RHS, -1LL, -1LL>
    {
        static const bool Value = RatioLess<Ratio<-RHS::Num, RHS::Den>, Ratio<-LHS::Num, LHS::Den>>::Value;
    };
}

template<typename LHS, typename RHS>
struct RatioLess : std::bool_constant<(Internal::RatioLess_t<LHS, RHS>::Value)>
{
};

template<typename LHS, typename RHS>
struct RatioLessEqual : std::bool_constant<(!RatioLess<RHS, LHS>::Value)>
{
};

template<typename LHS, typename RHS>
struct RatioGreater : std::bool_constant<(RatioLess<RHS, LHS>::Value)>
{
};

template<typename LHS, typename RHS>
struct RatioGreaterEqual : std::bool_constant<(!RatioLess<LHS, RHS>::Value)>
{
};

template<typename LHS, typename RHS>
struct RatioGreatestCommonDivisor
{
    typedef Ratio<Internal::GreatestCommonDivisor<LHS::Num, RHS::Num>::Value, Internal::LeastCommonMultiple<LHS::Den, RHS::Den>::Value> Type;
};

template<typename LHS, typename RHS>
constexpr bool RatioEqual_v = RatioEqual<LHS, RHS>::Value;

template<typename LHS, typename RHS>
constexpr bool RatioNotEqual_v = RatioNotEqual<LHS, RHS>::Value;

template<typename LHS, typename RHS>
constexpr bool RatioLess_v = RatioLess<LHS, RHS>::Value;

template<typename LHS, typename RHS>
constexpr bool RatioLessEqual_v = RatioLessEqual<LHS, RHS>::Value;

template<typename LHS, typename RHS>
constexpr bool RatioGreater_v = RatioGreater<LHS, RHS>::Value;

template<typename LHS, typename RHS>
constexpr bool RatioGreaterEqual_v = RatioGreaterEqual<LHS, RHS>::Value;

template<typename T>
struct is_ratio_helper
{
private:
    static T* create();

    template<int64 Numerator, int64 Denominator>
    static std::true_type test(const Ratio<Numerator, Denominator>*);

    template<int64 Numerator, int64 Denominator>
    static std::true_type test(const volatile Ratio<Numerator, Denominator>*);

    static std::false_type test(...);

public:
    using Type = decltype(test(create()));
};

template<typename T>
struct is_ratio : is_ratio_helper<T>::Type
{
};

// template<typename T>
// struct is_ratio : std::false_type
//{
//};

template<int64 Numerator, int64 Denominator>
struct is_ratio<Ratio<Numerator, Denominator>> : std::true_type
{
};

template<typename T>
struct is_ratio<T&> : std::false_type
{
};

template<typename T>
static constexpr bool is_ratio_v = is_ratio<T>::value;

template<int64 N1, int64 D1, int64 N2, int64 D2>
static constexpr auto operator+(Ratio<N1, D1>, Ratio<N2, D2>)
{
    return RatioAdd<Ratio<N1, D1>, Ratio<N2, D2>>();
}

template<int64 N1, int64 D1, int64 N2, int64 D2>
static constexpr auto operator-(Ratio<N1, D1>, Ratio<N2, D2>)
{
    return RatioSubtract<Ratio<N1, D1>, Ratio<N2, D2>>();
}

template<int64 N1, int64 D1, int64 N2, int64 D2>
static constexpr auto operator*(Ratio<N1, D1>, Ratio<N2, D2>)
{
    return RatioMultiply<Ratio<N1, D1>, Ratio<N2, D2>>();
}

template<int64 N1, int64 D1, int64 N2, int64 D2>
static constexpr auto operator/(Ratio<N1, D1>, Ratio<N2, D2>)
{
    return RatioDivide<Ratio<N1, D1>, Ratio<N2, D2>>();
}

template<int64 N1, int64 D1>
static constexpr auto operator-(Ratio<N1, D1>)
{
    return Ratio<-N1, D1>();
}

template<int64 N1, int64 D1, int64 N>
static constexpr auto operator^(Ratio<N1, D1>, int64)
{
    return RatioPow<Ratio<N1, D1>, N>();
}

template<int64 N1, int64 D1, int64 N>
static constexpr auto pow(Ratio<N1, D1>, int64)
{
    return RatioPow<Ratio<N1, D1>, N>();
}

template<int64 N1, int64 D1, int64 N2, int64 D2>
static constexpr auto operator==(Ratio<N1, D1>, Ratio<N2, D2>)
{
    return RatioEqual<Ratio<N1, D1>, Ratio<N2, D2>>();
}

template<int64 N1, int64 D1, int64 N2, int64 D2>
static constexpr auto operator!=(Ratio<N1, D1>, Ratio<N2, D2>)
{
    return RatioNotEqual<Ratio<N1, D1>, Ratio<N2, D2>>();
}

template<int64 N1, int64 D1, int64 N2, int64 D2>
static constexpr auto operator<(Ratio<N1, D1>, Ratio<N2, D2>)
{
    return RatioLess<Ratio<N1, D1>, Ratio<N2, D2>>();
}

template<int64 N1, int64 D1, int64 N2, int64 D2>
static constexpr auto operator<=(Ratio<N1, D1>, Ratio<N2, D2>)
{
    return RatioLessEqual<Ratio<N1, D1>, Ratio<N2, D2>>();
}

template<int64 N1, int64 D1, int64 N2, int64 D2>
static constexpr auto operator>(Ratio<N1, D1>, Ratio<N2, D2>)
{
    return RatioGreater<Ratio<N1, D1>, Ratio<N2, D2>>();
}

template<int64 N1, int64 D1, int64 N2, int64 D2>
static constexpr auto operator>=(Ratio<N1, D1>, Ratio<N2, D2>)
{
    return RatioGreaterEqual<Ratio<N1, D1>, Ratio<N2, D2>>();
}

// static constexpr auto ToFraction(const float value)
//{
//    const int64 sign = Internal::sign(value);
//    const int64 inValue = Internal::abs(value);
//
//    return ConvertPositiveDouble(sign, inValue);
//}

// template<typename ExprT>
// KOKKOS_FORCEINLINE_FUNCTION static auto get_view(VectorExpr<ExprT>& expr) -> std::enable_if_t<is_dense_vector_v<ExprT>, KokkosViewVector<ElementType_t<ExprT>>&>
//{
//    return get_view(~expr);
//}
