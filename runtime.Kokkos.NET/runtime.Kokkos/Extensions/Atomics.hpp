#pragma once

#if !defined(KOKKOS_EXTENSIONS)
#    error "Do not include directly. Include Extensions.hpp"
#endif

namespace Kokkos
{
    namespace Impl
    {
        //template<class Scalar1, class Scalar2>
        //struct LessThanOper
        //{
        //    KOKKOS_FORCEINLINE_FUNCTION static bool apply(const Scalar1& val1, const Scalar2& val2)
        //    {
        //        return (val1 < val2);
        //    }

        //    KOKKOS_FORCEINLINE_FUNCTION static bool apply(const volatile Scalar1& val1, const Scalar2& val2)
        //    {
        //        return (val1 < val2);
        //    }
        //};

        //template<class Scalar1, class Scalar2>
        //struct LessThanEqualToOper
        //{
        //    KOKKOS_FORCEINLINE_FUNCTION static bool apply(const Scalar1& val1, const Scalar2& val2)
        //    {
        //        return (val1 <= val2);
        //    }

        //    KOKKOS_FORCEINLINE_FUNCTION static bool apply(const volatile Scalar1& val1, const Scalar2& val2)
        //    {
        //        return (val1 <= val2);
        //    }
        //};

        //template<class Scalar1, class Scalar2>
        //struct GreaterThanOper
        //{
        //    KOKKOS_FORCEINLINE_FUNCTION static bool apply(const Scalar1& val1, const Scalar2& val2)
        //    {
        //        return (val1 > val2);
        //    }

        //    KOKKOS_FORCEINLINE_FUNCTION static bool apply(const volatile Scalar1& val1, const Scalar2& val2)
        //    {
        //        return (val1 > val2);
        //    }
        //};

        //template<class Scalar1, class Scalar2>
        //struct GreaterThanEqualToOper
        //{
        //    KOKKOS_FORCEINLINE_FUNCTION static bool apply(const Scalar1& val1, const Scalar2& val2)
        //    {
        //        return (val1 >= val2);
        //    }

        //    KOKKOS_FORCEINLINE_FUNCTION static bool apply(const volatile Scalar1& val1, const Scalar2& val2)
        //    {
        //        return (val1 >= val2);
        //    }
        //};

        //template<class Scalar1, class Scalar2>
        //struct EqualToOper
        //{
        //    KOKKOS_FORCEINLINE_FUNCTION static bool apply(const Scalar1& val1, const Scalar2& val2)
        //    {
        //        return (val1 == val2);
        //    }

        //    KOKKOS_FORCEINLINE_FUNCTION static bool apply(const volatile Scalar1& val1, const Scalar2& val2)
        //    {
        //        return (val1 == val2);
        //    }
        //};

        //template<class Scalar1, class Scalar2>
        //struct NotEqualToOper
        //{
        //    KOKKOS_FORCEINLINE_FUNCTION static bool apply(const Scalar1& val1, const Scalar2& val2)
        //    {
        //        return (val1 != val2);
        //    }

        //    KOKKOS_FORCEINLINE_FUNCTION static bool apply(const volatile Scalar1& val1, const Scalar2& val2)
        //    {
        //        return (val1 != val2);
        //    }
        //};

        //template<class Oper, typename T>
        //KOKKOS_INLINE_FUNCTION bool atomic_comparison_oper(const Oper& op, volatile T* const dest, const T val)
        //{
        //    Kokkos::store_fence();

        //    uint32_t i = 0;

        //    while (value == flag)
        //    {
        //        host_thread_yield(++i, WaitMode::ROOT);
        //    }

        //    Kokkos::load_fence();
        //}
    }

    //template<typename T>
    //KOKKOS_INLINE_FUNCTION bool atomic_less_than_fetch(volatile T* const dest, const T val)
    //{
    //    return Impl::atomic_comparison_oper_fetch(Impl::LessThanOper<T, const T>(), dest, val);
    //}

    //template<typename T>
    //KOKKOS_INLINE_FUNCTION bool atomic_less_than_equal_fetch(volatile T* const dest, const T val)
    //{
    //    return Impl::atomic_comparison_oper_fetch(Impl::LessThanEqualToOper<T, const T>(), dest, val);
    //}

    //template<typename T>
    //KOKKOS_INLINE_FUNCTION bool atomic_greater_than_fetch(volatile T* const dest, const T val)
    //{
    //    return Impl::atomic_comparison_oper_fetch(Impl::GreaterThanOper<T, const T>(), dest, val);
    //}

    //template<typename T>
    //KOKKOS_INLINE_FUNCTION bool atomic_greater_than_equal_fetch(volatile T* const dest, const T val)
    //{
    //    return Impl::atomic_comparison_oper_fetch(Impl::GreaterThanEqualToOper<T, const T>(), dest, val);
    //}

    //template<typename T>
    //KOKKOS_INLINE_FUNCTION bool atomic_equal_to_fetch(volatile T* const dest, const T val)
    //{
    //    return Impl::atomic_comparison_oper_fetch(Impl::EqualToOper<T, const T>(), dest, val);
    //}

    //template<typename T>
    //KOKKOS_INLINE_FUNCTION bool atomic_not_equal_to_fetch(volatile T* const dest, const T val)
    //{
    //    return Impl::atomic_comparison_oper_fetch(Impl::NotEqualToOper<T, const T>(), dest, val);
    //}

}
