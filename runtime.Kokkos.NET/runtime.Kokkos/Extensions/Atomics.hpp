#pragma once

#if !defined(KOKKOS_EXTENSIONS)
#    error "Do not include directly. Include Extensions.hpp"
#endif

namespace Kokkos
{
    namespace Impl
    {
        template<class Scalar1, class Scalar2>
        struct LessThanOper
        {
            KOKKOS_FORCEINLINE_FUNCTION static bool apply(const Scalar1& val1, const Scalar2& val2)
            {
                return (val1 < val2);
            }
        };

        template<class Scalar1, class Scalar2>
        struct LessThanEqualToOper
        {
            KOKKOS_FORCEINLINE_FUNCTION static bool apply(const Scalar1& val1, const Scalar2& val2)
            {
                return (val1 <= val2);
            }
        };

        template<class Scalar1, class Scalar2>
        struct GreaterThanOper
        {
            KOKKOS_FORCEINLINE_FUNCTION static bool apply(const Scalar1& val1, const Scalar2& val2)
            {
                return (val1 > val2);
            }
        };

        template<class Scalar1, class Scalar2>
        struct GreaterThanEqualToOper
        {
            KOKKOS_FORCEINLINE_FUNCTION static bool apply(const Scalar1& val1, const Scalar2& val2)
            {
                return (val1 >= val2);
            }
        };

        template<class Scalar1, class Scalar2>
        struct EqualToOper
        {
            KOKKOS_FORCEINLINE_FUNCTION static bool apply(const Scalar1& val1, const Scalar2& val2)
            {
                return (val1 == val2);
            }
        };

        template<class Scalar1, class Scalar2>
        struct NotEqualToOper
        {
            KOKKOS_FORCEINLINE_FUNCTION static bool apply(const Scalar1& val1, const Scalar2& val2)
            {
                return (val1 != val2);
            }
        };

        template<class Oper, typename T>
        KOKKOS_INLINE_FUNCTION bool atomic_boolean_fetch(const Oper&                                                                                                     op,
                                                         volatile T* const                                                                                               dest,
                                                         typename std::enable_if<sizeof(T) != sizeof(int) && sizeof(T) == sizeof(unsigned long long int), const T>::type val)
        {
            union U
            {
                unsigned long long int i;
                T                      t;
                KOKKOS_INLINE_FUNCTION U() {}
            } oldval, newval;

            oldval.t = *dest;

            bool result;

            do
            {
                result = op.apply(oldval.t, val);

                if (result)
                {
                    newval.t = val;
                    oldval.i = Kokkos::atomic_exchange((unsigned long long int*)dest, newval.i);
                }

            } while (result && newval.i != oldval.i);

            return result;
        }

        template<class Oper, typename T>
        KOKKOS_INLINE_FUNCTION T atomic_boolean_fetch(const Oper& op, volatile T* const dest, typename std::enable_if<sizeof(T) == sizeof(int), const T>::type val)
        {
            union U
            {
                int                    i;
                T                      t;
                KOKKOS_INLINE_FUNCTION U() {}
            } oldval, newval;

            oldval.t = *dest;

            bool result;

            do
            {
                result = op.apply(oldval.t, val);

                if (result)
                {
                    newval.t = val;
                    oldval.i = Kokkos::atomic_exchange((int*)dest, newval.i);
                }

            } while (result && newval.i != oldval.i);

            return result;
        }

        template<class Oper, typename T>
        KOKKOS_INLINE_FUNCTION T atomic_boolean_fetch(const Oper&       op,
                                                      volatile T* const dest,
                                                      typename std::enable_if<(sizeof(T) != 4) && (sizeof(T) != 8)
#if defined(KOKKOS_ENABLE_ASM) && defined(KOKKOS_ACTIVE_EXECUTION_MEMORY_SPACE_HOST)
                                                                                  && (sizeof(T) != 16)
#endif
                                                                                  ,
                                                                              const T>::type& val)
        {
            bool result;

#ifdef KOKKOS_ACTIVE_EXECUTION_MEMORY_SPACE_HOST
            while (!Impl::lock_address_host_space((void*)dest))
                ;
            Kokkos::memory_fence();

            result = op.apply(*dest, val);

            if (result)
            {
                *dest = val;
            }

            Kokkos::memory_fence();
            Impl::unlock_address_host_space((void*)dest);
            return result;
#elif defined(KOKKOS_ACTIVE_EXECUTION_MEMORY_SPACE_CUDA)
            // This is a way to (hopefully) avoid dead lock in a warp
            int done = 0;
#    ifdef KOKKOS_IMPL_CUDA_SYNCWARP_NEEDS_MASK
            unsigned int mask   = KOKKOS_IMPL_CUDA_ACTIVEMASK;
            unsigned int active = KOKKOS_IMPL_CUDA_BALLOT_MASK(mask, 1);
#    else
            unsigned int active = KOKKOS_IMPL_CUDA_BALLOT(1);
#    endif
            unsigned int done_active = 0;
            do
            {
                if (!done)
                {
                    if (Impl::lock_address_cuda_space((void*)dest))
                    {
                        Kokkos::memory_fence();
                        result = op.apply(*dest, val);
                        if (result)
                        {
                            *dest = val;
                        }
                        Kokkos::memory_fence();
                        Impl::unlock_address_cuda_space((void*)dest);
                        done = 1;
                    }
                }
#    ifdef KOKKOS_IMPL_CUDA_SYNCWARP_NEEDS_MASK
                done_active = KOKKOS_IMPL_CUDA_BALLOT_MASK(mask, done);
#    else
                done_active = KOKKOS_IMPL_CUDA_BALLOT(done);
#    endif
            } while (result && active != done_active);
            return result;
#elif defined(__HIP_DEVICE_COMPILE__)
            // FIXME_HIP
            Kokkos::abort("atomic_oper_fetch not implemented for large types.");
            int          done        = 0;
            unsigned int active      = __ballot(1);
            unsigned int done_active = 0;
            do
            {
                if (!done)
                {
                    // if (Impl::lock_address_hip_space((void*)dest))
                    {
                        result = op.apply(*dest, val);
                        if (result)
                        {
                            *dest = val;
                        }
                        // Impl::unlock_address_hip_space((void*)dest);
                        done = 1;
                    }
                }
                done_active = __ballot(done);
            } while (result && active != done_active);
            return result;
#endif
        }
    }

    template<typename T>
    KOKKOS_INLINE_FUNCTION bool atomic_less_than_fetch(volatile T* const dest, const T val)
    {
        return Impl::atomic_boolean_fetch(Impl::LessThanOper<T, const T>(), dest, val);
    }

    template<typename T>
    KOKKOS_INLINE_FUNCTION bool atomic_less_than_equal_fetch(volatile T* const dest, const T val)
    {
        return Impl::atomic_boolean_fetch(Impl::LessThanEqualToOper<T, const T>(), dest, val);
    }

    template<typename T>
    KOKKOS_INLINE_FUNCTION bool atomic_greater_than_fetch(volatile T* const dest, const T val)
    {
        return Impl::atomic_boolean_fetch(Impl::GreaterThanOper<T, const T>(), dest, val);
    }

    template<typename T>
    KOKKOS_INLINE_FUNCTION bool atomic_greater_than_equal_fetch(volatile T* const dest, const T val)
    {
        return Impl::atomic_boolean_fetch(Impl::GreaterThanEqualToOper<T, const T>(), dest, val);
    }

    template<typename T>
    KOKKOS_INLINE_FUNCTION bool atomic_equal_to_fetch(volatile T* const dest, const T val)
    {
        return Impl::atomic_boolean_fetch(Impl::EqualToOper<T, const T>(), dest, val);
    }

    template<typename T>
    KOKKOS_INLINE_FUNCTION bool atomic_not_equal_to_fetch(volatile T* const dest, const T val)
    {
        return Impl::atomic_boolean_fetch(Impl::NotEqualToOper<T, const T>(), dest, val);
    }
}
