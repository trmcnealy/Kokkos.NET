#pragma once

#if !defined(KOKKOS_EXTENSIONS)
#    error "Do not include directly. Include Extensions.hpp"
#endif

namespace Kokkos
{
    namespace Experimental
    {
        struct ScatterResidual
        {
        };

    }
}
namespace Kokkos
{
    namespace Impl
    {
        namespace Experimental
        {
            template<typename ValueType, typename DeviceType>
            struct ScatterValue<ValueType, Kokkos::Experimental::ScatterResidual, DeviceType, Kokkos::Experimental::ScatterNonAtomic>
            {
                ValueType& value;

            public:
                KOKKOS_FORCEINLINE_FUNCTION      ScatterValue(ValueType& value_in) : value(value_in) {}
                KOKKOS_FORCEINLINE_FUNCTION      ScatterValue(ScatterValue&& other) noexcept : value(other.value) {}
                KOKKOS_FORCEINLINE_FUNCTION void operator*=(ValueType const& rhs)
                {
                    value *= rhs;
                }
                KOKKOS_FORCEINLINE_FUNCTION void operator/=(ValueType const& rhs)
                {
                    value /= rhs;
                }

                KOKKOS_FORCEINLINE_FUNCTION void update(ValueType const& rhs)
                {
                    value *= rhs;
                }
                KOKKOS_FORCEINLINE_FUNCTION void reset()
                {
                    value = reduction_identity<ValueType>::prod();
                }
            };
        }
    }
}

namespace Kokkos
{
    namespace Impl
    {
        namespace Experimental
        {
            template<typename ValueType, typename DeviceType>
            struct ScatterValue<ValueType, Kokkos::Experimental::ScatterResidual, DeviceType, Kokkos::Experimental::ScatterAtomic>
            {
                ValueType& value;

            public:
                KOKKOS_FORCEINLINE_FUNCTION ScatterValue(ValueType& value_in) : value(value_in) {}
                KOKKOS_FORCEINLINE_FUNCTION ScatterValue(ScatterValue&& other) noexcept : value(other.value) {}

                KOKKOS_FORCEINLINE_FUNCTION void operator*=(ValueType const& rhs)
                {
                    Kokkos::atomic_mul(&value, rhs);
                }
                KOKKOS_FORCEINLINE_FUNCTION void operator/=(ValueType const& rhs)
                {
                    Kokkos::atomic_div(&value, rhs);
                }

                KOKKOS_FORCEINLINE_FUNCTION
                void atomic_prod(ValueType& dest, const ValueType& src) const
                {
                    bool success = false;
                    while (!success)
                    {
                        ValueType dest_old = dest;
                        ValueType dest_new = dest_old * src;
                        dest_new           = Kokkos::atomic_compare_exchange<ValueType>(&dest, dest_old, dest_new);
                        success            = ((dest_new - dest_old) / dest_old <= 1e-15);
                    }
                }

                KOKKOS_INLINE_FUNCTION
                void join(ValueType& dest, const ValueType& src) const
                {
                    atomic_prod(&dest, src);
                }

                KOKKOS_INLINE_FUNCTION
                void join(volatile ValueType& dest, const volatile ValueType& src) const
                {
                    atomic_prod(&dest, src);
                }

                KOKKOS_FORCEINLINE_FUNCTION void update(ValueType const& rhs)
                {
                    atomic_prod(&value, rhs);
                }
                KOKKOS_FORCEINLINE_FUNCTION void reset()
                {
                    value = reduction_identity<ValueType>::prod();
                }
            };
        }
    }
}

namespace Kokkos
{
    namespace Extension
    {
        namespace Internal
        {
            template<typename DataType, class ExecutionSpace, int Rank>
            struct IndexOfMinimum;

            template<typename DataType, class ExecutionSpace>
            struct IndexOfMinimum<DataType, ExecutionSpace, 1>
            {
                using ViewType  = View<DataType*, typename ExecutionSpace::array_layout, ExecutionSpace>;
                using ValueType = typename ViewType::traits::non_const_value_type;

                static size_type Find(const ViewType& view)
                {
                    size_type index     = -1;
                    ValueType min_value = Constants<ValueType>::Max();

                    for (size_type i = 0; i < view.extent(0); ++i)
                    {
                        if (view(i) < min_value)
                        {
                            min_value = view(i);
                            index     = i;
                        }
                    }

                    return index;
                }
            };

            template<typename DataType, class ExecutionSpace>
            struct IndexOfMinimum<DataType, ExecutionSpace, 2>
            {
                using ViewType  = View<DataType**, typename ExecutionSpace::array_layout, ExecutionSpace>;
                using ValueType = typename ViewType::traits::non_const_value_type;

                static size_type Find(const ViewType& view)
                {
                    size_type index     = -1;
                    ValueType min_value = Constants<ValueType>::Max();

                    for (size_type i = 0; i < view.extent(0); ++i)
                    {
                        for (size_type j = 0; j < view.extent(1); ++j)
                        {
                            if (view(i, j) < min_value)
                            {
                                min_value = view(i, j);
                                index     = i;
                            }
                        }
                    }

                    return index;
                }
            };

            template<typename LayoutT>
            struct TransposeLayout;

            template<>
            struct TransposeLayout<Kokkos::LayoutLeft>
            {
                using type = Kokkos::LayoutRight;
            };

            template<>
            struct TransposeLayout<Kokkos::LayoutRight>
            {
                using type = Kokkos::LayoutLeft;
            };
        }

        template<typename ViewType>
        __inline static constexpr size_type IndexOfMin(const ViewType& view)
        {
            using ExecutionSpace = typename ViewType::traits::execution_space;
            using DataType       = typename ViewType::traits::data_type;
            using ValueType      = typename ViewType::traits::non_const_value_type;

            return Internal::IndexOfMinimum<DataType, ExecutionSpace, ViewType::Rank>::Find(view);
        }
    }
}

namespace Kokkos::Extension
{
    template<typename KeyType, typename DataType, class ExecutionSpace>
    using UnorderedMap = UnorderedMap<KeyType, DataType, ExecutionSpace>;

    // template<typename DataType>
    // using ArithTraits = Kokkos::Details::ArithTraits<DataType>;
}

namespace Kokkos
{
    namespace Extension
    {

        //       //Calculates the minimum value in the given data set.
        //[MethodImpl(MethodImplOptions.AggressiveInlining)]
        // public static double MinValue(double[] data)
        //{
        //    double minimum = data[0];
        //    double d;

        //    for (int i = 1; i < data.Length; i++)
        //    {
        //        d = data[i];
        //        if (d < minimum)
        //        {
        //            minimum = d;
        //        }
        //    }
        //    return minimum;
        //}

        ////Calculates the minimum absolute value in the given data set.
        //[MethodImpl(MethodImplOptions.AggressiveInlining)]
        // public static double MinAbsValue(double[] data)
        //{
        //    double minimum = Math.Abs(data[0]);
        //    double d;

        //    for (int i = 1; i < data.Length; i++)
        //    {
        //        d = Math.Abs(data[i]);
        //        if (d < minimum)
        //        {
        //            minimum = d;
        //        }
        //    }
        //    return minimum;
        //}

        ////Calculates the index of the minimum value in the given data set.
        //[MethodImpl(MethodImplOptions.AggressiveInlining)]
        // public static int MinIndex(double[] data)
        //{
        //    double minimum = Double.MaxValue;
        //    int index = -1;
        //    double d;

        //    for (int i = 0; i < data.Length; i++)
        //    {
        //        d = data[i];
        //        if (d < minimum)
        //        {
        //            index = i;
        //            minimum = d;
        //        }
        //    }
        //    return index;
        //}

        ////Calculates the index of the minimum absolute value in the given data set.
        //[MethodImpl(MethodImplOptions.AggressiveInlining)]
        // public static int MinAbsIndex(double[] data)
        //{
        //    double minimum = Double.MaxValue;
        //    int index = -1;
        //    double d;

        //    for (int i = 0; i < data.Length; i++)
        //    {
        //        d = Math.Abs(data[i]);
        //        if (d < minimum)
        //        {
        //            index = i;
        //            minimum = d;
        //        }
        //    }
        //    return index;
        //}

        ////Calculates the maximum value in the given data set.
        //[MethodImpl(MethodImplOptions.AggressiveInlining)]
        // public static double MaxValue(double[] data)
        //{
        //    double maximum = data[0];
        //    double d;

        //    for (int i = 1; i < data.Length; i++)
        //    {
        //        d = data[i];
        //        if (d > maximum)
        //        {
        //            maximum = d;
        //        }
        //    }
        //    return maximum;
        //}

        ////Calculates the maximum absolute value in the given data set.
        //[MethodImpl(MethodImplOptions.AggressiveInlining)]
        // public static double MaxAbsValue(double[] data)
        //{
        //    double maximum = Math.Abs(data[0]);
        //    double d;

        //    for (int i = 1; i < data.Length; i++)
        //    {
        //        d = Math.Abs(data[i]);
        //        if (d > maximum)
        //        {
        //            maximum = d;
        //        }
        //    }
        //    return maximum;
        //}

        ////Calculates the index of the maximum value in the given data set.
        //[MethodImpl(MethodImplOptions.AggressiveInlining)]
        // public static int MaxIndex(double[] data)
        //{
        //    double maximum = Double.MinValue;
        //    int index = -1;
        //    double d;

        //    for (int i = 0; i < data.Length; i++)
        //    {
        //        d = data[i];
        //        if (d > maximum)
        //        {
        //            index = i;
        //            maximum = d;
        //        }
        //    }
        //    return index;
        //}

        ////Calculates the index of the maximum absolute value in the given data set.
        //[MethodImpl(MethodImplOptions.AggressiveInlining)]
        // public static int MaxAbsIndex(double[] data)
        //{
        //    double maximum = Double.MinValue;
        //    int index = -1;
        //    double d;

        //    for (int i = 0; i < data.Length; i++)
        //    {
        //        d = Math.Abs(data[i]);
        //        if (d > maximum)
        //        {
        //            index = i;
        //            maximum = d;
        //        }
        //    }
        //    return index;
        //}
    }
}

#if 0
        template<typename DataType>
        static int CompareTo(REF(DataType) lhs, REF(DataType) rhs)
        {
            if (lhs == nullptr)
            {
                return (rhs == nullptr) ? 0 : -1;
            }
            else if (rhs == nullptr)
            {
                return 1;
            }

            if (lhs < rhs)
            {
                return -1;
            }

            if (lhs > rhs)
            {
                return 1;
            }

            return 0;
        }

        template<typename DataType, class ExecutionSpace>
        static int BinarySearch(const View<DataType*, typename ExecutionSpace::array_layout, ExecutionSpace>& view, REF(DataType) comparable)
        {
            const int length = view.extent(0);

            int32 lo = 0;
            int32 hi = length - 1;

            int32 i = 0;
            int32 c;

            while (lo <= hi)
            {
                i += (int32)((hi + lo) >> 1);

                c = CompareTo<DataType>(view(i), comparable);

                if (c == 0)
                {
                    return i;
                }

                if (c > 0)
                {
                    lo = i + 1;
                }
                else
                {
                    hi = i - 1;
                }
            }

            return ~lo;
        }

        template<typename DataType, class ExecutionSpace>
        static int BinarySearch(const View<DataType**, typename ExecutionSpace::array_layout, ExecutionSpace>& data, uint8 dimensionToSearch, REF(DataType) comparable)
        {
            const int length = data.extent(dimensionToSearch);

            uint32 lo = 0;
            uint32 hi = length - 1;

            int32 i = 0;
            int32 c;

            auto dimensionView = Kokkos::subview(data, (dimensionToSearch == 0) ? Kokkos::ALL : 0, (dimensionToSearch == 1) ? Kokkos::ALL : 0);

            while (lo <= hi)
            {
                i += (int32)((hi + lo) >> 1);

                c = CompareTo<DataType>(dimensionView(i), comparable);

                if (c == 0)
                {
                    return i;
                }

                if (c > 0)
                {
                    lo = i + 1;
                }
                else
                {
                    hi = i - 1;
                }
            }

            return ~lo;
        }

        template<typename DataType, class ExecutionSpace>
        static int BinarySearch(const View<DataType***, typename ExecutionSpace::array_layout, ExecutionSpace>& data, uint8 dimensionToSearch, REF(DataType) comparable)
        {
            const int length = data.extent(dimensionToSearch);

            uint32 lo = 0;
            uint32 hi = length - 1;

            int32 i = 0;
            int32 c;

            auto dimensionView = Kokkos::subview(data,
                                                 (dimensionToSearch == 0) ? Kokkos::ALL : 0,
                                                 (dimensionToSearch == 1) ? Kokkos::ALL : 0,
                                                 (dimensionToSearch == 2) ? Kokkos::ALL : 0);

            while (lo <= hi)
            {
                i += (int32)((hi + lo) >> 1);

                c = CompareTo<DataType>(dimensionView(i), comparable);

                if (c == 0)
                {
                    return i;
                }

                if (c > 0)
                {
                    lo = i + 1;
                }
                else
                {
                    hi = i - 1;
                }
            }

            return ~lo;
        }

        template<typename DataType, class ExecutionSpace>
        static int BinarySearch(const View<DataType****, typename ExecutionSpace::array_layout, ExecutionSpace>& data, uint8 dimensionToSearch, REF(DataType) comparable)
        {
            const int length = data.extent(dimensionToSearch);

            uint32 lo = 0;
            uint32 hi = length - 1;

            int32 i = 0;
            int32 c;

            auto dimensionView = Kokkos::subview(data,
                                                 (dimensionToSearch == 0) ? Kokkos::ALL : 0,
                                                 (dimensionToSearch == 1) ? Kokkos::ALL : 0,
                                                 (dimensionToSearch == 2) ? Kokkos::ALL : 0,
                                                 (dimensionToSearch == 3) ? Kokkos::ALL : 0);

            while (lo <= hi)
            {
                i += (int32)((hi + lo) >> 1);

                c = CompareTo<DataType>(dimensionView(i), comparable);

                if (c == 0)
                {
                    return i;
                }

                if (c > 0)
                {
                    lo = i + 1;
                }
                else
                {
                    hi = i - 1;
                }
            }

            return ~lo;
        }

        template<typename DataType, class ExecutionSpace>
        static int BinarySearch(const View<DataType*****, typename ExecutionSpace::array_layout, ExecutionSpace>& data, uint8 dimensionToSearch, REF(DataType) comparable)
        {
            const int length = data.extent(dimensionToSearch);

            uint32 lo = 0;
            uint32 hi = length - 1;

            int32 i = 0;
            int32 c;

            auto dimensionView = Kokkos::subview(data,
                                                 (dimensionToSearch == 0) ? Kokkos::ALL : 0,
                                                 (dimensionToSearch == 1) ? Kokkos::ALL : 0,
                                                 (dimensionToSearch == 2) ? Kokkos::ALL : 0,
                                                 (dimensionToSearch == 3) ? Kokkos::ALL : 0,
                                                 (dimensionToSearch == 4) ? Kokkos::ALL : 0);

            while (lo <= hi)
            {
                i += (int32)((hi + lo) >> 1);

                c = CompareTo<DataType>(dimensionView(i), comparable);

                if (c == 0)
                {
                    return i;
                }

                if (c > 0)
                {
                    lo = i + 1;
                }
                else
                {
                    hi = i - 1;
                }
            }

            return ~lo;
        }

        template<typename DataType, class ExecutionSpace>
        static int BinarySearch(const View<DataType******, typename ExecutionSpace::array_layout, ExecutionSpace>& data, uint8 dimensionToSearch, REF(DataType) comparable)
        {
            const int length = data.extent(dimensionToSearch);

            uint32 lo = 0;
            uint32 hi = length - 1;

            int32 i = 0;
            int32 c;

            auto dimensionView = Kokkos::subview(data,
                                                 (dimensionToSearch == 0) ? Kokkos::ALL : 0,
                                                 (dimensionToSearch == 1) ? Kokkos::ALL : 0,
                                                 (dimensionToSearch == 2) ? Kokkos::ALL : 0,
                                                 (dimensionToSearch == 3) ? Kokkos::ALL : 0,
                                                 (dimensionToSearch == 4) ? Kokkos::ALL : 0,
                                                 (dimensionToSearch == 5) ? Kokkos::ALL : 0);

            while (lo <= hi)
            {
                i += (int32)((hi + lo) >> 1);

                c = CompareTo<DataType>(dimensionView(i), comparable);

                if (c == 0)
                {
                    return i;
                }

                if (c > 0)
                {
                    lo = i + 1;
                }
                else
                {
                    hi = i - 1;
                }
            }

            return ~lo;
        }

        template<typename DataType, class ExecutionSpace>
        static int BinarySearch(const View<DataType*******, typename ExecutionSpace::array_layout, ExecutionSpace>& data, uint8 dimensionToSearch, REF(DataType) comparable)
        {
            const int length = data.extent(dimensionToSearch);

            uint32 lo = 0;
            uint32 hi = length - 1;

            int32 i = 0;
            int32 c;

            auto dimensionView = Kokkos::subview(data,
                                                 (dimensionToSearch == 0) ? Kokkos::ALL : 0,
                                                 (dimensionToSearch == 1) ? Kokkos::ALL : 0,
                                                 (dimensionToSearch == 2) ? Kokkos::ALL : 0,
                                                 (dimensionToSearch == 3) ? Kokkos::ALL : 0,
                                                 (dimensionToSearch == 4) ? Kokkos::ALL : 0,
                                                 (dimensionToSearch == 5) ? Kokkos::ALL : 0,
                                                 (dimensionToSearch == 6) ? Kokkos::ALL : 0);

            while (lo <= hi)
            {
                i += (int32)((hi + lo) >> 1);

                c = CompareTo<DataType>(dimensionView(i), comparable);

                if (c == 0)
                {
                    return i;
                }

                if (c > 0)
                {
                    lo = i + 1;
                }
                else
                {
                    hi = i - 1;
                }
            }

            return ~lo;
        }
#endif
