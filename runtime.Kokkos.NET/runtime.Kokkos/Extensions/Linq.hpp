#pragma once

#if !defined(KOKKOS_EXTENSIONS)
#    error "Do not include directly. Include Extensions.hpp"
#endif

#if !defined(MATH_EXTENSIONS)
#    include <MathExtensions.hpp>
#endif

#include <Concepts.hpp>

namespace Kokkos
{
    namespace Extension
    {

        template<System::EquatableType DataType, UnsignedInteger Index, class ExecutionSpace, class LayoutType = typename ExecutionSpace::array_layout>
        struct ContainsFunctor
        {
            Kokkos::View<DataType*, LayoutType, ExecutionSpace> values;

            const DataType value_to_find;

            Kokkos::View<bool, LayoutType, ExecutionSpace> value_found;

            ContainsFunctor(const Kokkos::View<DataType*, LayoutType, ExecutionSpace>& values, const DataType value_to_find) : values(values), value_to_find(value_to_find), value_found("value_found")
            {
                value_found() = false;
            }

            KOKKOS_INLINE_FUNCTION void operator()(const Index& i) const
            {
                if (value_to_find == values(i))
                {
                    Kokkos::atomic_exchange(&value_found(), true);
                }
            }
        };

        template<Primitive DataType, UnsignedInteger Index, class ExecutionSpace, class LayoutType = typename ExecutionSpace::array_layout>
        struct ContainsPrimitiveFunctor
        {
            Kokkos::View<DataType*, LayoutType, ExecutionSpace> values;

            const DataType value_to_find;

            Kokkos::View<bool, LayoutType, ExecutionSpace> value_found;

            ContainsPrimitiveFunctor(const Kokkos::View<DataType*, LayoutType, ExecutionSpace>& values, const DataType value_to_find) : values(values), value_to_find(value_to_find), value_found("value_found")
            {
                value_found() = false;
            }

            KOKKOS_INLINE_FUNCTION void operator()(const Index& i) const
            {
                if (EqualTo<DataType>(value_to_find, values(i)))
                {
                    Kokkos::atomic_exchange(&value_found(), true);
                }
            }
        };

        template<typename DataType, UnsignedInteger Index, class ExecutionSpace, class LayoutType = typename ExecutionSpace::array_layout>
        __inline static auto Contains(const Kokkos::View<DataType*, LayoutType, ExecutionSpace>& values, const DataType value_to_find) -> std::enable_if_t<Primitive<DataType>, bool>
        {
            const Kokkos::RangePolicy<ExecutionSpace> range(0, values.extent(0));

            ContainsPrimitiveFunctor<DataType, Index, ExecutionSpace> functor(values, value_to_find);

            Kokkos::parallel_for(range, functor);

            return functor.value_found();
        }

        template<System::EquatableType DataType, UnsignedInteger Index, class ExecutionSpace, class LayoutType = typename ExecutionSpace::array_layout>
        __inline static auto Contains(const Kokkos::View<DataType*, LayoutType, ExecutionSpace>& values, const DataType value_to_find) -> std::enable_if_t<NotPrimitive<DataType>, bool>
        {
            const Kokkos::RangePolicy<ExecutionSpace> range(0, values.extent(0));

            ContainsFunctor<DataType, Index, ExecutionSpace> functor(values, value_to_find);

            Kokkos::parallel_for(range, functor);

            return functor.value_found();
        }
    }

}
