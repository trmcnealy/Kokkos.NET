#pragma once

#include <Types.hpp>
#include <StdExtensions.hpp>
#include <Constants.hpp>
#include <Print.hpp>

#include <Kokkos_Core.hpp>

namespace Kokkos
{
    namespace Extension
    {

        template<class ExecutionSpace>
        struct CountLineEndingsFunctor
        {
            using ViewType            = Kokkos::View<wchar_t*, typename ExecutionSpace::array_layout, ExecutionSpace>;
            using ViewIndexType       = Kokkos::View<uint64*, typename ExecutionSpace::array_layout, ExecutionSpace, MemoryTraits<Atomic>>;
            using ViewIndexScalarType = Kokkos::View<uint64, typename ExecutionSpace::array_layout, ExecutionSpace>;
            using ValueType           = typename ViewType::traits::non_const_value_type;

            ViewType            StringView;
            ViewIndexType       LineEndingsView;
            ViewIndexScalarType index;

            const uint64 n;

            CountLineEndingsFunctor(const ViewType& stringView) :
                StringView(stringView),
                LineEndingsView("LineEndings", stringView.extent(0)),
                index("index"),
                n(stringView.extent(0))
            {
            }

            KOKKOS_INLINE_FUNCTION void operator()(const uint64& i) const
            {

                if (StringView(i) == L'\n')// || (StringView(i) == L'\r' && StringView(i + 1) == L'\n'))
                {
                    //printf("%hu\n", (uint16)StringView(i));

                    const uint64 id = Kokkos::atomic_add_fetch(&index(), 1ULL);
                    LineEndingsView(id) = i;
                }
            }
        };

        template<class ExecutionSpace>
        KOKKOS_FORCEINLINE_FUNCTION static View<uint64*, typename ExecutionSpace::array_layout, ExecutionSpace> CountLineEndings(
            const View<wchar_t*, typename ExecutionSpace::array_layout, ExecutionSpace>& stringView)
        {
            CountLineEndingsFunctor<ExecutionSpace> functor(stringView);

            Kokkos::parallel_for(Kokkos::RangePolicy<ExecutionSpace, uint64>(0, stringView.extent(0)), functor);

            Kokkos::View<uint64*, typename ExecutionSpace::array_layout, ExecutionSpace> lineEndingsView = functor.LineEndingsView;

            return lineEndingsView;
        }

    }
}
