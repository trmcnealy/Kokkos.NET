#pragma once

#if !defined(KOKKOS_EXTENSIONS)
#    error "Do not include directly. Include Extensions.hpp"
#endif

#if !defined(MATH_EXTENSIONS)
#    include <MathExtensions.hpp>
#endif

#include <Kokkos_Sort.hpp>

namespace Kokkos
{
    namespace Extension
    {
        template<typename VectorType>
        struct MinFunctor
        {
            typedef typename VectorType::traits::execution_space      ExecutionSpace;
            typedef typename VectorType::traits::data_type            DataType;
            typedef typename VectorType::traits::non_const_value_type ValueType;

            VectorType Values;

            MinFunctor(const VectorType& values) : Values(values) {}

            KOKKOS_INLINE_FUNCTION void operator()(const uint32& i, ValueType& value) const
            {
                if (Values(i) < value)
                {
                    value = Values(i);
                }
            }
        };

        template<typename VectorType>
        KOKKOS_INLINE_FUNCTION static auto min(const VectorType& values) -> typename std::enable_if<VectorType::Rank == 1, typename VectorType::traits::non_const_value_type>::type
        {
            typedef typename VectorType::traits::execution_space      ExecutionSpace;
            typedef typename VectorType::traits::data_type            DataType;
            typedef typename VectorType::traits::non_const_value_type ValueType;

            MinFunctor<VectorType> f(values);

            ValueType              min_value = Constants<ValueType>::Max();
            Kokkos::Min<ValueType> reducer_scalar(min_value);

            Kokkos::parallel_reduce(Kokkos::RangePolicy<ExecutionSpace>(0, values.extent(0)), f, reducer_scalar);
            Kokkos::fence();

            return min_value;
        }

        template<typename VectorType>
        struct MaxFunctor
        {
            typedef typename VectorType::traits::execution_space      ExecutionSpace;
            typedef typename VectorType::traits::data_type            DataType;
            typedef typename VectorType::traits::non_const_value_type ValueType;

            VectorType Values;

            MaxFunctor(const VectorType& values) : Values(values) {}

            KOKKOS_INLINE_FUNCTION void operator()(const uint32& i, ValueType& value) const
            {
                if (Values(i) > value)
                {
                    value = Values(i);
                }
            }
        };

        template<typename VectorType>
        KOKKOS_INLINE_FUNCTION static auto max(const VectorType& values) -> typename std::enable_if<VectorType::Rank == 1, typename VectorType::traits::non_const_value_type>::type
        {
            typedef typename VectorType::traits::execution_space      ExecutionSpace;
            typedef typename VectorType::traits::data_type            DataType;
            typedef typename VectorType::traits::non_const_value_type ValueType;

            MaxFunctor<VectorType> f(values);

            ValueType              max_value = Constants<ValueType>::Min();
            Kokkos::Max<ValueType> reducer_scalar(max_value);

            Kokkos::parallel_reduce(Kokkos::RangePolicy<ExecutionSpace>(0, values.extent(0)), f, reducer_scalar);
            Kokkos::fence();

            return max_value;
        }

        template<typename VectorType>
        struct MinMaxFunctor
        {
            typedef typename VectorType::traits::execution_space      ExecutionSpace;
            typedef typename VectorType::traits::data_type            DataType;
            typedef typename VectorType::traits::non_const_value_type ValueType;
            using minmax_scalar = Kokkos::MinMaxScalar<typename VectorType::non_const_value_type>;

            VectorType Values;

            MinMaxFunctor(const VectorType& values) : Values(values) {}

            KOKKOS_INLINE_FUNCTION void operator()(const uint32& i, minmax_scalar& minmax) const
            {
                if (Values(i) < minmax.min_val)
                {
                    minmax.min_val = Values(i);
                }
                if (Values(i) > minmax.max_val)
                {
                    minmax.max_val = Values(i);
                }
            }
        };

        template<typename VectorType>
        KOKKOS_INLINE_FUNCTION static auto minmax(const VectorType& values) ->
            typename std::enable_if<VectorType::Rank == 1, Kokkos::pair<typename VectorType::traits::non_const_value_type, typename VectorType::traits::non_const_value_type>>::type
        {
            typedef typename VectorType::traits::execution_space      ExecutionSpace;
            typedef typename VectorType::traits::data_type            DataType;
            typedef typename VectorType::traits::non_const_value_type ValueType;

            MinMaxFunctor<VectorType> f(values);

            Kokkos::MinMaxScalar<ValueType> minmax_value;
            Kokkos::MinMax<ValueType>       reducer_scalar(minmax_value);

            Kokkos::parallel_reduce(Kokkos::RangePolicy<ExecutionSpace>(0, values.extent(0)), f, reducer_scalar);
            Kokkos::fence();

            return Kokkos::pair<ValueType, ValueType>(minmax_value.min_val, minmax_value.max_val);
        }

        template<typename VectorType>
        struct SumOfSquaresFunctor
        {
            typedef typename VectorType::traits::execution_space      ExecutionSpace;
            typedef typename VectorType::traits::data_type            DataType;
            typedef typename VectorType::traits::non_const_value_type ValueType;

            VectorType Values;

            SumOfSquaresFunctor(const VectorType& values) : Values(values) {}

            KOKKOS_INLINE_FUNCTION void operator()(const uint32& i, ValueType& value) const
            {
                Kokkos::atomic_fetch_add(&value, Values(i) * Values(i));
            }
        };

        template<typename VectorType>
        KOKKOS_INLINE_FUNCTION static auto sum_of_squares(const VectorType& values) -> typename std::enable_if<VectorType::Rank == 1, typename VectorType::traits::non_const_value_type>::type
        {
            typedef typename VectorType::traits::execution_space      ExecutionSpace;
            typedef typename VectorType::traits::data_type            DataType;
            typedef typename VectorType::traits::non_const_value_type ValueType;

            SumOfSquaresFunctor<VectorType> f(values);

            ValueType sum_value = Constants<ValueType>::Zero();

            Kokkos::parallel_reduce(Kokkos::RangePolicy<ExecutionSpace>(0, values.extent(0)), f, sum_value);
            Kokkos::fence();

            return sum_value;
        }
    }
}

namespace Kokkos
{
    namespace Extension
    {
        template<typename ViewType>
        static auto operator<<(std::ostream& s, const ViewType& A) -> typename std::enable_if<Kokkos::is_view<ViewType>::value, std::ostream&>::type
        {
            const size_type n = A.extent(0);

            s << std::endl;
            s << A.label() << " [" << n << "]";
            s << std::endl;

            if constexpr (ViewType::Rank == 1)
            {
                for (size_type i = 0; i < n; ++i)
                {
                    s << A(i) << std::endl;
                }
            }

            s << std::endl;

            return s;
        }

        template<typename DataType, class ExecutionSpace>
        static std::ostream& operator<<(std::ostream& s, const Extension::Vector<DataType, ExecutionSpace>& A)
        {
            const size_type n = A.extent(0);

            s << std::endl;
            s << A.label() << " [" << n << "]";
            s << std::endl;

            for (size_type i = 0; i < n; ++i)
            {
                s << A(i) << std::endl;
            }

            s << std::endl;

            return s;
        }

        template<typename DataType, class ExecutionSpace>
        KOKKOS_INLINE_FUNCTION System::print& operator<<(System::print& s, const Extension::Vector<DataType, ExecutionSpace>& A)
        {
            const size_type n = A.extent(0);

            s << System::endl;
#if !defined(__CUDA_ARCH__)
            s << A.label();
#endif
            s << " [" << n << "]" << System::endl;

            for (size_type i = 0; i < n; ++i)
            {
                s << A(i) << System::endl;
            }

            s << System::endl;

            return s;
        }

        template<typename DataType, class ExecutionSpace>
        static std::istream& operator>>(std::istream& s, Extension::Vector<DataType, ExecutionSpace>& A)
        {
            size_type n;

            s >> n;

            if (!(n == A.extent(0)))
            {
                Kokkos::resize(A, n);
            }

            for (size_type i = 0; i < n; ++i)
            {
                s >> A(i);
            }

            return s;
        }

        namespace VectorOperators
        {
            template<class RVector, class RhsVector>
            struct VectorNegateFunctor
            {
                static_assert(Kokkos::is_view<RVector>::value && RVector::Rank == 1, "RVector::Rank != 1");
                static_assert(Kokkos::is_view<RhsVector>::value && RhsVector::Rank == 1, "RhsVector::Rank != 1");

                typedef typename RVector::size_type size_type;

                RVector                        _r;
                typename RhsVector::const_type _rhs;

                VectorNegateFunctor(const RVector& r, const RhsVector& rhs) : _r(r), _rhs(rhs) {}

                KOKKOS_INLINE_FUNCTION void operator()(const size_type i) const
                {
                    _r(i) = -_rhs(i);
                }
            };

            template<typename RhsVectorType,
                     typename ReturnDataType   = decltype(-std::declval<typename RhsVectorType::non_const_value_type>()),
                     typename ReturnVectorType = Vector<ReturnDataType, typename RhsVectorType::execution_space, typename RhsVectorType::execution_space::array_layout>>
            __inline static auto operator-(const RhsVectorType& rhs) -> std::enable_if_t<Kokkos::is_view<RhsVectorType>::value, ReturnVectorType>
            {
                const size_type n = rhs.extent(0);

                Assert(n == rhs.extent(0));

                ReturnVectorType r("-" + rhs.label(), n);

                VectorNegateFunctor<ReturnVectorType, RhsVectorType> f(r, rhs);

                Kokkos::RangePolicy<typename RhsVectorType::execution_space> policy(0, n);

                Kokkos::parallel_for("VectorNegateFunctor", policy, f);
                Kokkos::fence();

                return r;
            }

#define VECTOR_OPS_FUNCTORS(OP_NAME, OP, ASSIGN_OP)                                                                                                                                                                                  \
    template<class RVector, class LhsVector, class RhsVector>                                                                                                                                                                        \
    struct VectorVector##OP_NAME##Functor                                                                                                                                                                                            \
    {                                                                                                                                                                                                                                \
        static_assert(Kokkos::is_view<RVector>::value && RVector::Rank == 1, "RVector::Rank != 1");                                                                                                                                  \
        static_assert(Kokkos::is_view<LhsVector>::value && LhsVector::Rank == 1, "LhsVector::Rank != 1");                                                                                                                            \
        static_assert(Kokkos::is_view<RhsVector>::value && RhsVector::Rank == 1, "RhsVector::Rank != 1");                                                                                                                            \
                                                                                                                                                                                                                                     \
        typedef typename RVector::size_type    size_type;                                                                                                                                                                            \
        typedef typename LhsVector::value_type value_type;                                                                                                                                                                           \
                                                                                                                                                                                                                                     \
        RVector                        _r;                                                                                                                                                                                           \
        typename LhsVector::const_type _lhs;                                                                                                                                                                                         \
        typename RhsVector::const_type _rhs;                                                                                                                                                                                         \
                                                                                                                                                                                                                                     \
        VectorVector##OP_NAME##Functor(const RVector& r, const LhsVector& lhs, const RhsVector& rhs) : _r(r), _lhs(lhs), _rhs(rhs) {}                                                                                                \
                                                                                                                                                                                                                                     \
        KOKKOS_INLINE_FUNCTION void operator()(const size_type i) const                                                                                                                                                              \
        {                                                                                                                                                                                                                            \
            _r(i) = _lhs(i) OP _rhs(i);                                                                                                                                                                                              \
        }                                                                                                                                                                                                                            \
    };                                                                                                                                                                                                                               \
                                                                                                                                                                                                                                     \
    template<class RVector, class LhsVector>                                                                                                                                                                                         \
    struct VectorScalar##OP_NAME##Functor                                                                                                                                                                                            \
    {                                                                                                                                                                                                                                \
        static_assert(Kokkos::is_view<RVector>::value && RVector::Rank == 1, "RVector::Rank != 1");                                                                                                                                  \
        static_assert(Kokkos::is_view<LhsVector>::value && LhsVector::Rank == 1, "LhsVector::Rank != 1");                                                                                                                            \
                                                                                                                                                                                                                                     \
        typedef typename RVector::size_type              size_type;                                                                                                                                                                  \
        typedef typename LhsVector::non_const_value_type value_type;                                                                                                                                                                 \
                                                                                                                                                                                                                                     \
        RVector                        _r;                                                                                                                                                                                           \
        typename LhsVector::const_type _lhs;                                                                                                                                                                                         \
        value_type                     _rhs;                                                                                                                                                                                         \
                                                                                                                                                                                                                                     \
        VectorScalar##OP_NAME##Functor(const RVector& r, const LhsVector& lhs, const value_type rhs) : _r(r), _lhs(lhs), _rhs(rhs) {}                                                                                                \
                                                                                                                                                                                                                                     \
        KOKKOS_INLINE_FUNCTION void operator()(const size_type i) const                                                                                                                                                              \
        {                                                                                                                                                                                                                            \
            _r(i) = _lhs(i) OP _rhs;                                                                                                                                                                                                 \
        }                                                                                                                                                                                                                            \
    };                                                                                                                                                                                                                               \
                                                                                                                                                                                                                                     \
    template<class RVector, class RhsVector>                                                                                                                                                                                         \
    struct ScalarVector##OP_NAME##Functor                                                                                                                                                                                            \
    {                                                                                                                                                                                                                                \
        static_assert(Kokkos::is_view<RVector>::value && RVector::Rank == 1, "RVector::Rank != 1");                                                                                                                                  \
        static_assert(Kokkos::is_view<RhsVector>::value && RhsVector::Rank == 1, "RhsVector::Rank != 1");                                                                                                                            \
                                                                                                                                                                                                                                     \
        typedef typename RVector::size_type              size_type;                                                                                                                                                                  \
        typedef typename RhsVector::non_const_value_type value_type;                                                                                                                                                                 \
                                                                                                                                                                                                                                     \
        RVector                        _r;                                                                                                                                                                                           \
        value_type                     _lhs;                                                                                                                                                                                         \
        typename RhsVector::const_type _rhs;                                                                                                                                                                                         \
                                                                                                                                                                                                                                     \
        ScalarVector##OP_NAME##Functor(const RVector& r, const value_type lhs, const RhsVector& rhs) : _r(r), _lhs(lhs), _rhs(rhs) {}                                                                                                \
                                                                                                                                                                                                                                     \
        KOKKOS_INLINE_FUNCTION void operator()(const size_type i) const                                                                                                                                                              \
        {                                                                                                                                                                                                                            \
            _r(i) = _lhs OP _rhs(i);                                                                                                                                                                                                 \
        }                                                                                                                                                                                                                            \
    };                                                                                                                                                                                                                               \
                                                                                                                                                                                                                                     \
    template<class LhsVector, class RhsVector>                                                                                                                                                                                       \
    struct VectorVector##OP_NAME##AssignFunctor                                                                                                                                                                                      \
    {                                                                                                                                                                                                                                \
        static_assert(Kokkos::is_view<LhsVector>::value && LhsVector::Rank == 1, "LhsVector::Rank != 1");                                                                                                                            \
        static_assert(Kokkos::is_view<RhsVector>::value && RhsVector::Rank == 1, "RhsVector::Rank != 1");                                                                                                                            \
                                                                                                                                                                                                                                     \
        typedef typename LhsVector::size_type  size_type;                                                                                                                                                                            \
        typedef typename LhsVector::value_type value_type;                                                                                                                                                                           \
                                                                                                                                                                                                                                     \
        LhsVector                      _lhs;                                                                                                                                                                                         \
        typename RhsVector::const_type _rhs;                                                                                                                                                                                         \
                                                                                                                                                                                                                                     \
        VectorVector##OP_NAME##AssignFunctor(const LhsVector& lhs, const RhsVector& rhs) : _lhs(lhs), _rhs(rhs) {}                                                                                                                   \
                                                                                                                                                                                                                                     \
        KOKKOS_INLINE_FUNCTION void operator()(const size_type i) const                                                                                                                                                              \
        {                                                                                                                                                                                                                            \
            _lhs(i) ASSIGN_OP _rhs(i);                                                                                                                                                                                               \
        }                                                                                                                                                                                                                            \
    };                                                                                                                                                                                                                               \
                                                                                                                                                                                                                                     \
    template<class LhsVector>                                                                                                                                                                                                        \
    struct VectorScalar##OP_NAME##AssignFunctor                                                                                                                                                                                      \
    {                                                                                                                                                                                                                                \
        static_assert(Kokkos::is_view<LhsVector>::value && LhsVector::Rank == 1, "LhsVector::Rank != 1");                                                                                                                            \
                                                                                                                                                                                                                                     \
        typedef typename LhsVector::size_type            size_type;                                                                                                                                                                  \
        typedef typename LhsVector::non_const_value_type value_type;                                                                                                                                                                 \
                                                                                                                                                                                                                                     \
        LhsVector  _lhs;                                                                                                                                                                                                             \
        value_type _rhs;                                                                                                                                                                                                             \
                                                                                                                                                                                                                                     \
        VectorScalar##OP_NAME##AssignFunctor(const LhsVector& lhs, const value_type rhs) : _lhs(lhs), _rhs(rhs) {}                                                                                                                   \
                                                                                                                                                                                                                                     \
        KOKKOS_INLINE_FUNCTION void operator()(const size_type i) const                                                                                                                                                              \
        {                                                                                                                                                                                                                            \
            _lhs(i) ASSIGN_OP _rhs;                                                                                                                                                                                                  \
        }                                                                                                                                                                                                                            \
    };                                                                                                                                                                                                                               \
                                                                                                                                                                                                                                     \
    template<typename LhsVectorType,                                                                                                                                                                                                 \
             typename RhsVectorType,                                                                                                                                                                                                 \
             typename ReturnDataType   = decltype(std::declval<typename LhsVectorType::non_const_value_type>() OP std::declval<typename RhsVectorType::non_const_value_type>()),                                                     \
             typename ReturnVectorType = Vector<ReturnDataType, typename LhsVectorType::execution_space, typename LhsVectorType::execution_space::array_layout>>                                                                     \
    __inline static auto operator OP(const LhsVectorType& lhs, const RhsVectorType& rhs)->std::enable_if_t<Kokkos::is_view<LhsVectorType>::value && Kokkos::is_view<RhsVectorType>::value, ReturnVectorType>                         \
    {                                                                                                                                                                                                                                \
        const size_type n = lhs.extent(0);                                                                                                                                                                                           \
                                                                                                                                                                                                                                     \
        Assert(n == lhs.extent(0));                                                                                                                                                                                                  \
                                                                                                                                                                                                                                     \
        ReturnVectorType r(lhs.label() + #OP + rhs.label(), n);                                                                                                                                                                      \
                                                                                                                                                                                                                                     \
        VectorOperators::VectorVector##OP_NAME##Functor<ReturnVectorType, LhsVectorType, RhsVectorType> f(r, lhs, rhs);                                                                                                              \
                                                                                                                                                                                                                                     \
        Kokkos::RangePolicy<typename LhsVectorType::execution_space> policy(0, n);                                                                                                                                                   \
                                                                                                                                                                                                                                     \
        Kokkos::parallel_for("V_" #OP_NAME, policy, f);                                                                                                                                                                              \
                                                                                                                                                                                                                                     \
        Kokkos::fence();                                                                                                                                                                                                             \
                                                                                                                                                                                                                                     \
        return r;                                                                                                                                                                                                                    \
    }                                                                                                                                                                                                                                \
                                                                                                                                                                                                                                     \
    template<FloatingPoint LhsScalarType, typename RhsVectorType>                                                                                                                                                                    \
    __inline static auto operator OP(const LhsScalarType& lhs, const RhsVectorType& rhs)->std::enable_if_t<Kokkos::is_view<RhsVectorType>::value, RhsVectorType>                                                                     \
    {                                                                                                                                                                                                                                \
        const size_type n = rhs.extent(0);                                                                                                                                                                                           \
                                                                                                                                                                                                                                     \
        Assert(n == rhs.extent(0));                                                                                                                                                                                                  \
                                                                                                                                                                                                                                     \
        RhsVectorType r(std::to_string(lhs) + #OP + rhs.label(), n);                                                                                                                                                                 \
                                                                                                                                                                                                                                     \
        VectorOperators::ScalarVector##OP_NAME##Functor<RhsVectorType, RhsVectorType> f(r, lhs, rhs);                                                                                                                                \
                                                                                                                                                                                                                                     \
        Kokkos::RangePolicy<typename RhsVectorType::execution_space> policy(0, n);                                                                                                                                                   \
                                                                                                                                                                                                                                     \
        Kokkos::parallel_for("V_" #OP_NAME, policy, f);                                                                                                                                                                              \
                                                                                                                                                                                                                                     \
        Kokkos::fence();                                                                                                                                                                                                             \
                                                                                                                                                                                                                                     \
        return r;                                                                                                                                                                                                                    \
    }                                                                                                                                                                                                                                \
                                                                                                                                                                                                                                     \
    template<typename LhsVectorType, FloatingPoint RhsScalarType>                                                                                                                                                                    \
    __inline static auto operator OP(const LhsVectorType& lhs, const RhsScalarType rhs)->std::enable_if_t<Kokkos::is_view<LhsVectorType>::value, LhsVectorType>                                                                      \
    {                                                                                                                                                                                                                                \
        const size_type n = lhs.extent(0);                                                                                                                                                                                           \
                                                                                                                                                                                                                                     \
        LhsVectorType r(lhs.label() + #OP + std::to_string(rhs), n);                                                                                                                                                                 \
                                                                                                                                                                                                                                     \
        VectorOperators::VectorScalar##OP_NAME##Functor<LhsVectorType, LhsVectorType> f(r, lhs, rhs);                                                                                                                                \
                                                                                                                                                                                                                                     \
        Kokkos::RangePolicy<typename LhsVectorType::execution_space> policy(0, n);                                                                                                                                                   \
                                                                                                                                                                                                                                     \
        Kokkos::parallel_for("V_" #OP_NAME, policy, f);                                                                                                                                                                              \
                                                                                                                                                                                                                                     \
        Kokkos::fence();                                                                                                                                                                                                             \
                                                                                                                                                                                                                                     \
        return r;                                                                                                                                                                                                                    \
    }                                                                                                                                                                                                                                \
                                                                                                                                                                                                                                     \
    template<typename LhsVectorType, typename RhsVectorType>                                                                                                                                                                         \
    __inline static auto operator ASSIGN_OP(LhsVectorType& lhs, const RhsVectorType& rhs)->typename std::enable_if<Kokkos::is_view<LhsVectorType>::value && Kokkos::is_view<RhsVectorType>::value, LhsVectorType>::type              \
    {                                                                                                                                                                                                                                \
        const size_type n = lhs.extent(0);                                                                                                                                                                                           \
                                                                                                                                                                                                                                     \
        Assert(n == lhs.extent(0));                                                                                                                                                                                                  \
                                                                                                                                                                                                                                     \
        VectorOperators::VectorVector##OP_NAME##AssignFunctor<LhsVectorType, RhsVectorType> f(lhs, rhs);                                                                                                                             \
                                                                                                                                                                                                                                     \
        Kokkos::RangePolicy<typename RhsVectorType::execution_space> policy(0, n);                                                                                                                                                   \
                                                                                                                                                                                                                                     \
        Kokkos::parallel_for("V_" #OP_NAME "Assign", policy, f);                                                                                                                                                                     \
                                                                                                                                                                                                                                     \
        Kokkos::fence();                                                                                                                                                                                                             \
                                                                                                                                                                                                                                     \
        return lhs;                                                                                                                                                                                                                  \
    }                                                                                                                                                                                                                                \
                                                                                                                                                                                                                                     \
    template<typename LhsVectorType, FloatingPoint RhsScalarType>                                                                                                                                                                    \
    __inline static auto operator ASSIGN_OP(LhsVectorType& lhs, const RhsScalarType rhs)->typename std::enable_if<Kokkos::is_view<LhsVectorType>::value, LhsVectorType>::type                                                        \
    {                                                                                                                                                                                                                                \
        const size_type n = lhs.extent(0);                                                                                                                                                                                           \
                                                                                                                                                                                                                                     \
        VectorOperators::VectorScalar##OP_NAME##AssignFunctor<LhsVectorType> f(lhs, rhs);                                                                                                                                            \
                                                                                                                                                                                                                                     \
        Kokkos::RangePolicy<typename LhsVectorType::execution_space> policy(0, n);                                                                                                                                                   \
                                                                                                                                                                                                                                     \
        Kokkos::parallel_for("V_" #OP_NAME "Scalar", policy, f);                                                                                                                                                                     \
                                                                                                                                                                                                                                     \
        Kokkos::fence();                                                                                                                                                                                                             \
                                                                                                                                                                                                                                     \
        return lhs;                                                                                                                                                                                                                  \
    }

            VECTOR_OPS_FUNCTORS(Plus, +, +=)
            VECTOR_OPS_FUNCTORS(Minus, -, -=)
            VECTOR_OPS_FUNCTORS(Multiply, *, *=)
            VECTOR_OPS_FUNCTORS(Divide, /, /=)

#undef VECTOR_OPS_FUNCTORS

            template<class LhsVector, class RhsVector>
            struct VectorInnerProductFunctor
            {
                static_assert(Kokkos::is_view<LhsVector>::value && LhsVector::Rank == 1, "LhsVector::Rank != 1");
                static_assert(Kokkos::is_view<RhsVector>::value && RhsVector::Rank == 1, "RhsVector::Rank != 1");

                typedef typename LhsVector::size_type            size_type;
                typedef typename LhsVector::non_const_value_type value_type;

                typename LhsVector::const_type _lhs;
                typename RhsVector::const_type _rhs;

                VectorInnerProductFunctor(const LhsVector& lhs, const RhsVector& rhs) : _lhs(lhs), _rhs(rhs) {}

                KOKKOS_INLINE_FUNCTION void operator()(const size_type i, value_type& sum) const
                {
                    Kokkos::atomic_fetch_add(&sum, _lhs(i) * _rhs(i));
                }
            };

            template<class RMatrix, class LhsVector, class RhsVector>
            struct VectorOuterProductFunctor
            {
                static_assert((Kokkos::is_view<RMatrix>::value || Kokkos::is_view<typename RMatrix::ViewType>::value) && RMatrix::Rank == 2, "RVector::Rank != 2");
                static_assert(Kokkos::is_view<LhsVector>::value && LhsVector::Rank == 1, "LhsVector::Rank != 1");
                static_assert(Kokkos::is_view<RhsVector>::value && RhsVector::Rank == 1, "RhsVector::Rank != 1");

                typedef typename LhsVector::size_type            size_type;
                typedef typename LhsVector::non_const_value_type value_type;

                RMatrix                        _r;
                typename LhsVector::const_type _lhs;
                typename RhsVector::const_type _rhs;

                VectorOuterProductFunctor(RMatrix& r, const LhsVector& lhs, const RhsVector& rhs) : _r(r), _lhs(lhs), _rhs(rhs) {}

                KOKKOS_INLINE_FUNCTION void operator()(const size_type i, const size_type j) const
                {
                    _r(i, j) = _lhs(i) * _rhs(j);
                }
            };

            template<class LhsVector>
            struct VectorNormFunctor
            {
                static_assert(Kokkos::is_view<LhsVector>::value && LhsVector::Rank == 1, "LhsVector::Rank != 1");

                typedef typename LhsVector::size_type            size_type;
                typedef typename LhsVector::non_const_value_type value_type;

                typename LhsVector::const_type _lhs;

                VectorNormFunctor(const LhsVector& lhs) : _lhs(lhs) {}

                KOKKOS_INLINE_FUNCTION void operator()(const size_type i, value_type& sum) const
                {
                    sum += abs(_lhs(i)) * abs(_lhs(i));
                }
            };
        }

        using VectorOperators::operator+;
        using VectorOperators::operator-;
        using VectorOperators::operator*;
        using VectorOperators::operator/;
        using VectorOperators::operator+=;
        using VectorOperators::operator-=;
        using VectorOperators::operator*=;
        using VectorOperators::operator/=;
    }
}

namespace Kokkos
{
    using Extension::operator+;
    using Extension::operator-;
    using Extension::operator*;
    using Extension::operator/;
    using Extension::operator+=;
    using Extension::operator-=;
    using Extension::operator*=;
    using Extension::operator/=;

    using Kokkos::Extension::operator<<;
    using Kokkos::Extension::operator>>;
}

namespace Kokkos
{
    namespace Extension
    {
        template<typename DataType, class ExecutionSpace>
        KOKKOS_INLINE_FUNCTION static int64 BinarySearch(Vector<DataType, ExecutionSpace>& view, const DataType& find, const bool sort = true)
        {
            const uint64 length = view.extent(0);

            if (sort)
            {
                Kokkos::sort(view, true);
            }

            uint64 lo = 0;
            uint64 hi = length - 1;
            uint64 i;

            if (find < view[0])
            {
                return ~((int64)lo);
            }

            if (find > view[length - 1])
            {
                return ~((int64)hi);
            }

            while (lo <= hi)
            {
                // i = lo + ((hi - lo) >> 1);
                i = ((hi + lo) >> 1);

                if (view[i] < find)
                {
                    lo = i + 1;
                }
                else if (view[i] > find)
                {
                    hi = i - 1;
                }
                else
                {
                    return i;
                }
            }

            lo -= 1;

            return lo;
        }
    }
}

namespace Kokkos
{
    namespace Extension
    {
        // clang-format off
#define UNARY_FUNCTION(NAME)                                                                                                                                                                                                         \
    template<FloatingPoint DataType, class ExecutionSpace>                                                                                                                                                                           \
    __inline static Vector<DataType, ExecutionSpace> NAME(const Vector<DataType, ExecutionSpace>& view)                                                                                                                              \
    {                                                                                                                                                                                                                                \
        const uint32                     N = view.extent(0);                                                                                                                                                                         \
        Vector<DataType, ExecutionSpace> result(STRINGIZER(NAME) "(" + view.label() + ")", N);                                                                                                                                       \
        Kokkos::parallel_for(STRINGIZER(NAME), N, KOKKOS_LAMBDA(const uint32 i)                                                                                                                                                      \
        {                                                                                                                                                                                                                            \
            result(i) = System::NAME(view(i));                                                                                                                                                                                          \
        });                                                                                                                                                                                                                          \
        Kokkos::fence();                                                                                                                                                                                                             \
        return result;                                                                                                                                                                                                               \
    }
        
        UNARY_FUNCTION(abs)
        UNARY_FUNCTION(cos)
        UNARY_FUNCTION(acos)
        UNARY_FUNCTION(cosh)
        UNARY_FUNCTION(acosh)
        UNARY_FUNCTION(sin)
        UNARY_FUNCTION(asin)
        UNARY_FUNCTION(sinh)
        UNARY_FUNCTION(asinh)
        UNARY_FUNCTION(tan)
        UNARY_FUNCTION(atan)
        UNARY_FUNCTION(tanh)
        UNARY_FUNCTION(atanh)
        UNARY_FUNCTION(cot)
        UNARY_FUNCTION(coth)
        UNARY_FUNCTION(acot)
        UNARY_FUNCTION(acoth)
        UNARY_FUNCTION(sec)
        UNARY_FUNCTION(sech)
        UNARY_FUNCTION(asec)
        UNARY_FUNCTION(asech)
        UNARY_FUNCTION(csc)
        UNARY_FUNCTION(csch)
        UNARY_FUNCTION(acsc)
        UNARY_FUNCTION(acsch)
        UNARY_FUNCTION(exp)
        UNARY_FUNCTION(exp2)
        UNARY_FUNCTION(expm1)
        UNARY_FUNCTION(log)
        UNARY_FUNCTION(log10)
        UNARY_FUNCTION(log1p)
        UNARY_FUNCTION(logb)
        UNARY_FUNCTION(log2)
        UNARY_FUNCTION(round)
        UNARY_FUNCTION(ceil)
        UNARY_FUNCTION(floor)
        UNARY_FUNCTION(trunc)
        UNARY_FUNCTION(lgamma)
        UNARY_FUNCTION(tgamma)
        UNARY_FUNCTION(erf)
        UNARY_FUNCTION(erfc)
        UNARY_FUNCTION(inv)
        UNARY_FUNCTION(sqr)
        UNARY_FUNCTION(sqrt)
        UNARY_FUNCTION(sign)

#undef UNARY_FUNCTION

#define BINARY_FUNCTION(NAME)                                                                                                                                                                                                         \
    template<FloatingPoint DataType, class ExecutionSpace>                                                                                                                                                                            \
    __inline static Vector<DataType, ExecutionSpace> NAME(const DataType x, const Vector<DataType, ExecutionSpace>& view_y)                                                                                                           \
    {                                                                                                                                                                                                                                 \
        const uint32                     N = view_y.extent(0);                                                                                                                                                                        \
        Vector<DataType, ExecutionSpace> result(STRINGIZER(NAME) "(" + std::to_string(x) + "," + view_y.label() + ")", N);                                                                                                            \
        Kokkos::parallel_for(STRINGIZER(NAME), N, KOKKOS_LAMBDA(const uint32 i)                                                                                                                                                       \
        {                                                                                                                                                                                                                             \
            result(i) = System::NAME(x, view_y(i));                                                                                                                                                                                      \
        });                                                                                                                                                                                                                           \
        Kokkos::fence();                                                                                                                                                                                                              \
        return result;                                                                                                                                                                                                                \
    }                                                                                                                                                                                                                                 \
    template<FloatingPoint DataType, class ExecutionSpace>                                                                                                                                                                            \
    __inline static Vector<DataType, ExecutionSpace> NAME(const Vector<DataType, ExecutionSpace>& view_x, const DataType y)                                                                                                           \
    {                                                                                                                                                                                                                                 \
        const uint32                     N = view_x.extent(0);                                                                                                                                                                        \
        Vector<DataType, ExecutionSpace> result(STRINGIZER(NAME) "(" + view_x.label() + "," + std::to_string(y) + ")", N);                                                                                                            \
        Kokkos::parallel_for(STRINGIZER(NAME), N, KOKKOS_LAMBDA(const uint32 i)                                                                                                                                                       \
        {                                                                                                                                                                                                                             \
            result(i) = System::NAME(view_x(i), y);                                                                                                                                                                                      \
        });                                                                                                                                                                                                                           \
        Kokkos::fence();                                                                                                                                                                                                              \
        return result;                                                                                                                                                                                                                \
    }                                                                                                                                                                                                                                 \
    template<FloatingPoint DataType, class ExecutionSpace>                                                                                                                                                                            \
    __inline static Vector<DataType, ExecutionSpace> NAME(const Vector<DataType, ExecutionSpace>& view_x, const Vector<DataType, ExecutionSpace>& view_y)                                                                             \
    {                                                                                                                                                                                                                                 \
        const uint32                     N = view_x.extent(0);                                                                                                                                                                        \
        Vector<DataType, ExecutionSpace> result(STRINGIZER(NAME) "(" + view_x.label() + "," + view_y.label() + ")", N);                                                                                                               \
        Kokkos::parallel_for(STRINGIZER(NAME), N, KOKKOS_LAMBDA(const uint32 i)                                                                                                                                                       \
        {                                                                                                                                                                                                                             \
            result(i) = System::NAME(view_x(i), view_y(i));                                                                                                                                                                              \
        });                                                                                                                                                                                                                           \
        Kokkos::fence();                                                                                                                                                                                                              \
        return result;                                                                                                                                                                                                                \
    }

        BINARY_FUNCTION(copysign)
        BINARY_FUNCTION(sign)
        BINARY_FUNCTION(fmin)
        BINARY_FUNCTION(fmax)
        BINARY_FUNCTION(fmod)
        BINARY_FUNCTION(hypot)
        BINARY_FUNCTION(pow)

        template<FloatingPoint DataType, class ExecutionSpace>
        __inline static Vector<DataType, ExecutionSpace> pow(const Vector<DataType, ExecutionSpace>& view_x, const int32 y)
        {
            const uint32 N = view_x.extent(0);
            Vector<DataType, ExecutionSpace> result(STRINGIZER(pow) "(" + view_x.label() + "," + std::to_string(y) + ")", N);
            Kokkos::parallel_for(STRINGIZER(pow), N, KOKKOS_LAMBDA(const uint32 i)
            {
                result(i) = System::pow(view_x(i), y);
            });
            Kokkos::fence();
            return result;
        }
    
#undef BINARY_FUNCTION

        template<FloatingPoint DataType, class ExecutionSpace>
        __inline static Vector<DataType, ExecutionSpace> atan2(const DataType y, const Vector<DataType, ExecutionSpace>& view_x)
        {
            const uint32                     N = view_x.extent(0);
            Vector<DataType, ExecutionSpace> result(STRINGIZER(atan2) "(" + std::to_string(y) + "," + view_x.label() + ")", N);
            Kokkos::parallel_for(STRINGIZER(atan2), N, KOKKOS_LAMBDA(const uint32 i)
            {
                result(i) = System::atan2(y, view_x(i));
            });
            Kokkos::fence();
            return result;
        }
        template<FloatingPoint DataType, class ExecutionSpace>
        __inline static Vector<DataType, ExecutionSpace> atan2(const Vector<DataType, ExecutionSpace>& view_y, const DataType x)
        {
            const uint32                     N = view_y.extent(0);
            Vector<DataType, ExecutionSpace> result(STRINGIZER(atan2) "(" + view_y.label() + "," + std::to_string(x) + ")", N);
            Kokkos::parallel_for(STRINGIZER(atan2), N, KOKKOS_LAMBDA(const uint32 i)
            {
                result(i) = System::atan2(view_y(i), x);
            });
            Kokkos::fence();
            return result;
        }
        template<FloatingPoint DataType, class ExecutionSpace>
        __inline static Vector<DataType, ExecutionSpace> atan2(const Vector<DataType, ExecutionSpace>& view_y, const Vector<DataType, ExecutionSpace>& view_x)
        {
            const uint32                     N = view_y.extent(0);
            Vector<DataType, ExecutionSpace> result(STRINGIZER(atan2) "(" + view_y.label() + "," + view_x.label() + ")", N);
            Kokkos::parallel_for(STRINGIZER(atan2), N, KOKKOS_LAMBDA(const uint32 i)
            {
                result(i) = System::atan2(view_y(i), view_x(i));
            });
            Kokkos::fence();
            return result;
        }



        template<FloatingPoint DataType, class ExecutionSpace>
        __inline static Vector<DataType, ExecutionSpace> fma(const Vector<DataType, ExecutionSpace>& view_x, const Vector<DataType, ExecutionSpace>& view_y, const Vector<DataType, ExecutionSpace>& view_z)
        {
            const uint32                     N = view_x.extent(0);
            Vector<DataType, ExecutionSpace> result(STRINGIZER(fma) "(" + view_x.label() + "," + view_y.label() + "," + view_z.label() + ")", N);
            Kokkos::parallel_for(STRINGIZER(fma), N, KOKKOS_LAMBDA(const uint32 i)
            {
                result(i) = System::fma(view_x(i), view_y(i), view_z(i));
            });
            Kokkos::fence();
            return result;
        }

        // clang-format on

        // void sincos (double __x, double *p_sin, double *p_cos)

    }
}

//namespace std
//{
//    using Kokkos::Extension::acos;
//    using Kokkos::Extension::acosh;
//    using Kokkos::Extension::acot;
//    using Kokkos::Extension::acoth;
//    using Kokkos::Extension::acsc;
//    using Kokkos::Extension::acsch;
//    using Kokkos::Extension::asec;
//    using Kokkos::Extension::asech;
//    using Kokkos::Extension::asin;
//    using Kokkos::Extension::asinh;
//    using Kokkos::Extension::atan;
//    using Kokkos::Extension::atanh;
//    using Kokkos::Extension::ceil;
//    using Kokkos::Extension::cos;
//    using Kokkos::Extension::cosh;
//    using Kokkos::Extension::cot;
//    using Kokkos::Extension::coth;
//    using Kokkos::Extension::csc;
//    using Kokkos::Extension::csch;
//    using Kokkos::Extension::erf;
//    using Kokkos::Extension::erfc;
//    using Kokkos::Extension::exp;
//    using Kokkos::Extension::exp2;
//    using Kokkos::Extension::expm1;
//    using Kokkos::Extension::floor;
//    using Kokkos::Extension::inv;
//    using Kokkos::Extension::lgamma;
//    using Kokkos::Extension::log;
//    using Kokkos::Extension::log10;
//    using Kokkos::Extension::log1p;
//    using Kokkos::Extension::log2;
//    using Kokkos::Extension::logb;
//    using Kokkos::Extension::round;
//    using Kokkos::Extension::sec;
//    using Kokkos::Extension::sech;
//    using Kokkos::Extension::sign;
//    using Kokkos::Extension::sin;
//    using Kokkos::Extension::sinh;
//    using Kokkos::Extension::sqr;
//    using Kokkos::Extension::sqrt;
//    using Kokkos::Extension::tan;
//    using Kokkos::Extension::tanh;
//    using Kokkos::Extension::tgamma;
//    using Kokkos::Extension::trunc;
//
//    using Kokkos::Extension::copysign;
//    using Kokkos::Extension::fmax;
//    using Kokkos::Extension::fmod;
//    using Kokkos::Extension::hypot;
//    using Kokkos::Extension::pow;
//    using Kokkos::Extension::sign;
//
//    using Kokkos::Extension::atan2;
//
//    using Kokkos::Extension::fma;
//}

namespace Kokkos
{
    namespace Extension
    {

        template<FloatingPoint DataType, class ExecutionSpace>
        __inline static Vector<DataType, ExecutionSpace> head(const Vector<DataType, ExecutionSpace>& view, const uint32 n)
        {
            const uint32 N = n;

            Vector<DataType, ExecutionSpace> result(view.label(), N);

            Kokkos::parallel_for(Kokkos::RangePolicy<ExecutionSpace>(0, N), [=] __host__ __device__(const uint32 i) { result(i) = view(i); });

            Kokkos::fence();

            return result;
        }

        template<FloatingPoint DataType, class ExecutionSpace>
        __inline static Vector<DataType, ExecutionSpace> tail(const Vector<DataType, ExecutionSpace>& view, const uint32 n)
        {
            const uint32 N = n;

            Vector<DataType, ExecutionSpace> result(view.label(), N);

            Kokkos::parallel_for(Kokkos::RangePolicy<ExecutionSpace>(view.extent(0) - N, view.extent(0)), [=] __host__ __device__(const uint32 i) { result(i) = view(i); });

            Kokkos::fence();

            return result;
        }

    }
}
