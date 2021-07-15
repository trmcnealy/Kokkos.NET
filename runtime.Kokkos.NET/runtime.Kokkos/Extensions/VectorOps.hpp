#pragma once

#if !defined(KOKKOS_EXTENSIONS)
#    error "Do not include directly. Include Extensions.hpp"
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
                Kokkos::atomic_fetch_min(&value, Values(i));
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
                Kokkos::atomic_fetch_max(&value, Values(i));
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
            // template<class RVector, class XVector, class YVector, int scalar_x, int scalar_y>
            // struct V_AddVectorFunctor
            //{
            //    typedef typename RVector::size_type  size_type;
            //    typedef typename XVector::value_type value_type;
            //
            //    RVector                      _r;
            //    typename XVector::const_type _x;
            //    typename YVector::const_type _y;
            //    const value_type             _a;
            //    const value_type             _b;
            //
            //    V_AddVectorFunctor(const RVector& r, const value_type& a, const XVector& x, const value_type& b, const YVector& y) : _r(r), _x(x), _y(y), _a(a), _b(b) {}
            //
            //    KOKKOS_INLINE_FUNCTION void operator()(const size_type i) const
            //    {
            //        if((scalar_x == 1.0) && (scalar_y == 1.0))
            //        {
            //            _r(i) = _x(i) + _y(i);
            //        }
            //
            //        if((scalar_x == 1.0) && (scalar_y == -1.0))
            //        {
            //            _r(i) = _x(i) - _y(i);
            //        }
            //
            //        if((scalar_x == -1.0) && (scalar_y == -1.0))
            //        {
            //            _r(i) = -_x(i) - _y(i);
            //        }
            //
            //        if((scalar_x == -1.0) && (scalar_y == 1.0))
            //        {
            //            _r(i) = -_x(i) + _y(i);
            //        }
            //
            //        if((scalar_x == 2.0) && (scalar_y == 1.0))
            //        {
            //            _r(i) = _a * _x(i) + _y(i);
            //        }
            //
            //        if((scalar_x == 2.0) && (scalar_y == -1.0))
            //        {
            //            _r(i) = _a * _x(i) - _y(i);
            //        }
            //
            //        if((scalar_x == 1.0) && (scalar_y == 2.0))
            //        {
            //            _r(i) = _x(i) + _b * _y(i);
            //        }
            //
            //        if((scalar_x == -1.0) && (scalar_y == 2.0))
            //        {
            //            _r(i) = -_x(i) + _b * _y(i);
            //        }
            //
            //        if((scalar_x == 2.0) && (scalar_y == 2.0))
            //        {
            //            _r(i) = _a * _x(i) + _b * _y(i);
            //        }
            //    }
            //};

#define VECTOR_OPS_FUNCTORS(OP_NAME, OP, ASSIGN_OP)                                                                                                                                                    \
    template<class RVector, class LhsVector, class RhsVector>                                                                                                                                          \
    struct VectorVector##OP_NAME##Functor                                                                                                                                                              \
    {                                                                                                                                                                                                  \
        static_assert(Kokkos::is_view<RVector>::value && RVector::Rank == 1, "RVector::Rank != 1");                                                                                                    \
        static_assert(Kokkos::is_view<LhsVector>::value && LhsVector::Rank == 1, "LhsVector::Rank != 1");                                                                                              \
        static_assert(Kokkos::is_view<RhsVector>::value && RhsVector::Rank == 1, "RhsVector::Rank != 1");                                                                                              \
                                                                                                                                                                                                       \
        typedef typename RVector::size_type    size_type;                                                                                                                                              \
        typedef typename LhsVector::value_type value_type;                                                                                                                                             \
                                                                                                                                                                                                       \
        RVector                        _r;                                                                                                                                                             \
        typename LhsVector::const_type _lhs;                                                                                                                                                           \
        typename RhsVector::const_type _rhs;                                                                                                                                                           \
                                                                                                                                                                                                       \
        VectorVector##OP_NAME##Functor(const RVector& r, const LhsVector& lhs, const RhsVector& rhs) : _r(r), _lhs(lhs), _rhs(rhs) {}                                                                  \
                                                                                                                                                                                                       \
        KOKKOS_INLINE_FUNCTION void operator()(const size_type i) const                                                                                                                                \
        {                                                                                                                                                                                              \
            _r(i) = _lhs(i) OP _rhs(i);                                                                                                                                                                \
        }                                                                                                                                                                                              \
    };                                                                                                                                                                                                 \
                                                                                                                                                                                                       \
    template<class RVector, class LhsVector>                                                                                                                                                           \
    struct VectorScalar##OP_NAME##Functor                                                                                                                                                              \
    {                                                                                                                                                                                                  \
        static_assert(Kokkos::is_view<RVector>::value && RVector::Rank == 1, "RVector::Rank != 1");                                                                                                    \
        static_assert(Kokkos::is_view<LhsVector>::value && LhsVector::Rank == 1, "LhsVector::Rank != 1");                                                                                              \
                                                                                                                                                                                                       \
        typedef typename RVector::size_type              size_type;                                                                                                                                    \
        typedef typename LhsVector::non_const_value_type value_type;                                                                                                                                   \
                                                                                                                                                                                                       \
        RVector                        _r;                                                                                                                                                             \
        typename LhsVector::const_type _lhs;                                                                                                                                                           \
        value_type                     _rhs;                                                                                                                                                           \
                                                                                                                                                                                                       \
        VectorScalar##OP_NAME##Functor(const RVector& r, const LhsVector& lhs, const value_type& rhs) : _r(r), _lhs(lhs), _rhs(rhs) {}                                                                 \
                                                                                                                                                                                                       \
        KOKKOS_INLINE_FUNCTION void operator()(const size_type i) const                                                                                                                                \
        {                                                                                                                                                                                              \
            _r(i) = _lhs(i) OP _rhs;                                                                                                                                                                   \
        }                                                                                                                                                                                              \
    };                                                                                                                                                                                                 \
                                                                                                                                                                                                       \
    template<class RVector, class RhsVector>                                                                                                                                                           \
    struct ScalarVector##OP_NAME##Functor                                                                                                                                                              \
    {                                                                                                                                                                                                  \
        static_assert(Kokkos::is_view<RVector>::value && RVector::Rank == 1, "RVector::Rank != 1");                                                                                                    \
        static_assert(Kokkos::is_view<RhsVector>::value && RhsVector::Rank == 1, "RhsVector::Rank != 1");                                                                                              \
                                                                                                                                                                                                       \
        typedef typename RVector::size_type              size_type;                                                                                                                                    \
        typedef typename RhsVector::non_const_value_type value_type;                                                                                                                                   \
                                                                                                                                                                                                       \
        RVector                        _r;                                                                                                                                                             \
        value_type                     _lhs;                                                                                                                                                           \
        typename RhsVector::const_type _rhs;                                                                                                                                                           \
                                                                                                                                                                                                       \
        ScalarVector##OP_NAME##Functor(const RVector& r, const value_type& lhs, const RhsVector& rhs) : _r(r), _lhs(lhs), _rhs(rhs) {}                                                                 \
                                                                                                                                                                                                       \
        KOKKOS_INLINE_FUNCTION void operator()(const size_type i) const                                                                                                                                \
        {                                                                                                                                                                                              \
            _r(i) = _lhs OP _rhs(i);                                                                                                                                                                   \
        }                                                                                                                                                                                              \
    };                                                                                                                                                                                                 \
                                                                                                                                                                                                       \
    template<class LhsVector, class RhsVector>                                                                                                                                                         \
    struct VectorVector##OP_NAME##AssignFunctor                                                                                                                                                        \
    {                                                                                                                                                                                                  \
        static_assert(Kokkos::is_view<LhsVector>::value && LhsVector::Rank == 1, "LhsVector::Rank != 1");                                                                                              \
        static_assert(Kokkos::is_view<RhsVector>::value && RhsVector::Rank == 1, "RhsVector::Rank != 1");                                                                                              \
                                                                                                                                                                                                       \
        typedef typename LhsVector::size_type  size_type;                                                                                                                                              \
        typedef typename LhsVector::value_type value_type;                                                                                                                                             \
                                                                                                                                                                                                       \
        LhsVector                      _lhs;                                                                                                                                                           \
        typename RhsVector::const_type _rhs;                                                                                                                                                           \
                                                                                                                                                                                                       \
        VectorVector##OP_NAME##AssignFunctor(const LhsVector& lhs, const RhsVector& rhs) : _lhs(lhs), _rhs(rhs) {}                                                                                     \
                                                                                                                                                                                                       \
        KOKKOS_INLINE_FUNCTION void operator()(const size_type i) const                                                                                                                                \
        {                                                                                                                                                                                              \
            _lhs(i) ASSIGN_OP _rhs(i);                                                                                                                                                                 \
        }                                                                                                                                                                                              \
    };                                                                                                                                                                                                 \
                                                                                                                                                                                                       \
    template<class LhsVector>                                                                                                                                                                          \
    struct VectorScalar##OP_NAME##AssignFunctor                                                                                                                                                        \
    {                                                                                                                                                                                                  \
        static_assert(Kokkos::is_view<LhsVector>::value && LhsVector::Rank == 1, "LhsVector::Rank != 1");                                                                                              \
                                                                                                                                                                                                       \
        typedef typename LhsVector::size_type            size_type;                                                                                                                                    \
        typedef typename LhsVector::non_const_value_type value_type;                                                                                                                                   \
                                                                                                                                                                                                       \
        LhsVector  _lhs;                                                                                                                                                                               \
        value_type _rhs;                                                                                                                                                                               \
                                                                                                                                                                                                       \
        VectorScalar##OP_NAME##AssignFunctor(const LhsVector& lhs, const value_type& rhs) : _lhs(lhs), _rhs(rhs) {}                                                                                    \
                                                                                                                                                                                                       \
        KOKKOS_INLINE_FUNCTION void operator()(const size_type i) const                                                                                                                                \
        {                                                                                                                                                                                              \
            _lhs(i) ASSIGN_OP _rhs;                                                                                                                                                                    \
        }                                                                                                                                                                                              \
    };                                                                                                                                                                                                 \
                                                                                                                                                                                                       \
    template<typename LhsVectorType,                                                                                                                                                                   \
             typename RhsVectorType,                                                                                                                                                                   \
             typename ReturnDataType   = decltype(std::declval<typename LhsVectorType::non_const_value_type>() OP std::declval<typename RhsVectorType::non_const_value_type>()),                       \
             typename ReturnVectorType = Vector<ReturnDataType, typename LhsVectorType::execution_space, typename LhsVectorType::execution_space::array_layout>>                                       \
    __inline static ReturnVectorType operator OP(const LhsVectorType& lhs, const RhsVectorType& rhs)                                                                                                   \
    {                                                                                                                                                                                                  \
        const size_type n = lhs.extent(0);                                                                                                                                                             \
                                                                                                                                                                                                       \
        Assert(n == rhs.extent(0));                                                                                                                                                                    \
                                                                                                                                                                                                       \
        ReturnVectorType r(lhs.label() + #OP + rhs.label(), n);                                                                                                                                        \
                                                                                                                                                                                                       \
        VectorOperators::VectorVector##OP_NAME##Functor<ReturnVectorType, LhsVectorType, RhsVectorType> f(r, lhs, rhs);                                                                                \
                                                                                                                                                                                                       \
        Kokkos::RangePolicy<typename LhsVectorType::execution_space> policy(0, n);                                                                                                                     \
                                                                                                                                                                                                       \
        Kokkos::parallel_for("V_" #OP_NAME, policy, f);                                                                                                                                                \
                                                                                                                                                                                                       \
        return r;                                                                                                                                                                                      \
    }                                                                                                                                                                                                  \
                                                                                                                                                                                                       \
    template<typename RhsVectorType, FloatingPoint LhsScalarType = std::enable_if_t<Kokkos::is_view<RhsVectorType>::value, typename RhsVectorType::non_const_value_type>>                              \
    __inline static RhsVectorType operator OP(const LhsScalarType& lhs, const RhsVectorType& rhs)                                                                                                      \
    {                                                                                                                                                                                                  \
        const size_type n = rhs.extent(0);                                                                                                                                                             \
                                                                                                                                                                                                       \
        Assert(n == rhs.extent(0));                                                                                                                                                                    \
                                                                                                                                                                                                       \
        RhsVectorType r(std::to_string(lhs) + #OP + rhs.label(), n);                                                                                                                                   \
                                                                                                                                                                                                       \
        VectorOperators::ScalarVector##OP_NAME##Functor<RhsVectorType, RhsVectorType> f(r, lhs, rhs);                                                                                                  \
                                                                                                                                                                                                       \
        Kokkos::RangePolicy<typename RhsVectorType::execution_space> policy(0, n);                                                                                                                     \
                                                                                                                                                                                                       \
        Kokkos::parallel_for("V_" #OP_NAME, policy, f);                                                                                                                                                \
                                                                                                                                                                                                       \
        return r;                                                                                                                                                                                      \
    }                                                                                                                                                                                                  \
                                                                                                                                                                                                       \
    template<typename LhsVectorType, FloatingPoint RhsScalarType = std::enable_if_t<Kokkos::is_view<LhsVectorType>::value, typename LhsVectorType::non_const_value_type>>                              \
    __inline static LhsVectorType operator OP(const LhsVectorType& lhs, const RhsScalarType& rhs)                                                                                                      \
    {                                                                                                                                                                                                  \
        const size_type n = lhs.extent(0);                                                                                                                                                             \
                                                                                                                                                                                                       \
        LhsVectorType r(lhs.label() + #OP + std::to_string(rhs), n);                                                                                                                                   \
                                                                                                                                                                                                       \
        VectorOperators::VectorScalar##OP_NAME##Functor<LhsVectorType, LhsVectorType> f(r, lhs, rhs);                                                                                                  \
                                                                                                                                                                                                       \
        Kokkos::RangePolicy<typename LhsVectorType::execution_space> policy(0, n);                                                                                                                     \
                                                                                                                                                                                                       \
        Kokkos::parallel_for("V_" #OP_NAME, policy, f);                                                                                                                                                \
                                                                                                                                                                                                       \
        return r;                                                                                                                                                                                      \
    }                                                                                                                                                                                                  \
                                                                                                                                                                                                       \
    template<typename LhsVectorType, typename RhsVectorType>                                                                                                                                           \
    __inline static LhsVectorType operator ASSIGN_OP(LhsVectorType& lhs, const RhsVectorType& rhs)                                                                                                     \
    {                                                                                                                                                                                                  \
        const size_type n = lhs.extent(0);                                                                                                                                                             \
                                                                                                                                                                                                       \
        Assert(n == lhs.extent(0));                                                                                                                                                                    \
                                                                                                                                                                                                       \
        VectorOperators::VectorVector##OP_NAME##AssignFunctor<LhsVectorType, RhsVectorType> f(lhs, rhs);                                                                                               \
                                                                                                                                                                                                       \
        Kokkos::RangePolicy<typename RhsVectorType::execution_space> policy(0, n);                                                                                                                     \
                                                                                                                                                                                                       \
        Kokkos::parallel_for("V_" #OP_NAME "Assign", policy, f);                                                                                                                                       \
                                                                                                                                                                                                       \
        return lhs;                                                                                                                                                                                    \
    }                                                                                                                                                                                                  \
                                                                                                                                                                                                       \
    template<typename LhsVectorType, FloatingPoint RhsScalarType = std::enable_if_t<Kokkos::is_view<LhsVectorType>::value, typename LhsVectorType::non_const_value_type>>                              \
    __inline static LhsVectorType operator ASSIGN_OP(LhsVectorType& lhs, const RhsScalarType& rhs)                                                                                                     \
    {                                                                                                                                                                                                  \
        const size_type n = lhs.extent(0);                                                                                                                                                             \
                                                                                                                                                                                                       \
        VectorOperators::VectorScalar##OP_NAME##AssignFunctor<LhsVectorType> f(lhs, rhs);                                                                                                              \
                                                                                                                                                                                                       \
        Kokkos::RangePolicy<typename LhsVectorType::execution_space> policy(0, n);                                                                                                                     \
                                                                                                                                                                                                       \
        Kokkos::parallel_for("V_" #OP_NAME "Scalar", policy, f);                                                                                                                                       \
                                                                                                                                                                                                       \
        return lhs;                                                                                                                                                                                    \
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
    }
}

namespace Kokkos
{
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

        // template<typename DataType, class ExecutionSpace>
        // static Vector<DataType, ExecutionSpace> Intersect(const Vector<DataType, ExecutionSpace>& first, const Vector<DataType, ExecutionSpace>& second, System::IEqualityComparer<DataType>*
        // comparer)
        //{
        //    var set = new HashSet<TSource>(second, comparer);

        //    for(DataType element : first)
        //    {
        //        if (set.Remove(element))
        //        {
        //            yield return element;
        //        }
        //    }
        //}

    }
}
