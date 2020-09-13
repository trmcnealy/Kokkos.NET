#pragma once

#include "runtime.Kokkos/ViewTypes.hpp"
#include "MathExtensions.hpp"
#include "StdExtensions.hpp"
#include "Constants.hpp"

#include <Kokkos_Core.hpp>
//#include <KokkosBlas.hpp>
//#include <Kokkos_ArithTraits.hpp>
#include <Kokkos_Random.hpp>
#include <Kokkos_UnorderedMap.hpp>

//#include <KokkosBatched_SolveLU_Decl.hpp>
//
//#include <KokkosSparse_trsv.hpp>
//#include <KokkosSparse_spmv.hpp>
//
//#include <KokkosKernels_IOUtils.hpp>
//#include <KokkosKernels_Utils.hpp>
//
//#include <Sacado.hpp>

namespace Kokkos
{
    namespace Impl
    {
        template<class Scalar1, class Scalar2>
        struct LessThanOper
        {
            KOKKOS_FORCEINLINE_FUNCTION static bool apply(const Scalar1& val1, const Scalar2& val2) { return (val1 < val2); }
        };

        template<class Scalar1, class Scalar2>
        struct LessThanEqualToOper
        {
            KOKKOS_FORCEINLINE_FUNCTION static bool apply(const Scalar1& val1, const Scalar2& val2) { return (val1 <= val2); }
        };

        template<class Scalar1, class Scalar2>
        struct GreaterThanOper
        {
            KOKKOS_FORCEINLINE_FUNCTION static bool apply(const Scalar1& val1, const Scalar2& val2) { return (val1 > val2); }
        };

        template<class Scalar1, class Scalar2>
        struct GreaterThanEqualToOper
        {
            KOKKOS_FORCEINLINE_FUNCTION static bool apply(const Scalar1& val1, const Scalar2& val2) { return (val1 >= val2); }
        };

        template<class Scalar1, class Scalar2>
        struct EqualToOper
        {
            KOKKOS_FORCEINLINE_FUNCTION static bool apply(const Scalar1& val1, const Scalar2& val2) { return (val1 == val2); }
        };

        template<class Scalar1, class Scalar2>
        struct NotEqualToOper
        {
            KOKKOS_FORCEINLINE_FUNCTION static bool apply(const Scalar1& val1, const Scalar2& val2) { return (val1 != val2); }
        };
    }

    template<typename T>
    KOKKOS_INLINE_FUNCTION bool atomic_less_than_fetch(volatile T* const dest, const T val)
    {
        return Impl::atomic_oper_fetch(Impl::LessThanOper<T, const T>(), dest, val);
    }

    template<typename T>
    KOKKOS_INLINE_FUNCTION bool atomic_less_than_equal_fetch(volatile T* const dest, const T val)
    {
        return Impl::atomic_oper_fetch(Impl::LessThanEqualToOper<T, const T>(), dest, val);
    }

    template<typename T>
    KOKKOS_INLINE_FUNCTION bool atomic_greater_than_fetch(volatile T* const dest, const T val)
    {
        return Impl::atomic_oper_fetch(Impl::GreaterThanOper<T, const T>(), dest, val);
    }

    template<typename T>
    KOKKOS_INLINE_FUNCTION bool atomic_greater_than_equal_fetch(volatile T* const dest, const T val)
    {
        return Impl::atomic_oper_fetch(Impl::GreaterThanEqualToOper<T, const T>(), dest, val);
    }

    template<typename T>
    KOKKOS_INLINE_FUNCTION bool atomic_equal_to_fetch(volatile T* const dest, const T val)
    {
        return Impl::atomic_oper_fetch(Impl::EqualToOper<T, const T>(), dest, val);
    }

    template<typename T>
    KOKKOS_INLINE_FUNCTION bool atomic_not_equal_to_fetch(volatile T* const dest, const T val)
    {
        return Impl::atomic_oper_fetch(Impl::NotEqualToOper<T, const T>(), dest, val);
    }

    namespace Extension
    {
        template<typename DataType, class ExecutionSpace, class Layout = typename ExecutionSpace::array_layout>
        using Vector = View<DataType*, Layout, ExecutionSpace>;

        template<typename DataType, class ExecutionSpace, class Layout = typename ExecutionSpace::array_layout>
        using Matrix = View<DataType**, Layout, ExecutionSpace>;

        template<class DataType>
        class SparseVectorElement
        {
            DataType  _value;
            size_type _index;

        public:
            SparseVectorElement(const DataType& a, const size_type& i) : _value(a), _index(i) {}

            KOKKOS_INLINE_FUNCTION constexpr DataType&       value() { return _value; }
            KOKKOS_INLINE_FUNCTION constexpr const DataType& value() const { return _value; }

            KOKKOS_INLINE_FUNCTION constexpr size_type&       index() { return _index; }
            KOKKOS_INLINE_FUNCTION constexpr const size_type& index() const { return _index; }
        };

        template<typename DataType, class ExecutionSpace>
        using SparseVector = View<SparseVectorElement<DataType>*, typename ExecutionSpace::array_layout, ExecutionSpace>;

        template<typename DataType, class ExecutionSpace>
        using SparseMatrix = View<SparseVectorElement<DataType>**, typename ExecutionSpace::array_layout, ExecutionSpace>;

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

                    for(size_type i = 0; i < view.extent(0); ++i)
                    {
                        if(view(i) < min_value)
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

                    for(size_type i = 0; i < view.extent(0); ++i)
                    {
                        for(size_type j = 0; j < view.extent(1); ++j)
                        {
                            if(view(i, j) < min_value)
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

namespace Kokkos
{
    namespace Extension
    {
        template<typename KeyType, typename DataType, class ExecutionSpace>
        using UnorderedMap = UnorderedMap<KeyType, DataType, ExecutionSpace>;

        // template<typename DataType>
        // using ArithTraits = Kokkos::Details::ArithTraits<DataType>;

        template<typename VectorType>
        struct MinFunctor
        {
            typedef typename VectorType::traits::execution_space      ExecutionSpace;
            typedef typename VectorType::traits::data_type            DataType;
            typedef typename VectorType::traits::non_const_value_type ValueType;

            VectorType Values;

            MinFunctor(const VectorType& values) : Values(values) {}

            KOKKOS_INLINE_FUNCTION void operator()(const uint32& i, ValueType& value) const { Kokkos::atomic_fetch_min(&value, (ValueType)Values(i)); }
        };

        template<typename VectorType>
        KOKKOS_INLINE_FUNCTION static auto min(const VectorType& values) ->
            typename std::enable_if<VectorType::Rank == 1, typename VectorType::traits::non_const_value_type>::type
        {
            typedef typename VectorType::traits::execution_space      ExecutionSpace;
            typedef typename VectorType::traits::data_type            DataType;
            typedef typename VectorType::traits::non_const_value_type ValueType;

            MinFunctor<VectorType> f(values);

            ValueType      min_value = Constants<ValueType>::max();
            Min<ValueType> reducer_scalar(min_value);

            Kokkos::parallel_reduce(Kokkos::RangePolicy<ExecutionSpace>(0, values.extent(0)), f, reducer_scalar);
            fence();

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

            KOKKOS_INLINE_FUNCTION void operator()(const uint32& i, ValueType& value) const { Kokkos::atomic_fetch_max(&value, (ValueType)Values(i)); }
        };

        template<typename VectorType>
        KOKKOS_INLINE_FUNCTION static auto max(const VectorType& values) ->
            typename std::enable_if<VectorType::Rank == 1, typename VectorType::traits::non_const_value_type>::type
        {
            typedef typename VectorType::traits::execution_space      ExecutionSpace;
            typedef typename VectorType::traits::data_type            DataType;
            typedef typename VectorType::traits::non_const_value_type ValueType;

            MaxFunctor<VectorType> f(values);

            ValueType      max_value = Constants<ValueType>::min();
            Max<ValueType> reducer_scalar(max_value);

            Kokkos::parallel_reduce(Kokkos::RangePolicy<ExecutionSpace>(0, values.extent(0)), f, reducer_scalar);
            fence();

            return max_value;
        }
    }
}

namespace Kokkos
{
    namespace Extension
    {
        template<class ExecutionSpace>
        using mdrange_type = Kokkos::MDRangePolicy<ExecutionSpace, Kokkos::Rank<2>, Kokkos::IndexType<size_type>>;

        template<class ExecutionSpace>
        using point_type = typename mdrange_type<ExecutionSpace>::point_type;
    }
}

namespace Kokkos
{
    namespace Extension
    {
        template<typename DataType, class ExecutionSpace>
        std::ostream& operator<<(std::ostream& s, const Extension::Vector<DataType, ExecutionSpace>& A)
        {
            const size_type n = A.extent(0);

            s << n << "\n";

            for(size_type i = 0; i < n; i++)
                s << A(i) << " "
                  << "\n";

            s << "\n";

            return s;
        }

        template<typename DataType, class ExecutionSpace>
        std::istream& operator>>(std::istream& s, Extension::Vector<DataType, ExecutionSpace>& A)
        {
            size_type n;

            s >> n;

            if(!(n == A.size()))
            {
                Kokkos::resize(A, n);
            }

            for(size_type i = 0; i < n; i++)
                s >> A(i);

            return s;
        }

        namespace Internal
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

#define VECTOR_OPS_FUNCTORS(OP_NAME, OP, ASSIGN_OP)                                                                                                                    \
    template<class RVector, class LhsVector, class RhsVector>                                                                                                          \
    struct VectorVector##OP_NAME##Functor                                                                                                                              \
    {                                                                                                                                                                  \
        static_assert(RVector::Rank == 1, "RVector::Rank != 1");                                                                                                       \
        static_assert(LhsVector::Rank == 1, "LhsVector::Rank != 1");                                                                                                   \
        static_assert(RhsVector::Rank == 1, "RhsVector::Rank != 1");                                                                                                   \
                                                                                                                                                                       \
        typedef typename RVector::size_type    size_type;                                                                                                              \
        typedef typename LhsVector::value_type value_type;                                                                                                             \
                                                                                                                                                                       \
        RVector                        _r;                                                                                                                             \
        typename LhsVector::const_type _lhs;                                                                                                                           \
        typename RhsVector::const_type _rhs;                                                                                                                           \
                                                                                                                                                                       \
        VectorVector##OP_NAME##Functor(const RVector& r, const LhsVector& lhs, const RhsVector& rhs) : _r(r), _lhs(lhs), _rhs(rhs) {}                                  \
                                                                                                                                                                       \
        KOKKOS_INLINE_FUNCTION void operator()(const size_type i) const { _r(i) = _lhs(i) OP _rhs(i); }                                                                \
    };                                                                                                                                                                 \
                                                                                                                                                                       \
    template<class RVector, class LhsVector>                                                                                                                           \
    struct VectorScalar##OP_NAME##Functor                                                                                                                              \
    {                                                                                                                                                                  \
        static_assert(RVector::Rank == 1, "RVector::Rank != 1");                                                                                                       \
        static_assert(LhsVector::Rank == 1, "LhsVector::Rank != 1");                                                                                                   \
                                                                                                                                                                       \
        typedef typename RVector::size_type              size_type;                                                                                                    \
        typedef typename LhsVector::non_const_value_type value_type;                                                                                                   \
                                                                                                                                                                       \
        RVector                        _r;                                                                                                                             \
        typename LhsVector::const_type _lhs;                                                                                                                           \
        value_type                     _rhs;                                                                                                                           \
                                                                                                                                                                       \
        VectorScalar##OP_NAME##Functor(const RVector& r, const LhsVector& lhs, const value_type& rhs) : _r(r), _lhs(lhs), _rhs(rhs) {}                                 \
                                                                                                                                                                       \
        KOKKOS_INLINE_FUNCTION void operator()(const size_type i) const { _r(i) = _lhs(i) OP _rhs; }                                                                   \
    };                                                                                                                                                                 \
                                                                                                                                                                       \
    template<class RVector, class RhsVector>                                                                                                                           \
    struct ScalarVector##OP_NAME##Functor                                                                                                                              \
    {                                                                                                                                                                  \
        static_assert(RVector::Rank == 1, "RVector::Rank != 1");                                                                                                       \
        static_assert(RhsVector::Rank == 1, "RhsVector::Rank != 1");                                                                                                   \
                                                                                                                                                                       \
        typedef typename RVector::size_type              size_type;                                                                                                    \
        typedef typename RhsVector::non_const_value_type value_type;                                                                                                   \
                                                                                                                                                                       \
        RVector                        _r;                                                                                                                             \
        value_type                     _lhs;                                                                                                                           \
        typename RhsVector::const_type _rhs;                                                                                                                           \
                                                                                                                                                                       \
        ScalarVector##OP_NAME##Functor(const RVector& r, const value_type& lhs, const RhsVector& rhs) : _r(r), _lhs(lhs), _rhs(rhs) {}                                 \
                                                                                                                                                                       \
        KOKKOS_INLINE_FUNCTION void operator()(const size_type i) const { _r(i) = _lhs OP _rhs(i); }                                                                   \
    };                                                                                                                                                                 \
                                                                                                                                                                       \
    template<class LhsVector, class RhsVector>                                                                                                                         \
    struct VectorVector##OP_NAME##AssignFunctor                                                                                                                        \
    {                                                                                                                                                                  \
        static_assert(LhsVector::Rank == 1, "LhsVector::Rank != 1");                                                                                                   \
        static_assert(RhsVector::Rank == 1, "RhsVector::Rank != 1");                                                                                                   \
                                                                                                                                                                       \
        typedef typename LhsVector::size_type  size_type;                                                                                                              \
        typedef typename LhsVector::value_type value_type;                                                                                                             \
                                                                                                                                                                       \
        LhsVector                      _lhs;                                                                                                                           \
        typename RhsVector::const_type _rhs;                                                                                                                           \
                                                                                                                                                                       \
        VectorVector##OP_NAME##AssignFunctor(const LhsVector& lhs, const RhsVector& rhs) : _lhs(lhs), _rhs(rhs) {}                                                     \
                                                                                                                                                                       \
        KOKKOS_INLINE_FUNCTION void operator()(const size_type i) const { _lhs(i) ASSIGN_OP _rhs(i); }                                                                 \
    };                                                                                                                                                                 \
                                                                                                                                                                       \
    template<class LhsVector>                                                                                                                                          \
    struct VectorScalar##OP_NAME##AssignFunctor                                                                                                                        \
    {                                                                                                                                                                  \
        static_assert(LhsVector::Rank == 1, "LhsVector::Rank != 1");                                                                                                   \
                                                                                                                                                                       \
        typedef typename LhsVector::size_type            size_type;                                                                                                    \
        typedef typename LhsVector::non_const_value_type value_type;                                                                                                   \
                                                                                                                                                                       \
        LhsVector  _lhs;                                                                                                                                               \
        value_type _rhs;                                                                                                                                               \
                                                                                                                                                                       \
        VectorScalar##OP_NAME##AssignFunctor(const LhsVector& lhs, const value_type& rhs) : _lhs(lhs), _rhs(rhs) {}                                                    \
                                                                                                                                                                       \
        KOKKOS_INLINE_FUNCTION void operator()(const size_type i) const { _lhs(i) ASSIGN_OP _rhs; }                                                                    \
    };                                                                                                                                                                 \
                                                                                                                                                                       \
    template<typename DataType, class ExecutionSpace>                                                                                                                  \
    __inline static Vector<DataType, ExecutionSpace> operator OP(const Vector<DataType, ExecutionSpace>& lhs, const Vector<DataType, ExecutionSpace>& rhs)             \
    {                                                                                                                                                                  \
        const size_type n = lhs.extent(0);                                                                                                                             \
                                                                                                                                                                       \
        assert(n == rhs.extent(0));                                                                                                                                    \
                                                                                                                                                                       \
        Vector<DataType, ExecutionSpace> r(lhs.label() + #OP + rhs.label(), n);                                                                                        \
                                                                                                                                                                       \
        Internal::VectorVector##OP_NAME##Functor<Vector<DataType, ExecutionSpace>, Vector<DataType, ExecutionSpace>, Vector<DataType, ExecutionSpace>> f(r, lhs, rhs); \
                                                                                                                                                                       \
        Kokkos::RangePolicy<ExecutionSpace> policy(0, n);                                                                                                              \
                                                                                                                                                                       \
        Kokkos::parallel_for("V_" #OP_NAME, policy, f);                                                                                                                \
                                                                                                                                                                       \
        return r;                                                                                                                                                      \
    }                                                                                                                                                                  \
                                                                                                                                                                       \
    template<typename DataType, class ExecutionSpace>                                                                                                                  \
    __inline static Vector<DataType, ExecutionSpace> operator OP(const DataType& lhs, const Vector<DataType, ExecutionSpace>& rhs)                                     \
    {                                                                                                                                                                  \
        const size_type n = lhs.extent(0);                                                                                                                             \
                                                                                                                                                                       \
        assert(n == rhs.extent(0));                                                                                                                                    \
                                                                                                                                                                       \
        Vector<DataType, ExecutionSpace> r(lhs.label() + #OP + rhs.label(), n);                                                                                        \
                                                                                                                                                                       \
        Internal::ScalarVector##OP_NAME##Functor<Vector<DataType, ExecutionSpace>, Vector<DataType, ExecutionSpace>> f(r, lhs, rhs);                                   \
                                                                                                                                                                       \
        Kokkos::RangePolicy<ExecutionSpace> policy(0, n);                                                                                                              \
                                                                                                                                                                       \
        Kokkos::parallel_for("V_" #OP_NAME, policy, f);                                                                                                                \
                                                                                                                                                                       \
        return r;                                                                                                                                                      \
    }                                                                                                                                                                  \
                                                                                                                                                                       \
    template<typename DataType, class ExecutionSpace>                                                                                                                  \
    __inline static Vector<DataType, ExecutionSpace> operator OP(const Vector<DataType, ExecutionSpace>& lhs, const DataType& rhs)                                     \
    {                                                                                                                                                                  \
        const size_type n = lhs.extent(0);                                                                                                                             \
                                                                                                                                                                       \
        assert(n == rhs.extent(0));                                                                                                                                    \
                                                                                                                                                                       \
        Vector<DataType, ExecutionSpace> r(lhs.label() + #OP + rhs.label(), n);                                                                                        \
                                                                                                                                                                       \
        Internal::VectorScalar##OP_NAME##Functor<Vector<DataType, ExecutionSpace>, Vector<DataType, ExecutionSpace>> f(r, lhs, rhs);                                   \
                                                                                                                                                                       \
        Kokkos::RangePolicy<ExecutionSpace> policy(0, n);                                                                                                              \
                                                                                                                                                                       \
        Kokkos::parallel_for("V_" #OP_NAME, policy, f);                                                                                                                \
                                                                                                                                                                       \
        return r;                                                                                                                                                      \
    }                                                                                                                                                                  \
                                                                                                                                                                       \
    template<typename DataType, class ExecutionSpace>                                                                                                                  \
    __inline static Vector<DataType, ExecutionSpace> operator ASSIGN_OP(Vector<DataType, ExecutionSpace>& lhs, const Vector<DataType, ExecutionSpace>& rhs)            \
    {                                                                                                                                                                  \
        const size_type n = lhs.extent(0);                                                                                                                             \
                                                                                                                                                                       \
        assert(n == rhs.extent(0));                                                                                                                                    \
                                                                                                                                                                       \
        Vector<DataType, ExecutionSpace> r(lhs.label() + #ASSIGN_OP + rhs.label(), n);                                                                                 \
                                                                                                                                                                       \
        Internal::VectorVector##OP_NAME##AssignFunctor<Vector<DataType, ExecutionSpace>, Vector<DataType, ExecutionSpace>> f(lhs, rhs);                                \
                                                                                                                                                                       \
        Kokkos::RangePolicy<ExecutionSpace> policy(0, n);                                                                                                              \
                                                                                                                                                                       \
        Kokkos::parallel_for("V_" #OP_NAME "Assign", policy, f);                                                                                                       \
                                                                                                                                                                       \
        return r;                                                                                                                                                      \
    }                                                                                                                                                                  \
                                                                                                                                                                       \
    template<typename DataType, class ExecutionSpace>                                                                                                                  \
    __inline static Vector<DataType, ExecutionSpace> operator ASSIGN_OP(Vector<DataType, ExecutionSpace>& lhs, const DataType& rhs)                                    \
    {                                                                                                                                                                  \
        const size_type n = lhs.extent(0);                                                                                                                             \
                                                                                                                                                                       \
        assert(n == rhs.extent(0));                                                                                                                                    \
                                                                                                                                                                       \
        Vector<DataType, ExecutionSpace> r(lhs.label() + #ASSIGN_OP + std::to_string(rhs), n);                                                                         \
                                                                                                                                                                       \
        Internal::VectorScalar##OP_NAME##AssignFunctor<Vector<DataType, ExecutionSpace>> f(lhs, rhs);                                                                  \
                                                                                                                                                                       \
        Kokkos::RangePolicy<ExecutionSpace> policy(0, n);                                                                                                              \
                                                                                                                                                                       \
        Kokkos::parallel_for("V_" #OP_NAME "Scalar", policy, f);                                                                                                       \
                                                                                                                                                                       \
        return r;                                                                                                                                                      \
    }

            VECTOR_OPS_FUNCTORS(Plus, +, +=)
            VECTOR_OPS_FUNCTORS(Minus, -, -=)
            VECTOR_OPS_FUNCTORS(Multiply, *, *=)
            VECTOR_OPS_FUNCTORS(Divide, /, /=)

#undef VECTOR_OPS_FUNCTORS

            template<class LhsVector, class RhsVector>
            struct VectorInnerProductFunctor
            {
                static_assert(LhsVector::Rank == 1, "LhsVector::Rank != 1");
                static_assert(RhsVector::Rank == 1, "RhsVector::Rank != 1");

                typedef typename LhsVector::size_type            size_type;
                typedef typename LhsVector::non_const_value_type value_type;

                typename LhsVector::const_type _lhs;
                typename RhsVector::const_type _rhs;

                VectorInnerProductFunctor(const LhsVector& lhs, const RhsVector& rhs) : _lhs(lhs), _rhs(rhs) {}

                KOKKOS_INLINE_FUNCTION void operator()(const size_type i, value_type& sum) const { sum += _lhs(i) * _rhs(i); }
            };

            template<class RVector, class LhsVector, class RhsVector>
            struct VectorOuterProductFunctor
            {
                static_assert(RVector::Rank == 2, "RVector::Rank != 2");
                static_assert(LhsVector::Rank == 1, "LhsVector::Rank != 1");
                static_assert(RhsVector::Rank == 1, "RhsVector::Rank != 1");

                typedef typename LhsVector::size_type            size_type;
                typedef typename LhsVector::non_const_value_type value_type;

                RVector                        _r;
                typename LhsVector::const_type _lhs;
                typename RhsVector::const_type _rhs;

                VectorOuterProductFunctor(const RVector& r, const LhsVector& lhs, const RhsVector& rhs) : _r(r), _lhs(lhs), _rhs(rhs) {}

                KOKKOS_INLINE_FUNCTION void operator()(const size_type i, const size_type j) const { _r(i, j) = _lhs(i) * _rhs(j); }
            };

            template<class LhsVector>
            struct VectorNormFunctor
            {
                static_assert(LhsVector::Rank == 1, "LhsVector::Rank != 1");

                typedef typename LhsVector::size_type            size_type;
                typedef typename LhsVector::non_const_value_type value_type;

                typename LhsVector::const_type _lhs;

                VectorNormFunctor(const LhsVector& lhs) : _lhs(lhs) {}

                KOKKOS_INLINE_FUNCTION void operator()(const size_type i, value_type& sum) const { sum += abs(_lhs(i)) * abs(_lhs(i)); }
            };
        }

        template<typename DataType, class ExecutionSpace>
        __inline static auto row(const Matrix<DataType, ExecutionSpace>& A, const size_type& r) ->
            typename std::enable_if<std::is_same_v<typename Matrix<DataType, ExecutionSpace>::traits::array_layout, Kokkos::LayoutLeft>,
                                    Vector<DataType, ExecutionSpace>>::type
        {
            return Kokkos::subview(A, Kokkos::ALL, r);
        }

        template<typename DataType, class ExecutionSpace>
        __inline static auto row(const Matrix<DataType, ExecutionSpace>& A, const size_type& r) ->
            typename std::enable_if<std::is_same_v<typename Matrix<DataType, ExecutionSpace>::traits::array_layout, Kokkos::LayoutRight>,
                                    Vector<DataType, ExecutionSpace>>::type
        {
            return Kokkos::subview(A, r, Kokkos::ALL);
        }

        template<typename DataType, class ExecutionSpace>
        __inline static auto column(const Matrix<DataType, ExecutionSpace>& A, const size_type& c) ->
            typename std::enable_if<std::is_same_v<typename Matrix<DataType, ExecutionSpace>::traits::array_layout, Kokkos::LayoutLeft>,
                                    Vector<DataType, ExecutionSpace, Kokkos::LayoutStride>>::type
        {
            return Kokkos::subview(A, c, Kokkos::ALL);
        }

        template<typename DataType, class ExecutionSpace>
        __inline static auto column(const Matrix<DataType, ExecutionSpace>& A, const size_type& c) ->
            typename std::enable_if<std::is_same_v<typename Matrix<DataType, ExecutionSpace>::traits::array_layout, Kokkos::LayoutRight>,
                                    Vector<DataType, ExecutionSpace, Kokkos::LayoutStride>>::type
        {
            return Kokkos::subview(A, Kokkos::ALL, c);
        }

        template<typename DataType, class ExecutionSpace>
        __inline static DataType norm(const Vector<DataType, ExecutionSpace>& lhs)
        {
            const size_type n = lhs.extent(0);

            Internal::VectorNormFunctor<Vector<DataType, ExecutionSpace>> f(lhs);

            DataType sum;

            Kokkos::RangePolicy<ExecutionSpace> policy(0, n);

            Kokkos::parallel_reduce("V_Norm", policy, f, sum);

            return sqrt(sum);
        }

        template<typename DataType, class ExecutionSpace>
        KOKKOS_INLINE_FUNCTION static DataType inner_product(const Vector<DataType, ExecutionSpace>& lhs, const Vector<DataType, ExecutionSpace>& rhs)
        {
            const size_type n = lhs.extent(0);
            Assert(n == rhs.extent(0));
            DataType sum;

#if defined(__CUDA_ARCH__)
            for(size_type i = 0; i < n; i++)
            {
                sum += lhs(i) * rhs(i);
            }
#else
            Internal::VectorInnerProductFunctor<Vector<DataType, ExecutionSpace>, Vector<DataType, ExecutionSpace>> f(lhs, rhs);

            Kokkos::RangePolicy<ExecutionSpace> policy(0, n);

            Kokkos::parallel_reduce("V_InnerProduct", policy, f, sum);
#endif

            return sum;
        }

        template<typename DataType, class ExecutionSpace>
        __inline static DataType operator%(const Vector<DataType, ExecutionSpace>& lhs, const Vector<DataType, ExecutionSpace>& rhs)
        {
            return inner_product<DataType, ExecutionSpace>(lhs, rhs);
        }

        template<typename DataType, class ExecutionSpace>
        __inline static Matrix<DataType, ExecutionSpace> outer_product(const Vector<DataType, ExecutionSpace>& lhs, const Vector<DataType, ExecutionSpace>& rhs)
        {
            const size_type n = lhs.extent(0);
            const size_type m = rhs.extent(0);
            Assert(n == rhs.extent(0));

            Matrix<DataType, ExecutionSpace> r(lhs.label() + " X " + rhs.label(), n);

            Internal::VectorOuterProductFunctor<Matrix<DataType, ExecutionSpace>, Vector<DataType, ExecutionSpace>, Vector<DataType, ExecutionSpace>> f(r, lhs, rhs);

            mdrange_type<ExecutionSpace> policy(point_type<ExecutionSpace> {{0, 0}}, point_type<ExecutionSpace> {{n, m}});

            Kokkos::parallel_for("V_OuterProduct", policy, f);

            return r;
        }

        template<typename DataType, class ExecutionSpace>
        __inline static Matrix<DataType, ExecutionSpace> operator^(const Vector<DataType, ExecutionSpace>& lhs, const Vector<DataType, ExecutionSpace>& rhs)
        {
            return outer_product<DataType, ExecutionSpace>(lhs, rhs);
        }

        using Internal::operator+;
        using Internal::operator-;
        using Internal::operator*;
        using Internal::operator/;
        using Internal::operator+=;
        using Internal::operator-=;
        using Internal::operator*=;
        using Internal::operator/=;
    }

    using Extension::operator+;
    using Extension::operator-;
    using Extension::operator*;
    using Extension::operator/;
    using Extension::operator+=;
    using Extension::operator-=;
    using Extension::operator*=;
    using Extension::operator/=;

    using Extension::operator%;
    using Extension::operator^;

    using Extension::row;
    using Extension::column;
    using Extension::norm;
    using Extension::inner_product;
    using Extension::outer_product;
}

namespace Kokkos
{
    namespace Extension
    {
        template<typename DataType, class ExecutionSpace>
        std::ostream& operator<<(std::ostream& s, const Extension::Matrix<DataType, ExecutionSpace>& A)
        {
            const size_type m = A.extent(0);
            const size_type n = A.extent(1);

            s << m << " " << n << "\n";
            for(size_type i = 0; i < m; i++)
            {
                for(size_type j = 0; j < n; j++)
                {
                    s << A(i, j) << " ";
                }
                s << "\n";
            }

            return s;
        }

        template<typename DataType, class ExecutionSpace>
        std::istream& operator>>(std::istream& s, Extension::Matrix<DataType, ExecutionSpace>& A)
        {
            size_type m, n;

            s >> m >> n;

            if(!(m == A.extent(0) && n == A.extent(1)))
            {
                Kokkos::resize(A, m, n);
            }

            for(size_type i = 0; i < m; i++)
                for(size_type j = 0; j < n; j++)
                {
                    s >> A(i, j);
                }

            return s;
        }

        namespace Internal
        {
#define MATRIX_OPS_FUNCTORS(OP_NAME, OP, ASSIGN_OP)                                                                                                                    \
    template<class RMatrix, class LhsMatrix, class RhsMatrix>                                                                                                          \
    struct MatrixMatrix##OP_NAME##Functor                                                                                                                              \
    {                                                                                                                                                                  \
        static_assert(RMatrix::Rank == 2, "RMatrix::Rank != 2");                                                                                                       \
        static_assert(LhsMatrix::Rank == 2, "LhsMatrix::Rank != 2");                                                                                                   \
        static_assert(RhsMatrix::Rank == 2, "RhsMatrix::Rank != 2");                                                                                                   \
                                                                                                                                                                       \
        typedef typename RMatrix::size_type    size_type;                                                                                                              \
        typedef typename LhsMatrix::value_type value_type;                                                                                                             \
                                                                                                                                                                       \
        RMatrix                        _r;                                                                                                                             \
        typename LhsMatrix::const_type _lhs;                                                                                                                           \
        typename RhsMatrix::const_type _rhs;                                                                                                                           \
                                                                                                                                                                       \
        MatrixMatrix##OP_NAME##Functor(const RMatrix& r, const LhsMatrix& lhs, const RhsMatrix& rhs) : _r(r), _lhs(lhs), _rhs(rhs) {}                                  \
                                                                                                                                                                       \
        KOKKOS_INLINE_FUNCTION void operator()(const size_type i) const { _r(i) = _lhs(i) OP _rhs(i); }                                                                \
    };                                                                                                                                                                 \
                                                                                                                                                                       \
    template<class RMatrix, class LhsMatrix>                                                                                                                           \
    struct MatrixScalar##OP_NAME##Functor                                                                                                                              \
    {                                                                                                                                                                  \
        static_assert(RMatrix::Rank == 2, "RMatrix::Rank != 2");                                                                                                       \
        static_assert(LhsMatrix::Rank == 2, "LhsMatrix::Rank != 2");                                                                                                   \
                                                                                                                                                                       \
        typedef typename RMatrix::size_type              size_type;                                                                                                    \
        typedef typename LhsMatrix::non_const_value_type value_type;                                                                                                   \
                                                                                                                                                                       \
        RMatrix                        _r;                                                                                                                             \
        typename LhsMatrix::const_type _lhs;                                                                                                                           \
        value_type                     _rhs;                                                                                                                           \
                                                                                                                                                                       \
        MatrixScalar##OP_NAME##Functor(const RMatrix& r, const LhsMatrix& lhs, const value_type& rhs) : _r(r), _lhs(lhs), _rhs(rhs) {}                                 \
                                                                                                                                                                       \
        KOKKOS_INLINE_FUNCTION void operator()(const size_type i) const { _r(i) = _lhs(i) OP _rhs; }                                                                   \
    };                                                                                                                                                                 \
                                                                                                                                                                       \
    template<class RMatrix, class RhsMatrix>                                                                                                                           \
    struct ScalarMatrix##OP_NAME##Functor                                                                                                                              \
    {                                                                                                                                                                  \
        static_assert(RMatrix::Rank == 2, "RMatrix::Rank != 2");                                                                                                       \
        static_assert(RhsMatrix::Rank == 2, "RhsMatrix::Rank != 2");                                                                                                   \
                                                                                                                                                                       \
        typedef typename RMatrix::size_type              size_type;                                                                                                    \
        typedef typename RhsMatrix::non_const_value_type value_type;                                                                                                   \
                                                                                                                                                                       \
        RMatrix                        _r;                                                                                                                             \
        value_type                     _lhs;                                                                                                                           \
        typename RhsMatrix::const_type _rhs;                                                                                                                           \
                                                                                                                                                                       \
        ScalarMatrix##OP_NAME##Functor(const RMatrix& r, const value_type& lhs, const RhsMatrix& rhs) : _r(r), _lhs(lhs), _rhs(rhs) {}                                 \
                                                                                                                                                                       \
        KOKKOS_INLINE_FUNCTION void operator()(const size_type i) const { _r(i) = _lhs OP _rhs(i); }                                                                   \
    };                                                                                                                                                                 \
                                                                                                                                                                       \
    template<class LhsMatrix, class RhsMatrix>                                                                                                                         \
    struct MatrixMatrix##OP_NAME##AssignFunctor                                                                                                                        \
    {                                                                                                                                                                  \
        static_assert(LhsMatrix::Rank == 2, "LhsMatrix::Rank != 2");                                                                                                   \
        static_assert(RhsMatrix::Rank == 2, "RhsMatrix::Rank != 2");                                                                                                   \
                                                                                                                                                                       \
        typedef typename LhsMatrix::size_type  size_type;                                                                                                              \
        typedef typename LhsMatrix::value_type value_type;                                                                                                             \
                                                                                                                                                                       \
        LhsMatrix                      _lhs;                                                                                                                           \
        typename RhsMatrix::const_type _rhs;                                                                                                                           \
                                                                                                                                                                       \
        MatrixMatrix##OP_NAME##AssignFunctor(const LhsMatrix& lhs, const RhsMatrix& rhs) : _lhs(lhs), _rhs(rhs) {}                                                     \
                                                                                                                                                                       \
        KOKKOS_INLINE_FUNCTION void operator()(const size_type i) const { _lhs(i) ASSIGN_OP _rhs(i); }                                                                 \
    };                                                                                                                                                                 \
                                                                                                                                                                       \
    template<class LhsMatrix>                                                                                                                                          \
    struct MatrixScalar##OP_NAME##AssignFunctor                                                                                                                        \
    {                                                                                                                                                                  \
        static_assert(LhsMatrix::Rank == 2, "LhsMatrix::Rank != 2");                                                                                                   \
                                                                                                                                                                       \
        typedef typename LhsMatrix::size_type            size_type;                                                                                                    \
        typedef typename LhsMatrix::non_const_value_type value_type;                                                                                                   \
                                                                                                                                                                       \
        LhsMatrix  _lhs;                                                                                                                                               \
        value_type _rhs;                                                                                                                                               \
                                                                                                                                                                       \
        MatrixScalar##OP_NAME##AssignFunctor(const LhsMatrix& lhs, const value_type& rhs) : _lhs(lhs), _rhs(rhs) {}                                                    \
                                                                                                                                                                       \
        KOKKOS_INLINE_FUNCTION void operator()(const size_type i) const { _lhs(i) ASSIGN_OP _rhs; }                                                                    \
    };                                                                                                                                                                 \
                                                                                                                                                                       \
    template<typename DataType, class ExecutionSpace>                                                                                                                  \
    __inline static Matrix<DataType, ExecutionSpace> operator OP(const Matrix<DataType, ExecutionSpace>& lhs, const Matrix<DataType, ExecutionSpace>& rhs)             \
    {                                                                                                                                                                  \
        const size_type m = lhs.extent(0);                                                                                                                             \
        const size_type n = lhs.extent(1);                                                                                                                             \
                                                                                                                                                                       \
        assert(n == rhs.extent(0));                                                                                                                                    \
                                                                                                                                                                       \
        Matrix<DataType, ExecutionSpace> r(lhs.label() + #OP + rhs.label(), n);                                                                                        \
                                                                                                                                                                       \
        Internal::MatrixMatrix##OP_NAME##Functor<Matrix<DataType, ExecutionSpace>, Matrix<DataType, ExecutionSpace>, Matrix<DataType, ExecutionSpace>> f(r, lhs, rhs); \
                                                                                                                                                                       \
        mdrange_type<ExecutionSpace> policy(point_type<ExecutionSpace> {{0, 0}}, point_type<ExecutionSpace> {{m, n}});                                                 \
                                                                                                                                                                       \
        Kokkos::parallel_for("V_" #OP_NAME, policy, f);                                                                                                                \
                                                                                                                                                                       \
        return r;                                                                                                                                                      \
    }                                                                                                                                                                  \
                                                                                                                                                                       \
    template<typename DataType, class ExecutionSpace>                                                                                                                  \
    __inline static Matrix<DataType, ExecutionSpace> operator OP(const DataType& lhs, const Matrix<DataType, ExecutionSpace>& rhs)                                     \
    {                                                                                                                                                                  \
        const size_type m = rhs.extent(0);                                                                                                                             \
        const size_type n = rhs.extent(1);                                                                                                                             \
                                                                                                                                                                       \
        assert(n == rhs.extent(0));                                                                                                                                    \
                                                                                                                                                                       \
        Matrix<DataType, ExecutionSpace> r(lhs.label() + #OP + rhs.label(), n);                                                                                        \
                                                                                                                                                                       \
        Internal::ScalarMatrix##OP_NAME##Functor<Matrix<DataType, ExecutionSpace>, Matrix<DataType, ExecutionSpace>> f(r, lhs, rhs);                                   \
                                                                                                                                                                       \
        mdrange_type<ExecutionSpace> policy(point_type<ExecutionSpace> {{0, 0}}, point_type<ExecutionSpace> {{m, n}});                                                 \
                                                                                                                                                                       \
        Kokkos::parallel_for("V_" #OP_NAME, policy, f);                                                                                                                \
                                                                                                                                                                       \
        return r;                                                                                                                                                      \
    }                                                                                                                                                                  \
                                                                                                                                                                       \
    template<typename DataType, class ExecutionSpace>                                                                                                                  \
    __inline static Matrix<DataType, ExecutionSpace> operator OP(const Matrix<DataType, ExecutionSpace>& lhs, const DataType& rhs)                                     \
    {                                                                                                                                                                  \
        const size_type m = lhs.extent(0);                                                                                                                             \
        const size_type n = lhs.extent(1);                                                                                                                             \
                                                                                                                                                                       \
        assert(n == rhs.extent(0));                                                                                                                                    \
                                                                                                                                                                       \
        Matrix<DataType, ExecutionSpace> r(lhs.label() + #OP + rhs.label(), n);                                                                                        \
                                                                                                                                                                       \
        Internal::MatrixScalar##OP_NAME##Functor<Matrix<DataType, ExecutionSpace>, Matrix<DataType, ExecutionSpace>> f(r, lhs, rhs);                                   \
                                                                                                                                                                       \
        mdrange_type<ExecutionSpace> policy(point_type<ExecutionSpace> {{0, 0}}, point_type<ExecutionSpace> {{m, n}});                                                 \
                                                                                                                                                                       \
        Kokkos::parallel_for("V_" #OP_NAME, policy, f);                                                                                                                \
                                                                                                                                                                       \
        return r;                                                                                                                                                      \
    }                                                                                                                                                                  \
                                                                                                                                                                       \
    template<typename DataType, class ExecutionSpace>                                                                                                                  \
    __inline static Matrix<DataType, ExecutionSpace> operator ASSIGN_OP(Matrix<DataType, ExecutionSpace>& lhs, const Matrix<DataType, ExecutionSpace>& rhs)            \
    {                                                                                                                                                                  \
        const size_type m = lhs.extent(0);                                                                                                                             \
        const size_type n = lhs.extent(1);                                                                                                                             \
                                                                                                                                                                       \
        assert(n == rhs.extent(0));                                                                                                                                    \
                                                                                                                                                                       \
        Matrix<DataType, ExecutionSpace> r(lhs.label() + #ASSIGN_OP + rhs.label(), n);                                                                                 \
                                                                                                                                                                       \
        Internal::MatrixMatrix##OP_NAME##AssignFunctor<Matrix<DataType, ExecutionSpace>, Matrix<DataType, ExecutionSpace>> f(lhs, rhs);                                \
                                                                                                                                                                       \
        mdrange_type<ExecutionSpace> policy(point_type<ExecutionSpace> {{0, 0}}, point_type<ExecutionSpace> {{m, n}});                                                 \
                                                                                                                                                                       \
        Kokkos::parallel_for("V_" #OP_NAME "Assign", policy, f);                                                                                                       \
                                                                                                                                                                       \
        return r;                                                                                                                                                      \
    }                                                                                                                                                                  \
                                                                                                                                                                       \
    template<typename DataType, class ExecutionSpace>                                                                                                                  \
    __inline static Matrix<DataType, ExecutionSpace> operator ASSIGN_OP(Matrix<DataType, ExecutionSpace>& lhs, const DataType& rhs)                                    \
    {                                                                                                                                                                  \
        const size_type m = rhs.extent(0);                                                                                                                             \
        const size_type n = rhs.extent(1);                                                                                                                             \
                                                                                                                                                                       \
        assert(n == rhs.extent(0));                                                                                                                                    \
                                                                                                                                                                       \
        Matrix<DataType, ExecutionSpace> r(lhs.label() + #ASSIGN_OP + std::to_string(rhs), n);                                                                         \
                                                                                                                                                                       \
        Internal::MatrixScalar##OP_NAME##AssignFunctor<Matrix<DataType, ExecutionSpace>> f(lhs, rhs);                                                                  \
                                                                                                                                                                       \
        mdrange_type<ExecutionSpace> policy(point_type<ExecutionSpace> {{0, 0}}, point_type<ExecutionSpace> {{m, n}});                                                 \
                                                                                                                                                                       \
        Kokkos::parallel_for("V_" #OP_NAME "Scalar", policy, f);                                                                                                       \
                                                                                                                                                                       \
        return r;                                                                                                                                                      \
    }

            MATRIX_OPS_FUNCTORS(Plus, +, +=)
            MATRIX_OPS_FUNCTORS(Minus, -, -=)

#undef MATRIX_OPS_FUNCTORS

#define MATRIX_OPS_FUNCTORS(OP_NAME, OP, ASSIGN_OP)                                                                                    \
    template<class RMatrix, class LhsMatrix>                                                                                           \
    struct MatrixScalar##OP_NAME##Functor                                                                                              \
    {                                                                                                                                  \
        static_assert(RMatrix::Rank == 2, "RMatrix::Rank != 2");                                                                       \
        static_assert(LhsMatrix::Rank == 2, "LhsMatrix::Rank != 2");                                                                   \
                                                                                                                                       \
        typedef typename RMatrix::size_type              size_type;                                                                    \
        typedef typename LhsMatrix::non_const_value_type value_type;                                                                   \
                                                                                                                                       \
        RMatrix                        _r;                                                                                             \
        typename LhsMatrix::const_type _lhs;                                                                                           \
        value_type                     _rhs;                                                                                           \
                                                                                                                                       \
        MatrixScalar##OP_NAME##Functor(const RMatrix& r, const LhsMatrix& lhs, const value_type& rhs) : _r(r), _lhs(lhs), _rhs(rhs) {} \
                                                                                                                                       \
        KOKKOS_INLINE_FUNCTION void operator()(const size_type i) const { _r(i) = _lhs(i) OP _rhs; }                                   \
    };                                                                                                                                 \
                                                                                                                                       \
    template<class RMatrix, class RhsMatrix>                                                                                           \
    struct ScalarMatrix##OP_NAME##Functor                                                                                              \
    {                                                                                                                                  \
        static_assert(RMatrix::Rank == 2, "RMatrix::Rank != 2");                                                                       \
        static_assert(RhsMatrix::Rank == 2, "RhsMatrix::Rank != 2");                                                                   \
                                                                                                                                       \
        typedef typename RMatrix::size_type              size_type;                                                                    \
        typedef typename RhsMatrix::non_const_value_type value_type;                                                                   \
                                                                                                                                       \
        RMatrix                        _r;                                                                                             \
        value_type                     _lhs;                                                                                           \
        typename RhsMatrix::const_type _rhs;                                                                                           \
                                                                                                                                       \
        ScalarMatrix##OP_NAME##Functor(const RMatrix& r, const value_type& lhs, const RhsMatrix& rhs) : _r(r), _lhs(lhs), _rhs(rhs) {} \
                                                                                                                                       \
        KOKKOS_INLINE_FUNCTION void operator()(const size_type i) const { _r(i) = _lhs OP _rhs(i); }                                   \
    };                                                                                                                                 \
                                                                                                                                       \
    template<class LhsMatrix>                                                                                                          \
    struct MatrixScalar##OP_NAME##AssignFunctor                                                                                        \
    {                                                                                                                                  \
        static_assert(LhsMatrix::Rank == 2, "LhsMatrix::Rank != 2");                                                                   \
                                                                                                                                       \
        typedef typename LhsMatrix::size_type            size_type;                                                                    \
        typedef typename LhsMatrix::non_const_value_type value_type;                                                                   \
                                                                                                                                       \
        LhsMatrix  _lhs;                                                                                                               \
        value_type _rhs;                                                                                                               \
                                                                                                                                       \
        MatrixScalar##OP_NAME##AssignFunctor(const LhsMatrix& lhs, const value_type& rhs) : _lhs(lhs), _rhs(rhs) {}                    \
                                                                                                                                       \
        KOKKOS_INLINE_FUNCTION void operator()(const size_type i) const { _lhs(i) ASSIGN_OP _rhs; }                                    \
    };                                                                                                                                 \
                                                                                                                                       \
    template<typename DataType, class ExecutionSpace>                                                                                  \
    __inline static Matrix<DataType, ExecutionSpace> operator OP(const DataType& lhs, const Matrix<DataType, ExecutionSpace>& rhs)     \
    {                                                                                                                                  \
        const size_type m = rhs.extent(0);                                                                                             \
        const size_type n = rhs.extent(1);                                                                                             \
                                                                                                                                       \
        assert(n == rhs.extent(0));                                                                                                    \
                                                                                                                                       \
        Matrix<DataType, ExecutionSpace> r(lhs.label() + #OP + rhs.label(), n);                                                        \
                                                                                                                                       \
        Internal::ScalarMatrix##OP_NAME##Functor<Matrix<DataType, ExecutionSpace>, Matrix<DataType, ExecutionSpace>> f(r, lhs, rhs);   \
                                                                                                                                       \
        mdrange_type<ExecutionSpace> policy(point_type<ExecutionSpace> {{0, 0}}, point_type<ExecutionSpace> {{m, n}});                 \
                                                                                                                                       \
        Kokkos::parallel_for("V_" #OP_NAME, policy, f);                                                                                \
                                                                                                                                       \
        return r;                                                                                                                      \
    }                                                                                                                                  \
                                                                                                                                       \
    template<typename DataType, class ExecutionSpace>                                                                                  \
    __inline static Matrix<DataType, ExecutionSpace> operator OP(const Matrix<DataType, ExecutionSpace>& lhs, const DataType& rhs)     \
    {                                                                                                                                  \
        const size_type m = lhs.extent(0);                                                                                             \
        const size_type n = lhs.extent(1);                                                                                             \
                                                                                                                                       \
        assert(n == rhs.extent(0));                                                                                                    \
                                                                                                                                       \
        Matrix<DataType, ExecutionSpace> r(lhs.label() + #OP + rhs.label(), n);                                                        \
                                                                                                                                       \
        Internal::MatrixScalar##OP_NAME##Functor<Matrix<DataType, ExecutionSpace>, Matrix<DataType, ExecutionSpace>> f(r, lhs, rhs);   \
                                                                                                                                       \
        mdrange_type<ExecutionSpace> policy(point_type<ExecutionSpace> {{0, 0}}, point_type<ExecutionSpace> {{m, n}});                 \
                                                                                                                                       \
        Kokkos::parallel_for("V_" #OP_NAME, policy, f);                                                                                \
                                                                                                                                       \
        return r;                                                                                                                      \
    }                                                                                                                                  \
                                                                                                                                       \
    template<typename DataType, class ExecutionSpace>                                                                                  \
    __inline static Matrix<DataType, ExecutionSpace> operator ASSIGN_OP(Matrix<DataType, ExecutionSpace>& lhs, const DataType& rhs)    \
    {                                                                                                                                  \
        const size_type m = rhs.extent(0);                                                                                             \
        const size_type n = rhs.extent(1);                                                                                             \
                                                                                                                                       \
        assert(n == rhs.extent(0));                                                                                                    \
                                                                                                                                       \
        Matrix<DataType, ExecutionSpace> r(lhs.label() + #ASSIGN_OP + std::to_string(rhs), n);                                         \
                                                                                                                                       \
        Internal::MatrixScalar##OP_NAME##AssignFunctor<Matrix<DataType, ExecutionSpace>> f(lhs, rhs);                                  \
                                                                                                                                       \
        mdrange_type<ExecutionSpace> policy(point_type<ExecutionSpace> {{0, 0}}, point_type<ExecutionSpace> {{m, n}});                 \
                                                                                                                                       \
        Kokkos::parallel_for("V_" #OP_NAME "Scalar", policy, f);                                                                       \
                                                                                                                                       \
        return r;                                                                                                                      \
    }
            MATRIX_OPS_FUNCTORS(Multiply, *, *=)
            MATRIX_OPS_FUNCTORS(Divide, /, /=)

#undef MATRIX_OPS_FUNCTORS

            template<class RMatrix, class LhsMatrix, class RhsMatrix>
            struct MatrixMatrixMultiplyFunctor
            {
                static_assert(RMatrix::Rank == 2, "RVector::Rank != 2");
                static_assert(LhsMatrix::Rank == 2, "LhsVector::Rank != 2");
                static_assert(RhsMatrix::Rank == 2, "RhsVector::Rank != 2");

                typedef typename LhsMatrix::size_type            size_type;
                typedef typename LhsMatrix::const_value_type     const_value_type;
                typedef typename LhsMatrix::non_const_value_type value_type;

                RMatrix                        _r;
                typename LhsMatrix::const_type _lhs;
                typename RhsMatrix::const_type _rhs;
                const_value_type               n;

                MatrixMatrixMultiplyFunctor(const RMatrix& r, const LhsMatrix& lhs, const RhsMatrix& rhs) : _r(r), _lhs(lhs), _rhs(rhs), n(lhs.extent(1)) {}

                KOKKOS_INLINE_FUNCTION void operator()(const size_type i, const size_type k) const
                {
                    value_type sum;

                    for(size_type j = 0; j < n; ++j)
                    {
                        sum += _lhs(i, j) * _rhs(j, k);
                    }

                    _r(i, k) = sum;
                }
            };

            template<class RVector, class LhsMatrix, class RhsVector>
            struct MatrixVectorMultiplyFunctor
            {
                static_assert(RVector::Rank == 1, "RVector::Rank != 1");
                static_assert(LhsMatrix::Rank == 2, "LhsVector::Rank != 2");
                static_assert(RhsVector::Rank == 1, "RhsVector::Rank != 1");

                typedef typename LhsMatrix::size_type            size_type;
                typedef typename LhsMatrix::const_value_type     const_value_type;
                typedef typename LhsMatrix::non_const_value_type value_type;

                RVector                        _r;
                typename LhsMatrix::const_type _lhs;
                typename RhsVector::const_type _rhs;
                const_value_type               n;

                MatrixVectorMultiplyFunctor(const RVector& r, const LhsMatrix& lhs, const RhsVector& rhs) : _r(r), _lhs(lhs), _rhs(rhs), n(rhs.extent(0)) {}

                KOKKOS_INLINE_FUNCTION void operator()(const size_type i) const
                {
                    value_type sum = 0.0;

                    for(size_type j = 0; j < n; ++j)
                    {
                        sum += _lhs(i, j) * _rhs(j);
                    }

                    _r(i) = sum;
                }
            };

            template<class RMatrix, class LhsMatrix>
            struct MatrixTransposeFunctor
            {
                static_assert(RMatrix::Rank == 2, "RVector::Rank != 2");
                static_assert(LhsMatrix::Rank == 2, "LhsVector::Rank != 2");

                typedef typename LhsMatrix::size_type            size_type;
                typedef typename LhsMatrix::const_value_type     const_value_type;
                typedef typename LhsMatrix::non_const_value_type value_type;

                RMatrix                        _r;
                typename LhsMatrix::const_type _lhs;

                MatrixTransposeFunctor(const RMatrix& r, const LhsMatrix& lhs) : _r(r), _lhs(lhs) {}

                KOKKOS_INLINE_FUNCTION void operator()(const size_type i, const size_type j) const { _r(j, j) = _lhs(i, j); }
            };

            template<class LhsMatrix>
            struct MatrixNormFunctor
            {
                static_assert(LhsMatrix::Rank == 2, "LhsVector::Rank != 2");

                typedef typename LhsMatrix::size_type            size_type;
                typedef typename LhsMatrix::non_const_value_type value_type;

                typename LhsMatrix::const_type _lhs;

                MatrixNormFunctor(const LhsMatrix& lhs) : _lhs(lhs) {}

                KOKKOS_INLINE_FUNCTION void operator()(const size_type i, const size_type j, value_type& sum) const { sum += abs(_lhs(i, j)) * abs(_lhs(i, j)); }
            };
        }

        using Internal::operator+;
        using Internal::operator-;
        using Internal::operator*;
        using Internal::operator/;
        using Internal::operator+=;
        using Internal::operator-=;
        using Internal::operator*=;
        using Internal::operator/=;

        template<typename DataType, class ExecutionSpace>
        __inline static Matrix<DataType, ExecutionSpace> operator*(const Matrix<DataType, ExecutionSpace>& lhs, const Matrix<DataType, ExecutionSpace>& rhs)
        {
            const size_type m = lhs.extent(0);
            const size_type n = lhs.extent(1);
            const size_type k = rhs.extent(1);

            Assert(n == rhs.extent(0));

            Matrix<DataType, ExecutionSpace> r(lhs.label() + " * " + rhs.label(), m, k);

            Internal::MatrixMatrixMultiplyFunctor<Matrix<DataType, ExecutionSpace>, Matrix<DataType, ExecutionSpace>, Matrix<DataType, ExecutionSpace>> f(r, lhs, rhs);

            mdrange_type<ExecutionSpace> policy(point_type<ExecutionSpace> {{0, 0}}, point_type<ExecutionSpace> {{m, k}});

            Kokkos::parallel_for("M_Multiply", policy, f);

            return r;
        }

        template<typename DataType, class ExecutionSpace>
        __inline static Vector<DataType, ExecutionSpace> operator*(const Matrix<DataType, ExecutionSpace>& lhs, const Vector<DataType, ExecutionSpace>& rhs)
        {
            const size_type m = lhs.extent(0);
            const size_type n = lhs.extent(1);

            Assert(n == rhs.extent(0));

            Vector<DataType, ExecutionSpace> r(lhs.label() + " * " + rhs.label(), m);

            Internal::MatrixVectorMultiplyFunctor<Vector<DataType, ExecutionSpace>, Matrix<DataType, ExecutionSpace>, Vector<DataType, ExecutionSpace>> f(r, lhs, rhs);

            Kokkos::RangePolicy<ExecutionSpace> policy(0, m);

            Kokkos::parallel_for("M_Multiply", policy, f);

            return r;
        }

        template<typename DataType, class ExecutionSpace>
        __inline static DataType norm(const Matrix<DataType, ExecutionSpace>& lhs)
        {
            const size_type m = lhs.extent(0);
            const size_type n = lhs.extent(1);

            Internal::MatrixNormFunctor<Vector<DataType, ExecutionSpace>> f(lhs);

            DataType sum;

            mdrange_type<ExecutionSpace> policy(point_type<ExecutionSpace> {{0, 0}}, point_type<ExecutionSpace> {{m, n}});

            Kokkos::parallel_reduce("V_Norm", policy, f, sum);

            return sqrt(sum);
        }

        template<typename DataType, class ExecutionSpace>
        __inline static Matrix<DataType, ExecutionSpace> transpose(const Matrix<DataType, ExecutionSpace>& lhs)
        {
            const size_type m = lhs.extent(0);
            const size_type n = lhs.extent(1);

            Assert(n == lhs.extent(0));

            Matrix<DataType, ExecutionSpace> r("t(" + lhs.label() + ")", n, m);

            Internal::MatrixTransposeFunctor<Matrix<DataType, ExecutionSpace>, Matrix<DataType, ExecutionSpace>> f(r, lhs);

            mdrange_type<ExecutionSpace> policy(point_type<ExecutionSpace> {{0, 0}}, point_type<ExecutionSpace> {{m, n}});

            Kokkos::parallel_for("M_Transpose", policy, f);

            return r;
        }

        template<typename DataType, class ExecutionSpace>
        KOKKOS_INLINE_FUNCTION static Vector<DataType, ExecutionSpace> upper_triangular_solve(const Matrix<DataType, ExecutionSpace>& A,
                                                                                              const Vector<DataType, ExecutionSpace>& b)
        {
            const int n = A.extent(0) < A.extent(1) ? A.extent(0) : A.extent(1);

            Vector<DataType, ExecutionSpace> x(new DataType[b], b);

            for(int k = n; k >= 1; --k)
            {
                x(k) /= A(k, k);

                for(int i = 1; i < k; i++)
                {
                    x(i) -= x(k) * A(i, k);
                }
            }

            return x;
        }

        template<typename DataType, class ExecutionSpace>
        KOKKOS_INLINE_FUNCTION static Vector<DataType, ExecutionSpace> lower_triangular_solve(const Matrix<DataType, ExecutionSpace>& A,
                                                                                              const Vector<DataType, ExecutionSpace>& b)
        {
            const int n = A.extent(0) < A.extent(1) ? A.extent(0) : A.extent(1);

            Vector<DataType, ExecutionSpace> x(new DataType[b], b);

            for(int k = 1; k <= n; k++)
            {
                x(k) /= A(k, k);

                for(int i = k + 1; i <= n; i++)
                {
                    x(i) -= x(k) * A(i, k);
                }
            }

            return x;
        }
    }

    using Extension::operator+;
    using Extension::operator-;
    using Extension::operator*;
    using Extension::operator/;
    using Extension::operator+=;
    using Extension::operator-=;
    using Extension::operator*=;
    using Extension::operator/=;

    using Extension::norm;
    using Extension::transpose;
}

namespace Kokkos
{
    namespace LinearAlgebra
    {
        namespace Internal
        {
            struct InitializeCholesky
            {
            };
            struct SolveCholesky
            {
            };

            template<typename DataType, class ExecutionSpace>
            struct CholeskyVectorFunctor
            {
                typedef Extension::Matrix<DataType, ExecutionSpace> Matrix;
                typedef Extension::Vector<DataType, ExecutionSpace> Vector;

                typedef typename Matrix::size_type        size_type;
                typedef typename Matrix::const_value_type const_value_type;

                const size_type _n;
                Matrix          _L;
                Matrix          _A;
                Vector          _b;
                Vector          _x;
                mutable int     isspd;

                CholeskyVectorFunctor(const Matrix& A, const Vector& b, const Vector& x) : _n(A.extent(0)), _L("L", _n, _n), _A(A), _b(b), _x(x), isspd(0)
                {
                    Kokkos::deep_copy(_x, _b);
                }

                KOKKOS_INLINE_FUNCTION void operator()(const InitializeCholesky&, const size_type) const
                {
                    DataType d;
                    DataType s;

                    for(size_type j = 0; j < _n; ++j)
                    {
                        d = DataType(0.0);

                        for(size_type k = 0; k < j; ++k)
                        {
                            s = DataType(0.0);

                            for(int i = 0; i < k; i++)
                            {
                                s += _L(k, i) * _L(j, i);
                            }

                            _L(j, k) = s = (_A(j, k) - s) / _L(k, k);

                            d = d + (s * s);

                            isspd = isspd && (_A(k, j) == _A(j, k));
                        }

                        d = _A(j, j) - d;

                        isspd = isspd && (d > DataType(0.0));

                        _L(j, j) = ::sqrt(d > DataType(0.0) ? d : DataType(0.0));

                        for(size_type k = j + 1; k < _n; ++k)
                        {
                            _L(j, k) = DataType(0.0);
                        }
                    }
                }

                KOKKOS_INLINE_FUNCTION void operator()(const SolveCholesky&, const size_type) const
                {
                    for(size_type k = 0; k < _n; ++k)
                    {
                        for(size_type i = 0; i < k; ++i)
                        {
                            _x(k) -= _x(i) * _L(k, i);
                        }

                        _x(k) /= _L(k, k);
                    }

                    for(size_type k = _n - 1; k >= 0 && k < _n; --k)
                    {
                        for(size_type i = k + 1; i < _n; ++i)
                        {
                            _x(k) -= _x(i) * _L(i, k);
                        }

                        _x(k) /= _L(k, k);
                    }
                }
            };

            template<typename DataType, class ExecutionSpace>
            struct CholeskyMatrixFunctor
            {
                typedef Extension::Matrix<DataType, ExecutionSpace> Matrix;

                typedef typename Matrix::size_type        size_type;
                typedef typename Matrix::const_value_type const_value_type;

                const size_type _n;
                Matrix          _L;
                Matrix          _A;
                Matrix          _B;
                Matrix          _X;
                int             isspd;

                CholeskyMatrixFunctor(const Matrix& A, const Matrix& B, const Matrix& X) : _n(A.extent(0)), _L("L", _n, _n), _A(A), _B(B), _X(X), isspd(0)
                {
                    Kokkos::deep_copy(_X, _B);
                }

                KOKKOS_INLINE_FUNCTION void operator()(const InitializeCholesky&, const size_type) const
                {
                    DataType d;
                    DataType s;

                    for(size_type j = 0; j < _n; ++j)
                    {
                        d = DataType(0.0);

                        for(size_type k = 0; k < j; ++k)
                        {
                            s = DataType(0.0);

                            for(int i = 0; i < k; i++)
                            {
                                s += _L(k, i) * _L(j, i);
                            }

                            _L(j, k) = s = (_A(j, k) - s) / _L(k, k);

                            d = d + s * s;

                            isspd = isspd && (_A(k, j) == _A(j, k));
                        }

                        d = _A(j, j) - d;

                        isspd = isspd && (d > DataType(0.0));

                        _L(j, j) = sqrt(d > DataType(0.0) ? d : DataType(0.0));

                        for(size_type k = j + 1; k < _n; ++k)
                        {
                            _L(j, k) = DataType(0.0);
                        }
                    }
                }

                KOKKOS_INLINE_FUNCTION void operator()(const SolveCholesky&, const size_type) const
                {
                    for(size_type j = 0; j < _n; ++j)
                    {
                        for(size_type k = 0; k < _n; ++k)
                        {
                            for(size_type i = 0; i < k; ++i)
                            {
                                X(k, j) -= X(i, j) * _L(k, i);
                            }

                            X(k, j) /= _L(k, k);
                        }
                    }
                    for(size_type j = 0; j < _n; ++j)
                    {
                        for(size_type k = _n - 1; k >= 0; --k)
                        {
                            for(size_type i = k + 1; i < _n; ++i)
                            {
                                X(k, j) -= X(i, j) * _L(i, k);
                            }

                            X(k, j) /= _L(k, k);
                        }
                    }
                }
            };
        }

        template<typename DataType, class ExecutionSpace>
        __inline static Extension::Vector<DataType, ExecutionSpace> Cholesky(const Extension::Matrix<DataType, ExecutionSpace>& A,
                                                                             const Extension::Vector<DataType, ExecutionSpace>& b)
        {
            if(A.extent(0) != b.extent(0))
            {
                return Extension::Vector<DataType, ExecutionSpace>("x", b.extent(0));
            }

            Extension::Vector<DataType, ExecutionSpace> x("x", b.extent(0));

            Internal::CholeskyVectorFunctor<DataType, ExecutionSpace> functor(A, b, x);

            Kokkos::RangePolicy<ExecutionSpace, IndexType<size_type>, Internal::InitializeCholesky> initializeCholesky(0, 1);

            Kokkos::parallel_for(initializeCholesky, functor);

            Kokkos::RangePolicy<ExecutionSpace, IndexType<size_type>, Internal::SolveCholesky> solveCholesky(0, 1);

            Kokkos::parallel_for(solveCholesky, functor);

            return x;
        }

        template<typename DataType, class ExecutionSpace>
        __inline static Extension::Matrix<DataType, ExecutionSpace> Cholesky(const Extension::Matrix<DataType, ExecutionSpace>& A,
                                                                             const Extension::Matrix<DataType, ExecutionSpace>& B)
        {
            if(A.extent(0) != B.extent(0))
            {
                return Extension::Matrix<DataType, ExecutionSpace>("X", B.extent(0), B.extent(1));
            }

            Extension::Matrix<DataType, ExecutionSpace> X("X", B.extent(0), B.extent(1));

            Internal::CholeskyMatrixFunctor<DataType, ExecutionSpace> functor(A, B, X);

            Kokkos::RangePolicy<ExecutionSpace, IndexType<size_type>, Internal::InitializeCholesky> initializeCholesky(0, 1);

            Kokkos::parallel_for(initializeCholesky, functor);

            Kokkos::RangePolicy<ExecutionSpace, IndexType<size_type>, Internal::SolveCholesky> solveCholesky(0, 1);

            Kokkos::parallel_for(solveCholesky, functor);

            return X;
        }
    }
}

namespace Kokkos
{
    namespace LinearAlgebra
    {
        namespace Internal
        {
            // template<typename DataType, class ExecutionSpace>
            // class Eigenvalue
            //{
            //    using Vector = Kokkos::Extension::Vector<DataType, ExecutionSpace>;
            //    using Matrix = Kokkos::Extension::Matrix<DataType, ExecutionSpace>;

            //    /** Row and column dimension (square matrix).  */
            //    int n;

            //    int issymmetric; /* boolean*/

            //    /** Arrays for internal storage of eigenvalues. */

            //    Vector d; /* real part */
            //    Vector e; /* img part */

            //    /** Array for internal storage of eigenvectors. */
            //    Matrix V;

            //    /* Array for internal storage of nonsymmetric Hessenberg form.
            //    @serial internal storage of nonsymmetric Hessenberg form.
            //    */
            //    Matrix H;

            //    /* Working storage for nonsymmetric algorithm.
            //    @serial working storage for nonsymmetric algorithm.
            //    */
            //    Vector ort;

            //    // Symmetric Householder reduction to tridiagonal form.

            //    void tred2()
            //    {
            //        //  This is derived from the Algol procedures tred2 by
            //        //  Bowdler, Martin, Reinsch, and Wilkinson, Handbook for
            //        //  Auto. Comp., Vol.ii-Linear Algebra, and the corresponding
            //        //  Fortran subroutine in EISPACK.

            //        for(int j = 0; j < n; j++)
            //        {
            //            d(j) = V(n - 1, j);
            //        }

            //        // Householder reduction to tridiagonal form.

            //        for(int i = n - 1; i > 0; i--)
            //        {
            //            // Scale to avoid under/overflow.

            //            DataType scale = DataType(0.0);
            //            DataType h     = DataType(0.0);
            //            for(int k = 0; k < i; k++)
            //            {
            //                scale = scale + abs(d(k));
            //            }
            //            if(scale == DataType(0.0))
            //            {
            //                e(i) = d(i - 1);
            //                for(int j = 0; j < i; j++)
            //                {
            //                    d(j)    = V(i - 1, j);
            //                    V(i, j) = DataType(0.0);
            //                    V(j, i) = DataType(0.0);
            //                }
            //            }
            //            else
            //            {
            //                // Generate Householder vector.

            //                for(int k = 0; k < i; k++)
            //                {
            //                    d(k) /= scale;
            //                    h += d(k) * d(k);
            //                }
            //                DataType f = d(i - 1);
            //                DataType g = sqrt(h);
            //                if(f > 0)
            //                {
            //                    g = -g;
            //                }
            //                e(i)     = scale * g;
            //                h        = h - f * g;
            //                d(i - 1) = f - g;
            //                for(int j = 0; j < i; j++)
            //                {
            //                    e(j) = DataType(0.0);
            //                }

            //                // Apply similarity transformation to remaining columns.

            //                for(int j = 0; j < i; j++)
            //                {
            //                    f       = d(j);
            //                    V(j, i) = f;
            //                    g       = e(j) + V(j, j) * f;
            //                    for(int k = j + 1; k <= i - 1; k++)
            //                    {
            //                        g += V(k, j) * d(k);
            //                        e(k) += V(k, j) * f;
            //                    }
            //                    e(j) = g;
            //                }
            //                f = DataType(0.0);
            //                for(int j = 0; j < i; j++)
            //                {
            //                    e(j) /= h;
            //                    f += e(j) * d(j);
            //                }
            //                DataType hh = f / (h + h);
            //                for(int j = 0; j < i; j++)
            //                {
            //                    e(j) -= hh * d(j);
            //                }
            //                for(int j = 0; j < i; j++)
            //                {
            //                    f = d(j);
            //                    g = e(j);
            //                    for(int k = j; k <= i - 1; k++)
            //                    {
            //                        V(k, j) -= (f * e(k) + g * d(k));
            //                    }
            //                    d(j)    = V(i - 1, j);
            //                    V(i, j) = DataType(0.0);
            //                }
            //            }
            //            d(i) = h;
            //        }

            //        // Accumulate transformations.

            //        for(int i = 0; i < n - 1; i++)
            //        {
            //            V(n - 1, i) = V(i, i);
            //            V(i, i)     = DataType(1.0);
            //            DataType h  = d(i + 1);
            //            if(h != DataType(0.0))
            //            {
            //                for(int k = 0; k <= i; k++)
            //                {
            //                    d(k) = V(k, i + 1) / h;
            //                }
            //                for(int j = 0; j <= i; j++)
            //                {
            //                    DataType g = DataType(0.0);
            //                    for(int k = 0; k <= i; k++)
            //                    {
            //                        g += V(k, i + 1) * V(k, j);
            //                    }
            //                    for(int k = 0; k <= i; k++)
            //                    {
            //                        V(k, j) -= g * d(k);
            //                    }
            //                }
            //            }
            //            for(int k = 0; k <= i; k++)
            //            {
            //                V(k, i + 1) = DataType(0.0);
            //            }
            //        }
            //        for(int j = 0; j < n; j++)
            //        {
            //            d(j)        = V(n - 1, j);
            //            V(n - 1, j) = DataType(0.0);
            //        }
            //        V(n - 1, n - 1) = DataType(1.0);
            //        e(0)            = DataType(0.0);
            //    }

            //    // Symmetric tridiagonal QL algorithm.

            //    void tql2()
            //    {
            //        //  This is derived from the Algol procedures tql2, by
            //        //  Bowdler, Martin, Reinsch, and Wilkinson, Handbook for
            //        //  Auto. Comp., Vol.ii-Linear Algebra, and the corresponding
            //        //  Fortran subroutine in EISPACK.

            //        for(int i = 1; i < n; i++)
            //        {
            //            e(i - 1) = e(i);
            //        }
            //        e(n - 1) = DataType(0.0);

            //        DataType f    = DataType(0.0);
            //        DataType tst1 = DataType(0.0);
            //        DataType eps  = pow(2.0, -52.0);
            //        for(int l = 0; l < n; l++)
            //        {
            //            // Find small subdiagonal element

            //            tst1  = max(tst1, abs(d(l)) + abs(e(l)));
            //            int m = l;

            //            // Original while-loop from Java code
            //            while(m < n)
            //            {
            //                if(abs(e(m)) <= eps * tst1)
            //                {
            //                    break;
            //                }
            //                m++;
            //            }

            //            // If m == l, d(l) is an eigenvalue,
            //            // otherwise, iterate.

            //            if(m > l)
            //            {
            //                int iter = 0;
            //                do
            //                {
            //                    iter = iter + 1; // (Could check iteration count here.)

            //                    // Compute implicit shift

            //                    DataType g = d(l);
            //                    DataType p = (d(l + 1) - g) / (2.0 * e(l));
            //                    DataType r = hypot(p, DataType(1.0));
            //                    if(p < 0)
            //                    {
            //                        r = -r;
            //                    }
            //                    d(l)         = e(l) / (p + r);
            //                    d(l + 1)     = e(l) * (p + r);
            //                    DataType dl1 = d(l + 1);
            //                    DataType h   = g - d(l);
            //                    for(int i = l + 2; i < n; i++)
            //                    {
            //                        d(i) -= h;
            //                    }
            //                    f = f + h;

            //                    // Implicit QL transformation.

            //                    p            = d(m);
            //                    DataType c   = DataType(1.0);
            //                    DataType c2  = c;
            //                    DataType c3  = c;
            //                    DataType el1 = e(l + 1);
            //                    DataType s   = DataType(0.0);
            //                    DataType s2  = DataType(0.0);
            //                    for(int i = m - 1; i >= l; i--)
            //                    {
            //                        c3       = c2;
            //                        c2       = c;
            //                        s2       = s;
            //                        g        = c * e(i);
            //                        h        = c * p;
            //                        r        = hypot(p, e(i));
            //                        e(i + 1) = s * r;
            //                        s        = e(i) / r;
            //                        c        = p / r;
            //                        p        = c * d(i) - s * g;
            //                        d(i + 1) = h + s * (c * g + s * d(i));

            //                        // Accumulate transformation.

            //                        for(int k = 0; k < n; k++)
            //                        {
            //                            h           = V(k, i + 1);
            //                            V(k, i + 1) = s * V(k, i) + c * h;
            //                            V(k, i)     = c * V(k, i) - s * h;
            //                        }
            //                    }
            //                    p    = -s * s2 * c3 * el1 * e(l) / dl1;
            //                    e(l) = s * p;
            //                    d(l) = c * p;

            //                    // Check for convergence.

            //                } while(abs(e(l)) > eps * tst1);
            //            }
            //            d(l) = d(l) + f;
            //            e(l) = DataType(0.0);
            //        }

            //        // Sort eigenvalues and corresponding vectors.

            //        for(int i = 0; i < n - 1; i++)
            //        {
            //            int      k = i;
            //            DataType p = d(i);
            //            for(int j = i + 1; j < n; j++)
            //            {
            //                if(d(j) < p)
            //                {
            //                    k = j;
            //                    p = d(j);
            //                }
            //            }
            //            if(k != i)
            //            {
            //                d(k) = d(i);
            //                d(i) = p;
            //                for(int j = 0; j < n; j++)
            //                {
            //                    p       = V(j, i);
            //                    V(j, i) = V(j, k);
            //                    V(j, k) = p;
            //                }
            //            }
            //        }
            //    }

            //    // Nonsymmetric reduction to Hessenberg form.

            //    void orthes()
            //    {
            //        //  This is derived from the Algol procedures orthes and ortran,
            //        //  by Martin and Wilkinson, Handbook for Auto. Comp.,
            //        //  Vol.ii-Linear Algebra, and the corresponding
            //        //  Fortran subroutines in EISPACK.

            //        int low  = 0;
            //        int high = n - 1;

            //        for(int m = low + 1; m <= high - 1; m++)
            //        {
            //            // Scale column.

            //            DataType scale = DataType(0.0);
            //            for(int i = m; i <= high; i++)
            //            {
            //                scale = scale + abs(H(i, m - 1));
            //            }
            //            if(scale != DataType(0.0))
            //            {
            //                // Compute Householder transformation.

            //                DataType h = DataType(0.0);
            //                for(int i = high; i >= m; i--)
            //                {
            //                    ort(i) = H(i, m - 1) / scale;
            //                    h += ort(i) * ort(i);
            //                }
            //                DataType g = sqrt(h);
            //                if(ort(m) > 0)
            //                {
            //                    g = -g;
            //                }
            //                h      = h - ort(m) * g;
            //                ort(m) = ort(m) - g;

            //                // Apply Householder similarity transformation
            //                // H = (I-u*u'/h)*H*(I-u*u')/h)

            //                for(int j = m; j < n; j++)
            //                {
            //                    DataType f = DataType(0.0);
            //                    for(int i = high; i >= m; i--)
            //                    {
            //                        f += ort(i) * H(i, j);
            //                    }
            //                    f = f / h;
            //                    for(int i = m; i <= high; i++)
            //                    {
            //                        H(i, j) -= f * ort(i);
            //                    }
            //                }

            //                for(int i = 0; i <= high; i++)
            //                {
            //                    DataType f = DataType(0.0);
            //                    for(int j = high; j >= m; j--)
            //                    {
            //                        f += ort(j) * H(i, j);
            //                    }
            //                    f = f / h;
            //                    for(int j = m; j <= high; j++)
            //                    {
            //                        H(i, j) -= f * ort(j);
            //                    }
            //                }
            //                ort(m)      = scale * ort(m);
            //                H(m, m - 1) = scale * g;
            //            }
            //        }

            //        // Accumulate transformations (Algol's ortran).

            //        for(int i = 0; i < n; i++)
            //        {
            //            for(int j = 0; j < n; j++)
            //            {
            //                V(i, j) = (i == j ? DataType(1.0) : DataType(0.0));
            //            }
            //        }

            //        for(int m = high - 1; m >= low + 1; m--)
            //        {
            //            if(H(m, m - 1) != DataType(0.0))
            //            {
            //                for(int i = m + 1; i <= high; i++)
            //                {
            //                    ort(i) = H(i, m - 1);
            //                }
            //                for(int j = m; j <= high; j++)
            //                {
            //                    DataType g = DataType(0.0);
            //                    for(int i = m; i <= high; i++)
            //                    {
            //                        g += ort(i) * V(i, j);
            //                    }
            //                    // Double division avoids possible underflow
            //                    g = (g / ort(m)) / H(m, m - 1);
            //                    for(int i = m; i <= high; i++)
            //                    {
            //                        V(i, j) += g * ort(i);
            //                    }
            //                }
            //            }
            //        }
            //    }

            //    // Complex scalar division.

            //    DataType cdivr, cdivi;
            //    void     cdiv(DataType xr, DataType xi, DataType yr, DataType yi)
            //    {
            //        DataType r, d;
            //        if(abs(yr) > abs(yi))
            //        {
            //            r     = yi / yr;
            //            d     = yr + r * yi;
            //            cdivr = (xr + r * xi) / d;
            //            cdivi = (xi - r * xr) / d;
            //        }
            //        else
            //        {
            //            r     = yr / yi;
            //            d     = yi + r * yr;
            //            cdivr = (r * xr + xi) / d;
            //            cdivi = (r * xi - xr) / d;
            //        }
            //    }

            //    // Nonsymmetric reduction from Hessenberg to real Schur form.

            //    void hqr2()
            //    {
            //        //  This is derived from the Algol procedure hqr2,
            //        //  by Martin and Wilkinson, Handbook for Auto. Comp.,
            //        //  Vol.ii-Linear Algebra, and the corresponding
            //        //  Fortran subroutine in EISPACK.

            //        // Initialize

            //        int      nn      = this->n;
            //        int      n       = nn - 1;
            //        int      low     = 0;
            //        int      high    = nn - 1;
            //        DataType eps     = pow(2.0, -52.0);
            //        DataType exshift = DataType(0.0);
            //        DataType p = 0, q = 0, r = 0, s = 0, z = 0, t, w, x, y;

            //        // Store roots isolated by balanc and compute matrix norm

            //        DataType norm = DataType(0.0);
            //        for(int i = 0; i < nn; i++)
            //        {
            //            if((i < low) || (i > high))
            //            {
            //                d(i) = H(i, i);
            //                e(i) = DataType(0.0);
            //            }
            //            for(int j = max(i - 1, 0); j < nn; j++)
            //            {
            //                norm = norm + abs(H(i, j));
            //            }
            //        }

            //        // Outer loop over eigenvalue index

            //        int iter = 0;
            //        while(n >= low)
            //        {
            //            // Look for single small sub-diagonal element

            //            int l = n;
            //            while(l > low)
            //            {
            //                s = abs(H(l - 1, l - 1)) + abs(H(l, l));
            //                if(s == DataType(0.0))
            //                {
            //                    s = norm;
            //                }
            //                if(abs(H(l, l - 1)) < eps * s)
            //                {
            //                    break;
            //                }
            //                l--;
            //            }

            //            // Check for convergence
            //            // One root found

            //            if(l == n)
            //            {
            //                H(n, n) = H(n, n) + exshift;
            //                d(n)    = H(n, n);
            //                e(n)    = DataType(0.0);
            //                n--;
            //                iter = 0;

            //                // Two roots found
            //            }
            //            else if(l == n - 1)
            //            {
            //                w               = H(n, n - 1) * H(n - 1, n);
            //                p               = (H(n - 1, n - 1) - H(n, n)) / 2.0;
            //                q               = p * p + w;
            //                z               = sqrt(abs(q));
            //                H(n, n)         = H(n, n) + exshift;
            //                H(n - 1, n - 1) = H(n - 1, n - 1) + exshift;
            //                x               = H(n, n);

            //                // DataType pair

            //                if(q >= 0)
            //                {
            //                    if(p >= 0)
            //                    {
            //                        z = p + z;
            //                    }
            //                    else
            //                    {
            //                        z = p - z;
            //                    }
            //                    d(n - 1) = x + z;
            //                    d(n)     = d(n - 1);
            //                    if(z != DataType(0.0))
            //                    {
            //                        d(n) = x - w / z;
            //                    }
            //                    e(n - 1) = DataType(0.0);
            //                    e(n)     = DataType(0.0);
            //                    x        = H(n, n - 1);
            //                    s        = abs(x) + abs(z);
            //                    p        = x / s;
            //                    q        = z / s;
            //                    r        = sqrt(p * p + q * q);
            //                    p        = p / r;
            //                    q        = q / r;

            //                    // Row modification

            //                    for(int j = n - 1; j < nn; j++)
            //                    {
            //                        z           = H(n - 1, j);
            //                        H(n - 1, j) = q * z + p * H(n, j);
            //                        H(n, j)     = q * H(n, j) - p * z;
            //                    }

            //                    // Column modification

            //                    for(int i = 0; i <= n; i++)
            //                    {
            //                        z           = H(i, n - 1);
            //                        H(i, n - 1) = q * z + p * H(i, n);
            //                        H(i, n)     = q * H(i, n) - p * z;
            //                    }

            //                    // Accumulate transformations

            //                    for(int i = low; i <= high; i++)
            //                    {
            //                        z           = V(i, n - 1);
            //                        V(i, n - 1) = q * z + p * V(i, n);
            //                        V(i, n)     = q * V(i, n) - p * z;
            //                    }

            //                    // Complex pair
            //                }
            //                else
            //                {
            //                    d(n - 1) = x + p;
            //                    d(n)     = x + p;
            //                    e(n - 1) = z;
            //                    e(n)     = -z;
            //                }
            //                n    = n - 2;
            //                iter = 0;

            //                // No convergence yet
            //            }
            //            else
            //            {
            //                // Form shift

            //                x = H(n, n);
            //                y = DataType(0.0);
            //                w = DataType(0.0);
            //                if(l < n)
            //                {
            //                    y = H(n - 1, n - 1);
            //                    w = H(n, n - 1) * H(n - 1, n);
            //                }

            //                // Wilkinson's original ad hoc shift

            //                if(iter == 10)
            //                {
            //                    exshift += x;
            //                    for(int i = low; i <= n; i++)
            //                    {
            //                        H(i, i) -= x;
            //                    }
            //                    s = abs(H(n, n - 1)) + abs(H(n - 1, n - 2));
            //                    x = y = 0.75 * s;
            //                    w     = -0.4375 * s * s;
            //                }

            //                // MATLAB's new ad hoc shift

            //                if(iter == 30)
            //                {
            //                    s = (y - x) / 2.0;
            //                    s = s * s + w;
            //                    if(s > 0)
            //                    {
            //                        s = sqrt(s);
            //                        if(y < x)
            //                        {
            //                            s = -s;
            //                        }
            //                        s = x - w / ((y - x) / 2.0 + s);
            //                        for(int i = low; i <= n; i++)
            //                        {
            //                            H(i, i) -= s;
            //                        }
            //                        exshift += s;
            //                        x = y = w = 0.964;
            //                    }
            //                }

            //                iter = iter + 1; // (Could check iteration count here.)

            //                // Look for two consecutive small sub-diagonal elements

            //                int m = n - 2;
            //                while(m >= l)
            //                {
            //                    z = H(m, m);
            //                    r = x - z;
            //                    s = y - z;
            //                    p = (r * s - w) / H(m + 1, m) + H(m, m + 1);
            //                    q = H(m + 1, m + 1) - z - r - s;
            //                    r = H(m + 2, m + 1);
            //                    s = abs(p) + abs(q) + abs(r);
            //                    p = p / s;
            //                    q = q / s;
            //                    r = r / s;
            //                    if(m == l)
            //                    {
            //                        break;
            //                    }
            //                    if(abs(H(m, m - 1)) * (abs(q) + abs(r)) < eps * (abs(p) * (abs(H(m - 1, m - 1)) + abs(z) + abs(H(m + 1, m + 1)))))
            //                    {
            //                        break;
            //                    }
            //                    m--;
            //                }

            //                for(int i = m + 2; i <= n; i++)
            //                {
            //                    H(i, i - 2) = DataType(0.0);
            //                    if(i > m + 2)
            //                    {
            //                        H(i, i - 3) = DataType(0.0);
            //                    }
            //                }

            //                // Double QR step involving rows l:n and columns m:n

            //                for(int k = m; k <= n - 1; k++)
            //                {
            //                    int notlast = (k != n - 1);
            //                    if(k != m)
            //                    {
            //                        p = H(k, k - 1);
            //                        q = H(k + 1, k - 1);
            //                        r = (notlast ? H(k + 2, k - 1) : DataType(0.0));
            //                        x = abs(p) + abs(q) + abs(r);
            //                        if(x != DataType(0.0))
            //                        {
            //                            p = p / x;
            //                            q = q / x;
            //                            r = r / x;
            //                        }
            //                    }
            //                    if(x == DataType(0.0))
            //                    {
            //                        break;
            //                    }
            //                    s = sqrt(p * p + q * q + r * r);
            //                    if(p < 0)
            //                    {
            //                        s = -s;
            //                    }
            //                    if(s != 0)
            //                    {
            //                        if(k != m)
            //                        {
            //                            H(k, k - 1) = -s * x;
            //                        }
            //                        else if(l != m)
            //                        {
            //                            H(k, k - 1) = -H(k, k - 1);
            //                        }
            //                        p = p + s;
            //                        x = p / s;
            //                        y = q / s;
            //                        z = r / s;
            //                        q = q / p;
            //                        r = r / p;

            //                        // Row modification

            //                        for(int j = k; j < nn; j++)
            //                        {
            //                            p = H(k, j) + q * H(k + 1, j);
            //                            if(notlast)
            //                            {
            //                                p           = p + r * H(k + 2, j);
            //                                H(k + 2, j) = H(k + 2, j) - p * z;
            //                            }
            //                            H(k, j)     = H(k, j) - p * x;
            //                            H(k + 1, j) = H(k + 1, j) - p * y;
            //                        }

            //                        // Column modification

            //                        for(int i = 0; i <= min(n, k + 3); i++)
            //                        {
            //                            p = x * H(i, k) + y * H(i, k + 1);
            //                            if(notlast)
            //                            {
            //                                p           = p + z * H(i, k + 2);
            //                                H(i, k + 2) = H(i, k + 2) - p * r;
            //                            }
            //                            H(i, k)     = H(i, k) - p;
            //                            H(i, k + 1) = H(i, k + 1) - p * q;
            //                        }

            //                        // Accumulate transformations

            //                        for(int i = low; i <= high; i++)
            //                        {
            //                            p = x * V(i, k) + y * V(i, k + 1);
            //                            if(notlast)
            //                            {
            //                                p           = p + z * V(i, k + 2);
            //                                V(i, k + 2) = V(i, k + 2) - p * r;
            //                            }
            //                            V(i, k)     = V(i, k) - p;
            //                            V(i, k + 1) = V(i, k + 1) - p * q;
            //                        }
            //                    } // (s != 0)
            //                } // k loop
            //            } // check convergence
            //        } // while (n >= low)

            //        // Backsubstitute to find vectors of upper triangular form

            //        if(norm == DataType(0.0))
            //        {
            //            return;
            //        }

            //        for(n = nn - 1; n >= 0; n--)
            //        {
            //            p = d(n);
            //            q = e(n);

            //            // DataType vector

            //            if(q == 0)
            //            {
            //                int l   = n;
            //                H(n, n) = DataType(1.0);
            //                for(int i = n - 1; i >= 0; i--)
            //                {
            //                    w = H(i, i) - p;
            //                    r = DataType(0.0);
            //                    for(int j = l; j <= n; j++)
            //                    {
            //                        r = r + H(i, j) * H(j, n);
            //                    }
            //                    if(e(i) < DataType(0.0))
            //                    {
            //                        z = w;
            //                        s = r;
            //                    }
            //                    else
            //                    {
            //                        l = i;
            //                        if(e(i) == DataType(0.0))
            //                        {
            //                            if(w != DataType(0.0))
            //                            {
            //                                H(i, n) = -r / w;
            //                            }
            //                            else
            //                            {
            //                                H(i, n) = -r / (eps * norm);
            //                            }

            //                            // Solve real equations
            //                        }
            //                        else
            //                        {
            //                            x       = H(i, i + 1);
            //                            y       = H(i + 1, i);
            //                            q       = (d(i) - p) * (d(i) - p) + e(i) * e(i);
            //                            t       = (x * s - z * r) / q;
            //                            H(i, n) = t;
            //                            if(abs(x) > abs(z))
            //                            {
            //                                H(i + 1, n) = (-r - w * t) / x;
            //                            }
            //                            else
            //                            {
            //                                H(i + 1, n) = (-s - y * t) / z;
            //                            }
            //                        }

            //                        // Overflow control

            //                        t = abs(H(i, n));
            //                        if((eps * t) * t > 1)
            //                        {
            //                            for(int j = i; j <= n; j++)
            //                            {
            //                                H(j, n) = H(j, n) / t;
            //                            }
            //                        }
            //                    }
            //                }

            //                // Complex vector
            //            }
            //            else if(q < 0)
            //            {
            //                int l = n - 1;

            //                // Last vector component imaginary so matrix is triangular

            //                if(abs(H(n, n - 1)) > abs(H(n - 1, n)))
            //                {
            //                    H(n - 1, n - 1) = q / H(n, n - 1);
            //                    H(n - 1, n)     = -(H(n, n) - p) / H(n, n - 1);
            //                }
            //                else
            //                {
            //                    cdiv(DataType(0.0), -H(n - 1, n), H(n - 1, n - 1) - p, q);
            //                    H(n - 1, n - 1) = cdivr;
            //                    H(n - 1, n)     = cdivi;
            //                }
            //                H(n, n - 1) = DataType(0.0);
            //                H(n, n)     = DataType(1.0);
            //                for(int i = n - 2; i >= 0; i--)
            //                {
            //                    DataType ra, sa, vr, vi;
            //                    ra = DataType(0.0);
            //                    sa = DataType(0.0);
            //                    for(int j = l; j <= n; j++)
            //                    {
            //                        ra = ra + H(i, j) * H(j, n - 1);
            //                        sa = sa + H(i, j) * H(j, n);
            //                    }
            //                    w = H(i, i) - p;

            //                    if(e(i) < DataType(0.0))
            //                    {
            //                        z = w;
            //                        r = ra;
            //                        s = sa;
            //                    }
            //                    else
            //                    {
            //                        l = i;
            //                        if(e(i) == 0)
            //                        {
            //                            cdiv(-ra, -sa, w, q);
            //                            H(i, n - 1) = cdivr;
            //                            H(i, n)     = cdivi;
            //                        }
            //                        else
            //                        {
            //                            // Solve complex equations

            //                            x  = H(i, i + 1);
            //                            y  = H(i + 1, i);
            //                            vr = (d(i) - p) * (d(i) - p) + e(i) * e(i) - q * q;
            //                            vi = (d(i) - p) * 2.0 * q;
            //                            if((vr == DataType(0.0)) && (vi == DataType(0.0)))
            //                            {
            //                                vr = eps * norm * (abs(w) + abs(q) + abs(x) + abs(y) + abs(z));
            //                            }
            //                            cdiv(x * r - z * ra + q * sa, x * s - z * sa - q * ra, vr, vi);
            //                            H(i, n - 1) = cdivr;
            //                            H(i, n)     = cdivi;
            //                            if(abs(x) > (abs(z) + abs(q)))
            //                            {
            //                                H(i + 1, n - 1) = (-ra - w * H(i, n - 1) + q * H(i, n)) / x;
            //                                H(i + 1, n)     = (-sa - w * H(i, n) - q * H(i, n - 1)) / x;
            //                            }
            //                            else
            //                            {
            //                                cdiv(-r - y * H(i, n - 1), -s - y * H(i, n), z, q);
            //                                H(i + 1, n - 1) = cdivr;
            //                                H(i + 1, n)     = cdivi;
            //                            }
            //                        }

            //                        // Overflow control

            //                        t = max(abs(H(i, n - 1)), abs(H(i, n)));
            //                        if((eps * t) * t > 1)
            //                        {
            //                            for(int j = i; j <= n; j++)
            //                            {
            //                                H(j, n - 1) = H(j, n - 1) / t;
            //                                H(j, n)     = H(j, n) / t;
            //                            }
            //                        }
            //                    }
            //                }
            //            }
            //        }

            //        for(int i = 0; i < nn; i++)
            //        {
            //            if(i < low || i > high)
            //            {
            //                for(int j = i; j < nn; j++)
            //                {
            //                    V(i, j) = H(i, j);
            //                }
            //            }
            //        }

            //        for(int j = nn - 1; j >= low; j--)
            //        {
            //            for(int i = low; i <= high; i++)
            //            {
            //                z = DataType(0.0);
            //                for(int k = low; k <= min(j, high); k++)
            //                {
            //                    z = z + V(i, k) * H(k, j);
            //                }
            //                V(i, j) = z;
            //            }
            //        }
            //    }

            // public:
            //    Eigenvalue(const Matrix& A)
            //    {
            //        n = A.num_cols();
            //        V = Matrix(n, n);
            //        d = Vector(n);
            //        e = Vector(n);

            //        issymmetric = 1;
            //        for(int j = 0; (j < n) && issymmetric; j++)
            //        {
            //            for(int i = 0; (i < n) && issymmetric; i++)
            //            {
            //                issymmetric = (A(i, j) == A(j, i));
            //            }
            //        }

            //        if(issymmetric)
            //        {
            //            for(int i = 0; i < n; i++)
            //            {
            //                for(int j = 0; j < n; j++)
            //                {
            //                    V(i, j) = A(i, j);
            //                }
            //            }

            //            // Tridiagonalize.
            //            tred2();

            //            // Diagonalize.
            //            tql2();
            //        }
            //        else
            //        {
            //            H   = Matrix(n, n);
            //            ort = Vector(n);

            //            for(int j = 0; j < n; j++)
            //            {
            //                for(int i = 0; i < n; i++)
            //                {
            //                    H(i, j) = A(i, j);
            //                }
            //            }

            //            // Reduce to Hessenberg form.
            //            orthes();

            //            // Reduce Hessenberg to real Schur form.
            //            hqr2();
            //        }
            //    }

            //    void getD(Matrix& D)
            //    {
            //        D = Matrix(n, n);
            //        for(int i = 0; i < n; i++)
            //        {
            //            for(int j = 0; j < n; j++)
            //            {
            //                D(i, j) = DataType(0.0);
            //            }
            //            D(i, i) = d(i);
            //            if(e(i) > 0)
            //            {
            //                D(i, i + 1) = e(i);
            //            }
            //            else if(e(i) < 0)
            //            {
            //                D(i, i - 1) = e(i);
            //            }
            //        }
            //    }
            //};
        }
    }
}

namespace Kokkos
{
    namespace LinearAlgebra
    {
        namespace Internal
        {
            template<typename DataType, class ExecutionSpace>
            class LU
            {
                using Vector    = Kokkos::Extension::Vector<DataType, ExecutionSpace>;
                using VectorInt = Kokkos::Extension::Vector<size_type, ExecutionSpace>;
                using Matrix    = Kokkos::Extension::Matrix<DataType, ExecutionSpace>;

                /* Array for internal storage of decomposition.  */
                Matrix    _LU;
                size_type _m;
                size_type _n;
                size_type _pivsign;
                VectorInt _piv;

                struct Decompose
                {
                };

                KOKKOS_INLINE_FUNCTION void operator()(const Decompose&, const size_type i) const {}

            public:
                LU(const Matrix& A) : _LU("LU", A.extent(0), A.extent(1)), _m(A.extent(0)), _n(A.extent(1)), _piv("piv", A.extent(0))
                {
                    Kokkos::deep_copy(_LU, A);

                    // Use a "left-looking", dot-product, Crout/Doolittle algorithm.

                    for(size_type i = 0; i < _m; i++)
                    {
                        _piv(i) = i;
                    }

                    _pivsign = 1;

                    Vector LUrowi;

                    Vector LUcolj("LUcolj", _m);

                    // Outer loop.

                    for(size_type j = 0; j < _n; j++)
                    {
                        // Make a copy of the j-th column to localize references.

                        for(size_type i = 0; i < _m; i++)
                        {
                            LUcolj(i) = _LU(i, j);
                        }

                        // Apply previous transformations.

                        for(size_type i = 0; i < _m; i++)
                        {
                            LUrowi = row(_LU, i);

                            // Most of the time is spent in the following dot product.

                            const size_type kmax = min(i, j);

                            double s = DataType(0.0);

                            for(size_type k = 0; k < kmax; k++)
                            {
                                s += LUrowi(k) * LUcolj(k);
                            }

                            LUrowi(j) = LUcolj(i) -= s;
                        }

                        // Find pivot and exchange if necessary.

                        size_type p = j;

                        for(size_type i = j + 1; i < _m; i++)
                        {
                            if(abs(LUcolj(i)) > abs(LUcolj(p)))
                            {
                                p = i;
                            }
                        }

                        if(p != j)
                        {
                            size_type k = 0;

                            for(k = 0; k < _n; k++)
                            {
                                double t  = _LU(p, k);
                                _LU(p, k) = _LU(j, k);
                                _LU(j, k) = t;
                            }

                            k        = _piv(p);
                            _piv(p)  = _piv(j);
                            _piv(j)  = k;
                            _pivsign = -_pivsign;
                        }

                        // Compute multipliers.

                        if((j < _m) && (_LU(j, j) != DataType(0.0)))
                        {
                            for(size_type i = j + 1; i < _m; i++)
                            {
                                _LU(i, j) /= _LU(j, j);
                            }
                        }
                    }
                }

            private:
                Matrix permute_copy(const Matrix& A, const VectorInt& piv, const size_type j0, const size_type j1)
                {
                    size_type piv_length = piv.dim();

                    Matrix X("X", piv_length, j1 - j0 + 1);

                    for(size_type i = 0; i < piv_length; i++)
                    {
                        for(size_type j = j0; j <= j1; j++)
                        {
                            X(i, j - j0) = A(piv(i), j);
                        }
                    }

                    return X;
                }

                Vector permute_copy(const Vector& A, const VectorInt& piv)
                {
                    size_type piv_length = piv.dim();

                    if(piv_length != A.dim())
                    {
                        return Vector();
                    }

                    Vector x(piv_length);

                    for(size_type i = 0; i < piv_length; i++)
                    {
                        x(i) = A(piv(i));
                    }

                    return x;
                }

            public:
                bool isNonsingular()
                {
                    for(size_type j = 0; j < _n; j++)
                    {
                        if(_LU(j, j) == 0)
                        {
                            return false;
                        }
                    }
                    return true;
                }

                Matrix getL()
                {
                    Matrix L_("L", _m, _n);

                    for(size_type i = 0; i < _m; i++)
                    {
                        for(size_type j = 0; j < _n; j++)
                        {
                            if(i > j)
                            {
                                L_(i, j) = _LU(i, j);
                            }
                            else if(i == j)
                            {
                                L_(i, j) = DataType(1.0);
                            }
                            else
                            {
                                L_(i, j) = DataType(0.0);
                            }
                        }
                    }
                    return L_;
                }

                Matrix getU()
                {
                    Matrix U_("U", _n, _n);

                    for(size_type i = 0; i < _n; i++)
                    {
                        for(size_type j = 0; j < _n; j++)
                        {
                            if(i <= j)
                            {
                                U_(i, j) = _LU(i, j);
                            }
                            else
                            {
                                U_(i, j) = DataType(0.0);
                            }
                        }
                    }
                    return U_;
                }

                VectorInt getPivot() { return _piv; }

                DataType determinant()
                {
                    if(_m != _n)
                    {
                        return DataType(0);
                    }

                    DataType d = DataType(_pivsign);

                    for(size_type j = 0; j < _n; j++)
                    {
                        d *= _LU(j, j);
                    }
                    return d;
                }

                /// <summary>
                /// Solve A*X = B
                /// </summary>
                /// <param name="B">A Matrix with as many rows as A and any number of columns.</param>
                /// <returns>X so that L*U*X = B(piv,:), if B is nonconformant, returns DataType(0.0) (null) array.</returns>
                Matrix Solve(const Matrix& B)
                {
                    /* Dimensions: A is mxn, X is nxk, B is mxk */

                    if(B.num_rows() != _m)
                    {
                        return Matrix(DataType(0.0));
                    }
                    if(!isNonsingular())
                    {
                        return Matrix(DataType(0.0));
                    }

                    // Copy right hand side with pivoting
                    const size_type nx = B.num_cols();

                    Matrix X = permute_copy(B, _piv, 0, nx - 1);

                    // Solve L*Y = B(piv,:)
                    for(size_type k = 0; k < _n; k++)
                    {
                        for(size_type i = k + 1; i < _n; i++)
                        {
                            for(size_type j = 0; j < nx; j++)
                            {
                                X(i, j) -= X(k, j) * _LU(i, k);
                            }
                        }
                    }

                    // Solve U*X = Y;
                    for(size_type k = _n - 1; k >= 0; k--)
                    {
                        for(size_type j = 0; j < nx; j++)
                        {
                            X(k, j) /= _LU(k, k);
                        }

                        for(size_type i = 0; i < k; i++)
                        {
                            for(size_type j = 0; j < nx; j++)
                            {
                                X(i, j) -= X(k, j) * _LU(i, k);
                            }
                        }
                    }

                    return X;
                }

                /// <summary>
                /// Solve A*x = b, where x and b are vectors of length equal to the number of rows in A.
                /// </summary>
                /// <param name="b">a vector (Vector> of length equal to the first dimension of A.</param>
                /// <returns>x a vector (Vector> so that L*U*x = b(piv), if B is nonconformant, returns DataType(0.0) (null) array.</returns>
                Vector Solve(const Vector& b)
                {
                    /* Dimensions: A is mxn, X is nxk, B is mxk */

                    if(b.dim() != _m)
                    {
                        return Vector();
                    }
                    if(!isNonsingular())
                    {
                        return Vector();
                    }

                    Vector x = permute_copy(b, _piv);

                    // Solve L*Y = B(piv)
                    for(size_type k = 0; k < _n; k++)
                    {
                        for(size_type i = k + 1; i < _n; i++)
                        {
                            x(i) -= x(k) * _LU(i, k);
                        }
                    }

                    // Solve U*X = Y;
                    for(size_type k = _n - 1; k >= 0; k--)
                    {
                        x(k) /= _LU(k, k);

                        for(size_type i = 0; i < k; i++)
                        {
                            x(i) -= x(k) * _LU(i, k);
                        }
                    }

                    return x;
                }
            };
        }
    }
}

namespace Kokkos
{
    namespace LinearAlgebra
    {
        namespace Internal
        {
            // template<typename DataType, class ExecutionSpace>
            // class QR
            //{
            //    using Vector = Kokkos::Extension::Vector<DataType, ExecutionSpace>;
            //    using Matrix = Kokkos::Extension::Matrix<DataType, ExecutionSpace>;

            //    /* Array for internal storage of decomposition.
            //    @serial internal array storage.
            //    */

            //    Matrix QR_;

            //    /* Row and column dimensions.
            //    @serial column dimension.
            //    @serial row dimension.
            //    */
            //    int m, n;

            //    /* Array for internal storage of diagonal of R.
            //    @serial diagonal of R.
            //    */
            //    Vector Rdiag;

            // public:
            //    /**
            //        Create a QR factorization object for A.

            //        @param A rectangular (m>=n) matrix.
            //    */
            //    QR(const Matrix& A) /* constructor */
            //    {
            //        QR_   = A;
            //        m     = A.num_rows();
            //        n     = A.num_cols();
            //        Rdiag = Vector(n);
            //        int i = 0, j = 0, k = 0;

            //        // Main loop.
            //        for(k = 0; k < n; k++)
            //        {
            //            // Compute 2-norm of k-th column without under/overflow.
            //            DataType nrm = 0;
            //            for(i = k; i < m; i++)
            //            {
            //                nrm = hypot(nrm, QR_(i, k));
            //            }

            //            if(nrm != DataType(0.0))
            //            {
            //                // Form k-th Householder vector.
            //                if(QR_(k, k) < 0)
            //                {
            //                    nrm = -nrm;
            //                }
            //                for(i = k; i < m; i++)
            //                {
            //                    QR_(i, k) /= nrm;
            //                }
            //                QR_(k, k) += DataType(1.0);

            //                // Apply transformation to remaining columns.
            //                for(j = k + 1; j < n; j++)
            //                {
            //                    DataType s = DataType(0.0);
            //                    for(i = k; i < m; i++)
            //                    {
            //                        s += QR_(i, k) * QR_(i, j);
            //                    }
            //                    s = -s / QR_(k, k);
            //                    for(i = k; i < m; i++)
            //                    {
            //                        QR_(i, j) += s * QR_(i, k);
            //                    }
            //                }
            //            }
            //            Rdiag(k) = -nrm;
            //        }
            //    }

            //    /**
            //        Flag to denote the matrix is of full rank.

            //        @return 1 if matrix is full rank, 0 otherwise.
            //    */
            //    int isFullRank() const
            //    {
            //        for(int j = 0; j < n; j++)
            //        {
            //            if(Rdiag(j) == 0)
            //                return 0;
            //        }
            //        return 1;
            //    }

            //    /**

            //    Retreive the Householder vectors from QR factorization
            //    @returns lower trapezoidal matrix whose columns define the reflections
            //    */

            //    Matrix getHouseholder(void) const
            //    {
            //        Matrix H(m, n);

            //        /* note: H is completely filled in by algorithm, so
            //           initializaiton of H is not necessary.
            //        */
            //        for(int i = 0; i < m; i++)
            //        {
            //            for(int j = 0; j < n; j++)
            //            {
            //                if(i >= j)
            //                {
            //                    H(i, j) = QR_(i, j);
            //                }
            //                else
            //                {
            //                    H(i, j) = DataType(0.0);
            //                }
            //            }
            //        }
            //        return H;
            //    }

            //    /** Return the upper triangular factor, R, of the QR factorization
            //    @return     R
            //    */

            //    Matrix getR() const
            //    {
            //        Matrix R(n, n);
            //        for(int i = 0; i < n; i++)
            //        {
            //            for(int j = 0; j < n; j++)
            //            {
            //                if(i < j)
            //                {
            //                    R(i, j) = QR_(i, j);
            //                }
            //                else if(i == j)
            //                {
            //                    R(i, j) = Rdiag(i);
            //                }
            //                else
            //                {
            //                    R(i, j) = DataType(0.0);
            //                }
            //            }
            //        }
            //        return R;
            //    }

            //    /**
            //    @return     Q the (ecnomy-sized) orthogonal factor (Q*R=A).
            //    */

            //    Matrix getQ() const
            //    {
            //        int i = 0, j = 0, k = 0;

            //        Matrix Q(m, n);
            //        for(k = n - 1; k >= 0; k--)
            //        {
            //            for(i = 0; i < m; i++)
            //            {
            //                Q(i, k) = DataType(0.0);
            //            }
            //            Q(k, k) = DataType(1.0);
            //            for(j = k; j < n; j++)
            //            {
            //                if(QR_(k, k) != 0)
            //                {
            //                    DataType s = DataType(0.0);
            //                    for(i = k; i < m; i++)
            //                    {
            //                        s += QR_(i, k) * Q(i, j);
            //                    }
            //                    s = -s / QR_(k, k);
            //                    for(i = k; i < m; i++)
            //                    {
            //                        Q(i, j) += s * QR_(i, k);
            //                    }
            //                }
            //            }
            //        }
            //        return Q;
            //    }

            //    /** Least squares solution of A*x = b
            //    @param b     right hand side  (m-length vector).
            //    @return x    n-length vector that minimizes the two norm of Q*R*X-B.
            //         If B is non-conformant, or if QR.isFullRank() is false,
            //                         the routine returns a null (0-length) vector.
            //    */

            //    Vector solve(const Vector& b) const
            //    {
            //        if(b.dim() != m) /* arrays must be conformant */
            //            return Vector();

            //        if(!isFullRank()) /* matrix is rank deficient */
            //        {
            //            return Vector();
            //        }

            //        Vector x = b;

            //        // Compute Y = transpose(Q)*b
            //        for(int k = 0; k < n; k++)
            //        {
            //            DataType s = DataType(0.0);
            //            for(int i = k; i < m; i++)
            //            {
            //                s += QR_(i, k) * x(i);
            //            }
            //            s = -s / QR_(k, k);
            //            for(int i = k; i < m; i++)
            //            {
            //                x(i) += s * QR_(i, k);
            //            }
            //        }
            //        // Solve R*X = Y;
            //        for(int k = n - 1; k >= 0; k--)
            //        {
            //            x(k) /= Rdiag(k);
            //            for(int i = 0; i < k; i++)
            //            {
            //                x(i) -= x(k) * QR_(i, k);
            //            }
            //        }

            //        /* return n x nx portion of X */
            //        Vector x_(n);
            //        for(int i = 0; i < n; i++)
            //            x_(i) = x(i);

            //        return x_;
            //    }

            //    /** Least squares solution of A*X = B
            //    @param B     m x k Array (must conform).
            //    @return X     n x k Array that minimizes the two norm of Q*R*X-B. If
            //                         B is non-conformant, or if QR.isFullRank() is false,
            //                         the routine returns a null (DataType(0.0)) array.
            //    */

            //    Matrix solve(const Matrix& B) const
            //    {
            //        if(B.num_rows() != m) /* arrays must be conformant */
            //            return Matrix(DataType(0.0));

            //        if(!isFullRank()) /* matrix is rank deficient */
            //        {
            //            return Matrix(DataType(0.0));
            //        }

            //        int                              nx = B.num_cols();
            //        Matrix X  = B;
            //        int                              i = 0, j = 0, k = 0;

            //        // Compute Y = transpose(Q)*B
            //        for(k = 0; k < n; k++)
            //        {
            //            for(j = 0; j < nx; j++)
            //            {
            //                DataType s = DataType(0.0);
            //                for(i = k; i < m; i++)
            //                {
            //                    s += QR_(i, k) * X(i, j);
            //                }
            //                s = -s / QR_(k, k);
            //                for(i = k; i < m; i++)
            //                {
            //                    X(i, j) += s * QR_(i, k);
            //                }
            //            }
            //        }
            //        // Solve R*X = Y;
            //        for(k = n - 1; k >= 0; k--)
            //        {
            //            for(j = 0; j < nx; j++)
            //            {
            //                X(k, j) /= Rdiag(k);
            //            }
            //            for(i = 0; i < k; i++)
            //            {
            //                for(j = 0; j < nx; j++)
            //                {
            //                    X(i, j) -= X(k, j) * QR_(i, k);
            //                }
            //            }
            //        }

            //        /* return n x nx portion of X */
            //        Matrix X_(n, nx);
            //        for(i = 0; i < n; i++)
            //            for(j = 0; j < nx; j++)
            //                X_(i, j) = X(i, j);

            //        return X_;
            //    }
            //};
        }
    }
}

namespace Kokkos
{
    namespace LinearAlgebra
    {
        namespace Internal
        {
            // template<typename DataType, class ExecutionSpace>
            // class SVD
            //{
            //    using Vector = Kokkos::Extension::Vector<DataType, ExecutionSpace>;
            //    using Matrix = Kokkos::Extension::Matrix<DataType, ExecutionSpace>;

            //    Matrix U, V;
            //    Vector s;
            //    size_type                        m, n;

            // public:
            //    SVD(const Matrix& Arg)
            //    {
            //        m            = Arg.num_rows();
            //        n            = Arg.num_cols();
            //        size_type nu = min(m, n);
            //        s            = Vector(min(m + 1, n));
            //        U            = Matrix(m, nu, DataType(0));
            //        V            = Matrix(n, n);
            //        Vector e(n);
            //        Vector work(m);
            //        Matrix A(Arg);
            //        size_type                        wantu = 1; /* boolean */
            //        size_type                        wantv = 1; /* boolean */
            //        size_type                        i = 0, j = 0, k = 0;

            //        // Reduce A to bidiagonal form, storing the diagonal elements
            //        // in s and the super-diagonal elements in e.

            //        size_type nct = min(m - 1, n);
            //        size_type nrt = max(0, min(n - 2, m));
            //        for(k = 0; k < max(nct, nrt); k++)
            //        {
            //            if(k < nct)
            //            {
            //                // Compute the transformation for the k-th column and
            //                // place the k-th diagonal in s(k).
            //                // Compute 2-norm of k-th column without under/overflow.
            //                s(k) = 0;
            //                for(i = k; i < m; i++)
            //                {
            //                    s(k) = hypot(s(k), A(i, k));
            //                }
            //                if(s(k) != DataType(0.0))
            //                {
            //                    if(A(k, k) < DataType(0.0))
            //                    {
            //                        s(k) = -s(k);
            //                    }
            //                    for(i = k; i < m; i++)
            //                    {
            //                        A(i, k) /= s(k);
            //                    }
            //                    A(k, k) += DataType(1.0);
            //                }
            //                s(k) = -s(k);
            //            }
            //            for(j = k + 1; j < n; j++)
            //            {
            //                if((k < nct) && (s(k) != DataType(0.0)))
            //                {
            //                    // Apply the transformation.

            //                    double t = 0;
            //                    for(i = k; i < m; i++)
            //                    {
            //                        t += A(i, k) * A(i, j);
            //                    }
            //                    t = -t / A(k, k);
            //                    for(i = k; i < m; i++)
            //                    {
            //                        A(i, j) += t * A(i, k);
            //                    }
            //                }

            //                // Place the k-th row of A into e for the
            //                // subsequent calculation of the row transformation.

            //                e(j) = A(k, j);
            //            }
            //            if(wantu & (k < nct))
            //            {
            //                // Place the transformation in U for subsequent back
            //                // multiplication.

            //                for(i = k; i < m; i++)
            //                {
            //                    U(i, k) = A(i, k);
            //                }
            //            }
            //            if(k < nrt)
            //            {
            //                // Compute the k-th row transformation and place the
            //                // k-th super-diagonal in e(k).
            //                // Compute 2-norm without under/overflow.
            //                e(k) = 0;
            //                for(i = k + 1; i < n; i++)
            //                {
            //                    e(k) = hypot(e(k), e(i));
            //                }
            //                if(e(k) != DataType(0.0))
            //                {
            //                    if(e(k + 1) < DataType(0.0))
            //                    {
            //                        e(k) = -e(k);
            //                    }
            //                    for(i = k + 1; i < n; i++)
            //                    {
            //                        e(i) /= e(k);
            //                    }
            //                    e(k + 1) += DataType(1.0);
            //                }
            //                e(k) = -e(k);
            //                if((k + 1 < m) & (e(k) != DataType(0.0)))
            //                {
            //                    // Apply the transformation.

            //                    for(i = k + 1; i < m; i++)
            //                    {
            //                        work(i) = DataType(0.0);
            //                    }
            //                    for(j = k + 1; j < n; j++)
            //                    {
            //                        for(i = k + 1; i < m; i++)
            //                        {
            //                            work(i) += e(j) * A(i, j);
            //                        }
            //                    }
            //                    for(j = k + 1; j < n; j++)
            //                    {
            //                        double t = -e(j) / e(k + 1);
            //                        for(i = k + 1; i < m; i++)
            //                        {
            //                            A(i, j) += t * work(i);
            //                        }
            //                    }
            //                }
            //                if(wantv)
            //                {
            //                    // Place the transformation in V for subsequent
            //                    // back multiplication.

            //                    for(i = k + 1; i < n; i++)
            //                    {
            //                        V(i, k) = e(i);
            //                    }
            //                }
            //            }
            //        }

            //        // Set up the final bidiagonal matrix or order p.

            //        size_type p = min(n, m + 1);
            //        if(nct < n)
            //        {
            //            s(nct) = A(nct, nct);
            //        }
            //        if(m < p)
            //        {
            //            s(p - 1) = DataType(0.0);
            //        }
            //        if(nrt + 1 < p)
            //        {
            //            e(nrt) = A(nrt, p - 1);
            //        }
            //        e(p - 1) = DataType(0.0);

            //        // If required, generate U.

            //        if(wantu)
            //        {
            //            for(j = nct; j < nu; j++)
            //            {
            //                for(i = 0; i < m; i++)
            //                {
            //                    U(i, j) = DataType(0.0);
            //                }
            //                U(j, j) = DataType(1.0);
            //            }
            //            for(k = nct - 1; k >= 0; k--)
            //            {
            //                if(s(k) != DataType(0.0))
            //                {
            //                    for(j = k + 1; j < nu; j++)
            //                    {
            //                        double t = 0;
            //                        for(i = k; i < m; i++)
            //                        {
            //                            t += U(i, k) * U(i, j);
            //                        }
            //                        t = -t / U(k, k);
            //                        for(i = k; i < m; i++)
            //                        {
            //                            U(i, j) += t * U(i, k);
            //                        }
            //                    }
            //                    for(i = k; i < m; i++)
            //                    {
            //                        U(i, k) = -U(i, k);
            //                    }
            //                    U(k, k) = DataType(1.0) + U(k, k);
            //                    for(i = 0; i < k - 1; i++)
            //                    {
            //                        U(i, k) = DataType(0.0);
            //                    }
            //                }
            //                else
            //                {
            //                    for(i = 0; i < m; i++)
            //                    {
            //                        U(i, k) = DataType(0.0);
            //                    }
            //                    U(k, k) = DataType(1.0);
            //                }
            //            }
            //        }

            //        // If required, generate V.

            //        if(wantv)
            //        {
            //            for(k = n - 1; k >= 0; k--)
            //            {
            //                if((k < nrt) & (e(k) != DataType(0.0)))
            //                {
            //                    for(j = k + 1; j < nu; j++)
            //                    {
            //                        double t = 0;
            //                        for(i = k + 1; i < n; i++)
            //                        {
            //                            t += V(i, k) * V(i, j);
            //                        }
            //                        t = -t / V(k + 1, k);
            //                        for(i = k + 1; i < n; i++)
            //                        {
            //                            V(i, j) += t * V(i, k);
            //                        }
            //                    }
            //                }
            //                for(i = 0; i < n; i++)
            //                {
            //                    V(i, k) = DataType(0.0);
            //                }
            //                V(k, k) = DataType(1.0);
            //            }
            //        }

            //        // Main iteration loop for the singular values.

            //        size_type pp   = p - 1;
            //        size_type iter = 0;
            //        double    eps  = ::pow(2.0, -52.0);
            //        while(p > 0)
            //        {
            //            size_type k    = 0;
            //            size_type kase = 0;

            //            // Here is where a test for too many iterations would go.

            //            // This section of the program inspects for
            //            // negligible elements in the s and e arrays.  On
            //            // completion the variables kase and k are set as follows.

            //            // kase = 1     if s(p) and e(k-1) are negligible and k<p
            //            // kase = 2     if s(k) is negligible and k<p
            //            // kase = 3     if e(k-1) is negligible, k<p, and
            //            //              s(k), ..., s(p) are not negligible (qr step).
            //            // kase = 4     if e(p-1) is negligible (convergence).

            //            for(k = p - 2; k >= -1; k--)
            //            {
            //                if(k == -1)
            //                {
            //                    break;
            //                }
            //                if(abs(e(k)) <= eps * (abs(s(k)) + abs(s(k + 1))))
            //                {
            //                    e(k) = DataType(0.0);
            //                    break;
            //                }
            //            }
            //            if(k == p - 2)
            //            {
            //                kase = 4;
            //            }
            //            else
            //            {
            //                size_type ks;
            //                for(ks = p - 1; ks >= k; ks--)
            //                {
            //                    if(ks == k)
            //                    {
            //                        break;
            //                    }
            //                    double t = (ks != p ? abs(e(ks)) : 0.) + (ks != k + 1 ? abs(e(ks - 1)) : 0.);
            //                    if(abs(s(ks)) <= eps * t)
            //                    {
            //                        s(ks) = DataType(0.0);
            //                        break;
            //                    }
            //                }
            //                if(ks == k)
            //                {
            //                    kase = 3;
            //                }
            //                else if(ks == p - 1)
            //                {
            //                    kase = 1;
            //                }
            //                else
            //                {
            //                    kase = 2;
            //                    k    = ks;
            //                }
            //            }
            //            k++;

            //            // Perform the task indicated by kase.

            //            switch(kase)
            //            {
            //                    // Deflate negligible s(p).

            //                case 1:
            //                {
            //                    double f = e(p - 2);
            //                    e(p - 2) = DataType(0.0);
            //                    for(j = p - 2; j >= k; j--)
            //                    {
            //                        double t  = hypot(s(j), f);
            //                        double cs = s(j) / t;
            //                        double sn = f / t;
            //                        s(j)      = t;
            //                        if(j != k)
            //                        {
            //                            f        = -sn * e(j - 1);
            //                            e(j - 1) = cs * e(j - 1);
            //                        }
            //                        if(wantv)
            //                        {
            //                            for(i = 0; i < n; i++)
            //                            {
            //                                t           = cs * V(i, j) + sn * V(i, p - 1);
            //                                V(i, p - 1) = -sn * V(i, j) + cs * V(i, p - 1);
            //                                V(i, j)     = t;
            //                            }
            //                        }
            //                    }
            //                }
            //                break;

            //                    // Split at negligible s(k).

            //                case 2:
            //                {
            //                    double f = e(k - 1);
            //                    e(k - 1) = DataType(0.0);
            //                    for(j = k; j < p; j++)
            //                    {
            //                        double t  = hypot(s(j), f);
            //                        double cs = s(j) / t;
            //                        double sn = f / t;
            //                        s(j)      = t;
            //                        f         = -sn * e(j);
            //                        e(j)      = cs * e(j);
            //                        if(wantu)
            //                        {
            //                            for(i = 0; i < m; i++)
            //                            {
            //                                t           = cs * U(i, j) + sn * U(i, k - 1);
            //                                U(i, k - 1) = -sn * U(i, j) + cs * U(i, k - 1);
            //                                U(i, j)     = t;
            //                            }
            //                        }
            //                    }
            //                }
            //                break;

            //                    // Perform one qr step.

            //                case 3:
            //                {
            //                    // Calculate the shift.

            //                    double scale = max(max(max(max(abs(s(p - 1)), abs(s(p - 2))), abs(e(p - 2))), abs(s(k))), abs(e(k)));
            //                    double sp    = s(p - 1) / scale;
            //                    double spm1  = s(p - 2) / scale;
            //                    double epm1  = e(p - 2) / scale;
            //                    double sk    = s(k) / scale;
            //                    double ek    = e(k) / scale;
            //                    double b     = ((spm1 + sp) * (spm1 - sp) + epm1 * epm1) / 2.0;
            //                    double c     = (sp * epm1) * (sp * epm1);
            //                    double shift = DataType(0.0);
            //                    if((b != DataType(0.0)) || (c != DataType(0.0)))
            //                    {
            //                        shift = sqrt(b * b + c);
            //                        if(b < DataType(0.0))
            //                        {
            //                            shift = -shift;
            //                        }
            //                        shift = c / (b + shift);
            //                    }
            //                    double f = (sk + sp) * (sk - sp) + shift;
            //                    double g = sk * ek;

            //                    // Chase zeros.

            //                    for(j = k; j < p - 1; j++)
            //                    {
            //                        double t  = hypot(f, g);
            //                        double cs = f / t;
            //                        double sn = g / t;
            //                        if(j != k)
            //                        {
            //                            e(j - 1) = t;
            //                        }
            //                        f        = cs * s(j) + sn * e(j);
            //                        e(j)     = cs * e(j) - sn * s(j);
            //                        g        = sn * s(j + 1);
            //                        s(j + 1) = cs * s(j + 1);
            //                        if(wantv)
            //                        {
            //                            for(i = 0; i < n; i++)
            //                            {
            //                                t           = cs * V(i, j) + sn * V(i, j + 1);
            //                                V(i, j + 1) = -sn * V(i, j) + cs * V(i, j + 1);
            //                                V(i, j)     = t;
            //                            }
            //                        }
            //                        t        = hypot(f, g);
            //                        cs       = f / t;
            //                        sn       = g / t;
            //                        s(j)     = t;
            //                        f        = cs * e(j) + sn * s(j + 1);
            //                        s(j + 1) = -sn * e(j) + cs * s(j + 1);
            //                        g        = sn * e(j + 1);
            //                        e(j + 1) = cs * e(j + 1);
            //                        if(wantu && (j < m - 1))
            //                        {
            //                            for(i = 0; i < m; i++)
            //                            {
            //                                t           = cs * U(i, j) + sn * U(i, j + 1);
            //                                U(i, j + 1) = -sn * U(i, j) + cs * U(i, j + 1);
            //                                U(i, j)     = t;
            //                            }
            //                        }
            //                    }
            //                    e(p - 2) = f;
            //                    iter     = iter + 1;
            //                }
            //                break;

            //                    // Convergence.

            //                case 4:
            //                {
            //                    // Make the singular values positive.

            //                    if(s(k) <= DataType(0.0))
            //                    {
            //                        s(k) = (s(k) < DataType(0.0) ? -s(k) : DataType(0.0));
            //                        if(wantv)
            //                        {
            //                            for(i = 0; i <= pp; i++)
            //                            {
            //                                V(i, k) = -V(i, k);
            //                            }
            //                        }
            //                    }

            //                    // Order the singular values.

            //                    while(k < pp)
            //                    {
            //                        if(s(k) >= s(k + 1))
            //                        {
            //                            break;
            //                        }
            //                        double t = s(k);
            //                        s(k)     = s(k + 1);
            //                        s(k + 1) = t;
            //                        if(wantv && (k < n - 1))
            //                        {
            //                            for(i = 0; i < n; i++)
            //                            {
            //                                t           = V(i, k + 1);
            //                                V(i, k + 1) = V(i, k);
            //                                V(i, k)     = t;
            //                            }
            //                        }
            //                        if(wantu && (k < m - 1))
            //                        {
            //                            for(i = 0; i < m; i++)
            //                            {
            //                                t           = U(i, k + 1);
            //                                U(i, k + 1) = U(i, k);
            //                                U(i, k)     = t;
            //                            }
            //                        }
            //                        k++;
            //                    }
            //                    iter = 0;
            //                    p--;
            //                }
            //                break;
            //            }
            //        }
            //    }

            //    void getU(Matrix& A)
            //    {
            //        size_type minm = min(m + 1, n);

            //        A = Matrix(m, minm);

            //        for(size_type i = 0; i < m; i++)
            //            for(size_type j = 0; j < minm; j++)
            //                A(i, j) = U(i, j);
            //    }

            //    /* Return the right singular vectors */

            //    void getV(Matrix& A) { A = V; }

            //    /** Return the one-dimensional array of singular values */

            //    void getSingularValues(Vector& x) { x = s; }

            //    /** Return the diagonal matrix of singular values
            //    @return     S
            //    */

            //    void getS(Matrix& A)
            //    {
            //        A = Matrix(n, n);
            //        for(size_type i = 0; i < n; i++)
            //        {
            //            for(size_type j = 0; j < n; j++)
            //            {
            //                A(i, j) = DataType(0.0);
            //            }
            //            A(i, i) = s(i);
            //        }
            //    }

            //    /** Two norm  (max(S)) */

            //    double norm2() { return s(0); }

            //    /** Two norm of condition number (max(S)/min(S)) */

            //    double cond() { return s(0) / s[::min(m, n) - 1]; }

            //    /** Effective numerical matrix rank
            //    @return     Number of nonnegligible singular values.
            //    */

            //    size_type rank()
            //    {
            //        double    eps = ::pow(2.0, -52.0);
            //        double    tol = max(m, n) * s(0) * eps;
            //        size_type r   = 0;
            //        for(size_type i = 0; i < s.dim(); i++)
            //        {
            //            if(s(i) > tol)
            //            {
            //                r++;
            //            }
            //        }
            //        return r;
            //    }
            //};
        }
    }
}

namespace Kokkos
{
    namespace LinearAlgebra
    {
        namespace Internal
        {
            template<typename DataType, class ExecutionSpace>
            struct IdentityFunctor
            {
                typedef Extension::Matrix<DataType, ExecutionSpace> Matrix;

                typedef typename Matrix::size_type        size_type;
                typedef typename Matrix::const_value_type const_value_type;

                Matrix _I;

                IdentityFunctor(const Matrix& I) : _I(I) {}

                KOKKOS_INLINE_FUNCTION void operator()(const size_type i) const { _I(i, i) = DataType(1.0); }
            };
        }

        template<typename DataType, class ExecutionSpace>
        __inline static Extension::Matrix<DataType, ExecutionSpace> Identity(const size_type& m, const size_type& n)
        {
            Extension::Matrix<DataType, ExecutionSpace> I("I", m, n);

            Internal::IdentityFunctor<DataType, ExecutionSpace> functor(I);

            Kokkos::RangePolicy<ExecutionSpace, IndexType<size_type>> initialize(0, min<size_type>(m, n));

            Kokkos::parallel_for(initialize, functor);

            return I;
        }
    }
}

// namespace Kokkos
//{
//    namespace LinearAlgebra
//    {
//        template<typename DataType, class ExecutionSpace>
//        __inline static Extension::Matrix<DataType, ExecutionSpace> Solve(const Extension::Matrix<DataType, ExecutionSpace>& A, const Extension::Matrix<DataType,
//        ExecutionSpace>& B)
//        {
//            return (m == n ? (new LUDecomposition(this)).solve(B) : (new QRDecomposition(this)).solve(B));
//        }
//
//        template<typename DataType, class ExecutionSpace>
//        __inline static Extension::Matrix<DataType, ExecutionSpace> inverse(const Extension::Matrix<DataType, ExecutionSpace>& A)
//        {
//            return solve(identity(m, m));
//        }
//    }
//}

namespace Kokkos
{
    namespace LinearAlgebra
    {
        namespace Internal
        {
            struct InitializeQMinRes
            {
            };
            struct SolveQMinRes
            {
            };

            template<typename DataType, class ExecutionSpace>
            struct QMinResVectorFunctor
            {
                typedef Extension::Matrix<DataType, ExecutionSpace> Matrix;
                typedef Extension::Vector<DataType, ExecutionSpace> Vector;

                typedef typename Matrix::size_type        size_type;
                typedef typename Matrix::value_type       value_type;
                typedef typename Matrix::const_value_type const_value_type;

                const size_type _n;
                Matrix          _L;
                Matrix          _A;
                Vector          _b;
                Vector          _x;

                Vector _v1;
                Vector _w1;
                Vector _r0;

                Matrix _chi1;
                Matrix _espilson1;

                value_type _lambda1;
                value_type _kappa0;
                value_type _mu1;

                value_type _p0;
                value_type _q0;
                value_type _u0;
                value_type _d0;
                value_type _f0;

                QMinResVectorFunctor(const Matrix& A, const Vector& b, const Vector& x) :
                    _n(A.extent(0)),
                    _L("L", _n, _n),
                    _A(A),
                    _b(b),
                    _x(x),
                    _v1("v1", x.extent(0)),
                    _w1("w1", x.extent(0)),
                    _r0("r0", x.extent(0)),
                    _lambda1(1.0),
                    _kappa0(1.0),
                    _mu1(0.0),
                    _p0(0.0),
                    _q0(0.0),
                    _u0(0.0),
                    _d0(0.0),
                    _f0(0.0),
                    _chi1("chi1", x.extent(0), x.extent(0)),
                    _espilson1("espilson1", x.extent(0), x.extent(0))
                {
                    Kokkos::deep_copy(_x, _b);
                }

                KOKKOS_INLINE_FUNCTION void operator()(const InitializeQMinRes&, const size_type) const {}

                KOKKOS_INLINE_FUNCTION void operator()(const SolveQMinRes&, const size_type) const {}
            };

        }

        template<typename DataType, class ExecutionSpace>
        __inline static Extension::Vector<DataType, ExecutionSpace> QMinRes(const Extension::Matrix<DataType, ExecutionSpace>& A,
                                                                            const Extension::Vector<DataType, ExecutionSpace>& b)
        {
            if(A.extent(0) != b.extent(0))
            {
                return Extension::Vector<DataType, ExecutionSpace>("x", b.extent(0));
            }

            Extension::Vector<DataType, ExecutionSpace> x("x", b.extent(0));

            Internal::CholeskyVectorFunctor<DataType, ExecutionSpace> functor(A, b, x);

            Kokkos::RangePolicy<ExecutionSpace, IndexType<size_type>, Internal::InitializeCholesky> initializeCholesky(0, 1);

            Kokkos::parallel_for(initializeCholesky, functor);

            Kokkos::RangePolicy<ExecutionSpace, IndexType<size_type>, Internal::SolveCholesky> solveCholesky(0, 1);

            Kokkos::parallel_for(solveCholesky, functor);

            return x;
        }
    }
}

namespace Kokkos
{
    namespace FFT
    {
        void ccopy(int n, double x[], double y[])
        {
            int i;

            for(i = 0; i < n; i++)
            {
                y[i * 2 + 0] = x[i * 2 + 0];
                y[i * 2 + 1] = x[i * 2 + 1];
            }
            return;
        }

        void step(int n, int mj, double a[], double b[], double c[], double d[], double w[], double sgn)
        {
            double ambr;
            double ambu;
            int    j;
            int    ja;
            int    jb;
            int    jc;
            int    jd;
            int    jw;
            int    k;
            int    lj;
            int    mj2;
            double wjw[2];

            mj2 = 2 * mj;
            lj  = n / mj2;

#pragma omp parallel shared(a, b, c, d, lj, mj, mj2, sgn, w) private(ambr, ambu, j, ja, jb, jc, jd, jw, k, wjw)

#pragma omp for nowait

            for(j = 0; j < lj; j++)
            {
                jw = j * mj;
                ja = jw;
                jb = ja;
                jc = j * mj2;
                jd = jc;

                wjw[0] = w[jw * 2 + 0];
                wjw[1] = w[jw * 2 + 1];

                if(sgn < 0.0)
                {
                    wjw[1] = -wjw[1];
                }

                for(k = 0; k < mj; k++)
                {
                    c[(jc + k) * 2 + 0] = a[(ja + k) * 2 + 0] + b[(jb + k) * 2 + 0];
                    c[(jc + k) * 2 + 1] = a[(ja + k) * 2 + 1] + b[(jb + k) * 2 + 1];

                    ambr = a[(ja + k) * 2 + 0] - b[(jb + k) * 2 + 0];
                    ambu = a[(ja + k) * 2 + 1] - b[(jb + k) * 2 + 1];

                    d[(jd + k) * 2 + 0] = wjw[0] * ambr - wjw[1] * ambu;
                    d[(jd + k) * 2 + 1] = wjw[1] * ambr + wjw[0] * ambu;
                }
            }

            return;
        }

        void cfft2(int n, double x[], double y[], double w[], double sgn)
        {
            int j;
            int m;
            int mj;
            int tgle;

            m  = (int)(log((double)n) / log(1.99));
            mj = 1;
            //
            //  Toggling switch for work array.
            //
            tgle = 1;
            step(n, mj, &x[0 * 2 + 0], &x[(n / 2) * 2 + 0], &y[0 * 2 + 0], &y[mj * 2 + 0], w, sgn);

            if(n == 2)
            {
                return;
            }

            for(j = 0; j < m - 2; j++)
            {
                mj = mj * 2;
                if(tgle)
                {
                    step(n, mj, &y[0 * 2 + 0], &y[(n / 2) * 2 + 0], &x[0 * 2 + 0], &x[mj * 2 + 0], w, sgn);
                    tgle = 0;
                }
                else
                {
                    step(n, mj, &x[0 * 2 + 0], &x[(n / 2) * 2 + 0], &y[0 * 2 + 0], &y[mj * 2 + 0], w, sgn);
                    tgle = 1;
                }
            }
            //
            //  Last pass thru data: move y to x if needed
            //
            if(tgle)
            {
                ccopy(n, y, x);
            }

            mj = n / 2;
            step(n, mj, &x[0 * 2 + 0], &x[(n / 2) * 2 + 0], &y[0 * 2 + 0], &y[mj * 2 + 0], w, sgn);

            return;
        }

        void cffti(int n, double w[])
        {
            double       arg;
            double       aw;
            int          i;
            int          n2;
            const double pi = 3.141592653589793;

            n2 = n / 2;
            aw = 2.0 * pi / ((double)n);

#pragma omp parallel shared(aw, n, w) private(arg, i)

#pragma omp for nowait

            for(i = 0; i < n2; i++)
            {
                arg          = aw * ((double)i);
                w[i * 2 + 0] = cos(arg);
                w[i * 2 + 1] = sin(arg);
            }
            return;
        }

        double ggl(double* seed)
        {
            double d2 = 0.2147483647e10;
            double t;
            double value;

            t     = (double)*seed;
            t     = fmod(16807.0 * t, d2);
            *seed = (double)t;
            value = (double)((t - 1.0) / (d2 - 1.0));

            return value;
        }

    }
}
