#pragma once

#if !defined(KOKKOS_EXTENSIONS)
#    error "Do not include directly. Include Extensions.hpp"
#endif
#include <Algebra/SVD.hpp>

namespace Kokkos
{
    namespace Extension
    {

        template<typename VectorType, uint32 column>
        struct MinColumnFunctor
        {
            typedef typename VectorType::traits::execution_space      ExecutionSpace;
            typedef typename VectorType::traits::data_type            DataType;
            typedef typename VectorType::traits::non_const_value_type ValueType;

            VectorType Values;

            MinColumnFunctor(const VectorType& values) : Values(values) {}

            KOKKOS_INLINE_FUNCTION constexpr void operator()(const uint32& i, ValueType& value) const
            {
                Kokkos::atomic_fetch_min(&value, Values(i, column));
            }
        };

        template<typename VectorType, uint32 column>
        KOKKOS_INLINE_FUNCTION static auto min(const VectorType& values) -> typename std::enable_if<VectorType::Rank == 2, typename VectorType::traits::non_const_value_type>::type
        {
            typedef typename VectorType::traits::execution_space      ExecutionSpace;
            typedef typename VectorType::traits::data_type            DataType;
            typedef typename VectorType::traits::non_const_value_type ValueType;

            MinColumnFunctor<VectorType, column> f(values);

            ValueType              min_value = Constants<ValueType>::Max();
            Kokkos::Min<ValueType> reducer_scalar(min_value);

            Kokkos::parallel_reduce(Kokkos::RangePolicy<ExecutionSpace>(0, values.extent(0)), f, reducer_scalar);
            Kokkos::fence();

            return min_value;
        }

        template<typename VectorType>
        KOKKOS_INLINE_FUNCTION static auto min(const VectorType& values) -> typename std::enable_if<VectorType::Rank == 2, typename VectorType::traits::non_const_value_type>::type
        {
            typedef typename VectorType::traits::execution_space      ExecutionSpace;
            typedef typename VectorType::traits::data_type            DataType;
            typedef typename VectorType::traits::non_const_value_type ValueType;

            ValueType min0 = min<VectorType, 0>(values);
            ValueType min1 = min<VectorType, 1>(values);

            return std::min(min0, min1);
        }

        template<typename VectorType, uint32 column>
        struct MaxColumnFunctor
        {
            typedef typename VectorType::traits::execution_space      ExecutionSpace;
            typedef typename VectorType::traits::data_type            DataType;
            typedef typename VectorType::traits::non_const_value_type ValueType;

            VectorType Values;

            MaxColumnFunctor(const VectorType& values) : Values(values) {}

            KOKKOS_INLINE_FUNCTION constexpr void operator()(const uint32& i, ValueType& value) const
            {
                Kokkos::atomic_fetch_max(&value, Values(i, column));
            }
        };

        template<typename VectorType, uint32 column>
        KOKKOS_INLINE_FUNCTION static auto max(const VectorType& values) -> typename std::enable_if<VectorType::Rank == 2, typename VectorType::traits::non_const_value_type>::type
        {
            typedef typename VectorType::traits::execution_space      ExecutionSpace;
            typedef typename VectorType::traits::data_type            DataType;
            typedef typename VectorType::traits::non_const_value_type ValueType;

            MaxColumnFunctor<VectorType, column> f(values);

            ValueType              max_value = Constants<ValueType>::Min();
            Kokkos::Max<ValueType> reducer_scalar(max_value);

            Kokkos::parallel_reduce(Kokkos::RangePolicy<ExecutionSpace>(0, values.extent(0)), f, reducer_scalar);
            Kokkos::fence();

            return max_value;
        }

        template<typename VectorType>
        KOKKOS_INLINE_FUNCTION static auto max(const VectorType& values) -> typename std::enable_if<VectorType::Rank == 2, typename VectorType::traits::non_const_value_type>::type
        {
            typedef typename VectorType::traits::execution_space      ExecutionSpace;
            typedef typename VectorType::traits::data_type            DataType;
            typedef typename VectorType::traits::non_const_value_type ValueType;

            ValueType max0 = max<VectorType, 0>(values);
            ValueType max1 = max<VectorType, 1>(values);

            return std::max(max0, max1);
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

        template<typename DataType, class ExecutionSpace>
        __inline static auto row(const Matrix<DataType, ExecutionSpace>& A, const size_type& r) ->
            typename std::enable_if<std::is_same_v<typename Matrix<DataType, ExecutionSpace>::traits::array_layout, Kokkos::LayoutLeft>, Vector<DataType, ExecutionSpace>>::type
        {
            return Kokkos::subview(A, Kokkos::ALL, r);
        }

        template<typename DataType, class ExecutionSpace>
        __inline static auto row(const Matrix<DataType, ExecutionSpace>& A, const size_type& r) ->
            typename std::enable_if<std::is_same_v<typename Matrix<DataType, ExecutionSpace>::traits::array_layout, Kokkos::LayoutRight>, Vector<DataType, ExecutionSpace>>::type
        {
            return Kokkos::subview(A, r, Kokkos::ALL);
        }

        template<typename DataType, class ExecutionSpace>
        __inline static auto column(const Matrix<DataType, ExecutionSpace>& A, const size_type& c) ->
            typename std::enable_if<std::is_same_v<typename Matrix<DataType, ExecutionSpace>::traits::array_layout, Kokkos::LayoutLeft>, Vector<DataType, ExecutionSpace, Kokkos::LayoutStride>>::type
        {
            return Kokkos::subview(A, c, Kokkos::ALL);
        }

        template<typename DataType, class ExecutionSpace>
        __inline static auto column(const Matrix<DataType, ExecutionSpace>& A, const size_type& c) ->
            typename std::enable_if<std::is_same_v<typename Matrix<DataType, ExecutionSpace>::traits::array_layout, Kokkos::LayoutRight>, Vector<DataType, ExecutionSpace, Kokkos::LayoutStride>>::type
        {
            return Kokkos::subview(A, Kokkos::ALL, c);
        }

        template<typename DataType, class ExecutionSpace>
        __inline static DataType norm(const Vector<DataType, ExecutionSpace>& lhs)
        {
            const size_type n = lhs.nrows();

            VectorOperators::VectorNormFunctor<Vector<DataType, ExecutionSpace>> f(lhs);

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
            for (size_type i = 0; i < n; ++i)
            {
                sum += lhs(i) * rhs(i);
            }
#else
            VectorOperators::VectorInnerProductFunctor<Vector<DataType, ExecutionSpace>, Vector<DataType, ExecutionSpace>> f(lhs, rhs);

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

            Matrix<DataType, ExecutionSpace> r(lhs.label() + " X " + rhs.label(), n, m);

            VectorOperators::VectorOuterProductFunctor<Matrix<DataType, ExecutionSpace>, Vector<DataType, ExecutionSpace>, Vector<DataType, ExecutionSpace>> f(r, lhs, rhs);

            mdrange_type<ExecutionSpace> policy(point_type<ExecutionSpace>{{0, 0}}, point_type<ExecutionSpace>{{n, m}});

            Kokkos::parallel_for("V_OuterProduct", policy, f);

            return r;
        }

        template<typename DataType, class ExecutionSpace>
        __inline static Matrix<DataType, ExecutionSpace> operator^(const Vector<DataType, ExecutionSpace>& lhs, const Vector<DataType, ExecutionSpace>& rhs)
        {
            return outer_product<DataType, ExecutionSpace>(lhs, rhs);
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

    using Extension::column;
    using Extension::inner_product;
    using Extension::norm;
    using Extension::outer_product;
    using Extension::row;
}

namespace Kokkos
{
    namespace Extension
    {
        template<typename DataType, class ExecutionSpace>
        static std::ostream& operator<<(std::ostream& s, const Matrix<DataType, ExecutionSpace>& A)
        {
            const size_type m = A.nrows();
            const size_type n = A.ncolumns();

            s << std::endl;
            s << A.label() << " [" << m << "x" << n << "]";
            s << std::endl;

            for (size_type i = 0; i < m; ++i)
            {
                for (size_type j = 0; j < n; ++j)
                {
                    s << A(i, j) << " ";
                }
                s << std::endl;
            }

            return s;
        }

        template<typename DataType, class ExecutionSpace>
        KOKKOS_INLINE_FUNCTION static System::print& operator<<(System::print& s, const Matrix<DataType, ExecutionSpace>& A)
        {
            const size_type m = A.nrows();
            const size_type n = A.ncolumns();

            s << System::endl;
#if !defined(__CUDA_ARCH__)
            s << A.label();
#endif
            s << " [" << m << "x" << n << "]" << System::endl;

            for (size_type i = 0; i < m; ++i)
            {
                for (size_type j = 0; j < n; ++j)
                {
                    s << A(i, j) << " ";
                }
                s << System::endl;
            }

            return s;
        }

        template<typename DataType, class ExecutionSpace>
        static std::istream& operator>>(std::istream& s, Extension::Matrix<DataType, ExecutionSpace>& A)
        {
            size_type m, n;

            s >> m >> n;

            if (!(m == A.nrows() && n == A.ncolumns()))
            {
                Kokkos::resize(A, m, n);
            }

            for (size_type i = 0; i < m; ++i)
            {
                for (size_type j = 0; j < n; ++j)
                {
                    s >> A(i, j);
                }
            }

            return s;
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

        namespace MatrixOperators
        {
#define MATRIX_OPS_FUNCTORS(OP_NAME, OP, ASSIGN_OP)                                                                                                                                                    \
    template<class RMatrix, class LhsMatrix, class RhsMatrix>                                                                                                                                          \
    struct MatrixMatrix##OP_NAME##Functor                                                                                                                                                              \
    {                                                                                                                                                                                                  \
        static_assert(RMatrix::Rank == 2, "RMatrix::Rank != 2");                                                                                                                                       \
        static_assert(LhsMatrix::Rank == 2, "LhsMatrix::Rank != 2");                                                                                                                                   \
        static_assert(RhsMatrix::Rank == 2, "RhsMatrix::Rank != 2");                                                                                                                                   \
                                                                                                                                                                                                       \
        typedef typename RMatrix::size_type                      size_type;                                                                                                                            \
        typedef typename LhsMatrix::traits::non_const_value_type value_type;                                                                                                                           \
                                                                                                                                                                                                       \
        RMatrix                        _r;                                                                                                                                                             \
        typename LhsMatrix::const_type _lhs;                                                                                                                                                           \
        typename RhsMatrix::const_type _rhs;                                                                                                                                                           \
                                                                                                                                                                                                       \
        MatrixMatrix##OP_NAME##Functor(RMatrix& r, const LhsMatrix& lhs, const RhsMatrix& rhs) : _r(r), _lhs(lhs), _rhs(rhs) {}                                                                        \
                                                                                                                                                                                                       \
        KOKKOS_INLINE_FUNCTION void operator()(const size_type i, const size_type j) const                                                                                                             \
        {                                                                                                                                                                                              \
            _r(i, j) = _lhs(i, j) OP _rhs(i, j);                                                                                                                                                       \
        }                                                                                                                                                                                              \
    };                                                                                                                                                                                                 \
                                                                                                                                                                                                       \
    template<class RMatrix, class LhsMatrix>                                                                                                                                                           \
    struct MatrixScalar##OP_NAME##Functor                                                                                                                                                              \
    {                                                                                                                                                                                                  \
        static_assert(RMatrix::Rank == 2, "RMatrix::Rank != 2");                                                                                                                                       \
        static_assert(LhsMatrix::Rank == 2, "LhsMatrix::Rank != 2");                                                                                                                                   \
                                                                                                                                                                                                       \
        typedef typename RMatrix::size_type                      size_type;                                                                                                                            \
        typedef typename LhsMatrix::traits::non_const_value_type value_type;                                                                                                                           \
                                                                                                                                                                                                       \
        RMatrix                        _r;                                                                                                                                                             \
        typename LhsMatrix::const_type _lhs;                                                                                                                                                           \
        value_type                     _rhs;                                                                                                                                                           \
                                                                                                                                                                                                       \
        MatrixScalar##OP_NAME##Functor(RMatrix& r, const LhsMatrix& lhs, const value_type& rhs) : _r(r), _lhs(lhs), _rhs(rhs) {}                                                                       \
                                                                                                                                                                                                       \
        KOKKOS_INLINE_FUNCTION void operator()(const size_type i, const size_type j) const                                                                                                             \
        {                                                                                                                                                                                              \
            _r(i, j) = _lhs(i, j) OP _rhs;                                                                                                                                                             \
        }                                                                                                                                                                                              \
    };                                                                                                                                                                                                 \
                                                                                                                                                                                                       \
    template<class RMatrix, class RhsMatrix>                                                                                                                                                           \
    struct ScalarMatrix##OP_NAME##Functor                                                                                                                                                              \
    {                                                                                                                                                                                                  \
        static_assert(RMatrix::Rank == 2, "RMatrix::Rank != 2");                                                                                                                                       \
        static_assert(RhsMatrix::Rank == 2, "RhsMatrix::Rank != 2");                                                                                                                                   \
                                                                                                                                                                                                       \
        typedef typename RMatrix::size_type                      size_type;                                                                                                                            \
        typedef typename RhsMatrix::traits::non_const_value_type value_type;                                                                                                                           \
                                                                                                                                                                                                       \
        RMatrix                        _r;                                                                                                                                                             \
        value_type                     _lhs;                                                                                                                                                           \
        typename RhsMatrix::const_type _rhs;                                                                                                                                                           \
                                                                                                                                                                                                       \
        ScalarMatrix##OP_NAME##Functor(RMatrix& r, const value_type& lhs, const RhsMatrix& rhs) : _r(r), _lhs(lhs), _rhs(rhs) {}                                                                       \
                                                                                                                                                                                                       \
        KOKKOS_INLINE_FUNCTION void operator()(const size_type i, const size_type j) const                                                                                                             \
        {                                                                                                                                                                                              \
            _r(i, j) = _lhs OP _rhs(i, j);                                                                                                                                                             \
        }                                                                                                                                                                                              \
    };                                                                                                                                                                                                 \
                                                                                                                                                                                                       \
    template<class LhsMatrix, class RhsMatrix>                                                                                                                                                         \
    struct MatrixMatrix##OP_NAME##AssignFunctor                                                                                                                                                        \
    {                                                                                                                                                                                                  \
        static_assert(LhsMatrix::Rank == 2, "LhsMatrix::Rank != 2");                                                                                                                                   \
        static_assert(RhsMatrix::Rank == 2, "RhsMatrix::Rank != 2");                                                                                                                                   \
                                                                                                                                                                                                       \
        typedef typename LhsMatrix::size_type                    size_type;                                                                                                                            \
        typedef typename LhsMatrix::traits::non_const_value_type value_type;                                                                                                                           \
                                                                                                                                                                                                       \
        LhsMatrix                      _lhs;                                                                                                                                                           \
        typename RhsMatrix::const_type _rhs;                                                                                                                                                           \
                                                                                                                                                                                                       \
        MatrixMatrix##OP_NAME##AssignFunctor(const LhsMatrix& lhs, const RhsMatrix& rhs) : _lhs(lhs), _rhs(rhs) {}                                                                                     \
                                                                                                                                                                                                       \
        KOKKOS_INLINE_FUNCTION void operator()(const size_type i, const size_type j) const                                                                                                             \
        {                                                                                                                                                                                              \
            _lhs(i, j) ASSIGN_OP _rhs(i, j);                                                                                                                                                           \
        }                                                                                                                                                                                              \
    };                                                                                                                                                                                                 \
                                                                                                                                                                                                       \
    template<class LhsMatrix>                                                                                                                                                                          \
    struct MatrixScalar##OP_NAME##AssignFunctor                                                                                                                                                        \
    {                                                                                                                                                                                                  \
        static_assert(LhsMatrix::Rank == 2, "LhsMatrix::Rank != 2");                                                                                                                                   \
                                                                                                                                                                                                       \
        typedef typename LhsMatrix::size_type                    size_type;                                                                                                                            \
        typedef typename LhsMatrix::traits::non_const_value_type value_type;                                                                                                                           \
                                                                                                                                                                                                       \
        LhsMatrix  _lhs;                                                                                                                                                                               \
        value_type _rhs;                                                                                                                                                                               \
                                                                                                                                                                                                       \
        MatrixScalar##OP_NAME##AssignFunctor(const LhsMatrix& lhs, const value_type& rhs) : _lhs(lhs), _rhs(rhs) {}                                                                                    \
                                                                                                                                                                                                       \
        KOKKOS_INLINE_FUNCTION void operator()(const size_type i, const size_type j) const                                                                                                             \
        {                                                                                                                                                                                              \
            _lhs(i, j) ASSIGN_OP _rhs;                                                                                                                                                                 \
        }                                                                                                                                                                                              \
    };                                                                                                                                                                                                 \
                                                                                                                                                                                                       \
    template<typename DataType, class ExecutionSpace>                                                                                                                                                  \
    __inline static Matrix<DataType, ExecutionSpace> operator OP(const Matrix<DataType, ExecutionSpace>& lhs, const Matrix<DataType, ExecutionSpace>& rhs)                                             \
    {                                                                                                                                                                                                  \
        const size_type m = lhs.nrows();                                                                                                                                                               \
        const size_type n = lhs.ncolumns();                                                                                                                                                            \
        const size_type k = rhs.ncolumns();                                                                                                                                                            \
                                                                                                                                                                                                       \
        Assert(n == rhs.nrows());                                                                                                                                                                      \
                                                                                                                                                                                                       \
        Matrix<DataType, ExecutionSpace> r(lhs.label() + #OP + rhs.label(), m, k);                                                                                                                     \
                                                                                                                                                                                                       \
        MatrixOperators::MatrixMatrix##OP_NAME##Functor<Matrix<DataType, ExecutionSpace>, Matrix<DataType, ExecutionSpace>, Matrix<DataType, ExecutionSpace>> f(r, lhs, rhs);                          \
                                                                                                                                                                                                       \
        mdrange_type<ExecutionSpace> policy(point_type<ExecutionSpace>{{0, 0}}, point_type<ExecutionSpace>{{m, k}});                                                                                   \
                                                                                                                                                                                                       \
        Kokkos::parallel_for("V_" #OP_NAME, policy, f);                                                                                                                                                \
                                                                                                                                                                                                       \
        return r;                                                                                                                                                                                      \
    }                                                                                                                                                                                                  \
                                                                                                                                                                                                       \
    template<typename DataType, class ExecutionSpace>                                                                                                                                                  \
    __inline static Matrix<DataType, ExecutionSpace> operator OP(const DataType& lhs, const Matrix<DataType, ExecutionSpace>& rhs)                                                                     \
    {                                                                                                                                                                                                  \
        const size_type m = rhs.nrows();                                                                                                                                                               \
        const size_type n = rhs.ncolumns();                                                                                                                                                            \
                                                                                                                                                                                                       \
        Matrix<DataType, ExecutionSpace> r(std::to_string(lhs) + #OP + rhs.label(), m, n);                                                                                                             \
                                                                                                                                                                                                       \
        MatrixOperators::ScalarMatrix##OP_NAME##Functor<Matrix<DataType, ExecutionSpace>, Matrix<DataType, ExecutionSpace>> f(r, lhs, rhs);                                                            \
                                                                                                                                                                                                       \
        mdrange_type<ExecutionSpace> policy(point_type<ExecutionSpace>{{0, 0}}, point_type<ExecutionSpace>{{m, n}});                                                                                   \
                                                                                                                                                                                                       \
        Kokkos::parallel_for("V_" #OP_NAME, policy, f);                                                                                                                                                \
                                                                                                                                                                                                       \
        return r;                                                                                                                                                                                      \
    }                                                                                                                                                                                                  \
                                                                                                                                                                                                       \
    template<typename DataType, class ExecutionSpace>                                                                                                                                                  \
    __inline static Matrix<DataType, ExecutionSpace> operator OP(const Matrix<DataType, ExecutionSpace>& lhs, const DataType& rhs)                                                                     \
    {                                                                                                                                                                                                  \
        const size_type m = lhs.nrows();                                                                                                                                                               \
        const size_type n = lhs.ncolumns();                                                                                                                                                            \
                                                                                                                                                                                                       \
        Matrix<DataType, ExecutionSpace> r(lhs.label() + #OP + std::to_string(rhs), m, n);                                                                                                             \
                                                                                                                                                                                                       \
        MatrixOperators::MatrixScalar##OP_NAME##Functor<Matrix<DataType, ExecutionSpace>, Matrix<DataType, ExecutionSpace>> f(r, lhs, rhs);                                                            \
                                                                                                                                                                                                       \
        mdrange_type<ExecutionSpace> policy(point_type<ExecutionSpace>{{0, 0}}, point_type<ExecutionSpace>{{m, n}});                                                                                   \
                                                                                                                                                                                                       \
        Kokkos::parallel_for("V_" #OP_NAME, policy, f);                                                                                                                                                \
                                                                                                                                                                                                       \
        return r;                                                                                                                                                                                      \
    }                                                                                                                                                                                                  \
                                                                                                                                                                                                       \
    template<typename DataType, class ExecutionSpace>                                                                                                                                                  \
    __inline static Matrix<DataType, ExecutionSpace> operator ASSIGN_OP(Matrix<DataType, ExecutionSpace>& lhs, const Matrix<DataType, ExecutionSpace>& rhs)                                            \
    {                                                                                                                                                                                                  \
        const size_type m = lhs.nrows();                                                                                                                                                               \
        const size_type n = lhs.ncolumns();                                                                                                                                                            \
        const size_type k = rhs.ncolumns();                                                                                                                                                            \
                                                                                                                                                                                                       \
        Assert(n == rhs.nrows());                                                                                                                                                                      \
                                                                                                                                                                                                       \
        Matrix<DataType, ExecutionSpace> r(lhs.label() + #ASSIGN_OP + rhs.label(), m, k);                                                                                                              \
                                                                                                                                                                                                       \
        MatrixOperators::MatrixMatrix##OP_NAME##AssignFunctor<Matrix<DataType, ExecutionSpace>, Matrix<DataType, ExecutionSpace>> f(lhs, rhs);                                                         \
                                                                                                                                                                                                       \
        mdrange_type<ExecutionSpace> policy(point_type<ExecutionSpace>{{0, 0}}, point_type<ExecutionSpace>{{m, k}});                                                                                   \
                                                                                                                                                                                                       \
        Kokkos::parallel_for("V_" #OP_NAME "Assign", policy, f);                                                                                                                                       \
                                                                                                                                                                                                       \
        return r;                                                                                                                                                                                      \
    }                                                                                                                                                                                                  \
                                                                                                                                                                                                       \
    template<typename DataType, class ExecutionSpace>                                                                                                                                                  \
    __inline static Matrix<DataType, ExecutionSpace> operator ASSIGN_OP(Matrix<DataType, ExecutionSpace>& lhs, const DataType& rhs)                                                                    \
    {                                                                                                                                                                                                  \
        const size_type m = rhs.nrows();                                                                                                                                                               \
        const size_type n = rhs.ncolumns();                                                                                                                                                            \
                                                                                                                                                                                                       \
        Matrix<DataType, ExecutionSpace> r(lhs.label() + #ASSIGN_OP + std::to_string(rhs), n);                                                                                                         \
                                                                                                                                                                                                       \
        MatrixOperators::MatrixScalar##OP_NAME##AssignFunctor<Matrix<DataType, ExecutionSpace>> f(lhs, rhs);                                                                                           \
                                                                                                                                                                                                       \
        mdrange_type<ExecutionSpace> policy(point_type<ExecutionSpace>{{0, 0}}, point_type<ExecutionSpace>{{m, n}});                                                                                   \
                                                                                                                                                                                                       \
        Kokkos::parallel_for("V_" #OP_NAME "Scalar", policy, f);                                                                                                                                       \
                                                                                                                                                                                                       \
        return r;                                                                                                                                                                                      \
    }

            MATRIX_OPS_FUNCTORS(Plus, +, +=)
            MATRIX_OPS_FUNCTORS(Minus, -, -=)

#undef MATRIX_OPS_FUNCTORS

#define MATRIX_OPS_FUNCTORS(OP_NAME, OP, ASSIGN_OP)                                                                                                                                                    \
    template<class RMatrix, class LhsMatrix>                                                                                                                                                           \
    struct MatrixScalar##OP_NAME##Functor                                                                                                                                                              \
    {                                                                                                                                                                                                  \
        static_assert(RMatrix::Rank == 2, "RMatrix::Rank != 2");                                                                                                                                       \
        static_assert(LhsMatrix::Rank == 2, "LhsMatrix::Rank != 2");                                                                                                                                   \
                                                                                                                                                                                                       \
        typedef typename RMatrix::size_type                      size_type;                                                                                                                            \
        typedef typename LhsMatrix::traits::non_const_value_type value_type;                                                                                                                           \
                                                                                                                                                                                                       \
        RMatrix                        _r;                                                                                                                                                             \
        typename LhsMatrix::const_type _lhs;                                                                                                                                                           \
        value_type                     _rhs;                                                                                                                                                           \
                                                                                                                                                                                                       \
        MatrixScalar##OP_NAME##Functor(RMatrix& r, const LhsMatrix& lhs, const value_type& rhs) : _r(r), _lhs(lhs), _rhs(rhs) {}                                                                       \
                                                                                                                                                                                                       \
        KOKKOS_INLINE_FUNCTION void operator()(const size_type i, const size_type j) const                                                                                                             \
        {                                                                                                                                                                                              \
            _r(i, j) = _lhs(i, j) OP _rhs;                                                                                                                                                             \
        }                                                                                                                                                                                              \
    };                                                                                                                                                                                                 \
                                                                                                                                                                                                       \
    template<class RMatrix, class RhsMatrix>                                                                                                                                                           \
    struct ScalarMatrix##OP_NAME##Functor                                                                                                                                                              \
    {                                                                                                                                                                                                  \
        static_assert(RMatrix::Rank == 2, "RMatrix::Rank != 2");                                                                                                                                       \
        static_assert(RhsMatrix::Rank == 2, "RhsMatrix::Rank != 2");                                                                                                                                   \
                                                                                                                                                                                                       \
        typedef typename RMatrix::size_type                      size_type;                                                                                                                            \
        typedef typename RhsMatrix::traits::non_const_value_type value_type;                                                                                                                           \
                                                                                                                                                                                                       \
        RMatrix                        _r;                                                                                                                                                             \
        value_type                     _lhs;                                                                                                                                                           \
        typename RhsMatrix::const_type _rhs;                                                                                                                                                           \
                                                                                                                                                                                                       \
        ScalarMatrix##OP_NAME##Functor(RMatrix& r, const value_type& lhs, const RhsMatrix& rhs) : _r(r), _lhs(lhs), _rhs(rhs) {}                                                                       \
                                                                                                                                                                                                       \
        KOKKOS_INLINE_FUNCTION void operator()(const size_type i, const size_type j) const                                                                                                             \
        {                                                                                                                                                                                              \
            _r(i, j) = _lhs OP _rhs(i, j);                                                                                                                                                             \
        }                                                                                                                                                                                              \
    };                                                                                                                                                                                                 \
                                                                                                                                                                                                       \
    template<class LhsMatrix>                                                                                                                                                                          \
    struct MatrixScalar##OP_NAME##AssignFunctor                                                                                                                                                        \
    {                                                                                                                                                                                                  \
        static_assert(LhsMatrix::Rank == 2, "LhsMatrix::Rank != 2");                                                                                                                                   \
                                                                                                                                                                                                       \
        typedef typename LhsMatrix::size_type                    size_type;                                                                                                                            \
        typedef typename LhsMatrix::traits::non_const_value_type value_type;                                                                                                                           \
                                                                                                                                                                                                       \
        LhsMatrix  _lhs;                                                                                                                                                                               \
        value_type _rhs;                                                                                                                                                                               \
                                                                                                                                                                                                       \
        MatrixScalar##OP_NAME##AssignFunctor(const LhsMatrix& lhs, const value_type& rhs) : _lhs(lhs), _rhs(rhs) {}                                                                                    \
                                                                                                                                                                                                       \
        KOKKOS_INLINE_FUNCTION void operator()(const size_type i, const size_type j) const                                                                                                             \
        {                                                                                                                                                                                              \
            _lhs(i, j) ASSIGN_OP _rhs;                                                                                                                                                                 \
        }                                                                                                                                                                                              \
    };                                                                                                                                                                                                 \
                                                                                                                                                                                                       \
    template<typename DataType, class ExecutionSpace>                                                                                                                                                  \
    __inline static Matrix<DataType, ExecutionSpace> operator OP(const DataType& lhs, const Matrix<DataType, ExecutionSpace>& rhs)                                                                     \
    {                                                                                                                                                                                                  \
        const size_type m = rhs.nrows();                                                                                                                                                               \
        const size_type n = rhs.ncolumns();                                                                                                                                                            \
                                                                                                                                                                                                       \
        Matrix<DataType, ExecutionSpace> r(std::to_string(lhs) + #OP + rhs.label(), m, n);                                                                                                             \
                                                                                                                                                                                                       \
        MatrixOperators::ScalarMatrix##OP_NAME##Functor<Matrix<DataType, ExecutionSpace>, Matrix<DataType, ExecutionSpace>> f(r, lhs, rhs);                                                            \
                                                                                                                                                                                                       \
        mdrange_type<ExecutionSpace> policy(point_type<ExecutionSpace>{{0, 0}}, point_type<ExecutionSpace>{{m, n}});                                                                                   \
                                                                                                                                                                                                       \
        Kokkos::parallel_for("V_" #OP_NAME, policy, f);                                                                                                                                                \
                                                                                                                                                                                                       \
        return r;                                                                                                                                                                                      \
    }                                                                                                                                                                                                  \
                                                                                                                                                                                                       \
    template<typename DataType, class ExecutionSpace>                                                                                                                                                  \
    __inline static Matrix<DataType, ExecutionSpace> operator OP(const Matrix<DataType, ExecutionSpace>& lhs, const DataType& rhs)                                                                     \
    {                                                                                                                                                                                                  \
        const size_type m = lhs.nrows();                                                                                                                                                               \
        const size_type n = lhs.ncolumns();                                                                                                                                                            \
                                                                                                                                                                                                       \
        Matrix<DataType, ExecutionSpace> r(lhs.label() + #OP + std::to_string(rhs), m, n);                                                                                                             \
                                                                                                                                                                                                       \
        MatrixOperators::MatrixScalar##OP_NAME##Functor<Matrix<DataType, ExecutionSpace>, Matrix<DataType, ExecutionSpace>> f(r, lhs, rhs);                                                            \
                                                                                                                                                                                                       \
        mdrange_type<ExecutionSpace> policy(point_type<ExecutionSpace>{{0, 0}}, point_type<ExecutionSpace>{{m, n}});                                                                                   \
                                                                                                                                                                                                       \
        Kokkos::parallel_for("V_" #OP_NAME, policy, f);                                                                                                                                                \
                                                                                                                                                                                                       \
        return r;                                                                                                                                                                                      \
    }                                                                                                                                                                                                  \
                                                                                                                                                                                                       \
    template<typename DataType, class ExecutionSpace>                                                                                                                                                  \
    __inline static Matrix<DataType, ExecutionSpace> operator ASSIGN_OP(Matrix<DataType, ExecutionSpace>& lhs, const DataType& rhs)                                                                    \
    {                                                                                                                                                                                                  \
        const size_type m = lhs.nrows();                                                                                                                                                               \
        const size_type n = lhs.ncolumns();                                                                                                                                                            \
                                                                                                                                                                                                       \
        Matrix<DataType, ExecutionSpace> r(lhs.label() + #ASSIGN_OP + std::to_string(rhs), m, n);                                                                                                      \
                                                                                                                                                                                                       \
        MatrixOperators::MatrixScalar##OP_NAME##AssignFunctor<Matrix<DataType, ExecutionSpace>> f(lhs, rhs);                                                                                           \
                                                                                                                                                                                                       \
        mdrange_type<ExecutionSpace> policy(point_type<ExecutionSpace>{{0, 0}}, point_type<ExecutionSpace>{{m, n}});                                                                                   \
                                                                                                                                                                                                       \
        Kokkos::parallel_for("V_" #OP_NAME "Scalar", policy, f);                                                                                                                                       \
                                                                                                                                                                                                       \
        return r;                                                                                                                                                                                      \
    }
            MATRIX_OPS_FUNCTORS(Multiply, *, *=)
            MATRIX_OPS_FUNCTORS(Divide, /, /=)

#undef MATRIX_OPS_FUNCTORS

            template<class RMatrix, class LhsMatrix, class RhsMatrix>
            struct MatrixMatrixMultiplyFunctor
            {
                static_assert(RMatrix::Rank == 2, "RMatrix::Rank != 2");
                static_assert(LhsMatrix::Rank == 2, "LhsMatrix::Rank != 2");
                static_assert(RhsMatrix::Rank == 2, "RhsMatrix::Rank != 2");

                typedef typename LhsMatrix::size_type                    size_type;
                typedef typename LhsMatrix::const_value_type             const_value_type;
                typedef typename LhsMatrix::traits::non_const_value_type value_type;

                RMatrix                        _r;
                typename LhsMatrix::const_type _lhs;
                typename RhsMatrix::const_type _rhs;
                const_value_type               n;

                MatrixMatrixMultiplyFunctor(RMatrix& r, const LhsMatrix& lhs, const RhsMatrix& rhs) : _r(r), _lhs(lhs), _rhs(rhs), n(lhs.ncolumns()) {}

                KOKKOS_INLINE_FUNCTION void operator()(const size_type i, const size_type k) const
                {
                    value_type sum = Constants<value_type>::Zero();

                    for (size_type j = 0; j < n; ++j)
                    {
                        sum += (_lhs(i, j) * _rhs(j, k));
                    }

                    _r(i, k) = sum;
                }
            };

            template<class RVector, class LhsMatrix, class RhsVector>
            struct MatrixVectorMultiplyFunctor
            {
                static_assert(RVector::Rank == 1, "RVector::Rank != 1");
                static_assert(LhsMatrix::Rank == 2, "LhsMatrix::Rank != 2");
                static_assert(RhsVector::Rank == 1, "RhsVector::Rank != 1");

                typedef typename LhsMatrix::size_type                    size_type;
                typedef typename LhsMatrix::const_value_type             const_value_type;
                typedef typename LhsMatrix::traits::non_const_value_type value_type;

                RVector                        _r;
                typename LhsMatrix::const_type _lhs;
                typename RhsVector::const_type _rhs;
                const_value_type               n;

                MatrixVectorMultiplyFunctor(RVector& r, const LhsMatrix& lhs, const RhsVector& rhs) : _r(r), _lhs(lhs), _rhs(rhs), n(lhs.ncolumns()) {}

                KOKKOS_INLINE_FUNCTION void operator()(const size_type i) const
                {
                    value_type sum = 0.0;

                    for (size_type j = 0; j < n; ++j)
                    {
                        sum += _lhs(i, j) * _rhs(j);
                    }

                    _r(i) = sum;
                }
            };

            template<class RMatrix, class LhsMatrix>
            struct MatrixNegateFunctor
            {
                static_assert(RMatrix::Rank == 2, "RMatrix::Rank != 2");
                static_assert(LhsMatrix::Rank == 2, "LhsMatrix::Rank != 2");

                typedef typename LhsMatrix::size_type                    size_type;
                typedef typename LhsMatrix::const_value_type             const_value_type;
                typedef typename LhsMatrix::traits::non_const_value_type value_type;

                RMatrix                        _r;
                typename LhsMatrix::const_type _lhs;

                MatrixNegateFunctor(RMatrix& r, const LhsMatrix& lhs) : _r(r), _lhs(lhs) {}

                KOKKOS_INLINE_FUNCTION void operator()(const size_type i, const size_type j) const
                {
                    _r(i, j) = -_lhs(i, j);
                }
            };

            template<class RVector, class LhsMatrix>
            struct MatrixDiagonalFunctor
            {
                static_assert(RVector::Rank == 1, "RVector::Rank != 1");
                static_assert(LhsMatrix::Rank == 2, "LhsMatrix::Rank != 2");

                typedef typename LhsMatrix::size_type                    size_type;
                typedef typename LhsMatrix::const_value_type             const_value_type;
                typedef typename LhsMatrix::traits::non_const_value_type value_type;

                RVector                        _r;
                typename LhsMatrix::const_type _lhs;

                MatrixDiagonalFunctor(RVector& r, const LhsMatrix& lhs) : _r(r), _lhs(lhs) {}

                KOKKOS_INLINE_FUNCTION void operator()(const size_type i) const
                {
                    _r(i) = _lhs(i, i);
                }
            };

            template<class RMatrix, class LhsMatrix>
            struct MatrixDiagonalMatrixFunctor
            {
                static_assert(RMatrix::Rank == 2, "RVector::Rank != 2");
                static_assert(LhsMatrix::Rank == 2, "LhsMatrix::Rank != 2");

                typedef typename LhsMatrix::size_type                    size_type;
                typedef typename LhsMatrix::const_value_type             const_value_type;
                typedef typename LhsMatrix::traits::non_const_value_type value_type;

                RMatrix                        _r;
                typename LhsMatrix::const_type _lhs;

                MatrixDiagonalMatrixFunctor(RMatrix& r, const LhsMatrix& lhs) : _r(r), _lhs(lhs) {}

                KOKKOS_INLINE_FUNCTION void operator()(const size_type i, const size_type j) const
                {
                    if (i == j)
                    {
                        _r(i, j) = _lhs(i, j);
                    }
                    else
                    {
                        _r(i, j) = 0.0;
                    }
                }
            };

            template<class RMatrix, class LhsMatrix>
            struct MatrixTransposeFunctor
            {
                static_assert(RMatrix::Rank == 2, "RMatrix::Rank != 2");
                static_assert(LhsMatrix::Rank == 2, "LhsMatrix::Rank != 2");

                typedef typename LhsMatrix::size_type                    size_type;
                typedef typename LhsMatrix::const_value_type             const_value_type;
                typedef typename LhsMatrix::traits::non_const_value_type value_type;

                RMatrix                        _r;
                typename LhsMatrix::const_type _lhs;

                MatrixTransposeFunctor(RMatrix& r, const LhsMatrix& lhs) : _r(r), _lhs(lhs) {}

                KOKKOS_INLINE_FUNCTION void operator()(const size_type i, const size_type j) const
                {
                    _r(j, i) = _lhs(i, j);
                }
            };

            template<class LhsMatrix>
            struct MatrixNormFunctor
            {
                static_assert(LhsMatrix::Rank == 2, "LhsMatrix::Rank != 2");

                typedef typename LhsMatrix::size_type                    size_type;
                typedef typename LhsMatrix::traits::non_const_value_type value_type;

                typename LhsMatrix::const_type _lhs;

                MatrixNormFunctor(const LhsMatrix& lhs) : _lhs(lhs) {}

                KOKKOS_INLINE_FUNCTION void operator()(const size_type i, const size_type j, value_type& sum) const
                {
                    sum += abs(_lhs(i, j)) * abs(_lhs(i, j));
                }
            };
        }

        using MatrixOperators::operator+;
        using MatrixOperators::operator-;
        using MatrixOperators::operator*;
        using MatrixOperators::operator/;
        using MatrixOperators::operator+=;
        using MatrixOperators::operator-=;
        using MatrixOperators::operator*=;
        using MatrixOperators::operator/=;

        template<typename DataType, class ExecutionSpace>
        __inline static Matrix<DataType, ExecutionSpace> operator-(const Matrix<DataType, ExecutionSpace>& lhs)
        {
            const size_type m = lhs.nrows();
            const size_type n = lhs.ncolumns();

            Matrix<DataType, ExecutionSpace> r("-" + lhs.label(), m, n);

            MatrixOperators::MatrixNegateFunctor<Matrix<DataType, ExecutionSpace>, Matrix<DataType, ExecutionSpace>> f(r, lhs);

            mdrange_type<ExecutionSpace> policy(point_type<ExecutionSpace>{{0, 0}}, point_type<ExecutionSpace>{{m, n}});

            Kokkos::parallel_for("M_Negate", policy, f);

            return r;
        }

        template<typename DataType, class ExecutionSpace>
        __inline static Matrix<DataType, ExecutionSpace> operator*(const Matrix<DataType, ExecutionSpace>& lhs, const Matrix<DataType, ExecutionSpace>& rhs)
        {
            const size_type m = lhs.nrows();
            const size_type n = lhs.ncolumns();
            const size_type k = rhs.ncolumns();

            Assert(n == rhs.nrows());

            Matrix<DataType, ExecutionSpace> r(lhs.label() + " * " + rhs.label(), m, k);

            MatrixOperators::MatrixMatrixMultiplyFunctor<Matrix<DataType, ExecutionSpace>, Matrix<DataType, ExecutionSpace>, Matrix<DataType, ExecutionSpace>> f(r, lhs, rhs);

            mdrange_type<ExecutionSpace> policy(point_type<ExecutionSpace>{{0, 0}}, point_type<ExecutionSpace>{{m, k}});

            Kokkos::parallel_for("M_Multiply", policy, f);

            return r;
        }

        template<typename DataType, class ExecutionSpace>
        __inline static Vector<DataType, ExecutionSpace> operator*(const Matrix<DataType, ExecutionSpace>& lhs, const Vector<DataType, ExecutionSpace>& rhs)
        {
            const size_type m = lhs.nrows();
            const size_type n = lhs.ncolumns();

            Assert(n == rhs.extent(0));

            Vector<DataType, ExecutionSpace> r(lhs.label() + " * " + rhs.label(), m);

            MatrixOperators::MatrixVectorMultiplyFunctor<Vector<DataType, ExecutionSpace>, Matrix<DataType, ExecutionSpace>, Vector<DataType, ExecutionSpace>> f(r, lhs, rhs);

            Kokkos::RangePolicy<ExecutionSpace> policy(0, m);

            Kokkos::parallel_for("M_Multiply", policy, f);

            return r;
        }

        template<typename DataType, class ExecutionSpace>
        __inline static DataType norm(const Matrix<DataType, ExecutionSpace>& lhs)
        {
            const size_type m = lhs.nrows();
            const size_type n = lhs.ncolumns();

            MatrixOperators::MatrixNormFunctor<Vector<DataType, ExecutionSpace>> f(lhs);

            DataType sum;

            mdrange_type<ExecutionSpace> policy(point_type<ExecutionSpace>{{0, 0}}, point_type<ExecutionSpace>{{m, n}});

            Kokkos::parallel_reduce("V_Norm", policy, f, sum);

            return sqrt(sum);
        }

        template<typename DataType, class ExecutionSpace>
        __inline static Vector<DataType, ExecutionSpace> diagonal(const Matrix<DataType, ExecutionSpace>& lhs)
        {
            const size_type m = lhs.nrows();
            const size_type n = lhs.ncolumns();

            const size_type v = std::min(n, m);

            Vector<DataType, ExecutionSpace> r(lhs.label(), v);

            MatrixOperators::MatrixDiagonalFunctor<Vector<DataType, ExecutionSpace>, Matrix<DataType, ExecutionSpace>> f(r, lhs);

            Kokkos::parallel_for(v, f);

            return r;
        }

        template<typename DataType, class ExecutionSpace>
        __inline static Matrix<std::remove_cv_t<DataType>, ExecutionSpace> diagonal_matrix(const Vector<DataType, ExecutionSpace>& lhs)
        {
            Matrix<std::remove_cv_t<DataType>, ExecutionSpace> r(lhs);

            return r;
        }

        template<typename DataType, class ExecutionSpace>
        __inline static Matrix<std::remove_cv_t<DataType>, ExecutionSpace> diagonal_matrix(const Matrix<DataType, ExecutionSpace>& lhs)
        {
            const size_type m = lhs.nrows();
            const size_type n = lhs.ncolumns();

            Matrix<std::remove_cv_t<DataType>, ExecutionSpace> r(lhs.label(), m, n);

            mdrange_type<ExecutionSpace> policy(point_type<ExecutionSpace>{{0, 0}}, point_type<ExecutionSpace>{{m, n}});

            MatrixOperators::MatrixDiagonalMatrixFunctor<Matrix<std::remove_cv_t<DataType>, ExecutionSpace>, Matrix<std::remove_cv_t<DataType>, ExecutionSpace>> f(r, lhs);

            Kokkos::parallel_for(policy, f);

            return r;
        }

        template<typename DataType, class ExecutionSpace>
        __inline static Matrix<DataType, ExecutionSpace> identity(const size_type m, const size_type n)
        {
            const size_type v = std::min(n, m);

            Matrix<DataType, ExecutionSpace> r("I", m, n);

            mdrange_type<ExecutionSpace> policy(point_type<ExecutionSpace>{{0, 0}}, point_type<ExecutionSpace>{{m, n}});

            Kokkos::parallel_for(policy,
                                 [=] __host__ __device__(const size_type i, const size_type j)
                                 {
                                     if (i == j)
                                     {
                                         r(i, j) = 1.0;
                                     }
                                     else
                                     {
                                         r(i, j) = 0.0;
                                     }
                                 });

            return r;
        }

        template<typename DataType, class ExecutionSpace>
        __inline static Vector<DataType, ExecutionSpace> transpose(const Vector<DataType, ExecutionSpace>& lhs)
        {
            return lhs;
        }

        // template<typename DataType, class ExecutionSpace>
        //__inline static Matrix<DataType, ExecutionSpace> transpose(const Matrix<DataType, ExecutionSpace>& lhs)
        //{
        //    const size_type m = lhs.nrows();
        //    const size_type n = lhs.ncolumns();
        //
        //    Matrix<DataType, ExecutionSpace> r("t(" + lhs.label() + ")", n, m);
        //
        //    MatrixOperators::MatrixTransposeFunctor<Matrix<DataType, ExecutionSpace>, Matrix<DataType, ExecutionSpace>> f(r, lhs);
        //
        //    mdrange_type<ExecutionSpace> policy(point_type<ExecutionSpace>{{0, 0}}, point_type<ExecutionSpace>{{m, n}});
        //
        //    Kokkos::parallel_for("M_Transpose", policy, f);
        //
        //    return r;
        //}

        template<typename DataType, class ExecutionSpace>
        __inline static Matrix<DataType, ExecutionSpace> transpose(const Matrix<DataType, ExecutionSpace>& lhs)
        {
            const size_type m = lhs.nrows();
            const size_type n = lhs.ncolumns();

            Matrix<DataType, ExecutionSpace> r(lhs.label() + STRING("^T"), n, m);

            if constexpr (std::is_same_v<ExecutionSpace, Kokkos::Cuda>)
            {
                cudaMemcpy(r.View().data(), lhs.View().data(), lhs.size() * sizeof(DataType), cudaMemcpyKind::cudaMemcpyDeviceToDevice);
            }
            else
            {
                cudaMemcpy(r.View().data(), lhs.View().data(), lhs.size() * sizeof(DataType), cudaMemcpyKind::cudaMemcpyHostToHost);
            }

            return r;
        }

        template<typename DataType, class ExecutionSpace>
        __inline static Matrix<DataType, ExecutionSpace> inverse(const Matrix<DataType, ExecutionSpace>& lhs)
        {
            const size_type m = lhs.nrows();
            const size_type n = lhs.ncolumns();

            Matrix<DataType, ExecutionSpace> a("a", m, n);

            Kokkos::deep_copy(a.View(), lhs.View());

            Matrix<DataType, ExecutionSpace> b("b", n, m);

            Kokkos::deep_copy(b.View(), 0.0);

            Vector<int32, ExecutionSpace> indxc("indxc", n);
            Vector<int32, ExecutionSpace> indxr("indxr", n);
            Vector<int32, ExecutionSpace> ipiv("indxr", n);

            Kokkos::deep_copy(ipiv, 0);

            Kokkos::parallel_for(1,
                                 [=] __host__ __device__(const int32)
                                 {
                                     int32  icol;
                                     int32  irow;
                                     double big;
                                     double dum;
                                     double pivinv;

                                     for (int32 i = 0; i < n; i++)
                                     {
                                         big = 0.0;

                                         for (int32 j = 0; j < n; j++)
                                         {
                                             if (ipiv(j) != 1)
                                             {
                                                 for (int32 k = 0; k < n; k++)
                                                 {
                                                     if (ipiv(k) == 0 && abs<DataType>(a(j, k)) >= big)
                                                     {
                                                         big  = abs<DataType>(a(j, k));
                                                         irow = j;
                                                         icol = k;
                                                     }
                                                 }
                                             }
                                         }

                                         ++ipiv(icol);

                                         if (irow != icol)
                                         {
                                             for (int32 l = 0; l < n; l++)
                                             {
                                                 swap(a(irow, l), a(icol, l));
                                             }

                                             for (int32 l = 0; l < m; l++)
                                             {
                                                 swap(b(irow, l), b(icol, l));
                                             }
                                         }

                                         indxr(i) = irow;
                                         indxc(i) = icol;

                                         if (a(icol, icol) == 0.0)
                                         {
                                             ThrowException("gaussj: Singular Matrix");
                                         }

                                         pivinv = 1.0 / a(icol, icol);

                                         a(icol, icol) = 1.0;

                                         for (int32 l = 0; l < n; l++)
                                         {
                                             a(icol, l) *= pivinv;
                                         }

                                         for (int32 l = 0; l < m; l++)
                                         {
                                             b(icol, l) *= pivinv;
                                         }

                                         for (int32 ll = 0; ll < n; ll++)
                                         {
                                             if (ll != icol)
                                             {
                                                 dum = a(ll, icol);

                                                 a(ll, icol) = 0.0;

                                                 for (int32 l = 0; l < n; l++)
                                                 {
                                                     a(ll, l) -= a(icol, l) * dum;
                                                 }

                                                 for (int32 l = 0; l < m; l++)
                                                 {
                                                     b(ll, l) -= b(icol, l) * dum;
                                                 }
                                             }
                                         }
                                     }
                                     // p
                                     for (int32 l = n - 1; l >= 0; --l)
                                     {
                                         if (indxr(l) != indxc(l))
                                         {
                                             for (int32 k = 0; k < n; k++)
                                             {
                                                 swap(a(k, indxr(l)), a(k, indxc(l)));
                                             }
                                         }
                                     }
                                 });

            return a;
        }

        template<typename DataType, class ExecutionSpace>
        __inline static Matrix<DataType, ExecutionSpace> pinverse(const Matrix<DataType, ExecutionSpace>& A, const DataType rcond = Constants<DataType>::Epsilon())
        {
            const uint32 m = A.nrows();
            const uint32 n = A.ncolumns();

            Matrix<DataType, ExecutionSpace> U("U", m, m);
            Matrix<DataType, ExecutionSpace> V("V", n, n);
            Vector<DataType, ExecutionSpace> u("u", n);

            Matrix<DataType, ExecutionSpace> Sigma_pinv("Sigma_pinv", n, m);
            Matrix<DataType, ExecutionSpace> A_pinv("A_pinv", n, m);
            Matrix<DataType, ExecutionSpace> _tmp_mat("_tmp_mat", n, m);

            NumericalMethods::Algebra::SVD svd(A);
            Kokkos::deep_copy(V.View(), svd.v.View());
            Kokkos::deep_copy(u, svd.w);

            Kokkos::deep_copy(U.View(), 0.0);

            for (uint32 i = 0; i < n; ++i)
            {
                for (uint32 j = 0; j < m; ++j)
                {
                    U(i, j) = svd.u(i, j);
                }
            }

            Kokkos::deep_copy(Sigma_pinv.View(), 0.0);

            DataType cutoff = rcond * max(u);

            DataType x;
            for (uint32 i = 0; i < m; ++i)
            {
                if (u(i) > cutoff)
                {
                    x = 1.0 / u(i);
                }
                else
                {
                    x = 0.0;
                }
                Sigma_pinv(i, i) = x;
            }

            KokkosBlas::gemm("N", "N", 1.0, V.View(), Sigma_pinv.View(), 0.0, _tmp_mat.View());

            KokkosBlas::gemm("N", "T", 1.0, _tmp_mat.View(), U.View(), 0.0, A_pinv.View());

            return A_pinv;
        }

        template<typename DataType, class ExecutionSpace>
        __inline static Matrix<DataType, ExecutionSpace> covariance(const Matrix<DataType, ExecutionSpace>& lhs)
        {
            const size_type m = lhs.nrows();
            const size_type n = lhs.ncolumns();

            Matrix<DataType, ExecutionSpace> r("covariance", m, n);

            Kokkos::Extension::Vector<DataType, ExecutionSpace> means("means", n);

            means = Kokkos::Extension::SumByColumn(lhs);
            means /= DataType(m);

            Kokkos::Extension::Matrix<DataType, ExecutionSpace> B("means", m, n);
            Kokkos::deep_copy(B.View(), lhs.View());

            Kokkos::parallel_for(Kokkos::RangePolicy<ExecutionSpace>(0, n),
                                 [=] __host__ __device__(const size_type col_idx)
                                 {
                                     for (size_type row_idx = 0; row_idx < m; ++row_idx)
                                     {
                                         B(row_idx, col_idx) -= means(col_idx);
                                     }
                                 });

            // Covariance Matrix
            Matrix CovarianceMatrix = (1.0 / (DataType(m) - 1.0)) * (transpose(B) * B);

            Kokkos::deep_copy(r, CovarianceMatrix);

            return r;
        }

        template<typename DataType, Integer TIndex>
        KOKKOS_FORCEINLINE_FUNCTION static constexpr DataType checkerboard_pattern(REF(TIndex) i, REF(TIndex) j)
        {
            return ((i + j) % 2 == 0) ? 1.0 : -1.0;
        }

        template<typename DataType, class ExecutionSpace>
        __inline static void cofactor(const Matrix<DataType, ExecutionSpace>& A, Matrix<DataType, ExecutionSpace>& cfm, const uint32 p, const uint32 q)
        {
            const uint32 col_end = A.ncolumns() - 1;

            uint32 i = 0;
            uint32 j = 0;

            for (uint32 row = 0; row < A.nrows(); row++)
            {
                for (uint32 col = 0; col < A.ncolumns(); col++)
                {
                    if (row != p && col != q)
                    {
                        cfm(i, j++) = A(row, col);

                        if (j == col_end)
                        {
                            j = 0;
                            i++;
                        }
                    }
                }
            }
        }

        template<typename DataType, class ExecutionSpace>
        __inline static DataType determinant(const Matrix<DataType, ExecutionSpace>& A)
        {
            const uint32 col_end = A.ncolumns() - 1;

            const uint32 m = A.nrows();
            const uint32 n = A.ncolumns();

            if (m == n)
            {
                ThrowException("determinant() can only only support a square Matrix.");
            }

            DataType D = 0.0;

            if (n == 1)
            {
                return A[0];
            }

            Matrix<DataType, ExecutionSpace> temp("temp", m, n);

            DataType sign = 1.0;

            // Iterate for each element of first row
            for (uint32 f = 0; f < n; f++)
            {
                // Getting Cofactor of A[0][f]
                cofactor(A, temp, 0, f);

                D += (sign * A(0, f) * determinant(temp, col_end));

                // terms are to be added with alternate sign
                sign = -sign;
            }

            return D;
        }

        template<typename DataType, class ExecutionSpace>
        __inline static void adjoint(const Matrix<DataType, ExecutionSpace>& A, Matrix<DataType, ExecutionSpace>& adj)
        {
            const uint32 col_end = A.ncolumns() - 1;

            const uint32 m = A.nrows();
            const uint32 n = A.ncolumns();

            if (n == 1)
            {
                adj(0, 0) = 1;

                return;
            }

            DataType sign = 1.0;

            Matrix<DataType, ExecutionSpace> temp("temp", m, n);

            for (uint32 i = 0; i < m; i++)
            {
                for (uint32 j = 0; j < n; j++)
                {
                    cofactor(A, temp, i, j);

                    sign = checkerboard_pattern<DataType, uint32>(i, j);

                    adj(i, j) = sign * determinant(temp, col_end);
                }
            }
        }

        template<typename DataType, class ExecutionSpace>
        KOKKOS_INLINE_FUNCTION static Vector<DataType, ExecutionSpace> upper_triangular_solve(const Matrix<DataType, ExecutionSpace>& A, const Vector<DataType, ExecutionSpace>& b)
        {
            const uint32 n = A.nrows() < A.ncolumns() ? A.nrows() : A.ncolumns();

            Vector<DataType, ExecutionSpace> x(new DataType[b], b);

            for (uint32 k = n; k >= 1; --k)
            {
                x(k) /= A(k, k);

                for (uint32 i = 1; i < k; ++i)
                {
                    x(i) -= x(k) * A(i, k);
                }
            }

            return x;
        }

        template<typename DataType, class ExecutionSpace>
        KOKKOS_INLINE_FUNCTION static Vector<DataType, ExecutionSpace> lower_triangular_solve(const Matrix<DataType, ExecutionSpace>& A, const Vector<DataType, ExecutionSpace>& b)
        {
            const uint32 n = A.nrows() < A.ncolumns() ? A.nrows() : A.ncolumns();

            Vector<DataType, ExecutionSpace> x(new DataType[b], b);

            for (uint32 k = 1; k <= n; k++)
            {
                x(k) /= A(k, k);

                for (uint32 i = k + 1; i <= n; ++i)
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

    using Extension::diagonal;
    using Extension::diagonal_matrix;
    using Extension::identity;
    using Extension::inverse;
    using Extension::norm;
    using Extension::pinverse;
    using Extension::transpose;
}

namespace Kokkos
{
    using Kokkos::Extension::diagonal;
    using Kokkos::Extension::diagonal_matrix;
    using Kokkos::Extension::identity;
    using Kokkos::Extension::inverse;
    using Kokkos::Extension::norm;
    using Kokkos::Extension::pinverse;
    using Kokkos::Extension::transpose;
}
