#pragma once

#if !defined(KOKKOS_EXTENSIONS)
#    error "Do not include directly. Include Extensions.hpp"
#endif

namespace Kokkos
{
    namespace Extension
    {

        template<typename VectorType, uint32 column>
        struct MaxColumnFunctor
        {
            typedef typename VectorType::traits::execution_space      ExecutionSpace;
            typedef typename VectorType::traits::data_type            DataType;
            typedef typename VectorType::traits::non_const_value_type ValueType;

            VectorType Values;

            MaxColumnFunctor(const VectorType& values) : Values(values) {}

            KOKKOS_INLINE_FUNCTION void operator()(const uint32& i, ValueType& value) const
            {
                Kokkos::atomic_fetch_max(&value, (ValueType)Values(i, column));
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
            const size_type                     n = lhs.extent(0);
            Assert(n == rhs.extent(0)) DataType sum;

#if defined(__CUDA_ARCH__)
            for (size_type i = 0; i < n; i++)
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

            mdrange_type<ExecutionSpace> policy(point_type<ExecutionSpace>{{0, 0}}, point_type<ExecutionSpace>{{n, m}});

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
        std::ostream& operator<<(std::ostream& s, const Extension::Matrix<DataType, ExecutionSpace>& A)
        {
            const size_type m = A.extent(0);
            const size_type n = A.extent(1);

            s << m << " " << n << "\n";
            for (size_type i = 0; i < m; i++)
            {
                for (size_type j = 0; j < n; j++)
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

            if (!(m == A.extent(0) && n == A.extent(1)))
            {
                Kokkos::resize(A, m, n);
            }

            for (size_type i = 0; i < m; i++)
                for (size_type j = 0; j < n; j++)
                {
                    s >> A(i, j);
                }

            return s;
        }

        namespace Internal
        {
#define MATRIX_OPS_FUNCTORS(OP_NAME, OP, ASSIGN_OP)                                                                                                                                \
    template<class RMatrix, class LhsMatrix, class RhsMatrix>                                                                                                                      \
    struct MatrixMatrix##OP_NAME##Functor                                                                                                                                          \
    {                                                                                                                                                                              \
        static_assert(RMatrix::Rank == 2, "RMatrix::Rank != 2");                                                                                                                   \
        static_assert(LhsMatrix::Rank == 2, "LhsMatrix::Rank != 2");                                                                                                               \
        static_assert(RhsMatrix::Rank == 2, "RhsMatrix::Rank != 2");                                                                                                               \
                                                                                                                                                                                   \
        typedef typename RMatrix::size_type    size_type;                                                                                                                          \
        typedef typename LhsMatrix::value_type value_type;                                                                                                                         \
                                                                                                                                                                                   \
        RMatrix                        _r;                                                                                                                                         \
        typename LhsMatrix::const_type _lhs;                                                                                                                                       \
        typename RhsMatrix::const_type _rhs;                                                                                                                                       \
                                                                                                                                                                                   \
        MatrixMatrix##OP_NAME##Functor(const RMatrix& r, const LhsMatrix& lhs, const RhsMatrix& rhs) : _r(r), _lhs(lhs), _rhs(rhs) {}                                              \
                                                                                                                                                                                   \
        KOKKOS_INLINE_FUNCTION void operator()(const size_type i) const                                                                                                            \
        {                                                                                                                                                                          \
            _r(i) = _lhs(i) OP _rhs(i);                                                                                                                                            \
        }                                                                                                                                                                          \
    };                                                                                                                                                                             \
                                                                                                                                                                                   \
    template<class RMatrix, class LhsMatrix>                                                                                                                                       \
    struct MatrixScalar##OP_NAME##Functor                                                                                                                                          \
    {                                                                                                                                                                              \
        static_assert(RMatrix::Rank == 2, "RMatrix::Rank != 2");                                                                                                                   \
        static_assert(LhsMatrix::Rank == 2, "LhsMatrix::Rank != 2");                                                                                                               \
                                                                                                                                                                                   \
        typedef typename RMatrix::size_type              size_type;                                                                                                                \
        typedef typename LhsMatrix::non_const_value_type value_type;                                                                                                               \
                                                                                                                                                                                   \
        RMatrix                        _r;                                                                                                                                         \
        typename LhsMatrix::const_type _lhs;                                                                                                                                       \
        value_type                     _rhs;                                                                                                                                       \
                                                                                                                                                                                   \
        MatrixScalar##OP_NAME##Functor(const RMatrix& r, const LhsMatrix& lhs, const value_type& rhs) : _r(r), _lhs(lhs), _rhs(rhs) {}                                             \
                                                                                                                                                                                   \
        KOKKOS_INLINE_FUNCTION void operator()(const size_type i) const                                                                                                            \
        {                                                                                                                                                                          \
            _r(i) = _lhs(i) OP _rhs;                                                                                                                                               \
        }                                                                                                                                                                          \
    };                                                                                                                                                                             \
                                                                                                                                                                                   \
    template<class RMatrix, class RhsMatrix>                                                                                                                                       \
    struct ScalarMatrix##OP_NAME##Functor                                                                                                                                          \
    {                                                                                                                                                                              \
        static_assert(RMatrix::Rank == 2, "RMatrix::Rank != 2");                                                                                                                   \
        static_assert(RhsMatrix::Rank == 2, "RhsMatrix::Rank != 2");                                                                                                               \
                                                                                                                                                                                   \
        typedef typename RMatrix::size_type              size_type;                                                                                                                \
        typedef typename RhsMatrix::non_const_value_type value_type;                                                                                                               \
                                                                                                                                                                                   \
        RMatrix                        _r;                                                                                                                                         \
        value_type                     _lhs;                                                                                                                                       \
        typename RhsMatrix::const_type _rhs;                                                                                                                                       \
                                                                                                                                                                                   \
        ScalarMatrix##OP_NAME##Functor(const RMatrix& r, const value_type& lhs, const RhsMatrix& rhs) : _r(r), _lhs(lhs), _rhs(rhs) {}                                             \
                                                                                                                                                                                   \
        KOKKOS_INLINE_FUNCTION void operator()(const size_type i) const                                                                                                            \
        {                                                                                                                                                                          \
            _r(i) = _lhs OP _rhs(i);                                                                                                                                               \
        }                                                                                                                                                                          \
    };                                                                                                                                                                             \
                                                                                                                                                                                   \
    template<class LhsMatrix, class RhsMatrix>                                                                                                                                     \
    struct MatrixMatrix##OP_NAME##AssignFunctor                                                                                                                                    \
    {                                                                                                                                                                              \
        static_assert(LhsMatrix::Rank == 2, "LhsMatrix::Rank != 2");                                                                                                               \
        static_assert(RhsMatrix::Rank == 2, "RhsMatrix::Rank != 2");                                                                                                               \
                                                                                                                                                                                   \
        typedef typename LhsMatrix::size_type  size_type;                                                                                                                          \
        typedef typename LhsMatrix::value_type value_type;                                                                                                                         \
                                                                                                                                                                                   \
        LhsMatrix                      _lhs;                                                                                                                                       \
        typename RhsMatrix::const_type _rhs;                                                                                                                                       \
                                                                                                                                                                                   \
        MatrixMatrix##OP_NAME##AssignFunctor(const LhsMatrix& lhs, const RhsMatrix& rhs) : _lhs(lhs), _rhs(rhs) {}                                                                 \
                                                                                                                                                                                   \
        KOKKOS_INLINE_FUNCTION void operator()(const size_type i) const                                                                                                            \
        {                                                                                                                                                                          \
            _lhs(i) ASSIGN_OP _rhs(i);                                                                                                                                             \
        }                                                                                                                                                                          \
    };                                                                                                                                                                             \
                                                                                                                                                                                   \
    template<class LhsMatrix>                                                                                                                                                      \
    struct MatrixScalar##OP_NAME##AssignFunctor                                                                                                                                    \
    {                                                                                                                                                                              \
        static_assert(LhsMatrix::Rank == 2, "LhsMatrix::Rank != 2");                                                                                                               \
                                                                                                                                                                                   \
        typedef typename LhsMatrix::size_type            size_type;                                                                                                                \
        typedef typename LhsMatrix::non_const_value_type value_type;                                                                                                               \
                                                                                                                                                                                   \
        LhsMatrix  _lhs;                                                                                                                                                           \
        value_type _rhs;                                                                                                                                                           \
                                                                                                                                                                                   \
        MatrixScalar##OP_NAME##AssignFunctor(const LhsMatrix& lhs, const value_type& rhs) : _lhs(lhs), _rhs(rhs) {}                                                                \
                                                                                                                                                                                   \
        KOKKOS_INLINE_FUNCTION void operator()(const size_type i) const                                                                                                            \
        {                                                                                                                                                                          \
            _lhs(i) ASSIGN_OP _rhs;                                                                                                                                                \
        }                                                                                                                                                                          \
    };                                                                                                                                                                             \
                                                                                                                                                                                   \
    template<typename DataType, class ExecutionSpace>                                                                                                                              \
    __inline static Matrix<DataType, ExecutionSpace> operator OP(const Matrix<DataType, ExecutionSpace>& lhs, const Matrix<DataType, ExecutionSpace>& rhs)                         \
    {                                                                                                                                                                              \
        const size_type m = lhs.extent(0);                                                                                                                                         \
        const size_type n = lhs.extent(1);                                                                                                                                         \
                                                                                                                                                                                   \
        assert(n == rhs.extent(0));                                                                                                                                                \
                                                                                                                                                                                   \
        Matrix<DataType, ExecutionSpace> r(lhs.label() + #OP + rhs.label(), n);                                                                                                    \
                                                                                                                                                                                   \
        Internal::MatrixMatrix##OP_NAME##Functor<Matrix<DataType, ExecutionSpace>, Matrix<DataType, ExecutionSpace>, Matrix<DataType, ExecutionSpace>> f(r, lhs, rhs);             \
                                                                                                                                                                                   \
        mdrange_type<ExecutionSpace> policy(point_type<ExecutionSpace>{{0, 0}}, point_type<ExecutionSpace>{{m, n}});                                                               \
                                                                                                                                                                                   \
        Kokkos::parallel_for("V_" #OP_NAME, policy, f);                                                                                                                            \
                                                                                                                                                                                   \
        return r;                                                                                                                                                                  \
    }                                                                                                                                                                              \
                                                                                                                                                                                   \
    template<typename DataType, class ExecutionSpace>                                                                                                                              \
    __inline static Matrix<DataType, ExecutionSpace> operator OP(const DataType& lhs, const Matrix<DataType, ExecutionSpace>& rhs)                                                 \
    {                                                                                                                                                                              \
        const size_type m = rhs.extent(0);                                                                                                                                         \
        const size_type n = rhs.extent(1);                                                                                                                                         \
                                                                                                                                                                                   \
        assert(n == rhs.extent(0));                                                                                                                                                \
                                                                                                                                                                                   \
        Matrix<DataType, ExecutionSpace> r(lhs.label() + #OP + rhs.label(), n);                                                                                                    \
                                                                                                                                                                                   \
        Internal::ScalarMatrix##OP_NAME##Functor<Matrix<DataType, ExecutionSpace>, Matrix<DataType, ExecutionSpace>> f(r, lhs, rhs);                                               \
                                                                                                                                                                                   \
        mdrange_type<ExecutionSpace> policy(point_type<ExecutionSpace>{{0, 0}}, point_type<ExecutionSpace>{{m, n}});                                                               \
                                                                                                                                                                                   \
        Kokkos::parallel_for("V_" #OP_NAME, policy, f);                                                                                                                            \
                                                                                                                                                                                   \
        return r;                                                                                                                                                                  \
    }                                                                                                                                                                              \
                                                                                                                                                                                   \
    template<typename DataType, class ExecutionSpace>                                                                                                                              \
    __inline static Matrix<DataType, ExecutionSpace> operator OP(const Matrix<DataType, ExecutionSpace>& lhs, const DataType& rhs)                                                 \
    {                                                                                                                                                                              \
        const size_type m = lhs.extent(0);                                                                                                                                         \
        const size_type n = lhs.extent(1);                                                                                                                                         \
                                                                                                                                                                                   \
        assert(n == rhs.extent(0));                                                                                                                                                \
                                                                                                                                                                                   \
        Matrix<DataType, ExecutionSpace> r(lhs.label() + #OP + rhs.label(), n);                                                                                                    \
                                                                                                                                                                                   \
        Internal::MatrixScalar##OP_NAME##Functor<Matrix<DataType, ExecutionSpace>, Matrix<DataType, ExecutionSpace>> f(r, lhs, rhs);                                               \
                                                                                                                                                                                   \
        mdrange_type<ExecutionSpace> policy(point_type<ExecutionSpace>{{0, 0}}, point_type<ExecutionSpace>{{m, n}});                                                               \
                                                                                                                                                                                   \
        Kokkos::parallel_for("V_" #OP_NAME, policy, f);                                                                                                                            \
                                                                                                                                                                                   \
        return r;                                                                                                                                                                  \
    }                                                                                                                                                                              \
                                                                                                                                                                                   \
    template<typename DataType, class ExecutionSpace>                                                                                                                              \
    __inline static Matrix<DataType, ExecutionSpace> operator ASSIGN_OP(Matrix<DataType, ExecutionSpace>& lhs, const Matrix<DataType, ExecutionSpace>& rhs)                        \
    {                                                                                                                                                                              \
        const size_type m = lhs.extent(0);                                                                                                                                         \
        const size_type n = lhs.extent(1);                                                                                                                                         \
                                                                                                                                                                                   \
        assert(n == rhs.extent(0));                                                                                                                                                \
                                                                                                                                                                                   \
        Matrix<DataType, ExecutionSpace> r(lhs.label() + #ASSIGN_OP + rhs.label(), n);                                                                                             \
                                                                                                                                                                                   \
        Internal::MatrixMatrix##OP_NAME##AssignFunctor<Matrix<DataType, ExecutionSpace>, Matrix<DataType, ExecutionSpace>> f(lhs, rhs);                                            \
                                                                                                                                                                                   \
        mdrange_type<ExecutionSpace> policy(point_type<ExecutionSpace>{{0, 0}}, point_type<ExecutionSpace>{{m, n}});                                                               \
                                                                                                                                                                                   \
        Kokkos::parallel_for("V_" #OP_NAME "Assign", policy, f);                                                                                                                   \
                                                                                                                                                                                   \
        return r;                                                                                                                                                                  \
    }                                                                                                                                                                              \
                                                                                                                                                                                   \
    template<typename DataType, class ExecutionSpace>                                                                                                                              \
    __inline static Matrix<DataType, ExecutionSpace> operator ASSIGN_OP(Matrix<DataType, ExecutionSpace>& lhs, const DataType& rhs)                                                \
    {                                                                                                                                                                              \
        const size_type m = rhs.extent(0);                                                                                                                                         \
        const size_type n = rhs.extent(1);                                                                                                                                         \
                                                                                                                                                                                   \
        assert(n == rhs.extent(0));                                                                                                                                                \
                                                                                                                                                                                   \
        Matrix<DataType, ExecutionSpace> r(lhs.label() + #ASSIGN_OP + std::to_string(rhs), n);                                                                                     \
                                                                                                                                                                                   \
        Internal::MatrixScalar##OP_NAME##AssignFunctor<Matrix<DataType, ExecutionSpace>> f(lhs, rhs);                                                                              \
                                                                                                                                                                                   \
        mdrange_type<ExecutionSpace> policy(point_type<ExecutionSpace>{{0, 0}}, point_type<ExecutionSpace>{{m, n}});                                                               \
                                                                                                                                                                                   \
        Kokkos::parallel_for("V_" #OP_NAME "Scalar", policy, f);                                                                                                                   \
                                                                                                                                                                                   \
        return r;                                                                                                                                                                  \
    }

            MATRIX_OPS_FUNCTORS(Plus, +, +=)
            MATRIX_OPS_FUNCTORS(Minus, -, -=)

#undef MATRIX_OPS_FUNCTORS

#define MATRIX_OPS_FUNCTORS(OP_NAME, OP, ASSIGN_OP)                                                                                                                                \
    template<class RMatrix, class LhsMatrix>                                                                                                                                       \
    struct MatrixScalar##OP_NAME##Functor                                                                                                                                          \
    {                                                                                                                                                                              \
        static_assert(RMatrix::Rank == 2, "RMatrix::Rank != 2");                                                                                                                   \
        static_assert(LhsMatrix::Rank == 2, "LhsMatrix::Rank != 2");                                                                                                               \
                                                                                                                                                                                   \
        typedef typename RMatrix::size_type              size_type;                                                                                                                \
        typedef typename LhsMatrix::non_const_value_type value_type;                                                                                                               \
                                                                                                                                                                                   \
        RMatrix                        _r;                                                                                                                                         \
        typename LhsMatrix::const_type _lhs;                                                                                                                                       \
        value_type                     _rhs;                                                                                                                                       \
                                                                                                                                                                                   \
        MatrixScalar##OP_NAME##Functor(const RMatrix& r, const LhsMatrix& lhs, const value_type& rhs) : _r(r), _lhs(lhs), _rhs(rhs) {}                                             \
                                                                                                                                                                                   \
        KOKKOS_INLINE_FUNCTION void operator()(const size_type i) const                                                                                                            \
        {                                                                                                                                                                          \
            _r(i) = _lhs(i) OP _rhs;                                                                                                                                               \
        }                                                                                                                                                                          \
    };                                                                                                                                                                             \
                                                                                                                                                                                   \
    template<class RMatrix, class RhsMatrix>                                                                                                                                       \
    struct ScalarMatrix##OP_NAME##Functor                                                                                                                                          \
    {                                                                                                                                                                              \
        static_assert(RMatrix::Rank == 2, "RMatrix::Rank != 2");                                                                                                                   \
        static_assert(RhsMatrix::Rank == 2, "RhsMatrix::Rank != 2");                                                                                                               \
                                                                                                                                                                                   \
        typedef typename RMatrix::size_type              size_type;                                                                                                                \
        typedef typename RhsMatrix::non_const_value_type value_type;                                                                                                               \
                                                                                                                                                                                   \
        RMatrix                        _r;                                                                                                                                         \
        value_type                     _lhs;                                                                                                                                       \
        typename RhsMatrix::const_type _rhs;                                                                                                                                       \
                                                                                                                                                                                   \
        ScalarMatrix##OP_NAME##Functor(const RMatrix& r, const value_type& lhs, const RhsMatrix& rhs) : _r(r), _lhs(lhs), _rhs(rhs) {}                                             \
                                                                                                                                                                                   \
        KOKKOS_INLINE_FUNCTION void operator()(const size_type i) const                                                                                                            \
        {                                                                                                                                                                          \
            _r(i) = _lhs OP _rhs(i);                                                                                                                                               \
        }                                                                                                                                                                          \
    };                                                                                                                                                                             \
                                                                                                                                                                                   \
    template<class LhsMatrix>                                                                                                                                                      \
    struct MatrixScalar##OP_NAME##AssignFunctor                                                                                                                                    \
    {                                                                                                                                                                              \
        static_assert(LhsMatrix::Rank == 2, "LhsMatrix::Rank != 2");                                                                                                               \
                                                                                                                                                                                   \
        typedef typename LhsMatrix::size_type            size_type;                                                                                                                \
        typedef typename LhsMatrix::non_const_value_type value_type;                                                                                                               \
                                                                                                                                                                                   \
        LhsMatrix  _lhs;                                                                                                                                                           \
        value_type _rhs;                                                                                                                                                           \
                                                                                                                                                                                   \
        MatrixScalar##OP_NAME##AssignFunctor(const LhsMatrix& lhs, const value_type& rhs) : _lhs(lhs), _rhs(rhs) {}                                                                \
                                                                                                                                                                                   \
        KOKKOS_INLINE_FUNCTION void operator()(const size_type i) const                                                                                                            \
        {                                                                                                                                                                          \
            _lhs(i) ASSIGN_OP _rhs;                                                                                                                                                \
        }                                                                                                                                                                          \
    };                                                                                                                                                                             \
                                                                                                                                                                                   \
    template<typename DataType, class ExecutionSpace>                                                                                                                              \
    __inline static Matrix<DataType, ExecutionSpace> operator OP(const DataType& lhs, const Matrix<DataType, ExecutionSpace>& rhs)                                                 \
    {                                                                                                                                                                              \
        const size_type m = rhs.extent(0);                                                                                                                                         \
        const size_type n = rhs.extent(1);                                                                                                                                         \
                                                                                                                                                                                   \
        assert(n == rhs.extent(0));                                                                                                                                                \
                                                                                                                                                                                   \
        Matrix<DataType, ExecutionSpace> r(lhs.label() + #OP + rhs.label(), n);                                                                                                    \
                                                                                                                                                                                   \
        Internal::ScalarMatrix##OP_NAME##Functor<Matrix<DataType, ExecutionSpace>, Matrix<DataType, ExecutionSpace>> f(r, lhs, rhs);                                               \
                                                                                                                                                                                   \
        mdrange_type<ExecutionSpace> policy(point_type<ExecutionSpace>{{0, 0}}, point_type<ExecutionSpace>{{m, n}});                                                               \
                                                                                                                                                                                   \
        Kokkos::parallel_for("V_" #OP_NAME, policy, f);                                                                                                                            \
                                                                                                                                                                                   \
        return r;                                                                                                                                                                  \
    }                                                                                                                                                                              \
                                                                                                                                                                                   \
    template<typename DataType, class ExecutionSpace>                                                                                                                              \
    __inline static Matrix<DataType, ExecutionSpace> operator OP(const Matrix<DataType, ExecutionSpace>& lhs, const DataType& rhs)                                                 \
    {                                                                                                                                                                              \
        const size_type m = lhs.extent(0);                                                                                                                                         \
        const size_type n = lhs.extent(1);                                                                                                                                         \
                                                                                                                                                                                   \
        assert(n == rhs.extent(0));                                                                                                                                                \
                                                                                                                                                                                   \
        Matrix<DataType, ExecutionSpace> r(lhs.label() + #OP + rhs.label(), n);                                                                                                    \
                                                                                                                                                                                   \
        Internal::MatrixScalar##OP_NAME##Functor<Matrix<DataType, ExecutionSpace>, Matrix<DataType, ExecutionSpace>> f(r, lhs, rhs);                                               \
                                                                                                                                                                                   \
        mdrange_type<ExecutionSpace> policy(point_type<ExecutionSpace>{{0, 0}}, point_type<ExecutionSpace>{{m, n}});                                                               \
                                                                                                                                                                                   \
        Kokkos::parallel_for("V_" #OP_NAME, policy, f);                                                                                                                            \
                                                                                                                                                                                   \
        return r;                                                                                                                                                                  \
    }                                                                                                                                                                              \
                                                                                                                                                                                   \
    template<typename DataType, class ExecutionSpace>                                                                                                                              \
    __inline static Matrix<DataType, ExecutionSpace> operator ASSIGN_OP(Matrix<DataType, ExecutionSpace>& lhs, const DataType& rhs)                                                \
    {                                                                                                                                                                              \
        const size_type m = rhs.extent(0);                                                                                                                                         \
        const size_type n = rhs.extent(1);                                                                                                                                         \
                                                                                                                                                                                   \
        assert(n == rhs.extent(0));                                                                                                                                                \
                                                                                                                                                                                   \
        Matrix<DataType, ExecutionSpace> r(lhs.label() + #ASSIGN_OP + std::to_string(rhs), n);                                                                                     \
                                                                                                                                                                                   \
        Internal::MatrixScalar##OP_NAME##AssignFunctor<Matrix<DataType, ExecutionSpace>> f(lhs, rhs);                                                                              \
                                                                                                                                                                                   \
        mdrange_type<ExecutionSpace> policy(point_type<ExecutionSpace>{{0, 0}}, point_type<ExecutionSpace>{{m, n}});                                                               \
                                                                                                                                                                                   \
        Kokkos::parallel_for("V_" #OP_NAME "Scalar", policy, f);                                                                                                                   \
                                                                                                                                                                                   \
        return r;                                                                                                                                                                  \
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

                    for (size_type j = 0; j < n; ++j)
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

                    for (size_type j = 0; j < n; ++j)
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

                KOKKOS_INLINE_FUNCTION void operator()(const size_type i, const size_type j) const
                {
                    _r(j, j) = _lhs(i, j);
                }
            };

            template<class LhsMatrix>
            struct MatrixNormFunctor
            {
                static_assert(LhsMatrix::Rank == 2, "LhsVector::Rank != 2");

                typedef typename LhsMatrix::size_type            size_type;
                typedef typename LhsMatrix::non_const_value_type value_type;

                typename LhsMatrix::const_type _lhs;

                MatrixNormFunctor(const LhsMatrix& lhs) : _lhs(lhs) {}

                KOKKOS_INLINE_FUNCTION void operator()(const size_type i, const size_type j, value_type& sum) const
                {
                    sum += abs(_lhs(i, j)) * abs(_lhs(i, j));
                }
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

            mdrange_type<ExecutionSpace> policy(point_type<ExecutionSpace>{{0, 0}}, point_type<ExecutionSpace>{{m, k}});

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

            mdrange_type<ExecutionSpace> policy(point_type<ExecutionSpace>{{0, 0}}, point_type<ExecutionSpace>{{m, n}});

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

            mdrange_type<ExecutionSpace> policy(point_type<ExecutionSpace>{{0, 0}}, point_type<ExecutionSpace>{{m, n}});

            Kokkos::parallel_for("M_Transpose", policy, f);

            return r;
        }

        template<typename DataType, class ExecutionSpace>
        KOKKOS_INLINE_FUNCTION static Vector<DataType, ExecutionSpace> upper_triangular_solve(const Matrix<DataType, ExecutionSpace>& A, const Vector<DataType, ExecutionSpace>& b)
        {
            const int n = A.extent(0) < A.extent(1) ? A.extent(0) : A.extent(1);

            Vector<DataType, ExecutionSpace> x(new DataType[b], b);

            for (int k = n; k >= 1; --k)
            {
                x(k) /= A(k, k);

                for (int i = 1; i < k; i++)
                {
                    x(i) -= x(k) * A(i, k);
                }
            }

            return x;
        }

        template<typename DataType, class ExecutionSpace>
        KOKKOS_INLINE_FUNCTION static Vector<DataType, ExecutionSpace> lower_triangular_solve(const Matrix<DataType, ExecutionSpace>& A, const Vector<DataType, ExecutionSpace>& b)
        {
            const int n = A.extent(0) < A.extent(1) ? A.extent(0) : A.extent(1);

            Vector<DataType, ExecutionSpace> x(new DataType[b], b);

            for (int k = 1; k <= n; k++)
            {
                x(k) /= A(k, k);

                for (int i = k + 1; i <= n; i++)
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
