#pragma once

#include "runtime.Kokkos/ViewTypes.hpp"
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
    namespace Extension
    {
        template<typename DataType, class ExecutionSpace>
        using Vector = View<DataType*, typename ExecutionSpace::array_layout, ExecutionSpace>;

        template<typename DataType, class ExecutionSpace>
        using Matrix = View<DataType**, typename ExecutionSpace::array_layout, ExecutionSpace>;

        namespace Internal
        {
            template<typename DataType, class ExecutionSpace, int Rank>
            struct IndexOfMinimum;

            template<typename DataType, class ExecutionSpace>
            struct IndexOfMinimum<DataType, ExecutionSpace, 1>
            {
                using ViewType  = View<DataType, typename ExecutionSpace::array_layout, ExecutionSpace>;
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
                using ViewType  = View<DataType, typename ExecutionSpace::array_layout, ExecutionSpace>;
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
        KOKKOS_INLINE_FUNCTION static auto min(const VectorType& values) -> typename std::enable_if<VectorType::Rank == 1, typename VectorType::traits::non_const_value_type>::type
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
        KOKKOS_INLINE_FUNCTION static auto max(const VectorType& values) -> typename std::enable_if<VectorType::Rank == 1, typename VectorType::traits::non_const_value_type>::type
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
        size_type N;

        s >> N;

        if(!(N == A.size()))
        {
            Kokkos::resize(A, N);
        }

        for(size_type i = 0; i < N; i++)
            s >> A(i);

        return s;
    }

    namespace Extension
    {
        template<typename DataType, class ExecutionSpace>
        KOKKOS_INLINE_FUNCTION static Vector<DataType, ExecutionSpace> operator+(const Vector<DataType, ExecutionSpace>& lhs, const Vector<DataType, ExecutionSpace>& rhs)
        {
            const size_type n = lhs.extent(0);

            assert(N == rhs.extent(0));

            Vector<DataType, ExecutionSpace> tmp(new DataType[n], n);

            for(size_type i = 0; i < n; i++)
                tmp(i) = lhs(i) + rhs(i);

            return tmp;
        }

        template<typename DataType, class ExecutionSpace>
        KOKKOS_INLINE_FUNCTION static Vector<DataType, ExecutionSpace> operator+=(Vector<DataType, ExecutionSpace>& lhs, const Vector<DataType, ExecutionSpace>& rhs)
        {
            const size_type n = lhs.extent(0);

            assert(N == rhs.extent(0));

            for(size_type i = 0; i < n; i++)
                lhs(i) += rhs(i);

            return lhs;
        }

        template<typename DataType, class ExecutionSpace>
        KOKKOS_INLINE_FUNCTION static Vector<DataType, ExecutionSpace> operator+=(Vector<DataType, ExecutionSpace>& lhs, const DataType& rhs)
        {
            const size_type n = lhs.extent(0);

            assert(N == rhs.extent(0));

            for(size_type i = 0; i < n; i++)
                lhs(i) += rhs(i);

            return lhs;
        }

        template<typename DataType, class ExecutionSpace>
        KOKKOS_INLINE_FUNCTION static Vector<DataType, ExecutionSpace> operator-(const Vector<DataType, ExecutionSpace>& lhs, const Vector<DataType, ExecutionSpace>& rhs)
        {
            const size_type n = lhs.extent(0);

            assert(N == rhs.extent(0));

            Vector<DataType, ExecutionSpace> tmp(new DataType[n], n);

            for(size_type i = 0; i < n; i++)
                tmp(i) = lhs(i) - rhs(i);

            return tmp;
        }

        template<typename DataType, class ExecutionSpace>
        KOKKOS_INLINE_FUNCTION static Vector<DataType, ExecutionSpace> operator-=(Vector<DataType, ExecutionSpace>& lhs, const Vector<DataType, ExecutionSpace>& rhs)
        {
            const size_type n = lhs.extent(0);

            assert(N == rhs.extent(0));

            for(size_type i = 0; i < n; i++)
                lhs(i) -= rhs(i);

            return lhs;
        }

        template<typename DataType, class ExecutionSpace>
        KOKKOS_INLINE_FUNCTION static Vector<DataType, ExecutionSpace> operator-=(Vector<DataType, ExecutionSpace>& lhs, const DataType& rhs)
        {
            const size_type n = lhs.extent(0);

            assert(N == rhs.extent(0));

            for(size_type i = 0; i < n; i++)
                lhs(i) -= rhs(i);

            return lhs;
        }

        template<typename DataType, class ExecutionSpace>
        KOKKOS_INLINE_FUNCTION static Vector<DataType, ExecutionSpace> elementwise_mult(const Vector<DataType, ExecutionSpace>& lhs, const Vector<DataType, ExecutionSpace>& rhs)
        {
            const size_type n = lhs.extent(0);

            assert(N == rhs.extent(0));

            Vector<DataType, ExecutionSpace> tmp(new DataType[n], n);

            for(size_type i = 0; i < n; i++)
                tmp(i) = lhs(i) * rhs(i);

            return tmp;
        }

        template<typename DataType, class ExecutionSpace>
        KOKKOS_INLINE_FUNCTION static Vector<DataType, ExecutionSpace> operator*=(Vector<DataType, ExecutionSpace>& lhs, const DataType& rhs)
        {
            const size_type n = lhs.extent(0);

            assert(N == rhs.extent(0));

            for(size_type i = 0; i < n; i++)
                lhs(i) *= rhs(i);

            return lhs;
        }

        template<typename DataType, class ExecutionSpace>
        KOKKOS_INLINE_FUNCTION static DataType norm(const Vector<DataType, ExecutionSpace>& lhs)
        {
            const size_type n = lhs.extent(0);

            DataType sum = 0.0;

            for(int i = 0; i < n; i++)
                sum += abs(lhs(i)) * abs(lhs(i));

            return sqrt(sum);
        }

        template<typename DataType, class ExecutionSpace>
        KOKKOS_INLINE_FUNCTION static DataType dot_prod(const Vector<DataType, ExecutionSpace>& lhs, const Vector<DataType, ExecutionSpace>& rhs)
        {
            const size_type n = lhs.extent(0);
            assert(N == rhs.extent(0));

            DataType sum = 0;

            for(size_type i = 0; i < n; i++)
                sum += lhs(i) * rhs(i);

            return sum;
        }

        template<typename DataType, class ExecutionSpace>
        KOKKOS_INLINE_FUNCTION static DataType dot_product(const Vector<DataType, ExecutionSpace>& lhs, const Vector<DataType, ExecutionSpace>& rhs)
        {
            return dot_prod(lhs, rhs);
        }

        template<typename DataType, class ExecutionSpace>
        KOKKOS_INLINE_FUNCTION static DataType operator*(const Vector<DataType, ExecutionSpace>& lhs, const Vector<DataType, ExecutionSpace>& rhs)
        {
            return dot_prod(lhs, rhs);
        }

        template<typename DataType, class ExecutionSpace>
        KOKKOS_INLINE_FUNCTION static Vector<DataType, ExecutionSpace> operator*(const DataType& lhs, const Vector<DataType, ExecutionSpace>& rhs)
        {
            const size_type                  n = rhs.extent(0);
            Vector<DataType, ExecutionSpace> r(new DataType[n], n);

            for(int i = 0; i < n; i++)
                r(i) = lhs * rhs(i);

            return r;
        }

        template<typename DataType, class ExecutionSpace>
        KOKKOS_INLINE_FUNCTION static Vector<DataType, ExecutionSpace> operator*(const Vector<DataType, ExecutionSpace>& lhs, const DataType& rhs)
        {
            return lhs * rhs;
        }

        template<typename DataType, class ExecutionSpace>
        KOKKOS_INLINE_FUNCTION static DataType operator/(const Vector<DataType, ExecutionSpace>& lhs, const Vector<DataType, ExecutionSpace>& rhs)
        {
            const size_type n = lhs.extent(0);

            assert(N == rhs.extent(0));

            Vector<DataType, ExecutionSpace> tmp(new DataType[n], n);

            for(size_type i = 0; i < n; i++)
                tmp(i) = lhs(i) / rhs(i);

            return tmp;
        }

        template<typename DataType, class ExecutionSpace>
        KOKKOS_INLINE_FUNCTION static Vector<DataType, ExecutionSpace> operator/=(Vector<DataType, ExecutionSpace>& lhs, const DataType& rhs)
        {
            const size_type n = lhs.extent(0);

            assert(N == rhs.extent(0));

            for(size_type i = 0; i < n; i++)
                lhs(i) /= rhs(i);

            return lhs;
        }

        template<typename DataType, class ExecutionSpace>
        KOKKOS_INLINE_FUNCTION static Vector<DataType, ExecutionSpace> operator/(const DataType& lhs, const Vector<DataType, ExecutionSpace>& rhs)
        {
            const size_type                  n = rhs.extent(0);
            Vector<DataType, ExecutionSpace> r(new DataType[n], n);

            for(int i = 0; i < n; i++)
                r(i) = lhs / rhs(i);

            return r;
        }

        template<typename DataType, class ExecutionSpace>
        KOKKOS_INLINE_FUNCTION static Vector<DataType, ExecutionSpace> operator/(const Vector<DataType, ExecutionSpace>& lhs, const DataType& rhs)
        {
            return lhs / rhs;
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

    // using Extension::operator==;
    // using Extension::operator!=;
    // using Extension::operator<;
    // using Extension::operator<=;
    // using Extension::operator>;
    // using Extension::operator>=;

    // using Kokkos::Extension::operator/;
    // using Kokkos::Extension::operator%;

    // using Kokkos::Extension::operator+=;
    // using Kokkos::Extension::operator-=;
    // using Kokkos::Extension::operator*=;
    // using Kokkos::Extension::operator/=;
}

namespace Kokkos
{
    template<typename DataType, class ExecutionSpace>
    std::ostream& operator<<(std::ostream& s, const Extension::Matrix<DataType, ExecutionSpace>& A)
    {
        size_type M = A.extent(0);
        size_type N = A.extent(1);

        s << M << " " << N << "\n";
        for(size_type i = 0; i < M; i++)
        {
            for(size_type j = 0; j < N; j++)
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
        size_type M, N;

        s >> M >> N;

        if(!(M == A.extent(0) && N == A.extent(1)))
        {
            Kokkos::resize(A, M, N);
        }

        for(size_type i = 0; i < M; i++)
            for(size_type j = 0; j < N; j++)
            {
                s >> A(i, j);
            }

        return s;
    }

    namespace Extension
    {
        template<typename DataType, class ExecutionSpace>
        KOKKOS_INLINE_FUNCTION static Matrix<DataType, ExecutionSpace>& mult(Matrix<DataType, ExecutionSpace>& C, const Matrix<DataType, ExecutionSpace>& A, const Matrix<DataType, ExecutionSpace>& B)
        {
#ifdef TNT_BOUNDS_CHECK
            assert(A.extent(1) == B.extent(0));
#endif

            const size_type M = A.extent(0);
            const size_type N = A.extent(1);
            const size_type K = B.extent(1);

#ifdef TNT_BOUNDS_CHECK
            assert(C.extent(0) == M);
            assert(C.extent(1) == K);
#endif

            DataType sum;

            for(size_type i = 0; i < M; i++)
                for(size_type k = 0; k < K; k++)
                {
                    const DataType* row_i = &(A(i, 0));
                    const DataType* col_k = &(B(0, k));
                    sum                   = 0;
                    for(size_type j = 0; j < N; j++)
                    {
                        sum += *row_i * *col_k;
                        ++row_i;
                        col_k += K;
                    }
                    C(i, k) = sum;
                }

            return C;
        }

        template<typename DataType, class ExecutionSpace>
        KOKKOS_INLINE_FUNCTION static Matrix<DataType, ExecutionSpace> mult(const Matrix<DataType, ExecutionSpace>& A, const Matrix<DataType, ExecutionSpace>& B)
        {
#ifdef TNT_BOUNDS_CHECK
            assert(A.extent(1) == B.extent(0));
#endif

            const size_type M = A.extent(0);
            const size_type N = A.extent(1);
            const size_type K = B.extent(1);

            Matrix<DataType, ExecutionSpace> tmp(new DataType[M * K], M, K);

            // for (size_type i=0; i<M; i++)
            // {
            // for (size_type k=0; k<K; k++)
            //  	{
            // 	    DataType sum = 0;
            //  	    for (size_type j=0; j<N; j++)
            // 	        sum = sum +  A(i,j) * B(j,k);
            //
            //         tmp(i,k) = sum;
            //   }
            // }

            mult(tmp, A, B); // tmp = A*B

            return tmp;
        }

        template<typename DataType, class ExecutionSpace>
        KOKKOS_INLINE_FUNCTION static Matrix<DataType, ExecutionSpace> operator*(const Matrix<DataType, ExecutionSpace>& A, const Matrix<DataType, ExecutionSpace>& B)
        {
            return mult(A, B);
        }

        template<typename DataType, class ExecutionSpace>
        KOKKOS_INLINE_FUNCTION static Vector<DataType, ExecutionSpace> mult(const Matrix<DataType, ExecutionSpace>& A, const Vector<DataType, ExecutionSpace>& b)
        {
#ifdef TNT_BOUNDS_CHECK
            assert(A.extent(1) == b.extent(0));
#endif

            const size_type M = A.extent(0);
            const size_type N = A.extent(1);

            Vector<DataType, ExecutionSpace> tmp(new DataType[M], M);
            DataType                         sum;

            for(size_type i = 0; i < M; i++)
            {
                sum = 0;
                for(size_type j = 0; j < N; j++)
                    sum = sum + A(i, j) * b(j);

                tmp(i) = sum;
            }

            return tmp;
        }

        template<typename DataType, class ExecutionSpace>
        KOKKOS_INLINE_FUNCTION static Vector<DataType, ExecutionSpace> operator*(const Matrix<DataType, ExecutionSpace>& A, const Vector<DataType, ExecutionSpace>& b)
        {
            return mult(A, b);
        }

        template<typename DataType, class ExecutionSpace>
        KOKKOS_INLINE_FUNCTION static Matrix<DataType, ExecutionSpace> mult(const DataType& s, const Matrix<DataType, ExecutionSpace>& A)
        {
            const size_type M = A.extent(0);
            const size_type N = A.extent(1);

            Matrix<DataType, ExecutionSpace> R(new DataType[M * N], M, N);
            for(int i = 0; i < M; i++)
                for(int j = 0; j < N; j++)
                    R(i, j) = s * A(i, j);

            return R;
        }

        template<typename DataType, class ExecutionSpace>
        KOKKOS_INLINE_FUNCTION static Matrix<DataType, ExecutionSpace> mult(const Matrix<DataType, ExecutionSpace>& A, const DataType& s)
        {
            return mult(s, A);
        }

        template<typename DataType, class ExecutionSpace>
        KOKKOS_INLINE_FUNCTION static Matrix<DataType, ExecutionSpace> mult_eq(const DataType& s, Matrix<DataType, ExecutionSpace>& A)
        {
            const size_type M = A.extent(0);
            const size_type N = A.extent(1);

            for(int i = 0; i < M; i++)
                for(int j = 0; j < N; j++)
                    A(i, j) *= s;

            return A;
        }

        template<typename DataType, class ExecutionSpace>
        KOKKOS_INLINE_FUNCTION static Matrix<DataType, ExecutionSpace> mult_eq(Matrix<DataType, ExecutionSpace>& A, const DataType& a)
        {
            return mult_eq(a, A);
        }

        template<typename DataType, class ExecutionSpace>
        KOKKOS_INLINE_FUNCTION static Matrix<DataType, ExecutionSpace> transpose_mult(const Matrix<DataType, ExecutionSpace>& A, const Matrix<DataType, ExecutionSpace>& B)
        {
#ifdef TNT_BOUNDS_CHECK
            assert(A.extent(0) == B.extent(0));
#endif

            const size_type M = A.extent(1);
            const size_type N = A.extent(0);
            const size_type K = B.extent(1);

            Matrix<DataType, ExecutionSpace> tmp(new DataType[M * K], M, K);
            DataType                         sum;

            for(size_type i = 0; i < N; i++)
                for(size_type k = 0; k < K; k++)
                {
                    sum = 0;
                    for(size_type j = 0; j < M; j++)
                        sum = sum + A(j, i) * B(j, k);

                    tmp(i, k) = sum;
                }

            return tmp;
        }

        template<typename DataType, class ExecutionSpace>
        KOKKOS_INLINE_FUNCTION static Vector<DataType, ExecutionSpace> transpose_mult(const Matrix<DataType, ExecutionSpace>& A, const Vector<DataType, ExecutionSpace>& b)
        {
#ifdef TNT_BOUNDS_CHECK
            assert(A.extent(0) == b.extent(0));
#endif

            const size_type M = A.extent(1);
            const size_type N = A.extent(0);

            Vector<DataType, ExecutionSpace> tmp(new DataType[M], M);

            for(size_type i = 0; i < M; i++)
            {
                DataType sum = 0;
                for(size_type j = 0; j < N; j++)
                    sum = sum + A(j, i) * b(j);

                tmp(i) = sum;
            }

            return tmp;
        }

        template<typename DataType, class ExecutionSpace>
        KOKKOS_INLINE_FUNCTION static Matrix<DataType, ExecutionSpace> add(const Matrix<DataType, ExecutionSpace>& A, const Matrix<DataType, ExecutionSpace>& B)
        {
            const size_type M = A.extent(0);
            const size_type N = A.extent(1);

            assert(M == B.extent(0));
            assert(N == B.extent(1));

            Matrix<DataType, ExecutionSpace> tmp(new DataType[M * N], M, N);

            for(size_type i = 0; i < M; i++)
                for(size_type j = 0; j < N; j++)
                    tmp(i, j) = A(i, j) + B(i, j);

            return tmp;
        }

        template<typename DataType, class ExecutionSpace>
        KOKKOS_INLINE_FUNCTION static Matrix<DataType, ExecutionSpace> operator+(const Matrix<DataType, ExecutionSpace>& A, const Matrix<DataType, ExecutionSpace>& B)
        {
            return add(A, B);
        }

        template<typename DataType, class ExecutionSpace>
        KOKKOS_INLINE_FUNCTION static Matrix<DataType, ExecutionSpace>& add_eq(Matrix<DataType, ExecutionSpace>& A, const Matrix<DataType, ExecutionSpace>& B)
        {
            const size_type M = A.extent(0);
            const size_type N = A.extent(1);

            assert(M == B.extent(0));
            assert(N == B.extent(1));

            Matrix<DataType, ExecutionSpace> tmp(new DataType[M * N], M, N);

            for(size_type i = 0; i < M; i++)
                for(size_type j = 0; j < N; j++)
                    tmp(i, j) = A(i, j) + B(i, j);

            return A += tmp;
        }

        template<typename DataType, class ExecutionSpace>
        KOKKOS_INLINE_FUNCTION static Matrix<DataType, ExecutionSpace> operator+=(Matrix<DataType, ExecutionSpace>& A, const Matrix<DataType, ExecutionSpace>& B)
        {
            return add_eq(A, B);
        }

        template<typename DataType, class ExecutionSpace>
        KOKKOS_INLINE_FUNCTION static Matrix<DataType, ExecutionSpace> minus(const Matrix<DataType, ExecutionSpace>& A, const Matrix<DataType, ExecutionSpace>& B)
        {
            const size_type M = A.extent(0);
            const size_type N = A.extent(1);

            assert(M == B.extent(0));
            assert(N == B.extent(1));

            Matrix<DataType, ExecutionSpace> tmp(new DataType[M * N], M, N);

            for(size_type i = 0; i < M; i++)
                for(size_type j = 0; j < N; j++)
                    tmp(i, j) = A(i, j) - B(i, j);

            return tmp;
        }

        template<typename DataType, class ExecutionSpace>
        KOKKOS_INLINE_FUNCTION static Matrix<DataType, ExecutionSpace> operator-(const Matrix<DataType, ExecutionSpace>& A, const Matrix<DataType, ExecutionSpace>& B)
        {
            return minus(A, B);
        }

        template<typename DataType, class ExecutionSpace>
        KOKKOS_INLINE_FUNCTION static Matrix<DataType, ExecutionSpace> mult_element(const Matrix<DataType, ExecutionSpace>& A, const Matrix<DataType, ExecutionSpace>& B)
        {
            const size_type M = A.extent(0);
            const size_type N = A.extent(1);

            assert(M == B.extent(0));
            assert(N == B.extent(1));

            Matrix<DataType, ExecutionSpace> tmp(new DataType[M * N], M, N);

            for(size_type i = 0; i < M; i++)
                for(size_type j = 0; j < N; j++)
                    tmp(i, j) = A(i, j) * B(i, j);

            return tmp;
        }

        template<typename DataType, class ExecutionSpace>
        KOKKOS_INLINE_FUNCTION static Matrix<DataType, ExecutionSpace>& mult_element_eq(Matrix<DataType, ExecutionSpace>& A, const Matrix<DataType, ExecutionSpace>& B)
        {
            const size_type M = A.extent(0);
            const size_type N = A.extent(1);

            assert(M == B.extent(0));
            assert(N == B.extent(1));

            for(size_type i = 0; i < M; i++)
                for(size_type j = 0; j < N; j++)
                    A(i, j) *= B(i, j);

            return A;
        }

        template<typename DataType, class ExecutionSpace>
        KOKKOS_INLINE_FUNCTION static DataType norm(const Matrix<DataType, ExecutionSpace>& A)
        {
            const size_type M = A.extent(0);
            const size_type N = A.extent(1);

            DataType sum = 0.0;
            for(int i = 1; i <= M; i++)
                for(int j = 1; j <= N; j++)
                    sum += A(i, j) * A(i, j);
            return sqrt(sum);
        }

        template<typename DataType, class ExecutionSpace>
        KOKKOS_INLINE_FUNCTION static Matrix<DataType, ExecutionSpace> transpose(const Matrix<DataType, ExecutionSpace>& A)
        {
            const size_type M = A.extent(0);
            const size_type N = A.extent(1);

            Matrix<DataType, ExecutionSpace> S(new DataType[M * N], N, M);

            for(size_type i = 0; i < M; i++)
                for(size_type j = 0; j < N; j++)
                    S(j, i) = A(i, j);

            return S;
        }

        template<typename DataType, class ExecutionSpace>
        KOKKOS_INLINE_FUNCTION static Vector<DataType, ExecutionSpace> upper_triangular_solve(const Matrix<DataType, ExecutionSpace>& A, const Vector<DataType, ExecutionSpace>& b)
        {
            const int                        n = A.extent(0) < A.extent(1) ? A.extent(0) : A.extent(1);
            Vector<DataType, ExecutionSpace> x(new DataType[b], b);
            for(int k = n; k >= 1; --k)
            {
                x(k) /= A(k, k);
                for(int i = 1; i < k; i++)
                    x(i) -= x(k) * A(i, k);
            }

            return x;
        }

        template<typename DataType, class ExecutionSpace>
        KOKKOS_INLINE_FUNCTION static Vector<DataType, ExecutionSpace> lower_triangular_solve(const Matrix<DataType, ExecutionSpace>& A, const Vector<DataType, ExecutionSpace>& b)
        {
            const int                        n = A.extent(0) < A.extent(1) ? A.extent(0) : A.extent(1);
            Vector<DataType, ExecutionSpace> x(new DataType[b], b);
            for(int k = 1; k <= n; k++)
            {
                x(k) /= A(k, k);
                for(int i = k + 1; i <= n; i++)
                    x(i) -= x(k) * A(i, k);
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
}

namespace Kokkos
{
    namespace Extension
    {
        // template<class T>
        // inline T dot_product(const Sparse_Vector<T>& s, const Vector<T>& x)
        // {
        //     return s.dot_product(x);
        // }
        //
        // template<class T>
        // inline T dot_product(const Vector<T>& x, const Sparse_Vector<T>& s)
        // {
        //     return s.dot_product(x);
        // }
        //
        // template<class T>
        // inline T operator*(const Vector<T>& x, const Sparse_Vector<T>& s)
        // {
        //     return dot_product(x, s);
        // }
        //
        // template<class T>
        // inline T operator*(const Sparse_Vector<T>& s, const Vector<T>& x)
        // {
        //     return dot_product(x, s);
        // }
        //
        // template<class T>
        // inline double norm(const Sparse_Vector<T>& s)
        // {
        //     return s.norm();
        // }
    }
}

namespace Kokkos
{
    namespace Extension
    {
        // template<class T>
        // inline Vector<T> operator*(const Sparse_Matrix<T>& S, const Vector<T>& x)
        // {
        //     return S.mult(x);
        // }
        //
        // template<class T>
        // inline double norm(const Sparse_Matrix<T>& S)
        // {
        //     return S.norm();
        // }
    }
}
