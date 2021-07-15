#pragma once

#if !defined(KOKKOS_EXTENSIONS)
#    error "Do not include directly. Include Extensions.hpp"
#endif

#include <KokkosSparse_CrsMatrix.hpp>

namespace KokkosSparse
{
    namespace Extension
    {

        template<typename DataType,
                 class DeviceType,
                 typename Ordinal    = size_type,
                 typename Offset     = size_type/*,
                 typename DeviceType = Kokkos::Device<ExecutionSpace, typename ExecutionSpace::memory_space>*/>
        static std::ostream& operator<<(std::ostream& s, const CrsMatrix<DataType, Ordinal, DeviceType, void, Offset>& A)
        {
            const size_type m = A.numRows();
            const size_type n = A.numCols();

            s << std::endl;
            s << " [" << m << "x" << n << "]";
            s << std::endl;

            for (size_type i = 0; i < m; ++i)
            {
                const auto& rowEntries = A.row(i);

                for (size_type j = 0; j < rowEntries.length; ++j)
                {
                    s << rowEntries.value(j) << " ";
                }
                s << std::endl;
            }

            return s;
        }

    }

    using Extension::operator<<;
    // using Kokkos::Extension::operator>>;
}

namespace Kokkos
{
    namespace Extension
    {

        // namespace SparseMatrixOperators
        //{

        //    template<class RVector, class LhsMatrix, class RhsVector>
        //    struct SparseMatrixVectorMultiplyFunctor
        //    {
        //        static_assert(RVector::Rank == 1, "RVector::Rank != 1");
        //        static_assert(LhsMatrix::Rank == 2, "LhsMatrix::Rank != 2");
        //        static_assert(RhsVector::Rank == 1, "RhsVector::Rank != 1");

        //        typedef typename LhsMatrix::size_type            size_type;
        //        typedef typename LhsMatrix::const_value_type     const_value_type;
        //        typedef typename LhsMatrix::non_const_value_type value_type;

        //        RVector                        _r;
        //        typename LhsMatrix::const_type _lhs;
        //        typename RhsVector::const_type _rhs;
        //        const_value_type               n;

        //        SparseMatrixVectorMultiplyFunctor(const RVector& r, const LhsMatrix& lhs, const RhsVector& rhs) : _r(r), _lhs(lhs), _rhs(rhs), n(rhs.nrows()) {}

        //        KOKKOS_INLINE_FUNCTION void operator()(const size_type j) const
        //        {
        //            for (int i = _lhs.ColumnPtr()[j]; i < _lhs.ColumnPtr()[j + 1]; ++i)
        //            {
        //                _r[_lhs.RowIndex()[i]] += _lhs.Values()[i] * _rhs[j];
        //            }
        //        }
        //    };

        //}

        // template<typename DataType, typename IndexType, class ExecutionSpace>
        //__inline static Vector<DataType, ExecutionSpace> operator*(const SparseMatrix<DataType, IndexType, ExecutionSpace>& lhs, const Vector<DataType, ExecutionSpace>& rhs)
        //{
        //    const size_type m = lhs.nrows();
        //    const size_type n = lhs.ncolumns();

        //    Assert(n == rhs.extent(0));

        //    Vector<DataType, ExecutionSpace> r(lhs.label() + " * " + rhs.label(), m);

        //    SparseMatrixOperators::SparseMatrixVectorMultiplyFunctor<Vector<DataType, ExecutionSpace>, SparseMatrix<DataType, IndexType, ExecutionSpace>, Vector<DataType, ExecutionSpace>> f(r,
        //                                                                                                                                                                                      lhs,
        //                                                                                                                                                                                      rhs);

        //    Kokkos::RangePolicy<ExecutionSpace> policy(0, n);

        //    Kokkos::parallel_for("SM_Multiply", policy, f);

        //    return r;
        //}

        // template<typename DataType, typename IndexType, class ExecutionSpace>
        //__inline static SparseMatrix<DataType, IndexType, ExecutionSpace> transpose(const SparseMatrix<DataType, IndexType, ExecutionSpace>& lhs)
        //{
        //    const size_type m = lhs.nrows();
        //    const size_type n = lhs.ncolumns();
        //    const size_type v = lhs.nvalues();

        //    SparseMatrix<DataType, IndexType, ExecutionSpace> r("t(" + lhs.label() + ")", n, m, v);

        //    // MatrixOperators::MatrixTransposeFunctor<Matrix<DataType, ExecutionSpace>, Matrix<DataType, ExecutionSpace>> f(r, lhs);

        //    // mdrange_type<ExecutionSpace> policy(point_type<ExecutionSpace>{{0, 0}}, point_type<ExecutionSpace>{{m, n}});

        //    // Kokkos::parallel_for("M_Transpose", policy, f);

        //    //VecInt count(m, 0);
        //    //for (i = 0; i < n; i++)
        //    //{
        //    //    for (j = col_ptr[i]; j < col_ptr[i + 1]; j++)
        //    //    {
        //    //        k = row_ind[j];
        //    //        count[k]++;
        //    //    }
        //    //}
        //    //for (j = 0; j < m; j++)
        //    //{
        //    //    at.col_ptr[j + 1] = at.col_ptr[j] + count[j];
        //    //}
        //    //for (j = 0; j < m; j++)
        //    //{
        //    //    count[j] = 0;
        //    //}
        //    //for (i = 0; i < n; i++)
        //    //{
        //    //    for (j = col_ptr[i]; j < col_ptr[i + 1]; j++)
        //    //    {
        //    //        k                 = row_ind[j];
        //    //        index             = at.col_ptr[k] + count[k];
        //    //        at.row_ind[index] = i;
        //    //        at.val[index]     = val[j];
        //    //        count[k]++;
        //    //    }
        //    //}

        //    return r;
        //}
    }
}
