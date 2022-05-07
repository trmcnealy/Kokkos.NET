#pragma once

#if !defined(KOKKOS_EXTENSIONS)
#    error "Do not include directly. Include Extensions.hpp"
#endif

#include <KokkosSparse_CrsMatrix.hpp>


//proc COO_to_CSC
//input: N, n_cols (integer scalars)
//input: values, rows, columns (COO arrays)
//allocate array col_offsets with length n_cols + 1
//forall j ∈ [0, n_cols]: col_offsets[j] ← 0
//forall i ∈ [0, N):
//j ← columns[i] + 1
//col_offsets[j] ← col_offsets[j] + 1
//forall j ∈ [1, n_cols]:
//col_offsets[j] ← col_offsets[j] + col_offsets[j-1]
//output: values, rows, col_offsets (CSC arrays)
//
//proc CSC_to_COO
//input: N, n_cols (integer scalars)
//input: values, rows, col_offsets (CSC arrays)
//allocate array columns with length N
//k ← 0
//forall j ∈ [0, n_cols):
//M ← col_offsets[j+1] − col_offsets[j]
//forall l ∈ [0, M):
//columns[k+l] ← j
//k ← k + M
//output: values, rows, columns (COO arrays)
//
//proc CSC_to_RBT
//input: N, n_rows, n_cols (integer scalars)
//input: values, rows, col_offsets (CSC arrays)
//declare red-black tree T
//forall j ∈ [0, n_cols):
//start ← col_offsets[j]
//end ← col_offsets[j+1]
//forall k ∈ [start,end):
//index ← row_indices[k] + j ∗ n_rows
//l ← (index, values[k])
//insert node l into T
//output: T (red-black tree)
//
//proc RBT_to_CSC
//input: N, n_rows, n_cols (integer scalars)
//input: T (red-black tree)
//allocate array values with length N
//allocate array row_indices with length N
//allocate array col_offsets with length n_cols + 1
//forall j ∈ [0, n_cols]: col_offsets[j] ← 0
//k ← 0
//foreach node l ∈ T, where l = (index,value):
//values[k] ← value
//row_indices[k] ← index mod n_rows
//j ← bindex/n_rowsc
//col_offsets[j+1] ← col_offsets[j+1] + 1
//k ← k + 1
//forall j ∈ [1, n_cols]:
//col_offsets[j] ← col_offsets[j] + col_offsets[j-1]
//output: values, rows, col_offsets (CSC arrays)









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
