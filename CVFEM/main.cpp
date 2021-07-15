
#include <Types.hpp>
#include <Constants.hpp>

#include <runtime.Kokkos/Extensions.hpp>

#include <KokkosBlas.hpp>
#include <KokkosSparse_CrsMatrix.hpp>
#include <KokkosSparse_spmv.hpp>

//#include <Sacado.hpp>

// using LocalMatrixType = KokkosSparse::CrsMatrix<double, LO, PHX::Device, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
// typename Kokkos::View<LocalMatrixType**, PHX::Device>::HostMirror hostJacTpetraBlocks("panzer::ScatterResidual_BlockTpetra<Jacobian>::hostJacTpetraBlocks", numFieldBlocks, numFieldBlocks);

// LocalMatrixType unmanagedMatrix(managedMatrix.values.label(), managedMatrix.numCols(), unmanagedValues, unmanagedGraph);
//
// new (&hostJacTpetraBlocks(row,col)) LocalMatrixType(unmanagedMatrix);
//
// for (int row=0; row < numFieldBlocks; ++row)
//{
//    for (int col=0; col < numFieldBlocks; ++col)
//    {
//      if (hostBlockExistsInJac(row,col))
//      {
//        hostJacTpetraBlocks(row,col).~CrsMatrix();
//      }
//    }
//  }

// typename Kokkos::View<LocalMatrixType**,PHX::Device> jacTpetraBlocks("panzer::ScatterResidual_BlockedTpetra<Jacobian>::jacTpetraBlocks",numFieldBlocks,numFieldBlocks);
// Kokkos::deep_copy(jacTpetraBlocks,hostJacTpetraBlocks);

template<typename DataType,
         class ExecutionSpace,
         typename Ordinal    = size_type,
         typename Offset     = size_type,
         typename DeviceType = Kokkos::Device<ExecutionSpace, typename ExecutionSpace::memory_space>>
using SparseMatrixType = KokkosSparse::CrsMatrix<DataType, Ordinal, DeviceType, void, Offset>;

using ExecutionSpace = Kokkos::Cuda;
// using ExecutionSpace = Kokkos::OpenMP;
// using ExecutionSpace = Kokkos::Serial;

static const int32 FadStride = std::is_same_v<ExecutionSpace, Kokkos::Cuda> ? 32 : 1;

// typedef Sacado::Fad::DFad<double> FadType;
// using FadType = Sacado::Fad::SFad<double, num_deriv>;

// using FadType = Sacado::Fad::DFad<double>;

// using ViewFadType = Sacado::Fad::ViewFad<double, num_deriv, FadStride, FadType>;
// typedef Sacado::Fad::SFad<FadType, 1>                               HessianType;
// typedef HessianType                                                 ScalarT;

// typedef Kokkos::LayoutContiguous<ExecutionSpace::array_layout, FadStride> ContLayout;
// template<size_type num_deriv>
// using ViewType = Kokkos::View<FadType*, ExecutionSpace>;

using CrsMatrixType = SparseMatrixType<double, ExecutionSpace>;

using graph_type      = typename CrsMatrixType::staticcrsgraph_type;
using row_map_type    = typename graph_type::row_map_type;
using column_idx_type = typename graph_type::entries_type;
using values_type     = typename CrsMatrixType::values_type;

template<typename DataType>
static DataType func(const DataType& c, const DataType& x)
{
    DataType r = c + std::sin(x);
    return r;
}

///// <summary>
///// ------------------------------------------------------------------------
///// IN-PLACE coo-csr conversion routine.
///// ------------------------------------------------------------------------
///// this subroutine converts a matrix stored in coordinate format into
///// the csr format. The conversion is done in place in that the arrays
///// a,ja,ia of the result are overwritten onto the original arrays.
///// ------------------------------------------------------------------------
///// on entry:
///// ---------
///// n        = int. row dimension of A.
///// nnz        = int. number of nonzero elements in A.
///// job   = int. Job indicator. when job=1, the real values in a are
///// filled. Otherwise a is not touched and the structure of the
///// array only (i.e. ja, ia)  is obtained.
///// a        = real array of size nnz (number of nonzero elements in A)
///// containing the nonzero elements
///// ja        = int array of length nnz containing the column positions
///// of the corresponding elements in a.
///// ia        = int array of length max(nnz,n+1) containing the row
///// positions of the corresponding elements in a.
///// iwk        = int work array of length n+1
///// on return:
///// ----------
///// a
///// ja
///// ia        = contains the compressed sparse row data structure for the
///// resulting matrix.
///// Note:
///// -------
///// the entries of the output matrix are not sorted (the column
///// indices in each are not in increasing order) use coocsr
///// if you want them sorted. Note also that ia has to have at
///// least n+1 locations
///// </summary>
//int coicsr(int* n, int* nnz, double* a, int* ja, int* ia, int* iwk)
//{
//    /* Local variables */
//    int i__, k;
//
//    /* Parameter adjustments */
//    --iwk;
//    --ja;
//    --a;
//    --ia;
//
//    /* Function Body */
//    double t     = (float)0.;
//    double tnext = (float)0.;
//
//    /* find pointer array for resulting matrix. */
//    int i_1 = *n + 1;
//    for (i__ = 1; i__ <= i_1; ++i__)
//    {
//        iwk[i__] = 0;
//        /* L35: */
//    }
//    i_1 = *nnz;
//    for (k = 1; k <= i_1; ++k)
//    {
//        i__ = ia[k];
//        ++iwk[i__ + 1];
//        /* L4: */
//    }
//    /* ------------------------------------------------------------------------ */
//    iwk[1] = 1;
//    i_1   = *n;
//    for (i__ = 2; i__ <= i_1; ++i__)
//    {
//        iwk[i__] = iwk[i__ - 1] + iwk[i__];
//        /* L44: */
//    }
//
//    /*     loop for a cycle in chasing process. */
//
//    int init = 1;
//    k        = 0;
//L5:
//    t = a[init];
//
//    i__      = ia[init];
//    int j    = ja[init];
//    ia[init] = -1;
///* ------------------------------------------------------------------------ */
//L6:
//    ++k;
//    /*     current row number is i.  determine  where to go. */
//    int ipos = iwk[i__];
//    /*     save the chased element. */
//
//    tnext = a[ipos];
//
//    int inext = ia[ipos];
//    int jnext = ja[ipos];
//    /*     then occupy its location. */
//
//    a[ipos] = t;
//
//    ja[ipos] = j;
//    /*     update pointer information for next element to come in row i. */
//    iwk[i__] = ipos + 1;
//    /*     determine  next element to be chased, */
//    if (ia[ipos] < 0)
//    {
//        goto L65;
//    }
//    t        = tnext;
//    i__      = inext;
//    j        = jnext;
//    ia[ipos] = -1;
//    if (k < *nnz)
//    {
//        goto L6;
//    }
//    goto L70;
//L65:
//    ++init;
//    if (init > *nnz)
//    {
//        goto L70;
//    }
//    if (ia[init] < 0)
//    {
//        goto L65;
//    }
//    /*     restart chasing -- */
//    goto L5;
//L70:
//    i_1 = *n;
//    for (i__ = 1; i__ <= i_1; ++i__)
//    {
//        ia[i__ + 1] = iwk[i__];
//        /* L80: */
//    }
//    ia[1] = 1;
//    return 0;
//
//}

int main(int argc, char* argv[])
{
    using namespace Kokkos::Extension;

    const int32 num_threads      = 16;
    const int32 num_numa         = 1;
    const int32 device_id        = 0;
    const int32 ndevices         = 3;
    const int32 skip_device      = 9999;
    const bool  disable_warnings = true;

    Kokkos::InitArguments arguments{};
    arguments.num_threads      = num_threads;
    arguments.num_numa         = num_numa;
    arguments.device_id        = device_id;
    arguments.ndevices         = ndevices;
    arguments.skip_device      = skip_device;
    arguments.disable_warnings = disable_warnings;

    Kokkos::ScopeGuard kokkos(arguments);
    {
        const int32 num_args       = 1;
        const int32 num_time_steps = 5;
        // const int32 num_deriv      = 1;

        Kokkos::View<double*, ExecutionSpace> args("args", num_args);
        args(0) = 1.0;

        Kokkos::View<double*, ExecutionSpace> x("x", num_time_steps);
        Kokkos::View<double*, ExecutionSpace> y("y", num_time_steps);

        for (size_type i = 0; i < num_time_steps; i++)
        {
            x(i) = (((double)i + 1) / (double)num_time_steps) * Constants<double>::PI();
            y(i) = func(args(0), x(i));
        }

        const int32 numNNZ = 1 + (2 * (num_time_steps - 1));

        using ViewValuesType = CrsMatrixType::values_type::non_const_type;
        // using DataType       = ViewValuesType::traits::non_const_value_type;

        ViewValuesType             input_layer("input_layer", num_time_steps);
        ViewValuesType::HostMirror input_layer_h = Kokkos::create_mirror_view(input_layer);

        ViewValuesType             middle_layer("middle_layer", num_time_steps);
        ViewValuesType::HostMirror middle_layer_h = Kokkos::create_mirror_view(middle_layer);

        ViewValuesType             output_layer("output_layer", num_time_steps);
        ViewValuesType::HostMirror output_layer_h = Kokkos::create_mirror_view(output_layer);

        for (size_type i = 0; i < input_layer_h.size(); i++)
        {
            input_layer_h(i) = x(i);

            middle_layer_h(i) = Kokkos::ArithTraits<double>::zero();

            output_layer_h(i) = Kokkos::ArithTraits<double>::zero();
        }

        Kokkos::deep_copy(input_layer, input_layer_h);
        Kokkos::deep_copy(middle_layer, middle_layer_h);
        Kokkos::deep_copy(output_layer, output_layer_h);

        row_map_type::non_const_type row_map("row pointers", num_time_steps + 1);
        row_map_type::HostMirror     row_map_h = Kokkos::create_mirror_view(row_map);

        column_idx_type::non_const_type column_idx("column indices", numNNZ);
        column_idx_type::HostMirror     column_idx_h = Kokkos::create_mirror_view(column_idx);

        values_type::non_const_type values("values", numNNZ);
        values_type::HostMirror     values_h = Kokkos::create_mirror_view(values);

        size_type col_idx = 0;
        for (size_type i = 0; i < numNNZ; i++)
        {
            values_h(i) = ((double)std::rand() / (double)RAND_MAX);

            col_idx += ((i - 1) % 2);

            column_idx_h(i) = col_idx - 1;
        }

        row_map_h(0) = 0;
        for (size_type i = 1; i < num_time_steps; i++)
        {
            row_map_h(i) = (2 * i) - 1;
        }
        row_map_h(num_time_steps) = numNNZ;

        Kokkos::deep_copy(values, values_h);
        Kokkos::deep_copy(column_idx, column_idx_h);
        Kokkos::deep_copy(row_map, row_map_h);

         //std::cout << row_map << std::endl;
         //std::cout << column_idx << std::endl;
         //std::cout << values_h << std::endl;

        graph_type    connection_graph(column_idx, row_map);
        CrsMatrixType input_connections("connections", num_time_steps, values, connection_graph);

        const double alpha = Kokkos::ArithTraits<double>::one();
        const double beta  = Kokkos::ArithTraits<double>::zero();

        KokkosSparse::spmv("N", alpha, input_connections, input_layer, beta, middle_layer);

        std::cout << input_layer << std::endl;
        std::cout << input_connections << std::endl;
        std::cout << middle_layer << std::endl;









































    }

    std::cout << "Press any key to exit." << std::endl;
    getchar();

    return 0;
}
