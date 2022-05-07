#pragma once

#include <runtime.Kokkos/ViewTypes.hpp>

#include <MathExtensions.hpp>
#include <StdExtensions.hpp>
#include <Print.hpp>
#include <Exceptions.h>
#include <Constants.hpp>
#include <Concepts.hpp>
#include <Complex.hpp>

#include <Kokkos_Core.hpp>
#include <Kokkos_ScatterView.hpp>
#include <Kokkos_UnorderedMap.hpp>

#include <KokkosBlas.hpp>

#include <type_traits>

#include <cuda.h>
#include <mutex>

/// KokkosBlas::abs(y,x):		            y[i] = |x[i]|
/// KokkosBlas::axpy(alpha,x,y):		    y[i] += alpha * x[i]
/// KokkosBlas::axpby(alpha,x,beta,y):		y[i] = beta * y + alpha * x[i]
/// KokkosBlas::dot(x,y):		            dot = SUM_i ( x[i] * y[i] )
/// KokkosBlas::fill(x,alpha):		        x[i] = alpha
/// KokkosBlas::iamax(x):		            i= min{i of MAX_i(x[i]) }
/// KokkosBlas::mult(gamma,y,alpha,A,x):    y[i] = gamma * y[i] + alpha * A[i] * x[i]
/// KokkosBlas::nrm1(x):		            nrm1 = SUM_i( |x[i]| )
/// KokkosBlas::nrm2(x):		            nrm2 = sqrt ( SUM_i( |x[i]| * |x[i]| ))
/// KokkosBlas::nrm2_squared:               nrm2 = SUM_i( |x[i]| * |x[i]| )
/// KokkosBlas::nrm2w(x,w):		            nrm2w = sqrt ( SUM_i( (|x[i]|/|w[i]|)^2 ))
/// KokkosBlas::nrm2w_squared:              nrm2 = SUM_i( |x[i]| / |w[i]| * |x[i]| / |w[i]| )
/// KokkosBlas::nrminf(x):		            nrminf = MAX_i( |x[i]| )
/// KokkosBlas::reciprocal(r,x):		    r[i] = 1 / x[i]
/// KokkosBlas::scal(y,alpha,x):		    y[i] = alpha * x[i]
/// KokkosBlas::sum(x):		                sum = SUM_i( x[i] )
/// KokkosBlas::update(a,x,b,y,g,z):        y[i] = g * z[i] + b * y[i] + a * x[i]
///
/// KokkosBlas::gemv(t,alph,A,x,bet,y):     y[i] = bet*y[i] + alph*SUM_j(A[i,j]*x[j])
/// KokkosBlas::gemm(tA,tB,alph,A,B,bet,C): C[i,j]=bet*C[i,j]+alph*SUM_k(A[i,k]*B[k,j])
///
/// KokkosBlas::trmm(s,uplo,t,d,alpha,A,B): B = alpha*op(A)*B or alpha*B*op(A)
/// KokkosBlas::trsm(s,uplo,t,d,alpha,A,B): X = op(A)\alpha*B or alpha*B/op(A)

namespace Kokkos
{
    namespace Extension
    {
        namespace Internal
        {

            template<typename DataType, class ExecutionSpace, class Layout = typename ExecutionSpace::array_layout>
            __inline static void NestedHybridLoop(View<DataType**, Layout, ExecutionSpace> view)
            {
                const size_type nCpu = view.extent(0);
                const size_type nGpu = view.extent(1);

                typedef Kokkos::RangePolicy<Kokkos::OpenMP, Kokkos::IndexType<size_type>, Kokkos::Schedule<Kokkos::Static>> OpenMPPolicy;
                typedef Kokkos::RangePolicy<Kokkos::Cuda, Kokkos::IndexType<size_type>, Kokkos::Schedule<Kokkos::Static>>   CudaPolicy;

                Kokkos::parallel_for(OpenMPPolicy(0, nCpu),
                                     [=](const size_type i)
                                     {
                                         cudaStream_t stream;
                                         cudaStreamCreate(&stream);
                                         const Kokkos::Cuda space0(stream);

                                         Kokkos::parallel_for(CudaPolicy(space0, 0, nGpu), [=] __host__ __device__(const size_type i0) { view(i, i0) = 1.0 * i0; });

                                         space0.fence();
                                     });
                Kokkos::fence();
            }
        }

        template<class ExecutionSpace>
        class SpaceInstance;

        // template<class ExecutionSpace>
        // class SpaceInstance
        //{
        //    static SpaceInstance  _instance;
        //    static std::once_flag _flag;
        //
        //    ExecutionSpace execution_space;
        //
        //    __inline SpaceInstance() : execution_space() {}
        //    __inline ~SpaceInstance() = default;
        //
        // public:
        //    __inline static SpaceInstance& GetInstance()
        //    {
        //        std::call_once(_flag, []() { _instance = SpaceInstance{}; });
        //        return _instance;
        //    }
        //
        //    __inline static ExecutionSpace& GetExecutionSpace() { return GetInstance().execution_space; }
        //};

        template<>
        class SpaceInstance<Kokkos::Cuda>
        {
            Kokkos::Cuda execution_space;

            __inline SpaceInstance()
            {
                cudaStream_t stream;
                cudaStreamCreate(&stream);

                execution_space = Kokkos::Cuda(stream);
            }

            __inline ~SpaceInstance()
            {
                cudaStream_t stream = execution_space.cuda_stream();
                cudaStreamDestroy(stream);
            }

        public:
            __inline static SpaceInstance<Kokkos::Cuda>& GetInstance()
            {
                static SpaceInstance<Kokkos::Cuda> instance;
                return instance;
            }

            __inline static Kokkos::Cuda& GetExecutionSpace()
            {
                return GetInstance().execution_space;
            }
        };

        template<>
        class SpaceInstance<Kokkos::OpenMP>
        {
            Kokkos::OpenMP execution_space;

            __inline SpaceInstance() {}

            __inline ~SpaceInstance() {}

        public:
            __inline static SpaceInstance<Kokkos::OpenMP>& GetInstance()
            {
                static SpaceInstance<Kokkos::OpenMP> instance;
                return instance;
            }

            __inline static Kokkos::OpenMP& GetExecutionSpace()
            {
                return GetInstance().execution_space;
            }
        };

        template<>
        class SpaceInstance<Kokkos::Serial>
        {
            Kokkos::Serial execution_space;

            __inline SpaceInstance() {}

            __inline ~SpaceInstance() {}

        public:
            __inline static SpaceInstance<Kokkos::Serial>& GetInstance()
            {
                static SpaceInstance<Kokkos::Serial> instance;
                return instance;
            }

            __inline static Kokkos::Serial& GetExecutionSpace()
            {
                return GetInstance().execution_space;
            }
        };

    }
}

namespace Kokkos
{
    struct VOID_TYPE
    {
    };

    namespace Extension
    {
        template<typename DataTypes, class ExecutionSpace, class Layout = typename ExecutionSpace::array_layout>
        using ScratchView = View<DataTypes, Layout, typename ExecutionSpace::scratch_memory_space, Kokkos::MemoryUnmanaged>;

        template<typename DataType, class ExecutionSpace>
        using Scalar = View<DataType, ExecutionSpace>;

        template<typename DataType, class ExecutionSpace, class Layout = typename ExecutionSpace::array_layout>
        using Vector = View<DataType*, Layout, ExecutionSpace>;

        template<typename DataType, class ExecutionSpace, class Layout = typename ExecutionSpace::array_layout>
        using ConstVector = View<const DataType*, Layout, ExecutionSpace>;

        template<typename DataType, class ExecutionSpace, class Layout = typename ExecutionSpace::array_layout>
        class Matrix
        {
        public:
            using ThisType = Matrix<DataType, ExecutionSpace, Layout>;
            typedef Layout                                   LayoutType;
            typedef View<DataType**, Layout, ExecutionSpace> ViewType;
            typedef typename ViewType::traits                traits;

            typedef typename ViewType::size_type                    size_type;
            typedef typename ViewType::const_value_type             const_value_type;
            typedef typename ViewType::traits::non_const_value_type non_const_value_type;

            using const_type     = Matrix<const DataType, ExecutionSpace, Layout>;
            using non_const_type = Matrix<DataType, ExecutionSpace, Layout>;

            inline static constexpr size_type row_index    = std::is_same<LayoutType, Kokkos::LayoutLeft>::value ? 1 : 0;
            inline static constexpr size_type column_index = std::is_same<LayoutType, Kokkos::LayoutLeft>::value ? 0 : 1;
            inline static constexpr size_type Rank         = 2;

        private:
            ViewType _view;

            KOKKOS_FORCEINLINE_FUNCTION static auto clone(const Matrix& src);

        public:
            template<Integer NRows, Integer NColumns>
            __inline explicit Matrix(std::string label, NRows nRows, NColumns nColumns) :
                _view(label, std::is_same<LayoutType, Kokkos::LayoutLeft>::value ? nColumns : nRows, std::is_same<LayoutType, Kokkos::LayoutLeft>::value ? nRows : nColumns)
            {
            }

            template<Integer NRows, Integer NColumns>
            KOKKOS_INLINE_FUNCTION explicit Matrix(non_const_value_type* ptr, NRows nRows, NColumns nColumns) :
                _view(ptr, std::is_same<LayoutType, Kokkos::LayoutLeft>::value ? nColumns : nRows, std::is_same<LayoutType, Kokkos::LayoutLeft>::value ? nRows : nColumns)
            {
            }

            template<typename OtherDataType>
            __inline Matrix(const Vector<OtherDataType, ExecutionSpace>& vector) : Matrix(vector.label(), vector.extent(0), vector.extent(0))
            {
                for(size_type i0 = 0; i0 < _view.extent(0); ++i0)
                {
                    for(size_type i1 = 0; i1 < _view.extent(1); ++i1)
                    {
                        if(i0 == i1)
                        {
                            _view(i0, i1) = vector(i0);
                        }
                        else
                        {
                            _view(i0, i1) = 0.0;
                        }
                    }
                }
            }

            __inline Matrix()  = default;
            __inline ~Matrix() = default;
            KOKKOS_INLINE_FUNCTION Matrix(const Matrix& other) : _view(other.View()) {}
            __inline Matrix(Matrix&&)              = default;
            KOKKOS_INLINE_FUNCTION Matrix& operator=(const Matrix& other)
            {
                if(this == &other)
                {
                    return *this;
                }
                _view = other.View();
                return *this;
            }
            __inline Matrix& operator=(Matrix&&) = default;

            template<typename OtherDataType, class OtherExecutionSpace, class OtherLayout = typename OtherExecutionSpace::array_layout>
            KOKKOS_INLINE_FUNCTION Matrix(std::enable_if_t<!std::is_same<DataType, OtherDataType>::value || !std::is_same<ExecutionSpace, OtherExecutionSpace>::value || !std::is_same<Layout, OtherLayout>::value,
                                                           const Matrix<OtherDataType, OtherExecutionSpace, OtherLayout>&> other) :
                _view(other.View())
            {
            }

            template<typename OtherDataType, class OtherExecutionSpace, class OtherLayout = typename OtherExecutionSpace::array_layout>
            KOKKOS_INLINE_FUNCTION Matrix& operator=(std::enable_if_t<!std::is_same<DataType, OtherDataType>::value || !std::is_same<ExecutionSpace, OtherExecutionSpace>::value || !std::is_same<Layout, OtherLayout>::value,
                                                                      const Matrix<OtherDataType, OtherExecutionSpace, OtherLayout>&> other)
            {
                if(this == &other)
                {
                    return *this;
                }
                _view = other.View();
                return *this;
            }

#pragma region Indexers
            template<Integer TRow, Integer TColumn>
            KOKKOS_FORCEINLINE_FUNCTION auto operator()(const TRow& row, const TColumn& column) ->
                typename std::enable_if<Integer<TRow> && std::is_same<LayoutType, Kokkos::LayoutLeft>::value, typename ViewType::reference_type>::type
            {
                return _view(column, row);
            }

            template<Integer TRow, Integer TColumn>
            KOKKOS_FORCEINLINE_FUNCTION auto operator()(const TRow& row, const TColumn& column) const ->
                typename std::enable_if<Integer<TRow> && std::is_same<LayoutType, Kokkos::LayoutLeft>::value, const typename ViewType::reference_type>::type
            {
                return _view(column, row);
            }

            template<Integer TRow, Integer TColumn>
            KOKKOS_FORCEINLINE_FUNCTION auto operator()(const TRow& row, const TColumn& column) ->
                typename std::enable_if<Integer<TRow> && std::is_same<LayoutType, Kokkos::LayoutRight>::value, typename ViewType::reference_type>::type
            {
                return _view(row, column);
            }

            template<Integer TRow, Integer TColumn>
            KOKKOS_FORCEINLINE_FUNCTION auto operator()(const TRow& row, const TColumn& column) const ->
                typename std::enable_if<Integer<TRow> && std::is_same<LayoutType, Kokkos::LayoutRight>::value, const typename ViewType::reference_type>::type
            {
                return _view(row, column);
            }

            template<typename MatrixViewType, Integer TRow, Integer TColumn>
            KOKKOS_FORCEINLINE_FUNCTION static auto Indexer(MatrixViewType& view, const TRow& row, const TColumn& column) ->
                typename std::enable_if<MatrixViewType::Rank == 2 && Integer<TRow> && std::is_same<typename MatrixViewType::traits::array_layout, Kokkos::LayoutLeft>::value, typename ViewType::reference_type>::type
            {
                return view(column, row);
            }

            template<typename MatrixViewType, Integer TRow, Integer TColumn>
            KOKKOS_FORCEINLINE_FUNCTION static auto Indexer(MatrixViewType& view, const TRow& row, const TColumn& column) ->
                typename std::enable_if<MatrixViewType::Rank == 2 && Integer<TRow> && std::is_same<typename MatrixViewType::traits::array_layout, Kokkos::LayoutRight>::value, typename ViewType::reference_type>::type
            {
                return view(row, column);
            }

            template<typename MatrixViewType, Integer TRow, Integer TColumn>
            KOKKOS_FORCEINLINE_FUNCTION static auto Indexer(MatrixViewType& view, const TRow& row, const TColumn& column) ->
                typename std::enable_if<MatrixViewType::Rank == 2 && Integer<TRow> && std::is_same<typename MatrixViewType::traits::array_layout, Kokkos::LayoutStride>::value, typename ViewType::reference_type>::type
            {
                return view(row, column);
            }

            template<typename MatrixViewType, Integer TRow, Integer TColumn>
            KOKKOS_FORCEINLINE_FUNCTION static auto Indexer(MatrixViewType& view, const TRow& row, const TColumn& column) ->
                typename std::enable_if<MatrixViewType::Rank == 1 && Integer<TRow>, typename ViewType::reference_type>::type
            {
                return view(row);
            }

#pragma endregion

#pragma region Member Functions
            __inline operator ViewType()
            {
                return _view;
            }

            __inline operator const ViewType() const
            {
                return _view;
            }

            KOKKOS_FORCEINLINE_FUNCTION ViewType& View()
            {
                return _view;
            }

            KOKKOS_FORCEINLINE_FUNCTION const ViewType& View() const
            {
                return _view;
            }

            KOKKOS_INLINE_FUNCTION constexpr typename ViewType::pointer_type data() const
            {
                return _view.data();
            }

            __inline const std::string label() const
            {
                return _view.label();
            }

            KOKKOS_FORCEINLINE_FUNCTION constexpr const size_type nrows() const
            {
                return _view.extent(row_index);
            }

            KOKKOS_FORCEINLINE_FUNCTION constexpr const size_type ncolumns() const
            {
                return _view.extent(column_index);
            }

            KOKKOS_INLINE_FUNCTION constexpr const size_type size() const
            {
                return nrows() * ncolumns();
            }
#pragma endregion

#pragma region Row& Column operator() SubViews
            template<Integer TRowStart, Integer TRowEnd>
            KOKKOS_FORCEINLINE_FUNCTION auto operator()(const Kokkos::pair<TRowStart, TRowEnd>& row_range_idx) ->
                typename std::enable_if<std::is_same<LayoutType, Kokkos::LayoutLeft>::value, Kokkos::Subview<ViewType, Kokkos::pair<TRowStart, TRowEnd>, Kokkos::Impl::ALL_t>>::type
            {
                return Kokkos::subview(_view, row_range_idx, Kokkos::ALL);
            }

            template<Integer TRowStart, Integer TRowEnd>
            KOKKOS_FORCEINLINE_FUNCTION auto operator()(const Kokkos::pair<TRowStart, TRowEnd>& row_range_idx) ->
                typename std::enable_if<std::is_same<LayoutType, Kokkos::LayoutRight>::value, Kokkos::Subview<ViewType, Kokkos::Impl::ALL_t, Kokkos::pair<TRowStart, TRowEnd>>>::type
            {
                return Kokkos::subview(_view, Kokkos::ALL, row_range_idx);
            }

            template<Integer TRow, Integer TColumnStart, Integer TColumnEnd>
            KOKKOS_FORCEINLINE_FUNCTION auto operator()(const TRow& row_idx, const Kokkos::pair<TColumnStart, TColumnEnd>& column_range_idx) ->
                typename std::enable_if<std::is_same<LayoutType, Kokkos::LayoutLeft>::value, Kokkos::Subview<ViewType, TRow, Kokkos::pair<TColumnStart, TColumnEnd>>>::type
            {
                return Kokkos::subview(_view, row_idx, column_range_idx);
            }

            template<Integer TRow, Integer TColumnStart, Integer TColumnEnd>
            KOKKOS_FORCEINLINE_FUNCTION auto operator()(const TRow& row_idx, const Kokkos::pair<TColumnStart, TColumnEnd>& column_range_idx) ->
                typename std::enable_if<std::is_same<LayoutType, Kokkos::LayoutRight>::value, Kokkos::Subview<ViewType, Kokkos::pair<TColumnStart, TColumnEnd>, TRow>>::type
            {
                return Kokkos::subview(_view, column_range_idx, row_idx);
            }

            template<Integer TRowStart, Integer TRowEnd, Integer TColumn>
            KOKKOS_FORCEINLINE_FUNCTION auto operator()(const Kokkos::pair<TRowStart, TRowEnd>& row_range_idx, const TColumn& column_idx) ->
                typename std::enable_if<std::is_same<LayoutType, Kokkos::LayoutLeft>::value, Kokkos::Subview<ViewType, Kokkos::pair<TRowStart, TRowEnd>, TColumn>>::type
            {
                return Kokkos::subview(_view, row_range_idx, column_idx);
            }

            template<Integer TRowStart, Integer TRowEnd, Integer TColumn>
            KOKKOS_FORCEINLINE_FUNCTION auto operator()(const Kokkos::pair<TRowStart, TRowEnd>& row_range_idx, const TColumn& column_idx) ->
                typename std::enable_if<std::is_same<LayoutType, Kokkos::LayoutRight>::value, Kokkos::Subview<ViewType, TColumn, Kokkos::pair<TRowStart, TRowEnd>>>::type
            {
                return Kokkos::subview(_view, column_idx, row_range_idx);
            }

            template<Integer TRowStart, Integer TRowEnd, Integer TColumnStart, Integer TColumnEnd>
            KOKKOS_FORCEINLINE_FUNCTION auto operator()(const Kokkos::pair<TRowStart, TRowEnd>& row_range_idx, const Kokkos::pair<TColumnStart, TColumnEnd>& column_range_idx) ->
                typename std::enable_if<std::is_same<LayoutType, Kokkos::LayoutLeft>::value, Kokkos::Subview<ViewType, Kokkos::pair<TRowStart, TRowEnd>, Kokkos::pair<TColumnStart, TColumnEnd>>>::type
            {
                return Kokkos::subview(_view, row_range_idx, column_range_idx);
            }

            template<Integer TRowStart, Integer TRowEnd, Integer TColumnStart, Integer TColumnEnd>
            KOKKOS_FORCEINLINE_FUNCTION auto operator()(const Kokkos::pair<TRowStart, TRowEnd>& row_range_idx, const Kokkos::pair<TColumnStart, TColumnEnd>& column_range_idx) ->
                typename std::enable_if<std::is_same<LayoutType, Kokkos::LayoutRight>::value, Kokkos::Subview<ViewType, Kokkos::pair<TColumnStart, TColumnEnd>, Kokkos::pair<TRowStart, TRowEnd>>>::type
            {
                return Kokkos::subview(_view, column_range_idx, row_range_idx);
            }
#pragma endregion

#pragma region Row SubViews
            template<Integer TRow>
            KOKKOS_FORCEINLINE_FUNCTION auto Row(const TRow& row_idx) -> typename std::enable_if<std::is_same<LayoutType, Kokkos::LayoutLeft>::value, Kokkos::Subview<ViewType, Kokkos::Impl::ALL_t, TRow>>::type
            {
                return Kokkos::subview(_view, Kokkos::ALL, row_idx);
            }

            template<Integer TRow>
            KOKKOS_FORCEINLINE_FUNCTION auto Row(const TRow& row_idx) const -> typename std::enable_if<std::is_same<LayoutType, Kokkos::LayoutLeft>::value, Kokkos::Subview<ViewType, Kokkos::Impl::ALL_t, TRow>>::type
            {
                return Kokkos::subview(_view, Kokkos::ALL, row_idx);
            }

            template<Integer TRow>
            KOKKOS_FORCEINLINE_FUNCTION auto Row(const TRow& row_idx) -> typename std::enable_if<std::is_same<LayoutType, Kokkos::LayoutRight>::value, const Kokkos::Subview<ViewType, TRow, Kokkos::Impl::ALL_t>>::type
            {
                return Kokkos::subview(_view, row_idx, Kokkos::ALL);
            }

            template<Integer TRow>
            KOKKOS_FORCEINLINE_FUNCTION auto Row(const TRow& row_idx) const -> typename std::enable_if<std::is_same<LayoutType, Kokkos::LayoutRight>::value, const Kokkos::Subview<ViewType, TRow, Kokkos::Impl::ALL_t>>::type
            {
                return Kokkos::subview(_view, row_idx, Kokkos::ALL);
            }

            template<Integer TRowStart, Integer TRowEnd>
            KOKKOS_FORCEINLINE_FUNCTION auto Rows(const TRowStart& row_start_idx, const TRowEnd& row_end_idx) ->
                typename std::enable_if<std::is_same<LayoutType, Kokkos::LayoutLeft>::value, Kokkos::Subview<ViewType, Kokkos::Impl::ALL_t, Kokkos::pair<TRowStart, TRowEnd>>>::type
            {
                return Kokkos::subview(_view, Kokkos::ALL, Kokkos::make_pair(row_start_idx, row_end_idx));
            }

            template<Integer TRowStart, Integer TRowEnd>
            KOKKOS_FORCEINLINE_FUNCTION auto Rows(const TRowStart& row_start_idx, const TRowEnd& row_end_idx) ->
                typename std::enable_if<std::is_same<LayoutType, Kokkos::LayoutRight>::value, Kokkos::Subview<ViewType, Kokkos::pair<TRowStart, TRowEnd>, Kokkos::Impl::ALL_t>>::type
            {
                return Kokkos::subview(_view, Kokkos::make_pair(row_start_idx, row_end_idx), Kokkos::ALL);
            }

            template<Integer TRowStart, Integer TRowEnd>
            KOKKOS_FORCEINLINE_FUNCTION auto Rows(const Kokkos::pair<TRowStart, TRowEnd>& row_range_idx) ->
                typename std::enable_if<std::is_same<LayoutType, Kokkos::LayoutLeft>::value, Kokkos::Subview<ViewType, Kokkos::Impl::ALL_t, Kokkos::pair<TRowStart, TRowEnd>>>::type
            {
                return Kokkos::subview(_view, Kokkos::ALL, row_range_idx);
            }

            template<Integer TRowStart, Integer TRowEnd>
            KOKKOS_FORCEINLINE_FUNCTION auto Rows(const Kokkos::pair<TRowStart, TRowEnd>& row_range_idx) ->
                typename std::enable_if<std::is_same<LayoutType, Kokkos::LayoutRight>::value, Kokkos::Subview<ViewType, Kokkos::pair<TRowStart, TRowEnd>, Kokkos::Impl::ALL_t>>::type
            {
                return Kokkos::subview(_view, row_range_idx, Kokkos::ALL);
            }
#pragma endregion

#pragma region Column SubViews
            template<Integer TColumn>
            KOKKOS_FORCEINLINE_FUNCTION auto Column(const TColumn& column_idx) ->
                typename std::enable_if<Integer<TColumn> && std::is_same<LayoutType, Kokkos::LayoutLeft>::value, Kokkos::Subview<ViewType, TColumn, Kokkos::Impl::ALL_t>>::type
            {
                return Kokkos::subview(_view, column_idx, Kokkos::ALL);
            }

            template<Integer TColumn>
            KOKKOS_FORCEINLINE_FUNCTION auto Column(const TColumn& column_idx) const ->
                typename std::enable_if<Integer<TColumn> && std::is_same<LayoutType, Kokkos::LayoutLeft>::value, Kokkos::Subview<ViewType, TColumn, Kokkos::Impl::ALL_t>>::type
            {
                return Kokkos::subview(_view, column_idx, Kokkos::ALL);
            }

            template<Integer TColumn>
            KOKKOS_FORCEINLINE_FUNCTION auto Column(const TColumn& column_idx) ->
                typename std::enable_if<Integer<TColumn> && std::is_same<LayoutType, Kokkos::LayoutRight>::value, const Kokkos::Subview<ViewType, Kokkos::Impl::ALL_t, TColumn>>::type
            {
                return Kokkos::subview(_view, Kokkos::ALL, column_idx);
            }

            template<Integer TColumn>
            KOKKOS_FORCEINLINE_FUNCTION auto Column(const TColumn& column_idx) const ->
                typename std::enable_if<Integer<TColumn> && std::is_same<LayoutType, Kokkos::LayoutRight>::value, const Kokkos::Subview<ViewType, Kokkos::Impl::ALL_t, TColumn>>::type
            {
                return Kokkos::subview(_view, Kokkos::ALL, column_idx);
            }

            template<Integer TColumnStart, Integer TColumnEnd>
            KOKKOS_FORCEINLINE_FUNCTION auto Columns(const TColumnStart& column_start_idx, const TColumnEnd& column_end_idx) ->
                typename std::enable_if<std::is_same<LayoutType, Kokkos::LayoutLeft>::value, Kokkos::Subview<ViewType, Kokkos::pair<TColumnStart, TColumnEnd>, Kokkos::Impl::ALL_t>>::type
            {
                return Kokkos::subview(_view, Kokkos::make_pair(column_start_idx, column_end_idx), Kokkos::ALL);
            }

            template<Integer TColumnStart, Integer TColumnEnd>
            KOKKOS_FORCEINLINE_FUNCTION auto Columns(const TColumnStart& column_start_idx, const TColumnEnd& column_end_idx) ->
                typename std::enable_if<std::is_same<LayoutType, Kokkos::LayoutRight>::value, Kokkos::Subview<ViewType, Kokkos::Impl::ALL_t, Kokkos::pair<TColumnStart, TColumnEnd>>>::type
            {
                return Kokkos::subview(_view, Kokkos::ALL, Kokkos::make_pair(column_start_idx, column_end_idx));
            }

            template<Integer TColumnStart, Integer TColumnEnd>
            KOKKOS_FORCEINLINE_FUNCTION auto Columns(const Kokkos::pair<TColumnStart, TColumnEnd>& column_range_idx) ->
                typename std::enable_if<std::is_same<LayoutType, Kokkos::LayoutLeft>::value, Kokkos::Subview<ViewType, Kokkos::pair<TColumnStart, TColumnEnd>, Kokkos::Impl::ALL_t>>::type
            {
                return Kokkos::subview(_view, column_range_idx, Kokkos::ALL);
            }

            template<Integer TColumnStart, Integer TColumnEnd>
            KOKKOS_FORCEINLINE_FUNCTION auto Columns(const Kokkos::pair<TColumnStart, TColumnEnd>& column_range_idx) ->
                typename std::enable_if<std::is_same<LayoutType, Kokkos::LayoutRight>::value, Kokkos::Subview<ViewType, Kokkos::Impl::ALL_t, Kokkos::pair<TColumnStart, TColumnEnd>>>::type
            {
                return Kokkos::subview(_view, Kokkos::ALL, column_range_idx);
            }
#pragma endregion
        };

        template<typename DataType, class ExecutionSpace, class Layout>
        KOKKOS_FORCEINLINE_FUNCTION auto Matrix<DataType, ExecutionSpace, Layout>::clone(const Matrix<DataType, ExecutionSpace, Layout>& src)
        {
#if !defined(__CUDA_ARCH__)
            Matrix<typename ViewType::traits::non_const_value_type, typename ViewType::traits::execution_space> dest(src.label(), src.nrows(), src.ncolumns());

            Kokkos::deep_copy(dest.View(), src.View());
#else
            Matrix<typename ViewType::traits::non_const_value_type, typename ViewType::traits::execution_space> dest(new typename ViewType::traits::non_const_value_type[src.size()], src.nrows(), src.ncolumns());

            memcpy(dest.data(), src.data(), src.size() * sizeof(typename ViewType::traits::non_const_value_type));
#endif
            Kokkos::fence();

            return dest;
        }

        template<typename DataType, class ExecutionSpace, class Layout = typename ExecutionSpace::array_layout>
        class Matrix3x3
        {
        public:
            using ThisType = Matrix3x3<DataType, ExecutionSpace, Layout>;

            inline static constexpr size_type NRows    = 3;
            inline static constexpr size_type NColumns = 3;

            typedef Layout                                                  LayoutType;
            typedef View<DataType[NRows][NColumns], Layout, ExecutionSpace> ViewType;
            typedef typename ViewType::traits                               traits;

            typedef typename ViewType::size_type                    size_type;
            typedef typename ViewType::const_value_type             const_value_type;
            typedef typename ViewType::traits::non_const_value_type non_const_value_type;

            using const_type     = Matrix3x3<const DataType, ExecutionSpace, Layout>;
            using non_const_type = Matrix3x3<DataType, ExecutionSpace, Layout>;

            inline static constexpr size_type row_index    = std::is_same<LayoutType, Kokkos::LayoutLeft>::value ? 1 : 0;
            inline static constexpr size_type column_index = std::is_same<LayoutType, Kokkos::LayoutLeft>::value ? 0 : 1;
            inline static constexpr size_type Rank         = 2;

        private:
            ViewType _view;

        public:
            __inline explicit Matrix3x3(std::string label) : _view(label, std::is_same<LayoutType, Kokkos::LayoutLeft>::value ? NColumns : NRows, std::is_same<LayoutType, Kokkos::LayoutLeft>::value ? NRows : NColumns) {}

            KOKKOS_INLINE_FUNCTION explicit Matrix3x3(non_const_value_type* ptr) :
                _view(ptr, std::is_same<LayoutType, Kokkos::LayoutLeft>::value ? NColumns : NRows, std::is_same<LayoutType, Kokkos::LayoutLeft>::value ? NRows : NColumns)
            {
            }

            template<typename OtherDataType>
            __inline Matrix3x3(const Vector<OtherDataType, ExecutionSpace>& vector) : Matrix3x3(vector.label(), vector.extent(0), vector.extent(0))
            {
                for(size_type i0 = 0; i0 < _view.extent(0); ++i0)
                {
                    for(size_type i1 = 0; i1 < _view.extent(1); ++i1)
                    {
                        if(i0 == i1)
                        {
                            _view(i0, i1) = vector(i0);
                        }
                        else
                        {
                            _view(i0, i1) = 0.0;
                        }
                    }
                }
            }

            __inline Matrix3x3()  = default;
            __inline ~Matrix3x3() = default;
            KOKKOS_INLINE_FUNCTION Matrix3x3(const Matrix3x3& other) : _view(other.View()) {}
            __inline Matrix3x3(Matrix3x3&&)           = default;
            KOKKOS_INLINE_FUNCTION Matrix3x3& operator=(const Matrix3x3& other)
            {
                if(this == &other)
                {
                    return *this;
                }
                _view = other.View();
                return *this;
            }
            __inline Matrix3x3& operator=(Matrix3x3&&) = default;

            template<typename OtherDataType, class OtherExecutionSpace, class OtherLayout = typename OtherExecutionSpace::array_layout>
            KOKKOS_INLINE_FUNCTION Matrix3x3(const Matrix3x3<OtherDataType, OtherExecutionSpace, OtherLayout>& other) : _view(other.View())
            {
            }

            template<typename OtherDataType, class OtherExecutionSpace, class OtherLayout = typename OtherExecutionSpace::array_layout>
            KOKKOS_INLINE_FUNCTION Matrix3x3& operator=(const Matrix3x3<OtherDataType, OtherExecutionSpace, OtherLayout>& other)
            {
                if(this == &other)
                {
                    return *this;
                }
                _view = other.View();
                return *this;
            }

#pragma region Indexers
            template<Integer TRow, Integer TColumn>
            KOKKOS_FORCEINLINE_FUNCTION auto operator()(const TRow& row, const TColumn& column) ->
                typename std::enable_if<Integer<TRow> && std::is_same<LayoutType, Kokkos::LayoutLeft>::value, typename ViewType::reference_type>::type
            {
                return _view(column, row);
            }

            template<Integer TRow, Integer TColumn>
            KOKKOS_FORCEINLINE_FUNCTION auto operator()(const TRow& row, const TColumn& column) const ->
                typename std::enable_if<Integer<TRow> && std::is_same<LayoutType, Kokkos::LayoutLeft>::value, const typename ViewType::reference_type>::type
            {
                return _view(column, row);
            }

            template<Integer TRow, Integer TColumn>
            KOKKOS_FORCEINLINE_FUNCTION auto operator()(const TRow& row, const TColumn& column) ->
                typename std::enable_if<Integer<TRow> && std::is_same<LayoutType, Kokkos::LayoutRight>::value, typename ViewType::reference_type>::type
            {
                return _view(row, column);
            }

            template<Integer TRow, Integer TColumn>
            KOKKOS_FORCEINLINE_FUNCTION auto operator()(const TRow& row, const TColumn& column) const ->
                typename std::enable_if<Integer<TRow> && std::is_same<LayoutType, Kokkos::LayoutRight>::value, const typename ViewType::reference_type>::type
            {
                return _view(row, column);
            }

            template<typename MatrixViewType, Integer TRow, Integer TColumn>
            KOKKOS_FORCEINLINE_FUNCTION static auto Indexer(MatrixViewType& view, const TRow& row, const TColumn& column) ->
                typename std::enable_if<MatrixViewType::Rank == 2 && Integer<TRow> && std::is_same<typename MatrixViewType::traits::array_layout, Kokkos::LayoutLeft>::value, typename ViewType::reference_type>::type
            {
                return view(column, row);
            }

            template<typename MatrixViewType, Integer TRow, Integer TColumn>
            KOKKOS_FORCEINLINE_FUNCTION static auto Indexer(MatrixViewType& view, const TRow& row, const TColumn& column) ->
                typename std::enable_if<MatrixViewType::Rank == 2 && Integer<TRow> && std::is_same<typename MatrixViewType::traits::array_layout, Kokkos::LayoutRight>::value, typename ViewType::reference_type>::type
            {
                return view(row, column);
            }

            template<typename MatrixViewType, Integer TRow, Integer TColumn>
            KOKKOS_FORCEINLINE_FUNCTION static auto Indexer(MatrixViewType& view, const TRow& row, const TColumn& column) ->
                typename std::enable_if<MatrixViewType::Rank == 2 && Integer<TRow> && std::is_same<typename MatrixViewType::traits::array_layout, Kokkos::LayoutStride>::value, typename ViewType::reference_type>::type
            {
                return view(row, column);
            }

            template<typename MatrixViewType, Integer TRow, Integer TColumn>
            KOKKOS_FORCEINLINE_FUNCTION static auto Indexer(MatrixViewType& view, const TRow& row, const TColumn& column) ->
                typename std::enable_if<MatrixViewType::Rank == 1 && Integer<TRow>, typename ViewType::reference_type>::type
            {
                return view(row);
            }

#pragma endregion

#pragma region Member Functions
            __inline operator ViewType()
            {
                return _view;
            }

            __inline operator const ViewType() const
            {
                return _view;
            }

            KOKKOS_FORCEINLINE_FUNCTION ViewType& View()
            {
                return _view;
            }

            KOKKOS_FORCEINLINE_FUNCTION const ViewType& View() const
            {
                return _view;
            }

            KOKKOS_INLINE_FUNCTION constexpr typename ViewType::pointer_type data() const
            {
                return _view.data();
            }

            __inline const std::string label() const
            {
                return _view.label();
            }

            KOKKOS_FORCEINLINE_FUNCTION constexpr const size_type nrows() const
            {
                return NRows;
            }

            KOKKOS_FORCEINLINE_FUNCTION constexpr const size_type ncolumns() const
            {
                return NColumns;
            }

            KOKKOS_INLINE_FUNCTION constexpr const size_type size() const
            {
                return nrows() * ncolumns();
            }
#pragma endregion

#pragma region Row& Column operator() SubViews
            template<Integer TRowStart, Integer TRowEnd>
            KOKKOS_FORCEINLINE_FUNCTION auto operator()(const Kokkos::pair<TRowStart, TRowEnd>& row_range_idx) ->
                typename std::enable_if<std::is_same<LayoutType, Kokkos::LayoutLeft>::value, Kokkos::Subview<ViewType, Kokkos::pair<TRowStart, TRowEnd>, Kokkos::Impl::ALL_t>>::type
            {
                return Kokkos::subview(_view, row_range_idx, Kokkos::ALL);
            }

            template<Integer TRowStart, Integer TRowEnd>
            KOKKOS_FORCEINLINE_FUNCTION auto operator()(const Kokkos::pair<TRowStart, TRowEnd>& row_range_idx) ->
                typename std::enable_if<std::is_same<LayoutType, Kokkos::LayoutRight>::value, Kokkos::Subview<ViewType, Kokkos::Impl::ALL_t, Kokkos::pair<TRowStart, TRowEnd>>>::type
            {
                return Kokkos::subview(_view, Kokkos::ALL, row_range_idx);
            }

            template<Integer TRow, Integer TColumnStart, Integer TColumnEnd>
            KOKKOS_FORCEINLINE_FUNCTION auto operator()(const TRow& row_idx, const Kokkos::pair<TColumnStart, TColumnEnd>& column_range_idx) ->
                typename std::enable_if<std::is_same<LayoutType, Kokkos::LayoutLeft>::value, Kokkos::Subview<ViewType, TRow, Kokkos::pair<TColumnStart, TColumnEnd>>>::type
            {
                return Kokkos::subview(_view, row_idx, column_range_idx);
            }

            template<Integer TRow, Integer TColumnStart, Integer TColumnEnd>
            KOKKOS_FORCEINLINE_FUNCTION auto operator()(const TRow& row_idx, const Kokkos::pair<TColumnStart, TColumnEnd>& column_range_idx) ->
                typename std::enable_if<std::is_same<LayoutType, Kokkos::LayoutRight>::value, Kokkos::Subview<ViewType, Kokkos::pair<TColumnStart, TColumnEnd>, TRow>>::type
            {
                return Kokkos::subview(_view, column_range_idx, row_idx);
            }

            template<Integer TRowStart, Integer TRowEnd, Integer TColumn>
            KOKKOS_FORCEINLINE_FUNCTION auto operator()(const Kokkos::pair<TRowStart, TRowEnd>& row_range_idx, const TColumn& column_idx) ->
                typename std::enable_if<std::is_same<LayoutType, Kokkos::LayoutLeft>::value, Kokkos::Subview<ViewType, Kokkos::pair<TRowStart, TRowEnd>, TColumn>>::type
            {
                return Kokkos::subview(_view, row_range_idx, column_idx);
            }

            template<Integer TRowStart, Integer TRowEnd, Integer TColumn>
            KOKKOS_FORCEINLINE_FUNCTION auto operator()(const Kokkos::pair<TRowStart, TRowEnd>& row_range_idx, const TColumn& column_idx) ->
                typename std::enable_if<std::is_same<LayoutType, Kokkos::LayoutRight>::value, Kokkos::Subview<ViewType, TColumn, Kokkos::pair<TRowStart, TRowEnd>>>::type
            {
                return Kokkos::subview(_view, column_idx, row_range_idx);
            }

            template<Integer TRowStart, Integer TRowEnd, Integer TColumnStart, Integer TColumnEnd>
            KOKKOS_FORCEINLINE_FUNCTION auto operator()(const Kokkos::pair<TRowStart, TRowEnd>& row_range_idx, const Kokkos::pair<TColumnStart, TColumnEnd>& column_range_idx) ->
                typename std::enable_if<std::is_same<LayoutType, Kokkos::LayoutLeft>::value, Kokkos::Subview<ViewType, Kokkos::pair<TRowStart, TRowEnd>, Kokkos::pair<TColumnStart, TColumnEnd>>>::type
            {
                return Kokkos::subview(_view, row_range_idx, column_range_idx);
            }

            template<Integer TRowStart, Integer TRowEnd, Integer TColumnStart, Integer TColumnEnd>
            KOKKOS_FORCEINLINE_FUNCTION auto operator()(const Kokkos::pair<TRowStart, TRowEnd>& row_range_idx, const Kokkos::pair<TColumnStart, TColumnEnd>& column_range_idx) ->
                typename std::enable_if<std::is_same<LayoutType, Kokkos::LayoutRight>::value, Kokkos::Subview<ViewType, Kokkos::pair<TColumnStart, TColumnEnd>, Kokkos::pair<TRowStart, TRowEnd>>>::type
            {
                return Kokkos::subview(_view, column_range_idx, row_range_idx);
            }
#pragma endregion

#pragma region Row SubViews
            template<Integer TRow>
            KOKKOS_FORCEINLINE_FUNCTION auto Row(const TRow& row_idx) -> typename std::enable_if<std::is_same<LayoutType, Kokkos::LayoutLeft>::value, Kokkos::Subview<ViewType, Kokkos::Impl::ALL_t, TRow>>::type
            {
                return Kokkos::subview(_view, Kokkos::ALL, row_idx);
            }

            template<Integer TRow>
            KOKKOS_FORCEINLINE_FUNCTION auto Row(const TRow& row_idx) const -> typename std::enable_if<std::is_same<LayoutType, Kokkos::LayoutLeft>::value, Kokkos::Subview<ViewType, Kokkos::Impl::ALL_t, TRow>>::type
            {
                return Kokkos::subview(_view, Kokkos::ALL, row_idx);
            }

            template<Integer TRow>
            KOKKOS_FORCEINLINE_FUNCTION auto Row(const TRow& row_idx) -> typename std::enable_if<std::is_same<LayoutType, Kokkos::LayoutRight>::value, const Kokkos::Subview<ViewType, TRow, Kokkos::Impl::ALL_t>>::type
            {
                return Kokkos::subview(_view, row_idx, Kokkos::ALL);
            }

            template<Integer TRow>
            KOKKOS_FORCEINLINE_FUNCTION auto Row(const TRow& row_idx) const -> typename std::enable_if<std::is_same<LayoutType, Kokkos::LayoutRight>::value, const Kokkos::Subview<ViewType, TRow, Kokkos::Impl::ALL_t>>::type
            {
                return Kokkos::subview(_view, row_idx, Kokkos::ALL);
            }

            template<Integer TRowStart, Integer TRowEnd>
            KOKKOS_FORCEINLINE_FUNCTION auto Rows(const TRowStart& row_start_idx, const TRowEnd& row_end_idx) ->
                typename std::enable_if<std::is_same<LayoutType, Kokkos::LayoutLeft>::value, Kokkos::Subview<ViewType, Kokkos::Impl::ALL_t, Kokkos::pair<TRowStart, TRowEnd>>>::type
            {
                return Kokkos::subview(_view, Kokkos::ALL, Kokkos::make_pair(row_start_idx, row_end_idx));
            }

            template<Integer TRowStart, Integer TRowEnd>
            KOKKOS_FORCEINLINE_FUNCTION auto Rows(const TRowStart& row_start_idx, const TRowEnd& row_end_idx) ->
                typename std::enable_if<std::is_same<LayoutType, Kokkos::LayoutRight>::value, Kokkos::Subview<ViewType, Kokkos::pair<TRowStart, TRowEnd>, Kokkos::Impl::ALL_t>>::type
            {
                return Kokkos::subview(_view, Kokkos::make_pair(row_start_idx, row_end_idx), Kokkos::ALL);
            }

            template<Integer TRowStart, Integer TRowEnd>
            KOKKOS_FORCEINLINE_FUNCTION auto Rows(const Kokkos::pair<TRowStart, TRowEnd>& row_range_idx) ->
                typename std::enable_if<std::is_same<LayoutType, Kokkos::LayoutLeft>::value, Kokkos::Subview<ViewType, Kokkos::Impl::ALL_t, Kokkos::pair<TRowStart, TRowEnd>>>::type
            {
                return Kokkos::subview(_view, Kokkos::ALL, row_range_idx);
            }

            template<Integer TRowStart, Integer TRowEnd>
            KOKKOS_FORCEINLINE_FUNCTION auto Rows(const Kokkos::pair<TRowStart, TRowEnd>& row_range_idx) ->
                typename std::enable_if<std::is_same<LayoutType, Kokkos::LayoutRight>::value, Kokkos::Subview<ViewType, Kokkos::pair<TRowStart, TRowEnd>, Kokkos::Impl::ALL_t>>::type
            {
                return Kokkos::subview(_view, row_range_idx, Kokkos::ALL);
            }
#pragma endregion

#pragma region Column SubViews
            template<Integer TColumn>
            KOKKOS_FORCEINLINE_FUNCTION auto Column(const TColumn& column_idx) ->
                typename std::enable_if<Integer<TColumn> && std::is_same<LayoutType, Kokkos::LayoutLeft>::value, Kokkos::Subview<ViewType, TColumn, Kokkos::Impl::ALL_t>>::type
            {
                return Kokkos::subview(_view, column_idx, Kokkos::ALL);
            }

            template<Integer TColumn>
            KOKKOS_FORCEINLINE_FUNCTION auto Column(const TColumn& column_idx) const ->
                typename std::enable_if<Integer<TColumn> && std::is_same<LayoutType, Kokkos::LayoutLeft>::value, Kokkos::Subview<ViewType, TColumn, Kokkos::Impl::ALL_t>>::type
            {
                return Kokkos::subview(_view, column_idx, Kokkos::ALL);
            }

            template<Integer TColumn>
            KOKKOS_FORCEINLINE_FUNCTION auto Column(const TColumn& column_idx) ->
                typename std::enable_if<Integer<TColumn> && std::is_same<LayoutType, Kokkos::LayoutRight>::value, const Kokkos::Subview<ViewType, Kokkos::Impl::ALL_t, TColumn>>::type
            {
                return Kokkos::subview(_view, Kokkos::ALL, column_idx);
            }

            template<Integer TColumn>
            KOKKOS_FORCEINLINE_FUNCTION auto Column(const TColumn& column_idx) const ->
                typename std::enable_if<Integer<TColumn> && std::is_same<LayoutType, Kokkos::LayoutRight>::value, const Kokkos::Subview<ViewType, Kokkos::Impl::ALL_t, TColumn>>::type
            {
                return Kokkos::subview(_view, Kokkos::ALL, column_idx);
            }

            template<Integer TColumnStart, Integer TColumnEnd>
            KOKKOS_FORCEINLINE_FUNCTION auto Columns(const TColumnStart& column_start_idx, const TColumnEnd& column_end_idx) ->
                typename std::enable_if<std::is_same<LayoutType, Kokkos::LayoutLeft>::value, Kokkos::Subview<ViewType, Kokkos::pair<TColumnStart, TColumnEnd>, Kokkos::Impl::ALL_t>>::type
            {
                return Kokkos::subview(_view, Kokkos::make_pair(column_start_idx, column_end_idx), Kokkos::ALL);
            }

            template<Integer TColumnStart, Integer TColumnEnd>
            KOKKOS_FORCEINLINE_FUNCTION auto Columns(const TColumnStart& column_start_idx, const TColumnEnd& column_end_idx) ->
                typename std::enable_if<std::is_same<LayoutType, Kokkos::LayoutRight>::value, Kokkos::Subview<ViewType, Kokkos::Impl::ALL_t, Kokkos::pair<TColumnStart, TColumnEnd>>>::type
            {
                return Kokkos::subview(_view, Kokkos::ALL, Kokkos::make_pair(column_start_idx, column_end_idx));
            }

            template<Integer TColumnStart, Integer TColumnEnd>
            KOKKOS_FORCEINLINE_FUNCTION auto Columns(const Kokkos::pair<TColumnStart, TColumnEnd>& column_range_idx) ->
                typename std::enable_if<std::is_same<LayoutType, Kokkos::LayoutLeft>::value, Kokkos::Subview<ViewType, Kokkos::pair<TColumnStart, TColumnEnd>, Kokkos::Impl::ALL_t>>::type
            {
                return Kokkos::subview(_view, column_range_idx, Kokkos::ALL);
            }

            template<Integer TColumnStart, Integer TColumnEnd>
            KOKKOS_FORCEINLINE_FUNCTION auto Columns(const Kokkos::pair<TColumnStart, TColumnEnd>& column_range_idx) ->
                typename std::enable_if<std::is_same<LayoutType, Kokkos::LayoutRight>::value, Kokkos::Subview<ViewType, Kokkos::Impl::ALL_t, Kokkos::pair<TColumnStart, TColumnEnd>>>::type
            {
                return Kokkos::subview(_view, Kokkos::ALL, column_range_idx);
            }
#pragma endregion
        };

        template<typename DataType, class ExecutionSpace, class Layout = typename ExecutionSpace::array_layout>
        using Tensor = View<DataType***, Layout, ExecutionSpace>;

        template<typename DataType, class ExecutionSpace, class Layout = typename ExecutionSpace::array_layout>
        using Tesseract = View<DataType****, Layout, ExecutionSpace>;

        // template<class DataType>
        // class SparseVectorElement
        //{
        //    size_type _index;
        //    DataType  _value;
        //
        // public:
        //    SparseVectorElement(const size_type& i, const DataType& a) : _index(i), _value(a) {}
        //
        //    __inline ~SparseVectorElement()                          = default;
        //    __inline SparseVectorElement(const SparseVectorElement&) = default;
        //    __inline SparseVectorElement(SparseVectorElement&&)      = default;
        //    __inline SparseVectorElement& operator=(const SparseVectorElement&) = default;
        //    __inline SparseVectorElement& operator=(SparseVectorElement&&) = default;
        //
        //    KOKKOS_INLINE_FUNCTION constexpr size_type& Index()
        //    {
        //        return _index;
        //    }
        //    KOKKOS_INLINE_FUNCTION constexpr const size_type& Index() const
        //    {
        //        return _index;
        //    }
        //
        //    KOKKOS_INLINE_FUNCTION constexpr DataType& Value()
        //    {
        //        return _value;
        //    }
        //    KOKKOS_INLINE_FUNCTION constexpr const DataType& Value() const
        //    {
        //        return _value;
        //    }
        //
        //    KOKKOS_INLINE_FUNCTION friend constexpr int compare(const SparseVectorElement& lhs, const SparseVectorElement& rhs)
        //    {
        //        if (lhs.Index() < rhs.Index())
        //        {
        //            return -1;
        //        }
        //
        //        if (lhs.Index() > rhs.Index())
        //        {
        //            return 1;
        //        }
        //
        //        return 0;
        //    }
        //};
        //
        // template<typename DataType, class ExecutionSpace>
        // using SparseVector = View<SparseVectorElement<DataType>*, typename ExecutionSpace::array_layout, ExecutionSpace>;
        //
        // template<typename DataType, typename IndexType, class ExecutionSpace, class Layout = typename ExecutionSpace::array_layout>
        // class SparseMatrix
        //{
        // public:
        //    typedef Layout                                   LayoutType;
        //    typedef View<IndexType*, Layout, ExecutionSpace> RowIndexViewType;
        //    typedef View<IndexType*, Layout, ExecutionSpace> ColumnPtrViewType;
        //    typedef View<DataType*, Layout, ExecutionSpace>  ValueViewType;
        //
        //    using const_type     = SparseMatrix<const DataType, ExecutionSpace, Layout>;
        //    using non_const_type = SparseMatrix<DataType, ExecutionSpace, Layout>;
        //
        //    inline static constexpr size_type Rank = 2;
        //
        // private:
        //    IndexType _nRows;
        //    IndexType _nColumns;
        //
        //    RowIndexViewType  _rowIndex;
        //    ColumnPtrViewType _columnPtr;
        //    ValueViewType     _values;
        //
        // public:
        //    template<Integer NRows, Integer NColumns, Integer NValues>
        //    __inline explicit SparseMatrix(std::string label, NRows nRows, NColumns nColumns, NValues nValues) :
        //        _nRows(nRows),
        //        _nColumns(nColumns),
        //        _rowIndex(nValues + 1, 0),
        //        _columnPtr(nValues, 0),
        //        _values(nValues)
        //    {
        //    }
        //
        //    __inline SparseMatrix()                    = default;
        //    __inline ~SparseMatrix()                   = default;
        //    __inline SparseMatrix(const SparseMatrix&) = default;
        //    __inline SparseMatrix(SparseMatrix&&)      = default;
        //    __inline SparseMatrix& operator=(const SparseMatrix&) = default;
        //    __inline SparseMatrix& operator=(SparseMatrix&&) = default;
        //
        //    // template<typename OtherDataType, class OtherExecutionSpace, class OtherLayout = typename OtherExecutionSpace::array_layout>
        //    //__inline SparseMatrix(const SparseMatrix<OtherDataType, OtherExecutionSpace, OtherLayout>& other) : _view(other.View())
        //    //{
        //    //}
        //
        //    // template<typename OtherDataType, class OtherExecutionSpace, class OtherLayout = typename OtherExecutionSpace::array_layout>
        //    //__inline SparseMatrix& operator=(const SparseMatrix<OtherDataType, OtherExecutionSpace, OtherLayout>& other)
        //    //{
        //    //    if (this == &other)
        //    //    {
        //    //        return *this;
        //    //    }
        //    //    _view = other.View();
        //    //    return *this;
        //    //}
        //
        //    // template<typename TRow, typename TColumn>
        //    // KOKKOS_FORCEINLINE_FUNCTION auto operator()(const TRow& row, const TColumn& column) ->
        //    //    typename std::enable_if<Integer<TRow>::value && Integer<TColumn> && std::is_same<LayoutType, Kokkos::LayoutLeft>::value,
        //    //                            typename ViewType::reference_type>::type
        //    //{
        //    //    return _view(column, row);
        //    //}
        //
        //    // template<typename TRow, typename TColumn>
        //    // KOKKOS_FORCEINLINE_FUNCTION auto operator()(const TRow& row, const TColumn& column) const ->
        //    //    typename std::enable_if<Integer<TRow>::value && Integer<TColumn> && std::is_same<LayoutType, Kokkos::LayoutLeft>::value,
        //    //                            const typename ViewType::reference_type>::type
        //    //{
        //    //    return _view(column, row);
        //    //}
        //
        //    // template<Integer TRow, Integer TColumn>
        //    // KOKKOS_FORCEINLINE_FUNCTION auto operator()(const TRow& row, const TColumn& column) ->
        //    //    typename std::enable_if<Integer<TRow>::value && Integer<TColumn> && std::is_same<LayoutType, Kokkos::LayoutRight>::value,
        //    //                            typename ViewType::reference_type>::type
        //    //{
        //    //    return _view(row, column);
        //    //}
        //
        //    // template<Integer TRow, Integer TColumn>
        //    // KOKKOS_FORCEINLINE_FUNCTION auto operator()(const TRow& row, const TColumn& column) const ->
        //    //    typename std::enable_if<Integer<TRow>::value && Integer<TColumn> && std::is_same<LayoutType, Kokkos::LayoutRight>::value,
        //    //                            const typename ViewType::reference_type>::type
        //    //{
        //    //    return _view(row, column);
        //    //}
        //
        //    __inline RowIndexViewType& RowIndex()
        //    {
        //        return _rowIndex;
        //    }
        //
        //    __inline const RowIndexViewType& RowIndex() const
        //    {
        //        return _rowIndex;
        //    }
        //
        //    __inline ColumnPtrViewType& ColumnPtr()
        //    {
        //        return _columnPtr;
        //    }
        //
        //    __inline const ColumnPtrViewType& ColumnPtr() const
        //    {
        //        return _columnPtr;
        //    }
        //
        //    __inline ValueViewType& Values()
        //    {
        //        return _values;
        //    }
        //
        //    __inline const ValueViewType& Values() const
        //    {
        //        return _values;
        //    }
        //
        //    //__inline const std::string label() const
        //    //{
        //    //    return _view.label();
        //    //}
        //
        //    KOKKOS_FORCEINLINE_FUNCTION constexpr IndexType nrows() const
        //    {
        //        return _nRows;
        //    }
        //
        //    KOKKOS_FORCEINLINE_FUNCTION constexpr IndexType ncolumns() const
        //    {
        //        return _nColumns;
        //    }
        //
        //    KOKKOS_FORCEINLINE_FUNCTION constexpr IndexType nvalues() const
        //    {
        //        return _values.extent(0);
        //    }
        //
        //    // template<typename TRow, typename TColumn>
        //    // KOKKOS_FORCEINLINE_FUNCTION auto Row(const TRow& row_idx) ->
        //    //    typename std::enable_if<Integer<TRow>::value && Integer<TColumn> && std::is_same<LayoutType, Kokkos::LayoutLeft>::value,
        //    //                            typename ViewType::reference_type>::type
        //    //{
        //    //    return Kokkos::subview(_view, row_idx, Kokkos::ALL);
        //    //}
        //
        //    // template<typename TRow, typename TColumn>
        //    // KOKKOS_FORCEINLINE_FUNCTION auto Row(const TRow& row_idx) const ->
        //    //    typename std::enable_if<Integer<TRow>::value && Integer<TColumn> && std::is_same<LayoutType, Kokkos::LayoutLeft>::value,
        //    //                            const typename ViewType::reference_type>::type
        //    //{
        //    //    return Kokkos::subview(_view, Kokkos::ALL, row_idx);
        //    //}
        //
        //    // template<Integer TRow, Integer TColumn>
        //    // KOKKOS_FORCEINLINE_FUNCTION auto Column(const TColumn& column_idx) ->
        //    //    typename std::enable_if<Integer<TRow>::value && Integer<TColumn> && std::is_same<LayoutType, Kokkos::LayoutRight>::value,
        //    //                            typename ViewType::reference_type>::type
        //    //{
        //    //    return Kokkos::subview(_view, Kokkos::ALL, column_idx);
        //    //}
        //
        //    // template<Integer TRow, Integer TColumn>
        //    // KOKKOS_FORCEINLINE_FUNCTION auto Column(const TColumn& column_idx) const ->
        //    //    typename std::enable_if<Integer<TRow>::value && Integer<TColumn> && std::is_same<LayoutType, Kokkos::LayoutRight>::value,
        //    //                            const typename ViewType::reference_type>::type
        //    //{
        //    //    return Kokkos::subview(_view, column_idx, Kokkos::ALL);
        //    //}
        //};

        template<typename DataType, class ExecutionSpace>
        __inline static Matrix<DataType, ExecutionSpace> JoinByColumn(const Matrix<DataType, ExecutionSpace>& lhs, const Matrix<DataType, ExecutionSpace>& rhs)
        {
            Assert(lhs.nrows() == rhs.nrows());

            Matrix<DataType, ExecutionSpace> matrix(lhs.label() + rhs.label(), lhs.nrows(), lhs.ncolumns() + rhs.ncolumns());

            auto first_half = matrix.Columns(0ULL, lhs.ncolumns());

            Kokkos::deep_copy(first_half, lhs.View());

            auto second_half = matrix.Columns(lhs.ncolumns(), lhs.ncolumns() + rhs.ncolumns());

            Kokkos::deep_copy(second_half, rhs.View());

            return matrix;
        }

        template<typename DataType, class ExecutionSpace>
        __inline static Matrix<DataType, ExecutionSpace> JoinByColumn(const Vector<DataType, ExecutionSpace>& lhs, const Matrix<DataType, ExecutionSpace>& rhs)
        {
            Assert(lhs.extent(0) == rhs.nrows());

            Matrix<DataType, ExecutionSpace> matrix(lhs.label() + rhs.label(), rhs.nrows(), 1 + rhs.ncolumns());

            for(size_type i = 0; i < rhs.nrows(); i++)
            {
                matrix(i, 0ULL) = lhs(i);
            }

            auto second_half = matrix.Columns(1ULL, 1ULL + rhs.ncolumns());

            Kokkos::deep_copy(second_half, rhs.View());

            return matrix;
        }

        template<typename DataType, class ExecutionSpace>
        __inline static Matrix<DataType, ExecutionSpace> JoinByColumn(const Matrix<DataType, ExecutionSpace>& lhs, const Vector<DataType, ExecutionSpace>& rhs)
        {
            Assert(lhs.nrows() == rhs.extent(0));

            Matrix<DataType, ExecutionSpace> matrix(lhs.label() + rhs.label(), lhs.nrows(), lhs.ncolumns() + 1);

            auto first_half = matrix.Columns(0ULL, lhs.ncolumns());

            Kokkos::deep_copy(first_half, lhs.View());

            const size_type last_column = lhs.ncolumns();

            for(size_type i = 0; i < lhs.nrows(); i++)
            {
                matrix(i, last_column) = rhs(i);
            }

            return matrix;
        }

    }

    namespace Extension
    {
        template<typename ViewType>
        KOKKOS_FORCEINLINE_FUNCTION static auto Clone(ViewType src) -> std::enable_if_t<ViewType::Rank == 1, Vector<typename ViewType::traits::non_const_value_type, typename ViewType::traits::execution_space>>
        {
#if !defined(__CUDA_ARCH__)
            Vector<typename ViewType::traits::non_const_value_type, typename ViewType::traits::execution_space> dest(src);

            Kokkos::deep_copy(dest, src);
#else
            Vector<typename ViewType::traits::non_const_value_type, typename ViewType::traits::execution_space> dest(new typename ViewType::traits::non_const_value_type[src.extent(0)], src.extent(0));

            memcpy(dest.data(), src.data(), src.size() * sizeof(typename ViewType::traits::non_const_value_type));
#endif
            Kokkos::fence();

            return dest;
        }

        template<typename ViewType>
        KOKKOS_FORCEINLINE_FUNCTION static auto Clone(ViewType src) -> std::enable_if_t<ViewType::Rank == 2, Matrix<typename ViewType::traits::non_const_value_type, typename ViewType::traits::execution_space>>
        {
#if !defined(__CUDA_ARCH__)
            Matrix<typename ViewType::traits::non_const_value_type, typename ViewType::traits::execution_space> dest(src.label(), src.nrows(), src.ncolumns());

            Kokkos::deep_copy(dest.View(), src.View());
#else
            Matrix<typename ViewType::traits::non_const_value_type, typename ViewType::traits::execution_space> dest(new typename ViewType::traits::non_const_value_type[src.size()], src.nrows(), src.ncolumns());

            memcpy(dest.data(), src.data(), src.size() * sizeof(typename ViewType::traits::non_const_value_type));
#endif
            Kokkos::fence();

            return dest;
        }

        template<typename ViewType>
        KOKKOS_FORCEINLINE_FUNCTION static auto Clone(ViewType src)
            -> std::enable_if_t<ViewType::Rank == 3, View<typename ViewType::traits::non_const_value_type***, typename ViewType::traits::array_layout, typename ViewType::traits::execution_space>>
        {
#if !defined(__CUDA_ARCH__)
            View<typename ViewType::traits::non_const_value_type***, typename ViewType::traits::array_layout, typename ViewType::traits::execution_space> dest(src.label(), src.extent(0), src.extent(1), src.extent(2));

            Kokkos::deep_copy(dest, src);
#else
            View<typename ViewType::traits::non_const_value_type***, typename ViewType::traits::array_layout, typename ViewType::traits::execution_space> dest(new typename ViewType::traits::non_const_value_type[src.size()],
                                                                                                                                                               src.extent(0),
                                                                                                                                                               src.extent(1),
                                                                                                                                                               src.extent(2));

            memcpy(dest.data(), src.data(), src.size() * sizeof(typename ViewType::traits::non_const_value_type));
#endif
            Kokkos::fence();

            return dest;
        }

        template<typename ViewType>
        KOKKOS_FORCEINLINE_FUNCTION static auto Clone(ViewType src)
            -> std::enable_if_t<ViewType::Rank == 4, View<typename ViewType::traits::non_const_value_type****, typename ViewType::traits::array_layout, typename ViewType::traits::execution_space>>
        {
#if !defined(__CUDA_ARCH__)
            View<typename ViewType::traits::non_const_value_type****, typename ViewType::traits::array_layout, typename ViewType::traits::execution_space> dest(src.label(),
                                                                                                                                                                src.extent(0),
                                                                                                                                                                src.extent(1),
                                                                                                                                                                src.extent(2),
                                                                                                                                                                src.extent(3));

            Kokkos::deep_copy(dest, src);
#else
            View<typename ViewType::traits::non_const_value_type****, typename ViewType::traits::array_layout, typename ViewType::traits::execution_space> dest(new typename ViewType::traits::non_const_value_type[src.size()],
                                                                                                                                                                src.extent(0),
                                                                                                                                                                src.extent(1),
                                                                                                                                                                src.extent(2),
                                                                                                                                                                src.extent(3));

            memcpy(dest.data(), src.data(), src.size() * sizeof(typename ViewType::traits::non_const_value_type));
#endif
            Kokkos::fence();

            return dest;
        }

        template<typename ViewType>
        KOKKOS_FORCEINLINE_FUNCTION static auto Clone(ViewType src)
            -> std::enable_if_t<ViewType::Rank == 4, View<typename ViewType::traits::non_const_value_type*****, typename ViewType::traits::array_layout, typename ViewType::traits::execution_space>>
        {
#if !defined(__CUDA_ARCH__)
            View<typename ViewType::traits::non_const_value_type*****, typename ViewType::traits::array_layout, typename ViewType::traits::execution_space> dest(src.label(),
                                                                                                                                                                 src.extent(0),
                                                                                                                                                                 src.extent(1),
                                                                                                                                                                 src.extent(2),
                                                                                                                                                                 src.extent(3),
                                                                                                                                                                 src.extent(4));

            Kokkos::deep_copy(dest, src);
#else
            View<typename ViewType::traits::non_const_value_type*****, typename ViewType::traits::array_layout, typename ViewType::traits::execution_space> dest(new typename ViewType::traits::non_const_value_type[src.size()],
                                                                                                                                                                 src.extent(0),
                                                                                                                                                                 src.extent(1),
                                                                                                                                                                 src.extent(2),
                                                                                                                                                                 src.extent(3),
                                                                                                                                                                 src.extent(4));

            memcpy(dest.data(), src.data(), src.size() * sizeof(typename ViewType::traits::non_const_value_type));
#endif
            Kokkos::fence();

            return dest;
        }

        template<typename ViewType>
        KOKKOS_FORCEINLINE_FUNCTION static auto Clone(ViewType src)
            -> std::enable_if_t<ViewType::Rank == 4, View<typename ViewType::traits::non_const_value_type******, typename ViewType::traits::array_layout, typename ViewType::traits::execution_space>>
        {
#if !defined(__CUDA_ARCH__)
            View<typename ViewType::traits::non_const_value_type******, typename ViewType::traits::array_layout, typename ViewType::traits::execution_space>
                dest(src.label(), src.extent(0), src.extent(1), src.extent(2), src.extent(3), src.extent(4), src.extent(5));

            Kokkos::deep_copy(dest, src);
#else
            View<typename ViewType::traits::non_const_value_type******, typename ViewType::traits::array_layout, typename ViewType::traits::execution_space>
                dest(new typename ViewType::traits::non_const_value_type[src.size()], src.extent(0), src.extent(1), src.extent(2), src.extent(3), src.extent(4), src.extent(5));

            memcpy(dest.data(), src.data(), src.size() * sizeof(typename ViewType::traits::non_const_value_type));
#endif
            Kokkos::fence();

            return dest;
        }

        template<typename ViewType>
        KOKKOS_FORCEINLINE_FUNCTION static auto Clone(ViewType src)
            -> std::enable_if_t<ViewType::Rank == 4, View<typename ViewType::traits::non_const_value_type*******, typename ViewType::traits::array_layout, typename ViewType::traits::execution_space>>
        {
#if !defined(__CUDA_ARCH__)
            View<typename ViewType::traits::non_const_value_type*******, typename ViewType::traits::array_layout, typename ViewType::traits::execution_space>
                dest(src.label(), src.extent(0), src.extent(1), src.extent(2), src.extent(3), src.extent(4), src.extent(5), src.extent(6));

            Kokkos::deep_copy(dest, src);
#else
            View<typename ViewType::traits::non_const_value_type*******, typename ViewType::traits::array_layout, typename ViewType::traits::execution_space>
                dest(new typename ViewType::traits::non_const_value_type[src.size()], src.extent(0), src.extent(1), src.extent(2), src.extent(3), src.extent(4), src.extent(5), src.extent(6));

            memcpy(dest.data(), src.data(), src.size() * sizeof(typename ViewType::traits::non_const_value_type));
#endif
            Kokkos::fence();

            return dest;
        }

        template<typename ViewType>
        KOKKOS_FORCEINLINE_FUNCTION static auto Clone(ViewType src)
            -> std::enable_if_t<ViewType::Rank == 4, View<typename ViewType::traits::non_const_value_type********, typename ViewType::traits::array_layout, typename ViewType::traits::execution_space>>
        {
#if !defined(__CUDA_ARCH__)
            View<typename ViewType::traits::non_const_value_type********, typename ViewType::traits::array_layout, typename ViewType::traits::execution_space>
                dest(src.label(), src.extent(0), src.extent(1), src.extent(2), src.extent(3), src.extent(4), src.extent(5), src.extent(6), src.extent(7));

            Kokkos::deep_copy(dest, src);
#else
            View<typename ViewType::traits::non_const_value_type********, typename ViewType::traits::array_layout, typename ViewType::traits::execution_space>
                dest(new typename ViewType::traits::non_const_value_type[src.size()], src.extent(0), src.extent(1), src.extent(2), src.extent(3), src.extent(4), src.extent(5), src.extent(6), src.extent(7));

            memcpy(dest.data(), src.data(), src.size() * sizeof(typename ViewType::traits::non_const_value_type));
#endif
            Kokkos::fence();

            return dest;
        }

    }
}

//#include <utility>

namespace Kokkos
{
    namespace Extension
    {

        // template<typename T>
        // using array = T[];

        //#if __has_builtin(__sync_swap)
        //        using swap = __sync_swap;
        //#else
        template<typename T>
        KOKKOS_INLINE_FUNCTION static void swap(volatile T& a, volatile T& b) noexcept
        {
            const T tmp = Kokkos::atomic_exchange(&a, b);
            Kokkos::atomic_exchange(&b, tmp);
            // a     = std::move(b);
            // b     = std::move(tmp);
        }
        //#endif

        // template<FloatingPoint DataType, class ExecutionSpace, class Layout = typename ExecutionSpace::array_layout, typename int_type = size_int<DataType>>
        // static void quick_sort(View<DataType*, Layout, ExecutionSpace>& x, int_type left, int_type right)
        //{
        //    int_type i     = left;
        //    int_type j     = right;
        //    DataType pivot = x[((DataType)(left + right)) / 2.0];
        //
        //    do
        //    {
        //        while ((x[i] < pivot) && (i < right))
        //        {
        //            ++i;
        //        }
        //        while ((pivot < x[j]) && (j > left))
        //        {
        //            --j;
        //        }
        //
        //        if (i <= j)
        //        {
        //            Kokkos::swap<DataType>(x[i], x[j]);
        //            ++i;
        //            --j;
        //        }
        //    } while (i <= j);
        //
        //    if (left < j)
        //    {
        //        quick_sort(x, left, j);
        //    }
        //
        //    if (i < right)
        //    {
        //        quick_sort(x, i, right);
        //    }
        //}
        //
        // template<FloatingPoint DataType, class ExecutionSpace, class Layout = typename ExecutionSpace::array_layout, typename int_type = size_int<DataType>>
        // static void QuickSort(View<DataType*, Layout, ExecutionSpace>& x)
        //{
        //    quick_sort(x, int_type(0), x.extent(0) - 1);
        //}
    }

    using Extension::swap;
}

namespace Kokkos
{
    namespace Extension
    {
        template<FloatingPoint DataType, class ExecutionSpace>
        struct ColumnSums
        {
            typedef DataType value_type[];

            const size_type value_count;

            Matrix<DataType, ExecutionSpace> X_;

            ColumnSums(const Matrix<DataType, ExecutionSpace>& X) : value_count(X.ncolumns()), X_(X) {}

            KOKKOS_INLINE_FUNCTION void operator()(const size_type i, const size_type j, value_type sum) const
            {
                sum[j] += X_(i, j);
            }
        };

        template<FloatingPoint DataType, class ExecutionSpace>
        __inline static Vector<DataType, ExecutionSpace> SumByColumn(const Matrix<DataType, ExecutionSpace>& view)
        {
            using MDRangeType = Kokkos::MDRangePolicy<ExecutionSpace, Kokkos::Rank<2>, Kokkos::IndexType<size_type>>;
            using PointType   = typename MDRangeType::point_type;

            const size_type nRows    = view.nrows();
            const size_type nColumns = view.ncolumns();

            ColumnSums cs(view);

            Vector<DataType, ExecutionSpace> sums("sums", nColumns);
            Kokkos::deep_copy(sums, 0.0);

            MDRangeType policy(PointType{{0, 0}}, PointType{{nRows, nColumns}});

            Kokkos::parallel_reduce(policy, cs, sums);
            Kokkos::fence();

            return sums;
        }

        template<FloatingPoint DataType, class ExecutionSpace>
        struct MatrixRowSums
        {
            typedef DataType value_type[];

            const size_type value_count;

            Matrix<DataType, ExecutionSpace> X_;

            MatrixRowSums(const Matrix<DataType, ExecutionSpace>& X) : value_count(X.nrows()), X_(X) {}

            KOKKOS_INLINE_FUNCTION void operator()(const size_type i, const size_type j, value_type sum) const
            {
                sum[i] += X_(i, j);
            }
        };

        template<FloatingPoint DataType, class ExecutionSpace>
        __inline static Vector<DataType, ExecutionSpace> SumByRow(const Matrix<DataType, ExecutionSpace>& view)
        {
            using MDRangeType = Kokkos::MDRangePolicy<ExecutionSpace, Kokkos::Rank<2>, Kokkos::IndexType<size_type>>;
            using PointType   = typename MDRangeType::point_type;

            const size_type nRows    = view.nrows();
            const size_type nColumns = view.ncolumns();

            MatrixRowSums rs(view);

            Vector<DataType, ExecutionSpace> sums("sums", nRows);
            Kokkos::deep_copy(sums, 0.0);

            MDRangeType policy(PointType{{0, 0}}, PointType{{nRows, nColumns}});

            Kokkos::parallel_reduce(policy, rs, sums);
            Kokkos::fence();

            return sums;
        }

        template<FloatingPoint DataType, class ExecutionSpace>
        struct TensorRowSums
        {
            typedef DataType value_type[];

            const size_type value_count;

            Tensor<DataType, ExecutionSpace> X_;

            TensorRowSums(const Tensor<DataType, ExecutionSpace>& X) : value_count(X.extent(0)), X_(X) {}

            KOKKOS_INLINE_FUNCTION void operator()(const size_type i, const size_type j, const size_type k, value_type sum) const
            {
                sum[i] += X_(i, j, k);
            }
        };

        template<FloatingPoint DataType, class ExecutionSpace>
        __inline static Vector<DataType, ExecutionSpace> SumByRow(const Tensor<DataType, ExecutionSpace>& view)
        {
            using MDRangeType = Kokkos::MDRangePolicy<ExecutionSpace, Kokkos::Rank<3>, Kokkos::IndexType<size_type>>;
            using PointType   = typename MDRangeType::point_type;

            const size_type nRows    = view.extent(0);
            const size_type nColumns = view.extent(0);
            const size_type nPages   = view.extent(0);

            TensorRowSums rs(view);

            Vector<DataType, ExecutionSpace> sums("sums", nRows);
            Kokkos::deep_copy(sums, 0.0);

            MDRangeType policy(PointType{{0, 0, 0}}, PointType{{nRows, nColumns, nPages}});

            Kokkos::parallel_reduce(policy, rs, sums);
            Kokkos::fence();

            return sums;
        }

        template<class ViewType>
        __inline static ViewType Zeros(const size_type arg_N0 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
                                       const size_type arg_N1 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
                                       const size_type arg_N2 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
                                       const size_type arg_N3 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
                                       const size_type arg_N4 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
                                       const size_type arg_N5 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
                                       const size_type arg_N6 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
                                       const size_type arg_N7 = KOKKOS_IMPL_CTOR_DEFAULT_ARG)
        {
            ViewType v("", arg_N0, arg_N1, arg_N2, arg_N3, arg_N4, arg_N5, arg_N6, arg_N7);

            Kokkos::deep_copy(v, 0.0);

            return v;
        }

        template<FloatingPoint DataType, class ExecutionSpace>
        KOKKOS_INLINE_FUNCTION static Vector<DataType, ExecutionSpace> Range(const DataType& min, const DataType& max, const DataType& step = 1.0)
        {
            const int32 n = (int32)(std::abs(max - min) / step) + 1;

            Vector<DataType, ExecutionSpace> range("", n);

            DataType curr = min;

            for(int32 i = 0; i < n; ++i)
            {
                range[i] = curr;
                curr += step;
            }

            return range;
        }
    }

    namespace Extension
    {
        template<class ReturnViewType, class ViewType, typename DataType = typename ViewType::traits::non_const_value_type>
        struct CumulativeSumFunctor
        {
            using value_type = DataType;

            ViewType       _v;
            ReturnViewType _r;

            CumulativeSumFunctor(const ViewType& v, ReturnViewType& r) : _v(v), _r(r) {}

            KOKKOS_INLINE_FUNCTION void operator()(const size_type idx, value_type& value, const bool& final) const
            {
                value += _v(idx);

                if(final)
                {
                    _r(idx) = value;
                }
            }
        };

        template<class ViewType, typename DataType = typename ViewType::traits::non_const_value_type, typename ExecutionSpace = typename ViewType::traits::execution_space>
        KOKKOS_INLINE_FUNCTION static Vector<DataType, ExecutionSpace> CumulativeSum(const ViewType& values)
        {
            const size_t N = values.extent(0);

            const Kokkos::RangePolicy<ExecutionSpace, Kokkos::IndexType<size_type>> range(0, N);

            Vector<DataType, ExecutionSpace> result("Cumulative" + values.label(), N);

            CumulativeSumFunctor<Vector<DataType, ExecutionSpace>, ViewType> functor(values, result);

            Kokkos::parallel_scan(range, functor);

            return result;
        }
    }
}

namespace Kokkos
{
    using Kokkos::Extension::CumulativeSum;
    using Kokkos::Extension::Range;
    using Kokkos::Extension::Zeros;
}

namespace Kokkos
{
    namespace Extension
    {
#if 0
        class MMAPAllocation
        {
            size_type                    sz;
            CUmemGenericAllocationHandle hdl;
            CUmemAccessDesc              accessDescriptor;
            CUdeviceptr                  ptr;

        public:
            MMAPAllocation(const size_type size, const int dev = 0)
            {
                CUmemAllocationProp prop = {};
                prop.type                = CU_MEM_ALLOCATION_TYPE_PINNED;
                prop.location.type       = CU_MEM_LOCATION_TYPE_DEVICE;
                prop.location.id         = dev;

                accessDescriptor.location = prop.location;
                accessDescriptor.flags    = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;

                size_type aligned_sz;

                ThrowIfFailed(cuMemGetAllocationGranularity(&aligned_sz, &prop, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED));

                sz = ((size + aligned_sz - 1) / aligned_sz) * aligned_sz;

                ThrowIfFailed(cuMemAddressReserve(&ptr, sz, 0ULL, 0ULL, 0ULL));
                ThrowIfFailed(cuMemCreate(&hdl, sz, &prop, 0));
                ThrowIfFailed(cuMemMap(ptr, sz, 0ULL, hdl, 0ULL));
                ThrowIfFailed(cuMemSetAccess(ptr, sz, &accessDescriptor, 1ULL));
            }

            ~MMAPAllocation()
            {
                ThrowIfFailed(cuMemUnmap(ptr, sz));
                ThrowIfFailed(cuMemAddressFree(ptr, sz));
                ThrowIfFailed(cuMemRelease(hdl));
            }
        };

        /// <summary>
        ///
        /// vector<CUdevice> mappingDevices;
        /// mappingDevices.push_back(cuDevice);
        ///
        /// vector<CUdevice> backingDevices = getBackingDevices(cuDevice);
        ///
        /// checkCudaErrors(simpleMallocMultiDeviceMmap(&d_A, &allocationSize, size, backingDevices, mappingDevices));
        /// checkCudaErrors(simpleMallocMultiDeviceMmap(&d_B, NULL, size, backingDevices, mappingDevices));
        /// checkCudaErrors(simpleMallocMultiDeviceMmap(&d_C, NULL, size, backingDevices, mappingDevices));
        ///
        ///
        ///
        /// </summary>
        class MultiDeviceMMAP
        {
            size_type                                 sz;
            std::vector<CUmemGenericAllocationHandle> memHandles;
            std::vector<CUmemAccessDesc>              accessDescriptors;

            CUdeviceptr ptr;

        public:
            MultiDeviceMMAP(const int dev, const std::vector<CUdevice>& residentDevices, const std::vector<CUdevice>& mappingDevices, const size_type size, const size_type align)
            {
                memHandles.resize(residentDevices.size());
                accessDescriptors.resize(mappingDevices.size());

                size_type min_granularity = 0;
                size_type granularity;

                CUmemAllocationProp prop = {};
                prop.type                = CU_MEM_ALLOCATION_TYPE_PINNED;
                prop.location.type       = CU_MEM_LOCATION_TYPE_DEVICE;

                for (int idx = 0; idx < residentDevices.size(); idx++)
                {
                    granularity = 0;

                    prop.location.id = residentDevices[idx];

                    ThrowIfFailed(cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED));

                    if (min_granularity < granularity)
                    {
                        min_granularity = granularity;
                    }
                }

                for (size_type idx = 0; idx < mappingDevices.size(); idx++)
                {
                    prop.location.id = mappingDevices[idx];

                    ThrowIfFailed(cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED));

                    if (min_granularity < granularity)
                    {
                        min_granularity = granularity;
                    }
                }

                sz = round_up(size, residentDevices.size() * min_granularity);

                size_type stripeSize = sz / residentDevices.size();

                ThrowIfFailed(cuMemAddressReserve(&ptr, sz, align, 0ULL, 0ULL));

                for (size_type idx = 0; idx < residentDevices.size(); idx++)
                {
                    prop.location.id = residentDevices[idx];

                    ThrowIfFailed(cuMemCreate(&memHandles[idx], stripeSize, &prop, 0ULL));
                    ThrowIfFailed(cuMemMap(ptr + (stripeSize * idx), stripeSize, 0ULL, &memHandles[idx], 0ULL));
                }

                for (size_type idx = 0; idx < mappingDevices.size(); idx++)
                {
                    accessDescriptors[idx].location.type = CU_MEM_LOCATION_TYPE_DEVICE;
                    accessDescriptors[idx].location.id   = mappingDevices[idx];
                    accessDescriptors[idx].flags         = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
                }

                ThrowIfFailed(cuMemSetAccess(ptr, sz, &accessDescriptors[0], accessDescriptors.size()));
            }

            ~MultiDeviceMMAP()
            {
                ThrowIfFailed(cuMemUnmap(ptr, sz));
                ThrowIfFailed(cuMemAddressFree(ptr, sz));

                for (size_type idx = 0; idx < memHandles.size(); idx++)
                {
                    ThrowIfFailed(cuMemRelease(memHandles[idx]));
                }
            }

            static vector<CUdevice> getBackingDevices(const CUdevice cuDevice)
            {
                int num_devices;

                ThrowIfFailed(cuDeviceGetCount(&num_devices));

                vector<CUdevice> backingDevices;
                backingDevices.push_back(cuDevice);
                for (int dev = 0; dev < num_devices; dev++)
                {
                    int capable      = 0;
                    int attributeVal = 0;

                    // The mapping device is already in the backingDevices vector
                    if (dev == cuDevice)
                    {
                        continue;
                    }

                    // Only peer capable devices can map each others memory
                    ThrowIfFailed(cuDeviceCanAccessPeer(&capable, cuDevice, dev));
                    if (!capable)
                    {
                        continue;
                    }

                    // The device needs to support virtual address management for the required apis to work
                    ThrowIfFailed(cuDeviceGetAttribute(&attributeVal, CU_DEVICE_ATTRIBUTE_VIRTUAL_ADDRESS_MANAGEMENT_SUPPORTED, cuDevice));
                    if (attributeVal == 0)
                    {
                        continue;
                    }

                    backingDevices.push_back(dev);
                }
                return backingDevices;
            }

            static CUresult MallocMultiDeviceMmap(CUdeviceptr*                 dptr,
                                                  size_type*                   allocationSize,
                                                  size_type                    size,
                                                  const std::vector<CUdevice>& residentDevices,
                                                  const std::vector<CUdevice>& mappingDevices,
                                                  size_type                    align)
            {
                CUresult  status          = CUDA_SUCCESS;
                size_type min_granularity = 0;
                size_type stripeSize;

                // Setup the properties common for all the chunks
                // The allocations will be device pinned memory.
                // This property structure describes the physical location where the memory will be allocated via cuMemCreate allong with additional properties
                // In this case, the allocation will be pinnded device memory local to a given device.
                CUmemAllocationProp prop = {};
                prop.type                = CU_MEM_ALLOCATION_TYPE_PINNED;
                prop.location.type       = CU_MEM_LOCATION_TYPE_DEVICE;

                // Get the minimum granularity needed for the resident devices
                // (the max of the minimum granularity of each participating device)
                for (int idx = 0; idx < residentDevices.size(); idx++)
                {
                    size_type granularity = 0;

                    // get the minnimum granularity for residentDevices[idx]
                    prop.location.id = residentDevices[idx];
                    status           = cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED);
                    if (status != CUDA_SUCCESS)
                    {
                        goto done;
                    }
                    if (min_granularity < granularity)
                    {
                        min_granularity = granularity;
                    }
                }

                // Get the minimum granularity needed for the accessing devices
                // (the max of the minimum granularity of each participating device)
                for (size_type idx = 0; idx < mappingDevices.size(); idx++)
                {
                    size_type granularity = 0;

                    // get the minnimum granularity for mappingDevices[idx]
                    prop.location.id = mappingDevices[idx];
                    status           = cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED);
                    if (status != CUDA_SUCCESS)
                    {
                        goto done;
                    }
                    if (min_granularity < granularity)
                    {
                        min_granularity = granularity;
                    }
                }

                // Round up the size such that we can evenly split it into a stripe size tha meets the granularity requirements
                // Essentially size = N * residentDevices.size() * min_granularity is the requirement,
                // since each piece of the allocation will be stripeSize = N * min_granularity
                // and the min_granularity requirement applies to each stripeSize piece of the allocation.
                size       = round_up(size, residentDevices.size() * min_granularity);
                stripeSize = size / residentDevices.size();

                // Return the rounded up size to the caller for use in the free
                if (allocationSize)
                {
                    *allocationSize = size;
                }

                // Reserve the required contiguous VA space for the allocations
                status = cuMemAddressReserve(dptr, size, align, 0, 0);
                if (status != CUDA_SUCCESS)
                {
                    goto done;
                }

                // Create and map the backings on each gpu
                // note: reusing CUmemAllocationProp prop from earlier with prop.type & prop.location.type already specified.
                for (size_type idx = 0; idx < residentDevices.size(); idx++)
                {
                    CUresult status2 = CUDA_SUCCESS;

                    // Set the location for this chunk to this device
                    prop.location.id = residentDevices[idx];

                    // Create the allocation as a pinned allocation on this device
                    CUmemGenericAllocationHandle allocationHandle;
                    status = cuMemCreate(&allocationHandle, stripeSize, &prop, 0);
                    if (status != CUDA_SUCCESS)
                    {
                        goto done;
                    }

                    // Assign the chunk to the appropriate VA range and release the handle.
                    // After mapping the memory, it can be referenced by virtual address.
                    // Since we do not need to make any other mappings of this memory or export it,
                    // we no longer need and can release the allocationHandle.
                    // The allocation will be kept live until it is unmapped.
                    status = cuMemMap(*dptr + (stripeSize * idx), stripeSize, 0, allocationHandle, 0);

                    // the handle needs to be released even if the mapping failed.
                    status2 = cuMemRelease(allocationHandle);
                    if (status == CUDA_SUCCESS)
                    {
                        // cuMemRelease should not have failed here
                        // as the handle was just allocated successfully
                        // however return an error if it does.
                        status = status2;
                    }

                    // Cleanup in case of any mapping failures.
                    if (status != CUDA_SUCCESS)
                    {
                        goto done;
                    }
                }

                {
                    // Each accessDescriptor will describe the mapping requirement for a single device
                    std::vector<CUmemAccessDesc> accessDescriptors;
                    accessDescriptors.resize(mappingDevices.size());

                    // Prepare the access descriptor array indicating where and how the backings should be visible.
                    for (size_type idx = 0; idx < mappingDevices.size(); idx++)
                    {
                        // Specify which device we are adding mappings for.
                        accessDescriptors[idx].location.type = CU_MEM_LOCATION_TYPE_DEVICE;
                        accessDescriptors[idx].location.id   = mappingDevices[idx];

                        // Specify both read and write access.
                        accessDescriptors[idx].flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
                    }

                    // Apply the access descriptors to the whole VA range.
                    status = cuMemSetAccess(*dptr, size, &accessDescriptors[0], accessDescriptors.size());
                    if (status != CUDA_SUCCESS)
                    {
                        goto done;
                    }
                }

            done:
                if (status != CUDA_SUCCESS)
                {
                    if (*dptr)
                    {
                        simpleFreeMultiDeviceMmap(*dptr, size);
                    }
                }

                return status;
            }

            static CUresult FreeMultiDeviceMmap(CUdeviceptr dptr, size_type size)
            {
                CUresult status = CUDA_SUCCESS;

                // Unmap the mapped virtual memory region
                // Since the handles to the mapped backing stores have already been released
                // by cuMemRelease, and these are the only/last mappings referencing them,
                // The backing stores will be freed.
                // Since the memory has been unmapped after this call, accessing the specified
                // va range will result in a fault (unitll it is remapped).
                status = cuMemUnmap(dptr, size);
                if (status != CUDA_SUCCESS)
                {
                    return status;
                }
                // Free the virtual address region.  This allows the virtual address region
                // to be reused by future cuMemAddressReserve calls.  This also allows the
                // virtual address region to be used by other allocation made through
                // opperating system calls like malloc & mmap.
                status = cuMemAddressFree(dptr, size);
                if (status != CUDA_SUCCESS)
                {
                    return status;
                }

                return status;
            }
        };

#endif
    }
}

#define KOKKOS_EXTENSIONS

#include <runtime.Kokkos/Extensions/Atomics.hpp>
#include <runtime.Kokkos/Extensions/IndexOf.hpp>
#include <runtime.Kokkos/Extensions/VectorOps.hpp>
#include <runtime.Kokkos/Extensions/MatrixOps.hpp>
#include <runtime.Kokkos/Extensions/SparseOps.hpp>
#include <runtime.Kokkos/Extensions/TensorOps.hpp>
//#include <runtime.Kokkos/Extensions/Solvers.hpp>

#include <runtime.Kokkos/Extensions/Linq.hpp>

#undef KOKKOS_EXTENSIONS

namespace Kokkos
{
    namespace Extension
    {
        using namespace Kokkos::Extension::VectorOperators;
        using namespace Kokkos::Extension::MatrixOperators;
    }
}

namespace Kokkos
{
    namespace Extension
    {
        template<Integer OrdinalType, class ExecutionSpace>
        using Policy = Kokkos::RangePolicy<ExecutionSpace, Kokkos::Schedule<Kokkos::Static>, Kokkos::IndexType<OrdinalType>>;

        template<Integer OrdinalType, class ExecutionSpace>
        __forceinline__ Policy<OrdinalType, ExecutionSpace> policy(OrdinalType n)
        {
            return Policy<OrdinalType, ExecutionSpace>(0, n);
        }
    }
}

// template<typename DataType, class TFunction>
// KOKKOS_INLINE_FUNCTION static DataType fold(DataType arr[], const int len, TFunction binop, DataType initialValue = DataType(0))
//{
//    DataType ans = initialValue;
//
//    for (int i = 0; i < len; ++i)
//    {
//        ans = binop(ans, arr[i]);
//    }
//
//    return ans;
//}
//
// template<typename T>
// struct comparator
//{
//    KOKKOS_INLINE_FUNCTION constexpr int operator()(const T& a, const T& b) const
//    {
//        return (a > b) - (a < b);
//    }
//
//    KOKKOS_INLINE_FUNCTION constexpr bool less(const T& a, const T& b) const
//    {
//        return a < b;
//    }
//
//    KOKKOS_INLINE_FUNCTION constexpr bool equal(const T& a, const T& b) const
//    {
//        return a == b;
//    }
//};
//
// struct search_strategy
//{
//    search_strategy() = delete;
//    ~search_strategy() = delete;
//};
//
// struct linear_search final : public search_strategy
//{
//
//    linear_search() = delete;
//
// public:
//    template<typename Key, typename Iter, typename Comp>
//    KOKKOS_INLINE_FUNCTION static constexpr Iter Execute(const Key& k, Iter a, Iter b, Comp& comp)
//    {
//        return lower_bound(k, a, b, comp);
//    }
//
//    template<typename Key, typename Iter, typename Comp>
//    KOKKOS_INLINE_FUNCTION static constexpr Iter LowerBound(const Key& k, Iter a, Iter b, Comp& comp)
//    {
//        auto c = a;
//        while (c < b)
//        {
//            auto r = comp(*c, k);
//            if (r >= 0)
//            {
//                return c;
//            }
//            ++c;
//        }
//        return b;
//    }
//
//    template<typename Key, typename Iter, typename Comp>
//    KOKKOS_INLINE_FUNCTION static constexpr Iter UpperBound(const Key& k, Iter a, Iter b, Comp& comp)
//    {
//        auto c = a;
//        while (c < b)
//        {
//            if (comp(*c, k) > 0)
//            {
//                return c;
//            }
//            ++c;
//        }
//        return b;
//    }
//};
//
// struct binary_search final : public search_strategy
//{
//    binary_search() = delete;
//
//    template<typename Key, typename Iter, typename Comp>
//    KOKKOS_INLINE_FUNCTION static constexpr Iter Execute(const Key& k, Iter a, Iter b, Comp& comp)
//    {
//        Iter c;
//        auto count = b - a;
//        while (count > 0)
//        {
//            auto step = count >> 1;
//            c         = a + step;
//            auto r    = comp(*c, k);
//            if (r == 0)
//            {
//                return c;
//            }
//            if (r < 0)
//            {
//                a = ++c;
//                count -= step + 1;
//            }
//            else
//            {
//                count = step;
//            }
//        }
//        return a;
//    }
//
//    template<typename Key, typename Iter, typename Comp>
//    KOKKOS_INLINE_FUNCTION static constexpr Iter LowerBound(const Key& k, Iter a, Iter b, Comp& comp)
//    {
//        Iter c;
//        auto count = b - a;
//        while (count > 0)
//        {
//            auto step = count >> 1;
//            c         = a + step;
//            if (comp(*c, k) < 0)
//            {
//                a = ++c;
//                count -= step + 1;
//            }
//            else
//            {
//                count = step;
//            }
//        }
//        return a;
//    }
//
//    template<typename Key, typename Iter, typename Comp>
//    KOKKOS_INLINE_FUNCTION static constexpr Iter UpperBound(const Key& k, Iter a, Iter b, Comp& comp)
//    {
//        Iter c;
//        auto count = b - a;
//        while (count > 0)
//        {
//            auto step = count >> 1;
//            c         = a + step;
//            if (comp(k, *c) >= 0)
//            {
//                a = ++c;
//                count -= step + 1;
//            }
//            else
//            {
//                count = step;
//            }
//        }
//        return a;
//    }
//};
//
// template<typename S>
// struct strategy_selection
//{
//    using type = S;
//};
//
// struct linear : public strategy_selection<linear_search>
//{
//};
// struct binary : public strategy_selection<binary_search>
//{
//};
//
//// by default every key utilizes binary search
// template<typename Key>
// struct default_strategy : public binary
//{
//};
//
// template<>
// struct default_strategy<int> : public linear
//{
//};
//
// template<typename... Ts>
// struct default_strategy<System::Tuple<Ts...>> : public linear
//{
//};
//
///**
// * The default non-updater
// */
// template<typename T>
// struct updater
//{
//    void update(T& /* old_t */, const T& /* new_t */) {}
//};
//
// template<typename Key,
//         typename Comparator,
//         typename Allocator, // is ignored so far - TODO: add support
//         unsigned blockSize,
//         typename SearchStrategy,
//         bool isSet,
//         typename WeakComparator = Comparator,
//         typename Updater        = updater<Key>>
// class btree
//{
// public:
//    class iterator;
//    using const_iterator = iterator;
//
//    using key_type     = Key;
//    using element_type = Key;
//    using chunk        = range<iterator>;
//
// protected:
//    /* ------------- static utilities ----------------- */
//
//    const static SearchStrategy search;
//
//    /* ---------- comparison utilities ---------------- */
//
//    mutable Comparator comp;
//
//    bool less(const Key& a, const Key& b) const
//    {
//        return comp.less(a, b);
//    }
//
//    bool equal(const Key& a, const Key& b) const
//    {
//        return comp.equal(a, b);
//    }
//
//    mutable WeakComparator weak_comp;
//
//    bool weak_less(const Key& a, const Key& b) const
//    {
//        return weak_comp.less(a, b);
//    }
//
//    bool weak_equal(const Key& a, const Key& b) const
//    {
//        return weak_comp.equal(a, b);
//    }
//
//    /* -------------- updater utilities ------------- */
//
//    mutable Updater upd;
//    void            update(Key& old_k, const Key& new_k)
//    {
//        upd.update(old_k, new_k);
//    }
//
//    /* -------------- the node type ----------------- */
//
//    using size_type        = std::size_type;
//    using field_index_type = uint8_t;
//    using lock_type        = OptimisticReadWriteLock;
//
//    struct node;
//
//    /**
//     * The base type of all node types containing essential
//     * book-keeping information.
//     */
//    struct base
//    {
//
//        // the parent node
//        node* volatile parent;
//
//        // a lock for synchronizing parallel operations on this node
//        lock_type lock;
//
//        // the number of keys in this node
//        volatile size_type numElements;
//
//        // the position in the parent node
//        volatile field_index_type position;
//
//
//        // a flag indicating whether this is a inner node or not
//        const bool inner;
//
//        /**
//         * A simple constructor for nodes
//         */
//        base(bool inner) : parent(nullptr), numElements(0), position(0), inner(inner) {}
//
//        bool isLeaf() const
//        {
//            return !inner;
//        }
//
//        bool isInner() const
//        {
//            return inner;
//        }
//
//        node* getParent() const
//        {
//            return parent;
//        }
//
//        field_index_type getPositionInParent() const
//        {
//            return position;
//        }
//
//        size_type getNumElements() const
//        {
//            return numElements;
//        }
//    };
//
//    struct inner_node;
//
//    /**
//     * The actual, generic node implementation covering the operations
//     * for both, inner and leaf nodes.
//     */
//    struct node : public base
//    {
//        /**
//         * The number of keys/node desired by the user.
//         */
//        static constexpr size_type desiredNumKeys = ((blockSize > sizeof(base)) ? blockSize - sizeof(base) : 0) / sizeof(Key);
//
//        /**
//         * The actual number of keys/node corrected by functional requirements.
//         */
//        static constexpr size_type maxKeys = (desiredNumKeys > 3) ? desiredNumKeys : 3;
//
//        // the keys stored in this node
//        Key keys[maxKeys];
//
//        // a simple constructor
//        node(bool inner) : base(inner) {}
//
//        /**
//         * A deep-copy operation creating a clone of this node.
//         */
//        node* clone() const
//        {
//            // create a clone of this node
//            node* res = (this->isInner()) ? static_cast<node*>(new inner_node()) : static_cast<node*>(new leaf_node());
//
//            // copy basic fields
//            res->position    = this->position;
//            res->numElements = this->numElements;
//
//            for (size_type i = 0; i < this->numElements; ++i)
//            {
//                res->keys[i] = this->keys[i];
//            }
//
//            // if this is a leaf we are done
//            if (this->isLeaf())
//            {
//                return res;
//            }
//
//            // copy child nodes recursively
//            auto* ires = (inner_node*)res;
//            for (size_type i = 0; i <= this->numElements; ++i)
//            {
//                ires->children[i]         = this->getChild(i)->clone();
//                ires->children[i]->parent = res;
//            }
//
//            // that's it
//            return res;
//        }
//
//        /**
//         * A utility function providing a reference to this node as
//         * an inner node.
//         */
//        inner_node& asInnerNode()
//        {
//            assert(this->inner && "Invalid cast!");
//            return *static_cast<inner_node*>(this);
//        }
//
//        /**
//         * A utility function providing a reference to this node as
//         * a const inner node.
//         */
//        const inner_node& asInnerNode() const
//        {
//            assert(this->inner && "Invalid cast!");
//            return *static_cast<const inner_node*>(this);
//        }
//
//        /**
//         * Computes the number of nested levels of the tree rooted
//         * by this node.
//         */
//        size_type getDepth() const
//        {
//            if (this->isLeaf())
//            {
//                return 1;
//            }
//            return getChild(0)->getDepth() + 1;
//        }
//
//        /**
//         * Counts the number of nodes contained in the sub-tree rooted
//         * by this node.
//         */
//        size_type countNodes() const
//        {
//            if (this->isLeaf())
//            {
//                return 1;
//            }
//            size_type sum = 1;
//            for (unsigned i = 0; i <= this->numElements; ++i)
//            {
//                sum += getChild(i)->countNodes();
//            }
//            return sum;
//        }
//
//        /**
//         * Counts the number of entries contained in the sub-tree rooted
//         * by this node.
//         */
//        size_type countEntries() const
//        {
//            if (this->isLeaf())
//            {
//                return this->numElements;
//            }
//            size_type sum = this->numElements;
//            for (unsigned i = 0; i <= this->numElements; ++i)
//            {
//                sum += getChild(i)->countEntries();
//            }
//            return sum;
//        }
//
//        /**
//         * Determines the amount of memory used by the sub-tree rooted
//         * by this node.
//         */
//        size_type getMemoryUsage() const
//        {
//            if (this->isLeaf())
//            {
//                return sizeof(leaf_node);
//            }
//            size_type res = sizeof(inner_node);
//            for (unsigned i = 0; i <= this->numElements; ++i)
//            {
//                res += getChild(i)->getMemoryUsage();
//            }
//            return res;
//        }
//
//        /**
//         * Obtains a pointer to the array of child-pointers
//         * of this node -- if it is an inner node.
//         */
//        node** getChildren()
//        {
//            return asInnerNode().children;
//        }
//
//        /**
//         * Obtains a pointer to the array of const child-pointers
//         * of this node -- if it is an inner node.
//         */
//        node* const* getChildren() const
//        {
//            return asInnerNode().children;
//        }
//
//        /**
//         * Obtains a reference to the child of the given index.
//         */
//        node* getChild(size_type s) const
//        {
//            return asInnerNode().children[s];
//        }
//
//        /**
//         * Checks whether this node is empty -- can happen due to biased insertion.
//         */
//        bool isEmpty() const
//        {
//            return this->numElements == 0;
//        }
//
//        /**
//         * Checks whether this node is full.
//         */
//        bool isFull() const
//        {
//            return this->numElements == maxKeys;
//        }
//
//        /**
//         * Obtains the point at which full nodes should be split.
//         * Conventional b-trees always split in half. However, in cases
//         * where in-order insertions are frequent, a split assigning
//         * larger portions to the right fragment provide higher performance
//         * and a better node-filling rate.
//         */
//        int getSplitPoint(int /*unused*/)
//        {
//            return static_cast<int>(std::min(3 * maxKeys / 4, maxKeys - 2));
//        }
//
//        /**
//         * Splits this node.
//         *
//         * @param root .. a pointer to the root-pointer of the enclosing b-tree
//         *                 (might have to be updated if the root-node needs to be split)
//         * @param idx  .. the position of the insert causing the split
//         */
//
//        void split(node** root, lock_type& root_lock, int idx, std::vector<node*>& locked_nodes)
//        {
//
//            assert(this->numElements == maxKeys);
//
//            // get middle element
//            int split_point = getSplitPoint(idx);
//
//            // create a new sibling node
//            node* sibling = (this->inner) ? static_cast<node*>(new inner_node()) : static_cast<node*>(new leaf_node());
//
//
//            // lock sibling
//            sibling->lock.start_write();
//            locked_nodes.push_back(sibling);
//
//            // move data over to the new node
//            for (unsigned i = split_point + 1, j = 0; i < maxKeys; ++i, ++j)
//            {
//                sibling->keys[j] = keys[i];
//            }
//
//            // move child pointers
//            if (this->inner)
//            {
//                // move pointers to sibling
//                auto* other = static_cast<inner_node*>(sibling);
//                for (unsigned i = split_point + 1, j = 0; i <= maxKeys; ++i, ++j)
//                {
//                    other->children[j]           = getChildren()[i];
//                    other->children[j]->parent   = other;
//                    other->children[j]->position = static_cast<field_index_type>(j);
//                }
//            }
//
//            // update number of elements
//            this->numElements    = split_point;
//            sibling->numElements = maxKeys - split_point - 1;
//
//            // update parent
//
//            grow_parent(root, root_lock, sibling, locked_nodes);
//
//        }
//
//        /**
//         * Moves keys from this node to one of its siblings or splits
//         * this node to make some space for the insertion of an element at
//         * position idx.
//         *
//         * Returns the number of elements moved to the left side, 0 in case
//         * of a split. The number of moved elements will be <= the given idx.
//         *
//         * @param root .. the root node of the b-tree being part of
//         * @param idx  .. the position of the insert triggering this operation
//         */
//
//        int rebalance_or_split(node** root, lock_type& root_lock, int idx, std::vector<node*>& locked_nodes)
//        {
//
//
//            // this node is full ... and needs some space
//            assert(this->numElements == maxKeys);
//
//            // get snap-shot of parent
//            auto parent = this->parent;
//            auto pos    = this->position;
//
//            // Option A) re-balance data
//            if (parent && pos > 0)
//            {
//                node* left = parent->getChild(pos - 1);
//
//
//                // lock access to left sibling
//                if (!left->lock.try_start_write())
//                {
//                    // left node is currently updated => skip balancing and split
//                    split(root, root_lock, idx, locked_nodes);
//                    return 0;
//                }
//
//
//                // compute number of elements to be movable to left
//                //    space available in left vs. insertion index
//                size_type num = static_cast<size_type>(std::min<int>(static_cast<int>(maxKeys - left->numElements), idx));
//
//                // if there are elements to move ..
//                if (num > 0)
//                {
//                    Key* splitter = &(parent->keys[this->position - 1]);
//
//                    // .. move keys to left node
//                    left->keys[left->numElements] = *splitter;
//                    for (size_type i = 0; i < num - 1; ++i)
//                    {
//                        left->keys[left->numElements + 1 + i] = keys[i];
//                    }
//                    *splitter = keys[num - 1];
//
//                    // shift keys in this node to the left
//                    for (size_type i = 0; i < this->numElements - num; ++i)
//                    {
//                        keys[i] = keys[i + num];
//                    }
//
//                    // .. and children if necessary
//                    if (this->isInner())
//                    {
//                        auto* ileft  = static_cast<inner_node*>(left);
//                        auto* iright = static_cast<inner_node*>(this);
//
//                        // move children
//                        for (field_index_type i = 0; i < num; ++i)
//                        {
//                            ileft->children[left->numElements + i + 1] = iright->children[i];
//                        }
//
//                        // update moved children
//                        for (size_type i = 0; i < num; ++i)
//                        {
//                            iright->children[i]->parent   = ileft;
//                            iright->children[i]->position = static_cast<field_index_type>(left->numElements + i) + 1;
//                        }
//
//                        // shift child-pointer to the left
//                        for (size_type i = 0; i < this->numElements - num + 1; ++i)
//                        {
//                            iright->children[i] = iright->children[i + num];
//                        }
//
//                        // update position of children
//                        for (size_type i = 0; i < this->numElements - num + 1; ++i)
//                        {
//                            iright->children[i]->position = static_cast<field_index_type>(i);
//                        }
//                    }
//
//                    // update node sizes
//                    left->numElements += num;
//                    this->numElements -= num;
//
//
//                    left->lock.end_write();
//
//                    // done
//                    return static_cast<int>(num);
//                }
//
//
//                left->lock.abort_write();
//
//            }
//
//            // Option B) split node
//
//            split(root, root_lock, idx, locked_nodes);
//
//            return 0; // = no re-balancing
//        }
//
//    private:
//        /**
//         * Inserts a new sibling into the parent of this node utilizing
//         * the last key of this node as a separation key. (for internal
//         * use only)
//         *
//         * @param root .. a pointer to the root-pointer of the containing tree
//         * @param sibling .. the new right-sibling to be add to the parent node
//         */
//
//        void grow_parent(node** root, lock_type& root_lock, node* sibling, std::vector<node*>& locked_nodes)
//        {
//            assert(this->lock.is_write_locked());
//            assert(!this->parent || this->parent->lock.is_write_locked());
//            assert((this->parent != nullptr) || root_lock.is_write_locked());
//            assert(this->isLeaf() || souffle::contains(locked_nodes, this));
//            assert(!this->parent || souffle::contains(locked_nodes, const_cast<node*>(this->parent)));
//
//
//            if (this->parent == nullptr)
//            {
//                assert(*root == this);
//
//                // create a new root node
//                auto* new_root        = new inner_node();
//                new_root->numElements = 1;
//                new_root->keys[0]     = keys[this->numElements];
//
//                new_root->children[0] = this;
//                new_root->children[1] = sibling;
//
//                // link this and the sibling node to new root
//                this->parent      = new_root;
//                sibling->parent   = new_root;
//                sibling->position = 1;
//
//                // switch root node
//                *root = new_root;
//            }
//            else
//            {
//                // insert new element in parent element
//                auto parent = this->parent;
//                auto pos    = this->position;
//
//
//                parent->insert_inner(root, root_lock, pos, this, keys[this->numElements], sibling, locked_nodes);
//
//            }
//        }
//
//        /**
//         * Inserts a new element into an inner node (for internal use only).
//         *
//         * @param root .. a pointer to the root-pointer of the containing tree
//         * @param pos  .. the position to insert the new key
//         * @param key  .. the key to insert
//         * @param newNode .. the new right-child of the inserted key
//         */
//
//        void insert_inner(node** root, lock_type& root_lock, unsigned pos, node* predecessor, const Key& key, node* newNode, std::vector<node*>& locked_nodes)
//        {
//            assert(this->lock.is_write_locked());
//            assert(souffle::contains(locked_nodes, this));
//
//
//            // check capacity
//            if (this->numElements >= maxKeys)
//            {
//
//
//                // split this node
//
//                pos -= rebalance_or_split(root, root_lock, pos, locked_nodes);
//
//                // complete insertion within new sibling if necessary
//                if (pos > this->numElements)
//                {
//                    // correct position
//                    pos = pos - static_cast<unsigned int>(this->numElements) - 1;
//
//                    // get new sibling
//                    auto other = this->parent->getChild(this->position + 1);
//
//
//                    // make sure other side is write locked
//                    assert(other->lock.is_write_locked());
//                    assert(souffle::contains(locked_nodes, other));
//
//                    // search for new position (since other may have been altered in the meanwhile)
//                    size_type i = 0;
//                    for (; i <= other->numElements; ++i)
//                    {
//                        if (other->getChild(i) == predecessor)
//                        {
//                            break;
//                        }
//                    }
//
//                    pos = (i > other->numElements) ? 0 : i;
//                    other->insert_inner(root, root_lock, pos, predecessor, key, newNode, locked_nodes);
//
//                    return;
//                }
//            }
//
//            // move bigger keys one forward
//            for (int i = static_cast<int>(this->numElements) - 1; i >= (int)pos; --i)
//            {
//                keys[i + 1]          = keys[i];
//                getChildren()[i + 2] = getChildren()[i + 1];
//                ++getChildren()[i + 2]->position;
//            }
//
//            // ensure proper position
//            assert(getChild(pos) == predecessor);
//
//            // insert new element
//            keys[pos]              = key;
//            getChildren()[pos + 1] = newNode;
//            newNode->parent        = this;
//            newNode->position      = static_cast<field_index_type>(pos) + 1;
//            ++this->numElements;
//        }
//
//    public:
//        /**
//         * Prints a textual representation of this tree to the given output stream.
//         * This feature is mainly intended for debugging and tuning purposes.
//         *
//         * @see btree::printTree
//         */
//        void printTree(std::ostream& out, const std::string& prefix) const
//        {
//            // print the header
//            out << prefix << "@" << this << "[" << ((int)(this->position)) << "] - " << (this->inner ? "i" : "") << "node : " << this->numElements << "/" << maxKeys << " [";
//
//            // print the keys
//            for (unsigned i = 0; i < this->numElements; ++i)
//            {
//                out << keys[i];
//                if (i != this->numElements - 1)
//                {
//                    out << ",";
//                }
//            }
//            out << "]";
//
//            // print references to children
//            if (this->inner)
//            {
//                out << " - [";
//                for (unsigned i = 0; i <= this->numElements; ++i)
//                {
//                    out << getChildren()[i];
//                    if (i != this->numElements)
//                    {
//                        out << ",";
//                    }
//                }
//                out << "]";
//            }
//
//
//            // print the lock state
//            if (this->lock.is_write_locked())
//            {
//                std::cout << " locked";
//            }
//
//
//            out << std::endl;
//
//            // print the children recursively
//            if (this->inner)
//            {
//                for (unsigned i = 0; i < this->numElements + 1; ++i)
//                {
//                    static_cast<const inner_node*>(this)->children[i]->printTree(out, prefix + "    ");
//                }
//            }
//        }
//
//        /**
//         * A function decomposing the sub-tree rooted by this node into approximately equally
//         * sized chunks. To minimize computational overhead, no strict load balance nor limit
//         * on the number of actual chunks is given.
//         *
//         * @see btree::getChunks()
//         *
//         * @param res   .. the list of chunks to be extended
//         * @param num   .. the number of chunks to be produced
//         * @param begin .. the iterator to start the first chunk with
//         * @param end   .. the iterator to end the last chunk with
//         * @return the handed in list of chunks extended by generated chunks
//         */
//        std::vector<chunk>& collectChunks(std::vector<chunk>& res, size_type num, const iterator& begin, const iterator& end) const
//        {
//            assert(num > 0);
//
//            // special case: this node is empty
//            if (isEmpty())
//            {
//                if (begin != end)
//                {
//                    res.push_back(chunk(begin, end));
//                }
//                return res;
//            }
//
//            // special case: a single chunk is requested
//            if (num == 1)
//            {
//                res.push_back(chunk(begin, end));
//                return res;
//            }
//
//            // cut-off
//            if (this->isLeaf() || num < (this->numElements + 1))
//            {
//                auto step = this->numElements / num;
//                if (step == 0)
//                {
//                    step = 1;
//                }
//
//                size_type i = 0;
//
//                // the first chunk starts at the begin
//                res.push_back(chunk(begin, iterator(this, static_cast<field_index_type>(step) - 1)));
//
//                // split up the main part
//                for (i = step - 1; i < this->numElements - step; i += step)
//                {
//                    res.push_back(chunk(iterator(this, static_cast<field_index_type>(i)), iterator(this, static_cast<field_index_type>(i + step))));
//                }
//
//                // the last chunk runs to the end
//                res.push_back(chunk(iterator(this, static_cast<field_index_type>(i)), end));
//
//                // done
//                return res;
//            }
//
//            // else: collect chunks of sub-set elements
//
//            auto part = num / (this->numElements + 1);
//            assert(part > 0);
//            getChild(0)->collectChunks(res, part, begin, iterator(this, 0));
//            for (size_type i = 1; i < this->numElements; ++i)
//            {
//                getChild(i)->collectChunks(res, part, iterator(this, static_cast<field_index_type>(i - 1)), iterator(this, static_cast<field_index_type>(i)));
//            }
//            getChild(this->numElements)->collectChunks(res, num - (part * this->numElements), iterator(this, static_cast<field_index_type>(this->numElements) - 1), end);
//
//            // done
//            return res;
//        }
//
//        /**
//         * A function to verify the consistency of this node.
//         *
//         * @param root ... a reference to the root of the enclosing tree.
//         * @return true if valid, false otherwise
//         */
//        template<typename Comp>
//        bool check(Comp& comp, const node* root) const
//        {
//            bool valid = true;
//
//            // check fill-state
//            if (this->numElements > maxKeys)
//            {
//                std::cout << "Node with " << this->numElements << "/" << maxKeys << " encountered!\n";
//                valid = false;
//            }
//
//            // check root state
//            if (root == this)
//            {
//                if (this->parent != nullptr)
//                {
//                    std::cout << "Root not properly linked!\n";
//                    valid = false;
//                }
//            }
//            else
//            {
//                // check parent relation
//                if (!this->parent)
//                {
//                    std::cout << "Invalid null-parent!\n";
//                    valid = false;
//                }
//                else
//                {
//                    if (this->parent->getChildren()[this->position] != this)
//                    {
//                        std::cout << "Parent reference invalid!\n";
//                        std::cout << "   Node:     " << this << std::endl;
//                        std::cout << "   Parent:   " << this->parent << std::endl;
//                        std::cout << "   Position: " << ((int)this->position) << std::endl;
//                        valid = false;
//                    }
//
//                    // check parent key
//                    if (valid && this->position != 0 && !(comp(this->parent->keys[this->position - 1], keys[0]) < ((isSet) ? 0 : 1)))
//                    {
//                        std::cout << "Left parent key not lower bound!\n";
//                        std::cout << "   Node:     " << this << std::endl;
//                        std::cout << "   Parent:   " << this->parent << std::endl;
//                        std::cout << "   Position: " << ((int)this->position) << std::endl;
//                        std::cout << "   Key:   " << (this->parent->keys[this->position]) << std::endl;
//                        std::cout << "   Lower: " << (keys[0]) << std::endl;
//                        valid = false;
//                    }
//
//                    // check parent key
//                    if (valid && this->position != this->parent->numElements && !(comp(keys[this->numElements - 1], this->parent->keys[this->position]) < ((isSet) ? 0 : 1)))
//                    {
//                        std::cout << "Right parent key not lower bound!\n";
//                        std::cout << "   Node:     " << this << std::endl;
//                        std::cout << "   Parent:   " << this->parent << std::endl;
//                        std::cout << "   Position: " << ((int)this->position) << std::endl;
//                        std::cout << "   Key:   " << (this->parent->keys[this->position]) << std::endl;
//                        std::cout << "   Upper: " << (keys[0]) << std::endl;
//                        valid = false;
//                    }
//                }
//            }
//
//            // check element order
//            if (this->numElements > 0)
//            {
//                for (unsigned i = 0; i < this->numElements - 1; ++i)
//                {
//                    if (valid && !(comp(keys[i], keys[i + 1]) < ((isSet) ? 0 : 1)))
//                    {
//                        std::cout << "Element order invalid!\n";
//                        std::cout << " @" << this << " key " << i << " is " << keys[i] << " vs " << keys[i + 1] << std::endl;
//                        valid = false;
//                    }
//                }
//            }
//
//            // check state of sub-nodes
//            if (this->inner)
//            {
//                for (unsigned i = 0; i <= this->numElements; ++i)
//                {
//                    valid &= getChildren()[i]->check(comp, root);
//                }
//            }
//
//            return valid;
//        }
//    }; // namespace detail
//
//    /**
//     * The data type representing inner nodes of the b-tree. It extends
//     * the generic implementation of a node by the storage locations
//     * of child pointers.
//     */
//    struct inner_node : public node
//    {
//        // references to child nodes owned by this node
//        node* children[node::maxKeys + 1];
//
//        // a simple default constructor initializing member fields
//        inner_node() : node(true) {}
//
//        // clear up child nodes recursively
//        ~inner_node()
//        {
//            for (unsigned i = 0; i <= this->numElements; ++i)
//            {
//                if (children[i] != nullptr)
//                {
//                    if (children[i]->isLeaf())
//                    {
//                        delete static_cast<leaf_node*>(children[i]);
//                    }
//                    else
//                    {
//                        delete static_cast<inner_node*>(children[i]);
//                    }
//                }
//            }
//        }
//    };
//
//    /**
//     * The data type representing leaf nodes of the b-tree. It does not
//     * add any capabilities to the generic node type.
//     */
//    struct leaf_node : public node
//    {
//        // a simple default constructor initializing member fields
//        leaf_node() : node(false) {}
//    };
//
//    // ------------------- iterators ------------------------
//
// public:
//    /**
//     * The iterator type to be utilized for scanning through btree instances.
//     */
//    class iterator
//    {
//        // a pointer to the node currently referred to
//        node const* cur;
//
//        // the index of the element currently addressed within the referenced node
//        field_index_type pos = 0;
//
//    public:
//        using iterator_category = std::forward_iterator_tag;
//        using value_type        = Key;
//        using difference_type   = ptrdiff_t;
//        using pointer           = value_type*;
//        using reference         = value_type&;
//
//        // default constructor -- creating an end-iterator
//        iterator() : cur(nullptr) {}
//
//        // creates an iterator referencing a specific element within a given node
//        iterator(node const* cur, field_index_type pos) : cur(cur), pos(pos) {}
//
//        // a copy constructor
//        iterator(const iterator& other) : cur(other.cur), pos(other.pos) {}
//
//        // an assignment operator
//        iterator& operator=(const iterator& other)
//        {
//            cur = other.cur;
//            pos = other.pos;
//            return *this;
//        }
//
//        // the equality operator as required by the iterator concept
//        bool operator==(const iterator& other) const
//        {
//            return cur == other.cur && pos == other.pos;
//        }
//
//        // the not-equality operator as required by the iterator concept
//        bool operator!=(const iterator& other) const
//        {
//            return !(*this == other);
//        }
//
//        // the deref operator as required by the iterator concept
//        const Key& operator*() const
//        {
//            return cur->keys[pos];
//        }
//
//        // the increment operator as required by the iterator concept
//        iterator& operator++()
//        {
//            // the quick mode -- if in a leaf and there are elements left
//            if (cur->isLeaf() && ++pos < cur->getNumElements())
//            {
//                return *this;
//            }
//
//            // otherwise it is a bit more tricky
//
//            // A) currently in an inner node => go to the left-most child
//            if (cur->isInner())
//            {
//                cur = cur->getChildren()[pos + 1];
//                while (!cur->isLeaf())
//                {
//                    cur = cur->getChildren()[0];
//                }
//                pos = 0;
//
//                // nodes may be empty due to biased insertion
//                if (!cur->isEmpty())
//                {
//                    return *this;
//                }
//            }
//
//            // B) we are at the right-most element of a leaf => go to next inner node
//            assert(cur->isLeaf());
//            assert(pos == cur->getNumElements());
//
//            while (cur != nullptr && pos == cur->getNumElements())
//            {
//                pos = cur->getPositionInParent();
//                cur = cur->getParent();
//            }
//            return *this;
//        }
//
//        // prints a textual representation of this iterator to the given stream (mainly for debugging)
//        void print(std::ostream& out = std::cout) const
//        {
//            out << cur << "[" << (int)pos << "]";
//        }
//    };
//
//    /**
//     * A collection of operation hints speeding up some of the involved operations
//     * by exploiting temporal locality.
//     */
//    template<unsigned size = 1>
//    struct btree_operation_hints
//    {
//        using node_cache = LRUCache<node*, size>;
//
//        // the node where the last insertion terminated
//        node_cache last_insert;
//
//        // the node where the last find-operation terminated
//        node_cache last_find_end;
//
//        // the node where the last lower-bound operation terminated
//        node_cache last_lower_bound_end;
//
//        // the node where the last upper-bound operation terminated
//        node_cache last_upper_bound_end;
//
//        // default constructor
//        btree_operation_hints() = default;
//
//        // resets all hints (to be triggered e.g. when deleting nodes)
//        void clear()
//        {
//            last_insert.clear(nullptr);
//            last_find_end.clear(nullptr);
//            last_lower_bound_end.clear(nullptr);
//            last_upper_bound_end.clear(nullptr);
//        }
//    };
//
//    using operation_hints = btree_operation_hints<1>;
//
// protected:
//    // a pointer to the root node of this tree
//    node* volatile root;
//
//    // a lock to synchronize update operations on the root pointer
//    lock_type root_lock;
//
//
//    // a pointer to the left-most node of this tree (initial note for iteration)
//    leaf_node* leftmost;
//
//    /* -------------- operator hint statistics ----------------- */
//
//    // an aggregation of statistical values of the hint utilization
//    struct hint_statistics
//    {
//        // the counter for insertion operations
//        CacheAccessCounter inserts;
//
//        // the counter for contains operations
//        CacheAccessCounter contains;
//
//        // the counter for lower_bound operations
//        CacheAccessCounter lower_bound;
//
//        // the counter for upper_bound operations
//        CacheAccessCounter upper_bound;
//    };
//
//    // the hint statistic of this b-tree instance
//    mutable hint_statistics hint_stats;
//
// public:
//    // the maximum number of keys stored per node
//    static constexpr size_type max_keys_per_node = node::maxKeys;
//
//    // -- ctors / dtors --
//
//    // the default constructor creating an empty tree
//    btree(Comparator comp = Comparator(), WeakComparator weak_comp = WeakComparator()) :
//        comp(std::move(comp)),
//        weak_comp(std::move(weak_comp)),
//        root(nullptr),
//        leftmost(nullptr)
//    {
//    }
//
//    // a constructor creating a tree from the given iterator range
//    template<typename Iter>
//    btree(const Iter& a, const Iter& b) : root(nullptr), leftmost(nullptr)
//    {
//        insert(a, b);
//    }
//
//    // a move constructor
//    btree(btree&& other) : comp(other.comp), weak_comp(other.weak_comp), root(other.root), leftmost(other.leftmost)
//    {
//        other.root     = nullptr;
//        other.leftmost = nullptr;
//    }
//
//    // a copy constructor
//    btree(const btree& set) : comp(set.comp), weak_comp(set.weak_comp), root(nullptr), leftmost(nullptr)
//    {
//        // use assignment operator for a deep copy
//        *this = set;
//    }
//
// protected:
//    /**
//     * An internal constructor enabling the specific creation of a tree
//     * based on internal parameters.
//     */
//    btree(size_type /* size */, node* root, leaf_node* leftmost) : root(root), leftmost(leftmost) {}
//
// public:
//    // the destructor freeing all contained nodes
//    ~btree()
//    {
//        clear();
//    }
//
//    // -- mutators and observers --
//
//    // emptiness check
//    bool empty() const
//    {
//        return root == nullptr;
//    }
//
//    // determines the number of elements in this tree
//    size_type size() const
//    {
//        return (root) ? root->countEntries() : 0;
//    }
//
//    /**
//     * Inserts the given key into this tree.
//     */
//    bool insert(const Key& k)
//    {
//        operation_hints hints;
//        return insert(k, hints);
//    }
//
//    /**
//     * Inserts the given key into this tree.
//     */
//    bool insert(const Key& k, operation_hints& hints)
//    {
//
//        // special handling for inserting first element
//        while (root == nullptr)
//        {
//            // try obtaining root-lock
//            if (!root_lock.try_start_write())
//            {
//                // somebody else was faster => re-check
//                continue;
//            }
//
//            // check loop condition again
//            if (root != nullptr)
//            {
//                // somebody else was faster => normal insert
//                root_lock.end_write();
//                break;
//            }
//
//            // create new node
//            leftmost              = new leaf_node();
//            leftmost->numElements = 1;
//            leftmost->keys[0]     = k;
//            root                  = leftmost;
//
//            // operation complete => we can release the root lock
//            root_lock.end_write();
//
//            hints.last_insert.access(leftmost);
//
//            return true;
//        }
//
//        // insert using iterative implementation
//
//        node* cur = nullptr;
//
//        // test last insert hints
//        lock_type::Lease cur_lease;
//
//        auto checkHint = [&](node* last_insert) {
//            // ignore null pointer
//            if (!last_insert)
//                return false;
//            // get a read lease on indicated node
//            auto hint_lease = last_insert->lock.start_read();
//            // check whether it covers the key
//            if (!weak_covers(last_insert, k))
//                return false;
//            // and if there was no concurrent modification
//            if (!last_insert->lock.validate(hint_lease))
//                return false;
//            // use hinted location
//            cur = last_insert;
//            // and keep lease
//            cur_lease = hint_lease;
//            // we found a hit
//            return true;
//        };
//
//        if (hints.last_insert.any(checkHint))
//        {
//            // register this as a hit
//            hint_stats.inserts.addHit();
//        }
//        else
//        {
//            // register this as a miss
//            hint_stats.inserts.addMiss();
//        }
//
//        // if there is no valid hint ..
//        if (!cur)
//        {
//            do
//            {
//                // get root - access lock
//                auto root_lease = root_lock.start_read();
//
//                // start with root
//                cur = root;
//
//                // get lease of the next node to be accessed
//                cur_lease = cur->lock.start_read();
//
//                // check validity of root pointer
//                if (root_lock.end_read(root_lease))
//                {
//                    break;
//                }
//
//            } while (true);
//        }
//
//        while (true)
//        {
//            // handle inner nodes
//            if (cur->inner)
//            {
//                auto a = &(cur->keys[0]);
//                auto b = &(cur->keys[cur->numElements]);
//
//                auto pos = search.lower_bound(k, a, b, weak_comp);
//                auto idx = pos - a;
//
//                // early exit for sets
//                if (isSet && pos != b && weak_equal(*pos, k))
//                {
//                    // validate results
//                    if (!cur->lock.validate(cur_lease))
//                    {
//                        // start over again
//                        return insert(k, hints);
//                    }
//
//                    // update provenance information
//                    if (typeid(Comparator) != typeid(WeakComparator) && less(k, *pos))
//                    {
//                        if (!cur->lock.try_upgrade_to_write(cur_lease))
//                        {
//                            // start again
//                            return insert(k, hints);
//                        }
//                        update(*pos, k);
//                        cur->lock.end_write();
//                        return true;
//                    }
//
//                    // we found the element => no check of lock necessary
//                    return false;
//                }
//
//                // get next pointer
//                auto next = cur->getChild(idx);
//
//                // get lease on next level
//                auto next_lease = next->lock.start_read();
//
//                // check whether there was a write
//                if (!cur->lock.end_read(cur_lease))
//                {
//                    // start over
//                    return insert(k, hints);
//                }
//
//                // go to next
//                cur = next;
//
//                // move on lease
//                cur_lease = next_lease;
//
//                continue;
//            }
//
//            // the rest is for leaf nodes
//            assert(!cur->inner);
//
//            // -- insert node in leaf node --
//
//            auto a = &(cur->keys[0]);
//            auto b = &(cur->keys[cur->numElements]);
//
//            auto pos = search.upper_bound(k, a, b, weak_comp);
//            auto idx = pos - a;
//
//            // early exit for sets
//            if (isSet && pos != a && weak_equal(*(pos - 1), k))
//            {
//                // validate result
//                if (!cur->lock.validate(cur_lease))
//                {
//                    // start over again
//                    return insert(k, hints);
//                }
//
//                // update provenance information
//                if (typeid(Comparator) != typeid(WeakComparator) && less(k, *(pos - 1)))
//                {
//                    if (!cur->lock.try_upgrade_to_write(cur_lease))
//                    {
//                        // start again
//                        return insert(k, hints);
//                    }
//                    update(*(pos - 1), k);
//                    cur->lock.end_write();
//                    return true;
//                }
//
//                // we found the element => done
//                return false;
//            }
//
//            // upgrade to write-permission
//            if (!cur->lock.try_upgrade_to_write(cur_lease))
//            {
//                // something has changed => restart
//                hints.last_insert.access(cur);
//                return insert(k, hints);
//            }
//
//            if (cur->numElements >= node::maxKeys)
//            {
//                // -- lock parents --
//                auto               priv   = cur;
//                auto               parent = priv->parent;
//                std::vector<node*> parents;
//                do
//                {
//                    if (parent)
//                    {
//                        parent->lock.start_write();
//                        while (true)
//                        {
//                            // check whether parent is correct
//                            if (parent == priv->parent)
//                            {
//                                break;
//                            }
//                            // switch parent
//                            parent->lock.abort_write();
//                            parent = priv->parent;
//                            parent->lock.start_write();
//                        }
//                    }
//                    else
//                    {
//                        // lock root lock => since cur is root
//                        root_lock.start_write();
//                    }
//
//                    // record locked node
//                    parents.push_back(parent);
//
//                    // stop at "sphere of influence"
//                    if (!parent || !parent->isFull())
//                    {
//                        break;
//                    }
//
//                    // go one step higher
//                    priv   = parent;
//                    parent = parent->parent;
//
//                } while (true);
//
//                // split this node
//                auto old_root = root;
//                idx -= cur->rebalance_or_split(const_cast<node**>(&root), root_lock, idx, parents);
//
//                // release parent lock
//                for (auto it = parents.rbegin(); it != parents.rend(); ++it)
//                {
//                    auto parent = *it;
//
//                    // release this lock
//                    if (parent)
//                    {
//                        parent->lock.end_write();
//                    }
//                    else
//                    {
//                        if (old_root != root)
//                        {
//                            root_lock.end_write();
//                        }
//                        else
//                        {
//                            root_lock.abort_write();
//                        }
//                    }
//                }
//
//                // insert element in right fragment
//                if (((size_type)idx) > cur->numElements)
//                {
//                    // release current lock
//                    cur->lock.end_write();
//
//                    // insert in sibling
//                    return insert(k, hints);
//                }
//            }
//
//            // ok - no split necessary
//            assert(cur->numElements < node::maxKeys && "Split required!");
//
//            // move keys
//            for (int j = cur->numElements; j > idx; --j)
//            {
//                cur->keys[j] = cur->keys[j - 1];
//            }
//
//            // insert new element
//            cur->keys[idx] = k;
//            cur->numElements++;
//
//            // release lock on current node
//            cur->lock.end_write();
//
//            // remember last insertion position
//            hints.last_insert.access(cur);
//            return true;
//        }
//
//    }
//
//    /**
//     * Inserts the given range of elements into this tree.
//     */
//    template<typename Iter>
//    void insert(const Iter& a, const Iter& b)
//    {
//        // TODO: improve this beyond a naive insert
//        operation_hints hints;
//        // a naive insert so far .. seems to work fine
//        for (auto it = a; it != b; ++it)
//        {
//            // use insert with hint
//            insert(*it, hints);
//        }
//    }
//
//    // Obtains an iterator referencing the first element of the tree.
//    iterator begin() const
//    {
//        return iterator(leftmost, 0);
//    }
//
//    // Obtains an iterator referencing the position after the last element of the tree.
//    iterator end() const
//    {
//        return iterator();
//    }
//
//    /**
//     * Partitions the full range of this set into up to a given number of chunks.
//     * The chunks will cover approximately the same number of elements. Also, the
//     * number of chunks will only approximate the desired number of chunks.
//     *
//     * @param num .. the number of chunks requested
//     * @return a list of chunks partitioning this tree
//     */
//    std::vector<chunk> partition(size_type num) const
//    {
//        return getChunks(num);
//    }
//
//    std::vector<chunk> getChunks(size_type num) const
//    {
//        std::vector<chunk> res;
//        if (empty())
//        {
//            return res;
//        }
//        return root->collectChunks(res, num, begin(), end());
//    }
//
//    /**
//     * Determines whether the given element is a member of this tree.
//     */
//    bool contains(const Key& k) const
//    {
//        operation_hints hints;
//        return contains(k, hints);
//    }
//
//    /**
//     * Determines whether the given element is a member of this tree.
//     */
//    bool contains(const Key& k, operation_hints& hints) const
//    {
//        return find(k, hints) != end();
//    }
//
//    /**
//     * Locates the given key within this tree and returns an iterator
//     * referencing its position. If not found, an end-iterator will be returned.
//     */
//    iterator find(const Key& k) const
//    {
//        operation_hints hints;
//        return find(k, hints);
//    }
//
//    /**
//     * Locates the given key within this tree and returns an iterator
//     * referencing its position. If not found, an end-iterator will be returned.
//     */
//    iterator find(const Key& k, operation_hints& hints) const
//    {
//        if (empty())
//        {
//            return end();
//        }
//
//        node* cur = root;
//
//        auto checkHints = [&](node* last_find_end) {
//            if (!last_find_end)
//                return false;
//            if (!covers(last_find_end, k))
//                return false;
//            cur = last_find_end;
//            return true;
//        };
//
//        // test last location searched (temporal locality)
//        if (hints.last_find_end.any(checkHints))
//        {
//            // register it as a hit
//            hint_stats.contains.addHit();
//        }
//        else
//        {
//            // register it as a miss
//            hint_stats.contains.addMiss();
//        }
//
//        // an iterative implementation (since 2/7 faster than recursive)
//
//        while (true)
//        {
//            auto a = &(cur->keys[0]);
//            auto b = &(cur->keys[cur->numElements]);
//
//            auto pos = search(k, a, b, comp);
//
//            if (pos < b && equal(*pos, k))
//            {
//                hints.last_find_end.access(cur);
//                return iterator(cur, static_cast<field_index_type>(pos - a));
//            }
//
//            if (!cur->inner)
//            {
//                hints.last_find_end.access(cur);
//                return end();
//            }
//
//            // continue search in child node
//            cur = cur->getChild(pos - a);
//        }
//    }
//
//    /**
//     * Obtains a lower boundary for the given key -- hence an iterator referencing
//     * the smallest value that is not less the given key. If there is no such element,
//     * an end-iterator will be returned.
//     */
//    iterator lower_bound(const Key& k) const
//    {
//        operation_hints hints;
//        return lower_bound(k, hints);
//    }
//
//    /**
//     * Obtains a lower boundary for the given key -- hence an iterator referencing
//     * the smallest value that is not less the given key. If there is no such element,
//     * an end-iterator will be returned.
//     */
//    iterator lower_bound(const Key& k, operation_hints& hints) const
//    {
//        if (empty())
//        {
//            return end();
//        }
//
//        node* cur = root;
//
//        auto checkHints = [&](node* last_lower_bound_end) {
//            if (!last_lower_bound_end)
//                return false;
//            if (!covers(last_lower_bound_end, k))
//                return false;
//            cur = last_lower_bound_end;
//            return true;
//        };
//
//        // test last searched node
//        if (hints.last_lower_bound_end.any(checkHints))
//        {
//            hint_stats.lower_bound.addHit();
//        }
//        else
//        {
//            hint_stats.lower_bound.addMiss();
//        }
//
//        iterator res = end();
//        while (true)
//        {
//            auto a = &(cur->keys[0]);
//            auto b = &(cur->keys[cur->numElements]);
//
//            auto pos = search.lower_bound(k, a, b, comp);
//            auto idx = static_cast<field_index_type>(pos - a);
//
//            if (!cur->inner)
//            {
//                hints.last_lower_bound_end.access(cur);
//                return (pos != b) ? iterator(cur, idx) : res;
//            }
//
//            if (isSet && pos != b && equal(*pos, k))
//            {
//                return iterator(cur, idx);
//            }
//
//            if (pos != b)
//            {
//                res = iterator(cur, idx);
//            }
//
//            cur = cur->getChild(idx);
//        }
//    }
//
//    /**
//     * Obtains an upper boundary for the given key -- hence an iterator referencing
//     * the first element that the given key is less than the referenced value. If
//     * there is no such element, an end-iterator will be returned.
//     */
//    iterator upper_bound(const Key& k) const
//    {
//        operation_hints hints;
//        return upper_bound(k, hints);
//    }
//
//    /**
//     * Obtains an upper boundary for the given key -- hence an iterator referencing
//     * the first element that the given key is less than the referenced value. If
//     * there is no such element, an end-iterator will be returned.
//     */
//    iterator upper_bound(const Key& k, operation_hints& hints) const
//    {
//        if (empty())
//        {
//            return end();
//        }
//
//        node* cur = root;
//
//        auto checkHints = [&](node* last_upper_bound_end) {
//            if (!last_upper_bound_end)
//                return false;
//            if (!coversUpperBound(last_upper_bound_end, k))
//                return false;
//            cur = last_upper_bound_end;
//            return true;
//        };
//
//        // test last search node
//        if (hints.last_upper_bound_end.any(checkHints))
//        {
//            hint_stats.upper_bound.addHit();
//        }
//        else
//        {
//            hint_stats.upper_bound.addMiss();
//        }
//
//        iterator res = end();
//        while (true)
//        {
//            auto a = &(cur->keys[0]);
//            auto b = &(cur->keys[cur->numElements]);
//
//            auto pos = search.upper_bound(k, a, b, comp);
//            auto idx = static_cast<field_index_type>(pos - a);
//
//            if (!cur->inner)
//            {
//                hints.last_upper_bound_end.access(cur);
//                return (pos != b) ? iterator(cur, idx) : res;
//            }
//
//            if (pos != b)
//            {
//                res = iterator(cur, idx);
//            }
//
//            cur = cur->getChild(idx);
//        }
//    }
//
//    /**
//     * Clears this tree.
//     */
//    void clear()
//    {
//        if (root != nullptr)
//        {
//            if (root->isLeaf())
//            {
//                delete static_cast<leaf_node*>(root);
//            }
//            else
//            {
//                delete static_cast<inner_node*>(root);
//            }
//        }
//        root     = nullptr;
//        leftmost = nullptr;
//    }
//
//    /**
//     * Swaps the content of this tree with the given tree. This
//     * is a much more efficient operation than creating a copy and
//     * realizing the swap utilizing assignment operations.
//     */
//    void swap(btree& other)
//    {
//        // swap the content
//        Kokkos::swap(root, other.root);
//        Kokkos::swap(leftmost, other.leftmost);
//    }
//
//    // Implementation of the assignment operation for trees.
//    btree& operator=(const btree& other)
//    {
//        // check identity
//        if (this == &other)
//        {
//            return *this;
//        }
//
//        // create a deep-copy of the content of the other tree
//        // shortcut for empty sets
//        if (other.empty())
//        {
//            return *this;
//        }
//
//        // clone content (deep copy)
//        root = other.root->clone();
//
//        // update leftmost reference
//        auto tmp = root;
//        while (!tmp->isLeaf())
//        {
//            tmp = tmp->getChild(0);
//        }
//        leftmost = static_cast<leaf_node*>(tmp);
//
//        // done
//        return *this;
//    }
//
//    // Implementation of an equality operation for trees.
//    bool operator==(const btree& other) const
//    {
//        // check identity
//        if (this == &other)
//        {
//            return true;
//        }
//
//        // check size
//        if (size() != other.size())
//        {
//            return false;
//        }
//        if (size() < other.size())
//        {
//            return other == *this;
//        }
//
//        // check content
//        for (const auto& key : other)
//        {
//            if (!contains(key))
//            {
//                return false;
//            }
//        }
//        return true;
//    }
//
//    // Implementation of an inequality operation for trees.
//    bool operator!=(const btree& other) const
//    {
//        return !(*this == other);
//    }
//
//    // -- for debugging --
//
//    // Determines the number of levels contained in this tree.
//    size_type getDepth() const
//    {
//        return (empty()) ? 0 : root->getDepth();
//    }
//
//    // Determines the number of nodes contained in this tree.
//    size_type getNumNodes() const
//    {
//        return (empty()) ? 0 : root->countNodes();
//    }
//
//    // Determines the amount of memory used by this data structure
//    size_type getMemoryUsage() const
//    {
//        return sizeof(*this) + (empty() ? 0 : root->getMemoryUsage());
//    }
//
//    /*
//     * Prints a textual representation of this tree to the given
//     * output stream (mostly for debugging and tuning).
//     */
//    void printTree(std::ostream& out = std::cout) const
//    {
//        out << "B-Tree with " << size() << " elements:\n";
//        if (empty())
//        {
//            out << " - empty - \n";
//        }
//        else
//        {
//            root->printTree(out, "");
//        }
//    }
//
//    /**
//     * Prints a textual summary of statistical properties of this
//     * tree to the given output stream (for debugging and tuning).
//     */
//    void printStats(std::ostream& out = std::cout) const
//    {
//        auto nodes = getNumNodes();
//        out << " ---------------------------------\n";
//        out << "  Elements: " << size() << std::endl;
//        out << "  Depth:    " << (empty() ? 0 : root->getDepth()) << std::endl;
//        out << "  Nodes:    " << nodes << std::endl;
//        out << " ---------------------------------\n";
//        out << "  Size of inner node: " << sizeof(inner_node) << std::endl;
//        out << "  Size of leaf node:  " << sizeof(leaf_node) << std::endl;
//        out << "  Size of Key:        " << sizeof(Key) << std::endl;
//        out << "  max keys / node:  " << node::maxKeys << std::endl;
//        out << "  avg keys / node:  " << (size() / (double)nodes) << std::endl;
//        out << "  avg filling rate: " << ((size() / (double)nodes) / node::maxKeys) << std::endl;
//        out << " ---------------------------------\n";
//        out << "  insert-hint (hits/misses/total): " << hint_stats.inserts.getHits() << "/" << hint_stats.inserts.getMisses() << "/" << hint_stats.inserts.getAccesses()
//            << std::endl;
//        out << "  contains-hint(hits/misses/total):" << hint_stats.contains.getHits() << "/" << hint_stats.contains.getMisses() << "/" << hint_stats.contains.getAccesses()
//            << std::endl;
//        out << "  lower-bound-hint (hits/misses/total):" << hint_stats.lower_bound.getHits() << "/" << hint_stats.lower_bound.getMisses() << "/"
//            << hint_stats.lower_bound.getAccesses() << std::endl;
//        out << "  upper-bound-hint (hits/misses/total):" << hint_stats.upper_bound.getHits() << "/" << hint_stats.upper_bound.getMisses() << "/"
//            << hint_stats.upper_bound.getAccesses() << std::endl;
//        out << " ---------------------------------\n";
//    }
//
//    /**
//     * Checks the consistency of this tree.
//     */
//    bool check()
//    {
//        auto ok = empty() || root->check(comp, root);
//        if (!ok)
//        {
//            printTree();
//        }
//        return ok;
//    }
//
//    /**
//     * A static member enabling the bulk-load of ordered data into an empty
//     * tree. This function is much more efficient in creating a index over
//     * an ordered set of elements than an iterative insertion of values.
//     *
//     * @tparam Iter .. the type of iterator specifying the range
//     *                     it must be a random-access iterator
//     */
//    template<typename R, typename Iter>
//    static typename std::enable_if<std::is_same<typename std::iterator_traits<Iter>::iterator_category, std::random_access_iterator_tag>::value, R>::type load(const Iter& a,
//                                                                                                                                                               const Iter& b)
//    {
//        // quick exit - empty range
//        if (a == b)
//        {
//            return R();
//        }
//
//        // resolve tree recursively
//        auto root = buildSubTree(a, b - 1);
//
//        // find leftmost node
//        node* leftmost = root;
//        while (!leftmost->isLeaf())
//        {
//            leftmost = leftmost->getChild(0);
//        }
//
//        // build result
//        return R(b - a, root, static_cast<leaf_node*>(leftmost));
//    }
//
// protected:
//    /**
//     * Determines whether the range covered by the given node is also
//     * covering the given key value.
//     */
//    bool covers(const node* node, const Key& k) const
//    {
//        if (isSet)
//        {
//            // in sets we can include the ends as covered elements
//            return !node->isEmpty() && !less(k, node->keys[0]) && !less(node->keys[node->numElements - 1], k);
//        }
//        // in multi-sets the ends may not be completely covered
//        return !node->isEmpty() && less(node->keys[0], k) && less(k, node->keys[node->numElements - 1]);
//    }
//
//    /**
//     * Determines whether the range covered by the given node is also
//     * covering the given key value.
//     */
//    bool weak_covers(const node* node, const Key& k) const
//    {
//        if (isSet)
//        {
//            // in sets we can include the ends as covered elements
//            return !node->isEmpty() && !weak_less(k, node->keys[0]) && !weak_less(node->keys[node->numElements - 1], k);
//        }
//        // in multi-sets the ends may not be completely covered
//        return !node->isEmpty() && weak_less(node->keys[0], k) && weak_less(k, node->keys[node->numElements - 1]);
//    }
//
// private:
//    /**
//     * Determines whether the range covered by this node covers
//     * the upper bound of the given key.
//     */
//    bool coversUpperBound(const node* node, const Key& k) const
//    {
//        // ignore edges
//        return !node->isEmpty() && !less(k, node->keys[0]) && less(k, node->keys[node->numElements - 1]);
//    }
//
//    // Utility function for the load operation above.
//    template<typename Iter>
//    static node* buildSubTree(const Iter& a, const Iter& b)
//    {
//        const int N = node::maxKeys;
//
//        // divide range in N+1 sub-ranges
//        int length = (b - a) + 1;
//
//        // terminal case: length is less then maxKeys
//        if (length <= N)
//        {
//            // create a leaf node
//            node* res        = new leaf_node();
//            res->numElements = length;
//
//            for (int i = 0; i < length; ++i)
//            {
//                res->keys[i] = a[i];
//            }
//
//            return res;
//        }
//
//        // recursive case - compute step size
//        int numKeys = N;
//        int step    = ((length - numKeys) / (numKeys + 1));
//
//        while (numKeys > 1 && (step < N / 2))
//        {
//            numKeys--;
//            step = ((length - numKeys) / (numKeys + 1));
//        }
//
//        // create inner node
//        node* res        = new inner_node();
//        res->numElements = numKeys;
//
//        Iter c = a;
//        for (int i = 0; i < numKeys; ++i)
//        {
//            // get dividing key
//            res->keys[i] = c[step];
//
//            // get sub-tree
//            auto child            = buildSubTree(c, c + (step - 1));
//            child->parent         = res;
//            child->position       = i;
//            res->getChildren()[i] = child;
//
//            c = c + (step + 1);
//        }
//
//        // and the remaining part
//        auto child                  = buildSubTree(c, b);
//        child->parent               = res;
//        child->position             = numKeys;
//        res->getChildren()[numKeys] = child;
//
//        // done
//        return res;
//    }
//}; // namespace souffle
//
//// Instantiation of static member search.
// template<typename Key, typename Comparator, typename Allocator, unsigned blockSize, typename SearchStrategy, bool isSet, typename WeakComparator, typename Updater>
// const SearchStrategy btree<Key, Comparator, Allocator, blockSize, SearchStrategy, isSet, WeakComparator, Updater>::search;
//
//} // end namespace detail
//
///**
// * A b-tree based set implementation.
// *
// * @tparam Key             .. the element type to be stored in this set
// * @tparam Comparator     .. a class defining an order on the stored elements
// * @tparam Allocator     .. utilized for allocating memory for required nodes
// * @tparam blockSize    .. determines the number of bytes/block utilized by leaf nodes
// * @tparam SearchStrategy .. enables switching between linear, binary or any other search strategy
// */
// template<typename Key,
//         typename Comparator     = detail::comparator<Key>,
//         typename Allocator      = std::allocator<Key>, // is ignored so far
//         unsigned blockSize      = 256,
//         typename SearchStrategy = typename souffle::detail::default_strategy<Key>::type,
//         typename WeakComparator = Comparator,
//         typename Updater        = souffle::detail::updater<Key>>
// class btree_set : public souffle::detail::btree<Key, Comparator, Allocator, blockSize, SearchStrategy, true, WeakComparator, Updater>
//{
//    using super = souffle::detail::btree<Key, Comparator, Allocator, blockSize, SearchStrategy, true, WeakComparator, Updater>;
//
//    friend class souffle::detail::btree<Key, Comparator, Allocator, blockSize, SearchStrategy, true, WeakComparator, Updater>;
//
// public:
//    /**
//     * A default constructor creating an empty set.
//     */
//    btree_set(const Comparator& comp = Comparator(), const WeakComparator& weak_comp = WeakComparator()) : super(comp, weak_comp) {}
//
//    /**
//     * A constructor creating a set based on the given range.
//     */
//    template<typename Iter>
//    btree_set(const Iter& a, const Iter& b)
//    {
//        this->insert(a, b);
//    }
//
//    // A copy constructor.
//    btree_set(const btree_set& other) : super(other) {}
//
//    // A move constructor.
//    btree_set(btree_set&& other) : super(std::move(other)) {}
//
// private:
//    // A constructor required by the bulk-load facility.
//    template<typename s, typename n, typename l>
//    btree_set(s size, n* root, l* leftmost) : super(size, root, leftmost)
//    {
//    }
//
// public:
//    // Support for the assignment operator.
//    btree_set& operator=(const btree_set& other)
//    {
//        super::operator=(other);
//        return *this;
//    }
//
//    // Support for the bulk-load operator.
//    template<typename Iter>
//    static btree_set load(const Iter& a, const Iter& b)
//    {
//        return super::template load<btree_set>(a, b);
//    }
//};
//
// template<typename Key,
//         typename Comparator     = detail::comparator<Key>,
//         typename Allocator      = std::allocator<Key>, // is ignored so far
//         unsigned blockSize      = 256,
//         typename SearchStrategy = typename souffle::detail::default_strategy<Key>::type,
//         typename WeakComparator = Comparator,
//         typename Updater        = souffle::detail::updater<Key>>
// class btree_multiset : public souffle::detail::btree<Key, Comparator, Allocator, blockSize, SearchStrategy, false, WeakComparator, Updater>
//{
//    using super = souffle::detail::btree<Key, Comparator, Allocator, blockSize, SearchStrategy, false, WeakComparator, Updater>;
//
//    friend class souffle::detail::btree<Key, Comparator, Allocator, blockSize, SearchStrategy, false, WeakComparator, Updater>;
//
// public:
//    /**
//     * A default constructor creating an empty set.
//     */
//    btree_multiset(const Comparator& comp = Comparator(), const WeakComparator& weak_comp = WeakComparator()) : super(comp, weak_comp) {}
//
//    /**
//     * A constructor creating a set based on the given range.
//     */
//    template<typename Iter>
//    btree_multiset(const Iter& a, const Iter& b)
//    {
//        this->insert(a, b);
//    }
//
//    // A copy constructor.
//    btree_multiset(const btree_multiset& other) : super(other) {}
//
//    // A move constructor.
//    btree_multiset(btree_multiset&& other) : super(std::move(other)) {}
//
// private:
//    // A constructor required by the bulk-load facility.
//    template<typename s, typename n, typename l>
//    btree_multiset(s size, n* root, l* leftmost) : super(size, root, leftmost)
//    {
//    }
//
// public:
//    // Support for the assignment operator.
//    btree_multiset& operator=(const btree_multiset& other)
//    {
//        super::operator=(other);
//        return *this;
//    }
//
//    // Support for the bulk-load operator.
//    template<typename Iter>
//    static btree_multiset load(const Iter& a, const Iter& b)
//    {
//        return super::template load<btree_multiset>(a, b);
//    }
//};
