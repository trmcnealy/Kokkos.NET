#pragma once

#include "KokkosAPI.hpp"

#include <Kokkos_Core.hpp>

#define NDARRAY_MAX_RANK 8

enum class LayoutKind : uint16
{
    Unknown = 0xFFFF,
    Left    = 0,
    Right,
    Stride
};

enum class ExecutionSpaceKind : uint16
{
    Unknown = 0xFFFF,
    Serial  = 0,
    Threads,
    Cuda
};

enum class DataTypeKind : uint16
{
    Unknown = 0xFFFF,
    Single  = 0,
    Double,
    Bool,
    Int8,
    UInt8,
    Int16,
    UInt16,
    Int32,
    UInt32,
    Int64,
    UInt64
};

struct NdArray
{
    DataTypeKind       data_type;
    uint16             rank;
    LayoutKind         layout;
    ExecutionSpaceKind execution_space;
    uint64             dims[NDARRAY_MAX_RANK];
    uint64             strides[NDARRAY_MAX_RANK];
    void*              data;
    const char*        label;
};

namespace Compatible
{
#define TEMPLATE(DEF, EXECUTION_SPACE)   \
    DEF(Single, float, EXECUTION_SPACE)  \
    DEF(Double, double, EXECUTION_SPACE) \
    DEF(Bool, bool, EXECUTION_SPACE)     \
    DEF(Int8, int8, EXECUTION_SPACE)     \
    DEF(UInt8, uint8, EXECUTION_SPACE)   \
    DEF(Int16, int16, EXECUTION_SPACE)   \
    DEF(UInt16, uint16, EXECUTION_SPACE) \
    DEF(Int32, int32, EXECUTION_SPACE)   \
    DEF(UInt32, uint32, EXECUTION_SPACE) \
    DEF(Int64, int64, EXECUTION_SPACE)   \
    DEF(UInt64, uint64, EXECUTION_SPACE)

#define DEF_TEMPLATE(TYPE_NAME, TYPE, EXECUTION_SPACE)                                                                                                                           \
    typedef Kokkos::View<TYPE, typename Kokkos::EXECUTION_SPACE::array_layout, Kokkos::EXECUTION_SPACE, Kokkos::MemoryUnmanaged>    View_##TYPE_NAME##_##EXECUTION_SPACE##_0d_t; \
    typedef Kokkos::View<TYPE*, typename Kokkos::EXECUTION_SPACE::array_layout, Kokkos::EXECUTION_SPACE, Kokkos::MemoryUnmanaged>   View_##TYPE_NAME##_##EXECUTION_SPACE##_1d_t; \
    typedef Kokkos::View<TYPE**, typename Kokkos::EXECUTION_SPACE::array_layout, Kokkos::EXECUTION_SPACE, Kokkos::MemoryUnmanaged>  View_##TYPE_NAME##_##EXECUTION_SPACE##_2d_t; \
    typedef Kokkos::View<TYPE***, typename Kokkos::EXECUTION_SPACE::array_layout, Kokkos::EXECUTION_SPACE, Kokkos::MemoryUnmanaged> View_##TYPE_NAME##_##EXECUTION_SPACE##_3d_t;

    // TEMPLATE(DEF_TEMPLATE, Serial)
    // TEMPLATE(DEF_TEMPLATE, Threads)
    // TEMPLATE(DEF_TEMPLATE, Cuda)

    // template<typename DataType>
    // Kokkos::View<DataType, Kokkos::LayoutStride, Kokkos::HostSpace, Kokkos::MemoryUnmanaged> view_from_ndarray(const NdArray& ndarray)
    //{
    //    size_type dimensions[Kokkos::ARRAY_LAYOUT_MAX_RANK] = {};
    //    size_type strides[Kokkos::ARRAY_LAYOUT_MAX_RANK]    = {};

    //    using traits        = Kokkos::ViewTraits<DataType>;
    //    using value_type    = typename traits::value_type;
    //    constexpr auto rank = Kokkos::ViewTraits<DataType>::rank;

    //    if(rank != ndarray.rank)
    //    {
    //        std::cerr << "Requested Kokkos view of rank " << rank << " for ndarray with rank " << ndarray.rank << "." << std::endl;
    //        std::exit(EXIT_FAILURE);
    //    }

    //    std::copy(ndarray.dims, ndarray.dims + ndarray.rank, dimensions);
    //    std::copy(ndarray.strides, ndarray.strides + ndarray.rank, strides);

    //    // clang-format off
    //    Kokkos::LayoutStride layout
    //    {
    //        dimensions[0], strides[0],
    //        dimensions[1], strides[1],
    //        dimensions[2], strides[2],
    //        dimensions[3], strides[3],
    //        dimensions[4], strides[4],
    //        dimensions[5], strides[5],
    //        dimensions[6], strides[6],
    //        dimensions[7], strides[7]
    //    };
    //    // clang-format on

    //    return Kokkos::View<DataType, Kokkos::LayoutStride, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>(reinterpret_cast<value_type*>(ndarray.data), layout);
    //}

#undef TEMPLATE
#undef DEF_TEMPLATE
}

template<DataTypeKind TDataType, unsigned Rank, ExecutionSpaceKind TExecutionSpace>
struct ViewBuilder;

#define TEMPLATE(DEF, EXECUTION_SPACE)   \
    DEF(Single, float, EXECUTION_SPACE)  \
    DEF(Double, double, EXECUTION_SPACE) \
    DEF(Bool, bool, EXECUTION_SPACE)     \
    DEF(Int8, int8, EXECUTION_SPACE)     \
    DEF(UInt8, uint8, EXECUTION_SPACE)   \
    DEF(Int16, int16, EXECUTION_SPACE)   \
    DEF(UInt16, uint16, EXECUTION_SPACE) \
    DEF(Int32, int32, EXECUTION_SPACE)   \
    DEF(UInt32, uint32, EXECUTION_SPACE) \
    DEF(Int64, int64, EXECUTION_SPACE)   \
    DEF(UInt64, uint64, EXECUTION_SPACE)

#define DEF_TEMPLATE(TYPE_NAME, TYPE, EXECUTION_SPACE)                                                                   \
    template<>                                                                                                           \
    struct ViewBuilder<DataTypeKind::TYPE_NAME, 0, ExecutionSpaceKind::EXECUTION_SPACE>                                  \
    {                                                                                                                    \
        using ViewType = Kokkos::View<TYPE, typename Kokkos::EXECUTION_SPACE::array_layout, Kokkos::EXECUTION_SPACE>;    \
    };                                                                                                                   \
    template<>                                                                                                           \
    struct ViewBuilder<DataTypeKind::TYPE_NAME, 1, ExecutionSpaceKind::EXECUTION_SPACE>                                  \
    {                                                                                                                    \
        using ViewType = Kokkos::View<TYPE*, typename Kokkos::EXECUTION_SPACE::array_layout, Kokkos::EXECUTION_SPACE>;   \
    };                                                                                                                   \
    template<>                                                                                                           \
    struct ViewBuilder<DataTypeKind::TYPE_NAME, 2, ExecutionSpaceKind::EXECUTION_SPACE>                                  \
    {                                                                                                                    \
        using ViewType = Kokkos::View<TYPE**, typename Kokkos::EXECUTION_SPACE::array_layout, Kokkos::EXECUTION_SPACE>;  \
    };                                                                                                                   \
    template<>                                                                                                           \
    struct ViewBuilder<DataTypeKind::TYPE_NAME, 3, ExecutionSpaceKind::EXECUTION_SPACE>                                  \
    {                                                                                                                    \
        using ViewType = Kokkos::View<TYPE***, typename Kokkos::EXECUTION_SPACE::array_layout, Kokkos::EXECUTION_SPACE>; \
    };

#include <Kokkos_Core.hpp>
#include <Teuchos_RCP.hpp>

TEMPLATE(DEF_TEMPLATE, Serial)
TEMPLATE(DEF_TEMPLATE, Threads)
TEMPLATE(DEF_TEMPLATE, Cuda)

#undef TEMPLATE
#undef DEF_TEMPLATE

#define DEF_TEMPLATE(TYPE_NAME, TYPE, EXECUTION_SPACE) ViewBuilder<DataTypeKind::TYPE_NAME, 0, EXECUTION_SPACE>::ViewType

#define TEMPLATE_RANK0(DEF, EXECUTION_SPACE)                                                                                                                \
    DEF(Single, float, EXECUTION_SPACE), DEF(Double, double, EXECUTION_SPACE), DEF(Int8, int8, EXECUTION_SPACE), DEF(UInt8, uint8, EXECUTION_SPACE),        \
        DEF(Int16, int16, EXECUTION_SPACE), DEF(UInt16, uint16, EXECUTION_SPACE), DEF(Int32, int32, EXECUTION_SPACE), DEF(UInt32, uint32, EXECUTION_SPACE), \
        DEF(Int64, int64, EXECUTION_SPACE), DEF(UInt64, uint64, EXECUTION_SPACE)

#define TEMPLATE_RANK1(DEF, EXECUTION_SPACE)                                                                                                                    \
    DEF(Single, float*, EXECUTION_SPACE), DEF(Double, double*, EXECUTION_SPACE), DEF(Int8, int8*, EXECUTION_SPACE), DEF(UInt8, uint8*, EXECUTION_SPACE),        \
        DEF(Int16, int16*, EXECUTION_SPACE), DEF(UInt16, uint16*, EXECUTION_SPACE), DEF(Int32, int32*, EXECUTION_SPACE), DEF(UInt32, uint32*, EXECUTION_SPACE), \
        DEF(Int64, int64*, EXECUTION_SPACE), DEF(UInt64, uint64*, EXECUTION_SPACE)

#define TEMPLATE_RANK2(DEF, EXECUTION_SPACE)                                                                                                                        \
    DEF(Single, float**, EXECUTION_SPACE), DEF(Double, double**, EXECUTION_SPACE), DEF(Int8, int8**, EXECUTION_SPACE), DEF(UInt8, uint8**, EXECUTION_SPACE),        \
        DEF(Int16, int16**, EXECUTION_SPACE), DEF(UInt16, uint16**, EXECUTION_SPACE), DEF(Int32, int32**, EXECUTION_SPACE), DEF(UInt32, uint32**, EXECUTION_SPACE), \
        DEF(Int64, int64**, EXECUTION_SPACE), DEF(UInt64, uint64**, EXECUTION_SPACE)

#define TEMPLATE_RANK3(DEF, EXECUTION_SPACE)                                                                                                                            \
    DEF(Single, float***, EXECUTION_SPACE), DEF(Double, double***, EXECUTION_SPACE), DEF(Int8, int8***, EXECUTION_SPACE), DEF(UInt8, uint8***, EXECUTION_SPACE),        \
        DEF(Int16, int16***, EXECUTION_SPACE), DEF(UInt16, uint16***, EXECUTION_SPACE), DEF(Int32, int32***, EXECUTION_SPACE), DEF(UInt32, uint32***, EXECUTION_SPACE), \
        DEF(Int64, int64***, EXECUTION_SPACE), DEF(UInt64, uint64***, EXECUTION_SPACE)

#undef TEMPLATE
#undef DEF_TEMPLATE
#undef TEMPLATE_RANK0
#undef TEMPLATE_RANK1
#undef TEMPLATE_RANK2
#undef TEMPLATE_RANK3
