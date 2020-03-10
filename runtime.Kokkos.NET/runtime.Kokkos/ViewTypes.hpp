#pragma once

#include "KokkosAPI.hpp"

#include <Kokkos_Core.hpp>
#include <Teuchos_RCP.hpp>

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
    OpenMP,
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
    UInt64,

    // ConstSingle = UInt64 + 1,
    // ConstDouble,
    // ConstBool,
    // ConstInt8,
    // ConstUInt8,
    // ConstInt16,
    // ConstUInt16,
    // ConstInt32,
    // ConstUInt32,
    // ConstInt64,
    // ConstUInt64
};

template<typename DataType, class ExecutionSpace, typename Layout, unsigned Rank>
struct __declspec(align(sizeof(uint64))) NdArrayTraits;

#define DEF_TEMPLATE(TYPE_NAME, TYPE, EXECUTION_SPACE, LAYOUT)                                     \
    template<>                                                                                     \
    struct NdArrayTraits<TYPE, Kokkos::EXECUTION_SPACE, Kokkos::Layout##LAYOUT, 0>                 \
    {                                                                                              \
        static constexpr DataTypeKind       data_type       = DataTypeKind::TYPE_NAME;             \
        static constexpr uint16             rank            = 0;                                   \
        static constexpr LayoutKind         layout          = LayoutKind::LAYOUT;                  \
        static constexpr ExecutionSpaceKind execution_space = ExecutionSpaceKind::EXECUTION_SPACE; \
    };                                                                                             \
    template<>                                                                                     \
    struct NdArrayTraits<TYPE, Kokkos::EXECUTION_SPACE, Kokkos::Layout##LAYOUT, 1>                 \
    {                                                                                              \
        static constexpr DataTypeKind       data_type       = DataTypeKind::TYPE_NAME;             \
        static constexpr uint16             rank            = 1;                                   \
        static constexpr LayoutKind         layout          = LayoutKind::LAYOUT;                  \
        static constexpr ExecutionSpaceKind execution_space = ExecutionSpaceKind::EXECUTION_SPACE; \
    };                                                                                             \
    template<>                                                                                     \
    struct NdArrayTraits<TYPE, Kokkos::EXECUTION_SPACE, Kokkos::Layout##LAYOUT, 2>                 \
    {                                                                                              \
        static constexpr DataTypeKind       data_type       = DataTypeKind::TYPE_NAME;             \
        static constexpr uint16             rank            = 2;                                   \
        static constexpr LayoutKind         layout          = LayoutKind::LAYOUT;                  \
        static constexpr ExecutionSpaceKind execution_space = ExecutionSpaceKind::EXECUTION_SPACE; \
    };                                                                                             \
    template<>                                                                                     \
    struct NdArrayTraits<TYPE, Kokkos::EXECUTION_SPACE, Kokkos::Layout##LAYOUT, 3>                 \
    {                                                                                              \
        static constexpr DataTypeKind       data_type       = DataTypeKind::TYPE_NAME;             \
        static constexpr uint16             rank            = 3;                                   \
        static constexpr LayoutKind         layout          = LayoutKind::LAYOUT;                  \
        static constexpr ExecutionSpaceKind execution_space = ExecutionSpaceKind::EXECUTION_SPACE; \
    };                                                                                             \
    template<>                                                                                     \
    struct NdArrayTraits<TYPE, Kokkos::EXECUTION_SPACE, Kokkos::Layout##LAYOUT, 4>                 \
    {                                                                                              \
        static constexpr DataTypeKind       data_type       = DataTypeKind::TYPE_NAME;             \
        static constexpr uint16             rank            = 4;                                   \
        static constexpr LayoutKind         layout          = LayoutKind::LAYOUT;                  \
        static constexpr ExecutionSpaceKind execution_space = ExecutionSpaceKind::EXECUTION_SPACE; \
    };                                                                                             \
    template<>                                                                                     \
    struct NdArrayTraits<TYPE, Kokkos::EXECUTION_SPACE, Kokkos::Layout##LAYOUT, 5>                 \
    {                                                                                              \
        static constexpr DataTypeKind       data_type       = DataTypeKind::TYPE_NAME;             \
        static constexpr uint16             rank            = 5;                                   \
        static constexpr LayoutKind         layout          = LayoutKind::LAYOUT;                  \
        static constexpr ExecutionSpaceKind execution_space = ExecutionSpaceKind::EXECUTION_SPACE; \
    };                                                                                             \
    template<>                                                                                     \
    struct NdArrayTraits<TYPE, Kokkos::EXECUTION_SPACE, Kokkos::Layout##LAYOUT, 6>                 \
    {                                                                                              \
        static constexpr DataTypeKind       data_type       = DataTypeKind::TYPE_NAME;             \
        static constexpr uint16             rank            = 6;                                   \
        static constexpr LayoutKind         layout          = LayoutKind::LAYOUT;                  \
        static constexpr ExecutionSpaceKind execution_space = ExecutionSpaceKind::EXECUTION_SPACE; \
    };                                                                                             \
    template<>                                                                                     \
    struct NdArrayTraits<TYPE, Kokkos::EXECUTION_SPACE, Kokkos::Layout##LAYOUT, 7>                 \
    {                                                                                              \
        static constexpr DataTypeKind       data_type       = DataTypeKind::TYPE_NAME;             \
        static constexpr uint16             rank            = 7;                                   \
        static constexpr LayoutKind         layout          = LayoutKind::LAYOUT;                  \
        static constexpr ExecutionSpaceKind execution_space = ExecutionSpaceKind::EXECUTION_SPACE; \
    };                                                                                             \
    template<>                                                                                     \
    struct NdArrayTraits<TYPE, Kokkos::EXECUTION_SPACE, Kokkos::Layout##LAYOUT, 8>                 \
    {                                                                                              \
        static constexpr DataTypeKind       data_type       = DataTypeKind::TYPE_NAME;             \
        static constexpr uint16             rank            = 8;                                   \
        static constexpr LayoutKind         layout          = LayoutKind::LAYOUT;                  \
        static constexpr ExecutionSpaceKind execution_space = ExecutionSpaceKind::EXECUTION_SPACE; \
    };

#define TEMPLATE(DEF, EXECUTION_SPACE, LAYOUT)   \
    DEF(Single, float, EXECUTION_SPACE, LAYOUT)  \
    DEF(Double, double, EXECUTION_SPACE, LAYOUT) \
    DEF(Bool, bool, EXECUTION_SPACE, LAYOUT)     \
    DEF(Int8, int8, EXECUTION_SPACE, LAYOUT)     \
    DEF(UInt8, uint8, EXECUTION_SPACE, LAYOUT)   \
    DEF(Int16, int16, EXECUTION_SPACE, LAYOUT)   \
    DEF(UInt16, uint16, EXECUTION_SPACE, LAYOUT) \
    DEF(Int32, int32, EXECUTION_SPACE, LAYOUT)   \
    DEF(UInt32, uint32, EXECUTION_SPACE, LAYOUT) \
    DEF(Int64, int64, EXECUTION_SPACE, LAYOUT)   \
    DEF(UInt64, uint64, EXECUTION_SPACE, LAYOUT)

TEMPLATE(DEF_TEMPLATE, Serial, Right)
TEMPLATE(DEF_TEMPLATE, OpenMP, Right)
TEMPLATE(DEF_TEMPLATE, Cuda, Left)
TEMPLATE(DEF_TEMPLATE, Serial, Left)
TEMPLATE(DEF_TEMPLATE, OpenMP, Left)
TEMPLATE(DEF_TEMPLATE, Cuda, Right)

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
    //#define TEMPLATE(DEF, EXECUTION_SPACE)   \
//    DEF(Single, float, EXECUTION_SPACE)  \
//    DEF(Double, double, EXECUTION_SPACE) \
//    DEF(Bool, bool, EXECUTION_SPACE)     \
//    DEF(Int8, int8, EXECUTION_SPACE)     \
//    DEF(UInt8, uint8, EXECUTION_SPACE)   \
//    DEF(Int16, int16, EXECUTION_SPACE)   \
//    DEF(UInt16, uint16, EXECUTION_SPACE) \
//    DEF(Int32, int32, EXECUTION_SPACE)   \
//    DEF(UInt32, uint32, EXECUTION_SPACE) \
//    DEF(Int64, int64, EXECUTION_SPACE)   \
//    DEF(UInt64, uint64, EXECUTION_SPACE)

    //#define DEF_TEMPLATE(TYPE_NAME, TYPE, EXECUTION_SPACE)                                                                                                                           \
//    typedef Kokkos::View<TYPE, typename Kokkos::EXECUTION_SPACE::array_layout, Kokkos::EXECUTION_SPACE, Kokkos::MemoryUnmanaged>    View_##TYPE_NAME##_##EXECUTION_SPACE##_0d_t; \
//    typedef Kokkos::View<TYPE*, typename Kokkos::EXECUTION_SPACE::array_layout, Kokkos::EXECUTION_SPACE, Kokkos::MemoryUnmanaged>   View_##TYPE_NAME##_##EXECUTION_SPACE##_1d_t; \
//    typedef Kokkos::View<TYPE**, typename Kokkos::EXECUTION_SPACE::array_layout, Kokkos::EXECUTION_SPACE, Kokkos::MemoryUnmanaged>  View_##TYPE_NAME##_##EXECUTION_SPACE##_2d_t; \
//    typedef Kokkos::View<TYPE***, typename Kokkos::EXECUTION_SPACE::array_layout, Kokkos::EXECUTION_SPACE, Kokkos::MemoryUnmanaged> View_##TYPE_NAME##_##EXECUTION_SPACE##_3d_t;

    // TEMPLATE(DEF_TEMPLATE, Serial)
    // TEMPLATE(DEF_TEMPLATE, OpenMP)
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

    //#undef TEMPLATE
    //#undef DEF_TEMPLATE
}

template<DataTypeKind TDataType, unsigned Rank, ExecutionSpaceKind TExecutionSpace>
struct ViewBuilder;

#undef TEMPLATE
#undef DEF_TEMPLATE

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

#define DEF_TEMPLATE(TYPE_NAME, TYPE, EXECUTION_SPACE)                                                                        \
    template<>                                                                                                                \
    struct ViewBuilder<DataTypeKind::TYPE_NAME, 0, ExecutionSpaceKind::EXECUTION_SPACE>                                       \
    {                                                                                                                         \
        using ViewType = Kokkos::View<TYPE, typename Kokkos::EXECUTION_SPACE::array_layout, Kokkos::EXECUTION_SPACE>;         \
    };                                                                                                                        \
    template<>                                                                                                                \
    struct ViewBuilder<DataTypeKind::TYPE_NAME, 1, ExecutionSpaceKind::EXECUTION_SPACE>                                       \
    {                                                                                                                         \
        using ViewType = Kokkos::View<TYPE*, typename Kokkos::EXECUTION_SPACE::array_layout, Kokkos::EXECUTION_SPACE>;        \
    };                                                                                                                        \
    template<>                                                                                                                \
    struct ViewBuilder<DataTypeKind::TYPE_NAME, 2, ExecutionSpaceKind::EXECUTION_SPACE>                                       \
    {                                                                                                                         \
        using ViewType = Kokkos::View<TYPE**, typename Kokkos::EXECUTION_SPACE::array_layout, Kokkos::EXECUTION_SPACE>;       \
    };                                                                                                                        \
    template<>                                                                                                                \
    struct ViewBuilder<DataTypeKind::TYPE_NAME, 3, ExecutionSpaceKind::EXECUTION_SPACE>                                       \
    {                                                                                                                         \
        using ViewType = Kokkos::View<TYPE***, typename Kokkos::EXECUTION_SPACE::array_layout, Kokkos::EXECUTION_SPACE>;      \
    };                                                                                                                        \
    template<>                                                                                                                \
    struct ViewBuilder<DataTypeKind::TYPE_NAME, 4, ExecutionSpaceKind::EXECUTION_SPACE>                                       \
    {                                                                                                                         \
        using ViewType = Kokkos::View<TYPE****, typename Kokkos::EXECUTION_SPACE::array_layout, Kokkos::EXECUTION_SPACE>;     \
    };                                                                                                                        \
    template<>                                                                                                                \
    struct ViewBuilder<DataTypeKind::TYPE_NAME, 5, ExecutionSpaceKind::EXECUTION_SPACE>                                       \
    {                                                                                                                         \
        using ViewType = Kokkos::View<TYPE*****, typename Kokkos::EXECUTION_SPACE::array_layout, Kokkos::EXECUTION_SPACE>;    \
    };                                                                                                                        \
    template<>                                                                                                                \
    struct ViewBuilder<DataTypeKind::TYPE_NAME, 6, ExecutionSpaceKind::EXECUTION_SPACE>                                       \
    {                                                                                                                         \
        using ViewType = Kokkos::View<TYPE******, typename Kokkos::EXECUTION_SPACE::array_layout, Kokkos::EXECUTION_SPACE>;   \
    };                                                                                                                        \
    template<>                                                                                                                \
    struct ViewBuilder<DataTypeKind::TYPE_NAME, 7, ExecutionSpaceKind::EXECUTION_SPACE>                                       \
    {                                                                                                                         \
        using ViewType = Kokkos::View<TYPE*******, typename Kokkos::EXECUTION_SPACE::array_layout, Kokkos::EXECUTION_SPACE>;  \
    };                                                                                                                        \
    template<>                                                                                                                \
    struct ViewBuilder<DataTypeKind::TYPE_NAME, 8, ExecutionSpaceKind::EXECUTION_SPACE>                                       \
    {                                                                                                                         \
        using ViewType = Kokkos::View<TYPE********, typename Kokkos::EXECUTION_SPACE::array_layout, Kokkos::EXECUTION_SPACE>; \
    };

TEMPLATE(DEF_TEMPLATE, Serial)
TEMPLATE(DEF_TEMPLATE, OpenMP)
TEMPLATE(DEF_TEMPLATE, Cuda)

#undef TEMPLATE
#undef DEF_TEMPLATE

//#define DEF_TEMPLATE(TYPE_NAME, TYPE, EXECUTION_SPACE) ViewBuilder<DataTypeKind::TYPE_NAME, 0, EXECUTION_SPACE>::ViewType
//
//#define TEMPLATE_RANK0(DEF, EXECUTION_SPACE)                                                                                                                \
//    DEF(Single, float, EXECUTION_SPACE) \
//     DEF(Double, double, EXECUTION_SPACE), DEF(Int8, int8, EXECUTION_SPACE), DEF(UInt8, uint8, EXECUTION_SPACE),        \
//        DEF(Int16, int16, EXECUTION_SPACE), DEF(UInt16, uint16, EXECUTION_SPACE), DEF(Int32, int32, EXECUTION_SPACE), DEF(UInt32, uint32, EXECUTION_SPACE), \
//        DEF(Int64, int64, EXECUTION_SPACE), DEF(UInt64, uint64, EXECUTION_SPACE)
//
//#define TEMPLATE_RANK1(DEF, EXECUTION_SPACE)                                                                                                                    \
//    DEF(Single, float*, EXECUTION_SPACE), DEF(Double, double*, EXECUTION_SPACE), DEF(Int8, int8*, EXECUTION_SPACE), DEF(UInt8, uint8*, EXECUTION_SPACE),        \
//        DEF(Int16, int16*, EXECUTION_SPACE), DEF(UInt16, uint16*, EXECUTION_SPACE), DEF(Int32, int32*, EXECUTION_SPACE), DEF(UInt32, uint32*, EXECUTION_SPACE), \
//        DEF(Int64, int64*, EXECUTION_SPACE), DEF(UInt64, uint64*, EXECUTION_SPACE)
//
//#define TEMPLATE_RANK2(DEF, EXECUTION_SPACE)                                                                                                                        \
//    DEF(Single, float**, EXECUTION_SPACE), DEF(Double, double**, EXECUTION_SPACE), DEF(Int8, int8**, EXECUTION_SPACE), DEF(UInt8, uint8**, EXECUTION_SPACE),        \
//        DEF(Int16, int16**, EXECUTION_SPACE), DEF(UInt16, uint16**, EXECUTION_SPACE), DEF(Int32, int32**, EXECUTION_SPACE), DEF(UInt32, uint32**, EXECUTION_SPACE), \
//        DEF(Int64, int64**, EXECUTION_SPACE), DEF(UInt64, uint64**, EXECUTION_SPACE)
//
//#define TEMPLATE_RANK3(DEF, EXECUTION_SPACE)                                                                                                                            \
//    DEF(Single, float***, EXECUTION_SPACE), DEF(Double, double***, EXECUTION_SPACE), DEF(Int8, int8***, EXECUTION_SPACE), DEF(UInt8, uint8***, EXECUTION_SPACE),        \
//        DEF(Int16, int16***, EXECUTION_SPACE), DEF(UInt16, uint16***, EXECUTION_SPACE), DEF(Int32, int32***, EXECUTION_SPACE), DEF(UInt32, uint32***, EXECUTION_SPACE), \
//        DEF(Int64, int64***, EXECUTION_SPACE), DEF(UInt64, uint64***, EXECUTION_SPACE)

#undef TEMPLATE
#undef DEF_TEMPLATE
#undef TEMPLATE_RANK0
#undef TEMPLATE_RANK1
#undef TEMPLATE_RANK2
#undef TEMPLATE_RANK3
