#pragma once

#include <Types.hpp>
#include <Constants.hpp>
#include <Guid.hpp>

#include <KokkosAPI.hpp>

#include <Kokkos_Core.hpp>
#include <KokkosBlas.hpp>

#include <Teuchos_RCP.hpp>

#include <Kokkos_Vector.hpp>

#include <string>

// template<typename DataType, class ExecutionSpace>
//__inline void* operator new(unsigned long long, Kokkos::View<DataType, typename ExecutionSpace::array_layout,ExecutionSpace>* view)
//{
//    return view;
//}
//
// template<typename DataType, class ExecutionSpace>
//__inline void operator delete(Kokkos::View<DataType, typename ExecutionSpace::array_layout,ExecutionSpace>* view)
//{
//    view->::~Kokkos::View<DataType, typename ExecutionSpace::array_layout,ExecutionSpace>();
//}

// template<typename T>
// void* operator new(unsigned long long count, const Teuchos::RCP<T>& view_rcp)
//{
//    return new Teuchos::RCP<T>(view_rcp);
//}
// template<typename T>
// void* operator new[](unsigned long long count, const Teuchos::RCP<T>& view_rcp)
//{
//    return new Teuchos::RCP<T>[count];
//}
// template<typename T>
// void operator delete(void* ptr, const Teuchos::RCP<T>& view_rcp)
//{
//    view_rcp.~RCP();
//
//    delete reinterpret_cast<Teuchos::RCP<T>*>(ptr);
//}
// void operator delete[](void* ptr) throw()
//{
//}

enum class LayoutKind : uint16
{
    Unknown = 0xFFFF,
    Left    = 0,
    Right   = 1,
    Stride  = 2
};

enum class ExecutionSpaceKind : uint16
{
    Unknown = 0xFFFF,
    Serial  = 0,
    OpenMP  = 1,
    Cuda    = 2
};

// View <SIMD <double >*

// using  simd_t = simd::simd <double ,simd:: simd_abi ::native >;
// View <simd_t*> a("A",N);
// View <double*> a_s(static_cast <double*>(a.data()),N*simd_t ::size ());
// View <double*> b("B",M);
// View <simd_t*> b_v(static_cast <simd_t*>(b.data()),M/simd_t ::size ()

// using  ABI = pack <8>; int V=1;
// using  ABI = cuda_warp <8>; int V=8;

//  Using  cuda_warp  abi
//  using  simd_t = simd::simd <T,simd:: simd_abi ::cuda_warp <V>  >;
//  Define  simd_storage  type
//  using  simd_storage_t = simd_t :: storage_type;
//  Allocate  memory
// View <simd_storage_t **> data("D",N,M); // will  hold N*M*V Ts

enum class DataTypeKind : uint16
{
    Unknown = 0xFFFF,
    Single  = 0,
    Double  = 1,
    Bool    = 2,
    Int8    = 3,
    UInt8   = 4,
    Int16   = 5,
    UInt16  = 6,
    Int32   = 7,
    UInt32  = 8,
    Int64   = 9,
    UInt64  = 10,
    Char    = 11

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

#define DEF_TEMPLATE(TYPE_NAME, TYPE, EXECUTION_SPACE, LAYOUT)                                                                                                                     \
    template<>                                                                                                                                                                     \
    struct NdArrayTraits<TYPE, Kokkos::EXECUTION_SPACE, Kokkos::Layout##LAYOUT, 0>                                                                                                 \
    {                                                                                                                                                                              \
        static constexpr DataTypeKind       data_type       = DataTypeKind::TYPE_NAME;                                                                                             \
        static constexpr uint16             rank            = 0;                                                                                                                   \
        static constexpr LayoutKind         layout          = LayoutKind::LAYOUT;                                                                                                  \
        static constexpr ExecutionSpaceKind execution_space = ExecutionSpaceKind::EXECUTION_SPACE;                                                                                 \
    };                                                                                                                                                                             \
    template<>                                                                                                                                                                     \
    struct NdArrayTraits<TYPE, Kokkos::EXECUTION_SPACE, Kokkos::Layout##LAYOUT, 1>                                                                                                 \
    {                                                                                                                                                                              \
        static constexpr DataTypeKind       data_type       = DataTypeKind::TYPE_NAME;                                                                                             \
        static constexpr uint16             rank            = 1;                                                                                                                   \
        static constexpr LayoutKind         layout          = LayoutKind::LAYOUT;                                                                                                  \
        static constexpr ExecutionSpaceKind execution_space = ExecutionSpaceKind::EXECUTION_SPACE;                                                                                 \
    };                                                                                                                                                                             \
    template<>                                                                                                                                                                     \
    struct NdArrayTraits<TYPE, Kokkos::EXECUTION_SPACE, Kokkos::Layout##LAYOUT, 2>                                                                                                 \
    {                                                                                                                                                                              \
        static constexpr DataTypeKind       data_type       = DataTypeKind::TYPE_NAME;                                                                                             \
        static constexpr uint16             rank            = 2;                                                                                                                   \
        static constexpr LayoutKind         layout          = LayoutKind::LAYOUT;                                                                                                  \
        static constexpr ExecutionSpaceKind execution_space = ExecutionSpaceKind::EXECUTION_SPACE;                                                                                 \
    };                                                                                                                                                                             \
    template<>                                                                                                                                                                     \
    struct NdArrayTraits<TYPE, Kokkos::EXECUTION_SPACE, Kokkos::Layout##LAYOUT, 3>                                                                                                 \
    {                                                                                                                                                                              \
        static constexpr DataTypeKind       data_type       = DataTypeKind::TYPE_NAME;                                                                                             \
        static constexpr uint16             rank            = 3;                                                                                                                   \
        static constexpr LayoutKind         layout          = LayoutKind::LAYOUT;                                                                                                  \
        static constexpr ExecutionSpaceKind execution_space = ExecutionSpaceKind::EXECUTION_SPACE;                                                                                 \
    };                                                                                                                                                                             \
    template<>                                                                                                                                                                     \
    struct NdArrayTraits<TYPE, Kokkos::EXECUTION_SPACE, Kokkos::Layout##LAYOUT, 4>                                                                                                 \
    {                                                                                                                                                                              \
        static constexpr DataTypeKind       data_type       = DataTypeKind::TYPE_NAME;                                                                                             \
        static constexpr uint16             rank            = 4;                                                                                                                   \
        static constexpr LayoutKind         layout          = LayoutKind::LAYOUT;                                                                                                  \
        static constexpr ExecutionSpaceKind execution_space = ExecutionSpaceKind::EXECUTION_SPACE;                                                                                 \
    };                                                                                                                                                                             \
    template<>                                                                                                                                                                     \
    struct NdArrayTraits<TYPE, Kokkos::EXECUTION_SPACE, Kokkos::Layout##LAYOUT, 5>                                                                                                 \
    {                                                                                                                                                                              \
        static constexpr DataTypeKind       data_type       = DataTypeKind::TYPE_NAME;                                                                                             \
        static constexpr uint16             rank            = 5;                                                                                                                   \
        static constexpr LayoutKind         layout          = LayoutKind::LAYOUT;                                                                                                  \
        static constexpr ExecutionSpaceKind execution_space = ExecutionSpaceKind::EXECUTION_SPACE;                                                                                 \
    };                                                                                                                                                                             \
    template<>                                                                                                                                                                     \
    struct NdArrayTraits<TYPE, Kokkos::EXECUTION_SPACE, Kokkos::Layout##LAYOUT, 6>                                                                                                 \
    {                                                                                                                                                                              \
        static constexpr DataTypeKind       data_type       = DataTypeKind::TYPE_NAME;                                                                                             \
        static constexpr uint16             rank            = 6;                                                                                                                   \
        static constexpr LayoutKind         layout          = LayoutKind::LAYOUT;                                                                                                  \
        static constexpr ExecutionSpaceKind execution_space = ExecutionSpaceKind::EXECUTION_SPACE;                                                                                 \
    };                                                                                                                                                                             \
    template<>                                                                                                                                                                     \
    struct NdArrayTraits<TYPE, Kokkos::EXECUTION_SPACE, Kokkos::Layout##LAYOUT, 7>                                                                                                 \
    {                                                                                                                                                                              \
        static constexpr DataTypeKind       data_type       = DataTypeKind::TYPE_NAME;                                                                                             \
        static constexpr uint16             rank            = 7;                                                                                                                   \
        static constexpr LayoutKind         layout          = LayoutKind::LAYOUT;                                                                                                  \
        static constexpr ExecutionSpaceKind execution_space = ExecutionSpaceKind::EXECUTION_SPACE;                                                                                 \
    };                                                                                                                                                                             \
    template<>                                                                                                                                                                     \
    struct NdArrayTraits<TYPE, Kokkos::EXECUTION_SPACE, Kokkos::Layout##LAYOUT, 8>                                                                                                 \
    {                                                                                                                                                                              \
        static constexpr DataTypeKind       data_type       = DataTypeKind::TYPE_NAME;                                                                                             \
        static constexpr uint16             rank            = 8;                                                                                                                   \
        static constexpr LayoutKind         layout          = LayoutKind::LAYOUT;                                                                                                  \
        static constexpr ExecutionSpaceKind execution_space = ExecutionSpaceKind::EXECUTION_SPACE;                                                                                 \
    };

#define TEMPLATE(DEF, EXECUTION_SPACE, LAYOUT)                                                                                                                                     \
    DEF(Single, float, EXECUTION_SPACE, LAYOUT)                                                                                                                                    \
    DEF(Double, double, EXECUTION_SPACE, LAYOUT)                                                                                                                                   \
    DEF(Bool, bool, EXECUTION_SPACE, LAYOUT)                                                                                                                                       \
    DEF(Int8, int8, EXECUTION_SPACE, LAYOUT)                                                                                                                                       \
    DEF(UInt8, uint8, EXECUTION_SPACE, LAYOUT)                                                                                                                                     \
    DEF(Int16, int16, EXECUTION_SPACE, LAYOUT)                                                                                                                                     \
    DEF(UInt16, uint16, EXECUTION_SPACE, LAYOUT)                                                                                                                                   \
    DEF(Int32, int32, EXECUTION_SPACE, LAYOUT)                                                                                                                                     \
    DEF(UInt32, uint32, EXECUTION_SPACE, LAYOUT)                                                                                                                                   \
    DEF(Int64, int64, EXECUTION_SPACE, LAYOUT)                                                                                                                                     \
    DEF(UInt64, uint64, EXECUTION_SPACE, LAYOUT)                                                                                                                                   \
    DEF(Char, wchar_t, EXECUTION_SPACE, LAYOUT)

TEMPLATE(DEF_TEMPLATE, Serial, Right)
TEMPLATE(DEF_TEMPLATE, OpenMP, Right)
TEMPLATE(DEF_TEMPLATE, Cuda, Right)
TEMPLATE(DEF_TEMPLATE, Serial, Left)
TEMPLATE(DEF_TEMPLATE, OpenMP, Left)
TEMPLATE(DEF_TEMPLATE, Cuda, Left)

template<StringType TString>
__inline static bool isNullTerminating(const TString& str)
{
    return str[str.size() - 1] == '\0';
}

__inline static bool isNullTerminating(const int& length, const char* bytes)
{
    return bytes[length - 1] == '\0';
}

struct KOKKOS_NET_API_EXPORT NativeString
{
    int64 Length;

    int8* Bytes;

    __inline constexpr NativeString() : Length(0), Bytes(nullptr) {}

    __inline constexpr NativeString(const NativeString& other) : Length(other.Length), Bytes(other.Bytes) {}

    __inline constexpr NativeString(NativeString&& other) noexcept : Length(other.Length), Bytes(other.Bytes) {}

    __inline constexpr NativeString& operator=(const NativeString& other)
    {
        if (this == &other)
        {
            return *this;
        }
        Length = other.Length;
        Bytes  = other.Bytes;
        return *this;
    }

    __inline constexpr NativeString& operator=(NativeString&& other) noexcept
    {
        if (this == &other)
        {
            return *this;
        }
        Length = other.Length;
        Bytes  = other.Bytes;
        return *this;
    }

    explicit __inline NativeString(const std::string& str) :
        Length(isNullTerminating(str) ? str.size() : str.size() + 1),
        Bytes((int8*)Kokkos::kokkos_malloc<Kokkos::Serial::memory_space>(Length))
    {
        // memcpy(const_cast<int8*>(Bytes), str.c_str(), isNullTerminating(str) ? str.size() - 1 : str.size());

        int index = 0;

        while (str[index] != '\0')
        {
            Bytes[index] = str[index];
            ++index;
        }
    }

    explicit __inline NativeString(const size_t& length, const char* bytes) :
        Length(isNullTerminating(length, bytes) ? length : length + 1),
        Bytes((int8*)Kokkos::kokkos_malloc<Kokkos::Serial::memory_space>(Length))
    {
        // memcpy(const_cast<int8*>(Bytes), bytes, isNullTerminating(length, bytes) ? length - 1 : length);

        int index = 0;

        if (bytes != nullptr)
        {
            while (bytes[index] != '\0')
            {
                Bytes[index] = bytes[index];
                ++index;
            }
        }
    }

    __inline ~NativeString()
    {
        Kokkos::kokkos_free<Kokkos::Serial::memory_space>(Bytes);
    }

    __inline std::string ToString() const
    {
        return std::string(Bytes, isNullTerminating(Length, Bytes) ? Length - 1 : Length);
    }
};

#define NDARRAY_MAX_RANK 8

//#pragma pack(8)
struct __declspec(align(2)) NdArray
{
    DataTypeKind       data_type;
    uint16             rank;
    LayoutKind         layout;
    ExecutionSpaceKind execution_space;
    uint64             dims[NDARRAY_MAX_RANK];
    uint64             strides[NDARRAY_MAX_RANK];
    void*              data;
    NativeString       label;

    __inline NdArray(const NdArray& other) :
        data_type(other.data_type),
        rank(other.rank),
        layout(other.layout),
        execution_space(other.execution_space),
        dims{},
        strides{},
        data(other.data),
        label(other.label)
    {
        dims[0] = other.dims[0];
        dims[1] = other.dims[1];
        dims[2] = other.dims[2];
        dims[3] = other.dims[3];
        dims[4] = other.dims[4];
        dims[5] = other.dims[5];
        dims[6] = other.dims[6];
        dims[7] = other.dims[7];

        strides[0] = other.strides[0];
        strides[1] = other.strides[1];
        strides[2] = other.strides[2];
        strides[3] = other.strides[3];
        strides[4] = other.strides[4];
        strides[5] = other.strides[5];
        strides[6] = other.strides[6];
        strides[7] = other.strides[7];
    }

    __inline NdArray(NdArray&& other) noexcept :
        data_type(other.data_type),
        rank(other.rank),
        layout(other.layout),
        execution_space(other.execution_space),
        dims{},
        strides{},
        data(other.data),
        label(std::move(other.label))
    {
        dims[0] = other.dims[0];
        dims[1] = other.dims[1];
        dims[2] = other.dims[2];
        dims[3] = other.dims[3];
        dims[4] = other.dims[4];
        dims[5] = other.dims[5];
        dims[6] = other.dims[6];
        dims[7] = other.dims[7];

        strides[0] = other.strides[0];
        strides[1] = other.strides[1];
        strides[2] = other.strides[2];
        strides[3] = other.strides[3];
        strides[4] = other.strides[4];
        strides[5] = other.strides[5];
        strides[6] = other.strides[6];
        strides[7] = other.strides[7];
    }

    __inline NdArray& operator=(const NdArray& other)
    {
        if (this == &other)
        {
            return *this;
        }
        data_type       = other.data_type;
        rank            = other.rank;
        layout          = other.layout;
        execution_space = other.execution_space;
        data            = other.data;
        label           = other.label;

        dims[0] = other.dims[0];
        dims[1] = other.dims[1];
        dims[2] = other.dims[2];
        dims[3] = other.dims[3];
        dims[4] = other.dims[4];
        dims[5] = other.dims[5];
        dims[6] = other.dims[6];
        dims[7] = other.dims[7];

        strides[0] = other.strides[0];
        strides[1] = other.strides[1];
        strides[2] = other.strides[2];
        strides[3] = other.strides[3];
        strides[4] = other.strides[4];
        strides[5] = other.strides[5];
        strides[6] = other.strides[6];
        strides[7] = other.strides[7];

        return *this;
    }

    __inline NdArray& operator=(NdArray&& other) noexcept
    {
        if (this == &other)
        {
            return *this;
        }
        data_type       = other.data_type;
        rank            = other.rank;
        layout          = other.layout;
        execution_space = other.execution_space;
        data            = other.data;
        label           = std::move(other.label);

        dims[0] = other.dims[0];
        dims[1] = other.dims[1];
        dims[2] = other.dims[2];
        dims[3] = other.dims[3];
        dims[4] = other.dims[4];
        dims[5] = other.dims[5];
        dims[6] = other.dims[6];
        dims[7] = other.dims[7];

        strides[0] = other.strides[0];
        strides[1] = other.strides[1];
        strides[2] = other.strides[2];
        strides[3] = other.strides[3];
        strides[4] = other.strides[4];
        strides[5] = other.strides[5];
        strides[6] = other.strides[6];
        strides[7] = other.strides[7];

        return *this;
    }

        explicit __inline NdArray(const DataTypeKind       data_type,
                              const uint16             rank,
                              const LayoutKind         layout,
                              const ExecutionSpaceKind execution_space,
                              void* const              data,
                              const std::string&       label) :
        data_type(data_type),
        rank(rank),
        layout(layout),
        execution_space(execution_space),
        dims{},
        strides{},
        data(data),
        label(label)
    {
    }

    __inline explicit NdArray(const DataTypeKind       data_type,
                              const uint16             rank,
                              const LayoutKind         layout,
                              const ExecutionSpaceKind execution_space,
                              void* const              data,
                              const NativeString&       label) :
        data_type(data_type),
        rank(rank),
        layout(layout),
        execution_space(execution_space),
        dims{},
        strides{},
        data(data),
        label(label)
    {
    }
};
//#pragma pack(8)

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

#define TEMPLATE(DEF, EXECUTION_SPACE)                                                                                                                                             \
    DEF(Single, float, EXECUTION_SPACE)                                                                                                                                            \
    DEF(Double, double, EXECUTION_SPACE)                                                                                                                                           \
    DEF(Bool, bool, EXECUTION_SPACE)                                                                                                                                               \
    DEF(Int8, int8, EXECUTION_SPACE)                                                                                                                                               \
    DEF(UInt8, uint8, EXECUTION_SPACE)                                                                                                                                             \
    DEF(Int16, int16, EXECUTION_SPACE)                                                                                                                                             \
    DEF(UInt16, uint16, EXECUTION_SPACE)                                                                                                                                           \
    DEF(Int32, int32, EXECUTION_SPACE)                                                                                                                                             \
    DEF(UInt32, uint32, EXECUTION_SPACE)                                                                                                                                           \
    DEF(Int64, int64, EXECUTION_SPACE)                                                                                                                                             \
    DEF(UInt64, uint64, EXECUTION_SPACE)                                                                                                                                           \
    DEF(Char, wchar_t, EXECUTION_SPACE)

#define DEF_TEMPLATE(TYPE_NAME, TYPE, EXECUTION_SPACE)                                                                                                                             \
    template<>                                                                                                                                                                     \
    struct ViewBuilder<DataTypeKind::TYPE_NAME, 0, ExecutionSpaceKind::EXECUTION_SPACE>                                                                                            \
    {                                                                                                                                                                              \
        using ViewType = Kokkos::View<TYPE, typename Kokkos::EXECUTION_SPACE::array_layout, Kokkos::EXECUTION_SPACE>;                                                              \
    };                                                                                                                                                                             \
    template<>                                                                                                                                                                     \
    struct ViewBuilder<DataTypeKind::TYPE_NAME, 1, ExecutionSpaceKind::EXECUTION_SPACE>                                                                                            \
    {                                                                                                                                                                              \
        using ViewType = Kokkos::View<TYPE*, typename Kokkos::EXECUTION_SPACE::array_layout, Kokkos::EXECUTION_SPACE>;                                                             \
    };                                                                                                                                                                             \
    template<>                                                                                                                                                                     \
    struct ViewBuilder<DataTypeKind::TYPE_NAME, 2, ExecutionSpaceKind::EXECUTION_SPACE>                                                                                            \
    {                                                                                                                                                                              \
        using ViewType = Kokkos::View<TYPE**, typename Kokkos::EXECUTION_SPACE::array_layout, Kokkos::EXECUTION_SPACE>;                                                            \
    };                                                                                                                                                                             \
    template<>                                                                                                                                                                     \
    struct ViewBuilder<DataTypeKind::TYPE_NAME, 3, ExecutionSpaceKind::EXECUTION_SPACE>                                                                                            \
    {                                                                                                                                                                              \
        using ViewType = Kokkos::View<TYPE***, typename Kokkos::EXECUTION_SPACE::array_layout, Kokkos::EXECUTION_SPACE>;                                                           \
    };                                                                                                                                                                             \
    template<>                                                                                                                                                                     \
    struct ViewBuilder<DataTypeKind::TYPE_NAME, 4, ExecutionSpaceKind::EXECUTION_SPACE>                                                                                            \
    {                                                                                                                                                                              \
        using ViewType = Kokkos::View<TYPE****, typename Kokkos::EXECUTION_SPACE::array_layout, Kokkos::EXECUTION_SPACE>;                                                          \
    };                                                                                                                                                                             \
    template<>                                                                                                                                                                     \
    struct ViewBuilder<DataTypeKind::TYPE_NAME, 5, ExecutionSpaceKind::EXECUTION_SPACE>                                                                                            \
    {                                                                                                                                                                              \
        using ViewType = Kokkos::View<TYPE*****, typename Kokkos::EXECUTION_SPACE::array_layout, Kokkos::EXECUTION_SPACE>;                                                         \
    };                                                                                                                                                                             \
    template<>                                                                                                                                                                     \
    struct ViewBuilder<DataTypeKind::TYPE_NAME, 6, ExecutionSpaceKind::EXECUTION_SPACE>                                                                                            \
    {                                                                                                                                                                              \
        using ViewType = Kokkos::View<TYPE******, typename Kokkos::EXECUTION_SPACE::array_layout, Kokkos::EXECUTION_SPACE>;                                                        \
    };                                                                                                                                                                             \
    template<>                                                                                                                                                                     \
    struct ViewBuilder<DataTypeKind::TYPE_NAME, 7, ExecutionSpaceKind::EXECUTION_SPACE>                                                                                            \
    {                                                                                                                                                                              \
        using ViewType = Kokkos::View<TYPE*******, typename Kokkos::EXECUTION_SPACE::array_layout, Kokkos::EXECUTION_SPACE>;                                                       \
    };                                                                                                                                                                             \
    template<>                                                                                                                                                                     \
    struct ViewBuilder<DataTypeKind::TYPE_NAME, 8, ExecutionSpaceKind::EXECUTION_SPACE>                                                                                            \
    {                                                                                                                                                                              \
        using ViewType = Kokkos::View<TYPE********, typename Kokkos::EXECUTION_SPACE::array_layout, Kokkos::EXECUTION_SPACE>;                                                      \
    };

TEMPLATE(DEF_TEMPLATE, Serial)
TEMPLATE(DEF_TEMPLATE, OpenMP)
TEMPLATE(DEF_TEMPLATE, Cuda)

#undef TEMPLATE
#undef DEF_TEMPLATE

template<ExecutionSpaceKind TExecutionSpace>
struct ToTrait;

template<>
struct ToTrait<ExecutionSpaceKind::Serial>
{
    using ExecutionSpace = Kokkos::Serial;
};

template<>
struct ToTrait<ExecutionSpaceKind::OpenMP>
{
    using ExecutionSpace = Kokkos::OpenMP;
};

template<>
struct ToTrait<ExecutionSpaceKind::Cuda>
{
    using ExecutionSpace = Kokkos::Cuda;
};

// rank 1
template<typename I0>
KOKKOS_INLINE_FUNCTION static constexpr size_type view_offset_right(CONST(I0) i0)
{
    return i0;
}

// rank 2
template<typename I0, typename I1>
KOKKOS_INLINE_FUNCTION static constexpr size_type view_offset_right(CONST(I0) i0, CONST(I0) N0, CONST(I1) i1)
{
    return i0 + N0 * i1;
}

// rank 3
template<typename I0, typename I1, typename I2>
KOKKOS_INLINE_FUNCTION static constexpr size_type view_offset_right(CONST(I0) i0, CONST(I0) N0, CONST(I1) i1, CONST(I1) N1, CONST(I2) i2)
{
    return i0 + N0 * (i1 + N1 * i2);
}

// rank 4
template<typename I0, typename I1, typename I2, typename I3>
KOKKOS_INLINE_FUNCTION static constexpr size_type view_offset_right(CONST(I0) i0, CONST(I0) N0, CONST(I1) i1, CONST(I1) N1, CONST(I2) i2, CONST(I2) N2, CONST(I3) i3)
{
    return i0 + N0 * (i1 + N1 * (i2 + N2 * i3));
}

// rank 5
template<typename I0, typename I1, typename I2, typename I3, typename I4>
KOKKOS_INLINE_FUNCTION static constexpr size_type view_offset_right(CONST(I0) i0, CONST(I0) N0, CONST(I1) i1, CONST(I1) N1, CONST(I2) i2, CONST(I2) N2, CONST(I3) i3, CONST(I3) N3, CONST(I4) i4)
{
    return i0 + N0 * (i1 + N1 * (i2 + N2 * (i3 + N3 * i4)));
}

// rank 6
template<typename I0, typename I1, typename I2, typename I3, typename I4, typename I5>
KOKKOS_INLINE_FUNCTION static constexpr size_type view_offset_right(CONST(I0) i0,
                                                                    CONST(I0) N0,
                                                                    CONST(I1) i1,
                                                                    CONST(I1) N1,
                                                                    CONST(I2) i2,
                                                                    CONST(I2) N2,
                                                                    CONST(I3) i3,
                                                                    CONST(I3) N3,
                                                                    CONST(I4) i4,
                                                                    CONST(I4) N4,
                                                                    CONST(I5) i5)
{
    return i0 + N0 * (i1 + N1 * (i2 + N2 * (i3 + N3 * (i4 + N4 * i5))));
}

// rank 7
template<typename I0, typename I1, typename I2, typename I3, typename I4, typename I5, typename I6>
KOKKOS_INLINE_FUNCTION static constexpr size_type view_offset_right(CONST(I0) i0,
                                                                    CONST(I0) N0,
                                                                    CONST(I1) i1,
                                                                    CONST(I1) N1,
                                                                    CONST(I2) i2,
                                                                    CONST(I2) N2,
                                                                    CONST(I3) i3,
                                                                    CONST(I3) N3,
                                                                    CONST(I4) i4,
                                                                    CONST(I4) N4,
                                                                    CONST(I5) i5,
                                                                    CONST(I5) N5,
                                                                    CONST(I6) i6)
{
    return i0 + N0 * (i1 + N1 * (i2 + N2 * (i3 + N3 * (i4 + N4 * (i5 + N5 * i6)))));
}

// rank 8
template<typename I0, typename I1, typename I2, typename I3, typename I4, typename I5, typename I6, typename I7>
KOKKOS_INLINE_FUNCTION static constexpr size_type view_offset_right(CONST(I0) i0,
                                                                    CONST(I0) N0,
                                                                    CONST(I1) i1,
                                                                    CONST(I1) N1,
                                                                    CONST(I2) i2,
                                                                    CONST(I2) N2,
                                                                    CONST(I3) i3,
                                                                    CONST(I3) N3,
                                                                    CONST(I4) i4,
                                                                    CONST(I4) N4,
                                                                    CONST(I5) i5,
                                                                    CONST(I5) N5,
                                                                    CONST(I6) i6,
                                                                    CONST(I6) N6,
                                                                    CONST(I7) i7)
{
    return i0 + N0 * (i1 + N1 * (i2 + N2 * (i3 + N3 * (i4 + N4 * (i5 + N5 * (i6 + N6 * i7))))));
}

// rank 1
template<typename I0>
KOKKOS_INLINE_FUNCTION static constexpr size_type view_offset_left(CONST(I0) i0)
{
    return i0;
}

// rank 2
template<typename I0, typename I1>
KOKKOS_INLINE_FUNCTION static constexpr size_type view_offset_left(CONST(I0) i0, CONST(I1) N1, CONST(I1) i1)
{
    return i1 + (N1 * i0);
}

// rank 3
template<typename I0, typename I1, typename I2>
KOKKOS_INLINE_FUNCTION static constexpr size_type view_offset_left(CONST(I0) i0, CONST(I1) N1, CONST(I1) i1, CONST(I2) N2, CONST(I2) i2)
{
    return i2 + N2 * (i1 + (N1 * i0));
}

// rank 4
template<typename I0, typename I1, typename I2, typename I3>
KOKKOS_INLINE_FUNCTION static constexpr size_type view_offset_left(CONST(I0) i0, CONST(I1) N1, CONST(I1) i1, CONST(I2) N2, CONST(I2) i2, CONST(I3) N3, CONST(I3) i3)
{
    return i3 + N3 * (i2 + N2 * (i1 + (N1 * i0)));
}

// rank 5
template<typename I0, typename I1, typename I2, typename I3, typename I4>
KOKKOS_INLINE_FUNCTION static constexpr size_type view_offset_left(CONST(I0) i0, CONST(I1) N1, CONST(I1) i1, CONST(I2) N2, CONST(I2) i2, CONST(I3) N3, CONST(I3) i3, CONST(I4) N4, CONST(I4) i4)
{
    return i4 + N4 * (i3 + N3 * (i2 + N2 * (i1 + (N1 * i0))));
}

// rank 6
template<typename I0, typename I1, typename I2, typename I3, typename I4, typename I5>
KOKKOS_INLINE_FUNCTION static constexpr size_type view_offset_left(CONST(I0) i0,
                                                                   CONST(I1) N1,
                                                                   CONST(I1) i1,
                                                                   CONST(I2) N2,
                                                                   CONST(I2) i2,
                                                                   CONST(I3) N3,
                                                                   CONST(I3) i3,
                                                                   CONST(I4) N4,
                                                                   CONST(I4) i4,
                                                                   CONST(I5) N5,
                                                                   CONST(I5) i5)
{
    return i5 + N5 * (i4 + N4 * (i3 + N3 * (i2 + N2 * (i1 + (N1 * i0)))));
}

// rank 7
template<typename I0, typename I1, typename I2, typename I3, typename I4, typename I5, typename I6>
KOKKOS_INLINE_FUNCTION static constexpr size_type view_offset_left(CONST(I0) i0,
                                                                   CONST(I1) N1,
                                                                   CONST(I1) i1,
                                                                   CONST(I2) N2,
                                                                   CONST(I2) i2,
                                                                   CONST(I3) N3,
                                                                   CONST(I3) i3,
                                                                   CONST(I4) N4,
                                                                   CONST(I4) i4,
                                                                   CONST(I5) N5,
                                                                   CONST(I5) i5,
                                                                   CONST(I6) N6,
                                                                   CONST(I6) i6)
{
    return i6 + N6 * (i5 + N5 * (i4 + N4 * (i3 + N3 * (i2 + N2 * (i1 + (N1 * i0))))));
}

// rank 8
template<typename I0, typename I1, typename I2, typename I3, typename I4, typename I5, typename I6, typename I7>
KOKKOS_INLINE_FUNCTION static constexpr size_type view_offset_left(CONST(I0) i0,
                                                                   CONST(I1) N1,
                                                                   CONST(I1) i1,
                                                                   CONST(I2) N2,
                                                                   CONST(I2) i2,
                                                                   CONST(I3) N3,
                                                                   CONST(I3) i3,
                                                                   CONST(I4) N4,
                                                                   CONST(I4) i4,
                                                                   CONST(I5) N5,
                                                                   CONST(I5) i5,
                                                                   CONST(I6) N6,
                                                                   CONST(I6) i6,
                                                                   CONST(I7) N7,
                                                                   CONST(I7) i7)
{
    return i7 + N7 * (i6 + N6 * (i5 + N5 * (i4 + N4 * (i3 + N3 * (i2 + N2 * (i1 + (N1 * i0)))))));
}

// rank 2
template<typename I0, typename I1>
KOKKOS_INLINE_FUNCTION static constexpr size_type view_triangle_offset_right(CONST(I0) i0, CONST(I1) i1)
{
    return ((i1 * (i1 + 1)) / 2) + i0;
}

// rank 3
template<typename I0, typename I1, typename I2>
KOKKOS_INLINE_FUNCTION static constexpr size_type view_triangle_offset_right(CONST(I0) i0, CONST(I1) i1, CONST(I2) i2)
{
    return ((i2 * (i2 + 1) * (i2 + 2)) / 6) + ((i1 * (i1 + 1)) / 2) + i0;
}

// rank 4
template<typename I0, typename I1, typename I2, typename I3>
KOKKOS_INLINE_FUNCTION static constexpr size_type view_triangle_offset_right(CONST(I0) i0, CONST(I1) i1, CONST(I2) i2, CONST(I3) i3)
{
    return ((i3 * (i3 + 1) * (i3 + 2) * (i3 + 3)) / 24) + ((i2 * (i2 + 1) * (i2 + 2)) / 6) + ((i1 * (i1 + 1)) / 2) + i0;
}
