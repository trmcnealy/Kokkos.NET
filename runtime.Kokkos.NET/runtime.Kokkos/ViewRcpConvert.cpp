
#include "runtime.Kokkos/KokkosApi.h"

#include <iostream>

#define TEMPLATE_RANK(DEF, EXECUTION_SPACE, RANK)                                                                                                                                  \
    DEF(Single, float, EXECUTION_SPACE, RANK)                                                                                                                                      \
    DEF(Double, double, EXECUTION_SPACE, RANK)                                                                                                                                     \
    DEF(Bool, bool, EXECUTION_SPACE, RANK)                                                                                                                                         \
    DEF(Int8, int8, EXECUTION_SPACE, RANK)                                                                                                                                         \
    DEF(UInt8, uint8, EXECUTION_SPACE, RANK)                                                                                                                                       \
    DEF(Int16, int16, EXECUTION_SPACE, RANK)                                                                                                                                       \
    DEF(UInt16, uint16, EXECUTION_SPACE, RANK)                                                                                                                                     \
    DEF(Int32, int32, EXECUTION_SPACE, RANK)                                                                                                                                       \
    DEF(UInt32, uint32, EXECUTION_SPACE, RANK)                                                                                                                                     \
    DEF(Int64, int64, EXECUTION_SPACE, RANK)                                                                                                                                       \
    DEF(UInt64, uint64, EXECUTION_SPACE, RANK)                                                                                                                                     \
    DEF(Char, wchar_t, EXECUTION_SPACE, RANK)

#define DEF_TEMPLATE(TYPE_NAME, TYPE, EXECUTION_SPACE, RANK)                                                                                                                       \
    case DataTypeKind::TYPE_NAME:                                                                                                                                                  \
    {                                                                                                                                                                              \
        switch (layout)                                                                                                                                                            \
        {                                                                                                                                                                          \
            case LayoutKind::Right:                                                                                                                                                \
            {                                                                                                                                                                      \
                *ndArray = RcpViewToNdArrayRank##RANK<TYPE, Kokkos::EXECUTION_SPACE, Kokkos::LayoutRight>(instance);                                                               \
                break;                                                                                                                                                             \
            }                                                                                                                                                                      \
            case LayoutKind::Left:                                                                                                                                                 \
            {                                                                                                                                                                      \
                *ndArray = RcpViewToNdArrayRank##RANK<TYPE, Kokkos::EXECUTION_SPACE, Kokkos::LayoutLeft>(instance);                                                                \
                break;                                                                                                                                                             \
            }                                                                                                                                                                      \
            case LayoutKind::Unknown:                                                                                                                                              \
            default:                                                                                                                                                               \
            {                                                                                                                                                                      \
                std::cout << "RcpViewToNdArray, LayoutKind is not supported." << std::endl;                                                                                        \
                break;                                                                                                                                                             \
            }                                                                                                                                                                      \
        }                                                                                                                                                                          \
        break;                                                                                                                                                                     \
    }

template<typename DataType, class ExecutionSpace, typename Layout>
__inline static NdArray RcpViewToNdArrayRank0(void* instance) noexcept
{
    typedef Kokkos::View<DataType, Layout, ExecutionSpace> view_type;

    typedef NdArrayTraits<typename view_type::traits::value_type, typename view_type::traits::execution_space, typename view_type::traits::array_layout, 0> ndarray_traits;

    view_type& view = *(*reinterpret_cast<Teuchos::RCP<view_type>*>(instance));

    const std::string label = view.label();

    NdArray ndArray(ndarray_traits::data_type,
                    0,
                    ndarray_traits::layout,
                    ndarray_traits::execution_space,
                    view.data(),
                    label);

    return ndArray;
}

template<typename DataType, class ExecutionSpace, typename Layout>
__inline static NdArray RcpViewToNdArrayRank1(void* instance) noexcept
{
    typedef Kokkos::View<DataType*, Layout, ExecutionSpace> view_type;

    typedef NdArrayTraits<typename view_type::traits::value_type, typename view_type::traits::execution_space, typename view_type::traits::array_layout, 1> ndarray_traits;

    view_type& view = *(*reinterpret_cast<Teuchos::RCP<view_type>*>(instance));

    const std::string label = view.label();

    NdArray ndArray(ndarray_traits::data_type,
                    1,
                    ndarray_traits::layout,
                    ndarray_traits::execution_space,
                    view.data(),
                    label);

    return ndArray;
}

template<typename DataType, class ExecutionSpace, typename Layout>
__inline static NdArray RcpViewToNdArrayRank2(void* instance) noexcept
{
    typedef Kokkos::View<DataType**, Layout, ExecutionSpace> view_type;

    typedef NdArrayTraits<typename view_type::traits::value_type, typename view_type::traits::execution_space, typename view_type::traits::array_layout, 2> ndarray_traits;

    view_type& view = *(*reinterpret_cast<Teuchos::RCP<view_type>*>(instance));

    const std::string label = view.label();

    NdArray ndArray(ndarray_traits::data_type,
                    2,
                    ndarray_traits::layout,
                    ndarray_traits::execution_space,
                    view.data(),
                    label);

    ndArray.dims[0] = view.extent(0);
    ndArray.dims[1] = view.extent(1);

    ndArray.strides[0] = view.stride(0);
    ndArray.strides[1] = view.stride(1);

    return ndArray;
}

template<typename DataType, class ExecutionSpace, typename Layout>
__inline static NdArray RcpViewToNdArrayRank3(void* instance) noexcept
{
    typedef Kokkos::View<DataType***, Layout, ExecutionSpace> view_type;

    typedef NdArrayTraits<typename view_type::traits::value_type, typename view_type::traits::execution_space, typename view_type::traits::array_layout, 3> ndarray_traits;

    view_type& view = *(*reinterpret_cast<Teuchos::RCP<view_type>*>(instance));

    const std::string label = view.label();

    NdArray ndArray(ndarray_traits::data_type,
                    3,
                    ndarray_traits::layout,
                    ndarray_traits::execution_space,
                    view.data(),
                    label);

    ndArray.dims[0] = view.extent(0);
    ndArray.dims[1] = view.extent(1);
    ndArray.dims[2] = view.extent(2);

    ndArray.strides[0] = view.stride(0);
    ndArray.strides[1] = view.stride(1);
    ndArray.strides[2] = view.stride(2);

    return ndArray;
}

template<typename DataType, class ExecutionSpace, typename Layout>
__inline static NdArray RcpViewToNdArrayRank4(void* instance) noexcept
{
    typedef Kokkos::View<DataType****, Layout, ExecutionSpace> view_type;

    typedef NdArrayTraits<typename view_type::traits::value_type, typename view_type::traits::execution_space, typename view_type::traits::array_layout, 4> ndarray_traits;

    view_type& view = *(*reinterpret_cast<Teuchos::RCP<view_type>*>(instance));

    const std::string label = view.label();

    NdArray ndArray(ndarray_traits::data_type,
                    4,
                    ndarray_traits::layout,
                    ndarray_traits::execution_space,
                    view.data(),
                    label);

    ndArray.dims[0] = view.extent(0);
    ndArray.dims[1] = view.extent(1);
    ndArray.dims[2] = view.extent(2);
    ndArray.dims[3] = view.extent(3);

    ndArray.strides[0] = view.stride(0);
    ndArray.strides[1] = view.stride(1);
    ndArray.strides[2] = view.stride(2);
    ndArray.strides[3] = view.stride(3);

    return ndArray;
}

template<typename DataType, class ExecutionSpace, typename Layout>
__inline static NdArray RcpViewToNdArrayRank5(void* instance) noexcept
{
    typedef Kokkos::View<DataType*****, Layout, ExecutionSpace> view_type;

    typedef NdArrayTraits<typename view_type::traits::value_type, typename view_type::traits::execution_space, typename view_type::traits::array_layout, 5> ndarray_traits;

    view_type& view = *(*reinterpret_cast<Teuchos::RCP<view_type>*>(instance));

    const std::string label = view.label();

    NdArray ndArray(ndarray_traits::data_type,
                    5,
                    ndarray_traits::layout,
                    ndarray_traits::execution_space,
                    view.data(),
                    label);

    ndArray.dims[0] = view.extent(0);
    ndArray.dims[1] = view.extent(1);
    ndArray.dims[2] = view.extent(2);
    ndArray.dims[3] = view.extent(3);
    ndArray.dims[4] = view.extent(4);

    ndArray.strides[0] = view.stride(0);
    ndArray.strides[1] = view.stride(1);
    ndArray.strides[2] = view.stride(2);
    ndArray.strides[3] = view.stride(3);
    ndArray.strides[4] = view.stride(4);

    return ndArray;
}

template<typename DataType, class ExecutionSpace, typename Layout>
__inline static NdArray RcpViewToNdArrayRank6(void* instance) noexcept
{
    typedef Kokkos::View<DataType******, Layout, ExecutionSpace> view_type;

    typedef NdArrayTraits<typename view_type::traits::value_type, typename view_type::traits::execution_space, typename view_type::traits::array_layout, 6> ndarray_traits;

    view_type& view = *(*reinterpret_cast<Teuchos::RCP<view_type>*>(instance));

    const std::string label = view.label();

    NdArray ndArray(ndarray_traits::data_type,
                    6,
                    ndarray_traits::layout,
                    ndarray_traits::execution_space,
                    view.data(),
                    label);

    ndArray.dims[0] = view.extent(0);
    ndArray.dims[1] = view.extent(1);
    ndArray.dims[2] = view.extent(2);
    ndArray.dims[3] = view.extent(3);
    ndArray.dims[4] = view.extent(4);
    ndArray.dims[5] = view.extent(5);

    ndArray.strides[0] = view.stride(0);
    ndArray.strides[1] = view.stride(1);
    ndArray.strides[2] = view.stride(2);
    ndArray.strides[3] = view.stride(3);
    ndArray.strides[4] = view.stride(4);
    ndArray.strides[5] = view.stride(5);

    return ndArray;
}

template<typename DataType, class ExecutionSpace, typename Layout>
__inline static NdArray RcpViewToNdArrayRank7(void* instance) noexcept
{
    typedef Kokkos::View<DataType*******, Layout, ExecutionSpace> view_type;

    typedef NdArrayTraits<typename view_type::traits::value_type, typename view_type::traits::execution_space, typename view_type::traits::array_layout, 7> ndarray_traits;

    view_type& view = *(*reinterpret_cast<Teuchos::RCP<view_type>*>(instance));

    const std::string label = view.label();

    NdArray ndArray(ndarray_traits::data_type,
                    7,
                    ndarray_traits::layout,
                    ndarray_traits::execution_space,
                    view.data(),
                    label);

    ndArray.dims[0] = view.extent(0);
    ndArray.dims[1] = view.extent(1);
    ndArray.dims[2] = view.extent(2);
    ndArray.dims[3] = view.extent(3);
    ndArray.dims[4] = view.extent(4);
    ndArray.dims[5] = view.extent(5);
    ndArray.dims[6] = view.extent(6);

    ndArray.strides[0] = view.stride(0);
    ndArray.strides[1] = view.stride(1);
    ndArray.strides[2] = view.stride(2);
    ndArray.strides[3] = view.stride(3);
    ndArray.strides[4] = view.stride(4);
    ndArray.strides[5] = view.stride(5);
    ndArray.strides[6] = view.stride(6);

    return ndArray;
}

template<typename DataType, class ExecutionSpace, typename Layout>
__inline static NdArray RcpViewToNdArrayRank8(void* instance) noexcept
{
    typedef Kokkos::View<DataType********, Layout, ExecutionSpace> view_type;

    typedef NdArrayTraits<typename view_type::traits::value_type, typename view_type::traits::execution_space, typename view_type::traits::array_layout, 8> ndarray_traits;

    view_type& view = *(*reinterpret_cast<Teuchos::RCP<view_type>*>(instance));

    const std::string label = view.label();

    NdArray ndArray(ndarray_traits::data_type,
                    8,
                    ndarray_traits::layout,
                    ndarray_traits::execution_space,
                    view.data(),
                    label);

    ndArray.dims[0] = view.extent(0);
    ndArray.dims[1] = view.extent(1);
    ndArray.dims[2] = view.extent(2);
    ndArray.dims[3] = view.extent(3);
    ndArray.dims[4] = view.extent(4);
    ndArray.dims[5] = view.extent(5);
    ndArray.dims[6] = view.extent(6);
    ndArray.dims[7] = view.extent(7);

    ndArray.strides[0] = view.stride(0);
    ndArray.strides[1] = view.stride(1);
    ndArray.strides[2] = view.stride(2);
    ndArray.strides[3] = view.stride(3);
    ndArray.strides[4] = view.stride(4);
    ndArray.strides[5] = view.stride(5);
    ndArray.strides[6] = view.stride(6);
    ndArray.strides[7] = view.stride(7);

    return ndArray;
}

void RcpViewToNdArray(void* instance, const ExecutionSpaceKind execution_space, const LayoutKind layout, const DataTypeKind data_type, const uint16 rank, NdArray* ndArray) noexcept
{
    switch (execution_space)
    {
        case ExecutionSpaceKind::Serial:
        {
            switch (rank)
            {
                case 0:
                {
                    switch (data_type)
                    {
                        TEMPLATE_RANK(DEF_TEMPLATE, Serial, 0)
                        case DataTypeKind::Unknown:
                        default:
                        {
                            std::cout << "RcpViewToNdArray::Serial, DataType is not supported." << std::endl;
                            break;
                        }
                    }
                    break;
                }
                case 1:
                {
                    switch (data_type)
                    {
                        TEMPLATE_RANK(DEF_TEMPLATE, Serial, 1)
                        case DataTypeKind::Unknown:
                        default:
                        {
                            std::cout << "RcpViewToNdArray::Serial, DataType is not supported." << std::endl;
                            break;
                        }
                    }
                    break;
                }
                case 2:
                {
                    switch (data_type)
                    {
                        TEMPLATE_RANK(DEF_TEMPLATE, Serial, 2)
                        case DataTypeKind::Unknown:
                        default:
                        {
                            std::cout << "RcpViewToNdArray::Serial, DataType is not supported." << std::endl;
                            break;
                        }
                    }
                    break;
                }
                case 3:
                {
                    switch (data_type)
                    {
                        TEMPLATE_RANK(DEF_TEMPLATE, Serial, 3)
                        case DataTypeKind::Unknown:
                        default:
                        {
                            std::cout << "RcpViewToNdArray::Serial, DataType is not supported." << std::endl;
                            break;
                        }
                    }
                    break;
                }
                case 4:
                {
                    switch (data_type)
                    {
                        TEMPLATE_RANK(DEF_TEMPLATE, Serial, 4)
                        case DataTypeKind::Unknown:
                        default:
                        {
                            std::cout << "RcpViewToNdArray::Serial, DataType is not supported." << std::endl;
                            break;
                        }
                    }
                    break;
                }
                case 5:
                {
                    switch (data_type)
                    {
                        TEMPLATE_RANK(DEF_TEMPLATE, Serial, 5)
                        case DataTypeKind::Unknown:
                        default:
                        {
                            std::cout << "RcpViewToNdArray::Serial, DataType is not supported." << std::endl;
                            break;
                        }
                    }
                    break;
                }
                case 6:
                {
                    switch (data_type)
                    {
                        TEMPLATE_RANK(DEF_TEMPLATE, Serial, 6)
                        case DataTypeKind::Unknown:
                        default:
                        {
                            std::cout << "RcpViewToNdArray::Serial, DataType is not supported." << std::endl;
                            break;
                        }
                    }
                    break;
                }
                case 7:
                {
                    switch (data_type)
                    {
                        TEMPLATE_RANK(DEF_TEMPLATE, Serial, 7)
                        case DataTypeKind::Unknown:
                        default:
                        {
                            std::cout << "RcpViewToNdArray::Serial, DataType is not supported." << std::endl;
                            break;
                        }
                    }
                    break;
                }
                case 8:
                {
                    switch (data_type)
                    {
                        TEMPLATE_RANK(DEF_TEMPLATE, Serial, 8)
                        case DataTypeKind::Unknown:
                        default:
                        {
                            std::cout << "RcpViewToNdArray::Serial, DataType is not supported." << std::endl;
                            break;
                        }
                    }
                    break;
                }
                default:
                {
                    std::cout << "RcpViewToNdArray::Serial, Rank is not supported." << std::endl;
                    break;
                }
            }
            break;
        }
        case ExecutionSpaceKind::OpenMP:
        {
            switch (rank)
            {
                case 0:
                {
                    switch (data_type)
                    {
                        TEMPLATE_RANK(DEF_TEMPLATE, OpenMP, 0)
                        case DataTypeKind::Unknown:
                        default:
                        {
                            std::cout << "RcpViewToNdArray::OpenMP, DataType is not supported." << std::endl;
                            break;
                        }
                    }
                    break;
                }
                case 1:
                {
                    switch (data_type)
                    {
                        TEMPLATE_RANK(DEF_TEMPLATE, OpenMP, 1)
                        case DataTypeKind::Unknown:
                        default:
                        {
                            std::cout << "RcpViewToNdArray::OpenMP, DataType is not supported." << std::endl;
                            break;
                        }
                    }
                    break;
                }
                case 2:
                {
                    switch (data_type)
                    {
                        TEMPLATE_RANK(DEF_TEMPLATE, OpenMP, 2)
                        case DataTypeKind::Unknown:
                        default:
                        {
                            std::cout << "RcpViewToNdArray::OpenMP, DataType is not supported." << std::endl;
                            break;
                        }
                    }
                    break;
                }
                case 3:
                {
                    switch (data_type)
                    {
                        TEMPLATE_RANK(DEF_TEMPLATE, OpenMP, 3)
                        case DataTypeKind::Unknown:
                        default:
                        {
                            std::cout << "RcpViewToNdArray::OpenMP, DataType is not supported." << std::endl;
                            break;
                        }
                    }
                    break;
                }
                case 4:
                {
                    switch (data_type)
                    {
                        TEMPLATE_RANK(DEF_TEMPLATE, OpenMP, 4)
                        case DataTypeKind::Unknown:
                        default:
                        {
                            std::cout << "RcpViewToNdArray::OpenMP, DataType is not supported." << std::endl;
                            break;
                        }
                    }
                    break;
                }
                case 5:
                {
                    switch (data_type)
                    {
                        TEMPLATE_RANK(DEF_TEMPLATE, OpenMP, 5)
                        case DataTypeKind::Unknown:
                        default:
                        {
                            std::cout << "RcpViewToNdArray::OpenMP, DataType is not supported." << std::endl;
                            break;
                        }
                    }
                    break;
                }
                case 6:
                {
                    switch (data_type)
                    {
                        TEMPLATE_RANK(DEF_TEMPLATE, OpenMP, 6)
                        case DataTypeKind::Unknown:
                        default:
                        {
                            std::cout << "RcpViewToNdArray::OpenMP, DataType is not supported." << std::endl;
                            break;
                        }
                    }
                    break;
                }
                case 7:
                {
                    switch (data_type)
                    {
                        TEMPLATE_RANK(DEF_TEMPLATE, OpenMP, 7)
                        case DataTypeKind::Unknown:
                        default:
                        {
                            std::cout << "RcpViewToNdArray::OpenMP, DataType is not supported." << std::endl;
                            break;
                        }
                    }
                    break;
                }
                case 8:
                {
                    switch (data_type)
                    {
                        TEMPLATE_RANK(DEF_TEMPLATE, OpenMP, 8)
                        case DataTypeKind::Unknown:
                        default:
                        {
                            std::cout << "RcpViewToNdArray::OpenMP, DataType is not supported." << std::endl;
                            break;
                        }
                    }
                    break;
                }
                default:
                {
                    std::cout << "RcpViewToNdArray::OpenMP, Rank is not supported." << std::endl;
                    break;
                }
            }
            break;
        }
        case ExecutionSpaceKind::Cuda:
        {
            switch (rank)
            {
                case 0:
                {
                    switch (data_type)
                    {
                        TEMPLATE_RANK(DEF_TEMPLATE, Cuda, 0)
                        case DataTypeKind::Unknown:
                        default:
                        {
                            std::cout << "RcpViewToNdArray::Cuda, DataType is not supported." << std::endl;
                            break;
                        }
                    }
                    break;
                }
                case 1:
                {
                    switch (data_type)
                    {
                        TEMPLATE_RANK(DEF_TEMPLATE, Cuda, 1)
                        case DataTypeKind::Unknown:
                        default:
                        {
                            std::cout << "RcpViewToNdArray::Cuda, DataType is not supported." << std::endl;
                            break;
                        }
                    }
                    break;
                }
                case 2:
                {
                    switch (data_type)
                    {
                        TEMPLATE_RANK(DEF_TEMPLATE, Cuda, 2)
                        case DataTypeKind::Unknown:
                        default:
                        {
                            std::cout << "RcpViewToNdArray::Cuda, DataType is not supported." << std::endl;
                            break;
                        }
                    }
                    break;
                }
                case 3:
                {
                    switch (data_type)
                    {
                        TEMPLATE_RANK(DEF_TEMPLATE, Cuda, 3)
                        case DataTypeKind::Unknown:
                        default:
                        {
                            std::cout << "RcpViewToNdArray::Cuda, DataType is not supported." << std::endl;
                            break;
                        }
                    }
                    break;
                }
                case 4:
                {
                    switch (data_type)
                    {
                        TEMPLATE_RANK(DEF_TEMPLATE, Cuda, 4)
                        case DataTypeKind::Unknown:
                        default:
                        {
                            std::cout << "RcpViewToNdArray::Cuda, DataType is not supported." << std::endl;
                            break;
                        }
                    }
                    break;
                }
                case 5:
                {
                    switch (data_type)
                    {
                        TEMPLATE_RANK(DEF_TEMPLATE, Cuda, 5)
                        case DataTypeKind::Unknown:
                        default:
                        {
                            std::cout << "RcpViewToNdArray::Cuda, DataType is not supported." << std::endl;
                            break;
                        }
                    }
                    break;
                }
                case 6:
                {
                    switch (data_type)
                    {
                        TEMPLATE_RANK(DEF_TEMPLATE, Cuda, 6)
                        case DataTypeKind::Unknown:
                        default:
                        {
                            std::cout << "RcpViewToNdArray::Cuda, DataType is not supported." << std::endl;
                            break;
                        }
                    }
                    break;
                }
                case 7:
                {
                    switch (data_type)
                    {
                        TEMPLATE_RANK(DEF_TEMPLATE, Cuda, 7)
                        case DataTypeKind::Unknown:
                        default:
                        {
                            std::cout << "RcpViewToNdArray::Cuda, DataType is not supported." << std::endl;
                            break;
                        }
                    }
                    break;
                }
                case 8:
                {
                    switch (data_type)
                    {
                        TEMPLATE_RANK(DEF_TEMPLATE, Cuda, 8)
                        case DataTypeKind::Unknown:
                        default:
                        {
                            std::cout << "RcpViewToNdArray::Cuda, DataType is not supported." << std::endl;
                            break;
                        }
                    }
                    break;
                }
                default:
                {
                    std::cout << "RcpViewToNdArray::Cuda, Rank is not supported." << std::endl;
                    break;
                }
            }
            break;
        }
        default:
        {
            std::cout << "RcpViewToNdArray ExecutionSpace is not supported." << std::endl;
            break;
        }

            // if (ndArray == nullptr)
            //{
            //    ndArray = new NdArray(data_type, ~uint16(0), layout, execution_space, nullptr, NativeString(0, nullptr));
            //}
    }
}

#undef TEMPLATE_RANK
#undef DEF_TEMPLATE
