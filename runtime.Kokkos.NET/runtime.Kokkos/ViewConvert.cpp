
#include "runtime.Kokkos/KokkosApi.h"

template<typename DataType, class ExecutionSpace, typename Layout>
__inline static NdArray ViewToNdArrayRank0(void* instance) noexcept
{
    typedef Kokkos::View<DataType, Layout, ExecutionSpace> view_type;

    typedef NdArrayTraits<typename view_type::traits::value_type, typename view_type::traits::execution_space, typename view_type::traits::array_layout, 0> ndarray_traits;

    view_type* view = reinterpret_cast<view_type*>(instance);

    NdArray ndArray;
    ndArray.data_type       = ndarray_traits::data_type;
    ndArray.rank            = 0;
    ndArray.layout          = ndarray_traits::layout;
    ndArray.execution_space = ndarray_traits::execution_space;
    ndArray.data            = view->data();
    ndArray.label           = view->label().c_str();

    return ndArray;
}

template<typename DataType, class ExecutionSpace, typename Layout>
__inline static NdArray ViewToNdArrayRank1(void* instance) noexcept
{
    typedef Kokkos::View<DataType*, Layout, ExecutionSpace> view_type;

    typedef NdArrayTraits<typename view_type::traits::value_type, typename view_type::traits::execution_space, typename view_type::traits::array_layout, 1> ndarray_traits;

    view_type* view = reinterpret_cast<view_type*>(instance);

    NdArray ndArray;
    ndArray.data_type       = ndarray_traits::data_type;
    ndArray.rank            = 1;
    ndArray.layout          = ndarray_traits::layout;
    ndArray.execution_space = ndarray_traits::execution_space;
    ndArray.data            = view->data();
    ndArray.label           = view->label().c_str();

    ndArray.dims[0] = view->extent(0);

    return ndArray;
}

template<typename DataType, class ExecutionSpace, typename Layout>
__inline static NdArray ViewToNdArrayRank2(void* instance) noexcept
{
    typedef Kokkos::View<DataType**, Layout, ExecutionSpace> view_type;

    typedef NdArrayTraits<typename view_type::traits::value_type, typename view_type::traits::execution_space, typename view_type::traits::array_layout, 2> ndarray_traits;

    view_type* view = reinterpret_cast<view_type*>(instance);

    NdArray ndArray;
    ndArray.data_type       = ndarray_traits::data_type;
    ndArray.rank            = 2;
    ndArray.layout          = ndarray_traits::layout;
    ndArray.execution_space = ndarray_traits::execution_space;
    ndArray.data            = view->data();
    ndArray.label           = view->label().c_str();

    ndArray.dims[0] = view->extent(0);
    ndArray.dims[1] = view->extent(1);

    return ndArray;
}

template<typename DataType, class ExecutionSpace, typename Layout>
__inline static NdArray ViewToNdArrayRank3(void* instance) noexcept
{
    typedef Kokkos::View<DataType***, Layout, ExecutionSpace> view_type;

    typedef NdArrayTraits<typename view_type::traits::value_type, typename view_type::traits::execution_space, typename view_type::traits::array_layout, 3> ndarray_traits;

    view_type* view = reinterpret_cast<view_type*>(instance);

    NdArray ndArray;
    ndArray.data_type       = ndarray_traits::data_type;
    ndArray.rank            = 3;
    ndArray.layout          = ndarray_traits::layout;
    ndArray.execution_space = ndarray_traits::execution_space;
    ndArray.data            = view->data();
    ndArray.label           = view->label().c_str();

    ndArray.dims[0] = view->extent(0);
    ndArray.dims[1] = view->extent(1);
    ndArray.dims[2] = view->extent(2);

    return ndArray;
}

template<typename DataType, class ExecutionSpace, typename Layout>
__inline static NdArray ViewToNdArrayRank4(void* instance) noexcept
{
    typedef Kokkos::View<DataType****, Layout, ExecutionSpace> view_type;

    typedef NdArrayTraits<typename view_type::traits::value_type, typename view_type::traits::execution_space, typename view_type::traits::array_layout, 4> ndarray_traits;

    view_type* view = reinterpret_cast<view_type*>(instance);

    NdArray ndArray;
    ndArray.data_type       = ndarray_traits::data_type;
    ndArray.rank            = 4;
    ndArray.layout          = ndarray_traits::layout;
    ndArray.execution_space = ndarray_traits::execution_space;
    ndArray.data            = view->data();
    ndArray.label           = view->label().c_str();

    ndArray.dims[0] = view->extent(0);
    ndArray.dims[1] = view->extent(1);
    ndArray.dims[2] = view->extent(2);
    ndArray.dims[3] = view->extent(3);

    return ndArray;
}

template<typename DataType, class ExecutionSpace, typename Layout>
__inline static NdArray ViewToNdArrayRank5(void* instance) noexcept
{
    typedef Kokkos::View<DataType*****, Layout, ExecutionSpace> view_type;

    typedef NdArrayTraits<typename view_type::traits::value_type, typename view_type::traits::execution_space, typename view_type::traits::array_layout, 5> ndarray_traits;

    view_type* view = reinterpret_cast<view_type*>(instance);

    NdArray ndArray;
    ndArray.data_type       = ndarray_traits::data_type;
    ndArray.rank            = 5;
    ndArray.layout          = ndarray_traits::layout;
    ndArray.execution_space = ndarray_traits::execution_space;
    ndArray.data            = view->data();
    ndArray.label           = view->label().c_str();

    ndArray.dims[0] = view->extent(0);
    ndArray.dims[1] = view->extent(1);
    ndArray.dims[2] = view->extent(2);
    ndArray.dims[3] = view->extent(3);
    ndArray.dims[4] = view->extent(4);

    return ndArray;
}

template<typename DataType, class ExecutionSpace, typename Layout>
__inline static NdArray ViewToNdArrayRank6(void* instance) noexcept
{
    typedef Kokkos::View<DataType******, Layout, ExecutionSpace> view_type;

    typedef NdArrayTraits<typename view_type::traits::value_type, typename view_type::traits::execution_space, typename view_type::traits::array_layout, 6> ndarray_traits;

    view_type* view = reinterpret_cast<view_type*>(instance);

    NdArray ndArray;
    ndArray.data_type       = ndarray_traits::data_type;
    ndArray.rank            = 6;
    ndArray.layout          = ndarray_traits::layout;
    ndArray.execution_space = ndarray_traits::execution_space;
    ndArray.data            = view->data();
    ndArray.label           = view->label().c_str();

    ndArray.dims[0] = view->extent(0);
    ndArray.dims[1] = view->extent(1);
    ndArray.dims[2] = view->extent(2);
    ndArray.dims[3] = view->extent(3);
    ndArray.dims[4] = view->extent(4);
    ndArray.dims[5] = view->extent(5);

    return ndArray;
}

template<typename DataType, class ExecutionSpace, typename Layout>
__inline static NdArray ViewToNdArrayRank7(void* instance) noexcept
{
    typedef Kokkos::View<DataType*******, Layout, ExecutionSpace> view_type;

    typedef NdArrayTraits<typename view_type::traits::value_type, typename view_type::traits::execution_space, typename view_type::traits::array_layout, 7> ndarray_traits;

    view_type* view = reinterpret_cast<view_type*>(instance);

    NdArray ndArray;
    ndArray.data_type       = ndarray_traits::data_type;
    ndArray.rank            = 7;
    ndArray.layout          = ndarray_traits::layout;
    ndArray.execution_space = ndarray_traits::execution_space;
    ndArray.data            = view->data();
    ndArray.label           = view->label().c_str();

    ndArray.dims[0] = view->extent(0);
    ndArray.dims[1] = view->extent(1);
    ndArray.dims[2] = view->extent(2);
    ndArray.dims[3] = view->extent(3);
    ndArray.dims[4] = view->extent(4);
    ndArray.dims[5] = view->extent(5);
    ndArray.dims[6] = view->extent(6);

    return ndArray;
}

template<typename DataType, class ExecutionSpace, typename Layout>
__inline static NdArray ViewToNdArrayRank8(void* instance) noexcept
{
    typedef Kokkos::View<DataType********, Layout, ExecutionSpace> view_type;

    typedef NdArrayTraits<typename view_type::traits::value_type, typename view_type::traits::execution_space, typename view_type::traits::array_layout, 8> ndarray_traits;

    view_type* view = reinterpret_cast<view_type*>(instance);

    NdArray ndArray;
    ndArray.data_type       = ndarray_traits::data_type;
    ndArray.rank            = 8;
    ndArray.layout          = ndarray_traits::layout;
    ndArray.execution_space = ndarray_traits::execution_space;
    ndArray.data            = view->data();
    ndArray.label           = view->label().c_str();

    ndArray.dims[0] = view->extent(0);
    ndArray.dims[1] = view->extent(1);
    ndArray.dims[2] = view->extent(2);
    ndArray.dims[3] = view->extent(3);
    ndArray.dims[4] = view->extent(4);
    ndArray.dims[5] = view->extent(5);
    ndArray.dims[6] = view->extent(6);
    ndArray.dims[7] = view->extent(7);

    return ndArray;
}

#define DEF_TEMPLATE(TYPE_NAME, TYPE, EXECUTION_SPACE, RANK)                                                  \
    case DataTypeKind::TYPE_NAME:                                                                             \
    {                                                                                                         \
        switch(layout)                                                                                        \
        {                                                                                                     \
            case LayoutKind::Right:                                                                           \
            {                                                                                                 \
                return ViewToNdArrayRank##RANK<TYPE, Kokkos::EXECUTION_SPACE, Kokkos::LayoutRight>(instance); \
            }                                                                                                 \
            case LayoutKind::Left:                                                                            \
            {                                                                                                 \
                return ViewToNdArrayRank##RANK<TYPE, Kokkos::EXECUTION_SPACE, Kokkos::LayoutLeft>(instance);  \
            }                                                                                                 \
        }                                                                                                     \
    }

#define TEMPLATE_RANK(DEF, EXECUTION_SPACE, RANK) \
    DEF(Single, float, EXECUTION_SPACE, RANK)     \
    DEF(Double, double, EXECUTION_SPACE, RANK)    \
    DEF(Bool, bool, EXECUTION_SPACE, RANK)        \
    DEF(Int8, int8, EXECUTION_SPACE, RANK)        \
    DEF(UInt8, uint8, EXECUTION_SPACE, RANK)      \
    DEF(Int16, int16, EXECUTION_SPACE, RANK)      \
    DEF(UInt16, uint16, EXECUTION_SPACE, RANK)    \
    DEF(Int32, int32, EXECUTION_SPACE, RANK)      \
    DEF(UInt32, uint32, EXECUTION_SPACE, RANK)    \
    DEF(Int64, int64, EXECUTION_SPACE, RANK)      \
    DEF(UInt64, uint64, EXECUTION_SPACE, RANK)

NdArray ViewToNdArray(void* instance, const ExecutionSpaceKind& execution_space, const LayoutKind& layout, const DataTypeKind& data_type, const uint16& rank) noexcept
{
    switch(execution_space)
    {
        case ExecutionSpaceKind::Serial:
        {
            switch(rank)
            {
                case 0:
                {
                    switch(data_type)
                    {
                        TEMPLATE_RANK(DEF_TEMPLATE, Serial, 0)
                        default:
                        {
                            std::cout << "ViewToNdArray::Serial, DataType is not supported." << std::endl;
                        }
                    }
                    break;
                }
                case 1:
                {
                    switch(data_type)
                    {
                        TEMPLATE_RANK(DEF_TEMPLATE, Serial, 1)
                        default:
                        {
                            std::cout << "ViewToNdArray::Serial, DataType is not supported." << std::endl;
                        }
                    }
                    break;
                }
                case 2:
                {
                    switch(data_type)
                    {
                        TEMPLATE_RANK(DEF_TEMPLATE, Serial, 2)
                        default:
                        {
                            std::cout << "ViewToNdArray::Serial, DataType is not supported." << std::endl;
                        }
                    }
                    break;
                }
                case 3:
                {
                    switch(data_type)
                    {
                        TEMPLATE_RANK(DEF_TEMPLATE, Serial, 3)
                        default:
                        {
                            std::cout << "ViewToNdArray::Serial, DataType is not supported." << std::endl;
                        }
                    }
                    break;
                }
                case 4:
                {
                    switch(data_type)
                    {
                        TEMPLATE_RANK(DEF_TEMPLATE, Serial, 4)
                        default:
                        {
                            std::cout << "ViewToNdArray::Serial, DataType is not supported." << std::endl;
                        }
                    }
                    break;
                }
                case 5:
                {
                    switch(data_type)
                    {
                        TEMPLATE_RANK(DEF_TEMPLATE, Serial, 5)
                        default:
                        {
                            std::cout << "ViewToNdArray::Serial, DataType is not supported." << std::endl;
                        }
                    }
                    break;
                }
                case 6:
                {
                    switch(data_type)
                    {
                        TEMPLATE_RANK(DEF_TEMPLATE, Serial, 6)
                        default:
                        {
                            std::cout << "ViewToNdArray::Serial, DataType is not supported." << std::endl;
                        }
                    }
                    break;
                }
                case 7:
                {
                    switch(data_type)
                    {
                        TEMPLATE_RANK(DEF_TEMPLATE, Serial, 7)
                        default:
                        {
                            std::cout << "ViewToNdArray::Serial, DataType is not supported." << std::endl;
                        }
                    }
                    break;
                }
                case 8:
                {
                    switch(data_type)
                    {
                        TEMPLATE_RANK(DEF_TEMPLATE, Serial, 8)
                        default:
                        {
                            std::cout << "ViewToNdArray::Serial, DataType is not supported." << std::endl;
                        }
                    }
                    break;
                }
                default:
                {
                    std::cout << "ViewToNdArray::Serial, Rank is not supported." << std::endl;
                }
            }
        }
        case ExecutionSpaceKind::OpenMP:
        {
            switch(rank)
            {
                case 0:
                {
                    switch(data_type)
                    {
                        TEMPLATE_RANK(DEF_TEMPLATE, OpenMP, 0)
                        default:
                        {
                            std::cout << "ViewToNdArray::OpenMP, DataType is not supported." << std::endl;
                        }
                    }
                    break;
                }
                case 1:
                {
                    switch(data_type)
                    {
                        TEMPLATE_RANK(DEF_TEMPLATE, OpenMP, 1)
                        default:
                        {
                            std::cout << "ViewToNdArray::OpenMP, DataType is not supported." << std::endl;
                        }
                    }
                    break;
                }
                case 2:
                {
                    switch(data_type)
                    {
                        TEMPLATE_RANK(DEF_TEMPLATE, OpenMP, 2)
                        default:
                        {
                            std::cout << "ViewToNdArray::OpenMP, DataType is not supported." << std::endl;
                        }
                    }
                    break;
                }
                case 3:
                {
                    switch(data_type)
                    {
                        TEMPLATE_RANK(DEF_TEMPLATE, OpenMP, 3)
                        default:
                        {
                            std::cout << "ViewToNdArray::OpenMP, DataType is not supported." << std::endl;
                        }
                    }
                    break;
                }
                case 4:
                {
                    switch(data_type)
                    {
                        TEMPLATE_RANK(DEF_TEMPLATE, OpenMP, 4)
                        default:
                        {
                            std::cout << "ViewToNdArray::OpenMP, DataType is not supported." << std::endl;
                        }
                    }
                    break;
                }
                case 5:
                {
                    switch(data_type)
                    {
                        TEMPLATE_RANK(DEF_TEMPLATE, OpenMP, 5)
                        default:
                        {
                            std::cout << "ViewToNdArray::OpenMP, DataType is not supported." << std::endl;
                        }
                    }
                    break;
                }
                case 6:
                {
                    switch(data_type)
                    {
                        TEMPLATE_RANK(DEF_TEMPLATE, OpenMP, 6)
                        default:
                        {
                            std::cout << "ViewToNdArray::OpenMP, DataType is not supported." << std::endl;
                        }
                    }
                    break;
                }
                case 7:
                {
                    switch(data_type)
                    {
                        TEMPLATE_RANK(DEF_TEMPLATE, OpenMP, 7)
                        default:
                        {
                            std::cout << "ViewToNdArray::OpenMP, DataType is not supported." << std::endl;
                        }
                    }
                    break;
                }
                case 8:
                {
                    switch(data_type)
                    {
                        TEMPLATE_RANK(DEF_TEMPLATE, OpenMP, 8)
                        default:
                        {
                            std::cout << "ViewToNdArray::OpenMP, DataType is not supported." << std::endl;
                        }
                    }
                    break;
                }
                default:
                {
                    std::cout << "ViewToNdArray::OpenMP, Rank is not supported." << std::endl;
                }
            }
        }
        case ExecutionSpaceKind::Cuda:
        {
            switch(rank)
            {
                case 0:
                {
                    switch(data_type)
                    {
                        TEMPLATE_RANK(DEF_TEMPLATE, Cuda, 0)
                        default:
                        {
                            std::cout << "ViewToNdArray::Cuda, DataType is not supported." << std::endl;
                        }
                    }
                    break;
                }
                case 1:
                {
                    switch(data_type)
                    {
                        TEMPLATE_RANK(DEF_TEMPLATE, Cuda, 1)
                        default:
                        {
                            std::cout << "ViewToNdArray::Cuda, DataType is not supported." << std::endl;
                        }
                    }
                    break;
                }
                case 2:
                {
                    switch(data_type)
                    {
                        TEMPLATE_RANK(DEF_TEMPLATE, Cuda, 2)
                        default:
                        {
                            std::cout << "ViewToNdArray::Cuda, DataType is not supported." << std::endl;
                        }
                    }
                    break;
                }
                case 3:
                {
                    switch(data_type)
                    {
                        TEMPLATE_RANK(DEF_TEMPLATE, Cuda, 3)
                        default:
                        {
                            std::cout << "ViewToNdArray::Cuda, DataType is not supported." << std::endl;
                        }
                    }
                    break;
                }
                case 4:
                {
                    switch(data_type)
                    {
                        TEMPLATE_RANK(DEF_TEMPLATE, Cuda, 4)
                        default:
                        {
                            std::cout << "ViewToNdArray::Cuda, DataType is not supported." << std::endl;
                        }
                    }
                    break;
                }
                case 5:
                {
                    switch(data_type)
                    {
                        TEMPLATE_RANK(DEF_TEMPLATE, Cuda, 5)
                        default:
                        {
                            std::cout << "ViewToNdArray::Cuda, DataType is not supported." << std::endl;
                        }
                    }
                    break;
                }
                case 6:
                {
                    switch(data_type)
                    {
                        TEMPLATE_RANK(DEF_TEMPLATE, Cuda, 6)
                        default:
                        {
                            std::cout << "ViewToNdArray::Cuda, DataType is not supported." << std::endl;
                        }
                    }
                    break;
                }
                case 7:
                {
                    switch(data_type)
                    {
                        TEMPLATE_RANK(DEF_TEMPLATE, Cuda, 7)
                        default:
                        {
                            std::cout << "ViewToNdArray::Cuda, DataType is not supported." << std::endl;
                        }
                    }
                    break;
                }
                case 8:
                {
                    switch(data_type)
                    {
                        TEMPLATE_RANK(DEF_TEMPLATE, Cuda, 8)
                        default:
                        {
                            std::cout << "ViewToNdArray::Cuda, DataType is not supported." << std::endl;
                        }
                    }
                    break;
                }
                default:
                {
                    std::cout << "ViewToNdArray::Cuda, Rank is not supported." << std::endl;
                }
            }
        }
        default:
        {
            std::cout << "ViewToNdArray ExecutionSpace is not supported." << std::endl;
        }

            return NdArray {};
    }
}

#undef DEF_TEMPLATE

#define DEF_TEMPLATE(TYPE_NAME, TYPE, EXECUTION_SPACE, RANK)                                                  \
    case DataTypeKind::TYPE_NAME:                                                                             \
    {                                                                                                         \
        switch(layout)                                                                                        \
        {                                                                                                     \
            case LayoutKind::Right:                                                                           \
            {                                                                                                 \
                return RcpViewToNdArrayRank##RANK<TYPE, Kokkos::EXECUTION_SPACE, Kokkos::LayoutRight>(instance); \
            }                                                                                                 \
            case LayoutKind::Left:                                                                            \
            {                                                                                                 \
                return RcpViewToNdArrayRank##RANK<TYPE, Kokkos::EXECUTION_SPACE, Kokkos::LayoutLeft>(instance);  \
            }                                                                                                 \
        }                                                                                                     \
    }

template<typename DataType, class ExecutionSpace, typename Layout>
__inline static NdArray RcpViewToNdArrayRank0(void* instance) noexcept
{
    typedef Kokkos::View<DataType, Layout, ExecutionSpace> view_type;

    typedef NdArrayTraits<typename view_type::traits::value_type, typename view_type::traits::execution_space, typename view_type::traits::array_layout, 0> ndarray_traits;

    Teuchos::RCP<view_type>* view = reinterpret_cast<Teuchos::RCP<view_type>*>(instance);

    NdArray ndArray;
    ndArray.data_type       = ndarray_traits::data_type;
    ndArray.rank            = 0;
    ndArray.layout          = ndarray_traits::layout;
    ndArray.execution_space = ndarray_traits::execution_space;
    ndArray.data            = (*view)->data();
    ndArray.label           = (*view)->label().c_str();

    return ndArray;
}

template<typename DataType, class ExecutionSpace, typename Layout>
__inline static NdArray RcpViewToNdArrayRank1(void* instance) noexcept
{
    typedef Kokkos::View<DataType*, Layout, ExecutionSpace> view_type;

    typedef NdArrayTraits<typename view_type::traits::value_type, typename view_type::traits::execution_space, typename view_type::traits::array_layout, 1> ndarray_traits;

    Teuchos::RCP<view_type>* view = reinterpret_cast<Teuchos::RCP<view_type>*>(instance);

    NdArray ndArray;
    ndArray.data_type       = ndarray_traits::data_type;
    ndArray.rank            = 1;
    ndArray.layout          = ndarray_traits::layout;
    ndArray.execution_space = ndarray_traits::execution_space;
    ndArray.data            = (*view)->data();
    ndArray.label           = (*view)->label().c_str();

    ndArray.dims[0] = (*view)->extent(0);

    return ndArray;
}

template<typename DataType, class ExecutionSpace, typename Layout>
__inline static NdArray RcpViewToNdArrayRank2(void* instance) noexcept
{
    typedef Kokkos::View<DataType**, Layout, ExecutionSpace> view_type;

    typedef NdArrayTraits<typename view_type::traits::value_type, typename view_type::traits::execution_space, typename view_type::traits::array_layout, 2> ndarray_traits;

    Teuchos::RCP<view_type>* view = reinterpret_cast<Teuchos::RCP<view_type>*>(instance);

    NdArray ndArray;
    ndArray.data_type       = ndarray_traits::data_type;
    ndArray.rank            = 2;
    ndArray.layout          = ndarray_traits::layout;
    ndArray.execution_space = ndarray_traits::execution_space;
    ndArray.data            = (*view)->data();
    ndArray.label           = (*view)->label().c_str();

    ndArray.dims[0] = (*view)->extent(0);
    ndArray.dims[1] = (*view)->extent(1);

    return ndArray;
}

template<typename DataType, class ExecutionSpace, typename Layout>
__inline static NdArray RcpViewToNdArrayRank3(void* instance) noexcept
{
    typedef Kokkos::View<DataType***, Layout, ExecutionSpace> view_type;

    typedef NdArrayTraits<typename view_type::traits::value_type, typename view_type::traits::execution_space, typename view_type::traits::array_layout, 3> ndarray_traits;

    Teuchos::RCP<view_type>* view = reinterpret_cast<Teuchos::RCP<view_type>*>(instance);

    NdArray ndArray;
    ndArray.data_type       = ndarray_traits::data_type;
    ndArray.rank            = 3;
    ndArray.layout          = ndarray_traits::layout;
    ndArray.execution_space = ndarray_traits::execution_space;
    ndArray.data            = (*view)->data();
    ndArray.label           = (*view)->label().c_str();

    ndArray.dims[0] = (*view)->extent(0);
    ndArray.dims[1] = (*view)->extent(1);
    ndArray.dims[2] = (*view)->extent(2);

    return ndArray;
}

template<typename DataType, class ExecutionSpace, typename Layout>
__inline static NdArray RcpViewToNdArrayRank4(void* instance) noexcept
{
    typedef Kokkos::View<DataType****, Layout, ExecutionSpace> view_type;

    typedef NdArrayTraits<typename view_type::traits::value_type, typename view_type::traits::execution_space, typename view_type::traits::array_layout, 4> ndarray_traits;

    Teuchos::RCP<view_type>* view = reinterpret_cast<Teuchos::RCP<view_type>*>(instance);

    NdArray ndArray;
    ndArray.data_type       = ndarray_traits::data_type;
    ndArray.rank            = 4;
    ndArray.layout          = ndarray_traits::layout;
    ndArray.execution_space = ndarray_traits::execution_space;
    ndArray.data            = (*view)->data();
    ndArray.label           = (*view)->label().c_str();

    ndArray.dims[0] = (*view)->extent(0);
    ndArray.dims[1] = (*view)->extent(1);
    ndArray.dims[2] = (*view)->extent(2);
    ndArray.dims[3] = (*view)->extent(3);

    return ndArray;
}

template<typename DataType, class ExecutionSpace, typename Layout>
__inline static NdArray RcpViewToNdArrayRank5(void* instance) noexcept
{
    typedef Kokkos::View<DataType*****, Layout, ExecutionSpace> view_type;

    typedef NdArrayTraits<typename view_type::traits::value_type, typename view_type::traits::execution_space, typename view_type::traits::array_layout, 5> ndarray_traits;

    Teuchos::RCP<view_type>* view = reinterpret_cast<Teuchos::RCP<view_type>*>(instance);

    NdArray ndArray;
    ndArray.data_type       = ndarray_traits::data_type;
    ndArray.rank            = 5;
    ndArray.layout          = ndarray_traits::layout;
    ndArray.execution_space = ndarray_traits::execution_space;
    ndArray.data            = (*view)->data();
    ndArray.label           = (*view)->label().c_str();

    ndArray.dims[0] = (*view)->extent(0);
    ndArray.dims[1] = (*view)->extent(1);
    ndArray.dims[2] = (*view)->extent(2);
    ndArray.dims[3] = (*view)->extent(3);
    ndArray.dims[4] = (*view)->extent(4);

    return ndArray;
}

template<typename DataType, class ExecutionSpace, typename Layout>
__inline static NdArray RcpViewToNdArrayRank6(void* instance) noexcept
{
    typedef Kokkos::View<DataType******, Layout, ExecutionSpace> view_type;

    typedef NdArrayTraits<typename view_type::traits::value_type, typename view_type::traits::execution_space, typename view_type::traits::array_layout, 6> ndarray_traits;

    Teuchos::RCP<view_type>* view = reinterpret_cast<Teuchos::RCP<view_type>*>(instance);

    NdArray ndArray;
    ndArray.data_type       = ndarray_traits::data_type;
    ndArray.rank            = 6;
    ndArray.layout          = ndarray_traits::layout;
    ndArray.execution_space = ndarray_traits::execution_space;
    ndArray.data            = (*view)->data();
    ndArray.label           = (*view)->label().c_str();

    ndArray.dims[0] = (*view)->extent(0);
    ndArray.dims[1] = (*view)->extent(1);
    ndArray.dims[2] = (*view)->extent(2);
    ndArray.dims[3] = (*view)->extent(3);
    ndArray.dims[4] = (*view)->extent(4);
    ndArray.dims[5] = (*view)->extent(5);

    return ndArray;
}

template<typename DataType, class ExecutionSpace, typename Layout>
__inline static NdArray RcpViewToNdArrayRank7(void* instance) noexcept
{
    typedef Kokkos::View<DataType*******, Layout, ExecutionSpace> view_type;

    typedef NdArrayTraits<typename view_type::traits::value_type, typename view_type::traits::execution_space, typename view_type::traits::array_layout, 7> ndarray_traits;

    Teuchos::RCP<view_type>* view = reinterpret_cast<Teuchos::RCP<view_type>*>(instance);

    NdArray ndArray;
    ndArray.data_type       = ndarray_traits::data_type;
    ndArray.rank            = 7;
    ndArray.layout          = ndarray_traits::layout;
    ndArray.execution_space = ndarray_traits::execution_space;
    ndArray.data            = (*view)->data();
    ndArray.label           = (*view)->label().c_str();

    ndArray.dims[0] = (*view)->extent(0);
    ndArray.dims[1] = (*view)->extent(1);
    ndArray.dims[2] = (*view)->extent(2);
    ndArray.dims[3] = (*view)->extent(3);
    ndArray.dims[4] = (*view)->extent(4);
    ndArray.dims[5] = (*view)->extent(5);
    ndArray.dims[6] = (*view)->extent(6);

    return ndArray;
}

template<typename DataType, class ExecutionSpace, typename Layout>
__inline static NdArray RcpViewToNdArrayRank8(void* instance) noexcept
{
    typedef Kokkos::View<DataType********, Layout, ExecutionSpace> view_type;

    typedef NdArrayTraits<typename view_type::traits::value_type, typename view_type::traits::execution_space, typename view_type::traits::array_layout, 8> ndarray_traits;

    Teuchos::RCP<view_type>* view = reinterpret_cast<Teuchos::RCP<view_type>*>(instance);

    NdArray ndArray;
    ndArray.data_type       = ndarray_traits::data_type;
    ndArray.rank            = 8;
    ndArray.layout          = ndarray_traits::layout;
    ndArray.execution_space = ndarray_traits::execution_space;
    ndArray.data            = (*view)->data();
    ndArray.label           = (*view)->label().c_str();

    ndArray.dims[0] = (*view)->extent(0);
    ndArray.dims[1] = (*view)->extent(1);
    ndArray.dims[2] = (*view)->extent(2);
    ndArray.dims[3] = (*view)->extent(3);
    ndArray.dims[4] = (*view)->extent(4);
    ndArray.dims[5] = (*view)->extent(5);
    ndArray.dims[6] = (*view)->extent(6);
    ndArray.dims[7] = (*view)->extent(7);

    return ndArray;
}



NdArray RcpViewToNdArray(void* instance, const ExecutionSpaceKind& execution_space, const LayoutKind& layout, const DataTypeKind& data_type, const uint16& rank) noexcept
{
    switch(execution_space)
    {
        case ExecutionSpaceKind::Serial:
        {
            switch(rank)
            {
                case 0:
                {
                    switch(data_type)
                    {
                        TEMPLATE_RANK(DEF_TEMPLATE, Serial, 0)
                        default:
                        {
                            std::cout << "ViewToNdArray::Serial, DataType is not supported." << std::endl;
                        }
                    }
                    break;
                }
                case 1:
                {
                    switch(data_type)
                    {
                        TEMPLATE_RANK(DEF_TEMPLATE, Serial, 1)
                        default:
                        {
                            std::cout << "ViewToNdArray::Serial, DataType is not supported." << std::endl;
                        }
                    }
                    break;
                }
                case 2:
                {
                    switch(data_type)
                    {
                        TEMPLATE_RANK(DEF_TEMPLATE, Serial, 2)
                        default:
                        {
                            std::cout << "ViewToNdArray::Serial, DataType is not supported." << std::endl;
                        }
                    }
                    break;
                }
                case 3:
                {
                    switch(data_type)
                    {
                        TEMPLATE_RANK(DEF_TEMPLATE, Serial, 3)
                        default:
                        {
                            std::cout << "ViewToNdArray::Serial, DataType is not supported." << std::endl;
                        }
                    }
                    break;
                }
                case 4:
                {
                    switch(data_type)
                    {
                        TEMPLATE_RANK(DEF_TEMPLATE, Serial, 4)
                        default:
                        {
                            std::cout << "ViewToNdArray::Serial, DataType is not supported." << std::endl;
                        }
                    }
                    break;
                }
                case 5:
                {
                    switch(data_type)
                    {
                        TEMPLATE_RANK(DEF_TEMPLATE, Serial, 5)
                        default:
                        {
                            std::cout << "ViewToNdArray::Serial, DataType is not supported." << std::endl;
                        }
                    }
                    break;
                }
                case 6:
                {
                    switch(data_type)
                    {
                        TEMPLATE_RANK(DEF_TEMPLATE, Serial, 6)
                        default:
                        {
                            std::cout << "ViewToNdArray::Serial, DataType is not supported." << std::endl;
                        }
                    }
                    break;
                }
                case 7:
                {
                    switch(data_type)
                    {
                        TEMPLATE_RANK(DEF_TEMPLATE, Serial, 7)
                        default:
                        {
                            std::cout << "ViewToNdArray::Serial, DataType is not supported." << std::endl;
                        }
                    }
                    break;
                }
                case 8:
                {
                    switch(data_type)
                    {
                        TEMPLATE_RANK(DEF_TEMPLATE, Serial, 8)
                        default:
                        {
                            std::cout << "ViewToNdArray::Serial, DataType is not supported." << std::endl;
                        }
                    }
                    break;
                }
                default:
                {
                    std::cout << "ViewToNdArray::Serial, Rank is not supported." << std::endl;
                }
            }
        }
        case ExecutionSpaceKind::OpenMP:
        {
            switch(rank)
            {
                case 0:
                {
                    switch(data_type)
                    {
                        TEMPLATE_RANK(DEF_TEMPLATE, OpenMP, 0)
                        default:
                        {
                            std::cout << "ViewToNdArray::OpenMP, DataType is not supported." << std::endl;
                        }
                    }
                    break;
                }
                case 1:
                {
                    switch(data_type)
                    {
                        TEMPLATE_RANK(DEF_TEMPLATE, OpenMP, 1)
                        default:
                        {
                            std::cout << "ViewToNdArray::OpenMP, DataType is not supported." << std::endl;
                        }
                    }
                    break;
                }
                case 2:
                {
                    switch(data_type)
                    {
                        TEMPLATE_RANK(DEF_TEMPLATE, OpenMP, 2)
                        default:
                        {
                            std::cout << "ViewToNdArray::OpenMP, DataType is not supported." << std::endl;
                        }
                    }
                    break;
                }
                case 3:
                {
                    switch(data_type)
                    {
                        TEMPLATE_RANK(DEF_TEMPLATE, OpenMP, 3)
                        default:
                        {
                            std::cout << "ViewToNdArray::OpenMP, DataType is not supported." << std::endl;
                        }
                    }
                    break;
                }
                case 4:
                {
                    switch(data_type)
                    {
                        TEMPLATE_RANK(DEF_TEMPLATE, OpenMP, 4)
                        default:
                        {
                            std::cout << "ViewToNdArray::OpenMP, DataType is not supported." << std::endl;
                        }
                    }
                    break;
                }
                case 5:
                {
                    switch(data_type)
                    {
                        TEMPLATE_RANK(DEF_TEMPLATE, OpenMP, 5)
                        default:
                        {
                            std::cout << "ViewToNdArray::OpenMP, DataType is not supported." << std::endl;
                        }
                    }
                    break;
                }
                case 6:
                {
                    switch(data_type)
                    {
                        TEMPLATE_RANK(DEF_TEMPLATE, OpenMP, 6)
                        default:
                        {
                            std::cout << "ViewToNdArray::OpenMP, DataType is not supported." << std::endl;
                        }
                    }
                    break;
                }
                case 7:
                {
                    switch(data_type)
                    {
                        TEMPLATE_RANK(DEF_TEMPLATE, OpenMP, 7)
                        default:
                        {
                            std::cout << "ViewToNdArray::OpenMP, DataType is not supported." << std::endl;
                        }
                    }
                    break;
                }
                case 8:
                {
                    switch(data_type)
                    {
                        TEMPLATE_RANK(DEF_TEMPLATE, OpenMP, 8)
                        default:
                        {
                            std::cout << "ViewToNdArray::OpenMP, DataType is not supported." << std::endl;
                        }
                    }
                    break;
                }
                default:
                {
                    std::cout << "ViewToNdArray::OpenMP, Rank is not supported." << std::endl;
                }
            }
        }
        case ExecutionSpaceKind::Cuda:
        {
            switch(rank)
            {
                case 0:
                {
                    switch(data_type)
                    {
                        TEMPLATE_RANK(DEF_TEMPLATE, Cuda, 0)
                        default:
                        {
                            std::cout << "ViewToNdArray::Cuda, DataType is not supported." << std::endl;
                        }
                    }
                    break;
                }
                case 1:
                {
                    switch(data_type)
                    {
                        TEMPLATE_RANK(DEF_TEMPLATE, Cuda, 1)
                        default:
                        {
                            std::cout << "ViewToNdArray::Cuda, DataType is not supported." << std::endl;
                        }
                    }
                    break;
                }
                case 2:
                {
                    switch(data_type)
                    {
                        TEMPLATE_RANK(DEF_TEMPLATE, Cuda, 2)
                        default:
                        {
                            std::cout << "ViewToNdArray::Cuda, DataType is not supported." << std::endl;
                        }
                    }
                    break;
                }
                case 3:
                {
                    switch(data_type)
                    {
                        TEMPLATE_RANK(DEF_TEMPLATE, Cuda, 3)
                        default:
                        {
                            std::cout << "ViewToNdArray::Cuda, DataType is not supported." << std::endl;
                        }
                    }
                    break;
                }
                case 4:
                {
                    switch(data_type)
                    {
                        TEMPLATE_RANK(DEF_TEMPLATE, Cuda, 4)
                        default:
                        {
                            std::cout << "ViewToNdArray::Cuda, DataType is not supported." << std::endl;
                        }
                    }
                    break;
                }
                case 5:
                {
                    switch(data_type)
                    {
                        TEMPLATE_RANK(DEF_TEMPLATE, Cuda, 5)
                        default:
                        {
                            std::cout << "ViewToNdArray::Cuda, DataType is not supported." << std::endl;
                        }
                    }
                    break;
                }
                case 6:
                {
                    switch(data_type)
                    {
                        TEMPLATE_RANK(DEF_TEMPLATE, Cuda, 6)
                        default:
                        {
                            std::cout << "ViewToNdArray::Cuda, DataType is not supported." << std::endl;
                        }
                    }
                    break;
                }
                case 7:
                {
                    switch(data_type)
                    {
                        TEMPLATE_RANK(DEF_TEMPLATE, Cuda, 7)
                        default:
                        {
                            std::cout << "ViewToNdArray::Cuda, DataType is not supported." << std::endl;
                        }
                    }
                    break;
                }
                case 8:
                {
                    switch(data_type)
                    {
                        TEMPLATE_RANK(DEF_TEMPLATE, Cuda, 8)
                        default:
                        {
                            std::cout << "ViewToNdArray::Cuda, DataType is not supported." << std::endl;
                        }
                    }
                    break;
                }
                default:
                {
                    std::cout << "ViewToNdArray::Cuda, Rank is not supported." << std::endl;
                }
            }
        }
        default:
        {
            std::cout << "ViewToNdArray ExecutionSpace is not supported." << std::endl;
        }

            return NdArray {};
    }
}

#undef TEMPLATE_RANK
#undef DEF_TEMPLATE
