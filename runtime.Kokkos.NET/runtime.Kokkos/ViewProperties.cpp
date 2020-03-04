
#include "runtime.Kokkos/KokkosApi.h"

#define DEF_TEMPLATE(TYPE_NAME, TYPE, EXECUTION_SPACE)                                                                                                                                \
    case DataTypeKind::TYPE_NAME:                                                                                                                                                     \
    {                                                                                                                                                                                 \
        typedef Kokkos::View<TYPE, typename Kokkos::EXECUTION_SPACE::array_layout, Kokkos::EXECUTION_SPACE> view_type;                                                                \
        const std::string&                                                                                  label = (*reinterpret_cast<Teuchos::RCP<view_type>*>(instance))->label(); \
        return NativeString(label);                                                                                                                                                   \
    }

#define TEMPLATE_RANK0(DEF, EXECUTION_SPACE) \
    DEF(Single, float, EXECUTION_SPACE)      \
    DEF(Double, double, EXECUTION_SPACE)     \
    DEF(Bool, bool, EXECUTION_SPACE)         \
    DEF(Int8, int8, EXECUTION_SPACE)         \
    DEF(UInt8, uint8, EXECUTION_SPACE)       \
    DEF(Int16, int16, EXECUTION_SPACE)       \
    DEF(UInt16, uint16, EXECUTION_SPACE)     \
    DEF(Int32, int32, EXECUTION_SPACE)       \
    DEF(UInt32, uint32, EXECUTION_SPACE)     \
    DEF(Int64, int64, EXECUTION_SPACE)       \
    DEF(UInt64, uint64, EXECUTION_SPACE)

#define TEMPLATE_RANK1(DEF, EXECUTION_SPACE) \
    DEF(Single, float*, EXECUTION_SPACE)     \
    DEF(Double, double*, EXECUTION_SPACE)    \
    DEF(Bool, bool*, EXECUTION_SPACE)        \
    DEF(Int8, int8*, EXECUTION_SPACE)        \
    DEF(UInt8, uint8*, EXECUTION_SPACE)      \
    DEF(Int16, int16*, EXECUTION_SPACE)      \
    DEF(UInt16, uint16*, EXECUTION_SPACE)    \
    DEF(Int32, int32*, EXECUTION_SPACE)      \
    DEF(UInt32, uint32*, EXECUTION_SPACE)    \
    DEF(Int64, int64*, EXECUTION_SPACE)      \
    DEF(UInt64, uint64*, EXECUTION_SPACE)

#define TEMPLATE_RANK2(DEF, EXECUTION_SPACE) \
    DEF(Single, float**, EXECUTION_SPACE)    \
    DEF(Double, double**, EXECUTION_SPACE)   \
    DEF(Bool, bool**, EXECUTION_SPACE)       \
    DEF(Int8, int8**, EXECUTION_SPACE)       \
    DEF(UInt8, uint8**, EXECUTION_SPACE)     \
    DEF(Int16, int16**, EXECUTION_SPACE)     \
    DEF(UInt16, uint16**, EXECUTION_SPACE)   \
    DEF(Int32, int32**, EXECUTION_SPACE)     \
    DEF(UInt32, uint32**, EXECUTION_SPACE)   \
    DEF(Int64, int64**, EXECUTION_SPACE)     \
    DEF(UInt64, uint64**, EXECUTION_SPACE)

#define TEMPLATE_RANK3(DEF, EXECUTION_SPACE) \
    DEF(Single, float***, EXECUTION_SPACE)   \
    DEF(Double, double***, EXECUTION_SPACE)  \
    DEF(Bool, bool***, EXECUTION_SPACE)      \
    DEF(Int8, int8***, EXECUTION_SPACE)      \
    DEF(UInt8, uint8***, EXECUTION_SPACE)    \
    DEF(Int16, int16***, EXECUTION_SPACE)    \
    DEF(UInt16, uint16***, EXECUTION_SPACE)  \
    DEF(Int32, int32***, EXECUTION_SPACE)    \
    DEF(UInt32, uint32***, EXECUTION_SPACE)  \
    DEF(Int64, int64***, EXECUTION_SPACE)    \
    DEF(UInt64, uint64***, EXECUTION_SPACE)

const NativeString GetLabel(void* instance, const NdArray& ndArray) noexcept
{
    switch(ndArray.execution_space)
    {
        case ExecutionSpaceKind::Serial:
        {
            switch(ndArray.rank)
            {
                case 0:
                {
                    switch(ndArray.data_type)
                    {
                        TEMPLATE_RANK0(DEF_TEMPLATE, Serial)
                        default:
                        {
                            std::cout << "GetLabel::Serial, DataType is not supported." << std::endl;
                        }
                    }
                    break;
                }
                case 1:
                {
                    switch(ndArray.data_type)
                    {
                        TEMPLATE_RANK1(DEF_TEMPLATE, Serial)
                        default:
                        {
                            std::cout << "GetLabel::Serial, DataType is not supported." << std::endl;
                        }
                    }
                    break;
                }
                case 2:
                {
                    switch(ndArray.data_type)
                    {
                        TEMPLATE_RANK2(DEF_TEMPLATE, Serial)
                        default:
                        {
                            std::cout << "GetLabel::Serial, DataType is not supported." << std::endl;
                        }
                    }
                    break;
                }
                case 3:
                {
                    switch(ndArray.data_type)
                    {
                        TEMPLATE_RANK3(DEF_TEMPLATE, Serial)
                        default:
                        {
                            std::cout << "GetLabel::Serial, DataType is not supported." << std::endl;
                        }
                    }
                    break;
                }
                default:
                {
                    std::cout << "GetLabel::Serial, Rank is not supported." << std::endl;
                }
            }
        }
        case ExecutionSpaceKind::OpenMP:
        {
            switch(ndArray.rank)
            {
                case 0:
                {
                    switch(ndArray.data_type)
                    {
                        TEMPLATE_RANK0(DEF_TEMPLATE, OpenMP)
                        default:
                        {
                            std::cout << "GetLabel::OpenMP, DataType is not supported." << std::endl;
                        }
                    }
                    break;
                }
                case 1:
                {
                    switch(ndArray.data_type)
                    {
                        TEMPLATE_RANK1(DEF_TEMPLATE, OpenMP)
                        default:
                        {
                            std::cout << "GetLabel::OpenMP, DataType is not supported." << std::endl;
                        }
                    }
                    break;
                }
                case 2:
                {
                    switch(ndArray.data_type)
                    {
                        TEMPLATE_RANK2(DEF_TEMPLATE, OpenMP)
                        default:
                        {
                            std::cout << "GetLabel::OpenMP, DataType is not supported." << std::endl;
                        }
                    }
                    break;
                }
                case 3:
                {
                    switch(ndArray.data_type)
                    {
                        TEMPLATE_RANK3(DEF_TEMPLATE, OpenMP)
                        default:
                        {
                            std::cout << "GetLabel::OpenMP, DataType is not supported." << std::endl;
                        }
                    }
                    break;
                }
                default:
                {
                    std::cout << "GetLabel::OpenMP, Rank is not supported." << std::endl;
                }
            }
        }
        case ExecutionSpaceKind::Cuda:
        {
            switch(ndArray.rank)
            {
                case 0:
                {
                    switch(ndArray.data_type)
                    {
                        TEMPLATE_RANK0(DEF_TEMPLATE, Cuda)
                        default:
                        {
                            std::cout << "GetLabel::Cuda, DataType is not supported." << std::endl;
                        }
                    }
                    break;
                }
                case 1:
                {
                    switch(ndArray.data_type)
                    {
                        TEMPLATE_RANK1(DEF_TEMPLATE, Cuda)
                        default:
                        {
                            std::cout << "GetLabel::Cuda, DataType is not supported." << std::endl;
                        }
                    }
                    break;
                }
                case 2:
                {
                    switch(ndArray.data_type)
                    {
                        TEMPLATE_RANK2(DEF_TEMPLATE, Cuda)
                        default:
                        {
                            std::cout << "GetLabel::Cuda, DataType is not supported." << std::endl;
                        }
                    }
                    break;
                }
                case 3:
                {
                    switch(ndArray.data_type)
                    {
                        TEMPLATE_RANK3(DEF_TEMPLATE, Cuda)
                        default:
                        {
                            std::cout << "GetLabel::Cuda, DataType is not supported." << std::endl;
                        }
                    }
                    break;
                }
                default:
                {
                    std::cout << "GetLabel::Cuda, Rank is not supported." << std::endl;
                }
            }
        }
        default:
        {
            std::cout << "GetLabel ExecutionSpace is not supported." << std::endl;
        }
    }
    return {0, nullptr};
}

#undef DEF_TEMPLATE

#define DEF_TEMPLATE(TYPE_NAME, TYPE, EXECUTION_SPACE)                                                                 \
    case DataTypeKind::TYPE_NAME:                                                                                      \
    {                                                                                                                  \
        typedef Kokkos::View<TYPE, typename Kokkos::EXECUTION_SPACE::array_layout, Kokkos::EXECUTION_SPACE> view_type; \
        return (*reinterpret_cast<Teuchos::RCP<view_type>*>(instance))->size();                                        \
    }

uint64 GetSize(void* instance, const NdArray& ndArray) noexcept
{
    switch(ndArray.execution_space)
    {
        case ExecutionSpaceKind::Serial:
        {
            switch(ndArray.rank)
            {
                case 0:
                {
                    switch(ndArray.data_type)
                    {
                        TEMPLATE_RANK0(DEF_TEMPLATE, Serial)
                        default:
                        {
                            std::cout << "GetSize::Serial, DataType is not supported." << std::endl;
                        }
                    }
                    break;
                }
                case 1:
                {
                    switch(ndArray.data_type)
                    {
                        TEMPLATE_RANK1(DEF_TEMPLATE, Serial)
                        default:
                        {
                            std::cout << "GetSize::Serial, DataType is not supported." << std::endl;
                        }
                    }
                    break;
                }
                case 2:
                {
                    switch(ndArray.data_type)
                    {
                        TEMPLATE_RANK2(DEF_TEMPLATE, Serial)
                        default:
                        {
                            std::cout << "GetSize::Serial, DataType is not supported." << std::endl;
                        }
                    }
                    break;
                }
                case 3:
                {
                    switch(ndArray.data_type)
                    {
                        TEMPLATE_RANK3(DEF_TEMPLATE, Serial)
                        default:
                        {
                            std::cout << "GetSize::Serial, DataType is not supported." << std::endl;
                        }
                    }
                    break;
                }
                default:
                {
                    std::cout << "GetSize::Serial, Rank is not supported." << std::endl;
                }
            }
            break;
        }
        case ExecutionSpaceKind::OpenMP:
        {
            switch(ndArray.rank)
            {
                case 0:
                {
                    switch(ndArray.data_type)
                    {
                        TEMPLATE_RANK0(DEF_TEMPLATE, OpenMP)
                        default:
                        {
                            std::cout << "GetSize::OpenMP, DataType is not supported." << std::endl;
                        }
                    }
                    break;
                }
                case 1:
                {
                    switch(ndArray.data_type)
                    {
                        TEMPLATE_RANK1(DEF_TEMPLATE, OpenMP)
                        default:
                        {
                            std::cout << "GetSize::OpenMP, DataType is not supported." << std::endl;
                        }
                    }
                    break;
                }
                case 2:
                {
                    switch(ndArray.data_type)
                    {
                        TEMPLATE_RANK2(DEF_TEMPLATE, OpenMP)
                        default:
                        {
                            std::cout << "GetSize::OpenMP, DataType is not supported." << std::endl;
                        }
                    }
                    break;
                }
                case 3:
                {
                    switch(ndArray.data_type)
                    {
                        TEMPLATE_RANK3(DEF_TEMPLATE, OpenMP)
                        default:
                        {
                            std::cout << "GetSize::OpenMP, DataType is not supported." << std::endl;
                        }
                    }
                    break;
                }
                default:
                {
                    std::cout << "GetSize::OpenMP, Rank is not supported." << std::endl;
                }
            }
            break;
        }
        case ExecutionSpaceKind::Cuda:
        {
            switch(ndArray.rank)
            {
                case 0:
                {
                    switch(ndArray.data_type)
                    {
                        TEMPLATE_RANK0(DEF_TEMPLATE, Cuda)
                        default:
                        {
                            std::cout << "GetSize::Cuda, DataType is not supported." << std::endl;
                        }
                    }
                    break;
                }
                case 1:
                {
                    switch(ndArray.data_type)
                    {
                        TEMPLATE_RANK1(DEF_TEMPLATE, Cuda)
                        default:
                        {
                            std::cout << "GetSize::Cuda, DataType is not supported." << std::endl;
                        }
                    }
                    break;
                }
                case 2:
                {
                    switch(ndArray.data_type)
                    {
                        TEMPLATE_RANK2(DEF_TEMPLATE, Cuda)
                        default:
                        {
                            std::cout << "GetSize::Cuda, DataType is not supported." << std::endl;
                        }
                    }
                    break;
                }
                case 3:
                {
                    switch(ndArray.data_type)
                    {
                        TEMPLATE_RANK3(DEF_TEMPLATE, Cuda)
                        default:
                        {
                            std::cout << "GetSize::Cuda, DataType is not supported." << std::endl;
                        }
                    }
                    break;
                }
                default:
                {
                    std::cout << "GetSize::Cuda, Rank is not supported." << std::endl;
                }
            }
            break;
        }
        default:
        {
            std::cout << "GetSize ExecutionSpace is not supported." << std::endl;
        }
    }
    return 0;
}

#undef DEF_TEMPLATE

#define DEF_TEMPLATE(TYPE_NAME, TYPE, EXECUTION_SPACE)                                                                 \
    case DataTypeKind::TYPE_NAME:                                                                                      \
    {                                                                                                                  \
        typedef Kokkos::View<TYPE, typename Kokkos::EXECUTION_SPACE::array_layout, Kokkos::EXECUTION_SPACE> view_type; \
        return (*reinterpret_cast<Teuchos::RCP<view_type>*>(instance))->stride(dim);                                   \
    }

uint64 GetStride(void* instance, const NdArray& ndArray, const uint32& dim) noexcept
{
    switch(ndArray.execution_space)
    {
        case ExecutionSpaceKind::Serial:
        {
            switch(ndArray.rank)
            {
                case 0:
                {
                    switch(ndArray.data_type)
                    {
                        TEMPLATE_RANK0(DEF_TEMPLATE, Serial)
                        default:
                        {
                            std::cout << "GetStride::Serial, DataType is not supported." << std::endl;
                        }
                    }
                    break;
                }
                case 1:
                {
                    switch(ndArray.data_type)
                    {
                        TEMPLATE_RANK1(DEF_TEMPLATE, Serial)
                        default:
                        {
                            std::cout << "GetStride::Serial, DataType is not supported." << std::endl;
                        }
                    }
                    break;
                }
                case 2:
                {
                    switch(ndArray.data_type)
                    {
                        TEMPLATE_RANK2(DEF_TEMPLATE, Serial)
                        default:
                        {
                            std::cout << "GetStride::Serial, DataType is not supported." << std::endl;
                        }
                    }
                    break;
                }
                case 3:
                {
                    switch(ndArray.data_type)
                    {
                        TEMPLATE_RANK3(DEF_TEMPLATE, Serial)
                        default:
                        {
                            std::cout << "GetStride::Serial, DataType is not supported." << std::endl;
                        }
                    }
                    break;
                }
                default:
                {
                    std::cout << "GetStride::Serial, Rank is not supported." << std::endl;
                }
            }
        }
        case ExecutionSpaceKind::OpenMP:
        {
            switch(ndArray.rank)
            {
                case 0:
                {
                    switch(ndArray.data_type)
                    {
                        TEMPLATE_RANK0(DEF_TEMPLATE, OpenMP)
                        default:
                        {
                            std::cout << "GetStride::OpenMP, DataType is not supported." << std::endl;
                        }
                    }
                    break;
                }
                case 1:
                {
                    switch(ndArray.data_type)
                    {
                        TEMPLATE_RANK1(DEF_TEMPLATE, OpenMP)
                        default:
                        {
                            std::cout << "GetStride::OpenMP, DataType is not supported." << std::endl;
                        }
                    }
                    break;
                }
                case 2:
                {
                    switch(ndArray.data_type)
                    {
                        TEMPLATE_RANK2(DEF_TEMPLATE, OpenMP)
                        default:
                        {
                            std::cout << "GetStride::OpenMP, DataType is not supported." << std::endl;
                        }
                    }
                    break;
                }
                case 3:
                {
                    switch(ndArray.data_type)
                    {
                        TEMPLATE_RANK3(DEF_TEMPLATE, OpenMP)
                        default:
                        {
                            std::cout << "GetStride::OpenMP, DataType is not supported." << std::endl;
                        }
                    }
                    break;
                }
                default:
                {
                    std::cout << "GetStride::OpenMP, Rank is not supported." << std::endl;
                }
            }
        }
        case ExecutionSpaceKind::Cuda:
        {
            switch(ndArray.rank)
            {
                case 0:
                {
                    switch(ndArray.data_type)
                    {
                        TEMPLATE_RANK0(DEF_TEMPLATE, Cuda)
                        default:
                        {
                            std::cout << "GetStride::Cuda, DataType is not supported." << std::endl;
                        }
                    }
                    break;
                }
                case 1:
                {
                    switch(ndArray.data_type)
                    {
                        TEMPLATE_RANK1(DEF_TEMPLATE, Cuda)
                        default:
                        {
                            std::cout << "GetStride::Cuda, DataType is not supported." << std::endl;
                        }
                    }
                    break;
                }
                case 2:
                {
                    switch(ndArray.data_type)
                    {
                        TEMPLATE_RANK2(DEF_TEMPLATE, Cuda)
                        default:
                        {
                            std::cout << "GetStride::Cuda, DataType is not supported." << std::endl;
                        }
                    }
                    break;
                }
                case 3:
                {
                    switch(ndArray.data_type)
                    {
                        TEMPLATE_RANK3(DEF_TEMPLATE, Cuda)
                        default:
                        {
                            std::cout << "GetStride::Cuda, DataType is not supported." << std::endl;
                        }
                    }
                    break;
                }
                default:
                {
                    std::cout << "GetStride::Cuda, Rank is not supported." << std::endl;
                }
            }
        }
        default:
        {
            std::cout << "GetStride ExecutionSpace is not supported." << std::endl;
        }
    }
    return 0;
}

#undef DEF_TEMPLATE

#define DEF_TEMPLATE(TYPE_NAME, TYPE, EXECUTION_SPACE)                                                                 \
    case DataTypeKind::TYPE_NAME:                                                                                      \
    {                                                                                                                  \
        typedef Kokkos::View<TYPE, typename Kokkos::EXECUTION_SPACE::array_layout, Kokkos::EXECUTION_SPACE> view_type; \
        return (*reinterpret_cast<Teuchos::RCP<view_type>*>(instance))->extent(dim);                                   \
    }

uint64 GetExtent(void* instance, const NdArray& ndArray, const uint32& dim) noexcept
{
    switch(ndArray.execution_space)
    {
        case ExecutionSpaceKind::Serial:
        {
            switch(ndArray.rank)
            {
                case 0:
                {
                    switch(ndArray.data_type)
                    {
                        TEMPLATE_RANK0(DEF_TEMPLATE, Serial)
                        default:
                        {
                            std::cout << "GetExtent::Serial, DataType is not supported." << std::endl;
                        }
                    }
                    break;
                }
                case 1:
                {
                    switch(ndArray.data_type)
                    {
                        TEMPLATE_RANK1(DEF_TEMPLATE, Serial)
                        default:
                        {
                            std::cout << "GetExtent::Serial, DataType is not supported." << std::endl;
                        }
                    }
                    break;
                }
                case 2:
                {
                    switch(ndArray.data_type)
                    {
                        TEMPLATE_RANK2(DEF_TEMPLATE, Serial)
                        default:
                        {
                            std::cout << "GetExtent::Serial, DataType is not supported." << std::endl;
                        }
                    }
                    break;
                }
                case 3:
                {
                    switch(ndArray.data_type)
                    {
                        TEMPLATE_RANK3(DEF_TEMPLATE, Serial)
                        default:
                        {
                            std::cout << "GetExtent::Serial, DataType is not supported." << std::endl;
                        }
                    }
                    break;
                }
                default:
                {
                    std::cout << "GetExtent::Serial, Rank is not supported." << std::endl;
                }
            }
        }
        case ExecutionSpaceKind::OpenMP:
        {
            switch(ndArray.rank)
            {
                case 0:
                {
                    switch(ndArray.data_type)
                    {
                        TEMPLATE_RANK0(DEF_TEMPLATE, OpenMP)
                        default:
                        {
                            std::cout << "GetExtent::OpenMP, DataType is not supported." << std::endl;
                        }
                    }
                    break;
                }
                case 1:
                {
                    switch(ndArray.data_type)
                    {
                        TEMPLATE_RANK1(DEF_TEMPLATE, OpenMP)
                        default:
                        {
                            std::cout << "GetExtent::OpenMP, DataType is not supported." << std::endl;
                        }
                    }
                    break;
                }
                case 2:
                {
                    switch(ndArray.data_type)
                    {
                        TEMPLATE_RANK2(DEF_TEMPLATE, OpenMP)
                        default:
                        {
                            std::cout << "GetExtent::OpenMP, DataType is not supported." << std::endl;
                        }
                    }
                    break;
                }
                case 3:
                {
                    switch(ndArray.data_type)
                    {
                        TEMPLATE_RANK3(DEF_TEMPLATE, OpenMP)
                        default:
                        {
                            std::cout << "GetExtent::OpenMP, DataType is not supported." << std::endl;
                        }
                    }
                    break;
                }
                default:
                {
                    std::cout << "GetExtent::OpenMP, Rank is not supported." << std::endl;
                }
            }
        }
        case ExecutionSpaceKind::Cuda:
        {
            switch(ndArray.rank)
            {
                case 0:
                {
                    switch(ndArray.data_type)
                    {
                        TEMPLATE_RANK0(DEF_TEMPLATE, Cuda)
                        default:
                        {
                            std::cout << "GetExtent::Cuda, DataType is not supported." << std::endl;
                        }
                    }
                    break;
                }
                case 1:
                {
                    switch(ndArray.data_type)
                    {
                        TEMPLATE_RANK1(DEF_TEMPLATE, Cuda)
                        default:
                        {
                            std::cout << "GetExtent::Cuda, DataType is not supported." << std::endl;
                        }
                    }
                    break;
                }
                case 2:
                {
                    switch(ndArray.data_type)
                    {
                        TEMPLATE_RANK2(DEF_TEMPLATE, Cuda)
                        default:
                        {
                            std::cout << "GetExtent::Cuda, DataType is not supported." << std::endl;
                        }
                    }
                    break;
                }
                case 3:
                {
                    switch(ndArray.data_type)
                    {
                        TEMPLATE_RANK3(DEF_TEMPLATE, Cuda)
                        default:
                        {
                            std::cout << "GetExtent::Cuda, DataType is not supported." << std::endl;
                        }
                    }
                    break;
                }
                default:
                {
                    std::cout << "GetExtent::Cuda, Rank is not supported." << std::endl;
                }
            }
        }
        default:
        {
            std::cout << "GetExtent ExecutionSpace is not supported." << std::endl;
        }
    }
    return 0;
}

#undef DEF_TEMPLATE
#undef TEMPLATE_RANK0
#undef TEMPLATE_RANK1
#undef TEMPLATE_RANK2
#undef TEMPLATE_RANK3

template<typename DataType, class ExecutionSpace, typename Layout>
__inline static NdArray ViewToNdArrayRank0(void* instance) noexcept
{
    typedef Kokkos::View<DataType, Layout, ExecutionSpace> view_type;

    typedef NdArrayTraits<typename view_type::traits::non_const_value_type, typename view_type::traits::execution_space, typename view_type::traits::array_layout, 0> ndarray_traits;

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
__inline static NdArray ViewToNdArrayRank1(void* instance) noexcept
{
    typedef Kokkos::View<DataType*, Layout, ExecutionSpace> view_type;

    typedef NdArrayTraits<typename view_type::traits::non_const_value_type, typename view_type::traits::execution_space, typename view_type::traits::array_layout, 1> ndarray_traits;

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
__inline static NdArray ViewToNdArrayRank2(void* instance) noexcept
{
    typedef Kokkos::View<DataType**, Layout, ExecutionSpace> view_type;

    typedef NdArrayTraits<typename view_type::traits::non_const_value_type, typename view_type::traits::execution_space, typename view_type::traits::array_layout, 2> ndarray_traits;

    Teuchos::RCP<view_type>* view = reinterpret_cast<Teuchos::RCP<view_type>*>(instance);

    NdArray ndArray;
    ndArray.data_type       = ndarray_traits::data_type;
    ndArray.rank            = 1;
    ndArray.layout          = ndarray_traits::layout;
    ndArray.execution_space = ndarray_traits::execution_space;
    ndArray.data            = (*view)->data();
    ndArray.label           = (*view)->label().c_str();

    ndArray.dims[0] = (*view)->extent(0);
    ndArray.dims[1] = (*view)->extent(1);

    return ndArray;
}

template<typename DataType, class ExecutionSpace, typename Layout>
__inline static NdArray ViewToNdArrayRank3(void* instance) noexcept
{
    typedef Kokkos::View<DataType***, Layout, ExecutionSpace> view_type;

    typedef NdArrayTraits<typename view_type::traits::non_const_value_type, typename view_type::traits::execution_space, typename view_type::traits::array_layout, 3> ndarray_traits;

    Teuchos::RCP<view_type>* view = reinterpret_cast<Teuchos::RCP<view_type>*>(instance);

    NdArray ndArray;
    ndArray.data_type       = ndarray_traits::data_type;
    ndArray.rank            = 1;
    ndArray.layout          = ndarray_traits::layout;
    ndArray.execution_space = ndarray_traits::execution_space;
    ndArray.data            = (*view)->data();
    ndArray.label           = (*view)->label().c_str();

    ndArray.dims[0] = (*view)->extent(0);
    ndArray.dims[1] = (*view)->extent(1);
    ndArray.dims[2] = (*view)->extent(2);

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
