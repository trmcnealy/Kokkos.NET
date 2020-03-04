
#include "runtime.Kokkos/KokkosApi.h"

#define DEF_TEMPLATE_RANK0(TYPE_NAME, TYPE, EXECUTION_SPACE)                                                        \
    case DataTypeKind::TYPE_NAME:                                                                                   \
    {                                                                                                               \
        typedef TYPE                                                                                   value_type;  \
        typedef ViewBuilder<DataTypeKind::TYPE_NAME, 0, ExecutionSpaceKind::EXECUTION_SPACE>::ViewType view_type;   \
        typedef typename view_type::HostMirror                                                         mirror_type; \
                                                                                                                    \
        view_type& view = *(*reinterpret_cast<Teuchos::RCP<view_type>*>(instance));                                 \
                                                                                                                    \
        value_type* data = reinterpret_cast<value_type*>(values);                                                   \
                                                                                                                    \
        mirror_type view_mirror = Kokkos::create_mirror_view(view);                                                 \
                                                                                                                    \
        view_mirror() = *data;                                                                                      \
                                                                                                                    \
        Kokkos::deep_copy(view, view_mirror);                                                                       \
        break;                                                                                                      \
    }

#define DEF_TEMPLATE_RANK1(TYPE_NAME, TYPE, EXECUTION_SPACE)                                                        \
    case DataTypeKind::TYPE_NAME:                                                                                   \
    {                                                                                                               \
        typedef TYPE                                                                                   value_type;  \
        typedef ViewBuilder<DataTypeKind::TYPE_NAME, 1, ExecutionSpaceKind::EXECUTION_SPACE>::ViewType view_type;   \
        typedef view_type::HostMirror                                                                  mirror_type; \
                                                                                                                    \
        view_type& view = *(*reinterpret_cast<Teuchos::RCP<view_type>*>(instance));                                 \
                                                                                                                    \
        value_type* data = reinterpret_cast<value_type*>(values);                                                   \
                                                                                                                    \
        mirror_type view_mirror = Kokkos::create_mirror_view(view);                                                 \
                                                                                                                    \
        for(size_type i0 = 0; i0 < view.extent(0); ++i0)                                                            \
        {                                                                                                           \
            view_mirror(i0) = data[i0];                                                                             \
        }                                                                                                           \
                                                                                                                    \
        Kokkos::deep_copy(view, view_mirror);                                                                       \
        break;                                                                                                      \
    }

#define DEF_TEMPLATE_RANK2(TYPE_NAME, TYPE, EXECUTION_SPACE)                                                        \
    case DataTypeKind::TYPE_NAME:                                                                                   \
    {                                                                                                               \
        typedef TYPE                                                                                   value_type;  \
        typedef ViewBuilder<DataTypeKind::TYPE_NAME, 2, ExecutionSpaceKind::EXECUTION_SPACE>::ViewType view_type;   \
        typedef view_type::HostMirror                                                                  mirror_type; \
                                                                                                                    \
        view_type& view = *(*reinterpret_cast<Teuchos::RCP<view_type>*>(instance));                                 \
                                                                                                                    \
        value_type* data = reinterpret_cast<value_type*>(values);                                                   \
                                                                                                                    \
        mirror_type view_mirror = Kokkos::create_mirror_view(view);                                                 \
                                                                                                                    \
        for(size_type i0 = 0; i0 < view.extent(0); ++i0)                                                            \
        {                                                                                                           \
            for(size_type i1 = 0; i1 < view.extent(1); ++i1)                                                        \
            {                                                                                                       \
                view_mirror(i0, i1) = data[i0 * view.extent(1) + i1];                                               \
            }                                                                                                       \
        }                                                                                                           \
                                                                                                                    \
        Kokkos::deep_copy(view, view_mirror);                                                                       \
        break;                                                                                                      \
    }

#define DEF_TEMPLATE_RANK3(TYPE_NAME, TYPE, EXECUTION_SPACE)                                                        \
    case DataTypeKind::TYPE_NAME:                                                                                   \
    {                                                                                                               \
        typedef TYPE                                                                                   value_type;  \
        typedef ViewBuilder<DataTypeKind::TYPE_NAME, 3, ExecutionSpaceKind::EXECUTION_SPACE>::ViewType view_type;   \
        typedef view_type::HostMirror                                                                  mirror_type; \
                                                                                                                    \
        view_type& view = *(*reinterpret_cast<Teuchos::RCP<view_type>*>(instance));                                 \
                                                                                                                    \
        value_type* data = reinterpret_cast<value_type*>(values);                                                   \
                                                                                                                    \
        mirror_type view_mirror = Kokkos::create_mirror_view(view);                                                 \
                                                                                                                    \
        for(size_type i0 = 0; i0 < view.extent(0); ++i0)                                                            \
        {                                                                                                           \
            for(size_type i1 = 0; i1 < view.extent(1); ++i1)                                                        \
            {                                                                                                       \
                for(size_type i2 = 0; i2 < view.extent(2); ++i2)                                                    \
                {                                                                                                   \
                    view_mirror(i0, i1, i2) = data[(i0 * view.extent(1) + i1) * view.extent(2) + i2];               \
                }                                                                                                   \
            }                                                                                                       \
        }                                                                                                           \
                                                                                                                    \
        Kokkos::deep_copy(view, view_mirror);                                                                       \
        break;                                                                                                      \
    }

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

void CopyTo(void* instance, const NdArray& ndArray, ValueType* values) noexcept
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
                        TEMPLATE(DEF_TEMPLATE_RANK0, Serial)
                        default:
                        {
                            std::cout << "CopyTo::Serial, DataType is not supported." << std::endl;
                        }
                    }
                    break;
                }
                case 1:
                {
                    switch(ndArray.data_type)
                    {
                        TEMPLATE(DEF_TEMPLATE_RANK1, Serial)
                        default:
                        {
                            std::cout << "CopyTo::Serial, DataType is not supported." << std::endl;
                        }
                    }
                    break;
                }
                case 2:
                {
                    switch(ndArray.data_type)
                    {
                        TEMPLATE(DEF_TEMPLATE_RANK2, Serial)
                        default:
                        {
                            std::cout << "CopyTo::Serial, DataType is not supported." << std::endl;
                        }
                    }
                    break;
                }
                case 3:
                {
                    switch(ndArray.data_type)
                    {
                        TEMPLATE(DEF_TEMPLATE_RANK3, Serial)
                        default:
                        {
                            std::cout << "CopyTo::Serial, DataType is not supported." << std::endl;
                        }
                    }
                    break;
                }
                default:
                {
                    std::cout << "CopyTo::Serial, Rank is not supported." << std::endl;
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
                        TEMPLATE(DEF_TEMPLATE_RANK0, OpenMP)
                        default:
                        {
                            std::cout << "CopyTo::OpenMP, DataType is not supported." << std::endl;
                        }
                    }
                    break;
                }
                case 1:
                {
                    switch(ndArray.data_type)
                    {
                        TEMPLATE(DEF_TEMPLATE_RANK1, OpenMP)
                        default:
                        {
                            std::cout << "CopyTo::OpenMP, DataType is not supported." << std::endl;
                        }
                    }
                    break;
                }
                case 2:
                {
                    switch(ndArray.data_type)
                    {
                        TEMPLATE(DEF_TEMPLATE_RANK2, OpenMP)
                        default:
                        {
                            std::cout << "CopyTo::OpenMP, DataType is not supported." << std::endl;
                        }
                    }
                    break;
                }
                case 3:
                {
                    switch(ndArray.data_type)
                    {
                        TEMPLATE(DEF_TEMPLATE_RANK3, OpenMP)
                        default:
                        {
                            std::cout << "CopyTo::OpenMP, DataType is not supported." << std::endl;
                        }
                    }
                    break;
                }
                default:
                {
                    std::cout << "CopyTo::OpenMP, Rank is not supported." << std::endl;
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
                        TEMPLATE(DEF_TEMPLATE_RANK0, Cuda)
                        default:
                        {
                            std::cout << "CopyTo::Cuda, DataType is not supported." << std::endl;
                        }
                    }
                    break;
                }
                case 1:
                {
                    switch(ndArray.data_type)
                    {
                        TEMPLATE(DEF_TEMPLATE_RANK1, Cuda)
                        default:
                        {
                            std::cout << "CopyTo::Cuda, DataType is not supported." << std::endl;
                        }
                    }
                    break;
                }
                case 2:
                {
                    switch(ndArray.data_type)
                    {
                        TEMPLATE(DEF_TEMPLATE_RANK2, Cuda)
                        default:
                        {
                            std::cout << "CopyTo::Cuda, DataType is not supported." << std::endl;
                        }
                    }
                    break;
                }
                case 3:
                {
                    switch(ndArray.data_type)
                    {
                        TEMPLATE(DEF_TEMPLATE_RANK3, Cuda)
                        default:
                        {
                            std::cout << "CopyTo::Cuda, DataType is not supported." << std::endl;
                        }
                    }
                    break;
                }
                default:
                {
                    std::cout << "CopyTo::Cuda, Rank is not supported." << std::endl;
                }
            }
            break;
        }
        default:
        {
            std::cout << "CopyTo ExecutionSpace is not supported." << std::endl;
        }
    }
}

#undef DEF_TEMPLATE_RANK0
#undef DEF_TEMPLATE_RANK1
#undef DEF_TEMPLATE_RANK2
#undef DEF_TEMPLATE_RANK3
