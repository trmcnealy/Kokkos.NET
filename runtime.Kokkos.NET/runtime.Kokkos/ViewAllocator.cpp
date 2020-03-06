
#include "runtime.Kokkos/KokkosApi.h"

#include <Teuchos_RCP.hpp>

// template<typename T>
//__inline void* operator new(unsigned long long, T* p)
//{
//    return p;
//}

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

#define DEF_TEMPLATE_RANK0(TYPE_NAME, TYPE, EXECUTION_SPACE)                                                                                             \
    case DataTypeKind::TYPE_NAME:                                                                                                                        \
    {                                                                                                                                                    \
        typedef ViewBuilder<DataTypeKind::TYPE_NAME, 0, ExecutionSpaceKind::EXECUTION_SPACE>::ViewType view_type;                                        \
        view_type*                                                                                     view = new view_type(std::string(ndArray.label)); \
        ndArray.data                                                                                        = (void*)view->data();                       \
        new(instance) Teuchos::RCP<view_type>(view);                                                                                                     \
        break;                                                                                                                                           \
    }

#define DEF_TEMPLATE_RANK1(TYPE_NAME, TYPE, EXECUTION_SPACE)                                                                                                 \
    case DataTypeKind::TYPE_NAME:                                                                                                                            \
    {                                                                                                                                                        \
        typedef ViewBuilder<DataTypeKind::TYPE_NAME, 1, ExecutionSpaceKind::EXECUTION_SPACE>::ViewType view_type;                                            \
        view_type*                                                                                     view = new view_type(std::string(ndArray.label), n0); \
        ndArray.data                                                                                        = (void*)view->data();                           \
        new(instance) Teuchos::RCP<view_type>(view);                                                                                                         \
        break;                                                                                                                                               \
    }

#define DEF_TEMPLATE_RANK2(TYPE_NAME, TYPE, EXECUTION_SPACE)                                                                                                     \
    case DataTypeKind::TYPE_NAME:                                                                                                                                \
    {                                                                                                                                                            \
        typedef ViewBuilder<DataTypeKind::TYPE_NAME, 2, ExecutionSpaceKind::EXECUTION_SPACE>::ViewType view_type;                                                \
        view_type*                                                                                     view = new view_type(std::string(ndArray.label), n0, n1); \
        ndArray.data                                                                                        = (void*)view->data();                               \
        new(instance) Teuchos::RCP<view_type>(view);                                                                                                             \
        break;                                                                                                                                                   \
    }

#define DEF_TEMPLATE_RANK3(TYPE_NAME, TYPE, EXECUTION_SPACE)                                                                                                         \
    case DataTypeKind::TYPE_NAME:                                                                                                                                    \
    {                                                                                                                                                                \
        typedef ViewBuilder<DataTypeKind::TYPE_NAME, 3, ExecutionSpaceKind::EXECUTION_SPACE>::ViewType view_type;                                                    \
        view_type*                                                                                     view = new view_type(std::string(ndArray.label), n0, n1, n2); \
        ndArray.data                                                                                        = (void*)view->data();                                   \
        new(instance) Teuchos::RCP<view_type>(view);                                                                                                                 \
        break;                                                                                                                                                       \
    }

void CreateViewRank0(void* instance, NdArray& ndArray) noexcept
{
    switch(ndArray.execution_space)
    {
        case ExecutionSpaceKind::Serial:
        {
            switch(ndArray.data_type)
            {
                TEMPLATE(DEF_TEMPLATE_RANK0, Serial)
                default:
                {
                    std::cout << "CreateViewRank0::Serial, DataType is not supported." << std::endl;
                }
            }
            break;
        }
        case ExecutionSpaceKind::OpenMP:
        {
            switch(ndArray.data_type)
            {
                TEMPLATE(DEF_TEMPLATE_RANK0, OpenMP)
                default:
                {
                    std::cout << "CreateViewRank0::OpenMP, DataType is not supported." << std::endl;
                }
            }
            break;
        }
        case ExecutionSpaceKind::Cuda:
        {
            switch(ndArray.data_type)
            {
                TEMPLATE(DEF_TEMPLATE_RANK0, Cuda)
                default:
                {
                    std::cout << "CreateViewRank0::Cuda, DataType is not supported." << std::endl;
                }
            }
            break;
        }
        default:
        {
            std::cout << "CreateViewRank0 ExecutionSpace is not supported." << std::endl;
        }
    }
}

void CreateViewRank1(void* instance, NdArray& ndArray, const size_type& n0) noexcept
{
    switch(ndArray.execution_space)
    {
        case ExecutionSpaceKind::Serial:
        {
            switch(ndArray.data_type)
            {
                TEMPLATE(DEF_TEMPLATE_RANK1, Serial)
                default:
                {
                    std::cout << "CreateViewRank1::Serial, DataType is not supported." << std::endl;
                }
            }
            break;
        }
        case ExecutionSpaceKind::OpenMP:
        {
            switch(ndArray.data_type)
            {
                TEMPLATE(DEF_TEMPLATE_RANK1, OpenMP)
                default:
                {
                    std::cout << "CreateViewRank1::OpenMP, DataType is not supported." << std::endl;
                }
            }
            break;
        }
        case ExecutionSpaceKind::Cuda:
        {
            switch(ndArray.data_type)
            {
                TEMPLATE(DEF_TEMPLATE_RANK1, Cuda)
                default:
                {
                    std::cout << "CreateViewRank1::Cuda, DataType is not supported." << std::endl;
                }
            }
            break;
        }
        default:
        {
            std::cout << "CreateViewRank1 ExecutionSpace is not supported." << std::endl;
        }
    }
}

void CreateViewRank2(void* instance, NdArray& ndArray, const size_type& n0, const size_type& n1) noexcept
{
    switch(ndArray.execution_space)
    {
        case ExecutionSpaceKind::Serial:
        {
            switch(ndArray.data_type)
            {
                TEMPLATE(DEF_TEMPLATE_RANK2, Serial)
                default:
                {
                    std::cout << "CreateViewRank2::Serial, DataType is not supported." << std::endl;
                }
            }
            break;
        }
        case ExecutionSpaceKind::OpenMP:
        {
            switch(ndArray.data_type)
            {
                TEMPLATE(DEF_TEMPLATE_RANK2, OpenMP)
                default:
                {
                    std::cout << "CreateViewRank2::OpenMP, DataType is not supported." << std::endl;
                }
            }
            break;
        }
        case ExecutionSpaceKind::Cuda:
        {
            switch(ndArray.data_type)
            {
                TEMPLATE(DEF_TEMPLATE_RANK2, Cuda)
                default:
                {
                    std::cout << "CreateViewRank2::Cuda, DataType is not supported." << std::endl;
                }
            }
            break;
        }
        default:
        {
            std::cout << "CreateViewRank2 ExecutionSpace is not supported." << std::endl;
        }
    }
}

void CreateViewRank3(void* instance, NdArray& ndArray, const size_type& n0, const size_type& n1, const size_type& n2) noexcept
{
    switch(ndArray.execution_space)
    {
        case ExecutionSpaceKind::Serial:
        {
            switch(ndArray.data_type)
            {
                TEMPLATE(DEF_TEMPLATE_RANK3, Serial)
                default:
                {
                    std::cout << "CreateViewRank3::Serial, DataType is not supported." << std::endl;
                }
            }
            break;
        }
        case ExecutionSpaceKind::OpenMP:
        {
            switch(ndArray.data_type)
            {
                TEMPLATE(DEF_TEMPLATE_RANK3, OpenMP)
                default:
                {
                    std::cout << "CreateViewRank3::OpenMP, DataType is not supported." << std::endl;
                }
            }
            break;
        }
        case ExecutionSpaceKind::Cuda:
        {
            switch(ndArray.data_type)
            {
                TEMPLATE(DEF_TEMPLATE_RANK3, Cuda)
                default:
                {
                    std::cout << "CreateViewRank3::Cuda, DataType is not supported." << std::endl;
                }
            }
            break;
        }
        default:
        {
            std::cout << "CreateViewRank3 ExecutionSpace is not supported." << std::endl;
        }
    }
}

void CreateView(void* instance, NdArray& ndArray) noexcept
{
    switch(ndArray.rank)
    {
        case 0:
        {
            CreateViewRank0(instance, ndArray);
            break;
        }
        case 1:
        {
            const uint64 dim0 = std::max(ndArray.dims[0], 1ull);
            CreateViewRank1(instance, ndArray, dim0);
            break;
        }
        case 2:
        {
            const uint64 dim0 = std::max(ndArray.dims[0], 1ull);
            const uint64 dim1 = std::max(ndArray.dims[1], 1ull);
            CreateViewRank2(instance, ndArray, dim0, dim1);
            break;
        }
        case 3:
        {
            const uint64 dim0 = std::max(ndArray.dims[0], 1ull);
            const uint64 dim1 = std::max(ndArray.dims[1], 1ull);
            const uint64 dim2 = std::max(ndArray.dims[2], 1ull);
            CreateViewRank3(instance, ndArray, dim0, dim1, dim2);
            break;
        }
        default:
        {
            break;
        }
    }
}

// KOKKOS_NET_API_EXTERN void DisposeView(KokkosViewRank0Ref view)
//{
//    unwrap(view)->~DualView();
//}

// t_host view_host() const
// subview
// sync
// need_sync
// sync_host
// modify_host
// span
// span_is_contiguous
// stride
// extent
// realloc
// resize
