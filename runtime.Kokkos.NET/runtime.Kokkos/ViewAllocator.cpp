
#include "runtime.Kokkos/KokkosApi.h"

#include <Teuchos_RCP.hpp>

// template<typename T>
//__inline void* operator new(unsigned long long, T* p)
//{
//    return p;
//}

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

#define DEF_TEMPLATE_RANK0(TYPE_NAME, TYPE, EXECUTION_SPACE)                                                                                                                       \
    case DataTypeKind::TYPE_NAME:                                                                                                                                                  \
    {                                                                                                                                                                              \
        typedef ViewBuilder<DataTypeKind::TYPE_NAME, 0, ExecutionSpaceKind::EXECUTION_SPACE>::ViewType view_type;                                                                  \
        view_type*                                                                                     view = new view_type(std::string(ndArray->label.ToString()));               \
        ndArray->data                                                                                       = (void*)view->data();                                                 \
        new (instance) Teuchos::RCP<view_type>(view);                                                                                                                              \
        break;                                                                                                                                                                     \
    }

#define DEF_TEMPLATE_RANK1(TYPE_NAME, TYPE, EXECUTION_SPACE)                                                                                                                       \
    case DataTypeKind::TYPE_NAME:                                                                                                                                                  \
    {                                                                                                                                                                              \
        typedef ViewBuilder<DataTypeKind::TYPE_NAME, 1, ExecutionSpaceKind::EXECUTION_SPACE>::ViewType view_type;                                                                  \
        view_type*                                                                                     view = new view_type(std::string(ndArray->label.ToString()), n0);           \
        ndArray->data                                                                                       = (void*)view->data();                                                 \
        new (instance) Teuchos::RCP<view_type>(view);                                                                                                                              \
        break;                                                                                                                                                                     \
    }

#define DEF_TEMPLATE_RANK2(TYPE_NAME, TYPE, EXECUTION_SPACE)                                                                                                                       \
    case DataTypeKind::TYPE_NAME:                                                                                                                                                  \
    {                                                                                                                                                                              \
        typedef ViewBuilder<DataTypeKind::TYPE_NAME, 2, ExecutionSpaceKind::EXECUTION_SPACE>::ViewType view_type;                                                                  \
        view_type*                                                                                     view = new view_type(std::string(ndArray->label.ToString()), n0, n1);       \
        ndArray->data                                                                                       = (void*)view->data();                                                 \
        new (instance) Teuchos::RCP<view_type>(view);                                                                                                                              \
        break;                                                                                                                                                                     \
    }

#define DEF_TEMPLATE_RANK3(TYPE_NAME, TYPE, EXECUTION_SPACE)                                                                                                                       \
    case DataTypeKind::TYPE_NAME:                                                                                                                                                  \
    {                                                                                                                                                                              \
        typedef ViewBuilder<DataTypeKind::TYPE_NAME, 3, ExecutionSpaceKind::EXECUTION_SPACE>::ViewType view_type;                                                                  \
        view_type*                                                                                     view = new view_type(std::string(ndArray->label.ToString()), n0, n1, n2);   \
        ndArray->data                                                                                       = (void*)view->data();                                                 \
        new (instance) Teuchos::RCP<view_type>(view);                                                                                                                              \
        break;                                                                                                                                                                     \
    }

#define DEF_TEMPLATE_RANK4(TYPE_NAME, TYPE, EXECUTION_SPACE)                                                                                                                         \
    case DataTypeKind::TYPE_NAME:                                                                                                                                                    \
    {                                                                                                                                                                                \
        typedef ViewBuilder<DataTypeKind::TYPE_NAME, 4, ExecutionSpaceKind::EXECUTION_SPACE>::ViewType view_type;                                                                    \
        view_type*                                                                                     view = new view_type(std::string(ndArray->label.ToString()), n0, n1, n2, n3); \
        ndArray->data                                                                                       = (void*)view->data();                                                   \
        new (instance) Teuchos::RCP<view_type>(view);                                                                                                                                \
        break;                                                                                                                                                                       \
    }

#define DEF_TEMPLATE_RANK5(TYPE_NAME, TYPE, EXECUTION_SPACE)                                                                                                                       \
    case DataTypeKind::TYPE_NAME:                                                                                                                                                  \
    {                                                                                                                                                                              \
        typedef ViewBuilder<DataTypeKind::TYPE_NAME, 5, ExecutionSpaceKind::EXECUTION_SPACE>::ViewType view_type;                                                                  \
        view_type* view = new view_type(std::string(ndArray->label.ToString()), n0, n1, n2, n3, n4);                                                                               \
        ndArray->data   = (void*)view->data();                                                                                                                                     \
        new (instance) Teuchos::RCP<view_type>(view);                                                                                                                              \
        break;                                                                                                                                                                     \
    }

#define DEF_TEMPLATE_RANK6(TYPE_NAME, TYPE, EXECUTION_SPACE)                                                                                                                       \
    case DataTypeKind::TYPE_NAME:                                                                                                                                                  \
    {                                                                                                                                                                              \
        typedef ViewBuilder<DataTypeKind::TYPE_NAME, 6, ExecutionSpaceKind::EXECUTION_SPACE>::ViewType view_type;                                                                  \
        view_type* view = new view_type(std::string(ndArray->label.ToString()), n0, n1, n2, n3, n4, n5);                                                                           \
        ndArray->data   = (void*)view->data();                                                                                                                                     \
        new (instance) Teuchos::RCP<view_type>(view);                                                                                                                              \
        break;                                                                                                                                                                     \
    }

#define DEF_TEMPLATE_RANK7(TYPE_NAME, TYPE, EXECUTION_SPACE)                                                                                                                       \
    case DataTypeKind::TYPE_NAME:                                                                                                                                                  \
    {                                                                                                                                                                              \
        typedef ViewBuilder<DataTypeKind::TYPE_NAME, 7, ExecutionSpaceKind::EXECUTION_SPACE>::ViewType view_type;                                                                  \
        view_type* view = new view_type(std::string(ndArray->label.ToString()), n0, n1, n2, n3, n4, n5, n6);                                                                       \
        ndArray->data   = (void*)view->data();                                                                                                                                     \
        new (instance) Teuchos::RCP<view_type>(view);                                                                                                                              \
        break;                                                                                                                                                                     \
    }

#define DEF_TEMPLATE_RANK8(TYPE_NAME, TYPE, EXECUTION_SPACE)                                                                                                                       \
    case DataTypeKind::TYPE_NAME:                                                                                                                                                  \
    {                                                                                                                                                                              \
        typedef ViewBuilder<DataTypeKind::TYPE_NAME, 8, ExecutionSpaceKind::EXECUTION_SPACE>::ViewType view_type;                                                                  \
        view_type* view = new view_type(std::string(ndArray->label.ToString()), n0, n1, n2, n3, n4, n5, n6, n7);                                                                   \
        ndArray->data   = (void*)view->data();                                                                                                                                     \
        new (instance) Teuchos::RCP<view_type>(view);                                                                                                                              \
        break;                                                                                                                                                                     \
    }

void CreateViewRank0(void* instance, NdArray* ndArray) noexcept
{
    switch (ndArray->execution_space)
    {
        case ExecutionSpaceKind::Serial:
        {
            switch (ndArray->data_type)
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
            switch (ndArray->data_type)
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
            switch (ndArray->data_type)
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
            break;        
        }
    }
}

void CreateViewRank1(void* instance, NdArray* ndArray, const size_type n0) noexcept
{
    switch (ndArray->execution_space)
    {
        case ExecutionSpaceKind::Serial:
        {
            switch (ndArray->data_type)
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
            switch (ndArray->data_type)
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
            switch (ndArray->data_type)
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
            break;        
        }
    }
}

void CreateViewRank2(void* instance, NdArray* ndArray, const size_type n0, const size_type n1) noexcept
{
    switch (ndArray->execution_space)
    {
        case ExecutionSpaceKind::Serial:
        {
            switch (ndArray->data_type)
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
            switch (ndArray->data_type)
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
            switch (ndArray->data_type)
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
            break;        
        }
    }
}

void CreateViewRank3(void* instance, NdArray* ndArray, const size_type n0, const size_type n1, const size_type n2) noexcept
{
    switch (ndArray->execution_space)
    {
        case ExecutionSpaceKind::Serial:
        {
            switch (ndArray->data_type)
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
            switch (ndArray->data_type)
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
            switch (ndArray->data_type)
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
            break;        
        }
    }
}

void CreateViewRank4(void* instance, NdArray* ndArray, const size_type n0, const size_type n1, const size_type n2, const size_type n3) noexcept
{
    switch (ndArray->execution_space)
    {
        case ExecutionSpaceKind::Serial:
        {
            switch (ndArray->data_type)
            {
                TEMPLATE(DEF_TEMPLATE_RANK4, Serial)
                default:
                {
                    std::cout << "CreateViewRank4::Serial, DataType is not supported." << std::endl;
                }
            }
            break;
        }
        case ExecutionSpaceKind::OpenMP:
        {
            switch (ndArray->data_type)
            {
                TEMPLATE(DEF_TEMPLATE_RANK4, OpenMP)
                default:
                {
                    std::cout << "CreateViewRank4::OpenMP, DataType is not supported." << std::endl;
                }
            }
            break;
        }
        case ExecutionSpaceKind::Cuda:
        {
            switch (ndArray->data_type)
            {
                TEMPLATE(DEF_TEMPLATE_RANK4, Cuda)
                default:
                {
                    std::cout << "CreateViewRank4::Cuda, DataType is not supported." << std::endl;
                }
            }
            break;
        }
        default:
        {
            std::cout << "CreateViewRank4 ExecutionSpace is not supported." << std::endl;
            break;        
        }
    }
}

void CreateViewRank5(void* instance, NdArray* ndArray, const size_type n0, const size_type n1, const size_type n2, const size_type n3, const size_type n4) noexcept
{
    switch (ndArray->execution_space)
    {
        case ExecutionSpaceKind::Serial:
        {
            switch (ndArray->data_type)
            {
                TEMPLATE(DEF_TEMPLATE_RANK5, Serial)
                default:
                {
                    std::cout << "CreateViewRank5::Serial, DataType is not supported." << std::endl;
                }
            }
            break;
        }
        case ExecutionSpaceKind::OpenMP:
        {
            switch (ndArray->data_type)
            {
                TEMPLATE(DEF_TEMPLATE_RANK5, OpenMP)
                default:
                {
                    std::cout << "CreateViewRank5::OpenMP, DataType is not supported." << std::endl;
                }
            }
            break;
        }
        case ExecutionSpaceKind::Cuda:
        {
            switch (ndArray->data_type)
            {
                TEMPLATE(DEF_TEMPLATE_RANK5, Cuda)
                default:
                {
                    std::cout << "CreateViewRank5::Cuda, DataType is not supported." << std::endl;
                }
            }
            break;
        }
        default:
        {
            std::cout << "CreateViewRank5 ExecutionSpace is not supported." << std::endl;
            break;        
        }
    }
}

void CreateViewRank6(void*           instance,
                     NdArray*        ndArray,
                     const size_type n0,
                     const size_type n1,
                     const size_type n2,
                     const size_type n3,
                     const size_type n4,
                     const size_type n5) noexcept
{
    switch (ndArray->execution_space)
    {
        case ExecutionSpaceKind::Serial:
        {
            switch (ndArray->data_type)
            {
                TEMPLATE(DEF_TEMPLATE_RANK6, Serial)
                default:
                {
                    std::cout << "CreateViewRank6::Serial, DataType is not supported." << std::endl;
                }
            }
            break;
        }
        case ExecutionSpaceKind::OpenMP:
        {
            switch (ndArray->data_type)
            {
                TEMPLATE(DEF_TEMPLATE_RANK6, OpenMP)
                default:
                {
                    std::cout << "CreateViewRank6::OpenMP, DataType is not supported." << std::endl;
                }
            }
            break;
        }
        case ExecutionSpaceKind::Cuda:
        {
            switch (ndArray->data_type)
            {
                TEMPLATE(DEF_TEMPLATE_RANK6, Cuda)
                default:
                {
                    std::cout << "CreateViewRank6::Cuda, DataType is not supported." << std::endl;
                }
            }
            break;
        }
        default:
        {
            std::cout << "CreateViewRank6 ExecutionSpace is not supported." << std::endl;
            break;        
        }
    }
}

void CreateViewRank7(void*           instance,
                     NdArray*        ndArray,
                     const size_type n0,
                     const size_type n1,
                     const size_type n2,
                     const size_type n3,
                     const size_type n4,
                     const size_type n5,
                     const size_type n6) noexcept
{
    switch (ndArray->execution_space)
    {
        case ExecutionSpaceKind::Serial:
        {
            switch (ndArray->data_type)
            {
                TEMPLATE(DEF_TEMPLATE_RANK7, Serial)
                default:
                {
                    std::cout << "CreateViewRank7::Serial, DataType is not supported." << std::endl;
                }
            }
            break;
        }
        case ExecutionSpaceKind::OpenMP:
        {
            switch (ndArray->data_type)
            {
                TEMPLATE(DEF_TEMPLATE_RANK7, OpenMP)
                default:
                {
                    std::cout << "CreateViewRank7::OpenMP, DataType is not supported." << std::endl;
                }
            }
            break;
        }
        case ExecutionSpaceKind::Cuda:
        {
            switch (ndArray->data_type)
            {
                TEMPLATE(DEF_TEMPLATE_RANK7, Cuda)
                default:
                {
                    std::cout << "CreateViewRank7::Cuda, DataType is not supported." << std::endl;
                }
            }
            break;
        }
        default:
        {
            std::cout << "CreateViewRank7 ExecutionSpace is not supported." << std::endl;
            break;        
        }
    }
}

void CreateViewRank8(void*           instance,
                     NdArray*        ndArray,
                     const size_type n0,
                     const size_type n1,
                     const size_type n2,
                     const size_type n3,
                     const size_type n4,
                     const size_type n5,
                     const size_type n6,
                     const size_type n7) noexcept
{
    switch (ndArray->execution_space)
    {
        case ExecutionSpaceKind::Serial:
        {
            switch (ndArray->data_type)
            {
                TEMPLATE(DEF_TEMPLATE_RANK8, Serial)
                default:
                {
                    std::cout << "CreateViewRank8::Serial, DataType is not supported." << std::endl;
                }
            }
            break;
        }
        case ExecutionSpaceKind::OpenMP:
        {
            switch (ndArray->data_type)
            {
                TEMPLATE(DEF_TEMPLATE_RANK8, OpenMP)
                default:
                {
                    std::cout << "CreateViewRank8::OpenMP, DataType is not supported." << std::endl;
                }
            }
            break;
        }
        case ExecutionSpaceKind::Cuda:
        {
            switch (ndArray->data_type)
            {
                TEMPLATE(DEF_TEMPLATE_RANK8, Cuda)
                default:
                {
                    std::cout << "CreateViewRank8::Cuda, DataType is not supported." << std::endl;
                }
            }
            break;
        }
        default:
        {
            std::cout << "CreateViewRank8 ExecutionSpace is not supported." << std::endl;
            break;        
        }
    }
}

void CreateView(void* instance, NdArray* ndArray) noexcept
{
    switch (ndArray->rank)
    {
        case 0:
        {
            CreateViewRank0(instance, ndArray);
            break;
        }
        case 1:
        {
            const uint64 dim0 = System::max(ndArray->dims[0], 1ull);
            CreateViewRank1(instance, ndArray, dim0);
            break;
        }
        case 2:
        {
            const uint64 dim0 = System::max(ndArray->dims[0], 1ull);
            const uint64 dim1 = System::max(ndArray->dims[1], 1ull);
            CreateViewRank2(instance, ndArray, dim0, dim1);
            break;
        }
        case 3:
        {
            const uint64 dim0 = System::max(ndArray->dims[0], 1ull);
            const uint64 dim1 = System::max(ndArray->dims[1], 1ull);
            const uint64 dim2 = System::max(ndArray->dims[2], 1ull);
            CreateViewRank3(instance, ndArray, dim0, dim1, dim2);
            break;
        }
        case 4:
        {
            const uint64 dim0 = System::max(ndArray->dims[0], 1ull);
            const uint64 dim1 = System::max(ndArray->dims[1], 1ull);
            const uint64 dim2 = System::max(ndArray->dims[2], 1ull);
            const uint64 dim3 = System::max(ndArray->dims[3], 1ull);
            CreateViewRank4(instance, ndArray, dim0, dim1, dim2, dim3);
            break;
        }
        case 5:
        {
            const uint64 dim0 = System::max(ndArray->dims[0], 1ull);
            const uint64 dim1 = System::max(ndArray->dims[1], 1ull);
            const uint64 dim2 = System::max(ndArray->dims[2], 1ull);
            const uint64 dim3 = System::max(ndArray->dims[3], 1ull);
            const uint64 dim4 = System::max(ndArray->dims[4], 1ull);
            CreateViewRank5(instance, ndArray, dim0, dim1, dim2, dim3, dim4);
            break;
        }
        case 6:
        {
            const uint64 dim0 = System::max(ndArray->dims[0], 1ull);
            const uint64 dim1 = System::max(ndArray->dims[1], 1ull);
            const uint64 dim2 = System::max(ndArray->dims[2], 1ull);
            const uint64 dim3 = System::max(ndArray->dims[3], 1ull);
            const uint64 dim4 = System::max(ndArray->dims[4], 1ull);
            const uint64 dim5 = System::max(ndArray->dims[5], 1ull);
            CreateViewRank6(instance, ndArray, dim0, dim1, dim2, dim3, dim4, dim5);
            break;
        }
        case 7:
        {
            const uint64 dim0 = System::max(ndArray->dims[0], 1ull);
            const uint64 dim1 = System::max(ndArray->dims[1], 1ull);
            const uint64 dim2 = System::max(ndArray->dims[2], 1ull);
            const uint64 dim3 = System::max(ndArray->dims[3], 1ull);
            const uint64 dim4 = System::max(ndArray->dims[4], 1ull);
            const uint64 dim5 = System::max(ndArray->dims[5], 1ull);
            const uint64 dim6 = System::max(ndArray->dims[6], 1ull);
            CreateViewRank7(instance, ndArray, dim0, dim1, dim2, dim3, dim4, dim5, dim6);
            break;
        }
        case 8:
        {
            const uint64 dim0 = System::max(ndArray->dims[0], 1ull);
            const uint64 dim1 = System::max(ndArray->dims[1], 1ull);
            const uint64 dim2 = System::max(ndArray->dims[2], 1ull);
            const uint64 dim3 = System::max(ndArray->dims[3], 1ull);
            const uint64 dim4 = System::max(ndArray->dims[4], 1ull);
            const uint64 dim5 = System::max(ndArray->dims[5], 1ull);
            const uint64 dim6 = System::max(ndArray->dims[6], 1ull);
            const uint64 dim7 = System::max(ndArray->dims[7], 1ull);
            CreateViewRank8(instance, ndArray, dim0, dim1, dim2, dim3, dim4, dim5, dim6, dim7);
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
