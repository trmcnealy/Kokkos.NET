
#include "runtime.Kokkos/KokkosApi.h"

#define DEF_TEMPLATE_RANK0(TYPE_NAME, TYPE, EXECUTION_SPACE)                                                       \
    case DataTypeKind::TYPE_NAME:                                                                                  \
    {                                                                                                              \
        typedef TYPE                                                                                   value_type; \
        typedef ViewBuilder<DataTypeKind::TYPE_NAME, 0, ExecutionSpaceKind::EXECUTION_SPACE>::ViewType view_type;  \
                                                                                                                   \
        view_type& view      = *(*reinterpret_cast<Teuchos::RCP<view_type>*>(instance));                           \
        ValueType  valuetype = view();                                                                             \
        return valuetype;                                                                                          \
    }

#define DEF_TEMPLATE_RANK1(TYPE_NAME, TYPE, EXECUTION_SPACE)                                                       \
    case DataTypeKind::TYPE_NAME:                                                                                  \
    {                                                                                                              \
        typedef TYPE                                                                                   value_type; \
        typedef ViewBuilder<DataTypeKind::TYPE_NAME, 1, ExecutionSpaceKind::EXECUTION_SPACE>::ViewType view_type;  \
                                                                                                                   \
        view_type& view      = *(*reinterpret_cast<Teuchos::RCP<view_type>*>(instance));                           \
        ValueType  valuetype = view(i0);                                                                           \
        return valuetype;                                                                                          \
    }

#define DEF_TEMPLATE_RANK2(TYPE_NAME, TYPE, EXECUTION_SPACE)                                                       \
    case DataTypeKind::TYPE_NAME:                                                                                  \
    {                                                                                                              \
        typedef TYPE                                                                                   value_type; \
        typedef ViewBuilder<DataTypeKind::TYPE_NAME, 2, ExecutionSpaceKind::EXECUTION_SPACE>::ViewType view_type;  \
                                                                                                                   \
        view_type& view      = *(*reinterpret_cast<Teuchos::RCP<view_type>*>(instance));                           \
        ValueType  valuetype = view(i0, i1);                                                                       \
        return valuetype;                                                                                          \
    }

#define DEF_TEMPLATE_RANK3(TYPE_NAME, TYPE, EXECUTION_SPACE)                                                       \
    case DataTypeKind::TYPE_NAME:                                                                                  \
    {                                                                                                              \
        typedef TYPE                                                                                   value_type; \
        typedef ViewBuilder<DataTypeKind::TYPE_NAME, 3, ExecutionSpaceKind::EXECUTION_SPACE>::ViewType view_type;  \
                                                                                                                   \
        view_type& view      = *(*reinterpret_cast<Teuchos::RCP<view_type>*>(instance));                           \
        ValueType  valuetype = view(i0, i1, i2);                                                                   \
        return valuetype;                                                                                          \
    }

#define TEMPLATE(DEF, EXECUTION_SPACE)              \
    DEF(Single, float, EXECUTION_SPACE)             \
    DEF(Double, double, EXECUTION_SPACE)            \
    DEF(Bool, bool, EXECUTION_SPACE)                \
    DEF(Int8, int8, EXECUTION_SPACE)                \
    DEF(UInt8, uint8, EXECUTION_SPACE)              \
    DEF(Int16, int16, EXECUTION_SPACE)              \
    DEF(UInt16, uint16, EXECUTION_SPACE)            \
    DEF(Int32, int32, EXECUTION_SPACE)              \
    DEF(UInt32, uint32, EXECUTION_SPACE)            \
    DEF(Int64, int64, EXECUTION_SPACE)              \
    DEF(UInt64, uint64, EXECUTION_SPACE)            \
    DEF(ConstSingle, const float, EXECUTION_SPACE)  \
    DEF(ConstDouble, const double, EXECUTION_SPACE) \
    DEF(ConstBool, const bool, EXECUTION_SPACE)     \
    DEF(ConstInt8, const int8, EXECUTION_SPACE)     \
    DEF(ConstUInt8, const uint8, EXECUTION_SPACE)   \
    DEF(ConstInt16, const int16, EXECUTION_SPACE)   \
    DEF(ConstUInt16, const uint16, EXECUTION_SPACE) \
    DEF(ConstInt32, const int32, EXECUTION_SPACE)   \
    DEF(ConstUInt32, const uint32, EXECUTION_SPACE) \
    DEF(ConstInt64, const int64, EXECUTION_SPACE)   \
    DEF(ConstUInt64, const uint64, EXECUTION_SPACE)

ValueType GetValue(void* instance, const NdArray& ndArray, const size_type& i0, const size_type& i1, const size_type& i2) noexcept
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
                            std::cout << "GetValue::Serial, DataType is not supported." << std::endl;
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
                            std::cout << "GetValue::Serial, DataType is not supported." << std::endl;
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
                            std::cout << "GetValue::Serial, DataType is not supported." << std::endl;
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
                            std::cout << "GetValue::Serial, DataType is not supported." << std::endl;
                        }
                    }
                    break;
                }
                default:
                {
                    std::cout << "GetValue::Serial, Rank is not supported." << std::endl;
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
                            std::cout << "GetValue::OpenMP, DataType is not supported." << std::endl;
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
                            std::cout << "GetValue::OpenMP, DataType is not supported." << std::endl;
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
                            std::cout << "GetValue::OpenMP, DataType is not supported." << std::endl;
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
                            std::cout << "GetValue::OpenMP, DataType is not supported." << std::endl;
                        }
                    }
                    break;
                }
                default:
                {
                    std::cout << "GetValue::OpenMP, Rank is not supported." << std::endl;
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
                            std::cout << "GetValue::Cuda, DataType is not supported." << std::endl;
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
                            std::cout << "GetValue::Cuda, DataType is not supported." << std::endl;
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
                            std::cout << "GetValue::Cuda, DataType is not supported." << std::endl;
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
                            std::cout << "GetValue::Cuda, DataType is not supported." << std::endl;
                        }
                    }
                    break;
                }
                default:
                {
                    std::cout << "GetValue::Cuda, Rank is not supported." << std::endl;
                }
            }
            break;
        }
        default:
        {
            std::cout << "GetValue ExecutionSpace is not supported." << std::endl;
        }
    }
    return {0};
}

#undef DEF_TEMPLATE_RANK0
#undef DEF_TEMPLATE_RANK1
#undef DEF_TEMPLATE_RANK2
#undef DEF_TEMPLATE_RANK3
