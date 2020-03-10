
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

#define TEMPLATE_RANK4(DEF, EXECUTION_SPACE) \
    DEF(Single, float****, EXECUTION_SPACE)  \
    DEF(Double, double****, EXECUTION_SPACE) \
    DEF(Bool, bool****, EXECUTION_SPACE)     \
    DEF(Int8, int8****, EXECUTION_SPACE)     \
    DEF(UInt8, uint8****, EXECUTION_SPACE)   \
    DEF(Int16, int16****, EXECUTION_SPACE)   \
    DEF(UInt16, uint16****, EXECUTION_SPACE) \
    DEF(Int32, int32****, EXECUTION_SPACE)   \
    DEF(UInt32, uint32****, EXECUTION_SPACE) \
    DEF(Int64, int64****, EXECUTION_SPACE)   \
    DEF(UInt64, uint64****, EXECUTION_SPACE)

#define TEMPLATE_RANK5(DEF, EXECUTION_SPACE)  \
    DEF(Single, float*****, EXECUTION_SPACE)  \
    DEF(Double, double*****, EXECUTION_SPACE) \
    DEF(Bool, bool*****, EXECUTION_SPACE)     \
    DEF(Int8, int8*****, EXECUTION_SPACE)     \
    DEF(UInt8, uint8*****, EXECUTION_SPACE)   \
    DEF(Int16, int16*****, EXECUTION_SPACE)   \
    DEF(UInt16, uint16*****, EXECUTION_SPACE) \
    DEF(Int32, int32*****, EXECUTION_SPACE)   \
    DEF(UInt32, uint32*****, EXECUTION_SPACE) \
    DEF(Int64, int64*****, EXECUTION_SPACE)   \
    DEF(UInt64, uint64*****, EXECUTION_SPACE)

#define TEMPLATE_RANK6(DEF, EXECUTION_SPACE)   \
    DEF(Single, float******, EXECUTION_SPACE)  \
    DEF(Double, double******, EXECUTION_SPACE) \
    DEF(Bool, bool******, EXECUTION_SPACE)     \
    DEF(Int8, int8******, EXECUTION_SPACE)     \
    DEF(UInt8, uint8******, EXECUTION_SPACE)   \
    DEF(Int16, int16******, EXECUTION_SPACE)   \
    DEF(UInt16, uint16******, EXECUTION_SPACE) \
    DEF(Int32, int32******, EXECUTION_SPACE)   \
    DEF(UInt32, uint32******, EXECUTION_SPACE) \
    DEF(Int64, int64******, EXECUTION_SPACE)   \
    DEF(UInt64, uint64******, EXECUTION_SPACE)

#define TEMPLATE_RANK7(DEF, EXECUTION_SPACE)    \
    DEF(Single, float*******, EXECUTION_SPACE)  \
    DEF(Double, double*******, EXECUTION_SPACE) \
    DEF(Bool, bool*******, EXECUTION_SPACE)     \
    DEF(Int8, int8*******, EXECUTION_SPACE)     \
    DEF(UInt8, uint8*******, EXECUTION_SPACE)   \
    DEF(Int16, int16*******, EXECUTION_SPACE)   \
    DEF(UInt16, uint16*******, EXECUTION_SPACE) \
    DEF(Int32, int32*******, EXECUTION_SPACE)   \
    DEF(UInt32, uint32*******, EXECUTION_SPACE) \
    DEF(Int64, int64*******, EXECUTION_SPACE)   \
    DEF(UInt64, uint64*******, EXECUTION_SPACE)

#define TEMPLATE_RANK8(DEF, EXECUTION_SPACE)     \
    DEF(Single, float********, EXECUTION_SPACE)  \
    DEF(Double, double********, EXECUTION_SPACE) \
    DEF(Bool, bool********, EXECUTION_SPACE)     \
    DEF(Int8, int8********, EXECUTION_SPACE)     \
    DEF(UInt8, uint8********, EXECUTION_SPACE)   \
    DEF(Int16, int16********, EXECUTION_SPACE)   \
    DEF(UInt16, uint16********, EXECUTION_SPACE) \
    DEF(Int32, int32********, EXECUTION_SPACE)   \
    DEF(UInt32, uint32********, EXECUTION_SPACE) \
    DEF(Int64, int64********, EXECUTION_SPACE)   \
    DEF(UInt64, uint64********, EXECUTION_SPACE)

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
                case 4:
                {
                    switch(ndArray.data_type)
                    {
                        TEMPLATE_RANK4(DEF_TEMPLATE, Serial)
                        default:
                        {
                            std::cout << "GetLabel::Serial, DataType is not supported." << std::endl;
                        }
                    }
                    break;
                }
                case 5:
                {
                    switch(ndArray.data_type)
                    {
                        TEMPLATE_RANK5(DEF_TEMPLATE, Serial)
                        default:
                        {
                            std::cout << "GetLabel::Serial, DataType is not supported." << std::endl;
                        }
                    }
                    break;
                }
                case 6:
                {
                    switch(ndArray.data_type)
                    {
                        TEMPLATE_RANK6(DEF_TEMPLATE, Serial)
                        default:
                        {
                            std::cout << "GetLabel::Serial, DataType is not supported." << std::endl;
                        }
                    }
                    break;
                }
                case 7:
                {
                    switch(ndArray.data_type)
                    {
                        TEMPLATE_RANK7(DEF_TEMPLATE, Serial)
                        default:
                        {
                            std::cout << "GetLabel::Serial, DataType is not supported." << std::endl;
                        }
                    }
                    break;
                }
                case 8:
                {
                    switch(ndArray.data_type)
                    {
                        TEMPLATE_RANK8(DEF_TEMPLATE, Serial)
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
                case 4:
                {
                    switch(ndArray.data_type)
                    {
                        TEMPLATE_RANK4(DEF_TEMPLATE, OpenMP)
                        default:
                        {
                            std::cout << "GetLabel::OpenMP, DataType is not supported." << std::endl;
                        }
                    }
                    break;
                }
                case 5:
                {
                    switch(ndArray.data_type)
                    {
                        TEMPLATE_RANK5(DEF_TEMPLATE, OpenMP)
                        default:
                        {
                            std::cout << "GetLabel::OpenMP, DataType is not supported." << std::endl;
                        }
                    }
                    break;
                }
                case 6:
                {
                    switch(ndArray.data_type)
                    {
                        TEMPLATE_RANK6(DEF_TEMPLATE, OpenMP)
                        default:
                        {
                            std::cout << "GetLabel::OpenMP, DataType is not supported." << std::endl;
                        }
                    }
                    break;
                }
                case 7:
                {
                    switch(ndArray.data_type)
                    {
                        TEMPLATE_RANK7(DEF_TEMPLATE, OpenMP)
                        default:
                        {
                            std::cout << "GetLabel::OpenMP, DataType is not supported." << std::endl;
                        }
                    }
                    break;
                }
                case 8:
                {
                    switch(ndArray.data_type)
                    {
                        TEMPLATE_RANK8(DEF_TEMPLATE, OpenMP)
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
                case 4:
                {
                    switch(ndArray.data_type)
                    {
                        TEMPLATE_RANK4(DEF_TEMPLATE, Cuda)
                        default:
                        {
                            std::cout << "GetLabel::Cuda, DataType is not supported." << std::endl;
                        }
                    }
                    break;
                }
                case 5:
                {
                    switch(ndArray.data_type)
                    {
                        TEMPLATE_RANK5(DEF_TEMPLATE, Cuda)
                        default:
                        {
                            std::cout << "GetLabel::Cuda, DataType is not supported." << std::endl;
                        }
                    }
                    break;
                }
                case 6:
                {
                    switch(ndArray.data_type)
                    {
                        TEMPLATE_RANK6(DEF_TEMPLATE, Cuda)
                        default:
                        {
                            std::cout << "GetLabel::Cuda, DataType is not supported." << std::endl;
                        }
                    }
                    break;
                }
                case 7:
                {
                    switch(ndArray.data_type)
                    {
                        TEMPLATE_RANK7(DEF_TEMPLATE, Cuda)
                        default:
                        {
                            std::cout << "GetLabel::Cuda, DataType is not supported." << std::endl;
                        }
                    }
                    break;
                }
                case 8:
                {
                    switch(ndArray.data_type)
                    {
                        TEMPLATE_RANK8(DEF_TEMPLATE, Cuda)
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
                case 4:
                {
                    switch(ndArray.data_type)
                    {
                        TEMPLATE_RANK4(DEF_TEMPLATE, Serial)
                        default:
                        {
                            std::cout << "GetSize::Serial, DataType is not supported." << std::endl;
                        }
                    }
                    break;
                }
                case 5:
                {
                    switch(ndArray.data_type)
                    {
                        TEMPLATE_RANK5(DEF_TEMPLATE, Serial)
                        default:
                        {
                            std::cout << "GetSize::Serial, DataType is not supported." << std::endl;
                        }
                    }
                    break;
                }
                case 6:
                {
                    switch(ndArray.data_type)
                    {
                        TEMPLATE_RANK6(DEF_TEMPLATE, Serial)
                        default:
                        {
                            std::cout << "GetSize::Serial, DataType is not supported." << std::endl;
                        }
                    }
                    break;
                }
                case 7:
                {
                    switch(ndArray.data_type)
                    {
                        TEMPLATE_RANK7(DEF_TEMPLATE, Serial)
                        default:
                        {
                            std::cout << "GetSize::Serial, DataType is not supported." << std::endl;
                        }
                    }
                    break;
                }
                case 8:
                {
                    switch(ndArray.data_type)
                    {
                        TEMPLATE_RANK8(DEF_TEMPLATE, Serial)
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
                case 4:
                {
                    switch(ndArray.data_type)
                    {
                        TEMPLATE_RANK4(DEF_TEMPLATE, OpenMP)
                        default:
                        {
                            std::cout << "GetSize::OpenMP, DataType is not supported." << std::endl;
                        }
                    }
                    break;
                }
                case 5:
                {
                    switch(ndArray.data_type)
                    {
                        TEMPLATE_RANK5(DEF_TEMPLATE, OpenMP)
                        default:
                        {
                            std::cout << "GetSize::OpenMP, DataType is not supported." << std::endl;
                        }
                    }
                    break;
                }
                case 6:
                {
                    switch(ndArray.data_type)
                    {
                        TEMPLATE_RANK6(DEF_TEMPLATE, OpenMP)
                        default:
                        {
                            std::cout << "GetSize::OpenMP, DataType is not supported." << std::endl;
                        }
                    }
                    break;
                }
                case 7:
                {
                    switch(ndArray.data_type)
                    {
                        TEMPLATE_RANK7(DEF_TEMPLATE, OpenMP)
                        default:
                        {
                            std::cout << "GetSize::OpenMP, DataType is not supported." << std::endl;
                        }
                    }
                    break;
                }
                case 8:
                {
                    switch(ndArray.data_type)
                    {
                        TEMPLATE_RANK8(DEF_TEMPLATE, OpenMP)
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
                case 4:
                {
                    switch(ndArray.data_type)
                    {
                        TEMPLATE_RANK4(DEF_TEMPLATE, Cuda)
                        default:
                        {
                            std::cout << "GetSize::Cuda, DataType is not supported." << std::endl;
                        }
                    }
                    break;
                }
                case 5:
                {
                    switch(ndArray.data_type)
                    {
                        TEMPLATE_RANK5(DEF_TEMPLATE, Cuda)
                        default:
                        {
                            std::cout << "GetSize::Cuda, DataType is not supported." << std::endl;
                        }
                    }
                    break;
                }
                case 6:
                {
                    switch(ndArray.data_type)
                    {
                        TEMPLATE_RANK6(DEF_TEMPLATE, Cuda)
                        default:
                        {
                            std::cout << "GetSize::Cuda, DataType is not supported." << std::endl;
                        }
                    }
                    break;
                }
                case 7:
                {
                    switch(ndArray.data_type)
                    {
                        TEMPLATE_RANK7(DEF_TEMPLATE, Cuda)
                        default:
                        {
                            std::cout << "GetSize::Cuda, DataType is not supported." << std::endl;
                        }
                    }
                    break;
                }
                case 8:
                {
                    switch(ndArray.data_type)
                    {
                        TEMPLATE_RANK8(DEF_TEMPLATE, Cuda)
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
                case 4:
                {
                    switch(ndArray.data_type)
                    {
                        TEMPLATE_RANK4(DEF_TEMPLATE, Serial)
                        default:
                        {
                            std::cout << "GetStride::Serial, DataType is not supported." << std::endl;
                        }
                    }
                    break;
                }
                case 5:
                {
                    switch(ndArray.data_type)
                    {
                        TEMPLATE_RANK5(DEF_TEMPLATE, Serial)
                        default:
                        {
                            std::cout << "GetStride::Serial, DataType is not supported." << std::endl;
                        }
                    }
                    break;
                }
                case 6:
                {
                    switch(ndArray.data_type)
                    {
                        TEMPLATE_RANK6(DEF_TEMPLATE, Serial)
                        default:
                        {
                            std::cout << "GetStride::Serial, DataType is not supported." << std::endl;
                        }
                    }
                    break;
                }
                case 7:
                {
                    switch(ndArray.data_type)
                    {
                        TEMPLATE_RANK7(DEF_TEMPLATE, Serial)
                        default:
                        {
                            std::cout << "GetStride::Serial, DataType is not supported." << std::endl;
                        }
                    }
                    break;
                }
                case 8:
                {
                    switch(ndArray.data_type)
                    {
                        TEMPLATE_RANK8(DEF_TEMPLATE, Serial)
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
                case 4:
                {
                    switch(ndArray.data_type)
                    {
                        TEMPLATE_RANK4(DEF_TEMPLATE, OpenMP)
                        default:
                        {
                            std::cout << "GetStride::OpenMP, DataType is not supported." << std::endl;
                        }
                    }
                    break;
                }
                case 5:
                {
                    switch(ndArray.data_type)
                    {
                        TEMPLATE_RANK5(DEF_TEMPLATE, OpenMP)
                        default:
                        {
                            std::cout << "GetStride::OpenMP, DataType is not supported." << std::endl;
                        }
                    }
                    break;
                }
                case 6:
                {
                    switch(ndArray.data_type)
                    {
                        TEMPLATE_RANK6(DEF_TEMPLATE, OpenMP)
                        default:
                        {
                            std::cout << "GetStride::OpenMP, DataType is not supported." << std::endl;
                        }
                    }
                    break;
                }
                case 7:
                {
                    switch(ndArray.data_type)
                    {
                        TEMPLATE_RANK7(DEF_TEMPLATE, OpenMP)
                        default:
                        {
                            std::cout << "GetStride::OpenMP, DataType is not supported." << std::endl;
                        }
                    }
                    break;
                }
                case 8:
                {
                    switch(ndArray.data_type)
                    {
                        TEMPLATE_RANK8(DEF_TEMPLATE, OpenMP)
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
                case 4:
                {
                    switch(ndArray.data_type)
                    {
                        TEMPLATE_RANK4(DEF_TEMPLATE, Cuda)
                        default:
                        {
                            std::cout << "GetStride::Cuda, DataType is not supported." << std::endl;
                        }
                    }
                    break;
                }
                case 5:
                {
                    switch(ndArray.data_type)
                    {
                        TEMPLATE_RANK5(DEF_TEMPLATE, Cuda)
                        default:
                        {
                            std::cout << "GetStride::Cuda, DataType is not supported." << std::endl;
                        }
                    }
                    break;
                }
                case 6:
                {
                    switch(ndArray.data_type)
                    {
                        TEMPLATE_RANK6(DEF_TEMPLATE, Cuda)
                        default:
                        {
                            std::cout << "GetStride::Cuda, DataType is not supported." << std::endl;
                        }
                    }
                    break;
                }
                case 7:
                {
                    switch(ndArray.data_type)
                    {
                        TEMPLATE_RANK7(DEF_TEMPLATE, Cuda)
                        default:
                        {
                            std::cout << "GetStride::Cuda, DataType is not supported." << std::endl;
                        }
                    }
                    break;
                }
                case 8:
                {
                    switch(ndArray.data_type)
                    {
                        TEMPLATE_RANK8(DEF_TEMPLATE, Cuda)
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
                case 4:
                {
                    switch(ndArray.data_type)
                    {
                        TEMPLATE_RANK4(DEF_TEMPLATE, Serial)
                        default:
                        {
                            std::cout << "GetExtent::Serial, DataType is not supported." << std::endl;
                        }
                    }
                    break;
                }
                case 5:
                {
                    switch(ndArray.data_type)
                    {
                        TEMPLATE_RANK5(DEF_TEMPLATE, Serial)
                        default:
                        {
                            std::cout << "GetExtent::Serial, DataType is not supported." << std::endl;
                        }
                    }
                    break;
                }
                case 6:
                {
                    switch(ndArray.data_type)
                    {
                        TEMPLATE_RANK6(DEF_TEMPLATE, Serial)
                        default:
                        {
                            std::cout << "GetExtent::Serial, DataType is not supported." << std::endl;
                        }
                    }
                    break;
                }
                case 7:
                {
                    switch(ndArray.data_type)
                    {
                        TEMPLATE_RANK7(DEF_TEMPLATE, Serial)
                        default:
                        {
                            std::cout << "GetExtent::Serial, DataType is not supported." << std::endl;
                        }
                    }
                    break;
                }
                case 8:
                {
                    switch(ndArray.data_type)
                    {
                        TEMPLATE_RANK8(DEF_TEMPLATE, Serial)
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
                case 4:
                {
                    switch(ndArray.data_type)
                    {
                        TEMPLATE_RANK4(DEF_TEMPLATE, OpenMP)
                        default:
                        {
                            std::cout << "GetExtent::OpenMP, DataType is not supported." << std::endl;
                        }
                    }
                    break;
                }
                case 5:
                {
                    switch(ndArray.data_type)
                    {
                        TEMPLATE_RANK5(DEF_TEMPLATE, OpenMP)
                        default:
                        {
                            std::cout << "GetExtent::OpenMP, DataType is not supported." << std::endl;
                        }
                    }
                    break;
                }
                case 6:
                {
                    switch(ndArray.data_type)
                    {
                        TEMPLATE_RANK6(DEF_TEMPLATE, OpenMP)
                        default:
                        {
                            std::cout << "GetExtent::OpenMP, DataType is not supported." << std::endl;
                        }
                    }
                    break;
                }
                case 7:
                {
                    switch(ndArray.data_type)
                    {
                        TEMPLATE_RANK7(DEF_TEMPLATE, OpenMP)
                        default:
                        {
                            std::cout << "GetExtent::OpenMP, DataType is not supported." << std::endl;
                        }
                    }
                    break;
                }
                case 8:
                {
                    switch(ndArray.data_type)
                    {
                        TEMPLATE_RANK8(DEF_TEMPLATE, OpenMP)
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
                case 4:
                {
                    switch(ndArray.data_type)
                    {
                        TEMPLATE_RANK4(DEF_TEMPLATE, Cuda)
                        default:
                        {
                            std::cout << "GetExtent::Cuda, DataType is not supported." << std::endl;
                        }
                    }
                    break;
                }
                case 5:
                {
                    switch(ndArray.data_type)
                    {
                        TEMPLATE_RANK5(DEF_TEMPLATE, Cuda)
                        default:
                        {
                            std::cout << "GetExtent::Cuda, DataType is not supported." << std::endl;
                        }
                    }
                    break;
                }
                case 6:
                {
                    switch(ndArray.data_type)
                    {
                        TEMPLATE_RANK6(DEF_TEMPLATE, Cuda)
                        default:
                        {
                            std::cout << "GetExtent::Cuda, DataType is not supported." << std::endl;
                        }
                    }
                    break;
                }
                case 7:
                {
                    switch(ndArray.data_type)
                    {
                        TEMPLATE_RANK7(DEF_TEMPLATE, Cuda)
                        default:
                        {
                            std::cout << "GetExtent::Cuda, DataType is not supported." << std::endl;
                        }
                    }
                    break;
                }
                case 8:
                {
                    switch(ndArray.data_type)
                    {
                        TEMPLATE_RANK8(DEF_TEMPLATE, Cuda)
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
#undef TEMPLATE_RANK4
#undef TEMPLATE_RANK5
#undef TEMPLATE_RANK6
#undef TEMPLATE_RANK7
#undef TEMPLATE_RANK8
