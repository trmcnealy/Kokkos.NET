

#include "runtime.Kokkos/KokkosVector.hpp"

#define TEMPLATE(DEF, EXECUTIONSPACE)                                                                                                                                         \
    DEF(Single, float, EXECUTIONSPACE)                                                                                                                                        \
    DEF(Double, double, EXECUTIONSPACE)                                                                                                                                       \
    DEF(Bool, bool, EXECUTIONSPACE)                                                                                                                                           \
    DEF(Int8, int8, EXECUTIONSPACE)                                                                                                                                           \
    DEF(UInt8, uint8, EXECUTIONSPACE)                                                                                                                                         \
    DEF(Int16, int16, EXECUTIONSPACE)                                                                                                                                         \
    DEF(UInt16, uint16, EXECUTIONSPACE)                                                                                                                                       \
    DEF(Int32, int32, EXECUTIONSPACE)                                                                                                                                         \
    DEF(UInt32, uint32, EXECUTIONSPACE)                                                                                                                                       \
    DEF(Int64, int64, EXECUTIONSPACE)                                                                                                                                         \
    DEF(UInt64, uint64, EXECUTIONSPACE)                                                                                                                                       \
    DEF(Char, wchar_t, EXECUTION_SPACE)

#define DEF_TEMPLATE(TYPE_NAME, TYPE, EXECUTIONSPACE) template struct Kokkos::Vector<TYPE, EXECUTIONSPACE>;

TEMPLATE(DEF_TEMPLATE, Kokkos::Serial)
TEMPLATE(DEF_TEMPLATE, Kokkos::OpenMP)
TEMPLATE(DEF_TEMPLATE, Kokkos::Cuda)

#undef TEMPLATE
#undef DEF_TEMPLATE
