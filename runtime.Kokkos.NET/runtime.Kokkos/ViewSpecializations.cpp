
//#include "runtime.Kokkos/KokkosApi.h"

//#define TEMPLATE(DEF, EXECUTION_SPACE)                                                                                                                                        \
//    DEF(Single, float, EXECUTION_SPACE)                                                                                                                                       \
//    DEF(Double, double, EXECUTION_SPACE)                                                                                                                                      \
//    DEF(Bool, bool, EXECUTION_SPACE)                                                                                                                                          \
//    DEF(Int8, int8, EXECUTION_SPACE)                                                                                                                                          \
//    DEF(UInt8, uint8, EXECUTION_SPACE)                                                                                                                                        \
//    DEF(Int16, int16, EXECUTION_SPACE)                                                                                                                                        \
//    DEF(UInt16, uint16, EXECUTION_SPACE)                                                                                                                                      \
//    DEF(Int32, int32, EXECUTION_SPACE)                                                                                                                                        \
//    DEF(UInt32, uint32, EXECUTION_SPACE)                                                                                                                                      \
//    DEF(Int64, int64, EXECUTION_SPACE)                                                                                                                                        \
//    DEF(UInt64, uint64, EXECUTION_SPACE)                                                                                                                                      \
//    DEF(Char, wchar_t, EXECUTION_SPACE)
//
//#define DEF_TEMPLATE(TYPE_NAME, TYPE, EXECUTION_SPACE)                                                                                                                        \
//    template struct __declspec(dllexport) Kokkos::View<TYPE, Kokkos::EXECUTION_SPACE::array_layout, Kokkos::EXECUTION_SPACE>;                                                 \
//    template struct __declspec(dllexport) Kokkos::View<TYPE*, Kokkos::EXECUTION_SPACE::array_layout, Kokkos::EXECUTION_SPACE>;                                                \
//    template struct __declspec(dllexport) Kokkos::View<TYPE**, Kokkos::EXECUTION_SPACE::array_layout, Kokkos::EXECUTION_SPACE>;                                               \
//    template struct __declspec(dllexport) Kokkos::View<TYPE***, Kokkos::EXECUTION_SPACE::array_layout, Kokkos::EXECUTION_SPACE>;                                              \
//    template struct __declspec(dllexport) Kokkos::View<TYPE****, Kokkos::EXECUTION_SPACE::array_layout, Kokkos::EXECUTION_SPACE>;                                             \
//    template struct __declspec(dllexport) Kokkos::View<TYPE*****, Kokkos::EXECUTION_SPACE::array_layout, Kokkos::EXECUTION_SPACE>;                                            \
//    template struct __declspec(dllexport) Kokkos::View<TYPE******, Kokkos::EXECUTION_SPACE::array_layout, Kokkos::EXECUTION_SPACE>;                                           \
//    template struct __declspec(dllexport) Kokkos::View<TYPE*******, Kokkos::EXECUTION_SPACE::array_layout, Kokkos::EXECUTION_SPACE>;                                          \
//    template struct __declspec(dllexport) Kokkos::View<TYPE********, Kokkos::EXECUTION_SPACE::array_layout, Kokkos::EXECUTION_SPACE>;                                         \
//                                                                                                                                                                              \
//    typedef Kokkos::View<TYPE, Kokkos::EXECUTION_SPACE::array_layout, Kokkos::EXECUTION_SPACE>         view_type_##TYPE_NAME##_##EXECUTION_SPACE##0;                          \
//    typedef Kokkos::View<TYPE*, Kokkos::EXECUTION_SPACE::array_layout, Kokkos::EXECUTION_SPACE>        view_type_##TYPE_NAME##_##EXECUTION_SPACE##1;                          \
//    typedef Kokkos::View<TYPE**, Kokkos::EXECUTION_SPACE::array_layout, Kokkos::EXECUTION_SPACE>       view_type_##TYPE_NAME##_##EXECUTION_SPACE##2;                          \
//    typedef Kokkos::View<TYPE***, Kokkos::EXECUTION_SPACE::array_layout, Kokkos::EXECUTION_SPACE>      view_type_##TYPE_NAME##_##EXECUTION_SPACE##3;                          \
//    typedef Kokkos::View<TYPE****, Kokkos::EXECUTION_SPACE::array_layout, Kokkos::EXECUTION_SPACE>     view_type_##TYPE_NAME##_##EXECUTION_SPACE##4;                          \
//    typedef Kokkos::View<TYPE*****, Kokkos::EXECUTION_SPACE::array_layout, Kokkos::EXECUTION_SPACE>    view_type_##TYPE_NAME##_##EXECUTION_SPACE##5;                          \
//    typedef Kokkos::View<TYPE******, Kokkos::EXECUTION_SPACE::array_layout, Kokkos::EXECUTION_SPACE>   view_type_##TYPE_NAME##_##EXECUTION_SPACE##6;                          \
//    typedef Kokkos::View<TYPE*******, Kokkos::EXECUTION_SPACE::array_layout, Kokkos::EXECUTION_SPACE>  view_type_##TYPE_NAME##_##EXECUTION_SPACE##7;                          \
//    typedef Kokkos::View<TYPE********, Kokkos::EXECUTION_SPACE::array_layout, Kokkos::EXECUTION_SPACE> view_type_##TYPE_NAME##_##EXECUTION_SPACE##8;                          \
//                                                                                                                                                                              \
//    template __declspec(dllexport) KOKKOS_FUNCTION uint64 view_type_##TYPE_NAME##_##EXECUTION_SPACE##0 ::extent<uint64>(const uint64&) const noexcept;                        \
//    template __declspec(dllexport) KOKKOS_FUNCTION uint64 view_type_##TYPE_NAME##_##EXECUTION_SPACE##0 ::stride<uint64>(uint64) const;                                        \
//                                                                                                                                                                              \
//    template __declspec(dllexport) KOKKOS_FUNCTION uint64 view_type_##TYPE_NAME##_##EXECUTION_SPACE##1 ::extent<uint64>(const uint64&) const noexcept;                        \
//    template __declspec(dllexport) KOKKOS_FUNCTION uint64 view_type_##TYPE_NAME##_##EXECUTION_SPACE##1 ::stride<uint64>(uint64) const;                                        \
//                                                                                                                                                                              \
//    template __declspec(dllexport) KOKKOS_FUNCTION uint64 view_type_##TYPE_NAME##_##EXECUTION_SPACE##2 ::extent<uint64>(const uint64&) const noexcept;                        \
//    template __declspec(dllexport) KOKKOS_FUNCTION uint64 view_type_##TYPE_NAME##_##EXECUTION_SPACE##2 ::stride<uint64>(uint64) const;                                        \
//                                                                                                                                                                              \
//    template __declspec(dllexport) KOKKOS_FUNCTION uint64 view_type_##TYPE_NAME##_##EXECUTION_SPACE##3 ::extent<uint64>(const uint64&) const noexcept;                        \
//    template __declspec(dllexport) KOKKOS_FUNCTION uint64 view_type_##TYPE_NAME##_##EXECUTION_SPACE##3 ::stride<uint64>(uint64) const;                                        \
//                                                                                                                                                                              \
//    template __declspec(dllexport) KOKKOS_FUNCTION uint64 view_type_##TYPE_NAME##_##EXECUTION_SPACE##4 ::extent<uint64>(const uint64&) const noexcept;                        \
//    template __declspec(dllexport) KOKKOS_FUNCTION uint64 view_type_##TYPE_NAME##_##EXECUTION_SPACE##4 ::stride<uint64>(uint64) const;                                        \
//                                                                                                                                                                              \
//    template __declspec(dllexport) KOKKOS_FUNCTION uint64 view_type_##TYPE_NAME##_##EXECUTION_SPACE##5 ::extent<uint64>(const uint64&) const noexcept;                        \
//    template __declspec(dllexport) KOKKOS_FUNCTION uint64 view_type_##TYPE_NAME##_##EXECUTION_SPACE##5 ::stride<uint64>(uint64) const;                                        \
//                                                                                                                                                                              \
//    template __declspec(dllexport) KOKKOS_FUNCTION uint64 view_type_##TYPE_NAME##_##EXECUTION_SPACE##6 ::extent<uint64>(const uint64&) const noexcept;                        \
//    template __declspec(dllexport) KOKKOS_FUNCTION uint64 view_type_##TYPE_NAME##_##EXECUTION_SPACE##6 ::stride<uint64>(uint64) const;                                        \
//                                                                                                                                                                              \
//    template __declspec(dllexport) KOKKOS_FUNCTION uint64 view_type_##TYPE_NAME##_##EXECUTION_SPACE##7 ::extent<uint64>(const uint64&) const noexcept;                        \
//    template __declspec(dllexport) KOKKOS_FUNCTION uint64 view_type_##TYPE_NAME##_##EXECUTION_SPACE##7 ::stride<uint64>(uint64) const;                                        \
//                                                                                                                                                                              \
//    template __declspec(dllexport) KOKKOS_FUNCTION uint64 view_type_##TYPE_NAME##_##EXECUTION_SPACE##8 ::extent<uint64>(const uint64&) const noexcept;                        \
//    template __declspec(dllexport) KOKKOS_FUNCTION uint64 view_type_##TYPE_NAME##_##EXECUTION_SPACE##8 ::stride<uint64>(uint64) const;                                        \
//                                                                                                                                                                              \
//    template __declspec(dllexport) KOKKOS_FUNCTION TYPE& view_type_##TYPE_NAME##_##EXECUTION_SPACE##1 ::operator()<uint64>(const uint64& i0) const;                           \
//    template __declspec(dllexport) KOKKOS_FUNCTION TYPE& view_type_##TYPE_NAME##_##EXECUTION_SPACE##2 ::operator()<uint64, uint64>(const uint64& i0, const uint64& i1) const; \
//    template __declspec(dllexport)                                                                                                                                            \
//        KOKKOS_FUNCTION TYPE& view_type_##TYPE_NAME##_##EXECUTION_SPACE##3 ::operator()<uint64, uint64, uint64>(const uint64& i0, const uint64& i1, const uint64& i2) const;  \
//    template __declspec(dllexport) KOKKOS_FUNCTION TYPE& view_type_##TYPE_NAME##_##EXECUTION_SPACE##4 ::operator()<uint64, uint64, uint64, uint64>(const uint64& i0,          \
//                                                                                                                                                   const uint64& i1,          \
//                                                                                                                                                   const uint64& i2,          \
//                                                                                                                                                   const uint64& i3) const;   \
//    template __declspec(dllexport) KOKKOS_FUNCTION TYPE& view_type_##TYPE_NAME##_##EXECUTION_SPACE##5 ::operator()<uint64, uint64, uint64, uint64, uint64>(const uint64& i0,  \
//                                                                                                                                                           const uint64& i1,  \
//                                                                                                                                                           const uint64& i2,  \
//                                                                                                                                                           const uint64& i3,  \
//                                                                                                                                                           const uint64& i4)  \
//        const;                                                                                                                                                                \
//    template __declspec(dllexport)                                                                                                                                            \
//        KOKKOS_FUNCTION TYPE& view_type_##TYPE_NAME##_##EXECUTION_SPACE##6 ::operator()<uint64, uint64, uint64, uint64, uint64, uint64>(const uint64& i0,                     \
//                                                                                                                                        const uint64& i1,                     \
//                                                                                                                                        const uint64& i2,                     \
//                                                                                                                                        const uint64& i3,                     \
//                                                                                                                                        const uint64& i4,                     \
//                                                                                                                                        const uint64& i5) const;              \
//    template __declspec(dllexport)                                                                                                                                            \
//        KOKKOS_FUNCTION TYPE& view_type_##TYPE_NAME##_##EXECUTION_SPACE##7 ::operator()<uint64, uint64, uint64, uint64, uint64, uint64, uint64>(const uint64& i0,             \
//                                                                                                                                                const uint64& i1,             \
//                                                                                                                                                const uint64& i2,             \
//                                                                                                                                                const uint64& i3,             \
//                                                                                                                                                const uint64& i4,             \
//                                                                                                                                                const uint64& i5,             \
//                                                                                                                                                const uint64& i6) const;      \
//    template __declspec(dllexport)                                                                                                                                            \
//        KOKKOS_FUNCTION TYPE& view_type_##TYPE_NAME##_##EXECUTION_SPACE##8 ::operator()<uint64, uint64, uint64, uint64, uint64, uint64, uint64, uint64>(const uint64& i0,     \
//                                                                                                                                                        const uint64& i1,     \
//                                                                                                                                                        const uint64& i2,     \
//                                                                                                                                                        const uint64& i3,     \
//                                                                                                                                                        const uint64& i4,     \
//                                                                                                                                                        const uint64& i5,     \
//                                                                                                                                                        const uint64& i6,     \
//                                                                                                                                                        const uint64& i7)     \
//            const;                                                                                                                                                            \
//    template __declspec(dllexport) KOKKOS_FUNCTION TYPE& view_type_##TYPE_NAME##_##EXECUTION_SPACE##1 ::access<uint64>(const uint64& i0) const;                               \
//    template __declspec(dllexport) KOKKOS_FUNCTION TYPE& view_type_##TYPE_NAME##_##EXECUTION_SPACE##2 ::access<uint64, uint64>(const uint64& i0, const uint64& i1) const;     \
//    template __declspec(dllexport)                                                                                                                                            \
//        KOKKOS_FUNCTION TYPE& view_type_##TYPE_NAME##_##EXECUTION_SPACE##3 ::access<uint64, uint64, uint64>(const uint64& i0, const uint64& i1, const uint64& i2) const;      \
//    template __declspec(dllexport) KOKKOS_FUNCTION TYPE& view_type_##TYPE_NAME##_##EXECUTION_SPACE##4 ::access<uint64, uint64, uint64, uint64>(const uint64& i0,              \
//                                                                                                                                               const uint64& i1,              \
//                                                                                                                                               const uint64& i2,              \
//                                                                                                                                               const uint64& i3) const;       \
//    template __declspec(dllexport) KOKKOS_FUNCTION TYPE& view_type_##TYPE_NAME##_##EXECUTION_SPACE##5 ::access<uint64, uint64, uint64, uint64, uint64>(const uint64& i0,      \
//                                                                                                                                                       const uint64& i1,      \
//                                                                                                                                                       const uint64& i2,      \
//                                                                                                                                                       const uint64& i3,      \
//                                                                                                                                                       const uint64& i4)      \
//        const;                                                                                                                                                                \
//    template __declspec(dllexport)                                                                                                                                            \
//        KOKKOS_FUNCTION TYPE& view_type_##TYPE_NAME##_##EXECUTION_SPACE##6 ::access<uint64, uint64, uint64, uint64, uint64, uint64>(const uint64& i0,                         \
//                                                                                                                                    const uint64& i1,                         \
//                                                                                                                                    const uint64& i2,                         \
//                                                                                                                                    const uint64& i3,                         \
//                                                                                                                                    const uint64& i4,                         \
//                                                                                                                                    const uint64& i5) const;                  \
//    template __declspec(dllexport)                                                                                                                                            \
//        KOKKOS_FUNCTION TYPE& view_type_##TYPE_NAME##_##EXECUTION_SPACE##7 ::access<uint64, uint64, uint64, uint64, uint64, uint64, uint64>(const uint64& i0,                 \
//                                                                                                                                            const uint64& i1,                 \
//                                                                                                                                            const uint64& i2,                 \
//                                                                                                                                            const uint64& i3,                 \
//                                                                                                                                            const uint64& i4,                 \
//                                                                                                                                            const uint64& i5,                 \
//                                                                                                                                            const uint64& i6) const;          \
//    template __declspec(dllexport)                                                                                                                                            \
//        KOKKOS_FUNCTION TYPE& view_type_##TYPE_NAME##_##EXECUTION_SPACE##8 ::access<uint64, uint64, uint64, uint64, uint64, uint64, uint64, uint64>(const uint64& i0,         \
//                                                                                                                                                    const uint64& i1,         \
//                                                                                                                                                    const uint64& i2,         \
//                                                                                                                                                    const uint64& i3,         \
//                                                                                                                                                    const uint64& i4,         \
//                                                                                                                                                    const uint64& i5,         \
//                                                                                                                                                    const uint64& i6,         \
//                                                                                                                                                    const uint64& i7) const;
//
//TEMPLATE(DEF_TEMPLATE, Serial)
//TEMPLATE(DEF_TEMPLATE, OpenMP)
//TEMPLATE(DEF_TEMPLATE, Cuda)
//
//#undef TEMPLATE
//#undef DEF_TEMPLATE
