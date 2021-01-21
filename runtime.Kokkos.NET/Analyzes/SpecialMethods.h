#pragma once

#include <runtime.Kokkos/ViewTypes.hpp>
#include <runtime.Kokkos/Extensions.hpp>

template<class ExecutionSpace>
using KokkosViewFloat = Kokkos::View<float*, typename ExecutionSpace::array_layout, ExecutionSpace>;

template<class ExecutionSpace>
using KokkosViewDouble = Kokkos::View<double*, typename ExecutionSpace::array_layout, ExecutionSpace>;

KOKKOS_NET_API_EXPORT KOKKOS_FUNCTION float BesselI(const int nOrder, const float arg);

KOKKOS_NET_API_EXPORT KOKKOS_FUNCTION float BesselI0(const float arg);

KOKKOS_NET_API_EXPORT KOKKOS_FUNCTION float BesselI1(const float arg);

KOKKOS_NET_API_EXPORT KOKKOS_FUNCTION float BesselIN(const int nOrder, const float arg);

KOKKOS_NET_API_EXPORT KOKKOS_FUNCTION double BesselI(const int nOrder, const double arg);

KOKKOS_NET_API_EXPORT KOKKOS_FUNCTION double BesselI0(const double arg);

KOKKOS_NET_API_EXPORT KOKKOS_FUNCTION double BesselI1(const double arg);

KOKKOS_NET_API_EXPORT KOKKOS_FUNCTION double BesselIN(const int nOrder, const double arg);

template<class ExecutionSpace>
KOKKOS_NET_API_EXPORT KOKKOS_FUNCTION KokkosViewFloat<ExecutionSpace> BesselI(const int nOrder, const KokkosViewFloat<ExecutionSpace>& arg, const int32 length);

template<class ExecutionSpace>
KOKKOS_NET_API_EXPORT KOKKOS_FUNCTION KokkosViewFloat<ExecutionSpace> BesselI0(const KokkosViewFloat<ExecutionSpace>& arg, const int32 length);

template<class ExecutionSpace>
KOKKOS_NET_API_EXPORT KOKKOS_FUNCTION KokkosViewFloat<ExecutionSpace> BesselI1(const KokkosViewFloat<ExecutionSpace>& arg, const int32 length);

template<class ExecutionSpace>
KOKKOS_NET_API_EXPORT KOKKOS_FUNCTION KokkosViewFloat<ExecutionSpace> BesselIN(const int nOrder, const KokkosViewFloat<ExecutionSpace>& arg, const int32 length);

template<class ExecutionSpace>
KOKKOS_NET_API_EXPORT KOKKOS_FUNCTION KokkosViewDouble<ExecutionSpace> BesselI(const int nOrder, const KokkosViewDouble<ExecutionSpace>& arg, const int32 length);

template<class ExecutionSpace>
KOKKOS_NET_API_EXPORT KOKKOS_FUNCTION KokkosViewDouble<ExecutionSpace> BesselI0(const KokkosViewDouble<ExecutionSpace>& arg, const int32 length);

template<class ExecutionSpace>
KOKKOS_NET_API_EXPORT KOKKOS_FUNCTION KokkosViewDouble<ExecutionSpace> BesselI1(const KokkosViewDouble<ExecutionSpace>& arg, const int32 length);

template<class ExecutionSpace>
KOKKOS_NET_API_EXPORT KOKKOS_FUNCTION KokkosViewDouble<ExecutionSpace> BesselIN(const int nOrder, const KokkosViewDouble<ExecutionSpace>& arg, const int32 length);

KOKKOS_NET_API_EXPORT KOKKOS_FUNCTION float BesselIE(const int nOrder, const float arg);

KOKKOS_NET_API_EXPORT KOKKOS_FUNCTION float BesselIE0(const float arg);

KOKKOS_NET_API_EXPORT KOKKOS_FUNCTION float BesselIE1(const float arg);

KOKKOS_NET_API_EXPORT KOKKOS_FUNCTION float BesselIEN(const int nOrder, const float arg);

KOKKOS_NET_API_EXPORT KOKKOS_FUNCTION double BesselIE(const int nOrder, const double arg);

KOKKOS_NET_API_EXPORT KOKKOS_FUNCTION double BesselIE0(const double arg);

KOKKOS_NET_API_EXPORT KOKKOS_FUNCTION double BesselIE1(const double arg);

KOKKOS_NET_API_EXPORT KOKKOS_FUNCTION double BesselIEN(const int nOrder, const double arg);

template<class ExecutionSpace>
KOKKOS_NET_API_EXPORT KOKKOS_FUNCTION KokkosViewFloat<ExecutionSpace> BesselIE(const int nOrder, const KokkosViewFloat<ExecutionSpace>& arg, const int32 length);

template<class ExecutionSpace>
KOKKOS_NET_API_EXPORT KOKKOS_FUNCTION KokkosViewFloat<ExecutionSpace> BesselIE0(const KokkosViewFloat<ExecutionSpace>& arg, const int32 length);

template<class ExecutionSpace>
KOKKOS_NET_API_EXPORT KOKKOS_FUNCTION KokkosViewFloat<ExecutionSpace> BesselIE1(const KokkosViewFloat<ExecutionSpace>& arg, const int32 length);

template<class ExecutionSpace>
KOKKOS_NET_API_EXPORT KOKKOS_FUNCTION KokkosViewFloat<ExecutionSpace> BesselIEN(const int nOrder, const KokkosViewFloat<ExecutionSpace>& arg, const int32 length);

template<class ExecutionSpace>
KOKKOS_NET_API_EXPORT KOKKOS_FUNCTION KokkosViewDouble<ExecutionSpace> BesselIE(const int nOrder, const KokkosViewDouble<ExecutionSpace>& arg, const int32 length);

template<class ExecutionSpace>
KOKKOS_NET_API_EXPORT KOKKOS_FUNCTION KokkosViewDouble<ExecutionSpace> BesselIE0(const KokkosViewDouble<ExecutionSpace>& arg, const int32 length);

template<class ExecutionSpace>
KOKKOS_NET_API_EXPORT KOKKOS_FUNCTION KokkosViewDouble<ExecutionSpace> BesselIE1(const KokkosViewDouble<ExecutionSpace>& arg, const int32 length);

template<class ExecutionSpace>
KOKKOS_NET_API_EXPORT KOKKOS_FUNCTION KokkosViewDouble<ExecutionSpace> BesselIEN(const int nOrder, const KokkosViewDouble<ExecutionSpace>& arg, const int32 length);

KOKKOS_NET_API_EXPORT KOKKOS_FUNCTION float BesselK(const int nOrder, const float arg);

KOKKOS_NET_API_EXPORT KOKKOS_FUNCTION float BesselK0(const float arg);

KOKKOS_NET_API_EXPORT KOKKOS_FUNCTION float BesselK1(const float arg);

KOKKOS_NET_API_EXPORT KOKKOS_FUNCTION float BesselKN(const int nOrder, const float arg);

KOKKOS_NET_API_EXPORT KOKKOS_FUNCTION double BesselK(const int nOrder, const double arg);

KOKKOS_NET_API_EXPORT KOKKOS_FUNCTION double BesselK0(const double arg);

KOKKOS_NET_API_EXPORT KOKKOS_FUNCTION double BesselK1(const double arg);

KOKKOS_NET_API_EXPORT KOKKOS_FUNCTION double BesselKN(const int nOrder, const double arg);

template<class ExecutionSpace>
KOKKOS_NET_API_EXPORT KOKKOS_FUNCTION KokkosViewFloat<ExecutionSpace> BesselK(const int nOrder, const KokkosViewFloat<ExecutionSpace>& arg, const int32 length);

template<class ExecutionSpace>
KOKKOS_NET_API_EXPORT KOKKOS_FUNCTION KokkosViewFloat<ExecutionSpace> BesselK0(const KokkosViewFloat<ExecutionSpace>& arg, const int32 length);

template<class ExecutionSpace>
KOKKOS_NET_API_EXPORT KOKKOS_FUNCTION KokkosViewFloat<ExecutionSpace> BesselK1(const KokkosViewFloat<ExecutionSpace>& arg, const int32 length);

template<class ExecutionSpace>
KOKKOS_NET_API_EXPORT KOKKOS_FUNCTION KokkosViewFloat<ExecutionSpace> BesselKN(const int nOrder, const KokkosViewFloat<ExecutionSpace>& arg, const int32 length);

template<class ExecutionSpace>
KOKKOS_NET_API_EXPORT KOKKOS_FUNCTION KokkosViewDouble<ExecutionSpace> BesselK(const int nOrder, const KokkosViewDouble<ExecutionSpace>& arg, const int32 length);

template<class ExecutionSpace>
KOKKOS_NET_API_EXPORT KOKKOS_FUNCTION KokkosViewDouble<ExecutionSpace> BesselK0(const KokkosViewDouble<ExecutionSpace>& arg, const int32 length);

template<class ExecutionSpace>
KOKKOS_NET_API_EXPORT KOKKOS_FUNCTION KokkosViewDouble<ExecutionSpace> BesselK1(const KokkosViewDouble<ExecutionSpace>& arg, const int32 length);

template<class ExecutionSpace>
KOKKOS_NET_API_EXPORT KOKKOS_FUNCTION KokkosViewDouble<ExecutionSpace> BesselKN(const int nOrder, const KokkosViewDouble<ExecutionSpace>& arg, const int32 length);

KOKKOS_NET_API_EXPORT KOKKOS_FUNCTION float BesselKE(const int nOrder, const float arg);

KOKKOS_NET_API_EXPORT KOKKOS_FUNCTION float BesselKE0(const float arg);

KOKKOS_NET_API_EXPORT KOKKOS_FUNCTION float BesselKE1(const float arg);

KOKKOS_NET_API_EXPORT KOKKOS_FUNCTION float BesselKEN(const int nOrder, const float arg);

KOKKOS_NET_API_EXPORT KOKKOS_FUNCTION double BesselKE(const int nOrder, const double arg);

KOKKOS_NET_API_EXPORT KOKKOS_FUNCTION double BesselKE0(const double arg);

KOKKOS_NET_API_EXPORT KOKKOS_FUNCTION double BesselKE1(const double arg);

KOKKOS_NET_API_EXPORT KOKKOS_FUNCTION double BesselKEN(const int nOrder, const double arg);

template<class ExecutionSpace>
KOKKOS_NET_API_EXPORT KOKKOS_FUNCTION KokkosViewFloat<ExecutionSpace> BesselKE(const int nOrder, const KokkosViewFloat<ExecutionSpace>& arg, const int32 length);

template<class ExecutionSpace>
KOKKOS_NET_API_EXPORT KOKKOS_FUNCTION KokkosViewFloat<ExecutionSpace> BesselKE0(const KokkosViewFloat<ExecutionSpace>& arg, const int32 length);

template<class ExecutionSpace>
KOKKOS_NET_API_EXPORT KOKKOS_FUNCTION KokkosViewFloat<ExecutionSpace> BesselKE1(const KokkosViewFloat<ExecutionSpace>& arg, const int32 length);

template<class ExecutionSpace>
KOKKOS_NET_API_EXPORT KOKKOS_FUNCTION KokkosViewFloat<ExecutionSpace> BesselKEN(const int nOrder, const KokkosViewFloat<ExecutionSpace>& arg, const int32 length);

template<class ExecutionSpace>
KOKKOS_NET_API_EXPORT KOKKOS_FUNCTION KokkosViewDouble<ExecutionSpace> BesselKE(const int nOrder, const KokkosViewDouble<ExecutionSpace>& arg, const int32 length);

template<class ExecutionSpace>
KOKKOS_NET_API_EXPORT KOKKOS_FUNCTION KokkosViewDouble<ExecutionSpace> BesselKE0(const KokkosViewDouble<ExecutionSpace>& arg, const int32 length);

template<class ExecutionSpace>
KOKKOS_NET_API_EXPORT KOKKOS_FUNCTION KokkosViewDouble<ExecutionSpace> BesselKE1(const KokkosViewDouble<ExecutionSpace>& arg, const int32 length);

template<class ExecutionSpace>
KOKKOS_NET_API_EXPORT KOKKOS_FUNCTION KokkosViewDouble<ExecutionSpace> BesselKEN(const int nOrder, const KokkosViewDouble<ExecutionSpace>& arg, const int32 length);

enum SpecialFunctions : unsigned short
{
    BESSELI_FLOAT,
    BESSELI0_FLOAT,
    BESSELI1_FLOAT,
    BESSELIN_FLOAT,
    BESSELI_DOUBLE,
    BESSELI0_DOUBLE,
    BESSELI1_DOUBLE,
    BESSELIN_DOUBLE,
    BESSELI_FLOAT_ARRAY,
    BESSELI0_FLOAT_ARRAY,
    BESSELI1_FLOAT_ARRAY,
    BESSELIN_FLOAT_ARRAY,
    BESSELI_DOUBLE_ARRAY,
    BESSELI0_DOUBLE_ARRAY,
    BESSELI1_DOUBLE_ARRAY,
    BESSELIN_DOUBLE_ARRAY,
    BESSELIE_FLOAT,
    BESSELIE0_FLOAT,
    BESSELIE1_FLOAT,
    BESSELIEN_FLOAT,
    BESSELIE_DOUBLE,
    BESSELIE0_DOUBLE,
    BESSELIE1_DOUBLE,
    BESSELIEN_DOUBLE,
    BESSELIE_FLOAT_ARRAY,
    BESSELIE0_FLOAT_ARRAY,
    BESSELIE1_FLOAT_ARRAY,
    BESSELIEN_FLOAT_ARRAY,
    BESSELIE_DOUBLE_ARRAY,
    BESSELIE0_DOUBLE_ARRAY,
    BESSELIE1_DOUBLE_ARRAY,
    BESSELIEN_DOUBLE_ARRAY,
    BESSELK_FLOAT,
    BESSELK0_FLOAT,
    BESSELK1_FLOAT,
    BESSELKN_FLOAT,
    BESSELK_DOUBLE,
    BESSELK0_DOUBLE,
    BESSELK1_DOUBLE,
    BESSELKN_DOUBLE,
    BESSELK_FLOAT_ARRAY,
    BESSELK0_FLOAT_ARRAY,
    BESSELK1_FLOAT_ARRAY,
    BESSELKN_FLOAT_ARRAY,
    BESSELK_DOUBLE_ARRAY,
    BESSELK0_DOUBLE_ARRAY,
    BESSELK1_DOUBLE_ARRAY,
    BESSELKN_DOUBLE_ARRAY,
    BESSELKE_FLOAT,
    BESSELKE0_FLOAT,
    BESSELKE1_FLOAT,
    BESSELKEN_FLOAT,
    BESSELKE_DOUBLE,
    BESSELKE0_DOUBLE,
    BESSELKE1_DOUBLE,
    BESSELKEN_DOUBLE,
    BESSELKE_FLOAT_ARRAY,
    BESSELKE0_FLOAT_ARRAY,
    BESSELKE1_FLOAT_ARRAY,
    BESSELKEN_FLOAT_ARRAY,
    BESSELKE_DOUBLE_ARRAY,
    BESSELKE0_DOUBLE_ARRAY,
    BESSELKE1_DOUBLE_ARRAY,
    BESSELKEN_DOUBLE_ARRAY,

    BESSEL_COUNT,

    BESSEL_UNKNOWN                 = 65535,
    BESSEL_UNKNOWN_PREDICATE       = BESSEL_UNKNOWN,
    BESSEL_UNKNOWN_ARRAY           = BESSEL_UNKNOWN,
    BESSEL_UNKNOWN_NATIVE_ARRAY    = BESSEL_UNKNOWN,
    BESSEL_UNKNOWN_OPENMP_ARRAY    = BESSEL_UNKNOWN,
    BESSEL_UNKNOWN_MKL_ARRAY       = BESSEL_UNKNOWN,
    BESSEL_UNKNOWN_ARRAY_PREDICATE = BESSEL_UNKNOWN
};

struct BesselMethods
{
    float (__host__ *BesselIfloat)(const int, const float);
    float (__host__ *BesselI0float)(const float);
    float (__host__ *BesselI1float)(const float);
    float (__host__ *BesselINfloat)(const int, const float);
    double (__host__ *BesselIdouble)(const int, const double);
    double (__host__ *BesselI0double)(const double);
    double (__host__ *BesselI1double)(const double);
    double (__host__ *BesselINdouble)(const int, const double);
    float (__host__ *BesselIEfloat)(const int, const float);
    float (__host__ *BesselIE0float)(const float);
    float (__host__ *BesselIE1float)(const float);
    float (__host__ *BesselIENfloat)(const int, const float);
    double (__host__ *BesselIEdouble)(const int, const double);
    double (__host__ *BesselIE0double)(const double);
    double (__host__ *BesselIE1double)(const double);
    double (__host__ *BesselIENdouble)(const int, const double);
    float (__host__ *BesselKfloat)(const int, const float);
    float (__host__ *BesselK0float)(const float);
    float (__host__ *BesselK1float)(const float);
    float (__host__ *BesselKNfloat)(const int, const float);
    double (__host__ *BesselKdouble)(const int, const double);
    double (__host__ *BesselK0double)(const double);
    double (__host__ *BesselK1double)(const double);
    double (__host__ *BesselKNdouble)(const int, const double);
    float (__host__ *BesselKEfloat)(const int, const float);
    float (__host__ *BesselKE0float)(const float);
    float (__host__ *BesselKE1float)(const float);
    float (__host__ *BesselKENfloat)(const int, const float);
    double (__host__ *BesselKEdouble)(const int, const double);
    double (__host__ *BesselKE0double)(const double);
    double (__host__ *BesselKE1double)(const double);
    double (__host__ *BesselKENdouble)(const int, const double);

#define ARRAY_METHODS(NAME, EXECUTIONSPACE)                                                                                           \
    KokkosViewFloat<EXECUTIONSPACE> (__host__ *BesselIfloat_ARRAY##NAME)(const int, const KokkosViewFloat<EXECUTIONSPACE>&, const int32);      \
    KokkosViewFloat<EXECUTIONSPACE> (__host__ *BesselI0float_ARRAY##NAME)(const KokkosViewFloat<EXECUTIONSPACE>&, const int32);                \
    KokkosViewFloat<EXECUTIONSPACE> (__host__ *BesselI1float_ARRAY##NAME)(const KokkosViewFloat<EXECUTIONSPACE>&, const int32);                \
    KokkosViewFloat<EXECUTIONSPACE> (__host__ *BesselINfloat_ARRAY##NAME)(const int, const KokkosViewFloat<EXECUTIONSPACE>&, const int32);     \
    KokkosViewDouble<EXECUTIONSPACE> (__host__ *BesselIdouble_ARRAY##NAME)(const int, const KokkosViewDouble<EXECUTIONSPACE>&, const int32);   \
    KokkosViewDouble<EXECUTIONSPACE> (__host__ *BesselI0double_ARRAY##NAME)(const KokkosViewDouble<EXECUTIONSPACE>&, const int32);             \
    KokkosViewDouble<EXECUTIONSPACE> (__host__ *BesselI1double_ARRAY##NAME)(const KokkosViewDouble<EXECUTIONSPACE>&, const int32);             \
    KokkosViewDouble<EXECUTIONSPACE> (__host__ *BesselINdouble_ARRAY##NAME)(const int, const KokkosViewDouble<EXECUTIONSPACE>&, const int32);  \
    KokkosViewFloat<EXECUTIONSPACE> (__host__ *BesselIEfloat_ARRAY##NAME)(const int, const KokkosViewFloat<EXECUTIONSPACE>&, const int32);     \
    KokkosViewFloat<EXECUTIONSPACE> (__host__ *BesselIE0float_ARRAY##NAME)(const KokkosViewFloat<EXECUTIONSPACE>&, const int32);               \
    KokkosViewFloat<EXECUTIONSPACE> (__host__ *BesselIE1float_ARRAY##NAME)(const KokkosViewFloat<EXECUTIONSPACE>&, const int32);               \
    KokkosViewFloat<EXECUTIONSPACE> (__host__ *BesselIENfloat_ARRAY##NAME)(const int, const KokkosViewFloat<EXECUTIONSPACE>&, const int32);    \
    KokkosViewDouble<EXECUTIONSPACE> (__host__ *BesselIEdouble_ARRAY##NAME)(const int, const KokkosViewDouble<EXECUTIONSPACE>&, const int32);  \
    KokkosViewDouble<EXECUTIONSPACE> (__host__ *BesselIE0double_ARRAY##NAME)(const KokkosViewDouble<EXECUTIONSPACE>&, const int32);            \
    KokkosViewDouble<EXECUTIONSPACE> (__host__ *BesselIE1double_ARRAY##NAME)(const KokkosViewDouble<EXECUTIONSPACE>&, const int32);            \
    KokkosViewDouble<EXECUTIONSPACE> (__host__ *BesselIENdouble_ARRAY##NAME)(const int, const KokkosViewDouble<EXECUTIONSPACE>&, const int32); \
    KokkosViewFloat<EXECUTIONSPACE> (__host__ *BesselKfloat_ARRAY##NAME)(const int, const KokkosViewFloat<EXECUTIONSPACE>&, const int32);      \
    KokkosViewFloat<EXECUTIONSPACE> (__host__ *BesselK0float_ARRAY##NAME)(const KokkosViewFloat<EXECUTIONSPACE>&, const int32);                \
    KokkosViewFloat<EXECUTIONSPACE> (__host__ *BesselK1float_ARRAY##NAME)(const KokkosViewFloat<EXECUTIONSPACE>&, const int32);                \
    KokkosViewFloat<EXECUTIONSPACE> (__host__ *BesselKNfloat_ARRAY##NAME)(const int, const KokkosViewFloat<EXECUTIONSPACE>&, const int32);     \
    KokkosViewDouble<EXECUTIONSPACE> (__host__ *BesselKdouble_ARRAY##NAME)(const int, const KokkosViewDouble<EXECUTIONSPACE>&, const int32);   \
    KokkosViewDouble<EXECUTIONSPACE> (__host__ *BesselK0double_ARRAY##NAME)(const KokkosViewDouble<EXECUTIONSPACE>&, const int32);             \
    KokkosViewDouble<EXECUTIONSPACE> (__host__ *BesselK1double_ARRAY##NAME)(const KokkosViewDouble<EXECUTIONSPACE>&, const int32);             \
    KokkosViewDouble<EXECUTIONSPACE> (__host__ *BesselKNdouble_ARRAY##NAME)(const int, const KokkosViewDouble<EXECUTIONSPACE>&, const int32);  \
    KokkosViewFloat<EXECUTIONSPACE> (__host__ *BesselKEfloat_ARRAY##NAME)(const int, const KokkosViewFloat<EXECUTIONSPACE>&, const int32);     \
    KokkosViewFloat<EXECUTIONSPACE> (__host__ *BesselKE0float_ARRAY##NAME)(const KokkosViewFloat<EXECUTIONSPACE>&, const int32);               \
    KokkosViewFloat<EXECUTIONSPACE> (__host__ *BesselKE1float_ARRAY##NAME)(const KokkosViewFloat<EXECUTIONSPACE>&, const int32);               \
    KokkosViewFloat<EXECUTIONSPACE> (__host__ *BesselKENfloat_ARRAY##NAME)(const int, const KokkosViewFloat<EXECUTIONSPACE>&, const int32);    \
    KokkosViewDouble<EXECUTIONSPACE> (__host__ *BesselKEdouble_ARRAY##NAME)(const int, const KokkosViewDouble<EXECUTIONSPACE>&, const int32);  \
    KokkosViewDouble<EXECUTIONSPACE> (__host__ *BesselKE0double_ARRAY##NAME)(const KokkosViewDouble<EXECUTIONSPACE>&, const int32);            \
    KokkosViewDouble<EXECUTIONSPACE> (__host__ *BesselKE1double_ARRAY##NAME)(const KokkosViewDouble<EXECUTIONSPACE>&, const int32);            \
    KokkosViewDouble<EXECUTIONSPACE> (__host__ *BesselKENdouble_ARRAY##NAME)(const int, const KokkosViewDouble<EXECUTIONSPACE>&, const int32);

    ARRAY_METHODS(Serial, Kokkos::Serial)

    ARRAY_METHODS(OpenMP, Kokkos::OpenMP)

    ARRAY_METHODS(Cuda, Kokkos::Cuda)

#undef ARRAY_METHODS
};
