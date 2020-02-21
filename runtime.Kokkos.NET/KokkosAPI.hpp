#pragma once

#include <Types.hpp>

#ifndef EXTERN
#    define EXTERN extern
#endif

#ifndef EXTERN_C
#    define EXTERN_C extern "C"
#endif

#ifdef KOKKOS_NET_EXPORTS

#    ifndef KOKKOS_NET_API_EXPORT
#        define KOKKOS_NET_API_EXPORT __declspec(dllexport)
#    endif

#    ifndef KOKKOS_NET_API_EXTERNC
#        define KOKKOS_NET_API_EXTERNC EXTERN_C __declspec(dllexport)
#    endif

#    ifndef KOKKOS_NET_API_EXTERN
#        define KOKKOS_NET_API_EXTERN EXTERN __declspec(dllexport)
#    endif

#    ifndef KOKKOS_NET_API_IMPORT
#        define KOKKOS_NET_API_IMPORT __declspec(dllimport)
#    endif

#else

#    ifndef KOKKOS_NET_API_EXPORT
#        define KOKKOS_NET_API_EXPORT __declspec(dllimport)
#    endif

#    ifndef KOKKOS_NET_API_EXTERNC
#        define KOKKOS_NET_API_EXTERNC EXTERN_C __declspec(dllimport)
#    endif

#    ifndef KOKKOS_NET_API_EXTERN
#        define KOKKOS_NET_API_EXTERN EXTERN __declspec(dllimport)
#    endif

#    ifndef KOKKOS_NET_API_IMPORT
#        define KOKKOS_NET_API_IMPORT __declspec(dllexport)
#    endif

#endif
