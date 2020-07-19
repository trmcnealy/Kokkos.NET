using System;
using System.Runtime.CompilerServices;

namespace Kokkos
{
    internal class KokkosLibraryException : Exception
    {
        public KokkosLibraryException()
        {
        }

        public KokkosLibraryException(string message)
            : base(message)
        {
        }

#if NETSTANDARD
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
#else
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
#endif
        internal static void Throw()
        {
            throw new KokkosLibraryException();
        }

#if NETSTANDARD
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
#else
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
#endif
        internal static void Throw(string message)
        {
            throw new KokkosLibraryException(message);
        }
    }
}