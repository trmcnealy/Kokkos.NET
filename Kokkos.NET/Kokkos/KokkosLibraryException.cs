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

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        internal static void Throw()
        {
            throw new KokkosLibraryException();
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        internal static void Throw(string message)
        {
            throw new KokkosLibraryException(message);
        }
    }

    internal class KokkosInitializedException : Exception
    {
        public KokkosInitializedException()
        {
        }

        public KokkosInitializedException(string message)
            : base(message)
        {
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        internal static void Throw()
        {
            throw new KokkosInitializedException("Kokkos has not been initialized.");
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        internal static void Throw(string message)
        {
            throw new KokkosInitializedException(message);
        }
    }
}