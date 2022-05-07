using System;
using System.Runtime.CompilerServices;

namespace Kokkos
{
    public static class HashCodeExtensions
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static void Add<T>(ref this HashCode hashCode,
                                  ReadOnlySpan<T>   span)
            where T : notnull
        {
            int hash = HashCode<T>.CombineValues(span);

            hashCode.Add(hash);
        }
    }
}