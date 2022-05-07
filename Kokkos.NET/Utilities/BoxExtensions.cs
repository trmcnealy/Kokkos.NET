using System.Diagnostics.Contracts;
using System.Runtime.CompilerServices;

namespace Kokkos
{
    public static class BoxExtensions
    {
        [Pure]
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static ref T GetReference<T>(this Box<T> box)
            where T : struct
        {
            return ref Unsafe.Unbox<T>(box);
        }
    }
}