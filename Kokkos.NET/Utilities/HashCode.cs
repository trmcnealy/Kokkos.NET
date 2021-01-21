#nullable enable
using System;
using System.Diagnostics.Contracts;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace Kokkos.Utilities
{
    public struct HashCode<T>
        where T : notnull
    {
        [Pure]
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static int Combine(ReadOnlySpan<T> span)
        {
            int hash = CombineValues(span);

            return HashCode.Combine(hash);
        }

        [Pure]
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static unsafe int CombineValues(ReadOnlySpan<T> span)
        {
            ref T r0 = ref MemoryMarshal.GetReference(span);

            if(RuntimeHelpers.IsReferenceOrContainsReferences<T>())
            {
                return SpanHelper.GetDjb2HashCode(ref r0, (nint)(void*)(uint)span.Length);
            }

            ref byte rb     = ref Unsafe.As<T, byte>(ref r0);
            nint   length = (nint)(void*)((uint)span.Length * (uint)Unsafe.SizeOf<T>());

            return SpanHelper.GetDjb2LikeByteHash(ref rb, length);
        }
    }
}