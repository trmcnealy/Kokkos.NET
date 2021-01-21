using System;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace Kokkos.Utilities
{
    public readonly ref struct ReadOnlyRef<T>
    {
        internal readonly ReadOnlySpan<T> Span;

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public ReadOnlyRef(in T value)
        {
            ref T r0 = ref Unsafe.AsRef(value);

            Span = MemoryMarshal.CreateReadOnlySpan(ref r0, 1);
        }

        public ref readonly T Value
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            get { return ref MemoryMarshal.GetReference(Span); }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static implicit operator ReadOnlyRef<T>(Ref<T> reference)
        {
            return new ReadOnlyRef<T>(reference.Value);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static implicit operator T(ReadOnlyRef<T> reference)
        {
            return reference.Value;
        }
    }
}