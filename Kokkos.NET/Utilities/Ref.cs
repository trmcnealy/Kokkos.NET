using System;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace Kokkos.Utilities
{
    public readonly ref struct Ref<T>
    {
        internal readonly Span<T> Span;

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public Ref(ref T value)
        {
            Span = MemoryMarshal.CreateSpan(ref value, 1);
        }

        public ref T Value
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            get { return ref MemoryMarshal.GetReference(Span); }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static implicit operator T(Ref<T> reference)
        {
            return reference.Value;
        }
    }
}