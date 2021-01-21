using System;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace Kokkos.Utilities
{
    public readonly ref struct NullableReadOnlyRef<T>
    {
        private readonly ReadOnlySpan<T> span;

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public NullableReadOnlyRef(in T value)
        {
            ref T r0 = ref Unsafe.AsRef(value);

            span = MemoryMarshal.CreateReadOnlySpan(ref r0, 1);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        private NullableReadOnlyRef(ReadOnlySpan<T> span)
        {
            this.span = span;
        }

        public static NullableReadOnlyRef<T> Null
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            get { return default; }
        }

        public bool HasValue
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            get
            {
                // See comment in NullableRef<T> about this
                byte length = unchecked((byte)span.Length);

                return Unsafe.As<byte, bool>(ref length);
            }
        }

        public ref readonly T Value
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            get
            {
                if(!HasValue)
                {
                    ThrowInvalidOperationException();
                }

                return ref MemoryMarshal.GetReference(span);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static implicit operator NullableReadOnlyRef<T>(Ref<T> reference)
        {
            return new NullableReadOnlyRef<T>(reference.Span);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static implicit operator NullableReadOnlyRef<T>(ReadOnlyRef<T> reference)
        {
            return new NullableReadOnlyRef<T>(reference.Span);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static implicit operator NullableReadOnlyRef<T>(NullableRef<T> reference)
        {
            return new NullableReadOnlyRef<T>(reference.Span);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static explicit operator T(NullableReadOnlyRef<T> reference)
        {
            return reference.Value;
        }

        private static void ThrowInvalidOperationException()
        {
            throw new InvalidOperationException("The current instance doesn't have a value that can be accessed");
        }
    }
}