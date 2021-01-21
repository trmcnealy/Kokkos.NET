using System;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace Kokkos.Utilities
{
    public readonly ref struct NullableRef<T>
    {
        internal readonly Span<T> Span;

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public NullableRef(ref T value)
        {
            Span = MemoryMarshal.CreateSpan(ref value, 1);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        private NullableRef(Span<T> span)
        {
            Span = span;
        }

        public static NullableRef<T> Null
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            get { return default; }
        }

        public bool HasValue
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            get
            {
                // We know that the span will always have a length of either
                // 1 or 0, so instead of using a cmp instruction and setting the
                // zero flag to produce our boolean value, we can just cast
                // the length to byte without overflow checks (doing a cast will
                // also account for the byte endianness of the current system),
                // and then reinterpret that value to a bool flag.
                // This results in a single movzx instruction on x86-64.
                byte length = unchecked((byte)Span.Length);

                return Unsafe.As<byte, bool>(ref length);
            }
        }

        public ref T Value
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            get
            {
                if(!HasValue)
                {
                    ThrowInvalidOperationException();
                }

                return ref MemoryMarshal.GetReference(Span);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static implicit operator NullableRef<T>(Ref<T> reference)
        {
            return new NullableRef<T>(reference.Span);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static explicit operator T(NullableRef<T> reference)
        {
            return reference.Value;
        }

        private static void ThrowInvalidOperationException()
        {
            throw new InvalidOperationException("The current instance doesn't have a value that can be accessed");
        }
    }
}