#nullable enable
using System;
using System.Diagnostics;
using System.Diagnostics.Contracts;
using System.Runtime.CompilerServices;

namespace Kokkos.Utilities
{
    [DebuggerDisplay("{ToString(),raw}")]
    public sealed class Box<T>
        where T : struct
    {
        private Box()
        {
            throw new InvalidOperationException("The Microsoft.Toolkit.HighPerformance.Box<T> constructor should never be used");
        }

        [Pure]
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static Box<T> GetFrom(object obj)
        {
            if(obj.GetType() != typeof(T))
            {
                ThrowInvalidCastExceptionForGetFrom();
            }

            return Unsafe.As<Box<T>>(obj);
        }

        [Pure]
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static Box<T> DangerousGetFrom(object obj)
        {
            return Unsafe.As<Box<T>>(obj);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static bool TryGetFrom(object      obj,
                                      out Box<T>? box)
        {
            if(obj.GetType() == typeof(T))
            {
                box = Unsafe.As<Box<T>>(obj);

                return true;
            }

            box = null;

            return false;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static implicit operator T(Box<T> box)
        {
            return (T)(object)box;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static implicit operator Box<T>(T value)
        {
            // The Box<T> type is never actually instantiated.
            // Here we are just boxing the input T value, and then reinterpreting
            // that object reference as a Box<T> reference. As such, the Box<T>
            // type is really only used as an interface to access the contents
            // of a boxed value type. This also makes it so that additional methods
            // like ToString() or GetHashCode() will automatically be referenced from
            // the method table of the boxed object, meaning that they don't need to
            // manually be implemented in the Box<T> type. For instance, boxing a float
            // and calling ToString() on it directly, on its boxed object or on a Box<T>
            // reference retrieved from it will produce the same result in all cases.
            return Unsafe.As<Box<T>>(value);
        }

        public override string ToString()
        {
            // Here we're overriding the base object virtual methods to ensure
            // calls to those methods have a correct results on all runtimes.
            // For instance, not doing so is causing issue on .NET Core 2.1 Release
            // due to how the runtime handles the Box<T> reference to an actual
            // boxed T value (not a concrete Box<T> instance as it would expect).
            // To fix that, the overrides will simply call the expected methods
            // directly on the boxed T values. These methods will be directly
            // invoked by the JIT compiler when using a Box<T> reference. When
            // an object reference is used instead, the call would be forwarded
            // to those same methods anyway, since the method table for an object
            // representing a T instance is the one of type T anyway.
            return this.GetReference().ToString()!;
        }

        public override bool Equals(object? obj)
        {
            return Equals(this, obj);
        }

        public override int GetHashCode()
        {
            return this.GetReference().GetHashCode();
        }

        private static void ThrowInvalidCastExceptionForGetFrom()
        {
            throw new InvalidCastException($"Can't cast the input object to the type Box<{typeof(T)}>");
        }
    }
}