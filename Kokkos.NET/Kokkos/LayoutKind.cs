using System.Runtime.CompilerServices;
using System.Runtime.Versioning;

namespace Kokkos
{
    public enum LayoutKind : ushort
    {
        Unknown = ushort.MaxValue,
        Left    = 0,
        Right,
        Stride
    }

    [NonVersionable]
    public struct LayoutLeft : ILayout
    {
    }

    [NonVersionable]
    public struct LayoutRight : ILayout
    {
    }

    [NonVersionable]
    public struct LayoutStride : ILayout
    {
    }

    public static class Layout<T>
        where T : ILayout
    {
#if NETSTANDARD
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
#else
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
#endif
        public static LayoutKind GetKind()
        {
            if(typeof(T) == typeof(LayoutLeft))
            {
                return LayoutKind.Left;
            }

            if(typeof(T) == typeof(LayoutRight))
            {
                return LayoutKind.Right;
            }

            if(typeof(T) == typeof(LayoutStride))
            {
                return LayoutKind.Stride;
            }

            return LayoutKind.Unknown;
        }
    }
}