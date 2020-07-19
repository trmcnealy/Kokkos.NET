using System.Runtime.CompilerServices;
using System.Runtime.Versioning;

namespace Kokkos
{
    public enum ExecutionSpaceKind : ushort
    {
        Unknown = ushort.MaxValue,
        Serial  = 0,
        OpenMP,
        Cuda
    }

    [NonVersionable]
    public struct Cuda : IExecutionSpace
    {
        public LayoutKind DefaultLayout
        {
    #if NETSTANDARD
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
#else
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
#endif
            get { return LayoutKind.Left; }
        }
    }

    [NonVersionable]
    public struct Serial : IExecutionSpace
    {
        public LayoutKind DefaultLayout
        {
    #if NETSTANDARD
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
#else
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
#endif
            get { return LayoutKind.Right; }
        }
    }

    [NonVersionable]
    public struct OpenMP : IExecutionSpace
    {
        public LayoutKind DefaultLayout
        {
    #if NETSTANDARD
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
#else
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
#endif
            get { return LayoutKind.Right; }
        }
    }

    public static class ExecutionSpace<T>
        where T : IExecutionSpace
    {
#if NETSTANDARD
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
#else
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
#endif
        public static ExecutionSpaceKind GetKind()
        {
            if(typeof(T) == typeof(Serial))
            {
                return ExecutionSpaceKind.Serial;
            }

            if(typeof(T) == typeof(OpenMP))
            {
                return ExecutionSpaceKind.OpenMP;
            }

            if(typeof(T) == typeof(Cuda))
            {
                return ExecutionSpaceKind.Cuda;
            }

            return ExecutionSpaceKind.Unknown;
        }


#if NETSTANDARD
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
#else
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
#endif
        public static LayoutKind GetLayout()
        {
            if(typeof(T) == typeof(Serial))
            {
                return LayoutKind.Right;
            }

            if(typeof(T) == typeof(OpenMP))
            {
                return LayoutKind.Right;
            }

            if(typeof(T) == typeof(Cuda))
            {
                return LayoutKind.Left;
            }

            return LayoutKind.Unknown;
        }
    }
}