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
    }

    [NonVersionable]
    public struct Serial : IExecutionSpace
    {
    }

    [NonVersionable]
    public struct OpenMP : IExecutionSpace
    {
    }

    public static class ExecutionSpace<T>
        where T : IExecutionSpace
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
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


        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
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