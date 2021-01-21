using System;
using System.ComponentModel;
using System.Runtime.CompilerServices;
using System.Runtime.Versioning;

namespace Kokkos
{
    //[TypeConverter(typeof(EnumDescriptionTypeConverter))]
    public enum ExecutionSpaceKind : ushort
    {
        Unknown = ushort.MaxValue,
        [Description("Serial processing on the CPU")]
        Serial  = 0,
        [Description("OpenMP parallel processing on the CPU")]
        OpenMP = 1,
        [Description("Cuda parallel processing on the GPU")]
        Cuda = 2
    }

    [NonVersionable]
    public struct Cuda : IExecutionSpace
    {
        public static int  Id              { get; set; } = 0;
        public static int  NumberOfDevices { get; set; } = 1;
        public static int  SkipDevice      { get; set; } = 9999;
        public static bool DisableWarnings { get; set; } = true;
        
        public LayoutKind DefaultLayout
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            get { return LayoutKind.Left; }
        }

        public override string ToString()
        {
            return "Cuda";
        }
    }

    [NonVersionable]
    public struct Serial : IExecutionSpace
    {
        public static bool DisableWarnings { get; set; } = true;
        
        public LayoutKind DefaultLayout
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            get { return LayoutKind.Right; }
        }

        public override string ToString()
        {
            return "Serial";
        }
    }

    [NonVersionable]
    public struct OpenMP : IExecutionSpace
    {
        public static int  NumberOfThreads { get; set; } = Environment.ProcessorCount;
        public static bool DisableWarnings { get; set; } = true;
    
        public LayoutKind DefaultLayout
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            get { return LayoutKind.Right; }
        }

        public override string ToString()
        {
            return "OpenMP";
        }
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