using System;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Versioning;
using System.Text.Json.Serialization;

using RuntimeGeneration;

namespace Kokkos
{
    //[ILGeneration(typeof(int),
    //              typeof(uint))]
    //internal static class Interop<T>
    //{
    //    internal delegate uint CudaGetDeviceCountDelegate();

    //    internal delegate T CudaGetComputeCapabilityDelegate(T device_id);

    //    internal delegate void PrintConfigurationDelegate(in bool detail);

    //    [NativeCall("runtime.Kokkos.NET.dll",
    //                0x20A0)]
    //    internal static CudaGetDeviceCountDelegate CudaGetDeviceCount;

    //    [NativeCall("runtime.Kokkos.NET.dll",
    //                0x20E0)]
    //    internal static CudaGetComputeCapabilityDelegate CudaGetComputeCapability;

    //    [NativeCall("runtime.Kokkos.NET.dll",
    //                "_Z18PrintConfigurationRKb")]
    //    internal static PrintConfigurationDelegate PrintConfiguration;
    //}

    //[StructLayout(System.Runtime.InteropServices.LayoutKind.Sequential)]
    //public sealed unsafe class FractureProperties<T>
    //    where T : unmanaged
    //{
    //    private static readonly Type t = typeof(T);

    //    private static readonly int widthOffset;

    //    private static readonly int heightOffset;

    //    static FractureProperties()
    //    {
    //        widthOffset  = Marshal.OffsetOf<FractureProperties<T>>("_width").ToInt32();
    //        heightOffset = Marshal.OffsetOf<FractureProperties<T>>("_height").ToInt32();
    //    }

    //    private NativePointer pointer;

    //    private T _width;

    //    public T Width
    //    {
    //        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
    //        get { return *(T*)(pointer.Data + widthOffset); }
    //        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
    //        set { *(T*)(pointer.Data + widthOffset) = value; }
    //    }

    //    private T _height;

    //    public T Height
    //    {
    //        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
    //        get { return *(T*)(pointer.Data + heightOffset); }
    //        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
    //        set { *(T*)(pointer.Data + heightOffset) = value; }
    //    }

    //    public FractureProperties()
    //    {
    //        pointer = NativePointer.Allocate(Unsafe.SizeOf<FractureProperties<T>>(), ExecutionSpaceKind.Cuda);
    //    }
    //}

    internal class Program
    {
        //static Program()
        //{
        //    RuntimeCil.Generate(typeof(Program).Assembly);
        //}

        //[STAThread]
        private static void Main(string[] args)
        {
            int  num_threads      = 4;
            int  num_numa         = 1;
            int  device_id        = 0;
            int  ndevices         = 1;
            int  skip_device      = 9999;
            bool disable_warnings = false;

            InitArguments arguments = new InitArguments(num_threads,
                                                        num_numa,
                                                        device_id,
                                                        ndevices,
                                                        skip_device,
                                                        disable_warnings);

            using(ScopeGuard sg = new ScopeGuard(arguments))
                //ParallelProcessor.Initialize();
            {
                //PrintConfiguration(false);

                //Console.WriteLine(CudaGetDeviceCount());

                //FractureProperties<double> fd = new FractureProperties<double>();

                //fd.Height = 654.123;
                //fd.Width  = 9876.2456;

                //Console.WriteLine(fd.Height);
                //Console.WriteLine(fd.Width);

                Tests.ViewTests p = new Tests.ViewTests();
                p.Run();
            }

            using(ScopeGuard sg = new ScopeGuard(arguments))
                //ParallelProcessor.Initialize();
            {
                //PrintConfiguration(false);

                //Console.WriteLine(CudaGetDeviceCount());

                //FractureProperties<double> fd = new FractureProperties<double>();

                //fd.Height = 654.123;
                //fd.Width  = 9876.2456;

                //Console.WriteLine(fd.Height);
                //Console.WriteLine(fd.Width);

                Tests.ViewTests p = new Tests.ViewTests();
                p.Run();
            }
            //ParallelProcessor.Shutdown();

#if DEBUG
            Console.WriteLine("press any key to exit.");
            Console.ReadKey();
#endif
        }
    }
}