using System;
using System.Diagnostics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Versioning;
using System.Text.Json.Serialization;

using Kokkos.Tests;

using PlatformApi;

namespace Kokkos
{
    internal class Program
    {
        static Program()
        {
            Environment.SetEnvironmentVariable("KMP_DUPLICATE_LIB_OK", "TRUE");
        }


        [STAThread]
        private static void Main(string[] args)
        {

            //int  num_threads      = 4;
            //int  num_numa         = 1;
            //int  device_id        = 0;
            //int  ndevices         = 1;
            //int  skip_device      = 9999;
            //bool disable_warnings = false;

            //InitArguments arguments = new (num_threads, num_numa, device_id, ndevices, skip_device, disable_warnings);

            //KokkosLibrary.Initialize(4, 0);

            //KokkosLibrary.PrintConfiguration(true);

            //KokkosLibrary.FinalizeAll();

            ViewTests<Cuda> testsCuda = new();
            testsCuda.Run();

            ViewTests<OpenMP> testsOpenMP = new();
            testsOpenMP.Run();

            //Test1();



            //Console.WriteLine(CpuUsage.GetByProcess().Value.UserUsage.);

#if DEBUG
            Console.WriteLine("press any key to exit.");
            Console.ReadKey();
#endif
        }

        private static void Test1()
        {
            //int  num_threads      = 4;
            //int  num_numa         = 1;
            //int  device_id        = 0;
            //int  ndevices         = 1;
            //int  skip_device      = 9999;
            //bool disable_warnings = false;

            //InitArguments arguments = new InitArguments(num_threads, num_numa, device_id, ndevices, skip_device, disable_warnings);

            //using(ScopeGuard.Get(arguments))
            //{
            //    View<long, Cuda> lineEndingsView = CsvReader<Cuda>.GetCountLineEndings("R:/DrillingPermits.csv");

            //    ulong counter = 0;

            //    for(ulong i0 = 0; i0 < lineEndingsView.Extent(0); ++i0)
            //    {
            //        if(lineEndingsView[i0] != 0)
            //        {
            //            ++counter;
            //        }
            //    }

            //    long[] lineEndings = new long[counter];

            //    counter = 0;

            //    for(ulong i0 = 0; i0 < lineEndingsView.Extent(0); ++i0)
            //    {
            //        if(lineEndingsView[i0] != 0)
            //        {
            //            lineEndings[counter++] = lineEndingsView[i0];
            //        }
            //    }

            //    Console.WriteLine(lineEndings.LongLength);

            //    Array.Sort(lineEndings);

            //    for(long i0 = 0; i0 < 100; ++i0)
            //    {
            //        Console.WriteLine(lineEndings[i0]);
            //    }

            //    //PrintConfiguration(false);

            //    //Console.WriteLine(CudaGetDeviceCount());

            //    //FractureProperties<double> fd = new FractureProperties<double>();

            //    //fd.Height = 654.123;
            //    //fd.Width  = 9876.2456;

            //    //Console.WriteLine(fd.Height);
            //    //Console.WriteLine(fd.Width);

            //    //Tests.ViewTests p = new Tests.ViewTests();
            //    //p.Run();
            //}

            //using (ScopeGuard.Get(arguments))
            ////ParallelProcessor.Initialize();
            //{
            //    //PrintConfiguration(false);

            //    //Console.WriteLine(CudaGetDeviceCount());

            //    //FractureProperties<double> fd = new FractureProperties<double>();

            //    //fd.Height = 654.123;
            //    //fd.Width  = 9876.2456;

            //    //Console.WriteLine(fd.Height);
            //    //Console.WriteLine(fd.Width);

            //    //Tests.ViewTests p = new Tests.ViewTests();
            //    //p.Run();
            //}
            ////ParallelProcessor.Shutdown();
        }
    }
}
