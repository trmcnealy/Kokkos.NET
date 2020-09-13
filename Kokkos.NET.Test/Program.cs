using System;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Versioning;
using System.Text.Json.Serialization;

using Kokkos.Tests;

using RuntimeGeneration;

namespace Kokkos
{
    internal class Program
    {

        [STAThread]
        private static void Main(string[] args)
        {
            MemoryMappedTest.Test();

#if DEBUG
            Console.WriteLine("press any key to exit.");
            Console.ReadKey();
#endif
        }
        static void Test1()
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

            using(ScopeGuard.Get(arguments))
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

            using(ScopeGuard.Get(arguments))
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
        }
    }
}