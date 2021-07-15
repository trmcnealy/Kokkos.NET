using System;
using System.Runtime.CompilerServices;

namespace Kokkos
{
    public sealed class ScopeGuard : IDisposable
    {
        private static bool _initialized;

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        static ScopeGuard()
        {
            AppDomain.CurrentDomain.ProcessExit += Dtor;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        private ScopeGuard()
        {
            //if(_initialized)
            //{
            //    Console.WriteLine("Kokkos can only be Initialized once per process.");
            //}
        }

        ~ScopeGuard()
        {
            Dispose();
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public void Dispose()
        {
            GC.Collect();
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        private static void Dtor(object?   sender,
                                 EventArgs e)
        {
            if(KokkosLibrary.IsInitialized() || _initialized)
            {
                KokkosLibrary.FinalizeAll();
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static ScopeGuard Get()
        {
            if(!_initialized)
            {
                try
                {
                    KokkosLibrary.Initialize(Environment.ProcessorCount, 0);
                }
                catch(System.Exception ex)
                {
                    Console.WriteLine("Kokkos failed to initialize.\n"+ex.Message);
                }
                _initialized = true;
            }

            return new ScopeGuard();
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static ScopeGuard Get(int gpuDeviceId)
        {
            if(!_initialized)
            {
                try
                {
                    KokkosLibrary.Initialize(Environment.ProcessorCount, gpuDeviceId);
                }
                catch(System.Exception ex)
                {
                    Console.WriteLine("Kokkos failed to initialize.\n" +ex.Message);
                }
                _initialized = true;
            }

            return new ScopeGuard();
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static ScopeGuard Get(int numCpuThreads,
                                     int gpuDeviceId)
        {
            if(!_initialized)
            {
                try
                {
                    KokkosLibrary.Initialize(numCpuThreads, gpuDeviceId);
                }
                catch(System.Exception ex)
                {
                    Console.WriteLine("Kokkos failed to initialize.\n" +ex.Message);
                }
                _initialized = true;
            }

            return new ScopeGuard();
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static ScopeGuard Get(InitArguments arguments)
        {
            if(!_initialized)
            {
                try
                {
                    KokkosLibrary.Initialize(arguments);
                }
                catch(System.Exception ex)
                {
                    Console.WriteLine("Kokkos failed to initialize.\n" +ex.Message);
                }
                _initialized = true;
            }

            return new ScopeGuard();
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static void Print()
        {
            KokkosLibrary.PrintConfiguration(true);
        }
    }
}
