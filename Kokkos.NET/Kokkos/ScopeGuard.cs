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
            AppDomain.CurrentDomain.ProcessExit += RrcTexasDataAdapter_Dtor;
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
        private static void RrcTexasDataAdapter_Dtor(object?    sender,
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
                KokkosLibrary.Initialize(Environment.ProcessorCount, 0);
                _initialized = true;
            }

            return new ScopeGuard();
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static ScopeGuard Get(int gpuDeviceId)
        {
            if(!_initialized)
            {
                KokkosLibrary.Initialize(Environment.ProcessorCount, gpuDeviceId);
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
                KokkosLibrary.Initialize(numCpuThreads, gpuDeviceId);
                _initialized = true;
            }

            return new ScopeGuard();
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static ScopeGuard Get(InitArguments arguments)
        {
            if(!_initialized)
            {
                KokkosLibrary.Initialize(arguments);
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