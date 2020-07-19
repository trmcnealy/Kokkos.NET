using System;
using System.Runtime.CompilerServices;

namespace Kokkos
{
    public sealed class ScopeGuard : IDisposable
    {
        private static bool _initialized;

#if NETSTANDARD
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
#else
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
#endif
        static ScopeGuard()
        {
            AppDomain.CurrentDomain.ProcessExit += RrcTexasDataAdapter_Dtor;
        }

#if NETSTANDARD
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
#else
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
#endif
        private ScopeGuard()
        {
            //if(_initialized)
            //{
            //    Console.WriteLine("Kokkos can only be Initialized once per process.");
            //}
        }

#if NETSTANDARD
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
#else
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
#endif
        public void Dispose()
        {
        }

#if NETSTANDARD
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
#else
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
#endif
        private static void RrcTexasDataAdapter_Dtor(object    sender,
                                                     EventArgs e)
        {
            if(KokkosLibrary.IsInitialized() || _initialized)
            {
                KokkosLibrary.FinalizeAll();
            }
        }

#if NETSTANDARD
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
#else
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
#endif
        public static ScopeGuard Get()
        {
            if(!_initialized)
            {
                KokkosLibrary.Initialize(Environment.ProcessorCount, 0);
                _initialized = true;
            }

            return new ScopeGuard();
        }

#if NETSTANDARD
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
#else
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
#endif
        public static ScopeGuard Get(int gpuDeviceId)
        {
            if(!_initialized)
            {
                KokkosLibrary.Initialize(Environment.ProcessorCount, gpuDeviceId);
                _initialized = true;
            }

            return new ScopeGuard();
        }

#if NETSTANDARD
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
#else
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
#endif
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

#if NETSTANDARD
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
#else
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
#endif
        public static ScopeGuard Get(in InitArguments arguments)
        {
            if(!_initialized)
            {
                KokkosLibrary.Initialize(arguments);
                _initialized = true;
            }

            return new ScopeGuard();
        }

#if NETSTANDARD
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
#else
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
#endif
        public static void Print()
        {
            KokkosLibrary.PrintConfiguration(true);
        }
    }
}
