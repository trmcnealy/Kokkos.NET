using System;
using System.Runtime.CompilerServices;

namespace Kokkos
{
    public static class ParallelProcessor
    {
#if NETSTANDARD
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
#else
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
#endif
        private static void RuntimeTest()
        {
            //if(Thread.CurrentThread.GetApartmentState() != ApartmentState.STA || Thread.CurrentThread.IsBackground || Thread.CurrentThread.IsThreadPoolThread || !Thread.CurrentThread.IsAlive)
            //{
            //    KokkosLibraryException.Throw("Kokkos Library must be initialized on the main (STA) thread.");
            //}

            if(KokkosLibrary.IsLoaded())
            {
                KokkosLibraryException.Throw("Kokkos Library has not been initialized.");
            }
        }

#if NETSTANDARD
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
#else
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
#endif
        public static void Initialize(int gpuDeviceId = 0)
        {
            KokkosLibrary.Initialize(Environment.ProcessorCount,
                                     gpuDeviceId);

            //AppDomain.CurrentDomain.ProcessExit += View_Dtor;
        }

#if NETSTANDARD
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
#else
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
#endif
        public static void Initialize(int numCpuThreads,
                                      int gpuDeviceId)
        {
            KokkosLibrary.Initialize(numCpuThreads <= 0 ? Environment.ProcessorCount : numCpuThreads,
                                     gpuDeviceId);

            //AppDomain.CurrentDomain.ProcessExit += View_Dtor;
        }

#if NETSTANDARD
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
#else
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
#endif
        public static void Initialize(InitArguments arguments)
        {
            KokkosLibrary.Initialize(arguments);

            //AppDomain.CurrentDomain.ProcessExit += View_Dtor;
        }

#if NETSTANDARD
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
#else
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
#endif
        public static void Shutdown()
        {
            KokkosLibrary.@Finalize();
        }

#if NETSTANDARD
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
#else
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
#endif
        public static bool IsRunning()
        {
            if(KokkosLibrary.IsLoaded())
            {
                return KokkosLibrary.IsInitialized();
            }

            return false;
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

        //[MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        //internal static void View_Dtor(object    sender,
        //                               EventArgs e)
        //{
        //    if (KokkosActivated)
        //    {
        //        KokkosLibrary.Unload();

        //        KokkosActivated = false;
        //    }
        //}
    }
}