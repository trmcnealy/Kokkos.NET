using System;
using System.Runtime.CompilerServices;
using System.Threading;

namespace Kokkos
{
    public static class ParallelProcessor
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        private static void RuntimeTest()
        {
            //if(Thread.CurrentThread.GetApartmentState() != ApartmentState.STA || Thread.CurrentThread.IsBackground || Thread.CurrentThread.IsThreadPoolThread || !Thread.CurrentThread.IsAlive)
            //{
            //    KokkosLibraryException.Throw("Kokkos Library must be initialized on the main (STA) thread.");
            //}

            if (KokkosLibrary.IsLoaded())
            {
                KokkosLibraryException.Throw("Kokkos Library has not been initialized.");
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static void Initialize(int numCpuThreads = -1,
                                      int gpuDeviceId   = 0)
        {
            KokkosLibrary.Initialize(numCpuThreads == -1 ? Environment.ProcessorCount : numCpuThreads,
                                     gpuDeviceId);

            //AppDomain.CurrentDomain.ProcessExit += View_Dtor;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static void Initialize(InitArguments arguments)
        {
            KokkosLibrary.Initialize(arguments);

            //AppDomain.CurrentDomain.ProcessExit += View_Dtor;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static void Shutdown()
        {
            KokkosLibrary.Finalize();
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static bool IsRunning()
        {
            if(KokkosLibrary.IsLoaded())
            {
                return KokkosLibrary.IsInitialized();
            }

            return false;
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