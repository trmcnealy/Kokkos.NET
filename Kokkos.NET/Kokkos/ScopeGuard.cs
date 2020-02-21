using System;
using System.Runtime.CompilerServices;

namespace Kokkos
{
    public sealed class ScopeGuard : IDisposable
    {
        public bool SgInit;

        //public ScopeGuard(int      narg,
        //                  string[] arg)
        //{
        //    SgInit = false;

        //    if(!KokkosLibrary.IsInitialized())
        //    {
        //        KokkosLibrary.Initialize(narg,
        //                          arg);

        //        SgInit = true;
        //    }
        //}

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public ScopeGuard(int num_cpu_threads = 16,
                          int gpu_device_id   = 0)
        {
            //SgInit = false;

            //if(!KokkosLibrary.IsInitialized())
            //{
            KokkosLibrary.Initialize(num_cpu_threads,
                                     gpu_device_id);

            //    SgInit = true;
            //}
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public void Dispose()
        {
            KokkosLibrary.Finalize();
            //ReleaseUnmanagedResources();
            //GC.SuppressFinalize(this);
        }

        ~ScopeGuard()
        {
            KokkosLibrary.Finalize();
            //ReleaseUnmanagedResources();
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public void Print()
        {
            KokkosLibrary.PrintConfiguration(true);
        }

        //private void ReleaseUnmanagedResources()
        //{
        //    if(Kokkos.IsInitialized() && SgInit)
        //    {
        //        Kokkos.Finalize();
        //    }
        //}
    }
}