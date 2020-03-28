using System;
using System.Runtime.CompilerServices;

namespace Kokkos
{
    public sealed class ScopeGuard : IDisposable
    {
        private readonly bool _sgInit;

        //public ScopeGuard(int      narg,
        //                  string[] arg)
        //{
        //    _sgInit = false;

        //    if(!KokkosLibrary.IsInitialized())
        //    {
        //        KokkosLibrary.Initialize(narg,
        //                          arg);

        //        _sgInit = true;
        //    }
        //}

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public ScopeGuard()
        {
            _sgInit = false;

            //if(KokkosLibrary.IsLoaded() && !KokkosLibrary.IsInitialized())
            //{
                KokkosLibrary.Initialize(Environment.ProcessorCount,
                                         0);

                _sgInit = true;
            //}
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public ScopeGuard(int gpu_device_id)
        {
            _sgInit = false;

            //if(KokkosLibrary.IsLoaded() && !KokkosLibrary.IsInitialized())
            //{
                KokkosLibrary.Initialize(Environment.ProcessorCount,
                                         gpu_device_id);

                _sgInit = true;
            //}
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public ScopeGuard(int num_cpu_threads,
                          int gpu_device_id)
        {
            _sgInit = false;

            //if(KokkosLibrary.IsLoaded() && !KokkosLibrary.IsInitialized())
            //{
                KokkosLibrary.Initialize(num_cpu_threads,
                                         gpu_device_id);

                _sgInit = true;
            //}
        }
        
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public ScopeGuard(in InitArguments arguments)
        {
            _sgInit = false;

            //if(KokkosLibrary.IsLoaded() && !KokkosLibrary.IsInitialized())
            //{
                KokkosLibrary.Initialize(arguments);

                _sgInit = true;
            //}
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public void Dispose()
        {
            ReleaseUnmanagedResources();
            //GC.SuppressFinalize(this);
        }
        
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        ~ScopeGuard()
        {
            ReleaseUnmanagedResources();
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public void Print()
        {
            KokkosLibrary.PrintConfiguration(true);
        }

        private void ReleaseUnmanagedResources()
        {
            if(_sgInit)
            {
                KokkosLibrary.Finalize();
            }
        }
    }
}