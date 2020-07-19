#pragma warning disable CS0465

using System;
using System.IO;
using System.Reflection;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Versioning;
using System.Threading;

using PlatformApi.Win32;

namespace Kokkos
{
    [ComVisible(true)]
    [Serializable]
    public delegate void KokkosLibraryEventHandler(object                 sender,
                                                   KokkosLibraryEventArgs e);

    public enum KokkosLibraryEventKind
    {
        Loaded,
        Unloaded
    }

    [Serializable]
    public sealed class KokkosLibraryEventArgs : EventArgs
    {
        public KokkosLibraryEventKind Type { get; }

        public KokkosLibraryEventArgs(KokkosLibraryEventKind type)
        {
            Type = type;
        }
    }

    [NonVersionable]
    public static class KokkosLibrary
    {
        public static event KokkosLibraryEventHandler Loaded;

        public static event KokkosLibraryEventHandler Unloaded;

        public const string LibraryName = "runtime.Kokkos.NET";

        public static readonly string RuntimeKokkosLibraryName;

        public static IntPtr Handle;
        public static IntPtr ModuleHandle;

        public static KokkosApi Api;

        public static volatile bool Initialized;

        private static readonly string nativeLibraryPath;

#if NETSTANDARD
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
#else
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
#endif
        static KokkosLibrary()
        {
            string operatingSystem      = RuntimeInformation.IsOSPlatform(OSPlatform.Windows) ? "win" : "linux";
            string platformArchitecture = RuntimeInformation.ProcessArchitecture == Architecture.X64 ? "x64" : "x86";

            nativeLibraryPath = $"runtimes\\{operatingSystem}-{platformArchitecture}\\native";

            RuntimeKokkosLibraryName = LibraryName + (RuntimeInformation.ProcessArchitecture == Architecture.X64 ? ".x64" : ".x86");
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static bool IsLoaded()
        {
            return Handle != IntPtr.Zero;
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void Load()
        {
            KokkosCoreLibrary.Load();

#if DEBUG
            Console.WriteLine("Loading " + RuntimeKokkosLibraryName);
#endif

            // if(!NativeLibrary.TryLoad(RuntimeKokkosLibraryName,
            //                           typeof(KokkosLibrary).Assembly,
            //                           DllImportSearchPath.UseDllDirectoryForDependencies,
            //                           out Handle))
            // {
            //     KokkosLibraryException.Throw();
            // }

            Handle = PlatformApi.NativeLibrary.Load(RuntimeKokkosLibraryName, out ulong _);

            ModuleHandle = Kernel32.Native.GetModuleHandle(RuntimeKokkosLibraryName + ".dll");

            IntPtr getApiHandle = PlatformApi.NativeLibrary.GetExport(ModuleHandle, "GetApi", out ulong _);

            if(getApiHandle == IntPtr.Zero)
            {
                getApiHandle = Handle + 0x000019E0;
            }

            if(getApiHandle != IntPtr.Zero) //NativeLibrary.TryGetExport(ModuleHandle, "GetApi", out IntPtr getApiHandle))
            {
                GetApi = Marshal.GetDelegateForFunctionPointer<GetApiDelegate>(getApiHandle);

                Api = GetApi(1);

                Allocate = Marshal.GetDelegateForFunctionPointer<AllocateDelegate>(Api.AllocatePtr);

                Reallocate = Marshal.GetDelegateForFunctionPointer<ReallocateDelegate>(Api.ReallocatePtr);

                Free = Marshal.GetDelegateForFunctionPointer<FreeDelegate>(Api.FreePtr);

                _initialize = Marshal.GetDelegateForFunctionPointer<InitializeDelegate>(Api.InitializePtr);

                _initializeThreads = Marshal.GetDelegateForFunctionPointer<InitializeThreadsDelegate>(Api.InitializeThreadsPtr);

                _initializeArguments = Marshal.GetDelegateForFunctionPointer<InitializeArgumentsDelegate>(Api.InitializeArgumentsPtr);

                _finalize = Marshal.GetDelegateForFunctionPointer<FinalizeDelegate>(Api.FinalizePtr);

                _finalizeAll = Marshal.GetDelegateForFunctionPointer<FinalizeAllDelegate>(Api.FinalizeAllPtr);

                _isInitialized = Marshal.GetDelegateForFunctionPointer<IsInitializedDelegate>(Api.IsInitializedPtr);

                PrintConfiguration = Marshal.GetDelegateForFunctionPointer<PrintConfigurationDelegate>(Api.PrintConfigurationPtr);

                GetComputeCapability = Marshal.GetDelegateForFunctionPointer<CudaGetComputeCapabilityDelegate>(Api.GetComputeCapabilityPtr);

                GetDeviceCount = Marshal.GetDelegateForFunctionPointer<CudaGetDeviceCountDelegate>(Api.GetDeviceCountPtr);

                CreateViewRank0 = Marshal.GetDelegateForFunctionPointer<CreateViewRank0Delegate>(Api.CreateViewRank0Ptr);

                CreateViewRank1 = Marshal.GetDelegateForFunctionPointer<CreateViewRank1Delegate>(Api.CreateViewRank1Ptr);

                CreateViewRank2 = Marshal.GetDelegateForFunctionPointer<CreateViewRank2Delegate>(Api.CreateViewRank2Ptr);

                CreateViewRank3 = Marshal.GetDelegateForFunctionPointer<CreateViewRank3Delegate>(Api.CreateViewRank3Ptr);

                CreateView = Marshal.GetDelegateForFunctionPointer<CreateViewDelegate>(Api.CreateViewPtr);

                GetLabel = Marshal.GetDelegateForFunctionPointer<GetLabelDelegate>(Api.GetLabelPtr);

                GetSize = Marshal.GetDelegateForFunctionPointer<GetSizeDelegate>(Api.GetSizePtr);

                GetStride = Marshal.GetDelegateForFunctionPointer<GetStrideDelegate>(Api.GetStridePtr);

                GetExtent = Marshal.GetDelegateForFunctionPointer<GetExtentDelegate>(Api.GetExtentPtr);

                CopyTo = Marshal.GetDelegateForFunctionPointer<CopyToDelegate>(Api.CopyToPtr);

                GetValue = Marshal.GetDelegateForFunctionPointer<GetValueDelegate>(Api.GetValuePtr);

                SetValue = Marshal.GetDelegateForFunctionPointer<SetValueDelegate>(Api.SetValuePtr);

                RcpViewToNdArray = Marshal.GetDelegateForFunctionPointer<RcpViewToNdArrayDelegate>(Api.RcpViewToNdArrayPtr);

                ViewToNdArray = Marshal.GetDelegateForFunctionPointer<ViewToNdArrayDelegate>(Api.ViewToNdArrayPtr);
                
                Shepard2dSingle = Marshal.GetDelegateForFunctionPointer<Shepard2dSingleDelegate>(PlatformApi.NativeLibrary.GetExport(ModuleHandle, "Shepard2dSingle", out ulong _));

                Shepard2dDouble = Marshal.GetDelegateForFunctionPointer<Shepard2dDoubleDelegate>(PlatformApi.NativeLibrary.GetExport(ModuleHandle, "Shepard2dDouble", out ulong _));
            }
            else
            {
                KokkosLibraryException.Throw("'runtime.Kokkos.NET::GetApi' not found.");
            }

#if DEBUG
            Console.WriteLine("Loaded " + RuntimeKokkosLibraryName + $"@ 0x{Handle.ToString("X")}");
#endif

            OnLoaded();
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void Unload()
        {
            if(!PlatformApi.NativeLibrary.Free(ModuleHandle, out ulong _))
            {
                KokkosLibraryException.Throw(RuntimeKokkosLibraryName + "failed to unload.");
            }
            else
            {
                Handle       = IntPtr.Zero;
                ModuleHandle = IntPtr.Zero;
            }

            //KokkosCoreLibrary.Unload();

            //Kernel32.UnmapViewOfFile(Kernel32.GetModuleHandleA(RuntimeKokkosLibraryName + ".dll"));
            //Kernel32.UnmapViewOfFile(Kernel32.GetModuleHandleA(KokkosCoreLibrary.KokkosCoreLibraryName));

            OnUnloaded();
        }

        #region Delegates

        public delegate ref KokkosApi GetApiDelegate(in uint version);

        public delegate IntPtr AllocateDelegate(in ExecutionSpaceKind execution_space,
                                                in ulong              arg_alloc_size);

        public delegate IntPtr ReallocateDelegate(in ExecutionSpaceKind execution_space,
                                                  IntPtr                instance,
                                                  in ulong              arg_alloc_size);

        public delegate void FreeDelegate(in ExecutionSpaceKind execution_space,
                                          IntPtr                instance);

        public delegate void InitializeDelegate(int                                                                             narg,
                                                [MarshalAs(UnmanagedType.LPArray, ArraySubType = UnmanagedType.LPStr)] string[] arg);

        public delegate void InitializeThreadsDelegate(int num_cpu_threads,
                                                       int gpu_device_id);

        public delegate void InitializeArgumentsDelegate(in InitArguments arguments);

        public delegate void FinalizeDelegate();

        public delegate void FinalizeAllDelegate();

        public delegate bool IsInitializedDelegate();

        public delegate void PrintConfigurationDelegate(bool detail);

        public delegate uint CudaGetDeviceCountDelegate();

        public delegate uint CudaGetComputeCapabilityDelegate(uint device_id);

        public delegate void CreateViewRank0Delegate(IntPtr      instance,
                                                     ref NdArray nArray);

        public delegate void CreateViewRank1Delegate(IntPtr      instance,
                                                     ref NdArray nArray,
                                                     in  ulong   n0);

        public delegate void CreateViewRank2Delegate(IntPtr      instance,
                                                     ref NdArray nArray,
                                                     in  ulong   n0,
                                                     in  ulong   n1);

        public delegate void CreateViewRank3Delegate(IntPtr      instance,
                                                     ref NdArray nArray,
                                                     in  ulong   n0,
                                                     in  ulong   n1,
                                                     in  ulong   n2);

        public delegate void CreateViewDelegate(IntPtr      instance,
                                                ref NdArray nArray);

        public delegate NativeString GetLabelDelegate(IntPtr     instance,
                                                      in NdArray nArray);

        public delegate ulong GetSizeDelegate(IntPtr     instance,
                                              in NdArray nArray);

        public delegate ulong GetStrideDelegate(IntPtr     instance,
                                                in NdArray nArray,
                                                in uint    dim);

        public delegate ulong GetExtentDelegate(IntPtr     instance,
                                                in NdArray nArray,
                                                in uint    dim);

        public delegate void CopyToDelegate(IntPtr      instance,
                                            in NdArray  nArray,
                                            ValueType[] values);

        public delegate ValueType GetValueDelegate(IntPtr     instance,
                                                   in NdArray nArray,
                                                   in ulong   i0 = ulong.MaxValue,
                                                   in ulong   i1 = ulong.MaxValue,
                                                   in ulong   i2 = ulong.MaxValue,
                                                   in ulong   i4 = ulong.MaxValue,
                                                   in ulong   i5 = ulong.MaxValue,
                                                   in ulong   i6 = ulong.MaxValue,
                                                   in ulong   i7 = ulong.MaxValue,
                                                   in ulong   i8 = ulong.MaxValue);

        public delegate void SetValueDelegate(IntPtr       instance,
                                              in NdArray   nArray,
                                              in ValueType value,
                                              in ulong     i0 = ulong.MaxValue,
                                              in ulong     i1 = ulong.MaxValue,
                                              in ulong     i2 = ulong.MaxValue,
                                              in ulong     i4 = ulong.MaxValue,
                                              in ulong     i5 = ulong.MaxValue,
                                              in ulong     i6 = ulong.MaxValue,
                                              in ulong     i7 = ulong.MaxValue,
                                              in ulong     i8 = ulong.MaxValue);

        public delegate NdArray RcpViewToNdArrayDelegate(IntPtr                instance,
                                                         in ExecutionSpaceKind execution_space,
                                                         in LayoutKind         layout,
                                                         in DataTypeKind       data_type,
                                                         in ushort             rank);

        public delegate NdArray ViewToNdArrayDelegate(IntPtr                instance,
                                                      in ExecutionSpaceKind execution_space,
                                                      in LayoutKind         layout,
                                                      in DataTypeKind       data_type,
                                                      in ushort             rank);

        public delegate IntPtr Shepard2dSingleDelegate(IntPtr                xd_rcp_view_ptr,
                                                       IntPtr                zd_rcp_view_ptr,
                                                       in float              p,
                                                       IntPtr                xi_rcp_view_ptr,
                                                       in ExecutionSpaceKind execution_space);

        public delegate IntPtr Shepard2dDoubleDelegate(IntPtr                xd_rcp_view_ptr,
                                                       IntPtr                zd_rcp_view_ptr,
                                                       in double              p,
                                                       IntPtr                xi_rcp_view_ptr,
                                                       in ExecutionSpaceKind execution_space);

        #endregion

        #region Calli

        // public static void CalliInitialize(int      narg,
        //                                   string[] arg,
        //                                   IntPtr   funcPtr)
        //{
        //    throw new NotImplementedException();
        //}

        // public static void CalliInitializeThreads(int    num_cpu_threads,
        //                                          int    gpu_device_id,
        //                                          IntPtr funcPtr)
        //{
        //    throw new NotImplementedException();
        //}

        // public static void CalliInitializeArguments(in InitArguments arguments,
        //                                            IntPtr           funcPtr)
        //{
        //    throw new NotImplementedException();
        //}

        // public static void CalliFinalize(IntPtr funcPtr)
        //{
        //    throw new NotImplementedException();
        //}

        // public static void CalliFinalizeAll(IntPtr funcPtr)
        //{
        //    throw new NotImplementedException();
        //}

        // public static bool CalliIsInitialized(IntPtr funcPtr)
        //{
        //    throw new NotImplementedException();
        //}

        // public static void CalliPrintConfiguration(bool   detail,
        //                                           IntPtr funcPtr)
        //{
        //    throw new NotImplementedException();
        //}

        // public static uint CalliCudaGetDeviceCount(IntPtr funcPtr)
        //{
        //    throw new NotImplementedException();
        //}

        // public static uint CalliCudaGetComputeCapability(uint   device_id,
        //                                                 IntPtr funcPtr)
        //{
        //    throw new NotImplementedException();
        //}

        // public static void CalliCreateViewRank0(IntPtr                instance,
        //                                        in DataTypeKind       data_type,
        //                                        in ExecutionSpaceKind
        //                                        execution_space, byte[] label,
        //                                        IntPtr                funcPtr)
        //{
        //    throw new NotImplementedException();
        //}

        // public static void CalliCreateViewRank1(IntPtr                instance,
        //                                        in DataTypeKind       data_type,
        //                                        in ExecutionSpaceKind
        //                                        execution_space, byte[] label, in
        //                                        ulong              n0, IntPtr
        //                                        funcPtr)
        //{
        //    throw new NotImplementedException();
        //}

        // public static void CalliCreateViewRank2(IntPtr                instance,
        //                                        in DataTypeKind       data_type,
        //                                        in ExecutionSpaceKind
        //                                        execution_space, byte[] label, in
        //                                        ulong              n0, in ulong n1,
        //                                        IntPtr                funcPtr)
        //{
        //    throw new NotImplementedException();
        //}

        // public static void CalliCreateViewRank3(IntPtr                instance,
        //                                        in DataTypeKind       data_type,
        //                                        in ExecutionSpaceKind
        //                                        execution_space, byte[] label, in
        //                                        ulong              n0, in ulong n1,
        //                                        in ulong              n2,
        //                                        IntPtr                funcPtr)
        //{
        //    throw new NotImplementedException();
        //}

        // public static byte[] CalliGetLabel(IntPtr                instance,
        //                                   in DataTypeKind       data_type,
        //                                   in ExecutionSpaceKind execution_space,
        //                                   in uint               rank,
        //                                   IntPtr                funcPtr)
        //{
        //    throw new NotImplementedException();
        //}

        // public static ulong CalliGetSize(IntPtr                instance,
        //                                 in DataTypeKind       data_type,
        //                                 in ExecutionSpaceKind execution_space,
        //                                 in uint               rank,
        //                                 IntPtr                funcPtr)
        //{
        //    throw new NotImplementedException();
        //}

        // public static ulong CalliGetStride(IntPtr                instance,
        //                                   in DataTypeKind       data_type,
        //                                   in ExecutionSpaceKind execution_space,
        //                                   in uint               rank,
        //                                   in uint               dim,
        //                                   IntPtr                funcPtr)
        //{
        //    throw new NotImplementedException();
        //}

        // public static ulong CalliGetExtent(IntPtr                instance,
        //                                   in DataTypeKind       data_type,
        //                                   in ExecutionSpaceKind execution_space,
        //                                   in uint               rank,
        //                                   in uint               dim,
        //                                   IntPtr                funcPtr)
        //{
        //    throw new NotImplementedException();
        //}

        #endregion

        #region Methods

        public static GetApiDelegate GetApi;

        public static AllocateDelegate Allocate;

        public static ReallocateDelegate Reallocate;

        public static FreeDelegate Free;

        private static InitializeDelegate _initialize;

#if NETSTANDARD
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
#else
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
#endif
        public static void Initialize(int      narg,
                                      string[] arg)
        {
            if(!IsInitialized())
            {
                Load();

                _initialize(narg, arg);

                Initialized = true;
            }
        }

        private static InitializeThreadsDelegate _initializeThreads;

#if NETSTANDARD
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
#else
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
#endif
        public static void Initialize(int num_cpu_threads,
                                      int gpu_device_id)
        {
            if(!IsInitialized())
            {
                Load();

                _initializeThreads(num_cpu_threads, gpu_device_id);

                Initialized = true;
            }
        }

        private static InitializeArgumentsDelegate _initializeArguments;

#if NETSTANDARD
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
#else
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
#endif
        public static void Initialize(in InitArguments arguments)
        {
            Load();

            _initializeArguments(arguments);

            Initialized = true;
        }

        private static FinalizeDelegate _finalize;

#if NETSTANDARD
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
#else
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
#endif
        public static void @Finalize()
        {
            _finalize();

            //Unload();

            Initialized = false;
        }

        private static FinalizeAllDelegate _finalizeAll;

#if NETSTANDARD
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
#else
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
#endif
        public static void FinalizeAll()
        {
            _finalizeAll();

            //Unload();

            Initialized = false;
        }

        private static IsInitializedDelegate _isInitialized;

#if NETSTANDARD
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
#else
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
#endif
        public static bool IsInitialized()
        {
            if(!Initialized)
            {
                return false;
            }

            return _isInitialized();
        }

        public static PrintConfigurationDelegate PrintConfiguration;

        public static CudaGetComputeCapabilityDelegate GetComputeCapability;

        public static CudaGetDeviceCountDelegate GetDeviceCount;

        public static CreateViewRank0Delegate CreateViewRank0;

        public static CreateViewRank1Delegate CreateViewRank1;

        public static CreateViewRank2Delegate CreateViewRank2;

        public static CreateViewRank3Delegate CreateViewRank3;

        public static CreateViewDelegate CreateView;

        public static GetLabelDelegate GetLabel;

        public static GetSizeDelegate GetSize;

        public static GetStrideDelegate GetStride;

        public static GetExtentDelegate GetExtent;

        public static CopyToDelegate CopyTo;

        public static GetValueDelegate GetValue;

        public static SetValueDelegate SetValue;

        public static RcpViewToNdArrayDelegate RcpViewToNdArray;

        public static ViewToNdArrayDelegate ViewToNdArray;

        public static Shepard2dSingleDelegate Shepard2dSingle;

        public static Shepard2dDoubleDelegate Shepard2dDouble;

        #endregion

        private static readonly KokkosLibraryEventArgs loadedEventArgs   = new KokkosLibraryEventArgs(KokkosLibraryEventKind.Loaded);
        private static readonly KokkosLibraryEventArgs unloadedEventArgs = new KokkosLibraryEventArgs(KokkosLibraryEventKind.Unloaded);

#if NETSTANDARD
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
#else
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
#endif
        private static void OnLoaded()
        {
            Loaded?.Invoke(null, loadedEventArgs);

            Console.WriteLine("KokkosLibrary Loaded.");
        }

#if NETSTANDARD
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
#else
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
#endif
        private static void OnUnloaded()
        {
            Unloaded?.Invoke(null, unloadedEventArgs);

            Console.WriteLine("KokkosLibrary Unloaded.");
        }
    }
}
