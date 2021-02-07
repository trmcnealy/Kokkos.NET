#pragma warning disable CS0465

using System;
using System.IO;
using System.Reflection;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Versioning;
using System.Security;

using PlatformApi.Win32;

namespace Kokkos
{
    [ComVisible(true)]
    [Serializable]
    public delegate void KokkosLibraryEventHandler(object?                sender,
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
        public const string LibraryName = "runtime.Kokkos.NET";

        public static readonly string RuntimeKokkosLibraryName;

        public static nint Handle;
        public static nint ModuleHandle;

        public static KokkosApi Api;

        public static volatile bool Initialized;

        public static volatile bool IsLoaded;

        private static readonly string nativeLibraryPath;

        private static readonly KokkosLibraryEventArgs loadedEventArgs   = new KokkosLibraryEventArgs(KokkosLibraryEventKind.Loaded);
        private static readonly KokkosLibraryEventArgs unloadedEventArgs = new KokkosLibraryEventArgs(KokkosLibraryEventKind.Unloaded);

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        static KokkosLibrary()
        {
            string operatingSystem      = RuntimeInformation.IsOSPlatform(OSPlatform.Windows) ? "win" : "linux";
            string platformArchitecture = RuntimeInformation.ProcessArchitecture == Architecture.X64 ? "x64" : "x86";

            nativeLibraryPath = $"runtimes\\{operatingSystem}-{platformArchitecture}\\native";

            RuntimeKokkosLibraryName = LibraryName + (RuntimeInformation.ProcessArchitecture == Architecture.X64 ? ".x64" : ".x86");
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static unsafe void Load()
        {
            if(!IsLoaded)
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

                PlatformApi.NativeLibrary.LoadLibrary("nvcuda");

                Handle = PlatformApi.NativeLibrary.Load(RuntimeKokkosLibraryName);

                ModuleHandle = Kernel32.Native.GetModuleHandle(RuntimeKokkosLibraryName + ".dll");

                nint getApiHandle = PlatformApi.NativeLibrary.GetExport(ModuleHandle, "GetApi");

                if(getApiHandle == 0)
                {
                    getApiHandle = Handle + 0x000019E0;
                }

                if(getApiHandle == 0)
                {
                    KokkosLibraryException.Throw("'runtime.Kokkos.NET::GetApi' not found.");
                }

                if(getApiHandle != 0) //NativeLibrary.TryGetExport(ModuleHandle, "GetApi", out nint getApiHandle))
                {
                    GetApi = Marshal.GetDelegateForFunctionPointer<GetApiDelegate>(getApiHandle);

                    Api = GetApi(1);

                    Allocate = Marshal.GetDelegateForFunctionPointer<AllocateDelegate>(Api.AllocatePtr);

                    Reallocate = Marshal.GetDelegateForFunctionPointer<ReallocateDelegate>(Api.ReallocatePtr);

                    _free = Marshal.GetDelegateForFunctionPointer<FreeDelegate>(Api.FreePtr);

                    _initialize = Marshal.GetDelegateForFunctionPointer<InitializeDelegate>(Api.InitializePtr);

                    InitializeSerial = Marshal.GetDelegateForFunctionPointer<InitializeSerialDelegate>(Api.InitializeSerialPtr);

                    InitializeOpenMP = Marshal.GetDelegateForFunctionPointer<InitializeOpenMPDelegate>(Api.InitializeOpenMPPtr);

                    InitializeCuda = Marshal.GetDelegateForFunctionPointer<InitializeCudaDelegate>(Api.InitializeCudaPtr);

                    _initializeThreads = Marshal.GetDelegateForFunctionPointer<InitializeThreadsDelegate>(Api.InitializeThreadsPtr);

                    _initializeArguments = Marshal.GetDelegateForFunctionPointer<InitializeArgumentsDelegate>(Api.InitializeArgumentsPtr);

                    _finalize = Marshal.GetDelegateForFunctionPointer<FinalizeDelegate>(Api.FinalizePtr);

                    FinalizeSerial = Marshal.GetDelegateForFunctionPointer<FinalizeSerialDelegate>(Api.FinalizeSerialPtr);

                    FinalizeOpenMP = Marshal.GetDelegateForFunctionPointer<FinalizeOpenMPDelegate>(Api.FinalizeOpenMPPtr);

                    FinalizeCuda = Marshal.GetDelegateForFunctionPointer<FinalizeCudaDelegate>(Api.FinalizeCudaPtr);

                    _finalizeAll = Marshal.GetDelegateForFunctionPointer<FinalizeAllDelegate>(Api.FinalizeAllPtr);

                    _isInitialized = Marshal.GetDelegateForFunctionPointer<IsInitializedDelegate>(Api.IsInitializedPtr);

                    PrintConfiguration = Marshal.GetDelegateForFunctionPointer<PrintConfigurationDelegate>(Api.PrintConfigurationPtr);

                    GetComputeCapability = Marshal.GetDelegateForFunctionPointer<CudaGetComputeCapabilityDelegate>(Api.GetComputeCapabilityPtr);

                    GetDeviceCount = Marshal.GetDelegateForFunctionPointer<CudaGetDeviceCountDelegate>(Api.GetDeviceCountPtr);

                    CreateViewRank0 = Marshal.GetDelegateForFunctionPointer<CreateViewRank0Delegate>(Api.CreateViewRank0Ptr);

                    CreateViewRank1 = Marshal.GetDelegateForFunctionPointer<CreateViewRank1Delegate>(Api.CreateViewRank1Ptr);

                    CreateViewRank2 = Marshal.GetDelegateForFunctionPointer<CreateViewRank2Delegate>(Api.CreateViewRank2Ptr);

                    CreateViewRank3 = Marshal.GetDelegateForFunctionPointer<CreateViewRank3Delegate>(Api.CreateViewRank3Ptr);

                    CreateViewRank4 = Marshal.GetDelegateForFunctionPointer<CreateViewRank4Delegate>(Api.CreateViewRank4Ptr);

                    CreateViewRank5 = Marshal.GetDelegateForFunctionPointer<CreateViewRank5Delegate>(Api.CreateViewRank5Ptr);

                    CreateViewRank6 = Marshal.GetDelegateForFunctionPointer<CreateViewRank6Delegate>(Api.CreateViewRank6Ptr);

                    CreateViewRank7 = Marshal.GetDelegateForFunctionPointer<CreateViewRank7Delegate>(Api.CreateViewRank7Ptr);

                    CreateViewRank8 = Marshal.GetDelegateForFunctionPointer<CreateViewRank8Delegate>(Api.CreateViewRank8Ptr);

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
                }

                GetNumaCount = Marshal.GetDelegateForFunctionPointer<GetNumaCountDelegate>(PlatformApi.NativeLibrary.GetExport(ModuleHandle, "GetNumaCount"));

                GetCoresPerNuma = Marshal.GetDelegateForFunctionPointer<GetCoresPerNumaDelegate>(PlatformApi.NativeLibrary.GetExport(ModuleHandle, "GetCoresPerNuma"));

                GetThreadsPerCore = Marshal.GetDelegateForFunctionPointer<GetThreadsPerCoreDelegate>(PlatformApi.NativeLibrary.GetExport(ModuleHandle, "GetThreadsPerCore"));

                Shepard2dSingle = Marshal.GetDelegateForFunctionPointer<Shepard2dSingleDelegate>(PlatformApi.NativeLibrary.GetExport(ModuleHandle, "Shepard2dSingle"));

                Shepard2dDouble = Marshal.GetDelegateForFunctionPointer<Shepard2dDoubleDelegate>(PlatformApi.NativeLibrary.GetExport(ModuleHandle, "Shepard2dDouble"));

                NearestNeighborSingle = Marshal.GetDelegateForFunctionPointer<NearestNeighborSingleDelegate>(PlatformApi.NativeLibrary.GetExport(ModuleHandle, "NearestNeighborSingle"));

                NearestNeighborDouble = Marshal.GetDelegateForFunctionPointer<NearestNeighborDoubleDelegate>(PlatformApi.NativeLibrary.GetExport(ModuleHandle, "NearestNeighborDouble"));

                CountLineEndingsSerial = Marshal.GetDelegateForFunctionPointer<CountLineEndingsSerialDelegate>(PlatformApi.NativeLibrary.GetExport(ModuleHandle, "CountLineEndingsSerial"));

                CountLineEndingsOpenMP = Marshal.GetDelegateForFunctionPointer<CountLineEndingsOpenMPDelegate>(PlatformApi.NativeLibrary.GetExport(ModuleHandle, "CountLineEndingsOpenMP"));

                CountLineEndingsCuda = Marshal.GetDelegateForFunctionPointer<CountLineEndingsCudaDelegate>(PlatformApi.NativeLibrary.GetExport(ModuleHandle, "CountLineEndingsCuda"));

                IpcCreate           = Marshal.GetDelegateForFunctionPointer<IpcCreateDelegate>(PlatformApi.NativeLibrary.GetExport(ModuleHandle,           "IpcCreate"));
                IpcCreateFrom       = Marshal.GetDelegateForFunctionPointer<IpcCreateFromDelegate>(PlatformApi.NativeLibrary.GetExport(ModuleHandle,       "IpcCreateFrom"));
                IpcOpenExisting     = Marshal.GetDelegateForFunctionPointer<IpcOpenExistingDelegate>(PlatformApi.NativeLibrary.GetExport(ModuleHandle,     "IpcOpenExisting"));
                IpcDestory          = Marshal.GetDelegateForFunctionPointer<IpcDestoryDelegate>(PlatformApi.NativeLibrary.GetExport(ModuleHandle,          "IpcDestory"));
                IpcClose            = Marshal.GetDelegateForFunctionPointer<IpcCloseDelegate>(PlatformApi.NativeLibrary.GetExport(ModuleHandle,            "IpcClose"));
                IpcGetMemoryPointer = Marshal.GetDelegateForFunctionPointer<IpcGetMemoryPointerDelegate>(PlatformApi.NativeLibrary.GetExport(ModuleHandle, "IpcGetMemoryPointer"));
                IpcGetDeviceHandle  = Marshal.GetDelegateForFunctionPointer<IpcGetDeviceHandleDelegate>(PlatformApi.NativeLibrary.GetExport(ModuleHandle,  "IpcGetDeviceHandle"));
                IpcGetSize          = Marshal.GetDelegateForFunctionPointer<IpcGetSizeDelegate>(PlatformApi.NativeLibrary.GetExport(ModuleHandle,          "IpcGetSize"));

                IpcMakeViewFromPointer = Marshal.GetDelegateForFunctionPointer<IpcMakeViewFromPointerDelegate>(PlatformApi.NativeLibrary.GetExport(ModuleHandle, "IpcMakeViewFromPointer"));

                IpcMakeViewFromHandle = Marshal.GetDelegateForFunctionPointer<IpcMakeViewFromHandleDelegate>(PlatformApi.NativeLibrary.GetExport(ModuleHandle, "IpcMakeViewFromHandle"));

#if DEBUG
                Console.WriteLine("Loaded " + RuntimeKokkosLibraryName + $"@ 0x{Handle.ToString("X")}");
#endif

                IsLoaded = true;
            }
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void Unload()
        {
            if(!PlatformApi.NativeLibrary.Free(ModuleHandle))
            {
                KokkosLibraryException.Throw(RuntimeKokkosLibraryName + "failed to unload.");
            }
            else
            {
                Handle       = (nint)0;
                ModuleHandle = (nint)0;
            }

            //KokkosCoreLibrary.Unload();

            //Kernel32.UnmapViewOfFile(Kernel32.GetModuleHandleA(RuntimeKokkosLibraryName + ".dll"));
            //Kernel32.UnmapViewOfFile(Kernel32.GetModuleHandleA(KokkosCoreLibrary.KokkosCoreLibraryName));

            IsLoaded = false;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static ulong CalculateRank(ulong arg_N0 = ulong.MaxValue,
                                          ulong arg_N1 = ulong.MaxValue,
                                          ulong arg_N2 = ulong.MaxValue,
                                          ulong arg_N3 = ulong.MaxValue,
                                          ulong arg_N4 = ulong.MaxValue,
                                          ulong arg_N5 = ulong.MaxValue,
                                          ulong arg_N6 = ulong.MaxValue,
                                          ulong arg_N7 = ulong.MaxValue)

        {
            if(arg_N0 == ulong.MaxValue)
            {
                return 0;
            }

            if(arg_N1 == ulong.MaxValue)
            {
                return 1;
            }

            if(arg_N2 == ulong.MaxValue)
            {
                return 2;
            }

            if(arg_N3 == ulong.MaxValue)
            {
                return 3;
            }

            if(arg_N4 == ulong.MaxValue)
            {
                return 4;
            }

            if(arg_N5 == ulong.MaxValue)
            {
                return 5;
            }

            if(arg_N6 == ulong.MaxValue)
            {
                return 6;
            }

            if(arg_N7 == ulong.MaxValue)
            {
                return 7;
            }

            return 8;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static ulong CalculateSize(ulong arg_N0 = 0ul,
                                          ulong arg_N1 = 0ul,
                                          ulong arg_N2 = 0ul,
                                          ulong arg_N3 = 0ul,
                                          ulong arg_N4 = 0ul,
                                          ulong arg_N5 = 0ul,
                                          ulong arg_N6 = 0ul,
                                          ulong arg_N7 = 0ul)

        {
            return (arg_N0 == 0ul ? 1ul : arg_N0) *
                   (arg_N1 == 0ul ? 1ul : arg_N1) *
                   (arg_N2 == 0ul ? 1ul : arg_N2) *
                   (arg_N3 == 0ul ? 1ul : arg_N3) *
                   (arg_N4 == 0ul ? 1ul : arg_N4) *
                   (arg_N5 == 0ul ? 1ul : arg_N5) *
                   (arg_N6 == 0ul ? 1ul : arg_N6) *
                   (arg_N7 == 0ul ? 1ul : arg_N7);
        }

        #region Delegates

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        [SuppressUnmanagedCodeSecurity]
        public delegate ref KokkosApi GetApiDelegate(uint version);

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        [SuppressUnmanagedCodeSecurity]
        public delegate nint AllocateDelegate(ExecutionSpaceKind execution_space,
                                              ulong              arg_alloc_size);

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        [SuppressUnmanagedCodeSecurity]
        public delegate nint ReallocateDelegate(ExecutionSpaceKind execution_space,
                                                nint               instance,
                                                ulong              arg_alloc_size);

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        [SuppressUnmanagedCodeSecurity]
        public delegate void FreeDelegate(ExecutionSpaceKind execution_space,
                                          nint               instance);

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        [SuppressUnmanagedCodeSecurity]
        public delegate void InitializeDelegate(int                                                                             narg,
                                                [MarshalAs(UnmanagedType.LPArray, ArraySubType = UnmanagedType.LPStr)] string[] arg);

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        [SuppressUnmanagedCodeSecurity]
        public delegate void InitializeSerialDelegate();

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        [SuppressUnmanagedCodeSecurity]
        public delegate void InitializeOpenMPDelegate(int num_threads);

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        [SuppressUnmanagedCodeSecurity]
        public delegate void InitializeCudaDelegate(int use_gpu);

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        [SuppressUnmanagedCodeSecurity]
        public delegate void InitializeThreadsDelegate(int num_cpu_threads,
                                                       int gpu_device_id);

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        [SuppressUnmanagedCodeSecurity]
        public delegate void InitializeArgumentsDelegate(InitArguments arguments);

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        [SuppressUnmanagedCodeSecurity]
        public delegate void FinalizeDelegate();

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        [SuppressUnmanagedCodeSecurity]
        public delegate void FinalizeSerialDelegate();

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        [SuppressUnmanagedCodeSecurity]
        public delegate void FinalizeOpenMPDelegate();

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        [SuppressUnmanagedCodeSecurity]
        public delegate void FinalizeCudaDelegate();

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        [SuppressUnmanagedCodeSecurity]
        public delegate void FinalizeAllDelegate();

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        [SuppressUnmanagedCodeSecurity]
        public delegate bool IsInitializedDelegate();

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        [SuppressUnmanagedCodeSecurity]
        public delegate void PrintConfigurationDelegate(bool detail);

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        [SuppressUnmanagedCodeSecurity]
        public delegate uint CudaGetDeviceCountDelegate();

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        [SuppressUnmanagedCodeSecurity]
        public delegate uint CudaGetComputeCapabilityDelegate(uint device_id);

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        [SuppressUnmanagedCodeSecurity]
        public delegate void CreateViewRank0Delegate(nint        instance,
                                                     ref NdArray nArray);

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        [SuppressUnmanagedCodeSecurity]
        public delegate void CreateViewRank1Delegate(nint        instance,
                                                     ref NdArray nArray,
                                                     ulong       n0);

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        [SuppressUnmanagedCodeSecurity]
        public delegate void CreateViewRank2Delegate(nint        instance,
                                                     ref NdArray nArray,
                                                     ulong       n0,
                                                     ulong       n1);

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        [SuppressUnmanagedCodeSecurity]
        public delegate void CreateViewRank3Delegate(nint        instance,
                                                     ref NdArray nArray,
                                                     ulong       n0,
                                                     ulong       n1,
                                                     ulong       n2);

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        [SuppressUnmanagedCodeSecurity]
        public delegate void CreateViewRank4Delegate(nint        instance,
                                                     ref NdArray nArray,
                                                     ulong       n0,
                                                     ulong       n1,
                                                     ulong       n2,
                                                     ulong       n3);

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        [SuppressUnmanagedCodeSecurity]
        public delegate void CreateViewRank5Delegate(nint        instance,
                                                     ref NdArray nArray,
                                                     ulong       n0,
                                                     ulong       n1,
                                                     ulong       n2,
                                                     ulong       n3,
                                                     ulong       n4);

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        [SuppressUnmanagedCodeSecurity]
        public delegate void CreateViewRank6Delegate(nint        instance,
                                                     ref NdArray nArray,
                                                     ulong       n0,
                                                     ulong       n1,
                                                     ulong       n2,
                                                     ulong       n3,
                                                     ulong       n4,
                                                     ulong       n5);

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        [SuppressUnmanagedCodeSecurity]
        public delegate void CreateViewRank7Delegate(nint        instance,
                                                     ref NdArray nArray,
                                                     ulong       n0,
                                                     ulong       n1,
                                                     ulong       n2,
                                                     ulong       n3,
                                                     ulong       n4,
                                                     ulong       n5,
                                                     ulong       n6);

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        [SuppressUnmanagedCodeSecurity]
        public delegate void CreateViewRank8Delegate(nint        instance,
                                                     ref NdArray nArray,
                                                     ulong       n0,
                                                     ulong       n1,
                                                     ulong       n2,
                                                     ulong       n3,
                                                     ulong       n4,
                                                     ulong       n5,
                                                     ulong       n6,
                                                     ulong       n7);

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        [SuppressUnmanagedCodeSecurity]
        public delegate void CreateViewDelegate(nint        instance,
                                                ref NdArray nArray);

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        [SuppressUnmanagedCodeSecurity]
        public delegate NativeString<Serial> GetLabelDelegate(nint    instance,
                                                              NdArray nArray);

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        [SuppressUnmanagedCodeSecurity]
        public delegate ulong GetSizeDelegate(nint    instance,
                                              NdArray nArray);

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        [SuppressUnmanagedCodeSecurity]
        public delegate ulong GetStrideDelegate(nint    instance,
                                                NdArray nArray,
                                                uint    dim);

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        [SuppressUnmanagedCodeSecurity]
        public delegate ulong GetExtentDelegate(nint    instance,
                                                NdArray nArray,
                                                uint    dim);

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        [SuppressUnmanagedCodeSecurity]
        public delegate void CopyToDelegate(nint        instance,
                                            NdArray     nArray,
                                            ValueType[] values);

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        [SuppressUnmanagedCodeSecurity]
        public delegate ValueType GetValueDelegate(nint    instance,
                                                   NdArray nArray,
                                                   ulong   i0 = ulong.MaxValue,
                                                   ulong   i1 = ulong.MaxValue,
                                                   ulong   i2 = ulong.MaxValue,
                                                   ulong   i3 = ulong.MaxValue,
                                                   ulong   i4 = ulong.MaxValue,
                                                   ulong   i5 = ulong.MaxValue,
                                                   ulong   i6 = ulong.MaxValue,
                                                   ulong   i7 = ulong.MaxValue);

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        [SuppressUnmanagedCodeSecurity]
        public delegate void SetValueDelegate(nint      instance,
                                              NdArray   nArray,
                                              ValueType value,
                                              ulong     i0 = ulong.MaxValue,
                                              ulong     i1 = ulong.MaxValue,
                                              ulong     i2 = ulong.MaxValue,
                                              ulong     i3 = ulong.MaxValue,
                                              ulong     i4 = ulong.MaxValue,
                                              ulong     i5 = ulong.MaxValue,
                                              ulong     i6 = ulong.MaxValue,
                                              ulong     i7 = ulong.MaxValue);

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        [SuppressUnmanagedCodeSecurity]
        public delegate void RcpViewToNdArrayDelegate(nint               instance,
                                                      ExecutionSpaceKind execution_space,
                                                      LayoutKind         layout,
                                                      DataTypeKind       data_type,
                                                      ushort             rank,
                                                      out NdArray        ndArray);

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        [SuppressUnmanagedCodeSecurity]
        public delegate ref NdArray ViewToNdArrayDelegate(nint               instance,
                                                          ExecutionSpaceKind execution_space,
                                                          LayoutKind         layout,
                                                          DataTypeKind       data_type,
                                                          ushort             rank);

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        [SuppressUnmanagedCodeSecurity]
        public delegate nint Shepard2dSingleDelegate(nint               xd_rcp_view_ptr,
                                                     nint               zd_rcp_view_ptr,
                                                     float              p,
                                                     nint               xi_rcp_view_ptr,
                                                     ExecutionSpaceKind execution_space);

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        [SuppressUnmanagedCodeSecurity]
        public delegate nint Shepard2dDoubleDelegate(nint               xd_rcp_view_ptr,
                                                     nint               zd_rcp_view_ptr,
                                                     double             p,
                                                     nint               xi_rcp_view_ptr,
                                                     ExecutionSpaceKind execution_space);

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        [SuppressUnmanagedCodeSecurity]
        public delegate nint NearestNeighborSingleDelegate(nint               latlongdegrees_rcp_view_ptr,
                                                           ExecutionSpaceKind execution_space);

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        [SuppressUnmanagedCodeSecurity]
        public delegate nint NearestNeighborDoubleDelegate(nint               latlongdegrees_rcp_view_ptr,
                                                           ExecutionSpaceKind execution_space);

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        [SuppressUnmanagedCodeSecurity]
        public delegate nint CountLineEndingsSerialDelegate(nint instance);

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        [SuppressUnmanagedCodeSecurity]
        public delegate nint CountLineEndingsOpenMPDelegate(nint instance);

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        [SuppressUnmanagedCodeSecurity]
        public delegate nint CountLineEndingsCudaDelegate(nint instance);

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        [SuppressUnmanagedCodeSecurity]
        public delegate nint IpcCreateDelegate(ExecutionSpaceKind   execution_space,
                                               ulong                size,
                                               NativeString<Serial> label);

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        [SuppressUnmanagedCodeSecurity]
        public delegate nint IpcCreateFromDelegate(ExecutionSpaceKind   execution_space,
                                                   nint                 memoryPtr,
                                                   ulong                size,
                                                   NativeString<Serial> label);

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        [SuppressUnmanagedCodeSecurity]
        public delegate nint IpcOpenExistingDelegate(ExecutionSpaceKind   execution_space,
                                                     NativeString<Serial> label);

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        [SuppressUnmanagedCodeSecurity]
        public delegate void IpcDestoryDelegate(ExecutionSpaceKind execution_space,
                                                nint               instance);

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        [SuppressUnmanagedCodeSecurity]
        public delegate void IpcCloseDelegate(ExecutionSpaceKind execution_space,
                                              nint               instance);

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        [SuppressUnmanagedCodeSecurity]
        public delegate nint IpcGetMemoryPointerDelegate(ExecutionSpaceKind execution_space,
                                                         nint               instance);

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        [SuppressUnmanagedCodeSecurity]
        public delegate nint IpcGetDeviceHandleDelegate(ExecutionSpaceKind execution_space,
                                                        nint               instance);

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        [SuppressUnmanagedCodeSecurity]
        public delegate ulong IpcGetSizeDelegate(ExecutionSpaceKind execution_space,
                                                 nint               instance);

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        [SuppressUnmanagedCodeSecurity]
        public delegate nint IpcMakeViewFromPointerDelegate(ExecutionSpaceKind execution_space,
                                                            DataTypeKind       data_type,
                                                            nint               instance,
                                                            ulong              arg_N0 = ulong.MaxValue,
                                                            ulong              arg_N1 = ulong.MaxValue,
                                                            ulong              arg_N2 = ulong.MaxValue,
                                                            ulong              arg_N3 = ulong.MaxValue,
                                                            ulong              arg_N4 = ulong.MaxValue,
                                                            ulong              arg_N5 = ulong.MaxValue,
                                                            ulong              arg_N6 = ulong.MaxValue,
                                                            ulong              arg_N7 = ulong.MaxValue);

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        [SuppressUnmanagedCodeSecurity]
        public delegate nint IpcMakeViewFromHandleDelegate(ExecutionSpaceKind execution_space,
                                                           DataTypeKind       data_type,
                                                           nint               instance,
                                                           ulong              arg_N0 = ulong.MaxValue,
                                                           ulong              arg_N1 = ulong.MaxValue,
                                                           ulong              arg_N2 = ulong.MaxValue,
                                                           ulong              arg_N3 = ulong.MaxValue,
                                                           ulong              arg_N4 = ulong.MaxValue,
                                                           ulong              arg_N5 = ulong.MaxValue,
                                                           ulong              arg_N6 = ulong.MaxValue,
                                                           ulong              arg_N7 = ulong.MaxValue);

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        [SuppressUnmanagedCodeSecurity]
        public delegate uint GetNumaCountDelegate();

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        [SuppressUnmanagedCodeSecurity]
        public delegate uint GetCoresPerNumaDelegate();

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        [SuppressUnmanagedCodeSecurity]
        public delegate uint GetThreadsPerCoreDelegate();

        #endregion

        #region Calli

        // public static void CalliInitialize(int      narg,
        //                                   string[] arg,
        //                                   nint   funcPtr)
        //{
        //    throw new NotImplementedException();
        //}

        // public static void CalliInitializeThreads(int    num_cpu_threads,
        //                                          int    gpu_device_id,
        //                                          nint funcPtr)
        //{
        //    throw new NotImplementedException();
        //}

        // public static void CalliInitializeArguments( InitArguments arguments,
        //                                            nint           funcPtr)
        //{
        //    throw new NotImplementedException();
        //}

        // public static void CalliFinalize(nint funcPtr)
        //{
        //    throw new NotImplementedException();
        //}

        // public static void CalliFinalizeAll(nint funcPtr)
        //{
        //    throw new NotImplementedException();
        //}

        // public static bool CalliIsInitialized(nint funcPtr)
        //{
        //    throw new NotImplementedException();
        //}

        // public static void CalliPrintConfiguration(bool   detail,
        //                                           nint funcPtr)
        //{
        //    throw new NotImplementedException();
        //}

        // public static uint CalliCudaGetDeviceCount(nint funcPtr)
        //{
        //    throw new NotImplementedException();
        //}

        // public static uint CalliCudaGetComputeCapability(uint   device_id,
        //                                                 nint funcPtr)
        //{
        //    throw new NotImplementedException();
        //}

        // public static void CalliCreateViewRank0(nint                instance,
        //                                         DataTypeKind       data_type,
        //                                         ExecutionSpaceKind
        //                                        execution_space, byte[] label,
        //                                        nint                funcPtr)
        //{
        //    throw new NotImplementedException();
        //}

        // public static void CalliCreateViewRank1(nint                instance,
        //                                         DataTypeKind       data_type,
        //                                         ExecutionSpaceKind
        //                                        execution_space, byte[] label, in
        //                                        ulong              n0, nint
        //                                        funcPtr)
        //{
        //    throw new NotImplementedException();
        //}

        // public static void CalliCreateViewRank2(nint                instance,
        //                                         DataTypeKind       data_type,
        //                                         ExecutionSpaceKind
        //                                        execution_space, byte[] label, in
        //                                        ulong              n0,  ulong n1,
        //                                        nint                funcPtr)
        //{
        //    throw new NotImplementedException();
        //}

        // public static void CalliCreateViewRank3(nint                instance,
        //                                         DataTypeKind       data_type,
        //                                         ExecutionSpaceKind
        //                                        execution_space, byte[] label, in
        //                                        ulong              n0,  ulong n1,
        //                                         ulong              n2,
        //                                        nint                funcPtr)
        //{
        //    throw new NotImplementedException();
        //}

        // public static byte[] CalliGetLabel(nint                instance,
        //                                    DataTypeKind       data_type,
        //                                    ExecutionSpaceKind execution_space,
        //                                    uint               rank,
        //                                   nint                funcPtr)
        //{
        //    throw new NotImplementedException();
        //}

        // public static ulong CalliGetSize(nint                instance,
        //                                  DataTypeKind       data_type,
        //                                  ExecutionSpaceKind execution_space,
        //                                  uint               rank,
        //                                 nint                funcPtr)
        //{
        //    throw new NotImplementedException();
        //}

        // public static ulong CalliGetStride(nint                instance,
        //                                    DataTypeKind       data_type,
        //                                    ExecutionSpaceKind execution_space,
        //                                    uint               rank,
        //                                    uint               dim,
        //                                   nint                funcPtr)
        //{
        //    throw new NotImplementedException();
        //}

        // public static ulong CalliGetExtent(nint                instance,
        //                                    DataTypeKind       data_type,
        //                                    ExecutionSpaceKind execution_space,
        //                                    uint               rank,
        //                                    uint               dim,
        //                                   nint                funcPtr)
        //{
        //    throw new NotImplementedException();
        //}

        #endregion

        #region Methods

        public static GetApiDelegate GetApi;

        public static AllocateDelegate Allocate;

        public static ReallocateDelegate Reallocate;

        private static FreeDelegate _free;

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static void Free(ExecutionSpaceKind execution_space,
                                nint               instance)
        {
            try
            {
                _free(execution_space, instance);
            }
            catch(Exception)
            {
                Console.WriteLine($"KokkosLibrary Free failed at 0x{instance:X} with {Enum.GetName(execution_space)}.");
            }
        }

        private static InitializeDelegate _initialize;

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
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
        
        public static InitializeSerialDelegate InitializeSerial;

        public static InitializeOpenMPDelegate InitializeOpenMP;

        public static InitializeCudaDelegate InitializeCuda;

        private static InitializeThreadsDelegate _initializeThreads;

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
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

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static void Initialize(InitArguments arguments)
        {
            if(!IsInitialized())
            {
                Load();

                _initializeArguments(arguments);

                Initialized = true;
            }
        }

        private static FinalizeDelegate _finalize;

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static void Finalize()
        {
            _finalize();

            //Unload();

            Initialized = false;
        }

        public static FinalizeSerialDelegate FinalizeSerial;

        public static FinalizeOpenMPDelegate FinalizeOpenMP;

        public static FinalizeCudaDelegate FinalizeCuda;

        private static FinalizeAllDelegate _finalizeAll;

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static void FinalizeAll()
        {
            _finalizeAll();

            //Unload();

            Initialized = false;
        }

        private static IsInitializedDelegate _isInitialized;

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
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

        public static CreateViewRank4Delegate CreateViewRank4;

        public static CreateViewRank5Delegate CreateViewRank5;

        public static CreateViewRank6Delegate CreateViewRank6;

        public static CreateViewRank7Delegate CreateViewRank7;

        public static CreateViewRank8Delegate CreateViewRank8;

        public static CreateViewDelegate CreateView;

        //public static delegate* unmanaged[Cdecl]<nint, NdArray, NativeString<Serial>> GetLabel;

        //public static delegate* unmanaged[Cdecl]<nint, NdArray, ulong> GetSize;

        //public static delegate* unmanaged[Cdecl]<nint, NdArray, uint, ulong> GetStride;

        //public static delegate* unmanaged[Cdecl]<nint, NdArray, uint, ulong> GetExtent;

        public static GetLabelDelegate GetLabel;

        public static GetSizeDelegate GetSize;

        public static GetStrideDelegate GetStride;

        public static GetExtentDelegate GetExtent;

        public static CopyToDelegate CopyTo;

        //public static delegate* unmanaged[Cdecl]<nint, NdArray, ulong, ulong, ulong, ulong, ulong, ulong, ulong, ulong, ValueType> GetValue;

        //public static delegate* unmanaged[Cdecl]<nint, NdArray, ValueType, ulong, ulong, ulong, ulong, ulong, ulong, ulong, ulong, void> SetValue;

        public static GetValueDelegate GetValue;

        public static SetValueDelegate SetValue;

        public static RcpViewToNdArrayDelegate RcpViewToNdArray;

        public static ViewToNdArrayDelegate ViewToNdArray;

        public static GetNumaCountDelegate      GetNumaCount;

        public static GetCoresPerNumaDelegate   GetCoresPerNuma;

        public static GetThreadsPerCoreDelegate GetThreadsPerCore;

        public static Shepard2dSingleDelegate Shepard2dSingle;

        public static Shepard2dDoubleDelegate Shepard2dDouble;

        public static NearestNeighborSingleDelegate NearestNeighborSingle;

        public static NearestNeighborDoubleDelegate NearestNeighborDouble;

        public static CountLineEndingsSerialDelegate CountLineEndingsSerial;

        public static CountLineEndingsOpenMPDelegate CountLineEndingsOpenMP;

        public static CountLineEndingsCudaDelegate CountLineEndingsCuda;

        public static IpcCreateDelegate              IpcCreate;
        public static IpcCreateFromDelegate          IpcCreateFrom;
        public static IpcOpenExistingDelegate        IpcOpenExisting;
        public static IpcDestoryDelegate             IpcDestory;
        public static IpcCloseDelegate               IpcClose;
        public static IpcGetMemoryPointerDelegate    IpcGetMemoryPointer;
        public static IpcGetDeviceHandleDelegate     IpcGetDeviceHandle;
        public static IpcGetSizeDelegate             IpcGetSize;
        public static IpcMakeViewFromPointerDelegate IpcMakeViewFromPointer;
        public static IpcMakeViewFromHandleDelegate  IpcMakeViewFromHandle;

        #endregion
    }
}