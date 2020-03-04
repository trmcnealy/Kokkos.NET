#pragma warning disable CS0465

using System;
using System.Reflection;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Versioning;

namespace Kokkos
{
    [NonVersionable]
    internal static class KokkosLibrary
    {
        public const string LibraryName = "runtime.Kokkos.NET";

        public static readonly IntPtr Handle;

        public static readonly KokkosApi Api;

        public static volatile bool Initialized;

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        static KokkosLibrary()
        {
            // NativeLibrary.SetDllImportResolver(typeof(KokkosLibrary).Assembly,
            // ImportResolver);

            KokkosCoreLibrary.Initialize();

            string runtimeKokkosLibraryName = LibraryName + (RuntimeInformation.ProcessArchitecture == Architecture.X64 ? ".x64" : ".x86");

            Console.WriteLine("Loading " + runtimeKokkosLibraryName);

            if(!NativeLibrary.TryLoad(runtimeKokkosLibraryName,
                                      typeof(KokkosLibrary).Assembly,
                                      DllImportSearchPath.UseDllDirectoryForDependencies,
                                      out Handle))
            {
                KokkosLibraryException.Throw();
            }

            if(NativeLibrary.TryGetExport(Handle,
                                          "GetApi",
                                          out IntPtr getApiHandle))
            {
                GetApi = Marshal.GetDelegateForFunctionPointer<GetApiDelegate>(getApiHandle);

                Api = GetApi(1);

                Allocate = Marshal.GetDelegateForFunctionPointer<AllocateDelegate>(Api.AllocatePtr);

                Reallocate = Marshal.GetDelegateForFunctionPointer<ReallocateDelegate>(Api.ReallocatePtr);

                Free = Marshal.GetDelegateForFunctionPointer<FreeDelegate>(Api.FreePtr);

                initialize = Marshal.GetDelegateForFunctionPointer<InitializeDelegate>(Api.InitializePtr);

                initializeThreads = Marshal.GetDelegateForFunctionPointer<InitializeThreadsDelegate>(Api.InitializeThreadsPtr);

                initializeArguments = Marshal.GetDelegateForFunctionPointer<InitializeArgumentsDelegate>(Api.InitializeArgumentsPtr);

                finalize = Marshal.GetDelegateForFunctionPointer<FinalizeDelegate>(Api.FinalizePtr);

                finalizeAll = Marshal.GetDelegateForFunctionPointer<FinalizeAllDelegate>(Api.FinalizeAllPtr);

                isInitialized = Marshal.GetDelegateForFunctionPointer<IsInitializedDelegate>(Api.IsInitializedPtr);

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

                ViewToNdArray = Marshal.GetDelegateForFunctionPointer<ViewToNdArrayDelegate>(Api.ViewToNdArrayPtr);
            }
            else
            {
                KokkosLibraryException.Throw("'runtime.Kokkos.NET::GetApi' not found.");
            }

            Console.WriteLine("Loaded " + runtimeKokkosLibraryName + $"@ 0x{Handle.ToString(" X ")}");
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        public static bool IsLoaded()
        {
            return Handle != IntPtr.Zero;
        }

        internal static void Unload()
        {
            NativeLibrary.Free(Handle);
        }

        // private static IntPtr ImportResolver(string               libraryName,
        //                                     Assembly             assembly,
        //                                     DllImportSearchPath? searchPath =
        //                                     DllImportSearchPath.UseDllDirectoryForDependencies)
        //{
        //    IntPtr libHandle = IntPtr.Zero;
        //
        //    if(libraryName == LibraryName)
        //    {
        //        NativeLibrary.TryLoad(LibraryName,
        //                              assembly,
        //                              searchPath,
        //                              out libHandle);
        //    }
        //
        //    return libHandle;
        //}

        #region Delegates

        internal delegate ref KokkosApi GetApiDelegate(in uint version);

        internal delegate IntPtr AllocateDelegate(in ExecutionSpaceKind execution_space,
                                                  in ulong              arg_alloc_size);

        internal delegate IntPtr ReallocateDelegate(in ExecutionSpaceKind execution_space,
                                                    IntPtr                instance,
                                                    in ulong              arg_alloc_size);

        internal delegate void FreeDelegate(in ExecutionSpaceKind execution_space,
                                            IntPtr                instance);

        internal delegate void InitializeDelegate(int narg,
                                                  [MarshalAs(UnmanagedType.LPArray,
                                                             ArraySubType = UnmanagedType.LPStr)]
                                                  string[] arg);

        internal delegate void InitializeThreadsDelegate(int num_cpu_threads,
                                                         int gpu_device_id);

        internal delegate void InitializeArgumentsDelegate(in InitArguments arguments);

        internal delegate void FinalizeDelegate();

        internal delegate void FinalizeAllDelegate();

        internal delegate bool IsInitializedDelegate();

        internal delegate void PrintConfigurationDelegate(bool detail);

        internal delegate uint CudaGetDeviceCountDelegate();

        internal delegate uint CudaGetComputeCapabilityDelegate(uint device_id);

        internal delegate void CreateViewRank0Delegate(IntPtr      instance,
                                                       ref NdArray nArray);

        internal delegate void CreateViewRank1Delegate(IntPtr      instance,
                                                       ref NdArray nArray,
                                                       in  ulong   n0);

        internal delegate void CreateViewRank2Delegate(IntPtr      instance,
                                                       ref NdArray nArray,
                                                       in  ulong   n0,
                                                       in  ulong   n1);

        internal delegate void CreateViewRank3Delegate(IntPtr      instance,
                                                       ref NdArray nArray,
                                                       in  ulong   n0,
                                                       in  ulong   n1,
                                                       in  ulong   n2);

        internal delegate void CreateViewDelegate(IntPtr      instance,
                                                  ref NdArray nArray);

        internal delegate NativeString GetLabelDelegate(IntPtr     instance,
                                                        in NdArray nArray);

        internal delegate ulong GetSizeDelegate(IntPtr     instance,
                                                in NdArray nArray);

        internal delegate ulong GetStrideDelegate(IntPtr     instance,
                                                  in NdArray nArray,
                                                  in uint    dim);

        internal delegate ulong GetExtentDelegate(IntPtr     instance,
                                                  in NdArray nArray,
                                                  in uint    dim);

        internal delegate void CopyToDelegate(IntPtr      instance,
                                              in NdArray  nArray,
                                              ValueType[] values);

        internal delegate ValueType GetValueDelegate(IntPtr     instance,
                                                     in NdArray nArray,
                                                     in ulong   i0 = ulong.MaxValue,
                                                     in ulong   i1 = ulong.MaxValue,
                                                     in ulong   i2 = ulong.MaxValue);

        internal delegate void SetValueDelegate(IntPtr       instance,
                                                in NdArray   nArray,
                                                in ValueType value,
                                                in ulong     i0 = ulong.MaxValue,
                                                in ulong     i1 = ulong.MaxValue,
                                                in ulong     i2 = ulong.MaxValue);

        internal delegate NdArray ViewToNdArrayDelegate(IntPtr                instance,
                                                        in ExecutionSpaceKind execution_space,
                                                        in LayoutKind         layout,
                                                        in DataTypeKind       data_type,
                                                        in ushort             rank);

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

        internal static readonly GetApiDelegate GetApi;

        internal static readonly AllocateDelegate Allocate;

        internal static readonly ReallocateDelegate Reallocate;

        internal static readonly FreeDelegate Free;

        private static readonly InitializeDelegate initialize;

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        internal static void Initialize(int      narg,
                                        string[] arg)
        {
            if(Initialized)
            {
                KokkosLibraryException.Throw("Kokkos Library has already been initialized.");
            }

            Initialized = true;

            initialize(narg,
                       arg);
        }

        private static readonly InitializeThreadsDelegate initializeThreads;

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        internal static void Initialize(int num_cpu_threads,
                                        int gpu_device_id)
        {
            if(Initialized)
            {
                KokkosLibraryException.Throw("Kokkos Library has already been initialized.");
            }

            Initialized = true;

            initializeThreads(num_cpu_threads,
                              gpu_device_id);
        }

        private static readonly InitializeArgumentsDelegate initializeArguments;

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        internal static void Initialize(in InitArguments arguments)
        {
            if(Initialized)
            {
                KokkosLibraryException.Throw("Kokkos Library has already been initialized.");
            }

            Initialized = true;

            initializeArguments(arguments);
        }

        private static readonly FinalizeDelegate finalize;

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        internal static void Finalize()
        {
            if(!Initialized)
            {
                KokkosLibraryException.Throw("Kokkos Library has not been initialized.");
            }

            Initialized = false;

            finalize();
        }

        private static readonly FinalizeAllDelegate finalizeAll;

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        internal static void FinalizeAll()
        {
            if(!Initialized)
            {
                KokkosLibraryException.Throw("Kokkos Library has not been initialized.");
            }

            Initialized = false;

            finalizeAll();
        }

        private static readonly IsInitializedDelegate isInitialized;

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        internal static bool IsInitialized()
        {
            if(!Initialized)
            {
                return false;
            }

            return isInitialized();
        }

        internal static readonly PrintConfigurationDelegate PrintConfiguration;

        internal static readonly CudaGetComputeCapabilityDelegate GetComputeCapability;

        internal static readonly CudaGetDeviceCountDelegate GetDeviceCount;

        internal static readonly CreateViewRank0Delegate CreateViewRank0;

        internal static readonly CreateViewRank1Delegate CreateViewRank1;

        internal static readonly CreateViewRank2Delegate CreateViewRank2;

        internal static readonly CreateViewRank3Delegate CreateViewRank3;

        internal static readonly CreateViewDelegate CreateView;

        internal static readonly GetLabelDelegate GetLabel;

        internal static readonly GetSizeDelegate GetSize;

        internal static readonly GetStrideDelegate GetStride;

        internal static readonly GetExtentDelegate GetExtent;

        internal static readonly CopyToDelegate CopyTo;

        internal static readonly GetValueDelegate GetValue;

        internal static readonly SetValueDelegate SetValue;

        internal static readonly ViewToNdArrayDelegate ViewToNdArray;
        
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static NdArray Convert(IntPtr             view_ptr,
                                      ExecutionSpaceKind execution_space,
                                      LayoutKind         layoutkind,
                                      DataTypeKind       data_type,
                                      ushort             rank)
        {
            return ViewToNdArray(view_ptr,
                                 execution_space,
                                 layoutkind,
                                 data_type,
                                 rank);
        }

        //[SuppressUnmanagedCodeSecurity]
        //[DllImport("kernel32.dll",
        //           CharSet       = CharSet.Ansi,
        //           ExactSpelling = true,
        //           EntryPoint    = "lstrlenA")]
        // internal static extern int lstrlenA(IntPtr ptr);

        //[MethodImpl(MethodImplOptions.AggressiveInlining)]
        // internal static int strlen(sbyte* buffer)
        //{
        //    return lstrlenA(*(IntPtr*)buffer);
        //}

        //[MethodImpl(MethodImplOptions.AggressiveInlining)]
        // internal static string ToString(sbyte* bytes)
        //{
        //    return new string(bytes,
        //                      0,
        //                      strlen(bytes),
        //                      Encoding.UTF8);
        //}

        //[MethodImpl(MethodImplOptions.AggressiveInlining)]
        // internal static string ToString(this byte[] bytes)
        //{
        //    return bytes.ToString();
        //}

        //[MethodImpl(MethodImplOptions.AggressiveInlining)]
        // internal static sbyte* ToSByte(this string @string)
        //{
        //    if(@string[^1] != '\0')
        //    {
        //        @string += '\0';
        //    }

        //    byte[] bytes = Encoding.UTF8.GetBytes(@string);

        //    fixed(byte* p = bytes)
        //    {
        //        //sbyte* sp = (sbyte*)p;
        //        return (sbyte*)p;
        //    }
        //}

        //[MethodImpl(MethodImplOptions.AggressiveInlining)]
        // internal static byte[] ToBytes(this string @string)
        //{
        //    if(@string[^1] != char.MinValue)
        //    {
        //        @string += char.MinValue;
        //    }

        //    if(RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
        //    {
        //        return Encoding.ASCII.GetBytes(@string);
        //    }

        //    return Encoding.UTF8.GetBytes(@string);
        //}

        //[MethodImpl(MethodImplOptions.AggressiveInlining)]
        // internal static string FromToBytes(this byte[] bytes)
        //{
        //    if(RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
        //    {
        //        return Encoding.ASCII.GetString(bytes);
        //    }

        //    return Encoding.UTF8.GetString(bytes);
        //}

        //[MethodImpl(MethodImplOptions.AggressiveInlining)]
        // internal static byte[] ToBytes(this string @string)
        //{
        //    if(@string[^1] != '\0')
        //    {
        //        @string += '\0';
        //    }

        //    byte[] bytes = Encoding.UTF8.GetBytes(@string);

        //    return bytes;
        //}
    }
}