using System;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Versioning;
using System.Security;

namespace Kokkos
{


    [NonVersionable]
    public sealed class Vector<TDataType, TExecutionSpace> : IDisposable
        where TDataType : unmanaged
        where TExecutionSpace : IExecutionSpace, new()
    {
        private static readonly DataTypeKind dataType;

        private static readonly IExecutionSpace executionSpace;

        private static readonly ExecutionSpaceKind executionSpaceType;

        public NativePointer Pointer
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            get;
        }


        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        static Vector()
        {
            dataType           = DataType<TDataType>.GetKind();
            executionSpace     = new TExecutionSpace();
            executionSpaceType = ExecutionSpace<TExecutionSpace>.GetKind();

            VectorUtilities<TDataType, TExecutionSpace>.Load();
        }
        
        ~Vector()
        {
            pointer.Dispose();
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public nint data()
        {
            throw null;
        }


    }

    //[System.Runtime.InteropServices.StructLayout(System.Runtime.InteropServices.LayoutKind.Explicit)]
    //public sealed class IVectorVTable
    //{
    //    [FieldOffset(sizeof(ulong) * 1)] public nint dcctorPtr;
    //    [FieldOffset(sizeof(ulong) * 2)] public nint sync_hostPtr;
    //    [FieldOffset(sizeof(ulong) * 3)] public nint sync_devicePtr;
    //    [FieldOffset(sizeof(ulong) * 4)] public nint need_sync_hostPtr;
    //    [FieldOffset(sizeof(ulong) * 5)] public nint need_sync_devicePtr;
    //    [FieldOffset(sizeof(ulong) * 6)] public nint assignPtr;
    //    [FieldOffset(sizeof(ulong) * 7)] public nint backPtr;
    //    [FieldOffset(sizeof(ulong) * 8)] public nint beginPtr;
    //    [FieldOffset(sizeof(ulong) * 9)] public nint clearPtr;
    //    [FieldOffset(sizeof(ulong) * 10)] public nint dataPtr;
    //    [FieldOffset(sizeof(ulong) * 11)] public nint device_to_hostPtr;
    //    [FieldOffset(sizeof(ulong) * 12)] public nint emptyPtr;
    //    [FieldOffset(sizeof(ulong) * 13)] public nint endPtr;
    //    [FieldOffset(sizeof(ulong) * 14)] public nint findPtr;
    //    [FieldOffset(sizeof(ulong) * 15)] public nint frontPtr;
    //    [FieldOffset(sizeof(ulong) * 16)] public nint host_to_devicePtr;
    //    [FieldOffset(sizeof(ulong) * 17)] public nint insertPtr;
    //    [FieldOffset(sizeof(ulong) * 18)] public nint insert2Ptr;
    //    [FieldOffset(sizeof(ulong) * 19)] public nint is_allocatedPtr;
    //    [FieldOffset(sizeof(ulong) * 20)] public nint is_sortedPtr;
    //    [FieldOffset(sizeof(ulong) * 21)] public nint lower_boundPtr;
    //    [FieldOffset(sizeof(ulong) * 22)] public nint max_sizePtr;
    //    [FieldOffset(sizeof(ulong) * 23)] public nint on_devicePtr;
    //    [FieldOffset(sizeof(ulong) * 24)] public nint on_hostPtr;
    //    [FieldOffset(sizeof(ulong) * 25)] public nint pop_backPtr;
    //    [FieldOffset(sizeof(ulong) * 26)] public nint push_backPtr;
    //    [FieldOffset(sizeof(ulong) * 27)] public nint reservePtr;
    //    [FieldOffset(sizeof(ulong) * 28)] public nint resizePtr;
    //    [FieldOffset(sizeof(ulong) * 29)] public nint resize2Ptr;
    //    [FieldOffset(sizeof(ulong) * 30)] public nint set_overallocationPtr;
    //    [FieldOffset(sizeof(ulong) * 31)] public nint sizePtr;
    //    [FieldOffset(sizeof(ulong) * 32)] public nint spanPtr;
    //}




    //[System.Runtime.InteropServices.StructLayout(System.Runtime.InteropServices.LayoutKind.Sequential)]
    //public unsafe class VectorBase<TDataType, TExecutionSpace>
    //    where TDataType : unmanaged
    //    where TExecutionSpace : IExecutionSpace
    //{
    //    void sync_host();

    //    void sync_device();

    //    bool need_sync_host();

    //    bool need_sync_device();

    //    void assign(ulong        n,
    //                in TDataType value);

    //    ref TDataType back();

    //    TDataType* begin();

    //    void clear();

    //    TDataType* data();

    //    void device_to_host();

    //    bool empty();

    //    TDataType* end();

    //    TDataType* find(TDataType value);

    //    ref TDataType front();

    //    void host_to_device();

    //    TDataType* insert(TDataType*   it,
    //                      in TDataType value);

    //    TDataType* insert(TDataType*   it,
    //                      ulong        count,
    //                      in TDataType value);

    //    bool is_allocated();

    //    bool is_sorted();

    //    ulong lower_bound(in ulong     start,
    //                      in ulong     theEnd,
    //                      in TDataType comp_val);

    //    ulong max_size();

    //    void on_device();

    //    void on_host();

    //    void pop_back();

    //    void push_back(TDataType value);

    //    void reserve(ulong n);

    //    void resize(ulong n);

    //    void resize(ulong        n,
    //                in TDataType value);

    //    void set_overallocation(float extra);

    //    ulong size();

    //    ulong span();
    //}














//    [ILGeneration(typeof(float),
//                  typeof(double),
//                  typeof(bool),
//                  typeof(sbyte),
//                  typeof(byte),
//                  typeof(short),
//                  typeof(ushort),
//                  typeof(int),
//                  typeof(uint),
//                  typeof(long),
//                  typeof(ulong))]
//    public static class VectorUtilities<TDataType, TExecutionSpace>
//        where TDataType : unmanaged
//        where TExecutionSpace : IExecutionSpace, new()
//    {
    
//        [SuppressUnmanagedCodeSecurity]
//        [UnmanagedFunctionPointer(CallingConvention.ThisCall)]
//        public delegate nint createDelegate1();

//        [SuppressUnmanagedCodeSecurity]
//        [UnmanagedFunctionPointer(CallingConvention.ThisCall)]
//        public delegate nint createDelegate2(int       n,
//                                               TDataType value);

//        [SuppressUnmanagedCodeSecurity]
//        [UnmanagedFunctionPointer(CallingConvention.ThisCall)]
//        public delegate void destoryDelegate(nint @this);



//        [SuppressUnmanagedCodeSecurity]
//        [UnmanagedFunctionPointer(CallingConvention.ThisCall)]
//        public delegate void sync_hostDelegate();

//        [SuppressUnmanagedCodeSecurity]
//        [UnmanagedFunctionPointer(CallingConvention.ThisCall)]
//        public delegate void sync_deviceDelegate();

//        [SuppressUnmanagedCodeSecurity]
//        [UnmanagedFunctionPointer(CallingConvention.ThisCall)]
//        public delegate bool need_sync_hostDelegate();

//        [SuppressUnmanagedCodeSecurity]
//        [UnmanagedFunctionPointer(CallingConvention.ThisCall)]
//        public delegate bool need_sync_deviceDelegate();



//        [SuppressUnmanagedCodeSecurity]
//        [UnmanagedFunctionPointer(CallingConvention.ThisCall)]
//        public delegate void assignDelegate(nint       @this,
//                                            ulong        n,
//                                            in TDataType val);

//        [SuppressUnmanagedCodeSecurity]
//        [UnmanagedFunctionPointer(CallingConvention.ThisCall)]
//        public delegate ref TDataType backDelegate(nint @this);

//        [SuppressUnmanagedCodeSecurity]
//        [UnmanagedFunctionPointer(CallingConvention.ThisCall)]
//        public delegate nint beginDelegate(nint @this);



//        [SuppressUnmanagedCodeSecurity]
//        [UnmanagedFunctionPointer(CallingConvention.ThisCall)]
//        public delegate void clearDelegate(nint @this);

//        [SuppressUnmanagedCodeSecurity]
//        [UnmanagedFunctionPointer(CallingConvention.ThisCall)]
//        public delegate nint dataDelegate(nint @this);

//        [SuppressUnmanagedCodeSecurity]
//        [UnmanagedFunctionPointer(CallingConvention.ThisCall)]
//        public delegate void device_to_hostDelegate(nint @this);

//        [SuppressUnmanagedCodeSecurity]
//        [UnmanagedFunctionPointer(CallingConvention.ThisCall)]
//        public delegate bool emptyDelegate(nint @this);

//        [SuppressUnmanagedCodeSecurity]
//        [UnmanagedFunctionPointer(CallingConvention.ThisCall)]
//        public delegate nint endDelegate(nint @this);

//        [SuppressUnmanagedCodeSecurity]
//        [UnmanagedFunctionPointer(CallingConvention.ThisCall)]
//        public delegate nint findDelegate(nint    @this,
//                                            TDataType val);

//        [SuppressUnmanagedCodeSecurity]
//        [UnmanagedFunctionPointer(CallingConvention.ThisCall)]
//        public delegate ref TDataType frontDelegate(nint @this);

//        [SuppressUnmanagedCodeSecurity]
//        [UnmanagedFunctionPointer(CallingConvention.ThisCall)]
//        public delegate void host_to_deviceDelegate(nint @this);

//        [SuppressUnmanagedCodeSecurity]
//        [UnmanagedFunctionPointer(CallingConvention.ThisCall)]
//        public delegate nint insertDelegate1(nint       @this,
//                                               nint       it,
//                                               in TDataType val);

//        [SuppressUnmanagedCodeSecurity]
//        [UnmanagedFunctionPointer(CallingConvention.ThisCall)]
//        public delegate nint insertDelegate2(nint       @this,
//                                               nint       it,
//                                               ulong        count,
//                                               in TDataType val);

//        [SuppressUnmanagedCodeSecurity]
//        [UnmanagedFunctionPointer(CallingConvention.ThisCall)]
//        public delegate bool is_allocatedDelegate(nint @this);

//        [SuppressUnmanagedCodeSecurity]
//        [UnmanagedFunctionPointer(CallingConvention.ThisCall)]
//        public delegate bool is_sortedDelegate(nint @this);

//        [SuppressUnmanagedCodeSecurity]
//        [UnmanagedFunctionPointer(CallingConvention.ThisCall)]
//        public delegate ulong lower_boundDelegate(nint       @this,
//                                                  in ulong     start,
//                                                  in ulong     theEnd,
//                                                  in TDataType comp_val);

//        [SuppressUnmanagedCodeSecurity]
//        [UnmanagedFunctionPointer(CallingConvention.ThisCall)]
//        public delegate ulong max_sizeDelegate(nint @this);

//        [SuppressUnmanagedCodeSecurity]
//        [UnmanagedFunctionPointer(CallingConvention.ThisCall)]
//        public delegate void on_deviceDelegate(nint @this);

//        [SuppressUnmanagedCodeSecurity]
//        [UnmanagedFunctionPointer(CallingConvention.ThisCall)]
//        public delegate void on_hostDelegate(nint @this);

//        [SuppressUnmanagedCodeSecurity]
//        [UnmanagedFunctionPointer(CallingConvention.ThisCall)]
//        public delegate ref TDataType operator_bracketsDelegate(nint @this,
//                                                                int    i);

//        [SuppressUnmanagedCodeSecurity]
//        [UnmanagedFunctionPointer(CallingConvention.ThisCall)]
//        public delegate ref TDataType operatorDelegate(nint @this,
//                                                       int    i);

//        [SuppressUnmanagedCodeSecurity]
//        [UnmanagedFunctionPointer(CallingConvention.ThisCall)]
//        public delegate void pop_backDelegate(nint @this);

//        [SuppressUnmanagedCodeSecurity]
//        [UnmanagedFunctionPointer(CallingConvention.ThisCall)]
//        public delegate void push_backDelegate(nint    @this,
//                                               TDataType val);

//        [SuppressUnmanagedCodeSecurity]
//        [UnmanagedFunctionPointer(CallingConvention.ThisCall)]
//        public delegate void reserveDelegate(nint @this,
//                                             ulong  n);

//        [SuppressUnmanagedCodeSecurity]
//        [UnmanagedFunctionPointer(CallingConvention.ThisCall)]
//        public delegate void resize1Delegate(nint @this,
//                                             ulong  n);

//        [SuppressUnmanagedCodeSecurity]
//        [UnmanagedFunctionPointer(CallingConvention.ThisCall)]
//        public delegate void resize2Delegate(nint       @this,
//                                             ulong        n,
//                                             in TDataType val);

//        [SuppressUnmanagedCodeSecurity]
//        [UnmanagedFunctionPointer(CallingConvention.ThisCall)]
//        public delegate void set_overallocationDelegate(nint @this,
//                                                        float  extra);

//        [SuppressUnmanagedCodeSecurity]
//        [UnmanagedFunctionPointer(CallingConvention.ThisCall)]
//        public delegate ulong sizeDelegate(nint @this);

//        [SuppressUnmanagedCodeSecurity]
//        [UnmanagedFunctionPointer(CallingConvention.ThisCall)]
//        public delegate ulong spanDelegate(nint @this);
        
//        public static readonly Func<char, string, string> vtable_Name = (data_type, execution_space) => $"_ZTVN6Kokkos6VectorI{data_type}NS_6{execution_space}EEE";
        
//        public static readonly Func<char, string, string> create1_Name = (data_type, execution_space) => $"_ZN6Kokkos6VectorI{data_type}NS_6{execution_space}EE6CreateEv";
//        public static readonly Func<char, string, string> create2_Name = (data_type, execution_space) => $"_ZN6Kokkos6VectorI{data_type}NS_6{execution_space}EE6CreateEld";

//        public static readonly Func<char, string, string> destory_Name = (data_type, execution_space) => $"_ZN6Kokkos6VectorI{data_type}NS_6{execution_space}EE7DestoryEPS2_";

//        public static readonly Func<char, string, string> sync_host_Name        = (data_type, execution_space) => $"_ZN6Kokkos6VectorI{data_type}NS_6{execution_space}EE9sync_hostEv";
//        public static readonly Func<char, string, string> sync_device_Name      = (data_type, execution_space) => $"_ZN6Kokkos6VectorI{data_type}NS_4{execution_space}EE11sync_deviceEv";
//        public static readonly Func<char, string, string> need_sync_host_Name   = (data_type, execution_space) => $"_ZN6Kokkos6VectorI{data_type}NS_4{execution_space}EE14need_sync_hostEv";
//        public static readonly Func<char, string, string> need_sync_device_Name = (data_type, execution_space) => $"_ZN6Kokkos6VectorI{data_type}NS_4{execution_space}EE16need_sync_deviceEv";

//        public static readonly Func<char, string, string> reserve_Name = (data_type, execution_space) => $"_ZN6Kokkos6vectorI{data_type}vE7reserveEy";

//        public static readonly Func<char, string, string> resize1_Name = (data_type, execution_space) => $"_ZN6Kokkos6vectorI{data_type}vE6resizeEy";
//        public static readonly Func<char, string, string> resize2_Name = (data_type, execution_space) => $"_ZN6Kokkos6vectorI{data_type}vE6resizeEyRK{data_type}";

//        public static readonly Func<char, string, string> data_Name = (data_type, execution_space) => $"_ZNK6Kokkos6vectorI{data_type}vE4dataEv";

//        public static readonly Func<char, string, string> is_allocated_Name = (data_type, execution_space) => $"_ZNK6Kokkos6vectorI{data_type}vE12is_allocatedEv";
//        public static readonly Func<char, string, string> is_sorted_Name    = (data_type, execution_space) => $"_ZN6Kokkos6vectorI{data_type}vE9is_sortedEv";

//        public static readonly Func<char, string, string> on_device_Name      = (data_type, execution_space) => $"_ZN6Kokkos6vectorI{data_type}vE9on_deviceEv";
//        public static readonly Func<char, string, string> on_host_Name        = (data_type, execution_space) => $"_ZN6Kokkos6vectorI{data_type}vE7on_hostEv";
//        public static readonly Func<char, string, string> device_to_host_Name = (data_type, execution_space) => $"_ZN6Kokkos6vectorI{data_type}vE14device_to_hostEv";
//        public static readonly Func<char, string, string> host_to_device_Name = (data_type, execution_space) => $"_ZNK6Kokkos6vectorI{data_type}vE14host_to_deviceEv";

//        public static readonly Func<char, string, string> set_overallocation_Name = (data_type, execution_space) => $"_ZN6Kokkos6vectorI{data_type}vE18set_overallocationEf";

//        public static readonly Func<char, string, string> assign_Name    = (data_type, execution_space) => $"_ZN6Kokkos6vectorI{data_type}vE6assignEyRK{data_type}";
//        public static readonly Func<char, string, string> push_back_Name = (data_type, execution_space) => $"_ZN6Kokkos6vectorI{data_type}vE9push_backE{data_type}";
//        public static readonly Func<char, string, string> pop_back_Name  = (data_type, execution_space) => $"_ZN6Kokkos6vectorI{data_type}vE8pop_backEv";
//        public static readonly Func<char, string, string> insert1_Name   = (data_type, execution_space) => $"_ZN6Kokkos6vectorI{data_type}vE6insertEP{data_type}RK{data_type}";
//        public static readonly Func<char, string, string> insert2_Name   = (data_type, execution_space) => $"_ZN6Kokkos6vectorI{data_type}vE6insertEP{data_type}yRK{data_type}";
//        public static readonly Func<char, string, string> clear_Name     = (data_type, execution_space) => $"_ZN6Kokkos6vectorI{data_type}vE5clearEv";

//        public static readonly Func<char, string, string> lower_bound_Name = (data_type, execution_space) => $"_ZNK6Kokkos6vectorI{data_type}vE11lower_boundERKyS3_RK{data_type}";
//        public static readonly Func<char, string, string> find_Name        = (data_type, execution_space) => $"_ZNK6Kokkos6vectorI{data_type}vE4findE{data_type}";

//        public static readonly Func<char, string, string> empty_Name    = (data_type, execution_space) => $"_ZNK6Kokkos6vectorI{data_type}vE5emptyEv";
//        public static readonly Func<char, string, string> size_Name     = (data_type, execution_space) => $"_ZNK6Kokkos6vectorI{data_type}vE4sizeEv";
//        public static readonly Func<char, string, string> max_size_Name = (data_type, execution_space) => $"_ZNK6Kokkos6vectorI{data_type}vE8max_sizeEv";
//        public static readonly Func<char, string, string> span_Name     = (data_type, execution_space) => $"_ZNK6Kokkos6vectorI{data_type}vE4spanEv";

//        public static readonly Func<char, string, string> begin_Name = (data_type, execution_space) => $"_ZNK6Kokkos6vectorI{data_type}vE5beginEv";
//        public static readonly Func<char, string, string> end_Name   = (data_type, execution_space) => $"_ZNK6Kokkos6vectorI{data_type}vE3endEv";

//        public static readonly Func<char, string, string> front1_Name = (data_type, execution_space) => $"_ZN6Kokkos6vectorI{data_type}vE5frontEv";
//        public static readonly Func<char, string, string> front2_Name = (data_type, execution_space) => $"_ZNK6Kokkos6vectorI{data_type}vE5frontEv";
//        public static readonly Func<char, string, string> back1_Name  = (data_type, execution_space) => $"_ZN6Kokkos6vectorI{data_type}vE4backEv";
//        public static readonly Func<char, string, string> back2_Name  = (data_type, execution_space) => $"_ZNK6Kokkos6vectorI{data_type}vE4backEv";

//        public static readonly Func<char, string, string> operator1_Name         = (data_type, execution_space) => $"_ZN6Kokkos6vectorI{data_type}vEaSERKS1_";
//        public static readonly Func<char, string, string> operator2_Name         = (data_type, execution_space) => $"_ZNK6Kokkos6vectorI{data_type}vEclEi";
//        public static readonly Func<char, string, string> operator_brackets_Name = (data_type, execution_space) => $"_ZNK6Kokkos6vectorI{data_type}vEixEi";

//        public static createDelegate1 create1;
//        public static createDelegate2 create2;

//        public static destoryDelegate destory;
        
//        public static sync_hostDelegate        sync_host;
//        public static sync_deviceDelegate      sync_device;
//        public static need_sync_hostDelegate   need_sync_host;
//        public static need_sync_deviceDelegate need_sync_device;

//        public static operatorDelegate          @operator;
//        public static operator_bracketsDelegate operator_brackets;

//        public static resize1Delegate   resize1;
//        public static resize2Delegate   resize2;
//        public static assignDelegate    assign;
//        public static reserveDelegate   reserve;
//        public static push_backDelegate push_back;

//        public static pop_backDelegate pop_back;

//        public static clearDelegate clear;

//        public static findDelegate find;

//        public static insertDelegate1 insert1;
//        public static insertDelegate2 insert2;

//        public static is_allocatedDelegate is_allocated;
//        public static sizeDelegate         size;
//        public static max_sizeDelegate     max_size;
//        public static spanDelegate         span;
//        public static emptyDelegate        empty;
//        public static dataDelegate         data;
//        public static beginDelegate        begin;
//        public static endDelegate          end;
//        public static frontDelegate        front;
//        public static backDelegate         back;

//        public static lower_boundDelegate        lower_bound;
//        public static is_sortedDelegate          is_sorted;
//        public static device_to_hostDelegate     device_to_host;
//        public static host_to_deviceDelegate     host_to_device;
//        public static on_deviceDelegate          on_device;
//        public static on_hostDelegate            on_host;
//        public static set_overallocationDelegate set_overallocation;

//        private static readonly Type TypeOfDataType = typeof(TDataType);

//        private static readonly DataTypeKind dataType;

//        private static readonly IExecutionSpace executionSpace;

//        private static readonly ExecutionSpaceKind executionSpaceType;

//        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
//        static VectorUtilities()
//        {
//            dataType           = DataType<TDataType>.GetKind();
//            executionSpace     = new TExecutionSpace();
//            executionSpaceType = ExecutionSpace<TExecutionSpace>.GetKind();


//            KokkosLibrary.Loaded += (sender,
//                                     e) =>
//                                    {
//                                        Load();
//                                    };

//            if(!KokkosLibrary.IsLoaded())
//            {
//                KokkosLibrary.Load();
//            }
//            else
//            {
//                Load();
//            }
//        }

//        [MethodImpl(MethodImplOptions.NoInlining)]
//        public static void Load()
//        {
//            if(KokkosLibrary.IsLoaded())
//            {
//                ulong errorCode;


//                nint vtablePtr = PlatformApi.NativeLibrary.GetExport(KokkosLibrary.ModuleHandle, vtable_Name(Mangling.GetType(TypeOfDataType), executionSpace.ToString()), out errorCode);

//                IVectorVTable vtable = new IVectorVTable();

//                Marshal.PtrToStructure(vtablePtr, vtable);


//                sync_host = Marshal.GetDelegateForFunctionPointer<sync_hostDelegate>(vtable.sync_hostPtr);


//                //vtable.sync_hostPtr = Marshal.GetFunctionPointerForDelegate<QueryInterfaceFn>(wrapper.QueryInterface);




//                //reserve = Marshal.GetDelegateForFunctionPointer<reserveDelegate>(PlatformApi.NativeLibrary.GetExport(KokkosLibrary.ModuleHandle,
//                //                                                                     reserve_Name(Mangling.GetType(TypeOfDataType), executionSpace.ToString()),
//                //                                                                     out errorCode));

//                //resize1 = Marshal.GetDelegateForFunctionPointer<resize1Delegate>(PlatformApi.NativeLibrary.GetExport(KokkosLibrary.ModuleHandle,
//                //                                                                     resize1_Name(Mangling.GetType(TypeOfDataType), executionSpace.ToString()),
//                //                                                                     out errorCode));

//                //resize2 = Marshal.GetDelegateForFunctionPointer<resize2Delegate>(PlatformApi.NativeLibrary.GetExport(KokkosLibrary.ModuleHandle,
//                //                                                                     resize2_Name(Mangling.GetType(TypeOfDataType), executionSpace.ToString()),
//                //                                                                     out errorCode));

//                //data = Marshal.GetDelegateForFunctionPointer<dataDelegate>(PlatformApi.NativeLibrary.GetExport(KokkosLibrary.ModuleHandle,
//                //                                                               data_Name(Mangling.GetType(TypeOfDataType), executionSpace.ToString()),
//                //                                                               out errorCode));

//                //is_allocated = Marshal.GetDelegateForFunctionPointer<is_allocatedDelegate>(PlatformApi.NativeLibrary.GetExport(KokkosLibrary.ModuleHandle,
//                //                                                                               is_allocated_Name(Mangling.GetType(TypeOfDataType), executionSpace.ToString()),
//                //                                                                               out errorCode));

//                //is_sorted = Marshal.GetDelegateForFunctionPointer<is_sortedDelegate>(PlatformApi.NativeLibrary.GetExport(KokkosLibrary.ModuleHandle,
//                //                                                                         is_sorted_Name(Mangling.GetType(TypeOfDataType), executionSpace.ToString()),
//                //                                                                         out errorCode));

//                //on_device = Marshal.GetDelegateForFunctionPointer<on_deviceDelegate>(PlatformApi.NativeLibrary.GetExport(KokkosLibrary.ModuleHandle,
//                //                                                                         on_device_Name(Mangling.GetType(TypeOfDataType), executionSpace.ToString()),
//                //                                                                         out errorCode));

//                //on_host = Marshal.GetDelegateForFunctionPointer<on_hostDelegate>(PlatformApi.NativeLibrary.GetExport(KokkosLibrary.ModuleHandle,
//                //                                                                     on_host_Name(Mangling.GetType(TypeOfDataType), executionSpace.ToString()),
//                //                                                                     out errorCode));

//                //device_to_host = Marshal.GetDelegateForFunctionPointer<device_to_hostDelegate>(PlatformApi.NativeLibrary.GetExport(KokkosLibrary.ModuleHandle,
//                //                                                                                   device_to_host_Name(Mangling.GetType(TypeOfDataType), executionSpace.ToString()),
//                //                                                                                   out errorCode));

//                //host_to_device = Marshal.GetDelegateForFunctionPointer<host_to_deviceDelegate>(PlatformApi.NativeLibrary.GetExport(KokkosLibrary.ModuleHandle,
//                //                                                                                   host_to_device_Name(Mangling.GetType(TypeOfDataType), executionSpace.ToString()),
//                //                                                                                   out errorCode));

//                //set_overallocation = Marshal.GetDelegateForFunctionPointer<set_overallocationDelegate>(PlatformApi.NativeLibrary.GetExport(KokkosLibrary.ModuleHandle,
//                //    set_overallocation_Name(Mangling.GetType(TypeOfDataType), executionSpace.ToString()),
//                //    out errorCode));

//                //assign = Marshal.GetDelegateForFunctionPointer<assignDelegate>(PlatformApi.NativeLibrary.GetExport(KokkosLibrary.ModuleHandle,
//                //                                                                   assign_Name(Mangling.GetType(TypeOfDataType), executionSpace.ToString()),
//                //                                                                   out errorCode));

//                //push_back = Marshal.GetDelegateForFunctionPointer<push_backDelegate>(PlatformApi.NativeLibrary.GetExport(KokkosLibrary.ModuleHandle,
//                //                                                                         push_back_Name(Mangling.GetType(TypeOfDataType), executionSpace.ToString()),
//                //                                                                         out errorCode));

//                //pop_back = Marshal.GetDelegateForFunctionPointer<pop_backDelegate>(PlatformApi.NativeLibrary.GetExport(KokkosLibrary.ModuleHandle,
//                //                                                                       pop_back_Name(Mangling.GetType(TypeOfDataType), executionSpace.ToString()),
//                //                                                                       out errorCode));

//                //insert1 = Marshal.GetDelegateForFunctionPointer<insertDelegate1>(PlatformApi.NativeLibrary.GetExport(KokkosLibrary.ModuleHandle,
//                //                                                                     insert1_Name(Mangling.GetType(TypeOfDataType), executionSpace.ToString()),
//                //                                                                     out errorCode));

//                //insert2 = Marshal.GetDelegateForFunctionPointer<insertDelegate2>(PlatformApi.NativeLibrary.GetExport(KokkosLibrary.ModuleHandle,
//                //                                                                     insert2_Name(Mangling.GetType(TypeOfDataType), executionSpace.ToString()),
//                //                                                                     out errorCode));

//                //clear = Marshal.GetDelegateForFunctionPointer<clearDelegate>(PlatformApi.NativeLibrary.GetExport(KokkosLibrary.ModuleHandle,
//                //                                                                 clear_Name(Mangling.GetType(TypeOfDataType), executionSpace.ToString()),
//                //                                                                 out errorCode));

//                //lower_bound = Marshal.GetDelegateForFunctionPointer<lower_boundDelegate>(PlatformApi.NativeLibrary.GetExport(KokkosLibrary.ModuleHandle,
//                //                                                                             lower_bound_Name(Mangling.GetType(TypeOfDataType), executionSpace.ToString()),
//                //                                                                             out errorCode));

//                //find = Marshal.GetDelegateForFunctionPointer<findDelegate>(PlatformApi.NativeLibrary.GetExport(KokkosLibrary.ModuleHandle,
//                //                                                               find_Name(Mangling.GetType(TypeOfDataType), executionSpace.ToString()),
//                //                                                               out errorCode));

//                //empty = Marshal.GetDelegateForFunctionPointer<emptyDelegate>(PlatformApi.NativeLibrary.GetExport(KokkosLibrary.ModuleHandle,
//                //                                                                 empty_Name(Mangling.GetType(TypeOfDataType), executionSpace.ToString()),
//                //                                                                 out errorCode));

//                //size = Marshal.GetDelegateForFunctionPointer<sizeDelegate>(PlatformApi.NativeLibrary.GetExport(KokkosLibrary.ModuleHandle,
//                //                                                               size_Name(Mangling.GetType(TypeOfDataType), executionSpace.ToString()),
//                //                                                               out errorCode));

//                //max_size = Marshal.GetDelegateForFunctionPointer<max_sizeDelegate>(PlatformApi.NativeLibrary.GetExport(KokkosLibrary.ModuleHandle,
//                //                                                                       max_size_Name(Mangling.GetType(TypeOfDataType), executionSpace.ToString()),
//                //                                                                       out errorCode));

//                //span = Marshal.GetDelegateForFunctionPointer<spanDelegate>(PlatformApi.NativeLibrary.GetExport(KokkosLibrary.ModuleHandle,
//                //                                                               span_Name(Mangling.GetType(TypeOfDataType), executionSpace.ToString()),
//                //                                                               out errorCode));

//                //begin = Marshal.GetDelegateForFunctionPointer<beginDelegate>(PlatformApi.NativeLibrary.GetExport(KokkosLibrary.ModuleHandle,
//                //                                                                 begin_Name(Mangling.GetType(TypeOfDataType), executionSpace.ToString()),
//                //                                                                 out errorCode));

//                //end = Marshal.GetDelegateForFunctionPointer<endDelegate>(PlatformApi.NativeLibrary.GetExport(KokkosLibrary.ModuleHandle,
//                //                                                             end_Name(Mangling.GetType(TypeOfDataType), executionSpace.ToString()),
//                //                                                             out errorCode));

//                //front = Marshal.GetDelegateForFunctionPointer<frontDelegate>(PlatformApi.NativeLibrary.GetExport(KokkosLibrary.ModuleHandle,
//                //                                                                 front1_Name(Mangling.GetType(TypeOfDataType), executionSpace.ToString()),
//                //                                                                 out errorCode));

//                //back = Marshal.GetDelegateForFunctionPointer<backDelegate>(PlatformApi.NativeLibrary.GetExport(KokkosLibrary.ModuleHandle,
//                //                                                               back1_Name(Mangling.GetType(TypeOfDataType), executionSpace.ToString()),
//                //                                                               out errorCode));

//                //@operator = Marshal.GetDelegateForFunctionPointer<operatorDelegate>(PlatformApi.NativeLibrary.GetExport(KokkosLibrary.ModuleHandle,
//                //                                                                        operator1_Name(Mangling.GetType(TypeOfDataType), executionSpace.ToString()),
//                //                                                                        out errorCode));

//                //operator_brackets =
//                //    Marshal.GetDelegateForFunctionPointer<operator_bracketsDelegate>(PlatformApi.NativeLibrary.GetExport(KokkosLibrary.ModuleHandle,
//                //                                                                         operator_brackets_Name(Mangling.GetType(TypeOfDataType), executionSpace.ToString()),
//                //                                                                         out errorCode));
//            }
//        }
//    }

//    [NonVersionable]
//    public sealed class Vector<TDataType, TExecutionSpace> : IDisposable
//        where TDataType : unmanaged
//        where TExecutionSpace : IExecutionSpace, new()
//    {
//        private static readonly DataTypeKind dataType;

//        private static readonly IExecutionSpace executionSpace;

//        private static readonly ExecutionSpaceKind executionSpaceType;

//        public NativePointer Pointer
//        {
//#if NETSTANDARD
//        [MethodImpl(MethodImplOptions.AggressiveInlining)]
//#else
//            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
//#endif
//            get;
//        }

//#if NETSTANDARD
//        [MethodImpl(MethodImplOptions.AggressiveInlining)]
//#else
//        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
//#endif
//        static Vector()
//        {
//            dataType           = DataType<TDataType>.GetKind();
//            executionSpace     = new TExecutionSpace();
//            executionSpaceType = ExecutionSpace<TExecutionSpace>.GetKind();

//            VectorUtilities<TDataType, TExecutionSpace>.Load();
//        }

//#if NETSTANDARD
//        [MethodImpl(MethodImplOptions.AggressiveInlining)]
//#else
//        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
//#endif
//        public Vector()
//        {


//            Pointer = new NativePointer(VectorUtilities<TDataType, TExecutionSpace>.create1(), Unsafe.SizeOf<IVectorVTable>());



//        }
//#if NETSTANDARD
//        [MethodImpl(MethodImplOptions.AggressiveInlining)]
//#else
//        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
//#endif
//        public Vector(NativePointer pointer)
//        {
//            Pointer = pointer;
//        }

//#if NETSTANDARD
//        [MethodImpl(MethodImplOptions.AggressiveInlining)]
//#else
//        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
//#endif
//        public Vector(int       n,
//                      TDataType val = default)
//        {
//            Pointer = new NativePointer();
//        }

//        public void Dispose()
//        {
//            Pointer?.Dispose();
//        }

//        public ref TDataType this[int i] { get { return ref VectorUtilities<TDataType, TExecutionSpace>.@operator(Pointer, i); } }

//        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
//        public void assign(ulong        n,
//                           in TDataType val)
//        {
//            VectorUtilities<TDataType, TExecutionSpace>.assign(Pointer, n, val);
//        }

//        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
//        public ref TDataType back()
//        {
//            return ref VectorUtilities<TDataType, TExecutionSpace>.back(Pointer);
//        }

//        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
//        public nint begin()
//        {
//            return VectorUtilities<TDataType, TExecutionSpace>.begin(Pointer);
//        }
        

//        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
//        public static nint Create()
//        {
//            return VectorUtilities<TDataType, TExecutionSpace>.create1();
//        }

//        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
//        public nint Create(int       n,
//                             TDataType value)
//        {
//            return VectorUtilities<TDataType, TExecutionSpace>.create2(n, value);
//        }

//        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
//        public void clear()
//        {
//            VectorUtilities<TDataType, TExecutionSpace>.clear(Pointer);
//        }

//        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
//        public nint data()
//        {
//            return VectorUtilities<TDataType, TExecutionSpace>.data(Pointer);
//        }

//        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
//        public void device_to_host()
//        {
//            VectorUtilities<TDataType, TExecutionSpace>.device_to_host(Pointer);
//        }

//        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
//        public bool empty()
//        {
//            return VectorUtilities<TDataType, TExecutionSpace>.empty(Pointer);
//        }

//        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
//        public nint end()
//        {
//            return VectorUtilities<TDataType, TExecutionSpace>.end(Pointer);
//        }

//        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
//        public nint find(TDataType val)
//        {
//            return VectorUtilities<TDataType, TExecutionSpace>.find(Pointer, val);
//        }

//        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
//        public ref TDataType front()
//        {
//            return ref VectorUtilities<TDataType, TExecutionSpace>.front(Pointer);
//        }

//        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
//        public void host_to_device()
//        {
//            VectorUtilities<TDataType, TExecutionSpace>.host_to_device(Pointer);
//        }

//        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
//        public nint insert(nint       it,
//                             in TDataType val)
//        {
//            return VectorUtilities<TDataType, TExecutionSpace>.insert1(Pointer, it, val);
//        }

//        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
//        public nint insert(nint       it,
//                             ulong        count,
//                             in TDataType val)
//        {
//            return VectorUtilities<TDataType, TExecutionSpace>.insert2(Pointer, it, count, val);
//        }

//        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
//        public bool is_allocated()
//        {
//            return VectorUtilities<TDataType, TExecutionSpace>.is_allocated(Pointer);
//        }

//        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
//        public bool is_sorted()
//        {
//            return VectorUtilities<TDataType, TExecutionSpace>.is_sorted(Pointer);
//        }

//        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
//        public ulong lower_bound(in ulong     start,
//                                 in ulong     theEnd,
//                                 in TDataType comp_val)
//        {
//            return VectorUtilities<TDataType, TExecutionSpace>.lower_bound(Pointer, start, theEnd, comp_val);
//        }

//        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
//        public ulong max_size()
//        {
//            return VectorUtilities<TDataType, TExecutionSpace>.max_size(Pointer);
//        }

//        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
//        public void on_device()
//        {
//            VectorUtilities<TDataType, TExecutionSpace>.on_device(Pointer);
//        }

//        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
//        public void on_host()
//        {
//            VectorUtilities<TDataType, TExecutionSpace>.on_host(Pointer);
//        }

//        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
//        public ref TDataType operator_brackets(int i)
//        {
//            return ref VectorUtilities<TDataType, TExecutionSpace>.operator_brackets(Pointer, i);
//        }

//        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
//        public ref TDataType @operator(int i)
//        {
//            return ref VectorUtilities<TDataType, TExecutionSpace>.@operator(Pointer, i);
//        }

//        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
//        public void pop_back()
//        {
//            VectorUtilities<TDataType, TExecutionSpace>.pop_back(Pointer);
//        }

//        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
//        public void push_back(TDataType val)
//        {
//            VectorUtilities<TDataType, TExecutionSpace>.push_back(Pointer, val);
//        }

//        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
//        public void reserve(ulong n)
//        {
//            VectorUtilities<TDataType, TExecutionSpace>.reserve(Pointer, n);
//        }

//        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
//        public void resize(ulong n)
//        {
//            VectorUtilities<TDataType, TExecutionSpace>.resize1(Pointer, n);
//        }

//        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
//        public void resize(ulong        n,
//                           in TDataType val)
//        {
//            VectorUtilities<TDataType, TExecutionSpace>.resize2(Pointer, n, val);
//        }

//        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
//        public void set_overallocation(float extra)
//        {
//            VectorUtilities<TDataType, TExecutionSpace>.set_overallocation(Pointer, extra);
//        }

//        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
//        public ulong size()
//        {
//            return VectorUtilities<TDataType, TExecutionSpace>.size(Pointer);
//        }

//        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
//        public ulong span()
//        {
//            return VectorUtilities<TDataType, TExecutionSpace>.span(Pointer);
//        }
//    }
//}