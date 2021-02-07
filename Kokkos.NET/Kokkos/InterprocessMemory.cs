using System;
using System.Runtime.CompilerServices;
using System.Runtime.Versioning;

using PlatformApi;

namespace Kokkos
{
    [NonVersionable]
    public sealed class InterprocessMemory<TExecutionSpace> : IDisposable
        where TExecutionSpace : IExecutionSpace, new()
    {
        private static readonly IExecutionSpace executionSpace;

        private static readonly ExecutionSpaceKind executionSpaceType;

        public NativePointer Pointer
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            get;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        static InterprocessMemory()
        {
            executionSpace     = new TExecutionSpace();
            executionSpaceType = ExecutionSpace<TExecutionSpace>.GetKind();
        }

        public InterprocessMemory(ulong  size,
                                  string label)
        {
            Pointer = new NativePointer(KokkosLibrary.IpcCreate(executionSpaceType, size, new NativeString<Serial>(label)), size);
        }

        public InterprocessMemory(nint memoryPtr,
                                  ulong size,
                                  string label)
        {
            Pointer = new NativePointer(KokkosLibrary.IpcCreateFrom(executionSpaceType, memoryPtr, size, new NativeString<Serial>(label)), size);
        }
        
        public InterprocessMemory(string label)
        {
            nint ptr = KokkosLibrary.IpcOpenExisting(executionSpaceType, new NativeString<Serial>(label));

            ulong size = KokkosLibrary.IpcGetSize(executionSpaceType, ptr);

            Pointer = new NativePointer(ptr, size);
        }
        
        ~InterprocessMemory()
        {
            Dispose();
        }

        public void Dispose()
        {
            KokkosLibrary.IpcClose(executionSpaceType, Pointer.Data);
            KokkosLibrary.IpcDestory(executionSpaceType, Pointer.Data);

            Pointer?.Dispose();

            GC.SuppressFinalize(this);
        }


        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public  nint GetMemoryPointer()
        {
            return KokkosLibrary.IpcGetMemoryPointer(executionSpaceType, Pointer.Data);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public  nint GetDeviceHandle()
        {
            return KokkosLibrary.IpcGetDeviceHandle(executionSpaceType, Pointer.Data);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public ulong GetSize()
        {
            return KokkosLibrary.IpcGetSize(executionSpaceType, Pointer.Data);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public View<TDataType, TExecutionSpace> MakeViewFromPointer<TDataType>(ulong n0,
                                                                               ulong n1 = ulong.MaxValue,
                                                                               ulong n2 = ulong.MaxValue,
                                                                               ulong n3 = ulong.MaxValue,
                                                                               ulong n4 = ulong.MaxValue,
                                                                               ulong n5 = ulong.MaxValue,
                                                                               ulong n6 = ulong.MaxValue,
                                                                               ulong n7 = ulong.MaxValue)
            where TDataType : struct
        {
            int dataTypeSize = Unsafe.SizeOf<TDataType>();

            DataTypeKind dataType = DataType<TDataType>.GetKind();

            nint viewPtr = KokkosLibrary.IpcMakeViewFromPointer(executionSpaceType, dataType, Pointer, n0, n1, n2, n3, n4, n5, n6, n7);

            ushort rank = (ushort)KokkosLibrary.CalculateRank(n0, n1, n2, n3, n4, n5, n6, n7);

            NdArray ndArray = View<TDataType, TExecutionSpace>.RcpConvert(viewPtr, rank);

            ulong size = KokkosLibrary.CalculateSize(ndArray[0].Dim, ndArray[1].Dim, ndArray[2].Dim, ndArray[3].Dim, ndArray[4].Dim, ndArray[5].Dim, ndArray[6].Dim, ndArray[7].Dim);

            View<TDataType, TExecutionSpace> view = new View<TDataType, TExecutionSpace>(new NativePointer(viewPtr, (ulong)dataTypeSize * size), ndArray);

            return view;
        }

        //public static IpcCreateDelegate              IpcCreate;
        //public static IpcCreateFromDelegate          IpcCreateFrom;
        //public static IpcOpenExistingDelegate        IpcOpenExisting;
        //public static IpcDestoryDelegate             IpcDestory;
        //public static IpcCloseDelegate               IpcClose;
        //public static IpcGetMemoryPointerDelegate    IpcGetMemoryPointer;
        //public static IpcGetDeviceHandleDelegate     IpcGetDeviceHandle;
        //public static IpcGetSizeDelegate             IpcGetSize;
        //public static IpcMakeViewFromPointerDelegate IpcMakeViewFromPointer;
        //public static IpcMakeViewFromHandleDelegate  IpcMakeViewFromHandle;
    }
}