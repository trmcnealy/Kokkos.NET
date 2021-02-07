using PlatformApi;

using System;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Versioning;

namespace Kokkos
{
    [NonVersionable]
    [StructLayout(System.Runtime.InteropServices.LayoutKind.Explicit, Pack = 2, Size = 160)]
    public unsafe struct NdArray
    {
        [FieldOffset(0)]
        public DataTypeKind DataType;

        [FieldOffset(2)]
        public ushort Rank;

        [FieldOffset(4)]
        public LayoutKind Layout;

        [FieldOffset(6)]
        public ExecutionSpaceKind ExecutionSpace;

        [FieldOffset(8)]
        public fixed ulong Dims[8];

        [FieldOffset(72)]
        public fixed ulong Strides[8];

        [FieldOffset(136)]
        public nint Data;

        //[MarshalAs(UnmanagedType.LPUTF8Str)] string
        [FieldOffset(144)]
        public NativeString<Serial> Label;

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public NdArray(DataTypeKind       dataType,
                       ushort             rank,
                       LayoutKind         layout,
                       ExecutionSpaceKind executionSpace,
                       string             label,
                       ulong              n0 = ulong.MaxValue,
                       ulong              n1 = ulong.MaxValue,
                       ulong              n2 = ulong.MaxValue,
                       ulong              n3 = ulong.MaxValue,
                       ulong              n4 = ulong.MaxValue,
                       ulong              n5 = ulong.MaxValue,
                       ulong              n6 = ulong.MaxValue,
                       ulong              n7 = ulong.MaxValue)
        {
            DataType       = dataType;
            Rank           = rank;
            Layout         = layout;
            ExecutionSpace = executionSpace;
            Data           = (nint)0;

            Label = new NativeString<Serial>(label);

            Dims[0] = n0 == ulong.MaxValue ? 0 : n0;
            Dims[1] = n1 == ulong.MaxValue ? 0 : n1;
            Dims[2] = n2 == ulong.MaxValue ? 0 : n2;
            Dims[3] = n3 == ulong.MaxValue ? 0 : n3;
            Dims[4] = n4 == ulong.MaxValue ? 0 : n4;
            Dims[5] = n5 == ulong.MaxValue ? 0 : n5;
            Dims[6] = n6 == ulong.MaxValue ? 0 : n6;
            Dims[7] = n7 == ulong.MaxValue ? 0 : n7;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public ulong Extent(uint rank)
        {
            return Dims[rank];
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public ulong Stride(uint rank)
        {
            return Strides[rank];
        }

        #region Indices

        public (ulong Dim, ulong Stride) this[int i0]
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            get { return (Dims[i0], Strides[i0]); }
        }

        public (ulong Dim, ulong Stride) this[uint i0]
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            get { return (Dims[i0], Strides[i0]); }
        }

        public (ulong Dim, ulong Stride) this[long i0]
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            get { return (Dims[i0], Strides[i0]); }
        }

        public (ulong Dim, ulong Stride) this[ulong i0]
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            get { return (Dims[i0], Strides[i0]); }
        }

        #endregion
    }

    [NonVersionable]
    public abstract class View : IDisposable
    {
        public NativePointer Pointer
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            get;
        }

        public NdArray NdArray
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            get;
        }

        public uint Rank
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            get { return NdArray.Rank; }
        }

        //[MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        //static View()
        //{
        //    if(!KokkosLibrary.IsInitialized())
        //    {
        //        KokkosLibraryException.Throw("Kokkos Library is not initialized. Use ParallelProcessor.Initialize/Shutdown in the main routine/thread.");
        //    }
        //}

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        protected View(NativePointer pointer,
                       NdArray       ndArray)
        {
            Pointer = pointer;
            NdArray = ndArray;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        protected View(DataTypeKind       dataType,
                       ushort             rank,
                       LayoutKind         layout,
                       ExecutionSpaceKind executionSpace,
                       string             label)
        {
            Pointer = new NativePointer();

            NdArray ndArray = new(dataType, rank, layout, executionSpace, label);

            KokkosLibrary.CreateView(Pointer, ref ndArray);

            NdArray = ndArray;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        protected View(DataTypeKind       dataType,
                       ushort             rank,
                       LayoutKind         layout,
                       ExecutionSpaceKind executionSpace,
                       string             label,
                       ulong              n0,
                       ulong              n1 = ulong.MaxValue,
                       ulong              n2 = ulong.MaxValue,
                       ulong              n3 = ulong.MaxValue,
                       ulong              n4 = ulong.MaxValue,
                       ulong              n5 = ulong.MaxValue,
                       ulong              n6 = ulong.MaxValue,
                       ulong              n7 = ulong.MaxValue)
        {
            Pointer = new NativePointer();

            NdArray ndArray = new NdArray(dataType, rank, layout, executionSpace, label, n0, n1, n2, n3, n4, n5, n6, n7);

            KokkosLibrary.CreateView(Pointer, ref ndArray);

            NdArray = ndArray;
        }
        
        ~View()
        {
        }

        public void Dispose()
        {
            Pointer.Dispose();
            GC.SuppressFinalize(this);
        }
    }

    public interface IView<TDataType>
        where TDataType : struct
    {
        TDataType this[int i0]
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            get;
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            set;
        }

        TDataType this[int i0,
                       int i1]
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            get;
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            set;
        }

        TDataType this[int i0,
                       int i1,
                       int i2]
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            get;
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            set;
        }

        TDataType this[int i0,
                       int i1,
                       int i2,
                       int i3]
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            get;
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            set;
        }

        TDataType this[int i0,
                       int i1,
                       int i2,
                       int i3,
                       int i4]
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            get;
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            set;
        }

        TDataType this[int i0,
                       int i1,
                       int i2,
                       int i3,
                       int i4,
                       int i5]
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            get;
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            set;
        }

        TDataType this[int i0,
                       int i1,
                       int i2,
                       int i3,
                       int i4,
                       int i5,
                       int i6]
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            get;
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            set;
        }

        TDataType this[int i0,
                       int i1,
                       int i2,
                       int i3,
                       int i4,
                       int i5,
                       int i6,
                       int i7]
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            get;
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            set;
        }

        TDataType this[uint i0]
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            get;
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            set;
        }

        TDataType this[uint i0,
                       uint i1]
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            get;
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            set;
        }

        TDataType this[uint i0,
                       uint i1,
                       uint i2]
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            get;
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            set;
        }

        TDataType this[uint i0,
                       uint i1,
                       uint i2,
                       uint i3]
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            get;
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            set;
        }

        TDataType this[uint i0,
                       uint i1,
                       uint i2,
                       uint i3,
                       uint i4]
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            get;
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            set;
        }

        TDataType this[uint i0,
                       uint i1,
                       uint i2,
                       uint i3,
                       uint i4,
                       uint i5]
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            get;
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            set;
        }

        TDataType this[uint i0,
                       uint i1,
                       uint i2,
                       uint i3,
                       uint i4,
                       uint i5,
                       uint i6]
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            get;
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            set;
        }

        TDataType this[uint i0,
                       uint i1,
                       uint i2,
                       uint i3,
                       uint i4,
                       uint i5,
                       uint i6,
                       uint i7]
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            get;
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            set;
        }

        TDataType this[long i0]
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            get;
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            set;
        }

        TDataType this[long i0,
                       long i1]
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            get;
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            set;
        }

        TDataType this[long i0,
                       long i1,
                       long i2]
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            get;
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            set;
        }

        TDataType this[long i0,
                       long i1,
                       long i2,
                       long i3]
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            get;
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            set;
        }

        TDataType this[long i0,
                       long i1,
                       long i2,
                       long i3,
                       long i4]
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            get;
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            set;
        }

        TDataType this[long i0,
                       long i1,
                       long i2,
                       long i3,
                       long i4,
                       long i5]
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            get;
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            set;
        }

        TDataType this[long i0,
                       long i1,
                       long i2,
                       long i3,
                       long i4,
                       long i5,
                       long i6]
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            get;
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            set;
        }

        TDataType this[long i0,
                       long i1,
                       long i2,
                       long i3,
                       long i4,
                       long i5,
                       long i6,
                       long i7]
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            get;
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            set;
        }

        TDataType this[ulong i0]
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            get;
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            set;
        }

        TDataType this[ulong i0,
                       ulong i1]
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            get;
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            set;
        }

        TDataType this[ulong i0,
                       ulong i1,
                       ulong i2]
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            get;
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            set;
        }

        TDataType this[ulong i0,
                       ulong i1,
                       ulong i2,
                       ulong i3]
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            get;
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            set;
        }

        TDataType this[ulong i0,
                       ulong i1,
                       ulong i2,
                       ulong i3,
                       ulong i4]
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            get;
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            set;
        }

        TDataType this[ulong i0,
                       ulong i1,
                       ulong i2,
                       ulong i3,
                       ulong i4,
                       ulong i5]
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            get;
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            set;
        }

        TDataType this[ulong i0,
                       ulong i1,
                       ulong i2,
                       ulong i3,
                       ulong i4,
                       ulong i5,
                       ulong i6]
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            get;
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            set;
        }

        TDataType this[ulong i0,
                       ulong i1,
                       ulong i2,
                       ulong i3,
                       ulong i4,
                       ulong i5,
                       ulong i6,
                       ulong i7]
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            get;
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            set;
        }

        uint Rank
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            get;
        }

        string Label();

        ulong Size();

        ulong Stride(uint dim);

        ulong Extent(uint dim);

        void CopyTo(TDataType[] values);
    }

    [NonVersionable]
    public sealed class View<TDataType, TExecutionSpace> : View, IView<TDataType>
        where TDataType : struct
        where TExecutionSpace : IExecutionSpace, new()
    {
        //private static readonly int dataTypeSize = Unsafe.SizeOf<TDataType>();

        private static readonly DataTypeKind dataType;

        private static readonly IExecutionSpace executionSpace;

        private static readonly ExecutionSpaceKind executionSpaceType;

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        static View()
        {
            dataType           = DataType<TDataType>.GetKind();
            executionSpace     = new TExecutionSpace();
            executionSpaceType = ExecutionSpace<TExecutionSpace>.GetKind();
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public View(NativePointer pointer,
                    NdArray       ndArray)
            : base(pointer, ndArray)
        {
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public View(string label)
            : base(dataType, 0, ExecutionSpace<TExecutionSpace>.GetLayout(), executionSpaceType, label)
        {
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public View(string label,
                    int    n0,
                    int    n1 = int.MaxValue,
                    int    n2 = int.MaxValue,
                    int    n3 = int.MaxValue,
                    int    n4 = int.MaxValue,
                    int    n5 = int.MaxValue,
                    int    n6 = int.MaxValue,
                    int    n7 = int.MaxValue)
            : base(dataType,
                   GetRank(n0, n1, n2, n3, n4, n5, n6, n7),
                   ExecutionSpace<TExecutionSpace>.GetLayout(),
                   executionSpaceType,
                   label,
                   (ulong)n0,
                   n1 == int.MaxValue ? ulong.MaxValue : (ulong)n1,
                   n2 == int.MaxValue ? ulong.MaxValue : (ulong)n2,
                   n3 == int.MaxValue ? ulong.MaxValue : (ulong)n3,
                   n4 == int.MaxValue ? ulong.MaxValue : (ulong)n4,
                   n5 == int.MaxValue ? ulong.MaxValue : (ulong)n5,
                   n6 == int.MaxValue ? ulong.MaxValue : (ulong)n6,
                   n7 == int.MaxValue ? ulong.MaxValue : (ulong)n7)
        {
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public View(string label,
                    uint   n0,
                    uint   n1 = uint.MaxValue,
                    uint   n2 = uint.MaxValue,
                    uint   n3 = uint.MaxValue,
                    uint   n4 = uint.MaxValue,
                    uint   n5 = uint.MaxValue,
                    uint   n6 = uint.MaxValue,
                    uint   n7 = uint.MaxValue)
            : base(dataType,
                   GetRank(n0, n1, n2, n3, n4, n5, n6, n7),
                   ExecutionSpace<TExecutionSpace>.GetLayout(),
                   executionSpaceType,
                   label,
                   n0,
                   n1 == uint.MaxValue ? ulong.MaxValue : n1,
                   n2 == uint.MaxValue ? ulong.MaxValue : n2,
                   n3 == uint.MaxValue ? ulong.MaxValue : n3,
                   n4 == uint.MaxValue ? ulong.MaxValue : n4,
                   n5 == uint.MaxValue ? ulong.MaxValue : n5,
                   n6 == uint.MaxValue ? ulong.MaxValue : n6,
                   n7 == uint.MaxValue ? ulong.MaxValue : n7)
        {
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public View(string label,
                    long   n0,
                    long   n1 = long.MaxValue,
                    long   n2 = long.MaxValue,
                    long   n3 = long.MaxValue,
                    long   n4 = long.MaxValue,
                    long   n5 = long.MaxValue,
                    long   n6 = long.MaxValue,
                    long   n7 = long.MaxValue)
            : base(dataType,
                   GetRank(n0, n1, n2, n3, n4, n5, n6, n7),
                   ExecutionSpace<TExecutionSpace>.GetLayout(),
                   executionSpaceType,
                   label,
                   (ulong)n0,
                   n1 == long.MaxValue ? ulong.MaxValue : (ulong)n1,
                   n2 == long.MaxValue ? ulong.MaxValue : (ulong)n2,
                   n3 == long.MaxValue ? ulong.MaxValue : (ulong)n3,
                   n4 == long.MaxValue ? ulong.MaxValue : (ulong)n4,
                   n5 == long.MaxValue ? ulong.MaxValue : (ulong)n5,
                   n6 == long.MaxValue ? ulong.MaxValue : (ulong)n6,
                   n7 == long.MaxValue ? ulong.MaxValue : (ulong)n7)
        {
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public View(string label,
                    ulong  n0,
                    ulong  n1 = ulong.MaxValue,
                    ulong  n2 = ulong.MaxValue,
                    ulong  n3 = ulong.MaxValue,
                    ulong  n4 = ulong.MaxValue,
                    ulong  n5 = ulong.MaxValue,
                    ulong  n6 = ulong.MaxValue,
                    ulong  n7 = ulong.MaxValue)
            : base(dataType, GetRank(n0, n1, n2, n3, n4, n5, n6, n7), ExecutionSpace<TExecutionSpace>.GetLayout(), executionSpaceType, label, n0, n1, n2, n3, n4, n5, n6, n7)
        {
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public string Label()
        {
            return KokkosLibrary.GetLabel(Pointer, NdArray).ToString();
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public ulong Size()
        {
            return KokkosLibrary.GetSize(Pointer, NdArray);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public ulong Stride(uint dim)
        {
            return KokkosLibrary.GetStride(Pointer, NdArray, dim);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public ulong Extent(uint dim)
        {
            return KokkosLibrary.GetExtent(Pointer, NdArray, dim);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public void CopyTo(TDataType[] values)
        {
            //GCHandle handle = GCHandle.Alloc(values[0],
            //                                 GCHandleType.Pinned);

            ValueType[] valueTypes = Array.ConvertAll(values, ValueType.From);

            KokkosLibrary.CopyTo(Pointer, NdArray, valueTypes);

            //handle.AddrOfPinnedObject()handle.Free();
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static NdArray RcpConvert(nint   rcp_view_ptr,
                                         ushort rank)
        {
            KokkosLibrary.RcpViewToNdArray(rcp_view_ptr, executionSpaceType, executionSpace.DefaultLayout, dataType, rank, out NdArray ndArray);

            return ndArray;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static NdArray Convert(nint   view_ptr,
                                      ushort rank)
        {
            return KokkosLibrary.ViewToNdArray(view_ptr, executionSpaceType, executionSpace.DefaultLayout, dataType, rank);
        }

        #region GetRank

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        private static ushort GetRank(int n0,
                                      int n1,
                                      int n2,
                                      int n3,
                                      int n4,
                                      int n5,
                                      int n6,
                                      int n7)
        {
            if(n0 == int.MaxValue)
            {
                return 0;
            }

            if(n1 == int.MaxValue)
            {
                return 1;
            }

            if(n2 == int.MaxValue)
            {
                return 2;
            }

            if(n3 == int.MaxValue)
            {
                return 3;
            }

            if(n4 == int.MaxValue)
            {
                return 4;
            }

            if(n5 == int.MaxValue)
            {
                return 5;
            }

            if(n6 == int.MaxValue)
            {
                return 6;
            }

            if(n7 == int.MaxValue)
            {
                return 7;
            }

            return 8;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        private static ushort GetRank(uint n0,
                                      uint n1,
                                      uint n2,
                                      uint n3,
                                      uint n4,
                                      uint n5,
                                      uint n6,
                                      uint n7)
        {
            if(n0 == uint.MaxValue)
            {
                return 0;
            }

            if(n1 == uint.MaxValue)
            {
                return 1;
            }

            if(n2 == uint.MaxValue)
            {
                return 2;
            }

            if(n3 == uint.MaxValue)
            {
                return 3;
            }

            if(n4 == uint.MaxValue)
            {
                return 4;
            }

            if(n5 == uint.MaxValue)
            {
                return 5;
            }

            if(n6 == uint.MaxValue)
            {
                return 6;
            }

            if(n7 == uint.MaxValue)
            {
                return 7;
            }

            return 8;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        private static ushort GetRank(long n0,
                                      long n1,
                                      long n2,
                                      long n3,
                                      long n4,
                                      long n5,
                                      long n6,
                                      long n7)
        {
            if(n0 == long.MaxValue)
            {
                return 0;
            }

            if(n1 == long.MaxValue)
            {
                return 1;
            }

            if(n2 == long.MaxValue)
            {
                return 2;
            }

            if(n3 == long.MaxValue)
            {
                return 3;
            }

            if(n4 == long.MaxValue)
            {
                return 4;
            }

            if(n5 == long.MaxValue)
            {
                return 5;
            }

            if(n6 == long.MaxValue)
            {
                return 6;
            }

            if(n7 == long.MaxValue)
            {
                return 7;
            }

            return 8;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        private static ushort GetRank(ulong n0,
                                      ulong n1,
                                      ulong n2,
                                      ulong n3,
                                      ulong n4,
                                      ulong n5,
                                      ulong n6,
                                      ulong n7)
        {
            if(n0 == ulong.MaxValue)
            {
                return 0;
            }

            if(n1 == ulong.MaxValue)
            {
                return 1;
            }

            if(n2 == ulong.MaxValue)
            {
                return 2;
            }

            if(n3 == ulong.MaxValue)
            {
                return 3;
            }

            if(n4 == ulong.MaxValue)
            {
                return 4;
            }

            if(n5 == ulong.MaxValue)
            {
                return 5;
            }

            if(n6 == ulong.MaxValue)
            {
                return 6;
            }

            if(n7 == ulong.MaxValue)
            {
                return 7;
            }

            return 8;
        }

        #endregion

        #region int Indices

        public TDataType this[int i0]
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            get
            {
                ValueType valuePtr = KokkosLibrary.GetValue(Pointer, NdArray, (ulong)i0);

                return valuePtr.As<TDataType>();
            }
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            set
            {
                //GCHandle handle = GCHandle.ToIntPtr(value);

                //byte* pvt = stackalloc byte[dataTypeSze];

                //nint valuePtr = new nint(pvt);
                ////Marshal.AllocHGlobal(Marshal.SizeOf(value));

                //Marshal.GetNativeVariantForObject(value,
                //                                  valuePtr);

                KokkosLibrary.SetValue(Pointer, NdArray, ValueType.From(value), (ulong)i0);

                //handle.Free();
            }
        }

        public TDataType this[int i0,
                              int i1]
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            get
            {
                ValueType valuePtr = KokkosLibrary.GetValue(Pointer, NdArray, (ulong)i0, (ulong)i1);

                return valuePtr.As<TDataType>();
            }
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            set { KokkosLibrary.SetValue(Pointer, NdArray, ValueType.From(value), (ulong)i0, (ulong)i1); }
        }

        public TDataType this[int i0,
                              int i1,
                              int i2]
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            get
            {
                ValueType valuePtr = KokkosLibrary.GetValue(Pointer, NdArray, (ulong)i0, (ulong)i1, (ulong)i2);

                return valuePtr.As<TDataType>();
            }
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            set { KokkosLibrary.SetValue(Pointer, NdArray, ValueType.From(value), (ulong)i0, (ulong)i1, (ulong)i2); }
        }

        public TDataType this[int i0,
                              int i1,
                              int i2,
                              int i3]
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            get
            {
                ValueType valuePtr = KokkosLibrary.GetValue(Pointer, NdArray, (ulong)i0, (ulong)i1, (ulong)i2, (ulong)i3);

                return valuePtr.As<TDataType>();
            }
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            set { KokkosLibrary.SetValue(Pointer, NdArray, ValueType.From(value), (ulong)i0, (ulong)i1, (ulong)i2, (ulong)i3); }
        }

        public TDataType this[int i0,
                              int i1,
                              int i2,
                              int i3,
                              int i4]
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            get
            {
                ValueType valuePtr = KokkosLibrary.GetValue(Pointer, NdArray, (ulong)i0, (ulong)i1, (ulong)i2, (ulong)i3, (ulong)i4);

                return valuePtr.As<TDataType>();
            }
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            set { KokkosLibrary.SetValue(Pointer, NdArray, ValueType.From(value), (ulong)i0, (ulong)i1, (ulong)i2, (ulong)i3, (ulong)i4); }
        }

        public TDataType this[int i0,
                              int i1,
                              int i2,
                              int i3,
                              int i4,
                              int i5]
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            get
            {
                ValueType valuePtr = KokkosLibrary.GetValue(Pointer, NdArray, (ulong)i0, (ulong)i1, (ulong)i2, (ulong)i3, (ulong)i4, (ulong)i5);

                return valuePtr.As<TDataType>();
            }
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            set { KokkosLibrary.SetValue(Pointer, NdArray, ValueType.From(value), (ulong)i0, (ulong)i1, (ulong)i2, (ulong)i3, (ulong)i4, (ulong)i5); }
        }

        public TDataType this[int i0,
                              int i1,
                              int i2,
                              int i3,
                              int i4,
                              int i5,
                              int i6]
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            get
            {
                ValueType valuePtr = KokkosLibrary.GetValue(Pointer, NdArray, (ulong)i0, (ulong)i1, (ulong)i2, (ulong)i3, (ulong)i4, (ulong)i5, (ulong)i6);

                return valuePtr.As<TDataType>();
            }
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            set { KokkosLibrary.SetValue(Pointer, NdArray, ValueType.From(value), (ulong)i0, (ulong)i1, (ulong)i2, (ulong)i3, (ulong)i4, (ulong)i5, (ulong)i6); }
        }

        public TDataType this[int i0,
                              int i1,
                              int i2,
                              int i3,
                              int i4,
                              int i5,
                              int i6,
                              int i7]
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            get
            {
                ValueType valuePtr = KokkosLibrary.GetValue(Pointer, NdArray, (ulong)i0, (ulong)i1, (ulong)i2, (ulong)i3, (ulong)i4, (ulong)i5, (ulong)i6, (ulong)i7);

                return valuePtr.As<TDataType>();
            }
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            set { KokkosLibrary.SetValue(Pointer, NdArray, ValueType.From(value), (ulong)i0, (ulong)i1, (ulong)i2, (ulong)i3, (ulong)i4, (ulong)i5, (ulong)i6, (ulong)i7); }
        }

        #endregion

        #region uint Indices

        public TDataType this[uint i0]
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            get
            {
                ValueType valuePtr = KokkosLibrary.GetValue(Pointer, NdArray, i0);

                return valuePtr.As<TDataType>();
            }
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            set
            {
                //GCHandle handle = GCHandle.ToIntPtr(value);

                //byte* pvt = stackalloc byte[dataTypeSze];

                //nint valuePtr = new nint(pvt);
                ////Marshal.AllocHGlobal(Marshal.SizeOf(value));

                //Marshal.GetNativeVariantForObject(value,
                //                                  valuePtr);

                KokkosLibrary.SetValue(Pointer, NdArray, ValueType.From(value), i0);

                //handle.Free();
            }
        }

        public TDataType this[uint i0,
                              uint i1]
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            get
            {
                ValueType valuePtr = KokkosLibrary.GetValue(Pointer, NdArray, i0, i1);

                return valuePtr.As<TDataType>();
            }
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            set { KokkosLibrary.SetValue(Pointer, NdArray, ValueType.From(value), i0, i1); }
        }

        public TDataType this[uint i0,
                              uint i1,
                              uint i2]
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            get
            {
                ValueType valuePtr = KokkosLibrary.GetValue(Pointer, NdArray, i0, i1, i2);

                return valuePtr.As<TDataType>();
            }
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            set { KokkosLibrary.SetValue(Pointer, NdArray, ValueType.From(value), i0, i1, i2); }
        }

        public TDataType this[uint i0,
                              uint i1,
                              uint i2,
                              uint i3]
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            get
            {
                ValueType valuePtr = KokkosLibrary.GetValue(Pointer, NdArray, i0, i1, i2, i3);

                return valuePtr.As<TDataType>();
            }
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            set { KokkosLibrary.SetValue(Pointer, NdArray, ValueType.From(value), i0, i1, i2, i3); }
        }

        public TDataType this[uint i0,
                              uint i1,
                              uint i2,
                              uint i3,
                              uint i4]
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            get
            {
                ValueType valuePtr = KokkosLibrary.GetValue(Pointer, NdArray, i0, i1, i2, i3, i4);

                return valuePtr.As<TDataType>();
            }
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            set { KokkosLibrary.SetValue(Pointer, NdArray, ValueType.From(value), i0, i1, i2, i3, i4); }
        }

        public TDataType this[uint i0,
                              uint i1,
                              uint i2,
                              uint i3,
                              uint i4,
                              uint i5]
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            get
            {
                ValueType valuePtr = KokkosLibrary.GetValue(Pointer, NdArray, i0, i1, i2, i3, i4, i5);

                return valuePtr.As<TDataType>();
            }
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            set { KokkosLibrary.SetValue(Pointer, NdArray, ValueType.From(value), i0, i1, i2, i3, i4, i5); }
        }

        public TDataType this[uint i0,
                              uint i1,
                              uint i2,
                              uint i3,
                              uint i4,
                              uint i5,
                              uint i6]
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            get
            {
                ValueType valuePtr = KokkosLibrary.GetValue(Pointer, NdArray, i0, i1, i2, i3, i4, i5, i6);

                return valuePtr.As<TDataType>();
            }
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            set { KokkosLibrary.SetValue(Pointer, NdArray, ValueType.From(value), i0, i1, i2, i3, i4, i5, i6); }
        }

        public TDataType this[uint i0,
                              uint i1,
                              uint i2,
                              uint i3,
                              uint i4,
                              uint i5,
                              uint i6,
                              uint i7]
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            get
            {
                ValueType valuePtr = KokkosLibrary.GetValue(Pointer, NdArray, i0, i1, i2, i3, i4, i5, i6, i7);

                return valuePtr.As<TDataType>();
            }
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            set { KokkosLibrary.SetValue(Pointer, NdArray, ValueType.From(value), i0, i1, i2, i3, i4, i5, i6, i7); }
        }

        #endregion

        #region long Indices

        public TDataType this[long i0]
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            get
            {
                ValueType valuePtr = KokkosLibrary.GetValue(Pointer, NdArray, (ulong)i0);

                return valuePtr.As<TDataType>();
            }
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            set
            {
                //GCHandle handle = GCHandle.ToIntPtr(value);

                //byte* pvt = stackalloc byte[dataTypeSze];

                //nint valuePtr = new nint(pvt);
                ////Marshal.AllocHGlobal(Marshal.SizeOf(value));

                //Marshal.GetNativeVariantForObject(value,
                //                                  valuePtr);

                KokkosLibrary.SetValue(Pointer, NdArray, ValueType.From(value), (ulong)i0);

                //handle.Free();
            }
        }

        public TDataType this[long i0,
                              long i1]
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            get
            {
                ValueType valuePtr = KokkosLibrary.GetValue(Pointer, NdArray, (ulong)i0, (ulong)i1);

                return valuePtr.As<TDataType>();
            }
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            set { KokkosLibrary.SetValue(Pointer, NdArray, ValueType.From(value), (ulong)i0, (ulong)i1); }
        }

        public TDataType this[long i0,
                              long i1,
                              long i2]
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            get
            {
                ValueType valuePtr = KokkosLibrary.GetValue(Pointer, NdArray, (ulong)i0, (ulong)i1, (ulong)i2);

                return valuePtr.As<TDataType>();
            }
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            set { KokkosLibrary.SetValue(Pointer, NdArray, ValueType.From(value), (ulong)i0, (ulong)i1, (ulong)i2); }
        }

        public TDataType this[long i0,
                              long i1,
                              long i2,
                              long i3]
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            get
            {
                ValueType valuePtr = KokkosLibrary.GetValue(Pointer, NdArray, (ulong)i0, (ulong)i1, (ulong)i2, (ulong)i3);

                return valuePtr.As<TDataType>();
            }
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            set { KokkosLibrary.SetValue(Pointer, NdArray, ValueType.From(value), (ulong)i0, (ulong)i1, (ulong)i2, (ulong)i3); }
        }

        public TDataType this[long i0,
                              long i1,
                              long i2,
                              long i3,
                              long i4]
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            get
            {
                ValueType valuePtr = KokkosLibrary.GetValue(Pointer, NdArray, (ulong)i0, (ulong)i1, (ulong)i2, (ulong)i3, (ulong)i4);

                return valuePtr.As<TDataType>();
            }
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            set { KokkosLibrary.SetValue(Pointer, NdArray, ValueType.From(value), (ulong)i0, (ulong)i1, (ulong)i2, (ulong)i3, (ulong)i4); }
        }

        public TDataType this[long i0,
                              long i1,
                              long i2,
                              long i3,
                              long i4,
                              long i5]
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            get
            {
                ValueType valuePtr = KokkosLibrary.GetValue(Pointer, NdArray, (ulong)i0, (ulong)i1, (ulong)i2, (ulong)i3, (ulong)i4, (ulong)i5);

                return valuePtr.As<TDataType>();
            }
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            set { KokkosLibrary.SetValue(Pointer, NdArray, ValueType.From(value), (ulong)i0, (ulong)i1, (ulong)i2, (ulong)i3, (ulong)i4, (ulong)i5); }
        }

        public TDataType this[long i0,
                              long i1,
                              long i2,
                              long i3,
                              long i4,
                              long i5,
                              long i6]
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            get
            {
                ValueType valuePtr = KokkosLibrary.GetValue(Pointer, NdArray, (ulong)i0, (ulong)i1, (ulong)i2, (ulong)i3, (ulong)i4, (ulong)i5, (ulong)i6);

                return valuePtr.As<TDataType>();
            }
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            set { KokkosLibrary.SetValue(Pointer, NdArray, ValueType.From(value), (ulong)i0, (ulong)i1, (ulong)i2, (ulong)i3, (ulong)i4, (ulong)i5, (ulong)i6); }
        }

        public TDataType this[long i0,
                              long i1,
                              long i2,
                              long i3,
                              long i4,
                              long i5,
                              long i6,
                              long i7]
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            get
            {
                ValueType valuePtr = KokkosLibrary.GetValue(Pointer, NdArray, (ulong)i0, (ulong)i1, (ulong)i2, (ulong)i3, (ulong)i4, (ulong)i5, (ulong)i6, (ulong)i7);

                return valuePtr.As<TDataType>();
            }
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            set { KokkosLibrary.SetValue(Pointer, NdArray, ValueType.From(value), (ulong)i0, (ulong)i1, (ulong)i2, (ulong)i3, (ulong)i4, (ulong)i5, (ulong)i6, (ulong)i7); }
        }

        #endregion

        #region ulong Indices

        public TDataType this[ulong i0]
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            get
            {
                ValueType valuePtr = KokkosLibrary.GetValue(Pointer, NdArray, i0);

                return valuePtr.As<TDataType>();
            }
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            set
            {
                //GCHandle handle = GCHandle.ToIntPtr(value);

                //byte* pvt = stackalloc byte[dataTypeSze];

                //nint valuePtr = new nint(pvt);
                ////Marshal.AllocHGlobal(Marshal.SizeOf(value));

                //Marshal.GetNativeVariantForObject(value,
                //                                  valuePtr);

                KokkosLibrary.SetValue(Pointer, NdArray, ValueType.From(value), i0);

                //handle.Free();
            }
        }

        public TDataType this[ulong i0,
                              ulong i1]
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            get
            {
                ValueType valuePtr = KokkosLibrary.GetValue(Pointer, NdArray, i0, i1);

                return valuePtr.As<TDataType>();
            }
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            set { KokkosLibrary.SetValue(Pointer, NdArray, ValueType.From(value), i0, i1); }
        }

        public TDataType this[ulong i0,
                              ulong i1,
                              ulong i2]
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            get
            {
                ValueType valuePtr = KokkosLibrary.GetValue(Pointer, NdArray, i0, i1, i2);

                return valuePtr.As<TDataType>();
            }
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            set { KokkosLibrary.SetValue(Pointer, NdArray, ValueType.From(value), i0, i1, i2); }
        }

        public TDataType this[ulong i0,
                              ulong i1,
                              ulong i2,
                              ulong i3]
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            get
            {
                ValueType valuePtr = KokkosLibrary.GetValue(Pointer, NdArray, i0, i1, i2, i3);

                return valuePtr.As<TDataType>();
            }
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            set { KokkosLibrary.SetValue(Pointer, NdArray, ValueType.From(value), i0, i1, i2, i3); }
        }

        public TDataType this[ulong i0,
                              ulong i1,
                              ulong i2,
                              ulong i3,
                              ulong i4]
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            get
            {
                ValueType valuePtr = KokkosLibrary.GetValue(Pointer, NdArray, i0, i1, i2, i3, i4);

                return valuePtr.As<TDataType>();
            }
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            set { KokkosLibrary.SetValue(Pointer, NdArray, ValueType.From(value), i0, i1, i2, i3, i4); }
        }

        public TDataType this[ulong i0,
                              ulong i1,
                              ulong i2,
                              ulong i3,
                              ulong i4,
                              ulong i5]
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            get
            {
                ValueType valuePtr = KokkosLibrary.GetValue(Pointer, NdArray, i0, i1, i2, i3, i4, i5);

                return valuePtr.As<TDataType>();
            }
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            set { KokkosLibrary.SetValue(Pointer, NdArray, ValueType.From(value), i0, i1, i2, i3, i4, i5); }
        }

        public TDataType this[ulong i0,
                              ulong i1,
                              ulong i2,
                              ulong i3,
                              ulong i4,
                              ulong i5,
                              ulong i6]
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            get
            {
                ValueType valuePtr = KokkosLibrary.GetValue(Pointer, NdArray, i0, i1, i2, i3, i4, i5, i6);

                return valuePtr.As<TDataType>();
            }
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            set { KokkosLibrary.SetValue(Pointer, NdArray, ValueType.From(value), i0, i1, i2, i3, i4, i5, i6); }
        }

        public TDataType this[ulong i0,
                              ulong i1,
                              ulong i2,
                              ulong i3,
                              ulong i4,
                              ulong i5,
                              ulong i6,
                              ulong i7]
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            get
            {
                ValueType valuePtr = KokkosLibrary.GetValue(Pointer, NdArray, i0, i1, i2, i3, i4, i5, i6, i7);

                return valuePtr.As<TDataType>();
            }
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            set { KokkosLibrary.SetValue(Pointer, NdArray, ValueType.From(value), i0, i1, i2, i3, i4, i5, i6, i7); }
        }

        #endregion
    }

    //[NonVersionable]
    //public sealed class ConstView<TDataType, TExecutionSpace> : View
    //    where TDataType : struct
    //    where TExecutionSpace : IExecutionSpace, new()
    //{
    //    //private static readonly int dataTypeSize = Unsafe.SizeOf<TDataType>();

    //    private static readonly DataTypeKind dataType;

    //    private static readonly IExecutionSpace executionSpace;

    //    private static readonly ExecutionSpaceKind executionSpaceType;

    //    public TDataType this[ulong i0]
    //    {
    //#if NETSTANDARD
    //        [MethodImpl(MethodImplOptions.AggressiveInlining)]
    //#else
    //        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
    //#endif
    //        get
    //        {
    //            ValueType valuePtr = KokkosLibrary.GetValue(Pointer,
    //                                                        NdArray,
    //                                                        i0);

    //            return valuePtr.As<TDataType>();
    //        }
    //    }

    //    public TDataType this[ulong i0,
    //                          ulong i1]
    //    {
    //#if NETSTANDARD
    //        [MethodImpl(MethodImplOptions.AggressiveInlining)]
    //#else
    //        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
    //#endif
    //        get
    //        {
    //            ValueType valuePtr = KokkosLibrary.GetValue(Pointer,
    //                                                        NdArray,
    //                                                        i0,
    //                                                        i1);

    //            return valuePtr.As<TDataType>(); 
    //        }
    //    }

    //    public TDataType this[ulong i0,
    //                          ulong i1,
    //                          ulong i2]
    //    {
    //#if NETSTANDARD
    //        [MethodImpl(MethodImplOptions.AggressiveInlining)]
    //#else
    //        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
    //#endif
    //        get
    //        {
    //            ValueType valuePtr = KokkosLibrary.GetValue(Pointer,
    //                                                        NdArray,
    //                                                        i0,
    //                                                        i1,
    //                                                        i2);

    //            return valuePtr.As<TDataType>(); 
    //        }
    //    }

    //    [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
    //    static ConstView()
    //    {
    //        dataType           = DataType<TDataType>.GetKind(true);
    //        executionSpace     = new TExecutionSpace();
    //        executionSpaceType = ExecutionSpace<TExecutionSpace>.GetKind();
    //    }

    //    [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
    //    public ConstView(View<TDataType, TExecutionSpace> view)
    //        : base(pointer,
    //               ndArray)
    //    {
    //    }

    //    [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
    //    public string Label()
    //    {
    //        return KokkosLibrary.GetLabel(Pointer,
    //                                      NdArray).ToString();
    //    }

    //    [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
    //    public ulong Size()
    //    {
    //        return KokkosLibrary.GetSize(Pointer,
    //                                     NdArray);
    //    }

    //    [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
    //    public ulong Stride(uint dim)
    //    {
    //        return KokkosLibrary.GetStride(Pointer,
    //                                       NdArray,
    //                                       dim);
    //    }

    //    [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
    //    public ulong Extent(uint dim)
    //    {
    //        return KokkosLibrary.GetExtent(Pointer,
    //                                       NdArray,
    //                                       dim);
    //    }

    //    [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
    //    public void CopyTo(TDataType[] values)
    //    {
    //        //GCHandle handle = GCHandle.Alloc(values[0],
    //        //                                 GCHandleType.Pinned);

    //        ValueType[] valueTypes = Array.ConvertAll(values,
    //                                                  ValueType.From);

    //        KokkosLibrary.CopyTo(Pointer,
    //                             NdArray,
    //                             valueTypes);

    //        //handle.AddrOfPinnedObject()handle.Free();
    //    }

    //    [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
    //    public static NdArray Convert(nint view_ptr,
    //                                  ushort rank)
    //    {
    //        return KokkosLibrary.ViewToNdArray(view_ptr,
    //                                           executionSpaceType,
    //                                           executionSpace.DefaultLayout,
    //                                           dataType,
    //                                           rank);
    //    }
    //}
}