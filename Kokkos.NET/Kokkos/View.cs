using System;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Versioning;

namespace Kokkos
{
    [NonVersionable]
    [StructLayout(System.Runtime.InteropServices.LayoutKind.Sequential)]
    public unsafe struct NdArray
    {
        public DataTypeKind DataType;

        public ushort Rank;

        public LayoutKind Layout;

        public ExecutionSpaceKind ExecutionSpace;

        public fixed ulong Dims[8];

        public fixed ulong Strides[8];

        public IntPtr Data;

        //[MarshalAs(UnmanagedType.LPUTF8Str)] string
        public IntPtr Label;

        public NdArray(DataTypeKind       dataType,
                       ushort             rank,
                       LayoutKind         layout,
                       ExecutionSpaceKind executionSpace,
                       string             label)
        {
            DataType       = dataType;
            Rank           = rank;
            Layout         = layout;
            ExecutionSpace = executionSpace;
            Data           = IntPtr.Zero;
            Label          = Marshal.StringToHGlobalAnsi(label);
        }

        public NdArray(DataTypeKind       dataType,
                       ushort             rank,
                       LayoutKind         layout,
                       ExecutionSpaceKind executionSpace,
                       string             label,
                       ulong              n0)
            : this(dataType,
                   rank,
                   layout,
                   executionSpace,
                   label)
        {
            Dims[0] = n0;
        }

        public NdArray(DataTypeKind       dataType,
                       ushort             rank,
                       LayoutKind         layout,
                       ExecutionSpaceKind executionSpace,
                       string             label,
                       ulong              n0,
                       ulong              n1)
            : this(dataType,
                   rank,
                   layout,
                   executionSpace,
                   label)
        {
            Dims[0] = n0;
            Dims[1] = n1;
        }

        public NdArray(DataTypeKind       dataType,
                       ushort             rank,
                       LayoutKind         layout,
                       ExecutionSpaceKind executionSpace,
                       string             label,
                       ulong              n0,
                       ulong              n1,
                       ulong              n2)
            : this(dataType,
                   rank,
                   layout,
                   executionSpace,
                   label)
        {
            Dims[0] = n0;
            Dims[1] = n1;
            Dims[2] = n2;
        }
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

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        static View()
        {
            if(!KokkosLibrary.IsInitialized())
            {
                KokkosLibraryException.Throw("Kokkos Library is not initialized. Use ParallelProcessor.Initialize/Shutdown in the main routine/thread.");
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        protected View(DataTypeKind       dataType,
                       ushort             rank,
                       LayoutKind         layout,
                       ExecutionSpaceKind executionSpace,
                       string             label)
        {
            Pointer = new NativePointer();

            NdArray ndArray = new NdArray(dataType,
                                          rank,
                                          layout,
                                          executionSpace,
                                          label);

            KokkosLibrary.CreateView(Pointer,
                                     ref ndArray);

            NdArray = ndArray;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        protected View(DataTypeKind       dataType,
                       ushort             rank,
                       LayoutKind         layout,
                       ExecutionSpaceKind executionSpace,
                       string             label,
                       ulong              n0)
        {
            Pointer = new NativePointer();

            NdArray ndArray = new NdArray(dataType,
                                          rank,
                                          layout,
                                          executionSpace,
                                          label,
                                          n0);

            KokkosLibrary.CreateView(Pointer,
                                     ref ndArray);

            NdArray = ndArray;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        protected View(DataTypeKind       dataType,
                       ushort             rank,
                       LayoutKind         layout,
                       ExecutionSpaceKind executionSpace,
                       string             label,
                       ulong              n0,
                       ulong              n1)
        {
            Pointer = new NativePointer();

            NdArray ndArray = new NdArray(dataType,
                                          rank,
                                          layout,
                                          executionSpace,
                                          label,
                                          n0,
                                          n1);

            KokkosLibrary.CreateView(Pointer,
                                     ref ndArray);

            NdArray = ndArray;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        protected View(DataTypeKind       dataType,
                       ushort             rank,
                       LayoutKind         layout,
                       ExecutionSpaceKind executionSpace,
                       string             label,
                       ulong              n0,
                       ulong              n1,
                       ulong              n2)
        {
            Pointer = new NativePointer();

            NdArray ndArray = new NdArray(dataType,
                                          rank,
                                          layout,
                                          executionSpace,
                                          label,
                                          n0,
                                          n1,
                                          n2);

            KokkosLibrary.CreateView(Pointer,
                                     ref ndArray);

            NdArray = ndArray;
        }

        public void Dispose()
        {
            Pointer?.Dispose();
        }
    }

    [NonVersionable]
    public sealed class View<TDataType, TExecutionSpace> : View
        where TDataType : struct
        where TExecutionSpace : IExecutionSpace
    {
        //private static readonly int dataTypeSize = Unsafe.SizeOf<TDataType>();

        private static readonly DataTypeKind dataType;

        private static readonly ExecutionSpaceKind executionSpace;

        //public View(string label,
        //            bool   isConst = false)
        //{
        //    Pointer = new NativePointer();
        //
        //    unsafe
        //    {
        //        if(isConst)
        //        {
        //            View.KokkosLibrary.CreateViewRank0(Pointer,
        //                                         DataType + 10,
        //                                         ExecutionSpace,
        //                                         label);
        //        }
        //        else
        //        {
        //            View.KokkosLibrary.CreateViewRank0(Pointer,
        //                                         DataType,
        //                                         ExecutionSpace,
        //                                         label);
        //        }
        //    }
        //}

        //public View(string label,
        //            ulong  n0,
        //            bool   isConst = false)
        //{
        //    Pointer = new NativePointer();
        //
        //    unsafe
        //    {
        //        if(isConst)
        //        {
        //            View.KokkosLibrary.CreateViewRank1(Pointer,
        //                                         DataType + 10,
        //                                         ExecutionSpace,
        //                                         label,
        //                                         n0);
        //        }
        //        else
        //        {
        //            View.KokkosLibrary.CreateViewRank1(Pointer,
        //                                         DataType,
        //                                         ExecutionSpace,
        //                                         label,
        //                                         n0);
        //        }
        //    }
        //}

        //public View(string label,
        //            ulong  n0,
        //            ulong  n1,
        //            bool   isConst = false)
        //{
        //    Pointer = new NativePointer();
        //
        //    unsafe
        //    {
        //        if(isConst)
        //        {
        //            View.KokkosLibrary.CreateViewRank2(Pointer,
        //                                         DataType + 10,
        //                                         ExecutionSpace,
        //                                         label,
        //                                         n0,
        //                                         n1);
        //        }
        //        else
        //        {
        //            View.KokkosLibrary.CreateViewRank2(Pointer,
        //                                         DataType,
        //                                         ExecutionSpace,
        //                                         label,
        //                                         n0,
        //                                         n1);
        //        }
        //    }
        //}

        //public View(string label,
        //            ulong  n0,
        //            ulong  n1,
        //            ulong  n2,
        //            bool   isConst = false)
        //{
        //    Pointer = new NativePointer();
        //
        //    unsafe
        //    {
        //        if(isConst)
        //        {
        //            View.KokkosLibrary.CreateViewRank3(Pointer,
        //                                         DataType + 10,
        //                                         ExecutionSpace,
        //                                         label,
        //                                         n0,
        //                                         n1,
        //                                         n2);
        //        }
        //        else
        //        {
        //            View.KokkosLibrary.CreateViewRank3(Pointer,
        //                                         DataType,
        //                                         ExecutionSpace,
        //                                         label,
        //                                         n0,
        //                                         n1,
        //                                         n2);
        //        }
        //    }
        //}

        public TDataType this[ulong i0]
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            get
            {
                ValueType valuePtr = KokkosLibrary.GetValue(Pointer,
                                                            NdArray,
                                                            i0);

                return valuePtr.As<TDataType>();
            }
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            set
            {
                //GCHandle handle = GCHandle.ToIntPtr(value);

                //byte* pvt = stackalloc byte[dataTypeSze];

                //IntPtr valuePtr = new IntPtr(pvt);
                ////Marshal.AllocHGlobal(Marshal.SizeOf(value));

                //Marshal.GetNativeVariantForObject(value,
                //                                  valuePtr);

                KokkosLibrary.SetValue(Pointer,
                                       NdArray,
                                       ValueType.From(value),
                                       i0);

                //handle.Free();
            }
        }

        public TDataType this[ulong i0,
                              ulong i1]
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            get
            {
                ValueType valuePtr = KokkosLibrary.GetValue(Pointer,
                                                            NdArray,
                                                            i0,
                                                            i1);

                return valuePtr.As<TDataType>(); //return Unsafe.AsRef<TDataType>(valuePtr.ToPointer());
            }
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            set
            {
                //IntPtr valuePtr = (IntPtr)Unsafe.AsPointer(ref value);

                KokkosLibrary.SetValue(Pointer,
                                       NdArray,
                                       ValueType.From(value),
                                       i0,
                                       i1);
            }
        }

        public TDataType this[ulong i0,
                              ulong i1,
                              ulong i2]
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            get
            {
                ValueType valuePtr = KokkosLibrary.GetValue(Pointer,
                                                            NdArray,
                                                            i0,
                                                            i1,
                                                            i2);

                return valuePtr.As<TDataType>(); //return Unsafe.AsRef<TDataType>(valuePtr.ToPointer());
            }
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            set
            {
                //IntPtr valuePtr = (IntPtr)Unsafe.AsPointer(ref value);

                KokkosLibrary.SetValue(Pointer,
                                       NdArray,
                                       ValueType.From(value),
                                       i0,
                                       i1,
                                       i2);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        static View()
        {
            dataType       = DataType<TDataType>.GetKind();
            executionSpace = ExecutionSpace<TExecutionSpace>.GetKind();
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public View(string label)
            : base(dataType,
                   0,
                   ExecutionSpace<TExecutionSpace>.GetLayout(),
                   executionSpace,
                   label)
        {
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public View(string label,
                    int    n0)
            : base(dataType,
                   1,
                   ExecutionSpace<TExecutionSpace>.GetLayout(),
                   executionSpace,
                   label,
                   (ulong)n0)
        {
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public View(string label,
                    int    n0,
                    int    n1)
            : base(dataType,
                   2,
                   ExecutionSpace<TExecutionSpace>.GetLayout(),
                   executionSpace,
                   label,
                   (ulong)n0,
                   (ulong)n1)
        {
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public View(string label,
                    int    n0,
                    int    n1,
                    int    n2)
            : base(dataType,
                   3,
                   ExecutionSpace<TExecutionSpace>.GetLayout(),
                   executionSpace,
                   label,
                   (ulong)n0,
                   (ulong)n1,
                   (ulong)n2)
        {
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public View(string label,
                    uint   n0)
            : base(dataType,
                   1,
                   ExecutionSpace<TExecutionSpace>.GetLayout(),
                   executionSpace,
                   label,
                   n0)
        {
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public View(string label,
                    uint   n0,
                    uint   n1)
            : base(dataType,
                   2,
                   ExecutionSpace<TExecutionSpace>.GetLayout(),
                   executionSpace,
                   label,
                   n0,
                   n1)
        {
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public View(string label,
                    uint   n0,
                    uint   n1,
                    uint   n2)
            : base(dataType,
                   3,
                   ExecutionSpace<TExecutionSpace>.GetLayout(),
                   executionSpace,
                   label,
                   n0,
                   n1,
                   n2)
        {
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public View(string label,
                    long   n0)
            : base(dataType,
                   1,
                   ExecutionSpace<TExecutionSpace>.GetLayout(),
                   executionSpace,
                   label,
                   (ulong)n0)
        {
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public View(string label,
                    long   n0,
                    long   n1)
            : base(dataType,
                   2,
                   ExecutionSpace<TExecutionSpace>.GetLayout(),
                   executionSpace,
                   label,
                   (ulong)n0,
                   (ulong)n1)
        {
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public View(string label,
                    long   n0,
                    long   n1,
                    long   n2)
            : base(dataType,
                   3,
                   ExecutionSpace<TExecutionSpace>.GetLayout(),
                   executionSpace,
                   label,
                   (ulong)n0,
                   (ulong)n1,
                   (ulong)n2)
        {
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public View(string label,
                    ulong  n0)
            : base(dataType,
                   1,
                   ExecutionSpace<TExecutionSpace>.GetLayout(),
                   executionSpace,
                   label,
                   n0)
        {
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public View(string label,
                    ulong  n0,
                    ulong  n1)
            : base(dataType,
                   2,
                   ExecutionSpace<TExecutionSpace>.GetLayout(),
                   executionSpace,
                   label,
                   n0,
                   n1)
        {
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public View(string label,
                    ulong  n0,
                    ulong  n1,
                    ulong  n2)
            : base(dataType,
                   3,
                   ExecutionSpace<TExecutionSpace>.GetLayout(),
                   executionSpace,
                   label,
                   n0,
                   n1,
                   n2)
        {
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public string Label()
        {
            return KokkosLibrary.GetLabel(Pointer,
                                          NdArray).ToString();
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public ulong Size()
        {
            return KokkosLibrary.GetSize(Pointer,
                                         NdArray);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public ulong Stride(uint dim)
        {
            return KokkosLibrary.GetStride(Pointer,
                                           NdArray,
                                           dim);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public ulong Extent(uint dim)
        {
            return KokkosLibrary.GetExtent(Pointer,
                                           NdArray,
                                           dim);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public void CopyTo(TDataType[] values)
        {
            //GCHandle handle = GCHandle.Alloc(values[0],
            //                                 GCHandleType.Pinned);

            ValueType[] valueTypes = Array.ConvertAll(values,
                                                      ValueType.From);

            KokkosLibrary.CopyTo(Pointer,
                                 NdArray,
                                 valueTypes);

            //handle.AddrOfPinnedObject()handle.Free();
        }
    }
}