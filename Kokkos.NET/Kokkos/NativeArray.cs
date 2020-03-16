using System;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace Kokkos
{
    [System.Runtime.Versioning.NonVersionable]
    public sealed unsafe class NativeArray<T, TExecutionSpace> : IDisposable
        where T : unmanaged
        where TExecutionSpace : IExecutionSpace
    {
        [StructLayout(System.Runtime.InteropServices.LayoutKind.Explicit,
                      Size = 16)]
        public struct NativeStruct
        {
            [FieldOffset(0)]
            public int Length;

            [FieldOffset(sizeof(int))]
            public IntPtr Data;
        }

        private static readonly Type _T = typeof(T);

        private static readonly int elementSize;

        private static readonly ExecutionSpaceKind executionSpace;

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        static NativeArray()
        {
            //if(!KokkosLibrary.IsInitialized())
            //{
            //    KokkosLibraryException.Throw("Kokkos Library is not initialized. Use ParallelProcessor.Initialize/Shutdown in the main routine/thread.");
            //}

            executionSpace = ExecutionSpace<TExecutionSpace>.GetKind();

            elementSize = sizeof(T);
        }

        private readonly NativePointer _pointer;
        private          bool          _isOwner;
        private readonly T*            data_pointer;

        public int Length
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            get { return ((NativeStruct*)Instance)->Length; }
        }

        public T* Pointer
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            get { return data_pointer; }
        }

        public NativePointer Instance
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            get { return _pointer; }
        }

        public T this[in int index]
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            get { return data_pointer[index]; }
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            set { data_pointer[index] = value; }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public NativeArray(int size)
        {
            _pointer = NativePointer.Allocate(sizeof(NativeStruct),
                                              executionSpace);

            ((NativeStruct*)Instance)->Length = size;

            ((NativeStruct*)Instance)->Data = KokkosLibrary.Allocate(executionSpace,
                                                                     (ulong)(size * elementSize));

            _isOwner = true;

            data_pointer = (T*)((NativeStruct*)Instance)->Data;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public NativeArray(IntPtr nativeStructPointer)
        {
            _pointer = new NativePointer(nativeStructPointer,
                                         ((NativeStruct*)nativeStructPointer)->Length * elementSize);

            _isOwner = false;

            data_pointer = (T*)((NativeStruct*)Instance)->Data;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public NativeArray(IntPtr nativePointer,
                           int    length)
        {
            _pointer = new NativePointer(nativePointer,
                                         length * elementSize);

            _isOwner = false;

            data_pointer = (T*)((NativeStruct*)Instance)->Data;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public NativeArray(T[] array)
        {
            _pointer = NativePointer.Allocate(sizeof(NativeStruct),
                                              executionSpace);

            _isOwner = true;

            ((NativeStruct*)Instance)->Length = array.Length;

            ((NativeStruct*)Instance)->Data = KokkosLibrary.Allocate(executionSpace,
                                                                     (ulong)(array.Length * elementSize));

            data_pointer = (T*)((NativeStruct*)Instance)->Data;

            if(data_pointer != null)
            {
                for(int i = 0; i < array.Length; i++)
                {
                    data_pointer[i] = array[i];
                }
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public T[] ToManaged()
        {
            T[] result = new T[Length];

            if(Instance == IntPtr.Zero)
            {
                return result;
            }

            for(int i = 0; i < Length; ++i)
            {
                result[i] = data_pointer[i];
            }

            return result;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static T[] ToManaged(IntPtr nativeArray,
                                    int    length)
        {
            NativeArray<T, TExecutionSpace> array = new NativeArray<T, TExecutionSpace>(nativeArray,
                                                                                        length);

            return array.ToManaged();
        }

        #region IDisposable

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        private void ReleaseUnmanagedResources()
        {
            if(_isOwner)
            {
                KokkosLibrary.Free(executionSpace,
                                   ((NativeStruct*)Instance)->Data);
            }
        }

        public void Dispose()
        {
            ReleaseUnmanagedResources();
            GC.SuppressFinalize(this);
        }

        ~NativeArray()
        {
            ReleaseUnmanagedResources();
        }

        #endregion
    }
}