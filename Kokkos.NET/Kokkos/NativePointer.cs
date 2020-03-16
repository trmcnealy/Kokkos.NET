using System;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Versioning;

namespace Kokkos
{
    //[Serializer(typeof(NativePointer.CustomSerializer))]
    [NonVersionable]
    public sealed class NativePointer : IDisposable
    {
        //[MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        //static NativePointer()
        //{
        //    if(!KokkosLibrary.IsInitialized())
        //    {
        //        KokkosLibraryException.Throw("Kokkos Library is not initialized. Use ParallelProcessor.Initialize/Shutdown in the main routine/thread.");
        //    }
        //}

        private IntPtr _data;

        private int _size;

        private bool _mustDeallocate;

        private ExecutionSpaceKind _executionSpace;

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public NativePointer(IntPtr             data,
                             int                size,
                             bool               mustDeallocate = false,
                             ExecutionSpaceKind executionSpace = ExecutionSpaceKind.Serial)
        {
            _data           = data;
            _size           = size;
            _mustDeallocate = mustDeallocate;
            _executionSpace = executionSpace;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public NativePointer(IntPtr             data,
                             uint               size,
                             bool               mustDeallocate = false,
                             ExecutionSpaceKind executionSpace = ExecutionSpaceKind.Serial)
        {
            _data           = data;
            _size           = (int)size;
            _mustDeallocate = mustDeallocate;
            _executionSpace = executionSpace;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public NativePointer(IntPtr             data,
                             long               size,
                             bool               mustDeallocate = false,
                             ExecutionSpaceKind executionSpace = ExecutionSpaceKind.Serial)
        {
            _data           = data;
            _size           = (int)size;
            _mustDeallocate = mustDeallocate;
            _executionSpace = executionSpace;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public NativePointer(IntPtr             data,
                             ulong              size,
                             bool               mustDeallocate = false,
                             ExecutionSpaceKind executionSpace = ExecutionSpaceKind.Serial)
        {
            _data           = data;
            _size           = (int)size;
            _mustDeallocate = mustDeallocate;
            _executionSpace = executionSpace;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public NativePointer(int                size,
                             ExecutionSpaceKind executionSpace)
            : this(KokkosLibrary.Allocate(executionSpace,
                                          (ulong)size),
                   size,
                   true,
                   executionSpace)
        {
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public NativePointer(ExecutionSpaceKind executionSpace = ExecutionSpaceKind.Serial)
            : this(Unsafe.SizeOf<IntPtr>(),
                   executionSpace)
        {
        }

        ~NativePointer()
        {
            DisposeUnmanaged();
        }

        public ref IntPtr Data
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            get { return ref _data; }
        }

        public int Size
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            get { return _size; }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static implicit operator IntPtr(NativePointer pointer)
        {
            return pointer.Data;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static unsafe implicit operator void*(NativePointer pointer)
        {
            return pointer.Data.ToPointer();
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static NativePointer Allocate(int                size,
                                             ExecutionSpaceKind executionSpace = ExecutionSpaceKind.Serial)
        {
            IntPtr data = KokkosLibrary.Allocate(executionSpace,
                                                 (ulong)size);

            unsafe
            {
                byte* d = (byte*)data.ToPointer();

                for(byte* i = d; i < d + size; i++)
                {
                    *i = 0;
                }
            }

            return new NativePointer(data,
                                     size,
                                     true,
                                     executionSpace);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static NativePointer Allocate(uint               size,
                                             ExecutionSpaceKind executionSpace = ExecutionSpaceKind.Serial)
        {
            IntPtr data = KokkosLibrary.Allocate(executionSpace,
                                                 size);

            unsafe
            {
                byte* d = (byte*)data.ToPointer();

                for(byte* i = d; i < d + size; i++)
                {
                    *i = 0;
                }
            }

            return new NativePointer(data,
                                     size,
                                     true,
                                     executionSpace);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static NativePointer Allocate(long               size,
                                             ExecutionSpaceKind executionSpace = ExecutionSpaceKind.Serial)
        {
            IntPtr data = KokkosLibrary.Allocate(executionSpace,
                                                 (ulong)size);

            unsafe
            {
                byte* d = (byte*)data.ToPointer();

                for(byte* i = d; i < d + size; i++)
                {
                    *i = 0;
                }
            }

            return new NativePointer(data,
                                     size,
                                     true,
                                     executionSpace);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static NativePointer Allocate(ulong              size,
                                             ExecutionSpaceKind executionSpace = ExecutionSpaceKind.Serial)
        {
            IntPtr data = KokkosLibrary.Allocate(executionSpace,
                                                 size);

            unsafe
            {
                byte* d = (byte*)data.ToPointer();

                for(byte* i = d; i < d + size; i++)
                {
                    *i = 0;
                }
            }

            return new NativePointer(data,
                                     size,
                                     true,
                                     executionSpace);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public void Dispose()
        {
            DisposeUnmanaged();
            GC.SuppressFinalize(this);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        private void DisposeUnmanaged()
        {
            if(_mustDeallocate && _data != IntPtr.Zero)
            {
                KokkosLibrary.Free(_executionSpace,
                                   _data);

                _data           = IntPtr.Zero;
                _size           = 0;
                _mustDeallocate = false;
            }
        }

        //public static NativePointer WrapIntPtr(IntPtr data,
        //                                       int    size)
        //{
        //    return new NativePointer(data,
        //                             size,
        //                             false);
        //}

        //public static NativePointer CreateCopyFrom(IntPtr data,
        //                                           int    size)
        //{
        //    IntPtr newData = Marshal.AllocHGlobal(size);

        //    CopyUnmanagedMemory(newData,
        //                        data,
        //                        size);

        //    return new NativePointer(newData,
        //                             size,
        //                             true);
        //}

        //public static NativePointer CreateCopyFrom(byte[] data)
        //{
        //    NativePointer result = Allocate(data.Length);
        //    result.CopyFrom(data);

        //    return result;
        //}

        //public NativePointer Clone()
        //{
        //    IntPtr newData = Marshal.AllocHGlobal(size);

        //    CopyUnmanagedMemory(newData,
        //                        data,
        //                        size);

        //    return new NativePointer(newData,
        //                             size,
        //                             true);
        //}

        //public void CopyTo(NativePointer destination)
        //{
        //    if(destination == null)
        //    {
        //        throw new ArgumentException("Destination unmanaged array is null.");
        //    }
        //    else if(destination.size != size)
        //    {
        //        throw new ArgumentException("Destination unmanaged array is not of the same size.");
        //    }
        //    else
        //    {
        //        CopyUnmanagedMemory(destination.data,
        //                            data,
        //                            size);
        //    }
        //}

        //public byte[] ReadBytes(int count,
        //                        int offset = 0)
        //{
        //    byte[] result = new byte[count];

        //    Marshal.Copy(IntPtr.Add(data,
        //                            offset),
        //                 result,
        //                 0,
        //                 count);

        //    return result;
        //}

        //public void CopyTo(byte[] destination)
        //{
        //    if(destination == null)
        //    {
        //        throw new ArgumentException("Destination buffer is null.");
        //    }
        //    else if(size != destination.Length)
        //    {
        //        throw new ArgumentException("Destination buffer is not of the same size.");
        //    }

        //    Marshal.Copy(data,
        //                 destination,
        //                 0,
        //                 destination.Length);
        //}

        //public void CopyTo(IntPtr destination,
        //                   int    size)
        //{
        //    if(size != this.size)
        //    {
        //        throw new ArgumentException("Destination size is not the same as source.");
        //    }

        //    CopyUnmanagedMemory(destination,
        //                        data,
        //                        this.size);
        //}

        //public void CopyFrom(NativePointer source)
        //{
        //    if(source == null)
        //    {
        //        throw new ArgumentException("Source unmanaged array is null.");
        //    }
        //    else if(size != source.Size)
        //    {
        //        throw new ArgumentException("Source unmanaged array is not of the same size.");
        //    }

        //    CopyUnmanagedMemory(data,
        //                        source.data,
        //                        size);
        //}

        //public void CopyFrom(byte[] source)
        //{
        //    if(source == null)
        //    {
        //        throw new ArgumentException("Source buffer is null.");
        //    }
        //    else if(size != source.Length)
        //    {
        //        throw new ArgumentException("Source buffer is not of the same size.");
        //    }

        //    Marshal.Copy(source,
        //                 0,
        //                 data,
        //                 source.Length);
        //}

        //public void CopyFrom(IntPtr source,
        //                     int    size)
        //{
        //    if(size != this.size)
        //    {
        //        throw new ArgumentException("Destination size is not the same as source.");
        //    }

        //    CopyUnmanagedMemory(data,
        //                        source,
        //                        this.size);
        //}

        //private static unsafe void CopyUnmanagedMemory(IntPtr dst,
        //                                               IntPtr src,
        //                                               int    count)
        //{
        //    unsafe
        //    {
        //        Buffer.MemoryCopy(src.ToPointer(),
        //                          dst.ToPointer(),
        //                          count,
        //                          count);
        //    }
        //}

        //private class CustomSerializer : ISerializer<NativePointer>
        //{
        //    public const int Version = 2;
        //
        //    public TypeSchema Initialize(KnownSerializers serializers, TypeSchema targetSchema)
        //    {
        //        serializers.GetHandler<byte>(); // register element type
        //        var type = typeof(byte[]);
        //        var name = TypeSchema.GetContractName(type, serializers.RuntimeVersion);
        //        var elementsMember = new TypeMemberSchema("Elements", typeof(byte).AssemblyQualifiedName, true);
        //        var schema = new TypeSchema(name, TypeSchema.GetId(name), type.AssemblyQualifiedName, TypeFlags.IsCollection, new TypeMemberSchema[] { elementsMember }, Version);
        //        return targetSchema ?? schema;
        //    }
        //
        //    public void Serialize(BufferWriter writer, NativePointer instance, SerializationContext context)
        //    {
        //        unsafe
        //        {
        //            writer.Write(instance.Size);
        //            writer.Write(instance.Data.ToPointer(), instance.Size);
        //        }
        //    }
        //
        //    public void PrepareCloningTarget(NativePointer instance, ref NativePointer target, SerializationContext context)
        //    {
        //        if (target == null || target.Size != instance.Size)
        //        {
        //            target?.Dispose();
        //            target = new NativePointer(instance.Size);
        //        }
        //    }
        //
        //    public void Clone(NativePointer instance, ref NativePointer target, SerializationContext context)
        //    {
        //        CopyUnmanagedMemory(target.data, instance.data, instance.size);
        //    }
        //
        //    public void PrepareDeserializationTarget(BufferReader reader, ref NativePointer target, SerializationContext context)
        //    {
        //        int size = reader.ReadInt32();
        //        if (target == null || target.Size != size)
        //        {
        //            target?.Dispose();
        //            target = new NativePointer(size);
        //        }
        //    }
        //
        //    public void Deserialize(BufferReader reader, ref NativePointer target, SerializationContext context)
        //    {
        //        unsafe
        //        {
        //            reader.Read(target.Data.ToPointer(), target.Size);
        //        }
        //    }
        //
        //    public void Clear(ref NativePointer target, SerializationContext context)
        //    {
        //        // nothing to clear in an unmanaged buffer
        //    }
        //}
    }
}