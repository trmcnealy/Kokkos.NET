using System;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Security;

using RuntimeGeneration;
#if TARGET_64BIT
using nuint = System.UInt64;

#else
using nuint = System.UInt32;
#endif

namespace System
{
    // ByReference<T> is meant to be used to represent "ref T" fields. It is working
    // around lack of first class support for byref fields in C# and IL. The JIT and
    // type loader has special handling for it that turns it into a thin wrapper around ref T.

    public static unsafe class Mem
    {
        [SuppressUnmanagedCodeSecurity]
        [DllImport("Kernel32.dll", EntryPoint = "RtlMoveMemory", SetLastError = false)]
        public static extern void Memmove(void* dest,
                                          void* src,
                                          int   size);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void Memmove<T>(ref T destination,
                                      ref T source,
                                      ulong elementCount)
        {
            Memmove(Unsafe.AsPointer(ref destination), Unsafe.AsPointer(ref source), (int)elementCount * Unsafe.SizeOf<T>());
        }
    }
}

namespace Kokkos
{
    public sealed class MemoryMapped : IDisposable
    {
        /// <summary>tweak performance</summary>
        public enum CacheHint : uint
        {
            /// <summary>good overall performance</summary>
            Normal,

            /// <summary>read file only once with few seeks</summary>
            SequentialScan,

            /// <summary>jump around</summary>
            RandomAccess
        }

        /// <summary>how much should be mappend</summary>
        public enum MapRange : uint
        {
            /// <summary>everything ... be careful when file is larger than memory</summary>
            WholeFile = 0
        }

        public IntPtr Pointer;

        public MemoryMapped()
        {
            Pointer = Native.Create();
        }

        public MemoryMapped(string    filename,
                            ulong     mappedBytes = (ulong)MapRange.WholeFile,
                            CacheHint hint        = CacheHint.Normal)
        {
            Pointer = Native.CreateAndOpen(Marshal.StringToHGlobalAnsi(filename), mappedBytes, hint);
        }

        public void Dispose()
        {
            ReleaseUnmanagedResources();
            GC.SuppressFinalize(this);
        }

        private void ReleaseUnmanagedResources()
        {
            Native.Destory(Pointer);
        }

        ~MemoryMapped()
        {
            ReleaseUnmanagedResources();
        }

        public bool Open(string    filename,
                         ulong     mappedBytes = (ulong)MapRange.WholeFile,
                         CacheHint hint        = CacheHint.Normal)
        {
            return Native.Open(Pointer, Marshal.StringToHGlobalAnsi(filename), mappedBytes, hint);
        }

        public void Close()
        {
            Native.Close(Pointer);
        }

        public byte At(ulong offset)
        {
            return Native.At(Pointer, offset);
        }

        public byte[] GetData()
        {
            int length = Size();

            byte[] bytes = new byte[length];

            IntPtr data = Native.GetData(Pointer);

            Marshal.Copy(bytes, 0, data, length);

            return bytes;
        }

        //public unsafe ReadOnlySpan<byte> GetReadOnlySpan()
        //{
        //    return new ReadOnlySpan<byte>(Native.GetData(Pointer).ToPointer(), (int)Size());
        //}

        public unsafe ReadOnlySpan<byte> GetDataPointer()
        {
            return new ReadOnlySpan<byte>(Native.GetData(Pointer).ToPointer(), Size());
        }

        public bool IsValid()
        {
            return Native.IsValid(Pointer);
        }

        public int Size()
        {
            return (int)Native.Size(Pointer);
        }

        public ulong MappedSize()
        {
            return Native.MappedSize(Pointer);
        }

        public bool Remap(ulong offset,
                          ulong mappedBytes)
        {
            return Native.Remap(Pointer, offset, mappedBytes);
        }

        public static class Native
        {
            public delegate byte AtDelegate(IntPtr mm,
                                            ulong  offset);

            public delegate void CloseDelegate(IntPtr mm);

            public delegate IntPtr CreateAndOpenDelegate(IntPtr    filename,
                                                         ulong     mappedBytes,
                                                         CacheHint hint);

            public delegate IntPtr CreateDelegate();

            public delegate void DestoryDelegate(IntPtr mm);

            public delegate IntPtr GetDataDelegate(IntPtr mm);

            public delegate bool IsValidDelegate(IntPtr mm);

            public delegate ulong MappedSizeDelegate(IntPtr mm);

            public delegate bool OpenDelegate(IntPtr    mm,
                                              IntPtr    filename,
                                              ulong     mappedBytes,
                                              CacheHint hint);

            public delegate bool RemapDelegate(IntPtr mm,
                                               ulong  offset,
                                               ulong  mappedBytes);

            public delegate ulong SizeDelegate(IntPtr mm);

            public const string LibraryName = "runtime.Kokkos.NET";

            [NativeCall(LibraryName, "Create", true)]
            public static CreateDelegate Create;

            [NativeCall(LibraryName, "CreateAndOpen", true)]
            public static CreateAndOpenDelegate CreateAndOpen;

            [NativeCall(LibraryName, "Destory", true)]
            public static DestoryDelegate Destory;

            [NativeCall(LibraryName, "Open", true)]
            public static OpenDelegate Open;

            [NativeCall(LibraryName, "Close", true)]
            public static CloseDelegate Close;

            [NativeCall(LibraryName, "At", true)]
            public static AtDelegate At;

            [NativeCall(LibraryName, "GetData", true)]
            public static GetDataDelegate GetData;

            [NativeCall(LibraryName, "IsValid", true)]
            public static IsValidDelegate IsValid;

            [NativeCall(LibraryName, "Size", true)]
            public static SizeDelegate Size;

            [NativeCall(LibraryName, "MappedSize", true)]
            public static MappedSizeDelegate MappedSize;

            [NativeCall(LibraryName, "Remap", true)]
            public static RemapDelegate Remap;

            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            static Native()
            {
                RuntimeCil.Generate(typeof(MemoryMapped).Assembly);
            }
        }
    }
}
