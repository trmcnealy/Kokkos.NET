using System;
using System.IO;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Security;
using System.Threading;
using System.Threading.Tasks;

using RuntimeGeneration;
#if X86
using nuint = System.UInt32;
#else
using nuint = System.UInt64;
#endif

namespace System
{
    // ByReference<T> is meant to be used to represent "ref T" fields. It is working
    // around lack of first class support for byref fields in C# and IL. The JIT and
    // type loader has special handling for it that turns it into a thin wrapper around ref T.

    internal static unsafe class Mem
    {
        [SuppressGCTransition, SuppressUnmanagedCodeSecurity, DllImport("Kernel32.dll", EntryPoint = "RtlMoveMemory", SetLastError = false)]
        public static extern void move(void* dest,
                                          void* src,
                                          int   size);

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static void move<T>(ref T destination,
                                      ref T source,
                                      ulong elementCount)
        {
            move(Unsafe.AsPointer(ref destination), Unsafe.AsPointer(ref source), (int)elementCount * Unsafe.SizeOf<T>());
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

        public nint Pointer;

        public MemoryMapped()
        {
            Pointer = Native.Create();
        }

        public MemoryMapped(string    filename,
                            ulong     mappedBytes = (ulong)MapRange.WholeFile,
                            CacheHint hint        = CacheHint.SequentialScan)
        {
            if(!File.Exists(filename))
            {
                throw new FileNotFoundException();
            }

            Pointer = Native.CreateAndOpen(Marshal.StringToHGlobalAnsi(filename), mappedBytes, hint);
        }

        public void Dispose()
        {
            ReleaseUnmanagedResources();
            GC.SuppressFinalize(this);
        }

        private void ReleaseUnmanagedResources()
        {
            Native.Close(Pointer);
            Native.Destory(Pointer);
        }

        ~MemoryMapped()
        {
            Dispose();
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
            int length = Length();

            byte[] bytes = new byte[length];

            nint data = Native.GetData(Pointer);

            Marshal.Copy(bytes, 0, data, length);

            return bytes;
        }

        public UnmanagedMemoryStream AsStream()
        {
            unsafe
            {
                return new UnmanagedMemoryStream((byte*)Pointer, (long)Size());
            }
        }

        //public unsafe ReadOnlySpan<byte> GetReadOnlySpan()
        //{
        //    return new ReadOnlySpan<byte>(Native.GetData(Pointer).ToPointer(), (int)Size());
        //}

        public unsafe ReadOnlySpan<byte> GetDataPointer()
        {
            if(Size() > 2147483647)
            {
                throw new Exception("Retards at Microsoft limited ReadOnlySpan to 2147483647 bytes. Use the GetPointer method.");
            }

            return new ReadOnlySpan<byte>((void*)Native.GetData(Pointer), Length());
        }

        public unsafe T* GetPointer<T>()
            where T : unmanaged
        {
            return (T*)Native.GetData(Pointer);
        }

        public bool IsValid()
        {
            return Native.IsValid(Pointer);
        }

        public int Length()
        {
            return (int)Native.Size(Pointer);
        }

        public ulong Size()
        {
            return Native.Size(Pointer);
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
            
            
            
            public delegate byte AtDelegate(nint mm,
                                            ulong  offset);

            public delegate void CloseDelegate(nint mm);

            public delegate nint CreateAndOpenDelegate(nint    filename,
                                                         ulong     mappedBytes,
                                                         CacheHint hint);

            public delegate nint CreateDelegate();

            public delegate void DestoryDelegate(nint mm);

            public delegate nint GetDataDelegate(nint mm);

            public delegate bool IsValidDelegate(nint mm);

            public delegate ulong MappedSizeDelegate(nint mm);

            public delegate bool OpenDelegate(nint    mm,
                                              nint    filename,
                                              ulong     mappedBytes,
                                              CacheHint hint);

            public delegate bool RemapDelegate(nint mm,
                                               ulong  offset,
                                               ulong  mappedBytes);

            public delegate ulong SizeDelegate(nint mm);

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
                //RuntimeCil.Generate(typeof(MemoryMapped).Assembly);
                
                Create = Marshal.GetDelegateForFunctionPointer<CreateDelegate>(PlatformApi.NativeLibrary.GetExport(KokkosLibrary.Handle, "Create"));

                CreateAndOpen = Marshal.GetDelegateForFunctionPointer<CreateAndOpenDelegate>(PlatformApi.NativeLibrary.GetExport(KokkosLibrary.Handle, "CreateAndOpen"));

                Destory = Marshal.GetDelegateForFunctionPointer<DestoryDelegate>(PlatformApi.NativeLibrary.GetExport(KokkosLibrary.Handle, "Destory"));

                Open = Marshal.GetDelegateForFunctionPointer<OpenDelegate>(PlatformApi.NativeLibrary.GetExport(KokkosLibrary.Handle, "Open"));

                Close = Marshal.GetDelegateForFunctionPointer<CloseDelegate>(PlatformApi.NativeLibrary.GetExport(KokkosLibrary.Handle, "Close"));

                At = Marshal.GetDelegateForFunctionPointer<AtDelegate>(PlatformApi.NativeLibrary.GetExport(KokkosLibrary.Handle, "At"));

                GetData = Marshal.GetDelegateForFunctionPointer<GetDataDelegate>(PlatformApi.NativeLibrary.GetExport(KokkosLibrary.Handle, "GetData"));

                IsValid = Marshal.GetDelegateForFunctionPointer<IsValidDelegate>(PlatformApi.NativeLibrary.GetExport(KokkosLibrary.Handle, "IsValid"));

                Size = Marshal.GetDelegateForFunctionPointer<SizeDelegate>(PlatformApi.NativeLibrary.GetExport(KokkosLibrary.Handle, "Size"));

                MappedSize = Marshal.GetDelegateForFunctionPointer<MappedSizeDelegate>(PlatformApi.NativeLibrary.GetExport(KokkosLibrary.Handle, "MappedSize"));

                Remap = Marshal.GetDelegateForFunctionPointer<RemapDelegate>(PlatformApi.NativeLibrary.GetExport(KokkosLibrary.Handle, "Remap"));
            }
        }
    }
}
