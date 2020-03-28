using System;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Text;

namespace Std
{
    namespace BasicString
    {
        [StructLayout(System.Runtime.InteropServices.LayoutKind.Explicit, Size = 32)]
        public struct basic_string
        {
            [FieldOffset(0)]
            internal AllocHider.Internal _M_dataplus;

            [FieldOffset(8)]
            internal ulong _M_string_length;

            [FieldOffset(16)]
            internal details.basic_string_char_traits_allocator details;
        }

        namespace AllocHider
        {
            [StructLayout(System.Runtime.InteropServices.LayoutKind.Explicit, Size = 8)]
            public struct Internal
            {
                [FieldOffset(0)]
                internal IntPtr _M_p;
            }

        }

        namespace details
        {
            [StructLayout(System.Runtime.InteropServices.LayoutKind.Explicit, Size = 16)]
            public unsafe struct basic_string_char_traits_allocator
            {
                [FieldOffset(0)]
                internal fixed byte _M_local_buf[16];

                [FieldOffset(0)]
                internal ulong _M_allocated_capacity;
            }
        }

    }
}

namespace Kokkos
{
    [StructLayout(System.Runtime.InteropServices.LayoutKind.Sequential)]
    public unsafe struct NativeString : IDisposable
    {
        //[MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        //static NativeString()
        //{
        //    if(!KokkosLibrary.IsInitialized())
        //    {
        //        KokkosLibraryException.Throw("Kokkos Library is not initialized. Use ParallelProcessor.Initialize/Shutdown in the main routine/thread.");
        //    }
        //}

        public int Length;

        public IntPtr Bytes;

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public NativeString(string @string)
        {
            if(@string[^1] != char.MinValue)
            {
                @string += char.MinValue;
            }

            Length = @string.Length;

            Bytes = KokkosLibrary.Allocate(ExecutionSpaceKind.Serial,
                                           (ulong)Length);

            byte[] bytes;

            if(RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
            {
                bytes = Encoding.ASCII.GetBytes(@string);
            }
            else
            {
                bytes = Encoding.UTF8.GetBytes(@string);
            }

            byte* bytePtr = (byte*)Bytes.ToPointer();

            for(int i = 0; i < Length; i++)
            {
                bytePtr[i] = bytes[i];
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public NativeString(byte* bytes,
                            int   length)
        {
            Length = length;

            Bytes = KokkosLibrary.Allocate(ExecutionSpaceKind.Serial,
                                           (ulong)Length);

            byte* bytePtr = (byte*)Bytes.ToPointer();

            int index = 0;

            while(bytes[index] != char.MinValue)
            {
                bytePtr[index] = bytes[index];
                ++index;
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public override string ToString()
        {
            return FromToBytes(Bytes,
                               Length - 1);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static byte* ToBytes(string @string)
        {
            if(@string[^1] != char.MinValue)
            {
                @string += char.MinValue;
            }

            int sizeInBytes = @string.Length * sizeof(byte);

            byte* bytesPtr = (byte*)KokkosLibrary.Allocate(ExecutionSpaceKind.Serial,
                                                           (ulong)sizeInBytes).ToPointer();

            byte[] bytes;

            if(RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
            {
                bytes = Encoding.ASCII.GetBytes(@string);
            }
            else
            {
                bytes = Encoding.UTF8.GetBytes(@string);
            }

            for(int i = 0; i < @string.Length; i++)
            {
                bytesPtr[i] = bytes[i];
            }

            return bytesPtr;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static string FromToBytes(IntPtr bytesIntPtr,
                                           int    length)
        {
            byte* bytesPtr = (byte*)bytesIntPtr.ToPointer();

            //int lengthToEncode = bytesPtr[length - 1] == char.MinValue ? length - 1 : length;

            byte[] bytes = new ReadOnlySpan<byte>(bytesPtr,
                                                  length).ToArray();

            if(RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
            {
                return Encoding.ASCII.GetString(bytes,
                                                0,
                                                length);
            }

            return Encoding.UTF8.GetString(bytes,
                                           0,
                                           length);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public void Dispose()
        {
            DisposeUnmanaged();
            //GC.SuppressFinalize(this);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        private void DisposeUnmanaged()
        {
            if(Bytes != IntPtr.Zero)
            {
                KokkosLibrary.Free(ExecutionSpaceKind.Serial,
                                   Bytes);

                Bytes  = IntPtr.Zero;
                Length = 0;
            }
        }
    }
}