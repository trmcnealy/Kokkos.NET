#nullable enable
using System;
using System.Buffers;
using System.Diagnostics;
using System.IO;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Threading;
using System.Threading.Tasks;
using System.Buffers.Binary;
using System.Linq;
using System.Text;

namespace Kokkos
{
    public sealed class MemoryMappedStream : Stream
    {
        private readonly nint _content;
        private readonly long   _length;

        private long _position;

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public MemoryMappedStream(nint content,
                                  long   length)
        {
            _content = content;
            _length  = length;
        }

        public override bool CanRead
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            get { return true; }
        }

        public override bool CanSeek
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            get { return true; }
        }

        public override bool CanWrite
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            get { return false; }
        }

        public override long Length
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            get { return _length; }
        }

        public override long Position
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            get { return _position; }
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            set { _position = value; }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public override long Seek(long       offset,
                                  SeekOrigin origin)
        {
            long pos = origin == SeekOrigin.Begin   ? offset :
                       origin == SeekOrigin.Current ? _position + offset :
                       origin == SeekOrigin.End     ? _length   + offset : throw new ArgumentOutOfRangeException(nameof(origin));

            if(pos > int.MaxValue)
            {
                throw new ArgumentOutOfRangeException(nameof(offset));
            }

            if(pos < 0)
            {
                throw new IOException("IO_SeekBeforeBegin");
            }

            _position = (int)pos;

            return _position;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public unsafe T* GetPointer<T>()
            where T : unmanaged
        {
            return (T*)_content.ToPointer();
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public Span<byte> Slice(long start)
        {
            unsafe
            {
                return new Span<byte>((byte*)_content + start, (int)(_length - start));
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public Span<byte> Slice(long start,
                                int  length)
        {
            unsafe
            {
                return new Span<byte>((byte*)_content + start, length);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public override int ReadByte()
        {
            unsafe
            {
                byte* s = GetPointer<byte>();

                return _position < _length ? s[_position++] : -1;
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public override int Read(byte[] buffer,
                                 int    offset,
                                 int    count)
        {
            return Read(new Span<byte>(buffer, offset, count));
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public override int Read(Span<byte> buffer)
        {
            long remaining = _length - _position;

            if(remaining <= 0 || buffer.Length == 0)
            {
                return 0;
            }

            if(remaining <= buffer.Length)
            {
                Slice(_position).CopyTo(buffer);

                _position = _length;

                return (int)remaining;
            }

            Slice(_position, buffer.Length).CopyTo(buffer);

            _position += buffer.Length;

            return buffer.Length;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public override Task<int> ReadAsync(byte[]            buffer,
                                            int               offset,
                                            int               count,
                                            CancellationToken cancellationToken)
        {
            return cancellationToken.IsCancellationRequested ? Task.FromCanceled<int>(cancellationToken) : Task.FromResult(Read(new Span<byte>(buffer, offset, count)));
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public override ValueTask<int> ReadAsync(Memory<byte>      buffer,
                                                 CancellationToken cancellationToken = default(CancellationToken))
        {
            return cancellationToken.IsCancellationRequested ? new ValueTask<int>(Task.FromCanceled<int>(cancellationToken)) : new ValueTask<int>(Read(buffer.Span));
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public override IAsyncResult BeginRead(byte[]         buffer,
                                               int            offset,
                                               int            count,
                                               AsyncCallback? callback,
                                               object?        state)
        {
            return Task.FromResult(ReadAsync(buffer, offset, count));
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public override int EndRead(IAsyncResult asyncResult)
        {
            if(asyncResult is Task twar && twar is Task<int> task)
            {
                return task.GetAwaiter().GetResult();
            }

            throw new ArgumentNullException(nameof(asyncResult));
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public override void CopyTo(Stream destination,
                                    int    bufferSize)
        {
            if(_length > _position)
            {
                destination.Write(Slice(_position));
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public override Task CopyToAsync(Stream            destination,
                                         int               bufferSize,
                                         CancellationToken cancellationToken)
        {
            return _length > _position ? destination.WriteAsync(new ReadOnlyMemory<byte>(Slice(_position).ToArray()), cancellationToken).AsTask() : Task.CompletedTask;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public override void Flush()
        {
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public override Task FlushAsync(CancellationToken cancellationToken)
        {
            return Task.CompletedTask;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public override void SetLength(long value)
        {
            throw new NotSupportedException();
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public override void Write(byte[] buffer,
                                   int    offset,
                                   int    count)
        {
            throw new NotSupportedException();
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        private byte InternalReadByte()
        {
            int b = ReadByte();

            if(b == -1)
            {
                throw new EndOfStreamException();
            }

            return (byte)b;
        }

        //[MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        //internal ReadOnlySpan<byte> InternalReadSpan(int count)
        //{
        //    long origPos = _position;
        //    long newPos  = origPos + count;

        //    if(newPos > _length)
        //    {
        //        _position = _length;

        //        throw new EndOfStreamException();
        //    }

        //    ReadOnlySpan<byte> span = new ReadOnlySpan<byte>(_content.ToPointer(), (int)origPos, count);

        //    _position = newPos;

        //    return span;
        //}

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        private unsafe byte[] InternalRead(int numBytes)
        {
            //return InternalReadSpan(numBytes);

            long origPos = _position;
            long newPos  = origPos + numBytes;

            if(newPos > _length)
            {
                _position = _length;

                throw new EndOfStreamException();
            }

            byte[] span = new byte[numBytes];

            Buffer.MemoryCopy(_content.ToPointer(), Unsafe.AsPointer(ref span), numBytes, numBytes);

            //for (int i = 0; i < numBytes; ++i)
            //{
            //    span[i] = bytes[i];
            //}

            _position = newPos;

            return span;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public byte ReadAsByte()
        {
            return InternalReadByte();
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public sbyte ReadAsSByte()
        {
            return (sbyte)InternalReadByte();
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public bool ReadAsBoolean()
        {
            return InternalReadByte() != 0;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public char ReadAsChar(int count = 2)
        {
            return Converter.ToChar(InternalRead(count));
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public short ReadAsInt16(int count = 2)
        {
            return Converter.ToInt16(InternalRead(count));
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public ushort ReadAsUInt16(int count = 2)
        {
            return Converter.ToUInt16(InternalRead(count));
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public int ReadAsInt32(int count = 4)
        {
            return Converter.ToInt32(InternalRead(count));
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public uint ReadAsUInt32(int count = 4)
        {
            return Converter.ToUInt32(InternalRead(count));
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public long ReadAsInt64(int count = 8)
        {
            return Converter.ToInt64(InternalRead(count));
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public ulong ReadAsUInt64(int count = 8)
        {
            return Converter.ToUInt64(InternalRead(count));
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public float ReadAsSingle(int count = 4)
        {
            return Converter.ToSingle(InternalRead(count));
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public double ReadAsDouble(int count = 8)
        {
            return Converter.ToDouble(InternalRead(count));
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public string ReadAsString(int count)
        {
            char[] chars = new char[count];

            byte[] bytes = InternalRead(count);

            for(int i = 0; i < count; ++i)
            {
                chars[i] = (char)bytes[i];
            }

            return new string(chars);

            //Converter.ToString(InternalRead(count).ToArray());
        }

        internal static class Converter
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            public static char ToChar(byte[] value,
                                      int    startIndex)
            {
                return unchecked((char)ToInt16(value, startIndex));
            }

            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            public static char ToChar(ReadOnlySpan<byte> value)
            {
                return Unsafe.ReadUnaligned<char>(ref MemoryMarshal.GetReference(value));
            }

            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            public static short ToInt16(byte[] value,
                                        int    startIndex)
            {
                return Unsafe.ReadUnaligned<short>(ref value[startIndex]);
            }

            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            public static short ToInt16(ReadOnlySpan<byte> value)
            {
                return Unsafe.ReadUnaligned<short>(ref MemoryMarshal.GetReference(value));
            }

            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            public static int ToInt32(byte[] value,
                                      int    startIndex)
            {
                return Unsafe.ReadUnaligned<int>(ref value[startIndex]);
            }

            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            public static int ToInt32(ReadOnlySpan<byte> value)
            {
                return Unsafe.ReadUnaligned<int>(ref MemoryMarshal.GetReference(value));
            }

            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            public static long ToInt64(byte[] value,
                                       int    startIndex)
            {
                return Unsafe.ReadUnaligned<long>(ref value[startIndex]);
            }

            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            public static long ToInt64(ReadOnlySpan<byte> value)
            {
                return Unsafe.ReadUnaligned<long>(ref MemoryMarshal.GetReference(value));
            }

            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            public static ushort ToUInt16(byte[] value,
                                          int    startIndex)
            {
                return unchecked((ushort)ToInt16(value, startIndex));
            }

            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            public static ushort ToUInt16(ReadOnlySpan<byte> value)
            {
                return Unsafe.ReadUnaligned<ushort>(ref MemoryMarshal.GetReference(value));
            }

            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            public static uint ToUInt32(byte[] value,
                                        int    startIndex)
            {
                return unchecked((uint)ToInt32(value, startIndex));
            }

            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            public static uint ToUInt32(ReadOnlySpan<byte> value)
            {
                return Unsafe.ReadUnaligned<uint>(ref MemoryMarshal.GetReference(value));
            }

            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            public static ulong ToUInt64(byte[] value,
                                         int    startIndex)
            {
                return unchecked((ulong)ToInt64(value, startIndex));
            }

            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            public static ulong ToUInt64(ReadOnlySpan<byte> value)
            {
                return Unsafe.ReadUnaligned<ulong>(ref MemoryMarshal.GetReference(value));
            }

            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            public static float ToSingle(byte[] value,
                                         int    startIndex)
            {
                return Int32BitsToSingle(ToInt32(value, startIndex));
            }

            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            public static float ToSingle(ReadOnlySpan<byte> value)
            {
                return Unsafe.ReadUnaligned<float>(ref MemoryMarshal.GetReference(value));
            }

            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            public static double ToDouble(byte[] value,
                                          int    startIndex)
            {
                return Int64BitsToDouble(ToInt64(value, startIndex));
            }

            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            public static double ToDouble(ReadOnlySpan<byte> value)
            {
                return Unsafe.ReadUnaligned<double>(ref MemoryMarshal.GetReference(value));
            }

            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            public static string ToString(byte[] value)
            {
                return ToString(value, 0, value.Length);
            }

            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            public static string ToString(byte[] value,
                                          int    startIndex)
            {
                return ToString(value, startIndex, value.Length - startIndex);
            }

            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            public static string ToString(byte[] value,
                                          int    startIndex,
                                          int    length)
            {
                return Encoding.UTF8.GetString(value, startIndex, length);
            }

            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            public static unsafe long DoubleToInt64Bits(double value)
            {
                return *((long*)&value);
            }

            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            public static unsafe double Int64BitsToDouble(long value)
            {
                return *((double*)&value);
            }

            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            public static unsafe int SingleToInt32Bits(float value)
            {
                return *((int*)&value);
            }

            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            public static unsafe float Int32BitsToSingle(int value)
            {
                return *((float*)&value);
            }
        }
    }
}
