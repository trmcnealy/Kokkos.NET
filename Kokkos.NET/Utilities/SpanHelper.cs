using System;
using System.Diagnostics.Contracts;
using System.Numerics;
using System.Runtime.CompilerServices;

namespace Kokkos.Utilities
{
    internal static class SpanHelper
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static unsafe int GetDjb2HashCode<T>(ref T  r0,
                                                    nint n)
            where T : notnull
        {
            nint length = n;
            int    hash   = 5381;

            nint offset = default;
            
            T _r0 = r0;

            while((byte*)length >= (byte*)8)
            {
                hash = unchecked(((hash << 5) + hash) ^ Unsafe.Add(ref _r0, offset + 0).GetHashCode());
                hash = unchecked(((hash << 5) + hash) ^ Unsafe.Add(ref _r0, offset + 1).GetHashCode());
                hash = unchecked(((hash << 5) + hash) ^ Unsafe.Add(ref _r0, offset + 2).GetHashCode());
                hash = unchecked(((hash << 5) + hash) ^ Unsafe.Add(ref _r0, offset + 3).GetHashCode());
                hash = unchecked(((hash << 5) + hash) ^ Unsafe.Add(ref _r0, offset + 4).GetHashCode());
                hash = unchecked(((hash << 5) + hash) ^ Unsafe.Add(ref _r0, offset + 5).GetHashCode());
                hash = unchecked(((hash << 5) + hash) ^ Unsafe.Add(ref _r0, offset + 6).GetHashCode());
                hash = unchecked(((hash << 5) + hash) ^ Unsafe.Add(ref _r0, offset + 7).GetHashCode());

                length -= 8;
                offset += 8;
            }

            if((byte*)length >= (byte*)4)
            {
                hash = unchecked(((hash << 5) + hash) ^ Unsafe.Add(ref _r0, offset + 0).GetHashCode());
                hash = unchecked(((hash << 5) + hash) ^ Unsafe.Add(ref _r0, offset + 1).GetHashCode());
                hash = unchecked(((hash << 5) + hash) ^ Unsafe.Add(ref _r0, offset + 2).GetHashCode());
                hash = unchecked(((hash << 5) + hash) ^ Unsafe.Add(ref _r0, offset + 3).GetHashCode());

                length -= 4;
                offset += 4;
            }

            while((byte*)length > (byte*)0)
            {
                hash = unchecked(((hash << 5) + hash) ^ Unsafe.Add(ref _r0, offset).GetHashCode());

                length -= 1;
                offset += 1;
            }
            
            r0 = _r0;

            return hash;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static unsafe int GetDjb2LikeByteHash(ref byte r0,
                                                     nint   n)
        {
            nint length = n;

            int    hash   = 5381;

            nint offset = default;
            
            byte _r0 = r0;

            if(Vector.IsHardwareAccelerated && (byte*)length >= (byte*)(Vector<byte>.Count << 3))
            {
                Vector<int> vh  = new Vector<int>(5381);
                Vector<int> v33 = new Vector<int>(33);

                while((byte*)length >= (byte*)(Vector<byte>.Count << 3))
                {
                    ref byte    ri0 = ref Unsafe.Add(ref _r0, offset + Vector<byte>.Count * 0);
                    Vector<int> vi0 = Unsafe.ReadUnaligned<Vector<int>>(ref ri0);
                    Vector<int> vp0 = Vector.Multiply(vh, v33);
                    vh = Vector.Xor(vp0, vi0);

                    ref byte    ri1 = ref Unsafe.Add(ref _r0, offset + Vector<byte>.Count * 1);
                    Vector<int> vi1 = Unsafe.ReadUnaligned<Vector<int>>(ref ri1);
                    Vector<int> vp1 = Vector.Multiply(vh, v33);
                    vh = Vector.Xor(vp1, vi1);

                    ref byte    ri2 = ref Unsafe.Add(ref _r0, offset + Vector<byte>.Count * 2);
                    Vector<int> vi2 = Unsafe.ReadUnaligned<Vector<int>>(ref ri2);
                    Vector<int> vp2 = Vector.Multiply(vh, v33);
                    vh = Vector.Xor(vp2, vi2);

                    ref byte    ri3 = ref Unsafe.Add(ref _r0, offset + Vector<byte>.Count * 3);
                    Vector<int> vi3 = Unsafe.ReadUnaligned<Vector<int>>(ref ri3);
                    Vector<int> vp3 = Vector.Multiply(vh, v33);
                    vh = Vector.Xor(vp3, vi3);

                    ref byte    ri4 = ref Unsafe.Add(ref _r0, offset + Vector<byte>.Count * 4);
                    Vector<int> vi4 = Unsafe.ReadUnaligned<Vector<int>>(ref ri4);
                    Vector<int> vp4 = Vector.Multiply(vh, v33);
                    vh = Vector.Xor(vp4, vi4);

                    ref byte    ri5 = ref Unsafe.Add(ref _r0, offset + Vector<byte>.Count * 5);
                    Vector<int> vi5 = Unsafe.ReadUnaligned<Vector<int>>(ref ri5);
                    Vector<int> vp5 = Vector.Multiply(vh, v33);
                    vh = Vector.Xor(vp5, vi5);

                    ref byte    ri6 = ref Unsafe.Add(ref _r0, offset + Vector<byte>.Count * 6);
                    Vector<int> vi6 = Unsafe.ReadUnaligned<Vector<int>>(ref ri6);
                    Vector<int> vp6 = Vector.Multiply(vh, v33);
                    vh = Vector.Xor(vp6, vi6);

                    ref byte    ri7 = ref Unsafe.Add(ref _r0, offset + Vector<byte>.Count * 7);
                    Vector<int> vi7 = Unsafe.ReadUnaligned<Vector<int>>(ref ri7);
                    Vector<int> vp7 = Vector.Multiply(vh, v33);
                    vh = Vector.Xor(vp7, vi7);

                    length -= Vector<byte>.Count << 3;
                    offset += Vector<byte>.Count << 3;
                }

                while((byte*)length >= (byte*)Vector<byte>.Count)
                {
                    ref byte    ri = ref Unsafe.Add(ref _r0, offset);
                    Vector<int> vi = Unsafe.ReadUnaligned<Vector<int>>(ref ri);
                    Vector<int> vp = Vector.Multiply(vh, v33);
                    vh = Vector.Xor(vp, vi);

                    length -= Vector<byte>.Count;
                    offset += Vector<byte>.Count;
                }

                for(int j = 0; j < Vector<int>.Count; ++j)
                {
                    hash = unchecked(((hash << 5) + hash) ^ vh[j]);
                }
            }
            else
            {
                if(sizeof(nint) == sizeof(ulong))
                {
                    while((byte*)length >= (byte*)(sizeof(ulong) << 3))
                    {
                        ref byte ri0    = ref Unsafe.Add(ref _r0, offset + sizeof(ulong) * 0);
                        ulong    value0 = Unsafe.ReadUnaligned<ulong>(ref ri0);
                        hash = unchecked(((hash << 5) + hash) ^ (int)value0 ^ (int)(value0 >> 32));

                        ref byte ri1    = ref Unsafe.Add(ref _r0, offset + sizeof(ulong) * 1);
                        ulong    value1 = Unsafe.ReadUnaligned<ulong>(ref ri1);
                        hash = unchecked(((hash << 5) + hash) ^ (int)value1 ^ (int)(value1 >> 32));

                        ref byte ri2    = ref Unsafe.Add(ref _r0, offset + sizeof(ulong) * 2);
                        ulong    value2 = Unsafe.ReadUnaligned<ulong>(ref ri2);
                        hash = unchecked(((hash << 5) + hash) ^ (int)value2 ^ (int)(value2 >> 32));

                        ref byte ri3    = ref Unsafe.Add(ref _r0, offset + sizeof(ulong) * 3);
                        ulong    value3 = Unsafe.ReadUnaligned<ulong>(ref ri3);
                        hash = unchecked(((hash << 5) + hash) ^ (int)value3 ^ (int)(value3 >> 32));

                        ref byte ri4    = ref Unsafe.Add(ref _r0, offset + sizeof(ulong) * 4);
                        ulong    value4 = Unsafe.ReadUnaligned<ulong>(ref ri4);
                        hash = unchecked(((hash << 5) + hash) ^ (int)value4 ^ (int)(value4 >> 32));

                        ref byte ri5    = ref Unsafe.Add(ref _r0, offset + sizeof(ulong) * 5);
                        ulong    value5 = Unsafe.ReadUnaligned<ulong>(ref ri5);
                        hash = unchecked(((hash << 5) + hash) ^ (int)value5 ^ (int)(value5 >> 32));

                        ref byte ri6    = ref Unsafe.Add(ref _r0, offset + sizeof(ulong) * 6);
                        ulong    value6 = Unsafe.ReadUnaligned<ulong>(ref ri6);
                        hash = unchecked(((hash << 5) + hash) ^ (int)value6 ^ (int)(value6 >> 32));

                        ref byte ri7    = ref Unsafe.Add(ref _r0, offset + sizeof(ulong) * 7);
                        ulong    value7 = Unsafe.ReadUnaligned<ulong>(ref ri7);
                        hash = unchecked(((hash << 5) + hash) ^ (int)value7 ^ (int)(value7 >> 32));

                        length -= sizeof(ulong) << 3;
                        offset += sizeof(ulong) << 3;
                    }
                }

                while((byte*)length >= (byte*)(sizeof(uint) << 3))
                {
                    ref byte ri0    = ref Unsafe.Add(ref _r0, offset + sizeof(uint) * 0);
                    uint     value0 = Unsafe.ReadUnaligned<uint>(ref ri0);
                    hash = unchecked(((hash << 5) + hash) ^ (int)value0);

                    ref byte ri1    = ref Unsafe.Add(ref _r0, offset + sizeof(uint) * 1);
                    uint     value1 = Unsafe.ReadUnaligned<uint>(ref ri1);
                    hash = unchecked(((hash << 5) + hash) ^ (int)value1);

                    ref byte ri2    = ref Unsafe.Add(ref _r0, offset + sizeof(uint) * 2);
                    uint     value2 = Unsafe.ReadUnaligned<uint>(ref ri2);
                    hash = unchecked(((hash << 5) + hash) ^ (int)value2);

                    ref byte ri3    = ref Unsafe.Add(ref _r0, offset + sizeof(uint) * 3);
                    uint     value3 = Unsafe.ReadUnaligned<uint>(ref ri3);
                    hash = unchecked(((hash << 5) + hash) ^ (int)value3);

                    ref byte ri4    = ref Unsafe.Add(ref _r0, offset + sizeof(uint) * 4);
                    uint     value4 = Unsafe.ReadUnaligned<uint>(ref ri4);
                    hash = unchecked(((hash << 5) + hash) ^ (int)value4);

                    ref byte ri5    = ref Unsafe.Add(ref _r0, offset + sizeof(uint) * 5);
                    uint     value5 = Unsafe.ReadUnaligned<uint>(ref ri5);
                    hash = unchecked(((hash << 5) + hash) ^ (int)value5);

                    ref byte ri6    = ref Unsafe.Add(ref _r0, offset + sizeof(uint) * 6);
                    uint     value6 = Unsafe.ReadUnaligned<uint>(ref ri6);
                    hash = unchecked(((hash << 5) + hash) ^ (int)value6);

                    ref byte ri7    = ref Unsafe.Add(ref _r0, offset + sizeof(uint) * 7);
                    uint     value7 = Unsafe.ReadUnaligned<uint>(ref ri7);
                    hash = unchecked(((hash << 5) + hash) ^ (int)value7);

                    length -= sizeof(uint) << 3;
                    offset += sizeof(uint) << 3;
                }
            }

            if((byte*)length >= (byte*)(sizeof(ushort) << 3))
            {
                ref byte ri0    = ref Unsafe.Add(ref _r0, offset + sizeof(ushort) * 0);
                ushort   value0 = Unsafe.ReadUnaligned<ushort>(ref ri0);
                hash = unchecked(((hash << 5) + hash) ^ value0);

                ref byte ri1    = ref Unsafe.Add(ref _r0, offset + sizeof(ushort) * 1);
                ushort   value1 = Unsafe.ReadUnaligned<ushort>(ref ri1);
                hash = unchecked(((hash << 5) + hash) ^ value1);

                ref byte ri2    = ref Unsafe.Add(ref _r0, offset + sizeof(ushort) * 2);
                ushort   value2 = Unsafe.ReadUnaligned<ushort>(ref ri2);
                hash = unchecked(((hash << 5) + hash) ^ value2);

                ref byte ri3    = ref Unsafe.Add(ref _r0, offset + sizeof(ushort) * 3);
                ushort   value3 = Unsafe.ReadUnaligned<ushort>(ref ri3);
                hash = unchecked(((hash << 5) + hash) ^ value3);

                ref byte ri4    = ref Unsafe.Add(ref _r0, offset + sizeof(ushort) * 4);
                ushort   value4 = Unsafe.ReadUnaligned<ushort>(ref ri4);
                hash = unchecked(((hash << 5) + hash) ^ value4);

                ref byte ri5    = ref Unsafe.Add(ref _r0, offset + sizeof(ushort) * 5);
                ushort   value5 = Unsafe.ReadUnaligned<ushort>(ref ri5);
                hash = unchecked(((hash << 5) + hash) ^ value5);

                ref byte ri6    = ref Unsafe.Add(ref _r0, offset + sizeof(ushort) * 6);
                ushort   value6 = Unsafe.ReadUnaligned<ushort>(ref ri6);
                hash = unchecked(((hash << 5) + hash) ^ value6);

                ref byte ri7    = ref Unsafe.Add(ref _r0, offset + sizeof(ushort) * 7);
                ushort   value7 = Unsafe.ReadUnaligned<ushort>(ref ri7);
                hash = unchecked(((hash << 5) + hash) ^ value7);

                length -= sizeof(ushort) << 3;
                offset += sizeof(ushort) << 3;
            }

            while((byte*)length > (byte*)0)
            {
                hash = unchecked(((hash << 5) + hash) ^ Unsafe.Add(ref _r0, offset));

                length -= 1;
                offset += 1;
            }
            
            r0 = _r0;

            return hash;
        }

        [Pure]
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static int Count<T>(ref T  r0,
                                   nint n,
                                   T      value)
            where T : IEquatable<T>
        {
            nint length = n;

            if(!Vector.IsHardwareAccelerated)
            {
                return CountSequential(ref r0, length, value);
            }
            
            T _r0 = r0;

            // Special vectorized version when using a supported type
            if(typeof(T) == typeof(byte) || typeof(T) == typeof(sbyte) || typeof(T) == typeof(bool))
            {
                ref sbyte r1     = ref Unsafe.As<T, sbyte>(ref _r0);
                sbyte     target = Unsafe.As<T, sbyte>(ref value);

                return CountSimd(ref r1, length, target, (nint)sbyte.MaxValue);
            }

            if(typeof(T) == typeof(char) || typeof(T) == typeof(ushort) || typeof(T) == typeof(short))
            {
                ref short r1     = ref Unsafe.As<T, short>(ref _r0);
                short     target = Unsafe.As<T, short>(ref value);

                return CountSimd(ref r1, length, target, (nint)short.MaxValue);
            }

            if(typeof(T) == typeof(int) || typeof(T) == typeof(uint))
            {
                ref int r1     = ref Unsafe.As<T, int>(ref _r0);
                int     target = Unsafe.As<T, int>(ref value);

                return CountSimd(ref r1, length, target, (nint)int.MaxValue);
            }

            if(typeof(T) == typeof(long) || typeof(T) == typeof(ulong))
            {
                ref long r1     = ref Unsafe.As<T, long>(ref _r0);
                long     target = Unsafe.As<T, long>(ref value);

                return CountSimd(ref r1, length, target, (nint)int.MaxValue);
            }
            
            r0 = _r0;

            return CountSequential(ref r0, length, value);
        }

        [Pure]
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        private static unsafe int CountSequential<T>(ref T  r0,
                                                     nint n,
                                                     T      value)
            where T : IEquatable<T>
        {
            nint length = n;

            int    result = 0;

            nint offset = default;
            
            T _r0 = r0;

            // Main loop with 8 unrolled iterations
            while((byte*)length >= (byte*)8)
            {
                result += Unsafe.Add(ref _r0, offset + 0).Equals(value).ToInt();
                result += Unsafe.Add(ref _r0, offset + 1).Equals(value).ToInt();
                result += Unsafe.Add(ref _r0, offset + 2).Equals(value).ToInt();
                result += Unsafe.Add(ref _r0, offset + 3).Equals(value).ToInt();
                result += Unsafe.Add(ref _r0, offset + 4).Equals(value).ToInt();
                result += Unsafe.Add(ref _r0, offset + 5).Equals(value).ToInt();
                result += Unsafe.Add(ref _r0, offset + 6).Equals(value).ToInt();
                result += Unsafe.Add(ref _r0, offset + 7).Equals(value).ToInt();

                length -= 8;
                offset += 8;
            }

            if((byte*)length >= (byte*)4)
            {
                result += Unsafe.Add(ref _r0, offset + 0).Equals(value).ToInt();
                result += Unsafe.Add(ref _r0, offset + 1).Equals(value).ToInt();
                result += Unsafe.Add(ref _r0, offset + 2).Equals(value).ToInt();
                result += Unsafe.Add(ref _r0, offset + 3).Equals(value).ToInt();

                length -= 4;
                offset += 4;
            }

            // Iterate over the remaining values and count those that match
            while((byte*)length > (byte*)0)
            {
                result += Unsafe.Add(ref _r0, offset).Equals(value).ToInt();

                length -= 1;
                offset += 1;
            }
            
            r0 = _r0;

            return result;
        }

        [Pure]
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        private static unsafe int CountSimd<T>(ref T  r0,
                                               nint n,
                                               T      value,
                                               nint max)
            where T : unmanaged, IEquatable<T>
        {
            nint length = n;

            int result = 0;

            nint offset = default;
            
            T _r0 = r0;

            if((byte*)length >= (byte*)Vector<T>.Count)
            {
                Vector<T> vc = new Vector<T>(value);

                do
                {
                    // Calculate the maximum sequential area that can be processed in
                    // one pass without the risk of numeric overflow in the dot product
                    // to sum the partial results. We also backup the current offset to
                    // be able to track how many items have been processed, which lets
                    // us avoid updating a third counter (length) in the loop body.
                    nint chunkLength = Min(length, max), initialOffset = offset;

                    Vector<T> partials = Vector<T>.Zero;

                    while((byte*)chunkLength >= (byte*)Vector<T>.Count)
                    {
                        ref T ri = ref Unsafe.Add(ref _r0, offset);

                        // Load the current Vector<T> register, and then use
                        // Vector.Equals to check for matches. This API sets the
                        // values corresponding to matching pairs to all 1s.
                        // Since the input type is guaranteed to always be signed,
                        // this means that a value with all 1s represents -1, as
                        // signed numbers are represented in two's complement.
                        // So we can just subtract this intermediate value to the
                        // partial results, which effectively sums 1 for each match.
                        Vector<T> vi = Unsafe.As<T, Vector<T>>(ref ri);
                        Vector<T> ve = Vector.Equals(vi, vc);

                        partials -= ve;

                        chunkLength -= Vector<T>.Count;
                        offset      += Vector<T>.Count;
                    }

                    result += CastToInt(Vector.Dot(partials, Vector<T>.One));

                    length = Subtract(length, Subtract(offset, initialOffset));
                } while((byte*)length >= (byte*)Vector<T>.Count);
            }

            if(Vector<T>.Count > 8 && (byte*)length >= (byte*)8)
            {
                result += Unsafe.Add(ref _r0, offset + 0).Equals(value).ToInt();
                result += Unsafe.Add(ref _r0, offset + 1).Equals(value).ToInt();
                result += Unsafe.Add(ref _r0, offset + 2).Equals(value).ToInt();
                result += Unsafe.Add(ref _r0, offset + 3).Equals(value).ToInt();
                result += Unsafe.Add(ref _r0, offset + 4).Equals(value).ToInt();
                result += Unsafe.Add(ref _r0, offset + 5).Equals(value).ToInt();
                result += Unsafe.Add(ref _r0, offset + 6).Equals(value).ToInt();
                result += Unsafe.Add(ref _r0, offset + 7).Equals(value).ToInt();

                length -= 8;
                offset += 8;
            }

            if(Vector<T>.Count > 4 && (byte*)length >= (byte*)4)
            {
                result += Unsafe.Add(ref _r0, offset + 0).Equals(value).ToInt();
                result += Unsafe.Add(ref _r0, offset + 1).Equals(value).ToInt();
                result += Unsafe.Add(ref _r0, offset + 2).Equals(value).ToInt();
                result += Unsafe.Add(ref _r0, offset + 3).Equals(value).ToInt();

                length -= 4;
                offset += 4;
            }

            while((byte*)length > (byte*)0)
            {
                result += Unsafe.Add(ref _r0, offset).Equals(value).ToInt();

                length -= 1;
                offset += 1;
            }

            r0 = _r0;
            
            return result;
        }

        [Pure]
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        private static unsafe nint Min(nint a,
                                         nint b)
        {
            if(sizeof(nint) == 4)
            {
                return (nint)Math.Min((int)a, (int)b);
            }

            return (nint)Math.Min((long)a, (long)b);
        }

        [Pure]
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        private static unsafe nint Subtract(nint a,
                                              nint b)
        {
            if(sizeof(nint) == 4)
            {
                return (nint)((int)a - (int)b);
            }

            return (nint)((long)a - (long)b);
        }

        [Pure]
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        private static int CastToInt<T>(T value)
            where T : unmanaged
        {
            if(typeof(T) == typeof(sbyte))
            {
                return Unsafe.As<T, sbyte>(ref value);
            }

            if(typeof(T) == typeof(short))
            {
                return Unsafe.As<T, short>(ref value);
            }

            if(typeof(T) == typeof(int))
            {
                return Unsafe.As<T, int>(ref value);
            }

            if(typeof(T) == typeof(long))
            {
                return (int)Unsafe.As<T, long>(ref value);
            }

            throw new NotSupportedException($"Invalid input type {typeof(T)}");
        }
    }
}