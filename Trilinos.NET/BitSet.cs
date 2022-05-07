using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading.Tasks;

namespace Trilinos
{
    public sealed class BitSet
    {
        private readonly uint _length;
        private bool[] _bits;

        public uint Length
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            get { return _length; }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public BitSet(uint length)
        {
            _length = length;
            _bits = new bool[_length];
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public BitSet(BitSet bitSet)
        {
            _length = bitSet.Length;
            _bits = new bool[_length];

            Buffer.BlockCopy(bitSet._bits, 0, _bits, 0, (int)_length);
        }

        public bool this[uint index]
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            get { return _bits[index]; }
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            set { _bits[index] = value; }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public bool Equals(BitSet? b)
        {
            if (b == null)
            {
                return false;
            }

            if (Length != b.Length)
            {
                return false;
            }

            for (uint i = 0; i < _bits.Length; i++)
            {
                if (b._bits[i] != _bits[i])
                {
                    return false;
                }
            }

            return true;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public void Clear(uint index)
        {
            _bits[index] = false;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public void Clear(uint startIndex, uint endIndex)
        {
            for (uint i = startIndex; i < endIndex; i++)
            {
                Clear(i);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public bool Get(uint index)
        {
            return _bits[index];
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public BitSet Set(uint index)
        {
            _bits[index] = true;

            return this;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public BitSet Set(uint startIndex, uint endIndex)
        {
            for (uint i = startIndex; i < endIndex; i++)
            {
                Set(i);
            }

            return this;
        }
        
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public void Reset()
        {
            Array.Fill(_bits, false);
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public BitSet Flip()
        {
            for (uint i = 0; i < _length; i++)
            {
                _bits[i] = !_bits[i];
            }

            return this;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public void Xor(BitSet bitSet)
        {
            uint len = Math.Min(bitSet.Length, Length);

            for (uint i = 0; i < len; i++)
            {
                if (Get(i) == bitSet.Get(i))
                {
                    Clear(i);
                }
                else
                {
                    Set(i);
                }
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public void Or(BitSet bitSet)
        {
            uint len = Math.Min(bitSet.Length, Length);

            for (uint i = 0; i < len; i++)
            {
                if (Get(i) || bitSet.Get(i))
                {
                    Set(i);
                }
                else
                {
                    Clear(i);
                }
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public void And(BitSet bitSet)
        {
            uint len = Math.Min(bitSet.Length, Length);

            for (uint i = 0; i < len; i++)
            {
                if (Get(i) && bitSet.Get(i))
                {
                    Set(i);
                }
                else
                {
                    Clear(i);
                }
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public void AndNot(BitSet bitSet)
        {
            uint len = Math.Min(bitSet.Length, Length);

            for (uint i = 0; i < len; i++)
            {
                if (!Get(i) || !bitSet.Get(i))
                {
                    Set(i);
                }
                else
                {
                    Clear(i);
                }
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public bool Intersects(BitSet bitSet)
        {
            uint len = Math.Min(bitSet.Length, Length);

            bool v;
            for (uint i = 0; i < len; i++)
            {
                v = Get(i);

                if (v == bitSet.Get(i) && v == true)
                {
                    return true;
                }
            }

            return false;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static BitSet operator <<(BitSet lhs, int rhs)
        {
            BitSet bitSet = new BitSet(lhs.Length);

            if (rhs > lhs.Length)
            {
                return bitSet;
            }

            uint pos = (uint)rhs;

            Buffer.BlockCopy(lhs._bits, 0, bitSet._bits, (int)pos, (int)(lhs.Length - pos));

            return bitSet;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static BitSet operator >>(BitSet lhs, int rhs)
        {
            BitSet bitSet = new BitSet(lhs.Length);

            if (rhs > lhs.Length)
            {
                return bitSet;
            }

            uint pos = (uint)rhs;

            Buffer.BlockCopy(lhs._bits, (int)pos, bitSet._bits, 0, (int)(lhs.Length - pos));

            return bitSet;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static BitSet operator ^(BitSet lhs, BitSet rhs)
        {
            BitSet bitSet = new BitSet(lhs);

            bitSet.Xor(rhs);

            return bitSet;

        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static BitSet operator |(BitSet lhs, BitSet rhs)
        {
            BitSet bitSet = new BitSet(lhs);

            bitSet.Or(rhs);

            return bitSet;

        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static BitSet operator &(BitSet lhs, BitSet rhs)
        {
            BitSet bitSet = new BitSet(lhs);

            bitSet.And(rhs);

            return bitSet;

        }
        
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public override string ToString()
        {
            char[] str = new char[_length];

            for (uint i = 0; i < _length; i++)
            {
                str[i] = _bits[i] ? '1' : '0';
            }

            return new string(str);
        }
    }
}
