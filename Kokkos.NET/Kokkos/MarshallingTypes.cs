// ReSharper disable InconsistentNaming

using System;
using System.Net;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace Kokkos
{
    [StructLayout(System.Runtime.InteropServices.LayoutKind.Explicit, Size = sizeof(ulong))]
    public struct ValueType
    {
        [FieldOffset(0)]
        public float Single;

        [FieldOffset(0)]
        public double Double;

        [FieldOffset(0)]
        public bool Bool;

        [FieldOffset(0)]
        public sbyte Int8;

        [FieldOffset(0)]
        public byte UInt8;

        [FieldOffset(0)]
        public short Int16;

        [FieldOffset(0)]
        public ushort UInt16;

        [FieldOffset(0)]
        public int Int32;

        [FieldOffset(0)]
        public uint UInt32;

        [FieldOffset(0)]
        public long Int64;
        
        [FieldOffset(0)]
        public ulong UInt64;

        [FieldOffset(0)]
        public IntPtr ByRef;

        public TDataType As<TDataType>() where TDataType : struct
        {
            if(typeof(TDataType) == typeof(float))
            {
                return (TDataType)(System.ValueType)Single;
            }
            if(typeof(TDataType) == typeof(double))
            {
                return (TDataType)(System.ValueType)Double;
            }
            if(typeof(TDataType) == typeof(bool))
            {
                return (TDataType)(System.ValueType)Bool;
            }
            if(typeof(TDataType) == typeof(sbyte))
            {
                return (TDataType)(System.ValueType)Int8;
            }
            if(typeof(TDataType) == typeof(byte))
            {
                return (TDataType)(System.ValueType)UInt8;
            }
            if(typeof(TDataType) == typeof(short))
            {
                return (TDataType)(System.ValueType)Int16;
            }
            if(typeof(TDataType) == typeof(ushort))
            {
                return (TDataType)(System.ValueType)UInt16;
            }
            if(typeof(TDataType) == typeof(int))
            {
                return (TDataType)(System.ValueType)Int32;
            }
            if(typeof(TDataType) == typeof(uint))
            {
                return (TDataType)(System.ValueType)UInt32;
            }
            if(typeof(TDataType) == typeof(long))
            {
                return (TDataType)(System.ValueType)Int64;
            }
            if(typeof(TDataType) == typeof(ulong))
            {
                return (TDataType)(System.ValueType)UInt64;
            }

            throw new NotSupportedException();
        }

        public static ValueType From<TDataType>(TDataType value) where TDataType : struct
        {
            if(typeof(TDataType) == typeof(float))
            {
                return new ValueType() {Single = (float)(System.ValueType)value};
            }
            if(typeof(TDataType) == typeof(double))
            {
                return new ValueType() {Double = (double)(System.ValueType)value};
            }
            if(typeof(TDataType) == typeof(bool))
            {
                return new ValueType() {Bool = (bool)(System.ValueType)value};
            }
            if(typeof(TDataType) == typeof(sbyte))
            {
                return new ValueType() {Int8 = (sbyte)(System.ValueType)value};
            }
            if(typeof(TDataType) == typeof(byte))
            {
                return new ValueType() {UInt8 = (byte)(System.ValueType)value};
            }
            if(typeof(TDataType) == typeof(short))
            {
                return new ValueType() {Int16 = (short)(System.ValueType)value};
            }
            if(typeof(TDataType) == typeof(ushort))
            {
                return new ValueType() {UInt16 = (ushort)(System.ValueType)value};
            }
            if(typeof(TDataType) == typeof(int))
            {
                return new ValueType() {Int32 = (int)(System.ValueType)value};
            }
            if(typeof(TDataType) == typeof(uint))
            {
                return new ValueType() {UInt32 = (uint)(System.ValueType)value};
            }
            if(typeof(TDataType) == typeof(long))
            {
                return new ValueType() {Int64 = (long)(System.ValueType)value};
            }
            if(typeof(TDataType) == typeof(ulong))
            {
                return new ValueType() {UInt64 = (ulong)(System.ValueType)value};
            }

            throw new NotSupportedException();
        }

        public static implicit operator float(ValueType from) => from.Single;
        public static implicit operator double (ValueType from) => from.Double;
        public static implicit operator bool (ValueType from) => from.Bool;
        public static implicit operator sbyte (ValueType from) => from.Int8;
        public static implicit operator byte (ValueType from) => from.UInt8;
        public static implicit operator short (ValueType from) => from.Int16;
        public static implicit operator ushort (ValueType from) => from.UInt16;
        public static implicit operator int (ValueType from) => from.Int32;
        public static implicit operator uint (ValueType from) => from.UInt32;
        public static implicit operator long (ValueType from) => from.Int64;
        public static implicit operator ulong (ValueType from) => from.UInt64;

        public static explicit operator ValueType(float to) => new ValueType(){Single =to};
        public static explicit operator ValueType(double  to) => new ValueType(){Double=to};
        public static explicit operator ValueType(bool to) => new ValueType(){Bool =to};
        public static explicit operator ValueType(sbyte to) => new ValueType(){Int8 =to};
        public static explicit operator ValueType(byte to) => new ValueType(){UInt8=to};
        public static explicit operator ValueType(short to) => new ValueType(){Int16=to};
        public static explicit operator ValueType(ushort to) => new ValueType(){UInt16=to};
        public static explicit operator ValueType(int to) => new ValueType(){Int32=to};
        public static explicit operator ValueType(uint to) => new ValueType(){UInt32=to};
        public static explicit operator ValueType(long to) => new ValueType(){Int64=to};
        public static explicit operator ValueType(ulong to) => new ValueType(){UInt64=to};
    }

    [StructLayout(System.Runtime.InteropServices.LayoutKind.Sequential)]
    public struct type_info : IEquatable<type_info>
    {
        [MarshalAs(UnmanagedType.LPStr)]
        public readonly string name;

        public type_info(string value)
        {
            name = value;
        }

        public bool Equals(type_info other)
        {
            return name == other.name;
        }

        public override bool Equals(object obj)
        {
            return obj is type_info other && Equals(other);
        }

        public override int GetHashCode()
        {
            return name != null ? name.GetHashCode() : 0;
        }

        public static bool operator ==(type_info left,
                                       type_info right)
        {
            return left.Equals(right);
        }

        public static bool operator !=(type_info left,
                                       type_info right)
        {
            return !left.Equals(right);
        }
    }

    [StructLayout(System.Runtime.InteropServices.LayoutKind.Sequential)]
    public struct any
    {
        private IntPtr /*placeholder*/
            content;

        //[System.Runtime.CompilerServices.SpecialName]
        public static any Create<TValueType>(TValueType value)
        {
            unsafe
            {
                holder<TValueType> new_holder = new holder<TValueType>(value);

                return new any()
                {
                    content = (IntPtr)Unsafe.AsPointer(ref new_holder)
                };
            }
        }

        public any(any other)
        {
            content = other.content;
        }

        //public any swap(any rhs)
        //{
        //    std::swap(content,
        //              rhs.content);

        //    return this;
        //}

        //public any CopyFrom<TValueType>(TValueType rhs)
        //{
        //    any(rhs).swap(this);

        //    return this;
        //}

        //public any CopyFrom(any rhs)
        //{
        //    any(rhs).swap(this);

        //    return this;
        //}

        public bool empty()
        {
            return content == IntPtr.Zero;
        }

        //public type_info type()
        //{
        //    return content != null ? content.type() : typeid();
        //}

        //public string typeName()
        //{
        //    return content != null ? content.typeName() : "NONE";
        //}

        public bool same(any other)
        {
            if(empty() && other.empty())
            {
                return true;
            }

            if(empty() && !other.empty())
            {
                return false;
            }

            if(!empty() && other.empty())
            {
                return false;
            }

            return content == other.content;
        }

        //public void print(std::ostream os)
        //{
        //    if(content != null)
        //    {
        //        content.print(os);
        //    }
        //}

        public interface placeholder
        {
            //public abstract type_info type();

            //public abstract string typeName();

            //public abstract placeholder clone();

            //public abstract bool same(placeholder other);

            //public abstract void print(std::ostream os);
        }

        [StructLayout(System.Runtime.InteropServices.LayoutKind.Sequential)]
        public struct holder<TValueType> : placeholder
        {
            public TValueType held;

            public holder(TValueType value)
            {
                held = value;
            }

            //public override type_info type()
            //{
            //    return typeid(TValueType);
            //}

            //public override string typeName()
            //{
            //    return TypeNameTraits<TValueType>.name();
            //}

            public placeholder clone()
            {
                return new holder<TValueType>(held);
            }

            public bool same(placeholder other)
            {
                //if(type() != other.type())
                //{
                //    return false;
                //}

                TValueType other_held = ((holder<TValueType>)other).held;

                return held.Equals(other_held);
            }

            //public override void print(std::ostream os)
            //{
            //    global::Teuchos.print < TValueType >

            //    {
            //    }

            //    (os, held);
            //}
        }

        public placeholder access_content()
        {
            unsafe
            {
                return Unsafe.AsRef<placeholder>(content.ToPointer());
            }
        }

        public static TValueType any_cast<TValueType>(any operand)
        {
            holder<TValueType> dyn_cast_content = (holder<TValueType>)operand.access_content();

            return dyn_cast_content.held;
        }

        public static TValueType any_ref_cast<TValueType>(any operand)
        {
            return any_cast<TValueType>(operand);
        }
    }
}