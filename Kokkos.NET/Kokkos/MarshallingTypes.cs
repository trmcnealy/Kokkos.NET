// ReSharper disable InconsistentNaming

using System;
using System.Collections.Generic;
using System.Net;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Security;

namespace Kokkos
{
    [UnmanagedFunctionPointer(CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
    public unsafe delegate TResult Functor<out TResult>();
    [UnmanagedFunctionPointer(CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
    public unsafe delegate TResult Functor<out TResult, in T>(T arg);
    [UnmanagedFunctionPointer(CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
    public unsafe delegate TResult Functor<out TResult, in T1, in T2>(T1 arg1, T2 arg2);
    [UnmanagedFunctionPointer(CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
    public unsafe delegate TResult Functor<out TResult, in T1, in T2, in T3>(T1 arg1, T2 arg2, T3 arg3);
    [UnmanagedFunctionPointer(CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
    public unsafe delegate TResult Functor<out TResult, in T1, in T2, in T3, in T4>(T1 arg1, T2 arg2, T3 arg3, T4 arg4);
    [UnmanagedFunctionPointer(CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
    public unsafe delegate TResult Functor<out TResult, in T1, in T2, in T3, in T4, in T5>(T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5);
    [UnmanagedFunctionPointer(CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
    public unsafe delegate TResult Functor<out TResult, in T1, in T2, in T3, in T4, in T5, in T6>(T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6);
    [UnmanagedFunctionPointer(CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
    public unsafe delegate TResult Functor<out TResult, in T1, in T2, in T3, in T4, in T5, in T6, in T7>(T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7);
    [UnmanagedFunctionPointer(CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
    public unsafe delegate TResult Functor<out TResult, in T1, in T2, in T3, in T4, in T5, in T6, in T7, in T8>(T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8);
    [UnmanagedFunctionPointer(CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
    public unsafe delegate TResult Functor<out TResult, in T1, in T2, in T3, in T4, in T5, in T6, in T7, in T8, in T9>(T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9);
    [UnmanagedFunctionPointer(CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
    public unsafe delegate TResult Functor<out TResult, in T1, in T2, in T3, in T4, in T5, in T6, in T7, in T8, in T9, in T10>(T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9, T10 arg10);
    [UnmanagedFunctionPointer(CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
    public unsafe delegate TResult Functor<out TResult, in T1, in T2, in T3, in T4, in T5, in T6, in T7, in T8, in T9, in T10, in T11>(T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9, T10 arg10, T11 arg11);
    [UnmanagedFunctionPointer(CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
    public unsafe delegate TResult Functor<out TResult, in T1, in T2, in T3, in T4, in T5, in T6, in T7, in T8, in T9, in T10, in T11, in T12>(T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9, T10 arg10, T11 arg11, T12 arg12);
    [UnmanagedFunctionPointer(CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
    public unsafe delegate TResult Functor<out TResult, in T1, in T2, in T3, in T4, in T5, in T6, in T7, in T8, in T9, in T10, in T11, in T12, in T13>(T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9, T10 arg10, T11 arg11, T12 arg12, T13 arg13);
    [UnmanagedFunctionPointer(CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
    public unsafe delegate TResult Functor<out TResult, in T1, in T2, in T3, in T4, in T5, in T6, in T7, in T8, in T9, in T10, in T11, in T12, in T13, in T14>(T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9, T10 arg10, T11 arg11, T12 arg12, T13 arg13, T14 arg14);
    [UnmanagedFunctionPointer(CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
    public unsafe delegate TResult Functor<out TResult, in T1, in T2, in T3, in T4, in T5, in T6, in T7, in T8, in T9, in T10, in T11, in T12, in T13, in T14, in T15>(T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9, T10 arg10, T11 arg11, T12 arg12, T13 arg13, T14 arg14, T15 arg15);
    [UnmanagedFunctionPointer(CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
    public unsafe delegate TResult Functor<out TResult, in T1, in T2, in T3, in T4, in T5, in T6, in T7, in T8, in T9, in T10, in T11, in T12, in T13, in T14, in T15, in T16>(T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9, T10 arg10, T11 arg11, T12 arg12, T13 arg13, T14 arg14, T15 arg15, T16 arg16);
}

namespace Kokkos
{
    public static class Mangling
    {
        private static readonly Dictionary<Type, char> _typeCodes = new Dictionary<Type, char>
        {
            {
                typeof(bool), 'b'
            },
            {
                typeof(sbyte), 'c'
            },
            {
                typeof(double), 'd'
            },
            {
                typeof(decimal), 'e'
            },
            {
                typeof(float), 'f'
            },
            {
                typeof(byte), 'h'
            },
            {
                typeof(int), 'l'
            },
            {
                typeof(uint), 'm'
            },
            {
                typeof(short), 's'
            },
            {
                typeof(ushort), 't'
            },
            {
                typeof(long), 'x'
            },
            {
                typeof(ulong), 'y'
            },
            {
                typeof(nint), 'P'
            },

            //{typeof(Complex<float>), 'C'},
            //{typeof(Complex<double>), 'C'},
            {
                typeof(char), 'w'
            },
        };

        public static char GetType(Type type)
        {
            return _typeCodes[type];
        }
    }

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
        public char Char16;

        [FieldOffset(0)]
        public nint ByRef;

        public TDataType As<TDataType>()
            where TDataType : struct
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

            if(typeof(TDataType) == typeof(char))
            {
                return (TDataType)(System.ValueType)Char16;
            }

            throw new NotSupportedException();
        }

        public static ValueType From<TDataType>(TDataType value)
            where TDataType : struct
        {
            if(typeof(TDataType) == typeof(float))
            {
                return new ValueType
                {
                    Single = (float)(System.ValueType)value
                };
            }

            if(typeof(TDataType) == typeof(double))
            {
                return new ValueType
                {
                    Double = (double)(System.ValueType)value
                };
            }

            if(typeof(TDataType) == typeof(bool))
            {
                return new ValueType
                {
                    Bool = (bool)(System.ValueType)value
                };
            }

            if(typeof(TDataType) == typeof(sbyte))
            {
                return new ValueType
                {
                    Int8 = (sbyte)(System.ValueType)value
                };
            }

            if(typeof(TDataType) == typeof(byte))
            {
                return new ValueType
                {
                    UInt8 = (byte)(System.ValueType)value
                };
            }

            if(typeof(TDataType) == typeof(short))
            {
                return new ValueType
                {
                    Int16 = (short)(System.ValueType)value
                };
            }

            if(typeof(TDataType) == typeof(ushort))
            {
                return new ValueType
                {
                    UInt16 = (ushort)(System.ValueType)value
                };
            }

            if(typeof(TDataType) == typeof(int))
            {
                return new ValueType
                {
                    Int32 = (int)(System.ValueType)value
                };
            }

            if(typeof(TDataType) == typeof(uint))
            {
                return new ValueType
                {
                    UInt32 = (uint)(System.ValueType)value
                };
            }

            if(typeof(TDataType) == typeof(long))
            {
                return new ValueType
                {
                    Int64 = (long)(System.ValueType)value
                };
            }

            if(typeof(TDataType) == typeof(ulong))
            {
                return new ValueType
                {
                    UInt64 = (ulong)(System.ValueType)value
                };
            }

            if(typeof(TDataType) == typeof(char))
            {
                return new ValueType
                {
                    Char16 = (char)(System.ValueType)value
                };
            }

            throw new NotSupportedException();
        }

        public static implicit operator float(ValueType from)
        {
            return from.Single;
        }

        public static implicit operator double(ValueType from)
        {
            return from.Double;
        }

        public static implicit operator bool(ValueType from)
        {
            return from.Bool;
        }

        public static implicit operator sbyte(ValueType from)
        {
            return from.Int8;
        }

        public static implicit operator byte(ValueType from)
        {
            return from.UInt8;
        }

        public static implicit operator short(ValueType from)
        {
            return from.Int16;
        }

        public static implicit operator ushort(ValueType from)
        {
            return from.UInt16;
        }

        public static implicit operator int(ValueType from)
        {
            return from.Int32;
        }

        public static implicit operator uint(ValueType from)
        {
            return from.UInt32;
        }

        public static implicit operator long(ValueType from)
        {
            return from.Int64;
        }

        public static implicit operator ulong(ValueType from)
        {
            return from.UInt64;
        }

        public static implicit operator char(ValueType from)
        {
            return from.Char16;
        }

        public static explicit operator ValueType(float to)
        {
            return new ValueType
            {
                Single = to
            };
        }

        public static explicit operator ValueType(double to)
        {
            return new ValueType
            {
                Double = to
            };
        }

        public static explicit operator ValueType(bool to)
        {
            return new ValueType
            {
                Bool = to
            };
        }

        public static explicit operator ValueType(sbyte to)
        {
            return new ValueType
            {
                Int8 = to
            };
        }

        public static explicit operator ValueType(byte to)
        {
            return new ValueType
            {
                UInt8 = to
            };
        }

        public static explicit operator ValueType(short to)
        {
            return new ValueType
            {
                Int16 = to
            };
        }

        public static explicit operator ValueType(ushort to)
        {
            return new ValueType
            {
                UInt16 = to
            };
        }

        public static explicit operator ValueType(int to)
        {
            return new ValueType
            {
                Int32 = to
            };
        }

        public static explicit operator ValueType(uint to)
        {
            return new ValueType
            {
                UInt32 = to
            };
        }

        public static explicit operator ValueType(long to)
        {
            return new ValueType
            {
                Int64 = to
            };
        }

        public static explicit operator ValueType(ulong to)
        {
            return new ValueType
            {
                UInt64 = to
            };
        }

        public static explicit operator ValueType(char to)
        {
            return new ValueType
            {
                Char16 = to
            };
        }
    }

    //[StructLayout(System.Runtime.InteropServices.LayoutKind.Sequential)]
    //public struct type_info : IEquatable<type_info>
    //{
    //    [MarshalAs(UnmanagedType.LPStr)]
    //    public readonly string? name;

    //    public type_info(string value)
    //    {
    //        name = value;
    //    }

    //    public bool Equals(type_info other)
    //    {
    //        return name == other.name;
    //    }

    //    public override bool Equals(object? obj)
    //    {
    //        return obj is type_info other && Equals(other);
    //    }

    //    public override int GetHashCode()
    //    {
    //        return name != null ? name.GetHashCode() : 0;
    //    }

    //    public static bool operator ==(type_info left,
    //                                   type_info right)
    //    {
    //        return left.Equals(right);
    //    }

    //    public static bool operator !=(type_info left,
    //                                   type_info right)
    //    {
    //        return !left.Equals(right);
    //    }
    //}

    //[StructLayout(System.Runtime.InteropServices.LayoutKind.Sequential)]
    //public struct any
    //{
    //    private nint /*placeholder*/
    //        content;

    //    //[System.Runtime.CompilerServices.SpecialName]
    //    public static any Create<TValueType>(TValueType value)
    //    {
    //        unsafe
    //        {
    //            holder<TValueType> new_holder = new holder<TValueType>(value);

    //            return new any
    //            {
    //                content = (nint)Unsafe.AsPointer(ref new_holder)
    //            };
    //        }
    //    }

    //    public any(any other)
    //    {
    //        content = other.content;
    //    }

    //    //public any swap(any rhs)
    //    //{
    //    //    std::swap(content,
    //    //              rhs.content);

    //    //    return this;
    //    //}

    //    //public any CopyFrom<TValueType>(TValueType rhs)
    //    //{
    //    //    any(rhs).swap(this);

    //    //    return this;
    //    //}

    //    //public any CopyFrom(any rhs)
    //    //{
    //    //    any(rhs).swap(this);

    //    //    return this;
    //    //}

    //    public bool empty()
    //    {
    //        return content == 0;
    //    }

    //    //public type_info type()
    //    //{
    //    //    return content != null ? content.type() : typeid();
    //    //}

    //    //public string typeName()
    //    //{
    //    //    return content != null ? content.typeName() : "NONE";
    //    //}

    //    public bool same(any other)
    //    {
    //        if(empty() && other.empty())
    //        {
    //            return true;
    //        }

    //        if(empty() && !other.empty())
    //        {
    //            return false;
    //        }

    //        if(!empty() && other.empty())
    //        {
    //            return false;
    //        }

    //        return content == other.content;
    //    }

    //    //public void print(std::ostream os)
    //    //{
    //    //    if(content != null)
    //    //    {
    //    //        content.print(os);
    //    //    }
    //    //}

    //    public interface placeholder
    //    {
    //        //public abstract type_info type();

    //        //public abstract string typeName();

    //        //public abstract placeholder clone();

    //        //public abstract bool same(placeholder other);

    //        //public abstract void print(std::ostream os);
    //    }

    //    [StructLayout(System.Runtime.InteropServices.LayoutKind.Sequential)]
    //    public struct holder<TValueType> : placeholder
    //    {
    //        public TValueType held;

    //        public holder(TValueType value)
    //        {
    //            held = value;
    //        }

    //        //public override type_info type()
    //        //{
    //        //    return typeid(TValueType);
    //        //}

    //        //public override string typeName()
    //        //{
    //        //    return TypeNameTraits<TValueType>.name();
    //        //}

    //        public placeholder clone()
    //        {
    //            return new holder<TValueType>(held);
    //        }

    //        public bool same(placeholder other)
    //        {
    //            //if(type() != other.type())
    //            //{
    //            //    return false;
    //            //}

    //            TValueType other_held = ((holder<TValueType>)other).held;

    //            return held.Equals(other_held);
    //        }

    //        //public override void print(std::ostream os)
    //        //{
    //        //    global::Teuchos.print < TValueType >

    //        //    {
    //        //    }

    //        //    (os, held);
    //        //}
    //    }

    //    public placeholder access_content()
    //    {
    //        unsafe
    //        {
    //            return Unsafe.AsRef<placeholder>(content.ToPointer());
    //        }
    //    }

    //    public static TValueType any_cast<TValueType>(any operand)
    //    {
    //        holder<TValueType> dyn_cast_content = (holder<TValueType>)operand.access_content();

    //        return dyn_cast_content.held;
    //    }

    //    public static TValueType any_ref_cast<TValueType>(any operand)
    //    {
    //        return any_cast<TValueType>(operand);
    //    }
    //}
}