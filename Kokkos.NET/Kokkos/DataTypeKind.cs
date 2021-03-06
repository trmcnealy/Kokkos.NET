﻿using System.Runtime.CompilerServices;

namespace Kokkos
{
    public enum DataTypeKind : ushort
    {
        Unknown = ushort.MaxValue,
        Single  = 0,
        Double  = 1,
        Bool    = 2,
        Int8    = 3,
        UInt8   = 4,
        Int16   = 5,
        UInt16  = 6,
        Int32   = 7,
        UInt32  = 8,
        Int64   = 9,
        UInt64  = 10,
        Char16  = 11

        //ConstSingle = UInt64 + 1,
        //ConstDouble,
        //ConstBool,
        //ConstInt8,
        //ConstUInt8,
        //ConstInt16,
        //ConstUInt16,
        //ConstInt32,
        //ConstUInt32,
        //ConstInt64,
        //ConstUInt64
    }

    internal static class DataType<T>
        where T : struct
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static DataTypeKind GetKind() //bool const_value = false)
        {
            //if(const_value)
            //{
            if(typeof(T) == typeof(float))
            {
                return DataTypeKind.Single;
            }

            if(typeof(T) == typeof(double))
            {
                return DataTypeKind.Double;
            }

            if(typeof(T) == typeof(bool))
            {
                return DataTypeKind.Bool;
            }

            if(typeof(T) == typeof(sbyte))
            {
                return DataTypeKind.Int8;
            }

            if(typeof(T) == typeof(byte))
            {
                return DataTypeKind.UInt8;
            }

            if(typeof(T) == typeof(short))
            {
                return DataTypeKind.Int16;
            }

            if(typeof(T) == typeof(ushort))
            {
                return DataTypeKind.UInt16;
            }

            if(typeof(T) == typeof(int))
            {
                return DataTypeKind.Int32;
            }

            if(typeof(T) == typeof(uint))
            {
                return DataTypeKind.UInt32;
            }

            if(typeof(T) == typeof(long))
            {
                return DataTypeKind.Int64;
            }

            if(typeof(T) == typeof(ulong))
            {
                return DataTypeKind.UInt64;
            }

            if(typeof(T) == typeof(char))
            {
                return DataTypeKind.Char16;
            }
            //}
            //else
            //{
            //    if(typeof(T) == typeof(float))
            //    {
            //        return DataTypeKind.ConstSingle;
            //    }

            //    if(typeof(T) == typeof(double))
            //    {
            //        return DataTypeKind.ConstDouble;
            //    }

            //    if(typeof(T) == typeof(bool))
            //    {
            //        return DataTypeKind.ConstBool;
            //    }

            //    if(typeof(T) == typeof(sbyte))
            //    {
            //        return DataTypeKind.ConstInt8;
            //    }

            //    if(typeof(T) == typeof(byte))
            //    {
            //        return DataTypeKind.ConstUInt8;
            //    }

            //    if(typeof(T) == typeof(short))
            //    {
            //        return DataTypeKind.ConstInt16;
            //    }

            //    if(typeof(T) == typeof(ushort))
            //    {
            //        return DataTypeKind.ConstUInt16;
            //    }

            //    if(typeof(T) == typeof(int))
            //    {
            //        return DataTypeKind.ConstInt32;
            //    }

            //    if(typeof(T) == typeof(uint))
            //    {
            //        return DataTypeKind.ConstUInt32;
            //    }

            //    if(typeof(T) == typeof(long))
            //    {
            //        return DataTypeKind.ConstInt64;
            //    }

            //    if(typeof(T) == typeof(ulong))
            //    {
            //        return DataTypeKind.ConstUInt64;
            //    }
            //}

            return DataTypeKind.Unknown;
        }
    }
}