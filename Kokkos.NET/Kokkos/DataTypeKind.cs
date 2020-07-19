using System.Runtime.CompilerServices;

namespace Kokkos
{
    public enum DataTypeKind : ushort
    {
        Unknown = ushort.MaxValue,
        Single  = 0,
        Double,
        Bool,
        Int8,
        UInt8,
        Int16,
        UInt16,
        Int32,
        UInt32,
        Int64,
        UInt64

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
#if NETSTANDARD
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
#else
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
#endif
        public static DataTypeKind GetKind()//bool const_value = false)
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