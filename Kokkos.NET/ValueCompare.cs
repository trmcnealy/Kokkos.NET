using System;
using System.Runtime.CompilerServices;


public static class Compare
{
    [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
    public static bool AreEqual(sbyte lhs,
                                sbyte rhs)
    {
        return lhs == rhs;
    }
    
    [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
    public static bool AreEqual(byte lhs, 
                                byte rhs)
    {
        return lhs == rhs;
    }
    
    [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
    public static bool AreEqual(short lhs,
                                short rhs)
    {
        return lhs == rhs;
    }
    
    [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
    public static bool AreEqual(ushort lhs, 
                                ushort rhs)
    {
        return lhs == rhs;
    }
    
    [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
    public static bool AreEqual(int lhs,
                                int rhs)
    {
        return lhs == rhs;
    }
    
    [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
    public static bool AreEqual(uint lhs, 
                                uint rhs)
    {
        return lhs == rhs;
    }
    
    [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
    public static bool AreEqual(long lhs,
                                long rhs)
    {
        return lhs == rhs;
    }
    
    [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
    public static bool AreEqual(ulong lhs, 
                                ulong rhs)
    {
        return lhs == rhs;
    }
    
    [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
    public static bool AreEqual(Half lhs, 
                                Half rhs)
    {
        return lhs == rhs;
    }
    
    [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
    public static bool AreEqual(float lhs, 
                                float rhs)
    {
        return Math.Abs(lhs - rhs) <= float.Epsilon;
    }
    
    [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
    public static bool AreEqual(double lhs, 
                                double rhs)
    {
        return Math.Abs(lhs - rhs) <= double.Epsilon;
    }
    
    
    
    
    
    //Math.Abs(view[0] - values[0]) <= double.Epsilon
}