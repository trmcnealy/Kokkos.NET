using System.Diagnostics.Contracts;
using System.Runtime.CompilerServices;

namespace Kokkos.Utilities
{
    public static class BoolExtensions
    {
        [Pure]
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static int ToInt(this bool flag)
        {
            return Unsafe.As<bool, byte>(ref flag);
        }

        [Pure]
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static int ToBitwiseMask32(this bool flag)
        {
            byte rangeFlag = Unsafe.As<bool, byte>(ref flag);

            int negativeFlag = rangeFlag - 1;

            int mask = ~negativeFlag;

            return mask;
        }

        [Pure]
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static long ToBitwiseMask64(this bool flag)
        {
            byte rangeFlag = Unsafe.As<bool, byte>(ref flag);

            long negativeFlag = (long)rangeFlag - 1;

            long mask = ~negativeFlag;

            return mask;
        }
    }
}