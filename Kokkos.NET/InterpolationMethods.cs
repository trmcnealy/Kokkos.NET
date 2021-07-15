using System;
using System.Runtime.CompilerServices;

namespace Kokkos
{
    public static class InterpolationMethods<T, TExecutionSpace>
        where T : unmanaged
        where TExecutionSpace : IExecutionSpace, new()
    {
        private static readonly ExecutionSpaceKind executionSpace;

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        static InterpolationMethods()
        {
            executionSpace = ExecutionSpace<TExecutionSpace>.GetKind();
        }

        /// <summary>
        /// Shepard 2d interpolant.
        /// </summary>
        /// <param name="xd">the data point[M*ND], M: the spatial dimension, ND: the number of data points</param>
        /// <param name="zd">data values[ND]</param>
        /// <param name="p">power</param>
        /// <param name="xi">interpolation points[M*NI], NI: the number of interpolation points</param>
        /// <returns>interpolated values[NI]</returns>
        public static View<float, TExecutionSpace> Shepard2d(View<float, TExecutionSpace> xd,
                                                             View<float, TExecutionSpace> zd,
                                                             float                        p,
                                                             View<float, TExecutionSpace> xi)
        {
            nint result = KokkosLibrary.Shepard2dSingle(xd.Pointer, zd.Pointer, p, xi.Pointer, executionSpace);

            NdArray ndArray = View<float, TExecutionSpace>.RcpConvert(result, 1);

            View<float, TExecutionSpace> zi = new View<float, TExecutionSpace>(new NativePointer(result, sizeof(float) * xi.Size()), ndArray);

            return zi;
        }

        public static View<double, TExecutionSpace> Shepard2d(View<double, TExecutionSpace> xd,
                                                              View<double, TExecutionSpace> zd,
                                                              double                        p,
                                                              View<double, TExecutionSpace> xi)
        {
            nint result = KokkosLibrary.Shepard2dDouble(xd.Pointer, zd.Pointer, p, xi.Pointer, executionSpace);

            NdArray ndArray = View<double, TExecutionSpace>.RcpConvert(result, 1);

            View<double, TExecutionSpace> zi = new View<double, TExecutionSpace>(new NativePointer(result, sizeof(double) * xi.Size()), ndArray);

            return zi;
        }
    }
}
