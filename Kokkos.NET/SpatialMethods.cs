using System;
using System.Runtime.CompilerServices;

namespace Kokkos
{
    public static class SpatialMethods<T, TExecutionSpace>
        where T : unmanaged
        where TExecutionSpace : IExecutionSpace, new()
    {
        private static readonly ExecutionSpaceKind executionSpace;

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        static SpatialMethods()
        {
            executionSpace = ExecutionSpace<TExecutionSpace>.GetKind();
        }
        
        public static View<float, TExecutionSpace> NearestNeighbor(View<float, TExecutionSpace> latlongdegrees)
        {
            nint result = KokkosLibrary.NearestNeighborSingle(latlongdegrees.Pointer, executionSpace);

            NdArray ndArray = View<float, TExecutionSpace>.RcpConvert(result, 1);

            View<float, TExecutionSpace> neighbors = new View<float, TExecutionSpace>(new NativePointer(result, sizeof(float) * latlongdegrees.Extent(0)), ndArray);

            return neighbors;
        }

        public static View<double, TExecutionSpace> NearestNeighbor(View<double, TExecutionSpace> latlongdegrees)
        {
            nint result = KokkosLibrary.NearestNeighborDouble(latlongdegrees.Pointer, executionSpace);

            NdArray ndArray = View<double, TExecutionSpace>.RcpConvert(result, 1);

            View<double, TExecutionSpace> neighbors = new View<double, TExecutionSpace>(new NativePointer(result, sizeof(double) * latlongdegrees.Extent(0)), ndArray);

            return neighbors;
        }
    }
}
