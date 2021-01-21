using System;
using System.Runtime.InteropServices;

namespace Kokkos
{
    [StructLayout(System.Runtime.InteropServices.LayoutKind.Sequential)]
    public unsafe struct KokkosApi
    {
        public nint AllocatePtr;

        public nint ReallocatePtr;

        public nint FreePtr;

        public nint InitializePtr;

        public nint InitializeThreadsPtr;

        public nint InitializeArgumentsPtr;

        public nint FinalizePtr;

        public nint FinalizeAllPtr;

        public nint IsInitializedPtr;

        public nint PrintConfigurationPtr;

        public nint GetDeviceCountPtr;

        public nint GetComputeCapabilityPtr;

        public nint CreateViewRank0Ptr;

        public nint CreateViewRank1Ptr;

        public nint CreateViewRank2Ptr;

        public nint CreateViewRank3Ptr;

        public nint CreateViewRank4Ptr;

        public nint CreateViewRank5Ptr;

        public nint CreateViewRank6Ptr;

        public nint CreateViewRank7Ptr;

        public nint CreateViewRank8Ptr;

        public nint CreateViewPtr;

        public nint GetLabelPtr;

        public nint GetSizePtr;

        public nint GetStridePtr;

        public nint GetExtentPtr;

        public nint CopyToPtr;

        public nint GetValuePtr;

        public nint SetValuePtr;

        public /*delegate* unmanaged[Cdecl]<nint, ExecutionSpaceKind, LayoutKind, DataTypeKind, ushort, out NdArray, void>*/ nint RcpViewToNdArrayPtr;

        public nint ViewToNdArrayPtr;
    }
}