using System;
using System.Runtime.InteropServices;

namespace Kokkos
{
    [StructLayout(System.Runtime.InteropServices.LayoutKind.Sequential)]
    internal struct KokkosApi
    {
        public IntPtr AllocatePtr;

        public IntPtr ReallocatePtr;

        public IntPtr FreePtr;

        public IntPtr InitializePtr;

        public IntPtr InitializeThreadsPtr;

        public IntPtr InitializeArgumentsPtr;

        public IntPtr FinalizePtr;

        public IntPtr FinalizeAllPtr;

        public IntPtr IsInitializedPtr;

        public IntPtr PrintConfigurationPtr;

        public IntPtr GetDeviceCountPtr;

        public IntPtr GetComputeCapabilityPtr;

        public IntPtr CreateViewRank0Ptr;

        public IntPtr CreateViewRank1Ptr;

        public IntPtr CreateViewRank2Ptr;

        public IntPtr CreateViewRank3Ptr;

        public IntPtr CreateViewPtr;

        public IntPtr GetLabelPtr;

        public IntPtr GetSizePtr;

        public IntPtr GetStridePtr;

        public IntPtr GetExtentPtr;

        public IntPtr CopyToPtr;

        public IntPtr GetValuePtr;

        public IntPtr SetValuePtr;

        public IntPtr ViewToNdArrayPtr;
    }
}