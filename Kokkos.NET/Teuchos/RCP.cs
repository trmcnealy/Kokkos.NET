// ReSharper disable InconsistentNaming
// ReSharper disable CheckNamespace

using System;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Versioning;

namespace Teuchos
{
    public enum ERCPStrength
    {
        RCP_STRONG = 0,
        RCP_WEAK   = 1
    }

    [NonVersionable]
    [StructLayout(LayoutKind.Sequential)]
    public struct RCPNodeHandle
    {
        public IntPtr       node;
        public ERCPStrength strength;
    }

    [NonVersionable]
    [StructLayout(LayoutKind.Sequential)]
    public struct RCP/*<T>*/
    {
        public /* T* */ IntPtr ptr;

        public RCPNodeHandle node;

        //public ref T Get()
        //{
        //    return Unsafe.AsRef<T>();
        //}
    }
}