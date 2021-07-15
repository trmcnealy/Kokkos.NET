using System;
using System.Runtime.InteropServices;

using std;

namespace std
{
    [StructLayout(LayoutKind.Explicit, Size = 16, Pack = 8)]
    public unsafe struct __Internal
    {
        [FieldOffset(0)]
        internal fixed sbyte _Buf[16];

        [FieldOffset(0)]
        internal IntPtr _Ptr;

        [FieldOffset(0)]
        internal fixed sbyte _Alias[16];
    }

    [StructLayout(LayoutKind.Explicit, Size = 32, Pack = 8)]
    public struct __Internalc__N_std_S__String_val____N_std_S__Simple_types__C
    {
        [FieldOffset(0)]
        internal __Internal _Bx;

        [FieldOffset(16)]
        internal ulong _Mysize;

        [FieldOffset(24)]
        internal ulong _Myres;
    }

    [StructLayout(LayoutKind.Explicit, Size = 32, Pack = 8)]
    public struct Compressed_pair
    {
        [FieldOffset(0)]
        internal __Internalc__N_std_S__String_val____N_std_S__Simple_types__C _Myval2;
    }

    [StructLayout(LayoutKind.Explicit, Size = 32, Pack = 8)]
    public struct basic_string
    {
        [FieldOffset(0)]
        internal Compressed_pair _Mypair;
    }
}

namespace Kokkos
{
    [StructLayout(System.Runtime.InteropServices.LayoutKind.Sequential)]
    public struct InitArguments
    {
        public int  num_threads;
        public int  num_numa;
        public int  device_id;
        public int  ndevices;
        public int  skip_device;
        [MarshalAs(UnmanagedType.Bool)]
        public bool disable_warnings;
        [MarshalAs(UnmanagedType.Bool)]
        public bool tune_internals;
        [MarshalAs(UnmanagedType.Bool)]
        public bool tool_help;
        std.basic_string tool_lib;
        std.basic_string tool_args;

        //public InitArguments()
        //{
        //    num_threads      = -1;
        //    num_numa         = -1;
        //    device_id        = -1;
        //    ndevices         = -1;
        //    skip_device      = 9999;
        //    disable_warnings = false;
        //}

        public InitArguments(int  nt = -1,
                             int  nn = -1,
                             int  dv = -1,
                             bool dw = false,
                             bool ti = false)
        {
            num_threads      = nt;
            num_numa         = nn;
            device_id        = dv;
            ndevices         = -1;
            skip_device      = 9999;
            disable_warnings = dw;
            tool_help        = false;
            tune_internals   = ti;
            tool_lib         = new basic_string();
            tool_args        = new basic_string();
        }

        public InitArguments(int  num_threads,
                             int  num_numa,
                             int  device_id,
                             int  ndevices,
                             int  skip_device,
                             bool disable_warnings,
                             bool ti = false)
        {
            this.num_threads      = num_threads;
            this.num_numa         = num_numa;
            this.device_id        = device_id;
            this.ndevices         = ndevices;
            this.skip_device      = skip_device;
            this.disable_warnings = disable_warnings;
            tool_help             = false;
            tune_internals        = ti;
            tool_lib              = new basic_string();
            tool_args             = new basic_string();
        }
    }
}
