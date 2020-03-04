using System;
using System.Runtime.InteropServices;

namespace Kokkos
{
    [StructLayout(System.Runtime.InteropServices.LayoutKind.Explicit,
                  Size = sizeof(int) * 5 + sizeof(bool))]
    public struct InitArguments
    {
        [FieldOffset(0)]
        public int num_threads;

        [FieldOffset(4)]
        public int num_numa;

        [FieldOffset(8)]
        public int device_id;

        [FieldOffset(12)]
        public int ndevices;

        [FieldOffset(16)]
        public int skip_device;

        [FieldOffset(20)]
        public bool disable_warnings;

        public InitArguments(int  nt = -1,
                             int  nn = -1,
                             int  dv = -1,
                             bool dw = false)
        {
            num_threads      = nt;
            num_numa         = nn;
            device_id        = dv;
            ndevices         = -1;
            skip_device      = 9999;
            disable_warnings = dw;
        }

        public InitArguments(int num_threads,
                             int num_numa,
                             int device_id,
                             int ndevices,
                             int skip_device,
                             bool disable_warnings)
        {
            this.num_threads = num_threads;
            this.num_numa = num_numa;
            this.device_id = device_id;
            this.ndevices = ndevices;
            this.skip_device = skip_device;
            this.disable_warnings = disable_warnings;
        }
    }
}