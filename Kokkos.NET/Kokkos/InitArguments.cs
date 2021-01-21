using System;
using System.Runtime.InteropServices;

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
        public bool disable_warnings;

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