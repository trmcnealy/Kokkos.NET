using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Kokkos
{
    public struct SharedMemoryData
    {
        public ulong Size;
        public nint  HostAddress;
        public nint  DeviceAddress;
        public nint  Handle;
    }
}
