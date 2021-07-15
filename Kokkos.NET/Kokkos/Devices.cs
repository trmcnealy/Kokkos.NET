using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;

using Kokkos;

using NvAPIWrapper.GPU;
using NvAPIWrapper.Native.GPU.Structures;

using PlatformApi.Win32;

using static Kokkos.Devices;

namespace Kokkos
{
    public struct DeviceArch
    {
        public string Name;
        public int    Version;

        public DeviceArch(string name)
        {
            Name    = name;
            Version = 0;
        }

        public DeviceArch(int version)
        {
            Name    = string.Empty;
            Version = version;
        }

        public DeviceArch(string name,
                          int    version)
        {
            Name    = name;
            Version = version;
        }
    }

    public abstract class Device
    {
        public int Id { get; protected set; }

        public string Platform { get; protected set; }

        public int CoreCount { get; protected set; }

        public int ThreadCount { get; protected set; }

        public DeviceArch Architecture { get; protected set; }

        protected Device(int        id,
                         string     platform,
                         int        coreCount,
                         int        threadCount,
                         DeviceArch architecture)
        {
            Id           = id;
            Platform     = platform;
            CoreCount    = coreCount;
            ThreadCount  = threadCount;
            Architecture = architecture;
        }

        public abstract int GetUsage();
    }

    public sealed class CpuDevice : Device
    {

        static CpuDevice()
        {
        }

        public CpuDevice(int        id,
                         string     platform,
                         int        coreCount,
                         int        threadCount,
                         DeviceArch architecture)
            : base(id, platform, coreCount, threadCount, architecture)
        {
        }

        public override int GetUsage()
        {
            return 0;
        }
    }

    public class GpuDevice : Device
    {
        public GpuDevice(int        id,
                         string     platform,
                         int        coreCount,
                         int        threadCount,
                         DeviceArch architecture)
            : base(id, platform, coreCount, threadCount, architecture)
        {
        }

        public override int GetUsage()
        {
            return -1;
        }
    }

    public sealed class CudaGpuDevice : GpuDevice
    {
        public GPUUsageInformation UsageInformation { get; }

        public CudaGpuDevice(int                 id,
                             string              platform,
                             int                 coreCount,
                             int                 threadCount,
                             DeviceArch          architecture,
                             GPUUsageInformation usageInformation)
            : base(id, platform, coreCount, threadCount, architecture)
        {
            UsageInformation = usageInformation;
        }

        public override int GetUsage()
        {
            return UsageInformation.UtilizationDomainsStatus.Sum(gpu => gpu.Percentage);
        }
    }

    public sealed class Devices
    {

        [StructLayout(System.Runtime.InteropServices.LayoutKind.Sequential)]
        internal unsafe struct SYSTEM_INFO
        {
            [StructLayout(System.Runtime.InteropServices.LayoutKind.Explicit)]
            public struct SYSTEM_INFOunion
            {
                [StructLayout(System.Runtime.InteropServices.LayoutKind.Sequential)]
                public struct SYSTEM_INFOunionstruct
                {
                    public ushort wProcessorArchitecture;
                    
                    public ushort wReserved;
                }
                
                [FieldOffset(0)]
                public uint dwOemId;

                [FieldOffset(0)]
                public SYSTEM_INFOunionstruct DUMMYSTRUCTNAME;
            }

            public SYSTEM_INFOunion DUMMYUNIONNAME;

            public uint dwPageSize;
            
            public void* lpMinimumApplicationAddress;
            
            public void* lpMaximumApplicationAddress;
            
            public uint* dwActiveProcessorMask;
            
            public uint dwNumberOfProcessors;
            
            public uint dwProcessorType;
            
            public uint dwAllocationGranularity;
            
            public ushort wProcessorLevel;
            
            public ushort wProcessorRevision;
        }

        [DllImport("kernel32.dll", ExactSpelling = true)]
        internal static extern void GetSystemInfo(out SYSTEM_INFO lpSystemInfo);


        public List<CpuDevice> Cpus { get; }

        public List<GpuDevice> Gpus { get; }

        public Devices()
        {
            Cpus = new List<CpuDevice>((int)KokkosLibrary.GetNumaCount());

            OperatingSystem os = Environment.OSVersion;

            if(Cpus.Capacity == 1)
            {
                GetSystemInfo(out SYSTEM_INFO lpSystemInfo);

                Cpus.Add(new CpuDevice(0,
                                       $"{os.Platform:G}",
                                       (int)lpSystemInfo.dwNumberOfProcessors,
                                       (int)lpSystemInfo.dwNumberOfProcessors,
                                       new DeviceArch(RuntimeInformation.ProcessArchitecture == Architecture.X64 ? "x64" : "x86")));
            }
            else
            {
                for(int i = 0; i < (int)KokkosLibrary.GetNumaCount(); ++i)
                {
                    Cpus.Add(new CpuDevice(i,
                                           $"{os.Platform:G}",
                                           (int)KokkosLibrary.GetCoresPerNuma(),
                                           (int)KokkosLibrary.GetCoresPerNuma() * (int)KokkosLibrary.GetThreadsPerCore(),
                                           new DeviceArch(RuntimeInformation.ProcessArchitecture == Architecture.X64 ? "x64" : "x86")));
                }
            }

            Gpus = new List<GpuDevice>((int)KokkosLibrary.GetDeviceCount());

            uint gpuId = 0;

            foreach(PhysicalGPU physicalGpu in PhysicalGPU.GetPhysicalGPUs())
            {
                uint gpuVersion = KokkosLibrary.GetComputeCapability(gpuId);

                Gpus.Add(new CudaGpuDevice((int)gpuId,
                                          "Cuda",
                                          physicalGpu.ArchitectInformation.NumberOfCores,
                                          physicalGpu.ArchitectInformation.NumberOfCores,
                                          new DeviceArch(GetCudaDeviceName(gpuVersion), (int)gpuVersion),
                                          physicalGpu.UsageInformation));

                ++gpuId;
            }
        }

        private static string GetCudaDeviceName(uint version)
        {
            switch(version)
            {
                case 200:
                {
                    return "Kepler";
                }
                case 210:
                {
                    return "Kepler";
                }
                case 300:
                {
                    return "Kepler";
                }
                case 320:
                {
                    return "Kepler";
                }
                case 350:
                {
                    return "Kepler";
                }
                case 370:
                {
                    return "Kepler";
                }
                case 500:
                {
                    return "Maxwell";
                }
                case 520:
                {
                    return "Maxwell";
                }
                case 530:
                {
                    return "Maxwell";
                }
                case 600:
                {
                    return "Pascal";
                }
                case 610:
                {
                    return "Pascal";
                }
                case 620:
                {
                    return "Pascal";
                }
                case 700:
                {
                    return "Volta";
                }
                case 720:
                {
                    return "Volta";
                }
                case 1000:
                {
                    return "Turing";
                }
            }

            return "Unknown";
        }
    }
}
