using System;
using System.Collections.Generic;
using System.Linq;
using System.Management;
using System.Runtime.InteropServices;

using Kokkos;

using NvAPIWrapper.GPU;

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
    }

    public sealed class CpuDevice : Device
    {
        public CpuDevice(int        id,
                         string     platform,
                         int        coreCount,
                         int        threadCount,
                         DeviceArch architecture)
            : base(id,
                   platform,
                   coreCount,
                   threadCount,
                   architecture)
        {
        }
    }

    public sealed class GpuDevice : Device
    {
        public GpuDevice(int        id,
                         string     platform,
                         int        coreCount,
                         int        threadCount,
                         DeviceArch architecture)
            : base(id,
                   platform,
                   coreCount,
                   threadCount,
                   architecture)
        {
        }
    }

    public sealed class Devices
    {
        public List<CpuDevice> Cpu { get; }

        public List<GpuDevice> Gpu { get; }

        public Devices()
        {
            int       numberOfProcessors        = 0;
            List<int> numberOfCores             = new List<int>(numberOfProcessors);
            List<int> numberOfHyperThreads      = new List<int>(numberOfProcessors);
            int       numberOfLogicalProcessors = 0;

            foreach(ManagementBaseObject item in new ManagementObjectSearcher("Select NumberOfProcessors from Win32_ComputerSystem").Get())
            {
                numberOfProcessors = int.Parse(item["NumberOfProcessors"].ToString());
            }

            Cpu = new List<CpuDevice>(numberOfProcessors);

            foreach(ManagementBaseObject item in new ManagementObjectSearcher("Select NumberOfLogicalProcessors from Win32_ComputerSystem").Get())
            {
                numberOfLogicalProcessors = int.Parse(item["NumberOfLogicalProcessors"].ToString());
            }

            foreach(ManagementBaseObject item in new ManagementObjectSearcher("Select NumberOfCores from Win32_Processor").Get())
            {
                numberOfCores.Add(int.Parse(item["NumberOfCores"].ToString()));
                numberOfHyperThreads.Add(numberOfLogicalProcessors / numberOfCores.Last());
            }

            uint cpuId = 0;

            OperatingSystem os = Environment.OSVersion;

            foreach(int numberOfCore in numberOfCores)
            {
                Cpu.Add(new CpuDevice((int)cpuId,
                                      $"{os.Platform:G}",
                                      numberOfCores[(int)cpuId],
                                      numberOfCores[(int)cpuId] * numberOfHyperThreads[(int)cpuId],
                                      new DeviceArch(RuntimeInformation.ProcessArchitecture == Architecture.X64 ? "x64" : "x86")));

                ++cpuId;
            }

            Gpu = new List<GpuDevice>((int)KokkosLibrary.GetDeviceCount());

            uint gpuId = 0;

            foreach(PhysicalGPU physicalGpu in PhysicalGPU.GetPhysicalGPUs())
            {
                Gpu.Add(new GpuDevice((int)gpuId,
                                      "Cuda",
                                      physicalGpu.ArchitectInformation.NumberOfCores,
                                      physicalGpu.ArchitectInformation.NumberOfCores,
                                      new DeviceArch(KokkosLibrary.GetComputeCapability(gpuId).ToString())));

                ++gpuId;
            }
        }
    }
}