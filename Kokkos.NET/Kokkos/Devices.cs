using System;
using System.Collections.Generic;
using System.Linq;
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
            : base(id, platform, coreCount, threadCount, architecture)
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
            : base(id, platform, coreCount, threadCount, architecture)
        {
        }
    }

    public sealed class Devices
    {
        public List<CpuDevice> Cpu { get; }

        public List<GpuDevice> Gpu { get; }

        public Devices()
        {
            Cpu = new List<CpuDevice>((int)KokkosLibrary.GetNumaCount());

            OperatingSystem os = Environment.OSVersion;

            for(int i = 0; i < (int)KokkosLibrary.GetNumaCount(); ++i)
            {
                Cpu.Add(new CpuDevice(i,
                                      $"{os.Platform:G}",
                                      (int)KokkosLibrary.GetCoresPerNuma(),
                                      (int)KokkosLibrary.GetCoresPerNuma() * (int)KokkosLibrary.GetThreadsPerCore(),
                                      new DeviceArch(RuntimeInformation.ProcessArchitecture == Architecture.X64 ? "x64" : "x86")));
            }

            Gpu = new List<GpuDevice>((int)KokkosLibrary.GetDeviceCount());

            uint gpuId = 0;

            foreach(PhysicalGPU physicalGpu in PhysicalGPU.GetPhysicalGPUs())
            {
                uint gpuVersion = KokkosLibrary.GetComputeCapability(gpuId);

                Gpu.Add(new GpuDevice((int)gpuId,
                                      "Cuda",
                                      physicalGpu.ArchitectInformation.NumberOfCores,
                                      physicalGpu.ArchitectInformation.NumberOfCores,
                                      new DeviceArch(GetCudaDeviceName(gpuVersion), (int)gpuVersion)));

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