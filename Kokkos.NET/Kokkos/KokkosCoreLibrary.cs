// ReSharper disable InconsistentNaming
// ReSharper disable UnusedMember.Local

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Versioning;
using System.Threading;

using ThreadState = System.Diagnostics.ThreadState;

namespace Kokkos
{
    [NonVersionable]
    internal static class KokkosCoreLibrary
    {
        public const string KokkosCoreLibraryName = "libkokkoscore";

        public const string KokkosContainersLibraryName = "libkokkoscontainers";

        public const string OpenMpLibraryName = "libomp";

        public const string MpiLibraryName = "mpi";

        public const string CudartLibraryName = "cudart64_102";

        public static IntPtr KokkosCoreHandle;

        public static IntPtr KokkosCoreModuleHandle;

        public static IntPtr KokkosContainersHandle;

        public static IntPtr KokkosContainersModuleHandle;

        public static readonly IntPtr OpenMpHandle;

        public static readonly IntPtr MpiHandle;

        public static readonly IntPtr CudartHandle;

        public static volatile bool Initialized;

        private static readonly string nativeLibraryPath;

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        private static string GetNativePackagePath(string nativePackagePath)
        {
            Version lastestVersion = new Version(0, 0, 0, 0);

            Version currentVersion;

            foreach(DirectoryInfo di in new DirectoryInfo(nativePackagePath).GetDirectories())
            {
                currentVersion = new Version(di.Name);

                if(lastestVersion < currentVersion)
                {
                    lastestVersion = currentVersion;
                }
            }

            return Path.Combine(nativePackagePath, lastestVersion.ToString());
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        private static string GetLibraryPath()
        {
            string fullPath = System.Reflection.Assembly.GetExecutingAssembly().Location;

            if(!string.IsNullOrEmpty(fullPath) && !fullPath.Contains(".nuget"))
            {
                int lastIndex = fullPath.LastIndexOf("\\", StringComparison.Ordinal);

                return fullPath.Substring(0, lastIndex);
            }

            string nugetPackagesEnvironmentVariable = Environment.GetEnvironmentVariable("NUGET_PACKAGES");

            if(!string.IsNullOrEmpty(nugetPackagesEnvironmentVariable))
            {
                string nativePackagePath = Path.Combine(nugetPackagesEnvironmentVariable, "native.kokkos.net");

                return GetNativePackagePath(nativePackagePath);
            }

            //const string dotnetProfileDirectoryName = ".dotnet";
            //const string toolsShimFolderName        = "tools";

            string userProfile = Environment.GetEnvironmentVariable(RuntimeInformation.IsOSPlatform(OSPlatform.Windows) ? "USERPROFILE" : "HOME");

            if(!string.IsNullOrEmpty(userProfile))
            {
                string nativePackagePath = Path.Combine(userProfile, ".nuget", "packages", "native.kokkos.net");

                return GetNativePackagePath(nativePackagePath);
            }

            return null;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        static KokkosCoreLibrary()
        {
            if(Thread.CurrentThread.TrySetApartmentState(ApartmentState.STA))
            {
                Console.WriteLine("TrySetApartmentState Failed.");
            }

            string operatingSystem      = RuntimeInformation.IsOSPlatform(OSPlatform.Windows) ? "win" : "linux";
            string platformArchitecture = RuntimeInformation.ProcessArchitecture == Architecture.X64 ? "x64" : "x86";

            //string libraryPath = GetLibraryPath() ?? throw new NullReferenceException("typeof(KokkosCoreLibrary).Assembly.Location is empty.");

// #if DEBUG
//             Console.WriteLine("libraryPath: " + libraryPath);
// #endif

            //Path.Combine(libraryPath,
            nativeLibraryPath = $"runtimes\\{operatingSystem}-{platformArchitecture}\\native";

            //nativeLibraryPath = Kernel32.GetShortPath(nativeLibraryPath);

            //Kernel32.AddToPath(nativeLibraryPath);
#if DEBUG
            Console.WriteLine("nativeLibraryPath: " + nativeLibraryPath);
#endif
            OpenMpHandle = Kernel32.LoadLibraryEx(Path.Combine(nativeLibraryPath, OpenMpLibraryName + ".dll"), Kernel32.LoadLibraryFlags.LOAD_WITH_ALTERED_SEARCH_PATH, out ErrorCode _);
#if DEBUG
            Console.WriteLine($"OpenMpHandle: 0x{OpenMpHandle.ToString("X")}");
#endif

            MpiHandle = Kernel32.LoadLibraryEx(Path.Combine(nativeLibraryPath, MpiLibraryName + ".dll"), Kernel32.LoadLibraryFlags.LOAD_WITH_ALTERED_SEARCH_PATH, out ErrorCode _);
#if DEBUG
            Console.WriteLine($"MpiHandle: 0x{MpiHandle.ToString("X")}");
#endif

            CudartHandle = Kernel32.LoadLibraryEx(Path.Combine(nativeLibraryPath, CudartLibraryName + ".dll"), Kernel32.LoadLibraryFlags.LOAD_WITH_ALTERED_SEARCH_PATH, out ErrorCode _);
#if DEBUG
            Console.WriteLine($"CudartHandle: 0x{CudartHandle.ToString("X")}");
#endif
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        internal static void Load()
        {
            if(OpenMpHandle == IntPtr.Zero)
            {
                KokkosLibraryException.Throw(OpenMpLibraryName + "failed to load.");
            }

            if(MpiHandle == IntPtr.Zero)
            {
                KokkosLibraryException.Throw(MpiLibraryName + "failed to load.");
            }

            if(CudartHandle == IntPtr.Zero)
            {
                KokkosLibraryException.Throw(CudartLibraryName + "failed to load.");
            }

            KokkosCoreHandle = Kernel32.LoadLibraryEx(Path.Combine(nativeLibraryPath, KokkosCoreLibraryName + ".dll"),
                                                      Kernel32.LoadLibraryFlags.LOAD_WITH_ALTERED_SEARCH_PATH,
                                                      out ErrorCode _);

            Kernel32.DisableThreadLibraryCalls(KokkosCoreHandle);

            KokkosCoreModuleHandle = Kernel32.GetModuleHandleA(KokkosCoreLibraryName);

#if DEBUG
            Console.WriteLine($"KokkosCoreHandle: 0x{KokkosCoreHandle.ToString("X")}");
#endif

            if(KokkosCoreHandle == IntPtr.Zero)
            {
                KokkosLibraryException.Throw(KokkosCoreLibraryName + "failed to load.");
            }

            KokkosContainersHandle = Kernel32.LoadLibraryEx(Path.Combine(nativeLibraryPath, KokkosContainersLibraryName + ".dll"),
                                                            Kernel32.LoadLibraryFlags.LOAD_WITH_ALTERED_SEARCH_PATH,
                                                            out ErrorCode _);

            Kernel32.DisableThreadLibraryCalls(KokkosContainersHandle);

            KokkosContainersModuleHandle = Kernel32.GetModuleHandleA(KokkosContainersLibraryName);

#if DEBUG
            Console.WriteLine($"KokkosContainersHandle: 0x{KokkosContainersHandle.ToString("X")}");
#endif
            if(KokkosContainersHandle == IntPtr.Zero)
            {
                KokkosLibraryException.Throw(KokkosContainersLibraryName + "failed to load.");
            }

            Initialized = true;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        internal static void TerminateThreads()
        {
            Process process = Process.GetCurrentProcess();

            uint dwOwnerPID = (uint)process.Id;

            THREADENTRY32 te32 = new THREADENTRY32();

            IntPtr hThreadSnap = Kernel32.CreateToolhelp32Snapshot((uint)SnapshotFlags.Thread, 0);

            te32.dwSize = (uint)Unsafe.SizeOf<THREADENTRY32>();

            long nvcudaAddress = Kernel32.GetModuleHandleA("nvcuda.dll").ToInt64();

            if(Kernel32.Thread32First(hThreadSnap, ref te32))
            {
                do
                {
                    if(te32.th32OwnerProcessID == dwOwnerPID)
                    {
                        IntPtr ptrThread = Kernel32.OpenThread(0x0001, false, te32.th32ThreadID);

                        long startAddress = Kernel32.GetThreadStartAddress(process.Handle, te32.th32ThreadID).ToInt64();

                        //Console.WriteLine($"{te32.th32ThreadID.ToString("X")} {startAddress.ToString("X")}");
                        
                        if((startAddress > OpenMpHandle.ToInt64()) && (startAddress < OpenMpHandle.ToInt64() + 0x95344))
                        {
                            Kernel32.TerminateThread(ptrThread, 1);
                        }

                        if((startAddress > nvcudaAddress) && (startAddress < nvcudaAddress + 0x10C3000))
                        {
                            Kernel32.TerminateThread(ptrThread, 1);
                        }
                    }
                } while(Kernel32.Thread32Next(hThreadSnap, ref te32));
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        internal static void Unload()
        {
            TerminateThreads();

            if(!Kernel32.FreeLibrary(Kernel32.GetModuleHandleA(KokkosContainersLibraryName)))
            {
                KokkosLibraryException.Throw(KokkosContainersLibraryName + "failed to unload.");
            }
            else
            {
                KokkosContainersHandle       = IntPtr.Zero;
                KokkosContainersModuleHandle = IntPtr.Zero;
            }

            if(!Kernel32.FreeLibrary(Kernel32.GetModuleHandleA(KokkosCoreLibraryName)))
            {
                KokkosLibraryException.Throw(KokkosCoreLibraryName + "failed to unload.");
            }
            else
            {
                KokkosCoreHandle       = IntPtr.Zero;
                KokkosCoreModuleHandle = IntPtr.Zero;
            }

            Initialized = false;
        }
    }
}
