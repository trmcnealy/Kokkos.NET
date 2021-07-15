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

using PlatformApi;
using PlatformApi.Win32;


using ThreadState = System.Diagnostics.ThreadState;

namespace Trilinos
{
    internal class TrilinosLibraryException : Exception
    {
        public TrilinosLibraryException()
        {
        }

        public TrilinosLibraryException(string message)
            : base(message)
        {
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        internal static void Throw()
        {
            throw new TrilinosLibraryException();
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        internal static void Throw(string message)
        {
            throw new TrilinosLibraryException(message);
        }
    }

    [NonVersionable]
    public static class MiscellaneousLibraries
    {
        public const string OpenMPLibraryName = "libomp.dll";

        public const string MpiLibraryName = "mpi.dll";

        public const string CudaLibraryName = "nvcuda.dll";

        public const string CudartLibraryName = "cudart64_110.dll";

        public const string CublasLtLibraryName = "cublasLt64_11.dll";

        public const string CublasLibraryName = "cublas64_11.dll";

        public const string CusparseLibraryName = "cusparse64_11.dll";

        
        public static readonly nint OpenMPHandle;

        public static readonly nint MpiHandle;

        public static readonly nint CudaHandle;

        public static readonly nint CudartHandle;

        public static readonly nint CublasLtHandle;

        public static readonly nint CublasHandle;

        public static readonly nint CusparseHandle;

        public static volatile bool IsLoaded;

        //private static readonly string nativeLibraryPath;

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
        private static string? GetLibraryPath()
        {
            string fullPath = System.Reflection.Assembly.GetExecutingAssembly().Location;

            if(!string.IsNullOrEmpty(fullPath) && !fullPath.Contains(".nuget"))
            {
                int lastIndex = fullPath.LastIndexOf("\\", StringComparison.Ordinal);

                return fullPath.Substring(0, lastIndex);
            }

            string? nugetPackagesEnvironmentVariable = Environment.GetEnvironmentVariable("NUGET_PACKAGES");

            if(!string.IsNullOrEmpty(nugetPackagesEnvironmentVariable))
            {
                string nativePackagePath = Path.Combine(nugetPackagesEnvironmentVariable, "native.kokkos.net");

                return GetNativePackagePath(nativePackagePath);
            }

            //const string dotnetProfileDirectoryName = ".dotnet";
            //const string toolsShimFolderName        = "tools";

            string? userProfile = Environment.GetEnvironmentVariable(RuntimeInformation.IsOSPlatform(OSPlatform.Windows) ? "USERPROFILE" : "HOME");

            if(!string.IsNullOrEmpty(userProfile))
            {
                string nativePackagePath = Path.Combine(userProfile, ".nuget", "packages", "native.kokkos.net");

                return GetNativePackagePath(nativePackagePath);
            }

            return null;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        static MiscellaneousLibraries()
        {
            OpenMPHandle = PlatformApi.NativeLibrary.LoadLibrary(OpenMPLibraryName);

            MpiHandle = PlatformApi.NativeLibrary.LoadLibrary(MpiLibraryName);

            CudaHandle = PlatformApi.NativeLibrary.LoadLibrary(CudaLibraryName);
#if DEBUG
            Console.WriteLine($"CudaHandle: 0x{CudaHandle.ToString("X")}");
#endif
            CudartHandle = PlatformApi.NativeLibrary.LoadLibrary(CudartLibraryName);
#if DEBUG
            Console.WriteLine($"CudartHandle: 0x{CudartHandle.ToString("X")}");
#endif
            CublasLtHandle = PlatformApi.NativeLibrary.LoadLibrary(CublasLtLibraryName);
#if DEBUG
            Console.WriteLine($"CublasLtHandle: 0x{CublasLtHandle.ToString("X")}");
#endif
            CublasHandle = PlatformApi.NativeLibrary.LoadLibrary(CublasLibraryName);
#if DEBUG
            Console.WriteLine($"CublasHandle: 0x{CublasHandle.ToString("X")}");
#endif
            CusparseHandle = PlatformApi.NativeLibrary.LoadLibrary(CusparseLibraryName);
#if DEBUG
            Console.WriteLine($"CusparseHandle: 0x{CusparseHandle.ToString("X")}");
#endif
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static void Load()
        {
            if(!IsLoaded)
            {
                if(CudartHandle == 0)
                {
                    TrilinosLibraryException.Throw(CudartLibraryName + "failed to load.");
                }

//                KokkosCoreHandle = PlatformApi.NativeLibrary.LoadLibrary(KokkosCoreLibraryName);

//                Kernel32.Native.DisableThreadLibraryCalls(KokkosCoreHandle);

//                KokkosCoreModuleHandle = Kernel32.Native.GetModuleHandle(KokkosCoreLibraryName);

//#if DEBUG
//                Console.WriteLine($"KokkosCoreHandle: 0x{KokkosCoreHandle.ToString("X")}");
//#endif

//                if(KokkosCoreHandle == 0)
//                {
//                    KokkosLibraryException.Throw(KokkosCoreLibraryName + "failed to load.");
//                }

//                KokkosContainersHandle = PlatformApi.NativeLibrary.LoadLibrary(KokkosContainersLibraryName);

//                Kernel32.Native.DisableThreadLibraryCalls(KokkosContainersHandle);

//                KokkosContainersModuleHandle = Kernel32.Native.GetModuleHandle(KokkosContainersLibraryName);

//#if DEBUG
//                Console.WriteLine($"KokkosContainersHandle: 0x{KokkosContainersHandle.ToString("X")}");
//#endif
//                if(KokkosContainersHandle == 0)
//                {
//                    KokkosLibraryException.Throw(KokkosContainersLibraryName + "failed to load.");
//                }

                //KokkosKernelsHandle = PlatformApi.NativeLibrary.Load(KokkosKernelsLibraryName);

                //Kernel32.Native.DisableThreadLibraryCalls(KokkosKernelsHandle);

                //KokkosKernelsModuleHandle = Kernel32.Native.GetModuleHandle(KokkosKernelsLibraryName);

                IsLoaded = true;
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static void TerminateThreads()
        {
            Process process = Process.GetCurrentProcess();

            uint dwOwnerPID = (uint)process.Id;

            THREADENTRY32 te32 = new THREADENTRY32();

            nint hThreadSnap = Kernel32.Native.CreateToolhelp32Snapshot((uint)SnapshotFlags.Thread, 0);

            te32.dwSize = (uint)Unsafe.SizeOf<THREADENTRY32>();

            long nvcudaAddress = (long)Kernel32.Native.GetModuleHandle("nvcuda.dll");

            if(Kernel32.Native.Thread32First(hThreadSnap, ref te32))
            {
                do
                {
                    if(te32.th32OwnerProcessID == dwOwnerPID)
                    {
                        nint ptrThread = Kernel32.Native.OpenThread(0x0001, false, te32.th32ThreadID);

                        long startAddress = (long)Kernel32.GetThreadStartAddress(process.Handle, te32.th32ThreadID);

                        //Console.WriteLine($"{te32.th32ThreadID.ToString("X")} {startAddress.ToString("X")}");

                        //if(startAddress > OpenMpHandle && startAddress < OpenMpHandle + 0x95344)
                        //{
                        //    Kernel32.Native.TerminateThread(ptrThread, 1);
                        //}

                        if(startAddress > nvcudaAddress && startAddress < nvcudaAddress + 0x10C3000)
                        {
                            Kernel32.Native.TerminateThread(ptrThread, 1);
                        }
                    }
                } while(Kernel32.Native.Thread32Next(hThreadSnap, ref te32));
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static void Unload()
        {
            TerminateThreads();

            //if(!PlatformApi.NativeLibrary.Free(Kernel32.Native.GetModuleHandle(KokkosContainersLibraryName)))
            //{
            //    KokkosLibraryException.Throw(KokkosContainersLibraryName + "failed to unload.");
            //}
            //else
            //{
            //    KokkosContainersHandle = 0;
            //    KokkosContainersModuleHandle = 0;
            //}

            //if(!PlatformApi.NativeLibrary.Free(Kernel32.Native.GetModuleHandle(KokkosCoreLibraryName)))
            //{
            //    KokkosLibraryException.Throw(KokkosCoreLibraryName + "failed to unload.");
            //}
            //else
            //{
            //    KokkosCoreHandle = 0;
            //    KokkosCoreModuleHandle = 0;
            //}

            IsLoaded = false;
        }
    }
}
