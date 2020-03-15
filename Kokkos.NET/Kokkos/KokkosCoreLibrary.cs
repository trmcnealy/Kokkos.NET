﻿using System;
using System.IO;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Versioning;
using System.Security;
using System.Text;

namespace Kokkos
{
    [NonVersionable]
    internal static class Kernel32
    {
        [Flags]
        public enum LoadLibraryFlags : uint
        {
            None                                = 0,
            DONT_RESOLVE_DLL_REFERENCES         = 0x00000001,
            LOAD_IGNORE_CODE_AUTHZ_LEVEL        = 0x00000010,
            LOAD_LIBRARY_AS_DATAFILE            = 0x00000002,
            LOAD_LIBRARY_AS_DATAFILE_EXCLUSIVE  = 0x00000040,
            LOAD_LIBRARY_AS_IMAGE_RESOURCE      = 0x00000020,
            LOAD_LIBRARY_SEARCH_APPLICATION_DIR = 0x00000200,
            LOAD_LIBRARY_SEARCH_DEFAULT_DIRS    = 0x00001000,
            LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR    = 0x00000100,
            LOAD_LIBRARY_SEARCH_SYSTEM32        = 0x00000800,
            LOAD_LIBRARY_SEARCH_USER_DIRS       = 0x00000400,
            LOAD_WITH_ALTERED_SEARCH_PATH       = 0x00000008
        }

        public const int MaxPath = 255;

        [SuppressUnmanagedCodeSecurity]
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        [DllImport("kernel32",
                   EntryPoint   = "LoadLibraryA",
                   CharSet      = CharSet.Ansi,
                   SetLastError = true)]
        private static extern IntPtr LoadLibrary([In] [MarshalAs(UnmanagedType.LPStr)] string libName);

        [SuppressUnmanagedCodeSecurity]
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        [DllImport("kernel32",
                   EntryPoint   = "LoadLibraryW",
                   CharSet      = CharSet.Unicode,
                   SetLastError = true)]
        private static extern IntPtr LoadLibraryW([In] [MarshalAs(UnmanagedType.LPWStr)] string libName);

        [SuppressUnmanagedCodeSecurity]
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        [DllImport("kernel32",
                   EntryPoint   = "LoadLibraryExA",
                   CharSet      = CharSet.Ansi,
                   SetLastError = true)]
        private static extern IntPtr LoadLibraryEx([In] [MarshalAs(UnmanagedType.LPStr)] string lpFileName,
                                                   IntPtr                                       hReservedNull,
                                                   uint                                         dwFlags);

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        [SuppressUnmanagedCodeSecurity]
        [DllImport("kernel32",
                   ExactSpelling = true,
                   CharSet       = CharSet.Unicode,
                   SetLastError  = true)]
        private static extern IntPtr LoadLibraryExW([In] [MarshalAs(UnmanagedType.LPWStr)] string lpwLibFileName,
                                                    [In]                                   IntPtr hFile,
                                                    [In]                                   uint   dwFlags);

        [SuppressUnmanagedCodeSecurity]
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        [DllImport("kernel32",
                   EntryPoint   = "GetShortPathNameW",
                   CharSet      = CharSet.Unicode,
                   SetLastError = true)]
        public static extern int GetShortPathName([MarshalAs(UnmanagedType.LPWStr)] string        path,
                                                  [MarshalAs(UnmanagedType.LPWStr)] StringBuilder shortPath,
                                                  uint                                            shortPathLength);

        [SuppressUnmanagedCodeSecurity]
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        [DllImport("kernel32",
                   EntryPoint   = "AddDllDirectory",
                   CharSet      = CharSet.Unicode,
                   SetLastError = true)]
        public static extern IntPtr AddDllDirectory([In] [MarshalAs(UnmanagedType.LPWStr)] string newDirectory);

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static string GetShortPath(string path)
        {
            StringBuilder shortPath = new StringBuilder(MaxPath);

            if(path.EndsWith("\\"))
            {
                path = path.Substring(0,
                                      path.Length - 1);
            }

            GetShortPathName(path,
                             shortPath,
                             MaxPath);

            return shortPath.ToString();
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static IntPtr AddDllDirectory(string        newDirectory,
                                             out ErrorCode errorCode)
        {
            IntPtr result = AddDllDirectory(newDirectory);
            errorCode = ((ErrorCode)Marshal.GetLastWin32Error()).IfErrorThrow();

            return result;
        }

        //[MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        //public static IntPtr LoadLibrary(string libName, out ErrorCode errorCode)
        //{
        //    IntPtr result = LoadLibrary(libName);
        //    errorCode = ((ErrorCode)Marshal.GetLastWin32Error()).IfErrorThrow();
        //    return result;
        //}

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static IntPtr LoadLibraryEx(string           lpFileName,
                                           LoadLibraryFlags dwFlags,
                                           out ErrorCode    errorCode)
        {
            //LoadLibraryFlags.LOAD_LIBRARY_AS_DATAFILE | LoadLibraryFlags.LOAD_LIBRARY_AS_IMAGE_RESOURCE | LoadLibraryFlags.LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR
            IntPtr result = LoadLibraryEx(lpFileName,
                                          IntPtr.Zero,
                                          (uint)dwFlags); // & 0xFFFFFF00

            errorCode = ((ErrorCode)Marshal.GetLastWin32Error()).IfErrorThrow();

            return result;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static bool AddToPath(string dirToAdd)
        {
            if(string.IsNullOrEmpty(dirToAdd))
            {
                return false;
            }

            if(!Directory.Exists(dirToAdd))
            {
                return false;
            }

            string text = Environment.GetEnvironmentVariable("PATH");

            if(text == null)
            {
                return false;
            }

            string[] array = text.Split(Path.PathSeparator);
            text += Path.PathSeparator;
            text =  text.Replace(dirToAdd + Path.PathSeparator, "");

            if(text[^1] == Path.PathSeparator)
            {
                text = text.Substring(0, text.Length - 1);
            }

            string value = dirToAdd + Path.PathSeparator + text;
            Environment.SetEnvironmentVariable("PATH", value);
#if DEBUG
            string PATH = Environment.GetEnvironmentVariable("PATH");
#endif
            return true;
        }

    }

    [NonVersionable]
    internal static class KokkosCoreLibrary
    {
        public const string KokkosCoreLibraryName = "libkokkoscore";

        public const string KokkosContainersLibraryName = "libkokkoscontainers";

        public const string OpenMpLibraryName = "libomp";

        public const string MpiLibraryName = "mpi";

        public const string CudartLibraryName = "cudart64_101";

        public static readonly IntPtr KokkosCoreHandle;

        public static readonly IntPtr KokkosContainersHandle;

        public static readonly IntPtr OpenMpHandle;

        public static readonly IntPtr MpiHandle;

        public static readonly IntPtr CudartHandle;

        public static volatile bool Initialized;
        
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        private static string GetLibraryPath()
        {
            string fullPath = typeof(KokkosCoreLibrary).Assembly.Location;

            if(string.IsNullOrEmpty(fullPath))
            {
                return null;
            }

            int lastIndex = fullPath.LastIndexOf("\\", StringComparison.Ordinal);


            return fullPath.Substring(0,
                                      lastIndex);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        static KokkosCoreLibrary()
        {
            string operatingSystem      = RuntimeInformation.IsOSPlatform(OSPlatform.Windows) ? "win" : "linux";
            string platformArchitecture = RuntimeInformation.ProcessArchitecture == Architecture.X64 ? "x64" : "x86";

            string nativeLibraryPath = Path.Combine(GetLibraryPath() ?? throw new NullReferenceException("typeof(KokkosCoreLibrary).Assembly.Location is empty."),
                                                    $"runtimes\\{operatingSystem}-{platformArchitecture}\\native");

            nativeLibraryPath = Kernel32.GetShortPath(nativeLibraryPath);
            
            bool addedToPath = Kernel32.AddToPath(nativeLibraryPath);

            Console.WriteLine("nativeLibraryPath:" + nativeLibraryPath);

            //Kernel32.AddDllDirectory(nativeLibraryPath,
            //                         out ErrorCode _);

            OpenMpHandle = Kernel32.LoadLibraryEx(OpenMpLibraryName + ".dll",
                                                  Kernel32.LoadLibraryFlags.None,
                                                  out ErrorCode _);

            Console.WriteLine($"OpenMpHandle: 0x{OpenMpHandle.ToString("X")}");

            MpiHandle = Kernel32.LoadLibraryEx(MpiLibraryName + ".dll",
                                               Kernel32.LoadLibraryFlags.None,
                                               out ErrorCode _);

            Console.WriteLine($"MpiHandle: 0x{MpiHandle.ToString("X")}");

            CudartHandle = Kernel32.LoadLibraryEx(CudartLibraryName + ".dll",
                                                  Kernel32.LoadLibraryFlags.None,
                                                  out ErrorCode _);

            Console.WriteLine($"CudartHandle: 0x{CudartHandle.ToString("X")}");

            KokkosCoreHandle = Kernel32.LoadLibraryEx(KokkosCoreLibraryName + ".dll",
                                                      Kernel32.LoadLibraryFlags.None,
                                                      out ErrorCode _);

            Console.WriteLine($"KokkosCoreHandle: 0x{KokkosCoreHandle.ToString("X")}");

            KokkosContainersHandle = Kernel32.LoadLibraryEx(KokkosContainersLibraryName + ".dll",
                                                            Kernel32.LoadLibraryFlags.None,
                                                            out ErrorCode _);

            Console.WriteLine($"KokkosContainersHandle: 0x{KokkosContainersHandle.ToString("X")}");
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        internal static void Initialize()
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

            if(KokkosCoreHandle == IntPtr.Zero)
            {
                KokkosLibraryException.Throw(KokkosCoreLibraryName + "failed to load.");
            }

            if(KokkosContainersHandle == IntPtr.Zero)
            {
                KokkosLibraryException.Throw(KokkosContainersLibraryName + "failed to load.");
            }

            Initialized = true;
        }
    }
}