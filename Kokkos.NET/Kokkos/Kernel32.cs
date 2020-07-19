#nullable enable
using System;
using System.Diagnostics.CodeAnalysis;
using System.IO;
using System.Reflection;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Versioning;
using System.Security;
using System.Text;

namespace Kokkos
{
    [Flags]
    public enum ThreadAccess : int
    {
        TERMINATE                = (0x0001),
        SUSPEND_RESUME           = (0x0002),
        GET_CONTEXT              = (0x0008),
        SET_CONTEXT              = (0x0010),
        SET_INFORMATION          = (0x0020),
        QUERY_INFORMATION        = (0x0040),
        SET_THREAD_TOKEN         = (0x0080),
        IMPERSONATE              = (0x0100),
        DIRECT_IMPERSONATION     = (0x0200),
        STANDARD_RIGHTS_REQUIRED = (0x000F0000),
        SYNCHRONIZE              = (0x00100000),

        // vista and later
        THREAD_ALL_ACCESS = (STANDARD_RIGHTS_REQUIRED | SYNCHRONIZE | 0xFFFF)
    }

    [StructLayout(System.Runtime.InteropServices.LayoutKind.Sequential)]
    public struct MODULEINFO
    {
        public IntPtr lpBaseOfDll;
        public uint   SizeOfImage;
        public IntPtr EntryPoint;
    }

    [Flags]
    public enum SnapshotFlags : uint
    {
        HeapList = 0x00000001,
        Process  = 0x00000002,
        Thread   = 0x00000004,
        Module   = 0x00000008,
        Module32 = 0x00000010,
        Inherit  = 0x80000000,
        All      = HeapList | Module | Process | Thread
    }

    public enum ThreadInfoClass : int
    {
        ThreadBasicInformation          = 0,
        ThreadQuerySetWin32StartAddress = 9
    }

    [NonVersionable]
    public static class Ntdll
    {
        static Ntdll()
        {
            Marshal.PrelinkAll(typeof(Psapi));
        }

        [SuppressUnmanagedCodeSecurity]
#if NETSTANDARD
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
#else
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
#endif
        [DllImport("ntdll", ExactSpelling = true, SetLastError = true)]
        public static extern int NtQueryInformationThread(IntPtr                       ThreadHandle,
                                                          ThreadInfoClass              ThreadInformationClass,
                                                          IntPtr ThreadInformation,
                                                          int                          ThreadInformationLength,
                                                          out int                      ReturnLength);
    }

    [NonVersionable]
    public static class Psapi
    {
        static Psapi()
        {
            Marshal.PrelinkAll(typeof(Psapi));
        }

        [SuppressUnmanagedCodeSecurity]
#if NETSTANDARD
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
#else
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
#endif
        [DllImport("psapi", ExactSpelling = true, SetLastError = true)]
        public static extern bool GetModuleInformation(IntPtr         hProcess,
                                                       IntPtr         hModule,
                                                       ref MODULEINFO lpmodinfo,
                                                       uint           cb);
    }

    [NonVersionable]
    public static class Ole32
    {
        static Ole32()
        {
            Marshal.PrelinkAll(typeof(Ole32));
        }

        [SuppressUnmanagedCodeSecurity]
#if NETSTANDARD
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
#else
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
#endif
        [DllImport("ole32", EntryPoint = "CoFreeUnusedLibraries", CharSet = CharSet.Ansi, SetLastError = true)]
        public static extern void CoFreeUnusedLibraries();
    }

    [NonVersionable]
    public static class Kernel32
    {
        static Kernel32()
        {
            Marshal.PrelinkAll(typeof(Kernel32));
        }

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

        [Flags]
        public enum GetModuleFlags : uint
        {
            GET_MODULE_HANDLE_EX_FLAG_PIN                = 0x00000001,
            GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT = 0x00000002,
            GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS       = 0x00000004
        }

        public const int MaxPath = 255;

        [SuppressUnmanagedCodeSecurity]
#if NETSTANDARD
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
#else
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
#endif
        [DllImport("kernel32", EntryPoint = "GetLastError")]
        public static extern ErrorCode GetLastError();

#if NETSTANDARD
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
#else
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
#endif
        [SuppressUnmanagedCodeSecurity]
        [DllImport("kernel32", ExactSpelling = true, SetLastError = true)]
        public static extern IntPtr GetCurrentProcess();

        [SuppressUnmanagedCodeSecurity]
#if NETSTANDARD
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
#else
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
#endif
        [DllImport("kernel32", EntryPoint = "LoadLibraryA", CharSet = CharSet.Ansi, SetLastError = true)]
        public static extern IntPtr LoadLibrary([In] [MarshalAs(UnmanagedType.LPStr)] string libName);

        [SuppressUnmanagedCodeSecurity]
#if NETSTANDARD
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
#else
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
#endif
        [DllImport("kernel32", EntryPoint = "LoadLibraryW", CharSet = CharSet.Unicode, SetLastError = true)]
        public static extern IntPtr LoadLibraryW([In] [MarshalAs(UnmanagedType.LPWStr)] string libName);

        [SuppressUnmanagedCodeSecurity]
#if NETSTANDARD
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
#else
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
#endif
        [DllImport("kernel32", EntryPoint = "LoadLibraryExA", CharSet = CharSet.Ansi, SetLastError = true)]
        public static extern IntPtr LoadLibraryEx([In] [MarshalAs(UnmanagedType.LPStr)] string lpFileName,
                                                  IntPtr                                       hReservedNull,
                                                  uint                                         dwFlags);

#if NETSTANDARD
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
#else
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
#endif
        [SuppressUnmanagedCodeSecurity]
        [DllImport("kernel32", ExactSpelling = true, CharSet = CharSet.Unicode, SetLastError = true)]
        public static extern IntPtr LoadLibraryExW([In] [MarshalAs(UnmanagedType.LPWStr)] string lpwLibFileName,
                                                   [In]                                   IntPtr hFile,
                                                   [In]                                   uint   dwFlags);

#if NETSTANDARD
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
#else
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
#endif
        [SuppressUnmanagedCodeSecurity]
        [DllImport("kernel32", ExactSpelling = true, SetLastError = true)]
        public static extern bool FreeLibrary(IntPtr hModule);

#if NETSTANDARD
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
#else
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
#endif
        [SuppressUnmanagedCodeSecurity]
        [DllImport("kernel32", ExactSpelling = true, SetLastError = true)]
        public static extern void FreeLibraryAndExitThread(IntPtr hLibModule,
                                                           uint   dwExitCode);

        [SuppressUnmanagedCodeSecurity]
#if NETSTANDARD
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
#else
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
#endif
        [DllImport("kernel32", EntryPoint = "GetShortPathNameW", CharSet = CharSet.Unicode, SetLastError = true)]
        public static extern int GetShortPathName([MarshalAs(UnmanagedType.LPWStr)] string        path,
                                                  [MarshalAs(UnmanagedType.LPWStr)] StringBuilder shortPath,
                                                  uint                                            shortPathLength);

#if NETSTANDARD
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
#else
#if NETSTANDARD
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
#else
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
#endif
#endif
        [SuppressUnmanagedCodeSecurity]
        [DllImport("kernel32", EntryPoint = "AddDllDirectory", CharSet = CharSet.Unicode, SetLastError = true)]
        public static extern IntPtr AddDllDirectory([In] [MarshalAs(UnmanagedType.LPWStr)] string newDirectory);

#if NETSTANDARD
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
#else
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
#endif
        public static string GetShortPath(string path)
        {
            StringBuilder shortPath = new StringBuilder(MaxPath);

            if(path.EndsWith("\\"))
            {
                path = path.Substring(0, path.Length - 1);
            }

            GetShortPathName(path, shortPath, MaxPath);

            return shortPath.ToString();
        }

#if NETSTANDARD
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
#else
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
#endif
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

#if NETSTANDARD
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
#else
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
#endif
        public static IntPtr LoadLibraryEx(string           lpFileName,
                                           LoadLibraryFlags dwFlags,
                                           out ErrorCode    errorCode)
        {
            //LoadLibraryFlags.LOAD_LIBRARY_AS_DATAFILE | LoadLibraryFlags.LOAD_LIBRARY_AS_IMAGE_RESOURCE | LoadLibraryFlags.LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR
            IntPtr result = LoadLibraryEx(lpFileName, IntPtr.Zero, (uint)dwFlags); // & 0xFFFFFF00

            errorCode = ((ErrorCode)Marshal.GetLastWin32Error()).IfErrorThrow();

            return result;
        }

#if NETSTANDARD
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
#else
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
#endif
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

            string? text = Environment.GetEnvironmentVariable("PATH");

            if(text == null)
            {
                return false;
            }

            //string[] array = text.Split(Path.PathSeparator);

            text += Path.PathSeparator;

            text = text.Replace(dirToAdd + Path.PathSeparator, "");

            if(text[^1] == Path.PathSeparator)
            {
                text = text.Substring(0, text.Length - 1);
            }

            string value = dirToAdd + Path.PathSeparator + text;

            Environment.SetEnvironmentVariable("PATH", value);
#if DEBUG
            string? path = Environment.GetEnvironmentVariable("PATH");
#endif
            return true;
        }

        [SuppressUnmanagedCodeSecurity]
#if NETSTANDARD
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
#else
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
#endif
        [DllImport("kernel32", ExactSpelling = true, CharSet = CharSet.Ansi, SetLastError = true)]
        public static extern IntPtr GetModuleHandleA(string lpModuleName);

        [SuppressUnmanagedCodeSecurity]
#if NETSTANDARD
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
#else
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
#endif
        [DllImport("kernel32", ExactSpelling = true, CharSet = CharSet.Ansi, SetLastError = true)]
        public static extern bool GetModuleHandleExA(uint       dwFlags,
                                                     string     lpModuleName,
                                                     out IntPtr phModule);

        [SuppressUnmanagedCodeSecurity]
#if NETSTANDARD
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
#else
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
#endif
        [DllImport("kernel32", ExactSpelling = true, CharSet = CharSet.Unicode, SetLastError = true)]
        public static extern IntPtr GetModuleHandleW(string lpModuleName);

        [SuppressUnmanagedCodeSecurity]
#if NETSTANDARD
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
#else
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
#endif
        [DllImport("kernel32", ExactSpelling = true, CharSet = CharSet.Unicode, SetLastError = true)]
        public static extern bool GetModuleHandleExW(uint       dwFlags,
                                                     string     lpModuleName,
                                                     out IntPtr phModule);

        [SuppressUnmanagedCodeSecurity]
#if NETSTANDARD
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
#else
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
#endif
        [DllImport("kernel32", EntryPoint = "GetProcAddress", SetLastError = true)]
        public static extern IntPtr GetProcAddress(IntPtr hModule,
                                                   string lpProcName);

        [SuppressUnmanagedCodeSecurity]
#if NETSTANDARD
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
#else
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
#endif
        [DllImport("kernel32", ExactSpelling = true, SetLastError = true)]
        public static extern bool DisableThreadLibraryCalls(IntPtr hLibModule);

        [SuppressUnmanagedCodeSecurity]
#if NETSTANDARD
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
#else
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
#endif
        [DllImport("kernel32", ExactSpelling = true, SetLastError = true)]
        public static extern IntPtr OpenThread(uint dwDesiredAccess,
                                               bool bInheritHandle,
                                               uint dwThreadId);

        [SuppressUnmanagedCodeSecurity]
#if NETSTANDARD
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
#else
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
#endif
        [DllImport("kernel32", ExactSpelling = true, SetLastError = true)]
        public static extern bool TerminateThread(IntPtr hThread,
                                                  uint   dwExitCode);

        [SuppressUnmanagedCodeSecurity]
#if NETSTANDARD
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
#else
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
#endif
        [DllImport("kernel32", ExactSpelling = true, SetLastError = true)]
        public static extern IntPtr CreateToolhelp32Snapshot(uint flags,
                                                             uint toolhelp32ProcessID);

        [SuppressUnmanagedCodeSecurity]
#if NETSTANDARD
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
#else
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
#endif
        [DllImport("kernel32", ExactSpelling = true, SetLastError = true)]
        public static extern bool Process32First(IntPtr             snapshotHandle,
                                                 ref PROCESSENTRY32 processEntryPointer);

        [SuppressUnmanagedCodeSecurity]
#if NETSTANDARD
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
#else
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
#endif
        [DllImport("kernel32", ExactSpelling = true, SetLastError = true)]
        public static extern bool Process32Next(IntPtr             snapshotHandle,
                                                ref PROCESSENTRY32 processEntryPointer);

        [SuppressUnmanagedCodeSecurity]
#if NETSTANDARD
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
#else
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
#endif
        [DllImport("kernel32", ExactSpelling = true, SetLastError = true)]
        public static extern bool Thread32First(IntPtr            hSnapshot,
                                                ref THREADENTRY32 lpte);

        [SuppressUnmanagedCodeSecurity]
#if NETSTANDARD
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
#else
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
#endif
        [DllImport("kernel32", ExactSpelling = true, SetLastError = true)]
        public static extern bool Thread32Next(IntPtr            hSnapshot,
                                               ref THREADENTRY32 lpte);

        [SuppressUnmanagedCodeSecurity]
#if NETSTANDARD
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
#else
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
#endif
        [DllImport("kernel32", ExactSpelling = true, SetLastError = true)]
        public static extern uint GetThreadId(IntPtr Thread);

        [SuppressUnmanagedCodeSecurity]
#if NETSTANDARD
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
#else
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
#endif
        [DllImport("kernel32", ExactSpelling = true, SetLastError = true)]
        public static extern HRESULT GetThreadDescription(IntPtr     hThread,
                                                          ref string ppszThreadDescription);

#if NETSTANDARD
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
#else
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
#endif
        [DllImport("kernel32", ExactSpelling = true, SetLastError = true)]
        public static extern bool CloseHandle(IntPtr hObject);

#if NETSTANDARD
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
#else
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
#endif
        [DllImport("kernel32", ExactSpelling = true, SetLastError = true)]
        public static extern bool ReadProcessMemory(IntPtr lpProcess,
                                                    IntPtr lpBaseAddress,
                                                    IntPtr lpBuffer,
                                                    int    nSize,
                                                    IntPtr bytesRead);

#if NETSTANDARD
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
#else
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
#endif
        [DllImport("kernel32", ExactSpelling = true, SetLastError = true)]
        public static extern bool UnmapViewOfFile(IntPtr lpBaseAddress);

        public static IntPtr GetThreadStartAddress(IntPtr hProcess,
                                                   uint   threadId)
        {
            IntPtr hThread = Kernel32.OpenThread((uint)ThreadAccess.QUERY_INFORMATION, false, threadId);

            if(hThread == IntPtr.Zero)
            {
                return hThread;
            }

            byte[] buf = new byte[4];

            try
            {
                unsafe
                {
                    ulong startAddress = 0;

                    int status = Ntdll.NtQueryInformationThread(hThread, ThreadInfoClass.ThreadQuerySetWin32StartAddress, new IntPtr(&startAddress), 8, out _);




                    // if(result != 0)
                    // {
                    //     throw new Exception("NtQueryInformationThread failed; NTSTATUS = {0:X8}", result);
                    // }

                    return (IntPtr)startAddress;
                }
            }
            finally
            {
                Kernel32.CloseHandle(hThread);
            }
        }
    }

    [StructLayout(System.Runtime.InteropServices.LayoutKind.Sequential)]
    public struct CLIENT_ID
    {
        public IntPtr UniqueProcess; // original: PVOID
        public IntPtr UniqueThread;  // original: PVOID
    }

    [StructLayout(System.Runtime.InteropServices.LayoutKind.Sequential)]
    public struct THREAD_BASIC_INFORMATION
    {
        public IntPtr    ExitStatus;     // original: LONG NTSTATUS
        public IntPtr    TebBaseAddress; // original: PVOID
        public CLIENT_ID ClientId;
        public IntPtr    AffinityMask; // original: ULONG_PTR
        public uint      Priority;     // original: DWORD
        public uint      BasePriority; // original: DWORD
    }

    [StructLayout(System.Runtime.InteropServices.LayoutKind.Sequential)]
    public struct PROCESSENTRY32
    {
        public uint   size;
        public uint   countUsage;
        public uint   toolHelp32ProcessID;
        public IntPtr toolHelp32DefaultHeapID;
        public uint   toolHelp32ModuleID;
        public uint   countThreads;
        public uint   toolHelp32ParentProcessID;
        public int    pcPriorityClassBase;
        public uint   flags;

        [MarshalAs(UnmanagedType.ByValTStr, SizeConst = 255)]
        public string exeFile;
    }

    [StructLayout(System.Runtime.InteropServices.LayoutKind.Sequential)]
    public struct THREADENTRY32
    {
        public uint dwSize;

        public uint cntUsage;

        public uint th32ThreadID;

        public uint th32OwnerProcessID;

        public int tpBasePri;

        public int tpDeltaPri;

        public uint dwFlags;
    }

    [SuppressMessage("ReSharper", "InconsistentNaming")]
    public readonly struct HRESULT //: IEquatable<HRESULT>, IComparable<HRESULT>, IComparable
    {
        public const uint E_UNEXPECTED = 0x8000FFFF;

        public const uint E_NOTIMPL = 0x80004001;

        public const uint E_OUTOFMEMORY = 0x8007000E;

        public const uint E_INVALIDARG = 0x80070057;

        public const uint E_NOINTERFACE = 0x80004002;

        public const uint E_POINTER = 0x80004003;

        public const uint E_HANDLE = 0x80070006;

        public const uint E_ABORT = 0x80004004;

        public const uint E_FAIL = 0x80004005;

        public const uint E_ACCESSDENIED = 0x80070005;

        public const uint E_PENDING = 0x8000000A;

        public const uint E_BOUNDS = 0x8000000B;

        public const uint E_CHANGED_STATE = 0x8000000C;

        public const uint E_ILLEGAL_STATE_CHANGE = 0x8000000D;

        public const uint E_ILLEGAL_METHOD_CALL = 0x8000000E;

        public const uint E_STRING_NOT_NULL_TERMINATED = 0x80000017;

        public const uint E_ILLEGAL_DELEGATE_ASSIGNMENT = 0x80000018;

        public const uint E_ASYNC_OPERATION_NOT_STARTED = 0x80000019;

        public const uint E_APPLICATION_EXITING = 0x8000001A;

        public const uint E_APPLICATION_VIEW_EXITING = 0x8000001B;

        public const uint S_OK    = 0;
        public const uint S_FALSE = 1;

        public readonly uint value;

        public uint Value { get { return value; } }

        [NonVersionable]
        public HRESULT(uint value)
        {
            this.value = value;
        }

        public override string? ToString()
        {
            return value.ToString();
        }

        [NonVersionable]
        public static implicit operator HRESULT(uint value)
        {
            return new HRESULT(value);
        }

        [NonVersionable]
        public static explicit operator uint(in HRESULT value)
        {
            return value!.Value;
        }

        public bool Equals(in HRESULT other)
        {
            return value == other.value;
        }

        public override bool Equals(object? obj)
        {
            return obj is HRESULT other && Equals(other);
        }

        public override int GetHashCode()
        {
            return (int)value;
        }

        public static bool operator ==(in HRESULT left,
                                       in HRESULT right)
        {
            return left.Equals(right);
        }

        public static bool operator !=(in HRESULT left,
                                       in HRESULT right)
        {
            return !left.Equals(right);
        }

        public int CompareTo(in HRESULT other)
        {
            return value.CompareTo(other.value);
        }

        public int CompareTo(object? obj)
        {
            if(ReferenceEquals(null, obj))
            {
                return 1;
            }

            return obj is HRESULT other ? CompareTo(other) : throw new ArgumentException($"Object must be of type {nameof(HRESULT)}");
        }

        public static bool operator <(in HRESULT left,
                                      in HRESULT right)
        {
            return left.CompareTo(right) < 0;
        }

        public static bool operator >(in HRESULT left,
                                      in HRESULT right)
        {
            return left.CompareTo(right) > 0;
        }

        public static bool operator <=(in HRESULT left,
                                       in HRESULT right)
        {
            return left.CompareTo(right) <= 0;
        }

        public static bool operator >=(in HRESULT left,
                                       in HRESULT right)
        {
            return left.CompareTo(right) >= 0;
        }
    }
}
