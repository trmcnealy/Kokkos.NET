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
    [NonVersionable]
    public static class UcrtLibraries
    {
        //public const string CoreConsoleLibraryName            = "api-ms-win-core-console-l1-1-0.dll";
        //public const string CoreDatetimeLibraryName           = "api-ms-win-core-datetime-l1-1-0.dll";
        //public const string CoreDebugLibraryName              = "api-ms-win-core-debug-l1-1-0.dll";
        //public const string CoreErrorhandlingLibraryName      = "api-ms-win-core-errorhandling-l1-1-0.dll";
        //public const string CoreFilel11LibraryName            = "api-ms-win-core-file-l1-1-0.dll";
        //public const string CoreFilel12LibraryName            = "api-ms-win-core-file-l1-2-0.dll";
        //public const string CoreFilel2LibraryName             = "api-ms-win-core-file-l2-1-0.dll";
        //public const string CoreHandleLibraryName             = "api-ms-win-core-handle-l1-1-0.dll";
        //public const string CoreHeapLibraryName               = "api-ms-win-core-heap-l1-1-0.dll";
        //public const string CoreInterlockedLibraryName        = "api-ms-win-core-interlocked-l1-1-0.dll";
        //public const string CoreLibraryloaderLibraryName      = "api-ms-win-core-libraryloader-l1-1-0.dll";
        //public const string CoreLocalizationLibraryName       = "api-ms-win-core-localization-l1-2-0.dll";
        //public const string CoreMemoryLibraryName             = "api-ms-win-core-memory-l1-1-0.dll";
        //public const string CoreNamedpipeLibraryName          = "api-ms-win-core-namedpipe-l1-1-0.dll";
        //public const string CoreProcessenvironmentLibraryName = "api-ms-win-core-processenvironment-l1-1-0.dll";
        //public const string CoreProcessthreadsl110LibraryName = "api-ms-win-core-processthreads-l1-1-0.dll";
        //public const string CoreProcessthreadsl111LibraryName = "api-ms-win-core-processthreads-l1-1-1.dll";
        //public const string CoreProfileLibraryName            = "api-ms-win-core-profile-l1-1-0.dll";
        //public const string CoreRtlsupportLibraryName         = "api-ms-win-core-rtlsupport-l1-1-0.dll";
        //public const string CoreStringLibraryName             = "api-ms-win-core-string-l1-1-0.dll";
        //public const string CoreSynchl11LibraryName           = "api-ms-win-core-synch-l1-1-0.dll";
        //public const string CoreSynchl12LibraryName           = "api-ms-win-core-synch-l1-2-0.dll";
        //public const string CoreSysinfoLibraryName            = "api-ms-win-core-sysinfo-l1-1-0.dll";
        //public const string CoreTimezoneLibraryName           = "api-ms-win-core-timezone-l1-1-0.dll";
        //public const string CoreUtilLibraryName               = "api-ms-win-core-util-l1-1-0.dll";
        public const string CrtConioLibraryName = "api-ms-win-crt-conio-l1-1-0.dll";
        public const string CrtConvertLibraryName = "api-ms-win-crt-convert-l1-1-0.dll";
        public const string CrtEnvironmentLibraryName = "api-ms-win-crt-environment-l1-1-0.dll";
        public const string CrtFilesystemLibraryName = "api-ms-win-crt-filesystem-l1-1-0.dll";
        public const string CrtHeapLibraryName = "api-ms-win-crt-heap-l1-1-0.dll";
        public const string CrtLocaleLibraryName = "api-ms-win-crt-locale-l1-1-0.dll";
        public const string CrtMathLibraryName = "api-ms-win-crt-math-l1-1-0.dll";
        public const string CrtMultibyteLibraryName = "api-ms-win-crt-multibyte-l1-1-0.dll";
        public const string CrtPrivateLibraryName = "api-ms-win-crt-private-l1-1-0.dll";
        public const string CrtProcessLibraryName = "api-ms-win-crt-process-l1-1-0.dll";
        public const string CrtRuntimeLibraryName = "api-ms-win-crt-runtime-l1-1-0.dll";
        public const string CrtStdioLibraryName = "api-ms-win-crt-stdio-l1-1-0.dll";
        public const string CrtStringLibraryName = "api-ms-win-crt-string-l1-1-0.dll";
        public const string CrtTimeLibraryName = "api-ms-win-crt-time-l1-1-0.dll";
        public const string CrtUtilityLibraryName = "api-ms-win-crt-utility-l1-1-0.dll";
        //public const string UcrtbaseLibraryName               = "ucrtbase.dll";

        //public static readonly nint CoreConsoleHandle;
        //public static readonly nint CoreDatetimeHandle;
        //public static readonly nint CoreDebugHandle;
        //public static readonly nint CoreErrorhandlingHandle;
        //public static readonly nint CoreFilel11Handle;
        //public static readonly nint CoreFilel12Handle;
        //public static readonly nint CoreFilel2Handle;
        //public static readonly nint CoreHandleHandle;
        //public static readonly nint CoreHeapHandle;
        //public static readonly nint CoreInterlockedHandle;
        //public static readonly nint CoreLibraryloaderHandle;
        //public static readonly nint CoreLocalizationHandle;
        //public static readonly nint CoreMemoryHandle;
        //public static readonly nint CoreNamedpipeHandle;
        //public static readonly nint CoreProcessenvironmentHandle;
        //public static readonly nint CoreProcessthreadsl110Handle;
        //public static readonly nint CoreProcessthreadsl111Handle;
        //public static readonly nint CoreProfileHandle;
        //public static readonly nint CoreRtlsupportHandle;
        //public static readonly nint CoreStringHandle;
        //public static readonly nint CoreSynchl11Handle;
        //public static readonly nint CoreSynchl12Handle;
        //public static readonly nint CoreSysinfoHandle;
        //public static readonly nint CoreTimezoneHandle;
        //public static readonly nint CoreUtilHandle;
        public static readonly nint CrtConioHandle;
        public static readonly nint CrtConvertHandle;
        public static readonly nint CrtEnvironmentHandle;
        public static readonly nint CrtFilesystemHandle;
        public static readonly nint CrtHeapHandle;
        public static readonly nint CrtLocaleHandle;
        public static readonly nint CrtMathHandle;
        public static readonly nint CrtMultibyteHandle;
        public static readonly nint CrtPrivateHandle;
        public static readonly nint CrtProcessHandle;
        public static readonly nint CrtRuntimeHandle;
        public static readonly nint CrtStdioHandle;
        public static readonly nint CrtStringHandle;
        public static readonly nint CrtTimeHandle;
        public static readonly nint CrtUtilityHandle;
        //public static readonly nint UcrtbaseHandle;

        public static unsafe delegate* unmanaged<in string, nint> GetModuleHandleW;

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        static unsafe UcrtLibraries()
        {
            nint Kernel32 = PlatformApi.NativeLibrary.LoadLibrary("Kernel32.dll");

            GetModuleHandleW = (delegate* unmanaged<in string, nint>)PlatformApi.NativeLibrary.GetExport(Kernel32, "GetModuleHandleW");

            //CoreConsoleHandle            = PlatformApi.NativeLibrary.LoadLibrary(CoreConsoleLibraryName);
            //CoreDatetimeHandle           = PlatformApi.NativeLibrary.LoadLibrary(CoreDatetimeLibraryName);
            //CoreDebugHandle              = PlatformApi.NativeLibrary.LoadLibrary(CoreDebugLibraryName);
            //CoreErrorhandlingHandle      = PlatformApi.NativeLibrary.LoadLibrary(CoreErrorhandlingLibraryName);
            //CoreFilel11Handle            = PlatformApi.NativeLibrary.LoadLibrary(CoreFilel11LibraryName);
            //CoreFilel12Handle            = PlatformApi.NativeLibrary.LoadLibrary(CoreFilel12LibraryName);
            //CoreFilel2Handle             = PlatformApi.NativeLibrary.LoadLibrary(CoreFilel2LibraryName);
            //CoreHandleHandle             = PlatformApi.NativeLibrary.LoadLibrary(CoreHandleLibraryName);
            //CoreHeapHandle               = PlatformApi.NativeLibrary.LoadLibrary(CoreHeapLibraryName);
            //CoreInterlockedHandle        = PlatformApi.NativeLibrary.LoadLibrary(CoreInterlockedLibraryName);
            //CoreLibraryloaderHandle      = PlatformApi.NativeLibrary.LoadLibrary(CoreLibraryloaderLibraryName);
            //CoreLocalizationHandle       = PlatformApi.NativeLibrary.LoadLibrary(CoreLocalizationLibraryName);
            //CoreMemoryHandle             = PlatformApi.NativeLibrary.LoadLibrary(CoreMemoryLibraryName);
            //CoreNamedpipeHandle          = PlatformApi.NativeLibrary.LoadLibrary(CoreNamedpipeLibraryName);
            //CoreProcessenvironmentHandle = PlatformApi.NativeLibrary.LoadLibrary(CoreProcessenvironmentLibraryName);
            //CoreProcessthreadsl110Handle = PlatformApi.NativeLibrary.LoadLibrary(CoreProcessthreadsl110LibraryName);
            //CoreProcessthreadsl111Handle = PlatformApi.NativeLibrary.LoadLibrary(CoreProcessthreadsl111LibraryName);
            //CoreProfileHandle            = PlatformApi.NativeLibrary.LoadLibrary(CoreProfileLibraryName);
            //CoreRtlsupportHandle         = PlatformApi.NativeLibrary.LoadLibrary(CoreRtlsupportLibraryName);
            //CoreStringHandle             = PlatformApi.NativeLibrary.LoadLibrary(CoreStringLibraryName);
            //CoreSynchl11Handle           = PlatformApi.NativeLibrary.LoadLibrary(CoreSynchl11LibraryName);
            //CoreSynchl12Handle           = PlatformApi.NativeLibrary.LoadLibrary(CoreSynchl12LibraryName);
            //CoreSysinfoHandle            = PlatformApi.NativeLibrary.LoadLibrary(CoreSysinfoLibraryName);
            //CoreTimezoneHandle           = PlatformApi.NativeLibrary.LoadLibrary(CoreTimezoneLibraryName);
            //CoreUtilHandle               = PlatformApi.NativeLibrary.LoadLibrary(CoreUtilLibraryName);

            if(GetModuleHandleW(CrtConioLibraryName) == 0)
            {
                CrtConioHandle = PlatformApi.NativeLibrary.LoadLibrary(CrtConioLibraryName);
            }

            if(GetModuleHandleW(CrtConioLibraryName) == 0)
            {
                CrtConvertHandle = PlatformApi.NativeLibrary.LoadLibrary(CrtConvertLibraryName);
            }

            if(GetModuleHandleW(CrtConioLibraryName) == 0)
            {
                CrtEnvironmentHandle = PlatformApi.NativeLibrary.LoadLibrary(CrtEnvironmentLibraryName);
            }

            if(GetModuleHandleW(CrtConioLibraryName) == 0)
            {
                CrtFilesystemHandle = PlatformApi.NativeLibrary.LoadLibrary(CrtFilesystemLibraryName);
            }

            if(GetModuleHandleW(CrtConioLibraryName) == 0)
            {
                CrtHeapHandle = PlatformApi.NativeLibrary.LoadLibrary(CrtHeapLibraryName);
            }

            if(GetModuleHandleW(CrtConioLibraryName) == 0)
            {
                CrtLocaleHandle = PlatformApi.NativeLibrary.LoadLibrary(CrtLocaleLibraryName);
            }

            if(GetModuleHandleW(CrtConioLibraryName) == 0)
            {
                CrtMathHandle = PlatformApi.NativeLibrary.LoadLibrary(CrtMathLibraryName);
            }

            if(GetModuleHandleW(CrtConioLibraryName) == 0)
            {
                CrtMultibyteHandle = PlatformApi.NativeLibrary.LoadLibrary(CrtMultibyteLibraryName);
            }

            if(GetModuleHandleW(CrtConioLibraryName) == 0)
            {
                CrtPrivateHandle = PlatformApi.NativeLibrary.LoadLibrary(CrtPrivateLibraryName);
            }

            if(GetModuleHandleW(CrtConioLibraryName) == 0)
            {
                CrtProcessHandle = PlatformApi.NativeLibrary.LoadLibrary(CrtProcessLibraryName);
            }

            if(GetModuleHandleW(CrtConioLibraryName) == 0)
            {
                CrtRuntimeHandle = PlatformApi.NativeLibrary.LoadLibrary(CrtRuntimeLibraryName);
            }

            if(GetModuleHandleW(CrtConioLibraryName) == 0)
            {
                CrtStdioHandle = PlatformApi.NativeLibrary.LoadLibrary(CrtStdioLibraryName);
            }

            if(GetModuleHandleW(CrtConioLibraryName) == 0)
            {
                CrtStringHandle = PlatformApi.NativeLibrary.LoadLibrary(CrtStringLibraryName);
            }

            if(GetModuleHandleW(CrtConioLibraryName) == 0)
            {
                CrtTimeHandle = PlatformApi.NativeLibrary.LoadLibrary(CrtTimeLibraryName);
            }

            if(GetModuleHandleW(CrtConioLibraryName) == 0)
            {
                CrtUtilityHandle = PlatformApi.NativeLibrary.LoadLibrary(CrtUtilityLibraryName);
            }

            //UcrtbaseHandle               = PlatformApi.NativeLibrary.LoadLibrary(UcrtbaseLibraryName);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static void Load()
        {
        }
    }
}
