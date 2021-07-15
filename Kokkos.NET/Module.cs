using System;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

using Kokkos;

//[module:]
[assembly: DefaultDllImportSearchPaths(DllImportSearchPath.AssemblyDirectory | DllImportSearchPath.System32)]
public static class Module
{
    [ModuleInitializer]
    internal static void Initialize()
    {
        //CreateEnvironmentVariableIfMissing("KMP_DUPLICATE_LIB_OK", "TRUE");
        //CreateEnvironmentVariableIfMissing("KMP_AFFINITY",         "granularity=thread,scatter");
        
        //CreateEnvironmentVariableIfMissing("OMP_NUM_THREADS", $"{Environment.ProcessorCount}");
        //CreateEnvironmentVariableIfMissing("OMP_THREAD_LIMIT", $"{Environment.ProcessorCount}");
        //CreateEnvironmentVariableIfMissing("OMP_SCHEDULE",    "AUTO");
        //CreateEnvironmentVariableIfMissing("OMP_PROC_BIND",   "TRUE");
        //CreateEnvironmentVariableIfMissing("OMP_PLACES",      "THREADS");
        //CreateEnvironmentVariableIfMissing("OMP_MAX_ACTIVE_LEVELS",      "100");
        //CreateEnvironmentVariableIfMissing("OMP_DYNAMIC",      "false");

        CreateEnvironmentVariableIfMissing("CUDA_VISIBLE_DEVICES",                                     "0"); //A comma-separated sequence of GPU identifiers
        CreateEnvironmentVariableIfMissing("CUDA_MANAGED_FORCE_DEVICE_ALLOC",                          "1"); //0 or 1 (default is 0)
        CreateEnvironmentVariableIfMissing("CUDA_AUTO_BOOST",                                          "1"); //0 or 1
        CreateEnvironmentVariableIfMissing("CUDA_LAUNCH_BLOCKING",                                     "1"); //0 or 1 (default is 0)
        //CreateEnvironmentVariableIfMissing("CUDA_DEVICE_ORDER",                                        "FASTEST_FIRST"); //FASTEST_FIRST, PCI_BUS_ID, (default is FASTEST_FIRST)
        //CreateEnvironmentVariableIfMissing("CUDA_CACHE_DISABLE",                                       "0"); //0 or 1 (default is 0)
        //CreateEnvironmentVariableIfMissing("CUDA_CACHE_PATH",                                          "%TEMP%\\NVIDIA\\ComputeCache"); //filepath
        //CreateEnvironmentVariableIfMissing("CUDA_CACHE_MAXSIZE",                                       "1073741824"); //integer (default is 268,435,456 (256 MiB) and maximum is 4,294,967,296 (4 GiB))
        //CreateEnvironmentVariableIfMissing("CUDA_FORCE_PTX_JIT",                                       "0"); //0 or 1 (default is 0)
        //CreateEnvironmentVariableIfMissing("CUDA_DISABLE_PTX_JIT",                                     "1"); //0 or 1 (default is 0)
        //CreateEnvironmentVariableIfMissing("CUDA_DEVICE_MAX_CONNECTIONS",                              "32"); //1 to 32 (default is 8)
        //CreateEnvironmentVariableIfMissing("CUDA_DEVICE_WAITS_ON_EXCEPTION",                           "1"); //0 or 1 (default is 0)
        //CreateEnvironmentVariableIfMissing("CUDA_DEVICE_DEFAULT_PERSISTING_L2_CACHE_PERCENTAGE_LIMIT", "50"); //Percentage value (between 0 - 100, default is 0)

        Trilinos.UcrtLibraries.Load();
        Trilinos.MiscellaneousLibraries.Load();
        Trilinos.TrilinosLibraries.Load();
        KokkosLibrary.Load();
    }

    internal static bool EnvironmentVariableExist(string variable)
    {
        return Environment.GetEnvironmentVariable(variable) is not null;
    }

    internal static void CreateEnvironmentVariableIfMissing(string variable, string value)
    {
        if(!EnvironmentVariableExist(variable))
        {
            Environment.SetEnvironmentVariable(variable, value);
        }
    }
}
