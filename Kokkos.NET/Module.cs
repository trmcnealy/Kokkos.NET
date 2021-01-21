using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

using Kokkos;

[assembly: DefaultDllImportSearchPaths(DllImportSearchPath.AssemblyDirectory | DllImportSearchPath.System32)]
public static class Module
{
    [ModuleInitializer]
    internal static void Initialize()
    {
        KokkosLibrary.Load();
    }
}