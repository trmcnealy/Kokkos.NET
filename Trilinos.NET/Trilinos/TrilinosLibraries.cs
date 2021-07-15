// ReSharper disable InconsistentNaming
// ReSharper disable UnusedMember.Local

using System.Runtime.CompilerServices;
using System.Runtime.Versioning;

namespace Trilinos
{
    [NonVersionable]
    public static class TrilinosLibraries
    {
        public const string libexodusLibraryName = "libexodus.dll";
        public const string libintrepid2LibraryName = "libintrepid2.dll";
        public const string libIoexLibraryName = "libIoex.dll";
        public const string libIognLibraryName = "libIogn.dll";
        public const string libIogsLibraryName = "libIogs.dll";
        public const string libIohbLibraryName = "libIohb.dll";
        public const string libIonitLibraryName = "libIonit.dll";
        public const string libIossLibraryName = "libIoss.dll";
        public const string libIotrLibraryName = "libIotr.dll";
        public const string libIovsLibraryName = "libIovs.dll";
        public const string libio_info_libLibraryName = "libio_info_lib.dll";
        public const string libkokkosalgorithmsLibraryName = "libkokkosalgorithms.dll";
        public const string libkokkoscontainersLibraryName = "libkokkoscontainers.dll";
        public const string libkokkoscoreLibraryName = "libkokkoscore.dll";
        public const string libkokkoskernelsLibraryName = "libkokkoskernels.dll";
        public const string libsacadoLibraryName = "libsacado.dll";
        public const string libshardsLibraryName = "libshards.dll";
        public const string libstk_mesh_baseLibraryName = "libstk_mesh_base.dll";
        public const string libstk_ngp_testLibraryName = "libstk_ngp_test.dll";
        public const string libstk_topologyLibraryName = "libstk_topology.dll";
        public const string libstk_util_command_lineLibraryName = "libstk_util_command_line.dll";
        public const string libstk_util_diagLibraryName = "libstk_util_diag.dll";
        public const string libstk_util_envLibraryName = "libstk_util_env.dll";
        public const string libstk_util_parallelLibraryName = "libstk_util_parallel.dll";
        public const string libstk_util_registryLibraryName = "libstk_util_registry.dll";
        public const string libstk_util_utilLibraryName = "libstk_util_util.dll";
        public const string libteuchoscommLibraryName = "libteuchoscomm.dll";
        public const string libteuchoscoreLibraryName = "libteuchoscore.dll";
        public const string libteuchoskokkoscommLibraryName = "libteuchoskokkoscomm.dll";
        public const string libteuchoskokkoscompatLibraryName = "libteuchoskokkoscompat.dll";
        public const string libteuchosnumericsLibraryName = "libteuchosnumerics.dll";
        public const string libteuchosparameterlistLibraryName = "libteuchosparameterlist.dll";
        public const string libteuchosparserLibraryName = "libteuchosparser.dll";
        public const string libteuchosremainderLibraryName = "libteuchosremainder.dll";


        public static readonly nint libexodusHandle;
        public static readonly nint libintrepid2Handle;
        public static readonly nint libIoexHandle;
        public static readonly nint libIognHandle;
        public static readonly nint libIogsHandle;
        public static readonly nint libIohbHandle;
        public static readonly nint libIonitHandle;
        public static readonly nint libIossHandle;
        public static readonly nint libIotrHandle;
        public static readonly nint libIovsHandle;
        public static readonly nint libio_info_libHandle;
        public static readonly nint libkokkosalgorithmsHandle;
        public static readonly nint libkokkoscontainersHandle;
        public static readonly nint libkokkoscoreHandle;
        public static readonly nint libkokkoskernelsHandle;
        public static readonly nint libsacadoHandle;
        public static readonly nint libshardsHandle;
        public static readonly nint libstk_mesh_baseHandle;
        public static readonly nint libstk_ngp_testHandle;
        public static readonly nint libstk_topologyHandle;
        public static readonly nint libstk_util_command_lineHandle;
        public static readonly nint libstk_util_diagHandle;
        public static readonly nint libstk_util_envHandle;
        public static readonly nint libstk_util_parallelHandle;
        public static readonly nint libstk_util_registryHandle;
        public static readonly nint libstk_util_utilHandle;
        public static readonly nint libteuchoscommHandle;
        public static readonly nint libteuchoscoreHandle;
        public static readonly nint libteuchoskokkoscommHandle;
        public static readonly nint libteuchoskokkoscompatHandle;
        public static readonly nint libteuchosnumericsHandle;
        public static readonly nint libteuchosparameterlistHandle;
        public static readonly nint libteuchosparserHandle;
        public static readonly nint libteuchosremainderHandle;

        public static volatile bool IsLoaded;

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        static TrilinosLibraries()
        {
            libkokkoscoreHandle       = PlatformApi.NativeLibrary.LoadLibrary(libkokkoscoreLibraryName);
            libkokkoscontainersHandle = PlatformApi.NativeLibrary.LoadLibrary(libkokkoscontainersLibraryName);
            libkokkosalgorithmsHandle = PlatformApi.NativeLibrary.LoadLibrary(libkokkosalgorithmsLibraryName);
            libkokkoskernelsHandle    = PlatformApi.NativeLibrary.LoadLibrary(libkokkoskernelsLibraryName);

            libexodusHandle           = PlatformApi.NativeLibrary.LoadLibrary(libexodusLibraryName);

            libshardsHandle = PlatformApi.NativeLibrary.LoadLibrary(libshardsLibraryName);

            libsacadoHandle = PlatformApi.NativeLibrary.LoadLibrary(libsacadoLibraryName);

            libteuchoscoreHandle          = PlatformApi.NativeLibrary.LoadLibrary(libteuchoscoreLibraryName);
            libteuchosparserHandle        = PlatformApi.NativeLibrary.LoadLibrary(libteuchosparserLibraryName);
            libteuchosnumericsHandle      = PlatformApi.NativeLibrary.LoadLibrary(libteuchosnumericsLibraryName);
            libteuchosremainderHandle     = PlatformApi.NativeLibrary.LoadLibrary(libteuchosremainderLibraryName);
            libteuchoskokkoscommHandle    = PlatformApi.NativeLibrary.LoadLibrary(libteuchoskokkoscommLibraryName);
            libteuchoskokkoscompatHandle  = PlatformApi.NativeLibrary.LoadLibrary(libteuchoskokkoscompatLibraryName);
            libteuchosparameterlistHandle = PlatformApi.NativeLibrary.LoadLibrary(libteuchosparameterlistLibraryName);
            libteuchoscommHandle          = PlatformApi.NativeLibrary.LoadLibrary(libteuchoscommLibraryName);

            libIossHandle        = PlatformApi.NativeLibrary.LoadLibrary(libIossLibraryName);
            libIoexHandle        = PlatformApi.NativeLibrary.LoadLibrary(libIoexLibraryName);
            libio_info_libHandle = PlatformApi.NativeLibrary.LoadLibrary(libio_info_libLibraryName);
            libIohbHandle        = PlatformApi.NativeLibrary.LoadLibrary(libIohbLibraryName);
            libIognHandle        = PlatformApi.NativeLibrary.LoadLibrary(libIognLibraryName);
            libIovsHandle        = PlatformApi.NativeLibrary.LoadLibrary(libIovsLibraryName);
            libIogsHandle        = PlatformApi.NativeLibrary.LoadLibrary(libIogsLibraryName);
            libIotrHandle        = PlatformApi.NativeLibrary.LoadLibrary(libIotrLibraryName);
            libIonitHandle       = PlatformApi.NativeLibrary.LoadLibrary(libIonitLibraryName);

            libstk_ngp_testHandle          = PlatformApi.NativeLibrary.LoadLibrary(libstk_ngp_testLibraryName);
            libstk_util_utilHandle         = PlatformApi.NativeLibrary.LoadLibrary(libstk_util_utilLibraryName);
            libstk_topologyHandle          = PlatformApi.NativeLibrary.LoadLibrary(libstk_topologyLibraryName);
            libstk_util_parallelHandle     = PlatformApi.NativeLibrary.LoadLibrary(libstk_util_parallelLibraryName);
            libstk_mesh_baseHandle         = PlatformApi.NativeLibrary.LoadLibrary(libstk_mesh_baseLibraryName);
            libstk_util_envHandle          = PlatformApi.NativeLibrary.LoadLibrary(libstk_util_envLibraryName);
            libstk_util_diagHandle         = PlatformApi.NativeLibrary.LoadLibrary(libstk_util_diagLibraryName);
            libstk_util_registryHandle     = PlatformApi.NativeLibrary.LoadLibrary(libstk_util_registryLibraryName);
            libstk_util_command_lineHandle = PlatformApi.NativeLibrary.LoadLibrary(libstk_util_command_lineLibraryName);

            libintrepid2Handle            = PlatformApi.NativeLibrary.LoadLibrary(libintrepid2LibraryName);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static void Load()
        {
            if(!IsLoaded)
            {
                IsLoaded = true;
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static void Unload()
        {
            IsLoaded = false;
        }
    }
}
