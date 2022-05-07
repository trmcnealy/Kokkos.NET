// ReSharper disable InconsistentNaming
// ReSharper disable UnusedMember.Local

using System;
using System.Collections;
using System.Collections.Generic;
using System.Reflection.Emit;
using System.Runtime.CompilerServices;
using System.Runtime.Versioning;

namespace Trilinos
{
    [NonVersionable]
    public class TrilinosLibrary
    {
        private readonly string _libraryName;
        private readonly BitSet _flag;

        public string LibraryName
        {
            get { return _libraryName + ".dll"; }
        }

        public nint Handle;

        public TrilinosLibrary[] Dependents;

        public TrilinosLibrary(string libraryName, BitSet flag)
        {
            _libraryName = libraryName;
            _flag = flag;
            Dependents = Array.Empty<TrilinosLibrary>();
        }

        public bool HasFlag(BitSet flag)
        {
            return flag.Intersects(_flag);
        }

        public void AddDependents(params TrilinosLibrary[] dependents)
        {
            Dependents = dependents;
        }

        public bool Load()
        {
            if (Handle == 0 && LoadDependents())
            {
                Handle = PlatformApi.NativeLibrary.LoadLibrary(LibraryName);
            }

            return Handle != 0;
        }

        public bool LoadDependents()
        {
            bool dependentsLoaded = true;

            foreach (TrilinosLibrary library in Dependents)
            {
                dependentsLoaded |= library.Load();
            }

            return dependentsLoaded;
        }
    }


    public static class TrilinosEnablePackageFlags
    {
        public static readonly BitSet None = new BitSet(93);
        public static readonly BitSet Trilinosss = new BitSet(93).Set(1) << 0;
        public static readonly BitSet Kokkoscore = new BitSet(93).Set(1) << 1;
        public static readonly BitSet Teuchoscore = new BitSet(93).Set(1) << 2;
        public static readonly BitSet Teuchosparser = new BitSet(93).Set(1) << 3;
        public static readonly BitSet Teuchosparameterlist = new BitSet(93).Set(1) << 4;
        public static readonly BitSet Teuchoscomm = new BitSet(93).Set(1) << 5;
        public static readonly BitSet Epetra = new BitSet(93).Set(1) << 6;
        public static readonly BitSet Epetraext = new BitSet(93).Set(1) << 7;
        public static readonly BitSet Amesos = new BitSet(93).Set(1) << 8;
        public static readonly BitSet Tpetra = new BitSet(93).Set(1) << 9;
        public static readonly BitSet Tpetraext = new BitSet(93).Set(1) << 10;
        public static readonly BitSet Teuchoskokkoscompat = new BitSet(93).Set(1) << 11;
        public static readonly BitSet Teuchosremainder = new BitSet(93).Set(1) << 12;
        public static readonly BitSet Teuchosnumerics = new BitSet(93).Set(1) << 13;
        public static readonly BitSet Amesos2 = new BitSet(93).Set(1) << 14;
        public static readonly BitSet Anasazi = new BitSet(93).Set(1) << 15;
        public static readonly BitSet Anasaziepetra = new BitSet(93).Set(1) << 16;
        public static readonly BitSet Belos = new BitSet(93).Set(1) << 17;
        public static readonly BitSet Thyratpetra = new BitSet(93).Set(1) << 18;
        public static readonly BitSet Xpetra = new BitSet(93).Set(1) << 19;
        public static readonly BitSet Xpetra_sup = new BitSet(93).Set(1) << 20;
        public static readonly BitSet Belosxpetra = new BitSet(93).Set(1) << 21;
        public static readonly BitSet Anasazitpetra = new BitSet(93).Set(1) << 22;
        public static readonly BitSet Aztecoo = new BitSet(93).Set(1) << 23;
        public static readonly BitSet Belosepetra = new BitSet(93).Set(1) << 24;
        public static readonly BitSet Belostpetra = new BitSet(93).Set(1) << 25;
        public static readonly BitSet Exodus = new BitSet(93).Set(1) << 26;
        public static readonly BitSet Gtest = new BitSet(93).Set(1) << 27;
        public static readonly BitSet Ifpack = new BitSet(93).Set(1) << 28;
        public static readonly BitSet Ifpack2_adapters = new BitSet(93).Set(1) << 29;
        public static readonly BitSet Zoltan2 = new BitSet(93).Set(1) << 30;
        public static readonly BitSet Kokkoscontainers = new BitSet(93).Set(1) << 31;
        public static readonly BitSet Ifpack2 = new BitSet(93).Set(1) << 32;
        public static readonly BitSet Shards = new BitSet(93).Set(1) << 33;
        public static readonly BitSet Intrepid2 = new BitSet(93).Set(1) << 34;
        public static readonly BitSet Zoltan = new BitSet(93).Set(1) << 35;
        public static readonly BitSet Ioss = new BitSet(93).Set(1) << 36;
        public static readonly BitSet Ioex = new BitSet(93).Set(1) << 37;
        public static readonly BitSet Iogn = new BitSet(93).Set(1) << 38;
        public static readonly BitSet Iogs = new BitSet(93).Set(1) << 39;
        public static readonly BitSet Iohb = new BitSet(93).Set(1) << 40;
        public static readonly BitSet Iotr = new BitSet(93).Set(1) << 41;
        public static readonly BitSet Iovs = new BitSet(93).Set(1) << 42;
        public static readonly BitSet Ionit = new BitSet(93).Set(1) << 43;
        public static readonly BitSet Io_info_lib = new BitSet(93).Set(1) << 44;
        public static readonly BitSet Kokkosalgorithms = new BitSet(93).Set(1) << 45;
        public static readonly BitSet Kokkoskernels = new BitSet(93).Set(1) << 46;
        public static readonly BitSet Kokkostsqr = new BitSet(93).Set(1) << 47;
        public static readonly BitSet Thyracore = new BitSet(93).Set(1) << 48;
        public static readonly BitSet Nox = new BitSet(93).Set(1) << 49;
        public static readonly BitSet Loca = new BitSet(93).Set(1) << 50;
        public static readonly BitSet Ml = new BitSet(93).Set(1) << 51;
        public static readonly BitSet Rtop = new BitSet(93).Set(1) << 52;
        public static readonly BitSet Thyraepetra = new BitSet(93).Set(1) << 53;
        public static readonly BitSet Stratimikosifpack = new BitSet(93).Set(1) << 54;
        public static readonly BitSet Stratimikosml = new BitSet(93).Set(1) << 55;
        public static readonly BitSet Stratimikosamesos = new BitSet(93).Set(1) << 56;
        public static readonly BitSet Stratimikosaztecoo = new BitSet(93).Set(1) << 57;
        public static readonly BitSet Stratimikosamesos2 = new BitSet(93).Set(1) << 58;
        public static readonly BitSet Stratimikosbelos = new BitSet(93).Set(1) << 59;
        public static readonly BitSet Stratimikos = new BitSet(93).Set(1) << 60;
        public static readonly BitSet Thyraepetraext = new BitSet(93).Set(1) << 61;
        public static readonly BitSet Teko = new BitSet(93).Set(1) << 62;
        public static readonly BitSet Noxepetra = new BitSet(93).Set(1) << 63;
        public static readonly BitSet Locaepetra = new BitSet(93).Set(1) << 64;
        public static readonly BitSet Noxlapack = new BitSet(93).Set(1) << 65;
        public static readonly BitSet Localapack = new BitSet(93).Set(1) << 66;
        public static readonly BitSet Locathyra = new BitSet(93).Set(1) << 67;
        public static readonly BitSet ModeLaplace = new BitSet(93).Set(1) << 68;
        public static readonly BitSet Muelu = new BitSet(93).Set(1) << 69;
        public static readonly BitSet Muelu_adapters = new BitSet(93).Set(1) << 70;
        public static readonly BitSet Muelu_interface = new BitSet(93).Set(1) << 71;
        public static readonly BitSet Piro = new BitSet(93).Set(1) << 72;
        public static readonly BitSet Rythmos = new BitSet(93).Set(1) << 73;
        public static readonly BitSet Sacado = new BitSet(93).Set(1) << 74;
        public static readonly BitSet Shylu_nodehts = new BitSet(93).Set(1) << 75;
        public static readonly BitSet Stk_topology = new BitSet(93).Set(1) << 76;
        public static readonly BitSet Stk_util_util = new BitSet(93).Set(1) << 77;
        public static readonly BitSet Stk_util_parallel = new BitSet(93).Set(1) << 78;
        public static readonly BitSet Stk_mesh_base = new BitSet(93).Set(1) << 79;
        public static readonly BitSet Stk_util_env = new BitSet(93).Set(1) << 80;
        public static readonly BitSet Stk_util_diag = new BitSet(93).Set(1) << 81;
        public static readonly BitSet Stk_io = new BitSet(93).Set(1) << 82;
        public static readonly BitSet Stk_io_util = new BitSet(93).Set(1) << 83;
        public static readonly BitSet Stk_ngp_test = new BitSet(93).Set(1) << 84;
        public static readonly BitSet Stk_util_registry = new BitSet(93).Set(1) << 85;
        public static readonly BitSet Stk_util_command_line = new BitSet(93).Set(1) << 86;
        public static readonly BitSet Tempus = new BitSet(93).Set(1) << 87;
        public static readonly BitSet Teuchoskokkoscomm = new BitSet(93).Set(1) << 88;
        public static readonly BitSet Tpetraclassic = new BitSet(93).Set(1) << 89;
        public static readonly BitSet Tpetraclassiclinalg = new BitSet(93).Set(1) << 90;
        public static readonly BitSet Tpetraclassicnodeapi = new BitSet(93).Set(1) << 91;
        public static readonly BitSet Tpetrainout = new BitSet(93).Set(1) << 92;
        public static readonly BitSet All = Trilinosss | Kokkoscore | Teuchoscore | Teuchosparser | Teuchosparameterlist | Teuchoscomm | Epetra | Epetraext | Amesos | Tpetra | Tpetraext | Teuchoskokkoscompat | Teuchosremainder | Teuchosnumerics | Amesos2 | Anasazi | Anasaziepetra | Belos | Thyratpetra | Xpetra | Xpetra_sup | Belosxpetra | Anasazitpetra | Aztecoo | Belosepetra | Belostpetra | Exodus | Gtest | Ifpack | Ifpack2_adapters | Zoltan2 | Kokkoscontainers | Ifpack2 | Shards | Intrepid2 | Zoltan | Ioss | Ioex | Iogn | Iogs | Iohb | Iotr | Iovs | Ionit | Io_info_lib | Kokkosalgorithms | Kokkoskernels | Kokkostsqr | Thyracore | Nox | Loca | Ml | Rtop | Thyraepetra | Stratimikosifpack | Stratimikosml | Stratimikosamesos | Stratimikosaztecoo | Stratimikosamesos2 | Stratimikosbelos | Stratimikos | Thyraepetraext | Teko | Noxepetra | Locaepetra | Noxlapack | Localapack | Locathyra | ModeLaplace | Muelu | Muelu_adapters | Muelu_interface | Piro | Rythmos | Sacado | Shylu_nodehts | Stk_topology | Stk_util_util | Stk_util_parallel | Stk_mesh_base | Stk_util_env | Stk_util_diag | Stk_io | Stk_io_util | Stk_ngp_test | Stk_util_registry | Stk_util_command_line | Tempus | Teuchoskokkoscomm | Tpetraclassic | Tpetraclassiclinalg | Tpetraclassicnodeapi | Tpetrainout;
    }

    [NonVersionable]
    public static class TrilinosLibraries
    {
        public static readonly TrilinosLibrary libamesos;
        public static readonly TrilinosLibrary libamesos2;
        public static readonly TrilinosLibrary libanasazi;
        public static readonly TrilinosLibrary libanasaziepetra;
        public static readonly TrilinosLibrary libanasazitpetra;
        public static readonly TrilinosLibrary libaztecoo;
        public static readonly TrilinosLibrary libbelos;
        public static readonly TrilinosLibrary libbelosepetra;
        public static readonly TrilinosLibrary libbelostpetra;
        public static readonly TrilinosLibrary libbelosxpetra;
        public static readonly TrilinosLibrary libepetra;
        public static readonly TrilinosLibrary libepetraext;
        public static readonly TrilinosLibrary libexodus;
        public static readonly TrilinosLibrary libgtest;
        public static readonly TrilinosLibrary libifpack;
        public static readonly TrilinosLibrary libifpack2_adapters;
        public static readonly TrilinosLibrary libifpack2;
        public static readonly TrilinosLibrary libintrepid2;
        public static readonly TrilinosLibrary libIoex;
        public static readonly TrilinosLibrary libIogn;
        public static readonly TrilinosLibrary libIogs;
        public static readonly TrilinosLibrary libIohb;
        public static readonly TrilinosLibrary libIonit;
        public static readonly TrilinosLibrary libIoss;
        public static readonly TrilinosLibrary libIotr;
        public static readonly TrilinosLibrary libIovs;
        public static readonly TrilinosLibrary libio_info_lib;
        public static readonly TrilinosLibrary libkokkosalgorithms;
        public static readonly TrilinosLibrary libkokkoscontainers;
        public static readonly TrilinosLibrary libkokkoscore;
        public static readonly TrilinosLibrary libkokkoskernels;
        public static readonly TrilinosLibrary libkokkostsqr;
        public static readonly TrilinosLibrary libloca;
        public static readonly TrilinosLibrary liblocaepetra;
        public static readonly TrilinosLibrary liblocalapack;
        public static readonly TrilinosLibrary liblocathyra;
        public static readonly TrilinosLibrary libml;
        public static readonly TrilinosLibrary libModeLaplace;
        public static readonly TrilinosLibrary libmuelu_adapters;
        public static readonly TrilinosLibrary libmuelu_interface;
        public static readonly TrilinosLibrary libmuelu;
        public static readonly TrilinosLibrary libnox;
        public static readonly TrilinosLibrary libnoxepetra;
        public static readonly TrilinosLibrary libnoxlapack;
        public static readonly TrilinosLibrary libpiro;
        public static readonly TrilinosLibrary librtop;
        public static readonly TrilinosLibrary librythmos;
        public static readonly TrilinosLibrary libsacado;
        public static readonly TrilinosLibrary libshards;
        public static readonly TrilinosLibrary libshylu_nodehts;
        public static readonly TrilinosLibrary libstk_io;
        public static readonly TrilinosLibrary libstk_io_util;
        public static readonly TrilinosLibrary libstk_mesh_base;
        public static readonly TrilinosLibrary libstk_ngp_test;
        public static readonly TrilinosLibrary libstk_topology;
        public static readonly TrilinosLibrary libstk_util_command_line;
        public static readonly TrilinosLibrary libstk_util_diag;
        public static readonly TrilinosLibrary libstk_util_env;
        public static readonly TrilinosLibrary libstk_util_parallel;
        public static readonly TrilinosLibrary libstk_util_registry;
        public static readonly TrilinosLibrary libstk_util_util;
        public static readonly TrilinosLibrary libstratimikos;
        public static readonly TrilinosLibrary libstratimikosamesos;
        public static readonly TrilinosLibrary libstratimikosamesos2;
        public static readonly TrilinosLibrary libstratimikosaztecoo;
        public static readonly TrilinosLibrary libstratimikosbelos;
        public static readonly TrilinosLibrary libstratimikosifpack;
        public static readonly TrilinosLibrary libstratimikosml;
        public static readonly TrilinosLibrary libteko;
        public static readonly TrilinosLibrary libtempus;
        public static readonly TrilinosLibrary libteuchoscomm;
        public static readonly TrilinosLibrary libteuchoscore;
        public static readonly TrilinosLibrary libteuchoskokkoscomm;
        public static readonly TrilinosLibrary libteuchoskokkoscompat;
        public static readonly TrilinosLibrary libteuchosnumerics;
        public static readonly TrilinosLibrary libteuchosparameterlist;
        public static readonly TrilinosLibrary libteuchosparser;
        public static readonly TrilinosLibrary libteuchosremainder;
        public static readonly TrilinosLibrary libthyracore;
        public static readonly TrilinosLibrary libthyraepetra;
        public static readonly TrilinosLibrary libthyraepetraext;
        public static readonly TrilinosLibrary libthyratpetra;
        public static readonly TrilinosLibrary libtpetra;
        public static readonly TrilinosLibrary libtpetraclassic;
        public static readonly TrilinosLibrary libtpetraclassiclinalg;
        public static readonly TrilinosLibrary libtpetraclassicnodeapi;
        public static readonly TrilinosLibrary libtpetraext;
        public static readonly TrilinosLibrary libtpetrainout;
        public static readonly TrilinosLibrary libtrilinosss;
        public static readonly TrilinosLibrary libxpetra_sup;
        public static readonly TrilinosLibrary libxpetra;
        public static readonly TrilinosLibrary libzoltan;
        public static readonly TrilinosLibrary libzoltan2;

        public static readonly TrilinosLibrary[] Libraries;

        public static volatile bool IsLoaded;


        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        private static void DependentsSort(TrilinosLibrary v, Dictionary<TrilinosLibrary, bool> visited, Stack<TrilinosLibrary> stack)
        {
            visited[v] = true;

            foreach (TrilinosLibrary vertex in v.Dependents)
            {
                if (!visited[vertex])
                {
                    DependentsSort(vertex, visited, stack);
                }
            }

            stack.Push(v);
        }


        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        private static Stack<TrilinosLibrary> DependentsSort()
        {
            Stack<TrilinosLibrary> stack = new();

            Dictionary<TrilinosLibrary, bool> visited = new(Libraries.Length);

            foreach (TrilinosLibrary library in Libraries)
            {
                visited.Add(library, false);
            }

            for (int i = 0; i < Libraries.Length; i++)
            {
                if (visited[Libraries[i]] == false)
                {
                    DependentsSort(Libraries[i], visited, stack);
                }
            }

            return stack;
        }


        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        static TrilinosLibraries()
        {
            libamesos = new TrilinosLibrary("amesos", TrilinosEnablePackageFlags.Amesos);
            libamesos2 = new TrilinosLibrary("amesos2", TrilinosEnablePackageFlags.Amesos2);
            libanasazi = new TrilinosLibrary("anasazi", TrilinosEnablePackageFlags.Anasazi);
            libanasaziepetra = new TrilinosLibrary("anasaziepetra", TrilinosEnablePackageFlags.Anasaziepetra);
            libanasazitpetra = new TrilinosLibrary("anasazitpetra", TrilinosEnablePackageFlags.Anasazitpetra);
            libaztecoo = new TrilinosLibrary("aztecoo", TrilinosEnablePackageFlags.Aztecoo);
            libbelos = new TrilinosLibrary("belos", TrilinosEnablePackageFlags.Belos);
            libbelosepetra = new TrilinosLibrary("belosepetra", TrilinosEnablePackageFlags.Belosepetra);
            libbelostpetra = new TrilinosLibrary("belostpetra", TrilinosEnablePackageFlags.Belostpetra);
            libbelosxpetra = new TrilinosLibrary("belosxpetra", TrilinosEnablePackageFlags.Belosxpetra);
            libepetra = new TrilinosLibrary("epetra", TrilinosEnablePackageFlags.Epetra);
            libepetraext = new TrilinosLibrary("epetraext", TrilinosEnablePackageFlags.Epetraext);
            libexodus = new TrilinosLibrary("exodus", TrilinosEnablePackageFlags.Exodus);
            libgtest = new TrilinosLibrary("gtest", TrilinosEnablePackageFlags.Gtest);
            libifpack = new TrilinosLibrary("ifpack", TrilinosEnablePackageFlags.Ifpack);
            libifpack2_adapters = new TrilinosLibrary("ifpack2-adapters", TrilinosEnablePackageFlags.Ifpack2_adapters);
            libifpack2 = new TrilinosLibrary("ifpack2", TrilinosEnablePackageFlags.Ifpack2);
            libintrepid2 = new TrilinosLibrary("intrepid2", TrilinosEnablePackageFlags.Intrepid2);
            libIoex = new TrilinosLibrary("Ioex", TrilinosEnablePackageFlags.Ioex);
            libIogn = new TrilinosLibrary("Iogn", TrilinosEnablePackageFlags.Iogn);
            libIogs = new TrilinosLibrary("Iogs", TrilinosEnablePackageFlags.Iogs);
            libIohb = new TrilinosLibrary("Iohb", TrilinosEnablePackageFlags.Iohb);
            libIonit = new TrilinosLibrary("Ionit", TrilinosEnablePackageFlags.Ionit);
            libIoss = new TrilinosLibrary("Ioss", TrilinosEnablePackageFlags.Ioss);
            libIotr = new TrilinosLibrary("Iotr", TrilinosEnablePackageFlags.Iotr);
            libIovs = new TrilinosLibrary("Iovs", TrilinosEnablePackageFlags.Iovs);
            libio_info_lib = new TrilinosLibrary("io_info_lib", TrilinosEnablePackageFlags.Io_info_lib);
            libkokkosalgorithms = new TrilinosLibrary("kokkosalgorithms", TrilinosEnablePackageFlags.Kokkosalgorithms);
            libkokkoscontainers = new TrilinosLibrary("kokkoscontainers", TrilinosEnablePackageFlags.Kokkoscontainers);
            libkokkoscore = new TrilinosLibrary("kokkoscore", TrilinosEnablePackageFlags.Kokkoscore);
            libkokkoskernels = new TrilinosLibrary("kokkoskernels", TrilinosEnablePackageFlags.Kokkoskernels);
            libkokkostsqr = new TrilinosLibrary("kokkostsqr", TrilinosEnablePackageFlags.Kokkostsqr);
            libloca = new TrilinosLibrary("loca", TrilinosEnablePackageFlags.Loca);
            liblocaepetra = new TrilinosLibrary("locaepetra", TrilinosEnablePackageFlags.Locaepetra);
            liblocalapack = new TrilinosLibrary("localapack", TrilinosEnablePackageFlags.Localapack);
            liblocathyra = new TrilinosLibrary("locathyra", TrilinosEnablePackageFlags.Locathyra);
            libml = new TrilinosLibrary("ml", TrilinosEnablePackageFlags.Ml);
            libModeLaplace = new TrilinosLibrary("ModeLaplace", TrilinosEnablePackageFlags.ModeLaplace);
            libmuelu_adapters = new TrilinosLibrary("muelu-adapters", TrilinosEnablePackageFlags.Muelu_adapters);
            libmuelu_interface = new TrilinosLibrary("muelu-interface", TrilinosEnablePackageFlags.Muelu_interface);
            libmuelu = new TrilinosLibrary("muelu", TrilinosEnablePackageFlags.Muelu);
            libnox = new TrilinosLibrary("nox", TrilinosEnablePackageFlags.Nox);
            libnoxepetra = new TrilinosLibrary("noxepetra", TrilinosEnablePackageFlags.Noxepetra);
            libnoxlapack = new TrilinosLibrary("noxlapack", TrilinosEnablePackageFlags.Noxlapack);
            libpiro = new TrilinosLibrary("piro", TrilinosEnablePackageFlags.Piro);
            librtop = new TrilinosLibrary("rtop", TrilinosEnablePackageFlags.Rtop);
            librythmos = new TrilinosLibrary("rythmos", TrilinosEnablePackageFlags.Rythmos);
            libsacado = new TrilinosLibrary("sacado", TrilinosEnablePackageFlags.Sacado);
            libshards = new TrilinosLibrary("shards", TrilinosEnablePackageFlags.Shards);
            libshylu_nodehts = new TrilinosLibrary("shylu_nodehts", TrilinosEnablePackageFlags.Shylu_nodehts);
            libstk_io = new TrilinosLibrary("stk_io", TrilinosEnablePackageFlags.Stk_io);
            libstk_io_util = new TrilinosLibrary("stk_io_util", TrilinosEnablePackageFlags.Stk_io_util);
            libstk_mesh_base = new TrilinosLibrary("stk_mesh_base", TrilinosEnablePackageFlags.Stk_mesh_base);
            libstk_ngp_test = new TrilinosLibrary("stk_ngp_test", TrilinosEnablePackageFlags.Stk_ngp_test);
            libstk_topology = new TrilinosLibrary("stk_topology", TrilinosEnablePackageFlags.Stk_topology);
            libstk_util_command_line = new TrilinosLibrary("stk_util_command_line", TrilinosEnablePackageFlags.Stk_util_command_line);
            libstk_util_diag = new TrilinosLibrary("stk_util_diag", TrilinosEnablePackageFlags.Stk_util_diag);
            libstk_util_env = new TrilinosLibrary("stk_util_env", TrilinosEnablePackageFlags.Stk_util_env);
            libstk_util_parallel = new TrilinosLibrary("stk_util_parallel", TrilinosEnablePackageFlags.Stk_util_parallel);
            libstk_util_registry = new TrilinosLibrary("stk_util_registry", TrilinosEnablePackageFlags.Stk_util_registry);
            libstk_util_util = new TrilinosLibrary("stk_util_util", TrilinosEnablePackageFlags.Stk_util_util);
            libstratimikos = new TrilinosLibrary("stratimikos", TrilinosEnablePackageFlags.Stratimikos);
            libstratimikosamesos = new TrilinosLibrary("stratimikosamesos", TrilinosEnablePackageFlags.Stratimikosamesos);
            libstratimikosamesos2 = new TrilinosLibrary("stratimikosamesos2", TrilinosEnablePackageFlags.Stratimikosamesos2);
            libstratimikosaztecoo = new TrilinosLibrary("stratimikosaztecoo", TrilinosEnablePackageFlags.Stratimikosaztecoo);
            libstratimikosbelos = new TrilinosLibrary("stratimikosbelos", TrilinosEnablePackageFlags.Stratimikosbelos);
            libstratimikosifpack = new TrilinosLibrary("stratimikosifpack", TrilinosEnablePackageFlags.Stratimikosifpack);
            libstratimikosml = new TrilinosLibrary("stratimikosml", TrilinosEnablePackageFlags.Stratimikosml);
            libteko = new TrilinosLibrary("teko", TrilinosEnablePackageFlags.Teko);
            libtempus = new TrilinosLibrary("tempus", TrilinosEnablePackageFlags.Tempus);
            libteuchoscomm = new TrilinosLibrary("teuchoscomm", TrilinosEnablePackageFlags.Teuchoscomm);
            libteuchoscore = new TrilinosLibrary("teuchoscore", TrilinosEnablePackageFlags.Teuchoscore);
            libteuchoskokkoscomm = new TrilinosLibrary("teuchoskokkoscomm", TrilinosEnablePackageFlags.Teuchoskokkoscomm);
            libteuchoskokkoscompat = new TrilinosLibrary("teuchoskokkoscompat", TrilinosEnablePackageFlags.Teuchoskokkoscompat);
            libteuchosnumerics = new TrilinosLibrary("teuchosnumerics", TrilinosEnablePackageFlags.Teuchosnumerics);
            libteuchosparameterlist = new TrilinosLibrary("teuchosparameterlist", TrilinosEnablePackageFlags.Teuchosparameterlist);
            libteuchosparser = new TrilinosLibrary("teuchosparser", TrilinosEnablePackageFlags.Teuchosparser);
            libteuchosremainder = new TrilinosLibrary("teuchosremainder", TrilinosEnablePackageFlags.Teuchosremainder);
            libthyracore = new TrilinosLibrary("thyracore", TrilinosEnablePackageFlags.Thyracore);
            libthyraepetra = new TrilinosLibrary("thyraepetra", TrilinosEnablePackageFlags.Thyraepetra);
            libthyraepetraext = new TrilinosLibrary("thyraepetraext", TrilinosEnablePackageFlags.Thyraepetraext);
            libthyratpetra = new TrilinosLibrary("thyratpetra", TrilinosEnablePackageFlags.Thyratpetra);
            libtpetra = new TrilinosLibrary("tpetra", TrilinosEnablePackageFlags.Tpetra);
            libtpetraclassic = new TrilinosLibrary("tpetraclassic", TrilinosEnablePackageFlags.Tpetraclassic);
            libtpetraclassiclinalg = new TrilinosLibrary("tpetraclassiclinalg", TrilinosEnablePackageFlags.Tpetraclassiclinalg);
            libtpetraclassicnodeapi = new TrilinosLibrary("tpetraclassicnodeapi", TrilinosEnablePackageFlags.Tpetraclassicnodeapi);
            libtpetraext = new TrilinosLibrary("tpetraext", TrilinosEnablePackageFlags.Tpetraext);
            libtpetrainout = new TrilinosLibrary("tpetrainout", TrilinosEnablePackageFlags.Tpetrainout);
            libtrilinosss = new TrilinosLibrary("trilinosss", TrilinosEnablePackageFlags.Trilinosss);
            libxpetra_sup = new TrilinosLibrary("xpetra-sup", TrilinosEnablePackageFlags.Xpetra_sup);
            libxpetra = new TrilinosLibrary("xpetra", TrilinosEnablePackageFlags.Xpetra);
            libzoltan = new TrilinosLibrary("zoltan", TrilinosEnablePackageFlags.Zoltan);
            libzoltan2 = new TrilinosLibrary("zoltan2", TrilinosEnablePackageFlags.Zoltan2);


            #region Add Dependents

            libamesos.AddDependents(libtrilinosss,
                                    libepetraext,
                                    libepetra,
                                    libteuchosparameterlist,
                                    libteuchoscore,
                                    libkokkoscore);
            libamesos2.AddDependents(libtrilinosss,
                                     libtpetraext,
                                     libtpetra,
                                     libepetra,
                                     libteuchoskokkoscompat,
                                     libteuchosremainder,
                                     libteuchosnumerics,
                                     libteuchoscomm,
                                     libteuchosparameterlist,
                                     libteuchoscore,
                                     libkokkoscore);
            libanasazi.AddDependents(libkokkoscore);
            libanasaziepetra.AddDependents(libepetra,
                                           libteuchoscore,
                                           libkokkoscore);
            libanasazitpetra.AddDependents(libteuchoscomm,
                                           libteuchoscore,
                                           libkokkoscore,
                                           libbelosxpetra);
            libaztecoo.AddDependents(libepetra,
                                     libteuchoscore,
                                     libkokkoscore);
            libbelos.AddDependents(libteuchosnumerics,
                                   libteuchoscomm,
                                   libteuchosparameterlist,
                                   libteuchoscore,
                                   libkokkoscore);
            libbelosepetra.AddDependents(libbelos,
                                         libepetra,
                                         libteuchosremainder,
                                         libteuchosnumerics,
                                         libteuchoscomm,
                                         libteuchosparameterlist,
                                         libteuchoscore,
                                         libkokkoscore);
            libbelostpetra.AddDependents(libbelos,
                                         libtpetra,
                                         libteuchoskokkoscompat,
                                         libteuchosremainder,
                                         libteuchosnumerics,
                                         libteuchoscomm,
                                         libteuchosparameterlist,
                                         libteuchoscore,
                                         libkokkoscore,
                                         libxpetra_sup);
            libbelosxpetra.AddDependents(libbelos,
                                         libxpetra_sup,
                                         libxpetra,
                                         libtpetra,
                                         libteuchoskokkoscompat,
                                         libteuchosnumerics,
                                         libteuchoscomm,
                                         libteuchosparameterlist,
                                         libteuchoscore,
                                         libkokkoscore);
            libepetra.AddDependents(libteuchoscomm,
                                    libteuchoscore,
                                    libkokkoscore);
            libepetraext.AddDependents(libepetra,
                                       libteuchoscomm,
                                       libteuchosparameterlist,
                                       libteuchoscore,
                                       libkokkoscore);
            libifpack.AddDependents(libamesos,
                                    libaztecoo,
                                    libepetraext,
                                    libepetra,
                                    libteuchosremainder,
                                    libteuchosnumerics,
                                    libteuchoscomm,
                                    libteuchosparameterlist,
                                    libteuchoscore,
                                    libtrilinosss,
                                    libkokkoscore);
            libifpack2_adapters.AddDependents(libteuchoscomm,
                                              libteuchoscore,
                                              libkokkoscore);
            libifpack2.AddDependents(libamesos2,
                                     libzoltan2,
                                     libtpetra,
                                     libkokkoscontainers,
                                     libteuchoskokkoscompat,
                                     libteuchosremainder,
                                     libteuchosnumerics,
                                     libteuchoscomm,
                                     libteuchosparameterlist,
                                     libteuchoscore,
                                     libkokkoscore);
            libintrepid2.AddDependents(libshards);
            libIoex.AddDependents(libIoss,
                                  libexodus,
                                  libkokkoscore);
            libIogn.AddDependents(libIoss,
                                  libkokkoscore);
            libIogs.AddDependents(libIoss,
                                  libkokkoscore);
            libIohb.AddDependents(libIoss,
                                  libkokkoscore);
            libIonit.AddDependents(libIoex,
                                   libIogn,
                                   libIogs,
                                   libIohb,
                                   libIotr,
                                   libIovs,
                                   libIoss,
                                   libkokkoscore);
            libIoss.AddDependents(libzoltan,
                                  libkokkoscore);
            libIotr.AddDependents(libIoss,
                                  libkokkoscore);
            libIovs.AddDependents(libIoss,
                                  libkokkoscore);
            libio_info_lib.AddDependents(libIoss,
                                         libexodus,
                                         libkokkoscore);
            libkokkosalgorithms.AddDependents();
            libkokkoscontainers.AddDependents(libkokkoscore);
            libkokkoscore.AddDependents();
            libkokkoskernels.AddDependents(libkokkoscore);
            libkokkostsqr.AddDependents(libteuchosnumerics,
                                        libteuchoscomm,
                                        libteuchoscore,
                                        libkokkoscore);
            libloca.AddDependents(libnox,
                                  libteuchosnumerics,
                                  libteuchoscomm,
                                  libteuchosparameterlist,
                                  libteuchoscore,
                                  libkokkoscore);
            liblocaepetra.AddDependents(libloca,
                                        libnoxepetra,
                                        libnox,
                                        libteko,
                                        libepetraext,
                                        libepetra,
                                        libteuchosnumerics,
                                        libteuchosparameterlist,
                                        libteuchoscore,
                                        libkokkoscore);
            liblocalapack.AddDependents(libloca,
                                        libnoxlapack,
                                        libnox,
                                        libteuchosnumerics,
                                        libteuchosparameterlist,
                                        libteuchoscore,
                                        libkokkoscore);
            liblocathyra.AddDependents(libloca,
                                       libnox,
                                       libteuchoscomm,
                                       libteuchosparameterlist,
                                       libteuchoscore,
                                       libkokkoscore);
            libml.AddDependents(libifpack,
                                libamesos,
                                libaztecoo,
                                libepetraext,
                                libzoltan,
                                libepetra,
                                libteuchosparameterlist,
                                libteuchoscore,
                                libkokkoscore);
            libModeLaplace.AddDependents(libepetra,
                                         libteuchosnumerics,
                                         libteuchoscore,
                                         libkokkoscore);
            libmuelu_adapters.AddDependents(libshards,
                                            libteuchoscomm,
                                            libteuchoscore,
                                            libteko,
                                            libmuelu,
                                            libstratimikos);
            libmuelu_interface.AddDependents(libmuelu,
                                             libshards,
                                             libteuchosparameterlist,
                                             libteuchoscore);
            libmuelu.AddDependents(libshards,
                                   libteuchoscomm,
                                   libteuchosparameterlist,
                                   libteuchoscore,
                                   libteko);
            libnox.AddDependents(libthyracore,
                                 libteuchosnumerics,
                                 libteuchoscomm,
                                 libteuchosparameterlist,
                                 libteuchoscore,
                                 libkokkoscore);
            libnoxepetra.AddDependents(libnox,
                                       libteko,
                                       libstratimikos,
                                       libml,
                                       libifpack,
                                       libamesos,
                                       libaztecoo,
                                       libthyraepetra,
                                       libepetraext,
                                       libepetra,
                                       libteuchoscomm,
                                       libteuchosparameterlist,
                                       libteuchoscore,
                                       libkokkoscore,
                                       libstratimikosml);
            libnoxlapack.AddDependents(libnox,
                                       libteuchosnumerics,
                                       libteuchoscore,
                                       libkokkoscore);
            libpiro.AddDependents(liblocathyra,
                                  liblocaepetra,
                                  libloca,
                                  libnoxepetra,
                                  libnox,
                                  libshards,
                                  libstratimikos,
                                  libthyraepetraext,
                                  libepetraext,
                                  libthyraepetra,
                                  libthyracore,
                                  libepetra,
                                  libteuchosnumerics,
                                  libteuchoscomm,
                                  libteuchosparameterlist,
                                  libteuchoscore,
                                  libmuelu_adapters);
            librtop.AddDependents(libteuchoscomm,
                                  libteuchoscore,
                                  libkokkoscore);
            librythmos.AddDependents(libteuchoscomm,
                                     libteuchoscore,
                                     libkokkoscore);
            libsacado.AddDependents(libkokkoscore);
            libstk_io.AddDependents(libstk_mesh_base,
                                    libIonit,
                                    libIoss,
                                    libteuchoscore,
                                    libstk_topology,
                                    libstk_util_diag,
                                    libstk_util_env,
                                    libstk_util_parallel,
                                    libstk_util_util,
                                    libshards);
            libstk_io_util.AddDependents(libstk_io,
                                         libstk_mesh_base,
                                         libshards,
                                         libIoss,
                                         libteuchoscore);
            libstk_mesh_base.AddDependents(libstk_topology,
                                           libstk_util_parallel,
                                           libstk_util_util,
                                           libshards,
                                           libkokkoscore);
            libstk_ngp_test.AddDependents(libkokkoscore);
            libstk_topology.AddDependents(libkokkoscore);
            libstk_util_command_line.AddDependents(libstk_util_registry,
                                                   libstk_util_env,
                                                   libstk_util_parallel,
                                                   libstk_util_util,
                                                   libkokkoscore);
            libstk_util_diag.AddDependents(libstk_util_env,
                                           libstk_util_parallel,
                                           libstk_util_util,
                                           libkokkoscore);
            libstk_util_env.AddDependents(libstk_util_parallel,
                                          libstk_util_util,
                                          libkokkoscore);
            libstk_util_parallel.AddDependents(libstk_util_util,
                                               libkokkoscore);
            libstk_util_registry.AddDependents(libstk_util_env,
                                               libkokkoscore);
            libstk_util_util.AddDependents(libkokkoscore);
            libstratimikos.AddDependents(libstratimikosifpack,
                                         libstratimikosml,
                                         libstratimikosamesos,
                                         libstratimikosaztecoo,
                                         libstratimikosamesos2,
                                         libstratimikosbelos,
                                         libamesos2,
                                         libbelos,
                                         libtrilinosss,
                                         libtpetra,
                                         libteuchoskokkoscompat,
                                         libteuchosremainder,
                                         libteuchosnumerics,
                                         libteuchoscomm,
                                         libteuchosparameterlist,
                                         libteuchoscore,
                                         libkokkoscore);
            libstratimikosamesos.AddDependents(libamesos,
                                               libthyraepetra,
                                               libepetra,
                                               libteuchosremainder,
                                               libteuchoscomm,
                                               libteuchosparameterlist,
                                               libteuchoscore,
                                               libkokkoscore);
            libstratimikosamesos2.AddDependents(libteuchosremainder,
                                                libteuchoscomm,
                                                libteuchoscore,
                                                libkokkoscore,
                                                libamesos2);
            libstratimikosaztecoo.AddDependents(libaztecoo,
                                                libthyraepetra,
                                                libepetraext,
                                                libepetra,
                                                libteuchoscomm,
                                                libteuchosparameterlist,
                                                libteuchoscore,
                                                libkokkoscore);
            libstratimikosbelos.AddDependents(libteuchoscomm,
                                              libteuchoscore,
                                              libkokkoscore,
                                              libbelosxpetra);
            libstratimikosifpack.AddDependents(libml,
                                               libifpack,
                                               libthyraepetra,
                                               libteuchoscomm,
                                               libteuchosparameterlist,
                                               libteuchoscore,
                                               libkokkoscore);
            libstratimikosml.AddDependents(libml,
                                           libthyraepetra,
                                           libteuchoscomm,
                                           libteuchosparameterlist,
                                           libteuchoscore,
                                           libkokkoscore);
            libteko.AddDependents(libstratimikos,
                                  libstratimikosaztecoo,
                                  libstratimikosml,
                                  libifpack2,
                                  libamesos2,
                                  libzoltan2,
                                  libthyratpetra,
                                  libthyraepetraext,
                                  libthyraepetra,
                                  libthyracore,
                                  librtop,
                                  libtpetra,
                                  libkokkoscontainers,
                                  libepetraext,
                                  libepetra,
                                  libteuchoskokkoscompat,
                                  libteuchosremainder,
                                  libteuchosnumerics,
                                  libteuchoscomm,
                                  libteuchosparameterlist,
                                  libteuchoscore,
                                  libkokkoscore);
            libtempus.AddDependents(libteuchoscore,
                                    libkokkoscore);
            libteuchoscomm.AddDependents(libteuchosparameterlist,
                                         libteuchoscore,
                                         libkokkoscore);
            libteuchoscore.AddDependents(libkokkoscore);
            libteuchoskokkoscompat.AddDependents(libteuchoscore,
                                                 libkokkoscore);
            libteuchosnumerics.AddDependents(libteuchoscore,
                                             libkokkoscore);
            libteuchosparameterlist.AddDependents(libteuchosparser,
                                                  libteuchoscore,
                                                  libkokkoscore);
            libteuchosparser.AddDependents(libteuchoscore,
                                           libkokkoscore);
            libteuchosremainder.AddDependents(libteuchoscore,
                                              libkokkoscore);
            libthyracore.AddDependents(libteuchoscomm,
                                       libteuchoscore,
                                       libkokkoscore);
            libthyraepetra.AddDependents(libthyracore,
                                         libepetra,
                                         librtop,
                                         libteuchosnumerics,
                                         libteuchoscomm,
                                         libteuchosparameterlist,
                                         libteuchoscore,
                                         libkokkoscore);
            libthyraepetraext.AddDependents(libthyraepetra,
                                            libthyracore,
                                            libepetraext,
                                            libepetra,
                                            libteuchoscomm,
                                            libteuchosparameterlist,
                                            libteuchoscore,
                                            libkokkoscore);
            libthyratpetra.AddDependents(libteuchoscomm,
                                         libteuchoscore,
                                         libkokkoscore,
                                         libtpetraext);
            libtpetra.AddDependents(libepetra,
                                    libteuchoscomm,
                                    libteuchosparameterlist,
                                    libteuchoscore,
                                    libkokkoscore);
            libtpetraclassicnodeapi.AddDependents(libteuchoscore,
                                                  libkokkoscore);
            libtpetraext.AddDependents(libteuchoscomm,
                                       libteuchoscore,
                                       libkokkoscore,
                                       libtpetra);
            libtpetrainout.AddDependents(libtpetra,
                                         libteuchoscomm,
                                         libteuchoscore,
                                         libkokkoscore);
            libxpetra_sup.AddDependents(libteuchoscomm,
                                        libteuchoscore,
                                        libkokkoscore,
                                        libxpetra);
            libxpetra.AddDependents(libteuchoscomm,
                                    libteuchoscore,
                                    libkokkoscore,
                                    libthyratpetra);
            libzoltan2.AddDependents(libteuchoscomm,
                                     libteuchosparameterlist,
                                     libteuchoscore,
                                     libkokkoscore,
                                     libxpetra_sup);

            #endregion

            Libraries = new TrilinosLibrary[]
            {
                libtrilinosss,
                libkokkoscore,
                libteuchoscore,
                libteuchosparser,
                libteuchosparameterlist,
                libteuchoscomm,
                libepetra,
                libepetraext,
                libamesos,
                libtpetra,
                libtpetraext,
                libteuchoskokkoscompat,
                libteuchosremainder,
                libteuchosnumerics,
                libamesos2,
                libanasazi,
                libanasaziepetra,
                libbelos,
                libthyratpetra,
                libxpetra,
                libxpetra_sup,
                libbelosxpetra,
                libanasazitpetra,
                libaztecoo,
                libbelosepetra,
                libbelostpetra,
                libexodus,
                libgtest,
                libifpack,
                libifpack2_adapters,
                libzoltan2,
                libkokkoscontainers,
                libifpack2,
                libshards,
                libintrepid2,
                libzoltan,
                libIoss,
                libIoex,
                libIogn,
                libIogs,
                libIohb,
                libIotr,
                libIovs,
                libIonit,
                libio_info_lib,
                libkokkosalgorithms,
                libkokkoskernels,
                libkokkostsqr,
                libthyracore,
                libnox,
                libloca,
                libml,
                librtop,
                libthyraepetra,
                libstratimikosifpack,
                libstratimikosml,
                libstratimikosamesos,
                libstratimikosaztecoo,
                libstratimikosamesos2,
                libstratimikosbelos,
                libstratimikos,
                libthyraepetraext,
                libteko,
                libnoxepetra,
                liblocaepetra,
                libnoxlapack,
                liblocalapack,
                liblocathyra,
                libModeLaplace,
                libmuelu,
                libmuelu_adapters,
                libmuelu_interface,
                libpiro,
                librythmos,
                libsacado,
                libshylu_nodehts,
                libstk_topology,
                libstk_util_util,
                libstk_util_parallel,
                libstk_mesh_base,
                libstk_util_env,
                libstk_util_diag,
                libstk_io,
                libstk_io_util,
                libstk_ngp_test,
                libstk_util_registry,
                libstk_util_command_line,
                libtempus,
                libteuchoskokkoscomm,
                libtpetraclassic,
                libtpetraclassiclinalg,
                libtpetraclassicnodeapi,
                libtpetrainout
            };

            //Stack<TrilinosLibrary> librariesOrdered = DependentsSort();


            //TrilinosLibrary libraryToLoad;
            //while(librariesOrdered.Count > 0)
            //{
            //    libraryToLoad = librariesOrdered.Pop();

            //    Console.WriteLine(libraryToLoad.LibraryName);

            //    libraryToLoad.Load();
            //}

            //libkokkoscoreHandle       = PlatformApi.NativeLibrary.LoadLibrary(libkokkoscoreLibraryName);
            //libkokkoscontainersHandle = PlatformApi.NativeLibrary.LoadLibrary(libkokkoscontainersLibraryName);
            //libkokkosalgorithmsHandle = PlatformApi.NativeLibrary.LoadLibrary(libkokkosalgorithmsLibraryName);
            //libkokkoskernelsHandle    = PlatformApi.NativeLibrary.LoadLibrary(libkokkoskernelsLibraryName);

            //libexodusHandle = PlatformApi.NativeLibrary.LoadLibrary(libexodusLibraryName);

            //libshardsHandle = PlatformApi.NativeLibrary.LoadLibrary(libshardsLibraryName);

            //libsacadoHandle = PlatformApi.NativeLibrary.LoadLibrary(libsacadoLibraryName);

            //libteuchoscoreHandle          = PlatformApi.NativeLibrary.LoadLibrary(libteuchoscoreLibraryName);
            //libteuchosparserHandle        = PlatformApi.NativeLibrary.LoadLibrary(libteuchosparserLibraryName);
            //libteuchosnumericsHandle      = PlatformApi.NativeLibrary.LoadLibrary(libteuchosnumericsLibraryName);
            //libteuchosremainderHandle     = PlatformApi.NativeLibrary.LoadLibrary(libteuchosremainderLibraryName);
            //libteuchoskokkoscommHandle    = PlatformApi.NativeLibrary.LoadLibrary(libteuchoskokkoscommLibraryName);
            //libteuchoskokkoscompatHandle  = PlatformApi.NativeLibrary.LoadLibrary(libteuchoskokkoscompatLibraryName);
            //libteuchosparameterlistHandle = PlatformApi.NativeLibrary.LoadLibrary(libteuchosparameterlistLibraryName);
            //libteuchoscommHandle          = PlatformApi.NativeLibrary.LoadLibrary(libteuchoscommLibraryName);

            //libIossHandle        = PlatformApi.NativeLibrary.LoadLibrary(libIossLibraryName);
            //libIoexHandle        = PlatformApi.NativeLibrary.LoadLibrary(libIoexLibraryName);
            //libio_info_libHandle = PlatformApi.NativeLibrary.LoadLibrary(libio_info_libLibraryName);
            //libIohbHandle        = PlatformApi.NativeLibrary.LoadLibrary(libIohbLibraryName);
            //libIognHandle        = PlatformApi.NativeLibrary.LoadLibrary(libIognLibraryName);
            //libIovsHandle        = PlatformApi.NativeLibrary.LoadLibrary(libIovsLibraryName);
            //libIogsHandle        = PlatformApi.NativeLibrary.LoadLibrary(libIogsLibraryName);
            //libIotrHandle        = PlatformApi.NativeLibrary.LoadLibrary(libIotrLibraryName);
            //libIonitHandle       = PlatformApi.NativeLibrary.LoadLibrary(libIonitLibraryName);

            //libstk_ngp_testHandle          = PlatformApi.NativeLibrary.LoadLibrary(libstk_ngp_testLibraryName);
            //libstk_util_utilHandle         = PlatformApi.NativeLibrary.LoadLibrary(libstk_util_utilLibraryName);
            //libstk_topologyHandle          = PlatformApi.NativeLibrary.LoadLibrary(libstk_topologyLibraryName);
            //libstk_util_parallelHandle     = PlatformApi.NativeLibrary.LoadLibrary(libstk_util_parallelLibraryName);
            //libstk_mesh_baseHandle         = PlatformApi.NativeLibrary.LoadLibrary(libstk_mesh_baseLibraryName);
            //libstk_util_envHandle          = PlatformApi.NativeLibrary.LoadLibrary(libstk_util_envLibraryName);
            //libstk_util_diagHandle         = PlatformApi.NativeLibrary.LoadLibrary(libstk_util_diagLibraryName);
            //libstk_util_registryHandle     = PlatformApi.NativeLibrary.LoadLibrary(libstk_util_registryLibraryName);
            //libstk_util_command_lineHandle = PlatformApi.NativeLibrary.LoadLibrary(libstk_util_command_lineLibraryName);

            //libintrepid2Handle = PlatformApi.NativeLibrary.LoadLibrary(libintrepid2LibraryName);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static void Load(BitSet? flags = null)
        {
            if (!IsLoaded)
            {
                IsLoaded = true;

                UcrtLibraries.Load();
                MiscellaneousLibraries.Load();

                if (flags == null)
                {
                    flags = TrilinosEnablePackageFlags.All;
                }

                for (int i = 0; i < Libraries.Length; ++i)
                {
                    if (Libraries[i].HasFlag(flags))
                    {
                        Libraries[i].Load();
                    }
                }
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static void Unload()
        {
            IsLoaded = false;
        }
    }
}
