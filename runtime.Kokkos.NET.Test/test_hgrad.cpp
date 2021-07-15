
#include <Types.hpp>

#include <runtime.Kokkos/ViewTypes.hpp>

//#include <Kokkos_Core.hpp>
#include <impl/Kokkos_Timer.hpp>

#include <Teuchos_CommandLineProcessor.hpp>

#include <Intrepid2_Types.hpp>
#include "test_hgrad.hpp"

#include <iomanip>

int main(int argc, char** argv)
{

    Teuchos::CommandLineProcessor clp;
    clp.setDocString("Intrepid2::DynRankView_PerfTest01.\n");

    int nworkset = 8;
    clp.setOption("nworkset", &nworkset, "# of worksets");

    int C = 4096;
    clp.setOption("C", &C, "# of Cells in a workset");

    int order = 2;
    clp.setOption("order", &order, "cubature order");

    bool verbose = true;
    clp.setOption("enable-verbose", "disable-verbose", &verbose, "Flag for verbose printing");

    clp.recogniseAllOptions(true);
    clp.throwExceptions(false);

    const Teuchos::CommandLineProcessor::EParseCommandLineReturn r_parse = clp.parse(argc, argv);

    if (r_parse == Teuchos::CommandLineProcessor::PARSE_HELP_PRINTED)
    {
        return 0;
    }
    if (r_parse != Teuchos::CommandLineProcessor::PARSE_SUCCESSFUL)
    {
        return -1;
    }

    const int  num_threads      = 16;
    const int  num_numa         = 1;
    const int  device_id        = 0;
    const int  ndevices         = 3;
    const int  skip_device      = 9999;
    const bool disable_warnings = true;

    Kokkos::InitArguments arguments{};
    arguments.num_threads      = num_threads;
    arguments.num_numa         = num_numa;
    arguments.device_id        = device_id;
    arguments.ndevices         = ndevices;
    arguments.skip_device      = skip_device;
    arguments.disable_warnings = disable_warnings;

    Kokkos::ScopeGuard kokkos(arguments);
    {
        if (verbose)
        {
            std::cout << "Testing datatype double\n";
        }

        const int r_val_double = Intrepid2::Test::ComputeBasis_HGRAD<double, Kokkos::OpenMP>(nworkset, C, order, verbose);

        std::cout << "Press any key to exit." << std::endl;
        getchar();

        return r_val_double;
    }
}
