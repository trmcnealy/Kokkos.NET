

#include <MultiPorosityModel/PorosityModelKind.hpp>
#include <MultiPorosityModel/InverseTransformPrecisionKind.hpp>
#include <MultiPorosityModel/ModelInputParameters.hpp>
#include <MultiPorosityModel/MultiPorosityRegression.hpp>

namespace Internal
{
    using DataType = fp64;

    using ExecutionSpace = Kokkos::Cuda;

    using LayoutType = typename ExecutionSpace::array_layout;

    template<typename Type>
    using Scalar = Kokkos::View<Type, LayoutType, ExecutionSpace>;
    template<typename Type>
    using Vector = Kokkos::View<Type*, LayoutType, ExecutionSpace>;
    template<typename Type>
    using ConstVector = Kokkos::View<const Type*, LayoutType, ExecutionSpace>;

    template<typename Type>
    using DataVector = Kokkos::View<Type**, LayoutType, ExecutionSpace>;
    template<typename Type>
    using ConstDataVector = Kokkos::View<const Type**, LayoutType, ExecutionSpace>;

    template<typename Type>
    using Matrix = Kokkos::View<Type**, LayoutType, ExecutionSpace>;
    template<typename Type>
    using ConstMatrix = Kokkos::View<const Type**, LayoutType, ExecutionSpace>;
    template<typename Type>
    using Tensor = Kokkos::View<Type***, LayoutType, ExecutionSpace>;

    using namespace Petroleum::Reservoir;

    using MultiPorositySolver  = MultiPorositySolver<DataType, ExecutionSpace, PorosityModel::Triple, InverseTransformPrecision::Medium>;
    using ModelInputParameters = ModelInputParameters<DataType, PorosityModel::Triple>;

    static constexpr uint16 Precision = TeamVectorSize(InverseTransformPrecision::Medium) - 1U;

    template<typename FunctionType>
    using ParticleSwarmOptimization        = NumericalMethods::Algorithms::ParticleSwarmOptimization<DataType, ExecutionSpace, FunctionType, 3, Precision>;
    using ParticleSwarmOptimizationOptions = NumericalMethods::Algorithms::ParticleSwarmOptimizationOptions<DataType>;
}

void RunMultiPorosityModel()
{
    ::Internal::ModelInputParameters multi_porosity_parameters;
    multi_porosity_parameters.FormationHeight            = 125.0;
    multi_porosity_parameters.FormationLength            = 6500.0;
    multi_porosity_parameters.Acw                        = 1625000.0;
    multi_porosity_parameters.InitPressure               = 7000.0;
    multi_porosity_parameters.WellboreFlowingPressure    = 3500.0;
    multi_porosity_parameters.BottomholeTemperature      = 275.0;
    multi_porosity_parameters.DeltaPressureOil           = 3500.0;
    multi_porosity_parameters.DeltaPressureWater         = 3500.0;
    multi_porosity_parameters.DeltaPressureGas           = 1308993218.9459749;
    multi_porosity_parameters.WF                         = 0.008333333333333334;
    multi_porosity_parameters.Wf                         = 0.0008333333333333334;
    multi_porosity_parameters.PhiF                       = 0.2;
    multi_porosity_parameters.Phif                       = 0.1;
    multi_porosity_parameters.Phim                       = 0.06;
    multi_porosity_parameters.OilSolutionGas             = 1301.598337295025;
    multi_porosity_parameters.OilVaporizedRatio          = 42.62126918331808;
    multi_porosity_parameters.OilSaturation              = 0.8250000000000001;
    multi_porosity_parameters.OilApiGravity              = 46.8;
    multi_porosity_parameters.OilViscosity               = 0.16525289334393976;
    multi_porosity_parameters.OilFormationVolumeFactor   = 1.7635084680500342;
    multi_porosity_parameters.OilCompressibility         = 0.000018775126163268967;
    multi_porosity_parameters.WaterSaturation            = 0.0;
    multi_porosity_parameters.WaterSpecificGravity       = 1.02;
    multi_porosity_parameters.WaterViscosity             = 0.4111300824817609;
    multi_porosity_parameters.WaterFormationVolumeFactor = 1.0473400098328994;
    multi_porosity_parameters.WaterCompressibility       = 0.000003226481210342687;
    multi_porosity_parameters.GasSaturation              = 0.175;
    multi_porosity_parameters.GasSpecificGravity         = 0.65;
    multi_porosity_parameters.GasViscosity               = 0.024396806595039679;
    multi_porosity_parameters.GasFormationVolumeFactor   = 0.0008162882694002733;
    multi_porosity_parameters.GasCompressibilityFactor   = 1.150765844646991;
    multi_porosity_parameters.GasCompressibility         = 0.00012621444306349592;
    multi_porosity_parameters.TotalCompressibility       = 0.000057577006620808689;
    multi_porosity_parameters.Krom                       = 0.18730048213226092;
    multi_porosity_parameters.Krwm                       = 0.0;
    multi_porosity_parameters.Krgm                       = 0.01129770435826271;
    multi_porosity_parameters.KroF                       = 0.6447641565643524;
    multi_porosity_parameters.KrwF                       = 0.0;
    multi_porosity_parameters.KrgF                       = 0.33811257397258806;
    multi_porosity_parameters.Krof                       = 0.8396292505331319;
    multi_porosity_parameters.Krwf                       = 0.0;
    multi_porosity_parameters.Krgf                       = 0.33334497969277807;

    ::Internal::MultiPorositySolver* mps = new ::Internal::MultiPorositySolver(multi_porosity_parameters);

    std::cout << "::Internal::MultiPorositySolver:" << sizeof(*mps) << std::endl;

    ::Internal::ParticleSwarmOptimizationOptions options(5, 5, 200, 0.00, 0.2, 1.0, nullptr, false);

    Teuchos::RCP<::Internal::DataVector<::Internal::DataType>>* actual_data =
        new Teuchos::RCP<::Internal::DataVector<::Internal::DataType>>(new ::Internal::DataVector<::Internal::DataType>("actual_data", 29, 3));

    (*actual_data)->operator()(0, 0)  = 0;
    (*actual_data)->operator()(1, 0)  = 715.5714286;
    (*actual_data)->operator()(2, 0)  = 306.95;
    (*actual_data)->operator()(3, 0)  = 215.5645161;
    (*actual_data)->operator()(4, 0)  = 186.7903226;
    (*actual_data)->operator()(5, 0)  = 201.6290323;
    (*actual_data)->operator()(6, 0)  = 143.7;
    (*actual_data)->operator()(7, 0)  = 138.4032258;
    (*actual_data)->operator()(8, 0)  = 85.35483871;
    (*actual_data)->operator()(9, 0)  = 65.75806452;
    (*actual_data)->operator()(10, 0) = 61.01612903;
    (*actual_data)->operator()(11, 0) = 67.33928571;
    (*actual_data)->operator()(12, 0) = 69.51612903;
    (*actual_data)->operator()(13, 0) = 60.06666667;
    (*actual_data)->operator()(14, 0) = 42.4516129;
    (*actual_data)->operator()(15, 0) = 26.82258065;
    (*actual_data)->operator()(16, 0) = 36.75;
    (*actual_data)->operator()(17, 0) = 28.83870968;
    (*actual_data)->operator()(18, 0) = 13.75806452;
    (*actual_data)->operator()(19, 0) = 15.10714286;
    (*actual_data)->operator()(20, 0) = 13.17741935;
    (*actual_data)->operator()(21, 0) = 15.45;
    (*actual_data)->operator()(22, 0) = 12.88709677;
    (*actual_data)->operator()(23, 0) = 10.22580645;
    (*actual_data)->operator()(24, 0) = 11.58333333;
    (*actual_data)->operator()(25, 0) = 10.62903226;
    (*actual_data)->operator()(26, 0) = 10.11666667;
    (*actual_data)->operator()(27, 0) = 8.822580645;
    (*actual_data)->operator()(28, 0) = 11.42857143;

    (*actual_data)->operator()(0, 1)  = 0;
    (*actual_data)->operator()(1, 1)  = 1071.857143;
    (*actual_data)->operator()(2, 1)  = 539.8;
    (*actual_data)->operator()(3, 1)  = 344.5;
    (*actual_data)->operator()(4, 1)  = 320.4032258;
    (*actual_data)->operator()(5, 1)  = 298.6290323;
    (*actual_data)->operator()(6, 1)  = 242.1;
    (*actual_data)->operator()(7, 1)  = 196.8387097;
    (*actual_data)->operator()(8, 1)  = 142.7419355;
    (*actual_data)->operator()(9, 1)  = 119.4516129;
    (*actual_data)->operator()(10, 1) = 98.56451613;
    (*actual_data)->operator()(11, 1) = 106.8035714;
    (*actual_data)->operator()(12, 1) = 100.1612903;
    (*actual_data)->operator()(13, 1) = 92.8;
    (*actual_data)->operator()(14, 1) = 65.09677419;
    (*actual_data)->operator()(15, 1) = 51.79032258;
    (*actual_data)->operator()(16, 1) = 66.56666667;
    (*actual_data)->operator()(17, 1) = 49.59677419;
    (*actual_data)->operator()(18, 1) = 25.24193548;
    (*actual_data)->operator()(19, 1) = 31.26785714;
    (*actual_data)->operator()(20, 1) = 20.61290323;
    (*actual_data)->operator()(21, 1) = 22.33333333;
    (*actual_data)->operator()(22, 1) = 24.16129032;
    (*actual_data)->operator()(23, 1) = 18.19354839;
    (*actual_data)->operator()(24, 1) = 18.15;
    (*actual_data)->operator()(25, 1) = 18.35483871;
    (*actual_data)->operator()(26, 1) = 18.25;
    (*actual_data)->operator()(27, 1) = 16.33870968;
    (*actual_data)->operator()(28, 1) = 15.85714286;

    Teuchos::RCP<::Internal::Vector<::Internal::DataType>>* actual_time = new Teuchos::RCP<::Internal::Vector<::Internal::DataType>>(new ::Internal::Vector<::Internal::DataType>("actual_time", 29));

    (*actual_time)->operator[](0)  = 1.0;
    (*actual_time)->operator[](1)  = 8.0;
    (*actual_time)->operator[](2)  = 30.0;
    (*actual_time)->operator[](3)  = 61.0;
    (*actual_time)->operator[](4)  = 106.0;
    (*actual_time)->operator[](5)  = 151.0;
    (*actual_time)->operator[](6)  = 181.0;
    (*actual_time)->operator[](7)  = 242.0;
    (*actual_time)->operator[](8)  = 349.0;
    (*actual_time)->operator[](9)  = 426.0;
    (*actual_time)->operator[](10) = 457.0;
    (*actual_time)->operator[](11) = 486.0;
    (*actual_time)->operator[](12) = 516.0;
    (*actual_time)->operator[](13) = 653.0;
    (*actual_time)->operator[](14) = 851.0;
    (*actual_time)->operator[](15) = 942.0;
    (*actual_time)->operator[](16) = 972.0;
    (*actual_time)->operator[](17) = 1262.0;
    (*actual_time)->operator[](18) = 1553.0;
    (*actual_time)->operator[](19) = 1582.0;
    (*actual_time)->operator[](20) = 1642.0;
    (*actual_time)->operator[](21) = 1703.0;
    (*actual_time)->operator[](22) = 1734.0;
    (*actual_time)->operator[](23) = 1765.0;
    (*actual_time)->operator[](24) = 1795.0;
    (*actual_time)->operator[](25) = 1826.0;
    (*actual_time)->operator[](26) = 1856.0;
    (*actual_time)->operator[](27) = 1902.0;
    (*actual_time)->operator[](28) = 1947.0;

    Teuchos::RCP<::Internal::Vector<::Internal::DataType>>* weights = new Teuchos::RCP<::Internal::Vector<::Internal::DataType>>(new ::Internal::Vector<::Internal::DataType>("weights", 29));

    (*weights)->operator[](0)  = 0.0;
    (*weights)->operator[](1)  = 1.0;
    (*weights)->operator[](2)  = 1.0;
    (*weights)->operator[](3)  = 1.0;
    (*weights)->operator[](4)  = 1.0;
    (*weights)->operator[](5)  = 1.0;
    (*weights)->operator[](6)  = 1.0;
    (*weights)->operator[](7)  = 1.0;
    (*weights)->operator[](8)  = 1.0;
    (*weights)->operator[](9)  = 1.0;
    (*weights)->operator[](10) = 1.0;
    (*weights)->operator[](11) = 1.0;
    (*weights)->operator[](12) = 1.0;
    (*weights)->operator[](13) = 1.0;
    (*weights)->operator[](14) = 1.0;
    (*weights)->operator[](15) = 1.0;
    (*weights)->operator[](16) = 1.0;
    (*weights)->operator[](17) = 1.0;
    (*weights)->operator[](18) = 1.0;
    (*weights)->operator[](19) = 1.0;
    (*weights)->operator[](20) = 1.0;
    (*weights)->operator[](21) = 1.0;
    (*weights)->operator[](22) = 1.0;
    (*weights)->operator[](23) = 1.0;
    (*weights)->operator[](24) = 1.0;
    (*weights)->operator[](25) = 1.0;
    (*weights)->operator[](26) = 1.0;
    (*weights)->operator[](27) = 1.0;
    (*weights)->operator[](28) = 1.0;

    System::NativeArray<ValueLimits<::Internal::DataType>>* limits = new System::NativeArray<ValueLimits<::Internal::DataType>>(6);

    limits->operator[](0) = ValueLimits<::Internal::DataType>(0.001, 0.01);
    limits->operator[](1) = ValueLimits<::Internal::DataType>(100, 20000);
    limits->operator[](2) = ValueLimits<::Internal::DataType>(10, 2000);
    limits->operator[](3) = ValueLimits<::Internal::DataType>(1, 500);
    limits->operator[](4) = ValueLimits<::Internal::DataType>(1, 500);
    limits->operator[](5) = ValueLimits<::Internal::DataType>(1, 500);

    Petroleum::Reservoir::MultiPorosityResult<::Internal::DataType, ::Internal::ExecutionSpace>* result = mps->HistoryMatch(options, actual_data, actual_time, weights, limits);

    std::cout << "Error:" << result->Error << std::endl;

    for (size_type i = 0; i < result->Args.extent(0); ++i)
    {
        std::cout << result->Args(i) << std::endl;
    }
}
