
#include "Tests.hpp"

#include <ValueLimits.hpp>
#include <Statistics.hpp>
#include <Combinations.hpp>

template<class ExecutionSpace>
extern void TestCombinations2()
{
    auto km = ValueLimits<double>(0.0001, 0.01);
    auto kF = ValueLimits<double>(10.0, 1000.0);
    auto kf = ValueLimits<double>(0.01, 10.0);
    auto ye = ValueLimits<double>(100.0, 1000.0);
    auto LF = ValueLimits<double>(50.0, 250.0);
    auto Lf = ValueLimits<double>(10.0, 150.0);

    const int32 n = 3;

    typedef std::vector<double>       Set;
    typedef std::vector<Set>          SetOfSets;
    typedef SetOfSets::const_iterator SetOfSetsCIt;

    auto kmV = std::LinearSpacing<double>(km.GetLower(), km.GetUpper(), n + 2);
    auto kFV = std::LinearSpacing<double>(kF.GetLower(), kF.GetUpper(), n + 2);
    auto kfV = std::LinearSpacing<double>(kf.GetLower(), kf.GetUpper(), n + 2);
    auto yeV = std::LinearSpacing<double>(ye.GetLower(), ye.GetUpper(), n + 2);
    auto LFV = std::LinearSpacing<double>(LF.GetLower(), LF.GetUpper(), n + 2);
    auto LfV = std::LinearSpacing<double>(Lf.GetLower(), Lf.GetUpper(), n + 2);

    kmV.erase(kmV.begin(), kmV.begin() + 1);
    kmV.erase(kmV.end() - 1, kmV.end());

    kFV.erase(kFV.begin(), kFV.begin() + 1);
    kFV.erase(kFV.end() - 1, kFV.end());

    kfV.erase(kfV.begin(), kfV.begin() + 1);
    kfV.erase(kfV.end() - 1, kfV.end());

    yeV.erase(yeV.begin(), yeV.begin() + 1);
    yeV.erase(yeV.end() - 1, yeV.end());

    LFV.erase(LFV.begin(), LFV.begin() + 1);
    LFV.erase(LFV.end() - 1, LFV.end());

    LfV.erase(LfV.begin(), LfV.begin() + 1);
    LfV.erase(LfV.end() - 1, LfV.end());

    for (const auto& item : kmV)
    {
        std::cout << std::to_string(item) << " ";
    }
    std::cout << std::endl;
    for (const auto& item : kFV)
    {
        std::cout << std::to_string(item) << " ";
    }
    std::cout << std::endl;
    for (const auto& item : kfV)
    {
        std::cout << std::to_string(item) << " ";
    }
    std::cout << std::endl;
    for (const auto& item : yeV)
    {
        std::cout << std::to_string(item) << " ";
    }
    std::cout << std::endl;
    for (const auto& item : LFV)
    {
        std::cout << std::to_string(item) << " ";
    }
    std::cout << std::endl;
    for (const auto& item : LfV)
    {
        std::cout << std::to_string(item) << " ";
    }
    std::cout << std::endl;

    const SetOfSets data = {kmV, kFV, kfV, yeV, LFV, LfV};

    {
        typedef NumericalMethods::Algorithms::Internal::Combinations<SetOfSetsCIt> Combinations;
        Combinations                                                               cmbs = NumericalMethods::Algorithms::Internal::make_combinations(data);

        std::cout << "km"
                  << " "
                  << "kF"
                  << " "
                  << "kf"
                  << " "
                  << "ye"
                  << " "
                  << "LF"
                  << " "
                  << "Lf" << std::endl;

        for (const auto& combination : cmbs)
        {
            for (const auto& elemIt : combination)
            {
                std::cout << std::to_string(*elemIt) << " ";
            }
            std::cout << std::endl;
        }

        // int swarm_index    = 0;
        // int particle_index = 0;
        // int parameter_index;

        // for (const auto& combination : cmbs)
        //{
        //    parameter_index = 0;

        //    for (const auto& elemIt : combination)
        //    {
        //        // std::cout << std::to_string(*elemIt) << " ";
        //        std::cout << std::to_string(swarm_index) << " " << std::to_string(particle_index) << " " << std::to_string(parameter_index) << std::endl;
        //        ++parameter_index;
        //    }
        //    // std::cout << std::endl;

        //    if (particle_index == 64 - 1)
        //    {
        //        ++swarm_index;
        //        particle_index = 0;
        //    }
        //    else
        //    {
        //        ++particle_index;
        //    }
        //}
    }
}


template __declspec(dllexport) void TestCombinations2<EXECUTION_SPACE>();