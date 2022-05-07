

#include <Utilities/SharedDataTable.h>

void SharedDataTableTest()
{

    int nRows = 2;

    typedef Engineering::DataSource::SharedDataTableColumnSchema<System::PrimitiveKind::UInt64> SharedDataTableColumnUInt64;
    typedef Engineering::DataSource::SharedDataTableColumnSchema<System::PrimitiveKind::FP64>   SharedDataTableColumnFP64;

    Engineering::DataSource::SharedDataTableColumnSchema<> columns[] = {
        SharedDataTableColumnUInt64(L"Iteration"s),   SharedDataTableColumnUInt64(L"Particle"s), SharedDataTableColumnFP64(L"RMS"s),           SharedDataTableColumnFP64(L"km"s),
        SharedDataTableColumnFP64(L"km_velocity"s),   SharedDataTableColumnFP64(L"kF"s),         SharedDataTableColumnFP64(L"kF_velocity"s),   SharedDataTableColumnFP64(L"kf"s),
        SharedDataTableColumnFP64(L"kf_velocity"s),   SharedDataTableColumnFP64(L"ye"s),         SharedDataTableColumnFP64(L"ye_velocity"s),   SharedDataTableColumnFP64(L"LF"s),
        SharedDataTableColumnFP64(L"LF_velocity"s),   SharedDataTableColumnFP64(L"Lf"s),         SharedDataTableColumnFP64(L"Lf_velocity"s),   SharedDataTableColumnFP64(L"Sg"s),
        SharedDataTableColumnFP64(L"Sg_velocity"s),   SharedDataTableColumnFP64(L"KrgF"s),       SharedDataTableColumnFP64(L"KrgF_velocity"s), SharedDataTableColumnFP64(L"Krgf"s),
        SharedDataTableColumnFP64(L"Krgf_velocity"s), SharedDataTableColumnFP64(L"Krgm"s),       SharedDataTableColumnFP64(L"Krgm_velocity"s), SharedDataTableColumnFP64(L"So"s),
        SharedDataTableColumnFP64(L"So_velocity"s),   SharedDataTableColumnFP64(L"KroF"s),       SharedDataTableColumnFP64(L"KroF_velocity"s), SharedDataTableColumnFP64(L"Krof"s),
        SharedDataTableColumnFP64(L"Krof_velocity"s), SharedDataTableColumnFP64(L"Krom"s),       SharedDataTableColumnFP64(L"Krom_velocity")};

    Engineering::DataSource::SharedDataTable<uint64, double>* sharedDataTable = Engineering::DataSource::SharedDataTable<uint64, double>::Create(L"PSO", nRows, columns);

    int32 columnIdx;

    columnIdx = 0;
    std::wcout << sharedDataTable->Header[columnIdx++] << std::endl;
    std::wcout << sharedDataTable->Header[columnIdx++] << std::endl;
    std::wcout << sharedDataTable->Header[columnIdx++] << std::endl;
    std::wcout << sharedDataTable->Header[columnIdx++] << std::endl;
    std::wcout << sharedDataTable->Header[columnIdx++] << std::endl;
    std::wcout << sharedDataTable->Header[columnIdx++] << std::endl;
    std::wcout << sharedDataTable->Header[columnIdx++] << std::endl;
    std::wcout << sharedDataTable->Header[columnIdx++] << std::endl;
    std::wcout << sharedDataTable->Header[columnIdx++] << std::endl;
    std::wcout << sharedDataTable->Header[columnIdx++] << std::endl;
    std::wcout << sharedDataTable->Header[columnIdx++] << std::endl;
    std::wcout << sharedDataTable->Header[columnIdx++] << std::endl;
    std::wcout << sharedDataTable->Header[columnIdx++] << std::endl;
    std::wcout << sharedDataTable->Header[columnIdx++] << std::endl;
    std::wcout << sharedDataTable->Header[columnIdx++] << std::endl;
    std::wcout << sharedDataTable->Header[columnIdx++] << std::endl;
    std::wcout << sharedDataTable->Header[columnIdx++] << std::endl;
    std::wcout << sharedDataTable->Header[columnIdx++] << std::endl;
    std::wcout << sharedDataTable->Header[columnIdx++] << std::endl;
    std::wcout << sharedDataTable->Header[columnIdx++] << std::endl;
    std::wcout << sharedDataTable->Header[columnIdx++] << std::endl;
    std::wcout << sharedDataTable->Header[columnIdx++] << std::endl;
    std::wcout << sharedDataTable->Header[columnIdx++] << std::endl;
    std::wcout << sharedDataTable->Header[columnIdx++] << std::endl;
    std::wcout << sharedDataTable->Header[columnIdx++] << std::endl;
    std::wcout << sharedDataTable->Header[columnIdx++] << std::endl;
    std::wcout << sharedDataTable->Header[columnIdx++] << std::endl;
    std::wcout << sharedDataTable->Header[columnIdx++] << std::endl;
    std::wcout << sharedDataTable->Header[columnIdx++] << std::endl;
    std::wcout << sharedDataTable->Header[columnIdx++] << std::endl;
    std::wcout << sharedDataTable->Header[columnIdx] << std::endl;

    std::cout << std::endl;
    std::cout << std::endl;

    columnIdx = 0;
    sharedDataTable->SetValue(0, columnIdx++, 0);
    sharedDataTable->SetValue(0, columnIdx++, 0);
    sharedDataTable->SetValue(0, columnIdx++, 207.91215266018088);
    sharedDataTable->SetValue(0, columnIdx++, 0.004301334876225365);
    sharedDataTable->SetValue(0, columnIdx++, -0.0002677032498810585);
    sharedDataTable->SetValue(0, columnIdx++, 11550.735503291184);
    sharedDataTable->SetValue(0, columnIdx++, -4224.632248354408);
    sharedDataTable->SetValue(0, columnIdx++, 1443.2455656682348);
    sharedDataTable->SetValue(0, columnIdx++, -230.56044151036696);
    sharedDataTable->SetValue(0, columnIdx++, 107.82624679980268);
    sharedDataTable->SetValue(0, columnIdx++, 17.95815327413011);
    sharedDataTable->SetValue(0, columnIdx++, 401.0578734338122);
    sharedDataTable->SetValue(0, columnIdx++, 10.55146024134962);
    sharedDataTable->SetValue(0, columnIdx++, 120.24569062574434);
    sharedDataTable->SetValue(0, columnIdx++, -72.35127100659713);
    sharedDataTable->SetValue(0, columnIdx++, 0.5324507014573712);
    sharedDataTable->SetValue(0, columnIdx++, 0.002046482475658619);
    sharedDataTable->SetValue(0, columnIdx++, 0.8343634717663323);
    sharedDataTable->SetValue(0, columnIdx++, -0.048516032808734164);
    sharedDataTable->SetValue(0, columnIdx++, 0.4735258820130159);
    sharedDataTable->SetValue(0, columnIdx++, 0.23676294100650783);
    sharedDataTable->SetValue(0, columnIdx++, 0.23728719242124663);
    sharedDataTable->SetValue(0, columnIdx++, 0.1186435962106232);
    sharedDataTable->SetValue(0, columnIdx++, 0.07485978696561646);
    sharedDataTable->SetValue(0, columnIdx++, -0.18855079823028187);
    sharedDataTable->SetValue(0, columnIdx++, 0.2238064513916355);
    sharedDataTable->SetValue(0, columnIdx++, 0.0224007674195905);
    sharedDataTable->SetValue(0, columnIdx++, 0.6267164240866504);
    sharedDataTable->SetValue(0, columnIdx++, 0.006333819705852502);
    sharedDataTable->SetValue(0, columnIdx++, 0.9338939089464722);
    sharedDataTable->SetValue(0, columnIdx, -0.03305304552676391);

    columnIdx = 0;
    sharedDataTable->SetValue(1, columnIdx++, 0);
    sharedDataTable->SetValue(1, columnIdx++, 1);
    sharedDataTable->SetValue(1, columnIdx++, 207.7570548219494);
    sharedDataTable->SetValue(1, columnIdx++, 0.0026257180715087186);
    sharedDataTable->SetValue(1, columnIdx++, 0.0006317431096925668);
    sharedDataTable->SetValue(1, columnIdx++, 8512.098545009594);
    sharedDataTable->SetValue(1, columnIdx++, 1080.3516979668575);
    sharedDataTable->SetValue(1, columnIdx++, 397.05922344604244);
    sharedDataTable->SetValue(1, columnIdx++, 2.1204828410805496);
    sharedDataTable->SetValue(1, columnIdx++, 351.9231510548292);
    sharedDataTable->SetValue(1, columnIdx++, 117.20753082876843);
    sharedDataTable->SetValue(1, columnIdx++, 395.32994003407845);
    sharedDataTable->SetValue(1, columnIdx++, 23.67874912674148);
    sharedDataTable->SetValue(1, columnIdx++, 281.5100327909447);
    sharedDataTable->SetValue(1, columnIdx++, 13.1473801900652);
    sharedDataTable->SetValue(1, columnIdx++, 0.9729865778337337);
    sharedDataTable->SetValue(1, columnIdx++, -0.013506711083133172);
    sharedDataTable->SetValue(1, columnIdx++, 0.5381910669276636);
    sharedDataTable->SetValue(1, columnIdx++, 0.022268301121861997);
    sharedDataTable->SetValue(1, columnIdx++, 0.320497237999601);
    sharedDataTable->SetValue(1, columnIdx++, 0.1602486189998004);
    sharedDataTable->SetValue(1, columnIdx++, 0.3385441917500026);
    sharedDataTable->SetValue(1, columnIdx++, -0.012671386795575133);
    sharedDataTable->SetValue(1, columnIdx++, 0.17888869589034917);
    sharedDataTable->SetValue(1, columnIdx++, -0.037824796733618195);
    sharedDataTable->SetValue(1, columnIdx++, 0.6592417054012351);
    sharedDataTable->SetValue(1, columnIdx++, -0.0042667371164846896);
    sharedDataTable->SetValue(1, columnIdx++, 0.5521585127333527);
    sharedDataTable->SetValue(1, columnIdx++, 0.15153758844110618);
    sharedDataTable->SetValue(1, columnIdx++, 0.717524028967944);
    sharedDataTable->SetValue(1, columnIdx++, -0.053241674918967474);

    std::cout << std::endl;
    std::cout << std::endl;

    columnIdx = 0;
    std::cout << std::to_string(*sharedDataTable->Get<uint64>(0, columnIdx++)) << std::endl;
    std::cout << std::to_string(*sharedDataTable->Get<uint64>(0, columnIdx++)) << std::endl;
    std::cout << std::to_string(*sharedDataTable->Get<fp64>(0, columnIdx++)) << std::endl;
    std::cout << std::to_string(*sharedDataTable->Get<fp64>(0, columnIdx++)) << std::endl;
    std::cout << std::to_string(*sharedDataTable->Get<fp64>(0, columnIdx++)) << std::endl;
    std::cout << std::to_string(*sharedDataTable->Get<fp64>(0, columnIdx++)) << std::endl;
    std::cout << std::to_string(*sharedDataTable->Get<fp64>(0, columnIdx++)) << std::endl;
    std::cout << std::to_string(*sharedDataTable->Get<fp64>(0, columnIdx++)) << std::endl;
    std::cout << std::to_string(*sharedDataTable->Get<fp64>(0, columnIdx++)) << std::endl;
    std::cout << std::to_string(*sharedDataTable->Get<fp64>(0, columnIdx++)) << std::endl;
    std::cout << std::to_string(*sharedDataTable->Get<fp64>(0, columnIdx++)) << std::endl;
    std::cout << std::to_string(*sharedDataTable->Get<fp64>(0, columnIdx++)) << std::endl;
    std::cout << std::to_string(*sharedDataTable->Get<fp64>(0, columnIdx++)) << std::endl;
    std::cout << std::to_string(*sharedDataTable->Get<fp64>(0, columnIdx++)) << std::endl;
    std::cout << std::to_string(*sharedDataTable->Get<fp64>(0, columnIdx++)) << std::endl;
    std::cout << std::to_string(*sharedDataTable->Get<fp64>(0, columnIdx++)) << std::endl;
    std::cout << std::to_string(*sharedDataTable->Get<fp64>(0, columnIdx++)) << std::endl;
    std::cout << std::to_string(*sharedDataTable->Get<fp64>(0, columnIdx++)) << std::endl;
    std::cout << std::to_string(*sharedDataTable->Get<fp64>(0, columnIdx++)) << std::endl;
    std::cout << std::to_string(*sharedDataTable->Get<fp64>(0, columnIdx++)) << std::endl;
    std::cout << std::to_string(*sharedDataTable->Get<fp64>(0, columnIdx++)) << std::endl;
    std::cout << std::to_string(*sharedDataTable->Get<fp64>(0, columnIdx++)) << std::endl;
    std::cout << std::to_string(*sharedDataTable->Get<fp64>(0, columnIdx++)) << std::endl;
    std::cout << std::to_string(*sharedDataTable->Get<fp64>(0, columnIdx++)) << std::endl;
    std::cout << std::to_string(*sharedDataTable->Get<fp64>(0, columnIdx++)) << std::endl;
    std::cout << std::to_string(*sharedDataTable->Get<fp64>(0, columnIdx++)) << std::endl;
    std::cout << std::to_string(*sharedDataTable->Get<fp64>(0, columnIdx++)) << std::endl;
    std::cout << std::to_string(*sharedDataTable->Get<fp64>(0, columnIdx++)) << std::endl;
    std::cout << std::to_string(*sharedDataTable->Get<fp64>(0, columnIdx++)) << std::endl;
    std::cout << std::to_string(*sharedDataTable->Get<fp64>(0, columnIdx++)) << std::endl;
    std::cout << std::to_string(*sharedDataTable->Get<fp64>(0, columnIdx)) << std::endl;

    columnIdx = 0;
    std::cout << std::to_string(*sharedDataTable->Get<uint64>(1, columnIdx++)) << std::endl;
    std::cout << std::to_string(*sharedDataTable->Get<uint64>(1, columnIdx++)) << std::endl;
    std::cout << std::to_string(*sharedDataTable->Get<fp64>(1, columnIdx++)) << std::endl;
    std::cout << std::to_string(*sharedDataTable->Get<fp64>(1, columnIdx++)) << std::endl;
    std::cout << std::to_string(*sharedDataTable->Get<fp64>(1, columnIdx++)) << std::endl;
    std::cout << std::to_string(*sharedDataTable->Get<fp64>(1, columnIdx++)) << std::endl;
    std::cout << std::to_string(*sharedDataTable->Get<fp64>(1, columnIdx++)) << std::endl;
    std::cout << std::to_string(*sharedDataTable->Get<fp64>(1, columnIdx++)) << std::endl;
    std::cout << std::to_string(*sharedDataTable->Get<fp64>(1, columnIdx++)) << std::endl;
    std::cout << std::to_string(*sharedDataTable->Get<fp64>(1, columnIdx++)) << std::endl;
    std::cout << std::to_string(*sharedDataTable->Get<fp64>(1, columnIdx++)) << std::endl;
    std::cout << std::to_string(*sharedDataTable->Get<fp64>(1, columnIdx++)) << std::endl;
    std::cout << std::to_string(*sharedDataTable->Get<fp64>(1, columnIdx++)) << std::endl;
    std::cout << std::to_string(*sharedDataTable->Get<fp64>(1, columnIdx++)) << std::endl;
    std::cout << std::to_string(*sharedDataTable->Get<fp64>(1, columnIdx++)) << std::endl;
    std::cout << std::to_string(*sharedDataTable->Get<fp64>(1, columnIdx++)) << std::endl;
    std::cout << std::to_string(*sharedDataTable->Get<fp64>(1, columnIdx++)) << std::endl;
    std::cout << std::to_string(*sharedDataTable->Get<fp64>(1, columnIdx++)) << std::endl;
    std::cout << std::to_string(*sharedDataTable->Get<fp64>(1, columnIdx++)) << std::endl;
    std::cout << std::to_string(*sharedDataTable->Get<fp64>(1, columnIdx++)) << std::endl;
    std::cout << std::to_string(*sharedDataTable->Get<fp64>(1, columnIdx++)) << std::endl;
    std::cout << std::to_string(*sharedDataTable->Get<fp64>(1, columnIdx++)) << std::endl;
    std::cout << std::to_string(*sharedDataTable->Get<fp64>(1, columnIdx++)) << std::endl;
    std::cout << std::to_string(*sharedDataTable->Get<fp64>(1, columnIdx++)) << std::endl;
    std::cout << std::to_string(*sharedDataTable->Get<fp64>(1, columnIdx++)) << std::endl;
    std::cout << std::to_string(*sharedDataTable->Get<fp64>(1, columnIdx++)) << std::endl;
    std::cout << std::to_string(*sharedDataTable->Get<fp64>(1, columnIdx++)) << std::endl;
    std::cout << std::to_string(*sharedDataTable->Get<fp64>(1, columnIdx++)) << std::endl;
    std::cout << std::to_string(*sharedDataTable->Get<fp64>(1, columnIdx++)) << std::endl;
    std::cout << std::to_string(*sharedDataTable->Get<fp64>(1, columnIdx++)) << std::endl;
    std::cout << std::to_string(*sharedDataTable->Get<fp64>(1, columnIdx)) << std::endl;
}
