
#include <Types.hpp>

#include <runtime.Kokkos/ViewTypes.hpp>
#include <runtime.Kokkos/Extensions.hpp>

//#include <windows.h>
//
//#include <cstring>
//#include <iostream>
//
// struct RawSMBIOSData
//{
//    BYTE  Used20CallingMethod;
//    BYTE  SMBIOSMajorVersion;
//    BYTE  SMBIOSMinorVersion;
//    BYTE  DmiRevision;
//    DWORD Length;
//    BYTE  SMBIOSTableData[];
//};
//
// void ACPITest()
//{
//    const DWORD smBiosDataSize = GetSystemFirmwareTable('ACPI', 0, nullptr, 0);
//
//    RawSMBIOSData* smBiosData = (RawSMBIOSData*)HeapAlloc(GetProcessHeap(), 0, smBiosDataSize);
//
//    DWORD bytesWritten = GetSystemFirmwareTable('ACPI', 0, smBiosData, smBiosDataSize);
//
//    std::cout << bytesWritten << std::endl;
//
//    std::cout << smBiosData->Used20CallingMethod << std::endl;
//    std::cout << smBiosData->SMBIOSMajorVersion << std::endl;
//    std::cout << smBiosData->SMBIOSMinorVersion << std::endl;
//    std::cout << smBiosData->DmiRevision << std::endl;
//    std::cout << smBiosData->Length << std::endl;
//
//    for (DWORD i = 0; i < smBiosData->Length; i++)
//    {
//        std::cout << smBiosData->SMBIOSTableData[i];
//    }
//    std::cout << std::endl;
//}
//
// void RSMBTest()
//{
//    const DWORD smBiosDataSize = GetSystemFirmwareTable('RSMB', 0, nullptr, 0);
//
//    RawSMBIOSData* smBiosData = (RawSMBIOSData*)HeapAlloc(GetProcessHeap(), 0, smBiosDataSize);
//
//    DWORD bytesWritten = GetSystemFirmwareTable('RSMB', 0, smBiosData, smBiosDataSize);
//
//    std::cout << bytesWritten << std::endl;
//
//    std::cout << smBiosData->Used20CallingMethod << std::endl;
//    std::cout << smBiosData->SMBIOSMajorVersion << std::endl;
//    std::cout << smBiosData->SMBIOSMinorVersion << std::endl;
//    std::cout << smBiosData->DmiRevision << std::endl;
//    std::cout << smBiosData->Length << std::endl;
//
//    for (DWORD i = 0; i < smBiosData->Length; i++)
//    {
//        std::cout << smBiosData->SMBIOSTableData[i];
//    }
//    std::cout << std::endl;
//}

// struct LOADPARMS32
//{
//    LPSTR lpEnvAddress;
//    LPSTR lpCmdLine;
//    LPSTR lpCmdShow;
//    DWORD dwReserved;
//};
// HMODULE hlib = LoadLibraryA("runtime.MultiPorosity.x64.dll");
// HMODULE hModule = GetModuleHandleA("runtime.MultiPorosity.x64.dll");
// typedef BOOL (*DllMainT)(HANDLE, DWORD, LPVOID);
// DllMainT pDllMain = (DllMainT)GetProcAddress(hModule, "DllMain");
// pDllMain(hModule, DLL_PROCESS_ATTACH, nullptr);

void TestHardware()
{
    
    // DWORD EnumACPIBufferSize   = 0;
    // DWORD EnumSMBIOSBufferSize = 0;
    // DWORD EnumFIRMBufferSize   = 0;

    // EnumACPIBufferSize   = EnumSystemFirmwareTables('ACPI', nullptr, EnumACPIBufferSize);
    // EnumSMBIOSBufferSize = EnumSystemFirmwareTables('RSMB', nullptr, EnumSMBIOSBufferSize);
    // EnumFIRMBufferSize   = EnumSystemFirmwareTables('FIRM', nullptr, EnumFIRMBufferSize);

    // printf("EnumACPIBufferSize: %d\nEnumSMBIOSBufferSize: %d\nEnumFIRMBufferSize: %d\n", EnumACPIBufferSize, EnumSMBIOSBufferSize, EnumFIRMBufferSize);

    // const PVOID EnumACPIBuffer = malloc(EnumACPIBufferSize);
    // memset(EnumACPIBuffer, 0, EnumACPIBufferSize);
    // EnumSystemFirmwareTables('ACPI', EnumACPIBuffer, EnumACPIBufferSize);

    // const PVOID EnumRSMBBuffer = malloc(EnumSMBIOSBufferSize);
    // memset(EnumRSMBBuffer, 0, EnumSMBIOSBufferSize);
    // EnumSystemFirmwareTables('ACPI', EnumRSMBBuffer, EnumSMBIOSBufferSize);

    // const PVOID EnumFIRMBuffer = malloc(EnumFIRMBufferSize);
    // memset(EnumFIRMBuffer, 0, EnumFIRMBufferSize);
    // EnumSystemFirmwareTables('ACPI', EnumFIRMBuffer, EnumFIRMBufferSize);

    // RawSMBIOSData testmssmbiosbin;

    // PVOID ACPIBuffer       = nullptr;
    // PVOID RSMBBuffer       = nullptr;
    // PVOID FIRMBuffer       = nullptr;

    // DWORD ACPIBufferSize   = 0;
    // DWORD SMBIOSBufferSize = 0;
    // DWORD FIRMBufferSize   = 0;

    // ACPIBufferSize   = GetSystemFirmwareTable('ACPI', 'TDSS', nullptr, ACPIBufferSize);
    // SMBIOSBufferSize = GetSystemFirmwareTable('RSMB', 0x00, nullptr, SMBIOSBufferSize);
    // FIRMBufferSize   = GetSystemFirmwareTable('FIRM', 0x00, nullptr, FIRMBufferSize);

    // printf("ACPIBufferSize: %d\nSMBIOSBufferSize: %d\nFIRMBufferSize: %d\n", ACPIBufferSize, SMBIOSBufferSize, FIRMBufferSize);
    // printf("test: %d\n", GetSystemFirmwareTable('ACPI', 'TDSD', nullptr, 0));
    //// printf("%s", EnumACPIBuffer);
    // free(EnumACPIBuffer);
    // free(EnumRSMBBuffer);
    // free(EnumFIRMBuffer);

    // ACPITest();
    // RSMBTest();

    ////// void*           ptr;
    ////// const size_type arg_alloc_size = 80;
    ////// cudaError_t err = cudaMallocManaged(&ptr, arg_alloc_size, cudaMemAttachGlobal);
}
