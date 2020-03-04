
#include <windows.h>
//#include <memory.h>
//
//#define SHMEMSIZE 4096
//
//static LPVOID lpvMem     = nullptr; // pointer to shared memory
//static HANDLE hMapObject = nullptr; // handle to file mapping

BOOL WINAPI DllMain(HANDLE hModule, const DWORD ul_reason_for_call, LPVOID lpReserved)
{
    switch(ul_reason_for_call)
    {
        case DLL_PROCESS_ATTACH:
        {
            DisableThreadLibraryCalls(static_cast<HINSTANCE>(hModule));

            //// Create a named file mapping object
            //hMapObject = CreateFileMapping(INVALID_HANDLE_VALUE, // use paging file
            //                               nullptr, // default security attributes
            //                               PAGE_READWRITE, // read/write access
            //                               0, // size: high 32-bits
            //                               SHMEMSIZE, // size: low 32-bits
            //                               TEXT("dllmemfilemap")); // name of map object

            //if(hMapObject == nullptr)
            //{
            //    return FALSE;
            //}

            //// The first process to attach initializes memory
            //const BOOL fInit = (GetLastError() != ERROR_ALREADY_EXISTS);

            //// Get a pointer to the file-mapped shared memory
            //lpvMem = MapViewOfFile(hMapObject, // object to map view of
            //                       FILE_MAP_WRITE, // read/write access
            //                       0, // high offset:  map from
            //                       0, // low offset:   beginning
            //                       0); // default: map entire file

            //if(lpvMem == nullptr)
            //{
            //    return FALSE;
            //}

            //// Initialize memory if this is the first process
            //if(fInit)
            //{
            //    memset(lpvMem, '\0', SHMEMSIZE);
            //}

            break;
        }
        case DLL_PROCESS_DETACH:
        {
            //// Unmap shared memory from the process's address space
            //BOOL fIgnore = UnmapViewOfFile(lpvMem);

            //// Close the process's handle to the file-mapping object
            //fIgnore = CloseHandle(hMapObject);

            break;
        }
        case DLL_THREAD_ATTACH:
        case DLL_THREAD_DETACH:
        default: break;
    }
    return TRUE;
}

//#ifdef __cplusplus
//extern "C"
//{
//#endif
//
//    // SetSharedMem sets the contents of the shared memory
//    __declspec(dllexport) void __cdecl SetSharedMem(LPWSTR lpszBuf)
//    {
//        DWORD dwCount = 1;
//
//        // Get the address of the shared memory block
//        LPWSTR lpszTmp = (LPWSTR)lpvMem;
//
//        // Copy the null-terminated string into shared memory
//        while(*lpszBuf && dwCount < SHMEMSIZE)
//        {
//            *lpszTmp++ = *lpszBuf++;
//            dwCount++;
//        }
//        *lpszTmp = '\0';
//    }
//
//    // GetSharedMem gets the contents of the shared memory
//    __declspec(dllexport) void __cdecl GetSharedMem(LPWSTR lpszBuf, DWORD cchSize)
//    {
//        // Get the address of the shared memory block
//        LPWSTR lpszTmp = (LPWSTR)lpvMem;
//
//        // Copy from shared memory into the caller's buffer
//        while(*lpszTmp && --cchSize)
//        {
//            *lpszBuf++ = *lpszTmp++;
//        }
//
//        *lpszBuf = '\0';
//    }
//
//#ifdef __cplusplus
//}
//#endif
