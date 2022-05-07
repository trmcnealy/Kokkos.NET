

#include <Types.hpp>
#include <Kokkos_Core.hpp>

#define ENABLE_DETOUR_ABORT
#include <ErrorFuncs.hpp>
#undef ENABLE_DETOUR_ABORT

#include <exception>
#include <string>

#include <windows.h>

BOOL WINAPI DllMain(HANDLE hModule, const DWORD ul_reason_for_call, LPVOID lpReserved)
{
    switch (ul_reason_for_call)
    {
        case DLL_PROCESS_ATTACH:
        {
            DisableThreadLibraryCalls(static_cast<HINSTANCE>(hModule));

            if (std::get_terminate() == nullptr)
            {
                _set_abort_behavior(0, _WRITE_ABORT_MSG);
                std::set_terminate(onTerminate);
            }
            break;
        }
        case DLL_PROCESS_DETACH:
        {
            break;
        }
        case DLL_THREAD_ATTACH:
        case DLL_THREAD_DETACH:
        default:
            break;
    }
    return TRUE;
}

//.def
// LIBRARY
//
// EXPORTS
//    DllGetActivationFactory PRIVATE
//    DllGetClassObject       PRIVATE
//    DllCanUnloadNow         PRIVATE

//#include <wrl\module.h>
//
// using namespace Microsoft::WRL;
//
//#if !defined(__WRL_CLASSIC_COM__)
// EXTERN_C HRESULT __stdcall DllGetActivationFactory(_In_ HSTRING activatibleClassId, IActivationFactory** factory)
//{
//    return Module<InProc>::GetModule().GetActivationFactory(activatibleClassId, factory);
//}
//#endif
//
//#if !defined(__WRL_WINRT_STRICT__)
// EXTERN_C HRESULT __stdcall DllGetClassObject(REFCLSID rclsid, REFIID riid, void** ppv)
//{
//    return Module<InProc>::GetModule().GetClassObject(rclsid, riid, ppv);
//}
//#endif
//
// EXTERN_C HRESULT __stdcall DllCanUnloadNow()
//{
//    return Module<InProc>::GetModule().Terminate() ? S_OK : S_FALSE;
//}
