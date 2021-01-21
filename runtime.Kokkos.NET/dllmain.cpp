

#include <Types.hpp>
#include <Kokkos_Core.hpp>

#include <exception>
#include <string>

#include <windows.h>

[[noreturn]] void onTerminate() noexcept;

[[noreturn]] void onatexit() noexcept;

BOOL WINAPI DllMain(HANDLE hModule, const DWORD ul_reason_for_call, LPVOID lpReserved)
{
    switch (ul_reason_for_call)
    {
        case DLL_PROCESS_ATTACH:
        {
            DisableThreadLibraryCalls(static_cast<HINSTANCE>(hModule));

            std::set_terminate(&onTerminate);
            //atexit(&onatexit);
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

[[noreturn]] void onTerminate() noexcept
{
#if defined(KOKKOS_ENABLE_CUDA)
    const cudaError_t cuda_error = cudaGetLastError();

    if (cuda_error != 0)
    {
        std::cerr << cudaGetErrorString(cuda_error);
    }
#endif

    if (const std::exception_ptr exc = std::current_exception())
    {
        try
        {
            std::rethrow_exception(exc);
        }
        catch (const std::exception& exp)
        {
            std::cerr << exp.what() << std::endl;
        }
        catch (const std::logic_error& exp)
        {
            std::cerr << exp.what() << std::endl;
        }
        catch (const std::runtime_error& exp)
        {
            std::cerr << exp.what() << std::endl;
        }
        catch (const std::bad_exception& exp)
        {
            std::cerr << exp.what() << std::endl;
        }
        catch (...)
        {
            std::cerr << "Caught unknown exception." << std::endl;
        }
    }

    if(IsDebuggerPresent())
    {
        __debugbreak();
    }

    if (Kokkos::is_initialized())
    {
        Kokkos::finalize_all();
    }

    std::_Exit(EXIT_FAILURE);
}

[[noreturn]] void onatexit() noexcept
{
    if(IsDebuggerPresent())
    {
        __debugbreak();
    }
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
