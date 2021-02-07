#pragma once

#ifndef _WIN32
#define _WIN32  // NOLINT(clang-diagnostic-reserved-id-macro)
#endif

#include <atomic>
#include <comdef.h>

template <COINIT TM>
struct COMController
{
    const HRESULT Result;

    COMController() : Result{CoInitializeEx(nullptr, TM)} {}

    ~COMController()
    {
        if (SUCCEEDED(Result))
        {
            CoUninitialize();
        }
    }
};

using COM_MT = COMController<COINIT_MULTITHREADED>;

struct CoreShim_COMActivation
{
    CoreShim_COMActivation(const wchar_t* assemblyName, const wchar_t* typeName) { Set(assemblyName, typeName); }

    ~CoreShim_COMActivation() { Set(nullptr, nullptr); }

private:
    void Set(const wchar_t* assemblyName, const wchar_t* typeName)
    {
        SetEnvironmentVariableW(__CRT_WIDE("CORESHIM_COMACT_ASSEMBLYNAME"), assemblyName);
        SetEnvironmentVariableW(__CRT_WIDE("CORESHIM_COMACT_TYPENAME"), typeName);
    }
};

// class DECLSPEC_UUID("53169A33-E85D-4E3C-B668-24E438D0929B") NumericTesting;
//#define CLSID_NumericTesting __uuidof(NumericTesting)
//#define IID_INumericTesting  __uuidof(INumericTesting)

derived_class UnknownBase : public IUnknown
{
    std::atomic<unsigned int> _ref_count = 1;

public:
    constexpr UnknownBase() = default;
    virtual ~UnknownBase()  = default;

    constexpr UnknownBase(const UnknownBase&) = delete;
    constexpr UnknownBase& operator=(const UnknownBase&) = delete;
    constexpr UnknownBase(UnknownBase&&)                 = delete;
    constexpr UnknownBase& operator=(UnknownBase&&) = delete;

    template <typename I>
    static HRESULT __QueryInterfaceImpl(
        /* [in] */ REFIID                     riid,
        /* [iid_is][out] */ _COM_Outptr_ void __RPC_FAR* __RPC_FAR* ppvObject,
        /* [in] */ I                                                obj)
    {
        if (riid == __uuidof(I))
        {
            *ppvObject = static_cast<I>(obj);
        }
        else
        {
            *ppvObject = nullptr;
            return E_NOINTERFACE;
        }

        return S_OK;
    }

    template <typename I1, typename... IR>
    static HRESULT __QueryInterfaceImpl(
        /* [in] */ REFIID                     riid,
        /* [iid_is][out] */ _COM_Outptr_ void __RPC_FAR* __RPC_FAR* ppvObject,
        /* [in] */ I1                                               i1,
        /* [in] */ IR... remain)
    {
        if (riid == __mingw_uuidof<decltype(I1)>())
        {
            *ppvObject = static_cast<I1>(i1);
            return S_OK;
        }

        return __QueryInterfaceImpl(riid, ppvObject, remain...);
    }

    template <typename I1, typename... IR>
    HRESULT DoQueryInterface(
        /* [in] */ REFIID                       riid,
        /* [iid_is][out] */ _COM_Outptr_ void** ppvObject,
        /* [in] */ I1                           i1,
        /* [in] */ IR... remain)
    {
        if (ppvObject == nullptr)
        {
            return E_POINTER;
        }

        if (riid == __uuidof(IUnknown))
        {
            *ppvObject = static_cast<IUnknown*>(i1);
        }
        else
        {
            HRESULT hr = Internal::__QueryInterfaceImpl(riid, ppvObject, i1, remain...);
            if (hr != S_OK)
            {
                return hr;
            }
        }

        DoAddRef();
        return S_OK;
    }

    constexpr unsigned int DoAddRef() override { return (++_ref_count); }

    constexpr unsigned int DoRelease() override
    {
        unsigned int c = (--_ref_count);
        if (c == 0)
        {
            delete this;
        }
        return c;
    }
};
