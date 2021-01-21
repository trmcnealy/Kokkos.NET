#pragma once

#include "runtime.Kokkos/ViewTypes.hpp"

namespace Kokkos
{

    MAKE_UID_INTERFACE_T(IVector, "D21E4706-D308-448A-B776-35EDADF609E2", template<typename TDataType, typename TExecustionSpace>)
    KOKKOS_NET_API_EXPORT IVector
    {
        virtual ~IVector() = 0;

        virtual void sync_host()        = 0;
        virtual void sync_device()      = 0;
        virtual bool need_sync_host()   = 0;
        virtual bool need_sync_device() = 0;

        virtual void       assign(uint64 n, const TDataType& value)                                          = 0;
        virtual TDataType& back()                                                                            = 0;
        virtual TDataType* begin()                                                                           = 0;
        virtual void       clear()                                                                           = 0;
        virtual TDataType* data()                                                                            = 0;
        virtual void       device_to_host()                                                                  = 0;
        virtual bool       empty()                                                                           = 0;
        virtual TDataType* end()                                                                             = 0;
        virtual TDataType* find(TDataType value)                                                             = 0;
        virtual TDataType& front()                                                                           = 0;
        virtual void       host_to_device()                                                                  = 0;
        virtual TDataType* insert(TDataType * it, const TDataType& value)                                    = 0;
        virtual TDataType* insert(TDataType * it, uint64 count, const TDataType& value)                      = 0;
        virtual bool       is_allocated()                                                                    = 0;
        virtual bool       is_sorted()                                                                       = 0;
        virtual uint64     lower_bound(const uint64& start, const uint64& theEnd, const TDataType& comp_val) = 0;
        virtual uint64     max_size()                                                                        = 0;
        virtual void       on_device()                                                                       = 0;
        virtual void       on_host()                                                                         = 0;
        virtual void       pop_back()                                                                        = 0;
        virtual void       push_back(TDataType value)                                                        = 0;
        virtual void       reserve(uint64 n)                                                                 = 0;
        virtual void       resize(uint64 n)                                                                  = 0;
        virtual void       resize(uint64 n, const TDataType& value)                                          = 0;
        virtual void       set_overallocation(float extra)                                                   = 0;
        virtual uint64     size()                                                                            = 0;
        virtual uint64     span()                                                                            = 0;
    };

}

// namespace Microsoft
//{
//    namespace WRL
//    {
//
//        enum RuntimeClassType
//        {
//            WinRt                   = 0x0001,
//            ClassicCom              = 0x0002,
//            WinRtClassicComMix      = WinRt | ClassicCom,
//            InhibitWeakReference    = 0x0004,
//            Delegate                = ClassicCom,
//            InhibitFtmBase          = 0x0008,
//            InhibitRoOriginateError = 0x0010
//        };
//
//        template<unsigned int flags>
//        struct RuntimeClassFlags
//        {
//            static const unsigned int value = flags;
//        };
//
//        template<class RuntimeClassFlagsT, bool implementsWeakReferenceSource, bool implementsInspectable, bool implementsFtmBase, typename... TInterfaces>
//        class __declspec(novtable) RuntimeClassImpl;
//
//        struct __declspec(uuid("00000000-0000-0000-C000-000000000046")) __declspec(novtable) IUnknown
//        {
//        public:
//            virtual int32 __cdecl QueryInterface(const GUID* const riid, void** ppvObject) = 0;
//
//            virtual uint32 __cdecl AddRef() = 0;
//
//            virtual uint32 __cdecl Release() = 0;
//        };
//
//        struct __declspec(uuid("b63ea76d-1f85-456f-a19c-48159efa858b")) __declspec(novtable) IShellItemArray : public IUnknown
//        {
//        public:
//        };
//
//        struct __declspec(uuid("a08ce4d0-fa25-44ab-b57c-c7b1c323e0b9")) __declspec(novtable) IExplorerCommand : public IUnknown
//        {
//        public:
//            virtual void* __cdecl GetToolTip(IShellItemArray* psiItemArray, wchar_t** ppszInfotip) = 0;
//        };
//
//#undef throw
//
//        template<class RuntimeClassFlagsT, bool implementsWeakReferenceSource, bool implementsFtmBase, typename... TInterfaces>
//        class __declspec(novtable) RuntimeClassImpl
//        {
//        private:
//            template<typename T>
//            static bool CanCastTo(T* ptr, REFIID riid, void** ppv) noexcept
//            {
//                if (InlineIsEqualGUID(riid, __uuidof(Base)))
//                {
//                    *ppv = static_cast<Base*>(ptr);
//                    return true;
//                }
//
//                return false;
//            }
//
//        public:
//            void QueryInterface(REFIID riid, void** ppvObject)
//            {
//
//                *ppvObject          = nullptr;
//                bool isRefDelegated = false;
//
//                if (InlineIsEqualGUID(riid, __uuidof(IUnknown)) || ((RuntimeClassTypeT & WinRt) != 0 && InlineIsEqualGUID(riid, __uuidof(IInspectable))))
//                {
//                    *ppvObject = implements->CastToUnknown();
//
//                    static_cast<IUnknown*>(*ppvObject)->AddRef();
//
//                    return S_OK;
//                }
//
//                uint32 hr = implements->CanCastTo(riid, ppvObject, &isRefDelegated);
//
//                if (hr == 0 && !isRefDelegated)
//                {
//                    static_cast<IUnknown*>(*ppvObject)->AddRef();
//                }
//
//                return hr;
//
//                return AsIID(this, riid, ppvObject);
//            }
//
//            uint32 AddRef()
//            {
//                return 0;
//            }
//
//            uint32 Release()
//            {
//                return 0;
//            }
//
//        protected:
//            RuntimeClassImpl() noexcept {}
//
//            virtual ~RuntimeClassImpl() noexcept {}
//        };
//
//        template<typename ILst,
//                 class RuntimeClassFlagsT,
//                 bool implementsWeakReferenceSource = (RuntimeClassFlagsT::value & InhibitWeakReference) == 0,
//                 bool implementsInspectable         = (RuntimeClassFlagsT::value & WinRt) == WinRt,
//                 bool implementsFtmBase             = RuntimeClassFlagsT::value>
//        class RuntimeClass;
//
//        template<typename... TInterfaces>
//        class RuntimeClass : public RuntimeClassImpl<RuntimeClassFlags<WinRt>,
//                                    (RuntimeClassFlags<WinRt>::value & InhibitWeakReference) == 0,
//                                    (RuntimeClassFlags<WinRt>::value & WinRt) == WinRt,
//                                    RuntimeClassFlags<WinRt>::value,
//                                    TInterfaces...>
//        {
//            RuntimeClass(const RuntimeClass&) = default;
//            RuntimeClass& operator=(const RuntimeClass&) = default;
//
//        protected:
//            uint32 CustomQueryInterface(REFIID /*riid*/, void** /*ppvObject*/, bool* handled)
//            {
//                *handled = false;
//                return 0;
//            }
//
//        public:
//            RuntimeClass() noexcept
//            {
//                auto modulePtr = ::Microsoft::WRL::GetModuleBase();
//
//                if (modulePtr != nullptr)
//                {
//                    modulePtr->IncrementObjectCount();
//                }
//            }
//            typedef RuntimeClass RuntimeClassT;
//        };
//
//    }
//}
//
// struct __declspec(uuid("9f156763-7844-4dc4-b2b1-901f640f5155")) OpenTerminalHere :
//    public Microsoft::WRL::RuntimeClass<Microsoft::WRL::RuntimeClassFlags<Microsoft::WRL::ClassicCom | Microsoft::WRL::InhibitFtmBase>, Microsoft::WRL::IExplorerCommand>
//{
//};
//
//
//
// CoCreatableClass(OpenTerminalHere);
