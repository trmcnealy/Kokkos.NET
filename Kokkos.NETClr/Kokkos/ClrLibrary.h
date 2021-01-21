#pragma once

#pragma unmanaged

//#include <Types.hpp>

//#include <Kokkos_Core.hpp>

#include <string>

#include <D:/TFS_Sources/Github/Compilation/trmcnealy/Kokkos.NET/Kokkos.NETClr/Kokkos/Native.h>

#pragma managed

using namespace System;

//template<class NativeType>
//public ref class ManagedObject
//{
//protected:
//    NativeType* _instance;
//
//public:
//    ManagedObject() : _instance(new NativeType()) {}
//
//    template<class... Types>
//    ManagedObject(Types... types) : _instance(new NativeType(types...))
//    {
//    }
//
//    ManagedObject(void* instance) : _instance(reinterpret_cast<NativeType*>(instance)) {}
//
//    ManagedObject(NativeType* instance) : _instance(instance) {}
//
//    ManagedObject(NativeType& instance) : _instance(&instance) {}
//
//    virtual ~ManagedObject()
//    {
//        if (_instance != nullptr)
//        {
//            delete _instance;
//        }
//    }
//
//    !ManagedObject()
//    {
//        if (_instance != nullptr)
//        {
//            delete _instance;
//        }
//    }
//
//    NativeType* GetPointer()
//    {
//        return _instance;
//    }
//
//    NativeType& GetRawRef()
//    {
//        return (*_instance);
//    }
//};


namespace ManagedKokkos{

    public ref class Serial abstract sealed
    {

        static String^ ToString(double value)
        {
            double result = ieee754_atanh(value);

            return gcnew String(std::to_string(result).c_str());
        }
    };

}








//namespace ManagedKokkos
//{
//public
//    enum class LayoutKind : int
//    {
//        Unknown = UInt16::MaxValue,
//        Left    = 0,
//        Right,
//        Stride
//    };
//
//public
//    interface class ILayout
//    {
//    };
//
//    public value struct LayoutLeft : ILayout
//    {
//    };
//
//    public value struct LayoutRight : ILayout
//    {
//    };
//
//    public value struct LayoutStride : ILayout
//    {
//    };
//
//public
//    enum class ExecutionSpaceKind : int
//    {
//        Unknown = UInt16::MaxValue,
//        Serial  = 0,
//        OpenMP,
//        Cuda
//    };
//
//public
//    interface class IExecutionSpace
//    {
//        property LayoutKind DefaultLayout
//        {
//            LayoutKind get();
//        }
//    };
//
//    public value struct Cuda : IExecutionSpace
//    {
//    public:
//        property LayoutKind DefaultLayout
//        {
//            virtual LayoutKind get() { return LayoutKind::Left; }
//        }
//
//        String^ ToString() override
//        {
//            return "Cuda";
//        }
//    };
//
//    public value struct Serial : IExecutionSpace
//    {
//    public:
//        
//        property LayoutKind DefaultLayout
//        {
//            virtual LayoutKind get() { return LayoutKind::Right; }
//        }
//
//        String^ ToString() override
//        {
//            return "Serial";
//        }
//    };
//
//    public value struct OpenMP : IExecutionSpace
//    {
//    public:
//        
//        property LayoutKind DefaultLayout
//        {
//            virtual LayoutKind get() { return LayoutKind::Right; }
//        }
//
//        String^ ToString() override
//        {
//            return "OpenMP";
//        }
//    };
//
//    generic<typename T> where T : IExecutionSpace
//    public ref class ExecutionSpace sealed abstract
//    {
//
//    public:
//        static ExecutionSpaceKind GetKind()
//        {
//            if (T::typeid == Serial::typeid)
//            {
//                return ExecutionSpaceKind::Serial;
//            }
//
//            if (T::typeid == OpenMP::typeid)
//            {
//                return ExecutionSpaceKind::OpenMP;
//            }
//
//            if (T::typeid == Cuda::typeid)
//            {
//                return ExecutionSpaceKind::Cuda;
//            }
//
//            return ExecutionSpaceKind::Unknown;
//        }
//
//        static LayoutKind ^ GetLayout()
//        {
//            if (T::typeid == Serial::typeid)
//            {
//                return LayoutKind::Right;
//            }
//
//            if (T::typeid == OpenMP::typeid)
//            {
//                return LayoutKind::Right;
//            }
//
//            if (T::typeid == Cuda::typeid)
//            {
//                return LayoutKind::Left;
//            }
//
//            return LayoutKind::Unknown;
//        }
//    };
//
//
//    template<typename T>
//    value struct ToTrait;
//    
//    //template<>
//    //value struct ToTrait<ExecutionSpaceKind::Serial>
//    //{
//    //    using ExecutionSpace = Kokkos::Serial;
//    //};
//
//    template<>
//    value struct ToTrait<ManagedKokkos::Serial>
//    {
//        using Value = Kokkos::Serial;
//    };
//
//    //template<>
//    //value struct ToTrait<ExecutionSpaceKind::OpenMP>
//    //{
//    //    using ExecutionSpace = Kokkos::OpenMP;
//    //};
//
//    template<>
//    value struct ToTrait<ManagedKokkos::OpenMP>
//    {
//        using Value = Kokkos::OpenMP;
//    };
//
//    //template<>
//    //value struct ToTrait<ExecutionSpaceKind::Cuda>
//    //{
//    //    using ExecutionSpace = Kokkos::Cuda;
//    //};
//
//    template<>
//    value struct ToTrait<ManagedKokkos::Cuda>
//    {
//        using Value = Kokkos::Cuda;
//    };
//
//
//
//
//
//
//    template<> value struct ToTrait<System::Single>
//    {
//        using Value = float;
//    };
//
//    template<> value struct ToTrait<System::Double>
//    {
//        using Value = Kokkos::Cuda;
//    };
//
//    template<> value struct ToTrait<System::Boolean>
//    {
//        using Value = bool;
//    };
//
//    template<> value struct ToTrait<System::SByte>
//    {
//        using Value = int8;
//    };
//
//    template<> value struct ToTrait<System::Byte>
//    {
//        using Value = uint8;
//    };
//
//    template<> value struct ToTrait<System::Int16>
//    {
//        using Value = int16;
//    };
//
//    template<> value struct ToTrait<System::UInt16>
//    {
//        using Value = uint16;
//    };
//
//    template<> value struct ToTrait<System::Int32>
//    {
//        using Value = int32;
//    };
//
//    template<> value struct ToTrait<System::UInt32>
//    {
//        using Value = uint32;
//    };
//
//    template<> value struct ToTrait<System::Int64>
//    {
//        using Value = int64;
//    };
//    
//    template<> value struct ToTrait<System::UInt64>
//    {
//        using Value = uint64;
//    };
//
//    template<> value struct ToTrait<System::Char>
//    {
//        using Value = wchar_t;
//    };
//
//
//
//    template<typename TDataType, typename TExecutionSpace>
//        //where TDataType : value struct
//        //where TExecutionSpace : IExecutionSpace, gcnew()
//    public ref class KokkosView : public ManagedObject<Kokkos::View<typename ToTrait<TDataType>::Value, typename ToTrait<TExecutionSpace>::Value>>
//    {
//    };
//
//}

//#include <Kokkos_Core.hpp>
//#include <Teuchos_RCP.hpp>
//
// namespace Kokkos
//{
//
//    template<class NativeType, class ExecutionSpace>
//    public ref class KokkosManagedObject
//    {
//    protected:
//        Kokkos::RCP<NativeType> _instance;
//
//        using KokkosMalloc = Kokkos::kokkos_malloc<ExecutionSpace::memory_space>;
//        using KokkosFree   = Kokkos::kokkos_free<ExecutionSpace::memory_space>;
//
//    public:
//        KokkosManagedObject()
//        {
//            _instance = KokkosMalloc(sizeof(NativeType));
//
//            new (instance) NativeType();
//        }
//
//        template<class... Types>
//        KokkosManagedObject(Types... types)
//        {
//            _instance = KokkosMalloc(sizeof(NativeType));
//
//            new (_instance) NativeType(types...);
//        }
//
//        KokkosManagedObject(void* instance) : _instance(Kokkos::RCP<NativeType>(instance)) {}
//
//        KokkosManagedObject(NativeType* instance) : _instance(Kokkos::RCP<NativeType>(instance)) {}
//
//        KokkosManagedObject(NativeType& instance) : _instance(Kokkos::rcpFromRef<NativeType>(instance)) {}
//
//        virtual ~KokkosManagedObject()
//        {
//            if (_instance != nullptr)
//            {
//                KokkosFree(_instance);
//            }
//        }
//
//        !KokkosManagedObject()
//        {
//            if (_instance != nullptr)
//            {
//                KokkosFree(_instance);
//            }
//        }
//
//        Kokkos::RCP<NativeType>& GetInstance()
//        {
//            return _instance;
//        }
//
//        NativeType* GetPointer()
//        {
//            return (_instance.get());
//        }
//
//        NativeType* GetRawPointer()
//        {
//            return (_instance.getRawPtr());
//        }
//
//        NativeType& GetRef()
//        {
//            return (*_instance);
//        }
//    };
//}
//
//
// namespace Kokkos
//{
//    template<class NativeType>
//    public ref class TeuchosManagedObject
//    {
//    protected:
//        Teuchos::RCP<NativeType> _instance;
//
//    public:
//        TeuchosManagedObject() : _instance(Teuchos::RCP<NativeType>(new NativeType())) {}
//
//        template<class... Types>
//        TeuchosManagedObject(Types... types) : _instance(Teuchos::RCP<NativeType>(new NativeType(types...)))
//        {
//        }
//
//        TeuchosManagedObject(void* instance) : _instance(Teuchos::RCP<NativeType>(instance)) {}
//
//        TeuchosManagedObject(NativeType* instance) : _instance(Teuchos::RCP<NativeType>(instance)) {}
//
//        TeuchosManagedObject(NativeType& instance) : _instance(Teuchos::rcpFromRef<NativeType>(instance)) {}
//
//        virtual ~TeuchosManagedObject()
//        {
//            if (_instance != nullptr)
//            {
//                delete _instance.getRawPtr();
//
//                _instance.~RCP();
//            }
//        }
//
//        !TeuchosManagedObject()
//        {
//            if (_instance != nullptr)
//            {
//                delete _instance.getRawPtr();
//
//                _instance.~RCP();
//            }
//        }
//
//        Teuchos::RCP<NativeType>& GetInstance()
//        {
//            return _instance;
//        }
//
//        NativeType* GetPointer()
//        {
//            return (_instance.get());
//        }
//
//        NativeType* GetRawPointer()
//        {
//            return (_instance.getRawPtr());
//        }
//
//        NativeType& GetRef()
//        {
//            return (*_instance);
//        }
//    };
//
//}
