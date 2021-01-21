#pragma once

#include "runtime.Kokkos/ViewTypes.hpp"

//#include <fileapi.h>
#include <handleapi.h>

// template<typename DataType, class ExecutionSpace, class Layout = typename ExecutionSpace::array_layout>
// using Vector = View<DataType*, Layout, ExecutionSpace>;

namespace Kokkos
{
    template<class ExecutionSpace>
    struct InterprocessMemory;
}

KOKKOS_NET_API_EXTERNC void* IpcCreate(REF(ExecutionSpaceKind) execution_space, REF(size_type) size, REF(NativeString) label) noexcept;

KOKKOS_NET_API_EXTERNC void* IpcCreateFrom(REF(ExecutionSpaceKind) execution_space, void* memoryPtr, REF(size_type) size, REF(NativeString) label) noexcept;

KOKKOS_NET_API_EXTERNC void* IpcOpenExisting(REF(ExecutionSpaceKind) execution_space, void* instance) noexcept;

KOKKOS_NET_API_EXTERNC void IpcDestory(REF(ExecutionSpaceKind) execution_space, void* instance) noexcept;

KOKKOS_NET_API_EXTERNC void IpcClose(REF(ExecutionSpaceKind) execution_space, void* instance) noexcept;

KOKKOS_NET_API_EXTERNC void* IpcGetMemoryPointer(REF(ExecutionSpaceKind) execution_space, void* instance) noexcept;

KOKKOS_NET_API_EXTERNC void* IpcGetDeviceHandle(REF(ExecutionSpaceKind) execution_space, void* instance) noexcept;

KOKKOS_NET_API_EXTERNC size_type IpcGetSize(REF(ExecutionSpaceKind) execution_space, void* instance) noexcept;

KOKKOS_NET_API_EXTERNC void* IpcMakeViewFromPointer(REF(ExecutionSpaceKind) execution_space,
                                                    REF(DataTypeKind) data_type,
                                                    void* instance,
                                                    REF(size_type) arg_N0 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
                                                    REF(size_type) arg_N1 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
                                                    REF(size_type) arg_N2 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
                                                    REF(size_type) arg_N3 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
                                                    REF(size_type) arg_N4 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
                                                    REF(size_type) arg_N5 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
                                                    REF(size_type) arg_N6 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
                                                    REF(size_type) arg_N7 = KOKKOS_IMPL_CTOR_DEFAULT_ARG) noexcept;

KOKKOS_NET_API_EXTERNC void* IpcMakeViewFromHandle(REF(ExecutionSpaceKind) execution_space,
                                                   REF(DataTypeKind) data_type,
                                                   void* instance,
                                                   REF(size_type) arg_N0 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
                                                   REF(size_type) arg_N1 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
                                                   REF(size_type) arg_N2 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
                                                   REF(size_type) arg_N3 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
                                                   REF(size_type) arg_N4 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
                                                   REF(size_type) arg_N5 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
                                                   REF(size_type) arg_N6 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
                                                   REF(size_type) arg_N7 = KOKKOS_IMPL_CTOR_DEFAULT_ARG) noexcept;

namespace Kokkos
{
    KOKKOS_INLINE_FUNCTION static size_type GetRank(REF(size_type) arg_N0 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
                                                    REF(size_type) arg_N1 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
                                                    REF(size_type) arg_N2 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
                                                    REF(size_type) arg_N3 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
                                                    REF(size_type) arg_N4 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
                                                    REF(size_type) arg_N5 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
                                                    REF(size_type) arg_N6 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
                                                    REF(size_type) arg_N7 = KOKKOS_IMPL_CTOR_DEFAULT_ARG)
    {
        if (arg_N0 == KOKKOS_IMPL_CTOR_DEFAULT_ARG)
        {
            return 0;
        }
        if (arg_N1 == KOKKOS_IMPL_CTOR_DEFAULT_ARG)
        {
            return 1;
        }
        if (arg_N2 == KOKKOS_IMPL_CTOR_DEFAULT_ARG)
        {
            return 2;
        }
        if (arg_N3 == KOKKOS_IMPL_CTOR_DEFAULT_ARG)
        {
            return 3;
        }
        if (arg_N4 == KOKKOS_IMPL_CTOR_DEFAULT_ARG)
        {
            return 4;
        }
        if (arg_N5 == KOKKOS_IMPL_CTOR_DEFAULT_ARG)
        {
            return 5;
        }
        if (arg_N6 == KOKKOS_IMPL_CTOR_DEFAULT_ARG)
        {
            return 6;
        }
        if (arg_N7 == KOKKOS_IMPL_CTOR_DEFAULT_ARG)
        {
            return 7;
        }
        return 8;
    }

    template<>
    struct InterprocessMemory<Kokkos::Serial>
    {
    private:
        void*       _memoryPtr;
        void*       _memoryHandle;
        size_type   _size;
        const char* _label;

    public:
        explicit InterprocessMemory(void* memory_ptr, void* memory_handle, REF(size_type) size, const char* label) :
            _memoryPtr(memory_ptr),
            _memoryHandle(memory_handle),
            _size(size),
            _label(label)
        {
        }

        ~InterprocessMemory()
        {
            IpcDestory(ExecutionSpaceKind::Serial, this);
        }

        [[nodiscard]] __inline __host__ void* GetMemoryPointer() const
        {
            return _memoryPtr;
        }

        [[nodiscard]] __inline __host__ void* GetDeviceHandle() const
        {
            return _memoryHandle;
        }

        [[nodiscard]] __inline __host__ size_type GetSize() const
        {
            return _size;
        }

        [[nodiscard]] __inline __host__ const char* GetLabel() const
        {
            return _label;
        }

        template<typename DataType, class Layout = Kokkos::Serial::array_layout>
        [[nodiscard]] __inline __host__ void* MakeViewFromPointer(REF(size_type) arg_N0 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
                                                                  REF(size_type) arg_N1 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
                                                                  REF(size_type) arg_N2 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
                                                                  REF(size_type) arg_N3 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
                                                                  REF(size_type) arg_N4 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
                                                                  REF(size_type) arg_N5 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
                                                                  REF(size_type) arg_N6 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
                                                                  REF(size_type) arg_N7 = KOKKOS_IMPL_CTOR_DEFAULT_ARG) const
        {
            if (arg_N7 != KOKKOS_IMPL_CTOR_DEFAULT_ARG)
            {
                auto view = new View<DataType********, Layout, Kokkos::Serial>(Kokkos::ViewAllocateWithoutInitializing(_label),
                                                                               arg_N0,
                                                                               arg_N1,
                                                                               arg_N2,
                                                                               arg_N3,
                                                                               arg_N4,
                                                                               arg_N5,
                                                                               arg_N6,
                                                                               arg_N7);
                view->assign_data(reinterpret_cast<DataType*>(_memoryPtr));
                return view;
            }
            if (arg_N6 != KOKKOS_IMPL_CTOR_DEFAULT_ARG)
            {
                auto view = new View<DataType*******, Layout, Kokkos::Serial>(Kokkos::ViewAllocateWithoutInitializing(_label),
                                                                              arg_N0,
                                                                              arg_N1,
                                                                              arg_N2,
                                                                              arg_N3,
                                                                              arg_N4,
                                                                              arg_N5,
                                                                              arg_N6);
                view->assign_data(reinterpret_cast<DataType*>(_memoryPtr));
                return view;
            }
            if (arg_N5 != KOKKOS_IMPL_CTOR_DEFAULT_ARG)
            {
                auto view = new View<DataType******, Layout, Kokkos::Serial>(Kokkos::ViewAllocateWithoutInitializing(_label), arg_N0, arg_N1, arg_N2, arg_N3, arg_N4, arg_N5);
                view->assign_data(reinterpret_cast<DataType*>(_memoryPtr));
                return view;
            }
            if (arg_N4 != KOKKOS_IMPL_CTOR_DEFAULT_ARG)
            {
                auto view = new View<DataType*****, Layout, Kokkos::Serial>(Kokkos::ViewAllocateWithoutInitializing(_label), arg_N0, arg_N1, arg_N2, arg_N3, arg_N4);
                view->assign_data(reinterpret_cast<DataType*>(_memoryPtr));
                return view;
            }
            if (arg_N3 != KOKKOS_IMPL_CTOR_DEFAULT_ARG)
            {
                auto view = new View<DataType****, Layout, Kokkos::Serial>(Kokkos::ViewAllocateWithoutInitializing(_label), arg_N0, arg_N1, arg_N2, arg_N3);
                view->assign_data(reinterpret_cast<DataType*>(_memoryPtr));
                return view;
            }
            if (arg_N2 != KOKKOS_IMPL_CTOR_DEFAULT_ARG)
            {
                auto view = new View<DataType***, Layout, Kokkos::Serial>(Kokkos::ViewAllocateWithoutInitializing(_label), arg_N0, arg_N1, arg_N2);
                view->assign_data(reinterpret_cast<DataType*>(_memoryPtr));
                return view;
            }
            if (arg_N1 != KOKKOS_IMPL_CTOR_DEFAULT_ARG)
            {
                auto view = new View<DataType**, Layout, Kokkos::Serial>(Kokkos::ViewAllocateWithoutInitializing(_label), arg_N0, arg_N1);
                view->assign_data(reinterpret_cast<DataType*>(_memoryPtr));
                return view;
            }
            if (arg_N0 != KOKKOS_IMPL_CTOR_DEFAULT_ARG)
            {
                auto view = new View<DataType**, Layout, Kokkos::Serial>(Kokkos::ViewAllocateWithoutInitializing(_label), arg_N0);
                view->assign_data(reinterpret_cast<DataType*>(_memoryPtr));
                return view;
            }

            auto view = new View<DataType*, Layout, Kokkos::Serial>(Kokkos::ViewAllocateWithoutInitializing(_label));
            view->assign_data(reinterpret_cast<DataType*>(_memoryPtr));
            return view;
        }

        template<typename DataType, class Layout = Kokkos::Serial::array_layout>
        [[nodiscard]] __inline __host__ void* MakeViewFromHandle(REF(size_type) arg_N0 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
                                                                 REF(size_type) arg_N1 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
                                                                 REF(size_type) arg_N2 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
                                                                 REF(size_type) arg_N3 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
                                                                 REF(size_type) arg_N4 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
                                                                 REF(size_type) arg_N5 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
                                                                 REF(size_type) arg_N6 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
                                                                 REF(size_type) arg_N7 = KOKKOS_IMPL_CTOR_DEFAULT_ARG) const
        {

            if (arg_N7 != KOKKOS_IMPL_CTOR_DEFAULT_ARG)
            {
                auto view = new View<DataType********, Layout, Kokkos::Serial>(Kokkos::ViewAllocateWithoutInitializing(_label),
                                                                               arg_N0,
                                                                               arg_N1,
                                                                               arg_N2,
                                                                               arg_N3,
                                                                               arg_N4,
                                                                               arg_N5,
                                                                               arg_N6,
                                                                               arg_N7);
                view->assign_data(reinterpret_cast<DataType*>(_memoryHandle));
                return view;
            }
            if (arg_N6 != KOKKOS_IMPL_CTOR_DEFAULT_ARG)
            {
                auto view = new View<DataType*******, Layout, Kokkos::Serial>(Kokkos::ViewAllocateWithoutInitializing(_label),
                                                                              arg_N0,
                                                                              arg_N1,
                                                                              arg_N2,
                                                                              arg_N3,
                                                                              arg_N4,
                                                                              arg_N5,
                                                                              arg_N6);
                view->assign_data(reinterpret_cast<DataType*>(_memoryHandle));
                return view;
            }
            if (arg_N5 != KOKKOS_IMPL_CTOR_DEFAULT_ARG)
            {
                auto view = new View<DataType******, Layout, Kokkos::Serial>(Kokkos::ViewAllocateWithoutInitializing(_label), arg_N0, arg_N1, arg_N2, arg_N3, arg_N4, arg_N5);
                view->assign_data(reinterpret_cast<DataType*>(_memoryHandle));
                return view;
            }
            if (arg_N4 != KOKKOS_IMPL_CTOR_DEFAULT_ARG)
            {
                auto view = new View<DataType*****, Layout, Kokkos::Serial>(Kokkos::ViewAllocateWithoutInitializing(_label), arg_N0, arg_N1, arg_N2, arg_N3, arg_N4);
                view->assign_data(reinterpret_cast<DataType*>(_memoryHandle));
                return view;
            }
            if (arg_N3 != KOKKOS_IMPL_CTOR_DEFAULT_ARG)
            {
                auto view = new View<DataType****, Layout, Kokkos::Serial>(Kokkos::ViewAllocateWithoutInitializing(_label), arg_N0, arg_N1, arg_N2, arg_N3);
                view->assign_data(reinterpret_cast<DataType*>(_memoryHandle));
                return view;
            }
            if (arg_N2 != KOKKOS_IMPL_CTOR_DEFAULT_ARG)
            {
                auto view = new View<DataType***, Layout, Kokkos::Serial>(Kokkos::ViewAllocateWithoutInitializing(_label), arg_N0, arg_N1, arg_N2);
                view->assign_data(reinterpret_cast<DataType*>(_memoryHandle));
                return view;
            }
            if (arg_N1 != KOKKOS_IMPL_CTOR_DEFAULT_ARG)
            {
                auto view = new View<DataType**, Layout, Kokkos::Serial>(Kokkos::ViewAllocateWithoutInitializing(_label), arg_N0, arg_N1);
                view->assign_data(reinterpret_cast<DataType*>(_memoryHandle));
                return view;
            }
            if (arg_N0 != KOKKOS_IMPL_CTOR_DEFAULT_ARG)
            {
                auto view = new View<DataType**, Layout, Kokkos::Serial>(Kokkos::ViewAllocateWithoutInitializing(_label), arg_N0);
                view->assign_data(reinterpret_cast<DataType*>(_memoryHandle));
                return view;
            }

            auto view = new View<DataType*, Layout, Kokkos::Serial>(Kokkos::ViewAllocateWithoutInitializing(_label));
            view->assign_data(reinterpret_cast<DataType*>(_memoryHandle));
            return view;
        }

    private:
        InterprocessMemory(const InterprocessMemory&) = default;
        InterprocessMemory& operator=(const InterprocessMemory&) = default;
    };

    template<>
    struct InterprocessMemory<Kokkos::OpenMP>
    {
    private:
        void*       _memoryPtr;
        void*       _memoryHandle;
        size_type   _size;
        const char* _label;

    public:
        explicit InterprocessMemory(void* memory_ptr, void* memory_handle, REF(size_type) size, const char* label) :
            _memoryPtr(memory_ptr),
            _memoryHandle(memory_handle),
            _size(size),
            _label(label)
        {
        }

        ~InterprocessMemory()
        {
            IpcDestory(ExecutionSpaceKind::OpenMP, this);
        }

        [[nodiscard]] __inline __host__ void* GetMemoryPointer() const
        {
            return _memoryPtr;
        }

        [[nodiscard]] __inline __host__ void* GetDeviceHandle() const
        {
            return _memoryHandle;
        }

        [[nodiscard]] __inline __host__ size_type GetSize() const
        {
            return _size;
        }

        [[nodiscard]] __inline __host__ const char* GetLabel() const
        {
            return _label;
        }

        template<typename DataType, class Layout = Kokkos::OpenMP::array_layout>
        [[nodiscard]] __inline __host__ void* MakeViewFromPointer(REF(size_type) arg_N0 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
                                                                  REF(size_type) arg_N1 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
                                                                  REF(size_type) arg_N2 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
                                                                  REF(size_type) arg_N3 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
                                                                  REF(size_type) arg_N4 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
                                                                  REF(size_type) arg_N5 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
                                                                  REF(size_type) arg_N6 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
                                                                  REF(size_type) arg_N7 = KOKKOS_IMPL_CTOR_DEFAULT_ARG) const
        {
            if (arg_N7 != KOKKOS_IMPL_CTOR_DEFAULT_ARG)
            {
                auto view = new View<DataType********, Layout, Kokkos::OpenMP>(Kokkos::ViewAllocateWithoutInitializing(_label),
                                                                               arg_N0,
                                                                               arg_N1,
                                                                               arg_N2,
                                                                               arg_N3,
                                                                               arg_N4,
                                                                               arg_N5,
                                                                               arg_N6,
                                                                               arg_N7);
                view->assign_data(reinterpret_cast<DataType*>(_memoryPtr));
                return view;
            }
            if (arg_N6 != KOKKOS_IMPL_CTOR_DEFAULT_ARG)
            {
                auto view = new View<DataType*******, Layout, Kokkos::OpenMP>(Kokkos::ViewAllocateWithoutInitializing(_label),
                                                                              arg_N0,
                                                                              arg_N1,
                                                                              arg_N2,
                                                                              arg_N3,
                                                                              arg_N4,
                                                                              arg_N5,
                                                                              arg_N6);
                view->assign_data(reinterpret_cast<DataType*>(_memoryPtr));
                return view;
            }
            if (arg_N5 != KOKKOS_IMPL_CTOR_DEFAULT_ARG)
            {
                auto view = new View<DataType******, Layout, Kokkos::OpenMP>(Kokkos::ViewAllocateWithoutInitializing(_label), arg_N0, arg_N1, arg_N2, arg_N3, arg_N4, arg_N5);
                view->assign_data(reinterpret_cast<DataType*>(_memoryPtr));
                return view;
            }
            if (arg_N4 != KOKKOS_IMPL_CTOR_DEFAULT_ARG)
            {
                auto view = new View<DataType*****, Layout, Kokkos::OpenMP>(Kokkos::ViewAllocateWithoutInitializing(_label), arg_N0, arg_N1, arg_N2, arg_N3, arg_N4);
                view->assign_data(reinterpret_cast<DataType*>(_memoryPtr));
                return view;
            }
            if (arg_N3 != KOKKOS_IMPL_CTOR_DEFAULT_ARG)
            {
                auto view = new View<DataType****, Layout, Kokkos::OpenMP>(Kokkos::ViewAllocateWithoutInitializing(_label), arg_N0, arg_N1, arg_N2, arg_N3);
                view->assign_data(reinterpret_cast<DataType*>(_memoryPtr));
                return view;
            }
            if (arg_N2 != KOKKOS_IMPL_CTOR_DEFAULT_ARG)
            {
                auto view = new View<DataType***, Layout, Kokkos::OpenMP>(Kokkos::ViewAllocateWithoutInitializing(_label), arg_N0, arg_N1, arg_N2);
                view->assign_data(reinterpret_cast<DataType*>(_memoryPtr));
                return view;
            }
            if (arg_N1 != KOKKOS_IMPL_CTOR_DEFAULT_ARG)
            {
                auto view = new View<DataType**, Layout, Kokkos::OpenMP>(Kokkos::ViewAllocateWithoutInitializing(_label), arg_N0, arg_N1);
                view->assign_data(reinterpret_cast<DataType*>(_memoryPtr));
                return view;
            }
            if (arg_N0 != KOKKOS_IMPL_CTOR_DEFAULT_ARG)
            {
                auto view = new View<DataType**, Layout, Kokkos::OpenMP>(Kokkos::ViewAllocateWithoutInitializing(_label), arg_N0);
                view->assign_data(reinterpret_cast<DataType*>(_memoryPtr));
                return view;
            }

            auto view = new View<DataType*, Layout, Kokkos::OpenMP>(Kokkos::ViewAllocateWithoutInitializing(_label));
            view->assign_data(reinterpret_cast<DataType*>(_memoryPtr));
            return view;
        }

        template<typename DataType, class Layout = Kokkos::OpenMP::array_layout>
        [[nodiscard]] __inline __host__ void* MakeViewFromHandle(REF(size_type) arg_N0 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
                                                                 REF(size_type) arg_N1 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
                                                                 REF(size_type) arg_N2 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
                                                                 REF(size_type) arg_N3 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
                                                                 REF(size_type) arg_N4 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
                                                                 REF(size_type) arg_N5 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
                                                                 REF(size_type) arg_N6 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
                                                                 REF(size_type) arg_N7 = KOKKOS_IMPL_CTOR_DEFAULT_ARG) const
        {

            if (arg_N7 != KOKKOS_IMPL_CTOR_DEFAULT_ARG)
            {
                auto view = new View<DataType********, Layout, Kokkos::OpenMP>(Kokkos::ViewAllocateWithoutInitializing(_label),
                                                                               arg_N0,
                                                                               arg_N1,
                                                                               arg_N2,
                                                                               arg_N3,
                                                                               arg_N4,
                                                                               arg_N5,
                                                                               arg_N6,
                                                                               arg_N7);
                view->assign_data(reinterpret_cast<DataType*>(_memoryHandle));
                return view;
            }
            if (arg_N6 != KOKKOS_IMPL_CTOR_DEFAULT_ARG)
            {
                auto view = new View<DataType*******, Layout, Kokkos::OpenMP>(Kokkos::ViewAllocateWithoutInitializing(_label),
                                                                              arg_N0,
                                                                              arg_N1,
                                                                              arg_N2,
                                                                              arg_N3,
                                                                              arg_N4,
                                                                              arg_N5,
                                                                              arg_N6);
                view->assign_data(reinterpret_cast<DataType*>(_memoryHandle));
                return view;
            }
            if (arg_N5 != KOKKOS_IMPL_CTOR_DEFAULT_ARG)
            {
                auto view = new View<DataType******, Layout, Kokkos::OpenMP>(Kokkos::ViewAllocateWithoutInitializing(_label), arg_N0, arg_N1, arg_N2, arg_N3, arg_N4, arg_N5);
                view->assign_data(reinterpret_cast<DataType*>(_memoryHandle));
                return view;
            }
            if (arg_N4 != KOKKOS_IMPL_CTOR_DEFAULT_ARG)
            {
                auto view = new View<DataType*****, Layout, Kokkos::OpenMP>(Kokkos::ViewAllocateWithoutInitializing(_label), arg_N0, arg_N1, arg_N2, arg_N3, arg_N4);
                view->assign_data(reinterpret_cast<DataType*>(_memoryHandle));
                return view;
            }
            if (arg_N3 != KOKKOS_IMPL_CTOR_DEFAULT_ARG)
            {
                auto view = new View<DataType****, Layout, Kokkos::OpenMP>(Kokkos::ViewAllocateWithoutInitializing(_label), arg_N0, arg_N1, arg_N2, arg_N3);
                view->assign_data(reinterpret_cast<DataType*>(_memoryHandle));
                return view;
            }
            if (arg_N2 != KOKKOS_IMPL_CTOR_DEFAULT_ARG)
            {
                auto view = new View<DataType***, Layout, Kokkos::OpenMP>(Kokkos::ViewAllocateWithoutInitializing(_label), arg_N0, arg_N1, arg_N2);
                view->assign_data(reinterpret_cast<DataType*>(_memoryHandle));
                return view;
            }
            if (arg_N1 != KOKKOS_IMPL_CTOR_DEFAULT_ARG)
            {
                auto view = new View<DataType**, Layout, Kokkos::OpenMP>(Kokkos::ViewAllocateWithoutInitializing(_label), arg_N0, arg_N1);
                view->assign_data(reinterpret_cast<DataType*>(_memoryHandle));
                return view;
            }
            if (arg_N0 != KOKKOS_IMPL_CTOR_DEFAULT_ARG)
            {
                auto view = new View<DataType**, Layout, Kokkos::OpenMP>(Kokkos::ViewAllocateWithoutInitializing(_label), arg_N0);
                view->assign_data(reinterpret_cast<DataType*>(_memoryHandle));
                return view;
            }

            auto view = new View<DataType*, Layout, Kokkos::OpenMP>(Kokkos::ViewAllocateWithoutInitializing(_label));
            view->assign_data(reinterpret_cast<DataType*>(_memoryHandle));
            return view;
        }

    private:
        InterprocessMemory(const InterprocessMemory&) = default;
        InterprocessMemory& operator=(const InterprocessMemory&) = default;
    };

    template<>
    struct InterprocessMemory<Kokkos::Cuda>
    {
    private:
        void*       _memoryPtr;
        void*       _memoryHandle;
        size_type   _size;
        const char* _label;

    public:
        explicit InterprocessMemory(void* memory_ptr, void* memory_handle, REF(size_type) size, const char* label) :
            _memoryPtr(memory_ptr),
            _memoryHandle(memory_handle),
            _size(size),
            _label(label)
        {
        }

        ~InterprocessMemory()
        {
            IpcDestory(ExecutionSpaceKind::Cuda, this);
        }

        [[nodiscard]] __inline __host__ void* GetMemoryPointer() const
        {
            return _memoryPtr;
        }

        [[nodiscard]] __inline __host__ void* GetDeviceHandle() const
        {
            return _memoryHandle;
        }

        [[nodiscard]] __inline __host__ size_type GetSize() const
        {
            return _size;
        }

        [[nodiscard]] __inline __host__ const char* GetLabel() const
        {
            return _label;
        }

        template<typename DataType, class Layout = Kokkos::Cuda::array_layout>
        [[nodiscard]] __inline __host__ void* MakeViewFromPointer(REF(size_type) arg_N0 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
                                                                  REF(size_type) arg_N1 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
                                                                  REF(size_type) arg_N2 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
                                                                  REF(size_type) arg_N3 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
                                                                  REF(size_type) arg_N4 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
                                                                  REF(size_type) arg_N5 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
                                                                  REF(size_type) arg_N6 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
                                                                  REF(size_type) arg_N7 = KOKKOS_IMPL_CTOR_DEFAULT_ARG) const
        {
            if (arg_N7 != KOKKOS_IMPL_CTOR_DEFAULT_ARG)
            {
                auto view = new View<DataType********, Layout, Kokkos::Cuda>(Kokkos::ViewAllocateWithoutInitializing(_label),
                                                                             arg_N0,
                                                                             arg_N1,
                                                                             arg_N2,
                                                                             arg_N3,
                                                                             arg_N4,
                                                                             arg_N5,
                                                                             arg_N6,
                                                                             arg_N7);
                view->assign_data(reinterpret_cast<DataType*>(_memoryPtr));
                return view;
            }
            if (arg_N6 != KOKKOS_IMPL_CTOR_DEFAULT_ARG)
            {
                auto view = new View<DataType*******, Layout, Kokkos::Cuda>(Kokkos::ViewAllocateWithoutInitializing(_label),
                                                                            arg_N0,
                                                                            arg_N1,
                                                                            arg_N2,
                                                                            arg_N3,
                                                                            arg_N4,
                                                                            arg_N5,
                                                                            arg_N6);
                view->assign_data(reinterpret_cast<DataType*>(_memoryPtr));
                return view;
            }
            if (arg_N5 != KOKKOS_IMPL_CTOR_DEFAULT_ARG)
            {
                auto view = new View<DataType******, Layout, Kokkos::Cuda>(Kokkos::ViewAllocateWithoutInitializing(_label), arg_N0, arg_N1, arg_N2, arg_N3, arg_N4, arg_N5);
                view->assign_data(reinterpret_cast<DataType*>(_memoryPtr));
                return view;
            }
            if (arg_N4 != KOKKOS_IMPL_CTOR_DEFAULT_ARG)
            {
                auto view = new View<DataType*****, Layout, Kokkos::Cuda>(Kokkos::ViewAllocateWithoutInitializing(_label), arg_N0, arg_N1, arg_N2, arg_N3, arg_N4);
                view->assign_data(reinterpret_cast<DataType*>(_memoryPtr));
                return view;
            }
            if (arg_N3 != KOKKOS_IMPL_CTOR_DEFAULT_ARG)
            {
                auto view = new View<DataType****, Layout, Kokkos::Cuda>(Kokkos::ViewAllocateWithoutInitializing(_label), arg_N0, arg_N1, arg_N2, arg_N3);
                view->assign_data(reinterpret_cast<DataType*>(_memoryPtr));
                return view;
            }
            if (arg_N2 != KOKKOS_IMPL_CTOR_DEFAULT_ARG)
            {
                auto view = new View<DataType***, Layout, Kokkos::Cuda>(Kokkos::ViewAllocateWithoutInitializing(_label), arg_N0, arg_N1, arg_N2);
                view->assign_data(reinterpret_cast<DataType*>(_memoryPtr));
                return view;
            }
            if (arg_N1 != KOKKOS_IMPL_CTOR_DEFAULT_ARG)
            {
                auto view = new View<DataType**, Layout, Kokkos::Cuda>(Kokkos::ViewAllocateWithoutInitializing(_label), arg_N0, arg_N1);
                view->assign_data(reinterpret_cast<DataType*>(_memoryPtr));
                return view;
            }
            if (arg_N0 != KOKKOS_IMPL_CTOR_DEFAULT_ARG)
            {
                auto view = new View<DataType**, Layout, Kokkos::Cuda>(Kokkos::ViewAllocateWithoutInitializing(_label), arg_N0);
                view->assign_data(reinterpret_cast<DataType*>(_memoryPtr));
                return view;
            }

            auto view = new View<DataType*, Layout, Kokkos::Cuda>(Kokkos::ViewAllocateWithoutInitializing(_label));
            view->assign_data(reinterpret_cast<DataType*>(_memoryPtr));
            return view;
        }

        template<typename DataType, class Layout = Kokkos::Cuda::array_layout>
        [[nodiscard]] __inline __host__ void* MakeViewFromHandle(REF(size_type) arg_N0 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
                                                                 REF(size_type) arg_N1 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
                                                                 REF(size_type) arg_N2 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
                                                                 REF(size_type) arg_N3 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
                                                                 REF(size_type) arg_N4 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
                                                                 REF(size_type) arg_N5 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
                                                                 REF(size_type) arg_N6 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
                                                                 REF(size_type) arg_N7 = KOKKOS_IMPL_CTOR_DEFAULT_ARG) const
        {

            if (arg_N7 != KOKKOS_IMPL_CTOR_DEFAULT_ARG)
            {
                auto view = new View<DataType********, Layout, Kokkos::Cuda>(Kokkos::ViewAllocateWithoutInitializing(_label),
                                                                             arg_N0,
                                                                             arg_N1,
                                                                             arg_N2,
                                                                             arg_N3,
                                                                             arg_N4,
                                                                             arg_N5,
                                                                             arg_N6,
                                                                             arg_N7);
                view->assign_data(reinterpret_cast<DataType*>(_memoryHandle));
                return view;
            }
            if (arg_N6 != KOKKOS_IMPL_CTOR_DEFAULT_ARG)
            {
                auto view = new View<DataType*******, Layout, Kokkos::Cuda>(Kokkos::ViewAllocateWithoutInitializing(_label),
                                                                            arg_N0,
                                                                            arg_N1,
                                                                            arg_N2,
                                                                            arg_N3,
                                                                            arg_N4,
                                                                            arg_N5,
                                                                            arg_N6);
                view->assign_data(reinterpret_cast<DataType*>(_memoryHandle));
                return view;
            }
            if (arg_N5 != KOKKOS_IMPL_CTOR_DEFAULT_ARG)
            {
                auto view = new View<DataType******, Layout, Kokkos::Cuda>(Kokkos::ViewAllocateWithoutInitializing(_label), arg_N0, arg_N1, arg_N2, arg_N3, arg_N4, arg_N5);
                view->assign_data(reinterpret_cast<DataType*>(_memoryHandle));
                return view;
            }
            if (arg_N4 != KOKKOS_IMPL_CTOR_DEFAULT_ARG)
            {
                auto view = new View<DataType*****, Layout, Kokkos::Cuda>(Kokkos::ViewAllocateWithoutInitializing(_label), arg_N0, arg_N1, arg_N2, arg_N3, arg_N4);
                view->assign_data(reinterpret_cast<DataType*>(_memoryHandle));
                return view;
            }
            if (arg_N3 != KOKKOS_IMPL_CTOR_DEFAULT_ARG)
            {
                auto view = new View<DataType****, Layout, Kokkos::Cuda>(Kokkos::ViewAllocateWithoutInitializing(_label), arg_N0, arg_N1, arg_N2, arg_N3);
                view->assign_data(reinterpret_cast<DataType*>(_memoryHandle));
                return view;
            }
            if (arg_N2 != KOKKOS_IMPL_CTOR_DEFAULT_ARG)
            {
                auto view = new View<DataType***, Layout, Kokkos::Cuda>(Kokkos::ViewAllocateWithoutInitializing(_label), arg_N0, arg_N1, arg_N2);
                view->assign_data(reinterpret_cast<DataType*>(_memoryHandle));
                return view;
            }
            if (arg_N1 != KOKKOS_IMPL_CTOR_DEFAULT_ARG)
            {
                auto view = new View<DataType**, Layout, Kokkos::Cuda>(Kokkos::ViewAllocateWithoutInitializing(_label), arg_N0, arg_N1);
                view->assign_data(reinterpret_cast<DataType*>(_memoryHandle));
                return view;
            }
            if (arg_N0 != KOKKOS_IMPL_CTOR_DEFAULT_ARG)
            {
                auto view = new View<DataType**, Layout, Kokkos::Cuda>(Kokkos::ViewAllocateWithoutInitializing(_label), arg_N0);
                view->assign_data(reinterpret_cast<DataType*>(_memoryHandle));
                return view;
            }

            auto view = new View<DataType*, Layout, Kokkos::Cuda>(Kokkos::ViewAllocateWithoutInitializing(_label));
            view->assign_data(reinterpret_cast<DataType*>(_memoryHandle));
            return view;
        }

    private:
        InterprocessMemory(const InterprocessMemory&) = default;
        InterprocessMemory& operator=(const InterprocessMemory&) = default;
    };

}
