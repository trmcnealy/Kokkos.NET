#pragma once

#include <runtime.Kokkos/ViewTypes.hpp>

//#include <fileapi.h>
#include <handleapi.h>

// template<typename DataType, class ExecutionSpace, class Layout = typename ExecutionSpace::array_layout>
// using Vector = View<DataType*, Layout, ExecutionSpace>;

namespace Kokkos
{
    template<class ExecutionSpace>
    struct InterprocessMemory;
}

KOKKOS_NET_API_EXTERNC void* IpcCreate(CONST(ExecutionSpaceKind) execution_space, CONST(size_type) size, CONST(NativeString) label) noexcept;

KOKKOS_NET_API_EXTERNC void* IpcCreateFrom(CONST(ExecutionSpaceKind) execution_space, void* memoryPtr, CONST(size_type) size, CONST(NativeString) label) noexcept;

KOKKOS_NET_API_EXTERNC void* IpcOpenExisting(CONST(ExecutionSpaceKind) execution_space, void* instance) noexcept;

KOKKOS_NET_API_EXTERNC void IpcDestory(CONST(ExecutionSpaceKind) execution_space, void* instance) noexcept;

KOKKOS_NET_API_EXTERNC void IpcClose(CONST(ExecutionSpaceKind) execution_space, void* instance) noexcept;

KOKKOS_NET_API_EXTERNC void* IpcGetMemoryPointer(CONST(ExecutionSpaceKind) execution_space, void* instance) noexcept;

KOKKOS_NET_API_EXTERNC void* IpcGetDeviceHandle(CONST(ExecutionSpaceKind) execution_space, void* instance) noexcept;

KOKKOS_NET_API_EXTERNC size_type IpcGetSize(CONST(ExecutionSpaceKind) execution_space, void* instance) noexcept;

KOKKOS_NET_API_EXTERNC void* IpcMakeViewFromPointer(CONST(ExecutionSpaceKind) execution_space,
                                                    CONST(DataTypeKind) data_type,
                                                    void* instance,
                                                    CONST(size_type) arg_N0 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
                                                    CONST(size_type) arg_N1 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
                                                    CONST(size_type) arg_N2 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
                                                    CONST(size_type) arg_N3 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
                                                    CONST(size_type) arg_N4 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
                                                    CONST(size_type) arg_N5 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
                                                    CONST(size_type) arg_N6 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
                                                    CONST(size_type) arg_N7 = KOKKOS_IMPL_CTOR_DEFAULT_ARG) noexcept;

KOKKOS_NET_API_EXTERNC void* IpcMakeViewFromHandle(CONST(ExecutionSpaceKind) execution_space,
                                                   CONST(DataTypeKind) data_type,
                                                   void* instance,
                                                   CONST(size_type) arg_N0 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
                                                   CONST(size_type) arg_N1 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
                                                   CONST(size_type) arg_N2 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
                                                   CONST(size_type) arg_N3 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
                                                   CONST(size_type) arg_N4 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
                                                   CONST(size_type) arg_N5 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
                                                   CONST(size_type) arg_N6 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
                                                   CONST(size_type) arg_N7 = KOKKOS_IMPL_CTOR_DEFAULT_ARG) noexcept;

namespace Kokkos
{
    KOKKOS_INLINE_FUNCTION static size_type GetRank(CONST(size_type) arg_N0 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
                                                    CONST(size_type) arg_N1 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
                                                    CONST(size_type) arg_N2 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
                                                    CONST(size_type) arg_N3 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
                                                    CONST(size_type) arg_N4 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
                                                    CONST(size_type) arg_N5 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
                                                    CONST(size_type) arg_N6 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
                                                    CONST(size_type) arg_N7 = KOKKOS_IMPL_CTOR_DEFAULT_ARG)
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
    struct InterprocessMemory<Serial>
    {
    private:
        void*       _memoryPtr;
        void*       _memoryHandle;
        size_type   _size;
        const char* _label;

    public:
        explicit InterprocessMemory(void* memory_ptr, void* memory_handle, CONST(size_type) size, const char* label) : _memoryPtr(memory_ptr), _memoryHandle(memory_handle), _size(size), _label(label) {}

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

        template<typename DataType, class Layout = Serial::array_layout>
        [[nodiscard]] __inline __host__ void* MakeViewFromPointer(CONST(size_type) arg_N0 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
                                                                  CONST(size_type) arg_N1 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
                                                                  CONST(size_type) arg_N2 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
                                                                  CONST(size_type) arg_N3 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
                                                                  CONST(size_type) arg_N4 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
                                                                  CONST(size_type) arg_N5 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
                                                                  CONST(size_type) arg_N6 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
                                                                  CONST(size_type) arg_N7 = KOKKOS_IMPL_CTOR_DEFAULT_ARG) const
        {
            if (arg_N7 != KOKKOS_IMPL_CTOR_DEFAULT_ARG)
            {
                auto view = new View<DataType********, Layout, Serial>(ViewAllocateWithoutInitializing(_label), arg_N0, arg_N1, arg_N2, arg_N3, arg_N4, arg_N5, arg_N6, arg_N7);
                view->assign_data(reinterpret_cast<DataType*>(_memoryPtr));
                return view;
            }
            if (arg_N6 != KOKKOS_IMPL_CTOR_DEFAULT_ARG)
            {
                auto view = new View<DataType*******, Layout, Serial>(ViewAllocateWithoutInitializing(_label), arg_N0, arg_N1, arg_N2, arg_N3, arg_N4, arg_N5, arg_N6);
                view->assign_data(reinterpret_cast<DataType*>(_memoryPtr));
                return view;
            }
            if (arg_N5 != KOKKOS_IMPL_CTOR_DEFAULT_ARG)
            {
                auto view = new View<DataType******, Layout, Serial>(ViewAllocateWithoutInitializing(_label), arg_N0, arg_N1, arg_N2, arg_N3, arg_N4, arg_N5);
                view->assign_data(reinterpret_cast<DataType*>(_memoryPtr));
                return view;
            }
            if (arg_N4 != KOKKOS_IMPL_CTOR_DEFAULT_ARG)
            {
                auto view = new View<DataType*****, Layout, Serial>(ViewAllocateWithoutInitializing(_label), arg_N0, arg_N1, arg_N2, arg_N3, arg_N4);
                view->assign_data(reinterpret_cast<DataType*>(_memoryPtr));
                return view;
            }
            if (arg_N3 != KOKKOS_IMPL_CTOR_DEFAULT_ARG)
            {
                auto view = new View<DataType****, Layout, Serial>(ViewAllocateWithoutInitializing(_label), arg_N0, arg_N1, arg_N2, arg_N3);
                view->assign_data(reinterpret_cast<DataType*>(_memoryPtr));
                return view;
            }
            if (arg_N2 != KOKKOS_IMPL_CTOR_DEFAULT_ARG)
            {
                auto view = new View<DataType***, Layout, Serial>(ViewAllocateWithoutInitializing(_label), arg_N0, arg_N1, arg_N2);
                view->assign_data(reinterpret_cast<DataType*>(_memoryPtr));
                return view;
            }
            if (arg_N1 != KOKKOS_IMPL_CTOR_DEFAULT_ARG)
            {
                auto view = new View<DataType**, Layout, Serial>(ViewAllocateWithoutInitializing(_label), arg_N0, arg_N1);
                view->assign_data(reinterpret_cast<DataType*>(_memoryPtr));
                return view;
            }
            if (arg_N0 != KOKKOS_IMPL_CTOR_DEFAULT_ARG)
            {
                auto view = new View<DataType**, Layout, Serial>(ViewAllocateWithoutInitializing(_label), arg_N0);
                view->assign_data(reinterpret_cast<DataType*>(_memoryPtr));
                return view;
            }

            auto view = new View<DataType*, Layout, Serial>(ViewAllocateWithoutInitializing(_label));
            view->assign_data(reinterpret_cast<DataType*>(_memoryPtr));
            return view;
        }

        template<typename DataType, class Layout = Serial::array_layout>
        [[nodiscard]] __inline __host__ void* MakeViewFromHandle(CONST(size_type) arg_N0 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
                                                                 CONST(size_type) arg_N1 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
                                                                 CONST(size_type) arg_N2 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
                                                                 CONST(size_type) arg_N3 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
                                                                 CONST(size_type) arg_N4 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
                                                                 CONST(size_type) arg_N5 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
                                                                 CONST(size_type) arg_N6 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
                                                                 CONST(size_type) arg_N7 = KOKKOS_IMPL_CTOR_DEFAULT_ARG) const
        {
            if (arg_N7 != KOKKOS_IMPL_CTOR_DEFAULT_ARG)
            {
                auto view = new View<DataType********, Layout, Serial>(ViewAllocateWithoutInitializing(_label), arg_N0, arg_N1, arg_N2, arg_N3, arg_N4, arg_N5, arg_N6, arg_N7);
                view->assign_data(reinterpret_cast<DataType*>(_memoryHandle));
                return view;
            }
            if (arg_N6 != KOKKOS_IMPL_CTOR_DEFAULT_ARG)
            {
                auto view = new View<DataType*******, Layout, Serial>(ViewAllocateWithoutInitializing(_label), arg_N0, arg_N1, arg_N2, arg_N3, arg_N4, arg_N5, arg_N6);
                view->assign_data(reinterpret_cast<DataType*>(_memoryHandle));
                return view;
            }
            if (arg_N5 != KOKKOS_IMPL_CTOR_DEFAULT_ARG)
            {
                auto view = new View<DataType******, Layout, Serial>(ViewAllocateWithoutInitializing(_label), arg_N0, arg_N1, arg_N2, arg_N3, arg_N4, arg_N5);
                view->assign_data(reinterpret_cast<DataType*>(_memoryHandle));
                return view;
            }
            if (arg_N4 != KOKKOS_IMPL_CTOR_DEFAULT_ARG)
            {
                auto view = new View<DataType*****, Layout, Serial>(ViewAllocateWithoutInitializing(_label), arg_N0, arg_N1, arg_N2, arg_N3, arg_N4);
                view->assign_data(reinterpret_cast<DataType*>(_memoryHandle));
                return view;
            }
            if (arg_N3 != KOKKOS_IMPL_CTOR_DEFAULT_ARG)
            {
                auto view = new View<DataType****, Layout, Serial>(ViewAllocateWithoutInitializing(_label), arg_N0, arg_N1, arg_N2, arg_N3);
                view->assign_data(reinterpret_cast<DataType*>(_memoryHandle));
                return view;
            }
            if (arg_N2 != KOKKOS_IMPL_CTOR_DEFAULT_ARG)
            {
                auto view = new View<DataType***, Layout, Serial>(ViewAllocateWithoutInitializing(_label), arg_N0, arg_N1, arg_N2);
                view->assign_data(reinterpret_cast<DataType*>(_memoryHandle));
                return view;
            }
            if (arg_N1 != KOKKOS_IMPL_CTOR_DEFAULT_ARG)
            {
                auto view = new View<DataType**, Layout, Serial>(ViewAllocateWithoutInitializing(_label), arg_N0, arg_N1);
                view->assign_data(reinterpret_cast<DataType*>(_memoryHandle));
                return view;
            }
            if (arg_N0 != KOKKOS_IMPL_CTOR_DEFAULT_ARG)
            {
                auto view = new View<DataType**, Layout, Serial>(ViewAllocateWithoutInitializing(_label), arg_N0);
                view->assign_data(reinterpret_cast<DataType*>(_memoryHandle));
                return view;
            }

            auto view = new View<DataType*, Layout, Serial>(ViewAllocateWithoutInitializing(_label));
            view->assign_data(reinterpret_cast<DataType*>(_memoryHandle));
            return view;
        }

    private:
        InterprocessMemory(const InterprocessMemory&) = default;
        InterprocessMemory& operator=(const InterprocessMemory&) = default;
    };

    template<>
    struct InterprocessMemory<OpenMP>
    {
    private:
        void*       _memoryPtr;
        void*       _memoryHandle;
        size_type   _size;
        const char* _label;

    public:
        explicit InterprocessMemory(void* memory_ptr, void* memory_handle, CONST(size_type) size, const char* label) : _memoryPtr(memory_ptr), _memoryHandle(memory_handle), _size(size), _label(label) {}

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

        template<typename DataType, class Layout = OpenMP::array_layout>
        [[nodiscard]] __inline __host__ void* MakeViewFromPointer(CONST(size_type) arg_N0 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
                                                                  CONST(size_type) arg_N1 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
                                                                  CONST(size_type) arg_N2 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
                                                                  CONST(size_type) arg_N3 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
                                                                  CONST(size_type) arg_N4 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
                                                                  CONST(size_type) arg_N5 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
                                                                  CONST(size_type) arg_N6 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
                                                                  CONST(size_type) arg_N7 = KOKKOS_IMPL_CTOR_DEFAULT_ARG) const
        {
            if (arg_N7 != KOKKOS_IMPL_CTOR_DEFAULT_ARG)
            {
                auto view = new View<DataType********, Layout, OpenMP>(ViewAllocateWithoutInitializing(_label), arg_N0, arg_N1, arg_N2, arg_N3, arg_N4, arg_N5, arg_N6, arg_N7);
                view->assign_data(reinterpret_cast<DataType*>(_memoryPtr));
                return view;
            }
            if (arg_N6 != KOKKOS_IMPL_CTOR_DEFAULT_ARG)
            {
                auto view = new View<DataType*******, Layout, OpenMP>(ViewAllocateWithoutInitializing(_label), arg_N0, arg_N1, arg_N2, arg_N3, arg_N4, arg_N5, arg_N6);
                view->assign_data(reinterpret_cast<DataType*>(_memoryPtr));
                return view;
            }
            if (arg_N5 != KOKKOS_IMPL_CTOR_DEFAULT_ARG)
            {
                auto view = new View<DataType******, Layout, OpenMP>(ViewAllocateWithoutInitializing(_label), arg_N0, arg_N1, arg_N2, arg_N3, arg_N4, arg_N5);
                view->assign_data(reinterpret_cast<DataType*>(_memoryPtr));
                return view;
            }
            if (arg_N4 != KOKKOS_IMPL_CTOR_DEFAULT_ARG)
            {
                auto view = new View<DataType*****, Layout, OpenMP>(ViewAllocateWithoutInitializing(_label), arg_N0, arg_N1, arg_N2, arg_N3, arg_N4);
                view->assign_data(reinterpret_cast<DataType*>(_memoryPtr));
                return view;
            }
            if (arg_N3 != KOKKOS_IMPL_CTOR_DEFAULT_ARG)
            {
                auto view = new View<DataType****, Layout, OpenMP>(ViewAllocateWithoutInitializing(_label), arg_N0, arg_N1, arg_N2, arg_N3);
                view->assign_data(reinterpret_cast<DataType*>(_memoryPtr));
                return view;
            }
            if (arg_N2 != KOKKOS_IMPL_CTOR_DEFAULT_ARG)
            {
                auto view = new View<DataType***, Layout, OpenMP>(ViewAllocateWithoutInitializing(_label), arg_N0, arg_N1, arg_N2);
                view->assign_data(reinterpret_cast<DataType*>(_memoryPtr));
                return view;
            }
            if (arg_N1 != KOKKOS_IMPL_CTOR_DEFAULT_ARG)
            {
                auto view = new View<DataType**, Layout, OpenMP>(ViewAllocateWithoutInitializing(_label), arg_N0, arg_N1);
                view->assign_data(reinterpret_cast<DataType*>(_memoryPtr));
                return view;
            }
            if (arg_N0 != KOKKOS_IMPL_CTOR_DEFAULT_ARG)
            {
                auto view = new View<DataType**, Layout, OpenMP>(ViewAllocateWithoutInitializing(_label), arg_N0);
                view->assign_data(reinterpret_cast<DataType*>(_memoryPtr));
                return view;
            }

            auto view = new View<DataType*, Layout, OpenMP>(ViewAllocateWithoutInitializing(_label));
            view->assign_data(reinterpret_cast<DataType*>(_memoryPtr));
            return view;
        }

        template<typename DataType, class Layout = OpenMP::array_layout>
        [[nodiscard]] __inline __host__ void* MakeViewFromHandle(CONST(size_type) arg_N0 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
                                                                 CONST(size_type) arg_N1 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
                                                                 CONST(size_type) arg_N2 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
                                                                 CONST(size_type) arg_N3 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
                                                                 CONST(size_type) arg_N4 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
                                                                 CONST(size_type) arg_N5 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
                                                                 CONST(size_type) arg_N6 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
                                                                 CONST(size_type) arg_N7 = KOKKOS_IMPL_CTOR_DEFAULT_ARG) const
        {
            if (arg_N7 != KOKKOS_IMPL_CTOR_DEFAULT_ARG)
            {
                auto view = new View<DataType********, Layout, OpenMP>(ViewAllocateWithoutInitializing(_label), arg_N0, arg_N1, arg_N2, arg_N3, arg_N4, arg_N5, arg_N6, arg_N7);
                view->assign_data(reinterpret_cast<DataType*>(_memoryHandle));
                return view;
            }
            if (arg_N6 != KOKKOS_IMPL_CTOR_DEFAULT_ARG)
            {
                auto view = new View<DataType*******, Layout, OpenMP>(ViewAllocateWithoutInitializing(_label), arg_N0, arg_N1, arg_N2, arg_N3, arg_N4, arg_N5, arg_N6);
                view->assign_data(reinterpret_cast<DataType*>(_memoryHandle));
                return view;
            }
            if (arg_N5 != KOKKOS_IMPL_CTOR_DEFAULT_ARG)
            {
                auto view = new View<DataType******, Layout, OpenMP>(ViewAllocateWithoutInitializing(_label), arg_N0, arg_N1, arg_N2, arg_N3, arg_N4, arg_N5);
                view->assign_data(reinterpret_cast<DataType*>(_memoryHandle));
                return view;
            }
            if (arg_N4 != KOKKOS_IMPL_CTOR_DEFAULT_ARG)
            {
                auto view = new View<DataType*****, Layout, OpenMP>(ViewAllocateWithoutInitializing(_label), arg_N0, arg_N1, arg_N2, arg_N3, arg_N4);
                view->assign_data(reinterpret_cast<DataType*>(_memoryHandle));
                return view;
            }
            if (arg_N3 != KOKKOS_IMPL_CTOR_DEFAULT_ARG)
            {
                auto view = new View<DataType****, Layout, OpenMP>(ViewAllocateWithoutInitializing(_label), arg_N0, arg_N1, arg_N2, arg_N3);
                view->assign_data(reinterpret_cast<DataType*>(_memoryHandle));
                return view;
            }
            if (arg_N2 != KOKKOS_IMPL_CTOR_DEFAULT_ARG)
            {
                auto view = new View<DataType***, Layout, OpenMP>(ViewAllocateWithoutInitializing(_label), arg_N0, arg_N1, arg_N2);
                view->assign_data(reinterpret_cast<DataType*>(_memoryHandle));
                return view;
            }
            if (arg_N1 != KOKKOS_IMPL_CTOR_DEFAULT_ARG)
            {
                auto view = new View<DataType**, Layout, OpenMP>(ViewAllocateWithoutInitializing(_label), arg_N0, arg_N1);
                view->assign_data(reinterpret_cast<DataType*>(_memoryHandle));
                return view;
            }
            if (arg_N0 != KOKKOS_IMPL_CTOR_DEFAULT_ARG)
            {
                auto view = new View<DataType**, Layout, OpenMP>(ViewAllocateWithoutInitializing(_label), arg_N0);
                view->assign_data(reinterpret_cast<DataType*>(_memoryHandle));
                return view;
            }

            auto view = new View<DataType*, Layout, OpenMP>(ViewAllocateWithoutInitializing(_label));
            view->assign_data(reinterpret_cast<DataType*>(_memoryHandle));
            return view;
        }

    private:
        InterprocessMemory(const InterprocessMemory&) = default;
        InterprocessMemory& operator=(const InterprocessMemory&) = default;
    };

    template<>
    struct InterprocessMemory<Cuda>
    {
    private:
        void*       _memoryPtr;
        void*       _memoryHandle;
        size_type   _size;
        const char* _label;

    public:
        explicit InterprocessMemory(void* memory_ptr, void* memory_handle, CONST(size_type) size, const char* label) : _memoryPtr(memory_ptr), _memoryHandle(memory_handle), _size(size), _label(label) {}

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

        template<typename DataType, class Layout = Cuda::array_layout>
        [[nodiscard]] __inline __host__ void* MakeViewFromPointer(CONST(size_type) arg_N0 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
                                                                  CONST(size_type) arg_N1 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
                                                                  CONST(size_type) arg_N2 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
                                                                  CONST(size_type) arg_N3 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
                                                                  CONST(size_type) arg_N4 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
                                                                  CONST(size_type) arg_N5 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
                                                                  CONST(size_type) arg_N6 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
                                                                  CONST(size_type) arg_N7 = KOKKOS_IMPL_CTOR_DEFAULT_ARG) const
        {
            if (arg_N7 != KOKKOS_IMPL_CTOR_DEFAULT_ARG)
            {
                auto view = new View<DataType********, Layout, Cuda>(ViewAllocateWithoutInitializing(_label), arg_N0, arg_N1, arg_N2, arg_N3, arg_N4, arg_N5, arg_N6, arg_N7);
                view->assign_data(reinterpret_cast<DataType*>(_memoryPtr));
                return view;
            }
            if (arg_N6 != KOKKOS_IMPL_CTOR_DEFAULT_ARG)
            {
                auto view = new View<DataType*******, Layout, Cuda>(ViewAllocateWithoutInitializing(_label), arg_N0, arg_N1, arg_N2, arg_N3, arg_N4, arg_N5, arg_N6);
                view->assign_data(reinterpret_cast<DataType*>(_memoryPtr));
                return view;
            }
            if (arg_N5 != KOKKOS_IMPL_CTOR_DEFAULT_ARG)
            {
                auto view = new View<DataType******, Layout, Cuda>(ViewAllocateWithoutInitializing(_label), arg_N0, arg_N1, arg_N2, arg_N3, arg_N4, arg_N5);
                view->assign_data(reinterpret_cast<DataType*>(_memoryPtr));
                return view;
            }
            if (arg_N4 != KOKKOS_IMPL_CTOR_DEFAULT_ARG)
            {
                auto view = new View<DataType*****, Layout, Cuda>(ViewAllocateWithoutInitializing(_label), arg_N0, arg_N1, arg_N2, arg_N3, arg_N4);
                view->assign_data(reinterpret_cast<DataType*>(_memoryPtr));
                return view;
            }
            if (arg_N3 != KOKKOS_IMPL_CTOR_DEFAULT_ARG)
            {
                auto view = new View<DataType****, Layout, Cuda>(ViewAllocateWithoutInitializing(_label), arg_N0, arg_N1, arg_N2, arg_N3);
                view->assign_data(reinterpret_cast<DataType*>(_memoryPtr));
                return view;
            }
            if (arg_N2 != KOKKOS_IMPL_CTOR_DEFAULT_ARG)
            {
                auto view = new View<DataType***, Layout, Cuda>(ViewAllocateWithoutInitializing(_label), arg_N0, arg_N1, arg_N2);
                view->assign_data(reinterpret_cast<DataType*>(_memoryPtr));
                return view;
            }
            if (arg_N1 != KOKKOS_IMPL_CTOR_DEFAULT_ARG)
            {
                auto view = new View<DataType**, Layout, Cuda>(ViewAllocateWithoutInitializing(_label), arg_N0, arg_N1);
                view->assign_data(reinterpret_cast<DataType*>(_memoryPtr));
                return view;
            }
            if (arg_N0 != KOKKOS_IMPL_CTOR_DEFAULT_ARG)
            {
                auto view = new View<DataType**, Layout, Cuda>(ViewAllocateWithoutInitializing(_label), arg_N0);
                view->assign_data(reinterpret_cast<DataType*>(_memoryPtr));
                return view;
            }

            auto view = new View<DataType*, Layout, Cuda>(ViewAllocateWithoutInitializing(_label));
            view->assign_data(reinterpret_cast<DataType*>(_memoryPtr));
            return view;
        }

        template<typename DataType, class Layout = Cuda::array_layout>
        [[nodiscard]] __inline __host__ void* MakeViewFromHandle(CONST(size_type) arg_N0 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
                                                                 CONST(size_type) arg_N1 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
                                                                 CONST(size_type) arg_N2 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
                                                                 CONST(size_type) arg_N3 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
                                                                 CONST(size_type) arg_N4 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
                                                                 CONST(size_type) arg_N5 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
                                                                 CONST(size_type) arg_N6 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
                                                                 CONST(size_type) arg_N7 = KOKKOS_IMPL_CTOR_DEFAULT_ARG) const
        {
            if (arg_N7 != KOKKOS_IMPL_CTOR_DEFAULT_ARG)
            {
                auto view = new View<DataType********, Layout, Cuda>(ViewAllocateWithoutInitializing(_label), arg_N0, arg_N1, arg_N2, arg_N3, arg_N4, arg_N5, arg_N6, arg_N7);
                view->assign_data(reinterpret_cast<DataType*>(_memoryHandle));
                return view;
            }
            if (arg_N6 != KOKKOS_IMPL_CTOR_DEFAULT_ARG)
            {
                auto view = new View<DataType*******, Layout, Cuda>(ViewAllocateWithoutInitializing(_label), arg_N0, arg_N1, arg_N2, arg_N3, arg_N4, arg_N5, arg_N6);
                view->assign_data(reinterpret_cast<DataType*>(_memoryHandle));
                return view;
            }
            if (arg_N5 != KOKKOS_IMPL_CTOR_DEFAULT_ARG)
            {
                auto view = new View<DataType******, Layout, Cuda>(ViewAllocateWithoutInitializing(_label), arg_N0, arg_N1, arg_N2, arg_N3, arg_N4, arg_N5);
                view->assign_data(reinterpret_cast<DataType*>(_memoryHandle));
                return view;
            }
            if (arg_N4 != KOKKOS_IMPL_CTOR_DEFAULT_ARG)
            {
                auto view = new View<DataType*****, Layout, Cuda>(ViewAllocateWithoutInitializing(_label), arg_N0, arg_N1, arg_N2, arg_N3, arg_N4);
                view->assign_data(reinterpret_cast<DataType*>(_memoryHandle));
                return view;
            }
            if (arg_N3 != KOKKOS_IMPL_CTOR_DEFAULT_ARG)
            {
                auto view = new View<DataType****, Layout, Cuda>(ViewAllocateWithoutInitializing(_label), arg_N0, arg_N1, arg_N2, arg_N3);
                view->assign_data(reinterpret_cast<DataType*>(_memoryHandle));
                return view;
            }
            if (arg_N2 != KOKKOS_IMPL_CTOR_DEFAULT_ARG)
            {
                auto view = new View<DataType***, Layout, Cuda>(ViewAllocateWithoutInitializing(_label), arg_N0, arg_N1, arg_N2);
                view->assign_data(reinterpret_cast<DataType*>(_memoryHandle));
                return view;
            }
            if (arg_N1 != KOKKOS_IMPL_CTOR_DEFAULT_ARG)
            {
                auto view = new View<DataType**, Layout, Cuda>(ViewAllocateWithoutInitializing(_label), arg_N0, arg_N1);
                view->assign_data(reinterpret_cast<DataType*>(_memoryHandle));
                return view;
            }
            if (arg_N0 != KOKKOS_IMPL_CTOR_DEFAULT_ARG)
            {
                auto view = new View<DataType**, Layout, Cuda>(ViewAllocateWithoutInitializing(_label), arg_N0);
                view->assign_data(reinterpret_cast<DataType*>(_memoryHandle));
                return view;
            }

            auto view = new View<DataType*, Layout, Cuda>(ViewAllocateWithoutInitializing(_label));
            view->assign_data(reinterpret_cast<DataType*>(_memoryHandle));
            return view;
        }

    private:
        InterprocessMemory(const InterprocessMemory&) = default;
        InterprocessMemory& operator=(const InterprocessMemory&) = default;
    };
}
