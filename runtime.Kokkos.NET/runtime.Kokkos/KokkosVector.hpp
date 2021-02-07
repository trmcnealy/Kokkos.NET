#pragma once

#include "runtime.Kokkos/KokkosIVector.hpp"

namespace Kokkos
{
    template<typename TDataType, typename TExecustionSpace>
    derived_struct KOKKOS_NET_API_EXPORT Vector final : public IVector<TDataType, TExecustionSpace>
    {
    private:
        Kokkos::vector<TDataType, TExecustionSpace>* _vector;

    public:
        Vector() : _vector(new Kokkos::vector<TDataType, TExecustionSpace>()) {}

        Vector(int32 n, TDataType value = TDataType()) : _vector(new Kokkos::vector<TDataType, TExecustionSpace>(n, value)) {}

        static Vector<TDataType, TExecustionSpace>* Create()
        {
            return new Vector<TDataType, TExecustionSpace>();
        }

        static Vector<TDataType, TExecustionSpace>* Create(int32 n, TDataType value = TDataType())
        {
            return new Vector<TDataType, TExecustionSpace>(n, value);
        }

        static void Destory(Vector<TDataType, TExecustionSpace>* vector)
        {
            vector->Vector<TDataType, TExecustionSpace>::~Vector();
        }

        ~Vector() override
        {
            delete _vector;
        }

#if 0
        KOKKOS_INLINE_FUNCTION static void* operator new(std::size_t sz)
        {
            return Kokkos::kokkos_malloc<typename TExecustionSpace::memory_space>(sz * sizeof(Vector<TDataType, TExecustionSpace>));
        }
        KOKKOS_INLINE_FUNCTION static void* operator new[](std::size_t sz)
        {
            return Kokkos::kokkos_malloc<typename TExecustionSpace::memory_space>(sz * sizeof(Vector<TDataType, TExecustionSpace>));
        }
        KOKKOS_INLINE_FUNCTION static void* operator new(std::size_t sz, void* ptr)
        {
            return ptr;
        }
        KOKKOS_INLINE_FUNCTION static void* operator new[](std::size_t sz, void* ptr)
        {
            return ptr;
        }
        KOKKOS_INLINE_FUNCTION static void operator delete(void* ptr)
        {
            Kokkos::kokkos_free<typename TExecustionSpace::memory_space>(ptr);
        }
        KOKKOS_INLINE_FUNCTION static void operator delete[](void* ptr)
        {
            Kokkos::kokkos_free<typename TExecustionSpace::memory_space>(ptr);
        }
        KOKKOS_INLINE_FUNCTION static void operator delete(void* ptr, void*)
        {
            Kokkos::kokkos_free<typename TExecustionSpace::memory_space>(ptr);
        }
        KOKKOS_INLINE_FUNCTION static void operator delete[](void* ptr, void*)
        {
            Kokkos::kokkos_free<typename TExecustionSpace::memory_space>(ptr);
        }
#endif

        void sync_host() override
        {
            _vector->sync_host();
        }
        void sync_device() override
        {
            _vector->sync_device();
        }
        bool need_sync_host() override
        {
            return _vector->need_sync_host();
        }
        bool need_sync_device() override
        {
            return _vector->need_sync_device();
        }

        TDataType* data() override
        {
            return _vector->data();
        }

        void assign(uint64 n, const TDataType& value) override
        {
            _vector->assign(n, value);
        }

        TDataType& back() override
        {
            return _vector->back();
        }
        TDataType& front() override
        {
            return _vector->front();
        }

        TDataType* begin() override
        {
            return _vector->begin();
        }
        TDataType* end() override
        {
            return _vector->end();
        }

        void clear() override
        {
            _vector->clear();
        }
        bool empty() override
        {
            return _vector->empty();
        }

        TDataType* find(TDataType value) override
        {
            return _vector->find(value);
        }

        void device_to_host() override
        {
            _vector->device_to_host();
        }
        void host_to_device() override
        {
            _vector->host_to_device();
        }

        TDataType* insert(TDataType* it, const TDataType& value) override
        {
            return _vector->insert(it, value);
        }
        TDataType* insert(TDataType* it, uint64 count, const TDataType& value) override
        {
            return _vector->insert(it, count, value);
        }

        bool is_allocated() override
        {
            return _vector->is_allocated();
        }
        bool is_sorted() override
        {
            return _vector->is_sorted();
        }

        void on_device() override
        {
            _vector->on_device();
        }
        void on_host() override
        {
            _vector->on_host();
        }

        uint64 lower_bound(const uint64& start, const uint64& theEnd, const TDataType& comp_val) override
        {
            return _vector->lower_bound(start, theEnd, comp_val);
        }
        uint64 max_size() override
        {
            return _vector->max_size();
        }

        void pop_back() override
        {
            _vector->pop_back();
        }
        void push_back(TDataType value) override
        {
            _vector->push_back(value);
        }

        void reserve(uint64 n) override
        {
            _vector->reserve(n);
        }

        void resize(uint64 n) override
        {
            _vector->resize(n);
        }
        void resize(uint64 n, const TDataType& value) override
        {
            _vector->resize(n, value);
        }

        void set_overallocation(float extra) override
        {
            _vector->set_overallocation(extra);
        }

        uint64 size() override
        {
            return _vector->size();
        }
        uint64 span() override
        {
            return _vector->span();
        }
    };
}
