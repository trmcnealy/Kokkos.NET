#pragma once

#include <Types.hpp>
#include <Kokkos_Core.hpp>

#include <algorithm>nnn
#include <climits>
#include <exception>
#include <functional>
#include <iterator>
#include <list>
#include <map>
#include <numeric>
#include <set>
#include <string>
#include <type_traits>
#include <vector>

namespace Linq
{
#define __always_inline __attribute__((always_inline))
#define __flatten __attribute__((flatten))
#define __forceinline __inline

    typedef std::size_t size_type;

    struct BaseException : std::exception
    {
        virtual const char* what() const noexcept { return "base_exception"; }
    };

    struct ProgrammingErrorException : BaseException
    {
        virtual const char* what() const noexcept { return "programming_error_exception"; }
    };

    struct SequenceEmptyException : BaseException
    {
        virtual const char* what() const noexcept { return "sequence_empty_exception"; }
    };

    // Tedious implementation details of cpplinq

    namespace detail
    {
        size_type const invalid_size = static_cast<size_type>(-1);

        template<typename TValue>
        struct CleanupType
        {
            typedef typename std::remove_const<typename std::remove_reference<TValue>::type>::type type;
        };

        template<typename TRangeBuilder, typename TRange>
        struct GetBuiltupType
        {
            static TRangeBuilder GetBuilder();
            static TRange        GetRange();

            typedef decltype(GetBuilder().build(GetRange())) type;
        };

        template<typename TPredicate, typename TValue>
        struct GetTransformedType
        {
            static TValue     GetValue();
            static TPredicate GetPredicate();

            typedef decltype(GetPredicate()(GetValue())) raw_type;
            typedef typename CleanupType<raw_type>::type type;
        };

        template<typename TArray>
        struct GetArrayProperties;

        template<typename TValue, int Size>
        struct GetArrayProperties<TValue[Size]>
        {
            enum
            {
                size = Size,
            };

            typedef typename CleanupType<TValue>::type value_type;
            typedef value_type const*                  iterator_type;
        };

        template<typename TValue>
        struct Opt
        {
            typedef TValue value_type;

            __forceinline Opt() noexcept : is_initialized(false) {}

            __forceinline explicit Opt(value_type&& value) : is_initialized(true) { new(&storage) value_type(std::move(value)); }

            __forceinline explicit Opt(value_type const& value) : is_initialized(true) { new(&storage) value_type(value); }

            __forceinline ~Opt() noexcept
            {
                auto ptr = get_ptr();
                if(ptr)
                {
                    ptr->~value_type();
                }
                is_initialized = false;
            }

            __forceinline Opt(Opt const& v) : is_initialized(v.is_initialized)
            {
                if(v.is_initialized)
                {
                    copy(&storage, &v.storage);
                }
            }

            __forceinline Opt(Opt&& v) noexcept : is_initialized(v.is_initialized)
            {
                if(v.is_initialized)
                {
                    move(&storage, &v.storage);
                }
                v.is_initialized = false;
            }

            KOKKOS_FUNCTION void swap(Opt& v) noexcept
            {
                if(is_initialized && v.is_initialized)
                {
                    storage_type tmp;

                    move(&tmp, &storage);
                    move(&storage, &v.storage);
                    move(&v.storage, &tmp);
                }
                else if(is_initialized)
                {
                    move(&v.storage, &storage);
                    v.is_initialized = true;
                    is_initialized   = false;
                }
                else if(v.is_initialized)
                {
                    move(&storage, &v.storage);
                    v.is_initialized = false;
                    is_initialized   = true;
                }
                else
                {
                    // Do nothing
                }
            }

            __forceinline Opt& operator=(Opt const& v)
            {
                if(this == std::addressof(v))
                {
                    return *this;
                }

                Opt<value_type> o(v);

                swap(o);

                return *this;
            }

            __forceinline Opt& operator=(Opt&& v) noexcept
            {
                if(this == std::addressof(v))
                {
                    return *this;
                }

                swap(v);

                return *this;
            }

            __forceinline Opt& operator=(value_type v) { return *this = Opt(std::move(v)); }

            __forceinline void clear() noexcept
            {
                Opt empty;
                swap(empty);
            }

            __forceinline value_type const* get_ptr() const noexcept
            {
                if(is_initialized)
                {
                    return reinterpret_cast<value_type const*>(&storage);
                }
                return nullptr;
            }

            __forceinline value_type* get_ptr() noexcept
            {
                if(is_initialized)
                {
                    return reinterpret_cast<value_type*>(&storage);
                }
                return nullptr;
            }

            __forceinline value_type const& get() const noexcept
            {
                Assert(is_initialized);
                return *get_ptr();
            }

            __forceinline value_type& get() noexcept
            {
                Assert(is_initialized);
                return *get_ptr();
            }

            __forceinline bool has_value() const noexcept { return is_initialized; }

            // TODO: To be replaced with explicit operator bool ()
            typedef bool (Opt::*type_safe_bool_type)() const;

            __forceinline operator type_safe_bool_type() const noexcept { return is_initialized ? &Opt::has_value : nullptr; }

            __forceinline value_type const& operator*() const noexcept { return get(); }

            __forceinline value_type& operator*() noexcept { return get(); }

            __forceinline value_type const* operator->() const noexcept { return get_ptr(); }

            __forceinline value_type* operator->() noexcept { return get_ptr(); }

        private:
            typedef typename std::aligned_storage<sizeof(value_type), std::alignment_of<value_type>::value>::type storage_type;

            storage_type storage;
            bool         is_initialized;

            __forceinline static void move(storage_type* to, storage_type* from) noexcept
            {
                auto f = reinterpret_cast<value_type*>(from);
                new(to) value_type(std::move(*f));
                f->~value_type();
            }

            __forceinline static void copy(storage_type* to, storage_type const* from)
            {
                auto f = reinterpret_cast<value_type const*>(from);
                new(to) value_type(*f);
            }
        };

        // The generic interface

        // _range classes:
        //      inherit base_range
        //      COPYABLE
        //      MOVEABLE (movesemantics)
        //      typedef                 ...         this_type       ;
        //      typedef                 ...         value_type      ;
        //      typedef                 ...         return_type     ;   // value_type | value_type const &
        //      enum { returns_reference = 0|1 };
        //      return_type front () const
        //      bool next ()
        //      template<typename TRangeBuilder>
        //      typename get_builtup_type<TRangeBuilder, this_type>::type operator>>(TRangeBuilder range_builder) const

        // _builder classes:
        //      inherit base_builder
        //      COPYABLE
        //      MOVEABLE (movesemantics)
        //      typedef                 ...         this_type       ;
        //      template<typename TRange>
        //      TAggregated build (TRange range) const

        struct BaseRange
        {
#ifdef CPPLINQ_DETECT_INVALID_METHODS
        protected:
            // In order to prevent object slicing

            __forceinline base_range() noexcept {}

            __forceinline base_range(base_range const&) noexcept {}

            __forceinline base_range(base_range&&) noexcept {}

            __forceinline ~base_range() noexcept {}

        private:
            __forceinline base_range& operator=(base_range const&);
            __forceinline base_range& operator=(base_range&&);
#endif
        };

        struct BaseBuilder
        {
#ifdef CPPLINQ_DETECT_INVALID_METHODS
        protected:
            // In order to prevent object slicing

            __forceinline base_builder() noexcept {}

            __forceinline base_builder(base_builder const&) noexcept {}

            __forceinline base_builder(base_builder&&) noexcept {}

            __forceinline ~base_builder() noexcept {}

        private:
            __forceinline base_builder& operator=(base_builder const&);
            __forceinline base_builder& operator=(base_builder&&);
#endif
        };

        template<typename TValueIterator>
        struct FromRange : BaseRange
        {
            static TValueIterator GetIterator();

            typedef FromRange<TValueIterator> this_type;
            typedef TValueIterator            iterator_type;

            typedef decltype(*GetIterator())                   raw_value_type;
            typedef typename CleanupType<raw_value_type>::type value_type;
            typedef value_type const&                          return_type;
            enum
            {
                returns_reference = 1,
            };

            iterator_type current;
            iterator_type upcoming;
            iterator_type end;

            __forceinline FromRange(iterator_type begin, iterator_type end) noexcept : current(std::move(begin)), upcoming(current), end(std::move(end)) {}

            __forceinline FromRange(FromRange const& v) noexcept : current(v.current), upcoming(v.upcoming), end(v.end) {}

            __forceinline FromRange(FromRange&& v) noexcept : current(std::move(v.current)), upcoming(std::move(v.upcoming)), end(std::move(v.end)) {}

            template<typename TRangeBuilder>
            __forceinline typename GetBuiltupType<TRangeBuilder, this_type>::type operator>>(TRangeBuilder range_builder) const
            {
                return range_builder.build(*this);
            }

            __forceinline return_type Front() const
            {
                Assert(current != upcoming);
                Assert(current != end);

                return *current;
            }

            __forceinline bool Next() noexcept
            {
                if(upcoming == end)
                {
                    return false;
                }

                current = upcoming;
                ++upcoming;
                return true;
            }
        };

        template<typename TContainer>
        struct FromCopyRange : BaseRange
        {
            typedef FromCopyRange<TContainer> this_type;

            typedef TContainer                          container_type;
            typedef typename TContainer::const_iterator iterator_type;
            typedef typename TContainer::value_type     value_type;
            typedef value_type const&                   return_type;
            enum
            {
                returns_reference = 1,
            };

            container_type container;

            iterator_type current;
            iterator_type upcoming;
            iterator_type end;

            __forceinline FromCopyRange(container_type&& container) :
                container(std::move(container)), current(container.begin()), upcoming(container.begin()), end(container.end())
            {
            }

            __forceinline FromCopyRange(container_type const& container) : container(container), current(container.begin()), upcoming(container.begin()), end(container.end())
            {
            }

            __forceinline FromCopyRange(FromCopyRange const& v) : container(v.container), current(v.current), upcoming(v.upcoming), end(v.end) {}

            __forceinline FromCopyRange(FromCopyRange&& v) noexcept :
                container(std::move(v.container)), current(std::move(v.current)), upcoming(std::move(v.upcoming)), end(std::move(v.end))
            {
            }

            template<typename TRangeBuilder>
            __forceinline typename GetBuiltupType<TRangeBuilder, this_type>::type operator>>(TRangeBuilder range_builder) const
            {
                return range_builder.build(*this);
            }

            __forceinline return_type Front() const
            {
                Assert(current != upcoming);
                Assert(current != end);

                return *current;
            }

            __forceinline bool Next() noexcept
            {
                if(upcoming == end)
                {
                    return false;
                }

                current = upcoming;
                ++upcoming;
                return true;
            }
        };

        struct IntRange : BaseRange
        {
            typedef IntRange this_type;
            typedef int      value_type;
            typedef int      return_type;
            enum
            {
                returns_reference = 0,
            };

            int current;
            int end;

            static int GetCurrent(int begin, int end)
            {
                return (begin < end ? begin : end) - 1; // -1 in order to start one-step before the first element
            }

            static int GetEnd(int begin, int end) // -1 in order to avoid an extra test in next
            {
                return (begin < end ? end : begin) - 1;
            }

            __forceinline IntRange(const int begin, const int end) noexcept : current(GetCurrent(begin, end)), end(GetEnd(begin, end)) {}

            __forceinline IntRange(IntRange const& v) noexcept : current(v.current), end(v.end) {}

            __forceinline IntRange(IntRange&& v) noexcept : current(std::move(v.current)), end(std::move(v.end)) {}

            template<typename TRangeBuilder>
            __forceinline typename GetBuiltupType<TRangeBuilder, this_type>::type operator>>(TRangeBuilder range_builder) const
            {
                return range_builder.build(*this);
            }

            __forceinline return_type Front() const { return current; }

            __forceinline bool Next() noexcept
            {
                if(current >= end)
                {
                    return false;
                }

                ++current;

                return true;
            }
        };

        template<typename TValue>
        struct RepeatRange : BaseRange
        {
            typedef RepeatRange<TValue> this_type;
            typedef TValue              value_type;
            typedef TValue              return_type;
            enum
            {
                returns_reference = 0,
            };

            TValue    value;
            size_type remaining;

            __forceinline RepeatRange(value_type element, const size_type count) noexcept : value(std::move(element)), remaining(count) {}

            __forceinline RepeatRange(RepeatRange const& v) noexcept : value(v.value), remaining(v.remaining) {}

            __forceinline RepeatRange(RepeatRange&& v) noexcept : value(std::move(v.value)), remaining(std::move(v.remaining)) {}

            template<typename TRangeBuilder>
            __forceinline typename GetBuiltupType<TRangeBuilder, this_type>::type operator>>(TRangeBuilder range_builder) const
            {
                return range_builder.build(*this);
            }

            __forceinline return_type Front() const { return value; }

            __forceinline bool Next() noexcept
            {
                if(remaining == 0U)
                {
                    return false;
                }

                --remaining;

                return true;
            }
        };

        template<typename TValue>
        struct EmptyRange : BaseRange
        {
            typedef EmptyRange<TValue> this_type;
            typedef TValue             value_type;
            typedef TValue             return_type;
            enum
            {
                returns_reference = 0,
            };

            __forceinline EmptyRange() noexcept {}

            __forceinline EmptyRange(EmptyRange const& v) noexcept {}

            __forceinline EmptyRange(EmptyRange&& v) noexcept {}

            template<typename TRangeBuilder>
            __forceinline typename GetBuiltupType<TRangeBuilder, this_type>::type operator>>(TRangeBuilder range_builder) const
            {
                return range_builder.build(*this);
            }

            __forceinline return_type Front() const
            {
                Assert(false);
                throw ProgrammingErrorException();
            }

            __forceinline bool Next() noexcept { return false; }
        };

        template<typename TValue>
        struct SingletonRange : BaseRange
        {
            typedef SingletonRange<TValue> this_type;
            typedef TValue                 value_type;
            typedef TValue const&          return_type;

            enum
            {
                returns_reference = 1,
            };

            value_type value;
            bool       done;

            __forceinline SingletonRange(TValue const& value) : value(value), done(false) {}

            __forceinline SingletonRange(TValue&& value) noexcept : value(std::move(value)), done(false) {}

            __forceinline SingletonRange(SingletonRange const& v) noexcept : value(v.value), done(v.done) {}

            __forceinline SingletonRange(SingletonRange&& v) noexcept : value(std::move(v.value)), done(std::move(v.done)) {}

            template<typename TRangeBuilder>
            __forceinline typename GetBuiltupType<TRangeBuilder, this_type>::type operator>>(TRangeBuilder range_builder) const
            {
                return range_builder.build(*this);
            }

            __forceinline return_type Front() const noexcept { return value; }

            __forceinline bool Next() noexcept
            {
                const auto d = done;
                done         = true;
                return !d;
            }
        };

        struct SortingRange : BaseRange
        {
#ifdef CPPLINQ_DETECT_INVALID_METHODS
        protected:
            // In order to prevent object slicing

            __forceinline sorting_range() noexcept {}

            __forceinline sorting_range(sorting_range const&) noexcept {}

            __forceinline sorting_range(sorting_range&&) noexcept {}

            __forceinline ~sorting_range() noexcept {}

        private:
            __forceinline sorting_range& operator=(sorting_range const&);
            __forceinline sorting_range& operator=(sorting_range&&);
#endif
        };

        template<typename TRange, typename TPredicate>
        struct OrderbyRange : SortingRange
        {
            typedef OrderbyRange<TRange, TPredicate> this_type;
            typedef TRange                           range_type;
            typedef TPredicate                       predicate_type;

            typedef typename TRange::value_type  value_type;
            typedef typename TRange::return_type forwarding_return_type;
            typedef value_type const&            return_type;
            enum
            {
                forward_returns_reference = TRange::returns_reference,
                returns_reference         = 1,
            };

            range_type     range;
            predicate_type predicate;
            bool           sort_ascending;

            size_type               current;
            std::vector<value_type> sorted_values;

            __forceinline OrderbyRange(range_type range, predicate_type predicate, const bool sort_ascending) noexcept :
                range(std::move(range)), predicate(std::move(predicate)), sort_ascending(sort_ascending), current(invalid_size)
            {
                static_assert(!std::is_convertible<range_type, SortingRange>::value, "orderby may not follow orderby or thenby");
            }

            __forceinline OrderbyRange(OrderbyRange const& v) :
                range(v.range), predicate(v.predicate), sort_ascending(v.sort_ascending), current(v.current), sorted_values(v.sorted_values)
            {
            }

            __forceinline OrderbyRange(OrderbyRange&& v) noexcept :
                range(std::move(v.range)),
                predicate(std::move(v.predicate)),
                sort_ascending(std::move(v.sort_ascending)),
                current(std::move(v.current)),
                sorted_values(std::move(v.sorted_values))
            {
            }

            __forceinline forwarding_return_type ForwardingFront() const { return range.Front(); }

            __forceinline bool ForwardingNext() { return range.Next(); }

            __forceinline bool CompareValues(value_type const& l, value_type const& r) const
            {
                if(sort_ascending)
                {
                    return predicate(l) < predicate(r);
                }
                return predicate(r) < predicate(l);
            }

            template<typename TRangeBuilder>
            __forceinline typename GetBuiltupType<TRangeBuilder, this_type>::type operator>>(TRangeBuilder range_builder) const
            {
                return range_builder.build(*this);
            }

            __forceinline return_type Front() const { return sorted_values[current]; }

            KOKKOS_FUNCTION bool Next()
            {
                if(current == invalid_size)
                {
                    sorted_values.clear();

                    while(range.Next())
                    {
                        sorted_values.push_back(range.Front());
                    }

                    if(sorted_values.size() == 0)
                    {
                        return false;
                    }

                    std::sort(sorted_values.begin(), sorted_values.end(), [this](value_type const& l, value_type const& r) { return this->CompareValues(l, r); });

                    current = 0U;
                    return true;
                }

                if(current < sorted_values.size())
                {
                    ++current;
                }

                return current < sorted_values.size();
            }
        };

        template<typename TPredicate>
        struct OrderbyBuilder : BaseBuilder
        {
            typedef OrderbyBuilder<TPredicate> this_type;
            typedef TPredicate                 predicate_type;

            predicate_type predicate;
            bool           sort_ascending;

            __forceinline explicit OrderbyBuilder(predicate_type predicate, const bool sort_ascending) noexcept :
                predicate(std::move(predicate)), sort_ascending(sort_ascending)
            {
            }

            __forceinline OrderbyBuilder(OrderbyBuilder const& v) : predicate(v.predicate), sort_ascending(v.sort_ascending) {}

            __forceinline OrderbyBuilder(OrderbyBuilder&& v) noexcept : predicate(std::move(v.predicate)), sort_ascending(std::move(v.sort_ascending)) {}

            template<typename TRange>
            __forceinline OrderbyRange<TRange, TPredicate> Build(TRange range) const
            {
                return OrderbyRange<TRange, TPredicate>(std::move(range), predicate, sort_ascending);
            }
        };

        template<typename TRange, typename TPredicate>
        struct ThenbyRange : SortingRange
        {
            typedef ThenbyRange<TRange, TPredicate> this_type;
            typedef TRange                          range_type;
            typedef TPredicate                      predicate_type;

            typedef typename TRange::value_type             value_type;
            typedef typename TRange::forwarding_return_type forwarding_return_type;
            typedef value_type const&                       return_type;
            enum
            {
                forward_returns_reference = TRange::forward_returns_reference,
                returns_reference         = 1,
            };

            range_type     range;
            predicate_type predicate;
            bool           sort_ascending;

            size_type               current;
            std::vector<value_type> sorted_values;

            __forceinline ThenbyRange(range_type range, predicate_type predicate, const bool sort_ascending) noexcept :
                range(std::move(range)), predicate(std::move(predicate)), sort_ascending(sort_ascending), current(invalid_size)
            {
                static_assert(std::is_convertible<range_type, SortingRange>::value, "thenby may only follow orderby or thenby");
            }

            __forceinline ThenbyRange(ThenbyRange const& v) :
                range(v.range), predicate(v.predicate), sort_ascending(v.sort_ascending), current(v.current), sorted_values(v.sorted_values)
            {
            }

            __forceinline ThenbyRange(ThenbyRange&& v) noexcept :
                range(std::move(v.range)),
                predicate(std::move(v.predicate)),
                sort_ascending(std::move(v.sort_ascending)),
                current(std::move(v.current)),
                sorted_values(std::move(v.sorted_values))
            {
            }

            template<typename TRangeBuilder>
            __forceinline typename GetBuiltupType<TRangeBuilder, this_type>::type operator>>(TRangeBuilder range_builder) const
            {
                return range_builder.build(*this);
            }

            __forceinline forwarding_return_type ForwardingFront() const { return range.Front(); }

            __forceinline bool ForwardingNext() { return range.Next(); }

            __forceinline bool CompareValues(value_type const& l, value_type const& r) const
            {
                auto pless = range.compare_values(l, r);
                if(pless)
                {
                    return true;
                }

                auto pgreater = range.compare_values(r, l);
                if(pgreater)
                {
                    return false;
                }

                if(sort_ascending)
                {
                    return predicate(l) < predicate(r);
                }
                return predicate(r) < predicate(l);
            }

            __forceinline return_type Front() const { return sorted_values[current]; }

            KOKKOS_FUNCTION bool Next()
            {
                if(current == invalid_size)
                {
                    sorted_values.clear();

                    while(range.forwarding_next())
                    {
                        sorted_values.push_back(range.forwarding_front());
                    }

                    if(sorted_values.size() == 0)
                    {
                        return false;
                    }

                    std::sort(sorted_values.begin(), sorted_values.end(), [this](value_type const& l, value_type const& r) { return this->CompareValues(l, r); });

                    current = 0U;
                    return true;
                }

                if(current < sorted_values.size())
                {
                    ++current;
                }

                return current < sorted_values.size();
            }
        };

        template<typename TPredicate>
        struct ThenbyBuilder : BaseBuilder
        {
            typedef ThenbyBuilder<TPredicate> this_type;
            typedef TPredicate                predicate_type;

            predicate_type predicate;
            bool           sort_ascending;

            __forceinline explicit ThenbyBuilder(predicate_type predicate, bool sort_ascending) noexcept;

            __forceinline ThenbyBuilder(ThenbyBuilder const& v) : predicate(v.predicate), sort_ascending(v.sort_ascending) {}

            __forceinline ThenbyBuilder(ThenbyBuilder&& v) noexcept : predicate(std::move(v.predicate)), sort_ascending(std::move(v.sort_ascending)) {}

            template<typename TRange>
            __forceinline ThenbyRange<TRange, TPredicate> Build(TRange range) const
            {
                return ThenbyRange<TRange, TPredicate>(std::move(range), predicate, sort_ascending);
            }
        };

        template<typename TPredicate>
        ThenbyBuilder<TPredicate>::ThenbyBuilder(predicate_type predicate, const bool sort_ascending) noexcept : predicate(std::move(predicate)), sort_ascending(sort_ascending)
        {
        }

        template<typename TRange>
        struct ReverseRange : BaseRange
        {
            typedef ReverseRange<TRange> this_type;
            typedef TRange               range_type;

            typedef typename TRange::value_type value_type;
            typedef value_type const&           return_type;

            typedef std::vector<value_type> stack_type;

            enum
            {
                returns_reference = 1,
            };

            range_type              range;
            size_type               capacity;
            std::vector<value_type> reversed;
            bool                    start;

            __forceinline ReverseRange(range_type range, const size_type capacity) noexcept : range(std::move(range)), capacity(capacity), start(true) {}

            __forceinline ReverseRange(ReverseRange const& v) noexcept : range(v.range), capacity(v.capacity), reversed(v.reversed), start(v.start) {}

            __forceinline ReverseRange(ReverseRange&& v) noexcept :
                range(std::move(v.range)), capacity(std::move(v.capacity)), reversed(std::move(v.reversed)), start(std::move(v.start))
            {
            }

            template<typename TRangeBuilder>
            __forceinline typename GetBuiltupType<TRangeBuilder, this_type>::type operator>>(TRangeBuilder range_builder) const
            {
                return range_builder.build(*this);
            }

            __forceinline return_type Front() const noexcept
            {
                Assert(!start);
                Assert(!reversed.empty());
                return reversed[reversed.size() - 1];
            }

            __forceinline bool Next()
            {
                if(start)
                {
                    start = false;

                    reversed.clear();
                    reversed.reserve(capacity);

                    while(range.Next())
                    {
                        reversed.push_back(range.Front());
                    }

                    return !reversed.empty();
                }

                if(reversed.empty())
                {
                    return false;
                }

                reversed.pop_back();

                return !reversed.empty();
            }
        };

        struct ReverseBuilder : BaseBuilder
        {
            typedef ReverseBuilder this_type;

            size_type capacity;

            __forceinline ReverseBuilder(const size_type capacity) noexcept : capacity(capacity) {}

            __forceinline ReverseBuilder(ReverseBuilder const& v) noexcept : capacity(v.capacity) {}

            __forceinline ReverseBuilder(ReverseBuilder&& v) noexcept : capacity(std::move(v.capacity)) {}

            template<typename TRange>
            __forceinline ReverseRange<TRange> Build(TRange range) const
            {
                return ReverseRange<TRange>(std::move(range), capacity);
            }
        };

        template<typename TRange, typename TPredicate>
        struct WhereRange : BaseRange
        {
            typedef WhereRange<TRange, TPredicate> this_type;
            typedef TRange                         range_type;
            typedef TPredicate                     predicate_type;

            typedef typename TRange::value_type  value_type;
            typedef typename TRange::return_type return_type;
            enum
            {
                returns_reference = TRange::returns_reference,
            };

            range_type     range;
            predicate_type predicate;

            __forceinline WhereRange(range_type range, predicate_type predicate) noexcept : range(std::move(range)), predicate(std::move(predicate)) {}

            __forceinline WhereRange(WhereRange const& v) : range(v.range), predicate(v.predicate) {}

            __forceinline WhereRange(WhereRange&& v) noexcept : range(std::move(v.range)), predicate(std::move(v.predicate)) {}

            template<typename TRangeBuilder>
            __forceinline typename GetBuiltupType<TRangeBuilder, this_type>::type operator>>(TRangeBuilder range_builder) const
            {
                return range_builder.build(*this);
            }

            __forceinline return_type Front() const { return range.Front(); }

            __forceinline bool Next()
            {
                while(range.Next())
                {
                    if(predicate(range.Front()))
                    {
                        return true;
                    }
                }

                return false;
            }
        };

        template<typename TPredicate>
        struct WhereBuilder : BaseBuilder
        {
            typedef WhereBuilder<TPredicate> this_type;
            typedef TPredicate               predicate_type;

            predicate_type predicate;

            __forceinline explicit WhereBuilder(predicate_type predicate) noexcept : predicate(std::move(predicate)) {}

            __forceinline WhereBuilder(WhereBuilder const& v) : predicate(v.predicate) {}

            __forceinline WhereBuilder(WhereBuilder&& v) noexcept : predicate(std::move(v.predicate)) {}

            template<typename TRange>
            __forceinline WhereRange<TRange, TPredicate> Build(TRange range) const
            {
                return WhereRange<TRange, TPredicate>(std::move(range), predicate);
            }
        };

        template<typename TRange>
        struct TakeRange : BaseRange
        {
            typedef TakeRange<TRange> this_type;
            typedef TRange            range_type;

            typedef typename TRange::value_type  value_type;
            typedef typename TRange::return_type return_type;
            enum
            {
                returns_reference = TRange::returns_reference,
            };

            range_type range;
            size_type  count;
            size_type  current;

            __forceinline TakeRange(range_type range, size_type count) noexcept : range(std::move(range)), count(std::move(count)), current(0) {}

            __forceinline TakeRange(TakeRange const& v) : range(v.range), count(v.count), current(v.current) {}

            __forceinline TakeRange(TakeRange&& v) noexcept : range(std::move(v.range)), count(std::move(v.count)), current(std::move(v.current)) {}

            template<typename TRangeBuilder>
            __forceinline typename GetBuiltupType<TRangeBuilder, this_type>::type operator>>(TRangeBuilder range_builder) const
            {
                return range_builder.build(*this);
            }

            __forceinline return_type Front() const { return range.Front(); }

            __forceinline bool Next()
            {
                if(current >= count)
                {
                    return false;
                }

                ++current;
                return range.Next();
            }
        };

        struct TakeBuilder : BaseBuilder
        {
            typedef TakeBuilder this_type;

            size_type count;

            __forceinline explicit TakeBuilder(size_type count) noexcept : count(std::move(count)) {}

            __forceinline TakeBuilder(TakeBuilder const& v) noexcept : count(v.count) {}

            __forceinline TakeBuilder(TakeBuilder&& v) noexcept : count(std::move(v.count)) {}

            template<typename TRange>
            __forceinline TakeRange<TRange> Build(TRange range) const
            {
                return TakeRange<TRange>(std::move(range), count);
            }
        };

        template<typename TRange, typename TPredicate>
        struct TakeWhileRange : BaseRange
        {
            typedef TakeWhileRange<TRange, TPredicate> this_type;
            typedef TRange                             range_type;
            typedef TPredicate                         predicate_type;

            typedef typename TRange::value_type  value_type;
            typedef typename TRange::return_type return_type;
            enum
            {
                returns_reference = TRange::returns_reference,
            };

            range_type     range;
            predicate_type predicate;
            bool           done;

            __forceinline TakeWhileRange(range_type range, predicate_type predicate) noexcept : range(std::move(range)), predicate(std::move(predicate)), done(false) {}

            __forceinline TakeWhileRange(TakeWhileRange const& v) : range(v.range), predicate(v.predicate), done(v.done) {}

            __forceinline TakeWhileRange(TakeWhileRange&& v) noexcept : range(std::move(v.range)), predicate(std::move(v.predicate)), done(std::move(v.done)) {}

            template<typename TRangeBuilder>
            __forceinline typename GetBuiltupType<TRangeBuilder, this_type>::type operator>>(TRangeBuilder range_builder) const
            {
                return range_builder.build(*this);
            }

            __forceinline return_type Front() const { return range.Front(); }

            __forceinline bool Next()
            {
                if(done)
                {
                    return false;
                }

                if(!range.Next())
                {
                    done = true;
                    return false;
                }

                if(!predicate(range.Front()))
                {
                    done = true;
                    return false;
                }

                return true;
            }
        };

        template<typename TPredicate>
        struct TakeWhileBuilder : BaseBuilder
        {
            typedef TakeWhileBuilder<TPredicate> this_type;
            typedef TPredicate                   predicate_type;

            predicate_type predicate;

            __forceinline TakeWhileBuilder(predicate_type predicate) noexcept : predicate(std::move(predicate)) {}

            __forceinline TakeWhileBuilder(TakeWhileBuilder const& v) noexcept : predicate(v.predicate) {}

            __forceinline TakeWhileBuilder(TakeWhileBuilder&& v) noexcept : predicate(std::move(v.predicate)) {}

            template<typename TRange>
            __forceinline TakeWhileRange<TRange, TPredicate> Build(TRange range) const
            {
                return TakeWhileRange<TRange, TPredicate>(std::move(range), predicate);
            }
        };

        template<typename TRange>
        struct SkipRange : BaseRange
        {
            typedef SkipRange<TRange> this_type;
            typedef TRange            range_type;

            typedef typename TRange::value_type  value_type;
            typedef typename TRange::return_type return_type;
            enum
            {
                returns_reference = TRange::returns_reference,
            };

            range_type range;
            size_type  count;
            size_type  current;

            __forceinline SkipRange(range_type range, size_type count) noexcept : range(std::move(range)), count(std::move(count)), current(0) {}

            __forceinline SkipRange(SkipRange const& v) : range(v.range), count(v.count), current(v.current) {}

            __forceinline SkipRange(SkipRange&& v) noexcept : range(std::move(v.range)), count(std::move(v.count)), current(std::move(v.current)) {}

            template<typename TRangeBuilder>
            __forceinline typename GetBuiltupType<TRangeBuilder, this_type>::type operator>>(TRangeBuilder range_builder) const
            {
                return range_builder.build(*this);
            }

            __forceinline return_type Front() const { return range.Front(); }

            __forceinline bool Next()
            {
                if(current == invalid_size)
                {
                    return false;
                }

                while(current < count && range.Next())
                {
                    ++current;
                }

                if(current < count)
                {
                    current = invalid_size;
                    return false;
                }

                return range.Next();
            }
        };

        struct SkipBuilder : BaseBuilder
        {
            typedef SkipBuilder this_type;

            size_type count;

            __forceinline explicit SkipBuilder(size_type count) noexcept : count(std::move(count)) {}

            __forceinline SkipBuilder(SkipBuilder const& v) noexcept : count(v.count) {}

            __forceinline SkipBuilder(SkipBuilder&& v) noexcept : count(std::move(v.count)) {}

            template<typename TRange>
            __forceinline SkipRange<TRange> Build(TRange range) const
            {
                return SkipRange<TRange>(std::move(range), count);
            }
        };

        template<typename TRange, typename TPredicate>
        struct SkipWhileRange : BaseRange
        {
            typedef SkipWhileRange<TRange, TPredicate> this_type;
            typedef TRange                             range_type;
            typedef TPredicate                         predicate_type;

            typedef typename TRange::value_type  value_type;
            typedef typename TRange::return_type return_type;
            enum
            {
                returns_reference = TRange::returns_reference,
            };

            range_type     range;
            predicate_type predicate;
            bool           skipping;

            __forceinline SkipWhileRange(range_type range, predicate_type predicate) noexcept : range(std::move(range)), predicate(std::move(predicate)), skipping(true) {}

            __forceinline SkipWhileRange(SkipWhileRange const& v) : range(v.range), predicate(v.predicate), skipping(v.skipping) {}

            __forceinline SkipWhileRange(SkipWhileRange&& v) noexcept : range(std::move(v.range)), predicate(std::move(v.predicate)), skipping(std::move(v.skipping)) {}

            template<typename TRangeBuilder>
            __forceinline typename GetBuiltupType<TRangeBuilder, this_type>::type operator>>(TRangeBuilder range_builder) const
            {
                return range_builder.build(*this);
            }

            __forceinline return_type Front() const { return range.Front(); }

            __forceinline bool Next()
            {
                if(!skipping)
                {
                    return range.Next();
                }

                while(range.Next())
                {
                    if(!predicate(range.Front()))
                    {
                        skipping = false;
                        return true;
                    }
                }

                return false;
            }
        };

        template<typename TPredicate>
        struct SkipWhileBuilder : BaseBuilder
        {
            typedef SkipWhileBuilder<TPredicate> this_type;
            typedef TPredicate                   predicate_type;

            predicate_type predicate;

            __forceinline SkipWhileBuilder(predicate_type predicate) noexcept : predicate(std::move(predicate)) {}

            __forceinline SkipWhileBuilder(SkipWhileBuilder const& v) noexcept : predicate(v.predicate) {}

            __forceinline SkipWhileBuilder(SkipWhileBuilder&& v) noexcept : predicate(std::move(v.predicate)) {}

            template<typename TRange>
            __forceinline SkipWhileRange<TRange, TPredicate> Build(TRange range) const
            {
                return SkipWhileRange<TRange, TPredicate>(std::move(range), predicate);
            }
        };

        template<typename TRange>
        struct RefRange : BaseRange
        {
            typedef std::reference_wrapper<typename TRange::value_type const> value_type;
            typedef value_type                                                return_type;
            enum
            {
                returns_reference = 0,
            };

            typedef RefRange<TRange> this_type;
            typedef TRange           range_type;

            range_type range;

            __forceinline RefRange(range_type range) noexcept : range(std::move(range))
            {
                static_assert(TRange::returns_reference, "ref may only follow a range that returns references");
            }

            __forceinline RefRange(RefRange const& v) : range(v.range) {}

            __forceinline RefRange(RefRange&& v) noexcept : range(std::move(v.range)) {}

            template<typename TRangeBuilder>
            __forceinline typename GetBuiltupType<TRangeBuilder, this_type>::type operator>>(TRangeBuilder range_builder) const
            {
                return range_builder.build(*this);
            }

            __forceinline return_type Front() const { return value_type(range.Front()); }

            __forceinline bool Next() { return range.Next(); }
        };

        struct RefBuilder : BaseBuilder
        {
            typedef RefBuilder this_type;

            __forceinline RefBuilder() noexcept {}

            __forceinline RefBuilder(RefBuilder const& v) {}

            __forceinline RefBuilder(RefBuilder&& v) noexcept {}

            template<typename TRange>
            __forceinline RefRange<TRange> Build(TRange range) const
            {
                return RefRange<TRange>(std::move(range));
            }
        };

        template<typename TRange, typename TPredicate>
        struct SelectRange : BaseRange
        {
            static typename TRange::value_type GetSource();
            static TPredicate                  GetPredicate();

            typedef decltype(GetPredicate()(GetSource()))      raw_value_type;
            typedef typename CleanupType<raw_value_type>::type value_type;
            typedef value_type const&                          return_type;
            enum
            {
                returns_reference = 1,
            };

            typedef SelectRange<TRange, TPredicate> this_type;
            typedef TRange                          range_type;
            typedef TPredicate                      predicate_type;

            range_type     range;
            predicate_type predicate;

            Opt<value_type> cache_value;

            __forceinline SelectRange(range_type range, predicate_type predicate) noexcept : range(std::move(range)), predicate(std::move(predicate)) {}

            __forceinline SelectRange(SelectRange const& v) : range(v.range), predicate(v.predicate), cache_value(v.cache_value) {}

            __forceinline SelectRange(SelectRange&& v) noexcept : range(std::move(v.range)), predicate(std::move(v.predicate)), cache_value(std::move(v.cache_value)) {}

            template<typename TRangeBuilder>
            __forceinline typename GetBuiltupType<TRangeBuilder, this_type>::type operator>>(TRangeBuilder range_builder) const
            {
                return range_builder.build(*this);
            }

            __forceinline return_type Front() const
            {
                Assert(cache_value);
                return *cache_value;
            }

            __forceinline bool Next()
            {
                if(range.Next())
                {
                    cache_value = predicate(range.Front());
                    return true;
                }

                cache_value.clear();

                return false;
            }
        };

        template<typename TPredicate>
        struct SelectBuilder : BaseBuilder
        {
            typedef SelectBuilder<TPredicate> this_type;
            typedef TPredicate                predicate_type;

            predicate_type predicate;

            __forceinline explicit SelectBuilder(predicate_type predicate) noexcept : predicate(std::move(predicate)) {}

            __forceinline SelectBuilder(SelectBuilder const& v) : predicate(v.predicate) {}

            __forceinline SelectBuilder(SelectBuilder&& v) noexcept : predicate(std::move(v.predicate)) {}

            template<typename TRange>
            __forceinline SelectRange<TRange, TPredicate> Build(TRange range) const
            {
                return SelectRange<TRange, TPredicate>(std::move(range), predicate);
            }
        };

        // Some trickery in order to force the code to compile on VS2012
        template<typename TRange, typename TPredicate>
        struct SelectManyRangeHelper
        {
            static typename TRange::value_type GetSource();
            static TPredicate                  GetPredicate();

            typedef decltype(GetPredicate()(GetSource()))            raw_inner_range_type;
            typedef typename CleanupType<raw_inner_range_type>::type inner_range_type;

            static inner_range_type GetInnerRange();

            typedef decltype(GetInnerRange().Front())          raw_value_type;
            typedef typename CleanupType<raw_value_type>::type value_type;
        };

        template<typename TRange, typename TPredicate>
        struct SelectManyRange : BaseRange
        {
            typedef SelectManyRangeHelper<TRange, TPredicate> helper_type;

            typedef typename helper_type::inner_range_type inner_range_type;
            typedef typename helper_type::value_type       value_type;
            typedef value_type                             return_type;
            enum
            {
                returns_reference = 0,
            };

            typedef SelectManyRange<TRange, TPredicate> this_type;
            typedef TRange                              range_type;
            typedef TPredicate                          predicate_type;

            range_type     range;
            predicate_type predicate;

            Opt<inner_range_type> inner_range;

            __forceinline SelectManyRange(range_type range, predicate_type predicate) noexcept : range(std::move(range)), predicate(std::move(predicate)) {}

            __forceinline SelectManyRange(SelectManyRange const& v) : range(v.range), predicate(v.predicate), inner_range(v.inner_range) {}

            __forceinline SelectManyRange(SelectManyRange&& v) noexcept : range(std::move(v.range)), predicate(std::move(v.predicate)), inner_range(std::move(v.inner_range)) {}

            template<typename TRangeBuilder>
            __forceinline typename GetBuiltupType<TRangeBuilder, this_type>::type operator>>(TRangeBuilder range_builder) const
            {
                return range_builder.build(*this);
            }

            __forceinline return_type Front() const
            {
                Assert(inner_range);
                return inner_range->Front();
            }

            __forceinline bool Next()
            {
                if(inner_range && inner_range->Next())
                {
                    return true;
                }

                if(range.Next())
                {
                    inner_range = predicate(range.Front());
                    return inner_range && inner_range->Next();
                }

                inner_range.clear();

                return false;
            }
        };

        template<typename TPredicate>
        struct SelectManyBuilder : BaseBuilder
        {
            typedef SelectManyBuilder<TPredicate> this_type;
            typedef TPredicate                    predicate_type;

            predicate_type predicate;

            __forceinline explicit SelectManyBuilder(predicate_type predicate) noexcept : predicate(std::move(predicate)) {}

            __forceinline SelectManyBuilder(SelectManyBuilder const& v) : predicate(v.predicate) {}

            __forceinline SelectManyBuilder(SelectManyBuilder&& v) noexcept : predicate(std::move(v.predicate)) {}

            template<typename TRange>
            __forceinline SelectManyRange<TRange, TPredicate> Build(TRange range) const
            {
                return SelectManyRange<TRange, TPredicate>(std::move(range), predicate);
            }
        };

        template<typename TRange, typename TOtherRange, typename TKeySelector, typename TOtherKeySelector, typename TCombiner>
        struct JoinRange : BaseRange
        {
            static typename TRange::value_type      GetSource();
            static typename TOtherRange::value_type GetOtherSource();
            static TKeySelector                     GetKeySelector();
            static TOtherKeySelector                GetOtherKeySelector();
            static TCombiner                        GetCombiner();

            typedef decltype(GetKeySelector()(GetSource()))  raw_key_type;
            typedef typename CleanupType<raw_key_type>::type key_type;

            typedef decltype(GetOtherKeySelector()(GetOtherSource())) raw_other_key_type;
            typedef typename CleanupType<raw_other_key_type>::type    other_key_type;

            typedef decltype(GetCombiner()(GetSource(), GetOtherSource())) raw_value_type;
            typedef typename CleanupType<raw_value_type>::type             value_type;
            typedef value_type                                             return_type;
            enum
            {
                returns_reference = 0,
            };

            typedef JoinRange<TRange, TOtherRange, TKeySelector, TOtherKeySelector, TCombiner> this_type;
            typedef TRange                                                                     range_type;
            typedef TOtherRange                                                                other_range_type;
            typedef TKeySelector                                                               key_selector_type;
            typedef TOtherKeySelector                                                          other_key_selector_type;
            typedef TCombiner                                                                  combiner_type;
            typedef std::multimap<other_key_type, typename TOtherRange::value_type>            map_type;
            typedef typename map_type::const_iterator                                          map_iterator_type;

            range_type              range;
            other_range_type        other_range;
            key_selector_type       key_selector;
            other_key_selector_type other_key_selector;
            combiner_type           combiner;

            bool              start;
            map_type          map;
            map_iterator_type current;

            __forceinline JoinRange(range_type              range,
                                             other_range_type        other_range,
                                             key_selector_type       key_selector,
                                             other_key_selector_type other_key_selector,
                                             combiner_type           combiner) noexcept :
                range(std::move(range)),
                other_range(std::move(other_range)),
                key_selector(std::move(key_selector)),
                other_key_selector(std::move(other_key_selector)),
                combiner(std::move(combiner)),
                start(true)
            {
            }

            __forceinline JoinRange(JoinRange const& v) :
                range(v.range),
                other_range(v.other_range),
                key_selector(v.key_selector),
                other_key_selector(v.other_key_selector),
                combiner(v.combiner),
                start(v.start),
                map(v.map),
                current(v.current)
            {
            }

            __forceinline JoinRange(JoinRange&& v) noexcept :
                range(std::move(v.range)),
                other_range(std::move(v.other_range)),
                key_selector(std::move(v.key_selector)),
                other_key_selector(std::move(v.other_key_selector)),
                combiner(std::move(v.combiner)),
                start(std::move(v.start)),
                map(std::move(v.map)),
                current(std::move(v.current))
            {
            }

            template<typename TRangeBuilder>
            __forceinline typename GetBuiltupType<TRangeBuilder, this_type>::type operator>>(TRangeBuilder range_builder) const
            {
                return range_builder.build(*this);
            }

            __forceinline return_type Front() const
            {
                Assert(current != map.end());
                return combiner(range.Front(), current->second);
            }

            __forceinline bool Next()
            {
                if(start)
                {
                    start = false;
                    while(other_range.Next())
                    {
                        auto other_value = other_range.Front();
                        auto other_key   = other_key_selector(other_value);
                        map.insert(typename map_type::value_type(std::move(other_key), std::move(other_value)));
                    }

                    current = map.end();
                    if(map.size() == 0U)
                    {
                        return false;
                    }
                }

                if(current != map.end())
                {
                    auto previous = current;
                    ++current;
                    if(current != map.end() && !(previous->first < current->first))
                    {
                        return true;
                    }
                }

                while(range.Next())
                {
                    auto value = range.Front();
                    auto key   = key_selector(value);

                    current = map.find(key);
                    if(current != map.end())
                    {
                        return true;
                    }
                }

                return false;
            }
        };

        template<typename TOtherRange, typename TKeySelector, typename TOtherKeySelector, typename TCombiner>
        struct JoinBuilder : BaseBuilder
        {
            typedef JoinBuilder<TOtherRange, TKeySelector, TOtherKeySelector, TCombiner> this_type;

            typedef TOtherRange       other_range_type;
            typedef TKeySelector      key_selector_type;
            typedef TOtherKeySelector other_key_selector_type;
            typedef TCombiner         combiner_type;

            other_range_type        other_range;
            key_selector_type       key_selector;
            other_key_selector_type other_key_selector;
            combiner_type           combiner;

            __forceinline JoinBuilder(other_range_type other_range, key_selector_type key_selector, other_key_selector_type other_key_selector, combiner_type combiner) noexcept
                :
                other_range(std::move(other_range)), key_selector(std::move(key_selector)), other_key_selector(std::move(other_key_selector)), combiner(std::move(combiner))
            {
            }

            __forceinline JoinBuilder(JoinBuilder const& v) :
                other_range(v.other_range), key_selector(v.key_selector), other_key_selector(v.other_key_selector), combiner(v.combiner)
            {
            }

            __forceinline JoinBuilder(JoinBuilder&& v) noexcept :
                other_range(std::move(v.other_range)), key_selector(std::move(v.key_selector)), other_key_selector(std::move(v.other_key_selector)), combiner(std::move(v.combiner))
            {
            }

            template<typename TRange>
            __forceinline JoinRange<TRange, TOtherRange, TKeySelector, TOtherKeySelector, TCombiner> Build(TRange range) const
            {
                return JoinRange<TRange, TOtherRange, TKeySelector, TOtherKeySelector, TCombiner>(std::move(range), other_range, key_selector, other_key_selector, combiner);
            }
        };

        template<typename TRange>
        struct DistinctRange : BaseRange
        {
            typedef DistinctRange<TRange> this_type;
            typedef TRange                range_type;

            typedef typename CleanupType<typename TRange::value_type>::type value_type;
            typedef value_type const&                                       return_type;
            enum
            {
                returns_reference = 1,
            };

            typedef std::set<value_type>              set_type;
            typedef typename set_type::const_iterator set_iterator_type;

            range_type        range;
            set_type          set;
            set_iterator_type current;

            __forceinline DistinctRange(range_type range) noexcept : range(std::move(range)) {}

            __forceinline DistinctRange(DistinctRange const& v) noexcept : range(v.range), set(v.set), current(v.current) {}

            __forceinline DistinctRange(DistinctRange&& v) noexcept : range(std::move(v.range)), set(std::move(v.set)), current(std::move(v.current)) {}

            template<typename TRangeBuilder>
            __forceinline typename GetBuiltupType<TRangeBuilder, this_type>::type operator>>(TRangeBuilder range_builder) const
            {
                return range_builder.build(*this);
            }

            __forceinline return_type Front() const { return *current; }

            __forceinline bool Next()
            {
                while(range.Next())
                {
                    auto result = set.insert(range.Front());
                    if(result.second)
                    {
                        current = result.first;
                        return true;
                    }
                }

                return false;
            }
        };

        struct DistinctBuilder : BaseBuilder
        {
            typedef DistinctBuilder this_type;

            __forceinline DistinctBuilder() noexcept {}

            __forceinline DistinctBuilder(DistinctBuilder const& v) noexcept {}

            __forceinline DistinctBuilder(DistinctBuilder&& v) noexcept {}

            template<typename TRange>
            __forceinline DistinctRange<TRange> Build(TRange range) const
            {
                return DistinctRange<TRange>(std::move(range));
            }
        };

        template<typename TRange, typename TOtherRange>
        struct UnionRange : BaseRange
        {
            typedef UnionRange<TRange, TOtherRange> this_type;
            typedef TRange                          range_type;
            typedef TOtherRange                     other_range_type;

            typedef typename CleanupType<typename TRange::value_type>::type value_type;
            typedef value_type const&                                       return_type;
            enum
            {
                returns_reference = 1,
            };

            typedef std::set<value_type>              set_type;
            typedef typename set_type::const_iterator set_iterator_type;

            range_type        range;
            other_range_type  other_range;
            set_type          set;
            set_iterator_type current;

            __forceinline UnionRange(range_type range, other_range_type other_range) noexcept : range(std::move(range)), other_range(std::move(other_range)) {}

            __forceinline UnionRange(UnionRange const& v) noexcept : range(v.range), other_range(v.other_range), set(v.set), current(v.current) {}

            __forceinline UnionRange(UnionRange&& v) noexcept :
                range(std::move(v.range)), other_range(std::move(v.other_range)), set(std::move(v.set)), current(std::move(v.current))
            {
            }

            template<typename TRangeBuilder>
            __forceinline typename GetBuiltupType<TRangeBuilder, this_type>::type operator>>(TRangeBuilder range_builder) const
            {
                return range_builder.build(*this);
            }

            __forceinline return_type Front() const { return *current; }

            __forceinline bool Next()
            {
                while(range.Next())
                {
                    auto result = set.insert(range.Front());
                    if(result.second)
                    {
                        current = result.first;
                        return true;
                    }
                }

                while(other_range.Next())
                {
                    auto result = set.insert(other_range.Front());
                    if(result.second)
                    {
                        current = result.first;
                        return true;
                    }
                }

                return false;
            }
        };

        template<typename TOtherRange>
        struct UnionBuilder : BaseBuilder
        {
            typedef UnionBuilder<TOtherRange> this_type;
            typedef TOtherRange               other_range_type;

            other_range_type other_range;

            __forceinline UnionBuilder(TOtherRange other_range) noexcept : other_range(std::move(other_range)) {}

            __forceinline UnionBuilder(UnionBuilder const& v) noexcept : other_range(v.other_range) {}

            __forceinline UnionBuilder(UnionBuilder&& v) noexcept : other_range(std::move(v.other_range)) {}

            template<typename TRange>
            __forceinline UnionRange<TRange, TOtherRange> Build(TRange range) const
            {
                return UnionRange<TRange, TOtherRange>(std::move(range), std::move(other_range));
            }
        };

        template<typename TRange, typename TOtherRange>
        struct IntersectRange : BaseRange
        {
            typedef IntersectRange<TRange, TOtherRange> this_type;
            typedef TRange                              range_type;
            typedef TOtherRange                         other_range_type;

            typedef typename CleanupType<typename TRange::value_type>::type value_type;
            typedef value_type const&                                       return_type;
            enum
            {
                returns_reference = 1,
            };

            typedef std::set<value_type>              set_type;
            typedef typename set_type::const_iterator set_iterator_type;

            range_type        range;
            other_range_type  other_range;
            set_type          set;
            set_iterator_type current;
            bool              start;

            __forceinline IntersectRange(range_type range, other_range_type other_range) noexcept : range(std::move(range)), other_range(std::move(other_range)), start(true) {}

            __forceinline IntersectRange(IntersectRange const& v) noexcept : range(v.range), other_range(v.other_range), set(v.set), current(v.current), start(v.start) {}

            __forceinline IntersectRange(IntersectRange&& v) noexcept :
                range(std::move(v.range)), other_range(std::move(v.other_range)), set(std::move(v.set)), current(std::move(v.current)), start(std::move(v.start))
            {
            }

            template<typename TRangeBuilder>
            __forceinline typename GetBuiltupType<TRangeBuilder, this_type>::type operator>>(TRangeBuilder range_builder) const
            {
                return range_builder.build(*this);
            }

            __forceinline return_type Front() const
            {
                Assert(!start);
                return *current;
            }

            __forceinline bool Next()
            {
                if(start)
                {
                    start = false;

                    while(other_range.Next())
                    {
                        set.insert(other_range.Front());
                    }

                    while(range.Next())
                    {
                        current = set.find(range.Front());
                        if(current != set.end())
                        {
                            return true;
                        }
                    }

                    set.clear();

                    return false;
                }

                if(set.empty())
                {
                    return false;
                }

                set.erase(current);

                while(range.Next())
                {
                    current = set.find(range.Front());
                    if(current != set.end())
                    {
                        return true;
                    }
                }

                return false;
            }
        };

        template<typename TOtherRange>
        struct IntersectBuilder : BaseBuilder
        {
            typedef IntersectBuilder<TOtherRange> this_type;
            typedef TOtherRange                   other_range_type;

            other_range_type other_range;

            __forceinline IntersectBuilder(TOtherRange other_range) noexcept : other_range(std::move(other_range)) {}

            __forceinline IntersectBuilder(IntersectBuilder const& v) noexcept : other_range(v.other_range) {}

            __forceinline IntersectBuilder(IntersectBuilder&& v) noexcept : other_range(std::move(v.other_range)) {}

            template<typename TRange>
            __forceinline IntersectRange<TRange, TOtherRange> Build(TRange range) const
            {
                return IntersectRange<TRange, TOtherRange>(std::move(range), std::move(other_range));
            }
        };

        template<typename TRange, typename TOtherRange>
        struct ExceptRange : BaseRange
        {
            typedef ExceptRange<TRange, TOtherRange> this_type;
            typedef TRange                           range_type;
            typedef TOtherRange                      other_range_type;

            typedef typename CleanupType<typename TRange::value_type>::type value_type;
            typedef value_type const&                                       return_type;
            enum
            {
                returns_reference = 1,
            };

            typedef std::set<value_type>              set_type;
            typedef typename set_type::const_iterator set_iterator_type;

            range_type        range;
            other_range_type  other_range;
            set_type          set;
            set_iterator_type current;
            bool              start;

            __forceinline ExceptRange(range_type range, other_range_type other_range) noexcept : range(std::move(range)), other_range(std::move(other_range)), start(true) {}

            __forceinline ExceptRange(ExceptRange const& v) noexcept : range(v.range), other_range(v.other_range), set(v.set), current(v.current), start(v.start) {}

            __forceinline ExceptRange(ExceptRange&& v) noexcept :
                range(std::move(v.range)), other_range(std::move(v.other_range)), set(std::move(v.set)), current(std::move(v.current)), start(std::move(v.start))
            {
            }

            template<typename TRangeBuilder>
            __forceinline typename GetBuiltupType<TRangeBuilder, this_type>::type operator>>(TRangeBuilder range_builder) const
            {
                return range_builder.build(*this);
            }

            __forceinline return_type Front() const { return *current; }

            __forceinline bool Next()
            {
                if(start)
                {
                    start = false;
                    while(other_range.Next())
                    {
                        set.insert(other_range.Front());
                    }
                }

                while(range.Next())
                {
                    auto result = set.insert(range.Front());
                    if(result.second)
                    {
                        current = result.first;
                        return true;
                    }
                }

                return false;
            }
        };

        template<typename TOtherRange>
        struct ExceptBuilder : BaseBuilder
        {
            typedef UnionBuilder<TOtherRange> this_type;
            typedef TOtherRange               other_range_type;

            other_range_type other_range;

            __forceinline ExceptBuilder(TOtherRange other_range) noexcept : other_range(std::move(other_range)) {}

            __forceinline ExceptBuilder(ExceptBuilder const& v) noexcept : other_range(v.other_range) {}

            __forceinline ExceptBuilder(ExceptBuilder&& v) noexcept : other_range(std::move(v.other_range)) {}

            template<typename TRange>
            __forceinline ExceptRange<TRange, TOtherRange> Build(TRange range) const
            {
                return ExceptRange<TRange, TOtherRange>(std::move(range), std::move(other_range));
            }
        };

        template<typename TRange, typename TOtherRange>
        struct ConcatRange : BaseRange
        {
            typedef ConcatRange<TRange, TOtherRange> this_type;
            typedef TRange                           range_type;
            typedef TOtherRange                      other_range_type;

            typedef typename CleanupType<typename TRange::value_type>::type      value_type;
            typedef typename CleanupType<typename TOtherRange::value_type>::type other_value_type;
            typedef value_type                                                   return_type;

            enum
            {
                returns_reference = 0,
            };

            enum state
            {
                state_initial,
                state_iterating_range,
                state_iterating_other_range,
                state_end,
            };

            range_type       range;
            other_range_type other_range;
            state            state;

            __forceinline ConcatRange(range_type range, other_range_type other_range) noexcept :
                range(std::move(range)), other_range(std::move(other_range)), state(state_initial)
            {
            }

            __forceinline ConcatRange(ConcatRange const& v) noexcept : range(v.range), other_range(v.other_range), state(v.state) {}

            __forceinline ConcatRange(ConcatRange&& v) noexcept : range(std::move(v.range)), other_range(std::move(v.other_range)), state(std::move(v.state)) {}

            template<typename TRangeBuilder>
            __forceinline typename GetBuiltupType<TRangeBuilder, this_type>::type operator>>(TRangeBuilder range_builder) const
            {
                return range_builder.build(*this);
            }

            __forceinline return_type Front() const
            {
                switch(state)
                {
                    case state_initial:
                    case state_end:
                    default: Assert(false); // Intentionally falls through
                    case state_iterating_range: return range.Front();
                    case state_iterating_other_range: return other_range.Front();
                };
            }

            __forceinline bool Next()
            {
                switch(state)
                {
                    case state_initial:
                        if(range.Next())
                        {
                            state = state_iterating_range;
                            return true;
                        }

                        if(other_range.Next())
                        {
                            state = state_iterating_other_range;
                            return true;
                        }

                        state = state_end;
                        return false;
                    case state_iterating_range:
                        if(range.Next())
                        {
                            return true;
                        }

                        if(other_range.Next())
                        {
                            state = state_iterating_other_range;
                            return true;
                        }

                        state = state_end;
                        return false;
                    case state_iterating_other_range:
                        if(other_range.Next())
                        {
                            return true;
                        }

                        state = state_end;
                        return false;
                    case state_end:
                    default: return false;
                }
            }
        };

        template<typename TOtherRange>
        struct ConcatBuilder : BaseBuilder
        {
            typedef ConcatBuilder<TOtherRange> this_type;
            typedef TOtherRange                other_range_type;

            other_range_type other_range;

            __forceinline ConcatBuilder(TOtherRange other_range) noexcept : other_range(std::move(other_range)) {}

            __forceinline ConcatBuilder(ConcatBuilder const& v) noexcept : other_range(v.other_range) {}

            __forceinline ConcatBuilder(ConcatBuilder&& v) noexcept : other_range(std::move(v.other_range)) {}

            template<typename TRange>
            __forceinline ConcatRange<TRange, TOtherRange> Build(TRange range) const
            {
                return ConcatRange<TRange, TOtherRange>(std::move(range), std::move(other_range));
            }
        };

        namespace experimental
        {
            // TODO: Verify that container range aggregator has the right semantics

            template<typename TRange>
            struct ContainerIterator
            {
                typedef std::forward_iterator_tag    iterator_category;
                typedef typename TRange::value_type  value_type;
                typedef typename TRange::return_type return_type;
                enum
                {
                    returns_reference = TRange::returns_reference,
                };

                typedef std::ptrdiff_t difference_type;
                typedef value_type*    pointer;
                typedef value_type&    reference;

                typedef ContainerIterator<TRange> this_type;
                typedef TRange                    range_type;

                bool            has_value;
                Opt<range_type> range;

                __forceinline ContainerIterator() noexcept : has_value(false) {}

                __forceinline ContainerIterator(range_type r) noexcept : range(std::move(r)) { has_value = range && range->Next(); }

                __forceinline ContainerIterator(ContainerIterator const& v) noexcept : has_value(v.has_value), range(v.range) {}

                __forceinline ContainerIterator(ContainerIterator&& v) noexcept : has_value(std::move(v.has_value)), range(std::move(v.range)) {}

                __forceinline return_type operator*() const
                {
                    Assert(has_value);
                    Assert(range);
                    return range->Front();
                }

                __forceinline value_type const* operator->() const
                {
                    static_assert(returns_reference, "operator-> requires a range that returns a reference, typically select causes ranges to return values not references");
                    return &range->Front();
                }

                __forceinline this_type& operator++()
                {
                    if(has_value && range)
                    {
                        has_value = range->Next();
                    }

                    return *this;
                }

                __forceinline bool operator==(this_type const& v) const noexcept
                {
                    if(!has_value && !v.has_value)
                    {
                        return true;
                    }

                    if(has_value && v.has_value && range.get_ptr() == v.range.get_ptr())
                    {
                        return true;
                    }

                    return false;
                }

                __forceinline bool operator!=(this_type const& v) const noexcept { return !(*this == v); }
            };

            template<typename TRange>
            struct Container
            {
                typedef Container<TRange>            this_type;
                typedef TRange                       range_type;
                typedef typename TRange::value_type  value_type;
                typedef typename TRange::return_type return_type;
                enum
                {
                    returns_reference = TRange::returns_reference,
                };

                range_type range;

                __forceinline explicit Container(TRange range) : range(std::move(range)) {}

                __forceinline Container(Container const& v) noexcept : range(v.range) {}

                __forceinline Container(Container&& v) noexcept : range(std::move(v.range)) {}

                __forceinline ContainerIterator<TRange> begin() noexcept { return ContainerIterator<TRange>(range); }

                __forceinline ContainerIterator<TRange> end() noexcept { return ContainerIterator<TRange>(); }
            };

            struct ContainerBuilder : BaseBuilder
            {
                typedef ContainerBuilder this_type;

                __forceinline ContainerBuilder() noexcept {}

                __forceinline ContainerBuilder(ContainerBuilder const& v) noexcept {}

                __forceinline ContainerBuilder(ContainerBuilder&& v) noexcept {}

                template<typename TRange>
                KOKKOS_FUNCTION Container<TRange> Build(TRange range) const
                {
                    return Container<TRange>(std::move(range));
                }
            };
        }

        struct ToVectorBuilder : BaseBuilder
        {
            typedef ToVectorBuilder this_type;

            size_type capacity;

            __forceinline explicit ToVectorBuilder(const size_type capacity = 16U) noexcept : capacity(capacity) {}

            __forceinline ToVectorBuilder(ToVectorBuilder const& v) noexcept : capacity(v.capacity) {}

            __forceinline ToVectorBuilder(ToVectorBuilder&& v) noexcept : capacity(std::move(v.capacity)) {}

            template<typename TRange>
            KOKKOS_FUNCTION std::vector<typename TRange::value_type> Build(TRange range) const
            {
                std::vector<typename TRange::value_type> result;
                result.reserve(capacity);

                while(range.Next())
                {
                    result.push_back(range.Front());
                }

                return result;
            }
        };

        struct ToListBuilder : BaseBuilder
        {
            typedef ToListBuilder this_type;

            __forceinline explicit ToListBuilder() noexcept {}

            __forceinline ToListBuilder(ToListBuilder const& v) noexcept {}

            __forceinline ToListBuilder(ToListBuilder&& v) noexcept {}

            template<typename TRange>
            KOKKOS_FUNCTION std::list<typename TRange::value_type> Build(TRange range) const
            {
                std::list<typename TRange::value_type> result;

                while(range.Next())
                {
                    result.push_back(range.Front());
                }

                return result;
            }
        };

        template<typename TKeyPredicate>
        struct ToMapBuilder : BaseBuilder
        {
            static TKeyPredicate GetKeyPredicate();

            typedef ToMapBuilder<TKeyPredicate> this_type;
            typedef TKeyPredicate               key_predicate_type;

            key_predicate_type key_predicate;

            __forceinline explicit ToMapBuilder(key_predicate_type key_predicate) noexcept : key_predicate(std::move(key_predicate)) {}

            __forceinline ToMapBuilder(ToMapBuilder const& v) : key_predicate(v.key_predicate) {}

            __forceinline ToMapBuilder(ToMapBuilder&& v) noexcept : key_predicate(std::move(v.key_predicate)) {}

            template<typename TRange>
            KOKKOS_FUNCTION std::map<typename GetTransformedType<key_predicate_type, typename TRange::value_type>::type, typename TRange::value_type> Build(TRange range) const
            {
                typedef std::map<typename GetTransformedType<key_predicate_type, typename TRange::value_type>::type, typename TRange::value_type> result_type;

                result_type result;

                while(range.Next())
                {
                    auto v = range.Front();
                    auto k = key_predicate(v);

                    result.insert(typename result_type::value_type(std::move(k), std::move(v)));
                }

                return result;
            }
        };

        template<typename TKey, typename TValue>
        struct Lookup
        {
            typedef TKey   key_type;
            typedef TValue value_type;

            typedef std::vector<std::pair<key_type, size_type>> keys_type;
            typedef std::vector<value_type>                     values_type;

            typedef typename values_type::const_iterator values_iterator_type;

            template<typename TRange, typename TSelector>
            KOKKOS_FUNCTION Lookup(size_type capacity, TRange range, TSelector selector)
            {
                keys_type   k;
                values_type v;
                k.reserve(capacity);
                v.reserve(capacity);

                auto index = 0U;
                while(range.Next())
                {
                    auto value = range.Front();
                    auto key   = selector(value);
                    v.push_back(std::move(value));
                    k.push_back(typename keys_type::value_type(std::move(key), index));
                    ++index;
                }

                if(v.size() == 0)
                {
                    return;
                }

                std::sort(k.begin(), k.end(), [](typename keys_type::value_type const& l, typename keys_type::value_type const& r) { return l.first < r.first; });

                keys.reserve(k.size());
                values.reserve(v.size());

                auto iter = k.begin();
                auto end  = k.end();

                index = 0U;

                if(iter != end)
                {
                    values.push_back(std::move(v[iter->second]));
                    keys.push_back(typename keys_type::value_type(iter->first, index));
                }

                auto previous = iter;
                ++iter;
                ++index;

                while(iter != end)
                {
                    values.push_back(v[iter->second]);

                    if(previous->first < iter->first)
                    {
                        keys.push_back(typename keys_type::value_type(iter->first, index));
                    }

                    previous = iter;
                    ++iter;
                    ++index;
                }
            }

            __forceinline Lookup(Lookup const& v) : values(v.values), keys(v.keys) {}

            __forceinline Lookup(Lookup&& v) noexcept : values(std::move(v.values)), keys(std::move(v.keys)) {}

            __forceinline void swap(Lookup& v) noexcept
            {
                values.swap(v.values);
                keys.swap(v.keys);
            }

            __forceinline Lookup& operator=(Lookup const& v)
            {
                if(this == std::addressof(v))
                {
                    return *this;
                }

                Lookup tmp(v);

                swap(tmp);

                return *this;
            }

            __forceinline Lookup& operator=(Lookup&& v) noexcept
            {
                if(this == std::addressof(v))
                {
                    return *this;
                }

                swap(v);

                return *this;
            }

            struct LookupRange : BaseRange
            {
                typedef LookupRange this_type;

                enum
                {
                    returns_reference = 1,
                };

                typedef TValue            value_type;
                typedef value_type const& return_type;

                enum state
                {
                    state_initial,
                    state_iterating,
                    state_end,
                };

                values_type const* values;
                size_type          iter;
                size_type          end;
                state              state;

                __forceinline LookupRange(values_type const* values, const size_type iter, const size_type end) noexcept :
                    values(values), iter(iter), end(end), state(state_initial)
                {
                    Assert(values);
                }

                __forceinline LookupRange(LookupRange const& v) noexcept : values(v.values), iter(v.iter), end(v.end), state(v.state) {}

                __forceinline LookupRange(LookupRange&& v) noexcept : values(std::move(v.values)), iter(std::move(v.iter)), end(std::move(v.end)), state(std::move(v.state)) {}

                template<typename TRangeBuilder>
                __forceinline typename GetBuiltupType<TRangeBuilder, this_type>::type operator>>(TRangeBuilder range_builder) const
                {
                    return range_builder.build(*this);
                }

                __forceinline return_type Front() const noexcept
                {
                    Assert(state == state_iterating);
                    Assert(iter < end);

                    return (*values)[iter];
                }

                __forceinline bool Next() noexcept
                {
                    switch(state)
                    {
                        case state_initial:
                        {
                            auto has_elements = iter < end;
                            state             = has_elements ? state_iterating : state_end;
                            return has_elements;
                        }
                        case state_iterating:
                        {
                            ++iter;

                            auto has_elements = iter < end;
                            state             = has_elements ? state_iterating : state_end;
                            return has_elements;
                        }
                        case state_end:
                        default: return false;
                    }
                }
            };

            KOKKOS_FUNCTION LookupRange operator[](key_type const& key) const noexcept
            {
                if(values.empty())
                {
                    return LookupRange(std::addressof(values), 0U, 0U);
                }

                auto find = std::lower_bound(keys.begin(),
                                             keys.end(),
                                             typename keys_type::value_type(key, 0U),
                                             [](typename keys_type::value_type const& l, typename keys_type::value_type const& r) { return l.first < r.first; });

                if(find == keys.end())
                {
                    return LookupRange(std::addressof(values), 0U, 0U);
                }

                auto next = find + 1;
                if(next == keys.end())
                {
                    return LookupRange(std::addressof(values), find->second, values.size());
                }

                return LookupRange(std::addressof(values), find->second, next->second);
            }

            __forceinline size_type size_of_keys() const noexcept { return keys.size(); }

            __forceinline size_type size_of_values() const noexcept { return values.size(); }

            __forceinline FromRange<values_iterator_type> range_of_values() const noexcept { return FromRange<values_iterator_type>(values.begin(), values.end()); }

        private:
            values_type values;
            keys_type   keys;
        };

        template<typename TKeyPredicate>
        struct ToLookupBuilder : BaseBuilder
        {
            static TKeyPredicate GetKeyPredicate();

            typedef ToLookupBuilder<TKeyPredicate> this_type;
            typedef TKeyPredicate                  key_predicate_type;

            key_predicate_type key_predicate;

            __forceinline explicit ToLookupBuilder(key_predicate_type key_predicate) noexcept : key_predicate(std::move(key_predicate)) {}

            __forceinline ToLookupBuilder(ToLookupBuilder const& v) : key_predicate(v.key_predicate) {}

            __forceinline ToLookupBuilder(ToLookupBuilder&& v) noexcept : key_predicate(std::move(v.key_predicate)) {}

            template<typename TRange>
            __forceinline Lookup<typename GetTransformedType<key_predicate_type, typename TRange::value_type>::type, typename TRange::value_type> Build(TRange range) const
            {
                typedef Lookup<typename GetTransformedType<key_predicate_type, typename TRange::value_type>::type, typename TRange::value_type> result_type;

                result_type result(16U, range, key_predicate);

                return result;
            }
        };

        template<typename TPredicate>
        struct ForEachBuilder : BaseBuilder
        {
            typedef ForEachBuilder<TPredicate> this_type;
            typedef TPredicate                 predicate_type;

            predicate_type predicate;

            __forceinline explicit ForEachBuilder(predicate_type predicate) noexcept : predicate(std::move(predicate)) {}

            __forceinline ForEachBuilder(ForEachBuilder const& v) noexcept : predicate(v.predicate) {}

            __forceinline ForEachBuilder(ForEachBuilder&& v) noexcept : predicate(std::move(v.predicate)) {}

            template<typename TRange>
            __forceinline void build(TRange range) const
            {
                while(range.Next())
                {
                    predicate(range.Front());
                }
            }
        };

        template<typename TPredicate>
        struct FirstPredicateBuilder : BaseBuilder
        {
            typedef FirstPredicateBuilder<TPredicate> this_type;
            typedef TPredicate                        predicate_type;

            predicate_type predicate;

            __forceinline FirstPredicateBuilder(predicate_type predicate) noexcept : predicate(std::move(predicate)) {}

            __forceinline FirstPredicateBuilder(FirstPredicateBuilder const& v) noexcept : predicate(v.predicate) {}

            __forceinline FirstPredicateBuilder(FirstPredicateBuilder&& v) noexcept : predicate(std::move(v.predicate)) {}

            template<typename TRange>
            __forceinline typename TRange::value_type build(TRange range)
            {
                while(range.Next())
                {
                    if(predicate(range.Front()))
                    {
                        return range.Front();
                    }
                }

                throw SequenceEmptyException();
            }
        };

        struct FirstBuilder : BaseBuilder
        {
            typedef FirstBuilder this_type;

            __forceinline FirstBuilder() noexcept {}

            __forceinline FirstBuilder(FirstBuilder const& v) noexcept {}

            __forceinline FirstBuilder(FirstBuilder&& v) noexcept {}

            template<typename TRange>
            __forceinline typename TRange::value_type build(TRange range)
            {
                if(range.Next())
                {
                    return range.Front();
                }

                throw SequenceEmptyException();
            }
        };

        template<typename TPredicate>
        struct FirstOrDefaultPredicateBuilder : BaseBuilder
        {
            typedef FirstOrDefaultPredicateBuilder<TPredicate> this_type;
            typedef TPredicate                                 predicate_type;

            predicate_type predicate;

            __forceinline FirstOrDefaultPredicateBuilder(predicate_type predicate) noexcept : predicate(std::move(predicate)) {}

            __forceinline FirstOrDefaultPredicateBuilder(FirstOrDefaultPredicateBuilder const& v) noexcept : predicate(v.predicate) {}

            __forceinline FirstOrDefaultPredicateBuilder(FirstOrDefaultPredicateBuilder&& v) noexcept : predicate(std::move(v.predicate)) {}

            template<typename TRange>
            __forceinline typename TRange::value_type build(TRange range) const
            {
                while(range.Next())
                {
                    if(predicate(range.Front()))
                    {
                        return range.Front();
                    }
                }

                return typename TRange::value_type();
            }
        };

        struct FirstOrDefaultBuilder : BaseBuilder
        {
            typedef FirstOrDefaultBuilder this_type;

            __forceinline FirstOrDefaultBuilder() noexcept {}

            __forceinline FirstOrDefaultBuilder(FirstOrDefaultBuilder const& v) noexcept {}

            __forceinline FirstOrDefaultBuilder(FirstOrDefaultBuilder&& v) noexcept {}

            template<typename TRange>
            __forceinline typename TRange::value_type build(TRange range) const
            {
                if(range.Next())
                {
                    return range.Front();
                }

                return typename TRange::value_type();
            }
        };

        template<typename TPredicate>
        struct LastOrDefaultPredicateBuilder : BaseBuilder
        {
            typedef LastOrDefaultPredicateBuilder<TPredicate> this_type;
            typedef TPredicate                                predicate_type;

            predicate_type predicate;

            __forceinline LastOrDefaultPredicateBuilder(predicate_type predicate) noexcept : predicate(std::move(predicate)) {}

            __forceinline LastOrDefaultPredicateBuilder(LastOrDefaultPredicateBuilder const& v) noexcept : predicate(v.predicate) {}

            __forceinline LastOrDefaultPredicateBuilder(LastOrDefaultPredicateBuilder&& v) noexcept : predicate(std::move(v.predicate)) {}

            template<typename TRange>
            __forceinline typename TRange::value_type build(TRange range) const
            {
                auto current = typename TRange::value_type();

                while(range.Next())
                {
                    if(predicate(range.Front()))
                    {
                        current = std::move(range.Front());
                    }
                }

                return current;
            }
        };

        struct LastOrDefaultBuilder : BaseBuilder
        {
            typedef LastOrDefaultBuilder this_type;

            __forceinline LastOrDefaultBuilder() noexcept {}

            __forceinline LastOrDefaultBuilder(LastOrDefaultBuilder const& v) noexcept {}

            __forceinline LastOrDefaultBuilder(LastOrDefaultBuilder&& v) noexcept {}

            template<typename TRange>
            __forceinline typename TRange::value_type build(TRange range) const
            {
                auto current = typename TRange::value_type();

                while(range.Next())
                {
                    current = std::move(range.Front());
                }

                return current;
            }
        };

        template<typename TPredicate>
        struct CountPredicateBuilder : BaseBuilder
        {
            typedef CountPredicateBuilder<TPredicate> this_type;
            typedef TPredicate                        predicate_type;

            predicate_type predicate;

            __forceinline CountPredicateBuilder(predicate_type predicate) noexcept : predicate(std::move(predicate)) {}

            __forceinline CountPredicateBuilder(CountPredicateBuilder const& v) noexcept : predicate(v.predicate) {}

            __forceinline CountPredicateBuilder(CountPredicateBuilder&& v) noexcept : predicate(std::move(v.predicate)) {}

            template<typename TRange>
            __forceinline size_type build(TRange range) const
            {
                size_type count = 0U;
                while(range.Next())
                {
                    if(predicate(range.Front()))
                    {
                        ++count;
                    }
                }
                return count;
            }
        };

        struct CountBuilder : BaseBuilder
        {
            typedef CountBuilder this_type;

            __forceinline CountBuilder() noexcept {}

            __forceinline CountBuilder(CountBuilder const& v) noexcept {}

            __forceinline CountBuilder(CountBuilder&& v) noexcept {}

            template<typename TRange>
            __forceinline size_type build(TRange range) const
            {
                size_type count = 0U;
                while(range.Next())
                {
                    ++count;
                }
                return count;
            }
        };

        template<typename TSelector>
        struct SumSelectorBuilder : BaseBuilder
        {
            typedef SumSelectorBuilder<TSelector> this_type;
            typedef TSelector                     selector_type;

            selector_type selector;

            __forceinline SumSelectorBuilder(selector_type selector) noexcept : selector(std::move(selector)) {}

            __forceinline SumSelectorBuilder(SumSelectorBuilder const& v) noexcept : selector(v.selector) {}

            __forceinline SumSelectorBuilder(SumSelectorBuilder&& v) noexcept : selector(std::move(v.selector)) {}

            template<typename TRange>
            __forceinline typename GetTransformedType<selector_type, typename TRange::value_type>::type build(TRange range) const
            {
                typedef typename GetTransformedType<selector_type, typename TRange::value_type>::type value_type;

                auto sum = value_type();
                while(range.Next())
                {
                    sum += selector(range.Front());
                }
                return sum;
            }
        };

        struct SumBuilder : BaseBuilder
        {
            typedef SumBuilder this_type;

            __forceinline SumBuilder() noexcept {}

            __forceinline SumBuilder(SumBuilder const& v) noexcept {}

            __forceinline SumBuilder(SumBuilder&& v) noexcept {}

            template<typename TRange>
            __forceinline typename TRange::value_type build(TRange range) const
            {
                auto sum = typename TRange::value_type();
                while(range.Next())
                {
                    sum += range.Front();
                }
                return sum;
            }
        };

        template<typename TSelector>
        struct MaxSelectorBuilder : BaseBuilder
        {
            typedef MaxSelectorBuilder<TSelector> this_type;
            typedef TSelector                     selector_type;

            selector_type selector;

            __forceinline MaxSelectorBuilder(selector_type selector) noexcept : selector(std::move(selector)) {}

            __forceinline MaxSelectorBuilder(MaxSelectorBuilder const& v) noexcept : selector(v.selector) {}

            __forceinline MaxSelectorBuilder(MaxSelectorBuilder&& v) noexcept : selector(std::move(v.selector)) {}

            template<typename TRange>
            __forceinline typename GetTransformedType<selector_type, typename TRange::value_type>::type build(TRange range) const
            {
                typedef typename GetTransformedType<selector_type, typename TRange::value_type>::type value_type;

                auto current = std::numeric_limits<value_type>::lowest();
                while(range.Next())
                {
                    auto v = selector(range.Front());
                    if(current < v)
                    {
                        current = std::move(v);
                    }
                }

                return current;
            }
        };

        struct MaxBuilder : BaseBuilder
        {
            typedef MaxBuilder this_type;

            __forceinline MaxBuilder() noexcept {}

            __forceinline MaxBuilder(MaxBuilder const& v) noexcept {}

            __forceinline MaxBuilder(MaxBuilder&& v) noexcept {}

            template<typename TRange>
            __forceinline typename TRange::value_type build(TRange range) const
            {
                auto current = std::numeric_limits<typename TRange::value_type>::lowest();
                while(range.Next())
                {
                    auto v = range.Front();
                    if(current < v)
                    {
                        current = std::move(v);
                    }
                }

                return current;
            }
        };

        template<typename TSelector>
        struct MinSelectorBuilder : BaseBuilder
        {
            typedef MinSelectorBuilder<TSelector> this_type;
            typedef TSelector                     selector_type;

            selector_type selector;

            __forceinline MinSelectorBuilder(selector_type selector) noexcept : selector(std::move(selector)) {}

            __forceinline MinSelectorBuilder(MinSelectorBuilder const& v) noexcept : selector(v.selector) {}

            __forceinline MinSelectorBuilder(MinSelectorBuilder&& v) noexcept : selector(std::move(v.selector)) {}

            template<typename TRange>
            __forceinline typename GetTransformedType<selector_type, typename TRange::value_type>::type build(TRange range) const
            {
                typedef typename GetTransformedType<selector_type, typename TRange::value_type>::type value_type;

                auto current = std::numeric_limits<value_type>::max();
                while(range.Next())
                {
                    auto v = selector(range.Front());
                    if(v < current)
                    {
                        current = std::move(v);
                    }
                }

                return current;
            }
        };

        struct MinBuilder : BaseBuilder
        {
            typedef MinBuilder this_type;

            __forceinline MinBuilder() noexcept {}

            __forceinline MinBuilder(MinBuilder const& v) noexcept {}

            __forceinline MinBuilder(MinBuilder&& v) noexcept {}

            template<typename TRange>
            __forceinline typename TRange::value_type build(TRange range) const
            {
                auto current = std::numeric_limits<typename TRange::value_type>::max();
                while(range.Next())
                {
                    auto v = range.Front();
                    if(v < current)
                    {
                        current = std::move(v);
                    }
                }

                return current;
            }
        };

        template<typename TSelector>
        struct AvgSelectorBuilder : BaseBuilder
        {
            typedef AvgSelectorBuilder<TSelector> this_type;
            typedef TSelector                     selector_type;

            selector_type selector;

            __forceinline AvgSelectorBuilder(selector_type selector) noexcept : selector(std::move(selector)) {}

            __forceinline AvgSelectorBuilder(AvgSelectorBuilder const& v) noexcept : selector(v.selector) {}

            __forceinline AvgSelectorBuilder(AvgSelectorBuilder&& v) noexcept : selector(std::move(v.selector)) {}

            template<typename TRange>
            __forceinline typename GetTransformedType<selector_type, typename TRange::value_type>::type build(TRange range) const
            {
                typedef typename GetTransformedType<selector_type, typename TRange::value_type>::type value_type;

                auto sum   = value_type();
                int  count = 0;
                while(range.Next())
                {
                    sum += selector(range.Front());
                    ++count;
                }

                if(count == 0)
                {
                    return sum;
                }

                return sum / count;
            }
        };

        struct AvgBuilder : BaseBuilder
        {
            typedef AvgBuilder this_type;

            __forceinline AvgBuilder() noexcept {}

            __forceinline AvgBuilder(AvgBuilder const& v) noexcept {}

            __forceinline AvgBuilder(AvgBuilder&& v) noexcept {}

            template<typename TRange>
            __forceinline typename TRange::value_type build(TRange range) const
            {
                auto sum   = typename TRange::value_type();
                int  count = 0;
                while(range.Next())
                {
                    sum += range.Front();
                    ++count;
                }

                if(count == 0)
                {
                    return sum;
                }

                return sum / count;
            }
        };

        template<typename TAccumulate, typename TAccumulator>
        struct AggregateBuilder : BaseBuilder
        {
            typedef AggregateBuilder<TAccumulate, TAccumulator> this_type;
            typedef TAccumulator                                accumulator_type;
            typedef TAccumulate                                 seed_type;

            seed_type        seed;
            accumulator_type accumulator;

            __forceinline AggregateBuilder(seed_type seed, accumulator_type accumulator) noexcept : seed(std::move(seed)), accumulator(std::move(accumulator)) {}

            __forceinline AggregateBuilder(AggregateBuilder const& v) noexcept : seed(v.seed), accumulator(v.accumulator) {}

            __forceinline AggregateBuilder(AggregateBuilder&& v) noexcept : seed(std::move(v.seed)), accumulator(std::move(v.accumulator)) {}

            template<typename TRange>
            __forceinline seed_type build(TRange range) const
            {
                auto sum = seed;
                while(range.Next())
                {
                    sum = accumulator(sum, range.Front());
                }
                return sum;
            }
        };

        template<typename TAccumulate, typename TAccumulator, typename TSelector>
        struct AggregateResultSelectorBuilder : BaseBuilder
        {
            typedef AggregateResultSelectorBuilder<TAccumulate, TAccumulator, TSelector> this_type;
            typedef TAccumulator                                                         accumulator_type;
            typedef TAccumulate                                                          seed_type;
            typedef TSelector                                                            result_selector_type;

            seed_type            seed;
            accumulator_type     accumulator;
            result_selector_type result_selector;

            __forceinline AggregateResultSelectorBuilder(seed_type seed, accumulator_type accumulator, result_selector_type result_selector) noexcept :
                seed(std::move(seed)), accumulator(std::move(accumulator)), result_selector(std::move(result_selector))
            {
            }

            __forceinline AggregateResultSelectorBuilder(AggregateResultSelectorBuilder const& v) noexcept :
                seed(v.seed), accumulator(v.accumulator), result_selector(v.result_selector)
            {
            }

            __forceinline AggregateResultSelectorBuilder(AggregateResultSelectorBuilder&& v) noexcept :
                seed(std::move(v.seed)), accumulator(std::move(v.accumulator)), result_selector(std::move(v.result_selector))
            {
            }

            template<typename TRange>
            __forceinline auto build(TRange range) const -> decltype(result_selector(seed))
            {
                auto sum = seed;
                while(range.Next())
                {
                    sum = accumulator(sum, range.Front());
                }

                return result_selector(sum);
            }
        };

        template<typename TOtherRange, typename TComparer>
        struct SequenceEqualPredicateBuilder : BaseBuilder
        {
            typedef SequenceEqualPredicateBuilder<TOtherRange, TComparer> this_type;
            typedef TOtherRange                                           other_range_type;
            typedef TComparer                                             comparer_type;

            other_range_type other_range;
            comparer_type    comparer;

            __forceinline SequenceEqualPredicateBuilder(TOtherRange other_range, comparer_type comparer) noexcept :
                other_range(std::move(other_range)), comparer(std::move(comparer))
            {
            }

            __forceinline SequenceEqualPredicateBuilder(SequenceEqualPredicateBuilder const& v) noexcept : other_range(v.other_range), comparer(v.comparer) {}

            __forceinline SequenceEqualPredicateBuilder(SequenceEqualPredicateBuilder&& v) noexcept : other_range(std::move(v.other_range)), comparer(std::move(v.comparer)) {}

            template<typename TRange>
            __forceinline bool Build(TRange range) const
            {
                auto copy = other_range;
                for(;;)
                {
                    const bool next1 = range.Next();
                    const bool next2 = copy.Next();

                    // sequences are not of same length
                    if(next1 != next2)
                    {
                        return false;
                    }

                    // both sequences are over, next1 = next2 = false
                    if(!next1)
                    {
                        return true;
                    }

                    if(!comparer(range.Front(), copy.Front()))
                    {
                        return false;
                    }
                }
            }
        };

        template<typename TOtherRange>
        struct SequenceEqualBuilder : BaseBuilder
        {
            typedef SequenceEqualBuilder<TOtherRange> this_type;
            typedef TOtherRange                       other_range_type;

            other_range_type other_range;

            __forceinline SequenceEqualBuilder(TOtherRange other_range) noexcept : other_range(std::move(other_range)) {}

            __forceinline SequenceEqualBuilder(SequenceEqualBuilder const& v) noexcept : other_range(v.other_range) {}

            __forceinline SequenceEqualBuilder(SequenceEqualBuilder&& v) noexcept : other_range(std::move(v.other_range)) {}

            template<typename TRange>
            __forceinline bool Build(TRange range) const
            {
                auto copy = other_range;
                for(;;)
                {
                    const bool next1 = range.Next();
                    const bool next2 = copy.Next();

                    // sequences are not of same length
                    if(next1 != next2)
                    {
                        return false;
                    }

                    // both sequences are over, next1 = next2 = false
                    if(!next1)
                    {
                        return true;
                    }

                    if(range.Front() != copy.Front())
                    {
                        return false;
                    }
                }
            }
        };

        template<typename TCharType>
        struct ConcatenateBuilder : BaseBuilder
        {
            typedef ConcatenateBuilder<TCharType> this_type;

            std::basic_string<TCharType> separator;
            size_type                    capacity;

            __forceinline ConcatenateBuilder(std::basic_string<TCharType> separator, const size_type capacity) noexcept : separator(std::move(separator)), capacity(capacity) {}

            __forceinline ConcatenateBuilder(ConcatenateBuilder const& v) noexcept : separator(v.separator), capacity(v.capacity) {}

            __forceinline ConcatenateBuilder(ConcatenateBuilder&& v) noexcept : separator(std::move(v.separator)), capacity(std::move(v.capacity)) {}

            template<typename TRange>
            __forceinline typename std::basic_string<TCharType> Build(TRange range) const
            {
                auto                   first = true;
                std::vector<TCharType> buffer;

                buffer.reserve(capacity);

                while(range.Next())
                {
                    if(first)
                    {
                        first = false;
                    }
                    else
                    {
                        buffer.insert(buffer.end(), separator.begin(), separator.end());
                    }

                    auto v = range.Front();

                    buffer.insert(buffer.end(), v.begin(), v.end());
                }

                return std::basic_string<TCharType>(buffer.begin(), buffer.end());
            }
        };

        template<typename TPredicate>
        struct AnyPredicateBuilder : BaseBuilder
        {
            typedef AnyPredicateBuilder<TPredicate> this_type;
            typedef TPredicate                      predicate_type;

            predicate_type predicate;

            __forceinline AnyPredicateBuilder(predicate_type predicate) noexcept : predicate(std::move(predicate)) {}

            __forceinline AnyPredicateBuilder(AnyPredicateBuilder const& v) noexcept : predicate(v.predicate) {}

            __forceinline AnyPredicateBuilder(AnyPredicateBuilder&& v) noexcept : predicate(std::move(v.predicate)) {}

            template<typename TRange>
            __forceinline bool build(TRange range) const
            {
                bool any = false;
                while(range.Next() && !any)
                {
                    any = predicate(range.Front());
                }
                return any;
            }
        };

        struct AnyBuilder : BaseBuilder
        {
            typedef AnyBuilder this_type;

            __forceinline AnyBuilder() noexcept {}

            __forceinline AnyBuilder(AnyBuilder const& v) noexcept {}

            __forceinline AnyBuilder(AnyBuilder&& v) noexcept {}

            template<typename TRange>
            __forceinline bool build(TRange range) const
            {
                return range.Next();
            }
        };

        template<typename TPredicate>
        struct AllPredicateBuilder : BaseBuilder
        {
            typedef AllPredicateBuilder<TPredicate> this_type;
            typedef TPredicate                      predicate_type;

            predicate_type predicate;

            __forceinline AllPredicateBuilder(predicate_type predicate) noexcept : predicate(std::move(predicate)) {}

            __forceinline AllPredicateBuilder(AllPredicateBuilder const& v) noexcept : predicate(v.predicate) {}

            __forceinline AllPredicateBuilder(AllPredicateBuilder&& v) noexcept : predicate(std::move(v.predicate)) {}

            template<typename TRange>
            __forceinline bool build(TRange range) const
            {
                while(range.Next())
                {
                    if(!predicate(range.Front()))
                    {
                        return false;
                    }
                }

                return true;
            }
        };

        template<typename TValue>
        struct ContainsBuilder : BaseBuilder
        {
            typedef ContainsBuilder<TValue> this_type;
            typedef TValue                  value_type;

            value_type value;

            __forceinline ContainsBuilder(value_type value) noexcept : value(std::move(value)) {}

            __forceinline ContainsBuilder(ContainsBuilder const& v) noexcept : value(v.value) {}

            __forceinline ContainsBuilder(ContainsBuilder&& v) noexcept : value(std::move(v.value)) {}

            template<typename TRange>
            __forceinline bool build(TRange range) const
            {
                while(range.Next())
                {
                    if(range.Front() == value)
                    {
                        return true;
                    }
                }

                return false;
            }
        };

        template<typename TValue, typename TPredicate>
        struct ContainsPredicateBuilder : BaseBuilder
        {
            typedef ContainsPredicateBuilder<TValue, TPredicate> this_type;
            typedef TValue                                       value_type;
            typedef TPredicate                                   predicate_type;

            value_type     value;
            predicate_type predicate;

            __forceinline ContainsPredicateBuilder(value_type value, predicate_type predicate) noexcept : value(std::move(value)), predicate(std::move(predicate)) {}

            __forceinline ContainsPredicateBuilder(ContainsPredicateBuilder const& v) noexcept : value(v.value), predicate(v.predicate) {}

            __forceinline ContainsPredicateBuilder(ContainsPredicateBuilder&& v) noexcept : value(std::move(v.value)), predicate(std::move(v.predicate)) {}

            template<typename TRange>
            __forceinline bool build(TRange range) const
            {
                while(range.Next())
                {
                    if(predicate(range.Front(), value))
                    {
                        return true;
                    }
                }

                return false;
            }
        };

        struct ElementAtOrDefaultBuilder : BaseBuilder
        {
            typedef ElementAtOrDefaultBuilder this_type;

            size_type index;

            __forceinline ElementAtOrDefaultBuilder(size_type index) noexcept : index(std::move(index)) {}

            __forceinline ElementAtOrDefaultBuilder(ElementAtOrDefaultBuilder const& v) noexcept : index(v.index) {}

            __forceinline ElementAtOrDefaultBuilder(ElementAtOrDefaultBuilder&& v) noexcept : index(std::move(v.index)) {}

            template<typename TRange>
            __forceinline typename TRange::value_type build(TRange range) const
            {
                size_type current = 0U;

                while(range.Next())
                {
                    if(current < index)
                    {
                        ++current;
                    }
                    else
                    {
                        return range.Front();
                    }
                }

                return typename TRange::value_type();
            }
        };

        template<typename TRange>
        struct PairwiseRange : BaseRange
        {
            typedef PairwiseRange<TRange> this_type;
            typedef TRange                range_type;

            typedef typename TRange::value_type           element_type;
            typedef std::pair<element_type, element_type> value_type;
            typedef value_type                            return_type;

            enum
            {
                returns_reference = 0,
            };

            range_type        range;
            Opt<element_type> previous;
            Opt<element_type> current;

            __forceinline PairwiseRange(range_type range) noexcept : range(std::move(range)) {}

            __forceinline PairwiseRange(PairwiseRange const& v) noexcept : range(v.range), previous(v.previous), current(v.current) {}

            __forceinline PairwiseRange(PairwiseRange&& v) noexcept : range(std::move(v.range)), previous(std::move(v.previous)), current(std::move(v.current)) {}

            template<typename TPairwiseBuilder>
            __forceinline typename GetBuiltupType<TPairwiseBuilder, this_type>::type operator>>(TPairwiseBuilder pairwise_builder) const
            {
                return pairwise_builder.build(*this);
            }

            __forceinline return_type Front() const
            {
                Assert(previous.has_value());
                Assert(current.has_value());
                return std::make_pair(previous.get(), current.get());
            }

            __forceinline bool Next()
            {
                if(!previous.has_value())
                {
                    if(range.Next())
                    {
                        current = range.Front();
                    }
                    else
                    {
                        return false;
                    }
                }

                previous.swap(current);

                if(range.Next())
                {
                    current = range.Front();
                    return true;
                }

                previous.clear();
                current.clear();

                return false;
            }
        };

        struct PairwiseBuilder : BaseBuilder
        {
            typedef PairwiseBuilder this_type;

            __forceinline PairwiseBuilder() noexcept {}

            __forceinline PairwiseBuilder(PairwiseBuilder const& v) noexcept {}

            __forceinline PairwiseBuilder(PairwiseBuilder&& v) noexcept {}

            template<typename TRange>
            __forceinline PairwiseRange<TRange> Build(TRange range) const
            {
                return PairwiseRange<TRange>(std::move(range));
            }
        };

        template<typename TRange, typename TOtherRange>
        struct ZipWithRange : BaseRange
        {
            typedef ZipWithRange<TRange, TOtherRange> this_type;
            typedef TRange                            range_type;
            typedef TOtherRange                       other_range_type;

            typedef typename CleanupType<typename TRange::value_type>::type      left_element_type;
            typedef typename CleanupType<typename TOtherRange::value_type>::type right_element_type;
            typedef std::pair<left_element_type, right_element_type>             value_type;
            typedef value_type                                                   return_type;
            enum
            {
                returns_reference = 0,
            };

            range_type       range;
            other_range_type other_range;

            __forceinline ZipWithRange(range_type range, other_range_type other_range) noexcept : range(std::move(range)), other_range(std::move(other_range)) {}

            __forceinline ZipWithRange(ZipWithRange const& v) noexcept : range(v.range), other_range(v.other_range) {}

            __forceinline ZipWithRange(ZipWithRange&& v) noexcept : range(std::move(v.range)), other_range(std::move(v.other_range)) {}

            template<typename TRangeBuilder>
            __forceinline typename GetBuiltupType<TRangeBuilder, this_type>::type operator>>(TRangeBuilder range_builder) const
            {
                return range_builder.build(*this);
            }

            __forceinline return_type Front() const { return std::make_pair(range.Front(), other_range.Front()); }

            __forceinline bool Next() { return range.Next() && other_range.Next(); }
        };

        template<typename TOtherRange>
        struct ZipWithBuilder : BaseBuilder
        {
            typedef ZipWithBuilder<TOtherRange> this_type;
            typedef TOtherRange                 other_range_type;

            other_range_type other_range;

            __forceinline ZipWithBuilder(TOtherRange other_range) noexcept : other_range(std::move(other_range)) {}

            __forceinline ZipWithBuilder(ZipWithBuilder const& v) noexcept : other_range(v.other_range) {}

            __forceinline ZipWithBuilder(ZipWithBuilder&& v) noexcept : other_range(std::move(v.other_range)) {}

            template<typename TRange>
            __forceinline ZipWithRange<TRange, TOtherRange> Build(TRange range) const
            {
                return ZipWithRange<TRange, TOtherRange>(std::move(range), std::move(other_range));
            }
        };

        template<typename TPredicate>
        struct GenerateRange : BaseRange
        {
            static TPredicate GetPredicate();

            typedef decltype(GetPredicate()())                     raw_opt_value_type;
            typedef typename CleanupType<raw_opt_value_type>::type opt_value_type;

            typedef decltype(*(GetPredicate()()))              raw_value_type;
            typedef typename CleanupType<raw_value_type>::type value_type;

            typedef GenerateRange<TPredicate> this_type;
            typedef TPredicate                predicate_type;
            typedef value_type const&         return_type;

            enum
            {
                returns_reference = 1,
            };

            TPredicate     predicate;
            opt_value_type current_value;

            __forceinline GenerateRange(TPredicate predicate) noexcept : predicate(std::move(predicate)) {}

            __forceinline GenerateRange(GenerateRange const& v) noexcept : predicate(v.predicate), current_value(v.current_value) {}

            __forceinline GenerateRange(GenerateRange&& v) noexcept : predicate(std::move(v.predicate)), current_value(std::move(v.current_value)) {}

            template<typename TRangeBuilder>
            __forceinline typename GetBuiltupType<TRangeBuilder, this_type>::type operator>>(TRangeBuilder range_builder) const
            {
                return range_builder.build(*this);
            }

            __forceinline return_type Front() const
            {
                Assert(current_value);
                return *current_value;
            }

            __forceinline bool Next() noexcept
            {
                current_value = predicate();
                return current_value;
            }
        };

    }

    // The interface of cpplinq

    // Range sources

    template<typename TValueIterator>
    __forceinline detail::FromRange<TValueIterator> FromIterators(TValueIterator begin, TValueIterator end) noexcept
    {
        return detail::FromRange<TValueIterator>(std::move(begin), std::move(end));
    }

    template<typename TContainer>
    __forceinline detail::FromRange<typename TContainer::const_iterator> From(TContainer const& container)
    {
        return detail::FromRange<typename TContainer::const_iterator>(container.begin(), container.end());
    }

    /**
     * \brief
     * \tparam TValueArray
     * \param a
     * \return
     */
    template<typename TValueArray>
    __forceinline detail::FromRange<typename detail::GetArrayProperties<TValueArray>::iterator_type> from_array(TValueArray& a) noexcept
    {
        typedef detail::GetArrayProperties<TValueArray>  array_properties;
        typedef typename array_properties::iterator_type iterator_type;

        iterator_type begin = a;
        iterator_type end   = begin + array_properties::size;

        return detail::FromRange<typename array_properties::iterator_type>(std::move(begin), std::move(end));
    }

    template<typename TContainer>
    __forceinline detail::FromCopyRange<typename detail::CleanupType<TContainer>::type> FromCopy(TContainer&& container)
    {
        typedef typename detail::CleanupType<TContainer>::type container_type;

        return detail::FromCopyRange<container_type>(std::forward<TContainer>(container));
    }

    template<typename TPredicate>
    __forceinline detail::GenerateRange<TPredicate> Generate(TPredicate predicate) noexcept
    {
        return detail::GenerateRange<TPredicate>(std::move(predicate));
    }

    // Restriction operators

    template<typename TPredicate>
    __forceinline detail::WhereBuilder<TPredicate> Where(TPredicate predicate) noexcept
    {
        return detail::WhereBuilder<TPredicate>(std::move(predicate));
    }

    // Projection operators

    __forceinline detail::RefBuilder Ref() noexcept { return detail::RefBuilder(); }

    template<typename TPredicate>
    __forceinline detail::SelectBuilder<TPredicate> Select(TPredicate predicate) noexcept
    {
        return detail::SelectBuilder<TPredicate>(std::move(predicate));
    }

    template<typename TPredicate>
    __forceinline detail::SelectManyBuilder<TPredicate> SelectMany(TPredicate predicate) noexcept
    {
        return detail::SelectManyBuilder<TPredicate>(std::move(predicate));
    }

    template<typename TOtherRange, typename TKeySelector, typename TOtherKeySelector, typename TCombiner>
    __forceinline detail::JoinBuilder<TOtherRange, TKeySelector, TOtherKeySelector, TCombiner> Join(TOtherRange       other_range,
                                                                                                    TKeySelector      key_selector,
                                                                                                    TOtherKeySelector other_key_selector,
                                                                                                    TCombiner         combiner) noexcept
    {
        return detail::JoinBuilder<TOtherRange, TKeySelector, TOtherKeySelector, TCombiner>(
            std::move(other_range), std::move(key_selector), std::move(other_key_selector), std::move(combiner));
    }

    // Concatenation operators

    template<typename TOtherRange>
    __forceinline detail::ConcatBuilder<TOtherRange> Concat(TOtherRange other_range) noexcept
    {
        return detail::ConcatBuilder<TOtherRange>(std::move(other_range));
    }

    // Partitioning operators

    template<typename TPredicate>
    __forceinline detail::TakeWhileBuilder<TPredicate> TakeWhile(TPredicate predicate) noexcept
    {
        return detail::TakeWhileBuilder<TPredicate>(std::move(predicate));
    }

    __forceinline detail::TakeBuilder Take(const size_type count) noexcept { return detail::TakeBuilder(count); }

    template<typename TPredicate>
    __forceinline detail::SkipWhileBuilder<TPredicate> SkipWhile(TPredicate predicate) noexcept
    {
        return detail::SkipWhileBuilder<TPredicate>(predicate);
    }

    __forceinline detail::SkipBuilder Skip(const size_type count) noexcept { return detail::SkipBuilder(count); }

    // Ordering operators

    template<typename TPredicate>
    __forceinline detail::OrderbyBuilder<TPredicate> Orderby(TPredicate predicate, bool sort_ascending = true) noexcept
    {
        return detail::OrderbyBuilder<TPredicate>(std::move(predicate), sort_ascending);
    }

    template<typename TPredicate>
    __forceinline detail::OrderbyBuilder<TPredicate> OrderbyAscending(TPredicate predicate) noexcept
    {
        return detail::OrderbyBuilder<TPredicate>(std::move(predicate), true);
    }

    template<typename TPredicate>
    __forceinline detail::OrderbyBuilder<TPredicate> OrderbyDescending(TPredicate predicate) noexcept
    {
        return detail::OrderbyBuilder<TPredicate>(std::move(predicate), false);
    }

    template<typename TPredicate>
    __forceinline detail::ThenbyBuilder<TPredicate> Thenby(TPredicate predicate, bool sort_ascending = true) noexcept
    {
        return detail::ThenbyBuilder<TPredicate>(std::move(predicate), sort_ascending);
    }

    template<typename TPredicate>
    __forceinline detail::ThenbyBuilder<TPredicate> ThenbyAscending(TPredicate predicate) noexcept
    {
        return detail::ThenbyBuilder<TPredicate>(std::move(predicate), true);
    }

    template<typename TPredicate>
    __forceinline detail::ThenbyBuilder<TPredicate> ThenbyDescending(TPredicate predicate) noexcept
    {
        return detail::ThenbyBuilder<TPredicate>(std::move(predicate), false);
    }

    __forceinline detail::ReverseBuilder Reverse(const size_type capacity = 16U) noexcept { return detail::ReverseBuilder(capacity); }

    // Conversion operators

    namespace experimental
    {
        __forceinline detail::experimental::ContainerBuilder Container() noexcept { return detail::experimental::ContainerBuilder(); }
    }

    template<typename TValue>
    __forceinline detail::Opt<typename detail::CleanupType<TValue>::type> ToOpt(TValue&& v)
    {
        return detail::Opt<typename detail::CleanupType<TValue>::type>(std::forward<TValue>(v));
    }

    template<typename TValue>
    __forceinline detail::Opt<TValue> ToOpt()
    {
        return detail::Opt<TValue>();
    }

    __forceinline detail::ToVectorBuilder ToVector(const size_type capacity = 16U) noexcept { return detail::ToVectorBuilder(capacity); }

    __forceinline detail::ToListBuilder ToList() noexcept { return detail::ToListBuilder(); }

    template<typename TKeyPredicate>
    __forceinline detail::ToMapBuilder<TKeyPredicate> ToMap(TKeyPredicate key_predicate) noexcept
    {
        return detail::ToMapBuilder<TKeyPredicate>(std::move(key_predicate));
    }

    template<typename TKeyPredicate>
    __forceinline detail::ToLookupBuilder<TKeyPredicate> ToLookup(TKeyPredicate key_predicate) noexcept
    {
        return detail::ToLookupBuilder<TKeyPredicate>(std::move(key_predicate));
    }

    // Equality operators
    template<typename TOtherRange>
    __forceinline detail::SequenceEqualBuilder<TOtherRange> SequenceEqual(TOtherRange other_range) noexcept
    {
        return detail::SequenceEqualBuilder<TOtherRange>(std::move(other_range));
    }

    template<typename TOtherRange, typename TComparer>
    __forceinline detail::SequenceEqualPredicateBuilder<TOtherRange, TComparer> SequenceEqual(TOtherRange other_range, TComparer comparer) noexcept
    {
        return detail::SequenceEqualPredicateBuilder<TOtherRange, TComparer>(std::move(other_range), std::move(comparer));
    }

    // Element operators

    template<typename TPredicate>
    __forceinline detail::FirstPredicateBuilder<TPredicate> First(TPredicate predicate)
    {
        return detail::FirstPredicateBuilder<TPredicate>(std::move(predicate));
    }

    __forceinline detail::FirstBuilder First() { return detail::FirstBuilder(); }

    template<typename TPredicate>
    __forceinline detail::FirstOrDefaultPredicateBuilder<TPredicate> FirstOrDefault(TPredicate predicate) noexcept
    {
        return detail::FirstOrDefaultPredicateBuilder<TPredicate>(predicate);
    }

    __forceinline detail::FirstOrDefaultBuilder FirstOrDefault() noexcept { return detail::FirstOrDefaultBuilder(); }

    template<typename TPredicate>
    __forceinline detail::LastOrDefaultPredicateBuilder<TPredicate> LastOrDefault(TPredicate predicate) noexcept
    {
        return detail::LastOrDefaultPredicateBuilder<TPredicate>(predicate);
    }

    __forceinline detail::LastOrDefaultBuilder LastOrDefault() noexcept { return detail::LastOrDefaultBuilder(); }

    __forceinline detail::ElementAtOrDefaultBuilder ElementAtOrDefault(const size_type index) noexcept { return detail::ElementAtOrDefaultBuilder(index); }

    // Generation operators

    __forceinline detail::IntRange Range(const int start, const int count) noexcept
    {
        const auto c   = count > 0 ? count : 0;
        const auto end = (INT_MAX - c) > start ? (start + c) : INT_MAX;
        return detail::IntRange(start, end);
    }

    template<typename TValue>
    __forceinline detail::RepeatRange<TValue> Repeat(TValue element, const int count) noexcept
    {
        auto c = count > 0 ? count : 0;
        return detail::RepeatRange<TValue>(element, c);
    }

    template<typename TValue>
    __forceinline detail::EmptyRange<TValue> Empty() noexcept
    {
        return detail::EmptyRange<TValue>();
    }

    template<typename TValue>
    __forceinline detail::SingletonRange<typename detail::CleanupType<TValue>::type> Singleton(TValue&& value) noexcept
    {
        return detail::SingletonRange<typename detail::CleanupType<TValue>::type>(std::forward<TValue>(value));
    }

    // Quantifiers

    template<typename TPredicate>
    __forceinline detail::AnyPredicateBuilder<TPredicate> Any(TPredicate predicate) noexcept
    {
        return detail::AnyPredicateBuilder<TPredicate>(std::move(predicate));
    }

    __forceinline detail::AnyBuilder Any() noexcept { return detail::AnyBuilder(); }

    template<typename TPredicate>
    __forceinline detail::AllPredicateBuilder<TPredicate> All(TPredicate predicate) noexcept
    {
        return detail::AllPredicateBuilder<TPredicate>(std::move(predicate));
    }

    template<typename TValue>
    __forceinline detail::ContainsBuilder<TValue> Contains(TValue value) noexcept
    {
        return detail::ContainsBuilder<TValue>(value);
    }

    template<typename TValue, typename TPredicate>
    __forceinline detail::ContainsPredicateBuilder<TValue, TPredicate> Contains(TValue value, TPredicate predicate) noexcept
    {
        return detail::ContainsPredicateBuilder<TValue, TPredicate>(value, predicate);
    }

    // Aggregate operators

    template<typename TPredicate>
    __forceinline detail::CountPredicateBuilder<TPredicate> Count(TPredicate predicate) noexcept
    {
        return detail::CountPredicateBuilder<TPredicate>(std::move(predicate));
    }

    __forceinline detail::CountBuilder Count() noexcept { return detail::CountBuilder(); }

    template<typename TSelector>
    __forceinline detail::SumSelectorBuilder<TSelector> Sum(TSelector selector) noexcept
    {
        return detail::SumSelectorBuilder<TSelector>(std::move(selector));
    }

    __forceinline detail::SumBuilder Sum() noexcept { return detail::SumBuilder(); }

    template<typename TSelector>
    __forceinline detail::MaxSelectorBuilder<TSelector> Max(TSelector selector) noexcept
    {
        return detail::MaxSelectorBuilder<TSelector>(std::move(selector));
    }

    __forceinline detail::MaxBuilder Max() noexcept { return detail::MaxBuilder(); }

    template<typename TSelector>
    __forceinline detail::MinSelectorBuilder<TSelector> Min(TSelector selector) noexcept
    {
        return detail::MinSelectorBuilder<TSelector>(std::move(selector));
    }

    __forceinline detail::MinBuilder Min() noexcept { return detail::MinBuilder(); }

    template<typename TSelector>
    __forceinline detail::AvgSelectorBuilder<TSelector> Average(TSelector selector) noexcept
    {
        return detail::AvgSelectorBuilder<TSelector>(std::move(selector));
    }

    __forceinline detail::AvgBuilder Average() noexcept { return detail::AvgBuilder(); }

    template<typename TAccumulate, typename TAccumulator>
    __forceinline detail::AggregateBuilder<TAccumulate, TAccumulator> Aggregate(TAccumulate seed, TAccumulator accumulator) noexcept
    {
        return detail::AggregateBuilder<TAccumulate, TAccumulator>(seed, accumulator);
    }

    template<typename TAccumulate, typename TAccumulator, typename TSelector>
    __forceinline detail::AggregateResultSelectorBuilder<TAccumulate, TAccumulator, TSelector> Aggregate(TAccumulate seed, TAccumulator accumulator, TSelector result_selector) noexcept
    {
        return detail::AggregateResultSelectorBuilder<TAccumulate, TAccumulator, TSelector>(seed, accumulator, result_selector);
    }

    // set operators
    __forceinline detail::DistinctBuilder Distinct() noexcept { return detail::DistinctBuilder(); }

    template<typename TOtherRange>
    __forceinline detail::UnionBuilder<TOtherRange> UnionWith(TOtherRange other_range) noexcept
    {
        return detail::UnionBuilder<TOtherRange>(std::move(other_range));
    }

    template<typename TOtherRange>
    __forceinline detail::IntersectBuilder<TOtherRange> IntersectWith(TOtherRange other_range) noexcept
    {
        return detail::IntersectBuilder<TOtherRange>(std::move(other_range));
    }

    template<typename TOtherRange>
    __forceinline detail::ExceptBuilder<TOtherRange> Except(TOtherRange other_range) noexcept
    {
        return detail::ExceptBuilder<TOtherRange>(std::move(other_range));
    }

    // other operators

    template<typename TPredicate>
    __forceinline detail::ForEachBuilder<TPredicate> ForEach(TPredicate predicate) noexcept
    {
        return detail::ForEachBuilder<TPredicate>(std::move(predicate));
    }

    __forceinline detail::ConcatenateBuilder<char> Concatenate(std::string separator, const size_type capacity = 16U) noexcept
    {
        return detail::ConcatenateBuilder<char>(std::move(separator), capacity);
    }

    __forceinline detail::ConcatenateBuilder<wchar_t> Concatenate(std::wstring separator, const size_type capacity = 16U) noexcept
    {
        return detail::ConcatenateBuilder<wchar_t>(std::move(separator), capacity);
    }

    __forceinline detail::PairwiseBuilder Pairwise() noexcept { return detail::PairwiseBuilder(); }

    template<typename TOtherRange>
    __forceinline detail::ZipWithBuilder<TOtherRange> ZipWith(TOtherRange other_range) noexcept
    {
        return detail::ZipWithBuilder<TOtherRange>(std::move(other_range));
    }

}
