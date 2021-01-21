#pragma once

#include "runtime.Kokkos/ViewTypes.hpp"
#include "runtime.Kokkos/Extensions.hpp"

#include <MathExtensions.hpp>

#include <StdExtensions.hpp>

#include <Print.hpp>

#include "Analyzes/Variant.hpp"

#include <unordered_map>

namespace Spatial
{
    struct empty
    {
    }; //  this Geometry type represents the empty point set, ∅, for the coordinate space (OGC Simple Features).

    constexpr bool operator==(empty, empty) { return true; }
    constexpr bool operator!=(empty, empty) { return false; }
    constexpr bool operator<(empty, empty) { return false; }
    constexpr bool operator>(empty, empty) { return false; }
    constexpr bool operator<=(empty, empty) { return true; }
    constexpr bool operator>=(empty, empty) { return true; }

    template<typename T>
    struct point
    {
        using coordinate_type = T;

        constexpr point() : x(), y() {}
        constexpr point(T x_, T y_) : x(x_), y(y_) {}

        T x;
        T y;
    };

    template<typename T>
    constexpr bool operator==(const point<T> & lhs, const point<T> & rhs)
    {
        return lhs.x == rhs.x && lhs.y == rhs.y;
    }

    template<typename T>
    constexpr bool operator!=(const point<T> & lhs, const point<T> & rhs)
    {
        return !(lhs == rhs);
    }

    template<typename T>
    struct box
    {
        using coordinate_type = T;
        using point_type      = point<coordinate_type>;

        constexpr box(const point_type & min_, const point_type & max_) : min(min_), max(max_) {}

        point_type min;
        point_type max;
    };

    template<typename T>
    constexpr bool operator==(const box<T> & lhs, const box<T> & rhs)
    {
        return lhs.min == rhs.min && lhs.max == rhs.max;
    }

    template<typename T>
    constexpr bool operator!=(const box<T> & lhs, const box<T> & rhs)
    {
        return lhs.min != rhs.min || lhs.max != rhs.max;
    }

    template<typename G, typename T = typename G::coordinate_type>
    box<T> envelope(const G & geometry)
    {
        using limits = std::numeric_limits<T>;

        T min_t = limits::has_infinity ? -limits::infinity() : limits::min();
        T max_t = limits::has_infinity ? limits::infinity() : limits::max();

        point<T> min(max_t, max_t);
        point<T> max(min_t, min_t);

        for_each_point(geometry, [&](const point<T> & point) {
            if(min.x > point.x)
                min.x = point.x;
            if(min.y > point.y)
                min.y = point.y;
            if(max.x < point.x)
                max.x = point.x;
            if(max.y < point.y)
                max.y = point.y;
        });

        return box<T>(min, max);
    }

    template<typename T, template<typename...> class Cont = std::vector>
    struct multi_point : Cont<point<T>>
    {
        using coordinate_type = T;
        using point_type      = point<T>;
        using container_type  = Cont<point_type>;
        using size_type       = typename container_type::size_type;

        template<class... Args>
        multi_point(Args&&... args) : container_type(std::forward<Args>(args)...)
        {
        }

        multi_point(std::initializer_list<point_type> args) : container_type(std::move(args)) {}
    };

    template<typename T, template<typename...> class Cont = std::vector>
    struct line_string : Cont<point<T>>
    {
        using coordinate_type = T;
        using point_type      = point<T>;
        using container_type  = Cont<point_type>;
        using size_type       = typename container_type::size_type;

        template<class... Args>
        line_string(Args&&... args) : container_type(std::forward<Args>(args)...)
        {
        }
        line_string(std::initializer_list<point_type> args) : container_type(std::move(args)) {}
    };

    template<typename T, template<typename...> class Cont = std::vector>
    struct multi_line_string : Cont<line_string<T>>
    {
        using coordinate_type  = T;
        using line_string_type = line_string<T>;
        using container_type   = Cont<line_string_type>;
        using size_type        = typename container_type::size_type;

        template<class... Args>
        multi_line_string(Args&&... args) : container_type(std::forward<Args>(args)...)
        {
        }
        multi_line_string(std::initializer_list<line_string_type> args) : container_type(std::move(args)) {}
    };

    template<typename T, template<typename...> class Cont = std::vector>
    struct linear_ring : Cont<point<T>>
    {
        using coordinate_type = T;
        using point_type      = point<T>;
        using container_type  = Cont<point_type>;
        using size_type       = typename container_type::size_type;

        template<class... Args>
        linear_ring(Args&&... args) : container_type(std::forward<Args>(args)...)
        {
        }
        linear_ring(std::initializer_list<point_type> args) : container_type(std::move(args)) {}
    };

    template<typename T, template<typename...> class Cont = std::vector>
    struct polygon : Cont<linear_ring<T>>
    {
        using coordinate_type  = T;
        using linear_ring_type = linear_ring<T>;
        using container_type   = Cont<linear_ring_type>;
        using size_type        = typename container_type::size_type;

        template<class... Args>
        polygon(Args&&... args) : container_type(std::forward<Args>(args)...)
        {
        }
        polygon(std::initializer_list<linear_ring_type> args) : container_type(std::move(args)) {}
    };

    template<typename T, template<typename...> class Cont = std::vector>
    struct multi_polygon : Cont<polygon<T>>
    {
        using coordinate_type = T;
        using polygon_type    = polygon<T>;
        using container_type  = Cont<polygon_type>;
        using size_type       = typename container_type::size_type;

        template<class... Args>
        multi_polygon(Args&&... args) : container_type(std::forward<Args>(args)...)
        {
        }
        multi_polygon(std::initializer_list<polygon_type> args) : container_type(std::move(args)) {}
    };

    template<typename T, template<typename...> class Cont = std::vector>
    struct geometry_collection;

    template<typename T, template<typename...> class Cont = std::vector>
    using geometry_base = System::variant<empty,
                                          point<T>,
                                          line_string<T, Cont>,
                                          polygon<T, Cont>,
                                          multi_point<T, Cont>,
                                          multi_line_string<T, Cont>,
                                          multi_polygon<T, Cont>,
                                          geometry_collection<T, Cont>>;

    template<typename T, template<typename...> class Cont = std::vector>
    struct geometry : geometry_base<T, Cont>
    {
        using coordinate_type = T;
        using geometry_base<T>::geometry_base;
    };

    template<typename T, template<typename...> class Cont>
    struct geometry_collection : Cont<geometry<T>>
    {
        using coordinate_type = T;
        using geometry_type   = geometry<T>;
        using container_type  = Cont<geometry_type>;
        using size_type       = typename container_type::size_type;

        template<class... Args>
        geometry_collection(Args&&... args) : container_type(std::forward<Args>(args)...)
        {
        }
        geometry_collection(std::initializer_list<geometry_type> args) : container_type(std::move(args)) {}
    };

    struct equal_comp_shared_ptr
    {
        template<typename T>
        bool operator()(const T & lhs, const T & rhs) const
        {
            return lhs == rhs;
        }

        template<typename T>
        bool operator()(const std::shared_ptr<T> & lhs, const std::shared_ptr<T> & rhs) const
        {
            if(lhs == rhs)
            {
                return true;
            }
            return *lhs == *rhs;
        }
    };

    struct null_value_t
    {
    };

    constexpr bool operator==(const null_value_t&, const null_value_t&) { return true; }
    constexpr bool operator!=(const null_value_t&, const null_value_t&) { return false; }
    constexpr bool operator<(const null_value_t&, const null_value_t&) { return false; }

    constexpr null_value_t null_value = null_value_t();

#define DECLARE_VALUE_TYPE_ACCESOR(NAME, TYPE)                                                         \
    TYPE* get##NAME() noexcept                                                                         \
    {                                                                                                  \
        return match([](TYPE& val) -> TYPE* { return &val; }, [](auto&) -> TYPE* { return nullptr; }); \
    }                                                                                                  \
    const TYPE* get##NAME() const noexcept { return const_cast<value*>(this)->get##NAME(); }

    struct value;

    using value_base = System::
        variant<null_value_t, bool, uint64_t, int64_t, double, std::string, std::shared_ptr<std::vector<value>>, std::shared_ptr<std::unordered_map<std::string, value>>>;

    struct value : public value_base
    {
        using array_type            = std::vector<value>;
        using array_ptr_type        = std::shared_ptr<std::vector<value>>;
        using const_array_ptr_type  = std::shared_ptr<const std::vector<value>>;
        using object_type           = std::unordered_map<std::string, value>;
        using object_ptr_type       = std::shared_ptr<std::unordered_map<std::string, value>>;
        using const_object_ptr_type = std::shared_ptr<const std::unordered_map<std::string, value>>;

        value() : value_base(null_value) {}
        value(null_value_t) : value_base(null_value) {}
        value(bool v) : value_base(v) {}
        value(const char* c) : value_base(std::string(c)) {}
        value(std::string str) : value_base(std::move(str)) {}

        template<typename T, std::enable_if_t<std::is_integral<T>::value, int> = 0, std::enable_if_t<std::is_signed<T>::value, int> = 0>
        value(T t) : value_base(int64_t(t))
        {
        }

        template<typename T, std::enable_if_t<std::is_integral<T>::value, int> = 0, std::enable_if_t<!std::is_signed<T>::value, int> = 0>
        value(T t) : value_base(uint64_t(t))
        {
        }

        template<typename T, std::enable_if_t<std::is_floating_point<T>::value, int> = 0>
        value(T t) : value_base(double(t))
        {
        }
        value(array_type array) : value_base(std::make_shared<array_type>(std::forward<array_type>(array))) {}
        value(array_ptr_type array) : value_base(array) {}
        value(object_type object) : value_base(std::make_shared<object_type>(std::forward<object_type>(object))) {}
        value(object_ptr_type object) : value_base(object) {}

        bool operator==(const value & rhs) const
        {
            assert(valid() && rhs.valid());
            if(this->which() != rhs.which())
            {
                return false;
            }
            System::detail::comparer<value, equal_comp_shared_ptr> visitor(*this);
            return visit(rhs, visitor);
        }

        explicit operator bool() const { return !is<null_value_t>(); }

        DECLARE_VALUE_TYPE_ACCESOR(Int, int64_t)
        DECLARE_VALUE_TYPE_ACCESOR(Uint, uint64_t)
        DECLARE_VALUE_TYPE_ACCESOR(Bool, bool)
        DECLARE_VALUE_TYPE_ACCESOR(Double, double)
        DECLARE_VALUE_TYPE_ACCESOR(String, std::string)

        array_ptr_type getArray() noexcept
        {
            return match([](array_ptr_type& val) -> array_ptr_type { return val; }, [](auto&) -> array_ptr_type { return nullptr; });
        }
        const_array_ptr_type getArray() const noexcept { return const_cast<value*>(this)->getArray(); }

        object_ptr_type getObject() noexcept
        {
            return match([](object_ptr_type& val) -> object_ptr_type { return val; }, [](auto&) -> object_ptr_type { return nullptr; });
        }
        const_object_ptr_type getObject() const noexcept { return const_cast<value*>(this)->getObject(); }
    };

#undef DECLARE_VALUE_TYPE_ACCESOR

    using property_map = value::object_type;

    using identifier = System::variant<null_value_t, uint64_t, int64_t, double, std::string>;

    template<class T>
    struct feature
    {
        using coordinate_type = T;
        using geometry_type   = geometry<T>; // Fully qualified to avoid GCC -fpermissive error.

        geometry_type geometry;
        property_map  properties;
        identifier    id;

        feature() : geometry(), properties(), id() {}
        feature(const geometry_type & geom_) : geometry(geom_), properties(), id() {}
        feature(geometry_type&& geom_) : geometry(std::move(geom_)), properties(), id() {}
        feature(const geometry_type & geom_, const property_map & prop_) : geometry(geom_), properties(prop_), id() {}
        feature(geometry_type&& geom_, property_map&& prop_) : geometry(std::move(geom_)), properties(std::move(prop_)), id() {}
        feature(const geometry_type & geom_, const property_map & prop_, const identifier & id_) : geometry(geom_), properties(prop_), id(id_) {}
        feature(geometry_type&& geom_, property_map&& prop_, identifier&& id_) : geometry(std::move(geom_)), properties(std::move(prop_)), id(std::move(id_)) {}
    };

    template<class T>
    constexpr bool operator==(const feature<T> & lhs, const feature<T> & rhs)
    {
        return lhs.id == rhs.id && lhs.geometry == rhs.geometry && lhs.properties == rhs.properties;
    }

    template<class T>
    constexpr bool operator!=(const feature<T> & lhs, const feature<T> & rhs)
    {
        return !(lhs == rhs);
    }

    template<class T, template<typename...> class Cont = std::vector>
    struct feature_collection : Cont<feature<T>>
    {
        using coordinate_type = T;
        using feature_type    = feature<T>;
        using container_type  = Cont<feature_type>;
        using size_type       = typename container_type::size_type;

        template<class... Args>
        feature_collection(Args&&... args) : container_type(std::forward<Args>(args)...)
        {
        }
        feature_collection(std::initializer_list<feature_type> args) : container_type(std::move(args)) {}
    };
}
