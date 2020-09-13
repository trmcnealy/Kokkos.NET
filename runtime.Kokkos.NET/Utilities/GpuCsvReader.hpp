#pragma once

#include <Types.hpp>
#include <StdExtensions.hpp>
#include <Constants.hpp>
#include <Print.hpp>

#include <Kokkos_Core.hpp>

namespace Kokkos
{
    template<>
    struct reduction_identity<int8>
    {
        KOKKOS_FORCEINLINE_FUNCTION constexpr static int8 sum() { return static_cast<int8>(0); }
        KOKKOS_FORCEINLINE_FUNCTION constexpr static int8 prod() { return static_cast<int8>(1); }
        KOKKOS_FORCEINLINE_FUNCTION constexpr static int8 max() { return static_cast<int8>(0); }
        KOKKOS_FORCEINLINE_FUNCTION constexpr static int8 min() { return UCHAR_MAX; }
        KOKKOS_FORCEINLINE_FUNCTION constexpr static int8 bor() { return static_cast<int8>(0x0); }
        KOKKOS_FORCEINLINE_FUNCTION constexpr static int8 band() { return ~static_cast<int8>(0x0); }
        KOKKOS_FORCEINLINE_FUNCTION constexpr static int8 lor() { return static_cast<int8>(0); }
        KOKKOS_FORCEINLINE_FUNCTION constexpr static int8 land() { return static_cast<int8>(1); }
    };
}

template<typename T>
class BasicString;

// char
// wchar_t

template<>
class BasicString<int8>
{ /*
     static constexpr int8 _nullChar = '\0';

     static constexpr BasicString<int8> Empty = "";*/

    Kokkos::View<int8*, Kokkos::Cuda> _chars;

    using Reference      = int8&;
    using ConstReference = const Reference;
    using Pointer        = int8*;
    using ConstPointer   = const Pointer;

public:
    BasicString(const std::string& value) : _chars("", value.size())
    {
        Kokkos::deep_copy(_chars, Kokkos::View<const int8*, Kokkos::HostSpace>(value.data(), value.size()));

        // Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::Cuda, int>(0, dim), [=] __host__ __device__(const int& i) { });
    }

    template<unsigned N>
    BasicString(int8 values[N]) : _chars("", N)
    {
        Kokkos::deep_copy(_chars, Kokkos::View<const int8*, Kokkos::HostSpace>(values, N));
    }

    BasicString(int8* values, const unsigned length) : _chars("", length) { Kokkos::deep_copy(_chars, Kokkos::View<const int8*, Kokkos::HostSpace>(values, length)); }

    BasicString(const int8* values, const unsigned length) : _chars("", length) { Kokkos::deep_copy(_chars, Kokkos::View<const int8*, Kokkos::HostSpace>(values, length)); }

    KOKKOS_INLINE_FUNCTION constexpr std::size_t size() const { return _chars.size(); }

#pragma region Indexers
    template<typename IntT>
    KOKKOS_FORCEINLINE_FUNCTION auto operator[](const IntT& i) noexcept -> std::enable_if_t<std::is_integral_v<IntT>, Reference>
    {
        return _chars[i];
    }

    template<typename IntT>
    KOKKOS_FORCEINLINE_FUNCTION auto operator[](const IntT& i) const noexcept -> std::enable_if_t<std::is_integral_v<IntT>, ConstReference>
    {
        return _chars[i];
    }

    template<typename IntT>
    KOKKOS_FORCEINLINE_FUNCTION auto operator()(const IntT& i) -> std::enable_if_t<std::is_integral_v<IntT>, Reference>
    {
        return _chars(i);
    }

    template<typename IntT>
    KOKKOS_FORCEINLINE_FUNCTION auto operator()(const IntT& i) const -> std::enable_if_t<std::is_integral_v<IntT>, ConstReference>
    {
        return _chars(i);
    }
#pragma endregion

#pragma region operator+

    __inline BasicString& operator+=(const std::string& value)
    {
        const unsigned new_length = _chars.size() + value.size();

        Kokkos::resize(_chars, new_length);

        auto rhs = subview(_chars, Kokkos::pair<unsigned, unsigned>(_chars.size() + 1, new_length));

        Kokkos::deep_copy(rhs, Kokkos::View<const int8*, Kokkos::HostSpace>(value.data(), value.size()));

        return *this;
    }

    __inline BasicString& operator+=(const BasicString& value)
    {
        const unsigned new_length = _chars.size() + value.size();

        Kokkos::resize(_chars, new_length);

        auto rhs = subview(_chars, Kokkos::pair<unsigned, unsigned>(_chars.size() + 1, new_length));

        Kokkos::deep_copy(rhs, value._chars);

        return *this;
    }

#pragma endregion

    template<typename Scalar, class ExecSpace>
    struct IndexOfFunctor
    {
        Kokkos::View<const Scalar*, ExecSpace> Values;

        Kokkos::View<Scalar, ExecSpace> Value;
        Kokkos::View<int, ExecSpace>    Index;

        IndexOfFunctor(const Kokkos::View<const Scalar*, ExecSpace>& values, const Scalar& input) : Values(values), Value("value"), Index("Index")
        {
            Value() = input;
            Index() = Constants<int>::Max();
        }

        KOKKOS_INLINE_FUNCTION void operator()(const int& i) const
        {
            if(Value() == Values(i))
            {
                Kokkos::atomic_fetch_min(&Index(), i);
            }
        }
    };

    __inline int IndexOf(const int8 value, const unsigned startIndex, const unsigned length) const
    {
        // TODO mod number of cuda threads

        const IndexOfFunctor<int8, Kokkos::Cuda> functor(_chars, value);

        const Kokkos::RangePolicy<Kokkos::Cuda, int> range(startIndex, startIndex + length);

        Kokkos::parallel_for(range, functor);

        Kokkos::fence();

        const int index = functor.Index();

        return index >= 0 && index < _chars.size() ? index : -1;
    }

    __inline int IndexOf(const int8 value) const { return IndexOf(value, 0, _chars.size()); }

    template<typename Scalar, class ExecSpace>
    struct IndexOfFunctor2
    {
        Kokkos::View<const Scalar*, ExecSpace> Values;

        Kokkos::View<Scalar*, ExecSpace> Value;
        Kokkos::View<int, ExecSpace>     Index;

        IndexOfFunctor2(const Kokkos::View<const Scalar*, ExecSpace>& values, const Kokkos::View<Scalar*, ExecSpace>& input) : Values(values), Value(input), Index("Index")
        {
            Index() = Constants<int>::Max();
        }

        KOKKOS_INLINE_FUNCTION void operator()(const int& i) const
        {
            bool found = true;

            for(int j = 0; j < Value.size(); ++j)
            {
                if(i + j >= Values.size())
                {
                    found = false;
                    break;
                }

                if(Value(j) != Values(i + j))
                {
                    found = false;
                    break;
                }
            }

            if(found)
            {
                Kokkos::atomic_fetch_min(&Index(), i);
            }
        }
    };

    __inline int IndexOf(const std::string& value, const unsigned startIndex, const unsigned length) const
    {
        const BasicString<int8> input(value.data(), value.size());

        const IndexOfFunctor2<int8, Kokkos::Cuda> functor(_chars, input._chars);

        const Kokkos::RangePolicy<Kokkos::Cuda, int> range(startIndex, startIndex + length);

        Kokkos::parallel_for(range, functor);

        Kokkos::fence();

        const int index = functor.Index();

        return index >= 0 && index < _chars.size() ? index : -1;
    }

    __inline int IndexOf(const std::string& value) const { return IndexOf(value, 0, _chars.size()); }

    template<typename Scalar, class ExecSpace>
    struct LastIndexOfFunctor
    {
        Kokkos::View<const Scalar*, ExecSpace> Values;

        Kokkos::View<Scalar, ExecSpace> Value;
        Kokkos::View<int, ExecSpace>    Index;

        LastIndexOfFunctor(const Kokkos::View<const Scalar*, ExecSpace>& values, const Scalar& input) : Values(values), Value("value"), Index("Index")
        {
            Value() = input;
            Index() = -1;
        }

        KOKKOS_INLINE_FUNCTION void operator()(const int& i) const
        {
            if(Value() == Values(i))
            {
                Kokkos::atomic_fetch_max(&Index(), i);
            }
        }
    };

    __inline int LastIndexOf(const int8 value, const unsigned startIndex, const unsigned length) const
    {
        const IndexOfFunctor<int8, Kokkos::Cuda> functor(_chars, value);

        const Kokkos::RangePolicy<Kokkos::Cuda, int> range(startIndex, startIndex + length);

        Kokkos::parallel_for(range, functor);

        Kokkos::fence();

        const int index = functor.Index();

        return index >= 0 && index < _chars.size() ? index : -1;
    }

    __inline int LastIndexOf(const int8 value) const { return LastIndexOf(value, 0, _chars.size()); }

    template<typename Scalar, class ExecSpace>
    struct LastIndexOfFunctor2
    {
        Kokkos::View<const Scalar*, ExecSpace> Values;

        Kokkos::View<Scalar*, ExecSpace> Value;
        Kokkos::View<int, ExecSpace>     Index;

        LastIndexOfFunctor2(const Kokkos::View<const Scalar*, ExecSpace>& values, const Kokkos::View<Scalar*, ExecSpace>& input) : Values(values), Value(input), Index("Index") { Index() = -1; }

        KOKKOS_INLINE_FUNCTION void operator()(const int& i) const
        {
            bool found = true;

            for(int j = 0; j < Value.size(); ++j)
            {
                if(i + j >= Values.size())
                {
                    found = false;
                    break;
                }

                if(Value(j) != Values(i + j))
                {
                    found = false;
                    break;
                }
            }

            if(found)
            {
                Kokkos::atomic_fetch_max(&Index(), i);
            }
        }
    };

    __inline int LastIndexOf(const std::string& value, const unsigned startIndex, const unsigned length) const
    {
        const BasicString<int8> input(value.data(), value.size());

        const IndexOfFunctor2<int8, Kokkos::Cuda> functor(_chars, input._chars);

        const Kokkos::RangePolicy<Kokkos::Cuda, int> range(startIndex, startIndex + length);

        Kokkos::parallel_for(range, functor);

        Kokkos::fence();

        const int index = functor.Index();

        return index >= 0 && index < _chars.size() ? index : -1;
    }

    __inline int LastIndexOf(const std::string& value) const { return IndexOf(value, 0, _chars.size()); }







    // Clone
    // Compare
    // CompareOrdinal
    // CompareTo
    // Concat
    // Contains
    // Copy
    // CopyTo
    // Create
    // EndsWith
    // EnumerateRunes
    // Equals
    // Format
    // GetEnumerator
    // GetHashCode
    // GetPinnableReference
    // GetTypeCode
    // IndexOfAny
    // Insert
    // Intern
    // IsInterned
    // IsNormalized
    // IsNullOrEmpty
    // IsNullOrWhiteSpace
    // Join
    // LastIndexOfAny
    // Normalize
    // PadLeft
    // PadRight
    // Remove
    // Replace
    // Split
    // StartsWith
    // Substring
    // ToCharArray
    // ToLower
    // ToLowerInvariant
    // ToString
    // ToUpper
    // ToUpperInvariant
    // Trim
    // TrimEnd
    // TrimStart
    //
    // static bool IsNullOrEmpty(std::wstring* value)
    //{
    //    // Using 0u >= (uint)value.Length rather than
    //    // value.Length == 0 as it will elide the bounds check to
    //    // the first char: value[0] if that is performed following the test
    //    // for the same test cost.
    //    // Ternary operator returning true/false prevents redundant asm generation:
    //    // https://github.com/dotnet/runtime/issues/4207
    //    return (value == nullptr || 0u >= static_cast<uint32>(value.Length)) ? true : false;
    //}
    //
    // static bool IsNullOrWhiteSpace(std::wstring* value)
    //{
    //    if(value == nullptr)
    //    {
    //        return true;
    //    }
    //
    //    for(int i = 0; i < value.Length; i++)
    //    {
    //        if(!std::isspace(value[i]))
    //        {
    //            return false;
    //        }
    //    }
    //
    //    return true;
    //}

    friend decltype(auto)                 operator+(const std::string& lhs, const BasicString<int8>& rhs);
    friend decltype(auto)                 operator+(const BasicString<int8>& lhs, const std::string& rhs);
    friend KOKKOS_FUNCTION decltype(auto) operator+(const BasicString<int8>& lhs, const BasicString<int8>& rhs);
};

__inline __attribute__((always_inline)) static decltype(auto) operator+(const std::string& lhs, const BasicString<int8>& rhs)
{
    const unsigned new_length = lhs.size() + rhs.size();

    BasicString<int8> newString(new int8[new_length], new_length);

    auto new_lhs = subview(newString._chars, Kokkos::pair<unsigned, unsigned>(0, lhs.size()));

    Kokkos::deep_copy(new_lhs, Kokkos::View<const int8*, Kokkos::HostSpace>(lhs.data(), lhs.size()));

    const unsigned rhs_index = lhs.size() + 1;

    Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::Cuda, int>(0, rhs.size()), [=] __host__ __device__(const int i) { new_lhs(rhs_index + i) = rhs(i); });

    return newString;
}

__inline __attribute__((always_inline)) static decltype(auto) operator+(const BasicString<int8>& lhs, const std::string& rhs)
{
    const unsigned new_length = lhs.size() + rhs.size();

    BasicString<int8> newString(new int8[new_length], new_length);

    Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::Cuda, int>(0, rhs.size()), [=] __host__ __device__(const int i) { newString(i) = lhs(i); });

    const unsigned rhs_index = lhs.size() + 1;

    auto new_lhs = subview(newString._chars, Kokkos::pair<unsigned, unsigned>(lhs.size() + 1, new_length));

    Kokkos::deep_copy(new_lhs, Kokkos::View<const int8*, Kokkos::HostSpace>(rhs.data(), rhs.size()));

    return newString;
}

KOKKOS_FORCEINLINE_FUNCTION static decltype(auto) operator+(const BasicString<int8>& lhs, const BasicString<int8>& rhs)
{
    const unsigned new_length = lhs.size() + rhs.size();

    BasicString<int8> newString(new int8[new_length], new_length);

    Kokkos::parallel_for(
        Kokkos::RangePolicy<Kokkos::Cuda, int>(0, rhs.size()), KOKKOS_LAMBDA(const int i) { newString(i) = rhs(i); });

    const unsigned rhs_index = lhs.size() + 1;

    Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::Cuda, int>(rhs.size(), new_length), [=] __host__ __device__(const int i) { newString(rhs_index + i) = rhs(i); });

    return newString;
}

using String = BasicString<int8>;

/*namespace Test
{
using ignore_column                              = unsigned int;
static const ignore_column ignore_no_column      = 0;
static const ignore_column ignore_extra_column   = 1;
static const ignore_column ignore_missing_column = 2;

template<char... trim_char_list>
struct trim_chars
{
private:
constexpr static bool is_trim_char(char) { return false; }

template<class... OtherTrimChars>
constexpr static bool is_trim_char(char c, const char trim_char, OtherTrimChars... other_trim_chars)
{
    return c == trim_char || is_trim_char(c, other_trim_chars...);
}

public:
static void trim(char*& str_begin, char*& str_end)
{
    while(str_begin != str_end && is_trim_char(*str_begin, trim_char_list...))
    {
        ++str_begin;
    }

    while(str_begin != str_end && is_trim_char(*(str_end - 1), trim_char_list...))
    {
        --str_end;
    }

    *str_end = '\0';
}
};

struct no_comment
{
static constexpr bool is_comment(const char*) { return false; }
};

template<char... comment_start_char_list>
struct single_line_comment
{
private:
constexpr static bool is_comment_start_char(char) { return false; }

template<class... OtherCommentStartChars>
constexpr static bool is_comment_start_char(char c, const char comment_start_char, OtherCommentStartChars... other_comment_start_chars)
{
    return c == comment_start_char || is_comment_start_char(c, other_comment_start_chars...);
}

public:
static bool is_comment(const char* line) { return is_comment_start_char(*line, comment_start_char_list...); }
};

struct empty_line_comment
{
static bool is_comment(const char* line)
{
    if(*line == '\0')
    {
        return true;
    }

    while(*line == ' ' || *line == '\t')
    {
        ++line;

        if(*line == 0)
        {
            return true;
        }
    }
    return false;
}
};

template<char... comment_start_char_list>
struct single_and_empty_line_comment
{
static bool is_comment(const char* line) { return single_line_comment<comment_start_char_list...>::is_comment(line) || empty_line_comment::is_comment(line); }
};

template<char sep>
struct no_quote_escape
{
static const char* find_next_column_end(const char* col_begin)
{
    while(*col_begin != sep && *col_begin != '\0')
    {
        ++col_begin;
    }

    return col_begin;
}

static void unescape(char*&, char*&) {}
};

template<char sep, char quote>
struct double_quote_escape
{
static const char* find_next_column_end(const char* col_begin)
{
    while(*col_begin != sep && *col_begin != '\0')
    {
        if(*col_begin != quote)
        {
            ++col_begin;
        }
        else
        {
            do
            {
                ++col_begin;

                while(*col_begin != quote)
                {
                    if(*col_begin == '\0')
                    {
                        return nullptr;
                    }

                    ++col_begin;
                }
                ++col_begin;
            } while(*col_begin == quote);
        }
    }
    return col_begin;
}

static void unescape(char*& col_begin, char*& col_end)
{
    if(col_end - col_begin >= 2)
    {
        if(*col_begin == quote && *(col_end - 1) == quote)
        {
            ++col_begin;
            --col_end;

            char* out = col_begin;

            for(char* in = col_begin; in != col_end; ++in)
            {
                if(*in == quote && (in + 1) != col_end && *(in + 1) == quote)
                {
                    ++in;
                }

                *out = *in;
                ++out;
            }

            col_end  = out;
            *col_end = '\0';
        }
    }
}
};

struct throw_on_overflow
{
template<class T>
static void on_overflow(T&)
{
    throw "integer_overflow";
}

template<class T>
static void on_underflow(T&)
{
    throw "integer_underflow";
}
};

struct ignore_overflow
{
template<class T>
static void on_overflow(T&)
{
}

template<class T>
static void on_underflow(T&)
{
}
};

struct set_to_max_on_overflow
{
template<class T>
static void on_overflow(T& x)
{
    x = std::numeric_limits<T>::max();
}

template<class T>
static void on_underflow(T& x)
{
    x = std::numeric_limits<T>::min();
}
};

class LineReader
{
static const int        block_len = 1 << 20;
std::unique_ptr<char[]> buffer; // must be constructed before (and thus destructed after) the reader!

#ifdef CSV_IO_NO_THREAD
detail::SynchronousReader reader;
#else
detail::AsynchronousReader reader;
#endif
int data_begin;
int data_end;

char     file_name[error::max_file_name_length + 1];
unsigned file_line;

static std::unique_ptr<ByteSourceBase> open_file(const char* file_name)
{
    // We open the file in binary mode as it makes no difference under *nix
    // and under Windows we handle \r\n newlines ourself.
    FILE* file = std::fopen(file_name, "rb");
    if(file == 0)
    {
        int                      x = errno; // store errno as soon as possible, doing it after constructor call can fail.
        error::can_not_open_file err;
        err.set_errno(x);
        err.set_file_name(file_name);
        throw err;
    }

    return std::unique_ptr<ByteSourceBase>(new detail::OwningStdIOByteSourceBase(file));
}

void init(std::unique_ptr<ByteSourceBase> byte_source)
{
    file_line = 0;

    buffer     = std::unique_ptr<char[]>(new char[3 * block_len]);
    data_begin = 0;
    data_end   = byte_source->read(buffer.get(), 2 * block_len);

    // Ignore UTF-8 BOM
    if(data_end >= 3 && buffer[0] == '\xEF' && buffer[1] == '\xBB' && buffer[2] == '\xBF')
    {
        data_begin = 3;
    }

    if(data_end == 2 * block_len)
    {
        reader.init(std::move(byte_source));
        reader.start_read(buffer.get() + 2 * block_len, block_len);
    }
}

public:
LineReader()                  = delete;
LineReader(const LineReader&) = delete;
LineReader& operator=(const LineReader&) = delete;

explicit LineReader(const char* file_name)
{
    set_file_name(file_name);
    init(open_file(file_name));
}

explicit LineReader(const std::string& file_name)
{
    set_file_name(file_name.c_str());
    init(open_file(file_name.c_str()));
}

LineReader(const char* file_name, std::unique_ptr<ByteSourceBase> byte_source)
{
    set_file_name(file_name);
    init(std::move(byte_source));
}

LineReader(const std::string& file_name, std::unique_ptr<ByteSourceBase> byte_source)
{
    set_file_name(file_name.c_str());
    init(std::move(byte_source));
}

LineReader(const char* file_name, const char* data_begin, const char* data_end)
{
    set_file_name(file_name);
    init(std::unique_ptr<ByteSourceBase>(new detail::NonOwningStringByteSource(data_begin, data_end - data_begin)));
}

LineReader(const std::string& file_name, const char* data_begin, const char* data_end)
{
    set_file_name(file_name.c_str());
    init(std::unique_ptr<ByteSourceBase>(new detail::NonOwningStringByteSource(data_begin, data_end - data_begin)));
}

LineReader(const char* file_name, FILE* file)
{
    set_file_name(file_name);
    init(std::unique_ptr<ByteSourceBase>(new detail::OwningStdIOByteSourceBase(file)));
}

LineReader(const std::string& file_name, FILE* file)
{
    set_file_name(file_name.c_str());
    init(std::unique_ptr<ByteSourceBase>(new detail::OwningStdIOByteSourceBase(file)));
}

LineReader(const char* file_name, std::istream& in)
{
    set_file_name(file_name);
    init(std::unique_ptr<ByteSourceBase>(new detail::NonOwningIStreamByteSource(in)));
}

LineReader(const std::string& file_name, std::istream& in)
{
    set_file_name(file_name.c_str());
    init(std::unique_ptr<ByteSourceBase>(new detail::NonOwningIStreamByteSource(in)));
}

void set_file_name(const std::string& file_name) { set_file_name(file_name.c_str()); }

void set_file_name(const char* file_name)
{
    if(file_name != nullptr)
    {
        strncpy(this->file_name, file_name, sizeof(this->file_name));
        this->file_name[sizeof(this->file_name) - 1] = '\0';
    }
    else
    {
        this->file_name[0] = '\0';
    }
}

const char* get_truncated_file_name() const { return file_name; }

void set_file_line(unsigned file_line) { this->file_line = file_line; }

unsigned get_file_line() const { return file_line; }

char* next_line()
{
    if(data_begin == data_end)
    {
        return nullptr;
    }

    ++file_line;

    Assert(data_begin < data_end);
    Assert(data_end <= block_len * 2);

    if(data_begin >= block_len)
    {
        std::memcpy(buffer.get(), buffer.get() + block_len, block_len);
        data_begin -= block_len;
        data_end -= block_len;
        if(reader.is_valid())
        {
            data_end += reader.finish_read();
            std::memcpy(buffer.get() + block_len, buffer.get() + 2 * block_len, block_len);
            reader.start_read(buffer.get() + 2 * block_len, block_len);
        }
    }

    int line_end = data_begin;
    while(buffer[line_end] != '\n' && line_end != data_end)
    {
        ++line_end;
    }

    if(line_end - data_begin + 1 > block_len)
    {
        error::line_length_limit_exceeded err;
        err.set_file_name(file_name);
        err.set_file_line(file_line);
        throw err;
    }

    if(buffer[line_end] == '\n' && line_end != data_end)
    {
        buffer[line_end] = '\0';
    }
    else
    {
        // some files are missing the newline at the end of the
        // last line
        ++data_end;
        buffer[line_end] = '\0';
    }

    // handle windows \r\n-line breaks
    if(line_end != data_begin && buffer[line_end - 1] == '\r')
    {
        buffer[line_end - 1] = '\0';
    }

    char* ret  = buffer.get() + data_begin;
    data_begin = line_end + 1;
    return ret;
}
};

template<unsigned column_count, class trim_policy = trim_chars<' ', '\t'>, class quote_policy = no_quote_escape<','>, class overflow_policy = throw_on_overflow, class comment_policy = no_comment>
class CSVReader
{
private:
LineReader in;

char*       row[column_count];
std::string column_names[column_count];

std::vector<int> col_order;

template<class... ColNames>
void set_column_names(std::string s, ColNames... cols)
{
    column_names[column_count - sizeof...(ColNames) - 1] = std::move(s);
    set_column_names(std::forward<ColNames>(cols)...);
}

void set_column_names() {}

public:
CSVReader()                 = delete;
CSVReader(const CSVReader&) = delete;
CSVReader& operator=(const CSVReader&) = delete;

template<class... Args>
explicit CSVReader(Args&&... args) : in(std::forward<Args>(args)...)
{
    std::fill(row, row + column_count, nullptr);
    col_order.resize(column_count);
    for(unsigned i = 0; i < column_count; ++i)
    {
        col_order[i] = i;
    }
    for(unsigned i = 1; i <= column_count; ++i)
    {
        column_names[i - 1] = "col" + std::to_string(i);
    }
}

char* next_line() { return in.next_line(); }

template<class... ColNames>
void read_header(ignore_column ignore_policy, ColNames... cols)
{
    static_assert(sizeof...(ColNames) >= column_count, "not enough column names specified");
    static_assert(sizeof...(ColNames) <= column_count, "too many column names specified");
    try
    {
        set_column_names(std::forward<ColNames>(cols)...);

        char* line;
        do
        {
            line = in.next_line();
            if(!line)
                throw error::header_missing();
        } while(comment_policy::is_comment(line));

        detail::parse_header_line<column_count, trim_policy, quote_policy>(line, col_order, column_names, ignore_policy);
    }
    catch(error::with_file_name& err)
    {
        err.set_file_name(in.get_truncated_file_name());
        throw;
    }
}

template<class... ColNames>
void set_header(ColNames... cols)
{
    static_assert(sizeof...(ColNames) >= column_count, "not enough column names specified");
    static_assert(sizeof...(ColNames) <= column_count, "too many column names specified");

    set_column_names(std::forward<ColNames>(cols)...);
    std::fill(row, row + column_count, nullptr);
    col_order.resize(column_count);

    for(unsigned i = 0; i < column_count; ++i)
    {
        col_order[i] = i;
    }
}

bool has_column(const std::string& name) const
{
    return col_order.end() != std::find(col_order.begin(), col_order.end(), std::find(std::begin(column_names), std::end(column_names), name) - std::begin(column_names));
}

void set_file_name(const std::string& file_name) { in.set_file_name(file_name); }

void set_file_name(const char* file_name) { in.set_file_name(file_name); }

const char* get_truncated_file_name() const { return in.get_truncated_file_name(); }

void set_file_line(unsigned file_line) { in.set_file_line(file_line); }

unsigned get_file_line() const { return in.get_file_line(); }

private:
void parse_helper(std::size_t) {}

template<class T, class... ColType>
void parse_helper(std::size_t r, T& t, ColType&... cols)
{
    if(row[r])
    {
        try
        {
            try
            {
                ::io::detail::parse<overflow_policy>(row[r], t);
            }
            catch(error::with_column_content& err)
            {
                err.set_column_content(row[r]);
                throw;
            }
        }
        catch(error::with_column_name& err)
        {
            err.set_column_name(column_names[r].c_str());
            throw;
        }
    }
    parse_helper(r + 1, cols...);
}

public:
template<class... ColType>
bool read_row(ColType&... cols)
{
    static_assert(sizeof...(ColType) >= column_count, "not enough columns specified");
    static_assert(sizeof...(ColType) <= column_count, "too many columns specified");
    try
    {
        try
        {
            char* line;
            do
            {
                line = in.next_line();
                if(!line)
                {
                    return false;
                }
            } while(comment_policy::is_comment(line));

            detail::parse_line<trim_policy, quote_policy>(line, row, col_order);

            parse_helper(0, cols...);
        }
        catch(error::with_file_name& err)
        {
            err.set_file_name(in.get_truncated_file_name());
            throw;
        }
    }
    catch(error::with_file_line& err)
    {
        err.set_file_line(in.get_file_line());
        throw;
    }

    return true;
}
};

}
*/
