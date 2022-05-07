#pragma once

#include <Types.hpp>
#include <Memory.hpp>

#include <runtime.Kokkos/ViewTypes.hpp>
#include <Utilities/SharedMemory.h>

#include <Sacado_TemplateContainer.hpp>
//#include <Sacado_mpl_at.hpp>
#include <Sacado_mpl_vector.hpp>
#include <Sacado_mpl_placeholders.hpp>

namespace Engineering::DataSource
{

    class ISharedDataColumn
    {
    protected:
        int32 _relativeOffset;
        int32 _thisSize;

        SharedMemoryData _data;

        __inline constexpr ISharedDataColumn(const int32 relativeOffset, const int32 thisSize, const SharedMemoryData data) : _relativeOffset(relativeOffset), _thisSize(thisSize), _data(data) {}

    public:
        virtual ~ISharedDataColumn() = default;

        KOKKOS_INLINE_FUNCTION constexpr ISharedDataColumn(const ISharedDataColumn& other) : _relativeOffset(other._relativeOffset), _thisSize(other._thisSize), _data(other._data) {}

        KOKKOS_INLINE_FUNCTION constexpr ISharedDataColumn(ISharedDataColumn&& other) noexcept : _relativeOffset(other._relativeOffset), _thisSize(other._thisSize), _data(std::move(other._data)) {}

        KOKKOS_INLINE_FUNCTION constexpr ISharedDataColumn& operator=(const ISharedDataColumn& other)
        {
            if (this == &other)
            {
                return *this;
            }
            _relativeOffset = other._relativeOffset;
            _thisSize       = other._thisSize;
            _data           = other._data;
            return *this;
        }

        KOKKOS_INLINE_FUNCTION constexpr ISharedDataColumn& operator=(ISharedDataColumn&& other) noexcept
        {
            if (this == &other)
            {
                return *this;
            }
            _relativeOffset = other._relativeOffset;
            _thisSize       = other._thisSize;
            _data           = std::move(other._data);
            return *this;
        }

        KOKKOS_INLINE_FUNCTION virtual void* operator[](const int32 index) const = 0;

        KOKKOS_INLINE_FUNCTION virtual void* operator()(const int32 index) const = 0;

        KOKKOS_INLINE_FUNCTION virtual void* Get(const int32 index) const = 0;

        KOKKOS_INLINE_FUNCTION const int32& Size() const
        {
            return _thisSize;
        }

        __inline virtual const System::PrimitiveKind& DataType() const = 0;

        template<typename T>
        __inline static T* As(void* address)
        {
            return reinterpret_cast<T*>(address); // System::addressof(
        }
    };

    template<typename T>
    class SharedDataColumn : public ISharedDataColumn
    {
        __inline static const System::PrimitiveKind& type()
        {
            static const System::PrimitiveKind t(System::TypeToKind<T>());
            return t;
        }

        KOKKOS_INLINE_FUNCTION static constexpr int32 sizeOfType()
        {
            return sizeof(T);
        }

    public:
        constexpr SharedDataColumn() = default;
        //~SharedDataColumn() override = default;

        __inline constexpr SharedDataColumn(const int32 offset, const int32 nRows, const SharedMemoryData data) : ISharedDataColumn(offset, (nRows * sizeOfType()), data) {}

        KOKKOS_INLINE_FUNCTION constexpr SharedDataColumn(const SharedDataColumn& other) : ISharedDataColumn(other._relativeOffset, other._thisSize, other._data) {}

        KOKKOS_INLINE_FUNCTION constexpr SharedDataColumn(SharedDataColumn&& other) noexcept : ISharedDataColumn(other._relativeOffset, other._thisSize, std::move(other._data)) {}

        KOKKOS_INLINE_FUNCTION constexpr SharedDataColumn& operator=(const SharedDataColumn& other)
        {
            if (this == &other)
            {
                return *this;
            }

            ISharedDataColumn::operator=(other);

            return *this;
        }

        KOKKOS_INLINE_FUNCTION constexpr SharedDataColumn& operator=(SharedDataColumn&& other) noexcept
        {
            if (this == &other)
            {
                return *this;
            }

            ISharedDataColumn::operator=(other);

            return *this;
        }

        KOKKOS_INLINE_FUNCTION constexpr uint64 Offset() const
        {
            return (reinterpret_cast<uint64>(_data.HostAddress) + _relativeOffset);
        }

        KOKKOS_INLINE_FUNCTION void* operator[](const int32 index) const override
        {
            return reinterpret_cast<void*>(Offset() + (sizeOfType() * index));
        }

        KOKKOS_INLINE_FUNCTION void* operator()(const int32 index) const override
        {
            return reinterpret_cast<void*>(Offset() + (sizeOfType() * index));
        }

        KOKKOS_INLINE_FUNCTION void* Get(const int32 index) const override
        {
            return reinterpret_cast<void*>(Offset() + (sizeOfType() * index));
        }

        __inline const System::PrimitiveKind& DataType() const override
        {
            return type();
        }
    };

    class SharedDataColumnHeader
    {
        int32 _thisSize;

        size_type length;

        int32* _titleLengths;

        uint64* _titleIndices;

        SharedMemoryData _data;

    public:
        SharedDataColumnHeader() : _thisSize(0), length(0), _titleLengths(nullptr), _titleIndices(nullptr), _data{} {}

        SharedDataColumnHeader(const SharedMemoryData data, const std::vector<std::wstring>& columnNames) : length(columnNames.size())
        {
            _data = data;

            _titleLengths = new int32[length];
            _titleIndices = new uint64[length];

            _titleIndices[0] = 0;

            void* root;

            for (uint32 i = 0; i < length; ++i)
            {
                _titleLengths[i] = (columnNames.begin() + i)->size() * sizeof(wchar_t);

                if (i < length - 1)
                {
                    _titleIndices[i + 1] = _titleIndices[i] + _titleLengths[i];
                }

                root = reinterpret_cast<void*>(reinterpret_cast<uint64>(_data.HostAddress) + _titleIndices[i]);

                const int32 length = _titleLengths[i] / sizeof(wchar_t);

                wchar_t* memory = reinterpret_cast<wchar_t*>(root);

                for (int32 j = 0; j < length; ++j)
                {
                    memory[j] = (columnNames.begin() + i)->at(j);
                }
            }

            int32 this_size = 0;

            for (size_type i = 0; i < length; ++i)
            {
                this_size += _titleLengths[i];
            }

            _thisSize = this_size;
        }

        SharedDataColumnHeader(const SharedDataColumnHeader& other) : _thisSize(other._thisSize), length(other.length), _titleLengths(new int32[length]), _titleIndices(new uint64[length]), _data(other._data)
        {
            System::Memory::Copy<int32>(_titleLengths, other._titleLengths, length);
            System::Memory::Copy<uint64>(_titleIndices, other._titleIndices, length);
        }

        SharedDataColumnHeader(SharedDataColumnHeader&& other) noexcept : _thisSize(other._thisSize), length(other.length), _titleLengths(new int32[length]), _titleIndices(new uint64[length]), _data(std::move(other._data)) {}

        SharedDataColumnHeader& operator=(const SharedDataColumnHeader& other)
        {
            if (this == &other)
            {
                return *this;
            }
            _thisSize     = other._thisSize;
            length        = other.length;
            _titleLengths = new int32[length];
            _titleIndices = new uint64[length];
            System::Memory::Copy<int32>(_titleLengths, other._titleLengths, length);
            System::Memory::Copy<uint64>(_titleIndices, other._titleIndices, length);
            _data = other._data;
            return *this;
        }

        SharedDataColumnHeader& operator=(SharedDataColumnHeader&& other) noexcept
        {
            if (this == &other)
            {
                return *this;
            }
            _thisSize     = other._thisSize;
            length        = other.length;
            _titleLengths = new int32[length];
            _titleIndices = new uint64[length];
            System::Memory::Copy<int32>(_titleLengths, other._titleLengths, length);
            System::Memory::Copy<uint64>(_titleIndices, other._titleIndices, length);
            _data = std::move(other._data);
            return *this;
        }

        __inline std::wstring operator[](int32 index) const
        {
            const int32 length = _titleLengths[index] / sizeof(wchar_t);

            const wchar_t* memory = reinterpret_cast<wchar_t*>(reinterpret_cast<uint64>(_data.HostAddress) + _titleIndices[index]);

            return std::wstring(memory, length);
        }

        __inline std::wstring operator()(int32 index) const
        {
            const int32 length = _titleLengths[index] / sizeof(wchar_t);

            const wchar_t* memory = reinterpret_cast<wchar_t*>(reinterpret_cast<uint64>(_data.HostAddress) + _titleIndices[index]);

            return std::wstring(memory, length);
        }

        KOKKOS_INLINE_FUNCTION constexpr int32 operator[](const wchar_t* name) const
        {
            for (int32 i = 0; i < length; ++i)
            {
                if (name == (*this)(i))
                {
                    return i;
                }
            }

            return -1;
        }

        KOKKOS_INLINE_FUNCTION constexpr int32 operator()(const wchar_t* name) const
        {
            for (int32 i = 0; i < length; ++i)
            {
                if (name == (*this)(i))
                {
                    return i;
                }
            }

            return -1;
        }

        KOKKOS_INLINE_FUNCTION constexpr const int32& Size() const
        {
            return _thisSize;
        }

        KOKKOS_INLINE_FUNCTION constexpr const size_type& Length() const
        {
            return length;
        }
    };

    enum class SharedDataTableMode
    {
        Create,
        Open
    };

    template<System::PrimitiveKind PrimitiveType = System::PrimitiveKind::NaN>
    struct SharedDataTableColumnSchema;

    template<>
    struct SharedDataTableColumnSchema<System::PrimitiveKind::NaN>
    {
        System::PrimitiveKind type;
        std::wstring          name;

        SharedDataTableColumnSchema()  = delete;
        ~SharedDataTableColumnSchema() = default;

        SharedDataTableColumnSchema(const System::PrimitiveKind type, const std::wstring& name) : type(type), name(name) {}

        SharedDataTableColumnSchema(const SharedDataTableColumnSchema& other) : type(other.type), name(other.name) {}

        SharedDataTableColumnSchema(SharedDataTableColumnSchema&& other) noexcept : type(std::move(other.type)), name(std::move(other.name)) {}

        SharedDataTableColumnSchema& operator=(const SharedDataTableColumnSchema& other)
        {
            if (this == &other)
            {
                return *this;
            }
            type = other.type;
            name = other.name;
            return *this;
        }

        SharedDataTableColumnSchema& operator=(SharedDataTableColumnSchema&& other) noexcept
        {
            if (this == &other)
            {
                return *this;
            }
            type = std::move(other.type);
            name = std::move(other.name);
            return *this;
        }
    };

    template<System::PrimitiveKind PrimitiveType>
    struct SharedDataTableColumnSchema : public SharedDataTableColumnSchema<>
    {
        SharedDataTableColumnSchema()  = delete;
        ~SharedDataTableColumnSchema() = default;

        SharedDataTableColumnSchema(const std::wstring& name) : SharedDataTableColumnSchema<>(PrimitiveType, name) {}

        SharedDataTableColumnSchema(const System::PrimitiveKind type, const std::wstring& name) : SharedDataTableColumnSchema<>(type, name) {}

        SharedDataTableColumnSchema(const SharedDataTableColumnSchema& other) : SharedDataTableColumnSchema<>(other.type, other.name) {}

        SharedDataTableColumnSchema(SharedDataTableColumnSchema&& other) noexcept : SharedDataTableColumnSchema<>(std::move(other.type), std::move(other.name)) {}

        SharedDataTableColumnSchema& operator=(const SharedDataTableColumnSchema& other)
        {
            if (this == &other)
            {
                return *this;
            }

            SharedDataTableColumnSchema<>::operator=(other);

            return *this;
        }

        SharedDataTableColumnSchema& operator=(SharedDataTableColumnSchema&& other) noexcept
        {
            if (this == &other)
            {
                return *this;
            }

            SharedDataTableColumnSchema<>::operator=(other);

            return *this;
        }
    };

    template<System::PrimitiveKind PrimitiveType>
    using SharedDataTableColumnType = SharedDataTableColumnSchema<System::PrimitiveKind::NaN>;

    template<class T>
    struct SharedDataTableColumnData
    {
        using Type = T;

        System::PrimitiveKind type;
        std::wstring          name;
        int32                 size;

        SharedDataTableColumnData() : type(System::TypeToKind<T>()), name(), size(0) {}
    };

    template<class Container>
    struct SharedDataTableSetFunctor
    {
        Container& column;

        SharedDataTableSetFunctor(Container& c) : column(c) {}

        template<typename T>
        void operator()(T) const
        {
            auto& col = column.template get<T>();

            System::PrimitiveKind ti = System::TypeToKind<T>();

            col.type = ti;

            const std::string name = System::TypeToString<T>();

            col.name = std::wstring(name.begin(), name.end());

            col.size = sizeof(T);
        }
    };

    template<typename... TArgs>
    struct SharedDataTableStaticConstructor
    {
        typedef Sacado::mpl::placeholders::_ _;

        typedef Sacado::mpl::vector<TArgs...> ColumnDataTypes;

        typedef Sacado::TemplateContainer<ColumnDataTypes, SharedDataTableColumnData<_>> ColumnContainerType;

        static ColumnContainerType Constructor()
        {
            ColumnContainerType ColumnContainer;

            Sacado::container_for_each_no_kokkos(ColumnContainer, SharedDataTableSetFunctor<ColumnContainerType>(ColumnContainer));

            return ColumnContainer;
        }
    };

    template<typename... TArgs>
    struct SharedDataTable
    {
        typedef SharedDataTableStaticConstructor<TArgs...> SharedDataTableStaticConstructorType;

        typedef typename SharedDataTableStaticConstructorType::ColumnDataTypes ColumnDataTypes;

        typedef typename SharedDataTableStaticConstructorType::ColumnContainerType ColumnContainerType;

        __inline static ColumnContainerType ColumnData;

        uint64 _thisSize;

        SharedMemoryData _data;

        SharedDataColumnHeader Header;

        ISharedDataColumn** Columns;

        __inline SharedDataTable(const SharedDataTableMode mode, const std::wstring sharedMemoryName, const int32 nRows, const int32 nColumns, const SharedDataTableColumnSchema<>* columns)
        {
            // if (ColumnData.template get<Sacado::mpl::at<typename SharedDataTableStaticConstructorType::ColumnDataTypes, 0>>().size == 0)
            {
                ColumnData = SharedDataTableStaticConstructorType::Constructor();
            }

            const size_type length = nColumns;

            int32 headerSize = 0;

            std::vector<std::wstring> columnNames;
            columnNames.reserve(length);

            for (uint32 i = 0; i < length; ++i)
            {
                const std::wstring name = columns[i].name;

                headerSize += name.size() * sizeof(wchar_t);
                columnNames.push_back(name);
            }

            std::vector<int32> dataTypeSizes;
            dataTypeSizes.reserve(length);

            int32 dataSize = 0;

            for (uint32 i = 0; i < length; ++i)
            {
                Sacado::mpl::for_each_no_kokkos<ColumnContainerType>(
                    [&]<typename T>(T)
                    {
                        // auto index = Sacado::mpl::find<ColumnContainerType,T>::value;

                        if (columns[i].type == System::TypeToKind<T>())
                        {
                            const int32 dataTypeSize = ColumnData.template get<T>().size;
                            dataTypeSizes.push_back(dataTypeSize);
                            dataSize += dataTypeSize * nRows;
                        }
                    });
            }

            _thisSize = headerSize + dataSize;

            const wchar_t* wsharedMemoryName = sharedMemoryName.c_str();

            if (mode == SharedDataTableMode::Create)
            {
                SharedMemoryCreate(wsharedMemoryName, _thisSize, &_data);
            }
            else
            {
                SharedMemoryOpen(wsharedMemoryName, _thisSize, &_data);
            }

            Columns = new ISharedDataColumn*[nColumns];

            Header = SharedDataColumnHeader(_data, columnNames);

            for (uint32 i = 0; i < length; ++i)
            {
                // Sacado::mpl::find<ColumnData, >

                Sacado::mpl::for_each_no_kokkos<ColumnContainerType>(
                    [&]<typename T>(T)
                    {
                        if (columns[i].type == System::TypeToKind<T>()) // ColumnData.template get<T>().type)
                        {
                            Columns[i] = new SharedDataColumn<T>((int32)(headerSize + (i * dataTypeSizes[i] * nRows)), nRows, _data);
                        }
                    });
            }

            // Columns = new ValueTuple<SharedDataColumn<T1>, SharedDataColumn<T2>>(new SharedDataColumn<T1>(_data, headerSize, nRows), new SharedDataColumn<T2>(_data, headerSize + (Unsafe.SizeOf<T2>() * nRows), nRows));
        }

        __inline ~SharedDataTable()
        {
            if (_data.Handle == nullptr)
            {
                SharedMemoryClose(&_data);
            }
        }

        __inline SharedDataTable(const SharedDataTable& other) : _thisSize(other._thisSize), _data(other._data), Header(other.Header), Columns(other.Columns) {}

        __inline SharedDataTable(SharedDataTable&& other) noexcept : _thisSize(other._thisSize), _data(std::move(other._data)), Header(std::move(other.Header)), Columns(other.Columns) {}

        __inline SharedDataTable& operator=(const SharedDataTable& other)
        {
            if (this == &other)
            {
                return *this;
            }
            _thisSize = other._thisSize;
            _data     = other._data;
            Header    = other.Header;
            Columns   = other.Columns;
            return *this;
        }

        __inline SharedDataTable& operator=(SharedDataTable&& other) noexcept
        {
            if (this == &other)
            {
                return *this;
            }
            _thisSize = other._thisSize;
            _data     = std::move(other._data);
            Header    = std::move(other.Header);
            Columns   = other.Columns;
            return *this;
        }

        template<uint32 nColummns>
        KOKKOS_INLINE_FUNCTION constexpr static SharedDataTable<TArgs...>* Create(const std::wstring& sharedMemoryName, const int32 nRows, const SharedDataTableColumnSchema<> (&columns)[nColummns])
        {
            return new SharedDataTable<TArgs...>(SharedDataTableMode::Create, sharedMemoryName, nRows, nColummns, columns);
        }

        template<uint32 nColummns>
        KOKKOS_INLINE_FUNCTION constexpr static SharedDataTable<TArgs...>* Open(const std::wstring& sharedMemoryName, const int32 nRows, const SharedDataTableColumnSchema<> (&columns)[nColummns])
        {
            return new SharedDataTable<TArgs...>(SharedDataTableMode::Open, sharedMemoryName, nRows, nColummns, columns);
        }

        KOKKOS_INLINE_FUNCTION constexpr void* operator()(const int32 rowIndex, const int32 columnIndex) const
        {
            const ISharedDataColumn* column = Columns[columnIndex];

            return GetValue(rowIndex, column);

            // Sacado::mpl::for_each_no_kokkos<ColumnContainerType>(
            //    [&]<typename T>(T)
            //    {
            //        if (column->DataType() == System::TypeToKind<T>()) // ColumnData.template get<T>().type)
            //        {

            //        }
            //    });

            // return nullptr;

            // SetValue(rowIndex, column, (T1)value);
        }

        template<typename TValue>
        KOKKOS_INLINE_FUNCTION constexpr void SetValue(const int32 rowIndex, const ISharedDataColumn* column, const TValue value)
        {
            *ISharedDataColumn::As<TValue>((*column)(rowIndex)) = value;
        }

        template<typename TValue>
        KOKKOS_INLINE_FUNCTION constexpr void SetValue(const int32 rowIndex, const int32 columnIndex, const TValue value)
        {
            const ISharedDataColumn* column                     = Columns[columnIndex];
            *ISharedDataColumn::As<TValue>((*column)(rowIndex)) = value;
        }

        template<typename TValue>
        KOKKOS_INLINE_FUNCTION constexpr void GetValue(const int32 rowIndex, const ISharedDataColumn* column, TValue* value)
        {
            *value = ISharedDataColumn::As<TValue>((*column)(rowIndex));
        }

        template<typename TValue>
        KOKKOS_INLINE_FUNCTION constexpr void GetValue(const int32 rowIndex, const int32 columnIndex, TValue* value)
        {
            const ISharedDataColumn* column = Columns[columnIndex];
            *value                          = ISharedDataColumn::As<TValue>((*column)(rowIndex));
        }

        KOKKOS_INLINE_FUNCTION static constexpr void* GetValue(const int32 rowIndex, const ISharedDataColumn* column)
        {
            return (*column)(rowIndex);
        }

        KOKKOS_INLINE_FUNCTION constexpr void* GetValue(const int32 rowIndex, int32 columnIndex) const
        {
            const ISharedDataColumn* column = Columns[columnIndex];

            return (*column)(rowIndex);
        }

        template<typename TValue>
        KOKKOS_INLINE_FUNCTION constexpr TValue* Get(const int32 rowIndex, int32 columnIndex)
        {
            const ISharedDataColumn* column = Columns[columnIndex];

            return ISharedDataColumn::As<TValue>((*column)(rowIndex));
        }

        KOKKOS_INLINE_FUNCTION constexpr const uint64& Size() const
        {
            return _thisSize;
        }
    };

}

namespace Engineering::DataSource::Layout
{
    //struct Null
    //{
    //};

    //struct Struct
    //{
    //};

    //struct List
    //{
    //};

    //struct LargeList
    //{
    //};

    //struct FixedSizeList
    //{
    //    int listSize;
    //};

    //struct Map
    //{
    //    bool keysSorted;
    //};

    //enum class UnionMode : short
    //{
    //    Sparse,
    //    Dense
    //};

    //struct Union
    //{
    //    UnionMode mode;
    //    int       typeIds[];
    //};

    //struct Int
    //{
    //    int  bitWidth;
    //    bool is_signed;
    //};

    //enum class Precision : short
    //{
    //    HALF,
    //    SINGLE,
    //    DOUBLE
    //};

    //struct FloatingPoint
    //{
    //    Precision precision;
    //};

    //struct Utf8
    //{
    //};

    //struct Binary
    //{
    //};

    //struct LargeUtf8
    //{
    //};

    //struct LargeBinary
    //{
    //};

    //struct FixedSizeBinary
    //{
    //    int byteWidth;
    //};

    //struct Bool
    //{
    //};

    //struct Decimal
    //{
    //    int precision;

    //    int scale;

    //    int bitWidth = 128;
    //};

    //enum class DateUnit : short
    //{
    //    DAY,
    //    MILLISECOND
    //};

    //struct Date
    //{
    //    DateUnit unit = DateUnit::MILLISECOND;
    //};

    //enum class TimeUnit : short
    //{
    //    SECOND,
    //    MILLISECOND,
    //    MICROSECOND,
    //    NANOSECOND
    //};

    //struct Time
    //{
    //    TimeUnit unit     = TimeUnit::MILLISECOND;
    //    int      bitWidth = 32;
    //};

    //struct Timestamp
    //{
    //    TimeUnit unit;

    //    std::string timezone;
    //};

    //enum class IntervalUnit : short
    //{
    //    YEAR_MONTH,
    //    DAY_TIME,
    //    MONTH_DAY_NANO
    //};

    //struct Interval
    //{
    //    IntervalUnit unit;
    //};

    //struct Duration
    //{
    //    TimeUnit unit = TimeUnit::MILLISECOND;
    //};

    //union Type
    //{
    //    Null            NullType;
    //    Int             IntType;
    //    FloatingPoint   FloatingPointType;
    //    Binary          BinaryType;
    //    Utf8            Utf8Type;
    //    Bool            BoolType;
    //    Decimal         DecimalType;
    //    Date            DateType;
    //    Time            TimeType;
    //    Timestamp       TimestampType;
    //    Interval        IntervalType;
    //    List            ListType;
    //    Struct          StructType;
    //    Union           UnionType;
    //    FixedSizeBinary FixedSizeBinaryType;
    //    FixedSizeList   FixedSizeListType;
    //    Map             MapType;
    //    Duration        DurationType;
    //    LargeBinary     LargeBinaryType;
    //    LargeUtf8       LargeUtf8Type;
    //    LargeList       LargeListType;
    //};

    //struct KeyValue
    //{
    //    std::string key;
    //    std::string value;
    //};

    //enum class DictionaryKind : short
    //{
    //    DenseArray
    //};

    //struct DictionaryEncoding
    //{
    //    long id;

    //    Int indexType;

    //    bool isOrdered;

    //    DictionaryKind dictionaryKind;
    //};

    //struct Field
    //{
    //    std::string name;

    //    bool nullable;

    //    Type* type;

    //    DictionaryEncoding* dictionary;

    //    Field* children[];

    //    KeyValue custom_metadata[];
    //};

    //enum class Feature : long
    //{
    //    UNUSED                 = 0,
    //    DICTIONARY_REPLACEMENT = 1,
    //    COMPRESSED_BODY        = 2
    //};

    //struct Schema
    //{
    //    Field fields[];

    //    KeyValue custom_metadata[];

    //    Feature features[];
    //};

}
