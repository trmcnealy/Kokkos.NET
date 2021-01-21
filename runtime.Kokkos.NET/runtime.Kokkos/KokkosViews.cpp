
#include "runtime.Kokkos/ViewTypes.hpp"

// MAKE_UID_INTERFACE_T(IView, "C12ECB0C-365E-4F85-8084-6D1E1261A5F0", template<typename TDataType, typename TExecustionSpace>)

extern "C" __declspec(dllexport) __declspec(selectany) constexpr char IViewUid[] = "C12ECB0C-365E-4F85-8084-6D1E1261A5F0";
typedef GuidType<IViewUid> IViewGuid;

template<unsigned Rank, typename TDataType, typename TExecustionSpace>
struct __declspec(novtable) __declspec(uuid("C12ECB0C-365E-4F85-8084-6D1E1261A5F0")) __declspec(dllexport) IView
{
    virtual ~IView() = default;

    virtual NativeString GetLabel() const = 0;

    KOKKOS_FUNCTION virtual uint64 GetSize() const = 0;

    KOKKOS_FUNCTION virtual uint32 GetRank() const = 0;

    KOKKOS_FUNCTION virtual uint64 GetStride(uint32 dim) const = 0;

    KOKKOS_FUNCTION virtual uint64 GetExtent(uint32 dim) const = 0;

    virtual TDataType& operator()()
    {
        throw "This method is not supported for this rank.";
    }

    virtual TDataType& operator()(const int32 i0) const
    {
        throw "This method is not supported for this rank.";
    }

    virtual TDataType& operator()(const int32 i0, const int32 i1) const
    {
        throw "This method is not supported for this rank.";
    }

    virtual TDataType& operator()(const int32 i0, const int32 i1, const int32 i2) const
    {
        throw "This method is not supported for this rank.";
    }

    virtual TDataType& operator()(const int32 i0, const int32 i1, const int32 i2, const int32 i3) const
    {
        throw "This method is not supported for this rank.";
    }

    virtual TDataType& operator()(const int32 i0, const int32 i1, const int32 i2, const int32 i3, const int32 i4) const
    {
        throw "This method is not supported for this rank.";
    }

    virtual TDataType& operator()(const int32 i0, const int32 i1, const int32 i2, const int32 i3, const int32 i4, const int32 i5) const
    {
        throw "This method is not supported for this rank.";
    }

    virtual TDataType& operator()(const int32 i0, const int32 i1, const int32 i2, const int32 i3, const int32 i4, const int32 i5, const int32 i6) const
    {
        throw "This method is not supported for this rank.";
    }

    virtual TDataType& operator()(const int32 i0, const int32 i1, const int32 i2, const int32 i3, const int32 i4, const int32 i5, const int32 i6, const int32 i7) const
    {
        throw "This method is not supported for this rank.";
    }
};

#undef TEMPLATE
#undef DEF_TEMPLATE

#define TEMPLATE(DEF, EXECUTION_SPACE)                                                                                                                                             \
    DEF(Single, float, EXECUTION_SPACE)                                                                                                                                            \
    DEF(Double, double, EXECUTION_SPACE)                                                                                                                                           \
    DEF(Bool, bool, EXECUTION_SPACE)                                                                                                                                               \
    DEF(Int8, int8, EXECUTION_SPACE)                                                                                                                                               \
    DEF(UInt8, uint8, EXECUTION_SPACE)                                                                                                                                             \
    DEF(Int16, int16, EXECUTION_SPACE)                                                                                                                                             \
    DEF(UInt16, uint16, EXECUTION_SPACE)                                                                                                                                           \
    DEF(Int32, int32, EXECUTION_SPACE)                                                                                                                                             \
    DEF(UInt32, uint32, EXECUTION_SPACE)                                                                                                                                           \
    DEF(Int64, int64, EXECUTION_SPACE)                                                                                                                                             \
    DEF(UInt64, uint64, EXECUTION_SPACE)                                                                                                                                           \
    DEF(Char, wchar_t, EXECUTION_SPACE)

#define DEF_TEMPLATE(TYPE_NAME, TYPE, EXECUTION_SPACE)                                                                                                                             \
    class __declspec(dllexport) View_##TYPE_NAME##_##EXECUTION_SPACE##_0D : IView<0, TYPE, Kokkos::EXECUTION_SPACE>                                                                \
    {                                                                                                                                                                              \
        using ViewType = Kokkos::View<TYPE, Kokkos::EXECUTION_SPACE::array_layout, Kokkos::EXECUTION_SPACE>;                                                                       \
                                                                                                                                                                                   \
        ViewType view;                                                                                                                                                             \
                                                                                                                                                                                   \
    public:                                                                                                                                                                        \
        View_##TYPE_NAME##_##EXECUTION_SPACE##_0D(const NativeString arg_label);                                                                                                   \
                                                                                                                                                                                   \
        ~View_##TYPE_NAME##_##EXECUTION_SPACE##_0D() = default;                                                                                                                    \
                                                                                                                                                                                   \
        TYPE&                  operator()() override;                                                                                                                              \
        NativeString           GetLabel() const override;                                                                                                                          \
        KOKKOS_FUNCTION uint64 GetSize() const override;                                                                                                                           \
        KOKKOS_FUNCTION uint32 GetRank() const override;                                                                                                                           \
        KOKKOS_FUNCTION uint64 GetStride(uint32 dim) const override;                                                                                                               \
        KOKKOS_FUNCTION uint64 GetExtent(uint32 dim) const override;                                                                                                               \
    };                                                                                                                                                                             \
    __declspec(dllexport) View_##TYPE_NAME##_##EXECUTION_SPACE##_0D::View_##TYPE_NAME##_##EXECUTION_SPACE##_0D(const NativeString arg_label) : view(arg_label.ToString()) {}       \
    __declspec(dllexport) TYPE& View_##TYPE_NAME##_##EXECUTION_SPACE##_0D::operator()()                                                                                            \
    {                                                                                                                                                                              \
        return view();                                                                                                                                                             \
    }                                                                                                                                                                              \
    __declspec(dllexport) NativeString View_##TYPE_NAME##_##EXECUTION_SPACE##_0D::GetLabel() const                                                                                 \
    {                                                                                                                                                                              \
        return NativeString(view.label());                                                                                                                                         \
    }                                                                                                                                                                              \
    __declspec(dllexport) KOKKOS_FUNCTION uint64 View_##TYPE_NAME##_##EXECUTION_SPACE##_0D::GetSize() const                                                                        \
    {                                                                                                                                                                              \
        return view.size();                                                                                                                                                        \
    }                                                                                                                                                                              \
    __declspec(dllexport) KOKKOS_FUNCTION uint32 View_##TYPE_NAME##_##EXECUTION_SPACE##_0D::GetRank() const                                                                        \
    {                                                                                                                                                                              \
        return ViewType::Rank;                                                                                                                                                     \
    }                                                                                                                                                                              \
    __declspec(dllexport) KOKKOS_FUNCTION uint64 View_##TYPE_NAME##_##EXECUTION_SPACE##_0D::GetStride(uint32 dim) const                                                            \
    {                                                                                                                                                                              \
        return view.stride(dim);                                                                                                                                                   \
    }                                                                                                                                                                              \
    __declspec(dllexport) KOKKOS_FUNCTION uint64 View_##TYPE_NAME##_##EXECUTION_SPACE##_0D::GetExtent(uint32 dim) const                                                            \
    {                                                                                                                                                                              \
        return view.extent(dim);                                                                                                                                                   \
    }                                                                                                                                                                              \
    class __declspec(dllexport) View_##TYPE_NAME##_##EXECUTION_SPACE##_1D : IView<1, TYPE, Kokkos::EXECUTION_SPACE>                                                                \
    {                                                                                                                                                                              \
        using ViewType = Kokkos::View<TYPE*, Kokkos::EXECUTION_SPACE::array_layout, Kokkos::EXECUTION_SPACE>;                                                                      \
                                                                                                                                                                                   \
        ViewType view;                                                                                                                                                             \
                                                                                                                                                                                   \
    public:                                                                                                                                                                        \
        View_##TYPE_NAME##_##EXECUTION_SPACE##_1D(const NativeString arg_label, const size_t arg_N0);                                                                              \
                                                                                                                                                                                   \
        ~View_##TYPE_NAME##_##EXECUTION_SPACE##_1D() = default;                                                                                                                    \
                                                                                                                                                                                   \
        TYPE&                  operator()(const int32 i0) const override;                                                                                                          \
        NativeString           GetLabel() const override;                                                                                                                          \
        KOKKOS_FUNCTION uint64 GetSize() const override;                                                                                                                           \
        KOKKOS_FUNCTION uint32 GetRank() const override;                                                                                                                           \
        KOKKOS_FUNCTION uint64 GetStride(uint32 dim) const override;                                                                                                               \
        KOKKOS_FUNCTION uint64 GetExtent(uint32 dim) const override;                                                                                                               \
    };                                                                                                                                                                             \
                                                                                                                                                                                   \
    __declspec(dllexport) View_##TYPE_NAME##_##EXECUTION_SPACE##_1D::View_##TYPE_NAME##_##EXECUTION_SPACE##_1D(const NativeString arg_label, const size_t arg_N0) :                \
        view(arg_label.ToString(), arg_N0)                                                                                                                                         \
    {                                                                                                                                                                              \
    }                                                                                                                                                                              \
    __declspec(dllexport) TYPE& View_##TYPE_NAME##_##EXECUTION_SPACE##_1D::operator()(const int32 i0) const                                                                        \
    {                                                                                                                                                                              \
        return view(i0);                                                                                                                                                           \
    }                                                                                                                                                                              \
    __declspec(dllexport) NativeString View_##TYPE_NAME##_##EXECUTION_SPACE##_1D::GetLabel() const                                                                                 \
    {                                                                                                                                                                              \
        return NativeString(view.label());                                                                                                                                         \
    }                                                                                                                                                                              \
    __declspec(dllexport) KOKKOS_FUNCTION uint64 View_##TYPE_NAME##_##EXECUTION_SPACE##_1D::GetSize() const                                                                        \
    {                                                                                                                                                                              \
        return view.size();                                                                                                                                                        \
    }                                                                                                                                                                              \
    __declspec(dllexport) KOKKOS_FUNCTION uint32 View_##TYPE_NAME##_##EXECUTION_SPACE##_1D::GetRank() const                                                                        \
    {                                                                                                                                                                              \
        return ViewType::Rank;                                                                                                                                                     \
    }                                                                                                                                                                              \
    __declspec(dllexport) KOKKOS_FUNCTION uint64 View_##TYPE_NAME##_##EXECUTION_SPACE##_1D::GetStride(uint32 dim) const                                                            \
    {                                                                                                                                                                              \
        return view.stride(dim);                                                                                                                                                   \
    }                                                                                                                                                                              \
    __declspec(dllexport) KOKKOS_FUNCTION uint64 View_##TYPE_NAME##_##EXECUTION_SPACE##_1D::GetExtent(uint32 dim) const                                                            \
    {                                                                                                                                                                              \
        return view.extent(dim);                                                                                                                                                   \
    }                                                                                                                                                                              \
    class __declspec(dllexport) View_##TYPE_NAME##_##EXECUTION_SPACE##_2D : IView<2, TYPE, Kokkos::EXECUTION_SPACE>                                                                \
    {                                                                                                                                                                              \
        using ViewType = Kokkos::View<TYPE**, Kokkos::EXECUTION_SPACE::array_layout, Kokkos::EXECUTION_SPACE>;                                                                     \
                                                                                                                                                                                   \
        ViewType view;                                                                                                                                                             \
                                                                                                                                                                                   \
    public:                                                                                                                                                                        \
        View_##TYPE_NAME##_##EXECUTION_SPACE##_2D(const NativeString arg_label, const size_t arg_N0, const size_t arg_N1);                                                         \
                                                                                                                                                                                   \
        ~View_##TYPE_NAME##_##EXECUTION_SPACE##_2D() = default;                                                                                                                    \
                                                                                                                                                                                   \
        TYPE&                  operator()(const int32 i0, const int32 i1) const override;                                                                                          \
        NativeString           GetLabel() const override;                                                                                                                          \
        KOKKOS_FUNCTION uint64 GetSize() const override;                                                                                                                           \
        KOKKOS_FUNCTION uint32 GetRank() const override;                                                                                                                           \
        KOKKOS_FUNCTION uint64 GetStride(uint32 dim) const override;                                                                                                               \
        KOKKOS_FUNCTION uint64 GetExtent(uint32 dim) const override;                                                                                                               \
    };                                                                                                                                                                             \
                                                                                                                                                                                   \
    __declspec(dllexport)                                                                                                                                                          \
        View_##TYPE_NAME##_##EXECUTION_SPACE##_2D::View_##TYPE_NAME##_##EXECUTION_SPACE##_2D(const NativeString arg_label, const size_t arg_N0, const size_t arg_N1) :             \
        view(arg_label.ToString(), arg_N0, arg_N1)                                                                                                                                 \
    {                                                                                                                                                                              \
    }                                                                                                                                                                              \
    __declspec(dllexport) TYPE& View_##TYPE_NAME##_##EXECUTION_SPACE##_2D::operator()(const int32 i0, const int32 i1) const                                                        \
    {                                                                                                                                                                              \
        return view(i0, i1);                                                                                                                                                       \
    }                                                                                                                                                                              \
    __declspec(dllexport) NativeString View_##TYPE_NAME##_##EXECUTION_SPACE##_2D::GetLabel() const                                                                                 \
    {                                                                                                                                                                              \
        return NativeString(view.label());                                                                                                                                         \
    }                                                                                                                                                                              \
    __declspec(dllexport) KOKKOS_FUNCTION uint64 View_##TYPE_NAME##_##EXECUTION_SPACE##_2D::GetSize() const                                                                        \
    {                                                                                                                                                                              \
        return view.size();                                                                                                                                                        \
    }                                                                                                                                                                              \
    __declspec(dllexport) KOKKOS_FUNCTION uint32 View_##TYPE_NAME##_##EXECUTION_SPACE##_2D::GetRank() const                                                                        \
    {                                                                                                                                                                              \
        return ViewType::Rank;                                                                                                                                                     \
    }                                                                                                                                                                              \
    __declspec(dllexport) KOKKOS_FUNCTION uint64 View_##TYPE_NAME##_##EXECUTION_SPACE##_2D::GetStride(uint32 dim) const                                                            \
    {                                                                                                                                                                              \
        return view.stride(dim);                                                                                                                                                   \
    }                                                                                                                                                                              \
    __declspec(dllexport) KOKKOS_FUNCTION uint64 View_##TYPE_NAME##_##EXECUTION_SPACE##_2D::GetExtent(uint32 dim) const                                                            \
    {                                                                                                                                                                              \
        return view.extent(dim);                                                                                                                                                   \
    }                                                                                                                                                                              \
    class __declspec(dllexport) View_##TYPE_NAME##_##EXECUTION_SPACE##_3D : IView<3, TYPE, Kokkos::EXECUTION_SPACE>                                                                \
    {                                                                                                                                                                              \
        using ViewType = Kokkos::View<TYPE***, Kokkos::EXECUTION_SPACE::array_layout, Kokkos::EXECUTION_SPACE>;                                                                    \
                                                                                                                                                                                   \
        ViewType view;                                                                                                                                                             \
                                                                                                                                                                                   \
    public:                                                                                                                                                                        \
        View_##TYPE_NAME##_##EXECUTION_SPACE##_3D(const NativeString arg_label, const size_t arg_N0, const size_t arg_N1, const size_t arg_N2);                                    \
                                                                                                                                                                                   \
        ~View_##TYPE_NAME##_##EXECUTION_SPACE##_3D() = default;                                                                                                                    \
                                                                                                                                                                                   \
        TYPE&                  operator()(const int32 i0, const int32 i1, const int32 i2) const override;                                                                          \
        NativeString           GetLabel() const override;                                                                                                                          \
        KOKKOS_FUNCTION uint64 GetSize() const override;                                                                                                                           \
        KOKKOS_FUNCTION uint32 GetRank() const override;                                                                                                                           \
        KOKKOS_FUNCTION uint64 GetStride(uint32 dim) const override;                                                                                                               \
        KOKKOS_FUNCTION uint64 GetExtent(uint32 dim) const override;                                                                                                               \
    };                                                                                                                                                                             \
                                                                                                                                                                                   \
    __declspec(dllexport) View_##TYPE_NAME##_##EXECUTION_SPACE##_3D::View_##TYPE_NAME##_##EXECUTION_SPACE##_3D(const NativeString arg_label,                                       \
                                                                                                               const size_t       arg_N0,                                          \
                                                                                                               const size_t       arg_N1,                                          \
                                                                                                               const size_t       arg_N2) :                                              \
        view(arg_label.ToString(), arg_N0, arg_N1, arg_N2)                                                                                                                         \
    {                                                                                                                                                                              \
    }                                                                                                                                                                              \
    __declspec(dllexport) TYPE& View_##TYPE_NAME##_##EXECUTION_SPACE##_3D::operator()(const int32 i0, const int32 i1, const int32 i2) const                                        \
    {                                                                                                                                                                              \
        return view(i0, i1, i2);                                                                                                                                                   \
    }                                                                                                                                                                              \
    __declspec(dllexport) NativeString View_##TYPE_NAME##_##EXECUTION_SPACE##_3D::GetLabel() const                                                                                 \
    {                                                                                                                                                                              \
        return NativeString(view.label());                                                                                                                                         \
    }                                                                                                                                                                              \
    __declspec(dllexport) KOKKOS_FUNCTION uint64 View_##TYPE_NAME##_##EXECUTION_SPACE##_3D::GetSize() const                                                                        \
    {                                                                                                                                                                              \
        return view.size();                                                                                                                                                        \
    }                                                                                                                                                                              \
    __declspec(dllexport) KOKKOS_FUNCTION uint32 View_##TYPE_NAME##_##EXECUTION_SPACE##_3D::GetRank() const                                                                        \
    {                                                                                                                                                                              \
        return ViewType::Rank;                                                                                                                                                     \
    }                                                                                                                                                                              \
    __declspec(dllexport) KOKKOS_FUNCTION uint64 View_##TYPE_NAME##_##EXECUTION_SPACE##_3D::GetStride(uint32 dim) const                                                            \
    {                                                                                                                                                                              \
        return view.stride(dim);                                                                                                                                                   \
    }                                                                                                                                                                              \
    __declspec(dllexport) KOKKOS_FUNCTION uint64 View_##TYPE_NAME##_##EXECUTION_SPACE##_3D::GetExtent(uint32 dim) const                                                            \
    {                                                                                                                                                                              \
        return view.extent(dim);                                                                                                                                                   \
    }                                                                                                                                                                              \
    class __declspec(dllexport) View_##TYPE_NAME##_##EXECUTION_SPACE##_4D : IView<4, TYPE, Kokkos::EXECUTION_SPACE>                                                                \
    {                                                                                                                                                                              \
        using ViewType = Kokkos::View<TYPE****, Kokkos::EXECUTION_SPACE::array_layout, Kokkos::EXECUTION_SPACE>;                                                                   \
                                                                                                                                                                                   \
        ViewType view;                                                                                                                                                             \
                                                                                                                                                                                   \
    public:                                                                                                                                                                        \
        View_##TYPE_NAME##_##EXECUTION_SPACE##_4D(const NativeString arg_label, const size_t arg_N0, const size_t arg_N1, const size_t arg_N2, const size_t arg_N3);               \
                                                                                                                                                                                   \
        ~View_##TYPE_NAME##_##EXECUTION_SPACE##_4D() = default;                                                                                                                    \
                                                                                                                                                                                   \
        TYPE&                  operator()(const int32 i0, const int32 i1, const int32 i2, const int32 i3) const override;                                                          \
        NativeString           GetLabel() const override;                                                                                                                          \
        KOKKOS_FUNCTION uint64 GetSize() const override;                                                                                                                           \
        KOKKOS_FUNCTION uint32 GetRank() const override;                                                                                                                           \
        KOKKOS_FUNCTION uint64 GetStride(uint32 dim) const override;                                                                                                               \
        KOKKOS_FUNCTION uint64 GetExtent(uint32 dim) const override;                                                                                                               \
    };                                                                                                                                                                             \
                                                                                                                                                                                   \
    __declspec(dllexport) View_##TYPE_NAME##_##EXECUTION_SPACE##_4D::View_##TYPE_NAME##_##EXECUTION_SPACE##_4D(const NativeString arg_label,                                       \
                                                                                                               const size_t       arg_N0,                                          \
                                                                                                               const size_t       arg_N1,                                          \
                                                                                                               const size_t       arg_N2,                                          \
                                                                                                               const size_t       arg_N3) :                                              \
        view(arg_label.ToString(), arg_N0, arg_N1, arg_N2, arg_N3)                                                                                                                 \
    {                                                                                                                                                                              \
    }                                                                                                                                                                              \
    __declspec(dllexport) TYPE& View_##TYPE_NAME##_##EXECUTION_SPACE##_4D::operator()(const int32 i0, const int32 i1, const int32 i2, const int32 i3) const                        \
    {                                                                                                                                                                              \
        return view(i0, i1, i2, i3);                                                                                                                                               \
    }                                                                                                                                                                              \
    __declspec(dllexport) NativeString View_##TYPE_NAME##_##EXECUTION_SPACE##_4D::GetLabel() const                                                                                 \
    {                                                                                                                                                                              \
        return NativeString(view.label());                                                                                                                                         \
    }                                                                                                                                                                              \
    __declspec(dllexport) KOKKOS_FUNCTION uint64 View_##TYPE_NAME##_##EXECUTION_SPACE##_4D::GetSize() const                                                                        \
    {                                                                                                                                                                              \
        return view.size();                                                                                                                                                        \
    }                                                                                                                                                                              \
    __declspec(dllexport) KOKKOS_FUNCTION uint32 View_##TYPE_NAME##_##EXECUTION_SPACE##_4D::GetRank() const                                                                        \
    {                                                                                                                                                                              \
        return ViewType::Rank;                                                                                                                                                     \
    }                                                                                                                                                                              \
    __declspec(dllexport) KOKKOS_FUNCTION uint64 View_##TYPE_NAME##_##EXECUTION_SPACE##_4D::GetStride(uint32 dim) const                                                            \
    {                                                                                                                                                                              \
        return view.stride(dim);                                                                                                                                                   \
    }                                                                                                                                                                              \
    __declspec(dllexport) KOKKOS_FUNCTION uint64 View_##TYPE_NAME##_##EXECUTION_SPACE##_4D::GetExtent(uint32 dim) const                                                            \
    {                                                                                                                                                                              \
        return view.extent(dim);                                                                                                                                                   \
    }                                                                                                                                                                              \
    class __declspec(dllexport) View_##TYPE_NAME##_##EXECUTION_SPACE##_5D : IView<5, TYPE, Kokkos::EXECUTION_SPACE>                                                                \
    {                                                                                                                                                                              \
        using ViewType = Kokkos::View<TYPE*****, Kokkos::EXECUTION_SPACE::array_layout, Kokkos::EXECUTION_SPACE>;                                                                  \
                                                                                                                                                                                   \
        ViewType view;                                                                                                                                                             \
                                                                                                                                                                                   \
    public:                                                                                                                                                                        \
        View_##TYPE_NAME##_##EXECUTION_SPACE##_5D(const NativeString arg_label,                                                                                                    \
                                                  const size_t       arg_N0,                                                                                                       \
                                                  const size_t       arg_N1,                                                                                                       \
                                                  const size_t       arg_N2,                                                                                                       \
                                                  const size_t       arg_N3,                                                                                                       \
                                                  const size_t       arg_N4);                                                                                                            \
                                                                                                                                                                                   \
        ~View_##TYPE_NAME##_##EXECUTION_SPACE##_5D() = default;                                                                                                                    \
                                                                                                                                                                                   \
        TYPE&                  operator()(const int32 i0, const int32 i1, const int32 i2, const int32 i3, const int32 i4) const override;                                          \
        NativeString           GetLabel() const override;                                                                                                                          \
        KOKKOS_FUNCTION uint64 GetSize() const override;                                                                                                                           \
        KOKKOS_FUNCTION uint32 GetRank() const override;                                                                                                                           \
        KOKKOS_FUNCTION uint64 GetStride(uint32 dim) const override;                                                                                                               \
        KOKKOS_FUNCTION uint64 GetExtent(uint32 dim) const override;                                                                                                               \
    };                                                                                                                                                                             \
                                                                                                                                                                                   \
    __declspec(dllexport) View_##TYPE_NAME##_##EXECUTION_SPACE##_5D::View_##TYPE_NAME##_##EXECUTION_SPACE##_5D(const NativeString arg_label,                                       \
                                                                                                               const size_t       arg_N0,                                          \
                                                                                                               const size_t       arg_N1,                                          \
                                                                                                               const size_t       arg_N2,                                          \
                                                                                                               const size_t       arg_N3,                                          \
                                                                                                               const size_t       arg_N4) :                                              \
        view(arg_label.ToString(), arg_N0, arg_N1, arg_N2, arg_N3, arg_N4)                                                                                                         \
    {                                                                                                                                                                              \
    }                                                                                                                                                                              \
    __declspec(dllexport) TYPE& View_##TYPE_NAME##_##EXECUTION_SPACE##_5D::operator()(const int32 i0, const int32 i1, const int32 i2, const int32 i3, const int32 i4) const        \
    {                                                                                                                                                                              \
        return view(i0, i1, i2, i3, i4);                                                                                                                                           \
    }                                                                                                                                                                              \
    __declspec(dllexport) NativeString View_##TYPE_NAME##_##EXECUTION_SPACE##_5D::GetLabel() const                                                                                 \
    {                                                                                                                                                                              \
        return NativeString(view.label());                                                                                                                                         \
    }                                                                                                                                                                              \
    __declspec(dllexport) KOKKOS_FUNCTION uint64 View_##TYPE_NAME##_##EXECUTION_SPACE##_5D::GetSize() const                                                                        \
    {                                                                                                                                                                              \
        return view.size();                                                                                                                                                        \
    }                                                                                                                                                                              \
    __declspec(dllexport) KOKKOS_FUNCTION uint32 View_##TYPE_NAME##_##EXECUTION_SPACE##_5D::GetRank() const                                                                        \
    {                                                                                                                                                                              \
        return ViewType::Rank;                                                                                                                                                     \
    }                                                                                                                                                                              \
    __declspec(dllexport) KOKKOS_FUNCTION uint64 View_##TYPE_NAME##_##EXECUTION_SPACE##_5D::GetStride(uint32 dim) const                                                            \
    {                                                                                                                                                                              \
        return view.stride(dim);                                                                                                                                                   \
    }                                                                                                                                                                              \
    __declspec(dllexport) KOKKOS_FUNCTION uint64 View_##TYPE_NAME##_##EXECUTION_SPACE##_5D::GetExtent(uint32 dim) const                                                            \
    {                                                                                                                                                                              \
        return view.extent(dim);                                                                                                                                                   \
    }                                                                                                                                                                              \
    class __declspec(dllexport) View_##TYPE_NAME##_##EXECUTION_SPACE##_6D : IView<6, TYPE, Kokkos::EXECUTION_SPACE>                                                                \
    {                                                                                                                                                                              \
        using ViewType = Kokkos::View<TYPE******, Kokkos::EXECUTION_SPACE::array_layout, Kokkos::EXECUTION_SPACE>;                                                                 \
                                                                                                                                                                                   \
        ViewType view;                                                                                                                                                             \
                                                                                                                                                                                   \
    public:                                                                                                                                                                        \
        View_##TYPE_NAME##_##EXECUTION_SPACE##_6D(const NativeString arg_label,                                                                                                    \
                                                  const size_t       arg_N0,                                                                                                       \
                                                  const size_t       arg_N1,                                                                                                       \
                                                  const size_t       arg_N2,                                                                                                       \
                                                  const size_t       arg_N3,                                                                                                       \
                                                  const size_t       arg_N4,                                                                                                       \
                                                  const size_t       arg_N5);                                                                                                            \
                                                                                                                                                                                   \
        ~View_##TYPE_NAME##_##EXECUTION_SPACE##_6D() = default;                                                                                                                    \
                                                                                                                                                                                   \
        TYPE&                  operator()(const int32 i0, const int32 i1, const int32 i2, const int32 i3, const int32 i4, const int32 i5) const override;                          \
        NativeString           GetLabel() const override;                                                                                                                          \
        KOKKOS_FUNCTION uint64 GetSize() const override;                                                                                                                           \
        KOKKOS_FUNCTION uint32 GetRank() const override;                                                                                                                           \
        KOKKOS_FUNCTION uint64 GetStride(uint32 dim) const override;                                                                                                               \
        KOKKOS_FUNCTION uint64 GetExtent(uint32 dim) const override;                                                                                                               \
    };                                                                                                                                                                             \
                                                                                                                                                                                   \
    __declspec(dllexport) View_##TYPE_NAME##_##EXECUTION_SPACE##_6D::View_##TYPE_NAME##_##EXECUTION_SPACE##_6D(const NativeString arg_label,                                       \
                                                                                                               const size_t       arg_N0,                                          \
                                                                                                               const size_t       arg_N1,                                          \
                                                                                                               const size_t       arg_N2,                                          \
                                                                                                               const size_t       arg_N3,                                          \
                                                                                                               const size_t       arg_N4,                                          \
                                                                                                               const size_t       arg_N5) :                                              \
        view(arg_label.ToString(), arg_N0, arg_N1, arg_N2, arg_N3, arg_N4, arg_N5)                                                                                                 \
    {                                                                                                                                                                              \
    }                                                                                                                                                                              \
    __declspec(dllexport)                                                                                                                                                          \
        TYPE& View_##TYPE_NAME##_##EXECUTION_SPACE##_6D::operator()(const int32 i0, const int32 i1, const int32 i2, const int32 i3, const int32 i4, const int32 i5) const          \
    {                                                                                                                                                                              \
        return view(i0, i1, i2, i3, i4, i5);                                                                                                                                       \
    }                                                                                                                                                                              \
    __declspec(dllexport) NativeString View_##TYPE_NAME##_##EXECUTION_SPACE##_6D::GetLabel() const                                                                                 \
    {                                                                                                                                                                              \
        return NativeString(view.label());                                                                                                                                         \
    }                                                                                                                                                                              \
    __declspec(dllexport) KOKKOS_FUNCTION uint64 View_##TYPE_NAME##_##EXECUTION_SPACE##_6D::GetSize() const                                                                        \
    {                                                                                                                                                                              \
        return view.size();                                                                                                                                                        \
    }                                                                                                                                                                              \
    __declspec(dllexport) KOKKOS_FUNCTION uint32 View_##TYPE_NAME##_##EXECUTION_SPACE##_6D::GetRank() const                                                                        \
    {                                                                                                                                                                              \
        return ViewType::Rank;                                                                                                                                                     \
    }                                                                                                                                                                              \
    __declspec(dllexport) KOKKOS_FUNCTION uint64 View_##TYPE_NAME##_##EXECUTION_SPACE##_6D::GetStride(uint32 dim) const                                                            \
    {                                                                                                                                                                              \
        return view.stride(dim);                                                                                                                                                   \
    }                                                                                                                                                                              \
    __declspec(dllexport) KOKKOS_FUNCTION uint64 View_##TYPE_NAME##_##EXECUTION_SPACE##_6D::GetExtent(uint32 dim) const                                                            \
    {                                                                                                                                                                              \
        return view.extent(dim);                                                                                                                                                   \
    }                                                                                                                                                                              \
    class __declspec(dllexport) View_##TYPE_NAME##_##EXECUTION_SPACE##_7D : IView<7, TYPE, Kokkos::EXECUTION_SPACE>                                                                \
    {                                                                                                                                                                              \
        using ViewType = Kokkos::View<TYPE*******, Kokkos::EXECUTION_SPACE::array_layout, Kokkos::EXECUTION_SPACE>;                                                                \
                                                                                                                                                                                   \
        ViewType view;                                                                                                                                                             \
                                                                                                                                                                                   \
    public:                                                                                                                                                                        \
        View_##TYPE_NAME##_##EXECUTION_SPACE##_7D(const NativeString arg_label,                                                                                                    \
                                                  const size_t       arg_N0,                                                                                                       \
                                                  const size_t       arg_N1,                                                                                                       \
                                                  const size_t       arg_N2,                                                                                                       \
                                                  const size_t       arg_N3,                                                                                                       \
                                                  const size_t       arg_N4,                                                                                                       \
                                                  const size_t       arg_N5,                                                                                                       \
                                                  const size_t       arg_N6);                                                                                                            \
                                                                                                                                                                                   \
        ~View_##TYPE_NAME##_##EXECUTION_SPACE##_7D() = default;                                                                                                                    \
                                                                                                                                                                                   \
        TYPE&                  operator()(const int32 i0, const int32 i1, const int32 i2, const int32 i3, const int32 i4, const int32 i5, const int32 i6) const override;          \
        NativeString           GetLabel() const override;                                                                                                                          \
        KOKKOS_FUNCTION uint64 GetSize() const override;                                                                                                                           \
        KOKKOS_FUNCTION uint32 GetRank() const override;                                                                                                                           \
        KOKKOS_FUNCTION uint64 GetStride(uint32 dim) const override;                                                                                                               \
        KOKKOS_FUNCTION uint64 GetExtent(uint32 dim) const override;                                                                                                               \
    };                                                                                                                                                                             \
                                                                                                                                                                                   \
    __declspec(dllexport) View_##TYPE_NAME##_##EXECUTION_SPACE##_7D::View_##TYPE_NAME##_##EXECUTION_SPACE##_7D(const NativeString arg_label,                                       \
                                                                                                               const size_t       arg_N0,                                          \
                                                                                                               const size_t       arg_N1,                                          \
                                                                                                               const size_t       arg_N2,                                          \
                                                                                                               const size_t       arg_N3,                                          \
                                                                                                               const size_t       arg_N4,                                          \
                                                                                                               const size_t       arg_N5,                                          \
                                                                                                               const size_t       arg_N6) :                                              \
        view(arg_label.ToString(), arg_N0, arg_N1, arg_N2, arg_N3, arg_N4, arg_N5, arg_N6)                                                                                         \
    {                                                                                                                                                                              \
    }                                                                                                                                                                              \
    __declspec(dllexport) TYPE&                                                                                                                                                    \
        View_##TYPE_NAME##_##EXECUTION_SPACE##_7D::operator()(const int32 i0, const int32 i1, const int32 i2, const int32 i3, const int32 i4, const int32 i5, const int32 i6)      \
            const                                                                                                                                                                  \
    {                                                                                                                                                                              \
        return view(i0, i1, i2, i3, i4, i5, i6);                                                                                                                                   \
    }                                                                                                                                                                              \
    __declspec(dllexport) NativeString View_##TYPE_NAME##_##EXECUTION_SPACE##_7D::GetLabel() const                                                                                 \
    {                                                                                                                                                                              \
        return NativeString(view.label());                                                                                                                                         \
    }                                                                                                                                                                              \
    __declspec(dllexport) KOKKOS_FUNCTION uint64 View_##TYPE_NAME##_##EXECUTION_SPACE##_7D::GetSize() const                                                                        \
    {                                                                                                                                                                              \
        return view.size();                                                                                                                                                        \
    }                                                                                                                                                                              \
    __declspec(dllexport) KOKKOS_FUNCTION uint32 View_##TYPE_NAME##_##EXECUTION_SPACE##_7D::GetRank() const                                                                        \
    {                                                                                                                                                                              \
        return ViewType::Rank;                                                                                                                                                     \
    }                                                                                                                                                                              \
    __declspec(dllexport) KOKKOS_FUNCTION uint64 View_##TYPE_NAME##_##EXECUTION_SPACE##_7D::GetStride(uint32 dim) const                                                            \
    {                                                                                                                                                                              \
        return view.stride(dim);                                                                                                                                                   \
    }                                                                                                                                                                              \
    __declspec(dllexport) KOKKOS_FUNCTION uint64 View_##TYPE_NAME##_##EXECUTION_SPACE##_7D::GetExtent(uint32 dim) const                                                            \
    {                                                                                                                                                                              \
        return view.extent(dim);                                                                                                                                                   \
    }                                                                                                                                                                              \
    class __declspec(dllexport) View_##TYPE_NAME##_##EXECUTION_SPACE##_8D : IView<8, TYPE, Kokkos::EXECUTION_SPACE>                                                                \
    {                                                                                                                                                                              \
        using ViewType = Kokkos::View<TYPE********, Kokkos::EXECUTION_SPACE::array_layout, Kokkos::EXECUTION_SPACE>;                                                               \
                                                                                                                                                                                   \
        ViewType view;                                                                                                                                                             \
                                                                                                                                                                                   \
    public:                                                                                                                                                                        \
        View_##TYPE_NAME##_##EXECUTION_SPACE##_8D(const NativeString arg_label,                                                                                                    \
                                                  const size_t       arg_N0,                                                                                                       \
                                                  const size_t       arg_N1,                                                                                                       \
                                                  const size_t       arg_N2,                                                                                                       \
                                                  const size_t       arg_N3,                                                                                                       \
                                                  const size_t       arg_N4,                                                                                                       \
                                                  const size_t       arg_N5,                                                                                                       \
                                                  const size_t       arg_N6,                                                                                                       \
                                                  const size_t       arg_N7);                                                                                                            \
                                                                                                                                                                                   \
        ~View_##TYPE_NAME##_##EXECUTION_SPACE##_8D() = default;                                                                                                                    \
                                                                                                                                                                                   \
        TYPE&        operator()(const int32 i0, const int32 i1, const int32 i2, const int32 i3, const int32 i4, const int32 i5, const int32 i6, const int32 i7) const override;    \
        NativeString GetLabel() const override;                                                                                                                                    \
        KOKKOS_FUNCTION uint64 GetSize() const override;                                                                                                                           \
        KOKKOS_FUNCTION uint32 GetRank() const override;                                                                                                                           \
        KOKKOS_FUNCTION uint64 GetStride(uint32 dim) const override;                                                                                                               \
        KOKKOS_FUNCTION uint64 GetExtent(uint32 dim) const override;                                                                                                               \
    };                                                                                                                                                                             \
                                                                                                                                                                                   \
    __declspec(dllexport) View_##TYPE_NAME##_##EXECUTION_SPACE##_8D::View_##TYPE_NAME##_##EXECUTION_SPACE##_8D(const NativeString arg_label,                                       \
                                                                                                               const size_t       arg_N0,                                          \
                                                                                                               const size_t       arg_N1,                                          \
                                                                                                               const size_t       arg_N2,                                          \
                                                                                                               const size_t       arg_N3,                                          \
                                                                                                               const size_t       arg_N4,                                          \
                                                                                                               const size_t       arg_N5,                                          \
                                                                                                               const size_t       arg_N6,                                          \
                                                                                                               const size_t       arg_N7) :                                              \
        view(arg_label.ToString(), arg_N0, arg_N1, arg_N2, arg_N3, arg_N4, arg_N5, arg_N6, arg_N7)                                                                                 \
    {                                                                                                                                                                              \
    }                                                                                                                                                                              \
    __declspec(dllexport) TYPE& View_##TYPE_NAME##_##EXECUTION_SPACE##_8D::operator()(const int32 i0,                                                                              \
                                                                                      const int32 i1,                                                                              \
                                                                                      const int32 i2,                                                                              \
                                                                                      const int32 i3,                                                                              \
                                                                                      const int32 i4,                                                                              \
                                                                                      const int32 i5,                                                                              \
                                                                                      const int32 i6,                                                                              \
                                                                                      const int32 i7) const                                                                        \
    {                                                                                                                                                                              \
        return view(i0, i1, i2, i3, i4, i5, i6, i7);                                                                                                                               \
    }                                                                                                                                                                              \
    __declspec(dllexport) NativeString View_##TYPE_NAME##_##EXECUTION_SPACE##_8D::GetLabel() const                                                                                 \
    {                                                                                                                                                                              \
        return NativeString(view.label());                                                                                                                                         \
    }                                                                                                                                                                              \
    __declspec(dllexport) KOKKOS_FUNCTION uint64 View_##TYPE_NAME##_##EXECUTION_SPACE##_8D::GetSize() const                                                                        \
    {                                                                                                                                                                              \
        return view.size();                                                                                                                                                        \
    }                                                                                                                                                                              \
    __declspec(dllexport) KOKKOS_FUNCTION uint32 View_##TYPE_NAME##_##EXECUTION_SPACE##_8D::GetRank() const                                                                        \
    {                                                                                                                                                                              \
        return ViewType::Rank;                                                                                                                                                     \
    }                                                                                                                                                                              \
    __declspec(dllexport) KOKKOS_FUNCTION uint64 View_##TYPE_NAME##_##EXECUTION_SPACE##_8D::GetStride(uint32 dim) const                                                            \
    {                                                                                                                                                                              \
        return view.stride(dim);                                                                                                                                                   \
    }                                                                                                                                                                              \
    __declspec(dllexport) KOKKOS_FUNCTION uint64 View_##TYPE_NAME##_##EXECUTION_SPACE##_8D::GetExtent(uint32 dim) const                                                            \
    {                                                                                                                                                                              \
        return view.extent(dim);                                                                                                                                                   \
    }

TEMPLATE(DEF_TEMPLATE, Serial)
TEMPLATE(DEF_TEMPLATE, OpenMP)
TEMPLATE(DEF_TEMPLATE, Cuda)

#undef TEMPLATE
#undef DEF_TEMPLATE
