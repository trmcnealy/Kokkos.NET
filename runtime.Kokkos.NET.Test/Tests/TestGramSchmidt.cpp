
#include "Tests.hpp"

namespace Kokkos
{
    template<class Type>
    struct Dot
    {
        using execution_space = typename Type::execution_space;

        static_assert(static_cast<unsigned>(Type::Rank) == static_cast<unsigned>(1), "Dot static_assert Fail: Rank != 1");

        using value_type = double;

#if 1
        typename Type::const_type X;
        typename Type::const_type Y;
#else
        Type            X;
        Type            Y;
#endif

        Dot(const Type& arg_x, const Type& arg_y) : X(arg_x), Y(arg_y) {}

        KOKKOS_INLINE_FUNCTION void operator()(int i, value_type& update) const
        {
            update += X[i] * Y[i];
        }

        KOKKOS_INLINE_FUNCTION static void join(volatile value_type& update, const volatile value_type& source)
        {
            update += source;
        }

        KOKKOS_INLINE_FUNCTION static void init(value_type& update)
        {
            update = 0;
        }
    };

    template<class Type>
    struct DotSingle
    {
        using execution_space = typename Type::execution_space;

        static_assert(static_cast<unsigned>(Type::Rank) == static_cast<unsigned>(1), "DotSingle static_assert Fail: Rank != 1");

        using value_type = double;

#if 1
        typename Type::const_type X;
#else
        Type            X;
#endif

        DotSingle(const Type& arg_x) : X(arg_x) {}

        KOKKOS_INLINE_FUNCTION void operator()(int i, value_type& update) const
        {
            const typename Type::value_type& x = X[i];
            update += x * x;
        }

        KOKKOS_INLINE_FUNCTION static void join(volatile value_type& update, const volatile value_type& source)
        {
            update += source;
        }

        KOKKOS_INLINE_FUNCTION static void init(value_type& update)
        {
            update = 0;
        }
    };

    template<class ScalarType, class VectorType>
    struct Scale
    {
        using execution_space = typename VectorType::execution_space;

        static_assert(static_cast<unsigned>(ScalarType::Rank) == static_cast<unsigned>(0), "Scale static_assert Fail: ScalarType::Rank != 0");

        static_assert(static_cast<unsigned>(VectorType::Rank) == static_cast<unsigned>(1), "Scale static_assert Fail: VectorType::Rank != 1");

#if 1
        typename ScalarType::const_type alpha;
#else
        ScalarType      alpha;
#endif

        VectorType Y;

        Scale(const ScalarType& arg_alpha, const VectorType& arg_Y) : alpha(arg_alpha), Y(arg_Y) {}

        KOKKOS_INLINE_FUNCTION void operator()(int i) const
        {
            Y[i] *= alpha();
        }
    };

    template<class ScalarType, class ConstVectorType, class VectorType>
    struct AXPBY
    {
        using execution_space = typename VectorType::execution_space;

        static_assert(static_cast<unsigned>(ScalarType::Rank) == static_cast<unsigned>(0), "AXPBY static_assert Fail: ScalarType::Rank != 0");

        static_assert(static_cast<unsigned>(ConstVectorType::Rank) == static_cast<unsigned>(1), "AXPBY static_assert Fail: ConstVectorType::Rank != 1");

        static_assert(static_cast<unsigned>(VectorType::Rank) == static_cast<unsigned>(1), "AXPBY static_assert Fail: VectorType::Rank != 1");

#if 1
        typename ScalarType::const_type      alpha, beta;
        typename ConstVectorType::const_type X;
#else
        ScalarType      alpha, beta;
        ConstVectorType X;
#endif

        VectorType Y;

        AXPBY(const ScalarType& arg_alpha, const ConstVectorType& arg_X, const ScalarType& arg_beta, const VectorType& arg_Y) : alpha(arg_alpha), beta(arg_beta), X(arg_X), Y(arg_Y) {}

        KOKKOS_INLINE_FUNCTION void operator()(int i) const
        {
            Y[i] = alpha() * X[i] + beta() * Y[i];
        }
    };

    template<class ConstScalarType, class ConstVectorType, class VectorType>
    __inline static void axpby(const ConstScalarType& alpha, const ConstVectorType& X, const ConstScalarType& beta, const VectorType& Y)
    {
        using functor = AXPBY<ConstScalarType, ConstVectorType, VectorType>;

        parallel_for(Y.extent(0), functor(alpha, X, beta, Y));
    }

    /** \brief  Y *= alpha */
    template<class ConstScalarType, class VectorType>
    __inline static void scale(const ConstScalarType& alpha, const VectorType& Y)
    {
        using functor = Scale<ConstScalarType, VectorType>;

        parallel_for(Y.extent(0), functor(alpha, Y));
    }

    template<class ConstVectorType, class Finalize>
    __inline static void dot(const ConstVectorType& X, const ConstVectorType& Y, const Finalize& finalize)
    {
        using functor = Dot<ConstVectorType>;

        parallel_reduce(X.extent(0), functor(X, Y), finalize);
    }

    template<class ConstVectorType, class Finalize>
    __inline static void dot(const ConstVectorType& X, const Finalize& finalize)
    {
        using functor = DotSingle<ConstVectorType>;

        parallel_reduce(X.extent(0), functor(X), finalize);
    }
}

// Reduction   : result = dot( Q(:,j) , Q(:,j) );
// PostProcess : R(j,j) = result ; inv = 1 / result ;
template<class VectorView, class ValueView>
struct InvNorm2 : public Kokkos::DotSingle<VectorView>
{
    using value_type = typename Kokkos::DotSingle<VectorView>::value_type;

    ValueView Rjj;
    ValueView inv;

    InvNorm2(const VectorView& argX, const ValueView& argR, const ValueView& argInv) : Kokkos::DotSingle<VectorView>(argX), Rjj(argR), inv(argInv) {}

    KOKKOS_INLINE_FUNCTION void final(value_type& result) const
    {
        result = Kokkos::Experimental::sqrt(result);
        Rjj()  = result;
        inv()  = (0 < result) ? 1.0 / result : 0;
    }
};

template<class VectorView, class ValueView>
inline static void invnorm2(const VectorView& x, const ValueView& r, const ValueView& r_inv)
{
    Kokkos::parallel_reduce(Kokkos::RangePolicy<typename VectorView::traits::execution_space>(0, x.extent(0)), InvNorm2<VectorView, ValueView>(x, r, r_inv));
}

// PostProcess : tmp = - ( R(j,k) = result );
template<class VectorView, class ValueView>
struct DotM : public Kokkos::Dot<VectorView>
{
    using value_type = typename Kokkos::Dot<VectorView>::value_type;

    ValueView Rjk;
    ValueView tmp;

    DotM(const VectorView& argX, const VectorView& argY, const ValueView& argR, const ValueView& argTmp) : Kokkos::Dot<VectorView>(argX, argY), Rjk(argR), tmp(argTmp) {}

    KOKKOS_INLINE_FUNCTION void final(value_type& result) const
    {
        Rjk() = result;
        tmp() = -result;
    }
};

template<class VectorView, class ValueView>
inline static void dot_neg(const VectorView& x, const VectorView& y, const ValueView& r, const ValueView& r_neg)
{
    Kokkos::parallel_reduce(Kokkos::RangePolicy<typename VectorView::traits::execution_space>(0, x.extent(0)), DotM<VectorView, ValueView>(x, y, r, r_neg));
}

template<typename Scalar, class DeviceType>
struct ModifiedGramSchmidt
{
    using execution_space = DeviceType;
    using size_type       = typename execution_space::size_type;

    using multivector_type = Kokkos::View<Scalar**, Kokkos::LayoutLeft, execution_space>;

    using vector_type = Kokkos::View<Scalar*, Kokkos::LayoutLeft, execution_space>;

    using value_view = Kokkos::View<Scalar, Kokkos::LayoutLeft, execution_space>;

    multivector_type Q;
    multivector_type R;

    static double factorization(const multivector_type Q_, const multivector_type R_)
    {
        const size_type count = Q_.extent(1);
        value_view      tmp("tmp");
        value_view      one("one");

        Kokkos::deep_copy(one, (Scalar)1);

        Kokkos::Timer timer;

        for (size_type j = 0; j < count; ++j)
        {
            // Reduction   : tmp = dot( Q(:,j) , Q(:,j) );
            // PostProcess : tmp = std::sqrt( tmp ); R(j,j) = tmp ; tmp = 1 / tmp ;
            const vector_type Qj  = Kokkos::subview(Q_, Kokkos::ALL(), j);
            const value_view  Rjj = Kokkos::subview(R_, j, j);

            invnorm2(Qj, Rjj, tmp);

            // Q(:,j) *= ( 1 / R(j,j) ); => Q(:,j) *= tmp ;
            Kokkos::scale(tmp, Qj);

            for (size_type k = j + 1; k < count; ++k)
            {
                const vector_type Qk  = Kokkos::subview(Q_, Kokkos::ALL(), k);
                const value_view  Rjk = Kokkos::subview(R_, j, k);

                // Reduction   : R(j,k) = dot( Q(:,j) , Q(:,k) );
                // PostProcess : tmp = - R(j,k);
                dot_neg(Qj, Qk, Rjk, tmp);

                // Q(:,k) -= R(j,k) * Q(:,j); => Q(:,k) += tmp * Q(:,j)
                Kokkos::axpby(tmp, Qj, one, Qk);
            }
        }

        execution_space().fence();

        return timer.seconds();
    }

    //--------------------------------------------------------------------------

    static double test(const size_type length, const size_type count, const size_t iter = 1)
    {
        multivector_type Q_("Q", length, count);
        multivector_type R_("R", count, count);

        typename multivector_type::HostMirror A = Kokkos::create_mirror(Q_);

        // Create and fill A on the host

        for (size_type j = 0; j < count; ++j)
        {
            for (size_type i = 0; i < length; ++i)
            {
                A(i, j) = (i + 1) * (j + 1);
            }
        }

        double dt_min = 0;

        for (size_t i = 0; i < iter; ++i)
        {
            Kokkos::deep_copy(Q_, A);

            // A = Q * R

            const double dt = factorization(Q_, R_);

            if (0 == i)
                dt_min = dt;
            else
                dt_min = dt < dt_min ? dt : dt_min;
        }

        return dt_min;
    }
};

template<class DeviceType>
void run_test_gramschmidt(const int exp_beg, const int exp_end, const int num_trials, const char deviceTypeName[])
{
    std::string label_gramschmidt;
    label_gramschmidt.append("\"GramSchmidt< double , ");
    label_gramschmidt.append(deviceTypeName);
    label_gramschmidt.append(" >\"");

    for (int i = exp_beg; i < exp_end; ++i)
    {
        double min_seconds = 0.0;
        double max_seconds = 0.0;
        double avg_seconds = 0.0;

        const int parallel_work_length = 1 << i;

        for (int j = 0; j < num_trials; ++j)
        {
            const double seconds = ModifiedGramSchmidt<double, DeviceType>::test(parallel_work_length, 32);

            if (0 == j)
            {
                min_seconds = seconds;
                max_seconds = seconds;
            }
            else
            {
                if (seconds < min_seconds)
                    min_seconds = seconds;
                if (seconds > max_seconds)
                    max_seconds = seconds;
            }
            avg_seconds += seconds;
        }
        avg_seconds /= num_trials;

        std::cout << label_gramschmidt << " , " << parallel_work_length << " , " << min_seconds << " , " << (min_seconds / parallel_work_length) << std::endl;
    }
}

template<class ExecutionSpace>
static void TestGramSchmidt()
{
    const int exp_beg    = 10;
    const int exp_end    = 20;
    const int num_trials = 5;

    run_test_gramschmidt<ExecutionSpace>(exp_beg, exp_end, num_trials, ExecutionSpace::name());

    //const double seconds = ModifiedGramSchmidt<double, ExecutionSpace>::test(1 << exp_end, 32);
}
