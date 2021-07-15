#pragma once

#include "runtime.Kokkos/ViewTypes.hpp"
#include "runtime.Kokkos/Extensions.hpp"

#include <MathExtensions.hpp>

#include <KokkosBlas.hpp>

#include "StdExtensions.hpp"

//#include <Algebra/Eigenvalue.hpp>

namespace Geo
{
    template<typename DataType, class ExecutionSpace>
    KOKKOS_INLINE_FUNCTION static void rot(Kokkos::Extension::Vector<DataType, ExecutionSpace>& x,
                                           const int                                            incx,
                                           Kokkos::Extension::Vector<DataType, ExecutionSpace>& y,
                                           const int                                            incy,
                                           const DataType&                                      c,
                                           const DataType&                                      s)
    {
        const int n = x.extent(0);

        int      ix;
        int      iy;
        DataType stemp;

        if(n <= 0) {}
        else if(incx == 1 && incy == 1)
        {
            for(int i = 0; i < n; ++i)
            {
                stemp = c * x[i] + s * y[i];
                y[i]  = c * y[i] - s * x[i];
                x[i]  = stemp;
            }
        }
        else
        {
            if(0 <= incx)
            {
                ix = 0;
            }
            else
            {
                ix = (-n + 1) * incx;
            }

            if(0 <= incy)
            {
                iy = 0;
            }
            else
            {
                iy = (-n + 1) * incy;
            }

            for(int i = 0; i < n; ++i)
            {
                stemp = c * x[ix] + s * y[iy];
                y[iy] = c * y[iy] - s * x[ix];
                x[ix] = stemp;
                ix    = ix + incx;
                iy    = iy + incy;
            }
        }
    }

    template<typename DataType, class ExecutionSpace>
    KOKKOS_INLINE_FUNCTION static void rotg(DataType* sa, DataType* sb, DataType* c, DataType* s)
    {
        DataType r;
        DataType roe;
        DataType z;

        if(fabs(*sb) < fabs(*sa))
        {
            roe = *sa;
        }
        else
        {
            roe = *sb;
        }

        DataType scale = fabs(*sa) + fabs(*sb);

        if(scale == 0.0)
        {
            *c = 1.0;
            *s = 0.0;
            r  = 0.0;
        }
        else
        {
            r  = scale * sqrt((*sa / scale) * (*sa / scale) + (*sb / scale) * (*sb / scale));
            r  = sign(roe) * r;
            *c = *sa / r;
            *s = *sb / r;
        }

        if(0.0 < fabs(*c) && fabs(*c) <= *s)
        {
            z = 1.0 / *c;
        }
        else
        {
            z = *s;
        }

        *sa = r;
        *sb = z;
    }

    template<typename DataType, class ExecutionSpace>
    KOKKOS_INLINE_FUNCTION static int Svdc(Kokkos::Extension::Vector<DataType, ExecutionSpace>& a,
                                           int                                                  lda,
                                           int                                                  m,
                                           int                                                  n,
                                           Kokkos::Extension::Vector<DataType, ExecutionSpace>& s,
                                           Kokkos::Extension::Vector<DataType, ExecutionSpace>& e,
                                           Kokkos::Extension::Vector<DataType, ExecutionSpace>& u,
                                           int                                                  ldu,
                                           Kokkos::Extension::Vector<DataType, ExecutionSpace>& v,
                                           int                                                  ldv,
                                           Kokkos::Extension::Vector<DataType, ExecutionSpace>& work,
                                           int                                                  job)
    {
        DataType b;
        DataType c;
        DataType cs;
        DataType el;
        DataType emm1;
        DataType f;
        DataType g;
        int      i;
        int      info;
        int      iter;
        int      j;
        int      jobu;
        int      k;
        int      kase;
        int      kk;
        int      l;
        int      ll;
        int      lls;
        int      ls;
        int      lu;
        int      maxit = 30;
        int      mm;
        int      mm1;
        int      mn;
        int      nct;
        int      nctp1;
        int      ncu;
        int      nrt;
        int      nrtp1;
        DataType scale;
        DataType shift;
        DataType sl;
        DataType sm;
        DataType smm1;
        DataType sn;
        DataType t;
        DataType t1;
        DataType test;
        bool     wantu;
        bool     wantv;
        DataType ztest;
        //
        //  Determine what is to be computed.
        //
        info  = 0;
        wantu = false;
        wantv = false;
        jobu  = (job % 100) / 10;

        if(1 < jobu)
        {
            ncu = std::min(m, n);
        }
        else
        {
            ncu = m;
        }

        if(jobu != 0)
        {
            wantu = true;
        }

        if((job % 10) != 0)
        {
            wantv = true;
        }
        //
        //  Reduce A to bidiagonal form, storing the diagonal elements
        //  in S and the super-diagonal elements in E.
        //
        nct = std::min(m - 1, n);
        nrt = max(0, std::min(m, n - 2));
        lu  = max(nct, nrt);

        for(l = 1; l <= lu; ++l)
        {
            //
            //  Compute the transformation for the L-th column and
            //  place the L-th diagonal in S(L).
            //
            if(l <= nct)
            {
                s[l - 1] = KokkosBlas::nrm2(m - l + 1, a + l - 1 + (l - 1) * lda);

                if(s[l - 1] != 0.0)
                {
                    if(a[l - 1 + (l - 1) * lda] != 0.0)
                    {
                        s[l - 1] = sign(a[l - 1 + (l - 1) * lda]) * fabs(s[l - 1]);
                    }
                    KokkosBlas::scal(m - l + 1, 1.0 / s[l - 1], a + l - 1 + (l - 1) * lda);
                    a[l - 1 + (l - 1) * lda] = 1.0 + a[l - 1 + (l - 1) * lda];
                }
                s[l - 1] = -s[l - 1];
            }

            for(j = l + 1; j <= n; ++j)
            {
                //
                //  Apply the transformation.
                //
                if(l <= nct && s[l - 1] != 0.0)
                {
                    t = -KokkosBlas::dot(m - l + 1, a + l - 1 + (l - 1) * lda, a + l - 1 + (j - 1) * lda) / a[l - 1 + (l - 1) * lda];
                    KokkosBlas::axpy(m - l + 1, t, a + l - 1 + (l - 1) * lda, a + l - 1 + (j - 1) * lda);
                }
                //
                //  Place the L-th row of A into E for the
                //  subsequent calculation of the row transformation.
                //
                e[j - 1] = a[l - 1 + (j - 1) * lda];
            }
            //
            //  Place the transformation in U for subsequent back multiplication.
            //
            if(wantu && l <= nct)
            {
                for(i = l; i <= m; ++i)
                {
                    u[i - 1 + (l - 1) * ldu] = a[i - 1 + (l - 1) * lda];
                }
            }

            if(l <= nrt)
            {
                //
                //  Compute the L-th row transformation and place the
                //  L-th superdiagonal in E(L).
                //
                e[l - 1] = KokkosBlas::nrm2(n - l, e + l);

                if(e[l - 1] != 0.0)
                {
                    if(e[l] != 0.0)
                    {
                        e[l - 1] = sign(e[l]) * fabs(e[l - 1]);
                    }
                    KokkosBlas::scal(n - l, 1.0 / e[l - 1], e + l);
                    e[l] = 1.0 + e[l];
                }

                e[l - 1] = -e[l - 1];
                //
                //  Apply the transformation.
                //
                if(l + 1 <= m && e[l - 1] != 0.0)
                {
                    for(j = l + 1; j <= m; ++j)
                    {
                        work[j - 1] = 0.0;
                    }

                    for(j = l + 1; j <= n; ++j)
                    {
                        KokkosBlas::axpy(m - l, e[j - 1], a + l + (j - 1) * lda, work + l);
                    }

                    for(j = l + 1; j <= n; ++j)
                    {
                        KokkosBlas::axpy(m - l, -e[j - 1] / e[l], work + l, a + l + (j - 1) * lda);
                    }
                }
                //
                //  Place the transformation in V for subsequent back multiplication.
                //
                if(wantv)
                {
                    for(j = l + 1; j <= n; ++j)
                    {
                        v[j - 1 + (l - 1) * ldv] = e[j - 1];
                    }
                }
            }
        }
        //
        //  Set up the final bidiagonal matrix of order MN.
        //
        mn    = std::min(m + 1, n);
        nctp1 = nct + 1;
        nrtp1 = nrt + 1;

        if(nct < n)
        {
            s[nctp1 - 1] = a[nctp1 - 1 + (nctp1 - 1) * lda];
        }

        if(m < mn)
        {
            s[mn - 1] = 0.0;
        }

        if(nrtp1 < mn)
        {
            e[nrtp1 - 1] = a[nrtp1 - 1 + (mn - 1) * lda];
        }

        e[mn - 1] = 0.0;
        //
        //  If required, generate U.
        //
        if(wantu)
        {
            for(i = 1; i <= m; ++i)
            {
                for(j = nctp1; j <= ncu; ++j)
                {
                    u[(i - 1) + (j - 1) * ldu] = 0.0;
                }
            }

            for(j = nctp1; j <= ncu; ++j)
            {
                u[j - 1 + (j - 1) * ldu] = 1.0;
            }

            for(ll = 1; ll <= nct; ++ll)
            {
                l = nct - ll + 1;

                if(s[l - 1] != 0.0)
                {
                    for(j = l + 1; j <= ncu; ++j)
                    {
                        t = -KokkosBlas::dot(m - l + 1, u + (l - 1) + (l - 1) * ldu, u + (l - 1) + (j - 1) * ldu) / u[l - 1 + (l - 1) * ldu];
                        KokkosBlas::axpy(m - l + 1, t, u + (l - 1) + (l - 1) * ldu, u + (l - 1) + (j - 1) * ldu);
                    }

                    KokkosBlas::scal(m - l + 1, -1.0, u + (l - 1) + (l - 1) * ldu);
                    u[l - 1 + (l - 1) * ldu] = 1.0 + u[l - 1 + (l - 1) * ldu];
                    for(i = 1; i <= l - 1; ++i)
                    {
                        u[i - 1 + (l - 1) * ldu] = 0.0;
                    }
                }
                else
                {
                    for(i = 1; i <= m; ++i)
                    {
                        u[i - 1 + (l - 1) * ldu] = 0.0;
                    }
                    u[l - 1 + (l - 1) * ldu] = 1.0;
                }
            }
        }
        //
        //  If it is required, generate V.
        //
        if(wantv)
        {
            for(ll = 1; ll <= n; ++ll)
            {
                l = n - ll + 1;

                if(l <= nrt && e[l - 1] != 0.0)
                {
                    for(j = l + 1; j <= n; ++j)
                    {
                        t = -KokkosBlas::dot(n - l, v + l + (l - 1) * ldv, v + l + (j - 1) * ldv) / v[l + (l - 1) * ldv];
                        KokkosBlas::axpy(n - l, t, v + l + (l - 1) * ldv, v + l + (j - 1) * ldv);
                    }
                }
                for(i = 1; i <= n; ++i)
                {
                    v[i - 1 + (l - 1) * ldv] = 0.0;
                }
                v[l - 1 + (l - 1) * ldv] = 1.0;
            }
        }
        //
        //  Main iteration loop for the singular values.
        //
        mm   = mn;
        iter = 0;

        while(0 < mn)
        {
            //
            //  If too many iterations have been performed, set flag and return.
            //
            if(maxit <= iter)
            {
                info = mn;
                return info;
            }
            //
            //  This section of the program inspects for
            //  negligible elements in the S and E arrays.
            //
            //  On completion the variables KASE and L are set as follows:
            //
            //  KASE = 1     if S(MN) and E(L-1) are negligible and L < MN
            //  KASE = 2     if S(L) is negligible and L < MN
            //  KASE = 3     if E(L-1) is negligible, L < MN, and
            //               S(L), ..., S(MN) are not negligible (QR step).
            //  KASE = 4     if E(MN-1) is negligible (convergence).
            //
            for(ll = 1; ll <= mn; ++ll)
            {
                l = mn - ll;

                if(l == 0)
                {
                    break;
                }

                test  = fabs(s[l - 1]) + fabs(s[l]);
                ztest = test + fabs(e[l - 1]);

                if(ztest == test)
                {
                    e[l - 1] = 0.0;
                    break;
                }
            }

            if(l == mn - 1)
            {
                kase = 4;
            }
            else
            {
                for(lls = l + 1; lls <= mn + 1; ++lls)
                {
                    ls = mn - lls + l + 1;

                    if(ls == l)
                    {
                        break;
                    }

                    test = 0.0;
                    if(ls != mn)
                    {
                        test = test + fabs(e[ls - 1]);
                    }

                    if(ls != l + 1)
                    {
                        test = test + fabs(e[ls - 2]);
                    }

                    ztest = test + fabs(s[ls - 1]);

                    if(ztest == test)
                    {
                        s[ls - 1] = 0.0;
                        break;
                    }
                }

                if(ls == l)
                {
                    kase = 3;
                }
                else if(ls == mn)
                {
                    kase = 1;
                }
                else
                {
                    kase = 2;
                    l    = ls;
                }
            }

            l = l + 1;
            //
            //  Deflate negligible S(MN).
            //
            if(kase == 1)
            {
                mm1       = mn - 1;
                f         = e[mn - 2];
                e[mn - 2] = 0.0;

                for(kk = 1; kk <= mm1; ++kk)
                {
                    k  = mm1 - kk + l;
                    t1 = s[k - 1];
                    rotg(&t1, &f, &cs, &sn);
                    s[k - 1] = t1;

                    if(k != l)
                    {
                        f        = -sn * e[k - 2];
                        e[k - 2] = cs * e[k - 2];
                    }

                    if(wantv)
                    {
                        rot(v + 0 + (k - 1) * ldv, 1, v + 0 + (mn - 1) * ldv, 1, cs, sn);
                    }
                }
            }
            //
            //  Split at negligible S(L).
            //
            else if(kase == 2)
            {
                f        = e[l - 2];
                e[l - 2] = 0.0;

                for(k = l; k <= mn; ++k)
                {
                    t1 = s[k - 1];
                    rotg(&t1, &f, &cs, &sn);
                    s[k - 1] = t1;
                    f        = -sn * e[k - 1];
                    e[k - 1] = cs * e[k - 1];
                    if(wantu)
                    {
                        rot(m, u + 0 + (k - 1) * ldu, 1, u + 0 + (l - 2) * ldu, 1, cs, sn);
                    }
                }
            }
            //
            //  Perform one QR step.
            //
            else if(kase == 3)
            {
                //
                //  Calculate the shift.
                //
                scale = max(fabs(s[mn - 1]), max(fabs(s[mn - 2]), max(fabs(e[mn - 2]), max(fabs(s[l - 1]), fabs(e[l - 1])))));

                sm    = s[mn - 1] / scale;
                smm1  = s[mn - 2] / scale;
                emm1  = e[mn - 2] / scale;
                sl    = s[l - 1] / scale;
                el    = e[l - 1] / scale;
                b     = ((smm1 + sm) * (smm1 - sm) + emm1 * emm1) / 2.0;
                c     = (sm * emm1) * (sm * emm1);
                shift = 0.0;

                if(b != 0.0 || c != 0.0)
                {
                    shift = sqrt(b * b + c);
                    if(b < 0.0)
                    {
                        shift = -shift;
                    }
                    shift = c / (b + shift);
                }

                f = (sl + sm) * (sl - sm) - shift;
                g = sl * el;
                //
                //  Chase zeros.
                //
                mm1 = mn - 1;

                for(k = l; k <= mm1; ++k)
                {
                    rotg(&f, &g, &cs, &sn);

                    if(k != l)
                    {
                        e[k - 2] = f;
                    }

                    f        = cs * s[k - 1] + sn * e[k - 1];
                    e[k - 1] = cs * e[k - 1] - sn * s[k - 1];
                    g        = sn * s[k];
                    s[k]     = cs * s[k];

                    if(wantv)
                    {
                        rot(v + 0 + (k - 1) * ldv, 1, v + 0 + k * ldv, 1, cs, sn);
                    }

                    rotg(&f, &g, &cs, &sn);
                    s[k - 1] = f;
                    f        = cs * e[k - 1] + sn * s[k];
                    s[k]     = -sn * e[k - 1] + cs * s[k];
                    g        = sn * e[k];
                    e[k]     = cs * e[k];

                    if(wantu && k < m)
                    {
                        rot(m, u + 0 + (k - 1) * ldu, 1, u + 0 + k * ldu, 1, cs, sn);
                    }
                }
                e[mn - 2] = f;
                iter      = iter + 1;
            }
            //
            //  Convergence.
            //
            else if(kase == 4)
            {
                //
                //  Make the singular value nonnegative.
                //
                if(s[l - 1] < 0.0)
                {
                    s[l - 1] = -s[l - 1];
                    if(wantv)
                    {
                        KokkosBlas::scal(n, -1.0, v + 0 + (l - 1) * ldv);
                    }
                }
                //
                //  Order the singular value.
                //
                for(;;)
                {
                    if(l == mm)
                    {
                        break;
                    }

                    if(s[l] <= s[l - 1])
                    {
                        break;
                    }

                    t        = s[l - 1];
                    s[l - 1] = s[l];
                    s[l]     = t;

                    if(wantv && l < n)
                    {
                        swap(v + 0 + (l - 1) * ldv, 1, v + 0 + l * ldv, 1);
                    }

                    if(wantu && l < m)
                    {
                        swap(u + 0 + (l - 1) * ldu, 1, u + 0 + l * ldu, 1);
                    }

                    l = l + 1;
                }
                iter = 0;
                mn   = mn - 1;
            }
        }

        return info;
    }

    template<typename DataType, class ExecutionSpace>
    KOKKOS_INLINE_FUNCTION static void swap(Kokkos::Extension::Vector<DataType, ExecutionSpace>& x,
                                            const int                                            incx,
                                            Kokkos::Extension::Vector<DataType, ExecutionSpace>& y,
                                            const int                                            incy)
    {
        const int n = x.extent(0);

        int      ix;
        int      iy;
        DataType temp;

        if(n <= 0) {}
        else if(incx == 1 && incy == 1)
        {
            const int m = n % 3;

            for(int i = 0; i < m; ++i)
            {
                temp = x[i];
                x[i] = y[i];
                y[i] = temp;
            }

            for(int i = m; i < n; i = i + 3)
            {
                temp = x[i];
                x[i] = y[i];
                y[i] = temp;

                temp     = x[i + 1];
                x[i + 1] = y[i + 1];
                y[i + 1] = temp;

                temp     = x[i + 2];
                x[i + 2] = y[i + 2];
                y[i + 2] = temp;
            }
        }
        else
        {
            if(0 <= incx)
            {
                ix = 0;
            }
            else
            {
                ix = (-n + 1) * incx;
            }

            if(0 <= incy)
            {
                iy = 0;
            }
            else
            {
                iy = (-n + 1) * incy;
            }

            for(int i = 0; i < n; ++i)
            {
                temp  = x[ix];
                x[ix] = y[iy];
                y[iy] = temp;
                ix    = ix + incx;
                iy    = iy + incy;
            }
        }
    }

    template<typename DataType, class ExecutionSpace>
    using phi_funcPtr = void(const Kokkos::Extension::Vector<DataType, ExecutionSpace>&, const DataType&, Kokkos::Extension::Vector<DataType, ExecutionSpace>&);

    template<typename DataType, class ExecutionSpace>
    KOKKOS_INLINE_FUNCTION static void phi1(const Kokkos::Extension::Vector<DataType, ExecutionSpace>& r, const DataType& r0, Kokkos::Extension::Vector<DataType, ExecutionSpace>& v)
    {
        for(int i = 0; i < v.extent(0); ++i)
        {
            v[i] = sqrt(r[i] * r[i] + r0 * r0);
        }
    }

    template<typename DataType, class ExecutionSpace>
    KOKKOS_INLINE_FUNCTION static void phi2(const Kokkos::Extension::Vector<DataType, ExecutionSpace>& r, const DataType& r0, Kokkos::Extension::Vector<DataType, ExecutionSpace>& v)
    {
        for(int i = 0; i < v.extent(0); ++i)
        {
            v[i] = 1.0 / sqrt(r[i] * r[i] + r0 * r0);
        }
    }

    template<typename DataType, class ExecutionSpace>
    KOKKOS_INLINE_FUNCTION static void phi3(const Kokkos::Extension::Vector<DataType, ExecutionSpace>& r, const DataType& r0, Kokkos::Extension::Vector<DataType, ExecutionSpace>& v)
    {
        for(int i = 0; i < v.extent(0); ++i)
        {
            if(r[i] <= 0.0)
            {
                v[i] = 0.0;
            }
            else
            {
                v[i] = r[i] * r[i] * log(r[i] / r0);
            }
        }
    }

    template<typename DataType, class ExecutionSpace>
    KOKKOS_INLINE_FUNCTION static void phi4(const Kokkos::Extension::Vector<DataType, ExecutionSpace>& r, const DataType& r0, Kokkos::Extension::Vector<DataType, ExecutionSpace>& v)
    {
        for(int i = 0; i < v.extent(0); ++i)
        {
            v[i] = exp(-0.5 * r[i] * r[i] / r0 / r0);
        }
    }

    template<typename DataType, class ExecutionSpace>
    KOKKOS_INLINE_FUNCTION static Kokkos::Extension::Vector<DataType, ExecutionSpace> SolveSvd(int                                                  m,
                                                                                               int                                                  n,
                                                                                               Kokkos::Extension::Vector<DataType, ExecutionSpace>& a,
                                                                                               Kokkos::Extension::Vector<DataType, ExecutionSpace>& b)
    {
        const int nm      = n * m;
        const int nn      = n * n;
        const int mm      = m * m;
        const int max_m_n = max(m + 1, n);
        const int lda     = m;
        const int ldu     = m;
        const int ldv     = n;
        const int job     = 11;

        Kokkos::Extension::Vector<DataType, ExecutionSpace> a_copy(new DataType[nm], nm);
        Kokkos::deep_copy(a_copy, a);

        Kokkos::Extension::Vector<DataType, ExecutionSpace> a_pseudo(new DataType[nm], nm);
        Kokkos::Extension::Vector<DataType, ExecutionSpace> e(new DataType[max_m_n], max_m_n);
        Kokkos::Extension::Vector<DataType, ExecutionSpace> s(new DataType[nm], nm);
        Kokkos::Extension::Vector<DataType, ExecutionSpace> sp(new DataType[nm], nm);
        Kokkos::Extension::Vector<DataType, ExecutionSpace> sdiag(new DataType[max_m_n], max_m_n);
        Kokkos::Extension::Vector<DataType, ExecutionSpace> u(new DataType[mm], mm);
        Kokkos::Extension::Vector<DataType, ExecutionSpace> v(new DataType[nn], nn);
        Kokkos::Extension::Vector<DataType, ExecutionSpace> work(new DataType[m], m);

        Kokkos::Extension::Vector<DataType, ExecutionSpace> x(new DataType[b.extent(0)], b.extent(0));

        //
        //  Compute the SVD decomposition.
        //

        const int info = Svdc(a_copy, lda, m, n, sdiag, e, u, ldu, v, ldv, work, job);

        if(info != 0)
        {
            // cerr << std::endl;
            // cerr << "R8MAT_SOLVE_SVD - Fatal error!\n";
            // cerr << "  The SVD could not be calculated.\n";
            // cerr << "  LINPACK routine DSVDC returned a nonzero\n";
            // cerr << "  value of the error flag, INFO = " << info << std::endl;
            return;
        }

        for(int j = 0; j < n; ++j)
        {
            for(int i = 0; i < m; ++i)
            {
                s[i + j * m] = 0.0;
            }
        }
        for(int i = 0; i < std::min(m, n); ++i)
        {
            s[i + i * m] = sdiag[i];
        }
        //
        //  Compute the pseudo inverse.
        //

        for(int j = 0; j < m; ++j)
        {
            for(int i = 0; i < n; ++i)
            {
                sp[i + j * m] = 0.0;
            }
        }
        for(int i = 0; i < std::min(m, n); ++i)
        {
            if(s[i + i * m] != 0.0)
            {
                sp[i + i * n] = 1.0 / s[i + i * m];
            }
        }

        for(int j = 0; j < m; ++j)
        {
            for(int i = 0; i < n; ++i)
            {
                a_pseudo[i + j * n] = 0.0;

                for(int k = 0; k < n; ++k)
                {
                    for(int l = 0; l < m; ++l)
                    {
                        a_pseudo[i + j * n] = a_pseudo[i + j * n] + v[i + k * n] * sp[k + l * n] * u[j + l * m];
                    }
                }
            }
        }
        //
        //  Compute x = A_pseudo * b.
        //
        //
        Kokkos::Extension::Matrix<DataType, ExecutionSpace> aMatrix(a_pseudo.data(), n, m);

        KokkosBlas::gemv("N", 1.0, aMatrix, x, 1.0, b);

        return x;
    }

    template<typename DataType, class ExecutionSpace>
    KOKKOS_INLINE_FUNCTION static Kokkos::Extension::Vector<DataType, ExecutionSpace> rbf_interp_nd(const int                                            m,
                                                                                                    int                                                  nd,
                                                                                                    Kokkos::Extension::Vector<DataType, ExecutionSpace>& xd,
                                                                                                    const DataType&                                      r0,
                                                                                                    phi_funcPtr<DataType, ExecutionSpace>                phi,
                                                                                                    Kokkos::Extension::Vector<DataType, ExecutionSpace>& w,
                                                                                                    int                                                  ni,
                                                                                                    Kokkos::Extension::Vector<DataType, ExecutionSpace>& xi)
    {
        Kokkos::Extension::Vector<DataType, ExecutionSpace> fi(new DataType[ni], ni);
        Kokkos::Extension::Vector<DataType, ExecutionSpace> r(new DataType[nd], nd);
        Kokkos::Extension::Vector<DataType, ExecutionSpace> v(new DataType[nd], nd);

        for(int i = 0; i < ni; ++i)
        {
            for(int j = 0; j < nd; ++j)
            {
                r[j] = 0.0;
                for(int k = 0; k < m; ++k)
                {
                    r[j] = r[j] + pow(xi[k + i * m] - xd[k + j * m], 2);
                }
                r[j] = sqrt(r[j]);
            }

            phi(nd, r, r0, v);

            fi[i] = KokkosBlas::dot(v, w);
        }

        return fi;
    }

    template<typename DataType, class ExecutionSpace>
    KOKKOS_INLINE_FUNCTION static Kokkos::Extension::Vector<DataType, ExecutionSpace> rbf_weight(const int                                            m,
                                                                                                 const int                                            nd,
                                                                                                 Kokkos::Extension::Vector<DataType, ExecutionSpace>& xd,
                                                                                                 const DataType&                                      r0,
                                                                                                 phi_funcPtr<DataType, ExecutionSpace>                phi,
                                                                                                 Kokkos::Extension::Vector<DataType, ExecutionSpace>& fd)
    {
        Kokkos::Extension::Vector<DataType, ExecutionSpace> a(new DataType[nd * nd], nd * nd);
        Kokkos::Extension::Vector<DataType, ExecutionSpace> r(new DataType[nd], nd);
        Kokkos::Extension::Vector<DataType, ExecutionSpace> v(new DataType[nd], nd);

        for(int i = 0; i < nd; ++i)
        {
            for(int j = 0; j < nd; ++j)
            {
                r[j] = 0.0;

                for(int k = 0; k < m; ++k)
                {
                    r[j] = r[j] + pow(xd[k + i * m] - xd[k + j * m], 2);
                }

                r[j] = sqrt(r[j]);
            }

            phi(nd, r, r0, v);

            for(int j = 0; j < nd; ++j)
            {
                a[i + j * nd] = v[j];
            }
        }

        //
        //  Solve for the weights.
        //

        return SolveSvd(nd, nd, a, fd);
    }

}
