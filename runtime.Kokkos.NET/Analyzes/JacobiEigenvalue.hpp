#pragma once

#include "runtime.Kokkos/ViewTypes.hpp"
#include "runtime.Kokkos/Extensions.hpp"

/// <summary>
///
/// </summary>
/// <param name="a">Input, double A[N*N], the matrix, which must be square, real, and symmetric.</param>
/// <param name="it_max">Input, int IT_MAX, the maximum number of iterations.</param>
/// <param name="v">Output, double V[N*N], the matrix of eigenvectors.</param>
/// <param name="d">Output, double D[N], the eigenvalues, in descending order.</param>
/// <param name="it_num">Output, int &IT_NUM, the total number of iterations.</param>
/// <param name="rot_num">Output, int &ROT_NUM, the total number of rotations.</param>
template<typename DataType, class ExecutionSpace>
static void JacobiEigenvalue(Kokkos::Extension::Matrix<DataType, ExecutionSpace> a,
                             const int                                           it_max,
                             Kokkos::Extension::Matrix<DataType, ExecutionSpace> v,
                             Kokkos::Extension::Vector<DataType, ExecutionSpace> d,
                             int&                                                it_num,
                             int&                                                rot_num)
{
    const int n = a.extent(0);

    double c;
    double g;
    double gapq;
    double h;
    int    i;
    int    j;
    int    k;
    int    l;
    int    m;
    int    p;
    int    q;
    double s;
    double t;
    double tau;
    double term;
    double termp;
    double termq;
    double theta;
    double thresh;
    double w;

    r8mat_identity(n, v);

    r8mat_diag_get_vector(n, a, d);

    Kokkos::Extension::Vector<DataType, ExecutionSpace> bw(new double[n], n);
    Kokkos::Extension::Vector<DataType, ExecutionSpace> zw(new double[n], n);

    for(i = 0; i < n; ++i)
    {
        bw[i] = d[i];
        zw[i] = 0.0;
    }
    it_num  = 0;
    rot_num = 0;

    while(it_num < it_max)
    {
        it_num = it_num + 1;
        //
        //  The convergence threshold is based on the size of the elements in
        //  the strict upper triangle of the matrix.
        //
        thresh = 0.0;
        for(j = 0; j < n; ++j)
        {
            for(i = 0; i < j; ++i)
            {
                thresh = thresh + a[i + j * n] * a[i + j * n];
            }
        }

        thresh = sqrt(thresh) / (double)(4 * n);

        if(thresh == 0.0)
        {
            break;
        }

        for(p = 0; p < n; p++)
        {
            for(q = p + 1; q < n; q++)
            {
                gapq  = 10.0 * fabs(a[p + q * n]);
                termp = gapq + fabs(d[p]);
                termq = gapq + fabs(d[q]);
                //
                //  Annihilate tiny offdiagonal elements.
                //
                if(4 < it_num && termp == fabs(d[p]) && termq == fabs(d[q]))
                {
                    a[p + q * n] = 0.0;
                }
                //
                //  Otherwise, apply a rotation.
                //
                else if(thresh <= fabs(a[p + q * n]))
                {
                    h    = d[q] - d[p];
                    term = fabs(h) + gapq;

                    if(term == fabs(h))
                    {
                        t = a[p + q * n] / h;
                    }
                    else
                    {
                        theta = 0.5 * h / a[p + q * n];
                        t     = 1.0 / (fabs(theta) + sqrt(1.0 + theta * theta));
                        if(theta < 0.0)
                        {
                            t = -t;
                        }
                    }
                    c   = 1.0 / sqrt(1.0 + t * t);
                    s   = t * c;
                    tau = s / (1.0 + c);
                    h   = t * a[p + q * n];
                    //
                    //  Accumulate corrections to diagonal elements.
                    //
                    zw[p] = zw[p] - h;
                    zw[q] = zw[q] + h;
                    d[p]  = d[p] - h;
                    d[q]  = d[q] + h;

                    a[p + q * n] = 0.0;
                    //
                    //  Rotate, using information from the upper triangle of A only.
                    //
                    for(j = 0; j < p; ++j)
                    {
                        g            = a[j + p * n];
                        h            = a[j + q * n];
                        a[j + p * n] = g - s * (h + g * tau);
                        a[j + q * n] = h + s * (g - h * tau);
                    }

                    for(j = p + 1; j < q; ++j)
                    {
                        g            = a[p + j * n];
                        h            = a[j + q * n];
                        a[p + j * n] = g - s * (h + g * tau);
                        a[j + q * n] = h + s * (g - h * tau);
                    }

                    for(j = q + 1; j < n; ++j)
                    {
                        g            = a[p + j * n];
                        h            = a[q + j * n];
                        a[p + j * n] = g - s * (h + g * tau);
                        a[q + j * n] = h + s * (g - h * tau);
                    }
                    //
                    //  Accumulate information in the eigenvector matrix.
                    //
                    for(j = 0; j < n; ++j)
                    {
                        g            = v[j + p * n];
                        h            = v[j + q * n];
                        v[j + p * n] = g - s * (h + g * tau);
                        v[j + q * n] = h + s * (g - h * tau);
                    }
                    rot_num = rot_num + 1;
                }
            }
        }

        for(i = 0; i < n; ++i)
        {
            bw[i] = bw[i] + zw[i];
            d[i]  = bw[i];
            zw[i] = 0.0;
        }
    }
    //
    //  Restore upper triangle of input matrix.
    //
    for(j = 0; j < n; ++j)
    {
        for(i = 0; i < j; ++i)
        {
            a[i + j * n] = a[j + i * n];
        }
    }
    //
    //  Ascending sort the eigenvalues and eigenvectors.
    //
    for(k = 0; k < n - 1; k++)
    {
        m = k;
        for(l = k + 1; l < n; l++)
        {
            if(d[l] < d[m])
            {
                m = l;
            }
        }

        if(m != k)
        {
            t    = d[m];
            d[m] = d[k];
            d[k] = t;
            for(i = 0; i < n; ++i)
            {
                w            = v[i + m * n];
                v[i + m * n] = v[i + k * n];
                v[i + k * n] = w;
            }
        }
    }

    delete[] bw;
    delete[] zw;

    return;
}

void r8mat_diag_get_vector(int n, double a[], double v[])
{
    for(int i = 0; i < n; ++i)
    {
        v[i] = a[i + i * n];
    }

    return;
}

void r8mat_identity(int n, double a[])
{
    int k = 0;
    for(int j = 0; j < n; ++j)
    {
        for(int i = 0; i < n; ++i)
        {
            if(i == j)
            {
                a[k] = 1.0;
            }
            else
            {
                a[k] = 0.0;
            }
            k = k + 1;
        }
    }

    return;
}

double r8mat_is_eigen_right(int n, int k, double a[], double x[], double lambda[])
{
    int i;
    int j;

    double* c = new double[n * k];

    for(j = 0; j < k; ++j)
    {
        for(i = 0; i < n; ++i)
        {
            c[i + j * n] = 0.0;
            for(int l = 0; l < n; l++)
            {
                c[i + j * n] = c[i + j * n] + a[i + l * n] * x[l + j * n];
            }
        }
    }

    for(j = 0; j < k; ++j)
    {
        for(i = 0; i < n; ++i)
        {
            c[i + j * n] = c[i + j * n] - lambda[j] * x[i + j * n];
        }
    }

    double error_frobenius = r8mat_norm_fro(n, k, c);

    delete[] c;

    return error_frobenius;
}

double r8mat_norm_fro(int m, int n, double a[])
{
    double value = 0.0;
    for(int j = 0; j < n; ++j)
    {
        for(int i = 0; i < m; ++i)
        {
            value = value + pow(a[i + j * m], 2);
        }
    }
    value = sqrt(value);

    return value;
}

void r8mat_print(int m, int n, double a[], std::string title)
{
    r8mat_print_some(m, n, a, 1, 1, m, n, title);

    return;
}

void r8mat_print_some(int m, int n, double a[], int ilo, int jlo, int ihi, int jhi, std::string title)
{
#define INCX 5

    int i;
    int i2hi;
    int i2lo;
    int j;
    int j2hi;

    std::cout << std::endl;
    std::cout << title << std::endl;

    if(m <= 0 || n <= 0)
    {
        std::cout << std::endl;
        std::cout << "  (None)\n";
        return;
    }
    //
    //  Print the columns of the matrix, in strips of 5.
    //
    for(int j2lo = jlo; j2lo <= jhi; j2lo = j2lo + INCX)
    {
        j2hi = j2lo + INCX - 1;
        if(n < j2hi)
        {
            j2hi = n;
        }
        if(jhi < j2hi)
        {
            j2hi = jhi;
        }
        std::cout << std::endl;
        //
        //  For each column J in the current range...
        //
        //  Write the header.
        //
        std::cout << "  Col:    ";
        for(j = j2lo; j <= j2hi; ++j)
        {
            std::cout << setw(7) << j - 1 << "       ";
        }
        std::cout << std::endl;
        std::cout << "  Row\n";
        std::cout << std::endl;
        //
        //  Determine the range of the rows in this strip.
        //
        if(1 < ilo)
        {
            i2lo = ilo;
        }
        else
        {
            i2lo = 1;
        }
        if(ihi < m)
        {
            i2hi = ihi;
        }
        else
        {
            i2hi = m;
        }

        for(i = i2lo; i <= i2hi; ++i)
        {
            //
            //  Print out (up to) 5 entries in row I, that lie in the current strip.
            //
            std::cout << std::setw(5) << i - 1 << ": ";
            for(j = j2lo; j <= j2hi; ++j)
            {
                std::cout << std::setw(12) << a[i - 1 + (j - 1) * m] << "  ";
            }
            std::cout << std::endl;
        }
    }

    return;
#undef INCX
}

void r8vec_print(int n, double a[], std::string title)
{
    std::cout << std::endl;
    std::cout << title << std::endl;
    std::cout << std::endl;
    for(int i = 0; i < n; ++i)
    {
        std::cout << "  " << std::setw(8) << i << ": " << std::setw(14) << a[i] << std::endl;
    }

    return;
}
