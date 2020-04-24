#pragma once

#include "runtime.Kokkos/ViewTypes.hpp"
#include "runtime.Kokkos/Extensions.hpp"

#include "StdExtensions.hpp"

#include <NumericalMethods/Algebra/Eigenvalue.hpp>

namespace PCA
{
    template<typename DataType, class ExecutionSpace>
    static void adjust_data(Kokkos::Extension::Matrix<DataType, ExecutionSpace>& d, Kokkos::Extension::Vector<DataType, ExecutionSpace>& means)
    {
        Kokkos::parallel_for(Kokkos::RangePolicy<ExecutionSpace, int>(0, d.extent(1)), [=] __host__ __device__(const int& i) {
            DataType mean = 0;
            for(int j = 0; j < d.extent(0); ++j)
            {
                mean += d(j, i);
            }

            mean /= d.extent(0);

            // store the mean
            means(i) = mean;

            // subtract the mean
            for(int j = 0; j < d.extent(0); ++j)
            {
                d(j, i) -= mean;
            }
        });
    }

    template<typename DataType, class ExecutionSpace>
    KOKKOS_INLINE_FUNCTION static DataType compute_covariance(const Kokkos::Extension::Matrix<DataType, ExecutionSpace>& d, int i, int j)
    {
        DataType cov = 0;

        for(int k = 0; k < d.extent(0); ++k)
        {
            cov += d(k, i) * d(k, j);
        }

        return cov / (d.extent(0) - 1);
    }

    template<typename DataType, class ExecutionSpace>
    static void compute_covariance_matrix(const Kokkos::Extension::Matrix<DataType, ExecutionSpace>& d, Kokkos::Extension::Matrix<DataType, ExecutionSpace>& covar_matrix)
    {
        int dim = d.extent(1);
        assert(dim == covar_matrix.extent(0));
        assert(dim == covar_matrix.extent(1));

        Kokkos::parallel_for(Kokkos::RangePolicy<ExecutionSpace, int>(0, dim), [=] __host__ __device__(const int& i) {
            for(int j = i; j < dim; ++j)
            {
                covar_matrix(i, j) = compute_covariance(d, i, j);
            }
        });

        Kokkos::parallel_for(Kokkos::RangePolicy<ExecutionSpace, int>(1, dim), [=] __host__ __device__(const int& i) {
            for(int j = 0; j < i; ++j)
            {
                covar_matrix(i, j) = covar_matrix(j, i);
            }
        });
    }

    // Calculate the eigenvectors and eigenvalues of the covariance
    // matrix
    template<typename DataType, class ExecutionSpace>
    static void eigen(const Kokkos::Extension::Matrix<DataType, ExecutionSpace>& covar_matrix,
                      Kokkos::Extension::Matrix<DataType, ExecutionSpace>&       eigenvector,
                      Kokkos::Extension::Matrix<DataType, ExecutionSpace>&       eigenvalue)
    {
        NumericalMethods::Algebra::Eigenvalue<DataType, ExecutionSpace> eig(covar_matrix);
        eig.getV(eigenvector);
        eig.getD(eigenvalue);
    }

    template<typename DataType, class ExecutionSpace>
    static void transpose(const Kokkos::Extension::Matrix<DataType, ExecutionSpace>& src, Kokkos::Extension::Matrix<DataType, ExecutionSpace>& target)
    {
        Kokkos::parallel_for(Kokkos::RangePolicy<ExecutionSpace, int>(0, src.extent(0)), [=] __host__ __device__(const int& i) {
            for(int j = 0; j < src.extent(1); ++j)
            {
                target(j, i) = src(i, j);
            }
        });
    }

    // z = x * y
    template<typename DataType, class ExecutionSpace>
    static void multiply(const Kokkos::Extension::Matrix<DataType, ExecutionSpace>& x,
                         const Kokkos::Extension::Matrix<DataType, ExecutionSpace>& y,
                         Kokkos::Extension::Matrix<DataType, ExecutionSpace>&       z)
    {
        assert(x.extent(1) == y.extent(0));

        Kokkos::parallel_for(Kokkos::RangePolicy<ExecutionSpace, int>(0, x.extent(0)), [=] __host__ __device__(const int& i) {
            for(int j = 0; j < y.extent(1); ++j)
            {
                DataType sum = 0;

                int d = y.extent(0);

                for(int k = 0; k < d; k++)
                {
                    sum += x(i, k) * y(k, j);
                }
                z(i, j) = sum;
            }
        });
    }
}

template<typename DataType, class ExecutionSpace>
static int PrincipalComponentAnalysis()
{
    using namespace PCA;

    const int row = 10;
    const int col = 2;

    Kokkos::Extension::Matrix<DataType, ExecutionSpace> d("d", row, col);

    d(0, 0) = 2.5;
    d(0, 1) = 2.4;
    d(1, 0) = 0.5;
    d(1, 1) = 0.7;
    d(2, 0) = 2.2;
    d(2, 1) = 2.9;
    d(3, 0) = 1.9;
    d(3, 1) = 2.2;
    d(4, 0) = 3.1;
    d(4, 1) = 3.0;
    d(5, 0) = 2.3;
    d(5, 1) = 2.7;
    d(6, 0) = 2.0;
    d(6, 1) = 1.6;
    d(7, 0) = 1.0;
    d(7, 1) = 1.1;
    d(8, 0) = 1.5;
    d(8, 1) = 1.6;
    d(9, 0) = 1.1;
    d(9, 1) = 0.9;

    Kokkos::Extension::Vector<DataType, ExecutionSpace> means("means", col);

    adjust_data(d, means);

    std::cout << "adjust data" << std::endl;
    std::cout << d << std::endl;

    Kokkos::Extension::Matrix<DataType, ExecutionSpace> covar_matrix("covar_matrix", col, col);

    compute_covariance_matrix(d, covar_matrix);

    std::cout << "the covar matrix" << std::endl;
    std::cout << covar_matrix << std::endl;

    int dim = covar_matrix.extent(0);

    // get the eigenvectors
    Kokkos::Extension::Matrix<DataType, ExecutionSpace> eigenvector("eigenvector", dim, dim);

    // get the eigenvalues
    Kokkos::Extension::Matrix<DataType, ExecutionSpace> eigenvalue("eigenvalue", dim, dim);

    eigen(covar_matrix, eigenvector, eigenvalue);

    std::cout << "The eigenvectors:" << std::endl;
    std::cout << eigenvector << std::endl;

    std::cout << "The eigenvalues:" << std::endl;
    std::cout << eigenvalue << std::endl;

    // restore the old data
    // final_data = RowFeatureVector * RowDataAdjust
    //
    Kokkos::Extension::Matrix<DataType, ExecutionSpace> final_data("final_data", row, col);
    Kokkos::Extension::Matrix<DataType, ExecutionSpace> transpose_data("transpose_data", col, row);

    transpose(d, transpose_data);
    multiply(eigenvector, transpose_data, final_data);

    std::cout << "the final data" << std::endl;
    std::cout << final_data << std::endl;

    // eigenvalues
    // 0.0490833989 1.28402771

    return 0;
}
