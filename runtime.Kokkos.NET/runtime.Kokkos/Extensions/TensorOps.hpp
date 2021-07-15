#pragma once

#if !defined(KOKKOS_EXTENSIONS)
#    error "Do not include directly. Include Extensions.hpp"
#endif

namespace Kokkos
{
    namespace Extension
    {

        template<typename DataType, class ExecutionSpace>
        __inline static Tensor<DataType, ExecutionSpace> operator+(const Tensor<DataType, ExecutionSpace>& Lft, const Tensor<DataType, ExecutionSpace>& Rgt)
        {
            const size_type N0 = Lft.extent(0);
            const size_type N1 = Lft.extent(1);
            const size_type N2 = Lft.extent(2);

            Matrix<DataType, ExecutionSpace> res(Lft.label() + "+" + Rgt.label(), N0, N1, N2);

            for (int i = 0; i < N0; ++i)
            {
                for (int j = 0; j < N1; ++j)
                {
                    for (int k = 0; k < N2; ++k)
                    {
                        res(i, j, k) = Lft(i, j, k) + Rgt(i, j, k);
                    }
                }
            }

            return res;
        }

        template<typename DataType, class ExecutionSpace>
        __inline static Tensor<DataType, ExecutionSpace> operator-(const Tensor<DataType, ExecutionSpace>& Lft, const Tensor<DataType, ExecutionSpace>& Rgt)
        {
            const size_type N0 = Lft.extent(0);
            const size_type N1 = Lft.extent(1);
            const size_type N2 = Lft.extent(2);

            Matrix<DataType, ExecutionSpace> res(Lft.label() + "-" + Rgt.label(), N0, N1, N2);

            for (int i = 0; i < N0; ++i)
            {
                for (int j = 0; j < N1; ++j)
                {
                    for (int k = 0; k < N2; ++k)
                    {
                        res(i, j, k) = Lft(i, j, k) - Rgt(i, j, k);
                    }
                }
            }

            return res;
        }

        template<typename DataType, class ExecutionSpace>
        __inline static Tensor<DataType, ExecutionSpace> operator+(const Tensor<DataType, ExecutionSpace>& Lft, const DataType& Rgt)
        {
            const size_type N0 = Lft.extent(0);
            const size_type N1 = Lft.extent(1);
            const size_type N2 = Lft.extent(2);

            Tensor<DataType, ExecutionSpace> res(Lft.label() + "+" + std::to_string(Rgt), N0, N1, N2);

            for (int i = 0; i < N0; ++i)
            {
                for (int j = 0; j < N1; ++j)
                {
                    for (int k = 0; k < N2; ++k)
                    {
                        res(i, j, k) = Lft(i, j, k) + Rgt;
                    }
                }
            }

            return res;
        }

        template<typename DataType, class ExecutionSpace>
        __inline static Tensor<DataType, ExecutionSpace> operator+(const DataType& Lft, const Tensor<DataType, ExecutionSpace>& Rgt)
        {
            const size_type N0 = Rgt.extent(0);
            const size_type N1 = Rgt.extent(1);
            const size_type N2 = Rgt.extent(2);

            Tensor<DataType, ExecutionSpace> res(std::to_string(Lft) + "+" + Rgt.label(), N0, N1, N2);

            for (int i = 0; i < N0; ++i)
            {
                for (int j = 0; j < N1; ++j)
                {
                    for (int k = 0; k < N2; ++k)
                    {
                        res(i, j, k) = Lft + Rgt(i, j, k);
                    }
                }
            }

            return res;
        }

        template<typename DataType, class ExecutionSpace>
        __inline static Tensor<DataType, ExecutionSpace> operator-(const Tensor<DataType, ExecutionSpace>& Lft, const DataType& Rgt)
        {
            const size_type N0 = Lft.extent(0);
            const size_type N1 = Lft.extent(1);
            const size_type N2 = Lft.extent(2);

            Tensor<DataType, ExecutionSpace> res(Lft.label() + "-" + std::to_string(Rgt), N0, N1, N2);

            for (int i = 0; i < N0; ++i)
            {
                for (int j = 0; j < N1; ++j)
                {
                    for (int k = 0; k < N2; ++k)
                    {
                        res(i, j, k) = Lft(i, j, k) - Rgt;
                    }
                }
            }

            return res;
        }

        template<typename DataType, class ExecutionSpace>
        __inline static Tensor<DataType, ExecutionSpace> operator-(const DataType& Lft, const Tensor<DataType, ExecutionSpace>& Rgt)
        {
            const size_type N0 = Rgt.extent(0);
            const size_type N1 = Rgt.extent(1);
            const size_type N2 = Rgt.extent(2);

            Tensor<DataType, ExecutionSpace> res(std::to_string(Lft) + "-" + Rgt.label(), N0, N1, N2);

            for (int i = 0; i < N0; ++i)
            {
                for (int j = 0; j < N1; ++j)
                {
                    for (int k = 0; k < N2; ++k)
                    {
                        res(i, j, k) = Lft - Rgt(i, j, k);
                    }
                }
            }

            return res;
        }

        template<typename DataType, class ExecutionSpace>
        __inline static Tensor<DataType, ExecutionSpace> operator*(const Tensor<DataType, ExecutionSpace>& Lft, const DataType& Rgt)
        {
            const size_type N0 = Lft.extent(0);
            const size_type N1 = Lft.extent(1);
            const size_type N2 = Lft.extent(2);

            Tensor<DataType, ExecutionSpace> res(Lft.label() + "*" + std::to_string(Rgt), N0, N1, N2);

            for (int i = 0; i < N0; ++i)
            {
                for (int j = 0; j < N1; ++j)
                {
                    for (int k = 0; k < N2; ++k)
                    {
                        res(i, j, k) = Lft(i, j, k) * Rgt;
                    }
                }
            }

            return res;
        }

        template<typename DataType, class ExecutionSpace>
        __inline static Tensor<DataType, ExecutionSpace> operator*(const DataType& Lft, const Tensor<DataType, ExecutionSpace>& Rgt)
        {
            const size_type N0 = Rgt.extent(0);
            const size_type N1 = Rgt.extent(1);
            const size_type N2 = Rgt.extent(2);

            Tensor<DataType, ExecutionSpace> res(std::to_string(Lft) + "*" + Rgt.label(), N0, N1, N2);

            for (int i = 0; i < N0; ++i)
            {
                for (int j = 0; j < N1; ++j)
                {
                    for (int k = 0; k < N2; ++k)
                    {
                        res(i, j, k) = Lft * Rgt(i, j, k);
                    }
                }
            }

            return res;
        }

        template<typename DataType, class ExecutionSpace>
        __inline static Tensor<DataType, ExecutionSpace> operator/(const Tensor<DataType, ExecutionSpace>& Lft, const DataType& Rgt)
        {
            const size_type N0 = Lft.extent(0);
            const size_type N1 = Lft.extent(1);
            const size_type N2 = Lft.extent(2);

            Tensor<DataType, ExecutionSpace> res(Lft.label() + "/" + std::to_string(Rgt), N0, N1, N2);

            for (int i = 0; i < N0; ++i)
            {
                for (int j = 0; j < N1; ++j)
                {
                    for (int k = 0; k < N2; ++k)
                    {
                        res(i, j, k) = Lft(i, j, k) / Rgt;
                    }
                }
            }

            return res;
        }

        template<typename DataType, class ExecutionSpace>
        __inline static Tensor<DataType, ExecutionSpace> operator/(const DataType& Lft, const Tensor<DataType, ExecutionSpace>& Rgt)
        {
            const size_type N0 = Rgt.extent(0);
            const size_type N1 = Rgt.extent(1);
            const size_type N2 = Rgt.extent(2);

            Tensor<DataType, ExecutionSpace> res(std::to_string(Lft) + "/" + Rgt.label(), N0, N1, N2);

            for (int i = 0; i < N0; ++i)
            {
                for (int j = 0; j < N1; ++j)
                {
                    for (int k = 0; k < N2; ++k)
                    {
                        res(i, j, k) = Lft / Rgt(i, j, k);
                    }
                }
            }

            return res;
        }
    }
}

namespace Kokkos
{
    namespace Extension
    {

        template<typename DataType, class ExecutionSpace>
        __inline static Tesseract<DataType, ExecutionSpace> operator+(const Tesseract<DataType, ExecutionSpace>& Lft, const Tesseract<DataType, ExecutionSpace>& Rgt)
        {
            const size_type N0 = Lft.extent(0);
            const size_type N1 = Lft.extent(1);
            const size_type N2 = Lft.extent(2);
            const size_type N3 = Lft.extent(3);

            Matrix<DataType, ExecutionSpace> res(Lft.label() + "+" + Rgt.label(), N0, N1, N2, N3);

            for (int i = 0; i < N0; ++i)
            {
                for (int j = 0; j < N1; ++j)
                {
                    for (int k = 0; k < N2; ++k)
                    {
                        for (int l = 0; l < N3; ++l)
                        {
                            res(i, j, k, l) = Lft(i, j, k, l) + Rgt(i, j, k, l);
                        }
                    }
                }
            }

            return res;
        }

        template<typename DataType, class ExecutionSpace>
        __inline static Tesseract<DataType, ExecutionSpace> operator-(const Tesseract<DataType, ExecutionSpace>& Lft, const Tesseract<DataType, ExecutionSpace>& Rgt)
        {
            const size_type N0 = Lft.extent(0);
            const size_type N1 = Lft.extent(1);
            const size_type N2 = Lft.extent(2);
            const size_type N3 = Lft.extent(3);

            Matrix<DataType, ExecutionSpace> res(Lft.label() + "-" + Rgt.label(), N0, N1, N2, N3);

            for (int i = 0; i < N0; ++i)
            {
                for (int j = 0; j < N1; ++j)
                {
                    for (int k = 0; k < N2; ++k)
                    {
                        for (int l = 0; l < N3; ++l)
                        {
                            res(i, j, k, l) = Lft(i, j, k, l) - Rgt(i, j, k, l);
                        }
                    }
                }
            }

            return res;
        }

        template<typename DataType, class ExecutionSpace>
        __inline static Tesseract<DataType, ExecutionSpace> operator+(const Tesseract<DataType, ExecutionSpace>& Lft, const DataType& Rgt)
        {
            const size_type N0 = Lft.extent(0);
            const size_type N1 = Lft.extent(1);
            const size_type N2 = Lft.extent(2);
            const size_type N3 = Lft.extent(3);

            Tesseract<DataType, ExecutionSpace> res(Lft.label() + "+" + std::to_string(Rgt), N0, N1, N2, N3);

            for (int i = 0; i < N0; ++i)
            {
                for (int j = 0; j < N1; ++j)
                {
                    for (int k = 0; k < N2; ++k)
                    {
                        for (int l = 0; l < N3; ++l)
                        {
                            res(i, j, k, l) = Lft(i, j, k, l) + Rgt;
                        }
                    }
                }
            }

            return res;
        }

        template<typename DataType, class ExecutionSpace>
        __inline static Tesseract<DataType, ExecutionSpace> operator+(const DataType& Lft, const Tesseract<DataType, ExecutionSpace>& Rgt)
        {
            const size_type N0 = Rgt.extent(0);
            const size_type N1 = Rgt.extent(1);
            const size_type N2 = Rgt.extent(2);
            const size_type N3 = Rgt.extent(3);

            Tesseract<DataType, ExecutionSpace> res(std::to_string(Lft) + "+" + Rgt.label(), N0, N1, N2, N3);

            for (int i = 0; i < N0; ++i)
            {
                for (int j = 0; j < N1; ++j)
                {
                    for (int k = 0; k < N2; ++k)
                    {
                        for (int l = 0; l < N3; ++l)
                        {
                            res(i, j, k, l) = Lft + Rgt(i, j, k, l);
                        }
                    }
                }
            }

            return res;
        }

        template<typename DataType, class ExecutionSpace>
        __inline static Tesseract<DataType, ExecutionSpace> operator-(const Tesseract<DataType, ExecutionSpace>& Lft, const DataType& Rgt)
        {
            const size_type N0 = Lft.extent(0);
            const size_type N1 = Lft.extent(1);
            const size_type N2 = Lft.extent(2);
            const size_type N3 = Lft.extent(3);

            Tesseract<DataType, ExecutionSpace> res(Lft.label() + "-" + std::to_string(Rgt), N0, N1, N2, N3);

            for (int i = 0; i < N0; ++i)
            {
                for (int j = 0; j < N1; ++j)
                {
                    for (int k = 0; k < N2; ++k)
                    {
                        for (int l = 0; l < N3; ++l)
                        {
                            res(i, j, k, l) = Lft(i, j, k, l) - Rgt;
                        }
                    }
                }
            }

            return res;
        }

        template<typename DataType, class ExecutionSpace>
        __inline static Tesseract<DataType, ExecutionSpace> operator-(const DataType& Lft, const Tesseract<DataType, ExecutionSpace>& Rgt)
        {
            const size_type N0 = Rgt.extent(0);
            const size_type N1 = Rgt.extent(1);
            const size_type N2 = Rgt.extent(2);
            const size_type N3 = Rgt.extent(3);

            Tesseract<DataType, ExecutionSpace> res(std::to_string(Lft) + "-" + Rgt.label(), N0, N1, N2, N3);

            for (int i = 0; i < N0; ++i)
            {
                for (int j = 0; j < N1; ++j)
                {
                    for (int k = 0; k < N2; ++k)
                    {
                        for (int l = 0; l < N3; ++l)
                        {
                            res(i, j, k, l) = Lft - Rgt(i, j, k, l);
                        }
                    }
                }
            }

            return res;
        }

        template<typename DataType, class ExecutionSpace>
        __inline static Tesseract<DataType, ExecutionSpace> operator*(const Tesseract<DataType, ExecutionSpace>& Lft, const DataType& Rgt)
        {
            const size_type N0 = Lft.extent(0);
            const size_type N1 = Lft.extent(1);
            const size_type N2 = Lft.extent(2);
            const size_type N3 = Lft.extent(3);

            Tesseract<DataType, ExecutionSpace> res(Lft.label() + "*" + std::to_string(Rgt), N0, N1, N2, N3);

            for (int i = 0; i < N0; ++i)
            {
                for (int j = 0; j < N1; ++j)
                {
                    for (int k = 0; k < N2; ++k)
                    {
                        for (int l = 0; l < N3; ++l)
                        {
                            res(i, j, k, l) = Lft(i, j, k, l) * Rgt;
                        }
                    }
                }
            }

            return res;
        }

        template<typename DataType, class ExecutionSpace>
        __inline static Tesseract<DataType, ExecutionSpace> operator*(const DataType& Lft, const Tesseract<DataType, ExecutionSpace>& Rgt)
        {
            const size_type N0 = Rgt.extent(0);
            const size_type N1 = Rgt.extent(1);
            const size_type N2 = Rgt.extent(2);
            const size_type N3 = Rgt.extent(3);

            Tesseract<DataType, ExecutionSpace> res(std::to_string(Lft) + "*" + Rgt.label(), N0, N1, N2, N3);

            for (int i = 0; i < N0; ++i)
            {
                for (int j = 0; j < N1; ++j)
                {
                    for (int k = 0; k < N2; ++k)
                    {
                        for (int l = 0; l < N3; ++l)
                        {
                            res(i, j, k, l) = Lft * Rgt(i, j, k, l);
                        }
                    }
                }
            }

            return res;
        }

        template<typename DataType, class ExecutionSpace>
        __inline static Tesseract<DataType, ExecutionSpace> operator/(const Tesseract<DataType, ExecutionSpace>& Lft, const DataType& Rgt)
        {
            const size_type N0 = Lft.extent(0);
            const size_type N1 = Lft.extent(1);
            const size_type N2 = Lft.extent(2);
            const size_type N3 = Lft.extent(3);

            Tesseract<DataType, ExecutionSpace> res(Lft.label() + "/" + std::to_string(Rgt), N0, N1, N2, N3);

            for (int i = 0; i < N0; ++i)
            {
                for (int j = 0; j < N1; ++j)
                {
                    for (int k = 0; k < N2; ++k)
                    {
                        for (int l = 0; l < N3; ++l)
                        {
                            res(i, j, k, l) = Lft(i, j, k, l) / Rgt;
                        }
                    }
                }
            }

            return res;
        }

        template<typename DataType, class ExecutionSpace>
        __inline static Tesseract<DataType, ExecutionSpace> operator/(const DataType& Lft, const Tesseract<DataType, ExecutionSpace>& Rgt)
        {
            const size_type N0 = Rgt.extent(0);
            const size_type N1 = Rgt.extent(1);
            const size_type N2 = Rgt.extent(2);
            const size_type N3 = Rgt.extent(3);

            Tesseract<DataType, ExecutionSpace> res(std::to_string(Lft) + "/" + Rgt.label(), N0, N1, N2, N3);

            for (int i = 0; i < N0; ++i)
            {
                for (int j = 0; j < N1; ++j)
                {
                    for (int k = 0; k < N2; ++k)
                    {
                        for (int l = 0; l < N3; ++l)
                        {
                            res(i, j, k, l) = Lft / Rgt(i, j, k, l);
                        }
                    }
                }
            }

            return res;
        }
    }
}

namespace Kokkos
{
    namespace Extension
    {

        template<typename DataType, class ExecutionSpace>
        __inline static DataType operator*(const Vector<DataType, ExecutionSpace>& Lft, const Vector<DataType, ExecutionSpace>& Rgt)
        {
            const size_type N0 = Rgt.extent(0);

            DataType res = static_cast<DataType>(0);

            for (int i = 0; i < N0; ++i)
            {
                res += Lft(i) * Rgt(i);
            }

            return res;
        }

        template<typename DataType, class ExecutionSpace>
        __inline static Vector<DataType, ExecutionSpace> operator*(const Vector<DataType, ExecutionSpace>& Lft, const Matrix<DataType, ExecutionSpace>& Rgt)
        {
            const size_type N0 = Rgt.nrows();
            const size_type N1 = Rgt.ncolumns();

            Vector<DataType, ExecutionSpace> res(Lft.label() + "*" + Rgt.label(), N1);

            for (int j = 0; j < N1; ++j)
            {
                for (int i = 0; i < N0; ++i)
                {
                    res(j) += Lft(i) * Rgt(i, j);
                }
            }

            return res;
        }

        template<typename DataType, class ExecutionSpace>
        __inline static Matrix<DataType, ExecutionSpace> operator*(const Vector<DataType, ExecutionSpace>& Lft, const Tensor<DataType, ExecutionSpace>& Rgt)
        {
            const size_type N0 = Rgt.extent(0);
            const size_type N1 = Rgt.extent(1);
            const size_type N2 = Rgt.extent(2);

            Matrix<DataType, ExecutionSpace> res(Lft.label() + "*" + Rgt.label(), N2, N1);

            for (int j = 0; j < N2; ++j)
            {
                for (int k = 0; k < N1; ++k)
                {
                    for (int i = 0; i < N0; ++i)
                    {
                        res(j, k) += Lft(i) * Rgt(i, j, k);
                    }
                }
            }

            return res;
        }

        template<typename DataType, class ExecutionSpace>
        __inline static Tensor<DataType, ExecutionSpace> operator*(const Vector<DataType, ExecutionSpace>& Lft, const Tesseract<DataType, ExecutionSpace>& Rgt)
        {
            const size_type N0 = Rgt.extent(0);
            const size_type N1 = Rgt.extent(1);
            const size_type N2 = Rgt.extent(2);
            const size_type N3 = Rgt.extent(3);

            Tensor<DataType, ExecutionSpace> res(Lft.label() + "*" + Rgt.label(), N1, N2, N3);

            for (int j = 0; j < N1; ++j)
            {
                for (int k = 0; k < N2; ++k)
                {
                    for (int l = 0; l < N3; ++l)
                    {
                        for (int i = 0; i < N0; ++i)
                        {
                            res(j, k, l) += Lft(i) * Rgt(i, j, k, l);
                        }
                    }
                }
            }
            return res;
        }

        // template<typename DataType, class ExecutionSpace>
        //__inline static Vector<DataType, ExecutionSpace> operator*(const Matrix<DataType, ExecutionSpace>& Lft, const Vector<DataType, ExecutionSpace>& Rgt)
        //{
        //    Vector<DataType, ExecutionSpace> res;
        //    for (int i = 0; i < Dim; ++i)
        //    {
        //        for (int j = 0; j < Dim; ++j)
        //        {
        //            res(i) = res(i) + Lft(i, j) * Rgt(j);
        //        }
        //    }
        //    return res;
        //}

        // template<typename DataType, class ExecutionSpace>
        //__inline static Matrix<DataType, ExecutionSpace> operator*(const Matrix<DataType, ExecutionSpace>& Lft, const Matrix<DataType, ExecutionSpace>& Rgt)
        //{
        //    Matrix<DataType, ExecutionSpace> res;
        //    for (int i = 0; i < Dim; ++i)
        //    {
        //        for (int k = 0; k < Dim; ++k)
        //        {
        //            for (int j = 0; j < Dim; ++j)
        //            {
        //                res(i, k) = res(i, k) + Lft(i, j) * Rgt(j, k);
        //            }
        //        }
        //    }
        //    return res;
        //}

        template<typename DataType, class ExecutionSpace>
        __inline static Tensor<DataType, ExecutionSpace> operator*(const Matrix<DataType, ExecutionSpace>& Lft, const Tensor<DataType, ExecutionSpace>& Rgt)
        {
            const size_type N0 = Rgt.extent(0);
            const size_type N1 = Rgt.extent(1);
            const size_type N2 = Rgt.extent(2);

            const size_type N3 = Lft.extent(0);

            Tensor<DataType, ExecutionSpace> res(Lft.label() + "*" + Rgt.label(), N3, N1, N2);

            for (int i = 0; i < N3; ++i)
            {
                for (int k = 0; k < N1; ++k)
                {
                    for (int l = 0; l < N2; ++l)
                    {
                        for (int j = 0; j < N0; ++j)
                        {
                            res(i, k, l) += Lft(i, j) * Rgt(j, k, l);
                        }
                    }
                }
            }
            return res;
        }

        template<typename DataType, class ExecutionSpace>
        __inline static Tesseract<DataType, ExecutionSpace> operator*(const Matrix<DataType, ExecutionSpace>& Lft, const Tesseract<DataType, ExecutionSpace>& Rgt)
        {
            const size_type N0 = Rgt.extent(0);
            const size_type N1 = Rgt.extent(1);
            const size_type N2 = Rgt.extent(2);
            const size_type N3 = Rgt.extent(3);

            const size_type N4 = Lft.extent(0);

            Tesseract<DataType, ExecutionSpace> res(Lft.label() + "*" + Rgt.label(), N4, N1, N2, N3);

            for (int i = 0; i < N4; ++i)
            {
                for (int k = 0; k < N1; ++k)
                {
                    for (int l = 0; l < N2; ++l)
                    {
                        for (int m = 0; m < N3; ++m)
                        {
                            for (int j = 0; j < N0; ++j)
                            {
                                res(i, k, l, m) += Lft(i, j) * Rgt(j, k, l, m);
                            }
                        }
                    }
                }
            }

            return res;
        }

        template<typename DataType, class ExecutionSpace>
        __inline static Matrix<DataType, ExecutionSpace> operator*(const Tensor<DataType, ExecutionSpace>& Lft, const Vector<DataType, ExecutionSpace>& Rgt)
        {
            const size_type N0 = Lft.extent(0);
            const size_type N1 = Lft.extent(1);
            const size_type N2 = Lft.extent(2);

            Matrix<DataType, ExecutionSpace> res(Lft.label() + "*" + Rgt.label(), N0, N1);

            for (int i = 0; i < N0; ++i)
            {
                for (int j = 0; j < N1; ++j)
                {
                    for (int k = 0; k < N2; ++k)
                    {
                        res(i, j) += Lft(i, j, k) * Rgt(k);
                    }
                }
            }

            return res;
        }

        template<typename DataType, class ExecutionSpace>
        __inline static Tensor<DataType, ExecutionSpace> operator*(const Tensor<DataType, ExecutionSpace>& Lft, const Matrix<DataType, ExecutionSpace>& Rgt)
        {
            const size_type N0 = Lft.extent(0);
            const size_type N1 = Lft.extent(1);
            const size_type N2 = Lft.extent(2);

            const size_type N3 = Rgt.extent(1);

            Tensor<DataType, ExecutionSpace> res(Lft.label() + "*" + Rgt.label(), N0, N1, N3);

            for (int i = 0; i < Lft; ++i)
            {
                for (int j = 0; j < N1; ++j)
                {
                    for (int l = 0; l < N3; ++l)
                    {
                        for (int k = 0; k < N2; ++k)
                        {
                            res(i, j, l) += Lft(i, j, k) * Rgt(k, l);
                        }
                    }
                }
            }

            return res;
        }

        template<typename DataType, class ExecutionSpace>
        __inline static Tesseract<DataType, ExecutionSpace> operator*(const Tensor<DataType, ExecutionSpace>& Lft, const Tensor<DataType, ExecutionSpace>& Rgt)
        {
            const size_type N0 = Lft.extent(0);
            const size_type N1 = Lft.extent(1);
            const size_type N2 = Lft.extent(2);

            const size_type N3 = Rgt.extent(1);
            const size_type N4 = Rgt.extent(2);

            Tesseract<DataType, ExecutionSpace> res(Lft.label() + "*" + Rgt.label(), N0, N1, N3, N4);

            for (int i = 0; i < N0; ++i)
            {
                for (int j = 0; j < N1; ++j)
                {
                    for (int l = 0; l < N3; ++l)
                    {
                        for (int m = 0; m < N4; ++m)
                        {
                            for (int k = 0; k < N2; ++k)
                            {
                                res(i, j, l, m) += Lft(i, j, k) * Rgt(k, l, m);
                            }
                        }
                    }
                }
            }

            return res;
        }

        template<typename DataType, class ExecutionSpace>
        __inline static Tensor<DataType, ExecutionSpace> operator*(const Tesseract<DataType, ExecutionSpace>& Lft, const Vector<DataType, ExecutionSpace>& Rgt)
        {
            const size_type N0 = Lft.extent(0);
            const size_type N1 = Lft.extent(1);
            const size_type N2 = Lft.extent(2);
            const size_type N3 = Lft.extent(3);

            Tensor<DataType, ExecutionSpace> res(Lft.label() + "*" + Rgt.label(), N0, N1, N2);

            for (int i = 0; i < N0; ++i)
            {
                for (int j = 0; j < N1; ++j)
                {
                    for (int k = 0; k < N2; ++k)
                    {
                        for (int l = 0; l < N3; ++l)
                        {
                            res(i, j, k) += Lft(i, j, k, l) * Rgt(l);
                        }
                    }
                }
            }

            return res;
        }

        template<typename DataType, class ExecutionSpace>
        __inline static Tesseract<DataType, ExecutionSpace> operator*(const Tesseract<DataType, ExecutionSpace>& Lft, const Matrix<DataType, ExecutionSpace>& Rgt)
        {
            const size_type N0 = Lft.extent(0);
            const size_type N1 = Lft.extent(1);
            const size_type N2 = Lft.extent(2);
            const size_type N3 = Lft.extent(3);

            const size_type N4 = Rgt.extent(1);

            Tesseract<DataType, ExecutionSpace> res(Lft.label() + "*" + Rgt.label(), N0, N1, N2, N4);

            for (int i = 0; i < N0; ++i)
            {
                for (int j = 0; j < N1; ++j)
                {
                    for (int k = 0; k < N2; ++k)
                    {
                        for (int m = 0; m < N4; ++m)
                        {
                            for (int l = 0; l < N3; ++l)
                            {
                                res(i, j, k, m) += Lft(i, j, k, l) * Rgt(l, m);
                            }
                        }
                    }
                }
            }

            return res;
        }

        //// -------------------------------------------------------- operator %  =>  double-dot

        //// 2.#)

        //// 2.2) r = sum(i=1,Dim) sum(j=1,Dim) { L(i, j)*R(i, j) }         scalar = T2 d-dot T2
        // template<typename DataType, class ExecutionSpace>
        //__inline static DataType operator%(const Matrix<DataType, ExecutionSpace>& Lft, const Matrix<DataType, ExecutionSpace>& Rgt)
        //{
        //    DataType res = static_cast<DataType>(0);
        //    for (int i = 0; i < Dim; ++i)
        //        for (int j = 0; j < Dim; ++j)
        //            res = res + Lft(i, j) * Rgt(i, j);
        //    return res;
        //}

        //// 2.3) R(k) = sum(i=1,Dim) sum(j=1,Dim) { L(i, j)*R(i, j, k) }       T1 = T2 d-dot T3
        // template<typename DataType, class ExecutionSpace>
        //__inline static Vector<DataType, ExecutionSpace> operator%(const Matrix<DataType, ExecutionSpace>& Lft, const Tensor<DataType, ExecutionSpace>& Rgt)
        //{
        //    Vector<DataType, ExecutionSpace> res;
        //    for (int k = 0; k < Dim; ++k)
        //        for (int i = 0; i < Dim; ++i)
        //            for (int j = 0; j < Dim; ++j)
        //                res(k) = res(k) + Lft(i, j) * Rgt(i, j, k);
        //    return res;
        //}

        //// 2.4) R(k, l) = sum(i=1,Dim) sum(j=1,Dim) { L(i, j)*R(i, j, k, l) } T2 = T2 d-dot T4
        // template<typename DataType, class ExecutionSpace>
        //__inline static Matrix<DataType, ExecutionSpace> operator%(const Matrix<DataType, ExecutionSpace>& Lft, const Tesseract<DataType, ExecutionSpace>& Rgt)
        //{
        //    Matrix<DataType, ExecutionSpace> res;
        //    for (int k = 0; k < Dim; ++k)
        //        for (int l = 0; l < Dim; ++l)
        //            for (int i = 0; i < Dim; ++i)
        //                for (int j = 0; j < Dim; ++j)
        //                    res(k, l) = res(k, l) + Lft(i, j) * Rgt(i, j, k, l);
        //    return res;
        //}

        //// 3.#)

        //// 3.2) R(i) = sum(j=1,Dim) sum(k=1,Dim) { L(i, j, k)*R(j, k) }       T1 = T3 d-dot T2
        // template<typename DataType, class ExecutionSpace>
        //__inline static Vector<DataType, ExecutionSpace> operator%(const Tensor<DataType, ExecutionSpace>& Lft, const Matrix<DataType, ExecutionSpace>& Rgt)
        //{
        //    Vector<DataType, ExecutionSpace> res;
        //    for (int i = 0; i < Dim; ++i)
        //        for (int j = 0; j < Dim; ++j)
        //            for (int k = 0; k < Dim; ++k)
        //                res(i) = res(i) + Lft(i, j, k) * Rgt(j, k);
        //    return res;
        //}

        //// 3.3) R(i, l) = sum(j=1,Dim) sum(k=1,Dim) { L(i, j, k)*R(j, k, l) }
        ////                                                                    T2 = T3 d-dot T3
        // template<typename DataType, class ExecutionSpace>
        //__inline static Matrix<DataType, ExecutionSpace> operator%(const Tensor<DataType, ExecutionSpace>& Lft, const Tensor<DataType, ExecutionSpace>& Rgt)
        //{
        //    Matrix<DataType, ExecutionSpace> res;
        //    for (int i = 0; i < Dim; ++i)
        //        for (int l = 0; l < Dim; ++l)
        //            for (int j = 0; j < Dim; ++j)
        //                for (int k = 0; k < Dim; ++k)
        //                    res(i, l) = res(i, l) + Lft(i, j, k) * Rgt(j, k, l);
        //    return res;
        //}

        //// 3.4) R(i, l, m) = sum(j=1,Dim) sum(k=1,Dim) { L(i, j, k)*R(j, k, l, m) }
        ////                                                                    T3 = T3 d-dot T4
        // template<typename DataType, class ExecutionSpace>
        //__inline static Tensor<DataType, ExecutionSpace> operator%(const Tensor<DataType, ExecutionSpace>& Lft, const Tesseract<DataType, ExecutionSpace>& Rgt)
        //{
        //    Tensor<DataType, ExecutionSpace> res;
        //    for (int i = 0; i < Dim; ++i)
        //        for (int l = 0; l < Dim; ++l)
        //            for (int m = 0; m < Dim; ++m)
        //                for (int j = 0; j < Dim; ++j)
        //                    for (int k = 0; k < Dim; ++k)
        //                        res(i, l, m) = res(i, l, m) + Lft(i, j, k) * Rgt(j, k, l, m);
        //    return res;
        //}

        //// 4.#)

        //// 4.2) R(i, j) = sum(k=1,Dim) sum(l=1,Dim){ L(i, j, k, l) * R(k, l) }
        ////                                                                    T2 = T4 d-dot T2
        // template<typename DataType, class ExecutionSpace>
        //__inline static Matrix<DataType, ExecutionSpace> operator%(const Tesseract<DataType, ExecutionSpace>& Lft, const Matrix<DataType, ExecutionSpace>& Rgt)
        //{
        //    Matrix<DataType, ExecutionSpace> res;
        //    for (int i = 0; i < Dim; ++i)
        //        for (int j = 0; j < Dim; ++j)
        //            for (int k = 0; k < Dim; ++k)
        //                for (int l = 0; l < Dim; ++l)
        //                    res(i, j) = res(i, j) + Lft(i, j, k, l) * Rgt(k, l);
        //    return res;
        //}

        //// 4.3) R(i, j, m) = sum(k=1,Dim) sum(l=1,Dim){ L(i, j, k, l) * R(k, l, m) }
        ////                                                                    T3 = T4 d-dot T3
        // template<typename DataType, class ExecutionSpace>
        //__inline static Tensor<DataType, ExecutionSpace> operator%(const Tesseract<DataType, ExecutionSpace>& Lft, const Tensor<DataType, ExecutionSpace>& Rgt)
        //{
        //    Tensor<DataType, ExecutionSpace> res;
        //    for (int i = 0; i < Dim; ++i)
        //        for (int j = 0; j < Dim; ++j)
        //            for (int m = 0; m < Dim; ++m)
        //                for (int k = 0; k < Dim; ++k)
        //                    for (int l = 0; l < Dim; ++l)
        //                        res(i, j, m) = res(i, j, m) + Lft(i, j, k, l) * Rgt(k, l, m);
        //    return res;
        //}

        //// 4.4) R(i, j, m, n) = sum(k=1,Dim)sum(l=1,Dim){L(i, j, k, l)*R(k, l, m, n)}
        ////                                                                    T4 = T4 d-dot T4
        // template<typename DataType, class ExecutionSpace>
        //__inline static Tesseract<DataType, ExecutionSpace> operator%(const Tesseract<DataType, ExecutionSpace>& Lft, const Tesseract<DataType, ExecutionSpace>& Rgt)
        //{
        //    Tesseract<DataType, ExecutionSpace> res;
        //    for (int i = 0; i < Dim; ++i)
        //        for (int j = 0; j < Dim; ++j)
        //            for (int m = 0; m < Dim; ++m)
        //                for (int n = 0; n < Dim; ++n)
        //                    for (int k = 0; k < Dim; ++k)
        //                        for (int l = 0; l < Dim; ++l)
        //                            res(i, j, m, n) = res(i, j, m, n) + Lft(i, j, k, l) * Rgt(k, l, m, n);
        //    return res;
        //}

        //// ------------------------------------------------------------ operator &  =>  dyadic

        //// 1.#)

        //// 1.1) R(i, j) = L(i)*R(j)                                              T2 = T1 dy T1
        // template<typename DataType, class ExecutionSpace>
        //__inline static Matrix<DataType, ExecutionSpace> operator&(const Vector<DataType, ExecutionSpace>& Lft, const Vector<DataType, ExecutionSpace>& Rgt)
        //{
        //    Matrix<DataType, ExecutionSpace> res;
        //    for (int i = 0; i < Dim; ++i)
        //        for (int j = 0; j < Dim; ++j)
        //            res(i, j) = Lft(i) * Rgt(j);
        //    return res;
        //}

        //// 1.2) R(i, j, k) = L(i)*R(j, k)                                        T3 = T1 dy T2
        // template<typename DataType, class ExecutionSpace>
        //__inline static Tensor<DataType, ExecutionSpace> operator&(const Vector<DataType, ExecutionSpace>& Lft, const Matrix<DataType, ExecutionSpace>& Rgt)
        //{
        //    Tensor<DataType, ExecutionSpace> res;
        //    for (int i = 0; i < Dim; ++i)
        //        for (int j = 0; j < Dim; ++j)
        //            for (int k = 0; k < Dim; ++k)
        //                res(i, j, k) = Lft(i) * Rgt(j, k);
        //    return res;
        //}

        //// 1.3) R(i, j, k, l) = L(i)*R(j, k, l)                                  T4 = T1 dy T3
        // template<typename DataType, class ExecutionSpace>
        //__inline static Tesseract<DataType, ExecutionSpace> operator&(const Vector<DataType, ExecutionSpace>& Lft, const Tensor<DataType, ExecutionSpace>& Rgt)
        //{
        //    Tesseract<DataType, ExecutionSpace> res;
        //    for (int i = 0; i < Dim; ++i)
        //        for (int j = 0; j < Dim; ++j)
        //            for (int k = 0; k < Dim; ++k)
        //                for (int l = 0; l < Dim; ++l)
        //                    res(i, j, k, l) = Lft(i) * Rgt(j, k, l);
        //    return res;
        //}

        //// 2.#)

        //// 2.1) R(i, j, k) = L(i, j)*R(k)                                        T3 = T2 dy T1
        // template<typename DataType, class ExecutionSpace>
        //__inline static Tensor<DataType, ExecutionSpace> operator&(const Matrix<DataType, ExecutionSpace>& Lft, const Vector<DataType, ExecutionSpace>& Rgt)
        //{
        //    Tensor<DataType, ExecutionSpace> res;
        //    for (int i = 0; i < Dim; ++i)
        //        for (int j = 0; j < Dim; ++j)
        //            for (int k = 0; k < Dim; ++k)
        //                res(i, j, k) = Lft(i, j) * Rgt(k);
        //    return res;
        //}

        //// 2.2) R(i, j, k, l) = L(i, j)*R(k, l)                                  T4 = T2 dy T2
        // template<typename DataType, class ExecutionSpace>
        // Tesseract<DataType, ExecutionSpace> operator&(const Matrix<DataType, ExecutionSpace>& Lft, const Matrix<DataType, ExecutionSpace>& Rgt)
        //{
        //    Tesseract<DataType, ExecutionSpace> res;
        //    for (int i = 0; i < Dim; ++i)
        //        for (int j = 0; j < Dim; ++j)
        //            for (int k = 0; k < Dim; ++k)
        //                for (int l = 0; l < Dim; ++l)
        //                    res(i, j, k, l) = Lft(i, j) * Rgt(k, l);
        //    return res;
        //}

        //// ------------------------------------------------------- operator ^  =>  leaf-dyadic

        //// 2.#)

        //// 2.2) R(i, j, k, l) = L(i, k)*R(j, l)                             T4 = T2 leaf-dy T2
        // template<typename DataType, class ExecutionSpace>
        //__inline static __inline static Tesseract<DataType, ExecutionSpace> operator^(const Matrix<DataType, ExecutionSpace>& Lft, const Matrix<DataType, ExecutionSpace>& Rgt)
        //{
        //    Tesseract<DataType, ExecutionSpace> res;
        //    for (int i = 0; i < Dim; ++i)
        //        for (int j = 0; j < Dim; ++j)
        //            for (int k = 0; k < Dim; ++k)
        //                for (int l = 0; l < Dim; ++l)
        //                    res(i, j, k, l) = Lft(i, k) * Rgt(j, l);
        //    return res;
        //}

        //// ------------------------------------------------------- operator |  =>  palm-dyadic

        //// 2.#)

        //// 2.2) R(i, j, k, l) = L(i, l)*R(j, k)                             T4 = T2 palm-dy T2
        // template<typename DataType, class ExecutionSpace>
        //__inline static Tesseract<DataType, ExecutionSpace> operator|(const Matrix<DataType, ExecutionSpace>& Lft, const Matrix<DataType, ExecutionSpace>& Rgt)
        //{
        //    Tesseract<DataType, ExecutionSpace> res;
        //    for (int i = 0; i < Dim; ++i)
        //        for (int j = 0; j < Dim; ++j)
        //            for (int k = 0; k < Dim; ++k)
        //                for (int l = 0; l < Dim; ++l)
        //                    res(i, j, k, l) = Lft(i, l) * Rgt(j, k);
        //    return res;
        //}

        ////////////////////////////////////////////////////////////////////////////// Functions

        ///** Trace. */
        // template<typename DataType, class ExecutionSpace>
        //__inline static DataType Tr(Matrix<DataType, ExecutionSpace> const& A)
        //{
        //    DataType res = static_cast<DataType>(0);
        //    for (int i = 0; i < Dim; ++i)
        //        for (int j = 0; j < Dim; ++j)
        //        {
        //            if (i == j)
        //                res += A(i, j);
        //        }
        //    return res;
        //}

        ///** Characteristic invariants. */
        // template<typename DataType, class ExecutionSpace>
        //__inline static void CharInvs(Matrix<DataType, ExecutionSpace> const& A, DataType& Ia, DataType& IIa, DataType& IIIa)
        //{
        //    // characteristic invariants
        //    DataType two   = static_cast<DataType>(2);
        //    DataType three = static_cast<DataType>(3);
        //    Ia             = Tr(A);
        //    IIa            = (Ia * Ia - Tr(A * A)) / two;
        //    IIIa           = (Tr(A * A * A) - three * Tr(A * A) * Ia / two + Ia * Ia * Ia / two) / three;
        //}

        ///** Inverse. */
        // template<typename DataType, class ExecutionSpace>
        //__inline static void Inv(Matrix<DataType, ExecutionSpace> const& A, Matrix<DataType, ExecutionSpace>& Ai)
        //{
        //    // characteristic invariants
        //    DataType Ia, IIa, IIIa;
        //    CharInvs(A, Ia, IIa, IIIa);

        //    // identity
        //    Matrix<DataType, ExecutionSpace> I;
        //    I.SetKronecker();

        //    // inverse (should check |IIIa|>0)
        //    Ai = (A * A - A * Ia + I * IIa) / IIIa; // TODO: Check if transpose is necessary
        //}

    }
}
