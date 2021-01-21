using System;
using System.Runtime.CompilerServices;

namespace Kokkos.Tests
{
    public class VectorTests
    {
        private test_vector_combinations(uint size)
        {
            reference = run_me<std::vector<double>>(size);
            result    = run_me<Kokkos::vector<double, Device>>(size);
        }

        private TEST(TEST_CATEGORY,
                     vector_combination)
        {
            test_vector_allocate<int, TEST_EXECSPACE>(10);
            test_vector_combinations<int, TEST_EXECSPACE>(10);
            test_vector_combinations<int, TEST_EXECSPACE>(3057);
        }

        private TEST(TEST_CATEGORY,
                     vector_insert)
        {
            Impl::test_vector_insert<int, TEST_EXECSPACE>(3057);
        }

        public void Run()
        {
            Vector<double, Cuda> a = new Vector<double, Cuda>();

            test_vector_insert(ref a);
        }

        public void test_vector_insert(ref Vector<double> a)
        {
            unsafe
            {
                View<double, Cuda> view = new View<double, Cuda>("CreateScalar");

                ulong n = a.size();

                IntPtr it = a.begin();

                if(n > 0)
                {
                    Assert.AreEqual(a.data(), Unsafe.AsPointer(ref a[0]));
                }

                it += 15;

                Assert.AreEqual(Unsafe.Read<double>(it), 1.0);

                IntPtr it_return = a.insert(it, 3.0);
                Assert.AreEqual(a.size(), n + 1);
                //Assert.AreEqual(std::distance(it_return, a.begin() + 15), 0);

                it =  a.begin();
                it += 17;

                it_return = a.insert(it, n + 5, 5.0);

                it_return = a.insert(it, n + 5, 5.0);

                Assert.AreEqual(a.size(), n + 1 + n + 5);
                //Assert.AreEqual(std::distance(it_return, a.begin() + 17), 0);

                Vector<double> b = new Vector<double>();

                b.insert(b.begin(), 7, 9);

                Assert.AreEqual(b.size(), 7);
                Assert.AreEqual(b[0],     9.0);

                it = a.begin();

                Unsafe.Add(it, 27 + n);

                it_return = a.insert(it, b.begin(), b.end());

                Assert.AreEqual(a.size(), n + 1 + n + 5 + 7);
                //Assert.AreEqual(std::distance(it_return, a.begin() + 27 + n), 0);

                // Testing insert at end via all three function interfaces
                a.insert(a.end(), 11);

                a.insert(a.end(), 2, 12);

                //a.insert(a.end(), b.begin(), b.end());
            }
        }

        private void check_test(ref Vector<double> a,
                                int                n)
        {
            for(int i = 0; i < (int)a.size(); i++)
            {
                if(i == 15)
                {
                    Assert.AreEqual(a[i], 3.0);
                }
                else if(i > 16 && i < 16 + 6 + n)
                {
                    Assert.AreEqual(a[i], 5.0);
                }
                else if(i > 26 + n && i < 34 + n)
                {
                    Assert.AreEqual(a[i], 9.0);
                }
                else if(i == (int)a.size() - 10)
                {
                    Assert.AreEqual(a[i], 11.0);
                }
                else if(i == (int)a.size() - 9 || i == (int)a.size() - 8)
                {
                    Assert.AreEqual(a[i], 12.0);
                }
                else if(i > (int)a.size() - 8)
                {
                    Assert.AreEqual(a[i], 9.0);
                }
                else
                {
                    Assert.AreEqual(a[i], 1.0);
                }
            }
        }

        private void test_vector_insert(uint size)
        {
            //{
            //    std::vector<double> a(size, (1.0));
            //    run_test(a);
            //    check_test(a, size);
            //}
            {
                Vector<double> a = new Vector<double>(size, 1.0);
                a.sync_device();
                run_test(a);
                a.sync_host();
                check_test(a, size);
            }

            {
                Vector<double> a = new Vector<double>(size, 1.0);
                a.sync_host();
                run_test(a);
                check_test(a, size);
            }
        }

        //double test_vector_allocate (uint n) {
        //    {
        //        Vector v1;
        //        if (v1.is_allocated() == true) return false;

        //        v1 = Vector(n, 1);
        //        Vector v2(v1);
        //        Vector v3(n, 1);

        //        if (v1.is_allocated() == false) return false;
        //        if (v2.is_allocated() == false) return false;
        //        if (v3.is_allocated() == false) return false;
        //    }
        //    return true;
        //}

        private void test_vector_allocate(uint size)
        {
            result = run_me<Kokkos::vector<double, Device>>(size);
        }

        private double test_vector_combinations(uint n)
        {
            Vector<double> a = new Vector<double>(n, 1);

            a.push_back(2);
            a.resize(n + 4);
            a[n + 1] = 3;
            a[n + 2] = 4;
            a[n + 3] = 5;

            double temp1 = a[2];
            double temp2 = a[n];
            double temp3 = a[n + 1];

            a.assign(n + 2, -1);

            a[2]     = temp1;
            a[n]     = temp2;
            a[n + 1] = temp3;

            double test1 = 0;

            for(uint i = 0; i < a.size(); i++)
            {
                test1 += a[i];
            }

            a.assign(n + 1, -2);
            double test2 = 0;

            for(uint i = 0; i < a.size(); i++)
            {
                test2 += a[i];
            }

            a.reserve(n + 10);

            double test3 = 0;

            for(uint i = 0; i < a.size(); i++)
            {
                test3 += a[i];
            }

            return (test1 * test2 + test3) * test2 + test1 * test3;
        }

        //private void test_vector_combinations(uint size)
        //{
        //    Impl::test_vector_combinations<double, Device> test(size);
        //    ASSERT_EQ(test.reference, test.result);
        //}
        //private void test_vector_allocate(uint size)
        //{
        //    Impl::test_vector_allocate<double, Device> test(size);
        //    ASSERT_TRUE(test.result);
        //}
    }
}