using System;
using System.Transactions;

namespace Kokkos.Tests
{
    public class ViewTests<TExecutionSpace>
        where TExecutionSpace : IExecutionSpace, new()
    {
        public void Run()
        {
            int  num_threads      = 4;
            int  num_numa         = -1;
            int  device_id        = 0;
            int  ndevices         = 1;
            int  skip_device      = 9999;
            bool disable_warnings = false;

            InitArguments arguments = new(num_threads, num_numa, device_id, ndevices, skip_device, disable_warnings);

            using(ScopeGuard.Get(arguments))
            {
                CreateScalar();
                CreateVector();
                CreateMatrix();
                CreateTensor();

                CreateN4();
                CreateN5();
                CreateN6();
                CreateN7();
                CreateN8();

                CopyToVector();

                GetSetValueVector();

                Convert();
            }
        }

        public void CopyToVector()
        {
            View<double, TExecutionSpace> view = new("CreateVector", 3);

            double[] values =
            {
                101684.65, 651681.26, 987612.20
            };

            view.CopyTo(values);

            Assert.IsTrue(Math.Abs(view[0] - values[0]) <= double.Epsilon);

            Assert.IsTrue(Math.Abs(view[1] - values[1]) <= double.Epsilon);

            Assert.IsTrue(Math.Abs(view[2] - values[2]) <= double.Epsilon);
        }

        public void GetSetValueVector()
        {
            View<double, TExecutionSpace> view = new("CreateVector", 2);

            view[0] = 1321.258;
            view[1] = 123123.12;

            double value0 = view[0];
            double value1 = view[1];

            Assert.IsTrue(Math.Abs(value0 - 1321.258) <= double.Epsilon);

            Assert.IsTrue(Math.Abs(value1 - 123123.12) <= double.Epsilon);
        }

        public void Convert()
        {
            View<double, TExecutionSpace> view = new("CreateVector", 2);

            view[0] = 1321.258;
            view[1] = 123123.12;

            NdArray viewNdArray = View<double, TExecutionSpace>.RcpConvert(view.Pointer, 1);

            View<double, TExecutionSpace> cachedView = new(new NativePointer(view.Pointer, sizeof(double) * viewNdArray.Extent(0)), viewNdArray);

            double viewValue0 = view[0];
            double viewValue1 = view[1];

            double cachedViewValue0 = cachedView[0];
            double cachedViewValue1 = cachedView[1];

            Assert.IsTrue(Math.Abs(viewValue0 - cachedViewValue0) <= double.Epsilon);

            Assert.IsTrue(Math.Abs(viewValue1 - cachedViewValue1) <= double.Epsilon);
        }

        #region Create Tests

        public void CreateScalar()
        {
            View<double, TExecutionSpace> view = new("CreateScalar");

            Assert.AreEqual(view.Label(), "CreateScalar");

            Assert.AreEqual(view.Size(), 1ul);

            Assert.AreEqual(view.Extent(0), 1ul);
        }

        public void CreateVector()
        {
            View<double, TExecutionSpace> view = new("CreateVector", 10);

            Assert.AreEqual(view.Label(), "CreateVector");

            Assert.AreEqual(view.Size(), 10ul);

            Assert.AreEqual(view.Extent(0), 10ul);
        }

        public void CreateMatrix()
        {
            View<double, TExecutionSpace> view = new("CreateMatrix", 10, 10);

            Assert.AreEqual(view.Label(), "CreateMatrix");

            Assert.AreEqual(view.Size(), 10ul * 10ul);

            Assert.AreEqual(view.Extent(0), 10ul);

            Assert.AreEqual(view.Extent(1), 10ul);
        }

        public void CreateTensor()
        {
            View<double, TExecutionSpace> view = new("CreateTensor", 10, 10, 10);

            Assert.AreEqual(view.Label(), "CreateTensor");

            Assert.AreEqual(view.Size(), 10ul * 10ul * 10ul);

            Assert.AreEqual(view.Extent(0), 10ul);

            Assert.AreEqual(view.Extent(1), 10ul);

            Assert.AreEqual(view.Extent(2), 10ul);
        }

        public void CreateN4()
        {
            View<double, TExecutionSpace> view = new("CreateN4", 10, 10, 10, 10);

            Assert.AreEqual(view.Label(), "CreateN4");

            Assert.AreEqual(view.Size(), 10ul * 10ul * 10ul * 10ul);

            Assert.AreEqual(view.Extent(0), 10ul);

            Assert.AreEqual(view.Extent(1), 10ul);

            Assert.AreEqual(view.Extent(2), 10ul);

            Assert.AreEqual(view.Extent(3), 10ul);
        }

        public void CreateN5()
        {
            View<double, TExecutionSpace> view = new("CreateN5", 10, 10, 10, 10, 10);

            Assert.AreEqual(view.Label(), "CreateN5");

            Assert.AreEqual(view.Size(), 10ul * 10ul * 10ul * 10ul * 10ul);

            Assert.AreEqual(view.Extent(0), 10ul);

            Assert.AreEqual(view.Extent(1), 10ul);

            Assert.AreEqual(view.Extent(2), 10ul);

            Assert.AreEqual(view.Extent(3), 10ul);

            Assert.AreEqual(view.Extent(4), 10ul);
        }

        public void CreateN6()
        {
            View<double, TExecutionSpace> view = new("CreateN6", 10, 10, 10, 10, 10, 10);

            Assert.AreEqual(view.Label(), "CreateN6");

            Assert.AreEqual(view.Size(), 10ul * 10ul * 10ul * 10ul * 10ul * 10ul);

            Assert.AreEqual(view.Extent(0), 10ul);

            Assert.AreEqual(view.Extent(1), 10ul);

            Assert.AreEqual(view.Extent(2), 10ul);

            Assert.AreEqual(view.Extent(3), 10ul);

            Assert.AreEqual(view.Extent(4), 10ul);

            Assert.AreEqual(view.Extent(5), 10ul);
        }

        public void CreateN7()
        {
            View<double, TExecutionSpace> view = new("CreateN7", 10, 10, 10, 10, 10, 10, 10);

            Assert.AreEqual(view.Label(), "CreateN7");

            Assert.AreEqual(view.Size(), 10ul * 10ul * 10ul * 10ul * 10ul * 10ul * 10ul);

            Assert.AreEqual(view.Extent(0), 10ul);

            Assert.AreEqual(view.Extent(1), 10ul);

            Assert.AreEqual(view.Extent(2), 10ul);

            Assert.AreEqual(view.Extent(3), 10ul);

            Assert.AreEqual(view.Extent(4), 10ul);

            Assert.AreEqual(view.Extent(5), 10ul);

            Assert.AreEqual(view.Extent(6), 10ul);
        }

        public void CreateN8()
        {
            View<double, TExecutionSpace> view = new("CreateN8", 10, 10, 10, 10, 10, 10, 10, 10);

            Assert.AreEqual(view.Label(), "CreateN8");

            Assert.AreEqual(view.Size(), 10ul * 10ul * 10ul * 10ul * 10ul * 10ul * 10ul * 10ul);

            Assert.AreEqual(view.Extent(0), 10ul);

            Assert.AreEqual(view.Extent(1), 10ul);

            Assert.AreEqual(view.Extent(2), 10ul);

            Assert.AreEqual(view.Extent(3), 10ul);

            Assert.AreEqual(view.Extent(4), 10ul);

            Assert.AreEqual(view.Extent(5), 10ul);

            Assert.AreEqual(view.Extent(6), 10ul);

            Assert.AreEqual(view.Extent(7), 10ul);
        }

        #endregion
    }
}