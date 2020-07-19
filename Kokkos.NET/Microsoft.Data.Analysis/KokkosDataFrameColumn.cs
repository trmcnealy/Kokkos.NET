using System;
using System.Collections;
using System.Collections.Generic;

using Kokkos;

namespace Microsoft.Data.Analysis
{
    public class KokkosDataFrameColumn<TDataType, TExecutionSpace> : DataFrameColumn, IEnumerable<TDataType>
        where TDataType : struct
        where TExecutionSpace : IExecutionSpace, new()
    {
        private readonly View<TDataType, TExecutionSpace> _dataBuffer;

        public KokkosDataFrameColumn(string name, long length)
            : base(name, length, typeof(TDataType))
        {
            _dataBuffer = new View<TDataType, TExecutionSpace>(name, length);
        }

        public KokkosDataFrameColumn(View<TDataType, TExecutionSpace> view)
            : base(view.Label(), (long)view.Size(), typeof(TDataType))
        {
            _dataBuffer = view;
        }

        protected override object GetValue(long rowIndex)
        {
            return _dataBuffer[rowIndex];
        }

        protected override IReadOnlyList<object> GetValues(long startIndex,
                                                           int  length)
        {
            List<object> ret = new List<object>();
            while (ret.Count < length)
            {
                ret.Add(_dataBuffer[startIndex++]);
            }
            return ret;
        }

        protected override void SetValue(long   rowIndex,
                                         object value)
        {
            _dataBuffer[rowIndex] = (TDataType)value;
        }

        protected override IEnumerator GetEnumeratorCore()
        {
            return GetEnumerator();
        }

        public override long NullCount { get { return 0; } }

        public IEnumerator<TDataType> GetEnumerator()
        {
            for (long i = 0; i < Length; i++)
            {
                yield return _dataBuffer[i];
            }
        }
    }
}
