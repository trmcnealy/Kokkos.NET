using System;
using System.Diagnostics;

namespace Kokkos.Tests
{
    public sealed class AssertException : Exception
    {
        public override string StackTrace { get; }

        public AssertException(string     message,
                               StackTrace st)
            : base(message)
        {
            StackTrace = st.ToString();
        }
    }
}