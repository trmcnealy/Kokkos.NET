using System;
using System.IO;
using System.Runtime.CompilerServices;

namespace Kokkos
{
    public static class CsvReader<TExecutionSpace>
        where TExecutionSpace : IExecutionSpace, new()
    {
        private static readonly ExecutionSpaceKind executionSpace;

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        static CsvReader()
        {
            executionSpace = ExecutionSpace<TExecutionSpace>.GetKind();
        }

        public static View<long, TExecutionSpace> GetCountLineEndings(string filename)
        {
            if(!File.Exists(filename))
            {
                throw new FileNotFoundException();
            }

            char[] chars = File.ReadAllText(filename).ToCharArray();

            long n = chars.LongLength;

            nint result = 0;

            View<char, TExecutionSpace> stringView = new View<char, TExecutionSpace>("stringView", n);

            for(long i = 0; i < n; ++i)
            {
                stringView[i] = chars[i];
            }

            switch(executionSpace)
            {
                case ExecutionSpaceKind.Serial:
                {
                    result = KokkosLibrary.CountLineEndingsSerial(stringView.Pointer);

                    break;
                }
                case ExecutionSpaceKind.OpenMP:
                {
                    result = KokkosLibrary.CountLineEndingsOpenMP(stringView.Pointer);

                    break;
                }
                case ExecutionSpaceKind.Cuda:
                {
                    result = KokkosLibrary.CountLineEndingsCuda(stringView.Pointer);

                    break;
                }
            }

            NdArray ndArray = View<long, TExecutionSpace>.RcpConvert(result, 1);

            View<long, TExecutionSpace> lineEndings = new View<long, TExecutionSpace>(new NativePointer(result, sizeof(long) * n), ndArray);

            return lineEndings;
        }
    }
}