using System;
using System.Diagnostics;

namespace Kokkos
{
    public sealed class Measure : IDisposable
    {
        private        Stopwatch time;
        private        string    name;
        private static long      duration;

        private Measure(string name = null)
        {
            this.name = name;
            time      = new Stopwatch();
            time.Start();
        }

        public static Measure Execution(string name = null)
        {
            return new Measure(name);
        }

        public void Dispose()
        {
            time.Stop();
            duration = time.ElapsedMilliseconds;

            if(!string.IsNullOrEmpty(name))
            {
                Console.WriteLine($"{name}: {duration} ms");
            }
        }
    }
}
