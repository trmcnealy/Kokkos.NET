using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.IO;
using System.Threading.Tasks;

using Engineering.DataSource;
using Engineering.DataSource.Tools;

namespace Kokkos.Tests
{
    readonly struct EagleFordLatLong
    {
        public readonly ApiNumber Api;
        public readonly double    SurfaceLatitude;
        public readonly double    SurfaceLongitude;
        public readonly double    BottomLatitude;
        public readonly double    BottomLongitude;

        public EagleFordLatLong(string[] row)
        {
            if(row.Length != 5)
            {
                throw new InvalidDataException();
            }

            Api              = row[0];
            SurfaceLatitude  = double.Parse(row[1]);
            SurfaceLongitude = double.Parse(row[2]);
            BottomLatitude   = double.Parse(row[3]);
            BottomLongitude  = double.Parse(row[4]);
        }
    }

    public static class MemoryMappedTest
    {
        public static void Test()
        {
            InitArguments arguments = new InitArguments(8, -1, 0, true);

            using(ScopeGuard.Get(arguments))
            {
                EagleFordLatLong[] EagleFordLatLongs;

                using(MemoryMap mm = new MemoryMap("T:/EagleFordLatLongs.csv"))
                {
                    MappedCsvReader csvReader = new MappedCsvReader(mm);

                    (_, List<string[]> rows) = csvReader.ReadFile(1);

                    EagleFordLatLongs = new EagleFordLatLong[rows.Count];

                    Parallel.ForEach(Partitioner.Create(0, rows.Count),
                                     (row) =>
                                     {
                                         for(int i = row.Item1; i < row.Item2; i++)
                                         {
                                             EagleFordLatLongs[i] = new EagleFordLatLong(rows[i]);
                                         }
                                     });
                }

                View<double, OpenMP> latlongdegrees = new View<double, OpenMP>("latlongdegrees", EagleFordLatLongs.Length, 2, 2);

                for(ulong i = 0; i < latlongdegrees.Extent(0); ++i)
                {
                    latlongdegrees[i, 0, 0] = EagleFordLatLongs[i].SurfaceLatitude;
                    latlongdegrees[i, 0, 1] = EagleFordLatLongs[i].SurfaceLongitude;
                    latlongdegrees[i, 1, 0] = EagleFordLatLongs[i].BottomLatitude;
                    latlongdegrees[i, 1, 1] = EagleFordLatLongs[i].BottomLongitude;
                }

                View<double, OpenMP> neighbors = SpatialMethods<double, OpenMP>.NearestNeighbor(latlongdegrees);

                neighbors.ToCsv("T:/neighbors.csv");

                //for(int i = 0; i < EagleFordLatLongs.Length; ++i)
                //{
                //    Console.WriteLine($"{EagleFordLatLongs[i].Api} {neighbors[i]}");
                //}
            }
        }
    }
}