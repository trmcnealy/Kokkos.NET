#pragma once

#include "runtime.Kokkos/Extensions.hpp"
//#include "NumericalMethods/Statistics/UniformRandom.hpp"

#include <Kokkos_Sort.hpp>

namespace MeansClustering
{
    template<size_type Dimensions>
    struct Tag
    {
    };

    template<typename DataType, class ExecutionSpace, size_type Dimensions>
    struct DistanceFunctor
    {
        static_assert(Dimensions >= 1 && Dimensions <= 10, "Dimensions must be between 1 && 10");

        using Dataset = Kokkos::View<DataType* [Dimensions], typename ExecutionSpace::array_layout, ExecutionSpace>;

        using Matrix = Kokkos::View<DataType**, typename ExecutionSpace::array_layout, ExecutionSpace>;

        Dataset _dataset;
        Matrix  _distances;

        DistanceFunctor(const Dataset& dataset) : _dataset(dataset), _distances("distances", dataset.extent(0), dataset.extent(0)) {}

        KOKKOS_INLINE_FUNCTION void operator()(const Tag<1>&, const size_type i, const size_type j) const
        {
            _distances(i, j) = 0.0;

            if (i == j)
            {
                return;
            }

            _distances(i, j) += pow(_dataset(i, 0) - _dataset(j, 0), 2);
        }

        KOKKOS_INLINE_FUNCTION void operator()(const Tag<2>&, const size_type i, const size_type j) const
        {
            _distances(i, j) = 0.0;

            if (i == j)
            {
                return;
            }

            _distances(i, j) += pow(_dataset(i, 0) - _dataset(j, 0), 2);
            _distances(i, j) += pow(_dataset(i, 1) - _dataset(j, 1), 2);

            _distances(i, j) = sqrt(_distances(i, j));
        }

        KOKKOS_INLINE_FUNCTION void operator()(const Tag<3>&, const size_type i, const size_type j) const
        {
            _distances(i, j) = 0.0;

            if (i == j)
            {
                return;
            }

            _distances(i, j) += pow(_dataset(i, 0) - _dataset(j, 0), 2);
            _distances(i, j) += pow(_dataset(i, 1) - _dataset(j, 1), 2);
            _distances(i, j) += pow(_dataset(i, 2) - _dataset(j, 2), 2);

            _distances(i, j) = sqrt(_distances(i, j));
        }

        KOKKOS_INLINE_FUNCTION void operator()(const Tag<4>&, const size_type i, const size_type j) const
        {
            _distances(i, j) = 0.0;

            if (i == j)
            {
                return;
            }

            _distances(i, j) += pow(_dataset(i, 0) - _dataset(j, 0), 2);
            _distances(i, j) += pow(_dataset(i, 1) - _dataset(j, 1), 2);
            _distances(i, j) += pow(_dataset(i, 2) - _dataset(j, 2), 2);
            _distances(i, j) += pow(_dataset(i, 3) - _dataset(j, 3), 2);

            _distances(i, j) = sqrt(_distances(i, j));
        }

        KOKKOS_INLINE_FUNCTION void operator()(const Tag<5>&, const size_type i, const size_type j) const
        {
            _distances(i, j) = 0.0;

            if (i == j)
            {
                return;
            }

            _distances(i, j) += pow(_dataset(i, 0) - _dataset(j, 0), 2);
            _distances(i, j) += pow(_dataset(i, 1) - _dataset(j, 1), 2);
            _distances(i, j) += pow(_dataset(i, 2) - _dataset(j, 2), 2);
            _distances(i, j) += pow(_dataset(i, 3) - _dataset(j, 3), 2);
            _distances(i, j) += pow(_dataset(i, 4) - _dataset(j, 4), 2);

            _distances(i, j) = sqrt(_distances(i, j));
        }

        KOKKOS_INLINE_FUNCTION void operator()(const Tag<6>&, const size_type i, const size_type j) const
        {
            _distances(i, j) = 0.0;

            if (i == j)
            {
                return;
            }

            _distances(i, j) += pow(_dataset(i, 0) - _dataset(j, 0), 2);
            _distances(i, j) += pow(_dataset(i, 1) - _dataset(j, 1), 2);
            _distances(i, j) += pow(_dataset(i, 2) - _dataset(j, 2), 2);
            _distances(i, j) += pow(_dataset(i, 3) - _dataset(j, 3), 2);
            _distances(i, j) += pow(_dataset(i, 4) - _dataset(j, 4), 2);
            _distances(i, j) += pow(_dataset(i, 5) - _dataset(j, 5), 2);

            _distances(i, j) = sqrt(_distances(i, j));
        }

        KOKKOS_INLINE_FUNCTION void operator()(const Tag<7>&, const size_type i, const size_type j) const
        {
            _distances(i, j) = 0.0;

            if (i == j)
            {
                return;
            }

            _distances(i, j) += pow(_dataset(i, 0) - _dataset(j, 0), 2);
            _distances(i, j) += pow(_dataset(i, 1) - _dataset(j, 1), 2);
            _distances(i, j) += pow(_dataset(i, 2) - _dataset(j, 2), 2);
            _distances(i, j) += pow(_dataset(i, 3) - _dataset(j, 3), 2);
            _distances(i, j) += pow(_dataset(i, 4) - _dataset(j, 4), 2);
            _distances(i, j) += pow(_dataset(i, 5) - _dataset(j, 5), 2);
            _distances(i, j) += pow(_dataset(i, 6) - _dataset(j, 6), 2);

            _distances(i, j) = sqrt(_distances(i, j));
        }

        KOKKOS_INLINE_FUNCTION void operator()(const Tag<8>&, const size_type i, const size_type j) const
        {
            _distances(i, j) = 0.0;

            if (i == j)
            {
                return;
            }

            _distances(i, j) += pow(_dataset(i, 0) - _dataset(j, 0), 2);
            _distances(i, j) += pow(_dataset(i, 1) - _dataset(j, 1), 2);
            _distances(i, j) += pow(_dataset(i, 2) - _dataset(j, 2), 2);
            _distances(i, j) += pow(_dataset(i, 3) - _dataset(j, 3), 2);
            _distances(i, j) += pow(_dataset(i, 4) - _dataset(j, 4), 2);
            _distances(i, j) += pow(_dataset(i, 5) - _dataset(j, 5), 2);
            _distances(i, j) += pow(_dataset(i, 6) - _dataset(j, 6), 2);
            _distances(i, j) += pow(_dataset(i, 7) - _dataset(j, 7), 2);

            _distances(i, j) = sqrt(_distances(i, j));
        }

        KOKKOS_INLINE_FUNCTION void operator()(const Tag<9>&, const size_type i, const size_type j) const
        {
            _distances(i, j) = 0.0;

            if (i == j)
            {
                return;
            }

            _distances(i, j) += pow(_dataset(i, 0) - _dataset(j, 0), 2);
            _distances(i, j) += pow(_dataset(i, 1) - _dataset(j, 1), 2);
            _distances(i, j) += pow(_dataset(i, 2) - _dataset(j, 2), 2);
            _distances(i, j) += pow(_dataset(i, 3) - _dataset(j, 3), 2);
            _distances(i, j) += pow(_dataset(i, 4) - _dataset(j, 4), 2);
            _distances(i, j) += pow(_dataset(i, 5) - _dataset(j, 5), 2);
            _distances(i, j) += pow(_dataset(i, 6) - _dataset(j, 6), 2);
            _distances(i, j) += pow(_dataset(i, 7) - _dataset(j, 7), 2);
            _distances(i, j) += pow(_dataset(i, 8) - _dataset(j, 8), 2);

            _distances(i, j) = sqrt(_distances(i, j));
        }

        KOKKOS_INLINE_FUNCTION void operator()(const Tag<10>&, const size_type i, const size_type j) const
        {
            _distances(i, j) = 0.0;

            if (i == j)
            {
                return;
            }

            _distances(i, j) += pow(_dataset(i, 0) - _dataset(j, 0), 2);
            _distances(i, j) += pow(_dataset(i, 1) - _dataset(j, 1), 2);
            _distances(i, j) += pow(_dataset(i, 2) - _dataset(j, 2), 2);
            _distances(i, j) += pow(_dataset(i, 3) - _dataset(j, 3), 2);
            _distances(i, j) += pow(_dataset(i, 4) - _dataset(j, 4), 2);
            _distances(i, j) += pow(_dataset(i, 5) - _dataset(j, 5), 2);
            _distances(i, j) += pow(_dataset(i, 6) - _dataset(j, 6), 2);
            _distances(i, j) += pow(_dataset(i, 7) - _dataset(j, 7), 2);
            _distances(i, j) += pow(_dataset(i, 8) - _dataset(j, 8), 2);
            _distances(i, j) += pow(_dataset(i, 9) - _dataset(j, 9), 2);

            _distances(i, j) = sqrt(_distances(i, j));
        }
    };

    template<typename DataType, class ExecutionSpace, size_type Dimensions>
    struct KMeansClusteringResult
    {
        Kokkos::View<DataType**, typename ExecutionSpace::array_layout, ExecutionSpace> Centroids;
        Kokkos::View<int*, typename ExecutionSpace::array_layout, ExecutionSpace>       ClusterIndices;

        KMeansClusteringResult(const Kokkos::View<DataType**, typename ExecutionSpace::array_layout, ExecutionSpace>& centroids,
                               const Kokkos::View<int*, typename ExecutionSpace::array_layout, ExecutionSpace>&       cluster_indices) :
            Centroids(centroids),
            ClusterIndices(cluster_indices)
        {
        }
    };

    template<typename DataType, class ExecutionSpace, size_type Dimensions>
    class KMeansClustering
    {
        using Matrix    = Kokkos::View<DataType**, typename ExecutionSpace::array_layout, ExecutionSpace>;
        using VectorInt = Kokkos::View<int*, typename ExecutionSpace::array_layout, ExecutionSpace>;

        struct NormalizeData
        {
            DataType Min;
            DataType Max;
            DataType Mean;
            DataType StdDev;
        };



        struct DataIdx
        {
            static constexpr uint8 Min = 0;
             
            static constexpr uint8 Max = 1;
             
            static constexpr uint8 Mean = 2;
             
            static constexpr uint8 StdDev = 3;
             
            static constexpr uint8 SIZE = 4;
        };

        Kokkos::View<DataType*[DataIdx::SIZE], typename ExecutionSpace::array_layout, ExecutionSpace> NormalizeDataList;

        KOKKOS_INLINE_FUNCTION static MeansClustering::KMeansClusteringResult<DataType, ExecutionSpace, Dimensions> Execute(Matrix rawData, int numClusters)
        {

            Matrix means;

            VectorInt clustering = Cluster(rawData, numClusters, out means); // this is it

            Parallel.ForEach(
                Partitioner.Create(0, means.Length),
                range = >
                        {
                            for (int i = range.Item1; i < range.Item2; ++i) // each col
                            {
                                for (int j = 0; j < means[i].Length; ++j)
                                {
                                    means[i][j] = (means[i][j] * NormalizeDataList(j, DataIdx::StdDev) + NormalizeDataList(j, DataIdx::Mean);
                                    // means[i][j] = ((means[i][j] * (NormalizeDataList[j].Max - NormalizeDataList[j].Min)) + NormalizeDataList[j].Min);
                                    //((result[i][j] - min) / (max - min));
                                    //(result[i][j] - mean) / sd;
                                }
                            }
                        });

            if (clustering == null)
            {
                return null;
            }

            return new KMeansClusteringResult(means, clustering);
        }

        KOKKOS_INLINE_FUNCTION static VectorInt Cluster(Matrix rawData, int numClusters, out Matrix means)
        {
            // k-means clustering
            // index of return is tuple ID, cell is cluster ID
            // ex: [2 1 0 0 2 2] means tuple 0 is cluster 2, tuple 1 is cluster 1, tuple 2 is cluster 0, tuple 3 is cluster 0, etc.
            // an alternative clustering DS to save space is to use the .NET BitArray class
            Matrix data = Normalized(rawData); // so large values don't dominate

            bool changed = true; // was there a change in at least one cluster assignment?
            bool success = true; // were all means able to be computed? (no zero-count clusters)

            // init clustering[] to get things started
            // an alternative is to initialize means to randomly selected tuples
            // then the processing loop is
            // loop
            //    update clustering
            //    update means
            // end loop
            int restartCounter = 0;

            // RESTART:

            VectorInt clustering = InitClustering(data.Length, numClusters, DateTime.Now.Millisecond); // semi-random initialization
            means                = Allocate(numClusters, data[0].Length);                              // small convenience

            int maxCount = data.Length * 3; // sanity check
            int ct       = 0;
            while (changed == true && success == true && ct < maxCount)
            {
                ++ct;                                                // k-means typically converges very quickly
                success = UpdateMeans(data, clustering, means);      // compute new cluster means if possible. no effect if fail
                changed = UpdateClustering(data, clustering, means); // (re)assign tuples to clusters. no effect if fail
            }

            if ((!success || !changed) && restartCounter < 2)
            {
                return null;
                // changed = true;
                // success = true;
                // restartCounter++;
                // goto RESTART;
            }

            // consider adding means[][] as an out parameter - the final means could be computed
            // the final means are useful in some scenarios (e.g., discretization and RBF centroids)
            // and even though you can compute final means from final clustering, in some cases it
            // makes sense to return the means (at the expense of some method signature uglinesss)
            //
            // another alternative is to return, as an out parameter, some measure of cluster goodness
            // such as the average distance between cluster means, or the average distance between tuples in
            // a cluster, or a weighted combination of both
            return clustering;
        }

        KOKKOS_INLINE_FUNCTION static Matrix Normalized(Matrix rawData)
        {
            // normalize raw data by computing (x - mean) / stddev
            // primary alternative is min-max:
            // v' = (v - min) / (max - min)

            // make a copy of input data
            Matrix result = new DataType[rawData.Length][];
            Parallel.ForEach(
                Partitioner.Create(0, rawData.Length),
                range = >
                        {
                            for (int i = range.Item1; i < range.Item2; ++i) // each col
                            {
                                result[i] = new DataType[rawData[i].Length];
                                Array.Copy(rawData[i], result[i], rawData[i].Length);
                            }
                        });

            NormalizeDataList = new NormalizeData[result[0].Length];

            for (int j = 0; j < result[0].Length; ++j) // each col
            {
                NormalizeData normData = new NormalizeData();

                DataType colSum = 0.0;
                DataType min    = DataType.MaxValue;
                DataType max    = DataType.MinValue;

                for (int i = 0; i < result.Length; ++i)
                {
                    colSum += result[i][j];

                    // if(DataType.IsNaN(colSum))
                    //{
                    //    throw new Exception();
                    //}

                    if (result[i][j] > max)
                    {
                        max = result[i][j];
                    }

                    if (result[i][j] < min)
                    {
                        min = result[i][j];
                    }
                }

                DataType mean = colSum / result.Length;
                DataType sum  = 0.0;

                // if(DataType.IsNaN(mean))
                //{
                //    throw new Exception();
                //}

                for (int i = 0; i < result.Length; ++i)
                {
                    sum += (result[i][j] - mean) * (result[i][j] - mean);
                }

                DataType sd = Math.Sqrt(sum / result.Length);

                Parallel.ForEach(
                    Partitioner.Create(0, result.Length),
                    range = >
                            {
                                for (int i = range.Item1; i < range.Item2; ++i) // each col
                                {
                                    result[i][j] = (result[i][j] - mean) / sd;
                                    // result[i][j] = ((result[i][j] - min) / (max - min));
                                }
                            });

                normData.Min    = min;
                normData.Max    = max;
                normData.Mean   = mean;
                normData.StdDev = sd;

                NormalizeDataList[j] = normData;
            }
            return result;
        }

        KOKKOS_INLINE_FUNCTION static VectorInt InitClustering(int numTuples, int numClusters, int randomSeed)
        {
            // init clustering semi-randomly (at least one tuple in each cluster)
            // consider alternatives, especially k-means++ initialization,
            // or instead of randomly assigning each tuple to a cluster, pick
            // numClusters of the tuples as initial centroids/means then use
            // those means to assign each tuple to an initial cluster.
            Random    random     = new Random(randomSeed);
            VectorInt clustering = new int[numTuples];

            Parallel.ForEach(
                Partitioner.Create(0, numClusters),
                range = >
                        {
                            for (int i = range.Item1; i < range.Item2; ++i) // make sure each cluster has at least one tuple
                            {
                                clustering[i] = i;
                            }
                        });

            // Parallel.ForEach(Partitioner.Create(numClusters, clustering.Length), range =>
            //{
            //    for(int i = range.Item1; i < range.Item2; ++i)
            //    {
            //        clustering[i] = random.Next(0, numClusters); // other assignments random
            //    }
            //});

            for (int i = numClusters; i < clustering.Length; ++i)
            {
                clustering[i] = random.Next(0, numClusters); // other assignments random
            }

            return clustering;
        }

        KOKKOS_INLINE_FUNCTION static Matrix Allocate(int numClusters, int numColumns)
        {
            // convenience matrix allocator for Cluster()
            Matrix result = new DataType[numClusters][];
            Parallel.ForEach(
                Partitioner.Create(0, numClusters),
                range = >
                        {
                            for (int i = range.Item1; i < range.Item2; ++i)
                            {
                                result[i] = new DataType[numColumns];
                            }
                        });
            return result;
        }

        KOKKOS_INLINE_FUNCTION static bool UpdateMeans(Matrix data, VectorInt clustering, Matrix means)
        {
            int       numClusters   = means.Length;
            VectorInt clusterCounts = new int[numClusters];

            for (int i = 0; i < data.Length; ++i)
            {
                int cluster = clustering[i];
                ++clusterCounts[cluster];
            }

            // CancellationTokenSource cts = new CancellationTokenSource();
            // ParallelOptions options = new ParallelOptions();
            // options.CancellationToken = cts.Token;

            bool goodClustering = true;

            // try
            //{
            //    Parallel.ForEach(Partitioner.Create(0, clusterCounts.Length), options, (range, loopState) =>
            //    {
            //        for(int i = range.Item1; i < range.Item2; ++i)
            for (int i = 0; i < clusterCounts.Length; ++i)
            {
                if (clusterCounts[i] == 0)
                {
                    goodClustering = false; // bad clustering. no change to means[][]
                                            // loopState.Stop();
                                            // options.CancellationToken.ThrowIfCancellationRequested();
                                            // throw new OperationCanceledException(options.CancellationToken);
                    break;
                }
            }
            //    });
            //}
            // catch(OperationCanceledException)
            //{
            //    return false;
            //}
            // catch
            //{
            //    return false;
            //}
            // finally
            //{
            //    cts.Dispose();
            //}

            if (!goodClustering)
            {
                return false;
            }

            // update, zero-out means so it can be used as scratch matrix
            means.FillZerosParallel();
            // Parallel.ForEach(Partitioner.Create(0, means.Length), range =>
            //{
            //    for(int i = range.Item1; i < range.Item2; ++i)
            //    {
            //        for(int j = 0; j < means[i].Length; ++j)
            //        {
            //            means[i][j] = 0.0;
            //        }
            //    }
            //});

            for (int i = 0; i < data.Length; ++i)
            {
                int cluster = clustering[i];
                for (int j = 0; j < data[i].Length; ++j)
                {
                    means[cluster][j] += data[i][j]; // accumulate sum
                }
            }

            for (int k = 0; k < means.Length; ++k)
            {
                for (int j = 0; j < means[k].Length; ++j)
                {
                    means[k][j] /= clusterCounts[k]; // danger of div by 0
                }
            }

            return true;
        }

        KOKKOS_INLINE_FUNCTION static bool UpdateClustering(Matrix data, VectorInt clustering, Matrix means)
        {
            // (re)assign each tuple to a cluster (closest mean)
            // returns false if no tuple assignments change OR
            // if the reassignment would result in a clustering where
            // one or more clusters have no tuples.

            int  numClusters = means.Length;
            bool changed     = false;

            VectorInt newClustering = new int[clustering.Length]; // proposed result
            Array.Copy(clustering, newClustering, clustering.Length);

            DataType[] distances = new DataType[numClusters]; // distances from curr tuple to each mean

            for (int i = 0; i < data.Length; ++i) // walk thru each tuple
            {
                Parallel.ForEach(
                    Partitioner.Create(0, numClusters),
                    range = >
                            {
                                for (int k = range.Item1; k < range.Item2; k++) // walk thru each tuple
                                {
                                    distances[k] = Distance(data[i], means[k]); // compute distances from curr tuple to all k means
                                }
                            });

                int newClusterID = MinIndex(distances); // find closest mean ID
                if (newClusterID != newClustering[i])
                {
                    changed          = true;
                    newClustering[i] = newClusterID; // update
                }
            }

            if (changed == false)
            {
                return false; // no change so bail and don't update clustering[][]
            }

            // check proposed clustering[] cluster counts
            VectorInt clusterCounts = new int[numClusters];
            for (int i = 0; i < data.Length; ++i)
            {
                int cluster = newClustering[i];
                ++clusterCounts[cluster];
            }

            bool goodClustering = true;

            // CancellationTokenSource cts = new CancellationTokenSource();
            // ParallelOptions options = new ParallelOptions();
            // options.CancellationToken = cts.Token;

            // try
            //{
            //    Parallel.ForEach(Partitioner.Create(0, clusterCounts.Length), options, (range, loopState) =>
            //    {
            //        for(int i = range.Item1; i < range.Item2; ++i)
            for (int i = 0; i < clusterCounts.Length; ++i)
            {
                if (clusterCounts[i] == 0)
                {
                    goodClustering = false; // bad clustering. no change to means[][]
                                            // loopState.Stop();
                                            // options.CancellationToken.ThrowIfCancellationRequested();
                                            // throw new OperationCanceledException(options.CancellationToken);
                    break;
                }
            }
            //    });
            //}
            // catch(OperationCanceledException)
            //{
            //    return false;
            //}
            // catch
            //{
            //    return false;
            //}
            // finally
            //{
            //    cts.Dispose();
            //}

            if (!goodClustering)
            {
                return false;
            }

            Array.Copy(newClustering, clustering, newClustering.Length); // update
            return true;                                                 // good clustering and at least one change
        }

        KOKKOS_INLINE_FUNCTION static DataType Distance(DataType[] tuple, DataType[] mean)
        {
            DataType sumSquaredDiffs = 0.0;
            for (int j = 0; j < tuple.Length; ++j)
            {
                sumSquaredDiffs += Math.Pow((tuple[j] - mean[j]), 2);
            }
            return Math.Sqrt(sumSquaredDiffs);
        }

        KOKKOS_INLINE_FUNCTION static int MinIndex(DataType[] distances)
        {
            int indexOfMin = 0;

            DataType smallDist = DataType.MaxValue;

            for (int k = 0; k < distances.Length; ++k)
            {
                if (distances[k] < smallDist)
                {
                    smallDist  = distances[k];
                    indexOfMin = k;
                }
            }
            return indexOfMin;
        }

        KOKKOS_INLINE_FUNCTION static void ShowData(Matrix data, int decimals, bool indices, bool newLine)
        {
            for (int i = 0; i < data.Length; ++i)
            {
                if (indices)
                    Console.Write(i.ToString().PadLeft(3) + " ");
                for (int j = 0; j < data[i].Length; ++j)
                {
                    // if(data[i][j] >= 0.0)
                    //    Console.Write(" ");
                    Console.Write("{0,12} ", data[i][j].ToString("F" + decimals));
                }
                Console.WriteLine("");
            }
            if (newLine)
                Console.WriteLine("");
        }

        KOKKOS_INLINE_FUNCTION static void ShowVector(VectorInt vector, bool newLine)
        {
            for (int i = 0; i < vector.Length; ++i)
                Console.Write(vector[i] + " ");
            if (newLine)
                Console.WriteLine(std::endl);
        }

        KOKKOS_INLINE_FUNCTION static void ShowClustered(Matrix data, VectorInt clustering, int numClusters, int decimals)
        {
            for (int k = 0; k < numClusters; ++k)
            {
                Console.WriteLine("===================");
                for (int i = 0; i < data.Length; ++i)
                {
                    int clusterID = clustering[i];
                    if (clusterID != k)
                        continue;
                    Console.Write(i.ToString().PadLeft(3) + " ");
                    for (int j = 0; j < data[i].Length; ++j)
                    {
                        if (data[i][j] >= 0.0)
                            Console.Write(" ");
                        Console.Write(data[i][j].ToString("F" + decimals) + " ");
                    }
                    Console.WriteLine("");
                }
                Console.WriteLine("===================");
            }
        }
    };
}

template<typename DataType, class ExecutionSpace, size_type Dimensions>
__inline static Kokkos::View<DataType**, typename ExecutionSpace::array_layout, ExecutionSpace> kMeansClustering(
    const int                                                                                         k,
    const Kokkos::View<DataType* [Dimensions], typename ExecutionSpace::array_layout, ExecutionSpace> dataset)
{
    using mdrange_type = Kokkos::MDRangePolicy<ExecutionSpace, Kokkos::Rank<2>, Kokkos::IndexType<size_type>, NearestNeighbor::Tag<Dimensions>>;

    using point_type = typename mdrange_type::point_type;

    mdrange_type policy(point_type{{0, 0}}, point_type{{dataset.extent(0), dataset.extent(0)}});

    using DistanceFunctor = MeansClustering::DistanceFunctor<DataType, ExecutionSpace, Dimensions>;
    DistanceFunctor f(dataset);

    Kokkos::parallel_for("Distance", policy, f);

    Kokkos::View<DataType**, typename ExecutionSpace::array_layout, ExecutionSpace> distances("distances", dataset.extent(0), dataset.extent(0));
    Kokkos::deep_copy(distances, f._distances);

    for (size_type i = 0; i < dataset.extent(0); ++i)
    {
        auto _row = Kokkos::Extension::row(distances, i);

        Kokkos::sort(_row);
    }

    // Kokkos::View<int*, typename ExecutionSpace::array_layout, ExecutionSpace> classification("classification", dataset.extent(0));

    // const Kokkos::Random_XorShift1024_Pool<ExecutionSpace> pool(Kokkos::Impl::clock_tic());
    // Kokkos::fill_random(classification, pool, k);

    // for (int i = 0; i < k; ++i)
    //{
    //    if (arr[i].val == 0)
    //        freq1++;
    //    else if (arr[i].val == 1)
    //        freq2++;
    //}

    return distances;
}
