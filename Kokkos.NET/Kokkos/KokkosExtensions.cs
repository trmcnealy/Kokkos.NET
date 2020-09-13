using System.IO;
using System.Text;

namespace Kokkos
{
    public static class KokkosExtensions
    {
        public static void ToCsv<TDataType, TExecutionSpace>(this View<TDataType, TExecutionSpace> view,
                                                             string                                file_path)
            where TDataType : struct
            where TExecutionSpace : IExecutionSpace, new()
        {
            StringBuilder sb = new StringBuilder();

            sb.Append(view.Label());

            sb.Append("\n");

            if(view.Rank == 1)
            {
                for(ulong i0 = 0; i0 < view.Extent(0); ++i0)
                {
                    sb.Append(view[i0]);
                    sb.Append("\n");
                }
            }
            else if(view.Rank == 2)
            {
                for(ulong i0 = 0; i0 < view.Extent(0); ++i0)
                {
                    sb.Append(view[i0, 0]);

                    for(ulong i1 = 1; i1 < view.Extent(1); ++i1)
                    {
                        sb.Append(",");
                        sb.Append(view[i0, i1]);
                    }

                    sb.Append("\n");
                }
            }
            else if(view.Rank == 3)
            {
                for(ulong i0 = 0; i0 < view.Extent(0); ++i0)
                {
                    for(ulong i1 = 0; i1 < view.Extent(1); ++i1)
                    {
                        sb.Append(view[i0, i1, 0]);

                        for(ulong i2 = 1; i2 < view.Extent(2); ++i2)
                        {
                            sb.Append(",");
                            sb.Append(view[i0, i1, i2]);
                        }
                    }

                    sb.Append("\n");
                }
            }

            File.WriteAllText(file_path, sb.ToString());
        }
    }
}