using System;
using System.Runtime.CompilerServices;

using PlatformApi;

namespace Kokkos
{
    public class View_Double_Cuda_1D
    {
        internal static readonly nint vtable;
        
        internal static readonly        nint                                                GetExtentAddress;
        internal static readonly unsafe delegate*<nint, uint, ulong> GetExtentPtr;

        static View_Double_Cuda_1D()
        {
            unsafe
            {
                vtable = NativeLibrary.GetExport(KokkosLibrary.Handle, "_ZTV19View_Double_Cuda_1D");

                GetExtentAddress = *(nint*)(vtable + 5 * UnManaged.Unsafe.SizeOf<nint>());
                
                GetExtentPtr     = (delegate*<nint, uint, ulong >)GetExtentAddress;


                //type_info + 1

                //dctor + 2
                //dctor + 3

                //GetLabel + 4
                //GetSize + 5
                //GetRank + 6
                //GetStride + 7
                //GetExtent + 8

                //operator + 9
                //operator + 10
                //operator + 11
                //operator + 12
                //operator + 13
                //operator + 14
                //operator + 15
                //operator + 16
                //operator + 17






            }
        }

        public NativePointer Pointer
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            get;
        }

        public View_Double_Cuda_1D(string arg_label)
        {
            Pointer = new NativePointer();
        }

        ~View_Double_Cuda_1D()
        {
            Pointer.Dispose();
        }

        public ref double this[int t0]
        {
            get { throw null!; }
        }

        public NativeString<Serial> GetLabel()
        {
            throw null!;
        }

        public ulong GetSize()
        {
            throw null!;
        }

        public uint GetRank()
        {
            throw null!;
        }

        public ulong GetStride(uint dim)
        {
            throw null!;
        }

        public ulong GetExtent(uint dim)
        {
            unsafe
            {
                return GetExtentPtr(Pointer.Data,  dim);
            }
        }
    }
}

// 2129	850	003D39F0	_ZTV17View_Bool_Cuda_0D	vtable for View_Bool_Cuda_0D
// 2130	851	003D3A80	_ZTV17View_Bool_Cuda_1D	vtable for View_Bool_Cuda_1D
// 2131	852	003D3B10	_ZTV17View_Bool_Cuda_2D	vtable for View_Bool_Cuda_2D
// 2132	853	003D3BA0	_ZTV17View_Bool_Cuda_3D	vtable for View_Bool_Cuda_3D
// 2133	854	003D3C30	_ZTV17View_Bool_Cuda_4D	vtable for View_Bool_Cuda_4D
// 2134	855	003D3CC0	_ZTV17View_Bool_Cuda_5D	vtable for View_Bool_Cuda_5D
// 2135	856	003D3D50	_ZTV17View_Bool_Cuda_6D	vtable for View_Bool_Cuda_6D
// 2136	857	003D3DE0	_ZTV17View_Bool_Cuda_7D	vtable for View_Bool_Cuda_7D
// 2137	858	003D3E70	_ZTV17View_Bool_Cuda_8D	vtable for View_Bool_Cuda_8D
// 2138	859	003D6780	_ZTV17View_Char_Cuda_0D	vtable for View_Char_Cuda_0D
// 2139	85A	003D6810	_ZTV17View_Char_Cuda_1D	vtable for View_Char_Cuda_1D
// 2140	85B	003D68A0	_ZTV17View_Char_Cuda_2D	vtable for View_Char_Cuda_2D
// 2141	85C	003D6930	_ZTV17View_Char_Cuda_3D	vtable for View_Char_Cuda_3D
// 2142	85D	003D69C0	_ZTV17View_Char_Cuda_4D	vtable for View_Char_Cuda_4D
// 2143	85E	003D6A50	_ZTV17View_Char_Cuda_5D	vtable for View_Char_Cuda_5D
// 2144	85F	003D6AE0	_ZTV17View_Char_Cuda_6D	vtable for View_Char_Cuda_6D
// 2145	860	003D6B70	_ZTV17View_Char_Cuda_7D	vtable for View_Char_Cuda_7D
// 2146	861	003D6C00	_ZTV17View_Char_Cuda_8D	vtable for View_Char_Cuda_8D
// 2147	862	003D3F00	_ZTV17View_Int8_Cuda_0D	vtable for View_Int8_Cuda_0D
// 2148	863	003D3F90	_ZTV17View_Int8_Cuda_1D	vtable for View_Int8_Cuda_1D
// 2149	864	003D4020	_ZTV17View_Int8_Cuda_2D	vtable for View_Int8_Cuda_2D
// 2150	865	003D40B0	_ZTV17View_Int8_Cuda_3D	vtable for View_Int8_Cuda_3D
// 2151	866	003D4140	_ZTV17View_Int8_Cuda_4D	vtable for View_Int8_Cuda_4D
// 2152	867	003D41D0	_ZTV17View_Int8_Cuda_5D	vtable for View_Int8_Cuda_5D
// 2153	868	003D4260	_ZTV17View_Int8_Cuda_6D	vtable for View_Int8_Cuda_6D
// 2154	869	003D42F0	_ZTV17View_Int8_Cuda_7D	vtable for View_Int8_Cuda_7D
// 2155	86A	003D4380	_ZTV17View_Int8_Cuda_8D	vtable for View_Int8_Cuda_8D
// 2156	86B	003D4920	_ZTV18View_Int16_Cuda_0D	vtable for View_Int16_Cuda_0D
// 2157	86C	003D49B0	_ZTV18View_Int16_Cuda_1D	vtable for View_Int16_Cuda_1D
// 2158	86D	003D4A40	_ZTV18View_Int16_Cuda_2D	vtable for View_Int16_Cuda_2D
// 2159	86E	003D4AD0	_ZTV18View_Int16_Cuda_3D	vtable for View_Int16_Cuda_3D
// 2160	86F	003D4B60	_ZTV18View_Int16_Cuda_4D	vtable for View_Int16_Cuda_4D
// 2161	870	003D4BF0	_ZTV18View_Int16_Cuda_5D	vtable for View_Int16_Cuda_5D
// 2162	871	003D4C80	_ZTV18View_Int16_Cuda_6D	vtable for View_Int16_Cuda_6D
// 2163	872	003D4D10	_ZTV18View_Int16_Cuda_7D	vtable for View_Int16_Cuda_7D
// 2164	873	003D4DA0	_ZTV18View_Int16_Cuda_8D	vtable for View_Int16_Cuda_8D
// 2165	874	003D5340	_ZTV18View_Int32_Cuda_0D	vtable for View_Int32_Cuda_0D
// 2166	875	003D53D0	_ZTV18View_Int32_Cuda_1D	vtable for View_Int32_Cuda_1D
// 2167	876	003D5460	_ZTV18View_Int32_Cuda_2D	vtable for View_Int32_Cuda_2D
// 2168	877	003D54F0	_ZTV18View_Int32_Cuda_3D	vtable for View_Int32_Cuda_3D
// 2169	878	003D5580	_ZTV18View_Int32_Cuda_4D	vtable for View_Int32_Cuda_4D
// 2170	879	003D5610	_ZTV18View_Int32_Cuda_5D	vtable for View_Int32_Cuda_5D
// 2171	87A	003D56A0	_ZTV18View_Int32_Cuda_6D	vtable for View_Int32_Cuda_6D
// 2172	87B	003D5730	_ZTV18View_Int32_Cuda_7D	vtable for View_Int32_Cuda_7D
// 2173	87C	003D57C0	_ZTV18View_Int32_Cuda_8D	vtable for View_Int32_Cuda_8D
// 2174	87D	003D5D60	_ZTV18View_Int64_Cuda_0D	vtable for View_Int64_Cuda_0D
// 2175	87E	003D5DF0	_ZTV18View_Int64_Cuda_1D	vtable for View_Int64_Cuda_1D
// 2176	87F	003D5E80	_ZTV18View_Int64_Cuda_2D	vtable for View_Int64_Cuda_2D
// 2177	880	003D5F10	_ZTV18View_Int64_Cuda_3D	vtable for View_Int64_Cuda_3D
// 2178	881	003D5FA0	_ZTV18View_Int64_Cuda_4D	vtable for View_Int64_Cuda_4D
// 2179	882	003D6030	_ZTV18View_Int64_Cuda_5D	vtable for View_Int64_Cuda_5D
// 2180	883	003D60C0	_ZTV18View_Int64_Cuda_6D	vtable for View_Int64_Cuda_6D
// 2181	884	003D6150	_ZTV18View_Int64_Cuda_7D	vtable for View_Int64_Cuda_7D
// 2182	885	003D61E0	_ZTV18View_Int64_Cuda_8D	vtable for View_Int64_Cuda_8D
// 2183	886	003D4410	_ZTV18View_UInt8_Cuda_0D	vtable for View_UInt8_Cuda_0D
// 2184	887	003D44A0	_ZTV18View_UInt8_Cuda_1D	vtable for View_UInt8_Cuda_1D
// 2185	888	003D4530	_ZTV18View_UInt8_Cuda_2D	vtable for View_UInt8_Cuda_2D
// 2186	889	003D45C0	_ZTV18View_UInt8_Cuda_3D	vtable for View_UInt8_Cuda_3D
// 2187	88A	003D4650	_ZTV18View_UInt8_Cuda_4D	vtable for View_UInt8_Cuda_4D
// 2188	88B	003D46E0	_ZTV18View_UInt8_Cuda_5D	vtable for View_UInt8_Cuda_5D
// 2189	88C	003D4770	_ZTV18View_UInt8_Cuda_6D	vtable for View_UInt8_Cuda_6D
// 2190	88D	003D4800	_ZTV18View_UInt8_Cuda_7D	vtable for View_UInt8_Cuda_7D
// 2191	88E	003D4890	_ZTV18View_UInt8_Cuda_8D	vtable for View_UInt8_Cuda_8D
// 2192	88F	003CFD30	_ZTV19View_Bool_OpenMP_0D	vtable for View_Bool_OpenMP_0D
// 2193	890	003CFDC0	_ZTV19View_Bool_OpenMP_1D	vtable for View_Bool_OpenMP_1D
// 2194	891	003CFE50	_ZTV19View_Bool_OpenMP_2D	vtable for View_Bool_OpenMP_2D
// 2195	892	003CFEE0	_ZTV19View_Bool_OpenMP_3D	vtable for View_Bool_OpenMP_3D
// 2196	893	003CFF70	_ZTV19View_Bool_OpenMP_4D	vtable for View_Bool_OpenMP_4D
// 2197	894	003D0000	_ZTV19View_Bool_OpenMP_5D	vtable for View_Bool_OpenMP_5D
// 2198	895	003D0090	_ZTV19View_Bool_OpenMP_6D	vtable for View_Bool_OpenMP_6D
// 2199	896	003D0120	_ZTV19View_Bool_OpenMP_7D	vtable for View_Bool_OpenMP_7D
// 2200	897	003D01B0	_ZTV19View_Bool_OpenMP_8D	vtable for View_Bool_OpenMP_8D
// 2201	898	003CC070	_ZTV19View_Bool_Serial_0D	vtable for View_Bool_Serial_0D
// 2202	899	003CC100	_ZTV19View_Bool_Serial_1D	vtable for View_Bool_Serial_1D
// 2203	89A	003CC190	_ZTV19View_Bool_Serial_2D	vtable for View_Bool_Serial_2D
// 2204	89B	003CC220	_ZTV19View_Bool_Serial_3D	vtable for View_Bool_Serial_3D
// 2205	89C	003CC2B0	_ZTV19View_Bool_Serial_4D	vtable for View_Bool_Serial_4D
// 2206	89D	003CC340	_ZTV19View_Bool_Serial_5D	vtable for View_Bool_Serial_5D
// 2207	89E	003CC3D0	_ZTV19View_Bool_Serial_6D	vtable for View_Bool_Serial_6D
// 2208	89F	003CC460	_ZTV19View_Bool_Serial_7D	vtable for View_Bool_Serial_7D
// 2209	8A0	003CC4F0	_ZTV19View_Bool_Serial_8D	vtable for View_Bool_Serial_8D
// 2210	8A1	003D2AC0	_ZTV19View_Char_OpenMP_0D	vtable for View_Char_OpenMP_0D
// 2211	8A2	003D2B50	_ZTV19View_Char_OpenMP_1D	vtable for View_Char_OpenMP_1D
// 2212	8A3	003D2BE0	_ZTV19View_Char_OpenMP_2D	vtable for View_Char_OpenMP_2D
// 2213	8A4	003D2C70	_ZTV19View_Char_OpenMP_3D	vtable for View_Char_OpenMP_3D
// 2214	8A5	003D2D00	_ZTV19View_Char_OpenMP_4D	vtable for View_Char_OpenMP_4D
// 2215	8A6	003D2D90	_ZTV19View_Char_OpenMP_5D	vtable for View_Char_OpenMP_5D
// 2216	8A7	003D2E20	_ZTV19View_Char_OpenMP_6D	vtable for View_Char_OpenMP_6D
// 2217	8A8	003D2EB0	_ZTV19View_Char_OpenMP_7D	vtable for View_Char_OpenMP_7D
// 2218	8A9	003D2F40	_ZTV19View_Char_OpenMP_8D	vtable for View_Char_OpenMP_8D
// 2219	8AA	003CEE00	_ZTV19View_Char_Serial_0D	vtable for View_Char_Serial_0D
// 2220	8AB	003CEE90	_ZTV19View_Char_Serial_1D	vtable for View_Char_Serial_1D
// 2221	8AC	003CEF20	_ZTV19View_Char_Serial_2D	vtable for View_Char_Serial_2D
// 2222	8AD	003CEFB0	_ZTV19View_Char_Serial_3D	vtable for View_Char_Serial_3D
// 2223	8AE	003CF040	_ZTV19View_Char_Serial_4D	vtable for View_Char_Serial_4D
// 2224	8AF	003CF0D0	_ZTV19View_Char_Serial_5D	vtable for View_Char_Serial_5D
// 2225	8B0	003CF160	_ZTV19View_Char_Serial_6D	vtable for View_Char_Serial_6D
// 2226	8B1	003CF1F0	_ZTV19View_Char_Serial_7D	vtable for View_Char_Serial_7D
// 2227	8B2	003CF280	_ZTV19View_Char_Serial_8D	vtable for View_Char_Serial_8D
// 2228	8B3	003D34E0	_ZTV19View_Double_Cuda_0D	vtable for View_Double_Cuda_0D
// 2229	8B4	003D3570	_ZTV19View_Double_Cuda_1D	vtable for View_Double_Cuda_1D
// 2230	8B5	003D3600	_ZTV19View_Double_Cuda_2D	vtable for View_Double_Cuda_2D
// 2231	8B6	003D3690	_ZTV19View_Double_Cuda_3D	vtable for View_Double_Cuda_3D
// 2232	8B7	003D3720	_ZTV19View_Double_Cuda_4D	vtable for View_Double_Cuda_4D
// 2233	8B8	003D37B0	_ZTV19View_Double_Cuda_5D	vtable for View_Double_Cuda_5D
// 2234	8B9	003D3840	_ZTV19View_Double_Cuda_6D	vtable for View_Double_Cuda_6D
// 2235	8BA	003D38D0	_ZTV19View_Double_Cuda_7D	vtable for View_Double_Cuda_7D
// 2236	8BB	003D3960	_ZTV19View_Double_Cuda_8D	vtable for View_Double_Cuda_8D
// 2237	8BC	003D0240	_ZTV19View_Int8_OpenMP_0D	vtable for View_Int8_OpenMP_0D
// 2238	8BD	003D02D0	_ZTV19View_Int8_OpenMP_1D	vtable for View_Int8_OpenMP_1D
// 2239	8BE	003D0360	_ZTV19View_Int8_OpenMP_2D	vtable for View_Int8_OpenMP_2D
// 2240	8BF	003D03F0	_ZTV19View_Int8_OpenMP_3D	vtable for View_Int8_OpenMP_3D
// 2241	8C0	003D0480	_ZTV19View_Int8_OpenMP_4D	vtable for View_Int8_OpenMP_4D
// 2242	8C1	003D0510	_ZTV19View_Int8_OpenMP_5D	vtable for View_Int8_OpenMP_5D
// 2243	8C2	003D05A0	_ZTV19View_Int8_OpenMP_6D	vtable for View_Int8_OpenMP_6D
// 2244	8C3	003D0630	_ZTV19View_Int8_OpenMP_7D	vtable for View_Int8_OpenMP_7D
// 2245	8C4	003D06C0	_ZTV19View_Int8_OpenMP_8D	vtable for View_Int8_OpenMP_8D
// 2246	8C5	003CC580	_ZTV19View_Int8_Serial_0D	vtable for View_Int8_Serial_0D
// 2247	8C6	003CC610	_ZTV19View_Int8_Serial_1D	vtable for View_Int8_Serial_1D
// 2248	8C7	003CC6A0	_ZTV19View_Int8_Serial_2D	vtable for View_Int8_Serial_2D
// 2249	8C8	003CC730	_ZTV19View_Int8_Serial_3D	vtable for View_Int8_Serial_3D
// 2250	8C9	003CC7C0	_ZTV19View_Int8_Serial_4D	vtable for View_Int8_Serial_4D
// 2251	8CA	003CC850	_ZTV19View_Int8_Serial_5D	vtable for View_Int8_Serial_5D
// 2252	8CB	003CC8E0	_ZTV19View_Int8_Serial_6D	vtable for View_Int8_Serial_6D
// 2253	8CC	003CC970	_ZTV19View_Int8_Serial_7D	vtable for View_Int8_Serial_7D
// 2254	8CD	003CCA00	_ZTV19View_Int8_Serial_8D	vtable for View_Int8_Serial_8D
// 2255	8CE	003D2FD0	_ZTV19View_Single_Cuda_0D	vtable for View_Single_Cuda_0D
// 2256	8CF	003D3060	_ZTV19View_Single_Cuda_1D	vtable for View_Single_Cuda_1D
// 2257	8D0	003D30F0	_ZTV19View_Single_Cuda_2D	vtable for View_Single_Cuda_2D
// 2258	8D1	003D3180	_ZTV19View_Single_Cuda_3D	vtable for View_Single_Cuda_3D
// 2259	8D2	003D3210	_ZTV19View_Single_Cuda_4D	vtable for View_Single_Cuda_4D
// 2260	8D3	003D32A0	_ZTV19View_Single_Cuda_5D	vtable for View_Single_Cuda_5D
// 2261	8D4	003D3330	_ZTV19View_Single_Cuda_6D	vtable for View_Single_Cuda_6D
// 2262	8D5	003D33C0	_ZTV19View_Single_Cuda_7D	vtable for View_Single_Cuda_7D
// 2263	8D6	003D3450	_ZTV19View_Single_Cuda_8D	vtable for View_Single_Cuda_8D
// 2264	8D7	003D4E30	_ZTV19View_UInt16_Cuda_0D	vtable for View_UInt16_Cuda_0D
// 2265	8D8	003D4EC0	_ZTV19View_UInt16_Cuda_1D	vtable for View_UInt16_Cuda_1D
// 2266	8D9	003D4F50	_ZTV19View_UInt16_Cuda_2D	vtable for View_UInt16_Cuda_2D
// 2267	8DA	003D4FE0	_ZTV19View_UInt16_Cuda_3D	vtable for View_UInt16_Cuda_3D
// 2268	8DB	003D5070	_ZTV19View_UInt16_Cuda_4D	vtable for View_UInt16_Cuda_4D
// 2269	8DC	003D5100	_ZTV19View_UInt16_Cuda_5D	vtable for View_UInt16_Cuda_5D
// 2270	8DD	003D5190	_ZTV19View_UInt16_Cuda_6D	vtable for View_UInt16_Cuda_6D
// 2271	8DE	003D5220	_ZTV19View_UInt16_Cuda_7D	vtable for View_UInt16_Cuda_7D
// 2272	8DF	003D52B0	_ZTV19View_UInt16_Cuda_8D	vtable for View_UInt16_Cuda_8D
// 2273	8.00E+00	003D5850	_ZTV19View_UInt32_Cuda_0D	vtable for View_UInt32_Cuda_0D
// 2274	8.00E+01	003D58E0	_ZTV19View_UInt32_Cuda_1D	vtable for View_UInt32_Cuda_1D
// 2275	8.00E+02	003D5970	_ZTV19View_UInt32_Cuda_2D	vtable for View_UInt32_Cuda_2D
// 2276	8.00E+03	003D5A00	_ZTV19View_UInt32_Cuda_3D	vtable for View_UInt32_Cuda_3D
// 2277	8.00E+04	003D5A90	_ZTV19View_UInt32_Cuda_4D	vtable for View_UInt32_Cuda_4D
// 2278	8.00E+05	003D5B20	_ZTV19View_UInt32_Cuda_5D	vtable for View_UInt32_Cuda_5D
// 2279	8.00E+06	003D5BB0	_ZTV19View_UInt32_Cuda_6D	vtable for View_UInt32_Cuda_6D
// 2280	8.00E+07	003D5C40	_ZTV19View_UInt32_Cuda_7D	vtable for View_UInt32_Cuda_7D
// 2281	8.00E+08	003D5CD0	_ZTV19View_UInt32_Cuda_8D	vtable for View_UInt32_Cuda_8D
// 2282	8.00E+09	003D6270	_ZTV19View_UInt64_Cuda_0D	vtable for View_UInt64_Cuda_0D
// 2283	8EA	003D6300	_ZTV19View_UInt64_Cuda_1D	vtable for View_UInt64_Cuda_1D
// 2284	8EB	003D6390	_ZTV19View_UInt64_Cuda_2D	vtable for View_UInt64_Cuda_2D
// 2285	8EC	003D6420	_ZTV19View_UInt64_Cuda_3D	vtable for View_UInt64_Cuda_3D
// 2286	8ED	003D64B0	_ZTV19View_UInt64_Cuda_4D	vtable for View_UInt64_Cuda_4D
// 2287	8EE	003D6540	_ZTV19View_UInt64_Cuda_5D	vtable for View_UInt64_Cuda_5D
// 2288	8EF	003D65D0	_ZTV19View_UInt64_Cuda_6D	vtable for View_UInt64_Cuda_6D
// 2289	8F0	003D6660	_ZTV19View_UInt64_Cuda_7D	vtable for View_UInt64_Cuda_7D
// 2290	8F1	003D66F0	_ZTV19View_UInt64_Cuda_8D	vtable for View_UInt64_Cuda_8D
// 2291	8F2	003D0C60	_ZTV20View_Int16_OpenMP_0D	vtable for View_Int16_OpenMP_0D
// 2292	8F3	003D0CF0	_ZTV20View_Int16_OpenMP_1D	vtable for View_Int16_OpenMP_1D
// 2293	8F4	003D0D80	_ZTV20View_Int16_OpenMP_2D	vtable for View_Int16_OpenMP_2D
// 2294	8F5	003D0E10	_ZTV20View_Int16_OpenMP_3D	vtable for View_Int16_OpenMP_3D
// 2295	8F6	003D0EA0	_ZTV20View_Int16_OpenMP_4D	vtable for View_Int16_OpenMP_4D
// 2296	8F7	003D0F30	_ZTV20View_Int16_OpenMP_5D	vtable for View_Int16_OpenMP_5D
// 2297	8F8	003D0FC0	_ZTV20View_Int16_OpenMP_6D	vtable for View_Int16_OpenMP_6D
// 2298	8F9	003D1050	_ZTV20View_Int16_OpenMP_7D	vtable for View_Int16_OpenMP_7D
// 2299	8FA	003D10E0	_ZTV20View_Int16_OpenMP_8D	vtable for View_Int16_OpenMP_8D
// 2300	8FB	003CCFA0	_ZTV20View_Int16_Serial_0D	vtable for View_Int16_Serial_0D
// 2301	8FC	003CD030	_ZTV20View_Int16_Serial_1D	vtable for View_Int16_Serial_1D
// 2302	8FD	003CD0C0	_ZTV20View_Int16_Serial_2D	vtable for View_Int16_Serial_2D
// 2303	8FE	003CD150	_ZTV20View_Int16_Serial_3D	vtable for View_Int16_Serial_3D
// 2304	8FF	003CD1E0	_ZTV20View_Int16_Serial_4D	vtable for View_Int16_Serial_4D
// 2305	900	003CD270	_ZTV20View_Int16_Serial_5D	vtable for View_Int16_Serial_5D
// 2306	901	003CD300	_ZTV20View_Int16_Serial_6D	vtable for View_Int16_Serial_6D
// 2307	902	003CD390	_ZTV20View_Int16_Serial_7D	vtable for View_Int16_Serial_7D
// 2308	903	003CD420	_ZTV20View_Int16_Serial_8D	vtable for View_Int16_Serial_8D
// 2309	904	003D1680	_ZTV20View_Int32_OpenMP_0D	vtable for View_Int32_OpenMP_0D
// 2310	905	003D1710	_ZTV20View_Int32_OpenMP_1D	vtable for View_Int32_OpenMP_1D
// 2311	906	003D17A0	_ZTV20View_Int32_OpenMP_2D	vtable for View_Int32_OpenMP_2D
// 2312	907	003D1830	_ZTV20View_Int32_OpenMP_3D	vtable for View_Int32_OpenMP_3D
// 2313	908	003D18C0	_ZTV20View_Int32_OpenMP_4D	vtable for View_Int32_OpenMP_4D
// 2314	909	003D1950	_ZTV20View_Int32_OpenMP_5D	vtable for View_Int32_OpenMP_5D
// 2315	90A	003D19E0	_ZTV20View_Int32_OpenMP_6D	vtable for View_Int32_OpenMP_6D
// 2316	90B	003D1A70	_ZTV20View_Int32_OpenMP_7D	vtable for View_Int32_OpenMP_7D
// 2317	90C	003D1B00	_ZTV20View_Int32_OpenMP_8D	vtable for View_Int32_OpenMP_8D
// 2318	90D	003CD9C0	_ZTV20View_Int32_Serial_0D	vtable for View_Int32_Serial_0D
// 2319	90E	003CDA50	_ZTV20View_Int32_Serial_1D	vtable for View_Int32_Serial_1D
// 2320	90F	003CDAE0	_ZTV20View_Int32_Serial_2D	vtable for View_Int32_Serial_2D
// 2321	910	003CDB70	_ZTV20View_Int32_Serial_3D	vtable for View_Int32_Serial_3D
// 2322	911	003CDC00	_ZTV20View_Int32_Serial_4D	vtable for View_Int32_Serial_4D
// 2323	912	003CDC90	_ZTV20View_Int32_Serial_5D	vtable for View_Int32_Serial_5D
// 2324	913	003CDD20	_ZTV20View_Int32_Serial_6D	vtable for View_Int32_Serial_6D
// 2325	914	003CDDB0	_ZTV20View_Int32_Serial_7D	vtable for View_Int32_Serial_7D
// 2326	915	003CDE40	_ZTV20View_Int32_Serial_8D	vtable for View_Int32_Serial_8D
// 2327	916	003D20A0	_ZTV20View_Int64_OpenMP_0D	vtable for View_Int64_OpenMP_0D
// 2328	917	003D2130	_ZTV20View_Int64_OpenMP_1D	vtable for View_Int64_OpenMP_1D
// 2329	918	003D21C0	_ZTV20View_Int64_OpenMP_2D	vtable for View_Int64_OpenMP_2D
// 2330	919	003D2250	_ZTV20View_Int64_OpenMP_3D	vtable for View_Int64_OpenMP_3D
// 2331	91A	003D22E0	_ZTV20View_Int64_OpenMP_4D	vtable for View_Int64_OpenMP_4D
// 2332	91B	003D2370	_ZTV20View_Int64_OpenMP_5D	vtable for View_Int64_OpenMP_5D
// 2333	91C	003D2400	_ZTV20View_Int64_OpenMP_6D	vtable for View_Int64_OpenMP_6D
// 2334	91D	003D2490	_ZTV20View_Int64_OpenMP_7D	vtable for View_Int64_OpenMP_7D
// 2335	91E	003D2520	_ZTV20View_Int64_OpenMP_8D	vtable for View_Int64_OpenMP_8D
// 2336	91F	003CE3E0	_ZTV20View_Int64_Serial_0D	vtable for View_Int64_Serial_0D
// 2337	920	003CE470	_ZTV20View_Int64_Serial_1D	vtable for View_Int64_Serial_1D
// 2338	921	003CE500	_ZTV20View_Int64_Serial_2D	vtable for View_Int64_Serial_2D
// 2339	922	003CE590	_ZTV20View_Int64_Serial_3D	vtable for View_Int64_Serial_3D
// 2340	923	003CE620	_ZTV20View_Int64_Serial_4D	vtable for View_Int64_Serial_4D
// 2341	924	003CE6B0	_ZTV20View_Int64_Serial_5D	vtable for View_Int64_Serial_5D
// 2342	925	003CE740	_ZTV20View_Int64_Serial_6D	vtable for View_Int64_Serial_6D
// 2343	926	003CE7D0	_ZTV20View_Int64_Serial_7D	vtable for View_Int64_Serial_7D
// 2344	927	003CE860	_ZTV20View_Int64_Serial_8D	vtable for View_Int64_Serial_8D
// 2345	928	003D0750	_ZTV20View_UInt8_OpenMP_0D	vtable for View_UInt8_OpenMP_0D
// 2346	929	003D07E0	_ZTV20View_UInt8_OpenMP_1D	vtable for View_UInt8_OpenMP_1D
// 2347	92A	003D0870	_ZTV20View_UInt8_OpenMP_2D	vtable for View_UInt8_OpenMP_2D
// 2348	92B	003D0900	_ZTV20View_UInt8_OpenMP_3D	vtable for View_UInt8_OpenMP_3D
// 2349	92C	003D0990	_ZTV20View_UInt8_OpenMP_4D	vtable for View_UInt8_OpenMP_4D
// 2350	92D	003D0A20	_ZTV20View_UInt8_OpenMP_5D	vtable for View_UInt8_OpenMP_5D
// 2351	92E	003D0AB0	_ZTV20View_UInt8_OpenMP_6D	vtable for View_UInt8_OpenMP_6D
// 2352	92F	003D0B40	_ZTV20View_UInt8_OpenMP_7D	vtable for View_UInt8_OpenMP_7D
// 2353	930	003D0BD0	_ZTV20View_UInt8_OpenMP_8D	vtable for View_UInt8_OpenMP_8D
// 2354	931	003CCA90	_ZTV20View_UInt8_Serial_0D	vtable for View_UInt8_Serial_0D
// 2355	932	003CCB20	_ZTV20View_UInt8_Serial_1D	vtable for View_UInt8_Serial_1D
// 2356	933	003CCBB0	_ZTV20View_UInt8_Serial_2D	vtable for View_UInt8_Serial_2D
// 2357	934	003CCC40	_ZTV20View_UInt8_Serial_3D	vtable for View_UInt8_Serial_3D
// 2358	935	003CCCD0	_ZTV20View_UInt8_Serial_4D	vtable for View_UInt8_Serial_4D
// 2359	936	003CCD60	_ZTV20View_UInt8_Serial_5D	vtable for View_UInt8_Serial_5D
// 2360	937	003CCDF0	_ZTV20View_UInt8_Serial_6D	vtable for View_UInt8_Serial_6D
// 2361	938	003CCE80	_ZTV20View_UInt8_Serial_7D	vtable for View_UInt8_Serial_7D
// 2362	939	003CCF10	_ZTV20View_UInt8_Serial_8D	vtable for View_UInt8_Serial_8D
// 2363	93A	003CF820	_ZTV21View_Double_OpenMP_0D	vtable for View_Double_OpenMP_0D
// 2364	93B	003CF8B0	_ZTV21View_Double_OpenMP_1D	vtable for View_Double_OpenMP_1D
// 2365	93C	003CF940	_ZTV21View_Double_OpenMP_2D	vtable for View_Double_OpenMP_2D
// 2366	93D	003CF9D0	_ZTV21View_Double_OpenMP_3D	vtable for View_Double_OpenMP_3D
// 2367	93E	003CFA60	_ZTV21View_Double_OpenMP_4D	vtable for View_Double_OpenMP_4D
// 2368	93F	003CFAF0	_ZTV21View_Double_OpenMP_5D	vtable for View_Double_OpenMP_5D
// 2369	940	003CFB80	_ZTV21View_Double_OpenMP_6D	vtable for View_Double_OpenMP_6D
// 2370	941	003CFC10	_ZTV21View_Double_OpenMP_7D	vtable for View_Double_OpenMP_7D
// 2371	942	003CFCA0	_ZTV21View_Double_OpenMP_8D	vtable for View_Double_OpenMP_8D
// 2372	943	003CBB60	_ZTV21View_Double_Serial_0D	vtable for View_Double_Serial_0D
// 2373	944	003CBBF0	_ZTV21View_Double_Serial_1D	vtable for View_Double_Serial_1D
// 2374	945	003CBC80	_ZTV21View_Double_Serial_2D	vtable for View_Double_Serial_2D
// 2375	946	003CBD10	_ZTV21View_Double_Serial_3D	vtable for View_Double_Serial_3D
// 2376	947	003CBDA0	_ZTV21View_Double_Serial_4D	vtable for View_Double_Serial_4D
// 2377	948	003CBE30	_ZTV21View_Double_Serial_5D	vtable for View_Double_Serial_5D
// 2378	949	003CBEC0	_ZTV21View_Double_Serial_6D	vtable for View_Double_Serial_6D
// 2379	94A	003CBF50	_ZTV21View_Double_Serial_7D	vtable for View_Double_Serial_7D
// 2380	94B	003CBFE0	_ZTV21View_Double_Serial_8D	vtable for View_Double_Serial_8D
// 2381	94C	003CF310	_ZTV21View_Single_OpenMP_0D	vtable for View_Single_OpenMP_0D
// 2382	94D	003CF3A0	_ZTV21View_Single_OpenMP_1D	vtable for View_Single_OpenMP_1D
// 2383	94E	003CF430	_ZTV21View_Single_OpenMP_2D	vtable for View_Single_OpenMP_2D
// 2384	94F	003CF4C0	_ZTV21View_Single_OpenMP_3D	vtable for View_Single_OpenMP_3D
// 2385	950	003CF550	_ZTV21View_Single_OpenMP_4D	vtable for View_Single_OpenMP_4D
// 2386	951	003CF5E0	_ZTV21View_Single_OpenMP_5D	vtable for View_Single_OpenMP_5D
// 2387	952	003CF670	_ZTV21View_Single_OpenMP_6D	vtable for View_Single_OpenMP_6D
// 2388	953	003CF700	_ZTV21View_Single_OpenMP_7D	vtable for View_Single_OpenMP_7D
// 2389	954	003CF790	_ZTV21View_Single_OpenMP_8D	vtable for View_Single_OpenMP_8D
// 2390	955	003CB650	_ZTV21View_Single_Serial_0D	vtable for View_Single_Serial_0D
// 2391	956	003CB6E0	_ZTV21View_Single_Serial_1D	vtable for View_Single_Serial_1D
// 2392	957	003CB770	_ZTV21View_Single_Serial_2D	vtable for View_Single_Serial_2D
// 2393	958	003CB800	_ZTV21View_Single_Serial_3D	vtable for View_Single_Serial_3D
// 2394	959	003CB890	_ZTV21View_Single_Serial_4D	vtable for View_Single_Serial_4D
// 2395	95A	003CB920	_ZTV21View_Single_Serial_5D	vtable for View_Single_Serial_5D
// 2396	95B	003CB9B0	_ZTV21View_Single_Serial_6D	vtable for View_Single_Serial_6D
// 2397	95C	003CBA40	_ZTV21View_Single_Serial_7D	vtable for View_Single_Serial_7D
// 2398	95D	003CBAD0	_ZTV21View_Single_Serial_8D	vtable for View_Single_Serial_8D
// 2399	95E	003D1170	_ZTV21View_UInt16_OpenMP_0D	vtable for View_UInt16_OpenMP_0D
// 2400	95F	003D1200	_ZTV21View_UInt16_OpenMP_1D	vtable for View_UInt16_OpenMP_1D
// 2401	960	003D1290	_ZTV21View_UInt16_OpenMP_2D	vtable for View_UInt16_OpenMP_2D
// 2402	961	003D1320	_ZTV21View_UInt16_OpenMP_3D	vtable for View_UInt16_OpenMP_3D
// 2403	962	003D13B0	_ZTV21View_UInt16_OpenMP_4D	vtable for View_UInt16_OpenMP_4D
// 2404	963	003D1440	_ZTV21View_UInt16_OpenMP_5D	vtable for View_UInt16_OpenMP_5D
// 2405	964	003D14D0	_ZTV21View_UInt16_OpenMP_6D	vtable for View_UInt16_OpenMP_6D
// 2406	965	003D1560	_ZTV21View_UInt16_OpenMP_7D	vtable for View_UInt16_OpenMP_7D
// 2407	966	003D15F0	_ZTV21View_UInt16_OpenMP_8D	vtable for View_UInt16_OpenMP_8D
// 2408	967	003CD4B0	_ZTV21View_UInt16_Serial_0D	vtable for View_UInt16_Serial_0D
// 2409	968	003CD540	_ZTV21View_UInt16_Serial_1D	vtable for View_UInt16_Serial_1D
// 2410	969	003CD5D0	_ZTV21View_UInt16_Serial_2D	vtable for View_UInt16_Serial_2D
// 2411	96A	003CD660	_ZTV21View_UInt16_Serial_3D	vtable for View_UInt16_Serial_3D
// 2412	96B	003CD6F0	_ZTV21View_UInt16_Serial_4D	vtable for View_UInt16_Serial_4D
// 2413	96C	003CD780	_ZTV21View_UInt16_Serial_5D	vtable for View_UInt16_Serial_5D
// 2414	96D	003CD810	_ZTV21View_UInt16_Serial_6D	vtable for View_UInt16_Serial_6D
// 2415	96E	003CD8A0	_ZTV21View_UInt16_Serial_7D	vtable for View_UInt16_Serial_7D
// 2416	96F	003CD930	_ZTV21View_UInt16_Serial_8D	vtable for View_UInt16_Serial_8D
// 2417	970	003D1B90	_ZTV21View_UInt32_OpenMP_0D	vtable for View_UInt32_OpenMP_0D
// 2418	971	003D1C20	_ZTV21View_UInt32_OpenMP_1D	vtable for View_UInt32_OpenMP_1D
// 2419	972	003D1CB0	_ZTV21View_UInt32_OpenMP_2D	vtable for View_UInt32_OpenMP_2D
// 2420	973	003D1D40	_ZTV21View_UInt32_OpenMP_3D	vtable for View_UInt32_OpenMP_3D
// 2421	974	003D1DD0	_ZTV21View_UInt32_OpenMP_4D	vtable for View_UInt32_OpenMP_4D
// 2422	975	003D1E60	_ZTV21View_UInt32_OpenMP_5D	vtable for View_UInt32_OpenMP_5D
// 2423	976	003D1EF0	_ZTV21View_UInt32_OpenMP_6D	vtable for View_UInt32_OpenMP_6D
// 2424	977	003D1F80	_ZTV21View_UInt32_OpenMP_7D	vtable for View_UInt32_OpenMP_7D
// 2425	978	003D2010	_ZTV21View_UInt32_OpenMP_8D	vtable for View_UInt32_OpenMP_8D
// 2426	979	003CDED0	_ZTV21View_UInt32_Serial_0D	vtable for View_UInt32_Serial_0D
// 2427	97A	003CDF60	_ZTV21View_UInt32_Serial_1D	vtable for View_UInt32_Serial_1D
// 2428	97B	003CDFF0	_ZTV21View_UInt32_Serial_2D	vtable for View_UInt32_Serial_2D
// 2429	97C	003CE080	_ZTV21View_UInt32_Serial_3D	vtable for View_UInt32_Serial_3D
// 2430	97D	003CE110	_ZTV21View_UInt32_Serial_4D	vtable for View_UInt32_Serial_4D
// 2431	97E	003CE1A0	_ZTV21View_UInt32_Serial_5D	vtable for View_UInt32_Serial_5D
// 2432	97F	003CE230	_ZTV21View_UInt32_Serial_6D	vtable for View_UInt32_Serial_6D
// 2433	980	003CE2C0	_ZTV21View_UInt32_Serial_7D	vtable for View_UInt32_Serial_7D
// 2434	981	003CE350	_ZTV21View_UInt32_Serial_8D	vtable for View_UInt32_Serial_8D
// 2435	982	003D25B0	_ZTV21View_UInt64_OpenMP_0D	vtable for View_UInt64_OpenMP_0D
// 2436	983	003D2640	_ZTV21View_UInt64_OpenMP_1D	vtable for View_UInt64_OpenMP_1D
// 2437	984	003D26D0	_ZTV21View_UInt64_OpenMP_2D	vtable for View_UInt64_OpenMP_2D
// 2438	985	003D2760	_ZTV21View_UInt64_OpenMP_3D	vtable for View_UInt64_OpenMP_3D
// 2439	986	003D27F0	_ZTV21View_UInt64_OpenMP_4D	vtable for View_UInt64_OpenMP_4D
// 2440	987	003D2880	_ZTV21View_UInt64_OpenMP_5D	vtable for View_UInt64_OpenMP_5D
// 2441	988	003D2910	_ZTV21View_UInt64_OpenMP_6D	vtable for View_UInt64_OpenMP_6D
// 2442	989	003D29A0	_ZTV21View_UInt64_OpenMP_7D	vtable for View_UInt64_OpenMP_7D
// 2443	98A	003D2A30	_ZTV21View_UInt64_OpenMP_8D	vtable for View_UInt64_OpenMP_8D
// 2444	98B	003CE8F0	_ZTV21View_UInt64_Serial_0D	vtable for View_UInt64_Serial_0D
// 2445	98C	003CE980	_ZTV21View_UInt64_Serial_1D	vtable for View_UInt64_Serial_1D
// 2446	98D	003CEA10	_ZTV21View_UInt64_Serial_2D	vtable for View_UInt64_Serial_2D
// 2447	98E	003CEAA0	_ZTV21View_UInt64_Serial_3D	vtable for View_UInt64_Serial_3D
// 2448	98F	003CEB30	_ZTV21View_UInt64_Serial_4D	vtable for View_UInt64_Serial_4D
// 2449	990	003CEBC0	_ZTV21View_UInt64_Serial_5D	vtable for View_UInt64_Serial_5D
// 2450	991	003CEC50	_ZTV21View_UInt64_Serial_6D	vtable for View_UInt64_Serial_6D
// 2451	992	003CECE0	_ZTV21View_UInt64_Serial_7D	vtable for View_UInt64_Serial_7D
// 2452	993	003CED70	_ZTV21View_UInt64_Serial_8D	vtable for View_UInt64_Serial_8D
// 2453	994	003DBB28	_ZTV5IViewIbN6Kokkos4CudaEE	vtable for IView<bool, Kokkos::Cuda>
// 2454	995	003D9738	_ZTV5IViewIbN6Kokkos6OpenMPEE	vtable for IView<bool, Kokkos::OpenMP>
// 2455	996	003D7338	_ZTV5IViewIbN6Kokkos6SerialEE	vtable for IView<bool, Kokkos::Serial>
// 2456	997	003DBE20	_ZTV5IViewIcN6Kokkos4CudaEE	vtable for IView<char, Kokkos::Cuda>
// 2457	998	003D9A38	_ZTV5IViewIcN6Kokkos6OpenMPEE	vtable for IView<char, Kokkos::OpenMP>
// 2458	999	003D7638	_ZTV5IViewIcN6Kokkos6SerialEE	vtable for IView<char, Kokkos::Serial>
// 2459	99A	003DB830	_ZTV5IViewIdN6Kokkos4CudaEE	vtable for IView<double, Kokkos::Cuda>
// 2460	99B	003D9438	_ZTV5IViewIdN6Kokkos6OpenMPEE	vtable for IView<double, Kokkos::OpenMP>
// 2461	99C	003D7038	_ZTV5IViewIdN6Kokkos6SerialEE	vtable for IView<double, Kokkos::Serial>
// 2462	99D	003DB538	_ZTV5IViewIfN6Kokkos4CudaEE	vtable for IView<float, Kokkos::Cuda>
// 2463	99E	003D9138	_ZTV5IViewIfN6Kokkos6OpenMPEE	vtable for IView<float, Kokkos::OpenMP>
// 2464	99F	003D6D38	_ZTV5IViewIfN6Kokkos6SerialEE	vtable for IView<float, Kokkos::Serial>
// 2465	9A0	003DC118	_ZTV5IViewIhN6Kokkos4CudaEE	vtable for IView<unsigned char, Kokkos::Cuda>
// 2466	9A1	003D9D38	_ZTV5IViewIhN6Kokkos6OpenMPEE	vtable for IView<unsigned char, Kokkos::OpenMP>
// 2467	9A2	003D7938	_ZTV5IViewIhN6Kokkos6SerialEE	vtable for IView<unsigned char, Kokkos::Serial>
// 2468	9A3	003DCA00	_ZTV5IViewIlN6Kokkos4CudaEE	vtable for IView<long, Kokkos::Cuda>
// 2469	9A4	003DA638	_ZTV5IViewIlN6Kokkos6OpenMPEE	vtable for IView<long, Kokkos::OpenMP>
// 2470	9A5	003D8238	_ZTV5IViewIlN6Kokkos6SerialEE	vtable for IView<long, Kokkos::Serial>
// 2471	9A6	003DCCF8	_ZTV5IViewImN6Kokkos4CudaEE	vtable for IView<unsigned long, Kokkos::Cuda>
// 2472	9A7	003DA938	_ZTV5IViewImN6Kokkos6OpenMPEE	vtable for IView<unsigned long, Kokkos::OpenMP>
// 2473	9A8	003D8538	_ZTV5IViewImN6Kokkos6SerialEE	vtable for IView<unsigned long, Kokkos::Serial>
// 2474	9A9	003DC410	_ZTV5IViewIsN6Kokkos4CudaEE	vtable for IView<short, Kokkos::Cuda>
// 2475	9AA	003DA038	_ZTV5IViewIsN6Kokkos6OpenMPEE	vtable for IView<short, Kokkos::OpenMP>
// 2476	9AB	003D7C38	_ZTV5IViewIsN6Kokkos6SerialEE	vtable for IView<short, Kokkos::Serial>
// 2477	9AC	003DC708	_ZTV5IViewItN6Kokkos4CudaEE	vtable for IView<unsigned short, Kokkos::Cuda>
// 2478	9AD	003DA338	_ZTV5IViewItN6Kokkos6OpenMPEE	vtable for IView<unsigned short, Kokkos::OpenMP>
// 2479	9AE	003D7F38	_ZTV5IViewItN6Kokkos6SerialEE	vtable for IView<unsigned short, Kokkos::Serial>
// 2480	9AF	003DD5E0	_ZTV5IViewIwN6Kokkos4CudaEE	vtable for IView<wchar_t, Kokkos::Cuda>
// 2481	9B0	003DB238	_ZTV5IViewIwN6Kokkos6OpenMPEE	vtable for IView<wchar_t, Kokkos::OpenMP>
// 2482	9B1	003D8E38	_ZTV5IViewIwN6Kokkos6SerialEE	vtable for IView<wchar_t, Kokkos::Serial>
// 2483	9B2	003DCFF0	_ZTV5IViewIxN6Kokkos4CudaEE	vtable for IView<long long, Kokkos::Cuda>
// 2484	9B3	003DAC38	_ZTV5IViewIxN6Kokkos6OpenMPEE	vtable for IView<long long, Kokkos::OpenMP>
// 2485	9B4	003D8838	_ZTV5IViewIxN6Kokkos6SerialEE	vtable for IView<long long, Kokkos::Serial>
// 2486	9B5	003DD2E8	_ZTV5IViewIyN6Kokkos4CudaEE	vtable for IView<unsigned long long, Kokkos::Cuda>
// 2487	9B6	003DAF38	_ZTV5IViewIyN6Kokkos6OpenMPEE	vtable for IView<unsigned long long, Kokkos::OpenMP>
// 2488	9B7	003D8B38	_ZTV5IViewIyN6Kokkos6SerialEE	vtable for IView<unsigned long long, Kokkos::Serial>
