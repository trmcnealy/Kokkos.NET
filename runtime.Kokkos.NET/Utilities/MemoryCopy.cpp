
#include "..\KokkosAPI.hpp"

#include <MathExtensions.hpp>


static void unroll8_memcpy(void* dst, const void* src, uint64 size)
{
    const uint64_t* r  = (const uint64_t*)src;
    uint64_t*       w  = (uint64_t*)dst;
    uint64          nw = size / sizeof(*r);

    while (nw)
    {
        if (0 == (nw & 3))
        {
            uint64_t r0 = r[0];
            uint64_t r1 = r[1];
            uint64_t r2 = r[2];
            uint64_t r3 = r[3];
            w[0]        = r0;
            w[1]        = r1;
            w[2]        = r2;
            w[3]        = r3;
            r += 4;
            w += 4;
            nw -= 4;
        }
        else if (0 == (nw & 1))
        {
            uint64_t r0 = r[0];
            uint64_t r1 = r[1];
            w[0]        = r0;
            w[1]        = r1;
            r += 2;
            w += 2;
            nw -= 2;
        }
        else
        {
            w[0] = r[0];
            ++w;
            ++r;
            --nw;
        }
    }
}

static void unroll4_memcpy(void* dst, const void* src, uint64 size)
{
    const uint32_t* r  = (const uint32_t*)src;
    uint32_t*       w  = (uint32_t*)dst;
    uint64          nw = size / sizeof(*r);

    while (nw)
    {
        if (0 == (nw & 3))
        {
            uint32_t r0 = r[0];
            uint32_t r1 = r[1];
            uint32_t r2 = r[2];
            uint32_t r3 = r[3];
            w[0]        = r0;
            w[1]        = r1;
            w[2]        = r2;
            w[3]        = r3;
            r += 4;
            w += 4;
            nw -= 4;
        }
        else if (0 == (nw & 1))
        {
            uint32_t r0 = r[0];
            uint32_t r1 = r[1];
            w[0]        = r0;
            w[1]        = r1;
            r += 2;
            w += 2;
            nw -= 2;
        }
        else
        {
            w[0] = r[0];
            ++w;
            ++r;
            --nw;
        }
    }
}

int memcpy_uncached_load_sse41(void* dest, const void* src, uint64 n_bytes)
{
    int ret = 0;
#ifdef __SSE4_1__
    char*       d     = (char*)dest;
    uintptr_t   d_int = (uintptr_t)d;
    const char* s     = (const char*)src;
    uintptr_t   s_int = (uintptr_t)s;
    uint64      n     = n_bytes;

    // align src to 128-bits
    if (s_int & 0xf)
    {
        uint64 nh = std::std::min(0x10 - (s_int & 0x0f), n);
        memcpy(d, s, nh);
        d += nh;
        d_int += nh;
        s += nh;
        s_int += nh;
        n -= nh;
    }

    if (d_int & 0xf)
    { // dest is not aligned to 128-bits
        __m128i r0, r1, r2, r3, r4, r5, r6, r7;
        // unroll 8
        while (n >= 8 * sizeof(__m128i))
        {
            r0 = _mm_stream_load_si128((__m128i*)(s + 0 * sizeof(__m128i)));
            r1 = _mm_stream_load_si128((__m128i*)(s + 1 * sizeof(__m128i)));
            r2 = _mm_stream_load_si128((__m128i*)(s + 2 * sizeof(__m128i)));
            r3 = _mm_stream_load_si128((__m128i*)(s + 3 * sizeof(__m128i)));
            r4 = _mm_stream_load_si128((__m128i*)(s + 4 * sizeof(__m128i)));
            r5 = _mm_stream_load_si128((__m128i*)(s + 5 * sizeof(__m128i)));
            r6 = _mm_stream_load_si128((__m128i*)(s + 6 * sizeof(__m128i)));
            r7 = _mm_stream_load_si128((__m128i*)(s + 7 * sizeof(__m128i)));
            _mm_storeu_si128((__m128i*)(d + 0 * sizeof(__m128i)), r0);
            _mm_storeu_si128((__m128i*)(d + 1 * sizeof(__m128i)), r1);
            _mm_storeu_si128((__m128i*)(d + 2 * sizeof(__m128i)), r2);
            _mm_storeu_si128((__m128i*)(d + 3 * sizeof(__m128i)), r3);
            _mm_storeu_si128((__m128i*)(d + 4 * sizeof(__m128i)), r4);
            _mm_storeu_si128((__m128i*)(d + 5 * sizeof(__m128i)), r5);
            _mm_storeu_si128((__m128i*)(d + 6 * sizeof(__m128i)), r6);
            _mm_storeu_si128((__m128i*)(d + 7 * sizeof(__m128i)), r7);
            s += 8 * sizeof(__m128i);
            d += 8 * sizeof(__m128i);
            n -= 8 * sizeof(__m128i);
        }
        while (n >= sizeof(__m128i))
        {
            r0 = _mm_stream_load_si128((__m128i*)(s + 0 * sizeof(__m128i)));
            _mm_storeu_si128((__m128i*)(d + 0 * sizeof(__m128i)), r0);
            s += sizeof(__m128i);
            d += sizeof(__m128i);
            n -= sizeof(__m128i);
        }
    }
    else
    { // or it IS aligned
        __m128i r0, r1, r2, r3, r4, r5, r6, r7;
        // unroll 8
        while (n >= 8 * sizeof(__m128i))
        {
            r0 = _mm_stream_load_si128((__m128i*)(s + 0 * sizeof(__m128i)));
            r1 = _mm_stream_load_si128((__m128i*)(s + 1 * sizeof(__m128i)));
            r2 = _mm_stream_load_si128((__m128i*)(s + 2 * sizeof(__m128i)));
            r3 = _mm_stream_load_si128((__m128i*)(s + 3 * sizeof(__m128i)));
            r4 = _mm_stream_load_si128((__m128i*)(s + 4 * sizeof(__m128i)));
            r5 = _mm_stream_load_si128((__m128i*)(s + 5 * sizeof(__m128i)));
            r6 = _mm_stream_load_si128((__m128i*)(s + 6 * sizeof(__m128i)));
            r7 = _mm_stream_load_si128((__m128i*)(s + 7 * sizeof(__m128i)));
            _mm_stream_si128((__m128i*)(d + 0 * sizeof(__m128i)), r0);
            _mm_stream_si128((__m128i*)(d + 1 * sizeof(__m128i)), r1);
            _mm_stream_si128((__m128i*)(d + 2 * sizeof(__m128i)), r2);
            _mm_stream_si128((__m128i*)(d + 3 * sizeof(__m128i)), r3);
            _mm_stream_si128((__m128i*)(d + 4 * sizeof(__m128i)), r4);
            _mm_stream_si128((__m128i*)(d + 5 * sizeof(__m128i)), r5);
            _mm_stream_si128((__m128i*)(d + 6 * sizeof(__m128i)), r6);
            _mm_stream_si128((__m128i*)(d + 7 * sizeof(__m128i)), r7);
            s += 8 * sizeof(__m128i);
            d += 8 * sizeof(__m128i);
            n -= 8 * sizeof(__m128i);
        }
        while (n >= sizeof(__m128i))
        {
            r0 = _mm_stream_load_si128((__m128i*)(s + 0 * sizeof(__m128i)));
            _mm_stream_si128((__m128i*)(d + 0 * sizeof(__m128i)), r0);
            s += sizeof(__m128i);
            d += sizeof(__m128i);
            n -= sizeof(__m128i);
        }
    }

    if (n)
        memcpy(d, s, n);

    // fencing because of NT stores
    // potential optimization: issue only when NT stores are actually emitted
    _mm_sfence();

#else
#    error "this file should be compiled with -msse4.1"
#endif
    return ret;
}

int memcpy_uncached_store_sse(void* dest, const void* src, uint64 n_bytes)
{
    int ret = 0;
#ifdef __SSE__
    char*       d     = (char*)dest;
    uintptr_t   d_int = (uintptr_t)d;
    const char* s     = (const char*)src;
    uintptr_t   s_int = (uintptr_t)s;
    uint64      n     = n_bytes;

    // align dest to 128-bits
    if (d_int & 0xf)
    {
        uint64 nh = std::min(0x10 - (d_int & 0x0f), n);
        memcpy(d, s, nh);
        d += nh;
        d_int += nh;
        s += nh;
        s_int += nh;
        n -= nh;
    }

    if (s_int & 0xf)
    { // src is not aligned to 128-bits
        __m128 r0, r1, r2, r3;
        // unroll 4
        while (n >= 4 * 4 * sizeof(float))
        {
            r0 = _mm_loadu_ps((float*)(s + 0 * 4 * sizeof(float)));
            r1 = _mm_loadu_ps((float*)(s + 1 * 4 * sizeof(float)));
            r2 = _mm_loadu_ps((float*)(s + 2 * 4 * sizeof(float)));
            r3 = _mm_loadu_ps((float*)(s + 3 * 4 * sizeof(float)));
            _mm_stream_ps((float*)(d + 0 * 4 * sizeof(float)), r0);
            _mm_stream_ps((float*)(d + 1 * 4 * sizeof(float)), r1);
            _mm_stream_ps((float*)(d + 2 * 4 * sizeof(float)), r2);
            _mm_stream_ps((float*)(d + 3 * 4 * sizeof(float)), r3);
            s += 4 * 4 * sizeof(float);
            d += 4 * 4 * sizeof(float);
            n -= 4 * 4 * sizeof(float);
        }
        while (n >= 4 * sizeof(float))
        {
            r0 = _mm_loadu_ps((float*)(s));
            _mm_stream_ps((float*)(d), r0);
            s += 4 * sizeof(float);
            d += 4 * sizeof(float);
            n -= 4 * sizeof(float);
        }
    }
    else
    { // or it IS aligned
        __m128 r0, r1, r2, r3;
        // unroll 4
        while (n >= 4 * 4 * sizeof(float))
        {
            r0 = _mm_load_ps((float*)(s + 0 * 4 * sizeof(float)));
            r1 = _mm_load_ps((float*)(s + 1 * 4 * sizeof(float)));
            r2 = _mm_load_ps((float*)(s + 2 * 4 * sizeof(float)));
            r3 = _mm_load_ps((float*)(s + 3 * 4 * sizeof(float)));
            _mm_stream_ps((float*)(d + 0 * 4 * sizeof(float)), r0);
            _mm_stream_ps((float*)(d + 1 * 4 * sizeof(float)), r1);
            _mm_stream_ps((float*)(d + 2 * 4 * sizeof(float)), r2);
            _mm_stream_ps((float*)(d + 3 * 4 * sizeof(float)), r3);
            s += 4 * 4 * sizeof(float);
            d += 4 * 4 * sizeof(float);
            n -= 4 * 4 * sizeof(float);
        }
        while (n >= 4 * sizeof(float))
        {
            r0 = _mm_load_ps((float*)(s));
            _mm_stream_ps((float*)(d), r0);
            s += 4 * sizeof(float);
            d += 4 * sizeof(float);
            n -= 4 * sizeof(float);
        }
    }

    if (n)
        memcpy(d, s, n);

    // fencing is needed even for plain memcpy(), due to performance
    // being hit by delayed flushing of WC buffers
    _mm_sfence();
#else
#    error "this file should be compiled with -msse"
#endif
    return ret;
}

int memcpy_cached_store_sse(void* dest, const void* src, uint64 n_bytes)
{
    int ret = 0;
#ifdef __SSE__
    char*       d     = (char*)dest;
    uintptr_t   d_int = (uintptr_t)d;
    const char* s     = (const char*)src;
    uintptr_t   s_int = (uintptr_t)s;
    uint64      n     = n_bytes;

    // align dest to 128-bits
    if (d_int & 0xf)
    {
        uint64 nh = std::min(0x10 - (d_int & 0x0f), n);
        memcpy(d, s, nh);
        d += nh;
        d_int += nh;
        s += nh;
        s_int += nh;
        n -= nh;
    }

    if (s_int & 0xf)
    { // src is not aligned to 128-bits
        __m128 r0, r1, r2, r3;
        // unroll 4
        while (n >= 4 * 4 * sizeof(float))
        {
            r0 = _mm_loadu_ps((float*)(s + 0 * 4 * sizeof(float)));
            r1 = _mm_loadu_ps((float*)(s + 1 * 4 * sizeof(float)));
            r2 = _mm_loadu_ps((float*)(s + 2 * 4 * sizeof(float)));
            r3 = _mm_loadu_ps((float*)(s + 3 * 4 * sizeof(float)));
            _mm_store_ps((float*)(d + 0 * 4 * sizeof(float)), r0);
            _mm_store_ps((float*)(d + 1 * 4 * sizeof(float)), r1);
            _mm_store_ps((float*)(d + 2 * 4 * sizeof(float)), r2);
            _mm_store_ps((float*)(d + 3 * 4 * sizeof(float)), r3);
            s += 4 * 4 * sizeof(float);
            d += 4 * 4 * sizeof(float);
            n -= 4 * 4 * sizeof(float);
        }
        while (n >= 4 * sizeof(float))
        {
            r0 = _mm_loadu_ps((float*)(s));
            _mm_store_ps((float*)(d), r0);
            s += 4 * sizeof(float);
            d += 4 * sizeof(float);
            n -= 4 * sizeof(float);
        }
    }
    else
    { // or it IS aligned
        __m128 r0, r1, r2, r3;
        // unroll 4
        while (n >= 4 * 4 * sizeof(float))
        {
            r0 = _mm_load_ps((float*)(s + 0 * 4 * sizeof(float)));
            r1 = _mm_load_ps((float*)(s + 1 * 4 * sizeof(float)));
            r2 = _mm_load_ps((float*)(s + 2 * 4 * sizeof(float)));
            r3 = _mm_load_ps((float*)(s + 3 * 4 * sizeof(float)));
            _mm_store_ps((float*)(d + 0 * 4 * sizeof(float)), r0);
            _mm_store_ps((float*)(d + 1 * 4 * sizeof(float)), r1);
            _mm_store_ps((float*)(d + 2 * 4 * sizeof(float)), r2);
            _mm_store_ps((float*)(d + 3 * 4 * sizeof(float)), r3);
            s += 4 * 4 * sizeof(float);
            d += 4 * 4 * sizeof(float);
            n -= 4 * 4 * sizeof(float);
        }
        while (n >= 4 * sizeof(float))
        {
            r0 = _mm_load_ps((float*)(s));
            _mm_store_ps((float*)(d), r0);
            s += 4 * sizeof(float);
            d += 4 * sizeof(float);
            n -= 4 * sizeof(float);
        }
    }

    if (n)
        memcpy(d, s, n);

    // fencing because of NT stores
    // potential optimization: issue only when NT stores are actually emitted
    _mm_sfence();

#else
#    error "this file should be compiled with -msse"
#endif
    return ret;
}

int memcpy_uncached_store_avx(void* dest, const void* src, uint64 n_bytes)
{
    int ret = 0;
#ifdef __AVX__
    char*       d     = (char*)dest;
    uintptr_t   d_int = (uintptr_t)d;
    const char* s     = (const char*)src;
    uintptr_t   s_int = (uintptr_t)s;
    uint64      n     = n_bytes;

    // align dest to 256-bits
    if (d_int & 0x1f)
    {
        uint64 nh = std::min(0x20 - (d_int & 0x1f), n);
        memcpy(d, s, nh);
        d += nh;
        d_int += nh;
        s += nh;
        s_int += nh;
        n -= nh;
    }

    if (s_int & 0x1f)
    { // src is not aligned to 256-bits
        __m256d r0, r1, r2, r3;
        // unroll 4
        while (n >= 4 * sizeof(__m256d))
        {
            r0 = _mm256_loadu_pd((double*)(s + 0 * sizeof(__m256d)));
            r1 = _mm256_loadu_pd((double*)(s + 1 * sizeof(__m256d)));
            r2 = _mm256_loadu_pd((double*)(s + 2 * sizeof(__m256d)));
            r3 = _mm256_loadu_pd((double*)(s + 3 * sizeof(__m256d)));
            _mm256_stream_pd((double*)(d + 0 * sizeof(__m256d)), r0);
            _mm256_stream_pd((double*)(d + 1 * sizeof(__m256d)), r1);
            _mm256_stream_pd((double*)(d + 2 * sizeof(__m256d)), r2);
            _mm256_stream_pd((double*)(d + 3 * sizeof(__m256d)), r3);
            s += 4 * sizeof(__m256d);
            d += 4 * sizeof(__m256d);
            n -= 4 * sizeof(__m256d);
        }
        while (n >= sizeof(__m256d))
        {
            r0 = _mm256_loadu_pd((double*)(s));
            _mm256_stream_pd((double*)(d), r0);
            s += sizeof(__m256d);
            d += sizeof(__m256d);
            n -= sizeof(__m256d);
        }
    }
    else
    { // or it IS aligned
        __m256d r0, r1, r2, r3, r4, r5, r6, r7;
        // unroll 8
        while (n >= 8 * sizeof(__m256d))
        {
            r0 = _mm256_load_pd((double*)(s + 0 * sizeof(__m256d)));
            r1 = _mm256_load_pd((double*)(s + 1 * sizeof(__m256d)));
            r2 = _mm256_load_pd((double*)(s + 2 * sizeof(__m256d)));
            r3 = _mm256_load_pd((double*)(s + 3 * sizeof(__m256d)));
            r4 = _mm256_load_pd((double*)(s + 4 * sizeof(__m256d)));
            r5 = _mm256_load_pd((double*)(s + 5 * sizeof(__m256d)));
            r6 = _mm256_load_pd((double*)(s + 6 * sizeof(__m256d)));
            r7 = _mm256_load_pd((double*)(s + 7 * sizeof(__m256d)));
            _mm256_stream_pd((double*)(d + 0 * sizeof(__m256d)), r0);
            _mm256_stream_pd((double*)(d + 1 * sizeof(__m256d)), r1);
            _mm256_stream_pd((double*)(d + 2 * sizeof(__m256d)), r2);
            _mm256_stream_pd((double*)(d + 3 * sizeof(__m256d)), r3);
            _mm256_stream_pd((double*)(d + 4 * sizeof(__m256d)), r4);
            _mm256_stream_pd((double*)(d + 5 * sizeof(__m256d)), r5);
            _mm256_stream_pd((double*)(d + 6 * sizeof(__m256d)), r6);
            _mm256_stream_pd((double*)(d + 7 * sizeof(__m256d)), r7);
            s += 8 * sizeof(__m256d);
            d += 8 * sizeof(__m256d);
            n -= 8 * sizeof(__m256d);
        }
        while (n >= sizeof(__m256d))
        {
            r0 = _mm256_load_pd((double*)(s));
            _mm256_stream_pd((double*)(d), r0);
            s += sizeof(__m256d);
            d += sizeof(__m256d);
            n -= sizeof(__m256d);
        }
    }

    if (n)
        memcpy(d, s, n);

    // fencing is needed even for plain memcpy(), due to performance
    // being hit by delayed flushing of WC buffers
    _mm_sfence();

#else
#    error "this file should be compiled with -mavx"
#endif
    return ret;
}

int memcpy_cached_store_avx(void* dest, const void* src, uint64 n_bytes)
{
    int ret = 0;
#ifdef __AVX__
    char*       d     = (char*)dest;
    uintptr_t   d_int = (uintptr_t)d;
    const char* s     = (const char*)src;
    uintptr_t   s_int = (uintptr_t)s;
    uint64      n     = n_bytes;

    // align dest to 256-bits
    if (d_int & 0x1f)
    {
        uint64 nh = std::min(0x20 - (d_int & 0x1f), n);
        memcpy(d, s, nh);
        d += nh;
        d_int += nh;
        s += nh;
        s_int += nh;
        n -= nh;
    }

    if (s_int & 0x1f)
    { // src is not aligned to 256-bits
        __m256d r0, r1, r2, r3;
        // unroll 4
        while (n >= 4 * sizeof(__m256d))
        {
            r0 = _mm256_loadu_pd((double*)(s + 0 * sizeof(__m256d)));
            r1 = _mm256_loadu_pd((double*)(s + 1 * sizeof(__m256d)));
            r2 = _mm256_loadu_pd((double*)(s + 2 * sizeof(__m256d)));
            r3 = _mm256_loadu_pd((double*)(s + 3 * sizeof(__m256d)));
            _mm256_store_pd((double*)(d + 0 * sizeof(__m256d)), r0);
            _mm256_store_pd((double*)(d + 1 * sizeof(__m256d)), r1);
            _mm256_store_pd((double*)(d + 2 * sizeof(__m256d)), r2);
            _mm256_store_pd((double*)(d + 3 * sizeof(__m256d)), r3);
            s += 4 * sizeof(__m256d);
            d += 4 * sizeof(__m256d);
            n -= 4 * sizeof(__m256d);
        }
        while (n >= sizeof(__m256d))
        {
            r0 = _mm256_loadu_pd((double*)(s));
            _mm256_store_pd((double*)(d), r0);
            s += sizeof(__m256d);
            d += sizeof(__m256d);
            n -= sizeof(__m256d);
        }
    }
    else
    { // or it IS aligned
        __m256d r0, r1, r2, r3;
        // unroll 4
        while (n >= 4 * sizeof(__m256d))
        {
            r0 = _mm256_load_pd((double*)(s + 0 * sizeof(__m256d)));
            r1 = _mm256_load_pd((double*)(s + 1 * sizeof(__m256d)));
            r2 = _mm256_load_pd((double*)(s + 2 * sizeof(__m256d)));
            r3 = _mm256_load_pd((double*)(s + 3 * sizeof(__m256d)));
            _mm256_store_pd((double*)(d + 0 * sizeof(__m256d)), r0);
            _mm256_store_pd((double*)(d + 1 * sizeof(__m256d)), r1);
            _mm256_store_pd((double*)(d + 2 * sizeof(__m256d)), r2);
            _mm256_store_pd((double*)(d + 3 * sizeof(__m256d)), r3);
            s += 4 * sizeof(__m256d);
            d += 4 * sizeof(__m256d);
            n -= 4 * sizeof(__m256d);
        }
        while (n >= sizeof(__m256d))
        {
            r0 = _mm256_load_pd((double*)(s));
            _mm256_store_pd((double*)(d), r0);
            s += sizeof(__m256d);
            d += sizeof(__m256d);
            n -= sizeof(__m256d);
        }
    }
    if (n)
        memcpy(d, s, n);

    // fencing is needed because of the use of non-temporal stores
    _mm_sfence();

#else
#    error "this file should be compiled with -mavx"
#endif
    return ret;
}
