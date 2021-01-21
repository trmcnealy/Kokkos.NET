#pragma unmanaged

#include <cmath>

using int8   = char;
using uint8  = unsigned char;
using int16  = short;
using uint16 = unsigned short;
using int32  = long;
using uint32 = unsigned long;
using int64  = long long;
using uint64 = unsigned long long;

static const double one = 1.0, huge = 1e300;
static const double zero = 0.0;

typedef union
{
  double value;
  struct
  {
    uint32 msw;
    uint32 lsw;
  } parts;
  struct
  {
    uint64 w;
  } xparts;
} ieee_double_shape_type;

#define EXTRACT_WORDS(ix0, ix1, d)                                                                                                                                            \
    {                                                                                                                                                                         \
        ieee_double_shape_type ew_u;                                                                                                                                          \
        ew_u.value = (d);                                                                                                                                                     \
        (ix0)      = ew_u.parts.msw;                                                                                                                                          \
        (ix1)      = ew_u.parts.lsw;                                                                                                                                          \
    }

#define SET_HIGH_WORD(d, v)                                                                                                                                                   \
    {                                                                                                                                                                         \
        ieee_double_shape_type sh_u;                                                                                                                                          \
        sh_u.value     = (d);                                                                                                                                                 \
        sh_u.parts.msw = (v);                                                                                                                                                 \
        (d)            = sh_u.value;                                                                                                                                          \
    }

__declspec(dllexport) double ieee754_atanh(double x)
{
    double t;
    int32  hx;
    int32  ix;
    uint32 lx;

    EXTRACT_WORDS(hx, lx, x);

    ix = hx & 0x7fffffff;

    if ((ix | ((lx | (-lx)) >> 31)) > 0x3ff00000) /* |x|>1 */
    {
        return (x - x) / (x - x);
    }

    if (ix == 0x3ff00000)
    {
        return x / zero;
    }

    if (ix < 0x3e300000 && (huge + x) > zero)
    {
        return x;
    } /* x<2**-28 */

    SET_HIGH_WORD(x, ix);

    if (ix < 0x3fe00000)
    { /* x < 0.5 */
        t = x + x;
        t = 0.5 * log1p(t + t * x / (one - x));
    }
    else
    {
        t = 0.5 * log1p((x + x) / (one - x));
    }

    if (hx >= 0)
    {
        return t;
    }

    return -t;
}
