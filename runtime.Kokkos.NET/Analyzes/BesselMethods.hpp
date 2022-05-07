#pragma once

#include <runtime.Kokkos/ViewTypes.hpp>
#include <runtime.Kokkos/Extensions.hpp>

#include <Complex.hpp>

//template<typename DataType, class ExecutionSpace>
//using KokkosView = Kokkos::View<float*, typename ExecutionSpace::array_layout, ExecutionSpace>;

static int32    c__1   = 1;
static System::Complex<float>    c_b17  = {(float)2., (float)0.};
static int32    c__2   = 2;
static int32    c__4   = 4;
static int32    c__12  = 12;
static int32    c__13  = 13;
static int32    c__5   = 5;
static int32    c__11  = 11;
static int32    c__9   = 9;
static int32    c__0   = 0;
static int32    c__14  = 14;
static int32    c__15  = 15;
static int32    c__16  = 16;

static double c_b876 = .5;
static double c_b877 = 0.;


static int cacai_(System::Complex<float>* z__, float* fnu, int32* kode, int32* mr, int32* n,  System::Complex<float>  * y, int32* nz, float* rl, float* tol, float* elim, float* alim)
{
    /* Initialized data */

    static float pi = (float)3.14159265358979324;

    /* System generated locals */
    System::Complex<float> q__1, q__2, q__3;



    
    System::Complex<float>                     c1, c2;
    float                        az;
    System::Complex<float>                     cy[2];
    int32                     nn, nw;
    System::Complex<float>                     zn;
    float                        yy, arg, cpn;
    int32                     iuf;
    float                        fmr, sgn;
    int32                     inu;
    float                        spn;
    System::Complex<float>                     csgn;
    float                        dfnu;
    System::Complex<float>                     cspn;
    float                        ascle;


    /* *********************************************************************72 */

    /* ***BEGIN PROLOGUE  CACAI */
    /* ***REFER TO  CAIRY */

    /* C CACAI APPLIES THE ANALYTIC CONTINUATION FORMULA */

    /*         K(FNU,ZN*EXP(MP))=K(FNU,ZN)*EXP(-MP*FNU) - MP*I(FNU,ZN) */
    /*                 MP=PI*MR*CMPLX(0.0,1.0) */

    /*     TO CONTINUE THE K FUNCTION FROM THE RIGHT HALF TO THE LEFT */
    /*     HALF Z PLANE FOR USE WITH CAIRY WHERE FNU=1/3 OR 2/3 AND N=1. */
    /*     CACAI IS THE SAME AS CACON WITH THE PARTS FOR LARGER ORDERS AND */
    /*     RECURRENCE REMOVED. A RECURSIVE CALL TO CACON CAN RESULT IF CACON */
    /*     IS CALLED FROM CAIRY. */

    /* ***ROUTINES CALLED  CASYI,CBKNU,CMLRI,CSERI,CS1S2,R1MACH */
    /* ***END PROLOGUE  CACAI */
    /* Parameter adjustments */
    --y;

    /* Function Body */
    *nz    = 0;
    q__1 = -*z__;
    zn = q__1;

    az   = c_abs(z__);
    nn   = *n;
    dfnu = *fnu + (float)(*n - 1);
    if(az <= (float)2.)
    {
        goto L10;
    }
    if(az * az * (float).25 > dfnu + (float)1.)
    {
        goto L20;
    }
L10:
    /* ----------------------------------------------------------------------- */
    /*     POWER SERIES FOR THE I FUNCTION */
    /* ----------------------------------------------------------------------- */
    cseri_(&zn, fnu, kode, &nn, &y[1], &nw, tol, elim, alim);
    goto L40;
L20:
    if(az < *rl)
    {
        goto L30;
    }
    /* ----------------------------------------------------------------------- */
    /*     ASYMPTOTIC EXPANSION FOR LARGE Z FOR THE I FUNCTION */
    /* ----------------------------------------------------------------------- */
    casyi_(&zn, fnu, kode, &nn, &y[1], &nw, rl, tol, elim, alim);
    if(nw < 0)
    {
        goto L70;
    }
    goto L40;
L30:
    /* ----------------------------------------------------------------------- */
    /*     MILLER ALGORITHM NORMALIZED BY THE SERIES FOR THE I FUNCTION */
    /* ----------------------------------------------------------------------- */
    cmlri_(&zn, fnu, kode, &nn, &y[1], &nw, tol);
    if(nw < 0)
    {
        goto L70;
    }
L40:
    /* ----------------------------------------------------------------------- */
    /*     ANALYTIC CONTINUATION TO THE LEFT HALF PLANE FOR THE K FUNCTION */
    /* ----------------------------------------------------------------------- */
    cbknu_(&zn, fnu, kode, &c__1, cy, &nw, tol, elim, alim);
    if(nw != 0)
    {
        goto L70;
    }
    fmr    = (float)(*mr);
    sgn    = -r_sign(&pi, &fmr);
    q__1.real() = (float)0., q__1.imag() = sgn;
    csgn.real() = q__1.real(), csgn.imag() = q__1.imag();
    if(*kode == 1)
    {
        goto L50;
    }
    yy     = -r_imag(&zn);
    cpn    = cos(yy);
    spn    = sin(yy);
    q__2.real() = cpn, q__2.imag() = spn;
    q__1.real() = csgn.real() * q__2.real() - csgn.imag() * q__2.imag(), q__1.imag() = csgn.real() * q__2.imag() + csgn.imag() * q__2.real();
    csgn.real() = q__1.real(), csgn.imag() = q__1.imag();
L50:
    /* ----------------------------------------------------------------------- */
    /*     CALCULATE CSPN=EXP(FNU*PI*I) TO MINIMIZE LOSSES OF SIGNIFICANCE */
    /*     WHEN FNU IS LARGE */
    /* ----------------------------------------------------------------------- */
    inu    = (int32)(*fnu);
    arg    = (*fnu - (float)inu) * sgn;
    cpn    = cos(arg);
    spn    = sin(arg);
    q__1.real() = cpn, q__1.imag() = spn;
    cspn.real() = q__1.real(), cspn.imag() = q__1.imag();
    if(inu % 2 == 1)
    {
        q__1.real() = -cspn.real(), q__1.imag() = -cspn.imag();
        cspn.real() = q__1.real(), cspn.imag() = q__1.imag();
    }
    c1.real() = cy[0].real(), c1.imag() = cy[0].imag();
    c2.real() = y[1].real(), c2.imag() = y[1].imag();
    if(*kode == 1)
    {
        goto L60;
    }
    iuf   = 0;
    ascle = r1mach_(&c__1) * (float)1e3 / *tol;
    cs1s2_(&zn, &c1, &c2, &nw, &ascle, alim, &iuf);
    *nz += nw;
L60:
    q__2.real() = cspn.real() * c1.real() - cspn.imag() * c1.imag(), q__2.imag() = cspn.real() * c1.imag() + cspn.imag() * c1.real();
    q__3.real() = csgn.real() * c2.real() - csgn.imag() * c2.imag(), q__3.imag() = csgn.real() * c2.imag() + csgn.imag() * c2.real();
    q__1.real() = q__2.real() + q__3.real(), q__1.imag() = q__2.imag() + q__3.imag();
    y[1].real() = q__1.real(), y[1].imag() = q__1.imag();
    return 0;
L70:
    *nz = -1;
    if(nw == -2)
    {
        *nz = -2;
    }
    return 0;
} /* cacai_ */

static  int cacon_(System::Complex<float>* z__, float* fnu, int32* kode, int32* mr, int32* n, System::Complex<float>* y, int32* nz, float* rl, float* fnul, float* tol, float* elim, float* alim)
{
    /* Initialized data */

    static float    pi   = (float)3.14159265358979324;
    static System::Complex<float> cone = {(float)1., (float)0.};

    /* System generated locals */
    int32 i__1, i__2;
    float    r__1;
    System::Complex<float> q__1, q__2, q__3;

   
    

    
    int32                     i__;
    System::Complex<float>                     c1, c2, s1, s2, ck, cs, cy[2];
    int32                     nn, nw;
    System::Complex<float>                     st, zn, rz;
    float                        yy, c1i, c1m, as2;
    System::Complex<float>                     sc1, sc2;
    float                        c1r, arg, cpn;
    int32                     iuf;
    float                        fmr;
    System::Complex<float>                     csr[3], css[3];
    float                        sgn;
    int32                     inu;
    float                        bry[3], spn;
    System::Complex<float>                     cscl, cscr, csgn;
    System::Complex<float>                     cspn;
    int32                     kflag;
    float                        ascle, bscle;


    /* *********************************************************************72 */

    /* ***BEGIN PROLOGUE  CACON */
    /* ***REFER TO  CBESK,CBESH */

    /* C CACON APPLIES THE ANALYTIC CONTINUATION FORMULA */

    /*         K(FNU,ZN*EXP(MP))=K(FNU,ZN)*EXP(-MP*FNU) - MP*I(FNU,ZN) */
    /*                 MP=PI*MR*CMPLX(0.0,1.0) */

    /*     TO CONTINUE THE K FUNCTION FROM THE RIGHT HALF TO THE LEFT */
    /*     HALF Z PLANE */

    /* ***ROUTINES CALLED  CBINU,CBKNU,CS1S2,R1MACH */
    /* ***END PROLOGUE  CACON */
    /* Parameter adjustments */
    --y;

    /* Function Body */
    *nz    = 0;
    q__1.real() = -z__->real(), q__1.imag() = -z__->imag();
    zn.real() = q__1.real(), zn.imag() = q__1.imag();
    nn = *n;
    cbinu_(&zn, fnu, kode, &nn, &y[1], &nw, rl, fnul, tol, elim, alim);
    if(nw < 0)
    {
        goto L80;
    }
    /* ----------------------------------------------------------------------- */
    /*     ANALYTIC CONTINUATION TO THE LEFT HALF PLANE FOR THE K FUNCTION */
    /* ----------------------------------------------------------------------- */
    nn = std::min(2L, *n);
    cbknu_(&zn, fnu, kode, &nn, cy, &nw, tol, elim, alim);
    if(nw != 0)
    {
        goto L80;
    }
    s1.real() = cy[0].real(), s1.imag() = cy[0].imag();
    fmr    = (float)(*mr);
    sgn    = -r_sign(&pi, &fmr);
    q__1.real() = (float)0., q__1.imag() = sgn;
    csgn.real() = q__1.real(), csgn.imag() = q__1.imag();
    if(*kode == 1)
    {
        goto L10;
    }
    yy     = -r_imag(&zn);
    cpn    = cos(yy);
    spn    = sin(yy);
    q__2.real() = cpn, q__2.imag() = spn;
    q__1.real() = csgn.real() * q__2.real() - csgn.imag() * q__2.imag(), q__1.imag() = csgn.real() * q__2.imag() + csgn.imag() * q__2.real();
    csgn.real() = q__1.real(), csgn.imag() = q__1.imag();
L10:
    /* ----------------------------------------------------------------------- */
    /*     CALCULATE CSPN=EXP(FNU*PI*I) TO MINIMIZE LOSSES OF SIGNIFICANCE */
    /*     WHEN FNU IS LARGE */
    /* ----------------------------------------------------------------------- */
    inu    = (int32)(*fnu);
    arg    = (*fnu - (float)inu) * sgn;
    cpn    = cos(arg);
    spn    = sin(arg);
    q__1.real() = cpn, q__1.imag() = spn;
    cspn.real() = q__1.real(), cspn.imag() = q__1.imag();
    if(inu % 2 == 1)
    {
        q__1.real() = -cspn.real(), q__1.imag() = -cspn.imag();
        cspn.real() = q__1.real(), cspn.imag() = q__1.imag();
    }
    iuf  = 0;
    c1.real() = s1.real(), c1.imag() = s1.imag();
    c2.real() = y[1].real(), c2.imag() = y[1].imag();
    ascle = r1mach_(&c__1) * (float)1e3 / *tol;
    if(*kode == 1)
    {
        goto L20;
    }
    cs1s2_(&zn, &c1, &c2, &nw, &ascle, alim, &iuf);
    *nz += nw;
    sc1.real() = c1.real(), sc1.imag() = c1.imag();
L20:
    q__2.real() = cspn.real() * c1.real() - cspn.imag() * c1.imag(), q__2.imag() = cspn.real() * c1.imag() + cspn.imag() * c1.real();
    q__3.real() = csgn.real() * c2.real() - csgn.imag() * c2.imag(), q__3.imag() = csgn.real() * c2.imag() + csgn.imag() * c2.real();
    q__1.real() = q__2.real() + q__3.real(), q__1.imag() = q__2.imag() + q__3.imag();
    y[1].real() = q__1.real(), y[1].imag() = q__1.imag();
    if(*n == 1)
    {
        return 0;
    }
    q__1.real() = -cspn.real(), q__1.imag() = -cspn.imag();
    cspn.real() = q__1.real(), cspn.imag() = q__1.imag();
    s2.real() = cy[1].real(), s2.imag() = cy[1].imag();
    c1.real() = s2.real(), c1.imag() = s2.imag();
    c2.real() = y[2].real(), c2.imag() = y[2].imag();
    if(*kode == 1)
    {
        goto L30;
    }
    cs1s2_(&zn, &c1, &c2, &nw, &ascle, alim, &iuf);
    *nz += nw;
    sc2.real() = c1.real(), sc2.imag() = c1.imag();
L30:
    q__2.real() = cspn.real() * c1.real() - cspn.imag() * c1.imag(), q__2.imag() = cspn.real() * c1.imag() + cspn.imag() * c1.real();
    q__3.real() = csgn.real() * c2.real() - csgn.imag() * c2.imag(), q__3.imag() = csgn.real() * c2.imag() + csgn.imag() * c2.real();
    q__1.real() = q__2.real() + q__3.real(), q__1.imag() = q__2.imag() + q__3.imag();
    y[2].real() = q__1.real(), y[2].imag() = q__1.imag();
    if(*n == 2)
    {
        return 0;
    }
    q__1.real() = -cspn.real(), q__1.imag() = -cspn.imag();
    cspn.real() = q__1.real(), cspn.imag() = q__1.imag();
    c_div(&q__1, &c_b17, &zn);
    rz.real() = q__1.real(), rz.imag() = q__1.imag();
    r__1   = *fnu + (float)1.;
    q__2.real() = r__1, q__2.imag() = (float)0.;
    q__1.real() = q__2.real() * rz.real() - q__2.imag() * rz.imag(), q__1.imag() = q__2.real() * rz.imag() + q__2.imag() * rz.real();
    ck.real() = q__1.real(), ck.imag() = q__1.imag();
    /* ----------------------------------------------------------------------- */
    /*     SCALE NEAR EXPONENT EXTREMES DURING RECURRENCE ON K FUNCTIONS */
    /* ----------------------------------------------------------------------- */
    r__1   = (float)1. / *tol;
    q__1.real() = r__1, q__1.imag() = (float)0.;
    cscl.real() = q__1.real(), cscl.imag() = q__1.imag();
    q__1.real() = *tol, q__1.imag() = (float)0.;
    cscr.real() = q__1.real(), cscr.imag() = q__1.imag();
    css[0].real() = cscl.real(), css[0].imag() = cscl.imag();
    css[1].real() = cone.real(), css[1].imag() = cone.imag();
    css[2].real() = cscr.real(), css[2].imag() = cscr.imag();
    csr[0].real() = cscr.real(), csr[0].imag() = cscr.imag();
    csr[1].real() = cone.real(), csr[1].imag() = cone.imag();
    csr[2].real() = cscl.real(), csr[2].imag() = cscl.imag();
    bry[0] = ascle;
    bry[1] = (float)1. / ascle;
    bry[2] = r1mach_(&c__2);
    as2    = c_abs(&s2);
    kflag  = 2;
    if(as2 > bry[0])
    {
        goto L40;
    }
    kflag = 1;
    goto L50;
L40:
    if(as2 < bry[1])
    {
        goto L50;
    }
    kflag = 3;
L50:
    bscle  = bry[kflag - 1];
    i__1   = kflag - 1;
    q__1.real() = s1.real() * css[i__1].real() - s1.imag() * css[i__1].imag(), q__1.imag() = s1.real() * css[i__1].imag() + s1.imag() * css[i__1].real();
    s1.real() = q__1.real(), s1.imag() = q__1.imag();
    i__1   = kflag - 1;
    q__1.real() = s2.real() * css[i__1].real() - s2.imag() * css[i__1].imag(), q__1.imag() = s2.real() * css[i__1].imag() + s2.imag() * css[i__1].real();
    s2.real() = q__1.real(), s2.imag() = q__1.imag();
    i__1 = kflag - 1;
    cs.real() = csr[i__1].real(), cs.imag() = csr[i__1].imag();
    i__1 = *n;
    for(i__ = 3; i__ <= i__1; ++i__)
    {
        st.real() = s2.real(), st.imag() = s2.imag();
        q__2.real() = ck.real() * s2.real() - ck.imag() * s2.imag(), q__2.imag() = ck.real() * s2.imag() + ck.imag() * s2.real();
        q__1.real() = q__2.real() + s1.real(), q__1.imag() = q__2.imag() + s1.imag();
        s2.real() = q__1.real(), s2.imag() = q__1.imag();
        s1.real() = st.real(), s1.imag() = st.imag();
        q__1.real() = s2.real() * cs.real() - s2.imag() * cs.imag(), q__1.imag() = s2.real() * cs.imag() + s2.imag() * cs.real();
        c1.real() = q__1.real(), c1.imag() = q__1.imag();
        st.real() = c1.real(), st.imag() = c1.imag();
        i__2 = i__;
        c2.real() = y[i__2].real(), c2.imag() = y[i__2].imag();
        if(*kode == 1)
        {
            goto L60;
        }
        if(iuf < 0)
        {
            goto L60;
        }
        cs1s2_(&zn, &c1, &c2, &nw, &ascle, alim, &iuf);
        *nz += nw;
        sc1.real() = sc2.real(), sc1.imag() = sc2.imag();
        sc2.real() = c1.real(), sc2.imag() = c1.imag();
        if(iuf != 3)
        {
            goto L60;
        }
        iuf    = -4;
        i__2   = kflag - 1;
        q__1.real() = sc1.real() * css[i__2].real() - sc1.imag() * css[i__2].imag(), q__1.imag() = sc1.real() * css[i__2].imag() + sc1.imag() * css[i__2].real();
        s1.real() = q__1.real(), s1.imag() = q__1.imag();
        i__2   = kflag - 1;
        q__1.real() = sc2.real() * css[i__2].real() - sc2.imag() * css[i__2].imag(), q__1.imag() = sc2.real() * css[i__2].imag() + sc2.imag() * css[i__2].real();
        s2.real() = q__1.real(), s2.imag() = q__1.imag();
        st.real() = sc2.real(), st.imag() = sc2.imag();
    L60:
        i__2   = i__;
        q__2.real() = cspn.real() * c1.real() - cspn.imag() * c1.imag(), q__2.imag() = cspn.real() * c1.imag() + cspn.imag() * c1.real();
        q__3.real() = csgn.real() * c2.real() - csgn.imag() * c2.imag(), q__3.imag() = csgn.real() * c2.imag() + csgn.imag() * c2.real();
        q__1.real() = q__2.real() + q__3.real(), q__1.imag() = q__2.imag() + q__3.imag();
        y[i__2].real() = q__1.real(), y[i__2].imag() = q__1.imag();
        q__1.real() = ck.real() + rz.real(), q__1.imag() = ck.imag() + rz.imag();
        ck.real() = q__1.real(), ck.imag() = q__1.imag();
        q__1.real() = -cspn.real(), q__1.imag() = -cspn.imag();
        cspn.real() = q__1.real(), cspn.imag() = q__1.imag();
        if(kflag >= 3)
        {
            goto L70;
        }
        c1r = c1.real();
        c1i = r_imag(&c1);
        c1r = abs(c1r);
        c1i = abs(c1i);
        c1m = max(c1r, c1i);
        if(c1m <= bscle)
        {
            goto L70;
        }
        ++kflag;
        bscle  = bry[kflag - 1];
        q__1.real() = s1.real() * cs.real() - s1.imag() * cs.imag(), q__1.imag() = s1.real() * cs.imag() + s1.imag() * cs.real();
        s1.real() = q__1.real(), s1.imag() = q__1.imag();
        s2.real() = st.real(), s2.imag() = st.imag();
        i__2   = kflag - 1;
        q__1.real() = s1.real() * css[i__2].real() - s1.imag() * css[i__2].imag(), q__1.imag() = s1.real() * css[i__2].imag() + s1.imag() * css[i__2].real();
        s1.real() = q__1.real(), s1.imag() = q__1.imag();
        i__2   = kflag - 1;
        q__1.real() = s2.real() * css[i__2].real() - s2.imag() * css[i__2].imag(), q__1.imag() = s2.real() * css[i__2].imag() + s2.imag() * css[i__2].real();
        s2.real() = q__1.real(), s2.imag() = q__1.imag();
        i__2 = kflag - 1;
        cs.real() = csr[i__2].real(), cs.imag() = csr[i__2].imag();
    L70:;
    }
    return 0;
L80:
    *nz = -1;
    if(nw == -2)
    {
        *nz = -2;
    }
    return 0;
} /* cacon_ */

 static int cairy_(System::Complex<float>* z__, int32* id, int32* kode, System::Complex<float>* ai, int32* nz, int32* ierr)
{
    /* Initialized data */

    static float    tth  = (float).666666666666666667;
    static float    c1   = (float).35502805388781724;
    static float    c2   = (float).258819403792806799;
    static float    coef = (float).183776298473930683;
    static System::Complex<float> cone = {(float)1., (float)0.};

    /* System generated locals */
    int32    i__1, i__2;
    float       r__1, r__2;
    double d__1, d__2;
    System::Complex<float>    q__1, q__2, q__3, q__4, q__5, q__6;


    
    int32                     k;
    float                        d1, d2;
    int32                     k1, k2;
    System::Complex<float>                     s1, s2, z3;
    float                        aa, bb, ad, ak, bk, ck, dk, az;
    System::Complex<float>                     cy[1];
    int32                     nn;
    float                        rl;
    int32                     mr;
    float                        zi, zr, az3, z3i, z3r, fid, dig, r1m5;
    System::Complex<float>                     csq;
    float                        fnu;
    System::Complex<float>                     zta;
    float                        tol;
    System::Complex<float>                     trm1, trm2;
    float                        sfac, alim, elim, alaz, atrm;
    int32                     iflag;


    /* *********************************************************************72 */

    /* c CAIRY computes the System::Complex<float> Airy function AI(Z) or its derivative. */

    /* ***BEGIN PROLOGUE  CAIRY */
    /* ***DATE WRITTEN   830501   (YYMMDD) */
    /* ***REVISION DATE  890801, 930101   (YYMMDD) */
    /* ***CATEGORY NO.  B5K */
    /* ***KEYWORDS  AIRY FUNCTION,BESSEL FUNCTIONS OF ORDER ONE THIRD */
    /* ***AUTHOR  AMOS, DONALD E., SANDIA NATIONAL LABORATORIES */
    /* ***PURPOSE  TO COMPUTE AIRY FUNCTIONS AI(Z) AND DAI(Z) FOR COMPLEX Z */
    /* ***DESCRIPTION */

    /*         ON KODE=1, CAIRY COMPUTES THE COMPLEX AIRY FUNCTION AI(Z) OR */
    /*         ITS DERIVATIVE DAI(Z)/DZ ON ID=0 OR ID=1 RESPECTIVELY. ON */
    /*         KODE=2, A SCALING OPTION CEXP(ZTA)*AI(Z) OR CEXP(ZTA)* */
    /*         DAI(Z)/DZ IS PROVIDED TO REMOVE THE EXPONENTIAL DECAY IN */
    /*         -PI/3.LT.ARG(Z).LT.PI/3 AND THE EXPONENTIAL GROWTH IN */
    /*         PI/3.LT.ABS(ARG(Z)).LT.PI WHERE ZTA=(2/3)*Z*CSQRT(Z) */

    /*         WHILE THE AIRY FUNCTIONS AI(Z) AND DAI(Z)/DZ ARE ANALYTIC IN */
    /*         THE WHOLE Z PLANE, THE CORRESPONDING SCALED FUNCTIONS DEFINED */
    /*         FOR KODE=2 HAVE A CUT ALONG THE NEGATIVE REAL AXIS. */
    /*         DEFINITIONS AND NOTATION ARE FOUND IN THE NBS HANDBOOK OF */
    /*         MATHEMATICAL FUNCTIONS (CONST. 1). */

    /*         INPUT */
    /*           Z      - Z=CMPLX(X,Y) */
    /*           ID     - ORDER OF DERIVATIVE, ID=0 OR ID=1 */
    /*           KODE   - A PARAMETER TO INDICATE THE SCALING OPTION */
    /*                    KODE= 1  RETURNS */
    /*                             AI=AI(Z)                ON ID=0 OR */
    /*                             AI=DAI(Z)/DZ            ON ID=1 */
    /*                        = 2  RETURNS */
    /*                             AI=CEXP(ZTA)*AI(Z)       ON ID=0 OR */
    /*                             AI=CEXP(ZTA)*DAI(Z)/DZ   ON ID=1 WHERE */
    /*                             ZTA=(2/3)*Z*CSQRT(Z) */

    /*         OUTPUT */
    /*           AI     - COMPLEX ANSWER DEPENDING ON THE CHOICES FOR ID AND */
    /*                    KODE */
    /*           NZ     - UNDERFLOW INDICATOR */
    /*                    NZ= 0   , NORMAL RETURN */
    /*                    NZ= 1   , AI=CMPLX(0.0,0.0) DUE TO UNDERFLOW IN */
    /*                              -PI/3.LT.ARG(Z).LT.PI/3 ON KODE=1 */
    /*           IERR   - ERROR FLAG */
    /*                    IERR=0, NORMAL RETURN - COMPUTATION COMPLETED */
    /*                    IERR=1, INPUT ERROR   - NO COMPUTATION */
    /*                    IERR=2, OVERFLOW      - NO COMPUTATION, REAL(ZTA) */
    /*                            TOO LARGE WITH KODE=1. */
    /*                    IERR=3, CABS(Z) LARGE      - COMPUTATION COMPLETED */
    /*                            LOSSES OF SIGNIFCANCE BY ARGUMENT REDUCTION */
    /*                            PRODUCE LESS THAN HALF OF MACHINE ACCURACY */
    /*                    IERR=4, CABS(Z) TOO LARGE  - NO COMPUTATION */
    /*                            COMPLETE LOSS OF ACCURACY BY ARGUMENT */
    /*                            REDUCTION */
    /*                    IERR=5, ERROR              - NO COMPUTATION, */
    /*                            ALGORITHM TERMINATION CONDITION NOT MET */

    /* ***LONG DESCRIPTION */

    /*         AI AND DAI ARE COMPUTED FOR CABS(Z).GT.1.0 FROM THE K BESSEL */
    /*         FUNCTIONS BY */

    /*            AI(Z)=C*SQRT(Z)*K(1/3,ZTA) , DAI(Z)=-C*Z*K(2/3,ZTA) */
    /*                           C=1.0/(PI*SQRT(3.0)) */
    /*                           ZTA=(2/3)*Z**(3/2) */

    /*         WITH THE POWER SERIES FOR CABS(Z).LE.1.0. */

    /*         IN MOST COMPLEX VARIABLE COMPUTATION, ONE MUST EVALUATE ELE- */
    /*         MENTARY FUNCTIONS. WHEN THE MAGNITUDE OF Z IS LARGE, LOSSES */
    /*         OF SIGNIFICANCE BY ARGUMENT REDUCTION OCCUR. CONSEQUENTLY, IF */
    /*         THE MAGNITUDE OF ZETA=(2/3)*Z**1.5 EXCEEDS U1=SQRT(0.5/UR), */
    /*         THEN LOSSES EXCEEDING HALF PRECISION ARE LIKELY AND AN ERROR */
    /*         FLAG IERR=3 IS TRIGGERED WHERE UR=R1MACH(4)=UNIT ROUNDOFF. */
    /*         ALSO, IF THE MAGNITUDE OF ZETA IS LARGER THAN U2=0.5/UR, THEN */
    /*         ALL SIGNIFICANCE IS LOST AND IERR=4. IN ORDER TO USE THE INT */
    /*         FUNCTION, ZETA MUST BE FURTHER RESTRICTED NOT TO EXCEED THE */
    /*         LARGEST INTEGER, U3=I1MACH(9). THUS, THE MAGNITUDE OF ZETA */
    /*         MUST BE RESTRICTED BY MIN(U2,U3). ON 32 BIT MACHINES, U1,U2, */
    /*         AND U3 ARE APPROXIMATELY 2.0E+3, 4.2E+6, 2.1E+9 IN SINGLE */
    /*         PRECISION ARITHMETIC AND 1.3E+8, 1.8E+16, 2.1E+9 IN DOUBLE */
    /*         PRECISION ARITHMETIC RESPECTIVELY. THIS MAKES U2 AND U3 LIMIT- */
    /*         ING IN THEIR RESPECTIVE ARITHMETICS. THIS MEANS THAT THE MAG- */
    /*         NITUDE OF Z CANNOT EXCEED 3.1E+4 IN SINGLE AND 2.1E+6 IN */
    /*         DOUBLE PRECISION ARITHMETIC. THIS ALSO MEANS THAT ONE CAN */
    /*         EXPECT TO RETAIN, IN THE WORST CASES ON 32 BIT MACHINES, */
    /*         NO DIGITS IN SINGLE PRECISION AND ONLY 7 DIGITS IN DOUBLE */
    /*         PRECISION ARITHMETIC. SIMILAR CONSIDERATIONS HOLD FOR OTHER */
    /*         MACHINES. */

    /*         THE APPROXIMATE RELATIVE ERROR IN THE MAGNITUDE OF A COMPLEX */
    /*         BESSEL FUNCTION CAN BE EXPRESSED BY P*10**S WHERE P=MAX(UNIT */
    /*         ROUNDOFF,1.0E-18) IS THE NOMINAL PRECISION AND 10**S REPRE- */
    /*         SENTS THE INCREASE IN ERROR DUE TO ARGUMENT REDUCTION IN THE */
    /*         ELEMENTARY FUNCTIONS. HERE, S=MAX(1,ABS(LOG10(CABS(Z))), */
    /*         ABS(LOG10(FNU))) APPROXIMATELY (I.E. S=MAX(1,ABS(EXPONENT OF */
    /*         CABS(Z),ABS(EXPONENT OF FNU)) ). HOWEVER, THE PHASE ANGLE MAY */
    /*         HAVE ONLY ABSOLUTE ACCURACY. THIS IS MOST LIKELY TO OCCUR WHEN */
    /*         ONE COMPONENT (IN ABSOLUTE VALUE) IS LARGER THAN THE OTHER BY */
    /*         SEVERAL ORDERS OF MAGNITUDE. IF ONE COMPONENT IS 10**K LARGER */
    /*         THAN THE OTHER, THEN ONE CAN EXPECT ONLY MAX(ABS(LOG10(P))-K, */
    /*         0) SIGNIFICANT DIGITS; OR, STATED ANOTHER WAY, WHEN K EXCEEDS */
    /*         THE EXPONENT OF P, NO SIGNIFICANT DIGITS REMAIN IN THE SMALLER */
    /*         COMPONENT. HOWEVER, THE PHASE ANGLE RETAINS ABSOLUTE ACCURACY */
    /*         BECAUSE, IN COMPLEX ARITHMETIC WITH PRECISION P, THE SMALLER */
    /*         COMPONENT WILL NOT (AS A RULE) DECREASE BELOW P TIMES THE */
    /*         MAGNITUDE OF THE LARGER COMPONENT. IN THESE EXTREME CASES, */
    /*         THE PRINCIPAL PHASE ANGLE IS ON THE ORDER OF +P, -P, PI/2-P, */
    /*         OR -PI/2+P. */

    /* ***REFERENCES  HANDBOOK OF MATHEMATICAL FUNCTIONS BY M. ABRAMOWITZ */
    /*                 AND I. A. STEGUN, NBS AMS SERIES 55, U.S. DEPT. OF */
    /*                 COMMERCE, 1955. */

    /*               COMPUTATION OF BESSEL FUNCTIONS OF COMPLEX ARGUMENT */
    /*                 AND LARGE ORDER BY D. E. AMOS, SAND83-0643, MAY, 1983 */

    /*               A SUBROUTINE PACKAGE FOR BESSEL FUNCTIONS OF A COMPLEX */
    /*                 ARGUMENT AND NONNEGATIVE ORDER BY D. E. AMOS, SAND85- */
    /*                 1018, MAY, 1985 */

    /*               A PORTABLE PACKAGE FOR BESSEL FUNCTIONS OF A COMPLEX */
    /*                 ARGUMENT AND NONNEGATIVE ORDER BY D. E. AMOS, ACM */
    /*                 TRANS. MATH. SOFTWARE, VOL. 12, NO. 3, SEPTEMBER 1986, */
    /*                 PP 265-273. */

    /* ***ROUTINES CALLED  CACAI,CBKNU,I1MACH,R1MACH */
    /* ***END PROLOGUE  CAIRY */
    /* ***FIRST EXECUTABLE STATEMENT  CAIRY */
    *ierr = 0;
    *nz   = 0;
    if(*id < 0 || *id > 1)
    {
        *ierr = 1;
    }
    if(*kode < 1 || *kode > 2)
    {
        *ierr = 1;
    }
    if(*ierr != 0)
    {
        return 0;
    }
    az = c_abs(z__);
    /* Computing MAX */
    r__1 = r1mach_(&c__4);
    tol  = max(r__1, (float)1e-18);
    fid  = (float)(*id);
    if(az > (float)1.)
    {
        goto L60;
    }
    /* ----------------------------------------------------------------------- */
    /*     POWER SERIES FOR CABS(Z).LE.1. */
    /* ----------------------------------------------------------------------- */
    s1.real() = cone.real(), s1.imag() = cone.imag();
    s2.real() = cone.real(), s2.imag() = cone.imag();
    if(az < tol)
    {
        goto L160;
    }
    aa = az * az;
    if(aa < tol / az)
    {
        goto L40;
    }
    trm1.real() = cone.real(), trm1.imag() = cone.imag();
    trm2.real() = cone.real(), trm2.imag() = cone.imag();
    atrm   = (float)1.;
    q__2.real() = z__->real() * z__->real() - z__->imag() * z__->imag(), q__2.imag() = z__->real() * z__->imag() + z__->imag() * z__->real();
    q__1.real() = q__2.real() * z__->real() - q__2.imag() * z__->imag(), q__1.imag() = q__2.real() * z__->imag() + q__2.imag() * z__->real();
    z3.real() = q__1.real(), z3.imag() = q__1.imag();
    az3 = az * aa;
    ak  = fid + (float)2.;
    bk  = (float)3. - fid - fid;
    ck  = (float)4. - fid;
    dk  = fid + (float)3. + fid;
    d1  = ak * dk;
    d2  = bk * ck;
    ad  = std::min(d1, d2);
    ak  = fid * (float)9. + (float)24.;
    bk  = (float)30. - fid * (float)9.;
    z3r = z3.real();
    z3i = r_imag(&z3);
    for(k = 1; k <= 25; ++k)
    {
        r__1   = z3r / d1;
        r__2   = z3i / d1;
        q__2.real() = r__1, q__2.imag() = r__2;
        q__1.real() = trm1.real() * q__2.real() - trm1.imag() * q__2.imag(), q__1.imag() = trm1.real() * q__2.imag() + trm1.imag() * q__2.real();
        trm1.real() = q__1.real(), trm1.imag() = q__1.imag();
        q__1.real() = s1.real() + trm1.real(), q__1.imag() = s1.imag() + trm1.imag();
        s1.real() = q__1.real(), s1.imag() = q__1.imag();
        r__1   = z3r / d2;
        r__2   = z3i / d2;
        q__2.real() = r__1, q__2.imag() = r__2;
        q__1.real() = trm2.real() * q__2.real() - trm2.imag() * q__2.imag(), q__1.imag() = trm2.real() * q__2.imag() + trm2.imag() * q__2.real();
        trm2.real() = q__1.real(), trm2.imag() = q__1.imag();
        q__1.real() = s2.real() + trm2.real(), q__1.imag() = s2.imag() + trm2.imag();
        s2.real() = q__1.real(), s2.imag() = q__1.imag();
        atrm = atrm * az3 / ad;
        d1 += ak;
        d2 += bk;
        ad = std::min(d1, d2);
        if(atrm < tol * ad)
        {
            goto L40;
        }
        ak += (float)18.;
        bk += (float)18.;
        /* L30: */
    }
L40:
    if(*id == 1)
    {
        goto L50;
    }
    q__3.real() = c1, q__3.imag() = (float)0.;
    q__2.real() = s1.real() * q__3.real() - s1.imag() * q__3.imag(), q__2.imag() = s1.real() * q__3.imag() + s1.imag() * q__3.real();
    q__5.real() = z__->real() * s2.real() - z__->imag() * s2.imag(), q__5.imag() = z__->real() * s2.imag() + z__->imag() * s2.real();
    q__6.real() = c2, q__6.imag() = (float)0.;
    q__4.real() = q__5.real() * q__6.real() - q__5.imag() * q__6.imag(), q__4.imag() = q__5.real() * q__6.imag() + q__5.imag() * q__6.real();
    q__1.real() = q__2.real() - q__4.real(), q__1.imag() = q__2.imag() - q__4.imag();
    ai->real() = q__1.real(), ai->imag() = q__1.imag();
    if(*kode == 1)
    {
        return 0;
    }
    c_sqrt(&q__3, z__);
    q__2.real() = z__->real() * q__3.real() - z__->imag() * q__3.imag(), q__2.imag() = z__->real() * q__3.imag() + z__->imag() * q__3.real();
    q__4.real() = tth, q__4.imag() = (float)0.;
    q__1.real() = q__2.real() * q__4.real() - q__2.imag() * q__4.imag(), q__1.imag() = q__2.real() * q__4.imag() + q__2.imag() * q__4.real();
    zta.real() = q__1.real(), zta.imag() = q__1.imag();
    c_exp(&q__2, &zta);
    q__1.real() = ai->real() * q__2.real() - ai->imag() * q__2.imag(), q__1.imag() = ai->real() * q__2.imag() + ai->imag() * q__2.real();
    ai->real() = q__1.real(), ai->imag() = q__1.imag();
    return 0;
L50:
    q__2.real() = -s2.real(), q__2.imag() = -s2.imag();
    q__3.real() = c2, q__3.imag() = (float)0.;
    q__1.real() = q__2.real() * q__3.real() - q__2.imag() * q__3.imag(), q__1.imag() = q__2.real() * q__3.imag() + q__2.imag() * q__3.real();
    ai->real() = q__1.real(), ai->imag() = q__1.imag();
    if(az > tol)
    {
        q__4.real() = z__->real() * z__->real() - z__->imag() * z__->imag(), q__4.imag() = z__->real() * z__->imag() + z__->imag() * z__->real();
        q__3.real() = q__4.real() * s1.real() - q__4.imag() * s1.imag(), q__3.imag() = q__4.real() * s1.imag() + q__4.imag() * s1.real();
        r__1   = c1 / (fid + (float)1.);
        q__5.real() = r__1, q__5.imag() = (float)0.;
        q__2.real() = q__3.real() * q__5.real() - q__3.imag() * q__5.imag(), q__2.imag() = q__3.real() * q__5.imag() + q__3.imag() * q__5.real();
        q__1.real() = ai->real() + q__2.real(), q__1.imag() = ai->imag() + q__2.imag();
        ai->real() = q__1.real(), ai->imag() = q__1.imag();
    }
    if(*kode == 1)
    {
        return 0;
    }
    c_sqrt(&q__3, z__);
    q__2.real() = z__->real() * q__3.real() - z__->imag() * q__3.imag(), q__2.imag() = z__->real() * q__3.imag() + z__->imag() * q__3.real();
    q__4.real() = tth, q__4.imag() = (float)0.;
    q__1.real() = q__2.real() * q__4.real() - q__2.imag() * q__4.imag(), q__1.imag() = q__2.real() * q__4.imag() + q__2.imag() * q__4.real();
    zta.real() = q__1.real(), zta.imag() = q__1.imag();
    c_exp(&q__2, &zta);
    q__1.real() = ai->real() * q__2.real() - ai->imag() * q__2.imag(), q__1.imag() = ai->real() * q__2.imag() + ai->imag() * q__2.real();
    ai->real() = q__1.real(), ai->imag() = q__1.imag();
    return 0;
/* ----------------------------------------------------------------------- */
/*     CASE FOR CABS(Z).GT.1.0 */
/* ----------------------------------------------------------------------- */
L60:
    fnu = (fid + (float)1.) / (float)3.;
    /* ----------------------------------------------------------------------- */
    /*     SET PARAMETERS RELATED TO MACHINE CONSTANTS. */
    /*     TOL IS THE APPROXIMATE UNIT ROUNDOFF LIMITED TO 1.0E-18. */
    /*     ELIM IS THE APPROXIMATE EXPONENTIAL OVER- AND UNDERFLOW LIMIT. */
    /*     EXP(-ELIM).LT.EXP(-ALIM)=EXP(-ELIM)/TOL    AND */
    /*     EXP(ELIM).GT.EXP(ALIM)=EXP(ELIM)*TOL       ARE INTERVALS NEAR */
    /*     UNDERFLOW AND OVERFLOW LIMITS WHERE SCALED ARITHMETIC IS DONE. */
    /*     RL IS THE LOWER BOUNDARY OF THE ASYMPTOTIC EXPANSION FOR LARGE Z. */
    /*     DIG = NUMBER OF BASE 10 DIGITS IN TOL = 10**(-DIG). */
    /* ----------------------------------------------------------------------- */
    k1   = i1mach_(&c__12);
    k2   = i1mach_(&c__13);
    r1m5 = r1mach_(&c__5);
    /* Computing MIN */
    i__1 = abs(k1), i__2 = abs(k2);
    k    = std::min(i__1, i__2);
    elim = ((float)k * r1m5 - (float)3.) * (float)2.303;
    k1   = i1mach_(&c__11) - 1;
    aa   = r1m5 * (float)k1;
    dig  = std::min(aa, (float)18.);
    aa *= (float)2.303;
    /* Computing MAX */
    r__1 = -aa;
    alim = elim + max(r__1, (float)-41.45);
    rl   = dig * (float)1.2 + (float)3.;
    alaz = log(az);
    /* ----------------------------------------------------------------------- */
    /*     TEST FOR RANGE */
    /* ----------------------------------------------------------------------- */
    aa   = (float).5 / tol;
    bb   = (float)i1mach_(&c__9) * (float).5;
    aa   = std::min(aa, bb);
    d__1 = (double)aa;
    d__2 = (double)tth;
    aa   = pow_dd(&d__1, &d__2);
    if(az > aa)
    {
        goto L260;
    }
    aa = sqrt(aa);
    if(az > aa)
    {
        *ierr = 3;
    }
    c_sqrt(&q__1, z__);
    csq.real() = q__1.real(), csq.imag() = q__1.imag();
    q__2.real() = z__->real() * csq.real() - z__->imag() * csq.imag(), q__2.imag() = z__->real() * csq.imag() + z__->imag() * csq.real();
    q__3.real() = tth, q__3.imag() = (float)0.;
    q__1.real() = q__2.real() * q__3.real() - q__2.imag() * q__3.imag(), q__1.imag() = q__2.real() * q__3.imag() + q__2.imag() * q__3.real();
    zta.real() = q__1.real(), zta.imag() = q__1.imag();
    /* ----------------------------------------------------------------------- */
    /*     RE(ZTA).LE.0 WHEN RE(Z).LT.0, ESPECIALLY WHEN IM(Z) IS SMALL */
    /* ----------------------------------------------------------------------- */
    iflag = 0;
    sfac  = (float)1.;
    zi    = r_imag(z__);
    zr    = z__->real();
    ak    = r_imag(&zta);
    if(zr >= (float)0.)
    {
        goto L70;
    }
    bk     = zta.real();
    ck     = -abs(bk);
    q__1.real() = ck, q__1.imag() = ak;
    zta.real() = q__1.real(), zta.imag() = q__1.imag();
L70:
    if(zi != (float)0.)
    {
        goto L80;
    }
    if(zr > (float)0.)
    {
        goto L80;
    }
    q__1.real() = (float)0., q__1.imag() = ak;
    zta.real() = q__1.real(), zta.imag() = q__1.imag();
L80:
    aa = zta.real();
    if(aa >= (float)0. && zr > (float)0.)
    {
        goto L100;
    }
    if(*kode == 2)
    {
        goto L90;
    }
    /* ----------------------------------------------------------------------- */
    /*     OVERFLOW TEST */
    /* ----------------------------------------------------------------------- */
    if(aa > -alim)
    {
        goto L90;
    }
    aa    = -aa + alaz * (float).25;
    iflag = 1;
    sfac  = tol;
    if(aa > elim)
    {
        goto L240;
    }
L90:
    /* ----------------------------------------------------------------------- */
    /*     CBKNU AND CACAI RETURN EXP(ZTA)*K(FNU,ZTA) ON KODE=2 */
    /* ----------------------------------------------------------------------- */
    mr = 1;
    if(zi < (float)0.)
    {
        mr = -1;
    }
    cacai_(&zta, &fnu, kode, &mr, &c__1, cy, &nn, &rl, &tol, &elim, &alim);
    if(nn < 0)
    {
        goto L250;
    }
    *nz += nn;
    goto L120;
L100:
    if(*kode == 2)
    {
        goto L110;
    }
    /* ----------------------------------------------------------------------- */
    /*     UNDERFLOW TEST */
    /* ----------------------------------------------------------------------- */
    if(aa < alim)
    {
        goto L110;
    }
    aa    = -aa - alaz * (float).25;
    iflag = 2;
    sfac  = (float)1. / tol;
    if(aa < -elim)
    {
        goto L180;
    }
L110:
    cbknu_(&zta, &fnu, kode, &c__1, cy, nz, &tol, &elim, &alim);
L120:
    q__2.real() = coef, q__2.imag() = (float)0.;
    q__1.real() = cy[0].real() * q__2.real() - cy[0].imag() * q__2.imag(), q__1.imag() = cy[0].real() * q__2.imag() + cy[0].imag() * q__2.real();
    s1.real() = q__1.real(), s1.imag() = q__1.imag();
    if(iflag != 0)
    {
        goto L140;
    }
    if(*id == 1)
    {
        goto L130;
    }
    q__1.real() = csq.real() * s1.real() - csq.imag() * s1.imag(), q__1.imag() = csq.real() * s1.imag() + csq.imag() * s1.real();
    ai->real() = q__1.real(), ai->imag() = q__1.imag();
    return 0;
L130:
    q__2.real() = -z__->real(), q__2.imag() = -z__->imag();
    q__1.real() = q__2.real() * s1.real() - q__2.imag() * s1.imag(), q__1.imag() = q__2.real() * s1.imag() + q__2.imag() * s1.real();
    ai->real() = q__1.real(), ai->imag() = q__1.imag();
    return 0;
L140:
    q__2.real() = sfac, q__2.imag() = (float)0.;
    q__1.real() = s1.real() * q__2.real() - s1.imag() * q__2.imag(), q__1.imag() = s1.real() * q__2.imag() + s1.imag() * q__2.real();
    s1.real() = q__1.real(), s1.imag() = q__1.imag();
    if(*id == 1)
    {
        goto L150;
    }
    q__1.real() = s1.real() * csq.real() - s1.imag() * csq.imag(), q__1.imag() = s1.real() * csq.imag() + s1.imag() * csq.real();
    s1.real() = q__1.real(), s1.imag() = q__1.imag();
    r__1   = (float)1. / sfac;
    q__2.real() = r__1, q__2.imag() = (float)0.;
    q__1.real() = s1.real() * q__2.real() - s1.imag() * q__2.imag(), q__1.imag() = s1.real() * q__2.imag() + s1.imag() * q__2.real();
    ai->real() = q__1.real(), ai->imag() = q__1.imag();
    return 0;
L150:
    q__2.real() = -s1.real(), q__2.imag() = -s1.imag();
    q__1.real() = q__2.real() * z__->real() - q__2.imag() * z__->imag(), q__1.imag() = q__2.real() * z__->imag() + q__2.imag() * z__->real();
    s1.real() = q__1.real(), s1.imag() = q__1.imag();
    r__1   = (float)1. / sfac;
    q__2.real() = r__1, q__2.imag() = (float)0.;
    q__1.real() = s1.real() * q__2.real() - s1.imag() * q__2.imag(), q__1.imag() = s1.real() * q__2.imag() + s1.imag() * q__2.real();
    ai->real() = q__1.real(), ai->imag() = q__1.imag();
    return 0;
L160:
    aa   = r1mach_(&c__1) * (float)1e3;
    s1.real() = (float)0., s1.imag() = (float)0.;
    if(*id == 1)
    {
        goto L170;
    }
    if(az > aa)
    {
        q__2.real() = c2, q__2.imag() = (float)0.;
        q__1.real() = q__2.real() * z__->real() - q__2.imag() * z__->imag(), q__1.imag() = q__2.real() * z__->imag() + q__2.imag() * z__->real();
        s1.real() = q__1.real(), s1.imag() = q__1.imag();
    }
    q__2.real() = c1, q__2.imag() = (float)0.;
    q__1.real() = q__2.real() - s1.real(), q__1.imag() = q__2.imag() - s1.imag();
    ai->real() = q__1.real(), ai->imag() = q__1.imag();
    return 0;
L170:
    q__2.real() = c2, q__2.imag() = (float)0.;
    q__1.real() = -q__2.real(), q__1.imag() = -q__2.imag();
    ai->real() = q__1.real(), ai->imag() = q__1.imag();
    aa = sqrt(aa);
    if(az > aa)
    {
        q__2.real() = z__->real() * z__->real() - z__->imag() * z__->imag(), q__2.imag() = z__->real() * z__->imag() + z__->imag() * z__->real();
        q__1.real() = q__2.real() * (float).5 - q__2.imag() * (float)0., q__1.imag() = q__2.real() * (float)0. + q__2.imag() * (float).5;
        s1.real() = q__1.real(), s1.imag() = q__1.imag();
    }
    q__3.real() = c1, q__3.imag() = (float)0.;
    q__2.real() = s1.real() * q__3.real() - s1.imag() * q__3.imag(), q__2.imag() = s1.real() * q__3.imag() + s1.imag() * q__3.real();
    q__1.real() = ai->real() + q__2.real(), q__1.imag() = ai->imag() + q__2.imag();
    ai->real() = q__1.real(), ai->imag() = q__1.imag();
    return 0;
L180:
    *nz   = 1;
    ai->real() = (float)0., ai->imag() = (float)0.;
    return 0;
L240:
    *nz   = 0;
    *ierr = 2;
    return 0;
L250:
    if(nn == -1)
    {
        goto L240;
    }
    *nz   = 0;
    *ierr = 5;
    return 0;
L260:
    *ierr = 4;
    *nz   = 0;
    return 0;
} /* cairy_ */

 static int casyi_(System::Complex<float>* z__, float* fnu, int32* kode, int32* n, System::Complex<float>* y, int32* nz, float* rl, float* tol, float* elim, float* alim)
{
    /* Initialized data */

    static float    pi    = (float)3.14159265358979324;
    static float    rtpi  = (float).159154943091895336;
    static System::Complex<float> czero = {(float)0., (float)0.};
    static System::Complex<float> cone  = {(float)1., (float)0.};

    /* System generated locals */
    int32 i__1, i__2, i__3, i__4;
    float    r__1;
    System::Complex<float> q__1, q__2, q__3, q__4, q__5, q__6;

    

    
    int32           i__, j, k, m;
    float              s, x;
    System::Complex<float>           p1, s2;
    float              aa, bb;
    int32           ib;
    float              ak, bk;
    System::Complex<float>           ck, dk;
    int32           il, jl;
    float              az;
    int32           nn;
    System::Complex<float>           cz, ez, rz;
    float              yy;
    System::Complex<float>           ak1, cs1, cs2;
    float              fdn, arg, acz, aez, arm, sgn;
    int32           inu;
    float              sqk, dnu2, rtr1, dfnu, atol;
    int32           koded;
    

    /* *********************************************************************72 */

    /* c CASYI computs the I Bessel function. */

    /* ***BEGIN PROLOGUE  CASYI */
    /* ***REFER TO  CBESI,CBESK */

    /*     CASYI COMPUTES THE I BESSEL FUNCTION FOR REAL(Z).GE.0.0 BY */
    /*     MEANS OF THE ASYMPTOTIC EXPANSION FOR LARGE CABS(Z) IN THE */
    /*     REGION CABS(Z).GT.MAX(RL,FNU*FNU/2). NZ=0 IS A NORMAL RETURN. */
    /*     NZ.LT.0 INDICATES AN OVERFLOW ON KODE=1. */

    /* ***ROUTINES CALLED  R1MACH */
    /* ***END PROLOGUE  CASYI */
    /* Parameter adjustments */
    --y;

    /* Function Body */

    *nz  = 0;
    az   = c_abs(z__);
    x    = z__->real();
    arm  = r1mach_(&c__1) * (float)1e3;
    rtr1 = sqrt(arm);
    il   = std::min(2L, *n);
    dfnu = *fnu + (float)(*n - il);
    /* ----------------------------------------------------------------------- */
    /*     OVERFLOW TEST */
    /* ----------------------------------------------------------------------- */
    q__2.real() = rtpi, q__2.imag() = (float)0.;
    c_div(&q__1, &q__2, z__);
    ak1.real() = q__1.real(), ak1.imag() = q__1.imag();
    c_sqrt(&q__1, &ak1);
    ak1.real() = q__1.real(), ak1.imag() = q__1.imag();
    cz.real() = z__->real(), cz.imag() = z__->imag();
    if(*kode == 2)
    {
        q__2.real() = x, q__2.imag() = (float)0.;
        q__1.real() = z__->real() - q__2.real(), q__1.imag() = z__->imag() - q__2.imag();
        cz.real() = q__1.real(), cz.imag() = q__1.imag();
    }
    acz = cz.real();
    if(abs(acz) > *elim)
    {
        goto L80;
    }
    dnu2  = dfnu + dfnu;
    koded = 1;
    if(abs(acz) > *alim && *n > 2)
    {
        goto L10;
    }
    koded = 0;
    c_exp(&q__2, &cz);
    q__1.real() = ak1.real() * q__2.real() - ak1.imag() * q__2.imag(), q__1.imag() = ak1.real() * q__2.imag() + ak1.imag() * q__2.real();
    ak1.real() = q__1.real(), ak1.imag() = q__1.imag();
L10:
    fdn = (float)0.;
    if(dnu2 > rtr1)
    {
        fdn = dnu2 * dnu2;
    }
    q__1.real() = z__->real() * (float)8. - z__->imag() * (float)0., q__1.imag() = z__->real() * (float)0. + z__->imag() * (float)8.;
    ez.real() = q__1.real(), ez.imag() = q__1.imag();
    /* ----------------------------------------------------------------------- */
    /*     WHEN Z IS IMAGINARY, THE ERROR TEST MUST BE MADE RELATIVE TO THE */
    /*     FIRST RECIPROCAL POWER SINCE THIS IS THE LEADING TERM OF THE */
    /*     EXPANSION FOR THE IMAGINARY PART. */
    /* ----------------------------------------------------------------------- */
    aez  = az * (float)8.;
    s    = *tol / aez;
    jl   = (int32)(*rl + *rl) + 2;
    yy   = r_imag(z__);
    p1.real() = czero.real(), p1.imag() = czero.imag();
    if(yy == (float)0.)
    {
        goto L20;
    }
    /* ----------------------------------------------------------------------- */
    /*     CALCULATE EXP(PI*(0.5+FNU+N-IL)*I) TO MINIMIZE LOSSES OF */
    /*     SIGNIFICANCE WHEN FNU OR N IS LARGE */
    /* ----------------------------------------------------------------------- */
    inu = (int32)(*fnu);
    arg = (*fnu - (float)inu) * pi;
    inu = inu + *n - il;
    ak  = -sin(arg);
    bk  = cos(arg);
    if(yy < (float)0.)
    {
        bk = -bk;
    }
    q__1.real() = ak, q__1.imag() = bk;
    p1.real() = q__1.real(), p1.imag() = q__1.imag();
    if(inu % 2 == 1)
    {
        q__1.real() = -p1.real(), q__1.imag() = -p1.imag();
        p1.real() = q__1.real(), p1.imag() = q__1.imag();
    }
L20:
    i__1 = il;
    for(k = 1; k <= i__1; ++k)
    {
        sqk   = fdn - (float)1.;
        atol  = s * abs(sqk);
        sgn   = (float)1.;
        cs1.real() = cone.real(), cs1.imag() = cone.imag();
        cs2.real() = cone.real(), cs2.imag() = cone.imag();
        ck.real() = cone.real(), ck.imag() = cone.imag();
        ak   = (float)0.;
        aa   = (float)1.;
        bb   = aez;
        dk.real() = ez.real(), dk.imag() = ez.imag();
        i__2 = jl;
        for(j = 1; j <= i__2; ++j)
        {
            q__3.real() = sqk, q__3.imag() = (float)0.;
            q__2.real() = ck.real() * q__3.real() - ck.imag() * q__3.imag(), q__2.imag() = ck.real() * q__3.imag() + ck.imag() * q__3.real();
            c_div(&q__1, &q__2, &dk);
            ck.real() = q__1.real(), ck.imag() = q__1.imag();
            q__1.real() = cs2.real() + ck.real(), q__1.imag() = cs2.imag() + ck.imag();
            cs2.real() = q__1.real(), cs2.imag() = q__1.imag();
            sgn    = -sgn;
            q__3.real() = sgn, q__3.imag() = (float)0.;
            q__2.real() = ck.real() * q__3.real() - ck.imag() * q__3.imag(), q__2.imag() = ck.real() * q__3.imag() + ck.imag() * q__3.real();
            q__1.real() = cs1.real() + q__2.real(), q__1.imag() = cs1.imag() + q__2.imag();
            cs1.real() = q__1.real(), cs1.imag() = q__1.imag();
            q__1.real() = dk.real() + ez.real(), q__1.imag() = dk.imag() + ez.imag();
            dk.real() = q__1.real(), dk.imag() = q__1.imag();
            aa = aa * abs(sqk) / bb;
            bb += aez;
            ak += (float)8.;
            sqk -= ak;
            if(aa <= atol)
            {
                goto L40;
            }
            /* L30: */
        }
        goto L90;
    L40:
        s2.real() = cs1.real(), s2.imag() = cs1.imag();
        if(x + x < *elim)
        {
            q__3.real() = p1.real() * cs2.real() - p1.imag() * cs2.imag(), q__3.imag() = p1.real() * cs2.imag() + p1.imag() * cs2.real();
            q__6.real() = -z__->real(), q__6.imag() = -z__->imag();
            q__5.real() = q__6.real() - z__->real(), q__5.imag() = q__6.imag() - z__->imag();
            c_exp(&q__4, &q__5);
            q__2.real() = q__3.real() * q__4.real() - q__3.imag() * q__4.imag(), q__2.imag() = q__3.real() * q__4.imag() + q__3.imag() * q__4.real();
            q__1.real() = s2.real() + q__2.real(), q__1.imag() = s2.imag() + q__2.imag();
            s2.real() = q__1.real(), s2.imag() = q__1.imag();
        }
        fdn    = fdn + dfnu * (float)8. + (float)4.;
        q__1.real() = -p1.real(), q__1.imag() = -p1.imag();
        p1.real() = q__1.real(), p1.imag() = q__1.imag();
        m      = *n - il + k;
        i__2   = m;
        q__1.real() = s2.real() * ak1.real() - s2.imag() * ak1.imag(), q__1.imag() = s2.real() * ak1.imag() + s2.imag() * ak1.real();
        y[i__2].real() = q__1.real(), y[i__2].imag() = q__1.imag();
        /* L50: */
    }
    if(*n <= 2)
    {
        return 0;
    }
    nn     = *n;
    k      = nn - 2;
    ak     = (float)k;
    q__2.real() = cone.real() + cone.real(), q__2.imag() = cone.imag() + cone.imag();
    c_div(&q__1, &q__2, z__);
    rz.real() = q__1.real(), rz.imag() = q__1.imag();
    ib   = 3;
    i__1 = nn;
    for(i__ = ib; i__ <= i__1; ++i__)
    {
        i__2   = k;
        r__1   = ak + *fnu;
        q__4.real() = r__1, q__4.imag() = (float)0.;
        q__3.real() = q__4.real() * rz.real() - q__4.imag() * rz.imag(), q__3.imag() = q__4.real() * rz.imag() + q__4.imag() * rz.real();
        i__3   = k + 1;
        q__2.real() = q__3.real() * y[i__3].real() - q__3.imag() * y[i__3].imag(), q__2.imag() = q__3.real() * y[i__3].imag() + q__3.imag() * y[i__3].real();
        i__4   = k + 2;
        q__1.real() = q__2.real() + y[i__4].real(), q__1.imag() = q__2.imag() + y[i__4].imag();
        y[i__2].real() = q__1.real(), y[i__2].imag() = q__1.imag();
        ak += (float)-1.;
        --k;
        /* L60: */
    }
    if(koded == 0)
    {
        return 0;
    }
    c_exp(&q__1, &cz);
    ck.real() = q__1.real(), ck.imag() = q__1.imag();
    i__1 = nn;
    for(i__ = 1; i__ <= i__1; ++i__)
    {
        i__2   = i__;
        i__3   = i__;
        q__1.real() = y[i__3].real() * ck.real() - y[i__3].imag() * ck.imag(), q__1.imag() = y[i__3].real() * ck.imag() + y[i__3].imag() * ck.real();
        y[i__2].real() = q__1.real(), y[i__2].imag() = q__1.imag();
        /* L70: */
    }
    return 0;
L80:
    *nz = -1;
    return 0;
L90:
    *nz = -2;
    return 0;
} /* casyi_ */

 static int cbesh_(System::Complex<float>* z__, float* fnu, int32* kode, int32* m, int32* n, System::Complex<float>* cy, int32* nz, int32* ierr)
{
    /* Initialized data */

    static float hpi = (float)1.57079632679489662;

    /* System generated locals */
    int32 i__1, i__2;
    float    r__1, r__2;
    System::Complex<float> q__1, q__2;


    
    int32                     i__, k, k1, k2;
    float                        aa, bb, fn;
    int32                     mm;
    float                        az;
    int32                     ir;
    float                        rl;
    int32                     mr, nn, nw;
    float                        xn, yn;
    System::Complex<float>                     zn, zt;
    float                        xx, yy, dig, arg, aln, fmm, cpn, r1m5, ufl, sgn;
    int32                     nuf, inu;
    float                        tol, spn, alim, elim;
    System::Complex<float>                     csgn;
    float                        atol, rhpi;
    int32                     inuh;
    float                        fnul, rtol;
    float                        ascle;
    

    /* *********************************************************************72 */

    /* c CBESH computes a sequence of System::Complex<float> Hankel functions. */

    /* ***BEGIN PROLOGUE  CBESH */
    /* ***DATE WRITTEN   830501   (YYMMDD) */
    /* ***REVISION DATE  890801, 930101   (YYMMDD) */
    /* ***CATEGORY NO.  B5K */
    /* ***KEYWORDS  H-BESSEL FUNCTIONS,BESSEL FUNCTIONS OF COMPLEX ARGUMENT, */
    /*             BESSEL FUNCTIONS OF THIRD KIND,HANKEL FUNCTIONS */
    /* ***AUTHOR  AMOS, DONALD E., SANDIA NATIONAL LABORATORIES */
    /* ***PURPOSE  TO COMPUTE THE H-BESSEL FUNCTIONS OF A COMPLEX ARGUMENT */
    /* ***DESCRIPTION */

    /*         ON KODE=1, CBESH COMPUTES AN N MEMBER SEQUENCE OF COMPLEX */
    /*         HANKEL (BESSEL) FUNCTIONS CY(J)=H(M,FNU+J-1,Z) FOR KINDS M=1 */
    /*         OR 2, REAL, NONNEGATIVE ORDERS FNU+J-1, J=1,...,N, AND COMPLEX */
    /*         Z.NE.CMPLX(0.0E0,0.0E0) IN THE CUT PLANE -PI.LT.ARG(Z).LE.PI. */
    /*         ON KODE=2, CBESH COMPUTES THE SCALED HANKEL FUNCTIONS */

    /*         CY(I)=H(M,FNU+J-1,Z)*EXP(-MM*Z*I)       MM=3-2M,      I**2=-1. */

    /*         WHICH REMOVES THE EXPONENTIAL BEHAVIOR IN BOTH THE UPPER */
    /*         AND LOWER HALF PLANES. DEFINITIONS AND NOTATION ARE FOUND IN */
    /*         THE NBS HANDBOOK OF MATHEMATICAL FUNCTIONS (CONST. 1). */

    /*         INPUT */
    /*           Z      - Z=CMPLX(X,Y), Z.NE.CMPLX(0.,0.),-PI.LT.ARG(Z).LE.PI */
    /*           FNU    - ORDER OF INITIAL H FUNCTION, FNU.GE.0.0E0 */
    /*           KODE   - A PARAMETER TO INDICATE THE SCALING OPTION */
    /*                    KODE= 1  RETURNS */
    /*                             CY(J)=H(M,FNU+J-1,Z),      J=1,...,N */
    /*                        = 2  RETURNS */
    /*                             CY(J)=H(M,FNU+J-1,Z)*EXP(-I*Z*(3-2M)) */
    /*                                  J=1,...,N  ,  I**2=-1 */
    /*           M      - KIND OF HANKEL FUNCTION, M=1 OR 2 */
    /*           N      - NUMBER OF MEMBERS OF THE SEQUENCE, N.GE.1 */

    /*         OUTPUT */
    /*           CY     - A COMPLEX VECTOR WHOSE FIRST N COMPONENTS CONTAIN */
    /*                    VALUES FOR THE SEQUENCE */
    /*                    CY(J)=H(M,FNU+J-1,Z)  OR */
    /*                    CY(J)=H(M,FNU+J-1,Z)*EXP(-I*Z*(3-2M))  J=1,...,N */
    /*                    DEPENDING ON KODE, I**2=-1. */
    /*           NZ     - NUMBER OF COMPONENTS SET TO ZERO DUE TO UNDERFLOW, */
    /*                    NZ= 0   , NORMAL RETURN */
    /*                    NZ.GT.0 , FIRST NZ COMPONENTS OF CY SET TO ZERO */
    /*                              DUE TO UNDERFLOW, CY(J)=CMPLX(0.0,0.0) */
    /*                              J=1,...,NZ WHEN Y.GT.0.0 AND M=1 OR */
    /*                              Y.LT.0.0 AND M=2. FOR THE COMPLMENTARY */
    /*                              HALF PLANES, NZ STATES ONLY THE NUMBER */
    /*                              OF UNDERFLOWS. */
    /*           IERR    -ERROR FLAG */
    /*                    IERR=0, NORMAL RETURN - COMPUTATION COMPLETED */
    /*                    IERR=1, INPUT ERROR   - NO COMPUTATION */
    /*                    IERR=2, OVERFLOW      - NO COMPUTATION, FNU+N-1 TOO */
    /*                            LARGE OR CABS(Z) TOO SMALL OR BOTH */
    /*                    IERR=3, CABS(Z) OR FNU+N-1 LARGE - COMPUTATION DONE */
    /*                            BUT LOSSES OF SIGNIFCANCE BY ARGUMENT */
    /*                            REDUCTION PRODUCE LESS THAN HALF OF MACHINE */
    /*                            ACCURACY */
    /*                    IERR=4, CABS(Z) OR FNU+N-1 TOO LARGE - NO COMPUTA- */
    /*                            TION BECAUSE OF COMPLETE LOSSES OF SIGNIFI- */
    /*                            CANCE BY ARGUMENT REDUCTION */
    /*                    IERR=5, ERROR              - NO COMPUTATION, */
    /*                            ALGORITHM TERMINATION CONDITION NOT MET */

    /* ***LONG DESCRIPTION */

    /*         THE COMPUTATION IS CARRIED OUT BY THE RELATION */

    /*         H(M,FNU,Z)=(1/MP)*EXP(-MP*FNU)*K(FNU,Z*EXP(-MP)) */
    /*             MP=MM*HPI*I,  MM=3-2*M,  HPI=PI/2,  I**2=-1 */

    /*         FOR M=1 OR 2 WHERE THE K BESSEL FUNCTION IS COMPUTED FOR THE */
    /*         RIGHT HALF PLANE RE(Z).GE.0.0. THE K FUNCTION IS CONTINUED */
    /*         TO THE LEFT HALF PLANE BY THE RELATION */

    /*         K(FNU,Z*EXP(MP)) = EXP(-MP*FNU)*K(FNU,Z)-MP*I(FNU,Z) */
    /*         MP=MR*PI*I, MR=+1 OR -1, RE(Z).GT.0, I**2=-1 */

    /*         WHERE I(FNU,Z) IS THE I BESSEL FUNCTION. */

    /*         EXPONENTIAL DECAY OF H(M,FNU,Z) OCCURS IN THE UPPER HALF Z */
    /*         PLANE FOR M=1 AND THE LOWER HALF Z PLANE FOR M=2.  EXPONENTIAL */
    /*         GROWTH OCCURS IN THE COMPLEMENTARY HALF PLANES.  SCALING */
    /*         BY EXP(-MM*Z*I) REMOVES THE EXPONENTIAL BEHAVIOR IN THE */
    /*         WHOLE Z PLANE FOR Z TO INFINITY. */

    /*         FOR NEGATIVE ORDERS,THE FORMULAE */

    /*               H(1,-FNU,Z) = H(1,FNU,Z)*CEXP( PI*FNU*I) */
    /*               H(2,-FNU,Z) = H(2,FNU,Z)*CEXP(-PI*FNU*I) */
    /*                         I**2=-1 */

    /*         CAN BE USED. */

    /*         IN MOST COMPLEX VARIABLE COMPUTATION, ONE MUST EVALUATE ELE- */
    /*         MENTARY FUNCTIONS. WHEN THE MAGNITUDE OF Z OR FNU+N-1 IS */
    /*         LARGE, LOSSES OF SIGNIFICANCE BY ARGUMENT REDUCTION OCCUR. */
    /*         CONSEQUENTLY, IF EITHER ONE EXCEEDS U1=SQRT(0.5/UR), THEN */
    /*         LOSSES EXCEEDING HALF PRECISION ARE LIKELY AND AN ERROR FLAG */
    /*         IERR=3 IS TRIGGERED WHERE UR=R1MACH(4)=UNIT ROUNDOFF. ALSO */
    /*         IF EITHER IS LARGER THAN U2=0.5/UR, THEN ALL SIGNIFICANCE IS */
    /*         LOST AND IERR=4. IN ORDER TO USE THE INT FUNCTION, ARGUMENTS */
    /*         MUST BE FURTHER RESTRICTED NOT TO EXCEED THE LARGEST MACHINE */
    /*         INTEGER, U3=I1MACH(9). THUS, THE MAGNITUDE OF Z AND FNU+N-1 IS */
    /*         RESTRICTED BY MIN(U2,U3). ON 32 BIT MACHINES, U1,U2, AND U3 */
    /*         ARE APPROXIMATELY 2.0E+3, 4.2E+6, 2.1E+9 IN SINGLE PRECISION */
    /*         ARITHMETIC AND 1.3E+8, 1.8E+16, 2.1E+9 IN DOUBLE PRECISION */
    /*         ARITHMETIC RESPECTIVELY. THIS MAKES U2 AND U3 LIMITING IN */
    /*         THEIR RESPECTIVE ARITHMETICS. THIS MEANS THAT ONE CAN EXPECT */
    /*         TO RETAIN, IN THE WORST CASES ON 32 BIT MACHINES, NO DIGITS */
    /*         IN SINGLE AND ONLY 7 DIGITS IN DOUBLE PRECISION ARITHMETIC. */
    /*         SIMILAR CONSIDERATIONS HOLD FOR OTHER MACHINES. */

    /*         THE APPROXIMATE RELATIVE ERROR IN THE MAGNITUDE OF A COMPLEX */
    /*         BESSEL FUNCTION CAN BE EXPRESSED BY P*10**S WHERE P=MAX(UNIT */
    /*         ROUNDOFF,1.0E-18) IS THE NOMINAL PRECISION AND 10**S REPRE- */
    /*         SENTS THE INCREASE IN ERROR DUE TO ARGUMENT REDUCTION IN THE */
    /*         ELEMENTARY FUNCTIONS. HERE, S=MAX(1,ABS(LOG10(CABS(Z))), */
    /*         ABS(LOG10(FNU))) APPROXIMATELY (I.E. S=MAX(1,ABS(EXPONENT OF */
    /*         CABS(Z),ABS(EXPONENT OF FNU)) ). HOWEVER, THE PHASE ANGLE MAY */
    /*         HAVE ONLY ABSOLUTE ACCURACY. THIS IS MOST LIKELY TO OCCUR WHEN */
    /*         ONE COMPONENT (IN ABSOLUTE VALUE) IS LARGER THAN THE OTHER BY */
    /*         SEVERAL ORDERS OF MAGNITUDE. IF ONE COMPONENT IS 10**K LARGER */
    /*         THAN THE OTHER, THEN ONE CAN EXPECT ONLY MAX(ABS(LOG10(P))-K, */
    /*         0) SIGNIFICANT DIGITS; OR, STATED ANOTHER WAY, WHEN K EXCEEDS */
    /*         THE EXPONENT OF P, NO SIGNIFICANT DIGITS REMAIN IN THE SMALLER */
    /*         COMPONENT. HOWEVER, THE PHASE ANGLE RETAINS ABSOLUTE ACCURACY */
    /*         BECAUSE, IN COMPLEX ARITHMETIC WITH PRECISION P, THE SMALLER */
    /*         COMPONENT WILL NOT (AS A RULE) DECREASE BELOW P TIMES THE */
    /*         MAGNITUDE OF THE LARGER COMPONENT. IN THESE EXTREME CASES, */
    /*         THE PRINCIPAL PHASE ANGLE IS ON THE ORDER OF +P, -P, PI/2-P, */
    /*         OR -PI/2+P. */

    /* ***REFERENCES  HANDBOOK OF MATHEMATICAL FUNCTIONS BY M. ABRAMOWITZ */
    /*                 AND I. A. STEGUN, NBS AMS SERIES 55, U.S. DEPT. OF */
    /*                 COMMERCE, 1955. */

    /*               COMPUTATION OF BESSEL FUNCTIONS OF COMPLEX ARGUMENT */
    /*                 BY D. E. AMOS, SAND83-0083, MAY, 1983. */

    /*               COMPUTATION OF BESSEL FUNCTIONS OF COMPLEX ARGUMENT */
    /*                 AND LARGE ORDER BY D. E. AMOS, SAND83-0643, MAY, 1983 */

    /*               A SUBROUTINE PACKAGE FOR BESSEL FUNCTIONS OF A COMPLEX */
    /*                 ARGUMENT AND NONNEGATIVE ORDER BY D. E. AMOS, SAND85- */
    /*                 1018, MAY, 1985 */

    /*               A PORTABLE PACKAGE FOR BESSEL FUNCTIONS OF A COMPLEX */
    /*                 ARGUMENT AND NONNEGATIVE ORDER BY D. E. AMOS, ACM */
    /*                 TRANS. MATH. SOFTWARE, VOL. 12, NO. 3, SEPTEMBER 1986, */
    /*                 PP 265-273. */

    /* ***ROUTINES CALLED  CACON,CBKNU,CBUNK,CUOIK,I1MACH,R1MACH */
    /* ***END PROLOGUE  CBESH */

    /* Parameter adjustments */
    --cy;

    /* Function Body */

    /* ***FIRST EXECUTABLE STATEMENT  CBESH */
    *nz   = 0;
    xx    = z__->real();
    yy    = r_imag(z__);
    *ierr = 0;
    if(xx == (float)0. && yy == (float)0.)
    {
        *ierr = 1;
    }
    if(*fnu < (float)0.)
    {
        *ierr = 1;
    }
    if(*m < 1 || *m > 2)
    {
        *ierr = 1;
    }
    if(*kode < 1 || *kode > 2)
    {
        *ierr = 1;
    }
    if(*n < 1)
    {
        *ierr = 1;
    }
    if(*ierr != 0)
    {
        return 0;
    }
    nn = *n;
    /* ----------------------------------------------------------------------- */
    /*     SET PARAMETERS RELATED TO MACHINE CONSTANTS. */
    /*     TOL IS THE APPROXIMATE UNIT ROUNDOFF LIMITED TO 1.0E-18. */
    /*     ELIM IS THE APPROXIMATE EXPONENTIAL OVER- AND UNDERFLOW LIMIT. */
    /*     EXP(-ELIM).LT.EXP(-ALIM)=EXP(-ELIM)/TOL    AND */
    /*     EXP(ELIM).GT.EXP(ALIM)=EXP(ELIM)*TOL       ARE INTERVALS NEAR */
    /*     UNDERFLOW AND OVERFLOW LIMITS WHERE SCALED ARITHMETIC IS DONE. */
    /*     RL IS THE LOWER BOUNDARY OF THE ASYMPTOTIC EXPANSION FOR LARGE Z. */
    /*     DIG = NUMBER OF BASE 10 DIGITS IN TOL = 10**(-DIG). */
    /*     FNUL IS THE LOWER BOUNDARY OF THE ASYMPTOTIC SERIES FOR LARGE FNU */
    /* ----------------------------------------------------------------------- */
    /* Computing MAX */
    r__1 = r1mach_(&c__4);
    tol  = max(r__1, (float)1e-18);
    k1   = i1mach_(&c__12);
    k2   = i1mach_(&c__13);
    r1m5 = r1mach_(&c__5);
    /* Computing MIN */
    i__1 = abs(k1), i__2 = abs(k2);
    k    = std::min(i__1, i__2);
    elim = ((float)k * r1m5 - (float)3.) * (float)2.303;
    k1   = i1mach_(&c__11) - 1;
    aa   = r1m5 * (float)k1;
    dig  = std::min(aa, (float)18.);
    aa *= (float)2.303;
    /* Computing MAX */
    r__1   = -aa;
    alim   = elim + max(r__1, (float)-41.45);
    fnul   = (dig - (float)3.) * (float)6. + (float)10.;
    rl     = dig * (float)1.2 + (float)3.;
    fn     = *fnu + (float)(nn - 1);
    mm     = 3 - *m - *m;
    fmm    = (float)mm;
    r__1   = -fmm;
    q__2.real() = (float)0., q__2.imag() = r__1;
    q__1.real() = z__->real() * q__2.real() - z__->imag() * q__2.imag(), q__1.imag() = z__->real() * q__2.imag() + z__->imag() * q__2.real();
    zn.real() = q__1.real(), zn.imag() = q__1.imag();
    xn = zn.real();
    yn = r_imag(&zn);
    az = c_abs(z__);
    /* ----------------------------------------------------------------------- */
    /*     TEST FOR RANGE */
    /* ----------------------------------------------------------------------- */
    aa = (float).5 / tol;
    bb = (float)i1mach_(&c__9) * (float).5;
    aa = std::min(aa, bb);
    if(az > aa)
    {
        goto L240;
    }
    if(fn > aa)
    {
        goto L240;
    }
    aa = sqrt(aa);
    if(az > aa)
    {
        *ierr = 3;
    }
    if(fn > aa)
    {
        *ierr = 3;
    }
    /* ----------------------------------------------------------------------- */
    /*     OVERFLOW TEST ON THE LAST MEMBER OF THE SEQUENCE */
    /* ----------------------------------------------------------------------- */
    ufl = r1mach_(&c__1) * (float)1e3;
    if(az < ufl)
    {
        goto L220;
    }
    if(*fnu > fnul)
    {
        goto L90;
    }
    if(fn <= (float)1.)
    {
        goto L70;
    }
    if(fn > (float)2.)
    {
        goto L60;
    }
    if(az > tol)
    {
        goto L70;
    }
    arg = az * (float).5;
    aln = -fn * log(arg);
    if(aln > elim)
    {
        goto L220;
    }
    goto L70;
L60:
    cuoik_(&zn, fnu, kode, &c__2, &nn, &cy[1], &nuf, &tol, &elim, &alim);
    if(nuf < 0)
    {
        goto L220;
    }
    *nz += nuf;
    nn -= nuf;
    /* ----------------------------------------------------------------------- */
    /*     HERE NN=N OR NN=0 SINCE NUF=0,NN, OR -1 ON RETURN FROM CUOIK */
    /*     IF NUF=NN, THEN CY(I)=CZERO FOR ALL I */
    /* ----------------------------------------------------------------------- */
    if(nn == 0)
    {
        goto L130;
    }
L70:
    if(xn < (float)0. || xn == (float)0. && yn < (float)0. && *m == 2)
    {
        goto L80;
    }
    /* ----------------------------------------------------------------------- */
    /*     RIGHT HALF PLANE COMPUTATION, XN.GE.0. .AND. (XN.NE.0. .OR. */
    /*     YN.GE.0. .OR. M=1) */
    /* ----------------------------------------------------------------------- */
    cbknu_(&zn, fnu, kode, &nn, &cy[1], nz, &tol, &elim, &alim);
    goto L110;
/* ----------------------------------------------------------------------- */
/*     LEFT HALF PLANE COMPUTATION */
/* ----------------------------------------------------------------------- */
L80:
    mr = -mm;
    cacon_(&zn, fnu, kode, &mr, &nn, &cy[1], &nw, &rl, &fnul, &tol, &elim, &alim);
    if(nw < 0)
    {
        goto L230;
    }
    *nz = nw;
    goto L110;
L90:
    /* ----------------------------------------------------------------------- */
    /*     UNIFORM ASYMPTOTIC EXPANSIONS FOR FNU.GT.FNUL */
    /* ----------------------------------------------------------------------- */
    mr = 0;
    if(xn >= (float)0. && (xn != (float)0. || yn >= (float)0. || *m != 2))
    {
        goto L100;
    }
    mr = -mm;
    if(xn == (float)0. && yn < (float)0.)
    {
        q__1.real() = -zn.real(), q__1.imag() = -zn.imag();
        zn.real() = q__1.real(), zn.imag() = q__1.imag();
    }
L100:
    cbunk_(&zn, fnu, kode, &mr, &nn, &cy[1], &nw, &tol, &elim, &alim);
    if(nw < 0)
    {
        goto L230;
    }
    *nz += nw;
L110:
    /* ----------------------------------------------------------------------- */
    /*     H(M,FNU,Z) = -FMM*(I/HPI)*(ZT**FNU)*K(FNU,-Z*ZT) */

    /*     ZT=EXP(-FMM*HPI*I) = CMPLX(0.0,-FMM), FMM=3-2*M, M=1,2 */
    /* ----------------------------------------------------------------------- */
    r__1 = -fmm;
    sgn  = r_sign(&hpi, &r__1);
    /* ----------------------------------------------------------------------- */
    /*     CALCULATE EXP(FNU*HPI*I) TO MINIMIZE LOSSES OF SIGNIFICANCE */
    /*     WHEN FNU IS LARGE */
    /* ----------------------------------------------------------------------- */
    inu  = (int32)(*fnu);
    inuh = inu / 2;
    ir   = inu - (inuh << 1);
    arg  = (*fnu - (float)(inu - ir)) * sgn;
    rhpi = (float)1. / sgn;
    cpn  = rhpi * cos(arg);
    spn  = rhpi * sin(arg);
    /*     ZN = CMPLX(-SPN,CPN) */
    r__1   = -spn;
    q__1.real() = r__1, q__1.imag() = cpn;
    csgn.real() = q__1.real(), csgn.imag() = q__1.imag();
    /*     IF (MOD(INUH,2).EQ.1) ZN = -ZN */
    if(inuh % 2 == 1)
    {
        q__1.real() = -csgn.real(), q__1.imag() = -csgn.imag();
        csgn.real() = q__1.real(), csgn.imag() = q__1.imag();
    }
    r__1   = -fmm;
    q__1.real() = (float)0., q__1.imag() = r__1;
    zt.real() = q__1.real(), zt.imag() = q__1.imag();
    rtol  = (float)1. / tol;
    ascle = ufl * rtol;
    i__1  = nn;
    for(i__ = 1; i__ <= i__1; ++i__)
    {
        /*       CY(I) = CY(I)*ZN */
        /*       ZN = ZN*ZT */
        i__2 = i__;
        zn.real() = cy[i__2].real(), zn.imag() = cy[i__2].imag();
        aa   = zn.real();
        bb   = r_imag(&zn);
        atol = (float)1.;
        /* Computing MAX */
        r__1 = abs(aa), r__2 = abs(bb);
        if(max(r__1, r__2) > ascle)
        {
            goto L125;
        }
        q__2.real() = rtol, q__2.imag() = (float)0.;
        q__1.real() = zn.real() * q__2.real() - zn.imag() * q__2.imag(), q__1.imag() = zn.real() * q__2.imag() + zn.imag() * q__2.real();
        zn.real() = q__1.real(), zn.imag() = q__1.imag();
        atol = tol;
    L125:
        q__1.real() = zn.real() * csgn.real() - zn.imag() * csgn.imag(), q__1.imag() = zn.real() * csgn.imag() + zn.imag() * csgn.real();
        zn.real() = q__1.real(), zn.imag() = q__1.imag();
        i__2   = i__;
        q__2.real() = atol, q__2.imag() = (float)0.;
        q__1.real() = zn.real() * q__2.real() - zn.imag() * q__2.imag(), q__1.imag() = zn.real() * q__2.imag() + zn.imag() * q__2.real();
        cy[i__2].real() = q__1.real(), cy[i__2].imag() = q__1.imag();
        q__1.real() = csgn.real() * zt.real() - csgn.imag() * zt.imag(), q__1.imag() = csgn.real() * zt.imag() + csgn.imag() * zt.real();
        csgn.real() = q__1.real(), csgn.imag() = q__1.imag();
        /* L120: */
    }
    return 0;
L130:
    if(xn < (float)0.)
    {
        goto L220;
    }
    return 0;
L220:
    *ierr = 2;
    *nz   = 0;
    return 0;
L230:
    if(nw == -1)
    {
        goto L220;
    }
    *nz   = 0;
    *ierr = 5;
    return 0;
L240:
    *nz   = 0;
    *ierr = 4;
    return 0;
} /* cbesh_ */

 static int cbesi_(System::Complex<float>* z__, float* fnu, int32* kode, int32* n, System::Complex<float>* cy, int32* nz, int32* ierr)
{
    /* Initialized data */

    static float    pi   = (float)3.14159265358979324;
    static System::Complex<float> cone = {(float)1., (float)0.};

    /* System generated locals */
    int32 i__1, i__2;
    float    r__1, r__2;
    System::Complex<float> q__1, q__2;



    
    int32                     i__, k, k1, k2;
    float                        s1, s2, aa, bb, fn, az;
    int32                     nn;
    float                        rl;
    System::Complex<float>                     zn;
    float                        xx, yy, dig, arg, r1m5;
    int32                     inu;
    float                        tol, alim, elim;
    System::Complex<float>                     csgn;
    float                        atol, fnul, rtol, ascle;

    /* *********************************************************************72 */

    /* c CBESI computes a sequence of System::Complex<float> Bessel I functions. */

    /* ***BEGIN PROLOGUE  CBESI */
    /* ***DATE WRITTEN   830501   (YYMMDD) */
    /* ***REVISION DATE  890801, 930101   (YYMMDD) */
    /* ***CATEGORY NO.  B5K */
    /* ***KEYWORDS  I-BESSEL FUNCTION,COMPLEX BESSEL FUNCTION, */
    /*             MODIFIED BESSEL FUNCTION OF THE FIRST KIND */
    /* ***AUTHOR  AMOS, DONALD E., SANDIA NATIONAL LABORATORIES */
    /* ***PURPOSE  TO COMPUTE I-BESSEL FUNCTIONS OF COMPLEX ARGUMENT */
    /* ***DESCRIPTION */

    /*         ON KODE=1, CBESI COMPUTES AN N MEMBER SEQUENCE OF COMPLEX */
    /*         BESSEL FUNCTIONS CY(J)=I(FNU+J-1,Z) FOR REAL, NONNEGATIVE */
    /*         ORDERS FNU+J-1, J=1,...,N AND COMPLEX Z IN THE CUT PLANE */
    /*         -PI.LT.ARG(Z).LE.PI. ON KODE=2, CBESI RETURNS THE SCALED */
    /*         FUNCTIONS */

    /*         CY(J)=EXP(-ABS(X))*I(FNU+J-1,Z)   J = 1,...,N , X=REAL(Z) */

    /*         WITH THE EXPONENTIAL GROWTH REMOVED IN BOTH THE LEFT AND */
    /*         RIGHT HALF PLANES FOR Z TO INFINITY. DEFINITIONS AND */
    /*         NOTATION ARE FOUND IN THE NBS HANDBOOK OF MATHEMATICAL */
    /*         FUNCTIONS (CONST.1) */

    /*         INPUT */
    /*           Z      - Z=CMPLX(X,Y),  -PI.LT.ARG(Z).LE.PI */
    /*           FNU    - ORDER OF INITIAL I FUNCTION, FNU.GE.0.0E0 */
    /*           KODE   - A PARAMETER TO INDICATE THE SCALING OPTION */
    /*                    KODE= 1  RETURNS */
    /*                             CY(J)=I(FNU+J-1,Z), J=1,...,N */
    /*                        = 2  RETURNS */
    /*                             CY(J)=I(FNU+J-1,Z)*EXP(-ABS(X)), J=1,...,N */
    /*           N      - NUMBER OF MEMBERS OF THE SEQUENCE, N.GE.1 */

    /*         OUTPUT */
    /*           CY     - A COMPLEX VECTOR WHOSE FIRST N COMPONENTS CONTAIN */
    /*                    VALUES FOR THE SEQUENCE */
    /*                    CY(J)=I(FNU+J-1,Z)  OR */
    /*                    CY(J)=I(FNU+J-1,Z)*EXP(-ABS(X))  J=1,...,N */
    /*                    DEPENDING ON KODE, X=REAL(Z) */
    /*           NZ     - NUMBER OF COMPONENTS SET TO ZERO DUE TO UNDERFLOW, */
    /*                    NZ= 0   , NORMAL RETURN */
    /*                    NZ.GT.0 , LAST NZ COMPONENTS OF CY SET TO ZERO */
    /*                              DUE TO UNDERFLOW, CY(J)=CMPLX(0.0,0.0), */
    /*                              J = N-NZ+1,...,N */
    /*           IERR   - ERROR FLAG */
    /*                    IERR=0, NORMAL RETURN - COMPUTATION COMPLETED */
    /*                    IERR=1, INPUT ERROR   - NO COMPUTATION */
    /*                    IERR=2, OVERFLOW      - NO COMPUTATION, REAL(Z) TOO */
    /*                            LARGE ON KODE=1 */
    /*                    IERR=3, CABS(Z) OR FNU+N-1 LARGE - COMPUTATION DONE */
    /*                            BUT LOSSES OF SIGNIFCANCE BY ARGUMENT */
    /*                            REDUCTION PRODUCE LESS THAN HALF OF MACHINE */
    /*                            ACCURACY */
    /*                    IERR=4, CABS(Z) OR FNU+N-1 TOO LARGE - NO COMPUTA- */
    /*                            TION BECAUSE OF COMPLETE LOSSES OF SIGNIFI- */
    /*                            CANCE BY ARGUMENT REDUCTION */
    /*                    IERR=5, ERROR              - NO COMPUTATION, */
    /*                            ALGORITHM TERMINATION CONDITION NOT MET */

    /* ***LONG DESCRIPTION */

    /*         THE COMPUTATION IS CARRIED OUT BY THE POWER SERIES FOR */
    /*         SMALL CABS(Z), THE ASYMPTOTIC EXPANSION FOR LARGE CABS(Z), */
    /*         THE MILLER ALGORITHM NORMALIZED BY THE WRONSKIAN AND A */
    /*         NEUMANN SERIES FOR IMTERMEDIATE MAGNITUDES, AND THE */
    /*         UNIFORM ASYMPTOTIC EXPANSIONS FOR I(FNU,Z) AND J(FNU,Z) */
    /*         FOR LARGE ORDERS. BACKWARD RECURRENCE IS USED TO GENERATE */
    /*         SEQUENCES OR REDUCE ORDERS WHEN NECESSARY. */

    /*         THE CALCULATIONS ABOVE ARE DONE IN THE RIGHT HALF PLANE AND */
    /*         CONTINUED INTO THE LEFT HALF PLANE BY THE FORMULA */

    /*         I(FNU,Z*EXP(M*PI)) = EXP(M*PI*FNU)*I(FNU,Z)  REAL(Z).GT.0.0 */
    /*                       M = +I OR -I,  I**2=-1 */

    /*         FOR NEGATIVE ORDERS,THE FORMULA */

    /*              I(-FNU,Z) = I(FNU,Z) + (2/PI)*SIN(PI*FNU)*K(FNU,Z) */

    /*         CAN BE USED. HOWEVER,FOR LARGE ORDERS CLOSE TO INTEGERS, THE */
    /*         THE FUNCTION CHANGES RADICALLY. WHEN FNU IS A LARGE POSITIVE */
    /*         INTEGER,THE MAGNITUDE OF I(-FNU,Z)=I(FNU,Z) IS A LARGE */
    /*         NEGATIVE POWER OF TEN. BUT WHEN FNU IS NOT AN INTEGER, */
    /*         K(FNU,Z) DOMINATES IN MAGNITUDE WITH A LARGE POSITIVE POWER OF */
    /*         TEN AND THE MOST THAT THE SECOND TERM CAN BE REDUCED IS BY */
    /*         UNIT ROUNDOFF FROM THE COEFFICIENT. THUS, WIDE CHANGES CAN */
    /*         OCCUR WITHIN UNIT ROUNDOFF OF A LARGE INTEGER FOR FNU. HERE, */
    /*         LARGE MEANS FNU.GT.CABS(Z). */

    /*         IN MOST COMPLEX VARIABLE COMPUTATION, ONE MUST EVALUATE ELE- */
    /*         MENTARY FUNCTIONS. WHEN THE MAGNITUDE OF Z OR FNU+N-1 IS */
    /*         LARGE, LOSSES OF SIGNIFICANCE BY ARGUMENT REDUCTION OCCUR. */
    /*         CONSEQUENTLY, IF EITHER ONE EXCEEDS U1=SQRT(0.5/UR), THEN */
    /*         LOSSES EXCEEDING HALF PRECISION ARE LIKELY AND AN ERROR FLAG */
    /*         IERR=3 IS TRIGGERED WHERE UR=R1MACH(4)=UNIT ROUNDOFF. ALSO */
    /*         IF EITHER IS LARGER THAN U2=0.5/UR, THEN ALL SIGNIFICANCE IS */
    /*         LOST AND IERR=4. IN ORDER TO USE THE INT FUNCTION, ARGUMENTS */
    /*         MUST BE FURTHER RESTRICTED NOT TO EXCEED THE LARGEST MACHINE */
    /*         INTEGER, U3=I1MACH(9). THUS, THE MAGNITUDE OF Z AND FNU+N-1 IS */
    /*         RESTRICTED BY MIN(U2,U3). ON 32 BIT MACHINES, U1,U2, AND U3 */
    /*         ARE APPROXIMATELY 2.0E+3, 4.2E+6, 2.1E+9 IN SINGLE PRECISION */
    /*         ARITHMETIC AND 1.3E+8, 1.8E+16, 2.1E+9 IN DOUBLE PRECISION */
    /*         ARITHMETIC RESPECTIVELY. THIS MAKES U2 AND U3 LIMITING IN */
    /*         THEIR RESPECTIVE ARITHMETICS. THIS MEANS THAT ONE CAN EXPECT */
    /*         TO RETAIN, IN THE WORST CASES ON 32 BIT MACHINES, NO DIGITS */
    /*         IN SINGLE AND ONLY 7 DIGITS IN DOUBLE PRECISION ARITHMETIC. */
    /*         SIMILAR CONSIDERATIONS HOLD FOR OTHER MACHINES. */

    /*         THE APPROXIMATE RELATIVE ERROR IN THE MAGNITUDE OF A COMPLEX */
    /*         BESSEL FUNCTION CAN BE EXPRESSED BY P*10**S WHERE P=MAX(UNIT */
    /*         ROUNDOFF,1.0E-18) IS THE NOMINAL PRECISION AND 10**S REPRE- */
    /*         SENTS THE INCREASE IN ERROR DUE TO ARGUMENT REDUCTION IN THE */
    /*         ELEMENTARY FUNCTIONS. HERE, S=MAX(1,ABS(LOG10(CABS(Z))), */
    /*         ABS(LOG10(FNU))) APPROXIMATELY (I.E. S=MAX(1,ABS(EXPONENT OF */
    /*         CABS(Z),ABS(EXPONENT OF FNU)) ). HOWEVER, THE PHASE ANGLE MAY */
    /*         HAVE ONLY ABSOLUTE ACCURACY. THIS IS MOST LIKELY TO OCCUR WHEN */
    /*         ONE COMPONENT (IN ABSOLUTE VALUE) IS LARGER THAN THE OTHER BY */
    /*         SEVERAL ORDERS OF MAGNITUDE. IF ONE COMPONENT IS 10**K LARGER */
    /*         THAN THE OTHER, THEN ONE CAN EXPECT ONLY MAX(ABS(LOG10(P))-K, */
    /*         0) SIGNIFICANT DIGITS; OR, STATED ANOTHER WAY, WHEN K EXCEEDS */
    /*         THE EXPONENT OF P, NO SIGNIFICANT DIGITS REMAIN IN THE SMALLER */
    /*         COMPONENT. HOWEVER, THE PHASE ANGLE RETAINS ABSOLUTE ACCURACY */
    /*         BECAUSE, IN COMPLEX ARITHMETIC WITH PRECISION P, THE SMALLER */
    /*         COMPONENT WILL NOT (AS A RULE) DECREASE BELOW P TIMES THE */
    /*         MAGNITUDE OF THE LARGER COMPONENT. IN THESE EXTREME CASES, */
    /*         THE PRINCIPAL PHASE ANGLE IS ON THE ORDER OF +P, -P, PI/2-P, */
    /*         OR -PI/2+P. */

    /* ***REFERENCES  HANDBOOK OF MATHEMATICAL FUNCTIONS BY M. ABRAMOWITZ */
    /*                 AND I. A. STEGUN, NBS AMS SERIES 55, U.S. DEPT. OF */
    /*                 COMMERCE, 1955. */

    /*               COMPUTATION OF BESSEL FUNCTIONS OF COMPLEX ARGUMENT */
    /*                 BY D. E. AMOS, SAND83-0083, MAY, 1983. */

    /*               COMPUTATION OF BESSEL FUNCTIONS OF COMPLEX ARGUMENT */
    /*                 AND LARGE ORDER BY D. E. AMOS, SAND83-0643, MAY, 1983 */

    /*               A SUBROUTINE PACKAGE FOR BESSEL FUNCTIONS OF A COMPLEX */
    /*                 ARGUMENT AND NONNEGATIVE ORDER BY D. E. AMOS, SAND85- */
    /*                 1018, MAY, 1985 */

    /*               A PORTABLE PACKAGE FOR BESSEL FUNCTIONS OF A COMPLEX */
    /*                 ARGUMENT AND NONNEGATIVE ORDER BY D. E. AMOS, ACM */
    /*                 TRANS. MATH. SOFTWARE, VOL. 12, NO. 3, SEPTEMBER 1986, */
    /*                 PP 265-273. */

    /* ***ROUTINES CALLED  CBINU,I1MACH,R1MACH */
    /* ***END PROLOGUE  CBESI */
    /* Parameter adjustments */
    --cy;

    /* Function Body */

    /* ***FIRST EXECUTABLE STATEMENT  CBESI */
    *ierr = 0;
    *nz   = 0;
    if(*fnu < (float)0.)
    {
        *ierr = 1;
    }
    if(*kode < 1 || *kode > 2)
    {
        *ierr = 1;
    }
    if(*n < 1)
    {
        *ierr = 1;
    }
    if(*ierr != 0)
    {
        return 0;
    }
    xx = z__->real();
    yy = r_imag(z__);
    /* ----------------------------------------------------------------------- */
    /*     SET PARAMETERS RELATED TO MACHINE CONSTANTS. */
    /*     TOL IS THE APPROXIMATE UNIT ROUNDOFF LIMITED TO 1.0E-18. */
    /*     ELIM IS THE APPROXIMATE EXPONENTIAL OVER- AND UNDERFLOW LIMIT. */
    /*     EXP(-ELIM).LT.EXP(-ALIM)=EXP(-ELIM)/TOL    AND */
    /*     EXP(ELIM).GT.EXP(ALIM)=EXP(ELIM)*TOL       ARE INTERVALS NEAR */
    /*     UNDERFLOW AND OVERFLOW LIMITS WHERE SCALED ARITHMETIC IS DONE. */
    /*     RL IS THE LOWER BOUNDARY OF THE ASYMPTOTIC EXPANSION FOR LARGE Z. */
    /*     DIG = NUMBER OF BASE 10 DIGITS IN TOL = 10**(-DIG). */
    /*     FNUL IS THE LOWER BOUNDARY OF THE ASYMPTOTIC SERIES FOR LARGE FNU. */
    /* ----------------------------------------------------------------------- */
    /* Computing MAX */
    r__1 = r1mach_(&c__4);
    tol  = max(r__1, (float)1e-18);
    k1   = i1mach_(&c__12);
    k2   = i1mach_(&c__13);
    r1m5 = r1mach_(&c__5);
    /* Computing MIN */
    i__1 = abs(k1), i__2 = abs(k2);
    k    = std::min(i__1, i__2);
    elim = ((float)k * r1m5 - (float)3.) * (float)2.303;
    k1   = i1mach_(&c__11) - 1;
    aa   = r1m5 * (float)k1;
    dig  = std::min(aa, (float)18.);
    aa *= (float)2.303;
    /* Computing MAX */
    r__1 = -aa;
    alim = elim + max(r__1, (float)-41.45);
    rl   = dig * (float)1.2 + (float)3.;
    fnul = (dig - (float)3.) * (float)6. + (float)10.;
    az   = c_abs(z__);
    /* ----------------------------------------------------------------------- */
    /*     TEST FOR RANGE */
    /* ----------------------------------------------------------------------- */
    aa = (float).5 / tol;
    bb = (float)i1mach_(&c__9) * (float).5;
    aa = std::min(aa, bb);
    if(az > aa)
    {
        goto L140;
    }
    fn = *fnu + (float)(*n - 1);
    if(fn > aa)
    {
        goto L140;
    }
    aa = sqrt(aa);
    if(az > aa)
    {
        *ierr = 3;
    }
    if(fn > aa)
    {
        *ierr = 3;
    }
    zn.real() = z__->real(), zn.imag() = z__->imag();
    csgn.real() = cone.real(), csgn.imag() = cone.imag();
    if(xx >= (float)0.)
    {
        goto L40;
    }
    q__1.real() = -z__->real(), q__1.imag() = -z__->imag();
    zn.real() = q__1.real(), zn.imag() = q__1.imag();
    /* ----------------------------------------------------------------------- */
    /*     CALCULATE CSGN=EXP(FNU*PI*I) TO MINIMIZE LOSSES OF SIGNIFICANCE */
    /*     WHEN FNU IS LARGE */
    /* ----------------------------------------------------------------------- */
    inu = (int32)(*fnu);
    arg = (*fnu - (float)inu) * pi;
    if(yy < (float)0.)
    {
        arg = -arg;
    }
    s1     = cos(arg);
    s2     = sin(arg);
    q__1.real() = s1, q__1.imag() = s2;
    csgn.real() = q__1.real(), csgn.imag() = q__1.imag();
    if(inu % 2 == 1)
    {
        q__1.real() = -csgn.real(), q__1.imag() = -csgn.imag();
        csgn.real() = q__1.real(), csgn.imag() = q__1.imag();
    }
L40:
    cbinu_(&zn, fnu, kode, n, &cy[1], nz, &rl, &fnul, &tol, &elim, &alim);
    if(*nz < 0)
    {
        goto L120;
    }
    if(xx >= (float)0.)
    {
        return 0;
    }
    /* ----------------------------------------------------------------------- */
    /*     ANALYTIC CONTINUATION TO THE LEFT HALF PLANE */
    /* ----------------------------------------------------------------------- */
    nn = *n - *nz;
    if(nn == 0)
    {
        return 0;
    }
    rtol  = (float)1. / tol;
    ascle = r1mach_(&c__1) * rtol * (float)1e3;
    i__1  = nn;
    for(i__ = 1; i__ <= i__1; ++i__)
    {
        /*       CY(I) = CY(I)*CSGN */
        i__2 = i__;
        zn.real() = cy[i__2].real(), zn.imag() = cy[i__2].imag();
        aa   = zn.real();
        bb   = r_imag(&zn);
        atol = (float)1.;
        /* Computing MAX */
        r__1 = abs(aa), r__2 = abs(bb);
        if(max(r__1, r__2) > ascle)
        {
            goto L55;
        }
        q__2.real() = rtol, q__2.imag() = (float)0.;
        q__1.real() = zn.real() * q__2.real() - zn.imag() * q__2.imag(), q__1.imag() = zn.real() * q__2.imag() + zn.imag() * q__2.real();
        zn.real() = q__1.real(), zn.imag() = q__1.imag();
        atol = tol;
    L55:
        q__1.real() = zn.real() * csgn.real() - zn.imag() * csgn.imag(), q__1.imag() = zn.real() * csgn.imag() + zn.imag() * csgn.real();
        zn.real() = q__1.real(), zn.imag() = q__1.imag();
        i__2   = i__;
        q__2.real() = atol, q__2.imag() = (float)0.;
        q__1.real() = zn.real() * q__2.real() - zn.imag() * q__2.imag(), q__1.imag() = zn.real() * q__2.imag() + zn.imag() * q__2.real();
        cy[i__2].real() = q__1.real(), cy[i__2].imag() = q__1.imag();
        q__1.real() = -csgn.real(), q__1.imag() = -csgn.imag();
        csgn.real() = q__1.real(), csgn.imag() = q__1.imag();
        /* L50: */
    }
    return 0;
L120:
    if(*nz == -2)
    {
        goto L130;
    }
    *nz   = 0;
    *ierr = 2;
    return 0;
L130:
    *nz   = 0;
    *ierr = 5;
    return 0;
L140:
    *nz   = 0;
    *ierr = 4;
    return 0;
} /* cbesi_ */

 int cbesj_(System::Complex<float>* z__, float* fnu, int32* kode, int32* n, System::Complex<float>* cy, int32* nz, int32* ierr)
{
    /* Initialized data */

    static float hpi = (float)1.57079632679489662;

    /* System generated locals */
    int32 i__1, i__2;
    float    r__1, r__2;
    System::Complex<float> q__1, q__2;

  

    
    int32                     i__, k, k1, k2;
    float                        r1, r2, aa, bb;
    System::Complex<float>                     ci;
    float                        fn;
    int32                     nl;
    float                        az;
    int32                     ir;
    float                        rl;
    System::Complex<float>                     zn;
    float                        yy, dig, arg, r1m5;
    int32                     inu;
    float                        tol, alim, elim;
    System::Complex<float>                     csgn;
    float                        atol;
    int32                     inuh;
    float                        fnul, rtol, ascle;

    /* *********************************************************************72 */

    /* c CBESJ computes a sequence of System::Complex<float> Bessel J functions. */

    /* ***BEGIN PROLOGUE  CBESJ */
    /* ***DATE WRITTEN   830501   (YYMMDD) */
    /* ***REVISION DATE  890801, 930101   (YYMMDD) */
    /* ***CATEGORY NO.  B5K */
    /* ***KEYWORDS  J-BESSEL FUNCTION,BESSEL FUNCTION OF COMPLEX ARGUMENT, */
    /*             BESSEL FUNCTION OF FIRST KIND */
    /* ***AUTHOR  AMOS, DONALD E., SANDIA NATIONAL LABORATORIES */
    /* ***PURPOSE  TO COMPUTE THE J-BESSEL FUNCTION OF A COMPLEX ARGUMENT */
    /* ***DESCRIPTION */

    /*         ON KODE=1, CBESJ COMPUTES AN N MEMBER  SEQUENCE OF COMPLEX */
    /*         BESSEL FUNCTIONS CY(I)=J(FNU+I-1,Z) FOR REAL, NONNEGATIVE */
    /*         ORDERS FNU+I-1, I=1,...,N AND COMPLEX Z IN THE CUT PLANE */
    /*         -PI.LT.ARG(Z).LE.PI. ON KODE=2, CBESJ RETURNS THE SCALED */
    /*         FUNCTIONS */

    /*         CY(I)=EXP(-ABS(Y))*J(FNU+I-1,Z)   I = 1,...,N , Y=AIMAG(Z) */

    /*         WHICH REMOVE THE EXPONENTIAL GROWTH IN BOTH THE UPPER AND */
    /*         LOWER HALF PLANES FOR Z TO INFINITY. DEFINITIONS AND NOTATION */
    /*         ARE FOUND IN THE NBS HANDBOOK OF MATHEMATICAL FUNCTIONS */
    /*         (CONST. 1). */

    /*         INPUT */
    /*           Z      - Z=CMPLX(X,Y),  -PI.LT.ARG(Z).LE.PI */
    /*           FNU    - ORDER OF INITIAL J FUNCTION, FNU.GE.0.0E0 */
    /*           KODE   - A PARAMETER TO INDICATE THE SCALING OPTION */
    /*                    KODE= 1  RETURNS */
    /*                             CY(I)=J(FNU+I-1,Z), I=1,...,N */
    /*                        = 2  RETURNS */
    /*                             CY(I)=J(FNU+I-1,Z)*EXP(-ABS(Y)), I=1,... */
    /*           N      - NUMBER OF MEMBERS OF THE SEQUENCE, N.GE.1 */

    /*         OUTPUT */
    /*           CY     - A COMPLEX VECTOR WHOSE FIRST N COMPONENTS CONTAIN */
    /*                    VALUES FOR THE SEQUENCE */
    /*                    CY(I)=J(FNU+I-1,Z)  OR */
    /*                    CY(I)=J(FNU+I-1,Z)*EXP(-ABS(Y))  I=1,...,N */
    /*                    DEPENDING ON KODE, Y=AIMAG(Z). */
    /*           NZ     - NUMBER OF COMPONENTS SET TO ZERO DUE TO UNDERFLOW, */
    /*                    NZ= 0   , NORMAL RETURN */
    /*                    NZ.GT.0 , LAST NZ COMPONENTS OF CY SET TO ZERO */
    /*                              DUE TO UNDERFLOW, CY(I)=CMPLX(0.0,0.0), */
    /*                              I = N-NZ+1,...,N */
    /*           IERR   - ERROR FLAG */
    /*                    IERR=0, NORMAL RETURN - COMPUTATION COMPLETED */
    /*                    IERR=1, INPUT ERROR   - NO COMPUTATION */
    /*                    IERR=2, OVERFLOW      - NO COMPUTATION, AIMAG(Z) */
    /*                            TOO LARGE ON KODE=1 */
    /*                    IERR=3, CABS(Z) OR FNU+N-1 LARGE - COMPUTATION DONE */
    /*                            BUT LOSSES OF SIGNIFCANCE BY ARGUMENT */
    /*                            REDUCTION PRODUCE LESS THAN HALF OF MACHINE */
    /*                            ACCURACY */
    /*                    IERR=4, CABS(Z) OR FNU+N-1 TOO LARGE - NO COMPUTA- */
    /*                            TION BECAUSE OF COMPLETE LOSSES OF SIGNIFI- */
    /*                            CANCE BY ARGUMENT REDUCTION */
    /*                    IERR=5, ERROR              - NO COMPUTATION, */
    /*                            ALGORITHM TERMINATION CONDITION NOT MET */

    /* ***LONG DESCRIPTION */

    /*         THE COMPUTATION IS CARRIED OUT BY THE FORMULA */

    /*         J(FNU,Z)=EXP( FNU*PI*I/2)*I(FNU,-I*Z)    AIMAG(Z).GE.0.0 */

    /*         J(FNU,Z)=EXP(-FNU*PI*I/2)*I(FNU, I*Z)    AIMAG(Z).LT.0.0 */

    /*         WHERE I**2 = -1 AND I(FNU,Z) IS THE I BESSEL FUNCTION. */

    /*         FOR NEGATIVE ORDERS,THE FORMULA */

    /*              J(-FNU,Z) = J(FNU,Z)*COS(PI*FNU) - Y(FNU,Z)*SIN(PI*FNU) */

    /*         CAN BE USED. HOWEVER,FOR LARGE ORDERS CLOSE TO INTEGERS, THE */
    /*         THE FUNCTION CHANGES RADICALLY. WHEN FNU IS A LARGE POSITIVE */
    /*         INTEGER,THE MAGNITUDE OF J(-FNU,Z)=J(FNU,Z)*COS(PI*FNU) IS A */
    /*         LARGE NEGATIVE POWER OF TEN. BUT WHEN FNU IS NOT AN INTEGER, */
    /*         Y(FNU,Z) DOMINATES IN MAGNITUDE WITH A LARGE POSITIVE POWER OF */
    /*         TEN AND THE MOST THAT THE SECOND TERM CAN BE REDUCED IS BY */
    /*         UNIT ROUNDOFF FROM THE COEFFICIENT. THUS, WIDE CHANGES CAN */
    /*         OCCUR WITHIN UNIT ROUNDOFF OF A LARGE INTEGER FOR FNU. HERE, */
    /*         LARGE MEANS FNU.GT.CABS(Z). */

    /*         IN MOST COMPLEX VARIABLE COMPUTATION, ONE MUST EVALUATE ELE- */
    /*         MENTARY FUNCTIONS. WHEN THE MAGNITUDE OF Z OR FNU+N-1 IS */
    /*         LARGE, LOSSES OF SIGNIFICANCE BY ARGUMENT REDUCTION OCCUR. */
    /*         CONSEQUENTLY, IF EITHER ONE EXCEEDS U1=SQRT(0.5/UR), THEN */
    /*         LOSSES EXCEEDING HALF PRECISION ARE LIKELY AND AN ERROR FLAG */
    /*         IERR=3 IS TRIGGERED WHERE UR=R1MACH(4)=UNIT ROUNDOFF. ALSO */
    /*         IF EITHER IS LARGER THAN U2=0.5/UR, THEN ALL SIGNIFICANCE IS */
    /*         LOST AND IERR=4. IN ORDER TO USE THE INT FUNCTION, ARGUMENTS */
    /*         MUST BE FURTHER RESTRICTED NOT TO EXCEED THE LARGEST MACHINE */
    /*         INTEGER, U3=I1MACH(9). THUS, THE MAGNITUDE OF Z AND FNU+N-1 IS */
    /*         RESTRICTED BY MIN(U2,U3). ON 32 BIT MACHINES, U1,U2, AND U3 */
    /*         ARE APPROXIMATELY 2.0E+3, 4.2E+6, 2.1E+9 IN SINGLE PRECISION */
    /*         ARITHMETIC AND 1.3E+8, 1.8E+16, 2.1E+9 IN DOUBLE PRECISION */
    /*         ARITHMETIC RESPECTIVELY. THIS MAKES U2 AND U3 LIMITING IN */
    /*         THEIR RESPECTIVE ARITHMETICS. THIS MEANS THAT ONE CAN EXPECT */
    /*         TO RETAIN, IN THE WORST CASES ON 32 BIT MACHINES, NO DIGITS */
    /*         IN SINGLE AND ONLY 7 DIGITS IN DOUBLE PRECISION ARITHMETIC. */
    /*         SIMILAR CONSIDERATIONS HOLD FOR OTHER MACHINES. */

    /*         THE APPROXIMATE RELATIVE ERROR IN THE MAGNITUDE OF A COMPLEX */
    /*         BESSEL FUNCTION CAN BE EXPRESSED BY P*10**S WHERE P=MAX(UNIT */
    /*         ROUNDOFF,1.0E-18) IS THE NOMINAL PRECISION AND 10**S REPRE- */
    /*         SENTS THE INCREASE IN ERROR DUE TO ARGUMENT REDUCTION IN THE */
    /*         ELEMENTARY FUNCTIONS. HERE, S=MAX(1,ABS(LOG10(CABS(Z))), */
    /*         ABS(LOG10(FNU))) APPROXIMATELY (I.E. S=MAX(1,ABS(EXPONENT OF */
    /*         CABS(Z),ABS(EXPONENT OF FNU)) ). HOWEVER, THE PHASE ANGLE MAY */
    /*         HAVE ONLY ABSOLUTE ACCURACY. THIS IS MOST LIKELY TO OCCUR WHEN */
    /*         ONE COMPONENT (IN ABSOLUTE VALUE) IS LARGER THAN THE OTHER BY */
    /*         SEVERAL ORDERS OF MAGNITUDE. IF ONE COMPONENT IS 10**K LARGER */
    /*         THAN THE OTHER, THEN ONE CAN EXPECT ONLY MAX(ABS(LOG10(P))-K, */
    /*         0) SIGNIFICANT DIGITS; OR, STATED ANOTHER WAY, WHEN K EXCEEDS */
    /*         THE EXPONENT OF P, NO SIGNIFICANT DIGITS REMAIN IN THE SMALLER */
    /*         COMPONENT. HOWEVER, THE PHASE ANGLE RETAINS ABSOLUTE ACCURACY */
    /*         BECAUSE, IN COMPLEX ARITHMETIC WITH PRECISION P, THE SMALLER */
    /*         COMPONENT WILL NOT (AS A RULE) DECREASE BELOW P TIMES THE */
    /*         MAGNITUDE OF THE LARGER COMPONENT. IN THESE EXTREME CASES, */
    /*         THE PRINCIPAL PHASE ANGLE IS ON THE ORDER OF +P, -P, PI/2-P, */
    /*         OR -PI/2+P. */

    /* ***REFERENCES  HANDBOOK OF MATHEMATICAL FUNCTIONS BY M. ABRAMOWITZ */
    /*                 AND I. A. STEGUN, NBS AMS SERIES 55, U.S. DEPT. OF */
    /*                 COMMERCE, 1955. */

    /*               COMPUTATION OF BESSEL FUNCTIONS OF COMPLEX ARGUMENT */
    /*                 BY D. E. AMOS, SAND83-0083, MAY, 1983. */

    /*               COMPUTATION OF BESSEL FUNCTIONS OF COMPLEX ARGUMENT */
    /*                 AND LARGE ORDER BY D. E. AMOS, SAND83-0643, MAY, 1983 */

    /*               A SUBROUTINE PACKAGE FOR BESSEL FUNCTIONS OF A COMPLEX */
    /*                 ARGUMENT AND NONNEGATIVE ORDER BY D. E. AMOS, SAND85- */
    /*                 1018, MAY, 1985 */

    /*               A PORTABLE PACKAGE FOR BESSEL FUNCTIONS OF A COMPLEX */
    /*                 ARGUMENT AND NONNEGATIVE ORDER BY D. E. AMOS, ACM */
    /*                 TRANS. MATH. SOFTWARE, VOL. 12, NO. 3, SEPTEMBER 1986, */
    /*                 PP 265-273. */

    /* ***ROUTINES CALLED  CBINU,I1MACH,R1MACH */
    /* ***END PROLOGUE  CBESJ */

    /* Parameter adjustments */
    --cy;

    /* Function Body */

    /* ***FIRST EXECUTABLE STATEMENT  CBESJ */
    *ierr = 0;
    *nz   = 0;
    if(*fnu < (float)0.)
    {
        *ierr = 1;
    }
    if(*kode < 1 || *kode > 2)
    {
        *ierr = 1;
    }
    if(*n < 1)
    {
        *ierr = 1;
    }
    if(*ierr != 0)
    {
        return 0;
    }
    /* ----------------------------------------------------------------------- */
    /*     SET PARAMETERS RELATED TO MACHINE CONSTANTS. */
    /*     TOL IS THE APPROXIMATE UNIT ROUNDOFF LIMITED TO 1.0E-18. */
    /*     ELIM IS THE APPROXIMATE EXPONENTIAL OVER- AND UNDERFLOW LIMIT. */
    /*     EXP(-ELIM).LT.EXP(-ALIM)=EXP(-ELIM)/TOL    AND */
    /*     EXP(ELIM).GT.EXP(ALIM)=EXP(ELIM)*TOL       ARE INTERVALS NEAR */
    /*     UNDERFLOW AND OVERFLOW LIMITS WHERE SCALED ARITHMETIC IS DONE. */
    /*     RL IS THE LOWER BOUNDARY OF THE ASYMPTOTIC EXPANSION FOR LARGE Z. */
    /*     DIG = NUMBER OF BASE 10 DIGITS IN TOL = 10**(-DIG). */
    /*     FNUL IS THE LOWER BOUNDARY OF THE ASYMPTOTIC SERIES FOR LARGE FNU. */
    /* ----------------------------------------------------------------------- */
    /* Computing MAX */
    r__1 = r1mach_(&c__4);
    tol  = max(r__1, (float)1e-18);
    k1   = i1mach_(&c__12);
    k2   = i1mach_(&c__13);
    r1m5 = r1mach_(&c__5);
    /* Computing MIN */
    i__1 = abs(k1), i__2 = abs(k2);
    k    = std::min(i__1, i__2);
    elim = ((float)k * r1m5 - (float)3.) * (float)2.303;
    k1   = i1mach_(&c__11) - 1;
    aa   = r1m5 * (float)k1;
    dig  = std::min(aa, (float)18.);
    aa *= (float)2.303;
    /* Computing MAX */
    r__1 = -aa;
    alim = elim + max(r__1, (float)-41.45);
    rl   = dig * (float)1.2 + (float)3.;
    fnul = (dig - (float)3.) * (float)6. + (float)10.;
    ci.real() = (float)0., ci.imag() = (float)1.;
    yy = r_imag(z__);
    az = c_abs(z__);
    /* ----------------------------------------------------------------------- */
    /*     TEST FOR RANGE */
    /* ----------------------------------------------------------------------- */
    aa = (float).5 / tol;
    bb = (float)i1mach_(&c__9) * (float).5;
    aa = std::min(aa, bb);
    fn = *fnu + (float)(*n - 1);
    if(az > aa)
    {
        goto L140;
    }
    if(fn > aa)
    {
        goto L140;
    }
    aa = sqrt(aa);
    if(az > aa)
    {
        *ierr = 3;
    }
    if(fn > aa)
    {
        *ierr = 3;
    }
    /* ----------------------------------------------------------------------- */
    /*     CALCULATE CSGN=EXP(FNU*HPI*I) TO MINIMIZE LOSSES OF SIGNIFICANCE */
    /*     WHEN FNU IS LARGE */
    /* ----------------------------------------------------------------------- */
    inu    = (int32)(*fnu);
    inuh   = inu / 2;
    ir     = inu - (inuh << 1);
    arg    = (*fnu - (float)(inu - ir)) * hpi;
    r1     = cos(arg);
    r2     = sin(arg);
    q__1.real() = r1, q__1.imag() = r2;
    csgn.real() = q__1.real(), csgn.imag() = q__1.imag();
    if(inuh % 2 == 1)
    {
        q__1.real() = -csgn.real(), q__1.imag() = -csgn.imag();
        csgn.real() = q__1.real(), csgn.imag() = q__1.imag();
    }
    /* ----------------------------------------------------------------------- */
    /*     ZN IS IN THE RIGHT HALF PLANE */
    /* ----------------------------------------------------------------------- */
    q__2.real() = -z__->real(), q__2.imag() = -z__->imag();
    q__1.real() = q__2.real() * ci.real() - q__2.imag() * ci.imag(), q__1.imag() = q__2.real() * ci.imag() + q__2.imag() * ci.real();
    zn.real() = q__1.real(), zn.imag() = q__1.imag();
    if(yy >= (float)0.)
    {
        goto L40;
    }
    q__1.real() = -zn.real(), q__1.imag() = -zn.imag();
    zn.real() = q__1.real(), zn.imag() = q__1.imag();
    r_cnjg(&q__1, &csgn);
    csgn.real() = q__1.real(), csgn.imag() = q__1.imag();
    r_cnjg(&q__1, &ci);
    ci.real() = q__1.real(), ci.imag() = q__1.imag();
L40:
    cbinu_(&zn, fnu, kode, n, &cy[1], nz, &rl, &fnul, &tol, &elim, &alim);
    if(*nz < 0)
    {
        goto L120;
    }
    nl = *n - *nz;
    if(nl == 0)
    {
        return 0;
    }
    rtol  = (float)1. / tol;
    ascle = r1mach_(&c__1) * rtol * (float)1e3;
    i__1  = nl;
    for(i__ = 1; i__ <= i__1; ++i__)
    {
        /*       CY(I)=CY(I)*CSGN */
        i__2 = i__;
        zn.real() = cy[i__2].real(), zn.imag() = cy[i__2].imag();
        aa   = zn.real();
        bb   = r_imag(&zn);
        atol = (float)1.;
        /* Computing MAX */
        r__1 = abs(aa), r__2 = abs(bb);
        if(max(r__1, r__2) > ascle)
        {
            goto L55;
        }
        q__2.real() = rtol, q__2.imag() = (float)0.;
        q__1.real() = zn.real() * q__2.real() - zn.imag() * q__2.imag(), q__1.imag() = zn.real() * q__2.imag() + zn.imag() * q__2.real();
        zn.real() = q__1.real(), zn.imag() = q__1.imag();
        atol = tol;
    L55:
        q__1.real() = zn.real() * csgn.real() - zn.imag() * csgn.imag(), q__1.imag() = zn.real() * csgn.imag() + zn.imag() * csgn.real();
        zn.real() = q__1.real(), zn.imag() = q__1.imag();
        i__2   = i__;
        q__2.real() = atol, q__2.imag() = (float)0.;
        q__1.real() = zn.real() * q__2.real() - zn.imag() * q__2.imag(), q__1.imag() = zn.real() * q__2.imag() + zn.imag() * q__2.real();
        cy[i__2].real() = q__1.real(), cy[i__2].imag() = q__1.imag();
        q__1.real() = csgn.real() * ci.real() - csgn.imag() * ci.imag(), q__1.imag() = csgn.real() * ci.imag() + csgn.imag() * ci.real();
        csgn.real() = q__1.real(), csgn.imag() = q__1.imag();
        /* L50: */
    }
    return 0;
L120:
    if(*nz == -2)
    {
        goto L130;
    }
    *nz   = 0;
    *ierr = 2;
    return 0;
L130:
    *nz   = 0;
    *ierr = 5;
    return 0;
L140:
    *nz   = 0;
    *ierr = 4;
    return 0;
} /* cbesj_ */

 int cbesk_(System::Complex<float>* z__, float* fnu, int32* kode, int32* n, System::Complex<float>* cy, int32* nz, int32* ierr)
{
    /* System generated locals */
    int32 i__1, i__2;
    float    r__1;


    
    int32                     k, k1, k2;
    float                        aa, bb, fn, az;
    int32                     nn;
    float                        rl;
    int32                     mr, nw;
    float                        xx, yy, dig, arg, aln, r1m5, ufl;
    int32                     nuf;
    float                        tol, alim, elim, fnul;
    

    /* *********************************************************************72 */

    /* c CBESK computes a sequence of System::Complex<float> Bessel K functions. */

    /* ***BEGIN PROLOGUE  CBESK */
    /* ***DATE WRITTEN   830501   (YYMMDD) */
    /* ***REVISION DATE  890801, 930101   (YYMMDD) */
    /* ***CATEGORY NO.  B5K */
    /* ***KEYWORDS  K-BESSEL FUNCTION,COMPLEX BESSEL FUNCTION, */
    /*             MODIFIED BESSEL FUNCTION OF THE SECOND KIND, */
    /*             BESSEL FUNCTION OF THE THIRD KIND */
    /* ***AUTHOR  AMOS, DONALD E., SANDIA NATIONAL LABORATORIES */
    /* ***PURPOSE  TO COMPUTE K-BESSEL FUNCTIONS OF COMPLEX ARGUMENT */
    /* ***DESCRIPTION */

    /*         ON KODE=1, CBESK COMPUTES AN N MEMBER SEQUENCE OF COMPLEX */
    /*         BESSEL FUNCTIONS CY(J)=K(FNU+J-1,Z) FOR REAL, NONNEGATIVE */
    /*         ORDERS FNU+J-1, J=1,...,N AND COMPLEX Z.NE.CMPLX(0.0,0.0) */
    /*         IN THE CUT PLANE -PI.LT.ARG(Z).LE.PI. ON KODE=2, CBESK */
    /*         RETURNS THE SCALED K FUNCTIONS, */

    /*         CY(J)=EXP(Z)*K(FNU+J-1,Z) , J=1,...,N, */

    /*         WHICH REMOVE THE EXPONENTIAL BEHAVIOR IN BOTH THE LEFT AND */
    /*         RIGHT HALF PLANES FOR Z TO INFINITY. DEFINITIONS AND */
    /*         NOTATION ARE FOUND IN THE NBS HANDBOOK OF MATHEMATICAL */
    /*         FUNCTIONS (CONST. 1). */

    /*         INPUT */
    /*           Z      - Z=CMPLX(X,Y),Z.NE.CMPLX(0.,0.),-PI.LT.ARG(Z).LE.PI */
    /*           FNU    - ORDER OF INITIAL K FUNCTION, FNU.GE.0.0E0 */
    /*           N      - NUMBER OF MEMBERS OF THE SEQUENCE, N.GE.1 */
    /*           KODE   - A PARAMETER TO INDICATE THE SCALING OPTION */
    /*                    KODE= 1  RETURNS */
    /*                             CY(I)=K(FNU+I-1,Z), I=1,...,N */
    /*                        = 2  RETURNS */
    /*                             CY(I)=K(FNU+I-1,Z)*EXP(Z), I=1,...,N */

    /*         OUTPUT */
    /*           CY     - A COMPLEX VECTOR WHOSE FIRST N COMPONENTS CONTAIN */
    /*                    VALUES FOR THE SEQUENCE */
    /*                    CY(I)=K(FNU+I-1,Z), I=1,...,N OR */
    /*                    CY(I)=K(FNU+I-1,Z)*EXP(Z), I=1,...,N */
    /*                    DEPENDING ON KODE */
    /*           NZ     - NUMBER OF COMPONENTS SET TO ZERO DUE TO UNDERFLOW. */
    /*                    NZ= 0   , NORMAL RETURN */
    /*                    NZ.GT.0 , FIRST NZ COMPONENTS OF CY SET TO ZERO */
    /*                              DUE TO UNDERFLOW, CY(I)=CMPLX(0.0,0.0), */
    /*                              I=1,...,N WHEN X.GE.0.0. WHEN X.LT.0.0 */
    /*                              NZ STATES ONLY THE NUMBER OF UNDERFLOWS */
    /*                              IN THE SEQUENCE. */
    /*           IERR   - ERROR FLAG */
    /*                    IERR=0, NORMAL RETURN - COMPUTATION COMPLETED */
    /*                    IERR=1, INPUT ERROR   - NO COMPUTATION */
    /*                    IERR=2, OVERFLOW      - NO COMPUTATION, FNU+N-1 IS */
    /*                            TOO LARGE OR CABS(Z) IS TOO SMALL OR BOTH */
    /*                    IERR=3, CABS(Z) OR FNU+N-1 LARGE - COMPUTATION DONE */
    /*                            BUT LOSSES OF SIGNIFCANCE BY ARGUMENT */
    /*                            REDUCTION PRODUCE LESS THAN HALF OF MACHINE */
    /*                            ACCURACY */
    /*                    IERR=4, CABS(Z) OR FNU+N-1 TOO LARGE - NO COMPUTA- */
    /*                            TION BECAUSE OF COMPLETE LOSSES OF SIGNIFI- */
    /*                            CANCE BY ARGUMENT REDUCTION */
    /*                    IERR=5, ERROR              - NO COMPUTATION, */
    /*                            ALGORITHM TERMINATION CONDITION NOT MET */

    /* ***LONG DESCRIPTION */

    /*         EQUATIONS OF THE REFERENCE ARE IMPLEMENTED FOR SMALL ORDERS */
    /*         DNU AND DNU+1.0 IN THE RIGHT HALF PLANE X.GE.0.0. FORWARD */
    /*         RECURRENCE GENERATES HIGHER ORDERS. K IS CONTINUED TO THE LEFT */
    /*         HALF PLANE BY THE RELATION */

    /*         K(FNU,Z*EXP(MP)) = EXP(-MP*FNU)*K(FNU,Z)-MP*I(FNU,Z) */
    /*         MP=MR*PI*I, MR=+1 OR -1, RE(Z).GT.0, I**2=-1 */

    /*         WHERE I(FNU,Z) IS THE I BESSEL FUNCTION. */

    /*         FOR LARGE ORDERS, FNU.GT.FNUL, THE K FUNCTION IS COMPUTED */
    /*         BY MEANS OF ITS UNIFORM ASYMPTOTIC EXPANSIONS. */

    /*         FOR NEGATIVE ORDERS, THE FORMULA */

    /*                       K(-FNU,Z) = K(FNU,Z) */

    /*         CAN BE USED. */

    /*         CBESK ASSUMES THAT A SIGNIFICANT DIGIT SINH(X) FUNCTION IS */
    /*         AVAILABLE. */

    /*         IN MOST COMPLEX VARIABLE COMPUTATION, ONE MUST EVALUATE ELE- */
    /*         MENTARY FUNCTIONS. WHEN THE MAGNITUDE OF Z OR FNU+N-1 IS */
    /*         LARGE, LOSSES OF SIGNIFICANCE BY ARGUMENT REDUCTION OCCUR. */
    /*         CONSEQUENTLY, IF EITHER ONE EXCEEDS U1=SQRT(0.5/UR), THEN */
    /*         LOSSES EXCEEDING HALF PRECISION ARE LIKELY AND AN ERROR FLAG */
    /*         IERR=3 IS TRIGGERED WHERE UR=R1MACH(4)=UNIT ROUNDOFF. ALSO */
    /*         IF EITHER IS LARGER THAN U2=0.5/UR, THEN ALL SIGNIFICANCE IS */
    /*         LOST AND IERR=4. IN ORDER TO USE THE INT FUNCTION, ARGUMENTS */
    /*         MUST BE FURTHER RESTRICTED NOT TO EXCEED THE LARGEST MACHINE */
    /*         INTEGER, U3=I1MACH(9). THUS, THE MAGNITUDE OF Z AND FNU+N-1 IS */
    /*         RESTRICTED BY MIN(U2,U3). ON 32 BIT MACHINES, U1,U2, AND U3 */
    /*         ARE APPROXIMATELY 2.0E+3, 4.2E+6, 2.1E+9 IN SINGLE PRECISION */
    /*         ARITHMETIC AND 1.3E+8, 1.8E+16, 2.1E+9 IN DOUBLE PRECISION */
    /*         ARITHMETIC RESPECTIVELY. THIS MAKES U2 AND U3 LIMITING IN */
    /*         THEIR RESPECTIVE ARITHMETICS. THIS MEANS THAT ONE CAN EXPECT */
    /*         TO RETAIN, IN THE WORST CASES ON 32 BIT MACHINES, NO DIGITS */
    /*         IN SINGLE AND ONLY 7 DIGITS IN DOUBLE PRECISION ARITHMETIC. */
    /*         SIMILAR CONSIDERATIONS HOLD FOR OTHER MACHINES. */

    /*         THE APPROXIMATE RELATIVE ERROR IN THE MAGNITUDE OF A COMPLEX */
    /*         BESSEL FUNCTION CAN BE EXPRESSED BY P*10**S WHERE P=MAX(UNIT */
    /*         ROUNDOFF,1.0E-18) IS THE NOMINAL PRECISION AND 10**S REPRE- */
    /*         SENTS THE INCREASE IN ERROR DUE TO ARGUMENT REDUCTION IN THE */
    /*         ELEMENTARY FUNCTIONS. HERE, S=MAX(1,ABS(LOG10(CABS(Z))), */
    /*         ABS(LOG10(FNU))) APPROXIMATELY (I.E. S=MAX(1,ABS(EXPONENT OF */
    /*         CABS(Z),ABS(EXPONENT OF FNU)) ). HOWEVER, THE PHASE ANGLE MAY */
    /*         HAVE ONLY ABSOLUTE ACCURACY. THIS IS MOST LIKELY TO OCCUR WHEN */
    /*         ONE COMPONENT (IN ABSOLUTE VALUE) IS LARGER THAN THE OTHER BY */
    /*         SEVERAL ORDERS OF MAGNITUDE. IF ONE COMPONENT IS 10**K LARGER */
    /*         THAN THE OTHER, THEN ONE CAN EXPECT ONLY MAX(ABS(LOG10(P))-K, */
    /*         0) SIGNIFICANT DIGITS; OR, STATED ANOTHER WAY, WHEN K EXCEEDS */
    /*         THE EXPONENT OF P, NO SIGNIFICANT DIGITS REMAIN IN THE SMALLER */
    /*         COMPONENT. HOWEVER, THE PHASE ANGLE RETAINS ABSOLUTE ACCURACY */
    /*         BECAUSE, IN COMPLEX ARITHMETIC WITH PRECISION P, THE SMALLER */
    /*         COMPONENT WILL NOT (AS A RULE) DECREASE BELOW P TIMES THE */
    /*         MAGNITUDE OF THE LARGER COMPONENT. IN THESE EXTREME CASES, */
    /*         THE PRINCIPAL PHASE ANGLE IS ON THE ORDER OF +P, -P, PI/2-P, */
    /*         OR -PI/2+P. */

    /* ***REFERENCES  HANDBOOK OF MATHEMATICAL FUNCTIONS BY M. ABRAMOWITZ */
    /*                 AND I. A. STEGUN, NBS AMS SERIES 55, U.S. DEPT. OF */
    /*                 COMMERCE, 1955. */

    /*               COMPUTATION OF BESSEL FUNCTIONS OF COMPLEX ARGUMENT */
    /*                 BY D. E. AMOS, SAND83-0083, MAY, 1983. */

    /*               COMPUTATION OF BESSEL FUNCTIONS OF COMPLEX ARGUMENT */
    /*                 AND LARGE ORDER BY D. E. AMOS, SAND83-0643, MAY, 1983. */

    /*               A SUBROUTINE PACKAGE FOR BESSEL FUNCTIONS OF A COMPLEX */
    /*                 ARGUMENT AND NONNEGATIVE ORDER BY D. E. AMOS, SAND85- */
    /*                 1018, MAY, 1985 */

    /*               A PORTABLE PACKAGE FOR BESSEL FUNCTIONS OF A COMPLEX */
    /*                 ARGUMENT AND NONNEGATIVE ORDER BY D. E. AMOS, ACM */
    /*                 TRANS. MATH. SOFTWARE, VOL. 12, NO. 3, SEPTEMBER 1986, */
    /*                 PP 265-273. */

    /* ***ROUTINES CALLED  CACON,CBKNU,CBUNK,CUOIK,I1MACH,R1MACH */
    /* ***END PROLOGUE  CBESK */

    /* ***FIRST EXECUTABLE STATEMENT  CBESK */
    /* Parameter adjustments */
    --cy;

    /* Function Body */
    *ierr = 0;
    *nz   = 0;
    xx    = z__->real();
    yy    = r_imag(z__);
    if(yy == (float)0. && xx == (float)0.)
    {
        *ierr = 1;
    }
    if(*fnu < (float)0.)
    {
        *ierr = 1;
    }
    if(*kode < 1 || *kode > 2)
    {
        *ierr = 1;
    }
    if(*n < 1)
    {
        *ierr = 1;
    }
    if(*ierr != 0)
    {
        return 0;
    }
    nn = *n;
    /* ----------------------------------------------------------------------- */
    /*     SET PARAMETERS RELATED TO MACHINE CONSTANTS. */
    /*     TOL IS THE APPROXIMATE UNIT ROUNDOFF LIMITED TO 1.0E-18. */
    /*     ELIM IS THE APPROXIMATE EXPONENTIAL OVER- AND UNDERFLOW LIMIT. */
    /*     EXP(-ELIM).LT.EXP(-ALIM)=EXP(-ELIM)/TOL    AND */
    /*     EXP(ELIM).GT.EXP(ALIM)=EXP(ELIM)*TOL       ARE INTERVALS NEAR */
    /*     UNDERFLOW AND OVERFLOW LIMITS WHERE SCALED ARITHMETIC IS DONE. */
    /*     RL IS THE LOWER BOUNDARY OF THE ASYMPTOTIC EXPANSION FOR LARGE Z. */
    /*     DIG = NUMBER OF BASE 10 DIGITS IN TOL = 10**(-DIG). */
    /*     FNUL IS THE LOWER BOUNDARY OF THE ASYMPTOTIC SERIES FOR LARGE FNU */
    /* ----------------------------------------------------------------------- */
    /* Computing MAX */
    r__1 = r1mach_(&c__4);
    tol  = max(r__1, (float)1e-18);
    k1   = i1mach_(&c__12);
    k2   = i1mach_(&c__13);
    r1m5 = r1mach_(&c__5);
    /* Computing MIN */
    i__1 = abs(k1), i__2 = abs(k2);
    k    = std::min(i__1, i__2);
    elim = ((float)k * r1m5 - (float)3.) * (float)2.303;
    k1   = i1mach_(&c__11) - 1;
    aa   = r1m5 * (float)k1;
    dig  = std::min(aa, (float)18.);
    aa *= (float)2.303;
    /* Computing MAX */
    r__1 = -aa;
    alim = elim + max(r__1, (float)-41.45);
    fnul = (dig - (float)3.) * (float)6. + (float)10.;
    rl   = dig * (float)1.2 + (float)3.;
    az   = c_abs(z__);
    fn   = *fnu + (float)(nn - 1);
    /* ----------------------------------------------------------------------- */
    /*     TEST FOR RANGE */
    /* ----------------------------------------------------------------------- */
    aa = (float).5 / tol;
    bb = (float)i1mach_(&c__9) * (float).5;
    aa = std::min(aa, bb);
    if(az > aa)
    {
        goto L210;
    }
    if(fn > aa)
    {
        goto L210;
    }
    aa = sqrt(aa);
    if(az > aa)
    {
        *ierr = 3;
    }
    if(fn > aa)
    {
        *ierr = 3;
    }
    /* ----------------------------------------------------------------------- */
    /*     OVERFLOW TEST ON THE LAST MEMBER OF THE SEQUENCE */
    /* ----------------------------------------------------------------------- */
    /*     UFL = EXP(-ELIM) */
    ufl = r1mach_(&c__1) * (float)1e3;
    if(az < ufl)
    {
        goto L180;
    }
    if(*fnu > fnul)
    {
        goto L80;
    }
    if(fn <= (float)1.)
    {
        goto L60;
    }
    if(fn > (float)2.)
    {
        goto L50;
    }
    if(az > tol)
    {
        goto L60;
    }
    arg = az * (float).5;
    aln = -fn * log(arg);
    if(aln > elim)
    {
        goto L180;
    }
    goto L60;
L50:
    cuoik_(z__, fnu, kode, &c__2, &nn, &cy[1], &nuf, &tol, &elim, &alim);
    if(nuf < 0)
    {
        goto L180;
    }
    *nz += nuf;
    nn -= nuf;
    /* ----------------------------------------------------------------------- */
    /*     HERE NN=N OR NN=0 SINCE NUF=0,NN, OR -1 ON RETURN FROM CUOIK */
    /*     IF NUF=NN, THEN CY(I)=CZERO FOR ALL I */
    /* ----------------------------------------------------------------------- */
    if(nn == 0)
    {
        goto L100;
    }
L60:
    if(xx < (float)0.)
    {
        goto L70;
    }
    /* ----------------------------------------------------------------------- */
    /*     RIGHT HALF PLANE COMPUTATION, REAL(Z).GE.0. */
    /* ----------------------------------------------------------------------- */
    cbknu_(z__, fnu, kode, &nn, &cy[1], &nw, &tol, &elim, &alim);
    if(nw < 0)
    {
        goto L200;
    }
    *nz = nw;
    return 0;
/* ----------------------------------------------------------------------- */
/*     LEFT HALF PLANE COMPUTATION */
/*     PI/2.LT.ARG(Z).LE.PI AND -PI.LT.ARG(Z).LT.-PI/2. */
/* ----------------------------------------------------------------------- */
L70:
    if(*nz != 0)
    {
        goto L180;
    }
    mr = 1;
    if(yy < (float)0.)
    {
        mr = -1;
    }
    cacon_(z__, fnu, kode, &mr, &nn, &cy[1], &nw, &rl, &fnul, &tol, &elim, &alim);
    if(nw < 0)
    {
        goto L200;
    }
    *nz = nw;
    return 0;
/* ----------------------------------------------------------------------- */
/*     UNIFORM ASYMPTOTIC EXPANSIONS FOR FNU.GT.FNUL */
/* ----------------------------------------------------------------------- */
L80:
    mr = 0;
    if(xx >= (float)0.)
    {
        goto L90;
    }
    mr = 1;
    if(yy < (float)0.)
    {
        mr = -1;
    }
L90:
    cbunk_(z__, fnu, kode, &mr, &nn, &cy[1], &nw, &tol, &elim, &alim);
    if(nw < 0)
    {
        goto L200;
    }
    *nz += nw;
    return 0;
L100:
    if(xx < (float)0.)
    {
        goto L180;
    }
    return 0;
L180:
    *nz   = 0;
    *ierr = 2;
    return 0;
L200:
    if(nw == -1)
    {
        goto L180;
    }
    *nz   = 0;
    *ierr = 5;
    return 0;
L210:
    *nz   = 0;
    *ierr = 4;
    return 0;
} /* cbesk_ */


 int cbesy_(System::Complex<float>* z__, float* fnu, int32* kode, int32* n, System::Complex<float>* cy, int32* nz, System::Complex<float>* cwrk, int32* ierr)
{
    /* Initialized data */

    static System::Complex<float> cip[4] = {{(float)1., (float)0.}, {(float)0., (float)1.}, {(float)-1., (float)0.}, {(float)0., (float)-1.}};
    static float    hpi    = (float)1.57079632679489662;

    /* System generated locals */
    int32 i__1, i__2, i__3, i__4;
    float    r__1, r__2;
    System::Complex<float> q__1, q__2, q__3;


    
    int32                     i__, k, k1, k2, i4;
    float                        r1, r2, aa, bb;
    System::Complex<float>                     ci, ex;
    float                        ey;
    System::Complex<float>                     zn, zu, zv;
    float                        xx, yy;
    System::Complex<float>                     zz;
    int32                     nz1, nz2;
    float                        arg, r1m5, tay, tol, elim;
    System::Complex<float>                     csgn;
    float                        ffnu, atol, rhpi;
    System::Complex<float>                     cspn;
    int32                     ifnu;
    float                        rtol;
    float                        ascle;

    /* *********************************************************************72 */

    /* c CBESY computes a sequence of System::Complex<float> Bessel Y functions. */

    /* ***BEGIN PROLOGUE  CBESY */
    /* ***DATE WRITTEN   830501   (YYMMDD) */
    /* ***REVISION DATE  890801, 930101  (YYMMDD) */
    /* ***CATEGORY NO.  B5K */
    /* ***KEYWORDS  Y-BESSEL FUNCTION,BESSEL FUNCTION OF COMPLEX ARGUMENT, */
    /*             BESSEL FUNCTION OF SECOND KIND */
    /* ***AUTHOR  AMOS, DONALD E., SANDIA NATIONAL LABORATORIES */
    /* ***PURPOSE  TO COMPUTE THE Y-BESSEL FUNCTION OF A COMPLEX ARGUMENT */
    /* ***DESCRIPTION */

    /*         ON KODE=1, CBESY COMPUTES AN N MEMBER SEQUENCE OF COMPLEX */
    /*         BESSEL FUNCTIONS CY(I)=Y(FNU+I-1,Z) FOR REAL, NONNEGATIVE */
    /*         ORDERS FNU+I-1, I=1,...,N AND COMPLEX Z IN THE CUT PLANE */
    /*         -PI.LT.ARG(Z).LE.PI. ON KODE=2, CBESY RETURNS THE SCALED */
    /*         FUNCTIONS */

    /*         CY(I)=EXP(-ABS(Y))*Y(FNU+I-1,Z)   I = 1,...,N , Y=AIMAG(Z) */

    /*         WHICH REMOVE THE EXPONENTIAL GROWTH IN BOTH THE UPPER AND */
    /*         LOWER HALF PLANES FOR Z TO INFINITY. DEFINITIONS AND NOTATION */
    /*         ARE FOUND IN THE NBS HANDBOOK OF MATHEMATICAL FUNCTIONS */
    /*         (CONST. 1). */

    /*         INPUT */
    /*           Z      - Z=CMPLX(X,Y), Z.NE.CMPLX(0.,0.),-PI.LT.ARG(Z).LE.PI */
    /*           FNU    - ORDER OF INITIAL Y FUNCTION, FNU.GE.0.0E0 */
    /*           KODE   - A PARAMETER TO INDICATE THE SCALING OPTION */
    /*                    KODE= 1  RETURNS */
    /*                             CY(I)=Y(FNU+I-1,Z), I=1,...,N */
    /*                        = 2  RETURNS */
    /*                             CY(I)=Y(FNU+I-1,Z)*EXP(-ABS(Y)), I=1,...,N */
    /*                             WHERE Y=AIMAG(Z) */
    /*           N      - NUMBER OF MEMBERS OF THE SEQUENCE, N.GE.1 */
    /*           CWRK   - A COMPLEX WORK VECTOR OF DIMENSION AT LEAST N */

    /*         OUTPUT */
    /*           CY     - A COMPLEX VECTOR WHOSE FIRST N COMPONENTS CONTAIN */
    /*                    VALUES FOR THE SEQUENCE */
    /*                    CY(I)=Y(FNU+I-1,Z)  OR */
    /*                    CY(I)=Y(FNU+I-1,Z)*EXP(-ABS(Y))  I=1,...,N */
    /*                    DEPENDING ON KODE. */
    /*           NZ     - NZ=0 , A NORMAL RETURN */
    /*                    NZ.GT.0 , NZ COMPONENTS OF CY SET TO ZERO DUE TO */
    /*                    UNDERFLOW (GENERALLY ON KODE=2) */
    /*           IERR   - ERROR FLAG */
    /*                    IERR=0, NORMAL RETURN - COMPUTATION COMPLETED */
    /*                    IERR=1, INPUT ERROR   - NO COMPUTATION */
    /*                    IERR=2, OVERFLOW      - NO COMPUTATION, FNU+N-1 IS */
    /*                            TOO LARGE OR CABS(Z) IS TOO SMALL OR BOTH */
    /*                    IERR=3, CABS(Z) OR FNU+N-1 LARGE - COMPUTATION DONE */
    /*                            BUT LOSSES OF SIGNIFCANCE BY ARGUMENT */
    /*                            REDUCTION PRODUCE LESS THAN HALF OF MACHINE */
    /*                            ACCURACY */
    /*                    IERR=4, CABS(Z) OR FNU+N-1 TOO LARGE - NO COMPUTA- */
    /*                            TION BECAUSE OF COMPLETE LOSSES OF SIGNIFI- */
    /*                            CANCE BY ARGUMENT REDUCTION */
    /*                    IERR=5, ERROR              - NO COMPUTATION, */
    /*                            ALGORITHM TERMINATION CONDITION NOT MET */

    /* ***LONG DESCRIPTION */

    /*         THE COMPUTATION IS CARRIED OUT IN TERMS OF THE I(FNU,Z) AND */
    /*         K(FNU,Z) BESSEL FUNCTIONS IN THE RIGHT HALF PLANE BY */

    /*             Y(FNU,Z) = I*CC*I(FNU,ARG) - (2/PI)*CONJG(CC)*K(FNU,ARG) */

    /*             Y(FNU,Z) = CONJG(Y(FNU,CONJG(Z))) */

    /*         FOR AIMAG(Z).GE.0 AND AIMAG(Z).LT.0 RESPECTIVELY, WHERE */
    /*         CC=EXP(I*PI*FNU/2), ARG=Z*EXP(-I*PI/2) AND I**2=-1. */

    /*         FOR NEGATIVE ORDERS,THE FORMULA */

    /*             Y(-FNU,Z) = Y(FNU,Z)*COS(PI*FNU) + J(FNU,Z)*SIN(PI*FNU) */

    /*         CAN BE USED. HOWEVER,FOR LARGE ORDERS CLOSE TO HALF ODD */
    /*         INTEGERS THE FUNCTION CHANGES RADICALLY. WHEN FNU IS A LARGE */
    /*         POSITIVE HALF ODD INTEGER,THE MAGNITUDE OF Y(-FNU,Z)=J(FNU,Z)* */
    /*         SIN(PI*FNU) IS A LARGE NEGATIVE POWER OF TEN. BUT WHEN FNU IS */
    /*         NOT A HALF ODD INTEGER, Y(FNU,Z) DOMINATES IN MAGNITUDE WITH A */
    /*         LARGE POSITIVE POWER OF TEN AND THE MOST THAT THE SECOND TERM */
    /*         CAN BE REDUCED IS BY UNIT ROUNDOFF FROM THE COEFFICIENT. THUS, */
    /*         WIDE CHANGES CAN OCCUR WITHIN UNIT ROUNDOFF OF A LARGE HALF */
    /*         ODD INTEGER. HERE, LARGE MEANS FNU.GT.CABS(Z). */

    /*         IN MOST COMPLEX VARIABLE COMPUTATION, ONE MUST EVALUATE ELE- */
    /*         MENTARY FUNCTIONS. WHEN THE MAGNITUDE OF Z OR FNU+N-1 IS */
    /*         LARGE, LOSSES OF SIGNIFICANCE BY ARGUMENT REDUCTION OCCUR. */
    /*         CONSEQUENTLY, IF EITHER ONE EXCEEDS U1=SQRT(0.5/UR), THEN */
    /*         LOSSES EXCEEDING HALF PRECISION ARE LIKELY AND AN ERROR FLAG */
    /*         IERR=3 IS TRIGGERED WHERE UR=R1MACH(4)=UNIT ROUNDOFF. ALSO */
    /*         IF EITHER IS LARGER THAN U2=0.5/UR, THEN ALL SIGNIFICANCE IS */
    /*         LOST AND IERR=4. IN ORDER TO USE THE INT FUNCTION, ARGUMENTS */
    /*         MUST BE FURTHER RESTRICTED NOT TO EXCEED THE LARGEST MACHINE */
    /*         INTEGER, U3=I1MACH(9). THUS, THE MAGNITUDE OF Z AND FNU+N-1 IS */
    /*         RESTRICTED BY MIN(U2,U3). ON 32 BIT MACHINES, U1,U2, AND U3 */
    /*         ARE APPROXIMATELY 2.0E+3, 4.2E+6, 2.1E+9 IN SINGLE PRECISION */
    /*         ARITHMETIC AND 1.3E+8, 1.8E+16, 2.1E+9 IN DOUBLE PRECISION */
    /*         ARITHMETIC RESPECTIVELY. THIS MAKES U2 AND U3 LIMITING IN */
    /*         THEIR RESPECTIVE ARITHMETICS. THIS MEANS THAT ONE CAN EXPECT */
    /*         TO RETAIN, IN THE WORST CASES ON 32 BIT MACHINES, NO DIGITS */
    /*         IN SINGLE AND ONLY 7 DIGITS IN DOUBLE PRECISION ARITHMETIC. */
    /*         SIMILAR CONSIDERATIONS HOLD FOR OTHER MACHINES. */

    /*         THE APPROXIMATE RELATIVE ERROR IN THE MAGNITUDE OF A COMPLEX */
    /*         BESSEL FUNCTION CAN BE EXPRESSED BY P*10**S WHERE P=MAX(UNIT */
    /*         ROUNDOFF,1.0E-18) IS THE NOMINAL PRECISION AND 10**S REPRE- */
    /*         SENTS THE INCREASE IN ERROR DUE TO ARGUMENT REDUCTION IN THE */
    /*         ELEMENTARY FUNCTIONS. HERE, S=MAX(1,ABS(LOG10(CABS(Z))), */
    /*         ABS(LOG10(FNU))) APPROXIMATELY (I.E. S=MAX(1,ABS(EXPONENT OF */
    /*         CABS(Z),ABS(EXPONENT OF FNU)) ). HOWEVER, THE PHASE ANGLE MAY */
    /*         HAVE ONLY ABSOLUTE ACCURACY. THIS IS MOST LIKELY TO OCCUR WHEN */
    /*         ONE COMPONENT (IN ABSOLUTE VALUE) IS LARGER THAN THE OTHER BY */
    /*         SEVERAL ORDERS OF MAGNITUDE. IF ONE COMPONENT IS 10**K LARGER */
    /*         THAN THE OTHER, THEN ONE CAN EXPECT ONLY MAX(ABS(LOG10(P))-K, */
    /*         0) SIGNIFICANT DIGITS; OR, STATED ANOTHER WAY, WHEN K EXCEEDS */
    /*         THE EXPONENT OF P, NO SIGNIFICANT DIGITS REMAIN IN THE SMALLER */
    /*         COMPONENT. HOWEVER, THE PHASE ANGLE RETAINS ABSOLUTE ACCURACY */
    /*         BECAUSE, IN COMPLEX ARITHMETIC WITH PRECISION P, THE SMALLER */
    /*         COMPONENT WILL NOT (AS A RULE) DECREASE BELOW P TIMES THE */
    /*         MAGNITUDE OF THE LARGER COMPONENT. IN THESE EXTREME CASES, */
    /*         THE PRINCIPAL PHASE ANGLE IS ON THE ORDER OF +P, -P, PI/2-P, */
    /*         OR -PI/2+P. */

    /* ***REFERENCES  HANDBOOK OF MATHEMATICAL FUNCTIONS BY M. ABRAMOWITZ */
    /*                 AND I. A. STEGUN, NBS AMS SERIES 55, U.S. DEPT. OF */
    /*                 COMMERCE, 1955. */

    /*               COMPUTATION OF BESSEL FUNCTIONS OF COMPLEX ARGUMENT */
    /*                 BY D. E. AMOS, SAND83-0083, MAY, 1983. */

    /*               COMPUTATION OF BESSEL FUNCTIONS OF COMPLEX ARGUMENT */
    /*                 AND LARGE ORDER BY D. E. AMOS, SAND83-0643, MAY, 1983 */

    /*               A SUBROUTINE PACKAGE FOR BESSEL FUNCTIONS OF A COMPLEX */
    /*                 ARGUMENT AND NONNEGATIVE ORDER BY D. E. AMOS, SAND85- */
    /*                 1018, MAY, 1985 */

    /*               A PORTABLE PACKAGE FOR BESSEL FUNCTIONS OF A COMPLEX */
    /*                 ARGUMENT AND NONNEGATIVE ORDER BY D. E. AMOS, ACM */
    /*                 TRANS. MATH. SOFTWARE, VOL. 12, NO. 3, SEPTEMBER 1986, */
    /*                 PP 265-273. */

    /* ***ROUTINES CALLED  CBESI,CBESK,I1MACH,R1MACH */
    /* ***END PROLOGUE  CBESY */

    /* Parameter adjustments */
    --cwrk;
    --cy;

    /* Function Body */
    /* ***FIRST EXECUTABLE STATEMENT  CBESY */
    xx    = z__->real();
    yy    = r_imag(z__);
    *ierr = 0;
    *nz   = 0;
    if(xx == (float)0. && yy == (float)0.)
    {
        *ierr = 1;
    }
    if(*fnu < (float)0.)
    {
        *ierr = 1;
    }
    if(*kode < 1 || *kode > 2)
    {
        *ierr = 1;
    }
    if(*n < 1)
    {
        *ierr = 1;
    }
    if(*ierr != 0)
    {
        return 0;
    }
    ci.real() = (float)0., ci.imag() = (float)1.;
    zz.real() = z__->real(), zz.imag() = z__->imag();
    if(yy < (float)0.)
    {
        r_cnjg(&q__1, z__);
        zz.real() = q__1.real(), zz.imag() = q__1.imag();
    }
    q__2.real() = -ci.real(), q__2.imag() = -ci.imag();
    q__1.real() = q__2.real() * zz.real() - q__2.imag() * zz.imag(), q__1.imag() = q__2.real() * zz.imag() + q__2.imag() * zz.real();
    zn.real() = q__1.real(), zn.imag() = q__1.imag();
    cbesi_(&zn, fnu, kode, n, &cy[1], &nz1, ierr);
    if(*ierr != 0 && *ierr != 3)
    {
        goto L90;
    }
    cbesk_(&zn, fnu, kode, n, &cwrk[1], &nz2, ierr);
    if(*ierr != 0 && *ierr != 3)
    {
        goto L90;
    }
    *nz    = std::min(nz1, nz2);
    ifnu   = (int32)(*fnu);
    ffnu   = *fnu - (float)ifnu;
    arg    = hpi * ffnu;
    r__1   = cos(arg);
    r__2   = sin(arg);
    q__1.real() = r__1, q__1.imag() = r__2;
    csgn.real() = q__1.real(), csgn.imag() = q__1.imag();
    i4     = ifnu % 4 + 1;
    i__1   = i4 - 1;
    q__1.real() = csgn.real() * cip[i__1].real() - csgn.imag() * cip[i__1].imag(), q__1.imag() = csgn.real() * cip[i__1].imag() + csgn.imag() * cip[i__1].real();
    csgn.real() = q__1.real(), csgn.imag() = q__1.imag();
    rhpi = (float)1. / hpi;
    r_cnjg(&q__2, &csgn);
    q__3.real() = rhpi, q__3.imag() = (float)0.;
    q__1.real() = q__2.real() * q__3.real() - q__2.imag() * q__3.imag(), q__1.imag() = q__2.real() * q__3.imag() + q__2.imag() * q__3.real();
    cspn.real() = q__1.real(), cspn.imag() = q__1.imag();
    q__1.real() = csgn.real() * ci.real() - csgn.imag() * ci.imag(), q__1.imag() = csgn.real() * ci.imag() + csgn.imag() * ci.real();
    csgn.real() = q__1.real(), csgn.imag() = q__1.imag();
    if(*kode == 2)
    {
        goto L60;
    }
    i__1 = *n;
    for(i__ = 1; i__ <= i__1; ++i__)
    {
        i__2   = i__;
        i__3   = i__;
        q__2.real() = csgn.real() * cy[i__3].real() - csgn.imag() * cy[i__3].imag(), q__2.imag() = csgn.real() * cy[i__3].imag() + csgn.imag() * cy[i__3].real();
        i__4   = i__;
        q__3.real() = cspn.real() * cwrk[i__4].real() - cspn.imag() * cwrk[i__4].imag(), q__3.imag() = cspn.real() * cwrk[i__4].imag() + cspn.imag() * cwrk[i__4].real();
        q__1.real() = q__2.real() - q__3.real(), q__1.imag() = q__2.imag() - q__3.imag();
        cy[i__2].real() = q__1.real(), cy[i__2].imag() = q__1.imag();
        q__1.real() = ci.real() * csgn.real() - ci.imag() * csgn.imag(), q__1.imag() = ci.real() * csgn.imag() + ci.imag() * csgn.real();
        csgn.real() = q__1.real(), csgn.imag() = q__1.imag();
        q__2.real() = -ci.real(), q__2.imag() = -ci.imag();
        q__1.real() = q__2.real() * cspn.real() - q__2.imag() * cspn.imag(), q__1.imag() = q__2.real() * cspn.imag() + q__2.imag() * cspn.real();
        cspn.real() = q__1.real(), cspn.imag() = q__1.imag();
        /* L50: */
    }
    if(yy < (float)0.)
    {
        i__1 = *n;
        for(i__ = 1; i__ <= i__1; ++i__)
        {
            i__2 = i__;
            r_cnjg(&q__1, &cy[i__]);
            cy[i__2].real() = q__1.real(), cy[i__2].imag() = q__1.imag();
            /* L55: */
        }
    }
    return 0;
L60:
    r1     = cos(xx);
    r2     = sin(xx);
    q__1.real() = r1, q__1.imag() = r2;
    ex.real() = q__1.real(), ex.imag() = q__1.imag();
    /* Computing MAX */
    r__1 = r1mach_(&c__4);
    tol  = max(r__1, (float)1e-18);
    k1   = i1mach_(&c__12);
    k2   = i1mach_(&c__13);
    /* Computing MIN */
    i__1 = abs(k1), i__2 = abs(k2);
    k    = std::min(i__1, i__2);
    r1m5 = r1mach_(&c__5);
    /* ----------------------------------------------------------------------- */
    /*     ELIM IS THE APPROXIMATE EXPONENTIAL UNDER- AND OVERFLOW LIMIT */
    /* ----------------------------------------------------------------------- */
    elim = ((float)k * r1m5 - (float)3.) * (float)2.303;
    ey   = (float)0.;
    tay  = (r__1 = yy + yy, abs(r__1));
    if(tay < elim)
    {
        ey = exp(-tay);
    }
    q__3.real() = ey, q__3.imag() = (float)0.;
    q__2.real() = ex.real() * q__3.real() - ex.imag() * q__3.imag(), q__2.imag() = ex.real() * q__3.imag() + ex.imag() * q__3.real();
    q__1.real() = q__2.real() * cspn.real() - q__2.imag() * cspn.imag(), q__1.imag() = q__2.real() * cspn.imag() + q__2.imag() * cspn.real();
    cspn.real() = q__1.real(), cspn.imag() = q__1.imag();
    *nz   = 0;
    rtol  = (float)1. / tol;
    ascle = r1mach_(&c__1) * rtol * (float)1e3;
    i__1  = *n;
    for(i__ = 1; i__ <= i__1; ++i__)
    {
        /* ---------------------------------------------------------------------- */
        /*       CY(I) = CSGN*CY(I)-CSPN*CWRK(I): PRODUCTS ARE COMPUTED IN */
        /*       SCALED MODE IF CY(I) OR CWRK(I) ARE CLOSE TO UNDERFLOW TO */
        /*       PREVENT UNDERFLOW IN AN INTERMEDIATE COMPUTATION. */
        /* ---------------------------------------------------------------------- */
        i__2 = i__;
        zv.real() = cwrk[i__2].real(), zv.imag() = cwrk[i__2].imag();
        aa   = zv.real();
        bb   = r_imag(&zv);
        atol = (float)1.;
        /* Computing MAX */
        r__1 = abs(aa), r__2 = abs(bb);
        if(max(r__1, r__2) > ascle)
        {
            goto L75;
        }
        q__2.real() = rtol, q__2.imag() = (float)0.;
        q__1.real() = zv.real() * q__2.real() - zv.imag() * q__2.imag(), q__1.imag() = zv.real() * q__2.imag() + zv.imag() * q__2.real();
        zv.real() = q__1.real(), zv.imag() = q__1.imag();
        atol = tol;
    L75:
        q__1.real() = zv.real() * cspn.real() - zv.imag() * cspn.imag(), q__1.imag() = zv.real() * cspn.imag() + zv.imag() * cspn.real();
        zv.real() = q__1.real(), zv.imag() = q__1.imag();
        q__2.real() = atol, q__2.imag() = (float)0.;
        q__1.real() = zv.real() * q__2.real() - zv.imag() * q__2.imag(), q__1.imag() = zv.real() * q__2.imag() + zv.imag() * q__2.real();
        zv.real() = q__1.real(), zv.imag() = q__1.imag();
        i__2 = i__;
        zu.real() = cy[i__2].real(), zu.imag() = cy[i__2].imag();
        aa   = zu.real();
        bb   = r_imag(&zu);
        atol = (float)1.;
        /* Computing MAX */
        r__1 = abs(aa), r__2 = abs(bb);
        if(max(r__1, r__2) > ascle)
        {
            goto L85;
        }
        q__2.real() = rtol, q__2.imag() = (float)0.;
        q__1.real() = zu.real() * q__2.real() - zu.imag() * q__2.imag(), q__1.imag() = zu.real() * q__2.imag() + zu.imag() * q__2.real();
        zu.real() = q__1.real(), zu.imag() = q__1.imag();
        atol = tol;
    L85:
        q__1.real() = zu.real() * csgn.real() - zu.imag() * csgn.imag(), q__1.imag() = zu.real() * csgn.imag() + zu.imag() * csgn.real();
        zu.real() = q__1.real(), zu.imag() = q__1.imag();
        q__2.real() = atol, q__2.imag() = (float)0.;
        q__1.real() = zu.real() * q__2.real() - zu.imag() * q__2.imag(), q__1.imag() = zu.real() * q__2.imag() + zu.imag() * q__2.real();
        zu.real() = q__1.real(), zu.imag() = q__1.imag();
        i__2   = i__;
        q__1.real() = zu.real() - zv.real(), q__1.imag() = zu.imag() - zv.imag();
        cy[i__2].real() = q__1.real(), cy[i__2].imag() = q__1.imag();
        if(yy < (float)0.)
        {
            i__2 = i__;
            r_cnjg(&q__1, &cy[i__]);
            cy[i__2].real() = q__1.real(), cy[i__2].imag() = q__1.imag();
        }
        i__2 = i__;
        if(cy[i__2].real() == (float)0. && cy[i__2].imag() == (float)0. && ey == (float)0.)
        {
            ++(*nz);
        }
        q__1.real() = ci.real() * csgn.real() - ci.imag() * csgn.imag(), q__1.imag() = ci.real() * csgn.imag() + ci.imag() * csgn.real();
        csgn.real() = q__1.real(), csgn.imag() = q__1.imag();
        q__2.real() = -ci.real(), q__2.imag() = -ci.imag();
        q__1.real() = q__2.real() * cspn.real() - q__2.imag() * cspn.imag(), q__1.imag() = q__2.real() * cspn.imag() + q__2.imag() * cspn.real();
        cspn.real() = q__1.real(), cspn.imag() = q__1.imag();
        /* L80: */
    }
    return 0;
L90:
    *nz = 0;
    return 0;
} /* cbesy_ */

 int cbinu_(System::Complex<float>* z__, float* fnu, int32* kode, int32* n, System::Complex<float>* cy, int32* nz, float* rl, float* fnul, float* tol, float* elim, float* alim)
{
    /* Initialized data */

    static System::Complex<float> czero = {(float)0., (float)0.};

    /* System generated locals */
    int32 i__1, i__2;


    
    int32                     i__;
    System::Complex<float>                     cw[2];
    float                        az;
    int32                     nn, nw, nui, inw;
    float                        dfnu;
    int32                     nlast;

    /* *********************************************************************72 */

    /* ***BEGIN PROLOGUE  CBINU */
    /* ***REFER TO  CBESH,CBESI,CBESJ,CBESK,CAIRY,CBIRY */

    /* c CBINU COMPUTES THE I FUNCTION IN THE RIGHT HALF Z PLANE. */

    /* ***ROUTINES CALLED  CASYI,CBUNI,CMLRI,CSERI,CUOIK,CWRSK */
    /* ***END PROLOGUE  CBINU */
    /* Parameter adjustments */
    --cy;

    /* Function Body */

    *nz  = 0;
    az   = c_abs(z__);
    nn   = *n;
    dfnu = *fnu + (float)(*n - 1);
    if(az <= (float)2.)
    {
        goto L10;
    }
    if(az * az * (float).25 > dfnu + (float)1.)
    {
        goto L20;
    }
L10:
    /* ----------------------------------------------------------------------- */
    /*     POWER SERIES */
    /* ----------------------------------------------------------------------- */
    cseri_(z__, fnu, kode, &nn, &cy[1], &nw, tol, elim, alim);
    inw = abs(nw);
    *nz += inw;
    nn -= inw;
    if(nn == 0)
    {
        return 0;
    }
    if(nw >= 0)
    {
        goto L120;
    }
    dfnu = *fnu + (float)(nn - 1);
L20:
    if(az < *rl)
    {
        goto L40;
    }
    if(dfnu <= (float)1.)
    {
        goto L30;
    }
    if(az + az < dfnu * dfnu)
    {
        goto L50;
    }
/* ----------------------------------------------------------------------- */
/*     ASYMPTOTIC EXPANSION FOR LARGE Z */
/* ----------------------------------------------------------------------- */
L30:
    casyi_(z__, fnu, kode, &nn, &cy[1], &nw, rl, tol, elim, alim);
    if(nw < 0)
    {
        goto L130;
    }
    goto L120;
L40:
    if(dfnu <= (float)1.)
    {
        goto L70;
    }
L50:
    /* ----------------------------------------------------------------------- */
    /*     OVERFLOW AND UNDERFLOW TEST ON I SEQUENCE FOR MILLER ALGORITHM */
    /* ----------------------------------------------------------------------- */
    cuoik_(z__, fnu, kode, &c__1, &nn, &cy[1], &nw, tol, elim, alim);
    if(nw < 0)
    {
        goto L130;
    }
    *nz += nw;
    nn -= nw;
    if(nn == 0)
    {
        return 0;
    }
    dfnu = *fnu + (float)(nn - 1);
    if(dfnu > *fnul)
    {
        goto L110;
    }
    if(az > *fnul)
    {
        goto L110;
    }
L60:
    if(az > *rl)
    {
        goto L80;
    }
L70:
    /* ----------------------------------------------------------------------- */
    /*     MILLER ALGORITHM NORMALIZED BY THE SERIES */
    /* ----------------------------------------------------------------------- */
    cmlri_(z__, fnu, kode, &nn, &cy[1], &nw, tol);
    if(nw < 0)
    {
        goto L130;
    }
    goto L120;
L80:
    /* ----------------------------------------------------------------------- */
    /*     MILLER ALGORITHM NORMALIZED BY THE WRONSKIAN */
    /* ----------------------------------------------------------------------- */
    /* ----------------------------------------------------------------------- */
    /*     OVERFLOW TEST ON K FUNCTIONS USED IN WRONSKIAN */
    /* ----------------------------------------------------------------------- */
    cuoik_(z__, fnu, kode, &c__2, &c__2, cw, &nw, tol, elim, alim);
    if(nw >= 0)
    {
        goto L100;
    }
    *nz  = nn;
    i__1 = nn;
    for(i__ = 1; i__ <= i__1; ++i__)
    {
        i__2       = i__;
        cy[i__2].real() = czero.real(), cy[i__2].imag() = czero.imag();
        /* L90: */
    }
    return 0;
L100:
    if(nw > 0)
    {
        goto L130;
    }
    cwrsk_(z__, fnu, kode, &nn, &cy[1], &nw, cw, tol, elim, alim);
    if(nw < 0)
    {
        goto L130;
    }
    goto L120;
L110:
    /* ----------------------------------------------------------------------- */
    /*     INCREMENT FNU+NN-1 UP TO FNUL, COMPUTE AND RECUR BACKWARD */
    /* ----------------------------------------------------------------------- */
    nui = (int32)(*fnul - dfnu) + 1;
    nui = max(nui, 0);
    cbuni_(z__, fnu, kode, &nn, &cy[1], &nw, &nui, &nlast, fnul, tol, elim, alim);
    if(nw < 0)
    {
        goto L130;
    }
    *nz += nw;
    if(nlast == 0)
    {
        goto L120;
    }
    nn = nlast;
    goto L60;
L120:
    return 0;
L130:
    *nz = -1;
    if(nw == -2)
    {
        *nz = -2;
    }
    return 0;
} /* cbinu_ */

 int cbiry_(System::Complex<float>* z__, int32* id, int32* kode, System::Complex<float>* bi, int32* ierr)
{
    /* Initialized data */

    static float    tth  = (float).666666666666666667;
    static float    c1   = (float).614926627446000736;
    static float    c2   = (float).448288357353826359;
    static float    coef = (float).577350269189625765;
    static float    pi   = (float)3.14159265358979324;
    static System::Complex<float> cone = {(float)1., (float)0.};

    /* System generated locals */
    int32    i__1, i__2;
    float       r__1, r__2;
    double d__1, d__2;
    System::Complex<float>    q__1, q__2, q__3, q__4, q__5, q__6;



    
    int32                     k;
    float                        d1, d2;
    int32                     k1, k2;
    System::Complex<float>                     s1, s2, z3;
    float                        aa, bb, ad, ak, bk, ck, dk, az;
    System::Complex<float>                     cy[2];
    float                        rl, zi;
    int32                     nz;
    float                        zr, az3, z3i, z3r, fid, dig, fmr, r1m5;
    System::Complex<float>                     csq;
    float                        fnu;
    System::Complex<float>                     zta;
    float                        tol;
    System::Complex<float>                     trm1, trm2;
    float                        sfac, alim, elim, atrm, fnul;


    /* *********************************************************************72 */

    /* ***BEGIN PROLOGUE  CBIRY */
    /* ***DATE WRITTEN   830501   (YYMMDD) */
    /* ***REVISION DATE  890801, 930101   (YYMMDD) */
    /* ***CATEGORY NO.  B5K */
    /* ***KEYWORDS  AIRY FUNCTION,BESSEL FUNCTIONS OF ORDER ONE THIRD */
    /* ***AUTHOR  AMOS, DONALD E., SANDIA NATIONAL LABORATORIES */
    /* ***PURPOSE  TO COMPUTE AIRY FUNCTIONS BI(Z) AND DBI(Z) FOR COMPLEX Z */
    /* ***DESCRIPTION */

    /*         ON KODE=1, CBIRY COMPUTES THE COMPLEX AIRY FUNCTION BI(Z) OR */
    /*         ITS DERIVATIVE DBI(Z)/DZ ON ID=0 OR ID=1 RESPECTIVELY. ON */
    /*         KODE=2, A SCALING OPTION CEXP(-AXZTA)*BI(Z) OR CEXP(-AXZTA)* */
    /*         DBI(Z)/DZ IS PROVIDED TO REMOVE THE EXPONENTIAL BEHAVIOR IN */
    /*         BOTH THE LEFT AND RIGHT HALF PLANES WHERE */
    /*         ZTA=(2/3)*Z*CSQRT(Z)=CMPLX(XZTA,YZTA) AND AXZTA=ABS(XZTA). */
    /*         DEFINITIONS AND NOTATION ARE FOUND IN THE NBS HANDBOOK OF */
    /*         MATHEMATICAL FUNCTIONS (CONST. 1). */

    /*         INPUT */
    /*           Z      - Z=CMPLX(X,Y) */
    /*           ID     - ORDER OF DERIVATIVE, ID=0 OR ID=1 */
    /*           KODE   - A PARAMETER TO INDICATE THE SCALING OPTION */
    /*                    KODE= 1  RETURNS */
    /*                             BI=BI(Z)                 ON ID=0 OR */
    /*                             BI=DBI(Z)/DZ             ON ID=1 */
    /*                        = 2  RETURNS */
    /*                             BI=CEXP(-AXZTA)*BI(Z)     ON ID=0 OR */
    /*                             BI=CEXP(-AXZTA)*DBI(Z)/DZ ON ID=1 WHERE */
    /*                             ZTA=(2/3)*Z*CSQRT(Z)=CMPLX(XZTA,YZTA) */
    /*                             AND AXZTA=ABS(XZTA) */

    /*         OUTPUT */
    /*           BI     - COMPLEX ANSWER DEPENDING ON THE CHOICES FOR ID AND */
    /*                    KODE */
    /*           IERR   - ERROR FLAG */
    /*                    IERR=0, NORMAL RETURN - COMPUTATION COMPLETED */
    /*                    IERR=1, INPUT ERROR   - NO COMPUTATION */
    /*                    IERR=2, OVERFLOW      - NO COMPUTATION, REAL(Z) */
    /*                            TOO LARGE WITH KODE=1 */
    /*                    IERR=3, CABS(Z) LARGE      - COMPUTATION COMPLETED */
    /*                            LOSSES OF SIGNIFCANCE BY ARGUMENT REDUCTION */
    /*                            PRODUCE LESS THAN HALF OF MACHINE ACCURACY */
    /*                    IERR=4, CABS(Z) TOO LARGE  - NO COMPUTATION */
    /*                            COMPLETE LOSS OF ACCURACY BY ARGUMENT */
    /*                            REDUCTION */
    /*                    IERR=5, ERROR              - NO COMPUTATION, */
    /*                            ALGORITHM TERMINATION CONDITION NOT MET */

    /* ***LONG DESCRIPTION */

    /*         BI AND DBI ARE COMPUTED FOR CABS(Z).GT.1.0 FROM THE I BESSEL */
    /*         FUNCTIONS BY */

    /*                BI(Z)=C*SQRT(Z)*( I(-1/3,ZTA) + I(1/3,ZTA) ) */
    /*               DBI(Z)=C *  Z  * ( I(-2/3,ZTA) + I(2/3,ZTA) ) */
    /*                               C=1.0/SQRT(3.0) */
    /*                               ZTA=(2/3)*Z**(3/2) */

    /*         WITH THE POWER SERIES FOR CABS(Z).LE.1.0. */

    /*         IN MOST COMPLEX VARIABLE COMPUTATION, ONE MUST EVALUATE ELE- */
    /*         MENTARY FUNCTIONS. WHEN THE MAGNITUDE OF Z IS LARGE, LOSSES */
    /*         OF SIGNIFICANCE BY ARGUMENT REDUCTION OCCUR. CONSEQUENTLY, IF */
    /*         THE MAGNITUDE OF ZETA=(2/3)*Z**1.5 EXCEEDS U1=SQRT(0.5/UR), */
    /*         THEN LOSSES EXCEEDING HALF PRECISION ARE LIKELY AND AN ERROR */
    /*         FLAG IERR=3 IS TRIGGERED WHERE UR=R1MACH(4)=UNIT ROUNDOFF. */
    /*         ALSO, IF THE MAGNITUDE OF ZETA IS LARGER THAN U2=0.5/UR, THEN */
    /*         ALL SIGNIFICANCE IS LOST AND IERR=4. IN ORDER TO USE THE INT */
    /*         FUNCTION, ZETA MUST BE FURTHER RESTRICTED NOT TO EXCEED THE */
    /*         LARGEST INTEGER, U3=I1MACH(9). THUS, THE MAGNITUDE OF ZETA */
    /*         MUST BE RESTRICTED BY MIN(U2,U3). ON 32 BIT MACHINES, U1,U2, */
    /*         AND U3 ARE APPROXIMATELY 2.0E+3, 4.2E+6, 2.1E+9 IN SINGLE */
    /*         PRECISION ARITHMETIC AND 1.3E+8, 1.8E+16, 2.1E+9 IN DOUBLE */
    /*         PRECISION ARITHMETIC RESPECTIVELY. THIS MAKES U2 AND U3 LIMIT- */
    /*         ING IN THEIR RESPECTIVE ARITHMETICS. THIS MEANS THAT THE MAG- */
    /*         NITUDE OF Z CANNOT EXCEED 3.1E+4 IN SINGLE AND 2.1E+6 IN */
    /*         DOUBLE PRECISION ARITHMETIC. THIS ALSO MEANS THAT ONE CAN */
    /*         EXPECT TO RETAIN, IN THE WORST CASES ON 32 BIT MACHINES, */
    /*         NO DIGITS IN SINGLE PRECISION AND ONLY 7 DIGITS IN DOUBLE */
    /*         PRECISION ARITHMETIC. */

    /*         THE APPROXIMATE RELATIVE ERROR IN THE MAGNITUDE OF A COMPLEX */
    /*         BESSEL FUNCTION CAN BE EXPRESSED BY P*10**S WHERE P=MAX(UNIT */
    /*         ROUNDOFF,1.0E-18) IS THE NOMINAL PRECISION AND 10**S REPRE- */
    /*         SENTS THE INCREASE IN ERROR DUE TO ARGUMENT REDUCTION IN THE */
    /*         ELEMENTARY FUNCTIONS. HERE, S=MAX(1,ABS(LOG10(CABS(Z))), */
    /*         ABS(LOG10(FNU))) APPROXIMATELY (I.E. S=MAX(1,ABS(EXPONENT OF */
    /*         CABS(Z),ABS(EXPONENT OF FNU)) ). HOWEVER, THE PHASE ANGLE MAY */
    /*         HAVE ONLY ABSOLUTE ACCURACY. THIS IS MOST LIKELY TO OCCUR WHEN */
    /*         ONE COMPONENT (IN ABSOLUTE VALUE) IS LARGER THAN THE OTHER BY */
    /*         SEVERAL ORDERS OF MAGNITUDE. IF ONE COMPONENT IS 10**K LARGER */
    /*         THAN THE OTHER, THEN ONE CAN EXPECT ONLY MAX(ABS(LOG10(P))-K, */
    /*         0) SIGNIFICANT DIGITS; OR, STATED ANOTHER WAY, WHEN K EXCEEDS */
    /*         THE EXPONENT OF P, NO SIGNIFICANT DIGITS REMAIN IN THE SMALLER */
    /*         COMPONENT. HOWEVER, THE PHASE ANGLE RETAINS ABSOLUTE ACCURACY */
    /*         BECAUSE, IN COMPLEX ARITHMETIC WITH PRECISION P, THE SMALLER */
    /*         COMPONENT WILL NOT (AS A RULE) DECREASE BELOW P TIMES THE */
    /*         MAGNITUDE OF THE LARGER COMPONENT. IN THESE EXTREME CASES, */
    /*         THE PRINCIPAL PHASE ANGLE IS ON THE ORDER OF +P, -P, PI/2-P, */
    /*         OR -PI/2+P. */

    /* ***REFERENCES  HANDBOOK OF MATHEMATICAL FUNCTIONS BY M. ABRAMOWITZ */
    /*                 AND I. A. STEGUN, NBS AMS SERIES 55, U.S. DEPT. OF */
    /*                 COMMERCE, 1955. */

    /*               COMPUTATION OF BESSEL FUNCTIONS OF COMPLEX ARGUMENT */
    /*                 AND LARGE ORDER BY D. E. AMOS, SAND83-0643, MAY, 1983 */

    /*               A SUBROUTINE PACKAGE FOR BESSEL FUNCTIONS OF A COMPLEX */
    /*                 ARGUMENT AND NONNEGATIVE ORDER BY D. E. AMOS, SAND85- */
    /*                 1018, MAY, 1985 */

    /*               A PORTABLE PACKAGE FOR BESSEL FUNCTIONS OF A COMPLEX */
    /*                 ARGUMENT AND NONNEGATIVE ORDER BY D. E. AMOS, ACM */
    /*                 TRANS. MATH. SOFTWARE, VOL. 12, NO. 3, SEPTEMBER 1986, */
    /*                 PP 265-273. */

    /* ***ROUTINES CALLED  CBINU,I1MACH,R1MACH */
    /* ***END PROLOGUE  CBIRY */
    /* ***FIRST EXECUTABLE STATEMENT  CBIRY */
    *ierr = 0;
    nz    = 0;
    if(*id < 0 || *id > 1)
    {
        *ierr = 1;
    }
    if(*kode < 1 || *kode > 2)
    {
        *ierr = 1;
    }
    if(*ierr != 0)
    {
        return 0;
    }
    az = c_abs(z__);
    /* Computing MAX */
    r__1 = r1mach_(&c__4);
    tol  = max(r__1, (float)1e-18);
    fid  = (float)(*id);
    if(az > (float)1.)
    {
        goto L60;
    }
    /* ----------------------------------------------------------------------- */
    /*     POWER SERIES FOR CABS(Z).LE.1. */
    /* ----------------------------------------------------------------------- */
    s1.real() = cone.real(), s1.imag() = cone.imag();
    s2.real() = cone.real(), s2.imag() = cone.imag();
    if(az < tol)
    {
        goto L110;
    }
    aa = az * az;
    if(aa < tol / az)
    {
        goto L40;
    }
    trm1.real() = cone.real(), trm1.imag() = cone.imag();
    trm2.real() = cone.real(), trm2.imag() = cone.imag();
    atrm   = (float)1.;
    q__2.real() = z__->real() * z__->real() - z__->imag() * z__->imag(), q__2.imag() = z__->real() * z__->imag() + z__->imag() * z__->real();
    q__1.real() = q__2.real() * z__->real() - q__2.imag() * z__->imag(), q__1.imag() = q__2.real() * z__->imag() + q__2.imag() * z__->real();
    z3.real() = q__1.real(), z3.imag() = q__1.imag();
    az3 = az * aa;
    ak  = fid + (float)2.;
    bk  = (float)3. - fid - fid;
    ck  = (float)4. - fid;
    dk  = fid + (float)3. + fid;
    d1  = ak * dk;
    d2  = bk * ck;
    ad  = std::min(d1, d2);
    ak  = fid * (float)9. + (float)24.;
    bk  = (float)30. - fid * (float)9.;
    z3r = z3.real();
    z3i = r_imag(&z3);
    for(k = 1; k <= 25; ++k)
    {
        r__1   = z3r / d1;
        r__2   = z3i / d1;
        q__2.real() = r__1, q__2.imag() = r__2;
        q__1.real() = trm1.real() * q__2.real() - trm1.imag() * q__2.imag(), q__1.imag() = trm1.real() * q__2.imag() + trm1.imag() * q__2.real();
        trm1.real() = q__1.real(), trm1.imag() = q__1.imag();
        q__1.real() = s1.real() + trm1.real(), q__1.imag() = s1.imag() + trm1.imag();
        s1.real() = q__1.real(), s1.imag() = q__1.imag();
        r__1   = z3r / d2;
        r__2   = z3i / d2;
        q__2.real() = r__1, q__2.imag() = r__2;
        q__1.real() = trm2.real() * q__2.real() - trm2.imag() * q__2.imag(), q__1.imag() = trm2.real() * q__2.imag() + trm2.imag() * q__2.real();
        trm2.real() = q__1.real(), trm2.imag() = q__1.imag();
        q__1.real() = s2.real() + trm2.real(), q__1.imag() = s2.imag() + trm2.imag();
        s2.real() = q__1.real(), s2.imag() = q__1.imag();
        atrm = atrm * az3 / ad;
        d1 += ak;
        d2 += bk;
        ad = std::min(d1, d2);
        if(atrm < tol * ad)
        {
            goto L40;
        }
        ak += (float)18.;
        bk += (float)18.;
        /* L30: */
    }
L40:
    if(*id == 1)
    {
        goto L50;
    }
    q__3.real() = c1, q__3.imag() = (float)0.;
    q__2.real() = s1.real() * q__3.real() - s1.imag() * q__3.imag(), q__2.imag() = s1.real() * q__3.imag() + s1.imag() * q__3.real();
    q__5.real() = z__->real() * s2.real() - z__->imag() * s2.imag(), q__5.imag() = z__->real() * s2.imag() + z__->imag() * s2.real();
    q__6.real() = c2, q__6.imag() = (float)0.;
    q__4.real() = q__5.real() * q__6.real() - q__5.imag() * q__6.imag(), q__4.imag() = q__5.real() * q__6.imag() + q__5.imag() * q__6.real();
    q__1.real() = q__2.real() + q__4.real(), q__1.imag() = q__2.imag() + q__4.imag();
    bi->real() = q__1.real(), bi->imag() = q__1.imag();
    if(*kode == 1)
    {
        return 0;
    }
    c_sqrt(&q__3, z__);
    q__2.real() = z__->real() * q__3.real() - z__->imag() * q__3.imag(), q__2.imag() = z__->real() * q__3.imag() + z__->imag() * q__3.real();
    q__4.real() = tth, q__4.imag() = (float)0.;
    q__1.real() = q__2.real() * q__4.real() - q__2.imag() * q__4.imag(), q__1.imag() = q__2.real() * q__4.imag() + q__2.imag() * q__4.real();
    zta.real() = q__1.real(), zta.imag() = q__1.imag();
    aa     = zta.real();
    aa     = -abs(aa);
    r__1   = exp(aa);
    q__2.real() = r__1, q__2.imag() = (float)0.;
    q__1.real() = bi->real() * q__2.real() - bi->imag() * q__2.imag(), q__1.imag() = bi->real() * q__2.imag() + bi->imag() * q__2.real();
    bi->real() = q__1.real(), bi->imag() = q__1.imag();
    return 0;
L50:
    q__2.real() = c2, q__2.imag() = (float)0.;
    q__1.real() = s2.real() * q__2.real() - s2.imag() * q__2.imag(), q__1.imag() = s2.real() * q__2.imag() + s2.imag() * q__2.real();
    bi->real() = q__1.real(), bi->imag() = q__1.imag();
    if(az > tol)
    {
        q__4.real() = z__->real() * z__->real() - z__->imag() * z__->imag(), q__4.imag() = z__->real() * z__->imag() + z__->imag() * z__->real();
        q__3.real() = q__4.real() * s1.real() - q__4.imag() * s1.imag(), q__3.imag() = q__4.real() * s1.imag() + q__4.imag() * s1.real();
        r__1   = c1 / (fid + (float)1.);
        q__5.real() = r__1, q__5.imag() = (float)0.;
        q__2.real() = q__3.real() * q__5.real() - q__3.imag() * q__5.imag(), q__2.imag() = q__3.real() * q__5.imag() + q__3.imag() * q__5.real();
        q__1.real() = bi->real() + q__2.real(), q__1.imag() = bi->imag() + q__2.imag();
        bi->real() = q__1.real(), bi->imag() = q__1.imag();
    }
    if(*kode == 1)
    {
        return 0;
    }
    c_sqrt(&q__3, z__);
    q__2.real() = z__->real() * q__3.real() - z__->imag() * q__3.imag(), q__2.imag() = z__->real() * q__3.imag() + z__->imag() * q__3.real();
    q__4.real() = tth, q__4.imag() = (float)0.;
    q__1.real() = q__2.real() * q__4.real() - q__2.imag() * q__4.imag(), q__1.imag() = q__2.real() * q__4.imag() + q__2.imag() * q__4.real();
    zta.real() = q__1.real(), zta.imag() = q__1.imag();
    aa     = zta.real();
    aa     = -abs(aa);
    r__1   = exp(aa);
    q__2.real() = r__1, q__2.imag() = (float)0.;
    q__1.real() = bi->real() * q__2.real() - bi->imag() * q__2.imag(), q__1.imag() = bi->real() * q__2.imag() + bi->imag() * q__2.real();
    bi->real() = q__1.real(), bi->imag() = q__1.imag();
    return 0;
/* ----------------------------------------------------------------------- */
/*     CASE FOR CABS(Z).GT.1.0 */
/* ----------------------------------------------------------------------- */
L60:
    fnu = (fid + (float)1.) / (float)3.;
    /* ----------------------------------------------------------------------- */
    /*     SET PARAMETERS RELATED TO MACHINE CONSTANTS. */
    /*     TOL IS THE APPROXIMATE UNIT ROUNDOFF LIMITED TO 1.0E-18. */
    /*     ELIM IS THE APPROXIMATE EXPONENTIAL OVER- AND UNDERFLOW LIMIT. */
    /*     EXP(-ELIM).LT.EXP(-ALIM)=EXP(-ELIM)/TOL    AND */
    /*     EXP(ELIM).GT.EXP(ALIM)=EXP(ELIM)*TOL       ARE INTERVALS NEAR */
    /*     UNDERFLOW AND OVERFLOW LIMITS WHERE SCALED ARITHMETIC IS DONE. */
    /*     RL IS THE LOWER BOUNDARY OF THE ASYMPTOTIC EXPANSION FOR LARGE Z. */
    /*     DIG = NUMBER OF BASE 10 DIGITS IN TOL = 10**(-DIG). */
    /*     FNUL IS THE LOWER BOUNDARY OF THE ASYMPTOTIC SERIES FOR LARGE FNU. */
    /* ----------------------------------------------------------------------- */
    k1   = i1mach_(&c__12);
    k2   = i1mach_(&c__13);
    r1m5 = r1mach_(&c__5);
    /* Computing MIN */
    i__1 = abs(k1), i__2 = abs(k2);
    k    = std::min(i__1, i__2);
    elim = ((float)k * r1m5 - (float)3.) * (float)2.303;
    k1   = i1mach_(&c__11) - 1;
    aa   = r1m5 * (float)k1;
    dig  = std::min(aa, (float)18.);
    aa *= (float)2.303;
    /* Computing MAX */
    r__1 = -aa;
    alim = elim + max(r__1, (float)-41.45);
    rl   = dig * (float)1.2 + (float)3.;
    fnul = (dig - (float)3.) * (float)6. + (float)10.;
    /* ----------------------------------------------------------------------- */
    /*     TEST FOR RANGE */
    /* ----------------------------------------------------------------------- */
    aa   = (float).5 / tol;
    bb   = (float)i1mach_(&c__9) * (float).5;
    aa   = std::min(aa, bb);
    d__1 = (double)aa;
    d__2 = (double)tth;
    aa   = pow_dd(&d__1, &d__2);
    if(az > aa)
    {
        goto L190;
    }
    aa = sqrt(aa);
    if(az > aa)
    {
        *ierr = 3;
    }
    c_sqrt(&q__1, z__);
    csq.real() = q__1.real(), csq.imag() = q__1.imag();
    q__2.real() = z__->real() * csq.real() - z__->imag() * csq.imag(), q__2.imag() = z__->real() * csq.imag() + z__->imag() * csq.real();
    q__3.real() = tth, q__3.imag() = (float)0.;
    q__1.real() = q__2.real() * q__3.real() - q__2.imag() * q__3.imag(), q__1.imag() = q__2.real() * q__3.imag() + q__2.imag() * q__3.real();
    zta.real() = q__1.real(), zta.imag() = q__1.imag();
    /* ----------------------------------------------------------------------- */
    /*     RE(ZTA).LE.0 WHEN RE(Z).LT.0, ESPECIALLY WHEN IM(Z) IS SMALL */
    /* ----------------------------------------------------------------------- */
    sfac = (float)1.;
    zi   = r_imag(z__);
    zr   = z__->real();
    ak   = r_imag(&zta);
    if(zr >= (float)0.)
    {
        goto L70;
    }
    bk     = zta.real();
    ck     = -abs(bk);
    q__1.real() = ck, q__1.imag() = ak;
    zta.real() = q__1.real(), zta.imag() = q__1.imag();
L70:
    if(zi == (float)0. && zr <= (float)0.)
    {
        q__1.real() = (float)0., q__1.imag() = ak;
        zta.real() = q__1.real(), zta.imag() = q__1.imag();
    }
    aa = zta.real();
    if(*kode == 2)
    {
        goto L80;
    }
    /* ----------------------------------------------------------------------- */
    /*     OVERFLOW TEST */
    /* ----------------------------------------------------------------------- */
    bb = abs(aa);
    if(bb < alim)
    {
        goto L80;
    }
    bb += log(az) * (float).25;
    sfac = tol;
    if(bb > elim)
    {
        goto L170;
    }
L80:
    fmr = (float)0.;
    if(aa >= (float)0. && zr > (float)0.)
    {
        goto L90;
    }
    fmr = pi;
    if(zi < (float)0.)
    {
        fmr = -pi;
    }
    q__1.real() = -zta.real(), q__1.imag() = -zta.imag();
    zta.real() = q__1.real(), zta.imag() = q__1.imag();
L90:
    /* ----------------------------------------------------------------------- */
    /*     AA=FACTOR FOR ANALYTIC CONTINUATION OF I(FNU,ZTA) */
    /*     KODE=2 RETURNS EXP(-ABS(XZTA))*I(FNU,ZTA) FROM CBINU */
    /* ----------------------------------------------------------------------- */
    cbinu_(&zta, &fnu, kode, &c__1, cy, &nz, &rl, &fnul, &tol, &elim, &alim);
    if(nz < 0)
    {
        goto L180;
    }
    aa     = fmr * fnu;
    q__1.real() = sfac, q__1.imag() = (float)0.;
    z3.real() = q__1.real(), z3.imag() = q__1.imag();
    r__1   = cos(aa);
    r__2   = sin(aa);
    q__3.real() = r__1, q__3.imag() = r__2;
    q__2.real() = cy[0].real() * q__3.real() - cy[0].imag() * q__3.imag(), q__2.imag() = cy[0].real() * q__3.imag() + cy[0].imag() * q__3.real();
    q__1.real() = q__2.real() * z3.real() - q__2.imag() * z3.imag(), q__1.imag() = q__2.real() * z3.imag() + q__2.imag() * z3.real();
    s1.real() = q__1.real(), s1.imag() = q__1.imag();
    fnu = ((float)2. - fid) / (float)3.;
    cbinu_(&zta, &fnu, kode, &c__2, cy, &nz, &rl, &fnul, &tol, &elim, &alim);
    q__1.real() = cy[0].real() * z3.real() - cy[0].imag() * z3.imag(), q__1.imag() = cy[0].real() * z3.imag() + cy[0].imag() * z3.real();
    cy[0].real() = q__1.real(), cy[0].imag() = q__1.imag();
    q__1.real() = cy[1].real() * z3.real() - cy[1].imag() * z3.imag(), q__1.imag() = cy[1].real() * z3.imag() + cy[1].imag() * z3.real();
    cy[1].real() = q__1.real(), cy[1].imag() = q__1.imag();
    /* ----------------------------------------------------------------------- */
    /*     BACKWARD RECUR ONE STEP FOR ORDERS -1/3 OR -2/3 */
    /* ----------------------------------------------------------------------- */
    r__1   = fnu + fnu;
    q__4.real() = r__1, q__4.imag() = (float)0.;
    q__3.real() = cy[0].real() * q__4.real() - cy[0].imag() * q__4.imag(), q__3.imag() = cy[0].real() * q__4.imag() + cy[0].imag() * q__4.real();
    c_div(&q__2, &q__3, &zta);
    q__1.real() = q__2.real() + cy[1].real(), q__1.imag() = q__2.imag() + cy[1].imag();
    s2.real() = q__1.real(), s2.imag() = q__1.imag();
    aa     = fmr * (fnu - (float)1.);
    r__1   = cos(aa);
    r__2   = sin(aa);
    q__4.real() = r__1, q__4.imag() = r__2;
    q__3.real() = s2.real() * q__4.real() - s2.imag() * q__4.imag(), q__3.imag() = s2.real() * q__4.imag() + s2.imag() * q__4.real();
    q__2.real() = s1.real() + q__3.real(), q__2.imag() = s1.imag() + q__3.imag();
    q__5.real() = coef, q__5.imag() = (float)0.;
    q__1.real() = q__2.real() * q__5.real() - q__2.imag() * q__5.imag(), q__1.imag() = q__2.real() * q__5.imag() + q__2.imag() * q__5.real();
    s1.real() = q__1.real(), s1.imag() = q__1.imag();
    if(*id == 1)
    {
        goto L100;
    }
    q__1.real() = csq.real() * s1.real() - csq.imag() * s1.imag(), q__1.imag() = csq.real() * s1.imag() + csq.imag() * s1.real();
    s1.real() = q__1.real(), s1.imag() = q__1.imag();
    r__1   = (float)1. / sfac;
    q__2.real() = r__1, q__2.imag() = (float)0.;
    q__1.real() = s1.real() * q__2.real() - s1.imag() * q__2.imag(), q__1.imag() = s1.real() * q__2.imag() + s1.imag() * q__2.real();
    bi->real() = q__1.real(), bi->imag() = q__1.imag();
    return 0;
L100:
    q__1.real() = z__->real() * s1.real() - z__->imag() * s1.imag(), q__1.imag() = z__->real() * s1.imag() + z__->imag() * s1.real();
    s1.real() = q__1.real(), s1.imag() = q__1.imag();
    r__1   = (float)1. / sfac;
    q__2.real() = r__1, q__2.imag() = (float)0.;
    q__1.real() = s1.real() * q__2.real() - s1.imag() * q__2.imag(), q__1.imag() = s1.real() * q__2.imag() + s1.imag() * q__2.real();
    bi->real() = q__1.real(), bi->imag() = q__1.imag();
    return 0;
L110:
    aa     = c1 * ((float)1. - fid) + fid * c2;
    q__1.real() = aa, q__1.imag() = (float)0.;
    bi->real() = q__1.real(), bi->imag() = q__1.imag();
    return 0;
L170:
    nz    = 0;
    *ierr = 2;
    return 0;
L180:
    if(nz == -1)
    {
        goto L170;
    }
    nz    = 0;
    *ierr = 5;
    return 0;
L190:
    *ierr = 4;
    nz    = 0;
    return 0;
} /* cbiry_ */

 int cbknu_(System::Complex<float>* z__, float* fnu, int32* kode, int32* n, System::Complex<float>* y, int32* nz, float* tol, float* elim, float* alim)
{
    /* Initialized data */

    static int32 kmax  = 30;
    static float    r1    = (float)2.;
    static System::Complex<float> czero = {(float)0., (float)0.};
    static System::Complex<float> cone  = {(float)1., (float)0.};
    static System::Complex<float> ctwo  = {(float)2., (float)0.};
    static float    pi    = (float)3.14159265358979324;
    static float    rthpi = (float)1.25331413731550025;
    static float    spi   = (float)1.90985931710274403;
    static float    hpi   = (float)1.57079632679489662;
    static float    fpi   = (float)1.89769999331517738;
    static float    tth   = (float).666666666666666666;
    static float    cc[8] = {(float).577215664901532861,
                         (float)-.0420026350340952355,
                         (float)-.0421977345555443367,
                         (float).00721894324666309954,
                         (float)-2.15241674114950973e-4,
                         (float)-2.01348547807882387e-5,
                         (float)1.13302723198169588e-6,
                         (float)6.11609510448141582e-9};

    /* System generated locals */
    int32 i__1, i__2;
    float    r__1, r__2;
    System::Complex<float> q__1, q__2, q__3, q__4, q__5, q__6;

 

    
    System::Complex<float>                     f;
    int32                     i__, j, k;
    System::Complex<float>                     p, q;
    float                        s, a1, a2, g1, g2;
    System::Complex<float>                     p1, p2, s1, s2;
    float                        t1, t2, aa, bb, fc, ak, bk;
    System::Complex<float>                     ck;
    int32                     ic;
    float                        fk, as;
    System::Complex<float>                     cs;
    int32                     kk;
    System::Complex<float>                     cy[2], cz, zd;
    float                        rk, xd, tm, yd;
    System::Complex<float>                     pt;
    int32                     nw;
    System::Complex<float>                     st, rz;
    float                        xx, yy, p2i, p2m, p2r;
    System::Complex<float>                     cch, csh;
    float                        caz, fhs, elm, fks, dnu;
    System::Complex<float>                     csr[3], css[3], fmu;
    float                        bry[3];
    int32                     inu;
    System::Complex<float>                     smu;
    float                        dnu2;
    System::Complex<float>                     coef, celm;
    float                        alas;
    System::Complex<float>                     cscl, crsc;
    int32                     inub, idum, iflag, kflag, koded;
    float                        ascle;
    float                        helim;
    float                        etest;

    /* *********************************************************************72 */

    /* ***BEGIN PROLOGUE  CBKNU */
    /* ***REFER TO  CBESI,CBESK,CAIRY,CBESH */

    /*     CBKNU COMPUTES THE K BESSEL FUNCTION IN THE RIGHT HALF Z PLANE */

    /* ***ROUTINES CALLED  CKSCL,CSHCH,GAMLN,I1MACH,R1MACH,CUCHK */
    /* ***END PROLOGUE  CBKNU */

    /* Parameter adjustments */
    --y;

    /* Function Body */

    xx     = z__->real();
    yy     = r_imag(z__);
    caz    = c_abs(z__);
    r__1   = (float)1. / *tol;
    q__1.real() = r__1, q__1.imag() = (float)0.;
    cscl.real() = q__1.real(), cscl.imag() = q__1.imag();
    q__1.real() = *tol, q__1.imag() = (float)0.;
    crsc.real() = q__1.real(), crsc.imag() = q__1.imag();
    css[0].real() = cscl.real(), css[0].imag() = cscl.imag();
    css[1].real() = cone.real(), css[1].imag() = cone.imag();
    css[2].real() = crsc.real(), css[2].imag() = crsc.imag();
    csr[0].real() = crsc.real(), csr[0].imag() = crsc.imag();
    csr[1].real() = cone.real(), csr[1].imag() = cone.imag();
    csr[2].real() = cscl.real(), csr[2].imag() = cscl.imag();
    bry[0] = r1mach_(&c__1) * (float)1e3 / *tol;
    bry[1] = (float)1. / bry[0];
    bry[2] = r1mach_(&c__2);
    *nz    = 0;
    iflag  = 0;
    koded  = *kode;
    c_div(&q__1, &ctwo, z__);
    rz.real() = q__1.real(), rz.imag() = q__1.imag();
    inu = (int32)(*fnu + (float).5);
    dnu = *fnu - (float)inu;
    if(abs(dnu) == (float).5)
    {
        goto L110;
    }
    dnu2 = (float)0.;
    if(abs(dnu) > *tol)
    {
        dnu2 = dnu * dnu;
    }
    if(caz > r1)
    {
        goto L110;
    }
    /* ----------------------------------------------------------------------- */
    /*     SERIES FOR CABS(Z).LE.R1 */
    /* ----------------------------------------------------------------------- */
    fc = (float)1.;
    c_log(&q__1, &rz);
    smu.real() = q__1.real(), smu.imag() = q__1.imag();
    q__2.real() = dnu, q__2.imag() = (float)0.;
    q__1.real() = smu.real() * q__2.real() - smu.imag() * q__2.imag(), q__1.imag() = smu.real() * q__2.imag() + smu.imag() * q__2.real();
    fmu.real() = q__1.real(), fmu.imag() = q__1.imag();
    cshch_(&fmu, &csh, &cch);
    if(dnu == (float)0.)
    {
        goto L10;
    }
    fc = dnu * pi;
    fc /= sin(fc);
    r__1   = (float)1. / dnu;
    q__2.real() = r__1, q__2.imag() = (float)0.;
    q__1.real() = csh.real() * q__2.real() - csh.imag() * q__2.imag(), q__1.imag() = csh.real() * q__2.imag() + csh.imag() * q__2.real();
    smu.real() = q__1.real(), smu.imag() = q__1.imag();
L10:
    a2 = dnu + (float)1.;
    /* ----------------------------------------------------------------------- */
    /*     GAM(1-Z)*GAM(1+Z)=PI*Z/SIN(PI*Z), T1=1/GAM(1-DNU), T2=1/GAM(1+DNU) */
    /* ----------------------------------------------------------------------- */
    t2 = exp(-gamln_(&a2, &idum));
    t1 = (float)1. / (t2 * fc);
    if(abs(dnu) > (float).1)
    {
        goto L40;
    }
    /* ----------------------------------------------------------------------- */
    /*     SERIES FOR F0 TO RESOLVE INDETERMINACY FOR SMALL ABS(DNU) */
    /* ----------------------------------------------------------------------- */
    ak = (float)1.;
    s  = cc[0];
    for(k = 2; k <= 8; ++k)
    {
        ak *= dnu2;
        tm = cc[k - 1] * ak;
        s += tm;
        if(abs(tm) < *tol)
        {
            goto L30;
        }
        /* L20: */
    }
L30:
    g1 = -s;
    goto L50;
L40:
    g1 = (t1 - t2) / (dnu + dnu);
L50:
    g2 = (t1 + t2) * (float).5 * fc;
    g1 *= fc;
    q__3.real() = g1, q__3.imag() = (float)0.;
    q__2.real() = q__3.real() * cch.real() - q__3.imag() * cch.imag(), q__2.imag() = q__3.real() * cch.imag() + q__3.imag() * cch.real();
    q__5.real() = g2, q__5.imag() = (float)0.;
    q__4.real() = smu.real() * q__5.real() - smu.imag() * q__5.imag(), q__4.imag() = smu.real() * q__5.imag() + smu.imag() * q__5.real();
    q__1.real() = q__2.real() + q__4.real(), q__1.imag() = q__2.imag() + q__4.imag();
    f.real() = q__1.real(), f.imag() = q__1.imag();
    c_exp(&q__1, &fmu);
    pt.real() = q__1.real(), pt.imag() = q__1.imag();
    r__1   = (float).5 / t2;
    q__2.real() = r__1, q__2.imag() = (float)0.;
    q__1.real() = q__2.real() * pt.real() - q__2.imag() * pt.imag(), q__1.imag() = q__2.real() * pt.imag() + q__2.imag() * pt.real();
    p.real() = q__1.real(), p.imag() = q__1.imag();
    r__1   = (float).5 / t1;
    q__2.real() = r__1, q__2.imag() = (float)0.;
    c_div(&q__1, &q__2, &pt);
    q.real() = q__1.real(), q.imag() = q__1.imag();
    s1.real() = f.real(), s1.imag() = f.imag();
    s2.real() = p.real(), s2.imag() = p.imag();
    ak   = (float)1.;
    a1   = (float)1.;
    ck.real() = cone.real(), ck.imag() = cone.imag();
    bk = (float)1. - dnu2;
    if(inu > 0 || *n > 1)
    {
        goto L80;
    }
    /* ----------------------------------------------------------------------- */
    /*     GENERATE K(FNU,Z), 0.0D0 .LE. FNU .LT. 0.5D0 AND N=1 */
    /* ----------------------------------------------------------------------- */
    if(caz < *tol)
    {
        goto L70;
    }
    q__2.real() = z__->real() * z__->real() - z__->imag() * z__->imag(), q__2.imag() = z__->real() * z__->imag() + z__->imag() * z__->real();
    q__1.real() = q__2.real() * (float).25 - q__2.imag() * (float)0., q__1.imag() = q__2.real() * (float)0. + q__2.imag() * (float).25;
    cz.real() = q__1.real(), cz.imag() = q__1.imag();
    t1 = caz * (float).25 * caz;
L60:
    q__5.real() = ak, q__5.imag() = (float)0.;
    q__4.real() = f.real() * q__5.real() - f.imag() * q__5.imag(), q__4.imag() = f.real() * q__5.imag() + f.imag() * q__5.real();
    q__3.real() = q__4.real() + p.real(), q__3.imag() = q__4.imag() + p.imag();
    q__2.real() = q__3.real() + q.real(), q__2.imag() = q__3.imag() + q.imag();
    r__1   = (float)1. / bk;
    q__6.real() = r__1, q__6.imag() = (float)0.;
    q__1.real() = q__2.real() * q__6.real() - q__2.imag() * q__6.imag(), q__1.imag() = q__2.real() * q__6.imag() + q__2.imag() * q__6.real();
    f.real() = q__1.real(), f.imag() = q__1.imag();
    r__1   = (float)1. / (ak - dnu);
    q__2.real() = r__1, q__2.imag() = (float)0.;
    q__1.real() = p.real() * q__2.real() - p.imag() * q__2.imag(), q__1.imag() = p.real() * q__2.imag() + p.imag() * q__2.real();
    p.real() = q__1.real(), p.imag() = q__1.imag();
    r__1   = (float)1. / (ak + dnu);
    q__2.real() = r__1, q__2.imag() = (float)0.;
    q__1.real() = q.real() * q__2.real() - q.imag() * q__2.imag(), q__1.imag() = q.real() * q__2.imag() + q.imag() * q__2.real();
    q.real() = q__1.real(), q.imag() = q__1.imag();
    rk     = (float)1. / ak;
    q__2.real() = ck.real() * cz.real() - ck.imag() * cz.imag(), q__2.imag() = ck.real() * cz.imag() + ck.imag() * cz.real();
    q__3.real() = rk, q__3.imag() = (float)0.;
    q__1.real() = q__2.real() * q__3.real() - q__2.imag() * q__3.imag(), q__1.imag() = q__2.real() * q__3.imag() + q__2.imag() * q__3.real();
    ck.real() = q__1.real(), ck.imag() = q__1.imag();
    q__2.real() = ck.real() * f.real() - ck.imag() * f.imag(), q__2.imag() = ck.real() * f.imag() + ck.imag() * f.real();
    q__1.real() = s1.real() + q__2.real(), q__1.imag() = s1.imag() + q__2.imag();
    s1.real() = q__1.real(), s1.imag() = q__1.imag();
    a1 = a1 * t1 * rk;
    bk = bk + ak + ak + (float)1.;
    ak += (float)1.;
    if(a1 > *tol)
    {
        goto L60;
    }
L70:
    y[1].real() = s1.real(), y[1].imag() = s1.imag();
    if(koded == 1)
    {
        return 0;
    }
    c_exp(&q__2, z__);
    q__1.real() = s1.real() * q__2.real() - s1.imag() * q__2.imag(), q__1.imag() = s1.real() * q__2.imag() + s1.imag() * q__2.real();
    y[1].real() = q__1.real(), y[1].imag() = q__1.imag();
    return 0;
/* ----------------------------------------------------------------------- */
/*     GENERATE K(DNU,Z) AND K(DNU+1,Z) FOR FORWARD RECURRENCE */
/* ----------------------------------------------------------------------- */
L80:
    if(caz < *tol)
    {
        goto L100;
    }
    q__2.real() = z__->real() * z__->real() - z__->imag() * z__->imag(), q__2.imag() = z__->real() * z__->imag() + z__->imag() * z__->real();
    q__1.real() = q__2.real() * (float).25 - q__2.imag() * (float)0., q__1.imag() = q__2.real() * (float)0. + q__2.imag() * (float).25;
    cz.real() = q__1.real(), cz.imag() = q__1.imag();
    t1 = caz * (float).25 * caz;
L90:
    q__5.real() = ak, q__5.imag() = (float)0.;
    q__4.real() = f.real() * q__5.real() - f.imag() * q__5.imag(), q__4.imag() = f.real() * q__5.imag() + f.imag() * q__5.real();
    q__3.real() = q__4.real() + p.real(), q__3.imag() = q__4.imag() + p.imag();
    q__2.real() = q__3.real() + q.real(), q__2.imag() = q__3.imag() + q.imag();
    r__1   = (float)1. / bk;
    q__6.real() = r__1, q__6.imag() = (float)0.;
    q__1.real() = q__2.real() * q__6.real() - q__2.imag() * q__6.imag(), q__1.imag() = q__2.real() * q__6.imag() + q__2.imag() * q__6.real();
    f.real() = q__1.real(), f.imag() = q__1.imag();
    r__1   = (float)1. / (ak - dnu);
    q__2.real() = r__1, q__2.imag() = (float)0.;
    q__1.real() = p.real() * q__2.real() - p.imag() * q__2.imag(), q__1.imag() = p.real() * q__2.imag() + p.imag() * q__2.real();
    p.real() = q__1.real(), p.imag() = q__1.imag();
    r__1   = (float)1. / (ak + dnu);
    q__2.real() = r__1, q__2.imag() = (float)0.;
    q__1.real() = q.real() * q__2.real() - q.imag() * q__2.imag(), q__1.imag() = q.real() * q__2.imag() + q.imag() * q__2.real();
    q.real() = q__1.real(), q.imag() = q__1.imag();
    rk     = (float)1. / ak;
    q__2.real() = ck.real() * cz.real() - ck.imag() * cz.imag(), q__2.imag() = ck.real() * cz.imag() + ck.imag() * cz.real();
    q__3.real() = rk, q__3.imag() = (float)0.;
    q__1.real() = q__2.real() * q__3.real() - q__2.imag() * q__3.imag(), q__1.imag() = q__2.real() * q__3.imag() + q__2.imag() * q__3.real();
    ck.real() = q__1.real(), ck.imag() = q__1.imag();
    q__2.real() = ck.real() * f.real() - ck.imag() * f.imag(), q__2.imag() = ck.real() * f.imag() + ck.imag() * f.real();
    q__1.real() = s1.real() + q__2.real(), q__1.imag() = s1.imag() + q__2.imag();
    s1.real() = q__1.real(), s1.imag() = q__1.imag();
    q__5.real() = ak, q__5.imag() = (float)0.;
    q__4.real() = f.real() * q__5.real() - f.imag() * q__5.imag(), q__4.imag() = f.real() * q__5.imag() + f.imag() * q__5.real();
    q__3.real() = p.real() - q__4.real(), q__3.imag() = p.imag() - q__4.imag();
    q__2.real() = ck.real() * q__3.real() - ck.imag() * q__3.imag(), q__2.imag() = ck.real() * q__3.imag() + ck.imag() * q__3.real();
    q__1.real() = s2.real() + q__2.real(), q__1.imag() = s2.imag() + q__2.imag();
    s2.real() = q__1.real(), s2.imag() = q__1.imag();
    a1 = a1 * t1 * rk;
    bk = bk + ak + ak + (float)1.;
    ak += (float)1.;
    if(a1 > *tol)
    {
        goto L90;
    }
L100:
    kflag = 2;
    bk    = smu.real();
    a1    = *fnu + (float)1.;
    ak    = a1 * abs(bk);
    if(ak > *alim)
    {
        kflag = 3;
    }
    i__1   = kflag - 1;
    q__1.real() = s2.real() * css[i__1].real() - s2.imag() * css[i__1].imag(), q__1.imag() = s2.real() * css[i__1].imag() + s2.imag() * css[i__1].real();
    p2.real() = q__1.real(), p2.imag() = q__1.imag();
    q__1.real() = p2.real() * rz.real() - p2.imag() * rz.imag(), q__1.imag() = p2.real() * rz.imag() + p2.imag() * rz.real();
    s2.real() = q__1.real(), s2.imag() = q__1.imag();
    i__1   = kflag - 1;
    q__1.real() = s1.real() * css[i__1].real() - s1.imag() * css[i__1].imag(), q__1.imag() = s1.real() * css[i__1].imag() + s1.imag() * css[i__1].real();
    s1.real() = q__1.real(), s1.imag() = q__1.imag();
    if(koded == 1)
    {
        goto L210;
    }
    c_exp(&q__1, z__);
    f.real() = q__1.real(), f.imag() = q__1.imag();
    q__1.real() = s1.real() * f.real() - s1.imag() * f.imag(), q__1.imag() = s1.real() * f.imag() + s1.imag() * f.real();
    s1.real() = q__1.real(), s1.imag() = q__1.imag();
    q__1.real() = s2.real() * f.real() - s2.imag() * f.imag(), q__1.imag() = s2.real() * f.imag() + s2.imag() * f.real();
    s2.real() = q__1.real(), s2.imag() = q__1.imag();
    goto L210;
/* ----------------------------------------------------------------------- */
/*     IFLAG=0 MEANS NO UNDERFLOW OCCURRED */
/*     IFLAG=1 MEANS AN UNDERFLOW OCCURRED- COMPUTATION PROCEEDS WITH */
/*     KODED=2 AND A TEST FOR ON SCALE VALUES IS MADE DURING FORWARD */
/*     RECURSION */
/* ----------------------------------------------------------------------- */
L110:
    q__2.real() = rthpi, q__2.imag() = (float)0.;
    c_sqrt(&q__3, z__);
    c_div(&q__1, &q__2, &q__3);
    coef.real() = q__1.real(), coef.imag() = q__1.imag();
    kflag = 2;
    if(koded == 2)
    {
        goto L120;
    }
    if(xx > *alim)
    {
        goto L290;
    }
    /*     BLANK LINE */
    i__1   = kflag - 1;
    a1     = exp(-xx) * css[i__1].real();
    q__2.real() = a1, q__2.imag() = (float)0.;
    r__1   = cos(yy);
    r__2   = -sin(yy);
    q__3.real() = r__1, q__3.imag() = r__2;
    q__1.real() = q__2.real() * q__3.real() - q__2.imag() * q__3.imag(), q__1.imag() = q__2.real() * q__3.imag() + q__2.imag() * q__3.real();
    pt.real() = q__1.real(), pt.imag() = q__1.imag();
    q__1.real() = coef.real() * pt.real() - coef.imag() * pt.imag(), q__1.imag() = coef.real() * pt.imag() + coef.imag() * pt.real();
    coef.real() = q__1.real(), coef.imag() = q__1.imag();
L120:
    if(abs(dnu) == (float).5)
    {
        goto L300;
    }
    /* ----------------------------------------------------------------------- */
    /*     MILLER ALGORITHM FOR CABS(Z).GT.R1 */
    /* ----------------------------------------------------------------------- */
    ak = cos(pi * dnu);
    ak = abs(ak);
    if(ak == (float)0.)
    {
        goto L300;
    }
    fhs = (r__1 = (float).25 - dnu2, abs(r__1));
    if(fhs == (float)0.)
    {
        goto L300;
    }
    /* ----------------------------------------------------------------------- */
    /*     COMPUTE R2=F(E). IF CABS(Z).GE.R2, USE FORWARD RECURRENCE TO */
    /*     DETERMINE THE BACKWARD INDEX K. R2=F(E) IS A STRAIGHT LINE ON */
    /*     12.LE.E.LE.60. E IS COMPUTED FROM 2**(-E)=B**(1-I1MACH(11))= */
    /*     TOL WHERE B IS THE BASE OF THE ARITHMETIC. */
    /* ----------------------------------------------------------------------- */
    t1 = (float)(i1mach_(&c__11) - 1) * r1mach_(&c__5) * (float)3.321928094;
    t1 = max(t1, (float)12.);
    t1 = std::min(t1, (float)60.);
    t2 = tth * t1 - (float)6.;
    if(xx != (float)0.)
    {
        goto L130;
    }
    t1 = hpi;
    goto L140;
L130:
    t1 = atan(yy / xx);
    t1 = abs(t1);
L140:
    if(t2 > caz)
    {
        goto L170;
    }
    /* ----------------------------------------------------------------------- */
    /*     FORWARD RECURRENCE LOOP WHEN CABS(Z).GE.R2 */
    /* ----------------------------------------------------------------------- */
    etest = ak / (pi * caz * *tol);
    fk    = (float)1.;
    if(etest < (float)1.)
    {
        goto L180;
    }
    fks  = (float)2.;
    rk   = caz + caz + (float)2.;
    a1   = (float)0.;
    a2   = (float)1.;
    i__1 = kmax;
    for(i__ = 1; i__ <= i__1; ++i__)
    {
        ak = fhs / fks;
        bk = rk / (fk + (float)1.);
        tm = a2;
        a2 = bk * a2 - ak * a1;
        a1 = tm;
        rk += (float)2.;
        fks = fks + fk + fk + (float)2.;
        fhs = fhs + fk + fk;
        fk += (float)1.;
        tm = abs(a2) * fk;
        if(etest < tm)
        {
            goto L160;
        }
        /* L150: */
    }
    goto L310;
L160:
    fk += spi * t1 * sqrt(t2 / caz);
    fhs = (r__1 = (float).25 - dnu2, abs(r__1));
    goto L180;
L170:
    /* ----------------------------------------------------------------------- */
    /*     COMPUTE BACKWARD INDEX K FOR CABS(Z).LT.R2 */
    /* ----------------------------------------------------------------------- */
    a2 = sqrt(caz);
    ak = fpi * ak / (*tol * sqrt(a2));
    aa = t1 * (float)3. / (caz + (float)1.);
    bb = t1 * (float)14.7 / (caz + (float)28.);
    ak = (log(ak) + caz * cos(aa) / (caz * (float).008 + (float)1.)) / cos(bb);
    fk = ak * (float).12125 * ak / caz + (float)1.5;
L180:
    k = (int32)fk;
    /* ----------------------------------------------------------------------- */
    /*     BACKWARD RECURRENCE LOOP FOR MILLER ALGORITHM */
    /* ----------------------------------------------------------------------- */
    fk   = (float)k;
    fks  = fk * fk;
    p1.real() = czero.real(), p1.imag() = czero.imag();
    q__1.real() = *tol, q__1.imag() = (float)0.;
    p2.real() = q__1.real(), p2.imag() = q__1.imag();
    cs.real() = p2.real(), cs.imag() = p2.imag();
    i__1 = k;
    for(i__ = 1; i__ <= i__1; ++i__)
    {
        a1   = fks - fk;
        a2   = (fks + fk) / (a1 + fhs);
        rk   = (float)2. / (fk + (float)1.);
        t1   = (fk + xx) * rk;
        t2   = yy * rk;
        pt.real() = p2.real(), pt.imag() = p2.imag();
        q__4.real() = t1, q__4.imag() = t2;
        q__3.real() = p2.real() * q__4.real() - p2.imag() * q__4.imag(), q__3.imag() = p2.real() * q__4.imag() + p2.imag() * q__4.real();
        q__2.real() = q__3.real() - p1.real(), q__2.imag() = q__3.imag() - p1.imag();
        q__5.real() = a2, q__5.imag() = (float)0.;
        q__1.real() = q__2.real() * q__5.real() - q__2.imag() * q__5.imag(), q__1.imag() = q__2.real() * q__5.imag() + q__2.imag() * q__5.real();
        p2.real() = q__1.real(), p2.imag() = q__1.imag();
        p1.real() = pt.real(), p1.imag() = pt.imag();
        q__1.real() = cs.real() + p2.real(), q__1.imag() = cs.imag() + p2.imag();
        cs.real() = q__1.real(), cs.imag() = q__1.imag();
        fks = a1 - fk + (float)1.;
        fk += (float)-1.;
        /* L190: */
    }
    /* ----------------------------------------------------------------------- */
    /*     COMPUTE (P2/CS)=(P2/CABS(CS))*(CONJG(CS)/CABS(CS)) FOR BETTER */
    /*     SCALING */
    /* ----------------------------------------------------------------------- */
    tm     = c_abs(&cs);
    r__1   = (float)1. / tm;
    q__1.real() = r__1, q__1.imag() = (float)0.;
    pt.real() = q__1.real(), pt.imag() = q__1.imag();
    q__1.real() = pt.real() * p2.real() - pt.imag() * p2.imag(), q__1.imag() = pt.real() * p2.imag() + pt.imag() * p2.real();
    s1.real() = q__1.real(), s1.imag() = q__1.imag();
    r_cnjg(&q__2, &cs);
    q__1.real() = q__2.real() * pt.real() - q__2.imag() * pt.imag(), q__1.imag() = q__2.real() * pt.imag() + q__2.imag() * pt.real();
    cs.real() = q__1.real(), cs.imag() = q__1.imag();
    q__2.real() = coef.real() * s1.real() - coef.imag() * s1.imag(), q__2.imag() = coef.real() * s1.imag() + coef.imag() * s1.real();
    q__1.real() = q__2.real() * cs.real() - q__2.imag() * cs.imag(), q__1.imag() = q__2.real() * cs.imag() + q__2.imag() * cs.real();
    s1.real() = q__1.real(), s1.imag() = q__1.imag();
    if(inu > 0 || *n > 1)
    {
        goto L200;
    }
    zd.real() = z__->real(), zd.imag() = z__->imag();
    if(iflag == 1)
    {
        goto L270;
    }
    goto L240;
L200:
    /* ----------------------------------------------------------------------- */
    /*     COMPUTE P1/P2=(P1/CABS(P2)*CONJG(P2)/CABS(P2) FOR SCALING */
    /* ----------------------------------------------------------------------- */
    tm     = c_abs(&p2);
    r__1   = (float)1. / tm;
    q__1.real() = r__1, q__1.imag() = (float)0.;
    pt.real() = q__1.real(), pt.imag() = q__1.imag();
    q__1.real() = pt.real() * p1.real() - pt.imag() * p1.imag(), q__1.imag() = pt.real() * p1.imag() + pt.imag() * p1.real();
    p1.real() = q__1.real(), p1.imag() = q__1.imag();
    r_cnjg(&q__2, &p2);
    q__1.real() = q__2.real() * pt.real() - q__2.imag() * pt.imag(), q__1.imag() = q__2.real() * pt.imag() + q__2.imag() * pt.real();
    p2.real() = q__1.real(), p2.imag() = q__1.imag();
    q__1.real() = p1.real() * p2.real() - p1.imag() * p2.imag(), q__1.imag() = p1.real() * p2.imag() + p1.imag() * p2.real();
    pt.real() = q__1.real(), pt.imag() = q__1.imag();
    r__1   = dnu + (float).5;
    q__5.real() = r__1, q__5.imag() = (float)0.;
    q__4.real() = q__5.real() - pt.real(), q__4.imag() = q__5.imag() - pt.imag();
    c_div(&q__3, &q__4, z__);
    q__2.real() = cone.real() + q__3.real(), q__2.imag() = cone.imag() + q__3.imag();
    q__1.real() = s1.real() * q__2.real() - s1.imag() * q__2.imag(), q__1.imag() = s1.real() * q__2.imag() + s1.imag() * q__2.real();
    s2.real() = q__1.real(), s2.imag() = q__1.imag();
/* ----------------------------------------------------------------------- */
/*     FORWARD RECURSION ON THE THREE TERM RECURSION RELATION WITH */
/*     SCALING NEAR EXPONENT EXTREMES ON KFLAG=1 OR KFLAG=3 */
/* ----------------------------------------------------------------------- */
L210:
    r__1   = dnu + (float)1.;
    q__2.real() = r__1, q__2.imag() = (float)0.;
    q__1.real() = q__2.real() * rz.real() - q__2.imag() * rz.imag(), q__1.imag() = q__2.real() * rz.imag() + q__2.imag() * rz.real();
    ck.real() = q__1.real(), ck.imag() = q__1.imag();
    if(*n == 1)
    {
        --inu;
    }
    if(inu > 0)
    {
        goto L220;
    }
    if(*n == 1)
    {
        s1.real() = s2.real(), s1.imag() = s2.imag();
    }
    zd.real() = z__->real(), zd.imag() = z__->imag();
    if(iflag == 1)
    {
        goto L270;
    }
    goto L240;
L220:
    inub = 1;
    if(iflag == 1)
    {
        goto L261;
    }
L225:
    i__1 = kflag - 1;
    p1.real() = csr[i__1].real(), p1.imag() = csr[i__1].imag();
    ascle = bry[kflag - 1];
    i__1  = inu;
    for(i__ = inub; i__ <= i__1; ++i__)
    {
        st.real() = s2.real(), st.imag() = s2.imag();
        q__2.real() = ck.real() * s2.real() - ck.imag() * s2.imag(), q__2.imag() = ck.real() * s2.imag() + ck.imag() * s2.real();
        q__1.real() = q__2.real() + s1.real(), q__1.imag() = q__2.imag() + s1.imag();
        s2.real() = q__1.real(), s2.imag() = q__1.imag();
        s1.real() = st.real(), s1.imag() = st.imag();
        q__1.real() = ck.real() + rz.real(), q__1.imag() = ck.imag() + rz.imag();
        ck.real() = q__1.real(), ck.imag() = q__1.imag();
        if(kflag >= 3)
        {
            goto L230;
        }
        q__1.real() = s2.real() * p1.real() - s2.imag() * p1.imag(), q__1.imag() = s2.real() * p1.imag() + s2.imag() * p1.real();
        p2.real() = q__1.real(), p2.imag() = q__1.imag();
        p2r = p2.real();
        p2i = r_imag(&p2);
        p2r = abs(p2r);
        p2i = abs(p2i);
        p2m = max(p2r, p2i);
        if(p2m <= ascle)
        {
            goto L230;
        }
        ++kflag;
        ascle  = bry[kflag - 1];
        q__1.real() = s1.real() * p1.real() - s1.imag() * p1.imag(), q__1.imag() = s1.real() * p1.imag() + s1.imag() * p1.real();
        s1.real() = q__1.real(), s1.imag() = q__1.imag();
        s2.real() = p2.real(), s2.imag() = p2.imag();
        i__2   = kflag - 1;
        q__1.real() = s1.real() * css[i__2].real() - s1.imag() * css[i__2].imag(), q__1.imag() = s1.real() * css[i__2].imag() + s1.imag() * css[i__2].real();
        s1.real() = q__1.real(), s1.imag() = q__1.imag();
        i__2   = kflag - 1;
        q__1.real() = s2.real() * css[i__2].real() - s2.imag() * css[i__2].imag(), q__1.imag() = s2.real() * css[i__2].imag() + s2.imag() * css[i__2].real();
        s2.real() = q__1.real(), s2.imag() = q__1.imag();
        i__2 = kflag - 1;
        p1.real() = csr[i__2].real(), p1.imag() = csr[i__2].imag();
    L230:;
    }
    if(*n == 1)
    {
        s1.real() = s2.real(), s1.imag() = s2.imag();
    }
L240:
    i__1   = kflag - 1;
    q__1.real() = s1.real() * csr[i__1].real() - s1.imag() * csr[i__1].imag(), q__1.imag() = s1.real() * csr[i__1].imag() + s1.imag() * csr[i__1].real();
    y[1].real() = q__1.real(), y[1].imag() = q__1.imag();
    if(*n == 1)
    {
        return 0;
    }
    i__1   = kflag - 1;
    q__1.real() = s2.real() * csr[i__1].real() - s2.imag() * csr[i__1].imag(), q__1.imag() = s2.real() * csr[i__1].imag() + s2.imag() * csr[i__1].real();
    y[2].real() = q__1.real(), y[2].imag() = q__1.imag();
    if(*n == 2)
    {
        return 0;
    }
    kk = 2;
L250:
    ++kk;
    if(kk > *n)
    {
        return 0;
    }
    i__1 = kflag - 1;
    p1.real() = csr[i__1].real(), p1.imag() = csr[i__1].imag();
    ascle = bry[kflag - 1];
    i__1  = *n;
    for(i__ = kk; i__ <= i__1; ++i__)
    {
        p2.real() = s2.real(), p2.imag() = s2.imag();
        q__2.real() = ck.real() * s2.real() - ck.imag() * s2.imag(), q__2.imag() = ck.real() * s2.imag() + ck.imag() * s2.real();
        q__1.real() = q__2.real() + s1.real(), q__1.imag() = q__2.imag() + s1.imag();
        s2.real() = q__1.real(), s2.imag() = q__1.imag();
        s1.real() = p2.real(), s1.imag() = p2.imag();
        q__1.real() = ck.real() + rz.real(), q__1.imag() = ck.imag() + rz.imag();
        ck.real() = q__1.real(), ck.imag() = q__1.imag();
        q__1.real() = s2.real() * p1.real() - s2.imag() * p1.imag(), q__1.imag() = s2.real() * p1.imag() + s2.imag() * p1.real();
        p2.real() = q__1.real(), p2.imag() = q__1.imag();
        i__2      = i__;
        y[i__2].real() = p2.real(), y[i__2].imag() = p2.imag();
        if(kflag >= 3)
        {
            goto L260;
        }
        p2r = p2.real();
        p2i = r_imag(&p2);
        p2r = abs(p2r);
        p2i = abs(p2i);
        p2m = max(p2r, p2i);
        if(p2m <= ascle)
        {
            goto L260;
        }
        ++kflag;
        ascle  = bry[kflag - 1];
        q__1.real() = s1.real() * p1.real() - s1.imag() * p1.imag(), q__1.imag() = s1.real() * p1.imag() + s1.imag() * p1.real();
        s1.real() = q__1.real(), s1.imag() = q__1.imag();
        s2.real() = p2.real(), s2.imag() = p2.imag();
        i__2   = kflag - 1;
        q__1.real() = s1.real() * css[i__2].real() - s1.imag() * css[i__2].imag(), q__1.imag() = s1.real() * css[i__2].imag() + s1.imag() * css[i__2].real();
        s1.real() = q__1.real(), s1.imag() = q__1.imag();
        i__2   = kflag - 1;
        q__1.real() = s2.real() * css[i__2].real() - s2.imag() * css[i__2].imag(), q__1.imag() = s2.real() * css[i__2].imag() + s2.imag() * css[i__2].real();
        s2.real() = q__1.real(), s2.imag() = q__1.imag();
        i__2 = kflag - 1;
        p1.real() = csr[i__2].real(), p1.imag() = csr[i__2].imag();
    L260:;
    }
    return 0;
/* ----------------------------------------------------------------------- */
/*     IFLAG=1 CASES, FORWARD RECURRENCE ON SCALED VALUES ON UNDERFLOW */
/* ----------------------------------------------------------------------- */
L261:
    helim  = *elim * (float).5;
    elm    = exp(-(*elim));
    q__1.real() = elm, q__1.imag() = (float)0.;
    celm.real() = q__1.real(), celm.imag() = q__1.imag();
    ascle = bry[0];
    zd.real() = z__->real(), zd.imag() = z__->imag();
    xd   = xx;
    yd   = yy;
    ic   = -1;
    j    = 2;
    i__1 = inu;
    for(i__ = 1; i__ <= i__1; ++i__)
    {
        st.real() = s2.real(), st.imag() = s2.imag();
        q__2.real() = ck.real() * s2.real() - ck.imag() * s2.imag(), q__2.imag() = ck.real() * s2.imag() + ck.imag() * s2.real();
        q__1.real() = q__2.real() + s1.real(), q__1.imag() = q__2.imag() + s1.imag();
        s2.real() = q__1.real(), s2.imag() = q__1.imag();
        s1.real() = st.real(), s1.imag() = st.imag();
        q__1.real() = ck.real() + rz.real(), q__1.imag() = ck.imag() + rz.imag();
        ck.real() = q__1.real(), ck.imag() = q__1.imag();
        as   = c_abs(&s2);
        alas = log(as);
        p2r  = -xd + alas;
        if(p2r < -(*elim))
        {
            goto L263;
        }
        q__2.real() = -zd.real(), q__2.imag() = -zd.imag();
        c_log(&q__3, &s2);
        q__1.real() = q__2.real() + q__3.real(), q__1.imag() = q__2.imag() + q__3.imag();
        p2.real() = q__1.real(), p2.imag() = q__1.imag();
        p2r    = p2.real();
        p2i    = r_imag(&p2);
        p2m    = exp(p2r) / *tol;
        q__2.real() = p2m, q__2.imag() = (float)0.;
        r__1   = cos(p2i);
        r__2   = sin(p2i);
        q__3.real() = r__1, q__3.imag() = r__2;
        q__1.real() = q__2.real() * q__3.real() - q__2.imag() * q__3.imag(), q__1.imag() = q__2.real() * q__3.imag() + q__2.imag() * q__3.real();
        p1.real() = q__1.real(), p1.imag() = q__1.imag();
        cuchk_(&p1, &nw, &ascle, tol);
        if(nw != 0)
        {
            goto L263;
        }
        j          = 3 - j;
        i__2       = j - 1;
        cy[i__2].real() = p1.real(), cy[i__2].imag() = p1.imag();
        if(ic == i__ - 1)
        {
            goto L264;
        }
        ic = i__;
        goto L262;
    L263:
        if(alas < helim)
        {
            goto L262;
        }
        xd -= *elim;
        q__1.real() = s1.real() * celm.real() - s1.imag() * celm.imag(), q__1.imag() = s1.real() * celm.imag() + s1.imag() * celm.real();
        s1.real() = q__1.real(), s1.imag() = q__1.imag();
        q__1.real() = s2.real() * celm.real() - s2.imag() * celm.imag(), q__1.imag() = s2.real() * celm.imag() + s2.imag() * celm.real();
        s2.real() = q__1.real(), s2.imag() = q__1.imag();
        q__1.real() = xd, q__1.imag() = yd;
        zd.real() = q__1.real(), zd.imag() = q__1.imag();
    L262:;
    }
    if(*n == 1)
    {
        s1.real() = s2.real(), s1.imag() = s2.imag();
    }
    goto L270;
L264:
    kflag = 1;
    inub  = i__ + 1;
    i__1  = j - 1;
    s2.real() = cy[i__1].real(), s2.imag() = cy[i__1].imag();
    j    = 3 - j;
    i__1 = j - 1;
    s1.real() = cy[i__1].real(), s1.imag() = cy[i__1].imag();
    if(inub <= inu)
    {
        goto L225;
    }
    if(*n == 1)
    {
        s1.real() = s2.real(), s1.imag() = s2.imag();
    }
    goto L240;
L270:
    y[1].real() = s1.real(), y[1].imag() = s1.imag();
    if(*n == 1)
    {
        goto L280;
    }
    y[2].real() = s2.real(), y[2].imag() = s2.imag();
L280:
    ascle = bry[0];
    ckscl_(&zd, fnu, n, &y[1], nz, &rz, &ascle, tol, elim);
    inu = *n - *nz;
    if(inu <= 0)
    {
        return 0;
    }
    kk   = *nz + 1;
    i__1 = kk;
    s1.real() = y[i__1].real(), s1.imag() = y[i__1].imag();
    i__1   = kk;
    q__1.real() = s1.real() * csr[0].real() - s1.imag() * csr[0].imag(), q__1.imag() = s1.real() * csr[0].imag() + s1.imag() * csr[0].real();
    y[i__1].real() = q__1.real(), y[i__1].imag() = q__1.imag();
    if(inu == 1)
    {
        return 0;
    }
    kk   = *nz + 2;
    i__1 = kk;
    s2.real() = y[i__1].real(), s2.imag() = y[i__1].imag();
    i__1   = kk;
    q__1.real() = s2.real() * csr[0].real() - s2.imag() * csr[0].imag(), q__1.imag() = s2.real() * csr[0].imag() + s2.imag() * csr[0].real();
    y[i__1].real() = q__1.real(), y[i__1].imag() = q__1.imag();
    if(inu == 2)
    {
        return 0;
    }
    t2     = *fnu + (float)(kk - 1);
    q__2.real() = t2, q__2.imag() = (float)0.;
    q__1.real() = q__2.real() * rz.real() - q__2.imag() * rz.imag(), q__1.imag() = q__2.real() * rz.imag() + q__2.imag() * rz.real();
    ck.real() = q__1.real(), ck.imag() = q__1.imag();
    kflag = 1;
    goto L250;
L290:
    /* ----------------------------------------------------------------------- */
    /*     SCALE BY EXP(Z), IFLAG = 1 CASES */
    /* ----------------------------------------------------------------------- */
    koded = 2;
    iflag = 1;
    kflag = 2;
    goto L120;
/* ----------------------------------------------------------------------- */
/*     FNU=HALF ODD INTEGER CASE, DNU=-0.5 */
/* ----------------------------------------------------------------------- */
L300:
    s1.real() = coef.real(), s1.imag() = coef.imag();
    s2.real() = coef.real(), s2.imag() = coef.imag();
    goto L210;
L310:
    *nz = -2;
    return 0;
} /* cbknu_ */

 int cbuni_(System::Complex<float>* z__,
                            float*    fnu,
                            int32* kode,
                            int32* n,
                            System::Complex<float>* y,
                            int32* nz,
                            int32* nui,
                            int32* nlast,
                            float*    fnul,
                            float*    tol,
                            float*    elim,
                            float*    alim)
{
    /* System generated locals */
    int32 i__1, i__2;
    float    r__1;
    System::Complex<float> q__1, q__2, q__3, q__4;


    
    int32                     i__, k;
    System::Complex<float>                     s1, s2;
    float                        ax, ay;
    int32                     nl;
    System::Complex<float>                     cy[2];
    int32                     nw;
    System::Complex<float>                     st, rz;
    float                        xx, yy, gnu, bry[3], sti, stm, str;
    System::Complex<float>                     cscl, cscr;
    float                        dfnu, fnui;

    int32           iflag;
    float              ascle;
    int32           iform;


    /* *********************************************************************72 */

    /* ***BEGIN PROLOGUE  CBUNI */
    /* ***REFER TO  CBESI,CBESK */

    /*     CBUNI COMPUTES THE I BESSEL FUNCTION FOR LARGE CABS(Z).GT. */
    /*     FNUL AND FNU+N-1.LT.FNUL. THE ORDER IS INCREASED FROM */
    /*     FNU+N-1 GREATER THAN FNUL BY ADDING NUI AND COMPUTING */
    /*     ACCORDING TO THE UNIFORM ASYMPTOTIC EXPANSION FOR I(FNU,Z) */
    /*     ON IFORM=1 AND THE EXPANSION FOR J(FNU,Z) ON IFORM=2 */

    /* ***ROUTINES CALLED  CUNI1,CUNI2,R1MACH */
    /* ***END PROLOGUE  CBUNI */
    /* Parameter adjustments */
    --y;

    /* Function Body */
    *nz   = 0;
    xx    = z__->real();
    yy    = z__->imag();
    ax    = abs(xx) * (float)1.7321;
    ay    = abs(yy);
    iform = 1;
    if(ay > ax)
    {
        iform = 2;
    }
    if(*nui == 0)
    {
        goto L60;
    }
    fnui = (float)(*nui);
    dfnu = *fnu + (float)(*n - 1);
    gnu  = dfnu + fnui;
    if(iform == 2)
    {
        goto L10;
    }
    /* ----------------------------------------------------------------------- */
    /*     ASYMPTOTIC EXPANSION FOR I(FNU,Z) FOR LARGE FNU APPLIED IN */
    /*     -PI/3.LE.ARG(Z).LE.PI/3 */
    /* ----------------------------------------------------------------------- */
    cuni1_(z__, &gnu, kode, &c__2, cy, &nw, nlast, fnul, tol, elim, alim);
    goto L20;
L10:
    /* ----------------------------------------------------------------------- */
    /*     ASYMPTOTIC EXPANSION FOR J(FNU,Z*EXP(M*HPI)) FOR LARGE FNU */
    /*     APPLIED IN PI/3.LT.ABS(ARG(Z)).LE.PI/2 WHERE M=+I OR -I */
    /*     AND HPI=PI/2 */
    /* ----------------------------------------------------------------------- */
    cuni2_(z__, &gnu, kode, &c__2, cy, &nw, nlast, fnul, tol, elim, alim);
L20:
    if(nw < 0)
    {
        goto L50;
    }
    if(nw != 0)
    {
        goto L90;
    }
    ay = c_abs(cy);
    /* ---------------------------------------------------------------------- */
    /*     SCALE BACKWARD RECURRENCE, BRY(3) IS DEFINED BUT NEVER USED */
    /* ---------------------------------------------------------------------- */
    bry[0] = r1mach_(&c__1) * (float)1e3 / *tol;
    bry[1] = (float)1. / bry[0];
    bry[2] = bry[1];
    iflag  = 2;
    ascle  = bry[1];
    ax     = (float)1.;
    q__1.real() = ax, q__1.imag() = (float)0.;
    cscl.real() = q__1.real(), cscl.imag() = q__1.imag();
    if(ay > bry[0])
    {
        goto L21;
    }
    iflag  = 1;
    ascle  = bry[0];
    ax     = (float)1. / *tol;
    q__1.real() = ax, q__1.imag() = (float)0.;
    cscl.real() = q__1.real(), cscl.imag() = q__1.imag();
    goto L25;
L21:
    if(ay < bry[1])
    {
        goto L25;
    }
    iflag  = 3;
    ascle  = bry[2];
    ax     = *tol;
    q__1.real() = ax, q__1.imag() = (float)0.;
    cscl.real() = q__1.real(), cscl.imag() = q__1.imag();
L25:
    ay     = (float)1. / ax;
    q__1.real() = ay, q__1.imag() = (float)0.;
    cscr.real() = q__1.real(), cscr.imag() = q__1.imag();
    q__1.real() = cy[1].real() * cscl.real() - cy[1].imag() * cscl.imag(), q__1.imag() = cy[1].real() * cscl.imag() + cy[1].imag() * cscl.real();
    s1.real() = q__1.real(), s1.imag() = q__1.imag();
    q__1.real() = cy[0].real() * cscl.real() - cy[0].imag() * cscl.imag(), q__1.imag() = cy[0].real() * cscl.imag() + cy[0].imag() * cscl.real();
    s2.real() = q__1.real(), s2.imag() = q__1.imag();
    c_div(&q__1, &c_b17, z__);
    rz.real() = q__1.real(), rz.imag() = q__1.imag();
    i__1 = *nui;
    for(i__ = 1; i__ <= i__1; ++i__)
    {
        st.real() = s2.real(), st.imag() = s2.imag();
        r__1   = dfnu + fnui;
        q__4.real() = r__1, q__4.imag() = (float)0.;
        q__3.real() = q__4.real() * rz.real() - q__4.imag() * rz.imag(), q__3.imag() = q__4.real() * rz.imag() + q__4.imag() * rz.real();
        q__2.real() = q__3.real() * s2.real() - q__3.imag() * s2.imag(), q__2.imag() = q__3.real() * s2.imag() + q__3.imag() * s2.real();
        q__1.real() = q__2.real() + s1.real(), q__1.imag() = q__2.imag() + s1.imag();
        s2.real() = q__1.real(), s2.imag() = q__1.imag();
        s1.real() = st.real(), s1.imag() = st.imag();
        fnui += (float)-1.;
        if(iflag >= 3)
        {
            goto L30;
        }
        q__1.real() = s2.real() * cscr.real() - s2.imag() * cscr.imag(), q__1.imag() = s2.real() * cscr.imag() + s2.imag() * cscr.real();
        st.real() = q__1.real(), st.imag() = q__1.imag();
        str = st.real();
        sti = r_imag(&st);
        str = abs(str);
        sti = abs(sti);
        stm = max(str, sti);
        if(stm <= ascle)
        {
            goto L30;
        }
        ++iflag;
        ascle  = bry[iflag - 1];
        q__1.real() = s1.real() * cscr.real() - s1.imag() * cscr.imag(), q__1.imag() = s1.real() * cscr.imag() + s1.imag() * cscr.real();
        s1.real() = q__1.real(), s1.imag() = q__1.imag();
        s2.real() = st.real(), s2.imag() = st.imag();
        ax *= *tol;
        ay     = (float)1. / ax;
        q__1.real() = ax, q__1.imag() = (float)0.;
        cscl.real() = q__1.real(), cscl.imag() = q__1.imag();
        q__1.real() = ay, q__1.imag() = (float)0.;
        cscr.real() = q__1.real(), cscr.imag() = q__1.imag();
        q__1.real() = s1.real() * cscl.real() - s1.imag() * cscl.imag(), q__1.imag() = s1.real() * cscl.imag() + s1.imag() * cscl.real();
        s1.real() = q__1.real(), s1.imag() = q__1.imag();
        q__1.real() = s2.real() * cscl.real() - s2.imag() * cscl.imag(), q__1.imag() = s2.real() * cscl.imag() + s2.imag() * cscl.real();
        s2.real() = q__1.real(), s2.imag() = q__1.imag();
    L30:;
    }
    i__1   = *n;
    q__1.real() = s2.real() * cscr.real() - s2.imag() * cscr.imag(), q__1.imag() = s2.real() * cscr.imag() + s2.imag() * cscr.real();
    y[i__1].real() = q__1.real(), y[i__1].imag() = q__1.imag();
    if(*n == 1)
    {
        return 0;
    }
    nl   = *n - 1;
    fnui = (float)nl;
    k    = nl;
    i__1 = nl;
    for(i__ = 1; i__ <= i__1; ++i__)
    {
        st.real() = s2.real(), st.imag() = s2.imag();
        r__1   = *fnu + fnui;
        q__4.real() = r__1, q__4.imag() = (float)0.;
        q__3.real() = q__4.real() * rz.real() - q__4.imag() * rz.imag(), q__3.imag() = q__4.real() * rz.imag() + q__4.imag() * rz.real();
        q__2.real() = q__3.real() * s2.real() - q__3.imag() * s2.imag(), q__2.imag() = q__3.real() * s2.imag() + q__3.imag() * s2.real();
        q__1.real() = q__2.real() + s1.real(), q__1.imag() = q__2.imag() + s1.imag();
        s2.real() = q__1.real(), s2.imag() = q__1.imag();
        s1.real() = st.real(), s1.imag() = st.imag();
        q__1.real() = s2.real() * cscr.real() - s2.imag() * cscr.imag(), q__1.imag() = s2.real() * cscr.imag() + s2.imag() * cscr.real();
        st.real() = q__1.real(), st.imag() = q__1.imag();
        i__2      = k;
        y[i__2].real() = st.real(), y[i__2].imag() = st.imag();
        fnui += (float)-1.;
        --k;
        if(iflag >= 3)
        {
            goto L40;
        }
        str = st.real();
        sti = r_imag(&st);
        str = abs(str);
        sti = abs(sti);
        stm = max(str, sti);
        if(stm <= ascle)
        {
            goto L40;
        }
        ++iflag;
        ascle  = bry[iflag - 1];
        q__1.real() = s1.real() * cscr.real() - s1.imag() * cscr.imag(), q__1.imag() = s1.real() * cscr.imag() + s1.imag() * cscr.real();
        s1.real() = q__1.real(), s1.imag() = q__1.imag();
        s2.real() = st.real(), s2.imag() = st.imag();
        ax *= *tol;
        ay     = (float)1. / ax;
        q__1.real() = ax, q__1.imag() = (float)0.;
        cscl.real() = q__1.real(), cscl.imag() = q__1.imag();
        q__1.real() = ay, q__1.imag() = (float)0.;
        cscr.real() = q__1.real(), cscr.imag() = q__1.imag();
        q__1.real() = s1.real() * cscl.real() - s1.imag() * cscl.imag(), q__1.imag() = s1.real() * cscl.imag() + s1.imag() * cscl.real();
        s1.real() = q__1.real(), s1.imag() = q__1.imag();
        q__1.real() = s2.real() * cscl.real() - s2.imag() * cscl.imag(), q__1.imag() = s2.real() * cscl.imag() + s2.imag() * cscl.real();
        s2.real() = q__1.real(), s2.imag() = q__1.imag();
    L40:;
    }
    return 0;
L50:
    *nz = -1;
    if(nw == -2)
    {
        *nz = -2;
    }
    return 0;
L60:
    if(iform == 2)
    {
        goto L70;
    }
    /* ----------------------------------------------------------------------- */
    /*     ASYMPTOTIC EXPANSION FOR I(FNU,Z) FOR LARGE FNU APPLIED IN */
    /*     -PI/3.LE.ARG(Z).LE.PI/3 */
    /* ----------------------------------------------------------------------- */
    cuni1_(z__, fnu, kode, n, &y[1], &nw, nlast, fnul, tol, elim, alim);
    goto L80;
L70:
    /* ----------------------------------------------------------------------- */
    /*     ASYMPTOTIC EXPANSION FOR J(FNU,Z*EXP(M*HPI)) FOR LARGE FNU */
    /*     APPLIED IN PI/3.LT.ABS(ARG(Z)).LE.PI/2 WHERE M=+I OR -I */
    /*     AND HPI=PI/2 */
    /* ----------------------------------------------------------------------- */
    cuni2_(z__, fnu, kode, n, &y[1], &nw, nlast, fnul, tol, elim, alim);
L80:
    if(nw < 0)
    {
        goto L50;
    }
    *nz = nw;
    return 0;
L90:
    *nlast = *n;
    return 0;
} /* cbuni_ */

 int cbunk_(System::Complex<float>* z__, float* fnu, int32* kode, int32* mr, int32* n, System::Complex<float>* y, int32* nz, float* tol, float* elim, float* alim)
{
   

    
    float                        ax, ay, xx, yy;

    /* *********************************************************************72 */

    /* ***BEGIN PROLOGUE  CBUNK */
    /* ***REFER TO  CBESK,CBESH */

    /*     CBUNK COMPUTES THE K BESSEL FUNCTION FOR FNU.GT.FNUL. */
    /*     ACCORDING TO THE UNIFORM ASYMPTOTIC EXPANSION FOR K(FNU,Z) */
    /*     IN CUNK1 AND THE EXPANSION FOR H(2,FNU,Z) IN CUNK2 */

    /* ***ROUTINES CALLED  CUNK1,CUNK2 */
    /* ***END PROLOGUE  CBUNK */
    /* Parameter adjustments */
    --y;

    /* Function Body */
    *nz = 0;
    xx  = z__->real();
    yy  = r_imag(z__);
    ax  = abs(xx) * (float)1.7321;
    ay  = abs(yy);
    if(ay > ax)
    {
        goto L10;
    }
    /* ----------------------------------------------------------------------- */
    /*     ASYMPTOTIC EXPANSION FOR K(FNU,Z) FOR LARGE FNU APPLIED IN */
    /*     -PI/3.LE.ARG(Z).LE.PI/3 */
    /* ----------------------------------------------------------------------- */
    cunk1_(z__, fnu, kode, mr, n, &y[1], nz, tol, elim, alim);
    goto L20;
L10:
    /* ----------------------------------------------------------------------- */
    /*     ASYMPTOTIC EXPANSION FOR H(2,FNU,Z*EXP(M*HPI)) FOR LARGE FNU */
    /*     APPLIED IN PI/3.LT.ABS(ARG(Z)).LE.PI/2 WHERE M=+I OR -I */
    /*     AND HPI=PI/2 */
    /* ----------------------------------------------------------------------- */
    cunk2_(z__, fnu, kode, mr, n, &y[1], nz, tol, elim, alim);
L20:
    return 0;
} /* cbunk_ */

 int ckscl_(System::Complex<float>* zr, float* fnu, int32* n, System::Complex<float>* y, int32* nz, System::Complex<float>* rz, float* ascle, float* tol, float* elim)
{
    /* Initialized data */

    static System::Complex<float> czero = {(float)0., (float)0.};

    /* System generated locals */
    int32 i__1, i__2;
    float    r__1, r__2;
    System::Complex<float> q__1, q__2, q__3;

   

    
    int32                     i__, k;
    System::Complex<float>                     s1, s2;
    float                        aa;
    int32                     ic;
    System::Complex<float>                     ck;
    float                        as, fn;
    System::Complex<float>                     cs;
    int32                     kk;
    System::Complex<float>                     cy[2];
    int32                     nn;
    System::Complex<float>                     zd;
    int32                     nw;
    float                        xx, acs, elm, csi, csr, zri;
    System::Complex<float>                     celm;
    float                        alas;
    float                        helim;

    /* *********************************************************************72 */

    /* ***BEGIN PROLOGUE  CKSCL */
    /* ***REFER TO  CBKNU,CUNK1,CUNK2 */

    /*     SET K FUNCTIONS TO ZERO ON UNDERFLOW, CONTINUE RECURRENCE */
    /*     ON SCALED FUNCTIONS UNTIL TWO MEMBERS COME ON SCALE, THEN */
    /*     RETURN WITH MIN(NZ+2,N) VALUES SCALED BY 1/TOL. */

    /* ***ROUTINES CALLED  CUCHK */
    /* ***END PROLOGUE  CKSCL */
    /* Parameter adjustments */
    --y;

    /* Function Body */

    *nz  = 0;
    ic   = 0;
    xx   = zr->real();
    nn   = std::min(2L, *n);
    i__1 = nn;
    for(i__ = 1; i__ <= i__1; ++i__)
    {
        i__2 = i__;
        s1.real() = y[i__2].real(), s1.imag() = y[i__2].imag();
        i__2       = i__ - 1;
        cy[i__2].real() = s1.real(), cy[i__2].imag() = s1.imag();
        as  = c_abs(&s1);
        acs = -xx + log(as);
        ++(*nz);
        i__2      = i__;
        y[i__2].real() = czero.real(), y[i__2].imag() = czero.imag();
        if(acs < -(*elim))
        {
            goto L10;
        }
        q__2.real() = -zr->real(), q__2.imag() = -zr->imag();
        c_log(&q__3, &s1);
        q__1.real() = q__2.real() + q__3.real(), q__1.imag() = q__2.imag() + q__3.imag();
        cs.real() = q__1.real(), cs.imag() = q__1.imag();
        csr    = cs.real();
        csi    = r_imag(&cs);
        aa     = exp(csr) / *tol;
        q__2.real() = aa, q__2.imag() = (float)0.;
        r__1   = cos(csi);
        r__2   = sin(csi);
        q__3.real() = r__1, q__3.imag() = r__2;
        q__1.real() = q__2.real() * q__3.real() - q__2.imag() * q__3.imag(), q__1.imag() = q__2.real() * q__3.imag() + q__2.imag() * q__3.real();
        cs.real() = q__1.real(), cs.imag() = q__1.imag();
        cuchk_(&cs, &nw, ascle, tol);
        if(nw != 0)
        {
            goto L10;
        }
        i__2      = i__;
        y[i__2].real() = cs.real(), y[i__2].imag() = cs.imag();
        --(*nz);
        ic = i__;
    L10:;
    }
    if(*n == 1)
    {
        return 0;
    }
    if(ic > 1)
    {
        goto L20;
    }
    y[1].real() = czero.real(), y[1].imag() = czero.imag();
    *nz = 2;
L20:
    if(*n == 2)
    {
        return 0;
    }
    if(*nz == 0)
    {
        return 0;
    }
    fn     = *fnu + (float)1.;
    q__2.real() = fn, q__2.imag() = (float)0.;
    q__1.real() = q__2.real() * rz->real() - q__2.imag() * rz->imag(), q__1.imag() = q__2.real() * rz->imag() + q__2.imag() * rz->real();
    ck.real() = q__1.real(), ck.imag() = q__1.imag();
    s1.real() = cy[0].real(), s1.imag() = cy[0].imag();
    s2.real() = cy[1].real(), s2.imag() = cy[1].imag();
    helim  = *elim * (float).5;
    elm    = exp(-(*elim));
    q__1.real() = elm, q__1.imag() = (float)0.;
    celm.real() = q__1.real(), celm.imag() = q__1.imag();
    zri  = r_imag(zr);
    zd.real() = zr->real(), zd.imag() = zr->imag();

    /*     FIND TWO CONSECUTIVE Y VALUES ON SCALE. SCALE RECURRENCE IF */
    /*     S2 GETS LARGER THAN EXP(ELIM/2) */

    i__1 = *n;
    for(i__ = 3; i__ <= i__1; ++i__)
    {
        kk   = i__;
        cs.real() = s2.real(), cs.imag() = s2.imag();
        q__2.real() = ck.real() * s2.real() - ck.imag() * s2.imag(), q__2.imag() = ck.real() * s2.imag() + ck.imag() * s2.real();
        q__1.real() = q__2.real() + s1.real(), q__1.imag() = q__2.imag() + s1.imag();
        s2.real() = q__1.real(), s2.imag() = q__1.imag();
        s1.real() = cs.real(), s1.imag() = cs.imag();
        q__1.real() = ck.real() + rz->real(), q__1.imag() = ck.imag() + rz->imag();
        ck.real() = q__1.real(), ck.imag() = q__1.imag();
        as   = c_abs(&s2);
        alas = log(as);
        acs  = -xx + alas;
        ++(*nz);
        i__2      = i__;
        y[i__2].real() = czero.real(), y[i__2].imag() = czero.imag();
        if(acs < -(*elim))
        {
            goto L25;
        }
        q__2.real() = -zd.real(), q__2.imag() = -zd.imag();
        c_log(&q__3, &s2);
        q__1.real() = q__2.real() + q__3.real(), q__1.imag() = q__2.imag() + q__3.imag();
        cs.real() = q__1.real(), cs.imag() = q__1.imag();
        csr    = cs.real();
        csi    = r_imag(&cs);
        aa     = exp(csr) / *tol;
        q__2.real() = aa, q__2.imag() = (float)0.;
        r__1   = cos(csi);
        r__2   = sin(csi);
        q__3.real() = r__1, q__3.imag() = r__2;
        q__1.real() = q__2.real() * q__3.real() - q__2.imag() * q__3.imag(), q__1.imag() = q__2.real() * q__3.imag() + q__2.imag() * q__3.real();
        cs.real() = q__1.real(), cs.imag() = q__1.imag();
        cuchk_(&cs, &nw, ascle, tol);
        if(nw != 0)
        {
            goto L25;
        }
        i__2      = i__;
        y[i__2].real() = cs.real(), y[i__2].imag() = cs.imag();
        --(*nz);
        if(ic == kk - 1)
        {
            goto L40;
        }
        ic = kk;
        goto L30;
    L25:
        if(alas < helim)
        {
            goto L30;
        }
        xx -= *elim;
        q__1.real() = s1.real() * celm.real() - s1.imag() * celm.imag(), q__1.imag() = s1.real() * celm.imag() + s1.imag() * celm.real();
        s1.real() = q__1.real(), s1.imag() = q__1.imag();
        q__1.real() = s2.real() * celm.real() - s2.imag() * celm.imag(), q__1.imag() = s2.real() * celm.imag() + s2.imag() * celm.real();
        s2.real() = q__1.real(), s2.imag() = q__1.imag();
        q__1.real() = xx, q__1.imag() = zri;
        zd.real() = q__1.real(), zd.imag() = q__1.imag();
    L30:;
    }
    *nz = *n;
    if(ic == *n)
    {
        *nz = *n - 1;
    }
    goto L45;
L40:
    *nz = kk - 2;
L45:
    i__1 = *nz;
    for(k = 1; k <= i__1; ++k)
    {
        i__2      = k;
        y[i__2].real() = czero.real(), y[i__2].imag() = czero.imag();
        /* L50: */
    }
    return 0;
} /* ckscl_ */

 int cmlri_(System::Complex<float>* z__, float* fnu, int32* kode, int32* n, System::Complex<float>* y, int32* nz, float* tol)
{
    /* Initialized data */

    static System::Complex<float> czero = {(float)0., (float)0.};
    static System::Complex<float> cone  = {(float)1., (float)0.};
    static System::Complex<float> ctwo  = {(float)2., (float)0.};

    /* System generated locals */
    int32 i__1, i__2, i__3;
    float    r__1, r__2, r__3;
    System::Complex<float> q__1, q__2, q__3, q__4, q__5;

 
    
    int32           i__, k, m;
    float              x;
    System::Complex<float>           p1, p2;
    float              ak, bk;
    System::Complex<float>           ck;
    float              ap, at;
    int32           kk, km;
    float              az;
    System::Complex<float>           pt, rz;
    float              ack, fnf, fkk;
    int32           iaz;
    float              rho;
    int32           inu;
    System::Complex<float>           sum;
    float              tst, rho2, flam, fkap, scle, tfnf;
    int32           idum, ifnu;
    int32           itime;
    System::Complex<float>           cnorm;
    

    /* *********************************************************************72 */

    /* ***BEGIN PROLOGUE  CMLRI */
    /* ***REFER TO  CBESI,CBESK */

    /*     CMLRI COMPUTES THE I BESSEL FUNCTION FOR RE(Z).GE.0.0 BY THE */
    /*     MILLER ALGORITHM NORMALIZED BY A NEUMANN SERIES. */

    /* ***ROUTINES CALLED  GAMLN,R1MACH */
    /* ***END PROLOGUE  CMLRI */
    /* Parameter adjustments */
    --y;

    /* Function Body */
    scle   = (float)1e3 * r1mach_(&c__1) / *tol;
    *nz    = 0;
    az     = c_abs(z__);
    x      = z__->real();
    iaz    = (int32)az;
    ifnu   = (int32)(*fnu);
    inu    = ifnu + *n - 1;
    at     = (float)iaz + (float)1.;
    q__2.real() = at, q__2.imag() = (float)0.;
    c_div(&q__1, &q__2, z__);
    ck.real() = q__1.real(), ck.imag() = q__1.imag();
    c_div(&q__1, &ctwo, z__);
    rz.real() = q__1.real(), rz.imag() = q__1.imag();
    p1.real() = czero.real(), p1.imag() = czero.imag();
    p2.real() = cone.real(), p2.imag() = cone.imag();
    ack  = (at + (float)1.) / az;
    rho  = ack + sqrt(ack * ack - (float)1.);
    rho2 = rho * rho;
    tst  = (rho2 + rho2) / ((rho2 - (float)1.) * (rho - (float)1.));
    tst /= *tol;
    /* ----------------------------------------------------------------------- */
    /*     COMPUTE RELATIVE TRUNCATION ERROR INDEX FOR SERIES */
    /* ----------------------------------------------------------------------- */
    ak = at;
    for(i__ = 1; i__ <= 80; ++i__)
    {
        pt.real() = p2.real(), pt.imag() = p2.imag();
        q__2.real() = ck.real() * p2.real() - ck.imag() * p2.imag(), q__2.imag() = ck.real() * p2.imag() + ck.imag() * p2.real();
        q__1.real() = p1.real() - q__2.real(), q__1.imag() = p1.imag() - q__2.imag();
        p2.real() = q__1.real(), p2.imag() = q__1.imag();
        p1.real() = pt.real(), p1.imag() = pt.imag();
        q__1.real() = ck.real() + rz.real(), q__1.imag() = ck.imag() + rz.imag();
        ck.real() = q__1.real(), ck.imag() = q__1.imag();
        ap = c_abs(&p2);
        if(ap > tst * ak * ak)
        {
            goto L20;
        }
        ak += (float)1.;
        /* L10: */
    }
    goto L110;
L20:
    ++i__;
    k = 0;
    if(inu < iaz)
    {
        goto L40;
    }
    /* ----------------------------------------------------------------------- */
    /*     COMPUTE RELATIVE TRUNCATION ERROR FOR RATIOS */
    /* ----------------------------------------------------------------------- */
    p1.real() = czero.real(), p1.imag() = czero.imag();
    p2.real() = cone.real(), p2.imag() = cone.imag();
    at     = (float)inu + (float)1.;
    q__2.real() = at, q__2.imag() = (float)0.;
    c_div(&q__1, &q__2, z__);
    ck.real() = q__1.real(), ck.imag() = q__1.imag();
    ack   = at / az;
    tst   = sqrt(ack / *tol);
    itime = 1;
    for(k = 1; k <= 80; ++k)
    {
        pt.real() = p2.real(), pt.imag() = p2.imag();
        q__2.real() = ck.real() * p2.real() - ck.imag() * p2.imag(), q__2.imag() = ck.real() * p2.imag() + ck.imag() * p2.real();
        q__1.real() = p1.real() - q__2.real(), q__1.imag() = p1.imag() - q__2.imag();
        p2.real() = q__1.real(), p2.imag() = q__1.imag();
        p1.real() = pt.real(), p1.imag() = pt.imag();
        q__1.real() = ck.real() + rz.real(), q__1.imag() = ck.imag() + rz.imag();
        ck.real() = q__1.real(), ck.imag() = q__1.imag();
        ap = c_abs(&p2);
        if(ap < tst)
        {
            goto L30;
        }
        if(itime == 2)
        {
            goto L40;
        }
        ack  = c_abs(&ck);
        flam = ack + sqrt(ack * ack - (float)1.);
        fkap = ap / c_abs(&p1);
        rho  = std::min(flam, fkap);
        tst *= sqrt(rho / (rho * rho - (float)1.));
        itime = 2;
    L30:;
    }
    goto L110;
L40:
    /* ----------------------------------------------------------------------- */
    /*     BACKWARD RECURRENCE AND SUM NORMALIZING RELATION */
    /* ----------------------------------------------------------------------- */
    ++k;
    /* Computing MAX */
    i__1 = i__ + iaz, i__2 = k + inu;
    kk   = max(i__1, i__2);
    fkk  = (float)kk;
    p1.real() = czero.real(), p1.imag() = czero.imag();
    /* ----------------------------------------------------------------------- */
    /*     SCALE P2 AND SUM BY SCLE */
    /* ----------------------------------------------------------------------- */
    q__1.real() = scle, q__1.imag() = (float)0.;
    p2.real() = q__1.real(), p2.imag() = q__1.imag();
    fnf   = *fnu - (float)ifnu;
    tfnf  = fnf + fnf;
    r__1  = fkk + tfnf + (float)1.;
    r__2  = fkk + (float)1.;
    r__3  = tfnf + (float)1.;
    bk    = gamln_(&r__1, &idum) - gamln_(&r__2, &idum) - gamln_(&r__3, &idum);
    bk    = exp(bk);
    sum.real() = czero.real(), sum.imag() = czero.imag();
    km   = kk - inu;
    i__1 = km;
    for(i__ = 1; i__ <= i__1; ++i__)
    {
        pt.real() = p2.real(), pt.imag() = p2.imag();
        r__1   = fkk + fnf;
        q__4.real() = r__1, q__4.imag() = (float)0.;
        q__3.real() = q__4.real() * rz.real() - q__4.imag() * rz.imag(), q__3.imag() = q__4.real() * rz.imag() + q__4.imag() * rz.real();
        q__2.real() = q__3.real() * p2.real() - q__3.imag() * p2.imag(), q__2.imag() = q__3.real() * p2.imag() + q__3.imag() * p2.real();
        q__1.real() = p1.real() + q__2.real(), q__1.imag() = p1.imag() + q__2.imag();
        p2.real() = q__1.real(), p2.imag() = q__1.imag();
        p1.real() = pt.real(), p1.imag() = pt.imag();
        ak     = (float)1. - tfnf / (fkk + tfnf);
        ack    = bk * ak;
        r__1   = ack + bk;
        q__3.real() = r__1, q__3.imag() = (float)0.;
        q__2.real() = q__3.real() * p1.real() - q__3.imag() * p1.imag(), q__2.imag() = q__3.real() * p1.imag() + q__3.imag() * p1.real();
        q__1.real() = sum.real() + q__2.real(), q__1.imag() = sum.imag() + q__2.imag();
        sum.real() = q__1.real(), sum.imag() = q__1.imag();
        bk = ack;
        fkk += (float)-1.;
        /* L50: */
    }
    i__1      = *n;
    y[i__1].real() = p2.real(), y[i__1].imag() = p2.imag();
    if(*n == 1)
    {
        goto L70;
    }
    i__1 = *n;
    for(i__ = 2; i__ <= i__1; ++i__)
    {
        pt.real() = p2.real(), pt.imag() = p2.imag();
        r__1   = fkk + fnf;
        q__4.real() = r__1, q__4.imag() = (float)0.;
        q__3.real() = q__4.real() * rz.real() - q__4.imag() * rz.imag(), q__3.imag() = q__4.real() * rz.imag() + q__4.imag() * rz.real();
        q__2.real() = q__3.real() * p2.real() - q__3.imag() * p2.imag(), q__2.imag() = q__3.real() * p2.imag() + q__3.imag() * p2.real();
        q__1.real() = p1.real() + q__2.real(), q__1.imag() = p1.imag() + q__2.imag();
        p2.real() = q__1.real(), p2.imag() = q__1.imag();
        p1.real() = pt.real(), p1.imag() = pt.imag();
        ak     = (float)1. - tfnf / (fkk + tfnf);
        ack    = bk * ak;
        r__1   = ack + bk;
        q__3.real() = r__1, q__3.imag() = (float)0.;
        q__2.real() = q__3.real() * p1.real() - q__3.imag() * p1.imag(), q__2.imag() = q__3.real() * p1.imag() + q__3.imag() * p1.real();
        q__1.real() = sum.real() + q__2.real(), q__1.imag() = sum.imag() + q__2.imag();
        sum.real() = q__1.real(), sum.imag() = q__1.imag();
        bk = ack;
        fkk += (float)-1.;
        m         = *n - i__ + 1;
        i__2      = m;
        y[i__2].real() = p2.real(), y[i__2].imag() = p2.imag();
        /* L60: */
    }
L70:
    if(ifnu <= 0)
    {
        goto L90;
    }
    i__1 = ifnu;
    for(i__ = 1; i__ <= i__1; ++i__)
    {
        pt.real() = p2.real(), pt.imag() = p2.imag();
        r__1   = fkk + fnf;
        q__4.real() = r__1, q__4.imag() = (float)0.;
        q__3.real() = q__4.real() * rz.real() - q__4.imag() * rz.imag(), q__3.imag() = q__4.real() * rz.imag() + q__4.imag() * rz.real();
        q__2.real() = q__3.real() * p2.real() - q__3.imag() * p2.imag(), q__2.imag() = q__3.real() * p2.imag() + q__3.imag() * p2.real();
        q__1.real() = p1.real() + q__2.real(), q__1.imag() = p1.imag() + q__2.imag();
        p2.real() = q__1.real(), p2.imag() = q__1.imag();
        p1.real() = pt.real(), p1.imag() = pt.imag();
        ak     = (float)1. - tfnf / (fkk + tfnf);
        ack    = bk * ak;
        r__1   = ack + bk;
        q__3.real() = r__1, q__3.imag() = (float)0.;
        q__2.real() = q__3.real() * p1.real() - q__3.imag() * p1.imag(), q__2.imag() = q__3.real() * p1.imag() + q__3.imag() * p1.real();
        q__1.real() = sum.real() + q__2.real(), q__1.imag() = sum.imag() + q__2.imag();
        sum.real() = q__1.real(), sum.imag() = q__1.imag();
        bk = ack;
        fkk += (float)-1.;
        /* L80: */
    }
L90:
    pt.real() = z__->real(), pt.imag() = z__->imag();
    if(*kode == 2)
    {
        q__2.real() = x, q__2.imag() = (float)0.;
        q__1.real() = pt.real() - q__2.real(), q__1.imag() = pt.imag() - q__2.imag();
        pt.real() = q__1.real(), pt.imag() = q__1.imag();
    }
    q__4.real() = fnf, q__4.imag() = (float)0.;
    q__3.real() = -q__4.real(), q__3.imag() = -q__4.imag();
    c_log(&q__5, &rz);
    q__2.real() = q__3.real() * q__5.real() - q__3.imag() * q__5.imag(), q__2.imag() = q__3.real() * q__5.imag() + q__3.imag() * q__5.real();
    q__1.real() = q__2.real() + pt.real(), q__1.imag() = q__2.imag() + pt.imag();
    p1.real() = q__1.real(), p1.imag() = q__1.imag();
    r__1   = fnf + (float)1.;
    ap     = gamln_(&r__1, &idum);
    q__2.real() = ap, q__2.imag() = (float)0.;
    q__1.real() = p1.real() - q__2.real(), q__1.imag() = p1.imag() - q__2.imag();
    pt.real() = q__1.real(), pt.imag() = q__1.imag();
    /* ----------------------------------------------------------------------- */
    /*     THE DIVISION CEXP(PT)/(SUM+P2) IS ALTERED TO AVOID OVERFLOW */
    /*     IN THE DENOMINATOR BY SQUARING LARGE QUANTITIES */
    /* ----------------------------------------------------------------------- */
    q__1.real() = p2.real() + sum.real(), q__1.imag() = p2.imag() + sum.imag();
    p2.real() = q__1.real(), p2.imag() = q__1.imag();
    ap     = c_abs(&p2);
    r__1   = (float)1. / ap;
    q__1.real() = r__1, q__1.imag() = (float)0.;
    p1.real() = q__1.real(), p1.imag() = q__1.imag();
    c_exp(&q__2, &pt);
    q__1.real() = q__2.real() * p1.real() - q__2.imag() * p1.imag(), q__1.imag() = q__2.real() * p1.imag() + q__2.imag() * p1.real();
    ck.real() = q__1.real(), ck.imag() = q__1.imag();
    r_cnjg(&q__2, &p2);
    q__1.real() = q__2.real() * p1.real() - q__2.imag() * p1.imag(), q__1.imag() = q__2.real() * p1.imag() + q__2.imag() * p1.real();
    pt.real() = q__1.real(), pt.imag() = q__1.imag();
    q__1.real() = ck.real() * pt.real() - ck.imag() * pt.imag(), q__1.imag() = ck.real() * pt.imag() + ck.imag() * pt.real();
    cnorm.real() = q__1.real(), cnorm.imag() = q__1.imag();
    i__1 = *n;
    for(i__ = 1; i__ <= i__1; ++i__)
    {
        i__2   = i__;
        i__3   = i__;
        q__1.real() = y[i__3].real() * cnorm.real() - y[i__3].imag() * cnorm.imag(), q__1.imag() = y[i__3].real() * cnorm.imag() + y[i__3].imag() * cnorm.real();
        y[i__2].real() = q__1.real(), y[i__2].imag() = q__1.imag();
        /* L100: */
    }
    return 0;
L110:
    *nz = -2;
    return 0;
} /* cmlri_ */

 int crati_(System::Complex<float>* z__, float* fnu, int32* n, System::Complex<float>* cy, float* tol)
{
    /* Initialized data */

    static System::Complex<float> czero = {(float)0., (float)0.};
    static System::Complex<float> cone  = {(float)1., (float)0.};

    /* System generated locals */
    int32 i__1, i__2;
    float    r__1;
    System::Complex<float> q__1, q__2, q__3, q__4;



    
    int32 i__, k;
    System::Complex<float> p1, p2, t1;
    float    ak;
    int32 id, kk;
    float    az;
    System::Complex<float> pt, rz;
    float    ap1, ap2, arg, rho;
    int32 inu;
    float    rap1, flam, dfnu, fdnu;
    int32 magz, idnu;
    float    fnup, test, test1;
    System::Complex<float> cdfnu;
    float    amagz;
    int32 itime;

    /* *********************************************************************72 */

    /* ***BEGIN PROLOGUE  CRATI */
    /* ***REFER TO  CBESI,CBESK,CBESH */

    /*     CRATI COMPUTES RATIOS OF I BESSEL FUNCTIONS BY BACKWARD */
    /*     RECURRENCE.  THE STARTING INDEX IS DETERMINED BY FORWARD */
    /*     RECURRENCE AS DESCRIBED IN J. RES. OF NAT. BUR. OF STANDARDS-B, */
    /*     MATHEMATICAL SCIENCES, VOL 77B, P111-114, SEPTEMBER, 1973, */
    /*     BESSEL FUNCTIONS I AND J OF COMPLEX ARGUMENT AND INTEGER ORDER, */
    /*     BY D. J. SOOKNE. */

    /* ***ROUTINES CALLED  (NONE) */
    /* ***END PROLOGUE  CRATI */
    /* Parameter adjustments */
    --cy;

    /* Function Body */
    az     = c_abs(z__);
    inu    = (int32)(*fnu);
    idnu   = inu + *n - 1;
    fdnu   = (float)idnu;
    magz   = (int32)az;
    amagz  = (float)(magz + 1);
    fnup   = max(amagz, fdnu);
    id     = idnu - magz - 1;
    itime  = 1;
    k      = 1;
    q__2.real() = cone.real() + cone.real(), q__2.imag() = cone.imag() + cone.imag();
    c_div(&q__1, &q__2, z__);
    rz.real() = q__1.real(), rz.imag() = q__1.imag();
    q__2.real() = fnup, q__2.imag() = (float)0.;
    q__1.real() = q__2.real() * rz.real() - q__2.imag() * rz.imag(), q__1.imag() = q__2.real() * rz.imag() + q__2.imag() * rz.real();
    t1.real() = q__1.real(), t1.imag() = q__1.imag();
    q__1.real() = -t1.real(), q__1.imag() = -t1.imag();
    p2.real() = q__1.real(), p2.imag() = q__1.imag();
    p1.real() = cone.real(), p1.imag() = cone.imag();
    q__1.real() = t1.real() + rz.real(), q__1.imag() = t1.imag() + rz.imag();
    t1.real() = q__1.real(), t1.imag() = q__1.imag();
    if(id > 0)
    {
        id = 0;
    }
    ap2 = c_abs(&p2);
    ap1 = c_abs(&p1);
    /* ----------------------------------------------------------------------- */
    /*     THE OVERFLOW TEST ON K(FNU+I-1,Z) BEFORE THE CALL TO CBKNX */
    /*     GUARANTEES THAT P2 IS ON SCALE. SCALE TEST1 AND ALL SUBSEQUENT */
    /*     P2 VALUES BY AP1 TO ENSURE THAT AN OVERFLOW DOES NOT OCCUR */
    /*     PREMATURELY. */
    /* ----------------------------------------------------------------------- */
    arg    = (ap2 + ap2) / (ap1 * *tol);
    test1  = sqrt(arg);
    test   = test1;
    rap1   = (float)1. / ap1;
    q__2.real() = rap1, q__2.imag() = (float)0.;
    q__1.real() = p1.real() * q__2.real() - p1.imag() * q__2.imag(), q__1.imag() = p1.real() * q__2.imag() + p1.imag() * q__2.real();
    p1.real() = q__1.real(), p1.imag() = q__1.imag();
    q__2.real() = rap1, q__2.imag() = (float)0.;
    q__1.real() = p2.real() * q__2.real() - p2.imag() * q__2.imag(), q__1.imag() = p2.real() * q__2.imag() + p2.imag() * q__2.real();
    p2.real() = q__1.real(), p2.imag() = q__1.imag();
    ap2 *= rap1;
L10:
    ++k;
    ap1  = ap2;
    pt.real() = p2.real(), pt.imag() = p2.imag();
    q__2.real() = t1.real() * p2.real() - t1.imag() * p2.imag(), q__2.imag() = t1.real() * p2.imag() + t1.imag() * p2.real();
    q__1.real() = p1.real() - q__2.real(), q__1.imag() = p1.imag() - q__2.imag();
    p2.real() = q__1.real(), p2.imag() = q__1.imag();
    p1.real() = pt.real(), p1.imag() = pt.imag();
    q__1.real() = t1.real() + rz.real(), q__1.imag() = t1.imag() + rz.imag();
    t1.real() = q__1.real(), t1.imag() = q__1.imag();
    ap2 = c_abs(&p2);
    if(ap1 <= test)
    {
        goto L10;
    }
    if(itime == 2)
    {
        goto L20;
    }
    ak   = c_abs(&t1) * (float).5;
    flam = ak + sqrt(ak * ak - (float)1.);
    /* Computing MIN */
    r__1  = ap2 / ap1;
    rho   = std::min(r__1, flam);
    test  = test1 * sqrt(rho / (rho * rho - (float)1.));
    itime = 2;
    goto L10;
L20:
    kk     = k + 1 - id;
    ak     = (float)kk;
    dfnu   = *fnu + (float)(*n - 1);
    q__1.real() = dfnu, q__1.imag() = (float)0.;
    cdfnu.real() = q__1.real(), cdfnu.imag() = q__1.imag();
    q__1.real() = ak, q__1.imag() = (float)0.;
    t1.real() = q__1.real(), t1.imag() = q__1.imag();
    r__1   = (float)1. / ap2;
    q__1.real() = r__1, q__1.imag() = (float)0.;
    p1.real() = q__1.real(), p1.imag() = q__1.imag();
    p2.real() = czero.real(), p2.imag() = czero.imag();
    i__1 = kk;
    for(i__ = 1; i__ <= i__1; ++i__)
    {
        pt.real() = p1.real(), pt.imag() = p1.imag();
        q__4.real() = cdfnu.real() + t1.real(), q__4.imag() = cdfnu.imag() + t1.imag();
        q__3.real() = rz.real() * q__4.real() - rz.imag() * q__4.imag(), q__3.imag() = rz.real() * q__4.imag() + rz.imag() * q__4.real();
        q__2.real() = q__3.real() * p1.real() - q__3.imag() * p1.imag(), q__2.imag() = q__3.real() * p1.imag() + q__3.imag() * p1.real();
        q__1.real() = q__2.real() + p2.real(), q__1.imag() = q__2.imag() + p2.imag();
        p1.real() = q__1.real(), p1.imag() = q__1.imag();
        p2.real() = pt.real(), p2.imag() = pt.imag();
        q__1.real() = t1.real() - cone.real(), q__1.imag() = t1.imag() - cone.imag();
        t1.real() = q__1.real(), t1.imag() = q__1.imag();
        /* L30: */
    }
    if(p1.real() != (float)0. || r_imag(&p1) != (float)0.)
    {
        goto L40;
    }
    q__1.real() = *tol, q__1.imag() = *tol;
    p1.real() = q__1.real(), p1.imag() = q__1.imag();
L40:
    i__1 = *n;
    c_div(&q__1, &p2, &p1);
    cy[i__1].real() = q__1.real(), cy[i__1].imag() = q__1.imag();
    if(*n == 1)
    {
        return 0;
    }
    k      = *n - 1;
    ak     = (float)k;
    q__1.real() = ak, q__1.imag() = (float)0.;
    t1.real() = q__1.real(), t1.imag() = q__1.imag();
    q__2.real() = *fnu, q__2.imag() = (float)0.;
    q__1.real() = q__2.real() * rz.real() - q__2.imag() * rz.imag(), q__1.imag() = q__2.real() * rz.imag() + q__2.imag() * rz.real();
    cdfnu.real() = q__1.real(), cdfnu.imag() = q__1.imag();
    i__1 = *n;
    for(i__ = 2; i__ <= i__1; ++i__)
    {
        q__3.real() = t1.real() * rz.real() - t1.imag() * rz.imag(), q__3.imag() = t1.real() * rz.imag() + t1.imag() * rz.real();
        q__2.real() = cdfnu.real() + q__3.real(), q__2.imag() = cdfnu.imag() + q__3.imag();
        i__2   = k + 1;
        q__1.real() = q__2.real() + cy[i__2].real(), q__1.imag() = q__2.imag() + cy[i__2].imag();
        pt.real() = q__1.real(), pt.imag() = q__1.imag();
        if(pt.real() != (float)0. || r_imag(&pt) != (float)0.)
        {
            goto L50;
        }
        q__1.real() = *tol, q__1.imag() = *tol;
        pt.real() = q__1.real(), pt.imag() = q__1.imag();
    L50:
        i__2 = k;
        c_div(&q__1, &cone, &pt);
        cy[i__2].real() = q__1.real(), cy[i__2].imag() = q__1.imag();
        q__1.real() = t1.real() - cone.real(), q__1.imag() = t1.imag() - cone.imag();
        t1.real() = q__1.real(), t1.imag() = q__1.imag();
        --k;
        /* L60: */
    }
    return 0;
} /* crati_ */

 int cs1s2_(System::Complex<float>* zr, System::Complex<float>* s1, System::Complex<float>* s2, int32* nz, float* ascle, float* alim, int32* iuf)
{
    /* Initialized data */

    static System::Complex<float> czero = {(float)0., (float)0.};

    /* System generated locals */
    System::Complex<float> q__1, q__2, q__3;

    /*  functions */
    , r_imag(System::Complex<float>*);
    , c_exp(System::Complex<float>*, System::Complex<float>*);

    
    System::Complex<float> c1;
    float    aa, xx, as1, as2;
    System::Complex<float> s1d;
    float    aln;

    /* *********************************************************************72 */

    /* ***BEGIN PROLOGUE  CS1S2 */
    /* ***REFER TO  CBESK,CAIRY */

    /*     CS1S2 TESTS FOR A POSSIBLE UNDERFLOW RESULTING FROM THE */
    /*     ADDITION OF THE I AND K FUNCTIONS IN THE ANALYTIC CON- */
    /*     TINUATION FORMULA WHERE S1=K FUNCTION AND S2=I FUNCTION. */
    /*     ON KODE=1 THE I AND K FUNCTIONS ARE DIFFERENT ORDERS OF */
    /*     MAGNITUDE, BUT FOR KODE=2 THEY CAN BE OF THE SAME ORDER */
    /*     OF MAGNITUDE AND THE MAXIMUM MUST BE AT LEAST ONE */
    /*     PRECISION ABOVE THE UNDERFLOW LIMIT. */

    /* ***ROUTINES CALLED  (NONE) */
    /* ***END PROLOGUE  CS1S2 */
    *nz = 0;
    as1 = c_abs(s1);
    as2 = c_abs(s2);
    aa  = s1->real();
    aln = r_imag(s1);
    if(aa == (float)0. && aln == (float)0.)
    {
        goto L10;
    }
    if(as1 == (float)0.)
    {
        goto L10;
    }
    xx    = zr->real();
    aln   = -xx - xx + log(as1);
    s1d.real() = s1->real(), s1d.imag() = s1->imag();
    s1->real() = czero.real(), s1->imag() = czero.imag();
    as1 = (float)0.;
    if(aln < -(*alim))
    {
        goto L10;
    }
    c_log(&q__3, &s1d);
    q__2.real() = q__3.real() - zr->real(), q__2.imag() = q__3.imag() - zr->imag();
    q__1.real() = q__2.real() - zr->real(), q__1.imag() = q__2.imag() - zr->imag();
    c1.real() = q__1.real(), c1.imag() = q__1.imag();
    c_exp(&q__1, &c1);
    s1->real() = q__1.real(), s1->imag() = q__1.imag();
    as1 = c_abs(s1);
    ++(*iuf);
L10:
    aa = max(as1, as2);
    if(aa > *ascle)
    {
        return 0;
    }
    s1->real() = czero.real(), s1->imag() = czero.imag();
    s2->real() = czero.real(), s2->imag() = czero.imag();
    *nz  = 1;
    *iuf = 0;
    return 0;
} /* cs1s2_ */

 int cseri_(System::Complex<float>* z__, float* fnu, int32* kode, int32* n, System::Complex<float>* y, int32* nz, float* tol, float* elim, float* alim)
{
    /* Initialized data */

    static System::Complex<float> czero = {(float)0., (float)0.};
    static System::Complex<float> cone  = {(float)1., (float)0.};

    /* System generated locals */
    int32 i__1, i__2, i__3, i__4;
    float    r__1, r__2;
    System::Complex<float> q__1, q__2, q__3, q__4;

  
    
    int32                     i__, k, l, m;
    float                        s;
    System::Complex<float>                     w[2];
    float                        x;
    System::Complex<float>                     s1, s2;
    float                        aa;
    int32                     ib;
    float                        ak;
    System::Complex<float>                     ck;
    int32                     il;
    float                        az;
    int32                     nn;
    System::Complex<float>                     cz, hz;
    float                        rs, ss;
    int32                     nw;
    System::Complex<float>                     rz, ak1;
    float                        acz, arm, rak1, rtr1;
    System::Complex<float>                     coef, crsc;
    float                        dfnu;
    int32                     idum;
    float                        atol, fnup;
    int32                     iflag;
    float                        ascle;

    /* *********************************************************************72 */

    /* ***BEGIN PROLOGUE  CSERI */
    /* ***REFER TO  CBESI,CBESK */

    /*     CSERI COMPUTES THE I BESSEL FUNCTION FOR REAL(Z).GE.0.0 BY */
    /*     MEANS OF THE POWER SERIES FOR LARGE CABS(Z) IN THE */
    /*     REGION CABS(Z).LE.2*SQRT(FNU+1). NZ=0 IS A NORMAL RETURN. */
    /*     NZ.GT.0 MEANS THAT THE LAST NZ COMPONENTS WERE SET TO ZERO */
    /*     DUE TO UNDERFLOW. NZ.LT.0 MEANS UNDERFLOW OCCURRED, BUT THE */
    /*     CONDITION CABS(Z).LE.2*SQRT(FNU+1) WAS VIOLATED AND THE */
    /*     COMPUTATION MUST BE COMPLETED IN ANOTHER ROUTINE WITH N=N-ABS(NZ). */

    /* ***ROUTINES CALLED  CUCHK,GAMLN,R1MACH */
    /* ***END PROLOGUE  CSERI */
    /* Parameter adjustments */
    --y;

    /* Function Body */

    *nz = 0;
    az  = c_abs(z__);
    if(az == (float)0.)
    {
        goto L150;
    }
    x      = z__->real();
    arm    = r1mach_(&c__1) * (float)1e3;
    rtr1   = sqrt(arm);
    crsc.real() = (float)1., crsc.imag() = (float)0.;
    iflag = 0;
    if(az < arm)
    {
        goto L140;
    }
    q__1.real() = z__->real() * (float).5 - z__->imag() * (float)0., q__1.imag() = z__->real() * (float)0. + z__->imag() * (float).5;
    hz.real() = q__1.real(), hz.imag() = q__1.imag();
    cz.real() = czero.real(), cz.imag() = czero.imag();
    if(az > rtr1)
    {
        q__1.real() = hz.real() * hz.real() - hz.imag() * hz.imag(), q__1.imag() = hz.real() * hz.imag() + hz.imag() * hz.real();
        cz.real() = q__1.real(), cz.imag() = q__1.imag();
    }
    acz = c_abs(&cz);
    nn  = *n;
    c_log(&q__1, &hz);
    ck.real() = q__1.real(), ck.imag() = q__1.imag();
L10:
    dfnu = *fnu + (float)(nn - 1);
    fnup = dfnu + (float)1.;
    /* ----------------------------------------------------------------------- */
    /*     UNDERFLOW TEST */
    /* ----------------------------------------------------------------------- */
    q__2.real() = dfnu, q__2.imag() = (float)0.;
    q__1.real() = ck.real() * q__2.real() - ck.imag() * q__2.imag(), q__1.imag() = ck.real() * q__2.imag() + ck.imag() * q__2.real();
    ak1.real() = q__1.real(), ak1.imag() = q__1.imag();
    ak     = gamln_(&fnup, &idum);
    q__2.real() = ak, q__2.imag() = (float)0.;
    q__1.real() = ak1.real() - q__2.real(), q__1.imag() = ak1.imag() - q__2.imag();
    ak1.real() = q__1.real(), ak1.imag() = q__1.imag();
    if(*kode == 2)
    {
        q__2.real() = x, q__2.imag() = (float)0.;
        q__1.real() = ak1.real() - q__2.real(), q__1.imag() = ak1.imag() - q__2.imag();
        ak1.real() = q__1.real(), ak1.imag() = q__1.imag();
    }
    rak1 = ak1.real();
    if(rak1 > -(*elim))
    {
        goto L30;
    }
L20:
    ++(*nz);
    i__1      = nn;
    y[i__1].real() = czero.real(), y[i__1].imag() = czero.imag();
    if(acz > dfnu)
    {
        goto L170;
    }
    --nn;
    if(nn == 0)
    {
        return 0;
    }
    goto L10;
L30:
    if(rak1 > -(*alim))
    {
        goto L40;
    }
    iflag  = 1;
    ss     = (float)1. / *tol;
    q__1.real() = *tol, q__1.imag() = (float)0.;
    crsc.real() = q__1.real(), crsc.imag() = q__1.imag();
    ascle = arm * ss;
L40:
    ak = r_imag(&ak1);
    aa = exp(rak1);
    if(iflag == 1)
    {
        aa *= ss;
    }
    q__2.real() = aa, q__2.imag() = (float)0.;
    r__1   = cos(ak);
    r__2   = sin(ak);
    q__3.real() = r__1, q__3.imag() = r__2;
    q__1.real() = q__2.real() * q__3.real() - q__2.imag() * q__3.imag(), q__1.imag() = q__2.real() * q__3.imag() + q__2.imag() * q__3.real();
    coef.real() = q__1.real(), coef.imag() = q__1.imag();
    atol = *tol * acz / fnup;
    il   = std::min(2L, nn);
    i__1 = il;
    for(i__ = 1; i__ <= i__1; ++i__)
    {
        dfnu = *fnu + (float)(nn - i__);
        fnup = dfnu + (float)1.;
        s1.real() = cone.real(), s1.imag() = cone.imag();
        if(acz < *tol * fnup)
        {
            goto L60;
        }
        ak1.real() = cone.real(), ak1.imag() = cone.imag();
        ak = fnup + (float)2.;
        s  = fnup;
        aa = (float)2.;
    L50:
        rs     = (float)1. / s;
        q__2.real() = ak1.real() * cz.real() - ak1.imag() * cz.imag(), q__2.imag() = ak1.real() * cz.imag() + ak1.imag() * cz.real();
        q__3.real() = rs, q__3.imag() = (float)0.;
        q__1.real() = q__2.real() * q__3.real() - q__2.imag() * q__3.imag(), q__1.imag() = q__2.real() * q__3.imag() + q__2.imag() * q__3.real();
        ak1.real() = q__1.real(), ak1.imag() = q__1.imag();
        q__1.real() = s1.real() + ak1.real(), q__1.imag() = s1.imag() + ak1.imag();
        s1.real() = q__1.real(), s1.imag() = q__1.imag();
        s += ak;
        ak += (float)2.;
        aa = aa * acz * rs;
        if(aa > atol)
        {
            goto L50;
        }
    L60:
        m      = nn - i__ + 1;
        q__1.real() = s1.real() * coef.real() - s1.imag() * coef.imag(), q__1.imag() = s1.real() * coef.imag() + s1.imag() * coef.real();
        s2.real() = q__1.real(), s2.imag() = q__1.imag();
        i__2      = i__ - 1;
        w[i__2].real() = s2.real(), w[i__2].imag() = s2.imag();
        if(iflag == 0)
        {
            goto L70;
        }
        cuchk_(&s2, &nw, &ascle, tol);
        if(nw != 0)
        {
            goto L20;
        }
    L70:
        i__2   = m;
        q__1.real() = s2.real() * crsc.real() - s2.imag() * crsc.imag(), q__1.imag() = s2.real() * crsc.imag() + s2.imag() * crsc.real();
        y[i__2].real() = q__1.real(), y[i__2].imag() = q__1.imag();
        if(i__ != il)
        {
            q__3.real() = dfnu, q__3.imag() = (float)0.;
            q__2.real() = coef.real() * q__3.real() - coef.imag() * q__3.imag(), q__2.imag() = coef.real() * q__3.imag() + coef.imag() * q__3.real();
            c_div(&q__1, &q__2, &hz);
            coef.real() = q__1.real(), coef.imag() = q__1.imag();
        }
        /* L80: */
    }
    if(nn <= 2)
    {
        return 0;
    }
    k      = nn - 2;
    ak     = (float)k;
    q__2.real() = cone.real() + cone.real(), q__2.imag() = cone.imag() + cone.imag();
    c_div(&q__1, &q__2, z__);
    rz.real() = q__1.real(), rz.imag() = q__1.imag();
    if(iflag == 1)
    {
        goto L110;
    }
    ib = 3;
L90:
    i__1 = nn;
    for(i__ = ib; i__ <= i__1; ++i__)
    {
        i__2   = k;
        r__1   = ak + *fnu;
        q__4.real() = r__1, q__4.imag() = (float)0.;
        q__3.real() = q__4.real() * rz.real() - q__4.imag() * rz.imag(), q__3.imag() = q__4.real() * rz.imag() + q__4.imag() * rz.real();
        i__3   = k + 1;
        q__2.real() = q__3.real() * y[i__3].real() - q__3.imag() * y[i__3].imag(), q__2.imag() = q__3.real() * y[i__3].imag() + q__3.imag() * y[i__3].real();
        i__4   = k + 2;
        q__1.real() = q__2.real() + y[i__4].real(), q__1.imag() = q__2.imag() + y[i__4].imag();
        y[i__2].real() = q__1.real(), y[i__2].imag() = q__1.imag();
        ak += (float)-1.;
        --k;
        /* L100: */
    }
    return 0;
/* ----------------------------------------------------------------------- */
/*     RECUR BACKWARD WITH SCALED VALUES */
/* ----------------------------------------------------------------------- */
L110:
    /* ----------------------------------------------------------------------- */
    /*     EXP(-ALIM)=EXP(-ELIM)/TOL=APPROX. ONE PRECISION ABOVE THE */
    /*     UNDERFLOW LIMIT = ASCLE = R1MACH(1)*CSCL*1.0E+3 */
    /* ----------------------------------------------------------------------- */
    s1.real() = w[0].real(), s1.imag() = w[0].imag();
    s2.real() = w[1].real(), s2.imag() = w[1].imag();
    i__1 = nn;
    for(l = 3; l <= i__1; ++l)
    {
        ck.real() = s2.real(), ck.imag() = s2.imag();
        r__1   = ak + *fnu;
        q__4.real() = r__1, q__4.imag() = (float)0.;
        q__3.real() = q__4.real() * rz.real() - q__4.imag() * rz.imag(), q__3.imag() = q__4.real() * rz.imag() + q__4.imag() * rz.real();
        q__2.real() = q__3.real() * s2.real() - q__3.imag() * s2.imag(), q__2.imag() = q__3.real() * s2.imag() + q__3.imag() * s2.real();
        q__1.real() = s1.real() + q__2.real(), q__1.imag() = s1.imag() + q__2.imag();
        s2.real() = q__1.real(), s2.imag() = q__1.imag();
        s1.real() = ck.real(), s1.imag() = ck.imag();
        q__1.real() = s2.real() * crsc.real() - s2.imag() * crsc.imag(), q__1.imag() = s2.real() * crsc.imag() + s2.imag() * crsc.real();
        ck.real() = q__1.real(), ck.imag() = q__1.imag();
        i__2      = k;
        y[i__2].real() = ck.real(), y[i__2].imag() = ck.imag();
        ak += (float)-1.;
        --k;
        if(c_abs(&ck) > ascle)
        {
            goto L130;
        }
        /* L120: */
    }
    return 0;
L130:
    ib = l + 1;
    if(ib > nn)
    {
        return 0;
    }
    goto L90;
L140:
    *nz = *n;
    if(*fnu == (float)0.)
    {
        --(*nz);
    }
L150:
    y[1].real() = czero.real(), y[1].imag() = czero.imag();
    if(*fnu == (float)0.)
    {
        y[1].real() = cone.real(), y[1].imag() = cone.imag();
    }
    if(*n == 1)
    {
        return 0;
    }
    i__1 = *n;
    for(i__ = 2; i__ <= i__1; ++i__)
    {
        i__2      = i__;
        y[i__2].real() = czero.real(), y[i__2].imag() = czero.imag();
        /* L160: */
    }
    return 0;
/* ----------------------------------------------------------------------- */
/*     RETURN WITH NZ.LT.0 IF CABS(Z*Z/4).GT.FNU+N-NZ-1 COMPLETE */
/*     THE CALCULATION IN CBINU WITH N=N-IABS(NZ) */
/* ----------------------------------------------------------------------- */
L170:
    *nz = -(*nz);
    return 0;
} /* cseri_ */

 int cshch_(System::Complex<float>* z__, System::Complex<float>* csh, System::Complex<float>* cch)
{
    /* System generated locals */
    System::Complex<float> q__1;

 

    
    float x, y, ch, cn, sh, sn, cchi, cchr, cshi, cshr;

    /* *********************************************************************72 */

    /* ***BEGIN PROLOGUE  CSHCH */
    /* ***REFER TO  CBESK,CBESH */

    /*     CSHCH COMPUTES THE COMPLEX HYPERBOLIC FUNCTIONS CSH=SINH(X+I*Y) */
    /*     AND CCH=COSH(X+I*Y), WHERE I**2=-1. */

    /* ***ROUTINES CALLED  (NONE) */
    /* ***END PROLOGUE  CSHCH */
    x      = z__->real();
    y      = r_imag(z__);
    sh     = sinh(x);
    ch     = cosh(x);
    sn     = sin(y);
    cn     = cos(y);
    cshr   = sh * cn;
    cshi   = ch * sn;
    q__1.real() = cshr, q__1.imag() = cshi;
    csh->real() = q__1.real(), csh->imag() = q__1.imag();
    cchr   = ch * cn;
    cchi   = sh * sn;
    q__1.real() = cchr, q__1.imag() = cchi;
    cch->real() = q__1.real(), cch->imag() = q__1.imag();
    return 0;
} /* cshch_ */

 int cuchk_(System::Complex<float>* y, int32* nz, float* ascle, float* tol)
{


    
    float yi, ss, st, yr;

    /* *********************************************************************72 */

    /* ***BEGIN PROLOGUE  CUCHK */
    /* ***REFER TO CSERI,CUOIK,CUNK1,CUNK2,CUNI1,CUNI2,CKSCL */

    /*      Y ENTERS AS A SCALED QUANTITY WHOSE MAGNITUDE IS GREATER THAN */
    /*      EXP(-ALIM)=ASCLE=1.0E+3*R1MACH(1)/TOL. THE TEST IS MADE TO SEE */
    /*      IF THE MAGNITUDE OF THE REAL OR IMAGINARY PART WOULD UNDER FLOW */
    /*      WHEN Y IS SCALED (BY TOL) TO ITS PROPER VALUE. Y IS ACCEPTED */
    /*      IF THE UNDERFLOW IS AT LEAST ONE PRECISION BELOW THE MAGNITUDE */
    /*      OF THE LARGEST COMPONENT; OTHERWISE THE PHASE ANGLE DOES NOT HAVE */
    /*      ABSOLUTE ACCURACY AND AN UNDERFLOW IS ASSUMED. */

    /* ***ROUTINES CALLED  (NONE) */
    /* ***END PROLOGUE  CUCHK */

    *nz = 0;
    yr  = y->real();
    yi  = r_imag(y);
    yr  = abs(yr);
    yi  = abs(yi);
    st  = std::min(yr, yi);
    if(st > *ascle)
    {
        return 0;
    }
    ss = max(yr, yi);
    st /= *tol;
    if(ss < st)
    {
        *nz = 1;
    }
    return 0;
} /* cuchk_ */

 int cunhj_(System::Complex<float>* z__, float* fnu, int32* ipmtr, float* tol, System::Complex<float>* phi, System::Complex<float>* arg, System::Complex<float>* zeta1, System::Complex<float>* zeta2, System::Complex<float>* asum, System::Complex<float>* bsum)
{
    /* Initialized data */

    static float ar[14]    = {(float)1.,
                          (float).104166666666666667,
                          (float).0835503472222222222,
                          (float).12822657455632716,
                          (float).291849026464140464,
                          (float).881627267443757652,
                          (float)3.32140828186276754,
                          (float)14.9957629868625547,
                          (float)78.9230130115865181,
                          (float)474.451538868264323,
                          (float)3207.49009089066193,
                          (float)24086.5496408740049,
                          (float)198923.119169509794,
                          (float)1791902.00777534383};
    static float br[14]    = {(float)1.,
                          (float)-.145833333333333333,
                          (float)-.0987413194444444444,
                          (float)-.143312053915895062,
                          (float)-.317227202678413548,
                          (float)-.942429147957120249,
                          (float)-3.51120304082635426,
                          (float)-15.7272636203680451,
                          (float)-82.2814390971859444,
                          (float)-492.355370523670524,
                          (float)-3316.21856854797251,
                          (float)-24827.6742452085896,
                          (float)-204526.587315129788,
                          (float)-1838444.9170682099};
    static float c__[105]  = {(float)1.,
                            (float)-.208333333333333333,
                            (float).125,
                            (float).334201388888888889,
                            (float)-.401041666666666667,
                            (float).0703125,
                            (float)-1.02581259645061728,
                            (float)1.84646267361111111,
                            (float)-.8912109375,
                            (float).0732421875,
                            (float)4.66958442342624743,
                            (float)-11.2070026162229938,
                            (float)8.78912353515625,
                            (float)-2.3640869140625,
                            (float).112152099609375,
                            (float)-28.2120725582002449,
                            (float)84.6362176746007346,
                            (float)-91.8182415432400174,
                            (float)42.5349987453884549,
                            (float)-7.3687943594796317,
                            (float).227108001708984375,
                            (float)212.570130039217123,
                            (float)-765.252468141181642,
                            (float)1059.99045252799988,
                            (float)-699.579627376132541,
                            (float)218.19051174421159,
                            (float)-26.4914304869515555,
                            (float).572501420974731445,
                            (float)-1919.457662318407,
                            (float)8061.72218173730938,
                            (float)-13586.5500064341374,
                            (float)11655.3933368645332,
                            (float)-5305.64697861340311,
                            (float)1200.90291321635246,
                            (float)-108.090919788394656,
                            (float)1.7277275025844574,
                            (float)20204.2913309661486,
                            (float)-96980.5983886375135,
                            (float)192547.001232531532,
                            (float)-203400.177280415534,
                            (float)122200.46498301746,
                            (float)-41192.6549688975513,
                            (float)7109.51430248936372,
                            (float)-493.915304773088012,
                            (float)6.07404200127348304,
                            (float)-242919.187900551333,
                            (float)1311763.6146629772,
                            (float)-2998015.91853810675,
                            (float)3763271.297656404,
                            (float)-2813563.22658653411,
                            (float)1268365.27332162478,
                            (float)-331645.172484563578,
                            (float)45218.7689813627263,
                            (float)-2499.83048181120962,
                            (float)24.3805296995560639,
                            (float)3284469.85307203782,
                            (float)-19706819.1184322269,
                            (float)50952602.4926646422,
                            (float)-74105148.2115326577,
                            (float)66344512.2747290267,
                            (float)-37567176.6607633513,
                            (float)13288767.1664218183,
                            (float)-2785618.12808645469,
                            (float)308186.404612662398,
                            (float)-13886.0897537170405,
                            (float)110.017140269246738,
                            (float)-49329253.664509962,
                            (float)325573074.185765749,
                            (float)-939462359.681578403,
                            (float)1553596899.57058006,
                            (float)-1621080552.10833708,
                            (float)1106842816.82301447,
                            (float)-495889784.275030309,
                            (float)142062907.797533095,
                            (float)-24474062.7257387285,
                            (float)2243768.17792244943,
                            (float)-84005.4336030240853,
                            (float)551.335896122020586,
                            (float)814789096.118312115,
                            (float)-5866481492.05184723,
                            (float)18688207509.2958249,
                            (float)-34632043388.1587779,
                            (float)41280185579.753974,
                            (float)-33026599749.8007231,
                            (float)17954213731.1556001,
                            (float)-6563293792.61928433,
                            (float)1559279864.87925751,
                            (float)-225105661.889415278,
                            (float)17395107.5539781645,
                            (float)-549842.327572288687,
                            (float)3038.09051092238427,
                            (float)-14679261247.6956167,
                            (float)114498237732.02581,
                            (float)-399096175224.466498,
                            (float)819218669548.577329,
                            (float)-1098375156081.22331,
                            (float)1008158106865.38209,
                            (float)-645364869245.376503,
                            (float)287900649906.150589,
                            (float)-87867072178.0232657,
                            (float)17634730606.8349694,
                            (float)-2167164983.22379509,
                            (float)143157876.718888981,
                            (float)-3871833.44257261262,
                            (float)18257.7554742931747};
    static float alfa[180] = {
        (float)-.00444444444444444444,  (float)-9.22077922077922078e-4, (float)-8.84892884892884893e-5, (float)1.65927687832449737e-4,  (float)2.4669137274179291e-4,
        (float)2.6599558934625478e-4,   (float)2.61824297061500945e-4,  (float)2.48730437344655609e-4,  (float)2.32721040083232098e-4,  (float)2.16362485712365082e-4,
        (float)2.00738858762752355e-4,  (float)1.86267636637545172e-4,  (float)1.73060775917876493e-4,  (float)1.61091705929015752e-4,  (float)1.50274774160908134e-4,
        (float)1.40503497391269794e-4,  (float)1.31668816545922806e-4,  (float)1.23667445598253261e-4,  (float)1.16405271474737902e-4,  (float)1.09798298372713369e-4,
        (float)1.03772410422992823e-4,  (float)9.82626078369363448e-5,  (float)9.32120517249503256e-5,  (float)8.85710852478711718e-5,  (float)8.42963105715700223e-5,
        (float)8.03497548407791151e-5,  (float)7.66981345359207388e-5,  (float)7.33122157481777809e-5,  (float)7.01662625163141333e-5,  (float)6.72375633790160292e-5,
        (float)6.93735541354588974e-4,  (float)2.32241745182921654e-4,  (float)-1.41986273556691197e-5, (float)-1.1644493167204864e-4,  (float)-1.50803558053048762e-4,
        (float)-1.55121924918096223e-4, (float)-1.46809756646465549e-4, (float)-1.33815503867491367e-4, (float)-1.19744975684254051e-4, (float)-1.0618431920797402e-4,
        (float)-9.37699549891194492e-5, (float)-8.26923045588193274e-5, (float)-7.29374348155221211e-5, (float)-6.44042357721016283e-5, (float)-5.69611566009369048e-5,
        (float)-5.04731044303561628e-5, (float)-4.48134868008882786e-5, (float)-3.98688727717598864e-5, (float)-3.55400532972042498e-5, (float)-3.1741425660902248e-5,
        (float)-2.83996793904174811e-5, (float)-2.54522720634870566e-5, (float)-2.28459297164724555e-5, (float)-2.05352753106480604e-5, (float)-1.84816217627666085e-5,
        (float)-1.66519330021393806e-5, (float)-1.50179412980119482e-5, (float)-1.35554031379040526e-5, (float)-1.22434746473858131e-5, (float)-1.10641884811308169e-5,
        (float)-3.54211971457743841e-4, (float)-1.56161263945159416e-4, (float)3.0446550359493641e-5,   (float)1.30198655773242693e-4,  (float)1.67471106699712269e-4,
        (float)1.70222587683592569e-4,  (float)1.56501427608594704e-4,  (float)1.3633917097744512e-4,   (float)1.14886692029825128e-4,  (float)9.45869093034688111e-5,
        (float)7.64498419250898258e-5,  (float)6.07570334965197354e-5,  (float)4.74394299290508799e-5,  (float)3.62757512005344297e-5,  (float)2.69939714979224901e-5,
        (float)1.93210938247939253e-5,  (float)1.30056674793963203e-5,  (float)7.82620866744496661e-6,  (float)3.59257485819351583e-6,  (float)1.44040049814251817e-7,
        (float)-2.65396769697939116e-6, (float)-4.9134686709848591e-6,  (float)-6.72739296091248287e-6, (float)-8.17269379678657923e-6, (float)-9.31304715093561232e-6,
        (float)-1.02011418798016441e-5, (float)-1.0880596251059288e-5,  (float)-1.13875481509603555e-5, (float)-1.17519675674556414e-5, (float)-1.19987364870944141e-5,
        (float)3.78194199201772914e-4,  (float)2.02471952761816167e-4,  (float)-6.37938506318862408e-5, (float)-2.38598230603005903e-4, (float)-3.10916256027361568e-4,
        (float)-3.13680115247576316e-4, (float)-2.78950273791323387e-4, (float)-2.28564082619141374e-4, (float)-1.75245280340846749e-4, (float)-1.25544063060690348e-4,
        (float)-8.22982872820208365e-5, (float)-4.62860730588116458e-5, (float)-1.72334302366962267e-5, (float)5.60690482304602267e-6,  (float)2.313954431482868e-5,
        (float)3.62642745856793957e-5,  (float)4.58006124490188752e-5,  (float)5.2459529495911405e-5,   (float)5.68396208545815266e-5,  (float)5.94349820393104052e-5,
        (float)6.06478527578421742e-5,  (float)6.08023907788436497e-5,  (float)6.01577894539460388e-5,  (float)5.891996573446985e-5,    (float)5.72515823777593053e-5,
        (float)5.52804375585852577e-5,  (float)5.3106377380288017e-5,   (float)5.08069302012325706e-5,  (float)4.84418647620094842e-5,  (float)4.6056858160747537e-5,
        (float)-6.91141397288294174e-4, (float)-4.29976633058871912e-4, (float)1.83067735980039018e-4,  (float)6.60088147542014144e-4,  (float)8.75964969951185931e-4,
        (float)8.77335235958235514e-4,  (float)7.49369585378990637e-4,  (float)5.63832329756980918e-4,  (float)3.68059319971443156e-4,  (float)1.88464535514455599e-4,
        (float)3.70663057664904149e-5,  (float)-8.28520220232137023e-5, (float)-1.72751952869172998e-4, (float)-2.36314873605872983e-4, (float)-2.77966150694906658e-4,
        (float)-3.02079514155456919e-4, (float)-3.12594712643820127e-4, (float)-3.12872558758067163e-4, (float)-3.05678038466324377e-4, (float)-2.93226470614557331e-4,
        (float)-2.77255655582934777e-4, (float)-2.59103928467031709e-4, (float)-2.39784014396480342e-4, (float)-2.20048260045422848e-4, (float)-2.00443911094971498e-4,
        (float)-1.81358692210970687e-4, (float)-1.63057674478657464e-4, (float)-1.45712672175205844e-4, (float)-1.29425421983924587e-4, (float)-1.14245691942445952e-4,
        (float).00192821964248775885,   (float).00135592576302022234,   (float)-7.17858090421302995e-4, (float)-.00258084802575270346,  (float)-.00349271130826168475,
        (float)-.00346986299340960628,  (float)-.00282285233351310182,  (float)-.00188103076404891354,  (float)-8.895317183839476e-4,   (float)3.87912102631035228e-6,
        (float)7.28688540119691412e-4,  (float).00126566373053457758,   (float).00162518158372674427,   (float).00183203153216373172,   (float).00191588388990527909,
        (float).00190588846755546138,   (float).00182798982421825727,   (float).0017038950642112153,    (float).00155097127171097686,   (float).00138261421852276159,
        (float).00120881424230064774,   (float).00103676532638344962,   (float)8.71437918068619115e-4,  (float)7.16080155297701002e-4,  (float)5.72637002558129372e-4,
        (float)4.42089819465802277e-4,  (float)3.24724948503090564e-4,  (float)2.20342042730246599e-4,  (float)1.28412898401353882e-4,  (float)4.82005924552095464e-5};
    static float beta[210] = {
        (float).0179988721413553309,    (float).00559964911064388073,   (float).00288501402231132779,   (float).00180096606761053941,   (float).00124753110589199202,
        (float)9.22878876572938311e-4,  (float)7.14430421727287357e-4,  (float)5.71787281789704872e-4,  (float)4.69431007606481533e-4,  (float)3.93232835462916638e-4,
        (float)3.34818889318297664e-4,  (float)2.88952148495751517e-4,  (float)2.52211615549573284e-4,  (float)2.22280580798883327e-4,  (float)1.97541838033062524e-4,
        (float)1.76836855019718004e-4,  (float)1.59316899661821081e-4,  (float)1.44347930197333986e-4,  (float)1.31448068119965379e-4,  (float)1.20245444949302884e-4,
        (float)1.10449144504599392e-4,  (float)1.01828770740567258e-4,  (float)9.41998224204237509e-5,  (float)8.74130545753834437e-5,  (float)8.13466262162801467e-5,
        (float)7.59002269646219339e-5,  (float)7.09906300634153481e-5,  (float)6.65482874842468183e-5,  (float)6.25146958969275078e-5,  (float)5.88403394426251749e-5,
        (float)-.00149282953213429172,  (float)-8.78204709546389328e-4, (float)-5.02916549572034614e-4, (float)-2.94822138512746025e-4, (float)-1.75463996970782828e-4,
        (float)-1.04008550460816434e-4, (float)-5.96141953046457895e-5, (float)-3.1203892907609834e-5,  (float)-1.26089735980230047e-5, (float)-2.42892608575730389e-7,
        (float)8.05996165414273571e-6,  (float)1.36507009262147391e-5,  (float)1.73964125472926261e-5,  (float)1.9867297884213378e-5,   (float)2.14463263790822639e-5,
        (float)2.23954659232456514e-5,  (float)2.28967783814712629e-5,  (float)2.30785389811177817e-5,  (float)2.30321976080909144e-5,  (float)2.28236073720348722e-5,
        (float)2.25005881105292418e-5,  (float)2.20981015361991429e-5,  (float)2.16418427448103905e-5,  (float)2.11507649256220843e-5,  (float)2.06388749782170737e-5,
        (float)2.01165241997081666e-5,  (float)1.95913450141179244e-5,  (float)1.9068936791043674e-5,   (float)1.85533719641636667e-5,  (float)1.80475722259674218e-5,
        (float)5.5221307672129279e-4,   (float)4.47932581552384646e-4,  (float)2.79520653992020589e-4,  (float)1.52468156198446602e-4,  (float)6.93271105657043598e-5,
        (float)1.76258683069991397e-5,  (float)-1.35744996343269136e-5, (float)-3.17972413350427135e-5, (float)-4.18861861696693365e-5, (float)-4.69004889379141029e-5,
        (float)-4.87665447413787352e-5, (float)-4.87010031186735069e-5, (float)-4.74755620890086638e-5, (float)-4.55813058138628452e-5, (float)-4.33309644511266036e-5,
        (float)-4.09230193157750364e-5, (float)-3.84822638603221274e-5, (float)-3.60857167535410501e-5, (float)-3.37793306123367417e-5, (float)-3.15888560772109621e-5,
        (float)-2.95269561750807315e-5, (float)-2.75978914828335759e-5, (float)-2.58006174666883713e-5, (float)-2.413083567612802e-5,   (float)-2.25823509518346033e-5,
        (float)-2.11479656768912971e-5, (float)-1.98200638885294927e-5, (float)-1.85909870801065077e-5, (float)-1.74532699844210224e-5, (float)-1.63997823854497997e-5,
        (float)-4.74617796559959808e-4, (float)-4.77864567147321487e-4, (float)-3.20390228067037603e-4, (float)-1.61105016119962282e-4, (float)-4.25778101285435204e-5,
        (float)3.44571294294967503e-5,  (float)7.97092684075674924e-5,  (float)1.031382367082722e-4,    (float)1.12466775262204158e-4,  (float)1.13103642108481389e-4,
        (float)1.08651634848774268e-4,  (float)1.01437951597661973e-4,  (float)9.29298396593363896e-5,  (float)8.40293133016089978e-5,  (float)7.52727991349134062e-5,
        (float)6.69632521975730872e-5,  (float)5.92564547323194704e-5,  (float)5.22169308826975567e-5,  (float)4.58539485165360646e-5,  (float)4.01445513891486808e-5,
        (float)3.50481730031328081e-5,  (float)3.05157995034346659e-5,  (float)2.64956119950516039e-5,  (float)2.29363633690998152e-5,  (float)1.97893056664021636e-5,
        (float)1.70091984636412623e-5,  (float)1.45547428261524004e-5,  (float)1.23886640995878413e-5,  (float)1.04775876076583236e-5,  (float)8.79179954978479373e-6,
        (float)7.36465810572578444e-4,  (float)8.72790805146193976e-4,  (float)6.22614862573135066e-4,  (float)2.85998154194304147e-4,  (float)3.84737672879366102e-6,
        (float)-1.87906003636971558e-4, (float)-2.97603646594554535e-4, (float)-3.45998126832656348e-4, (float)-3.53382470916037712e-4, (float)-3.35715635775048757e-4,
        (float)-3.04321124789039809e-4, (float)-2.66722723047612821e-4, (float)-2.27654214122819527e-4, (float)-1.89922611854562356e-4, (float)-1.5505891859909387e-4,
        (float)-1.2377824076187363e-4,  (float)-9.62926147717644187e-5, (float)-7.25178327714425337e-5, (float)-5.22070028895633801e-5, (float)-3.50347750511900522e-5,
        (float)-2.06489761035551757e-5, (float)-8.70106096849767054e-6, (float)1.1369868667510029e-6,   (float)9.16426474122778849e-6,  (float)1.5647778542887262e-5,
        (float)2.08223629482466847e-5,  (float)2.48923381004595156e-5,  (float)2.80340509574146325e-5,  (float)3.03987774629861915e-5,  (float)3.21156731406700616e-5,
        (float)-.00180182191963885708,  (float)-.00243402962938042533,  (float)-.00183422663549856802,  (float)-7.62204596354009765e-4, (float)2.39079475256927218e-4,
        (float)9.49266117176881141e-4,  (float).00134467449701540359,   (float).00148457495259449178,   (float).00144732339830617591,   (float).00130268261285657186,
        (float).00110351597375642682,   (float)8.86047440419791759e-4,  (float)6.73073208165665473e-4,  (float)4.77603872856582378e-4,  (float)3.05991926358789362e-4,
        (float)1.6031569459472163e-4,   (float)4.00749555270613286e-5,  (float)-5.66607461635251611e-5, (float)-1.32506186772982638e-4, (float)-1.90296187989614057e-4,
        (float)-2.32811450376937408e-4, (float)-2.62628811464668841e-4, (float)-2.82050469867598672e-4, (float)-2.93081563192861167e-4, (float)-2.97435962176316616e-4,
        (float)-2.96557334239348078e-4, (float)-2.91647363312090861e-4, (float)-2.83696203837734166e-4, (float)-2.73512317095673346e-4, (float)-2.6175015580676858e-4,
        (float).00638585891212050914,   (float).00962374215806377941,   (float).00761878061207001043,   (float).00283219055545628054,   (float)-.0020984135201272009,
        (float)-.00573826764216626498,  (float)-.0077080424449541462,   (float)-.00821011692264844401,  (float)-.00765824520346905413,  (float)-.00647209729391045177,
        (float)-.00499132412004966473,  (float)-.0034561228971313328,   (float)-.00201785580014170775,  (float)-7.59430686781961401e-4, (float)2.84173631523859138e-4,
        (float).00110891667586337403,   (float).00172901493872728771,   (float).00216812590802684701,   (float).00245357710494539735,   (float).00261281821058334862,
        (float).00267141039656276912,   (float).0026520307339598043,    (float).00257411652877287315,   (float).00245389126236094427,   (float).00230460058071795494,
        (float).00213684837686712662,   (float).00195896528478870911,   (float).00177737008679454412,   (float).00159690280765839059,   (float).00142111975664438546};
    static float    gama[30] = {(float).629960524947436582,  (float).251984209978974633,  (float).154790300415655846,  (float).110713062416159013,  (float).0857309395527394825,
                            (float).0697161316958684292, (float).0586085671893713576, (float).0504698873536310685, (float).0442600580689154809, (float).0393720661543509966,
                            (float).0354283195924455368, (float).0321818857502098231, (float).0294646240791157679, (float).0271581677112934479, (float).0251768272973861779,
                            (float).0234570755306078891, (float).0219508390134907203, (float).020621082823564624,  (float).0194388240897880846, (float).0183810633800683158,
                            (float).0174293213231963172, (float).0165685837786612353, (float).0157865285987918445, (float).0150729501494095594, (float).0144193250839954639,
                            (float).0138184805735341786, (float).0132643378994276568, (float).0127517121970498651, (float).0122761545318762767, (float).0118338262398482403};
    static float    ex1      = (float).333333333333333333;
    static float    ex2      = (float).666666666666666667;
    static float    hpi      = (float)1.57079632679489662;
    static float    pi       = (float)3.14159265358979324;
    static float    thpi     = (float)4.71238898038468986;
    static System::Complex<float> czero    = {(float)0., (float)0.};
    static System::Complex<float> cone     = {(float)1., (float)0.};

    /* System generated locals */
    int32    i__1, i__2, i__3;
    float       r__1;
    double d__1, d__2;
    System::Complex<float>    q__1, q__2, q__3, q__4, q__5;



    
    int32           j, k, l, m;
    System::Complex<float>           p[30], w;
    int32           l1, l2;
    System::Complex<float>           t2, w2;
    float              ac, ap[30];
    System::Complex<float>           cr[14], dr[14], za, zb, zc;
    int32           is, jr;
    float              pp, wi;
    int32           ju, ks, lr;
    System::Complex<float>           up[14];
    float              wr, aw2;
    int32           kp1;
    float              ang, fn13, fn23;
    int32           ias, ibs;
    float              zci;
    System::Complex<float>           tfn;
    float              zcr;
    System::Complex<float>           zth;
    int32           lrp1;
    System::Complex<float>           rfn13, cfnu;
    float              atol, btol;
    int32           kmax;
    System::Complex<float>           zeta, ptfn, suma, sumb;
    float              azth, rfnu, zthi, test, tsti;
    System::Complex<float>           rzth;
    float              zthr, tstr, rfnu2, zetai, asumi, bsumi, zetar, asumr, bsumr;
    System::Complex<float>           rtzta, przth;
    

    /* *********************************************************************72 */

    /* ***BEGIN PROLOGUE  CUNHJ */
    /* ***REFER TO  CBESI,CBESK */

    /*     REFERENCES */
    /*         HANDBOOK OF MATHEMATICAL FUNCTIONS BY M. ABRAMOWITZ AND I.A. */
    /*         STEGUN, AMS55, NATIONAL BUREAU OF STANDARDS, 1965, CHAPTER 9. */

    /*         ASYMPTOTICS AND SPECIAL FUNCTIONS BY F.W.J. OLVER, ACADEMIC */
    /*         PRESS, N.Y., 1974, PAGE 420 */

    /*     ABSTRACT */
    /*         CUNHJ COMPUTES PARAMETERS FOR BESSEL FUNCTIONS C(FNU,Z) = */
    /*         J(FNU,Z), Y(FNU,Z) OR H(I,FNU,Z) I=1,2 FOR LARGE ORDERS FNU */
    /*         BY MEANS OF THE UNIFORM ASYMPTOTIC EXPANSION */

    /*         C(FNU,Z)=C1*PHI*( ASUM*AIRY(ARG) + C2*BSUM*DAIRY(ARG) ) */

    /*         FOR PROPER CHOICES OF C1, C2, AIRY AND DAIRY WHERE AIRY IS */
    /*         AN AIRY FUNCTION AND DAIRY IS ITS DERIVATIVE. */

    /*               (2/3)*FNU*ZETA**1.5 = ZETA1-ZETA2, */

    /*         ZETA1=0.5*FNU*CLOG((1+W)/(1-W)), ZETA2=FNU*W FOR SCALING */
    /*         PURPOSES IN AIRY FUNCTIONS FROM CAIRY OR CBIRY. */

    /*         MCONJ=SIGN OF AIMAG(Z), BUT IS AMBIGUOUS WHEN Z IS REAL AND */
    /*         MUST BE SPECIFIED. IPMTR=0 RETURNS ALL PARAMETERS. IPMTR= */
    /*         1 COMPUTES ALL EXCEPT ASUM AND BSUM. */

    /* ***ROUTINES CALLED  (NONE) */
    /* ***END PROLOGUE  CUNHJ */

    rfnu = (float)1. / *fnu;
    /*     ZB = Z*CMPLX(RFNU,0.0E0) */
    /* ----------------------------------------------------------------------- */
    /*     OVERFLOW TEST (Z/FNU TOO SMALL) */
    /* ----------------------------------------------------------------------- */
    tstr = z__->real();
    tsti = r_imag(z__);
    test = r1mach_(&c__1) * (float)1e3;
    ac   = *fnu * test;
    if(abs(tstr) > ac || abs(tsti) > ac)
    {
        goto L15;
    }
    ac     = (r__1 = log(test), abs(r__1)) * (float)2. + *fnu;
    q__1.real() = ac, q__1.imag() = (float)0.;
    zeta1->real() = q__1.real(), zeta1->imag() = q__1.imag();
    q__1.real() = *fnu, q__1.imag() = (float)0.;
    zeta2->real() = q__1.real(), zeta2->imag() = q__1.imag();
    phi->real() = cone.real(), phi->imag() = cone.imag();
    arg->real() = cone.real(), arg->imag() = cone.imag();
    return 0;
L15:
    q__2.real() = rfnu, q__2.imag() = (float)0.;
    q__1.real() = z__->real() * q__2.real() - z__->imag() * q__2.imag(), q__1.imag() = z__->real() * q__2.imag() + z__->imag() * q__2.real();
    zb.real() = q__1.real(), zb.imag() = q__1.imag();
    rfnu2 = rfnu * rfnu;
    /* ----------------------------------------------------------------------- */
    /*     COMPUTE IN THE FOURTH QUADRANT */
    /* ----------------------------------------------------------------------- */
    d__1   = (double)(*fnu);
    d__2   = (double)ex1;
    fn13   = pow_dd(&d__1, &d__2);
    fn23   = fn13 * fn13;
    r__1   = (float)1. / fn13;
    q__1.real() = r__1, q__1.imag() = (float)0.;
    rfn13.real() = q__1.real(), rfn13.imag() = q__1.imag();
    q__2.real() = zb.real() * zb.real() - zb.imag() * zb.imag(), q__2.imag() = zb.real() * zb.imag() + zb.imag() * zb.real();
    q__1.real() = cone.real() - q__2.real(), q__1.imag() = cone.imag() - q__2.imag();
    w2.real() = q__1.real(), w2.imag() = q__1.imag();
    aw2 = c_abs(&w2);
    if(aw2 > (float).25)
    {
        goto L130;
    }
    /* ----------------------------------------------------------------------- */
    /*     POWER SERIES FOR CABS(W2).LE.0.25E0 */
    /* ----------------------------------------------------------------------- */
    k      = 1;
    p[0].real() = cone.real(), p[0].imag() = cone.imag();
    q__1.real() = gama[0], q__1.imag() = (float)0.;
    suma.real() = q__1.real(), suma.imag() = q__1.imag();
    ap[0] = (float)1.;
    if(aw2 < *tol)
    {
        goto L20;
    }
    for(k = 2; k <= 30; ++k)
    {
        i__1   = k - 1;
        i__2   = k - 2;
        q__1.real() = p[i__2].real() * w2.real() - p[i__2].imag() * w2.imag(), q__1.imag() = p[i__2].real() * w2.imag() + p[i__2].imag() * w2.real();
        p[i__1].real() = q__1.real(), p[i__1].imag() = q__1.imag();
        i__1   = k - 1;
        i__2   = k - 1;
        q__3.real() = gama[i__2], q__3.imag() = (float)0.;
        q__2.real() = p[i__1].real() * q__3.real() - p[i__1].imag() * q__3.imag(), q__2.imag() = p[i__1].real() * q__3.imag() + p[i__1].imag() * q__3.real();
        q__1.real() = suma.real() + q__2.real(), q__1.imag() = suma.imag() + q__2.imag();
        suma.real() = q__1.real(), suma.imag() = q__1.imag();
        ap[k - 1] = ap[k - 2] * aw2;
        if(ap[k - 1] < *tol)
        {
            goto L20;
        }
        /* L10: */
    }
    k = 30;
L20:
    kmax   = k;
    q__1.real() = w2.real() * suma.real() - w2.imag() * suma.imag(), q__1.imag() = w2.real() * suma.imag() + w2.imag() * suma.real();
    zeta.real() = q__1.real(), zeta.imag() = q__1.imag();
    q__2.real() = fn23, q__2.imag() = (float)0.;
    q__1.real() = zeta.real() * q__2.real() - zeta.imag() * q__2.imag(), q__1.imag() = zeta.real() * q__2.imag() + zeta.imag() * q__2.real();
    arg->real() = q__1.real(), arg->imag() = q__1.imag();
    c_sqrt(&q__1, &suma);
    za.real() = q__1.real(), za.imag() = q__1.imag();
    c_sqrt(&q__2, &w2);
    q__3.real() = *fnu, q__3.imag() = (float)0.;
    q__1.real() = q__2.real() * q__3.real() - q__2.imag() * q__3.imag(), q__1.imag() = q__2.real() * q__3.imag() + q__2.imag() * q__3.real();
    zeta2->real() = q__1.real(), zeta2->imag() = q__1.imag();
    q__4.real() = zeta.real() * za.real() - zeta.imag() * za.imag(), q__4.imag() = zeta.real() * za.imag() + zeta.imag() * za.real();
    q__5.real() = ex2, q__5.imag() = (float)0.;
    q__3.real() = q__4.real() * q__5.real() - q__4.imag() * q__5.imag(), q__3.imag() = q__4.real() * q__5.imag() + q__4.imag() * q__5.real();
    q__2.real() = cone.real() + q__3.real(), q__2.imag() = cone.imag() + q__3.imag();
    q__1.real() = zeta2->real() * q__2.real() - zeta2->imag() * q__2.imag(), q__1.imag() = zeta2->real() * q__2.imag() + zeta2->imag() * q__2.real();
    zeta1->real() = q__1.real(), zeta1->imag() = q__1.imag();
    q__1.real() = za.real() + za.real(), q__1.imag() = za.imag() + za.imag();
    za.real() = q__1.real(), za.imag() = q__1.imag();
    c_sqrt(&q__2, &za);
    q__1.real() = q__2.real() * rfn13.real() - q__2.imag() * rfn13.imag(), q__1.imag() = q__2.real() * rfn13.imag() + q__2.imag() * rfn13.real();
    phi->real() = q__1.real(), phi->imag() = q__1.imag();
    if(*ipmtr == 1)
    {
        goto L120;
    }
    /* ----------------------------------------------------------------------- */
    /*     SUM SERIES FOR ASUM AND BSUM */
    /* ----------------------------------------------------------------------- */
    sumb.real() = czero.real(), sumb.imag() = czero.imag();
    i__1 = kmax;
    for(k = 1; k <= i__1; ++k)
    {
        i__2   = k - 1;
        i__3   = k - 1;
        q__3.real() = beta[i__3], q__3.imag() = (float)0.;
        q__2.real() = p[i__2].real() * q__3.real() - p[i__2].imag() * q__3.imag(), q__2.imag() = p[i__2].real() * q__3.imag() + p[i__2].imag() * q__3.real();
        q__1.real() = sumb.real() + q__2.real(), q__1.imag() = sumb.imag() + q__2.imag();
        sumb.real() = q__1.real(), sumb.imag() = q__1.imag();
        /* L30: */
    }
    asum->real() = czero.real(), asum->imag() = czero.imag();
    bsum->real() = sumb.real(), bsum->imag() = sumb.imag();
    l1   = 0;
    l2   = 30;
    btol = *tol * c_abs(bsum);
    atol = *tol;
    pp   = (float)1.;
    ias  = 0;
    ibs  = 0;
    if(rfnu2 < *tol)
    {
        goto L110;
    }
    for(is = 2; is <= 7; ++is)
    {
        atol /= rfnu2;
        pp *= rfnu2;
        if(ias == 1)
        {
            goto L60;
        }
        suma.real() = czero.real(), suma.imag() = czero.imag();
        i__1 = kmax;
        for(k = 1; k <= i__1; ++k)
        {
            m      = l1 + k;
            i__2   = k - 1;
            i__3   = m - 1;
            q__3.real() = alfa[i__3], q__3.imag() = (float)0.;
            q__2.real() = p[i__2].real() * q__3.real() - p[i__2].imag() * q__3.imag(), q__2.imag() = p[i__2].real() * q__3.imag() + p[i__2].imag() * q__3.real();
            q__1.real() = suma.real() + q__2.real(), q__1.imag() = suma.imag() + q__2.imag();
            suma.real() = q__1.real(), suma.imag() = q__1.imag();
            if(ap[k - 1] < atol)
            {
                goto L50;
            }
            /* L40: */
        }
    L50:
        q__3.real() = pp, q__3.imag() = (float)0.;
        q__2.real() = suma.real() * q__3.real() - suma.imag() * q__3.imag(), q__2.imag() = suma.real() * q__3.imag() + suma.imag() * q__3.real();
        q__1.real() = asum->real() + q__2.real(), q__1.imag() = asum->imag() + q__2.imag();
        asum->real() = q__1.real(), asum->imag() = q__1.imag();
        if(pp < *tol)
        {
            ias = 1;
        }
    L60:
        if(ibs == 1)
        {
            goto L90;
        }
        sumb.real() = czero.real(), sumb.imag() = czero.imag();
        i__1 = kmax;
        for(k = 1; k <= i__1; ++k)
        {
            m      = l2 + k;
            i__2   = k - 1;
            i__3   = m - 1;
            q__3.real() = beta[i__3], q__3.imag() = (float)0.;
            q__2.real() = p[i__2].real() * q__3.real() - p[i__2].imag() * q__3.imag(), q__2.imag() = p[i__2].real() * q__3.imag() + p[i__2].imag() * q__3.real();
            q__1.real() = sumb.real() + q__2.real(), q__1.imag() = sumb.imag() + q__2.imag();
            sumb.real() = q__1.real(), sumb.imag() = q__1.imag();
            if(ap[k - 1] < atol)
            {
                goto L80;
            }
            /* L70: */
        }
    L80:
        q__3.real() = pp, q__3.imag() = (float)0.;
        q__2.real() = sumb.real() * q__3.real() - sumb.imag() * q__3.imag(), q__2.imag() = sumb.real() * q__3.imag() + sumb.imag() * q__3.real();
        q__1.real() = bsum->real() + q__2.real(), q__1.imag() = bsum->imag() + q__2.imag();
        bsum->real() = q__1.real(), bsum->imag() = q__1.imag();
        if(pp < btol)
        {
            ibs = 1;
        }
    L90:
        if(ias == 1 && ibs == 1)
        {
            goto L110;
        }
        l1 += 30;
        l2 += 30;
        /* L100: */
    }
L110:
    q__1.real() = asum->real() + cone.real(), q__1.imag() = asum->imag() + cone.imag();
    asum->real() = q__1.real(), asum->imag() = q__1.imag();
    pp     = rfnu * rfn13.real();
    q__2.real() = pp, q__2.imag() = (float)0.;
    q__1.real() = bsum->real() * q__2.real() - bsum->imag() * q__2.imag(), q__1.imag() = bsum->real() * q__2.imag() + bsum->imag() * q__2.real();
    bsum->real() = q__1.real(), bsum->imag() = q__1.imag();
L120:
    return 0;
/* ----------------------------------------------------------------------- */
/*     CABS(W2).GT.0.25E0 */
/* ----------------------------------------------------------------------- */
L130:
    c_sqrt(&q__1, &w2);
    w.real() = q__1.real(), w.imag() = q__1.imag();
    wr = w.real();
    wi = r_imag(&w);
    if(wr < (float)0.)
    {
        wr = (float)0.;
    }
    if(wi < (float)0.)
    {
        wi = (float)0.;
    }
    q__1.real() = wr, q__1.imag() = wi;
    w.real() = q__1.real(), w.imag() = q__1.imag();
    q__2.real() = cone.real() + w.real(), q__2.imag() = cone.imag() + w.imag();
    c_div(&q__1, &q__2, &zb);
    za.real() = q__1.real(), za.imag() = q__1.imag();
    c_log(&q__1, &za);
    zc.real() = q__1.real(), zc.imag() = q__1.imag();
    zcr = zc.real();
    zci = r_imag(&zc);
    if(zci < (float)0.)
    {
        zci = (float)0.;
    }
    if(zci > hpi)
    {
        zci = hpi;
    }
    if(zcr < (float)0.)
    {
        zcr = (float)0.;
    }
    q__1.real() = zcr, q__1.imag() = zci;
    zc.real() = q__1.real(), zc.imag() = q__1.imag();
    q__2.real() = zc.real() - w.real(), q__2.imag() = zc.imag() - w.imag();
    q__1.real() = q__2.real() * (float)1.5 - q__2.imag() * (float)0., q__1.imag() = q__2.real() * (float)0. + q__2.imag() * (float)1.5;
    zth.real() = q__1.real(), zth.imag() = q__1.imag();
    q__1.real() = *fnu, q__1.imag() = (float)0.;
    cfnu.real() = q__1.real(), cfnu.imag() = q__1.imag();
    q__1.real() = zc.real() * cfnu.real() - zc.imag() * cfnu.imag(), q__1.imag() = zc.real() * cfnu.imag() + zc.imag() * cfnu.real();
    zeta1->real() = q__1.real(), zeta1->imag() = q__1.imag();
    q__1.real() = w.real() * cfnu.real() - w.imag() * cfnu.imag(), q__1.imag() = w.real() * cfnu.imag() + w.imag() * cfnu.real();
    zeta2->real() = q__1.real(), zeta2->imag() = q__1.imag();
    azth = c_abs(&zth);
    zthr = zth.real();
    zthi = r_imag(&zth);
    ang  = thpi;
    if(zthr >= (float)0. && zthi < (float)0.)
    {
        goto L140;
    }
    ang = hpi;
    if(zthr == (float)0.)
    {
        goto L140;
    }
    ang = atan(zthi / zthr);
    if(zthr < (float)0.)
    {
        ang += pi;
    }
L140:
    d__1 = (double)azth;
    d__2 = (double)ex2;
    pp   = pow_dd(&d__1, &d__2);
    ang *= ex2;
    zetar = pp * cos(ang);
    zetai = pp * sin(ang);
    if(zetai < (float)0.)
    {
        zetai = (float)0.;
    }
    q__1.real() = zetar, q__1.imag() = zetai;
    zeta.real() = q__1.real(), zeta.imag() = q__1.imag();
    q__2.real() = fn23, q__2.imag() = (float)0.;
    q__1.real() = zeta.real() * q__2.real() - zeta.imag() * q__2.imag(), q__1.imag() = zeta.real() * q__2.imag() + zeta.imag() * q__2.real();
    arg->real() = q__1.real(), arg->imag() = q__1.imag();
    c_div(&q__1, &zth, &zeta);
    rtzta.real() = q__1.real(), rtzta.imag() = q__1.imag();
    c_div(&q__1, &rtzta, &w);
    za.real() = q__1.real(), za.imag() = q__1.imag();
    q__3.real() = za.real() + za.real(), q__3.imag() = za.imag() + za.imag();
    c_sqrt(&q__2, &q__3);
    q__1.real() = q__2.real() * rfn13.real() - q__2.imag() * rfn13.imag(), q__1.imag() = q__2.real() * rfn13.imag() + q__2.imag() * rfn13.real();
    phi->real() = q__1.real(), phi->imag() = q__1.imag();
    if(*ipmtr == 1)
    {
        goto L120;
    }
    q__2.real() = rfnu, q__2.imag() = (float)0.;
    c_div(&q__1, &q__2, &w);
    tfn.real() = q__1.real(), tfn.imag() = q__1.imag();
    q__2.real() = rfnu, q__2.imag() = (float)0.;
    c_div(&q__1, &q__2, &zth);
    rzth.real() = q__1.real(), rzth.imag() = q__1.imag();
    q__2.real() = ar[1], q__2.imag() = (float)0.;
    q__1.real() = rzth.real() * q__2.real() - rzth.imag() * q__2.imag(), q__1.imag() = rzth.real() * q__2.imag() + rzth.imag() * q__2.real();
    zc.real() = q__1.real(), zc.imag() = q__1.imag();
    c_div(&q__1, &cone, &w2);
    t2.real() = q__1.real(), t2.imag() = q__1.imag();
    q__4.real() = c__[1], q__4.imag() = (float)0.;
    q__3.real() = t2.real() * q__4.real() - t2.imag() * q__4.imag(), q__3.imag() = t2.real() * q__4.imag() + t2.imag() * q__4.real();
    q__5.real() = c__[2], q__5.imag() = (float)0.;
    q__2.real() = q__3.real() + q__5.real(), q__2.imag() = q__3.imag() + q__5.imag();
    q__1.real() = q__2.real() * tfn.real() - q__2.imag() * tfn.imag(), q__1.imag() = q__2.real() * tfn.imag() + q__2.imag() * tfn.real();
    up[1].real() = q__1.real(), up[1].imag() = q__1.imag();
    q__1.real() = up[1].real() + zc.real(), q__1.imag() = up[1].imag() + zc.imag();
    bsum->real() = q__1.real(), bsum->imag() = q__1.imag();
    asum->real() = czero.real(), asum->imag() = czero.imag();
    if(rfnu < *tol)
    {
        goto L220;
    }
    przth.real() = rzth.real(), przth.imag() = rzth.imag();
    ptfn.real() = tfn.real(), ptfn.imag() = tfn.imag();
    up[0].real() = cone.real(), up[0].imag() = cone.imag();
    pp    = (float)1.;
    bsumr = bsum->real();
    bsumi = r_imag(bsum);
    btol  = *tol * (abs(bsumr) + abs(bsumi));
    ks    = 0;
    kp1   = 2;
    l     = 3;
    ias   = 0;
    ibs   = 0;
    for(lr = 2; lr <= 12; lr += 2)
    {
        lrp1 = lr + 1;
        /* ----------------------------------------------------------------------- */
        /*     COMPUTE TWO ADDITIONAL CR, DR, AND UP FOR TWO MORE TERMS IN */
        /*     NEXT SUMA AND SUMB */
        /* ----------------------------------------------------------------------- */
        i__1 = lrp1;
        for(k = lr; k <= i__1; ++k)
        {
            ++ks;
            ++kp1;
            ++l;
            i__2   = l - 1;
            q__1.real() = c__[i__2], q__1.imag() = (float)0.;
            za.real() = q__1.real(), za.imag() = q__1.imag();
            i__2 = kp1;
            for(j = 2; j <= i__2; ++j)
            {
                ++l;
                q__2.real() = za.real() * t2.real() - za.imag() * t2.imag(), q__2.imag() = za.real() * t2.imag() + za.imag() * t2.real();
                i__3   = l - 1;
                q__3.real() = c__[i__3], q__3.imag() = (float)0.;
                q__1.real() = q__2.real() + q__3.real(), q__1.imag() = q__2.imag() + q__3.imag();
                za.real() = q__1.real(), za.imag() = q__1.imag();
                /* L150: */
            }
            q__1.real() = ptfn.real() * tfn.real() - ptfn.imag() * tfn.imag(), q__1.imag() = ptfn.real() * tfn.imag() + ptfn.imag() * tfn.real();
            ptfn.real() = q__1.real(), ptfn.imag() = q__1.imag();
            i__2   = kp1 - 1;
            q__1.real() = ptfn.real() * za.real() - ptfn.imag() * za.imag(), q__1.imag() = ptfn.real() * za.imag() + ptfn.imag() * za.real();
            up[i__2].real() = q__1.real(), up[i__2].imag() = q__1.imag();
            i__2   = ks - 1;
            i__3   = ks;
            q__2.real() = br[i__3], q__2.imag() = (float)0.;
            q__1.real() = przth.real() * q__2.real() - przth.imag() * q__2.imag(), q__1.imag() = przth.real() * q__2.imag() + przth.imag() * q__2.real();
            cr[i__2].real() = q__1.real(), cr[i__2].imag() = q__1.imag();
            q__1.real() = przth.real() * rzth.real() - przth.imag() * rzth.imag(), q__1.imag() = przth.real() * rzth.imag() + przth.imag() * rzth.real();
            przth.real() = q__1.real(), przth.imag() = q__1.imag();
            i__2   = ks - 1;
            i__3   = ks + 1;
            q__2.real() = ar[i__3], q__2.imag() = (float)0.;
            q__1.real() = przth.real() * q__2.real() - przth.imag() * q__2.imag(), q__1.imag() = przth.real() * q__2.imag() + przth.imag() * q__2.real();
            dr[i__2].real() = q__1.real(), dr[i__2].imag() = q__1.imag();
            /* L160: */
        }
        pp *= rfnu2;
        if(ias == 1)
        {
            goto L180;
        }
        i__1   = lrp1 - 1;
        suma.real() = up[i__1].real(), suma.imag() = up[i__1].imag();
        ju   = lrp1;
        i__1 = lr;
        for(jr = 1; jr <= i__1; ++jr)
        {
            --ju;
            i__2   = jr - 1;
            i__3   = ju - 1;
            q__2.real() = cr[i__2].real() * up[i__3].real() - cr[i__2].imag() * up[i__3].imag(), q__2.imag() = cr[i__2].real() * up[i__3].imag() + cr[i__2].imag() * up[i__3].real();
            q__1.real() = suma.real() + q__2.real(), q__1.imag() = suma.imag() + q__2.imag();
            suma.real() = q__1.real(), suma.imag() = q__1.imag();
            /* L170: */
        }
        q__1.real() = asum->real() + suma.real(), q__1.imag() = asum->imag() + suma.imag();
        asum->real() = q__1.real(), asum->imag() = q__1.imag();
        asumr = asum->real();
        asumi = r_imag(asum);
        test  = abs(asumr) + abs(asumi);
        if(pp < *tol && test < *tol)
        {
            ias = 1;
        }
    L180:
        if(ibs == 1)
        {
            goto L200;
        }
        i__1   = lr + 1;
        i__2   = lrp1 - 1;
        q__2.real() = up[i__2].real() * zc.real() - up[i__2].imag() * zc.imag(), q__2.imag() = up[i__2].real() * zc.imag() + up[i__2].imag() * zc.real();
        q__1.real() = up[i__1].real() + q__2.real(), q__1.imag() = up[i__1].imag() + q__2.imag();
        sumb.real() = q__1.real(), sumb.imag() = q__1.imag();
        ju   = lrp1;
        i__1 = lr;
        for(jr = 1; jr <= i__1; ++jr)
        {
            --ju;
            i__2   = jr - 1;
            i__3   = ju - 1;
            q__2.real() = dr[i__2].real() * up[i__3].real() - dr[i__2].imag() * up[i__3].imag(), q__2.imag() = dr[i__2].real() * up[i__3].imag() + dr[i__2].imag() * up[i__3].real();
            q__1.real() = sumb.real() + q__2.real(), q__1.imag() = sumb.imag() + q__2.imag();
            sumb.real() = q__1.real(), sumb.imag() = q__1.imag();
            /* L190: */
        }
        q__1.real() = bsum->real() + sumb.real(), q__1.imag() = bsum->imag() + sumb.imag();
        bsum->real() = q__1.real(), bsum->imag() = q__1.imag();
        bsumr = bsum->real();
        bsumi = r_imag(bsum);
        test  = abs(bsumr) + abs(bsumi);
        if(pp < btol && test < *tol)
        {
            ibs = 1;
        }
    L200:
        if(ias == 1 && ibs == 1)
        {
            goto L220;
        }
        /* L210: */
    }
L220:
    q__1.real() = asum->real() + cone.real(), q__1.imag() = asum->imag() + cone.imag();
    asum->real() = q__1.real(), asum->imag() = q__1.imag();
    q__3.real() = -bsum->real(), q__3.imag() = -bsum->imag();
    q__2.real() = q__3.real() * rfn13.real() - q__3.imag() * rfn13.imag(), q__2.imag() = q__3.real() * rfn13.imag() + q__3.imag() * rfn13.real();
    c_div(&q__1, &q__2, &rtzta);
    bsum->real() = q__1.real(), bsum->imag() = q__1.imag();
    goto L120;
} /* cunhj_ */

 int cuni1_(System::Complex<float>* z__, float* fnu, int32* kode, int32* n, System::Complex<float>* y, int32* nz, int32* nlast, float* fnul, float* tol, float* elim, float* alim)
{
    /* Initialized data */

    static System::Complex<float> czero = {(float)0., (float)0.};
    static System::Complex<float> cone  = {(float)1., (float)0.};

    /* System generated locals */
    int32 i__1, i__2, i__3;
    float    r__1, r__2;
    System::Complex<float> q__1, q__2, q__3, q__4, q__5, q__6, q__7;

 

    
    int32                     i__, k, m;
    System::Complex<float>                     c1, c2, s1, s2;
    int32                     nd;
    float                        fn;
    System::Complex<float>                     cy[2];
    int32                     nn, nw;
    System::Complex<float>                     rz;
    float                        yy, c2i, c2m, c2r, rs1;
    System::Complex<float>                     cfn, phi, csr[3], css[3];
    int32                     nuf;
    float                        bry[3];
    System::Complex<float>                     sum;
    float                        aphi;
    System::Complex<float>                     cscl, crsc;
    int32                     init;
    System::Complex<float>                     cwrk[16], zeta1, zeta2;
    int32                     iflag;
    float                        ascle;
    

    /* *********************************************************************72 */

    /* ***BEGIN PROLOGUE  CUNI1 */
    /* ***REFER TO  CBESI,CBESK */

    /*     CUNI1 COMPUTES I(FNU,Z)  BY MEANS OF THE UNIFORM ASYMPTOTIC */
    /*     EXPANSION FOR I(FNU,Z) IN -PI/3.LE.ARG Z.LE.PI/3. */

    /*     FNUL IS THE SMALLEST ORDER PERMITTED FOR THE ASYMPTOTIC */
    /*     EXPANSION. NLAST=0 MEANS ALL OF THE Y VALUES WERE SET. */
    /*     NLAST.NE.0 IS THE NUMBER LEFT TO BE COMPUTED BY ANOTHER */
    /*     FORMULA FOR ORDERS FNU TO FNU+NLAST-1 BECAUSE FNU+NLAST-1.LT.FNUL. */
    /*     Y(I)=CZERO FOR I=NLAST+1,N */

    /* ***ROUTINES CALLED  CUCHK,CUNIK,CUOIK,R1MACH */
    /* ***END PROLOGUE  CUNI1 */
    /* Parameter adjustments */
    --y;

    /* Function Body */

    *nz    = 0;
    nd     = *n;
    *nlast = 0;
    /* ----------------------------------------------------------------------- */
    /*     COMPUTED VALUES WITH EXPONENTS BETWEEN ALIM AND ELIM IN MAG- */
    /*     NITUDE ARE SCALED TO KEEP INTERMEDIATE ARITHMETIC ON SCALE, */
    /*     EXP(ALIM)=EXP(ELIM)*TOL */
    /* ----------------------------------------------------------------------- */
    r__1   = (float)1. / *tol;
    q__1.real() = r__1, q__1.imag() = (float)0.;
    cscl.real() = q__1.real(), cscl.imag() = q__1.imag();
    q__1.real() = *tol, q__1.imag() = (float)0.;
    crsc.real() = q__1.real(), crsc.imag() = q__1.imag();
    css[0].real() = cscl.real(), css[0].imag() = cscl.imag();
    css[1].real() = cone.real(), css[1].imag() = cone.imag();
    css[2].real() = crsc.real(), css[2].imag() = crsc.imag();
    csr[0].real() = crsc.real(), csr[0].imag() = crsc.imag();
    csr[1].real() = cone.real(), csr[1].imag() = cone.imag();
    csr[2].real() = cscl.real(), csr[2].imag() = cscl.imag();
    bry[0] = r1mach_(&c__1) * (float)1e3 / *tol;
    /* ----------------------------------------------------------------------- */
    /*     CHECK FOR UNDERFLOW AND OVERFLOW ON FIRST MEMBER */
    /* ----------------------------------------------------------------------- */
    fn   = max(*fnu, (float)1.);
    init = 0;
    cunik_(z__, &fn, &c__1, &c__1, tol, &init, &phi, &zeta1, &zeta2, &sum, cwrk);
    if(*kode == 1)
    {
        goto L10;
    }
    q__1.real() = fn, q__1.imag() = (float)0.;
    cfn.real() = q__1.real(), cfn.imag() = q__1.imag();
    q__2.real() = -zeta1.real(), q__2.imag() = -zeta1.imag();
    q__5.real() = z__->real() + zeta2.real(), q__5.imag() = z__->imag() + zeta2.imag();
    c_div(&q__4, &cfn, &q__5);
    q__3.real() = cfn.real() * q__4.real() - cfn.imag() * q__4.imag(), q__3.imag() = cfn.real() * q__4.imag() + cfn.imag() * q__4.real();
    q__1.real() = q__2.real() + q__3.real(), q__1.imag() = q__2.imag() + q__3.imag();
    s1.real() = q__1.real(), s1.imag() = q__1.imag();
    goto L20;
L10:
    q__2.real() = -zeta1.real(), q__2.imag() = -zeta1.imag();
    q__1.real() = q__2.real() + zeta2.real(), q__1.imag() = q__2.imag() + zeta2.imag();
    s1.real() = q__1.real(), s1.imag() = q__1.imag();
L20:
    rs1 = s1.real();
    if(abs(rs1) > *elim)
    {
        goto L130;
    }
L30:
    nn   = std::min(2L, nd);
    i__1 = nn;
    for(i__ = 1; i__ <= i__1; ++i__)
    {
        fn   = *fnu + (float)(nd - i__);
        init = 0;
        cunik_(z__, &fn, &c__1, &c__0, tol, &init, &phi, &zeta1, &zeta2, &sum, cwrk);
        if(*kode == 1)
        {
            goto L40;
        }
        q__1.real() = fn, q__1.imag() = (float)0.;
        cfn.real() = q__1.real(), cfn.imag() = q__1.imag();
        yy     = r_imag(z__);
        q__3.real() = -zeta1.real(), q__3.imag() = -zeta1.imag();
        q__6.real() = z__->real() + zeta2.real(), q__6.imag() = z__->imag() + zeta2.imag();
        c_div(&q__5, &cfn, &q__6);
        q__4.real() = cfn.real() * q__5.real() - cfn.imag() * q__5.imag(), q__4.imag() = cfn.real() * q__5.imag() + cfn.imag() * q__5.real();
        q__2.real() = q__3.real() + q__4.real(), q__2.imag() = q__3.imag() + q__4.imag();
        q__7.real() = (float)0., q__7.imag() = yy;
        q__1.real() = q__2.real() + q__7.real(), q__1.imag() = q__2.imag() + q__7.imag();
        s1.real() = q__1.real(), s1.imag() = q__1.imag();
        goto L50;
    L40:
        q__2.real() = -zeta1.real(), q__2.imag() = -zeta1.imag();
        q__1.real() = q__2.real() + zeta2.real(), q__1.imag() = q__2.imag() + zeta2.imag();
        s1.real() = q__1.real(), s1.imag() = q__1.imag();
    L50:
        /* ----------------------------------------------------------------------- */
        /*     TEST FOR UNDERFLOW AND OVERFLOW */
        /* ----------------------------------------------------------------------- */
        rs1 = s1.real();
        if(abs(rs1) > *elim)
        {
            goto L110;
        }
        if(i__ == 1)
        {
            iflag = 2;
        }
        if(abs(rs1) < *alim)
        {
            goto L60;
        }
        /* ----------------------------------------------------------------------- */
        /*     REFINE  TEST AND SCALE */
        /* ----------------------------------------------------------------------- */
        aphi = c_abs(&phi);
        rs1 += log(aphi);
        if(abs(rs1) > *elim)
        {
            goto L110;
        }
        if(i__ == 1)
        {
            iflag = 1;
        }
        if(rs1 < (float)0.)
        {
            goto L60;
        }
        if(i__ == 1)
        {
            iflag = 3;
        }
    L60:
        /* ----------------------------------------------------------------------- */
        /*     SCALE S1 IF CABS(S1).LT.ASCLE */
        /* ----------------------------------------------------------------------- */
        q__1.real() = phi.real() * sum.real() - phi.imag() * sum.imag(), q__1.imag() = phi.real() * sum.imag() + phi.imag() * sum.real();
        s2.real() = q__1.real(), s2.imag() = q__1.imag();
        c2r    = s1.real();
        c2i    = r_imag(&s1);
        i__2   = iflag - 1;
        c2m    = exp(c2r) * css[i__2].real();
        q__2.real() = c2m, q__2.imag() = (float)0.;
        r__1   = cos(c2i);
        r__2   = sin(c2i);
        q__3.real() = r__1, q__3.imag() = r__2;
        q__1.real() = q__2.real() * q__3.real() - q__2.imag() * q__3.imag(), q__1.imag() = q__2.real() * q__3.imag() + q__2.imag() * q__3.real();
        s1.real() = q__1.real(), s1.imag() = q__1.imag();
        q__1.real() = s2.real() * s1.real() - s2.imag() * s1.imag(), q__1.imag() = s2.real() * s1.imag() + s2.imag() * s1.real();
        s2.real() = q__1.real(), s2.imag() = q__1.imag();
        if(iflag != 1)
        {
            goto L70;
        }
        cuchk_(&s2, &nw, bry, tol);
        if(nw != 0)
        {
            goto L110;
        }
    L70:
        m          = nd - i__ + 1;
        i__2       = i__ - 1;
        cy[i__2].real() = s2.real(), cy[i__2].imag() = s2.imag();
        i__2   = m;
        i__3   = iflag - 1;
        q__1.real() = s2.real() * csr[i__3].real() - s2.imag() * csr[i__3].imag(), q__1.imag() = s2.real() * csr[i__3].imag() + s2.imag() * csr[i__3].real();
        y[i__2].real() = q__1.real(), y[i__2].imag() = q__1.imag();
        /* L80: */
    }
    if(nd <= 2)
    {
        goto L100;
    }
    c_div(&q__1, &c_b17, z__);
    rz.real() = q__1.real(), rz.imag() = q__1.imag();
    bry[1] = (float)1. / bry[0];
    bry[2] = r1mach_(&c__2);
    s1.real() = cy[0].real(), s1.imag() = cy[0].imag();
    s2.real() = cy[1].real(), s2.imag() = cy[1].imag();
    i__1 = iflag - 1;
    c1.real() = csr[i__1].real(), c1.imag() = csr[i__1].imag();
    ascle = bry[iflag - 1];
    k     = nd - 2;
    fn    = (float)k;
    i__1  = nd;
    for(i__ = 3; i__ <= i__1; ++i__)
    {
        c2.real() = s2.real(), c2.imag() = s2.imag();
        r__1   = *fnu + fn;
        q__4.real() = r__1, q__4.imag() = (float)0.;
        q__3.real() = q__4.real() * rz.real() - q__4.imag() * rz.imag(), q__3.imag() = q__4.real() * rz.imag() + q__4.imag() * rz.real();
        q__2.real() = q__3.real() * s2.real() - q__3.imag() * s2.imag(), q__2.imag() = q__3.real() * s2.imag() + q__3.imag() * s2.real();
        q__1.real() = s1.real() + q__2.real(), q__1.imag() = s1.imag() + q__2.imag();
        s2.real() = q__1.real(), s2.imag() = q__1.imag();
        s1.real() = c2.real(), s1.imag() = c2.imag();
        q__1.real() = s2.real() * c1.real() - s2.imag() * c1.imag(), q__1.imag() = s2.real() * c1.imag() + s2.imag() * c1.real();
        c2.real() = q__1.real(), c2.imag() = q__1.imag();
        i__2      = k;
        y[i__2].real() = c2.real(), y[i__2].imag() = c2.imag();
        --k;
        fn += (float)-1.;
        if(iflag >= 3)
        {
            goto L90;
        }
        c2r = c2.real();
        c2i = r_imag(&c2);
        c2r = abs(c2r);
        c2i = abs(c2i);
        c2m = max(c2r, c2i);
        if(c2m <= ascle)
        {
            goto L90;
        }
        ++iflag;
        ascle  = bry[iflag - 1];
        q__1.real() = s1.real() * c1.real() - s1.imag() * c1.imag(), q__1.imag() = s1.real() * c1.imag() + s1.imag() * c1.real();
        s1.real() = q__1.real(), s1.imag() = q__1.imag();
        s2.real() = c2.real(), s2.imag() = c2.imag();
        i__2   = iflag - 1;
        q__1.real() = s1.real() * css[i__2].real() - s1.imag() * css[i__2].imag(), q__1.imag() = s1.real() * css[i__2].imag() + s1.imag() * css[i__2].real();
        s1.real() = q__1.real(), s1.imag() = q__1.imag();
        i__2   = iflag - 1;
        q__1.real() = s2.real() * css[i__2].real() - s2.imag() * css[i__2].imag(), q__1.imag() = s2.real() * css[i__2].imag() + s2.imag() * css[i__2].real();
        s2.real() = q__1.real(), s2.imag() = q__1.imag();
        i__2 = iflag - 1;
        c1.real() = csr[i__2].real(), c1.imag() = csr[i__2].imag();
    L90:;
    }
L100:
    return 0;
/* ----------------------------------------------------------------------- */
/*     SET UNDERFLOW AND UPDATE PARAMETERS */
/* ----------------------------------------------------------------------- */
L110:
    if(rs1 > (float)0.)
    {
        goto L120;
    }
    i__1      = nd;
    y[i__1].real() = czero.real(), y[i__1].imag() = czero.imag();
    ++(*nz);
    --nd;
    if(nd == 0)
    {
        goto L100;
    }
    cuoik_(z__, fnu, kode, &c__1, &nd, &y[1], &nuf, tol, elim, alim);
    if(nuf < 0)
    {
        goto L120;
    }
    nd -= nuf;
    *nz += nuf;
    if(nd == 0)
    {
        goto L100;
    }
    fn = *fnu + (float)(nd - 1);
    if(fn >= *fnul)
    {
        goto L30;
    }
    *nlast = nd;
    return 0;
L120:
    *nz = -1;
    return 0;
L130:
    if(rs1 > (float)0.)
    {
        goto L120;
    }
    *nz  = *n;
    i__1 = *n;
    for(i__ = 1; i__ <= i__1; ++i__)
    {
        i__2      = i__;
        y[i__2].real() = czero.real(), y[i__2].imag() = czero.imag();
        /* L140: */
    }
    return 0;
} /* cuni1_ */

 int cuni2_(System::Complex<float>* z__, float* fnu, int32* kode, int32* n, System::Complex<float>* y, int32* nz, int32* nlast, float* fnul, float* tol, float* elim, float* alim)
{
    /* Initialized data */

    static System::Complex<float> czero  = {(float)0., (float)0.};
    static System::Complex<float> cone   = {(float)1., (float)0.};
    static System::Complex<float> ci     = {(float)0., (float)1.};
    static System::Complex<float> cip[4] = {{(float)1., (float)0.}, {(float)0., (float)1.}, {(float)-1., (float)0.}, {(float)0., (float)-1.}};
    static float    hpi    = (float)1.57079632679489662;
    static float    aic    = (float)1.265512123484645396;

    /* System generated locals */
    int32 i__1, i__2, i__3;
    float    r__1, r__2;
    System::Complex<float> q__1, q__2, q__3, q__4, q__5, q__6, q__7;


    
    int32                     i__, j, k;
    System::Complex<float>                     c1, c2, s1, s2, ai;
    int32                     nd;
    float                        fn;
    int32                     in;
    float                        ay;
    System::Complex<float>                     cy[2], zb;
    int32                     nn, nw;
    System::Complex<float>                     zn, rz;
    float                        yy, c2i, c2m, c2r, rs1;
    System::Complex<float>                     dai, cid;
    float                        ang;
    System::Complex<float>                     cfn;
    float                        car;
    int32                     nai;
    System::Complex<float>                     arg, phi;
    float                        sar;
    System::Complex<float>                     csr[3], css[3];
    int32                     nuf, inu;
    System::Complex<float>                     zar;
    float                        bry[3], aarg;
    int32                     ndai;
    float                        aphi;
    System::Complex<float>                     cscl, crsc;
    int32                     idum;
    System::Complex<float>                     asum, bsum, zeta1, zeta2;
    int32                     iflag;
    float                        ascle;

    /* *********************************************************************72 */

    /* ***BEGIN PROLOGUE  CUNI2 */
    /* ***REFER TO  CBESI,CBESK */

    /*     CUNI2 COMPUTES I(FNU,Z) IN THE RIGHT HALF PLANE BY MEANS OF */
    /*     UNIFORM ASYMPTOTIC EXPANSION FOR J(FNU,ZN) WHERE ZN IS Z*I */
    /*     OR -Z*I AND ZN IS IN THE RIGHT HALF PLANE ALSO. */

    /*     FNUL IS THE SMALLEST ORDER PERMITTED FOR THE ASYMPTOTIC */
    /*     EXPANSION. NLAST=0 MEANS ALL OF THE Y VALUES WERE SET. */
    /*     NLAST.NE.0 IS THE NUMBER LEFT TO BE COMPUTED BY ANOTHER */
    /*     FORMULA FOR ORDERS FNU TO FNU+NLAST-1 BECAUSE FNU+NLAST-1.LT.FNUL. */
    /*     Y(I)=CZERO FOR I=NLAST+1,N */

    /* ***ROUTINES CALLED  CAIRY,CUCHK,CUNHJ,CUOIK,R1MACH */
    /* ***END PROLOGUE  CUNI2 */
    /* Parameter adjustments */
    --y;

    /* Function Body */

    *nz    = 0;
    nd     = *n;
    *nlast = 0;
    /* ----------------------------------------------------------------------- */
    /*     COMPUTED VALUES WITH EXPONENTS BETWEEN ALIM AND ELIM IN MAG- */
    /*     NITUDE ARE SCALED TO KEEP INTERMEDIATE ARITHMETIC ON SCALE, */
    /*     EXP(ALIM)=EXP(ELIM)*TOL */
    /* ----------------------------------------------------------------------- */
    r__1   = (float)1. / *tol;
    q__1.real() = r__1, q__1.imag() = (float)0.;
    cscl.real() = q__1.real(), cscl.imag() = q__1.imag();
    q__1.real() = *tol, q__1.imag() = (float)0.;
    crsc.real() = q__1.real(), crsc.imag() = q__1.imag();
    css[0].real() = cscl.real(), css[0].imag() = cscl.imag();
    css[1].real() = cone.real(), css[1].imag() = cone.imag();
    css[2].real() = crsc.real(), css[2].imag() = crsc.imag();
    csr[0].real() = crsc.real(), csr[0].imag() = crsc.imag();
    csr[1].real() = cone.real(), csr[1].imag() = cone.imag();
    csr[2].real() = cscl.real(), csr[2].imag() = cscl.imag();
    bry[0] = r1mach_(&c__1) * (float)1e3 / *tol;
    yy     = r_imag(z__);
    /* ----------------------------------------------------------------------- */
    /*     ZN IS IN THE RIGHT HALF PLANE AFTER ROTATION BY CI OR -CI */
    /* ----------------------------------------------------------------------- */
    q__2.real() = -z__->real(), q__2.imag() = -z__->imag();
    q__1.real() = q__2.real() * ci.real() - q__2.imag() * ci.imag(), q__1.imag() = q__2.real() * ci.imag() + q__2.imag() * ci.real();
    zn.real() = q__1.real(), zn.imag() = q__1.imag();
    zb.real() = z__->real(), zb.imag() = z__->imag();
    q__1.real() = -ci.real(), q__1.imag() = -ci.imag();
    cid.real() = q__1.real(), cid.imag() = q__1.imag();
    inu    = (int32)(*fnu);
    ang    = hpi * (*fnu - (float)inu);
    car    = cos(ang);
    sar    = sin(ang);
    q__1.real() = car, q__1.imag() = sar;
    c2.real() = q__1.real(), c2.imag() = q__1.imag();
    zar.real() = c2.real(), zar.imag() = c2.imag();
    in = inu + *n - 1;
    in %= 4;
    i__1   = in;
    q__1.real() = c2.real() * cip[i__1].real() - c2.imag() * cip[i__1].imag(), q__1.imag() = c2.real() * cip[i__1].imag() + c2.imag() * cip[i__1].real();
    c2.real() = q__1.real(), c2.imag() = q__1.imag();
    if(yy > (float)0.)
    {
        goto L10;
    }
    q__2.real() = -zn.real(), q__2.imag() = -zn.imag();
    r_cnjg(&q__1, &q__2);
    zn.real() = q__1.real(), zn.imag() = q__1.imag();
    r_cnjg(&q__1, &zb);
    zb.real() = q__1.real(), zb.imag() = q__1.imag();
    q__1.real() = -cid.real(), q__1.imag() = -cid.imag();
    cid.real() = q__1.real(), cid.imag() = q__1.imag();
    r_cnjg(&q__1, &c2);
    c2.real() = q__1.real(), c2.imag() = q__1.imag();
L10:
    /* ----------------------------------------------------------------------- */
    /*     CHECK FOR UNDERFLOW AND OVERFLOW ON FIRST MEMBER */
    /* ----------------------------------------------------------------------- */
    fn = max(*fnu, (float)1.);
    cunhj_(&zn, &fn, &c__1, tol, &phi, &arg, &zeta1, &zeta2, &asum, &bsum);
    if(*kode == 1)
    {
        goto L20;
    }
    q__1.real() = *fnu, q__1.imag() = (float)0.;
    cfn.real() = q__1.real(), cfn.imag() = q__1.imag();
    q__2.real() = -zeta1.real(), q__2.imag() = -zeta1.imag();
    q__5.real() = zb.real() + zeta2.real(), q__5.imag() = zb.imag() + zeta2.imag();
    c_div(&q__4, &cfn, &q__5);
    q__3.real() = cfn.real() * q__4.real() - cfn.imag() * q__4.imag(), q__3.imag() = cfn.real() * q__4.imag() + cfn.imag() * q__4.real();
    q__1.real() = q__2.real() + q__3.real(), q__1.imag() = q__2.imag() + q__3.imag();
    s1.real() = q__1.real(), s1.imag() = q__1.imag();
    goto L30;
L20:
    q__2.real() = -zeta1.real(), q__2.imag() = -zeta1.imag();
    q__1.real() = q__2.real() + zeta2.real(), q__1.imag() = q__2.imag() + zeta2.imag();
    s1.real() = q__1.real(), s1.imag() = q__1.imag();
L30:
    rs1 = s1.real();
    if(abs(rs1) > *elim)
    {
        goto L150;
    }
L40:
    nn   = std::min(2L, nd);
    i__1 = nn;
    for(i__ = 1; i__ <= i__1; ++i__)
    {
        fn = *fnu + (float)(nd - i__);
        cunhj_(&zn, &fn, &c__0, tol, &phi, &arg, &zeta1, &zeta2, &asum, &bsum);
        if(*kode == 1)
        {
            goto L50;
        }
        q__1.real() = fn, q__1.imag() = (float)0.;
        cfn.real() = q__1.real(), cfn.imag() = q__1.imag();
        ay     = abs(yy);
        q__3.real() = -zeta1.real(), q__3.imag() = -zeta1.imag();
        q__6.real() = zb.real() + zeta2.real(), q__6.imag() = zb.imag() + zeta2.imag();
        c_div(&q__5, &cfn, &q__6);
        q__4.real() = cfn.real() * q__5.real() - cfn.imag() * q__5.imag(), q__4.imag() = cfn.real() * q__5.imag() + cfn.imag() * q__5.real();
        q__2.real() = q__3.real() + q__4.real(), q__2.imag() = q__3.imag() + q__4.imag();
        q__7.real() = (float)0., q__7.imag() = ay;
        q__1.real() = q__2.real() + q__7.real(), q__1.imag() = q__2.imag() + q__7.imag();
        s1.real() = q__1.real(), s1.imag() = q__1.imag();
        goto L60;
    L50:
        q__2.real() = -zeta1.real(), q__2.imag() = -zeta1.imag();
        q__1.real() = q__2.real() + zeta2.real(), q__1.imag() = q__2.imag() + zeta2.imag();
        s1.real() = q__1.real(), s1.imag() = q__1.imag();
    L60:
        /* ----------------------------------------------------------------------- */
        /*     TEST FOR UNDERFLOW AND OVERFLOW */
        /* ----------------------------------------------------------------------- */
        rs1 = s1.real();
        if(abs(rs1) > *elim)
        {
            goto L120;
        }
        if(i__ == 1)
        {
            iflag = 2;
        }
        if(abs(rs1) < *alim)
        {
            goto L70;
        }
        /* ----------------------------------------------------------------------- */
        /*     REFINE  TEST AND SCALE */
        /* ----------------------------------------------------------------------- */
        /* ----------------------------------------------------------------------- */
        aphi = c_abs(&phi);
        aarg = c_abs(&arg);
        rs1  = rs1 + log(aphi) - log(aarg) * (float).25 - aic;
        if(abs(rs1) > *elim)
        {
            goto L120;
        }
        if(i__ == 1)
        {
            iflag = 1;
        }
        if(rs1 < (float)0.)
        {
            goto L70;
        }
        if(i__ == 1)
        {
            iflag = 3;
        }
    L70:
        /* ----------------------------------------------------------------------- */
        /*     SCALE S1 TO KEEP INTERMEDIATE ARITHMETIC ON SCALE NEAR */
        /*     EXPONENT EXTREMES */
        /* ----------------------------------------------------------------------- */
        cairy_(&arg, &c__0, &c__2, &ai, &nai, &idum);
        cairy_(&arg, &c__1, &c__2, &dai, &ndai, &idum);
        q__3.real() = ai.real() * asum.real() - ai.imag() * asum.imag(), q__3.imag() = ai.real() * asum.imag() + ai.imag() * asum.real();
        q__4.real() = dai.real() * bsum.real() - dai.imag() * bsum.imag(), q__4.imag() = dai.real() * bsum.imag() + dai.imag() * bsum.real();
        q__2.real() = q__3.real() + q__4.real(), q__2.imag() = q__3.imag() + q__4.imag();
        q__1.real() = phi.real() * q__2.real() - phi.imag() * q__2.imag(), q__1.imag() = phi.real() * q__2.imag() + phi.imag() * q__2.real();
        s2.real() = q__1.real(), s2.imag() = q__1.imag();
        c2r    = s1.real();
        c2i    = r_imag(&s1);
        i__2   = iflag - 1;
        c2m    = exp(c2r) * css[i__2].real();
        q__2.real() = c2m, q__2.imag() = (float)0.;
        r__1   = cos(c2i);
        r__2   = sin(c2i);
        q__3.real() = r__1, q__3.imag() = r__2;
        q__1.real() = q__2.real() * q__3.real() - q__2.imag() * q__3.imag(), q__1.imag() = q__2.real() * q__3.imag() + q__2.imag() * q__3.real();
        s1.real() = q__1.real(), s1.imag() = q__1.imag();
        q__1.real() = s2.real() * s1.real() - s2.imag() * s1.imag(), q__1.imag() = s2.real() * s1.imag() + s2.imag() * s1.real();
        s2.real() = q__1.real(), s2.imag() = q__1.imag();
        if(iflag != 1)
        {
            goto L80;
        }
        cuchk_(&s2, &nw, bry, tol);
        if(nw != 0)
        {
            goto L120;
        }
    L80:
        if(yy <= (float)0.)
        {
            r_cnjg(&q__1, &s2);
            s2.real() = q__1.real(), s2.imag() = q__1.imag();
        }
        j      = nd - i__ + 1;
        q__1.real() = s2.real() * c2.real() - s2.imag() * c2.imag(), q__1.imag() = s2.real() * c2.imag() + s2.imag() * c2.real();
        s2.real() = q__1.real(), s2.imag() = q__1.imag();
        i__2       = i__ - 1;
        cy[i__2].real() = s2.real(), cy[i__2].imag() = s2.imag();
        i__2   = j;
        i__3   = iflag - 1;
        q__1.real() = s2.real() * csr[i__3].real() - s2.imag() * csr[i__3].imag(), q__1.imag() = s2.real() * csr[i__3].imag() + s2.imag() * csr[i__3].real();
        y[i__2].real() = q__1.real(), y[i__2].imag() = q__1.imag();
        q__1.real() = c2.real() * cid.real() - c2.imag() * cid.imag(), q__1.imag() = c2.real() * cid.imag() + c2.imag() * cid.real();
        c2.real() = q__1.real(), c2.imag() = q__1.imag();
        /* L90: */
    }
    if(nd <= 2)
    {
        goto L110;
    }
    c_div(&q__1, &c_b17, z__);
    rz.real() = q__1.real(), rz.imag() = q__1.imag();
    bry[1] = (float)1. / bry[0];
    bry[2] = r1mach_(&c__2);
    s1.real() = cy[0].real(), s1.imag() = cy[0].imag();
    s2.real() = cy[1].real(), s2.imag() = cy[1].imag();
    i__1 = iflag - 1;
    c1.real() = csr[i__1].real(), c1.imag() = csr[i__1].imag();
    ascle = bry[iflag - 1];
    k     = nd - 2;
    fn    = (float)k;
    i__1  = nd;
    for(i__ = 3; i__ <= i__1; ++i__)
    {
        c2.real() = s2.real(), c2.imag() = s2.imag();
        r__1   = *fnu + fn;
        q__4.real() = r__1, q__4.imag() = (float)0.;
        q__3.real() = q__4.real() * rz.real() - q__4.imag() * rz.imag(), q__3.imag() = q__4.real() * rz.imag() + q__4.imag() * rz.real();
        q__2.real() = q__3.real() * s2.real() - q__3.imag() * s2.imag(), q__2.imag() = q__3.real() * s2.imag() + q__3.imag() * s2.real();
        q__1.real() = s1.real() + q__2.real(), q__1.imag() = s1.imag() + q__2.imag();
        s2.real() = q__1.real(), s2.imag() = q__1.imag();
        s1.real() = c2.real(), s1.imag() = c2.imag();
        q__1.real() = s2.real() * c1.real() - s2.imag() * c1.imag(), q__1.imag() = s2.real() * c1.imag() + s2.imag() * c1.real();
        c2.real() = q__1.real(), c2.imag() = q__1.imag();
        i__2      = k;
        y[i__2].real() = c2.real(), y[i__2].imag() = c2.imag();
        --k;
        fn += (float)-1.;
        if(iflag >= 3)
        {
            goto L100;
        }
        c2r = c2.real();
        c2i = r_imag(&c2);
        c2r = abs(c2r);
        c2i = abs(c2i);
        c2m = max(c2r, c2i);
        if(c2m <= ascle)
        {
            goto L100;
        }
        ++iflag;
        ascle  = bry[iflag - 1];
        q__1.real() = s1.real() * c1.real() - s1.imag() * c1.imag(), q__1.imag() = s1.real() * c1.imag() + s1.imag() * c1.real();
        s1.real() = q__1.real(), s1.imag() = q__1.imag();
        s2.real() = c2.real(), s2.imag() = c2.imag();
        i__2   = iflag - 1;
        q__1.real() = s1.real() * css[i__2].real() - s1.imag() * css[i__2].imag(), q__1.imag() = s1.real() * css[i__2].imag() + s1.imag() * css[i__2].real();
        s1.real() = q__1.real(), s1.imag() = q__1.imag();
        i__2   = iflag - 1;
        q__1.real() = s2.real() * css[i__2].real() - s2.imag() * css[i__2].imag(), q__1.imag() = s2.real() * css[i__2].imag() + s2.imag() * css[i__2].real();
        s2.real() = q__1.real(), s2.imag() = q__1.imag();
        i__2 = iflag - 1;
        c1.real() = csr[i__2].real(), c1.imag() = csr[i__2].imag();
    L100:;
    }
L110:
    return 0;
L120:
    if(rs1 > (float)0.)
    {
        goto L140;
    }
    /* ----------------------------------------------------------------------- */
    /*     SET UNDERFLOW AND UPDATE PARAMETERS */
    /* ----------------------------------------------------------------------- */
    i__1      = nd;
    y[i__1].real() = czero.real(), y[i__1].imag() = czero.imag();
    ++(*nz);
    --nd;
    if(nd == 0)
    {
        goto L110;
    }
    cuoik_(z__, fnu, kode, &c__1, &nd, &y[1], &nuf, tol, elim, alim);
    if(nuf < 0)
    {
        goto L140;
    }
    nd -= nuf;
    *nz += nuf;
    if(nd == 0)
    {
        goto L110;
    }
    fn = *fnu + (float)(nd - 1);
    if(fn < *fnul)
    {
        goto L130;
    }
    /*      FN = AIMAG(CID) */
    /*      J = NUF + 1 */
    /*      K = MOD(J,4) + 1 */
    /*      S1 = CIP(K) */
    /*      IF (FN.LT.0.0E0) S1 = CONJG(S1) */
    /*      C2 = C2*S1 */
    in     = inu + nd - 1;
    in     = in % 4 + 1;
    i__1   = in - 1;
    q__1.real() = zar.real() * cip[i__1].real() - zar.imag() * cip[i__1].imag(), q__1.imag() = zar.real() * cip[i__1].imag() + zar.imag() * cip[i__1].real();
    c2.real() = q__1.real(), c2.imag() = q__1.imag();
    if(yy <= (float)0.)
    {
        r_cnjg(&q__1, &c2);
        c2.real() = q__1.real(), c2.imag() = q__1.imag();
    }
    goto L40;
L130:
    *nlast = nd;
    return 0;
L140:
    *nz = -1;
    return 0;
L150:
    if(rs1 > (float)0.)
    {
        goto L140;
    }
    *nz  = *n;
    i__1 = *n;
    for(i__ = 1; i__ <= i__1; ++i__)
    {
        i__2      = i__;
        y[i__2].real() = czero.real(), y[i__2].imag() = czero.imag();
        /* L160: */
    }
    return 0;
} /* cuni2_ */

 int cunik_(System::Complex<float>* zr,
                            float*    fnu,
                            int32* ikflg,
                            int32* ipmtr,
                            float*    tol,
                            int32* init,
                            System::Complex<float>* phi,
                            System::Complex<float>* zeta1,
                            System::Complex<float>* zeta2,
                            System::Complex<float>* sum,
                            System::Complex<float>* cwrk)
{
    /* Initialized data */

    static System::Complex<float> czero    = {(float)0., (float)0.};
    static System::Complex<float> cone     = {(float)1., (float)0.};
    static System::Complex<float> con[2]   = {{(float).398942280401432678, (float)0.}, {(float)1.25331413731550025, (float)0.}};
    static float    c__[120] = {(float)1.,
                            (float)-.208333333333333333,
                            (float).125,
                            (float).334201388888888889,
                            (float)-.401041666666666667,
                            (float).0703125,
                            (float)-1.02581259645061728,
                            (float)1.84646267361111111,
                            (float)-.8912109375,
                            (float).0732421875,
                            (float)4.66958442342624743,
                            (float)-11.2070026162229938,
                            (float)8.78912353515625,
                            (float)-2.3640869140625,
                            (float).112152099609375,
                            (float)-28.2120725582002449,
                            (float)84.6362176746007346,
                            (float)-91.8182415432400174,
                            (float)42.5349987453884549,
                            (float)-7.3687943594796317,
                            (float).227108001708984375,
                            (float)212.570130039217123,
                            (float)-765.252468141181642,
                            (float)1059.99045252799988,
                            (float)-699.579627376132541,
                            (float)218.19051174421159,
                            (float)-26.4914304869515555,
                            (float).572501420974731445,
                            (float)-1919.457662318407,
                            (float)8061.72218173730938,
                            (float)-13586.5500064341374,
                            (float)11655.3933368645332,
                            (float)-5305.64697861340311,
                            (float)1200.90291321635246,
                            (float)-108.090919788394656,
                            (float)1.7277275025844574,
                            (float)20204.2913309661486,
                            (float)-96980.5983886375135,
                            (float)192547.001232531532,
                            (float)-203400.177280415534,
                            (float)122200.46498301746,
                            (float)-41192.6549688975513,
                            (float)7109.51430248936372,
                            (float)-493.915304773088012,
                            (float)6.07404200127348304,
                            (float)-242919.187900551333,
                            (float)1311763.6146629772,
                            (float)-2998015.91853810675,
                            (float)3763271.297656404,
                            (float)-2813563.22658653411,
                            (float)1268365.27332162478,
                            (float)-331645.172484563578,
                            (float)45218.7689813627263,
                            (float)-2499.83048181120962,
                            (float)24.3805296995560639,
                            (float)3284469.85307203782,
                            (float)-19706819.1184322269,
                            (float)50952602.4926646422,
                            (float)-74105148.2115326577,
                            (float)66344512.2747290267,
                            (float)-37567176.6607633513,
                            (float)13288767.1664218183,
                            (float)-2785618.12808645469,
                            (float)308186.404612662398,
                            (float)-13886.0897537170405,
                            (float)110.017140269246738,
                            (float)-49329253.664509962,
                            (float)325573074.185765749,
                            (float)-939462359.681578403,
                            (float)1553596899.57058006,
                            (float)-1621080552.10833708,
                            (float)1106842816.82301447,
                            (float)-495889784.275030309,
                            (float)142062907.797533095,
                            (float)-24474062.7257387285,
                            (float)2243768.17792244943,
                            (float)-84005.4336030240853,
                            (float)551.335896122020586,
                            (float)814789096.118312115,
                            (float)-5866481492.05184723,
                            (float)18688207509.2958249,
                            (float)-34632043388.1587779,
                            (float)41280185579.753974,
                            (float)-33026599749.8007231,
                            (float)17954213731.1556001,
                            (float)-6563293792.61928433,
                            (float)1559279864.87925751,
                            (float)-225105661.889415278,
                            (float)17395107.5539781645,
                            (float)-549842.327572288687,
                            (float)3038.09051092238427,
                            (float)-14679261247.6956167,
                            (float)114498237732.02581,
                            (float)-399096175224.466498,
                            (float)819218669548.577329,
                            (float)-1098375156081.22331,
                            (float)1008158106865.38209,
                            (float)-645364869245.376503,
                            (float)287900649906.150589,
                            (float)-87867072178.0232657,
                            (float)17634730606.8349694,
                            (float)-2167164983.22379509,
                            (float)143157876.718888981,
                            (float)-3871833.44257261262,
                            (float)18257.7554742931747,
                            (float)286464035717.679043,
                            (float)-2406297900028.50396,
                            (float)9109341185239.89896,
                            (float)-20516899410934.4374,
                            (float)30565125519935.3206,
                            (float)-31667088584785.1584,
                            (float)23348364044581.8409,
                            (float)-12320491305598.2872,
                            (float)4612725780849.13197,
                            (float)-1196552880196.1816,
                            (float)205914503232.410016,
                            (float)-21822927757.5292237,
                            (float)1247009293.51271032,
                            (float)-29188388.1222208134,
                            (float)118838.426256783253};

    /* System generated locals */
    int32 i__1, i__2;
    float    r__1;
    System::Complex<float> q__1, q__2, q__3;


    int32           i__, j, k, l;
    System::Complex<float>           s, t, t2;
    float              ac;
    System::Complex<float>           sr, zn, cfn;
    float              rfn;
    System::Complex<float>           crfn;
    float              test, tsti, tstr;
    

    /* *********************************************************************72 */

    /* ***BEGIN PROLOGUE  CUNIK */
    /* ***REFER TO  CBESI,CBESK */

    /*        CUNIK COMPUTES PARAMETERS FOR THE UNIFORM ASYMPTOTIC */
    /*        EXPANSIONS OF THE I AND K FUNCTIONS ON IKFLG= 1 OR 2 */
    /*        RESPECTIVELY BY */

    /*        W(FNU,ZR) = PHI*EXP(ZETA)*SUM */

    /*        WHERE       ZETA=-ZETA1 + ZETA2       OR */
    /*                          ZETA1 - ZETA2 */

    /*        THE FIRST CALL MUST HAVE INIT=0. SUBSEQUENT CALLS WITH THE */
    /*        SAME ZR AND FNU WILL RETURN THE I OR K FUNCTION ON IKFLG= */
    /*        1 OR 2 WITH NO CHANGE IN INIT. CWRK IS A COMPLEX WORK */
    /*        ARRAY. IPMTR=0 COMPUTES ALL PARAMETERS. IPMTR=1 COMPUTES PHI, */
    /*        ZETA1,ZETA2. */

    /* ***ROUTINES CALLED  (NONE) */
    /* ***END PROLOGUE  CUNIK */
    /* Parameter adjustments */
    --cwrk;

    /* Function Body */

    if(*init != 0)
    {
        goto L40;
    }
    /* ----------------------------------------------------------------------- */
    /*     INITIALIZE ALL VARIABLES */
    /* ----------------------------------------------------------------------- */
    rfn    = (float)1. / *fnu;
    q__1.real() = rfn, q__1.imag() = (float)0.;
    crfn.real() = q__1.real(), crfn.imag() = q__1.imag();
    /*     T = ZR*CRFN */
    /* ----------------------------------------------------------------------- */
    /*     OVERFLOW TEST (ZR/FNU TOO SMALL) */
    /* ----------------------------------------------------------------------- */
    tstr = zr->real();
    tsti = r_imag(zr);
    test = r1mach_(&c__1) * (float)1e3;
    ac   = *fnu * test;
    if(abs(tstr) > ac || abs(tsti) > ac)
    {
        goto L15;
    }
    ac     = (r__1 = log(test), abs(r__1)) * (float)2. + *fnu;
    q__1.real() = ac, q__1.imag() = (float)0.;
    zeta1->real() = q__1.real(), zeta1->imag() = q__1.imag();
    q__1.real() = *fnu, q__1.imag() = (float)0.;
    zeta2->real() = q__1.real(), zeta2->imag() = q__1.imag();
    phi->real() = cone.real(), phi->imag() = cone.imag();
    return 0;
L15:
    q__1.real() = zr->real() * crfn.real() - zr->imag() * crfn.imag(), q__1.imag() = zr->real() * crfn.imag() + zr->imag() * crfn.real();
    t.real() = q__1.real(), t.imag() = q__1.imag();
    q__2.real() = t.real() * t.real() - t.imag() * t.imag(), q__2.imag() = t.real() * t.imag() + t.imag() * t.real();
    q__1.real() = cone.real() + q__2.real(), q__1.imag() = cone.imag() + q__2.imag();
    s.real() = q__1.real(), s.imag() = q__1.imag();
    c_sqrt(&q__1, &s);
    sr.real() = q__1.real(), sr.imag() = q__1.imag();
    q__1.real() = *fnu, q__1.imag() = (float)0.;
    cfn.real() = q__1.real(), cfn.imag() = q__1.imag();
    q__2.real() = cone.real() + sr.real(), q__2.imag() = cone.imag() + sr.imag();
    c_div(&q__1, &q__2, &t);
    zn.real() = q__1.real(), zn.imag() = q__1.imag();
    c_log(&q__2, &zn);
    q__1.real() = cfn.real() * q__2.real() - cfn.imag() * q__2.imag(), q__1.imag() = cfn.real() * q__2.imag() + cfn.imag() * q__2.real();
    zeta1->real() = q__1.real(), zeta1->imag() = q__1.imag();
    q__1.real() = cfn.real() * sr.real() - cfn.imag() * sr.imag(), q__1.imag() = cfn.real() * sr.imag() + cfn.imag() * sr.real();
    zeta2->real() = q__1.real(), zeta2->imag() = q__1.imag();
    c_div(&q__1, &cone, &sr);
    t.real() = q__1.real(), t.imag() = q__1.imag();
    q__1.real() = t.real() * crfn.real() - t.imag() * crfn.imag(), q__1.imag() = t.real() * crfn.imag() + t.imag() * crfn.real();
    sr.real() = q__1.real(), sr.imag() = q__1.imag();
    c_sqrt(&q__1, &sr);
    cwrk[16].real() = q__1.real(), cwrk[16].imag() = q__1.imag();
    i__1   = *ikflg - 1;
    q__1.real() = cwrk[16].real() * con[i__1].real() - cwrk[16].imag() * con[i__1].imag(), q__1.imag() = cwrk[16].real() * con[i__1].imag() + cwrk[16].imag() * con[i__1].real();
    phi->real() = q__1.real(), phi->imag() = q__1.imag();
    if(*ipmtr != 0)
    {
        return 0;
    }
    c_div(&q__1, &cone, &s);
    t2.real() = q__1.real(), t2.imag() = q__1.imag();
    cwrk[1].real() = cone.real(), cwrk[1].imag() = cone.imag();
    crfn.real() = cone.real(), crfn.imag() = cone.imag();
    ac = (float)1.;
    l  = 1;
    for(k = 2; k <= 15; ++k)
    {
        s.real() = czero.real(), s.imag() = czero.imag();
        i__1 = k;
        for(j = 1; j <= i__1; ++j)
        {
            ++l;
            q__2.real() = s.real() * t2.real() - s.imag() * t2.imag(), q__2.imag() = s.real() * t2.imag() + s.imag() * t2.real();
            i__2   = l - 1;
            q__3.real() = c__[i__2], q__3.imag() = (float)0.;
            q__1.real() = q__2.real() + q__3.real(), q__1.imag() = q__2.imag() + q__3.imag();
            s.real() = q__1.real(), s.imag() = q__1.imag();
            /* L10: */
        }
        q__1.real() = crfn.real() * sr.real() - crfn.imag() * sr.imag(), q__1.imag() = crfn.real() * sr.imag() + crfn.imag() * sr.real();
        crfn.real() = q__1.real(), crfn.imag() = q__1.imag();
        i__1   = k;
        q__1.real() = crfn.real() * s.real() - crfn.imag() * s.imag(), q__1.imag() = crfn.real() * s.imag() + crfn.imag() * s.real();
        cwrk[i__1].real() = q__1.real(), cwrk[i__1].imag() = q__1.imag();
        ac *= rfn;
        i__1 = k;
        tstr = cwrk[i__1].real();
        tsti = r_imag(&cwrk[k]);
        test = abs(tstr) + abs(tsti);
        if(ac < *tol && test < *tol)
        {
            goto L30;
        }
        /* L20: */
    }
    k = 15;
L30:
    *init = k;
L40:
    if(*ikflg == 2)
    {
        goto L60;
    }
    /* ----------------------------------------------------------------------- */
    /*     COMPUTE SUM FOR THE I FUNCTION */
    /* ----------------------------------------------------------------------- */
    s.real() = czero.real(), s.imag() = czero.imag();
    i__1 = *init;
    for(i__ = 1; i__ <= i__1; ++i__)
    {
        i__2   = i__;
        q__1.real() = s.real() + cwrk[i__2].real(), q__1.imag() = s.imag() + cwrk[i__2].imag();
        s.real() = q__1.real(), s.imag() = q__1.imag();
        /* L50: */
    }
    sum->real() = s.real(), sum->imag() = s.imag();
    q__1.real() = cwrk[16].real() * con[0].real() - cwrk[16].imag() * con[0].imag(), q__1.imag() = cwrk[16].real() * con[0].imag() + cwrk[16].imag() * con[0].real();
    phi->real() = q__1.real(), phi->imag() = q__1.imag();
    return 0;
L60:
    /* ----------------------------------------------------------------------- */
    /*     COMPUTE SUM FOR THE K FUNCTION */
    /* ----------------------------------------------------------------------- */
    s.real() = czero.real(), s.imag() = czero.imag();
    t.real() = cone.real(), t.imag() = cone.imag();
    i__1 = *init;
    for(i__ = 1; i__ <= i__1; ++i__)
    {
        i__2   = i__;
        q__2.real() = t.real() * cwrk[i__2].real() - t.imag() * cwrk[i__2].imag(), q__2.imag() = t.real() * cwrk[i__2].imag() + t.imag() * cwrk[i__2].real();
        q__1.real() = s.real() + q__2.real(), q__1.imag() = s.imag() + q__2.imag();
        s.real() = q__1.real(), s.imag() = q__1.imag();
        q__1.real() = -t.real(), q__1.imag() = -t.imag();
        t.real() = q__1.real(), t.imag() = q__1.imag();
        /* L70: */
    }
    sum->real() = s.real(), sum->imag() = s.imag();
    q__1.real() = cwrk[16].real() * con[1].real() - cwrk[16].imag() * con[1].imag(), q__1.imag() = cwrk[16].real() * con[1].imag() + cwrk[16].imag() * con[1].real();
    phi->real() = q__1.real(), phi->imag() = q__1.imag();
    return 0;
} /* cunik_ */

 int cunk1_(System::Complex<float>* z__, float* fnu, int32* kode, int32* mr, int32* n, System::Complex<float>* y, int32* nz, float* tol, float* elim, float* alim)
{
    /* Initialized data */

    static System::Complex<float> czero = {(float)0., (float)0.};
    static System::Complex<float> cone  = {(float)1., (float)0.};
    static float    pi    = (float)3.14159265358979324;

    /* System generated locals */
    int32 i__1, i__2, i__3;
    float    r__1, r__2;
    System::Complex<float> q__1, q__2, q__3, q__4, q__5;


    
    int32                     i__, j, k, m;
    float                        x;
    System::Complex<float>                     c1, c2, s1, s2;
    int32                     ib, ic;
    System::Complex<float>                     ck;
    float                        fn;
    int32                     il;
    System::Complex<float>                     cs;
    int32                     kk;
    System::Complex<float>                     cy[2];
    int32                     nw;
    System::Complex<float>                     rz, zr;
    float                        c2i, c2m, c2r, rs1, ang;
    System::Complex<float>                     cfn;
    float                        asc, fnf;
    int32                     ifn;
    System::Complex<float>                     phi[2];
    float                        cpn;
    int32                     iuf;
    float                        fmr;
    System::Complex<float>                     csr[3], css[3];
    float                        sgn;
    int32                     inu;
    float                        bry[3], spn;
    System::Complex<float>                     sum[2];
    float                        aphi;
    System::Complex<float>                     cscl, phid, crsc, csgn;
    System::Complex<float>                     cspn;
    int32                     init[2];
    System::Complex<float>                     cwrk[48] /* was [16][3] */, sumd, zeta1[2], zeta2[2];
    int32                     iflag, kflag;
    float                        ascle;
    int32                     kdflg;
   
    int32                     ipard, initd;
   
    System::Complex<float>                     zeta1d, zeta2d;

#define cwrk_subscr(a_1, a_2) (a_2) * 16 + a_1 - 17
#define cwrk_ref(a_1, a_2) cwrk[cwrk_subscr(a_1, a_2)]

    /* *********************************************************************72 */

    /* ***BEGIN PROLOGUE  CUNK1 */
    /* ***REFER TO  CBESK */

    /*     CUNK1 COMPUTES K(FNU,Z) AND ITS ANALYTIC CONTINUATION FROM THE */
    /*     RIGHT HALF PLANE TO THE LEFT HALF PLANE BY MEANS OF THE */
    /*     UNIFORM ASYMPTOTIC EXPANSION. */
    /*     MR INDICATES THE DIRECTION OF ROTATION FOR ANALYTIC CONTINUATION. */
    /*     NZ=-1 MEANS AN OVERFLOW WILL OCCUR */

    /* ***ROUTINES CALLED  CS1S2,CUCHK,CUNIK,R1MACH */
    /* ***END PROLOGUE  CUNK1 */
    /* Parameter adjustments */
    --y;

    /* Function Body */

    kdflg = 1;
    *nz   = 0;
    /* ----------------------------------------------------------------------- */
    /*     EXP(-ALIM)=EXP(-ELIM)/TOL=APPROX. ONE PRECISION GREATER THAN */
    /*     THE UNDERFLOW LIMIT */
    /* ----------------------------------------------------------------------- */
    r__1   = (float)1. / *tol;
    q__1.real() = r__1, q__1.imag() = (float)0.;
    cscl.real() = q__1.real(), cscl.imag() = q__1.imag();
    q__1.real() = *tol, q__1.imag() = (float)0.;
    crsc.real() = q__1.real(), crsc.imag() = q__1.imag();
    css[0].real() = cscl.real(), css[0].imag() = cscl.imag();
    css[1].real() = cone.real(), css[1].imag() = cone.imag();
    css[2].real() = crsc.real(), css[2].imag() = crsc.imag();
    csr[0].real() = crsc.real(), csr[0].imag() = crsc.imag();
    csr[1].real() = cone.real(), csr[1].imag() = cone.imag();
    csr[2].real() = cscl.real(), csr[2].imag() = cscl.imag();
    bry[0] = r1mach_(&c__1) * (float)1e3 / *tol;
    bry[1] = (float)1. / bry[0];
    bry[2] = r1mach_(&c__2);
    x      = z__->real();
    zr.real() = z__->real(), zr.imag() = z__->imag();
    if(x < (float)0.)
    {
        q__1.real() = -z__->real(), q__1.imag() = -z__->imag();
        zr.real() = q__1.real(), zr.imag() = q__1.imag();
    }
    j    = 2;
    i__1 = *n;
    for(i__ = 1; i__ <= i__1; ++i__)
    {
        /* ----------------------------------------------------------------------- */
        /*     J FLIP FLOPS BETWEEN 1 AND 2 IN J = 3 - J */
        /* ----------------------------------------------------------------------- */
        j           = 3 - j;
        fn          = *fnu + (float)(i__ - 1);
        init[j - 1] = 0;
        cunik_(&zr, &fn, &c__2, &c__0, tol, &init[j - 1], &phi[j - 1], &zeta1[j - 1], &zeta2[j - 1], &sum[j - 1], &cwrk_ref(1, j));
        if(*kode == 1)
        {
            goto L20;
        }
        q__1.real() = fn, q__1.imag() = (float)0.;
        cfn.real() = q__1.real(), cfn.imag() = q__1.imag();
        i__2   = j - 1;
        i__3   = j - 1;
        q__4.real() = zr.real() + zeta2[i__3].real(), q__4.imag() = zr.imag() + zeta2[i__3].imag();
        c_div(&q__3, &cfn, &q__4);
        q__2.real() = cfn.real() * q__3.real() - cfn.imag() * q__3.imag(), q__2.imag() = cfn.real() * q__3.imag() + cfn.imag() * q__3.real();
        q__1.real() = zeta1[i__2].real() - q__2.real(), q__1.imag() = zeta1[i__2].imag() - q__2.imag();
        s1.real() = q__1.real(), s1.imag() = q__1.imag();
        goto L30;
    L20:
        i__2   = j - 1;
        i__3   = j - 1;
        q__1.real() = zeta1[i__2].real() - zeta2[i__3].real(), q__1.imag() = zeta1[i__2].imag() - zeta2[i__3].imag();
        s1.real() = q__1.real(), s1.imag() = q__1.imag();
    L30:
        /* ----------------------------------------------------------------------- */
        /*     TEST FOR UNDERFLOW AND OVERFLOW */
        /* ----------------------------------------------------------------------- */
        rs1 = s1.real();
        if(abs(rs1) > *elim)
        {
            goto L60;
        }
        if(kdflg == 1)
        {
            kflag = 2;
        }
        if(abs(rs1) < *alim)
        {
            goto L40;
        }
        /* ----------------------------------------------------------------------- */
        /*     REFINE  TEST AND SCALE */
        /* ----------------------------------------------------------------------- */
        aphi = c_abs(&phi[j - 1]);
        rs1 += log(aphi);
        if(abs(rs1) > *elim)
        {
            goto L60;
        }
        if(kdflg == 1)
        {
            kflag = 1;
        }
        if(rs1 < (float)0.)
        {
            goto L40;
        }
        if(kdflg == 1)
        {
            kflag = 3;
        }
    L40:
        /* ----------------------------------------------------------------------- */
        /*     SCALE S1 TO KEEP INTERMEDIATE ARITHMETIC ON SCALE NEAR */
        /*     EXPONENT EXTREMES */
        /* ----------------------------------------------------------------------- */
        i__2   = j - 1;
        i__3   = j - 1;
        q__1.real() = phi[i__2].real() * sum[i__3].real() - phi[i__2].imag() * sum[i__3].imag(), q__1.imag() = phi[i__2].real() * sum[i__3].imag() + phi[i__2].imag() * sum[i__3].real();
        s2.real() = q__1.real(), s2.imag() = q__1.imag();
        c2r    = s1.real();
        c2i    = r_imag(&s1);
        i__2   = kflag - 1;
        c2m    = exp(c2r) * css[i__2].real();
        q__2.real() = c2m, q__2.imag() = (float)0.;
        r__1   = cos(c2i);
        r__2   = sin(c2i);
        q__3.real() = r__1, q__3.imag() = r__2;
        q__1.real() = q__2.real() * q__3.real() - q__2.imag() * q__3.imag(), q__1.imag() = q__2.real() * q__3.imag() + q__2.imag() * q__3.real();
        s1.real() = q__1.real(), s1.imag() = q__1.imag();
        q__1.real() = s2.real() * s1.real() - s2.imag() * s1.imag(), q__1.imag() = s2.real() * s1.imag() + s2.imag() * s1.real();
        s2.real() = q__1.real(), s2.imag() = q__1.imag();
        if(kflag != 1)
        {
            goto L50;
        }
        cuchk_(&s2, &nw, bry, tol);
        if(nw != 0)
        {
            goto L60;
        }
    L50:
        i__2       = kdflg - 1;
        cy[i__2].real() = s2.real(), cy[i__2].imag() = s2.imag();
        i__2   = i__;
        i__3   = kflag - 1;
        q__1.real() = s2.real() * csr[i__3].real() - s2.imag() * csr[i__3].imag(), q__1.imag() = s2.real() * csr[i__3].imag() + s2.imag() * csr[i__3].real();
        y[i__2].real() = q__1.real(), y[i__2].imag() = q__1.imag();
        if(kdflg == 2)
        {
            goto L75;
        }
        kdflg = 2;
        goto L70;
    L60:
        if(rs1 > (float)0.)
        {
            goto L290;
        }
        /* ----------------------------------------------------------------------- */
        /*     FOR X.LT.0.0, THE I FUNCTION TO BE ADDED WILL OVERFLOW */
        /* ----------------------------------------------------------------------- */
        if(x < (float)0.)
        {
            goto L290;
        }
        kdflg     = 1;
        i__2      = i__;
        y[i__2].real() = czero.real(), y[i__2].imag() = czero.imag();
        ++(*nz);
        if(i__ == 1)
        {
            goto L70;
        }
        i__2 = i__ - 1;
        if(y[i__2].real() == czero.real() && y[i__2].imag() == czero.imag())
        {
            goto L70;
        }
        i__2      = i__ - 1;
        y[i__2].real() = czero.real(), y[i__2].imag() = czero.imag();
        ++(*nz);
    L70:;
    }
    i__ = *n;
L75:
    c_div(&q__1, &c_b17, &zr);
    rz.real() = q__1.real(), rz.imag() = q__1.imag();
    q__2.real() = fn, q__2.imag() = (float)0.;
    q__1.real() = q__2.real() * rz.real() - q__2.imag() * rz.imag(), q__1.imag() = q__2.real() * rz.imag() + q__2.imag() * rz.real();
    ck.real() = q__1.real(), ck.imag() = q__1.imag();
    ib = i__ + 1;
    if(*n < ib)
    {
        goto L160;
    }
    /* ----------------------------------------------------------------------- */
    /*     TEST LAST MEMBER FOR UNDERFLOW AND OVERFLOW, SET SEQUENCE TO ZERO */
    /*     ON UNDERFLOW */
    /* ----------------------------------------------------------------------- */
    fn    = *fnu + (float)(*n - 1);
    ipard = 1;
    if(*mr != 0)
    {
        ipard = 0;
    }
    initd = 0;
    cunik_(&zr, &fn, &c__2, &ipard, tol, &initd, &phid, &zeta1d, &zeta2d, &sumd, &cwrk_ref(1, 3));
    if(*kode == 1)
    {
        goto L80;
    }
    q__1.real() = fn, q__1.imag() = (float)0.;
    cfn.real() = q__1.real(), cfn.imag() = q__1.imag();
    q__4.real() = zr.real() + zeta2d.real(), q__4.imag() = zr.imag() + zeta2d.imag();
    c_div(&q__3, &cfn, &q__4);
    q__2.real() = cfn.real() * q__3.real() - cfn.imag() * q__3.imag(), q__2.imag() = cfn.real() * q__3.imag() + cfn.imag() * q__3.real();
    q__1.real() = zeta1d.real() - q__2.real(), q__1.imag() = zeta1d.imag() - q__2.imag();
    s1.real() = q__1.real(), s1.imag() = q__1.imag();
    goto L90;
L80:
    q__1.real() = zeta1d.real() - zeta2d.real(), q__1.imag() = zeta1d.imag() - zeta2d.imag();
    s1.real() = q__1.real(), s1.imag() = q__1.imag();
L90:
    rs1 = s1.real();
    if(abs(rs1) > *elim)
    {
        goto L95;
    }
    if(abs(rs1) < *alim)
    {
        goto L100;
    }
    /* ----------------------------------------------------------------------- */
    /*     REFINE ESTIMATE AND TEST */
    /* ----------------------------------------------------------------------- */
    aphi = c_abs(&phid);
    rs1 += log(aphi);
    if(abs(rs1) < *elim)
    {
        goto L100;
    }
L95:
    if(rs1 > (float)0.)
    {
        goto L290;
    }
    /* ----------------------------------------------------------------------- */
    /*     FOR X.LT.0.0, THE I FUNCTION TO BE ADDED WILL OVERFLOW */
    /* ----------------------------------------------------------------------- */
    if(x < (float)0.)
    {
        goto L290;
    }
    *nz  = *n;
    i__1 = *n;
    for(i__ = 1; i__ <= i__1; ++i__)
    {
        i__2      = i__;
        y[i__2].real() = czero.real(), y[i__2].imag() = czero.imag();
        /* L96: */
    }
    return 0;
L100:
    /* ----------------------------------------------------------------------- */
    /*     RECUR FORWARD FOR REMAINDER OF THE SEQUENCE */
    /* ----------------------------------------------------------------------- */
    s1.real() = cy[0].real(), s1.imag() = cy[0].imag();
    s2.real() = cy[1].real(), s2.imag() = cy[1].imag();
    i__1 = kflag - 1;
    c1.real() = csr[i__1].real(), c1.imag() = csr[i__1].imag();
    ascle = bry[kflag - 1];
    i__1  = *n;
    for(i__ = ib; i__ <= i__1; ++i__)
    {
        c2.real() = s2.real(), c2.imag() = s2.imag();
        q__2.real() = ck.real() * s2.real() - ck.imag() * s2.imag(), q__2.imag() = ck.real() * s2.imag() + ck.imag() * s2.real();
        q__1.real() = q__2.real() + s1.real(), q__1.imag() = q__2.imag() + s1.imag();
        s2.real() = q__1.real(), s2.imag() = q__1.imag();
        s1.real() = c2.real(), s1.imag() = c2.imag();
        q__1.real() = ck.real() + rz.real(), q__1.imag() = ck.imag() + rz.imag();
        ck.real() = q__1.real(), ck.imag() = q__1.imag();
        q__1.real() = s2.real() * c1.real() - s2.imag() * c1.imag(), q__1.imag() = s2.real() * c1.imag() + s2.imag() * c1.real();
        c2.real() = q__1.real(), c2.imag() = q__1.imag();
        i__2      = i__;
        y[i__2].real() = c2.real(), y[i__2].imag() = c2.imag();
        if(kflag >= 3)
        {
            goto L120;
        }
        c2r = c2.real();
        c2i = r_imag(&c2);
        c2r = abs(c2r);
        c2i = abs(c2i);
        c2m = max(c2r, c2i);
        if(c2m <= ascle)
        {
            goto L120;
        }
        ++kflag;
        ascle  = bry[kflag - 1];
        q__1.real() = s1.real() * c1.real() - s1.imag() * c1.imag(), q__1.imag() = s1.real() * c1.imag() + s1.imag() * c1.real();
        s1.real() = q__1.real(), s1.imag() = q__1.imag();
        s2.real() = c2.real(), s2.imag() = c2.imag();
        i__2   = kflag - 1;
        q__1.real() = s1.real() * css[i__2].real() - s1.imag() * css[i__2].imag(), q__1.imag() = s1.real() * css[i__2].imag() + s1.imag() * css[i__2].real();
        s1.real() = q__1.real(), s1.imag() = q__1.imag();
        i__2   = kflag - 1;
        q__1.real() = s2.real() * css[i__2].real() - s2.imag() * css[i__2].imag(), q__1.imag() = s2.real() * css[i__2].imag() + s2.imag() * css[i__2].real();
        s2.real() = q__1.real(), s2.imag() = q__1.imag();
        i__2 = kflag - 1;
        c1.real() = csr[i__2].real(), c1.imag() = csr[i__2].imag();
    L120:;
    }
L160:
    if(*mr == 0)
    {
        return 0;
    }
    /* ----------------------------------------------------------------------- */
    /*     ANALYTIC CONTINUATION FOR RE(Z).LT.0.0E0 */
    /* ----------------------------------------------------------------------- */
    *nz = 0;
    fmr = (float)(*mr);
    sgn = -r_sign(&pi, &fmr);
    /* ----------------------------------------------------------------------- */
    /*     CSPN AND CSGN ARE COEFF OF K AND I FUNCIONS RESP. */
    /* ----------------------------------------------------------------------- */
    q__1.real() = (float)0., q__1.imag() = sgn;
    csgn.real() = q__1.real(), csgn.imag() = q__1.imag();
    inu    = (int32)(*fnu);
    fnf    = *fnu - (float)inu;
    ifn    = inu + *n - 1;
    ang    = fnf * sgn;
    cpn    = cos(ang);
    spn    = sin(ang);
    q__1.real() = cpn, q__1.imag() = spn;
    cspn.real() = q__1.real(), cspn.imag() = q__1.imag();
    if(ifn % 2 == 1)
    {
        q__1.real() = -cspn.real(), q__1.imag() = -cspn.imag();
        cspn.real() = q__1.real(), cspn.imag() = q__1.imag();
    }
    asc   = bry[0];
    kk    = *n;
    iuf   = 0;
    kdflg = 1;
    --ib;
    ic   = ib - 1;
    i__1 = *n;
    for(k = 1; k <= i__1; ++k)
    {
        fn = *fnu + (float)(kk - 1);
        /* ----------------------------------------------------------------------- */
        /*     LOGIC TO SORT OUT CASES WHOSE PARAMETERS WERE SET FOR THE K */
        /*     FUNCTION ABOVE */
        /* ----------------------------------------------------------------------- */
        m = 3;
        if(*n > 2)
        {
            goto L175;
        }
    L170:
        initd  = init[j - 1];
        i__2   = j - 1;
        phid.real() = phi[i__2].real(), phid.imag() = phi[i__2].imag();
        i__2     = j - 1;
        zeta1d.real() = zeta1[i__2].real(), zeta1d.imag() = zeta1[i__2].imag();
        i__2     = j - 1;
        zeta2d.real() = zeta2[i__2].real(), zeta2d.imag() = zeta2[i__2].imag();
        i__2   = j - 1;
        sumd.real() = sum[i__2].real(), sumd.imag() = sum[i__2].imag();
        m = j;
        j = 3 - j;
        goto L180;
    L175:
        if(kk == *n && ib < *n)
        {
            goto L180;
        }
        if(kk == ib || kk == ic)
        {
            goto L170;
        }
        initd = 0;
    L180:
        cunik_(&zr, &fn, &c__1, &c__0, tol, &initd, &phid, &zeta1d, &zeta2d, &sumd, &cwrk_ref(1, m));
        if(*kode == 1)
        {
            goto L190;
        }
        q__1.real() = fn, q__1.imag() = (float)0.;
        cfn.real() = q__1.real(), cfn.imag() = q__1.imag();
        q__2.real() = -zeta1d.real(), q__2.imag() = -zeta1d.imag();
        q__5.real() = zr.real() + zeta2d.real(), q__5.imag() = zr.imag() + zeta2d.imag();
        c_div(&q__4, &cfn, &q__5);
        q__3.real() = cfn.real() * q__4.real() - cfn.imag() * q__4.imag(), q__3.imag() = cfn.real() * q__4.imag() + cfn.imag() * q__4.real();
        q__1.real() = q__2.real() + q__3.real(), q__1.imag() = q__2.imag() + q__3.imag();
        s1.real() = q__1.real(), s1.imag() = q__1.imag();
        goto L200;
    L190:
        q__2.real() = -zeta1d.real(), q__2.imag() = -zeta1d.imag();
        q__1.real() = q__2.real() + zeta2d.real(), q__1.imag() = q__2.imag() + zeta2d.imag();
        s1.real() = q__1.real(), s1.imag() = q__1.imag();
    L200:
        /* ----------------------------------------------------------------------- */
        /*     TEST FOR UNDERFLOW AND OVERFLOW */
        /* ----------------------------------------------------------------------- */
        rs1 = s1.real();
        if(abs(rs1) > *elim)
        {
            goto L250;
        }
        if(kdflg == 1)
        {
            iflag = 2;
        }
        if(abs(rs1) < *alim)
        {
            goto L210;
        }
        /* ----------------------------------------------------------------------- */
        /*     REFINE  TEST AND SCALE */
        /* ----------------------------------------------------------------------- */
        aphi = c_abs(&phid);
        rs1 += log(aphi);
        if(abs(rs1) > *elim)
        {
            goto L250;
        }
        if(kdflg == 1)
        {
            iflag = 1;
        }
        if(rs1 < (float)0.)
        {
            goto L210;
        }
        if(kdflg == 1)
        {
            iflag = 3;
        }
    L210:
        q__2.real() = csgn.real() * phid.real() - csgn.imag() * phid.imag(), q__2.imag() = csgn.real() * phid.imag() + csgn.imag() * phid.real();
        q__1.real() = q__2.real() * sumd.real() - q__2.imag() * sumd.imag(), q__1.imag() = q__2.real() * sumd.imag() + q__2.imag() * sumd.real();
        s2.real() = q__1.real(), s2.imag() = q__1.imag();
        c2r    = s1.real();
        c2i    = r_imag(&s1);
        i__2   = iflag - 1;
        c2m    = exp(c2r) * css[i__2].real();
        q__2.real() = c2m, q__2.imag() = (float)0.;
        r__1   = cos(c2i);
        r__2   = sin(c2i);
        q__3.real() = r__1, q__3.imag() = r__2;
        q__1.real() = q__2.real() * q__3.real() - q__2.imag() * q__3.imag(), q__1.imag() = q__2.real() * q__3.imag() + q__2.imag() * q__3.real();
        s1.real() = q__1.real(), s1.imag() = q__1.imag();
        q__1.real() = s2.real() * s1.real() - s2.imag() * s1.imag(), q__1.imag() = s2.real() * s1.imag() + s2.imag() * s1.real();
        s2.real() = q__1.real(), s2.imag() = q__1.imag();
        if(iflag != 1)
        {
            goto L220;
        }
        cuchk_(&s2, &nw, bry, tol);
        if(nw != 0)
        {
            s2.real() = (float)0., s2.imag() = (float)0.;
        }
    L220:
        i__2       = kdflg - 1;
        cy[i__2].real() = s2.real(), cy[i__2].imag() = s2.imag();
        c2.real() = s2.real(), c2.imag() = s2.imag();
        i__2   = iflag - 1;
        q__1.real() = s2.real() * csr[i__2].real() - s2.imag() * csr[i__2].imag(), q__1.imag() = s2.real() * csr[i__2].imag() + s2.imag() * csr[i__2].real();
        s2.real() = q__1.real(), s2.imag() = q__1.imag();
        /* ----------------------------------------------------------------------- */
        /*     ADD I AND K FUNCTIONS, K SEQUENCE IN Y(I), I=1,N */
        /* ----------------------------------------------------------------------- */
        i__2 = kk;
        s1.real() = y[i__2].real(), s1.imag() = y[i__2].imag();
        if(*kode == 1)
        {
            goto L240;
        }
        cs1s2_(&zr, &s1, &s2, &nw, &asc, alim, &iuf);
        *nz += nw;
    L240:
        i__2   = kk;
        q__2.real() = s1.real() * cspn.real() - s1.imag() * cspn.imag(), q__2.imag() = s1.real() * cspn.imag() + s1.imag() * cspn.real();
        q__1.real() = q__2.real() + s2.real(), q__1.imag() = q__2.imag() + s2.imag();
        y[i__2].real() = q__1.real(), y[i__2].imag() = q__1.imag();
        --kk;
        q__1.real() = -cspn.real(), q__1.imag() = -cspn.imag();
        cspn.real() = q__1.real(), cspn.imag() = q__1.imag();
        if(c2.real() != czero.real() || c2.imag() != czero.imag())
        {
            goto L245;
        }
        kdflg = 1;
        goto L260;
    L245:
        if(kdflg == 2)
        {
            goto L265;
        }
        kdflg = 2;
        goto L260;
    L250:
        if(rs1 > (float)0.)
        {
            goto L290;
        }
        s2.real() = czero.real(), s2.imag() = czero.imag();
        goto L220;
    L260:;
    }
    k = *n;
L265:
    il = *n - k;
    if(il == 0)
    {
        return 0;
    }
    /* ----------------------------------------------------------------------- */
    /*     RECUR BACKWARD FOR REMAINDER OF I SEQUENCE AND ADD IN THE */
    /*     K FUNCTIONS, SCALING THE I SEQUENCE DURING RECURRENCE TO KEEP */
    /*     INTERMEDIATE ARITHMETIC ON SCALE NEAR EXPONENT EXTREMES. */
    /* ----------------------------------------------------------------------- */
    s1.real() = cy[0].real(), s1.imag() = cy[0].imag();
    s2.real() = cy[1].real(), s2.imag() = cy[1].imag();
    i__1 = iflag - 1;
    cs.real() = csr[i__1].real(), cs.imag() = csr[i__1].imag();
    ascle = bry[iflag - 1];
    fn    = (float)(inu + il);
    i__1  = il;
    for(i__ = 1; i__ <= i__1; ++i__)
    {
        c2.real() = s2.real(), c2.imag() = s2.imag();
        r__1   = fn + fnf;
        q__4.real() = r__1, q__4.imag() = (float)0.;
        q__3.real() = q__4.real() * rz.real() - q__4.imag() * rz.imag(), q__3.imag() = q__4.real() * rz.imag() + q__4.imag() * rz.real();
        q__2.real() = q__3.real() * s2.real() - q__3.imag() * s2.imag(), q__2.imag() = q__3.real() * s2.imag() + q__3.imag() * s2.real();
        q__1.real() = s1.real() + q__2.real(), q__1.imag() = s1.imag() + q__2.imag();
        s2.real() = q__1.real(), s2.imag() = q__1.imag();
        s1.real() = c2.real(), s1.imag() = c2.imag();
        fn += (float)-1.;
        q__1.real() = s2.real() * cs.real() - s2.imag() * cs.imag(), q__1.imag() = s2.real() * cs.imag() + s2.imag() * cs.real();
        c2.real() = q__1.real(), c2.imag() = q__1.imag();
        ck.real() = c2.real(), ck.imag() = c2.imag();
        i__2 = kk;
        c1.real() = y[i__2].real(), c1.imag() = y[i__2].imag();
        if(*kode == 1)
        {
            goto L270;
        }
        cs1s2_(&zr, &c1, &c2, &nw, &asc, alim, &iuf);
        *nz += nw;
    L270:
        i__2   = kk;
        q__2.real() = c1.real() * cspn.real() - c1.imag() * cspn.imag(), q__2.imag() = c1.real() * cspn.imag() + c1.imag() * cspn.real();
        q__1.real() = q__2.real() + c2.real(), q__1.imag() = q__2.imag() + c2.imag();
        y[i__2].real() = q__1.real(), y[i__2].imag() = q__1.imag();
        --kk;
        q__1.real() = -cspn.real(), q__1.imag() = -cspn.imag();
        cspn.real() = q__1.real(), cspn.imag() = q__1.imag();
        if(iflag >= 3)
        {
            goto L280;
        }
        c2r = ck.real();
        c2i = r_imag(&ck);
        c2r = abs(c2r);
        c2i = abs(c2i);
        c2m = max(c2r, c2i);
        if(c2m <= ascle)
        {
            goto L280;
        }
        ++iflag;
        ascle  = bry[iflag - 1];
        q__1.real() = s1.real() * cs.real() - s1.imag() * cs.imag(), q__1.imag() = s1.real() * cs.imag() + s1.imag() * cs.real();
        s1.real() = q__1.real(), s1.imag() = q__1.imag();
        s2.real() = ck.real(), s2.imag() = ck.imag();
        i__2   = iflag - 1;
        q__1.real() = s1.real() * css[i__2].real() - s1.imag() * css[i__2].imag(), q__1.imag() = s1.real() * css[i__2].imag() + s1.imag() * css[i__2].real();
        s1.real() = q__1.real(), s1.imag() = q__1.imag();
        i__2   = iflag - 1;
        q__1.real() = s2.real() * css[i__2].real() - s2.imag() * css[i__2].imag(), q__1.imag() = s2.real() * css[i__2].imag() + s2.imag() * css[i__2].real();
        s2.real() = q__1.real(), s2.imag() = q__1.imag();
        i__2 = iflag - 1;
        cs.real() = csr[i__2].real(), cs.imag() = csr[i__2].imag();
    L280:;
    }
    return 0;
L290:
    *nz = -1;
    return 0;
} /* cunk1_ */

#undef cwrk_ref
#undef cwrk_subscr

 int cunk2_(System::Complex<float>* z__, float* fnu, int32* kode, int32* mr, int32* n, System::Complex<float>* y, int32* nz, float* tol, float* elim, float* alim)
{
    /* Initialized data */

    static System::Complex<float> czero  = {(float)0., (float)0.};
    static System::Complex<float> cone   = {(float)1., (float)0.};
    static System::Complex<float> ci     = {(float)0., (float)1.};
    static System::Complex<float> cr1    = {(float)1., (float)1.73205080756887729};
    static System::Complex<float> cr2    = {(float)-.5, (float)-.866025403784438647};
    static float    hpi    = (float)1.57079632679489662;
    static float    pi     = (float)3.14159265358979324;
    static float    aic    = (float)1.26551212348464539;
    static System::Complex<float> cip[4] = {{(float)1., (float)0.}, {(float)0., (float)-1.}, {(float)-1., (float)0.}, {(float)0., (float)1.}};

    /* System generated locals */
    int32 i__1, i__2, i__3, i__4;
    float    r__1, r__2;
    System::Complex<float> q__1, q__2, q__3, q__4, q__5, q__6;


    
    int32                     i__, j, k;
    float                        x;
    System::Complex<float>                     c1, c2, s1, s2, ai;
    int32                     ib, ic;
    System::Complex<float>                     ck;
    float                        fn;
    int32                     il;
    System::Complex<float>                     cs;
    int32                     in, kk;
    System::Complex<float>                     cy[2], zb;
    int32                     nw;
    System::Complex<float>                     zn, rz, zr;
    float                        yy, c2i, c2m, c2r, rs1;
    System::Complex<float>                     dai;
    float                        ang;
    System::Complex<float>                     cfn;
    float                        asc, car;
    System::Complex<float>                     arg[2];
    float                        fnf;
    int32                     ifn, nai;
    System::Complex<float>                     phi[2];
    float                        cpn;
    int32                     iuf;
    float                        fmr, sar;
    System::Complex<float>                     csr[3], css[3];
    float                        sgn;
    int32                     inu;
    float                        bry[3], spn, aarg;
    int32                     ndai;
    System::Complex<float>                     argd;
    float                        aphi;
    System::Complex<float>                     cscl, phid, crsc, csgn;
    int32                     idum;
    System::Complex<float>                     cspn, asum[2], bsum[2], zeta1[2], zeta2[2];
    int32                     iflag, kflag;
    float                        ascle;
    int32                     kdflg;
    int32                     ipard;
    System::Complex<float>           asumd, bsumd;
    
    System::Complex<float>           zeta1d, zeta2d;

    /* *********************************************************************72 */

    /* ***BEGIN PROLOGUE  CUNK2 */
    /* ***REFER TO  CBESK */

    /*     CUNK2 COMPUTES K(FNU,Z) AND ITS ANALYTIC CONTINUATION FROM THE */
    /*     RIGHT HALF PLANE TO THE LEFT HALF PLANE BY MEANS OF THE */
    /*     UNIFORM ASYMPTOTIC EXPANSIONS FOR H(KIND,FNU,ZN) AND J(FNU,ZN) */
    /*     WHERE ZN IS IN THE RIGHT HALF PLANE, KIND=(3-MR)/2, MR=+1 OR */
    /*     -1. HERE ZN=ZR*I OR -ZR*I WHERE ZR=Z IF Z IS IN THE RIGHT */
    /*     HALF PLANE OR ZR=-Z IF Z IS IN THE LEFT HALF PLANE. MR INDIC- */
    /*     ATES THE DIRECTION OF ROTATION FOR ANALYTIC CONTINUATION. */
    /*     NZ=-1 MEANS AN OVERFLOW WILL OCCUR */

    /* ***ROUTINES CALLED  CAIRY,CS1S2,CUCHK,CUNHJ,R1MACH */
    /* ***END PROLOGUE  CUNK2 */
    /* Parameter adjustments */
    --y;

    /* Function Body */

    kdflg = 1;
    *nz   = 0;
    /* ----------------------------------------------------------------------- */
    /*     EXP(-ALIM)=EXP(-ELIM)/TOL=APPROX. ONE PRECISION GREATER THAN */
    /*     THE UNDERFLOW LIMIT */
    /* ----------------------------------------------------------------------- */
    r__1   = (float)1. / *tol;
    q__1.real() = r__1, q__1.imag() = (float)0.;
    cscl.real() = q__1.real(), cscl.imag() = q__1.imag();
    q__1.real() = *tol, q__1.imag() = (float)0.;
    crsc.real() = q__1.real(), crsc.imag() = q__1.imag();
    css[0].real() = cscl.real(), css[0].imag() = cscl.imag();
    css[1].real() = cone.real(), css[1].imag() = cone.imag();
    css[2].real() = crsc.real(), css[2].imag() = crsc.imag();
    csr[0].real() = crsc.real(), csr[0].imag() = crsc.imag();
    csr[1].real() = cone.real(), csr[1].imag() = cone.imag();
    csr[2].real() = cscl.real(), csr[2].imag() = cscl.imag();
    bry[0] = r1mach_(&c__1) * (float)1e3 / *tol;
    bry[1] = (float)1. / bry[0];
    bry[2] = r1mach_(&c__2);
    x      = z__->real();
    zr.real() = z__->real(), zr.imag() = z__->imag();
    if(x < (float)0.)
    {
        q__1.real() = -z__->real(), q__1.imag() = -z__->imag();
        zr.real() = q__1.real(), zr.imag() = q__1.imag();
    }
    yy     = r_imag(&zr);
    q__2.real() = -zr.real(), q__2.imag() = -zr.imag();
    q__1.real() = q__2.real() * ci.real() - q__2.imag() * ci.imag(), q__1.imag() = q__2.real() * ci.imag() + q__2.imag() * ci.real();
    zn.real() = q__1.real(), zn.imag() = q__1.imag();
    zb.real() = zr.real(), zb.imag() = zr.imag();
    inu    = (int32)(*fnu);
    fnf    = *fnu - (float)inu;
    ang    = -hpi * fnf;
    car    = cos(ang);
    sar    = sin(ang);
    cpn    = -hpi * car;
    spn    = -hpi * sar;
    r__1   = -spn;
    q__1.real() = r__1, q__1.imag() = cpn;
    c2.real() = q__1.real(), c2.imag() = q__1.imag();
    kk     = inu % 4 + 1;
    q__2.real() = cr1.real() * c2.real() - cr1.imag() * c2.imag(), q__2.imag() = cr1.real() * c2.imag() + cr1.imag() * c2.real();
    i__1   = kk - 1;
    q__1.real() = q__2.real() * cip[i__1].real() - q__2.imag() * cip[i__1].imag(), q__1.imag() = q__2.real() * cip[i__1].imag() + q__2.imag() * cip[i__1].real();
    cs.real() = q__1.real(), cs.imag() = q__1.imag();
    if(yy > (float)0.)
    {
        goto L10;
    }
    q__2.real() = -zn.real(), q__2.imag() = -zn.imag();
    r_cnjg(&q__1, &q__2);
    zn.real() = q__1.real(), zn.imag() = q__1.imag();
    r_cnjg(&q__1, &zb);
    zb.real() = q__1.real(), zb.imag() = q__1.imag();
L10:
    /* ----------------------------------------------------------------------- */
    /*     K(FNU,Z) IS COMPUTED FROM H(2,FNU,-I*Z) WHERE Z IS IN THE FIRST */
    /*     QUADRANT. FOURTH QUADRANT VALUES (YY.LE.0.0E0) ARE COMPUTED BY */
    /*     CONJUGATION SINCE THE K FUNCTION IS REAL ON THE POSITIVE REAL AXIS */
    /* ----------------------------------------------------------------------- */
    j    = 2;
    i__1 = *n;
    for(i__ = 1; i__ <= i__1; ++i__)
    {
        /* ----------------------------------------------------------------------- */
        /*     J FLIP FLOPS BETWEEN 1 AND 2 IN J = 3 - J */
        /* ----------------------------------------------------------------------- */
        j  = 3 - j;
        fn = *fnu + (float)(i__ - 1);
        cunhj_(&zn, &fn, &c__0, tol, &phi[j - 1], &arg[j - 1], &zeta1[j - 1], &zeta2[j - 1], &asum[j - 1], &bsum[j - 1]);
        if(*kode == 1)
        {
            goto L20;
        }
        q__1.real() = fn, q__1.imag() = (float)0.;
        cfn.real() = q__1.real(), cfn.imag() = q__1.imag();
        i__2   = j - 1;
        i__3   = j - 1;
        q__4.real() = zb.real() + zeta2[i__3].real(), q__4.imag() = zb.imag() + zeta2[i__3].imag();
        c_div(&q__3, &cfn, &q__4);
        q__2.real() = cfn.real() * q__3.real() - cfn.imag() * q__3.imag(), q__2.imag() = cfn.real() * q__3.imag() + cfn.imag() * q__3.real();
        q__1.real() = zeta1[i__2].real() - q__2.real(), q__1.imag() = zeta1[i__2].imag() - q__2.imag();
        s1.real() = q__1.real(), s1.imag() = q__1.imag();
        goto L30;
    L20:
        i__2   = j - 1;
        i__3   = j - 1;
        q__1.real() = zeta1[i__2].real() - zeta2[i__3].real(), q__1.imag() = zeta1[i__2].imag() - zeta2[i__3].imag();
        s1.real() = q__1.real(), s1.imag() = q__1.imag();
    L30:
        /* ----------------------------------------------------------------------- */
        /*     TEST FOR UNDERFLOW AND OVERFLOW */
        /* ----------------------------------------------------------------------- */
        rs1 = s1.real();
        if(abs(rs1) > *elim)
        {
            goto L60;
        }
        if(kdflg == 1)
        {
            kflag = 2;
        }
        if(abs(rs1) < *alim)
        {
            goto L40;
        }
        /* ----------------------------------------------------------------------- */
        /*     REFINE  TEST AND SCALE */
        /* ----------------------------------------------------------------------- */
        aphi = c_abs(&phi[j - 1]);
        aarg = c_abs(&arg[j - 1]);
        rs1  = rs1 + log(aphi) - log(aarg) * (float).25 - aic;
        if(abs(rs1) > *elim)
        {
            goto L60;
        }
        if(kdflg == 1)
        {
            kflag = 1;
        }
        if(rs1 < (float)0.)
        {
            goto L40;
        }
        if(kdflg == 1)
        {
            kflag = 3;
        }
    L40:
        /* ----------------------------------------------------------------------- */
        /*     SCALE S1 TO KEEP INTERMEDIATE ARITHMETIC ON SCALE NEAR */
        /*     EXPONENT EXTREMES */
        /* ----------------------------------------------------------------------- */
        i__2   = j - 1;
        q__1.real() = arg[i__2].real() * cr2.real() - arg[i__2].imag() * cr2.imag(), q__1.imag() = arg[i__2].real() * cr2.imag() + arg[i__2].imag() * cr2.real();
        c2.real() = q__1.real(), c2.imag() = q__1.imag();
        cairy_(&c2, &c__0, &c__2, &ai, &nai, &idum);
        cairy_(&c2, &c__1, &c__2, &dai, &ndai, &idum);
        i__2   = j - 1;
        q__2.real() = cs.real() * phi[i__2].real() - cs.imag() * phi[i__2].imag(), q__2.imag() = cs.real() * phi[i__2].imag() + cs.imag() * phi[i__2].real();
        i__3   = j - 1;
        q__4.real() = ai.real() * asum[i__3].real() - ai.imag() * asum[i__3].imag(), q__4.imag() = ai.real() * asum[i__3].imag() + ai.imag() * asum[i__3].real();
        q__6.real() = cr2.real() * dai.real() - cr2.imag() * dai.imag(), q__6.imag() = cr2.real() * dai.imag() + cr2.imag() * dai.real();
        i__4   = j - 1;
        q__5.real() = q__6.real() * bsum[i__4].real() - q__6.imag() * bsum[i__4].imag(), q__5.imag() = q__6.real() * bsum[i__4].imag() + q__6.imag() * bsum[i__4].real();
        q__3.real() = q__4.real() + q__5.real(), q__3.imag() = q__4.imag() + q__5.imag();
        q__1.real() = q__2.real() * q__3.real() - q__2.imag() * q__3.imag(), q__1.imag() = q__2.real() * q__3.imag() + q__2.imag() * q__3.real();
        s2.real() = q__1.real(), s2.imag() = q__1.imag();
        c2r    = s1.real();
        c2i    = r_imag(&s1);
        i__2   = kflag - 1;
        c2m    = exp(c2r) * css[i__2].real();
        q__2.real() = c2m, q__2.imag() = (float)0.;
        r__1   = cos(c2i);
        r__2   = sin(c2i);
        q__3.real() = r__1, q__3.imag() = r__2;
        q__1.real() = q__2.real() * q__3.real() - q__2.imag() * q__3.imag(), q__1.imag() = q__2.real() * q__3.imag() + q__2.imag() * q__3.real();
        s1.real() = q__1.real(), s1.imag() = q__1.imag();
        q__1.real() = s2.real() * s1.real() - s2.imag() * s1.imag(), q__1.imag() = s2.real() * s1.imag() + s2.imag() * s1.real();
        s2.real() = q__1.real(), s2.imag() = q__1.imag();
        if(kflag != 1)
        {
            goto L50;
        }
        cuchk_(&s2, &nw, bry, tol);
        if(nw != 0)
        {
            goto L60;
        }
    L50:
        if(yy <= (float)0.)
        {
            r_cnjg(&q__1, &s2);
            s2.real() = q__1.real(), s2.imag() = q__1.imag();
        }
        i__2       = kdflg - 1;
        cy[i__2].real() = s2.real(), cy[i__2].imag() = s2.imag();
        i__2   = i__;
        i__3   = kflag - 1;
        q__1.real() = s2.real() * csr[i__3].real() - s2.imag() * csr[i__3].imag(), q__1.imag() = s2.real() * csr[i__3].imag() + s2.imag() * csr[i__3].real();
        y[i__2].real() = q__1.real(), y[i__2].imag() = q__1.imag();
        q__2.real() = -ci.real(), q__2.imag() = -ci.imag();
        q__1.real() = q__2.real() * cs.real() - q__2.imag() * cs.imag(), q__1.imag() = q__2.real() * cs.imag() + q__2.imag() * cs.real();
        cs.real() = q__1.real(), cs.imag() = q__1.imag();
        if(kdflg == 2)
        {
            goto L75;
        }
        kdflg = 2;
        goto L70;
    L60:
        if(rs1 > (float)0.)
        {
            goto L300;
        }
        /* ----------------------------------------------------------------------- */
        /*     FOR X.LT.0.0, THE I FUNCTION TO BE ADDED WILL OVERFLOW */
        /* ----------------------------------------------------------------------- */
        if(x < (float)0.)
        {
            goto L300;
        }
        kdflg     = 1;
        i__2      = i__;
        y[i__2].real() = czero.real(), y[i__2].imag() = czero.imag();
        q__2.real() = -ci.real(), q__2.imag() = -ci.imag();
        q__1.real() = q__2.real() * cs.real() - q__2.imag() * cs.imag(), q__1.imag() = q__2.real() * cs.imag() + q__2.imag() * cs.real();
        cs.real() = q__1.real(), cs.imag() = q__1.imag();
        ++(*nz);
        if(i__ == 1)
        {
            goto L70;
        }
        i__2 = i__ - 1;
        if(y[i__2].real() == czero.real() && y[i__2].imag() == czero.imag())
        {
            goto L70;
        }
        i__2      = i__ - 1;
        y[i__2].real() = czero.real(), y[i__2].imag() = czero.imag();
        ++(*nz);
    L70:;
    }
    i__ = *n;
L75:
    c_div(&q__1, &c_b17, &zr);
    rz.real() = q__1.real(), rz.imag() = q__1.imag();
    q__2.real() = fn, q__2.imag() = (float)0.;
    q__1.real() = q__2.real() * rz.real() - q__2.imag() * rz.imag(), q__1.imag() = q__2.real() * rz.imag() + q__2.imag() * rz.real();
    ck.real() = q__1.real(), ck.imag() = q__1.imag();
    ib = i__ + 1;
    if(*n < ib)
    {
        goto L170;
    }
    /* ----------------------------------------------------------------------- */
    /*     TEST LAST MEMBER FOR UNDERFLOW AND OVERFLOW, SET SEQUENCE TO ZERO */
    /*     ON UNDERFLOW */
    /* ----------------------------------------------------------------------- */
    fn    = *fnu + (float)(*n - 1);
    ipard = 1;
    if(*mr != 0)
    {
        ipard = 0;
    }
    cunhj_(&zn, &fn, &ipard, tol, &phid, &argd, &zeta1d, &zeta2d, &asumd, &bsumd);
    if(*kode == 1)
    {
        goto L80;
    }
    q__1.real() = fn, q__1.imag() = (float)0.;
    cfn.real() = q__1.real(), cfn.imag() = q__1.imag();
    q__4.real() = zb.real() + zeta2d.real(), q__4.imag() = zb.imag() + zeta2d.imag();
    c_div(&q__3, &cfn, &q__4);
    q__2.real() = cfn.real() * q__3.real() - cfn.imag() * q__3.imag(), q__2.imag() = cfn.real() * q__3.imag() + cfn.imag() * q__3.real();
    q__1.real() = zeta1d.real() - q__2.real(), q__1.imag() = zeta1d.imag() - q__2.imag();
    s1.real() = q__1.real(), s1.imag() = q__1.imag();
    goto L90;
L80:
    q__1.real() = zeta1d.real() - zeta2d.real(), q__1.imag() = zeta1d.imag() - zeta2d.imag();
    s1.real() = q__1.real(), s1.imag() = q__1.imag();
L90:
    rs1 = s1.real();
    if(abs(rs1) > *elim)
    {
        goto L95;
    }
    if(abs(rs1) < *alim)
    {
        goto L100;
    }
    /* ----------------------------------------------------------------------- */
    /*     REFINE ESTIMATE AND TEST */
    /* ----------------------------------------------------------------------- */
    aphi = c_abs(&phid);
    aarg = c_abs(&argd);
    rs1  = rs1 + log(aphi) - log(aarg) * (float).25 - aic;
    if(abs(rs1) < *elim)
    {
        goto L100;
    }
L95:
    if(rs1 > (float)0.)
    {
        goto L300;
    }
    /* ----------------------------------------------------------------------- */
    /*     FOR X.LT.0.0, THE I FUNCTION TO BE ADDED WILL OVERFLOW */
    /* ----------------------------------------------------------------------- */
    if(x < (float)0.)
    {
        goto L300;
    }
    *nz  = *n;
    i__1 = *n;
    for(i__ = 1; i__ <= i__1; ++i__)
    {
        i__2      = i__;
        y[i__2].real() = czero.real(), y[i__2].imag() = czero.imag();
        /* L96: */
    }
    return 0;
L100:
    /* ----------------------------------------------------------------------- */
    /*     SCALED FORWARD RECURRENCE FOR REMAINDER OF THE SEQUENCE */
    /* ----------------------------------------------------------------------- */
    s1.real() = cy[0].real(), s1.imag() = cy[0].imag();
    s2.real() = cy[1].real(), s2.imag() = cy[1].imag();
    i__1 = kflag - 1;
    c1.real() = csr[i__1].real(), c1.imag() = csr[i__1].imag();
    ascle = bry[kflag - 1];
    i__1  = *n;
    for(i__ = ib; i__ <= i__1; ++i__)
    {
        c2.real() = s2.real(), c2.imag() = s2.imag();
        q__2.real() = ck.real() * s2.real() - ck.imag() * s2.imag(), q__2.imag() = ck.real() * s2.imag() + ck.imag() * s2.real();
        q__1.real() = q__2.real() + s1.real(), q__1.imag() = q__2.imag() + s1.imag();
        s2.real() = q__1.real(), s2.imag() = q__1.imag();
        s1.real() = c2.real(), s1.imag() = c2.imag();
        q__1.real() = ck.real() + rz.real(), q__1.imag() = ck.imag() + rz.imag();
        ck.real() = q__1.real(), ck.imag() = q__1.imag();
        q__1.real() = s2.real() * c1.real() - s2.imag() * c1.imag(), q__1.imag() = s2.real() * c1.imag() + s2.imag() * c1.real();
        c2.real() = q__1.real(), c2.imag() = q__1.imag();
        i__2      = i__;
        y[i__2].real() = c2.real(), y[i__2].imag() = c2.imag();
        if(kflag >= 3)
        {
            goto L120;
        }
        c2r = c2.real();
        c2i = r_imag(&c2);
        c2r = abs(c2r);
        c2i = abs(c2i);
        c2m = max(c2r, c2i);
        if(c2m <= ascle)
        {
            goto L120;
        }
        ++kflag;
        ascle  = bry[kflag - 1];
        q__1.real() = s1.real() * c1.real() - s1.imag() * c1.imag(), q__1.imag() = s1.real() * c1.imag() + s1.imag() * c1.real();
        s1.real() = q__1.real(), s1.imag() = q__1.imag();
        s2.real() = c2.real(), s2.imag() = c2.imag();
        i__2   = kflag - 1;
        q__1.real() = s1.real() * css[i__2].real() - s1.imag() * css[i__2].imag(), q__1.imag() = s1.real() * css[i__2].imag() + s1.imag() * css[i__2].real();
        s1.real() = q__1.real(), s1.imag() = q__1.imag();
        i__2   = kflag - 1;
        q__1.real() = s2.real() * css[i__2].real() - s2.imag() * css[i__2].imag(), q__1.imag() = s2.real() * css[i__2].imag() + s2.imag() * css[i__2].real();
        s2.real() = q__1.real(), s2.imag() = q__1.imag();
        i__2 = kflag - 1;
        c1.real() = csr[i__2].real(), c1.imag() = csr[i__2].imag();
    L120:;
    }
L170:
    if(*mr == 0)
    {
        return 0;
    }
    /* ----------------------------------------------------------------------- */
    /*     ANALYTIC CONTINUATION FOR RE(Z).LT.0.0E0 */
    /* ----------------------------------------------------------------------- */
    *nz = 0;
    fmr = (float)(*mr);
    sgn = -r_sign(&pi, &fmr);
    /* ----------------------------------------------------------------------- */
    /*     CSPN AND CSGN ARE COEFF OF K AND I FUNCTIONS RESP. */
    /* ----------------------------------------------------------------------- */
    q__1.real() = (float)0., q__1.imag() = sgn;
    csgn.real() = q__1.real(), csgn.imag() = q__1.imag();
    if(yy <= (float)0.)
    {
        r_cnjg(&q__1, &csgn);
        csgn.real() = q__1.real(), csgn.imag() = q__1.imag();
    }
    ifn    = inu + *n - 1;
    ang    = fnf * sgn;
    cpn    = cos(ang);
    spn    = sin(ang);
    q__1.real() = cpn, q__1.imag() = spn;
    cspn.real() = q__1.real(), cspn.imag() = q__1.imag();
    if(ifn % 2 == 1)
    {
        q__1.real() = -cspn.real(), q__1.imag() = -cspn.imag();
        cspn.real() = q__1.real(), cspn.imag() = q__1.imag();
    }
    /* ----------------------------------------------------------------------- */
    /*     CS=COEFF OF THE J FUNCTION TO GET THE I FUNCTION. I(FNU,Z) IS */
    /*     COMPUTED FROM EXP(I*FNU*HPI)*J(FNU,-I*Z) WHERE Z IS IN THE FIRST */
    /*     QUADRANT. FOURTH QUADRANT VALUES (YY.LE.0.0E0) ARE COMPUTED BY */
    /*     CONJUGATION SINCE THE I FUNCTION IS REAL ON THE POSITIVE REAL AXIS */
    /* ----------------------------------------------------------------------- */
    r__1   = -sar;
    q__2.real() = car, q__2.imag() = r__1;
    q__1.real() = q__2.real() * csgn.real() - q__2.imag() * csgn.imag(), q__1.imag() = q__2.real() * csgn.imag() + q__2.imag() * csgn.real();
    cs.real() = q__1.real(), cs.imag() = q__1.imag();
    in   = ifn % 4 + 1;
    i__1 = in - 1;
    c2.real() = cip[i__1].real(), c2.imag() = cip[i__1].imag();
    r_cnjg(&q__2, &c2);
    q__1.real() = cs.real() * q__2.real() - cs.imag() * q__2.imag(), q__1.imag() = cs.real() * q__2.imag() + cs.imag() * q__2.real();
    cs.real() = q__1.real(), cs.imag() = q__1.imag();
    asc   = bry[0];
    kk    = *n;
    kdflg = 1;
    --ib;
    ic   = ib - 1;
    iuf  = 0;
    i__1 = *n;
    for(k = 1; k <= i__1; ++k)
    {
        /* ----------------------------------------------------------------------- */
        /*     LOGIC TO SORT OUT CASES WHOSE PARAMETERS WERE SET FOR THE K */
        /*     FUNCTION ABOVE */
        /* ----------------------------------------------------------------------- */
        fn = *fnu + (float)(kk - 1);
        if(*n > 2)
        {
            goto L180;
        }
    L175:
        i__2   = j - 1;
        phid.real() = phi[i__2].real(), phid.imag() = phi[i__2].imag();
        i__2   = j - 1;
        argd.real() = arg[i__2].real(), argd.imag() = arg[i__2].imag();
        i__2     = j - 1;
        zeta1d.real() = zeta1[i__2].real(), zeta1d.imag() = zeta1[i__2].imag();
        i__2     = j - 1;
        zeta2d.real() = zeta2[i__2].real(), zeta2d.imag() = zeta2[i__2].imag();
        i__2    = j - 1;
        asumd.real() = asum[i__2].real(), asumd.imag() = asum[i__2].imag();
        i__2    = j - 1;
        bsumd.real() = bsum[i__2].real(), bsumd.imag() = bsum[i__2].imag();
        j = 3 - j;
        goto L190;
    L180:
        if(kk == *n && ib < *n)
        {
            goto L190;
        }
        if(kk == ib || kk == ic)
        {
            goto L175;
        }
        cunhj_(&zn, &fn, &c__0, tol, &phid, &argd, &zeta1d, &zeta2d, &asumd, &bsumd);
    L190:
        if(*kode == 1)
        {
            goto L200;
        }
        q__1.real() = fn, q__1.imag() = (float)0.;
        cfn.real() = q__1.real(), cfn.imag() = q__1.imag();
        q__2.real() = -zeta1d.real(), q__2.imag() = -zeta1d.imag();
        q__5.real() = zb.real() + zeta2d.real(), q__5.imag() = zb.imag() + zeta2d.imag();
        c_div(&q__4, &cfn, &q__5);
        q__3.real() = cfn.real() * q__4.real() - cfn.imag() * q__4.imag(), q__3.imag() = cfn.real() * q__4.imag() + cfn.imag() * q__4.real();
        q__1.real() = q__2.real() + q__3.real(), q__1.imag() = q__2.imag() + q__3.imag();
        s1.real() = q__1.real(), s1.imag() = q__1.imag();
        goto L210;
    L200:
        q__2.real() = -zeta1d.real(), q__2.imag() = -zeta1d.imag();
        q__1.real() = q__2.real() + zeta2d.real(), q__1.imag() = q__2.imag() + zeta2d.imag();
        s1.real() = q__1.real(), s1.imag() = q__1.imag();
    L210:
        /* ----------------------------------------------------------------------- */
        /*     TEST FOR UNDERFLOW AND OVERFLOW */
        /* ----------------------------------------------------------------------- */
        rs1 = s1.real();
        if(abs(rs1) > *elim)
        {
            goto L260;
        }
        if(kdflg == 1)
        {
            iflag = 2;
        }
        if(abs(rs1) < *alim)
        {
            goto L220;
        }
        /* ----------------------------------------------------------------------- */
        /*     REFINE  TEST AND SCALE */
        /* ----------------------------------------------------------------------- */
        aphi = c_abs(&phid);
        aarg = c_abs(&argd);
        rs1  = rs1 + log(aphi) - log(aarg) * (float).25 - aic;
        if(abs(rs1) > *elim)
        {
            goto L260;
        }
        if(kdflg == 1)
        {
            iflag = 1;
        }
        if(rs1 < (float)0.)
        {
            goto L220;
        }
        if(kdflg == 1)
        {
            iflag = 3;
        }
    L220:
        cairy_(&argd, &c__0, &c__2, &ai, &nai, &idum);
        cairy_(&argd, &c__1, &c__2, &dai, &ndai, &idum);
        q__2.real() = cs.real() * phid.real() - cs.imag() * phid.imag(), q__2.imag() = cs.real() * phid.imag() + cs.imag() * phid.real();
        q__4.real() = ai.real() * asumd.real() - ai.imag() * asumd.imag(), q__4.imag() = ai.real() * asumd.imag() + ai.imag() * asumd.real();
        q__5.real() = dai.real() * bsumd.real() - dai.imag() * bsumd.imag(), q__5.imag() = dai.real() * bsumd.imag() + dai.imag() * bsumd.real();
        q__3.real() = q__4.real() + q__5.real(), q__3.imag() = q__4.imag() + q__5.imag();
        q__1.real() = q__2.real() * q__3.real() - q__2.imag() * q__3.imag(), q__1.imag() = q__2.real() * q__3.imag() + q__2.imag() * q__3.real();
        s2.real() = q__1.real(), s2.imag() = q__1.imag();
        c2r    = s1.real();
        c2i    = r_imag(&s1);
        i__2   = iflag - 1;
        c2m    = exp(c2r) * css[i__2].real();
        q__2.real() = c2m, q__2.imag() = (float)0.;
        r__1   = cos(c2i);
        r__2   = sin(c2i);
        q__3.real() = r__1, q__3.imag() = r__2;
        q__1.real() = q__2.real() * q__3.real() - q__2.imag() * q__3.imag(), q__1.imag() = q__2.real() * q__3.imag() + q__2.imag() * q__3.real();
        s1.real() = q__1.real(), s1.imag() = q__1.imag();
        q__1.real() = s2.real() * s1.real() - s2.imag() * s1.imag(), q__1.imag() = s2.real() * s1.imag() + s2.imag() * s1.real();
        s2.real() = q__1.real(), s2.imag() = q__1.imag();
        if(iflag != 1)
        {
            goto L230;
        }
        cuchk_(&s2, &nw, bry, tol);
        if(nw != 0)
        {
            s2.real() = (float)0., s2.imag() = (float)0.;
        }
    L230:
        if(yy <= (float)0.)
        {
            r_cnjg(&q__1, &s2);
            s2.real() = q__1.real(), s2.imag() = q__1.imag();
        }
        i__2       = kdflg - 1;
        cy[i__2].real() = s2.real(), cy[i__2].imag() = s2.imag();
        c2.real() = s2.real(), c2.imag() = s2.imag();
        i__2   = iflag - 1;
        q__1.real() = s2.real() * csr[i__2].real() - s2.imag() * csr[i__2].imag(), q__1.imag() = s2.real() * csr[i__2].imag() + s2.imag() * csr[i__2].real();
        s2.real() = q__1.real(), s2.imag() = q__1.imag();
        /* ----------------------------------------------------------------------- */
        /*     ADD I AND K FUNCTIONS, K SEQUENCE IN Y(I), I=1,N */
        /* ----------------------------------------------------------------------- */
        i__2 = kk;
        s1.real() = y[i__2].real(), s1.imag() = y[i__2].imag();
        if(*kode == 1)
        {
            goto L250;
        }
        cs1s2_(&zr, &s1, &s2, &nw, &asc, alim, &iuf);
        *nz += nw;
    L250:
        i__2   = kk;
        q__2.real() = s1.real() * cspn.real() - s1.imag() * cspn.imag(), q__2.imag() = s1.real() * cspn.imag() + s1.imag() * cspn.real();
        q__1.real() = q__2.real() + s2.real(), q__1.imag() = q__2.imag() + s2.imag();
        y[i__2].real() = q__1.real(), y[i__2].imag() = q__1.imag();
        --kk;
        q__1.real() = -cspn.real(), q__1.imag() = -cspn.imag();
        cspn.real() = q__1.real(), cspn.imag() = q__1.imag();
        q__2.real() = -cs.real(), q__2.imag() = -cs.imag();
        q__1.real() = q__2.real() * ci.real() - q__2.imag() * ci.imag(), q__1.imag() = q__2.real() * ci.imag() + q__2.imag() * ci.real();
        cs.real() = q__1.real(), cs.imag() = q__1.imag();
        if(c2.real() != czero.real() || c2.imag() != czero.imag())
        {
            goto L255;
        }
        kdflg = 1;
        goto L270;
    L255:
        if(kdflg == 2)
        {
            goto L275;
        }
        kdflg = 2;
        goto L270;
    L260:
        if(rs1 > (float)0.)
        {
            goto L300;
        }
        s2.real() = czero.real(), s2.imag() = czero.imag();
        goto L230;
    L270:;
    }
    k = *n;
L275:
    il = *n - k;
    if(il == 0)
    {
        return 0;
    }
    /* ----------------------------------------------------------------------- */
    /*     RECUR BACKWARD FOR REMAINDER OF I SEQUENCE AND ADD IN THE */
    /*     K FUNCTIONS, SCALING THE I SEQUENCE DURING RECURRENCE TO KEEP */
    /*     INTERMEDIATE ARITHMETIC ON SCALE NEAR EXPONENT EXTREMES. */
    /* ----------------------------------------------------------------------- */
    s1.real() = cy[0].real(), s1.imag() = cy[0].imag();
    s2.real() = cy[1].real(), s2.imag() = cy[1].imag();
    i__1 = iflag - 1;
    cs.real() = csr[i__1].real(), cs.imag() = csr[i__1].imag();
    ascle = bry[iflag - 1];
    fn    = (float)(inu + il);
    i__1  = il;
    for(i__ = 1; i__ <= i__1; ++i__)
    {
        c2.real() = s2.real(), c2.imag() = s2.imag();
        r__1   = fn + fnf;
        q__4.real() = r__1, q__4.imag() = (float)0.;
        q__3.real() = q__4.real() * rz.real() - q__4.imag() * rz.imag(), q__3.imag() = q__4.real() * rz.imag() + q__4.imag() * rz.real();
        q__2.real() = q__3.real() * s2.real() - q__3.imag() * s2.imag(), q__2.imag() = q__3.real() * s2.imag() + q__3.imag() * s2.real();
        q__1.real() = s1.real() + q__2.real(), q__1.imag() = s1.imag() + q__2.imag();
        s2.real() = q__1.real(), s2.imag() = q__1.imag();
        s1.real() = c2.real(), s1.imag() = c2.imag();
        fn += (float)-1.;
        q__1.real() = s2.real() * cs.real() - s2.imag() * cs.imag(), q__1.imag() = s2.real() * cs.imag() + s2.imag() * cs.real();
        c2.real() = q__1.real(), c2.imag() = q__1.imag();
        ck.real() = c2.real(), ck.imag() = c2.imag();
        i__2 = kk;
        c1.real() = y[i__2].real(), c1.imag() = y[i__2].imag();
        if(*kode == 1)
        {
            goto L280;
        }
        cs1s2_(&zr, &c1, &c2, &nw, &asc, alim, &iuf);
        *nz += nw;
    L280:
        i__2   = kk;
        q__2.real() = c1.real() * cspn.real() - c1.imag() * cspn.imag(), q__2.imag() = c1.real() * cspn.imag() + c1.imag() * cspn.real();
        q__1.real() = q__2.real() + c2.real(), q__1.imag() = q__2.imag() + c2.imag();
        y[i__2].real() = q__1.real(), y[i__2].imag() = q__1.imag();
        --kk;
        q__1.real() = -cspn.real(), q__1.imag() = -cspn.imag();
        cspn.real() = q__1.real(), cspn.imag() = q__1.imag();
        if(iflag >= 3)
        {
            goto L290;
        }
        c2r = ck.real();
        c2i = r_imag(&ck);
        c2r = abs(c2r);
        c2i = abs(c2i);
        c2m = max(c2r, c2i);
        if(c2m <= ascle)
        {
            goto L290;
        }
        ++iflag;
        ascle  = bry[iflag - 1];
        q__1.real() = s1.real() * cs.real() - s1.imag() * cs.imag(), q__1.imag() = s1.real() * cs.imag() + s1.imag() * cs.real();
        s1.real() = q__1.real(), s1.imag() = q__1.imag();
        s2.real() = ck.real(), s2.imag() = ck.imag();
        i__2   = iflag - 1;
        q__1.real() = s1.real() * css[i__2].real() - s1.imag() * css[i__2].imag(), q__1.imag() = s1.real() * css[i__2].imag() + s1.imag() * css[i__2].real();
        s1.real() = q__1.real(), s1.imag() = q__1.imag();
        i__2   = iflag - 1;
        q__1.real() = s2.real() * css[i__2].real() - s2.imag() * css[i__2].imag(), q__1.imag() = s2.real() * css[i__2].imag() + s2.imag() * css[i__2].real();
        s2.real() = q__1.real(), s2.imag() = q__1.imag();
        i__2 = iflag - 1;
        cs.real() = csr[i__2].real(), cs.imag() = csr[i__2].imag();
    L290:;
    }
    return 0;
L300:
    *nz = -1;
    return 0;
} /* cunk2_ */

 int cuoik_(System::Complex<float>* z__, float* fnu, int32* kode, int32* ikflg, int32* n, System::Complex<float>* y, int32* nuf, float* tol, float* elim, float* alim)
{
    /* Initialized data */

    static System::Complex<float> czero = {(float)0., (float)0.};
    static float    aic   = (float)1.265512123484645396;

    /* System generated locals */
    int32 i__1, i__2;
    float    r__1, r__2;
    System::Complex<float> q__1, q__2, q__3, q__4, q__5;


    
    int32                     i__;
    float                        x, ax, ay;
    System::Complex<float>                     zb, cz;
    int32                     nn, nw;
    System::Complex<float>                     zn, zr;
    float                        yy;
    System::Complex<float>                     arg, phi;
    float                        fnn, gnn, gnu, rcz;
    System::Complex<float>                     sum;
    float                        aarg, aphi;
    int32                     init;
    System::Complex<float>                     asum, bsum, cwrk[16], zeta1, zeta2;
    float                        ascle;
   int32           iform;
    

    /* *********************************************************************72 */

    /* ***BEGIN PROLOGUE  CUOIK */
    /* ***REFER TO  CBESI,CBESK,CBESH */

    /*     CUOIK COMPUTES THE LEADING TERMS OF THE UNIFORM ASYMPTOTIC */
    /*     EXPANSIONS FOR THE I AND K FUNCTIONS AND COMPARES THEM */
    /*     (IN LOGARITHMIC FORM) TO ALIM AND ELIM FOR OVER AND UNDERFLOW */
    /*     WHERE ALIM.LT.ELIM. IF THE MAGNITUDE, BASED ON THE LEADING */
    /*     EXPONENTIAL, IS LESS THAN ALIM OR GREATER THAN -ALIM, THEN */
    /*     THE RESULT IS ON SCALE. IF NOT, THEN A REFINED TEST USING OTHER */
    /*     MULTIPLIERS (IN LOGARITHMIC FORM) IS MADE BASED ON ELIM. HERE */
    /*     EXP(-ELIM)=SMALLEST MACHINE NUMBER*1.0E+3 AND EXP(-ALIM)= */
    /*     EXP(-ELIM)/TOL */

    /*     IKFLG=1 MEANS THE I SEQUENCE IS TESTED */
    /*          =2 MEANS THE K SEQUENCE IS TESTED */
    /*     NUF = 0 MEANS THE LAST MEMBER OF THE SEQUENCE IS ON SCALE */
    /*         =-1 MEANS AN OVERFLOW WOULD OCCUR */
    /*     IKFLG=1 AND NUF.GT.0 MEANS THE LAST NUF Y VALUES WERE SET TO ZERO */
    /*             THE FIRST N-NUF VALUES MUST BE SET BY ANOTHER ROUTINE */
    /*     IKFLG=2 AND NUF.EQ.N MEANS ALL Y VALUES WERE SET TO ZERO */
    /*     IKFLG=2 AND 0.LT.NUF.LT.N NOT CONSIDERED. Y MUST BE SET BY */
    /*             ANOTHER ROUTINE */

    /* ***ROUTINES CALLED  CUCHK,CUNHJ,CUNIK,R1MACH */
    /* ***END PROLOGUE  CUOIK */
    /* Parameter adjustments */
    --y;

    /* Function Body */
    *nuf = 0;
    nn   = *n;
    x    = z__->real();
    zr.real() = z__->real(), zr.imag() = z__->imag();
    if(x < (float)0.)
    {
        q__1.real() = -z__->real(), q__1.imag() = -z__->imag();
        zr.real() = q__1.real(), zr.imag() = q__1.imag();
    }
    zb.real() = zr.real(), zb.imag() = zr.imag();
    yy    = r_imag(&zr);
    ax    = abs(x) * (float)1.7321;
    ay    = abs(yy);
    iform = 1;
    if(ay > ax)
    {
        iform = 2;
    }
    gnu = max(*fnu, (float)1.);
    if(*ikflg == 1)
    {
        goto L10;
    }
    fnn = (float)nn;
    gnn = *fnu + fnn - (float)1.;
    gnu = max(gnn, fnn);
L10:
    /* ----------------------------------------------------------------------- */
    /*     ONLY THE MAGNITUDE OF ARG AND PHI ARE NEEDED ALONG WITH THE */
    /*     REAL PARTS OF ZETA1, ZETA2 AND ZB. NO ATTEMPT IS MADE TO GET */
    /*     THE SIGN OF THE IMAGINARY PART CORRECT. */
    /* ----------------------------------------------------------------------- */
    if(iform == 2)
    {
        goto L20;
    }
    init = 0;
    cunik_(&zr, &gnu, ikflg, &c__1, tol, &init, &phi, &zeta1, &zeta2, &sum, cwrk);
    q__2.real() = -zeta1.real(), q__2.imag() = -zeta1.imag();
    q__1.real() = q__2.real() + zeta2.real(), q__1.imag() = q__2.imag() + zeta2.imag();
    cz.real() = q__1.real(), cz.imag() = q__1.imag();
    goto L40;
L20:
    q__2.real() = -zr.real(), q__2.imag() = -zr.imag();
    q__1.real() = q__2.real() * (float)0. - q__2.imag() * (float)1., q__1.imag() = q__2.real() * (float)1. + q__2.imag() * (float)0.;
    zn.real() = q__1.real(), zn.imag() = q__1.imag();
    if(yy > (float)0.)
    {
        goto L30;
    }
    q__2.real() = -zn.real(), q__2.imag() = -zn.imag();
    r_cnjg(&q__1, &q__2);
    zn.real() = q__1.real(), zn.imag() = q__1.imag();
L30:
    cunhj_(&zn, &gnu, &c__1, tol, &phi, &arg, &zeta1, &zeta2, &asum, &bsum);
    q__2.real() = -zeta1.real(), q__2.imag() = -zeta1.imag();
    q__1.real() = q__2.real() + zeta2.real(), q__1.imag() = q__2.imag() + zeta2.imag();
    cz.real() = q__1.real(), cz.imag() = q__1.imag();
    aarg = c_abs(&arg);
L40:
    if(*kode == 2)
    {
        q__1.real() = cz.real() - zb.real(), q__1.imag() = cz.imag() - zb.imag();
        cz.real() = q__1.real(), cz.imag() = q__1.imag();
    }
    if(*ikflg == 2)
    {
        q__1.real() = -cz.real(), q__1.imag() = -cz.imag();
        cz.real() = q__1.real(), cz.imag() = q__1.imag();
    }
    aphi = c_abs(&phi);
    rcz  = cz.real();
    /* ----------------------------------------------------------------------- */
    /*     OVERFLOW TEST */
    /* ----------------------------------------------------------------------- */
    if(rcz > *elim)
    {
        goto L170;
    }
    if(rcz < *alim)
    {
        goto L50;
    }
    rcz += log(aphi);
    if(iform == 2)
    {
        rcz = rcz - log(aarg) * (float).25 - aic;
    }
    if(rcz > *elim)
    {
        goto L170;
    }
    goto L100;
L50:
    /* ----------------------------------------------------------------------- */
    /*     UNDERFLOW TEST */
    /* ----------------------------------------------------------------------- */
    if(rcz < -(*elim))
    {
        goto L60;
    }
    if(rcz > -(*alim))
    {
        goto L100;
    }
    rcz += log(aphi);
    if(iform == 2)
    {
        rcz = rcz - log(aarg) * (float).25 - aic;
    }
    if(rcz > -(*elim))
    {
        goto L80;
    }
L60:
    i__1 = nn;
    for(i__ = 1; i__ <= i__1; ++i__)
    {
        i__2      = i__;
        y[i__2].real() = czero.real(), y[i__2].imag() = czero.imag();
        /* L70: */
    }
    *nuf = nn;
    return 0;
L80:
    ascle = r1mach_(&c__1) * (float)1e3 / *tol;
    c_log(&q__2, &phi);
    q__1.real() = cz.real() + q__2.real(), q__1.imag() = cz.imag() + q__2.imag();
    cz.real() = q__1.real(), cz.imag() = q__1.imag();
    if(iform == 1)
    {
        goto L90;
    }
    c_log(&q__4, &arg);
    q__3.real() = q__4.real() * (float).25 - q__4.imag() * (float)0., q__3.imag() = q__4.real() * (float)0. + q__4.imag() * (float).25;
    q__2.real() = cz.real() - q__3.real(), q__2.imag() = cz.imag() - q__3.imag();
    q__5.real() = aic, q__5.imag() = (float)0.;
    q__1.real() = q__2.real() - q__5.real(), q__1.imag() = q__2.imag() - q__5.imag();
    cz.real() = q__1.real(), cz.imag() = q__1.imag();
L90:
    ax     = exp(rcz) / *tol;
    ay     = r_imag(&cz);
    q__2.real() = ax, q__2.imag() = (float)0.;
    r__1   = cos(ay);
    r__2   = sin(ay);
    q__3.real() = r__1, q__3.imag() = r__2;
    q__1.real() = q__2.real() * q__3.real() - q__2.imag() * q__3.imag(), q__1.imag() = q__2.real() * q__3.imag() + q__2.imag() * q__3.real();
    cz.real() = q__1.real(), cz.imag() = q__1.imag();
    cuchk_(&cz, &nw, &ascle, tol);
    if(nw == 1)
    {
        goto L60;
    }
L100:
    if(*ikflg == 2)
    {
        return 0;
    }
    if(*n == 1)
    {
        return 0;
    }
/* ----------------------------------------------------------------------- */
/*     SET UNDERFLOWS ON I SEQUENCE */
/* ----------------------------------------------------------------------- */
L110:
    gnu = *fnu + (float)(nn - 1);
    if(iform == 2)
    {
        goto L120;
    }
    init = 0;
    cunik_(&zr, &gnu, ikflg, &c__1, tol, &init, &phi, &zeta1, &zeta2, &sum, cwrk);
    q__2.real() = -zeta1.real(), q__2.imag() = -zeta1.imag();
    q__1.real() = q__2.real() + zeta2.real(), q__1.imag() = q__2.imag() + zeta2.imag();
    cz.real() = q__1.real(), cz.imag() = q__1.imag();
    goto L130;
L120:
    cunhj_(&zn, &gnu, &c__1, tol, &phi, &arg, &zeta1, &zeta2, &asum, &bsum);
    q__2.real() = -zeta1.real(), q__2.imag() = -zeta1.imag();
    q__1.real() = q__2.real() + zeta2.real(), q__1.imag() = q__2.imag() + zeta2.imag();
    cz.real() = q__1.real(), cz.imag() = q__1.imag();
    aarg = c_abs(&arg);
L130:
    if(*kode == 2)
    {
        q__1.real() = cz.real() - zb.real(), q__1.imag() = cz.imag() - zb.imag();
        cz.real() = q__1.real(), cz.imag() = q__1.imag();
    }
    aphi = c_abs(&phi);
    rcz  = cz.real();
    if(rcz < -(*elim))
    {
        goto L140;
    }
    if(rcz > -(*alim))
    {
        return 0;
    }
    rcz += log(aphi);
    if(iform == 2)
    {
        rcz = rcz - log(aarg) * (float).25 - aic;
    }
    if(rcz > -(*elim))
    {
        goto L150;
    }
L140:
    i__1      = nn;
    y[i__1].real() = czero.real(), y[i__1].imag() = czero.imag();
    --nn;
    ++(*nuf);
    if(nn == 0)
    {
        return 0;
    }
    goto L110;
L150:
    ascle = r1mach_(&c__1) * (float)1e3 / *tol;
    c_log(&q__2, &phi);
    q__1.real() = cz.real() + q__2.real(), q__1.imag() = cz.imag() + q__2.imag();
    cz.real() = q__1.real(), cz.imag() = q__1.imag();
    if(iform == 1)
    {
        goto L160;
    }
    c_log(&q__4, &arg);
    q__3.real() = q__4.real() * (float).25 - q__4.imag() * (float)0., q__3.imag() = q__4.real() * (float)0. + q__4.imag() * (float).25;
    q__2.real() = cz.real() - q__3.real(), q__2.imag() = cz.imag() - q__3.imag();
    q__5.real() = aic, q__5.imag() = (float)0.;
    q__1.real() = q__2.real() - q__5.real(), q__1.imag() = q__2.imag() - q__5.imag();
    cz.real() = q__1.real(), cz.imag() = q__1.imag();
L160:
    ax     = exp(rcz) / *tol;
    ay     = r_imag(&cz);
    q__2.real() = ax, q__2.imag() = (float)0.;
    r__1   = cos(ay);
    r__2   = sin(ay);
    q__3.real() = r__1, q__3.imag() = r__2;
    q__1.real() = q__2.real() * q__3.real() - q__2.imag() * q__3.imag(), q__1.imag() = q__2.real() * q__3.imag() + q__2.imag() * q__3.real();
    cz.real() = q__1.real(), cz.imag() = q__1.imag();
    cuchk_(&cz, &nw, &ascle, tol);
    if(nw == 1)
    {
        goto L140;
    }
    return 0;
L170:
    *nuf = -1;
    return 0;
} /* cuoik_ */

 int cwrsk_(System::Complex<float>* zr, float* fnu, int32* kode, int32* n, System::Complex<float>* y, int32* nz, System::Complex<float>* cw, float* tol, float* elim, float* alim)
{
    /* System generated locals */
    int32 i__1, i__2;
    float    r__1;
    System::Complex<float> q__1, q__2, q__3;

    
    int32                     i__;
    System::Complex<float>                     c1, c2;
    float                        s1, s2;
    System::Complex<float>                     ct;
    int32                     nw;
    System::Complex<float>                     st;
    float                        yy, act, acw;
    System::Complex<float>                     rct, cscl, cinu;
    float                        ascle;

    /* *********************************************************************72 */

    /* ***BEGIN PROLOGUE  CWRSK */
    /* ***REFER TO  CBESI,CBESK */

    /*     CWRSK COMPUTES THE I BESSEL FUNCTION FOR RE(Z).GE.0.0 BY */
    /*     NORMALIZING THE I FUNCTION RATIOS FROM CRATI BY THE WRONSKIAN */

    /* ***ROUTINES CALLED  CBKNU,CRATI,R1MACH */
    /* ***END PROLOGUE  CWRSK */
    /* ----------------------------------------------------------------------- */
    /*     I(FNU+I-1,Z) BY BACKWARD RECURRENCE FOR RATIOS */
    /*     Y(I)=I(FNU+I,Z)/I(FNU+I-1,Z) FROM CRATI NORMALIZED BY THE */
    /*     WRONSKIAN WITH K(FNU,Z) AND K(FNU+1,Z) FROM CBKNU. */
    /* ----------------------------------------------------------------------- */
    /* Parameter adjustments */
    --y;
    --cw;

    /* Function Body */
    *nz = 0;
    cbknu_(zr, fnu, kode, &c__2, &cw[1], &nw, tol, elim, alim);
    if(nw != 0)
    {
        goto L50;
    }
    crati_(zr, fnu, n, &y[1], tol);
    /* ----------------------------------------------------------------------- */
    /*     RECUR FORWARD ON I(FNU+1,Z) = R(FNU,Z)*I(FNU,Z), */
    /*     R(FNU+J-1,Z)=Y(J),  J=1,...,N */
    /* ----------------------------------------------------------------------- */
    cinu.real() = (float)1., cinu.imag() = (float)0.;
    if(*kode == 1)
    {
        goto L10;
    }
    yy     = r_imag(zr);
    s1     = cos(yy);
    s2     = sin(yy);
    q__1.real() = s1, q__1.imag() = s2;
    cinu.real() = q__1.real(), cinu.imag() = q__1.imag();
L10:
    /* ----------------------------------------------------------------------- */
    /*     ON LOW EXPONENT MACHINES THE K FUNCTIONS CAN BE CLOSE TO BOTH */
    /*     THE UNDER AND OVERFLOW LIMITS AND THE NORMALIZATION MUST BE */
    /*     SCALED TO PREVENT OVER OR UNDERFLOW. CUOIK HAS DETERMINED THAT */
    /*     THE RESULT IS ON SCALE. */
    /* ----------------------------------------------------------------------- */
    acw    = c_abs(&cw[2]);
    ascle  = r1mach_(&c__1) * (float)1e3 / *tol;
    cscl.real() = (float)1., cscl.imag() = (float)0.;
    if(acw > ascle)
    {
        goto L20;
    }
    r__1   = (float)1. / *tol;
    q__1.real() = r__1, q__1.imag() = (float)0.;
    cscl.real() = q__1.real(), cscl.imag() = q__1.imag();
    goto L30;
L20:
    ascle = (float)1. / ascle;
    if(acw < ascle)
    {
        goto L30;
    }
    q__1.real() = *tol, q__1.imag() = (float)0.;
    cscl.real() = q__1.real(), cscl.imag() = q__1.imag();
L30:
    q__1.real() = cw[1].real() * cscl.real() - cw[1].imag() * cscl.imag(), q__1.imag() = cw[1].real() * cscl.imag() + cw[1].imag() * cscl.real();
    c1.real() = q__1.real(), c1.imag() = q__1.imag();
    q__1.real() = cw[2].real() * cscl.real() - cw[2].imag() * cscl.imag(), q__1.imag() = cw[2].real() * cscl.imag() + cw[2].imag() * cscl.real();
    c2.real() = q__1.real(), c2.imag() = q__1.imag();
    st.real() = y[1].real(), st.imag() = y[1].imag();
    /* ----------------------------------------------------------------------- */
    /*     CINU=CINU*(CONJG(CT)/CABS(CT))*(1.0E0/CABS(CT) PREVENTS */
    /*     UNDER- OR OVERFLOW PREMATURELY BY SQUARING CABS(CT) */
    /* ----------------------------------------------------------------------- */
    q__3.real() = st.real() * c1.real() - st.imag() * c1.imag(), q__3.imag() = st.real() * c1.imag() + st.imag() * c1.real();
    q__2.real() = c2.real() + q__3.real(), q__2.imag() = c2.imag() + q__3.imag();
    q__1.real() = zr->real() * q__2.real() - zr->imag() * q__2.imag(), q__1.imag() = zr->real() * q__2.imag() + zr->imag() * q__2.real();
    ct.real() = q__1.real(), ct.imag() = q__1.imag();
    act    = c_abs(&ct);
    r__1   = (float)1. / act;
    q__1.real() = r__1, q__1.imag() = (float)0.;
    rct.real() = q__1.real(), rct.imag() = q__1.imag();
    r_cnjg(&q__2, &ct);
    q__1.real() = q__2.real() * rct.real() - q__2.imag() * rct.imag(), q__1.imag() = q__2.real() * rct.imag() + q__2.imag() * rct.real();
    ct.real() = q__1.real(), ct.imag() = q__1.imag();
    q__2.real() = cinu.real() * rct.real() - cinu.imag() * rct.imag(), q__2.imag() = cinu.real() * rct.imag() + cinu.imag() * rct.real();
    q__1.real() = q__2.real() * ct.real() - q__2.imag() * ct.imag(), q__1.imag() = q__2.real() * ct.imag() + q__2.imag() * ct.real();
    cinu.real() = q__1.real(), cinu.imag() = q__1.imag();
    q__1.real() = cinu.real() * cscl.real() - cinu.imag() * cscl.imag(), q__1.imag() = cinu.real() * cscl.imag() + cinu.imag() * cscl.real();
    y[1].real() = q__1.real(), y[1].imag() = q__1.imag();
    if(*n == 1)
    {
        return 0;
    }
    i__1 = *n;
    for(i__ = 2; i__ <= i__1; ++i__)
    {
        q__1.real() = st.real() * cinu.real() - st.imag() * cinu.imag(), q__1.imag() = st.real() * cinu.imag() + st.imag() * cinu.real();
        cinu.real() = q__1.real(), cinu.imag() = q__1.imag();
        i__2 = i__;
        st.real() = y[i__2].real(), st.imag() = y[i__2].imag();
        i__2   = i__;
        q__1.real() = cinu.real() * cscl.real() - cinu.imag() * cscl.imag(), q__1.imag() = cinu.real() * cscl.imag() + cinu.imag() * cscl.real();
        y[i__2].real() = q__1.real(), y[i__2].imag() = q__1.imag();
        /* L40: */
    }
    return 0;
L50:
    *nz = -1;
    if(nw == -2)
    {
        *nz = -2;
    }
    return 0;
} /* cwrsk_ */

//double d1mach_(int32* i__)
//{
//    /* System generated locals */
//    double ret_val;
//    cilist     ci__1;
//
//    /* Builtin functions */
//    int32              s_wsfe(cilist*), do_fio(int32*, char*, ftnlen), e_wsfe();
//     int s_stop(char*, ftnlen);
//
//    /* *********************************************************************72 */
//
//    /* c D1MACH returns double precision float machine-dependent constants. */
//
//    /*  Discussion: */
//
//    /*    D1MACH can be used to obtain machine-dependent parameters */
//    /*    for the local machine environment.  It is a function */
//    /*    with one input argument, and can be called as follows: */
//
//    /*      D = D1MACH ( I ) */
//
//    /*    where I=1,...,5.  The output value of D above is */
//    /*    determined by the input value of I:. */
//
//    /*    D1MACH ( 1) = B**(EMIN-1), the smallest positive magnitude. */
//    /*    D1MACH ( 2) = B**EMAX*(1 - B**(-T)), the largest magnitude. */
//    /*    D1MACH ( 3) = B**(-T), the smallest relative spacing. */
//    /*    D1MACH ( 4) = B**(1-T), the largest relative spacing. */
//    /*    D1MACH ( 5) = LOG10(B) */
//
//    /*  Licensing: */
//
//    /*    This code is distributed under the GNU LGPL license. */
//
//    /*  Modified: */
//
//    /*    25 April 2007 */
//
//    /*  Author: */
//
//    /*    Original FORTRAN77 version by Phyllis Fox, Andrew Hall, Norman Schryer. */
//    /*    This FORTRAN77 version by John Burkardt. */
//
//    /*  Reference: */
//
//    /*    Phyllis Fox, Andrew Hall, Norman Schryer, */
//    /*    Algorithm 528: */
//    /*    Framework for a Portable Library, */
//    /*    ACM Transactions on Mathematical Software, */
//    /*    Volume 4, Number 2, June 1978, page 176-188. */
//
//    /*  Parameters: */
//
//    /*    Input, int32 I, the index of the desired constant. */
//
//    /*    Output, double precision D1MACH, the value of the constant. */
//
//    if(*i__ < 1)
//    {
//        ci__1.cierr  = 0;
//        ci__1.ciunit = 6;
//        ci__1.cifmt  = "(a)";
//        s_wsfe(&ci__1);
//        do_fio(&c__1, " ", (ftnlen)1);
//        e_wsfe();
//        ci__1.cierr  = 0;
//        ci__1.ciunit = 6;
//        ci__1.cifmt  = "(a)";
//        s_wsfe(&ci__1);
//        do_fio(&c__1, "D1MACH - Fatal error!", (ftnlen)21);
//        e_wsfe();
//        ci__1.cierr  = 0;
//        ci__1.ciunit = 6;
//        ci__1.cifmt  = "(a)";
//        s_wsfe(&ci__1);
//        do_fio(&c__1, "  The input argument I is out of bounds.", (ftnlen)40);
//        e_wsfe();
//        ci__1.cierr  = 0;
//        ci__1.ciunit = 6;
//        ci__1.cifmt  = "(a)";
//        s_wsfe(&ci__1);
//        do_fio(&c__1, "  Legal values satisfy 1 <= I <= 5.", (ftnlen)35);
//        e_wsfe();
//        ci__1.cierr  = 0;
//        ci__1.ciunit = 6;
//        ci__1.cifmt  = "(a,i12)";
//        s_wsfe(&ci__1);
//        do_fio(&c__1, "  I = ", (ftnlen)6);
//        do_fio(&c__1, (char*)&(*i__), (ftnlen)sizeof(int32));
//        e_wsfe();
//        ret_val = 0.;
//        s_stop("", (ftnlen)0);
//    }
//    else if(*i__ == 1)
//    {
//        ret_val = 4.450147717014403e-308;
//    }
//    else if(*i__ == 2)
//    {
//        ret_val = 8.988465674311579e307;
//    }
//    else if(*i__ == 3)
//    {
//        ret_val = 1.110223024625157e-16;
//    }
//    else if(*i__ == 4)
//    {
//        ret_val = 2.220446049250313e-16;
//    }
//    else if(*i__ == 5)
//    {
//        ret_val = .301029995663981;
//    }
//    else if(5 < *i__)
//    {
//        ci__1.cierr  = 0;
//        ci__1.ciunit = 6;
//        ci__1.cifmt  = "(a)";
//        s_wsfe(&ci__1);
//        do_fio(&c__1, " ", (ftnlen)1);
//        e_wsfe();
//        ci__1.cierr  = 0;
//        ci__1.ciunit = 6;
//        ci__1.cifmt  = "(a)";
//        s_wsfe(&ci__1);
//        do_fio(&c__1, "D1MACH - Fatal error!", (ftnlen)21);
//        e_wsfe();
//        ci__1.cierr  = 0;
//        ci__1.ciunit = 6;
//        ci__1.cifmt  = "(a)";
//        s_wsfe(&ci__1);
//        do_fio(&c__1, "  The input argument I is out of bounds.", (ftnlen)40);
//        e_wsfe();
//        ci__1.cierr  = 0;
//        ci__1.ciunit = 6;
//        ci__1.cifmt  = "(a)";
//        s_wsfe(&ci__1);
//        do_fio(&c__1, "  Legal values satisfy 1 <= I <= 5.", (ftnlen)35);
//        e_wsfe();
//        ci__1.cierr  = 0;
//        ci__1.ciunit = 6;
//        ci__1.cifmt  = "(a,i12)";
//        s_wsfe(&ci__1);
//        do_fio(&c__1, "  I = ", (ftnlen)6);
//        do_fio(&c__1, (char*)&(*i__), (ftnlen)sizeof(int32));
//        e_wsfe();
//        ret_val = 0.;
//        s_stop("", (ftnlen)0);
//    }
//    return ret_val;
//} /* d1mach_ */

double dgamln_(double* z__, int32* ierr)
{
    /* Initialized data */

    static double gln[100] = {0.,
                                  0.,
                                  .693147180559945309,
                                  1.791759469228055,
                                  3.17805383034794562,
                                  4.78749174278204599,
                                  6.579251212010101,
                                  8.5251613610654143,
                                  10.6046029027452502,
                                  12.8018274800814696,
                                  15.1044125730755153,
                                  17.5023078458738858,
                                  19.9872144956618861,
                                  22.5521638531234229,
                                  25.1912211827386815,
                                  27.8992713838408916,
                                  30.6718601060806728,
                                  33.5050734501368889,
                                  36.3954452080330536,
                                  39.339884187199494,
                                  42.335616460753485,
                                  45.380138898476908,
                                  48.4711813518352239,
                                  51.6066755677643736,
                                  54.7847293981123192,
                                  58.0036052229805199,
                                  61.261701761002002,
                                  64.5575386270063311,
                                  67.889743137181535,
                                  71.257038967168009,
                                  74.6582363488301644,
                                  78.0922235533153106,
                                  81.5579594561150372,
                                  85.0544670175815174,
                                  88.5808275421976788,
                                  92.1361756036870925,
                                  95.7196945421432025,
                                  99.3306124547874269,
                                  102.968198614513813,
                                  106.631760260643459,
                                  110.320639714757395,
                                  114.034211781461703,
                                  117.771881399745072,
                                  121.533081515438634,
                                  125.317271149356895,
                                  129.123933639127215,
                                  132.95257503561631,
                                  136.802722637326368,
                                  140.673923648234259,
                                  144.565743946344886,
                                  148.477766951773032,
                                  152.409592584497358,
                                  156.360836303078785,
                                  160.331128216630907,
                                  164.320112263195181,
                                  168.327445448427652,
                                  172.352797139162802,
                                  176.395848406997352,
                                  180.456291417543771,
                                  184.533828861449491,
                                  188.628173423671591,
                                  192.739047287844902,
                                  196.866181672889994,
                                  201.009316399281527,
                                  205.168199482641199,
                                  209.342586752536836,
                                  213.532241494563261,
                                  217.736934113954227,
                                  221.956441819130334,
                                  226.190548323727593,
                                  230.439043565776952,
                                  234.701723442818268,
                                  238.978389561834323,
                                  243.268849002982714,
                                  247.572914096186884,
                                  251.890402209723194,
                                  256.221135550009525,
                                  260.564940971863209,
                                  264.921649798552801,
                                  269.291097651019823,
                                  273.673124285693704,
                                  278.067573440366143,
                                  282.474292687630396,
                                  286.893133295426994,
                                  291.323950094270308,
                                  295.766601350760624,
                                  300.220948647014132,
                                  304.686856765668715,
                                  309.164193580146922,
                                  313.652829949879062,
                                  318.152639620209327,
                                  322.663499126726177,
                                  327.185287703775217,
                                  331.717887196928473,
                                  336.261181979198477,
                                  340.815058870799018,
                                  345.379407062266854,
                                  349.954118040770237,
                                  354.539085519440809,
                                  359.134205369575399};
    static double cf[22] = {.0833333333333333333,  -.00277777777777777778, 7.93650793650793651e-4, -5.95238095238095238e-4, 8.41750841750841751e-4, -.00191752691752691753,
                                .00641025641025641026, -.0295506535947712418,  .179644372368830573,    -1.39243221690590112,    13.402864044168392,     -156.848284626002017,
                                2193.10333333333333,   -36108.7712537249894,   691472.268851313067,    -15238221.5394074162,    382900751.391414141,    -10882266035.7843911,
                                347320283765.002252,   -12369602142269.2745,   488788064793079.335,    -21320333960919373.9};
    static double con    = 1.83787706640934548;

    /* System generated locals */
    int32    i__1;
    double ret_val;


    
    int32           i__, k;
    double        s, t1, fz, zm;
    int32           mz, nz;
    double        zp;
    int32           i1m;
    double        fln, tlg, rln, trm, tst, zsq, zinc, zmin, zdmy, wdtol;

    /* *********************************************************************72 */

    /* ***BEGIN PROLOGUE  DGAMLN */
    /* ***DATE WRITTEN   830501   (YYMMDD) */
    /* ***REVISION DATE  830501   (YYMMDD) */
    /* ***CATEGORY NO.  B5F */
    /* ***KEYWORDS  GAMMA FUNCTION,LOGARITHM OF GAMMA FUNCTION */
    /* ***AUTHOR  AMOS, DONALD E., SANDIA NATIONAL LABORATORIES */
    /* ***PURPOSE  TO COMPUTE THE LOGARITHM OF THE GAMMA FUNCTION */
    /* ***DESCRIPTION */

    /*               **** A DOUBLE PRECISION ROUTINE **** */
    /*         DGAMLN COMPUTES THE NATURAL LOG OF THE GAMMA FUNCTION FOR */
    /*         Z.GT.0.  THE ASYMPTOTIC EXPANSION IS USED TO GENERATE VALUES */
    /*         GREATER THAN ZMIN WHICH ARE ADJUSTED BY THE RECURSION */
    /*         G(Z+1)=Z*G(Z) FOR Z.LE.ZMIN.  THE FUNCTION WAS MADE AS */
    /*         PORTABLE AS POSSIBLE BY COMPUTIMG ZMIN FROM THE NUMBER OF BASE */
    /*         10 DIGITS IN A WORD, RLN=AMAX1(-ALOG10(R1MACH(4)),0.5E-18) */
    /*         LIMITED TO 18 DIGITS OF (RELATIVE) ACCURACY. */

    /*         SINCE INTEGER ARGUMENTS ARE COMMON, A TABLE LOOK UP ON 100 */
    /*         VALUES IS USED FOR SPEED OF EXECUTION. */

    /*     DESCRIPTION OF ARGUMENTS */

    /*         INPUT      Z IS D0UBLE PRECISION */
    /*           Z      - ARGUMENT, Z.GT.0.0D0 */

    /*         OUTPUT      DGAMLN IS DOUBLE PRECISION */
    /*           DGAMLN  - NATURAL LOG OF THE GAMMA FUNCTION AT Z.NE.0.0D0 */
    /*           IERR    - ERROR FLAG */
    /*                     IERR=0, NORMAL RETURN, COMPUTATION COMPLETED */
    /*                     IERR=1, Z.LE.0.0D0,    NO COMPUTATION */

    /* ***REFERENCES  COMPUTATION OF BESSEL FUNCTIONS OF COMPLEX ARGUMENT */
    /*                 BY D. E. AMOS, SAND83-0083, MAY, 1983. */
    /* ***ROUTINES CALLED  I1MACH,D1MACH */
    /* ***END PROLOGUE  DGAMLN */
    /*           LNGAMMA(N), N=1,100 */
    /*             COEFFICIENTS OF ASYMPTOTIC EXPANSION */

    /*             LN(2*PI) */

    /* ***FIRST EXECUTABLE STATEMENT  DGAMLN */
    *ierr = 0;
    if(*z__ <= 0.)
    {
        goto L70;
    }
    if(*z__ > 101.)
    {
        goto L10;
    }
    nz = (int32)(*z__);
    fz = *z__ - (float)nz;
    if(fz > 0.)
    {
        goto L10;
    }
    if(nz > 100)
    {
        goto L10;
    }
    ret_val = gln[nz - 1];
    return ret_val;
L10:
    wdtol = d1mach_(&c__4);
    wdtol = max(wdtol, 5e-19);
    i1m   = i1mach_(&c__14);
    rln   = d1mach_(&c__5) * (float)i1m;
    fln   = std::min(rln, 20.);
    fln   = max(fln, 3.);
    fln += -3.;
    zm   = fln * .3875 + 1.8;
    mz   = (int32)((float)zm) + 1;
    zmin = (float)mz;
    zdmy = *z__;
    zinc = 0.;
    if(*z__ >= zmin)
    {
        goto L20;
    }
    zinc = zmin - (float)nz;
    zdmy = *z__ + zinc;
L20:
    zp = 1. / zdmy;
    t1 = cf[0] * zp;
    s  = t1;
    if(zp < wdtol)
    {
        goto L40;
    }
    zsq = zp * zp;
    tst = t1 * wdtol;
    for(k = 2; k <= 22; ++k)
    {
        zp *= zsq;
        trm = cf[k - 1] * zp;
        if(abs(trm) < tst)
        {
            goto L40;
        }
        s += trm;
        /* L30: */
    }
L40:
    if(zinc != 0.)
    {
        goto L50;
    }
    tlg     = log(*z__);
    ret_val = *z__ * (tlg - 1.) + (con - tlg) * .5 + s;
    return ret_val;
L50:
    zp   = 1.;
    nz   = (int32)((float)zinc);
    i__1 = nz;
    for(i__ = 1; i__ <= i__1; ++i__)
    {
        zp *= *z__ + (float)(i__ - 1);
        /* L60: */
    }
    tlg     = log(zdmy);
    ret_val = zdmy * (tlg - 1.) - log(zp) + (con - tlg) * .5 + s;
    return ret_val;

L70:
    *ierr = 1;
    return ret_val;
} /* dgamln_ */

double gamln_(float* z__, int32* ierr)
{
    /* Initialized data */

    static float gln[100] = {(float)0.,
                            (float)0.,
                            (float).693147180559945309,
                            (float)1.791759469228055,
                            (float)3.17805383034794562,
                            (float)4.78749174278204599,
                            (float)6.579251212010101,
                            (float)8.5251613610654143,
                            (float)10.6046029027452502,
                            (float)12.8018274800814696,
                            (float)15.1044125730755153,
                            (float)17.5023078458738858,
                            (float)19.9872144956618861,
                            (float)22.5521638531234229,
                            (float)25.1912211827386815,
                            (float)27.8992713838408916,
                            (float)30.6718601060806728,
                            (float)33.5050734501368889,
                            (float)36.3954452080330536,
                            (float)39.339884187199494,
                            (float)42.335616460753485,
                            (float)45.380138898476908,
                            (float)48.4711813518352239,
                            (float)51.6066755677643736,
                            (float)54.7847293981123192,
                            (float)58.0036052229805199,
                            (float)61.261701761002002,
                            (float)64.5575386270063311,
                            (float)67.889743137181535,
                            (float)71.257038967168009,
                            (float)74.6582363488301644,
                            (float)78.0922235533153106,
                            (float)81.5579594561150372,
                            (float)85.0544670175815174,
                            (float)88.5808275421976788,
                            (float)92.1361756036870925,
                            (float)95.7196945421432025,
                            (float)99.3306124547874269,
                            (float)102.968198614513813,
                            (float)106.631760260643459,
                            (float)110.320639714757395,
                            (float)114.034211781461703,
                            (float)117.771881399745072,
                            (float)121.533081515438634,
                            (float)125.317271149356895,
                            (float)129.123933639127215,
                            (float)132.95257503561631,
                            (float)136.802722637326368,
                            (float)140.673923648234259,
                            (float)144.565743946344886,
                            (float)148.477766951773032,
                            (float)152.409592584497358,
                            (float)156.360836303078785,
                            (float)160.331128216630907,
                            (float)164.320112263195181,
                            (float)168.327445448427652,
                            (float)172.352797139162802,
                            (float)176.395848406997352,
                            (float)180.456291417543771,
                            (float)184.533828861449491,
                            (float)188.628173423671591,
                            (float)192.739047287844902,
                            (float)196.866181672889994,
                            (float)201.009316399281527,
                            (float)205.168199482641199,
                            (float)209.342586752536836,
                            (float)213.532241494563261,
                            (float)217.736934113954227,
                            (float)221.956441819130334,
                            (float)226.190548323727593,
                            (float)230.439043565776952,
                            (float)234.701723442818268,
                            (float)238.978389561834323,
                            (float)243.268849002982714,
                            (float)247.572914096186884,
                            (float)251.890402209723194,
                            (float)256.221135550009525,
                            (float)260.564940971863209,
                            (float)264.921649798552801,
                            (float)269.291097651019823,
                            (float)273.673124285693704,
                            (float)278.067573440366143,
                            (float)282.474292687630396,
                            (float)286.893133295426994,
                            (float)291.323950094270308,
                            (float)295.766601350760624,
                            (float)300.220948647014132,
                            (float)304.686856765668715,
                            (float)309.164193580146922,
                            (float)313.652829949879062,
                            (float)318.152639620209327,
                            (float)322.663499126726177,
                            (float)327.185287703775217,
                            (float)331.717887196928473,
                            (float)336.261181979198477,
                            (float)340.815058870799018,
                            (float)345.379407062266854,
                            (float)349.954118040770237,
                            (float)354.539085519440809,
                            (float)359.134205369575399};
    static float cf[22]   = {(float).0833333333333333333,   (float)-.00277777777777777778, (float)7.93650793650793651e-4, (float)-5.95238095238095238e-4,
                          (float)8.41750841750841751e-4, (float)-.00191752691752691753, (float).00641025641025641026,  (float)-.0295506535947712418,
                          (float).179644372368830573,    (float)-1.39243221690590112,   (float)13.402864044168392,     (float)-156.848284626002017,
                          (float)2193.10333333333333,    (float)-36108.7712537249894,   (float)691472.268851313067,    (float)-15238221.5394074162,
                          (float)382900751.391414141,    (float)-10882266035.7843911,   (float)347320283765.002252,    (float)-12369602142269.2745,
                          (float)488788064793079.335,    (float)-21320333960919373.9};
    static float con      = (float)1.83787706640934548;

    /* System generated locals */
    int32 i__1;
    float    ret_val;

    
    int32           i__, k;
    float              s, t1, fz;
    int32           mz, nz;
    float              zm, zp;
    int32           i1m;
    float              fln, tlg, rln, trm, tst, zsq, zinc, zmin, zdmy, wdtol;

    

    /* *********************************************************************72 */

    /* ***BEGIN PROLOGUE  GAMLN */
    /* ***DATE WRITTEN   830501   (YYMMDD) */
    /* ***REVISION DATE  830501   (YYMMDD) */
    /* ***CATEGORY NO.  B5F */
    /* ***KEYWORDS  GAMMA FUNCTION,LOGARITHM OF GAMMA FUNCTION */
    /* ***AUTHOR  AMOS, DONALD E., SANDIA NATIONAL LABORATORIES */
    /* ***PURPOSE  TO COMPUTE THE LOGARITHM OF THE GAMMA FUNCTION */
    /* ***DESCRIPTION */

    /*         GAMLN COMPUTES THE NATURAL LOG OF THE GAMMA FUNCTION FOR */
    /*         Z.GT.0.  THE ASYMPTOTIC EXPANSION IS USED TO GENERATE VALUES */
    /*         GREATER THAN ZMIN WHICH ARE ADJUSTED BY THE RECURSION */
    /*         G(Z+1)=Z*G(Z) FOR Z.LE.ZMIN.  THE FUNCTION WAS MADE AS */
    /*         PORTABLE AS POSSIBLE BY COMPUTIMG ZMIN FROM THE NUMBER OF BASE */
    /*         10 DIGITS IN A WORD, RLN=AMAX1(-ALOG10(R1MACH(4)),0.5E-18) */
    /*         LIMITED TO 18 DIGITS OF (RELATIVE) ACCURACY. */

    /*         SINCE INTEGER ARGUMENTS ARE COMMON, A TABLE LOOK UP ON 100 */
    /*         VALUES IS USED FOR SPEED OF EXECUTION. */

    /*     DESCRIPTION OF ARGUMENTS */

    /*         INPUT */
    /*           Z      - REAL ARGUMENT, Z.GT.0.0E0 */

    /*         OUTPUT */
    /*           GAMLN  - NATURAL LOG OF THE GAMMA FUNCTION AT Z */
    /*           IERR   - ERROR FLAG */
    /*                    IERR=0, NORMAL RETURN, COMPUTATION COMPLETED */
    /*                    IERR=1, Z.LE.0.0E0,    NO COMPUTATION */

    /* ***REFERENCES  COMPUTATION OF BESSEL FUNCTIONS OF COMPLEX ARGUMENT */
    /*                 BY D. E. AMOS, SAND83-0083, MAY, 1983. */
    /* ***ROUTINES CALLED  I1MACH,R1MACH */
    /* ***END PROLOGUE  GAMLN */

    /*           LNGAMMA(N), N=1,100 */
    /*             COEFFICIENTS OF ASYMPTOTIC EXPANSION */

    /*             LN(2*PI) */

    /* ***FIRST EXECUTABLE STATEMENT  GAMLN */
    *ierr = 0;
    if(*z__ <= (float)0.)
    {
        goto L70;
    }
    if(*z__ > (float)101.)
    {
        goto L10;
    }
    nz = (int32)(*z__);
    fz = *z__ - (float)nz;
    if(fz > (float)0.)
    {
        goto L10;
    }
    if(nz > 100)
    {
        goto L10;
    }
    ret_val = gln[nz - 1];
    return ret_val;
L10:
    wdtol = r1mach_(&c__4);
    wdtol = max(wdtol, (float)5e-19);
    i1m   = i1mach_(&c__11);
    rln   = r1mach_(&c__5) * (float)i1m;
    fln   = std::min(rln, (float)20.);
    fln   = max(fln, (float)3.);
    fln += (float)-3.;
    zm   = fln * (float).3875 + (float)1.8;
    mz   = (int32)zm + 1;
    zmin = (float)mz;
    zdmy = *z__;
    zinc = (float)0.;
    if(*z__ >= zmin)
    {
        goto L20;
    }
    zinc = zmin - (float)nz;
    zdmy = *z__ + zinc;
L20:
    zp = (float)1. / zdmy;
    t1 = cf[0] * zp;
    s  = t1;
    if(zp < wdtol)
    {
        goto L40;
    }
    zsq = zp * zp;
    tst = t1 * wdtol;
    for(k = 2; k <= 22; ++k)
    {
        zp *= zsq;
        trm = cf[k - 1] * zp;
        if(abs(trm) < tst)
        {
            goto L40;
        }
        s += trm;
        /* L30: */
    }
L40:
    if(zinc != (float)0.)
    {
        goto L50;
    }
    tlg     = log(*z__);
    ret_val = *z__ * (tlg - (float)1.) + (con - tlg) * (float).5 + s;
    return ret_val;
L50:
    zp   = (float)1.;
    nz   = (int32)zinc;
    i__1 = nz;
    for(i__ = 1; i__ <= i__1; ++i__)
    {
        zp *= *z__ + (float)(i__ - 1);
        /* L60: */
    }
    tlg     = log(zdmy);
    ret_val = zdmy * (tlg - (float)1.) - log(zp) + (con - tlg) * (float).5 + s;
    return ret_val;

L70:
    *ierr = 1;
    return ret_val;
} /* gamln_ */

//int32 i1mach_(int32* i__)
//{
//    /* System generated locals */
//    int32 ret_val;
//    cilist  ci__1;
//
//    /* Builtin functions */
//    int32              s_wsfe(cilist*), do_fio(int32*, char*, ftnlen), e_wsfe();
//     int s_stop(char*, ftnlen);
//
//    /* *********************************************************************72 */
//
//    /* c I1MACH returns int32 machine dependent constants. */
//
//    /*  Discussion: */
//
//    /*    Input/output unit numbers. */
//
//    /*      I1MACH(1) = the standard input unit. */
//    /*      I1MACH(2) = the standard output unit. */
//    /*      I1MACH(3) = the standard punch unit. */
//    /*      I1MACH(4) = the standard error message unit. */
//
//    /*    Words. */
//
//    /*      I1MACH(5) = the number of bits per int32 storage unit. */
//    /*      I1MACH(6) = the number of characters per int32 storage unit. */
//
//    /*    Integers. */
//
//    /*    Assume integers are represented in the S digit base A form: */
//
//    /*      Sign * (X(S-1)*A**(S-1) + ... + X(1)*A + X(0)) */
//
//    /*    where 0 <= X(1:S-1) < A. */
//
//    /*      I1MACH(7) = A, the base. */
//    /*      I1MACH(8) = S, the number of base A digits. */
//    /*      I1MACH(9) = A**S-1, the largest int32. */
//
//    /*    Floating point numbers */
//
//    /*    Assume floating point numbers are represented in the T digit */
//    /*    base B form: */
//
//    /*      Sign * (B**E) * ((X(1)/B) + ... + (X(T)/B**T) ) */
//
//    /*    where 0 <= X(I) < B for I=1 to T, 0 < X(1) and EMIN <= E <= EMAX. */
//
//    /*      I1MACH(10) = B, the base. */
//
//    /*    Single precision */
//
//    /*      I1MACH(11) = T, the number of base B digits. */
//    /*      I1MACH(12) = EMIN, the smallest exponent E. */
//    /*      I1MACH(13) = EMAX, the largest exponent E. */
//
//    /*    Double precision */
//
//    /*      I1MACH(14) = T, the number of base B digits. */
//    /*      I1MACH(15) = EMIN, the smallest exponent E. */
//    /*      I1MACH(16) = EMAX, the largest exponent E. */
//
//    /*  Licensing: */
//
//    /*    This code is distributed under the GNU LGPL license. */
//
//    /*  Modified: */
//
//    /*    25 April 2007 */
//
//    /*  Author: */
//
//    /*    Original FORTRAN77 version by Phyllis Fox, Andrew Hall, Norman Schryer. */
//    /*    This FORTRAN77 version by John Burkardt. */
//
//    /*  Reference: */
//
//    /*    Phyllis Fox, Andrew Hall, Norman Schryer, */
//    /*    Algorithm 528, */
//    /*    Framework for a Portable Library, */
//    /*    ACM Transactions on Mathematical Software, */
//    /*    Volume 4, Number 2, June 1978, page 176-188. */
//
//    /*  Parameters: */
//
//    /*    Input, int32 I, chooses the parameter to be returned. */
//    /*    1 <= I <= 16. */
//
//    /*    Output, int32 I1MACH, the value of the chosen parameter. */
//
//    if(*i__ < 1)
//    {
//        ci__1.cierr  = 0;
//        ci__1.ciunit = 6;
//        ci__1.cifmt  = "(a)";
//        s_wsfe(&ci__1);
//        do_fio(&c__1, " ", (ftnlen)1);
//        e_wsfe();
//        ci__1.cierr  = 0;
//        ci__1.ciunit = 6;
//        ci__1.cifmt  = "(a)";
//        s_wsfe(&ci__1);
//        do_fio(&c__1, "I1MACH - Fatal error!", (ftnlen)21);
//        e_wsfe();
//        ci__1.cierr  = 0;
//        ci__1.ciunit = 6;
//        ci__1.cifmt  = "(a)";
//        s_wsfe(&ci__1);
//        do_fio(&c__1, "  The input argument I is out of bounds.", (ftnlen)40);
//        e_wsfe();
//        ci__1.cierr  = 0;
//        ci__1.ciunit = 6;
//        ci__1.cifmt  = "(a)";
//        s_wsfe(&ci__1);
//        do_fio(&c__1, "  Legal values satisfy 1 <= I <= 16.", (ftnlen)36);
//        e_wsfe();
//        ci__1.cierr  = 0;
//        ci__1.ciunit = 6;
//        ci__1.cifmt  = "(a,i12)";
//        s_wsfe(&ci__1);
//        do_fio(&c__1, "  I = ", (ftnlen)6);
//        do_fio(&c__1, (char*)&(*i__), (ftnlen)sizeof(int32));
//        e_wsfe();
//        ret_val = 0;
//        s_stop("", (ftnlen)0);
//    }
//    else if(*i__ == 1)
//    {
//        ret_val = 5;
//    }
//    else if(*i__ == 2)
//    {
//        ret_val = 6;
//    }
//    else if(*i__ == 3)
//    {
//        ret_val = 7;
//    }
//    else if(*i__ == 4)
//    {
//        ret_val = 6;
//    }
//    else if(*i__ == 5)
//    {
//        ret_val = 32;
//    }
//    else if(*i__ == 6)
//    {
//        ret_val = 4;
//    }
//    else if(*i__ == 7)
//    {
//        ret_val = 2;
//    }
//    else if(*i__ == 8)
//    {
//        ret_val = 31;
//    }
//    else if(*i__ == 9)
//    {
//        ret_val = 2147483647;
//    }
//    else if(*i__ == 10)
//    {
//        ret_val = 2;
//    }
//    else if(*i__ == 11)
//    {
//        ret_val = 24;
//    }
//    else if(*i__ == 12)
//    {
//        ret_val = -125;
//    }
//    else if(*i__ == 13)
//    {
//        ret_val = 128;
//    }
//    else if(*i__ == 14)
//    {
//        ret_val = 53;
//    }
//    else if(*i__ == 15)
//    {
//        ret_val = -1021;
//    }
//    else if(*i__ == 16)
//    {
//        ret_val = 1024;
//    }
//    else if(16 < *i__)
//    {
//        ci__1.cierr  = 0;
//        ci__1.ciunit = 6;
//        ci__1.cifmt  = "(a)";
//        s_wsfe(&ci__1);
//        do_fio(&c__1, " ", (ftnlen)1);
//        e_wsfe();
//        ci__1.cierr  = 0;
//        ci__1.ciunit = 6;
//        ci__1.cifmt  = "(a)";
//        s_wsfe(&ci__1);
//        do_fio(&c__1, "I1MACH - Fatal error!", (ftnlen)21);
//        e_wsfe();
//        ci__1.cierr  = 0;
//        ci__1.ciunit = 6;
//        ci__1.cifmt  = "(a)";
//        s_wsfe(&ci__1);
//        do_fio(&c__1, "  The input argument I is out of bounds.", (ftnlen)40);
//        e_wsfe();
//        ci__1.cierr  = 0;
//        ci__1.ciunit = 6;
//        ci__1.cifmt  = "(a)";
//        s_wsfe(&ci__1);
//        do_fio(&c__1, "  Legal values satisfy 1 <= I <= 16.", (ftnlen)36);
//        e_wsfe();
//        ci__1.cierr  = 0;
//        ci__1.ciunit = 6;
//        ci__1.cifmt  = "(a,i12)";
//        s_wsfe(&ci__1);
//        do_fio(&c__1, "  I = ", (ftnlen)6);
//        do_fio(&c__1, (char*)&(*i__), (ftnlen)sizeof(int32));
//        e_wsfe();
//        ret_val = 0;
//        s_stop("", (ftnlen)0);
//    }
//    return ret_val;
//} /* i1mach_ */

//double r1mach_(int32* i__)
//{
//    /* System generated locals */
//    float   ret_val;
//    cilist ci__1;
//
//    /* Builtin functions */
//    int32              s_wsfe(cilist*), do_fio(int32*, char*, ftnlen), e_wsfe();
//     int s_stop(char*, ftnlen);
//
//    /* *********************************************************************72 */
//
//    /* c R1MACH returns single precision float machine constants. */
//
//    /*  Discussion: */
//
//    /*    Assume that single precision float numbers are stored with a mantissa */
//    /*    of T digits in base B, with an exponent whose value must lie */
//    /*    between EMIN and EMAX.  Then for values of I between 1 and 5, */
//    /*    R1MACH will return the following values: */
//
//    /*      R1MACH(1) = B**(EMIN-1), the smallest positive magnitude. */
//    /*      R1MACH(2) = B**EMAX*(1-B**(-T)), the largest magnitude. */
//    /*      R1MACH(3) = B**(-T), the smallest relative spacing. */
//    /*      R1MACH(4) = B**(1-T), the largest relative spacing. */
//    /*      R1MACH(5) = log10(B) */
//
//    /*  Licensing: */
//
//    /*    This code is distributed under the GNU LGPL license. */
//
//    /*  Modified: */
//
//    /*    25 April 2007 */
//
//    /*  Author: */
//
//    /*    Original FORTRAN77 version by Phyllis Fox, Andrew Hall, Norman Schryer. */
//    /*    This FORTRAN77 version by John Burkardt. */
//
//    /*  Reference: */
//
//    /*    Phyllis Fox, Andrew Hall, Norman Schryer, */
//    /*    Algorithm 528, */
//    /*    Framework for a Portable Library, */
//    /*    ACM Transactions on Mathematical Software, */
//    /*    Volume 4, Number 2, June 1978, page 176-188. */
//
//    /*  Parameters: */
//
//    /*    Input, int32 I, chooses the parameter to be returned. */
//    /*    1 <= I <= 5. */
//
//    /*    Output, float R1MACH, the value of the chosen parameter. */
//
//    if(*i__ < 1)
//    {
//        ci__1.cierr  = 0;
//        ci__1.ciunit = 6;
//        ci__1.cifmt  = "(a)";
//        s_wsfe(&ci__1);
//        do_fio(&c__1, " ", (ftnlen)1);
//        e_wsfe();
//        ci__1.cierr  = 0;
//        ci__1.ciunit = 6;
//        ci__1.cifmt  = "(a)";
//        s_wsfe(&ci__1);
//        do_fio(&c__1, "R1MACH - Fatal error!", (ftnlen)21);
//        e_wsfe();
//        ci__1.cierr  = 0;
//        ci__1.ciunit = 6;
//        ci__1.cifmt  = "(a)";
//        s_wsfe(&ci__1);
//        do_fio(&c__1, "  The input argument I is out of bounds.", (ftnlen)40);
//        e_wsfe();
//        ci__1.cierr  = 0;
//        ci__1.ciunit = 6;
//        ci__1.cifmt  = "(a)";
//        s_wsfe(&ci__1);
//        do_fio(&c__1, "  Legal values satisfy 1 <= I <= 5.", (ftnlen)35);
//        e_wsfe();
//        ci__1.cierr  = 0;
//        ci__1.ciunit = 6;
//        ci__1.cifmt  = "(a,i12)";
//        s_wsfe(&ci__1);
//        do_fio(&c__1, "  I = ", (ftnlen)6);
//        do_fio(&c__1, (char*)&(*i__), (ftnlen)sizeof(int32));
//        e_wsfe();
//        ret_val = (float)0.;
//        s_stop("", (ftnlen)0);
//    }
//    else if(*i__ == 1)
//    {
//        ret_val = (float)1.1754944e-38;
//    }
//    else if(*i__ == 2)
//    {
//        ret_val = (float)3.4028235e38;
//    }
//    else if(*i__ == 3)
//    {
//        ret_val = (float)5.9604645e-8;
//    }
//    else if(*i__ == 4)
//    {
//        ret_val = (float)1.1920929e-7;
//    }
//    else if(*i__ == 5)
//    {
//        ret_val = (float).30103;
//    }
//    else if(5 < *i__)
//    {
//        ci__1.cierr  = 0;
//        ci__1.ciunit = 6;
//        ci__1.cifmt  = "(a)";
//        s_wsfe(&ci__1);
//        do_fio(&c__1, " ", (ftnlen)1);
//        e_wsfe();
//        ci__1.cierr  = 0;
//        ci__1.ciunit = 6;
//        ci__1.cifmt  = "(a)";
//        s_wsfe(&ci__1);
//        do_fio(&c__1, "R1MACH - Fatal error!", (ftnlen)21);
//        e_wsfe();
//        ci__1.cierr  = 0;
//        ci__1.ciunit = 6;
//        ci__1.cifmt  = "(a)";
//        s_wsfe(&ci__1);
//        do_fio(&c__1, "  The input argument I is out of bounds.", (ftnlen)40);
//        e_wsfe();
//        ci__1.cierr  = 0;
//        ci__1.ciunit = 6;
//        ci__1.cifmt  = "(a)";
//        s_wsfe(&ci__1);
//        do_fio(&c__1, "  Legal values satisfy 1 <= I <= 5.", (ftnlen)35);
//        e_wsfe();
//        ci__1.cierr  = 0;
//        ci__1.ciunit = 6;
//        ci__1.cifmt  = "(a,i12)";
//        s_wsfe(&ci__1);
//        do_fio(&c__1, "  I = ", (ftnlen)6);
//        do_fio(&c__1, (char*)&(*i__), (ftnlen)sizeof(int32));
//        e_wsfe();
//        ret_val = (float)0.;
//        s_stop("", (ftnlen)0);
//    }
//    return ret_val;
//} /* r1mach_ */


#undef month_ref

double zabs_(double* zr, double* zi)
{
    /* System generated locals */
    double ret_val;

 
    
    double q, s, u, v;

    /* *********************************************************************72 */

    /* c ZABS carries out double precision System::Complex<float> absolute values. */

    /* ***BEGIN PROLOGUE  ZABS */
    /* ***REFER TO  ZBESH,ZBESI,ZBESJ,ZBESK,ZBESY,ZAIRY,ZBIRY */

    /*     ZABS COMPUTES THE ABSOLUTE VALUE OR MAGNITUDE OF A DOUBLE */
    /*     PRECISION COMPLEX VARIABLE CMPLX(ZR,ZI) */

    /* ***ROUTINES CALLED  (NONE) */
    /* ***END PROLOGUE  ZABS */
    u = abs(*zr);
    v = abs(*zi);
    s = u + v;
    /* ----------------------------------------------------------------------- */
    /*     S*1.0D0 MAKES AN UNNORMALIZED UNDERFLOW ON CDC MACHINES INTO A */
    /*     TRUE FLOATING ZERO */
    /* ----------------------------------------------------------------------- */
    s *= 1.;
    if(s == 0.)
    {
        goto L20;
    }
    if(u > v)
    {
        goto L10;
    }
    q       = u / v;
    ret_val = v * sqrt(q * q + 1.);
    return ret_val;
L10:
    q       = v / u;
    ret_val = u * sqrt(q * q + 1.);
    return ret_val;
L20:
    ret_val = 0.;
    return ret_val;
} /* zabs_ */

 int zacai_(double* zr,
                            double* zi,
                            double* fnu,
                            int32*    kode,
                            int32*    mr,
                            int32*    n,
                            double* yr,
                            double* yi,
                            int32*    nz,
                            double* rl,
                            double* tol,
                            double* elim,
                            double* alim)
{
    /* Initialized data */

    static double pi = 3.14159265358979324;


    
    double                  az;
    int32                     nn, nw;
    double                  yy, c1i, c2i, c1r, c2r, arg;
    int32                     iuf;
    double                  cyi[2], fmr, sgn;
    int32                     inu;
    double                  cyr[2], zni, znr, dfnu;
    
    double                  ascle, csgni, csgnr, cspni, cspnr;

    /* *********************************************************************72 */

    /* ***BEGIN PROLOGUE  ZACAI */
    /* ***REFER TO  ZAIRY */

    /*     ZACAI APPLIES THE ANALYTIC CONTINUATION FORMULA */

    /*         K(FNU,ZN*EXP(MP))=K(FNU,ZN)*EXP(-MP*FNU) - MP*I(FNU,ZN) */
    /*                 MP=PI*MR*CMPLX(0.0,1.0) */

    /*     TO CONTINUE THE K FUNCTION FROM THE RIGHT HALF TO THE LEFT */
    /*     HALF Z PLANE FOR USE WITH ZAIRY WHERE FNU=1/3 OR 2/3 AND N=1. */
    /*     ZACAI IS THE SAME AS ZACON WITH THE PARTS FOR LARGER ORDERS AND */
    /*     RECURRENCE REMOVED. A RECURSIVE CALL TO ZACON CAN RESULT IF ZACON */
    /*     IS CALLED FROM ZAIRY. */

    /* ***ROUTINES CALLED  ZASYI,ZBKNU,ZMLRI,ZSERI,ZS1S2,D1MACH,ZABS */
    /* ***END PROLOGUE  ZACAI */
    /*     COMPLEX CSGN,CSPN,C1,C2,Y,Z,ZN,CY */
    /* Parameter adjustments */
    --yi;
    --yr;

    /* Function Body */
    *nz  = 0;
    znr  = -(*zr);
    zni  = -(*zi);
    az   = zabs_(zr, zi);
    nn   = *n;
    dfnu = *fnu + (double)((float)(*n - 1));
    if(az <= 2.)
    {
        goto L10;
    }
    if(az * az * .25 > dfnu + 1.)
    {
        goto L20;
    }
L10:
    /* ----------------------------------------------------------------------- */
    /*     POWER SERIES FOR THE I FUNCTION */
    /* ----------------------------------------------------------------------- */
    zseri_(&znr, &zni, fnu, kode, &nn, &yr[1], &yi[1], &nw, tol, elim, alim);
    goto L40;
L20:
    if(az < *rl)
    {
        goto L30;
    }
    /* ----------------------------------------------------------------------- */
    /*     ASYMPTOTIC EXPANSION FOR LARGE Z FOR THE I FUNCTION */
    /* ----------------------------------------------------------------------- */
    zasyi_(&znr, &zni, fnu, kode, &nn, &yr[1], &yi[1], &nw, rl, tol, elim, alim);
    if(nw < 0)
    {
        goto L80;
    }
    goto L40;
L30:
    /* ----------------------------------------------------------------------- */
    /*     MILLER ALGORITHM NORMALIZED BY THE SERIES FOR THE I FUNCTION */
    /* ----------------------------------------------------------------------- */
    zmlri_(&znr, &zni, fnu, kode, &nn, &yr[1], &yi[1], &nw, tol);
    if(nw < 0)
    {
        goto L80;
    }
L40:
    /* ----------------------------------------------------------------------- */
    /*     ANALYTIC CONTINUATION TO THE LEFT HALF PLANE FOR THE K FUNCTION */
    /* ----------------------------------------------------------------------- */
    zbknu_(&znr, &zni, fnu, kode, &c__1, cyr, cyi, &nw, tol, elim, alim);
    if(nw != 0)
    {
        goto L80;
    }
    fmr   = (double)((float)(*mr));
    sgn   = -d_sign(&pi, &fmr);
    csgnr = 0.;
    csgni = sgn;
    if(*kode == 1)
    {
        goto L50;
    }
    yy    = -zni;
    csgnr = -csgni * sin(yy);
    csgni *= cos(yy);
L50:
    /* ----------------------------------------------------------------------- */
    /*     CALCULATE CSPN=EXP(FNU*PI*I) TO MINIMIZE LOSSES OF SIGNIFICANCE */
    /*     WHEN FNU IS LARGE */
    /* ----------------------------------------------------------------------- */
    inu   = (int32)((float)(*fnu));
    arg   = (*fnu - (double)((float)inu)) * sgn;
    cspnr = cos(arg);
    cspni = sin(arg);
    if(inu % 2 == 0)
    {
        goto L60;
    }
    cspnr = -cspnr;
    cspni = -cspni;
L60:
    c1r = cyr[0];
    c1i = cyi[0];
    c2r = yr[1];
    c2i = yi[1];
    if(*kode == 1)
    {
        goto L70;
    }
    iuf   = 0;
    ascle = d1mach_(&c__1) * 1e3 / *tol;
    zs1s2_(&znr, &zni, &c1r, &c1i, &c2r, &c2i, &nw, &ascle, alim, &iuf);
    *nz += nw;
L70:
    yr[1] = cspnr * c1r - cspni * c1i + csgnr * c2r - csgni * c2i;
    yi[1] = cspnr * c1i + cspni * c1r + csgnr * c2i + csgni * c2r;
    return 0;
L80:
    *nz = -1;
    if(nw == -2)
    {
        *nz = -2;
    }
    return 0;
} /* zacai_ */

 int zacon_(double* zr,
                            double* zi,
                            double* fnu,
                            int32*    kode,
                            int32*    mr,
                            int32*    n,
                            double* yr,
                            double* yi,
                            int32*    nz,
                            double* rl,
                            double* fnul,
                            double* tol,
                            double* elim,
                            double* alim)
{
    /* Initialized data */

    static double pi    = 3.14159265358979324;
    static double zeror = 0.;
    static double coner = 1.;

    /* System generated locals */
    int32 i__1;

    
    int32                     i__;
    double                  fn;
    int32                     nn, nw;
    double                  yy, c1i, c2i, c1m, as2, c1r, c2r, s1i, s2i, s1r, s2r, cki, arg, ckr, cpn;
    int32                     iuf;
    double                  cyi[2], fmr, csr, azn, sgn;
    int32                     inu;
    double                  bry[3], cyr[2], pti, spn, sti, zni, rzi, ptr, str, znr, rzr, sc1i, sc2i, sc1r, sc2r, cscl, cscr;
    
    double                  csrr[3], cssr[3], razn;
    int32                     kflag;
    double                  ascle, bscle, csgni, csgnr, cspni, cspnr;

    /* *********************************************************************72 */

    /* ***BEGIN PROLOGUE  ZACON */
    /* ***REFER TO  ZBESK,ZBESH */

    /*     ZACON APPLIES THE ANALYTIC CONTINUATION FORMULA */

    /*         K(FNU,ZN*EXP(MP))=K(FNU,ZN)*EXP(-MP*FNU) - MP*I(FNU,ZN) */
    /*                 MP=PI*MR*CMPLX(0.0,1.0) */

    /*     TO CONTINUE THE K FUNCTION FROM THE RIGHT HALF TO THE LEFT */
    /*     HALF Z PLANE */

    /* ***ROUTINES CALLED  ZBINU,ZBKNU,ZS1S2,D1MACH,ZABS,ZMLT */
    /* ***END PROLOGUE  ZACON */
    /*     COMPLEX CK,CONE,CSCL,CSCR,CSGN,CSPN,CY,CZERO,C1,C2,RZ,SC1,SC2,ST, */
    /*    *S1,S2,Y,Z,ZN */
    /* Parameter adjustments */
    --yi;
    --yr;

    /* Function Body */
    *nz = 0;
    znr = -(*zr);
    zni = -(*zi);
    nn  = *n;
    zbinu_(&znr, &zni, fnu, kode, &nn, &yr[1], &yi[1], &nw, rl, fnul, tol, elim, alim);
    if(nw < 0)
    {
        goto L90;
    }
    /* ----------------------------------------------------------------------- */
    /*     ANALYTIC CONTINUATION TO THE LEFT HALF PLANE FOR THE K FUNCTION */
    /* ----------------------------------------------------------------------- */
    nn = std::min(2L, *n);
    zbknu_(&znr, &zni, fnu, kode, &nn, cyr, cyi, &nw, tol, elim, alim);
    if(nw != 0)
    {
        goto L90;
    }
    s1r   = cyr[0];
    s1i   = cyi[0];
    fmr   = (double)((float)(*mr));
    sgn   = -d_sign(&pi, &fmr);
    csgnr = zeror;
    csgni = sgn;
    if(*kode == 1)
    {
        goto L10;
    }
    yy  = -zni;
    cpn = cos(yy);
    spn = sin(yy);
    zmlt_(&csgnr, &csgni, &cpn, &spn, &csgnr, &csgni);
L10:
    /* ----------------------------------------------------------------------- */
    /*     CALCULATE CSPN=EXP(FNU*PI*I) TO MINIMIZE LOSSES OF SIGNIFICANCE */
    /*     WHEN FNU IS LARGE */
    /* ----------------------------------------------------------------------- */
    inu   = (int32)((float)(*fnu));
    arg   = (*fnu - (double)((float)inu)) * sgn;
    cpn   = cos(arg);
    spn   = sin(arg);
    cspnr = cpn;
    cspni = spn;
    if(inu % 2 == 0)
    {
        goto L20;
    }
    cspnr = -cspnr;
    cspni = -cspni;
L20:
    iuf   = 0;
    c1r   = s1r;
    c1i   = s1i;
    c2r   = yr[1];
    c2i   = yi[1];
    ascle = d1mach_(&c__1) * 1e3 / *tol;
    if(*kode == 1)
    {
        goto L30;
    }
    zs1s2_(&znr, &zni, &c1r, &c1i, &c2r, &c2i, &nw, &ascle, alim, &iuf);
    *nz += nw;
    sc1r = c1r;
    sc1i = c1i;
L30:
    zmlt_(&cspnr, &cspni, &c1r, &c1i, &str, &sti);
    zmlt_(&csgnr, &csgni, &c2r, &c2i, &ptr, &pti);
    yr[1] = str + ptr;
    yi[1] = sti + pti;
    if(*n == 1)
    {
        return 0;
    }
    cspnr = -cspnr;
    cspni = -cspni;
    s2r   = cyr[1];
    s2i   = cyi[1];
    c1r   = s2r;
    c1i   = s2i;
    c2r   = yr[2];
    c2i   = yi[2];
    if(*kode == 1)
    {
        goto L40;
    }
    zs1s2_(&znr, &zni, &c1r, &c1i, &c2r, &c2i, &nw, &ascle, alim, &iuf);
    *nz += nw;
    sc2r = c1r;
    sc2i = c1i;
L40:
    zmlt_(&cspnr, &cspni, &c1r, &c1i, &str, &sti);
    zmlt_(&csgnr, &csgni, &c2r, &c2i, &ptr, &pti);
    yr[2] = str + ptr;
    yi[2] = sti + pti;
    if(*n == 2)
    {
        return 0;
    }
    cspnr = -cspnr;
    cspni = -cspni;
    azn   = zabs_(&znr, &zni);
    razn  = 1. / azn;
    str   = znr * razn;
    sti   = -zni * razn;
    rzr   = (str + str) * razn;
    rzi   = (sti + sti) * razn;
    fn    = *fnu + 1.;
    ckr   = fn * rzr;
    cki   = fn * rzi;
    /* ----------------------------------------------------------------------- */
    /*     SCALE NEAR EXPONENT EXTREMES DURING RECURRENCE ON K FUNCTIONS */
    /* ----------------------------------------------------------------------- */
    cscl    = 1. / *tol;
    cscr    = *tol;
    cssr[0] = cscl;
    cssr[1] = coner;
    cssr[2] = cscr;
    csrr[0] = cscr;
    csrr[1] = coner;
    csrr[2] = cscl;
    bry[0]  = ascle;
    bry[1]  = 1. / ascle;
    bry[2]  = d1mach_(&c__2);
    as2     = zabs_(&s2r, &s2i);
    kflag   = 2;
    if(as2 > bry[0])
    {
        goto L50;
    }
    kflag = 1;
    goto L60;
L50:
    if(as2 < bry[1])
    {
        goto L60;
    }
    kflag = 3;
L60:
    bscle = bry[kflag - 1];
    s1r *= cssr[kflag - 1];
    s1i *= cssr[kflag - 1];
    s2r *= cssr[kflag - 1];
    s2i *= cssr[kflag - 1];
    csr  = csrr[kflag - 1];
    i__1 = *n;
    for(i__ = 3; i__ <= i__1; ++i__)
    {
        str = s2r;
        sti = s2i;
        s2r = ckr * str - cki * sti + s1r;
        s2i = ckr * sti + cki * str + s1i;
        s1r = str;
        s1i = sti;
        c1r = s2r * csr;
        c1i = s2i * csr;
        str = c1r;
        sti = c1i;
        c2r = yr[i__];
        c2i = yi[i__];
        if(*kode == 1)
        {
            goto L70;
        }
        if(iuf < 0)
        {
            goto L70;
        }
        zs1s2_(&znr, &zni, &c1r, &c1i, &c2r, &c2i, &nw, &ascle, alim, &iuf);
        *nz += nw;
        sc1r = sc2r;
        sc1i = sc2i;
        sc2r = c1r;
        sc2i = c1i;
        if(iuf != 3)
        {
            goto L70;
        }
        iuf = -4;
        s1r = sc1r * cssr[kflag - 1];
        s1i = sc1i * cssr[kflag - 1];
        s2r = sc2r * cssr[kflag - 1];
        s2i = sc2i * cssr[kflag - 1];
        str = sc2r;
        sti = sc2i;
    L70:
        ptr     = cspnr * c1r - cspni * c1i;
        pti     = cspnr * c1i + cspni * c1r;
        yr[i__] = ptr + csgnr * c2r - csgni * c2i;
        yi[i__] = pti + csgnr * c2i + csgni * c2r;
        ckr += rzr;
        cki += rzi;
        cspnr = -cspnr;
        cspni = -cspni;
        if(kflag >= 3)
        {
            goto L80;
        }
        ptr = abs(c1r);
        pti = abs(c1i);
        c1m = max(ptr, pti);
        if(c1m <= bscle)
        {
            goto L80;
        }
        ++kflag;
        bscle = bry[kflag - 1];
        s1r *= csr;
        s1i *= csr;
        s2r = str;
        s2i = sti;
        s1r *= cssr[kflag - 1];
        s1i *= cssr[kflag - 1];
        s2r *= cssr[kflag - 1];
        s2i *= cssr[kflag - 1];
        csr = csrr[kflag - 1];
    L80:;
    }
    return 0;
L90:
    *nz = -1;
    if(nw == -2)
    {
        *nz = -2;
    }
    return 0;
} /* zacon_ */

 int zairy_(double* zr, double* zi, int32* id, int32* kode, double* air, double* aii, int32* nz, int32* ierr)
{
    /* Initialized data */

    static double tth   = .666666666666666667;
    static double c1    = .35502805388781724;
    static double c2    = .258819403792806799;
    static double coef  = .183776298473930683;
    static double zeror = 0.;
    static double zeroi = 0.;
    static double coner = 1.;
    static double conei = 0.;

    /* System generated locals */
    int32    i__1, i__2;
    double d__1;

    
    int32                     k;
    double                  d1, d2;
    int32                     k1, k2;
    double                  aa, bb, ad, cc, ak, bk, ck, dk, az;
    int32                     nn;
    double                  rl;
    int32                     mr;
    double                  s1i, az3, s2i, s1r, s2r, z3i, z3r, dig, fid, cyi[1], r1m5, fnu, cyr[1], tol, sti, ptr, str, sfac, alim, elim, alaz;

    double                  csqi, atrm, ztai, csqr, ztar;
    double                  trm1i, trm2i, trm1r, trm2r;
    int32                     iflag;


    /* *********************************************************************72 */

    /* c ZAIRY computes a sequence of System::Complex<float> Airy Ai functions. */

    /* ***BEGIN PROLOGUE  ZAIRY */
    /* ***DATE WRITTEN   830501   (YYMMDD) */
    /* ***REVISION DATE  890801, 930101   (YYMMDD) */
    /* ***CATEGORY NO.  B5K */
    /* ***KEYWORDS  AIRY FUNCTION,BESSEL FUNCTIONS OF ORDER ONE THIRD */
    /* ***AUTHOR  AMOS, DONALD E., SANDIA NATIONAL LABORATORIES */
    /* ***PURPOSE  TO COMPUTE AIRY FUNCTIONS AI(Z) AND DAI(Z) FOR COMPLEX Z */
    /* ***DESCRIPTION */

    /*                      ***A DOUBLE PRECISION ROUTINE*** */
    /*         ON KODE=1, ZAIRY COMPUTES THE COMPLEX AIRY FUNCTION AI(Z) OR */
    /*         ITS DERIVATIVE DAI(Z)/DZ ON ID=0 OR ID=1 RESPECTIVELY. ON */
    /*         KODE=2, A SCALING OPTION CEXP(ZTA)*AI(Z) OR CEXP(ZTA)* */
    /*         DAI(Z)/DZ IS PROVIDED TO REMOVE THE EXPONENTIAL DECAY IN */
    /*         -PI/3.LT.ARG(Z).LT.PI/3 AND THE EXPONENTIAL GROWTH IN */
    /*         PI/3.LT.ABS(ARG(Z)).LT.PI WHERE ZTA=(2/3)*Z*CSQRT(Z). */

    /*         WHILE THE AIRY FUNCTIONS AI(Z) AND DAI(Z)/DZ ARE ANALYTIC IN */
    /*         THE WHOLE Z PLANE, THE CORRESPONDING SCALED FUNCTIONS DEFINED */
    /*         FOR KODE=2 HAVE A CUT ALONG THE NEGATIVE REAL AXIS. */
    /*         DEFINTIONS AND NOTATION ARE FOUND IN THE NBS HANDBOOK OF */
    /*         MATHEMATICAL FUNCTIONS (CONST. 1). */

    /*         INPUT      ZR,ZI ARE DOUBLE PRECISION */
    /*           ZR,ZI  - Z=CMPLX(ZR,ZI) */
    /*           ID     - ORDER OF DERIVATIVE, ID=0 OR ID=1 */
    /*           KODE   - A PARAMETER TO INDICATE THE SCALING OPTION */
    /*                    KODE= 1  RETURNS */
    /*                             AI=AI(Z)                ON ID=0 OR */
    /*                             AI=DAI(Z)/DZ            ON ID=1 */
    /*                        = 2  RETURNS */
    /*                             AI=CEXP(ZTA)*AI(Z)       ON ID=0 OR */
    /*                             AI=CEXP(ZTA)*DAI(Z)/DZ   ON ID=1 WHERE */
    /*                             ZTA=(2/3)*Z*CSQRT(Z) */

    /*         OUTPUT     AIR,AII ARE DOUBLE PRECISION */
    /*           AIR,AII- COMPLEX ANSWER DEPENDING ON THE CHOICES FOR ID AND */
    /*                    KODE */
    /*           NZ     - UNDERFLOW INDICATOR */
    /*                    NZ= 0   , NORMAL RETURN */
    /*                    NZ= 1   , AI=CMPLX(0.0D0,0.0D0) DUE TO UNDERFLOW IN */
    /*                              -PI/3.LT.ARG(Z).LT.PI/3 ON KODE=1 */
    /*           IERR   - ERROR FLAG */
    /*                    IERR=0, NORMAL RETURN - COMPUTATION COMPLETED */
    /*                    IERR=1, INPUT ERROR   - NO COMPUTATION */
    /*                    IERR=2, OVERFLOW      - NO COMPUTATION, REAL(ZTA) */
    /*                            TOO LARGE ON KODE=1 */
    /*                    IERR=3, CABS(Z) LARGE      - COMPUTATION COMPLETED */
    /*                            LOSSES OF SIGNIFCANCE BY ARGUMENT REDUCTION */
    /*                            PRODUCE LESS THAN HALF OF MACHINE ACCURACY */
    /*                    IERR=4, CABS(Z) TOO LARGE  - NO COMPUTATION */
    /*                            COMPLETE LOSS OF ACCURACY BY ARGUMENT */
    /*                            REDUCTION */
    /*                    IERR=5, ERROR              - NO COMPUTATION, */
    /*                            ALGORITHM TERMINATION CONDITION NOT MET */

    /* ***LONG DESCRIPTION */

    /*         AI AND DAI ARE COMPUTED FOR CABS(Z).GT.1.0 FROM THE K BESSEL */
    /*         FUNCTIONS BY */

    /*            AI(Z)=C*SQRT(Z)*K(1/3,ZTA) , DAI(Z)=-C*Z*K(2/3,ZTA) */
    /*                           C=1.0/(PI*SQRT(3.0)) */
    /*                            ZTA=(2/3)*Z**(3/2) */

    /*         WITH THE POWER SERIES FOR CABS(Z).LE.1.0. */

    /*         IN MOST COMPLEX VARIABLE COMPUTATION, ONE MUST EVALUATE ELE- */
    /*         MENTARY FUNCTIONS. WHEN THE MAGNITUDE OF Z IS LARGE, LOSSES */
    /*         OF SIGNIFICANCE BY ARGUMENT REDUCTION OCCUR. CONSEQUENTLY, IF */
    /*         THE MAGNITUDE OF ZETA=(2/3)*Z**1.5 EXCEEDS U1=SQRT(0.5/UR), */
    /*         THEN LOSSES EXCEEDING HALF PRECISION ARE LIKELY AND AN ERROR */
    /*         FLAG IERR=3 IS TRIGGERED WHERE UR=DMAX1(D1MACH(4),1.0D-18) IS */
    /*         DOUBLE PRECISION UNIT ROUNDOFF LIMITED TO 18 DIGITS PRECISION. */
    /*         ALSO, IF THE MAGNITUDE OF ZETA IS LARGER THAN U2=0.5/UR, THEN */
    /*         ALL SIGNIFICANCE IS LOST AND IERR=4. IN ORDER TO USE THE INT */
    /*         FUNCTION, ZETA MUST BE FURTHER RESTRICTED NOT TO EXCEED THE */
    /*         LARGEST INTEGER, U3=I1MACH(9). THUS, THE MAGNITUDE OF ZETA */
    /*         MUST BE RESTRICTED BY MIN(U2,U3). ON 32 BIT MACHINES, U1,U2, */
    /*         AND U3 ARE APPROXIMATELY 2.0E+3, 4.2E+6, 2.1E+9 IN SINGLE */
    /*         PRECISION ARITHMETIC AND 1.3E+8, 1.8E+16, 2.1E+9 IN DOUBLE */
    /*         PRECISION ARITHMETIC RESPECTIVELY. THIS MAKES U2 AND U3 LIMIT- */
    /*         ING IN THEIR RESPECTIVE ARITHMETICS. THIS MEANS THAT THE MAG- */
    /*         NITUDE OF Z CANNOT EXCEED 3.1E+4 IN SINGLE AND 2.1E+6 IN */
    /*         DOUBLE PRECISION ARITHMETIC. THIS ALSO MEANS THAT ONE CAN */
    /*         EXPECT TO RETAIN, IN THE WORST CASES ON 32 BIT MACHINES, */
    /*         NO DIGITS IN SINGLE PRECISION AND ONLY 7 DIGITS IN DOUBLE */
    /*         PRECISION ARITHMETIC. SIMILAR CONSIDERATIONS HOLD FOR OTHER */
    /*         MACHINES. */

    /*         THE APPROXIMATE RELATIVE ERROR IN THE MAGNITUDE OF A COMPLEX */
    /*         BESSEL FUNCTION CAN BE EXPRESSED BY P*10**S WHERE P=MAX(UNIT */
    /*         ROUNDOFF,1.0E-18) IS THE NOMINAL PRECISION AND 10**S REPRE- */
    /*         SENTS THE INCREASE IN ERROR DUE TO ARGUMENT REDUCTION IN THE */
    /*         ELEMENTARY FUNCTIONS. HERE, S=MAX(1,ABS(LOG10(CABS(Z))), */
    /*         ABS(LOG10(FNU))) APPROXIMATELY (I.E. S=MAX(1,ABS(EXPONENT OF */
    /*         CABS(Z),ABS(EXPONENT OF FNU)) ). HOWEVER, THE PHASE ANGLE MAY */
    /*         HAVE ONLY ABSOLUTE ACCURACY. THIS IS MOST LIKELY TO OCCUR WHEN */
    /*         ONE COMPONENT (IN ABSOLUTE VALUE) IS LARGER THAN THE OTHER BY */
    /*         SEVERAL ORDERS OF MAGNITUDE. IF ONE COMPONENT IS 10**K LARGER */
    /*         THAN THE OTHER, THEN ONE CAN EXPECT ONLY MAX(ABS(LOG10(P))-K, */
    /*         0) SIGNIFICANT DIGITS; OR, STATED ANOTHER WAY, WHEN K EXCEEDS */
    /*         THE EXPONENT OF P, NO SIGNIFICANT DIGITS REMAIN IN THE SMALLER */
    /*         COMPONENT. HOWEVER, THE PHASE ANGLE RETAINS ABSOLUTE ACCURACY */
    /*         BECAUSE, IN COMPLEX ARITHMETIC WITH PRECISION P, THE SMALLER */
    /*         COMPONENT WILL NOT (AS A RULE) DECREASE BELOW P TIMES THE */
    /*         MAGNITUDE OF THE LARGER COMPONENT. IN THESE EXTREME CASES, */
    /*         THE PRINCIPAL PHASE ANGLE IS ON THE ORDER OF +P, -P, PI/2-P, */
    /*         OR -PI/2+P. */

    /* ***REFERENCES  HANDBOOK OF MATHEMATICAL FUNCTIONS BY M. ABRAMOWITZ */
    /*                 AND I. A. STEGUN, NBS AMS SERIES 55, U.S. DEPT. OF */
    /*                 COMMERCE, 1955. */

    /*               COMPUTATION OF BESSEL FUNCTIONS OF COMPLEX ARGUMENT */
    /*                 AND LARGE ORDER BY D. E. AMOS, SAND83-0643, MAY, 1983 */

    /*               A SUBROUTINE PACKAGE FOR BESSEL FUNCTIONS OF A COMPLEX */
    /*                 ARGUMENT AND NONNEGATIVE ORDER BY D. E. AMOS, SAND85- */
    /*                 1018, MAY, 1985 */

    /*               A PORTABLE PACKAGE FOR BESSEL FUNCTIONS OF A COMPLEX */
    /*                 ARGUMENT AND NONNEGATIVE ORDER BY D. E. AMOS, ACM */
    /*                 TRANS. MATH. SOFTWARE, VOL. 12, NO. 3, SEPTEMBER 1986, */
    /*                 PP 265-273. */

    /* ***ROUTINES CALLED  ZACAI,ZBKNU,ZEXP,ZSQRT,ZABS,I1MACH,D1MACH */
    /* ***END PROLOGUE  ZAIRY */
    /*     COMPLEX AI,CONE,CSQ,CY,S1,S2,TRM1,TRM2,Z,ZTA,Z3 */
    /* ***FIRST EXECUTABLE STATEMENT  ZAIRY */
    *ierr = 0;
    *nz   = 0;
    if(*id < 0 || *id > 1)
    {
        *ierr = 1;
    }
    if(*kode < 1 || *kode > 2)
    {
        *ierr = 1;
    }
    if(*ierr != 0)
    {
        return 0;
    }
    az = zabs_(zr, zi);
    /* Computing MAX */
    d__1 = d1mach_(&c__4);
    tol  = max(d__1, 1e-18);
    fid  = (double)((float)(*id));
    if(az > 1.)
    {
        goto L70;
    }
    /* ----------------------------------------------------------------------- */
    /*     POWER SERIES FOR CABS(Z).LE.1. */
    /* ----------------------------------------------------------------------- */
    s1r = coner;
    s1i = conei;
    s2r = coner;
    s2i = conei;
    if(az < tol)
    {
        goto L170;
    }
    aa = az * az;
    if(aa < tol / az)
    {
        goto L40;
    }
    trm1r = coner;
    trm1i = conei;
    trm2r = coner;
    trm2i = conei;
    atrm  = 1.;
    str   = *zr * *zr - *zi * *zi;
    sti   = *zr * *zi + *zi * *zr;
    z3r   = str * *zr - sti * *zi;
    z3i   = str * *zi + sti * *zr;
    az3   = az * aa;
    ak    = fid + 2.;
    bk    = 3. - fid - fid;
    ck    = 4. - fid;
    dk    = fid + 3. + fid;
    d1    = ak * dk;
    d2    = bk * ck;
    ad    = std::min(d1, d2);
    ak    = fid * 9. + 24.;
    bk    = 30. - fid * 9.;
    for(k = 1; k <= 25; ++k)
    {
        str   = (trm1r * z3r - trm1i * z3i) / d1;
        trm1i = (trm1r * z3i + trm1i * z3r) / d1;
        trm1r = str;
        s1r += trm1r;
        s1i += trm1i;
        str   = (trm2r * z3r - trm2i * z3i) / d2;
        trm2i = (trm2r * z3i + trm2i * z3r) / d2;
        trm2r = str;
        s2r += trm2r;
        s2i += trm2i;
        atrm = atrm * az3 / ad;
        d1 += ak;
        d2 += bk;
        ad = std::min(d1, d2);
        if(atrm < tol * ad)
        {
            goto L40;
        }
        ak += 18.;
        bk += 18.;
        /* L30: */
    }
L40:
    if(*id == 1)
    {
        goto L50;
    }
    *air = s1r * c1 - c2 * (*zr * s2r - *zi * s2i);
    *aii = s1i * c1 - c2 * (*zr * s2i + *zi * s2r);
    if(*kode == 1)
    {
        return 0;
    }
    zsqrt_(zr, zi, &str, &sti);
    ztar = tth * (*zr * str - *zi * sti);
    ztai = tth * (*zr * sti + *zi * str);
    zexp_(&ztar, &ztai, &str, &sti);
    ptr  = *air * str - *aii * sti;
    *aii = *air * sti + *aii * str;
    *air = ptr;
    return 0;
L50:
    *air = -s2r * c2;
    *aii = -s2i * c2;
    if(az <= tol)
    {
        goto L60;
    }
    str = *zr * s1r - *zi * s1i;
    sti = *zr * s1i + *zi * s1r;
    cc  = c1 / (fid + 1.);
    *air += cc * (str * *zr - sti * *zi);
    *aii += cc * (str * *zi + sti * *zr);
L60:
    if(*kode == 1)
    {
        return 0;
    }
    zsqrt_(zr, zi, &str, &sti);
    ztar = tth * (*zr * str - *zi * sti);
    ztai = tth * (*zr * sti + *zi * str);
    zexp_(&ztar, &ztai, &str, &sti);
    ptr  = str * *air - sti * *aii;
    *aii = str * *aii + sti * *air;
    *air = ptr;
    return 0;
/* ----------------------------------------------------------------------- */
/*     CASE FOR CABS(Z).GT.1.0 */
/* ----------------------------------------------------------------------- */
L70:
    fnu = (fid + 1.) / 3.;
    /* ----------------------------------------------------------------------- */
    /*     SET PARAMETERS RELATED TO MACHINE CONSTANTS. */
    /*     TOL IS THE APPROXIMATE UNIT ROUNDOFF LIMITED TO 1.0D-18. */
    /*     ELIM IS THE APPROXIMATE EXPONENTIAL OVER- AND UNDERFLOW LIMIT. */
    /*     EXP(-ELIM).LT.EXP(-ALIM)=EXP(-ELIM)/TOL    AND */
    /*     EXP(ELIM).GT.EXP(ALIM)=EXP(ELIM)*TOL       ARE INTERVALS NEAR */
    /*     UNDERFLOW AND OVERFLOW LIMITS WHERE SCALED ARITHMETIC IS DONE. */
    /*     RL IS THE LOWER BOUNDARY OF THE ASYMPTOTIC EXPANSION FOR LARGE Z. */
    /*     DIG = NUMBER OF BASE 10 DIGITS IN TOL = 10**(-DIG). */
    /* ----------------------------------------------------------------------- */
    k1   = i1mach_(&c__15);
    k2   = i1mach_(&c__16);
    r1m5 = d1mach_(&c__5);
    /* Computing MIN */
    i__1 = abs(k1), i__2 = abs(k2);
    k    = std::min(i__1, i__2);
    elim = ((double)((float)k) * r1m5 - 3.) * 2.303;
    k1   = i1mach_(&c__14) - 1;
    aa   = r1m5 * (double)((float)k1);
    dig  = std::min(aa, 18.);
    aa *= 2.303;
    /* Computing MAX */
    d__1 = -aa;
    alim = elim + max(d__1, -41.45);
    rl   = dig * 1.2 + 3.;
    alaz = log(az);
    /* -------------------------------------------------------------------------- */
    /*     TEST FOR PROPER RANGE */
    /* ----------------------------------------------------------------------- */
    aa = .5 / tol;
    bb = (double)((float)i1mach_(&c__9)) * .5;
    aa = std::min(aa, bb);
    aa = pow_dd(&aa, &tth);
    if(az > aa)
    {
        goto L260;
    }
    aa = sqrt(aa);
    if(az > aa)
    {
        *ierr = 3;
    }
    zsqrt_(zr, zi, &csqr, &csqi);
    ztar = tth * (*zr * csqr - *zi * csqi);
    ztai = tth * (*zr * csqi + *zi * csqr);
    /* ----------------------------------------------------------------------- */
    /*     RE(ZTA).LE.0 WHEN RE(Z).LT.0, ESPECIALLY WHEN IM(Z) IS SMALL */
    /* ----------------------------------------------------------------------- */
    iflag = 0;
    sfac  = 1.;
    ak    = ztai;
    if(*zr >= 0.)
    {
        goto L80;
    }
    bk   = ztar;
    ck   = -abs(bk);
    ztar = ck;
    ztai = ak;
L80:
    if(*zi != 0.)
    {
        goto L90;
    }
    if(*zr > 0.)
    {
        goto L90;
    }
    ztar = 0.;
    ztai = ak;
L90:
    aa = ztar;
    if(aa >= 0. && *zr > 0.)
    {
        goto L110;
    }
    if(*kode == 2)
    {
        goto L100;
    }
    /* ----------------------------------------------------------------------- */
    /*     OVERFLOW TEST */
    /* ----------------------------------------------------------------------- */
    if(aa > -alim)
    {
        goto L100;
    }
    aa    = -aa + alaz * .25;
    iflag = 1;
    sfac  = tol;
    if(aa > elim)
    {
        goto L270;
    }
L100:
    /* ----------------------------------------------------------------------- */
    /*     CBKNU AND CACON RETURN EXP(ZTA)*K(FNU,ZTA) ON KODE=2 */
    /* ----------------------------------------------------------------------- */
    mr = 1;
    if(*zi < 0.)
    {
        mr = -1;
    }
    zacai_(&ztar, &ztai, &fnu, kode, &mr, &c__1, cyr, cyi, &nn, &rl, &tol, &elim, &alim);
    if(nn < 0)
    {
        goto L280;
    }
    *nz += nn;
    goto L130;
L110:
    if(*kode == 2)
    {
        goto L120;
    }
    /* ----------------------------------------------------------------------- */
    /*     UNDERFLOW TEST */
    /* ----------------------------------------------------------------------- */
    if(aa < alim)
    {
        goto L120;
    }
    aa    = -aa - alaz * .25;
    iflag = 2;
    sfac  = 1. / tol;
    if(aa < -elim)
    {
        goto L210;
    }
L120:
    zbknu_(&ztar, &ztai, &fnu, kode, &c__1, cyr, cyi, nz, &tol, &elim, &alim);
L130:
    s1r = cyr[0] * coef;
    s1i = cyi[0] * coef;
    if(iflag != 0)
    {
        goto L150;
    }
    if(*id == 1)
    {
        goto L140;
    }
    *air = csqr * s1r - csqi * s1i;
    *aii = csqr * s1i + csqi * s1r;
    return 0;
L140:
    *air = -(*zr * s1r - *zi * s1i);
    *aii = -(*zr * s1i + *zi * s1r);
    return 0;
L150:
    s1r *= sfac;
    s1i *= sfac;
    if(*id == 1)
    {
        goto L160;
    }
    str  = s1r * csqr - s1i * csqi;
    s1i  = s1r * csqi + s1i * csqr;
    s1r  = str;
    *air = s1r / sfac;
    *aii = s1i / sfac;
    return 0;
L160:
    str  = -(s1r * *zr - s1i * *zi);
    s1i  = -(s1r * *zi + s1i * *zr);
    s1r  = str;
    *air = s1r / sfac;
    *aii = s1i / sfac;
    return 0;
L170:
    aa  = d1mach_(&c__1) * 1e3;
    s1r = zeror;
    s1i = zeroi;
    if(*id == 1)
    {
        goto L190;
    }
    if(az <= aa)
    {
        goto L180;
    }
    s1r = c2 * *zr;
    s1i = c2 * *zi;
L180:
    *air = c1 - s1r;
    *aii = -s1i;
    return 0;
L190:
    *air = -c2;
    *aii = 0.;
    aa   = sqrt(aa);
    if(az <= aa)
    {
        goto L200;
    }
    s1r = (*zr * *zr - *zi * *zi) * .5;
    s1i = *zr * *zi;
L200:
    *air += c1 * s1r;
    *aii += c1 * s1i;
    return 0;
L210:
    *nz  = 1;
    *air = zeror;
    *aii = zeroi;
    return 0;
L270:
    *nz   = 0;
    *ierr = 2;
    return 0;
L280:
    if(nn == -1)
    {
        goto L270;
    }
    *nz   = 0;
    *ierr = 5;
    return 0;
L260:
    *ierr = 4;
    *nz   = 0;
    return 0;
} /* zairy_ */

 int zasyi_(double* zr,
                            double* zi,
                            double* fnu,
                            int32*    kode,
                            int32*    n,
                            double* yr,
                            double* yi,
                            int32*    nz,
                            double* rl,
                            double* tol,
                            double* elim,
                            double* alim)
{
    /* Initialized data */

    static double pi    = 3.14159265358979324;
    static double rtpi  = .159154943091895336;
    static double zeror = 0.;
    static double zeroi = 0.;
    static double coner = 1.;
    static double conei = 0.;

    /* System generated locals */
    int32    i__1, i__2;
    double d__1, d__2;


    
    int32                     i__, j, k, m;
    double                  s, aa, bb;
    int32                     ib;
    double                  ak, bk;
    int32                     il, jl;
    double                  az;
    int32                     nn;
    double                  p1i, s2i, p1r, s2r, cki, dki, fdn, arg, aez, arm, ckr, dkr, czi, ezi, sgn;
    int32                     inu;
    double                  raz, czr, ezr, sqk, sti, rzi, tzi, str, rzr, tzr, ak1i, ak1r, cs1i, cs2i, cs1r, cs2r, dnu2, rtr1, dfnu;
    
    double                  atol;
    
    int32                     koded;

    /* *********************************************************************72 */

    /* ***BEGIN PROLOGUE  ZASYI */
    /* ***REFER TO  ZBESI,ZBESK */

    /*     ZASYI COMPUTES THE I BESSEL FUNCTION FOR REAL(Z).GE.0.0 BY */
    /*     MEANS OF THE ASYMPTOTIC EXPANSION FOR LARGE CABS(Z) IN THE */
    /*     REGION CABS(Z).GT.MAX(RL,FNU*FNU/2). NZ=0 IS A NORMAL RETURN. */
    /*     NZ.LT.0 INDICATES AN OVERFLOW ON KODE=1. */

    /* ***ROUTINES CALLED  D1MACH,ZABS,ZDIV,ZEXP,ZMLT,ZSQRT */
    /* ***END PROLOGUE  ZASYI */
    /*     COMPLEX AK1,CK,CONE,CS1,CS2,CZ,CZERO,DK,EZ,P1,RZ,S2,Y,Z */
    /* Parameter adjustments */
    --yi;
    --yr;

    /* Function Body */

    *nz  = 0;
    az   = zabs_(zr, zi);
    arm  = d1mach_(&c__1) * 1e3;
    rtr1 = sqrt(arm);
    il   = std::min(2L, *n);
    dfnu = *fnu + (double)((float)(*n - il));
    /* ----------------------------------------------------------------------- */
    /*     OVERFLOW TEST */
    /* ----------------------------------------------------------------------- */
    raz  = 1. / az;
    str  = *zr * raz;
    sti  = -(*zi) * raz;
    ak1r = rtpi * str * raz;
    ak1i = rtpi * sti * raz;
    zsqrt_(&ak1r, &ak1i, &ak1r, &ak1i);
    czr = *zr;
    czi = *zi;
    if(*kode != 2)
    {
        goto L10;
    }
    czr = zeror;
    czi = *zi;
L10:
    if(abs(czr) > *elim)
    {
        goto L100;
    }
    dnu2  = dfnu + dfnu;
    koded = 1;
    if(abs(czr) > *alim && *n > 2)
    {
        goto L20;
    }
    koded = 0;
    zexp_(&czr, &czi, &str, &sti);
    zmlt_(&ak1r, &ak1i, &str, &sti, &ak1r, &ak1i);
L20:
    fdn = 0.;
    if(dnu2 > rtr1)
    {
        fdn = dnu2 * dnu2;
    }
    ezr = *zr * 8.;
    ezi = *zi * 8.;
    /* ----------------------------------------------------------------------- */
    /*     WHEN Z IS IMAGINARY, THE ERROR TEST MUST BE MADE RELATIVE TO THE */
    /*     FIRST RECIPROCAL POWER SINCE THIS IS THE LEADING TERM OF THE */
    /*     EXPANSION FOR THE IMAGINARY PART. */
    /* ----------------------------------------------------------------------- */
    aez = az * 8.;
    s   = *tol / aez;
    jl  = (int32)((float)(*rl + *rl)) + 2;
    p1r = zeror;
    p1i = zeroi;
    if(*zi == 0.)
    {
        goto L30;
    }
    /* ----------------------------------------------------------------------- */
    /*     CALCULATE EXP(PI*(0.5+FNU+N-IL)*I) TO MINIMIZE LOSSES OF */
    /*     SIGNIFICANCE WHEN FNU OR N IS LARGE */
    /* ----------------------------------------------------------------------- */
    inu = (int32)((float)(*fnu));
    arg = (*fnu - (double)((float)inu)) * pi;
    inu = inu + *n - il;
    ak  = -sin(arg);
    bk  = cos(arg);
    if(*zi < 0.)
    {
        bk = -bk;
    }
    p1r = ak;
    p1i = bk;
    if(inu % 2 == 0)
    {
        goto L30;
    }
    p1r = -p1r;
    p1i = -p1i;
L30:
    i__1 = il;
    for(k = 1; k <= i__1; ++k)
    {
        sqk  = fdn - 1.;
        atol = s * abs(sqk);
        sgn  = 1.;
        cs1r = coner;
        cs1i = conei;
        cs2r = coner;
        cs2i = conei;
        ckr  = coner;
        cki  = conei;
        ak   = 0.;
        aa   = 1.;
        bb   = aez;
        dkr  = ezr;
        dki  = ezi;
        i__2 = jl;
        for(j = 1; j <= i__2; ++j)
        {
            zdiv_(&ckr, &cki, &dkr, &dki, &str, &sti);
            ckr = str * sqk;
            cki = sti * sqk;
            cs2r += ckr;
            cs2i += cki;
            sgn = -sgn;
            cs1r += ckr * sgn;
            cs1i += cki * sgn;
            dkr += ezr;
            dki += ezi;
            aa = aa * abs(sqk) / bb;
            bb += aez;
            ak += 8.;
            sqk -= ak;
            if(aa <= atol)
            {
                goto L50;
            }
            /* L40: */
        }
        goto L110;
    L50:
        s2r = cs1r;
        s2i = cs1i;
        if(*zr + *zr >= *elim)
        {
            goto L60;
        }
        tzr  = *zr + *zr;
        tzi  = *zi + *zi;
        d__1 = -tzr;
        d__2 = -tzi;
        zexp_(&d__1, &d__2, &str, &sti);
        zmlt_(&str, &sti, &p1r, &p1i, &str, &sti);
        zmlt_(&str, &sti, &cs2r, &cs2i, &str, &sti);
        s2r += str;
        s2i += sti;
    L60:
        fdn   = fdn + dfnu * 8. + 4.;
        p1r   = -p1r;
        p1i   = -p1i;
        m     = *n - il + k;
        yr[m] = s2r * ak1r - s2i * ak1i;
        yi[m] = s2r * ak1i + s2i * ak1r;
        /* L70: */
    }
    if(*n <= 2)
    {
        return 0;
    }
    nn   = *n;
    k    = nn - 2;
    ak   = (double)((float)k);
    str  = *zr * raz;
    sti  = -(*zi) * raz;
    rzr  = (str + str) * raz;
    rzi  = (sti + sti) * raz;
    ib   = 3;
    i__1 = nn;
    for(i__ = ib; i__ <= i__1; ++i__)
    {
        yr[k] = (ak + *fnu) * (rzr * yr[k + 1] - rzi * yi[k + 1]) + yr[k + 2];
        yi[k] = (ak + *fnu) * (rzr * yi[k + 1] + rzi * yr[k + 1]) + yi[k + 2];
        ak += -1.;
        --k;
        /* L80: */
    }
    if(koded == 0)
    {
        return 0;
    }
    zexp_(&czr, &czi, &ckr, &cki);
    i__1 = nn;
    for(i__ = 1; i__ <= i__1; ++i__)
    {
        str     = yr[i__] * ckr - yi[i__] * cki;
        yi[i__] = yr[i__] * cki + yi[i__] * ckr;
        yr[i__] = str;
        /* L90: */
    }
    return 0;
L100:
    *nz = -1;
    return 0;
L110:
    *nz = -2;
    return 0;
} /* zasyi_ */

 int zbesh_(double* zr,
                            double* zi,
                            double* fnu,
                            int32*    kode,
                            int32*    m,
                            int32*    n,
                            double* cyr,
                            double* cyi,
                            int32*    nz,
                            int32*    ierr)
{
    /* Initialized data */

    static double hpi = 1.57079632679489662;

    /* System generated locals */
    int32    i__1, i__2;
    double d__1, d__2;


    
    int32                     i__, k, k1, k2;
    double                  aa, bb, fn;
    int32                     mm;
    double                  az;
    int32                     ir, nn;
    double                  rl;
    int32                     mr, nw;
    double                  dig, arg, aln, fmm, r1m5, ufl, sgn;
    int32                     nuf, inu;
    double                  tol, sti, zni, zti, str, znr, alim, elim;
    
    double                  atol, rhpi;
    int32                     inuh;
    double                  fnul, rtol, ascle, csgni;
   
    double                  csgnr;

    /* *********************************************************************72 */

    /* c ZBESH computes a sequence of System::Complex<float> Hankel functions. */

    /* ***BEGIN PROLOGUE  ZBESH */
    /* ***DATE WRITTEN   830501   (YYMMDD) */
    /* ***REVISION DATE  890801, 930101   (YYMMDD) */
    /* ***CATEGORY NO.  B5K */
    /* ***KEYWORDS  H-BESSEL FUNCTIONS,BESSEL FUNCTIONS OF COMPLEX ARGUMENT, */
    /*             BESSEL FUNCTIONS OF THIRD KIND,HANKEL FUNCTIONS */
    /* ***AUTHOR  AMOS, DONALD E., SANDIA NATIONAL LABORATORIES */
    /* ***PURPOSE  TO COMPUTE THE H-BESSEL FUNCTIONS OF A COMPLEX ARGUMENT */
    /* ***DESCRIPTION */

    /*                      ***A DOUBLE PRECISION ROUTINE*** */
    /*         ON KODE=1, ZBESH COMPUTES AN N MEMBER SEQUENCE OF COMPLEX */
    /*         HANKEL (BESSEL) FUNCTIONS CY(J)=H(M,FNU+J-1,Z) FOR KINDS M=1 */
    /*         OR 2, REAL, NONNEGATIVE ORDERS FNU+J-1, J=1,...,N, AND COMPLEX */
    /*         Z.NE.CMPLX(0.0,0.0) IN THE CUT PLANE -PI.LT.ARG(Z).LE.PI. */
    /*         ON KODE=2, ZBESH RETURNS THE SCALED HANKEL FUNCTIONS */

    /*         CY(I)=EXP(-MM*Z*I)*H(M,FNU+J-1,Z)       MM=3-2*M,   I**2=-1. */

    /*         WHICH REMOVES THE EXPONENTIAL BEHAVIOR IN BOTH THE UPPER AND */
    /*         LOWER HALF PLANES. DEFINITIONS AND NOTATION ARE FOUND IN THE */
    /*         NBS HANDBOOK OF MATHEMATICAL FUNCTIONS (CONST. 1). */

    /*         INPUT      ZR,ZI,FNU ARE DOUBLE PRECISION */
    /*           ZR,ZI  - Z=CMPLX(ZR,ZI), Z.NE.CMPLX(0.0D0,0.0D0), */
    /*                    -PT.LT.ARG(Z).LE.PI */
    /*           FNU    - ORDER OF INITIAL H FUNCTION, FNU.GE.0.0D0 */
    /*           KODE   - A PARAMETER TO INDICATE THE SCALING OPTION */
    /*                    KODE= 1  RETURNS */
    /*                             CY(J)=H(M,FNU+J-1,Z),   J=1,...,N */
    /*                        = 2  RETURNS */
    /*                             CY(J)=H(M,FNU+J-1,Z)*EXP(-I*Z*(3-2M)) */
    /*                                  J=1,...,N  ,  I**2=-1 */
    /*           M      - KIND OF HANKEL FUNCTION, M=1 OR 2 */
    /*           N      - NUMBER OF MEMBERS IN THE SEQUENCE, N.GE.1 */

    /*         OUTPUT     CYR,CYI ARE DOUBLE PRECISION */
    /*           CYR,CYI- DOUBLE PRECISION VECTORS WHOSE FIRST N COMPONENTS */
    /*                    CONTAIN REAL AND IMAGINARY PARTS FOR THE SEQUENCE */
    /*                    CY(J)=H(M,FNU+J-1,Z)  OR */
    /*                    CY(J)=H(M,FNU+J-1,Z)*EXP(-I*Z*(3-2M))  J=1,...,N */
    /*                    DEPENDING ON KODE, I**2=-1. */
    /*           NZ     - NUMBER OF COMPONENTS SET TO ZERO DUE TO UNDERFLOW, */
    /*                    NZ= 0   , NORMAL RETURN */
    /*                    NZ.GT.0 , FIRST NZ COMPONENTS OF CY SET TO ZERO DUE */
    /*                              TO UNDERFLOW, CY(J)=CMPLX(0.0D0,0.0D0) */
    /*                              J=1,...,NZ WHEN Y.GT.0.0 AND M=1 OR */
    /*                              Y.LT.0.0 AND M=2. FOR THE COMPLMENTARY */
    /*                              HALF PLANES, NZ STATES ONLY THE NUMBER */
    /*                              OF UNDERFLOWS. */
    /*           IERR   - ERROR FLAG */
    /*                    IERR=0, NORMAL RETURN - COMPUTATION COMPLETED */
    /*                    IERR=1, INPUT ERROR   - NO COMPUTATION */
    /*                    IERR=2, OVERFLOW      - NO COMPUTATION, FNU TOO */
    /*                            LARGE OR CABS(Z) TOO SMALL OR BOTH */
    /*                    IERR=3, CABS(Z) OR FNU+N-1 LARGE - COMPUTATION DONE */
    /*                            BUT LOSSES OF SIGNIFCANCE BY ARGUMENT */
    /*                            REDUCTION PRODUCE LESS THAN HALF OF MACHINE */
    /*                            ACCURACY */
    /*                    IERR=4, CABS(Z) OR FNU+N-1 TOO LARGE - NO COMPUTA- */
    /*                            TION BECAUSE OF COMPLETE LOSSES OF SIGNIFI- */
    /*                            CANCE BY ARGUMENT REDUCTION */
    /*                    IERR=5, ERROR              - NO COMPUTATION, */
    /*                            ALGORITHM TERMINATION CONDITION NOT MET */

    /* ***LONG DESCRIPTION */

    /*         THE COMPUTATION IS CARRIED OUT BY THE RELATION */

    /*         H(M,FNU,Z)=(1/MP)*EXP(-MP*FNU)*K(FNU,Z*EXP(-MP)) */
    /*             MP=MM*HPI*I,  MM=3-2*M,  HPI=PI/2,  I**2=-1 */

    /*         FOR M=1 OR 2 WHERE THE K BESSEL FUNCTION IS COMPUTED FOR THE */
    /*         RIGHT HALF PLANE RE(Z).GE.0.0. THE K FUNCTION IS CONTINUED */
    /*         TO THE LEFT HALF PLANE BY THE RELATION */

    /*         K(FNU,Z*EXP(MP)) = EXP(-MP*FNU)*K(FNU,Z)-MP*I(FNU,Z) */
    /*         MP=MR*PI*I, MR=+1 OR -1, RE(Z).GT.0, I**2=-1 */

    /*         WHERE I(FNU,Z) IS THE I BESSEL FUNCTION. */

    /*         EXPONENTIAL DECAY OF H(M,FNU,Z) OCCURS IN THE UPPER HALF Z */
    /*         PLANE FOR M=1 AND THE LOWER HALF Z PLANE FOR M=2.  EXPONENTIAL */
    /*         GROWTH OCCURS IN THE COMPLEMENTARY HALF PLANES.  SCALING */
    /*         BY EXP(-MM*Z*I) REMOVES THE EXPONENTIAL BEHAVIOR IN THE */
    /*         WHOLE Z PLANE FOR Z TO INFINITY. */

    /*         FOR NEGATIVE ORDERS,THE FORMULAE */

    /*               H(1,-FNU,Z) = H(1,FNU,Z)*CEXP( PI*FNU*I) */
    /*               H(2,-FNU,Z) = H(2,FNU,Z)*CEXP(-PI*FNU*I) */
    /*                         I**2=-1 */

    /*         CAN BE USED. */

    /*         IN MOST COMPLEX VARIABLE COMPUTATION, ONE MUST EVALUATE ELE- */
    /*         MENTARY FUNCTIONS. WHEN THE MAGNITUDE OF Z OR FNU+N-1 IS */
    /*         LARGE, LOSSES OF SIGNIFICANCE BY ARGUMENT REDUCTION OCCUR. */
    /*         CONSEQUENTLY, IF EITHER ONE EXCEEDS U1=SQRT(0.5/UR), THEN */
    /*         LOSSES EXCEEDING HALF PRECISION ARE LIKELY AND AN ERROR FLAG */
    /*         IERR=3 IS TRIGGERED WHERE UR=DMAX1(D1MACH(4),1.0D-18) IS */
    /*         DOUBLE PRECISION UNIT ROUNDOFF LIMITED TO 18 DIGITS PRECISION. */
    /*         IF EITHER IS LARGER THAN U2=0.5/UR, THEN ALL SIGNIFICANCE IS */
    /*         LOST AND IERR=4. IN ORDER TO USE THE INT FUNCTION, ARGUMENTS */
    /*         MUST BE FURTHER RESTRICTED NOT TO EXCEED THE LARGEST MACHINE */
    /*         INTEGER, U3=I1MACH(9). THUS, THE MAGNITUDE OF Z AND FNU+N-1 IS */
    /*         RESTRICTED BY MIN(U2,U3). ON 32 BIT MACHINES, U1,U2, AND U3 */
    /*         ARE APPROXIMATELY 2.0E+3, 4.2E+6, 2.1E+9 IN SINGLE PRECISION */
    /*         ARITHMETIC AND 1.3E+8, 1.8E+16, 2.1E+9 IN DOUBLE PRECISION */
    /*         ARITHMETIC RESPECTIVELY. THIS MAKES U2 AND U3 LIMITING IN */
    /*         THEIR RESPECTIVE ARITHMETICS. THIS MEANS THAT ONE CAN EXPECT */
    /*         TO RETAIN, IN THE WORST CASES ON 32 BIT MACHINES, NO DIGITS */
    /*         IN SINGLE AND ONLY 7 DIGITS IN DOUBLE PRECISION ARITHMETIC. */
    /*         SIMILAR CONSIDERATIONS HOLD FOR OTHER MACHINES. */

    /*         THE APPROXIMATE RELATIVE ERROR IN THE MAGNITUDE OF A COMPLEX */
    /*         BESSEL FUNCTION CAN BE EXPRESSED BY P*10**S WHERE P=MAX(UNIT */
    /*         ROUNDOFF,1.0D-18) IS THE NOMINAL PRECISION AND 10**S REPRE- */
    /*         SENTS THE INCREASE IN ERROR DUE TO ARGUMENT REDUCTION IN THE */
    /*         ELEMENTARY FUNCTIONS. HERE, S=MAX(1,ABS(LOG10(CABS(Z))), */
    /*         ABS(LOG10(FNU))) APPROXIMATELY (I.E. S=MAX(1,ABS(EXPONENT OF */
    /*         CABS(Z),ABS(EXPONENT OF FNU)) ). HOWEVER, THE PHASE ANGLE MAY */
    /*         HAVE ONLY ABSOLUTE ACCURACY. THIS IS MOST LIKELY TO OCCUR WHEN */
    /*         ONE COMPONENT (IN ABSOLUTE VALUE) IS LARGER THAN THE OTHER BY */
    /*         SEVERAL ORDERS OF MAGNITUDE. IF ONE COMPONENT IS 10**K LARGER */
    /*         THAN THE OTHER, THEN ONE CAN EXPECT ONLY MAX(ABS(LOG10(P))-K, */
    /*         0) SIGNIFICANT DIGITS; OR, STATED ANOTHER WAY, WHEN K EXCEEDS */
    /*         THE EXPONENT OF P, NO SIGNIFICANT DIGITS REMAIN IN THE SMALLER */
    /*         COMPONENT. HOWEVER, THE PHASE ANGLE RETAINS ABSOLUTE ACCURACY */
    /*         BECAUSE, IN COMPLEX ARITHMETIC WITH PRECISION P, THE SMALLER */
    /*         COMPONENT WILL NOT (AS A RULE) DECREASE BELOW P TIMES THE */
    /*         MAGNITUDE OF THE LARGER COMPONENT. IN THESE EXTREME CASES, */
    /*         THE PRINCIPAL PHASE ANGLE IS ON THE ORDER OF +P, -P, PI/2-P, */
    /*         OR -PI/2+P. */

    /* ***REFERENCES  HANDBOOK OF MATHEMATICAL FUNCTIONS BY M. ABRAMOWITZ */
    /*                 AND I. A. STEGUN, NBS AMS SERIES 55, U.S. DEPT. OF */
    /*                 COMMERCE, 1955. */

    /*               COMPUTATION OF BESSEL FUNCTIONS OF COMPLEX ARGUMENT */
    /*                 BY D. E. AMOS, SAND83-0083, MAY, 1983. */

    /*               COMPUTATION OF BESSEL FUNCTIONS OF COMPLEX ARGUMENT */
    /*                 AND LARGE ORDER BY D. E. AMOS, SAND83-0643, MAY, 1983 */

    /*               A SUBROUTINE PACKAGE FOR BESSEL FUNCTIONS OF A COMPLEX */
    /*                 ARGUMENT AND NONNEGATIVE ORDER BY D. E. AMOS, SAND85- */
    /*                 1018, MAY, 1985 */

    /*               A PORTABLE PACKAGE FOR BESSEL FUNCTIONS OF A COMPLEX */
    /*                 ARGUMENT AND NONNEGATIVE ORDER BY D. E. AMOS, ACM */
    /*                 TRANS. MATH. SOFTWARE, VOL. 12, NO. 3, SEPTEMBER 1986, */
    /*                 PP 265-273. */

    /* ***ROUTINES CALLED  ZACON,ZBKNU,ZBUNK,ZUOIK,ZABS,I1MACH,D1MACH */
    /* ***END PROLOGUE  ZBESH */

    /*     COMPLEX CY,Z,ZN,ZT,CSGN */

    /* Parameter adjustments */
    --cyi;
    --cyr;

    /* Function Body */

    /* ***FIRST EXECUTABLE STATEMENT  ZBESH */
    *ierr = 0;
    *nz   = 0;
    if(*zr == 0. && *zi == 0.)
    {
        *ierr = 1;
    }
    if(*fnu < 0.)
    {
        *ierr = 1;
    }
    if(*m < 1 || *m > 2)
    {
        *ierr = 1;
    }
    if(*kode < 1 || *kode > 2)
    {
        *ierr = 1;
    }
    if(*n < 1)
    {
        *ierr = 1;
    }
    if(*ierr != 0)
    {
        return 0;
    }
    nn = *n;
    /* ----------------------------------------------------------------------- */
    /*     SET PARAMETERS RELATED TO MACHINE CONSTANTS. */
    /*     TOL IS THE APPROXIMATE UNIT ROUNDOFF LIMITED TO 1.0E-18. */
    /*     ELIM IS THE APPROXIMATE EXPONENTIAL OVER- AND UNDERFLOW LIMIT. */
    /*     EXP(-ELIM).LT.EXP(-ALIM)=EXP(-ELIM)/TOL    AND */
    /*     EXP(ELIM).GT.EXP(ALIM)=EXP(ELIM)*TOL       ARE INTERVALS NEAR */
    /*     UNDERFLOW AND OVERFLOW LIMITS WHERE SCALED ARITHMETIC IS DONE. */
    /*     RL IS THE LOWER BOUNDARY OF THE ASYMPTOTIC EXPANSION FOR LARGE Z. */
    /*     DIG = NUMBER OF BASE 10 DIGITS IN TOL = 10**(-DIG). */
    /*     FNUL IS THE LOWER BOUNDARY OF THE ASYMPTOTIC SERIES FOR LARGE FNU */
    /* ----------------------------------------------------------------------- */
    /* Computing MAX */
    d__1 = d1mach_(&c__4);
    tol  = max(d__1, 1e-18);
    k1   = i1mach_(&c__15);
    k2   = i1mach_(&c__16);
    r1m5 = d1mach_(&c__5);
    /* Computing MIN */
    i__1 = abs(k1), i__2 = abs(k2);
    k    = std::min(i__1, i__2);
    elim = ((double)((float)k) * r1m5 - 3.) * 2.303;
    k1   = i1mach_(&c__14) - 1;
    aa   = r1m5 * (double)((float)k1);
    dig  = std::min(aa, 18.);
    aa *= 2.303;
    /* Computing MAX */
    d__1 = -aa;
    alim = elim + max(d__1, -41.45);
    fnul = (dig - 3.) * 6. + 10.;
    rl   = dig * 1.2 + 3.;
    fn   = *fnu + (double)((float)(nn - 1));
    mm   = 3 - *m - *m;
    fmm  = (double)((float)mm);
    znr  = fmm * *zi;
    zni  = -fmm * *zr;
    /* ----------------------------------------------------------------------- */
    /*     TEST FOR PROPER RANGE */
    /* ----------------------------------------------------------------------- */
    az = zabs_(zr, zi);
    aa = .5 / tol;
    bb = (double)((float)i1mach_(&c__9)) * .5;
    aa = std::min(aa, bb);
    if(az > aa)
    {
        goto L260;
    }
    if(fn > aa)
    {
        goto L260;
    }
    aa = sqrt(aa);
    if(az > aa)
    {
        *ierr = 3;
    }
    if(fn > aa)
    {
        *ierr = 3;
    }
    /* ----------------------------------------------------------------------- */
    /*     OVERFLOW TEST ON THE LAST MEMBER OF THE SEQUENCE */
    /* ----------------------------------------------------------------------- */
    ufl = d1mach_(&c__1) * 1e3;
    if(az < ufl)
    {
        goto L230;
    }
    if(*fnu > fnul)
    {
        goto L90;
    }
    if(fn <= 1.)
    {
        goto L70;
    }
    if(fn > 2.)
    {
        goto L60;
    }
    if(az > tol)
    {
        goto L70;
    }
    arg = az * .5;
    aln = -fn * log(arg);
    if(aln > elim)
    {
        goto L230;
    }
    goto L70;
L60:
    zuoik_(&znr, &zni, fnu, kode, &c__2, &nn, &cyr[1], &cyi[1], &nuf, &tol, &elim, &alim);
    if(nuf < 0)
    {
        goto L230;
    }
    *nz += nuf;
    nn -= nuf;
    /* ----------------------------------------------------------------------- */
    /*     HERE NN=N OR NN=0 SINCE NUF=0,NN, OR -1 ON RETURN FROM CUOIK */
    /*     IF NUF=NN, THEN CY(I)=CZERO FOR ALL I */
    /* ----------------------------------------------------------------------- */
    if(nn == 0)
    {
        goto L140;
    }
L70:
    if(znr < 0. || znr == 0. && zni < 0. && *m == 2)
    {
        goto L80;
    }
    /* ----------------------------------------------------------------------- */
    /*     RIGHT HALF PLANE COMPUTATION, XN.GE.0. .AND. (XN.NE.0. .OR. */
    /*     YN.GE.0. .OR. M=1) */
    /* ----------------------------------------------------------------------- */
    zbknu_(&znr, &zni, fnu, kode, &nn, &cyr[1], &cyi[1], nz, &tol, &elim, &alim);
    goto L110;
/* ----------------------------------------------------------------------- */
/*     LEFT HALF PLANE COMPUTATION */
/* ----------------------------------------------------------------------- */
L80:
    mr = -mm;
    zacon_(&znr, &zni, fnu, kode, &mr, &nn, &cyr[1], &cyi[1], &nw, &rl, &fnul, &tol, &elim, &alim);
    if(nw < 0)
    {
        goto L240;
    }
    *nz = nw;
    goto L110;
L90:
    /* ----------------------------------------------------------------------- */
    /*     UNIFORM ASYMPTOTIC EXPANSIONS FOR FNU.GT.FNUL */
    /* ----------------------------------------------------------------------- */
    mr = 0;
    if(znr >= 0. && (znr != 0. || zni >= 0. || *m != 2))
    {
        goto L100;
    }
    mr = -mm;
    if(znr != 0. || zni >= 0.)
    {
        goto L100;
    }
    znr = -znr;
    zni = -zni;
L100:
    zbunk_(&znr, &zni, fnu, kode, &mr, &nn, &cyr[1], &cyi[1], &nw, &tol, &elim, &alim);
    if(nw < 0)
    {
        goto L240;
    }
    *nz += nw;
L110:
    /* ----------------------------------------------------------------------- */
    /*     H(M,FNU,Z) = -FMM*(I/HPI)*(ZT**FNU)*K(FNU,-Z*ZT) */

    /*     ZT=EXP(-FMM*HPI*I) = CMPLX(0.0,-FMM), FMM=3-2*M, M=1,2 */
    /* ----------------------------------------------------------------------- */
    d__1 = -fmm;
    sgn  = d_sign(&hpi, &d__1);
    /* ----------------------------------------------------------------------- */
    /*     CALCULATE EXP(FNU*HPI*I) TO MINIMIZE LOSSES OF SIGNIFICANCE */
    /*     WHEN FNU IS LARGE */
    /* ----------------------------------------------------------------------- */
    inu  = (int32)((float)(*fnu));
    inuh = inu / 2;
    ir   = inu - (inuh << 1);
    arg  = (*fnu - (double)((float)(inu - ir))) * sgn;
    rhpi = 1. / sgn;
    /*     ZNI = RHPI*DCOS(ARG) */
    /*     ZNR = -RHPI*DSIN(ARG) */
    csgni = rhpi * cos(arg);
    csgnr = -rhpi * sin(arg);
    if(inuh % 2 == 0)
    {
        goto L120;
    }
    /*     ZNR = -ZNR */
    /*     ZNI = -ZNI */
    csgnr = -csgnr;
    csgni = -csgni;
L120:
    zti   = -fmm;
    rtol  = 1. / tol;
    ascle = ufl * rtol;
    i__1  = nn;
    for(i__ = 1; i__ <= i__1; ++i__)
    {
        /*       STR = CYR(I)*ZNR - CYI(I)*ZNI */
        /*       CYI(I) = CYR(I)*ZNI + CYI(I)*ZNR */
        /*       CYR(I) = STR */
        /*       STR = -ZNI*ZTI */
        /*       ZNI = ZNR*ZTI */
        /*       ZNR = STR */
        aa   = cyr[i__];
        bb   = cyi[i__];
        atol = 1.;
        /* Computing MAX */
        d__1 = abs(aa), d__2 = abs(bb);
        if(max(d__1, d__2) > ascle)
        {
            goto L135;
        }
        aa *= rtol;
        bb *= rtol;
        atol = tol;
    L135:
        str      = aa * csgnr - bb * csgni;
        sti      = aa * csgni + bb * csgnr;
        cyr[i__] = str * atol;
        cyi[i__] = sti * atol;
        str      = -csgni * zti;
        csgni    = csgnr * zti;
        csgnr    = str;
        /* L130: */
    }
    return 0;
L140:
    if(znr < 0.)
    {
        goto L230;
    }
    return 0;
L230:
    *nz   = 0;
    *ierr = 2;
    return 0;
L240:
    if(nw == -1)
    {
        goto L230;
    }
    *nz   = 0;
    *ierr = 5;
    return 0;
L260:
    *nz   = 0;
    *ierr = 4;
    return 0;
} /* zbesh_ */

 int zbesi_(double* zr, double* zi, double* fnu, int32* kode, int32* n, double* cyr, double* cyi, int32* nz, int32* ierr)
{
    /* Initialized data */

    static double pi    = 3.14159265358979324;
    static double coner = 1.;
    static double conei = 0.;

    /* System generated locals */
    int32    i__1, i__2;
    double d__1, d__2;

    

    
    int32                     i__, k, k1, k2;
    double                  aa, bb, fn, az;
    int32                     nn;
    double                  rl, dig, arg, r1m5;
    int32                     inu;
    double                  tol, sti, zni, str, znr, alim, elim;
    
    double                  atol, fnul, rtol, ascle, csgni, csgnr;
    
    

    /* *********************************************************************72 */

    /* c ZBESI computes a sequence of System::Complex<float> Bessel I functions. */

    /* ***BEGIN PROLOGUE  ZBESI */
    /* ***DATE WRITTEN   830501   (YYMMDD) */
    /* ***REVISION DATE  890801, 930101   (YYMMDD) */
    /* ***CATEGORY NO.  B5K */
    /* ***KEYWORDS  I-BESSEL FUNCTION,COMPLEX BESSEL FUNCTION, */
    /*             MODIFIED BESSEL FUNCTION OF THE FIRST KIND */
    /* ***AUTHOR  AMOS, DONALD E., SANDIA NATIONAL LABORATORIES */
    /* ***PURPOSE  TO COMPUTE I-BESSEL FUNCTIONS OF COMPLEX ARGUMENT */
    /* ***DESCRIPTION */

    /*                    ***A DOUBLE PRECISION ROUTINE*** */
    /*         ON KODE=1, ZBESI COMPUTES AN N MEMBER SEQUENCE OF COMPLEX */
    /*         BESSEL FUNCTIONS CY(J)=I(FNU+J-1,Z) FOR REAL, NONNEGATIVE */
    /*         ORDERS FNU+J-1, J=1,...,N AND COMPLEX Z IN THE CUT PLANE */
    /*         -PI.LT.ARG(Z).LE.PI. ON KODE=2, ZBESI RETURNS THE SCALED */
    /*         FUNCTIONS */

    /*         CY(J)=EXP(-ABS(X))*I(FNU+J-1,Z)   J = 1,...,N , X=REAL(Z) */

    /*         WITH THE EXPONENTIAL GROWTH REMOVED IN BOTH THE LEFT AND */
    /*         RIGHT HALF PLANES FOR Z TO INFINITY. DEFINITIONS AND NOTATION */
    /*         ARE FOUND IN THE NBS HANDBOOK OF MATHEMATICAL FUNCTIONS */
    /*         (CONST. 1). */

    /*         INPUT      ZR,ZI,FNU ARE DOUBLE PRECISION */
    /*           ZR,ZI  - Z=CMPLX(ZR,ZI),  -PI.LT.ARG(Z).LE.PI */
    /*           FNU    - ORDER OF INITIAL I FUNCTION, FNU.GE.0.0D0 */
    /*           KODE   - A PARAMETER TO INDICATE THE SCALING OPTION */
    /*                    KODE= 1  RETURNS */
    /*                             CY(J)=I(FNU+J-1,Z), J=1,...,N */
    /*                        = 2  RETURNS */
    /*                             CY(J)=I(FNU+J-1,Z)*EXP(-ABS(X)), J=1,...,N */
    /*           N      - NUMBER OF MEMBERS OF THE SEQUENCE, N.GE.1 */

    /*         OUTPUT     CYR,CYI ARE DOUBLE PRECISION */
    /*           CYR,CYI- DOUBLE PRECISION VECTORS WHOSE FIRST N COMPONENTS */
    /*                    CONTAIN REAL AND IMAGINARY PARTS FOR THE SEQUENCE */
    /*                    CY(J)=I(FNU+J-1,Z)  OR */
    /*                    CY(J)=I(FNU+J-1,Z)*EXP(-ABS(X))  J=1,...,N */
    /*                    DEPENDING ON KODE, X=REAL(Z) */
    /*           NZ     - NUMBER OF COMPONENTS SET TO ZERO DUE TO UNDERFLOW, */
    /*                    NZ= 0   , NORMAL RETURN */
    /*                    NZ.GT.0 , LAST NZ COMPONENTS OF CY SET TO ZERO */
    /*                              TO UNDERFLOW, CY(J)=CMPLX(0.0D0,0.0D0) */
    /*                              J = N-NZ+1,...,N */
    /*           IERR   - ERROR FLAG */
    /*                    IERR=0, NORMAL RETURN - COMPUTATION COMPLETED */
    /*                    IERR=1, INPUT ERROR   - NO COMPUTATION */
    /*                    IERR=2, OVERFLOW      - NO COMPUTATION, REAL(Z) TOO */
    /*                            LARGE ON KODE=1 */
    /*                    IERR=3, CABS(Z) OR FNU+N-1 LARGE - COMPUTATION DONE */
    /*                            BUT LOSSES OF SIGNIFCANCE BY ARGUMENT */
    /*                            REDUCTION PRODUCE LESS THAN HALF OF MACHINE */
    /*                            ACCURACY */
    /*                    IERR=4, CABS(Z) OR FNU+N-1 TOO LARGE - NO COMPUTA- */
    /*                            TION BECAUSE OF COMPLETE LOSSES OF SIGNIFI- */
    /*                            CANCE BY ARGUMENT REDUCTION */
    /*                    IERR=5, ERROR              - NO COMPUTATION, */
    /*                            ALGORITHM TERMINATION CONDITION NOT MET */

    /* ***LONG DESCRIPTION */

    /*         THE COMPUTATION IS CARRIED OUT BY THE POWER SERIES FOR */
    /*         SMALL CABS(Z), THE ASYMPTOTIC EXPANSION FOR LARGE CABS(Z), */
    /*         THE MILLER ALGORITHM NORMALIZED BY THE WRONSKIAN AND A */
    /*         NEUMANN SERIES FOR IMTERMEDIATE MAGNITUDES, AND THE */
    /*         UNIFORM ASYMPTOTIC EXPANSIONS FOR I(FNU,Z) AND J(FNU,Z) */
    /*         FOR LARGE ORDERS. BACKWARD RECURRENCE IS USED TO GENERATE */
    /*         SEQUENCES OR REDUCE ORDERS WHEN NECESSARY. */

    /*         THE CALCULATIONS ABOVE ARE DONE IN THE RIGHT HALF PLANE AND */
    /*         CONTINUED INTO THE LEFT HALF PLANE BY THE FORMULA */

    /*         I(FNU,Z*EXP(M*PI)) = EXP(M*PI*FNU)*I(FNU,Z)  REAL(Z).GT.0.0 */
    /*                       M = +I OR -I,  I**2=-1 */

    /*         FOR NEGATIVE ORDERS,THE FORMULA */

    /*              I(-FNU,Z) = I(FNU,Z) + (2/PI)*SIN(PI*FNU)*K(FNU,Z) */

    /*         CAN BE USED. HOWEVER,FOR LARGE ORDERS CLOSE TO INTEGERS, THE */
    /*         THE FUNCTION CHANGES RADICALLY. WHEN FNU IS A LARGE POSITIVE */
    /*         INTEGER,THE MAGNITUDE OF I(-FNU,Z)=I(FNU,Z) IS A LARGE */
    /*         NEGATIVE POWER OF TEN. BUT WHEN FNU IS NOT AN INTEGER, */
    /*         K(FNU,Z) DOMINATES IN MAGNITUDE WITH A LARGE POSITIVE POWER OF */
    /*         TEN AND THE MOST THAT THE SECOND TERM CAN BE REDUCED IS BY */
    /*         UNIT ROUNDOFF FROM THE COEFFICIENT. THUS, WIDE CHANGES CAN */
    /*         OCCUR WITHIN UNIT ROUNDOFF OF A LARGE INTEGER FOR FNU. HERE, */
    /*         LARGE MEANS FNU.GT.CABS(Z). */

    /*         IN MOST COMPLEX VARIABLE COMPUTATION, ONE MUST EVALUATE ELE- */
    /*         MENTARY FUNCTIONS. WHEN THE MAGNITUDE OF Z OR FNU+N-1 IS */
    /*         LARGE, LOSSES OF SIGNIFICANCE BY ARGUMENT REDUCTION OCCUR. */
    /*         CONSEQUENTLY, IF EITHER ONE EXCEEDS U1=SQRT(0.5/UR), THEN */
    /*         LOSSES EXCEEDING HALF PRECISION ARE LIKELY AND AN ERROR FLAG */
    /*         IERR=3 IS TRIGGERED WHERE UR=DMAX1(D1MACH(4),1.0D-18) IS */
    /*         DOUBLE PRECISION UNIT ROUNDOFF LIMITED TO 18 DIGITS PRECISION. */
    /*         IF EITHER IS LARGER THAN U2=0.5/UR, THEN ALL SIGNIFICANCE IS */
    /*         LOST AND IERR=4. IN ORDER TO USE THE INT FUNCTION, ARGUMENTS */
    /*         MUST BE FURTHER RESTRICTED NOT TO EXCEED THE LARGEST MACHINE */
    /*         INTEGER, U3=I1MACH(9). THUS, THE MAGNITUDE OF Z AND FNU+N-1 IS */
    /*         RESTRICTED BY MIN(U2,U3). ON 32 BIT MACHINES, U1,U2, AND U3 */
    /*         ARE APPROXIMATELY 2.0E+3, 4.2E+6, 2.1E+9 IN SINGLE PRECISION */
    /*         ARITHMETIC AND 1.3E+8, 1.8E+16, 2.1E+9 IN DOUBLE PRECISION */
    /*         ARITHMETIC RESPECTIVELY. THIS MAKES U2 AND U3 LIMITING IN */
    /*         THEIR RESPECTIVE ARITHMETICS. THIS MEANS THAT ONE CAN EXPECT */
    /*         TO RETAIN, IN THE WORST CASES ON 32 BIT MACHINES, NO DIGITS */
    /*         IN SINGLE AND ONLY 7 DIGITS IN DOUBLE PRECISION ARITHMETIC. */
    /*         SIMILAR CONSIDERATIONS HOLD FOR OTHER MACHINES. */

    /*         THE APPROXIMATE RELATIVE ERROR IN THE MAGNITUDE OF A COMPLEX */
    /*         BESSEL FUNCTION CAN BE EXPRESSED BY P*10**S WHERE P=MAX(UNIT */
    /*         ROUNDOFF,1.0E-18) IS THE NOMINAL PRECISION AND 10**S REPRE- */
    /*         SENTS THE INCREASE IN ERROR DUE TO ARGUMENT REDUCTION IN THE */
    /*         ELEMENTARY FUNCTIONS. HERE, S=MAX(1,ABS(LOG10(CABS(Z))), */
    /*         ABS(LOG10(FNU))) APPROXIMATELY (I.E. S=MAX(1,ABS(EXPONENT OF */
    /*         CABS(Z),ABS(EXPONENT OF FNU)) ). HOWEVER, THE PHASE ANGLE MAY */
    /*         HAVE ONLY ABSOLUTE ACCURACY. THIS IS MOST LIKELY TO OCCUR WHEN */
    /*         ONE COMPONENT (IN ABSOLUTE VALUE) IS LARGER THAN THE OTHER BY */
    /*         SEVERAL ORDERS OF MAGNITUDE. IF ONE COMPONENT IS 10**K LARGER */
    /*         THAN THE OTHER, THEN ONE CAN EXPECT ONLY MAX(ABS(LOG10(P))-K, */
    /*         0) SIGNIFICANT DIGITS; OR, STATED ANOTHER WAY, WHEN K EXCEEDS */
    /*         THE EXPONENT OF P, NO SIGNIFICANT DIGITS REMAIN IN THE SMALLER */
    /*         COMPONENT. HOWEVER, THE PHASE ANGLE RETAINS ABSOLUTE ACCURACY */
    /*         BECAUSE, IN COMPLEX ARITHMETIC WITH PRECISION P, THE SMALLER */
    /*         COMPONENT WILL NOT (AS A RULE) DECREASE BELOW P TIMES THE */
    /*         MAGNITUDE OF THE LARGER COMPONENT. IN THESE EXTREME CASES, */
    /*         THE PRINCIPAL PHASE ANGLE IS ON THE ORDER OF +P, -P, PI/2-P, */
    /*         OR -PI/2+P. */

    /* ***REFERENCES  HANDBOOK OF MATHEMATICAL FUNCTIONS BY M. ABRAMOWITZ */
    /*                 AND I. A. STEGUN, NBS AMS SERIES 55, U.S. DEPT. OF */
    /*                 COMMERCE, 1955. */

    /*               COMPUTATION OF BESSEL FUNCTIONS OF COMPLEX ARGUMENT */
    /*                 BY D. E. AMOS, SAND83-0083, MAY, 1983. */

    /*               COMPUTATION OF BESSEL FUNCTIONS OF COMPLEX ARGUMENT */
    /*                 AND LARGE ORDER BY D. E. AMOS, SAND83-0643, MAY, 1983 */

    /*               A SUBROUTINE PACKAGE FOR BESSEL FUNCTIONS OF A COMPLEX */
    /*                 ARGUMENT AND NONNEGATIVE ORDER BY D. E. AMOS, SAND85- */
    /*                 1018, MAY, 1985 */

    /*               A PORTABLE PACKAGE FOR BESSEL FUNCTIONS OF A COMPLEX */
    /*                 ARGUMENT AND NONNEGATIVE ORDER BY D. E. AMOS, ACM */
    /*                 TRANS. MATH. SOFTWARE, VOL. 12, NO. 3, SEPTEMBER 1986, */
    /*                 PP 265-273. */

    /* ***ROUTINES CALLED  ZBINU,ZABS,I1MACH,D1MACH */
    /* ***END PROLOGUE  ZBESI */
    /*     COMPLEX CONE,CSGN,CW,CY,CZERO,Z,ZN */
    /* Parameter adjustments */
    --cyi;
    --cyr;

    /* Function Body */

    /* ***FIRST EXECUTABLE STATEMENT  ZBESI */
    *ierr = 0;
    *nz   = 0;
    if(*fnu < 0.)
    {
        *ierr = 1;
    }
    if(*kode < 1 || *kode > 2)
    {
        *ierr = 1;
    }
    if(*n < 1)
    {
        *ierr = 1;
    }
    if(*ierr != 0)
    {
        return 0;
    }
    /* ----------------------------------------------------------------------- */
    /*     SET PARAMETERS RELATED TO MACHINE CONSTANTS. */
    /*     TOL IS THE APPROXIMATE UNIT ROUNDOFF LIMITED TO 1.0E-18. */
    /*     ELIM IS THE APPROXIMATE EXPONENTIAL OVER- AND UNDERFLOW LIMIT. */
    /*     EXP(-ELIM).LT.EXP(-ALIM)=EXP(-ELIM)/TOL    AND */
    /*     EXP(ELIM).GT.EXP(ALIM)=EXP(ELIM)*TOL       ARE INTERVALS NEAR */
    /*     UNDERFLOW AND OVERFLOW LIMITS WHERE SCALED ARITHMETIC IS DONE. */
    /*     RL IS THE LOWER BOUNDARY OF THE ASYMPTOTIC EXPANSION FOR LARGE Z. */
    /*     DIG = NUMBER OF BASE 10 DIGITS IN TOL = 10**(-DIG). */
    /*     FNUL IS THE LOWER BOUNDARY OF THE ASYMPTOTIC SERIES FOR LARGE FNU. */
    /* ----------------------------------------------------------------------- */
    /* Computing MAX */
    d__1 = d1mach_(&c__4);
    tol  = max(d__1, 1e-18);
    k1   = i1mach_(&c__15);
    k2   = i1mach_(&c__16);
    r1m5 = d1mach_(&c__5);
    /* Computing MIN */
    i__1 = abs(k1), i__2 = abs(k2);
    k    = std::min(i__1, i__2);
    elim = ((double)((float)k) * r1m5 - 3.) * 2.303;
    k1   = i1mach_(&c__14) - 1;
    aa   = r1m5 * (double)((float)k1);
    dig  = std::min(aa, 18.);
    aa *= 2.303;
    /* Computing MAX */
    d__1 = -aa;
    alim = elim + max(d__1, -41.45);
    rl   = dig * 1.2 + 3.;
    fnul = (dig - 3.) * 6. + 10.;
    /* ----------------------------------------------------------------------------- */
    /*     TEST FOR PROPER RANGE */
    /* ----------------------------------------------------------------------- */
    az = zabs_(zr, zi);
    fn = *fnu + (double)((float)(*n - 1));
    aa = .5 / tol;
    bb = (double)((float)i1mach_(&c__9)) * .5;
    aa = std::min(aa, bb);
    if(az > aa)
    {
        goto L260;
    }
    if(fn > aa)
    {
        goto L260;
    }
    aa = sqrt(aa);
    if(az > aa)
    {
        *ierr = 3;
    }
    if(fn > aa)
    {
        *ierr = 3;
    }
    znr   = *zr;
    zni   = *zi;
    csgnr = coner;
    csgni = conei;
    if(*zr >= 0.)
    {
        goto L40;
    }
    znr = -(*zr);
    zni = -(*zi);
    /* ----------------------------------------------------------------------- */
    /*     CALCULATE CSGN=EXP(FNU*PI*I) TO MINIMIZE LOSSES OF SIGNIFICANCE */
    /*     WHEN FNU IS LARGE */
    /* ----------------------------------------------------------------------- */
    inu = (int32)((float)(*fnu));
    arg = (*fnu - (double)((float)inu)) * pi;
    if(*zi < 0.)
    {
        arg = -arg;
    }
    csgnr = cos(arg);
    csgni = sin(arg);
    if(inu % 2 == 0)
    {
        goto L40;
    }
    csgnr = -csgnr;
    csgni = -csgni;
L40:
    zbinu_(&znr, &zni, fnu, kode, n, &cyr[1], &cyi[1], nz, &rl, &fnul, &tol, &elim, &alim);
    if(*nz < 0)
    {
        goto L120;
    }
    if(*zr >= 0.)
    {
        return 0;
    }
    /* ----------------------------------------------------------------------- */
    /*     ANALYTIC CONTINUATION TO THE LEFT HALF PLANE */
    /* ----------------------------------------------------------------------- */
    nn = *n - *nz;
    if(nn == 0)
    {
        return 0;
    }
    rtol  = 1. / tol;
    ascle = d1mach_(&c__1) * rtol * 1e3;
    i__1  = nn;
    for(i__ = 1; i__ <= i__1; ++i__)
    {
        /*       STR = CYR(I)*CSGNR - CYI(I)*CSGNI */
        /*       CYI(I) = CYR(I)*CSGNI + CYI(I)*CSGNR */
        /*       CYR(I) = STR */
        aa   = cyr[i__];
        bb   = cyi[i__];
        atol = 1.;
        /* Computing MAX */
        d__1 = abs(aa), d__2 = abs(bb);
        if(max(d__1, d__2) > ascle)
        {
            goto L55;
        }
        aa *= rtol;
        bb *= rtol;
        atol = tol;
    L55:
        str      = aa * csgnr - bb * csgni;
        sti      = aa * csgni + bb * csgnr;
        cyr[i__] = str * atol;
        cyi[i__] = sti * atol;
        csgnr    = -csgnr;
        csgni    = -csgni;
        /* L50: */
    }
    return 0;
L120:
    if(*nz == -2)
    {
        goto L130;
    }
    *nz   = 0;
    *ierr = 2;
    return 0;
L130:
    *nz   = 0;
    *ierr = 5;
    return 0;
L260:
    *nz   = 0;
    *ierr = 4;
    return 0;
} /* zbesi_ */

 int zbesj_(double* zr, double* zi, double* fnu, int32* kode, int32* n, double* cyr, double* cyi, int32* nz, int32* ierr)
{
    /* Initialized data */

    static double hpi = 1.57079632679489662;

    /* System generated locals */
    int32    i__1, i__2;
    double d__1, d__2;

    

    
    int32                     i__, k, k1, k2;
    double                  aa, bb, fn;
    int32                     nl;
    double                  az;
    int32                     ir;
    double                  rl, dig, cii, arg, r1m5;
    int32                     inu;
    double                  tol, sti, zni, str, znr, alim, elim;
    
    double                  atol;
    int32                     inuh;
    double                  fnul, rtol, ascle, csgni, csgnr;
   
    

    /* *********************************************************************72 */

    /* c ZBESJ computes a sequence of System::Complex<float> Bessel J functions. */

    /* ***BEGIN PROLOGUE  ZBESJ */
    /* ***DATE WRITTEN   830501   (YYMMDD) */
    /* ***REVISION DATE  890801, 930101   (YYMMDD) */
    /* ***CATEGORY NO.  B5K */
    /* ***KEYWORDS  J-BESSEL FUNCTION,BESSEL FUNCTION OF COMPLEX ARGUMENT, */
    /*             BESSEL FUNCTION OF FIRST KIND */
    /* ***AUTHOR  AMOS, DONALD E., SANDIA NATIONAL LABORATORIES */
    /* ***PURPOSE  TO COMPUTE THE J-BESSEL FUNCTION OF A COMPLEX ARGUMENT */
    /* ***DESCRIPTION */

    /*                      ***A DOUBLE PRECISION ROUTINE*** */
    /*         ON KODE=1, ZBESJ COMPUTES AN N MEMBER  SEQUENCE OF COMPLEX */
    /*         BESSEL FUNCTIONS CY(I)=J(FNU+I-1,Z) FOR REAL, NONNEGATIVE */
    /*         ORDERS FNU+I-1, I=1,...,N AND COMPLEX Z IN THE CUT PLANE */
    /*         -PI.LT.ARG(Z).LE.PI. ON KODE=2, ZBESJ RETURNS THE SCALED */
    /*         FUNCTIONS */

    /*         CY(I)=EXP(-ABS(Y))*J(FNU+I-1,Z)   I = 1,...,N , Y=AIMAG(Z) */

    /*         WHICH REMOVE THE EXPONENTIAL GROWTH IN BOTH THE UPPER AND */
    /*         LOWER HALF PLANES FOR Z TO INFINITY. DEFINITIONS AND NOTATION */
    /*         ARE FOUND IN THE NBS HANDBOOK OF MATHEMATICAL FUNCTIONS */
    /*         (CONST. 1). */

    /*         INPUT      ZR,ZI,FNU ARE DOUBLE PRECISION */
    /*           ZR,ZI  - Z=CMPLX(ZR,ZI),  -PI.LT.ARG(Z).LE.PI */
    /*           FNU    - ORDER OF INITIAL J FUNCTION, FNU.GE.0.0D0 */
    /*           KODE   - A PARAMETER TO INDICATE THE SCALING OPTION */
    /*                    KODE= 1  RETURNS */
    /*                             CY(I)=J(FNU+I-1,Z), I=1,...,N */
    /*                        = 2  RETURNS */
    /*                             CY(I)=J(FNU+I-1,Z)EXP(-ABS(Y)), I=1,...,N */
    /*           N      - NUMBER OF MEMBERS OF THE SEQUENCE, N.GE.1 */

    /*         OUTPUT     CYR,CYI ARE DOUBLE PRECISION */
    /*           CYR,CYI- DOUBLE PRECISION VECTORS WHOSE FIRST N COMPONENTS */
    /*                    CONTAIN REAL AND IMAGINARY PARTS FOR THE SEQUENCE */
    /*                    CY(I)=J(FNU+I-1,Z)  OR */
    /*                    CY(I)=J(FNU+I-1,Z)EXP(-ABS(Y))  I=1,...,N */
    /*                    DEPENDING ON KODE, Y=AIMAG(Z). */
    /*           NZ     - NUMBER OF COMPONENTS SET TO ZERO DUE TO UNDERFLOW, */
    /*                    NZ= 0   , NORMAL RETURN */
    /*                    NZ.GT.0 , LAST NZ COMPONENTS OF CY SET  ZERO DUE */
    /*                              TO UNDERFLOW, CY(I)=CMPLX(0.0D0,0.0D0), */
    /*                              I = N-NZ+1,...,N */
    /*           IERR   - ERROR FLAG */
    /*                    IERR=0, NORMAL RETURN - COMPUTATION COMPLETED */
    /*                    IERR=1, INPUT ERROR   - NO COMPUTATION */
    /*                    IERR=2, OVERFLOW      - NO COMPUTATION, AIMAG(Z) */
    /*                            TOO LARGE ON KODE=1 */
    /*                    IERR=3, CABS(Z) OR FNU+N-1 LARGE - COMPUTATION DONE */
    /*                            BUT LOSSES OF SIGNIFCANCE BY ARGUMENT */
    /*                            REDUCTION PRODUCE LESS THAN HALF OF MACHINE */
    /*                            ACCURACY */
    /*                    IERR=4, CABS(Z) OR FNU+N-1 TOO LARGE - NO COMPUTA- */
    /*                            TION BECAUSE OF COMPLETE LOSSES OF SIGNIFI- */
    /*                            CANCE BY ARGUMENT REDUCTION */
    /*                    IERR=5, ERROR              - NO COMPUTATION, */
    /*                            ALGORITHM TERMINATION CONDITION NOT MET */

    /* ***LONG DESCRIPTION */

    /*         THE COMPUTATION IS CARRIED OUT BY THE FORMULA */

    /*         J(FNU,Z)=EXP( FNU*PI*I/2)*I(FNU,-I*Z)    AIMAG(Z).GE.0.0 */

    /*         J(FNU,Z)=EXP(-FNU*PI*I/2)*I(FNU, I*Z)    AIMAG(Z).LT.0.0 */

    /*         WHERE I**2 = -1 AND I(FNU,Z) IS THE I BESSEL FUNCTION. */

    /*         FOR NEGATIVE ORDERS,THE FORMULA */

    /*              J(-FNU,Z) = J(FNU,Z)*COS(PI*FNU) - Y(FNU,Z)*SIN(PI*FNU) */

    /*         CAN BE USED. HOWEVER,FOR LARGE ORDERS CLOSE TO INTEGERS, THE */
    /*         THE FUNCTION CHANGES RADICALLY. WHEN FNU IS A LARGE POSITIVE */
    /*         INTEGER,THE MAGNITUDE OF J(-FNU,Z)=J(FNU,Z)*COS(PI*FNU) IS A */
    /*         LARGE NEGATIVE POWER OF TEN. BUT WHEN FNU IS NOT AN INTEGER, */
    /*         Y(FNU,Z) DOMINATES IN MAGNITUDE WITH A LARGE POSITIVE POWER OF */
    /*         TEN AND THE MOST THAT THE SECOND TERM CAN BE REDUCED IS BY */
    /*         UNIT ROUNDOFF FROM THE COEFFICIENT. THUS, WIDE CHANGES CAN */
    /*         OCCUR WITHIN UNIT ROUNDOFF OF A LARGE INTEGER FOR FNU. HERE, */
    /*         LARGE MEANS FNU.GT.CABS(Z). */

    /*         IN MOST COMPLEX VARIABLE COMPUTATION, ONE MUST EVALUATE ELE- */
    /*         MENTARY FUNCTIONS. WHEN THE MAGNITUDE OF Z OR FNU+N-1 IS */
    /*         LARGE, LOSSES OF SIGNIFICANCE BY ARGUMENT REDUCTION OCCUR. */
    /*         CONSEQUENTLY, IF EITHER ONE EXCEEDS U1=SQRT(0.5/UR), THEN */
    /*         LOSSES EXCEEDING HALF PRECISION ARE LIKELY AND AN ERROR FLAG */
    /*         IERR=3 IS TRIGGERED WHERE UR=DMAX1(D1MACH(4),1.0D-18) IS */
    /*         DOUBLE PRECISION UNIT ROUNDOFF LIMITED TO 18 DIGITS PRECISION. */
    /*         IF EITHER IS LARGER THAN U2=0.5/UR, THEN ALL SIGNIFICANCE IS */
    /*         LOST AND IERR=4. IN ORDER TO USE THE INT FUNCTION, ARGUMENTS */
    /*         MUST BE FURTHER RESTRICTED NOT TO EXCEED THE LARGEST MACHINE */
    /*         INTEGER, U3=I1MACH(9). THUS, THE MAGNITUDE OF Z AND FNU+N-1 IS */
    /*         RESTRICTED BY MIN(U2,U3). ON 32 BIT MACHINES, U1,U2, AND U3 */
    /*         ARE APPROXIMATELY 2.0E+3, 4.2E+6, 2.1E+9 IN SINGLE PRECISION */
    /*         ARITHMETIC AND 1.3E+8, 1.8E+16, 2.1E+9 IN DOUBLE PRECISION */
    /*         ARITHMETIC RESPECTIVELY. THIS MAKES U2 AND U3 LIMITING IN */
    /*         THEIR RESPECTIVE ARITHMETICS. THIS MEANS THAT ONE CAN EXPECT */
    /*         TO RETAIN, IN THE WORST CASES ON 32 BIT MACHINES, NO DIGITS */
    /*         IN SINGLE AND ONLY 7 DIGITS IN DOUBLE PRECISION ARITHMETIC. */
    /*         SIMILAR CONSIDERATIONS HOLD FOR OTHER MACHINES. */

    /*         THE APPROXIMATE RELATIVE ERROR IN THE MAGNITUDE OF A COMPLEX */
    /*         BESSEL FUNCTION CAN BE EXPRESSED BY P*10**S WHERE P=MAX(UNIT */
    /*         ROUNDOFF,1.0E-18) IS THE NOMINAL PRECISION AND 10**S REPRE- */
    /*         SENTS THE INCREASE IN ERROR DUE TO ARGUMENT REDUCTION IN THE */
    /*         ELEMENTARY FUNCTIONS. HERE, S=MAX(1,ABS(LOG10(CABS(Z))), */
    /*         ABS(LOG10(FNU))) APPROXIMATELY (I.E. S=MAX(1,ABS(EXPONENT OF */
    /*         CABS(Z),ABS(EXPONENT OF FNU)) ). HOWEVER, THE PHASE ANGLE MAY */
    /*         HAVE ONLY ABSOLUTE ACCURACY. THIS IS MOST LIKELY TO OCCUR WHEN */
    /*         ONE COMPONENT (IN ABSOLUTE VALUE) IS LARGER THAN THE OTHER BY */
    /*         SEVERAL ORDERS OF MAGNITUDE. IF ONE COMPONENT IS 10**K LARGER */
    /*         THAN THE OTHER, THEN ONE CAN EXPECT ONLY MAX(ABS(LOG10(P))-K, */
    /*         0) SIGNIFICANT DIGITS; OR, STATED ANOTHER WAY, WHEN K EXCEEDS */
    /*         THE EXPONENT OF P, NO SIGNIFICANT DIGITS REMAIN IN THE SMALLER */
    /*         COMPONENT. HOWEVER, THE PHASE ANGLE RETAINS ABSOLUTE ACCURACY */
    /*         BECAUSE, IN COMPLEX ARITHMETIC WITH PRECISION P, THE SMALLER */
    /*         COMPONENT WILL NOT (AS A RULE) DECREASE BELOW P TIMES THE */
    /*         MAGNITUDE OF THE LARGER COMPONENT. IN THESE EXTREME CASES, */
    /*         THE PRINCIPAL PHASE ANGLE IS ON THE ORDER OF +P, -P, PI/2-P, */
    /*         OR -PI/2+P. */

    /* ***REFERENCES  HANDBOOK OF MATHEMATICAL FUNCTIONS BY M. ABRAMOWITZ */
    /*                 AND I. A. STEGUN, NBS AMS SERIES 55, U.S. DEPT. OF */
    /*                 COMMERCE, 1955. */

    /*               COMPUTATION OF BESSEL FUNCTIONS OF COMPLEX ARGUMENT */
    /*                 BY D. E. AMOS, SAND83-0083, MAY, 1983. */

    /*               COMPUTATION OF BESSEL FUNCTIONS OF COMPLEX ARGUMENT */
    /*                 AND LARGE ORDER BY D. E. AMOS, SAND83-0643, MAY, 1983 */

    /*               A SUBROUTINE PACKAGE FOR BESSEL FUNCTIONS OF A COMPLEX */
    /*                 ARGUMENT AND NONNEGATIVE ORDER BY D. E. AMOS, SAND85- */
    /*                 1018, MAY, 1985 */

    /*               A PORTABLE PACKAGE FOR BESSEL FUNCTIONS OF A COMPLEX */
    /*                 ARGUMENT AND NONNEGATIVE ORDER BY D. E. AMOS, ACM */
    /*                 TRANS. MATH. SOFTWARE, VOL. 12, NO. 3, SEPTEMBER 1986, */
    /*                 PP 265-273. */

    /* ***ROUTINES CALLED  ZBINU,ZABS,I1MACH,D1MACH */
    /* ***END PROLOGUE  ZBESJ */

    /*     COMPLEX CI,CSGN,CY,Z,ZN */
    /* Parameter adjustments */
    --cyi;
    --cyr;

    /* Function Body */

    /* ***FIRST EXECUTABLE STATEMENT  ZBESJ */
    *ierr = 0;
    *nz   = 0;
    if(*fnu < 0.)
    {
        *ierr = 1;
    }
    if(*kode < 1 || *kode > 2)
    {
        *ierr = 1;
    }
    if(*n < 1)
    {
        *ierr = 1;
    }
    if(*ierr != 0)
    {
        return 0;
    }
    /* ----------------------------------------------------------------------- */
    /*     SET PARAMETERS RELATED TO MACHINE CONSTANTS. */
    /*     TOL IS THE APPROXIMATE UNIT ROUNDOFF LIMITED TO 1.0E-18. */
    /*     ELIM IS THE APPROXIMATE EXPONENTIAL OVER- AND UNDERFLOW LIMIT. */
    /*     EXP(-ELIM).LT.EXP(-ALIM)=EXP(-ELIM)/TOL    AND */
    /*     EXP(ELIM).GT.EXP(ALIM)=EXP(ELIM)*TOL       ARE INTERVALS NEAR */
    /*     UNDERFLOW AND OVERFLOW LIMITS WHERE SCALED ARITHMETIC IS DONE. */
    /*     RL IS THE LOWER BOUNDARY OF THE ASYMPTOTIC EXPANSION FOR LARGE Z. */
    /*     DIG = NUMBER OF BASE 10 DIGITS IN TOL = 10**(-DIG). */
    /*     FNUL IS THE LOWER BOUNDARY OF THE ASYMPTOTIC SERIES FOR LARGE FNU. */
    /* ----------------------------------------------------------------------- */
    /* Computing MAX */
    d__1 = d1mach_(&c__4);
    tol  = max(d__1, 1e-18);
    k1   = i1mach_(&c__15);
    k2   = i1mach_(&c__16);
    r1m5 = d1mach_(&c__5);
    /* Computing MIN */
    i__1 = abs(k1), i__2 = abs(k2);
    k    = std::min(i__1, i__2);
    elim = ((double)((float)k) * r1m5 - 3.) * 2.303;
    k1   = i1mach_(&c__14) - 1;
    aa   = r1m5 * (double)((float)k1);
    dig  = std::min(aa, 18.);
    aa *= 2.303;
    /* Computing MAX */
    d__1 = -aa;
    alim = elim + max(d__1, -41.45);
    rl   = dig * 1.2 + 3.;
    fnul = (dig - 3.) * 6. + 10.;
    /* ----------------------------------------------------------------------- */
    /*     TEST FOR PROPER RANGE */
    /* ----------------------------------------------------------------------- */
    az = zabs_(zr, zi);
    fn = *fnu + (double)((float)(*n - 1));
    aa = .5 / tol;
    bb = (double)((float)i1mach_(&c__9)) * .5;
    aa = std::min(aa, bb);
    if(az > aa)
    {
        goto L260;
    }
    if(fn > aa)
    {
        goto L260;
    }
    aa = sqrt(aa);
    if(az > aa)
    {
        *ierr = 3;
    }
    if(fn > aa)
    {
        *ierr = 3;
    }
    /* ----------------------------------------------------------------------- */
    /*     CALCULATE CSGN=EXP(FNU*HPI*I) TO MINIMIZE LOSSES OF SIGNIFICANCE */
    /*     WHEN FNU IS LARGE */
    /* ----------------------------------------------------------------------- */
    cii   = 1.;
    inu   = (int32)((float)(*fnu));
    inuh  = inu / 2;
    ir    = inu - (inuh << 1);
    arg   = (*fnu - (double)((float)(inu - ir))) * hpi;
    csgnr = cos(arg);
    csgni = sin(arg);
    if(inuh % 2 == 0)
    {
        goto L40;
    }
    csgnr = -csgnr;
    csgni = -csgni;
L40:
    /* ----------------------------------------------------------------------- */
    /*     ZN IS IN THE RIGHT HALF PLANE */
    /* ----------------------------------------------------------------------- */
    znr = *zi;
    zni = -(*zr);
    if(*zi >= 0.)
    {
        goto L50;
    }
    znr   = -znr;
    zni   = -zni;
    csgni = -csgni;
    cii   = -cii;
L50:
    zbinu_(&znr, &zni, fnu, kode, n, &cyr[1], &cyi[1], nz, &rl, &fnul, &tol, &elim, &alim);
    if(*nz < 0)
    {
        goto L130;
    }
    nl = *n - *nz;
    if(nl == 0)
    {
        return 0;
    }
    rtol  = 1. / tol;
    ascle = d1mach_(&c__1) * rtol * 1e3;
    i__1  = nl;
    for(i__ = 1; i__ <= i__1; ++i__)
    {
        /*       STR = CYR(I)*CSGNR - CYI(I)*CSGNI */
        /*       CYI(I) = CYR(I)*CSGNI + CYI(I)*CSGNR */
        /*       CYR(I) = STR */
        aa   = cyr[i__];
        bb   = cyi[i__];
        atol = 1.;
        /* Computing MAX */
        d__1 = abs(aa), d__2 = abs(bb);
        if(max(d__1, d__2) > ascle)
        {
            goto L55;
        }
        aa *= rtol;
        bb *= rtol;
        atol = tol;
    L55:
        str      = aa * csgnr - bb * csgni;
        sti      = aa * csgni + bb * csgnr;
        cyr[i__] = str * atol;
        cyi[i__] = sti * atol;
        str      = -csgni * cii;
        csgni    = csgnr * cii;
        csgnr    = str;
        /* L60: */
    }
    return 0;
L130:
    if(*nz == -2)
    {
        goto L140;
    }
    *nz   = 0;
    *ierr = 2;
    return 0;
L140:
    *nz   = 0;
    *ierr = 5;
    return 0;
L260:
    *nz   = 0;
    *ierr = 4;
    return 0;
} /* zbesj_ */

 int zbesk_(double* zr, double* zi, double* fnu, int32* kode, int32* n, double* cyr, double* cyi, int32* nz, int32* ierr)
{
    /* System generated locals */
    int32    i__1, i__2;
    double d__1;

    

    
    int32                     k, k1, k2;
    double                  aa, bb, fn, az;
    int32                     nn;
    double                  rl;
    int32                     mr, nw;
    double                  dig, arg, aln, r1m5, ufl;
    int32                     nuf;
    double                  tol, alim, elim;
    
    double                  fnul;
   

    /* *********************************************************************72 */

    /* c ZBESK computes a sequence of System::Complex<float> Bessel K functions. */

    /* ***BEGIN PROLOGUE  ZBESK */
    /* ***DATE WRITTEN   830501   (YYMMDD) */
    /* ***REVISION DATE  890801, 930101   (YYMMDD) */
    /* ***CATEGORY NO.  B5K */
    /* ***KEYWORDS  K-BESSEL FUNCTION,COMPLEX BESSEL FUNCTION, */
    /*             MODIFIED BESSEL FUNCTION OF THE SECOND KIND, */
    /*             BESSEL FUNCTION OF THE THIRD KIND */
    /* ***AUTHOR  AMOS, DONALD E., SANDIA NATIONAL LABORATORIES */
    /* ***PURPOSE  TO COMPUTE K-BESSEL FUNCTIONS OF COMPLEX ARGUMENT */
    /* ***DESCRIPTION */

    /*                      ***A DOUBLE PRECISION ROUTINE*** */

    /*         ON KODE=1, ZBESK COMPUTES AN N MEMBER SEQUENCE OF COMPLEX */
    /*         BESSEL FUNCTIONS CY(J)=K(FNU+J-1,Z) FOR REAL, NONNEGATIVE */
    /*         ORDERS FNU+J-1, J=1,...,N AND COMPLEX Z.NE.CMPLX(0.0,0.0) */
    /*         IN THE CUT PLANE -PI.LT.ARG(Z).LE.PI. ON KODE=2, ZBESK */
    /*         RETURNS THE SCALED K FUNCTIONS, */

    /*         CY(J)=EXP(Z)*K(FNU+J-1,Z) , J=1,...,N, */

    /*         WHICH REMOVE THE EXPONENTIAL BEHAVIOR IN BOTH THE LEFT AND */
    /*         RIGHT HALF PLANES FOR Z TO INFINITY. DEFINITIONS AND */
    /*         NOTATION ARE FOUND IN THE NBS HANDBOOK OF MATHEMATICAL */
    /*         FUNCTIONS (CONST. 1). */

    /*         INPUT      ZR,ZI,FNU ARE DOUBLE PRECISION */
    /*           ZR,ZI  - Z=CMPLX(ZR,ZI), Z.NE.CMPLX(0.0D0,0.0D0), */
    /*                    -PI.LT.ARG(Z).LE.PI */
    /*           FNU    - ORDER OF INITIAL K FUNCTION, FNU.GE.0.0D0 */
    /*           N      - NUMBER OF MEMBERS OF THE SEQUENCE, N.GE.1 */
    /*           KODE   - A PARAMETER TO INDICATE THE SCALING OPTION */
    /*                    KODE= 1  RETURNS */
    /*                             CY(I)=K(FNU+I-1,Z), I=1,...,N */
    /*                        = 2  RETURNS */
    /*                             CY(I)=K(FNU+I-1,Z)*EXP(Z), I=1,...,N */

    /*         OUTPUT     CYR,CYI ARE DOUBLE PRECISION */
    /*           CYR,CYI- DOUBLE PRECISION VECTORS WHOSE FIRST N COMPONENTS */
    /*                    CONTAIN REAL AND IMAGINARY PARTS FOR THE SEQUENCE */
    /*                    CY(I)=K(FNU+I-1,Z), I=1,...,N OR */
    /*                    CY(I)=K(FNU+I-1,Z)*EXP(Z), I=1,...,N */
    /*                    DEPENDING ON KODE */
    /*           NZ     - NUMBER OF COMPONENTS SET TO ZERO DUE TO UNDERFLOW. */
    /*                    NZ= 0   , NORMAL RETURN */
    /*                    NZ.GT.0 , FIRST NZ COMPONENTS OF CY SET TO ZERO DUE */
    /*                              TO UNDERFLOW, CY(I)=CMPLX(0.0D0,0.0D0), */
    /*                              I=1,...,N WHEN X.GE.0.0. WHEN X.LT.0.0 */
    /*                              NZ STATES ONLY THE NUMBER OF UNDERFLOWS */
    /*                              IN THE SEQUENCE. */

    /*           IERR   - ERROR FLAG */
    /*                    IERR=0, NORMAL RETURN - COMPUTATION COMPLETED */
    /*                    IERR=1, INPUT ERROR   - NO COMPUTATION */
    /*                    IERR=2, OVERFLOW      - NO COMPUTATION, FNU IS */
    /*                            TOO LARGE OR CABS(Z) IS TOO SMALL OR BOTH */
    /*                    IERR=3, CABS(Z) OR FNU+N-1 LARGE - COMPUTATION DONE */
    /*                            BUT LOSSES OF SIGNIFCANCE BY ARGUMENT */
    /*                            REDUCTION PRODUCE LESS THAN HALF OF MACHINE */
    /*                            ACCURACY */
    /*                    IERR=4, CABS(Z) OR FNU+N-1 TOO LARGE - NO COMPUTA- */
    /*                            TION BECAUSE OF COMPLETE LOSSES OF SIGNIFI- */
    /*                            CANCE BY ARGUMENT REDUCTION */
    /*                    IERR=5, ERROR              - NO COMPUTATION, */
    /*                            ALGORITHM TERMINATION CONDITION NOT MET */

    /* ***LONG DESCRIPTION */

    /*         EQUATIONS OF THE REFERENCE ARE IMPLEMENTED FOR SMALL ORDERS */
    /*         DNU AND DNU+1.0 IN THE RIGHT HALF PLANE X.GE.0.0. FORWARD */
    /*         RECURRENCE GENERATES HIGHER ORDERS. K IS CONTINUED TO THE LEFT */
    /*         HALF PLANE BY THE RELATION */

    /*         K(FNU,Z*EXP(MP)) = EXP(-MP*FNU)*K(FNU,Z)-MP*I(FNU,Z) */
    /*         MP=MR*PI*I, MR=+1 OR -1, RE(Z).GT.0, I**2=-1 */

    /*         WHERE I(FNU,Z) IS THE I BESSEL FUNCTION. */

    /*         FOR LARGE ORDERS, FNU.GT.FNUL, THE K FUNCTION IS COMPUTED */
    /*         BY MEANS OF ITS UNIFORM ASYMPTOTIC EXPANSIONS. */

    /*         FOR NEGATIVE ORDERS, THE FORMULA */

    /*                       K(-FNU,Z) = K(FNU,Z) */

    /*         CAN BE USED. */

    /*         ZBESK ASSUMES THAT A SIGNIFICANT DIGIT SINH(X) FUNCTION IS */
    /*         AVAILABLE. */

    /*         IN MOST COMPLEX VARIABLE COMPUTATION, ONE MUST EVALUATE ELE- */
    /*         MENTARY FUNCTIONS. WHEN THE MAGNITUDE OF Z OR FNU+N-1 IS */
    /*         LARGE, LOSSES OF SIGNIFICANCE BY ARGUMENT REDUCTION OCCUR. */
    /*         CONSEQUENTLY, IF EITHER ONE EXCEEDS U1=SQRT(0.5/UR), THEN */
    /*         LOSSES EXCEEDING HALF PRECISION ARE LIKELY AND AN ERROR FLAG */
    /*         IERR=3 IS TRIGGERED WHERE UR=DMAX1(D1MACH(4),1.0D-18) IS */
    /*         DOUBLE PRECISION UNIT ROUNDOFF LIMITED TO 18 DIGITS PRECISION. */
    /*         IF EITHER IS LARGER THAN U2=0.5/UR, THEN ALL SIGNIFICANCE IS */
    /*         LOST AND IERR=4. IN ORDER TO USE THE INT FUNCTION, ARGUMENTS */
    /*         MUST BE FURTHER RESTRICTED NOT TO EXCEED THE LARGEST MACHINE */
    /*         INTEGER, U3=I1MACH(9). THUS, THE MAGNITUDE OF Z AND FNU+N-1 IS */
    /*         RESTRICTED BY MIN(U2,U3). ON 32 BIT MACHINES, U1,U2, AND U3 */
    /*         ARE APPROXIMATELY 2.0E+3, 4.2E+6, 2.1E+9 IN SINGLE PRECISION */
    /*         ARITHMETIC AND 1.3E+8, 1.8E+16, 2.1E+9 IN DOUBLE PRECISION */
    /*         ARITHMETIC RESPECTIVELY. THIS MAKES U2 AND U3 LIMITING IN */
    /*         THEIR RESPECTIVE ARITHMETICS. THIS MEANS THAT ONE CAN EXPECT */
    /*         TO RETAIN, IN THE WORST CASES ON 32 BIT MACHINES, NO DIGITS */
    /*         IN SINGLE AND ONLY 7 DIGITS IN DOUBLE PRECISION ARITHMETIC. */
    /*         SIMILAR CONSIDERATIONS HOLD FOR OTHER MACHINES. */

    /*         THE APPROXIMATE RELATIVE ERROR IN THE MAGNITUDE OF A COMPLEX */
    /*         BESSEL FUNCTION CAN BE EXPRESSED BY P*10**S WHERE P=MAX(UNIT */
    /*         ROUNDOFF,1.0E-18) IS THE NOMINAL PRECISION AND 10**S REPRE- */
    /*         SENTS THE INCREASE IN ERROR DUE TO ARGUMENT REDUCTION IN THE */
    /*         ELEMENTARY FUNCTIONS. HERE, S=MAX(1,ABS(LOG10(CABS(Z))), */
    /*         ABS(LOG10(FNU))) APPROXIMATELY (I.E. S=MAX(1,ABS(EXPONENT OF */
    /*         CABS(Z),ABS(EXPONENT OF FNU)) ). HOWEVER, THE PHASE ANGLE MAY */
    /*         HAVE ONLY ABSOLUTE ACCURACY. THIS IS MOST LIKELY TO OCCUR WHEN */
    /*         ONE COMPONENT (IN ABSOLUTE VALUE) IS LARGER THAN THE OTHER BY */
    /*         SEVERAL ORDERS OF MAGNITUDE. IF ONE COMPONENT IS 10**K LARGER */
    /*         THAN THE OTHER, THEN ONE CAN EXPECT ONLY MAX(ABS(LOG10(P))-K, */
    /*         0) SIGNIFICANT DIGITS; OR, STATED ANOTHER WAY, WHEN K EXCEEDS */
    /*         THE EXPONENT OF P, NO SIGNIFICANT DIGITS REMAIN IN THE SMALLER */
    /*         COMPONENT. HOWEVER, THE PHASE ANGLE RETAINS ABSOLUTE ACCURACY */
    /*         BECAUSE, IN COMPLEX ARITHMETIC WITH PRECISION P, THE SMALLER */
    /*         COMPONENT WILL NOT (AS A RULE) DECREASE BELOW P TIMES THE */
    /*         MAGNITUDE OF THE LARGER COMPONENT. IN THESE EXTREME CASES, */
    /*         THE PRINCIPAL PHASE ANGLE IS ON THE ORDER OF +P, -P, PI/2-P, */
    /*         OR -PI/2+P. */

    /* ***REFERENCES  HANDBOOK OF MATHEMATICAL FUNCTIONS BY M. ABRAMOWITZ */
    /*                 AND I. A. STEGUN, NBS AMS SERIES 55, U.S. DEPT. OF */
    /*                 COMMERCE, 1955. */

    /*               COMPUTATION OF BESSEL FUNCTIONS OF COMPLEX ARGUMENT */
    /*                 BY D. E. AMOS, SAND83-0083, MAY, 1983. */

    /*               COMPUTATION OF BESSEL FUNCTIONS OF COMPLEX ARGUMENT */
    /*                 AND LARGE ORDER BY D. E. AMOS, SAND83-0643, MAY, 1983. */

    /*               A SUBROUTINE PACKAGE FOR BESSEL FUNCTIONS OF A COMPLEX */
    /*                 ARGUMENT AND NONNEGATIVE ORDER BY D. E. AMOS, SAND85- */
    /*                 1018, MAY, 1985 */

    /*               A PORTABLE PACKAGE FOR BESSEL FUNCTIONS OF A COMPLEX */
    /*                 ARGUMENT AND NONNEGATIVE ORDER BY D. E. AMOS, ACM */
    /*                 TRANS. MATH. SOFTWARE, VOL. 12, NO. 3, SEPTEMBER 1986, */
    /*                 PP 265-273. */

    /* ***ROUTINES CALLED  ZACON,ZBKNU,ZBUNK,ZUOIK,ZABS,I1MACH,D1MACH */
    /* ***END PROLOGUE  ZBESK */

    /*     COMPLEX CY,Z */
    /* ***FIRST EXECUTABLE STATEMENT  ZBESK */
    /* Parameter adjustments */
    --cyi;
    --cyr;

    /* Function Body */
    *ierr = 0;
    *nz   = 0;
    if(*zi == (float)0. && *zr == (float)0.)
    {
        *ierr = 1;
    }
    if(*fnu < 0.)
    {
        *ierr = 1;
    }
    if(*kode < 1 || *kode > 2)
    {
        *ierr = 1;
    }
    if(*n < 1)
    {
        *ierr = 1;
    }
    if(*ierr != 0)
    {
        return 0;
    }
    nn = *n;
    /* ----------------------------------------------------------------------- */
    /*     SET PARAMETERS RELATED TO MACHINE CONSTANTS. */
    /*     TOL IS THE APPROXIMATE UNIT ROUNDOFF LIMITED TO 1.0E-18. */
    /*     ELIM IS THE APPROXIMATE EXPONENTIAL OVER- AND UNDERFLOW LIMIT. */
    /*     EXP(-ELIM).LT.EXP(-ALIM)=EXP(-ELIM)/TOL    AND */
    /*     EXP(ELIM).GT.EXP(ALIM)=EXP(ELIM)*TOL       ARE INTERVALS NEAR */
    /*     UNDERFLOW AND OVERFLOW LIMITS WHERE SCALED ARITHMETIC IS DONE. */
    /*     RL IS THE LOWER BOUNDARY OF THE ASYMPTOTIC EXPANSION FOR LARGE Z. */
    /*     DIG = NUMBER OF BASE 10 DIGITS IN TOL = 10**(-DIG). */
    /*     FNUL IS THE LOWER BOUNDARY OF THE ASYMPTOTIC SERIES FOR LARGE FNU */
    /* ----------------------------------------------------------------------- */
    /* Computing MAX */
    d__1 = d1mach_(&c__4);
    tol  = max(d__1, 1e-18);
    k1   = i1mach_(&c__15);
    k2   = i1mach_(&c__16);
    r1m5 = d1mach_(&c__5);
    /* Computing MIN */
    i__1 = abs(k1), i__2 = abs(k2);
    k    = std::min(i__1, i__2);
    elim = ((double)((float)k) * r1m5 - 3.) * 2.303;
    k1   = i1mach_(&c__14) - 1;
    aa   = r1m5 * (double)((float)k1);
    dig  = std::min(aa, 18.);
    aa *= 2.303;
    /* Computing MAX */
    d__1 = -aa;
    alim = elim + max(d__1, -41.45);
    fnul = (dig - 3.) * 6. + 10.;
    rl   = dig * 1.2 + 3.;
    /* ----------------------------------------------------------------------------- */
    /*     TEST FOR PROPER RANGE */
    /* ----------------------------------------------------------------------- */
    az = zabs_(zr, zi);
    fn = *fnu + (double)((float)(nn - 1));
    aa = .5 / tol;
    bb = (double)((float)i1mach_(&c__9)) * .5;
    aa = std::min(aa, bb);
    if(az > aa)
    {
        goto L260;
    }
    if(fn > aa)
    {
        goto L260;
    }
    aa = sqrt(aa);
    if(az > aa)
    {
        *ierr = 3;
    }
    if(fn > aa)
    {
        *ierr = 3;
    }
    /* ----------------------------------------------------------------------- */
    /*     OVERFLOW TEST ON THE LAST MEMBER OF THE SEQUENCE */
    /* ----------------------------------------------------------------------- */
    /*     UFL = DEXP(-ELIM) */
    ufl = d1mach_(&c__1) * 1e3;
    if(az < ufl)
    {
        goto L180;
    }
    if(*fnu > fnul)
    {
        goto L80;
    }
    if(fn <= 1.)
    {
        goto L60;
    }
    if(fn > 2.)
    {
        goto L50;
    }
    if(az > tol)
    {
        goto L60;
    }
    arg = az * .5;
    aln = -fn * log(arg);
    if(aln > elim)
    {
        goto L180;
    }
    goto L60;
L50:
    zuoik_(zr, zi, fnu, kode, &c__2, &nn, &cyr[1], &cyi[1], &nuf, &tol, &elim, &alim);
    if(nuf < 0)
    {
        goto L180;
    }
    *nz += nuf;
    nn -= nuf;
    /* ----------------------------------------------------------------------- */
    /*     HERE NN=N OR NN=0 SINCE NUF=0,NN, OR -1 ON RETURN FROM CUOIK */
    /*     IF NUF=NN, THEN CY(I)=CZERO FOR ALL I */
    /* ----------------------------------------------------------------------- */
    if(nn == 0)
    {
        goto L100;
    }
L60:
    if(*zr < 0.)
    {
        goto L70;
    }
    /* ----------------------------------------------------------------------- */
    /*     RIGHT HALF PLANE COMPUTATION, REAL(Z).GE.0. */
    /* ----------------------------------------------------------------------- */
    zbknu_(zr, zi, fnu, kode, &nn, &cyr[1], &cyi[1], &nw, &tol, &elim, &alim);
    if(nw < 0)
    {
        goto L200;
    }
    *nz = nw;
    return 0;
/* ----------------------------------------------------------------------- */
/*     LEFT HALF PLANE COMPUTATION */
/*     PI/2.LT.ARG(Z).LE.PI AND -PI.LT.ARG(Z).LT.-PI/2. */
/* ----------------------------------------------------------------------- */
L70:
    if(*nz != 0)
    {
        goto L180;
    }
    mr = 1;
    if(*zi < 0.)
    {
        mr = -1;
    }
    zacon_(zr, zi, fnu, kode, &mr, &nn, &cyr[1], &cyi[1], &nw, &rl, &fnul, &tol, &elim, &alim);
    if(nw < 0)
    {
        goto L200;
    }
    *nz = nw;
    return 0;
/* ----------------------------------------------------------------------- */
/*     UNIFORM ASYMPTOTIC EXPANSIONS FOR FNU.GT.FNUL */
/* ----------------------------------------------------------------------- */
L80:
    mr = 0;
    if(*zr >= 0.)
    {
        goto L90;
    }
    mr = 1;
    if(*zi < 0.)
    {
        mr = -1;
    }
L90:
    zbunk_(zr, zi, fnu, kode, &mr, &nn, &cyr[1], &cyi[1], &nw, &tol, &elim, &alim);
    if(nw < 0)
    {
        goto L200;
    }
    *nz += nw;
    return 0;
L100:
    if(*zr < 0.)
    {
        goto L180;
    }
    return 0;
L180:
    *nz   = 0;
    *ierr = 2;
    return 0;
L200:
    if(nw == -1)
    {
        goto L180;
    }
    *nz   = 0;
    *ierr = 5;
    return 0;
L260:
    *nz   = 0;
    *ierr = 4;
    return 0;
} /* zbesk_ */

 int zbesy_(double* zr,
                            double* zi,
                            double* fnu,
                            int32*    kode,
                            int32*    n,
                            double* cyr,
                            double* cyi,
                            int32*    nz,
                            double* cwrkr,
                            double* cwrki,
                            int32*    ierr)
{
    /* Initialized data */

    static double cipr[4] = {1., 0., -1., 0.};
    static double cipi[4] = {0., 1., 0., -1.};
    static double hpi     = 1.57079632679489662;

    /* System generated locals */
    int32    i__1, i__2;
    double d__1, d__2;


    
    int32                     i__, k, k1, i4, k2;
    double                  ey;
    int32                     nz1, nz2;
    double                  d1m5, arg, exi, exr, sti, tay, tol, zni, zui, str, znr, zvi, zzi, zur, zvr, zzr, elim, ffnu, atol, rhpi;
    int32                     ifnu;
    double                  rtol, ascle, csgni, csgnr, cspni;
    double        cspnr;
    
    

    /* *********************************************************************72 */

    /* c ZBESY computes a sequence of System::Complex<float> Bessel Y functions. */

    /* ***BEGIN PROLOGUE  ZBESY */
    /* ***DATE WRITTEN   830501   (YYMMDD) */
    /* ***REVISION DATE  890801, 930101   (YYMMDD) */
    /* ***CATEGORY NO.  B5K */
    /* ***KEYWORDS  Y-BESSEL FUNCTION,BESSEL FUNCTION OF COMPLEX ARGUMENT, */
    /*             BESSEL FUNCTION OF SECOND KIND */
    /* ***AUTHOR  AMOS, DONALD E., SANDIA NATIONAL LABORATORIES */
    /* ***PURPOSE  TO COMPUTE THE Y-BESSEL FUNCTION OF A COMPLEX ARGUMENT */
    /* ***DESCRIPTION */

    /*                      ***A DOUBLE PRECISION ROUTINE*** */

    /*         ON KODE=1, ZBESY COMPUTES AN N MEMBER SEQUENCE OF COMPLEX */
    /*         BESSEL FUNCTIONS CY(I)=Y(FNU+I-1,Z) FOR REAL, NONNEGATIVE */
    /*         ORDERS FNU+I-1, I=1,...,N AND COMPLEX Z IN THE CUT PLANE */
    /*         -PI.LT.ARG(Z).LE.PI. ON KODE=2, ZBESY RETURNS THE SCALED */
    /*         FUNCTIONS */

    /*         CY(I)=EXP(-ABS(Y))*Y(FNU+I-1,Z)   I = 1,...,N , Y=AIMAG(Z) */

    /*         WHICH REMOVE THE EXPONENTIAL GROWTH IN BOTH THE UPPER AND */
    /*         LOWER HALF PLANES FOR Z TO INFINITY. DEFINITIONS AND NOTATION */
    /*         ARE FOUND IN THE NBS HANDBOOK OF MATHEMATICAL FUNCTIONS */
    /*         (CONST. 1). */

    /*         INPUT      ZR,ZI,FNU ARE DOUBLE PRECISION */
    /*           ZR,ZI  - Z=CMPLX(ZR,ZI), Z.NE.CMPLX(0.0D0,0.0D0), */
    /*                    -PI.LT.ARG(Z).LE.PI */
    /*           FNU    - ORDER OF INITIAL Y FUNCTION, FNU.GE.0.0D0 */
    /*           KODE   - A PARAMETER TO INDICATE THE SCALING OPTION */
    /*                    KODE= 1  RETURNS */
    /*                             CY(I)=Y(FNU+I-1,Z), I=1,...,N */
    /*                        = 2  RETURNS */
    /*                             CY(I)=Y(FNU+I-1,Z)*EXP(-ABS(Y)), I=1,...,N */
    /*                             WHERE Y=AIMAG(Z) */
    /*           N      - NUMBER OF MEMBERS OF THE SEQUENCE, N.GE.1 */
    /*           CWRKR, - DOUBLE PRECISION WORK VECTORS OF DIMENSION AT */
    /*           CWRKI    AT LEAST N */

    /*         OUTPUT     CYR,CYI ARE DOUBLE PRECISION */
    /*           CYR,CYI- DOUBLE PRECISION VECTORS WHOSE FIRST N COMPONENTS */
    /*                    CONTAIN REAL AND IMAGINARY PARTS FOR THE SEQUENCE */
    /*                    CY(I)=Y(FNU+I-1,Z)  OR */
    /*                    CY(I)=Y(FNU+I-1,Z)*EXP(-ABS(Y))  I=1,...,N */
    /*                    DEPENDING ON KODE. */
    /*           NZ     - NZ=0 , A NORMAL RETURN */
    /*                    NZ.GT.0 , NZ COMPONENTS OF CY SET TO ZERO DUE TO */
    /*                    UNDERFLOW (GENERALLY ON KODE=2) */
    /*           IERR   - ERROR FLAG */
    /*                    IERR=0, NORMAL RETURN - COMPUTATION COMPLETED */
    /*                    IERR=1, INPUT ERROR   - NO COMPUTATION */
    /*                    IERR=2, OVERFLOW      - NO COMPUTATION, FNU IS */
    /*                            TOO LARGE OR CABS(Z) IS TOO SMALL OR BOTH */
    /*                    IERR=3, CABS(Z) OR FNU+N-1 LARGE - COMPUTATION DONE */
    /*                            BUT LOSSES OF SIGNIFCANCE BY ARGUMENT */
    /*                            REDUCTION PRODUCE LESS THAN HALF OF MACHINE */
    /*                            ACCURACY */
    /*                    IERR=4, CABS(Z) OR FNU+N-1 TOO LARGE - NO COMPUTA- */
    /*                            TION BECAUSE OF COMPLETE LOSSES OF SIGNIFI- */
    /*                            CANCE BY ARGUMENT REDUCTION */
    /*                    IERR=5, ERROR              - NO COMPUTATION, */
    /*                            ALGORITHM TERMINATION CONDITION NOT MET */

    /* ***LONG DESCRIPTION */

    /*         THE COMPUTATION IS CARRIED OUT IN TERMS OF THE I(FNU,Z) AND */
    /*         K(FNU,Z) BESSEL FUNCTIONS IN THE RIGHT HALF PLANE BY */

    /*             Y(FNU,Z) = I*CC*I(FNU,ARG) - (2/PI)*CONJG(CC)*K(FNU,ARG) */

    /*             Y(FNU,Z) = CONJG(Y(FNU,CONJG(Z))) */

    /*         FOR AIMAG(Z).GE.0 AND AIMAG(Z).LT.0 RESPECTIVELY, WHERE */
    /*         CC=EXP(I*PI*FNU/2), ARG=Z*EXP(-I*PI/2) AND I**2=-1. */

    /*         FOR NEGATIVE ORDERS,THE FORMULA */

    /*              Y(-FNU,Z) = Y(FNU,Z)*COS(PI*FNU) + J(FNU,Z)*SIN(PI*FNU) */

    /*         CAN BE USED. HOWEVER,FOR LARGE ORDERS CLOSE TO HALF ODD */
    /*         INTEGERS THE FUNCTION CHANGES RADICALLY. WHEN FNU IS A LARGE */
    /*         POSITIVE HALF ODD INTEGER,THE MAGNITUDE OF Y(-FNU,Z)=J(FNU,Z)* */
    /*         SIN(PI*FNU) IS A LARGE NEGATIVE POWER OF TEN. BUT WHEN FNU IS */
    /*         NOT A HALF ODD INTEGER, Y(FNU,Z) DOMINATES IN MAGNITUDE WITH A */
    /*         LARGE POSITIVE POWER OF TEN AND THE MOST THAT THE SECOND TERM */
    /*         CAN BE REDUCED IS BY UNIT ROUNDOFF FROM THE COEFFICIENT. THUS, */
    /*         WIDE CHANGES CAN OCCUR WITHIN UNIT ROUNDOFF OF A LARGE HALF */
    /*         ODD INTEGER. HERE, LARGE MEANS FNU.GT.CABS(Z). */

    /*         IN MOST COMPLEX VARIABLE COMPUTATION, ONE MUST EVALUATE ELE- */
    /*         MENTARY FUNCTIONS. WHEN THE MAGNITUDE OF Z OR FNU+N-1 IS */
    /*         LARGE, LOSSES OF SIGNIFICANCE BY ARGUMENT REDUCTION OCCUR. */
    /*         CONSEQUENTLY, IF EITHER ONE EXCEEDS U1=SQRT(0.5/UR), THEN */
    /*         LOSSES EXCEEDING HALF PRECISION ARE LIKELY AND AN ERROR FLAG */
    /*         IERR=3 IS TRIGGERED WHERE UR=DMAX1(D1MACH(4),1.0D-18) IS */
    /*         DOUBLE PRECISION UNIT ROUNDOFF LIMITED TO 18 DIGITS PRECISION. */
    /*         IF EITHER IS LARGER THAN U2=0.5/UR, THEN ALL SIGNIFICANCE IS */
    /*         LOST AND IERR=4. IN ORDER TO USE THE INT FUNCTION, ARGUMENTS */
    /*         MUST BE FURTHER RESTRICTED NOT TO EXCEED THE LARGEST MACHINE */
    /*         INTEGER, U3=I1MACH(9). THUS, THE MAGNITUDE OF Z AND FNU+N-1 IS */
    /*         RESTRICTED BY MIN(U2,U3). ON 32 BIT MACHINES, U1,U2, AND U3 */
    /*         ARE APPROXIMATELY 2.0E+3, 4.2E+6, 2.1E+9 IN SINGLE PRECISION */
    /*         ARITHMETIC AND 1.3E+8, 1.8E+16, 2.1E+9 IN DOUBLE PRECISION */
    /*         ARITHMETIC RESPECTIVELY. THIS MAKES U2 AND U3 LIMITING IN */
    /*         THEIR RESPECTIVE ARITHMETICS. THIS MEANS THAT ONE CAN EXPECT */
    /*         TO RETAIN, IN THE WORST CASES ON 32 BIT MACHINES, NO DIGITS */
    /*         IN SINGLE AND ONLY 7 DIGITS IN DOUBLE PRECISION ARITHMETIC. */
    /*         SIMILAR CONSIDERATIONS HOLD FOR OTHER MACHINES. */

    /*         THE APPROXIMATE RELATIVE ERROR IN THE MAGNITUDE OF A COMPLEX */
    /*         BESSEL FUNCTION CAN BE EXPRESSED BY P*10**S WHERE P=MAX(UNIT */
    /*         ROUNDOFF,1.0E-18) IS THE NOMINAL PRECISION AND 10**S REPRE- */
    /*         SENTS THE INCREASE IN ERROR DUE TO ARGUMENT REDUCTION IN THE */
    /*         ELEMENTARY FUNCTIONS. HERE, S=MAX(1,ABS(LOG10(CABS(Z))), */
    /*         ABS(LOG10(FNU))) APPROXIMATELY (I.E. S=MAX(1,ABS(EXPONENT OF */
    /*         CABS(Z),ABS(EXPONENT OF FNU)) ). HOWEVER, THE PHASE ANGLE MAY */
    /*         HAVE ONLY ABSOLUTE ACCURACY. THIS IS MOST LIKELY TO OCCUR WHEN */
    /*         ONE COMPONENT (IN ABSOLUTE VALUE) IS LARGER THAN THE OTHER BY */
    /*         SEVERAL ORDERS OF MAGNITUDE. IF ONE COMPONENT IS 10**K LARGER */
    /*         THAN THE OTHER, THEN ONE CAN EXPECT ONLY MAX(ABS(LOG10(P))-K, */
    /*         0) SIGNIFICANT DIGITS; OR, STATED ANOTHER WAY, WHEN K EXCEEDS */
    /*         THE EXPONENT OF P, NO SIGNIFICANT DIGITS REMAIN IN THE SMALLER */
    /*         COMPONENT. HOWEVER, THE PHASE ANGLE RETAINS ABSOLUTE ACCURACY */
    /*         BECAUSE, IN COMPLEX ARITHMETIC WITH PRECISION P, THE SMALLER */
    /*         COMPONENT WILL NOT (AS A RULE) DECREASE BELOW P TIMES THE */
    /*         MAGNITUDE OF THE LARGER COMPONENT. IN THESE EXTREME CASES, */
    /*         THE PRINCIPAL PHASE ANGLE IS ON THE ORDER OF +P, -P, PI/2-P, */
    /*         OR -PI/2+P. */

    /* ***REFERENCES  HANDBOOK OF MATHEMATICAL FUNCTIONS BY M. ABRAMOWITZ */
    /*                 AND I. A. STEGUN, NBS AMS SERIES 55, U.S. DEPT. OF */
    /*                 COMMERCE, 1955. */

    /*               COMPUTATION OF BESSEL FUNCTIONS OF COMPLEX ARGUMENT */
    /*                 BY D. E. AMOS, SAND83-0083, MAY, 1983. */

    /*               COMPUTATION OF BESSEL FUNCTIONS OF COMPLEX ARGUMENT */
    /*                 AND LARGE ORDER BY D. E. AMOS, SAND83-0643, MAY, 1983 */

    /*               A SUBROUTINE PACKAGE FOR BESSEL FUNCTIONS OF A COMPLEX */
    /*                 ARGUMENT AND NONNEGATIVE ORDER BY D. E. AMOS, SAND85- */
    /*                 1018, MAY, 1985 */

    /*               A PORTABLE PACKAGE FOR BESSEL FUNCTIONS OF A COMPLEX */
    /*                 ARGUMENT AND NONNEGATIVE ORDER BY D. E. AMOS, ACM */
    /*                 TRANS. MATH. SOFTWARE, VOL. 12, NO. 3, SEPTEMBER 1986, */
    /*                 PP 265-273. */

    /* ***ROUTINES CALLED  ZBESI,ZBESK,I1MACH,D1MACH */
    /* ***END PROLOGUE  ZBESY */

    /*     COMPLEX CWRK,CY,C1,C2,EX,HCI,Z,ZU,ZV */
    /* Parameter adjustments */
    --cwrki;
    --cwrkr;
    --cyi;
    --cyr;

    /* Function Body */
    /* ***FIRST EXECUTABLE STATEMENT  ZBESY */
    *ierr = 0;
    *nz   = 0;
    if(*zr == 0. && *zi == 0.)
    {
        *ierr = 1;
    }
    if(*fnu < 0.)
    {
        *ierr = 1;
    }
    if(*kode < 1 || *kode > 2)
    {
        *ierr = 1;
    }
    if(*n < 1)
    {
        *ierr = 1;
    }
    if(*ierr != 0)
    {
        return 0;
    }
    zzr = *zr;
    zzi = *zi;
    if(*zi < 0.)
    {
        zzi = -zzi;
    }
    znr = zzi;
    zni = -zzr;
    zbesi_(&znr, &zni, fnu, kode, n, &cyr[1], &cyi[1], &nz1, ierr);
    if(*ierr != 0 && *ierr != 3)
    {
        goto L90;
    }
    zbesk_(&znr, &zni, fnu, kode, n, &cwrkr[1], &cwrki[1], &nz2, ierr);
    if(*ierr != 0 && *ierr != 3)
    {
        goto L90;
    }
    *nz   = std::min(nz1, nz2);
    ifnu  = (int32)((float)(*fnu));
    ffnu  = *fnu - (double)((float)ifnu);
    arg   = hpi * ffnu;
    csgnr = cos(arg);
    csgni = sin(arg);
    i4    = ifnu % 4 + 1;
    str   = csgnr * cipr[i4 - 1] - csgni * cipi[i4 - 1];
    csgni = csgnr * cipi[i4 - 1] + csgni * cipr[i4 - 1];
    csgnr = str;
    rhpi  = 1. / hpi;
    cspnr = csgnr * rhpi;
    cspni = -csgni * rhpi;
    str   = -csgni;
    csgni = csgnr;
    csgnr = str;
    if(*kode == 2)
    {
        goto L60;
    }
    i__1 = *n;
    for(i__ = 1; i__ <= i__1; ++i__)
    {
        /*       CY(I) = CSGN*CY(I)-CSPN*CWRK(I) */
        str = csgnr * cyr[i__] - csgni * cyi[i__];
        str -= cspnr * cwrkr[i__] - cspni * cwrki[i__];
        sti = csgnr * cyi[i__] + csgni * cyr[i__];
        sti -= cspnr * cwrki[i__] + cspni * cwrkr[i__];
        cyr[i__] = str;
        cyi[i__] = sti;
        str      = -csgni;
        csgni    = csgnr;
        csgnr    = str;
        str      = cspni;
        cspni    = -cspnr;
        cspnr    = str;
        /* L50: */
    }
    if(*zi < 0.)
    {
        i__1 = *n;
        for(i__ = 1; i__ <= i__1; ++i__)
        {
            cyi[i__] = -cyi[i__];
            /* L55: */
        }
    }
    return 0;
L60:
    exr = cos(*zr);
    exi = sin(*zr);
    /* Computing MAX */
    d__1 = d1mach_(&c__4);
    tol  = max(d__1, 1e-18);
    k1   = i1mach_(&c__15);
    k2   = i1mach_(&c__16);
    /* Computing MIN */
    i__1 = abs(k1), i__2 = abs(k2);
    k    = std::min(i__1, i__2);
    d1m5 = d1mach_(&c__5);
    /* ----------------------------------------------------------------------- */
    /*     ELIM IS THE APPROXIMATE EXPONENTIAL UNDER- AND OVERFLOW LIMIT */
    /* ----------------------------------------------------------------------- */
    elim = ((double)((float)k) * d1m5 - 3.) * 2.303;
    ey   = 0.;
    tay  = (d__1 = *zi + *zi, abs(d__1));
    if(tay < elim)
    {
        ey = exp(-tay);
    }
    str   = (exr * cspnr - exi * cspni) * ey;
    cspni = (exr * cspni + exi * cspnr) * ey;
    cspnr = str;
    *nz   = 0;
    rtol  = 1. / tol;
    ascle = d1mach_(&c__1) * rtol * 1e3;
    i__1  = *n;
    for(i__ = 1; i__ <= i__1; ++i__)
    {
        /* ---------------------------------------------------------------------- */
        /*       CY(I) = CSGN*CY(I)-CSPN*CWRK(I): PRODUCTS ARE COMPUTED IN */
        /*       SCALED MODE IF CY(I) OR CWRK(I) ARE CLOSE TO UNDERFLOW TO */
        /*       PREVENT UNDERFLOW IN AN INTERMEDIATE COMPUTATION. */
        /* ---------------------------------------------------------------------- */
        zvr  = cwrkr[i__];
        zvi  = cwrki[i__];
        atol = 1.;
        /* Computing MAX */
        d__1 = abs(zvr), d__2 = abs(zvi);
        if(max(d__1, d__2) > ascle)
        {
            goto L75;
        }
        zvr *= rtol;
        zvi *= rtol;
        atol = tol;
    L75:
        str  = (zvr * cspnr - zvi * cspni) * atol;
        zvi  = (zvr * cspni + zvi * cspnr) * atol;
        zvr  = str;
        zur  = cyr[i__];
        zui  = cyi[i__];
        atol = 1.;
        /* Computing MAX */
        d__1 = abs(zur), d__2 = abs(zui);
        if(max(d__1, d__2) > ascle)
        {
            goto L85;
        }
        zur *= rtol;
        zui *= rtol;
        atol = tol;
    L85:
        str      = (zur * csgnr - zui * csgni) * atol;
        zui      = (zur * csgni + zui * csgnr) * atol;
        zur      = str;
        cyr[i__] = zur - zvr;
        cyi[i__] = zui - zvi;
        if(*zi < 0.)
        {
            cyi[i__] = -cyi[i__];
        }
        if(cyr[i__] == 0. && cyi[i__] == 0. && ey == 0.)
        {
            ++(*nz);
        }
        str   = -csgni;
        csgni = csgnr;
        csgnr = str;
        str   = cspni;
        cspni = -cspnr;
        cspnr = str;
        /* L80: */
    }
    return 0;
L90:
    *nz = 0;
    return 0;
} /* zbesy_ */

 int zbinu_(double* zr,
                            double* zi,
                            double* fnu,
                            int32*    kode,
                            int32*    n,
                            double* cyr,
                            double* cyi,
                            int32*    nz,
                            double* rl,
                            double* fnul,
                            double* tol,
                            double* elim,
                            double* alim)
{
    /* Initialized data */

    static double zeror = 0.;
    static double zeroi = 0.;

    /* System generated locals */
    int32 i__1;

    
    int32                     i__;
    double                  az;
    int32                     nn, nw;
    double                  cwi[2], cwr[2];
    int32                     nui, inw;
    double                  dfnu;
    
    int32                     nlast;
  

    /* *********************************************************************72 */

    /* ***BEGIN PROLOGUE  ZBINU */
    /* ***REFER TO  ZBESH,ZBESI,ZBESJ,ZBESK,ZAIRY,ZBIRY */

    /*     ZBINU COMPUTES THE I FUNCTION IN THE RIGHT HALF Z PLANE */

    /* ***ROUTINES CALLED  ZABS,ZASYI,ZBUNI,ZMLRI,ZSERI,ZUOIK,ZWRSK */
    /* ***END PROLOGUE  ZBINU */
    /* Parameter adjustments */
    --cyi;
    --cyr;

    /* Function Body */

    *nz  = 0;
    az   = zabs_(zr, zi);
    nn   = *n;
    dfnu = *fnu + (double)((float)(*n - 1));
    if(az <= 2.)
    {
        goto L10;
    }
    if(az * az * .25 > dfnu + 1.)
    {
        goto L20;
    }
L10:
    /* ----------------------------------------------------------------------- */
    /*     POWER SERIES */
    /* ----------------------------------------------------------------------- */
    zseri_(zr, zi, fnu, kode, &nn, &cyr[1], &cyi[1], &nw, tol, elim, alim);
    inw = abs(nw);
    *nz += inw;
    nn -= inw;
    if(nn == 0)
    {
        return 0;
    }
    if(nw >= 0)
    {
        goto L120;
    }
    dfnu = *fnu + (double)((float)(nn - 1));
L20:
    if(az < *rl)
    {
        goto L40;
    }
    if(dfnu <= 1.)
    {
        goto L30;
    }
    if(az + az < dfnu * dfnu)
    {
        goto L50;
    }
/* ----------------------------------------------------------------------- */
/*     ASYMPTOTIC EXPANSION FOR LARGE Z */
/* ----------------------------------------------------------------------- */
L30:
    zasyi_(zr, zi, fnu, kode, &nn, &cyr[1], &cyi[1], &nw, rl, tol, elim, alim);
    if(nw < 0)
    {
        goto L130;
    }
    goto L120;
L40:
    if(dfnu <= 1.)
    {
        goto L70;
    }
L50:
    /* ----------------------------------------------------------------------- */
    /*     OVERFLOW AND UNDERFLOW TEST ON I SEQUENCE FOR MILLER ALGORITHM */
    /* ----------------------------------------------------------------------- */
    zuoik_(zr, zi, fnu, kode, &c__1, &nn, &cyr[1], &cyi[1], &nw, tol, elim, alim);
    if(nw < 0)
    {
        goto L130;
    }
    *nz += nw;
    nn -= nw;
    if(nn == 0)
    {
        return 0;
    }
    dfnu = *fnu + (double)((float)(nn - 1));
    if(dfnu > *fnul)
    {
        goto L110;
    }
    if(az > *fnul)
    {
        goto L110;
    }
L60:
    if(az > *rl)
    {
        goto L80;
    }
L70:
    /* ----------------------------------------------------------------------- */
    /*     MILLER ALGORITHM NORMALIZED BY THE SERIES */
    /* ----------------------------------------------------------------------- */
    zmlri_(zr, zi, fnu, kode, &nn, &cyr[1], &cyi[1], &nw, tol);
    if(nw < 0)
    {
        goto L130;
    }
    goto L120;
L80:
    /* ----------------------------------------------------------------------- */
    /*     MILLER ALGORITHM NORMALIZED BY THE WRONSKIAN */
    /* ----------------------------------------------------------------------- */
    /* ----------------------------------------------------------------------- */
    /*     OVERFLOW TEST ON K FUNCTIONS USED IN WRONSKIAN */
    /* ----------------------------------------------------------------------- */
    zuoik_(zr, zi, fnu, kode, &c__2, &c__2, cwr, cwi, &nw, tol, elim, alim);
    if(nw >= 0)
    {
        goto L100;
    }
    *nz  = nn;
    i__1 = nn;
    for(i__ = 1; i__ <= i__1; ++i__)
    {
        cyr[i__] = zeror;
        cyi[i__] = zeroi;
        /* L90: */
    }
    return 0;
L100:
    if(nw > 0)
    {
        goto L130;
    }
    zwrsk_(zr, zi, fnu, kode, &nn, &cyr[1], &cyi[1], &nw, cwr, cwi, tol, elim, alim);
    if(nw < 0)
    {
        goto L130;
    }
    goto L120;
L110:
    /* ----------------------------------------------------------------------- */
    /*     INCREMENT FNU+NN-1 UP TO FNUL, COMPUTE AND RECUR BACKWARD */
    /* ----------------------------------------------------------------------- */
    nui = (int32)((float)(*fnul - dfnu)) + 1;
    nui = max(nui, 0);
    zbuni_(zr, zi, fnu, kode, &nn, &cyr[1], &cyi[1], &nw, &nui, &nlast, fnul, tol, elim, alim);
    if(nw < 0)
    {
        goto L130;
    }
    *nz += nw;
    if(nlast == 0)
    {
        goto L120;
    }
    nn = nlast;
    goto L60;
L120:
    return 0;
L130:
    *nz = -1;
    if(nw == -2)
    {
        *nz = -2;
    }
    return 0;
} /* zbinu_ */

 int zbiry_(double* zr, double* zi, int32* id, int32* kode, double* bir, double* bii, int32* ierr)
{
    /* Initialized data */

    static double tth   = .666666666666666667;
    static double c1    = .614926627446000736;
    static double c2    = .448288357353826359;
    static double coef  = .577350269189625765;
    static double pi    = 3.14159265358979324;
    static double coner = 1.;
    static double conei = 0.;

    /* System generated locals */
    int32    i__1, i__2;
    double d__1;



    
    int32                     k;
    double                  d1, d2;
    int32                     k1, k2;
    double                  aa, bb, ad, cc, ak, bk, ck, dk, az, rl;
    int32                     nz;
    double                  s1i, az3, s2i, s1r, s2r, z3i, z3r, eaa, fid, dig, cyi[2], fmr, r1m5, fnu, cyr[2], tol, sti, str, sfac, alim, elim;
    
    double                  csqi, atrm, fnul, ztai, csqr;
    double                  ztar, trm1i, trm2i, trm1r, trm2r;


    /* *********************************************************************72 */

    /* c ZBIRY computes a sequence of System::Complex<float> Airy Bi functions. */

    /* ***BEGIN PROLOGUE  ZBIRY */
    /* ***DATE WRITTEN   830501   (YYMMDD) */
    /* ***REVISION DATE  890801, 930101   (YYMMDD) */
    /* ***CATEGORY NO.  B5K */
    /* ***KEYWORDS  AIRY FUNCTION,BESSEL FUNCTIONS OF ORDER ONE THIRD */
    /* ***AUTHOR  AMOS, DONALD E., SANDIA NATIONAL LABORATORIES */
    /* ***PURPOSE  TO COMPUTE AIRY FUNCTIONS BI(Z) AND DBI(Z) FOR COMPLEX Z */
    /* ***DESCRIPTION */

    /*                      ***A DOUBLE PRECISION ROUTINE*** */
    /*         ON KODE=1, CBIRY COMPUTES THE COMPLEX AIRY FUNCTION BI(Z) OR */
    /*         ITS DERIVATIVE DBI(Z)/DZ ON ID=0 OR ID=1 RESPECTIVELY. ON */
    /*         KODE=2, A SCALING OPTION CEXP(-AXZTA)*BI(Z) OR CEXP(-AXZTA)* */
    /*         DBI(Z)/DZ IS PROVIDED TO REMOVE THE EXPONENTIAL BEHAVIOR IN */
    /*         BOTH THE LEFT AND RIGHT HALF PLANES WHERE */
    /*         ZTA=(2/3)*Z*CSQRT(Z)=CMPLX(XZTA,YZTA) AND AXZTA=ABS(XZTA). */
    /*         DEFINTIONS AND NOTATION ARE FOUND IN THE NBS HANDBOOK OF */
    /*         MATHEMATICAL FUNCTIONS (CONST. 1). */

    /*         INPUT      ZR,ZI ARE DOUBLE PRECISION */
    /*           ZR,ZI  - Z=CMPLX(ZR,ZI) */
    /*           ID     - ORDER OF DERIVATIVE, ID=0 OR ID=1 */
    /*           KODE   - A PARAMETER TO INDICATE THE SCALING OPTION */
    /*                    KODE= 1  RETURNS */
    /*                             BI=BI(Z)                 ON ID=0 OR */
    /*                             BI=DBI(Z)/DZ             ON ID=1 */
    /*                        = 2  RETURNS */
    /*                             BI=CEXP(-AXZTA)*BI(Z)     ON ID=0 OR */
    /*                             BI=CEXP(-AXZTA)*DBI(Z)/DZ ON ID=1 WHERE */
    /*                             ZTA=(2/3)*Z*CSQRT(Z)=CMPLX(XZTA,YZTA) */
    /*                             AND AXZTA=ABS(XZTA) */

    /*         OUTPUT     BIR,BII ARE DOUBLE PRECISION */
    /*           BIR,BII- COMPLEX ANSWER DEPENDING ON THE CHOICES FOR ID AND */
    /*                    KODE */
    /*           IERR   - ERROR FLAG */
    /*                    IERR=0, NORMAL RETURN - COMPUTATION COMPLETED */
    /*                    IERR=1, INPUT ERROR   - NO COMPUTATION */
    /*                    IERR=2, OVERFLOW      - NO COMPUTATION, REAL(Z) */
    /*                            TOO LARGE ON KODE=1 */
    /*                    IERR=3, CABS(Z) LARGE      - COMPUTATION COMPLETED */
    /*                            LOSSES OF SIGNIFCANCE BY ARGUMENT REDUCTION */
    /*                            PRODUCE LESS THAN HALF OF MACHINE ACCURACY */
    /*                    IERR=4, CABS(Z) TOO LARGE  - NO COMPUTATION */
    /*                            COMPLETE LOSS OF ACCURACY BY ARGUMENT */
    /*                            REDUCTION */
    /*                    IERR=5, ERROR              - NO COMPUTATION, */
    /*                            ALGORITHM TERMINATION CONDITION NOT MET */

    /* ***LONG DESCRIPTION */

    /*         BI AND DBI ARE COMPUTED FOR CABS(Z).GT.1.0 FROM THE I BESSEL */
    /*         FUNCTIONS BY */

    /*                BI(Z)=C*SQRT(Z)*( I(-1/3,ZTA) + I(1/3,ZTA) ) */
    /*               DBI(Z)=C *  Z  * ( I(-2/3,ZTA) + I(2/3,ZTA) ) */
    /*                               C=1.0/SQRT(3.0) */
    /*                             ZTA=(2/3)*Z**(3/2) */

    /*         WITH THE POWER SERIES FOR CABS(Z).LE.1.0. */

    /*         IN MOST COMPLEX VARIABLE COMPUTATION, ONE MUST EVALUATE ELE- */
    /*         MENTARY FUNCTIONS. WHEN THE MAGNITUDE OF Z IS LARGE, LOSSES */
    /*         OF SIGNIFICANCE BY ARGUMENT REDUCTION OCCUR. CONSEQUENTLY, IF */
    /*         THE MAGNITUDE OF ZETA=(2/3)*Z**1.5 EXCEEDS U1=SQRT(0.5/UR), */
    /*         THEN LOSSES EXCEEDING HALF PRECISION ARE LIKELY AND AN ERROR */
    /*         FLAG IERR=3 IS TRIGGERED WHERE UR=DMAX1(D1MACH(4),1.0D-18) IS */
    /*         DOUBLE PRECISION UNIT ROUNDOFF LIMITED TO 18 DIGITS PRECISION. */
    /*         ALSO, IF THE MAGNITUDE OF ZETA IS LARGER THAN U2=0.5/UR, THEN */
    /*         ALL SIGNIFICANCE IS LOST AND IERR=4. IN ORDER TO USE THE INT */
    /*         FUNCTION, ZETA MUST BE FURTHER RESTRICTED NOT TO EXCEED THE */
    /*         LARGEST INTEGER, U3=I1MACH(9). THUS, THE MAGNITUDE OF ZETA */
    /*         MUST BE RESTRICTED BY MIN(U2,U3). ON 32 BIT MACHINES, U1,U2, */
    /*         AND U3 ARE APPROXIMATELY 2.0E+3, 4.2E+6, 2.1E+9 IN SINGLE */
    /*         PRECISION ARITHMETIC AND 1.3E+8, 1.8E+16, 2.1E+9 IN DOUBLE */
    /*         PRECISION ARITHMETIC RESPECTIVELY. THIS MAKES U2 AND U3 LIMIT- */
    /*         ING IN THEIR RESPECTIVE ARITHMETICS. THIS MEANS THAT THE MAG- */
    /*         NITUDE OF Z CANNOT EXCEED 3.1E+4 IN SINGLE AND 2.1E+6 IN */
    /*         DOUBLE PRECISION ARITHMETIC. THIS ALSO MEANS THAT ONE CAN */
    /*         EXPECT TO RETAIN, IN THE WORST CASES ON 32 BIT MACHINES, */
    /*         NO DIGITS IN SINGLE PRECISION AND ONLY 7 DIGITS IN DOUBLE */
    /*         PRECISION ARITHMETIC. SIMILAR CONSIDERATIONS HOLD FOR OTHER */
    /*         MACHINES. */

    /*         THE APPROXIMATE RELATIVE ERROR IN THE MAGNITUDE OF A COMPLEX */
    /*         BESSEL FUNCTION CAN BE EXPRESSED BY P*10**S WHERE P=MAX(UNIT */
    /*         ROUNDOFF,1.0E-18) IS THE NOMINAL PRECISION AND 10**S REPRE- */
    /*         SENTS THE INCREASE IN ERROR DUE TO ARGUMENT REDUCTION IN THE */
    /*         ELEMENTARY FUNCTIONS. HERE, S=MAX(1,ABS(LOG10(CABS(Z))), */
    /*         ABS(LOG10(FNU))) APPROXIMATELY (I.E. S=MAX(1,ABS(EXPONENT OF */
    /*         CABS(Z),ABS(EXPONENT OF FNU)) ). HOWEVER, THE PHASE ANGLE MAY */
    /*         HAVE ONLY ABSOLUTE ACCURACY. THIS IS MOST LIKELY TO OCCUR WHEN */
    /*         ONE COMPONENT (IN ABSOLUTE VALUE) IS LARGER THAN THE OTHER BY */
    /*         SEVERAL ORDERS OF MAGNITUDE. IF ONE COMPONENT IS 10**K LARGER */
    /*         THAN THE OTHER, THEN ONE CAN EXPECT ONLY MAX(ABS(LOG10(P))-K, */
    /*         0) SIGNIFICANT DIGITS; OR, STATED ANOTHER WAY, WHEN K EXCEEDS */
    /*         THE EXPONENT OF P, NO SIGNIFICANT DIGITS REMAIN IN THE SMALLER */
    /*         COMPONENT. HOWEVER, THE PHASE ANGLE RETAINS ABSOLUTE ACCURACY */
    /*         BECAUSE, IN COMPLEX ARITHMETIC WITH PRECISION P, THE SMALLER */
    /*         COMPONENT WILL NOT (AS A RULE) DECREASE BELOW P TIMES THE */
    /*         MAGNITUDE OF THE LARGER COMPONENT. IN THESE EXTREME CASES, */
    /*         THE PRINCIPAL PHASE ANGLE IS ON THE ORDER OF +P, -P, PI/2-P, */
    /*         OR -PI/2+P. */

    /* ***REFERENCES  HANDBOOK OF MATHEMATICAL FUNCTIONS BY M. ABRAMOWITZ */
    /*                 AND I. A. STEGUN, NBS AMS SERIES 55, U.S. DEPT. OF */
    /*                 COMMERCE, 1955. */

    /*               COMPUTATION OF BESSEL FUNCTIONS OF COMPLEX ARGUMENT */
    /*                 AND LARGE ORDER BY D. E. AMOS, SAND83-0643, MAY, 1983 */

    /*               A SUBROUTINE PACKAGE FOR BESSEL FUNCTIONS OF A COMPLEX */
    /*                 ARGUMENT AND NONNEGATIVE ORDER BY D. E. AMOS, SAND85- */
    /*                 1018, MAY, 1985 */

    /*               A PORTABLE PACKAGE FOR BESSEL FUNCTIONS OF A COMPLEX */
    /*                 ARGUMENT AND NONNEGATIVE ORDER BY D. E. AMOS, ACM */
    /*                 TRANS. MATH. SOFTWARE, VOL. 12, NO. 3, SEPTEMBER 1986, */
    /*                 PP 265-273. */

    /* ***ROUTINES CALLED  ZBINU,ZABS,ZDIV,ZSQRT,D1MACH,I1MACH */
    /* ***END PROLOGUE  ZBIRY */
    /*     COMPLEX BI,CONE,CSQ,CY,S1,S2,TRM1,TRM2,Z,ZTA,Z3 */
    /* ***FIRST EXECUTABLE STATEMENT  ZBIRY */
    *ierr = 0;
    nz    = 0;
    if(*id < 0 || *id > 1)
    {
        *ierr = 1;
    }
    if(*kode < 1 || *kode > 2)
    {
        *ierr = 1;
    }
    if(*ierr != 0)
    {
        return 0;
    }
    az = zabs_(zr, zi);
    /* Computing MAX */
    d__1 = d1mach_(&c__4);
    tol  = max(d__1, 1e-18);
    fid  = (double)((float)(*id));
    if(az > (float)1.)
    {
        goto L70;
    }
    /* ----------------------------------------------------------------------- */
    /*     POWER SERIES FOR CABS(Z).LE.1. */
    /* ----------------------------------------------------------------------- */
    s1r = coner;
    s1i = conei;
    s2r = coner;
    s2i = conei;
    if(az < tol)
    {
        goto L130;
    }
    aa = az * az;
    if(aa < tol / az)
    {
        goto L40;
    }
    trm1r = coner;
    trm1i = conei;
    trm2r = coner;
    trm2i = conei;
    atrm  = 1.;
    str   = *zr * *zr - *zi * *zi;
    sti   = *zr * *zi + *zi * *zr;
    z3r   = str * *zr - sti * *zi;
    z3i   = str * *zi + sti * *zr;
    az3   = az * aa;
    ak    = fid + 2.;
    bk    = 3. - fid - fid;
    ck    = 4. - fid;
    dk    = fid + 3. + fid;
    d1    = ak * dk;
    d2    = bk * ck;
    ad    = std::min(d1, d2);
    ak    = fid * 9. + 24.;
    bk    = 30. - fid * 9.;
    for(k = 1; k <= 25; ++k)
    {
        str   = (trm1r * z3r - trm1i * z3i) / d1;
        trm1i = (trm1r * z3i + trm1i * z3r) / d1;
        trm1r = str;
        s1r += trm1r;
        s1i += trm1i;
        str   = (trm2r * z3r - trm2i * z3i) / d2;
        trm2i = (trm2r * z3i + trm2i * z3r) / d2;
        trm2r = str;
        s2r += trm2r;
        s2i += trm2i;
        atrm = atrm * az3 / ad;
        d1 += ak;
        d2 += bk;
        ad = std::min(d1, d2);
        if(atrm < tol * ad)
        {
            goto L40;
        }
        ak += 18.;
        bk += 18.;
        /* L30: */
    }
L40:
    if(*id == 1)
    {
        goto L50;
    }
    *bir = c1 * s1r + c2 * (*zr * s2r - *zi * s2i);
    *bii = c1 * s1i + c2 * (*zr * s2i + *zi * s2r);
    if(*kode == 1)
    {
        return 0;
    }
    zsqrt_(zr, zi, &str, &sti);
    ztar = tth * (*zr * str - *zi * sti);
    ztai = tth * (*zr * sti + *zi * str);
    aa   = ztar;
    aa   = -abs(aa);
    eaa  = exp(aa);
    *bir *= eaa;
    *bii *= eaa;
    return 0;
L50:
    *bir = s2r * c2;
    *bii = s2i * c2;
    if(az <= tol)
    {
        goto L60;
    }
    cc  = c1 / (fid + 1.);
    str = s1r * *zr - s1i * *zi;
    sti = s1r * *zi + s1i * *zr;
    *bir += cc * (str * *zr - sti * *zi);
    *bii += cc * (str * *zi + sti * *zr);
L60:
    if(*kode == 1)
    {
        return 0;
    }
    zsqrt_(zr, zi, &str, &sti);
    ztar = tth * (*zr * str - *zi * sti);
    ztai = tth * (*zr * sti + *zi * str);
    aa   = ztar;
    aa   = -abs(aa);
    eaa  = exp(aa);
    *bir *= eaa;
    *bii *= eaa;
    return 0;
/* ----------------------------------------------------------------------- */
/*     CASE FOR CABS(Z).GT.1.0 */
/* ----------------------------------------------------------------------- */
L70:
    fnu = (fid + 1.) / 3.;
    /* ----------------------------------------------------------------------- */
    /*     SET PARAMETERS RELATED TO MACHINE CONSTANTS. */
    /*     TOL IS THE APPROXIMATE UNIT ROUNDOFF LIMITED TO 1.0E-18. */
    /*     ELIM IS THE APPROXIMATE EXPONENTIAL OVER- AND UNDERFLOW LIMIT. */
    /*     EXP(-ELIM).LT.EXP(-ALIM)=EXP(-ELIM)/TOL    AND */
    /*     EXP(ELIM).GT.EXP(ALIM)=EXP(ELIM)*TOL       ARE INTERVALS NEAR */
    /*     UNDERFLOW AND OVERFLOW LIMITS WHERE SCALED ARITHMETIC IS DONE. */
    /*     RL IS THE LOWER BOUNDARY OF THE ASYMPTOTIC EXPANSION FOR LARGE Z. */
    /*     DIG = NUMBER OF BASE 10 DIGITS IN TOL = 10**(-DIG). */
    /*     FNUL IS THE LOWER BOUNDARY OF THE ASYMPTOTIC SERIES FOR LARGE FNU. */
    /* ----------------------------------------------------------------------- */
    k1   = i1mach_(&c__15);
    k2   = i1mach_(&c__16);
    r1m5 = d1mach_(&c__5);
    /* Computing MIN */
    i__1 = abs(k1), i__2 = abs(k2);
    k    = std::min(i__1, i__2);
    elim = ((double)((float)k) * r1m5 - 3.) * 2.303;
    k1   = i1mach_(&c__14) - 1;
    aa   = r1m5 * (double)((float)k1);
    dig  = std::min(aa, 18.);
    aa *= 2.303;
    /* Computing MAX */
    d__1 = -aa;
    alim = elim + max(d__1, -41.45);
    rl   = dig * 1.2 + 3.;
    fnul = (dig - 3.) * 6. + 10.;
    /* ----------------------------------------------------------------------- */
    /*     TEST FOR RANGE */
    /* ----------------------------------------------------------------------- */
    aa = .5 / tol;
    bb = (double)((float)i1mach_(&c__9)) * .5;
    aa = std::min(aa, bb);
    aa = pow_dd(&aa, &tth);
    if(az > aa)
    {
        goto L260;
    }
    aa = sqrt(aa);
    if(az > aa)
    {
        *ierr = 3;
    }
    zsqrt_(zr, zi, &csqr, &csqi);
    ztar = tth * (*zr * csqr - *zi * csqi);
    ztai = tth * (*zr * csqi + *zi * csqr);
    /* ----------------------------------------------------------------------- */
    /*     RE(ZTA).LE.0 WHEN RE(Z).LT.0, ESPECIALLY WHEN IM(Z) IS SMALL */
    /* ----------------------------------------------------------------------- */
    sfac = 1.;
    ak   = ztai;
    if(*zr >= 0.)
    {
        goto L80;
    }
    bk   = ztar;
    ck   = -abs(bk);
    ztar = ck;
    ztai = ak;
L80:
    if(*zi != 0. || *zr > 0.)
    {
        goto L90;
    }
    ztar = 0.;
    ztai = ak;
L90:
    aa = ztar;
    if(*kode == 2)
    {
        goto L100;
    }
    /* ----------------------------------------------------------------------- */
    /*     OVERFLOW TEST */
    /* ----------------------------------------------------------------------- */
    bb = abs(aa);
    if(bb < alim)
    {
        goto L100;
    }
    bb += log(az) * .25;
    sfac = tol;
    if(bb > elim)
    {
        goto L190;
    }
L100:
    fmr = 0.;
    if(aa >= 0. && *zr > 0.)
    {
        goto L110;
    }
    fmr = pi;
    if(*zi < 0.)
    {
        fmr = -pi;
    }
    ztar = -ztar;
    ztai = -ztai;
L110:
    /* ----------------------------------------------------------------------- */
    /*     AA=FACTOR FOR ANALYTIC CONTINUATION OF I(FNU,ZTA) */
    /*     KODE=2 RETURNS EXP(-ABS(XZTA))*I(FNU,ZTA) FROM ZBESI */
    /* ----------------------------------------------------------------------- */
    zbinu_(&ztar, &ztai, &fnu, kode, &c__1, cyr, cyi, &nz, &rl, &fnul, &tol, &elim, &alim);
    if(nz < 0)
    {
        goto L200;
    }
    aa  = fmr * fnu;
    z3r = sfac;
    str = cos(aa);
    sti = sin(aa);
    s1r = (str * cyr[0] - sti * cyi[0]) * z3r;
    s1i = (str * cyi[0] + sti * cyr[0]) * z3r;
    fnu = (2. - fid) / 3.;
    zbinu_(&ztar, &ztai, &fnu, kode, &c__2, cyr, cyi, &nz, &rl, &fnul, &tol, &elim, &alim);
    cyr[0] *= z3r;
    cyi[0] *= z3r;
    cyr[1] *= z3r;
    cyi[1] *= z3r;
    /* ----------------------------------------------------------------------- */
    /*     BACKWARD RECUR ONE STEP FOR ORDERS -1/3 OR -2/3 */
    /* ----------------------------------------------------------------------- */
    zdiv_(cyr, cyi, &ztar, &ztai, &str, &sti);
    s2r = (fnu + fnu) * str + cyr[1];
    s2i = (fnu + fnu) * sti + cyi[1];
    aa  = fmr * (fnu - 1.);
    str = cos(aa);
    sti = sin(aa);
    s1r = coef * (s1r + s2r * str - s2i * sti);
    s1i = coef * (s1i + s2r * sti + s2i * str);
    if(*id == 1)
    {
        goto L120;
    }
    str  = csqr * s1r - csqi * s1i;
    s1i  = csqr * s1i + csqi * s1r;
    s1r  = str;
    *bir = s1r / sfac;
    *bii = s1i / sfac;
    return 0;
L120:
    str  = *zr * s1r - *zi * s1i;
    s1i  = *zr * s1i + *zi * s1r;
    s1r  = str;
    *bir = s1r / sfac;
    *bii = s1i / sfac;
    return 0;
L130:
    aa   = c1 * (1. - fid) + fid * c2;
    *bir = aa;
    *bii = 0.;
    return 0;
L190:
    *ierr = 2;
    nz    = 0;
    return 0;
L200:
    if(nz == -1)
    {
        goto L190;
    }
    nz    = 0;
    *ierr = 5;
    return 0;
L260:
    *ierr = 4;
    nz    = 0;
    return 0;
} /* zbiry_ */

 int zbknu_(double* zr,
                            double* zi,
                            double* fnu,
                            int32*    kode,
                            int32*    n,
                            double* yr,
                            double* yi,
                            int32*    nz,
                            double* tol,
                            double* elim,
                            double* alim)
{
    /* Initialized data */

    static int32    kmax   = 30;
    static double czeror = 0.;
    static double czeroi = 0.;
    static double coner  = 1.;
    static double conei  = 0.;
    static double ctwor  = 2.;
    static double r1     = 2.;
    static double dpi    = 3.14159265358979324;
    static double rthpi  = 1.25331413731550025;
    static double spi    = 1.90985931710274403;
    static double hpi    = 1.57079632679489662;
    static double fpi    = 1.89769999331517738;
    static double tth    = .666666666666666666;
    static double cc[8]  = {.577215664901532861,
                               -.0420026350340952355,
                               -.0421977345555443367,
                               .00721894324666309954,
                               -2.15241674114950973e-4,
                               -2.01348547807882387e-5,
                               1.13302723198169588e-6,
                               6.11609510448141582e-9};

    /* System generated locals */
    int32    i__1;
    double d__1;

    
    int32    i__, j, k;
    double s, a1, a2, g1, g2, t1, t2, aa, bb, fc, ak, bk;
    int32    ic;
    double fi, fk, as;
    int32    kk;
    double fr, pi, qi, tm, pr, qr;
    int32    nw;
    double p1i, p2i, s1i, s2i, p2m, p1r, p2r, s1r, s2r, cbi, cbr, cki, caz, csi, ckr, fhs, fks, rak, czi, dnu, csr, elm, zdi, bry[3], pti, czr, sti, zdr, cyr[2], rzi, ptr,
        cyi[2];
    int32                     inu;
    double                  str, rzr, dnu2, cchi, cchr, alas, cshi;
    int32                     inub, idum;
    
    double                  cshr, fmui, rcaz, csrr[3], cssr[3], fmur;
    
    double                  smui;
    double                  smur;
    int32                     iflag, kflag;
    double                  coefi;
    int32                     koded;
    double                  ascle, coefr, helim, celmr, csclr, crscr;
    double                  etest;

    /* *********************************************************************72 */

    /* c ZBKNU computes the K Bessel function in the right half Z plane. */

    /* ***BEGIN PROLOGUE  ZBKNU */
    /* ***REFER TO  ZBESI,ZBESK,ZAIRY,ZBESH */

    /*     ZBKNU COMPUTES THE K BESSEL FUNCTION IN THE RIGHT HALF Z PLANE. */

    /* ***ROUTINES CALLED  DGAMLN,I1MACH,D1MACH,ZKSCL,ZSHCH,ZUCHK,ZABS,ZDIV, */
    /*                    ZEXP,ZLOG,ZMLT,ZSQRT */
    /* ***END PROLOGUE  ZBKNU */

    /*     COMPLEX Z,Y,A,B,RZ,SMU,FU,FMU,F,FLRZ,CZ,S1,S2,CSH,CCH */
    /*     COMPLEX CK,P,Q,COEF,P1,P2,CBK,PT,CZERO,CONE,CTWO,ST,EZ,CS,DK */

    /* Parameter adjustments */
    --yi;
    --yr;

    /* Function Body */

    caz     = zabs_(zr, zi);
    csclr   = 1. / *tol;
    crscr   = *tol;
    cssr[0] = csclr;
    cssr[1] = 1.;
    cssr[2] = crscr;
    csrr[0] = crscr;
    csrr[1] = 1.;
    csrr[2] = csclr;
    bry[0]  = d1mach_(&c__1) * 1e3 / *tol;
    bry[1]  = 1. / bry[0];
    bry[2]  = d1mach_(&c__2);
    *nz     = 0;
    iflag   = 0;
    koded   = *kode;
    rcaz    = 1. / caz;
    str     = *zr * rcaz;
    sti     = -(*zi) * rcaz;
    rzr     = (str + str) * rcaz;
    rzi     = (sti + sti) * rcaz;
    inu     = (int32)(*fnu + .5);
    dnu     = *fnu - (double)((float)inu);
    if(abs(dnu) == .5)
    {
        goto L110;
    }
    dnu2 = 0.;
    if(abs(dnu) > *tol)
    {
        dnu2 = dnu * dnu;
    }
    if(caz > r1)
    {
        goto L110;
    }
    /* ----------------------------------------------------------------------- */
    /*     SERIES FOR CABS(Z).LE.R1 */
    /* ----------------------------------------------------------------------- */
    fc = 1.;
    zlog_(&rzr, &rzi, &smur, &smui, &idum);
    fmur = smur * dnu;
    fmui = smui * dnu;
    zshch_(&fmur, &fmui, &cshr, &cshi, &cchr, &cchi);
    if(dnu == 0.)
    {
        goto L10;
    }
    fc = dnu * dpi;
    fc /= sin(fc);
    smur = cshr / dnu;
    smui = cshi / dnu;
L10:
    a2 = dnu + 1.;
    /* ----------------------------------------------------------------------- */
    /*     GAM(1-Z)*GAM(1+Z)=PI*Z/SIN(PI*Z), T1=1/GAM(1-DNU), T2=1/GAM(1+DNU) */
    /* ----------------------------------------------------------------------- */
    t2 = exp(-dgamln_(&a2, &idum));
    t1 = 1. / (t2 * fc);
    if(abs(dnu) > .1)
    {
        goto L40;
    }
    /* ----------------------------------------------------------------------- */
    /*     SERIES FOR F0 TO RESOLVE INDETERMINACY FOR SMALL ABS(DNU) */
    /* ----------------------------------------------------------------------- */
    ak = 1.;
    s  = cc[0];
    for(k = 2; k <= 8; ++k)
    {
        ak *= dnu2;
        tm = cc[k - 1] * ak;
        s += tm;
        if(abs(tm) < *tol)
        {
            goto L30;
        }
        /* L20: */
    }
L30:
    g1 = -s;
    goto L50;
L40:
    g1 = (t1 - t2) / (dnu + dnu);
L50:
    g2 = (t1 + t2) * .5;
    fr = fc * (cchr * g1 + smur * g2);
    fi = fc * (cchi * g1 + smui * g2);
    zexp_(&fmur, &fmui, &str, &sti);
    pr = str * .5 / t2;
    pi = sti * .5 / t2;
    zdiv_(&c_b876, &c_b877, &str, &sti, &ptr, &pti);
    qr  = ptr / t1;
    qi  = pti / t1;
    s1r = fr;
    s1i = fi;
    s2r = pr;
    s2i = pi;
    ak  = 1.;
    a1  = 1.;
    ckr = coner;
    cki = conei;
    bk  = 1. - dnu2;
    if(inu > 0 || *n > 1)
    {
        goto L80;
    }
    /* ----------------------------------------------------------------------- */
    /*     GENERATE K(FNU,Z), 0.0D0 .LE. FNU .LT. 0.5D0 AND N=1 */
    /* ----------------------------------------------------------------------- */
    if(caz < *tol)
    {
        goto L70;
    }
    zmlt_(zr, zi, zr, zi, &czr, &czi);
    czr *= .25;
    czi *= .25;
    t1 = caz * .25 * caz;
L60:
    fr  = (fr * ak + pr + qr) / bk;
    fi  = (fi * ak + pi + qi) / bk;
    str = 1. / (ak - dnu);
    pr *= str;
    pi *= str;
    str = 1. / (ak + dnu);
    qr *= str;
    qi *= str;
    str = ckr * czr - cki * czi;
    rak = 1. / ak;
    cki = (ckr * czi + cki * czr) * rak;
    ckr = str * rak;
    s1r = ckr * fr - cki * fi + s1r;
    s1i = ckr * fi + cki * fr + s1i;
    a1  = a1 * t1 * rak;
    bk  = bk + ak + ak + 1.;
    ak += 1.;
    if(a1 > *tol)
    {
        goto L60;
    }
L70:
    yr[1] = s1r;
    yi[1] = s1i;
    if(koded == 1)
    {
        return 0;
    }
    zexp_(zr, zi, &str, &sti);
    zmlt_(&s1r, &s1i, &str, &sti, &yr[1], &yi[1]);
    return 0;
/* ----------------------------------------------------------------------- */
/*     GENERATE K(DNU,Z) AND K(DNU+1,Z) FOR FORWARD RECURRENCE */
/* ----------------------------------------------------------------------- */
L80:
    if(caz < *tol)
    {
        goto L100;
    }
    zmlt_(zr, zi, zr, zi, &czr, &czi);
    czr *= .25;
    czi *= .25;
    t1 = caz * .25 * caz;
L90:
    fr  = (fr * ak + pr + qr) / bk;
    fi  = (fi * ak + pi + qi) / bk;
    str = 1. / (ak - dnu);
    pr *= str;
    pi *= str;
    str = 1. / (ak + dnu);
    qr *= str;
    qi *= str;
    str = ckr * czr - cki * czi;
    rak = 1. / ak;
    cki = (ckr * czi + cki * czr) * rak;
    ckr = str * rak;
    s1r = ckr * fr - cki * fi + s1r;
    s1i = ckr * fi + cki * fr + s1i;
    str = pr - fr * ak;
    sti = pi - fi * ak;
    s2r = ckr * str - cki * sti + s2r;
    s2i = ckr * sti + cki * str + s2i;
    a1  = a1 * t1 * rak;
    bk  = bk + ak + ak + 1.;
    ak += 1.;
    if(a1 > *tol)
    {
        goto L90;
    }
L100:
    kflag = 2;
    a1    = *fnu + 1.;
    ak    = a1 * abs(smur);
    if(ak > *alim)
    {
        kflag = 3;
    }
    str = cssr[kflag - 1];
    p2r = s2r * str;
    p2i = s2i * str;
    zmlt_(&p2r, &p2i, &rzr, &rzi, &s2r, &s2i);
    s1r *= str;
    s1i *= str;
    if(koded == 1)
    {
        goto L210;
    }
    zexp_(zr, zi, &fr, &fi);
    zmlt_(&s1r, &s1i, &fr, &fi, &s1r, &s1i);
    zmlt_(&s2r, &s2i, &fr, &fi, &s2r, &s2i);
    goto L210;
/* ----------------------------------------------------------------------- */
/*     IFLAG=0 MEANS NO UNDERFLOW OCCURRED */
/*     IFLAG=1 MEANS AN UNDERFLOW OCCURRED- COMPUTATION PROCEEDS WITH */
/*     KODED=2 AND A TEST FOR ON SCALE VALUES IS MADE DURING FORWARD */
/*     RECURSION */
/* ----------------------------------------------------------------------- */
L110:
    zsqrt_(zr, zi, &str, &sti);
    zdiv_(&rthpi, &czeroi, &str, &sti, &coefr, &coefi);
    kflag = 2;
    if(koded == 2)
    {
        goto L120;
    }
    if(*zr > *alim)
    {
        goto L290;
    }
    /*     BLANK LINE */
    str = exp(-(*zr)) * cssr[kflag - 1];
    sti = -str * sin(*zi);
    str *= cos(*zi);
    zmlt_(&coefr, &coefi, &str, &sti, &coefr, &coefi);
L120:
    if(abs(dnu) == .5)
    {
        goto L300;
    }
    /* ----------------------------------------------------------------------- */
    /*     MILLER ALGORITHM FOR CABS(Z).GT.R1 */
    /* ----------------------------------------------------------------------- */
    ak = cos(dpi * dnu);
    ak = abs(ak);
    if(ak == czeror)
    {
        goto L300;
    }
    fhs = (d__1 = .25 - dnu2, abs(d__1));
    if(fhs == czeror)
    {
        goto L300;
    }
    /* ----------------------------------------------------------------------- */
    /*     COMPUTE R2=F(E). IF CABS(Z).GE.R2, USE FORWARD RECURRENCE TO */
    /*     DETERMINE THE BACKWARD INDEX K. R2=F(E) IS A STRAIGHT LINE ON */
    /*     12.LE.E.LE.60. E IS COMPUTED FROM 2**(-E)=B**(1-I1MACH(14))= */
    /*     TOL WHERE B IS THE BASE OF THE ARITHMETIC. */
    /* ----------------------------------------------------------------------- */
    t1 = (double)((float)(i1mach_(&c__14) - 1));
    t1 = t1 * d1mach_(&c__5) * 3.321928094;
    t1 = max(t1, 12.);
    t1 = std::min(t1, 60.);
    t2 = tth * t1 - 6.;
    if(*zr != 0.)
    {
        goto L130;
    }
    t1 = hpi;
    goto L140;
L130:
    t1 = atan(*zi / *zr);
    t1 = abs(t1);
L140:
    if(t2 > caz)
    {
        goto L170;
    }
    /* ----------------------------------------------------------------------- */
    /*     FORWARD RECURRENCE LOOP WHEN CABS(Z).GE.R2 */
    /* ----------------------------------------------------------------------- */
    etest = ak / (dpi * caz * *tol);
    fk    = coner;
    if(etest < coner)
    {
        goto L180;
    }
    fks  = ctwor;
    ckr  = caz + caz + ctwor;
    p1r  = czeror;
    p2r  = coner;
    i__1 = kmax;
    for(i__ = 1; i__ <= i__1; ++i__)
    {
        ak  = fhs / fks;
        cbr = ckr / (fk + coner);
        ptr = p2r;
        p2r = cbr * p2r - p1r * ak;
        p1r = ptr;
        ckr += ctwor;
        fks = fks + fk + fk + ctwor;
        fhs = fhs + fk + fk;
        fk += coner;
        str = abs(p2r) * fk;
        if(etest < str)
        {
            goto L160;
        }
        /* L150: */
    }
    goto L310;
L160:
    fk += spi * t1 * sqrt(t2 / caz);
    fhs = (d__1 = .25 - dnu2, abs(d__1));
    goto L180;
L170:
    /* ----------------------------------------------------------------------- */
    /*     COMPUTE BACKWARD INDEX K FOR CABS(Z).LT.R2 */
    /* ----------------------------------------------------------------------- */
    a2 = sqrt(caz);
    ak = fpi * ak / (*tol * sqrt(a2));
    aa = t1 * 3. / (caz + 1.);
    bb = t1 * 14.7 / (caz + 28.);
    ak = (log(ak) + caz * cos(aa) / (caz * .008 + 1.)) / cos(bb);
    fk = ak * .12125 * ak / caz + 1.5;
L180:
    /* ----------------------------------------------------------------------- */
    /*     BACKWARD RECURRENCE LOOP FOR MILLER ALGORITHM */
    /* ----------------------------------------------------------------------- */
    k    = (int32)((float)fk);
    fk   = (double)((float)k);
    fks  = fk * fk;
    p1r  = czeror;
    p1i  = czeroi;
    p2r  = *tol;
    p2i  = czeroi;
    csr  = p2r;
    csi  = p2i;
    i__1 = k;
    for(i__ = 1; i__ <= i__1; ++i__)
    {
        a1  = fks - fk;
        ak  = (fks + fk) / (a1 + fhs);
        rak = 2. / (fk + coner);
        cbr = (fk + *zr) * rak;
        cbi = *zi * rak;
        ptr = p2r;
        pti = p2i;
        p2r = (ptr * cbr - pti * cbi - p1r) * ak;
        p2i = (pti * cbr + ptr * cbi - p1i) * ak;
        p1r = ptr;
        p1i = pti;
        csr += p2r;
        csi += p2i;
        fks = a1 - fk + coner;
        fk -= coner;
        /* L190: */
    }
    /* ----------------------------------------------------------------------- */
    /*     COMPUTE (P2/CS)=(P2/CABS(CS))*(CONJG(CS)/CABS(CS)) FOR BETTER */
    /*     SCALING */
    /* ----------------------------------------------------------------------- */
    tm  = zabs_(&csr, &csi);
    ptr = 1. / tm;
    s1r = p2r * ptr;
    s1i = p2i * ptr;
    csr *= ptr;
    csi = -csi * ptr;
    zmlt_(&coefr, &coefi, &s1r, &s1i, &str, &sti);
    zmlt_(&str, &sti, &csr, &csi, &s1r, &s1i);
    if(inu > 0 || *n > 1)
    {
        goto L200;
    }
    zdr = *zr;
    zdi = *zi;
    if(iflag == 1)
    {
        goto L270;
    }
    goto L240;
L200:
    /* ----------------------------------------------------------------------- */
    /*     COMPUTE P1/P2=(P1/CABS(P2)*CONJG(P2)/CABS(P2) FOR SCALING */
    /* ----------------------------------------------------------------------- */
    tm  = zabs_(&p2r, &p2i);
    ptr = 1. / tm;
    p1r *= ptr;
    p1i *= ptr;
    p2r *= ptr;
    p2i = -p2i * ptr;
    zmlt_(&p1r, &p1i, &p2r, &p2i, &ptr, &pti);
    str = dnu + .5 - ptr;
    sti = -pti;
    zdiv_(&str, &sti, zr, zi, &str, &sti);
    str += 1.;
    zmlt_(&str, &sti, &s1r, &s1i, &s2r, &s2i);
/* ----------------------------------------------------------------------- */
/*     FORWARD RECURSION ON THE THREE TERM RECURSION WITH RELATION WITH */
/*     SCALING NEAR EXPONENT EXTREMES ON KFLAG=1 OR KFLAG=3 */
/* ----------------------------------------------------------------------- */
L210:
    str = dnu + 1.;
    ckr = str * rzr;
    cki = str * rzi;
    if(*n == 1)
    {
        --inu;
    }
    if(inu > 0)
    {
        goto L220;
    }
    if(*n > 1)
    {
        goto L215;
    }
    s1r = s2r;
    s1i = s2i;
L215:
    zdr = *zr;
    zdi = *zi;
    if(iflag == 1)
    {
        goto L270;
    }
    goto L240;
L220:
    inub = 1;
    if(iflag == 1)
    {
        goto L261;
    }
L225:
    p1r   = csrr[kflag - 1];
    ascle = bry[kflag - 1];
    i__1  = inu;
    for(i__ = inub; i__ <= i__1; ++i__)
    {
        str = s2r;
        sti = s2i;
        s2r = ckr * str - cki * sti + s1r;
        s2i = ckr * sti + cki * str + s1i;
        s1r = str;
        s1i = sti;
        ckr += rzr;
        cki += rzi;
        if(kflag >= 3)
        {
            goto L230;
        }
        p2r = s2r * p1r;
        p2i = s2i * p1r;
        str = abs(p2r);
        sti = abs(p2i);
        p2m = max(str, sti);
        if(p2m <= ascle)
        {
            goto L230;
        }
        ++kflag;
        ascle = bry[kflag - 1];
        s1r *= p1r;
        s1i *= p1r;
        s2r = p2r;
        s2i = p2i;
        str = cssr[kflag - 1];
        s1r *= str;
        s1i *= str;
        s2r *= str;
        s2i *= str;
        p1r = csrr[kflag - 1];
    L230:;
    }
    if(*n != 1)
    {
        goto L240;
    }
    s1r = s2r;
    s1i = s2i;
L240:
    str   = csrr[kflag - 1];
    yr[1] = s1r * str;
    yi[1] = s1i * str;
    if(*n == 1)
    {
        return 0;
    }
    yr[2] = s2r * str;
    yi[2] = s2i * str;
    if(*n == 2)
    {
        return 0;
    }
    kk = 2;
L250:
    ++kk;
    if(kk > *n)
    {
        return 0;
    }
    p1r   = csrr[kflag - 1];
    ascle = bry[kflag - 1];
    i__1  = *n;
    for(i__ = kk; i__ <= i__1; ++i__)
    {
        p2r = s2r;
        p2i = s2i;
        s2r = ckr * p2r - cki * p2i + s1r;
        s2i = cki * p2r + ckr * p2i + s1i;
        s1r = p2r;
        s1i = p2i;
        ckr += rzr;
        cki += rzi;
        p2r     = s2r * p1r;
        p2i     = s2i * p1r;
        yr[i__] = p2r;
        yi[i__] = p2i;
        if(kflag >= 3)
        {
            goto L260;
        }
        str = abs(p2r);
        sti = abs(p2i);
        p2m = max(str, sti);
        if(p2m <= ascle)
        {
            goto L260;
        }
        ++kflag;
        ascle = bry[kflag - 1];
        s1r *= p1r;
        s1i *= p1r;
        s2r = p2r;
        s2i = p2i;
        str = cssr[kflag - 1];
        s1r *= str;
        s1i *= str;
        s2r *= str;
        s2i *= str;
        p1r = csrr[kflag - 1];
    L260:;
    }
    return 0;
/* ----------------------------------------------------------------------- */
/*     IFLAG=1 CASES, FORWARD RECURRENCE ON SCALED VALUES ON UNDERFLOW */
/* ----------------------------------------------------------------------- */
L261:
    helim = *elim * .5;
    elm   = exp(-(*elim));
    celmr = elm;
    ascle = bry[0];
    zdr   = *zr;
    zdi   = *zi;
    ic    = -1;
    j     = 2;
    i__1  = inu;
    for(i__ = 1; i__ <= i__1; ++i__)
    {
        str = s2r;
        sti = s2i;
        s2r = str * ckr - sti * cki + s1r;
        s2i = sti * ckr + str * cki + s1i;
        s1r = str;
        s1i = sti;
        ckr += rzr;
        cki += rzi;
        as   = zabs_(&s2r, &s2i);
        alas = log(as);
        p2r  = -zdr + alas;
        if(p2r < -(*elim))
        {
            goto L263;
        }
        zlog_(&s2r, &s2i, &str, &sti, &idum);
        p2r = -zdr + str;
        p2i = -zdi + sti;
        p2m = exp(p2r) / *tol;
        p1r = p2m * cos(p2i);
        p1i = p2m * sin(p2i);
        zuchk_(&p1r, &p1i, &nw, &ascle, tol);
        if(nw != 0)
        {
            goto L263;
        }
        j          = 3 - j;
        cyr[j - 1] = p1r;
        cyi[j - 1] = p1i;
        if(ic == i__ - 1)
        {
            goto L264;
        }
        ic = i__;
        goto L262;
    L263:
        if(alas < helim)
        {
            goto L262;
        }
        zdr -= *elim;
        s1r *= celmr;
        s1i *= celmr;
        s2r *= celmr;
        s2i *= celmr;
    L262:;
    }
    if(*n != 1)
    {
        goto L270;
    }
    s1r = s2r;
    s1i = s2i;
    goto L270;
L264:
    kflag = 1;
    inub  = i__ + 1;
    s2r   = cyr[j - 1];
    s2i   = cyi[j - 1];
    j     = 3 - j;
    s1r   = cyr[j - 1];
    s1i   = cyi[j - 1];
    if(inub <= inu)
    {
        goto L225;
    }
    if(*n != 1)
    {
        goto L240;
    }
    s1r = s2r;
    s1i = s2i;
    goto L240;
L270:
    yr[1] = s1r;
    yi[1] = s1i;
    if(*n == 1)
    {
        goto L280;
    }
    yr[2] = s2r;
    yi[2] = s2i;
L280:
    ascle = bry[0];
    zkscl_(&zdr, &zdi, fnu, n, &yr[1], &yi[1], nz, &rzr, &rzi, &ascle, tol, elim);
    inu = *n - *nz;
    if(inu <= 0)
    {
        return 0;
    }
    kk     = *nz + 1;
    s1r    = yr[kk];
    s1i    = yi[kk];
    yr[kk] = s1r * csrr[0];
    yi[kk] = s1i * csrr[0];
    if(inu == 1)
    {
        return 0;
    }
    kk     = *nz + 2;
    s2r    = yr[kk];
    s2i    = yi[kk];
    yr[kk] = s2r * csrr[0];
    yi[kk] = s2i * csrr[0];
    if(inu == 2)
    {
        return 0;
    }
    t2    = *fnu + (double)((float)(kk - 1));
    ckr   = t2 * rzr;
    cki   = t2 * rzi;
    kflag = 1;
    goto L250;
L290:
    /* ----------------------------------------------------------------------- */
    /*     SCALE BY DEXP(Z), IFLAG = 1 CASES */
    /* ----------------------------------------------------------------------- */
    koded = 2;
    iflag = 1;
    kflag = 2;
    goto L120;
/* ----------------------------------------------------------------------- */
/*     FNU=HALF ODD INTEGER CASE, DNU=-0.5 */
/* ----------------------------------------------------------------------- */
L300:
    s1r = coefr;
    s1i = coefi;
    s2r = coefr;
    s2i = coefi;
    goto L210;

L310:
    *nz = -2;
    return 0;
} /* zbknu_ */

 int zbuni_(double* zr,
                            double* zi,
                            double* fnu,
                            int32*    kode,
                            int32*    n,
                            double* yr,
                            double* yi,
                            int32*    nz,
                            int32*    nui,
                            int32*    nlast,
                            double* fnul,
                            double* tol,
                            double* elim,
                            double* alim)
{
    /* System generated locals */
    int32 i__1;

    
    int32                     i__, k;
    double                  ax, ay;
    int32                     nl, nw;
    double                  c1i, c1m, c1r, s1i, s2i, s1r, s2r, cyi[2], gnu, raz, cyr[2], sti, bry[3], rzi, str, rzr, dfnu;
    
    double                  fnui;
   
    int32           iflag;
    double        ascle, csclr, cscrr;
    int32           iform;
    

    /* *********************************************************************72 */

    /* ***BEGIN PROLOGUE  ZBUNI */
    /* ***REFER TO  ZBESI,ZBESK */

    /*     ZBUNI COMPUTES THE I BESSEL FUNCTION FOR LARGE CABS(Z).GT. */
    /*     FNUL AND FNU+N-1.LT.FNUL. THE ORDER IS INCREASED FROM */
    /*     FNU+N-1 GREATER THAN FNUL BY ADDING NUI AND COMPUTING */
    /*     ACCORDING TO THE UNIFORM ASYMPTOTIC EXPANSION FOR I(FNU,Z) */
    /*     ON IFORM=1 AND THE EXPANSION FOR J(FNU,Z) ON IFORM=2 */

    /* ***ROUTINES CALLED  ZUNI1,ZUNI2,ZABS,D1MACH */
    /* ***END PROLOGUE  ZBUNI */
    /*     COMPLEX CSCL,CSCR,CY,RZ,ST,S1,S2,Y,Z */
    /* Parameter adjustments */
    --yi;
    --yr;

    /* Function Body */
    *nz   = 0;
    ax    = abs(*zr) * 1.7321;
    ay    = abs(*zi);
    iform = 1;
    if(ay > ax)
    {
        iform = 2;
    }
    if(*nui == 0)
    {
        goto L60;
    }
    fnui = (double)((float)(*nui));
    dfnu = *fnu + (double)((float)(*n - 1));
    gnu  = dfnu + fnui;
    if(iform == 2)
    {
        goto L10;
    }
    /* ----------------------------------------------------------------------- */
    /*     ASYMPTOTIC EXPANSION FOR I(FNU,Z) FOR LARGE FNU APPLIED IN */
    /*     -PI/3.LE.ARG(Z).LE.PI/3 */
    /* ----------------------------------------------------------------------- */
    zuni1_(zr, zi, &gnu, kode, &c__2, cyr, cyi, &nw, nlast, fnul, tol, elim, alim);
    goto L20;
L10:
    /* ----------------------------------------------------------------------- */
    /*     ASYMPTOTIC EXPANSION FOR J(FNU,Z*EXP(M*HPI)) FOR LARGE FNU */
    /*     APPLIED IN PI/3.LT.ABS(ARG(Z)).LE.PI/2 WHERE M=+I OR -I */
    /*     AND HPI=PI/2 */
    /* ----------------------------------------------------------------------- */
    zuni2_(zr, zi, &gnu, kode, &c__2, cyr, cyi, &nw, nlast, fnul, tol, elim, alim);
L20:
    if(nw < 0)
    {
        goto L50;
    }
    if(nw != 0)
    {
        goto L90;
    }
    str = zabs_(cyr, cyi);
    /* ---------------------------------------------------------------------- */
    /*     SCALE BACKWARD RECURRENCE, BRY(3) IS DEFINED BUT NEVER USED */
    /* ---------------------------------------------------------------------- */
    bry[0] = d1mach_(&c__1) * 1e3 / *tol;
    bry[1] = 1. / bry[0];
    bry[2] = bry[1];
    iflag  = 2;
    ascle  = bry[1];
    csclr  = 1.;
    if(str > bry[0])
    {
        goto L21;
    }
    iflag = 1;
    ascle = bry[0];
    csclr = 1. / *tol;
    goto L25;
L21:
    if(str < bry[1])
    {
        goto L25;
    }
    iflag = 3;
    ascle = bry[2];
    csclr = *tol;
L25:
    cscrr = 1. / csclr;
    s1r   = cyr[1] * csclr;
    s1i   = cyi[1] * csclr;
    s2r   = cyr[0] * csclr;
    s2i   = cyi[0] * csclr;
    raz   = 1. / zabs_(zr, zi);
    str   = *zr * raz;
    sti   = -(*zi) * raz;
    rzr   = (str + str) * raz;
    rzi   = (sti + sti) * raz;
    i__1  = *nui;
    for(i__ = 1; i__ <= i__1; ++i__)
    {
        str = s2r;
        sti = s2i;
        s2r = (dfnu + fnui) * (rzr * str - rzi * sti) + s1r;
        s2i = (dfnu + fnui) * (rzr * sti + rzi * str) + s1i;
        s1r = str;
        s1i = sti;
        fnui += -1.;
        if(iflag >= 3)
        {
            goto L30;
        }
        str = s2r * cscrr;
        sti = s2i * cscrr;
        c1r = abs(str);
        c1i = abs(sti);
        c1m = max(c1r, c1i);
        if(c1m <= ascle)
        {
            goto L30;
        }
        ++iflag;
        ascle = bry[iflag - 1];
        s1r *= cscrr;
        s1i *= cscrr;
        s2r = str;
        s2i = sti;
        csclr *= *tol;
        cscrr = 1. / csclr;
        s1r *= csclr;
        s1i *= csclr;
        s2r *= csclr;
        s2i *= csclr;
    L30:;
    }
    yr[*n] = s2r * cscrr;
    yi[*n] = s2i * cscrr;
    if(*n == 1)
    {
        return 0;
    }
    nl   = *n - 1;
    fnui = (double)((float)nl);
    k    = nl;
    i__1 = nl;
    for(i__ = 1; i__ <= i__1; ++i__)
    {
        str   = s2r;
        sti   = s2i;
        s2r   = (*fnu + fnui) * (rzr * str - rzi * sti) + s1r;
        s2i   = (*fnu + fnui) * (rzr * sti + rzi * str) + s1i;
        s1r   = str;
        s1i   = sti;
        str   = s2r * cscrr;
        sti   = s2i * cscrr;
        yr[k] = str;
        yi[k] = sti;
        fnui += -1.;
        --k;
        if(iflag >= 3)
        {
            goto L40;
        }
        c1r = abs(str);
        c1i = abs(sti);
        c1m = max(c1r, c1i);
        if(c1m <= ascle)
        {
            goto L40;
        }
        ++iflag;
        ascle = bry[iflag - 1];
        s1r *= cscrr;
        s1i *= cscrr;
        s2r = str;
        s2i = sti;
        csclr *= *tol;
        cscrr = 1. / csclr;
        s1r *= csclr;
        s1i *= csclr;
        s2r *= csclr;
        s2i *= csclr;
    L40:;
    }
    return 0;
L50:
    *nz = -1;
    if(nw == -2)
    {
        *nz = -2;
    }
    return 0;
L60:
    if(iform == 2)
    {
        goto L70;
    }
    /* ----------------------------------------------------------------------- */
    /*     ASYMPTOTIC EXPANSION FOR I(FNU,Z) FOR LARGE FNU APPLIED IN */
    /*     -PI/3.LE.ARG(Z).LE.PI/3 */
    /* ----------------------------------------------------------------------- */
    zuni1_(zr, zi, fnu, kode, n, &yr[1], &yi[1], &nw, nlast, fnul, tol, elim, alim);
    goto L80;
L70:
    /* ----------------------------------------------------------------------- */
    /*     ASYMPTOTIC EXPANSION FOR J(FNU,Z*EXP(M*HPI)) FOR LARGE FNU */
    /*     APPLIED IN PI/3.LT.ABS(ARG(Z)).LE.PI/2 WHERE M=+I OR -I */
    /*     AND HPI=PI/2 */
    /* ----------------------------------------------------------------------- */
    zuni2_(zr, zi, fnu, kode, n, &yr[1], &yi[1], &nw, nlast, fnul, tol, elim, alim);
L80:
    if(nw < 0)
    {
        goto L50;
    }
    *nz = nw;
    return 0;
L90:
    *nlast = *n;
    return 0;
} /* zbuni_ */

 int zbunk_(double* zr,
                            double* zi,
                            double* fnu,
                            int32*    kode,
                            int32*    mr,
                            int32*    n,
                            double* yr,
                            double* yi,
                            int32*    nz,
                            double* tol,
                            double* elim,
                            double* alim)
{
    double                  ax, ay;

    /* *********************************************************************72 */

    /* ***BEGIN PROLOGUE  ZBUNK */
    /* ***REFER TO  ZBESK,ZBESH */

    /*     ZBUNK COMPUTES THE K BESSEL FUNCTION FOR FNU.GT.FNUL. */
    /*     ACCORDING TO THE UNIFORM ASYMPTOTIC EXPANSION FOR K(FNU,Z) */
    /*     IN ZUNK1 AND THE EXPANSION FOR H(2,FNU,Z) IN ZUNK2 */

    /* ***ROUTINES CALLED  ZUNK1,ZUNK2 */
    /* ***END PROLOGUE  ZBUNK */
    /*     COMPLEX Y,Z */
    /* Parameter adjustments */
    --yi;
    --yr;

    /* Function Body */
    *nz = 0;
    ax  = abs(*zr) * 1.7321;
    ay  = abs(*zi);
    if(ay > ax)
    {
        goto L10;
    }
    /* ----------------------------------------------------------------------- */
    /*     ASYMPTOTIC EXPANSION FOR K(FNU,Z) FOR LARGE FNU APPLIED IN */
    /*     -PI/3.LE.ARG(Z).LE.PI/3 */
    /* ----------------------------------------------------------------------- */
    zunk1_(zr, zi, fnu, kode, mr, n, &yr[1], &yi[1], nz, tol, elim, alim);
    goto L20;
L10:
    /* ----------------------------------------------------------------------- */
    /*     ASYMPTOTIC EXPANSION FOR H(2,FNU,Z*EXP(M*HPI)) FOR LARGE FNU */
    /*     APPLIED IN PI/3.LT.ABS(ARG(Z)).LE.PI/2 WHERE M=+I OR -I */
    /*     AND HPI=PI/2 */
    /* ----------------------------------------------------------------------- */
    zunk2_(zr, zi, fnu, kode, mr, n, &yr[1], &yi[1], nz, tol, elim, alim);
L20:
    return 0;
} /* zbunk_ */

 int zdiv_(double* ar, double* ai, double* br, double* bi, double* cr, double* ci)
{
    double        ca, cb, cc, cd, bm;

    /* *********************************************************************72 */

    /* c ZDIV carries out double precision System::Complex<float> division. */

    /* ***BEGIN PROLOGUE  ZDIV */
    /* ***REFER TO  ZBESH,ZBESI,ZBESJ,ZBESK,ZBESY,ZAIRY,ZBIRY */

    /*     DOUBLE PRECISION COMPLEX DIVIDE C=A/B. */

    /* ***ROUTINES CALLED  ZABS */
    /* ***END PROLOGUE  ZDIV */
    bm  = 1. / zabs_(br, bi);
    cc  = *br * bm;
    cd  = *bi * bm;
    ca  = (*ar * cc + *ai * cd) * bm;
    cb  = (*ai * cc - *ar * cd) * bm;
    *cr = ca;
    *ci = cb;
    return 0;
} /* zdiv_ */

 int zexp_(double* ar, double* ai, double* br, double* bi)
{
   

    
    double ca, cb, zm;

    /* *********************************************************************72 */

    /* c ZEXP carries out double precision System::Complex<float> exponentiation. */

    /* ***BEGIN PROLOGUE  ZEXP */
    /* ***REFER TO  ZBESH,ZBESI,ZBESJ,ZBESK,ZBESY,ZAIRY,ZBIRY */

    /*     DOUBLE PRECISION COMPLEX EXPONENTIAL FUNCTION B=EXP(A) */

    /* ***ROUTINES CALLED  (NONE) */
    /* ***END PROLOGUE  ZEXP */
    zm  = exp(*ar);
    ca  = zm * cos(*ai);
    cb  = zm * sin(*ai);
    *br = ca;
    *bi = cb;
    return 0;
} /* zexp_ */

 int zkscl_(double* zrr,
                            double* zri,
                            double* fnu,
                            int32*    n,
                            double* yr,
                            double* yi,
                            int32*    nz,
                            double* rzr,
                            double* rzi,
                            double* ascle,
                            double* tol,
                            double* elim)
{
    /* Initialized data */

    static double zeror = 0.;
    static double zeroi = 0.;

    /* System generated locals */
    int32 i__1;

 

    
    int32                     i__, ic;
    double                  as, fn;
    int32                     kk, nn, nw;
    double                  s1i, s2i, s1r, s2r, acs, cki, elm, csi, ckr, cyi[2], zdi, csr, cyr[2], zdr, str, alas;
    int32                     idum;
    
    double                  helim, celmr;

    /* *********************************************************************72 */

    /* ***BEGIN PROLOGUE  ZKSCL */
    /* ***REFER TO  ZBESK */

    /*     SET K FUNCTIONS TO ZERO ON UNDERFLOW, CONTINUE RECURRENCE */
    /*     ON SCALED FUNCTIONS UNTIL TWO MEMBERS COME ON SCALE, THEN */
    /*     RETURN WITH MIN(NZ+2,N) VALUES SCALED BY 1/TOL. */

    /* ***ROUTINES CALLED  ZUCHK,ZABS,ZLOG */
    /* ***END PROLOGUE  ZKSCL */
    /*     COMPLEX CK,CS,CY,CZERO,RZ,S1,S2,Y,ZR,ZD,CELM */
    /* Parameter adjustments */
    --yi;
    --yr;

    /* Function Body */

    *nz  = 0;
    ic   = 0;
    nn   = std::min(2L, *n);
    i__1 = nn;
    for(i__ = 1; i__ <= i__1; ++i__)
    {
        s1r          = yr[i__];
        s1i          = yi[i__];
        cyr[i__ - 1] = s1r;
        cyi[i__ - 1] = s1i;
        as           = zabs_(&s1r, &s1i);
        acs          = -(*zrr) + log(as);
        ++(*nz);
        yr[i__] = zeror;
        yi[i__] = zeroi;
        if(acs < -(*elim))
        {
            goto L10;
        }
        zlog_(&s1r, &s1i, &csr, &csi, &idum);
        csr -= *zrr;
        csi -= *zri;
        str = exp(csr) / *tol;
        csr = str * cos(csi);
        csi = str * sin(csi);
        zuchk_(&csr, &csi, &nw, ascle, tol);
        if(nw != 0)
        {
            goto L10;
        }
        yr[i__] = csr;
        yi[i__] = csi;
        ic      = i__;
        --(*nz);
    L10:;
    }
    if(*n == 1)
    {
        return 0;
    }
    if(ic > 1)
    {
        goto L20;
    }
    yr[1] = zeror;
    yi[1] = zeroi;
    *nz   = 2;
L20:
    if(*n == 2)
    {
        return 0;
    }
    if(*nz == 0)
    {
        return 0;
    }
    fn    = *fnu + 1.;
    ckr   = fn * *rzr;
    cki   = fn * *rzi;
    s1r   = cyr[0];
    s1i   = cyi[0];
    s2r   = cyr[1];
    s2i   = cyi[1];
    helim = *elim * .5;
    elm   = exp(-(*elim));
    celmr = elm;
    zdr   = *zrr;
    zdi   = *zri;

    /*     FIND TWO CONSECUTIVE Y VALUES ON SCALE. SCALE RECURRENCE IF */
    /*     S2 GETS LARGER THAN EXP(ELIM/2) */

    i__1 = *n;
    for(i__ = 3; i__ <= i__1; ++i__)
    {
        kk  = i__;
        csr = s2r;
        csi = s2i;
        s2r = ckr * csr - cki * csi + s1r;
        s2i = cki * csr + ckr * csi + s1i;
        s1r = csr;
        s1i = csi;
        ckr += *rzr;
        cki += *rzi;
        as   = zabs_(&s2r, &s2i);
        alas = log(as);
        acs  = -zdr + alas;
        ++(*nz);
        yr[i__] = zeror;
        yi[i__] = zeroi;
        if(acs < -(*elim))
        {
            goto L25;
        }
        zlog_(&s2r, &s2i, &csr, &csi, &idum);
        csr -= zdr;
        csi -= zdi;
        str = exp(csr) / *tol;
        csr = str * cos(csi);
        csi = str * sin(csi);
        zuchk_(&csr, &csi, &nw, ascle, tol);
        if(nw != 0)
        {
            goto L25;
        }
        yr[i__] = csr;
        yi[i__] = csi;
        --(*nz);
        if(ic == kk - 1)
        {
            goto L40;
        }
        ic = kk;
        goto L30;
    L25:
        if(alas < helim)
        {
            goto L30;
        }
        zdr -= *elim;
        s1r *= celmr;
        s1i *= celmr;
        s2r *= celmr;
        s2i *= celmr;
    L30:;
    }
    *nz = *n;
    if(ic == *n)
    {
        *nz = *n - 1;
    }
    goto L45;
L40:
    *nz = kk - 2;
L45:
    i__1 = *nz;
    for(i__ = 1; i__ <= i__1; ++i__)
    {
        yr[i__] = zeror;
        yi[i__] = zeroi;
        /* L50: */
    }
    return 0;
} /* zkscl_ */

 int zlog_(double* ar, double* ai, double* br, double* bi, int32* ierr)
{
    /* Initialized data */

    static double dpi  = 3.141592653589793238462643383;
    static double dhpi = 1.570796326794896619231321696;


    
    double        zm;
    double        dtheta;

    /* *********************************************************************72 */

    /* c ZLOG carries out double precision System::Complex<float> logarithms */

    /* ***BEGIN PROLOGUE  ZLOG */
    /* ***REFER TO  ZBESH,ZBESI,ZBESJ,ZBESK,ZBESY,ZAIRY,ZBIRY */

    /*     DOUBLE PRECISION COMPLEX LOGARITHM B=CLOG(A) */
    /*     IERR=0,NORMAL RETURN      IERR=1, Z=CMPLX(0.0,0.0) */
    /* ***ROUTINES CALLED  ZABS */
    /* ***END PROLOGUE  ZLOG */

    *ierr = 0;
    if(*ar == 0.)
    {
        goto L10;
    }
    if(*ai == 0.)
    {
        goto L20;
    }
    dtheta = atan(*ai / *ar);
    if(dtheta <= 0.)
    {
        goto L40;
    }
    if(*ar < 0.)
    {
        dtheta -= dpi;
    }
    goto L50;
L10:
    if(*ai == 0.)
    {
        goto L60;
    }
    *bi = dhpi;
    *br = log((abs(*ai)));
    if(*ai < 0.)
    {
        *bi = -(*bi);
    }
    return 0;
L20:
    if(*ar > 0.)
    {
        goto L30;
    }
    *br = log((abs(*ar)));
    *bi = dpi;
    return 0;
L30:
    *br = log(*ar);
    *bi = 0.;
    return 0;
L40:
    if(*ar < 0.)
    {
        dtheta += dpi;
    }
L50:
    zm  = zabs_(ar, ai);
    *br = log(zm);
    *bi = dtheta;
    return 0;
L60:
    *ierr = 1;
    return 0;
} /* zlog_ */

 int zmlri_(double* zr, double* zi, double* fnu, int32* kode, int32* n, double* yr, double* yi, int32* nz, double* tol)
{
    /* Initialized data */

    static double zeror = 0.;
    static double zeroi = 0.;
    static double coner = 1.;
    static double conei = 0.;

    /* System generated locals */
    int32    i__1, i__2;
    double d__1, d__2, d__3;

    

    
    int32                     i__, k, m;
    double                  ak, bk, ap, at;
    int32                     kk, km;
    double                  az, p1i, p2i, p1r, p2r, ack, cki, fnf, fkk, ckr;
    int32                     iaz;
    double                  rho;
    int32                     inu;
    double                  pti, raz, sti, rzi, ptr, str, tst, rzr, rho2, flam, fkap, scle, tfnf;
    int32                     idum;
    
    int32                     ifnu;
    double                  sumi, sumr;
    int32                     itime;
    double                  cnormi, cnormr;

    /* *********************************************************************72 */

    /* ***BEGIN PROLOGUE  ZMLRI */
    /* ***REFER TO  ZBESI,ZBESK */

    /*     ZMLRI COMPUTES THE I BESSEL FUNCTION FOR RE(Z).GE.0.0 BY THE */
    /*     MILLER ALGORITHM NORMALIZED BY A NEUMANN SERIES. */

    /* ***ROUTINES CALLED  DGAMLN,D1MACH,ZABS,ZEXP,ZLOG,ZMLT */
    /* ***END PROLOGUE  ZMLRI */
    /*     COMPLEX CK,CNORM,CONE,CTWO,CZERO,PT,P1,P2,RZ,SUM,Y,Z */
    /* Parameter adjustments */
    --yi;
    --yr;

    /* Function Body */
    scle = d1mach_(&c__1) / *tol;
    *nz  = 0;
    az   = zabs_(zr, zi);
    iaz  = (int32)((float)az);
    ifnu = (int32)((float)(*fnu));
    inu  = ifnu + *n - 1;
    at   = (double)((float)iaz) + 1.;
    raz  = 1. / az;
    str  = *zr * raz;
    sti  = -(*zi) * raz;
    ckr  = str * at * raz;
    cki  = sti * at * raz;
    rzr  = (str + str) * raz;
    rzi  = (sti + sti) * raz;
    p1r  = zeror;
    p1i  = zeroi;
    p2r  = coner;
    p2i  = conei;
    ack  = (at + 1.) * raz;
    rho  = ack + sqrt(ack * ack - 1.);
    rho2 = rho * rho;
    tst  = (rho2 + rho2) / ((rho2 - 1.) * (rho - 1.));
    tst /= *tol;
    /* ----------------------------------------------------------------------- */
    /*     COMPUTE RELATIVE TRUNCATION ERROR INDEX FOR SERIES */
    /* ----------------------------------------------------------------------- */
    ak = at;
    for(i__ = 1; i__ <= 80; ++i__)
    {
        ptr = p2r;
        pti = p2i;
        p2r = p1r - (ckr * ptr - cki * pti);
        p2i = p1i - (cki * ptr + ckr * pti);
        p1r = ptr;
        p1i = pti;
        ckr += rzr;
        cki += rzi;
        ap = zabs_(&p2r, &p2i);
        if(ap > tst * ak * ak)
        {
            goto L20;
        }
        ak += 1.;
        /* L10: */
    }
    goto L110;
L20:
    ++i__;
    k = 0;
    if(inu < iaz)
    {
        goto L40;
    }
    /* ----------------------------------------------------------------------- */
    /*     COMPUTE RELATIVE TRUNCATION ERROR FOR RATIOS */
    /* ----------------------------------------------------------------------- */
    p1r   = zeror;
    p1i   = zeroi;
    p2r   = coner;
    p2i   = conei;
    at    = (double)((float)inu) + 1.;
    str   = *zr * raz;
    sti   = -(*zi) * raz;
    ckr   = str * at * raz;
    cki   = sti * at * raz;
    ack   = at * raz;
    tst   = sqrt(ack / *tol);
    itime = 1;
    for(k = 1; k <= 80; ++k)
    {
        ptr = p2r;
        pti = p2i;
        p2r = p1r - (ckr * ptr - cki * pti);
        p2i = p1i - (ckr * pti + cki * ptr);
        p1r = ptr;
        p1i = pti;
        ckr += rzr;
        cki += rzi;
        ap = zabs_(&p2r, &p2i);
        if(ap < tst)
        {
            goto L30;
        }
        if(itime == 2)
        {
            goto L40;
        }
        ack  = zabs_(&ckr, &cki);
        flam = ack + sqrt(ack * ack - 1.);
        fkap = ap / zabs_(&p1r, &p1i);
        rho  = std::min(flam, fkap);
        tst *= sqrt(rho / (rho * rho - 1.));
        itime = 2;
    L30:;
    }
    goto L110;
L40:
    /* ----------------------------------------------------------------------- */
    /*     BACKWARD RECURRENCE AND SUM NORMALIZING RELATION */
    /* ----------------------------------------------------------------------- */
    ++k;
    /* Computing MAX */
    i__1 = i__ + iaz, i__2 = k + inu;
    kk  = max(i__1, i__2);
    fkk = (double)((float)kk);
    p1r = zeror;
    p1i = zeroi;
    /* ----------------------------------------------------------------------- */
    /*     SCALE P2 AND SUM BY SCLE */
    /* ----------------------------------------------------------------------- */
    p2r  = scle;
    p2i  = zeroi;
    fnf  = *fnu - (double)((float)ifnu);
    tfnf = fnf + fnf;
    d__1 = fkk + tfnf + 1.;
    d__2 = fkk + 1.;
    d__3 = tfnf + 1.;
    bk   = dgamln_(&d__1, &idum) - dgamln_(&d__2, &idum) - dgamln_(&d__3, &idum);
    bk   = exp(bk);
    sumr = zeror;
    sumi = zeroi;
    km   = kk - inu;
    i__1 = km;
    for(i__ = 1; i__ <= i__1; ++i__)
    {
        ptr = p2r;
        pti = p2i;
        p2r = p1r + (fkk + fnf) * (rzr * ptr - rzi * pti);
        p2i = p1i + (fkk + fnf) * (rzi * ptr + rzr * pti);
        p1r = ptr;
        p1i = pti;
        ak  = 1. - tfnf / (fkk + tfnf);
        ack = bk * ak;
        sumr += (ack + bk) * p1r;
        sumi += (ack + bk) * p1i;
        bk = ack;
        fkk += -1.;
        /* L50: */
    }
    yr[*n] = p2r;
    yi[*n] = p2i;
    if(*n == 1)
    {
        goto L70;
    }
    i__1 = *n;
    for(i__ = 2; i__ <= i__1; ++i__)
    {
        ptr = p2r;
        pti = p2i;
        p2r = p1r + (fkk + fnf) * (rzr * ptr - rzi * pti);
        p2i = p1i + (fkk + fnf) * (rzi * ptr + rzr * pti);
        p1r = ptr;
        p1i = pti;
        ak  = 1. - tfnf / (fkk + tfnf);
        ack = bk * ak;
        sumr += (ack + bk) * p1r;
        sumi += (ack + bk) * p1i;
        bk = ack;
        fkk += -1.;
        m     = *n - i__ + 1;
        yr[m] = p2r;
        yi[m] = p2i;
        /* L60: */
    }
L70:
    if(ifnu <= 0)
    {
        goto L90;
    }
    i__1 = ifnu;
    for(i__ = 1; i__ <= i__1; ++i__)
    {
        ptr = p2r;
        pti = p2i;
        p2r = p1r + (fkk + fnf) * (rzr * ptr - rzi * pti);
        p2i = p1i + (fkk + fnf) * (rzr * pti + rzi * ptr);
        p1r = ptr;
        p1i = pti;
        ak  = 1. - tfnf / (fkk + tfnf);
        ack = bk * ak;
        sumr += (ack + bk) * p1r;
        sumi += (ack + bk) * p1i;
        bk = ack;
        fkk += -1.;
        /* L80: */
    }
L90:
    ptr = *zr;
    pti = *zi;
    if(*kode == 2)
    {
        ptr = zeror;
    }
    zlog_(&rzr, &rzi, &str, &sti, &idum);
    p1r  = -fnf * str + ptr;
    p1i  = -fnf * sti + pti;
    d__1 = fnf + 1.;
    ap   = dgamln_(&d__1, &idum);
    ptr  = p1r - ap;
    pti  = p1i;
    /* ----------------------------------------------------------------------- */
    /*     THE DIVISION CEXP(PT)/(SUM+P2) IS ALTERED TO AVOID OVERFLOW */
    /*     IN THE DENOMINATOR BY SQUARING LARGE QUANTITIES */
    /* ----------------------------------------------------------------------- */
    p2r += sumr;
    p2i += sumi;
    ap  = zabs_(&p2r, &p2i);
    p1r = 1. / ap;
    zexp_(&ptr, &pti, &str, &sti);
    ckr = str * p1r;
    cki = sti * p1r;
    ptr = p2r * p1r;
    pti = -p2i * p1r;
    zmlt_(&ckr, &cki, &ptr, &pti, &cnormr, &cnormi);
    i__1 = *n;
    for(i__ = 1; i__ <= i__1; ++i__)
    {
        str     = yr[i__] * cnormr - yi[i__] * cnormi;
        yi[i__] = yr[i__] * cnormi + yi[i__] * cnormr;
        yr[i__] = str;
        /* L100: */
    }
    return 0;
L110:
    *nz = -2;
    return 0;
} /* zmlri_ */

 int zmlt_(double* ar, double* ai, double* br, double* bi, double* cr, double* ci)
{
    double ca, cb;

    /* *********************************************************************72 */

    /* c ZMLT carries out double precision System::Complex<float> multiplication. */

    /* ***BEGIN PROLOGUE  ZMLT */
    /* ***REFER TO  ZBESH,ZBESI,ZBESJ,ZBESK,ZBESY,ZAIRY,ZBIRY */

    /*     DOUBLE PRECISION COMPLEX MULTIPLY, C=A*B. */

    /* ***ROUTINES CALLED  (NONE) */
    /* ***END PROLOGUE  ZMLT */
    ca  = *ar * *br - *ai * *bi;
    cb  = *ar * *bi + *ai * *br;
    *cr = ca;
    *ci = cb;
    return 0;
} /* zmlt_ */

 int zrati_(double* zr, double* zi, double* fnu, int32* n, double* cyr, double* cyi, double* tol)
{
    /* Initialized data */

    static double czeror = 0.;
    static double czeroi = 0.;
    static double coner  = 1.;
    static double conei  = 0.;
    static double rt2    = 1.41421356237309505;

    /* System generated locals */
    int32    i__1;
    double d__1;

   
    

    
    int32                     i__, k;
    double                  ak;
    int32                     id, kk;
    double                  az, ap1, ap2, p1i, p2i, t1i, p1r, p2r, t1r, arg, rak, rho;
    int32                     inu;
    double                  pti, tti, rzi, ptr, ttr, rzr, rap1, flam, dfnu, fdnu;
    int32                     magz;
    
    int32                     idnu;
    double                  fnup;
    double                  test, test1, amagz;
    int32                     itime;
    double                  cdfnui, cdfnur;

    /* *********************************************************************72 */

    /* ***BEGIN PROLOGUE  ZRATI */
    /* ***REFER TO  ZBESI,ZBESK,ZBESH */

    /*     ZRATI COMPUTES RATIOS OF I BESSEL FUNCTIONS BY BACKWARD */
    /*     RECURRENCE.  THE STARTING INDEX IS DETERMINED BY FORWARD */
    /*     RECURRENCE AS DESCRIBED IN J. RES. OF NAT. BUR. OF STANDARDS-B, */
    /*     MATHEMATICAL SCIENCES, VOL 77B, P111-114, SEPTEMBER, 1973, */
    /*     BESSEL FUNCTIONS I AND J OF COMPLEX ARGUMENT AND INTEGER ORDER, */
    /*     BY D. J. SOOKNE. */

    /* ***ROUTINES CALLED  ZABS,ZDIV */
    /* ***END PROLOGUE  ZRATI */
    /*     COMPLEX Z,CY(1),CONE,CZERO,P1,P2,T1,RZ,PT,CDFNU */
    /* Parameter adjustments */
    --cyi;
    --cyr;

    /* Function Body */
    az    = zabs_(zr, zi);
    inu   = (int32)((float)(*fnu));
    idnu  = inu + *n - 1;
    magz  = (int32)((float)az);
    amagz = (double)((float)(magz + 1));
    fdnu  = (double)((float)idnu);
    fnup  = max(amagz, fdnu);
    id    = idnu - magz - 1;
    itime = 1;
    k     = 1;
    ptr   = 1. / az;
    rzr   = ptr * (*zr + *zr) * ptr;
    rzi   = -ptr * (*zi + *zi) * ptr;
    t1r   = rzr * fnup;
    t1i   = rzi * fnup;
    p2r   = -t1r;
    p2i   = -t1i;
    p1r   = coner;
    p1i   = conei;
    t1r += rzr;
    t1i += rzi;
    if(id > 0)
    {
        id = 0;
    }
    ap2 = zabs_(&p2r, &p2i);
    ap1 = zabs_(&p1r, &p1i);
    /* ----------------------------------------------------------------------- */
    /*     THE OVERFLOW TEST ON K(FNU+I-1,Z) BEFORE THE CALL TO CBKNU */
    /*     GUARANTEES THAT P2 IS ON SCALE. SCALE TEST1 AND ALL SUBSEQUENT */
    /*     P2 VALUES BY AP1 TO ENSURE THAT AN OVERFLOW DOES NOT OCCUR */
    /*     PREMATURELY. */
    /* ----------------------------------------------------------------------- */
    arg   = (ap2 + ap2) / (ap1 * *tol);
    test1 = sqrt(arg);
    test  = test1;
    rap1  = 1. / ap1;
    p1r *= rap1;
    p1i *= rap1;
    p2r *= rap1;
    p2i *= rap1;
    ap2 *= rap1;
L10:
    ++k;
    ap1 = ap2;
    ptr = p2r;
    pti = p2i;
    p2r = p1r - (t1r * ptr - t1i * pti);
    p2i = p1i - (t1r * pti + t1i * ptr);
    p1r = ptr;
    p1i = pti;
    t1r += rzr;
    t1i += rzi;
    ap2 = zabs_(&p2r, &p2i);
    if(ap1 <= test)
    {
        goto L10;
    }
    if(itime == 2)
    {
        goto L20;
    }
    ak   = zabs_(&t1r, &t1i) * .5;
    flam = ak + sqrt(ak * ak - 1.);
    /* Computing MIN */
    d__1  = ap2 / ap1;
    rho   = std::min(d__1, flam);
    test  = test1 * sqrt(rho / (rho * rho - 1.));
    itime = 2;
    goto L10;
L20:
    kk   = k + 1 - id;
    ak   = (double)((float)kk);
    t1r  = ak;
    t1i  = czeroi;
    dfnu = *fnu + (double)((float)(*n - 1));
    p1r  = 1. / ap2;
    p1i  = czeroi;
    p2r  = czeror;
    p2i  = czeroi;
    i__1 = kk;
    for(i__ = 1; i__ <= i__1; ++i__)
    {
        ptr  = p1r;
        pti  = p1i;
        rap1 = dfnu + t1r;
        ttr  = rzr * rap1;
        tti  = rzi * rap1;
        p1r  = ptr * ttr - pti * tti + p2r;
        p1i  = ptr * tti + pti * ttr + p2i;
        p2r  = ptr;
        p2i  = pti;
        t1r -= coner;
        /* L30: */
    }
    if(p1r != czeror || p1i != czeroi)
    {
        goto L40;
    }
    p1r = *tol;
    p1i = *tol;
L40:
    zdiv_(&p2r, &p2i, &p1r, &p1i, &cyr[*n], &cyi[*n]);
    if(*n == 1)
    {
        return 0;
    }
    k      = *n - 1;
    ak     = (double)((float)k);
    t1r    = ak;
    t1i    = czeroi;
    cdfnur = *fnu * rzr;
    cdfnui = *fnu * rzi;
    i__1   = *n;
    for(i__ = 2; i__ <= i__1; ++i__)
    {
        ptr = cdfnur + (t1r * rzr - t1i * rzi) + cyr[k + 1];
        pti = cdfnui + (t1r * rzi + t1i * rzr) + cyi[k + 1];
        ak  = zabs_(&ptr, &pti);
        if(ak != czeror)
        {
            goto L50;
        }
        ptr = *tol;
        pti = *tol;
        ak  = *tol * rt2;
    L50:
        rak    = coner / ak;
        cyr[k] = rak * ptr * rak;
        cyi[k] = -rak * pti * rak;
        t1r -= coner;
        --k;
        /* L60: */
    }
    return 0;
} /* zrati_ */

 int zs1s2_(double* zrr,
                            double* zri,
                            double* s1r,
                            double* s1i,
                            double* s2r,
                            double* s2i,
                            int32*    nz,
                            double* ascle,
                            double* alim,
                            int32*    iuf)
{
    /* Initialized data */

    static double zeror = 0.;
    static double zeroi = 0.;

   

    
    double                  aa, c1i, as1, as2, c1r, aln, s1di, s1dr;
    int32                     idum;
    

    /* *********************************************************************72 */

    /* ***BEGIN PROLOGUE  ZS1S2 */
    /* ***REFER TO  ZBESK,ZAIRY */

    /*     ZS1S2 TESTS FOR A POSSIBLE UNDERFLOW RESULTING FROM THE */
    /*     ADDITION OF THE I AND K FUNCTIONS IN THE ANALYTIC CON- */
    /*     TINUATION FORMULA WHERE S1=K FUNCTION AND S2=I FUNCTION. */
    /*     ON KODE=1 THE I AND K FUNCTIONS ARE DIFFERENT ORDERS OF */
    /*     MAGNITUDE, BUT FOR KODE=2 THEY CAN BE OF THE SAME ORDER */
    /*     OF MAGNITUDE AND THE MAXIMUM MUST BE AT LEAST ONE */
    /*     PRECISION ABOVE THE UNDERFLOW LIMIT. */

    /* ***ROUTINES CALLED  ZABS,ZEXP,ZLOG */
    /* ***END PROLOGUE  ZS1S2 */
    /*     COMPLEX CZERO,C1,S1,S1D,S2,ZR */
    *nz = 0;
    as1 = zabs_(s1r, s1i);
    as2 = zabs_(s2r, s2i);
    if(*s1r == 0. && *s1i == 0.)
    {
        goto L10;
    }
    if(as1 == 0.)
    {
        goto L10;
    }
    aln  = -(*zrr) - *zrr + log(as1);
    s1dr = *s1r;
    s1di = *s1i;
    *s1r = zeror;
    *s1i = zeroi;
    as1  = zeror;
    if(aln < -(*alim))
    {
        goto L10;
    }
    zlog_(&s1dr, &s1di, &c1r, &c1i, &idum);
    c1r = c1r - *zrr - *zrr;
    c1i = c1i - *zri - *zri;
    zexp_(&c1r, &c1i, s1r, s1i);
    as1 = zabs_(s1r, s1i);
    ++(*iuf);
L10:
    aa = max(as1, as2);
    if(aa > *ascle)
    {
        return 0;
    }
    *s1r = zeror;
    *s1i = zeroi;
    *s2r = zeror;
    *s2i = zeroi;
    *nz  = 1;
    *iuf = 0;
    return 0;
} /* zs1s2_ */

 int zseri_(double* zr,
                            double* zi,
                            double* fnu,
                            int32*    kode,
                            int32*    n,
                            double* yr,
                            double* yi,
                            int32*    nz,
                            double* tol,
                            double* elim,
                            double* alim)
{
    /* Initialized data */

    static double zeror = 0.;
    static double zeroi = 0.;
    static double coner = 1.;
    static double conei = 0.;

    /* System generated locals */
    int32 i__1;

    

    
    int32                     i__, k, l, m;
    double                  s, aa;
    int32                     ib;
    double                  ak;
    int32                     il;
    double                  az;
    int32                     nn;
    double                  wi[2], rs, ss;
    int32                     nw;
    double                  wr[2], s1i, s2i, s1r, s2r, cki, acz, arm, ckr, czi, hzi, raz, czr, sti, hzr, rzi, str, rzr, ak1i, ak1r, rtr1, dfnu;
    int32                     idum;
    
    double                  atol, fnup;
    int32                     iflag;
    double                  coefi, ascle, coefr, crscr;

    /* *********************************************************************72 */

    /* ***BEGIN PROLOGUE  ZSERI */
    /* ***REFER TO  ZBESI,ZBESK */

    /*     ZSERI COMPUTES THE I BESSEL FUNCTION FOR REAL(Z).GE.0.0 BY */
    /*     MEANS OF THE POWER SERIES FOR LARGE CABS(Z) IN THE */
    /*     REGION CABS(Z).LE.2*SQRT(FNU+1). NZ=0 IS A NORMAL RETURN. */
    /*     NZ.GT.0 MEANS THAT THE LAST NZ COMPONENTS WERE SET TO ZERO */
    /*     DUE TO UNDERFLOW. NZ.LT.0 MEANS UNDERFLOW OCCURRED, BUT THE */
    /*     CONDITION CABS(Z).LE.2*SQRT(FNU+1) WAS VIOLATED AND THE */
    /*     COMPUTATION MUST BE COMPLETED IN ANOTHER ROUTINE WITH N=N-ABS(NZ). */

    /* ***ROUTINES CALLED  DGAMLN,D1MACH,ZUCHK,ZABS,ZDIV,ZLOG,ZMLT */
    /* ***END PROLOGUE  ZSERI */
    /*     COMPLEX AK1,CK,COEF,CONE,CRSC,CSCL,CZ,CZERO,HZ,RZ,S1,S2,Y,Z */
    /* Parameter adjustments */
    --yi;
    --yr;

    /* Function Body */

    *nz = 0;
    az  = zabs_(zr, zi);
    if(az == 0.)
    {
        goto L160;
    }
    arm   = d1mach_(&c__1) * 1e3;
    rtr1  = sqrt(arm);
    crscr = 1.;
    iflag = 0;
    if(az < arm)
    {
        goto L150;
    }
    hzr = *zr * .5;
    hzi = *zi * .5;
    czr = zeror;
    czi = zeroi;
    if(az <= rtr1)
    {
        goto L10;
    }
    zmlt_(&hzr, &hzi, &hzr, &hzi, &czr, &czi);
L10:
    acz = zabs_(&czr, &czi);
    nn  = *n;
    zlog_(&hzr, &hzi, &ckr, &cki, &idum);
L20:
    dfnu = *fnu + (double)((float)(nn - 1));
    fnup = dfnu + 1.;
    /* ----------------------------------------------------------------------- */
    /*     UNDERFLOW TEST */
    /* ----------------------------------------------------------------------- */
    ak1r = ckr * dfnu;
    ak1i = cki * dfnu;
    ak   = dgamln_(&fnup, &idum);
    ak1r -= ak;
    if(*kode == 2)
    {
        ak1r -= *zr;
    }
    if(ak1r > -(*elim))
    {
        goto L40;
    }
L30:
    ++(*nz);
    yr[nn] = zeror;
    yi[nn] = zeroi;
    if(acz > dfnu)
    {
        goto L190;
    }
    --nn;
    if(nn == 0)
    {
        return 0;
    }
    goto L20;
L40:
    if(ak1r > -(*alim))
    {
        goto L50;
    }
    iflag = 1;
    ss    = 1. / *tol;
    crscr = *tol;
    ascle = arm * ss;
L50:
    aa = exp(ak1r);
    if(iflag == 1)
    {
        aa *= ss;
    }
    coefr = aa * cos(ak1i);
    coefi = aa * sin(ak1i);
    atol  = *tol * acz / fnup;
    il    = std::min(2L, nn);
    i__1  = il;
    for(i__ = 1; i__ <= i__1; ++i__)
    {
        dfnu = *fnu + (double)((float)(nn - i__));
        fnup = dfnu + 1.;
        s1r  = coner;
        s1i  = conei;
        if(acz < *tol * fnup)
        {
            goto L70;
        }
        ak1r = coner;
        ak1i = conei;
        ak   = fnup + 2.;
        s    = fnup;
        aa   = 2.;
    L60:
        rs   = 1. / s;
        str  = ak1r * czr - ak1i * czi;
        sti  = ak1r * czi + ak1i * czr;
        ak1r = str * rs;
        ak1i = sti * rs;
        s1r += ak1r;
        s1i += ak1i;
        s += ak;
        ak += 2.;
        aa = aa * acz * rs;
        if(aa > atol)
        {
            goto L60;
        }
    L70:
        s2r         = s1r * coefr - s1i * coefi;
        s2i         = s1r * coefi + s1i * coefr;
        wr[i__ - 1] = s2r;
        wi[i__ - 1] = s2i;
        if(iflag == 0)
        {
            goto L80;
        }
        zuchk_(&s2r, &s2i, &nw, &ascle, tol);
        if(nw != 0)
        {
            goto L30;
        }
    L80:
        m     = nn - i__ + 1;
        yr[m] = s2r * crscr;
        yi[m] = s2i * crscr;
        if(i__ == il)
        {
            goto L90;
        }
        zdiv_(&coefr, &coefi, &hzr, &hzi, &str, &sti);
        coefr = str * dfnu;
        coefi = sti * dfnu;
    L90:;
    }
    if(nn <= 2)
    {
        return 0;
    }
    k   = nn - 2;
    ak  = (double)((float)k);
    raz = 1. / az;
    str = *zr * raz;
    sti = -(*zi) * raz;
    rzr = (str + str) * raz;
    rzi = (sti + sti) * raz;
    if(iflag == 1)
    {
        goto L120;
    }
    ib = 3;
L100:
    i__1 = nn;
    for(i__ = ib; i__ <= i__1; ++i__)
    {
        yr[k] = (ak + *fnu) * (rzr * yr[k + 1] - rzi * yi[k + 1]) + yr[k + 2];
        yi[k] = (ak + *fnu) * (rzr * yi[k + 1] + rzi * yr[k + 1]) + yi[k + 2];
        ak += -1.;
        --k;
        /* L110: */
    }
    return 0;
/* ----------------------------------------------------------------------- */
/*     RECUR BACKWARD WITH SCALED VALUES */
/* ----------------------------------------------------------------------- */
L120:
    /* ----------------------------------------------------------------------- */
    /*     EXP(-ALIM)=EXP(-ELIM)/TOL=APPROX. ONE PRECISION ABOVE THE */
    /*     UNDERFLOW LIMIT = ASCLE = D1MACH(1)*SS*1.0D+3 */
    /* ----------------------------------------------------------------------- */
    s1r  = wr[0];
    s1i  = wi[0];
    s2r  = wr[1];
    s2i  = wi[1];
    i__1 = nn;
    for(l = 3; l <= i__1; ++l)
    {
        ckr   = s2r;
        cki   = s2i;
        s2r   = s1r + (ak + *fnu) * (rzr * ckr - rzi * cki);
        s2i   = s1i + (ak + *fnu) * (rzr * cki + rzi * ckr);
        s1r   = ckr;
        s1i   = cki;
        ckr   = s2r * crscr;
        cki   = s2i * crscr;
        yr[k] = ckr;
        yi[k] = cki;
        ak += -1.;
        --k;
        if(zabs_(&ckr, &cki) > ascle)
        {
            goto L140;
        }
        /* L130: */
    }
    return 0;
L140:
    ib = l + 1;
    if(ib > nn)
    {
        return 0;
    }
    goto L100;
L150:
    *nz = *n;
    if(*fnu == 0.)
    {
        --(*nz);
    }
L160:
    yr[1] = zeror;
    yi[1] = zeroi;
    if(*fnu != 0.)
    {
        goto L170;
    }
    yr[1] = coner;
    yi[1] = conei;
L170:
    if(*n == 1)
    {
        return 0;
    }
    i__1 = *n;
    for(i__ = 2; i__ <= i__1; ++i__)
    {
        yr[i__] = zeror;
        yi[i__] = zeroi;
        /* L180: */
    }
    return 0;
/* ----------------------------------------------------------------------- */
/*     RETURN WITH NZ.LT.0 IF CABS(Z*Z/4).GT.FNU+N-NZ-1 COMPLETE */
/*     THE CALCULATION IN CBINU WITH N=N-IABS(NZ) */
/* ----------------------------------------------------------------------- */
L190:
    *nz = -(*nz);
    return 0;
} /* zseri_ */

 int zshch_(double* zr, double* zi, double* cshr, double* cshi, double* cchr, double* cchi)
{
   
    
    double ch, cn, sh, sn;

    /* *********************************************************************72 */

    /* ***BEGIN PROLOGUE  ZSHCH */
    /* ***REFER TO  ZBESK,ZBESH */

    /*     ZSHCH COMPUTES THE COMPLEX HYPERBOLIC FUNCTIONS CSH=SINH(X+I*Y) */
    /*     AND CCH=COSH(X+I*Y), WHERE I**2=-1. */

    /* ***ROUTINES CALLED  (NONE) */
    /* ***END PROLOGUE  ZSHCH */

    sh    = sinh(*zr);
    ch    = cosh(*zr);
    sn    = sin(*zi);
    cn    = cos(*zi);
    *cshr = sh * cn;
    *cshi = ch * sn;
    *cchr = ch * cn;
    *cchi = sh * sn;
    return 0;
} /* zshch_ */

 int zsqrt_(double* ar, double* ai, double* br, double* bi)
{
    /* Initialized data */

    static double drt = .7071067811865475244008443621;
    static double dpi = 3.141592653589793238462643383;


    
    double        zm;
    double        dtheta;

    /* *********************************************************************72 */

    /* c ZSQRT carries out double precision System::Complex<float> square roots. */

    /* ***BEGIN PROLOGUE  ZSQRT */
    /* ***REFER TO  ZBESH,ZBESI,ZBESJ,ZBESK,ZBESY,ZAIRY,ZBIRY */

    /*     DOUBLE PRECISION COMPLEX SQUARE ROOT, B=CSQRT(A) */

    /* ***ROUTINES CALLED  ZABS */
    /* ***END PROLOGUE  ZSQRT */
    zm = zabs_(ar, ai);
    zm = sqrt(zm);
    if(*ar == 0.)
    {
        goto L10;
    }
    if(*ai == 0.)
    {
        goto L20;
    }
    dtheta = atan(*ai / *ar);
    if(dtheta <= 0.)
    {
        goto L40;
    }
    if(*ar < 0.)
    {
        dtheta -= dpi;
    }
    goto L50;
L10:
    if(*ai > 0.)
    {
        goto L60;
    }
    if(*ai < 0.)
    {
        goto L70;
    }
    *br = 0.;
    *bi = 0.;
    return 0;
L20:
    if(*ar > 0.)
    {
        goto L30;
    }
    *br = 0.;
    *bi = sqrt((abs(*ar)));
    return 0;
L30:
    *br = sqrt(*ar);
    *bi = 0.;
    return 0;
L40:
    if(*ar < 0.)
    {
        dtheta += dpi;
    }
L50:
    dtheta *= .5;
    *br = zm * cos(dtheta);
    *bi = zm * sin(dtheta);
    return 0;
L60:
    *br = zm * drt;
    *bi = zm * drt;
    return 0;
L70:
    *br = zm * drt;
    *bi = -zm * drt;
    return 0;
} /* zsqrt_ */

 int zuchk_(double* yr, double* yi, int32* nz, double* ascle, double* tol)
{
    double wi, ss, st, wr;

    /* *********************************************************************72 */

    /* ***BEGIN PROLOGUE  ZUCHK */
    /* ***REFER TO ZSERI,ZUOIK,ZUNK1,ZUNK2,ZUNI1,ZUNI2,ZKSCL */

    /*      Y ENTERS AS A SCALED QUANTITY WHOSE MAGNITUDE IS GREATER THAN */
    /*      EXP(-ALIM)=ASCLE=1.0E+3*D1MACH(1)/TOL. THE TEST IS MADE TO SEE */
    /*      IF THE MAGNITUDE OF THE REAL OR IMAGINARY PART WOULD UNDERFLOW */
    /*      WHEN Y IS SCALED (BY TOL) TO ITS PROPER VALUE. Y IS ACCEPTED */
    /*      IF THE UNDERFLOW IS AT LEAST ONE PRECISION BELOW THE MAGNITUDE */
    /*      OF THE LARGEST COMPONENT; OTHERWISE THE PHASE ANGLE DOES NOT HAVE */
    /*      ABSOLUTE ACCURACY AND AN UNDERFLOW IS ASSUMED. */

    /* ***ROUTINES CALLED  (NONE) */
    /* ***END PROLOGUE  ZUCHK */

    /*     COMPLEX Y */
    *nz = 0;
    wr  = abs(*yr);
    wi  = abs(*yi);
    st  = std::min(wr, wi);
    if(st > *ascle)
    {
        return 0;
    }
    ss = max(wr, wi);
    st /= *tol;
    if(ss < st)
    {
        *nz = 1;
    }
    return 0;
} /* zuchk_ */

 int zunhj_(double* zr,
                            double* zi,
                            double* fnu,
                            int32*    ipmtr,
                            double* tol,
                            double* phir,
                            double* phii,
                            double* argr,
                            double* argi,
                            double* zeta1r,
                            double* zeta1i,
                            double* zeta2r,
                            double* zeta2i,
                            double* asumr,
                            double* asumi,
                            double* bsumr,
                            double* bsumi)
{
    /* Initialized data */

    static double ar[14]    = {1.,
                                .104166666666666667,
                                .0835503472222222222,
                                .12822657455632716,
                                .291849026464140464,
                                .881627267443757652,
                                3.32140828186276754,
                                14.9957629868625547,
                                78.9230130115865181,
                                474.451538868264323,
                                3207.49009089066193,
                                24086.5496408740049,
                                198923.119169509794,
                                1791902.00777534383};
    static double br[14]    = {1.,
                                -.145833333333333333,
                                -.0987413194444444444,
                                -.143312053915895062,
                                -.317227202678413548,
                                -.942429147957120249,
                                -3.51120304082635426,
                                -15.7272636203680451,
                                -82.2814390971859444,
                                -492.355370523670524,
                                -3316.21856854797251,
                                -24827.6742452085896,
                                -204526.587315129788,
                                -1838444.9170682099};
    static double c__[105]  = {1.,
                                  -.208333333333333333,
                                  .125,
                                  .334201388888888889,
                                  -.401041666666666667,
                                  .0703125,
                                  -1.02581259645061728,
                                  1.84646267361111111,
                                  -.8912109375,
                                  .0732421875,
                                  4.66958442342624743,
                                  -11.2070026162229938,
                                  8.78912353515625,
                                  -2.3640869140625,
                                  .112152099609375,
                                  -28.2120725582002449,
                                  84.6362176746007346,
                                  -91.8182415432400174,
                                  42.5349987453884549,
                                  -7.3687943594796317,
                                  .227108001708984375,
                                  212.570130039217123,
                                  -765.252468141181642,
                                  1059.99045252799988,
                                  -699.579627376132541,
                                  218.19051174421159,
                                  -26.4914304869515555,
                                  .572501420974731445,
                                  -1919.457662318407,
                                  8061.72218173730938,
                                  -13586.5500064341374,
                                  11655.3933368645332,
                                  -5305.64697861340311,
                                  1200.90291321635246,
                                  -108.090919788394656,
                                  1.7277275025844574,
                                  20204.2913309661486,
                                  -96980.5983886375135,
                                  192547.001232531532,
                                  -203400.177280415534,
                                  122200.46498301746,
                                  -41192.6549688975513,
                                  7109.51430248936372,
                                  -493.915304773088012,
                                  6.07404200127348304,
                                  -242919.187900551333,
                                  1311763.6146629772,
                                  -2998015.91853810675,
                                  3763271.297656404,
                                  -2813563.22658653411,
                                  1268365.27332162478,
                                  -331645.172484563578,
                                  45218.7689813627263,
                                  -2499.83048181120962,
                                  24.3805296995560639,
                                  3284469.85307203782,
                                  -19706819.1184322269,
                                  50952602.4926646422,
                                  -74105148.2115326577,
                                  66344512.2747290267,
                                  -37567176.6607633513,
                                  13288767.1664218183,
                                  -2785618.12808645469,
                                  308186.404612662398,
                                  -13886.0897537170405,
                                  110.017140269246738,
                                  -49329253.664509962,
                                  325573074.185765749,
                                  -939462359.681578403,
                                  1553596899.57058006,
                                  -1621080552.10833708,
                                  1106842816.82301447,
                                  -495889784.275030309,
                                  142062907.797533095,
                                  -24474062.7257387285,
                                  2243768.17792244943,
                                  -84005.4336030240853,
                                  551.335896122020586,
                                  814789096.118312115,
                                  -5866481492.05184723,
                                  18688207509.2958249,
                                  -34632043388.1587779,
                                  41280185579.753974,
                                  -33026599749.8007231,
                                  17954213731.1556001,
                                  -6563293792.61928433,
                                  1559279864.87925751,
                                  -225105661.889415278,
                                  17395107.5539781645,
                                  -549842.327572288687,
                                  3038.09051092238427,
                                  -14679261247.6956167,
                                  114498237732.02581,
                                  -399096175224.466498,
                                  819218669548.577329,
                                  -1098375156081.22331,
                                  1008158106865.38209,
                                  -645364869245.376503,
                                  287900649906.150589,
                                  -87867072178.0232657,
                                  17634730606.8349694,
                                  -2167164983.22379509,
                                  143157876.718888981,
                                  -3871833.44257261262,
                                  18257.7554742931747};
    static double alfa[180] = {
        -.00444444444444444444,  -9.22077922077922078e-4, -8.84892884892884893e-5, 1.65927687832449737e-4,  2.4669137274179291e-4,   2.6599558934625478e-4,
        2.61824297061500945e-4,  2.48730437344655609e-4,  2.32721040083232098e-4,  2.16362485712365082e-4,  2.00738858762752355e-4,  1.86267636637545172e-4,
        1.73060775917876493e-4,  1.61091705929015752e-4,  1.50274774160908134e-4,  1.40503497391269794e-4,  1.31668816545922806e-4,  1.23667445598253261e-4,
        1.16405271474737902e-4,  1.09798298372713369e-4,  1.03772410422992823e-4,  9.82626078369363448e-5,  9.32120517249503256e-5,  8.85710852478711718e-5,
        8.42963105715700223e-5,  8.03497548407791151e-5,  7.66981345359207388e-5,  7.33122157481777809e-5,  7.01662625163141333e-5,  6.72375633790160292e-5,
        6.93735541354588974e-4,  2.32241745182921654e-4,  -1.41986273556691197e-5, -1.1644493167204864e-4,  -1.50803558053048762e-4, -1.55121924918096223e-4,
        -1.46809756646465549e-4, -1.33815503867491367e-4, -1.19744975684254051e-4, -1.0618431920797402e-4,  -9.37699549891194492e-5, -8.26923045588193274e-5,
        -7.29374348155221211e-5, -6.44042357721016283e-5, -5.69611566009369048e-5, -5.04731044303561628e-5, -4.48134868008882786e-5, -3.98688727717598864e-5,
        -3.55400532972042498e-5, -3.1741425660902248e-5,  -2.83996793904174811e-5, -2.54522720634870566e-5, -2.28459297164724555e-5, -2.05352753106480604e-5,
        -1.84816217627666085e-5, -1.66519330021393806e-5, -1.50179412980119482e-5, -1.35554031379040526e-5, -1.22434746473858131e-5, -1.10641884811308169e-5,
        -3.54211971457743841e-4, -1.56161263945159416e-4, 3.0446550359493641e-5,   1.30198655773242693e-4,  1.67471106699712269e-4,  1.70222587683592569e-4,
        1.56501427608594704e-4,  1.3633917097744512e-4,   1.14886692029825128e-4,  9.45869093034688111e-5,  7.64498419250898258e-5,  6.07570334965197354e-5,
        4.74394299290508799e-5,  3.62757512005344297e-5,  2.69939714979224901e-5,  1.93210938247939253e-5,  1.30056674793963203e-5,  7.82620866744496661e-6,
        3.59257485819351583e-6,  1.44040049814251817e-7,  -2.65396769697939116e-6, -4.9134686709848591e-6,  -6.72739296091248287e-6, -8.17269379678657923e-6,
        -9.31304715093561232e-6, -1.02011418798016441e-5, -1.0880596251059288e-5,  -1.13875481509603555e-5, -1.17519675674556414e-5, -1.19987364870944141e-5,
        3.78194199201772914e-4,  2.02471952761816167e-4,  -6.37938506318862408e-5, -2.38598230603005903e-4, -3.10916256027361568e-4, -3.13680115247576316e-4,
        -2.78950273791323387e-4, -2.28564082619141374e-4, -1.75245280340846749e-4, -1.25544063060690348e-4, -8.22982872820208365e-5, -4.62860730588116458e-5,
        -1.72334302366962267e-5, 5.60690482304602267e-6,  2.313954431482868e-5,    3.62642745856793957e-5,  4.58006124490188752e-5,  5.2459529495911405e-5,
        5.68396208545815266e-5,  5.94349820393104052e-5,  6.06478527578421742e-5,  6.08023907788436497e-5,  6.01577894539460388e-5,  5.891996573446985e-5,
        5.72515823777593053e-5,  5.52804375585852577e-5,  5.3106377380288017e-5,   5.08069302012325706e-5,  4.84418647620094842e-5,  4.6056858160747537e-5,
        -6.91141397288294174e-4, -4.29976633058871912e-4, 1.83067735980039018e-4,  6.60088147542014144e-4,  8.75964969951185931e-4,  8.77335235958235514e-4,
        7.49369585378990637e-4,  5.63832329756980918e-4,  3.68059319971443156e-4,  1.88464535514455599e-4,  3.70663057664904149e-5,  -8.28520220232137023e-5,
        -1.72751952869172998e-4, -2.36314873605872983e-4, -2.77966150694906658e-4, -3.02079514155456919e-4, -3.12594712643820127e-4, -3.12872558758067163e-4,
        -3.05678038466324377e-4, -2.93226470614557331e-4, -2.77255655582934777e-4, -2.59103928467031709e-4, -2.39784014396480342e-4, -2.20048260045422848e-4,
        -2.00443911094971498e-4, -1.81358692210970687e-4, -1.63057674478657464e-4, -1.45712672175205844e-4, -1.29425421983924587e-4, -1.14245691942445952e-4,
        .00192821964248775885,   .00135592576302022234,   -7.17858090421302995e-4, -.00258084802575270346,  -.00349271130826168475,  -.00346986299340960628,
        -.00282285233351310182,  -.00188103076404891354,  -8.895317183839476e-4,   3.87912102631035228e-6,  7.28688540119691412e-4,  .00126566373053457758,
        .00162518158372674427,   .00183203153216373172,   .00191588388990527909,   .00190588846755546138,   .00182798982421825727,   .0017038950642112153,
        .00155097127171097686,   .00138261421852276159,   .00120881424230064774,   .00103676532638344962,   8.71437918068619115e-4,  7.16080155297701002e-4,
        5.72637002558129372e-4,  4.42089819465802277e-4,  3.24724948503090564e-4,  2.20342042730246599e-4,  1.28412898401353882e-4,  4.82005924552095464e-5};
    static double beta[210] = {
        .0179988721413553309,    .00559964911064388073,   .00288501402231132779,   .00180096606761053941,   .00124753110589199202,   9.22878876572938311e-4,
        7.14430421727287357e-4,  5.71787281789704872e-4,  4.69431007606481533e-4,  3.93232835462916638e-4,  3.34818889318297664e-4,  2.88952148495751517e-4,
        2.52211615549573284e-4,  2.22280580798883327e-4,  1.97541838033062524e-4,  1.76836855019718004e-4,  1.59316899661821081e-4,  1.44347930197333986e-4,
        1.31448068119965379e-4,  1.20245444949302884e-4,  1.10449144504599392e-4,  1.01828770740567258e-4,  9.41998224204237509e-5,  8.74130545753834437e-5,
        8.13466262162801467e-5,  7.59002269646219339e-5,  7.09906300634153481e-5,  6.65482874842468183e-5,  6.25146958969275078e-5,  5.88403394426251749e-5,
        -.00149282953213429172,  -8.78204709546389328e-4, -5.02916549572034614e-4, -2.94822138512746025e-4, -1.75463996970782828e-4, -1.04008550460816434e-4,
        -5.96141953046457895e-5, -3.1203892907609834e-5,  -1.26089735980230047e-5, -2.42892608575730389e-7, 8.05996165414273571e-6,  1.36507009262147391e-5,
        1.73964125472926261e-5,  1.9867297884213378e-5,   2.14463263790822639e-5,  2.23954659232456514e-5,  2.28967783814712629e-5,  2.30785389811177817e-5,
        2.30321976080909144e-5,  2.28236073720348722e-5,  2.25005881105292418e-5,  2.20981015361991429e-5,  2.16418427448103905e-5,  2.11507649256220843e-5,
        2.06388749782170737e-5,  2.01165241997081666e-5,  1.95913450141179244e-5,  1.9068936791043674e-5,   1.85533719641636667e-5,  1.80475722259674218e-5,
        5.5221307672129279e-4,   4.47932581552384646e-4,  2.79520653992020589e-4,  1.52468156198446602e-4,  6.93271105657043598e-5,  1.76258683069991397e-5,
        -1.35744996343269136e-5, -3.17972413350427135e-5, -4.18861861696693365e-5, -4.69004889379141029e-5, -4.87665447413787352e-5, -4.87010031186735069e-5,
        -4.74755620890086638e-5, -4.55813058138628452e-5, -4.33309644511266036e-5, -4.09230193157750364e-5, -3.84822638603221274e-5, -3.60857167535410501e-5,
        -3.37793306123367417e-5, -3.15888560772109621e-5, -2.95269561750807315e-5, -2.75978914828335759e-5, -2.58006174666883713e-5, -2.413083567612802e-5,
        -2.25823509518346033e-5, -2.11479656768912971e-5, -1.98200638885294927e-5, -1.85909870801065077e-5, -1.74532699844210224e-5, -1.63997823854497997e-5,
        -4.74617796559959808e-4, -4.77864567147321487e-4, -3.20390228067037603e-4, -1.61105016119962282e-4, -4.25778101285435204e-5, 3.44571294294967503e-5,
        7.97092684075674924e-5,  1.031382367082722e-4,    1.12466775262204158e-4,  1.13103642108481389e-4,  1.08651634848774268e-4,  1.01437951597661973e-4,
        9.29298396593363896e-5,  8.40293133016089978e-5,  7.52727991349134062e-5,  6.69632521975730872e-5,  5.92564547323194704e-5,  5.22169308826975567e-5,
        4.58539485165360646e-5,  4.01445513891486808e-5,  3.50481730031328081e-5,  3.05157995034346659e-5,  2.64956119950516039e-5,  2.29363633690998152e-5,
        1.97893056664021636e-5,  1.70091984636412623e-5,  1.45547428261524004e-5,  1.23886640995878413e-5,  1.04775876076583236e-5,  8.79179954978479373e-6,
        7.36465810572578444e-4,  8.72790805146193976e-4,  6.22614862573135066e-4,  2.85998154194304147e-4,  3.84737672879366102e-6,  -1.87906003636971558e-4,
        -2.97603646594554535e-4, -3.45998126832656348e-4, -3.53382470916037712e-4, -3.35715635775048757e-4, -3.04321124789039809e-4, -2.66722723047612821e-4,
        -2.27654214122819527e-4, -1.89922611854562356e-4, -1.5505891859909387e-4,  -1.2377824076187363e-4,  -9.62926147717644187e-5, -7.25178327714425337e-5,
        -5.22070028895633801e-5, -3.50347750511900522e-5, -2.06489761035551757e-5, -8.70106096849767054e-6, 1.1369868667510029e-6,   9.16426474122778849e-6,
        1.5647778542887262e-5,   2.08223629482466847e-5,  2.48923381004595156e-5,  2.80340509574146325e-5,  3.03987774629861915e-5,  3.21156731406700616e-5,
        -.00180182191963885708,  -.00243402962938042533,  -.00183422663549856802,  -7.62204596354009765e-4, 2.39079475256927218e-4,  9.49266117176881141e-4,
        .00134467449701540359,   .00148457495259449178,   .00144732339830617591,   .00130268261285657186,   .00110351597375642682,   8.86047440419791759e-4,
        6.73073208165665473e-4,  4.77603872856582378e-4,  3.05991926358789362e-4,  1.6031569459472163e-4,   4.00749555270613286e-5,  -5.66607461635251611e-5,
        -1.32506186772982638e-4, -1.90296187989614057e-4, -2.32811450376937408e-4, -2.62628811464668841e-4, -2.82050469867598672e-4, -2.93081563192861167e-4,
        -2.97435962176316616e-4, -2.96557334239348078e-4, -2.91647363312090861e-4, -2.83696203837734166e-4, -2.73512317095673346e-4, -2.6175015580676858e-4,
        .00638585891212050914,   .00962374215806377941,   .00761878061207001043,   .00283219055545628054,   -.0020984135201272009,   -.00573826764216626498,
        -.0077080424449541462,   -.00821011692264844401,  -.00765824520346905413,  -.00647209729391045177,  -.00499132412004966473,  -.0034561228971313328,
        -.00201785580014170775,  -7.59430686781961401e-4, 2.84173631523859138e-4,  .00110891667586337403,   .00172901493872728771,   .00216812590802684701,
        .00245357710494539735,   .00261281821058334862,   .00267141039656276912,   .0026520307339598043,    .00257411652877287315,   .00245389126236094427,
        .00230460058071795494,   .00213684837686712662,   .00195896528478870911,   .00177737008679454412,   .00159690280765839059,   .00142111975664438546};
    static double gama[30] = {.629960524947436582,  .251984209978974633,  .154790300415655846,  .110713062416159013,  .0857309395527394825, .0697161316958684292,
                                  .0586085671893713576, .0504698873536310685, .0442600580689154809, .0393720661543509966, .0354283195924455368, .0321818857502098231,
                                  .0294646240791157679, .0271581677112934479, .0251768272973861779, .0234570755306078891, .0219508390134907203, .020621082823564624,
                                  .0194388240897880846, .0183810633800683158, .0174293213231963172, .0165685837786612353, .0157865285987918445, .0150729501494095594,
                                  .0144193250839954639, .0138184805735341786, .0132643378994276568, .0127517121970498651, .0122761545318762767, .0118338262398482403};
    static double ex1      = .333333333333333333;
    static double ex2      = .666666666666666667;
    static double hpi      = 1.57079632679489662;
    static double gpi      = 3.14159265358979324;
    static double thpi     = 4.71238898038468986;
    static double zeror    = 0.;
    static double zeroi    = 0.;
    static double coner    = 1.;
    static double conei    = 0.;

    /* System generated locals */
    int32    i__1, i__2;
    double d__1;



    
    int32                     j, k, l, m, l1, l2;
    double                  ac, ap[30], pi[30];
    int32                     is, jr, ks, ju;
    double                  pp, wi, pr[30];
    int32                     lr;
    double                  wr, aw2;
    int32                     kp1;
    double                  t2i, w2i, t2r, w2r, ang, fn13, fn23;
    int32                     ias;
    double                  cri[14], dri[14];
    int32                     ibs;
    double                  zai, zbi, zci, crr[14], drr[14], raw, zar, upi[14], sti, zbr, zcr, upr[14], str, raw2;
    int32                     lrp1;
    double                  rfn13;
    int32                     idum;
    
    double                  atol, btol, tfni;
    int32                     kmax;
    double                  azth, tzai, tfnr, rfnu;
    double                  zthi, test, tzar, zthr, rfnu2, zetai, ptfni, sumai, sumbi, zetar, ptfnr, razth, sumar, sumbr, rzthi;
    double                  rzthr, rtzti;
    double                  rtztr, przthi, przthr;

    /* *********************************************************************72 */

    /* ***BEGIN PROLOGUE  ZUNHJ */
    /* ***REFER TO  ZBESI,ZBESK */

    /*     REFERENCES */
    /*         HANDBOOK OF MATHEMATICAL FUNCTIONS BY M. ABRAMOWITZ AND I.A. */
    /*         STEGUN, AMS55, NATIONAL BUREAU OF STANDARDS, 1965, CHAPTER 9. */

    /*         ASYMPTOTICS AND SPECIAL FUNCTIONS BY F.W.J. OLVER, ACADEMIC */
    /*         PRESS, N.Y., 1974, PAGE 420 */

    /*     ABSTRACT */
    /*         ZUNHJ COMPUTES PARAMETERS FOR BESSEL FUNCTIONS C(FNU,Z) = */
    /*         J(FNU,Z), Y(FNU,Z) OR H(I,FNU,Z) I=1,2 FOR LARGE ORDERS FNU */
    /*         BY MEANS OF THE UNIFORM ASYMPTOTIC EXPANSION */

    /*         C(FNU,Z)=C1*PHI*( ASUM*AIRY(ARG) + C2*BSUM*DAIRY(ARG) ) */

    /*         FOR PROPER CHOICES OF C1, C2, AIRY AND DAIRY WHERE AIRY IS */
    /*         AN AIRY FUNCTION AND DAIRY IS ITS DERIVATIVE. */

    /*               (2/3)*FNU*ZETA**1.5 = ZETA1-ZETA2, */

    /*         ZETA1=0.5*FNU*CLOG((1+W)/(1-W)), ZETA2=FNU*W FOR SCALING */
    /*         PURPOSES IN AIRY FUNCTIONS FROM CAIRY OR CBIRY. */

    /*         MCONJ=SIGN OF AIMAG(Z), BUT IS AMBIGUOUS WHEN Z IS REAL AND */
    /*         MUST BE SPECIFIED. IPMTR=0 RETURNS ALL PARAMETERS. IPMTR= */
    /*         1 COMPUTES ALL EXCEPT ASUM AND BSUM. */

    /* ***ROUTINES CALLED  ZABS,ZDIV,ZLOG,ZSQRT,D1MACH */
    /* ***END PROLOGUE  ZUNHJ */
    /*     COMPLEX ARG,ASUM,BSUM,CFNU,CONE,CR,CZERO,DR,P,PHI,PRZTH,PTFN, */
    /*    *RFN13,RTZTA,RZTH,SUMA,SUMB,TFN,T2,UP,W,W2,Z,ZA,ZB,ZC,ZETA,ZETA1, */
    /*    *ZETA2,ZTH */

    rfnu = 1. / *fnu;
    /* ----------------------------------------------------------------------- */
    /*     OVERFLOW TEST (Z/FNU TOO SMALL) */
    /* ----------------------------------------------------------------------- */
    test = d1mach_(&c__1) * 1e3;
    ac   = *fnu * test;
    if(abs(*zr) > ac || abs(*zi) > ac)
    {
        goto L15;
    }
    *zeta1r = (d__1 = log(test), abs(d__1)) * 2. + *fnu;
    *zeta1i = 0.;
    *zeta2r = *fnu;
    *zeta2i = 0.;
    *phir   = 1.;
    *phii   = 0.;
    *argr   = 1.;
    *argi   = 0.;
    return 0;
L15:
    zbr   = *zr * rfnu;
    zbi   = *zi * rfnu;
    rfnu2 = rfnu * rfnu;
    /* ----------------------------------------------------------------------- */
    /*     COMPUTE IN THE FOURTH QUADRANT */
    /* ----------------------------------------------------------------------- */
    fn13  = pow_dd(fnu, &ex1);
    fn23  = fn13 * fn13;
    rfn13 = 1. / fn13;
    w2r   = coner - zbr * zbr + zbi * zbi;
    w2i   = conei - zbr * zbi - zbr * zbi;
    aw2   = zabs_(&w2r, &w2i);
    if(aw2 > .25)
    {
        goto L130;
    }
    /* ----------------------------------------------------------------------- */
    /*     POWER SERIES FOR CABS(W2).LE.0.25D0 */
    /* ----------------------------------------------------------------------- */
    k     = 1;
    pr[0] = coner;
    pi[0] = conei;
    sumar = gama[0];
    sumai = zeroi;
    ap[0] = 1.;
    if(aw2 < *tol)
    {
        goto L20;
    }
    for(k = 2; k <= 30; ++k)
    {
        pr[k - 1] = pr[k - 2] * w2r - pi[k - 2] * w2i;
        pi[k - 1] = pr[k - 2] * w2i + pi[k - 2] * w2r;
        sumar += pr[k - 1] * gama[k - 1];
        sumai += pi[k - 1] * gama[k - 1];
        ap[k - 1] = ap[k - 2] * aw2;
        if(ap[k - 1] < *tol)
        {
            goto L20;
        }
        /* L10: */
    }
    k = 30;
L20:
    kmax  = k;
    zetar = w2r * sumar - w2i * sumai;
    zetai = w2r * sumai + w2i * sumar;
    *argr = zetar * fn23;
    *argi = zetai * fn23;
    zsqrt_(&sumar, &sumai, &zar, &zai);
    zsqrt_(&w2r, &w2i, &str, &sti);
    *zeta2r = str * *fnu;
    *zeta2i = sti * *fnu;
    str     = coner + ex2 * (zetar * zar - zetai * zai);
    sti     = conei + ex2 * (zetar * zai + zetai * zar);
    *zeta1r = str * *zeta2r - sti * *zeta2i;
    *zeta1i = str * *zeta2i + sti * *zeta2r;
    zar += zar;
    zai += zai;
    zsqrt_(&zar, &zai, &str, &sti);
    *phir = str * rfn13;
    *phii = sti * rfn13;
    if(*ipmtr == 1)
    {
        goto L120;
    }
    /* ----------------------------------------------------------------------- */
    /*     SUM SERIES FOR ASUM AND BSUM */
    /* ----------------------------------------------------------------------- */
    sumbr = zeror;
    sumbi = zeroi;
    i__1  = kmax;
    for(k = 1; k <= i__1; ++k)
    {
        sumbr += pr[k - 1] * beta[k - 1];
        sumbi += pi[k - 1] * beta[k - 1];
        /* L30: */
    }
    *asumr = zeror;
    *asumi = zeroi;
    *bsumr = sumbr;
    *bsumi = sumbi;
    l1     = 0;
    l2     = 30;
    btol   = *tol * (abs(*bsumr) + abs(*bsumi));
    atol   = *tol;
    pp     = 1.;
    ias    = 0;
    ibs    = 0;
    if(rfnu2 < *tol)
    {
        goto L110;
    }
    for(is = 2; is <= 7; ++is)
    {
        atol /= rfnu2;
        pp *= rfnu2;
        if(ias == 1)
        {
            goto L60;
        }
        sumar = zeror;
        sumai = zeroi;
        i__1  = kmax;
        for(k = 1; k <= i__1; ++k)
        {
            m = l1 + k;
            sumar += pr[k - 1] * alfa[m - 1];
            sumai += pi[k - 1] * alfa[m - 1];
            if(ap[k - 1] < atol)
            {
                goto L50;
            }
            /* L40: */
        }
    L50:
        *asumr += sumar * pp;
        *asumi += sumai * pp;
        if(pp < *tol)
        {
            ias = 1;
        }
    L60:
        if(ibs == 1)
        {
            goto L90;
        }
        sumbr = zeror;
        sumbi = zeroi;
        i__1  = kmax;
        for(k = 1; k <= i__1; ++k)
        {
            m = l2 + k;
            sumbr += pr[k - 1] * beta[m - 1];
            sumbi += pi[k - 1] * beta[m - 1];
            if(ap[k - 1] < atol)
            {
                goto L80;
            }
            /* L70: */
        }
    L80:
        *bsumr += sumbr * pp;
        *bsumi += sumbi * pp;
        if(pp < btol)
        {
            ibs = 1;
        }
    L90:
        if(ias == 1 && ibs == 1)
        {
            goto L110;
        }
        l1 += 30;
        l2 += 30;
        /* L100: */
    }
L110:
    *asumr += coner;
    pp = rfnu * rfn13;
    *bsumr *= pp;
    *bsumi *= pp;
L120:
    return 0;
/* ----------------------------------------------------------------------- */
/*     CABS(W2).GT.0.25D0 */
/* ----------------------------------------------------------------------- */
L130:
    zsqrt_(&w2r, &w2i, &wr, &wi);
    if(wr < 0.)
    {
        wr = 0.;
    }
    if(wi < 0.)
    {
        wi = 0.;
    }
    str = coner + wr;
    sti = wi;
    zdiv_(&str, &sti, &zbr, &zbi, &zar, &zai);
    zlog_(&zar, &zai, &zcr, &zci, &idum);
    if(zci < 0.)
    {
        zci = 0.;
    }
    if(zci > hpi)
    {
        zci = hpi;
    }
    if(zcr < 0.)
    {
        zcr = 0.;
    }
    zthr    = (zcr - wr) * 1.5;
    zthi    = (zci - wi) * 1.5;
    *zeta1r = zcr * *fnu;
    *zeta1i = zci * *fnu;
    *zeta2r = wr * *fnu;
    *zeta2i = wi * *fnu;
    azth    = zabs_(&zthr, &zthi);
    ang     = thpi;
    if(zthr >= 0. && zthi < 0.)
    {
        goto L140;
    }
    ang = hpi;
    if(zthr == 0.)
    {
        goto L140;
    }
    ang = atan(zthi / zthr);
    if(zthr < 0.)
    {
        ang += gpi;
    }
L140:
    pp = pow_dd(&azth, &ex2);
    ang *= ex2;
    zetar = pp * cos(ang);
    zetai = pp * sin(ang);
    if(zetai < 0.)
    {
        zetai = 0.;
    }
    *argr = zetar * fn23;
    *argi = zetai * fn23;
    zdiv_(&zthr, &zthi, &zetar, &zetai, &rtztr, &rtzti);
    zdiv_(&rtztr, &rtzti, &wr, &wi, &zar, &zai);
    tzar = zar + zar;
    tzai = zai + zai;
    zsqrt_(&tzar, &tzai, &str, &sti);
    *phir = str * rfn13;
    *phii = sti * rfn13;
    if(*ipmtr == 1)
    {
        goto L120;
    }
    raw    = 1. / sqrt(aw2);
    str    = wr * raw;
    sti    = -wi * raw;
    tfnr   = str * rfnu * raw;
    tfni   = sti * rfnu * raw;
    razth  = 1. / azth;
    str    = zthr * razth;
    sti    = -zthi * razth;
    rzthr  = str * razth * rfnu;
    rzthi  = sti * razth * rfnu;
    zcr    = rzthr * ar[1];
    zci    = rzthi * ar[1];
    raw2   = 1. / aw2;
    str    = w2r * raw2;
    sti    = -w2i * raw2;
    t2r    = str * raw2;
    t2i    = sti * raw2;
    str    = t2r * c__[1] + c__[2];
    sti    = t2i * c__[1];
    upr[1] = str * tfnr - sti * tfni;
    upi[1] = str * tfni + sti * tfnr;
    *bsumr = upr[1] + zcr;
    *bsumi = upi[1] + zci;
    *asumr = zeror;
    *asumi = zeroi;
    if(rfnu < *tol)
    {
        goto L220;
    }
    przthr = rzthr;
    przthi = rzthi;
    ptfnr  = tfnr;
    ptfni  = tfni;
    upr[0] = coner;
    upi[0] = conei;
    pp     = 1.;
    btol   = *tol * (abs(*bsumr) + abs(*bsumi));
    ks     = 0;
    kp1    = 2;
    l      = 3;
    ias    = 0;
    ibs    = 0;
    for(lr = 2; lr <= 12; lr += 2)
    {
        lrp1 = lr + 1;
        /* ----------------------------------------------------------------------- */
        /*     COMPUTE TWO ADDITIONAL CR, DR, AND UP FOR TWO MORE TERMS IN */
        /*     NEXT SUMA AND SUMB */
        /* ----------------------------------------------------------------------- */
        i__1 = lrp1;
        for(k = lr; k <= i__1; ++k)
        {
            ++ks;
            ++kp1;
            ++l;
            zar  = c__[l - 1];
            zai  = zeroi;
            i__2 = kp1;
            for(j = 2; j <= i__2; ++j)
            {
                ++l;
                str = zar * t2r - t2i * zai + c__[l - 1];
                zai = zar * t2i + zai * t2r;
                zar = str;
                /* L150: */
            }
            str          = ptfnr * tfnr - ptfni * tfni;
            ptfni        = ptfnr * tfni + ptfni * tfnr;
            ptfnr        = str;
            upr[kp1 - 1] = ptfnr * zar - ptfni * zai;
            upi[kp1 - 1] = ptfni * zar + ptfnr * zai;
            crr[ks - 1]  = przthr * br[ks];
            cri[ks - 1]  = przthi * br[ks];
            str          = przthr * rzthr - przthi * rzthi;
            przthi       = przthr * rzthi + przthi * rzthr;
            przthr       = str;
            drr[ks - 1]  = przthr * ar[ks + 1];
            dri[ks - 1]  = przthi * ar[ks + 1];
            /* L160: */
        }
        pp *= rfnu2;
        if(ias == 1)
        {
            goto L180;
        }
        sumar = upr[lrp1 - 1];
        sumai = upi[lrp1 - 1];
        ju    = lrp1;
        i__1  = lr;
        for(jr = 1; jr <= i__1; ++jr)
        {
            --ju;
            sumar = sumar + crr[jr - 1] * upr[ju - 1] - cri[jr - 1] * upi[ju - 1];
            sumai = sumai + crr[jr - 1] * upi[ju - 1] + cri[jr - 1] * upr[ju - 1];
            /* L170: */
        }
        *asumr += sumar;
        *asumi += sumai;
        test = abs(sumar) + abs(sumai);
        if(pp < *tol && test < *tol)
        {
            ias = 1;
        }
    L180:
        if(ibs == 1)
        {
            goto L200;
        }
        sumbr = upr[lr + 1] + upr[lrp1 - 1] * zcr - upi[lrp1 - 1] * zci;
        sumbi = upi[lr + 1] + upr[lrp1 - 1] * zci + upi[lrp1 - 1] * zcr;
        ju    = lrp1;
        i__1  = lr;
        for(jr = 1; jr <= i__1; ++jr)
        {
            --ju;
            sumbr = sumbr + drr[jr - 1] * upr[ju - 1] - dri[jr - 1] * upi[ju - 1];
            sumbi = sumbi + drr[jr - 1] * upi[ju - 1] + dri[jr - 1] * upr[ju - 1];
            /* L190: */
        }
        *bsumr += sumbr;
        *bsumi += sumbi;
        test = abs(sumbr) + abs(sumbi);
        if(pp < btol && test < btol)
        {
            ibs = 1;
        }
    L200:
        if(ias == 1 && ibs == 1)
        {
            goto L220;
        }
        /* L210: */
    }
L220:
    *asumr += coner;
    str = -(*bsumr) * rfn13;
    sti = -(*bsumi) * rfn13;
    zdiv_(&str, &sti, &rtztr, &rtzti, bsumr, bsumi);
    goto L120;
} /* zunhj_ */

 int zuni1_(double* zr,
                            double* zi,
                            double* fnu,
                            int32*    kode,
                            int32*    n,
                            double* yr,
                            double* yi,
                            int32*    nz,
                            int32*    nlast,
                            double* fnul,
                            double* tol,
                            double* elim,
                            double* alim)
{
    /* Initialized data */

    static double zeror = 0.;
    static double zeroi = 0.;
    static double coner = 1.;

    /* System generated locals */
    int32 i__1;

   
    
    int32                     i__, k, m, nd;
    double                  fn;
    int32                     nn, nw;
    double                  c2i, c2m, c1r, c2r, s1i, s2i, rs1, s1r, s2r, cyi[2];
    int32                     nuf;
    double                  bry[3], cyr[2], sti, rzi, str, rzr, aphi, cscl, phii, crsc;
    
    double                  phir;
    int32                     init;
    double                  csrr[3], cssr[3], rast, sumi, sumr;
    int32                     iflag;
    double                  ascle, cwrki[16];
    double                  cwrkr[16];
   
    double zeta1i, zeta2i, zeta1r, zeta2r;

    /* *********************************************************************72 */

    /* ***BEGIN PROLOGUE  ZUNI1 */
    /* ***REFER TO  ZBESI,ZBESK */

    /*     ZUNI1 COMPUTES I(FNU,Z)  BY MEANS OF THE UNIFORM ASYMPTOTIC */
    /*     EXPANSION FOR I(FNU,Z) IN -PI/3.LE.ARG Z.LE.PI/3. */

    /*     FNUL IS THE SMALLEST ORDER PERMITTED FOR THE ASYMPTOTIC */
    /*     EXPANSION. NLAST=0 MEANS ALL OF THE Y VALUES WERE SET. */
    /*     NLAST.NE.0 IS THE NUMBER LEFT TO BE COMPUTED BY ANOTHER */
    /*     FORMULA FOR ORDERS FNU TO FNU+NLAST-1 BECAUSE FNU+NLAST-1.LT.FNUL. */
    /*     Y(I)=CZERO FOR I=NLAST+1,N */

    /* ***ROUTINES CALLED  ZUCHK,ZUNIK,ZUOIK,D1MACH,ZABS */
    /* ***END PROLOGUE  ZUNI1 */
    /*     COMPLEX CFN,CONE,CRSC,CSCL,CSR,CSS,CWRK,CZERO,C1,C2,PHI,RZ,SUM,S1, */
    /*    *S2,Y,Z,ZETA1,ZETA2 */
    /* Parameter adjustments */
    --yi;
    --yr;

    /* Function Body */

    *nz    = 0;
    nd     = *n;
    *nlast = 0;
    /* ----------------------------------------------------------------------- */
    /*     COMPUTED VALUES WITH EXPONENTS BETWEEN ALIM AND ELIM IN MAG- */
    /*     NITUDE ARE SCALED TO KEEP INTERMEDIATE ARITHMETIC ON SCALE, */
    /*     EXP(ALIM)=EXP(ELIM)*TOL */
    /* ----------------------------------------------------------------------- */
    cscl    = 1. / *tol;
    crsc    = *tol;
    cssr[0] = cscl;
    cssr[1] = coner;
    cssr[2] = crsc;
    csrr[0] = crsc;
    csrr[1] = coner;
    csrr[2] = cscl;
    bry[0]  = d1mach_(&c__1) * 1e3 / *tol;
    /* ----------------------------------------------------------------------- */
    /*     CHECK FOR UNDERFLOW AND OVERFLOW ON FIRST MEMBER */
    /* ----------------------------------------------------------------------- */
    fn   = max(*fnu, 1.);
    init = 0;
    zunik_(zr, zi, &fn, &c__1, &c__1, tol, &init, &phir, &phii, &zeta1r, &zeta1i, &zeta2r, &zeta2i, &sumr, &sumi, cwrkr, cwrki);
    if(*kode == 1)
    {
        goto L10;
    }
    str  = *zr + zeta2r;
    sti  = *zi + zeta2i;
    rast = fn / zabs_(&str, &sti);
    str  = str * rast * rast;
    sti  = -sti * rast * rast;
    s1r  = -zeta1r + str;
    s1i  = -zeta1i + sti;
    goto L20;
L10:
    s1r = -zeta1r + zeta2r;
    s1i = -zeta1i + zeta2i;
L20:
    rs1 = s1r;
    if(abs(rs1) > *elim)
    {
        goto L130;
    }
L30:
    nn   = std::min(2L, nd);
    i__1 = nn;
    for(i__ = 1; i__ <= i__1; ++i__)
    {
        fn   = *fnu + (double)((float)(nd - i__));
        init = 0;
        zunik_(zr, zi, &fn, &c__1, &c__0, tol, &init, &phir, &phii, &zeta1r, &zeta1i, &zeta2r, &zeta2i, &sumr, &sumi, cwrkr, cwrki);
        if(*kode == 1)
        {
            goto L40;
        }
        str  = *zr + zeta2r;
        sti  = *zi + zeta2i;
        rast = fn / zabs_(&str, &sti);
        str  = str * rast * rast;
        sti  = -sti * rast * rast;
        s1r  = -zeta1r + str;
        s1i  = -zeta1i + sti + *zi;
        goto L50;
    L40:
        s1r = -zeta1r + zeta2r;
        s1i = -zeta1i + zeta2i;
    L50:
        /* ----------------------------------------------------------------------- */
        /*     TEST FOR UNDERFLOW AND OVERFLOW */
        /* ----------------------------------------------------------------------- */
        rs1 = s1r;
        if(abs(rs1) > *elim)
        {
            goto L110;
        }
        if(i__ == 1)
        {
            iflag = 2;
        }
        if(abs(rs1) < *alim)
        {
            goto L60;
        }
        /* ----------------------------------------------------------------------- */
        /*     REFINE  TEST AND SCALE */
        /* ----------------------------------------------------------------------- */
        aphi = zabs_(&phir, &phii);
        rs1 += log(aphi);
        if(abs(rs1) > *elim)
        {
            goto L110;
        }
        if(i__ == 1)
        {
            iflag = 1;
        }
        if(rs1 < 0.)
        {
            goto L60;
        }
        if(i__ == 1)
        {
            iflag = 3;
        }
    L60:
        /* ----------------------------------------------------------------------- */
        /*     SCALE S1 IF CABS(S1).LT.ASCLE */
        /* ----------------------------------------------------------------------- */
        s2r = phir * sumr - phii * sumi;
        s2i = phir * sumi + phii * sumr;
        str = exp(s1r) * cssr[iflag - 1];
        s1r = str * cos(s1i);
        s1i = str * sin(s1i);
        str = s2r * s1r - s2i * s1i;
        s2i = s2r * s1i + s2i * s1r;
        s2r = str;
        if(iflag != 1)
        {
            goto L70;
        }
        zuchk_(&s2r, &s2i, &nw, bry, tol);
        if(nw != 0)
        {
            goto L110;
        }
    L70:
        cyr[i__ - 1] = s2r;
        cyi[i__ - 1] = s2i;
        m            = nd - i__ + 1;
        yr[m]        = s2r * csrr[iflag - 1];
        yi[m]        = s2i * csrr[iflag - 1];
        /* L80: */
    }
    if(nd <= 2)
    {
        goto L100;
    }
    rast   = 1. / zabs_(zr, zi);
    str    = *zr * rast;
    sti    = -(*zi) * rast;
    rzr    = (str + str) * rast;
    rzi    = (sti + sti) * rast;
    bry[1] = 1. / bry[0];
    bry[2] = d1mach_(&c__2);
    s1r    = cyr[0];
    s1i    = cyi[0];
    s2r    = cyr[1];
    s2i    = cyi[1];
    c1r    = csrr[iflag - 1];
    ascle  = bry[iflag - 1];
    k      = nd - 2;
    fn     = (double)((float)k);
    i__1   = nd;
    for(i__ = 3; i__ <= i__1; ++i__)
    {
        c2r   = s2r;
        c2i   = s2i;
        s2r   = s1r + (*fnu + fn) * (rzr * c2r - rzi * c2i);
        s2i   = s1i + (*fnu + fn) * (rzr * c2i + rzi * c2r);
        s1r   = c2r;
        s1i   = c2i;
        c2r   = s2r * c1r;
        c2i   = s2i * c1r;
        yr[k] = c2r;
        yi[k] = c2i;
        --k;
        fn += -1.;
        if(iflag >= 3)
        {
            goto L90;
        }
        str = abs(c2r);
        sti = abs(c2i);
        c2m = max(str, sti);
        if(c2m <= ascle)
        {
            goto L90;
        }
        ++iflag;
        ascle = bry[iflag - 1];
        s1r *= c1r;
        s1i *= c1r;
        s2r = c2r;
        s2i = c2i;
        s1r *= cssr[iflag - 1];
        s1i *= cssr[iflag - 1];
        s2r *= cssr[iflag - 1];
        s2i *= cssr[iflag - 1];
        c1r = csrr[iflag - 1];
    L90:;
    }
L100:
    return 0;
/* ----------------------------------------------------------------------- */
/*     SET UNDERFLOW AND UPDATE PARAMETERS */
/* ----------------------------------------------------------------------- */
L110:
    if(rs1 > 0.)
    {
        goto L120;
    }
    yr[nd] = zeror;
    yi[nd] = zeroi;
    ++(*nz);
    --nd;
    if(nd == 0)
    {
        goto L100;
    }
    zuoik_(zr, zi, fnu, kode, &c__1, &nd, &yr[1], &yi[1], &nuf, tol, elim, alim);
    if(nuf < 0)
    {
        goto L120;
    }
    nd -= nuf;
    *nz += nuf;
    if(nd == 0)
    {
        goto L100;
    }
    fn = *fnu + (double)((float)(nd - 1));
    if(fn >= *fnul)
    {
        goto L30;
    }
    *nlast = nd;
    return 0;
L120:
    *nz = -1;
    return 0;
L130:
    if(rs1 > 0.)
    {
        goto L120;
    }
    *nz  = *n;
    i__1 = *n;
    for(i__ = 1; i__ <= i__1; ++i__)
    {
        yr[i__] = zeror;
        yi[i__] = zeroi;
        /* L140: */
    }
    return 0;
} /* zuni1_ */

 int zuni2_(double* zr,
                            double* zi,
                            double* fnu,
                            int32*    kode,
                            int32*    n,
                            double* yr,
                            double* yi,
                            int32*    nz,
                            int32*    nlast,
                            double* fnul,
                            double* tol,
                            double* elim,
                            double* alim)
{
    /* Initialized data */

    static double zeror   = 0.;
    static double zeroi   = 0.;
    static double coner   = 1.;
    static double cipr[4] = {1., 0., -1., 0.};
    static double cipi[4] = {0., 1., 0., -1.};
    static double hpi     = 1.57079632679489662;
    static double aic     = 1.265512123484645396;

    /* System generated locals */
    int32 i__1;

   

    
    int32                     i__, j, k, nd;
    double                  fn;
    int32                     in, nn, nw;
    double                  c2i, c2m, c1r, c2r, s1i, s2i, rs1, s1r, s2r, aii, ang, car;
    int32                     nai;
    double                  air, zbi, cyi[2], sar;
    int32                     nuf, inu;
    double                  bry[3], raz, sti, zbr, zni, cyr[2], rzi, str, znr, rzr, daii, cidi, aarg;
    int32                     ndai;
    double                  dair, aphi, argi, cscl, phii, crsc, argr;
    int32                     idum;
    
    double                  phir, csrr[3], cssr[3], rast;
    int32                     iflag;
    double                  ascle, asumi, bsumi;
    double                  asumr, bsumr;

    double zeta1i, zeta2i, zeta1r, zeta2r;

    /* *********************************************************************72 */

    /* ***BEGIN PROLOGUE  ZUNI2 */
    /* ***REFER TO  ZBESI,ZBESK */

    /*     ZUNI2 COMPUTES I(FNU,Z) IN THE RIGHT HALF PLANE BY MEANS OF */
    /*     UNIFORM ASYMPTOTIC EXPANSION FOR J(FNU,ZN) WHERE ZN IS Z*I */
    /*     OR -Z*I AND ZN IS IN THE RIGHT HALF PLANE ALSO. */

    /*     FNUL IS THE SMALLEST ORDER PERMITTED FOR THE ASYMPTOTIC */
    /*     EXPANSION. NLAST=0 MEANS ALL OF THE Y VALUES WERE SET. */
    /*     NLAST.NE.0 IS THE NUMBER LEFT TO BE COMPUTED BY ANOTHER */
    /*     FORMULA FOR ORDERS FNU TO FNU+NLAST-1 BECAUSE FNU+NLAST-1.LT.FNUL. */
    /*     Y(I)=CZERO FOR I=NLAST+1,N */

    /* ***ROUTINES CALLED  ZAIRY,ZUCHK,ZUNHJ,ZUOIK,D1MACH,ZABS */
    /* ***END PROLOGUE  ZUNI2 */
    /*     COMPLEX AI,ARG,ASUM,BSUM,CFN,CI,CID,CIP,CONE,CRSC,CSCL,CSR,CSS, */
    /*    *CZERO,C1,C2,DAI,PHI,RZ,S1,S2,Y,Z,ZB,ZETA1,ZETA2,ZN */
    /* Parameter adjustments */
    --yi;
    --yr;

    /* Function Body */

    *nz    = 0;
    nd     = *n;
    *nlast = 0;
    /* ----------------------------------------------------------------------- */
    /*     COMPUTED VALUES WITH EXPONENTS BETWEEN ALIM AND ELIM IN MAG- */
    /*     NITUDE ARE SCALED TO KEEP INTERMEDIATE ARITHMETIC ON SCALE, */
    /*     EXP(ALIM)=EXP(ELIM)*TOL */
    /* ----------------------------------------------------------------------- */
    cscl    = 1. / *tol;
    crsc    = *tol;
    cssr[0] = cscl;
    cssr[1] = coner;
    cssr[2] = crsc;
    csrr[0] = crsc;
    csrr[1] = coner;
    csrr[2] = cscl;
    bry[0]  = d1mach_(&c__1) * 1e3 / *tol;
    /* ----------------------------------------------------------------------- */
    /*     ZN IS IN THE RIGHT HALF PLANE AFTER ROTATION BY CI OR -CI */
    /* ----------------------------------------------------------------------- */
    znr  = *zi;
    zni  = -(*zr);
    zbr  = *zr;
    zbi  = *zi;
    cidi = -coner;
    inu  = (int32)((float)(*fnu));
    ang  = hpi * (*fnu - (double)((float)inu));
    c2r  = cos(ang);
    c2i  = sin(ang);
    car  = c2r;
    sar  = c2i;
    in   = inu + *n - 1;
    in   = in % 4 + 1;
    str  = c2r * cipr[in - 1] - c2i * cipi[in - 1];
    c2i  = c2r * cipi[in - 1] + c2i * cipr[in - 1];
    c2r  = str;
    if(*zi > 0.)
    {
        goto L10;
    }
    znr  = -znr;
    zbi  = -zbi;
    cidi = -cidi;
    c2i  = -c2i;
L10:
    /* ----------------------------------------------------------------------- */
    /*     CHECK FOR UNDERFLOW AND OVERFLOW ON FIRST MEMBER */
    /* ----------------------------------------------------------------------- */
    fn = max(*fnu, 1.);
    zunhj_(&znr, &zni, &fn, &c__1, tol, &phir, &phii, &argr, &argi, &zeta1r, &zeta1i, &zeta2r, &zeta2i, &asumr, &asumi, &bsumr, &bsumi);
    if(*kode == 1)
    {
        goto L20;
    }
    str  = zbr + zeta2r;
    sti  = zbi + zeta2i;
    rast = fn / zabs_(&str, &sti);
    str  = str * rast * rast;
    sti  = -sti * rast * rast;
    s1r  = -zeta1r + str;
    s1i  = -zeta1i + sti;
    goto L30;
L20:
    s1r = -zeta1r + zeta2r;
    s1i = -zeta1i + zeta2i;
L30:
    rs1 = s1r;
    if(abs(rs1) > *elim)
    {
        goto L150;
    }
L40:
    nn   = std::min(2L, nd);
    i__1 = nn;
    for(i__ = 1; i__ <= i__1; ++i__)
    {
        fn = *fnu + (double)((float)(nd - i__));
        zunhj_(&znr, &zni, &fn, &c__0, tol, &phir, &phii, &argr, &argi, &zeta1r, &zeta1i, &zeta2r, &zeta2i, &asumr, &asumi, &bsumr, &bsumi);
        if(*kode == 1)
        {
            goto L50;
        }
        str  = zbr + zeta2r;
        sti  = zbi + zeta2i;
        rast = fn / zabs_(&str, &sti);
        str  = str * rast * rast;
        sti  = -sti * rast * rast;
        s1r  = -zeta1r + str;
        s1i  = -zeta1i + sti + abs(*zi);
        goto L60;
    L50:
        s1r = -zeta1r + zeta2r;
        s1i = -zeta1i + zeta2i;
    L60:
        /* ----------------------------------------------------------------------- */
        /*     TEST FOR UNDERFLOW AND OVERFLOW */
        /* ----------------------------------------------------------------------- */
        rs1 = s1r;
        if(abs(rs1) > *elim)
        {
            goto L120;
        }
        if(i__ == 1)
        {
            iflag = 2;
        }
        if(abs(rs1) < *alim)
        {
            goto L70;
        }
        /* ----------------------------------------------------------------------- */
        /*     REFINE  TEST AND SCALE */
        /* ----------------------------------------------------------------------- */
        /* ----------------------------------------------------------------------- */
        aphi = zabs_(&phir, &phii);
        aarg = zabs_(&argr, &argi);
        rs1  = rs1 + log(aphi) - log(aarg) * .25 - aic;
        if(abs(rs1) > *elim)
        {
            goto L120;
        }
        if(i__ == 1)
        {
            iflag = 1;
        }
        if(rs1 < 0.)
        {
            goto L70;
        }
        if(i__ == 1)
        {
            iflag = 3;
        }
    L70:
        /* ----------------------------------------------------------------------- */
        /*     SCALE S1 TO KEEP INTERMEDIATE ARITHMETIC ON SCALE NEAR */
        /*     EXPONENT EXTREMES */
        /* ----------------------------------------------------------------------- */
        zairy_(&argr, &argi, &c__0, &c__2, &air, &aii, &nai, &idum);
        zairy_(&argr, &argi, &c__1, &c__2, &dair, &daii, &ndai, &idum);
        str = dair * bsumr - daii * bsumi;
        sti = dair * bsumi + daii * bsumr;
        str += air * asumr - aii * asumi;
        sti += air * asumi + aii * asumr;
        s2r = phir * str - phii * sti;
        s2i = phir * sti + phii * str;
        str = exp(s1r) * cssr[iflag - 1];
        s1r = str * cos(s1i);
        s1i = str * sin(s1i);
        str = s2r * s1r - s2i * s1i;
        s2i = s2r * s1i + s2i * s1r;
        s2r = str;
        if(iflag != 1)
        {
            goto L80;
        }
        zuchk_(&s2r, &s2i, &nw, bry, tol);
        if(nw != 0)
        {
            goto L120;
        }
    L80:
        if(*zi <= 0.)
        {
            s2i = -s2i;
        }
        str          = s2r * c2r - s2i * c2i;
        s2i          = s2r * c2i + s2i * c2r;
        s2r          = str;
        cyr[i__ - 1] = s2r;
        cyi[i__ - 1] = s2i;
        j            = nd - i__ + 1;
        yr[j]        = s2r * csrr[iflag - 1];
        yi[j]        = s2i * csrr[iflag - 1];
        str          = -c2i * cidi;
        c2i          = c2r * cidi;
        c2r          = str;
        /* L90: */
    }
    if(nd <= 2)
    {
        goto L110;
    }
    raz    = 1. / zabs_(zr, zi);
    str    = *zr * raz;
    sti    = -(*zi) * raz;
    rzr    = (str + str) * raz;
    rzi    = (sti + sti) * raz;
    bry[1] = 1. / bry[0];
    bry[2] = d1mach_(&c__2);
    s1r    = cyr[0];
    s1i    = cyi[0];
    s2r    = cyr[1];
    s2i    = cyi[1];
    c1r    = csrr[iflag - 1];
    ascle  = bry[iflag - 1];
    k      = nd - 2;
    fn     = (double)((float)k);
    i__1   = nd;
    for(i__ = 3; i__ <= i__1; ++i__)
    {
        c2r   = s2r;
        c2i   = s2i;
        s2r   = s1r + (*fnu + fn) * (rzr * c2r - rzi * c2i);
        s2i   = s1i + (*fnu + fn) * (rzr * c2i + rzi * c2r);
        s1r   = c2r;
        s1i   = c2i;
        c2r   = s2r * c1r;
        c2i   = s2i * c1r;
        yr[k] = c2r;
        yi[k] = c2i;
        --k;
        fn += -1.;
        if(iflag >= 3)
        {
            goto L100;
        }
        str = abs(c2r);
        sti = abs(c2i);
        c2m = max(str, sti);
        if(c2m <= ascle)
        {
            goto L100;
        }
        ++iflag;
        ascle = bry[iflag - 1];
        s1r *= c1r;
        s1i *= c1r;
        s2r = c2r;
        s2i = c2i;
        s1r *= cssr[iflag - 1];
        s1i *= cssr[iflag - 1];
        s2r *= cssr[iflag - 1];
        s2i *= cssr[iflag - 1];
        c1r = csrr[iflag - 1];
    L100:;
    }
L110:
    return 0;
L120:
    if(rs1 > 0.)
    {
        goto L140;
    }
    /* ----------------------------------------------------------------------- */
    /*     SET UNDERFLOW AND UPDATE PARAMETERS */
    /* ----------------------------------------------------------------------- */
    yr[nd] = zeror;
    yi[nd] = zeroi;
    ++(*nz);
    --nd;
    if(nd == 0)
    {
        goto L110;
    }
    zuoik_(zr, zi, fnu, kode, &c__1, &nd, &yr[1], &yi[1], &nuf, tol, elim, alim);
    if(nuf < 0)
    {
        goto L140;
    }
    nd -= nuf;
    *nz += nuf;
    if(nd == 0)
    {
        goto L110;
    }
    fn = *fnu + (double)((float)(nd - 1));
    if(fn < *fnul)
    {
        goto L130;
    }
    /*      FN = CIDI */
    /*      J = NUF + 1 */
    /*      K = MOD(J,4) + 1 */
    /*      S1R = CIPR(K) */
    /*      S1I = CIPI(K) */
    /*      IF (FN.LT.0.0D0) S1I = -S1I */
    /*      STR = C2R*S1R - C2I*S1I */
    /*      C2I = C2R*S1I + C2I*S1R */
    /*      C2R = STR */
    in  = inu + nd - 1;
    in  = in % 4 + 1;
    c2r = car * cipr[in - 1] - sar * cipi[in - 1];
    c2i = car * cipi[in - 1] + sar * cipr[in - 1];
    if(*zi <= 0.)
    {
        c2i = -c2i;
    }
    goto L40;
L130:
    *nlast = nd;
    return 0;
L140:
    *nz = -1;
    return 0;
L150:
    if(rs1 > 0.)
    {
        goto L140;
    }
    *nz  = *n;
    i__1 = *n;
    for(i__ = 1; i__ <= i__1; ++i__)
    {
        yr[i__] = zeror;
        yi[i__] = zeroi;
        /* L160: */
    }
    return 0;
} /* zuni2_ */

 int zunik_(double* zrr,
                            double* zri,
                            double* fnu,
                            int32*    ikflg,
                            int32*    ipmtr,
                            double* tol,
                            int32*    init,
                            double* phir,
                            double* phii,
                            double* zeta1r,
                            double* zeta1i,
                            double* zeta2r,
                            double* zeta2i,
                            double* sumr,
                            double* sumi,
                            double* cwrkr,
                            double* cwrki)
{
    /* Initialized data */

    static double zeror    = 0.;
    static double zeroi    = 0.;
    static double coner    = 1.;
    static double conei    = 0.;
    static double con[2]   = {.398942280401432678, 1.25331413731550025};
    static double c__[120] = {1.,
                                  -.208333333333333333,
                                  .125,
                                  .334201388888888889,
                                  -.401041666666666667,
                                  .0703125,
                                  -1.02581259645061728,
                                  1.84646267361111111,
                                  -.8912109375,
                                  .0732421875,
                                  4.66958442342624743,
                                  -11.2070026162229938,
                                  8.78912353515625,
                                  -2.3640869140625,
                                  .112152099609375,
                                  -28.2120725582002449,
                                  84.6362176746007346,
                                  -91.8182415432400174,
                                  42.5349987453884549,
                                  -7.3687943594796317,
                                  .227108001708984375,
                                  212.570130039217123,
                                  -765.252468141181642,
                                  1059.99045252799988,
                                  -699.579627376132541,
                                  218.19051174421159,
                                  -26.4914304869515555,
                                  .572501420974731445,
                                  -1919.457662318407,
                                  8061.72218173730938,
                                  -13586.5500064341374,
                                  11655.3933368645332,
                                  -5305.64697861340311,
                                  1200.90291321635246,
                                  -108.090919788394656,
                                  1.7277275025844574,
                                  20204.2913309661486,
                                  -96980.5983886375135,
                                  192547.001232531532,
                                  -203400.177280415534,
                                  122200.46498301746,
                                  -41192.6549688975513,
                                  7109.51430248936372,
                                  -493.915304773088012,
                                  6.07404200127348304,
                                  -242919.187900551333,
                                  1311763.6146629772,
                                  -2998015.91853810675,
                                  3763271.297656404,
                                  -2813563.22658653411,
                                  1268365.27332162478,
                                  -331645.172484563578,
                                  45218.7689813627263,
                                  -2499.83048181120962,
                                  24.3805296995560639,
                                  3284469.85307203782,
                                  -19706819.1184322269,
                                  50952602.4926646422,
                                  -74105148.2115326577,
                                  66344512.2747290267,
                                  -37567176.6607633513,
                                  13288767.1664218183,
                                  -2785618.12808645469,
                                  308186.404612662398,
                                  -13886.0897537170405,
                                  110.017140269246738,
                                  -49329253.664509962,
                                  325573074.185765749,
                                  -939462359.681578403,
                                  1553596899.57058006,
                                  -1621080552.10833708,
                                  1106842816.82301447,
                                  -495889784.275030309,
                                  142062907.797533095,
                                  -24474062.7257387285,
                                  2243768.17792244943,
                                  -84005.4336030240853,
                                  551.335896122020586,
                                  814789096.118312115,
                                  -5866481492.05184723,
                                  18688207509.2958249,
                                  -34632043388.1587779,
                                  41280185579.753974,
                                  -33026599749.8007231,
                                  17954213731.1556001,
                                  -6563293792.61928433,
                                  1559279864.87925751,
                                  -225105661.889415278,
                                  17395107.5539781645,
                                  -549842.327572288687,
                                  3038.09051092238427,
                                  -14679261247.6956167,
                                  114498237732.02581,
                                  -399096175224.466498,
                                  819218669548.577329,
                                  -1098375156081.22331,
                                  1008158106865.38209,
                                  -645364869245.376503,
                                  287900649906.150589,
                                  -87867072178.0232657,
                                  17634730606.8349694,
                                  -2167164983.22379509,
                                  143157876.718888981,
                                  -3871833.44257261262,
                                  18257.7554742931747,
                                  286464035717.679043,
                                  -2406297900028.50396,
                                  9109341185239.89896,
                                  -20516899410934.4374,
                                  30565125519935.3206,
                                  -31667088584785.1584,
                                  23348364044581.8409,
                                  -12320491305598.2872,
                                  4612725780849.13197,
                                  -1196552880196.1816,
                                  205914503232.410016,
                                  -21822927757.5292237,
                                  1247009293.51271032,
                                  -29188388.1222208134,
                                  118838.426256783253};

    /* System generated locals */
    int32    i__1;
    double d__1, d__2;


    
    int32                     i__, j, k, l;
    double                  ac, si, ti, sr, tr, t2i, t2r, rfn, sri, sti, zni, srr, str, znr;
    int32                     idum;
    double                  test, crfni, crfnr;

    /* *********************************************************************72 */

    /* ***BEGIN PROLOGUE  ZUNIK */
    /* ***REFER TO  ZBESI,ZBESK */

    /*        ZUNIK COMPUTES PARAMETERS FOR THE UNIFORM ASYMPTOTIC */
    /*        EXPANSIONS OF THE I AND K FUNCTIONS ON IKFLG= 1 OR 2 */
    /*        RESPECTIVELY BY */

    /*        W(FNU,ZR) = PHI*EXP(ZETA)*SUM */

    /*        WHERE       ZETA=-ZETA1 + ZETA2       OR */
    /*                          ZETA1 - ZETA2 */

    /*        THE FIRST CALL MUST HAVE INIT=0. SUBSEQUENT CALLS WITH THE */
    /*        SAME ZR AND FNU WILL RETURN THE I OR K FUNCTION ON IKFLG= */
    /*        1 OR 2 WITH NO CHANGE IN INIT. CWRK IS A COMPLEX WORK */
    /*        ARRAY. IPMTR=0 COMPUTES ALL PARAMETERS. IPMTR=1 COMPUTES PHI, */
    /*        ZETA1,ZETA2. */

    /* ***ROUTINES CALLED  ZDIV,ZLOG,ZSQRT,D1MACH */
    /* ***END PROLOGUE  ZUNIK */
    /*     COMPLEX CFN,CON,CONE,CRFN,CWRK,CZERO,PHI,S,SR,SUM,T,T2,ZETA1, */
    /*    *ZETA2,ZN,ZR */
    /* Parameter adjustments */
    --cwrki;
    --cwrkr;

    /* Function Body */

    if(*init != 0)
    {
        goto L40;
    }
    /* ----------------------------------------------------------------------- */
    /*     INITIALIZE ALL VARIABLES */
    /* ----------------------------------------------------------------------- */
    rfn = 1. / *fnu;
    /* ----------------------------------------------------------------------- */
    /*     OVERFLOW TEST (ZR/FNU TOO SMALL) */
    /* ----------------------------------------------------------------------- */
    test = d1mach_(&c__1) * 1e3;
    ac   = *fnu * test;
    if(abs(*zrr) > ac || abs(*zri) > ac)
    {
        goto L15;
    }
    *zeta1r = (d__1 = log(test), abs(d__1)) * 2. + *fnu;
    *zeta1i = 0.;
    *zeta2r = *fnu;
    *zeta2i = 0.;
    *phir   = 1.;
    *phii   = 0.;
    return 0;
L15:
    tr = *zrr * rfn;
    ti = *zri * rfn;
    sr = coner + (tr * tr - ti * ti);
    si = conei + (tr * ti + ti * tr);
    zsqrt_(&sr, &si, &srr, &sri);
    str = coner + srr;
    sti = conei + sri;
    zdiv_(&str, &sti, &tr, &ti, &znr, &zni);
    zlog_(&znr, &zni, &str, &sti, &idum);
    *zeta1r = *fnu * str;
    *zeta1i = *fnu * sti;
    *zeta2r = *fnu * srr;
    *zeta2i = *fnu * sri;
    zdiv_(&coner, &conei, &srr, &sri, &tr, &ti);
    srr = tr * rfn;
    sri = ti * rfn;
    zsqrt_(&srr, &sri, &cwrkr[16], &cwrki[16]);
    *phir = cwrkr[16] * con[*ikflg - 1];
    *phii = cwrki[16] * con[*ikflg - 1];
    if(*ipmtr != 0)
    {
        return 0;
    }
    zdiv_(&coner, &conei, &sr, &si, &t2r, &t2i);
    cwrkr[1] = coner;
    cwrki[1] = conei;
    crfnr    = coner;
    crfni    = conei;
    ac       = 1.;
    l        = 1;
    for(k = 2; k <= 15; ++k)
    {
        sr   = zeror;
        si   = zeroi;
        i__1 = k;
        for(j = 1; j <= i__1; ++j)
        {
            ++l;
            str = sr * t2r - si * t2i + c__[l - 1];
            si  = sr * t2i + si * t2r;
            sr  = str;
            /* L10: */
        }
        str      = crfnr * srr - crfni * sri;
        crfni    = crfnr * sri + crfni * srr;
        crfnr    = str;
        cwrkr[k] = crfnr * sr - crfni * si;
        cwrki[k] = crfnr * si + crfni * sr;
        ac *= rfn;
        test = (d__1 = cwrkr[k], abs(d__1)) + (d__2 = cwrki[k], abs(d__2));
        if(ac < *tol && test < *tol)
        {
            goto L30;
        }
        /* L20: */
    }
    k = 15;
L30:
    *init = k;
L40:
    if(*ikflg == 2)
    {
        goto L60;
    }
    /* ----------------------------------------------------------------------- */
    /*     COMPUTE SUM FOR THE I FUNCTION */
    /* ----------------------------------------------------------------------- */
    sr   = zeror;
    si   = zeroi;
    i__1 = *init;
    for(i__ = 1; i__ <= i__1; ++i__)
    {
        sr += cwrkr[i__];
        si += cwrki[i__];
        /* L50: */
    }
    *sumr = sr;
    *sumi = si;
    *phir = cwrkr[16] * con[0];
    *phii = cwrki[16] * con[0];
    return 0;
L60:
    /* ----------------------------------------------------------------------- */
    /*     COMPUTE SUM FOR THE K FUNCTION */
    /* ----------------------------------------------------------------------- */
    sr   = zeror;
    si   = zeroi;
    tr   = coner;
    i__1 = *init;
    for(i__ = 1; i__ <= i__1; ++i__)
    {
        sr += tr * cwrkr[i__];
        si += tr * cwrki[i__];
        tr = -tr;
        /* L70: */
    }
    *sumr = sr;
    *sumi = si;
    *phir = cwrkr[16] * con[1];
    *phii = cwrki[16] * con[1];
    return 0;
} /* zunik_ */

 int zunk1_(double* zr,
                            double* zi,
                            double* fnu,
                            int32*    kode,
                            int32*    mr,
                            int32*    n,
                            double* yr,
                            double* yi,
                            int32*    nz,
                            double* tol,
                            double* elim,
                            double* alim)
{
    /* Initialized data */

    static double zeror = 0.;
    static double zeroi = 0.;
    static double coner = 1.;
    static double pi    = 3.14159265358979324;

    /* System generated locals */
    int32 i__1;

   
    
    int32                     i__, j, k, m, ib, ic;
    double                  fn;
    int32                     il, kk, nw;
    double                  c1i, c2i, c2m, c1r, c2r, s1i, s2i, rs1, s1r, s2r, ang, asc, cki, fnf;
    int32                     ifn;
    double                  ckr;
    int32                     iuf;
    double                  cyi[2], fmr, csr, sgn;
    int32                     inu;
    double                  bry[3], cyr[2], sti, rzi, zri, str, rzr, zrr, aphi, cscl, phii[2], crsc;
    
    double                  phir[2];
    int32                     init[2];
    double                  csrr[3], cssr[3], rast, sumi[2], razr;
    double                  sumr[2];
    int32                     iflag, kflag;
    double                  ascle;
    int32                     kdflg;
    double                  phidi;
    int32                     ipard;
    double                  csgni, phidr;
    int32                     initd;
    double                  cspni, cwrki[48] /* was [16][3] */, sumdi;
    double                  cspnr, cwrkr[48] /* was [16][3] */, sumdr;
  
    double                  zeta1i[2], zeta2i[2], zet1di, zet2di, zeta1r[2], zeta2r[2], zet1dr, zet2dr;

#define cwrki_ref(a_1, a_2) cwrki[(a_2)*16 + a_1 - 17]
#define cwrkr_ref(a_1, a_2) cwrkr[(a_2)*16 + a_1 - 17]

    /* *********************************************************************72 */

    /* ***BEGIN PROLOGUE  ZUNK1 */
    /* ***REFER TO  ZBESK */

    /*     ZUNK1 COMPUTES K(FNU,Z) AND ITS ANALYTIC CONTINUATION FROM THE */
    /*     RIGHT HALF PLANE TO THE LEFT HALF PLANE BY MEANS OF THE */
    /*     UNIFORM ASYMPTOTIC EXPANSION. */
    /*     MR INDICATES THE DIRECTION OF ROTATION FOR ANALYTIC CONTINUATION. */
    /*     NZ=-1 MEANS AN OVERFLOW WILL OCCUR */

    /* ***ROUTINES CALLED  ZKSCL,ZS1S2,ZUCHK,ZUNIK,D1MACH,ZABS */
    /* ***END PROLOGUE  ZUNK1 */
    /*     COMPLEX CFN,CK,CONE,CRSC,CS,CSCL,CSGN,CSPN,CSR,CSS,CWRK,CY,CZERO, */
    /*    *C1,C2,PHI,PHID,RZ,SUM,SUMD,S1,S2,Y,Z,ZETA1,ZETA1D,ZETA2,ZETA2D,ZR */
    /* Parameter adjustments */
    --yi;
    --yr;

    /* Function Body */

    kdflg = 1;
    *nz   = 0;
    /* ----------------------------------------------------------------------- */
    /*     EXP(-ALIM)=EXP(-ELIM)/TOL=APPROX. ONE PRECISION GREATER THAN */
    /*     THE UNDERFLOW LIMIT */
    /* ----------------------------------------------------------------------- */
    cscl    = 1. / *tol;
    crsc    = *tol;
    cssr[0] = cscl;
    cssr[1] = coner;
    cssr[2] = crsc;
    csrr[0] = crsc;
    csrr[1] = coner;
    csrr[2] = cscl;
    bry[0]  = d1mach_(&c__1) * 1e3 / *tol;
    bry[1]  = 1. / bry[0];
    bry[2]  = d1mach_(&c__2);
    zrr     = *zr;
    zri     = *zi;
    if(*zr >= 0.)
    {
        goto L10;
    }
    zrr = -(*zr);
    zri = -(*zi);
L10:
    j    = 2;
    i__1 = *n;
    for(i__ = 1; i__ <= i__1; ++i__)
    {
        /* ----------------------------------------------------------------------- */
        /*     J FLIP FLOPS BETWEEN 1 AND 2 IN J = 3 - J */
        /* ----------------------------------------------------------------------- */
        j           = 3 - j;
        fn          = *fnu + (double)((float)(i__ - 1));
        init[j - 1] = 0;
        zunik_(&zrr,
               &zri,
               &fn,
               &c__2,
               &c__0,
               tol,
               &init[j - 1],
               &phir[j - 1],
               &phii[j - 1],
               &zeta1r[j - 1],
               &zeta1i[j - 1],
               &zeta2r[j - 1],
               &zeta2i[j - 1],
               &sumr[j - 1],
               &sumi[j - 1],
               &cwrkr_ref(1, j),
               &cwrki_ref(1, j));
        if(*kode == 1)
        {
            goto L20;
        }
        str  = zrr + zeta2r[j - 1];
        sti  = zri + zeta2i[j - 1];
        rast = fn / zabs_(&str, &sti);
        str  = str * rast * rast;
        sti  = -sti * rast * rast;
        s1r  = zeta1r[j - 1] - str;
        s1i  = zeta1i[j - 1] - sti;
        goto L30;
    L20:
        s1r = zeta1r[j - 1] - zeta2r[j - 1];
        s1i = zeta1i[j - 1] - zeta2i[j - 1];
    L30:
        rs1 = s1r;
        /* ----------------------------------------------------------------------- */
        /*     TEST FOR UNDERFLOW AND OVERFLOW */
        /* ----------------------------------------------------------------------- */
        if(abs(rs1) > *elim)
        {
            goto L60;
        }
        if(kdflg == 1)
        {
            kflag = 2;
        }
        if(abs(rs1) < *alim)
        {
            goto L40;
        }
        /* ----------------------------------------------------------------------- */
        /*     REFINE  TEST AND SCALE */
        /* ----------------------------------------------------------------------- */
        aphi = zabs_(&phir[j - 1], &phii[j - 1]);
        rs1 += log(aphi);
        if(abs(rs1) > *elim)
        {
            goto L60;
        }
        if(kdflg == 1)
        {
            kflag = 1;
        }
        if(rs1 < 0.)
        {
            goto L40;
        }
        if(kdflg == 1)
        {
            kflag = 3;
        }
    L40:
        /* ----------------------------------------------------------------------- */
        /*     SCALE S1 TO KEEP INTERMEDIATE ARITHMETIC ON SCALE NEAR */
        /*     EXPONENT EXTREMES */
        /* ----------------------------------------------------------------------- */
        s2r = phir[j - 1] * sumr[j - 1] - phii[j - 1] * sumi[j - 1];
        s2i = phir[j - 1] * sumi[j - 1] + phii[j - 1] * sumr[j - 1];
        str = exp(s1r) * cssr[kflag - 1];
        s1r = str * cos(s1i);
        s1i = str * sin(s1i);
        str = s2r * s1r - s2i * s1i;
        s2i = s1r * s2i + s2r * s1i;
        s2r = str;
        if(kflag != 1)
        {
            goto L50;
        }
        zuchk_(&s2r, &s2i, &nw, bry, tol);
        if(nw != 0)
        {
            goto L60;
        }
    L50:
        cyr[kdflg - 1] = s2r;
        cyi[kdflg - 1] = s2i;
        yr[i__]        = s2r * csrr[kflag - 1];
        yi[i__]        = s2i * csrr[kflag - 1];
        if(kdflg == 2)
        {
            goto L75;
        }
        kdflg = 2;
        goto L70;
    L60:
        if(rs1 > 0.)
        {
            goto L300;
        }
        /* ----------------------------------------------------------------------- */
        /*     FOR ZR.LT.0.0, THE I FUNCTION TO BE ADDED WILL OVERFLOW */
        /* ----------------------------------------------------------------------- */
        if(*zr < 0.)
        {
            goto L300;
        }
        kdflg   = 1;
        yr[i__] = zeror;
        yi[i__] = zeroi;
        ++(*nz);
        if(i__ == 1)
        {
            goto L70;
        }
        if(yr[i__ - 1] == zeror && yi[i__ - 1] == zeroi)
        {
            goto L70;
        }
        yr[i__ - 1] = zeror;
        yi[i__ - 1] = zeroi;
        ++(*nz);
    L70:;
    }
    i__ = *n;
L75:
    razr = 1. / zabs_(&zrr, &zri);
    str  = zrr * razr;
    sti  = -zri * razr;
    rzr  = (str + str) * razr;
    rzi  = (sti + sti) * razr;
    ckr  = fn * rzr;
    cki  = fn * rzi;
    ib   = i__ + 1;
    if(*n < ib)
    {
        goto L160;
    }
    /* ----------------------------------------------------------------------- */
    /*     TEST LAST MEMBER FOR UNDERFLOW AND OVERFLOW. SET SEQUENCE TO ZERO */
    /*     ON UNDERFLOW. */
    /* ----------------------------------------------------------------------- */
    fn    = *fnu + (double)((float)(*n - 1));
    ipard = 1;
    if(*mr != 0)
    {
        ipard = 0;
    }
    initd = 0;
    zunik_(&zrr, &zri, &fn, &c__2, &ipard, tol, &initd, &phidr, &phidi, &zet1dr, &zet1di, &zet2dr, &zet2di, &sumdr, &sumdi, &cwrkr_ref(1, 3), &cwrki_ref(1, 3));
    if(*kode == 1)
    {
        goto L80;
    }
    str  = zrr + zet2dr;
    sti  = zri + zet2di;
    rast = fn / zabs_(&str, &sti);
    str  = str * rast * rast;
    sti  = -sti * rast * rast;
    s1r  = zet1dr - str;
    s1i  = zet1di - sti;
    goto L90;
L80:
    s1r = zet1dr - zet2dr;
    s1i = zet1di - zet2di;
L90:
    rs1 = s1r;
    if(abs(rs1) > *elim)
    {
        goto L95;
    }
    if(abs(rs1) < *alim)
    {
        goto L100;
    }
    /* ---------------------------------------------------------------------------- */
    /*     REFINE ESTIMATE AND TEST */
    /* ------------------------------------------------------------------------- */
    aphi = zabs_(&phidr, &phidi);
    rs1 += log(aphi);
    if(abs(rs1) < *elim)
    {
        goto L100;
    }
L95:
    if(abs(rs1) > 0.)
    {
        goto L300;
    }
    /* ----------------------------------------------------------------------- */
    /*     FOR ZR.LT.0.0, THE I FUNCTION TO BE ADDED WILL OVERFLOW */
    /* ----------------------------------------------------------------------- */
    if(*zr < 0.)
    {
        goto L300;
    }
    *nz  = *n;
    i__1 = *n;
    for(i__ = 1; i__ <= i__1; ++i__)
    {
        yr[i__] = zeror;
        yi[i__] = zeroi;
        /* L96: */
    }
    return 0;
/* --------------------------------------------------------------------------- */
/*     FORWARD RECUR FOR REMAINDER OF THE SEQUENCE */
/* ---------------------------------------------------------------------------- */
L100:
    s1r   = cyr[0];
    s1i   = cyi[0];
    s2r   = cyr[1];
    s2i   = cyi[1];
    c1r   = csrr[kflag - 1];
    ascle = bry[kflag - 1];
    i__1  = *n;
    for(i__ = ib; i__ <= i__1; ++i__)
    {
        c2r = s2r;
        c2i = s2i;
        s2r = ckr * c2r - cki * c2i + s1r;
        s2i = ckr * c2i + cki * c2r + s1i;
        s1r = c2r;
        s1i = c2i;
        ckr += rzr;
        cki += rzi;
        c2r     = s2r * c1r;
        c2i     = s2i * c1r;
        yr[i__] = c2r;
        yi[i__] = c2i;
        if(kflag >= 3)
        {
            goto L120;
        }
        str = abs(c2r);
        sti = abs(c2i);
        c2m = max(str, sti);
        if(c2m <= ascle)
        {
            goto L120;
        }
        ++kflag;
        ascle = bry[kflag - 1];
        s1r *= c1r;
        s1i *= c1r;
        s2r = c2r;
        s2i = c2i;
        s1r *= cssr[kflag - 1];
        s1i *= cssr[kflag - 1];
        s2r *= cssr[kflag - 1];
        s2i *= cssr[kflag - 1];
        c1r = csrr[kflag - 1];
    L120:;
    }
L160:
    if(*mr == 0)
    {
        return 0;
    }
    /* ----------------------------------------------------------------------- */
    /*     ANALYTIC CONTINUATION FOR RE(Z).LT.0.0D0 */
    /* ----------------------------------------------------------------------- */
    *nz = 0;
    fmr = (double)((float)(*mr));
    sgn = -d_sign(&pi, &fmr);
    /* ----------------------------------------------------------------------- */
    /*     CSPN AND CSGN ARE COEFF OF K AND I FUNCTIONS RESP. */
    /* ----------------------------------------------------------------------- */
    csgni = sgn;
    inu   = (int32)((float)(*fnu));
    fnf   = *fnu - (double)((float)inu);
    ifn   = inu + *n - 1;
    ang   = fnf * sgn;
    cspnr = cos(ang);
    cspni = sin(ang);
    if(ifn % 2 == 0)
    {
        goto L170;
    }
    cspnr = -cspnr;
    cspni = -cspni;
L170:
    asc   = bry[0];
    iuf   = 0;
    kk    = *n;
    kdflg = 1;
    --ib;
    ic   = ib - 1;
    i__1 = *n;
    for(k = 1; k <= i__1; ++k)
    {
        fn = *fnu + (double)((float)(kk - 1));
        /* ----------------------------------------------------------------------- */
        /*     LOGIC TO SORT OUT CASES WHOSE PARAMETERS WERE SET FOR THE K */
        /*     FUNCTION ABOVE */
        /* ----------------------------------------------------------------------- */
        m = 3;
        if(*n > 2)
        {
            goto L175;
        }
    L172:
        initd  = init[j - 1];
        phidr  = phir[j - 1];
        phidi  = phii[j - 1];
        zet1dr = zeta1r[j - 1];
        zet1di = zeta1i[j - 1];
        zet2dr = zeta2r[j - 1];
        zet2di = zeta2i[j - 1];
        sumdr  = sumr[j - 1];
        sumdi  = sumi[j - 1];
        m      = j;
        j      = 3 - j;
        goto L180;
    L175:
        if(kk == *n && ib < *n)
        {
            goto L180;
        }
        if(kk == ib || kk == ic)
        {
            goto L172;
        }
        initd = 0;
    L180:
        zunik_(&zrr, &zri, &fn, &c__1, &c__0, tol, &initd, &phidr, &phidi, &zet1dr, &zet1di, &zet2dr, &zet2di, &sumdr, &sumdi, &cwrkr_ref(1, m), &cwrki_ref(1, m));
        if(*kode == 1)
        {
            goto L200;
        }
        str  = zrr + zet2dr;
        sti  = zri + zet2di;
        rast = fn / zabs_(&str, &sti);
        str  = str * rast * rast;
        sti  = -sti * rast * rast;
        s1r  = -zet1dr + str;
        s1i  = -zet1di + sti;
        goto L210;
    L200:
        s1r = -zet1dr + zet2dr;
        s1i = -zet1di + zet2di;
    L210:
        /* ----------------------------------------------------------------------- */
        /*     TEST FOR UNDERFLOW AND OVERFLOW */
        /* ----------------------------------------------------------------------- */
        rs1 = s1r;
        if(abs(rs1) > *elim)
        {
            goto L260;
        }
        if(kdflg == 1)
        {
            iflag = 2;
        }
        if(abs(rs1) < *alim)
        {
            goto L220;
        }
        /* ----------------------------------------------------------------------- */
        /*     REFINE  TEST AND SCALE */
        /* ----------------------------------------------------------------------- */
        aphi = zabs_(&phidr, &phidi);
        rs1 += log(aphi);
        if(abs(rs1) > *elim)
        {
            goto L260;
        }
        if(kdflg == 1)
        {
            iflag = 1;
        }
        if(rs1 < 0.)
        {
            goto L220;
        }
        if(kdflg == 1)
        {
            iflag = 3;
        }
    L220:
        str = phidr * sumdr - phidi * sumdi;
        sti = phidr * sumdi + phidi * sumdr;
        s2r = -csgni * sti;
        s2i = csgni * str;
        str = exp(s1r) * cssr[iflag - 1];
        s1r = str * cos(s1i);
        s1i = str * sin(s1i);
        str = s2r * s1r - s2i * s1i;
        s2i = s2r * s1i + s2i * s1r;
        s2r = str;
        if(iflag != 1)
        {
            goto L230;
        }
        zuchk_(&s2r, &s2i, &nw, bry, tol);
        if(nw == 0)
        {
            goto L230;
        }
        s2r = zeror;
        s2i = zeroi;
    L230:
        cyr[kdflg - 1] = s2r;
        cyi[kdflg - 1] = s2i;
        c2r            = s2r;
        c2i            = s2i;
        s2r *= csrr[iflag - 1];
        s2i *= csrr[iflag - 1];
        /* ----------------------------------------------------------------------- */
        /*     ADD I AND K FUNCTIONS, K SEQUENCE IN Y(I), I=1,N */
        /* ----------------------------------------------------------------------- */
        s1r = yr[kk];
        s1i = yi[kk];
        if(*kode == 1)
        {
            goto L250;
        }
        zs1s2_(&zrr, &zri, &s1r, &s1i, &s2r, &s2i, &nw, &asc, alim, &iuf);
        *nz += nw;
    L250:
        yr[kk] = s1r * cspnr - s1i * cspni + s2r;
        yi[kk] = cspnr * s1i + cspni * s1r + s2i;
        --kk;
        cspnr = -cspnr;
        cspni = -cspni;
        if(c2r != 0. || c2i != 0.)
        {
            goto L255;
        }
        kdflg = 1;
        goto L270;
    L255:
        if(kdflg == 2)
        {
            goto L275;
        }
        kdflg = 2;
        goto L270;
    L260:
        if(rs1 > 0.)
        {
            goto L300;
        }
        s2r = zeror;
        s2i = zeroi;
        goto L230;
    L270:;
    }
    k = *n;
L275:
    il = *n - k;
    if(il == 0)
    {
        return 0;
    }
    /* ----------------------------------------------------------------------- */
    /*     RECUR BACKWARD FOR REMAINDER OF I SEQUENCE AND ADD IN THE */
    /*     K FUNCTIONS, SCALING THE I SEQUENCE DURING RECURRENCE TO KEEP */
    /*     INTERMEDIATE ARITHMETIC ON SCALE NEAR EXPONENT EXTREMES. */
    /* ----------------------------------------------------------------------- */
    s1r   = cyr[0];
    s1i   = cyi[0];
    s2r   = cyr[1];
    s2i   = cyi[1];
    csr   = csrr[iflag - 1];
    ascle = bry[iflag - 1];
    fn    = (double)((float)(inu + il));
    i__1  = il;
    for(i__ = 1; i__ <= i__1; ++i__)
    {
        c2r = s2r;
        c2i = s2i;
        s2r = s1r + (fn + fnf) * (rzr * c2r - rzi * c2i);
        s2i = s1i + (fn + fnf) * (rzr * c2i + rzi * c2r);
        s1r = c2r;
        s1i = c2i;
        fn += -1.;
        c2r = s2r * csr;
        c2i = s2i * csr;
        ckr = c2r;
        cki = c2i;
        c1r = yr[kk];
        c1i = yi[kk];
        if(*kode == 1)
        {
            goto L280;
        }
        zs1s2_(&zrr, &zri, &c1r, &c1i, &c2r, &c2i, &nw, &asc, alim, &iuf);
        *nz += nw;
    L280:
        yr[kk] = c1r * cspnr - c1i * cspni + c2r;
        yi[kk] = c1r * cspni + c1i * cspnr + c2i;
        --kk;
        cspnr = -cspnr;
        cspni = -cspni;
        if(iflag >= 3)
        {
            goto L290;
        }
        c2r = abs(ckr);
        c2i = abs(cki);
        c2m = max(c2r, c2i);
        if(c2m <= ascle)
        {
            goto L290;
        }
        ++iflag;
        ascle = bry[iflag - 1];
        s1r *= csr;
        s1i *= csr;
        s2r = ckr;
        s2i = cki;
        s1r *= cssr[iflag - 1];
        s1i *= cssr[iflag - 1];
        s2r *= cssr[iflag - 1];
        s2i *= cssr[iflag - 1];
        csr = csrr[iflag - 1];
    L290:;
    }
    return 0;
L300:
    *nz = -1;
    return 0;
} /* zunk1_ */

#undef cwrkr_ref
#undef cwrki_ref

 int zunk2_(double* zr,
                            double* zi,
                            double* fnu,
                            int32*    kode,
                            int32*    mr,
                            int32*    n,
                            double* yr,
                            double* yi,
                            int32*    nz,
                            double* tol,
                            double* elim,
                            double* alim)
{
    /* Initialized data */

    static double zeror   = 0.;
    static double zeroi   = 0.;
    static double coner   = 1.;
    static double cr1r    = 1.;
    static double cr1i    = 1.73205080756887729;
    static double cr2r    = -.5;
    static double cr2i    = -.866025403784438647;
    static double hpi     = 1.57079632679489662;
    static double pi      = 3.14159265358979324;
    static double aic     = 1.26551212348464539;
    static double cipr[4] = {1., 0., -1., 0.};
    static double cipi[4] = {0., -1., 0., 1.};

    /* System generated locals */
    int32 i__1;


    
    int32                     i__, j, k, ib, ic;
    double                  fn;
    int32                     il, kk, in, nw;
    double                  yy, c1i, c2i, c2m, c1r, c2r, s1i, s2i, rs1, s1r, s2r, aii, ang, asc, car, cki, fnf;
    int32                     nai;
    double                  air;
    int32                     ifn;
    double                  csi, ckr;
    int32                     iuf;
    double                  cyi[2], fmr, sar, csr, sgn, zbi;
    int32                     inu;
    double                  bry[3], cyr[2], pti, sti, zbr, zni, rzi, ptr, zri, str, znr, rzr, zrr, daii, aarg;
    int32                     ndai;
    double                  dair, aphi, argi[2], cscl, phii[2], crsc, argr[2];
    int32                     idum;
    
    double                  phir[2], csrr[3], cssr[3], rast, razr;
    int32                     iflag, kflag;
    double                  argdi, ascle;
    int32                     kdflg;
    double                  phidi, argdr;
    int32                     ipard;
    double                  csgni, phidr, cspni, asumi[2], bsumi[2];
    double                  cspnr, asumr[2], bsumr[2];
 
    double zeta1i[2], zeta2i[2], zet1di, zet2di, zeta1r[2], zeta2r[2], zet1dr, zet2dr, asumdi, bsumdi, asumdr, bsumdr;

    /* *********************************************************************72 */

    /* ***BEGIN PROLOGUE  ZUNK2 */
    /* ***REFER TO  ZBESK */

    /*     ZUNK2 COMPUTES K(FNU,Z) AND ITS ANALYTIC CONTINUATION FROM THE */
    /*     RIGHT HALF PLANE TO THE LEFT HALF PLANE BY MEANS OF THE */
    /*     UNIFORM ASYMPTOTIC EXPANSIONS FOR H(KIND,FNU,ZN) AND J(FNU,ZN) */
    /*     WHERE ZN IS IN THE RIGHT HALF PLANE, KIND=(3-MR)/2, MR=+1 OR */
    /*     -1. HERE ZN=ZR*I OR -ZR*I WHERE ZR=Z IF Z IS IN THE RIGHT */
    /*     HALF PLANE OR ZR=-Z IF Z IS IN THE LEFT HALF PLANE. MR INDIC- */
    /*     ATES THE DIRECTION OF ROTATION FOR ANALYTIC CONTINUATION. */
    /*     NZ=-1 MEANS AN OVERFLOW WILL OCCUR */

    /* ***ROUTINES CALLED  ZAIRY,ZKSCL,ZS1S2,ZUCHK,ZUNHJ,D1MACH,ZABS */
    /* ***END PROLOGUE  ZUNK2 */
    /*     COMPLEX AI,ARG,ARGD,ASUM,ASUMD,BSUM,BSUMD,CFN,CI,CIP,CK,CONE,CRSC, */
    /*    *CR1,CR2,CS,CSCL,CSGN,CSPN,CSR,CSS,CY,CZERO,C1,C2,DAI,PHI,PHID,RZ, */
    /*    *S1,S2,Y,Z,ZB,ZETA1,ZETA1D,ZETA2,ZETA2D,ZN,ZR */
    /* Parameter adjustments */
    --yi;
    --yr;

    /* Function Body */

    kdflg = 1;
    *nz   = 0;
    /* ----------------------------------------------------------------------- */
    /*     EXP(-ALIM)=EXP(-ELIM)/TOL=APPROX. ONE PRECISION GREATER THAN */
    /*     THE UNDERFLOW LIMIT */
    /* ----------------------------------------------------------------------- */
    cscl    = 1. / *tol;
    crsc    = *tol;
    cssr[0] = cscl;
    cssr[1] = coner;
    cssr[2] = crsc;
    csrr[0] = crsc;
    csrr[1] = coner;
    csrr[2] = cscl;
    bry[0]  = d1mach_(&c__1) * 1e3 / *tol;
    bry[1]  = 1. / bry[0];
    bry[2]  = d1mach_(&c__2);
    zrr     = *zr;
    zri     = *zi;
    if(*zr >= 0.)
    {
        goto L10;
    }
    zrr = -(*zr);
    zri = -(*zi);
L10:
    yy  = zri;
    znr = zri;
    zni = -zrr;
    zbr = zrr;
    zbi = zri;
    inu = (int32)((float)(*fnu));
    fnf = *fnu - (double)((float)inu);
    ang = -hpi * fnf;
    car = cos(ang);
    sar = sin(ang);
    c2r = hpi * sar;
    c2i = -hpi * car;
    kk  = inu % 4 + 1;
    str = c2r * cipr[kk - 1] - c2i * cipi[kk - 1];
    sti = c2r * cipi[kk - 1] + c2i * cipr[kk - 1];
    csr = cr1r * str - cr1i * sti;
    csi = cr1r * sti + cr1i * str;
    if(yy > 0.)
    {
        goto L20;
    }
    znr = -znr;
    zbi = -zbi;
L20:
    /* ----------------------------------------------------------------------- */
    /*     K(FNU,Z) IS COMPUTED FROM H(2,FNU,-I*Z) WHERE Z IS IN THE FIRST */
    /*     QUADRANT. FOURTH QUADRANT VALUES (YY.LE.0.0E0) ARE COMPUTED BY */
    /*     CONJUGATION SINCE THE K FUNCTION IS REAL ON THE POSITIVE REAL AXIS */
    /* ----------------------------------------------------------------------- */
    j    = 2;
    i__1 = *n;
    for(i__ = 1; i__ <= i__1; ++i__)
    {
        /* ----------------------------------------------------------------------- */
        /*     J FLIP FLOPS BETWEEN 1 AND 2 IN J = 3 - J */
        /* ----------------------------------------------------------------------- */
        j  = 3 - j;
        fn = *fnu + (double)((float)(i__ - 1));
        zunhj_(&znr,
               &zni,
               &fn,
               &c__0,
               tol,
               &phir[j - 1],
               &phii[j - 1],
               &argr[j - 1],
               &argi[j - 1],
               &zeta1r[j - 1],
               &zeta1i[j - 1],
               &zeta2r[j - 1],
               &zeta2i[j - 1],
               &asumr[j - 1],
               &asumi[j - 1],
               &bsumr[j - 1],
               &bsumi[j - 1]);
        if(*kode == 1)
        {
            goto L30;
        }
        str  = zbr + zeta2r[j - 1];
        sti  = zbi + zeta2i[j - 1];
        rast = fn / zabs_(&str, &sti);
        str  = str * rast * rast;
        sti  = -sti * rast * rast;
        s1r  = zeta1r[j - 1] - str;
        s1i  = zeta1i[j - 1] - sti;
        goto L40;
    L30:
        s1r = zeta1r[j - 1] - zeta2r[j - 1];
        s1i = zeta1i[j - 1] - zeta2i[j - 1];
    L40:
        /* ----------------------------------------------------------------------- */
        /*     TEST FOR UNDERFLOW AND OVERFLOW */
        /* ----------------------------------------------------------------------- */
        rs1 = s1r;
        if(abs(rs1) > *elim)
        {
            goto L70;
        }
        if(kdflg == 1)
        {
            kflag = 2;
        }
        if(abs(rs1) < *alim)
        {
            goto L50;
        }
        /* ----------------------------------------------------------------------- */
        /*     REFINE  TEST AND SCALE */
        /* ----------------------------------------------------------------------- */
        aphi = zabs_(&phir[j - 1], &phii[j - 1]);
        aarg = zabs_(&argr[j - 1], &argi[j - 1]);
        rs1  = rs1 + log(aphi) - log(aarg) * .25 - aic;
        if(abs(rs1) > *elim)
        {
            goto L70;
        }
        if(kdflg == 1)
        {
            kflag = 1;
        }
        if(rs1 < 0.)
        {
            goto L50;
        }
        if(kdflg == 1)
        {
            kflag = 3;
        }
    L50:
        /* ----------------------------------------------------------------------- */
        /*     SCALE S1 TO KEEP INTERMEDIATE ARITHMETIC ON SCALE NEAR */
        /*     EXPONENT EXTREMES */
        /* ----------------------------------------------------------------------- */
        c2r = argr[j - 1] * cr2r - argi[j - 1] * cr2i;
        c2i = argr[j - 1] * cr2i + argi[j - 1] * cr2r;
        zairy_(&c2r, &c2i, &c__0, &c__2, &air, &aii, &nai, &idum);
        zairy_(&c2r, &c2i, &c__1, &c__2, &dair, &daii, &ndai, &idum);
        str = dair * bsumr[j - 1] - daii * bsumi[j - 1];
        sti = dair * bsumi[j - 1] + daii * bsumr[j - 1];
        ptr = str * cr2r - sti * cr2i;
        pti = str * cr2i + sti * cr2r;
        str = ptr + (air * asumr[j - 1] - aii * asumi[j - 1]);
        sti = pti + (air * asumi[j - 1] + aii * asumr[j - 1]);
        ptr = str * phir[j - 1] - sti * phii[j - 1];
        pti = str * phii[j - 1] + sti * phir[j - 1];
        s2r = ptr * csr - pti * csi;
        s2i = ptr * csi + pti * csr;
        str = exp(s1r) * cssr[kflag - 1];
        s1r = str * cos(s1i);
        s1i = str * sin(s1i);
        str = s2r * s1r - s2i * s1i;
        s2i = s1r * s2i + s2r * s1i;
        s2r = str;
        if(kflag != 1)
        {
            goto L60;
        }
        zuchk_(&s2r, &s2i, &nw, bry, tol);
        if(nw != 0)
        {
            goto L70;
        }
    L60:
        if(yy <= 0.)
        {
            s2i = -s2i;
        }
        cyr[kdflg - 1] = s2r;
        cyi[kdflg - 1] = s2i;
        yr[i__]        = s2r * csrr[kflag - 1];
        yi[i__]        = s2i * csrr[kflag - 1];
        str            = csi;
        csi            = -csr;
        csr            = str;
        if(kdflg == 2)
        {
            goto L85;
        }
        kdflg = 2;
        goto L80;
    L70:
        if(rs1 > 0.)
        {
            goto L320;
        }
        /* ----------------------------------------------------------------------- */
        /*     FOR ZR.LT.0.0, THE I FUNCTION TO BE ADDED WILL OVERFLOW */
        /* ----------------------------------------------------------------------- */
        if(*zr < 0.)
        {
            goto L320;
        }
        kdflg   = 1;
        yr[i__] = zeror;
        yi[i__] = zeroi;
        ++(*nz);
        str = csi;
        csi = -csr;
        csr = str;
        if(i__ == 1)
        {
            goto L80;
        }
        if(yr[i__ - 1] == zeror && yi[i__ - 1] == zeroi)
        {
            goto L80;
        }
        yr[i__ - 1] = zeror;
        yi[i__ - 1] = zeroi;
        ++(*nz);
    L80:;
    }
    i__ = *n;
L85:
    razr = 1. / zabs_(&zrr, &zri);
    str  = zrr * razr;
    sti  = -zri * razr;
    rzr  = (str + str) * razr;
    rzi  = (sti + sti) * razr;
    ckr  = fn * rzr;
    cki  = fn * rzi;
    ib   = i__ + 1;
    if(*n < ib)
    {
        goto L180;
    }
    /* ----------------------------------------------------------------------- */
    /*     TEST LAST MEMBER FOR UNDERFLOW AND OVERFLOW. SET SEQUENCE TO ZERO */
    /*     ON UNDERFLOW. */
    /* ----------------------------------------------------------------------- */
    fn    = *fnu + (double)((float)(*n - 1));
    ipard = 1;
    if(*mr != 0)
    {
        ipard = 0;
    }
    zunhj_(&znr, &zni, &fn, &ipard, tol, &phidr, &phidi, &argdr, &argdi, &zet1dr, &zet1di, &zet2dr, &zet2di, &asumdr, &asumdi, &bsumdr, &bsumdi);
    if(*kode == 1)
    {
        goto L90;
    }
    str  = zbr + zet2dr;
    sti  = zbi + zet2di;
    rast = fn / zabs_(&str, &sti);
    str  = str * rast * rast;
    sti  = -sti * rast * rast;
    s1r  = zet1dr - str;
    s1i  = zet1di - sti;
    goto L100;
L90:
    s1r = zet1dr - zet2dr;
    s1i = zet1di - zet2di;
L100:
    rs1 = s1r;
    if(abs(rs1) > *elim)
    {
        goto L105;
    }
    if(abs(rs1) < *alim)
    {
        goto L120;
    }
    /* ---------------------------------------------------------------------------- */
    /*     REFINE ESTIMATE AND TEST */
    /* ------------------------------------------------------------------------- */
    aphi = zabs_(&phidr, &phidi);
    rs1 += log(aphi);
    if(abs(rs1) < *elim)
    {
        goto L120;
    }
L105:
    if(rs1 > 0.)
    {
        goto L320;
    }
    /* ----------------------------------------------------------------------- */
    /*     FOR ZR.LT.0.0, THE I FUNCTION TO BE ADDED WILL OVERFLOW */
    /* ----------------------------------------------------------------------- */
    if(*zr < 0.)
    {
        goto L320;
    }
    *nz  = *n;
    i__1 = *n;
    for(i__ = 1; i__ <= i__1; ++i__)
    {
        yr[i__] = zeror;
        yi[i__] = zeroi;
        /* L106: */
    }
    return 0;
L120:
    s1r   = cyr[0];
    s1i   = cyi[0];
    s2r   = cyr[1];
    s2i   = cyi[1];
    c1r   = csrr[kflag - 1];
    ascle = bry[kflag - 1];
    i__1  = *n;
    for(i__ = ib; i__ <= i__1; ++i__)
    {
        c2r = s2r;
        c2i = s2i;
        s2r = ckr * c2r - cki * c2i + s1r;
        s2i = ckr * c2i + cki * c2r + s1i;
        s1r = c2r;
        s1i = c2i;
        ckr += rzr;
        cki += rzi;
        c2r     = s2r * c1r;
        c2i     = s2i * c1r;
        yr[i__] = c2r;
        yi[i__] = c2i;
        if(kflag >= 3)
        {
            goto L130;
        }
        str = abs(c2r);
        sti = abs(c2i);
        c2m = max(str, sti);
        if(c2m <= ascle)
        {
            goto L130;
        }
        ++kflag;
        ascle = bry[kflag - 1];
        s1r *= c1r;
        s1i *= c1r;
        s2r = c2r;
        s2i = c2i;
        s1r *= cssr[kflag - 1];
        s1i *= cssr[kflag - 1];
        s2r *= cssr[kflag - 1];
        s2i *= cssr[kflag - 1];
        c1r = csrr[kflag - 1];
    L130:;
    }
L180:
    if(*mr == 0)
    {
        return 0;
    }
    /* ----------------------------------------------------------------------- */
    /*     ANALYTIC CONTINUATION FOR RE(Z).LT.0.0D0 */
    /* ----------------------------------------------------------------------- */
    *nz = 0;
    fmr = (double)((float)(*mr));
    sgn = -d_sign(&pi, &fmr);
    /* ----------------------------------------------------------------------- */
    /*     CSPN AND CSGN ARE COEFF OF K AND I FUNCIONS RESP. */
    /* ----------------------------------------------------------------------- */
    csgni = sgn;
    if(yy <= 0.)
    {
        csgni = -csgni;
    }
    ifn   = inu + *n - 1;
    ang   = fnf * sgn;
    cspnr = cos(ang);
    cspni = sin(ang);
    if(ifn % 2 == 0)
    {
        goto L190;
    }
    cspnr = -cspnr;
    cspni = -cspni;
L190:
    /* ----------------------------------------------------------------------- */
    /*     CS=COEFF OF THE J FUNCTION TO GET THE I FUNCTION. I(FNU,Z) IS */
    /*     COMPUTED FROM EXP(I*FNU*HPI)*J(FNU,-I*Z) WHERE Z IS IN THE FIRST */
    /*     QUADRANT. FOURTH QUADRANT VALUES (YY.LE.0.0E0) ARE COMPUTED BY */
    /*     CONJUGATION SINCE THE I FUNCTION IS REAL ON THE POSITIVE REAL AXIS */
    /* ----------------------------------------------------------------------- */
    csr   = sar * csgni;
    csi   = car * csgni;
    in    = ifn % 4 + 1;
    c2r   = cipr[in - 1];
    c2i   = cipi[in - 1];
    str   = csr * c2r + csi * c2i;
    csi   = -csr * c2i + csi * c2r;
    csr   = str;
    asc   = bry[0];
    iuf   = 0;
    kk    = *n;
    kdflg = 1;
    --ib;
    ic   = ib - 1;
    i__1 = *n;
    for(k = 1; k <= i__1; ++k)
    {
        fn = *fnu + (double)((float)(kk - 1));
        /* ----------------------------------------------------------------------- */
        /*     LOGIC TO SORT OUT CASES WHOSE PARAMETERS WERE SET FOR THE K */
        /*     FUNCTION ABOVE */
        /* ----------------------------------------------------------------------- */
        if(*n > 2)
        {
            goto L175;
        }
    L172:
        phidr  = phir[j - 1];
        phidi  = phii[j - 1];
        argdr  = argr[j - 1];
        argdi  = argi[j - 1];
        zet1dr = zeta1r[j - 1];
        zet1di = zeta1i[j - 1];
        zet2dr = zeta2r[j - 1];
        zet2di = zeta2i[j - 1];
        asumdr = asumr[j - 1];
        asumdi = asumi[j - 1];
        bsumdr = bsumr[j - 1];
        bsumdi = bsumi[j - 1];
        j      = 3 - j;
        goto L210;
    L175:
        if(kk == *n && ib < *n)
        {
            goto L210;
        }
        if(kk == ib || kk == ic)
        {
            goto L172;
        }
        zunhj_(&znr, &zni, &fn, &c__0, tol, &phidr, &phidi, &argdr, &argdi, &zet1dr, &zet1di, &zet2dr, &zet2di, &asumdr, &asumdi, &bsumdr, &bsumdi);
    L210:
        if(*kode == 1)
        {
            goto L220;
        }
        str  = zbr + zet2dr;
        sti  = zbi + zet2di;
        rast = fn / zabs_(&str, &sti);
        str  = str * rast * rast;
        sti  = -sti * rast * rast;
        s1r  = -zet1dr + str;
        s1i  = -zet1di + sti;
        goto L230;
    L220:
        s1r = -zet1dr + zet2dr;
        s1i = -zet1di + zet2di;
    L230:
        /* ----------------------------------------------------------------------- */
        /*     TEST FOR UNDERFLOW AND OVERFLOW */
        /* ----------------------------------------------------------------------- */
        rs1 = s1r;
        if(abs(rs1) > *elim)
        {
            goto L280;
        }
        if(kdflg == 1)
        {
            iflag = 2;
        }
        if(abs(rs1) < *alim)
        {
            goto L240;
        }
        /* ----------------------------------------------------------------------- */
        /*     REFINE  TEST AND SCALE */
        /* ----------------------------------------------------------------------- */
        aphi = zabs_(&phidr, &phidi);
        aarg = zabs_(&argdr, &argdi);
        rs1  = rs1 + log(aphi) - log(aarg) * .25 - aic;
        if(abs(rs1) > *elim)
        {
            goto L280;
        }
        if(kdflg == 1)
        {
            iflag = 1;
        }
        if(rs1 < 0.)
        {
            goto L240;
        }
        if(kdflg == 1)
        {
            iflag = 3;
        }
    L240:
        zairy_(&argdr, &argdi, &c__0, &c__2, &air, &aii, &nai, &idum);
        zairy_(&argdr, &argdi, &c__1, &c__2, &dair, &daii, &ndai, &idum);
        str = dair * bsumdr - daii * bsumdi;
        sti = dair * bsumdi + daii * bsumdr;
        str += air * asumdr - aii * asumdi;
        sti += air * asumdi + aii * asumdr;
        ptr = str * phidr - sti * phidi;
        pti = str * phidi + sti * phidr;
        s2r = ptr * csr - pti * csi;
        s2i = ptr * csi + pti * csr;
        str = exp(s1r) * cssr[iflag - 1];
        s1r = str * cos(s1i);
        s1i = str * sin(s1i);
        str = s2r * s1r - s2i * s1i;
        s2i = s2r * s1i + s2i * s1r;
        s2r = str;
        if(iflag != 1)
        {
            goto L250;
        }
        zuchk_(&s2r, &s2i, &nw, bry, tol);
        if(nw == 0)
        {
            goto L250;
        }
        s2r = zeror;
        s2i = zeroi;
    L250:
        if(yy <= 0.)
        {
            s2i = -s2i;
        }
        cyr[kdflg - 1] = s2r;
        cyi[kdflg - 1] = s2i;
        c2r            = s2r;
        c2i            = s2i;
        s2r *= csrr[iflag - 1];
        s2i *= csrr[iflag - 1];
        /* ----------------------------------------------------------------------- */
        /*     ADD I AND K FUNCTIONS, K SEQUENCE IN Y(I), I=1,N */
        /* ----------------------------------------------------------------------- */
        s1r = yr[kk];
        s1i = yi[kk];
        if(*kode == 1)
        {
            goto L270;
        }
        zs1s2_(&zrr, &zri, &s1r, &s1i, &s2r, &s2i, &nw, &asc, alim, &iuf);
        *nz += nw;
    L270:
        yr[kk] = s1r * cspnr - s1i * cspni + s2r;
        yi[kk] = s1r * cspni + s1i * cspnr + s2i;
        --kk;
        cspnr = -cspnr;
        cspni = -cspni;
        str   = csi;
        csi   = -csr;
        csr   = str;
        if(c2r != 0. || c2i != 0.)
        {
            goto L255;
        }
        kdflg = 1;
        goto L290;
    L255:
        if(kdflg == 2)
        {
            goto L295;
        }
        kdflg = 2;
        goto L290;
    L280:
        if(rs1 > 0.)
        {
            goto L320;
        }
        s2r = zeror;
        s2i = zeroi;
        goto L250;
    L290:;
    }
    k = *n;
L295:
    il = *n - k;
    if(il == 0)
    {
        return 0;
    }
    /* ----------------------------------------------------------------------- */
    /*     RECUR BACKWARD FOR REMAINDER OF I SEQUENCE AND ADD IN THE */
    /*     K FUNCTIONS, SCALING THE I SEQUENCE DURING RECURRENCE TO KEEP */
    /*     INTERMEDIATE ARITHMETIC ON SCALE NEAR EXPONENT EXTREMES. */
    /* ----------------------------------------------------------------------- */
    s1r   = cyr[0];
    s1i   = cyi[0];
    s2r   = cyr[1];
    s2i   = cyi[1];
    csr   = csrr[iflag - 1];
    ascle = bry[iflag - 1];
    fn    = (double)((float)(inu + il));
    i__1  = il;
    for(i__ = 1; i__ <= i__1; ++i__)
    {
        c2r = s2r;
        c2i = s2i;
        s2r = s1r + (fn + fnf) * (rzr * c2r - rzi * c2i);
        s2i = s1i + (fn + fnf) * (rzr * c2i + rzi * c2r);
        s1r = c2r;
        s1i = c2i;
        fn += -1.;
        c2r = s2r * csr;
        c2i = s2i * csr;
        ckr = c2r;
        cki = c2i;
        c1r = yr[kk];
        c1i = yi[kk];
        if(*kode == 1)
        {
            goto L300;
        }
        zs1s2_(&zrr, &zri, &c1r, &c1i, &c2r, &c2i, &nw, &asc, alim, &iuf);
        *nz += nw;
    L300:
        yr[kk] = c1r * cspnr - c1i * cspni + c2r;
        yi[kk] = c1r * cspni + c1i * cspnr + c2i;
        --kk;
        cspnr = -cspnr;
        cspni = -cspni;
        if(iflag >= 3)
        {
            goto L310;
        }
        c2r = abs(ckr);
        c2i = abs(cki);
        c2m = max(c2r, c2i);
        if(c2m <= ascle)
        {
            goto L310;
        }
        ++iflag;
        ascle = bry[iflag - 1];
        s1r *= csr;
        s1i *= csr;
        s2r = ckr;
        s2i = cki;
        s1r *= cssr[iflag - 1];
        s1i *= cssr[iflag - 1];
        s2r *= cssr[iflag - 1];
        s2i *= cssr[iflag - 1];
        csr = csrr[iflag - 1];
    L310:;
    }
    return 0;
L320:
    *nz = -1;
    return 0;
} /* zunk2_ */

 int zuoik_(double* zr,
                            double* zi,
                            double* fnu,
                            int32*    kode,
                            int32*    ikflg,
                            int32*    n,
                            double* yr,
                            double* yi,
                            int32*    nuf,
                            double* tol,
                            double* elim,
                            double* alim)
{
    /* Initialized data */

    static double zeror = 0.;
    static double zeroi = 0.;
    static double aic   = 1.265512123484645396;

    /* System generated locals */
    int32 i__1;

    

    
    int32                     i__;
    double                  ax, ay;
    int32                     nn, nw;
    double                  fnn, gnn, zbi, czi, gnu, zbr, czr, rcz, sti, zni, zri, str, znr, zrr, aarg, aphi, argi, phii, argr;
    int32                     idum;
    
    double                  phir;
    int32                     init;
    double                  sumi, sumr, ascle;
    int32                     iform;
    double                  asumi, bsumi, cwrki[16];
    double                  asumr, bsumr, cwrkr[16];
   
    double zeta1i, zeta2i, zeta1r, zeta2r;

    /* *********************************************************************72 */

    /* ***BEGIN PROLOGUE  ZUOIK */
    /* ***REFER TO  ZBESI,ZBESK,ZBESH */

    /*     ZUOIK COMPUTES THE LEADING TERMS OF THE UNIFORM ASYMPTOTIC */
    /*     EXPANSIONS FOR THE I AND K FUNCTIONS AND COMPARES THEM */
    /*     (IN LOGARITHMIC FORM) TO ALIM AND ELIM FOR OVER AND UNDERFLOW */
    /*     WHERE ALIM.LT.ELIM. IF THE MAGNITUDE, BASED ON THE LEADING */
    /*     EXPONENTIAL, IS LESS THAN ALIM OR GREATER THAN -ALIM, THEN */
    /*     THE RESULT IS ON SCALE. IF NOT, THEN A REFINED TEST USING OTHER */
    /*     MULTIPLIERS (IN LOGARITHMIC FORM) IS MADE BASED ON ELIM. HERE */
    /*     EXP(-ELIM)=SMALLEST MACHINE NUMBER*1.0E+3 AND EXP(-ALIM)= */
    /*     EXP(-ELIM)/TOL */

    /*     IKFLG=1 MEANS THE I SEQUENCE IS TESTED */
    /*          =2 MEANS THE K SEQUENCE IS TESTED */
    /*     NUF = 0 MEANS THE LAST MEMBER OF THE SEQUENCE IS ON SCALE */
    /*         =-1 MEANS AN OVERFLOW WOULD OCCUR */
    /*     IKFLG=1 AND NUF.GT.0 MEANS THE LAST NUF Y VALUES WERE SET TO ZERO */
    /*             THE FIRST N-NUF VALUES MUST BE SET BY ANOTHER ROUTINE */
    /*     IKFLG=2 AND NUF.EQ.N MEANS ALL Y VALUES WERE SET TO ZERO */
    /*     IKFLG=2 AND 0.LT.NUF.LT.N NOT CONSIDERED. Y MUST BE SET BY */
    /*             ANOTHER ROUTINE */

    /* ***ROUTINES CALLED  ZUCHK,ZUNHJ,ZUNIK,D1MACH,ZABS,ZLOG */
    /* ***END PROLOGUE  ZUOIK */
    /*     COMPLEX ARG,ASUM,BSUM,CWRK,CZ,CZERO,PHI,SUM,Y,Z,ZB,ZETA1,ZETA2,ZN, */
    /*    *ZR */
    /* Parameter adjustments */
    --yi;
    --yr;

    /* Function Body */
    *nuf = 0;
    nn   = *n;
    zrr  = *zr;
    zri  = *zi;
    if(*zr >= 0.)
    {
        goto L10;
    }
    zrr = -(*zr);
    zri = -(*zi);
L10:
    zbr   = zrr;
    zbi   = zri;
    ax    = abs(*zr) * 1.7321;
    ay    = abs(*zi);
    iform = 1;
    if(ay > ax)
    {
        iform = 2;
    }
    gnu = max(*fnu, 1.);
    if(*ikflg == 1)
    {
        goto L20;
    }
    fnn = (double)((float)nn);
    gnn = *fnu + fnn - 1.;
    gnu = max(gnn, fnn);
L20:
    /* ----------------------------------------------------------------------- */
    /*     ONLY THE MAGNITUDE OF ARG AND PHI ARE NEEDED ALONG WITH THE */
    /*     REAL PARTS OF ZETA1, ZETA2 AND ZB. NO ATTEMPT IS MADE TO GET */
    /*     THE SIGN OF THE IMAGINARY PART CORRECT. */
    /* ----------------------------------------------------------------------- */
    if(iform == 2)
    {
        goto L30;
    }
    init = 0;
    zunik_(&zrr, &zri, &gnu, ikflg, &c__1, tol, &init, &phir, &phii, &zeta1r, &zeta1i, &zeta2r, &zeta2i, &sumr, &sumi, cwrkr, cwrki);
    czr = -zeta1r + zeta2r;
    czi = -zeta1i + zeta2i;
    goto L50;
L30:
    znr = zri;
    zni = -zrr;
    if(*zi > 0.)
    {
        goto L40;
    }
    znr = -znr;
L40:
    zunhj_(&znr, &zni, &gnu, &c__1, tol, &phir, &phii, &argr, &argi, &zeta1r, &zeta1i, &zeta2r, &zeta2i, &asumr, &asumi, &bsumr, &bsumi);
    czr  = -zeta1r + zeta2r;
    czi  = -zeta1i + zeta2i;
    aarg = zabs_(&argr, &argi);
L50:
    if(*kode == 1)
    {
        goto L60;
    }
    czr -= zbr;
    czi -= zbi;
L60:
    if(*ikflg == 1)
    {
        goto L70;
    }
    czr = -czr;
    czi = -czi;
L70:
    aphi = zabs_(&phir, &phii);
    rcz  = czr;
    /* ----------------------------------------------------------------------- */
    /*     OVERFLOW TEST */
    /* ----------------------------------------------------------------------- */
    if(rcz > *elim)
    {
        goto L210;
    }
    if(rcz < *alim)
    {
        goto L80;
    }
    rcz += log(aphi);
    if(iform == 2)
    {
        rcz = rcz - log(aarg) * .25 - aic;
    }
    if(rcz > *elim)
    {
        goto L210;
    }
    goto L130;
L80:
    /* ----------------------------------------------------------------------- */
    /*     UNDERFLOW TEST */
    /* ----------------------------------------------------------------------- */
    if(rcz < -(*elim))
    {
        goto L90;
    }
    if(rcz > -(*alim))
    {
        goto L130;
    }
    rcz += log(aphi);
    if(iform == 2)
    {
        rcz = rcz - log(aarg) * .25 - aic;
    }
    if(rcz > -(*elim))
    {
        goto L110;
    }
L90:
    i__1 = nn;
    for(i__ = 1; i__ <= i__1; ++i__)
    {
        yr[i__] = zeror;
        yi[i__] = zeroi;
        /* L100: */
    }
    *nuf = nn;
    return 0;
L110:
    ascle = d1mach_(&c__1) * 1e3 / *tol;
    zlog_(&phir, &phii, &str, &sti, &idum);
    czr += str;
    czi += sti;
    if(iform == 1)
    {
        goto L120;
    }
    zlog_(&argr, &argi, &str, &sti, &idum);
    czr = czr - str * .25 - aic;
    czi -= sti * .25;
L120:
    ax  = exp(rcz) / *tol;
    ay  = czi;
    czr = ax * cos(ay);
    czi = ax * sin(ay);
    zuchk_(&czr, &czi, &nw, &ascle, tol);
    if(nw != 0)
    {
        goto L90;
    }
L130:
    if(*ikflg == 2)
    {
        return 0;
    }
    if(*n == 1)
    {
        return 0;
    }
/* ----------------------------------------------------------------------- */
/*     SET UNDERFLOWS ON I SEQUENCE */
/* ----------------------------------------------------------------------- */
L140:
    gnu = *fnu + (double)((float)(nn - 1));
    if(iform == 2)
    {
        goto L150;
    }
    init = 0;
    zunik_(&zrr, &zri, &gnu, ikflg, &c__1, tol, &init, &phir, &phii, &zeta1r, &zeta1i, &zeta2r, &zeta2i, &sumr, &sumi, cwrkr, cwrki);
    czr = -zeta1r + zeta2r;
    czi = -zeta1i + zeta2i;
    goto L160;
L150:
    zunhj_(&znr, &zni, &gnu, &c__1, tol, &phir, &phii, &argr, &argi, &zeta1r, &zeta1i, &zeta2r, &zeta2i, &asumr, &asumi, &bsumr, &bsumi);
    czr  = -zeta1r + zeta2r;
    czi  = -zeta1i + zeta2i;
    aarg = zabs_(&argr, &argi);
L160:
    if(*kode == 1)
    {
        goto L170;
    }
    czr -= zbr;
    czi -= zbi;
L170:
    aphi = zabs_(&phir, &phii);
    rcz  = czr;
    if(rcz < -(*elim))
    {
        goto L180;
    }
    if(rcz > -(*alim))
    {
        return 0;
    }
    rcz += log(aphi);
    if(iform == 2)
    {
        rcz = rcz - log(aarg) * .25 - aic;
    }
    if(rcz > -(*elim))
    {
        goto L190;
    }
L180:
    yr[nn] = zeror;
    yi[nn] = zeroi;
    --nn;
    ++(*nuf);
    if(nn == 0)
    {
        return 0;
    }
    goto L140;
L190:
    ascle = d1mach_(&c__1) * 1e3 / *tol;
    zlog_(&phir, &phii, &str, &sti, &idum);
    czr += str;
    czi += sti;
    if(iform == 1)
    {
        goto L200;
    }
    zlog_(&argr, &argi, &str, &sti, &idum);
    czr = czr - str * .25 - aic;
    czi -= sti * .25;
L200:
    ax  = exp(rcz) / *tol;
    ay  = czi;
    czr = ax * cos(ay);
    czi = ax * sin(ay);
    zuchk_(&czr, &czi, &nw, &ascle, tol);
    if(nw != 0)
    {
        goto L180;
    }
    return 0;
L210:
    *nuf = -1;
    return 0;
} /* zuoik_ */

 int zwrsk_(double* zrr,
                            double* zri,
                            double* fnu,
                            int32*    kode,
                            int32*    n,
                            double* yr,
                            double* yi,
                            int32*    nz,
                            double* cwr,
                            double* cwi,
                            double* tol,
                            double* elim,
                            double* alim)
{
    /* System generated locals */
    int32 i__1;

 

    
    int32                     i__, nw;
    double                  c1i, c2i, c1r, c2r, act, acw, cti, ctr, pti, sti, ptr, str, ract;
    
    double                  ascle, csclr, cinui, cinur;
    

    /* *********************************************************************72 */

    /* ***BEGIN PROLOGUE  ZWRSK */
    /* ***REFER TO  ZBESI,ZBESK */

    /*     ZWRSK COMPUTES THE I BESSEL FUNCTION FOR RE(Z).GE.0.0 BY */
    /*     NORMALIZING THE I FUNCTION RATIOS FROM ZRATI BY THE WRONSKIAN */

    /* ***ROUTINES CALLED  D1MACH,ZBKNU,ZRATI,ZABS */
    /* ***END PROLOGUE  ZWRSK */
    /*     COMPLEX CINU,CSCL,CT,CW,C1,C2,RCT,ST,Y,ZR */
    /* ----------------------------------------------------------------------- */
    /*     I(FNU+I-1,Z) BY BACKWARD RECURRENCE FOR RATIOS */
    /*     Y(I)=I(FNU+I,Z)/I(FNU+I-1,Z) FROM CRATI NORMALIZED BY THE */
    /*     WRONSKIAN WITH K(FNU,Z) AND K(FNU+1,Z) FROM CBKNU. */
    /* ----------------------------------------------------------------------- */
    /* Parameter adjustments */
    --yi;
    --yr;
    --cwr;
    --cwi;

    /* Function Body */
    *nz = 0;
    zbknu_(zrr, zri, fnu, kode, &c__2, &cwr[1], &cwi[1], &nw, tol, elim, alim);
    if(nw != 0)
    {
        goto L50;
    }
    zrati_(zrr, zri, fnu, n, &yr[1], &yi[1], tol);
    /* ----------------------------------------------------------------------- */
    /*     RECUR FORWARD ON I(FNU+1,Z) = R(FNU,Z)*I(FNU,Z), */
    /*     R(FNU+J-1,Z)=Y(J),  J=1,...,N */
    /* ----------------------------------------------------------------------- */
    cinur = 1.;
    cinui = 0.;
    if(*kode == 1)
    {
        goto L10;
    }
    cinur = cos(*zri);
    cinui = sin(*zri);
L10:
    /* ----------------------------------------------------------------------- */
    /*     ON LOW EXPONENT MACHINES THE K FUNCTIONS CAN BE CLOSE TO BOTH */
    /*     THE UNDER AND OVERFLOW LIMITS AND THE NORMALIZATION MUST BE */
    /*     SCALED TO PREVENT OVER OR UNDERFLOW. CUOIK HAS DETERMINED THAT */
    /*     THE RESULT IS ON SCALE. */
    /* ----------------------------------------------------------------------- */
    acw   = zabs_(&cwr[2], &cwi[2]);
    ascle = d1mach_(&c__1) * 1e3 / *tol;
    csclr = 1.;
    if(acw > ascle)
    {
        goto L20;
    }
    csclr = 1. / *tol;
    goto L30;
L20:
    ascle = 1. / ascle;
    if(acw < ascle)
    {
        goto L30;
    }
    csclr = *tol;
L30:
    c1r = cwr[1] * csclr;
    c1i = cwi[1] * csclr;
    c2r = cwr[2] * csclr;
    c2i = cwi[2] * csclr;
    str = yr[1];
    sti = yi[1];
    /* ----------------------------------------------------------------------- */
    /*     CINU=CINU*(CONJG(CT)/CABS(CT))*(1.0D0/CABS(CT) PREVENTS */
    /*     UNDER- OR OVERFLOW PREMATURELY BY SQUARING CABS(CT) */
    /* ----------------------------------------------------------------------- */
    ptr = str * c1r - sti * c1i;
    pti = str * c1i + sti * c1r;
    ptr += c2r;
    pti += c2i;
    ctr  = *zrr * ptr - *zri * pti;
    cti  = *zrr * pti + *zri * ptr;
    act  = zabs_(&ctr, &cti);
    ract = 1. / act;
    ctr *= ract;
    cti   = -cti * ract;
    ptr   = cinur * ract;
    pti   = cinui * ract;
    cinur = ptr * ctr - pti * cti;
    cinui = ptr * cti + pti * ctr;
    yr[1] = cinur * csclr;
    yi[1] = cinui * csclr;
    if(*n == 1)
    {
        return 0;
    }
    i__1 = *n;
    for(i__ = 2; i__ <= i__1; ++i__)
    {
        ptr     = str * cinur - sti * cinui;
        cinui   = str * cinui + sti * cinur;
        cinur   = ptr;
        str     = yr[i__];
        sti     = yi[i__];
        yr[i__] = cinur * csclr;
        yi[i__] = cinui * csclr;
        /* L40: */
    }
    return 0;
L50:
    *nz = -1;
    if(nw == -2)
    {
        *nz = -2;
    }
    return 0;
} /* zwrsk_ */
