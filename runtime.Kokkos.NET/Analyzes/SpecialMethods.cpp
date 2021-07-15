
#include <Analyzes/SpecialMethods.h>

#include <Print.hpp>

#pragma region BesselI float
#pragma region BesselI

KOKKOS_FUNCTION float BesselI(const int nOrder, const float arg)
{
    if(nOrder == 0)
    {
        return BesselI0(arg);
    }
    if(nOrder == 1)
    {
        return BesselI1(arg);
    }

    return BesselIN(nOrder, arg);
}

KOKKOS_FUNCTION static int calci0(const float* arg, float* result, const int* jint)
{
    static const float one   = 1.0E0f;
    static const float one5  = 15.0E0f;
    static const float exp40 = 2.353852668370199854E17f;
    static const float forty = 40.0E0f;
    static const float rec15 = 6.6666666666666666666E-2f;
    static const float two25 = 225.0E0f;

    static const float xsmall = 2.98E-8f;
    static const float xinf   = 3.40E38f;
    static const float xmax   = 91.9E0f;

    static const float p[15] = {-5.2487866627945699800E-18f,
                                -1.5982226675653184646E-14f,
                                -2.6843448573468483278E-11f,
                                -3.0517226450451067446E-08f,
                                -2.5172644670688975051E-05f,
                                -1.5453977791786851041E-02f,
                                -7.0935347449210549190E+00f,
                                -2.4125195876041896775E+03f,
                                -5.9545626019847898221E+05f,
                                -1.0313066708737980747E+08f,
                                -1.1912746104985237192E+10f,
                                -8.4925101247114157499E+11f,
                                -3.2940087627407749166E+13f,
                                -5.5050369673018427753E+14f,
                                -2.2335582639474375249E+15f};

    static const float q[5] = {-3.7277560179962773046E+03f, 6.5158506418655165707E+06f, -6.5626560740833869295E+09f, 3.7604188704092954661E+12f, -9.7087946179594019126E+14f};

    static const float pp[8] = {-3.9843750000000000000E-01f,
                                2.9205384596336793945E+00f,
                                -2.4708469169133954315E+00f,
                                4.7914889422856814203E-01f,
                                -3.7384991926068969150E-03f,
                                -2.6801520353328635310E-03f,
                                9.9168777670983678974E-05f,
                                -2.1877128189032726730E-06f};

    static const float qq[7] = {-3.1446690275135491500E+01f,
                                8.5539563258012929600E+01f,
                                -6.0228002066743340583E+01f,
                                1.3982595353892851542E+01f,
                                -1.1151759188741312645E+00f,
                                3.2547697594819615062E-02f,
                                -5.5194330231005480228E-04f};

    float a;
    float b;
    float xx;
    float sump;
    float sumq;

    float x = abs(*arg);

    if(x < xsmall)
    {
        *result = one;
    }
    else if(x < one5)
    {
        xx   = x * x;
        sump = p[0];

        for(int i = 2; i <= 15; ++i)
        {
            sump = sump * xx + p[i - 1];
        }

        xx -= two25;

        sumq = ((((xx + q[0]) * xx + q[1]) * xx + q[2]) * xx + q[3]) * xx + q[4];

        *result = sump / sumq;

        if(*jint == 2)
        {
            *result *= exp(-x);
        }
    }
    else if(x >= one5)
    {
        if(*jint == 1 && x > xmax)
        {
            *result = xinf;
        }
        else
        {
            xx = one / x - rec15;

            sump = ((((((pp[0] * xx + pp[1]) * xx + pp[2]) * xx + pp[3]) * xx + pp[4]) * xx + pp[5]) * xx + pp[6]) * xx + pp[7];

            sumq = ((((((xx + qq[0]) * xx + qq[1]) * xx + qq[2]) * xx + qq[3]) * xx + qq[4]) * xx + qq[5]) * xx + qq[6];

            *result = sump / sumq;

            if(*jint == 2)
            {
                *result = (*result - pp[0]) / sqrt(x);
            }
            else
            {
                if(x <= xmax - one5)
                {
                    a = exp(x);
                    b = one;
                }
                else
                {
                    a = exp(x - forty);
                    b = exp40;
                }
                *result = (*result * a - pp[0] * a) / sqrt(x) * b;
            }
        }
    }

    return 0;
}

KOKKOS_FUNCTION float BesselI0(const float arg)
{
    int   jint;
    float result;

    jint = 1;
    calci0(&arg, &result, &jint);
    float ret_val = result;
    return ret_val;
}

KOKKOS_FUNCTION static int calci1(const float* arg, float* result, const int* jint)
{
    static const float one   = 1.0E0f;
    static const float one5  = 15.0E0f;
    static const float exp40 = 2.353852668370199854E17f;
    static const float forty = 40.0E0f;
    static const float rec15 = 6.6666666666666666666E-2f;
    static const float two25 = 225.0E0f;
    static const float half  = 0.5E0f;
    static const float zero  = 0.0E0f;

    static const float pbar = 3.98437500E-01f;

    static const float xsmall = 2.98E-8f;
    static const float xinf   = 3.40E38f;
    static const float xmax   = 91.9E0f;

    static const float p[15] = {-1.9705291802535139930E-19f,
                                -6.5245515583151902910E-16f,
                                -1.1928788903603238754E-12f,
                                -1.4831904935994647675E-09f,
                                -1.3466829827635152875E-06f,
                                -9.1746443287817501309E-04f,
                                -4.7207090827310162436E-01f,
                                -1.8225946631657315931E+02f,
                                -5.1894091982308017540E+04f,
                                -1.0588550724769347106E+07f,
                                -1.4828267606612366099E+09f,
                                -1.3357437682275493024E+11f,
                                -6.9876779648010090070E+12f,
                                -1.7732037840791591320E+14f,
                                -1.4577180278143463643E+15f};

    static const float q[5] = {-4.0076864679904189921E+03f, 7.4810580356655069138E+06f, -8.0059518998619764991E+09f, 4.8544714258273622913E+12f, -1.3218168307321442305E+15f};

    static const float pp[8] = {-6.0437159056137600000E-02f,
                                4.5748122901933459000E-01f,
                                -4.2843766903304806403E-01f,
                                9.7356000150886612134E-02f,
                                -3.2457723974465568321E-03f,
                                -3.6395264712121795296E-04f,
                                1.6258661867440836395E-05f,
                                -3.6347578404608223492E-07f};

    static const float qq[6] = {-3.8806586721556593450E+00f,
                                3.2593714889036996297E+00f,
                                -8.5017476463217924408E-01f,
                                7.4212010813186530069E-02f,
                                -2.2835624489492512649E-03f,
                                3.7510433111922824643E-05f};

    float a;
    float b;
    float xx;
    float sump;
    float sumq;

    float x = abs(*arg);
    if(x < xsmall)
    {
        *result = half * x;
    }
    else if(x < one5)
    {
        xx   = x * x;
        sump = p[0];
        for(int j = 2; j <= 15; ++j)
        {
            sump = sump * xx + p[j - 1];
            /* L50: */
        }
        xx -= two25;
        sumq    = ((((xx + q[0]) * xx + q[1]) * xx + q[2]) * xx + q[3]) * xx + q[4];
        *result = sump / sumq * x;
        if(*jint == 2)
        {
            *result *= exp(-x);
        }
    }
    else if(*jint == 1 && x > xmax)
    {
        *result = xinf;
    }
    else
    {
        xx   = one / x - rec15;
        sump = ((((((pp[0] * xx + pp[1]) * xx + pp[2]) * xx + pp[3]) * xx + pp[4]) * xx + pp[5]) * xx + pp[6]) * xx + pp[7];

        sumq = (((((xx + qq[0]) * xx + qq[1]) * xx + qq[2]) * xx + qq[3]) * xx + qq[4]) * xx + qq[5];

        *result = sump / sumq;
        if(*jint != 1)
        {
            *result = (*result + pbar) / sqrt(x);
        }
        else
        {
            if(x > xmax - one5)
            {
                a = exp(x - forty);
                b = exp40;
            }
            else
            {
                a = exp(x);
                b = one;
            }
            *result = (*result * a + pbar * a) / sqrt(x) * b;
        }
    }
    if(*arg < zero)
    {
        *result = -(*result);
    }
    return 0;
}

KOKKOS_FUNCTION float BesselI1(const float arg)
{
    int   jint;
    float result;

    jint = 1;
    calci1(&arg, &result, &jint);
    const float ret_val = result;
    return ret_val;
}

KOKKOS_FUNCTION float BesselIN(const int nOrder, const float arg)
{
    System::out << "BesselIN Exception" << System::endl;
    return Constants<float>::NaN();
}
#pragma endregion

#pragma region BesselIE

KOKKOS_FUNCTION float BesselIE(const int nOrder, const float arg)
{
    if(nOrder == 0)
    {
        return BesselIE0(arg);
    }
    if(nOrder == 1)
    {
        return BesselIE1(arg);
    }

    return BesselIEN(nOrder, arg);
}

KOKKOS_FUNCTION float BesselIE0(const float arg)
{
    int   jint;
    float result;

    jint = 2;
    calci0(&arg, &result, &jint);
    float ret_val = result;
    return ret_val;
}

KOKKOS_FUNCTION float BesselIE1(const float arg)
{
    int   jint;
    float result;

    jint = 2;
    calci1(&arg, &result, &jint);
    float ret_val = result;
    return ret_val;
}

KOKKOS_FUNCTION float BesselIEN(const int nOrder, const float arg)
{
    System::out << "BesselIEN Exception" << System::endl;
    return Constants<float>::NaN();
}
#pragma endregion
#pragma endregion

#pragma region BesselI double
#pragma region BesselI

KOKKOS_FUNCTION double BesselI(const int nOrder, const double arg)
{
    if(nOrder == 0)
    {
        return BesselI0(arg);
    }
    if(nOrder == 1)
    {
        return BesselI1(arg);
    }

    return BesselIN(nOrder, arg);
}

KOKKOS_FUNCTION static int calci0(const double* arg, double* result, const int* jint)
{
    static const double one   = 1.0E0;
    static const double one5  = 15.0E0;
    static const double exp40 = 2.353852668370199854E17;
    static const double forty = 40.0E0;
    static const double rec15 = 6.6666666666666666666E-2;
    static const double two25 = 225.0E0;

    static const double xsmall = 2.98E-8;
    static const double xinf   = 3.40E38;
    static const double xmax   = 91.9E0;

    static const double p[15] = {-5.2487866627945699800E-18,
                                 -1.5982226675653184646E-14,
                                 -2.6843448573468483278E-11,
                                 -3.0517226450451067446E-08,
                                 -2.5172644670688975051E-05,
                                 -1.5453977791786851041E-02,
                                 -7.0935347449210549190E+00,
                                 -2.4125195876041896775E+03,
                                 -5.9545626019847898221E+05,
                                 -1.0313066708737980747E+08,
                                 -1.1912746104985237192E+10,
                                 -8.4925101247114157499E+11,
                                 -3.2940087627407749166E+13,
                                 -5.5050369673018427753E+14,
                                 -2.2335582639474375249E+15};

    static const double q[5] = {-3.7277560179962773046E+03, 6.5158506418655165707E+06, -6.5626560740833869295E+09, 3.7604188704092954661E+12, -9.7087946179594019126E+14};

    static const double pp[8] = {-3.9843750000000000000E-01,
                                 2.9205384596336793945E+00,
                                 -2.4708469169133954315E+00,
                                 4.7914889422856814203E-01,
                                 -3.7384991926068969150E-03,
                                 -2.6801520353328635310E-03,
                                 9.9168777670983678974E-05,
                                 -2.1877128189032726730E-06};

    static const double qq[7] = {-3.1446690275135491500E+01,
                                 8.5539563258012929600E+01,
                                 -6.0228002066743340583E+01,
                                 1.3982595353892851542E+01,
                                 -1.1151759188741312645E+00,
                                 3.2547697594819615062E-02,
                                 -5.5194330231005480228E-04};

    double a;
    double b;
    double xx;
    double sump;
    double sumq;

    double x = abs(*arg);

    if(x < xsmall)
    {
        *result = one;
    }
    else if(x < one5)
    {
        xx   = x * x;
        sump = p[0];

        for(int i = 2; i <= 15; ++i)
        {
            sump = sump * xx + p[i - 1];
        }

        xx -= two25;

        sumq = ((((xx + q[0]) * xx + q[1]) * xx + q[2]) * xx + q[3]) * xx + q[4];

        *result = sump / sumq;

        if(*jint == 2)
        {
            *result *= exp(-x);
        }
    }
    else if(x >= one5)
    {
        if(*jint == 1 && x > xmax)
        {
            *result = xinf;
        }
        else
        {
            xx = one / x - rec15;

            sump = ((((((pp[0] * xx + pp[1]) * xx + pp[2]) * xx + pp[3]) * xx + pp[4]) * xx + pp[5]) * xx + pp[6]) * xx + pp[7];

            sumq = ((((((xx + qq[0]) * xx + qq[1]) * xx + qq[2]) * xx + qq[3]) * xx + qq[4]) * xx + qq[5]) * xx + qq[6];

            *result = sump / sumq;

            if(*jint == 2)
            {
                *result = (*result - pp[0]) / sqrt(x);
            }
            else
            {
                if(x <= xmax - one5)
                {
                    a = exp(x);
                    b = one;
                }
                else
                {
                    a = exp(x - forty);
                    b = exp40;
                }
                *result = (*result * a - pp[0] * a) / sqrt(x) * b;
            }
        }
    }

    return 0;
}

KOKKOS_FUNCTION double BesselI0(const double arg)
{
    int    jint;
    double result;

    jint = 1;
    calci0(&arg, &result, &jint);
    const double ret_val = result;
    return ret_val;
}

KOKKOS_FUNCTION static int calci1(const double* arg, double* result, const int* jint)
{
    static const double one   = 1.0E0;
    static const double one5  = 15.0E0;
    static const double exp40 = 2.353852668370199854E17;
    static const double forty = 40.0E0;
    static const double rec15 = 6.6666666666666666666E-2;
    static const double two25 = 225.0E0;
    static const double half  = 0.5E0;
    static const double zero  = 0.0E0;

    static const double pbar = 3.98437500E-01;

    static const double xsmall = 2.98E-8;
    static const double xinf   = 3.40E38;
    static const double xmax   = 91.9E0;

    static const double p[15] = {-1.9705291802535139930E-19,
                                 -6.5245515583151902910E-16,
                                 -1.1928788903603238754E-12,
                                 -1.4831904935994647675E-09,
                                 -1.3466829827635152875E-06,
                                 -9.1746443287817501309E-04,
                                 -4.7207090827310162436E-01,
                                 -1.8225946631657315931E+02,
                                 -5.1894091982308017540E+04,
                                 -1.0588550724769347106E+07,
                                 -1.4828267606612366099E+09,
                                 -1.3357437682275493024E+11,
                                 -6.9876779648010090070E+12,
                                 -1.7732037840791591320E+14,
                                 -1.4577180278143463643E+15};

    static const double q[5] = {-4.0076864679904189921E+03, 7.4810580356655069138E+06, -8.0059518998619764991E+09, 4.8544714258273622913E+12, -1.3218168307321442305E+15};

    static const double pp[8] = {-6.0437159056137600000E-02,
                                 4.5748122901933459000E-01,
                                 -4.2843766903304806403E-01,
                                 9.7356000150886612134E-02,
                                 -3.2457723974465568321E-03,
                                 -3.6395264712121795296E-04,
                                 1.6258661867440836395E-05,
                                 -3.6347578404608223492E-07};

    static const double qq[6] = {
        -3.8806586721556593450E+00, 3.2593714889036996297E+00, -8.5017476463217924408E-01, 7.4212010813186530069E-02, -2.2835624489492512649E-03, 3.7510433111922824643E-05};

    double a;
    double b;
    double xx;
    double sump;
    double sumq;

    double x = abs(*arg);
    if(x < xsmall)
    {
        *result = half * x;
    }
    else if(x < one5)
    {
        xx   = x * x;
        sump = p[0];
        for(int j = 2; j <= 15; ++j)
        {
            sump = sump * xx + p[j - 1];
            /* L50: */
        }
        xx -= two25;
        sumq    = ((((xx + q[0]) * xx + q[1]) * xx + q[2]) * xx + q[3]) * xx + q[4];
        *result = sump / sumq * x;
        if(*jint == 2)
        {
            *result *= exp(-x);
        }
    }
    else if(*jint == 1 && x > xmax)
    {
        *result = xinf;
    }
    else
    {
        xx   = one / x - rec15;
        sump = ((((((pp[0] * xx + pp[1]) * xx + pp[2]) * xx + pp[3]) * xx + pp[4]) * xx + pp[5]) * xx + pp[6]) * xx + pp[7];

        sumq = (((((xx + qq[0]) * xx + qq[1]) * xx + qq[2]) * xx + qq[3]) * xx + qq[4]) * xx + qq[5];

        *result = sump / sumq;
        if(*jint != 1)
        {
            *result = (*result + pbar) / sqrt(x);
        }
        else
        {
            if(x > xmax - one5)
            {
                a = exp(x - forty);
                b = exp40;
            }
            else
            {
                a = exp(x);
                b = one;
            }
            *result = (*result * a + pbar * a) / sqrt(x) * b;
        }
    }
    if(*arg < zero)
    {
        *result = -(*result);
    }
    return 0;
}

KOKKOS_FUNCTION double BesselI1(const double arg)
{
    int    jint;
    double result;

    jint = 1;
    calci1(&arg, &result, &jint);
    double ret_val = result;
    return ret_val;
}

KOKKOS_FUNCTION double BesselIN(const int nOrder, const double arg)
{
    System::out << "BesselIN Exception" << System::endl;
    return Constants<double>::NaN();
}
#pragma endregion

#pragma region BesselIE

KOKKOS_FUNCTION double BesselIE(const int nOrder, const double arg)
{
    if(nOrder == 0)
    {
        return BesselIE0(arg);
    }
    if(nOrder == 1)
    {
        return BesselIE1(arg);
    }

    return BesselIEN(nOrder, arg);
}

KOKKOS_FUNCTION double BesselIE0(const double arg)
{
    int    jint;
    double result;

    jint = 2;
    calci0(&arg, &result, &jint);
    double ret_val = result;
    return ret_val;
}

KOKKOS_FUNCTION double BesselIE1(const double arg)
{
    int    jint;
    double result;

    jint = 2;
    calci1(&arg, &result, &jint);
    double ret_val = result;
    return ret_val;
}

KOKKOS_FUNCTION double BesselIEN(const int nOrder, const double arg)
{
    System::out << "BesselIEN Exception" << System::endl;
    return Constants<double>::NaN();
}
#pragma endregion
#pragma endregion

//#pragma region BesselI ComplexF
//#pragma region BesselI
//
//    ComplexF BesselI<ComplexF>(const int nOrder, ComplexF arg)
//    {
//        if (nOrder == 0)
//        {
//            return BesselI0(arg);
//        }
//        else if (nOrder == 1)
//        {
//            return BesselI1(arg);
//        }
//
//        return  BesselIN(nOrder, arg);
//    }
//
//
//    ComplexF BesselI0<ComplexF>(ComplexF arg)
//    {
//        ComplexF result;
//
//        //int n = 1;
//
//        //complex* z = reinterpret_cast<complex*>(&arg);
//
//        //float *zi = &arg.Imaginary;
//        //float fnu = 0;
//        //int kode = 1;
//        //complex *cy = reinterpret_cast<complex*>(&result);
//        //int nz = 0;
//        //int ierr = 0;
//
//        //int res = cbesi_(z,
//        //    /*fnu : initial order, fnu>=0*/ &fnu,
//        //    /*kode: A parameter to indicate the scaling option*/ &kode,
//        //    /*n   : Number of terms in the sequence, n>=1*/ &n,
//        //    /*out CYR : real part of result vector */ cy,
//        //    /*out NZ  : Number of underflows set to zero */ &nz,
//        //    /*out IERR: Error flag */ &ierr);
//
//        return result;
//    }
//
//
//    ComplexF BesselI1<ComplexF>(ComplexF arg)
//    {
//        ComplexF result;
//
//        //int n = 1;
//
//        //complex* z = reinterpret_cast<complex*>(&arg);
//
//        //float *zi = &arg.Imaginary;
//        //float fnu = 1;
//        //int kode = 1;
//        //complex *cy = reinterpret_cast<complex*>(&result);
//        //int nz = 0;
//        //int ierr = 0;
//
//        //int res = cbesi_(z,
//        //    /*fnu : initial order, fnu>=0*/ &fnu,
//        //    /*kode: A parameter to indicate the scaling option*/ &kode,
//        //    /*n   : Number of terms in the sequence, n>=1*/ &n,
//        //    /*out CYR : real part of result vector */ cy,
//        //    /*out NZ  : Number of underflows set to zero */ &nz,
//        //    /*out IERR: Error flag */ &ierr);
//
//        return result;
//    }
//
//
//    ComplexF BesselIN<ComplexF>(const int nOrder, ComplexF arg)
//    {
//        ComplexF result;
//
//        //int n = 1;
//
//        //complex* z = reinterpret_cast<complex*>(&arg);
//
//        //float *zi = &arg.Imaginary;
//        //float fnu = nOrder;
//        //int kode = 1;
//        //complex *cy = reinterpret_cast<complex*>(&result);
//        //int nz = 0;
//        //int ierr = 0;
//
//        //int res = cbesi_(z,
//        //    /*fnu : initial order, fnu>=0*/ &fnu,
//        //    /*kode: A parameter to indicate the scaling option*/ &kode,
//        //    /*n   : Number of terms in the sequence, n>=1*/ &n,
//        //    /*out CYR : real part of result vector */ cy,
//        //    /*out NZ  : Number of underflows set to zero */ &nz,
//        //    /*out IERR: Error flag */ &ierr);
//
//        return result;
//    }
//#pragma endregion
//
//#pragma region BesselIE
//
//    ComplexF BesselIE<ComplexF>(const int nOrder, ComplexF arg)
//    {
//        if (nOrder == 0)
//        {
//            return BesselIE0(arg);
//        }
//        else if (nOrder == 1)
//        {
//            return BesselIE1(arg);
//        }
//
//        return  BesselIEN(nOrder, arg);
//    }
//
//
//    ComplexF BesselIE0<ComplexF>(ComplexF arg)
//    {
//        ComplexF result;
//
//        //int n = 1;
//
//        //complex* z = reinterpret_cast<complex*>(&arg);
//
//        //float *zi = &arg.Imaginary;
//        //float fnu = 0;
//        //int kode = 2;
//        //complex *cy = reinterpret_cast<complex*>(&result);
//        //int nz = 0;
//        //int ierr = 0;
//
//        //int res = cbesi_(z,
//        //    /*fnu : initial order, fnu>=0*/ &fnu,
//        //    /*kode: A parameter to indicate the scaling option*/ &kode,
//        //    /*n   : Number of terms in the sequence, n>=1*/ &n,
//        //    /*out CYR : real part of result vector */ cy,
//        //    /*out NZ  : Number of underflows set to zero */ &nz,
//        //    /*out IERR: Error flag */ &ierr);
//
//        return result;
//    }
//
//
//    ComplexF BesselIE1<ComplexF>(ComplexF arg)
//    {
//        ComplexF result;
//
//        //int n = 1;
//
//        //complex* z = reinterpret_cast<complex*>(&arg);
//
//        //float *zi = &arg.Imaginary;
//        //float fnu = 1;
//        //int kode = 2;
//        //complex *cy = reinterpret_cast<complex*>(&result);
//        //int nz = 0;
//        //int ierr = 0;
//
//        //int res = cbesi_(z,
//        //    /*fnu : initial order, fnu>=0*/ &fnu,
//        //    /*kode: A parameter to indicate the scaling option*/ &kode,
//        //    /*n   : Number of terms in the sequence, n>=1*/ &n,
//        //    /*out CYR : real part of result vector */ cy,
//        //    /*out NZ  : Number of underflows set to zero */ &nz,
//        //    /*out IERR: Error flag */ &ierr);
//
//        return result;
//    }
//
//
//    ComplexF BesselIEN<ComplexF>(const int nOrder, ComplexF arg)
//    {
//        ComplexF result;
//
//        //int n = 1;
//
//        //complex* z = reinterpret_cast<complex*>(&arg);
//
//        //float *zi = &arg.Imaginary;
//        //float fnu = nOrder;
//        //int kode = 2;
//        //complex *cy = reinterpret_cast<complex*>(&result);
//        //int nz = 0;
//        //int ierr = 0;
//
//        //int res = cbesi_(z,
//        //    /*fnu : initial order, fnu>=0*/ &fnu,
//        //    /*kode: A parameter to indicate the scaling option*/ &kode,
//        //    /*n   : Number of terms in the sequence, n>=1*/ &n,
//        //    /*out CYR : real part of result vector */ cy,
//        //    /*out NZ  : Number of underflows set to zero */ &nz,
//        //    /*out IERR: Error flag */ &ierr);
//
//        return result;
//    }
//#pragma endregion
//#pragma endregion
//
//#pragma region BesselI ComplexD
//#pragma region BesselI
//
//    ComplexD BesselI<ComplexD>(const int nOrder, ComplexD arg)
//    {
//        if (nOrder == 0)
//        {
//            return BesselI0(arg);
//        }
//        else if (nOrder == 1)
//        {
//            return BesselI1(arg);
//        }
//
//        return  BesselIN(nOrder, arg);
//    }
//
//
//    ComplexD BesselI0<ComplexD>(ComplexD arg)
//    {
//        ComplexD result;
//
//        //int n = 1;
//        ////double *zr = new double[1]{ arg.Real };
//        ////double *zi = new double[1]{ arg.Imaginary };
//        //double *zr = &arg.Real;
//        //double *zi = &arg.Imaginary;
//        //double fnu = 0;
//        //int kode = 1;
//        //double *cyr = &result.Real;
//        //double *cyi = &result.Imaginary;
//        //int nz = 0;
//        //int ierr = 0;
//
//        //int res = zbesi_(zr, zi,
//        //    /*fnu : initial order, fnu>=0*/ &fnu,
//        //    /*kode: A parameter to indicate the scaling option*/ &kode,
//        //    /*n   : Number of terms in the sequence, n>=1*/ &n,
//        //    /*out CYR : real part of result vector */ cyr,
//        //    /*out CYI : imag part of result vector */ cyi,
//        //    /*out NZ  : Number of underflows set to zero */ &nz,
//        //    /*out IERR: Error flag */ &ierr);
//
//        return result;
//    }
//
//
//    ComplexD BesselI1<ComplexD>(ComplexD arg)
//    {
//        ComplexD result;
//
//        //int n = 1;
//        ////double *zr = new double[1]{ arg.Real };
//        ////double *zi = new double[1]{ arg.Imaginary };
//        //double *zr = &arg.Real;
//        //double *zi = &arg.Imaginary;
//        //double fnu = 1;
//        //int kode = 1;
//        //double *cyr = &result.Real;
//        //double *cyi = &result.Imaginary;
//        //int nz = 0;
//        //int ierr = 0;
//
//        //int res = zbesi_(zr, zi,
//        //    /*fnu : initial order, fnu>=0*/ &fnu,
//        //    /*kode: A parameter to indicate the scaling option*/ &kode,
//        //    /*n   : Number of terms in the sequence, n>=1*/ &n,
//        //    /*out CYR : real part of result vector */ cyr,
//        //    /*out CYI : imag part of result vector */ cyi,
//        //    /*out NZ  : Number of underflows set to zero */ &nz,
//        //    /*out IERR: Error flag */ &ierr);
//
//        return result;
//    }
//
//
//    ComplexD BesselIN<ComplexD>(const int nOrder, ComplexD arg)
//    {
//        ComplexD result;
//
//        //int n = 1;
//        ////double *zr = new double[1]{ arg.Real };
//        ////double *zi = new double[1]{ arg.Imaginary };
//        //double *zr = &arg.Real;
//        //double *zi = &arg.Imaginary;
//        //double fnu = nOrder;
//        //int kode = 1;
//        //double *cyr = &result.Real;
//        //double *cyi = &result.Imaginary;
//        //int nz = 0;
//        //int ierr = 0;
//
//        //int res = zbesi_(zr, zi,
//        //    /*fnu : initial order, fnu>=0*/ &fnu,
//        //    /*kode: A parameter to indicate the scaling option*/ &kode,
//        //    /*n   : Number of terms in the sequence, n>=1*/ &n,
//        //    /*out CYR : real part of result vector */ cyr,
//        //    /*out CYI : imag part of result vector */ cyi,
//        //    /*out NZ  : Number of underflows set to zero */ &nz,
//        //    /*out IERR: Error flag */ &ierr);
//
//        return result;
//    }
//#pragma endregion
//
//#pragma region BesselIE
//
//    ComplexD BesselIE<ComplexD>(const int nOrder, ComplexD arg)
//    {
//        if (nOrder == 0)
//        {
//            return BesselIE0(arg);
//        }
//        else if (nOrder == 1)
//        {
//            return BesselIE1(arg);
//        }
//
//        return  BesselIEN(nOrder, arg);
//    }
//
//
//    ComplexD BesselIE0<ComplexD>(ComplexD arg)
//    {
//        ComplexD result;
//
//        //int n = 1;
//        ////double *zr = new double[1]{ arg.Real };
//        ////double *zi = new double[1]{ arg.Imaginary };
//        //double *zr = &arg.Real;
//        //double *zi = &arg.Imaginary;
//        //double fnu = 0;
//        //int kode = 2;
//        //double *cyr = &result.Real;
//        //double *cyi = &result.Imaginary;
//        //int nz = 0;
//        //int ierr = 0;
//
//        //int res = zbesi_(zr, zi,
//        //    /*fnu : initial order, fnu>=0*/ &fnu,
//        //    /*kode: A parameter to indicate the scaling option*/ &kode,
//        //    /*n   : Number of terms in the sequence, n>=1*/ &n,
//        //    /*out CYR : real part of result vector */ cyr,
//        //    /*out CYI : imag part of result vector */ cyi,
//        //    /*out NZ  : Number of underflows set to zero */ &nz,
//        //    /*out IERR: Error flag */ &ierr);
//
//        return result;
//    }
//
//
//    ComplexD BesselIE1<ComplexD>(ComplexD arg)
//    {
//        ComplexD result;
//
//        //int n = 1;
//        ////double *zr = new double[1]{ arg.Real };
//        ////double *zi = new double[1]{ arg.Imaginary };
//        //double *zr = &arg.Real;
//        //double *zi = &arg.Imaginary;
//        //double fnu = 1;
//        //int kode = 2;
//        //double *cyr = &result.Real;
//        //double *cyi = &result.Imaginary;
//        //int nz = 0;
//        //int ierr = 0;
//
//        //int res = zbesi_(zr, zi,
//        //    /*fnu : initial order, fnu>=0*/ &fnu,
//        //    /*kode: A parameter to indicate the scaling option*/ &kode,
//        //    /*n   : Number of terms in the sequence, n>=1*/ &n,
//        //    /*out CYR : real part of result vector */ cyr,
//        //    /*out CYI : imag part of result vector */ cyi,
//        //    /*out NZ  : Number of underflows set to zero */ &nz,
//        //    /*out IERR: Error flag */ &ierr);
//
//        return result;
//    }
//
//
//    ComplexD BesselIEN<ComplexD>(const int nOrder, ComplexD arg)
//    {
//        ComplexD result;
//
//        //int n = 1;
//        ////double *zr = new double[1]{ arg.Real };
//        ////double *zi = new double[1]{ arg.Imaginary };
//        //double *zr = &arg.Real;
//        //double *zi = &arg.Imaginary;
//        //double fnu = nOrder;
//        //int kode = 2;
//        //double *cyr = &result.Real;
//        //double *cyi = &result.Imaginary;
//        //int nz = 0;
//        //int ierr = 0;
//
//        //int res = zbesi_(zr, zi,
//        //    /*fnu : initial order, fnu>=0*/ &fnu,
//        //    /*kode: A parameter to indicate the scaling option*/ &kode,
//        //    /*n   : Number of terms in the sequence, n>=1*/ &n,
//        //    /*out CYR : real part of result vector */ cyr,
//        //    /*out CYI : imag part of result vector */ cyi,
//        //    /*out NZ  : Number of underflows set to zero */ &nz,
//        //    /*out IERR: Error flag */ &ierr);
//
//        return result;
//    }
//#pragma endregion
//#pragma endregion

#pragma region BesselK float
#pragma region BesselK

KOKKOS_FUNCTION float BesselK(const int nOrder, const float arg)
{
    if(nOrder == 0)
    {
        return BesselK0(arg);
    }
    if(nOrder == 1)
    {
        return BesselK1(arg);
    }

    return BesselKN(nOrder, arg);
}

KOKKOS_FUNCTION static int calck0(const float* arg, float* result, const int* jint)
{
    static const float one = 1.0E0f;
    // static const float one5 = 15.0E0f;
    // static const float exp40 = 2.353852668370199854E17f;
    // static const float forty = 40.0E0f;
    // static const float rec15 = 6.6666666666666666666E-2f;
    // static const float two25 = 225.0E0f;
    static const float zero = 0E0f;

    static const float xsmall = 2.98E-8f;
    static const float xinf   = 3.40E38f;
    static const float xmax   = 91.9E0f;

    static const float p[6] = {5.8599221412826100000E-04f,
                               1.3166052564989571850E-01f,
                               1.1999463724910714109E+01f,
                               4.6850901201934832188E+02f,
                               5.9169059852270512312E+03f,
                               2.4708152720399552679E+03f};

    static const float q[2] = {-2.4994418972832303646E+02f, 2.1312714303849120380E+04f};

    static const float f[4] = {-1.6414452837299064100E+00f, -2.9601657892958843866E+02f, -1.7733784684952985886E+04f, -4.0320340761145482298E+05f};

    static const float g[3] = {-2.5064972445877992730E+02f, 2.9865713163054025489E+04f, -1.6128136304458193998E+06f};

    static const float pp[10] = {1.1394980557384778174E+02f,
                                 3.6832589957340267940E+03f,
                                 3.1075408980684392399E+04f,
                                 1.0577068948034021957E+05f,
                                 1.7398867902565686251E+05f,
                                 1.5097646353289914539E+05f,
                                 7.1557062783764037541E+04f,
                                 1.8321525870183537725E+04f,
                                 2.3444738764199315021E+03f,
                                 1.1600249425076035558E+02f};

    static const float qq[10] = {2.0013443064949242491E+02f,
                                 4.4329628889746408858E+03f,
                                 3.1474655750295278825E+04f,
                                 9.7418829762268075784E+04f,
                                 1.5144644673520157801E+05f,
                                 1.2689839587977598727E+05f,
                                 5.8824616785857027752E+04f,
                                 1.4847228371802360957E+04f,
                                 1.8821890840982713696E+03f,
                                 9.2556599177304839811E+01f};

    float xx;
    float sump;
    float sumq;

    float x = *arg;
    if(x > zero)
    {
        if(x <= one)
        {
            float temp = log(x);
            if(x < xsmall)
            {
                *result = p[5] / q[1] - temp;
            }
            else
            {
                xx   = x * x;
                sump = ((((p[0] * xx + p[1]) * xx + p[2]) * xx + p[3]) * xx + p[4]) * xx + p[5];

                sumq = (xx + q[0]) * xx + q[1];

                float sumf = ((f[0] * xx + f[1]) * xx + f[2]) * xx + f[3];

                float sumg = ((xx + g[0]) * xx + g[1]) * xx + g[2];

                *result = sump / sumq - xx * sumf * temp / sumg - temp;

                if(*jint == 2)
                {
                    *result *= exp(x);
                }
            }
        }
        else if(*jint == 1 && x > xmax)
        {
            *result = zero;
        }
        else
        {
            xx   = one / x;
            sump = pp[0];
            for(int i = 2; i <= 10; ++i)
            {
                sump = sump * xx + pp[i - 1];
                /* L120: */
            }
            sumq = xx;
            for(int i = 1; i <= 9; ++i)
            {
                sumq = (sumq + qq[i - 1]) * xx;
                /* L140: */
            }
            sumq += qq[9];
            *result = sump / sumq / sqrt(x);
            if(*jint == 1)
            {
                *result *= exp(-x);
            }
        }
    }
    else
    {
        *result = xinf;
    }

    return 0;
}

KOKKOS_FUNCTION float BesselK0(const float arg)
{
    int   jint;
    float result;

    jint = 1;
    calck0(&arg, &result, &jint);
    float ret_val = result;
    return ret_val;
}

KOKKOS_FUNCTION static int calck1(const float* arg, float* result, const int* jint)
{
    static const float one = 1.0E0f;
    // static const float one5 = 15.0E0f;
    // static const float exp40 = 2.353852668370199854E17f;
    // static const float forty = 40.0E0f;
    // static const float rec15 = 6.6666666666666666666E-2f;
    // static const float two25 = 225.0E0f;
    // static const float half = 0.5E0f;
    static const float zero = 0.0E0f;

    // static const float pbar = 3.98437500E-01f;

    static const float xsmall = 2.98E-8f;
    static const float xinf   = 3.40E38f;
    static const float xmax   = 91.9E0f;
    static const float xleast = 1.18E-38f;

    static const float p[5] = {4.8127070456878442310E-1f, 9.9991373567429309922E+1f, 7.1885382604084798576E+3f, 1.7733324035147015630E+5f, 7.1938920065420586101E+5f};

    static const float q[3] = {-2.8143915754538725829E+2f, 3.7264298672067697862E+4f, -2.2149374878243304548E+6f};

    static const float f[5] = {-2.2795590826955002390E-1f, -5.3103913335180275253E+1f, -4.5051623763436087023E+3f, -1.4758069205414222471E+5f, -1.3531161492785421328E+6f};

    static const float g[3] = {-3.0507151578787595807E+2f, 4.3117653211351080007E+4f, -2.7062322985570842656E+6f};

    static const float pp[11] = {6.4257745859173138767E-2f,
                                 7.5584584631176030810E+0f,
                                 1.3182609918569941308E+2f,
                                 8.1094256146537402173E+2f,
                                 2.3123742209168871550E+3f,
                                 3.4540675585544584407E+3f,
                                 2.8590657697910288226E+3f,
                                 1.3319486433183221990E+3f,
                                 3.4122953486801312910E+2f,
                                 4.4137176114230414036E+1f,
                                 2.2196792496874548962E+0f};

    static const float qq[9] = {3.6001069306861518855E+1f,
                                3.3031020088765390854E+2f,
                                1.2082692316002348638E+3f,
                                2.1181000487171943810E+3f,
                                1.9448440788918006154E+3f,
                                9.6929165726802648634E+2f,
                                2.5951223655579051357E+2f,
                                3.4552228452758912848E+1f,
                                1.7710478032601086579E+0f};

    float xx;
    float sump;
    float sumq;

    float x = *arg;
    if(x < xleast)
    {
        *result = xinf;
    }
    else if(x <= one)
    {
        if(x < xsmall)
        {
            *result = one / x;
        }
        else
        {
            xx = x * x;

            sump = ((((p[0] * xx + p[1]) * xx + p[2]) * xx + p[3]) * xx + p[4]) * xx + q[2];

            sumq = ((xx + q[0]) * xx + q[1]) * xx + q[2];

            float sumf = (((f[0] * xx + f[1]) * xx + f[2]) * xx + f[3]) * xx + f[4];

            float sumg = ((xx + g[0]) * xx + g[1]) * xx + g[2];

            *result = (xx * log(x) * sumf / sumg + sump / sumq) / x;

            if(*jint == 2)
            {
                *result *= exp(x);
            }
        }
    }
    else if(*jint == 1 && x > xmax)
    {
        /* -------------------------------------------------------------------- */
        /*  Error return for  ARG  .GT. XMAX */
        /* -------------------------------------------------------------------- */
        *result = zero;
    }
    else
    {
        /* -------------------------------------------------------------------- */
        /*  1.0 .LT.  ARG */
        /* -------------------------------------------------------------------- */
        xx   = one / x;
        sump = pp[0];
        for(int i = 2; i <= 11; ++i)
        {
            sump = sump * xx + pp[i - 1];
            /* L120: */
        }
        sumq = xx;

        for(int i = 1; i <= 8; ++i)
        {
            sumq = (sumq + qq[i - 1]) * xx;
            /* L140: */
        }

        sumq += qq[8];

        *result = sump / sumq / sqrt(x);
        if(*jint == 1)
        {
            *result *= exp(-x);
        }
    }

    return 0;
}

KOKKOS_FUNCTION float BesselK1(const float arg)
{
    int   jint;
    float result;

    jint = 1;
    calck1(&arg, &result, &jint);
    float ret_val = result;
    return ret_val;
}

KOKKOS_FUNCTION float BesselKN(const int nOrder, const float arg)
{
    System::out << "BesselKN Exception" << System::endl;
    return Constants<float>::NaN();
}
#pragma endregion

#pragma region BesselKE

KOKKOS_FUNCTION float BesselKE(const int nOrder, const float arg)
{
    if(nOrder == 0)
    {
        return BesselKE0(arg);
    }
    if(nOrder == 1)
    {
        return BesselKE1(arg);
    }

    return BesselKEN(nOrder, arg);
}

KOKKOS_FUNCTION float BesselKE0(const float arg)
{
    int   jint;
    float result;

    jint = 2;
    calck0(&arg, &result, &jint);
    float ret_val = result;
    return ret_val;
}

KOKKOS_FUNCTION float BesselKE1(const float arg)
{
    int   jint;
    float result;

    jint = 2;
    calck1(&arg, &result, &jint);
    float ret_val = result;
    return ret_val;
}

KOKKOS_FUNCTION float BesselKEN(const int nOrder, const float arg)
{
    System::out << "BesselKEN Exception" << System::endl;
    return Constants<float>::NaN();
}
#pragma endregion
#pragma endregion

#pragma region BesselK double
#pragma region BesselK

KOKKOS_FUNCTION double BesselK(const int nOrder, const double arg)
{
    if(nOrder == 0)
    {
        return BesselK0(arg);
    }
    if(nOrder == 1)
    {
        return BesselK1(arg);
    }

    return BesselKN(nOrder, arg);
}

KOKKOS_FUNCTION static int calck0(const double* arg, double* result, const int* jint)
{
    static const double one = 1.0E0;
    // static const double one5 = 15.0E0;
    // static const double exp40 = 2.353852668370199854E17;
    // static const double forty = 40.0E0;
    // static const double rec15 = 6.6666666666666666666E-2;
    // static const double two25 = 225.0E0;
    static const double zero = 0E0;

    static const double xsmall = 2.98E-8;
    static const double xinf   = 3.40E38;
    static const double xmax   = 91.9E0;

    static const double p[6] = {
        5.8599221412826100000E-04, 1.3166052564989571850E-01, 1.1999463724910714109E+01, 4.6850901201934832188E+02, 5.9169059852270512312E+03, 2.4708152720399552679E+03};

    static const double q[2] = {-2.4994418972832303646E+02, 2.1312714303849120380E+04};

    static const double f[4] = {-1.6414452837299064100E+00, -2.9601657892958843866E+02, -1.7733784684952985886E+04, -4.0320340761145482298E+05};

    static const double g[3] = {-2.5064972445877992730E+02, 2.9865713163054025489E+04, -1.6128136304458193998E+06};

    static const double pp[10] = {1.1394980557384778174E+02,
                                  3.6832589957340267940E+03,
                                  3.1075408980684392399E+04,
                                  1.0577068948034021957E+05,
                                  1.7398867902565686251E+05,
                                  1.5097646353289914539E+05,
                                  7.1557062783764037541E+04,
                                  1.8321525870183537725E+04,
                                  2.3444738764199315021E+03,
                                  1.1600249425076035558E+02};

    static const double qq[10] = {2.0013443064949242491E+02,
                                  4.4329628889746408858E+03,
                                  3.1474655750295278825E+04,
                                  9.7418829762268075784E+04,
                                  1.5144644673520157801E+05,
                                  1.2689839587977598727E+05,
                                  5.8824616785857027752E+04,
                                  1.4847228371802360957E+04,
                                  1.8821890840982713696E+03,
                                  9.2556599177304839811E+01};

    double xx;
    double sump;
    double sumq;

    double x = *arg;
    if(x > zero)
    {
        if(x <= one)
        {
            double temp = log(x);
            if(x < xsmall)
            {
                *result = p[5] / q[1] - temp;
            }
            else
            {
                xx   = x * x;
                sump = ((((p[0] * xx + p[1]) * xx + p[2]) * xx + p[3]) * xx + p[4]) * xx + p[5];

                sumq = (xx + q[0]) * xx + q[1];

                double sumf = ((f[0] * xx + f[1]) * xx + f[2]) * xx + f[3];

                double sumg = ((xx + g[0]) * xx + g[1]) * xx + g[2];

                *result = sump / sumq - xx * sumf * temp / sumg - temp;

                if(*jint == 2)
                {
                    *result *= exp(x);
                }
            }
        }
        else if(*jint == 1 && x > xmax)
        {
            *result = zero;
        }
        else
        {
            xx   = one / x;
            sump = pp[0];
            for(int i = 2; i <= 10; ++i)
            {
                sump = sump * xx + pp[i - 1];
                /* L120: */
            }
            sumq = xx;
            for(int i = 1; i <= 9; ++i)
            {
                sumq = (sumq + qq[i - 1]) * xx;
                /* L140: */
            }
            sumq += qq[9];
            *result = sump / sumq / sqrt(x);
            if(*jint == 1)
            {
                *result *= exp(-x);
            }
        }
    }
    else
    {
        *result = xinf;
    }

    return 0;
}

KOKKOS_FUNCTION double BesselK0(const double arg)
{
    int    jint;
    double result;

    jint = 1;
    calck0(&arg, &result, &jint);
    double ret_val = result;
    return ret_val;
}

KOKKOS_FUNCTION static int calck1(const double* arg, double* result, const int* jint)
{
    static const double one = 1.0E0;
    // static const double one5 = 15.0E0;
    // static const double exp40 = 2.353852668370199854E17;
    // static const double forty = 40.0E0;
    // static const double rec15 = 6.6666666666666666666E-2;
    // static const double two25 = 225.0E0;
    // static const double half = 0.5E0;
    static const double zero = 0.0E0;

    // static const double pbar = 3.98437500E-01;

    static const double xsmall = 2.98E-8;
    static const double xinf   = 3.40E38;
    static const double xmax   = 91.9E0;
    static const double xleast = 1.18E-38;

    static const double p[5] = {4.8127070456878442310E-1, 9.9991373567429309922E+1, 7.1885382604084798576E+3, 1.7733324035147015630E+5, 7.1938920065420586101E+5};

    static const double q[3] = {-2.8143915754538725829E+2, 3.7264298672067697862E+4, -2.2149374878243304548E+6};

    static const double f[5] = {-2.2795590826955002390E-1, -5.3103913335180275253E+1, -4.5051623763436087023E+3, -1.4758069205414222471E+5, -1.3531161492785421328E+6};

    static const double g[3] = {-3.0507151578787595807E+2, 4.3117653211351080007E+4, -2.7062322985570842656E+6};

    static const double pp[11] = {6.4257745859173138767E-2,
                                  7.5584584631176030810E+0,
                                  1.3182609918569941308E+2,
                                  8.1094256146537402173E+2,
                                  2.3123742209168871550E+3,
                                  3.4540675585544584407E+3,
                                  2.8590657697910288226E+3,
                                  1.3319486433183221990E+3,
                                  3.4122953486801312910E+2,
                                  4.4137176114230414036E+1,
                                  2.2196792496874548962E+0};

    static const double qq[9] = {3.6001069306861518855E+1,
                                 3.3031020088765390854E+2,
                                 1.2082692316002348638E+3,
                                 2.1181000487171943810E+3,
                                 1.9448440788918006154E+3,
                                 9.6929165726802648634E+2,
                                 2.5951223655579051357E+2,
                                 3.4552228452758912848E+1,
                                 1.7710478032601086579E+0};

    double xx;
    double sump;
    double sumq;

    double x = *arg;
    if(x < xleast)
    {
        *result = xinf;
    }
    else if(x <= one)
    {
        if(x < xsmall)
        {
            *result = one / x;
        }
        else
        {
            xx = x * x;

            sump = ((((p[0] * xx + p[1]) * xx + p[2]) * xx + p[3]) * xx + p[4]) * xx + q[2];

            sumq = ((xx + q[0]) * xx + q[1]) * xx + q[2];

            double sumf = (((f[0] * xx + f[1]) * xx + f[2]) * xx + f[3]) * xx + f[4];

            double sumg = ((xx + g[0]) * xx + g[1]) * xx + g[2];

            *result = (xx * log(x) * sumf / sumg + sump / sumq) / x;

            if(*jint == 2)
            {
                *result *= exp(x);
            }
        }
    }
    else if(*jint == 1 && x > xmax)
    {
        /* -------------------------------------------------------------------- */
        /*  Error return for  ARG  .GT. XMAX */
        /* -------------------------------------------------------------------- */
        *result = zero;
    }
    else
    {
        /* -------------------------------------------------------------------- */
        /*  1.0 .LT.  ARG */
        /* -------------------------------------------------------------------- */
        xx   = one / x;
        sump = pp[0];
        for(int i = 2; i <= 11; ++i)
        {
            sump = sump * xx + pp[i - 1];
            /* L120: */
        }
        sumq = xx;

        for(int i = 1; i <= 8; ++i)
        {
            sumq = (sumq + qq[i - 1]) * xx;
            /* L140: */
        }

        sumq += qq[8];

        *result = sump / sumq / sqrt(x);
        if(*jint == 1)
        {
            *result *= exp(-x);
        }
    }
    return 0;
}

KOKKOS_FUNCTION double BesselK1(const double arg)
{
    double result;

    int jint = 1;
    calck1(&arg, &result, &jint);
    double ret_val = result;
    return ret_val;
}

KOKKOS_FUNCTION double BesselKN(const int nOrder, const double arg)
{
    System::out << "BesselKN Exception" << System::endl;
    return Constants<double>::NaN();
}
#pragma endregion

#pragma region BesselKE

KOKKOS_FUNCTION double BesselKE(const int nOrder, const double arg)
{
    if(nOrder == 0)
    {
        return BesselKE0(arg);
    }

    if(nOrder == 1)
    {
        return BesselKE1(arg);
    }

    return BesselKEN(nOrder, arg);
}

KOKKOS_FUNCTION double BesselKE0(const double arg)
{
    double result;

    int jint = 2;
    calck0(&arg, &result, &jint);
    const double ret_val = result;
    return ret_val;
}

KOKKOS_FUNCTION double BesselKE1(const double arg)
{
    int    jint;
    double result;

    jint = 2;
    calck1(&arg, &result, &jint);
    double ret_val = result;
    return ret_val;
}

KOKKOS_FUNCTION double BesselKEN(const int nOrder, const double arg)
{
    System::out << "BesselKEN Exception" << System::endl;
    return Constants<double>::NaN();
}
#pragma endregion
#pragma endregion

//#pragma region BesselK ComplexF
//#pragma region BesselK
//
//    ComplexF BesselK<ComplexF>(const int nOrder, ComplexF arg)
//    {
//        if (nOrder == 0)
//        {
//            return BesselK0(arg);
//        }
//        else if (nOrder == 1)
//        {
//            return BesselK1(arg);
//        }
//
//        return  BesselKN(nOrder, arg);
//    }
//
//
//    ComplexF BesselK0<ComplexF>(ComplexF arg)
//    {
//        ComplexF result;
//
//        //int n = 1;
//
//        //complex* z = reinterpret_cast<complex*>(&arg);
//
//        //float *zi = &arg.Imaginary;
//        //float fnu = 0;
//        //int kode = 1;
//        //complex *cy = reinterpret_cast<complex*>(&result);
//        //int nz = 0;
//        //int ierr = 0;
//
//        //int res = cbesk_(z,
//        //    /*fnu : initial order, fnu>=0*/ &fnu,
//        //    /*kode: A parameter to indicate the scaling option*/ &kode,
//        //    /*n   : Number of terms in the sequence, n>=1*/ &n,
//        //    /*out CYR : real part of result vector */ cy,
//        //    /*out NZ  : Number of underflows set to zero */ &nz,
//        //    /*out IERR: Error flag */ &ierr);
//
//        return result;
//    }
//
//
//    ComplexF BesselK1<ComplexF>(ComplexF arg)
//    {
//        ComplexF result;
//
//        //int n = 1;
//
//        //complex* z = reinterpret_cast<complex*>(&arg);
//
//        //float *zi = &arg.Imaginary;
//        //float fnu = 1;
//        //int kode = 1;
//        //complex *cy = reinterpret_cast<complex*>(&result);
//        //int nz = 0;
//        //int ierr = 0;
//
//        //int res = cbesk_(z,
//        //    /*fnu : initial order, fnu>=0*/ &fnu,
//        //    /*kode: A parameter to indicate the scaling option*/ &kode,
//        //    /*n   : Number of terms in the sequence, n>=1*/ &n,
//        //    /*out CYR : real part of result vector */ cy,
//        //    /*out NZ  : Number of underflows set to zero */ &nz,
//        //    /*out IERR: Error flag */ &ierr);
//
//        return result;
//    }
//
//
//    ComplexF BesselKN<ComplexF>(const int nOrder, ComplexF arg)
//    {
//        ComplexF result;
//
//        //int n = 1;
//
//        //complex* z = reinterpret_cast<complex*>(&arg);
//
//        //float *zi = &arg.Imaginary;
//        //float fnu = nOrder;
//        //int kode = 1;
//        //complex *cy = reinterpret_cast<complex*>(&result);
//        //int nz = 0;
//        //int ierr = 0;
//
//        //int res = cbesk_(z,
//        //    /*fnu : initial order, fnu>=0*/ &fnu,
//        //    /*kode: A parameter to indicate the scaling option*/ &kode,
//        //    /*n   : Number of terms in the sequence, n>=1*/ &n,
//        //    /*out CYR : real part of result vector */ cy,
//        //    /*out NZ  : Number of underflows set to zero */ &nz,
//        //    /*out IERR: Error flag */ &ierr);
//
//        return result;
//    }
//#pragma endregion
//
//#pragma region BesselKE
//
//    ComplexF BesselKE<ComplexF>(const int nOrder, ComplexF arg)
//    {
//        if (nOrder == 0)
//        {
//            return BesselKE0(arg);
//        }
//        else if (nOrder == 1)
//        {
//            return BesselKE1(arg);
//        }
//
//        return  BesselKEN(nOrder, arg);
//    }
//
//
//    ComplexF BesselKE0<ComplexF>(ComplexF arg)
//    {
//        ComplexF result;
//
//        //int n = 1;
//
//        //complex* z = reinterpret_cast<complex*>(&arg);
//
//        //float *zi = &arg.Imaginary;
//        //float fnu = 0;
//        //int kode = 2;
//        //complex *cy = reinterpret_cast<complex*>(&result);
//        //int nz = 0;
//        //int ierr = 0;
//
//        //int res = cbesk_(z,
//        //    /*fnu : initial order, fnu>=0*/ &fnu,
//        //    /*kode: A parameter to indicate the scaling option*/ &kode,
//        //    /*n   : Number of terms in the sequence, n>=1*/ &n,
//        //    /*out CYR : real part of result vector */ cy,
//        //    /*out NZ  : Number of underflows set to zero */ &nz,
//        //    /*out IERR: Error flag */ &ierr);
//
//        return result;
//    }
//
//
//    ComplexF BesselKE1<ComplexF>(ComplexF arg)
//    {
//        ComplexF result;
//
//        //int n = 1;
//
//        //complex* z = reinterpret_cast<complex*>(&arg);
//
//        //float *zi = &arg.Imaginary;
//        //float fnu = 1;
//        //int kode = 2;
//        //complex *cy = reinterpret_cast<complex*>(&result);
//        //int nz = 0;
//        //int ierr = 0;
//
//        //int res = cbesk_(z,
//        //    /*fnu : initial order, fnu>=0*/ &fnu,
//        //    /*kode: A parameter to indicate the scaling option*/ &kode,
//        //    /*n   : Number of terms in the sequence, n>=1*/ &n,
//        //    /*out CYR : real part of result vector */ cy,
//        //    /*out NZ  : Number of underflows set to zero */ &nz,
//        //    /*out IERR: Error flag */ &ierr);
//
//        return result;
//    }
//
//
//    ComplexF BesselKEN<ComplexF>(const int nOrder, ComplexF arg)
//    {
//        ComplexF result;
//
//        //int n = 1;
//
//        //complex* z = reinterpret_cast<complex*>(&arg);
//
//        //float *zi = &arg.Imaginary;
//        //float fnu = nOrder;
//        //int kode = 2;
//        //complex *cy = reinterpret_cast<complex*>(&result);
//        //int nz = 0;
//        //int ierr = 0;
//
//        //int res = cbesk_(z,
//        //    /*fnu : initial order, fnu>=0*/ &fnu,
//        //    /*kode: A parameter to indicate the scaling option*/ &kode,
//        //    /*n   : Number of terms in the sequence, n>=1*/ &n,
//        //    /*out CYR : real part of result vector */ cy,
//        //    /*out NZ  : Number of underflows set to zero */ &nz,
//        //    /*out IERR: Error flag */ &ierr);
//
//        return result;
//    }
//#pragma endregion
//#pragma endregion
//
//#pragma region BesselK ComplexD
//#pragma region BesselK
//
//    ComplexD BesselK<ComplexD>(const int nOrder, ComplexD arg)
//    {
//        if (nOrder == 0)
//        {
//            return BesselK0(arg);
//        }
//        else if (nOrder == 1)
//        {
//            return BesselK1(arg);
//        }
//
//        return  BesselKN(nOrder, arg);
//    }
//
//
//    ComplexD BesselK0<ComplexD>(ComplexD arg)
//    {
//        ComplexD result;
//
//        //int n = 1;
//        ////double *zr = new double[1]{ arg.Real };
//        ////double *zi = new double[1]{ arg.Imaginary };
//        //double *zr = &arg.Real;
//        //double *zi = &arg.Imaginary;
//        //double fnu = 0;
//        //int kode = 1;
//        //double *cyr = &result.Real;
//        //double *cyi = &result.Imaginary;
//        //int nz = 0;
//        //int ierr = 0;
//
//        //int res = zbesk_(zr, zi,
//        //    /*fnu : initial order, fnu>=0*/ &fnu,
//        //    /*kode: A parameter to indicate the scaling option*/ &kode,
//        //    /*n   : Number of terms in the sequence, n>=1*/ &n,
//        //    /*out CYR : real part of result vector */ cyr,
//        //    /*out CYI : imag part of result vector */ cyi,
//        //    /*out NZ  : Number of underflows set to zero */ &nz,
//        //    /*out IERR: Error flag */ &ierr);
//
//        return result;
//    }
//
//
//    ComplexD BesselK1<ComplexD>(ComplexD arg)
//    {
//        ComplexD result;
//
//        //int n = 1;
//        ////double *zr = new double[1]{ arg.Real };
//        ////double *zi = new double[1]{ arg.Imaginary };
//        //double *zr = &arg.Real;
//        //double *zi = &arg.Imaginary;
//        //double fnu = 1;
//        //int kode = 1;
//        //double *cyr = &result.Real;
//        //double *cyi = &result.Imaginary;
//        //int nz = 0;
//        //int ierr = 0;
//
//        //int res = zbesk_(zr, zi,
//        //    /*fnu : initial order, fnu>=0*/ &fnu,
//        //    /*kode: A parameter to indicate the scaling option*/ &kode,
//        //    /*n   : Number of terms in the sequence, n>=1*/ &n,
//        //    /*out CYR : real part of result vector */ cyr,
//        //    /*out CYI : imag part of result vector */ cyi,
//        //    /*out NZ  : Number of underflows set to zero */ &nz,
//        //    /*out IERR: Error flag */ &ierr);
//
//        return result;
//    }
//
//
//    ComplexD BesselKN<ComplexD>(const int nOrder, ComplexD arg)
//    {
//        ComplexD result;
//
//        //int n = 1;
//        ////double *zr = new double[1]{ arg.Real };
//        ////double *zi = new double[1]{ arg.Imaginary };
//        //double *zr = &arg.Real;
//        //double *zi = &arg.Imaginary;
//        //double fnu = nOrder;
//        //int kode = 1;
//        //double *cyr = &result.Real;
//        //double *cyi = &result.Imaginary;
//        //int nz = 0;
//        //int ierr = 0;
//
//        //int res = zbesk_(zr, zi,
//        //    /*fnu : initial order, fnu>=0*/ &fnu,
//        //    /*kode: A parameter to indicate the scaling option*/ &kode,
//        //    /*n   : Number of terms in the sequence, n>=1*/ &n,
//        //    /*out CYR : real part of result vector */ cyr,
//        //    /*out CYI : imag part of result vector */ cyi,
//        //    /*out NZ  : Number of underflows set to zero */ &nz,
//        //    /*out IERR: Error flag */ &ierr);
//
//        return result;
//    }
//#pragma endregion
//
//#pragma region BesselKE
//
//    ComplexD BesselKE<ComplexD>(const int nOrder, ComplexD arg)
//    {
//        if (nOrder == 0)
//        {
//            return BesselKE0(arg);
//        }
//        else if (nOrder == 1)
//        {
//            return BesselKE1(arg);
//        }
//
//        return  BesselKEN(nOrder, arg);
//    }
//
//
//    ComplexD BesselKE0<ComplexD>(ComplexD arg)
//    {
//        ComplexD result;
//
//        //int n = 1;
//        ////double *zr = new double[1]{ arg.Real };
//        ////double *zi = new double[1]{ arg.Imaginary };
//        //double *zr = &arg.Real;
//        //double *zi = &arg.Imaginary;
//        //double fnu = 0;
//        //int kode = 2;
//        //double *cyr = &result.Real;
//        //double *cyi = &result.Imaginary;
//        //int nz = 0;
//        //int ierr = 0;
//
//        //int res = zbesk_(zr, zi,
//        //    /*fnu : initial order, fnu>=0*/ &fnu,
//        //    /*kode: A parameter to indicate the scaling option*/ &kode,
//        //    /*n   : Number of terms in the sequence, n>=1*/ &n,
//        //    /*out CYR : real part of result vector */ cyr,
//        //    /*out CYI : imag part of result vector */ cyi,
//        //    /*out NZ  : Number of underflows set to zero */ &nz,
//        //    /*out IERR: Error flag */ &ierr);
//
//        return result;
//    }
//
//
//    ComplexD BesselKE1<ComplexD>(ComplexD arg)
//    {
//        ComplexD result;
//
//        //int n = 1;
//        ////double *zr = new double[1]{ arg.Real };
//        ////double *zi = new double[1]{ arg.Imaginary };
//        //double *zr = &arg.Real;
//        //double *zi = &arg.Imaginary;
//        //double fnu = 1;
//        //int kode = 2;
//        //double *cyr = &result.Real;
//        //double *cyi = &result.Imaginary;
//        //int nz = 0;
//        //int ierr = 0;
//
//        //int res = zbesk_(zr, zi,
//        //    /*fnu : initial order, fnu>=0*/ &fnu,
//        //    /*kode: A parameter to indicate the scaling option*/ &kode,
//        //    /*n   : Number of terms in the sequence, n>=1*/ &n,
//        //    /*out CYR : real part of result vector */ cyr,
//        //    /*out CYI : imag part of result vector */ cyi,
//        //    /*out NZ  : Number of underflows set to zero */ &nz,
//        //    /*out IERR: Error flag */ &ierr);
//
//        return result;
//    }
//
//
//    ComplexD BesselKEN<ComplexD>(const int nOrder, ComplexD arg)
//    {
//        ComplexD result;
//
//        //int n = 1;
//        ////double *zr = new double[1]{ arg.Real };
//        ////double *zi = new double[1]{ arg.Imaginary };
//        //double *zr = &arg.Real;
//        //double *zi = &arg.Imaginary;
//        //double fnu = nOrder;
//        //int kode = 2;
//        //double *cyr = &result.Real;
//        //double *cyi = &result.Imaginary;
//        //int nz = 0;
//        //int ierr = 0;
//
//        //int res = zbesk_(zr, zi,
//        //    /*fnu : initial order, fnu>=0*/ &fnu,
//        //    /*kode: A parameter to indicate the scaling option*/ &kode,
//        //    /*n   : Number of terms in the sequence, n>=1*/ &n,
//        //    /*out CYR : real part of result vector */ cyr,
//        //    /*out CYI : imag part of result vector */ cyi,
//        //    /*out NZ  : Number of underflows set to zero */ &nz,
//        //    /*out IERR: Error flag */ &ierr);
//
//        return result;
//    }
//#pragma endregion
//#pragma endregion

#pragma region BesselI float Array
#pragma region BesselI

template<class ExecutionSpace>
KOKKOS_FUNCTION KokkosViewFloat<ExecutionSpace> BesselI(const int nOrder, const KokkosViewFloat<ExecutionSpace>& arg, const int32 length)
{
    if(nOrder == 0)
    {
        return BesselI0(arg, length);
    }
    if(nOrder == 1)
    {
        return BesselI1(arg, length);
    }

    return BesselIN(nOrder, arg, length);
}

template<class ExecutionSpace>
KOKKOS_FUNCTION KokkosViewFloat<ExecutionSpace> BesselI0(const KokkosViewFloat<ExecutionSpace>& arg, const int32 length)
{
    KokkosViewFloat<ExecutionSpace> array(new float[length], length);

    for(int i = 0; i < length; ++i)
    {
        array[i] = BesselI0(arg[i]);
    }

    return array;
}

template<class ExecutionSpace>
KOKKOS_FUNCTION KokkosViewFloat<ExecutionSpace> BesselI1(const KokkosViewFloat<ExecutionSpace>& arg, const int32 length)
{
    KokkosViewFloat<ExecutionSpace> array(new float[length], length);

    for(int i = 0; i < length; ++i)
    {
        array[i] = BesselI1(arg[i]);
    }

    return array;
}

template<class ExecutionSpace>
KOKKOS_FUNCTION KokkosViewFloat<ExecutionSpace> BesselIN(const int nOrder, const KokkosViewFloat<ExecutionSpace>& arg, const int32 length)
{
    KokkosViewFloat<ExecutionSpace> array(new float[length], length);

    for(int i = 0; i < length; ++i)
    {
        array[i] = BesselIN(nOrder, arg[i]);
    }

    return array;
}
#pragma endregion

#pragma region BesselIE

template<class ExecutionSpace>
KOKKOS_FUNCTION KokkosViewFloat<ExecutionSpace> BesselIE(const int nOrder, const KokkosViewFloat<ExecutionSpace>& arg, const int32 length)
{
    if(nOrder == 0)
    {
        return BesselIE0(arg, length);
    }
    if(nOrder == 1)
    {
        return BesselIE1(arg, length);
    }

    return BesselIEN(nOrder, arg, length);
}

template<class ExecutionSpace>
KOKKOS_FUNCTION KokkosViewFloat<ExecutionSpace> BesselIE0(const KokkosViewFloat<ExecutionSpace>& arg, const int32 length)
{
    KokkosViewFloat<ExecutionSpace> array(new float[length], length);

    for(int i = 0; i < length; ++i)
    {
        array[i] = BesselIE0(arg[i]);
    }

    return array;
}

template<class ExecutionSpace>
KOKKOS_FUNCTION KokkosViewFloat<ExecutionSpace> BesselIE1(const KokkosViewFloat<ExecutionSpace>& arg, const int32 length)
{
    KokkosViewFloat<ExecutionSpace> array(new float[length], length);

    for(int i = 0; i < length; ++i)
    {
        array[i] = BesselIE1(arg[i]);
    }

    return array;
}

template<class ExecutionSpace>
KOKKOS_FUNCTION KokkosViewFloat<ExecutionSpace> BesselIEN(const int nOrder, const KokkosViewFloat<ExecutionSpace>& arg, const int32 length)
{
    KokkosViewFloat<ExecutionSpace> array(new float[length], length);

    for(int i = 0; i < length; ++i)
    {
        array[i] = BesselIEN(nOrder, arg[i]);
    }

    return array;
}
#pragma endregion
#pragma endregion

#pragma region BesselI double Array
#pragma region BesselI

template<class ExecutionSpace>
KOKKOS_FUNCTION KokkosViewDouble<ExecutionSpace> BesselI(const int nOrder, const KokkosViewDouble<ExecutionSpace>& arg, const int32 length)
{
    if(nOrder == 0)
    {
        return BesselI0(arg, length);
    }
    if(nOrder == 1)
    {
        return BesselI1(arg, length);
    }

    return BesselIN(nOrder, arg, length);
}

template<class ExecutionSpace>
KOKKOS_FUNCTION KokkosViewDouble<ExecutionSpace> BesselI0(const KokkosViewDouble<ExecutionSpace>& arg, const int32 length)
{
    KokkosViewDouble<ExecutionSpace> array(new double[length], length);

    for(int i = 0; i < length; ++i)
    {
        array[i] = BesselI0(arg[i]);
    }

    return array;
}

template<class ExecutionSpace>
KOKKOS_FUNCTION KokkosViewDouble<ExecutionSpace> BesselI1(const KokkosViewDouble<ExecutionSpace>& arg, const int32 length)
{
    KokkosViewDouble<ExecutionSpace> array(new double[length], length);

    for(int i = 0; i < length; ++i)
    {
        array[i] = BesselI1(arg[i]);
    }

    return array;
}

template<class ExecutionSpace>
KOKKOS_FUNCTION KokkosViewDouble<ExecutionSpace> BesselIN(const int nOrder, const KokkosViewDouble<ExecutionSpace>& arg, const int32 length)
{
    KokkosViewDouble<ExecutionSpace> array(new double[length], length);

    for(int i = 0; i < length; ++i)
    {
        array[i] = BesselIN(nOrder, arg[i]);
    }

    return array;
}
#pragma endregion

#pragma region BesselIE

template<class ExecutionSpace>
KOKKOS_FUNCTION KokkosViewDouble<ExecutionSpace> BesselIE(const int nOrder, const KokkosViewDouble<ExecutionSpace>& arg, const int32 length)
{
    if(nOrder == 0)
    {
        return BesselIE0(arg, length);
    }
    if(nOrder == 1)
    {
        return BesselIE1(arg, length);
    }

    return BesselIEN(nOrder, arg, length);
}

template<class ExecutionSpace>
KOKKOS_FUNCTION KokkosViewDouble<ExecutionSpace> BesselIE0(const KokkosViewDouble<ExecutionSpace>& arg, const int32 length)
{
    KokkosViewDouble<ExecutionSpace> array(new double[length], length);

    for(int i = 0; i < length; ++i)
    {
        array[i] = BesselIE0(arg[i]);
    }

    return array;
}

template<class ExecutionSpace>
KOKKOS_FUNCTION KokkosViewDouble<ExecutionSpace> BesselIE1(const KokkosViewDouble<ExecutionSpace>& arg, const int32 length)
{
    KokkosViewDouble<ExecutionSpace> array(new double[length], length);

    for(int i = 0; i < length; ++i)
    {
        array[i] = BesselIE1(arg[i]);
    }

    return array;
}

template<class ExecutionSpace>
KOKKOS_FUNCTION KokkosViewDouble<ExecutionSpace> BesselIEN(const int nOrder, const KokkosViewDouble<ExecutionSpace>& arg, const int32 length)
{
    KokkosViewDouble<ExecutionSpace> array(new double[length], length);

    for(int i = 0; i < length; ++i)
    {
        array[i] = BesselIEN(nOrder, arg[i]);
    }

    return array;
}
#pragma endregion
#pragma endregion

//#pragma region BesselI ComplexF Array
//#pragma region BesselI
//
//    cli::array<ComplexF>^ BesselI<cli::array<ComplexF>^>(const int nOrder, cli::array<ComplexF>^ arg)
//    {
//        if (nOrder == 0)
//        {
//            return BesselI0(arg);
//        }
//        else if (nOrder == 1)
//        {
//            return BesselI1(arg);
//        }
//
//        return  BesselIN(nOrder, arg);
//    }
//
//
//    cli::array<ComplexF>^ BesselI0<cli::array<ComplexF>^>(cli::array<ComplexF>^ arg)
//    {
//        const int length = sizeof(arg);
//
//        cli::array<ComplexF>^ array = gcnew cli::array<ComplexF>(length];
//
//        for (int i = 0; i < length; ++i)
//        {
//            array[i] = BesselI0(arg[i]);
//        }
//
//        return array;
//    }
//
//
//    cli::array<ComplexF>^ BesselI1<cli::array<ComplexF>^>(cli::array<ComplexF>^ arg)
//    {
//        const int length = sizeof(arg);
//
//        cli::array<ComplexF>^ array = gcnew cli::array<ComplexF>(length];
//
//        for (int i = 0; i < length; ++i)
//        {
//            array[i] = BesselI1(arg[i]);
//        }
//
//        return array;
//    }
//
//
//    cli::array<ComplexF>^ BesselIN<cli::array<ComplexF>^>(const int nOrder, cli::array<ComplexF>^ arg)
//    {
//        const int length = sizeof(arg);
//
//        cli::array<ComplexF>^ array = gcnew cli::array<ComplexF>(length];
//
//        for (int i = 0; i < length; ++i)
//        {
//            array[i] = BesselIN(nOrder, arg[i]);
//        }
//
//        return array;
//    }
//#pragma endregion
//
//#pragma region BesselIE
//
//    cli::array<ComplexF>^ BesselIE<cli::array<ComplexF>^>(const int nOrder, cli::array<ComplexF>^ arg)
//    {
//        if (nOrder == 0)
//        {
//            return BesselIE0(arg);
//        }
//        else if (nOrder == 1)
//        {
//            return BesselIE1(arg);
//        }
//
//        return  BesselIEN(nOrder, arg);
//    }
//
//
//    cli::array<ComplexF>^ BesselIE0<cli::array<ComplexF>^>(cli::array<ComplexF>^ arg)
//    {
//        const int length = sizeof(arg);
//
//        cli::array<ComplexF>^ array = gcnew cli::array<ComplexF>(length];
//
//        for (int i = 0; i < length; ++i)
//        {
//            array[i] = BesselIE0(arg[i]);
//        }
//
//        return array;
//    }
//
//
//    cli::array<ComplexF>^ BesselIE1<cli::array<ComplexF>^>(cli::array<ComplexF>^ arg)
//    {
//        const int length = sizeof(arg);
//
//        cli::array<ComplexF>^ array = gcnew cli::array<ComplexF>(length];
//
//        for (int i = 0; i < length; ++i)
//        {
//            array[i] = BesselIE1(arg[i]);
//        }
//
//        return array;
//    }
//
//
//    cli::array<ComplexF>^ BesselIEN<cli::array<ComplexF>^>(const int nOrder, cli::array<ComplexF>^ arg)
//    {
//        const int length = sizeof(arg);
//
//        cli::array<ComplexF>^ array = gcnew cli::array<ComplexF>(length];
//
//        for (int i = 0; i < length; ++i)
//        {
//            array[i] = BesselIEN(nOrder, arg[i]);
//        }
//
//        return array;
//    }
//#pragma endregion
//#pragma endregion
//
//#pragma region BesselI ComplexD Array
//#pragma region BesselI
//
//    cli::array<ComplexD>^ BesselI<cli::array<ComplexD>^>(const int nOrder, cli::array<ComplexD>^ arg)
//    {
//        if (nOrder == 0)
//        {
//            return BesselI0(arg);
//        }
//        else if (nOrder == 1)
//        {
//            return BesselI1(arg);
//        }
//
//        return  BesselIN(nOrder, arg);
//    }
//
//
//    cli::array<ComplexD>^ BesselI0<cli::array<ComplexD>^>(cli::array<ComplexD>^ arg)
//    {
//        const int length = sizeof(arg);
//
//        cli::array<ComplexD>^ array = gcnew cli::array<ComplexD>(length];
//
//        for (int i = 0; i < length; ++i)
//        {
//            array[i] = BesselI0(arg[i]);
//        }
//
//        return array;
//    }
//
//
//    cli::array<ComplexD>^ BesselI1<cli::array<ComplexD>^>(cli::array<ComplexD>^ arg)
//    {
//        const int length = sizeof(arg);
//
//        cli::array<ComplexD>^ array = gcnew cli::array<ComplexD>(length];
//
//        for (int i = 0; i < length; ++i)
//        {
//            array[i] = BesselI1(arg[i]);
//        }
//
//        return array;
//    }
//
//
//    cli::array<ComplexD>^ BesselIN<cli::array<ComplexD>^>(const int nOrder, cli::array<ComplexD>^ arg)
//    {
//        const int length = sizeof(arg);
//
//        cli::array<ComplexD>^ array = gcnew cli::array<ComplexD>(length];
//
//        for (int i = 0; i < length; ++i)
//        {
//            array[i] = BesselIN(nOrder, arg[i]);
//        }
//
//        return array;
//    }
//#pragma endregion
//
//#pragma region BesselIE
//
//    cli::array<ComplexD>^ BesselIE<cli::array<ComplexD>^>(const int nOrder, cli::array<ComplexD>^ arg)
//    {
//        if (nOrder == 0)
//        {
//            return BesselIE0(arg);
//        }
//        else if (nOrder == 1)
//        {
//            return BesselIE1(arg);
//        }
//
//        return  BesselIEN(nOrder, arg);
//    }
//
//
//    cli::array<ComplexD>^ BesselIE0<cli::array<ComplexD>^>(cli::array<ComplexD>^ arg)
//    {
//        const int length = sizeof(arg);
//
//        cli::array<ComplexD>^ array = gcnew cli::array<ComplexD>(length];
//
//        for (int i = 0; i < length; ++i)
//        {
//            array[i] = BesselIE0(arg[i]);
//        }
//
//        return array;
//    }
//
//
//    cli::array<ComplexD>^ BesselIE1<cli::array<ComplexD>^>(cli::array<ComplexD>^ arg)
//    {
//        const int length = sizeof(arg);
//
//        cli::array<ComplexD>^ array = gcnew cli::array<ComplexD>(length];
//
//        for (int i = 0; i < length; ++i)
//        {
//            array[i] = BesselIE1(arg[i]);
//        }
//
//        return array;
//    }
//
//
//    cli::array<ComplexD>^ BesselIEN<cli::array<ComplexD>^>(const int nOrder, cli::array<ComplexD>^ arg)
//    {
//        const int length = sizeof(arg);
//
//        cli::array<ComplexD>^ array = gcnew cli::array<ComplexD>(length];
//
//        for (int i = 0; i < length; ++i)
//        {
//            array[i] = BesselIEN(nOrder, arg[i]);
//        }
//
//        return array;
//    }
//#pragma endregion
//#pragma endregion

#pragma region BesselK float Array
#pragma region BesselK

template<class ExecutionSpace>
KOKKOS_FUNCTION KokkosViewFloat<ExecutionSpace> BesselK(const int nOrder, const KokkosViewFloat<ExecutionSpace>& arg, const int32 length)
{
    if(nOrder == 0)
    {
        return BesselK0(arg, length);
    }
    if(nOrder == 1)
    {
        return BesselK1(arg, length);
    }

    return BesselKN(nOrder, arg, length);
}

template<class ExecutionSpace>
KOKKOS_FUNCTION KokkosViewFloat<ExecutionSpace> BesselK0(const KokkosViewFloat<ExecutionSpace>& arg, const int32 length)
{
    KokkosViewFloat<ExecutionSpace> array(new float[length], length);

    for(int i = 0; i < length; ++i)
    {
        array[i] = BesselK0(arg[i]);
    }

    return array;
}

template<class ExecutionSpace>
KOKKOS_FUNCTION KokkosViewFloat<ExecutionSpace> BesselK1(const KokkosViewFloat<ExecutionSpace>& arg, const int32 length)
{
    KokkosViewFloat<ExecutionSpace> array(new float[length], length);

    for(int i = 0; i < length; ++i)
    {
        array[i] = BesselK1(arg[i]);
    }

    return array;
}

template<class ExecutionSpace>
KOKKOS_FUNCTION KokkosViewFloat<ExecutionSpace> BesselKN(const int nOrder, const KokkosViewFloat<ExecutionSpace>& arg, const int32 length)
{
    KokkosViewFloat<ExecutionSpace> array(new float[length], length);

    for(int i = 0; i < length; ++i)
    {
        array[i] = BesselKN(nOrder, arg[i]);
    }

    return array;
}
#pragma endregion

#pragma region BesselKE

template<class ExecutionSpace>
KOKKOS_FUNCTION KokkosViewFloat<ExecutionSpace> BesselKE(const int nOrder, const KokkosViewFloat<ExecutionSpace>& arg, const int32 length)
{
    if(nOrder == 0)
    {
        return BesselKE0(arg, length);
    }
    if(nOrder == 1)
    {
        return BesselKE1(arg, length);
    }

    return BesselKEN(nOrder, arg, length);
}

template<class ExecutionSpace>
KOKKOS_FUNCTION KokkosViewFloat<ExecutionSpace> BesselKE0(const KokkosViewFloat<ExecutionSpace>& arg, const int32 length)
{
    KokkosViewFloat<ExecutionSpace> array(new float[length], length);

    for(int i = 0; i < length; ++i)
    {
        array[i] = BesselKE0(arg[i]);
    }

    return array;
}

template<class ExecutionSpace>
KOKKOS_FUNCTION KokkosViewFloat<ExecutionSpace> BesselKE1(const KokkosViewFloat<ExecutionSpace>& arg, const int32 length)
{
    KokkosViewFloat<ExecutionSpace> array(new float[length], length);

    for(int i = 0; i < length; ++i)
    {
        array[i] = BesselKE1(arg[i]);
    }

    return array;
}

template<class ExecutionSpace>
KOKKOS_FUNCTION KokkosViewFloat<ExecutionSpace> BesselKEN(const int nOrder, const KokkosViewFloat<ExecutionSpace>& arg, const int32 length)
{
    KokkosViewFloat<ExecutionSpace> array(new float[length], length);

    for(int i = 0; i < length; ++i)
    {
        array[i] = BesselKEN(nOrder, arg[i]);
    }

    return array;
}
#pragma endregion
#pragma endregion

#pragma region BesselK double Array
#pragma region BesselK

template<class ExecutionSpace>
KOKKOS_FUNCTION KokkosViewDouble<ExecutionSpace> BesselK(const int nOrder, const KokkosViewDouble<ExecutionSpace>& arg, const int32 length)
{
    if(nOrder == 0)
    {
        return BesselK0(arg, length);
    }
    if(nOrder == 1)
    {
        return BesselK1(arg, length);
    }

    return BesselKN(nOrder, arg, length);
}

template<class ExecutionSpace>
KOKKOS_FUNCTION KokkosViewDouble<ExecutionSpace> BesselK0(const KokkosViewDouble<ExecutionSpace>& arg, const int32 length)
{
    KokkosViewDouble<ExecutionSpace> array(new double[length], length);

    for(int i = 0; i < length; ++i)
    {
        array[i] = BesselK0(arg[i]);
    }

    return array;
}

template<class ExecutionSpace>
KOKKOS_FUNCTION KokkosViewDouble<ExecutionSpace> BesselK1(const KokkosViewDouble<ExecutionSpace>& arg, const int32 length)
{
    KokkosViewDouble<ExecutionSpace> array(new double[length], length);

    for(int i = 0; i < length; ++i)
    {
        array[i] = BesselK1(arg[i]);
    }

    return array;
}

template<class ExecutionSpace>
KOKKOS_FUNCTION KokkosViewDouble<ExecutionSpace> BesselKN(const int nOrder, const KokkosViewDouble<ExecutionSpace>& arg, const int32 length)
{
    KokkosViewDouble<ExecutionSpace> array(new double[length], length);

    for(int i = 0; i < length; ++i)
    {
        array[i] = BesselKN(nOrder, arg[i]);
    }

    return array;
}
#pragma endregion

#pragma region BesselKE

template<class ExecutionSpace>
KOKKOS_FUNCTION KokkosViewDouble<ExecutionSpace> BesselKE(const int nOrder, const KokkosViewDouble<ExecutionSpace>& arg, const int32 length)
{
    if(nOrder == 0)
    {
        return BesselKE0(arg, length);
    }
    if(nOrder == 1)
    {
        return BesselKE1(arg, length);
    }

    return BesselKEN(nOrder, arg, length);
}

template<class ExecutionSpace>
KOKKOS_FUNCTION KokkosViewDouble<ExecutionSpace> BesselKE0(const KokkosViewDouble<ExecutionSpace>& arg, const int32 length)
{
    KokkosViewDouble<ExecutionSpace> array(new double[length], length);

    for(int i = 0; i < length; ++i)
    {
        array[i] = BesselKE0(arg[i]);
    }

    return array;
}

template<class ExecutionSpace>
KOKKOS_FUNCTION KokkosViewDouble<ExecutionSpace> BesselKE1(const KokkosViewDouble<ExecutionSpace>& arg, const int32 length)
{
    KokkosViewDouble<ExecutionSpace> array(new double[length], length);

    for(int i = 0; i < length; ++i)
    {
        array[i] = BesselKE1(arg[i]);
    }

    return array;
}

template<class ExecutionSpace>
KOKKOS_FUNCTION KokkosViewDouble<ExecutionSpace> BesselKEN(const int nOrder, const KokkosViewDouble<ExecutionSpace>& arg, const int32 length)
{
    KokkosViewDouble<ExecutionSpace> array(new double[length], length);

    for(int i = 0; i < length; ++i)
    {
        array[i] = BesselKEN(nOrder, arg[i]);
    }

    return array;
}
#pragma endregion
#pragma endregion

//#pragma region BesselK ComplexF Array
//#pragma region BesselK
//
//    cli::array<ComplexF>^ BesselK<cli::array<ComplexF>^>(const int nOrder, cli::array<ComplexF>^ arg)
//    {
//        if (nOrder == 0)
//        {
//            return BesselK0(arg);
//        }
//        else if (nOrder == 1)
//        {
//            return BesselK1(arg);
//        }
//
//        return  BesselKN(nOrder, arg);
//    }
//
//
//    cli::array<ComplexF>^ BesselK0<cli::array<ComplexF>^>(cli::array<ComplexF>^ arg)
//    {
//        const int length = sizeof(arg);
//
//        cli::array<ComplexF>^ array = gcnew cli::array<ComplexF>(length];
//
//        for (int i = 0; i < length; ++i)
//        {
//            array[i] = BesselK0(arg[i]);
//        }
//
//        return array;
//    }
//
//
//    cli::array<ComplexF>^ BesselK1<cli::array<ComplexF>^>(cli::array<ComplexF>^ arg)
//    {
//        const int length = sizeof(arg);
//
//        cli::array<ComplexF>^ array = gcnew cli::array<ComplexF>(length];
//
//        for (int i = 0; i < length; ++i)
//        {
//            array[i] = BesselK1(arg[i]);
//        }
//
//        return array;
//    }
//
//
//    cli::array<ComplexF>^ BesselKN<cli::array<ComplexF>^>(const int nOrder, cli::array<ComplexF>^ arg)
//    {
//        const int length = sizeof(arg);
//
//        cli::array<ComplexF>^ array = gcnew cli::array<ComplexF>(length];
//
//        for (int i = 0; i < length; ++i)
//        {
//            array[i] = BesselKN(nOrder, arg[i]);
//        }
//
//        return array;
//    }
//#pragma endregion
//
//#pragma region BesselKE
//
//    cli::array<ComplexF>^ BesselKE<cli::array<ComplexF>^>(const int nOrder, cli::array<ComplexF>^ arg)
//    {
//        if (nOrder == 0)
//        {
//            return BesselKE0(arg);
//        }
//        else if (nOrder == 1)
//        {
//            return BesselKE1(arg);
//        }
//
//        return  BesselKEN(nOrder, arg);
//    }
//
//
//    cli::array<ComplexF>^ BesselKE0<cli::array<ComplexF>^>(cli::array<ComplexF>^ arg)
//    {
//        const int length = sizeof(arg);
//
//        cli::array<ComplexF>^ array = gcnew cli::array<ComplexF>(length];
//
//        for (int i = 0; i < length; ++i)
//        {
//            array[i] = BesselKE0(arg[i]);
//        }
//
//        return array;
//    }
//
//
//    cli::array<ComplexF>^ BesselKE1<cli::array<ComplexF>^>(cli::array<ComplexF>^ arg)
//    {
//        const int length = sizeof(arg);
//
//        cli::array<ComplexF>^ array = gcnew cli::array<ComplexF>(length];
//
//        for (int i = 0; i < length; ++i)
//        {
//            array[i] = BesselKE1(arg[i]);
//        }
//
//        return array;
//    }
//
//
//    cli::array<ComplexF>^ BesselKEN<cli::array<ComplexF>^>(const int nOrder, cli::array<ComplexF>^ arg)
//    {
//        const int length = sizeof(arg);
//
//        cli::array<ComplexF>^ array = gcnew cli::array<ComplexF>(length];
//
//        for (int i = 0; i < length; ++i)
//        {
//            array[i] = BesselKEN(nOrder, arg[i]);
//        }
//
//        return array;
//    }
//#pragma endregion
//#pragma endregion
//
//#pragma region BesselK ComplexD Array
//#pragma region BesselK
//
//    cli::array<ComplexD>^ BesselK<cli::array<ComplexD>^>(const int nOrder, cli::array<ComplexD>^ arg)
//    {
//        if (nOrder == 0)
//        {
//            return BesselK0(arg);
//        }
//        else if (nOrder == 1)
//        {
//            return BesselK1(arg);
//        }
//
//        return  BesselKN(nOrder, arg);
//    }
//
//
//    cli::array<ComplexD>^ BesselK0<cli::array<ComplexD>^>(cli::array<ComplexD>^ arg)
//    {
//        const int length = sizeof(arg);
//
//        cli::array<ComplexD>^ array = gcnew cli::array<ComplexD>(length];
//
//        for (int i = 0; i < length; ++i)
//        {
//            array[i] = BesselK0(arg[i]);
//        }
//
//        return array;
//    }
//
//
//    cli::array<ComplexD>^ BesselK1<cli::array<ComplexD>^>(cli::array<ComplexD>^ arg)
//    {
//        const int length = sizeof(arg);
//
//        cli::array<ComplexD>^ array = gcnew cli::array<ComplexD>(length];
//
//        for (int i = 0; i < length; ++i)
//        {
//            array[i] = BesselK1(arg[i]);
//        }
//
//        return array;
//    }
//
//
//    cli::array<ComplexD>^ BesselKN<cli::array<ComplexD>^>(const int nOrder, cli::array<ComplexD>^ arg)
//    {
//        const int length = sizeof(arg);
//
//        cli::array<ComplexD>^ array = gcnew cli::array<ComplexD>(length];
//
//        for (int i = 0; i < length; ++i)
//        {
//            array[i] = BesselKN(nOrder, arg[i]);
//        }
//
//        return array;
//    }
//#pragma endregion
//
//#pragma region BesselKE
//
//    cli::array<ComplexD>^ BesselKE<cli::array<ComplexD>^>(const int nOrder, cli::array<ComplexD>^ arg)
//    {
//        if (nOrder == 0)
//        {
//            return BesselKE0(arg);
//        }
//        else if (nOrder == 1)
//        {
//            return BesselKE1(arg);
//        }
//
//        return  BesselKEN(nOrder, arg);
//    }
//
//
//    cli::array<ComplexD>^ BesselKE0<cli::array<ComplexD>^>(cli::array<ComplexD>^ arg)
//    {
//        const int length = sizeof(arg);
//
//        cli::array<ComplexD>^ array = gcnew cli::array<ComplexD>(length];
//
//        for (int i = 0; i < length; ++i)
//        {
//            array[i] = BesselKE0(arg[i]);
//        }
//
//        return array;
//    }
//
//
//    cli::array<ComplexD>^ BesselKE1<cli::array<ComplexD>^>(cli::array<ComplexD>^ arg)
//    {
//        const int length = sizeof(arg);
//
//        cli::array<ComplexD>^ array = gcnew cli::array<ComplexD>(length];
//
//        for (int i = 0; i < length; ++i)
//        {
//            array[i] = BesselKE1(arg[i]);
//        }
//
//        return array;
//    }
//
//
//    cli::array<ComplexD>^ BesselKEN<cli::array<ComplexD>^>(const int nOrder, cli::array<ComplexD>^ arg)
//    {
//        const int length = sizeof(arg);
//
//        cli::array<ComplexD>^ array = gcnew cli::array<ComplexD>(length];
//
//        for (int i = 0; i < length; ++i)
//        {
//            array[i] = BesselKEN(nOrder, arg[i]);
//        }
//
//        return array;
//    }
//#pragma endregion
//#pragma endregion

typedef float (__host__ *BesselIfloat)(const int, const float);
typedef float (__host__ *BesselI0float)(const float);
typedef float (__host__ *BesselI1float)(const float);
typedef float (__host__ *BesselINfloat)(const int, const float);
typedef double (__host__ *BesselIdouble)(const int, const double);
typedef double (__host__ *BesselI0double)(const double);
typedef double (__host__ *BesselI1double)(const double);
typedef double (__host__ *BesselINdouble)(const int, const double);
typedef float (__host__ *BesselIEfloat)(const int, const float);
typedef float (__host__ *BesselIE0float)(const float);
typedef float (__host__ *BesselIE1float)(const float);
typedef float (__host__ *BesselIENfloat)(const int, const float);
typedef double (__host__ *BesselIEdouble)(const int, const double);
typedef double (__host__ *BesselIE0double)(const double);
typedef double (__host__ *BesselIE1double)(const double);
typedef double (__host__ *BesselIENdouble)(const int, const double);
typedef float (__host__ *BesselKfloat)(const int, const float);
typedef float (__host__ *BesselK0float)(const float);
typedef float (__host__ *BesselK1float)(const float);
typedef float (__host__ *BesselKNfloat)(const int, const float);
typedef double (__host__ *BesselKdouble)(const int, const double);
typedef double (__host__ *BesselK0double)(const double);
typedef double (__host__ *BesselK1double)(const double);
typedef double (__host__ *BesselKNdouble)(const int, const double);
typedef float (__host__ *BesselKEfloat)(const int, const float);
typedef float (__host__ *BesselKE0float)(const float);
typedef float (__host__ *BesselKE1float)(const float);
typedef float (__host__ *BesselKENfloat)(const int, const float);
typedef double (__host__ *BesselKEdouble)(const int, const double);
typedef double (__host__ *BesselKE0double)(const double);
typedef double (__host__ *BesselKE1double)(const double);
typedef double (__host__ *BesselKENdouble)(const int, const double);

#define ARRAY_METHODS(NAME, EXECUTIONSPACE)                                                                                                   \
    typedef KokkosViewFloat<EXECUTIONSPACE> (__host__ *BesselIfloat_ARRAY##NAME)(const int, const KokkosViewFloat<EXECUTIONSPACE>&, const int32);      \
    typedef KokkosViewFloat<EXECUTIONSPACE> (__host__ *BesselI0float_ARRAY##NAME)(const KokkosViewFloat<EXECUTIONSPACE>&, const int32);                \
    typedef KokkosViewFloat<EXECUTIONSPACE> (__host__ *BesselI1float_ARRAY##NAME)(const KokkosViewFloat<EXECUTIONSPACE>&, const int32);                \
    typedef KokkosViewFloat<EXECUTIONSPACE> (__host__ *BesselINfloat_ARRAY##NAME)(const int, const KokkosViewFloat<EXECUTIONSPACE>&, const int32);     \
    typedef KokkosViewDouble<EXECUTIONSPACE> (__host__ *BesselIdouble_ARRAY##NAME)(const int, const KokkosViewDouble<EXECUTIONSPACE>&, const int32);   \
    typedef KokkosViewDouble<EXECUTIONSPACE> (__host__ *BesselI0double_ARRAY##NAME)(const KokkosViewDouble<EXECUTIONSPACE>&, const int32);             \
    typedef KokkosViewDouble<EXECUTIONSPACE> (__host__ *BesselI1double_ARRAY##NAME)(const KokkosViewDouble<EXECUTIONSPACE>&, const int32);             \
    typedef KokkosViewDouble<EXECUTIONSPACE> (__host__ *BesselINdouble_ARRAY##NAME)(const int, const KokkosViewDouble<EXECUTIONSPACE>&, const int32);  \
    typedef KokkosViewFloat<EXECUTIONSPACE> (__host__ *BesselIEfloat_ARRAY##NAME)(const int, const KokkosViewFloat<EXECUTIONSPACE>&, const int32);     \
    typedef KokkosViewFloat<EXECUTIONSPACE> (__host__ *BesselIE0float_ARRAY##NAME)(const KokkosViewFloat<EXECUTIONSPACE>&, const int32);               \
    typedef KokkosViewFloat<EXECUTIONSPACE> (__host__ *BesselIE1float_ARRAY##NAME)(const KokkosViewFloat<EXECUTIONSPACE>&, const int32);               \
    typedef KokkosViewFloat<EXECUTIONSPACE> (__host__ *BesselIENfloat_ARRAY##NAME)(const int, const KokkosViewFloat<EXECUTIONSPACE>&, const int32);    \
    typedef KokkosViewDouble<EXECUTIONSPACE> (__host__ *BesselIEdouble_ARRAY##NAME)(const int, const KokkosViewDouble<EXECUTIONSPACE>&, const int32);  \
    typedef KokkosViewDouble<EXECUTIONSPACE> (__host__ *BesselIE0double_ARRAY##NAME)(const KokkosViewDouble<EXECUTIONSPACE>&, const int32);            \
    typedef KokkosViewDouble<EXECUTIONSPACE> (__host__ *BesselIE1double_ARRAY##NAME)(const KokkosViewDouble<EXECUTIONSPACE>&, const int32);            \
    typedef KokkosViewDouble<EXECUTIONSPACE> (__host__ *BesselIENdouble_ARRAY##NAME)(const int, const KokkosViewDouble<EXECUTIONSPACE>&, const int32); \
    typedef KokkosViewFloat<EXECUTIONSPACE> (__host__ *BesselKfloat_ARRAY##NAME)(const int, const KokkosViewFloat<EXECUTIONSPACE>&, const int32);      \
    typedef KokkosViewFloat<EXECUTIONSPACE> (__host__ *BesselK0float_ARRAY##NAME)(const KokkosViewFloat<EXECUTIONSPACE>&, const int32);                \
    typedef KokkosViewFloat<EXECUTIONSPACE> (__host__ *BesselK1float_ARRAY##NAME)(const KokkosViewFloat<EXECUTIONSPACE>&, const int32);                \
    typedef KokkosViewFloat<EXECUTIONSPACE> (__host__ *BesselKNfloat_ARRAY##NAME)(const int, const KokkosViewFloat<EXECUTIONSPACE>&, const int32);     \
    typedef KokkosViewDouble<EXECUTIONSPACE> (__host__ *BesselKdouble_ARRAY##NAME)(const int, const KokkosViewDouble<EXECUTIONSPACE>&, const int32);   \
    typedef KokkosViewDouble<EXECUTIONSPACE> (__host__ *BesselK0double_ARRAY##NAME)(const KokkosViewDouble<EXECUTIONSPACE>&, const int32);             \
    typedef KokkosViewDouble<EXECUTIONSPACE> (__host__ *BesselK1double_ARRAY##NAME)(const KokkosViewDouble<EXECUTIONSPACE>&, const int32);             \
    typedef KokkosViewDouble<EXECUTIONSPACE> (__host__ *BesselKNdouble_ARRAY##NAME)(const int, const KokkosViewDouble<EXECUTIONSPACE>&, const int32);  \
    typedef KokkosViewFloat<EXECUTIONSPACE> (__host__ *BesselKEfloat_ARRAY##NAME)(const int, const KokkosViewFloat<EXECUTIONSPACE>&, const int32);     \
    typedef KokkosViewFloat<EXECUTIONSPACE> (__host__ *BesselKE0float_ARRAY##NAME)(const KokkosViewFloat<EXECUTIONSPACE>&, const int32);               \
    typedef KokkosViewFloat<EXECUTIONSPACE> (__host__ *BesselKE1float_ARRAY##NAME)(const KokkosViewFloat<EXECUTIONSPACE>&, const int32);               \
    typedef KokkosViewFloat<EXECUTIONSPACE> (__host__ *BesselKENfloat_ARRAY##NAME)(const int, const KokkosViewFloat<EXECUTIONSPACE>&, const int32);    \
    typedef KokkosViewDouble<EXECUTIONSPACE> (__host__ *BesselKEdouble_ARRAY##NAME)(const int, const KokkosViewDouble<EXECUTIONSPACE>&, const int32);  \
    typedef KokkosViewDouble<EXECUTIONSPACE> (__host__ *BesselKE0double_ARRAY##NAME)(const KokkosViewDouble<EXECUTIONSPACE>&, const int32);            \
    typedef KokkosViewDouble<EXECUTIONSPACE> (__host__ *BesselKE1double_ARRAY##NAME)(const KokkosViewDouble<EXECUTIONSPACE>&, const int32);            \
    typedef KokkosViewDouble<EXECUTIONSPACE> (__host__ *BesselKENdouble_ARRAY##NAME)(const int, const KokkosViewDouble<EXECUTIONSPACE>&, const int32);

ARRAY_METHODS(Serial, Kokkos::Serial)

ARRAY_METHODS(OpenMP, Kokkos::OpenMP)

ARRAY_METHODS(Cuda, Kokkos::Cuda)

#undef ARRAY_METHODS

// clang-format off
static constexpr BesselMethods bessel_methods_api_version_1 = 
{
    static_cast<BesselIfloat>(&BesselI),
    static_cast<BesselI0float>(&BesselI0),
    static_cast<BesselI1float>(&BesselI1),
    static_cast<BesselINfloat>(&BesselIN),
    static_cast<BesselIdouble>(&BesselI),
    static_cast<BesselI0double>(&BesselI0),
    static_cast<BesselI1double>(&BesselI1),
    static_cast<BesselINdouble>(&BesselIN),
    static_cast<BesselIEfloat>(&BesselIE),
    static_cast<BesselIE0float>(&BesselIE0),
    static_cast<BesselIE1float>(&BesselIE1),
    static_cast<BesselIENfloat>(&BesselIEN),
    static_cast<BesselIEdouble>(&BesselIE),
    static_cast<BesselIE0double>(&BesselIE0),
    static_cast<BesselIE1double>(&BesselIE1),
    static_cast<BesselIENdouble>(&BesselIEN),
    static_cast<BesselKfloat>(&BesselK),
    static_cast<BesselK0float>(&BesselK0),
    static_cast<BesselK1float>(&BesselK1),
    static_cast<BesselKNfloat>(&BesselKN),
    static_cast<BesselKdouble>(&BesselK),
    static_cast<BesselK0double>(&BesselK0),
    static_cast<BesselK1double>(&BesselK1),
    static_cast<BesselKNdouble>(&BesselKN),
    static_cast<BesselKEfloat>(&BesselKE),
    static_cast<BesselKE0float>(&BesselKE0),
    static_cast<BesselKE1float>(&BesselKE1),
    static_cast<BesselKENfloat>(&BesselKEN),
    static_cast<BesselKEdouble>(&BesselKE),
    static_cast<BesselKE0double>(&BesselKE0),
    static_cast<BesselKE1double>(&BesselKE1),
    static_cast<BesselKENdouble>(&BesselKEN),

    #define ARRAY_METHODS(NAME, EXECUTIONSPACE) \
    static_cast<BesselIfloat_ARRAY##NAME>(&BesselI),\
    static_cast<BesselI0float_ARRAY##NAME>(&BesselI0),\
    static_cast<BesselI1float_ARRAY##NAME>(&BesselI1),\
    static_cast<BesselINfloat_ARRAY##NAME>(&BesselIN),\
    static_cast<BesselIdouble_ARRAY##NAME>(&BesselI),\
    static_cast<BesselI0double_ARRAY##NAME>(&BesselI0),\
    static_cast<BesselI1double_ARRAY##NAME>(&BesselI1),\
    static_cast<BesselINdouble_ARRAY##NAME>(&BesselIN),\
    static_cast<BesselIEfloat_ARRAY##NAME>(&BesselIE),\
    static_cast<BesselIE0float_ARRAY##NAME>(&BesselIE0),\
    static_cast<BesselIE1float_ARRAY##NAME>(&BesselIE1),\
    static_cast<BesselIENfloat_ARRAY##NAME>(&BesselIEN),\
    static_cast<BesselIEdouble_ARRAY##NAME>(&BesselIE),\
    static_cast<BesselIE0double_ARRAY##NAME>(&BesselIE0),\
    static_cast<BesselIE1double_ARRAY##NAME>(&BesselIE1),\
    static_cast<BesselIENdouble_ARRAY##NAME>(&BesselIEN),\
    static_cast<BesselKfloat_ARRAY##NAME>(&BesselK),\
    static_cast<BesselK0float_ARRAY##NAME>(&BesselK0),\
    static_cast<BesselK1float_ARRAY##NAME>(&BesselK1),\
    static_cast<BesselKNfloat_ARRAY##NAME>(&BesselKN),\
    static_cast<BesselKdouble_ARRAY##NAME>(&BesselK),\
    static_cast<BesselK0double_ARRAY##NAME>(&BesselK0),\
    static_cast<BesselK1double_ARRAY##NAME>(&BesselK1),\
    static_cast<BesselKNdouble_ARRAY##NAME>(&BesselKN),\
    static_cast<BesselKEfloat_ARRAY##NAME>(&BesselKE),\
    static_cast<BesselKE0float_ARRAY##NAME>(&BesselKE0),\
    static_cast<BesselKE1float_ARRAY##NAME>(&BesselKE1),\
    static_cast<BesselKENfloat_ARRAY##NAME>(&BesselKEN),\
    static_cast<BesselKEdouble_ARRAY##NAME>(&BesselKE),\
    static_cast<BesselKE0double_ARRAY##NAME>(&BesselKE0),\
    static_cast<BesselKE1double_ARRAY##NAME>(&BesselKE1),\
    static_cast<BesselKENdouble_ARRAY##NAME>(&BesselKEN),\

    ARRAY_METHODS(Serial, Kokkos::Serial)

    ARRAY_METHODS(OpenMP, Kokkos::OpenMP)

    ARRAY_METHODS(Cuda, Kokkos::Cuda)

    #undef ARRAY_METHODS


};
// clang-format on

KOKKOS_NET_API_EXTERNC const BesselMethods* GetBesselMethodsApi(const uint32& version)
{
    if(version == 1)
    {
        return &bessel_methods_api_version_1;
    }
    return nullptr;
}
