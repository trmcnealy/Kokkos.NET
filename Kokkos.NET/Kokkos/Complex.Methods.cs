using System;
using System.Collections.Generic;
using System.Text;

namespace Kokkos
{
    public sealed unsafe partial class Complex<T>
        where T : unmanaged
    {
        public static Complex<T> Zero = new Complex<T>(UnManaged.Math<T>.Default(0.0), UnManaged.Math<T>.Default(0.0));

        public static Complex<T> One = new Complex<T>(UnManaged.Math<T>.Default(1.0), UnManaged.Math<T>.Default(0.0));
        public static Complex<T> i   = new Complex<T>(UnManaged.Math<T>.Default(0.0), UnManaged.Math<T>.Default(1.0));
        //public static Complex<T> NaN      = new Complex<T>(T.NaN,                T.NaN);
        //public static Complex<T> Infinity = new Complex<T>(T.PositiveInfinity,   T.PositiveInfinity);

        public Complex(T x, T y)
        {
            _real      = x;
            _imaginary = y;
        }

        public Complex(T x)
        {
            _real      = x;
            _imaginary = default;
        }


        public static implicit operator T(Complex<T> z)
        {
            return z._real;
        }

        public static explicit operator Complex<T>(T x)
        {
            return new Complex<T>(x);
        }
        
        public override int GetHashCode()
        {
            var hash = new HashCode();
            hash.Add(_real);
            hash.Add(_imaginary);
            return hash.ToHashCode();
        }

        //public bool IsCZero
        //{ get { return ((_real == 0.0) && (_imaginary == 0.0)); } }

        //public bool IsCOne
        //{ get { return ((_real == 1.0) && (_imaginary == 0.0)); } }

        //public bool Isi
        //{ get { return ((_real == 0.0) && (_imaginary == 1.0)); } }

        //public bool IsCNaN
        //{ get { return T.IsNaN(_real) || T.IsNaN(_imaginary); } }

        //public bool IsInfinity
        //{ get { return T.IsInfinity(_real) || T.IsInfinity(_imaginary); } }

        //public bool IsReal
        //{ get { return (_imaginary == 0.0); } }

        //public bool IsImag
        //{ get { return (_real == 0.0); } }

        //////////////////////////////////////////
        //////////////Complex Methods ////////////
        //////////////////////////////////////////
        //// Returns the argument of a complex number.
        //public static T Arg(Complex<T> z)
        //{
        //    return (Math.Atan2(z._imaginary, z._real));
        //}

        //// Returns the conjugate of a complex number z.
        //public static Complex<T> Conj(Complex<T> z)
        //{
        //    return (new Complex<T>(z._real, -z._imaginary));
        //}

        //// Returns the norm (or modulus or absolute value)
        //// of a complex number.
        //public static T Norm(Complex<T> z)
        //{
        //    return (Math.Sqrt((z._real * z._real) + (z._imaginary * z._imaginary)));
        //}

        //// Returns the norm (or modulus) of a complex number
        //// avoiding potential overflow and/or underflow for
        //// very small or very large numbers.
        //public static T CNorm2(Complex<T> z)
        //{
        //    T x = z._real;
        //    T y = z._imaginary;

        //    if(Math.Abs(x) < Math.Abs(y))
        //    {
        //        return (Math.Abs(y) * Math.Sqrt(1.0 + (x / y) * (x / y)));
        //    }
        //    else
        //    {
        //        return (Math.Abs(x) * Math.Sqrt(1.0 + (y / x) * (y / x)));
        //    }
        //}

        //// Returns the inverse of a complex number.
        //public static Complex<T> Inv(Complex<T> z)
        //{
        //    return Math<T>.Div(Math<T>.Default(1.0), z);
        //}

        // Returns the _real part of a complex number.
        public static T Re(Complex<T> z)
        {
            return (z._real);
        }

        // Returns the imaginary part of a complex number.
        public static T Im(Complex<T> z)
        {
            return (z._imaginary);
        }

        // Converts:(r,theta) ----> (x,y)
        //public static Complex<T> FromPolarToXY(T r, T theta)
        //{
        //    return (new Complex<T>((r * Math.Cos(theta)), (r * Math.Sin(theta))));
        //}

        //// Converts:    (x,y) ----> (r,theta)
        //public static Complex<T> FromXYToPolar(Complex<T> z)
        //{
        //    return (new Complex<T>(CNorm(z), CArg(z)));
        //}

        //// Returns the negation of a complex number.
        //public static Complex<T> Neg(Complex<T> z)
        //{
        //    return Math<T>.Negate(z);
        //}

        //// Returns the sum of two complex numbers z1 and z2.
        //public static Complex<T> Add(Complex<T> z1, Complex<T> z2)
        //{
        //    return (z1 + z2);
        //}

        //// Returns the sum of a _real number with a complex number.
        //public static Complex<T> Add(T x, Complex<T> z)
        //{
        //    return (x + z);
        //}

        //// Returns the sum of a complex number with a _real number.
        //public static Complex<T> Add(Complex<T> z, T x)
        //{
        //    return (z + x);
        //}

        //// Returns the difference between two complex numbers z1 and z2.
        //public static Complex<T> Sub(Complex<T> z1, Complex<T> z2)
        //{
        //    return (z1 - z2);
        //}

        //// Returns the difference between a _real number and a complex number.
        //public static Complex<T> Sub(T x, Complex<T> z)
        //{
        //    return (x - z);
        //}

        //// Returns the difference between a complex number and a _real number.
        //public static Complex<T> Sub(Complex<T> z, T x)
        //{
        //    return (z - x);
        //}

        //// Returns the product between two complex numbers z1 and z2.
        //public static Complex<T> Mult(Complex<T> z1, Complex<T> z2)
        //{
        //    return (z1 * z2);
        //}

        //// Returns the product of a _real number with a complex number.
        //public static Complex<T> Mult(T x, Complex<T> z)
        //{
        //    return (x * z);
        //}

        //// Returns the product of a complex number and a _real number.
        //public static Complex<T> Mult(Complex<T> z, T x)
        //{
        //    return (z * x);
        //}

        //// Returns the quotient of dividing two complex numbers z1 and z2.
        //public static Complex<T> Div(Complex<T> z1, Complex<T> z2)
        //{
        //    return (z1 / z2);
        //}

        //// Returns the quotient of dividing a _real number by a complex number.
        //public static Complex<T> Div(T x, Complex<T> z)
        //{
        //    return (x / z);
        //}

        //// Returns the quotient of dividing a complex number by a _real number.
        //public static Complex<T> Div(Complex<T> z, T x)
        //{
        //    return (z / x);
        //}

        //// Returns the quotient of dividing two complex
        //// numbers z1 and z2 avoiding potential underflow
        //// and/or overflow for very small or very large numbers
        //public static Complex<T> Div2(Complex<T> z1, Complex<T> z2)
        //{
        //    T x1 = z1._real;
        //    T y1 = z1._imaginary;
        //    T x2 = z2._real;
        //    T y2 = z2._imaginary;

        //    Complex<T> u;
        //    T denom;

        //    if(z2.IsCZero)
        //        return Complex<T>.Infinity;

        //    if(Math.Abs(x2) < Math.Abs(y2))
        //    {
        //        denom = x2 * (x2 / y2) + y2;
        //        u._real = (x1 * (x2 / y2) + y1) / denom;
        //        u._imaginary = (y1 * (x2 / y2) - x1) / denom;
        //    }
        //    else
        //    {
        //        denom = x2 + y2 * (y2 / x2);
        //        u._real = (x1 + y1 * (y2 / x2)) / denom;
        //        u._imaginary = (y1 - x1 * (y2 / x2)) / denom;
        //    }
        //    return u;
        //}

        //////////////////////////////////////////
        ///////////Overloaded Operators //////////
        //////////////////////////////////////////
        //public static Complex<T> operator +(Complex<T> z)
        //{
        //    return z;
        //}

        //public static Complex<T> operator +(Complex<T> z1, Complex<T> z2)
        //{
        //    return (new Complex<T>(z1._real + z2._real, z1._imaginary + z2._imaginary));
        //}

        //// Returns the sum of a _real number with a complex number.
        //public static Complex<T> operator +(T x, Complex<T> z)
        //{
        //    return (new Complex<T>(x + z._real, z._imaginary));
        //}

        //// Returns the sum of a complex number with a _real number.
        //public static Complex<T> operator +(Complex<T> z, T x)
        //{
        //    return (new Complex<T>(z._real + x, z._imaginary));
        //}

        //// Returns the negation of a complex number.
        //public static Complex<T> operator -(Complex<T> z)
        //{
        //    return (new Complex<T>(-z._real, -z._imaginary));
        //}

        //// Returns the difference between two complex numbers.
        //public static Complex<T> operator -(Complex<T> z1, Complex<T> z2)
        //{
        //    return (new Complex<T>(z1._real - z2._real, z1._imaginary - z2._imaginary));
        //}

        //// Returns the difference of a _real number with a complex number.
        //public static Complex<T> operator -(T x, Complex<T> z)
        //{
        //    return (new Complex<T>(x - z._real, -z._imaginary));
        //}

        //// Returns the difference of a complex number with a _real number.
        //public static Complex<T> operator -(Complex<T> z, T x)
        //{
        //    return (new Complex<T>(z._real - x, z._imaginary));
        //}

        //// Returns the product of two complex numbers z1 * z2.
        //public static Complex<T> operator *(Complex<T> z1, Complex<T> z2)
        //{
        //    T x = (z1._real * z2._real) - (z1._imaginary * z2._imaginary);
        //    T y = (z1._real * z2._imaginary) + (z1._imaginary * z2._real);
        //    return (new Complex<T>(x, y));
        //}

        //// Returns the product of a _real and a complex number.
        //public static Complex<T> operator *(T x, Complex<T> z)
        //{
        //    return (new Complex<T>(x * z._real, x * z._imaginary));
        //}

        //// Returns the product of a complex number and a _real.
        //public static Complex<T> operator *(Complex<T> z, T x)
        //{
        //    return (new Complex<T>(z._real * x, z._imaginary * x));
        //}

        //// Returns the quotient of two complex numbers z1 / z2.
        //public static Complex<T> operator /(Complex<T> z1, Complex<T> z2)
        //{
        //    if(z2.IsCZero)
        //        return Complex<T>.Infinity;
        //    T denom = Math.Pow(z2._real, 2.0) + Math.Pow(z2._imaginary, 2.0);
        //    T x = ((z1._real * z2._real) + (z1._imaginary * z2._imaginary)) / denom;
        //    T y = ((z1._imaginary * z2._real) - (z1._real * z2._imaginary)) / denom;
        //    return (new Complex<T>(x, y));
        //}

        //// Returns the quotient of dividing a _real number by a complex number.
        //public static Complex<T> operator /(T x, Complex<T> z)
        //{
        //    if(z.IsCZero)
        //        return Complex<T>.Infinity;
        //    T denom = Math.Pow(z._real, 2.0) + Math.Pow(z._imaginary, 2.0);
        //    T re = (x * z._real) / denom;
        //    T im = (0.0 - (x * z._imaginary)) / denom;
        //    return (new Complex<T>(re, im));
        //}

        //// Returns the quotient of dividing a complex number by a _real number.
        //public static Complex<T> operator /(Complex<T> z, T x)
        //{
        //    if(x == 0.0)
        //        return Complex<T>.Infinity;
        //    T re = z._real / x;
        //    T im = z._imaginary / x;
        //    return (new Complex<T>(re, im));
        //}

        //// Tests for equality of two complex numbers z1 and z2.
        //public static bool operator ==(Complex<T> z1, Complex<T> z2)
        //{
        //    return ((z1._real == z2._real) && (z1._imaginary == z2._imaginary));
        //}

        //// Tests for inequality of two complex numbers z1 and z2.
        //public static bool operator !=(Complex<T> z1, Complex<T> z2)
        //{
        //    return (!(z1 == z2));
        //}

        //// Tests for equality of this complex number and another complex number.
        //public override bool Equals(Object obj)
        //{
        //    return ((obj is Complex<T>) && (this == (Complex<T>)obj));
        //}

        //// Returns an integer hash code for this complex number.
        //// If you override Equals, override GetHashCode too.
        //public override int GetHashCode()
        //{
        //    //return this.ToString().GetHashCode();
        //    return (_real.GetHashCode() ^ _imaginary.GetHashCode());
        //}

        //// Returns a formatted string representation in
        //// the form z = x + iy for a complex number.
        //public override string ToString()
        //{
        //    return (String.Format("{0} + {1}i", _real, _imaginary));
        //}

        ///////////////////////////////////////////
        ////////////Exponential Functions /////////
        ///////////////////////////////////////////
        //public static Complex<T> Exp(Complex<T> z)
        //{
        //    T x = z.Real;
        //    T y = z._imaginary;
        //    T expx = Math.Exp(x);
        //    return (new Complex<T>(expx * Math.Cos(y),
        //                        expx * Math.Sin(y)));
        //}

        ///////////////////////////////////////////
        ////////////Logarithmic Functions /////////
        ///////////////////////////////////////////
        //// Logarithm of complex z to base e
        //public static Complex<T> Log(Complex<T> z)
        //{
        //    return (new Complex<T>(Math.Log(Complex<T>.CNorm(z)),
        //                        Math.Atan2(z._imaginary, z.Real)));
        //}

        //// Another version of logarithm of complex z to base e
        //public static Complex<T> Log2(Complex<T> z)
        //{
        //    T x = z.Real;
        //    T y = z._imaginary;
        //    T re = 0.5 * Math.Log(x * x + y * y);
        //    T im = Math.Atan2(y, x);
        //    return (new Complex<T>(re, im));
        //}

        //// Logarithm of complex z to base 10
        //public static Complex<T> Log10(Complex<T> z)
        //{
        //    return (Complex<T>.Log(z) / Complex<T>.Log((Complex<T>)10.0));
        //}

        //// Logarithm of complex z1 to complex base z2
        //public static Complex<T> Logb(Complex<T> z1, Complex<T> z2)
        //{
        //    return (Complex<T>.Log(z1) / Complex<T>.Log(z2));
        //}

        //// Logarithm of _real x to complex base z2
        //public static Complex<T> Logb(T x, Complex<T> z2)
        //{
        //    return (Complex<T>.Log((Complex<T>)x) / Complex<T>.Log(z2));
        //}

        //// Logarithm of complex z1 to _real base x
        //public static Complex<T> Logb(Complex<T> z1, T x)
        //{
        //    return (Complex<T>.Log(z1) / Complex<T>.Log((Complex<T>)x));
        //}

        ///////////////////////////////////////////
        //////////////Power Functions /////////////
        ///////////////////////////////////////////
        //// Complex<T> z raised to the power of complex w
        //public static Complex<T> Pow(Complex<T> z, Complex<T> w)
        //{
        //    return (Complex<T>.Exp(w * Complex<T>.Log(z)));
        //}

        //// Complex<T> z raised to the power of complex w (ver 2)
        //public static Complex<T> Pow2(Complex<T> z, Complex<T> w)
        //{
        //    T x1 = z.Real;
        //    T y1 = z._imaginary;
        //    T x2 = w.Real;
        //    T y2 = w._imaginary;

        //    T r1 = Math.Sqrt(x1 * x1 + y1 * y1);
        //    T theta1 = Math.Atan2(y1, x1);
        //    T phi = theta1 * x2 + y2 * Math.Log(r1);

        //    T re = Math.Pow(r1, x2) * Math.Exp(-theta1 * y2) * Math.Cos(phi);
        //    T im = Math.Pow(r1, x2) * Math.Exp(-theta1 * y2) * Math.Sin(phi);

        //    return (new Complex<T>(re, im));
        //}

        //// Complex<T> z raised to the power of _real x
        //public static Complex<T> Pow(Complex<T> z, T x)
        //{
        //    return (Complex<T>.Exp(x * Complex<T>.Log(z)));
        //}

        //// Complex<T> z raised to the power of _real x (ver 2)
        //public static Complex<T> Pow2(Complex<T> z, T x)
        //{
        //    T x1 = z.Real;
        //    T y1 = z._imaginary;

        //    T r1 = Math.Sqrt(x1 * x1 + y1 * y1);
        //    T theta1 = Math.Atan2(y1, x1);
        //    T phi = theta1 * x;

        //    T re = Math.Pow(r1, x) * Math.Cos(phi);
        //    T im = Math.Pow(r1, x) * Math.Sin(phi);

        //    return (new Complex<T>(re, im));
        //}

        //// Real x raised to the power of complex z
        //public static Complex<T> Pow(T x, Complex<T> z)
        //{
        //    return (Complex<T>.Exp(z * Math.Log(x)));
        //}

        //// Real x raised to the power of complex z (ver 2)
        //public static Complex<T> Pow2(T x, Complex<T> w)
        //{
        //    T x2 = w.Real;
        //    T y2 = w._imaginary;

        //    T r1 = Math.Sqrt(x * x);
        //    T theta1 = Math.Atan2(0.0, x);
        //    T phi = theta1 * x2 + y2 * Math.Log(r1);

        //    T re = Math.Pow(r1, x2) * Math.Cos(phi);
        //    T im = Math.Pow(r1, x2) * Math.Sin(phi);

        //    return (new Complex<T>(re, im));
        //}

        //////////////////////////////////////////
        //////////////Root Functions /////////////
        //////////////////////////////////////////
        //// Complex<T> root w of complex number z
        //public static Complex<T> Root(Complex<T> z, Complex<T> w)
        //{
        //    return (Complex<T>.Exp(Complex<T>.Log(z) / w));
        //}

        //// Real root x of complex number z
        //public static Complex<T> Root(Complex<T> z, T x)
        //{
        //    return (Complex<T>.Exp(Complex<T>.Log(z) / x));
        //}

        //// Complex<T> root z of _real number x
        //public static Complex<T> Root(T x, Complex<T> z)
        //{
        //    return (Complex<T>.Exp(Math.Log(x) / z));
        //}

        //// Complex<T> square root of complex number z
        //public static Complex<T> Sqrt(Complex<T> z)
        //{
        //    return (Complex<T>.Exp(Complex<T>.Log(z) / 2.0));
        //}

        ///////////////////////////////////////////
        //////Complex<T> Trigonometric Functions /////
        ///////////////////////////////////////////
        //// Complex<T> sine of complex number z
        //public static Complex<T> Sin(Complex<T> z)
        //{
        //    return ((Complex<T>.Exp(i * z) - Complex<T>.Exp(-i * z)) / (2.0 * i));
        //}

        //// Complex<T> sine of complex number z (ver 2)
        //public static Complex<T> Sin2(Complex<T> z)
        //{
        //    T x = z.Real;
        //    T y = z._imaginary;
        //    T re = Math.Sin(x) * Math.Cosh(y);
        //    T im = Math.Cos(x) * Math.Sinh(y);
        //    return (new Complex<T>(re, im));
        //}

        //// Complex<T> cosine of complex number z
        //public static Complex<T> Cos(Complex<T> z)
        //{
        //    return ((Complex<T>.Exp(i * z) + Complex<T>.Exp(-i * z)) / 2.0);
        //}

        //// Complex<T> cosine of complex number z (ver 2)
        //public static Complex<T> Cos2(Complex<T> z)
        //{
        //    T x = z.Real;
        //    T y = z._imaginary;
        //    T re = Math.Cos(x) * Math.Cosh(y);
        //    T im = -Math.Sin(x) * Math.Sinh(y);
        //    return (new Complex<T>(re, im));
        //}

        //// Complex<T> tangent of complex number z
        //public static Complex<T> Tan(Complex<T> z)
        //{
        //    return (Complex<T>.Sin(z) / Complex<T>.Cos(z));
        //}

        //// Complex<T> tangent of complex number z (ver 2)
        //public static Complex<T> Tan2(Complex<T> z)
        //{
        //    T x2 = 2.0 * z.Real;
        //    T y2 = 2.0 * z._imaginary;
        //    T denom = Math.Cos(x2) + Math.Cosh(y2);
        //    if(denom == 0.0)
        //        return Complex<T>.Infinity;
        //    T re = Math.Sin(x2) / denom;
        //    T im = Math.Sinh(y2) / denom;
        //    return (new Complex<T>(re, im));
        //}

        //// Complex<T> cotangent of complex number z
        //public static Complex<T> Cot(Complex<T> z)
        //{
        //    return (Complex<T>.Cos(z) / Complex<T>.Sin(z));
        //}

        //// Complex<T> cotangent of complex number z (ver 2)
        //public static Complex<T> Cot2(Complex<T> z)
        //{
        //    T x2 = 2.0 * z.Real;
        //    T y2 = 2.0 * z._imaginary;
        //    T denom = Math.Cosh(y2) - Math.Cos(x2);
        //    if(denom == 0.0)
        //        return Complex<T>.Infinity;
        //    T re = Math.Sin(x2) / denom;
        //    T im = -Math.Sinh(y2) / denom;
        //    return (new Complex<T>(re, im));
        //}

        //// Complex<T> secant of complex number z
        //public static Complex<T> Sec(Complex<T> z)
        //{
        //    return (1.0 / Complex<T>.Cos(z));
        //}

        //// Complex<T> secant of complex number z (ver 2)
        //public static Complex<T> Sec2(Complex<T> z)
        //{
        //    T x = z.Real;
        //    T y = z._imaginary;
        //    T denom = Math.Cos(x) * Math.Cos(x) + Math.Sinh(y) * Math.Sinh(y);
        //    if(denom == 0.0)
        //        return Complex<T>.Infinity;
        //    T re = Math.Cos(x) * Math.Cosh(y) / denom;
        //    T im = Math.Sin(x) * Math.Sinh(y) / denom;
        //    return (new Complex<T>(re, im));
        //}

        //// Complex<T> cosecant of complex number z
        //public static Complex<T> Csc(Complex<T> z)
        //{
        //    return (1.0 / Complex<T>.Sin(z));
        //}

        //// Complex<T> cosecant of complex number z (ver 2)
        //public static Complex<T> Csc2(Complex<T> z)
        //{
        //    T x = z.Real;
        //    T y = z._imaginary;
        //    T denom = Math.Sin(x) * Math.Sin(x) +
        //                   Math.Sinh(y) * Math.Sinh(y);
        //    if(denom == 0.0)
        //        return Complex<T>.Infinity;
        //    T re = Math.Sin(x) * Math.Cosh(y) / denom;
        //    T im = -Math.Cos(x) * Math.Sinh(y) / denom;
        //    return (new Complex<T>(re, im));
        //}

        /////////////////////////////////////////////////
        /////Complex<T> Inverse Trigonometric Functions ////
        /////////////////////////////////////////////////
        //// Complex<T> ArcSine of complex number z
        //public static Complex<T> ArcSin(Complex<T> z)
        //{
        //    return (-i * Complex<T>.Log((i * z) + Complex<T>.Sqrt(1.0 - (z * z))));
        //}

        //// Complex<T> ArcSine of complex number z (ver 2)
        //public static Complex<T> ArcSin2(Complex<T> z)
        //{
        //    T x = z.Real;
        //    T y = z._imaginary;

        //    T ysqd = y * y;
        //    T rtpos = Math.Sqrt(Math.Pow(x + 1.0, 2.0) + ysqd);
        //    T rtneg = Math.Sqrt(Math.Pow(x - 1.0, 2.0) + ysqd);
        //    T alpha = 0.5 * (rtpos + rtneg);
        //    T beta = 0.5 * (rtpos - rtneg);

        //    T InvSinZRe = Math.Asin(beta);
        //    T InvSinZIm = Math.Sign(y) * Math.Log(alpha + Math.Sqrt(alpha * alpha - 1.0));

        //    return (new Complex<T>(InvSinZRe, InvSinZIm));
        //}

        //// Complex<T> ArcCosine of complex number z
        //public static Complex<T> ArcCos(Complex<T> z)
        //{
        //    return (-i * Complex<T>.Log(z + i * Complex<T>.Sqrt(1.0 - (z * z))));
        //}

        //// Complex<T> ArcCosine of complex number z (ver 2)
        //public static Complex<T> ArcCos2(Complex<T> z)
        //{
        //    T x = z.Real;
        //    T y = z._imaginary;

        //    T ysqd = y * y;
        //    T rtpos = Math.Sqrt(Math.Pow(x + 1.0, 2.0) + ysqd);
        //    T rtneg = Math.Sqrt(Math.Pow(x - 1.0, 2.0) + ysqd);
        //    T alpha = 0.5 * (rtpos + rtneg);
        //    T beta = 0.5 * (rtpos - rtneg);

        //    T InvCosZRe = Math.Acos(beta);
        //    T InvCosZIm = -Math.Sign(y) * Math.Log(alpha + Math.Sqrt(alpha * alpha - 1.0));

        //    return (new Complex<T>(InvCosZRe, InvCosZIm));
        //}

        //// Complex<T> ArcTangent of complex number z
        //public static Complex<T> ArcTan(Complex<T> z)
        //{
        //    return ((i / 2.0) * Complex<T>.Log((i + z) / (i - z)));
        //}

        //// Complex<T> ArcTangent of complex number z (ver 2)
        //public static Complex<T> ArcTan2(Complex<T> z)
        //{
        //    T x = z.Real;
        //    T y = z._imaginary;

        //    T xsqd = x * x;
        //    T ysqd = y * y;

        //    T InvTanZRe = 0.5 * Math.Atan2(2.0 * x, 1.0 - xsqd - ysqd);
        //    T InvTanZIm = 0.25 * Math.Log((xsqd + Math.Pow(y + 1.0, 2.0)) / (xsqd + Math.Pow(y - 1.0, 2.0)));

        //    return (new Complex<T>(InvTanZRe, InvTanZIm));
        //}

        //// Complex<T> ArcCotangent of complex number z
        //public static Complex<T> ArcCot(Complex<T> z)
        //{
        //    return (Complex<T>.ArcTan(1.0 / z));
        //}

        //// Complex<T> ArcCotangent of complex number z (ver 2)
        //public static Complex<T> ArcCot2(Complex<T> z)
        //{
        //    return ((i / 2.0) * (Complex<T>.Log((z - i) / (z + i))));
        //}

        //// Complex<T> ArcSecant of complex number z
        //public static Complex<T> ArcSec(Complex<T> z)
        //{
        //    return (Complex<T>.ArcCos(1.0 / z));
        //}

        //// Complex<T> ArcSecant of complex number z (ver 2)
        //public static Complex<T> ArcSec2(Complex<T> z)
        //{
        //    return (i * Complex<T>.Log((Complex<T>.Sqrt(1.0 - (z * z)) + 1.0) / z));
        //}

        //// Complex<T> ArcCosecant of complex number z
        //public static Complex<T> ArcCsc(Complex<T> z)
        //{
        //    return (Complex<T>.ArcSin(1.0 / z));
        //}

        //// Complex<T> ArcCosecant of complex number z (ver 2)
        //public static Complex<T> ArcCsc2(Complex<T> z)
        //{
        //    return (-i * Complex<T>.Log((Complex<T>.Sqrt((z * z) - 1.0) + i) / z));
        //}

        ////////////////////////////////////////////
        ////////Complex<T> Hyperbolic Functions ///////
        ////////////////////////////////////////////
        //// Complex<T> hyperbolic sine of complex number z
        //public static Complex<T> Sinh(Complex<T> z)
        //{
        //    return ((Complex<T>.Exp(z) - Complex<T>.Exp(-z)) / 2.0);
        //}

        //// Complex<T> Hyperbolic Sine
        //// of complex number z (ver 2)
        //public static Complex<T> Sinh2(Complex<T> z)
        //{
        //    T x = z.Real;
        //    T y = z._imaginary;
        //    T SinhZRe = Math.Sinh(x) * Math.Cos(y);
        //    T SinhZIm = Math.Cosh(x) * Math.Sin(y);
        //    return (new Complex<T>(SinhZRe, SinhZIm));
        //}

        //// Complex<T> Hyperbolic Sine of complex number z (ver 3)
        //public static Complex<T> Sinh3(Complex<T> z)
        //{
        //    return (-i * Complex<T>.Sin(i * z));
        //}

        //// Complex<T> Hyperbolic Cosine of complex number z
        //public static Complex<T> Cosh(Complex<T> z)
        //{
        //    return ((Complex<T>.Exp(z) + Complex<T>.Exp(-z)) / 2.0);
        //}

        //// Complex<T> Hyperbolic Cosine of complex number z (ver 2)
        //public static Complex<T> Cosh2(Complex<T> z)
        //{
        //    T x = z.Real;
        //    T y = z._imaginary;
        //    T CoshZRe = Math.Cosh(x) * Math.Cos(y);
        //    T CoshZIm = Math.Sinh(x) * Math.Sin(y);
        //    return (new Complex<T>(CoshZRe, CoshZIm));
        //}

        //// Complex<T> Hyperbolic Cosine of complex number z (ver 3)
        //public static Complex<T> Cosh3(Complex<T> z)
        //{
        //    return (Complex<T>.Cos(i * z));
        //}

        //// Complex<T> Hyperbolic Tangent of complex number z
        //public static Complex<T> Tanh(Complex<T> z)
        //{
        //    return (Complex<T>.Sinh(z) / Complex<T>.Cosh(z));
        //}

        //// Complex<T> Hyperbolic Tangent of complex number z (ver 2)
        //public static Complex<T> Tanh2(Complex<T> z)
        //{
        //    T twox = 2.0 * z.Real;
        //    T twoy = 2.0 * z._imaginary;
        //    T denom = Math.Cosh(twox) + Math.Cos(twoy);

        //    T TanhZRe = Math.Sinh(twox) / denom;
        //    T TanhZIm = Math.Sin(twoy) / denom;
        //    return (new Complex<T>(TanhZRe, TanhZIm));
        //}

        //// Complex<T> Hyperbolic Tangent of complex number z (ver 3)
        //public static Complex<T> Tanh3(Complex<T> z)
        //{
        //    return (-i * Complex<T>.Tan(i * z));
        //}

        //// Complex<T> Hyperbolic Cotangent of complex number z
        //public static Complex<T> Coth(Complex<T> z)
        //{
        //    return (Complex<T>.Cosh(z) / Complex<T>.Sinh(z));
        //}

        //// Complex<T> Hyperbolic Cotangent of complex number z (ver 2)
        //public static Complex<T> Coth2(Complex<T> z)
        //{
        //    return (Complex<T>.Cosh2(z) / Complex<T>.Sinh2(z));
        //}

        //// Complex<T> Hyperbolic Cotangent of complex number z (ver 3)
        //public static Complex<T> Coth3(Complex<T> z)
        //{
        //    T twox = 2.0 * z.Real;
        //    T twoy = 2.0 * z._imaginary;
        //    T denom = Math.Cosh(twox) - Math.Cos(twoy);

        //    T CothZRe = Math.Sinh(twox) / denom;
        //    T CothZIm = -Math.Sin(twoy) / denom;
        //    return (new Complex<T>(CothZRe, CothZIm));
        //}

        //// Complex<T> Hyperbolic Cotangent of complex number z (ver 4)
        //public static Complex<T> Coth4(Complex<T> z)
        //{
        //    return (i * Complex<T>.Cot(i * z));
        //}

        //// Complex<T> Hyperbolic Secant of complex number z
        //public static Complex<T> Sech(Complex<T> z)
        //{
        //    return (1.0 / Complex<T>.Cosh(z));
        //}

        //// Complex<T> Hyperbolic Secant of complex number z (ver 2)
        //public static Complex<T> Sech2(Complex<T> z)
        //{
        //    return (1.0 / Complex<T>.Cosh2(z));
        //}

        //// Complex<T> Hyperbolic Secant of complex number z (ver 3)
        //public static Complex<T> Sech3(Complex<T> z)
        //{
        //    T CoshX = Math.Cosh(z.Real);
        //    T CosY = Math.Cos(z._imaginary);
        //    T SinhX = Math.Sinh(z.Real);
        //    T SinY = Math.Sin(z._imaginary);

        //    T denom = CosY * CosY + SinhX * SinhX;

        //    T CSechZRe = (CoshX * CosY) / denom;
        //    T CSechZIm = -(SinhX * SinY) / denom;
        //    return (new Complex<T>(CSechZRe, CSechZIm));
        //}

        //// Complex<T> Hyperbolic Secant of complex number z (ver 4)
        //public static Complex<T> Sech4(Complex<T> z)
        //{
        //    return (Complex<T>.Sec(i * z));
        //}

        //// Complex<T> Hyperbolic Cosecant of complex number z
        //public static Complex<T> Csch(Complex<T> z)
        //{
        //    return (1.0 / Complex<T>.Sinh(z));
        //}

        //// Complex<T> Hyperbolic Cosecant of complex number z (ver 2)
        //public static Complex<T> Csch2(Complex<T> z)
        //{
        //    return (1.0 / Complex<T>.Sinh2(z));
        //}

        //// Complex<T> Hyperbolic Cosecant of complex number z (ver 3)
        //public static Complex<T> Csch3(Complex<T> z)
        //{
        //    T CoshX = Math.Cosh(z.Real);
        //    T CosY = Math.Cos(z._imaginary);
        //    T SinhX = Math.Sinh(z.Real);
        //    T SinY = Math.Sin(z._imaginary);

        //    T denom = SinY * SinY + SinhX * SinhX;

        //    T CSechZRe = (SinhX * CosY) / denom;
        //    T CSechZIm = -(CoshX * SinY) / denom;
        //    return (new Complex<T>(CSechZRe, CSechZIm));
        //}

        //// Complex<T> Hyperbolic Cosecant of complex number z (ver 4)
        //public static Complex<T> Csch4(Complex<T> z)
        //{
        //    return (i * Complex<T>.Csc(i * z));
        //}

        ///////////////////////////////////////////////
        //////Complex<T> Inverse Hyperbolic Functions ////
        ///////////////////////////////////////////////
        //// Complex<T> Inverse Hyperbolic Sine of complex number z
        //public static Complex<T> ArcSinh(Complex<T> z)
        //{
        //    return (Complex<T>.Log(z + Complex<T>.Sqrt((z * z) + 1.0)));
        //}

        //// Complex<T> Inverse Hyperbolic Sine of complex number z (ver 2)
        //public static Complex<T> ArcSinh2(Complex<T> z)
        //{
        //    T x = z.Real;
        //    T y = z._imaginary;

        //    T xsqd = x * x;
        //    T rtpos = Math.Sqrt(Math.Pow(y - 1.0, 2.0) + xsqd);
        //    T rtneg = Math.Sqrt(Math.Pow(y + 1.0, 2.0) + xsqd);
        //    T alphap = 0.5 * (rtpos + rtneg);
        //    T betap = 0.5 * (rtpos - rtneg);

        //    T InvSinhZRe = Math.Sign(x) *
        //        Math.Log(alphap + Math.Sqrt(alphap * alphap - 1));
        //    T InvSinhZIm = -Math.Asin(betap);

        //    return (new Complex<T>(InvSinhZRe, InvSinhZIm));
        //}

        //// Complex<T> Inverse Hyperbolic Cosine of complex number z
        //public static Complex<T> ArcCosh(Complex<T> z)
        //{
        //    return (Complex<T>.Log(z + Complex<T>.Sqrt(z * z - 1.0)));
        //}

        //// Complex<T> Inverse Hyperbolic Cosine of complex number z (ver 2)
        //public static Complex<T> ArcCosh2(Complex<T> z)
        //{
        //    T x = z.Real;
        //    T y = z._imaginary;

        //    T ysqd = y * y;
        //    T rtpos = Math.Sqrt(Math.Pow(x + 1.0, 2.0) + ysqd);
        //    T rtneg = Math.Sqrt(Math.Pow(x - 1.0, 2.0) + ysqd);
        //    T alpha = 0.5 * (rtpos + rtneg);
        //    T beta = 0.5 * (rtpos - rtneg);

        //    T InvCoshZRe =
        //        Math.Log(alpha + Math.Sqrt(alpha * alpha - 1));
        //    T InvCoshZIm = Math.Sign(y) * Math.Acos(beta);

        //    return (new Complex<T>(InvCoshZRe, InvCoshZIm));
        //}

        //// Complex<T> Inverse Hyperbolic Tangent of complex number z
        //public static Complex<T> ArcTanh(Complex<T> z)
        //{
        //    return (0.5 * Complex<T>.Log((1.0 + z) / (1.0 - z)));
        //}

        //// Complex<T> Inverse Hyperbolic Tangent of complex number z (ver 2)
        //public static Complex<T> ArcTanh2(Complex<T> z)
        //{
        //    T x = z.Real;
        //    T y = z._imaginary;
        //    T xsqd = x * x;
        //    T ysqd = y * y;

        //    T InvTanhZRe = 0.25 * Math.Log((ysqd +
        //        Math.Pow(x + 1.0, 2.0)) / (ysqd + Math.Pow(x - 1.0, 2.0)));

        //    T InvTanhZIm = 0.5 * Math.Atan2(2.0 * y, 1.0 - xsqd - ysqd);

        //    return (new Complex<T>(InvTanhZRe, InvTanhZIm));
        //}

        //// Complex<T> Inverse Hyperbolic Cotangent of complex number z
        //public static Complex<T> ArcCoth(Complex<T> z)
        //{
        //    return (Complex<T>.ArcTanh(1.0 / z));
        //}

        //// Complex<T> Inverse Hyperbolic Cotangent of complex number z (ver 2)
        //public static Complex<T> ArcCoth2(Complex<T> z)
        //{
        //    return (0.5 * Complex<T>.Log((z + 1.0) / (z - 1.0)));
        //}

        //// Complex<T> Inverse Hyperbolic Secant of complex number z
        //public static Complex<T> ArcSech(Complex<T> z)
        //{
        //    return (Complex<T>.ArcCosh(1.0 / z));
        //}

        //// Complex<T> Inverse Hyperbolic Secant of complex number z (ver 2)
        //public static Complex<T> ArcSech2(Complex<T> z)
        //{
        //    return (Complex<T>.Log((1.0 + Complex<T>.Sqrt(1.0 - (z * z))) / z));
        //}

        //// Complex<T> Inverse Hyperbolic Cosecant of complex number z
        //public static Complex<T> ArcCsch(Complex<T> z)
        //{
        //    return (Complex<T>.ArcSinh(1.0 / z));
        //}

        //// Complex<T> Inverse Hyperbolic Cosecant of complex number z (ver 2)
        //public static Complex<T> ArcCsch2(Complex<T> z)
        //{
        //    return (Complex<T>.Log((1.0 + Complex<T>.Sqrt(1.0 + (z * z))) / z));
        //}
    
    }
}
