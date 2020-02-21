using System;
using System.Diagnostics;
using System.IO;
using System.Reflection;
using System.Runtime.CompilerServices;
using System.Security;
using System.Text;
using System.Transactions;

[System.Runtime.Versioning.NonVersionable]
public static class ConsoleMethods
{
    public static void WriteLine(this TextWriter textWriter,
                                 ConsoleColor    textColor,
                                 string          msg)
    {
        Console.ForegroundColor = textColor;
        textWriter.WriteLine(msg);
        Console.ResetColor();
    }

    public static void WriteLine(this TextWriter textWriter,
                                 ConsoleColor    textColor,
                                 string          format,
                                 object          arg0)
    {
        Console.ForegroundColor = textColor;

        textWriter.WriteLine(string.Format(format,
                                           arg0));

        Console.ResetColor();
    }

    public static void WriteLine<T>(this TextWriter textWriter,
                                    ConsoleColor    textColor,
                                    string          format,
                                    params T[]      arg)
    {
        Console.ForegroundColor = textColor;

        textWriter.WriteLine(string.Format(format,
                                           arg));

        Console.ResetColor();
    }

    public static void WriteLine(this TextWriter textWriter,
                                 ConsoleColor    textColor,
                                 string          description,
                                 string          title)
    {
        Console.ForegroundColor = textColor;

        textWriter.WriteLine(description,
                             title,
                             null,
                             null);

        Console.ResetColor();
    }

    public static void WriteLine(this TextWriter textWriter,
                                 ConsoleColor    textColor,
                                 string          description,
                                 string          file,
                                 int             line,
                                 int             column)
    {
        Console.ForegroundColor = textColor;

        textWriter.WriteLine(description,
                             file,
                             line,
                             column);

        Console.ResetColor();
    }
}

namespace Kokkos.Tests
{
    public sealed class AssertException : Exception
    {
        public override string StackTrace { get; }

        public AssertException(string     message,
                               StackTrace st)
            : base(message)
        {
            StackTrace = st.ToString();
        }
    }

    public static class Assert
    {
        public static void IsTrue(bool                      condition,
                                  [CallerMemberName] string caller   = null,
                                  [CallerFilePath]   string __FILE__ = "",
                                  [CallerLineNumber] int    __LINE__ = 0)
        {
            if(AreEqual(condition,
                        caller,
                        __FILE__,
                        __LINE__))
            {
                Console.Out.WriteLine(ConsoleColor.Green,
                                      $"{caller} is true.");
            }
        }

        public static void AreEqual<T>(T                         expected,
                                       T                         actual,
                                       [CallerMemberName] string caller   = null,
                                       [CallerFilePath]   string __FILE__ = "",
                                       [CallerLineNumber] int    __LINE__ = 0)
        {
            if(AreEqual(expected.Equals(actual),
                        caller,
                        __FILE__,
                        __LINE__))
            {
                Console.Out.WriteLine(ConsoleColor.Green,
                                      $"expected: {expected} == actual: {actual}");
            }
        }

        public static bool AreEqual(bool   condition,
                                    string caller,
                                    string __FILE__,
                                    int    __LINE__)
        {
            if(!condition)
            {
                string errorMessage = string.Empty;

                if(!string.IsNullOrEmpty(caller))
                {
                    errorMessage += caller;
                }

                if(!string.IsNullOrEmpty(__FILE__))
                {
                    errorMessage += " [" + __FILE__ + ":" + __LINE__ + "]";
                }

                StackTrace st = new StackTrace();
                StackFrame sf;

                int frameIndex = -1;

                do
                {
                    sf = st.GetFrame(++frameIndex);
                } while(sf.GetMethod().Name.StartsWith("AreEqual") && frameIndex < st.FrameCount - 1);

                st = new StackTrace(frameIndex);

                try
                {
                    throw new AssertException(errorMessage,
                                              st);
                }
                catch(Exception ex)
                {
                    Console.Out.WriteLine(ConsoleColor.Red,
                                          ex.Message);
                }

                return false;
            }

            return true;
        }

        private static string GetStackTrace(Exception exception)
        {
            // Output stacktrace in custom format (very similar to Exception.StackTrace property on English systems).
            // Include filenames where available, but no paths.
            StackTrace stackTrace = new StackTrace(exception,
                                                   true);

            StringBuilder b = new StringBuilder();

            for(int i = 0; i < stackTrace.FrameCount; i++)
            {
                StackFrame frame  = stackTrace.GetFrame(i);
                MethodBase method = frame.GetMethod();

                if(method == null)
                {
                    continue;
                }

                if(b.Length > 0)
                {
                    b.AppendLine();
                }

                b.Append("   at ");
                Type declaringType = method.DeclaringType;

                if(declaringType != null)
                {
                    b.Append(declaringType.FullName.Replace('+',
                                                            '.'));

                    b.Append('.');
                }

                b.Append(method.Name);

                // output type parameters, if any
                if(method is MethodInfo && ((MethodInfo)method).IsGenericMethod)
                {
                    Type[] genericArguments = ((MethodInfo)method).GetGenericArguments();
                    b.Append('[');

                    for(int j = 0; j < genericArguments.Length; j++)
                    {
                        if(j > 0)
                        {
                            b.Append(',');
                        }

                        b.Append(genericArguments[j].Name);
                    }

                    b.Append(']');
                }

                // output parameters, if any
                b.Append('(');
                ParameterInfo[] parameters = method.GetParameters();

                for(int j = 0; j < parameters.Length; j++)
                {
                    if(j > 0)
                    {
                        b.Append(", ");
                    }

                    if(parameters[j].ParameterType != null)
                    {
                        b.Append(parameters[j].ParameterType.Name);
                    }
                    else
                    {
                        b.Append('?');
                    }

                    if(!string.IsNullOrEmpty(parameters[j].Name))
                    {
                        b.Append(' ');
                        b.Append(parameters[j].Name);
                    }
                }

                b.Append(')');

                // source location
                if(frame.GetILOffset() >= 0)
                {
                    string filename = null;

                    try
                    {
                        string fullpath = frame.GetFileName();

                        if(fullpath != null)
                        {
                            filename = Path.GetFileName(fullpath);
                        }
                    }
                    catch(SecurityException)
                    {
                        // StackFrame.GetFileName requires PathDiscovery permission
                    }
                    catch(ArgumentException)
                    {
                        // Path.GetFileName might throw on paths with invalid chars
                    }

                    b.Append(" in ");

                    if(filename != null)
                    {
                        b.Append(filename);
                        b.Append(":line ");
                        b.Append(frame.GetFileLineNumber());
                    }
                    else
                    {
                        b.Append("offset ");
                        b.Append(frame.GetILOffset());
                    }
                }
            }

            return b.ToString();
        }
    }

    public class ViewTests
    {
        public void Run()
        {
            CreateScalar();
            CreateVector();
            CreateMatrix();
            CreateTensor();

            CopyToVector();

            GetSetValueVector();
        }

        #region Create Tests

        public void CreateScalar()
        {
            View<double, Cuda> view = new View<double, Cuda>("CreateScalar");

            Assert.AreEqual(view.Label(),
                            "CreateScalar");

            Assert.AreEqual(view.Size(),
                            1ul);

            Assert.AreEqual(view.Extent(0),
                            1ul);
        }

        public void CreateVector()
        {
            View<double, Cuda> view = new View<double, Cuda>("CreateVector",
                                                             10);

            Assert.AreEqual(view.Label(),
                            "CreateVector");

            Assert.AreEqual(view.Size(),
                            10ul);

            Assert.AreEqual(view.Extent(0),
                            10ul);
        }

        public void CreateMatrix()
        {
            View<double, Cuda> view = new View<double, Cuda>("CreateMatrix",
                                                             10,
                                                             10);

            Assert.AreEqual(view.Label(),
                            "CreateMatrix");

            Assert.AreEqual(view.Size(),
                            100ul);

            Assert.AreEqual(view.Extent(0),
                            10ul);

            Assert.AreEqual(view.Extent(1),
                            10ul);
        }

        public void CreateTensor()
        {
            View<double, Cuda> view = new View<double, Cuda>("CreateTensor",
                                                             10,
                                                             10,
                                                             10);

            Assert.AreEqual(view.Label(),
                            "CreateTensor");

            Assert.AreEqual(view.Size(),
                            1000ul);

            Assert.AreEqual(view.Extent(0),
                            10ul);

            Assert.AreEqual(view.Extent(1),
                            10ul);

            Assert.AreEqual(view.Extent(2),
                            10ul);
        }

        #endregion

        public void CopyToVector()
        {
            View<double, Cuda> view = new View<double, Cuda>("CreateVector",
                                                             3);

            double[] values = new double[]
            {
                351684.65, 651681.26, 987612.20
            };

            view.CopyTo(values);

            Assert.IsTrue(Math.Abs(view[0] - values[0]) < double.Epsilon);

            Assert.IsTrue(Math.Abs(view[1] - values[1]) < double.Epsilon);

            Assert.IsTrue(Math.Abs(view[2] - values[2]) < double.Epsilon);
        }

        public void GetSetValueVector()
        {
            View<double, Cuda> view = new View<double, Cuda>("CreateVector",
                                                             2);

            view[0] = 1321.258;
            view[1] = 123123.12;

            double value0 = view[0];
            double value1 = view[1];

            Assert.IsTrue(Math.Abs(value0 - 1321.258) < double.Epsilon);

            Assert.IsTrue(Math.Abs(value1 - 123123.12) < double.Epsilon);
        }
    }
}