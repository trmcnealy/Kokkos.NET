using System;
using System.Diagnostics;
using System.IO;
using System.Reflection;
using System.Runtime.CompilerServices;
using System.Security;
using System.Text;

namespace Kokkos.Tests
{
    public static class Assert
    {
        public static void IsTrue(bool                      condition,
                                  [CallerMemberName] string? caller   = null,
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
                                       [CallerMemberName] string? caller   = null,
                                       [CallerFilePath]   string __FILE__ = "",
                                       [CallerLineNumber] int    __LINE__ = 0) where  T:notnull
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
                                    string? caller,
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
                StackFrame? sf;

                int frameIndex = -1;

                do
                {
                    sf = st.GetFrame(++frameIndex);
                } while(sf is not null && sf.GetMethod()!.Name.StartsWith("AreEqual") && frameIndex < st.FrameCount - 1);

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

            for(int i = 0; i < stackTrace.FrameCount; ++i)
            {
                StackFrame? frame  = stackTrace.GetFrame(i);
                MethodBase? method = frame?.GetMethod();

                if(method == null)
                {
                    continue;
                }

                if(b.Length > 0)
                {
                    b.AppendLine();
                }

                b.Append("   at ");
                Type? declaringType = method.DeclaringType;

                if(declaringType != null)
                {
                    b.Append(declaringType.FullName?.Replace('+',
                                                             '.'));

                    b.Append('.');
                }

                b.Append(method.Name);

                // output type parameters, if any
                if(method is MethodInfo && ((MethodInfo)method).IsGenericMethod)
                {
                    Type[] genericArguments = ((MethodInfo)method).GetGenericArguments();
                    b.Append('[');

                    for(int j = 0; j < genericArguments.Length; ++j)
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

                for(int j = 0; j < parameters.Length; ++j)
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
                if(frame is not null && frame.GetILOffset() >= 0)
                {
                    string? filename = null;

                    try
                    {
                        string? fullpath = frame.GetFileName();

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
}
