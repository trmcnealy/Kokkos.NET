using System;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace Kokkos
{
    [StructLayout(System.Runtime.InteropServices.LayoutKind.Sequential)]
    public sealed unsafe partial class Complex<T> : IDisposable
        where T : unmanaged
    {
        private static readonly Type _T = typeof(T);
        public static readonly int ThisSize;
        
        private static readonly nint _realOffset;
        private static readonly nint _imaginaryOffset;
        
        static Complex()
        {
            _realOffset  = (nint)Marshal.OffsetOf<Complex<T>>(nameof(_real)).ToInt32();
            _imaginaryOffset  = (nint)Marshal.OffsetOf<Complex<T>>(nameof(_imaginary)).ToInt32();
            ThisSize = (int)_imaginaryOffset + Unsafe.SizeOf<T>();
        }
        
        private T _real;
        private T _imaginary;
        
        private readonly NativePointer pointer;
        
        public T Real
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            get { return *(T*)(pointer.Data + _realOffset); }
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            set { *(T*)(pointer.Data + _realOffset) = value; }
        }
        
        public T Imaginary
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            get { return *(T*)(pointer.Data + _imaginaryOffset); }
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            set { *(T*)(pointer.Data + _imaginaryOffset) = value; }
        }
        
        public NativePointer Instance
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            get { return pointer; }
        }
        
        public Complex(ExecutionSpaceKind executionSpace = ExecutionSpaceKind.Cuda)
        {
            pointer = NativePointer.Allocate(ThisSize, executionSpace);
        }
        
        internal Complex(IntPtr intPtr, ExecutionSpaceKind executionSpace = ExecutionSpaceKind.Cuda)
        {
            pointer = new NativePointer(intPtr, ThisSize, false, executionSpace);
        }
        
        internal Complex(Complex<T> copy, ExecutionSpaceKind executionSpace = ExecutionSpaceKind.Cuda)
        {
            pointer = new NativePointer(copy.Instance, executionSpace);
        }
        
        public void Dispose()
        {
            pointer.Dispose();
        }
        
        public static implicit operator Complex<T>(IntPtr intPtr)
        {
            return new Complex<T>(intPtr);
        }
    }
}


