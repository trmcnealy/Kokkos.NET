﻿
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
        
        private static readonly int _realOffset;
        private static readonly int _imaginaryOffset;
        
        static Complex()
        {
            _realOffset  = Marshal.OffsetOf<Complex<T>>(nameof(_real)).ToInt32();
            _imaginaryOffset  = Marshal.OffsetOf<Complex<T>>(nameof(_imaginary)).ToInt32();
            ThisSize = _imaginaryOffset + Unsafe.SizeOf<T>();
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
        
        ~Complex()
        {
        }
        public void Dispose()
        {
            pointer.Dispose();
            GC.SuppressFinalize(this);
        }
        
        internal Complex(IntPtr intPtr, ExecutionSpaceKind executionSpace = ExecutionSpaceKind.Cuda)
        {
            pointer = new NativePointer(intPtr, ThisSize, false, executionSpace);
        }
        
        public static implicit operator Complex<T>(IntPtr intPtr)
        {
            return new Complex<T>(intPtr);
        }
    }
}