
using System;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace Kokkos
{
    [StructLayout(System.Runtime.InteropServices.LayoutKind.Sequential)]
    public sealed unsafe class Pair<T1, T2> : IDisposable
        where T1 : unmanaged
        where T2 : unmanaged
    {
        public static readonly int ThisSize;
        
        private static readonly int _firstOffset;
        private static readonly int _secondOffset;
        
        static Pair()
        {
            _firstOffset  = Marshal.OffsetOf<Pair<T1, T2>>(nameof(_first)).ToInt32();
            _secondOffset = Marshal.OffsetOf<Pair<T1, T2>>(nameof(_second)).ToInt32();
            ThisSize      = _secondOffset + Unsafe.SizeOf<T2>();
        }
        
        private T1 _first;
        private T2 _second;
        
        private readonly NativePointer pointer;
        
        public T1 First
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            get { return *(T1*)(pointer.Data + _firstOffset); }
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            set { *(T1*)(pointer.Data + _firstOffset) = value; }
        }
        
        public T2 Second
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            get { return *(T2*)(pointer.Data + _secondOffset); }
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            set { *(T2*)(pointer.Data + _secondOffset) = value; }
        }
        
        public NativePointer Instance
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            get { return pointer; }
        }
        
        public Pair(ExecutionSpaceKind executionSpace = ExecutionSpaceKind.Cuda)
        {
            pointer = NativePointer.Allocate(ThisSize, executionSpace);
        }
        
        internal Pair(nint intPtr, ExecutionSpaceKind executionSpace = ExecutionSpaceKind.Cuda)
        {
            pointer = new NativePointer(intPtr, ThisSize, false, executionSpace);
        }
        
        internal Pair(Pair<T1, T2> copy, ExecutionSpaceKind executionSpace = ExecutionSpaceKind.Cuda)
        {
            pointer = new NativePointer(copy.Instance, executionSpace);
        }
        
        public void Dispose()
        {
            pointer.Dispose();
        }
        
        public static implicit operator Pair<T1, T2>(nint intPtr)
        {
            return new (intPtr);
        }
    }
}
