#nullable enable
using System.ComponentModel;
using System.Diagnostics;
using System.Runtime.CompilerServices;
using System.Runtime.Versioning;

namespace System
{
    [DebuggerDisplay("{ToString(),raw}")]
    [NonVersionable]
    public readonly ref struct MemoryMappedPtr<T>
    {
        private static readonly ulong SizeOfT = (ulong)Unsafe.SizeOf<T>();

        internal class RawArrayData
        {
#pragma warning disable CS0649, CA1823, 169
            public ulong Length; // Array._numComponents padded to IntPtr
#if TARGET_64BIT
            public ulong Padding;
#endif
            public byte Data;
#pragma warning restore CS0649, CA1823, 169
        }

        [Intrinsic]
        [NonVersionable]
        public static ref T1 GetArrayDataReference<T1>(T1[] array)
        {
            return ref Unsafe.As<byte, T1>(ref Unsafe.As<RawArrayData>(array).Data);
        }

        /// <summary>A byref or a native ptr.</summary>
        internal readonly ByReference<T> _pointer;

        /// <summary>The number of elements this MemoryMappedPtr contains.</summary>
        private readonly ulong _length;

        /// <summary>Creates a new read-only span over the entirety of the target array.</summary>
        /// <param name="array">The target array.</param>
        /// <remarks>Returns default when <paramref name="array" /> is null.</remarks>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public MemoryMappedPtr(T[]? array)
        {
            if(array == null)
            {
                this = default;

                return; // returns default
            }

            _pointer = new ByReference<T>(ref GetArrayDataReference(array));
            _length  = (ulong)array.LongLength;
        }

        /// <summary>
        ///     Creates a new read-only span over the portion of the target array beginning at 'start' index and ending at
        ///     'end' index (exclusive).
        /// </summary>
        /// <param name="array">The target array.</param>
        /// <param name="start">The index at which to begin the read-only span.</param>
        /// <param name="length">The number of items in the read-only span.</param>
        /// <remarks>Returns default when <paramref name="array" /> is null.</remarks>
        /// <exception cref="System.ArgumentOutOfRangeException">
        ///     Thrown when the specified <paramref name="start" /> or end index
        ///     is not in the range (&lt;0 or &gt;Length).
        /// </exception>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public unsafe MemoryMappedPtr(T[]?  array,
                                      ulong start,
                                      ulong length)
        {
            if(array == null)
            {
                if(start != 0 || length != 0)
                {
                    throw new ArgumentOutOfRangeException();
                }

                this = default;

                return; // returns default
            }
#if TARGET_64BIT
            // See comment in Span<T>.Slice for how this works.
            if(start + length > (ulong)array.LongLength)
            {
                throw new ArgumentOutOfRangeException();
            }
#else
        if ((ulong)start > (ulong)array.LongLength || (ulong)length > (ulong)(array.LongLength - start))
            throw new ArgumentOutOfRangeException();
#endif
            //(void*)(((ulong)_pointer._value) + (start * SizeOfT))
            _pointer = new ByReference<T>(ref Unsafe.AsRef<T>((void*)((ulong)Unsafe.AsPointer(ref GetArrayDataReference(array)) + (start * SizeOfT))));
            _length  = length;
        }

        /// <summary>
        ///     Creates a new read-only span over the target unmanaged buffer.  Clearly this is quite dangerous, because we
        ///     are creating arbitrarily typed T's out of a void*-typed block of memory.  And the length is not checked. But if
        ///     this creation is correct, then all subsequent uses are correct.
        /// </summary>
        /// <param name="pointer">An unmanaged pointer to memory.</param>
        /// <param name="length">The number of <typeparamref name="T" /> elements the memory contains.</param>
        /// <exception cref="System.ArgumentException">
        ///     Thrown when <typeparamref name="T" /> is reference type or contains pointers
        ///     and hence cannot be stored in unmanaged memory.
        /// </exception>
        /// <exception cref="System.ArgumentOutOfRangeException">Thrown when the specified <paramref name="length" /> is negative.</exception>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public unsafe MemoryMappedPtr(void* pointer,
                                      ulong length)
        {
            if(RuntimeHelpers.IsReferenceOrContainsReferences<T>())
            {
                throw new ArgumentOutOfRangeException();
            }

            if(length < 0)
            {
                throw new ArgumentOutOfRangeException();
            }

            _pointer = new ByReference<T>(ref Unsafe.As<byte, T>(ref *(byte*)pointer));
            _length  = length;
        }

        // Constructor for internal use only.
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal MemoryMappedPtr(ref T ptr,
                                 ulong length)
        {
            _pointer = new ByReference<T>(ref ptr);
            _length  = length;
        }

        /// <summary>Returns the specified element of the read-only span.</summary>
        /// <param name="index"></param>
        /// <returns></returns>
        /// <exception cref="System.IndexOutOfRangeException">
        ///     Thrown when index less than 0 or index greater than or equal to
        ///     LongLength
        /// </exception>
        public unsafe ref readonly T this[ulong index]
        {
            [Intrinsic]
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            [NonVersionable]
            get
            {
                if(index >= _length)
                {
                    throw new IndexOutOfRangeException();
                }

                return ref Unsafe.As<byte, T>(ref *(byte*)((ulong)_pointer._value + index * SizeOfT));
            }
        }

        /// <summary>The number of items in the read-only span.</summary>
        public ulong Length
        {
            [NonVersionable]
            get { return _length; }
        }

        /// <summary>Returns true if LongLength is 0.</summary>
        public bool IsEmpty
        {
            [NonVersionable]
            get { return 0 >= _length; } // Workaround for https://github.com/dotnet/runtime/issues/10950
        }

        /// <summary>
        ///     Returns false if left and right point at the same memory and have the same length.  Note that this does *not*
        ///     check to see if the *contents* are equal.
        /// </summary>
        public static bool operator !=(MemoryMappedPtr<T> left,
                                       MemoryMappedPtr<T> right)
        {
            return !(left == right);
        }

        /// <summary>
        ///     This method is not supported as spans cannot be boxed. To compare two spans, use operator==.
        ///     <exception cref="System.NotSupportedException">Always thrown by this method.</exception>
        /// </summary>
        [EditorBrowsable(EditorBrowsableState.Never)]
        public override bool Equals(object? obj)
        {
            throw new NotSupportedException("NotSupported_CannotCallEqualsOnSpan");
        }

        /// <summary>
        ///     This method is not supported as spans cannot be boxed.
        ///     <exception cref="System.NotSupportedException">Always thrown by this method.</exception>
        /// </summary>
        [EditorBrowsable(EditorBrowsableState.Never)]
        public override int GetHashCode()
        {
            throw new NotSupportedException("NotSupported_CannotCallGetHashCodeOnSpan");
        }

        /// <summary>Defines an implicit conversion of an array to a <see cref="MemoryMappedPtr{T}" /></summary>
        public static implicit operator MemoryMappedPtr<T>(T[]? array)
        {
            return new MemoryMappedPtr<T>(array);
        }

        /// <summary>
        ///     Defines an implicit conversion of a <see cref="ArraySegment{T}" /> to a <see cref="MemoryMappedPtr{T}" />
        /// </summary>
        public static implicit operator MemoryMappedPtr<T>(ArraySegment<T> segment)
        {
            return new MemoryMappedPtr<T>(segment.Array, (ulong)segment.Offset, (ulong)segment.Count);
        }

        /// <summary>Returns a 0-length read-only span whose base is the null pointer.</summary>
        public static MemoryMappedPtr<T> Empty { get { return default; } }

        /// <summary>Gets an enumerator for this span.</summary>
        public Enumerator GetEnumerator()
        {
            return new Enumerator(this);
        }

        /// <summary>Enumerates the elements of a <see cref="MemoryMappedPtr{T}" />.</summary>
        public ref struct Enumerator
        {
            /// <summary>The span being enumerated.</summary>
            private readonly MemoryMappedPtr<T> _span;

            /// <summary>The next index to yield.</summary>
            private ulong _index;

            /// <summary>Initialize the enumerator.</summary>
            /// <param name="span">The span to enumerate.</param>
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            internal Enumerator(MemoryMappedPtr<T> span)
            {
                _span  = span;
                _index = ulong.MaxValue;
            }

            /// <summary>Advances the enumerator to the next element of the span.</summary>
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            public bool MoveNext()
            {
                ulong index = _index + 1;

                if(index < _span.Length)
                {
                    _index = index;

                    return true;
                }

                return false;
            }

            /// <summary>Gets the element at the current position of the enumerator.</summary>
            public ref readonly T Current
            {
                [MethodImpl(MethodImplOptions.AggressiveInlining)]
                get { return ref _span[_index]; }
            }
        }

        /// <summary>
        ///     Returns a reference to the 0th element of the Span. If the Span is empty, returns null reference. It can be
        ///     used for pinning and is required to support the use of span within a fixed statement.
        /// </summary>
        [EditorBrowsable(EditorBrowsableState.Never)]
        public unsafe ref readonly T GetPinnableReference()
        {
            // Ensure that the native code has just one forward branch that is predicted-not-taken.
            ref T ret = ref Unsafe.AsRef<T>(null);

            if(_length != 0)
            {
                ret = ref _pointer.Value;
            }

            return ref ret;
        }

        /// <summary>
        ///     Copies the contents of this read-only span into destination span. If the source and destinations overlap, this
        ///     method behaves as if the original values in a temporary location before the destination is overwritten.
        ///     <param name="destination">The span to copy items into.</param>
        ///     <exception cref="System.ArgumentException">Thrown when the destination Span is shorter than the source Span.</exception>
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void CopyTo(MemoryMappedPtr<T> destination)
        {
            // Using "if (!TryCopyTo(...))" results in two branches: one for the length
            // check, and one for the result of TryCopyTo. Since these checks are equivalent,
            // we can optimize by performing the check once ourselves then calling Memmove directly.

            if(_length <= destination.Length)
            {
                Mem.Memmove(ref destination._pointer.Value, ref _pointer.Value, _length);
            }
            else
            {
                throw new ArgumentException();
            }
        }

        /// <summary>
        ///     Copies the contents of this read-only span into destination span. If the source and destinations overlap, this
        ///     method behaves as if the original values in a temporary location before the destination is overwritten.
        /// </summary>
        /// <returns>
        ///     If the destination span is shorter than the source span, this method return false and no data is written to
        ///     the destination.
        /// </returns>
        /// <param name="destination">The span to copy items into.</param>
        public bool TryCopyTo(MemoryMappedPtr<T> destination)
        {
            bool retVal = false;

            if(_length <= destination.Length)
            {
                Mem.Memmove(ref destination._pointer.Value, ref _pointer.Value, _length);
                retVal = true;
            }

            return retVal;
        }

        /// <summary>
        ///     Returns true if left and right point at the same memory and have the same length.  Note that this does *not*
        ///     check to see if the *contents* are equal.
        /// </summary>
        public static bool operator ==(MemoryMappedPtr<T> left,
                                       MemoryMappedPtr<T> right)
        {
            return left._length == right._length && Unsafe.AreSame(ref left._pointer.Value, ref right._pointer.Value);
        }

        /// <summary>
        ///     For <see cref="MemoryMappedPtr{Char}" />, returns a new instance of string that represents the characters
        ///     pointed to by the span. Otherwise, returns a <see cref="string" /> with the name of the type and the number of
        ///     elements.
        /// </summary>
        public override unsafe string ToString()
        {
            if(typeof(T) == typeof(char))
            {
                return new string((char*)Unsafe.AsPointer(ref _pointer.Value));
            }
#if FEATURE_UTF8STRING
        else if (typeof(T) == typeof(Char8))
        {
            // TODO_UTF8STRING: Call into optimized transcoding routine when it's available.
            return Encoding.UTF8.GetString(new MemoryMappedPtr<byte>(ref Unsafe.As<T, byte>(ref _pointer.Value), _length));
        }
#endif // FEATURE_UTF8STRING
            return string.Format("System.MemoryMappedPtr<{0}>[{1}]", typeof(T).Name, _length);
        }

        /// <summary>Forms a slice out of the given read-only span, beginning at 'start'.</summary>
        /// <param name="start">The index at which to begin this slice.</param>
        /// <exception cref="System.ArgumentOutOfRangeException">
        ///     Thrown when the specified <paramref name="start" /> index is not
        ///     in range (&lt;0 or &gt;LongLength).
        /// </exception>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public unsafe MemoryMappedPtr<T> Slice(ulong start)
        {
            if(start > _length)
            {
                throw new ArgumentOutOfRangeException();
            }

            return new MemoryMappedPtr<T>((void*)((ulong)_pointer._value + (start * SizeOfT)), _length - start);
        }

        /// <summary>Forms a slice out of the given read-only span, beginning at 'start', of given length</summary>
        /// <param name="start">The index at which to begin this slice.</param>
        /// <param name="length">The desired length for the slice (exclusive).</param>
        /// <exception cref="System.ArgumentOutOfRangeException">
        ///     Thrown when the specified <paramref name="start" /> or end index
        ///     is not in range (&lt;0 or &gt;LongLength).
        /// </exception>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public unsafe MemoryMappedPtr<T> Slice(ulong start,
                                               ulong length)
        {
#if TARGET_64BIT
            // See comment in Span<T>.Slice for how this works.
            if(start + length > _length)
            {
                throw new ArgumentOutOfRangeException();
            }
#else
        if ((ulong)start > (ulong)_length || (ulong)length > (ulong)(_length - start))
            throw new ArgumentOutOfRangeException();
#endif

            return new MemoryMappedPtr<T>((void*)((ulong)_pointer._value + (start * SizeOfT)), length);
        }

        /// <summary>
        ///     Copies the contents of this read-only span into a new array.  This heap allocates, so should generally be
        ///     avoided, however it is sometimes necessary to bridge the gap with APIs written in terms of arrays.
        /// </summary>
        public T[] ToArray()
        {
            if(_length == 0)
            {
                return Array.Empty<T>();
            }

            T[] destination = new T[_length];
            Mem.Memmove(ref GetArrayDataReference(destination), ref _pointer.Value, _length);

            return destination;
        }
    }
}
