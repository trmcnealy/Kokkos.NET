using System.Runtime.CompilerServices;
using System.Runtime.Versioning;

namespace System
{
    [NonVersionable]
    internal readonly unsafe ref struct ByReference<T>
    {
#pragma warning disable CA1823, 169
        internal readonly void* _value;
#pragma warning restore CA1823, 169

        [Intrinsic]
        public ByReference(ref T value)
        {
            _value = Unsafe.AsPointer(ref value);
        }

        public ref T Value
        {
            [Intrinsic]
            get { return ref Unsafe.AsRef<T>(_value); }
        }
    }
}
