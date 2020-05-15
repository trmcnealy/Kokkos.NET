// ReSharper disable InvalidXmlDocComment
// ReSharper disable InconsistentNaming

using System;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

using RuntimeGeneration;

namespace Kokkos
{
    public enum CNMeMStatus : uint
    {
        CNMEM_STATUS_SUCCESS = 0,

        CNMEM_STATUS_CUDA_ERROR,

        CNMEM_STATUS_INVALID_ARGUMENT,

        CNMEM_STATUS_NOT_INITIALIZED,

        CNMEM_STATUS_OUT_OF_MEMORY,

        CNMEM_STATUS_UNKNOWN_ERROR,
    }

    public enum CNMeMManagerFlags : uint
    {
        CNMEM_FLAGS_DEFAULT = 0,

        /// <summary>
        /// Default flags.
        /// </summary>
        CNMEM_FLAGS_CANNOT_GROW = 1,

        /// <summary>
        /// Prevent the manager from growing its memory consumption.
        /// </summary>
        CNMEM_FLAGS_CANNOT_STEAL = 2,

        /// <summary>
        /// Prevent the manager from stealing memory.
        /// </summary>
        CNMEM_FLAGS_MANAGED = 4
    }

    [StructLayout(System.Runtime.InteropServices.LayoutKind.Sequential, CharSet = CharSet.Ansi)]
    public struct CNMeMDevice
    {
        /// <summary>
        /// The device number.
        /// </summary>
        public int device;

        /// <summary>
        /// The size to allocate for that device. If 0, the implementation chooses the size.
        /// </summary>
        public ulong size;

        /// <summary>
        /// The number of named streams associated with the device. The NULL stream is not counted.
        /// </summary>
        public int numStreams;

        /// <summary>
        /// The streams associated with the device. It can be NULL. The NULL stream is managed.
        /// </summary>
        public IntPtr streams;

        /// <summary>
        /// The size reserved for each streams. It can be 0.
        /// </summary>
        public IntPtr streamSizes;
    }

    [StructLayout(System.Runtime.InteropServices.LayoutKind.Sequential, CharSet = CharSet.Ansi)]
    public readonly struct CUstream : IEquatable<CUstream>
    {
        private readonly IntPtr _handle;

        public CUstream(IntPtr handle) => _handle = handle;

        public IntPtr Handle => _handle;

        public bool Equals(CUstream other) => _handle.Equals(other._handle);

        public override bool Equals(object obj) => obj is CUstream other && Equals(other);

        public override int GetHashCode() => _handle.GetHashCode();

        public override string ToString() => "0x" + (IntPtr.Size == 8 ? _handle.ToString("X16") : _handle.ToString("X8"));

        public static bool operator ==(CUstream left,
                                       CUstream right) =>
            left.Equals(right);

        public static bool operator !=(CUstream left,
                                       CUstream right) =>
            !left.Equals(right);
    }

    [StructLayout(System.Runtime.InteropServices.LayoutKind.Sequential, CharSet = CharSet.Ansi)]
    public struct iobuf
    {
        public IntPtr _ptr;

        public int _cnt;

        public IntPtr _base;

        public int _flag;

        public int _file;

        public int _charbuf;

        public int _bufsiz;

        public IntPtr _tmpfname;
    }

    public static class CNMeM
    {
        public delegate CNMeMStatus cnmemInitDelegate(int             numDevices,
                                                      ref CNMeMDevice devices,
                                                      uint            flags);

        public delegate CNMeMStatus cnmemFinalizeDelegate();

        public delegate CNMeMStatus cnmemRetainDelegate();

        public delegate CNMeMStatus cnmemReleaseDelegate();

        public delegate CNMeMStatus cnmemRegisterStreamDelegate(ref CUstream stream);

        public delegate CNMeMStatus cnmemMallocDelegate(out IntPtr   ptr,
                                                        ulong        size,
                                                        ref CUstream stream);

        public delegate CNMeMStatus cnmemFreeDelegate(IntPtr       ptr,
                                                      ref CUstream stream);

        public delegate CNMeMStatus cnmemMemGetInfoDelegate(ref ulong    freeMem,
                                                            ref ulong    totalMem,
                                                            ref CUstream stream);

        public delegate CNMeMStatus cnmemPrintMemoryStateDelegate(ref iobuf    file,
                                                                  ref CUstream stream);

        public delegate IntPtr cnmemGetErrorStringDelegate(ref CNMeMStatus status);

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        static CNMeM()
        {
            RuntimeCil.Generate(typeof(CNMeM).Assembly);
        }

        /// <summary>
        /// 
        /// </summary>
        /// <returns>CNMEM_STATUS_SUCCESS,          if everything goes fine,
        /// CNMEM_STATUS_INVALID_ARGUMENT, if one of the argument is invalid,
        /// CNMEM_STATUS_OUT_OF_MEMORY,    if the requested size exceeds the available memory,
        /// CNMEM_STATUS_CUDA_ERROR,       if an error happens in a CUDA function.</returns>
        /// <remarks>
        /// @brief Initialize the library and allocate memory on the listed devices.For each device, an internal memory manager is created and the specified amount of memory is
        /// allocated (it is the size defined in device[i].size). For each, named stream an additional
        /// memory manager is created. Currently, it is implemented as a tree of memory managers: A root
        /// manager for the device and a list of children, one for each named stream.This function must be called before any other function in the library. It has to be called
        /// by a single thread since it is not thread-safe.
        /// </remarks>
        public const string LibraryName = "runtime.Kokkos.NET";

        /// <summary>
        /// 
        /// </summary>
        /// <param name="numDevices"></param>
        /// <param name="devices"></param>
        /// <param name="flags"></param>
        /// <returns></returns>
        [NativeCall(LibraryName, "cnmemInit", true)]
        public static readonly cnmemInitDelegate Init;

        /// <summary>
        /// 
        /// </summary>
        /// <returns>CNMEM_STATUS_SUCCESS,          if everything goes fine,
        /// CNMEM_STATUS_NOT_INITIALIZED,  if the ::cnmemInit function has not been called,
        /// CNMEM_STATUS_CUDA_ERROR,       if an error happens in one of the CUDA functions.</returns>
        /// <remarks>
        /// @brief Release all the allocated memory.This function must be called by a single thread and after all threads that called
        /// cnmemMalloc/cnmemFree have joined. This function is not thread-safe.
        /// </remarks>
        [NativeCall(LibraryName, "cnmemFinalize", true)]
        public static readonly cnmemFinalizeDelegate @Finalize;

        /// <summary>
        /// 
        /// </summary>
        /// <returns>CNMEM_STATUS_SUCCESS,          if everything goes fine,
        /// CNMEM_STATUS_NOT_INITIALIZED,  if the ::cnmemInit function has not been called,</returns>
        /// <remarks>
        /// @brief Increase the internal reference counter of the context object.This function increases the internal reference counter of the library. The purpose of that
        /// reference counting mechanism is to give more control to the user over the lifetime of the
        /// library. It is useful with scoped memory allocation which may be destroyed in a final
        /// memory collection after the end of main(). That function is thread-safe.
        /// </remarks>
        [NativeCall(LibraryName, "cnmemRetain", true)]
        public static readonly cnmemRetainDelegate Retain;

        /// <summary>
        /// 
        /// </summary>
        /// <returns>CNMEM_STATUS_SUCCESS,          if everything goes fine,
        /// CNMEM_STATUS_NOT_INITIALIZED,  if the ::cnmemInit function has not been called,</returns>
        /// <remarks>
        /// @brief Decrease the internal reference counter of the context object.This function decreases the internal reference counter of the library. The purpose of that
        /// reference counting mechanism is to give more control to the user over the lifetime of the
        /// library. It is useful with scoped memory allocation which may be destroyed in a final
        /// memory collection after the end of main(). That function is thread-safe.You can use @c cnmemRelease to explicitly finalize the library.
        /// </remarks>
        [NativeCall(LibraryName, "cnmemRelease", true)]
        public static readonly cnmemReleaseDelegate Release;

        /// <summary>
        /// 
        /// </summary>
        /// <returns>CNMEM_STATUS_SUCCESS,          if everything goes fine,
        /// CNMEM_STATUS_INVALID_ARGUMENT, if one of the argument is invalid,</returns>
        /// <remarks>
        /// @brief Add a new stream to the pool of managed streams on a device.This function registers a new stream into a device memory manager. It is thread-safe.
        /// </remarks>
        [NativeCall(LibraryName, "cnmemRegisterStream", true)]
        public static readonly cnmemRegisterStreamDelegate RegisterStream;

        /// <summary>
        /// 
        /// </summary>
        /// <returns>CNMEM_STATUS_SUCCESS,          if everything goes fine,
        /// CNMEM_STATUS_NOT_INITIALIZED,  if the ::cnmemInit function has not been called,
        /// CNMEM_STATUS_INVALID_ARGUMENT, if one of the argument is invalid. For example, ptr == 0,
        /// CNMEM_STATUS_OUT_OF_MEMORY,    if there is not enough memory available,
        /// CNMEM_STATUS_CUDA_ERROR,       if an error happens in one of the CUDA functions.</returns>
        /// <remarks>
        /// @brief Allocate memory.This function allocates memory and initializes a pointer to device memory. If no memory
        /// is available, it returns a CNMEM_STATUS_OUT_OF_MEMORY error. This function is thread safe.The behavior of that function is the following:- If the stream is NULL, the root memory manager is asked to allocate a buffer of device
        /// memory. If there's a buffer of size larger or equal to the requested size in the list of
        /// free blocks, it is returned. If there's no such buffer but the manager is allowed to grow
        /// its memory usage (the CNMEM_FLAGS_CANNOT_GROW flag is not set), the memory manager calls
        /// cudaMalloc. If cudaMalloc fails due to no more available memory or the manager is not
        /// allowed to grow, the manager attempts to steal memory from one of its children (unless
        /// CNMEM_FLAGS_CANNOT_STEAL is set). If that attempt also fails, the manager returns
        /// CNMEM_STATUS_OUT_OF_MEMORY.- If the stream is a named stream, the initial request goes to the memory manager associated
        /// with that stream. If a free node is available in the lists of that manager, it is returned.
        /// Otherwise, the request is passed to the root node and works as if the request were made on
        /// the NULL stream.The calls to cudaMalloc are potentially costly and may induce GPU synchronizations. Also the
        /// mechanism to steal memory from the children induces GPU synchronizations (the manager has to
        /// make sure no kernel uses a given buffer before stealing it) and it the execution is
        /// sequential (in a multi-threaded context, the code is executed in a critical section inside
        /// the cnmem library - no need for the user to wrap cnmemMalloc with locks).
        /// </remarks>
        [NativeCall(LibraryName, "cnmemMalloc", true)]
        public static readonly cnmemMallocDelegate Malloc;

        /// <summary>
        /// 
        /// </summary>
        /// <returns>CNMEM_STATUS_SUCCESS,          if everything goes fine,
        /// CNMEM_STATUS_NOT_INITIALIZED,  if the ::cnmemInit function has not been called,
        /// CNMEM_STATUS_INVALID_ARGUMENT, if one of the argument is invalid. For example, ptr == 0,
        /// CNMEM_STATUS_CUDA_ERROR,       if an error happens in one of the CUDA functions.</returns>
        /// <remarks>
        /// @brief Release memory.This function releases memory and recycles a memory block in the manager. This function is
        /// thread safe.
        /// </remarks>
        [NativeCall(LibraryName, "cnmemFree", true)]
        public static readonly cnmemFreeDelegate Free;

        /// <summary>
        /// 
        /// </summary>
        /// <returns>CNMEM_STATUS_SUCCESS,          if everything goes fine,
        /// CNMEM_STATUS_NOT_INITIALIZED,  if the ::cnmemInit function has not been called,
        /// CNMEM_STATUS_INVALID_ARGUMENT, if one of the argument is invalid,
        /// CNMEM_STATUS_CUDA_ERROR,       if an error happens in one of the CUDA functions.</returns>
        /// <remarks>
        /// @brief Returns the amount of memory managed by the memory manager associated with a stream.The pointers totalMem and freeMem must be valid. At the moment, this function has a comple-
        /// xity linear in the number of allocated blocks so do not call it in performance critical
        /// sections.
        /// </remarks>
        [NativeCall(LibraryName, "cnmemMemGetInfo", true)]
        public static readonly cnmemMemGetInfoDelegate Info;

        /// <summary>
        /// 
        /// </summary>
        /// <returns>CNMEM_STATUS_SUCCESS,          if everything goes fine,
        /// CNMEM_STATUS_NOT_INITIALIZED,  if the ::cnmemInit function has not been called,
        /// CNMEM_STATUS_INVALID_ARGUMENT, if one of the argument is invalid. For example, used_mem == 0
        /// or free_mem == 0,
        /// CNMEM_STATUS_CUDA_ERROR,       if an error happens in one of the CUDA functions.</returns>
        /// <remarks>
        /// @brief Print a list of nodes to a file.This function is intended to be used in case of complex scenarios to help understand the
        /// behaviour of the memory managers/application. It is thread safe.
        /// </remarks>
        [NativeCall(LibraryName, "cnmemPrintMemoryState", true)]
        public static readonly cnmemPrintMemoryStateDelegate PrintMemoryState;

        /// <summary>
        /// 
        /// </summary>
        /// <remarks>
        /// @brief Converts a cnmemStatus_t value to a string.
        /// </remarks>
        [NativeCall(LibraryName, "cnmemGetErrorString", true)]
        public static readonly cnmemGetErrorStringDelegate GetErrorString;
    }
}
