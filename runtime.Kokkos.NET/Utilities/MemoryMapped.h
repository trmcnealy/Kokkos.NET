// //////////////////////////////////////////////////////////
// MemoryMapped.h
// Copyright (c) 2013 Stephan Brumme. All rights reserved.
// see http://create.stephan-brumme.com/disclaimer.html
//

#pragma once

#include "KokkosAPI.hpp"

// define fixed size integer types
//#ifdef _WINDOWS
// typedef unsigned __int64 uint64;
//#else
//#include <stdint.h>
//#endif

#include <string>

/// Portable read-only memory mapping (Windows and Linux)
/** Filesize limited by unsigned __int64, usually 2^32 or 2^64 */
class MemoryMapped
{
public:
    /// tweak performance
    enum CacheHint
    {
        Normal, ///< good overall performance
        SequentialScan, ///< read file only once with few seeks
        RandomAccess ///< jump around
    };

    /// how much should be mappend
    enum MapRange
    {
        WholeFile = 0 ///< everything ... be careful when file is larger than memory
    };

    /// do nothing, must use open()
    MemoryMapped();
    /// open file, mappedBytes = 0 maps the whole file
    MemoryMapped(const std::string& filename, unsigned __int64 mappedBytes = WholeFile, CacheHint hint = Normal);
    /// close file (see close() )
    ~MemoryMapped();

    /// open file, mappedBytes = 0 maps the whole file
    bool Open(const std::string& filename, unsigned __int64 mappedBytes = WholeFile, CacheHint hint = Normal);
    /// close file
    void Close();

    /// access position, no range checking (faster)
    unsigned char operator[](unsigned __int64 offset) const;
    /// access position, including range checking
    unsigned char At(unsigned __int64 offset) const;

    /// raw access
    const unsigned char* GetData() const;

    /// true, if file successfully opened
    bool IsValid() const;

    /// get file size
    uint64 Size() const;
    /// get number of actually mapped bytes
    unsigned __int64 MappedSize() const;

    /// replace mapping by a new one of the same file, offset MUST be a multiple of the page size
    bool Remap(uint64 offset, unsigned __int64 mappedBytes);

private:
    /// don't copy object
    MemoryMapped(const MemoryMapped&) = default;
    /// don't copy object
    MemoryMapped& operator=(const MemoryMapped&) = default;

    /// get OS page size (for remap)
    static int getpagesize();

    /// file name
    std::string _filename;

    /// file size
    uint64 _filesize;

    /// caching strategy
    CacheHint _hint;

    /// mapped size
    unsigned __int64 _mappedBytes;

    /// define handle
#ifdef _WINDOWS
    typedef void* FileHandle;

    /// Windows handle to memory mapping of _file
    void* _mappedFile;
#else
    typedef int FileHandle;
#endif

    /// file handle
    FileHandle _file;
    /// pointer to the file contents mapped into memory
    void* _mappedView;
};

KOKKOS_NET_API_EXTERNC MemoryMapped* Create() noexcept;
KOKKOS_NET_API_EXTERNC MemoryMapped* CreateAndOpen(const char*             filename,
                                                   unsigned __int64        mappedBytes = MemoryMapped::WholeFile,
                                                   MemoryMapped::CacheHint hint        = MemoryMapped::CacheHint::Normal) noexcept;
KOKKOS_NET_API_EXTERNC void          Destory(MemoryMapped* mm) noexcept;

KOKKOS_NET_API_EXTERNC bool Open(MemoryMapped*           mm,
                                 const char*             filename,
                                 unsigned __int64        mappedBytes = MemoryMapped::WholeFile,
                                 MemoryMapped::CacheHint hint        = MemoryMapped::CacheHint::Normal) noexcept;
KOKKOS_NET_API_EXTERNC void Close(MemoryMapped* mm);

// KOKKOS_NET_API_EXTERN unsigned char operator[](unsigned __int64 offset) noexcept;

KOKKOS_NET_API_EXTERNC unsigned char        At(MemoryMapped* mm, unsigned __int64 offset) noexcept;
KOKKOS_NET_API_EXTERNC const unsigned char* GetData(MemoryMapped* mm) noexcept;
KOKKOS_NET_API_EXTERNC bool                 IsValid(MemoryMapped* mm) noexcept;
KOKKOS_NET_API_EXTERNC uint64               Size(MemoryMapped* mm) noexcept;
KOKKOS_NET_API_EXTERNC unsigned __int64     MappedSize(MemoryMapped* mm) noexcept;
KOKKOS_NET_API_EXTERNC bool                 Remap(MemoryMapped* mm, uint64 offset, unsigned __int64 mappedBytes);
