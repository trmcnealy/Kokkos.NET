
#include "GpuMemoryMap.h"

class Block
{
    /// The pointer to the memory region on the device.
    char* mData;
    /// The size of the memory buffer.
    uint64 mSize;
    /// The prev/next blocks in the linked list of blocks.
    Block* mNext;
    /// Is it a head node (i.e. a node obtained from parent->allocate or cudaMalloc).
    bool mIsHead;

public:
    /// Create a block.
    Block(char* data, uint64 size, Block* next, bool isHead) : mData(data), mSize(size), mNext(next), mIsHead(isHead) {}

    /// The data.
    inline const char* getData() const { return mData; }
    /// The data (mutable).
    inline char* getData() { return mData; }

    /// The size of the block.
    inline uint64 getSize() const { return mSize; }

    /// The next block in the linked list.
    inline const Block* getNext() const { return mNext; }
    /// The next block in the linked list (mutable).
    inline Block* getNext() { return mNext; }

    /// Is it a head block.
    inline bool isHead() const { return mIsHead; }

    /// Change the next block.
    inline void setNext(Block* next) { mNext = next; }
    /// Change the size of the block.
    inline void setSize(uint64 size) { mSize = size; }
    /// Set the head flag.
    inline void setHeadFlag(bool isHead) { mIsHead = isHead; }
};

class Manager
{
    /// The parent manager.
    Manager* mParent;
    /// The children managers.
    std::vector<Manager*> mChildren;
    /// The GPU device where the memory is allocated.
    int mDevice;
    /// The stream this manager is associated with. It could be NULL.
    cudaStream_t mStream;
    /// Is the stream blocking?
    bool mIsStreamBlocking;
    /// The list of used blocks.
    Block* mUsedBlocks;
    /// The list of free blocks.
    Block* mFreeBlocks;
    /// The managed memory size.
    uint64 mSize;
    /// The flags.
    unsigned mFlags;
    /// To support multi-threading. Each manager has its own mutex.
    Mutex mMutex;

public:
    /// Create an unitialized manager.
    Manager();
    /// Dtor.
    ~Manager();

    /// Allocate a block of memory.
    cnmemStatus_t allocate(void*& ptr, uint64 size, bool isBlocking = true);
    /// Release a block of memory.
    cnmemStatus_t release(void* ptr);
    /// Release memory. It returns true if we have no memory leak.
    cnmemStatus_t releaseAllUnsafe();
    /// Reserve memory for a manager.
    cnmemStatus_t reserve(uint64 size);
    /// Steal memory from another manager.
    cnmemStatus_t stealUnsafe(void*& ptr, uint64 size);

    /// Print the full memory state.
    cnmemStatus_t printMemoryState(FILE* file) const;

    /// The amount of used memory.
    inline cnmemStatus_t getUsedMemoryUnsafe(uint64& usedMemory) const { return getMemoryUnsafe(usedMemory, mUsedBlocks); }
    /// The amount of used memory.
    inline cnmemStatus_t getFreeMemoryUnsafe(uint64& freeMemory) const { return getMemoryUnsafe(freeMemory, mFreeBlocks); }

    /// Get a specific child based on the stream id.
    cnmemStatus_t getChildFromStream(Manager*& manager, cudaStream_t stream) const;
    /// Get a specific child based on the stream id.
    cnmemStatus_t getChild(Manager*& manager, uint64 i) const;
    /// Add a new child.
    cnmemStatus_t addChild(Manager* manager);
    /// The number of children.
    cnmemStatus_t getNumChildren(uint64& numChildren) const;

    /// The associated device.
    inline int getDevice() const { return mDevice; }
    /// The flags.
    inline unsigned getFlags() const { return mFlags; }
    /// Get the mutex.
    inline const Mutex* getMutex() const { return &mMutex; }
    /// The size allocated to that manager.
    inline uint64 getSize() const { return mSize; }
    /// The CUDA stream.
    inline cudaStream_t getStream() const { return mStream; }

    /// Define the parent.
    inline void setParent(Manager* parent) { mParent = parent; }
    /// Define the device.
    inline void setDevice(int device) { mDevice = device; }
    /// Define the stream.
    inline cnmemStatus_t setStream(cudaStream_t stream)
    {
        mStream = stream;
#ifdef CUDA_API_PER_THREAD_DEFAULT_STREAM
        mIsStreamBlocking = false;
#elif CUDART_VERSION < 5050
        mIsStreamBlocking = true;
#else
        unsigned flags = 0;
        CNMEM_CHECK_CUDA(cudaStreamGetFlags(mStream, &flags));
        mIsStreamBlocking = !mStream || !(flags & cudaStreamNonBlocking);
#endif
        return CNMEM_STATUS_SUCCESS;
    }
    /// Define the flags.
    inline void setFlags(unsigned flags) { mFlags = flags; }

private:
    /// The member functions below which are marked "Unsafe" are not thread-safe when called on a
    /// same Manager object. Make sure they are called by a single thread in that case.

    /// Allocate a new block and add it to the free list.
    cnmemStatus_t allocateBlockUnsafe(Block*& curr, Block*& prev, uint64 size);
    /// Release a block from the active list.
    cnmemStatus_t releaseBlockUnsafe(Block* curr, Block* prev);
    /// Find the best free node based on the size.
    cnmemStatus_t findBestBlockUnsafe(Block*& curr, Block*& prev, uint64 size);
    /// Extract a node from the list of free blocks.
    cnmemStatus_t extractBlockUnsafe(Block* curr, Block* prev, uint64 size, bool stolen);

    /// Give a free block from that manager.
    cnmemStatus_t giveBlockUnsafe(void*& data, uint64& dataSize, uint64 size);
    /// Steal a block from another manager.
    cnmemStatus_t stealBlockUnsafe(void*& data, uint64& dataSize, uint64 size);

    /// The memory consumption of a list.
    cnmemStatus_t getMemoryUnsafe(uint64& memSize, const Block* head) const;
    /// Print an internal linked list.
    cnmemStatus_t printListUnsafe(FILE* file, const char* name, const Block* head) const;
};

Manager::Manager() :
    mParent(NULL), mChildren(), mDevice(-1), mStream(NULL), mIsStreamBlocking(false), mUsedBlocks(NULL), mFreeBlocks(NULL), mSize(0), mFlags(CNMEM_FLAGS_DEFAULT), mMutex()
{
    mMutex.initialize();
}

Manager::~Manager()
{
    if(mDevice == -1 || cudaSetDevice(mDevice) != cudaSuccess)
    { // Invalid device, skip it.
        return;
    }
    releaseAllUnsafe();
    mMutex.finalize();
}

cnmemStatus_t Manager::addChild(Manager* manager)
{
    CNMEM_CHECK(mMutex.lock());
    mChildren.push_back(manager);
    CNMEM_CHECK(mMutex.unlock());
    return CNMEM_STATUS_SUCCESS;
}

cnmemStatus_t Manager::allocate(void*& ptr, uint64 size, bool isBlocking)
{
    CNMEM_CHECK(mMutex.lock());

    // If the client is not blocking, we have to explicitly synchronize before giving one buffer.
    if(!isBlocking)
    {
        CNMEM_CHECK_CUDA_OR_UNLOCK(cudaStreamSynchronize(mStream), mMutex);
    }

    // Find the best fit.
    Block *best = NULL, *prev = NULL;
    CNMEM_CHECK_OR_UNLOCK(findBestBlockUnsafe(best, prev, size), mMutex);

    // If there's no block left in the list of free blocks (with a sufficient size). Request a new block.
    if(best == NULL && !(mFlags & CNMEM_FLAGS_CANNOT_GROW))
    {
        CNMEM_CHECK_OR_UNLOCK(allocateBlockUnsafe(best, prev, size), mMutex);
    }

    // Make sure we do have a block or quit.
    if(!best)
    {
        ptr = NULL;
        CNMEM_CHECK(mMutex.unlock());
        return CNMEM_STATUS_OUT_OF_MEMORY;
    }

    // Split the free block if needed.
    CNMEM_CHECK_OR_UNLOCK(extractBlockUnsafe(best, prev, size, false), mMutex);

    // Push the node to the list of used nodes.
    best->setNext(mUsedBlocks);
    mUsedBlocks = best;

    // Return the new pointer into memory.
    ptr = mUsedBlocks->getData();
    CNMEM_CHECK(mMutex.unlock());
    return CNMEM_STATUS_SUCCESS;
}

cnmemStatus_t Manager::allocateBlockUnsafe(Block*& curr, Block*& prev, uint64 size)
{
    // Reset the outputs.
    curr = prev = NULL;

    // Try to allocate data from the parent or the device.
    void* data = NULL;
    if(mParent)
    {
        CNMEM_CHECK(mParent->allocate(data, size, mIsStreamBlocking));
    }
    else
    {
        if(mFlags & CNMEM_FLAGS_MANAGED)
        {
            CNMEM_DEBUG_INFO("cudaMallocManaged(%lu)\n", size);
            CNMEM_CHECK_CUDA(cudaMallocManaged(&data, size));
            CNMEM_CHECK_CUDA(cudaMemPrefetchAsync(data, size, mDevice));
        }
        else
        {
            CNMEM_DEBUG_INFO("cudaMalloc(%lu)\n", size);
            CNMEM_CHECK_CUDA(cudaMalloc(&data, size));
        }
        CNMEM_DEBUG_INFO(">> returned address=0x%016lx\n", (uint64)data);
    }

    // If it failed, there's an unexpected issue.
    CNMEM_ASSERT(data);

    // We have data, we now need to add it to the list of free nodes. We keep the list sorted.
    Block* next = mFreeBlocks;
    for(; next && next->getData() < data; next = next->getNext())
    {
        prev = next;
    }
    curr = new Block((char*)data, size, next, true);
    if(!curr)
    {
        return CNMEM_STATUS_OUT_OF_MEMORY;
    }
    if(prev)
    {
        prev->setNext(curr);
    }
    else
    {
        mFreeBlocks = curr;
    }

    return CNMEM_STATUS_SUCCESS;
}

cnmemStatus_t Manager::extractBlockUnsafe(Block* curr, Block* prev, uint64 size, bool stolen)
{
    // We have two cases: 1/ It is the right size so we keep it or 2/ it is too large and we split the node.
    Block* next;
    if(curr->getSize() == size)
    {
        next = curr->getNext();
    }
    else
    {
        uint64 remaining = curr->getSize() - size;
        Block*      newBlock  = new Block(curr->getData() + size, remaining, curr->getNext(), stolen);
        if(!newBlock)
        {
            return CNMEM_STATUS_OUT_OF_MEMORY;
        }
        next = newBlock;
        curr->setSize(size);
    }

    // Redo the "branching" in the nodes.
    if(prev)
    {
        prev->setNext(next);
    }
    else
    {
        mFreeBlocks = next;
    }
    return CNMEM_STATUS_SUCCESS;
}

cnmemStatus_t Manager::findBestBlockUnsafe(Block*& best, Block*& prev, uint64 size)
{
    best = NULL, prev = NULL;
    for(Block *temp = mFreeBlocks, *tempPrev = NULL; temp; temp = temp->getNext())
    {
        if(temp->getSize() >= size && (!best || temp->getSize() < best->getSize()))
        {
            best = temp;
            prev = tempPrev;
        }
        tempPrev = temp;
    }
    return CNMEM_STATUS_SUCCESS;
}

cnmemStatus_t Manager::getChildFromStream(Manager*& manager, cudaStream_t stream) const
{
    CNMEM_CHECK(mMutex.lock());
    uint64 i = 0, numChildren = mChildren.size();
    for(; i < numChildren; ++i)
    {
        if(mChildren[i]->mStream == stream)
        {
            manager = mChildren[i];
            break;
        }
    }
    CNMEM_CHECK(mMutex.unlock());
    return i < numChildren ? CNMEM_STATUS_SUCCESS : CNMEM_STATUS_INVALID_ARGUMENT;
}

cnmemStatus_t Manager::getChild(Manager*& manager, uint64 i) const
{
    CNMEM_CHECK(mMutex.lock());
    if(i >= mChildren.size())
    {
        CNMEM_CHECK(mMutex.unlock());
        return CNMEM_STATUS_INVALID_ARGUMENT;
    }
    manager = mChildren[i];

    CNMEM_CHECK(mMutex.unlock());
    return CNMEM_STATUS_SUCCESS;
}

cnmemStatus_t Manager::getMemoryUnsafe(uint64& size, const Block* head) const
{
    size = 0;
    for(Block* curr = (Block*)head; curr; curr = curr->getNext())
    {
        size += curr->getSize();
    }
    return CNMEM_STATUS_SUCCESS;
}

#if 0
cnmemStatus_t Manager::getMemory(uint64 &size, const Block *head) const {
    CNMEM_CHECK(mMutex.lock());
    CNMEM_CHECK_OR_UNLOCK(getMemoryUnsafe(size, head));
    CNMEM_CHECK(mMutex.unlock());
    return status;
}
#endif

cnmemStatus_t Manager::getNumChildren(uint64& numChildren) const
{
    CNMEM_CHECK(mMutex.lock());
    numChildren = mChildren.size();
    CNMEM_CHECK(mMutex.unlock());
    return CNMEM_STATUS_SUCCESS;
}

cnmemStatus_t Manager::giveBlockUnsafe(void*& blockData, uint64& blockSize, uint64 size)
{
    // Make sure the block is not in use any more. It could be too coarse grain and we may change
    // it in the future.
    CNMEM_CHECK_CUDA(cudaStreamSynchronize(mStream));

    // Init the returned values to 0.
    blockData = NULL;
    blockSize = 0;

    // Find the best node to steal and reserve it.
    Block *best = NULL, *prev = NULL;
    CNMEM_CHECK(findBestBlockUnsafe(best, prev, size));
    if(!best)
    {
        return CNMEM_STATUS_OUT_OF_MEMORY;
    }
    CNMEM_CHECK(extractBlockUnsafe(best, prev, size, true));
    blockData = best->getData();
    blockSize = best->getSize();

    // Release the memory used by that block.
    delete best;
    return CNMEM_STATUS_SUCCESS;
}

cnmemStatus_t Manager::printListUnsafe(FILE* file, const char* name, const Block* head) const
{
    uint64 size = 0;
    for(Block* curr = (Block*)head; curr; curr = curr->getNext())
    {
        size += curr->getSize();
    }
#ifdef CNMEM_BUILD_WITH_32_BIT_POINTERS
    fprintf(file, "| list=\"%s\", size=%u\n", name, size);
    for(Block* curr = (Block*)head; curr; curr = curr->getNext())
    {
        fprintf(file,
                "| | node=0x%08x, data=0x%08x, size=%u, next=0x%08x, head=%2u\n",
#else
    fprintf(file, "| list=\"%s\", size=%lu\n", name, size);
    for(Block* curr = (Block*)head; curr; curr = curr->getNext())
    {
        fprintf(file,
                "| | node=0x%016lx, data=0x%016lx, size=%lu, next=0x%016lx, head=%2lu\n",
#endif
                (uint64)curr,
                (uint64)curr->getData(),
                (uint64)curr->getSize(),
                (uint64)curr->getNext(),
                (uint64)curr->isHead());
    }
    fprintf(file, "|\n");
    return CNMEM_STATUS_SUCCESS;
}

cnmemStatus_t Manager::printMemoryState(FILE* file) const
{
    CNMEM_CHECK(mMutex.lock());
    uint64 streamCode = (uint64)mStream;
    uint64 usedMemory, freeMemory;
    CNMEM_CHECK_OR_UNLOCK(getUsedMemoryUnsafe(usedMemory), mMutex);
    CNMEM_CHECK_OR_UNLOCK(getFreeMemoryUnsafe(freeMemory), mMutex);

#ifdef CNMEM_BUILD_WITH_32_BIT_POINTERS
    fprintf(file,
            ">> [%s] device=%d, stream=0x%08x, used=%uB, free=%uB\n",
#else
    fprintf(file,
            ">> [%s] device=%d, stream=0x%016lx, used=%luB, free=%luB\n",
#endif
            mParent ? "child" : "root",
            mDevice,
            streamCode,
            usedMemory,
            freeMemory);
    CNMEM_CHECK_OR_UNLOCK(printListUnsafe(file, "used", mUsedBlocks), mMutex);
    CNMEM_CHECK_OR_UNLOCK(printListUnsafe(file, "free", mFreeBlocks), mMutex);
    fprintf(file, std::endl);
    CNMEM_CHECK(mMutex.unlock());

    if(mParent)
    {
        CNMEM_CHECK(mParent->printMemoryState(file));
    }
    return CNMEM_STATUS_SUCCESS;
}

cnmemStatus_t Manager::release(void* ptr)
{
    // Skip if ptr is NULL.
    if(ptr == NULL)
    {
        return CNMEM_STATUS_SUCCESS;
    }

    // Lock to make sure only one thread execute that fragment of code.
    CNMEM_CHECK(mMutex.lock());

    // Find the node in the list of used blocks.
    Block *curr = mUsedBlocks, *prev = NULL;
    for(; curr && curr->getData() != ptr; curr = curr->getNext())
    {
        prev = curr;
    }

    // Make sure we have found a node.
    if(curr == NULL)
    {
        CNMEM_CHECK(mMutex.unlock());
        return CNMEM_STATUS_INVALID_ARGUMENT;
    }

    // We have the node so release it.
    cnmemStatus_t result = releaseBlockUnsafe(curr, prev);
    CNMEM_CHECK(mMutex.unlock());
    return result;
}

cnmemStatus_t Manager::releaseAllUnsafe()
{
    // Destroy the children if any.
    for(uint64 i = 0; i < mChildren.size(); ++i)
    {
        Manager* child = mChildren[i];
        CNMEM_CHECK(child->releaseAllUnsafe());
        delete child;
    }
    mChildren.clear();

    // Destroy used blocks. It's a kind of panic mode to avoid leaks. NOTE: Do that only with roots!!!
    if(!mParent)
    {
        while(mUsedBlocks)
        {
            CNMEM_CHECK(releaseBlockUnsafe(mUsedBlocks, NULL));
        }
    }

    // We should be having only free blocks that are head blocks. Release those blocks.
    while(mFreeBlocks)
    {
        if(mParent)
        {
            CNMEM_CHECK(mParent->release(mFreeBlocks->getData()));
        }
        else if(mFreeBlocks->isHead())
        {
            void* data = mFreeBlocks->getData();
            CNMEM_DEBUG_INFO("cudaFree(%lu, 0x%016lx)\n", mFreeBlocks->getSize(), (uint64)data);
            CNMEM_CHECK_CUDA(cudaFree(data));
            CNMEM_DEBUG_INFO(">> success\n");
        }
        Block* block = mFreeBlocks;
        mFreeBlocks  = mFreeBlocks->getNext();
        delete block;
    }

    // We shouldn't have any used block left. Or, it means the user is causing memory leaks!
    return CNMEM_STATUS_SUCCESS;
}

cnmemStatus_t Manager::releaseBlockUnsafe(Block* curr, Block* prev)
{
    // The current node cannot be NULL!
    CNMEM_ASSERT(curr != NULL);

    // Change the connection of the node.
    if(prev)
    {
        prev->setNext(curr->getNext());
    }
    else
    {
        mUsedBlocks = curr->getNext();
    }

    // Find the location where this block should be added to the free list.
    prev        = NULL;
    Block* iter = mFreeBlocks;
    for(; iter && iter->getData() < curr->getData(); iter = iter->getNext())
    {
        prev = iter;
    }

    // Keep track of the successor of pred. We may lose track of it in the following "else".
    Block* next = prev ? prev->getNext() : mFreeBlocks;

    // We first check if we can merge the block with its predecessor in the list and curr can be merged.
    if(prev && prev->getData() + prev->getSize() == curr->getData() && !curr->isHead())
    {
        prev->setSize(prev->getSize() + curr->getSize());
        delete curr;
        curr = prev;
    }
    else if(prev)
    {
        prev->setNext(curr);
    }
    else
    {
        mFreeBlocks = curr;
    }

    // Check if we can merge curr and next. We can't merge over "cudaMalloc" boundaries.
    if(next && curr->getData() + curr->getSize() == next->getData() && !next->isHead())
    {
        curr->setSize(curr->getSize() + next->getSize());
        curr->setNext(next->getNext());
        delete next;
    }
    else
    {
        curr->setNext(next);
    }
    return CNMEM_STATUS_SUCCESS;
}

cnmemStatus_t Manager::reserve(uint64 size)
{
    CNMEM_CHECK(mMutex.lock());
    Block *curr, *prev;
    CNMEM_CHECK_OR_UNLOCK(allocateBlockUnsafe(curr, prev, size), mMutex);
    mSize = size;
    CNMEM_CHECK(mMutex.unlock());
    return CNMEM_STATUS_SUCCESS;
}

cnmemStatus_t Manager::stealUnsafe(void*& stolen, uint64 size)
{
    // If we cannot steal, don't even try.
    if(mFlags & CNMEM_FLAGS_CANNOT_STEAL)
    {
        stolen = NULL;
        return CNMEM_STATUS_INVALID_ARGUMENT;
    }

    // The stolen block.
    void*       data     = NULL;
    uint64 dataSize = 0;
    if(!mChildren.empty())
    {
        CNMEM_CHECK(stealBlockUnsafe(data, dataSize, size));
    }
    else if(mParent)
    {
        CNMEM_CHECK(mParent->stealBlockUnsafe(data, dataSize, size));
    }

    // Make sure we do have a block of memory or quit.
    if(!data)
    {
        stolen = NULL;
        return CNMEM_STATUS_OUT_OF_MEMORY;
    }

    // Push the block in the used list.
    mUsedBlocks = new Block((char*)data, dataSize, mUsedBlocks, true);
    if(!mUsedBlocks)
    {
        return CNMEM_STATUS_OUT_OF_MEMORY;
    }

    // Return the new pointer into memory.
    stolen = data;
    return CNMEM_STATUS_SUCCESS;
}

cnmemStatus_t Manager::stealBlockUnsafe(void*& data, uint64& dataSize, ::uint64 size)
{
    // No block found and no room to grow. Try to steal from a children (if we have any).
    data = NULL;
    for(uint64 i = 0; !data && i < mChildren.size(); ++i)
    {
        Manager* child = mChildren[i];
        if(child->giveBlockUnsafe(data, dataSize, size) == CNMEM_STATUS_SUCCESS)
        {
            break;
        }
    }

    // If no memory space found, simply return NULL. We have failed to allocate. Quit miserably.
    if(!data)
    {
        return CNMEM_STATUS_OUT_OF_MEMORY;
    }

    // We have got a node from a children. We need to update our "used" list before we can do
    // anything with it.
    Block *curr = mUsedBlocks, *prev = NULL;
    for(; curr; curr = curr->getNext())
    {
        if(curr->getData() <= data && data < curr->getData() + curr->getSize())
        {
            break;
        }
        prev = curr;
    }

    // Curr points to the node which contains that memory region.
    CNMEM_ASSERT(curr);

    // If it is exactly the same memory region, we are done!!!
    if(curr->getData() == data && curr->getSize() == dataSize)
    {
        return CNMEM_STATUS_SUCCESS;
    }

    // Track the blocks before and after curr.
    Block* next = curr->getNext();

    // We may have up to 3 blocks.
    uint64 sizeBefore = (uint64)((char*)data - curr->getData());
    uint64 sizeAfter  = (curr->getSize() - sizeBefore - dataSize);

    // The resulting block.
    Block* result = curr;

    // If we have no space between curr->getData and block->getData.
    if(sizeBefore == 0)
    {
        curr->setSize(dataSize);
    }
    else
    {
        curr->setSize(sizeBefore);
        Block* block = new Block((char*)data, dataSize, next, false);
        if(!block)
        {
            return CNMEM_STATUS_OUT_OF_MEMORY;
        }
        curr->setNext(block);
        curr     = block;
        data     = (char*)data + dataSize;
        dataSize = sizeAfter;
        result   = block;
    }

    // We have space at the end so we may need to add a new node.
    if(sizeAfter > 0)
    {
        Block* block = new Block(curr->getData() + curr->getSize(), sizeAfter, next, false);
        if(!block)
        {
            return CNMEM_STATUS_OUT_OF_MEMORY;
        }
        curr->setNext(block);
        curr = block;
    }
    return CNMEM_STATUS_SUCCESS;
}

class Context
{
    /// Use a magic number to specify that the context is valid.
    enum
    {
        CTX_VALID = 0x1f5632a3
    };

    /// The reference counting mechanism.
    int mRefCount;
    /// The mutex to increase/decrease the reference counter. TODO: Use atomics.
    Mutex mMutex;
    /// The memory managers.
    std::vector<Manager> mManagers;
    /// The global context.
    static Context* sCtx;
    /// Use a magic number to specify that the context was created.
    static int sCtxCheck;

public:
    /// Ctor.
    Context() : mRefCount(1) { mMutex.initialize(); }
    /// Dtor.
    ~Context();
    /// Get the managers.
    inline std::vector<Manager>& getManagers() { return mManagers; }
    /// Get a single manager associated with a device.
    inline Manager& getManager(int i) { return mManagers[i]; }

    /// Create the global context.
    static cnmemStatus_t create();
    /// Check that the context was created.
    static inline bool check() { return sCtxCheck == CTX_VALID && sCtx; }
    /// Get the global context.
    static Context* get();
    /// Retain.
    static cnmemStatus_t retain();
    /// Release.
    static cnmemStatus_t release();
};

Context* Context::sCtx;
int      Context::sCtxCheck;

Context::~Context()
{
    int oldDevice;
    cudaGetDevice(&oldDevice);
    for(uint64 i = 0; i < mManagers.size(); ++i)
    {
        if(mManagers[i].getDevice() != -1)
        { // Skip invalid managers.
            cudaSetDevice(mManagers[i].getDevice());
            mManagers[i].releaseAllUnsafe();
        }
    }
    mManagers.clear();
    mMutex.finalize();
    cudaSetDevice(oldDevice);
}

cnmemStatus_t Context::create()
{
    sCtx      = new Context;
    sCtxCheck = CTX_VALID;
    return CNMEM_STATUS_SUCCESS;
}

Context* Context::get()
{
    CNMEM_ASSERT(Context::check());
    return Context::sCtx;
}

cnmemStatus_t Context::retain()
{
    CNMEM_CHECK(sCtx->mMutex.lock());
    sCtx->mRefCount++;
    CNMEM_CHECK(sCtx->mMutex.unlock());
    return CNMEM_STATUS_SUCCESS;
}

cnmemStatus_t Context::release()
{
    CNMEM_CHECK(sCtx->mMutex.lock());
    int refCount = --sCtx->mRefCount;
    CNMEM_CHECK(sCtx->mMutex.unlock());

    if(refCount == 0)
    { // Kill the context.
        delete sCtx;
        Context::sCtx      = NULL;
        Context::sCtxCheck = 0;
    }
    return CNMEM_STATUS_SUCCESS;
}
}

KOKKOS_NET_API_EXTERNC cnmemStatus cnmemInit(const int numDevices, const cnmemDevice* devices, unsigned flags)
{
    // Make sure we have at least one device declared.
    CNMEM_CHECK_TRUE(numDevices > 0, CNMEM_STATUS_INVALID_ARGUMENT);

    // Find the largest ID of the device.
    int maxDevice = 0;
    for(int i = 0; i < numDevices; ++i)
    {
        if(devices[i].device > maxDevice)
        {
            maxDevice = devices[i].device;
        }
    }

    // Create the global context.
    cnmem::Context::create();
    cnmem::Context* ctx = cnmem::Context::get();

    // Allocate enough managers.
    CNMEM_CHECK_TRUE(maxDevice >= 0, CNMEM_STATUS_INVALID_ARGUMENT);
    std::vector<cnmem::Manager>& managers = ctx->getManagers();
    managers.resize(maxDevice + 1);

    // Create a root manager for each device and create the children.
    int oldDevice;
    CNMEM_CHECK_CUDA(cudaGetDevice(&oldDevice));
    for(int i = 0; i < numDevices; ++i)
    {
        CNMEM_CHECK_CUDA(cudaSetDevice(devices[i].device));
        uint64    size = devices[i].size;
        cudaDeviceProp props;
        CNMEM_CHECK_CUDA(cudaGetDeviceProperties(&props, devices[i].device));
        if(size == 0)
        {
            size = props.totalGlobalMem / 2;
        }
        CNMEM_CHECK_TRUE(size > 0 && size < props.totalGlobalMem, CNMEM_STATUS_INVALID_ARGUMENT);

        cnmem::Manager& manager = ctx->getManager(devices[i].device);
        manager.setDevice(devices[i].device);
        manager.setFlags(flags);

        size = cnmem::ceilInt(size, CNMEM_GRANULARITY);
        CNMEM_CHECK(manager.reserve(size));

        for(int j = 0; j < devices[i].numStreams; ++j)
        {
            cnmem::Manager* child = new cnmem::Manager;
            child->setParent(&manager);
            child->setDevice(devices[i].device);
            child->setStream(devices[i].streams[j]);
            child->setFlags(flags & ~CNMEM_FLAGS_CANNOT_GROW);
            if(devices[i].streamSizes && devices[i].streamSizes[j] > 0)
            {
                // https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#sequential-but-misaligned-access-pattern
                // Align stream blocks so stream base addresses are alligned to CNMEM_GRANULARITY
                devices[i].streamSizes[j] = cnmem::ceilInt(devices[i].streamSizes[j], CNMEM_GRANULARITY);
                CNMEM_CHECK(child->reserve(devices[i].streamSizes[j]));
            }
            CNMEM_CHECK(manager.addChild(child));
        }
    }
    CNMEM_CHECK_CUDA(cudaSetDevice(oldDevice));
    return CNMEM_STATUS_SUCCESS;
}

KOKKOS_NET_API_EXTERNC cnmemStatus cnmemFinalize()
{
    CNMEM_CHECK_TRUE(cnmem::Context::check(), CNMEM_STATUS_NOT_INITIALIZED);
    return cnmem::Context::release();
}

KOKKOS_NET_API_EXTERNC cnmemStatus cnmemRetain()
{
    CNMEM_CHECK_TRUE(cnmem::Context::check(), CNMEM_STATUS_NOT_INITIALIZED);
    return cnmem::Context::retain();
}

KOKKOS_NET_API_EXTERNC cnmemStatus cnmemRelease()
{
    CNMEM_CHECK_TRUE(cnmem::Context::check(), CNMEM_STATUS_NOT_INITIALIZED);
    return cnmem::Context::release();
}

KOKKOS_NET_API_EXTERNC cnmemStatus cnmemRegisterStream(cudaStream_t stream)
{
    CNMEM_CHECK_TRUE(cnmem::Context::check(), CNMEM_STATUS_NOT_INITIALIZED);
    CNMEM_CHECK_TRUE(stream, CNMEM_STATUS_INVALID_ARGUMENT);

    int device;
    CNMEM_CHECK_CUDA(cudaGetDevice(&device));

    cnmem::Manager& root  = cnmem::Context::get()->getManager(device);
    cnmem::Manager* child = new cnmem::Manager;
    child->setParent(&root);
    child->setDevice(device);
    child->setStream(stream);
    child->setFlags(root.getFlags() & ~CNMEM_FLAGS_CANNOT_GROW);
    root.addChild(child);

    return CNMEM_STATUS_SUCCESS;
}

KOKKOS_NET_API_EXTERNC cnmemStatus cnmemMalloc(void** ptr, uint64 size, cudaStream_t stream)
{
    CNMEM_CHECK_TRUE(cnmem::Context::check(), CNMEM_STATUS_NOT_INITIALIZED);
    if(!ptr && !size)
    {
        return CNMEM_STATUS_SUCCESS;
    }
    else if(!size)
    {
        ptr[0] = NULL;
        return CNMEM_STATUS_SUCCESS;
    }
    CNMEM_CHECK_TRUE(ptr, CNMEM_STATUS_INVALID_ARGUMENT);

    int device;
    CNMEM_CHECK_CUDA(cudaGetDevice(&device));

    cnmem::Manager& root    = cnmem::Context::get()->getManager(device);
    cnmem::Manager* manager = &root;
    if(stream)
    {
        CNMEM_CHECK(root.getChildFromStream(manager, stream));
    }
    CNMEM_ASSERT(manager);

    size               = cnmem::ceilInt(size, CNMEM_GRANULARITY);
    cnmemStatus result = manager->allocate(ptr[0], size);

    // We failed to allocate but there might still be a buffer available in another manager. Try to
    // steal it.
    if(result == CNMEM_STATUS_OUT_OF_MEMORY)
    {
        // Try to acquire locks on all the children.
        uint64 numChildren;
        CNMEM_CHECK(root.getNumChildren(numChildren));
        std::vector<const cnmem::Mutex*> mutexes(numChildren);

        uint64 numLocked = 0;
        for(uint64 i = 0; i < numChildren; ++i, ++numLocked)
        {
            cnmem::Manager* child;
            CNMEM_CHECK(root.getChild(child, i));
            mutexes[numLocked] = child->getMutex();
            if(mutexes[numLocked]->lock() != CNMEM_STATUS_SUCCESS)
            {
                break;
            }
        }

        // One lock failed, quit. Reduce the damage as much as possible, though.
        if(numLocked != numChildren)
        {
            for(uint64 i = 0; i < numLocked; ++i)
            {
                cnmemStatus lockStatus = mutexes[i]->unlock();
            }
            return CNMEM_STATUS_UNKNOWN_ERROR;
        }

        // Grab the lock on the root, first.
        const cnmem::Mutex* rootMutex = root.getMutex();
        CNMEM_CHECK(rootMutex->lock());

        // We acquired all the lock so we try to steal a node from another child.
        if(numLocked == mutexes.size())
        {
            result = manager->stealUnsafe(ptr[0], size);
        }
        for(uint64 i = 0; i < numLocked; ++i)
        {
            cnmemStatus lockStatus = mutexes[i]->unlock();
            if(lockStatus != CNMEM_STATUS_SUCCESS)
            {
                // Starting from now we are panicking!!! One lock failed to be released, we try
                // we others. We could also give up because we are already screwed. I don't know
                // what's best! Comment are welcome.
                result = lockStatus;
            }
        }
        CNMEM_CHECK(rootMutex->unlock());
    }
    return result;
}

KOKKOS_NET_API_EXTERNC cnmemStatus cnmemFree(void* ptr, cudaStream_t stream)
{
    CNMEM_CHECK_TRUE(cnmem::Context::check(), CNMEM_STATUS_NOT_INITIALIZED);
    if(ptr == NULL)
    {
        return CNMEM_STATUS_SUCCESS;
    }

    int device;
    CNMEM_CHECK_CUDA(cudaGetDevice(&device));

    cnmem::Manager& root    = cnmem::Context::get()->getManager(device);
    cnmem::Manager* manager = &root;
    if(stream)
    {
        CNMEM_CHECK(root.getChildFromStream(manager, stream));
    }
    CNMEM_ASSERT(manager);
    return manager->release(ptr);
}

KOKKOS_NET_API_EXTERNC cnmemStatus cnmemMemGetInfo(uint64* freeMem, uint64* totalMem, cudaStream_t stream)
{
    CNMEM_CHECK_TRUE(cnmem::Context::check(), CNMEM_STATUS_NOT_INITIALIZED);
    CNMEM_CHECK_TRUE(totalMem && freeMem, CNMEM_STATUS_INVALID_ARGUMENT);

    int device;
    CNMEM_CHECK_CUDA(cudaGetDevice(&device));
    cnmem::Manager& root    = cnmem::Context::get()->getManager(device);
    cnmem::Manager* manager = &root;
    if(stream)
    {
        CNMEM_CHECK(root.getChildFromStream(manager, stream));
    }
    CNMEM_ASSERT(manager);

    const cnmem::Mutex* mutex = manager->getMutex();
    CNMEM_CHECK(mutex->lock());
    CNMEM_CHECK_OR_UNLOCK(manager->getFreeMemoryUnsafe(*freeMem), *mutex);
    uint64 usedMem;
    CNMEM_CHECK_OR_UNLOCK(manager->getUsedMemoryUnsafe(usedMem), *mutex);
    CNMEM_CHECK(mutex->unlock());
    totalMem[0] = usedMem + freeMem[0];
    return CNMEM_STATUS_SUCCESS;
}

KOKKOS_NET_API_EXTERNC cnmemStatus cnmemPrintMemoryState(FILE* file, cudaStream_t stream)
{
    CNMEM_CHECK_TRUE(cnmem::Context::check(), CNMEM_STATUS_NOT_INITIALIZED);

    int device;
    CNMEM_CHECK_CUDA(cudaGetDevice(&device));
    cnmem::Manager& root    = cnmem::Context::get()->getManager(device);
    cnmem::Manager* manager = &root;
    if(stream)
    {
        CNMEM_CHECK(root.getChildFromStream(manager, stream));
    }
    CNMEM_ASSERT(manager);
    return manager->printMemoryState(file);
}
