
#include <Types.hpp>

#include <runtime.Kokkos/ViewTypes.hpp>
#include <runtime.Kokkos/Extensions.hpp>

void cudaDeviceProperties()
{
    cudaDeviceProp props;

    cudaError_t cudaError = cudaGetDeviceProperties(&props, 0);

    std::cout << "name=" << props.name << std::endl;
    // std::cout << "uuid=" << props.uuid << std::endl;
    std::cout << "luid=" << props.luid << std::endl;
    std::cout << "luidDeviceNodeMask=" << props.luidDeviceNodeMask << std::endl;
    std::cout << "totalGlobalMem=" << props.totalGlobalMem << std::endl;
    std::cout << "sharedMemPerBlock=" << props.sharedMemPerBlock << std::endl;
    std::cout << "regsPerBlock=" << props.regsPerBlock << std::endl;
    std::cout << "warpSize=" << props.warpSize << std::endl;
    std::cout << "memPitch=" << props.memPitch << std::endl;
    std::cout << "maxThreadsPerBlock=" << props.maxThreadsPerBlock << std::endl;
    std::cout << "maxThreadsDim[0]=" << props.maxThreadsDim[0] << std::endl;
    std::cout << "maxThreadsDim[1]=" << props.maxThreadsDim[1] << std::endl;
    std::cout << "maxThreadsDim[2]=" << props.maxThreadsDim[2] << std::endl;
    std::cout << "maxGridSize[0]=" << props.maxGridSize[0] << std::endl;
    std::cout << "maxGridSize[1]=" << props.maxGridSize[1] << std::endl;
    std::cout << "maxGridSize[2]=" << props.maxGridSize[2] << std::endl;
    std::cout << "clockRate=" << props.clockRate << std::endl;
    std::cout << "totalConstMem=" << props.totalConstMem << std::endl;
    std::cout << "major=" << props.major << std::endl;
    std::cout << "minor=" << props.minor << std::endl;
    std::cout << "textureAlignment=" << props.textureAlignment << std::endl;
    std::cout << "texturePitchAlignment=" << props.texturePitchAlignment << std::endl;
    std::cout << "deviceOverlap=" << props.deviceOverlap << std::endl;
    std::cout << "multiProcessorCount=" << props.multiProcessorCount << std::endl;
    std::cout << "kernelExecTimeoutEnabled=" << props.kernelExecTimeoutEnabled << std::endl;
    std::cout << "integrated=" << props.integrated << std::endl;
    std::cout << "canMapHostMemory=" << props.canMapHostMemory << std::endl;
    std::cout << "computeMode=" << props.computeMode << std::endl;
    std::cout << "maxTexture1D=" << props.maxTexture1D << std::endl;
    std::cout << "maxTexture1DMipmap=" << props.maxTexture1DMipmap << std::endl;
    std::cout << "maxTexture1DLinear=" << props.maxTexture1DLinear << std::endl;
    // std::cout << "maxTexture2D[2]=" << props.maxTexture2D[2] << std::endl;
    // std::cout << "maxTexture2DMipmap[2]=" << props.maxTexture2DMipmap[2] << std::endl;
    // std::cout << "maxTexture2DLinear[3]=" << props.maxTexture2DLinear[3] << std::endl;
    // std::cout << "maxTexture2DGather[2]=" << props.maxTexture2DGather[2] << std::endl;
    // std::cout << "maxTexture3D[3]=" << props.maxTexture3D[3] << std::endl;
    // std::cout << "maxTexture3DAlt[3]=" << props.maxTexture3DAlt[3] << std::endl;
    std::cout << "maxTextureCubemap=" << props.maxTextureCubemap << std::endl;
    // std::cout << "maxTexture1DLayered[2]=" << props.maxTexture1DLayered[2] << std::endl;
    // std::cout << "maxTexture2DLayered[3]=" << props.maxTexture2DLayered[3] << std::endl;
    // std::cout << "maxTextureCubemapLayered[2]=" << props.maxTextureCubemapLayered[2] << std::endl;
    std::cout << "maxSurface1D=" << props.maxSurface1D << std::endl;
    // std::cout << "maxSurface2D[2]=" << props.maxSurface2D[2] << std::endl;
    // std::cout << "maxSurface3D[3]=" << props.maxSurface3D[3] << std::endl;
    // std::cout << "maxSurface1DLayered[2]=" << props.maxSurface1DLayered[2] << std::endl;
    // std::cout << "maxSurface2DLayered[3]=" << props.maxSurface2DLayered[3] << std::endl;
    std::cout << "maxSurfaceCubemap=" << props.maxSurfaceCubemap << std::endl;
    // std::cout << "maxSurfaceCubemapLayered[2]=" << props.maxSurfaceCubemapLayered[2] << std::endl;
    std::cout << "surfaceAlignment=" << props.surfaceAlignment << std::endl;
    std::cout << "concurrentKernels=" << props.concurrentKernels << std::endl;
    std::cout << "ECCEnabled=" << props.ECCEnabled << std::endl;
    std::cout << "pciBusID=" << props.pciBusID << std::endl;
    std::cout << "pciDeviceID=" << props.pciDeviceID << std::endl;
    std::cout << "pciDomainID=" << props.pciDomainID << std::endl;
    std::cout << "tccDriver=" << props.tccDriver << std::endl;
    std::cout << "asyncEngineCount=" << props.asyncEngineCount << std::endl;
    std::cout << "unifiedAddressing=" << props.unifiedAddressing << std::endl;
    std::cout << "memoryClockRate=" << props.memoryClockRate << std::endl;
    std::cout << "memoryBusWidth=" << props.memoryBusWidth << std::endl;
    std::cout << "l2CacheSize=" << props.l2CacheSize << std::endl;
    std::cout << "persistingL2CacheMaxSize=" << props.persistingL2CacheMaxSize << std::endl;
    std::cout << "maxThreadsPerMultiProcessor=" << props.maxThreadsPerMultiProcessor << std::endl;
    std::cout << "streamPrioritiesSupported=" << props.streamPrioritiesSupported << std::endl;
    std::cout << "globalL1CacheSupported=" << props.globalL1CacheSupported << std::endl;
    std::cout << "localL1CacheSupported=" << props.localL1CacheSupported << std::endl;
    std::cout << "sharedMemPerMultiprocessor=" << props.sharedMemPerMultiprocessor << std::endl;
    std::cout << "regsPerMultiprocessor=" << props.regsPerMultiprocessor << std::endl;
    std::cout << "managedMemory=" << props.managedMemory << std::endl;
    std::cout << "isMultiGpuBoard=" << props.isMultiGpuBoard << std::endl;
    std::cout << "multiGpuBoardGroupID=" << props.multiGpuBoardGroupID << std::endl;
    std::cout << "hostNativeAtomicSupported=" << props.hostNativeAtomicSupported << std::endl;
    std::cout << "singleToDoublePrecisionPerfRatio=" << props.singleToDoublePrecisionPerfRatio << std::endl;
    std::cout << "pageableMemoryAccess=" << props.pageableMemoryAccess << std::endl;
    std::cout << "concurrentManagedAccess=" << props.concurrentManagedAccess << std::endl;
    std::cout << "computePreemptionSupported=" << props.computePreemptionSupported << std::endl;
    std::cout << "canUseHostPointerForRegisteredMem=" << props.canUseHostPointerForRegisteredMem << std::endl;
    std::cout << "cooperativeLaunch=" << props.cooperativeLaunch << std::endl;
    std::cout << "cooperativeMultiDeviceLaunch=" << props.cooperativeMultiDeviceLaunch << std::endl;
    std::cout << "sharedMemPerBlockOptin=" << props.sharedMemPerBlockOptin << std::endl;
    std::cout << "pageableMemoryAccessUsesHostPageTables=" << props.pageableMemoryAccessUsesHostPageTables << std::endl;
    std::cout << "directManagedMemAccessFromHost=" << props.directManagedMemAccessFromHost << std::endl;
    std::cout << "maxBlocksPerMultiProcessor=" << props.maxBlocksPerMultiProcessor << std::endl;
    std::cout << "accessPolicyMaxWindowSize=" << props.accessPolicyMaxWindowSize << std::endl;
    std::cout << "reservedSharedMemPerBlock=" << props.reservedSharedMemPerBlock << std::endl;

    CUdevice cuDevice;
    CHECK(cuDeviceGet(&cuDevice, 0));

    int attributeVal = 0;

    cuDeviceGetAttribute(&attributeVal, CU_DEVICE_ATTRIBUTE_VIRTUAL_ADDRESS_MANAGEMENT_SUPPORTED, cuDevice);

    std::cout << "CU_DEVICE_ATTRIBUTE_VIRTUAL_ADDRESS_MANAGEMENT_SUPPORTED=" << attributeVal << std::endl;
}
