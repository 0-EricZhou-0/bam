#ifndef __BENCHMARK_BUFFER_H__
#define __BENCHMARK_BUFFER_H__

#include <memory>
#include <cstddef>
#include <cstdint>
#include "cuda.h"
#include "nvm_types.h"
#include "nvm_dma.h"
#include "nvm_util.h"
#include "nvm_error.h"
#include <stdexcept>
#include <string>
#include <new>
#include <cstdlib>
#include <iostream>
#include "util.h"


/*
 * DmaPtr -- Shared pointer to an nvm_dma_t mapping.
 *
 * Created by createDma(). The custom deleter unmaps the DMA region and
 * frees the underlying memory (host or GPU) when the last reference is
 * released.
 */
typedef std::shared_ptr<nvm_dma_t> DmaPtr;

/*
 * BufferPtr -- Shared pointer to a raw GPU or host memory allocation.
 *
 * Created by createBuffer(). The custom deleter frees the allocation
 * via cudaFreeHost (host) or cudaFree (device).
 */
typedef std::shared_ptr<void> BufferPtr;


/* Forward declarations for createDma/createBuffer overloads. */
DmaPtr createDma(const nvm_ctrl_t* ctrl, size_t size);
DmaPtr createDma(const nvm_ctrl_t* ctrl, size_t size, int cudaDevice);
DmaPtr createDma(const nvm_ctrl_t* ctrl, size_t size, uint32_t adapter, uint32_t id);
DmaPtr createDma(const nvm_ctrl_t* ctrl, size_t size, int cudaDevice, uint32_t adapter, uint32_t id);
BufferPtr createBuffer(size_t size);
BufferPtr createBuffer(size_t size, int cudaDevice);

/* Forward-declare Controller so BamBuffer can reference it. */
struct Controller;


/*
 * getDeviceMemory -- Allocate GPU memory with 64KB alignment for DMA.
 *
 * Allocates @size + 64KB bytes via cudaMalloc, then aligns the returned
 * pointers up to a 64KB boundary. This alignment is required by NVMe
 * for PRP (Physical Region Page) entries.
 *
 * @device     CUDA device ordinal.
 * @bufferPtr  [out] Aligned GPU virtual address for the caller.
 * @devicePtr  [out] Aligned device pointer (from cudaPointerGetAttributes).
 * @size       Requested allocation size in bytes.
 * @origPtr    [out] Original unaligned pointer (needed for cudaFree).
 */
static void getDeviceMemory(int device, void*& bufferPtr, void*& devicePtr, size_t size, void*& origPtr)
{
    bufferPtr = nullptr;
    devicePtr = nullptr;

    cudaError_t err = cudaSetDevice(device);
    if (err != cudaSuccess)
    {
        throw std::runtime_error(std::string("Failed to set CUDA device: ") + cudaGetErrorString(err));
    }
    size += 64*1024;
    err = cudaMalloc(&bufferPtr, size);
    if (err != cudaSuccess)
    {
        throw std::runtime_error(std::string("Failed to allocate device memory: ") + cudaGetErrorString(err));
    }

    cudaPointerAttributes attrs;
    err = cudaPointerGetAttributes(&attrs, bufferPtr);
    if (err != cudaSuccess)
    {
        cudaFree(bufferPtr);
        throw std::runtime_error(std::string("Failed to get pointer attributes: ") + cudaGetErrorString(err));
    }

    origPtr = bufferPtr;
    devicePtr = (void*) ((((uint64_t)attrs.devicePointer) + (64*1024)) & 0xffffffffff0000);
    bufferPtr = (void*) ((((uint64_t)bufferPtr) + (64*1024))  & 0xffffffffff0000);
}

/*
 * getDeviceMemory2 -- Allocate GPU memory with 32-byte alignment.
 *
 * Used for smaller auxiliary structures (ticket arrays, CID pools, etc.)
 * that need alignment but not the full 64KB DMA alignment.
 *
 * @device     CUDA device ordinal.
 * @bufferPtr  [out] Aligned GPU virtual address.
 * @size       Requested allocation size in bytes.
 * @origPtr    [out] Original unaligned pointer (needed for cudaFree).
 */
static void getDeviceMemory2(int device, void*& bufferPtr, size_t size, void*& origPtr)
{
    bufferPtr = nullptr;
    size += 128;
    cudaError_t err = cudaSetDevice(device);
    if (err != cudaSuccess)
    {
        throw std::runtime_error(std::string("Failed to set CUDA device: ") + cudaGetErrorString(err));
    }
    err = cudaMalloc(&bufferPtr, size);
    if (err != cudaSuccess)
    {
        throw std::runtime_error(std::string("Failed to allocate device memory: ") + cudaGetErrorString(err));
    }

    err = cudaMemset(bufferPtr, 0, size);
    if (err != cudaSuccess)
    {
        cudaFree(bufferPtr);
        throw std::runtime_error(std::string("Failed to clear device memory: ") + cudaGetErrorString(err));
    }

    origPtr = bufferPtr;
    bufferPtr = (void*) ((((uint64_t)bufferPtr) + (128))  & 0xffffffffffffe0);
}

/* Convenience overload: allocate without returning devicePtr or origPtr. */
static void getDeviceMemory(int device, void*& bufferPtr, size_t size)
{
    void* notUsed = nullptr;
    getDeviceMemory(device, bufferPtr, notUsed, size, notUsed);
}


/*
 * createDma -- Allocate and DMA-map host memory (page-aligned via posix_memalign).
 *
 * The returned DmaPtr's deleter calls nvm_dma_unmap() and free().
 */
inline DmaPtr createDma(const nvm_ctrl_t* ctrl, size_t size)
{
    nvm_dma_t* dma = nullptr;
    void* buffer = nullptr;

    int err  = posix_memalign(&buffer, 4096, size);
    if (err) {
        throw std::runtime_error(std::string("Failed to allocate host memory: ") + std::to_string(err));
    }
    int status = nvm_dma_map_host(&dma, ctrl, buffer, size);
    if (!nvm_ok(status))
    {
        free(buffer);
        throw std::runtime_error(std::string("Failed to map host memory: ") + nvm_strerror(status));
    }

    return DmaPtr(dma, [buffer](nvm_dma_t* dma) {
        nvm_dma_unmap(dma);
        free(buffer);
    });
}


/*
 * createDma -- Allocate and DMA-map GPU device memory.
 *
 * Allocates 64KB-aligned GPU memory on @cudaDevice, maps it for NVMe DMA,
 * and zeroes the buffer. If @cudaDevice < 0, falls back to host allocation.
 *
 * The returned DmaPtr's deleter calls nvm_dma_unmap() and cudaFree().
 * The dma->vaddr field is set to the GPU virtual address.
 * The dma->ioaddrs[] array contains the physical IO addresses of each page.
 */
inline DmaPtr createDma(const nvm_ctrl_t* ctrl, size_t size, int cudaDevice)
{
    if (cudaDevice < 0)
    {
        return createDma(ctrl, size);
    }

    nvm_dma_t* dma = nullptr;
    void* bufferPtr = nullptr;
    void* devicePtr = nullptr;
    void* origPtr = nullptr;

    getDeviceMemory(cudaDevice, bufferPtr, devicePtr, size, origPtr);

    int status = nvm_dma_map_device(&dma, ctrl, bufferPtr, size);
    if (!nvm_ok(status))
    {
        throw std::runtime_error(std::string("Failed to map device memory: ") + nvm_strerror(status));
    }
    cudaError_t err = cudaMemset(bufferPtr, 0, size);
    if (err != cudaSuccess)
    {
        cudaFree(bufferPtr);
        throw std::runtime_error(std::string("Failed to clear device memory: ") + cudaGetErrorString(err));
    }
    dma->vaddr = bufferPtr;

    return DmaPtr(dma, [bufferPtr, origPtr](nvm_dma_t* dma) {
        nvm_dma_unmap(dma);
        cudaFree(origPtr);
    });
}


/*
 * createBuffer -- Allocate pinned host memory via cudaHostAlloc.
 *
 * The returned BufferPtr's deleter calls cudaFreeHost().
 */
inline BufferPtr createBuffer(size_t size)
{
    void* buffer = nullptr;

    cudaError_t err = cudaHostAlloc(&buffer, size, cudaHostAllocDefault);
    if (err != cudaSuccess)
    {
        throw std::runtime_error(std::string("Failed to allocate host memory: ") + cudaGetErrorString(err));
    }

    return BufferPtr(buffer, [](void* ptr) {
        cudaFreeHost(ptr);
    });
}


/*
 * createBuffer -- Allocate 32-byte-aligned GPU device memory.
 *
 * If @cudaDevice < 0, falls back to pinned host allocation.
 * The returned BufferPtr's deleter calls cudaFree().
 */
inline BufferPtr createBuffer(size_t size, int cudaDevice)
{
    if (cudaDevice < 0)
    {
        return createBuffer(size);
    }

    void* bufferPtr = nullptr;
    void* origPtr = nullptr;

    getDeviceMemory2(cudaDevice, bufferPtr, size, origPtr);

    return BufferPtr(bufferPtr, [origPtr](void* ptr) {
        __ignore(ptr);
        cudaFree(origPtr);
    });
}


/*
 * BamBuffer -- GPU-accessible DMA buffer for use with bam_read/bam_write.
 *
 * Wraps a DMA allocation on a specific CUDA device and copies the physical
 * IO addresses (ioaddrs) to device memory so that GPU threads can pass them
 * directly to bam_read() / bam_write() as prp1/prp2 arguments.
 *
 * Host-side setup:
 *   Controller ctrl("/dev/libnvm0", ns_id, cuda_device, queue_depth, num_queues);
 *   BamBuffer buf(ctrl, total_bytes);
 *
 * Device-side usage (inside CUDA kernel):
 *   uint64_t prp1 = buf.d_ioaddrs[page_index];
 *   bam_read(qp, lba, n_blocks, prp1, 0);
 *   // Data is now at ((uint8_t*)buf.vaddr) + page_index * buf.page_size
 */
struct BamBuffer {
    DmaPtr    dma;          /* DMA mapping (shared_ptr, manages lifetime)      */
    uint64_t* d_ioaddrs;    /* Device-accessible copy of dma->ioaddrs[]        */
    void*     vaddr;        /* GPU virtual address of the buffer data           */
    size_t    n_pages;      /* Number of controller-page-sized pages            */
    size_t    page_size;    /* Controller page size in bytes                    */

    /*
     * Construct a BamBuffer on a specific CUDA device.
     *
     * Allocates GPU memory, creates a DMA mapping, and copies the
     * IO address array to device memory for kernel access.
     *
     * @ctrl        Pointer to the libnvm controller handle.
     * @total_size  Total buffer size in bytes (will be page-aligned).
     * @cudaDevice  CUDA device ordinal for the allocation.
     */
    inline BamBuffer(const nvm_ctrl_t* ctrl, size_t total_size, int cudaDevice)
        : dma(createDma(ctrl, total_size, cudaDevice))
        , d_ioaddrs(nullptr)
        , vaddr(dma->vaddr)
        , n_pages(dma->n_ioaddrs)
        , page_size(dma->page_size)
    {
        size_t addrs_size = n_pages * sizeof(uint64_t);
        cudaError_t err = cudaMalloc((void**)&d_ioaddrs, addrs_size);
        if (err != cudaSuccess)
        {
            throw std::runtime_error(
                std::string("BamBuffer: failed to allocate ioaddrs on device: ")
                + cudaGetErrorString(err));
        }
        err = cudaMemcpy(d_ioaddrs, dma->ioaddrs, addrs_size, cudaMemcpyHostToDevice);
        if (err != cudaSuccess)
        {
            cudaFree(d_ioaddrs);
            throw std::runtime_error(
                std::string("BamBuffer: failed to copy ioaddrs to device: ")
                + cudaGetErrorString(err));
        }
    }

    /*
     * Convenience constructor: extract ctrl handle and CUDA device
     * from a Controller object.
     */
    inline BamBuffer(Controller& ctrl, size_t total_size);

    inline ~BamBuffer()
    {
        if (d_ioaddrs)
            cudaFree(d_ioaddrs);
    }

    /* Returns the GPU virtual address where data can be read/written. */
    void* data() const { return vaddr; }

    /* Non-copyable (d_ioaddrs is a raw allocation). */
    BamBuffer(const BamBuffer&) = delete;
    BamBuffer& operator=(const BamBuffer&) = delete;
};

#endif
