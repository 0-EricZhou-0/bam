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
 * bam_buf_t -- Device-side BamBuffer descriptor for GPU kernels.
 *
 * Passed to bam_read_pages/bam_write_pages as a device pointer.
 * Contains the physical IO addresses of data pages plus a PRP list
 * pool for multi-page (>2) transfers.
 *
 * The PRP list pool is a set of pre-allocated DMA-mapped pages, each
 * capable of holding page_size/8 PRP entries. A lock-bit allocator
 * (same pattern as get_cid/put_cid in nvm_parallel_queue.h) distributes
 * slots to GPU threads on demand.
 */
struct bam_buf_t {
    uint64_t*      ioaddrs;          /* Physical addresses of data pages       */
    uint8_t*       base_addr;        /* GPU virtual address of buffer data     */
    uint32_t       page_size;        /* Controller page size in bytes          */
    uint32_t       n_pages;          /* Total number of data pages             */

    /* PRP list pool for multi-page transfers */
    uint64_t*      prp_list_ioaddrs; /* Physical addr of each PRP list page    */
    uint8_t*       prp_list_base;    /* GPU vaddr base of PRP list DMA region  */
    padded_struct* prp_slots;        /* Lock-bit allocator array [pool_size]   */
    uint32_t       prp_pool_size;    /* Number of PRP list slots (0=disabled)  */
    uint32_t       prp_page_size;    /* Size of each PRP list page (=page_size)*/

    simt::atomic<uint32_t, simt::thread_scope_device> prp_ticket;
};


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
 * BamBuffer -- GPU-accessible DMA buffer for use with bam I/O functions.
 *
 * Wraps a DMA allocation on a specific CUDA device and copies the physical
 * IO addresses (ioaddrs) to device memory so that GPU threads can use them
 * with bam_read/bam_write (single-page) or bam_read_pages/bam_write_pages
 * (multi-page).
 *
 * When prp_pool_size > 0, the buffer also allocates a pool of DMA-mapped
 * PRP list pages for multi-page (>2) NVMe transfers. Each PRP list page
 * holds page_size/8 entries, supporting transfers up to
 * (page_size/8 + 1) pages per command.
 *
 * Host-side setup:
 *   Controller ctrl("/dev/libnvm0", ns_id, cuda_device, queue_depth, num_queues);
 *   BamBuffer buf(ctrl, 16*1024*1024, 256);  // 16MB data, 256 PRP list slots
 *
 * Device-side usage (inside CUDA kernel):
 *   // Single-page read (raw API):
 *   bam_read(qp, lba, n_blocks, buf.d_buf->ioaddrs[page_idx], 0);
 *
 *   // Multi-page read (8 consecutive pages):
 *   bam_read_pages(qp, start_lba, buf.d_buf, start_page, 8);
 */
struct BamBuffer {
    DmaPtr    dma;          /* DMA mapping (shared_ptr, manages lifetime)      */
    uint64_t* d_ioaddrs;    /* Device-accessible copy of dma->ioaddrs[]        */
    void*     vaddr;        /* GPU virtual address of the buffer data           */
    size_t    n_pages;      /* Number of controller-page-sized pages            */
    size_t    page_size;    /* Controller page size in bytes                    */

    /* PRP list pool (only allocated when prp_pool_size > 0) */
    DmaPtr       prp_list_dma;          /* DMA mapping for PRP list pages          */
    BufferPtr    prp_list_ioaddrs_buf;  /* Device copy of PRP list ioaddrs         */
    BufferPtr    prp_slots_buf;         /* Device copy of allocator lock-bit array  */
    uint32_t     prp_pool_size_;        /* Number of PRP list slots                */

    /* Device-side descriptor (always allocated) */
    BufferPtr    d_buf_mem;             /* Device memory backing the bam_buf_t     */
    bam_buf_t*   d_buf;                /* Device pointer (pass to kernel args)     */

    /*
     * Construct a BamBuffer on a specific CUDA device.
     *
     * @ctrl           Pointer to the libnvm controller handle.
     * @total_size     Total buffer size in bytes (will be page-aligned).
     * @cudaDevice     CUDA device ordinal for the allocation.
     * @prp_pool_size  Number of PRP list slots for multi-page I/O (0=disabled).
     */
    inline BamBuffer(const nvm_ctrl_t* ctrl, size_t total_size, int cudaDevice,
                     uint32_t prp_pool_size = 0)
        : dma(createDma(ctrl, total_size, cudaDevice))
        , d_ioaddrs(nullptr)
        , vaddr(dma->vaddr)
        , n_pages(dma->n_ioaddrs)
        , page_size(dma->page_size)
        , prp_pool_size_(prp_pool_size)
        , d_buf(nullptr)
    {
        /* Copy data page ioaddrs to device memory */
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

        /* Allocate PRP list pool if requested */
        uint64_t* d_prp_list_ioaddrs = nullptr;
        uint8_t*  d_prp_list_base = nullptr;
        padded_struct* d_prp_slots = nullptr;

        if (prp_pool_size_ > 0)
        {
            /* DMA-mapped GPU memory for PRP list pages */
            prp_list_dma = createDma(ctrl, page_size * prp_pool_size_, cudaDevice);
            d_prp_list_base = (uint8_t*)prp_list_dma->vaddr;

            /* Copy PRP list page ioaddrs to device */
            prp_list_ioaddrs_buf = createBuffer(
                prp_list_dma->n_ioaddrs * sizeof(uint64_t), cudaDevice);
            d_prp_list_ioaddrs = (uint64_t*)prp_list_ioaddrs_buf.get();
            err = cudaMemcpy(d_prp_list_ioaddrs, prp_list_dma->ioaddrs,
                             prp_list_dma->n_ioaddrs * sizeof(uint64_t),
                             cudaMemcpyHostToDevice);
            if (err != cudaSuccess)
            {
                throw std::runtime_error(
                    std::string("BamBuffer: failed to copy PRP list ioaddrs: ")
                    + cudaGetErrorString(err));
            }

            /* Allocator lock-bit array (zeroed = all UNLOCKED) */
            prp_slots_buf = createBuffer(
                prp_pool_size_ * sizeof(padded_struct), cudaDevice);
            d_prp_slots = (padded_struct*)prp_slots_buf.get();
        }

        /* Build and upload bam_buf_t device descriptor */
        d_buf_mem = createBuffer(sizeof(bam_buf_t), cudaDevice);
        d_buf = (bam_buf_t*)d_buf_mem.get();

        bam_buf_t host_desc;
        memset(&host_desc, 0, sizeof(host_desc));
        host_desc.ioaddrs          = d_ioaddrs;
        host_desc.base_addr        = (uint8_t*)vaddr;
        host_desc.page_size        = (uint32_t)page_size;
        host_desc.n_pages          = (uint32_t)n_pages;
        host_desc.prp_list_ioaddrs = d_prp_list_ioaddrs;
        host_desc.prp_list_base    = d_prp_list_base;
        host_desc.prp_slots        = d_prp_slots;
        host_desc.prp_pool_size    = prp_pool_size_;
        host_desc.prp_page_size    = (uint32_t)page_size;

        err = cudaMemcpy(d_buf, &host_desc, sizeof(bam_buf_t), cudaMemcpyHostToDevice);
        if (err != cudaSuccess)
        {
            throw std::runtime_error(
                std::string("BamBuffer: failed to upload device descriptor: ")
                + cudaGetErrorString(err));
        }
    }

    /*
     * Convenience constructor: extract ctrl handle and CUDA device
     * from a Controller object.
     */
    inline BamBuffer(Controller& ctrl, size_t total_size, uint32_t prp_pool_size = 0);

    inline ~BamBuffer()
    {
        if (d_ioaddrs)
            cudaFree(d_ioaddrs);
    }

    /* Returns the GPU virtual address where data can be read/written. */
    void* data() const { return vaddr; }

    /* Non-copyable (owns raw GPU allocations). */
    BamBuffer(const BamBuffer&) = delete;
    BamBuffer& operator=(const BamBuffer&) = delete;
};

#endif
