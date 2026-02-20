#ifndef __BENCHMARK_QUEUEPAIR_H__
#define __BENCHMARK_QUEUEPAIR_H__

#include <algorithm>
#include <cstdint>
#include "buffer.h"
#include "cuda.h"
#include "nvm_types.h"
#include "nvm_util.h"
#include "nvm_error.h"
#include "nvm_admin.h"
#include <stdexcept>
#include <string>
#include <iostream>
#include <cmath>
#include "util.h"


/*
 * QueuePair -- Wraps an NVMe submission/completion queue pair for GPU access.
 *
 * Each QueuePair owns one SQ and one CQ whose memory resides in GPU device
 * memory (via createDma). The doorbell registers are mapped as device
 * pointers so GPU threads can ring them directly via MMIO writes.
 *
 * Additionally, each QueuePair allocates GPU-side auxiliary arrays needed
 * by the lock-free parallel queue operations in nvm_parallel_queue.h:
 *   - sq_tickets:   Ticket array for ordered SQ enqueue (one per SQ entry)
 *   - sq_tail_mark: Marks which SQ tail entries are ready for doorbell write
 *   - sq_cid:       Command ID allocation pool (65536 slots, one per NVMe CID)
 *   - cq_head_mark: Marks which CQ entries have been consumed
 *   - cq_pos_locks: Per-entry locks for CQ dequeue coordination
 */
struct QueuePair
{
    uint32_t            pageSize;           /* Controller memory page size          */
    uint32_t            block_size;         /* Namespace LBA data size in bytes     */
    uint32_t            block_size_log;     /* log2(block_size)                     */
    uint32_t            block_size_minus_1; /* block_size - 1 (for masking)         */
    uint32_t            nvmNamespace;       /* NVMe namespace ID                    */
    nvm_queue_t         sq;                 /* Submission queue descriptor          */
    nvm_queue_t         cq;                 /* Completion queue descriptor          */
    uint16_t            qp_id;              /* Queue pair identifier (1-based)      */
    DmaPtr              sq_mem;             /* DMA mapping for SQ memory            */
    DmaPtr              cq_mem;             /* DMA mapping for CQ memory            */
    DmaPtr              prp_mem;            /* DMA mapping for PRP lists (if used)  */
    BufferPtr           sq_tickets;         /* GPU buffer: SQ ticket array          */
    BufferPtr           sq_tail_mark;       /* GPU buffer: SQ tail mark array       */
    BufferPtr           sq_cid;             /* GPU buffer: CID allocation pool      */
    BufferPtr           cq_head_mark;       /* GPU buffer: CQ head mark array       */
    BufferPtr           cq_pos_locks;       /* GPU buffer: CQ position locks        */


/*
 * Max queue entries that fit in a single 64KB-aligned page.
 * SQ entries are 64 bytes each: 64KB / 64 = 1024 entries.
 * CQ entries are 16 bytes each: 64KB / 16 = 4096 entries.
 */
#define MAX_SQ_ENTRIES_64K  (64*1024/64)
#define MAX_CQ_ENTRIES_64K  (64*1024/16)

    /*
     * init_gpu_specific_struct -- Allocate GPU-side arrays for parallel queue ops.
     *
     * Creates the ticket, mark, CID, and lock arrays that the lock-free
     * GPU queue functions (get_cid, sq_enqueue, cq_poll, cq_dequeue) use
     * for multi-thread coordination. Also computes qs_minus_1 and qs_log2
     * for fast modular arithmetic in queue indexing.
     */
    inline void init_gpu_specific_struct(const uint32_t cudaDevice) {
        /* SQ parallel operation arrays */
        this->sq_tickets = createBuffer(this->sq.qs * sizeof(padded_struct), cudaDevice);
        this->sq_tail_mark = createBuffer(this->sq.qs * sizeof(padded_struct), cudaDevice);
        this->sq_cid = createBuffer(65536 * sizeof(padded_struct), cudaDevice);
        this->sq.tickets = (padded_struct*) this->sq_tickets.get();
        this->sq.tail_mark = (padded_struct*) this->sq_tail_mark.get();
        this->sq.cid = (padded_struct*) this->sq_cid.get();
        this->sq.qs_minus_1 = this->sq.qs - 1;
        this->sq.qs_log2 = (uint32_t) std::log2(this->sq.qs);

        /* CQ parallel operation arrays */
        this->cq_head_mark = createBuffer(this->cq.qs * sizeof(padded_struct), cudaDevice);
        this->cq.head_mark = (padded_struct*) this->cq_head_mark.get();
        this->cq.qs_minus_1 = this->cq.qs - 1;
        this->cq.qs_log2 = (uint32_t) std::log2(this->cq.qs);
        this->cq_pos_locks = createBuffer(this->cq.qs * sizeof(padded_struct), cudaDevice);
        this->cq.pos_locks = (padded_struct*) this->cq_pos_locks.get();
    }


    /*
     * QueuePair constructor -- Create an NVMe I/O queue pair on a GPU.
     *
     * Reads the controller's CAP register to determine the maximum queue
     * entries supported (MQES field, bits 15:0) and whether contiguous queue
     * memory is required (CQR bit 16). Allocates DMA-mapped GPU memory for
     * the SQ and CQ, creates them via admin commands, maps the doorbell
     * registers as CUDA device pointers, and initializes the GPU-side
     * parallel queue structures.
     *
     * @ctrl        libnvm controller handle (for CAP register and DMA mapping).
     * @cudaDevice  CUDA device ordinal for GPU memory allocation.
     * @ns          Namespace info (for LBA size and namespace ID).
     * @info        Controller info (for page size).
     * @aq_ref      Admin queue reference for issuing create-queue commands.
     * @qp_id       Queue pair ID (1-based; used as both SQ and CQ ID).
     * @queueDepth  Requested queue depth (clamped to hardware maximum).
     */
    inline QueuePair(const nvm_ctrl_t* ctrl, const uint32_t cudaDevice, const struct nvm_ns_info ns, const struct nvm_ctrl_info info, nvm_aq_ref& aq_ref, const uint16_t qp_id, const uint64_t queueDepth)
    {
        /*
         * Read the CAP register to determine queue sizing constraints.
         * MQES (Maximum Queue Entries Supported) is in bits 15:0 as a 0-based value.
         * CQR (Contiguous Queues Required) is bit 16; when set, queue memory
         * must be physically contiguous, limiting entries to what fits in one
         * 64KB-aligned page.
         */
        uint64_t cap = ((volatile uint64_t*) ctrl->mm_ptr)[0];
        bool cqr = (cap & 0x0000000000010000) == 0x0000000000010000;

        uint64_t mqes = (((volatile uint16_t*) ctrl->mm_ptr)[0] + 1);
        uint64_t sq_size = cqr ? std::min((uint64_t)MAX_SQ_ENTRIES_64K, mqes) : mqes;
        uint64_t cq_size = cqr ? std::min((uint64_t)MAX_CQ_ENTRIES_64K, mqes) : mqes;
        sq_size = std::min(queueDepth, sq_size);
        cq_size = std::min(queueDepth, cq_size);

        bool sq_need_prp = false;
        bool cq_need_prp = false;

        size_t sq_mem_size =  sq_size * sizeof(nvm_cmd_t) + sq_need_prp*(64*1024);
        size_t cq_mem_size =  cq_size * sizeof(nvm_cpl_t) + cq_need_prp*(64*1024);

        /* Allocate 64KB-aligned GPU memory for SQ and CQ via DMA mapping */
        this->sq_mem = createDma(ctrl, NVM_PAGE_ALIGN(sq_mem_size, 1UL << 16), cudaDevice);
        this->cq_mem = createDma(ctrl, NVM_PAGE_ALIGN(cq_mem_size, 1UL << 16), cudaDevice);

        /* Store namespace and block size info for use in GPU kernels */
        this->pageSize = info.page_size;
        this->block_size = ns.lba_data_size;
        this->block_size_minus_1 = ns.lba_data_size-1;
        this->block_size_log = std::log2(ns.lba_data_size);
        this->nvmNamespace = ns.ns_id;
        this->qp_id = qp_id;

        /* Build PRP list for CQ if queue spans multiple pages */
        if (cq_need_prp) {
            size_t iters = (size_t)ceil(((float)cq_size*sizeof(nvm_cpl_t))/((float)ctrl->page_size));
            uint64_t* cpu_vaddrs = (uint64_t*) malloc(64*1024);
            memset((void*)cpu_vaddrs, 0, 64*1024);
            for (size_t i = 0; i < iters; i++) {
                size_t page_64  = i/(64*1024);
                size_t page_4 = i%(64*1024/ctrl->page_size);
                cpu_vaddrs[i] = this->cq_mem.get()->ioaddrs[1 + page_64] + (page_4 * ctrl->page_size);
            }

            if (this->cq_mem.get()->vaddr) {
                cuda_err_chk(cudaMemcpy(this->cq_mem.get()->vaddr, cpu_vaddrs, 64*1024, cudaMemcpyHostToDevice));
            }

            this->cq_mem.get()->vaddr = (void*)((uint64_t)this->cq_mem.get()->vaddr + 64*1024);
            free(cpu_vaddrs);
        }

        /* Build PRP list for SQ if queue spans multiple pages */
        if (sq_need_prp) {
            size_t iters = (size_t)ceil(((float)sq_size*sizeof(nvm_cpl_t))/((float)ctrl->page_size));
            uint64_t* cpu_vaddrs = (uint64_t*) malloc(64*1024);
            memset((void*)cpu_vaddrs, 0, 64*1024);
            for (size_t i = 0; i < iters; i++) {
                size_t page_64  = i/(64*1024);
                size_t page_4 = i%(64*1024/ctrl->page_size);
                cpu_vaddrs[i] = this->sq_mem.get()->ioaddrs[1 + page_64] + (page_4 * ctrl->page_size);
            }

            if (this->sq_mem.get()->vaddr) {
                cuda_err_chk(cudaMemcpy(this->sq_mem.get()->vaddr, cpu_vaddrs, 64*1024, cudaMemcpyHostToDevice));
            }

            this->sq_mem.get()->vaddr = (void*)((uint64_t)this->sq_mem.get()->vaddr + 64*1024);
            free(cpu_vaddrs);
        }

        /* Create the CQ via admin command */
        int status = nvm_admin_cq_create(aq_ref, &this->cq, qp_id, this->cq_mem.get(), 0, cq_size, cq_need_prp);
        if (!nvm_ok(status))
        {
            throw std::runtime_error(std::string("Failed to create completion queue: ") + nvm_strerror(status));
        }

        /* Map CQ doorbell register as a CUDA device pointer for GPU MMIO writes */
        void* devicePtr = nullptr;
        cudaError_t err = cudaHostGetDevicePointer(&devicePtr, (void*) this->cq.db, 0);
        if (err != cudaSuccess)
        {
            throw std::runtime_error(std::string("Failed to get device pointer") + cudaGetErrorString(err));
        }
        this->cq.db = (volatile uint32_t*) devicePtr;

        /* Create the SQ via admin command, referencing the paired CQ */
        status = nvm_admin_sq_create(aq_ref, &this->sq, &this->cq, qp_id, this->sq_mem.get(), 0, sq_size, sq_need_prp);
        if (!nvm_ok(status))
        {
            throw std::runtime_error(std::string("Failed to create submission queue: ") + nvm_strerror(status));
        }

        /* Map SQ doorbell register as a CUDA device pointer for GPU MMIO writes */
        err = cudaHostGetDevicePointer(&devicePtr, (void*) this->sq.db, 0);
        if (err != cudaSuccess)
        {
            throw std::runtime_error(std::string("Failed to get device pointer") + cudaGetErrorString(err));
        }
        this->sq.db = (volatile uint32_t*) devicePtr;

        /* Allocate GPU-side parallel queue operation structures */
        init_gpu_specific_struct(cudaDevice);
    }

};
#endif
