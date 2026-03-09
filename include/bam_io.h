#ifndef __BAM_IO_H__
#define __BAM_IO_H__

/*
 * bam_io.h -- Device-side NVMe I/O API for GPU-initiated storage access.
 *
 * Provides inline __device__ functions that GPU threads call to issue NVMe
 * read and write commands directly from CUDA kernels. Each function operates
 * on a single QueuePair (SQ/CQ pair) and uses the lock-free parallel queue
 * operations from nvm_parallel_queue.h.
 *
 * Synchronous API (blocks the calling GPU thread until I/O completes):
 *   bam_read()    -- Read LBAs from NVMe into a GPU buffer.
 *   bam_write()   -- Write LBAs from a GPU buffer to NVMe.
 *   bam_access()  -- Generic I/O with configurable opcode.
 *
 * Asynchronous API (submit I/O, do other work, then poll for completion):
 *   bam_read_async()  -- Submit a read, return immediately with a CID.
 *   bam_write_async() -- Submit a write, return immediately with a CID.
 *   bam_complete()    -- Wait for an async command to complete and release CID.
 *
 * Usage example (synchronous):
 *   // Inside a CUDA kernel:
 *   uint32_t queue = (tid / 32) % ctrl->n_qps;
 *   QueuePair* qp = ctrl->d_qps + queue;
 *   bam_read(qp, lba, n_blocks, buf.d_ioaddrs[page_idx]);
 *
 * Usage example (asynchronous):
 *   uint16_t cid;
 *   bam_read_async(qp, lba, n_blocks, prp1, 0, &cid);
 *   // ... do other work while I/O is in flight ...
 *   bam_complete(qp, cid);
 *
 * Buffer setup: Use BamBuffer (from buffer.h) to allocate DMA-mapped GPU
 * memory. Pass buf.d_ioaddrs[page_idx] as prp1 to specify the physical
 * address of the target GPU memory page.
 *
 * Multi-page API (uses bam_buf_t device descriptor from BamBuffer::d_buf):
 *   bam_read_pages()   -- Read N consecutive pages in one NVMe command.
 *   bam_write_pages()  -- Write N consecutive pages in one NVMe command.
 *   bam_access_pages() -- Generic multi-page I/O with configurable opcode.
 *
 * Multi-page transfers use PRP lists for >2 pages, allocated from a
 * lock-free pool inside bam_buf_t. Requires BamBuffer created with
 * prp_pool_size > 0.
 */

#include "nvm_types.h"
#include "nvm_cmd.h"
#include "nvm_parallel_queue.h"
#include "queue.h"
#include "buffer.h"


/*
 * bam_access -- Issue a synchronous NVMe read or write from a GPU thread.
 *
 * Builds an NVMe command, enqueues it on the submission queue, polls the
 * completion queue until the command finishes, and releases the command ID.
 * The calling thread blocks until the I/O is complete.
 *
 * NVMe command flow:
 *   get_cid -> nvm_cmd_header -> nvm_cmd_data_ptr -> nvm_cmd_rw_blks
 *   -> sq_enqueue -> cq_poll -> cq_dequeue -> put_cid
 *
 * @qp        Pointer to the QueuePair to use (must be in GPU-accessible memory).
 * @lba       Starting logical block address on the NVMe namespace.
 * @n_blocks  Number of LBAs to transfer.
 * @prp1      Physical address of the first data page (from BamBuffer::d_ioaddrs).
 * @prp2      Physical address of the second data page, or PRP list pointer.
 *            Pass 0 for single-page transfers.
 * @opcode    NVMe I/O opcode: NVM_IO_READ (0x02) or NVM_IO_WRITE (0x01).
 */
inline __device__
void bam_access(QueuePair* qp, const uint64_t lba, const uint64_t n_blocks,
                const uint64_t prp1, const uint64_t prp2, const uint8_t opcode)
{
    nvm_cmd_t cmd;
    uint16_t cid = get_cid(&(qp->sq));

    nvm_cmd_header(&cmd, cid, opcode, qp->nvmNamespace);
    nvm_cmd_data_ptr(&cmd, prp1, prp2);
    nvm_cmd_rw_blks(&cmd, lba, n_blocks);

    sq_enqueue(&qp->sq, &cmd);

    uint32_t cq_pos = cq_poll(&qp->cq, cid);
    cq_dequeue(&qp->cq, cq_pos, &qp->sq);

    put_cid(&qp->sq, cid);
}


/*
 * bam_read -- Synchronous NVMe read from a GPU thread.
 *
 * Reads @n_blocks logical blocks starting at @lba into the GPU buffer
 * at physical address @prp1. Blocks the calling thread until data is
 * available in GPU memory.
 *
 * @qp        Pointer to the QueuePair to use.
 * @lba       Starting logical block address.
 * @n_blocks  Number of LBAs to read.
 * @prp1      Physical address of the destination GPU page.
 * @prp2      Second PRP entry (0 for single-page reads).
 */
inline __device__
void bam_read(QueuePair* qp, const uint64_t lba, const uint64_t n_blocks,
              const uint64_t prp1, const uint64_t prp2 = 0)
{
    bam_access(qp, lba, n_blocks, prp1, prp2, NVM_IO_READ);
}


/*
 * bam_write -- Synchronous NVMe write from a GPU thread.
 *
 * Writes @n_blocks logical blocks starting at @lba from the GPU buffer
 * at physical address @prp1. Blocks the calling thread until the write
 * is acknowledged by the NVMe controller.
 *
 * @qp        Pointer to the QueuePair to use.
 * @lba       Starting logical block address.
 * @n_blocks  Number of LBAs to write.
 * @prp1      Physical address of the source GPU page.
 * @prp2      Second PRP entry (0 for single-page writes).
 */
inline __device__
void bam_write(QueuePair* qp, const uint64_t lba, const uint64_t n_blocks,
               const uint64_t prp1, const uint64_t prp2 = 0)
{
    bam_access(qp, lba, n_blocks, prp1, prp2, NVM_IO_WRITE);
}


/*
 * bam_submit -- Submit an NVMe command asynchronously (internal helper).
 *
 * Builds and enqueues an NVMe command but does NOT wait for completion.
 * The caller receives the command ID via @out_cid and must later call
 * bam_complete() to wait for the I/O and release the CID.
 *
 * @qp        Pointer to the QueuePair to use.
 * @lba       Starting logical block address.
 * @n_blocks  Number of LBAs to transfer.
 * @prp1      Physical address of the first data page.
 * @prp2      Physical address of the second data page (or 0).
 * @opcode    NVMe I/O opcode.
 * @out_cid   [out] Receives the allocated command ID for later completion.
 */
inline __device__
void bam_submit(QueuePair* qp, const uint64_t lba, const uint64_t n_blocks,
                const uint64_t prp1, const uint64_t prp2,
                const uint8_t opcode, uint16_t* out_cid)
{
    nvm_cmd_t cmd;
    uint16_t cid = get_cid(&(qp->sq));

    nvm_cmd_header(&cmd, cid, opcode, qp->nvmNamespace);
    nvm_cmd_data_ptr(&cmd, prp1, prp2);
    nvm_cmd_rw_blks(&cmd, lba, n_blocks);

    sq_enqueue(&qp->sq, &cmd);

    *out_cid = cid;
}


/*
 * bam_read_async -- Submit an asynchronous NVMe read.
 *
 * Enqueues a read command and returns immediately. The caller must later
 * call bam_complete() with the returned CID to wait for the data and
 * release the command slot.
 *
 * @qp        Pointer to the QueuePair to use.
 * @lba       Starting logical block address.
 * @n_blocks  Number of LBAs to read.
 * @prp1      Physical address of the destination GPU page.
 * @prp2      Second PRP entry (0 for single-page reads).
 * @out_cid   [out] Receives the command ID for bam_complete().
 */
inline __device__
void bam_read_async(QueuePair* qp, const uint64_t lba, const uint64_t n_blocks,
                    const uint64_t prp1, const uint64_t prp2, uint16_t* out_cid)
{
    bam_submit(qp, lba, n_blocks, prp1, prp2, NVM_IO_READ, out_cid);
}


/*
 * bam_write_async -- Submit an asynchronous NVMe write.
 *
 * Enqueues a write command and returns immediately. The caller must later
 * call bam_complete() with the returned CID to wait for the write
 * acknowledgment and release the command slot.
 *
 * @qp        Pointer to the QueuePair to use.
 * @lba       Starting logical block address.
 * @n_blocks  Number of LBAs to write.
 * @prp1      Physical address of the source GPU page.
 * @prp2      Second PRP entry (0 for single-page writes).
 * @out_cid   [out] Receives the command ID for bam_complete().
 */
inline __device__
void bam_write_async(QueuePair* qp, const uint64_t lba, const uint64_t n_blocks,
                     const uint64_t prp1, const uint64_t prp2, uint16_t* out_cid)
{
    bam_submit(qp, lba, n_blocks, prp1, prp2, NVM_IO_WRITE, out_cid);
}


/*
 * bam_complete -- Wait for an async command to complete and release the CID.
 *
 * Polls the completion queue for the command identified by @cid, dequeues
 * the completion entry, and releases the command ID back to the pool.
 * Must be called exactly once for each bam_read_async()/bam_write_async().
 *
 * @qp   Pointer to the QueuePair used for the original submission.
 * @cid  The command ID returned by bam_read_async() or bam_write_async().
 */
inline __device__
void bam_complete(QueuePair* qp, const uint16_t cid)
{
    uint32_t cq_pos = cq_poll(&qp->cq, cid);
    cq_dequeue(&qp->cq, cq_pos, &qp->sq);
    put_cid(&qp->sq, cid);
}


/* =========================================================================
 * Multi-page I/O API
 *
 * These functions read/write multiple consecutive controller pages in a
 * single NVMe command. They use the bam_buf_t device descriptor (obtained
 * from BamBuffer::d_buf) which contains the data page IO addresses and a
 * PRP list pool for transfers spanning >2 pages.
 *
 * PRP layout per NVMe spec:
 *   1 page:   PRP1 = page addr, PRP2 = 0
 *   2 pages:  PRP1 = page 0,    PRP2 = page 1
 *   >2 pages: PRP1 = page 0,    PRP2 = PRP list pointer
 *             PRP list contains addresses of pages 1..N-1
 * ========================================================================= */


/*
 * prp_alloc -- Acquire a PRP list slot from the pool.
 *
 * Uses the same lock-bit + ticket pattern as get_cid() in
 * nvm_parallel_queue.h. Spins until a slot is available.
 *
 * @buf  Device pointer to bam_buf_t (must have prp_pool_size > 0).
 * @return  Slot index [0, prp_pool_size).
 */
inline __device__
uint32_t prp_alloc(bam_buf_t* buf)
{
    uint32_t id;
    bool not_found = true;
    do {
        id = buf->prp_ticket.fetch_add(1, simt::memory_order_relaxed)
             % buf->prp_pool_size;
        uint32_t old = buf->prp_slots[id].val.fetch_or(LOCKED,
                       simt::memory_order_acquire);
        not_found = (old == LOCKED);
    } while (not_found);
    return id;
}


/*
 * prp_free -- Release a PRP list slot back to the pool.
 *
 * @buf   Device pointer to bam_buf_t.
 * @slot  Slot index returned by prp_alloc().
 */
inline __device__
void prp_free(bam_buf_t* buf, uint32_t slot)
{
    buf->prp_slots[slot].val.store(UNLOCKED, simt::memory_order_release);
}


/*
 * bam_access_pages -- Issue a multi-page NVMe read or write from a GPU thread.
 *
 * Reads or writes @n_pages consecutive controller pages starting at
 * @start_page in the buffer, targeting NVMe LBAs starting at @start_lba.
 * Automatically handles PRP list construction:
 *   - 1 page:   PRP1 only
 *   - 2 pages:  PRP1 + PRP2 (inline, no pool needed)
 *   - >2 pages: PRP1 + PRP list (allocated from pool, requires prp_pool_size > 0)
 *
 * The number of LBAs transferred is n_pages * (page_size / block_size).
 *
 * @qp          Pointer to the QueuePair to use.
 * @start_lba   Starting logical block address on the NVMe namespace.
 * @buf         Device pointer to bam_buf_t descriptor.
 * @start_page  Index of the first page in the buffer.
 * @n_pages     Number of consecutive pages to transfer.
 * @opcode      NVMe I/O opcode: NVM_IO_READ (0x02) or NVM_IO_WRITE (0x01).
 */
inline __device__
void bam_access_pages(QueuePair* qp, uint64_t start_lba,
                      bam_buf_t* buf, uint32_t start_page, uint32_t n_pages,
                      uint8_t opcode)
{
    nvm_cmd_t cmd;
    uint16_t cid = get_cid(&(qp->sq));
    uint32_t prp_slot = 0xFFFFFFFF;

    uint64_t prp1 = buf->ioaddrs[start_page];
    uint64_t prp2 = 0;

    if (n_pages == 2)
    {
        prp2 = buf->ioaddrs[start_page + 1];
    }
    else if (n_pages > 2)
    {
        /* Allocate a PRP list page and fill it with addresses of pages 1..N-1 */
        prp_slot = prp_alloc(buf);
        uint64_t* list = (uint64_t*)(buf->prp_list_base
                         + (uint64_t)prp_slot * buf->prp_page_size);
        for (uint32_t i = 0; i < n_pages - 1; i++)
            list[i] = buf->ioaddrs[start_page + 1 + i];
        prp2 = buf->prp_list_ioaddrs[prp_slot];
    }

    /* n_blocks = n_pages * blocks_per_page */
    uint32_t blocks_per_page = qp->pageSize / qp->block_size;
    uint64_t n_blocks = (uint64_t)n_pages * blocks_per_page;

    nvm_cmd_header(&cmd, cid, opcode, qp->nvmNamespace);
    nvm_cmd_data_ptr(&cmd, prp1, prp2);
    nvm_cmd_rw_blks(&cmd, start_lba, n_blocks);

    sq_enqueue(&qp->sq, &cmd);
    uint32_t cq_pos = cq_poll(&qp->cq, cid);
    cq_dequeue(&qp->cq, cq_pos, &qp->sq);
    put_cid(&qp->sq, cid);

    if (prp_slot != 0xFFFFFFFF)
        prp_free(buf, prp_slot);
}


/*
 * bam_read_pages -- Read N consecutive pages from NVMe into a BamBuffer.
 *
 * @qp          Pointer to the QueuePair to use.
 * @start_lba   Starting logical block address.
 * @buf         Device pointer to bam_buf_t descriptor.
 * @start_page  Index of the first destination page in the buffer.
 * @n_pages     Number of consecutive pages to read.
 */
inline __device__
void bam_read_pages(QueuePair* qp, uint64_t start_lba,
                    bam_buf_t* buf, uint32_t start_page, uint32_t n_pages)
{
    bam_access_pages(qp, start_lba, buf, start_page, n_pages, NVM_IO_READ);
}


/*
 * bam_write_pages -- Write N consecutive pages from a BamBuffer to NVMe.
 *
 * @qp          Pointer to the QueuePair to use.
 * @start_lba   Starting logical block address.
 * @buf         Device pointer to bam_buf_t descriptor.
 * @start_page  Index of the first source page in the buffer.
 * @n_pages     Number of consecutive pages to write.
 */
inline __device__
void bam_write_pages(QueuePair* qp, uint64_t start_lba,
                     bam_buf_t* buf, uint32_t start_page, uint32_t n_pages)
{
    bam_access_pages(qp, start_lba, buf, start_page, n_pages, NVM_IO_WRITE);
}


#endif /* __BAM_IO_H__ */
