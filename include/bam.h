#ifndef __BAM_H__
#define __BAM_H__

/*
 * bam.h -- Top-level include for the BaM GPU-initiated NVMe storage library.
 *
 * BaM enables CUDA GPU threads to issue NVMe read/write commands directly,
 * bypassing the CPU data path. This header pulls in everything needed for
 * both host-side setup and device-side I/O.
 *
 * Host-side setup:
 *   #include <bam.h>
 *
 *   // Open NVMe controller and create I/O queue pairs on GPU 0
 *   Controller ctrl("/dev/libnvm0", 1, 0, 1024, 128);
 *
 *   // Allocate a DMA-mapped GPU buffer with PRP list pool for multi-page I/O
 *   BamBuffer buf(ctrl, 16 * 1024 * 1024, 256);  // 16MB data, 256 PRP slots
 *
 * Device-side I/O (inside CUDA kernel, buf.d_buf passed as kernel arg):
 *   // Each warp picks a queue pair round-robin
 *   uint32_t qid = (threadIdx.x + blockIdx.x * blockDim.x) / 32 % n_qps;
 *   QueuePair* qp = d_qps + qid;
 *
 *   // Single-page read (raw PRP API):
 *   bam_read(qp, lba, n_blocks, d_buf->ioaddrs[page_idx]);
 *
 *   // Multi-page read (8 consecutive pages in one NVMe command):
 *   bam_read_pages(qp, start_lba, d_buf, start_page, 8);
 *
 *   // Asynchronous single-page read:
 *   uint16_t cid;
 *   bam_read_async(qp, lba, n_blocks, d_buf->ioaddrs[page_idx], 0, &cid);
 *   // ... do other work ...
 *   bam_complete(qp, cid);
 *
 * Headers included:
 *   ctrl.h    -- Controller: host-side NVMe controller and queue pair setup
 *   buffer.h  -- BamBuffer + bam_buf_t: DMA-mapped GPU buffer with PRP pool
 *   bam_io.h  -- bam_read, bam_write, bam_read_pages, etc.: device-side I/O
 */

#include "ctrl.h"
#include "buffer.h"
#include "bam_io.h"

#endif /* __BAM_H__ */
