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
 *   Controller ctrl("/dev/libnvm0", /*ns_id=*/ 1, /*cudaDevice=*/ 0,
 *                    /*queueDepth=*/ 1024, /*numQueues=*/ 128);
 *
 *   // Allocate a DMA-mapped GPU buffer (e.g. 16 pages)
 *   BamBuffer buf(ctrl, 16 * ctrl.page_size);
 *
 * Device-side I/O (inside CUDA kernel):
 *   // Each warp picks a queue pair round-robin
 *   uint32_t qid = (threadIdx.x + blockIdx.x * blockDim.x) / 32 % ctrl->n_qps;
 *   QueuePair* qp = ctrl->d_qps + qid;
 *
 *   // Synchronous read: thread blocks until data is in GPU memory
 *   bam_read(qp, lba, n_blocks, buf.d_ioaddrs[page_idx]);
 *
 *   // Asynchronous read: submit and complete separately
 *   uint16_t cid;
 *   bam_read_async(qp, lba, n_blocks, buf.d_ioaddrs[page_idx], 0, &cid);
 *   // ... do other work ...
 *   bam_complete(qp, cid);
 *
 * Headers included:
 *   ctrl.h    -- Controller: host-side NVMe controller and queue pair setup
 *   buffer.h  -- BamBuffer: DMA-mapped GPU buffer with device-accessible ioaddrs
 *   bam_io.h  -- bam_read, bam_write, bam_complete: device-side I/O functions
 */

#include "ctrl.h"
#include "buffer.h"
#include "bam_io.h"

#endif /* __BAM_H__ */
