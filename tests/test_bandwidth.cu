/*
 * test_bandwidth.cu -- Bandwidth measurement test for BaM NVMe I/O.
 *
 * Measures sequential write, sequential read, and random read throughput.
 * Each phase uses many concurrent GPU threads issuing single-page I/O.
 * Supports multi-device RAID 0 striping.
 *
 * Usage:
 *   nvm-test-bandwidth --device /dev/libnvm0 /dev/libnvm1 --gpu 0 --pages 4096
 */

#include "test_common.h"


/* -------------------------------------------------------------------------
 * Kernels
 * ------------------------------------------------------------------------- */

__global__
void fill_buffer_kernel(uint8_t* base, uint32_t n_pages, uint32_t page_size)
{
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_pages) return;

    uint8_t* page = base + (uint64_t)tid * page_size;
    fill_pattern(page, tid, page_size);
}


/*
 * seq_write_kernel -- Sequential write: each thread writes its assigned page.
 */
__global__
void seq_write_kernel(DeviceArray da, bam_buf_t* buf,
                      uint32_t n_pages, uint32_t blocks_per_page)
{
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_pages) return;

    uint32_t warp_id = tid / 32;
    QueuePair* qp = get_qp(&da, tid, warp_id);
    uint64_t lba = stripe_lba(tid, da.n_ctrls, blocks_per_page);

    bam_write(qp, lba, blocks_per_page, buf->ioaddrs[tid]);
}


/*
 * seq_read_kernel -- Sequential read: each thread reads its assigned page.
 */
__global__
void seq_read_kernel(DeviceArray da, bam_buf_t* buf,
                     uint32_t n_pages, uint32_t blocks_per_page)
{
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_pages) return;

    uint32_t warp_id = tid / 32;
    QueuePair* qp = get_qp(&da, tid, warp_id);
    uint64_t lba = stripe_lba(tid, da.n_ctrls, blocks_per_page);

    bam_read(qp, lba, blocks_per_page, buf->ioaddrs[tid]);
}


/*
 * rand_read_kernel -- Random read: each thread reads a random page.
 *
 * Uses a simple LCG PRNG to generate random page indices within the
 * device capacity. The read data lands in the thread's assigned buffer page.
 */
__global__
void rand_read_kernel(DeviceArray da, bam_buf_t* buf,
                      uint32_t n_pages, uint32_t blocks_per_page,
                      uint64_t max_page_per_ctrl)
{
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_pages) return;

    /* LCG PRNG: state = tid * 6364136223846793005 + 1442695040888963407 */
    uint64_t state = (uint64_t)tid * 6364136223846793005ULL + 1442695040888963407ULL;
    uint32_t rand_page = (uint32_t)(state >> 16) % (uint32_t)max_page_per_ctrl;

    uint32_t warp_id = tid / 32;
    /* Stripe controller by tid (same as sequential for queue selection) */
    uint32_t ctrl_idx = tid % da.n_ctrls;
    QueuePair* qps = da.all_qps[ctrl_idx];
    uint16_t n_qps = da.all_n_qps[ctrl_idx];
    QueuePair* qp = qps + (warp_id % n_qps);

    uint64_t lba = (uint64_t)rand_page * blocks_per_page;

    /* Read into this thread's buffer page */
    bam_read(qp, lba, blocks_per_page, buf->ioaddrs[tid]);
}


__global__
void verify_kernel(uint8_t* base, uint32_t n_pages, uint32_t page_size,
                   uint64_t* d_mismatches)
{
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_pages) return;

    uint8_t* page = base + (uint64_t)tid * page_size;
    uint32_t m = verify_pattern(page, tid, page_size);
    if (m > 0)
        atomicAdd((unsigned long long*)d_mismatches, (unsigned long long)m);
}


/* -------------------------------------------------------------------------
 * Main
 * ------------------------------------------------------------------------- */

int main(int argc, char** argv)
{
    TestSettings settings = parse_args(argc, argv);

    /* Default to more pages for bandwidth test */
    if (settings.num_pages == 1024)
        settings.num_pages = 4096;

    printf("=== Bandwidth Test ===\n");
    printf("  Devices:     ");
    for (size_t i = 0; i < settings.device_paths.size(); i++)
        printf("%s ", settings.device_paths[i]);
    printf("\n");
    printf("  GPU:         %u\n", settings.gpu_id);
    printf("  Queue depth: %lu\n", (unsigned long)settings.queue_depth);
    printf("  Num queues:  %lu per controller\n", (unsigned long)settings.num_queues);
    printf("  Pages:       %lu\n", (unsigned long)settings.num_pages);
    printf("\n");

    TEST_CUDA_CHECK(cudaSetDevice(settings.gpu_id));

    /* Create one Controller per device */
    std::vector<Controller*> ctrls;
    for (size_t i = 0; i < settings.device_paths.size(); i++)
    {
        printf("  Opening %s ...\n", settings.device_paths[i]);
        ctrls.push_back(new Controller(
            settings.device_paths[i], settings.ns_id, settings.gpu_id,
            settings.queue_depth, settings.num_queues));
    }

    uint32_t page_size = ctrls[0]->page_size;
    uint32_t blk_size  = ctrls[0]->blk_size;
    uint32_t blocks_per_page = page_size / blk_size;
    uint32_t n_pages = (uint32_t)settings.num_pages;
    size_t   total_bytes = (size_t)n_pages * page_size;

    /* Compute max pages per controller (from namespace capacity) */
    uint64_t ns_blocks = ctrls[0]->ns.size;
    uint64_t max_page_per_ctrl = ns_blocks / blocks_per_page;

    printf("  Page size:          %u bytes\n", page_size);
    printf("  Block size:         %u bytes\n", blk_size);
    printf("  Total data:         %zu bytes (%.2f MB)\n", total_bytes,
           (double)total_bytes / (1024.0 * 1024.0));
    printf("  NS capacity:        %lu LBAs (%lu pages)\n",
           (unsigned long)ns_blocks, (unsigned long)max_page_per_ctrl);
    printf("\n");

    /* Allocate BamBuffer (no PRP pool, single-page I/O) */
    BamBuffer buf(*ctrls[0], total_bytes, 0);

    /* Build device array for multi-controller striping */
    DeviceArray da = build_device_array(ctrls, settings.gpu_id);

    /* CUDA events for timing */
    cudaEvent_t t_start, t_end;
    TEST_CUDA_CHECK(cudaEventCreate(&t_start));
    TEST_CUDA_CHECK(cudaEventCreate(&t_end));

    /* Mismatch counter */
    uint64_t* d_mismatches;
    TEST_CUDA_CHECK(cudaMalloc((void**)&d_mismatches, sizeof(uint64_t)));

    uint32_t tpb = 256;
    uint32_t num_blocks = (n_pages + tpb - 1) / tpb;

    /* -----------------------------------------------------------------
     * Phase 1: Sequential Write
     * ----------------------------------------------------------------- */
    printf("Phase 1: Sequential Write\n");

    /* Fill buffer with pattern first */
    fill_buffer_kernel<<<num_blocks, tpb>>>(
        (uint8_t*)buf.vaddr, n_pages, page_size);
    TEST_CUDA_CHECK(cudaDeviceSynchronize());

    TEST_CUDA_CHECK(cudaEventRecord(t_start));
    seq_write_kernel<<<num_blocks, tpb>>>(da, buf.d_buf, n_pages, blocks_per_page);
    TEST_CUDA_CHECK(cudaEventRecord(t_end));
    TEST_CUDA_CHECK(cudaEventSynchronize(t_end));
    float seq_write_ms;
    TEST_CUDA_CHECK(cudaEventElapsedTime(&seq_write_ms, t_start, t_end));
    report_bw("Sequential write:", total_bytes, seq_write_ms);

    /* -----------------------------------------------------------------
     * Phase 2: Sequential Read (+ correctness verification)
     * ----------------------------------------------------------------- */
    printf("Phase 2: Sequential Read\n");

    /* Clear buffer to prove data comes from NVMe */
    TEST_CUDA_CHECK(cudaMemset(buf.vaddr, 0, total_bytes));

    TEST_CUDA_CHECK(cudaEventRecord(t_start));
    seq_read_kernel<<<num_blocks, tpb>>>(da, buf.d_buf, n_pages, blocks_per_page);
    TEST_CUDA_CHECK(cudaEventRecord(t_end));
    TEST_CUDA_CHECK(cudaEventSynchronize(t_end));
    float seq_read_ms;
    TEST_CUDA_CHECK(cudaEventElapsedTime(&seq_read_ms, t_start, t_end));
    report_bw("Sequential read:", total_bytes, seq_read_ms);

    /* Verify what we read matches what we wrote */
    TEST_CUDA_CHECK(cudaMemset(d_mismatches, 0, sizeof(uint64_t)));
    verify_kernel<<<num_blocks, tpb>>>(
        (uint8_t*)buf.vaddr, n_pages, page_size, d_mismatches);
    TEST_CUDA_CHECK(cudaDeviceSynchronize());

    uint64_t mismatches = 0;
    TEST_CUDA_CHECK(cudaMemcpy(&mismatches, d_mismatches, sizeof(uint64_t),
                               cudaMemcpyDeviceToHost));
    report_result("Seq read integrity:", mismatches, n_pages);

    /* -----------------------------------------------------------------
     * Phase 3: Random Read
     * ----------------------------------------------------------------- */
    printf("Phase 3: Random Read\n");

    TEST_CUDA_CHECK(cudaEventRecord(t_start));
    rand_read_kernel<<<num_blocks, tpb>>>(
        da, buf.d_buf, n_pages, blocks_per_page, max_page_per_ctrl);
    TEST_CUDA_CHECK(cudaEventRecord(t_end));
    TEST_CUDA_CHECK(cudaEventSynchronize(t_end));
    float rand_read_ms;
    TEST_CUDA_CHECK(cudaEventElapsedTime(&rand_read_ms, t_start, t_end));
    report_bw("Random read:", total_bytes, rand_read_ms);

    printf("\n");

    /* Summary */
    printf("=== Bandwidth Summary ===\n");
    report_bw("Sequential write:", total_bytes, seq_write_ms);
    report_bw("Sequential read:", total_bytes, seq_read_ms);
    report_bw("Random read:", total_bytes, rand_read_ms);
    report_result("Data integrity:", mismatches, n_pages);
    printf("\n");

    /* Cleanup */
    cudaFree(d_mismatches);
    cudaEventDestroy(t_start);
    cudaEventDestroy(t_end);
    free_device_array(da);
    for (auto* c : ctrls) delete c;

    return (mismatches == 0) ? 0 : 1;
}
