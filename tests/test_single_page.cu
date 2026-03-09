/*
 * test_single_page.cu -- Single-page read/write correctness and bandwidth test.
 *
 * Each GPU thread writes one page to NVMe via bam_write(), clears the GPU
 * buffer, reads it back via bam_read(), and verifies the data matches.
 * Supports multi-device RAID 0 striping.
 *
 * Usage:
 *   nvm-test-single-page --device /dev/libnvm0 /dev/libnvm1 --gpu 0 --pages 1024
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


__global__
void write_kernel(DeviceArray da, bam_buf_t* buf,
                  uint32_t n_pages, uint32_t blocks_per_page)
{
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_pages) return;

    uint32_t warp_id = tid / 32;
    QueuePair* qp = get_qp(&da, tid, warp_id);
    uint64_t lba = stripe_lba(tid, da.n_ctrls, blocks_per_page);

    bam_write(qp, lba, blocks_per_page, buf->ioaddrs[tid]);
}


__global__
void read_kernel(DeviceArray da, bam_buf_t* buf,
                 uint32_t n_pages, uint32_t blocks_per_page)
{
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_pages) return;

    uint32_t warp_id = tid / 32;
    QueuePair* qp = get_qp(&da, tid, warp_id);
    uint64_t lba = stripe_lba(tid, da.n_ctrls, blocks_per_page);

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

    printf("=== Single-Page Read/Write Test ===\n");
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

    printf("  Page size:   %u bytes\n", page_size);
    printf("  Block size:  %u bytes\n", blk_size);
    printf("  Total data:  %zu bytes (%.2f MB)\n\n", total_bytes,
           (double)total_bytes / (1024.0 * 1024.0));

    /* Allocate BamBuffer (no PRP pool needed for single-page I/O) */
    BamBuffer buf(*ctrls[0], total_bytes, 0);

    /* Build device array for multi-controller striping */
    DeviceArray da = build_device_array(ctrls, settings.gpu_id);

    /* Mismatch counter on device */
    uint64_t* d_mismatches;
    TEST_CUDA_CHECK(cudaMalloc((void**)&d_mismatches, sizeof(uint64_t)));

    /* CUDA events for timing */
    cudaEvent_t t_start, t_end;
    TEST_CUDA_CHECK(cudaEventCreate(&t_start));
    TEST_CUDA_CHECK(cudaEventCreate(&t_end));

    /* Kernel launch config */
    uint32_t threads_per_block = 256;
    uint32_t num_blocks = (n_pages + threads_per_block - 1) / threads_per_block;

    /* Step 1: Fill buffer with known pattern */
    fill_buffer_kernel<<<num_blocks, threads_per_block>>>(
        (uint8_t*)buf.vaddr, n_pages, page_size);
    TEST_CUDA_CHECK(cudaDeviceSynchronize());

    /* Step 2: Write to NVMe (timed) */
    TEST_CUDA_CHECK(cudaEventRecord(t_start));
    write_kernel<<<num_blocks, threads_per_block>>>(
        da, buf.d_buf, n_pages, blocks_per_page);
    TEST_CUDA_CHECK(cudaEventRecord(t_end));
    TEST_CUDA_CHECK(cudaEventSynchronize(t_end));
    float write_ms;
    TEST_CUDA_CHECK(cudaEventElapsedTime(&write_ms, t_start, t_end));

    /* Step 3: Clear GPU buffer */
    TEST_CUDA_CHECK(cudaMemset(buf.vaddr, 0, total_bytes));

    /* Step 4: Read back from NVMe (timed) */
    TEST_CUDA_CHECK(cudaEventRecord(t_start));
    read_kernel<<<num_blocks, threads_per_block>>>(
        da, buf.d_buf, n_pages, blocks_per_page);
    TEST_CUDA_CHECK(cudaEventRecord(t_end));
    TEST_CUDA_CHECK(cudaEventSynchronize(t_end));
    float read_ms;
    TEST_CUDA_CHECK(cudaEventElapsedTime(&read_ms, t_start, t_end));

    /* Step 5: Verify data */
    TEST_CUDA_CHECK(cudaMemset(d_mismatches, 0, sizeof(uint64_t)));
    verify_kernel<<<num_blocks, threads_per_block>>>(
        (uint8_t*)buf.vaddr, n_pages, page_size, d_mismatches);
    TEST_CUDA_CHECK(cudaDeviceSynchronize());

    uint64_t mismatches = 0;
    TEST_CUDA_CHECK(cudaMemcpy(&mismatches, d_mismatches, sizeof(uint64_t),
                               cudaMemcpyDeviceToHost));

    /* Report results */
    printf("Results:\n");
    report_bw("Write bandwidth:", total_bytes, write_ms);
    report_bw("Read bandwidth:", total_bytes, read_ms);
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
