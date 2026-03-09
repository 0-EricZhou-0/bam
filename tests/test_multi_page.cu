/*
 * test_multi_page.cu -- Multi-page read/write correctness and bandwidth test.
 *
 * Tests bam_write_pages() and bam_read_pages() with 2, 4, and 8 pages per
 * NVMe command. This exercises the PRP list construction path (>2 pages).
 * Supports multi-device RAID 0 striping.
 *
 * Usage:
 *   nvm-test-multi-page --device /dev/libnvm0 /dev/libnvm1 --gpu 0 --pages 1024
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
 * write_pages_kernel -- Each thread writes pages_per_cmd consecutive pages.
 *
 * Thread i handles pages [i*pages_per_cmd, (i+1)*pages_per_cmd).
 * Stripe controller is selected by the first page of the chunk.
 */
__global__
void write_pages_kernel(DeviceArray da, bam_buf_t* buf,
                        uint32_t n_chunks, uint32_t pages_per_cmd,
                        uint32_t blocks_per_page)
{
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_chunks) return;

    uint32_t start_page = tid * pages_per_cmd;
    uint32_t warp_id = tid / 32;

    /* For multi-page commands, stripe by chunk (all pages in one cmd go to same controller) */
    uint32_t ctrl_idx = (start_page / pages_per_cmd) % da.n_ctrls;
    QueuePair* qps = da.all_qps[ctrl_idx];
    uint16_t n_qps = da.all_n_qps[ctrl_idx];
    QueuePair* qp = qps + (warp_id % n_qps);

    uint64_t lba = (uint64_t)((start_page / pages_per_cmd) / da.n_ctrls)
                   * pages_per_cmd * blocks_per_page;

    bam_write_pages(qp, lba, buf, start_page, pages_per_cmd);
}


__global__
void read_pages_kernel(DeviceArray da, bam_buf_t* buf,
                       uint32_t n_chunks, uint32_t pages_per_cmd,
                       uint32_t blocks_per_page)
{
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_chunks) return;

    uint32_t start_page = tid * pages_per_cmd;
    uint32_t warp_id = tid / 32;

    uint32_t ctrl_idx = (start_page / pages_per_cmd) % da.n_ctrls;
    QueuePair* qps = da.all_qps[ctrl_idx];
    uint16_t n_qps = da.all_n_qps[ctrl_idx];
    QueuePair* qp = qps + (warp_id % n_qps);

    uint64_t lba = (uint64_t)((start_page / pages_per_cmd) / da.n_ctrls)
                   * pages_per_cmd * blocks_per_page;

    bam_read_pages(qp, lba, buf, start_page, pages_per_cmd);
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
 * Run one test case for a given pages_per_cmd
 * ------------------------------------------------------------------------- */

static int run_test(const char* label, uint32_t pages_per_cmd,
                    DeviceArray& da, BamBuffer& buf,
                    uint32_t n_pages, uint32_t page_size,
                    uint32_t blocks_per_page)
{
    /* Round down n_pages to be divisible by pages_per_cmd */
    uint32_t effective_pages = (n_pages / pages_per_cmd) * pages_per_cmd;
    uint32_t n_chunks = effective_pages / pages_per_cmd;
    size_t total_bytes = (size_t)effective_pages * page_size;

    printf("  --- %s (%u pages/cmd, %u pages total, %.2f MB) ---\n",
           label, pages_per_cmd, effective_pages,
           (double)total_bytes / (1024.0 * 1024.0));

    uint32_t tpb = 256;

    /* Mismatch counter */
    uint64_t* d_mismatches;
    TEST_CUDA_CHECK(cudaMalloc((void**)&d_mismatches, sizeof(uint64_t)));

    /* CUDA events */
    cudaEvent_t t_start, t_end;
    TEST_CUDA_CHECK(cudaEventCreate(&t_start));
    TEST_CUDA_CHECK(cudaEventCreate(&t_end));

    /* Fill buffer (one thread per page) */
    uint32_t fill_blocks = (effective_pages + tpb - 1) / tpb;
    fill_buffer_kernel<<<fill_blocks, tpb>>>(
        (uint8_t*)buf.vaddr, effective_pages, page_size);
    TEST_CUDA_CHECK(cudaDeviceSynchronize());

    /* Write (one thread per chunk) */
    uint32_t write_blocks = (n_chunks + tpb - 1) / tpb;
    TEST_CUDA_CHECK(cudaEventRecord(t_start));
    write_pages_kernel<<<write_blocks, tpb>>>(
        da, buf.d_buf, n_chunks, pages_per_cmd, blocks_per_page);
    TEST_CUDA_CHECK(cudaEventRecord(t_end));
    TEST_CUDA_CHECK(cudaEventSynchronize(t_end));
    float write_ms;
    TEST_CUDA_CHECK(cudaEventElapsedTime(&write_ms, t_start, t_end));

    /* Clear buffer */
    TEST_CUDA_CHECK(cudaMemset(buf.vaddr, 0, total_bytes));

    /* Read back */
    TEST_CUDA_CHECK(cudaEventRecord(t_start));
    read_pages_kernel<<<write_blocks, tpb>>>(
        da, buf.d_buf, n_chunks, pages_per_cmd, blocks_per_page);
    TEST_CUDA_CHECK(cudaEventRecord(t_end));
    TEST_CUDA_CHECK(cudaEventSynchronize(t_end));
    float read_ms;
    TEST_CUDA_CHECK(cudaEventElapsedTime(&read_ms, t_start, t_end));

    /* Verify */
    TEST_CUDA_CHECK(cudaMemset(d_mismatches, 0, sizeof(uint64_t)));
    verify_kernel<<<fill_blocks, tpb>>>(
        (uint8_t*)buf.vaddr, effective_pages, page_size, d_mismatches);
    TEST_CUDA_CHECK(cudaDeviceSynchronize());

    uint64_t mismatches = 0;
    TEST_CUDA_CHECK(cudaMemcpy(&mismatches, d_mismatches, sizeof(uint64_t),
                               cudaMemcpyDeviceToHost));

    report_bw("Write bandwidth:", total_bytes, write_ms);
    report_bw("Read bandwidth:", total_bytes, read_ms);
    report_result("Data integrity:", mismatches, effective_pages);
    printf("\n");

    cudaFree(d_mismatches);
    cudaEventDestroy(t_start);
    cudaEventDestroy(t_end);

    return (mismatches == 0) ? 0 : 1;
}


/* -------------------------------------------------------------------------
 * Main
 * ------------------------------------------------------------------------- */

int main(int argc, char** argv)
{
    TestSettings settings = parse_args(argc, argv);

    printf("=== Multi-Page Read/Write Test ===\n");
    printf("  Devices:     ");
    for (size_t i = 0; i < settings.device_paths.size(); i++)
        printf("%s ", settings.device_paths[i]);
    printf("\n");
    printf("  GPU:         %u\n", settings.gpu_id);
    printf("  Queue depth: %lu\n", (unsigned long)settings.queue_depth);
    printf("  Num queues:  %lu per controller\n", (unsigned long)settings.num_queues);
    printf("  Pages:       %lu\n", (unsigned long)settings.num_pages);
    printf("  PRP pool:    %u slots\n", settings.prp_pool_size);
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

    /* Allocate BamBuffer with PRP pool for multi-page commands */
    BamBuffer buf(*ctrls[0], total_bytes, settings.prp_pool_size);

    /* Build device array for multi-controller striping */
    DeviceArray da = build_device_array(ctrls, settings.gpu_id);

    /* Run tests for different pages_per_cmd values */
    int failures = 0;
    uint32_t test_sizes[] = {2, 4, 8};

    for (uint32_t ppc : test_sizes)
    {
        if (n_pages < ppc)
        {
            printf("  Skipping %u pages/cmd test (need at least %u pages)\n\n", ppc, ppc);
            continue;
        }
        char label[64];
        snprintf(label, sizeof(label), "%u-page commands", ppc);
        failures += run_test(label, ppc, da, buf, n_pages, page_size, blocks_per_page);
    }

    /* Cleanup */
    free_device_array(da);
    for (auto* c : ctrls) delete c;

    printf("=== %s ===\n", failures ? "SOME TESTS FAILED" : "ALL TESTS PASSED");
    return failures;
}
