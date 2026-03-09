#ifndef __TEST_COMMON_H__
#define __TEST_COMMON_H__

/*
 * test_common.h -- Shared helpers for BaM NVMe I/O tests.
 *
 * Provides argument parsing, multi-device striping infrastructure,
 * deterministic pattern generation/verification, and bandwidth reporting.
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <vector>
#include <string>
#include <stdexcept>
#include <cuda.h>
#include "bam.h"


/* =========================================================================
 * Test settings (parsed from command-line arguments)
 * ========================================================================= */

struct TestSettings {
    std::vector<const char*> device_paths;
    uint32_t    ns_id;
    uint32_t    gpu_id;
    uint64_t    queue_depth;
    uint64_t    num_queues;
    uint64_t    num_pages;
    uint32_t    prp_pool_size;

    TestSettings()
        : ns_id(1)
        , gpu_id(0)
        , queue_depth(1024)
        , num_queues(16)
        , num_pages(1024)
        , prp_pool_size(256)
    {
        device_paths.push_back("/dev/libnvm0");
    }
};

inline void print_usage(const char* prog)
{
    printf("Usage: %s [options]\n", prog);
    printf("  --device PATH [PATH ...]  NVMe device(s) (default: /dev/libnvm0)\n");
    printf("  --gpu ID                  CUDA device ID (default: 0)\n");
    printf("  --ns ID                   NVMe namespace ID (default: 1)\n");
    printf("  --qd DEPTH                Queue depth (default: 1024)\n");
    printf("  --nq COUNT                Number of queue pairs per controller (default: 16)\n");
    printf("  --pages COUNT             Number of pages to test (default: 1024)\n");
    printf("  --prp-pool SIZE           PRP list pool size (default: 256)\n");
    printf("  --help                    Show this message\n");
}

inline TestSettings parse_args(int argc, char** argv)
{
    TestSettings s;
    bool has_device = false;

    for (int i = 1; i < argc; i++)
    {
        if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0)
        {
            print_usage(argv[0]);
            exit(0);
        }
        else if (strcmp(argv[i], "--device") == 0)
        {
            if (!has_device)
            {
                s.device_paths.clear();
                has_device = true;
            }
            /* Collect all following non-flag arguments as device paths */
            while (i + 1 < argc && argv[i + 1][0] != '-')
            {
                s.device_paths.push_back(argv[++i]);
            }
            if (s.device_paths.empty())
            {
                fprintf(stderr, "Error: --device requires at least one path\n");
                exit(1);
            }
        }
        else if (strcmp(argv[i], "--gpu") == 0 && i + 1 < argc)
            s.gpu_id = (uint32_t)atoi(argv[++i]);
        else if (strcmp(argv[i], "--ns") == 0 && i + 1 < argc)
            s.ns_id = (uint32_t)atoi(argv[++i]);
        else if (strcmp(argv[i], "--qd") == 0 && i + 1 < argc)
            s.queue_depth = (uint64_t)atoll(argv[++i]);
        else if (strcmp(argv[i], "--nq") == 0 && i + 1 < argc)
            s.num_queues = (uint64_t)atoll(argv[++i]);
        else if (strcmp(argv[i], "--pages") == 0 && i + 1 < argc)
            s.num_pages = (uint64_t)atoll(argv[++i]);
        else if (strcmp(argv[i], "--prp-pool") == 0 && i + 1 < argc)
            s.prp_pool_size = (uint32_t)atoi(argv[++i]);
        else
        {
            fprintf(stderr, "Unknown option: %s\n", argv[i]);
            print_usage(argv[0]);
            exit(1);
        }
    }
    return s;
}


/* =========================================================================
 * Multi-device striping (RAID 0) infrastructure
 * ========================================================================= */

/*
 * DeviceArray -- Device-side struct holding queue pair pointers for all
 * controllers. Passed as a kernel argument for stripe-aware I/O.
 *
 * Page i goes to controller (i % n_ctrls). Within that controller,
 * queue pairs are selected round-robin by warp ID.
 */
struct DeviceArray {
    QueuePair** all_qps;    /* Device array: [ctrl0.d_qps, ctrl1.d_qps, ...]  */
    uint16_t*   all_n_qps;  /* Device array: [ctrl0.n_qps, ctrl1.n_qps, ...]  */
    uint32_t    n_ctrls;    /* Number of controllers                           */
};

/*
 * build_device_array -- Allocate and populate a DeviceArray on the GPU.
 *
 * Caller owns the returned DeviceArray and must call free_device_array().
 */
inline DeviceArray build_device_array(std::vector<Controller*>& ctrls, uint32_t gpu_id)
{
    DeviceArray da;
    da.n_ctrls = (uint32_t)ctrls.size();

    /* Host staging arrays */
    std::vector<QueuePair*> h_qps(da.n_ctrls);
    std::vector<uint16_t>   h_n_qps(da.n_ctrls);
    for (uint32_t i = 0; i < da.n_ctrls; i++)
    {
        h_qps[i]   = ctrls[i]->d_qps;
        h_n_qps[i] = ctrls[i]->n_qps;
    }

    /* Allocate and copy to device */
    cudaSetDevice(gpu_id);

    cudaMalloc((void**)&da.all_qps, da.n_ctrls * sizeof(QueuePair*));
    cudaMemcpy(da.all_qps, h_qps.data(), da.n_ctrls * sizeof(QueuePair*),
               cudaMemcpyHostToDevice);

    cudaMalloc((void**)&da.all_n_qps, da.n_ctrls * sizeof(uint16_t));
    cudaMemcpy(da.all_n_qps, h_n_qps.data(), da.n_ctrls * sizeof(uint16_t),
               cudaMemcpyHostToDevice);

    return da;
}

inline void free_device_array(DeviceArray& da)
{
    cudaFree(da.all_qps);
    cudaFree(da.all_n_qps);
    da.all_qps = nullptr;
    da.all_n_qps = nullptr;
}


/* =========================================================================
 * Device-side helpers
 * ========================================================================= */

/*
 * get_qp -- Select a QueuePair for a given page using RAID 0 striping.
 *
 * Controller is selected by page_idx % n_ctrls.
 * Queue pair within that controller is selected by warp_id % n_qps.
 */
inline __device__
QueuePair* get_qp(DeviceArray* da, uint32_t page_idx, uint32_t warp_id)
{
    uint32_t ctrl_idx = page_idx % da->n_ctrls;
    QueuePair* qps = da->all_qps[ctrl_idx];
    uint16_t n_qps = da->all_n_qps[ctrl_idx];
    return qps + (warp_id % n_qps);
}

/*
 * stripe_lba -- Compute the LBA for a page on its striped controller.
 *
 * Page i is assigned to controller (i % n_ctrls). On that controller,
 * it is the (i / n_ctrls)-th page, starting at LBA:
 *   (i / n_ctrls) * blocks_per_page
 */
inline __device__
uint64_t stripe_lba(uint32_t page_idx, uint32_t n_ctrls, uint32_t blocks_per_page)
{
    return (uint64_t)(page_idx / n_ctrls) * blocks_per_page;
}

/*
 * fill_pattern -- Fill a page with a deterministic pattern.
 *
 * Pattern: each uint32_t word at offset j is set to (page_idx << 16) | (j & 0xFFFF).
 * This allows per-page and per-word verification.
 */
inline __device__
void fill_pattern(uint8_t* page, uint32_t page_idx, uint32_t page_size)
{
    uint32_t* words = (uint32_t*)page;
    uint32_t n_words = page_size / sizeof(uint32_t);
    for (uint32_t j = 0; j < n_words; j++)
        words[j] = (page_idx << 16) | (j & 0xFFFF);
}

/*
 * verify_pattern -- Verify a page against the expected pattern.
 *
 * Returns the number of mismatched uint32_t words.
 */
inline __device__
uint32_t verify_pattern(uint8_t* page, uint32_t page_idx, uint32_t page_size)
{
    uint32_t* words = (uint32_t*)page;
    uint32_t n_words = page_size / sizeof(uint32_t);
    uint32_t mismatches = 0;
    for (uint32_t j = 0; j < n_words; j++)
    {
        uint32_t expected = (page_idx << 16) | (j & 0xFFFF);
        if (words[j] != expected)
            mismatches++;
    }
    return mismatches;
}


/* =========================================================================
 * Host-side reporting helpers
 * ========================================================================= */

inline void report_bw(const char* label, size_t bytes, float ms)
{
    double gb = (double)bytes / (1024.0 * 1024.0 * 1024.0);
    double sec = (double)ms / 1000.0;
    double gbps = (sec > 0) ? gb / sec : 0;
    printf("  %-30s %8.2f GB/s  (%zu bytes in %.2f ms)\n", label, gbps, bytes, ms);
}

inline void report_result(const char* label, uint64_t mismatches, uint64_t total_pages)
{
    if (mismatches == 0)
        printf("  %-30s PASS (%lu pages verified)\n", label, (unsigned long)total_pages);
    else
        printf("  %-30s FAIL (%lu mismatched words across %lu pages)\n",
               label, (unsigned long)mismatches, (unsigned long)total_pages);
}


/* =========================================================================
 * CUDA error checking macro
 * ========================================================================= */

#define TEST_CUDA_CHECK(call)                                               \
    do {                                                                    \
        cudaError_t err = (call);                                           \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                   \
                    __FILE__, __LINE__, cudaGetErrorString(err));            \
            exit(1);                                                        \
        }                                                                   \
    } while (0)


#endif /* __TEST_COMMON_H__ */
