#ifndef __UTIL_H__
#define __UTIL_H__

#ifndef __device__
#define __device__
#endif
#ifndef __host__
#define __host__
#endif
#ifndef __forceinline__
#define __forceinline__ inline
#endif

#include "cuda.h"
#include "nvm_util.h"
#include "host_util.h"
#include <cstdio>


/*
 * cuda_err_chk -- Assert wrapper for CUDA runtime calls.
 *
 * Prints error message with file and line number on failure.
 * Usage: cuda_err_chk(cudaMalloc(...));
 */
#define cuda_err_chk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

#ifndef __CUDACC__
inline void gpuAssert(int code, const char *file, int line, bool abort=false)
{
    if (code != 0)
    {
        fprintf(stderr,"Assert: %i %s %d\n", code, file, line);
        if (abort) exit(1);
    }
}
#else
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=false)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(1);
    }
}
#endif

/* Ceiling division: (X + Y - 1) / 2^Z */
#define CEIL(X, Y, Z) ((X + Y - 1) >> Z)


/*
 * hexdump -- Print a hex dump of a memory region from a GPU thread.
 *
 * Useful for debugging NVMe command/completion contents on the device.
 * Prints @len bytes starting at @mem in rows of HEXDUMP_COLS columns.
 */
#ifndef HEXDUMP_COLS
#define HEXDUMP_COLS 16
#endif
inline __device__ void hexdump(void *mem, unsigned int len)
{
        unsigned int i;

        for(i = 0; i < len + ((len % HEXDUMP_COLS) ? (HEXDUMP_COLS - len % HEXDUMP_COLS) : 0); i++)
        {
                /* print offset */
                if(i % HEXDUMP_COLS == 0)
                {
                        printf("\n0x%06x: ", i);
                }

                /* print hex data */
                if(i < len)
                {
                        printf("%02x ", 0xFF & ((char*)mem)[i]);
                }
                else /* end of block, just aligning for ASCII dump */
                {
                        printf("   ");
                }
        }
        printf("\n");
}

/* Suppress unused-variable warnings. */
template <typename T>
void __ignore(T &&)
{ }

/*
 * warp_memcpy -- Cooperatively copy @num elements across active warp lanes.
 *
 * Each active lane copies a strided subset of elements. Assumes both @dest
 * and @src are aligned to sizeof(T) and @num is a count of T elements.
 */
template <typename T>
inline __device__
void warp_memcpy(T* dest, const T* src, size_t num) {
#ifndef __CUDACC__
    uint32_t mask = 1;
#else
    uint32_t mask = __activemask();
#endif
        uint32_t active_cnt = __popc(mask);
        uint32_t lane = lane_id();
        uint32_t prior_mask = mask >> (32 - lane);
        uint32_t prior_count = __popc(prior_mask);

        for(size_t i = prior_count; i < num; i+=active_cnt)
                dest[i] = src[i];
}

#endif // __UTIL_H__
