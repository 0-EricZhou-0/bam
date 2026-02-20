
#ifndef __BENCHMARK_CTRL_H__
#define __BENCHMARK_CTRL_H__

#include <cstdint>
#include "buffer.h"
#include "nvm_types.h"
#include "nvm_ctrl.h"
#include "nvm_aq.h"
#include "nvm_admin.h"
#include "nvm_util.h"
#include "nvm_error.h"
#include <string>
#include <memory>
#include <vector>
#include <stdexcept>
#include <cstddef>
#include <cstring>
#include <cstdio>
#include <algorithm>
#include <unistd.h>
#include <fcntl.h>
#include <cstdlib>
#include <simt/atomic>

#include "queue.h"


#define MAX_QUEUES 1024


/*
 * Controller -- Manages an NVMe controller and its I/O queue pairs for GPU access.
 *
 * Wraps a libnvm controller handle (nvm_ctrl_t) together with an admin queue,
 * controller/namespace info, and an array of QueuePairs whose memory resides
 * on a specific CUDA device. A device-side copy of this struct is maintained
 * at d_ctrl_ptr so GPU kernels can access queue pairs and counters.
 *
 * Typical host-side usage:
 *   Controller ctrl("/dev/libnvm0", ns_id, cudaDevice, queueDepth, numQueues);
 *   // Launch kernels using ctrl.d_qps (device pointer to QueuePair array)
 *
 * Typical device-side usage (inside CUDA kernel):
 *   uint32_t queue = (tid / 32) % ctrl->n_qps;
 *   QueuePair* qp = ctrl->d_qps + queue;
 *   bam_read(qp, lba, n_blocks, prp1, 0);
 */
struct Controller
{
    simt::atomic<uint64_t, simt::thread_scope_device> access_counter;  /* Device-side I/O counter for stats   */
    nvm_ctrl_t*             ctrl;       /* libnvm controller handle                     */
    nvm_aq_ref              aq_ref;     /* Admin queue reference for admin commands      */
    DmaPtr                  aq_mem;     /* DMA mapping for admin queue memory            */
    struct nvm_ctrl_info    info;       /* Controller identification data                */
    struct nvm_ns_info      ns;         /* Namespace identification data                 */
    uint16_t                n_sqs;      /* Number of submission queues allocated          */
    uint16_t                n_cqs;      /* Number of completion queues allocated          */
    uint16_t                n_qps;      /* Number of queue pairs created                  */
    uint32_t                deviceId;   /* CUDA device ordinal                           */
    QueuePair**             h_qps;      /* Host-side array of QueuePair pointers          */
    QueuePair*              d_qps;      /* Device-side array of QueuePairs (cudaMalloc'd) */

    simt::atomic<uint64_t, simt::thread_scope_device> queue_counter;   /* Device-side queue selection counter  */

    uint32_t page_size;                 /* Controller memory page size in bytes           */
    uint32_t blk_size;                  /* Namespace LBA data size in bytes               */
    uint32_t blk_size_log;              /* log2(blk_size)                                */

    void* d_ctrl_ptr;                   /* Device copy of this Controller struct          */
    BufferPtr d_ctrl_buff;              /* Manages lifetime of d_ctrl_ptr allocation      */

#ifdef __DIS_CLUSTER__
    Controller(uint64_t controllerId, uint32_t nvmNamespace, uint32_t adapter, uint32_t segmentId);
#endif

    Controller(const char* path, uint32_t nvmNamespace, uint32_t cudaDevice, uint64_t queueDepth, uint64_t numQueues);

    /*
     * reserveQueues -- Request I/O queues from the controller via admin commands.
     *
     * Calls NVMe Set Features (Number of Queues) to allocate submission and
     * completion queues. Updates n_sqs and n_cqs with the actual counts granted.
     */
    void reserveQueues();
    void reserveQueues(uint16_t numSubmissionQueues);
    void reserveQueues(uint16_t numSubmissionQueues, uint16_t numCompletionQueues);

    /* Print and reset the device-side access_counter. */
    void print_reset_stats(void);

    ~Controller();
};


inline void Controller::print_reset_stats(void) {
    cuda_err_chk(cudaMemcpy(&access_counter, d_ctrl_ptr, sizeof(simt::atomic<uint64_t, simt::thread_scope_device>), cudaMemcpyDeviceToHost));
    std::cout << "------------------------------------" << std::endl;
    std::cout << std::dec << "#SSDAccesses:\t" << access_counter << std::endl;

    cuda_err_chk(cudaMemset(d_ctrl_ptr, 0, sizeof(simt::atomic<uint64_t, simt::thread_scope_device>)));
}

/*
 * initializeController -- Set up admin queue and identify controller/namespace.
 *
 * Creates the admin queue pair from pre-allocated DMA memory, then issues
 * NVMe Identify Controller, Identify Namespace, and Get Number of Queues
 * admin commands to populate ctrl.info, ctrl.ns, ctrl.n_sqs, and ctrl.n_cqs.
 */
static void initializeController(struct Controller& ctrl, uint32_t ns_id)
{
    int status = nvm_aq_create(&ctrl.aq_ref, ctrl.ctrl, ctrl.aq_mem.get());
    if (!nvm_ok(status))
    {
        throw std::runtime_error(std::string("Failed to reset controller: ") + nvm_strerror(status));
    }

    status = nvm_admin_ctrl_info(ctrl.aq_ref, &ctrl.info, NVM_DMA_OFFSET(ctrl.aq_mem, 2), ctrl.aq_mem->ioaddrs[2]);
    if (!nvm_ok(status))
    {
        throw std::runtime_error(nvm_strerror(status));
    }

    status = nvm_admin_ns_info(ctrl.aq_ref, &ctrl.ns, ns_id, NVM_DMA_OFFSET(ctrl.aq_mem, 2), ctrl.aq_mem->ioaddrs[2]);
    if (!nvm_ok(status))
    {
        throw std::runtime_error(nvm_strerror(status));
    }

    status = nvm_admin_get_num_queues(ctrl.aq_ref, &ctrl.n_cqs, &ctrl.n_sqs);
    if (!nvm_ok(status))
    {
        throw std::runtime_error(nvm_strerror(status));
    }
}



#ifdef __DIS_CLUSTER__
Controller::Controller(uint64_t ctrl_id, uint32_t ns_id, uint32_t)
    : ctrl(nullptr)
    , aq_ref(nullptr)
{
    int status = nvm_dis_ctrl_init(&ctrl, ctrl_id);
    if (!nvm_ok(status))
    {
        throw std::runtime_error(std::string("Failed to get controller reference: ") + nvm_strerror(status));
    }

    aq_mem = createDma(ctrl, ctrl->page_size * 3, 0, 0);

    initializeController(*this, ns_id);
}
#endif


/*
 * Controller constructor -- Open an NVMe device and prepare it for GPU I/O.
 *
 * 1. Opens the libnvm character device (e.g. /dev/libnvm0).
 * 2. Initializes the controller handle and admin queue.
 * 3. Identifies the controller and namespace.
 * 4. Registers the controller's BAR0 memory as a CUDA IO-mapped region
 *    so GPU threads can access doorbell registers via MMIO.
 * 5. Reserves up to MAX_QUEUES I/O queues.
 * 6. Creates QueuePairs in GPU memory and copies them to d_qps.
 * 7. Creates a device-side copy of this Controller struct at d_ctrl_ptr.
 */
inline Controller::Controller(const char* path, uint32_t ns_id, uint32_t cudaDevice, uint64_t queueDepth, uint64_t numQueues)
    : ctrl(nullptr)
    , aq_ref(nullptr)
    , deviceId(cudaDevice)
{
    int fd = open(path, O_RDWR);
    if (fd < 0)
    {
        throw std::runtime_error(std::string("Failed to open descriptor: ") + strerror(errno));
    }

    /* Initialize libnvm controller handle from the character device fd */
    int status = nvm_ctrl_init(&ctrl, fd);
    if (!nvm_ok(status))
    {
        throw std::runtime_error(std::string("Failed to get controller reference: ") + nvm_strerror(status));
    }

    /* Allocate admin queue memory (3 pages: ASQ + ACQ + scratch page for identify) */
    aq_mem = createDma(ctrl, ctrl->page_size * 3);

    initializeController(*this, ns_id);

    /* Register BAR0 as CUDA IO memory so GPU can write doorbell registers */
    cudaError_t err = cudaHostRegister((void*) ctrl->mm_ptr, NVM_CTRL_MEM_MINSIZE, cudaHostRegisterIoMemory);
    if (err != cudaSuccess)
    {
        throw std::runtime_error(std::string("Unexpected error while mapping IO memory (cudaHostRegister): ") + cudaGetErrorString(err));
    }

    queue_counter = 0;
    page_size = ctrl->page_size;
    blk_size = this->ns.lba_data_size;
    blk_size_log = std::log2(blk_size);

    /* Reserve maximum queues, then clamp to what's available and requested */
    reserveQueues(MAX_QUEUES, MAX_QUEUES);
    n_qps = std::min(n_sqs, n_cqs);
    n_qps = std::min(n_qps, (uint16_t)numQueues);
    printf("SQs: %d\tCQs: %d\tn_qps: %d\n", n_sqs, n_cqs, n_qps);

    /* Allocate host and device arrays for QueuePairs */
    h_qps = (QueuePair**) malloc(sizeof(QueuePair)*n_qps);
    cuda_err_chk(cudaMalloc((void**)&d_qps, sizeof(QueuePair)*n_qps));

    /* Create each QueuePair and copy it to the device-side array */
    for (size_t i = 0; i < n_qps; i++) {
        h_qps[i] = new QueuePair(ctrl, cudaDevice, ns, info, aq_ref, i+1, queueDepth);
        cuda_err_chk(cudaMemcpy(d_qps+i, h_qps[i], sizeof(QueuePair), cudaMemcpyHostToDevice));
    }

    close(fd);

    /* Create a device-side copy of this Controller struct */
    d_ctrl_buff = createBuffer(sizeof(Controller), cudaDevice);
    d_ctrl_ptr = d_ctrl_buff.get();
    cuda_err_chk(cudaMemcpy(d_ctrl_ptr, this, sizeof(Controller), cudaMemcpyHostToDevice));
}



inline Controller::~Controller()
{
    cudaFree(d_qps);
    for (size_t i = 0; i < n_qps; i++) {
        delete h_qps[i];
    }
    free(h_qps);
    nvm_aq_destroy(aq_ref);
    nvm_ctrl_free(ctrl);
}



inline void Controller::reserveQueues()
{
    reserveQueues(n_sqs, n_cqs);
}



inline void Controller::reserveQueues(uint16_t numSubmissionQueues)
{
    reserveQueues(numSubmissionQueues, n_cqs);
}



inline void Controller::reserveQueues(uint16_t numSubs, uint16_t numCpls)
{
    int status = nvm_admin_request_num_queues(aq_ref, &numSubs, &numCpls);
    if (!nvm_ok(status))
    {
        throw std::runtime_error(std::string("Failed to reserve queues: ") + nvm_strerror(status));
    }

    n_sqs = numSubs;
    n_cqs = numCpls;
}


/*
 * BamBuffer convenience constructor -- Extract ctrl handle and CUDA device
 * from a Controller object.
 */
inline BamBuffer::BamBuffer(Controller& ctrl, size_t total_size)
    : BamBuffer(ctrl.ctrl, total_size, ctrl.deviceId)
{
}


#endif
