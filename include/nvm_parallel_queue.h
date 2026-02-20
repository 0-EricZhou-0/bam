#ifndef __NVM_PARALLEL_QUEUE_H_
#define __NVM_PARALLEL_QUEUE_H_

#ifndef __device__
#define __device__
#endif
#ifndef __host__
#define __host__
#endif
#ifndef __forceinline__
#define __forceinline__ inline
#endif

#include "host_util.h"
#include "nvm_types.h"
#include "nvm_util.h"
#include <simt/atomic>

#define LOCKED   1
#define UNLOCKED 0


/*
 * get_id -- Convert a ticket number to the expected turn value.
 *
 * The ticket lock scheme uses a per-slot "ticket" counter. Each slot can
 * hold one command at a time. A thread's "turn" to use slot (ticket % qs)
 * is computed as (ticket / qs) * 2, doubling each generation so that the
 * ticket value advances monotonically across wrap-arounds.
 */
__forceinline__ __device__ uint64_t get_id(uint64_t x, uint64_t y) {
    return (x >> y) * 2;
}


/*
 * get_cid -- Acquire a unique NVMe command ID from the CID pool.
 *
 * Searches the 65536-entry CID pool using an atomic ticket counter.
 * Each slot is a lock bit: UNLOCKED means available. Uses fetch_or to
 * atomically claim a slot. Spins until a free slot is found.
 *
 * Returns the 16-bit command ID.
 */
inline __device__
uint16_t get_cid(nvm_queue_t* sq) {
    bool not_found = true;
    uint16_t id;

    do {
        id = sq->cid_ticket.fetch_add(1, simt::memory_order_relaxed) & (65535);
        uint64_t old = sq->cid[id].val.fetch_or(LOCKED, simt::memory_order_acquire);
        not_found = old == LOCKED;
    } while (not_found);

    return id;
}

/*
 * put_cid -- Release a command ID back to the CID pool.
 *
 * Stores UNLOCKED into the CID slot, making it available for reuse.
 */
inline __device__
void put_cid(nvm_queue_t* sq, uint16_t id) {
    sq->cid[id].val.store(UNLOCKED, simt::memory_order_release);
}

/*
 * move_tail -- Advance the SQ tail through contiguous ready entries.
 *
 * Starting from @cur_tail, scans forward through tail_mark slots. Each
 * slot marked LOCKED indicates a command has been written and is ready
 * for submission. Clears each mark as it advances. Stops when:
 *   - A slot is not marked (command not yet written), or
 *   - The next position would collide with the SQ head (queue full).
 *
 * Returns the number of entries advanced.
 */
inline __device__
uint32_t move_tail(nvm_queue_t* q, uint32_t cur_tail) {
    uint32_t count = 0;

    bool pass = true;
    while (pass) {
        pass = (((cur_tail+count+1) & q->qs_minus_1) != (q->head.load(simt::memory_order_relaxed) & q->qs_minus_1 ));
        if (pass) {
            pass = ((q->tail_mark[(cur_tail+count)&q->qs_minus_1].val.exchange(UNLOCKED, simt::memory_order_relaxed)) == LOCKED);
            if (pass)
                count++;
        }
    }

    q->head_lock.fetch_add(1, simt::memory_order_acq_rel);
    return (count);
}

/*
 * move_head_cq -- Advance the CQ head through contiguous consumed entries.
 *
 * Scans forward from @cur_head through head_mark slots. Each slot marked
 * LOCKED indicates a completion entry that has been processed. Clears each
 * mark as it advances.
 *
 * Also updates the paired SQ's head pointer based on the SQHD (SQ Head
 * Doorbell) value from the last completion entry, freeing SQ slots.
 *
 * Returns the number of CQ entries advanced.
 */
inline __device__
uint32_t move_head_cq(nvm_queue_t* q, uint32_t cur_head, nvm_queue_t* sq) {
    uint32_t count = 0;
    (void) sq;

    bool pass = true;
    while (pass) {
        uint32_t loc = (cur_head+count++)&q->qs_minus_1;
        pass = (q->head_mark[loc].val.exchange(UNLOCKED, simt::memory_order_relaxed)) == LOCKED;
    }
    count -= 1;

    if (count) {
        /* Extract SQHD from the last completion entry's DWORD2 */
        uint32_t loc_ = (cur_head + (count -1)) & q->qs_minus_1;
        uint32_t cpl_entry = ((nvm_cpl_t*)q->vaddr)[loc_].dword[2];
        uint16_t new_sq_head =  (cpl_entry & 0x0000ffff);
        uint32_t sq_move_count = 0;
        uint32_t cur_sq_head = sq->head.load(simt::memory_order_relaxed);
        uint32_t loc = cur_sq_head & sq->qs_minus_1;

        if (loc != new_sq_head) {
            for (; loc != new_sq_head; sq_move_count++, loc= ((loc+1)  & sq->qs_minus_1)) {
                sq->tickets[loc].val.fetch_add(1, simt::memory_order_relaxed);
            }
            sq->head.fetch_add(sq_move_count, simt::memory_order_acq_rel);
        }
    }
    return (count);
}

/*
 * clean_cids -- Batch-release CIDs from CQ entries (currently unused).
 */
inline __device__
void clean_cids(nvm_queue_t* cq, nvm_queue_t* sq, uint32_t count) {
    for (size_t i  = 0; i < count; i++) {
        put_cid(sq, cq->clean_cid[i]);
    }
}

/*
 * move_head_sq -- Advance the SQ head through contiguous completed entries.
 *
 * Scans forward from @cur_head through head_mark slots. Each slot marked
 * LOCKED indicates a command whose completion has been processed. Clears
 * marks and releases ticket slots as it advances.
 *
 * Returns the number of SQ entries advanced.
 */
inline __device__
uint32_t move_head_sq(nvm_queue_t* q, uint32_t cur_head) {
    uint32_t count = 0;

    bool pass = true;
    while (pass) {
        uint64_t loc = (cur_head + count)&q->qs_minus_1;
        pass = (q->head_mark[loc].val.exchange(UNLOCKED, simt::memory_order_relaxed)) == LOCKED;
        if (pass) {
            q->tickets[loc].val.fetch_add(1, simt::memory_order_relaxed);
            count++;
        }
    }
    return (count);
}

typedef ulonglong4 copy_type;

/*
 * sq_enqueue -- Enqueue an NVMe command to the submission queue.
 *
 * Uses a ticket-based ordering scheme to ensure commands are written to
 * the SQ in order even when many GPU threads enqueue concurrently:
 *
 * 1. Atomically obtain a ticket (position in the queue).
 * 2. Wait until the slot's ticket counter matches our expected turn,
 *    indicating all prior commands to this slot have completed.
 * 3. Copy the 64-byte command to the SQ slot using 128-bit writes.
 * 4. Mark the tail_mark slot as LOCKED (ready for doorbell write).
 * 5. Attempt to acquire the tail_lock and write the SQ doorbell via
 *    a direct MMIO store (PTX st.mmio instruction), advancing the tail
 *    through all contiguous ready entries.
 * 6. Release the ticket slot for the next generation.
 *
 * @sq       Pointer to the submission queue descriptor (device memory).
 * @cmd      Pointer to the 64-byte NVMe command to enqueue.
 * @pc_tail  Optional: atomic page-cache tail to snapshot (for page cache use).
 * @cur_pc_tail  Optional: [out] snapshot of pc_tail value.
 *
 * Returns the SQ slot position where the command was written.
 */
inline __device__
uint16_t sq_enqueue(nvm_queue_t* sq, nvm_cmd_t* cmd, simt::atomic<uint64_t, simt::thread_scope_device>* pc_tail =NULL, uint64_t * cur_pc_tail=NULL) {

    /* Step 1: Get a ticket (our position in the queue) */
    uint32_t ticket;
    ticket = sq->in_ticket.fetch_add(1, simt::memory_order_relaxed);

    uint32_t pos = ticket & (sq->qs_minus_1);
    uint64_t id = get_id(ticket, sq->qs_log2);

    /* Step 2: Wait for our turn (relaxed spin, then acquire fence) */
    unsigned int ns = 8;
    while ((sq->tickets[pos].val.load(simt::memory_order_relaxed) != id) ) {
#if defined(__CUDACC__) && (__CUDA_ARCH__ >= 700 || !defined(__CUDA_ARCH__))
        __nanosleep(ns);
        if (ns < 256) {
            ns *= 2;
        }
#endif
    }

    ns = 8;
    while ((sq->tickets[pos].val.load(simt::memory_order_acquire) != id) ) {
#if defined(__CUDACC__) && (__CUDA_ARCH__ >= 700 || !defined(__CUDA_ARCH__))
        __nanosleep(ns);
        if (ns < 256) {
            ns *= 2;
        }
#endif
    }

    /* Step 3: Copy the 64-byte command to the SQ slot */
    copy_type* queue_loc = ((copy_type*)(((nvm_cmd_t*)(sq->vaddr)) + pos));
    copy_type* cmd_ = ((copy_type*)(cmd->dword));

#pragma unroll
    for (uint32_t i = 0; i < 64/sizeof(copy_type); i++) {
        queue_loc[i] = cmd_[i];
    }

    /* Step 4: Snapshot page-cache tail if requested (for page cache use) */
    if (pc_tail) {
        *cur_pc_tail = pc_tail->load(simt::memory_order_relaxed);
    }

    /* Step 5: Mark this entry as ready and try to write the doorbell */
    sq->tail_mark[pos].val.store(LOCKED, simt::memory_order_release);

    bool cont = true;
    ns = 8;
    cont = sq->tail_mark[pos].val.load(simt::memory_order_relaxed) == LOCKED;
    while(cont) {
        bool new_cont = sq->tail_lock.load(simt::memory_order_relaxed) == LOCKED;
        if (!new_cont) {
            new_cont = sq->tail_lock.fetch_or(LOCKED, simt::memory_order_acquire) == LOCKED;
            if(!new_cont) {
                uint32_t cur_tail = sq->tail.load(simt::memory_order_relaxed);

                uint32_t tail_move_count = move_tail(sq, cur_tail);

                if (tail_move_count) {
                    uint32_t new_tail = cur_tail + tail_move_count;
                    uint32_t new_db = (new_tail) & (sq->qs_minus_1);
                    if (pc_tail) {
                        *cur_pc_tail = pc_tail->load(simt::memory_order_acquire);
                    }
                    /* Write SQ doorbell via direct MMIO store */
                    asm volatile ("st.mmio.relaxed.sys.global.u32 [%0], %1;" :: "l"(sq->db),"r"(new_db) : "memory");

                    sq->tail.store(new_tail, simt::memory_order_release);
                }
                sq->tail_lock.store(UNLOCKED, simt::memory_order_release);
            }
        }
        cont = sq->tail_mark[pos].val.load(simt::memory_order_relaxed) == LOCKED;
        if (cont) {
#if defined(__CUDACC__) && (__CUDA_ARCH__ >= 700 || !defined(__CUDA_ARCH__))
            __nanosleep(ns);
            if (ns < 256) {
                ns *= 2;
            }
#endif
        }
    }

    /* Step 6: Release ticket slot for next generation */
    sq->tickets[pos].val.fetch_add(1, simt::memory_order_acq_rel);
    return pos;
}

/*
 * sq_dequeue -- Mark an SQ entry as completed and advance the SQ head.
 *
 * Called after a completion has been processed. Marks the SQ head_mark
 * slot and attempts to advance the SQ head pointer through contiguous
 * completed entries.
 *
 * @sq   Pointer to the submission queue descriptor.
 * @pos  The SQ slot position to dequeue.
 */
inline __device__
void sq_dequeue(nvm_queue_t* sq, uint16_t pos) {

    sq->head_mark[pos].val.store(LOCKED, simt::memory_order_relaxed);
    bool cont = true;
    unsigned int ns = 8;
    cont = sq->head_mark[pos].val.load(simt::memory_order_relaxed) == LOCKED;
    while (cont) {
            bool new_cont = sq->head_lock.exchange(LOCKED, simt::memory_order_acquire) == LOCKED;
            if (!new_cont){
                uint32_t cur_head = sq->head.load(simt::memory_order_relaxed);;

                uint32_t head_move_count = move_head_sq(sq, cur_head);
                if (head_move_count) {
                    sq->head.store(cur_head + head_move_count, simt::memory_order_relaxed);
                }

                sq->head_lock.store(UNLOCKED, simt::memory_order_release);
            }
            cont = sq->head_mark[pos].val.load(simt::memory_order_relaxed) == LOCKED;
            if (cont) {
#if defined(__CUDACC__) && (__CUDA_ARCH__ >= 700 || !defined(__CUDA_ARCH__))
                __nanosleep(ns);
                if (ns < 256) {
                    ns *= 2;
                }
#endif
            }
    }
}

/*
 * cq_poll -- Poll the completion queue for a specific command ID.
 *
 * Scans through CQ entries starting from the current head, looking for
 * an entry whose CID matches @search_cid and whose phase bit matches
 * the expected phase for that position. The phase bit flips each time
 * the queue wraps around, allowing detection of new vs. stale entries.
 *
 * Uses exponential backoff via __nanosleep() when no matching entry is
 * found in a scan pass.
 *
 * @cq          Pointer to the completion queue descriptor.
 * @search_cid  The command ID to search for.
 * @loc_        [out] Optional: the logical position (head + offset) of the match.
 * @cq_head     [out] Optional: the CQ head value at the time of the match.
 *
 * Returns the physical CQ slot index where the matching entry was found.
 */
inline __device__
uint32_t cq_poll(nvm_queue_t* cq, uint16_t search_cid, uint32_t* loc_ = NULL, uint32_t* cq_head = NULL) {
    uint64_t j = 0;
    unsigned int ns = 8;

    while (true) {
        uint32_t head = cq->head.load(simt::memory_order_relaxed);

        for (size_t i = 0; i < cq->qs_minus_1; i++) {
            uint32_t cur_head = head + i;
            bool search_phase = ((~(cur_head >> cq->qs_log2)) & 0x01);
            uint32_t loc = cur_head & (cq->qs_minus_1);
            uint32_t cpl_entry = ((nvm_cpl_t*)cq->vaddr)[loc].dword[3];
            uint32_t cid = (cpl_entry & 0x0000ffff);
            bool phase = (cpl_entry & 0x00010000) >> 16;

            if ((cid == search_cid) && (phase == search_phase)){
                if (cq_head) *cq_head = head;
                if (loc_) *loc_ = cur_head;
                return loc;
            }
            if (phase != search_phase)
                break;
        }
        j++;
#if defined(__CUDACC__) && (__CUDA_ARCH__ >= 700 || !defined(__CUDA_ARCH__))
         __nanosleep(ns);
         if (ns < 256) {
             ns *= 2;
         }
#endif
    }
}

/*
 * cq_dequeue -- Dequeue a CQ entry and advance the CQ head.
 *
 * After cq_poll() finds a matching entry, this function:
 * 1. Increments the CQ tail counter (tracks in-flight completions).
 * 2. Acquires the pos_lock for this CQ slot (prevents concurrent dequeue
 *    of the same slot).
 * 3. Marks the CQ head_mark slot as consumed.
 * 4. Attempts to acquire the CQ head_lock and advance the CQ head
 *    through all contiguous consumed entries (via move_head_cq).
 * 5. Writes the CQ doorbell via MMIO to acknowledge processed entries.
 * 6. Waits until the CQ head has advanced past our entry's position
 *    (ensuring the controller knows we're done with this slot).
 * 7. Releases the pos_lock.
 *
 * @cq         Pointer to the completion queue descriptor.
 * @pos        The physical CQ slot index (from cq_poll return value).
 * @sq         Pointer to the paired submission queue (for SQ head update).
 * @loc_       Logical position of our entry (from cq_poll's loc_ output).
 *             Default 0 when not using the extended completion tracking.
 * @cur_head_  CQ head at time of poll (from cq_poll's cq_head output).
 *             Default 0 when not using the extended completion tracking.
 */
inline __device__
void cq_dequeue(nvm_queue_t* cq, uint16_t pos, nvm_queue_t* sq, uint32_t loc_ = 0, uint32_t cur_head_ = 0) {
    cq->tail.fetch_add(1, simt::memory_order_acq_rel);

    /* Wait for pos_lock to become available, then acquire it */
    unsigned int ns = 8;
    while ((cq->pos_locks[pos].val.load(simt::memory_order_relaxed) != 0) ) {
#if defined(__CUDACC__) && (__CUDA_ARCH__ >= 700 || !defined(__CUDA_ARCH__))
        __nanosleep(ns);
        if (ns < 256) {
            ns *= 2;
        }
#endif
    }

    ns = 8;
    while ((cq->pos_locks[pos].val.fetch_or(1, simt::memory_order_acquire) != 0) ) {
#if defined(__CUDACC__) && (__CUDA_ARCH__ >= 700 || !defined(__CUDA_ARCH__))
        __nanosleep(ns);
        if (ns < 256) {
            ns *= 2;
        }
#endif
    }

    /* Mark this entry as consumed */
    cq->head_mark[pos].val.store(LOCKED, simt::memory_order_release);

    /* Try to advance the CQ head and write the doorbell */
    bool cont = true;
    ns = 8;
    cont = cq->head_mark[pos].val.load(simt::memory_order_relaxed) == LOCKED;
    while (cont) {
            bool new_cont = cq->head_lock.fetch_or(LOCKED, simt::memory_order_acquire) == LOCKED;
            if (!new_cont) {
                uint32_t cur_head = cq->head.load(simt::memory_order_relaxed);;

                uint32_t head_move_count = move_head_cq(cq, cur_head, sq);

                if (head_move_count) {
                    uint32_t new_head = cur_head + head_move_count;
                    uint32_t new_db = (new_head) & (cq->qs_minus_1);

                    /* Write CQ doorbell via direct MMIO store */
                    asm volatile ("st.mmio.relaxed.sys.global.u32 [%0], %1;" :: "l"(cq->db),"r"(new_db) : "memory");

                    cq->head.store(new_head, simt::memory_order_release);
                }
                cq->head_lock.store(UNLOCKED, simt::memory_order_release);
            }
            cont = cq->head_mark[pos].val.load(simt::memory_order_relaxed) == LOCKED;
            if (cont) {
#if defined(__CUDACC__) && (__CUDA_ARCH__ >= 700 || !defined(__CUDA_ARCH__))
                __nanosleep(ns);
                if (ns < 256) {
                    ns *= 2;
                }
#endif
            }
    }

    /*
     * Wait until the CQ head has advanced past our entry's logical position.
     * This ensures the controller has been notified (via doorbell) that we
     * are done with this slot before we release the pos_lock.
     */
    uint64_t j = 0;
    uint32_t new_head = cq->head.load(simt::memory_order_relaxed);
    ns = 8;
    do {
        if (new_head > cur_head_) {
            if ((loc_ >= cur_head_) && (loc_ < new_head))
                break;
        }
        else if (new_head < cur_head_) {
            if ((loc_ >= cur_head_))
                break;
            if (loc_ < new_head)
                break;
        }

        j++;
        new_head = cq->head.load(simt::memory_order_relaxed);
#if defined(__CUDACC__) && (__CUDA_ARCH__ >= 700 || !defined(__CUDA_ARCH__))
        __nanosleep(ns);
        if (ns < 256) {
            ns *= 2;
        }
#endif
    } while(true);

    cq->pos_locks[pos].val.store(0, simt::memory_order_release);
}

#endif // __NVM_PARALLEL_QUEUE_H_
