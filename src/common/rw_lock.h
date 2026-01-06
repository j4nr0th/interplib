#ifndef INTERPLIB_RW_LOCK_H
#define INTERPLIB_RW_LOCK_H

#include "error.h"
#include <threads.h>

typedef struct rw_lock_t rw_lock_t;
struct rw_lock_t
{
    mtx_t lock;
    cnd_t cond_read;
    cnd_t cond_write;
    unsigned readers;
    unsigned writers;
};

INTERPLIB_INTERNAL
interp_result_t rw_lock_init(rw_lock_t *this);

INTERPLIB_INTERNAL
void rw_lock_destroy(rw_lock_t *this);

INTERPLIB_INTERNAL
void rw_lock_acquire_read(rw_lock_t *this);

INTERPLIB_INTERNAL
void rw_lock_acquire_write(rw_lock_t *this);

INTERPLIB_INTERNAL
void rw_lock_release_read(rw_lock_t *this);

INTERPLIB_INTERNAL
void rw_lock_release_write(rw_lock_t *this);

#endif // INTERPLIB_RW_LOCK_H
