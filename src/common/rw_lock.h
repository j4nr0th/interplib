//
// Created by jan on 2025-09-12.
//

#ifndef INTERPLIB_RW_LOCK_H
#define INTERPLIB_RW_LOCK_H

#include "allocator.h"
#include "error.h"
#include <threads.h>

typedef struct
{
    mtx_t lock;
    cnd_t cond_read;
    cnd_t cond_write;
    unsigned readers;
    unsigned writers;
} rw_lock_t;

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
