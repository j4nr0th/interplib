//
// Created by jan on 19.10.2024.
//

#ifndef ALLOCATOR_H
#define ALLOCATOR_H
#include "common_defines.h"
#include <stdlib.h>

typedef struct
{
    void *(*alloc)(void *state, size_t size);
    void *(*realloc)(void *state, void *ptr, size_t new_size);
    void (*free)(void *state, void *ptr);
    void *state;
} allocator_callbacks;

static inline void *allocate(const allocator_callbacks *allocator, const size_t sz)
{
    return allocator->alloc(allocator->state, sz);
}

static inline void *reallocate(const allocator_callbacks *allocator, void *ptr, const size_t new_sz)
{
    return allocator->realloc(allocator->state, ptr, new_sz);
}

static inline void deallocate(const allocator_callbacks *allocator, void *ptr)
{
    return allocator->free(allocator->state, ptr);
}

#endif // ALLOCATOR_H
