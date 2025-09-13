//
// Created by jan on 2025-09-07.
//
#include "common.h"

enum
{
    TEST_ALLOCATOR_MAGIC_NUMBER = 0x12345678,
};

static void *wrap_malloc(void *state, size_t size)
{
    TEST_ASSERTION((void *)TEST_ALLOCATOR_MAGIC_NUMBER == state, "Magic number mismatch.");
    return malloc(size);
}

static void *wrap_realloc(void *state, void *ptr, size_t new_size)
{
    TEST_ASSERTION((void *)TEST_ALLOCATOR_MAGIC_NUMBER == state, "Magic number mismatch.");
    return realloc(ptr, new_size);
}

static void wrap_free(void *state, void *ptr)
{
    TEST_ASSERTION((void *)TEST_ALLOCATOR_MAGIC_NUMBER == state, "Magic number mismatch.");
    free(ptr);
}

const allocator_callbacks TEST_ALLOCATOR = {
    .alloc = wrap_malloc,
    .free = wrap_free,
    .realloc = wrap_realloc,
    .state = (void *)TEST_ALLOCATOR_MAGIC_NUMBER,
};

// Initialize the PRNG state with a seed (non-zero recommended)
void test_prng_seed(test_prng_t *rng, uint32_t seed)
{
    if (seed == 0)
        seed = 1; // avoid zero seed which can degenerate sequence
    rng->state = seed;
}

// Generate the next random unsigned 32-bit integer
uint32_t test_prng_next_uint(test_prng_t *rng)
{
    // Constants from Numerical Recipes LCG
    rng->state = 1664525 * rng->state + 1013904223;
    return rng->state;
}

// Generate the next random double in [0, 1)
double test_prng_next_double(test_prng_t *rng)
{
    uint32_t val = test_prng_next_uint(rng);
    // Divide by 2^32 to get uniform double in [0,1)
    return (double)val / 4294967296.0;
}
