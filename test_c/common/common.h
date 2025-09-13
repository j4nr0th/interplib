//
// Created by jan on 2025-09-07.
//

#ifndef INTERPLIB_COMMON_H
#define INTERPLIB_COMMON_H
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>

#include "../../src/common/allocator.h"
#include "../../src/common/error.h"

#include <math.h>

static void failed_assertion(const char *file, const int line, const char *function, const char *expr, const char *msg,
                             ...)
{
    char buffer[1024];
    va_list args;
    va_start(args, msg);
    vsnprintf(buffer, sizeof(buffer), msg, args);
    va_end(args);
    fprintf(stderr, "%s:%d - %s (Assertion failed: %s): %s\n", file, line, function, expr, buffer);
#ifdef __GNUC__
    __builtin_trap();
#endif

    exit(EXIT_FAILURE);
}

static void test_interp_result(const char *file, const int line, const char *function, const char *expr,
                               const interp_result_t res)
{
    if (res == INTERP_SUCCESS)
        return;
    fprintf(stderr, "%s:%d - %s (Assertion failed: %s): %s (%s)\n", file, line, function, expr, interp_error_str(res),
            interp_error_msg(res));
#ifdef __GNUC__
    __builtin_trap();
#endif

    exit(EXIT_FAILURE);
}

#define TEST_ASSERTION(expr, msg, ...)                                                                                 \
    ((expr) ? (void)0 : failed_assertion(__FILE__, __LINE__, __func__, #expr, msg __VA_OPT__(, ) __VA_ARGS__))

#define TEST_INTERP_RESULT(expr) test_interp_result(__FILE__, __LINE__, __func__, #expr, expr)

static void test_numbers_close(const char *file, const int line, const char *function, const double x, const double y,
                               const double atol, const double rtol)
{
    TEST_ASSERTION(atol >= 0, "Absolute tolerance must be non-negative.");
    TEST_ASSERTION(rtol >= 0, "Relative tolerance must be non-negative.");
    const double mag_x = fabs(x);
    const double mag_y = fabs(y);
    const double max_mag = mag_x > mag_y ? mag_x : mag_y;
    const double relative_tol = rtol * max_mag;
    const double tol = relative_tol > atol ? relative_tol : atol;
    if (fabs(x - y) > tol)
        failed_assertion(file, line, function, "fabs(x - y) <= tol", "Numbers %g and %g are not close enough.", x, y);
}

#define TEST_NUMBERS_CLOSE(x, y, atol, rtol) test_numbers_close(__FILE__, __LINE__, __func__, x, y, atol, rtol)

extern const allocator_callbacks TEST_ALLOCATOR;

#include <stdint.h>

typedef struct
{
    uint32_t state;
} test_prng_t;

INTERPLIB_INTERNAL
void test_prng_seed(test_prng_t *rng, uint32_t seed);

INTERPLIB_INTERNAL
uint32_t test_prng_next_uint(test_prng_t *rng);

INTERPLIB_INTERNAL
double test_prng_next_double(test_prng_t *rng);

#endif // INTERPLIB_COMMON_H
