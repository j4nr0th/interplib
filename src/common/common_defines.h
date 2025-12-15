#ifndef COMMON_DEFINES_H
#define COMMON_DEFINES_H

#ifdef __GNUC__
#define INTERPLIB_INTERNAL __attribute__((visibility("hidden")))
#define INTERPLIB_EXTERNAL __attribute__((visibility("default")))
#ifdef _DEBUG
#define INTERPLIB_BREAK __builtin_trap()
#endif
#define INTERPLIB_ARRAY_ARG(arr, sz) arr[sz]

#define INTERPLIB_EXPECT_CONDITION(x) (__builtin_expect(x, 1))

#endif

#ifndef ASSERT
#ifdef INTERPLIB_ASSERTS
/**
 * @brief is a macro, which tests a condition and only evaluates it once. If it
 * is false, then it is reported to stderr and the program will terminate.
 *
 * @note ASSERT does all this only when building in Debug mode. For Release
 * configuration, the macro is replaced with a compiler-specific assume
 * directive, or a zero if that is not known for the specific compiler used.
 */
#include <stdio.h>
#include <stdlib.h>
#ifndef INTERPLIB_BREAK
#define INTERPLIB_BREAK exit(EXIT_FAILURE)
#endif
#define ASSERT(condition, message, ...)                                                                                \
    ((condition) ? (void)0                                                                                             \
                 : (fprintf(stderr, "%s:%d: %s: Assertion '%s' failed - " message "\n", __FILE__, __LINE__, __func__,  \
                            #condition __VA_OPT__(, ) __VA_ARGS__),                                                    \
                    INTERPLIB_BREAK))
#else
#ifndef ASSERT
#define ASSERT(condition, message) 0
#endif
#endif
#endif

#ifdef __GNUC__
#define ASSUME(condition, message) __attribute__((assume(condition)))
#endif

#ifndef ASSUME
#define ASSUME(condition, message) ASSUME(condition, message)
#endif

#ifndef INTERPLIB_INTERNAL
#define INTERPLIB_INTERNAL
#endif

#ifndef INTERPLIB_EXTERNAL
#define INTERPLIB_EXTERNAL
#endif

#ifndef INTERPLIB_ARRAY_ARG
#define INTERPLIB_ARRAY_ARG(arr, sz) *arr
#endif

#endif // COMMON_DEFINES_H
