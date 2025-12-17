#include "../../src/operations/permutations.h"
#include "../common/common.h"

#include <stddef.h>
#include <string.h>

// static void print_permutation(const permutation_iterator_t *p)
// {
//     const unsigned char *const val = permutation_iterator_current(p);
//     printf("%hhd", val[0]);
//     for (unsigned i = 1; i < p->r; ++i)
//         printf(" %hhd", val[i]);
// }

static int are_permutations_equal(const unsigned r, const unsigned char a[static r], const unsigned char b[static r])
{
    for (unsigned i = 0; i < r; ++i)
    {
        if (a[i] != b[i])
            return 0;
    }
    return 1;
}

static void test_permutations(const unsigned char n, const unsigned char r)
{
    printf("Testing n: %u r: %u\n", (unsigned)n, (unsigned)r);
    TEST_ASSERTION(n >= r,
                   "Number of elements must be greater than or equal to the number of elements per permutation.");
    permutation_iterator_t *const p = malloc(permutation_iterator_required_memory(n, r));
    TEST_ASSERTION(p, "Failed to allocate permutation iterator.");
    permutation_iterator_init(p, n, r);
    const unsigned total_permutations = permutation_iterator_total_count(p);
    unsigned cnt = 0;
    unsigned char *const previous_permutations = malloc((size_t)r * total_permutations);
    TEST_ASSERTION(previous_permutations, "Failed to allocate memory for previous permutations.");
    while (!permutation_iterator_is_done(p))
    {
        // Copy the current iteration to the buffer
        const unsigned char *const current_permutation = permutation_iterator_current(p);
        memcpy(previous_permutations + (size_t)cnt * r, current_permutation, r);
        // printf("Permutation %u: ", cnt + 1);
        // print_permutation(p);
        // printf("\n");

        // Check the current iteration does not repeat!
        for (unsigned i = 0; i < cnt; ++i)
        {
            TEST_ASSERTION(!are_permutations_equal(r, current_permutation, previous_permutations + (size_t)(i * r)),
                           "Permutation should not repeat, but permutation %u and %u are the same.", cnt + 1, i + 1);
        }

        cnt += 1;
        permutation_iterator_next(p);
    }

    TEST_ASSERTION(cnt == total_permutations, "Wrong number of permutations generated (expected %u, but only got %u).",
                   total_permutations, cnt);

    free(previous_permutations);
    free(p);
    printf("Finished n: %u r: %u\n", (unsigned)n, (unsigned)r);
}

int main(void)
{
    // test_permutations(3, 0);
    test_permutations(3, 1);
    test_permutations(3, 2);
    test_permutations(5, 2);
    test_permutations(5, 5);
    test_permutations(10, 3);
}
