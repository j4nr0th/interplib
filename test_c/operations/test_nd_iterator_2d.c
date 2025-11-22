#include "../../src/operations/nd_iteration.h"
#include "../common/common.h"

int main(const int argc, const char *argv[argc])
{
    TEST_ASSERTION(argc == 3, "Two argument is required, but %u were given", (unsigned)(argc - 1));
    char *end_ptr;
    const long d1 = strtol(argv[1], &end_ptr, 10);
    TEST_ASSERTION(end_ptr != argv[1], "Argument 1 is not a valid integer");
    TEST_ASSERTION(d1 > 0, "Argument 1 is not positive");
    const long d2 = strtol(argv[2], &end_ptr, 10);
    TEST_ASSERTION(end_ptr != argv[2], "Argument 2 is not a valid integer");
    TEST_ASSERTION(d2 > 0, "Argument 2 is not positive");

    nd_iterator_t *const iter = malloc(nd_iterator_needed_memory(2));
    TEST_ASSERTION(iter != NULL, "Failed to allocate memory for iterator");

    nd_iterator_init(iter, 2, (const size_t[]){d1, d2});

    // Check that flat iteration without any steps works well
    for (size_t i = 0; i < d1 * d2; ++i)
    {
        TEST_ASSERTION(nd_iterator_get_flat_index(iter) == i, "Unexpected index");
        nd_iterator_advance(iter, 1, 1);
    }
    TEST_ASSERTION(nd_iterator_is_done(iter), "Iterator is not done");
    TEST_ASSERTION(nd_iterator_get_flat_index(iter) == d1 * d2, "End index was not what was expected");

    // Check that iteration over the first dimension
    nd_iterator_restart(iter);
    for (size_t i = 0; i < d1; ++i)
    {
        TEST_ASSERTION(nd_iterator_get_flat_index(iter) == i * d2, "Unexpected index");
        nd_iterator_advance(iter, 0, 1);
    }
    TEST_ASSERTION(nd_iterator_is_done(iter), "Iterator is not done");
    TEST_ASSERTION(nd_iterator_get_flat_index(iter) == d1 * d2, "End index was not what was expected");

    // Check that different stride length works
    for (size_t stride = 1; stride < 25; ++stride)
    {
        nd_iterator_restart(iter);
        TEST_ASSERTION(nd_iterator_get_flat_index(iter) == 0, "Iterator did not correctly reset");
        unsigned last_index = 0;
        while (last_index < d1 * d2)
        {
            TEST_ASSERTION(nd_iterator_get_flat_index(iter) == last_index, "Unexpected index");
            last_index += stride;
            nd_iterator_advance(iter, 1, stride);
        }
        TEST_ASSERTION(nd_iterator_is_done(iter), "Iterator is not done");
        TEST_ASSERTION(nd_iterator_get_flat_index(iter) == d1 * d2, "End index was not what was expected");
    }

    free(iter);
    return 0;
}
