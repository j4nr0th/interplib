#include "../../src/operations/multidim_iteration.h"
#include "../common/common.h"

int main(const int argc, const char *argv[argc])
{
    TEST_ASSERTION(argc == 3, "Two arguments is required, but %u were given", (unsigned)(argc - 1));
    char *end_ptr;
    const long d1 = strtol(argv[1], &end_ptr, 10);
    TEST_ASSERTION(end_ptr != argv[1], "Argument 1 is not a valid integer");
    TEST_ASSERTION(d1 > 0, "Argument 1 is not positive");
    const long d2 = strtol(argv[2], &end_ptr, 10);
    TEST_ASSERTION(end_ptr != argv[2], "Argument 2 is not a valid integer");
    TEST_ASSERTION(d2 > 0, "Argument 2 is not positive");

    multidim_iterator_t *const iter = malloc(multidim_iterator_needed_memory(2));
    TEST_ASSERTION(iter != NULL, "Failed to allocate memory for iterator");

    multidim_iterator_init(iter, 2, (const size_t[]){d1, d2});

    // Test forward iteration
    multidim_iterator_set_to_start(iter);
    // Check that flat iteration without any steps works well
    for (size_t i = 0; i < d1 * d2; ++i)
    {
        TEST_ASSERTION(multidim_iterator_get_flat_index(iter) == i, "Unexpected index");
        multidim_iterator_advance(iter, 1, 1);
    }
    TEST_ASSERTION(multidim_iterator_is_at_end(iter), "Iterator is not done");
    TEST_ASSERTION(multidim_iterator_get_flat_index(iter) == d1 * d2, "End index was not what was expected");

    // Check that iteration over the first dimension
    multidim_iterator_set_to_start(iter);
    for (size_t i = 0; i < d1; ++i)
    {
        TEST_ASSERTION(multidim_iterator_get_flat_index(iter) == i * d2, "Unexpected index");
        multidim_iterator_advance(iter, 0, 1);
    }
    TEST_ASSERTION(multidim_iterator_is_at_end(iter), "Iterator is not done");
    TEST_ASSERTION(multidim_iterator_get_flat_index(iter) == d1 * d2, "End index was not what was expected");

    // Check that different stride length works
    for (size_t stride = 1; stride < 25; ++stride)
    {
        multidim_iterator_set_to_start(iter);
        TEST_ASSERTION(multidim_iterator_get_flat_index(iter) == 0, "Iterator did not correctly reset");
        unsigned last_index = 0;
        while (last_index < d1 * d2)
        {
            TEST_ASSERTION(multidim_iterator_get_flat_index(iter) == last_index, "Unexpected index");
            last_index += stride;
            multidim_iterator_advance(iter, 1, stride);
        }
        TEST_ASSERTION(multidim_iterator_is_at_end(iter), "Iterator is not done");
        TEST_ASSERTION(multidim_iterator_get_flat_index(iter) == d1 * d2, "End index was not what was expected");
    }

    // Test backward iteration
    multidim_iterator_set_to_end(iter);
    // Check that flat iteration without any steps works well
    for (size_t i = 0; i < d1 * d2; ++i)
    {
        TEST_ASSERTION(multidim_iterator_get_flat_index(iter) == d1 * d2 - i, "Unexpected index");
        multidim_iterator_recede(iter, 1, 1);
    }
    TEST_ASSERTION(multidim_iterator_is_at_start(iter), "Iterator is not at the start");
    TEST_ASSERTION(multidim_iterator_get_flat_index(iter) == 0, "Start index was not what was expected");

    // Check that iteration over the first dimension
    multidim_iterator_set_to_end(iter);
    for (size_t i = 0; i < d1; ++i)
    {
        TEST_ASSERTION(multidim_iterator_get_flat_index(iter) == d1 * d2 - i * d2, "Unexpected index");
        multidim_iterator_recede(iter, 0, 1);
    }
    TEST_ASSERTION(multidim_iterator_is_at_start(iter), "Iterator is not at the start");
    TEST_ASSERTION(multidim_iterator_get_flat_index(iter) == 0, "Start index was not what was expected");

    // Check that different stride length works
    for (size_t stride = 1; stride < 25; ++stride)
    {
        multidim_iterator_set_to_end(iter);
        TEST_ASSERTION(multidim_iterator_get_flat_index(iter) == d1 * d2, "Iterator did not correctly reset");
        unsigned last_index = 0;
        while (last_index < d1 * d2)
        {
            TEST_ASSERTION(multidim_iterator_get_flat_index(iter) == d1 * d2 - last_index, "Unexpected index");
            last_index += stride;
            multidim_iterator_recede(iter, 1, stride);
        }
        TEST_ASSERTION(multidim_iterator_is_at_start(iter), "Iterator is not at the start");
        TEST_ASSERTION(multidim_iterator_get_flat_index(iter) == 0, "Start index was not what was expected");
    }

    free(iter);
    return 0;
}
