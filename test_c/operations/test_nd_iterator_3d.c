#include "../../src/operations/nd_iteration.h"
#include "../common/common.h"

int main(const int argc, const char *argv[argc])
{
    TEST_ASSERTION(argc == 4, "Three argument is required, but %u were given", (unsigned)(argc - 1));
    char *end_ptr;
    const long d1 = strtol(argv[1], &end_ptr, 10);
    TEST_ASSERTION(end_ptr != argv[1], "Argument 1 is not a valid integer");
    TEST_ASSERTION(d1 > 0, "Argument 1 is not positive");
    const long d2 = strtol(argv[2], &end_ptr, 10);
    TEST_ASSERTION(end_ptr != argv[2], "Argument 2 is not a valid integer");
    TEST_ASSERTION(d2 > 0, "Argument 2 is not positive");
    const long d3 = strtol(argv[3], &end_ptr, 10);
    TEST_ASSERTION(end_ptr != argv[3], "Argument 3 is not a valid integer");
    TEST_ASSERTION(d3 > 0, "Argument 3 is not positive");

    nd_iterator_t *const iter = malloc(nd_iterator_needed_memory(3));
    TEST_ASSERTION(iter != NULL, "Failed to allocate memory for iterator");

    nd_iterator_init(iter, 3, (const size_t[]){d1, d2, d3});

    // Test forward iteration
    nd_iterator_set_to_start(iter);
    // Check that flat iteration without any steps works well
    for (size_t i = 0; i < d1 * d2 * d3; ++i)
    {
        TEST_ASSERTION(nd_iterator_get_flat_index(iter) == i, "Unexpected index");
        nd_iterator_advance(iter, 2, 1);
    }
    TEST_ASSERTION(nd_iterator_is_at_end(iter), "Iterator is not done");
    TEST_ASSERTION(nd_iterator_get_flat_index(iter) == d1 * d2 * d3, "End index was not what was expected");

    // Check that iteration over the second dimension
    nd_iterator_set_to_start(iter);
    for (size_t i = 0; i < d1 * d2; ++i)
    {
        TEST_ASSERTION(nd_iterator_get_flat_index(iter) == i * d3, "Unexpected index");
        nd_iterator_advance(iter, 1, 1);
    }
    TEST_ASSERTION(nd_iterator_is_at_end(iter), "Iterator is not done");
    TEST_ASSERTION(nd_iterator_get_flat_index(iter) == d1 * d2 * d3, "End index was not what was expected");

    // Check that iteration over the first dimension
    nd_iterator_set_to_start(iter);
    for (size_t i = 0; i < d1; ++i)
    {
        TEST_ASSERTION(nd_iterator_get_flat_index(iter) == i * d2 * d3, "Unexpected index");
        nd_iterator_advance(iter, 0, 1);
    }
    TEST_ASSERTION(nd_iterator_is_at_end(iter), "Iterator is not done");
    TEST_ASSERTION(nd_iterator_get_flat_index(iter) == d1 * d2 * d3, "End index was not what was expected");

    // Check that different stride length works
    for (size_t stride = 1; stride < 25; ++stride)
    {
        nd_iterator_set_to_start(iter);
        TEST_ASSERTION(nd_iterator_get_flat_index(iter) == 0, "Iterator did not correctly reset");
        unsigned last_index = 0;
        while (last_index < d1 * d2 * d3)
        {
            TEST_ASSERTION(nd_iterator_get_flat_index(iter) == last_index, "Unexpected index");
            last_index += stride;
            nd_iterator_advance(iter, 2, stride);
        }
        TEST_ASSERTION(nd_iterator_is_at_end(iter), "Iterator is not done");
        TEST_ASSERTION(nd_iterator_get_flat_index(iter) == d1 * d2 * d3, "End index was not what was expected");
    }

    // Test backward iteration
    nd_iterator_set_to_end(iter);
    // Check that flat iteration without any steps works well
    for (size_t i = 0; i < d1 * d2 * d3; ++i)
    {
        TEST_ASSERTION(nd_iterator_get_flat_index(iter) == d1 * d2 * d3 - i, "Unexpected index");
        nd_iterator_recede(iter, 2, 1);
    }
    TEST_ASSERTION(nd_iterator_is_at_start(iter), "Iterator is not at the start");
    TEST_ASSERTION(nd_iterator_get_flat_index(iter) == 0, "Start index was not what was expected");

    // Check that iteration over the second dimension
    nd_iterator_set_to_end(iter);
    for (size_t i = 0; i < d1 * d2; ++i)
    {
        TEST_ASSERTION(nd_iterator_get_flat_index(iter) == d1 * d2 * d3 - i * d3, "Unexpected index");
        nd_iterator_recede(iter, 1, 1);
    }
    TEST_ASSERTION(nd_iterator_is_at_start(iter), "Iterator is not at the start");
    TEST_ASSERTION(nd_iterator_get_flat_index(iter) == 0, "Start index was not what was expected");

    // Check that iteration over the first dimension
    nd_iterator_set_to_end(iter);
    for (size_t i = 0; i < d1; ++i)
    {
        TEST_ASSERTION(nd_iterator_get_flat_index(iter) == d1 * d2 * d3 - i * d2 * d3, "Unexpected index");
        nd_iterator_recede(iter, 0, 1);
    }
    TEST_ASSERTION(nd_iterator_is_at_start(iter), "Iterator is not at the start");
    TEST_ASSERTION(nd_iterator_get_flat_index(iter) == 0, "Start index was not what was expected");

    // Check that different stride length works
    for (size_t stride = 1; stride < 25; ++stride)
    {
        nd_iterator_set_to_end(iter);
        TEST_ASSERTION(nd_iterator_get_flat_index(iter) == d1 * d2 * d3, "Iterator did not correctly reset");
        unsigned last_index = 0;
        while (last_index < d1 * d2 * d3)
        {
            TEST_ASSERTION(nd_iterator_get_flat_index(iter) == d1 * d2 * d3 - last_index, "Unexpected index");
            last_index += stride;
            nd_iterator_recede(iter, 2, stride);
        }
        TEST_ASSERTION(nd_iterator_is_at_start(iter), "Iterator is not at the start");
        TEST_ASSERTION(nd_iterator_get_flat_index(iter) == 0, "Start index was not what was expected");
    }

    free(iter);
    return 0;
}
