#include "../../src/operations/multidim_iteration.h"
#include "../common/common.h"

int main(const int argc, const char *argv[argc])
{
    TEST_ASSERTION(argc == 2, "One argument is required, but %u were given", (unsigned)(argc - 1));
    char *end_ptr;
    const long v = strtol(argv[1], &end_ptr, 10);
    TEST_ASSERTION(end_ptr != argv[1], "Argument is not a valid integer");
    TEST_ASSERTION(v > 0, "Argument is not positive");

    multidim_iterator_t *const iter = malloc(multidim_iterator_needed_memory(1));
    TEST_ASSERTION(iter != NULL, "Failed to allocate memory for iterator");

    multidim_iterator_init(iter, 1, (const size_t[]){v});

    // Forward iteration tests

    multidim_iterator_set_to_start(iter);
    // Check that iteration without any steps works well
    for (size_t i = 0; i < v; ++i)
    {
        TEST_ASSERTION(multidim_iterator_get_flat_index(iter) == i, "Unexpected index");
        multidim_iterator_advance(iter, 0, 1);
    }
    TEST_ASSERTION(multidim_iterator_is_at_end(iter), "Iterator is not done");
    TEST_ASSERTION(multidim_iterator_get_flat_index(iter) == v, "End index was not what was expected");

    // Check that different stride length works
    for (size_t stride = 1; stride < 25; ++stride)
    {
        multidim_iterator_set_to_start(iter);
        TEST_ASSERTION(multidim_iterator_get_flat_index(iter) == 0, "Iterator did not correctly reset");
        unsigned last_index = 0;
        while (last_index < v)
        {
            TEST_ASSERTION(multidim_iterator_get_flat_index(iter) == last_index, "Unexpected index");
            last_index += stride;
            multidim_iterator_advance(iter, 0, stride);
        }
        TEST_ASSERTION(multidim_iterator_is_at_end(iter), "Iterator is not done");
        TEST_ASSERTION(multidim_iterator_get_flat_index(iter) == v, "End index was not what was expected");
    }

    // Backward iteration steps
    multidim_iterator_set_to_end(iter);
    // Check that iteration without any steps works well
    for (size_t i = 0; i < v; ++i)
    {
        TEST_ASSERTION(multidim_iterator_get_flat_index(iter) == v - i, "Unexpected index");
        multidim_iterator_recede(iter, 0, 1);
    }
    TEST_ASSERTION(multidim_iterator_is_at_start(iter), "Iterator is not at the start");
    TEST_ASSERTION(multidim_iterator_get_flat_index(iter) == 0, "Start index was not what was expected");

    // Check that different stride length works
    for (size_t stride = 1; stride < 25; ++stride)
    {
        multidim_iterator_set_to_end(iter);
        TEST_ASSERTION(multidim_iterator_get_flat_index(iter) == v, "Iterator did not correctly reset");
        unsigned last_index = 0;
        while (last_index < v)
        {
            TEST_ASSERTION(multidim_iterator_get_flat_index(iter) == v - last_index, "Unexpected index");
            last_index += stride;
            multidim_iterator_recede(iter, 0, stride);
        }
        TEST_ASSERTION(multidim_iterator_is_at_start(iter), "Iterator is not at the start");
        TEST_ASSERTION(multidim_iterator_get_flat_index(iter) == 0, "Start index was not what was expected");
    }

    free(iter);
    return 0;
}
