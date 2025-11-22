#include "../../src/operations/nd_iteration.h"
#include "../common/common.h"

int main(const int argc, const char *argv[argc])
{
    TEST_ASSERTION(argc == 2, "One argument is required, but %u were given", (unsigned)(argc - 1));
    char *end_ptr;
    const long v = strtol(argv[1], &end_ptr, 10);
    TEST_ASSERTION(end_ptr != argv[1], "Argument is not a valid integer");
    TEST_ASSERTION(v > 0, "Argument is not positive");

    nd_iterator_t *const iter = malloc(nd_iterator_needed_memory(1));
    TEST_ASSERTION(iter != NULL, "Failed to allocate memory for iterator");

    nd_iterator_init(iter, 1, (const size_t[]){v});

    // Check that iteration without any steps works well
    for (size_t i = 0; i < v; ++i)
    {
        TEST_ASSERTION(nd_iterator_get_flat_index(iter) == i, "Unexpected index");
        nd_iterator_advance(iter, 0, 1);
    }
    TEST_ASSERTION(nd_iterator_is_done(iter), "Iterator is not done");
    TEST_ASSERTION(nd_iterator_get_flat_index(iter) == v, "End index was not what was expected");

    // Check that different stride length works
    for (size_t stride = 1; stride < 25; ++stride)
    {
        nd_iterator_restart(iter);
        TEST_ASSERTION(nd_iterator_get_flat_index(iter) == 0, "Iterator did not correctly reset");
        unsigned last_index = 0;
        while (last_index < v)
        {
            TEST_ASSERTION(nd_iterator_get_flat_index(iter) == last_index, "Unexpected index");
            last_index += stride;
            nd_iterator_advance(iter, 0, stride);
        }
        TEST_ASSERTION(nd_iterator_is_done(iter), "Iterator is not done");
        TEST_ASSERTION(nd_iterator_get_flat_index(iter) == v, "End index was not what was expected");
    }

    free(iter);
    return 0;
}
