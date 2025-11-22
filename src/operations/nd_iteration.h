#ifndef INTERPLIB_ND_ITERATION_H
#define INTERPLIB_ND_ITERATION_H

#include "../common/common_defines.h"
#include <stddef.h>

typedef struct
{
    size_t ndims;
    size_t dims_and_offsets[];
} nd_iterator_t;

/**
 * Calculates the amount of memory needed to store an nd_iterator_t structure
 * with the specified number of dimensions.
 *
 * @param ndims The number of dimensions for the iterator.
 * @return The size in bytes of memory required to initialize the nd_iterator_t
 *         structure with the given number of dimensions.
 */
size_t nd_iterator_needed_memory(size_t ndims);

/**
 * Initializes an nd_iterator_t structure with the specified number of dimensions
 * and dimension sizes. This function sets up the necessary internal data for
 * an iterator to traverse multidimensional data structures.
 *
 * @param this Pointer to the nd_iterator_t instance to be initialized.
 * @param ndims The number of dimensions of the iterator.
 * @param dims An array of size `ndims` specifying the size of each dimension.
 */
void nd_iterator_init(nd_iterator_t *this, size_t ndims, const size_t INTERPLIB_ARRAY_ARG(dims, static ndims));

void nd_iterator_restart(nd_iterator_t *this);

void nd_iterator_advance(nd_iterator_t *this, size_t dim, size_t step);

void nd_iterator_recede(nd_iterator_t *this, size_t dim, size_t step);

int nd_iterator_is_done(const nd_iterator_t *this);

size_t nd_iterator_get_flat_index(const nd_iterator_t *this);

#endif // INTERPLIB_ND_ITERATION_H
