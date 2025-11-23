#ifndef INTERPLIB_ND_ITERATION_H
#define INTERPLIB_ND_ITERATION_H

#include "../common/common_defines.h"
#include <stddef.h>

typedef struct
{
    size_t ndims;
    size_t dims_and_offsets[];
} multidim_iterator_t;

/**
 * Calculates the amount of memory needed to store a multidim_iterator_t structure
 * with the specified number of dimensions.
 *
 * @param ndims The number of dimensions for the iterator.
 * @return The size in bytes of memory required to initialize the multidim_iterator_t
 *         structure with the given number of dimensions.
 */
size_t multidim_iterator_needed_memory(size_t ndims);

/**
 * Initializes a multidim_iterator_t structure with the specified number of dimensions
 * and dimension sizes. This function sets up the necessary internal data for
 * an iterator to traverse multidimensional data structures.
 *
 * @param this Pointer to the multidim_iterator_t instance to be initialized.
 * @param ndims The number of dimensions of the iterator.
 * @param dims An array of size `ndims` specifying the size of each dimension.
 */
void multidim_iterator_init(multidim_iterator_t *this, size_t ndims,
                            const size_t INTERPLIB_ARRAY_ARG(dims, static ndims));
/**
 * Alternative to `multidim_iterator_init` to initalize the iterator by iteratevly
 * initializing the dimensions. Note that this function has to be called for all
 * dimensions from the lowest to the highest in order.
 *
 * @param this Pointer to the multidim_iterator_t instance to be initialized.
 * @param dim Dimension to initialize.
 * @param size Size of the dimension.
 */
void multidim_iterator_init_dim(multidim_iterator_t *this, size_t dim, size_t size);

/**
 * Resets the offsets of the multidimensional iterator to their starting values.
 * This function repositions the iterator to the beginning of the multidimensional space.
 *
 * @param this A pointer to the multidim_iterator_t structure to be reset.
 *             The structure must be initialized before calling this function.
 */
void multidim_iterator_set_to_start(multidim_iterator_t *this);

/**
 * Sets the iterator to its end position based on its dimensions.
 *
 * This function modifies the internal structure of the given iterator to place
 * it at the end of its range for all dimensions. The final dimension is set
 * to its total size, while all preceding dimensions are set to their maximum
 * index (size - 1).
 *
 * @param this A pointer to the iterator object whose position is to be set
 *             to the end.
 */
void multidim_iterator_set_to_end(multidim_iterator_t *this);

/**
 * Advances the position of the multidimensional iterator by a specified step
 * in a specified dimension. If the step exceeds the bounds of the dimension,
 * it carries over to the next dimension, recursively handling overflows.
 *
 * @param this A pointer to the multidimensional iterator to update.
 * @param dim The dimension in which the iterator is to be advanced.
 * @param step The number of steps to advance the iterator in the specified dimension.
 */
void multidim_iterator_advance(multidim_iterator_t *this, size_t dim, size_t step);

/**
 * Moves the iterator backward by a specified step in the given dimension.
 *
 * If the specified step exceeds the current offset in the target dimension,
 * the iterator will wrap around in higher dimensions as necessary. If a wrap
 * exceeds the limits of the iterator, all offsets are reset to the start.
 *
 * @param this Pointer to the multidim_iterator_t structure to be modified.
 * @param dim The dimension in which to apply the backward step. Must be
 *            less than the number of dimensions in the iterator.
 * @param step The number of steps to move backward in the specified dimension.
 */
void multidim_iterator_recede(multidim_iterator_t *this, size_t dim, size_t step);

/**
 * Checks if the multidim_iterator_t structure is at its starting position.
 *
 * @param this Pointer to the multidim_iterator_t structure to check.
 * @return 1 if the iterator is at its starting position, otherwise 0.
 */
int multidim_iterator_is_at_start(const multidim_iterator_t *this);

/**
 * Checks whether the iterator has reached the end of the multidimensional space
 * it is iterating over.
 *
 * @param this A pointer to the multidim_iterator_t structure representing the iterator.
 * @return 1 if the iterator is at the end of its iteration space; otherwise, 0.
 */
int multidim_iterator_is_at_end(const multidim_iterator_t *this);

/**
 * Computes the flat index of the current position of the iterator in a
 * multidimensional iteration space.
 *
 * @param this A pointer to the multidim_iterator_t structure representing the
 *             iterator.
 * @return The flat index corresponding to the current position of the iterator.
 */
size_t multidim_iterator_get_flat_index(const multidim_iterator_t *this);

/**
 * Gets the number of dimensions managed by the multidim_iterator_t structure.
 *
 * @param this A pointer to the multidim_iterator_t structure whose number of
 *             dimensions is to be retrieved.
 * @return The number of dimensions associated with the given multidim_iterator_t
 *         structure.
 */
size_t multidim_iterator_get_ndims(const multidim_iterator_t *this);

/**
 * Retrieves a pointer to the dimension array of the specified ND iterator.
 *
 * @param this A pointer to the `multidim_iterator_t` structure.
 * @return A pointer to the array of dimensions and offsets stored within the
 *         specified ND iterator.
 */
const size_t *multidim_iterator_dims(const multidim_iterator_t *this);

/**
 * Retrieves a pointer to the offsets of a multidim_iterator_t structure.
 *
 * @param this A pointer to the multidim_iterator_t structure.
 * @return A pointer to the array of offsets corresponding to the iterator's dimensions.
 */
const size_t *multidim_iterator_offsets(const multidim_iterator_t *this);

size_t multidim_iterator_get_dim(const multidim_iterator_t *this, size_t dim);

size_t multidim_iterator_get_offset(const multidim_iterator_t *this, size_t dim);

size_t multidim_iterator_total_size(const multidim_iterator_t *this);

#endif // INTERPLIB_ND_ITERATION_H
