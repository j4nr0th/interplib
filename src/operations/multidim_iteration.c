
#include "multidim_iteration.h"

static inline size_t *multidim_iterator_dims_ptr(multidim_iterator_t *this)
{
    return this->dims_and_offsets;
}

static inline size_t *multidim_iterator_offsets_ptr(multidim_iterator_t *this)
{
    return this->dims_and_offsets + this->ndims;
}

static inline const size_t *multidim_iterator_dims_const_ptr(const multidim_iterator_t *this)
{
    return this->dims_and_offsets;
}

static inline const size_t *multidim_iterator_offsets_const_ptr(const multidim_iterator_t *this)
{
    return this->dims_and_offsets + this->ndims;
}

size_t multidim_iterator_needed_memory(const size_t ndims)
{
    return sizeof(multidim_iterator_t) + 2 * ndims * sizeof(*((multidim_iterator_t *)0xB00B5)->dims_and_offsets);
}

void multidim_iterator_init(multidim_iterator_t *const this, const size_t ndims,
                            const size_t INTERPLIB_ARRAY_ARG(dims, const static ndims))
{
    this->ndims = ndims;
    size_t *const iter_dims = multidim_iterator_dims_ptr(this);
    size_t *const iter_offsets = multidim_iterator_offsets_ptr(this);
    for (size_t i = 0; i < ndims; ++i)
    {
        iter_dims[i] = dims[i];
        iter_offsets[i] = 0;
    }
}
void multidim_iterator_init_dim(multidim_iterator_t *this, size_t dim, size_t size)
{
    this->ndims = dim + 1;
    multidim_iterator_dims_ptr(this)[dim] = size;
    multidim_iterator_offsets_ptr(this)[dim] = 0;
}

void multidim_iterator_set_to_start(multidim_iterator_t *this)
{
    // Reset offsets
    size_t *const iter_offsets = multidim_iterator_offsets_ptr(this);
    for (size_t i = 0; i < this->ndims; ++i)
    {
        iter_offsets[i] = 0;
    }
}

void multidim_iterator_set_to_end(multidim_iterator_t *this)
{
    size_t *const iter_offsets = multidim_iterator_offsets_ptr(this);
    const size_t *const iter_dims = multidim_iterator_dims_const_ptr(this);
    for (size_t i = 0; i < this->ndims - 1; ++i)
    {
        iter_offsets[i] = iter_dims[i] - 1;
    }
    iter_offsets[this->ndims - 1] = iter_dims[this->ndims - 1];
}

void multidim_iterator_advance(multidim_iterator_t *const this, size_t dim, size_t step)
{
    ASSERT(dim < this->ndims, "Dimension out of bounds");
    const size_t *const iter_dims = multidim_iterator_dims_const_ptr(this);
    size_t *const iter_offsets = multidim_iterator_offsets_ptr(this);

compute_offset:;
    size_t new_offset = iter_offsets[dim] + step;
    if (new_offset < iter_dims[dim])
    {
        // the step is fine
        iter_offsets[dim] = new_offset;
        return;
    }

    // We would step out of bounds
    if (dim == 0)
    {
        multidim_iterator_set_to_end(this);
        return;
    }

    // We should step the previous dimension as well

    const size_t stride_lower_dim = new_offset / iter_dims[dim];
    new_offset %= iter_dims[dim];
    iter_offsets[dim] = new_offset;

    // Tail recursion (but with goto)
    // multidim_iterator_advance(this, dim - 1, stride_lower_dim);
    dim = dim - 1;
    step = stride_lower_dim;

    goto compute_offset;
}

void multidim_iterator_recede(multidim_iterator_t *this, size_t dim, size_t step)
{
    ASSERT(dim < this->ndims, "Dimension out of bounds");
    const size_t *const iter_dims = multidim_iterator_dims_const_ptr(this);
    size_t *const iter_offsets = multidim_iterator_offsets_ptr(this);

compute_offset:;
    if (step <= iter_offsets[dim])
    {
        // the step is fine
        iter_offsets[dim] -= step;
        return;
    }
    // We would step out of bounds
    if (dim == 0)
    {
        multidim_iterator_set_to_start(this);
        return;
    }

    const size_t remaining = step - iter_offsets[dim];
    // Ceil-divide
    const size_t required_steps = (remaining + iter_dims[dim] - 1) / iter_dims[dim];

    iter_offsets[dim] = iter_offsets[dim] + required_steps * iter_dims[dim] - step;
    // Tail recursion (but with goto)
    // multidim_iterator_recede(this, dim - 1, required_steps);
    dim -= 1;
    step = required_steps;

    goto compute_offset;
}

int multidim_iterator_is_at_start(const multidim_iterator_t *this)
{
    const size_t *const iter_offsets = multidim_iterator_offsets_const_ptr(this);
    for (unsigned i = 0; i < this->ndims; ++i)
    {
        if (iter_offsets[i] != 0)
            return 0;
    }
    return 1;
}

int multidim_iterator_is_at_end(const multidim_iterator_t *this)
{
    const size_t *const iter_offsets = multidim_iterator_offsets_const_ptr(this);
    const size_t *const iter_dims = multidim_iterator_dims_const_ptr(this);
    for (unsigned i = 0; i < this->ndims - 1; ++i)
    {
        if (iter_offsets[i] != iter_dims[i] - 1)
            return 0;
    }
    return iter_offsets[this->ndims - 1] == iter_dims[this->ndims - 1];
}

size_t multidim_iterator_get_flat_index(const multidim_iterator_t *this)
{
    size_t index = 0;
    const size_t *const iter_offsets = multidim_iterator_offsets_const_ptr(this);
    const size_t *const iter_dims = multidim_iterator_dims_const_ptr(this);
    for (unsigned i = 0; i < this->ndims; ++i)
    {
        index *= iter_dims[i];
        index += iter_offsets[i];
    }
    return index;
}

size_t multidim_iterator_get_ndims(const multidim_iterator_t *this)
{
    return this->ndims;
}

const size_t *multidim_iterator_dims(const multidim_iterator_t *this)
{
    return multidim_iterator_dims_const_ptr(this);
}

const size_t *multidim_iterator_offsets(const multidim_iterator_t *this)
{
    return multidim_iterator_offsets_const_ptr(this);
}

size_t multidim_iterator_get_dim(const multidim_iterator_t *this, const size_t dim)
{
    return multidim_iterator_dims_const_ptr(this)[dim];
}

size_t multidim_iterator_get_offset(const multidim_iterator_t *this, const size_t dim)
{
    return multidim_iterator_offsets_const_ptr(this)[dim];
}

size_t multidim_iterator_total_size(const multidim_iterator_t *this)
{
    size_t total_size = 1;
    const size_t *const dims = multidim_iterator_dims_const_ptr(this);
    for (size_t i = 0; i < this->ndims; ++i)
    {
        total_size *= dims[i];
    }
    return total_size;
}
