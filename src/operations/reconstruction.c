#include "reconstruction.h"

struct reconstruction_state_t
{
    const allocator_callbacks *allocator;
    unsigned ndims;
    multidim_iterator_t *iter_int;
    multidim_iterator_t *iter_basis;
    const basis_set_t *basis_sets[];
};

interp_result_t reconstruction_state_create(const unsigned ndims, const basis_set_t *basis_sets[const static ndims],
                                            const allocator_callbacks *allocator, reconstruction_state_t **out)
{
    reconstruction_state_t *const this = allocate(allocator, sizeof *this + ndims * sizeof(*this->basis_sets));
    if (!this)
        return INTERP_ERROR_FAILED_ALLOCATION;
    *this = (reconstruction_state_t){.allocator = allocator, .ndims = ndims};

    this->iter_int = allocate(allocator, multidim_iterator_needed_memory(ndims));
    if (!this->iter_int)
    {
        deallocate(allocator, this);
        return INTERP_ERROR_FAILED_ALLOCATION;
    }

    this->iter_basis = allocate(allocator, multidim_iterator_needed_memory(ndims));
    if (!this->iter_basis)
    {
        deallocate(allocator, this->iter_int);
        deallocate(allocator, this);
        return INTERP_ERROR_FAILED_ALLOCATION;
    }

    for (unsigned idim = 0; idim < ndims; ++idim)
    {
        multidim_iterator_init_dim(this->iter_basis, idim, basis_sets[idim]->spec.order + 1);
        multidim_iterator_init_dim(this->iter_int, idim, basis_sets[idim]->integration_spec.order + 1);
    }
    for (unsigned idim = 0; idim < ndims; ++idim)
    {
        this->basis_sets[idim] = basis_sets[idim];
    }

    *out = this;
    return INTERP_SUCCESS;
}

void reconstruction_state_release(reconstruction_state_t *recon_state)
{
    const allocator_callbacks *allocator = recon_state->allocator;
    deallocate(allocator, recon_state->iter_int);
    deallocate(allocator, recon_state->iter_basis);
    // Clear the contents before the free.
    *recon_state = (reconstruction_state_t){};
    deallocate(allocator, recon_state);
}

void reconstruction_state_reset(reconstruction_state_t *recon_state)
{
    multidim_iterator_set_to_start(recon_state->iter_int);
    multidim_iterator_set_to_start(recon_state->iter_basis);
}

size_t reconstruction_state_integration_points_count(const reconstruction_state_t *state)
{
    return multidim_iterator_total_size(state->iter_int);
}

size_t reconstruction_state_basis_count(const reconstruction_state_t *state)
{
    return multidim_iterator_total_size(state->iter_basis);
}

double reconstruction_state_current_basis_value(const reconstruction_state_t *state, const size_t integration_index)
{
    double basis_value = 1;
    for (unsigned idim = 0; idim < state->ndims; ++idim)
    {
        basis_value *= basis_set_basis_values(state->basis_sets[idim],
                                              multidim_iterator_get_offset(state->iter_basis, idim))[integration_index];
    }
    return basis_value;
}

void reconstruction_state_basis_values_current(const reconstruction_state_t *state, double *values)
{
    const size_t integration_index = multidim_iterator_get_flat_index(state->iter_int);
    multidim_iterator_set_to_start(state->iter_basis);
    while (!multidim_iterator_is_at_end(state->iter_basis))
    {
        values[multidim_iterator_get_flat_index(state->iter_basis)] =
            reconstruction_state_current_basis_value(state, integration_index);
    }
}
