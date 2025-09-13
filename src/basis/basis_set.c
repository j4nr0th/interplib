//
// Created by jan on 2025-09-09.
//

#include "basis_set.h"
#include "../common/rw_lock.h"

#include "basis_bernstein.h"
#include "basis_lagrange.h"
#include "basis_legendre.h"
#include <string.h>

// Bucket containing all basis sets of the same integration type
typedef struct
{
    integration_rule_type_t type;
    unsigned count;
    unsigned capacity;
    _Atomic unsigned *ref_counts;
    basis_set_t **basis_sets;
} basis_bucket_itype_t;

// Bucket containing all basis sets of the same basis_sets type
typedef struct
{
    basis_set_type_t type;
    unsigned count;
    unsigned capacity;
    basis_bucket_itype_t *buckets;
} basis_bucket_btype_t;

struct basis_set_registry_t
{
    rw_lock_t lock;
    allocator_callbacks allocator;
    int should_cache;
    unsigned n_buckets;
    basis_bucket_btype_t *buckets;
};

interp_result_t basis_set_registry_create(basis_set_registry_t **out, int should_cache,
                                          const allocator_callbacks *allocator)
{
    basis_set_registry_t *const this = allocate(allocator, sizeof *this);
    if (!this)
        return INTERP_ERROR_FAILED_ALLOCATION;
    *this = (basis_set_registry_t){
        .allocator = *allocator, .should_cache = should_cache != 0, .n_buckets = 0, .buckets = NULL};
    const interp_result_t res = rw_lock_init(&this->lock);
    if (res != INTERP_SUCCESS)
    {
        deallocate(allocator, this);
        return res;
    }

    *out = this;
    return INTERP_SUCCESS;
}

static inline interp_result_t basis_set_bucket_btype_init(basis_bucket_btype_t *this, basis_set_type_t type,
                                                          unsigned starting_size, const allocator_callbacks *allocator)
{
    this->type = type;
    this->count = 0;
    this->capacity = starting_size;
    this->buckets = allocate(allocator, starting_size * sizeof *this->buckets);
    if (!this->buckets)
    {
        this->capacity = 0;
        return INTERP_ERROR_FAILED_ALLOCATION;
    }
    return INTERP_SUCCESS;
}

static inline interp_result_t basis_set_bucket_itype_init(basis_bucket_itype_t *this, integration_rule_type_t type,
                                                          unsigned starting_size, const allocator_callbacks *allocator)
{
    this->type = type;
    this->count = 0;
    this->capacity = starting_size;
    this->basis_sets = allocate(allocator, starting_size * sizeof *this->basis_sets);
    if (!this->basis_sets)
    {
        this->capacity = 0;
        return INTERP_ERROR_FAILED_ALLOCATION;
    }
    this->ref_counts = allocate(allocator, starting_size * sizeof *this->ref_counts);
    if (!this->ref_counts)
    {
        deallocate(allocator, this->basis_sets);
        this->capacity = 0;
        return INTERP_ERROR_FAILED_ALLOCATION;
    }
    return INTERP_SUCCESS;
}

static inline interp_result_t basis_set_create(basis_set_t **out, const integration_rule_t *integration_rule,
                                               const basis_spec_t spec, const allocator_callbacks *allocator)
{
    basis_set_t *const this = allocate(allocator, sizeof *this + 2 * sizeof(*this->_data) * (spec.order + 1) *
                                                                     (integration_rule->spec.order + 1));
    if (!this)
        return INTERP_ERROR_FAILED_ALLOCATION;
    this->integration_spec = integration_rule->spec;
    this->spec = spec;

    double *buffer;
    switch (spec.type)
    {
    case BASIS_LEGENDRE:
        legendre_basis_values(integration_rule->spec.order + 1, integration_rule_nodes_const(integration_rule),
                              spec.order, (double *)basis_set_values_all(this),
                              (double *)basis_set_derivatives_all(this));
        break;

    case BASIS_LAGRANGE_GAUSS:
    case BASIS_LAGRANGE_UNIFORM:
    case BASIS_LAGRANGE_GAUSS_LOBATTO:
    case BASIS_LAGRANGE_CHEBYSHEV_GAUSS:
        buffer = allocate(allocator, 3 * (spec.order + 1) * sizeof(*buffer));
        if (!buffer)
        {
            deallocate(allocator, this);
            return INTERP_ERROR_FAILED_ALLOCATION;
        }
        const interp_result_t res = lagrange_basis_values(
            integration_rule->spec.order + 1, integration_rule_nodes_const(integration_rule), spec.order,
            (double *)basis_set_values_all(this), (double *)basis_set_derivatives_all(this), buffer, spec.type);
        deallocate(allocator, buffer);
        if (res != INTERP_SUCCESS)
        {
            deallocate(allocator, this);
            return res;
        }
        break;

    case BASIS_BERNSTEIN:
        bernstein_basis_values(integration_rule->spec.order + 1, integration_rule_nodes_const(integration_rule),
                               spec.order, (double *)basis_set_values_all(this),
                               (double *)basis_set_derivatives_all(this));
        break;

    default:
        return INTERP_ERROR_INVALID_ENUM;
    }
    *out = this;

    return INTERP_SUCCESS;
}

static inline interp_result_t basis_set_registry_add_btype_bucket(basis_set_registry_t *this,
                                                                  const basis_set_type_t type,
                                                                  const unsigned starting_size)
{
    basis_bucket_btype_t *new_buckets =
        reallocate(&this->allocator, this->buckets, (this->n_buckets + 1) * sizeof *new_buckets);
    if (!new_buckets)
    {
        return INTERP_ERROR_FAILED_ALLOCATION;
    }
    this->buckets = new_buckets;
    basis_bucket_btype_t *const first_bucket = this->buckets + this->n_buckets;
    this->n_buckets += 1;
    const interp_result_t res = basis_set_bucket_btype_init(first_bucket, type, starting_size, &this->allocator);
    if (res != INTERP_SUCCESS)
    {
        return res;
    }
    return INTERP_SUCCESS;
}

static inline interp_result_t basis_set_registry_add_itype_bucket(const basis_set_registry_t *this,
                                                                  basis_bucket_btype_t *first_bucket,
                                                                  const integration_rule_type_t type,
                                                                  const unsigned starting_size)
{
    if (first_bucket->count == first_bucket->capacity)
    {
        const unsigned new_capacity = first_bucket->capacity * 2;
        basis_bucket_itype_t *const new_buckets =
            reallocate(&this->allocator, first_bucket->buckets, new_capacity * sizeof *new_buckets);
        if (!new_buckets)
            return INTERP_ERROR_FAILED_ALLOCATION;
        first_bucket->buckets = new_buckets;
        first_bucket->capacity = new_capacity;
    }
    basis_bucket_itype_t *const second_bucket = first_bucket->buckets + first_bucket->count;
    first_bucket->count += 1;
    const interp_result_t res = basis_set_bucket_itype_init(second_bucket, type, starting_size, &this->allocator);
    if (res != INTERP_SUCCESS)
        return res;
    return INTERP_SUCCESS;
}

static inline interp_result_t basis_set_registry_add_basis_set(const basis_set_registry_t *this,
                                                               basis_bucket_itype_t *second_bucket,
                                                               const basis_set_t **p_basis,
                                                               const integration_rule_t *integration_rule,
                                                               const basis_spec_t spec)
{
    if (second_bucket->count == second_bucket->capacity)
    {
        const unsigned new_capacity = second_bucket->capacity * 2;
        basis_set_t **const new_basis =
            reallocate(&this->allocator, second_bucket->basis_sets, new_capacity * sizeof *new_basis);
        if (!new_basis)
            return INTERP_ERROR_FAILED_ALLOCATION;
        second_bucket->basis_sets = new_basis;
        _Atomic unsigned *const new_ref_counts =
            reallocate(&this->allocator, second_bucket->ref_counts, new_capacity * sizeof *new_ref_counts);
        if (!new_ref_counts)
            return INTERP_ERROR_FAILED_ALLOCATION;
        second_bucket->ref_counts = new_ref_counts;
        second_bucket->capacity = new_capacity;
    }

    basis_set_t *basis;
    const interp_result_t res = basis_set_create(&basis, integration_rule, spec, &this->allocator);
    if (res != INTERP_SUCCESS)
        return res;

    second_bucket->basis_sets[second_bucket->count] = basis;
    second_bucket->ref_counts[second_bucket->count] = 1;
    second_bucket->count += 1;

    *p_basis = basis;
    return INTERP_SUCCESS;
}

interp_result_t basis_set_registry_get_basis_set(basis_set_registry_t *this, const basis_set_t **p_basis,
                                                 const integration_rule_t *integration_rule, basis_spec_t spec)
{
    rw_lock_acquire_read(&this->lock);
    enum
    {
        BUCKET_STARTING_SIZE = 8
    };
    basis_bucket_btype_t *first_bucket = NULL;
    basis_bucket_itype_t *second_bucket = NULL;

    for (unsigned i = 0; i < this->n_buckets; ++i)
    {
        basis_bucket_btype_t *const b_bucket = this->buckets + i;
        if (b_bucket->type == spec.type)
        {
            first_bucket = b_bucket;
            for (unsigned j = 0; j < first_bucket->count; ++j)
            {
                basis_bucket_itype_t *const i_bucket = first_bucket->buckets + j;
                if (i_bucket->type == integration_rule->spec.type)
                {
                    second_bucket = i_bucket;
                    for (unsigned k = 0; k < second_bucket->count; ++k)
                    {
                        const basis_set_t *const basis = second_bucket->basis_sets[k];
                        if (basis->spec.order == spec.order &&
                            basis->integration_spec.order == integration_rule->spec.order)
                        {
                            second_bucket->ref_counts[k] += 1;
                            *p_basis = basis;
                            rw_lock_release_read(&this->lock);
                            return INTERP_SUCCESS;
                        }
                    }
                    break;
                }
            }
            break;
        }
    }

    rw_lock_release_read(&this->lock);
    rw_lock_acquire_write(&this->lock);
    if (!first_bucket)
    {
        const interp_result_t res = basis_set_registry_add_btype_bucket(this, spec.type, BUCKET_STARTING_SIZE);
        if (res != INTERP_SUCCESS)
        {
            rw_lock_release_write(&this->lock);
            return res;
        }
        first_bucket = this->buckets + this->n_buckets - 1;
    }
    if (!second_bucket)
    {
        const interp_result_t res =
            basis_set_registry_add_itype_bucket(this, first_bucket, integration_rule->spec.type, BUCKET_STARTING_SIZE);
        if (res != INTERP_SUCCESS)
        {
            rw_lock_release_write(&this->lock);
            return res;
        }
        second_bucket = first_bucket->buckets + first_bucket->count - 1;
    }

    const basis_set_t *basis;
    const interp_result_t res = basis_set_registry_add_basis_set(this, second_bucket, &basis, integration_rule, spec);
    rw_lock_release_write(&this->lock);
    if (res != INTERP_SUCCESS)
    {
        return res;
    }
    *p_basis = basis;
    return INTERP_SUCCESS;
}

interp_result_t basis_set_registry_release_basis_set(basis_set_registry_t *this, const basis_set_t *basis)
{
    rw_lock_acquire_read(&this->lock);
    basis_bucket_btype_t *first_bucket = NULL;
    for (unsigned i = 0; i < this->n_buckets; ++i)
    {
        if (this->buckets[i].type == basis->spec.type)
        {
            first_bucket = this->buckets + i;
            break;
        }
    }
    if (!first_bucket)
    {
        rw_lock_release_read(&this->lock);
        return INTERP_ERROR_NOT_IN_REGISTRY;
    }

    basis_bucket_itype_t *second_bucket = NULL;
    for (unsigned i = 0; i < first_bucket->count; ++i)
    {
        if (first_bucket->buckets[i].type == basis->integration_spec.type)
        {
            second_bucket = first_bucket->buckets + i;
            break;
        }
    }
    if (!second_bucket)
    {
        rw_lock_release_read(&this->lock);
        return INTERP_ERROR_NOT_IN_REGISTRY;
    }

    for (unsigned position = 0; position < second_bucket->count; ++position)
    {
        if (second_bucket->basis_sets[position] == basis)
        {
            rw_lock_release_read(&this->lock);
            rw_lock_acquire_write(&this->lock);
            second_bucket->ref_counts[position] -= 1;
            if (second_bucket->ref_counts[position] == 0 && this->should_cache == 0)
            {
                deallocate(&this->allocator, second_bucket->basis_sets[position]);
                memmove(second_bucket->basis_sets + position, second_bucket->basis_sets + position + 1,
                        sizeof(*second_bucket->basis_sets) * (second_bucket->count - position - 1));
                second_bucket->count -= 1;
            }
            rw_lock_release_write(&this->lock);
            return INTERP_SUCCESS;
        }
    }

    rw_lock_release_read(&this->lock);
    return INTERP_ERROR_NOT_IN_REGISTRY;
}

void basis_set_registry_destroy(basis_set_registry_t *this)
{
    for (unsigned i = 0; i < this->n_buckets; ++i)
    {
        basis_bucket_btype_t *first_bucket = this->buckets + i;
        for (unsigned j = 0; j < first_bucket->count; ++j)
        {
            basis_bucket_itype_t *second_bucket = first_bucket->buckets + j;
            for (unsigned k = 0; k < second_bucket->count; ++k)
            {
                deallocate(&this->allocator, second_bucket->basis_sets[k]);
            }
            deallocate(&this->allocator, second_bucket->basis_sets);
            deallocate(&this->allocator, second_bucket->ref_counts);
        }
        deallocate(&this->allocator, first_bucket->buckets);
    }
    deallocate(&this->allocator, this->buckets);
    deallocate(&this->allocator, this);
}

void basis_set_registry_release_unused_basis_sets(basis_set_registry_t *const this)
{
    rw_lock_acquire_write(&this->lock);
    for (unsigned i = 0; i < this->n_buckets; ++i)
    {
        const basis_bucket_btype_t *const first_bucket = this->buckets + i;
        for (unsigned j = 0; j < first_bucket->count; ++j)
        {
            basis_bucket_itype_t *const second_bucket = first_bucket->buckets + j;
            for (unsigned k = 0; k < second_bucket->count; ++k)
            {
                if (second_bucket->ref_counts[k] == 0 && this->should_cache == 0)
                {
                    deallocate(&this->allocator, second_bucket->basis_sets[k]);
                    second_bucket->basis_sets[k] = NULL;
                }
            }
            unsigned pos, valid;
            for (pos = 0, valid = 0; pos < second_bucket->count; ++pos)
            {
                if (second_bucket->basis_sets[pos] != NULL)
                {
                    second_bucket->basis_sets[valid] = second_bucket->basis_sets[pos];
                    second_bucket->ref_counts[valid] = second_bucket->ref_counts[pos];
                    valid += 1;
                }
            }
            second_bucket->count = valid;
        }
    }
    rw_lock_release_write(&this->lock);
}
