//
// Created by jan on 2025-09-09.
//

#ifndef INTERPLIB_BASIS_H
#define INTERPLIB_BASIS_H
#include "../common/error.h"
#include "../integration/integration_rules.h"

typedef enum
{
    BASIS_INVALID = 0,
    BASIS_LEGENDRE,
    BASIS_LAGRANGE_GAUSS_LOBATTO,
    BASIS_LAGRANGE_GAUSS,
    BASIS_LAGRANGE_UNIFORM,
    BASIS_LAGRANGE_CHEBYSHEV_GAUSS,
    BASIS_BERNSTEIN,
} basis_set_type_t;

typedef struct
{
    basis_set_type_t type;
    unsigned order;
} basis_spec_t;

typedef struct
{
    basis_spec_t spec;                        // Specifications for the Basis
    integration_rule_spec_t integration_spec; // Specifications for the Integration rule
    double _data[];                           // Values of the basis_sets and their derivatives at integration nodes
} basis_set_t;

static inline const double *basis_set_values_all(const basis_set_t *this)
{
    return this->_data;
}

static inline const double *basis_set_basis_values(const basis_set_t *this, const unsigned index)
{
    ASSERT(index <= this->spec.order, "Index was out of bounds.");
    return this->_data + index * (this->integration_spec.order + 1);
}

static inline const double *basis_set_derivatives_all(const basis_set_t *this)
{
    return this->_data + (this->spec.order + 1) * (this->integration_spec.order + 1);
}

static inline const double *basis_set_basis_derivatives(const basis_set_t *this, const unsigned index)
{
    ASSERT(index <= this->spec.order, "Index was out of bounds.");
    return this->_data + index * (this->spec.order + 1 + index) * (this->integration_spec.order + 1);
}

typedef struct basis_set_registry_t basis_set_registry_t;

INTERPLIB_INTERNAL
interp_result_t basis_set_registry_create(basis_set_registry_t **out, int should_cache,
                                          const allocator_callbacks *allocator);

INTERPLIB_INTERNAL
interp_result_t basis_set_registry_get_basis_set(basis_set_registry_t *this, const basis_set_t **p_basis,
                                                 const integration_rule_t *integration_rule, basis_spec_t spec);

INTERPLIB_INTERNAL
interp_result_t basis_set_registry_release_basis_set(basis_set_registry_t *this, const basis_set_t *basis);

INTERPLIB_INTERNAL
void basis_set_registry_destroy(basis_set_registry_t *this);

INTERPLIB_INTERNAL
void basis_set_registry_release_unused_basis_sets(basis_set_registry_t *this);

INTERPLIB_INTERNAL
void basis_set_registry_release_all_basis_sets(const basis_set_registry_t *this);

INTERPLIB_INTERNAL
unsigned basis_set_registry_get_sets(basis_set_registry_t *this, unsigned max_count,
                                     basis_spec_t INTERPLIB_ARRAY_ARG(basis_spec, max_count),
                                     integration_rule_spec_t INTERPLIB_ARRAY_ARG(integration_spec, max_count));

#endif // INTERPLIB_BASIS_H
