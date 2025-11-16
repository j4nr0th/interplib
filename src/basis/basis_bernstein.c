//
// Created by jan on 2025-09-09.
//

#include "basis_bernstein.h"
#include "../polynomials/bernstein.h"

INTERPLIB_INTERNAL
interp_result_t bernstein_basis_create(basis_set_t **out, const basis_spec_t spec, const integration_rule_t *rule,
                                       const allocator_callbacks *allocator)
{
    basis_set_t *const this =
        allocate(allocator, sizeof *this + 2 * sizeof(*this->_data) * (spec.order + 1) * rule->n_nodes);
    if (!this)
        return INTERP_ERROR_FAILED_ALLOCATION;

    bernstein_interpolation_value_derivative_matrix(rule->n_nodes, integration_rule_nodes_const(rule), spec.order,
                                                    this->_data, this->_data + (spec.order + 1) * rule->n_nodes);
    this->integration_spec = rule->spec;
    this->spec = spec;
    *out = this;
    return INTERP_SUCCESS;
}
