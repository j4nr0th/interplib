//
// Created by jan on 2025-09-09.
//

#include "basis_lagrange.h"
#include "../integration/gauss_legendre.h"
#include "../integration/gauss_lobatto.h"
#include "../polynomials/lagrange.h"
#include <math.h>

INTERPLIB_INTERNAL
interp_result_t lagrange_basis_create(basis_set_t **out, const basis_spec_t spec, const integration_rule_t *rule,
                                      const allocator_callbacks *allocator)
{
    basis_set_t *const this =
        allocate(allocator, sizeof *this + sizeof(*this->_data) * (spec.order + 1) * (2 * (rule->spec.order + 1) + 1));
    if (!this)
        return INTERP_ERROR_FAILED_ALLOCATION;

    double *const roots = this->_data + 2 * (spec.order + 1) * (rule->spec.order + 1);

    this->integration_spec = rule->spec;
    this->spec = spec;
    // Find roots for Lagrange polynomials
    switch (spec.type)
    {
    case BASIS_LAGRANGE_UNIFORM:
        for (unsigned i = 0; i < spec.order + 1; ++i)
        {
            roots[i] = (2.0 * i) / (double)(spec.order) - 1.0;
        }
        break;

    case BASIS_LAGRANGE_CHEBYSHEV_GAUSS:
        for (unsigned i = 0; i < spec.order + 1; ++i)
        {
            roots[i] = -cos(M_PI * (double)(2 * i + 1) / (double)(2 * (spec.order + 1)));
        }
        break;

    case BASIS_LAGRANGE_GAUSS:
        gauss_legendre_nodes_weights(spec.order + 1, 1e-12, 100, roots, this->_data);
        break;

    case BASIS_LAGRANGE_GAUSS_LOBATTO:
        gauss_lobatto_nodes_weights(spec.order + 1, 1e-12, 100, roots, this->_data);
        break;

    default:
        return INTERP_ERROR_INVALID_ENUM;
    }

    lagrange_polynomial_values_transposed_2(rule->n_nodes, integration_rule_nodes_const(rule), spec.order + 1, roots,
                                            (double *)basis_set_values_all(this));
    lagrange_polynomial_first_derivative_transposed_2(rule->n_nodes, integration_rule_nodes_const(rule), spec.order + 1,
                                                      roots, (double *)basis_set_derivatives_all(this));
    this->spec = spec;
    this->integration_spec = rule->spec;
    *out = this;
    return INTERP_SUCCESS;
}
