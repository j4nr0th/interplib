//
// Created by jan on 2025-09-09.
//

#include "basis_lagrange.h"
#include "../integration/gauss_legendre.h"
#include "../integration/gauss_lobatto.h"
#include "../polynomials/lagrange.h"
#include <math.h>

INTERPLIB_INTERNAL
interp_result_t lagrange_basis_values(const unsigned n_pts, const double INTERPLIB_ARRAY_ARG(nodes, n_pts),
                                      const unsigned order,
                                      double INTERPLIB_ARRAY_ARG(values, restrict(order + 1) * n_pts),
                                      double INTERPLIB_ARRAY_ARG(derivatives, restrict(order + 1) * n_pts),
                                      double INTERPLIB_ARRAY_ARG(buffer, restrict 3 * (order + 1)),
                                      const basis_set_type_t type)
{
    // Find roots for Lagrange polynomials
    double *roots = buffer;
    switch (type)
    {
    case BASIS_LAGRANGE_UNIFORM:
        for (unsigned i = 0; i < order + 1; ++i)
        {
            roots[i] = (2.0 * i) / (double)(order)-1.0;
        }
        break;

    case BASIS_LAGRANGE_CHEBYSHEV_GAUSS:
        for (unsigned i = 0; i < order + 1; ++i)
        {
            roots[i] = cos(M_PI * (double)(2 * i + 1) / (double)(2 * (order + 1)));
        }
        break;

    case BASIS_LAGRANGE_GAUSS:
        gauss_legendre_nodes_weights(order + 1, 1e-12, 100, roots, buffer + order + 1);
        break;

    case BASIS_LAGRANGE_GAUSS_LOBATTO:
        gauss_lobatto_nodes_weights(order + 1, 1e-12, 100, roots, buffer + order + 1);
        break;

    default:
        return INTERP_ERROR_INVALID_ENUM;
    }

    double *const work_buffer_1 = buffer + 1 * (order + 1);
    double *const work_buffer_2 = buffer + 2 * (order + 1);
    lagrange_polynomial_values_transposed(n_pts, nodes, order + 1, roots, values, work_buffer_1);
    lagrange_polynomial_first_derivative_transposed(n_pts, nodes, order + 1, roots, derivatives, work_buffer_1,
                                                    work_buffer_2);
    return INTERP_SUCCESS;
}
