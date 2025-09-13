//
// Created by jan on 2025-09-09.
//

#include "basis_bernstein.h"
#include "../polynomials/bernstein.h"

INTERPLIB_INTERNAL
void bernstein_basis_values(const unsigned n_pts, const double INTERPLIB_ARRAY_ARG(nodes, n_pts), const unsigned order,
                            double INTERPLIB_ARRAY_ARG(values, restrict(order + 1) * n_pts),
                            double INTERPLIB_ARRAY_ARG(derivatives, restrict(order + 1) * n_pts))
{
    bernstein_interpolation_value_derivative_matrix(n_pts, nodes, order, values, derivatives);
}
