//
// Created by jan on 2025-09-09.
//

#include "basis_legendre.h"
#include "../polynomials/legendre.h"

INTERPLIB_INTERNAL
void legendre_basis_values(const unsigned n_pts, const double INTERPLIB_ARRAY_ARG(nodes, n_pts), const unsigned order,
                           double INTERPLIB_ARRAY_ARG(values, restrict(order + 1) * n_pts),
                           double INTERPLIB_ARRAY_ARG(derivatives, restrict(order + 1) * n_pts))
{
    for (unsigned i_pt = 0; i_pt < n_pts; ++i_pt)
    {
        const double node = nodes[i_pt];
        // Compute the different basis values
        legendre_eval_bonnet_all_stride(order, node, n_pts, i_pt, values);
        // Compute the different basis derivatives
        derivatives[i_pt] = 0; // The first basis has no derivative
        double deriv = 0;
        for (unsigned i_deriv = 1; i_deriv <= order; ++i_deriv)
        {
            // Use recurrence formula for subsequent derivatives
            deriv = i_deriv * derivatives[i_pt + (i_deriv - 1) * n_pts] + node * deriv;
            derivatives[i_pt + i_deriv * n_pts] = deriv;
        }
    }
}
