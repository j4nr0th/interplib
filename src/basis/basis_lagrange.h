//
// Created by jan on 2025-09-09.
//

#ifndef INTERPLIB_BASIS_LAGRANGE_H
#define INTERPLIB_BASIS_LAGRANGE_H
#include "basis_set.h"

INTERPLIB_INTERNAL
interp_result_t lagrange_basis_values(unsigned n_pts, const double INTERPLIB_ARRAY_ARG(nodes, n_pts), unsigned order,
                                      double INTERPLIB_ARRAY_ARG(values, restrict(order + 1) * n_pts),
                                      double INTERPLIB_ARRAY_ARG(derivatives, restrict(order + 1) * n_pts),
                                      double INTERPLIB_ARRAY_ARG(buffer, restrict 3 * (order + 1)),
                                      basis_set_type_t type);

#endif // INTERPLIB_BASIS_LAGRANGE_H
