//
// Created by jan on 2025-09-09.
//

#ifndef INTERPLIB_BASIS_LEGENDRE_H
#define INTERPLIB_BASIS_LEGENDRE_H
#include "basis_set.h"

INTERPLIB_INTERNAL
void legendre_basis_values(unsigned n_pts, const double INTERPLIB_ARRAY_ARG(nodes, n_pts), unsigned order,
                           double INTERPLIB_ARRAY_ARG(values, restrict(order + 1) * n_pts),
                           double INTERPLIB_ARRAY_ARG(derivatives, restrict(order + 1) * n_pts));

#endif // INTERPLIB_BASIS_LEGENDRE_H
