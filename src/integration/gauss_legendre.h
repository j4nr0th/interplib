//
// Created by jan on 2025-09-07.
//

#ifndef INTERPLIB_GAUSS_LEGENDRE_H
#define INTERPLIB_GAUSS_LEGENDRE_H
#include "../common/allocator.h"

INTERPLIB_INTERNAL
int gauss_legendre_nodes_weights(unsigned n, double tol, unsigned max_iter, double INTERPLIB_ARRAY_ARG(x, restrict n),
                                 double INTERPLIB_ARRAY_ARG(w, restrict n));

INTERPLIB_INTERNAL
int gauss_legendre_nodes(unsigned n, double tol, unsigned max_iter, double INTERPLIB_ARRAY_ARG(x, restrict n));

#endif // INTERPLIB_GAUSS_LEGENDRE_H
