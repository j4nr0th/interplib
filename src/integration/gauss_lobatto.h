//
// Created by jan on 27.1.2025.
//

#ifndef GAUSSLOBATTO_H
#define GAUSSLOBATTO_H
#include "../common/common_defines.h"

INTERPLIB_INTERNAL
int gauss_lobatto_nodes_weights(unsigned n, double tol, unsigned max_iter, double INTERPLIB_ARRAY_ARG(x, restrict n),
                                double INTERPLIB_ARRAY_ARG(w, restrict n));

INTERPLIB_INTERNAL
int gauss_lobatto_nodes(unsigned n, double tol, unsigned max_iter, double INTERPLIB_ARRAY_ARG(x, restrict n));

#endif // GAUSSLOBATTO_H
