//
// Created by jan on 2025-09-07.
//

#ifndef INTERPLIB_LEGENDRE_H
#define INTERPLIB_LEGENDRE_H

#include "../common/allocator.h"

INTERPLIB_INTERNAL
void legendre_eval_bonnet_two(unsigned n, double x, double INTERPLIB_ARRAY_ARG(out, 2));

INTERPLIB_INTERNAL
void legendre_eval_bonnet(unsigned n, double x, unsigned m, double INTERPLIB_ARRAY_ARG(out, m));

INTERPLIB_INTERNAL
void legendre_eval_bonnet_all(unsigned n, double x, double INTERPLIB_ARRAY_ARG(out, n + 1));

INTERPLIB_INTERNAL
void legendre_eval_bonnet_all_stride(unsigned n, double x, unsigned stride, unsigned offset,
                                     double INTERPLIB_ARRAY_ARG(out, (n + 1) * stride));

#endif // INTERPLIB_LEGENDRE_H
