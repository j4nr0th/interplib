//
// Created by jan on 5.11.2024.
//

#ifndef BERNSTEIN_H
#define BERNSTEIN_H

#include "../common_defines.h"

INTERPLIB_INTERNAL
void bernstein_from_power_series(unsigned n, double INTERPLIB_ARRAY_ARG(coeffs, static n));

INTERPLIB_INTERNAL
void bernstein_interpolation_vector(double t, unsigned n, double INTERPLIB_ARRAY_ARG(out, restrict n));

INTERPLIB_INTERNAL
PyObject *bernstein_interpolation_matrix(PyObject *self, PyObject *const *args, Py_ssize_t nargs);

INTERPLIB_INTERNAL
extern const char bernstein_interpolation_matrix_doc[];

INTERPLIB_INTERNAL
PyObject *bernstein_coefficients(PyObject *Py_UNUSED(self), PyObject *arg);

INTERPLIB_INTERNAL
extern const char bernstein_coefficients_doc[];

#endif // BERNSTEIN_H
