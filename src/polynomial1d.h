//
// Created by jan on 21.10.2024.
//

#ifndef POLYNOMIAL1D_H
#define POLYNOMIAL1D_H
#include <Python.h>

#include "common_defines.h"

typedef struct
{
    PyObject_HEAD unsigned n;
    vectorcallfunc call_poly;
    double k[];
} polynomial_basis_t;

INTERPLIB_INTERNAL
extern PyTypeObject polynomial1d_type_object;

#endif // POLYNOMIAL1D_H
