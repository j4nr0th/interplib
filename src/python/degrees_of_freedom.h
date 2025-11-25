#ifndef INTERPLIB_DEGREES_OF_FREEDOM_H
#define INTERPLIB_DEGREES_OF_FREEDOM_H

#include "module.h"

typedef struct
{
    PyObject_VAR_HEAD;
    unsigned n_dims;
    basis_spec_t *basis_specs;
    double values[];
} dof_object;

INTERPLIB_INTERNAL
extern PyType_Spec degrees_of_freedom_type_spec;

#endif // INTERPLIB_DEGREES_OF_FREEDOM_H
