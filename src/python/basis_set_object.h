#ifndef INTERPLIB_BASIS_SET_OBJECT_H
#define INTERPLIB_BASIS_SET_OBJECT_H

#include "module.h"

typedef struct
{
    PyObject_HEAD;
    basis_spec_t spec;
} basis_specs_object;

INTERPLIB_INTERNAL
extern PyType_Spec basis_specs_type_spec;

#endif // INTERPLIB_BASIS_SET_OBJECT_H
