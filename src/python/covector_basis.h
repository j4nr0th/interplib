#ifndef INTERPLIB_PYTHON_COVECTOR_BASIS_H
#define INTERPLIB_PYTHON_COVECTOR_BASIS_H

#include "../../src/basis/covector_basis.h"
#include "module.h"

typedef struct
{
    PyObject_HEAD;
    const covector_basis_t basis;
} covector_basis_object;

INTERPLIB_INTERNAL
extern PyType_Spec covector_basis_type_spec;

INTERPLIB_INTERNAL
covector_basis_object *covector_basis_object_create(PyTypeObject *type, covector_basis_t basis);

#endif // INTERPLIB_PYTHON_COVECTOR_BASIS_H
