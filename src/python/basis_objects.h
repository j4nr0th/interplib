#ifndef INTERPLIB_BASIS_SET_OBJECT_H
#define INTERPLIB_BASIS_SET_OBJECT_H

#include "module.h"

typedef struct
{
    PyObject_HEAD;
    basis_set_registry_t *registry;
} basis_registry_object;

INTERPLIB_INTERNAL
extern PyType_Spec basis_registry_type_specs;

basis_registry_object *basis_registry_object_create(PyTypeObject *type);

typedef struct
{
    PyObject_HEAD;
    basis_spec_t spec;
} basis_specs_object;

INTERPLIB_INTERNAL
extern PyType_Spec basis_specs_type_spec;

INTERPLIB_INTERNAL
basis_specs_object *basis_specs_object_create(PyTypeObject *type, basis_spec_t spec);

#endif // INTERPLIB_BASIS_SET_OBJECT_H
