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

INTERPLIB_INTERNAL
const basis_set_t **python_basis_sets_get(unsigned n_basis, const basis_spec_t specs[const static n_basis],
                                          const integration_rule_t *rules[const static n_basis],
                                          basis_set_registry_t *registry);

INTERPLIB_INTERNAL
void python_basis_sets_release(unsigned n_basis, const basis_set_t *sets[static n_basis],
                               basis_set_registry_t *registry);

INTERPLIB_INTERNAL
multidim_iterator_t *python_basis_iterator(unsigned n_basis, const basis_spec_t specs[const static n_basis]);

#endif // INTERPLIB_BASIS_SET_OBJECT_H
