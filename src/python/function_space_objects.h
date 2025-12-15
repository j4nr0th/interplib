#ifndef INTERPLIB_FUNCTION_SPACE_OBJECTS_H
#define INTERPLIB_FUNCTION_SPACE_OBJECTS_H

#include "../operations/multidim_iteration.h"
#include "module.h"

typedef struct
{
    PyObject_VAR_HEAD;
    basis_spec_t specs[];
} function_space_object;

INTERPLIB_INTERNAL
extern PyType_Spec function_space_type_spec;

INTERPLIB_INTERNAL
function_space_object *function_space_object_create(PyTypeObject *type, unsigned n_basis,
                                                    const basis_spec_t INTERPLIB_ARRAY_ARG(specs, static n_basis));

INTERPLIB_INTERNAL
multidim_iterator_t *function_space_iterator(const function_space_object *space);

#endif // INTERPLIB_FUNCTION_SPACE_OBJECTS_H
