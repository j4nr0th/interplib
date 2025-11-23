#ifndef INTERPLIB_FUNCTION_SPACE_OBJECTS_H
#define INTERPLIB_FUNCTION_SPACE_OBJECTS_H

#include "module.h"

typedef struct
{
    PyObject_VAR_HEAD;
    basis_spec_t specs[];
} function_space_object;

INTERPLIB_INTERNAL
extern PyType_Spec function_space_type_spec;

#endif // INTERPLIB_FUNCTION_SPACE_OBJECTS_H
