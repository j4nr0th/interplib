//
// Created by jan on 2025-09-11.
//

#ifndef INTERPLIB_BASIS_SET_OBJECT_H
#define INTERPLIB_BASIS_SET_OBJECT_H

#include "module.h"

typedef struct
{
    PyObject_HEAD;
    const basis_set_t *basis_set;
} basis_set_object;

INTERPLIB_INTERNAL
extern PyType_Spec basis_set_type_spec;

#endif // INTERPLIB_BASIS_SET_OBJECT_H
