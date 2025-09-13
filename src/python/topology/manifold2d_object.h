//
// Created by jan on 24.11.2024.
//

#ifndef MANIFOLD2D_H
#define MANIFOLD2D_H

#include "../../topology/topology.h"
#include "line_object.h"
#include "manifold_object.h"

typedef struct
{
    PyObject_HEAD;
    manifold2d_t manifold;
} manifold2d_object_t;

INTERPLIB_INTERNAL
extern PyType_Spec manifold2d_type_spec;

#endif // MANIFOLD2D_H
