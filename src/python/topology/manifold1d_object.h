//
// Created by jan on 18.1.2025.
//

#ifndef MANIFOLD1D_H
#define MANIFOLD1D_H

#include "../../topology/topology.h"
#include "line_object.h"
#include "manifold_object.h"

typedef struct
{
    PyObject_HEAD;
    manifold1d_t manifold;
} manifold1d_object_t;

INTERPLIB_INTERNAL
extern PyType_Spec manifold1d_type_spec;

#endif // MANIFOLD1D_H
