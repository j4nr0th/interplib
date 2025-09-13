//
// Created by jan on 23.11.2024.
//

#ifndef LINEOBJECT_H
#define LINEOBJECT_H

#include "../../topology/topology.h"
#include "../module.h"

typedef struct
{
    PyObject_HEAD;
    line_t value;
} line_object_t;

INTERPLIB_INTERNAL
extern PyType_Spec line_type_spec;

INTERPLIB_INTERNAL
line_object_t *line_from_indices(PyTypeObject *line_type, geo_id_t begin, geo_id_t end);

#endif // LINEOBJECT_H
