//
// Created by jan on 24.11.2024.
//

#ifndef SURFACEOBJECT_H
#define SURFACEOBJECT_H

#include "../../topology/topology.h"
#include "../module.h"

typedef struct
{
    PyObject_HEAD size_t n_lines;
    geo_id_t lines[];
} surface_object_t;

INTERPLIB_INTERNAL
extern PyType_Spec surface_type_spec;

INTERPLIB_INTERNAL
surface_object_t *surface_object_empty(PyTypeObject *surf_type, size_t count);

INTERPLIB_INTERNAL
surface_object_t *surface_object_from_value(PyTypeObject *surf_type, size_t count, geo_id_t ids[static count]);

#endif // SURFACEOBJECT_H
