#ifndef INTERPLIB_MAPPINGS_H
#define INTERPLIB_MAPPINGS_H

#include "module.h"

typedef struct
{
    PyObject_VAR_HEAD;
    unsigned ndim;
    integration_spec_t *int_specs;
    double values[];
} coordinate_map_object;

INTERPLIB_INTERNAL
extern PyType_Spec coordinate_map_type_spec;

INTERPLIB_INTERNAL
const double *coordinate_map_values(const coordinate_map_object *map);

INTERPLIB_INTERNAL
const double *coordinate_map_gradient(const coordinate_map_object *map, unsigned dim);

typedef struct
{
    PyObject_VAR_HEAD;
    unsigned ndim;
    integration_spec_t *int_specs;
    double *determinant;
    coordinate_map_object *maps[];
} space_map_object;

INTERPLIB_INTERNAL
extern PyType_Spec space_map_type_spec;

#endif // INTERPLIB_MAPPINGS_H
