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
