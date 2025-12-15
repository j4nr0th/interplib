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
    double *inverse_maps;
    coordinate_map_object *maps[];
} space_map_object;

INTERPLIB_INTERNAL
extern PyType_Spec space_map_type_spec;

/**
 * Retrieves the pointer to the start of the inverse mapping data at a specific
 * integration point within a space map object.
 *
 * Rows of inverse mapping correspond to the reference dimensions, while
 * the columns correspond to the physical dimensions.
 *
 * @param map Pointer to the space_map_object that contains the mapping.
 * @param flat_index The flat index of the integration point for which the
 *                   inverse mapping data is needed.
 *
 * @return Pointer to the starting element of the inverse mapping data
 *         corresponding to the specified integration point.
 */
INTERPLIB_INTERNAL
const double *space_map_inverse_at_integration_point(const space_map_object *map, size_t flat_index);

INTERPLIB_INTERNAL
size_t space_map_inverse_size_per_integration_point(const space_map_object *map);

#endif // INTERPLIB_MAPPINGS_H
