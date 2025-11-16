//
// Created by jan on 2025-09-14.
//

#ifndef INTERPLIB_COLOR_H
#define INTERPLIB_COLOR_H

#include "common.h"

/** Type used to see what color to associate with values in the range of [0, 1]
 *
 */
typedef struct
{
    unsigned entries;
    vec4b_t colors[];
} colormap_t;

typedef enum
{
    VALUE_MAPPING_LINEAR,
    VALUE_MAPPING_LOGARITHMIC,
} value_mapping_type_t;

typedef struct
{
    value_mapping_type_t type;
    float min, max;
} value_mapping_t;

#endif // INTERPLIB_COLOR_H
