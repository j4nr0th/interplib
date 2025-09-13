//
// Created by jan on 29.9.2024.
//

#ifndef ERROR_H
#define ERROR_H
#include "common_defines.h"

typedef enum
{
    INTERP_SUCCESS = 0,
    INTERP_ERROR_NOT_IN_DOMAIN,
    INTERP_ERROR_NOT_INCREASING,
    INTERP_ERROR_FAILED_ALLOCATION,
    INTERP_ERROR_BAD_SYSTEM,
    INTERP_ERROR_INVALID_ENUM,
    INTERP_ERROR_NOT_IN_REGISTRY,
    INTERP_ERROR_GEOID_OUT_OF_RANGE,
    INTERP_ERROR_GEOID_NOT_VALID,
    INTERP_ERROR_SURFACE_NOT_CLOSED,
    INTERP_ERROR_OBJECT_CONNECTED_TWICE,

    INTERP_ERROR_COUNT,
} interp_result_t;

INTERPLIB_INTERNAL
const char *interp_error_str(interp_result_t error);

INTERPLIB_INTERNAL
const char *interp_error_msg(interp_result_t error);

#endif // ERROR_H
