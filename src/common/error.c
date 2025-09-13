//
// Created by jan on 29.9.2024.
//

#include "error.h"

#define ERROR_ENUM_ENTRY(entry, msg) [(entry)] = {#entry, (msg)}

static const struct
{
    const char *str, *msg;
} error_messages[INTERP_ERROR_COUNT] = {
    ERROR_ENUM_ENTRY(INTERP_SUCCESS, "Success"),
    ERROR_ENUM_ENTRY(INTERP_ERROR_NOT_IN_DOMAIN, "Argument was not inside the domain."),
    ERROR_ENUM_ENTRY(INTERP_ERROR_NOT_INCREASING, "Input was not monotonically increasing."),
    ERROR_ENUM_ENTRY(INTERP_ERROR_FAILED_ALLOCATION, "Could not allocate desired amount of memory."),
    ERROR_ENUM_ENTRY(INTERP_ERROR_BAD_SYSTEM, "System of equations could not be solved."),
    ERROR_ENUM_ENTRY(INTERP_ERROR_INVALID_ENUM, "Enum had a value that was out of bounds."),
    ERROR_ENUM_ENTRY(INTERP_ERROR_NOT_IN_REGISTRY, "Object was not found in the registry."),
    ERROR_ENUM_ENTRY(INTERP_ERROR_GEOID_OUT_OF_RANGE, "GeoID was not within allowed range."),
    ERROR_ENUM_ENTRY(INTERP_ERROR_GEOID_NOT_VALID, "GeoID was invalid."),
    ERROR_ENUM_ENTRY(INTERP_ERROR_SURFACE_NOT_CLOSED, "Surface did not have a closed boundary."),
};

const char *interp_error_str(interp_result_t error)
{
    if (error < 0 || error >= INTERP_ERROR_COUNT)
        return "UNKNOWN";
    return error_messages[error].str;
}

const char *interp_error_msg(interp_result_t error)
{
    if (error < 0 || error >= INTERP_ERROR_COUNT)
        return "UNKNOWN";
    return error_messages[error].msg;
}
