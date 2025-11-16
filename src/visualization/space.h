#ifndef INTERPLIB_SPACE_H
#define INTERPLIB_SPACE_H
#include "common.h"

// Enum for specifying different types which must all map the space data is in to flat 2D space for rendering.
typedef enum
{
    SPACE_TYPE_INVALID,
    SPACE_TYPE_1D,
    SPACE_TYPE_2D,
    SPACE_TYPE_3D,
} space_type_t;

typedef struct
{
    space_type_t type; // Type
    float origin;      // What point is at the origin of the window
    float angle;       // How much to rotate the view
} space_1d_t;

typedef struct
{
    space_type_t type; // Type
    vec2_t origin;     // Point at the middle of the screen
    vec2_t right;      // Direction of the vector that will point right
    vec2_t up;         // Direction of the vector that will point up
} space_2d_t;

typedef struct
{
    space_type_t type; // Type
    vec3_t origin;     // Point at the middle of the screen
    vec3_t right;      // Direction of the vector that will point right
    vec3_t up;         // Direction of the vector that will point up
    vec3_t forward;    // Direction of the vector that will point forward
} space_3d_t;

typedef union {
    space_1d_t space_1d;
    space_2d_t space_2d;
    space_3d_t space_3d;
} space_t;

mat4x4_t space_view_mat4x4(const space_t *space);

#endif // INTERPLIB_SPACE_H
