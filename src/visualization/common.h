//
// Created by jan on 2025-09-14.
//

#ifndef INTERPLIB_COMMON_H
#define INTERPLIB_COMMON_H
#include <stdint.h>

typedef union {
    float data[2];
    struct
    {
        float x, y;
    };
    struct
    {
        float v0, v1;
    };
} vec2_t;

typedef union {
    float data[3];
    struct
    {
        float x, y, z;
    };
    struct
    {
        float v0, v1, v2;
    };
} vec3_t;

typedef union {
    float data[4];
    struct
    {
        float x, y, z, w;
    };
    struct
    {
        float v0, v1, v2, v3;
    };
} vec4_t;

typedef union {
    float data[16];
    vec4_t columns[4];
    struct
    {
        float m00, m10, m20, m30;
        float m01, m11, m21, m31;
        float m02, m12, m22, m32;
        float m03, m13, m23, m33;
    };
} mat4x4_t;

typedef union {
    uint8_t data[4];
    struct
    {
        uint8_t r, g, b, a;
    };
} vec4b_t;

static inline vec4b_t vec4b_from_vec4(const vec4_t v)
{
    return (vec4b_t){.r = (uint8_t)(v.x * 255.0f),
                     .g = (uint8_t)(v.y * 255.0f),
                     .b = (uint8_t)(v.z * 255.0f),
                     .a = (uint8_t)(v.w * 255.0f)};
}

static inline vec4_t vec4_from_vec4b(const vec4b_t v)
{
    return (vec4_t){
        .x = (float)v.r / 255.0f, .y = (float)v.g / 255.0f, .z = (float)v.b / 255.0f, .w = (float)v.a / 255.0f};
}

typedef enum
{
    VIZ_SUCCESS = 0,
    VIZ_ERROR = 1,
    VIZ_INVALID_ARGUMENT = 2,
    VIZ_FAILED_ALLOCATION = 3,
    VIZ_INVALID_STATE = 4,
    VIZ_NOT_IMPLEMENTED = 5,
    VIZ_NO_CONTEXT = 6,
    VIZ_FAILED_SHADER_CREATION = 7,
    VIZ_ERROR_INVALID_ENUM = 8,
} viz_result_t;

enum
{
    // Around a million of floats seems reasonable, right?
    VIZ_LARGEST_GL_BUFFER = sizeof(float) * (1 << 20),
};

#endif // INTERPLIB_COMMON_H
