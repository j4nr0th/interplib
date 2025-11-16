//
// Created by jan on 2025-09-14.
//

#ifndef INTERPLIB_RENDERER_H
#define INTERPLIB_RENDERER_H

#include "../color.h"
#include "../configuration.h"
#include "glad/glad.h"
#include <GL/gl.h>

typedef struct
{
    const char *name;
    GLenum type;
} uniform_spec_t;

typedef struct
{
    const uniform_spec_t *spec;
    GLint location;
    union {
        GLint v_int;
        GLfloat v_float;
        GLfloat v_vec2[2];
        GLuint v_uint;
    } value;
} uniform_t;

typedef struct
{
    GLuint program;

    union {
        struct
        {
            GLuint vertex_buffer_id;
            GLuint element_buffer_id;
            GLuint coefficient_buffer_id;
            GLuint colormap_buffer_id;
        };
        GLuint buffers[4];
    };

    GLint element_buffer_bind_point;
    GLint coefficient_buffer_bind_point;
    GLint colormap_buffer_bind_point;

    unsigned update_uniforms; // when non-zero, uniforms should be updated before the render
    unsigned uniform_count;
    uniform_t *uniforms;
} renderer_1d_t;

typedef struct
{
    GLfloat pos, val;
    GLuint ei;
} line_vertex_1d_data_t;

viz_result_t viz_renderer_init(renderer_1d_t *this, const scene_configuration_t *configuration,
                               const allocator_callbacks *allocator);

void viz_renderer_destroy(renderer_1d_t *this, const allocator_callbacks *allocator);

viz_result_t viz_renderer_draw(renderer_1d_t *this, const scene_configuration_t *scene,
                               const window_configuration_t *window);

#endif // INTERPLIB_RENDERER_H
