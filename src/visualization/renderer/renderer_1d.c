//
// Created by jan on 2025-09-14.
//

#include <stddef.h>

#include "renderer_1d.h"

#include "shaders/1d_line_frag.h"
#include "shaders/1d_line_vert.h"

const uniform_spec_t UNIFORM_SPEC_1D[] = {
    {.name = "cmap_length", .type = GL_UNSIGNED_INT},
    {.name = "cmap_min", .type = GL_FLOAT},
    {.name = "cmap_max", .type = GL_FLOAT},
    {.name = "cmap_is_log", .type = GL_BOOL},
    {.name = "line_width", .type = GL_FLOAT},
    {.name = "center", .type = GL_FLOAT},
    {.name = "direction", .type = GL_FLOAT_VEC2},
    {.name = "viewport", .type = GL_FLOAT_VEC2},
    {},
};

viz_result_t viz_renderer_init(renderer_1d_t *this, const scene_configuration_t *configuration,
                               const allocator_callbacks *allocator)
{
    // Create shaders
    GLint status;

    const GLuint vtx_shader = glCreateShader(GL_VERTEX_SHADER);
    if (vtx_shader == 0)
    {
        return VIZ_FAILED_SHADER_CREATION;
    }
    glShaderSource(vtx_shader, 1, &line_1d_vert, NULL);
    glCompileShader(vtx_shader);
    glGetShaderiv(vtx_shader, GL_COMPILE_STATUS, &status);
    if (status != GL_TRUE)
    {
        glDeleteShader(vtx_shader);
        return VIZ_FAILED_SHADER_CREATION;
    }

    const GLuint frg_shader = glCreateShader(GL_FRAGMENT_SHADER);
    if (frg_shader == 0)
    {
        glDeleteShader(vtx_shader);
        return VIZ_FAILED_SHADER_CREATION;
    }
    glShaderSource(frg_shader, 1, &line_1d_frag, NULL);
    glCompileShader(frg_shader);
    glGetShaderiv(frg_shader, GL_COMPILE_STATUS, &status);
    if (status != GL_TRUE)
    {
        glDeleteShader(vtx_shader);
        glDeleteShader(frg_shader);
        return VIZ_FAILED_SHADER_CREATION;
    }

    const GLuint program = glCreateProgram();

    glAttachShader(program, vtx_shader);
    glAttachShader(program, frg_shader);
    glLinkProgram(program);
    glGetProgramiv(program, GL_LINK_STATUS, &status);
    glDeleteShader(vtx_shader);
    glDeleteShader(frg_shader);

    if (status != GL_TRUE)
    {
        glDeleteProgram(program);
        return VIZ_FAILED_SHADER_CREATION;
    }

    this->element_buffer_bind_point = glGetProgramResourceIndex(program, GL_SHADER_STORAGE_BLOCK, "element_buffer");
    this->coefficient_buffer_bind_point =
        glGetProgramResourceIndex(program, GL_SHADER_STORAGE_BLOCK, "coefficient_buffer");
    this->colormap_buffer_bind_point = glGetProgramResourceIndex(program, GL_SHADER_STORAGE_BLOCK, "colormap_buffer");

    if (this->element_buffer_bind_point == GL_INVALID_INDEX ||
        this->coefficient_buffer_bind_point == GL_INVALID_INDEX || this->colormap_buffer_bind_point == GL_INVALID_INDEX

    )
    {
        glDeleteProgram(program);
        return VIZ_FAILED_SHADER_CREATION;
    }

    unsigned uniform_count;
    for (uniform_count = 0; UNIFORM_SPEC_1D[uniform_count].name != NULL; ++uniform_count)
    {
    }

    uniform_t *const uniforms = allocate(allocator, uniform_count * sizeof *uniforms);
    if (!uniforms)
    {
        glDeleteProgram(program);
        return VIZ_FAILED_ALLOCATION;
    }
    for (unsigned i = 0; i < uniform_count; ++i)
    {
        uniforms[i] = (uniform_t){.spec = UNIFORM_SPEC_1D + i,
                                  .location = glGetUniformLocation(program, UNIFORM_SPEC_1D[i].name)};
    }
    this->update_uniforms = 1;
    this->uniforms = uniforms;
    this->uniform_count = uniform_count;
    this->program = program;

    enum
    {
        VBO_BUFFER = 0,
        ELEMENT_BUFFER = 1,
        COEFF_BUFFER = 2,
        CMAP_BUFFER = 3,
        BUFFER_COUNT
    };
    GLuint buffers[BUFFER_COUNT];
    glGenBuffers(BUFFER_COUNT, buffers);
    this->vertex_buffer_id = buffers[VBO_BUFFER];
    this->element_buffer_id = buffers[ELEMENT_BUFFER];
    this->coefficient_buffer_id = buffers[COEFF_BUFFER];
    this->colormap_buffer_id = buffers[CMAP_BUFFER];

    glBindBuffer(GL_ARRAY_BUFFER, this->vertex_buffer_id);
    glBufferData(GL_ARRAY_BUFFER, sizeof(float) * 2, NULL, GL_DYNAMIC_DRAW);

    const GLint pos_attrib = glGetAttribLocation(program, "pos");
    glEnableVertexAttribArray(pos_attrib);
    glVertexAttribPointer(pos_attrib, 1, GL_FLOAT, GL_FALSE, sizeof(line_vertex_1d_data_t),
                          (void *)offsetof(line_vertex_1d_data_t, pos));

    const GLint val_attrib = glGetAttribLocation(program, "t_in");
    glEnableVertexAttribArray(val_attrib);
    glVertexAttribPointer(val_attrib, 1, GL_FLOAT, GL_FALSE, sizeof(line_vertex_1d_data_t),
                          (void *)offsetof(line_vertex_1d_data_t, val));

    const GLint ei_attrib = glGetAttribLocation(program, "ei_in");
    glEnableVertexAttribArray(ei_attrib);
    glVertexAttribPointer(ei_attrib, 1, GL_UNSIGNED_INT, GL_FALSE, sizeof(line_vertex_1d_data_t),
                          (void *)offsetof(line_vertex_1d_data_t, ei));

    glBindBuffer(GL_ARRAY_BUFFER, 0);

    return VIZ_SUCCESS;
}
void viz_renderer_destroy(renderer_1d_t *this, const allocator_callbacks *allocator)
{
    glDeleteProgram(this->program);
    deallocate(allocator, this->uniforms);
    glDeleteBuffers(4, this->buffers);
    *this = (renderer_1d_t){};
}

viz_result_t viz_renderer_draw(renderer_1d_t *this, const scene_configuration_t *scene,
                               const window_configuration_t *window)
{
    glBindBuffer(GL_ARRAY_BUFFER, this->vertex_buffer_id);
    glUseProgram(this->program);
    if (this->update_uniforms)
    {
        for (unsigned i = 0; i < this->uniform_count; ++i)
        {
            const uniform_t *const uniform = this->uniforms + i;
            switch (uniform->spec->type)
            {
            case GL_UNSIGNED_INT:
            case GL_BOOL:
                glUniform1ui(uniform->location, uniform->value.v_uint);
                break;
            case GL_FLOAT:
                glUniform1f(uniform->location, uniform->value.v_float);
                break;
            case GL_FLOAT_VEC2:
                glUniform2fv(uniform->location, 1, uniform->value.v_vec2);
                break;
            default:
                return VIZ_ERROR_INVALID_ENUM;
            }
        }
        this->update_uniforms = 0;
    }

    return VIZ_SUCCESS;
}
