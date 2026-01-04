//
// Created by jan on 2025-09-07.
//

#ifndef INTERPLIB_TOPOLOGY_H
#define INTERPLIB_TOPOLOGY_H

#include "../common/common_defines.h"
#include "../common/error.h"
#include <cutl/allocators.h>

typedef unsigned index_t;
_Static_assert(sizeof(index_t) == 4, "This must be for the geo_id_t to make sense");

typedef struct
{
    index_t index : 8 * sizeof(index_t) - 1;
    index_t reverse : 1;
} geo_id_t;

enum
{
    GEO_ID_INVALID = ~(1 << (8 * sizeof(index_t) - 1))
};

typedef union {
    struct
    {
        geo_id_t begin;
        geo_id_t end;
    };
    geo_id_t values[2];
} line_t;

typedef struct
{
    unsigned n_lines;
    geo_id_t *values;
} surface_t;

typedef enum
{
    MANIFOLD_DIMENSION_INVALID = 0,
    // These should match the dimension number
    MANIFOLD_1D = 1,
    MANIFOLD_2D = 2,
    // MANIFOLD_TYPE_3D = 3, // WIP
} manifold_dimension_t;

/*
 *
 *      1D Manifold
 *
 */
typedef struct
{
    manifold_dimension_t manifold_type;
    unsigned n_points;
    // geo_id *points; // no need for this, since we can just say points are given in order.
    unsigned n_lines;
    line_t *lines;
} manifold1d_t;

INTERPLIB_INTERNAL
interp_result_t manifold1d_new(manifold1d_t *manifold, unsigned n_points, unsigned n_lines,
                               const line_t INTERPLIB_ARRAY_ARG(lines, static n_lines),
                               const cutl_allocator_t *allocator);

INTERPLIB_INTERNAL
interp_result_t manifold1d_new_line(manifold1d_t *manifold, unsigned n_points, const cutl_allocator_t *allocator);

INTERPLIB_INTERNAL
void manifold1d_free(manifold1d_t *manifold, const cutl_allocator_t *allocator);

INTERPLIB_INTERNAL
interp_result_t manifold1d_dual(const manifold1d_t *manifold, manifold1d_t *dual, const cutl_allocator_t *allocator);

/*
 *
 *      2D Manifold
 *
 */

typedef struct
{
    manifold_dimension_t manifold_type;
    unsigned n_points;
    // geo_id *points; // no need for this, since we can just say points are given in order.
    unsigned n_lines;
    line_t *lines;
    size_t n_surfaces;
    size_t *surf_counts;  // number of lines per surface
    geo_id_t *surf_lines; // packed lines
} manifold2d_t;

// INTERPLIB_INTERNAL
// interp_result_t manifold2d_new_regular(manifold2d_t *manifold, unsigned n_points, unsigned n_lines,
//                                        const line_t INTERPLIB_ARRAY_ARG(lines, static n_lines), unsigned per_surface,
//                                        unsigned n_surfaces, unsigned points_per_surface,
//                                        const unsigned INTERPLIB_ARRAY_ARG(surface_points,
//                                                                           static n_surfaces *points_per_surface),
//                                        const cutl_allocator_t *allocator);

INTERPLIB_INTERNAL
interp_result_t manifold2d_new(manifold2d_t *manifold, unsigned n_points, unsigned n_lines,
                               const line_t INTERPLIB_ARRAY_ARG(lines, static n_lines), unsigned n_surfaces,
                               const surface_t INTERPLIB_ARRAY_ARG(surfaces, static n_surfaces),
                               const cutl_allocator_t *allocator);

INTERPLIB_INTERNAL
void manifold2d_free(manifold2d_t *manifold, const cutl_allocator_t *allocator);

INTERPLIB_INTERNAL
interp_result_t manifold2d_dual(const manifold2d_t *manifold, manifold2d_t *dual, const cutl_allocator_t *allocator);

typedef union {
    manifold_dimension_t manifold_type;
    manifold1d_t manifold1d;
    manifold2d_t manifold2d;
} manifold_t;

static inline unsigned manifold_dimension_count(const manifold_t *this)
{
    switch (this->manifold_type)
    {
    case MANIFOLD_1D:
        return 1;
    case MANIFOLD_2D:
        return 2;
    default:
        ASSERT(0, "Invalid manifold type.");
        return 0;
    }
}

#endif // INTERPLIB_TOPOLOGY_H
