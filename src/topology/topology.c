//
// Created by jan on 2025-09-08.
//
#include "topology.h"

#include <string.h>

INTERPLIB_INTERNAL
interp_result_t manifold1d_new(manifold1d_t *manifold, const unsigned n_points, const unsigned n_lines,
                               const line_t INTERPLIB_ARRAY_ARG(lines, static n_lines),
                               const cutl_allocator_t *allocator)
{
    manifold->manifold_type = MANIFOLD_1D;
    manifold->n_points = n_points;
    manifold->n_lines = n_lines;
    manifold->lines = cutl_alloc(allocator, n_lines * sizeof *manifold->lines);
    if (!manifold->lines)
        return INTERP_ERROR_FAILED_ALLOCATION;

    for (unsigned i_line = 0; i_line < n_lines; ++i_line)
    {
        const line_t *line = lines + i_line;
        if (line->begin.index >= n_points || line->end.index >= n_points)
        {
            return INTERP_ERROR_GEOID_OUT_OF_RANGE;
        }
        manifold->lines[i_line] = *line;
    }

    return INTERP_SUCCESS;
}

INTERPLIB_INTERNAL
interp_result_t manifold1d_new_line(manifold1d_t *manifold, unsigned n_points, const cutl_allocator_t *allocator)
{
    manifold->manifold_type = MANIFOLD_1D;
    manifold->n_points = n_points;
    manifold->n_lines = n_points - 1;
    manifold->lines = cutl_alloc(allocator, manifold->n_lines * sizeof *manifold->lines);
    if (!manifold->lines)
        return INTERP_ERROR_FAILED_ALLOCATION;
    for (unsigned i = 0; i < manifold->n_lines; ++i)
        manifold->lines[i] = (line_t){
            (geo_id_t){.index = i, .reverse = 0},
            (geo_id_t){.index = i + 1, .reverse = 0},
        };
    return INTERP_SUCCESS;
}

INTERPLIB_INTERNAL
void manifold1d_free(manifold1d_t *manifold, const cutl_allocator_t *allocator)
{
    cutl_dealloc(allocator, manifold->lines);
    *manifold = (manifold1d_t){};
}

INTERPLIB_INTERNAL
interp_result_t manifold1d_dual(const manifold1d_t *manifold, manifold1d_t *dual, const cutl_allocator_t *allocator)
{
    dual->manifold_type = MANIFOLD_1D;
    dual->n_points = manifold->n_lines;
    dual->n_lines = manifold->n_points;
    dual->lines = cutl_alloc(allocator, dual->n_lines * sizeof *dual->lines);
    if (!dual->lines)
        return INTERP_ERROR_FAILED_ALLOCATION;
    for (unsigned i_point = 0; i_point < manifold->n_points; ++i_point)
    {
        line_t dual_line = {.begin = {.index = GEO_ID_INVALID, .reverse = 0},
                            .end = {.index = GEO_ID_INVALID, .reverse = 0}};
        unsigned found = 0;
        for (unsigned i_line = 0; i_line < manifold->n_lines && found < 2; ++i_line)
        {
            if (manifold->lines[i_line].begin.index == i_point)
            {
                if (dual_line.begin.index != GEO_ID_INVALID)
                {
                    cutl_dealloc(allocator, dual->lines);
                    return INTERP_ERROR_OBJECT_CONNECTED_TWICE;
                }
                dual_line.begin.index = manifold->lines[i_line].begin.index;
                found += 1;
            }
            if (manifold->lines[i_line].end.index == i_point)
            {
                if (dual_line.end.index != GEO_ID_INVALID)
                {
                    cutl_dealloc(allocator, dual->lines);
                    return INTERP_ERROR_OBJECT_CONNECTED_TWICE;
                }
                dual_line.end.index = manifold->lines[i_line].end.index;
                found += 1;
            }
        }
        dual->lines[i_point] = dual_line;
    }

    return INTERP_SUCCESS;
}

INTERPLIB_INTERNAL
interp_result_t manifold2d_new(manifold2d_t *manifold, unsigned n_points, unsigned n_lines,
                               const line_t INTERPLIB_ARRAY_ARG(lines, static n_lines), unsigned n_surfaces,
                               const surface_t INTERPLIB_ARRAY_ARG(surfaces, static n_surfaces),
                               const cutl_allocator_t *allocator)
{
    manifold->n_points = n_points;
    manifold->n_lines = n_lines;
    manifold->n_surfaces = 0;

    manifold->lines = NULL;
    manifold->surf_lines = NULL;
    manifold->surf_counts = NULL;

    manifold->lines = cutl_alloc(allocator, sizeof *manifold->lines * n_lines);
    if (!manifold->lines)
    {
        return INTERP_ERROR_FAILED_ALLOCATION;
    }

    for (unsigned i_ln = 0; i_ln < n_lines; ++i_ln)
    {
        const geo_id_t begin = lines->begin;
        const geo_id_t end = lines->end;

        if (begin.index >= n_points || end.index >= n_points)
        {
            cutl_dealloc(allocator, manifold->lines);
            return INTERP_ERROR_GEOID_OUT_OF_RANGE;
        }
        manifold->lines[i_ln] = (line_t){
            .begin = begin,
            .end = end,
        };
    }

    manifold->n_surfaces = n_surfaces;

    manifold->surf_counts = cutl_alloc(allocator, sizeof *manifold->surf_counts * (manifold->n_surfaces + 1));
    if (!manifold->surf_counts)
    {
        cutl_dealloc(allocator, manifold->lines);
        return INTERP_ERROR_FAILED_ALLOCATION;
    }

    // Count up the offsets
    manifold->surf_counts[0] = 0;
    size_t n_surf_lines = 0;
    // int same_size = 1;
    // const size_t first_size = manifold->n_surfaces ? surfaces[0].n_lines : 0;

    for (unsigned i = 0; i < manifold->n_surfaces; ++i)
    {
        const surface_t *const surface = surfaces + i;
        n_surf_lines += surface->n_lines;
        manifold->surf_counts[i + 1] = n_surf_lines;
        // same_size = same_size && (surface->n_lines == first_size);
    }

    manifold->surf_lines = cutl_alloc(allocator, sizeof(*manifold->surf_lines) * n_surf_lines);
    if (!manifold->surf_lines)
    {
        cutl_dealloc(allocator, manifold->surf_counts);
        cutl_dealloc(allocator, manifold->lines);
        return INTERP_ERROR_FAILED_ALLOCATION;
    }

    for (unsigned i = 0; i < manifold->n_surfaces; ++i)
    {
        const size_t offset = manifold->surf_counts[i];
        const size_t len = manifold->surf_counts[i + 1] - offset;
        geo_id_t *const surf_lines = manifold->surf_lines + offset;
        const surface_t *const surface = surfaces + i;
        for (unsigned j = 0; j < len; ++j)
        {
            const geo_id_t id = surface->values[j];
            if (id.index != GEO_ID_INVALID)
            {
                cutl_dealloc(allocator, surf_lines);
                cutl_dealloc(allocator, manifold->surf_counts);
                cutl_dealloc(allocator, manifold->lines);
                return INTERP_ERROR_GEOID_NOT_VALID;
            }
            if (id.index >= manifold->n_lines)
            {
                cutl_dealloc(allocator, surf_lines);
                cutl_dealloc(allocator, manifold->surf_counts);
                cutl_dealloc(allocator, manifold->lines);
                return INTERP_ERROR_GEOID_OUT_OF_RANGE;
            }
            surf_lines[j] = id;
        }

        geo_id_t end;
        {
            const geo_id_t id1 = surf_lines[len - 1];
            if (id1.reverse)
            {
                end = manifold->lines[id1.index].begin;
            }
            else
            {
                end = manifold->lines[id1.index].end;
            }
        }
        for (unsigned j = 0; j < len; ++j)
        {

            geo_id_t begin, new_end;
            const geo_id_t id2 = surf_lines[j];
            if (id2.reverse)
            {
                begin = manifold->lines[id2.index].end;
                new_end = manifold->lines[id2.index].begin;
            }
            else
            {
                begin = manifold->lines[id2.index].begin;
                new_end = manifold->lines[id2.index].end;
            }

            if (begin.index != end.index)
            {
                cutl_dealloc(allocator, surf_lines);
                cutl_dealloc(allocator, manifold->surf_counts);
                cutl_dealloc(allocator, manifold->lines);
                return INTERP_ERROR_SURFACE_NOT_CLOSED;
            }
            end = new_end;
        }
    }
    // if (same_size)
    // {
    //     PyErr_WarnFormat(PyExc_UserWarning, 0,
    //                      "Consider calling the Manifold2D.from_regular, since all surfaces have the same length of"
    //                      "%u.", (unsigned)first_size);
    // }

    return INTERP_SUCCESS;
}

INTERPLIB_INTERNAL
void manifold2d_free(manifold2d_t *manifold, const cutl_allocator_t *allocator)
{
    cutl_dealloc(allocator, manifold->lines);
    cutl_dealloc(allocator, manifold->surf_counts);
    cutl_dealloc(allocator, manifold->surf_lines);
    *manifold = (manifold2d_t){};
}

INTERPLIB_INTERNAL
interp_result_t manifold2d_dual(const manifold2d_t *manifold, manifold2d_t *dual, const cutl_allocator_t *allocator)
{
    const unsigned n_lines = manifold->n_lines;
    line_t *const dual_lines = cutl_alloc(allocator, sizeof *dual_lines * n_lines);
    if (!dual_lines)
        return INTERP_ERROR_FAILED_ALLOCATION;

    dual->n_lines = n_lines;
    dual->lines = dual_lines;

    for (unsigned i_ln = 0; i_ln < n_lines; ++i_ln)
    {
        line_t line = {.begin = {.reverse = 0, .index = GEO_ID_INVALID},
                       .end = {.reverse = 0, .index = GEO_ID_INVALID}};
        size_t cnt_before = 0;
        // NOTE: this loop could be broken as soon as beginning and end are found, assuming the manifold is
        // not invalid. For now the function does not assume this, since this is likely not a huge performance
        // hit.
        for (unsigned i_surf = 0; i_surf < manifold->n_surfaces; ++i_surf)
        {
            const size_t cnt_after = manifold->surf_counts[i_surf + 1];
            for (size_t i = cnt_before; i < cnt_after; ++i)
            {
                const geo_id_t id = manifold->surf_lines[i];
                if (id.index != i_ln)
                {
                    continue;
                }
                if (id.reverse)
                {
                    if (line.begin.index != GEO_ID_INVALID)
                    {
                        cutl_dealloc(allocator, dual_lines);
                        return INTERP_ERROR_OBJECT_CONNECTED_TWICE;
                    }
                    line.begin.index = i_surf;
                }
                else
                {
                    if (line.end.index != GEO_ID_INVALID)
                    {
                        cutl_dealloc(allocator, dual_lines);
                        return INTERP_ERROR_OBJECT_CONNECTED_TWICE;
                    }
                    line.end.index = i_surf;
                }
            }
            cnt_before = cnt_after;
        }
        dual_lines[i_ln] = line;
    }

    const unsigned n_surf = manifold->n_points;
    dual->n_surfaces = n_surf;
    size_t *const surf_counts = cutl_alloc(allocator, sizeof *surf_counts * (n_surf + 1));
    if (!surf_counts)
    {
        cutl_dealloc(allocator, dual_lines);
        return INTERP_ERROR_FAILED_ALLOCATION;
    }
    dual->surf_counts = surf_counts;

    surf_counts[0] = 0;
    size_t acc_cnt = 0;
    for (unsigned pt_idx = 0; pt_idx < manifold->n_points; ++pt_idx)
    {
        for (unsigned i_ln = 0; i_ln < manifold->n_lines; ++i_ln)
        {
            const line_t *ln = manifold->lines + i_ln;
            acc_cnt += (ln->begin.index == pt_idx);
            acc_cnt += (ln->end.index == pt_idx);
        }
        surf_counts[pt_idx + 1] = acc_cnt;
    }

    geo_id_t *dual_surf = cutl_alloc(allocator, sizeof *dual_surf * surf_counts[n_surf]);
    if (!dual_surf)
    {
        cutl_dealloc(allocator, surf_counts);
        cutl_dealloc(allocator, dual_lines);
        return INTERP_ERROR_FAILED_ALLOCATION;
    }
    dual->surf_lines = dual_surf;

    size_t offset = 0;
    for (unsigned pt_idx = 0; pt_idx < manifold->n_points; ++pt_idx)
    {
        const size_t offset_end = surf_counts[pt_idx + 1];
        for (unsigned i_ln = 0; offset < offset_end; ++i_ln)
        {
            const line_t *ln = manifold->lines + i_ln;
            if (ln->begin.index == pt_idx)
            {
                dual_surf[offset] = (geo_id_t){.index = i_ln, .reverse = 0};
                offset += 1;
            }
            if (ln->end.index == pt_idx)
            {
                dual_surf[offset] = (geo_id_t){.index = i_ln, .reverse = 1};
                offset += 1;
            }
        }
    }
    dual->n_points = manifold->n_surfaces;
    return INTERP_SUCCESS;
}
