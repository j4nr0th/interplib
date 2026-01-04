//
// Created by jan on 24.11.2024.
//

#include "manifold2d_object.h"

#include <numpy/arrayobject.h>

#include "geoid_object.h"
#include "line_object.h"
#include "surface_object.h"

static PyObject *manifold2d_str(PyObject *self)
{
    const manifold2d_object_t *this = (manifold2d_object_t *)self;
    return PyUnicode_FromFormat("Manifold2D(%u points, %u lines, %u surfaces)", this->manifold.n_points,
                                this->manifold.n_lines, this->manifold.n_surfaces);
}

PyDoc_STRVAR(manifold2d_type_docstring, "Manifold2D(...)\n"
                                        "Two dimensional manifold consisting of surfaces made of lines.\n");

static void manifold2d_dealloc(PyObject *self)
{
    manifold2d_object_t *this = (manifold2d_object_t *)self;
    manifold2d_free(&this->manifold, &SYSTEM_ALLOCATOR);
    Py_TYPE(this)->tp_free(this);
}

static PyObject *manifold2d_get_dimension(PyObject *Py_UNUSED(self), void *Py_UNUSED(closure))
{
    return PyLong_FromLong(2);
}

static PyObject *manifold2d_get_n_points(PyObject *self, void *Py_UNUSED(closure))
{
    const manifold2d_object_t *this = (manifold2d_object_t *)self;
    return PyLong_FromUnsignedLong(this->manifold.n_points);
}

static PyObject *manifold2d_get_n_lines(PyObject *self, void *Py_UNUSED(closure))
{
    const manifold2d_object_t *this = (manifold2d_object_t *)self;
    return PyLong_FromUnsignedLong(this->manifold.n_lines);
}

static PyObject *manifold2d_get_n_surfaces(PyObject *self, void *Py_UNUSED(closure))
{
    const manifold2d_object_t *this = (manifold2d_object_t *)self;
    return PyLong_FromUnsignedLong(this->manifold.n_surfaces);
}

static PyGetSetDef manifold2d_getset[] = {
    {
        .name = "dimension",
        .get = manifold2d_get_dimension,
        .set = NULL,
        .doc = "int : Dimension of the manifold.",
        .closure = NULL,
    },
    {
        .name = "n_points",
        .get = manifold2d_get_n_points,
        .set = NULL,
        .doc = "Number of points in the mesh.",
        .closure = NULL,
    },
    {
        .name = "n_lines",
        .get = manifold2d_get_n_lines,
        .set = NULL,
        .doc = "Number of lines in the mesh.",
        .closure = NULL,
    },
    {
        .name = "n_surfaces",
        .get = manifold2d_get_n_surfaces,
        .set = NULL,
        .doc = "Number of surfaces in the mesh.",
        .closure = NULL,
    },
    {0},
};

static PyObject *manifold2d_get_line(PyObject *self, PyObject *arg)
{
    const manifold2d_object_t *this = (manifold2d_object_t *)self;
    const interplib_module_state_t *const state = PyType_GetModuleState(Py_TYPE(self));
    geo_id_t id;
    if (geo_id_from_object(state->geoid_type, arg, &id) < 0)
        return NULL;
    if (id.index == GEO_ID_INVALID)
    {
        PyErr_SetString(PyExc_ValueError, "Invalid GeoID was given.");
        return NULL;
    }
    if (id.index >= this->manifold.n_lines)
    {
        PyErr_Format(PyExc_IndexError, "Manifold has only %u lines, but line with index %u was requested.",
                     (unsigned)this->manifold.n_lines, (unsigned)id.index);
        return NULL;
    }

    line_object_t *const line =
        line_from_indices(state->line_type, this->manifold.lines[id.index].begin, this->manifold.lines[id.index].end);
    if (!line)
        return NULL;

    if (id.reverse)
    {
        const geo_id_t tmp = line->value.begin;
        line->value.begin = line->value.end;
        line->value.end = tmp;
    }

    return (PyObject *)line;
}

static PyObject *manifold2d_get_surface(PyObject *self, PyObject *arg)
{
    const manifold2d_object_t *this = (manifold2d_object_t *)self;
    const interplib_module_state_t *const state = PyType_GetModuleState(Py_TYPE(self));
    geo_id_t id;

    if (geo_id_from_object(state->geoid_type, arg, &id) < 0)
    {
        return NULL;
    }
    if (id.index == GEO_ID_INVALID)
    {
        PyErr_SetString(PyExc_ValueError, "Invalid GeoID was given.");
        return NULL;
    }

    if (id.index >= this->manifold.n_surfaces)
    {
        PyErr_Format(PyExc_IndexError, "Index %u is our of bounds for a mesh with %u surfaces.", (unsigned)id.index,
                     (unsigned)this->manifold.n_surfaces);
        return NULL;
    }

    const size_t offset_0 = this->manifold.surf_counts[id.index];
    const size_t offset_1 = this->manifold.surf_counts[id.index + 1];
    surface_object_t *surf =
        surface_object_from_value(state->surf_type, offset_1 - offset_0, this->manifold.surf_lines + offset_0);
    if (!surf)
    {
        return NULL;
    }

    if (id.reverse)
    {
        for (unsigned i = 0; i < surf->n_lines; ++i)
        {
            surf->lines[i].reverse = !surf->lines[i].reverse;
        }
    }
    return (PyObject *)surf;
}

// static int mesh_dual_from_primal(const manifold2d_object_t *const primal, manifold2d_object_t *const dual)
// {
//     const unsigned n_lines = primal->manifold.n_lines;
//     line_t *const dual_lines = PyObject_Malloc(sizeof *dual_lines * n_lines);
//     if (!dual_lines)
//         return -1;
//
//     dual->manifold.n_lines = n_lines;
//     dual->manifold.lines = dual_lines;
//
//     for (unsigned i_ln = 0; i_ln < n_lines; ++i_ln)
//     {
//         line_t line = {.begin = {.reverse = 0, .index = GEO_ID_INVALID},
//                        .end = {.reverse = 0, .index = GEO_ID_INVALID}};
//         size_t cnt_before = 0;
//         // NOTE: this loop could be broken as soon as beginning and end are found, assuming the manifold is
//         // not invalid. For now the function does not assume this, since this is likely not a huge performance
//         // hit.
//         for (unsigned i_surf = 0; i_surf < primal->manifold.n_surfaces; ++i_surf)
//         {
//             const size_t cnt_after = primal->manifold.surf_counts[i_surf + 1];
//             for (size_t i = cnt_before; i < cnt_after; ++i)
//             {
//                 const geo_id_t id = primal->manifold.surf_lines[i];
//                 if (id.index != i_ln)
//                 {
//                     continue;
//                 }
//                 if (id.reverse)
//                 {
//                     if (line.begin.index != GEO_ID_INVALID)
//                     {
//                         PyErr_Format(PyExc_ValueError,
//                                      "Line %u appears in at least two surfaces (%u and %u)"
//                                      " with negative orientation. Such manifold is invalid.",
//                                      i_ln, line.begin.index, i_surf);
//                         // PyObject_Free(dual_lines);
//                         return -1;
//                     }
//                     line.begin.index = i_surf;
//                 }
//                 else
//                 {
//                     if (line.end.index != GEO_ID_INVALID)
//                     {
//                         PyErr_Format(PyExc_ValueError,
//                                      "Line %u appears in at least two surfaces (%u and %u)"
//                                      " with positive orientation. Such manifold is invalid.",
//                                      i_ln, line.end.index, i_surf);
//                         // PyObject_Free(dual_lines);
//                         return -1;
//                     }
//                     line.end.index = i_surf;
//                 }
//             }
//             cnt_before = cnt_after;
//         }
//         dual_lines[i_ln] = line;
//     }
//
//     const unsigned n_surf = primal->manifold.n_points;
//     dual->manifold.n_surfaces = n_surf;
//     size_t *const surf_counts = PyObject_Malloc(sizeof *surf_counts * (n_surf + 1));
//     if (!surf_counts)
//     {
//         return -1;
//     }
//     dual->manifold.surf_counts = surf_counts;
//
//     surf_counts[0] = 0;
//     size_t acc_cnt = 0;
//     for (unsigned pt_idx = 0; pt_idx < primal->manifold.n_points; ++pt_idx)
//     {
//         for (unsigned i_ln = 0; i_ln < primal->manifold.n_lines; ++i_ln)
//         {
//             const line_t *ln = primal->manifold.lines + i_ln;
//             acc_cnt += (ln->begin.index == pt_idx);
//             acc_cnt += (ln->end.index == pt_idx);
//         }
//         surf_counts[pt_idx + 1] = acc_cnt;
//     }
//
//     geo_id_t *dual_surf = PyObject_Malloc(sizeof *dual_surf * surf_counts[n_surf]);
//     if (!dual_surf)
//     {
//         return -1;
//     }
//     dual->manifold.surf_lines = dual_surf;
//
//     size_t offset = 0;
//     for (unsigned pt_idx = 0; pt_idx < primal->manifold.n_points; ++pt_idx)
//     {
//         const size_t offset_end = surf_counts[pt_idx + 1];
//         for (unsigned i_ln = 0; offset < offset_end; ++i_ln)
//         {
//             const line_t *ln = primal->manifold.lines + i_ln;
//             if (ln->begin.index == pt_idx)
//             {
//                 dual_surf[offset] = (geo_id_t){.index = i_ln, .reverse = 0};
//                 offset += 1;
//             }
//             if (ln->end.index == pt_idx)
//             {
//                 dual_surf[offset] = (geo_id_t){.index = i_ln, .reverse = 1};
//                 offset += 1;
//             }
//         }
//     }
//     dual->manifold.n_points = primal->manifold.n_surfaces;
//     return 0;
// }

static PyObject *manifold2d_compute_dual(PyObject *self, PyObject *Py_UNUSED(arg))
{
    manifold2d_object_t *that = (manifold2d_object_t *)Py_TYPE(self)->tp_alloc(Py_TYPE(self), 0);
    if (!that)
    {
        return NULL;
    }
    const manifold2d_object_t *this = (manifold2d_object_t *)self;
    const interplib_module_state_t *const state = PyType_GetModuleState(Py_TYPE(self));
    if (!state)
    {
        Py_DECREF(that);
        return NULL;
    }
    const interp_result_t res = manifold2d_dual(&this->manifold, &that->manifold, &SYSTEM_ALLOCATOR);
    if (res != INTERP_SUCCESS)
    {
        PyErr_Format(PyExc_RuntimeError, "Failed to compute dual of manifold: %s (%s).", interp_error_msg(res),
                     interp_error_str(res));
        Py_DECREF(that);
        return NULL;
    }
    // const int stat = mesh_dual_from_primal(this, that);
    // if (stat != 0)
    // {
    //     Py_DECREF(that);
    //     return NULL;
    // }
    return (PyObject *)that;
}

static PyObject *maifold2d_from_irregular(PyObject *type, PyObject *args, PyObject *kwargs)
{
    unsigned npts;
    PyObject *arg_lines;
    PyObject *arg_surf;
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "IOO",
                                     (char *[4]){"n_points", "line_connectivity", "surface_connectivity", NULL}, &npts,
                                     &arg_lines, &arg_surf))
    {
        return NULL;
    }

    PyArrayObject *const line_array = (PyArrayObject *)PyArray_FromAny(
        arg_lines, PyArray_DescrFromType(NPY_INT), 2, 2, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED, NULL);
    if (!line_array)
        return NULL;
    const unsigned n_lines = PyArray_DIM(line_array, 0);
    if (PyArray_DIM(line_array, 1) != 2)
    {
        PyErr_Format(PyExc_ValueError,
                     "Connectivity array must have the shape (N, 2), but instead its shape was (%u, %u).",
                     (unsigned)PyArray_DIM(line_array, 0), (unsigned)PyArray_DIM(line_array, 1));
        Py_DECREF(line_array);
        return NULL;
    }
    PyTypeObject *const obj_type = (PyTypeObject *)type;
    manifold2d_object_t *const this = (manifold2d_object_t *)obj_type->tp_alloc(obj_type, 0);
    if (!this)
    {
        Py_DECREF(line_array);
        return NULL;
    }

    this->manifold.n_points = npts;
    this->manifold.n_lines = n_lines;
    this->manifold.n_surfaces = 0;

    this->manifold.lines = NULL;
    this->manifold.surf_lines = NULL;
    this->manifold.surf_counts = NULL;

    this->manifold.lines = cutl_alloc(&SYSTEM_ALLOCATOR, sizeof *this->manifold.lines * n_lines);
    if (!this->manifold.lines)
    {
        Py_DECREF(this);
        Py_DECREF(line_array);
        return NULL;
    }

    const int *restrict p_in = PyArray_DATA(line_array);
    for (unsigned i_ln = 0; i_ln < n_lines; ++i_ln)
    {
        const geo_id_t begin = geo_id_pack(p_in[0]);
        const geo_id_t end = geo_id_pack(p_in[1]);

        if (begin.index >= npts || end.index >= npts)
        {
            PyErr_Format(PyExc_ValueError, "Line %u has points (%u, %u), but there were only %u points specified.",
                         i_ln, begin.index, end.index, npts);
            Py_DECREF(this);
            Py_DECREF(line_array);
            return NULL;
        }
        this->manifold.lines[i_ln] = (line_t){
            .begin = begin,
            .end = end,
        };
        p_in += 2;
    }
    Py_DECREF(line_array);

    PyObject *seq = PySequence_Fast(arg_surf, "Surface connectivity must be a sequence");
    if (!seq)
    {
        Py_DECREF(this);
        return NULL;
    }
    this->manifold.n_surfaces = PySequence_Fast_GET_SIZE(seq);

    this->manifold.surf_counts =
        cutl_alloc(&SYSTEM_ALLOCATOR, sizeof *this->manifold.surf_counts * (this->manifold.n_surfaces + 1));
    if (!this->manifold.surf_counts)
    {
        Py_DECREF(this);
        return NULL;
    }

    // Count up the offsets
    this->manifold.surf_counts[0] = 0;
    size_t n_surf_lines = 0;

    PyObject *const *const seq_items = PySequence_Fast_ITEMS(seq);
    for (unsigned i = 0; i < this->manifold.n_surfaces; ++i)
    {
        PyObject *const o = seq_items[i];
        const Py_ssize_t sz = PySequence_Size(o);
        if (sz < 0)
        {
            Py_DECREF(seq);
            Py_DECREF(this);
            return NULL;
        }
        n_surf_lines += (size_t)sz;
        this->manifold.surf_counts[i + 1] = n_surf_lines;
    }

    this->manifold.surf_lines = cutl_alloc(&SYSTEM_ALLOCATOR, sizeof(*this->manifold.surf_lines) * n_surf_lines);
    if (!this->manifold.surf_lines)
    {
        Py_DECREF(seq);
        Py_DECREF(this);
        return NULL;
    }
    int same_size = 1;
    const size_t first_size =
        this->manifold.n_surfaces ? this->manifold.surf_counts[1] - this->manifold.surf_counts[0] : 0;
    for (unsigned i = 0; i < this->manifold.n_surfaces; ++i)
    {
        PyObject *const o = seq_items[i];
        PyArrayObject *const array = (PyArrayObject *)PyArray_FromAny(o, PyArray_DescrFromType(NPY_INT), 1, 1,
                                                                      NPY_ARRAY_ALIGNED | NPY_ARRAY_C_CONTIGUOUS, NULL);
        if (!array)
        {
            Py_DECREF(seq);
            Py_DECREF(this);
            return NULL;
        }
        const int *ptr = PyArray_DATA(array);
        const size_t offset = this->manifold.surf_counts[i];
        const size_t len = this->manifold.surf_counts[i + 1] - offset;
        geo_id_t *const surf_lines = this->manifold.surf_lines + offset;
        for (unsigned j = 0; j < len; ++j)
        {
            const geo_id_t id = geo_id_pack(ptr[j]);
            if (id.index != GEO_ID_INVALID && id.index >= this->manifold.n_lines)
            {
                PyErr_Format(PyExc_ValueError,
                             "Surface %u specified a line with index %u, which is larger than"
                             "the number of lines in the mesh (%u).",
                             i, id.index, this->manifold.n_lines);
                Py_DECREF(array);
                Py_DECREF(seq);
                Py_DECREF(this);
                return NULL;
            }
            surf_lines[j] = id;
        }
        Py_DECREF(array);

        geo_id_t end;
        {
            const geo_id_t id1 = surf_lines[len - 1];
            if (id1.reverse)
            {
                end = this->manifold.lines[id1.index].begin;
            }
            else
            {
                end = this->manifold.lines[id1.index].end;
            }
        }
        for (unsigned j = 0; j < len; ++j)
        {

            geo_id_t begin, new_end;
            const geo_id_t id2 = surf_lines[j];
            if (id2.reverse)
            {
                begin = this->manifold.lines[id2.index].end;
                new_end = this->manifold.lines[id2.index].begin;
            }
            else
            {
                begin = this->manifold.lines[id2.index].begin;
                new_end = this->manifold.lines[id2.index].end;
            }

            if (begin.index != end.index)
            {
                PyErr_Format(PyExc_ValueError,
                             "Line %u does not end (point %u) where line %u begins (point %u) for surface %u.",
                             j == 0 ? len - 1 : j - 1, end.index, j, begin.index, i);
                Py_DECREF(seq);
                Py_DECREF(this);
                return NULL;
            }
            end = new_end;
        }
        same_size = same_size && (len == first_size);
    }
    Py_DECREF(seq);
    if (same_size)
    {
        PyErr_WarnFormat(PyExc_UserWarning, 0,
                         "Consider calling the Manifold2D.from_regular,"
                         " since all surfaces have the same length of %u.",
                         (unsigned)first_size);
    }

    return (PyObject *)this;
}

static PyObject *maifold2d_from_regular(PyObject *type, PyObject *args, PyObject *kwargs)
{
    unsigned npts;
    PyObject *arg_lines;
    PyObject *arg_surf;
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "IOO",
                                     (char *[4]){"n_points", "line_connectivity", "surface_connectivity", NULL}, &npts,
                                     &arg_lines, &arg_surf))
    {
        return NULL;
    }

    PyArrayObject *const line_array = (PyArrayObject *)PyArray_FromAny(
        arg_lines, PyArray_DescrFromType(NPY_INT), 2, 2, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED, NULL);
    if (!line_array)
        return NULL;
    const unsigned n_lines = PyArray_DIM(line_array, 0);
    if (PyArray_DIM(line_array, 1) != 2)
    {
        PyErr_Format(PyExc_ValueError,
                     "Connectivity array must have the shape (N, 2), but instead its shape was (%u, %u).",
                     (unsigned)PyArray_DIM(line_array, 0), (unsigned)PyArray_DIM(line_array, 1));
        Py_DECREF(line_array);
        return NULL;
    }
    PyTypeObject *const obj_type = (PyTypeObject *)type;
    manifold2d_object_t *const this = (manifold2d_object_t *)obj_type->tp_alloc(obj_type, 0);
    if (!this)
    {
        Py_DECREF(line_array);
        return NULL;
    }

    this->manifold.n_points = npts;
    this->manifold.n_lines = n_lines;
    this->manifold.n_surfaces = 0;

    this->manifold.lines = NULL;
    this->manifold.surf_lines = NULL;
    this->manifold.surf_counts = NULL;

    this->manifold.lines = cutl_alloc(&SYSTEM_ALLOCATOR, sizeof *this->manifold.lines * n_lines);
    if (!this->manifold.lines)
    {
        Py_DECREF(this);
        Py_DECREF(line_array);
        return NULL;
    }

    const int *restrict p_in = PyArray_DATA(line_array);
    for (unsigned i_ln = 0; i_ln < n_lines; ++i_ln)
    {
        const geo_id_t begin = geo_id_pack(p_in[0]);
        const geo_id_t end = geo_id_pack(p_in[1]);

        if (begin.index >= npts || end.index >= npts)
        {
            PyErr_Format(PyExc_ValueError, "Line %u has points (%u, %u), but there were only %u points specified.",
                         i_ln, begin.index, end.index, npts);
            Py_DECREF(this);
            Py_DECREF(line_array);
            return NULL;
        }
        this->manifold.lines[i_ln] = (line_t){
            .begin = begin,
            .end = end,
        };
        p_in += 2;
    }
    Py_DECREF(line_array);

    PyArrayObject *const surfaces = (PyArrayObject *)PyArray_FromAny(arg_surf, PyArray_DescrFromType(NPY_INT), 2, 2,
                                                                     NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED, NULL);
    if (!surfaces)
    {
        Py_DECREF(this);
        return NULL;
    }
    this->manifold.n_surfaces = PyArray_DIM(surfaces, 0);
    const unsigned n_per_surf = PyArray_DIM(surfaces, 1);

    this->manifold.surf_counts =
        cutl_alloc(&SYSTEM_ALLOCATOR, sizeof *this->manifold.surf_counts * (this->manifold.n_surfaces + 1));
    if (!this->manifold.surf_counts)
    {
        Py_DECREF(this);
        return NULL;
    }
    this->manifold.surf_lines =
        cutl_alloc(&SYSTEM_ALLOCATOR, sizeof(*this->manifold.surf_lines) * this->manifold.n_surfaces * n_per_surf);
    if (!this->manifold.surf_lines)
    {
        Py_DECREF(surfaces);
        Py_DECREF(this);
        return NULL;
    }

    // Count up the offsets
    this->manifold.surf_counts[0] = 0;
    const int *const lines = PyArray_DATA(surfaces);
    for (unsigned i = 0; i < this->manifold.n_surfaces; ++i)
    {
        const size_t offset = this->manifold.surf_counts[i];
        this->manifold.surf_counts[i + 1] = offset + n_per_surf;
        geo_id_t *const surf_lines = this->manifold.surf_lines + offset;
        for (unsigned j = 0; j < n_per_surf; ++j)
        {
            const geo_id_t id = geo_id_pack(lines[offset + j]);
            if (id.index != GEO_ID_INVALID && id.index >= this->manifold.n_lines)
            {
                PyErr_Format(PyExc_ValueError,
                             "Surface %u specified a line with index %u, which is larger than"
                             "the number of lines in the mesh (%u).",
                             i, id.index, this->manifold.n_lines);
                Py_DECREF(surfaces);
                Py_DECREF(this);
                return NULL;
            }
            surf_lines[j] = id;
        }

        geo_id_t end;
        {
            const geo_id_t id1 = surf_lines[n_per_surf - 1];
            if (id1.reverse)
            {
                end = this->manifold.lines[id1.index].begin;
            }
            else
            {
                end = this->manifold.lines[id1.index].end;
            }
        }
        for (unsigned j = 0; j < n_per_surf; ++j)
        {

            geo_id_t begin, new_end;
            const geo_id_t id2 = surf_lines[j];
            if (id2.reverse)
            {
                begin = this->manifold.lines[id2.index].end;
                new_end = this->manifold.lines[id2.index].begin;
            }
            else
            {
                begin = this->manifold.lines[id2.index].begin;
                new_end = this->manifold.lines[id2.index].end;
            }

            if (begin.index != end.index)
            {
                PyErr_Format(PyExc_ValueError,
                             "Line %u does not end (point %u) where line %u begins (point %u) for surface %u.",
                             j == 0 ? n_per_surf - 1 : j - 1, end.index, j, begin.index, i);
                Py_DECREF(surfaces);
                Py_DECREF(this);
                return NULL;
            }
            end = new_end;
        }
    }
    Py_DECREF(surfaces);

    return (PyObject *)this;
}

static PyMethodDef manifold2d_object_methods[] = {
    {.ml_name = "get_line",
     .ml_meth = manifold2d_get_line,
     .ml_flags = METH_O,
     .ml_doc = "get_line(index: int | GeoID, /) -> Line\n"
               "Get the line from the mesh.\n"
               "\n"
               "Parameters\n"
               "----------\n"
               "index : int or GeoID\n"
               "   Id of the line to get in 1-based indexing or GeoID. If negative,\n"
               "   the orientation will be reversed.\n"
               "\n"
               "Returns\n"
               "-------\n"
               "Line\n"
               "   Line object corresponding to the ID.\n"},
    {.ml_name = "get_surface",
     .ml_meth = manifold2d_get_surface,
     .ml_flags = METH_O,
     .ml_doc = "get_surface(index: int | GeoID, /) -> Surface\n"
               "Get the surface from the mesh.\n"
               "\n"
               "Parameters\n"
               "----------\n"
               "index : int or GeoID\n"
               "   Id of the surface to get in 1-based indexing or GeoID. If negative,\n"
               "   the orientation will be reversed.\n"
               "\n"
               "Returns\n"
               "-------\n"
               "Surface\n"
               "   Surface object corresponding to the ID.\n"},
    {.ml_name = "compute_dual",
     .ml_meth = manifold2d_compute_dual,
     .ml_flags = METH_NOARGS,
     .ml_doc = "compute_dual() -> Manifold2D\n"
               "Compute the dual to the manifold.\n"
               "\n"
               "A dual of each k-dimensional object in an n-dimensional space is a\n"
               "(n-k)-dimensional object. This means that duals of surfaces are points,\n"
               "duals of lines are also lines, and that the duals of points are surfaces.\n"
               "\n"
               "A dual line connects the dual points which correspond to surfaces which\n"
               "the line was a part of. Since the change over a line is computed by\n"
               "subtracting the value at the beginning from that at the end, the dual point\n"
               "which corresponds to the primal surface where the primal line has a\n"
               "positive orientation is the end point of the dual line and conversely the end\n"
               "dual point is the one corresponding to the surface which contained the primal\n"
               "line in the negative orientation. Since lines may only be contained in a\n"
               "single primal surface, they may have an invalid ID as either their beginning or\n"
               "their end. This can be used to determine if the line is actually a boundary of\n"
               "the manifold.\n"
               "\n"
               "A dual surface corresponds to a point and contains dual lines which correspond\n"
               "to primal lines, which contained the primal point of which the dual surface is\n"
               "the result of. The orientation of dual lines in this dual surface is positive if\n"
               "the primal line of which they are duals originated in the primal point in question\n"
               "and negative if it was their end point.\n"
               "\n"
               "Returns\n"
               "-------\n"
               "Manifold2D\n"
               "    Dual manifold.\n"},
    {.ml_name = "from_irregular",
     .ml_meth = (void *)maifold2d_from_irregular,
     .ml_flags = METH_VARARGS | METH_CLASS | METH_KEYWORDS,
     .ml_doc = "from_irregular(\n"
               "    n_points: int,\n"
               "    line_connectivity: array_like,\n"
               "    surface_connectivity: Sequence[array_like],\n"
               ") -> Self\n"
               "Create Manifold2D from surfaces with non-constant number of lines.\n"
               "\n"
               "Parameters\n"
               "----------\n"
               "n_points : int\n"
               "    Number of points in the mesh.\n"
               "line_connectivity : (N, 2) array_like\n"
               "    Connectivity of points which form lines in 0-based indexing.\n"
               "surface_connectivity : Sequence of array_like\n"
               "    Sequence of arrays specifying connectivity of mesh surfaces in 1-based\n"
               "    indexing, where a negative value means that the line's orientation is\n"
               "    reversed.\n"
               "\n"
               "Returns\n"
               "-------\n"
               "Self\n"
               "    Two dimensional manifold.\n"},
    {.ml_name = "from_regular",
     .ml_meth = (void *)maifold2d_from_regular,
     .ml_flags = METH_CLASS | METH_VARARGS | METH_KEYWORDS,
     .ml_doc = "from_regular(\n"
               "    n_points: int,\n"
               "    line_connectivity: array_like,\n"
               "    surface_connectivity: array_like,\n"
               ") -> Self\n"
               "Create Manifold2D from surfaces with constant number of lines.\n"
               "\n"
               "Parameters\n"
               "----------\n"
               "n_points : int\n"
               "    Number of points in the mesh.\n"
               "line_connectivity : (N, 2) array_like\n"
               "    Connectivity of points which form lines in 0-based indexing.\n"
               "surface_connectivity : array_like\n"
               "    Two dimensional array-like object specifying connectivity of mesh\n"
               "    surfaces in 1-based indexing, where a negative value means that\n"
               "    the line's orientation is reversed.\n"
               "\n"
               "Returns\n"
               "-------\n"
               "Self\n"
               "    Two dimensional manifold.\n"},
    {0},
};

static PyType_Slot manifold2d_slots[] = {
    {Py_tp_str, manifold2d_str},
    {Py_tp_doc, (void *)manifold2d_type_docstring},
    {Py_tp_methods, manifold2d_object_methods},
    {Py_tp_getset, manifold2d_getset},
    {Py_tp_dealloc, manifold2d_dealloc},
    {0, NULL},
};

INTERPLIB_INTERNAL
PyType_Spec manifold2d_type_spec = {
    .name = "interplib._interp.Manifold2D",
    .basicsize = sizeof(manifold2d_object_t),
    .itemsize = 0,
    .flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_IMMUTABLETYPE | Py_TPFLAGS_BASETYPE | Py_TPFLAGS_HEAPTYPE |
             Py_TPFLAGS_DISALLOW_INSTANTIATION,
    .slots = manifold2d_slots,
};
