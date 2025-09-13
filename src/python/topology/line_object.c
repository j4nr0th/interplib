//
// Created by jan on 23.11.2024.
//

#include "line_object.h"

#include <stddef.h>

#include "geoid_object.h"
// This should be after other includes
#include <numpy/ndarrayobject.h>

static PyObject *line_object_repr(PyObject *self)
{
    const line_object_t *this = (line_object_t *)self;
    return PyUnicode_FromFormat("Line(GeoID(%u, %u), GeoID(%u, %u))", this->value.begin.index,
                                this->value.begin.reverse, this->value.end.index, this->value.end.reverse);
}

static PyObject *line_object_str(PyObject *self)
{
    const line_object_t *this = (line_object_t *)self;
    return PyUnicode_FromFormat("(%c%u -> %c%u)", this->value.begin.reverse ? '-' : '+', this->value.begin.index,
                                this->value.end.reverse ? '-' : '+', this->value.end.index);
}

line_object_t *line_from_indices(PyTypeObject *line_type, const geo_id_t begin, const geo_id_t end)
{
    line_object_t *const this = (line_object_t *)line_type->tp_alloc(line_type, 0);
    if (!this)
        return NULL;
    this->value.begin = begin;
    this->value.end = end;

    return this;
}

static PyObject *line_object_new(PyTypeObject *type, PyObject *args, PyObject *kwargs)
{
    PyObject *a1, *a2;
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OO", (char *[3]){"begin", "end", NULL}, &a1, &a2))
    {
        return NULL;
    }

    PyObject *const mod = PyType_GetModule(type);
    if (!mod)
        return NULL;
    const interplib_module_state_t *const state = (interplib_module_state_t *)PyModule_GetState(mod);
    if (!state)
        return NULL;

    geo_id_t begin, end;
    if (geo_id_from_object(state->geoid_type, a1, &begin) < 0 || geo_id_from_object(state->geoid_type, a2, &end) < 0)
        return NULL;

    line_object_t *const this = (line_object_t *)type->tp_alloc(type, 0);
    if (!this)
        return NULL;
    this->value.begin = begin;
    this->value.end = end;

    return (PyObject *)this;
}

static PyObject *line_object_rich_compare(PyObject *self, PyObject *other, const int op)
{
    if (op != Py_EQ && op != Py_NE)
    {
        Py_RETURN_NOTIMPLEMENTED;
    }
    const line_object_t *const this = (line_object_t *)self;
    if (!PyObject_TypeCheck(other, Py_TYPE(self)))
    {
        Py_RETURN_NOTIMPLEMENTED;
    }
    const line_object_t *const that = (line_object_t *)other;
    const int val =
        geo_id_compare(this->value.begin, that->value.begin) && geo_id_compare(this->value.end, that->value.end);
    if (op == Py_NE)
    {
        return PyBool_FromLong(!val);
    }
    return PyBool_FromLong(val);
}

PyDoc_STRVAR(line_object_type_docstring, "Line(begin: GeoID | int, end: GeoID | int)\n"
                                         "Geometrical object, which connects two points.\n"
                                         "\n"
                                         "Parameters\n"
                                         "----------\n"
                                         "begin : GeoID or int\n"
                                         "    ID of the point where the line beings.\n"
                                         "end : GeoID or int\n"
                                         "    ID of the point where the line ends.\n");

static PyObject *line_object_get_begin(PyObject *self, void *Py_UNUSED(closure))
{
    const line_object_t *this = (line_object_t *)self;

    PyObject *const mod = PyType_GetModule(Py_TYPE(self));
    if (!mod)
        return NULL;
    const interplib_module_state_t *const state = (interplib_module_state_t *)PyModule_GetState(mod);
    if (!state)
        return NULL;
    return (PyObject *)geo_id_object_from_value(state->geoid_type, this->value.begin);
}

static PyObject *line_object_get_end(PyObject *self, void *Py_UNUSED(closure))
{
    const line_object_t *this = (line_object_t *)self;
    PyObject *const mod = PyType_GetModule(Py_TYPE(self));
    if (!mod)
        return NULL;
    const interplib_module_state_t *const state = (interplib_module_state_t *)PyModule_GetState(mod);
    if (!state)
        return NULL;
    return (PyObject *)geo_id_object_from_value(state->geoid_type, this->value.end);
}

static PyGetSetDef line_object_getset[] = {
    {.name = "begin",
     .get = line_object_get_begin,
     .set = NULL,
     .doc = "GeoID : ID of the point where the line beings.",
     .closure = NULL},
    {.name = "end",
     .get = line_object_get_end,
     .set = NULL,
     .doc = "GeoID : ID of the point where the line ends.",
     .closure = NULL},
    {},
};

static PyObject *line_object_as_array(PyObject *self, PyObject *args, PyObject *kwds)
{
    PyArray_Descr *dtype = NULL;
    int b_copy = 1;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|Op", (char *[3]){"dtype", "copy", NULL}, &dtype, &b_copy))
    {
        return NULL;
    }

    if (!b_copy)
    {
        PyErr_SetString(PyExc_ValueError, "A copy is always created when converting to NDArray.");
        return NULL;
    }

    const line_object_t *this = (line_object_t *)self;
    const npy_intp size = 2;

    PyArrayObject *const out = (PyArrayObject *)PyArray_SimpleNew(1, &size, NPY_INT);
    if (!out)
        return NULL;

    int *const ptr = PyArray_DATA(out);
    ptr[0] = geo_id_unpack(this->value.begin);
    ptr[1] = geo_id_unpack(this->value.end);

    if (dtype)
    {
        PyObject *const new_out = PyArray_CastToType(out, dtype, 0);
        Py_DECREF(out);
        return new_out;
    }

    return (PyObject *)out;
}

static PyMethodDef line_methods[] = {
    {.ml_name = "__array__",
     .ml_meth = (void *)line_object_as_array,
     .ml_flags = METH_VARARGS | METH_KEYWORDS,
     .ml_doc = "__array__(self, dtype=None, copy=None) -> numpy.ndarray"},
    {},
};

static PyType_Slot line_type_slots[] = {
    {Py_tp_repr, line_object_repr},
    {Py_tp_str, line_object_str},
    {Py_tp_doc, (void *)line_object_type_docstring},
    {Py_tp_new, line_object_new},
    {Py_tp_richcompare, line_object_rich_compare},
    {Py_tp_getset, line_object_getset},
    {Py_tp_methods, line_methods},
    {0, NULL},
};

PyType_Spec line_type_spec = {
    .name = "interplib._interp.Line",
    .basicsize = sizeof(line_object_t),
    .itemsize = 0,
    .flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_IMMUTABLETYPE | Py_TPFLAGS_HEAPTYPE,
    .slots = line_type_slots,
};
