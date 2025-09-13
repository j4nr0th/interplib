//
// Created by jan on 23.11.2024.
//

#include "geoid_object.h"

PyObject *geoid_repr(PyObject *self)
{
    const geo_id_object_t *this = (geo_id_object_t *)self;
    return PyUnicode_FromFormat("GeoID(%u, %u)", (unsigned)this->id.index, (unsigned)this->id.reverse);
}

PyObject *geoid_str(PyObject *self)
{
    const geo_id_object_t *this = (geo_id_object_t *)self;
    return PyUnicode_FromFormat("%c%u", (unsigned)this->id.reverse ? '-' : '+', (unsigned)this->id.index);
}

static PyObject *geoid_get_orientation(PyObject *self, void *Py_UNUSED(closure))
{
    const geo_id_object_t *this = (geo_id_object_t *)self;
    return PyBool_FromLong(this->id.reverse);
}

static int geoid_set_orientation(PyObject *self, PyObject *value, void *Py_UNUSED(closure))
{
    geo_id_object_t *this = (geo_id_object_t *)self;
    const int val = PyObject_IsTrue(value);
    if (val < 0)
    {
        return val;
    }
    this->id.reverse = val == 0 ? 0 : 1;
    return 0;
}

static PyObject *geoid_get_index(PyObject *self, void *Py_UNUSED(closure))
{
    const geo_id_object_t *this = (geo_id_object_t *)self;
    return PyLong_FromUnsignedLong(this->id.index);
}

static int geoid_set_index(PyObject *self, PyObject *value, void *Py_UNUSED(closure))
{
    geo_id_object_t *this = (geo_id_object_t *)self;
    const unsigned v = PyLong_AsUnsignedLong(value);
    if (PyErr_Occurred())
    {
        return -1;
    }
    this->id.index = v;
    return 0;
}

static PyGetSetDef geoid_getset[] = {
    {.name = "reversed",
     .get = geoid_get_orientation,
     .set = geoid_set_orientation,
     .doc = "bool : Is the orientation of the object reversed.",
     .closure = NULL},
    {.name = "index",
     .get = geoid_get_index,
     .set = geoid_set_index,
     .doc = "int : Index of the object referenced by id.",
     .closure = NULL},
    {0}, // sentinel
};

INTERPLIB_INTERNAL
geo_id_object_t *geo_id_object_from_value(PyTypeObject *const geoid_type, const geo_id_t id)
{
    geo_id_object_t *const this = (geo_id_object_t *)geoid_type->tp_alloc(geoid_type, 0);
    if (!this)
        return NULL;

    this->id = id;

    return this;
}

static PyObject *geoid_new(PyTypeObject *type, PyObject *args, PyObject *kwargs)
{
    int value;
    int orientation = 0;
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "i|p", (char *[3]){"index", "reverse", NULL}, &value, &orientation))
    {
        return NULL;
    }
    if (value < 0)
    {
        PyErr_SetString(PyExc_ValueError, "Can not use a negative index for a GeoID.");
        return NULL;
    }

    geo_id_object_t *const this = (geo_id_object_t *)type->tp_alloc(type, 0);
    if (!this)
        return NULL;

    this->id.reverse = orientation;
    this->id.index = value;

    return (PyObject *)this;
}

static PyObject *geoid_rich_compare(PyObject *self, PyObject *other, const int op)
{
    if (op != Py_EQ && op != Py_NE)
    {
        Py_RETURN_NOTIMPLEMENTED;
    }
    const geo_id_object_t *const this = (geo_id_object_t *)self;
    if (!PyObject_TypeCheck(other, Py_TYPE(self)))
    {
        Py_RETURN_NOTIMPLEMENTED;
    }
    const geo_id_object_t *const that = (geo_id_object_t *)other;
    const int val = this->id.reverse == that->id.reverse && this->id.index == that->id.index;
    if (op == Py_NE)
    {
        return PyBool_FromLong(!val);
    }
    return PyBool_FromLong(val);
}

static int geoid_bool(PyObject *self)
{
    const geo_id_object_t *const this = (geo_id_object_t *)self;
    return this->id.index != GEO_ID_INVALID;
}

static PyObject *geoid_negative(PyObject *self)
{
    const geo_id_object_t *const this = (geo_id_object_t *)self;
    return (PyObject *)geo_id_object_from_value(Py_TYPE(self),
                                                (geo_id_t){.index = this->id.index, .reverse = !this->id.reverse});
}

PyDoc_STRVAR(geoid_type_docstring, "GeoID(index: int, reverse=False)\n"
                                   "Type used to identify a geometrical object with an index and orientation.\n"
                                   "\n"
                                   "Parameters\n"
                                   "----------\n"
                                   "index : int\n"
                                   "    Index of the geometrical object.\n"
                                   "reverse : any, default: False\n"
                                   "    The object's orientation should be reversed.\n");

static PyType_Slot geo_id_type_slots[] = {
    {.slot = Py_tp_getset, .pfunc = geoid_getset},
    {.slot = Py_tp_repr, .pfunc = geoid_repr},
    {.slot = Py_tp_str, .pfunc = geoid_str},
    {.slot = Py_tp_doc, .pfunc = (void *)geoid_type_docstring},
    {.slot = Py_tp_new, .pfunc = geoid_new},
    {.slot = Py_tp_richcompare, .pfunc = geoid_rich_compare},
    {.slot = Py_nb_bool, .pfunc = geoid_bool},
    {.slot = Py_nb_negative, .pfunc = geoid_negative},
    {0},
};

INTERPLIB_INTERNAL
PyType_Spec geo_id_type_spec = {
    .name = "interplib._interp.GeoID",
    .basicsize = sizeof(geo_id_object_t),
    .itemsize = 0,
    .flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HEAPTYPE | Py_TPFLAGS_IMMUTABLETYPE,
    .slots = geo_id_type_slots,
};

INTERPLIB_INTERNAL
int geo_id_from_object(PyTypeObject *geoid_type, PyObject *o, geo_id_t *p_out)
{
    if (PyObject_TypeCheck(o, geoid_type))
    {
        const geo_id_object_t *this = (geo_id_object_t *)o;
        *p_out = this->id;
        return 0;
    }
    const long value = PyLong_AsLong(o);
    if (PyErr_Occurred())
        return -1;

    if (!value)
    {
        *p_out = (geo_id_t){.index = GEO_ID_INVALID, .reverse = 0};
    }
    else
    {
        *p_out = (geo_id_t){.index = labs(value) - 1, value < 0};
    }
    return 0;
}
