//
// Created by jan on 2025-09-11.
//

#include "basis_objects.h"
#include "integration_objects.h"

#include <numpy/ndarrayobject.h>
#include <string.h>

/* Enumeration-like strings for basis_type */
// static const char *basis_type_strings[] = {
//     "lagrange-uniform",
//     "lagrange-gauss",
//     "lagrange-gauss-lobatto",
//     "legendre",
//     "bernstein",
//     NULL
// };

static basis_set_type_t get_basis_type(const char *str)
{
    if (strcmp(str, "lagrange-uniform") == 0)
    {
        return BASIS_LAGRANGE_UNIFORM;
    }

    if (strcmp(str, "lagrange-gauss") == 0)
    {
        return BASIS_LAGRANGE_GAUSS;
    }

    if (strcmp(str, "lagrange-gauss-lobatto") == 0)
    {
        return BASIS_LAGRANGE_GAUSS_LOBATTO;
    }

    if (strcmp(str, "lagrange-chebyshev-gauss") == 0)
    {
        return BASIS_LAGRANGE_CHEBYSHEV_GAUSS;
    }

    if (strcmp(str, "legendre") == 0)
    {
        return BASIS_LEGENDRE;
    }

    if (strcmp(str, "bernstein") == 0)
    {
        return BASIS_BERNSTEIN;
    }

    return BASIS_INVALID;
}

static const char *basis_type_string(const basis_set_type_t type)
{
    switch (type)
    {
    case BASIS_LAGRANGE_UNIFORM:
        return "lagrange-uniform";
    case BASIS_LAGRANGE_GAUSS:
        return "lagrange-gauss";
    case BASIS_LAGRANGE_GAUSS_LOBATTO:
        return "lagrange-gauss-lobatto";
    case BASIS_LEGENDRE:
        return "legendre";
    case BASIS_BERNSTEIN:
        return "bernstein";
    default:
        return "invalid";
    }
}

/* __new__ */
static PyObject *basis_specs_new(PyTypeObject *subtype, PyObject *args, PyObject *kwds)
{
    const char *basis_type_str;
    int order;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "si", (char *[]){"", "", NULL}, &basis_type_str, &order))
    {
        return NULL;
    }

    basis_set_type_t const basis_type = get_basis_type(basis_type_str);
    if (basis_type == BASIS_INVALID)
    {
        PyErr_Format(PyExc_ValueError, "Unknown basis type string: %s", basis_type_str);
        return NULL;
    }

    if (order < 0)
    {
        PyErr_Format(PyExc_ValueError, "Order must be positive, but was given as %i.", order);
        return NULL;
    }

    basis_specs_object *const self = (basis_specs_object *)subtype->tp_alloc(subtype, 0);
    if (!self)
        return NULL;
    self->spec = (basis_spec_t){.order = order, .type = basis_type};

    return (PyObject *)self;
}

/* Properties */

static PyObject *basis_specs_get_order(const basis_specs_object *self, void *Py_UNUSED(closure))
{
    return PyLong_FromLong(self->spec.order);
}

static PyObject *basis_specs_get_type(const basis_specs_object *self, void *Py_UNUSED(closure))
{
    return PyUnicode_FromString(basis_type_string(self->spec.type));
}

/* Get-set table */
static PyGetSetDef basis_getset[] = {
    {
        "order",
        (getter)basis_specs_get_order,
        NULL,
        "int : Order of the basis set.",
        NULL,
    },
    {
        "type",
        (getter)basis_specs_get_type,
        NULL,
        "_BasisTypeHint : Type of the basis used for the set.",
        NULL,
    },
    {},
};

PyDoc_STRVAR(basis_specs_docstring, "BasisSpecs(basis_type: interplib._typing.BasisType, order: int)\n"
                                    "Type that describes a set of basis functions.\n"
                                    "\n"
                                    "Parameters\n"
                                    "----------\n"
                                    "basis_type : interplib._typing.BasisType\n"
                                    "    Type of the basis used for the set.\n"
                                    "\n"
                                    "order : int\n"
                                    "    Order of the basis in the set.\n");

/* Spec for heap type */
PyType_Spec basis_specs_type_spec = {
    .name = "interplib._interp.BasisSpecs",
    .basicsize = sizeof(basis_specs_object),
    .flags =
        Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE | Py_TPFLAGS_HEAPTYPE | Py_TPFLAGS_HAVE_GC | Py_TPFLAGS_IMMUTABLETYPE,
    .slots =
        (PyType_Slot[]){
            {Py_tp_new, (void *)basis_specs_new},
            {Py_tp_getset, (void *)basis_getset},
            {Py_tp_doc, (void *)basis_specs_docstring},
            {Py_tp_traverse, heap_type_traverse_type},
            {},
        },
};
