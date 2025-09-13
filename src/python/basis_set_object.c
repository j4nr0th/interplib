//
// Created by jan on 2025-09-11.
//

#include "basis_set_object.h"
#include "integration_rule_object.h"

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
static PyObject *basis_new(PyTypeObject *subtype, PyObject *args, PyObject *kwds)
{
    const char *basis_type_str;
    int order;
    integration_rule_object *integration_rule;
    basis_set_registry_t *basis_registry;

    {
        PyObject *const mod = PyType_GetModuleByDef(subtype, &interplib_module);
        if (!mod)
        {
            return NULL;
        }
        const interplib_module_state_t *const state = PyModule_GetState(mod);
        if (!state)
        {
            return NULL;
        }
        PyTypeObject *const integration_rule_type = state->integration_rule_type;
        basis_registry = state->basis_registry;

        if (!PyArg_ParseTuple(args, "siO!", &basis_type_str, &order, integration_rule_type, &integration_rule))
        {
            return NULL;
        }
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

    basis_set_object *const self = (basis_set_object *)subtype->tp_alloc(subtype, 0);
    if (!self)
        return NULL;

    const interp_result_t res = basis_set_registry_get_basis_set(
        basis_registry, &self->basis_set, integration_rule->rule, (basis_spec_t){.type = basis_type, .order = order});

    if (res != INTERP_SUCCESS)
    {
        PyErr_Format(PyExc_RuntimeError, "Failed to get basis set: %s (%s)", interp_error_str(res),
                     interp_error_msg(res));
        Py_DECREF(self);
        return NULL;
    }

    return (PyObject *)self;
}

/* Destructor */
static void basis_dealloc(basis_set_object *self)
{
    const basis_set_registry_t *const basis_registry = interplib_get_basis_registry(Py_TYPE(self));
    if (basis_registry)
    {
        basis_set_registry_release_basis_set(basis_registry, self->basis_set);
        self->basis_set = NULL;
    }
    Py_TYPE(self)->tp_free((PyObject *)self);
}

/* Properties */
static PyObject *basis_get_values(const basis_set_object *self, void *Py_UNUSED(closure))
{
    const npy_intp dims[2] = {self->basis_set->spec.order + 1, self->basis_set->integration_spec.order + 1};
    PyArrayObject *const array = (PyArrayObject *)PyArray_SimpleNew(2, dims, NPY_DOUBLE);
    if (!array)
    {
        return NULL;
    }
    double *const p_vals = PyArray_DATA(array);
    memcpy(p_vals, basis_set_values_all(self->basis_set), dims[0] * dims[1] * sizeof(*p_vals));
    return (PyObject *)array;
}

static PyObject *basis_get_order(const basis_set_object *self, void *Py_UNUSED(closure))
{
    return PyLong_FromLong(self->basis_set->spec.order);
}

static PyObject *basis_get_pointer(const basis_set_object *self, void *Py_UNUSED(closure))
{
    return PyLong_FromVoidPtr((void *)self->basis_set);
}

/* Get-set table */
static PyGetSetDef basis_getset[] = {
    {"values", (getter)basis_get_values, NULL,
     "numpy.typing.NDArray[numpy.double] : Values of all basis at integration points.", NULL},
    {"order", (getter)basis_get_order, NULL, "int : Order of the basis set.", NULL},
    {"pointer", (getter)basis_get_pointer, NULL, "int : Pointer of the basis set.", NULL},
    {NULL},
};

PyDoc_STRVAR(basis_set_docstring,
             "BasisSet(basis_type: interplib._typing.BasisType, order: int, integration_rule: IntegrationRule)\n"
             "Type that describes a set of basis functions.\n"
             "\n"
             "Parameters\n"
             "----------\n"
             "basis_type : interplib._typing.BasisType\n"
             "    Type of the basis used for the set.\n"
             "\n"
             "order : int\n"
             "    Order of the basis in the set.\n"
             "\n"
             "integration_rule : IntegrationRule\n"
             "    Integration rule used with the basis set.\n");

/* Slots for heap type */
static PyType_Slot basis_slots[] = {
    {Py_tp_new, (void *)basis_new},
    {Py_tp_dealloc, (void *)basis_dealloc},
    {Py_tp_getset, (void *)basis_getset},
    {Py_tp_doc, (void *)basis_set_docstring},
    {0, NULL},
};

/* Spec for heap type */
PyType_Spec basis_set_type_spec = {
    .name = "interplib._interp.BasisSet",
    .basicsize = sizeof(basis_set_object),
    .flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    .slots = basis_slots,
};
