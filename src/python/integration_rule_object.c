//
// Created by jan on 2025-09-10.
//

#include "integration_rule_object.h"
#include <Python.h>
#include <numpy/ndarrayobject.h>

/**
 * Determines the integration type based on the provided method string.
 *
 * @param method_str A string representing the integration method. Supported values are:
 *                   - "gauss" for Gauss-Legendre integration
 *                   - "gauss-lobatto" for Gauss-Lobatto integration
 * @return The integration rule type as an `integration_rule_type_t` enum value.
 *         Returns:
 *         - `INTEGRATION_RULE_TYPE_GAUSS_LEGENDRE` if the method is "gauss".
 *         - `INTEGRATION_RULE_TYPE_GAUSS_LOBATTO` if the method is "gauss-lobatto".
 *         - `INTEGRATION_RULE_TYPE_NONE` if the method is unknown.
 *         If the method is unknown, it also raises a Python ValueError with a descriptive message.
 */
static integration_rule_type_t determine_integration_type(const char *method_str)
{
    if (strcmp(method_str, "gauss") == 0)
    {
        return INTEGRATION_RULE_TYPE_GAUSS_LEGENDRE;
    }

    if (strcmp(method_str, "gauss-lobatto") == 0)
    {
        return INTEGRATION_RULE_TYPE_GAUSS_LOBATTO;
    }

    PyErr_Format(PyExc_ValueError, "Unknown integration method \"%s\".", method_str);
    return INTEGRATION_RULE_TYPE_NONE;
}

static const char *integration_rule_type_to_str(const integration_rule_type_t type)
{
    switch (type)
    {
    case INTEGRATION_RULE_TYPE_GAUSS_LEGENDRE:
        return "gauss";
    case INTEGRATION_RULE_TYPE_GAUSS_LOBATTO:
        return "gauss-lobatto";
    default:
        return "unknown";
    }
}

/* __new__ method */
static PyObject *integration_rule_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    int order;
    const char *method_str = "gauss";

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "i|s", (char *[]){"order", "method", NULL}, &order, &method_str))
    {
        return NULL;
    }

    if (order < 0)
    {
        PyErr_SetString(PyExc_ValueError, "Order must be a non-negative integer.");
        return NULL;
    }

    const integration_rule_type_t integration_type = determine_integration_type(method_str);
    if (integration_type == INTEGRATION_RULE_TYPE_NONE)
        return NULL;

    integration_rule_object *self = (integration_rule_object *)type->tp_alloc(type, 0);
    if (!self)
        return NULL;

    integration_rule_registry_t *const registry = interplib_get_integration_registry(type);
    if (!registry)
    {
        Py_DECREF(self);
        return NULL;
    }

    const interp_result_t res = integration_rule_registry_get_rule(
        registry, (integration_rule_spec_t){.type = integration_type, .order = order}, &self->rule);

    if (res != INTERP_SUCCESS)
    {
        PyErr_Format(PyExc_RuntimeError, "Failed to get integration rule: %s (%s)", interp_error_str(res),
                     interp_error_msg(res));
        Py_DECREF(self);
        return NULL;
    }

    return (PyObject *)self;
}

/* Property: order */
static PyObject *integration_rule_get_order(const integration_rule_object *self, void *Py_UNUSED(closure))
{
    return PyLong_FromLong(self->rule->spec.order);
}

/* Property: accuracy */
static PyObject *integration_rule_get_accuracy(const integration_rule_object *self, void *Py_UNUSED(closure))
{
    return PyLong_FromLong(self->rule->accuracy);
}

/* Property: nodes */
static PyObject *integration_rule_get_nodes(const integration_rule_object *self, void *Py_UNUSED(closure))
{
    const npy_intp n_nodes = self->rule->n_nodes;
    PyArrayObject *nodes = (PyArrayObject *)PyArray_SimpleNew(1, &n_nodes, NPY_DOUBLE);
    if (!nodes)
    {
        return NULL;
    }
    double *const p_nodes = PyArray_DATA(nodes);
    memcpy(p_nodes, integration_rule_nodes_const(self->rule), n_nodes * sizeof(*p_nodes));
    return (PyObject *)nodes;
}

/* Property: weights */
static PyObject *integration_rule_get_weights(const integration_rule_object *self, void *Py_UNUSED(closure))
{
    const npy_intp n_weights = self->rule->n_nodes;
    PyArrayObject *weights = (PyArrayObject *)PyArray_SimpleNew(1, &n_weights, NPY_DOUBLE);
    if (!weights)
    {
        return NULL;
    }
    double *const p_weights = PyArray_DATA(weights);
    memcpy(p_weights, integration_rule_weights_const(self->rule), n_weights * sizeof(*p_weights));
    return (PyObject *)weights;
}

/* Property: pointer */
static PyObject *integration_rule_get_pointer(const integration_rule_object *self, void *Py_UNUSED(closure))
{
    return PyLong_FromVoidPtr((void *)self->rule);
}

/* Deallocation */
static void integration_rule_dealloc(integration_rule_object *self)
{
    const integration_rule_registry_t *const registry = interplib_get_integration_registry(Py_TYPE(self));
    if (registry)
    {
        integration_rule_registry_release_rule(registry, self->rule);
        self->rule = NULL;
    }
    Py_TYPE(self)->tp_free((PyObject *)self);
}

static PyObject *integration_rule_repr(PyObject *self)
{
    const integration_rule_object *const this = (integration_rule_object *)self;
    return PyUnicode_FromFormat("IntegrationRule(%d, method=%s)", this->rule->spec.order,
                                integration_rule_type_to_str(this->rule->spec.type));
}

/* Get/set definitions */
static PyGetSetDef integration_rule_getset[] = {
    {
        "order",
        (getter)integration_rule_get_order,
        NULL,
        PyDoc_STR("int : Order of the integration rule."),
        NULL,
    },
    {
        "accuracy",
        (getter)integration_rule_get_accuracy,
        NULL,
        PyDoc_STR("int : Highest order of polynomial that is integrated exactly."),
        NULL,
    },
    {
        "nodes",
        (getter)integration_rule_get_nodes,
        NULL,
        PyDoc_STR("numpy.typing.NDArray[numpy.double] : Integration points on the reference domain."),
        NULL,
    },
    {
        "weights",
        (getter)integration_rule_get_weights,
        NULL,
        PyDoc_STR("numpy.typing.NDArray[numpy.double] : Weights associated with the integration nodes."),
        NULL,
    },
    {
        "pointer",
        (getter)integration_rule_get_pointer,
        NULL,
        PyDoc_STR("int : Pointer of the integration rule."),
        NULL,
    },
    {NULL},
};

PyDoc_STRVAR(integration_rule_doc,
             "IntegrationRule(order : int , /, method: typing.Literal[\"gauss\", \"gauss-lobatto\"] = \"gauss\")\n"
             "Type that describes an integration rule.\n"
             "\n"
             "Parameters\n"
             "----------\n"
             "order : int\n"
             "    Order of the integration rule.\n"
             "\n"
             "method : typing.Literal[\"gauss\", \"gauss-lobatto\"], default: \"gauss\"\n"
             "    Method used for integration.\n");

static PyType_Slot integration_rule_slots[] = {
    {Py_tp_dealloc, (destructor)integration_rule_dealloc},
    {Py_tp_doc, (void *)integration_rule_doc},
    {Py_tp_getset, integration_rule_getset},
    {Py_tp_new, integration_rule_new},
    {Py_tp_repr, (reprfunc)integration_rule_repr},
    {0, NULL},
};

PyType_Spec integration_rule_type_spec = {
    .name = "interplib._interp.IntegrationRule",
    .basicsize = sizeof(integration_rule_object),
    .itemsize = 0,
    .flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_IMMUTABLETYPE | Py_TPFLAGS_HEAPTYPE,
    .slots = integration_rule_slots,
};
