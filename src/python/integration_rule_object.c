#include "integration_rule_object.h"
#include <Python.h>
#include <numpy/ndarrayobject.h>

integration_registry_object *integration_registry_object_create(PyTypeObject *type)
{
    integration_registry_object *const self = (integration_registry_object *)type->tp_alloc(type, 0);
    if (!self)
        return NULL;
    const interp_result_t res = integration_rule_registry_create(&self->registry, 1, &SYSTEM_ALLOCATOR);
    if (res != INTERP_SUCCESS)
    {
        PyErr_Format(PyExc_RuntimeError, "Could not initialize integration rule registry: %s (%s)",
                     interp_error_str(res), interp_error_msg(res));
        Py_DECREF(self);
        return NULL;
    }
    return self;
}

static PyObject *integration_registry_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    if (PyTuple_GET_SIZE(args) != 0 || (kwds && PyDict_Size(kwds) != 0))
    {
        PyErr_SetString(PyExc_TypeError, "IntegrationRegistry takes no arguments.");
        return NULL;
    }

    integration_registry_object *const self = integration_registry_object_create(type);
    return (PyObject *)self;
}

static int ensure_integration_registry_state(PyObject *self, PyTypeObject *defining_class,
                                             integration_registry_object **p_this,
                                             const interplib_module_state_t **p_state)
{
    const interplib_module_state_t *const state =
        defining_class ? PyType_GetModuleState(defining_class) : interplib_get_module_state(Py_TYPE(self));
    if (!state)
        return -1;
    if (!PyObject_TypeCheck(self, state->integration_registry_type))
    {
        PyErr_Format(PyExc_TypeError, "Expected %s, got %s", state->integration_registry_type->tp_name,
                     Py_TYPE(self)->tp_name);
        return -1;
    }
    *p_this = (integration_registry_object *)self;
    *p_state = state;

    return 0;
}

static PyObject *integration_registry_repr(PyObject *self)
{
    const interplib_module_state_t *state;
    integration_registry_object *this;
    if (ensure_integration_registry_state(self, NULL, &this, &state) < 0)
        return NULL;

    return PyUnicode_FromFormat("<%s at %p>", state->integration_registry_type->tp_name, this->registry);
}

static void integration_registry_dealloc(integration_registry_object *self)
{
    PyObject_GC_UnTrack(self);
    integration_rule_registry_destroy(self->registry);
    PyTypeObject *const type = Py_TYPE(self);
    type->tp_free((PyObject *)self);
    Py_DECREF(type);
}

PyDoc_STRVAR(integration_registry_doc, "Registry for integration rules.\n"
                                       "\n"
                                       "This registry contains all available integration rules and caches them for\n"
                                       "efficient retrieval.\n");

static PyObject *integration_registry_usage(PyObject *self, PyTypeObject *defining_class,
                                            PyObject *const *Py_UNUSED(args), const Py_ssize_t nargs, PyObject *kwnames)
{
    if (nargs != 0 || kwnames != NULL)
    {
        PyErr_SetString(PyExc_TypeError, "usage() takes no arguments.");
        return NULL;
    }

    const interplib_module_state_t *state;
    integration_registry_object *this;
    if (ensure_integration_registry_state(self, defining_class, &this, &state) < 0)
        return NULL;

    unsigned count = integration_rule_get_rules(this->registry, 0, NULL);

    integration_rule_spec_t *const specs = PyMem_Malloc(count * sizeof(*specs));
    if (!specs)
        return NULL;
    {
        const unsigned new_cnt = integration_rule_get_rules(this->registry, count, specs);
        count = new_cnt < count ? new_cnt : count;
    }

    PyTupleObject *const out = (PyTupleObject *)PyTuple_New(count);
    if (!out)
        return NULL;
    for (unsigned i = 0; i < count; ++i)
    {
        PyObject *const val = Py_BuildValue("Is", specs[i].order, integration_rule_type_to_str(specs[i].type));
        if (!val)
        {
            Py_DECREF(out);
            PyMem_Free(specs);
            return NULL;
        }
        PyTuple_SET_ITEM(out, i, val);
    }
    PyMem_Free(specs);
    return (PyObject *)out;
}

static PyObject *integration_registry_clear(PyObject *self, PyTypeObject *defining_class,
                                            PyObject *const *Py_UNUSED(args), const Py_ssize_t nargs, PyObject *kwnames)
{
    if (nargs != 0 || kwnames != NULL)
    {
        PyErr_SetString(PyExc_TypeError, "usage() takes no arguments.");
        return NULL;
    }

    const interplib_module_state_t *state;
    integration_registry_object *this;
    if (ensure_integration_registry_state(self, defining_class, &this, &state) < 0)
        return NULL;

    integration_rule_registry_release_all_rules(this->registry);

    Py_RETURN_NONE;
}

PyType_Spec integration_registry_type_spec = {
    .name = "interplib._interp.IntegrationRegistry",
    .basicsize = sizeof(integration_registry_object),
    .itemsize = 0,
    .flags = Py_TPFLAGS_IMMUTABLETYPE | Py_TPFLAGS_HAVE_GC | Py_TPFLAGS_HEAPTYPE | Py_TPFLAGS_DEFAULT,
    .slots =
        (PyType_Slot[]){
            {Py_tp_new, integration_registry_new},
            {Py_tp_repr, integration_registry_repr},
            {Py_tp_traverse, heap_type_traverse_type},
            {Py_tp_dealloc, integration_registry_dealloc},
            {Py_tp_doc, (void *)integration_registry_doc},
            {Py_tp_methods,
             (PyMethodDef[]){
                 {
                     "usage",
                     (PyCFunction)integration_registry_usage,
                     METH_METHOD | METH_KEYWORDS | METH_FASTCALL,
                     "usage() -> tuple[tuple[int, str], ...]\nReturns a list of currently stored rules.",
                 },
                 {
                     "clear",
                     (PyCFunction)integration_registry_clear,
                     METH_METHOD | METH_KEYWORDS | METH_FASTCALL,
                     "clear() -> None\nClears all stored rules.",
                 },
                 {},
             }},
            {},
        },
};

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

/* __new__ method */
static PyObject *integration_specs_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
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

    integration_specs_object *self = (integration_specs_object *)type->tp_alloc(type, 0);
    if (!self)
        return NULL;

    self->spec = (integration_rule_spec_t){.order = order, .type = integration_type};

    return (PyObject *)self;
}

/* Property: order */
static PyObject *integration_spec_get_order(const integration_specs_object *self, void *Py_UNUSED(closure))
{
    return PyLong_FromLong(self->spec.order);
}

/* Property: accuracy */
static PyObject *integration_spec_get_accuracy(const integration_specs_object *self, void *Py_UNUSED(closure))
{
    return PyLong_FromLong(integration_rule_spec_get_accuracy(self->spec));
}

/* Property: method */
static PyObject *integration_spec_get_method(const integration_specs_object *self, void *Py_UNUSED(closure))
{
    return PyUnicode_FromString(integration_rule_type_to_str(self->spec.type));
}

static PyObject *integration_rule_repr(PyObject *self)
{
    const integration_specs_object *const this = (integration_specs_object *)self;
    const interplib_module_state_t *state = interplib_get_module_state(Py_TYPE(self));
    if (!state)
    {
        return NULL;
    }
    if (!PyObject_TypeCheck(self, state->integration_spec_type))
    {
        PyErr_Format(PyExc_TypeError, "Expected %s, got %s", state->integration_spec_type->tp_name,
                     Py_TYPE(self)->tp_name);
        return NULL;
    }

    return PyUnicode_FromFormat("IntegrationSpecs(%d, method=%s)", this->spec.order,
                                integration_rule_type_to_str(this->spec.type));
}

/* Get/set definitions */
static PyGetSetDef integration_rule_getset[] = {
    {
        "order",
        (getter)integration_spec_get_order,
        NULL,
        PyDoc_STR("int : Order of the integration rule."),
        NULL,
    },
    {
        "accuracy",
        (getter)integration_spec_get_accuracy,
        NULL,
        PyDoc_STR("int : Highest order of polynomial that is integrated exactly."),
        NULL,
    },
    {
        "method",
        (getter)integration_spec_get_method,
        NULL,
        PyDoc_STR("_IntegrationMethodHint : Method used for integration."),
        NULL,
    },
    {},
};

static int ensure_integration_spec_state(PyObject *self, PyTypeObject *defining_class,
                                         integration_specs_object **p_this, const interplib_module_state_t **p_state)
{
    const interplib_module_state_t *const state =
        defining_class ? PyType_GetModuleState(defining_class) : interplib_get_module_state(Py_TYPE(self));
    if (!state)
        return -1;
    if (!PyObject_TypeCheck(self, state->integration_spec_type))
    {
        PyErr_Format(PyExc_TypeError, "Expected %s, got %s", state->integration_spec_type->tp_name,
                     Py_TYPE(self)->tp_name);
        return -1;
    }
    *p_this = (integration_specs_object *)self;
    *p_state = state;
    return 0;
}

static PyArrayObject *integration_specs_prepare_array(PyObject *self, PyTypeObject *defining_class,
                                                      PyObject *const *args, const Py_ssize_t nargs,
                                                      const PyObject *kwnames, const integration_rule_t **p_rule,
                                                      integration_rule_registry_t **p_registry)
{
    const interplib_module_state_t *state;
    integration_specs_object *this;
    if (ensure_integration_spec_state(self, defining_class, &this, &state) < 0)
        return NULL;

    if (nargs > 1 || kwnames != NULL)
    {
        PyErr_SetString(PyExc_TypeError, "Only one argument may be passed to the method.");
        return NULL;
    }

    const integration_registry_object *registry_object;
    if (nargs == 1)
    {
        if (!PyObject_TypeCheck(args[0], state->integration_registry_type))
        {
            PyErr_Format(PyExc_TypeError, "Expected %s, got %s", state->integration_registry_type->tp_name,
                         Py_TYPE(args[0]));
            return NULL;
        }
        registry_object = (integration_registry_object *)args[0];
    }
    else
    {
        registry_object = (integration_registry_object *)state->registry_integration;
    }

    integration_rule_registry_t *const registry = registry_object->registry;

    const integration_rule_t *rule;
    const interp_result_t res = integration_rule_registry_get_rule(registry, this->spec, &rule);
    if (res != INTERP_SUCCESS)
    {
        PyErr_Format(PyExc_RuntimeError, "Failed to retrieve integration rule: %s (%s).", interp_error_str(res),
                     interp_error_msg(res));
        return NULL;
    }

    *p_rule = rule;
    *p_registry = registry;

    const npy_intp dims[] = {rule->n_nodes};
    return (PyArrayObject *)PyArray_SimpleNew(1, dims, NPY_DOUBLE);
}

PyDoc_STRVAR(integration_specs_nodes_docstring,
             "nodes(registry: IntegrationRegistry = DEFAULT_INTEGRATION_REGISTRY) -> ndarray[numpy.float64_t]\n"
             "Get the integration nodes.\n"
             "\n"
             "Parameters\n"
             "----------\n"
             "registry : interplib.IntegrationRegistry, default: DEFAULT_INTEGRATION_REGISTRY\n"
             "    Registry used to retrieve the integration rule.\n"
             "\n"
             "Returns\n"
             "-------\n"
             "array\n"
             "    Array of integration nodes.\n");

static PyObject *integration_specs_nodes(PyObject *self, PyTypeObject *defining_class, PyObject *const *args,
                                         const Py_ssize_t nargs, const PyObject *kwnames)
{
    const integration_rule_t *rule;
    integration_rule_registry_t *registry;
    PyArrayObject *out = integration_specs_prepare_array(self, defining_class, args, nargs, kwnames, &rule, &registry);
    if (out)
    {
        npy_double *const p_out = PyArray_DATA(out);
        const double *const nodes = integration_rule_nodes_const(rule);
        for (unsigned i = 0; i < rule->n_nodes; ++i)
        {
            p_out[i] = nodes[i];
        }
    }
    const interp_result_t res = integration_rule_registry_release_rule(registry, rule);
    (void)res;
    ASSERT(res == INTERP_SUCCESS, "Rule from the registry had to be successfully returned.");
    return (PyObject *)out;
}

PyDoc_STRVAR(integration_specs_weights_docstring,
             "weights(registry: IntegrationRegistry = DEFAULT_INTEGRATION_REGISTRY) -> ndarray[numpy.float64_t]\n"
             "Get the integration weights.\n"
             "\n"
             "Parameters\n"
             "----------\n"
             "registry : interplib.IntegrationRegistry, default: DEFAULT_INTEGRATION_REGISTRY\n"
             "    Registry used to retrieve the integration rule.\n"
             "\n"
             "Returns\n"
             "-------\n"
             "array\n"
             "    Array of integration weights.\n");

static PyObject *integration_specs_weights(PyObject *self, PyTypeObject *defining_class, PyObject *const *args,
                                           const Py_ssize_t nargs, const PyObject *kwnames)
{
    const integration_rule_t *rule;
    integration_rule_registry_t *registry;
    PyArrayObject *out = integration_specs_prepare_array(self, defining_class, args, nargs, kwnames, &rule, &registry);
    if (out)
    {
        npy_double *const p_out = PyArray_DATA(out);
        const double *const weights = integration_rule_weights_const(rule);
        for (unsigned i = 0; i < rule->n_nodes; ++i)
        {
            p_out[i] = weights[i];
        }
    }
    const interp_result_t res = integration_rule_registry_release_rule(registry, rule);
    (void)res;
    ASSERT(res == INTERP_SUCCESS, "Rule from the registry had to be successfully returned.");
    return (PyObject *)out;
}

PyDoc_STRVAR(integration_specs_docstring,
             "IntegrationSpecs(order : int , /, method: typing.Literal[\"gauss\", \"gauss-lobatto\"] = \"gauss\")\n"
             "Type that describes an integration rule.\n"
             "\n"
             "Parameters\n"
             "----------\n"
             "order : int\n"
             "    Order of the integration rule.\n"
             "\n"
             "method : typing.Literal[\"gauss\", \"gauss-lobatto\"], default: \"gauss\"\n"
             "    Method used for integration.\n");

PyType_Spec integration_specs_type_spec = {
    .name = "interplib._interp.IntegrationSpecs",
    .basicsize = sizeof(integration_specs_object),
    .itemsize = 0,
    .flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_IMMUTABLETYPE | Py_TPFLAGS_HEAPTYPE | Py_TPFLAGS_HAVE_GC,
    .slots =
        (PyType_Slot[]){
            {Py_tp_doc, (void *)integration_specs_docstring},
            {Py_tp_getset, integration_rule_getset},
            {Py_tp_new, integration_specs_new},
            {Py_tp_repr, (reprfunc)integration_rule_repr},
            {Py_tp_traverse, heap_type_traverse_type},
            {Py_tp_methods,
             (PyMethodDef[]){
                 {
                     .ml_name = "nodes",
                     .ml_meth = (void *)integration_specs_nodes,
                     .ml_flags = METH_METHOD | METH_KEYWORDS | METH_FASTCALL,
                     .ml_doc = integration_specs_nodes_docstring,
                 },
                 {
                     .ml_name = "weights",
                     .ml_meth = (void *)integration_specs_weights,
                     .ml_flags = METH_METHOD | METH_KEYWORDS | METH_FASTCALL,
                     .ml_doc = integration_specs_weights_docstring,
                 },
                 {},
             }},
            {},
        },
};
