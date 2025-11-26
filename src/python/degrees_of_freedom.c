#include "degrees_of_freedom.h"

#include "basis_objects.h"
#include "function_space_objects.h"
#include "integration_objects.h"

/**
 * Checks if a given Numpy array has a specific shape determined by the number
 * of dimensions and a basis specification for each dimension.
 *
 * @param arr A pointer to the Numpy array object to be checked.
 * @param ndim The number of dimensions against which the array is to be verified.
 * @param specs An array of basis specifications representing the expected shape
 *              for each dimension, where each dimension must match `order + 1`.
 * @return 1 if the array matches the specified shape, 0 otherwise.
 */
static int array_has_shape(const PyArrayObject *const arr, const unsigned ndim,
                           const basis_spec_t specs[const static ndim])
{
    if (PyArray_NDIM(arr) != (int)ndim)
        return 0;

    for (unsigned i = 0; i < ndim; ++i)
    {
        if (PyArray_DIM(arr, (int)i) != specs[i].order + 1)
            return 0;
    }
    return 1;
}

static PyObject *dof_new(PyTypeObject *subtype, PyObject *args, PyObject *kwds)
{
    const interplib_module_state_t *state = interplib_get_module_state(subtype);
    if (!state)
        return NULL;

    const function_space_object *space;
    PyObject *py_vals = NULL;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O!|O", (char *[]){"", "", NULL}, state->function_space_type, &space,
                                     &py_vals))
        return NULL;

    unsigned total_dofs = 1;
    const Py_ssize_t ndim = Py_SIZE(space);
    for (unsigned i = 0; i < ndim; ++i)
        total_dofs *= space->specs[i].order + 1;

    dof_object *const self = (dof_object *)subtype->tp_alloc(subtype, total_dofs);
    if (!self)
        return NULL;
    self->n_dims = ndim;
    self->basis_specs = NULL;

    basis_spec_t *const basis_specs = PyMem_Malloc(ndim * sizeof(*basis_specs));
    if (!basis_specs)
    {
        Py_DECREF(self);
        return NULL;
    }
    self->basis_specs = basis_specs;
    for (unsigned i = 0; i < ndim; ++i)
    {
        basis_specs[i] = space->specs[i];
    }
    if (py_vals)
    {
        PyArrayObject *const vals =
            (PyArrayObject *)PyArray_FROMANY(py_vals, NPY_DOUBLE, 0, 0, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED);
        if (!vals)
        {
            Py_DECREF(self);
            return NULL;
        }
        if (array_has_shape(vals, ndim, basis_specs) == 0 && PyArray_SIZE(vals) != total_dofs)
        {
            PyErr_Format(PyExc_ValueError,
                         "Values must be given either as a flat array with the correct number of elements (%u) or with "
                         "exact matching %u-dimensional shape.",
                         total_dofs, ndim);
            Py_DECREF(vals);
            Py_DECREF(self);
            return NULL;
        }
        const npy_double *const vals_data = PyArray_DATA(vals);
        for (unsigned i = 0; i < total_dofs; ++i)
            self->values[i] = vals_data[i];
        Py_DECREF(vals);
    }
    else
    {
        memset(self->values, 0, total_dofs * sizeof(*self->values));
    }

    return (PyObject *)self;
}

PyDoc_STRVAR(dof_docstring,
             "DegreesOfFreedom(function_space : FunctionSpace, values : numpy.typing.ArrayLike | None = None)\n"
             "Degrees of freedom associated with a function space.\n"
             "\n"
             "Parameters\n"
             "----------\n"
             "function_space : FunctionSpace\n"
             "    Function space the degrees of freedom belong to.\n"
             "values : array_like, optional\n"
             "    Values of the degrees of freedom. When not specified, they are zero initialized.\n");

static int ensure_dof_and_state(PyObject *self, PyTypeObject *defining_class, const interplib_module_state_t **p_state,
                                dof_object **p_object)
{
    const interplib_module_state_t *const state =
        defining_class ? PyType_GetModuleState(defining_class) : interplib_get_module_state(Py_TYPE(self));
    if (!state)
    {
        return -1;
    }
    if (!PyObject_TypeCheck(self, state->degrees_of_freedom_type))
    {
        PyErr_Format(PyExc_TypeError, "Expected %s, got %s", state->degrees_of_freedom_type->tp_name,
                     Py_TYPE(self)->tp_name);
        return -1;
    }
    *p_state = state;
    *p_object = (dof_object *)self;
    return 0;
}

static PyObject *dof_get_function_space(PyObject *self, void *Py_UNUSED(closure))
{
    const interplib_module_state_t *state;
    dof_object *this = (dof_object *)self;
    if (ensure_dof_and_state(self, NULL, &state, &this) < 0)
        return NULL;
    return (PyObject *)function_space_object_create(state->function_space_type, this->n_dims, this->basis_specs);
}

static PyObject *dof_get_total_number(PyObject *self, void *Py_UNUSED(closure))
{
    return PyLong_FromUnsignedLong(Py_SIZE(self));
}

static PyObject *dof_get_shape(PyObject *self, void *Py_UNUSED(closure))
{
    const dof_object *this = (dof_object *)self;
    PyTupleObject *const out = (PyTupleObject *)PyTuple_New(this->n_dims);
    if (!out)
        return NULL;
    for (unsigned i = 0; i < this->n_dims; ++i)
    {
        PyObject *const dim = PyLong_FromLong(this->basis_specs[i].order + 1);
        if (!dim)
        {
            Py_DECREF(out);
            return NULL;
        }
        PyTuple_SET_ITEM(out, i, dim);
    }
    return (PyObject *)out;
}

static PyObject *dof_get_values(PyObject *self, void *Py_UNUSED(closure))
{
    const dof_object *this = (dof_object *)self;
    npy_intp *const dims = PyMem_Malloc(this->n_dims * sizeof(*dims));
    if (!dims)
        return NULL;
    for (unsigned i = 0; i < this->n_dims; ++i)
        dims[i] = this->basis_specs[i].order + 1;
    PyArrayObject *const arr =
        (PyArrayObject *)PyArray_SimpleNewFromData(this->n_dims, dims, NPY_DOUBLE, (void *)this->values);
    PyMem_Free(dims);
    if (!arr)
        return NULL;
    if (PyArray_SetBaseObject(arr, self) < 0)
    {
        Py_DECREF(arr);
        return NULL;
    }
    Py_INCREF(self);
    return (PyObject *)arr;
}

static int dof_set_values(PyObject *self, PyObject *value, void *Py_UNUSED(closure))
{
    dof_object *this = (dof_object *)self;
    PyArrayObject *const arr = (PyArrayObject *)PyArray_FROMANY(value, NPY_DOUBLE, 0, 0, NPY_ARRAY_C_CONTIGUOUS);
    if (!arr)
        return -1;
    if (array_has_shape(arr, this->n_dims, this->basis_specs) == 0 && PyArray_SIZE(arr) != Py_SIZE(self))
    {
        PyErr_Format(PyExc_ValueError,
                     "Values must either be flat with %u elements or have exact correct %u-dimensional shape.",
                     (unsigned)Py_SIZE(arr), this->n_dims);
        Py_DECREF(arr);
        return -1;
    }
    const npy_double *const vals_data = PyArray_DATA(arr);
    for (unsigned i = 0; i < Py_SIZE(self); ++i)
        this->values[i] = vals_data[i];

    Py_DECREF(arr);
    return 0;
}

PyDoc_STRVAR(dof_reconstruct_at_integration_points_docstring,
             "reconstruct_at_integration_points(integration_space: IntegrationSpace, integration_registry: "
             "IntegrationRegistry = DEFAULT_INTEGRATION_REGISTRY, basis_registry: BasisRegistry = "
             "DEFAULT_BASIS_REGISTRY, *, out: numpy.typing.NDArray[numpy.double] | None = None, ) -> "
             "numpy.typing.NDArray[numpy.double]\n"
             "Reconstruct the function at the integration points of the given space.\n"
             "\n"
             "Parameters\n"
             "----------\n"
             "integration_space : IntegrationSpace\n"
             "    Integration space where the function should be reconstructed.\n"
             "integration_registry : IntegrationRegistry, default: DEFAULT_INTEGRATION_REGISTRY\n"
             "    Registry used to retrieve the integration rules.\n"
             "basis_registry : BasisRegistry, default: DEFAULT_BASIS_REGISTRY\n"
             "    Registry used to retrieve the basis specifications.\n"
             "out : array, optional\n"
             "    Array where the results should be written to. If not given, a new one\n"
             "    will be created and returned. It should have the same shape as the\n"
             "    integration points.\n"
             "\n"
             "Returns\n"
             "-------\n"
             "array\n"
             "    Array of reconstructed function values at the integration points.\n");

static PyObject *dof_reconstruct_at_integration_points(PyObject *self, PyTypeObject *defining_class,
                                                       PyObject *const *args, const Py_ssize_t nargs,
                                                       const PyObject *kwnames)
{
    const interplib_module_state_t *state;
    dof_object *this;
    if (ensure_dof_and_state(self, defining_class, &state, &this) < 0)
        return NULL;

    // Parse the arguments
    const integration_space_object *integration_space;
    const integration_registry_object *python_integration_registry =
        (integration_registry_object *)state->registry_integration;
    const basis_registry_object *python_basis_registry = (basis_registry_object *)state->registry_basis;
    PyArrayObject *out_array = NULL;
    if (parse_arguments_check(
            (cpyutl_argument_t[]){
                {
                    .type = CPYARG_TYPE_PYTHON,
                    .p_val = &integration_space,
                    .type_check = state->integration_space_type,
                    .kwname = "integration_space",
                },
                {
                    .type = CPYARG_TYPE_PYTHON,
                    .p_val = &python_integration_registry,
                    .type_check = state->integration_registry_type,
                    .kwname = "integration_registry",
                    .optional = 1,
                },
                {
                    .type = CPYARG_TYPE_PYTHON,
                    .p_val = &python_basis_registry,
                    .type_check = state->basis_registry_type,
                    .kwname = "basis_registry",
                    .optional = 1,
                },
                {
                    .type = CPYARG_TYPE_PYTHON,
                    .p_val = &out_array,
                    .type_check = &PyArray_Type,
                    .kwname = "out",
                    .optional = 1,
                    .kw_only = 1,
                },
                {},
            },
            args, nargs, kwnames) < 0)
        return NULL;

    // Do dimensions match
    const unsigned n_dof = Py_SIZE(self);
    if (Py_SIZE(integration_space) != this->n_dims)
    {
        PyErr_Format(PyExc_ValueError, "Expected integration space with %u dimensions, but it had only %u.",
                     this->n_dims, Py_SIZE(integration_space));
        return NULL;
    }

    // Check or create output
    if (out_array)
    {
        if (check_input_array(out_array, 0, (const npy_intp[]){}, NPY_DOUBLE,
                              NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED, "out") < 0)
        {
            raise_exception_from_current(PyExc_ValueError,
                                         "Output array must be a contiguous, aligned array of doubles.");
            return NULL;
        }
        if ((unsigned)PyArray_NDIM(out_array) != this->n_dims)
        {
            PyErr_Format(PyExc_ValueError, "Output array must have %u dimensions, but it had %u.", this->n_dims,
                         (unsigned)PyArray_NDIM(out_array));
            return NULL;
        }
        for (unsigned idim = 0; idim < this->n_dims; ++idim)
        {
            if (PyArray_DIM(out_array, (int)idim) != integration_space->specs[idim].order + 1)
            {
                PyErr_Format(PyExc_ValueError,
                             "Output array must have the exact same shape as the integration space, but dimension %u "
                             "did not match (integration space: %u, array: %u).",
                             idim, integration_space->specs[idim].order + 1,
                             (unsigned)PyArray_DIM(out_array, (int)idim));
                return NULL;
            }
        }
        // Good, now incref it
        Py_INCREF(out_array);
    }
    else
    {
        // Create it
        npy_intp *const p_dim_out = PyMem_Malloc(sizeof(*p_dim_out) * this->n_dims);
        if (!p_dim_out)
            return NULL;
        for (unsigned idim = 0; idim < this->n_dims; ++idim)
            p_dim_out[idim] = integration_space->specs[idim].order + 1;
        out_array = (PyArrayObject *)PyArray_SimpleNew(this->n_dims, p_dim_out, NPY_DOUBLE);
        PyMem_Free(p_dim_out);
        if (!out_array)
            return NULL;
    }

    // Create iterators
    multidim_iterator_t *const iter_int = integration_space_iterator(integration_space);
    if (!iter_int)
    {
        Py_DECREF(out_array);
        return NULL;
    }
    multidim_iterator_t *const iter_basis = python_basis_iterator(this->n_dims, this->basis_specs);
    if (!iter_basis)
    {
        PyMem_Free(iter_int);
        Py_DECREF(out_array);
        return NULL;
    }

    // Extract registry pointers
    basis_set_registry_t *const basis_registry = (basis_set_registry_t *)python_basis_registry->registry;
    const basis_set_t **basis_sets;

    // Get basis (and first the integration rules)
    {
        integration_rule_registry_t *const integration_registry =
            (integration_rule_registry_t *)python_integration_registry->registry;
        // Get integration rules
        const integration_rule_t **const integration_rules =
            python_integration_rules_get(this->n_dims, integration_space->specs, integration_registry);
        if (!integration_rules)
        {
            PyMem_Free(iter_basis);
            PyMem_Free(iter_int);
            Py_DECREF(out_array);
            return NULL;
        }
        basis_sets = python_basis_sets_get(this->n_dims, this->basis_specs, integration_rules, basis_registry);
        python_integration_rules_release(this->n_dims, integration_rules, integration_registry);
    }
    if (!basis_sets)
    {
        PyMem_Free(iter_basis);
        PyMem_Free(iter_int);
        Py_DECREF(out_array);
        return NULL;
    }

    npy_double *const ptr = PyArray_DATA(out_array);
    // Compute the values
    CPYUTL_ASSERT(multidim_iterator_total_size(iter_basis) == n_dof,
                  "Basis iterator should have the same number of elements as there are DoFs (%zu vs %u)",
                  multidim_iterator_total_size(iter_basis), n_dof);
    multidim_iterator_set_to_start(iter_int);
    while (!multidim_iterator_is_at_end(iter_int))
    {
        // Compute the point value
        double val = 0;
        multidim_iterator_set_to_start(iter_basis);
        while (!multidim_iterator_is_at_end(iter_basis))
        {
            // For each basis compute value at the integration point
            double basis_val = 1;
            for (unsigned idim = 0; idim < this->n_dims; ++idim)
            {
                basis_val *= basis_set_basis_values(
                    basis_sets[idim],
                    multidim_iterator_get_offset(iter_basis, idim))[multidim_iterator_get_offset(iter_int, idim)];
            }
            // Scale the basis value by the degree of freedom
            val += basis_val * this->values[multidim_iterator_get_flat_index(iter_basis)];
            multidim_iterator_advance(iter_basis, this->n_dims - 1, 1);
        }

        // Write output and advance the integration iterator
        ptr[multidim_iterator_get_flat_index(iter_int)] = val;
        multidim_iterator_advance(iter_int, this->n_dims - 1, 1);
    }

    // Free the iterator memory and release the basis sets
    PyMem_Free(iter_basis);
    PyMem_Free(iter_int);
    python_basis_sets_release(this->n_dims, basis_sets, basis_registry);
    return (PyObject *)out_array;
}

PyType_Spec degrees_of_freedom_type_spec = {
    .name = "interplib._interp.DegreesOfFreedom",
    .basicsize = sizeof(dof_object),
    .itemsize = sizeof(double),
    .flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HEAPTYPE | Py_TPFLAGS_HAVE_GC | Py_TPFLAGS_IMMUTABLETYPE,
    .slots = (PyType_Slot[]){
        {Py_tp_traverse, heap_type_traverse_type},
        {Py_tp_new, dof_new},
        {Py_tp_doc, (void *)dof_docstring},
        {Py_tp_getset,
         (PyGetSetDef[]){
             {
                 .name = "function_space",
                 .get = dof_get_function_space,
                 .doc = "FunctionSpace : Function space the degrees of freedom belong to.",
             },
             {
                 .name = "n_dofs",
                 .get = dof_get_total_number,
                 .doc = "int : Total number of degrees of freedom.",
             },
             {
                 .name = "shape",
                 .get = dof_get_shape,
                 .doc = "tuple[int, ...] : Shape of the degrees of freedom.",
             },
             {
                 .name = "values",
                 .get = dof_get_values,
                 .set = dof_set_values,
                 .doc = "numpy.typing.NDArray[numpy.double] : Values of the degrees of freedom.",
             },
             {},
         }},
        {Py_tp_methods,
         (PyMethodDef[]){
             {
                 .ml_name = "reconstruct_at_integration_points",
                 .ml_meth = (void *)dof_reconstruct_at_integration_points,
                 .ml_flags = METH_FASTCALL | METH_KEYWORDS | METH_METHOD,
                 .ml_doc = (void *)dof_reconstruct_at_integration_points_docstring,
             },
             {},
         }},
        {},
    }};
