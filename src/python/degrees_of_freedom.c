#include "degrees_of_freedom.h"
#include "function_space_objects.h"

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
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O!O", (char *[]){"", "", NULL}, state->function_space_type, &space,
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
        {},
    }};
