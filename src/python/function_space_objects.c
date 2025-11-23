#include "function_space_objects.h"
#include "../operations/multidim_iteration.h"
#include "basis_objects.h"

static PyObject *function_space_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    if (kwds && PyDict_Size(kwds))
    {
        PyErr_SetString(PyExc_TypeError, "Constructor takes no keyword arguments.");
        return NULL;
    }

    const unsigned n = PyTuple_GET_SIZE(args);
    if (n == 0)
    {
        PyErr_SetString(PyExc_TypeError, "Constructor requires at least one argument.");
        return NULL;
    }

    const interplib_module_state_t *state = interplib_get_module_state(type);
    if (!state)
        return NULL;

    // Check we got basis sets
    for (unsigned i = 0; i < n; ++i)
    {
        const PyObject *const val = PyTuple_GET_ITEM(args, i);
        if (!PyObject_TypeCheck(val, state->basis_spec_type))
        {
            PyErr_Format(PyExc_TypeError, "Argument %i was not a BasisSpec, but was instead %R.", i, val);
            return NULL;
        }
    }

    // Allocate the memory
    function_space_object *const this = (function_space_object *)type->tp_alloc(type, n);
    if (!this)
        return NULL;

    // Write the basis
    for (unsigned i = 0; i < n; ++i)
    {
        const basis_specs_object *const val = (basis_specs_object *)PyTuple_GET_ITEM(args, i);
        this->specs[i] = val->spec;
    }

    return (PyObject *)this;
}

static PyObject *function_space_get_dimensions(PyObject *self, void *Py_UNUSED(closure))
{
    const function_space_object *const this = (function_space_object *)self;
    return PyLong_FromSsize_t(Py_SIZE(this));
}

static PyObject *function_space_get_specs(PyObject *self, void *Py_UNUSED(closure))
{
    const function_space_object *const this = (function_space_object *)self;
    const interplib_module_state_t *state = interplib_get_module_state(Py_TYPE(self));
    PyTupleObject *const out = (PyTupleObject *)PyTuple_New(Py_SIZE(this));
    for (unsigned i = 0; i < Py_SIZE(this); ++i)
    {
        basis_specs_object *const val = basis_specs_object_create(state->basis_spec_type, this->specs[i]);
        if (!val)
        {
            Py_DECREF(out);
            return NULL;
        }
        PyTuple_SET_ITEM(out, i, val);
    }
    return (PyObject *)out;
}

static PyObject *function_space_get_orders(PyObject *self, void *Py_UNUSED(closure))
{
    const function_space_object *const this = (function_space_object *)self;
    PyTupleObject *const out = (PyTupleObject *)PyTuple_New(Py_SIZE(this));
    for (unsigned i = 0; i < Py_SIZE(this); ++i)
    {
        PyObject *const dim = PyLong_FromLong(this->specs[i].order);
        if (!dim)
        {
            Py_DECREF(out);
            return NULL;
        }
        PyTuple_SET_ITEM(out, i, dim);
    }
    return (PyObject *)out;
}

static int ensure_function_space_state(PyObject *self, PyTypeObject *defining_class, function_space_object **p_this,
                                       const interplib_module_state_t **p_state)
{
    const interplib_module_state_t *const state =
        defining_class ? PyType_GetModuleState(defining_class) : interplib_get_module_state(Py_TYPE(self));
    if (!state)
        return -1;
    if (!PyObject_TypeCheck(self, state->function_space_type))
    {
        PyErr_Format(PyExc_TypeError, "Expected %s, got %s", state->function_space_type->tp_name,
                     Py_TYPE(self)->tp_name);
        return -1;
    }

    *p_this = (function_space_object *)self;
    *p_state = state;
    return 0;
}

PyDoc_STRVAR(function_space_evaluate_docstring,
             "evaluate(*x: numpy.typing.NDArray[numpy.double], out: numpy.typing.NDArray[numpy.double] | None = None) "
             "-> numpy.typing.NDArray[numpy.double]:\n"
             "Evaluate basis functions at given locations.\n"
             "\n"
             "Parameters\n"
             "----------\n"
             "*x : array\n"
             "    Coordinates where the basis functions should be evaluated.\n"
             "    Each array corresponds to a dimension in the function space.\n"
             "out : array, optional\n"
             "    Array where the results should be written to. If not given, a new one\n"
             "    will be created and returned. It should have the same shape as ``x``,\n"
             "    but with an extra dimension added, the length of which is the total\n"
             "    number of basis functions in the function space.\n"
             "\n"
             "Returns\n"
             "-------\n"
             "array\n"
             "    Array of basis function values at the specified locations.\n");

static PyObject *function_space_evaluate(PyObject *self, PyTypeObject *defining_class, PyObject *const *args,
                                         const Py_ssize_t nargs, const PyObject *kwnames)
{
    function_space_object *this;
    const interplib_module_state_t *state;
    if (ensure_function_space_state(self, defining_class, &this, &state) < 0)
        return NULL;

    PyArrayObject *out = NULL;
    if (kwnames)
    {
        if (PyTuple_GET_SIZE(kwnames) != 1)
        {
            PyErr_SetString(PyExc_TypeError, "evaluate() takes exactly one keyword argument.");
            return NULL;
        }
        PyObject *const kwname = PyTuple_GET_ITEM(kwnames, 0);
        const char *kwrd = PyUnicode_AsUTF8(kwname);
        if (!kwrd)
            return NULL;
        if (strcmp(kwrd, "out") != 0)
        {
            PyErr_Format(PyExc_TypeError, "evaluate() got an unexpected keyword argument '%s'.", kwrd);
            return NULL;
        }
        out = (PyArrayObject *)args[nargs];
    }
    const unsigned n_basis_dims = Py_SIZE(this);
    CPYUTL_ASSERT(n_basis_dims > 0, "Function space should have at least one dimension.");

    if (nargs != n_basis_dims)
    {
        PyErr_Format(PyExc_TypeError,
                     "evaluate() takes exactly the same number of positional arguments as the dimension count (%u), "
                     "but %u were given.",
                     n_basis_dims, (unsigned)nargs);
        return NULL;
    }

    // Check all input arrays have the same shape, correct dtype, and the same flags
    npy_intp n_dim_in = 0;
    const npy_intp *p_dim_in = NULL;
    for (unsigned i = 0; i < n_basis_dims; ++i)
    {

        const PyArrayObject *const in = (PyArrayObject *)args[i];
        if (p_dim_in == NULL)
        {
            p_dim_in = PyArray_DIMS(in);
            n_dim_in = PyArray_NDIM(in);
        }
        if (check_input_array(in, n_dim_in, p_dim_in, NPY_DOUBLE, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED, "input") <
            0)
        {
            raise_exception_from_current(
                PyExc_ValueError,
                "All input arrays must have the exact same shape and have the correct data type and flags.");
            return NULL;
        }
    }

    const npy_intp n_dim_out = n_dim_in + n_basis_dims;
    npy_intp *p_dim_out = PyMem_Malloc(n_dim_out * sizeof(*p_dim_out));
    if (!p_dim_out)
    {
        return NULL;
    }
    CPYUTL_ASSERT(p_dim_in, "Input dims were not collected.");
    npy_uintp size_in = 1, max_size_basis = 1;
    for (npy_intp i = 0; i < n_dim_in; ++i)
    {
        const npy_intp dim = p_dim_in[i];
        size_in *= dim;
        p_dim_out[i] = dim;
    }
    npy_uintp size_out = size_in;
    for (npy_intp i = n_dim_in; i < n_dim_out; ++i)
    {
        const unsigned dim = this->specs[i - n_dim_in].order + 1;
        p_dim_out[i] = dim;
        size_out *= dim;
        // outer_prod_basis_cnt *= dim;
        max_size_basis = dim > max_size_basis ? dim : max_size_basis;
    }

    if (out)
    {
        // Check the output has the correct dimensions if given
        if (check_input_array(out, n_dim_out, p_dim_out, NPY_DOUBLE, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED,
                              "out") < 0)
            return NULL;
        Py_INCREF(out);
    }
    else
    {
        out = (PyArrayObject *)PyArray_SimpleNew(n_dim_out, p_dim_out, NPY_DOUBLE);
        if (!out)
        {
            PyMem_Free(p_dim_out);
            return NULL;
        }
    }
    PyMem_Free(p_dim_out);
    npy_double *const p_out = (npy_double *)PyArray_DATA(out);
    CPYUTL_ASSERT(size_out == (size_t)PyArray_SIZE(out), "Incorrect output size.");
    const double **const input_ptrs = PyMem_Malloc(n_basis_dims * sizeof(*input_ptrs));
    if (!input_ptrs)
    {
        Py_DECREF(out);
        return NULL;
    }
    for (unsigned i = 0; i < n_basis_dims; ++i)
    {
        input_ptrs[i] = (npy_double *)PyArray_DATA((PyArrayObject *)args[i]);
    }

    Py_BEGIN_ALLOW_THREADS;
    unsigned out_elements, work_elements, tmp_elements;
    size_t iter_size;
    basis_compute_outer_product_basis_required_memory(n_basis_dims, this->specs, size_in, &out_elements, &work_elements,
                                                      &tmp_elements, &iter_size);
    double *const basis_buffer = PyMem_RawMalloc((tmp_elements + work_elements) * sizeof(*basis_buffer));
    multidim_iterator_t *const iterator = PyMem_RawMalloc(iter_size);

    if (basis_buffer && iterator)
    {
        basis_compute_outer_product_basis(n_basis_dims, this->specs, size_in, input_ptrs, p_out,
                                          basis_buffer + tmp_elements, basis_buffer, iterator);
    }
    PyMem_RawFree(iterator);
    PyMem_RawFree(basis_buffer);
    Py_END_ALLOW_THREADS;
    PyMem_Free(input_ptrs);
    return (PyObject *)out;
}

PyType_Spec function_space_type_spec = {
    .name = "interplib._interp.FunctionSpace",
    .basicsize = sizeof(function_space_object),
    .itemsize = sizeof(basis_spec_t),
    .flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HEAPTYPE | Py_TPFLAGS_HAVE_GC | Py_TPFLAGS_IMMUTABLETYPE,
    .slots = (PyType_Slot[]){
        {Py_tp_traverse, heap_type_traverse_type},
        {Py_tp_new, function_space_new},
        {
            Py_tp_getset,
            (PyGetSetDef[]){
                {
                    .name = "dimension",
                    .get = function_space_get_dimensions,
                    .doc = "int:Number of dimensions in the function space.",
                },
                {
                    .name = "basis_specs",
                    .get = function_space_get_specs,
                    .doc = "tuple[BasisSpecs, ...] : Basis specifications that define the function space.",
                },
                {
                    .name = "orders",
                    .get = function_space_get_orders,
                    .doc = "tuple[int, ...] : Orders of the basis functions in the function space.",
                },
                {},
            },
        },
        {Py_tp_methods,
         (PyMethodDef[]){
             {
                 .ml_name = "evaluate",
                 .ml_meth = (void *)function_space_evaluate,
                 .ml_flags = METH_METHOD | METH_FASTCALL | METH_KEYWORDS,
                 .ml_doc = function_space_evaluate_docstring,
             },
             {},
         }},
        {},
    }};
