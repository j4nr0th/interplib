#include "mappings.h"

#include "basis_objects.h"
#include "degrees_of_freedom.h"
#include "integration_objects.h"

static PyObject *coordinate_map_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    const interplib_module_state_t *const state = interplib_get_module_state(type);
    if (!state)
        return NULL;
    dof_object *dofs;
    const integration_space_object *integration_space;
    const integration_registry_object *integration_registry =
        (integration_registry_object *)state->registry_integration;
    const basis_registry_object *basis_registry = (basis_registry_object *)state->registry_basis;
    if (!PyArg_ParseTupleAndKeywords(
            args, kwds, "O!O!|O!O!:CoordinateMap",
            (char *[]){"dofs", "integration_space", "integration_registry", "basis_registry", NULL},
            state->degrees_of_freedom_type, &dofs, state->integration_space_type, &integration_space,
            state->integration_registry_type, &integration_registry, state->basis_registry_type, &basis_registry))
        return NULL;

    // Call the reconstruct function on the DoFs
    PyArrayObject *const res_array = (PyArrayObject *)dof_reconstruct_at_integration_points(
        (PyObject *)dofs, state->degrees_of_freedom_type,
        (PyObject *const[]){(PyObject *)integration_space, (PyObject *)integration_registry,
                            (PyObject *)basis_registry},
        3, NULL);

    if (!res_array)
    {
        raise_exception_from_current(PyExc_RuntimeError, "Failed to reconstruct the DoFs.");
        return NULL;
    }

    const unsigned ndim = Py_SIZE(integration_space);

    const Py_ssize_t n_vals = PyArray_SIZE(res_array);
    coordinate_map_object *const self = (coordinate_map_object *)type->tp_alloc(type, n_vals * (ndim + 1));
    if (!self)
    {
        Py_DECREF(res_array);
        return NULL;
    }

    self->ndim = ndim;
    self->int_specs = PyMem_Malloc(ndim * sizeof(*self->int_specs));

    if (!self->int_specs)
    {
        Py_DECREF(res_array);
        Py_DECREF(self);
        return NULL;
    }
    for (unsigned idim = 0; idim < ndim; ++idim)
    {
        self->int_specs[idim] = integration_space->specs[idim];
    }

    memcpy(self->values, PyArray_DATA(res_array), n_vals * sizeof(*self->values));
    Py_DECREF(res_array);

    // Compute the gradients
    for (unsigned i = 0; i < ndim; ++i)
    {
        PyObject *const index = PyLong_FromLong(i);
        if (!index)
        {
            Py_DECREF(self);
            return NULL;
        }
        PyArrayObject *const res = (PyArrayObject *)dof_reconstruct_derivative_at_integration_points(
            (PyObject *)dofs, state->degrees_of_freedom_type,
            (PyObject *const[]){(PyObject *)integration_space, index, (PyObject *)integration_registry,
                                (PyObject *)basis_registry},
            3, NULL);
        Py_DECREF(index);
        if (!res)
        {
            Py_DECREF(self);
            return NULL;
        }
        memcpy(self->values + n_vals * (i + 1), PyArray_DATA(res), n_vals * sizeof(double));
        Py_DECREF(res);
    }

    return (PyObject *)self;
}

static void coordinate_map_dealloc(coordinate_map_object *self)
{
    PyObject_GC_UnTrack(self);
    PyMem_Free(self->int_specs);
    self->int_specs = NULL;
    PyTypeObject *const type = Py_TYPE(self);
    type->tp_free((PyObject *)self);
    Py_DECREF(type);
}

static PyObject *coordinate_map_get_dimension(PyObject *self, void *Py_UNUSED(closure))
{
    const coordinate_map_object *const this = (coordinate_map_object *)self;
    return PyLong_FromLong(this->ndim);
}

static PyObject *coordinate_map_get_values(PyObject *self, void *Py_UNUSED(closure))
{
    const coordinate_map_object *const this = (coordinate_map_object *)self;
    npy_intp *const dims = PyMem_Malloc(this->ndim * sizeof(*dims));
    if (!dims)
        return NULL;

    for (unsigned idim = 0; idim < this->ndim; ++idim)
    {
        dims[idim] = this->int_specs[idim].order + 1;
    }

    PyArrayObject *const res =
        (PyArrayObject *)PyArray_SimpleNewFromData(this->ndim, dims, NPY_DOUBLE, (void *)this->values);
    PyMem_Free(dims);
    if (!res)
        return NULL;

    CPYUTL_ASSERT(PyArray_SIZE(res) * (this->ndim + 1) == Py_SIZE(self), "These sizes should match!");
    if (PyArray_SetBaseObject((PyArrayObject *)res, (PyObject *)self) < 0)
    {
        Py_DECREF(res);
        return NULL;
    }
    Py_INCREF(this);
    return (PyObject *)res;
}

static PyObject *coordinate_map_get_integration_space(PyObject *self, void *Py_UNUSED(closure))
{
    const coordinate_map_object *const this = (coordinate_map_object *)self;
    const interplib_module_state_t *const state = interplib_get_module_state(Py_TYPE(self));
    if (!state)
        return NULL;
    integration_space_object *const res =
        (integration_space_object *)state->integration_space_type->tp_alloc(state->integration_space_type, this->ndim);
    if (!res)
        return NULL;
    for (unsigned idim = 0; idim < this->ndim; ++idim)
    {
        res->specs[idim] = this->int_specs[idim];
    }
    return (PyObject *)res;
}

static int ensure_coordinate_map_and_state(PyObject *self, PyTypeObject *defining_class,
                                           const interplib_module_state_t **p_state, coordinate_map_object **p_this)
{
    const interplib_module_state_t *const state =
        defining_class ? PyType_GetModuleState(defining_class) : interplib_get_module_state(Py_TYPE(self));
    if (!state)
        return -1;

    if (!PyObject_TypeCheck(self, state->coordinate_mapping_type))
    {
        PyErr_Format(PyExc_TypeError, "Expected a %s, but got a %s.", state->coordinate_mapping_type->tp_name,
                     Py_TYPE(self)->tp_name);
        return -1;
    }
    *p_state = state;
    *p_this = (coordinate_map_object *)self;
    return 0;
}

static PyObject *coordinate_map_gradient(PyObject *self, PyTypeObject *defining_class, PyObject *const *args,
                                         const Py_ssize_t nargs, const PyObject *const kwnames)
{
    const interplib_module_state_t *state;
    coordinate_map_object *this;
    if (ensure_coordinate_map_and_state(self, defining_class, &state, &this) < 0)
        return NULL;
    Py_ssize_t idx;
    if (parse_arguments_check(
            (cpyutl_argument_t[]){
                {.type = CPYARG_TYPE_SSIZE, .p_val = &idx},
                {},
            },
            args, nargs, kwnames) < 0)
        return NULL;
    if (idx < 0 || idx >= this->ndim)
    {
        PyErr_Format(PyExc_ValueError, "Expected dimension index in range [0, %zd), but got %zd.", this->ndim, idx);
        return NULL;
    }

    npy_intp *const dims = PyMem_Malloc(this->ndim * sizeof(*dims));
    if (!dims)
        return NULL;

    size_t total_cnt = 1;
    for (unsigned idim = 0; idim < this->ndim; ++idim)
    {
        const unsigned dim_size = this->int_specs[idim].order + 1;
        total_cnt *= dim_size;
        dims[idim] = dim_size;
    }

    PyArrayObject *const res = (PyArrayObject *)PyArray_SimpleNewFromData(this->ndim, dims, NPY_DOUBLE,
                                                                          (void *)(this->values + total_cnt * idx));
    PyMem_Free(dims);
    if (!res)
    {
        return NULL;
    }
    if (PyArray_SetBaseObject((PyArrayObject *)res, (PyObject *)self) < 0)
    {
        Py_DECREF(res);
        return NULL;
    }
    Py_INCREF(this);
    return (PyObject *)res;
}

static_assert(sizeof(*((coordinate_map_object *)0xB00B1E5)->values) == sizeof(double), "Nice");

PyType_Spec coordinate_map_type_spec = {
    .name = "interplib._interp.CoordinateMap",
    .basicsize = sizeof(coordinate_map_object),
    .itemsize = sizeof(*((coordinate_map_object *)0xB00B1E5)->values),
    .flags = Py_TPFLAGS_HEAPTYPE | Py_TPFLAGS_HAVE_GC | Py_TPFLAGS_DEFAULT | Py_TPFLAGS_IMMUTABLETYPE,
    .slots = (PyType_Slot[]){
        {Py_tp_traverse, heap_type_traverse_type},
        {Py_tp_dealloc, coordinate_map_dealloc},
        {Py_tp_new, coordinate_map_new},
        {Py_tp_getset,
         (PyGetSetDef[]){
             {
                 .name = "dimension",
                 .get = coordinate_map_get_dimension,
                 .doc = "int : Number of dimensions in the coordinate map.",
             },
             {
                 .name = "values",
                 .get = coordinate_map_get_values,
                 .doc = "numpy.typing.NDArray[numpy.double] : Values of the coordinate map at the integration points.",
             },
             {
                 .name = "integration_space",
                 .get = coordinate_map_get_integration_space,
                 .doc = "IntegrationSpace : Integration space used for the mapping.",
             },
             {},
         }},
        {
            Py_tp_methods,
            (PyMethodDef[]){
                {.ml_name = "gradient",
                 .ml_meth = (void *)coordinate_map_gradient,
                 .ml_flags = METH_METHOD | METH_FASTCALL | METH_KEYWORDS,
                 .ml_doc = "gradient(idim: int, /) -> numpy.typing.NDArray[numpy.double]\nRetrieve the gradient of the "
                           "coordinate map in given dimension."},
                {},
            },
        },
        {},
    }};

static void space_map_object_dealloc(PyObject *self)
{
    PyObject_GC_UnTrack(self);
    space_map_object *const this = (space_map_object *)self;
    PyTypeObject *const type = Py_TYPE(this);
    this->ndim = 0;
    PyMem_Free(this->int_specs);
    this->int_specs = NULL;
    for (unsigned i = 0; i < Py_SIZE(this); ++i)
    {
        coordinate_map_object *const map = this->maps[i];
        this->maps[i] = NULL;
        Py_DECREF(map);
    }
    type->tp_free((PyObject *)this);
    Py_DECREF(type);
}

/**
 * Compute determinant of transformation from gradients of coordinates.
 *
 * This function performs pivoted Gaussian elimination (or pivoted LU decomposition I guess) to reduce
 * the gradient matrix to an upper triangular form, at which point the product of diagonal entries is
 * the determinant.
 *
 * @param ndims[in] Number of dimensions.
 * @param metric_tensor[in, out] Square matrix with gradients of a single coordinate contiguous. Will be overwritten.
 * @param pivoted[out] Buffer array used to store pivot indices. Will be overwritten
 * @return Determinant of the matrix.
 */
static double compute_metric_determinant(const unsigned ndims, double metric_tensor[ndims][ndims],
                                         unsigned pivoted[ndims])
{
    double det = 1;
    for (unsigned i = 0; i < ndims; ++i)
    {
        // Get a new pivot for the current row
        unsigned pivot = i;
        for (unsigned i_pivot = i + 1; i_pivot < ndims; ++i_pivot)
        {
            // If this is the first row that was available, choose it as a pivot.
            // Otherwise, check if this row is a good candidate
            if (fabs(metric_tensor[pivoted[i_pivot]][i]) > fabs(metric_tensor[pivoted[pivot]][i]))
            {
                pivot = i_pivot;
            }
        }

        // Swap the rows
        if (pivot != i)
        {
            const unsigned tmp = pivoted[pivot];
            pivoted[pivot] = pivoted[i];
            pivoted[i] = tmp;
            det *= -1;
        }

        // Eliminate the rows in the gradient matrix using already used pivots
        for (unsigned i_row = i + 1; i_row < ndims; ++i_row)
        {
            const double factor = metric_tensor[pivoted[i_row]][i] / metric_tensor[pivoted[i]][i];
            for (unsigned j = i; j < ndims; ++j)
            {
                metric_tensor[pivoted[i_row]][j] -= metric_tensor[pivoted[i]][j] * factor;
            }
        }
    }

    // Compute the determinant from the pivoted diagonal
    for (unsigned i = 0; i < ndims; ++i)
    {
        det *= metric_tensor[pivoted[i]][i];
    }

    CPYUTL_ASSERT(det > 0, "Determinant should be positive, but it is %g.", det);

    return sqrt(det);
}

static PyObject *space_map_new(PyTypeObject *subtype, PyObject *args, PyObject *kwds)
{
    const interplib_module_state_t *const state = interplib_get_module_state(subtype);
    if (!state)
        return NULL;
    if (kwds && PyDict_Size(kwds))
    {
        PyErr_SetString(PyExc_TypeError, "SpaceMap takes no keyword arguments.");
        return NULL;
    }
    const unsigned n_coords = PyTuple_GET_SIZE(args);
    if (n_coords == 0)
    {
        PyErr_SetString(PyExc_TypeError, "SpaceMap requires at least one argument.");
        return NULL;
    }

    for (unsigned i = 0; i < n_coords; ++i)
    {
        PyObject *const o = PyTuple_GET_ITEM(args, i);
        if (!PyObject_TypeCheck(o, state->coordinate_mapping_type))
        {
            PyErr_Format(PyExc_TypeError, "Expected a %s, but got a %s.", state->coordinate_mapping_type->tp_name,
                         Py_TYPE(o)->tp_name);
            return NULL;
        }
    }

    space_map_object *const this = (space_map_object *)subtype->tp_alloc(subtype, n_coords);
    if (!this)
        return NULL;
    // Zero initialize
    this->ndim = 0;
    this->int_specs = NULL;
    this->determinant = NULL;
    for (unsigned i = 0; i < n_coords; ++i)
        this->maps[i] = NULL;

    // Copy the integration space from the first space, then check all others comply
    coordinate_map_object *const first_map = (coordinate_map_object *)PyTuple_GET_ITEM(args, 0);
    this->ndim = first_map->ndim;
    this->int_specs = PyMem_Malloc(this->ndim * sizeof(*this->int_specs));
    if (!this->int_specs)
    {
        Py_DECREF(this);
        return NULL;
    }
    for (unsigned i = 0; i < this->ndim; ++i)
    {
        this->int_specs[i] = first_map->int_specs[i];
    }
    this->maps[0] = first_map;
    Py_INCREF(first_map);

    for (unsigned i = 1; i < n_coords; ++i)
    {
        coordinate_map_object *const map = (coordinate_map_object *)PyTuple_GET_ITEM(args, i);
        if (map->ndim != this->ndim)
        {
            PyErr_Format(PyExc_ValueError,
                         "Expected all coordinate maps to have the same number of dimensions, but "
                         "got %zd and %zd.",
                         this->ndim, map->ndim);
            Py_DECREF(this);
            return NULL;
        }
        if (map->ndim != first_map->ndim)
        {
            PyErr_Format(
                PyExc_ValueError,
                "Expected all coordinate maps to have the same integration space, but the first and %u space have "
                "different integration spaces.",
                i + 1);
            Py_DECREF(this);
            return NULL;
        }
        for (unsigned idim = 0; idim < this->ndim; ++idim)
        {
            if (map->int_specs[idim].order != first_map->int_specs[idim].order ||
                map->int_specs[idim].type != first_map->int_specs[idim].type)
            {
                PyErr_Format(PyExc_ValueError,
                             "Expected all coordinate maps to have the same integration order and type, but got "
                             "order %d and type %d for dimension %u.",
                             first_map->int_specs[idim].order, first_map->int_specs[idim].type, idim);
                Py_DECREF(this);
                return NULL;
            }
        }
        this->maps[i] = map;
        Py_INCREF(map);
    }

    // Create the iterator to loop over all integration points
    multidim_iterator_t *const it = integration_specs_iterator(this->ndim, this->int_specs);
    multidim_iterator_set_to_start(it);

    const size_t total_points = multidim_iterator_total_size(it);
    // Allocate the output array
    double *const determinant = PyMem_Malloc(sizeof(*determinant) * total_points);
    if (!determinant)
    {
        Py_DECREF(this);
        return NULL;
    }

    // Allocate work arrays
    const unsigned n_dim_in = this->ndim;
    const unsigned n_dim_out = n_coords;
    double *const metric_tensor = PyMem_RawMalloc(sizeof(*metric_tensor) * n_dim_out * n_dim_out);
    if (!metric_tensor)
    {
        PyMem_Free(determinant);
        Py_DECREF(this);
        return NULL;
    }
    unsigned *const pivoted = PyMem_RawMalloc(sizeof(*pivoted) * this->ndim);
    if (!pivoted)
    {
        PyMem_RawFree(metric_tensor);
        PyMem_Free(determinant);
        Py_DECREF(this);
        return NULL;
    }

    // Now we iterate over all the points
    while (!multidim_iterator_is_at_end(it))
    {
        // Get the index of the point
        const size_t i_pt = multidim_iterator_get_flat_index(it);

        for (unsigned idim = 0; idim < n_dim_out; ++idim)
        {
            // Exploit symmetry
            for (unsigned jdim = 0; jdim < idim + 1; ++jdim)
            {
                double metric_value = 0;
#pragma omp simd reduction(+ : metric_value)
                for (unsigned k = 0; k < n_dim_in; ++k)
                {
                    metric_value += this->maps[k]->values[(jdim + 1) * total_points + i_pt] *
                                    this->maps[k]->values[(idim + 1) * total_points + i_pt];
                }
                // Symmetry to the rescue!
                metric_tensor[idim * n_dim_out + jdim] = metric_value;
                metric_tensor[jdim * n_dim_out + idim] = metric_value;
            }
        }

        determinant[i_pt] = compute_metric_determinant(n_dim_out, (double (*)[n_dim_out])metric_tensor, pivoted);

        multidim_iterator_advance(it, this->ndim - 1, 1);
    }

    // Free work arrays
    PyMem_RawFree(pivoted);
    PyMem_RawFree(metric_tensor);

    // Free the iterator
    PyMem_Free(it);

    // Store the output
    this->determinant = determinant;
    // Return
    return (PyObject *)this;
}

static int ensure_space_map_and_state(PyObject *self, PyTypeObject *defining_class,
                                      const interplib_module_state_t **p_state, space_map_object **p_this)
{
    const interplib_module_state_t *const state =
        defining_class ? PyType_GetModuleState(defining_class) : interplib_get_module_state(Py_TYPE(self));
    if (!state)
        return -1;

    if (!PyObject_TypeCheck(self, state->space_mapping_type))
    {
        PyErr_Format(PyExc_TypeError, "Expected a %s, but got a %s.", state->space_mapping_type->tp_name,
                     Py_TYPE(self)->tp_name);
        return -1;
    }
    *p_state = state;
    *p_this = (space_map_object *)self;
    return 0;
}

static PyObject *space_map_get_coordinate_map(PyObject *self, PyTypeObject *defining_class, PyObject *const *args,
                                              const Py_ssize_t nargs, const PyObject *const kwnames)
{
    const interplib_module_state_t *state;
    space_map_object *this;
    if (ensure_space_map_and_state(self, defining_class, &state, &this) < 0)
        return NULL;
    Py_ssize_t idx;
    if (parse_arguments_check(
            (cpyutl_argument_t[]){
                {.type = CPYARG_TYPE_SSIZE, .p_val = &idx, .kwname = "idx"},
                {},
            },
            args, nargs, kwnames) < 0)
        return NULL;

    if (idx < 0 || idx >= Py_SIZE(this))
    {
        PyErr_Format(PyExc_ValueError, "Expected dimension index in range [0, %zd), but got %zd.", Py_SIZE(this), idx);
        return NULL;
    }

    coordinate_map_object *const res = this->maps[idx];
    Py_INCREF(res);
    return (PyObject *)res;
}

PyDoc_STRVAR(space_map_get_coordinate_map_docstring,
             "get_coordinate_map(idx: int) -> CoordinateMap\n"
             "Return the coordinate map for the specified dimension.\n"
             "\n"
             "Parameters\n"
             "----------\n"
             "idx : int\n"
             "    Index of the dimension for which the map shoudl be returned.\n"
             "\n"
             "Returns\n"
             "-------\n"
             "CoordinateMap\n"
             "    Map used for the specified coordinate.\n");

PyDoc_STRVAR(space_map_docstring, "SpaceMap(*coordinates: CoordinateMap)\n"
                                  "Mapping between a reference space and a physical space.\n"
                                  "\n"
                                  "A mapping from a reference space to a physical space, which maps the\n"
                                  ":math:`N`-dimensional reference space to an :math:`M`-dimensional\n"
                                  "physical space. With this mapping, it is possible to integrate a\n"
                                  "quantity on a deformed element.\n"
                                  "\n"
                                  "Parameters\n"
                                  "----------\n"
                                  "*coordinates : CoordinateMap\n"
                                  "Maps for each coordinate of physical space. All of these must be\n"
                                  "defined on the same :class:`IntegrationSpace`.\n");

static PyObject *space_map_get_input_dimension(PyObject *self, void *Py_UNUSED(closure))
{
    const space_map_object *const this = (space_map_object *)self;
    return PyLong_FromLong(this->ndim);
}

static PyObject *space_map_get_output_dimension(PyObject *self, void *Py_UNUSED(closure))
{
    const space_map_object *const this = (space_map_object *)self;
    return PyLong_FromLong(Py_SIZE(this));
}

static PyObject *space_map_get_determinant(PyObject *self, void *Py_UNUSED(closure))
{
    const space_map_object *const this = (space_map_object *)self;
    if (!this->determinant)
    {
        PyErr_SetString(PyExc_NotImplementedError, "The determinant of the mapping is not yet implemented.");
        return NULL;
    }

    // Create the dims array
    npy_intp *const dims = PyMem_Malloc(sizeof(*dims) * this->ndim);
    if (!dims)
        return NULL;
    for (unsigned idim = 0; idim < this->ndim; ++idim)
    {
        dims[idim] = this->int_specs[idim].order + 1;
    }
    PyArrayObject *const res =
        (PyArrayObject *)PyArray_SimpleNewFromData(this->ndim, dims, NPY_DOUBLE, this->determinant);
    PyMem_Free(dims);
    if (!res)
    {
        return NULL;
    }
    if (PyArray_SetBaseObject(res, (PyObject *)this) < 0)
    {
        Py_DECREF(res);
        return NULL;
    }
    Py_INCREF(this);
    return (PyObject *)res;
}

static PyObject *space_map_get_integration_space(PyObject *self, void *Py_UNUSED(closure))
{
    const space_map_object *const this = (space_map_object *)self;
    const interplib_module_state_t *const state = interplib_get_module_state(Py_TYPE(self));
    if (!state)
        return NULL;
    integration_space_object *const res =
        (integration_space_object *)state->integration_space_type->tp_alloc(state->integration_space_type, this->ndim);
    if (!res)
        return NULL;
    for (unsigned idim = 0; idim < this->ndim; ++idim)
    {
        res->specs[idim] = this->int_specs[idim];
    }
    return (PyObject *)res;
}

PyType_Spec space_map_type_spec = {
    .name = "interplib._interp.SpaceMap",
    .basicsize = sizeof(space_map_object),
    .itemsize = sizeof(coordinate_map_object),
    .flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HEAPTYPE | Py_TPFLAGS_IMMUTABLETYPE | Py_TPFLAGS_HAVE_GC,
    .slots =
        (PyType_Slot[]){
            {Py_tp_traverse, heap_type_traverse_type},
            {Py_tp_dealloc, space_map_object_dealloc},
            {Py_tp_new, space_map_new},
            {Py_tp_doc, (void *)space_map_docstring},
            {Py_tp_getset,
             (PyGetSetDef[]){
                 {
                     .name = "input_dimensions",
                     .get = space_map_get_input_dimension,
                     .doc = "int : Dimension of the input/reference space.",
                 },
                 {
                     .name = "output_dimensions",
                     .get = space_map_get_output_dimension,
                     .doc = "int : Dimension of the output/physical space.",
                 },
                 {
                     .name = "determinant",
                     .get = space_map_get_determinant,
                     .doc = "numpy.typing.NDArray[numpy.double] : Array with the values of determinant at integration "
                            "points.",
                 },
                 {
                     .name = "integration_space",
                     .get = space_map_get_integration_space,
                     .doc = "IntegrationSpace : Integration space used by the mapping.",
                 },
                 {},
             }},
            {Py_tp_methods,
             (PyMethodDef[]){
                 {
                     .ml_name = "coordinate_map",
                     .ml_meth = (void *)space_map_get_coordinate_map,
                     .ml_flags = METH_FASTCALL | METH_KEYWORDS | METH_METHOD,
                     .ml_doc = space_map_get_coordinate_map_docstring,
                 },
                 {},
             }},
            {},
        },
};
