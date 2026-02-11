#include "mappings.h"

#include "../operations/matrices.h"
#include "basis_objects.h"
#include "cutl/iterators/combination_iterator.h"
#include "cutl/iterators/permutation_iterator.h"
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

    // Create the reconstruction state
    reconstruction_state_t recon_state;
    if (dof_reconstruction_state_init(dofs, integration_space, integration_registry, basis_registry, &recon_state) < 0)
        return NULL;
    const unsigned ndim = Py_SIZE(integration_space);
    const Py_ssize_t n_vals = (Py_ssize_t)multidim_iterator_total_size(recon_state.iter_int);
    coordinate_map_object *const self = (coordinate_map_object *)type->tp_alloc(type, n_vals * (ndim + 1));
    if (!self)
    {
        return NULL;
    }
    // Copy integration specs first
    self->ndim = ndim;
    self->int_specs = PyMem_Malloc(ndim * sizeof(*self->int_specs));
    if (!self->int_specs)
    {
        Py_DECREF(self);
        return NULL;
    }
    for (unsigned idim = 0; idim < ndim; ++idim)
    {
        self->int_specs[idim] = integration_space->specs[idim];
    }

    // Call the reconstruct function on the DoFs
    const unsigned ndofs = Py_SIZE(dofs);
    const double *const restrict pdofs = dofs->values;
    compute_integration_point_values(ndim, recon_state.iter_int, recon_state.iter_basis, recon_state.basis_sets, n_vals,
                                     self->values + 0 * n_vals, ndofs, pdofs);

    int *const derivative_array = PyMem_Malloc(sizeof(int) * ndim);
    if (!derivative_array)
    {
        Py_DECREF(self);
        return NULL;
    }
    for (unsigned i = 0; i < ndim; ++i)
    {
        derivative_array[i] = 0;
    }
    for (unsigned i = 0; i < ndim; ++i)
    {
        derivative_array[i] = 1; // Set the current dimension to use derivative
        // Compute with the specified derivatives
        compute_integration_point_values_derivatives(ndim, recon_state.iter_int, recon_state.iter_basis,
                                                     recon_state.basis_sets, derivative_array, n_vals,
                                                     self->values + (i + 1) * n_vals, ndofs, pdofs);
        derivative_array[i] = 0; // Reset the current dimension
    }
    PyMem_Free(derivative_array);

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

const double *coordinate_map_values(const coordinate_map_object *map)
{
    return map->values;
}

const double *coordinate_map_gradient(const coordinate_map_object *map, const unsigned dim)
{
    CPYUTL_ASSERT(dim < map->ndim, "Dimension index out of bounds.");
    size_t total_points = 1;
    for (unsigned i = 0; i < map->ndim; ++i)
        total_points *= map->int_specs[i].order + 1;
    return map->values + (dim + 1) * total_points;
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

static PyObject *coordinate_map_object_gradient(PyObject *self, PyTypeObject *defining_class, PyObject *const *args,
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

    PyArrayObject *const res = (PyArrayObject *)PyArray_SimpleNewFromData(
        this->ndim, dims, NPY_DOUBLE, (void *)(this->values + total_cnt * (idx + 1)));
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
                {
                    .ml_name = "gradient",
                    .ml_meth = (void *)coordinate_map_object_gradient,
                    .ml_flags = METH_METHOD | METH_FASTCALL | METH_KEYWORDS,
                    .ml_doc =
                        "gradient(idim: int, /) -> numpy.typing.NDArray[numpy.double]\nRetrieve the gradient of the "
                        "coordinate map in given dimension.",
                },
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
    PyMem_Free(this->determinant);
    this->determinant = NULL;
    PyMem_Free(this->inverse_maps);
    this->inverse_maps = NULL;
    for (unsigned i = 0; i < Py_SIZE(this); ++i)
    {
        coordinate_map_object *const map = this->maps[i];
        this->maps[i] = NULL;
        Py_DECREF(map);
    }
    type->tp_free((PyObject *)this);
    Py_DECREF(type);
}

static void calculate_determinants_and_inverse_maps(
    const unsigned n_dim, const unsigned n_maps, const coordinate_map_object *maps[static n_maps],
    const size_t total_points, double determinant[restrict const total_points], const size_t jacobian_size,
    double inverse_maps[restrict const total_points * jacobian_size], double jacobian[restrict const jacobian_size],
    double q_mat[restrict const n_maps * n_maps])
{
    // Now we iterate over all the points
    for (size_t i_pt = 0; i_pt < total_points; ++i_pt)
    {
        // Fill in the Jacobian
        const unsigned rows = n_maps;
        const unsigned cols = n_dim;
        for (unsigned idim = 0; idim < rows; ++idim)
        {
            for (unsigned jdim = 0; jdim < cols; ++jdim)
            {
                jacobian[idim * cols + jdim] = coordinate_map_gradient(maps[idim], jdim)[i_pt];
            }
        }

        const matrix_t jacobian_mat = (matrix_t){.rows = rows, .cols = cols, .values = jacobian};
        const matrix_t q_matrix = (matrix_t){.rows = rows, .cols = rows, .values = q_mat};

        // Decompose Jacobian into QR decomposition
        interp_result_t res = matrix_qr_decompose(&jacobian_mat, &q_matrix);
        (void)res;
        CPYUTL_ASSERT(res == INTERP_SUCCESS, "QR decomposition failed.");
        // Compute the determinant from the diagonal of the matrix
        double det = 1;
        for (unsigned i = 0; i < cols; ++i)
        {
            det *= jacobian[i * cols + i];
        }
        determinant[i_pt] = det;

        double *const p_inv_map = inverse_maps + i_pt * jacobian_size;
        const matrix_t out_mat = (matrix_t){.rows = cols, .cols = rows, .values = p_inv_map};
        // Copy the top part of q into out
        for (unsigned irow = 0; irow < cols; ++irow)
        {
            for (unsigned icol = 0; icol < rows; ++icol)
            {
                p_inv_map[irow * rows + icol] = q_mat[irow * rows + icol];
            }
        }

        // Use decomposition to compute "inverse". This is done simply by applying inverse of the
        // upper triangular (rows x rows) part of the jacobian to the matrix q_mat.
        res = matrix_back_substitute(&jacobian_mat, &out_mat);
        CPYUTL_ASSERT(res == INTERP_SUCCESS, "Back substitution failed.");
        (void)res;
    }
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
    const unsigned n_maps = PyTuple_GET_SIZE(args);
    if (n_maps == 0)
    {
        PyErr_SetString(PyExc_TypeError, "SpaceMap requires at least one argument.");
        return NULL;
    }

    for (unsigned i = 0; i < n_maps; ++i)
    {
        PyObject *const o = PyTuple_GET_ITEM(args, i);
        if (!PyObject_TypeCheck(o, state->coordinate_mapping_type))
        {
            PyErr_Format(PyExc_TypeError, "Expected a %s, but got a %s.", state->coordinate_mapping_type->tp_name,
                         Py_TYPE(o)->tp_name);
            return NULL;
        }
    }

    space_map_object *const this = (space_map_object *)subtype->tp_alloc(subtype, n_maps);
    if (!this)
        return NULL;
    // Zero initialize
    this->ndim = 0;
    this->int_specs = NULL;
    this->determinant = NULL;
    this->inverse_maps = NULL;
    for (unsigned i = 0; i < n_maps; ++i)
        this->maps[i] = NULL;

    // Copy the integration space from the first space, then check all others comply
    coordinate_map_object *const first_map = (coordinate_map_object *)PyTuple_GET_ITEM(args, 0);
    if (first_map->ndim > n_maps)
    {
        PyErr_Format(PyExc_ValueError,
                     "Can not construct a space map from reference domain with %u dimensions to %u physical "
                     "dimensions. The number of physical dimensions must be equal to, or greater than the number of "
                     "dimensions of the reference space.",
                     first_map->ndim, n_maps);
        return NULL;
    }
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

    for (unsigned i = 1; i < n_maps; ++i)
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

    const size_t total_points = integration_specs_total_points(this->ndim, this->int_specs);
    const size_t jacobian_size = (size_t)this->ndim * n_maps;

    // Allocate the output arrays
    double *const determinant = PyMem_Malloc(sizeof(*determinant) * total_points);
    if (!determinant)
    {
        Py_DECREF(this);
        return NULL;
    }
    double *const inverse_maps = PyMem_Malloc(sizeof(*inverse_maps) * total_points * jacobian_size);
    if (!inverse_maps)
    {
        PyMem_Free(determinant);
        Py_DECREF(this);
        return NULL;
    }

    // Allocate work arrays
    double *const jacobian = PyMem_RawMalloc(sizeof(*jacobian) * jacobian_size);
    if (!jacobian)
    {
        PyMem_Free(determinant);
        Py_DECREF(this);
        return NULL;
    }
    double *const q_mat = PyMem_RawMalloc(sizeof(*q_mat) * n_maps * n_maps);
    if (!q_mat)
    {
        PyMem_Free(jacobian);
        PyMem_Free(determinant);
        Py_DECREF(this);
        return NULL;
    }

    calculate_determinants_and_inverse_maps(this->ndim, n_maps, (const coordinate_map_object **)this->maps,
                                            total_points, determinant, jacobian_size, inverse_maps, jacobian, q_mat);

    // Free work arrays
    PyMem_RawFree(q_mat);
    PyMem_RawFree(jacobian);

    // Store the output
    this->determinant = determinant;
    this->inverse_maps = inverse_maps;
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

static PyObject *space_map_get_inverse_map(PyObject *self, void *Py_UNUSED(closure))
{
    const space_map_object *const this = (space_map_object *)self;
    npy_intp *const dims = PyMem_Malloc(sizeof(*dims) * (this->ndim + 2));
    if (!dims)
        return NULL;

    for (unsigned idim = 0; idim < this->ndim; ++idim)
    {
        dims[idim] = this->int_specs[idim].order + 1;
    }
    dims[this->ndim] = this->ndim;
    dims[this->ndim + 1] = Py_SIZE(this);

    PyArrayObject *const res =
        (PyArrayObject *)PyArray_SimpleNewFromData(this->ndim + 2, dims, NPY_DOUBLE, this->inverse_maps);
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

PyDoc_STRVAR(space_map_get_inverse_map_docstring,
             "numpy.typing.NDArray[numpy.double] : Local inverse transformation at each integration point.\n"
             "\n"
             "This array contains inverse mapping matrix, which is used\n"
             "for the contravarying components. When the dimension of the\n"
             "mapping space (as counted by :meth:`SpaceMap.output_dimensions`)\n"
             "is greater than the dimension of the reference space, this is a\n"
             "rectangular matrix, such that it maps the (rectangular) Jacobian\n"
             "to the identity matrix.\n");

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
                 {
                     .name = "inverse_map",
                     .get = space_map_get_inverse_map,
                     .doc = space_map_get_inverse_map_docstring,
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

size_t space_map_inverse_size_per_integration_point(const space_map_object *map)
{
    return map->ndim * Py_SIZE(map);
}

double space_map_forward_derivative(const space_map_object *map, const size_t integration_point_index,
                                    const unsigned idx_dim, const unsigned idx_coord)
{
    const coordinate_map_object *const map_dim = map->maps[idx_coord];
    return coordinate_map_gradient(map_dim, idx_dim)[integration_point_index];
}

double space_map_backward_derivative(const space_map_object *map, const size_t integration_point_index,
                                     const unsigned idx_dim, const unsigned idx_coord)
{
    const size_t n_coords = Py_SIZE(map);
    const size_t jacobian_size = (size_t)map->ndim * n_coords;
    return map->inverse_maps[integration_point_index * jacobian_size + idx_dim * n_coords + idx_coord];
}

const double *space_map_inverse_at_integration_point(const space_map_object *map, const size_t flat_index)
{
    return map->inverse_maps + flat_index * space_map_inverse_size_per_integration_point(map);
}

PyArrayObject *compute_basis_transform_impl(const space_map_object *map, const Py_ssize_t order)
{
    const unsigned n_maps = Py_SIZE(map);
    const unsigned n_dims = map->ndim;
    if (order <= 0 || order > n_dims || order > n_maps)
    {
        PyErr_Format(PyExc_ValueError, "Expected order in range (0, %u], but got %zd.",
                     n_dims < n_maps ? n_dims : n_maps, order);
        return NULL;
    }

    permutation_iterator_t *const iter_out_perm = PyMem_Malloc(permutation_iterator_required_memory(order, order));
    if (!iter_out_perm)
        return NULL;
    combination_iterator_t *const iter_out_comb = PyMem_Malloc(combination_iterator_required_memory(order));
    if (!iter_out_comb)
    {
        PyMem_Free(iter_out_perm);
        return NULL;
    }
    combination_iterator_t *const iter_in_comb = PyMem_Malloc(combination_iterator_required_memory(order));
    if (!iter_in_comb)
    {
        PyMem_Free(iter_out_comb);
        PyMem_Free(iter_out_perm);
        return NULL;
    }
    multidim_iterator_t *const iter_int_pts = integration_specs_iterator(n_dims, map->int_specs);
    if (!iter_int_pts)
    {
        PyMem_Free(iter_out_comb);
        PyMem_Free(iter_in_comb);
        PyMem_Free(iter_out_perm);
        return NULL;
    }
    combination_iterator_init(iter_in_comb, n_dims, order);
    combination_iterator_init(iter_out_comb, n_maps, order);
    permutation_iterator_init(iter_out_perm, order, order);

    const size_t total_points = multidim_iterator_total_size(iter_int_pts);
    const npy_intp out_dims[3] = {
        combination_iterator_total_count(iter_in_comb),
        combination_iterator_total_count(iter_out_comb),
        (npy_intp)total_points,
    };
    PyArrayObject *const res = (PyArrayObject *)PyArray_SimpleNew(3, out_dims, NPY_DOUBLE);
    if (!res)
    {
        PyMem_Free(iter_int_pts);
        PyMem_Free(iter_out_comb);
        PyMem_Free(iter_in_comb);
        PyMem_Free(iter_out_perm);
        return NULL;
    }
    npy_double *const ptr_out = PyArray_DATA(res);

    // Iterate over bases in the inputs space
    size_t idx_in = 0;
    combination_iterator_init(iter_in_comb, n_dims, order);
    while (!combination_iterator_is_done(iter_in_comb))
    {
        // Indices of current input dimensions
        const uint8_t *const current_in = combination_iterator_current(iter_in_comb);
        // Iterate over bases in the output space.
        size_t idx_out = 0;
        combination_iterator_init(iter_out_comb, n_maps, order);
        while (!combination_iterator_is_done(iter_out_comb))
        {
            // Indices of current output dimensions.
            const uint8_t *const current_out = combination_iterator_current(iter_out_comb);

            // Loop over integration points
            for (size_t idx_pt = 0; idx_pt < total_points; ++idx_pt)
            {
                // Total transformation coefficient, which we will accumulate for each possible basis.
                double val = 0.0;

                // Loop over all permutations of the current basis indices.
                permutation_iterator_init(iter_out_perm, order, order);
                while (!permutation_iterator_is_done(iter_out_perm))
                {
                    // Indices for the current permutation
                    const uint8_t *const current_perm = permutation_iterator_current(iter_out_perm);
                    double basis_contribution = 1.0;

                    // Loop over the derivative terms and compute their product
                    for (unsigned idim = 0; idim < order; ++idim)
                    {
                        const unsigned idx_coord = current_out[current_perm[idim]];
                        const unsigned idx_dim = current_in[idim];
                        // AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
                        const double contribution = space_map_backward_derivative(map, idx_pt, idx_dim, idx_coord);
                        basis_contribution *= contribution;
                        // printf("Adding contribution of dxi_%u/dx_%u=%g to element(%zu, %zu, %zu)\n", idx_dim,
                        // idx_coord,
                        //        contribution, idx_in, idx_out, idx_pt);
                    }

                    // Check if we flip the sign (meaning subtract) for this contribution.
                    if (permutation_iterator_current_sign(iter_out_perm))
                    {
                        val -= basis_contribution;
                    }
                    else
                    {
                        val += basis_contribution;
                    }

                    permutation_iterator_next(iter_out_perm);
                }

                ptr_out[idx_in * out_dims[1] * out_dims[2] + idx_out * out_dims[2] + idx_pt] = val;
            }

            idx_out += 1;
            combination_iterator_next(iter_out_comb);
        }

        idx_in += 1;
        combination_iterator_next(iter_in_comb);
    }

    PyMem_Free(iter_int_pts);
    PyMem_Free(iter_out_comb);
    PyMem_Free(iter_in_comb);
    PyMem_Free(iter_out_perm);

    return (PyObject *)res;
}

static PyObject *compute_basis_transform(PyObject *mod, PyObject *const *args, const Py_ssize_t nargs, PyObject *kwds)
{
    interplib_module_state_t *const state = PyModule_GetState(mod);
    if (!state)
        return NULL;

    const space_map_object *map;
    Py_ssize_t order;

    if (parse_arguments_check(
            (cpyutl_argument_t[]){
                {.type = CPYARG_TYPE_PYTHON, .type_check = state->space_mapping_type, .p_val = &map, .kwname = "smap"},
                {.type = CPYARG_TYPE_SSIZE, .p_val = &order, .kwname = "order"},
                {},
            },
            args, nargs, kwds) < 0)
        return NULL;

    return (PyObject *)compute_basis_transform_impl(map, order);
}

PyMethodDef transformation_functions[] = {
    {
        .ml_name = "compute_basis_transform",
        .ml_meth = (void *)compute_basis_transform,
        .ml_flags = METH_FASTCALL | METH_KEYWORDS,
        .ml_doc = "TODO", // TODO
    },
    {}, // sentinel
};
