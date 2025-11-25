#include "mass_matrices.h"
#include "basis_objects.h"
#include "function_space_objects.h"
#include "integration_objects.h"

PyDoc_STRVAR(compute_mass_matrix_docstring,
             "compute_mass_matrix(space_in: FunctionSpace, space_out: FunctionSpace, integration_space: "
             "IntegrationSpace, *,    integration_registry: IntegrationRegistry = DEFAULT_INTEGRATION_REGISTRY, "
             "basis_registry: BasisRegistry = DEFAULT_BASIS_REGISTRY,) -> numpy.typing.NDArray[numpy.double]\n"
             "Compute the mass matrix between two function spaces.\n"
             "\n"
             "Parameters\n"
             "----------\n"
             "space_in : FunctionSpace\n"
             "Function space for the input functions.\n"
             "space_out : FunctionSpace\n"
             "Function space for the output functions.\n"
             "integration_space : IntegrationSpace\n"
             "Integration space used to compute the mass matrix.\n"
             "integration_registry : IntegrationRegistry, default: DEFAULT_INTEGRATION_REGISTRY\n"
             "Registry used to retrieve the integration rules.\n"
             "basis_registry : BasisRegistry, default: DEFAULT_BASIS_REGISTRY\n"
             "Registry used to retrieve the basis specifications.\n"
             "\n"
             "Returns\n"
             "-------\n"
             "array\n"
             "Mass matrix as a 2D array, which maps the primal degress of freedom of the input\n"
             "function space to dual degrees of freedom of the output function space.\n");

static multidim_iterator_t *function_space_iterator(const function_space_object *space)
{
    const Py_ssize_t ndims = Py_SIZE(space);
    multidim_iterator_t *iter = PyMem_Malloc(multidim_iterator_needed_memory(ndims));
    if (!iter)
        return NULL;
    for (unsigned idim = 0; idim < ndims; ++idim)
    {
        multidim_iterator_init_dim(iter, idim, space->specs[idim].order + 1);
    }
    return iter;
}

static multidim_iterator_t *integration_space_iterator(const integration_space_object *space)
{
    const Py_ssize_t ndims = Py_SIZE(space);
    multidim_iterator_t *iter = PyMem_Malloc(multidim_iterator_needed_memory(ndims));
    if (!iter)
        return NULL;
    for (unsigned idim = 0; idim < ndims; ++idim)
    {
        multidim_iterator_init_dim(iter, idim, space->specs[idim].order + 1);
    }
    return iter;
}

static const integration_rule_t **integration_rules_get(const unsigned n_rules,
                                                        const integration_spec_t specs[const static n_rules],
                                                        integration_rule_registry_t *registry)
{
    const integration_rule_t **const array = PyMem_Malloc(n_rules * sizeof(*array));
    if (!array)
        return NULL;
    for (unsigned irule = 0; irule < n_rules; ++irule)
    {
        const interp_result_t res = integration_rule_registry_get_rule(registry, specs[irule], array + irule);
        if (res != INTERP_SUCCESS)
        {
            PyErr_Format(PyExc_RuntimeError, "Failed to retrieve integration rule: %s (%s).", interp_error_str(res),
                         interp_error_msg(res));
            for (unsigned i = 0; i < irule; ++i)
            {
                integration_rule_registry_release_rule(registry, array[i]);
            }
            PyMem_Free(array);
            return NULL;
        }
    }
    return array;
}

static void integration_rules_release(const unsigned n_rules, const integration_rule_t *rules[static n_rules],
                                      integration_rule_registry_t *registry)
{
    for (unsigned irule = 0; irule < n_rules; ++irule)
    {
        integration_rule_registry_release_rule(registry, rules[irule]);
        rules[irule] = NULL;
    }
    PyMem_Free(rules);
}

static const basis_set_t **basis_sets_get(const unsigned n_basis, const basis_spec_t specs[const static n_basis],
                                          const integration_rule_t *rules[const static n_basis],
                                          basis_set_registry_t *registry)
{
    const basis_set_t **const array = PyMem_Malloc(n_basis * sizeof(*array));
    if (!array)
        return NULL;
    for (unsigned ibasis = 0; ibasis < n_basis; ++ibasis)
    {
        const interp_result_t res =
            basis_set_registry_get_basis_set(registry, array + ibasis, rules[ibasis], specs[ibasis]);
        if (res != INTERP_SUCCESS)
        {
            PyErr_Format(PyExc_RuntimeError, "Failed to retrieve basis set: %s (%s).", interp_error_str(res),
                         interp_error_msg(res));
            for (unsigned i = 0; i < ibasis; ++i)
            {
                basis_set_registry_release_basis_set(registry, array[i]);
            }
            PyMem_Free(array);
            return NULL;
        }
    }
    return array;
}

static void basis_sets_release(const unsigned n_basis, const basis_set_t *sets[static n_basis],
                               basis_set_registry_t *registry)
{
    for (unsigned ibasis = 0; ibasis < n_basis; ++ibasis)
    {
        basis_set_registry_release_basis_set(registry, sets[ibasis]);
        sets[ibasis] = NULL;
    }
    PyMem_Free(sets);
}

typedef struct
{
    multidim_iterator_t *iter_in;
    multidim_iterator_t *iter_out;
    multidim_iterator_t *iter_int;
    unsigned n_rules;
    const integration_rule_t **rules;
    unsigned n_dim_in;
    const basis_set_t **basis_in;
    unsigned n_dim_out;
    const basis_set_t **basis_out;
} mass_matrix_resources_t;

static void mass_matrix_release_resources(mass_matrix_resources_t *resources,
                                          integration_rule_registry_t *integration_registry,
                                          basis_set_registry_t *basis_registry)
{
    if (resources->basis_out)
        basis_sets_release(resources->n_dim_out, resources->basis_out, basis_registry);
    if (resources->basis_in)
        basis_sets_release(resources->n_dim_in, resources->basis_in, basis_registry);
    if (resources->rules)
        integration_rules_release(resources->n_rules, resources->rules, integration_registry);
    if (resources->iter_out)
        PyMem_Free(resources->iter_out);
    if (resources->iter_in)
        PyMem_Free(resources->iter_in);
    if (resources->iter_int)
        PyMem_Free(resources->iter_int);
    *resources = (mass_matrix_resources_t){};
}

static int mass_matrix_create_resources(const function_space_object *space_in, const function_space_object *space_out,
                                        const integration_space_object *integration_space,
                                        integration_rule_registry_t *integration_registry,
                                        basis_set_registry_t *basis_registry, mass_matrix_resources_t *resources)
{
    mass_matrix_resources_t res = {};
    // Create iterators for function spaces and integration rules
    res.iter_in = function_space_iterator(space_in);
    res.iter_out = function_space_iterator(space_out);
    res.iter_int = integration_space_iterator(integration_space);
    const Py_ssize_t n_rules = Py_SIZE(integration_space);
    // Get integration rules and basis sets
    res.rules = integration_rules_get(n_rules, integration_space->specs, integration_registry);
    res.n_rules = n_rules;
    const Py_ssize_t n_basis_in = Py_SIZE(space_in);
    res.basis_in = res.rules ? basis_sets_get(n_basis_in, space_in->specs, res.rules, basis_registry) : NULL;
    res.n_dim_in = n_basis_in;
    const Py_ssize_t n_basis_out = Py_SIZE(space_out);
    res.basis_out = res.rules ? basis_sets_get(n_basis_out, space_out->specs, res.rules, basis_registry) : NULL;
    res.n_dim_out = n_basis_out;
    if (!res.iter_in || !res.iter_out || !res.iter_int || !res.rules || !res.basis_in || !res.basis_out)
    {
        mass_matrix_release_resources(&res, integration_registry, basis_registry);
        return -1;
    }
    *resources = res;
    return 0;
}

static int function_spaces_match(const function_space_object *space_in, const function_space_object *space_out)
{
    if (space_in == space_out)
        return 1;

    const unsigned n_space_dim = Py_SIZE(space_in);
    if (n_space_dim != Py_SIZE(space_out))
        return 0;

    // Space contents might match instead
    for (unsigned i = 0; i < n_space_dim; ++i)
    {
        if ((space_in->specs[i].order != space_out->specs[i].order) ||
            (space_in->specs[i].type != space_out->specs[i].type))
            return 0;
    }
    return 1;
}
static PyObject *compute_mass_matrix(PyObject *module, PyObject *const *args, const Py_ssize_t nargs,
                                     const PyObject *kwnames)
{
    const interplib_module_state_t *state = PyModule_GetState(module);
    if (!state)
        return NULL;

    const function_space_object *space_in, *space_out;
    const integration_space_object *integration_space;
    const integration_registry_object *integration_registry =
        (const integration_registry_object *)state->registry_integration;
    const basis_registry_object *basis_registry = (const basis_registry_object *)state->registry_basis;

    if (parse_arguments_check(
            (cpyutl_argument_t[]){
                {.type = CPYARG_TYPE_PYTHON, .p_val = &space_in, .type_check = state->function_space_type},
                {.type = CPYARG_TYPE_PYTHON, .p_val = &space_out, .type_check = state->function_space_type},
                {.type = CPYARG_TYPE_PYTHON, .p_val = &integration_space, .type_check = state->integration_space_type},
                {.type = CPYARG_TYPE_PYTHON,
                 .p_val = &integration_registry,
                 .type_check = state->integration_registry_type,
                 .optional = 1,
                 .kwname = "integration_registry",
                 .kw_only = 1},
                {.type = CPYARG_TYPE_PYTHON,
                 .p_val = &basis_registry,
                 .type_check = state->basis_registry_type,
                 .optional = 1,
                 .kwname = "basis_registry",
                 .kw_only = 1},
                {},
            },
            args, nargs, kwnames) < 0)
        return NULL;

    const unsigned n_space_dim = Py_SIZE(space_in);
    if (Py_SIZE(space_out) != n_space_dim || Py_SIZE(integration_space) != n_space_dim)
    {
        PyErr_Format(
            PyExc_ValueError,
            "Function spaces must have the same dimensionality (space in: %u, space out: %u, integration space: %u).",
            (unsigned)Py_SIZE(space_in), (unsigned)Py_SIZE(space_out), (unsigned)Py_SIZE(integration_space));
        return NULL;
    }

    // Create resources
    mass_matrix_resources_t resources = {};
    if (mass_matrix_create_resources(space_in, space_out, integration_space, integration_registry->registry,
                                     basis_registry->registry, &resources))
        return NULL;

    const npy_intp dims[2] = {(npy_intp)multidim_iterator_total_size(resources.iter_out),
                              (npy_intp)multidim_iterator_total_size(resources.iter_in)};

    PyArrayObject *const out = (PyArrayObject *)PyArray_SimpleNew(2, dims, NPY_DOUBLE);
    if (!out)
    {
        mass_matrix_release_resources(&resources, integration_registry->registry, basis_registry->registry);
        return NULL;
    }
    npy_double *const p_out = PyArray_DATA(out);

    // Matrix is symmetric if spaces match
    const int is_symmetric = function_spaces_match(space_in, space_out);
    multidim_iterator_set_to_start(resources.iter_in);
    multidim_iterator_set_to_start(resources.iter_out);
    while (!multidim_iterator_is_at_end(resources.iter_out))
    {
        const size_t index_out = multidim_iterator_get_flat_index(resources.iter_out);
        CPYUTL_ASSERT(index_out < (size_t)dims[0], "Out index out of bounds.");
        const size_t index_in = multidim_iterator_get_flat_index(resources.iter_in);
        CPYUTL_ASSERT(index_in < (size_t)dims[1], "In index out of bounds.");

        // integrate the respective basis
        multidim_iterator_set_to_start(resources.iter_int);
        double result = 0;
        // Integrate the basis product
        while (!multidim_iterator_is_at_end(resources.iter_int))
        {
            // Compute weight and basis values for these outer product basis and integration
            double weight = 1, basis_in = 1, basis_out = 1;
            for (unsigned idim = 0; idim < n_space_dim; ++idim)
            {
                const size_t integration_point_idx = multidim_iterator_get_offset(resources.iter_int, idim);
                weight *= integration_rule_weights_const(resources.rules[idim])[integration_point_idx];
                basis_in *= basis_set_basis_values(
                    resources.basis_in[idim],
                    multidim_iterator_get_offset(resources.iter_in, idim))[integration_point_idx];
                basis_out *= basis_set_basis_values(
                    resources.basis_out[idim],
                    multidim_iterator_get_offset(resources.iter_out, idim))[integration_point_idx];
            }
            multidim_iterator_advance(resources.iter_int, n_space_dim - 1, 1);
            // Add the contributions to the result
            result += weight * basis_in * basis_out;
        }
        // Advance the input basis
        multidim_iterator_advance(resources.iter_in, n_space_dim - 1, 1);
        // If we've done enough input basis, we advance the output basis and reset the input iterator
        if ((is_symmetric && (index_in == index_out)) || multidim_iterator_is_at_end(resources.iter_in))
        {
            multidim_iterator_advance(resources.iter_out, n_space_dim - 1, 1);
            multidim_iterator_set_to_start(resources.iter_in);
        }
        p_out[index_out * dims[1] + index_in] = result;
    }

    // If we're symmetric, we have to fill up the upper diagonal part
    if (is_symmetric)
    {
        for (npy_intp i = 0; i < dims[0]; ++i)
        {
            for (npy_intp j = i + 1; j < dims[1]; ++j)
            {
                p_out[i * dims[1] + j] = p_out[j * dims[1] + i];
            }
        }
    }

    mass_matrix_release_resources(&resources, integration_registry->registry, basis_registry->registry);
    return (PyObject *)out;
}

PyMethodDef mass_matrices_methods[] = {
    {
        .ml_name = "compute_mass_matrix",
        .ml_meth = (void *)compute_mass_matrix,
        .ml_flags = METH_FASTCALL | METH_KEYWORDS,
        .ml_doc = compute_mass_matrix_docstring,
    },
    {},
};
