#include "incidence.h"
#include "../basis/basis_lagrange.h"
#include "../polynomials/lagrange.h"
#include "basis_objects.h"

void bernstein_apply_incidence_operator(
    const unsigned n, const size_t pre_stride, const size_t post_stride, const unsigned cols,
    const double INTERPLIB_ARRAY_ARG(values_in, restrict const static pre_stride *(n + 1) * post_stride * cols),
    double INTERPLIB_ARRAY_ARG(values_out, restrict const pre_stride * n * post_stride * cols))
{
    const size_t col_stride_in = cols * post_stride * pre_stride * (n + 1);
    const size_t col_stride_out = cols * post_stride * pre_stride * n;
#pragma omp simd
    for (unsigned i_col = 0; i_col < cols; ++i_col)
    {
        double *const vout = values_out + i_col * col_stride_out;
        const double *const vin = values_in + i_col * col_stride_in;

        for (size_t i_pre = 0; i_pre < pre_stride; ++i_pre)
        {
            for (size_t i_post = 0; i_post < post_stride; ++i_post)
            {
                double *const ptr_out = vout + i_pre * n * post_stride + i_post;
                const double *const ptr_in = vin + i_pre * (n + 1) * post_stride + i_post;

                const npy_double coeff = (double)n / 2.0;
                ptr_out[0] -= coeff * ptr_in[0];
                for (unsigned col = 1; col < n; ++col)
                {
                    const npy_double x = coeff * ptr_in[col * post_stride];
                    ptr_out[col * post_stride] -= x;
                    ptr_out[(col - 1) * post_stride] += x;
                }
                ptr_out[(n - 1) * post_stride] += coeff * ptr_in[n * post_stride];
            }
        }
    }
}

void legendre_apply_incidence_operator(
    const unsigned n, const size_t pre_stride, const size_t post_stride, const unsigned cols,
    const double INTERPLIB_ARRAY_ARG(values_in, restrict const static pre_stride *(n + 1) * post_stride * cols),
    double INTERPLIB_ARRAY_ARG(values_out, restrict const pre_stride * n * post_stride * cols))
{
    const size_t col_stride_in = cols * post_stride * pre_stride * (n + 1);
    const size_t col_stride_out = cols * post_stride * pre_stride * n;
#pragma omp simd
    for (unsigned i_col = 0; i_col < cols; ++i_col)
    {
        double *const vout = values_out + i_col * col_stride_out;
        const double *const vin = values_in + i_col * col_stride_in;

        for (size_t i_pre = 0; i_pre < pre_stride; ++i_pre)
        {
            for (size_t i_post = 0; i_post < post_stride; ++i_post)
            {
                double *const ptr_out = vout + i_pre * n * post_stride + i_post;
                const double *const ptr_in = vin + i_pre * (n + 1) * post_stride + i_post;

                for (unsigned col = n; col > 0; --col)
                {
                    unsigned coeff = 2 * col - 1;
                    for (unsigned c_row = 0; 2 * c_row < col; ++c_row)
                    {
                        const unsigned r = (col - 1 - 2 * c_row);
                        ptr_out[r * post_stride] += coeff * ptr_in[col * post_stride];
                        coeff -= 4;
                    }
                }
            }
        }
    }
}

void lagrange_apply_incidence_matrix(
    const basis_set_type_t type, const unsigned n, const size_t pre_stride, const size_t post_stride,
    const unsigned cols,
    const double INTERPLIB_ARRAY_ARG(values_in, restrict const static pre_stride *(n + 1) * post_stride * cols),
    double INTERPLIB_ARRAY_ARG(values_out, restrict const pre_stride * n * post_stride * cols),
    double INTERPLIB_ARRAY_ARG(work, restrict const n + (n + 1) + n * (n + 1)))
{
    // Divide up the work array
    double *restrict const out_nodes = work + 0;
    double *restrict const in_nodes = work + n;
    double *restrict const trans_matrix = work + n + (n + 1);

    // Compute nodes for the output set
    interp_result_t res = generate_lagrange_roots(n - 1, type, out_nodes);
    CPYUTL_ASSERT(res == INTERP_SUCCESS, "Somehow an invalid enum?");
    if (res != INTERP_SUCCESS)
        return;

    res = generate_lagrange_roots(n, type, in_nodes);
    CPYUTL_ASSERT(res == INTERP_SUCCESS, "Somehow an invalid enum?");
    if (res != INTERP_SUCCESS)
        return;

    lagrange_polynomial_first_derivative_2(n, out_nodes, n + 1, in_nodes, trans_matrix);
    const size_t col_stride_in = cols * post_stride * pre_stride * (n + 1);
    const size_t col_stride_out = cols * post_stride * pre_stride * n;
#pragma omp simd
    for (unsigned i_col = 0; i_col < cols; ++i_col)
    {
        double *const vout = values_out + i_col * col_stride_out;
        const double *const vin = values_in + i_col * col_stride_in;

        for (size_t i_pre = 0; i_pre < pre_stride; ++i_pre)
        {
            for (size_t i_post = 0; i_post < post_stride; ++i_post)
            {
                double *const ptr_out = vout + i_pre * n * post_stride + i_post;
                const double *const ptr_in = vin + i_pre * (n + 1) * post_stride + i_post;

                // Apply the transformation matrix
                for (unsigned row = 0; row < n; ++row)
                {
                    double v = 0;
                    for (unsigned col = 0; col < n + 1; ++col)
                    {
                        v += trans_matrix[row * (n + 1) + col] * ptr_in[col * post_stride];
                    }
                    ptr_out[row * post_stride] = v;
                }
            }
        }
    }
}

static PyObject *incidence_matrix(PyObject *mod, PyObject *const *args, const Py_ssize_t nargs, const PyObject *kwnames)
{
    const interplib_module_state_t *const state = PyModule_GetState(mod);
    if (!state)
        return NULL;

    const basis_specs_object *basis_specs;
    if (parse_arguments_check(
            (cpyutl_argument_t[]){
                {.type = CPYARG_TYPE_PYTHON,
                 .p_val = &basis_specs,
                 .type_check = state->basis_spec_type,
                 .kwname = "basis_specs"},
                {},
            },
            args, nargs, kwnames) < 0)
        return NULL;

    if (basis_specs->spec.order == 0)
    {
        PyErr_SetString(PyExc_ValueError, "Cannot compute the incidence matrix for a zero-dimensional basis.");
        return NULL;
    }

    const unsigned n = basis_specs->spec.order;
    const npy_intp dims[2] = {n, n + 1};
    PyArrayObject *const out = (PyArrayObject *)PyArray_SimpleNew(2, dims, NPY_DOUBLE);
    if (!out)
        return NULL;

    npy_double *const data = PyArray_DATA(out);
    memset(PyArray_DATA(out), 0, (size_t)n * (n + 1) * sizeof(*data));
    switch (basis_specs->spec.type)
    {
    case BASIS_BERNSTEIN:
        // Use recurrence relation
        {
            // Scale by 1/2 because we change variables from [0, 1] to [-1, +1]
            const npy_double coeff = (double)n / 2.0;
            data[0 * (n + 1) + 0] = -coeff;
            for (unsigned col = 1; col < n; ++col)
            {
                data[col * (n + 1) + col] = -coeff;
                data[(col - 1) * (n + 1) + col] = +coeff;
            }
            data[(n - 1) * (n + 1) + n] = +coeff;
        }
        break;

    case BASIS_LEGENDRE:
        // Use recurrence relation
        {
            for (unsigned col = n; col > 0; --col)
            {
                unsigned coeff = 2 * col - 1;
                for (unsigned c_row = 0; 2 * c_row < col; ++c_row)
                {
                    const unsigned r = (col - 1 - 2 * c_row);
                    data[r * (n + 1) + col] = coeff;
                    coeff -= 4;
                }
            }
        }
        break;

    case BASIS_LAGRANGE_UNIFORM:
    case BASIS_LAGRANGE_GAUSS:
    case BASIS_LAGRANGE_GAUSS_LOBATTO:
    case BASIS_LAGRANGE_CHEBYSHEV_GAUSS:
        // Use direct evaluation of the derivative at nodes
        {
            // Compute nodes for the output set
            double *const out_nodes = PyMem_Malloc(sizeof(*out_nodes) * n);
            if (!out_nodes)
            {
                Py_DECREF(out);
                return NULL;
            }
            interp_result_t res = generate_lagrange_roots(n - 1, basis_specs->spec.type, out_nodes);
            (void)res;
            CPYUTL_ASSERT(res == INTERP_SUCCESS, "Somehow an invalid enum?");
            double *const in_nodes = PyMem_Malloc(sizeof(*in_nodes) * (n + 1));
            if (!in_nodes)
            {
                PyMem_Free(out_nodes);
                Py_DECREF(out);
                return NULL;
            }
            res = generate_lagrange_roots(n, basis_specs->spec.type, in_nodes);
            (void)res;
            CPYUTL_ASSERT(res == INTERP_SUCCESS, "Somehow an invalid enum?");

            lagrange_polynomial_first_derivative_2(n, out_nodes, n + 1, in_nodes, data);

            PyMem_Free(out_nodes);
            PyMem_Free(in_nodes);
        }
        break;

    default:
        CPYUTL_ASSERT(0, "Unsupported basis type.");
        PyErr_SetString(PyExc_NotImplementedError, "Unsupported basis type (should not have happended).");
        Py_DECREF(out);
        return NULL;
    }

    return (PyObject *)out;
}

PyDoc_STRVAR(incidence_matrix_docstring,
             "incidence_matrix(basis_specs : BasisSpecs) -> numpy.typing.NDArray[numpy.double]\n"
             "Return the incidence matrix to transfer derivative degrees of freedom.\n"
             "\n"
             "Parameters\n"
             "----------\n"
             "specs : BasisSpecs\n"
             "    Basis specs for which this incidence matrix should be computed.\n"
             "\n"
             "Returns\n"
             "-------\n"
             "array\n"
             "    One dimensional incidence matrix. It transfers primal degrees of freedom\n"
             "    for a derivative to a function space one order less than the original.\n");

PyMethodDef incidence_methods[] = {
    {
        .ml_name = "incidence_matrix",
        .ml_meth = (void *)incidence_matrix,
        .ml_flags = METH_FASTCALL | METH_KEYWORDS,
        .ml_doc = incidence_matrix_docstring,
    },
    {}, // sentinel
};
